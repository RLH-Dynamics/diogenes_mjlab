"""Foot reference gait for the Harold biped (suspended walking-in-air task).

Each foot traces a closed loop, expressed in the base (``body_assy``) frame.
The loop is first built in the leg's hip-zero sagittal plane:

  * STANCE, phase in [0, STANCE_FRACTION): a straight line parallel to the
    ground, STANCE_HEIGHT below the base, moving BACKWARD (+y) at constant speed
    from the front end of the step to the back end.
  * SWING, phase in [STANCE_FRACTION, 1): y returns linearly from the back end
    to the front end while z follows a half sine, peaking SWING_HEIGHT above the
    line directly above its midpoint.

The whole loop is then rotated about the leg's hip joint axis by HIP_ABDUCTION,
outward (away from the body). This is exactly the motion of the hip joint, so
the reference is followed with the hip held at +HIP_ABDUCTION (left) /
-HIP_ABDUCTION (right) while the thigh and calf trace the original loop. The hip
axes point fore/aft, so the stance line stays parallel to the ground (about
5 mm higher than STANCE_HEIGHT and ~6 cm further out).

Position is continuous around the loop; velocity steps at the two joins (the
foot reverses direction there).

Frame conventions (verified in the viewer): +x = robot left, +y = robot BACK,
+z = up, so forward is -y. The right leg runs half a cycle behind the left
(180 deg out of phase).
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Gait parameters.
# ---------------------------------------------------------------------------

# Full cycle (stance + swing) duration, seconds.
GAIT_PERIOD: float = 2.0

# Fraction of the cycle spent on the stance line (1.2 s stance, 0.8 s swing).
STANCE_FRACTION: float = 0.6

# Length of the stance line along y (m).
STEP_LENGTH: float = 0.12

# Swing apex above the stance line (m), measured in the leg's own plane.
SWING_HEIGHT: float = 0.04

# Stance line height before the hip rotation: foot site z in the base frame (m).
STANCE_HEIGHT: float = -0.34

# Outward tilt of each leg's loop about its hip axis (rad).
HIP_ABDUCTION: float = math.radians(10.0)

# Foot sites tracked, in reward/observation order.
FOOT_SITE_NAMES: tuple[str, ...] = ("left_foot", "right_foot")

# Per-leg phase offsets (cycle fraction), same order as FOOT_SITE_NAMES.
LEG_PHASE_OFFSETS: tuple[float, ...] = (0.0, 0.5)

# Step centre (x, y) per foot in the base frame BEFORE the hip rotation: the foot
# site position at the zero joint pose, so the loop is centred where the leg
# naturally hangs. Around the loop the thigh and calf stay >= 29.8 deg from their
# limits; with the abduction the hips sit 15 deg inside their asymmetric
# (-5..45 / -45..5 deg) ranges.
FOOT_CENTER_XY: dict[str, tuple[float, float]] = {
  "left_foot": (0.0845, 0.08565),
  "right_foot": (-0.0845, 0.08565),
}

# Hip joint axis and a point on it, in the base frame at the init pose (read
# from the compiled model). Both axes point along -y.
HIP_AXIS_B: tuple[float, float, float] = (0.0, -1.0, 0.0)
HIP_ANCHOR_B: dict[str, tuple[float, float, float]] = {
  "left_foot": (0.087, 0.02215, 0.0),
  "right_foot": (-0.087, 0.02215, 0.0),
}

# Sign of the hip joint angle that moves each foot OUTWARD (left_hip +rad moves
# the left foot to +x; right_hip -rad moves the right foot to -x).
HIP_OUTWARD_SIGN: dict[str, float] = {"left_foot": 1.0, "right_foot": -1.0}


def foot_centers(
  device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
  """Step centres before the hip rotation, shape (num_feet, 2)."""
  return torch.tensor(
    [FOOT_CENTER_XY[name] for name in FOOT_SITE_NAMES], device=device, dtype=dtype
  )


def hip_angles(
  device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
  """Hip joint angle holding each loop outward, shape (num_feet,), rad."""
  return torch.tensor(
    [HIP_OUTWARD_SIGN[name] * HIP_ABDUCTION for name in FOOT_SITE_NAMES],
    device=device,
    dtype=dtype,
  )


def leg_phases(phase: torch.Tensor) -> torch.Tensor:
  """Per-leg phase in [0, 1). phase: (...,) global phase -> (..., num_feet)."""
  offsets = torch.tensor(LEG_PHASE_OFFSETS, device=phase.device, dtype=phase.dtype)
  return torch.remainder(phase.unsqueeze(-1) + offsets, 1.0)


def _rotate(vec: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  """Rodrigues rotation of vec (..., F, 3) about unit axis (3,) by angle (F,)."""
  cos = torch.cos(angle).unsqueeze(-1)
  sin = torch.sin(angle).unsqueeze(-1)
  axis = axis.expand_as(vec)
  dot = torch.sum(axis * vec, dim=-1, keepdim=True)
  return vec * cos + torch.cross(axis, vec, dim=-1) * sin + axis * dot * (1.0 - cos)


def foot_reference(
  phase: torch.Tensor,
  centers: torch.Tensor,
  period: float = GAIT_PERIOD,
  stance_fraction: float = STANCE_FRACTION,
  step_length: float = STEP_LENGTH,
  swing_height: float = SWING_HEIGHT,
  stance_height: float = STANCE_HEIGHT,
  hip_abduction: float = HIP_ABDUCTION,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reference foot position and velocity in the base frame.

  Args:
    phase: per-leg phase in [0, 1), shape (..., num_feet) (see leg_phases).
    centers: step centre (x, y) per foot before the hip rotation, shape
      (num_feet, 2), in FOOT_SITE_NAMES order.
    hip_abduction: outward loop tilt about each hip axis (rad); 0 disables it.

  Returns:
    (pos, vel), each shape (..., num_feet, 3), in m and m/s.
  """
  t_stance = stance_fraction * period
  t_swing = (1.0 - stance_fraction) * period

  in_stance = phase < stance_fraction
  s = phase / stance_fraction  # stance progress, 0 -> 1
  u = (phase - stance_fraction) / (1.0 - stance_fraction)  # swing progress, 0 -> 1
  zeros = torch.zeros_like(phase)

  # y relative to the step centre: front (-L/2) -> back (+L/2) in stance, then
  # back -> front in swing.
  y_rel = torch.where(
    in_stance, -0.5 * step_length + step_length * s, 0.5 * step_length - step_length * u
  )
  z = stance_height + torch.where(
    in_stance, zeros, swing_height * torch.sin(math.pi * u)
  )

  vy = torch.where(
    in_stance,
    torch.full_like(phase, step_length / t_stance),
    torch.full_like(phase, -step_length / t_swing),
  )
  vz = torch.where(
    in_stance, zeros, (swing_height * math.pi / t_swing) * torch.cos(math.pi * u)
  )

  x = centers[:, 0].expand_as(phase)
  y = centers[:, 1] + y_rel
  pos = torch.stack([x, y, z], dim=-1)
  vel = torch.stack([zeros, vy, vz], dim=-1)

  # Tilt each loop outward about its hip axis.
  device, dtype = pos.device, pos.dtype
  axis = torch.tensor(HIP_AXIS_B, device=device, dtype=dtype)
  anchors = torch.tensor(
    [HIP_ANCHOR_B[name] for name in FOOT_SITE_NAMES], device=device, dtype=dtype
  )
  angles = hip_angles(device, dtype) * (hip_abduction / HIP_ABDUCTION)
  pos = anchors + _rotate(pos - anchors, axis, angles)
  vel = _rotate(vel, axis, angles)
  return pos, vel


def nominal_joint_pos() -> dict[str, float]:
  """Joint angles holding each leg at the centre of its reference loop.

  The hips sit at their abduction angle (the loop is a rotation about the hip
  axis, so the hip holds still at +-HIP_ABDUCTION while the thigh and calf trace
  it); the thigh and calf sit at zero, where the foot hangs at FOOT_CENTER_XY.

  This -- not the all-zero pose -- is the gait's home configuration, and it is
  what a reset should start near: the all-zero pose leaves the hips only 5 deg
  off their hard limits, inside which a 2% termination band leaves almost no
  room to perturb.
  """
  pose = {f"{side}_{link}": 0.0 for side in ("left", "right")
          for link in ("hip", "thigh", "calf")}
  for site, sign in HIP_OUTWARD_SIGN.items():
    pose[f"{site.split('_')[0]}_hip"] = sign * HIP_ABDUCTION
  return pose
