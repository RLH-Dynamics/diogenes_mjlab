"""Custom MDP terms for the Diogenes periodic-hopping task.

This module adds the task-specific functions that mjlab does not ship out of the
box. Generic regularizers (action-rate, joint-limit, joint pos/vel observations,
last-action observation, time-out termination) are reused directly from
``mjlab.envs.mdp``. Generic contact rewards (foot slip, soft landing) and the
electrical-power penalty are reused from ``mjlab.tasks.velocity.mdp`` and
``mjlab.envs.mdp`` respectively; this file only adds what is genuinely specific
to the phase-driven hop stand.

Two carriage trajectories are provided, sharing the same phase clock, the same
foot-xy hold and the same regularizers:

  * ``slider_dual_parabola_tracking`` -- gravity-exact dual-parabolic motion.
    Highly dynamic (true free-fall flight arc). Its period is DERIVED from
    physics; there is no free period parameter.
  * ``slider_sinusoid_tracking`` -- a smooth sinusoid between a minimum and a
    maximum height. Much gentler (the peak vertical acceleration is A*omega^2,
    which you keep below g so the foot never goes ballistic). Its period is a
    FREE parameter you choose, because a sinusoid has no physics-derived period.
    Intended as an easier first sim-to-real target.

Geometry note (important for signs)
-----------------------------------
The leg hangs from an unactuated prismatic ``slider`` joint. In the URDF the
``leg_mount_assy`` body carries a ``quat="0 1 0 0"`` (a 180 deg rotation about
X), so the joint's local +Z axis points along world **-Z**. A *more negative*
``slider`` value therefore raises the carriage. With the default slider value
of 0.0, the height of the carriage above its starting position is simply::

    height_above_start = -slider_pos

This was verified directly against MuJoCo (``mj_forward`` sweep of the slider).

The phase clock
---------------
A single global hop phase ``phi in [0, 1)`` advances with simulation time and
wraps every ``hop_period`` seconds. It is derived from the per-env step counter
``env.episode_length_buf`` (reset to 0 on episode reset, incremented once per
control step) times ``env.step_dt``. The reward, the gait terms and the
observation all read the same counter on the same step, so they stay perfectly
in phase.

"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import sample_uniform

from .constants import GRAVITY  # re-exported so diogenes_mdp.GRAVITY still works
from .accessors import carriage_height as _carriage_height

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Entity config selecting just the unactuated prismatic slider joint.
SLIDER_CFG = SceneEntityCfg("robot", joint_names=("slider",))


##
# Phase clock.
##


def _phase(env: ManagerBasedRlEnv, hop_period: float) -> torch.Tensor:
  """Global hop phase in [0, 1), shape (num_envs,).

  Derived from the per-env control-step counter so it resets with the episode
  and is identical for the reward and the observation within a step.
  """
  t = env.episode_length_buf.float() * env.step_dt
  return torch.remainder(t / hop_period, 1.0)


def phase_clock(env: ManagerBasedRlEnv, hop_period: float = 0.6) -> torch.Tensor:
  """(sin, cos) of the hop phase. Shape (num_envs, 2).

  A sin/cos pair gives the policy a smooth, wrap-free representation of where it
  is in the hop cycle (no discontinuity at the phase=1->0 boundary).
  """
  phi = _phase(env, hop_period)
  angle = 2.0 * math.pi * phi
  return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def _height_above_start(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Carriage height above its start position. Shape (num_envs,).

  ``height_above_start = -slider_pos`` (see module docstring). Using index -1 of
  the selected columns is robust whether ``joint_ids`` resolves to a list like
  ``[0]`` or a slice; the slider is the only joint this cfg selects.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return _carriage_height(asset, asset_cfg)


##
# Slider (carriage) state observations.
##


def slider_pos(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SLIDER_CFG
) -> torch.Tensor:
  """Raw slider joint position. Shape (num_envs, 1).

  Not made relative to the default (which is 0.0 anyway) so the policy sees the
  carriage's absolute travel along the rail.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids]


def slider_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SLIDER_CFG
) -> torch.Tensor:
  """Slider joint velocity. Shape (num_envs, 1)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


##
# Contact-force penalties (foot vs. floor).
##
# These read a ContactSensor configured with reduce="netforce", which sums all
# foot/floor contacts into a single net wrench expressed in the GLOBAL frame.
# In the global frame the z-component is the vertical force and the xy-plane
# components are the lateral forces, so the two penalties below decompose
# cleanly without any frame math here.


def lateral_contact_force_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize lateral (world x, y) foot/ground contact force. Shape (num_envs,).

  Minimizing tangential ground reaction discourages the foot from shearing,
  scuffing or skating along the floor, yielding a smoother, more planted stance.
  Requires the contact sensor to use ``reduce="netforce"`` (global-frame force).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # [B, N, 3] global frame (netforce).
  assert force is not None, (
    f"Sensor '{sensor_name}' must request the 'force' field for "
    "lateral_contact_force_l2."
  )
  lateral_sq = torch.sum(torch.square(force[..., :2]), dim=-1)  # [B, N]
  return torch.sum(lateral_sq, dim=-1)  # [B]


def vertical_contact_force_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize vertical (world z) foot/ground contact force. Shape (num_envs,).

  A gentle bias toward lower peak ground reaction encourages compliant,
  low-impact footfalls rather than slamming the floor. Keep the weight small:
  the foot fundamentally MUST push on the floor to hop, so an over-large weight
  here fights the task. Requires ``reduce="netforce"`` (global-frame force).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # [B, N, 3] global frame (netforce).
  assert force is not None, (
    f"Sensor '{sensor_name}' must request the 'force' field for "
    "vertical_contact_force_l2."
  )
  vertical_sq = torch.square(force[..., 2])  # [B, N]
  return torch.sum(vertical_sq, dim=-1)  # [B]


def foot_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize foot sliding: squared xy velocity AT THE CONTACT POINT while in contact.

  Shape (num_envs,).

  Why the contact point (not the body origin): the foot is offset ~0.2 m (the
  calf link length) from the ``calf_assy`` body origin, which sits up at the
  knee. A rigid body's point velocity is ``v_point = v_origin + omega x r``; with
  a 0.2 m lever arm, ordinary stance rotation (the leg pivoting as the carriage
  rises) injects a large spurious ``omega x r`` term into the *origin* velocity
  even when the foot is perfectly planted. Reading the body-origin velocity would
  therefore penalize legitimate pivoting and barely see true sliding.

  We reconstruct the velocity at the actual contact point reported by the sensor
  (``data.pos``, global frame): ``v_contact = v_origin + omega x (pos - origin)``,
  using the calf body's world-frame linear and angular velocity. The xy component
  is the true slip: ~0 when the foot pivots in place, nonzero only when the
  contact point translates along the floor. ``asset_cfg`` must select the foot
  body, e.g. ``body_names=("calf_assy",)``.

  The sensor must request the ``found`` and ``pos`` fields. Only in-contact steps
  are penalized, so the policy is free to move the foot in flight.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  pos = sensor.data.pos  # [B, N, 3] contact point, global frame.
  assert found is not None and pos is not None, (
    f"Sensor '{sensor_name}' must request the 'found' and 'pos' fields for foot_slip."
  )
  in_contact = (found > 0).any(dim=-1).float()  # [B]

  # Calf body twist and origin (single body selected by asset_cfg).
  body_id = asset_cfg.body_ids
  v_origin = asset.data.body_link_lin_vel_w[:, body_id]  # [B, 1, 3]
  omega = asset.data.body_link_ang_vel_w[:, body_id]  # [B, 1, 3]
  origin = asset.data.body_link_pos_w[:, body_id]  # [B, 1, 3]

  # Collapse the (single) body axis to [B, 3].
  v_origin = v_origin[:, 0]
  omega = omega[:, 0]
  origin = origin[:, 0]
  # Use the first contact slot's position (netforce -> a single representative
  # slot); shape [B, 3].
  contact_pos = pos[:, 0]

  # v_contact = v_origin + omega x (contact_pos - origin).
  r = contact_pos - origin  # [B, 3]
  v_contact = v_origin + torch.cross(omega, r, dim=-1)  # [B, 3]

  vel_xy_sq = torch.sum(torch.square(v_contact[:, :2]), dim=-1)  # [B]
  cost = vel_xy_sq * in_contact  # [B]

  num_in_contact = torch.sum(in_contact)
  mean_slip = torch.sum(torch.sqrt(vel_xy_sq) * in_contact) / torch.clamp(
    num_in_contact, min=1.0
  )
  env.extras["log"]["Metrics/foot_slip_vel_mean"] = mean_slip
  return cost


##
# Terminations.
##


def joint_at_limit(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  margin: float = 0.02,
) -> torch.Tensor:
  """Terminate when ANY selected joint reaches (within ``margin``) a hard limit.

  Shape (num_envs,), bool.

  Motivation: with the contact/slip/landing penalties zeroed, the most
  energetically cheap policy can exploit the joint hard-stops as a free energy
  sink -- driving the leg so that one or more joints bottom out on landing, so
  the impact energy is absorbed by the position constraint (the boundary does
  the braking) rather than by controlled motor work. Penalizing limit proximity
  only taxes this; terminating on it removes the exploit outright, since an
  episode that slams a stop ends immediately and collects no further reward.

  Why a margin off the HARD limit (``joint_pos_limits``) rather than the soft
  band: mjlab's ``soft_joint_pos_limits`` collapse onto the hard limits unless
  ``articulation.soft_joint_pos_limit_factor`` is set (< 1), which this robot
  does not set -- so a soft-limit test would only trip exactly at the boundary,
  too late to discourage the approach and fragile to floating-point. Instead we
  inset each joint's hard range by ``margin`` (a fraction of that joint's range)
  and terminate when the joint crosses into the inset band. This catches the
  bottoming-out behaviour just before the stop and gives a clean tuning knob
  independent of the soft-limit factor.

  IMPORTANT -- default/rest pose: this is a symmetric test on BOTH ends of every
  selected joint, so the default (reset) pose MUST sit strictly inside the inset
  band for all of them, or every env terminates on the first step. With
  ``use_default_offset=True`` on the action, the default pose is also the centre
  of the policy's operating range, so it should sit where the leg naturally
  spends the hop cycle, not merely somewhere legal. For this robot the calf's
  range is [0, pi/2] with its as-built default at 0.0 (exactly the lower stop)
  and the thigh's default at 0.0 (near its upper stop), so ``DEFAULT_INIT`` in
  diogenes_constants.py must be moved off those limits before enabling this term
  (e.g. a mildly crouched stance). At margin=0.02 the safe interior is roughly
  hip [-0.75, 0.75], thigh [-1.53, 0.23], calf [0.03, 1.54].

  Args:
    asset_cfg: Entity config selecting the joints to monitor (typically the
      three actuated leg joints; the unactuated slider is excluded).
    margin: Inset from each hard limit as a fraction of that joint's full range.
      0.0 means terminate only at the exact hard limit; 0.02 leaves a 2% guard
      band at each end. Joints with no finite range are ignored.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  pos = asset.data.joint_pos[:, joint_ids]  # [B, J]
  limits = asset.data.joint_pos_limits[:, joint_ids]  # [B, J, 2] (lower, upper)
  lower = limits[..., 0]
  upper = limits[..., 1]

  # Inset the range by `margin` * range at each end. Guard against non-finite
  # ranges (unlimited joints -> [-inf, inf]); their inset bounds stay infinite,
  # so the comparisons below are never True for them.
  rng = upper - lower
  inset = margin * rng
  low_thresh = lower + inset
  high_thresh = upper - inset

  at_lower = pos <= low_thresh  # [B, J]
  at_upper = pos >= high_thresh  # [B, J]
  at_limit = at_lower | at_upper  # [B, J]

  # Log the fraction of (env, joint) pairs sitting at a limit, for monitoring.
  env.extras["log"]["Metrics/joint_at_limit_frac"] = at_limit.float().mean()

  return at_limit.any(dim=-1)  # [B]


##
# Reset-time domain randomization (joint start pose).
##


def reset_joints_uniform_legal(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  margin: float = 0.02,
  safety_eps: float = 1e-3,
  velocity_range: tuple[float, float] = (0.0, 0.0),
) -> None:
  """Reset selected joints to a uniformly-random LEGAL position (absolute).

  Unlike mjlab's ``reset_joints_by_offset`` (which perturbs around the default
  pose), this samples each selected joint's start angle uniformly across its
  FULL range, so the leg can begin in essentially any legal orientation -- the
  behaviour wanted for a start-pose-robust policy.

  Why a margin-inset range (not the raw hard limits): the ``joint_at_limit``
  termination ends an episode (with a large penalty) whenever any selected joint
  is within ``margin`` * range of a hard stop. This robot leaves
  ``soft_joint_pos_limit_factor`` at its 1.0 default, so the soft limits EQUAL
  the hard limits -- sampling up to them would place a joint exactly on a stop
  and fire ``joint_at_limit`` on step 0, killing the env immediately. To stay
  consistent, we inset each joint's range by the SAME ``margin`` used by the
  termination, plus a tiny ``safety_eps`` so floating-point can never tip the
  sampled start over the threshold. Pass the same ``margin`` you give
  ``joint_at_limit``.

  Joints with a non-finite (unlimited) range are left at their default position
  (an unbounded joint has no meaningful uniform-legal range). The slider is
  unaffected: select only the actuated joints via ``asset_cfg``.

  Use with ``mode="reset"``.

  Args:
    env_ids: Environment IDs to reset. If None, resets all environments
      (normalized via ``resolve_env_ids``).
    asset_cfg: entity config selecting the joints to randomize (the three
      actuated leg joints; exclude the slider).
    margin: inset from each hard limit as a fraction of that joint's range.
      MUST match the ``joint_at_limit`` termination's margin.
    safety_eps: extra absolute inset (rad) on top of ``margin`` so the sampled
      start sits strictly inside the termination's safe band.
    velocity_range: ``(min, max)`` uniform start joint velocity (rad/s). Defaults
      to ``(0.0, 0.0)`` -- start at rest, the sane default for a hop stand.
  """
  env_ids = resolve_env_ids(env, env_ids)

  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids

  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  limits = asset.data.soft_joint_pos_limits  # == hard limits here (factor 1.0)
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  assert limits is not None

  lower = limits[env_ids][:, joint_ids, 0].clone()  # [E, J]
  upper = limits[env_ids][:, joint_ids, 1].clone()  # [E, J]
  rng = upper - lower
  inset = margin * rng + safety_eps
  low_s = lower + inset
  high_s = upper - inset

  # Uniform sample in [0, 1) -> map into each joint's inset band.
  u = torch.rand((len(env_ids), lower.shape[1]), device=env.device)
  joint_pos = low_s + u * (high_s - low_s)  # [E, J]

  # Unlimited joints (non-finite range) have no legal band -> keep default.
  finite = torch.isfinite(rng)
  default_sel = default_joint_pos[env_ids][:, joint_ids]
  joint_pos = torch.where(finite, joint_pos, default_sel)

  # Start velocities (default: at rest).
  joint_vel = default_joint_vel[env_ids][:, joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  ids = joint_ids
  if isinstance(ids, list):
    ids = torch.tensor(ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=ids,
  )

  # Log the post-reset distance-to-nearest-limit (fraction of range) so you can
  # confirm starts sit inside the joint_at_limit safe band. ~0 would be alarming.
  with torch.no_grad():
    finite_rng = torch.where(finite, rng, torch.ones_like(rng))
    dist_lo = (joint_pos - lower) / finite_rng
    dist_hi = (upper - joint_pos) / finite_rng
    nearest = torch.minimum(dist_lo, dist_hi)
    nearest = torch.where(finite, nearest, torch.ones_like(nearest))
    env.extras["log"]["Metrics/reset_joint_margin_min"] = nearest.min()


##
# Trajectory-tracking rewards (gravity-exact dual-parabolic slider + foot xy hold).
##
# Geometry facts (verified by compiling the model; mjlab 1.4.0 / mujoco 3.9.0):
#   * The slider sits ABOVE the leg joints, so its zero is fixed in the
#     kinematic tree and does NOT depend on hip/thigh/calf:
#         carriage_height_above_start = -slider_pos  (exact, pose-independent).
#   * Foot world position decomposes additively:
#         foot_world = (-slider) * z_hat + p0(hip, thigh, calf).
#     The hip joint translates the foot in world x.
#   * Units are METERS and RADIANS; no <mesh scale> is applied, so STL/mesh
#     coordinates are in meters. The foot-center mesh coordinate (0, 0, -0.25) m
#     transformed through the calf geom's pos+quat gives the body-frame offset
#     FOOT_OFFSET_B below (verified two ways against MuJoCo FK to ~1e-7 m).

# Foot-center offset in the calf_assy body frame (meters).
FOOT_OFFSET_B: tuple[float, float, float] = (-0.176776, 0.176777, -0.014)

# Foot-center world (x, y) at the default pose.
# Recapture if you change the default pose or FOOT_OFFSET_B.
# Updated for the re-imported Onshape assembly: the leg_mount/hip stack shifted
# ~+7 mm, moving the foot's world y by -7.00 mm at the reference pose (x and the
# calf-internal FOOT_OFFSET_B are unchanged, since calf_assy geometry did not
# change). Recomputed via MuJoCo FK on the new scene.xml.
DEFAULT_FOOT_REF_XY: tuple[float, float] = (0.00250, -0.10679)

def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate vec by quat (w, x, y, z), batched. quat:[B,4], vec:[B,3] -> [B,3].

  MuJoCo / mjlab store quaternions as (w, x, y, z).
  """
  w = quat[:, 0:1]
  xyz = quat[:, 1:4]
  t = 2.0 * torch.cross(xyz, vec, dim=-1)
  return vec + w * t + torch.cross(xyz, t, dim=-1)


def dual_parabola_timing(
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> tuple[float, float, float, float, float]:
  """Solve the gravity-exact dual-parabola timing from geometry alone.

  Everything is derived from physics; there is NO free period parameter.

  FLIGHT arc (true free fall at -gravity), amplitude Hf = traj_max - traj_transition:
    * Leaves traj_transition moving UP at v0, rises Hf, falls back to
      traj_transition moving DOWN at v0, with
          v0        = sqrt(2 * gravity * Hf)          (speed at the transition)
          T_flight  = 2 * sqrt(2 * Hf / gravity)      (full up+down flight time)

  RECOVERY arc (constant acceleration +a), amplitude Hr = traj_transition - traj_min:
    * Velocity continuity forces the recovery to ENTER at -v0 (matching the end
      of flight) and LEAVE at +v0 (matching the start of the next flight). A
      constant-accel arc that decelerates from v0 to 0 over a drop of Hr needs
          a         = v0^2 / (2 * Hr) = gravity * Hf / Hr   (the constant accel)
          T_recovery = 2 * v0 / a                            (down+back up time)

  Total period and the phase split:
          T_total     = T_flight + T_recovery
          flight_frac = T_flight / T_total

  Args:
    traj_min, traj_max, traj_transition: trajectory heights, z rel origin (m),
      with traj_max >= traj_transition >= traj_min.
    gravity: free-fall acceleration for the flight arc (m/s^2).

  Returns:
    (T_total, flight_frac, v0, recovery_accel, T_flight) -- seconds / m·s / etc.
  """
  Hf = traj_max - traj_transition  # flight amplitude (>= 0)
  Hr = traj_transition - traj_min  # recovery amplitude (> 0 required)
  assert Hf >= 0.0, "Require traj_max >= traj_transition."
  assert Hr > 0.0, "Require traj_transition > traj_min (a finite recovery dip)."

  v0 = math.sqrt(2.0 * gravity * Hf)
  t_flight = 2.0 * math.sqrt(2.0 * Hf / gravity)
  recovery_accel = gravity * Hf / Hr
  t_recovery = 2.0 * v0 / recovery_accel
  t_total = t_flight + t_recovery
  flight_frac = t_flight / t_total
  return t_total, flight_frac, v0, recovery_accel, t_flight


def dual_parabola_reference(
  phi: torch.Tensor,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> torch.Tensor:
  """Reference carriage height h_ref(phi). Shape == phi.shape. All z's rel origin.

  The cycle (phase phi in [0,1)) is two parabolic arcs meeting at
  ``traj_transition``, parametrized DIRECTLY by the physical motion (see
  ``dual_parabola_timing``):

    * FLIGHT arc, phi in [0, flight_frac]: true free fall at -gravity. Real time
      within the arc is t = phi * T_total, so
          h = traj_transition + v0 * t - 0.5 * gravity * t^2,
      which rises to apex traj_max and returns to traj_transition. Its duration
      is EXACTLY the Earth-gravity ballistic time for amplitude Hf.

    * RECOVERY arc, phi in [flight_frac, 1): constant deceleration/acceleration
      at +recovery_accel. With t' = phi * T_total - T_flight,
          h = traj_transition - v0 * t' + 0.5 * recovery_accel * t'^2,
      which dips to traj_min (at zero velocity) and returns to traj_transition,
      entering at -v0 and leaving at +v0 so velocity is continuous at both joins.

  Because the flight time is fixed by gravity and the recovery time by velocity
  continuity, the whole period T_total is derived, not chosen.

  Args:
    traj_min: lowest carriage height in the cycle (recovery dip), z rel origin.
    traj_max: apex carriage height (flight peak), z rel origin.
    traj_transition: height where flight and recovery meet (the cycle boundary
      level), z rel origin.
    gravity: free-fall acceleration for the flight arc (m/s^2).
  """
  t_total, flight_frac, v0, recovery_accel, t_flight = dual_parabola_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  t = phi * t_total  # real time within the cycle, seconds

  # FLIGHT: free fall from traj_transition, up at v0, accel -gravity.
  flight_h = traj_transition + v0 * t - 0.5 * gravity * torch.square(t)

  # RECOVERY: constant accel from traj_transition, down at v0, accel +recovery_accel.
  tr = t - t_flight
  recovery_h = (
    traj_transition - v0 * tr + 0.5 * recovery_accel * torch.square(tr)
  )

  return torch.where(phi < flight_frac, flight_h, recovery_h)


##
# Contact-phase penalties (keep the foot planted / lift it on schedule).
##
# Both read the same ContactSensor ``found`` field as ``foot_slip`` and use the
# identical in-contact convention: a primary is "in contact" on a step iff any
# of its slots reports found > 0. Both return a per-step 0/1 cost (give them a
# NEGATIVE weight in the cfg to turn the cost into a penalty), matching the
# binary, weight-tunable design requested. Defined here, AFTER
# ``dual_parabola_timing`` and ``GRAVITY``, because the dual-parabola variant
# reads both at definition time (default arg) and call time.


def foot_contact_required(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Binary cost (1.0) on every step the foot is NOT in ground contact. (num_envs,).

  For the SINE task, where the carriage should glide smoothly up and down with
  the point-foot planted the whole cycle (never hopping off). Returns 1.0 when
  the foot is airborne, 0.0 when it is touching, so a negative cfg weight
  penalizes any loss of contact uniformly across the cycle. This directly targets
  the "short hops as the carriage drops" behaviour: any airborne step is taxed.

  The sensor must request the ``found`` field. Logs the airborne fraction.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, (
    f"Sensor '{sensor_name}' must request the 'found' field for "
    "foot_contact_required."
  )
  in_contact = (found > 0).any(dim=-1)  # [B] bool
  airborne = (~in_contact).float()  # [B] 1.0 when foot has left the ground

  env.extras["log"]["Metrics/foot_airborne_frac"] = airborne.mean()
  return airborne  # [B]


def foot_contact_phase_dual_parabola(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> torch.Tensor:
  """Binary cost (1.0) on every step whose contact state is WRONG for the phase.

  Shape (num_envs,). For the DUAL-PARABOLA task, the foot is supposed to be:
    * AIRBORNE during the FLIGHT (upper) arc, phi in [0, flight_frac), and
    * IN CONTACT during the STANCE/recovery (lower) arc, phi in [flight_frac, 1).

  The cost is 1.0 in either wrong state -- contact during flight OR air during
  stance -- and 0.0 when the contact state matches the phase. A negative cfg
  weight turns this into a penalty that both forces the foot to leave the ground
  for the ballistic arc and forbids it floating through the recovery arc.

  The flight fraction and period are taken from ``dual_parabola_timing`` with the
  SAME geometry/gravity as the slider reward, so the contact schedule and the
  height reference share one phase clock and never drift apart. The sensor must
  request the ``found`` field.

  Args:
    sensor_name: the foot/ground ContactSensor name.
    traj_min, traj_max, traj_transition: trajectory geometry, z rel origin (m)
      (same values passed to the slider reward).
    gravity: free-fall accel for the flight arc (m/s^2), same as the reward.
  """
  t_total, flight_frac, _, _, _ = dual_parabola_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, (
    f"Sensor '{sensor_name}' must request the 'found' field for "
    "foot_contact_phase_dual_parabola."
  )
  in_contact = (found > 0).any(dim=-1)  # [B] bool

  phi = _phase(env, t_total)  # [B], derived period -- matches the slider reward
  in_flight = phi < flight_frac  # [B] bool: should be AIRBORNE here

  # Wrong iff (in flight AND touching) OR (in stance AND airborne).
  wrong = (in_flight & in_contact) | ((~in_flight) & (~in_contact))  # [B]
  cost = wrong.float()  # [B] 1.0 when contact state mismatches the phase

  env.extras["log"]["Metrics/contact_phase_wrong_frac"] = cost.mean()
  env.extras["log"]["Metrics/foot_contact_frac"] = in_contact.float().mean()
  return cost  # [B]


def slider_dual_parabola_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  std: float,
  gravity: float = GRAVITY,
  asset_cfg: SceneEntityCfg = SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking the gravity-exact dual-parabola height. (num_envs,).

  carriage_height_above_start = -slider_pos (verified; see section notes). The
  cycle period is DERIVED from (traj_min, traj_max, traj_transition, gravity) via
  ``dual_parabola_timing`` -- there is no period argument. The same derived
  period must drive the phase clock (the observation and any other phase-keyed
  terms); see ``dual_parabola_period`` in env_cfgs for how this is shared.

  Args:
    traj_min, traj_max, traj_transition: trajectory geometry, z rel origin (m).
    std: Gaussian width on the height-tracking error (meters).
    gravity: free-fall acceleration for the flight arc (m/s^2).
    asset_cfg: entity config selecting the slider joint, joint_names=("slider",).
  """
  t_total, _, _, recovery_accel, _ = dual_parabola_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  asset: Entity = env.scene[asset_cfg.name]
  height = _carriage_height(asset, asset_cfg)  # [B], meters above start

  phi = _phase(env, t_total)  # [B], uses the derived period
  h_ref = dual_parabola_reference(
    phi, traj_min, traj_max, traj_transition, gravity
  )  # [B]

  err = height - h_ref
  reward = torch.exp(-torch.square(err) / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/slider_track_err_mean"] = err.abs().mean()
  env.extras["log"]["Metrics/slider_height_ref_mean"] = h_ref.mean()
  env.extras["log"]["Metrics/slider_height_mean"] = height.mean()
  return reward


##
# Sinusoidal slider trajectory (gentle, first sim-to-real target).
##
# Unlike the dual-parabola, a sinusoid has NO physics-derived period -- the
# period is a free design parameter you supply (``sine_period`` seconds). The
# carriage oscillates smoothly between traj_min and traj_max:
#
#     mid = (traj_max + traj_min) / 2          (vertical centre)
#     amp = (traj_max - traj_min) / 2          (amplitude)
#     h_ref(phi) = mid - amp * cos(2*pi*phi)
#
# Phase convention: with the leading -cos, phi=0 starts at the BOTTOM
# (traj_min), rises to the top (traj_max) at phi=0.5, and returns to the bottom
# at phi=1. That matches the dual-parabola's "start low, push up" feel and means
# both trajectories begin near the crouched DEFAULT_INIT pose, so neither trips
# ``joint_at_limit`` on step 0.
#
# SIM-TO-REAL LEVER (peak acceleration): the carriage acceleration is
#     a(phi) = amp * omega^2 * cos(2*pi*phi),  omega = 2*pi / sine_period
# so the PEAK vertical accel magnitude is amp * omega^2. Keep this comfortably
# BELOW g (9.81) and the foot never goes ballistic -- it stays loaded against the
# floor through the whole cycle, which is exactly the gentle, well-behaved motion
# you want for a first transfer. Shrinking sine_period (faster hop) or growing
# the amplitude both raise this quadratically/linearly; tune sine_period FIRST.


def slider_sinusoid_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  sine_period: float,
  std: float,
  asset_cfg: SceneEntityCfg = SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking a smooth sinusoidal carriage height. (num_envs,).

  carriage_height_above_start = -slider_pos (verified; see section notes). The
  reference is ``mid - amp*cos(2*pi*phi)`` between ``traj_min`` and ``traj_max``.

  The cycle period is the FREE parameter ``sine_period`` (seconds). The SAME
  value must drive the phase clock observation so the policy's phase input stays
  in lockstep with this reward; ``env_cfgs`` wires both from the one constant.

  Args:
    traj_min, traj_max: lowest / highest carriage height, z rel origin (m),
      with traj_max >= traj_min.
    sine_period: cycle period in seconds (a free design choice). The peak
      vertical acceleration is ((traj_max-traj_min)/2) * (2*pi/sine_period)**2;
      keep it below g for a non-ballistic, gentle motion.
    std: Gaussian width on the height-tracking error (meters).
    asset_cfg: entity config selecting the slider joint, joint_names=("slider",).
  """
  assert traj_max >= traj_min, "Require traj_max >= traj_min."
  assert sine_period > 0.0, "Require sine_period > 0."

  mid = 0.5 * (traj_max + traj_min)
  amp = 0.5 * (traj_max - traj_min)

  asset: Entity = env.scene[asset_cfg.name]
  height = _carriage_height(asset, asset_cfg)  # [B], meters above start

  phi = _phase(env, sine_period)  # [B], uses the same free period
  angle = 2.0 * math.pi * phi
  h_ref = mid - amp * torch.cos(angle)  # [B]; phi=0 -> traj_min, phi=0.5 -> traj_max

  err = height - h_ref
  reward = torch.exp(-torch.square(err) / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/slider_track_err_mean"] = err.abs().mean()
  env.extras["log"]["Metrics/slider_height_ref_mean"] = h_ref.mean()
  env.extras["log"]["Metrics/slider_height_mean"] = height.mean()
  return reward


def foot_xy_position_tracking(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  ref_xy: tuple[float, float] = DEFAULT_FOOT_REF_XY,
  std: float = 0.05,
  foot_offset_b: tuple[float, float, float] = FOOT_OFFSET_B,
) -> torch.Tensor:
  """Keep the point-foot world (x, y) at ``ref_xy``. Shape (num_envs,).

  Foot center = calf_assy body pose composed with ``foot_offset_b`` (body frame).
  The hip joint translates the foot in x, so penalizing xy drift stops the leg
  from swinging the foot sideways during the hop.

  Args:
    asset_cfg: entity config selecting the foot body, body_names=("calf_assy",).
    ref_xy: target world (x, y) for the foot center (meters).
    std: Gaussian width on the xy error (meters).
    foot_offset_b: foot-center offset in the calf_assy body frame (meters).
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_id = asset_cfg.body_ids  # single body

  origin = asset.data.body_link_pos_w[:, body_id][:, 0]  # [B, 3]
  quat = asset.data.body_link_quat_w[:, body_id][:, 0]  # [B, 4] (w, x, y, z)

  offset = torch.tensor(
    foot_offset_b, device=origin.device, dtype=origin.dtype
  ).expand(origin.shape[0], 3)  # [B, 3]
  foot_w = origin + _quat_rotate(quat, offset)  # [B, 3]

  ref = torch.tensor(ref_xy, device=origin.device, dtype=origin.dtype)  # [2]
  dist_sq = torch.sum(torch.square(foot_w[:, :2] - ref), dim=-1)  # [B]
  reward = torch.exp(-dist_sq / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/foot_xy_err_mean"] = torch.sqrt(dist_sq).mean()
  return reward
