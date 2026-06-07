"""Custom MDP terms for the Diogenes periodic-hopping task.

This module adds the task-specific functions that mjlab does not ship out of the
box. Generic regularizers (action-rate, joint-limit, joint pos/vel observations,
last-action observation, time-out termination) are reused directly from
``mjlab.envs.mdp``. Generic contact rewards (foot slip, soft landing) and the
electrical-power penalty are reused from ``mjlab.tasks.velocity.mdp`` and
``mjlab.envs.mdp`` respectively; this file only adds what is genuinely specific
to the phase-driven hop stand.

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

Hop gait reward
---------------
Hop amplitude is shaped by a single phase-keyed term, ``peak_hop_height_reward``:
once per cycle (at the phase wrap) it rewards how close the cycle's achieved
*apex* carriage height got to the desired ``hop_height``, via a Gaussian on the
error. There is deliberately no enforced flight/stance schedule: for a ballistic
apex of height ``h`` the flight duration is fixed by physics, but the liftoff
timing depends on how the leg extends and push, which the policy must discover
rather than have prescribed. Letting the apex reward stand alone keeps the timing
unconstrained.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

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
  slider = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return -slider[:, -1]


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


def thigh_ground_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Terminate when the thigh contacts the ground plane. Shape (num_envs,), bool.

  Targets the flat-landing exploit: the policy learned to land with the (nearly
  straight) leg lying lengthwise on the floor, so the impact runs along the limb
  axis and induces almost no knee torque -- sidestepping the joint-limit
  termination. With the thigh shells now collidable against the floor (see
  diogenes.xml), the thigh strikes the ground slightly before the foot in such a
  landing, so any thigh<->floor contact flags the exploit. Terminating on it (a
  genuine failure end, time_out=False) makes the flat landing collect no further
  reward, removing the incentive.

  Reads only the ``found`` field of the thigh contact sensor; the thigh shells
  collide solely with the floor, so a positive ``found`` always means a real
  thigh/ground contact.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found  # [B, N]
  assert found is not None, (
    f"Sensor '{sensor_name}' must request the 'found' field for thigh_ground_contact."
  )
  in_contact = (found > 0).any(dim=-1)  # [B]
  env.extras["log"]["Metrics/thigh_contact_frac"] = in_contact.float().mean()
  return in_contact


class peak_hop_height_reward:
  """Reward the cycle's peak carriage height AND the apex occurring at mid-cycle.

  Two coupled objectives, evaluated once per cycle:

    1. *Height*: the cycle's peak carriage height should match ``hop_height``.
    2. *Timing*: that peak (the apex) should occur at mid-cycle (phase 0.5).

  Why the timing term matters: a pure peak-height reward is indifferent to *when*
  and *how* the apex is reached, so the policy can launch from a partial crouch
  (carriage already above the zero point at takeoff), giving a small ballistic
  rise, a short flight, and a hop period far shorter than ``hop_period`` -- the
  clock ends up decorative. Requiring the apex at mid-cycle forces the apex to
  land at ``0.5 * hop_period`` into the cycle, which (for a ballistic arc that
  starts and ends a cycle on the ground) pins the flight duration and hence the
  whole hop period to ``hop_period``. The policy still chooses the trajectory;
  it just cannot earn full credit with an early, shallow-rise apex.

  Mechanics (mirrors the velocity task's ``feet_swing_height`` accumulator, but
  keyed to the phase clock):

    * Every step, track the running maximum carriage height in the current cycle
      (``running_peak``) AND the phase at which that maximum occurred
      (``peak_phase``).
    * Detect the cycle boundary by watching the phase wrap ``phi: ~1 -> ~0``.
      On the wrap step a cycle has *completed*, so we emit the combined reward
      and reset the accumulators. On all other steps the term emits 0.

  Reward shape (emitted on the wrap step):

      height_term = exp(-(peak - hop_height)**2 / std**2)            in [0, 1]
      timing_term = exp(-(peak_phase - 0.5)**2 / phase_std**2)       in [0, 1]
      reward      = height_term * timing_term

  Multiplying (rather than adding) means BOTH must be satisfied for credit: a
  perfectly-high apex at the wrong time, or a well-timed apex at the wrong
  height, both score low. Using a product keeps the term bounded in [0, 1].

  Because the reward fires only on the wrap step (one control step per
  ``hop_period``), the per-second contribution is sparse; tune ``weight``
  accordingly.

  Args:
    hop_height: Desired apex height, metres above start.
    hop_period: Cycle duration, seconds (also the phase-clock period).
    std: Gaussian width on the height error, metres.
    phase_std: Gaussian width on the apex-phase error, in phase units (fraction
      of a cycle). ~0.12 lets the apex sit comfortably within the middle ~quarter
      of the cycle before credit falls off; tighten to demand sharper timing.
    asset_cfg: Entity config selecting the slider joint.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.running_peak = torch.full(
      (env.num_envs,), -1.0e9, device=env.device, dtype=torch.float32
    )
    # Phase at which the running peak was last attained.
    self.peak_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    # Previous-step phase, used to detect the wrap (decrease) boundary.
    self.prev_phi = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.running_peak[env_ids] = -1.0e9
    self.peak_phase[env_ids] = 0.0
    self.prev_phi[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    hop_height: float = 0.35,
    hop_period: float = 0.6,
    std: float = 0.05,
    phase_std: float = 0.12,
    asset_cfg: SceneEntityCfg = SLIDER_CFG,
  ) -> torch.Tensor:
    height = _height_above_start(env, asset_cfg)  # [B]
    phi = _phase(env, hop_period)  # [B]

    # A cycle completes when the phase wraps (this step's phi < last step's phi).
    wrapped = phi < self.prev_phi  # [B]

    # The completed cycle's apex is what the accumulators hold BEFORE this step's
    # (wrap-step) height is folded in -- the wrap step's height belongs to the
    # NEW cycle. So evaluate the reward against the current accumulators first.
    height_err = self.running_peak - hop_height
    height_term = torch.exp(-torch.square(height_err) / (std**2))
    # Apex should occur at mid-cycle (phase 0.5).
    phase_err = self.peak_phase - 0.5
    timing_term = torch.exp(-torch.square(phase_err) / (phase_std**2))
    reward_at_apex = height_term * timing_term
    reward = torch.where(wrapped, reward_at_apex, torch.zeros_like(height_err))

    # Log the completed-cycle peak and apex phase at each cycle boundary.
    num_wraps = torch.sum(wrapped.float())
    denom = torch.clamp(num_wraps, min=1.0)
    env.extras["log"]["Metrics/peak_hop_height_mean"] = (
      torch.sum(self.running_peak * wrapped.float()) / denom
    )
    env.extras["log"]["Metrics/apex_phase_mean"] = (
      torch.sum(self.peak_phase * wrapped.float()) / denom
    )

    # Re-seed accumulators for wrapped envs with THIS step's sample (start of the
    # new cycle); otherwise fold this step's height into the running max.
    seed_peak = torch.where(
      wrapped, height, torch.maximum(self.running_peak, height)
    )
    is_new_peak = height > self.running_peak  # [B]
    seed_phase = torch.where(
      wrapped,
      phi,  # new cycle starts with this sample as its provisional apex
      torch.where(is_new_peak, phi, self.peak_phase),
    )
    self.running_peak = seed_peak
    self.peak_phase = seed_phase
    self.prev_phi = phi

    return reward
