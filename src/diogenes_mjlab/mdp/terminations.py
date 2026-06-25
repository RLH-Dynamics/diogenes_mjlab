"""Termination functions for the Diogenes periodic-hopping task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from ..constants import GRAVITY
from .observations import _phase
from .trajectories import dual_parabola_timing

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def joint_at_limit(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  margin: float = 0.02,
) -> torch.Tensor:
  """Terminate when ANY selected joint reaches (within ``margin``) a hard limit.

  Shape (num_envs,), bool.

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


def foot_contact_phase_wrong_dual_parabola(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
  phase_margin: float = 0.15,
) -> torch.Tensor:
  """Terminate when foot/ground contact is WRONG for the dual-parabola phase.

  Shape (num_envs,), bool.

  Expected contact state mirrors :func:`mdp.rewards.foot_contact_phase_dual_parabola`:
  the foot should be AIRBORNE during the flight arc (``phi < flight_frac``) and
  IN CONTACT during the recovery (stance) arc.  Contact-state transitions happen
  at ``phi == 0`` (liftoff) and ``phi == flight_frac`` (landing).

  A band of ``+- phase_margin`` (in cycle fraction) around each transition is
  EXEMPT, so neither the finite liftoff/landing time nor the grounded reset pose
  at ``phi ~ 0`` triggers a spurious termination.  Only a clear, sustained
  mismatch away from a transition ends the episode.

  Args:
    sensor_name: foot/ground contact sensor (must expose the ``found`` field).
    traj_min, traj_max, traj_transition, gravity: dual-parabola geometry; the
      derived period and flight fraction are recomputed from these so the phase
      here matches the slider-tracking reward exactly.
    phase_margin: half-width (cycle fraction) of the exempt band around each
      contact-state transition.
  """
  t_total, flight_frac, _, _, _ = dual_parabola_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, (
    f"Sensor '{sensor_name}' must request the 'found' field for "
    "foot_contact_phase_wrong_dual_parabola."
  )
  in_contact = (found > 0).any(dim=-1)  # [B] bool

  phi = _phase(env, t_total)  # [B], derived period -- matches the slider reward
  in_flight = phi < flight_frac  # [B] bool: should be AIRBORNE here

  # Wrong iff (in flight AND touching) OR (in stance AND airborne).
  wrong = (in_flight & in_contact) | ((~in_flight) & (~in_contact))  # [B]

  # Exempt a band around each transition: phi ~ 0/1 (liftoff) and phi ~
  # flight_frac (landing).  The 0/1 band also covers the grounded reset pose.
  near_transition = (
    (phi < phase_margin)
    | (phi > 1.0 - phase_margin)
    | (torch.abs(phi - flight_frac) < phase_margin)
  )  # [B] bool

  terminate = wrong & ~near_transition  # [B]

  env.extras["log"]["Metrics/contact_phase_terminate_frac"] = terminate.float().mean()
  return terminate  # [B]


def foot_not_in_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Terminate when the foot is NOT in ground contact.

  Shape (num_envs,), bool.

  For the sinusoidal squat trajectory the foot must stay planted at all times
  (mirrors the :func:`mdp.rewards.foot_contact_required` penalty), so any loss
  of contact is a phase violation that ends the episode.  The default reset pose
  starts grounded, so no transition tolerance is needed here.

  Args:
    sensor_name: foot/ground contact sensor (must expose the ``found`` field).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, (
    f"Sensor '{sensor_name}' must request the 'found' field for "
    "foot_not_in_contact."
  )
  airborne = ~(found > 0).any(dim=-1)  # [B] bool: True when the foot has left

  env.extras["log"]["Metrics/contact_phase_terminate_frac"] = airborne.float().mean()
  return airborne  # [B]
