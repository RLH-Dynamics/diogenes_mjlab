"""Reward functions for the Diogenes periodic-hopping task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from ..constants import GRAVITY
from ..accessors import carriage_height as _carriage_height
from ..accessors import carriage_vel as _carriage_vel
from ..accessors import carriage_acc as _carriage_acc
from .trajectories import (
  dual_parabola_timing,
  dual_parabola_reference,
  dual_parabola_velocity,
  dual_parabola_acceleration,
  spring_timing,
  spring_reference,
  spring_velocity,
  spring_acceleration,
)
from .observations import _phase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Default slider entity config (mirrors the module-level SLIDER_CFG in __init__).
_SLIDER_CFG = SceneEntityCfg("robot", joint_names=("slider",))

# Foot-center offset in the calf_assy body frame (meters).
FOOT_OFFSET_B: tuple[float, float, float] = (-0.176776, 0.176777, -0.014)

# Foot-center world (x, y) at the default pose.
DEFAULT_FOOT_REF_XY: tuple[float, float] = (0.00250, -0.10679)


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate vec by quat (w, x, y, z), batched. quat:[B,4], vec:[B,3] -> [B,3]."""
  w = quat[:, 0:1]
  xyz = quat[:, 1:4]
  t = 2.0 * torch.cross(xyz, vec, dim=-1)
  return vec + w * t + torch.cross(xyz, t, dim=-1)


def lateral_contact_force_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize lateral (world x, y) foot/ground contact force. Shape (num_envs,)."""
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
  """Penalize vertical (world z) foot/ground contact force. Shape (num_envs,)."""
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


def foot_contact_required(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Binary cost (1.0) on every step the foot is NOT in ground contact. (num_envs,)."""
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

  Shape (num_envs,).
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


def foot_contact_phase_spring(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> torch.Tensor:
  """Binary cost (1.0) on every step whose contact state is WRONG for the phase.

  Spring analogue of :func:`foot_contact_phase_dual_parabola`: the spring hop has
  the SAME flight arc, so the foot should be AIRBORNE during flight
  (``phi < flight_frac``) and IN CONTACT during the spring (stance) arc.

  Shape (num_envs,).
  """
  t_total, flight_frac, _, _, _ = spring_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, (
    f"Sensor '{sensor_name}' must request the 'found' field for "
    "foot_contact_phase_spring."
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


def is_specific_termination(
  env: ManagerBasedRlEnv,
  term_name: str,
) -> torch.Tensor:
  """Binary cost (1.0) on the step a SPECIFIC termination term fires. (num_envs,).

  Reads the named term's done buffer from the termination manager, which is
  computed immediately before the rewards each step, so the returned mask aligns
  with the exact step on which that termination triggers.  Pair with a negative
  weight to apply a penalty dedicated to one termination (independent of the
  blanket ``termination_penalty``).

  Args:
    term_name: the termination dict key registered in the env config.
  """
  return env.termination_manager.get_term(term_name).float()


def slider_dual_parabola_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  std: float,
  gravity: float = GRAVITY,
  asset_cfg: SceneEntityCfg = _SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking the gravity-exact dual-parabola height. (num_envs,).

  Also logs a diagnostic: the mean carriage acceleration (+up) measured over the
  envs currently in the non-flight "ground contact" recovery arc. See the comment
  block near the end for why it lives on this term.
  """
  t_total, flight_frac, _, _, _ = dual_parabola_timing(
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

  # Diagnostic: mean carriage acceleration (+up) over the envs currently in the
  # non-flight "ground contact" recovery arc (phi >= flight_frac). Hosted on this
  # always-on position term -- NOT the acceleration-tracking term, which the
  # RewardManager skips whenever its weight is 0 -- so it reliably reaches the
  # training terminal summary and W&B every run. (Sampled per step, then averaged
  # over the rollout by the runner; with many envs at random phases a near-constant
  # ~(1 - flight_frac) fraction is in contact each step, so the clamp below is just
  # a divide-by-zero guard.)
  acc = _carriage_acc(asset, asset_cfg)  # [B], +up (m/s^2)
  in_contact_phase = (phi >= flight_frac).float()  # [B], 1.0 during recovery arc
  contact_acc_mean = (acc * in_contact_phase).sum() / torch.clamp(
    in_contact_phase.sum(), min=1.0
  )
  env.extras["log"]["Metrics/slider_contact_acc_mean"] = contact_acc_mean
  env.extras["log"]["Metrics/slider_contact_phase_frac"] = in_contact_phase.mean()
  return reward


def slider_sinusoid_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  sine_period: float,
  std: float,
  asset_cfg: SceneEntityCfg = _SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking a smooth sinusoidal carriage height. (num_envs,)."""
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


def slider_dual_parabola_velocity_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  std: float,
  gravity: float = GRAVITY,
  asset_cfg: SceneEntityCfg = _SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking the dual-parabola velocity profile. (num_envs,)."""
  t_total, _, _, _, _ = dual_parabola_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  asset: Entity = env.scene[asset_cfg.name]
  vel = _carriage_vel(asset, asset_cfg)  # [B], +up (m/s)

  phi = _phase(env, t_total)  # [B], uses the derived period
  v_ref = dual_parabola_velocity(
    phi, traj_min, traj_max, traj_transition, gravity
  )  # [B]

  err = vel - v_ref
  reward = torch.exp(-torch.square(err) / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/slider_vel_track_err_mean"] = err.abs().mean()
  env.extras["log"]["Metrics/slider_vel_ref_mean"] = v_ref.mean()
  env.extras["log"]["Metrics/slider_vel_mean"] = vel.mean()
  return reward


def slider_dual_parabola_acceleration_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  std: float,
  gravity: float = GRAVITY,
  asset_cfg: SceneEntityCfg = _SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking the dual-parabola acceleration profile. (num_envs,)."""
  t_total, _, _, _, _ = dual_parabola_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  asset: Entity = env.scene[asset_cfg.name]
  acc = _carriage_acc(asset, asset_cfg)  # [B], +up (m/s^2)

  phi = _phase(env, t_total)  # [B], uses the derived period
  a_ref = dual_parabola_acceleration(
    phi, traj_min, traj_max, traj_transition, gravity
  )  # [B]

  err = acc - a_ref
  reward = torch.exp(-torch.square(err) / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/slider_acc_track_err_mean"] = err.abs().mean()
  env.extras["log"]["Metrics/slider_acc_ref_mean"] = a_ref.mean()
  env.extras["log"]["Metrics/slider_acc_mean"] = acc.mean()
  return reward


def slider_spring_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  std: float,
  gravity: float = GRAVITY,
  asset_cfg: SceneEntityCfg = _SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking the spring-hop height. (num_envs,).

  Free-fall flight arc joined to a Hooke's-law spring contact arc (see
  :func:`mdp.trajectories.spring_reference`). Mirrors the diagnostics hosted on
  :func:`slider_dual_parabola_tracking` (mean carriage accel over the contact arc).
  """
  t_total, flight_frac, _, _, _ = spring_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  asset: Entity = env.scene[asset_cfg.name]
  height = _carriage_height(asset, asset_cfg)  # [B], meters above start

  phi = _phase(env, t_total)  # [B], uses the derived period
  h_ref = spring_reference(
    phi, traj_min, traj_max, traj_transition, gravity
  )  # [B]

  err = height - h_ref
  reward = torch.exp(-torch.square(err) / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/slider_track_err_mean"] = err.abs().mean()
  env.extras["log"]["Metrics/slider_height_ref_mean"] = h_ref.mean()
  env.extras["log"]["Metrics/slider_height_mean"] = height.mean()

  # Diagnostic: mean carriage acceleration (+up) over the envs currently in the
  # non-flight spring (contact) arc (phi >= flight_frac). Hosted on this always-on
  # position term -- NOT the acceleration-tracking term, which the RewardManager
  # skips whenever its weight is 0 -- so it reliably reaches the training terminal
  # summary and W&B every run.
  acc = _carriage_acc(asset, asset_cfg)  # [B], +up (m/s^2)
  in_contact_phase = (phi >= flight_frac).float()  # [B], 1.0 during spring arc
  contact_acc_mean = (acc * in_contact_phase).sum() / torch.clamp(
    in_contact_phase.sum(), min=1.0
  )
  env.extras["log"]["Metrics/slider_contact_acc_mean"] = contact_acc_mean
  env.extras["log"]["Metrics/slider_contact_phase_frac"] = in_contact_phase.mean()
  return reward


def slider_spring_velocity_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  std: float,
  gravity: float = GRAVITY,
  asset_cfg: SceneEntityCfg = _SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking the spring-hop velocity profile. (num_envs,)."""
  t_total, _, _, _, _ = spring_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  asset: Entity = env.scene[asset_cfg.name]
  vel = _carriage_vel(asset, asset_cfg)  # [B], +up (m/s)

  phi = _phase(env, t_total)  # [B], uses the derived period
  v_ref = spring_velocity(
    phi, traj_min, traj_max, traj_transition, gravity
  )  # [B]

  err = vel - v_ref
  reward = torch.exp(-torch.square(err) / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/slider_vel_track_err_mean"] = err.abs().mean()
  env.extras["log"]["Metrics/slider_vel_ref_mean"] = v_ref.mean()
  env.extras["log"]["Metrics/slider_vel_mean"] = vel.mean()
  return reward


def slider_spring_acceleration_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  std: float,
  gravity: float = GRAVITY,
  asset_cfg: SceneEntityCfg = _SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking the spring-hop acceleration profile. (num_envs,)."""
  t_total, _, _, _, _ = spring_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  asset: Entity = env.scene[asset_cfg.name]
  acc = _carriage_acc(asset, asset_cfg)  # [B], +up (m/s^2)

  phi = _phase(env, t_total)  # [B], uses the derived period
  a_ref = spring_acceleration(
    phi, traj_min, traj_max, traj_transition, gravity
  )  # [B]

  err = acc - a_ref
  reward = torch.exp(-torch.square(err) / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/slider_acc_track_err_mean"] = err.abs().mean()
  env.extras["log"]["Metrics/slider_acc_ref_mean"] = a_ref.mean()
  env.extras["log"]["Metrics/slider_acc_mean"] = acc.mean()
  return reward


def slider_sinusoid_velocity_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  sine_period: float,
  std: float,
  asset_cfg: SceneEntityCfg = _SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking a smooth sinusoidal carriage velocity. (num_envs,)."""
  assert traj_max >= traj_min, "Require traj_max >= traj_min."
  assert sine_period > 0.0, "Require sine_period > 0."

  amp = 0.5 * (traj_max - traj_min)
  omega = 2.0 * math.pi / sine_period

  asset: Entity = env.scene[asset_cfg.name]
  vel = _carriage_vel(asset, asset_cfg)  # [B], +up (m/s)

  phi = _phase(env, sine_period)  # [B], same free period as the position term
  # h_ref = mid - amp*cos(2π·phi)  =>  dh/dt = amp*omega*sin(2π·phi).
  v_ref = amp * omega * torch.sin(2.0 * math.pi * phi)  # [B]

  err = vel - v_ref
  reward = torch.exp(-torch.square(err) / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/slider_vel_track_err_mean"] = err.abs().mean()
  env.extras["log"]["Metrics/slider_vel_ref_mean"] = v_ref.mean()
  env.extras["log"]["Metrics/slider_vel_mean"] = vel.mean()
  return reward


def slider_sinusoid_acceleration_tracking(
  env: ManagerBasedRlEnv,
  traj_min: float,
  traj_max: float,
  sine_period: float,
  std: float,
  asset_cfg: SceneEntityCfg = _SLIDER_CFG,
) -> torch.Tensor:
  """Dense Gaussian reward tracking a smooth sinusoidal carriage acceleration. (num_envs,)."""
  assert traj_max >= traj_min, "Require traj_max >= traj_min."
  assert sine_period > 0.0, "Require sine_period > 0."

  amp = 0.5 * (traj_max - traj_min)
  omega = 2.0 * math.pi / sine_period

  asset: Entity = env.scene[asset_cfg.name]
  acc = _carriage_acc(asset, asset_cfg)  # [B], +up (m/s^2)

  phi = _phase(env, sine_period)  # [B], same free period as the position term
  # h_ref = mid - amp*cos(2π·phi)  =>  d2h/dt2 = amp*omega^2*cos(2π·phi).
  a_ref = amp * (omega**2) * torch.cos(2.0 * math.pi * phi)  # [B]

  err = acc - a_ref
  reward = torch.exp(-torch.square(err) / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/slider_acc_track_err_mean"] = err.abs().mean()
  env.extras["log"]["Metrics/slider_acc_ref_mean"] = a_ref.mean()
  env.extras["log"]["Metrics/slider_acc_mean"] = acc.mean()
  return reward


def foot_xy_position_tracking(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  ref_xy: tuple[float, float] = DEFAULT_FOOT_REF_XY,
  std: float = 0.05,
  foot_offset_b: tuple[float, float, float] = FOOT_OFFSET_B,
) -> torch.Tensor:
  """Keep the point-foot (x, y) at ``ref_xy`` in the env-local frame. (num_envs,).

  ``ref_xy`` is expressed relative to each env's origin, so the foot world (x, y)
  must have its per-env grid origin subtracted before the comparison.  Without
  this, every env off the world origin (the scene lays robots on an
  ``env_spacing`` grid) carries a constant tens-of-metres offset, the Gaussian
  underflows to exactly 0 and the term -- and its gradient -- dies.
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_id = asset_cfg.body_ids  # single body

  origin = asset.data.body_link_pos_w[:, body_id][:, 0]  # [B, 3]
  quat = asset.data.body_link_quat_w[:, body_id][:, 0]  # [B, 4] (w, x, y, z)

  offset = torch.tensor(
    foot_offset_b, device=origin.device, dtype=origin.dtype
  ).expand(origin.shape[0], 3)  # [B, 3]
  foot_w = origin + _quat_rotate(quat, offset)  # [B, 3] world frame

  # Re-express the foot (x, y) relative to this env's grid origin so it lines up
  # with ``ref_xy`` (which is env-local).  env_origins is [B, 3].
  foot_xy = foot_w[:, :2] - env.scene.env_origins[:, :2]  # [B, 2] env-local

  ref = torch.tensor(ref_xy, device=origin.device, dtype=origin.dtype)  # [2]
  dist_sq = torch.sum(torch.square(foot_xy - ref), dim=-1)  # [B]
  reward = torch.exp(-dist_sq / (std**2))  # [B] in (0, 1]

  env.extras["log"]["Metrics/foot_xy_err_mean"] = torch.sqrt(dist_sq).mean()
  return reward
