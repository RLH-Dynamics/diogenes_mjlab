"""Observation and reward terms for the Harold biped tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.noise.noise_cfg import NoiseModelCfg
from mjlab.utils.noise.noise_model import NoiseModel

from ..mdp.observations import _phase
from . import gait

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _feet_state_b(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """Foot site position and velocity relative to the base, in the base frame.

  Works for both the welded (suspended) and floating (walking) base: the base
  motion is removed before rotating into the base frame.

  Returns (pos_b, vel_b), each shape (num_envs, num_feet, 3).
  """
  asset: Entity = env.scene[asset_cfg.name]
  site_pos = asset.data.site_pos_w[:, asset_cfg.site_ids]  # [B, F, 3]
  site_vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]  # [B, F, 3]

  root_pos = asset.data.root_link_pos_w.unsqueeze(1)  # [B, 1, 3]
  root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(
    -1, site_pos.shape[1], -1
  )  # [B, F, 4]
  root_lin_vel = asset.data.root_link_vel_w[:, 0:3].unsqueeze(1)  # [B, 1, 3]
  root_ang_vel = asset.data.root_link_vel_w[:, 3:6].unsqueeze(1).expand_as(site_pos)

  rel_w = site_pos - root_pos
  vel_rel_w = site_vel - root_lin_vel - torch.cross(root_ang_vel, rel_w, dim=-1)
  return quat_apply_inverse(root_quat, rel_w), quat_apply_inverse(root_quat, vel_rel_w)


def _feet_reference_b(
  env: ManagerBasedRlEnv, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reference foot (pos, vel) in the base frame, each (num_envs, num_feet, 3)."""
  phase = _phase(env, gait.GAIT_PERIOD).to(dtype)
  return gait.foot_reference(
    gait.leg_phases(phase), gait.foot_centers(device=device, dtype=dtype)
  )


##
# Observations.
##


def imu_angular_velocity(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Gyro reading (rad/s) in the IMU site frame. Shape (num_envs, 3)."""
  sensor: BuiltinSensor = env.scene[sensor_name]
  return sensor.data


def imu_projected_gravity(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Unit gravity direction in the IMU site frame. Shape (num_envs, 3).

  Reads a ``framezaxis`` sensor giving world +z (up) in the site frame, so this
  is its negation: it points down, e.g. (0, -1, 0) for a level BNO085 (+y up).
  """
  sensor: BuiltinSensor = env.scene[sensor_name]
  return -sensor.data


class GyroNoiseModel(NoiseModel):
  """Gyro error model: reading = true * (1 + scale) + bias + white noise.

  ``scale`` and ``bias`` are drawn per env and per axis at every episode reset
  (a constant calibration error while the policy runs); the white noise is
  ``noise_cfg``, drawn every step.
  """

  def __init__(self, noise_model_cfg: GyroNoiseModelCfg, num_envs: int, device: str):
    super().__init__(noise_model_cfg, num_envs, device)
    self._bias_range = noise_model_cfg.bias_range
    self._scale_range = noise_model_cfg.scale_range
    self.bias = torch.zeros((num_envs, 3), device=device)
    self.scale = torch.ones((num_envs, 3), device=device)
    self.reset()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    indices = slice(None) if env_ids is None else env_ids
    n = self.bias[indices].shape[0]

    def uniform(bounds: tuple[float, float]) -> torch.Tensor:
      lo, hi = bounds
      return lo + (hi - lo) * torch.rand((n, 3), device=self._device)

    self.bias[indices] = uniform(self._bias_range)
    self.scale[indices] = 1.0 + uniform(self._scale_range)

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    return super().__call__(data * self.scale + self.bias)


@dataclass(kw_only=True)
class GyroNoiseModelCfg(NoiseModelCfg, class_type=GyroNoiseModel):
  """Per-episode gyro bias (rad/s) and scale error (fraction), plus per-step noise."""

  bias_range: tuple[float, float] = (0.0, 0.0)
  scale_range: tuple[float, float] = (0.0, 0.0)


def foot_contact_forces_heading(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Net foot contact forces in the robot's heading (yaw-only) frame, signed-log
  scaled like mjlab's ``foot_contact_forces``. Shape (num_envs, 3 * num_feet).

  The sensor reports world-frame forces, whose left/right mirror depends on the
  robot's heading; in the heading frame it is a fixed map (see symmetry.py).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # [B, F, 3], world frame (reduce="netforce")
  assert force is not None, f"Sensor '{sensor_name}' must request the 'force' field."
  asset: Entity = env.scene[asset_cfg.name]
  heading = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, force.shape[1], -1)
  flat = quat_apply_inverse(heading, force).flatten(1)
  return torch.sign(flat) * torch.log1p(torch.abs(flat))


def feet_pos_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Foot site positions in the base frame. Shape (num_envs, 3 * num_feet)."""
  pos_b, _ = _feet_state_b(env, asset_cfg)
  return pos_b.flatten(1)


def feet_ref_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reference foot positions in the base frame. Shape (num_envs, 3 * num_feet)."""
  ref_pos, _ = _feet_reference_b(env, env.device, torch.float32)
  return ref_pos.flatten(1)


##
# Rewards.
##


def feet_position_tracking(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
  """Mean over feet of exp(-|p - p_ref|^2 / std^2), base frame. Shape (num_envs,)."""
  pos_b, _ = _feet_state_b(env, asset_cfg)
  ref_pos, _ = _feet_reference_b(env, pos_b.device, pos_b.dtype)
  err_sq = torch.sum(torch.square(pos_b - ref_pos), dim=-1)  # [B, F]

  err = torch.sqrt(err_sq)
  for i, name in enumerate(gait.FOOT_SITE_NAMES):
    env.extras["log"][f"Metrics/{name}_pos_err_mean"] = err[:, i].mean()
  return torch.exp(-err_sq / std**2).mean(dim=-1)


def feet_velocity_tracking(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
  """Mean over feet of exp(-|v - v_ref|^2 / std^2), base frame. Shape (num_envs,)."""
  _, vel_b = _feet_state_b(env, asset_cfg)
  _, ref_vel = _feet_reference_b(env, vel_b.device, vel_b.dtype)
  err_sq = torch.sum(torch.square(vel_b - ref_vel), dim=-1)  # [B, F]

  env.extras["log"]["Metrics/feet_vel_err_mean"] = torch.sqrt(err_sq).mean()
  return torch.exp(-err_sq / std**2).mean(dim=-1)


##
# Walking: torso height and contact schedule.
##


def base_height_tracking(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """exp(-(h - target)^2 / std^2) on the root height above flat ground. (num_envs,)."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  env.extras["log"]["Metrics/base_height_mean"] = height.mean()
  return torch.exp(-torch.square(height - target_height) / std**2)


def expected_stance(leg_phase: torch.Tensor, swing_fraction: float) -> torch.Tensor:
  """True where a foot should be on the ground.

  Each leg's cycle is stance for leg_phase in [0, 1 - swing_fraction), then
  swing (airborne) for the rest. With the legs half a cycle apart, both feet
  are down around each handover when swing_fraction < 0.5.
  """
  return leg_phase < (1.0 - swing_fraction)


def contact_schedule_match(
  in_contact: torch.Tensor,
  leg_phase: torch.Tensor,
  swing_fraction: float,
  transition_margin: float,
) -> torch.Tensor:
  """1.0 where contact matches the schedule, else 0.0. Shape of in_contact.

  Within ``transition_margin`` (cycle fraction) of a lift-off (leg_phase =
  1 - swing_fraction) or touch-down (leg_phase = 0) either state counts as a
  match, so finite lift-off/landing time isn't punished.
  """
  matches = in_contact == expected_stance(leg_phase, swing_fraction)
  to_touchdown = torch.minimum(leg_phase, 1.0 - leg_phase)
  to_liftoff = torch.abs(leg_phase - (1.0 - swing_fraction))
  near_transition = torch.minimum(to_touchdown, to_liftoff) < transition_margin
  return (matches | near_transition).float()


class feet_contact_schedule:
  """Reward feet for following the fixed-frequency stepping schedule.

  Returns the mean over feet of :func:`contact_schedule_match`. Also logs how
  often each foot actually touches down (an exponential moving average over
  ~2 s), which should settle at the configured step frequency.
  """

  RATE_TIME_CONSTANT_S = 2.0

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.step_dt = env.step_dt
    # [num_envs, num_feet]; sized from the sensor output on the first call.
    self.touchdown_rate: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    step_frequency: float,
    swing_fraction: float,
    transition_margin: float,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None, f"Sensor '{sensor_name}' must request the 'found' field."
    in_contact = found > 0  # [B, F], F in gait.FOOT_SITE_NAMES order

    phase = _phase(env, 1.0 / step_frequency)
    leg_phase = gait.leg_phases(phase)  # [B, F], right half a cycle behind left
    match = contact_schedule_match(in_contact, leg_phase, swing_fraction, transition_margin)

    touchdowns = sensor.compute_first_contact(dt=self.step_dt).float()
    if self.touchdown_rate is None:
      self.touchdown_rate = torch.zeros_like(touchdowns)
    alpha = self.step_dt / self.RATE_TIME_CONSTANT_S
    self.touchdown_rate = (1.0 - alpha) * self.touchdown_rate + alpha * touchdowns / self.step_dt

    log = env.extras["log"]
    log["Metrics/contact_schedule_match"] = match.mean()
    for i, name in enumerate(gait.FOOT_SITE_NAMES):
      log[f"Metrics/{name}_touchdown_hz"] = self.touchdown_rate[:, i].mean()
    return match.mean(dim=-1)


def _cycle_distance(leg_phase: torch.Tensor, target: float) -> torch.Tensor:
  """Shortest distance around the unit cycle from leg_phase to target."""
  diff = torch.remainder(leg_phase - target, 1.0)
  return torch.minimum(diff, 1.0 - diff)


def off_schedule_events(
  touchdown: torch.Tensor,
  liftoff: torch.Tensor,
  leg_phase: torch.Tensor,
  swing_fraction: float,
  event_margin: float,
) -> torch.Tensor:
  """Count lift-offs/touch-downs outside their scheduled windows. Shape of inputs.

  A touch-down is expected within ``event_margin`` (cycle fraction) of
  leg_phase = 0, a lift-off within ``event_margin`` of 1 - swing_fraction.
  """
  late_or_early_touchdown = touchdown & (_cycle_distance(leg_phase, 0.0) > event_margin)
  late_or_early_liftoff = liftoff & (
    _cycle_distance(leg_phase, 1.0 - swing_fraction) > event_margin
  )
  return late_or_early_touchdown.float() + late_or_early_liftoff.float()


def off_schedule_contact_events(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  step_frequency: float,
  swing_fraction: float,
  event_margin: float,
) -> torch.Tensor:
  """Number of off-schedule foot lift-offs/touch-downs this step. (num_envs,).

  Use with a negative weight: it penalises extra taps between the scheduled
  steps, which the contact-schedule reward alone barely notices.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  touchdown = sensor.compute_first_contact(dt=env.step_dt)  # [B, F]
  liftoff = sensor.compute_first_air(dt=env.step_dt)  # [B, F]
  leg_phase = gait.leg_phases(_phase(env, 1.0 / step_frequency))

  events = off_schedule_events(touchdown, liftoff, leg_phase, swing_fraction, event_margin)
  env.extras["log"]["Metrics/off_schedule_events_hz"] = events.sum(dim=-1).mean() / env.step_dt
  return events.sum(dim=-1)


##
# Events.
##


def reset_joints_by_offset_per_joint(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  offsets: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Reset joints to the default pose plus a uniform offset drawn per joint.

  ``offsets`` maps joint-name patterns to (low, high) in rad; every selected
  joint must match one. Positions are clamped into the soft joint limits and
  velocities zeroed.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  ids = asset_cfg.joint_ids
  if isinstance(ids, slice):
    ids = list(range(len(asset.joint_names)))[ids]
  names = [asset.joint_names[i] for i in ids]
  idx, _, ranges = resolve_matching_names_values(offsets, names, preserve_order=True)
  assert sorted(idx) == list(range(len(names))), f"offsets {offsets} must cover {names}"
  bounds = torch.zeros(len(names), 2, device=env.device)
  bounds[idx] = torch.tensor(ranges, dtype=torch.float, device=env.device)

  pos = asset.data.default_joint_pos[env_ids][:, ids].clone()
  u = torch.rand_like(pos)
  pos += bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u
  limits = asset.data.soft_joint_pos_limits[env_ids][:, ids]
  pos = pos.clamp(limits[..., 0], limits[..., 1])
  joint_ids = torch.as_tensor(ids, device=env.device)
  asset.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=env_ids, joint_ids=joint_ids)
