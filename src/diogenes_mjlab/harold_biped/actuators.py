"""RS03 actuator model for the Harold walking task.

The suspended task uses the XML <position> actuators (kp=60, kv=4, +-60 N.m,
targets clamped to the joint range). Walking loads the motors far harder, so it
swaps them for a PD with the same gains, clamp and torque limit plus the RS03's
torque-speed limit at Harold's battery voltage:

    torque limit(w) = min(60, SAT * (1 - w / w0))   (motoring; braking is looser)

  * w0, the no-load speed, scales with supply voltage: 195 rpm (20.4 rad/s) at
    the rated 48 V. Harold runs on two 18 V-nominal packs in series, ~40 V full,
    36 V nominal and ~32 V near empty, so w0 is ~13.6-17 rad/s; each env draws
    its battery voltage at every reset (``randomize_battery_voltage``).
  * SAT is chosen so the full 60 N.m holds up to 60% of w0, then falls linearly
    to zero at w0. The RS03 datasheet gives no curve; this is the usual shape of
    a current-limited motor running into its voltage limit, and errs towards
    less torque at speed.

The real motor runs the same PD law itself (MIT mode), and diogenes_control
clamps every target to the joint's ctrl_range (the joint range) before sending.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

#: RobStride RS03 ratings (N.m, rad/s, V).
RS03_PEAK_TORQUE = 60.0
RS03_NO_LOAD_SPEED_RATED = 195.0 * 2.0 * torch.pi / 60.0
RS03_RATED_VOLTAGE = 48.0

#: Fraction of the no-load speed up to which the full peak torque is available.
FULL_TORQUE_SPEED_FRACTION = 0.6
RS03_SATURATION_TORQUE = RS03_PEAK_TORQUE / (1.0 - FULL_TORQUE_SPEED_FRACTION)

#: Harold's battery: two 18 V-nominal packs in series.
BATTERY_NOMINAL_VOLTAGE = 36.0
BATTERY_VOLTAGE_RANGE = (32.0, 40.0)

#: MIT-mode gains, as the suspended task and diogenes_control.
KP = 60.0
KD = 4.0


def no_load_speed(voltage: float) -> float:
  """RS03 no-load speed (rad/s) at a supply voltage (V)."""
  return RS03_NO_LOAD_SPEED_RATED * voltage / RS03_RATED_VOLTAGE


@dataclass(kw_only=True)
class Rs03ActuatorCfg(DcMotorActuatorCfg):
  """PD + torque-speed limit, targets clamped to the joint range."""

  stiffness: float = KP
  damping: float = KD
  effort_limit: float = RS03_PEAK_TORQUE
  saturation_effort: float = RS03_SATURATION_TORQUE
  velocity_limit: float = no_load_speed(BATTERY_NOMINAL_VOLTAGE)

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> Rs03Actuator:
    return Rs03Actuator(self, entity, target_ids, target_names)


class Rs03Actuator(DcMotorActuator[Rs03ActuatorCfg]):
  """DC-motor PD whose targets are clamped to the joint range.

  The corner velocity mjlab caches at start-up is recomputed on every call, so
  per-env effort limits (DR) and no-load speeds (battery voltage) both apply.
  """

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    # target_ids index the entity's non-free joints, as cmd.pos does.
    limits = self.entity.data.joint_pos_limits[:, self._target_ids]
    target = torch.clamp(cmd.position_target, limits[..., 0], limits[..., 1])
    return super().compute(replace(cmd, position_target=target))

  def set_no_load_speed(self, env_ids: torch.Tensor | slice, speed: torch.Tensor) -> None:
    """Set the no-load speed (rad/s) per env; ``speed`` is (len(env_ids),) or (len, J)."""
    assert self.velocity_limit_motor is not None
    if speed.ndim == 1:
      speed = speed.unsqueeze(-1)
    self.velocity_limit_motor[env_ids] = speed

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self.velocity_limit_motor is not None and self.saturation_effort is not None
    self._vel_at_effort_lim = self.velocity_limit_motor * (
      1 + self.force_limit / self.saturation_effort
    )
    return super()._clip_effort(effort)


def randomize_battery_voltage(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  voltage_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Draw a battery voltage per env and set every RS03's no-load speed from it."""
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  lo, hi = voltage_range
  volts = lo + (hi - lo) * torch.rand(len(env_ids), device=env.device)
  speed = RS03_NO_LOAD_SPEED_RATED * volts / RS03_RATED_VOLTAGE
  for actuator in asset.actuators:
    if isinstance(actuator, Rs03Actuator):
      actuator.set_no_load_speed(env_ids, speed)
