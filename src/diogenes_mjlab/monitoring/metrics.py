"""Stateless scalar metric functions for the Diogenes hop stand.

Each returns ONE scalar per env (shape (num_envs,)) and is registered via
``cfg.metrics`` so mjlab's MetricsManager accumulates it and the Viser viewer
plots it live in the Metrics tab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from ..accessors import (
  carriage_acc,
  carriage_height,
  carriage_vel,
  slider_acc_scalar,
  slider_pos_scalar,
  slider_vel_scalar,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


##
# Small indexing helpers (shared by metrics + recorder).
##


def _resolve_joint_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Resolve and cache the long-tensor joint ids selected by ``asset_cfg``."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  if isinstance(ids, slice):
    n = asset.data.joint_pos.shape[1]
    ids = list(range(*ids.indices(n)))
  return torch.as_tensor(ids, device=env.device, dtype=torch.long)


def _joint_torque(env: ManagerBasedRlEnv, joint_ids: torch.Tensor) -> torch.Tensor:
  """Joint-space actuator torque for ``joint_ids``. Shape (num_envs, J), N*m."""
  asset: Entity = env.scene["robot"]
  return asset.data.qfrc_actuator[:, joint_ids]


def _joint_power(env: ManagerBasedRlEnv, joint_ids: torch.Tensor) -> torch.Tensor:
  """Per-joint mechanical power ``tau * qdot``. Shape (num_envs, J), watts."""
  asset: Entity = env.scene["robot"]
  tau = asset.data.qfrc_actuator[:, joint_ids]
  qd = asset.data.joint_vel[:, joint_ids]
  return tau * qd


##
# Scalar metric terms (shape (num_envs,)) -> live Viser plots.
##


def joint_torque_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  component: int,
) -> torch.Tensor:
  """Signed joint torque for a single joint (``component`` index into asset_cfg)."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  return _joint_torque(env, joint_ids)[:, component]


def joint_power_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  component: int,
) -> torch.Tensor:
  """Signed mechanical power ``tau*qdot`` for a single joint."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  return _joint_power(env, joint_ids)[:, component]


def total_mechanical_power_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Sum of per-joint mechanical power over the actuated joints. Watts."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  return torch.sum(_joint_power(env, joint_ids), dim=1)


def joint_pos_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  component: int,
) -> torch.Tensor:
  """Position of a single joint (rad)."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, joint_ids][:, component]


def joint_vel_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  component: int,
) -> torch.Tensor:
  """Velocity of a single joint (rad/s)."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, joint_ids][:, component]


def carriage_height_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Carriage height above start (m). ``= -slider_pos`` (see mdp.py)."""
  asset: Entity = env.scene[asset_cfg.name]
  return carriage_height(asset, asset_cfg)


def carriage_vel_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Carriage vertical velocity (m/s, +up). ``= -slider_vel``."""
  asset: Entity = env.scene[asset_cfg.name]
  return carriage_vel(asset, asset_cfg)


def carriage_acc_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Carriage vertical acceleration (m/s^2, +up). ``= -slider_acc``."""
  asset: Entity = env.scene[asset_cfg.name]
  return carriage_acc(asset, asset_cfg)


def slider_pos_metric(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Raw slider joint position (m, joint-space sign)."""
  asset: Entity = env.scene[asset_cfg.name]
  return slider_pos_scalar(asset, asset_cfg)


def slider_vel_metric(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Raw slider joint velocity (m/s, joint-space sign)."""
  asset: Entity = env.scene[asset_cfg.name]
  return slider_vel_scalar(asset, asset_cfg)


def slider_acc_metric(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Raw slider joint acceleration (m/s^2, joint-space sign)."""
  asset: Entity = env.scene[asset_cfg.name]
  return slider_acc_scalar(asset, asset_cfg)


def contact_force_component_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  component: int,
) -> torch.Tensor:
  """One world-frame component (0=x,1=y,2=z) of the net foot/ground force (N)."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # [B, N, 3], world frame (netforce).
  assert force is not None, (
    f"Sensor '{sensor_name}' must request the 'force' field for contact metrics."
  )
  return torch.sum(force[..., component], dim=-1)


def contact_force_magnitude_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Magnitude of the net foot/ground contact force (N)."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # [B, N, 3], world frame.
  assert force is not None, (
    f"Sensor '{sensor_name}' must request the 'force' field for contact metrics."
  )
  net = torch.sum(force, dim=1)  # [B, 3]
  return torch.linalg.norm(net, dim=-1)
