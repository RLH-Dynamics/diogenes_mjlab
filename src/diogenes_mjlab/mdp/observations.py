"""Observation functions for the Diogenes periodic-hopping task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Default slider entity config.
_SLIDER_CFG = SceneEntityCfg("robot", joint_names=("slider",))


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
  """Carriage height above its start position. Shape (num_envs,)."""
  from ..accessors import carriage_height as _carriage_height
  asset: Entity = env.scene[asset_cfg.name]
  return _carriage_height(asset, asset_cfg)


##
# Slider (carriage) state observations.
##


def slider_pos(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _SLIDER_CFG
) -> torch.Tensor:
  """Raw slider joint position. Shape (num_envs, 1)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids]


def slider_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _SLIDER_CFG
) -> torch.Tensor:
  """Slider joint velocity. Shape (num_envs, 1)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]
