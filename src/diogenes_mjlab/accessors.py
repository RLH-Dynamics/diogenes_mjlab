"""Pure joint/slider/contact indexing helpers for the Diogenes hop stand.

All helpers are thin wrappers around single indexing expressions. They encode
the sign conventions and index semantics exactly once, preventing drift between
mdp.py and monitoring.py.

Sign convention reminder (from mdp.py docstring and geometry notes):
    carriage_height_above_start = -slider_pos

This is because the leg_mount body carries a 180 deg rotation about X, so the
slider's local +Z points along world -Z. A MORE NEGATIVE slider value raises the
carriage. The sign flip is applied here for the carriage helpers; the raw slider
helpers preserve the joint-space sign so callers can choose.

Note: a later Phase 2 may move this module into an mdp/ sub-package.
All helpers use ``[:, -1]`` to index the last (only) column returned by
``joint_ids``, which is robust whether ``joint_ids`` resolves to a list like
``[0]`` or a slice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  pass  # no env import needed here


# ---------------------------------------------------------------------------
# Slider scalar helpers (joint-space sign: negative = carriage up).
# ---------------------------------------------------------------------------


def slider_pos_scalar(asset: Entity, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Raw slider joint position, scalar per env. Shape (num_envs,).

  Joint-space sign: a MORE NEGATIVE value means the carriage is HIGHER.
  Equivalent to ``asset.data.joint_pos[:, asset_cfg.joint_ids][:, -1]``.
  """
  return asset.data.joint_pos[:, asset_cfg.joint_ids][:, -1]


def slider_vel_scalar(asset: Entity, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Raw slider joint velocity, scalar per env. Shape (num_envs,).

  Joint-space sign: positive velocity moves the carriage DOWN.
  Equivalent to ``asset.data.joint_vel[:, asset_cfg.joint_ids][:, -1]``.
  """
  return asset.data.joint_vel[:, asset_cfg.joint_ids][:, -1]


def slider_acc_scalar(asset: Entity, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Raw slider joint acceleration, scalar per env. Shape (num_envs,).

  Joint-space sign: positive acceleration moves the carriage DOWN.
  Equivalent to ``asset.data.joint_acc[:, asset_cfg.joint_ids][:, -1]``.
  """
  return asset.data.joint_acc[:, asset_cfg.joint_ids][:, -1]


# ---------------------------------------------------------------------------
# Carriage helpers (sign-corrected: +up = -slider, verified via MuJoCo FK).
# ---------------------------------------------------------------------------


def carriage_height(asset: Entity, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Carriage height above its start position. Shape (num_envs,).

  carriage_height = -slider_pos  (verified; see mdp.py geometry notes).
  """
  return -slider_pos_scalar(asset, asset_cfg)


def carriage_vel(asset: Entity, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Carriage vertical velocity, +up. Shape (num_envs,).

  carriage_vel = -slider_vel.
  """
  return -slider_vel_scalar(asset, asset_cfg)


def carriage_acc(asset: Entity, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Carriage vertical acceleration, +up. Shape (num_envs,).

  carriage_acc = -slider_acc.
  """
  return -slider_acc_scalar(asset, asset_cfg)
