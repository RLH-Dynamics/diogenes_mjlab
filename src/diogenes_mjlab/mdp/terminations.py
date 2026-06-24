"""Termination functions for the Diogenes periodic-hopping task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
