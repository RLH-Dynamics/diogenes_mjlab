"""Event / reset-pose functions for the Diogenes periodic-hopping task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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
      to ``(0.0, 0.0)`` -- start at rest.
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

  # Log the post-reset distance-to-nearest-limit (fraction of range).
  with torch.no_grad():
    finite_rng = torch.where(finite, rng, torch.ones_like(rng))
    dist_lo = (joint_pos - lower) / finite_rng
    dist_hi = (upper - joint_pos) / finite_rng
    nearest = torch.minimum(dist_lo, dist_hi)
    nearest = torch.where(finite, nearest, torch.ones_like(nearest))
    env.extras["log"]["Metrics/reset_joint_margin_min"] = nearest.min()
