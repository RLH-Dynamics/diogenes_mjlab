"""Left/right mirror symmetry of the Harold walking task, for rsl-rl.

Harold is symmetric about its sagittal (x-z) plane. Mirroring a state swaps the
legs and flips the signs that change under y -> -y; the policy should act on a
mirrored state with the mirrored action. rsl-rl's Symmetry extension uses
``mirror_obs_and_actions`` to add the mirrored copy of every sample to each PPO
mini-batch (data augmentation), which roughly halves what the policy has to
learn and gives an even gait.

Joints, in HAROLD_JOINT_NAMES order (left hip, thigh, calf, right hip, thigh,
calf): mirroring swaps sides and negates the hip and thigh angles; the calf
keeps its sign (checked with forward kinematics in the tests).

Observation terms are mirrored per timestep of their history. A term missing
from ``TERM_MIRRORS`` raises, so a new observation cannot slip in unmirrored.
"""

from __future__ import annotations

import torch

#: (permutation, signs) of one timestep of each observation term.
_JOINTS = ((3, 4, 5, 0, 1, 2), (-1, -1, 1, -1, -1, 1))
_FEET = ((1, 0), (1, 1))

TERM_MIRRORS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
  # BNO085 chip frame: +x left, +y up, +z forward. Gravity is a vector, the
  # angular velocity a pseudovector.
  "imu_ang_vel": ((0, 1, 2), (1, -1, -1)),
  "imu_gravity": ((0, 1, 2), (-1, 1, 1)),
  # Standard base frame: +x forward, +y left, +z up.
  "base_ang_vel": ((0, 1, 2), (-1, 1, -1)),
  "projected_gravity": ((0, 1, 2), (1, -1, 1)),
  "base_lin_vel": ((0, 1, 2), (1, -1, 1)),
  "joint_pos": _JOINTS,
  "joint_vel": _JOINTS,
  "actions": _JOINTS,
  # (vx, vy, yaw rate).
  "command": ((0, 1, 2), (1, -1, -1)),
  # (sin, cos) of the gait phase: the mirrored robot's left leg does what the
  # right one did, which runs half a cycle behind.
  "gait_clock": ((0, 1), (-1, -1)),
  "foot_height": _FEET,
  "foot_air_time": _FEET,
  "foot_contact": _FEET,
  # Per foot (x, y, z) in the heading frame.
  "foot_contact_forces": ((3, 4, 5, 0, 1, 2), (1, -1, 1, 1, -1, 1)),
}

ACTION_MIRROR = _JOINTS

_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def group_mirror(
  term_names: list[str], term_dims: list[tuple[int, ...]], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
  """(index, sign) with mirrored[:, i] = sign[i] * obs[:, index[i]] for a group."""
  index, sign, offset = [], [], 0
  for name, dims in zip(term_names, term_dims, strict=True):
    if name not in TERM_MIRRORS:
      raise KeyError(f"No mirror defined for observation term {name!r}.")
    perm, signs = TERM_MIRRORS[name]
    width = int(torch.tensor(dims).prod())
    if width % len(perm):
      raise ValueError(f"Term {name!r} width {width} is not a multiple of {len(perm)}.")
    for step in range(width // len(perm)):
      base = offset + step * len(perm)
      index.extend(base + p for p in perm)
      sign.extend(signs)
    offset += width
  return (
    torch.tensor(index, dtype=torch.long, device=device),
    torch.tensor(sign, dtype=torch.float, device=device),
  )


def _group_tensors(env, group: str, device) -> tuple[torch.Tensor, torch.Tensor]:
  key = (id(env), group)
  if key not in _cache:
    manager = env.unwrapped.observation_manager
    _cache[key] = group_mirror(
      manager.active_terms[group], manager.group_obs_term_dim[group], device
    )
  return _cache[key]


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
  perm, signs = ACTION_MIRROR
  sign = torch.tensor(signs, dtype=actions.dtype, device=actions.device)
  return actions[:, list(perm)] * sign


def mirror_obs_and_actions(env, obs=None, actions=None):
  """rsl-rl data_augmentation_func: originals first, then their mirror images."""
  obs_aug = None
  if obs is not None:
    mirrored = obs.clone()
    for group in obs.keys():
      index, sign = _group_tensors(env, group, obs[group].device)
      mirrored[group] = obs[group][:, index] * sign
    obs_aug = torch.cat([obs, mirrored], dim=0)
  actions_aug = None
  if actions is not None:
    actions_aug = torch.cat([actions, mirror_actions(actions)], dim=0)
  return obs_aug, actions_aug
