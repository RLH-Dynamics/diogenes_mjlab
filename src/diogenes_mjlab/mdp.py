"""Custom MDP terms for the Diogenes periodic-hopping task.

This module adds the few task-specific functions that mjlab does not ship out
of the box. Everything else (action-rate, torque, joint-limit penalties, joint
pos/vel observations, last-action observation, time-out termination) is reused
directly from ``mjlab.envs.mdp``.

Geometry note (important for signs)
-----------------------------------
The leg hangs from an unactuated prismatic ``slider`` joint. In the URDF the
``leg_mount_assy`` body carries a ``quat="0 1 0 0"`` (a 180 deg rotation about
X), so the joint's local +Z axis points along world **-Z**. A *more negative*
``slider`` value therefore raises the carriage. With the default slider value
of 0.0, the height of the carriage above its starting position is simply::

    height_above_start = -slider_pos

This was verified directly against MuJoCo (``mj_forward`` sweep of the slider).

The phase clock
---------------
A single global hop phase ``phi in [0, 1)`` advances with simulation time and
wraps every ``hop_period`` seconds. It is derived from the per-env step counter
``env.episode_length_buf`` (reset to 0 on episode reset, incremented once per
control step) times ``env.step_dt``. Both the reward and the observation read
the same counter on the same step, so they stay perfectly in phase.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Entity config selecting just the unactuated prismatic slider joint.
SLIDER_CFG = SceneEntityCfg("robot", joint_names=("slider",))


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


##
# Slider (carriage) state.
##


def slider_pos(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SLIDER_CFG
) -> torch.Tensor:
  """Raw slider joint position. Shape (num_envs, 1).

  Not made relative to the default (which is 0.0 anyway) so the policy sees the
  carriage's absolute travel along the rail.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids]


def slider_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SLIDER_CFG
) -> torch.Tensor:
  """Slider joint velocity. Shape (num_envs, 1)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


##
# Hop reward.
##


def _target_height(
  phi: torch.Tensor, hop_height: float, flight_frac: float
) -> torch.Tensor:
  """Phase-conditioned target carriage height (m above start).

  A hop is modelled as a ballistic flight arc followed by a stance interval:

    * During the flight window ``phi in [0, flight_frac]`` the target traces a
      downward-opening parabola - exactly the shape of a projectile launched
      and landing under constant gravity - peaking at ``hop_height`` halfway
      through the flight.
    * During the remaining stance window the target is 0 (the foot is on the
      ground), so the leg is asked to settle back to its start height before
      the next hop.

  Normalising the flight window to ``x in [0, 1]`` the parabola is
  ``hop_height * 4 * x * (1 - x)``, which is 0 at the ends and ``hop_height``
  at ``x = 0.5``.
  """
  x = phi / flight_frac
  parabola = hop_height * 4.0 * x * (1.0 - x)
  in_flight = phi <= flight_frac
  return torch.where(in_flight, parabola, torch.zeros_like(phi))


def hop_height_tracking(
  env: ManagerBasedRlEnv,
  hop_height: float = 0.2,
  hop_period: float = 0.6,
  flight_frac: float = 0.7,
  std: float = 0.05,
  asset_cfg: SceneEntityCfg = SLIDER_CFG,
) -> torch.Tensor:
  """Reward for tracking the phase-conditioned hop height. Shape (num_envs,).

  ``height_above_start = -slider_pos`` (see module docstring). The reward is a
  Gaussian on the height error, equal to 1 when the carriage is exactly on the
  target arc and decaying smoothly with a length scale of ``std`` metres.

  Args:
    hop_height: Peak target height of the hop, in metres above start.
    hop_period: Duration of one hop cycle, in seconds.
    flight_frac: Fraction of the period spent in the (airborne) flight arc.
    std: Gaussian width on the height error, in metres.
    asset_cfg: Entity config selecting the slider joint.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Select the (single) slider column and collapse to shape (num_envs,). Using
  # index -1 of the selected columns is robust whether joint_ids resolves to a
  # list like [0] or a slice; the slider is the only joint this cfg selects.
  slider = asset.data.joint_pos[:, asset_cfg.joint_ids]
  height = -slider[:, -1]
  phi = _phase(env, hop_period)
  target = _target_height(phi, hop_height, flight_frac)
  error = height - target
  return torch.exp(-torch.square(error) / (std**2))
