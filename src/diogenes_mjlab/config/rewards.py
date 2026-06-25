"""Reward-term builders for the Diogenes hop stand."""

from dataclasses import dataclass
from typing import Literal

from mjlab.envs import mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..constants import (
  CONTACT_PHASE_TERM_NAME,
  FOOT_CONTACT_SENSOR,
  GRAVITY,
  SINE_PERIOD,
  TRAJ_MAX,
  TRAJ_MIN,
  TRAJ_TRANSITION,
)
from .. import mdp as diogenes_mdp
from .entities import (
  actuated_joints_cfg,
  actuators_cfg,
  calf_body_cfg,
  power_joints_cfg,
  slider_cfg,
)

TrajectoryType = Literal["dual_parabola", "sine"]


@dataclass(frozen=True)
class RewardWeights:
  """Tuneable per-term reward weights for the Diogenes hop stand.

  All fields map directly to the ``weight`` argument of the corresponding
  :class:`RewardTermCfg`.  Positive means reward; negative means penalty.

  The default values exactly reproduce the hand-tuned baseline.
  """

  # Trajectory-tracking
  slider_trajectory_dual_parabola: float = 1.0
  slider_trajectory_sine: float = 30.0

  # Foot position hold
  foot_xy_position: float = 1.0

  # Contact-phase penalties (trajectory-dependent)
  contact_phase_dual_parabola: float = 0.0
  contact_phase_sine: float = -2.0

  # Contact-force shaping
  lateral_contact_force: float = -0.0

  # Foot slip
  foot_slip: float = -0.0

  # Energy / torque / smoothness penalties
  electrical_power: float = -0.0
  torque: float = -0.0
  action_rate: float = -0.0
  action_acc: float = -0.0
  joint_acc: float = -0.0

  # Safety
  joint_limits: float = -0.0
  termination_penalty: float = -100.0
  # Dedicated penalty for the contact-phase termination (stacks on top of the
  # blanket termination_penalty above so a wrong-contact termination is
  # penalized more heavily than e.g. a joint-limit termination).
  contact_phase_termination: float = -200.0


# Default weights instance used by _build_rewards when no override is passed.
DEFAULT_REWARD_WEIGHTS = RewardWeights()

# Dual-parabola derived cycle period (seconds). Computed once so the phase clock,
# the trajectory reward, and any other phase-keyed term all share the SAME period.
TRAJ_T = diogenes_mdp.dual_parabola_timing(
  TRAJ_MIN, TRAJ_MAX, TRAJ_TRANSITION, GRAVITY
)[0]


def _slider_trajectory_reward(
  trajectory: TrajectoryType,
  weights: RewardWeights,
) -> tuple[RewardTermCfg, float]:
  """Build the slider-tracking reward term and its phase-clock period.

  Returns ``(reward_term, phase_period)``.
  """
  if trajectory == "dual_parabola":
    term = RewardTermCfg(
      func=diogenes_mdp.slider_dual_parabola_tracking,
      weight=weights.slider_trajectory_dual_parabola,
      params={
        "traj_min": TRAJ_MIN,
        "traj_max": TRAJ_MAX,
        "traj_transition": TRAJ_TRANSITION,
        "std": 0.1,
        "gravity": GRAVITY,
        "asset_cfg": slider_cfg(),
      },
    )
    return term, TRAJ_T
  elif trajectory == "sine":
    term = RewardTermCfg(
      func=diogenes_mdp.slider_sinusoid_tracking,
      weight=weights.slider_trajectory_sine,
      params={
        "traj_min": TRAJ_MIN,
        "traj_max": TRAJ_MAX,
        "sine_period": SINE_PERIOD,
        "std": 0.1,
        "asset_cfg": slider_cfg(),
      },
    )
    return term, SINE_PERIOD
  else:
    raise ValueError(
      f"Unknown trajectory {trajectory!r}; expected 'dual_parabola' or 'sine'."
    )


def _contact_phase_reward(
  trajectory: TrajectoryType,
  weights: RewardWeights,
) -> RewardTermCfg:
  """Build the trajectory-appropriate foot/ground contact-phase penalty."""
  if trajectory == "sine":
    return RewardTermCfg(
      func=diogenes_mdp.foot_contact_required,
      weight=weights.contact_phase_sine,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    )
  elif trajectory == "dual_parabola":
    return RewardTermCfg(
      func=diogenes_mdp.foot_contact_phase_dual_parabola,
      weight=weights.contact_phase_dual_parabola,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "traj_min": TRAJ_MIN,
        "traj_max": TRAJ_MAX,
        "traj_transition": TRAJ_TRANSITION,
        "gravity": GRAVITY,
      },
    )
  else:
    raise ValueError(
      f"Unknown trajectory {trajectory!r}; expected 'dual_parabola' or 'sine'."
    )


def _build_rewards(
  trajectory: TrajectoryType,
  weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> tuple[dict[str, RewardTermCfg], float]:
  """Build the full rewards dict and the phase-clock period.

  Args:
    trajectory: Which carriage trajectory to build reward terms for.
    weights: Per-term weight overrides.  Defaults to :data:`DEFAULT_REWARD_WEIGHTS`
      which reproduces the baseline training values exactly.

  Returns ``(rewards_dict, phase_period)``.
  """
  slider_reward, phase_period = _slider_trajectory_reward(trajectory, weights)
  contact_phase_reward = _contact_phase_reward(trajectory, weights)

  rewards = {
    "slider_trajectory": slider_reward,
    "foot_xy_position": RewardTermCfg(
      func=diogenes_mdp.foot_xy_position_tracking,
      weight=weights.foot_xy_position,
      params={
        "asset_cfg": calf_body_cfg(),
        "ref_xy": diogenes_mdp.DEFAULT_FOOT_REF_XY,
        "std": 0.05,
        "foot_offset_b": diogenes_mdp.FOOT_OFFSET_B,
      },
    ),
    "contact_phase": contact_phase_reward,
    "lateral_contact_force": RewardTermCfg(
      func=diogenes_mdp.lateral_contact_force_l2,
      weight=weights.lateral_contact_force,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    ),
    "foot_slip": RewardTermCfg(
      func=diogenes_mdp.foot_slip,
      weight=weights.foot_slip,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "asset_cfg": SceneEntityCfg("robot", body_names=("calf_assy",)),
      },
    ),
    "electrical_power": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=weights.electrical_power,
      params={"asset_cfg": power_joints_cfg()},
    ),
    "torque": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=weights.torque,
      params={"asset_cfg": actuators_cfg()},
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=weights.action_rate,
    ),
    "action_acc": RewardTermCfg(
      func=mdp.action_acc_l2,
      weight=weights.action_acc,
    ),
    "joint_acc": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=weights.joint_acc,
      params={"asset_cfg": actuated_joints_cfg()},
    ),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=weights.joint_limits,
      params={"asset_cfg": actuated_joints_cfg()},
    ),
    "termination_penalty": RewardTermCfg(
      func=mdp.is_terminated,
      weight=weights.termination_penalty,
    ),
    "contact_phase_termination": RewardTermCfg(
      func=diogenes_mdp.is_specific_termination,
      weight=weights.contact_phase_termination,
      params={"term_name": CONTACT_PHASE_TERM_NAME},
    ),
  }
  return rewards, phase_period
