"""Reward-term builders for the Diogenes hop stand."""

from typing import Literal

from mjlab.envs import mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from ..constants import (
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

# Dual-parabola derived cycle period (seconds). Computed once so the phase clock,
# the trajectory reward, and any other phase-keyed term all share the SAME period.
TRAJ_T = diogenes_mdp.dual_parabola_timing(
  TRAJ_MIN, TRAJ_MAX, TRAJ_TRANSITION, GRAVITY
)[0]


def _slider_trajectory_reward(trajectory: TrajectoryType) -> tuple[RewardTermCfg, float]:
  """Build the slider-tracking reward term and its phase-clock period.

  Returns ``(reward_term, phase_period)``.
  """
  if trajectory == "dual_parabola":
    term = RewardTermCfg(
      func=diogenes_mdp.slider_dual_parabola_tracking,
      weight=300.0,
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
      weight=30.0,
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


def _contact_phase_reward(trajectory: TrajectoryType) -> RewardTermCfg:
  """Build the trajectory-appropriate foot/ground contact-phase penalty."""
  if trajectory == "sine":
    return RewardTermCfg(
      func=diogenes_mdp.foot_contact_required,
      weight=-2.0,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    )
  elif trajectory == "dual_parabola":
    return RewardTermCfg(
      func=diogenes_mdp.foot_contact_phase_dual_parabola,
      weight=-1.0,
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
) -> tuple[dict[str, RewardTermCfg], float]:
  """Build the full rewards dict and the phase-clock period.

  Returns ``(rewards_dict, phase_period)``.
  """
  slider_reward, phase_period = _slider_trajectory_reward(trajectory)
  contact_phase_reward = _contact_phase_reward(trajectory)

  rewards = {
    "slider_trajectory": slider_reward,
    "foot_xy_position": RewardTermCfg(
      func=diogenes_mdp.foot_xy_position_tracking,
      weight=100.0,
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
      weight=-0.002,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    ),
    "vertical_contact_force": RewardTermCfg(
      func=diogenes_mdp.vertical_contact_force_l2,
      weight=-0.0,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    ),
    "foot_slip": RewardTermCfg(
      func=diogenes_mdp.foot_slip,
      weight=-0.100,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "asset_cfg": SceneEntityCfg("robot", body_names=("calf_assy",)),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=velocity_mdp.soft_landing,
      weight=-0.0,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "command_name": None,
      },
    ),
    "electrical_power": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=-0.10,
      params={"asset_cfg": power_joints_cfg()},
    ),
    "torque": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-0.001,
      params={"asset_cfg": actuators_cfg()},
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),
    "action_acc": RewardTermCfg(
      func=mdp.action_acc_l2,
      weight=-0.001,
    ),
    "joint_acc": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": actuated_joints_cfg()},
    ),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": actuated_joints_cfg()},
    ),
    "termination_penalty": RewardTermCfg(
      func=mdp.is_terminated,
      weight=-500.0,
    ),
  }
  return rewards, phase_period
