"""Diogenes environment configurations.

The Diogenes robot is a fixed-base, single 3-jointed leg (hip / thigh / calf)
mounted on an unactuated prismatic ``slider`` joint that constrains the body to
move vertically along a rail. This is a hop test stand, not a free-floating
locomotion robot, so the built-in ``velocity`` task does not apply and the env
is built from scratch.

The task here is *periodic hopping*: drive the three leg joints so that the
carriage (tracked by the slider) follows a phase-conditioned ballistic arc that
peaks ``HOP_HEIGHT`` metres above its starting position once per ``HOP_PERIOD``.

Slider sign convention (verified against MuJoCo): the leg_mount body is rotated
180 deg about X, so the slider axis points along world -Z and the carriage
height above start equals ``-slider_pos``. See ``mdp.py`` for details.

Run a trained policy with::

    uv run play Diogenes-Flat
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg

from . import mdp as diogenes_mdp
from .diogenes.diogenes_constants import get_diogenes_cfg

# Names of the XML-defined <position> actuators (== the actuated joint names).
DIOGENES_ACTUATOR_NAMES = ("hip", "thigh", "calf")

# Hop task parameters.
HOP_HEIGHT = 0.35  # Peak target height above start, in metres.
HOP_PERIOD = 0.6  # One hop cycle, in seconds.
HOP_FLIGHT_FRAC = 0.7  # Fraction of the period spent airborne.

# Entity-config factories. Each manager term must get its OWN SceneEntityCfg
# instance, passed via ``params`` so the manager resolves it. Sharing a single
# instance across terms is fragile (the manager mutates it in place during
# resolution), so we hand out a fresh one per term.


def slider_cfg() -> SceneEntityCfg:
  """Selects only the unactuated prismatic slider joint."""
  return SceneEntityCfg("robot", joint_names=("slider",))


def actuated_joints_cfg() -> SceneEntityCfg:
  """Selects the three actuated leg joints (excludes the slider)."""
  return SceneEntityCfg("robot", joint_names=DIOGENES_ACTUATOR_NAMES)


def actuators_cfg() -> SceneEntityCfg:
  """Selects the three position actuators (for the torque penalty)."""
  return SceneEntityCfg("robot", actuator_names=DIOGENES_ACTUATOR_NAMES)


def diogenes_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Diogenes periodic-hopping environment configuration.

  Args:
    play: When True, apply evaluation-friendly overrides (effectively infinite
      episode, no observation corruption).
  """
  # ---------------------------------------------------------------------------
  # Simulation: 2 ms physics step * decimation 10 -> 50 Hz control.
  # ---------------------------------------------------------------------------
  # njmax/nconmax set the per-world constraint and contact buffer sizes. With
  # foot-only collision (see diogenes.xml) the real counts are tiny, but we give
  # generous headroom so a transient contact spike never clamps ("nefc overflow"
  # warnings) and the physics stays deterministic. These live on SimulationCfg,
  # not MujocoCfg.
  sim_cfg = SimulationCfg(
    njmax=64,
    nconmax=16,
    mujoco=MujocoCfg(
      timestep=0.002,
    ),
  )

  # ---------------------------------------------------------------------------
  # Scene: the robot plus a ground plane (included by scene.xml).
  # ---------------------------------------------------------------------------
  scene_cfg = SceneCfg(
    num_envs=4096,
    env_spacing=2.0,
    entities={"robot": get_diogenes_cfg()},
  )

  # ---------------------------------------------------------------------------
  # Actions: position targets for the three actuated joints.
  # ---------------------------------------------------------------------------
  # use_default_offset=True means a zero action holds the default joint pose;
  # the policy learns deltas about it. The slider is unactuated and excluded.
  joint_pos_action = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=DIOGENES_ACTUATOR_NAMES,
    scale=1.0,
    use_default_offset=True,
  )

  # ---------------------------------------------------------------------------
  # Observations.
  # ---------------------------------------------------------------------------
  # The actuated-joint pos/vel (relative to default), the unactuated slider
  # pos/vel (absolute carriage travel), the last action, and the (sin, cos)
  # hop-phase clock so the policy knows where it is in the hop cycle.
  # NOTE: A SceneEntityCfg is only resolved (names -> joint_ids) if the manager
  # finds it inside a term's ``params``. A cfg passed only as a function default
  # is never resolved, so its ``joint_ids`` stays ``slice(None)`` and selects
  # ALL joints. Every term that needs a specific joint subset therefore passes
  # its ``asset_cfg`` explicitly via ``params`` (and uses a fresh instance so no
  # cfg object is shared/mutated across terms).
  proprio_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": actuated_joints_cfg()},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": actuated_joints_cfg()},
    ),
    "slider_pos": ObservationTermCfg(
      func=diogenes_mdp.slider_pos,
      params={"asset_cfg": slider_cfg()},
    ),
    "slider_vel": ObservationTermCfg(
      func=diogenes_mdp.slider_vel,
      params={"asset_cfg": slider_cfg()},
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action),
    "phase_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock,
      params={"hop_period": HOP_PERIOD},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=dict(proprio_terms),
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=dict(proprio_terms),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # ---------------------------------------------------------------------------
  # Rewards: hop-height tracking plus regularization.
  # ---------------------------------------------------------------------------
  # The task reward (positive) tracks the phase-conditioned parabolic arc.
  # The three regularization terms have negative weights to discourage jerky,
  # high-torque, limit-slamming behaviour. Reward values are scaled by the
  # control dt inside the manager, so weights are per-second rates.
  rewards = {
    "hop_height": RewardTermCfg(
      func=diogenes_mdp.hop_height_tracking,
      weight=1.0,
      params={
        "hop_height": HOP_HEIGHT,
        "hop_period": HOP_PERIOD,
        "flight_frac": HOP_FLIGHT_FRAC,
        "std": 0.05,
        "asset_cfg": slider_cfg(),
      },
    ),
    # Action rate penalty: smooth, non-jerky targets.
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),
    # Torque magnitude penalty: energy-efficient actuation.
    "torque": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1.0e-4,
      params={"asset_cfg": actuators_cfg()},
    ),
    # Joint-limit avoidance: stay off the soft position limits.
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": actuated_joints_cfg()},
    ),
  }

  # ---------------------------------------------------------------------------
  # Terminations: time-out only.
  # ---------------------------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=20.0,
    sim=sim_cfg,
    scene=scene_cfg,
    observations=observations,
    actions={"joint_pos": joint_pos_action},
    rewards=rewards,
    terminations=terminations,
  )

  # Point the viewer at the robot for a sensible default camera.
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  # ---------------------------------------------------------------------------
  # Play-mode overrides.
  # ---------------------------------------------------------------------------
  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
