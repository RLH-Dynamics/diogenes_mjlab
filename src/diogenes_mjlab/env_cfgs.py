"""Diogenes environment configurations.

The Diogenes robot is a fixed-base, single 3-jointed leg (hip / thigh / calf)
mounted on an unactuated prismatic ``slider`` joint. The slider's parent block
is welded to the world frame, so this is NOT a floating-base locomotion robot
and the built-in ``velocity`` task (``make_velocity_env_cfg``) does not apply:
that task assumes a free-floating body, body-velocity commands, foot-contact
sensors and terrain, none of which exist here.

This module therefore builds a minimal ``ManagerBasedRlEnvCfg`` from scratch.
The immediate goal is a zero-action sanity check:

    uv run play Diogenes-Flat --agent zero

Under ``--agent zero`` the policy emits an all-zeros action of shape
``env.action_space``, which is determined entirely by the action manager. So
the only thing strictly required to run is: a scene (robot + ground plane), one
action term over the three actuators, at least one observation group, and a
``time_out`` termination so episodes reset cleanly. A single trivial reward is
included so the reward manager has a term and logging is sensible.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg

from .diogenes.diogenes_constants import get_diogenes_cfg

# Names of the XML-defined <position> actuators (== the actuated joint names).
DIOGENES_ACTUATOR_NAMES = ("hip", "thigh", "calf")


def diogenes_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Diogenes flat-ground environment configuration.

  Args:
    play: When True, apply evaluation-friendly overrides (effectively infinite
      episode, no observation corruption). Kept for parity with the mjlab task
      registration API, which expects a separate play config.
  """
  # ---------------------------------------------------------------------------
  # Simulation.
  # ---------------------------------------------------------------------------
  # 2 ms physics step * decimation 10 -> 50 Hz control. This is a small,
  # contact-light scene so the default contact/constraint heuristics are fine.
  sim_cfg = SimulationCfg(
    mujoco=MujocoCfg(
      timestep=0.002,
    ),
  )

  # ---------------------------------------------------------------------------
  # Scene: the robot plus a ground plane to land on.
  # ---------------------------------------------------------------------------
  # ``get_diogenes_cfg()`` returns a fresh EntityCfg each call (the spec is
  # loaded from xmls/diogenes.xml). A flat plane is added via the terrain field
  # so the calf has something to contact under gravity.
  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=2.0,
    entities={"robot": get_diogenes_cfg()},
  )

  # ---------------------------------------------------------------------------
  # Actions: position targets for the three actuated joints.
  # ---------------------------------------------------------------------------
  # use_default_offset=True means a zero action holds the default joint pose
  # (from EntityCfg.InitialStateCfg), which is exactly the behaviour we want for
  # a zero-action test: the leg should hold its initial pose, then settle/fall
  # under gravity along the slider. The slider itself is unactuated and is
  # deliberately NOT listed here.
  joint_pos_action = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=DIOGENES_ACTUATOR_NAMES,
    scale=1.0,
    use_default_offset=True,
  )

  # ---------------------------------------------------------------------------
  # Observations: minimal proprioception.
  # ---------------------------------------------------------------------------
  # rsl_rl builds the actor/critic from the obs groups named in the runner's
  # ``obs_groups`` mapping, which defaults to {"actor": ("actor",),
  # "critic": ("critic",)}. The group names here MUST match those, so we use
  # "actor" and "critic". None of this is exercised by zero actions, but the
  # managers must have at least one well-formed group.
  proprio_terms = {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "last_action": ObservationTermCfg(func=mdp.last_action),
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
  # Rewards: a single trivial term so the reward manager is populated.
  # ---------------------------------------------------------------------------
  rewards = {
    "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
  }

  # ---------------------------------------------------------------------------
  # Terminations: time-out only, so episodes reset on a fixed horizon.
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
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
