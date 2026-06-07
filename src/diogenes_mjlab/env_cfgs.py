"""Diogenes environment configurations.

The Diogenes robot is a fixed-base, single 3-jointed leg (hip / thigh / calf)
mounted on an unactuated prismatic ``slider`` joint that constrains the body to
move vertically along a rail. This is a hop test stand, not a free-floating
locomotion robot, so the built-in ``velocity`` task does not apply and the env
is built from scratch.

The task here is *periodic hopping*: drive the three leg joints so the carriage
(tracked by the slider) follows a gravity-exact dual-parabolic vertical
trajectory, while the point-foot stays fixed in world (x, y).

Trajectory (gravity-exact, derived period)
------------------------------------------
The carriage reference is two parabolic arcs that meet at ``TRAJ_TRANSITION``:

  * FLIGHT arc: a TRUE free-fall parabola at -GRAVITY, rising from
    ``TRAJ_TRANSITION`` to ``TRAJ_MAX`` and back. Its duration is fixed by
    physics (T_flight = 2*sqrt(2*Hf/g), Hf = TRAJ_MAX - TRAJ_TRANSITION).
  * RECOVERY arc: a constant-acceleration parabola from ``TRAJ_TRANSITION`` down
    to ``TRAJ_MIN`` and back. The acceleration is SOLVED for velocity continuity
    at the transition (a = g * Hf / Hr, Hr = TRAJ_TRANSITION - TRAJ_MIN), and its
    duration (T_recovery = 2*v0/a) then follows from the dynamics.

There is therefore NO free period: the cycle period is DERIVED as
T_total = T_flight + T_recovery and shared with the phase clock. See
``mdp.dual_parabola_timing``.

Reward design
-------------
  * ``slider_dual_parabola``    : dense Gaussian tracking the gravity-exact
                                   carriage reference (owns amplitude AND timing).
  * ``foot_xy_position``        : dense Gaussian keeping the point-foot's world
                                   (x, y) fixed, so the leg hops straight up/down.
  * Naturalness/smoothness + actuation regularizers: unchanged.

Foot/ground contact is measured by a ``ContactSensor`` (primary ``calf_assy``,
secondary ``floor``, ``reduce="netforce"`` -> global-frame wrench). Slider sign
convention (verified against MuJoCo): the leg_mount body is rotated 180 deg about
X, so the slider axis points along world -Z and the carriage height above start
equals ``-slider_pos``. See ``mdp.py`` for details.

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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from . import mdp as diogenes_mdp
from .diogenes.diogenes_constants import get_diogenes_cfg

# Names of the XML-defined <position> actuators (== the actuated joint names).
DIOGENES_ACTUATOR_NAMES = ("hip", "thigh", "calf")

# ---------------------------------------------------------------------------
# Gravity-exact dual-parabolic slider trajectory geometry. All three heights are
# z-values RELATIVE TO THE SLIDER ORIGIN (== the carriage start position; height
# above start = -slider_pos, verified). Require TRAJ_MAX >= TRAJ_TRANSITION >
# TRAJ_MIN.
#   * TRAJ_MAX        : apex of the free-fall (flight) arc.
#   * TRAJ_MIN        : lowest point of the constant-accel (recovery) arc.
#   * TRAJ_TRANSITION : height where the two arcs meet (also the cycle boundary).
#   * GRAVITY         : free-fall acceleration for the flight arc (m/s^2).
# The cycle PERIOD is NOT specified here -- it is derived from these via physics
# (see TRAJ_T below / mdp.dual_parabola_timing).
# ---------------------------------------------------------------------------
TRAJ_MAX = 0.35
TRAJ_MIN = 0.05
TRAJ_TRANSITION = 0.15
GRAVITY = diogenes_mdp.GRAVITY  # 9.81 m/s^2

# Derived cycle period (seconds). Computed once so the phase clock, the
# trajectory reward, and any other phase-keyed term all share the SAME period.
# This is the single source of truth for timing.
TRAJ_T = diogenes_mdp.dual_parabola_timing(
  TRAJ_MIN, TRAJ_MAX, TRAJ_TRANSITION, GRAVITY
)[0]

# Name of the foot/ground contact sensor (referenced by several reward terms).
FOOT_CONTACT_SENSOR = "foot_ground_contact"

# Entity-config factories. Each manager term must get its OWN SceneEntityCfg
# instance, passed via ``params`` so the manager resolves it.


def slider_cfg() -> SceneEntityCfg:
  """Selects only the unactuated prismatic slider joint."""
  return SceneEntityCfg("robot", joint_names=("slider",))


def actuated_joints_cfg() -> SceneEntityCfg:
  """Selects the three actuated leg joints (excludes the slider)."""
  return SceneEntityCfg("robot", joint_names=DIOGENES_ACTUATOR_NAMES)


def actuators_cfg() -> SceneEntityCfg:
  """Selects the three position actuators (for the torque penalty)."""
  return SceneEntityCfg("robot", actuator_names=DIOGENES_ACTUATOR_NAMES)


def power_joints_cfg() -> SceneEntityCfg:
  """Selects the three actuated joints by NAME for the power penalty.

  ``electrical_power_cost`` resolves its ``asset_cfg`` via ``find_joints`` on the
  joint names, so this must carry ``joint_names`` (not ``actuator_names``).
  """
  return SceneEntityCfg("robot", joint_names=DIOGENES_ACTUATOR_NAMES)


def calf_body_cfg() -> SceneEntityCfg:
  """Selects the calf_assy body (the foot body) for foot-position tracking."""
  return SceneEntityCfg("robot", body_names=("calf_assy",))


def diogenes_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Diogenes periodic-hopping environment configuration.

  Args:
    play: When True, apply evaluation-friendly overrides (effectively infinite
      episode, no observation corruption).
  """
  # ---------------------------------------------------------------------------
  # Simulation: 2 ms physics step * decimation 10 -> 50 Hz control.
  # ---------------------------------------------------------------------------
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
  foot_contact_cfg = ContactSensorCfg(
    name=FOOT_CONTACT_SENSOR,
    primary=ContactMatch(mode="body", pattern="calf_assy", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="floor", entity="robot"),
    fields=("found", "force", "pos"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  scene_cfg = SceneCfg(
    num_envs=4096,
    env_spacing=2.0,
    entities={"robot": get_diogenes_cfg()},
    sensors=(foot_contact_cfg,),
  )

  # ---------------------------------------------------------------------------
  # Actions: position targets for the three actuated joints.
  # ---------------------------------------------------------------------------
  joint_pos_action = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=DIOGENES_ACTUATOR_NAMES,
    scale=1.0,
    use_default_offset=True,
  )

  # ---------------------------------------------------------------------------
  # Observations.
  # ---------------------------------------------------------------------------
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
    # Phase clock uses the DERIVED period so it stays in lockstep with the
    # gravity-exact trajectory reward.
    "phase_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock,
      params={"hop_period": TRAJ_T},
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
  # Rewards.
  # ---------------------------------------------------------------------------
  rewards = {
    # --- Gravity-exact dual-parabolic slider trajectory tracking (dense). ---
    # Owns the whole carriage trajectory: free-fall arc up to TRAJ_MAX and back
    # to TRAJ_TRANSITION at -GRAVITY, then a velocity-continuous constant-accel
    # arc down to TRAJ_MIN and back. Period is derived (no traj_T argument).
    "slider_dual_parabola": RewardTermCfg(
      func=diogenes_mdp.slider_dual_parabola_tracking,
      weight=30.0,
      params={
        "traj_min": TRAJ_MIN,
        "traj_max": TRAJ_MAX,
        "traj_transition": TRAJ_TRANSITION,
        "std": 0.1,
        "gravity": GRAVITY,
        "asset_cfg": slider_cfg(),
      },
    ),
    # --- Point-foot world (x, y) hold (dense). ---
    "foot_xy_position": RewardTermCfg(
      func=diogenes_mdp.foot_xy_position_tracking,
      weight=10.0,
      params={
        "asset_cfg": calf_body_cfg(),
        "ref_xy": diogenes_mdp.DEFAULT_FOOT_REF_XY,
        "std": 0.05,
        "foot_offset_b": diogenes_mdp.FOOT_OFFSET_B,
      },
    ),
    # --- Contact-force shaping (naturalness / smoothness). ---
    "lateral_contact_force": RewardTermCfg(
      func=diogenes_mdp.lateral_contact_force_l2,
      weight=-0.0,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    ),
    "vertical_contact_force": RewardTermCfg(
      func=diogenes_mdp.vertical_contact_force_l2,
      weight=-0.0,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    ),
    "foot_slip": RewardTermCfg(
      func=diogenes_mdp.foot_slip,
      weight=-0.0,
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
    # --- Actuation regularizers. ---
    "electrical_power": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=-0.01,
      params={"asset_cfg": power_joints_cfg()},
    ),
    "torque": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-0.01,
      params={"asset_cfg": actuators_cfg()},
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": actuated_joints_cfg()},
    ),
    # --- One-time penalty on any failure termination. ---
    "termination_penalty": RewardTermCfg(
      func=mdp.is_terminated,
      weight=-500.0,
    ),
  }

  # ---------------------------------------------------------------------------
  # Terminations.
  # ---------------------------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "joint_at_limit": TerminationTermCfg(
      func=diogenes_mdp.joint_at_limit,
      params={
        "asset_cfg": actuated_joints_cfg(),
        "margin": 0.02,
      },
    ),
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
