"""Diogenes environment configurations.

The Diogenes robot is a fixed-base, single 3-jointed leg (hip / thigh / calf)
mounted on an unactuated prismatic ``slider`` joint that constrains the body to
move vertically along a rail. This is a hop test stand, not a free-floating
locomotion robot, so the built-in ``velocity`` task does not apply and the env
is built from scratch.

The task here is *periodic hopping*: drive the three leg joints so the carriage
(tracked by the slider) hops once per ``HOP_PERIOD``, reaching ``HOP_HEIGHT``
metres above start at the apex.

Reward design (updated)
-----------------------
The original dense reward tracked the slider's deviation from a hand-authored
parabolic trajectory. That has been replaced with a single outcome-based apex
reward, plus naturalness/smoothness regularizers:

  * Hop amplitude
      - ``peak_hop_height``          : per cycle, reward how close the achieved
                                       apex gets to the desired height (Gaussian).
                                       No flight/stance schedule is enforced: the
                                       policy discovers the liftoff timing that
                                       the leg geometry and target apex imply.
  * Naturalness / smoothness
      - ``lateral_contact_force``    : minimize tangential ground reaction
                                       (kills foot slip / scuffing).
      - ``vertical_contact_force``   : gently minimize normal ground reaction
                                       (encourages soft, compliant footfalls).
      - ``foot_slip``                : penalize foot xy velocity while in contact
                                       (reuses the velocity task's term).
      - ``soft_landing``             : penalize landing impact force
                                       (reuses the velocity task's term).
      - ``electrical_power``         : penalize positive mechanical power
                                       (reuses mjlab's term; energy efficiency).
      - ``action_rate`` / ``joint_limits`` : unchanged regularizers.

Foot/ground contact is measured by a ``ContactSensor`` whose primary is the
``calf_assy`` body (the foot is the only collidable geom on it) and whose
secondary is the world ``floor`` geom. ``reduce="netforce"`` returns the summed
contact wrench in the GLOBAL frame, so its z-component is the vertical force and
its xy-components are the lateral forces. ``track_air_time=True`` provides the
air/contact-time accumulators the gait terms and ``soft_landing`` rely on.

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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from . import mdp as diogenes_mdp
from .diogenes.diogenes_constants import get_diogenes_cfg

# Names of the XML-defined <position> actuators (== the actuated joint names).
DIOGENES_ACTUATOR_NAMES = ("hip", "thigh", "calf")

# Hop task parameters.
HOP_HEIGHT = 0.35  # Peak target height above start, in metres.
HOP_PERIOD = 0.9  # One hop cycle, in seconds. A clean 0.35 m ballistic hop has
# ~0.53 s of flight; at 0.9 s the flight sits centered in the cycle (takeoff at
# phase ~0.20, apex at 0.50, landing at ~0.80) leaving ~41% stance. At this
# period the height target and the mid-cycle apex-timing target are mutually
# consistent: a centered ballistic arc peaking at mid-cycle peaks at exactly
# 0.35 m, so the two reward factors reinforce rather than fight.

# Name of the foot/ground contact sensor (referenced by several reward terms).
FOOT_CONTACT_SENSOR = "foot_ground_contact"

# Name of the thigh/ground contact sensor (drives the flat-landing termination).
THIGH_CONTACT_SENSOR = "thigh_ground_contact"

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


def power_joints_cfg() -> SceneEntityCfg:
  """Selects the three actuated joints by NAME for the power penalty.

  ``electrical_power_cost`` resolves its ``asset_cfg`` via ``find_joints`` on the
  joint names, so this must carry ``joint_names`` (not ``actuator_names``).
  """
  return SceneEntityCfg("robot", joint_names=DIOGENES_ACTUATOR_NAMES)


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
  # Foot/ground contact sensor:
  #   * primary  = the calf_assy body (the foot is its only collidable geom).
  #   * secondary = the "floor" geom. The floor lives inside the robot entity's
  #     spec (scene.xml is loaded as part of the entity), so it gets attached
  #     with the "robot/" prefix; setting entity="robot" lets the sensor resolve
  #     and prefix the name automatically (the bare literal "floor" would not be
  #     found in the compiled model).
  #   * reduce="netforce" -> single summed wrench per env in the GLOBAL frame,
  #     so force[..., 2] is vertical and force[..., :2] is lateral.
  #   * track_air_time=True -> enables air/contact-time accumulators used by the
  #     gait terms and soft_landing's first-contact detection.
  foot_contact_cfg = ContactSensorCfg(
    name=FOOT_CONTACT_SENSOR,
    primary=ContactMatch(mode="body", pattern="calf_assy", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="floor", entity="robot"),
    fields=("found", "force", "pos"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # Thigh/ground contact sensor: detects a flat, lengthwise landing where a
  # thigh shell strikes the floor (the exploit that lets the policy land along
  # the limb axis and induce ~zero knee torque). Only "found" is needed -- the
  # termination just asks whether any thigh<->floor contact exists. The thigh
  # shells are collidable only against the floor (see diogenes.xml contype/
  # conaffinity), so this sensor never sees self/limb contacts.
  thigh_contact_cfg = ContactSensorCfg(
    name=THIGH_CONTACT_SENSOR,
    primary=ContactMatch(mode="body", pattern="thigh_assy", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="floor", entity="robot"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
    track_air_time=False,
  )

  scene_cfg = SceneCfg(
    num_envs=4096,
    env_spacing=2.0,
    entities={"robot": get_diogenes_cfg()},
    sensors=(foot_contact_cfg, thigh_contact_cfg),
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
  # Rewards.
  # ---------------------------------------------------------------------------
  # Reward values are scaled by the control dt inside the manager, so weights
  # are per-second rates. Positive weights are rewards; negative are penalties.
  rewards = {
    # --- Hop amplitude + timing: hit the desired apex AT mid-cycle. ---
    # The main task driver. Sparse (fires only on the phase-wrap step) and
    # positive (product of two Gaussians, in [0, 1]). It rewards BOTH the peak
    # carriage height matching HOP_HEIGHT and the apex occurring at phase 0.5
    # (mid-cycle). The timing factor is what couples the realized hop period to
    # HOP_PERIOD: without it the policy launches from a crouch for a short, early
    # apex and hops faster than the clock. It still does not prescribe the
    # trajectory -- only when the apex must land.
    "peak_hop_height": RewardTermCfg(
      func=diogenes_mdp.peak_hop_height_reward,
      weight=60.0,
      params={
        "hop_height": HOP_HEIGHT,
        "hop_period": HOP_PERIOD,
        # std=0.15 (not 0.05): a tight std gives zero reward gradient once the
        # apex drifts above ~0.45 m, stranding an overshooting policy on a flat
        # plateau with no pull back to target. 0.15 keeps a usable gradient
        # across 0.35-0.57 m while still rewarding accuracy near the target.
        "std": 0.15,
        "phase_std": 0.12,
        "asset_cfg": slider_cfg(),
      },
    ),
    # --- Contact-force shaping (naturalness / smoothness). ---
    # Lateral GRF: strongly discourage shearing the floor (anti-slip).
    "lateral_contact_force": RewardTermCfg(
      func=diogenes_mdp.lateral_contact_force_l2,
      weight=-0.0,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    ),
    # Vertical GRF: gently bias toward softer pushes. Keep SMALL: the foot must
    # push on the floor to hop, so a large weight here fights the task.
    "vertical_contact_force": RewardTermCfg(
      func=diogenes_mdp.vertical_contact_force_l2,
      weight=-0.0,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    ),
    # Foot slip: xy velocity AT THE TRUE CONTACT POINT while in contact.
    # Reconstructed from the calf body twist evaluated at the sensor's reported
    # contact position, because the foot is offset ~0.2 m from the calf body
    # origin and the origin velocity is dominated by spurious omega x r from
    # ordinary stance pivoting. See foot_slip in mdp.py. Needs the sensor's
    # "pos" field (added to the contact sensor below).
    "foot_slip": RewardTermCfg(
      func=diogenes_mdp.foot_slip,
      weight=-0.0,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "asset_cfg": SceneEntityCfg("robot", body_names=("calf_assy",)),
      },
    ),
    # Soft landing: penalize impact force at first contact each cycle.
    "soft_landing": RewardTermCfg(
      func=velocity_mdp.soft_landing,
      weight=-0.0,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "command_name": None,
      },
    ),
    # --- Actuation regularizers. ---
    # Electrical/mechanical power: penalize positive actuator power (energy
    # efficiency, smoother motion). Reuses mjlab's term.
    "electrical_power": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=-0.01,
      params={"asset_cfg": power_joints_cfg()},
    ),
    # Torque magnitude penalty: energy-efficient actuation.
    "torque": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-0.01,
      params={"asset_cfg": actuators_cfg()},
    ),
    # Action rate penalty: smooth, non-jerky targets.
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),
    # Joint-limit avoidance: stay off the soft position limits.
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": actuated_joints_cfg()},
    ),
    # --- One-time penalty on any failure termination. ---
    # mjlab's standard term: fires (=1.0) on any non-timeout termination
    # (joint_at_limit or thigh_ground_contact here; the time_out truncation is
    # excluded). The env computes terminations before rewards, so the penalty
    # lands on the same step the episode ends, before reset. Rewards are scaled
    # by step_dt (0.02 s at 50 Hz control), so the realized one-step penalty is
    # weight * 0.02; weight=-500 -> a -10 hit per failure.
    "termination_penalty": RewardTermCfg(
      func=mdp.is_terminated,
      weight=-500.0,
    ),
  }

  # ---------------------------------------------------------------------------
  # Terminations.
  # ---------------------------------------------------------------------------
  # time_out: a non-failure episode end (the value function bootstraps from the
  # final state). joint_at_limit: a genuine FAILURE end (time_out defaults to
  # False -> no bootstrap), so the policy learns that slamming a joint stop ends
  # the episode with no future reward. This removes the "bottom out a joint on
  # landing to dump impact energy into the constraint" exploit at the source:
  # an episode that hits a limit terminates immediately and collects nothing
  # more, which a limit-proximity penalty alone would only tax, not forbid.
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "joint_at_limit": TerminationTermCfg(
      func=diogenes_mdp.joint_at_limit,
      params={
        "asset_cfg": actuated_joints_cfg(),
        "margin": 0.02,
      },
    ),
    # Flat-landing exploit guard: terminate if a thigh shell touches the floor.
    # A genuine failure end (time_out=False) so the lengthwise landing collects
    # no further reward.
    "thigh_ground_contact": TerminationTermCfg(
      func=diogenes_mdp.thigh_ground_contact,
      params={"sensor_name": THIGH_CONTACT_SENSOR},
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
