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

Monitoring / logging
--------------------
Two cooperating instrumentation layers are wired in from ``monitoring.py``:

  * ``cfg.metrics`` exposes every requested channel (foot contact forces, joint
    torques, joint mechanical powers, slider pos/vel/acc, joint pos/vel) as
    per-step scalar metric terms. mjlab's Viser viewer auto-plots these LIVE in
    its **Metrics** tab -- toggle any subset on/off and watch them in real time.
  * ``cfg.recorders`` adds a ``DiogenesCsvRecorder`` that writes one CSV row per
    control step (full raw channels) under ``logs/diogenes_monitor/`` for offline
    analysis in Excel and cross-run comparison.

Both are controlled by flags on ``diogenes_env_cfg`` (``monitor``, ``record_csv``)
and default ON for ``play`` (single env, the natural place to inspect) and OFF
for training (4096 envs -- the metric plots still work there, but per-step CSV of
one env is rarely what you want during bulk training; flip ``record_csv=True`` to
enable).

Toggling without editing code
-----------------------------
The same switches are exposed as environment variables read at config-build time,
so you can flip them straight from the command line (no source edit, no mjlab
fork). Most useful is enabling the CSV during training, or forcing it off in play::

    DIOGENES_RECORD_CSV=1 uv run train Diogenes-Flat            # log in training
    DIOGENES_RECORD_CSV=0 uv run play  Diogenes-Flat ...        # force off in play
    DIOGENES_CSV_TAG=ablationA DIOGENES_RECORD_CSV=1 uv run train Diogenes-Flat

Recognized vars: ``DIOGENES_RECORD_CSV`` and ``DIOGENES_MONITOR``
(1/true/yes/on or 0/false/no/off) and ``DIOGENES_CSV_TAG`` (filename label). An
explicit keyword argument to ``diogenes_env_cfg`` always wins over the env var.
See the "Terminal-toggle support" block near ``diogenes_env_cfg`` for the full
precedence rules and rationale.

Run a trained policy with::

    uv run play Diogenes-Flat
"""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from . import mdp as diogenes_mdp
from . import monitoring
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


def monitored_joints_cfg() -> SceneEntityCfg:
  """Selects the three actuated joints, order-preserved, for monitoring.

  Order matches ``monitoring.JOINT_NAMES`` = (hip, thigh, calf) so component
  indices line up with the metric/CSV column names.
  """
  return SceneEntityCfg(
    "robot", joint_names=monitoring.JOINT_NAMES, preserve_order=True
  )


def _monitoring_metrics() -> dict[str, MetricsTermCfg]:
  """Build the per-step scalar metric terms that the Viser viewer plots live.

  One term per channel so each gets its own toggle/curve in the Metrics tab:
    * contact force        : foot_force/{x,y,z}, foot_force/mag
    * joint torque         : torque/{hip,thigh,calf}
    * joint power          : power/{hip,thigh,calf}, power/total
    * slider (raw)         : slider/{pos,vel,acc}
    * carriage (+up)       : carriage/{height,vel,acc}
    * joint position       : qpos/{hip,thigh,calf}
    * joint velocity       : qvel/{hip,thigh,calf}
  """
  metrics: dict[str, MetricsTermCfg] = {}

  # --- Foot contact force (world frame, net). ---
  for axis, comp in (("x", 0), ("y", 1), ("z", 2)):
    metrics[f"foot_force/{axis}"] = MetricsTermCfg(
      func=monitoring.contact_force_component_metric,
      params={"sensor_name": FOOT_CONTACT_SENSOR, "component": comp},
    )
  metrics["foot_force/mag"] = MetricsTermCfg(
    func=monitoring.contact_force_magnitude_metric,
    params={"sensor_name": FOOT_CONTACT_SENSOR},
  )

  # --- Per-joint torque, power, position, velocity. ---
  for comp, jname in enumerate(monitoring.JOINT_NAMES):
    metrics[f"torque/{jname}"] = MetricsTermCfg(
      func=monitoring.joint_torque_metric,
      params={"asset_cfg": monitored_joints_cfg(), "component": comp},
    )
    metrics[f"power/{jname}"] = MetricsTermCfg(
      func=monitoring.joint_power_metric,
      params={"asset_cfg": monitored_joints_cfg(), "component": comp},
    )
    metrics[f"qpos/{jname}"] = MetricsTermCfg(
      func=monitoring.joint_pos_metric,
      params={"asset_cfg": monitored_joints_cfg(), "component": comp},
    )
    metrics[f"qvel/{jname}"] = MetricsTermCfg(
      func=monitoring.joint_vel_metric,
      params={"asset_cfg": monitored_joints_cfg(), "component": comp},
    )
  metrics["power/total"] = MetricsTermCfg(
    func=monitoring.total_mechanical_power_metric,
    params={"asset_cfg": monitored_joints_cfg()},
  )

  # --- Slider (raw joint-space) ---
  metrics["slider/pos"] = MetricsTermCfg(
    func=monitoring.slider_pos_metric, params={"asset_cfg": slider_cfg()}
  )
  metrics["slider/vel"] = MetricsTermCfg(
    func=monitoring.slider_vel_metric, params={"asset_cfg": slider_cfg()}
  )
  metrics["slider/acc"] = MetricsTermCfg(
    func=monitoring.slider_acc_metric, params={"asset_cfg": slider_cfg()}
  )

  # --- Carriage (sign-corrected: +up = -slider) ---
  metrics["carriage/height"] = MetricsTermCfg(
    func=monitoring.carriage_height_metric, params={"asset_cfg": slider_cfg()}
  )
  metrics["carriage/vel"] = MetricsTermCfg(
    func=monitoring.carriage_vel_metric, params={"asset_cfg": slider_cfg()}
  )
  metrics["carriage/acc"] = MetricsTermCfg(
    func=monitoring.carriage_acc_metric, params={"asset_cfg": slider_cfg()}
  )

  return metrics


def _monitoring_recorder(run_tag: str = "run") -> dict[str, RecorderTermCfg]:
  """Build the CSV recorder term (one row per control step, env 0)."""
  return {
    "csv": RecorderTermCfg(
      func=monitoring.DiogenesCsvRecorder,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "env_idx": 0,
        "run_tag": run_tag,
        # "path": "logs/diogenes_monitor/my_run.csv",  # optional explicit path
      },
    )
  }


# ---------------------------------------------------------------------------
# Terminal-toggle support via environment variables.
#
# ``register_mjlab_task`` (in __init__.py) builds BOTH the train and play configs
# eagerly at import time, and the play/train CLIs offer no custom flags for our
# monitoring. The cleanest knob that works for both, with NO edits to mjlab, is an
# env var read here at config-build time. Set it on the command line BEFORE the
# task is imported (which is what happens naturally when you launch the CLI):
#
#   DIOGENES_RECORD_CSV=1 uv run play  Diogenes-Flat --wandb-run-path ...
#   DIOGENES_RECORD_CSV=1 uv run train Diogenes-Flat
#   DIOGENES_RECORD_CSV=0 uv run play  Diogenes-Flat ...   # force OFF in play
#   DIOGENES_CSV_TAG=ablationA DIOGENES_RECORD_CSV=1 uv run train Diogenes-Flat
#
# Recognized vars (all optional):
#   DIOGENES_RECORD_CSV : 1/true/yes/on -> on, 0/false/no/off -> off
#   DIOGENES_MONITOR    : same truthy parsing, toggles the live metric plots
#   DIOGENES_CSV_TAG    : string folded into the CSV filename (overrides default)
#
# A direct keyword argument to ``diogenes_env_cfg`` always WINS over the env var;
# the env var only fills in a flag left at its ``None`` default.
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f"}


def _env_bool(name: str) -> bool | None:
  """Parse a boolean env var. Returns None if unset/blank, else True/False.

  Unrecognized non-empty values raise, so a typo (``=ture``) fails loudly rather
  than silently disabling logging.
  """
  raw = os.environ.get(name)
  if raw is None:
    return None
  val = raw.strip().lower()
  if val == "":
    return None
  if val in _TRUTHY:
    return True
  if val in _FALSY:
    return False
  raise ValueError(
    f"Environment variable {name}={raw!r} is not a recognized boolean. "
    f"Use one of {sorted(_TRUTHY)} or {sorted(_FALSY)}."
  )


def diogenes_env_cfg(
  play: bool = False,
  monitor: bool | None = None,
  record_csv: bool | None = None,
  csv_run_tag: str = "run",
) -> ManagerBasedRlEnvCfg:
  """Create the Diogenes periodic-hopping environment configuration.

  Args:
    play: When True, apply evaluation-friendly overrides (effectively infinite
      episode, no observation corruption).
    monitor: Register the live monitoring metric terms (Viser Metrics tab). If
      None, falls back to the ``DIOGENES_MONITOR`` env var, then defaults to True
      (cheap; one scalar per channel per step).
    record_csv: Register the per-step CSV recorder. If None, falls back to the
      ``DIOGENES_RECORD_CSV`` env var, then defaults to ``play`` (record in play,
      not during bulk training). Set True to log a training env-0 trace.
    csv_run_tag: Label folded into the auto-generated CSV filename. The
      ``DIOGENES_CSV_TAG`` env var overrides the default ("run") but not an
      explicitly-passed value.

  Toggle from the terminal without editing code (see the env-var notes above)::

      DIOGENES_RECORD_CSV=1 uv run train Diogenes-Flat
      DIOGENES_RECORD_CSV=0 uv run play  Diogenes-Flat ...   # force off in play
  """
  # Resolve each flag: explicit arg wins; else env var; else built-in default.
  if monitor is None:
    monitor = _env_bool("DIOGENES_MONITOR")
  if monitor is None:
    monitor = True
  if record_csv is None:
    record_csv = _env_bool("DIOGENES_RECORD_CSV")
  if record_csv is None:
    record_csv = play
  # Only let the env var set the tag when the caller left the default in place.
  if csv_run_tag == "run":
    csv_run_tag = os.environ.get("DIOGENES_CSV_TAG", "run") or "run"

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

  # ---------------------------------------------------------------------------
  # Monitoring metrics (live Viser plots) and CSV recorder (offline analysis).
  # ---------------------------------------------------------------------------
  metrics = _monitoring_metrics() if monitor else {}
  recorders = _monitoring_recorder(run_tag=csv_run_tag) if record_csv else {}

  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=20.0,
    sim=sim_cfg,
    scene=scene_cfg,
    observations=observations,
    actions={"joint_pos": joint_pos_action},
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    recorders=recorders,
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
