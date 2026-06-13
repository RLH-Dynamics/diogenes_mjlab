"""Diogenes environment configurations.

The Diogenes robot is a fixed-base, single 3-jointed leg (hip / thigh / calf)
mounted on an unactuated prismatic ``slider`` joint that constrains the body to
move vertically along a rail. This is a hop test stand, not a free-floating
locomotion robot, so the built-in ``velocity`` task does not apply and the env
is built from scratch.

Two carriage trajectories are supported, selected by the ``trajectory`` argument
to ``diogenes_env_cfg`` and registered as two separate tasks (see __init__.py):

  * ``"dual_parabola"`` (task ``Diogenes-Flat``): the original gravity-exact
    dual-parabolic motion. Highly dynamic; its period is derived from physics.
  * ``"sine"`` (task ``Diogenes-Flat-Sine``): a smooth sinusoidal up-and-down
    carriage motion between a minimum and a maximum height. Much gentler and
    intended as an easier FIRST sim-to-real target. Its period is a free design
    parameter (``SINE_PERIOD``); a sinusoid has no physics-derived period.

Both share the same point-foot (x, y) hold reward, the same naturalness /
smoothness / actuation regularizers, the same terminations, and the same phase
clock machinery (each fed with its own period).

Dual-parabola trajectory (gravity-exact, derived period)
--------------------------------------------------------
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

Sinusoidal trajectory (smooth, free period)
-------------------------------------------
The carriage reference is ``mid - amp*cos(2*pi*phi)`` between ``TRAJ_MIN`` and
``TRAJ_MAX`` (mid/amp are their average/half-difference). phi=0 starts at the
bottom, rises to the top at phi=0.5, returns to the bottom at phi=1 -- the same
"start low, push up" feel as the dual-parabola, so the crouched DEFAULT_INIT
pose is a valid start for either. The period ``SINE_PERIOD`` is chosen by you;
the peak vertical acceleration is ``amp * (2*pi/SINE_PERIOD)**2``. The default
below keeps that comfortably under g so the foot stays loaded (non-ballistic) --
the gentle motion wanted for a first transfer. See ``mdp.slider_sinusoid_tracking``.

Reward design
-------------
  * ``slider_dual_parabola``    : (dual_parabola) dense Gaussian tracking the
                                   gravity-exact carriage reference.
  * ``slider_sinusoid``         : (sine) dense Gaussian tracking the sinusoidal
                                   carriage reference.
  * ``foot_xy_position``        : dense Gaussian keeping the point-foot's world
                                   (x, y) fixed, so the leg hops straight up/down.
                                   IDENTICAL for both trajectories.
  * Naturalness/smoothness + actuation regularizers: IDENTICAL for both.

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

    uv run play Diogenes-Flat          # dual-parabola
    uv run play Diogenes-Flat-Sine     # sinusoid
"""

import os
from typing import Literal

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

# Selector for which carriage trajectory the env tracks.
TrajectoryType = Literal["dual_parabola", "sine"]

# Names of the XML-defined <position> actuators (== the actuated joint names).
DIOGENES_ACTUATOR_NAMES = ("hip", "thigh", "calf")

# ---------------------------------------------------------------------------
# Carriage trajectory geometry, SHARED by both trajectories. All three heights
# are z-values RELATIVE TO THE SLIDER ORIGIN (== the carriage start position;
# height above start = -slider_pos, verified). Require
# TRAJ_MAX >= TRAJ_TRANSITION > TRAJ_MIN.
#   * TRAJ_MAX        : top of the motion (dual-parabola flight apex; sine peak).
#   * TRAJ_MIN        : bottom of the motion (dual-parabola recovery dip; sine
#                       trough).
#   * TRAJ_TRANSITION : (dual-parabola only) height where flight and recovery
#                       arcs meet, also the cycle boundary. UNUSED by the sine.
#   * GRAVITY         : (dual-parabola only) free-fall accel for the flight arc.
# ---------------------------------------------------------------------------
TRAJ_MAX = 0.15
TRAJ_MIN = 0.05
TRAJ_TRANSITION = 0.12
GRAVITY = diogenes_mdp.GRAVITY  # 9.81 m/s^2

# Dual-parabola derived cycle period (seconds). Computed once so the phase clock,
# the trajectory reward, and any other phase-keyed term all share the SAME
# period. This is the single source of truth for dual-parabola timing.
TRAJ_T = diogenes_mdp.dual_parabola_timing(
  TRAJ_MIN, TRAJ_MAX, TRAJ_TRANSITION, GRAVITY
)[0]

# ---------------------------------------------------------------------------
# Sinusoid period (seconds) -- the FREE design parameter for the sine task.
#
# A sinusoid has no physics-derived period, so you pick it. It trades off
# directly against the carriage's peak vertical acceleration:
#
#     amp        = (TRAJ_MAX - TRAJ_MIN) / 2
#     peak_accel = amp * (2*pi / SINE_PERIOD)**2     [m/s^2]
#
# Keep peak_accel comfortably BELOW g (9.81) and the foot never goes ballistic --
# it stays loaded against the floor through the whole cycle, the gentle,
# well-behaved motion wanted for a first sim-to-real transfer. With the default
# amplitude (0.15 m) the peak accel at each period is roughly:
#     SINE_PERIOD = 0.8 s -> ~9.3 m/s^2   (near g; getting dynamic -- avoid first)
#     SINE_PERIOD = 1.0 s -> ~5.9 m/s^2
#     SINE_PERIOD = 1.2 s -> ~4.1 m/s^2   (default; gentle, non-ballistic)
#     SINE_PERIOD = 1.5 s -> ~2.6 m/s^2   (even gentler)
# Lower SINE_PERIOD (faster hop) raises the accel QUADRATICALLY -- tune this
# first if you later want a livelier motion.
# ---------------------------------------------------------------------------
SINE_PERIOD = 1.2

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


def _slider_trajectory_reward(trajectory: TrajectoryType) -> tuple[RewardTermCfg, float]:
  """Build the slider-tracking reward term and its phase-clock period.

  Returns ``(reward_term, phase_period)`` so the caller can wire the SAME period
  into both the reward (implicitly, via the term) and the phase-clock observation.
  Both trajectories use weight 30.0 and std 0.1 so the slider-tracking signal is
  identical in strength between the two tasks -- only the reference shape and the
  period differ.

  * dual_parabola: period derived from physics (``TRAJ_T``); reward owns the full
    free-fall + recovery reference.
  * sine: period is the free ``SINE_PERIOD``; reward tracks ``mid-amp*cos``.
  """
  if trajectory == "dual_parabola":
    term = RewardTermCfg(
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
  """Build the trajectory-appropriate foot/ground contact-phase penalty.

  Returns a RewardTermCfg whose func returns a per-step 0/1 cost; the NEGATIVE
  weight here turns it into a penalty. Tune the weight to taste -- it starts
  active (unlike the 0.0-weight shaping terms) because keeping the foot planted
  on schedule is core to the motion you want, not optional polish.

  * sine: penalize ANY airborne step (foot should stay planted all cycle). This
    is the term that targets the "short hops as the carriage drops" issue.
  * dual_parabola: penalize the WRONG contact state for the current phase --
    contact during the flight (upper) arc OR air during the stance (lower) arc.
    The flight/stance boundary is read from the SAME geometry as the slider
    reward, so the two share one phase clock.
  """
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


def diogenes_env_cfg(
  play: bool = False,
  trajectory: TrajectoryType = "dual_parabola",
  monitor: bool | None = None,
  record_csv: bool | None = None,
  csv_run_tag: str = "run",
) -> ManagerBasedRlEnvCfg:
  """Create the Diogenes periodic-hopping environment configuration.

  Args:
    play: When True, apply evaluation-friendly overrides (effectively infinite
      episode, no observation corruption).
    trajectory: Which carriage motion to track. ``"dual_parabola"`` (default)
      is the original gravity-exact dynamic motion; ``"sine"`` is the smooth,
      gentle sinusoidal motion intended for a first sim-to-real transfer. Only
      the slider-tracking reward and the phase-clock period differ between the
      two; the foot-xy hold, all regularizers and terminations are identical.
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

  # Resolve the slider-tracking reward and the matching phase-clock period for
  # the chosen trajectory. The SAME period drives the reward and the phase clock.
  slider_reward, phase_period = _slider_trajectory_reward(trajectory)

  # Trajectory-appropriate foot/ground contact-phase penalty (per-step 0/1 cost,
  # negative weight). Sine: penalize any airborne step. Dual-parabola: penalize
  # the wrong contact state for the phase (contact in flight, air in stance).
  contact_phase_reward = _contact_phase_reward(trajectory)

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
    # Phase clock uses the trajectory's period (derived for dual-parabola, the
    # free SINE_PERIOD for sine) so it stays in lockstep with the slider reward.
    "phase_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock,
      params={"hop_period": phase_period},
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
    # --- Slider trajectory tracking (dense). Trajectory-specific term; same
    #     weight/std for both so only the reference shape + period differ. ---
    "slider_trajectory": slider_reward,
    # --- Point-foot world (x, y) hold (dense). IDENTICAL for both trajectories. ---
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
    # --- Foot/ground contact-phase penalty (per-step 0/1 cost x negative weight).
    #     Trajectory-specific: sine penalizes any airborne step; dual-parabola
    #     penalizes contact-in-flight and air-in-stance. ---
    "contact_phase": contact_phase_reward,
    # --- Contact-force shaping (naturalness / smoothness). ---
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
