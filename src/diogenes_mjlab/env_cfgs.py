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

Sim-to-real preparation (domain randomization, observation noise/delay)
-----------------------------------------------------------------------
For an initial sim-to-real transfer the env adds, all behind independent toggle
flags (see "Terminal-toggle support" below):

  * Domain randomization (``cfg.events``): per-world randomization of PD gains,
    mass+inertia (physics-consistent pseudo-inertia), centre-of-mass offsets,
    joint armature and friction, slider rail friction (N, not N·m),
    foot/ground geom friction, and joint encoder
    bias (all ``mode="startup"``), PLUS a random LEGAL joint START POSE applied
    every reset (``mode="reset"``) so the leg learns to recover into the hop
    cycle from any legal orientation. Built by ``_domain_randomization_events``.
    A single scalar ``dr_scale`` widens every startup range about its nominal
    centre so you can sweep DR strength from one knob (it does NOT widen the
    reset-pose term, which is bounded by the joint ranges, not a tunable band).
    NO external pushes/perturbations are applied (the leg is on a fixed vertical
    stand).
  * Observation noise + delay (actor group only; the critic stays clean to keep
    the asymmetric value function exact): additive uniform sensor noise on the
    proprioceptive terms, plus a per-term observation DELAY of up to
    ``OBS_DELAY_MAX_LAG`` control steps (a "time delay" in the
    ManagerBasedRlEnv observation pipeline). At 50 Hz control, lag 3 == up to
    60 ms of sensorimotor latency, modelling the real read+actuate round-trip.

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

    DIOGENES_DOMAIN_RAND=0 uv run train Diogenes-Flat           # DR off (ablation)
    DIOGENES_OBS_NOISE=0   uv run train Diogenes-Flat           # noise/delay off
    DIOGENES_DR_SCALE=1.5  uv run train Diogenes-Flat           # widen DR ranges
    DIOGENES_RESET_JOINTS=0 uv run train Diogenes-Flat          # fixed start pose

Recognized vars: ``DIOGENES_RECORD_CSV``, ``DIOGENES_MONITOR``,
``DIOGENES_DOMAIN_RAND``, ``DIOGENES_OBS_NOISE`` (1/true/yes/on or
0/false/no/off), ``DIOGENES_DR_SCALE`` (float), ``DIOGENES_RESET_JOINTS``
(toggle the random start pose) and ``DIOGENES_CSV_TAG`` (filename label). An
explicit keyword argument to ``diogenes_env_cfg`` always wins over the env var.

Run a trained policy with::

    uv run play Diogenes-Flat          # dual-parabola
    uv run play Diogenes-Flat-Sine     # sinusoid
"""

import os
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
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
from mjlab.utils.noise import UniformNoiseCfg

from . import mdp as diogenes_mdp
from . import monitoring
from .constants import (
  DIOGENES_ACTUATOR_NAMES,
  FOOT_CONTACT_SENSOR,
  FOOT_GEOM_NAME,
  GRAVITY,
  JOINT_LIMIT_MARGIN,
  OBS_DELAY_MAX_LAG,
  OBS_DELAY_MIN_LAG,
  OBS_NOISE_JOINT_POS,
  OBS_NOISE_JOINT_VEL,
  OBS_NOISE_SLIDER_POS,
  OBS_NOISE_SLIDER_VEL,
  SINE_PERIOD,
  TRAJ_MAX,
  TRAJ_MIN,
  TRAJ_TRANSITION,
)
from .diogenes.diogenes_constants import get_diogenes_cfg

# Selector for which carriage trajectory the env tracks.
TrajectoryType = Literal["dual_parabola", "sine"]

# Dual-parabola derived cycle period (seconds). Computed once so the phase clock,
# the trajectory reward, and any other phase-keyed term all share the SAME
# period. This is the single source of truth for dual-parabola timing.
TRAJ_T = diogenes_mdp.dual_parabola_timing(
  TRAJ_MIN, TRAJ_MAX, TRAJ_TRANSITION, GRAVITY
)[0]

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


def all_links_cfg() -> SceneEntityCfg:
  """Selects the moving leg links for mass / inertia / COM randomization.

  Excludes ``base_link`` (a ~zero-mass anchor) so we only perturb the real
  inertial bodies of the leg.
  """
  return SceneEntityCfg(
    "robot",
    body_names=("leg_mount_assy", "hip_assy", "thigh_assy", "calf_assy"),
  )


def foot_geom_cfg() -> SceneEntityCfg:
  """Selects the named foot collision geom for friction randomization."""
  return SceneEntityCfg("robot", geom_names=(FOOT_GEOM_NAME,))


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
# Domain randomization events (sim-to-real).
#
# The startup terms run at ``mode="startup"`` (sampled once per world when the
# model is built), which is the right cadence for fixed physical properties:
# gains, mass, inertia, COM, armature, friction and encoder bias do not change
# within an episode on the real robot. The random START POSE term instead runs
# at ``mode="reset"`` (re-sampled every episode), because the start pose IS an
# episode-level quantity -- you want a fresh legal orientation each rollout.
# Every term is per-world (each of the 4096 envs gets its own draw), so the
# policy trains across a distribution of plants and start poses.
#
# ``dr_scale`` widens each STARTUP range symmetrically about its nominal centre.
# At 1.0 the ranges below are used verbatim (carried over from the previous Isaac
# Lab leg project where applicable); raise it (e.g. 1.5) to stress-test
# robustness, lower it (e.g. 0.5) to soften DR for an early, easier curriculum
# stage. It does NOT scale the reset-pose term, whose spread is fixed by the
# joint ranges (inset by JOINT_LIMIT_MARGIN), not a tunable band.
#
# NOTE: no push/perturbation event is included -- the leg is on a fixed vertical
# stand, so external base shoves are not physically meaningful here.
# ---------------------------------------------------------------------------


def _scale_range(
  lo: float, hi: float, dr_scale: float
) -> tuple[float, float]:
  """Widen ``(lo, hi)`` about its midpoint by ``dr_scale``.

  dr_scale == 1.0 returns the range unchanged; 2.0 doubles its width while
  keeping the same centre. Used so a single knob sweeps overall DR strength.
  """
  mid = 0.5 * (lo + hi)
  half = 0.5 * (hi - lo) * dr_scale
  return (mid - half, mid + half)


# ---------------------------------------------------------------------------
# Per-term enable switches for domain randomization.
#
# Flip any of these to False to drop that single DR term while leaving the rest
# active -- handy for isolating which term destabilizes a policy (e.g. play a
# clean-trained policy with only ``ENABLE_ENCODER_BIAS = True`` to see if the
# encoder bias alone is what drives the leg into its joint limits). These are the
# master on/off per term; ``DIOGENES_DOMAIN_RAND`` still gates ALL of them at
# once, and ``dr_scale`` still sets each enabled startup term's range width.
#
# ``ENABLE_RESET_JOINT_POSE`` toggles the random LEGAL start pose specifically;
# it can also be flipped from the terminal via ``DIOGENES_RESET_JOINTS`` (which
# takes precedence by gating the term at build time -- see diogenes_env_cfg).
# ---------------------------------------------------------------------------
ENABLE_PD_GAINS = True
ENABLE_LINK_INERTIAL = True
ENABLE_COM_OFFSET = True
ENABLE_JOINT_ARMATURE = True
ENABLE_JOINT_FRICTION = True
ENABLE_FOOT_FRICTION = True
ENABLE_SLIDER_FRICTION = True
ENABLE_ENCODER_BIAS = True
ENABLE_RESET_JOINT_POSE = True


def _domain_randomization_events(
  dr_scale: float, reset_joint_pose: bool = True
) -> dict[str, EventTermCfg]:
  """Build the domain-randomization event terms (startup + reset).

  The dict is assembled conditionally: each term is included only if its
  ``ENABLE_*`` module-level switch above is True. Toggle those flags to enable or
  disable individual terms (e.g. for an ablation or to isolate a destabilizing
  term) without commenting out code.

  Args:
    dr_scale: multiplier on every STARTUP randomization range's half-width
      (1.0 = the nominal ranges below). Applied via ``_scale_range`` so the
      centre is held. Does not affect the reset-pose term.
    reset_joint_pose: include the random LEGAL start-pose reset term. Gated
      separately (by ``diogenes_env_cfg``, which folds in the
      ``DIOGENES_RESET_JOINTS`` env var and the play default) so the start pose
      can be ablated without touching the startup DR. Still also requires the
      ``ENABLE_RESET_JOINT_POSE`` module switch.

  Returns:
    A dict of ``EventTermCfg`` keyed by a short term name, ready to pass as
    ``cfg.events``. Only the enabled terms are present.
  """
  s = dr_scale
  events: dict[str, EventTermCfg] = {}

  # --- PD gains (the single most important actuator DR for sim-to-real).
  #     Scale about the nominal kp=60 / kv=4 by roughly +-12% ---
  if ENABLE_PD_GAINS:
    events["pd_gains"] = EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", actuator_names=DIOGENES_ACTUATOR_NAMES
        ),
        "kp_range": _scale_range(0.89, 1.12, s),
        "kd_range": _scale_range(0.89, 1.12, s),
        "operation": "scale",
        "distribution": "uniform",
      },
    )

  # --- Mass + inertia, physics-consistent. pseudo_inertia scales mass AND
  #     inertia together (unlike randomizing body_mass alone, which leaves
  #     inertia stale). alpha_range is the uniform-density scale factor;---
  if ENABLE_LINK_INERTIAL:
    events["link_inertial"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": all_links_cfg(),
        "alpha_range": _scale_range(-0.11, 0.09, s),
        "distribution": "uniform",
      },
    )

  # --- Centre-of-mass offset, +-25 mm per axis (Isaac project's COM range). ---
  if ENABLE_COM_OFFSET:
    events["com_offset"] = EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": all_links_cfg(),
        "ranges": {
          0: _scale_range(-0.025, 0.025, s),
          1: _scale_range(-0.025, 0.025, s),
          2: _scale_range(-0.025, 0.025, s),
        },
        "operation": "add",
        "distribution": "uniform",
      },
    )

  # --- Joint armature (reflected rotor inertia), abs. Isaac range (0.008,
  #     0.020); the XML nominal is 0.005, so this also lifts it to a realistic
  #     band. ---
  if ENABLE_JOINT_ARMATURE:
    events["joint_armature"] = EventTermCfg(
      func=dr.joint_armature,
      mode="startup",
      params={
        "asset_cfg": actuated_joints_cfg(),
        "ranges": _scale_range(0.016, 0.024, s),
        "operation": "abs",
        "distribution": "uniform",
      },
    )

  # --- Joint dry friction (frictionloss), abs. Isaac range (0.15, 1.60) --
  #     the leg's biggest unmodelled real effect, so worth a wide band. ---
  if ENABLE_JOINT_FRICTION:
    events["joint_friction"] = EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={
        "asset_cfg": actuated_joints_cfg(),
        "ranges": _scale_range(0.15, 1.60, s),
        "operation": "abs",
        "distribution": "uniform",
      },
    )

  # --- Foot/ground tangential friction, abs. Matches the velocity task's
  #     foot-friction default band (0.3, 1.2); critical for a hopper. Selects
  #     the named "foot" geom added to diogenes.xml. ---
  if ENABLE_FOOT_FRICTION:
    events["foot_friction"] = EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": foot_geom_cfg(),
        "ranges": _scale_range(0.3, 1.2, s),
        "operation": "abs",
        "distribution": "uniform",
        "shared_random": True,
      },
    )

  # --- Slider (rail) Coulomb dry-friction, abs. Band is in NEWTONS (slide joint),
  #     NOT N·m like the rotary joint_friction above. Covers the dual MGN12H
  #     dual-rail system (4 blocks total): well-broken-in/warm rail at low end,
  #     cold/tight-preload or contaminated rail at high end. The XML nominal
  #     frictionloss="2.0" N is the clean-plant value used when DR is off
  #     (play mode); this term overwrites it per-world during training. ---
  if ENABLE_SLIDER_FRICTION:
    events["slider_friction"] = EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={
        "asset_cfg": slider_cfg(),
        "ranges": _scale_range(1.0, 3.5, s),
        "operation": "abs",
        "distribution": "uniform",
      },
    )

  # --- Joint encoder bias, +-0.015 rad. Models per-joint calibration offset
  #     on the real encoders (a constant added to the measured joint angle). ---
  if ENABLE_ENCODER_BIAS:
    events["encoder_bias"] = EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={
        "asset_cfg": actuated_joints_cfg(),
        "bias_range": _scale_range(-0.015, 0.015, s),
      },
    )

  # --- Random LEGAL joint start pose (mode="reset", NOT startup). Re-sampled
  #     every episode so the leg learns to recover into the hop cycle from any
  #     legal orientation. Each actuated joint starts uniformly within its range,
  #     inset by JOINT_LIMIT_MARGIN (the SAME margin the joint_at_limit
  #     termination uses) plus a small safety epsilon, so a fresh start never
  #     trips the limit termination on step 0. The slider is NOT randomized
  #     (only the actuated joints are selected), so the carriage still starts at
  #     its trajectory-consistent height. dr_scale does NOT widen this term --
  #     its spread is bounded by the joint ranges, not a tunable band. ---
  if reset_joint_pose and ENABLE_RESET_JOINT_POSE:
    events["reset_joint_pose"] = EventTermCfg(
      func=diogenes_mdp.reset_joints_uniform_legal,
      mode="reset",
      params={
        "asset_cfg": actuated_joints_cfg(),
        "margin": JOINT_LIMIT_MARGIN,  # lockstep with joint_at_limit
        "safety_eps": 1e-3,
        "velocity_range": (0.0, 0.0),  # start at rest
      },
    )

  return events


# ---------------------------------------------------------------------------
# Terminal-toggle support via environment variables.
#
# ``register_mjlab_task`` (in __init__.py) builds BOTH the train and play configs
# eagerly at import time, and the play/train CLIs offer no custom flags for our
# monitoring / sim-to-real switches. The cleanest knob that works for both, with
# NO edits to mjlab, is an env var read here at config-build time. Set it on the
# command line BEFORE the task is imported (which is what happens naturally when
# you launch the CLI):
#
#   DIOGENES_RECORD_CSV=1   uv run play  Diogenes-Flat --wandb-run-path ...
#   DIOGENES_RECORD_CSV=1   uv run train Diogenes-Flat
#   DIOGENES_RECORD_CSV=0   uv run play  Diogenes-Flat ...   # force OFF in play
#   DIOGENES_DOMAIN_RAND=0  uv run train Diogenes-Flat       # DR off (ablation)
#   DIOGENES_OBS_NOISE=0    uv run train Diogenes-Flat       # noise/delay off
#   DIOGENES_DR_SCALE=1.5   uv run train Diogenes-Flat       # widen DR ranges
#   DIOGENES_RESET_JOINTS=0 uv run train Diogenes-Flat       # fixed start pose
#   DIOGENES_CSV_TAG=ablationA DIOGENES_RECORD_CSV=1 uv run train Diogenes-Flat
#
# Recognized vars (all optional):
#   DIOGENES_RECORD_CSV   : 1/true/yes/on -> on, 0/false/no/off -> off
#   DIOGENES_MONITOR      : same truthy parsing, toggles the live metric plots
#   DIOGENES_DOMAIN_RAND  : same truthy parsing, toggles startup DR events
#   DIOGENES_OBS_NOISE    : same truthy parsing, toggles actor obs noise + delay
#   DIOGENES_RESET_JOINTS : same truthy parsing, toggles the random start pose
#   DIOGENES_DR_SCALE     : float, multiplies every startup DR range half-width
#   DIOGENES_CSV_TAG      : string folded into the CSV filename (overrides default)
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


def _env_float(name: str) -> float | None:
  """Parse a float env var. Returns None if unset/blank, else the value.

  Unparseable non-empty values raise so a typo fails loudly.
  """
  raw = os.environ.get(name)
  if raw is None:
    return None
  val = raw.strip()
  if val == "":
    return None
  try:
    return float(val)
  except ValueError as exc:
    raise ValueError(
      f"Environment variable {name}={raw!r} is not a valid float."
    ) from exc


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


def _proprio_terms(obs_noise: bool) -> dict[str, ObservationTermCfg]:
  """Build the proprioceptive observation terms shared by actor and critic.

  Args:
    obs_noise: when True, attach additive uniform sensor noise AND a per-term
      observation delay (up to ``OBS_DELAY_MAX_LAG`` control steps) to each
      term. Intended for the ACTOR group only; pass False for the critic so the
      value function sees the clean state (asymmetric actor-critic).

  The two groups must NOT share term instances (the delay buffer is per-term),
  so this returns a fresh dict on every call.
  """
  # Noise configs (None when obs_noise is False so the clean critic copy reuses
  # this same builder).
  jp_noise = UniformNoiseCfg(
    n_min=-OBS_NOISE_JOINT_POS, n_max=OBS_NOISE_JOINT_POS
  ) if obs_noise else None
  jv_noise = UniformNoiseCfg(
    n_min=-OBS_NOISE_JOINT_VEL, n_max=OBS_NOISE_JOINT_VEL
  ) if obs_noise else None
  sp_noise = UniformNoiseCfg(
    n_min=-OBS_NOISE_SLIDER_POS, n_max=OBS_NOISE_SLIDER_POS
  ) if obs_noise else None
  sv_noise = UniformNoiseCfg(
    n_min=-OBS_NOISE_SLIDER_VEL, n_max=OBS_NOISE_SLIDER_VEL
  ) if obs_noise else None

  # Delay lags (0 when obs_noise is False -> the pipeline skips the delay buffer
  # entirely for that term).
  dmin = OBS_DELAY_MIN_LAG if obs_noise else 0
  dmax = OBS_DELAY_MAX_LAG if obs_noise else 0

  return {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": actuated_joints_cfg()},
      noise=jp_noise,
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": actuated_joints_cfg()},
      noise=jv_noise,
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    "slider_pos": ObservationTermCfg(
      func=diogenes_mdp.slider_pos,
      params={"asset_cfg": slider_cfg()},
      noise=sp_noise,
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    "slider_vel": ObservationTermCfg(
      func=diogenes_mdp.slider_vel,
      params={"asset_cfg": slider_cfg()},
      noise=sv_noise,
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    "last_action": ObservationTermCfg(
      func=mdp.last_action,
      # No sensor noise on the policy's own previous action, but DO delay it so
      # the policy's notion of "what I just commanded" carries the same latency
      # as the proprioception it is paired with.
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    # Phase clock uses the trajectory's period (derived for dual-parabola, the
    # free SINE_PERIOD for sine) so it stays in lockstep with the slider reward.
    # The clock is an exact, noiseless timing signal (it is an onboard counter on
    # the real robot too), so it is never corrupted or delayed.
    "phase_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock,
      params={"hop_period": _PHASE_PERIOD_PLACEHOLDER},
    ),
  }


# Sentinel replaced per-call inside diogenes_env_cfg (the phase period depends on
# the chosen trajectory, resolved at build time).
_PHASE_PERIOD_PLACEHOLDER = 0.6


def diogenes_env_cfg(
  play: bool = False,
  trajectory: TrajectoryType = "dual_parabola",
  monitor: bool | None = None,
  record_csv: bool | None = None,
  csv_run_tag: str = "run",
  domain_rand: bool | None = None,
  obs_noise: bool | None = None,
  dr_scale: float | None = None,
  reset_joints: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the Diogenes periodic-hopping environment configuration.

  Args:
    play: When True, apply evaluation-friendly overrides (effectively infinite
      episode, no observation corruption, and -- unless explicitly forced on --
      domain randomization and observation noise/delay default OFF so playback
      reflects the clean nominal plant).
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
    domain_rand: Register the startup domain-randomization events (PD gains,
      mass/inertia, COM, armature, friction, slider rail friction, foot friction,
      encoder bias) AND, subject to ``reset_joints``, the random start-pose
      reset event. If None,
      falls back to ``DIOGENES_DOMAIN_RAND``, then defaults to ``True`` for
      training and ``False`` for ``play`` (clean playback). Turn OFF to ablate
      all DR (startup and reset pose) at once.
    obs_noise: Attach additive sensor noise AND the observation delay to the
      ACTOR proprio terms. If None, falls back to ``DIOGENES_OBS_NOISE``, then
      defaults to ``True`` for training and ``False`` for ``play``. The critic
      group is always clean.
    dr_scale: Multiplier on every startup DR range half-width (1.0 = nominal
      ranges). If None, falls back to ``DIOGENES_DR_SCALE``, then defaults to
      1.0. Does not affect the reset start-pose term.
    reset_joints: Register the random LEGAL joint start-pose reset event so the
      leg begins each episode in an arbitrary legal orientation. If None, falls
      back to ``DIOGENES_RESET_JOINTS``, then defaults to ``True`` for training
      and ``False`` for ``play`` (deterministic default-pose start during
      evaluation). This term also lives under the ``domain_rand`` gate, so it is
      only added when domain randomization is on AND ``reset_joints`` resolves
      True AND the ``ENABLE_RESET_JOINT_POSE`` module switch is set.

  Toggle from the terminal without editing code (see the env-var notes above)::

      DIOGENES_RECORD_CSV=1    uv run train Diogenes-Flat
      DIOGENES_RECORD_CSV=0    uv run play  Diogenes-Flat ...  # force off in play
      DIOGENES_DOMAIN_RAND=0   uv run train Diogenes-Flat      # ablate ALL DR
      DIOGENES_OBS_NOISE=0     uv run train Diogenes-Flat      # ablate noise/delay
      DIOGENES_DR_SCALE=1.5    uv run train Diogenes-Flat      # widen DR ranges
      DIOGENES_RESET_JOINTS=0  uv run train Diogenes-Flat      # fixed start pose
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

  # Sim-to-real switches: default ON for training, OFF for play (clean nominal
  # plant during evaluation) unless explicitly forced.
  if domain_rand is None:
    domain_rand = _env_bool("DIOGENES_DOMAIN_RAND")
  if domain_rand is None:
    domain_rand = not play
  if obs_noise is None:
    obs_noise = _env_bool("DIOGENES_OBS_NOISE")
  if obs_noise is None:
    obs_noise = not play
  if dr_scale is None:
    dr_scale = _env_float("DIOGENES_DR_SCALE")
  if dr_scale is None:
    dr_scale = 1.0
  # Random start pose: default ON for training, OFF for play. Resolved
  # independently of the other DR flags but still gated by domain_rand below.
  if reset_joints is None:
    reset_joints = _env_bool("DIOGENES_RESET_JOINTS")
  if reset_joints is None:
    reset_joints = not play

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
  # Observations. Actor proprio carries sensor noise + delay when obs_noise is
  # on; the critic copy is always clean (asymmetric actor-critic). Both groups
  # get their OWN term instances (the delay buffer is per-term), and the phase
  # clock's period is patched in for the chosen trajectory.
  # ---------------------------------------------------------------------------
  actor_terms = _proprio_terms(obs_noise=obs_noise)
  actor_terms["phase_clock"].params["hop_period"] = phase_period

  # Asymmetric actor-critic: the slider (carriage) state is PRIVILEGED -- the
  # critic may see it to stabilize value estimation, but the actor must NOT,
  # so the deployed policy needs no rail sensor. Drop the two slider terms from
  # the actor group only. (critic_terms is a separate dict from a fresh
  # _proprio_terms() call, so these deletes don't touch it.)
  del actor_terms["slider_pos"]
  del actor_terms["slider_vel"]

  critic_terms = _proprio_terms(obs_noise=False)
  critic_terms["phase_clock"].params["hop_period"] = phase_period

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # ---------------------------------------------------------------------------
  # Events: startup domain randomization + random reset start pose (sim-to-real).
  # Empty when domain_rand is off, so the env runs on the clean nominal plant and
  # from the default start pose. The reset-pose term is additionally gated by
  # ``reset_joints`` so the start pose can be ablated on its own while leaving the
  # startup DR active (pass reset_joints=False / DIOGENES_RESET_JOINTS=0).
  # ---------------------------------------------------------------------------
  events = (
    _domain_randomization_events(dr_scale, reset_joint_pose=reset_joints)
    if domain_rand
    else {}
  )

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
      weight=100.0,
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
    # --- Actuation regularizers. ---
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
    # --- Motion smoothness (sim-to-real): penalize jerky commands and jerky
    #     joint motion so the deployed policy is gentle on the hardware. ---
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),
    # Second-order action smoothness (jerk): penalizes the action acceleration,
    # the standard companion to action_rate for a smooth, deployable policy.
    "action_acc": RewardTermCfg(
      func=mdp.action_acc_l2,
      weight=-0.001,
    ),
    # Joint-acceleration penalty (was commented out in the Isaac Lab project);
    # discourages high-frequency joint chatter that does not transfer.
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
        "margin": JOINT_LIMIT_MARGIN,  # lockstep with reset_joints_uniform_legal
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
    events=events,
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
