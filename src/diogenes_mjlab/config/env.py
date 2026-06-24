"""Diogenes environment configuration orchestrator.

Assembles the full ``ManagerBasedRlEnvCfg`` from the sub-builders in the
``config`` package sub-modules.
"""

import os
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg
from mjlab.envs import mdp

from ..constants import (
  DIOGENES_ACTUATOR_NAMES,
  FOOT_CONTACT_SENSOR,
  JOINT_LIMIT_MARGIN,
)
from ..flags import _env_bool, _env_float
from ..diogenes.diogenes_constants import get_diogenes_cfg
from .. import mdp as diogenes_mdp

from .domain_rand import _domain_randomization_events
from .instrumentation import _monitoring_metrics, _monitoring_recorder
from .observations import _proprio_terms
from .rewards import _build_rewards
from .entities import actuated_joints_cfg

# Selector for which carriage trajectory the env tracks.
TrajectoryType = Literal["dual_parabola", "sine"]


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
      None, falls back to the ``DIOGENES_MONITOR`` env var, then defaults to True.
    record_csv: Register the per-step CSV recorder. If None, falls back to the
      ``DIOGENES_RECORD_CSV`` env var, then defaults to ``play``.
    csv_run_tag: Label folded into the auto-generated CSV filename.
    domain_rand: Register the startup domain-randomization events. If None,
      falls back to ``DIOGENES_DOMAIN_RAND``, then defaults to ``not play``.
    obs_noise: Attach additive sensor noise AND the observation delay to the
      ACTOR proprio terms. If None, falls back to ``DIOGENES_OBS_NOISE``, then
      defaults to ``not play``.
    dr_scale: Multiplier on every startup DR range half-width. If None, falls
      back to ``DIOGENES_DR_SCALE``, then defaults to 1.0.
    reset_joints: Register the random LEGAL joint start-pose reset event. If
      None, falls back to ``DIOGENES_RESET_JOINTS``, then defaults to ``not play``.
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
  if csv_run_tag == "run":
    csv_run_tag = os.environ.get("DIOGENES_CSV_TAG", "run") or "run"

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
  if reset_joints is None:
    reset_joints = _env_bool("DIOGENES_RESET_JOINTS")
  if reset_joints is None:
    reset_joints = not play

  # Build rewards and recover the phase-clock period for the chosen trajectory.
  rewards, phase_period = _build_rewards(trajectory)

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
  # on; the critic copy is always clean (asymmetric actor-critic).
  # ---------------------------------------------------------------------------
  actor_terms = _proprio_terms(obs_noise=obs_noise)
  actor_terms["phase_clock"].params["hop_period"] = phase_period

  # Asymmetric actor-critic: the slider (carriage) state is PRIVILEGED.
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
  # Events: startup domain randomization + random reset start pose.
  # ---------------------------------------------------------------------------
  events = (
    _domain_randomization_events(dr_scale, reset_joint_pose=reset_joints)
    if domain_rand
    else {}
  )

  # ---------------------------------------------------------------------------
  # Terminations.
  # ---------------------------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "joint_at_limit": TerminationTermCfg(
      func=diogenes_mdp.joint_at_limit,
      params={
        "asset_cfg": actuated_joints_cfg(),
        "margin": JOINT_LIMIT_MARGIN,
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
