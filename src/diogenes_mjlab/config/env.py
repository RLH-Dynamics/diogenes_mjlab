"""Diogenes environment configuration orchestrator.

Assembles the full ``ManagerBasedRlEnvCfg`` from the sub-builders in the
``config`` package sub-modules.
"""

import os
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.envs import mdp

from ..constants import (
  CONTACT_PHASE_MARGIN,
  CONTACT_PHASE_TERM_ENABLED,
  CONTACT_PHASE_TERM_NAME,
  DIOGENES_ACTUATOR_NAMES,
  FOOT_CONTACT_SENSOR,
  GRAVITY,
  JOINT_LIMIT_MARGIN,
  OBS_HISTORY_LENGTH,
  TRAJ_MAX,
  TRAJ_MIN,
  TRAJ_TRANSITION,
)
from ..flags import _env_bool, _env_float, _env_int
from ..diogenes.diogenes_constants import get_diogenes_cfg
from .. import mdp as diogenes_mdp

from .domain_rand import _domain_randomization_events
from .instrumentation import _monitoring_metrics, _monitoring_recorder
from .observations import (
  PRIVILEGED_OBS_TERMS,
  _actor_terms,
  _critic_terms,
)
from .rewards import _build_rewards
from .entities import actuated_joints_cfg

# Selector for which carriage trajectory the env tracks.
TrajectoryType = Literal["dual_parabola", "sine", "spring"]


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
  contact_phase_term: bool | None = None,
  obs_history: int | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the Diogenes periodic-hopping environment configuration.

  Args:
    play: When True, apply evaluation-friendly overrides (effectively infinite
      episode, no observation corruption, and -- unless explicitly forced on --
      domain randomization and observation noise/delay default OFF so playback
      reflects the clean nominal plant).
    trajectory: Which carriage motion to track. ``"dual_parabola"`` (default)
      is the original gravity-exact dynamic motion; ``"sine"`` is the smooth,
      gentle sinusoidal motion intended for a first sim-to-real transfer;
      ``"spring"`` keeps the dual-parabola free-fall flight arc but replaces the
      constant-acceleration recovery with a Hooke's-law spring stance. Only the
      slider-tracking reward, the phase-clock period and (for sine vs the two
      flight trajectories) the contact-phase term differ; the foot-xy hold and
      all regularizers are identical.
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
    contact_phase_term: Register the contact-phase-violation termination (and
      its dedicated penalty reward).  If None, falls back to the
      ``DIOGENES_CONTACT_PHASE_TERM`` env var, then defaults to True.  Set to
      False (or ``DIOGENES_CONTACT_PHASE_TERM=0``) to disable during curriculum
      warm-up or ablation runs.
    obs_history: Number of past timesteps included in the actor observation
      (0 = current step only; None falls back to ``DIOGENES_OBS_HISTORY`` env
      var, then to :data:`~diogenes_mjlab.constants.OBS_HISTORY_LENGTH`).
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
  if contact_phase_term is None:
    contact_phase_term = _env_bool("DIOGENES_CONTACT_PHASE_TERM")
  if contact_phase_term is None:
    contact_phase_term = CONTACT_PHASE_TERM_ENABLED
  if obs_history is None:
    obs_history = _env_int("DIOGENES_OBS_HISTORY")
  if obs_history is None:
    obs_history = OBS_HISTORY_LENGTH

  # Build rewards and recover the phase-clock period for the chosen trajectory.
  rewards, phase_period = _build_rewards(trajectory)
  if not contact_phase_term:
    del rewards["contact_phase_termination"]

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
    # The ground is the scene TerrainEntity plane (geom named "terrain",
    # attached with an empty prefix), not a robot-bundled floor. entity unset
    # => the pattern is taken as a literal MuJoCo geom name.
    secondary=ContactMatch(mode="geom", pattern="terrain"),
    fields=("found", "force", "pos"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # The TerrainEntity plane is the single shared ground AND the consumer of
  # env_spacing: the scene copies num_envs/env_spacing into it so it lays the
  # per-env origins out on a grid.  Without it env_origins stay all-zero and the
  # (fixed-base, mocap-wrapped) robots all stack at the world origin.
  scene_cfg = SceneCfg(
    num_envs=4096,
    env_spacing=2.0,
    terrain=TerrainEntityCfg(terrain_type="plane"),
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
  # The actor group is built WITHOUT privileged slider state; the critic group
  # includes it.  See config/observations.py for the canonical term lists.
  # ---------------------------------------------------------------------------
  actor_terms = _actor_terms(obs_noise=obs_noise)
  actor_terms["phase_clock"].params["hop_period"] = phase_period

  critic_terms = _critic_terms()
  critic_terms["phase_clock"].params["hop_period"] = phase_period

  # Guard: enforce the privileged-info split — actor must NOT contain slider
  # terms; critic MUST contain them.  This assertion is intentionally cheap
  # (pure dict-key check) so it fires at config-build time with no overhead
  # during rollout.
  _actor_keys = set(actor_terms.keys())
  _critic_keys = set(critic_terms.keys())
  for _priv in PRIVILEGED_OBS_TERMS:
    assert _priv not in _actor_keys, (
      f"Privileged obs term {_priv!r} must NOT appear in the actor group. "
      "Check config/observations.py _actor_terms()."
    )
    assert _priv in _critic_keys, (
      f"Privileged obs term {_priv!r} must appear in the critic group. "
      "Check config/observations.py _critic_terms()."
    )

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=obs_history or None,
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

  # Position each env's (fixed-base, mocap-wrapped) robot at its grid origin on
  # reset.  This writes mocap_pose = default_root_state + env_origins for the
  # robot; it touches only the root pose, never the joints, so it composes with
  # the random reset_joint_pose term above.  Without this the env_spacing grid
  # exists but nothing ever moves the robots onto it.  zero pose_range => no
  # extra jitter; asset_cfg defaults to the "robot" entity.
  events["reset_base_pose"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={"pose_range": {}},
  )

  # ---------------------------------------------------------------------------
  # Terminations.
  # ---------------------------------------------------------------------------
  # Terminate when the foot/ground contact state is wrong for the current phase.
  # The dual-parabola task has an explicit flight arc (foot must be airborne),
  # so it uses the phase-keyed check with a tolerance band around each
  # liftoff/landing transition; the sine squat keeps the foot planted at all
  # times, so any loss of contact ends the episode.  The dedicated negative
  # reward is wired in config/rewards.py via CONTACT_PHASE_TERM_NAME.
  if trajectory in ("dual_parabola", "spring"):
    # Both flight trajectories have a real airborne arc, so use the phase-keyed
    # check (with a tolerance band around each liftoff/landing transition). The
    # spring shares the dual-parabola flight arc but has its own derived period.
    contact_phase_wrong_fn = (
      diogenes_mdp.foot_contact_phase_wrong_dual_parabola
      if trajectory == "dual_parabola"
      else diogenes_mdp.foot_contact_phase_wrong_spring
    )
    contact_phase_termination = TerminationTermCfg(
      func=contact_phase_wrong_fn,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "traj_min": TRAJ_MIN,
        "traj_max": TRAJ_MAX,
        "traj_transition": TRAJ_TRANSITION,
        "gravity": GRAVITY,
        "phase_margin": CONTACT_PHASE_MARGIN,
      },
    )
  else:  # "sine"
    contact_phase_termination = TerminationTermCfg(
      func=diogenes_mdp.foot_not_in_contact,
      params={"sensor_name": FOOT_CONTACT_SENSOR},
    )

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "joint_at_limit": TerminationTermCfg(
      func=diogenes_mdp.joint_at_limit,
      params={
        "asset_cfg": actuated_joints_cfg(),
        "margin": JOINT_LIMIT_MARGIN,
      },
    ),
    **(
      {CONTACT_PHASE_TERM_NAME: contact_phase_termination}
      if contact_phase_term
      else {}
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

  # Point the viewer at the robot with a TRACKING camera.  The robot is
  # fixed-base (mocap-wrapped), so the default AUTO/WORLD origin has no moving
  # root body to follow and falls back to a free camera staring at the world
  # origin (0,0,0).  Once env_spacing places each env on a grid, env_idx 0 sits
  # away from the origin, so a world-origin camera would frame empty ground.
  # ASSET_BODY tracks base_link wherever its env cell is, in both the live
  # viewer and the offscreen video recorder.
  cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
  cfg.viewer.entity_name = "robot"
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -20.0
  cfg.viewer.azimuth = 135.0


  # ---------------------------------------------------------------------------
  # Play-mode overrides.
  # ---------------------------------------------------------------------------
  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
