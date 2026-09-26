"""Environment config for the Harold biped suspended-gait task.

Registered as ``Diogenes-Biped-Suspended``. The torso is welded to the world
1 m above the ground (no freejoint) and each leg is rewarded for tracking the
foot reference loop defined in ``gait.py``: a straight stance line followed by
a half-sine swing, left and right 180 deg out of phase.

Follows the structure of the hopping-leg config (``config/env.py``) and reuses
its hardware-level settings: observation noise/delay, joint-limit margin and the
domain-randomization ranges validated on the real leg (same RS03 actuators).
"""

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg

from ..config.domain_rand import DomainRandRanges, _scale_range
from ..constants import (
  JOINT_LIMIT_MARGIN,
  OBS_DELAY_MAX_LAG,
  OBS_DELAY_MIN_LAG,
  OBS_HISTORY_LENGTH,
  OBS_NOISE_JOINT_POS,
  OBS_NOISE_JOINT_VEL,
)
from ..flags import _env_bool, _env_float, _env_int
from .. import mdp as diogenes_mdp
from . import gait
from . import mdp as harold_mdp
from .harold_biped_constants import (
  HAROLD_JOINT_NAMES,
  ROOT_BODY_NAME,
  get_harold_biped_cfg,
)

#: Critic-only (privileged) observation terms.
PRIVILEGED_OBS_TERMS = ("feet_pos", "feet_ref_pos")

#: DR ranges for the biped. The leg's baseline (``DEFAULT_DR_RANGES``) is left
#: untouched -- it was validated against the real hop stand and the snapshot
#: test pins it. Two deviations, both aimed at the first biped transfer:
#:
#:   * PD gains +-30% instead of +-11%. The real controller's kp/kv are a free
#:     choice here, so the policy should tolerate a wide band rather than one
#:     narrow guess, and the width tells us how much gain mismatch it survives.
#:   * Effort limit 0.7-1.0 of the model's +-60 N.m RS03 forcerange.
HAROLD_DR_RANGES = DomainRandRanges(
  pd_gains_kp=(0.7, 1.3),
  pd_gains_kd=(0.7, 1.3),
)


# ---------------------------------------------------------------------------
# Scene entity selectors (each manager term gets its own instance).
# ---------------------------------------------------------------------------


def joints_cfg() -> SceneEntityCfg:
  """The six actuated leg joints, in HAROLD_JOINT_NAMES order."""
  return SceneEntityCfg("robot", joint_names=HAROLD_JOINT_NAMES, preserve_order=True)


def actuators_cfg() -> SceneEntityCfg:
  """The six position actuators."""
  return SceneEntityCfg("robot", actuator_names=HAROLD_JOINT_NAMES)


#: The moving leg links.
LEG_LINK_NAMES: tuple[str, ...] = (
  "l_hip_assy", "l_thigh_assy", "l_calf_assy",
  "r_hip_assy", "r_thigh_assy", "r_calf_assy",
)


def leg_links_cfg() -> SceneEntityCfg:
  """The moving leg links (the welded torso is excluded from inertial DR)."""
  return SceneEntityCfg("robot", body_names=LEG_LINK_NAMES)


def feet_cfg() -> SceneEntityCfg:
  """The foot sites, in gait.FOOT_SITE_NAMES order."""
  return SceneEntityCfg(
    "robot", site_names=gait.FOOT_SITE_NAMES, preserve_order=True
  )


# ---------------------------------------------------------------------------
# Rewards.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuspendedRewardWeights:
  """Per-term reward weights. Starting values, not yet tuned in training."""

  feet_position: float = 10.0
  feet_velocity: float = 2.0
  electrical_power: float = -0.0005
  torque: float = -0.002
  # -1.0 beat -4.0 on every axis in a controlled A/B (reward 128 vs 6, episode
  # length 567 vs 58, foot error 6 mm vs 36 mm): the stronger penalty suppressed
  # motion before the policy learned the gait. A policy that tracks a smooth
  # reference has a low action rate for free.
  action_rate: float = -1.0
  joint_limits: float = -1.0
  termination_penalty: float = -100.0


DEFAULT_REWARD_WEIGHTS = SuspendedRewardWeights()


def _build_rewards(
  weights: SuspendedRewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> dict[str, RewardTermCfg]:
  return {
    "feet_position": RewardTermCfg(
      func=harold_mdp.feet_position_tracking,
      weight=weights.feet_position,
      # The zero pose sits 0.05 m below the stance line, so a 0.05 m band keeps
      # the term live from the first step.
      params={"asset_cfg": feet_cfg(), "std": 0.05},
    ),
    "feet_velocity": RewardTermCfg(
      func=harold_mdp.feet_velocity_tracking,
      weight=weights.feet_velocity,
      # Reference speeds peak at ~0.16 m/s.
      params={"asset_cfg": feet_cfg(), "std": 0.2},
    ),
    "electrical_power": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=weights.electrical_power,
      params={"asset_cfg": joints_cfg()},
    ),
    "torque": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=weights.torque,
      params={"asset_cfg": actuators_cfg()},
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=weights.action_rate,
    ),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=weights.joint_limits,
      params={"asset_cfg": joints_cfg()},
    ),
    "termination_penalty": RewardTermCfg(
      func=mdp.is_terminated,
      weight=weights.termination_penalty,
    ),
  }


# ---------------------------------------------------------------------------
# Observations.
# ---------------------------------------------------------------------------


def _actor_terms(obs_noise: bool) -> dict[str, ObservationTermCfg]:
  """Proprioception + gait clock; noise and delay when obs_noise is on."""
  jp_noise = (
    UniformNoiseCfg(n_min=-OBS_NOISE_JOINT_POS, n_max=OBS_NOISE_JOINT_POS)
    if obs_noise else None
  )
  jv_noise = (
    UniformNoiseCfg(n_min=-OBS_NOISE_JOINT_VEL, n_max=OBS_NOISE_JOINT_VEL)
    if obs_noise else None
  )
  dmin = OBS_DELAY_MIN_LAG if obs_noise else 0
  dmax = OBS_DELAY_MAX_LAG if obs_noise else 0

  return {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": joints_cfg()},
      noise=jp_noise,
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": joints_cfg()},
      noise=jv_noise,
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    "last_action": ObservationTermCfg(
      func=mdp.last_action,
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    "gait_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock,
      params={"hop_period": gait.GAIT_PERIOD},
    ),
  }


def _critic_terms() -> dict[str, ObservationTermCfg]:
  """Clean proprioception + gait clock + privileged foot state and reference."""
  return {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"asset_cfg": joints_cfg()}
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, params={"asset_cfg": joints_cfg()}
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action),
    "gait_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock,
      params={"hop_period": gait.GAIT_PERIOD},
    ),
    "feet_pos": ObservationTermCfg(
      func=harold_mdp.feet_pos_b, params={"asset_cfg": feet_cfg()}
    ),
    "feet_ref_pos": ObservationTermCfg(func=harold_mdp.feet_ref_pos_b),
  }


# ---------------------------------------------------------------------------
# Events.
# ---------------------------------------------------------------------------


def _domain_randomization_events(
  dr_scale: float,
  reset_joint_pose: bool,
  ranges: DomainRandRanges = HAROLD_DR_RANGES,
  inertial_body_names: tuple[str, ...] = LEG_LINK_NAMES,
  mode: str = "reset",
) -> dict[str, EventTermCfg]:
  """Leg-validated DR terms that apply to the biped.

  Foot friction and slider friction are omitted: the suspended feet never touch
  the ground and there is no slider (the walking task adds foot friction).

  Args:
    inertial_body_names: bodies whose mass/inertia and COM are randomized. The
      suspended task excludes the welded torso; walking includes it.
    mode: when the physical parameters are resampled. ``"reset"`` draws a fresh
      robot every episode; ``"startup"`` draws once, which pins each env to one
      fixed robot for the whole run and lets a policy with observation history
      identify its env and specialise to it.
  """
  s = dr_scale
  events = {
    "pd_gains": EventTermCfg(
      func=dr.pd_gains,
      mode=mode,
      params={
        "asset_cfg": actuators_cfg(),
        "kp_range": _scale_range(*ranges.pd_gains_kp, s),
        "kd_range": _scale_range(*ranges.pd_gains_kd, s),
        "operation": "scale",
        "distribution": "uniform",
      },
    ),
    "link_inertial": EventTermCfg(
      func=dr.pseudo_inertia,
      mode=mode,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=inertial_body_names),
        "alpha_range": _scale_range(*ranges.inertia_alpha, s),
        "distribution": "uniform",
      },
    ),
    "com_offset": EventTermCfg(
      func=dr.body_com_offset,
      mode=mode,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=inertial_body_names),
        "ranges": {
          0: _scale_range(*ranges.com_offset_x, s),
          1: _scale_range(*ranges.com_offset_y, s),
          2: _scale_range(*ranges.com_offset_z, s),
        },
        "operation": "add",
        "distribution": "uniform",
      },
    ),
    "joint_armature": EventTermCfg(
      func=dr.joint_armature,
      mode=mode,
      params={
        "asset_cfg": joints_cfg(),
        "ranges": _scale_range(*ranges.joint_armature, s),
        "operation": "abs",
        "distribution": "uniform",
      },
    ),
    "joint_friction": EventTermCfg(
      func=dr.joint_friction,
      mode=mode,
      params={
        "asset_cfg": joints_cfg(),
        "ranges": _scale_range(*ranges.joint_friction, s),
        "operation": "abs",
        "distribution": "uniform",
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode=mode,
      params={
        "asset_cfg": joints_cfg(),
        "bias_range": _scale_range(*ranges.encoder_bias, s),
      },
    ),
    # Scales the model's +-60 N.m RS03 forcerange. Without a forcerange in the
    # XML the position law is torque-unbounded and a policy will happily learn
    # a transient the real actuator cannot deliver.
    "effort_limit": EventTermCfg(
      func=dr.effort_limits,
      mode=mode,
      params={
        "asset_cfg": actuators_cfg(),
        "effort_limit_range": _scale_range(*ranges.effort_limit, s),
        "operation": "scale",
        "distribution": "uniform",
      },
    ),
  }
  if reset_joint_pose:
    events["reset_joint_pose"] = EventTermCfg(
      func=diogenes_mdp.reset_joints_near_default,
      mode="reset",
      params={
        "asset_cfg": joints_cfg(),
        "noise_range": ranges.reset_joint_noise,
        # The gait's home pose, not the all-zero default (see gait.py).
        "nominal_pos": tuple(
          gait.nominal_joint_pos()[j] for j in HAROLD_JOINT_NAMES
        ),
        "margin": JOINT_LIMIT_MARGIN,  # lockstep with joint_at_limit
        "safety_eps": 1e-3,
        "velocity_range": ranges.reset_velocity,
      },
    )
  return events


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


def harold_suspended_env_cfg(
  play: bool = False,
  domain_rand: bool | None = None,
  obs_noise: bool | None = None,
  dr_scale: float | None = None,
  reset_joints: bool | None = None,
  obs_history: int | None = None,
  env_spacing: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the suspended-biped gait-tracking environment configuration.

  Flags resolve exactly as in ``diogenes_env_cfg``: explicit argument, then the
  matching ``DIOGENES_*`` env var, then the play-based default.

  Args:
    play: evaluation overrides (1 env, effectively infinite episode; DR, noise
      and random reset default OFF, but their flags below turn them on to
      preview the policy with real-robot errors).
    domain_rand: startup DR events (``DIOGENES_DOMAIN_RAND``; default not play).
    obs_noise: actor sensor noise + delay (``DIOGENES_OBS_NOISE``; default not play).
    dr_scale: DR range half-width multiplier (``DIOGENES_DR_SCALE``; default 1.0).
    reset_joints: random legal start pose (``DIOGENES_RESET_JOINTS``; default
      not play).
    obs_history: past actor timesteps (``DIOGENES_OBS_HISTORY``; default
      OBS_HISTORY_LENGTH).
    env_spacing: grid spacing between envs in m (``DIOGENES_ENV_SPACING``;
      default 2.0). Visual only: the welded robots never interact.
  """
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
  if obs_history is None:
    obs_history = _env_int("DIOGENES_OBS_HISTORY")
  if obs_history is None:
    obs_history = OBS_HISTORY_LENGTH
  if env_spacing is None:
    env_spacing = _env_float("DIOGENES_ENV_SPACING")
  if env_spacing is None:
    env_spacing = 2.0

  # 2 ms physics step * decimation 10 -> 50 Hz control, as for the leg.
  sim_cfg = SimulationCfg(
    njmax=64,
    nconmax=16,
    mujoco=MujocoCfg(timestep=0.002),
  )

  # The terrain plane lays out the per-env grid (env_spacing). The suspended
  # feet stay ~0.6 m above it, so it never generates contacts.
  scene_cfg = SceneCfg(
    num_envs=4096,
    env_spacing=env_spacing,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_harold_biped_cfg(fixed_base=True)},
  )

  joint_pos_action = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=HAROLD_JOINT_NAMES,
    scale=1.0,
    use_default_offset=True,
  )

  actor_terms = _actor_terms(obs_noise=obs_noise)
  critic_terms = _critic_terms()
  for name in PRIVILEGED_OBS_TERMS:
    assert name not in actor_terms, f"Privileged obs term {name!r} leaked into actor."
    assert name in critic_terms, f"Privileged obs term {name!r} missing from critic."

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      # Follows obs_noise, so DIOGENES_OBS_NOISE=1 also corrupts play observations.
      enable_corruption=obs_noise,
      history_length=obs_history or None,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  events = (
    _domain_randomization_events(dr_scale, reset_joint_pose=reset_joints)
    if domain_rand
    else {}
  )
  # Place each env's welded (mocap) robot on its grid cell; root pose only.
  events["reset_base_pose"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={"pose_range": {}},
  )

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "joint_at_limit": TerminationTermCfg(
      func=diogenes_mdp.joint_at_limit,
      params={"asset_cfg": joints_cfg(), "margin": JOINT_LIMIT_MARGIN},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=20.0,
    sim=sim_cfg,
    scene=scene_cfg,
    observations=observations,
    actions={"joint_pos": joint_pos_action},
    events=events,
    rewards=_build_rewards(),
    terminations=terminations,
  )

  # Track the torso so env 0's grid cell stays in frame (see config/env.py).
  cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
  cfg.viewer.entity_name = "robot"
  cfg.viewer.body_name = ROOT_BODY_NAME
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0
  cfg.viewer.azimuth = 45.0

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)

  return cfg
