"""Actor/critic observation-group builders for the Diogenes hop stand."""

from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg

from ..constants import (
  OBS_DELAY_MAX_LAG,
  OBS_DELAY_MIN_LAG,
  OBS_NOISE_JOINT_POS,
  OBS_NOISE_JOINT_VEL,
  OBS_NOISE_SLIDER_POS,
  OBS_NOISE_SLIDER_VEL,
)
from .. import mdp as diogenes_mdp
from .entities import actuated_joints_cfg, slider_cfg

# Sentinel replaced per-call inside diogenes_env_cfg (the phase period depends on
# the chosen trajectory, resolved at build time).
_PHASE_PERIOD_PLACEHOLDER = 0.6

# ---------------------------------------------------------------------------
# Public term-name lists that document the privileged-info split.
# These are the canonical ordering; callers that need to verify the split can
# import these lists instead of repeating the string literals.
# ---------------------------------------------------------------------------

#: Observation term names present in the ACTOR group (no privileged slider state).
ACTOR_OBS_TERMS = ("joint_pos", "joint_vel", "last_action", "phase_clock")

#: Observation term names present in the CRITIC group (includes slider state).
CRITIC_OBS_TERMS = ("joint_pos", "joint_vel", "slider_pos", "slider_vel", "last_action", "phase_clock")

#: Names of the privileged (slider) terms excluded from the actor.
PRIVILEGED_OBS_TERMS = ("slider_pos", "slider_vel")


def _actor_terms(obs_noise: bool) -> dict[str, ObservationTermCfg]:
  """Build the proprioceptive observation terms for the ACTOR group.

  The actor does NOT receive slider (carriage) state — that is privileged
  information available only to the critic (asymmetric actor-critic).

  Term order: joint_pos, joint_vel, last_action, phase_clock.

  Args:
    obs_noise: when True, attach additive uniform sensor noise AND a per-term
      observation delay (up to ``OBS_DELAY_MAX_LAG`` control steps) to each
      term. Pass False for a noise-free actor (play/eval mode).

  Returns a fresh dict on every call (delay buffers are per-term instance).
  """
  jp_noise = UniformNoiseCfg(
    n_min=-OBS_NOISE_JOINT_POS, n_max=OBS_NOISE_JOINT_POS
  ) if obs_noise else None
  jv_noise = UniformNoiseCfg(
    n_min=-OBS_NOISE_JOINT_VEL, n_max=OBS_NOISE_JOINT_VEL
  ) if obs_noise else None

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
    "last_action": ObservationTermCfg(
      func=mdp.last_action,
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    "phase_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock,
      params={"hop_period": _PHASE_PERIOD_PLACEHOLDER},
    ),
  }


def _critic_terms() -> dict[str, ObservationTermCfg]:
  """Build the proprioceptive observation terms for the CRITIC group.

  The critic receives the full state including privileged slider (carriage)
  position and velocity. No noise or delay is applied to the critic.

  Term order: joint_pos, joint_vel, slider_pos, slider_vel, last_action,
  phase_clock.

  Returns a fresh dict on every call (delay buffers are per-term instance).
  """
  return {
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
    "last_action": ObservationTermCfg(
      func=mdp.last_action,
    ),
    "phase_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock,
      params={"hop_period": _PHASE_PERIOD_PLACEHOLDER},
    ),
  }


def _proprio_terms(obs_noise: bool) -> dict[str, ObservationTermCfg]:
  """Build the proprioceptive observation terms shared by actor and critic.

  .. deprecated::
    Use :func:`_actor_terms` and :func:`_critic_terms` directly.  This shim
    is kept for call-site compatibility while :mod:`config.env` is updated.

  Args:
    obs_noise: when True, attach additive uniform sensor noise AND a per-term
      observation delay (up to ``OBS_DELAY_MAX_LAG`` control steps) to each
      term. Intended for the ACTOR group only; pass False for the critic.

  The two groups must NOT share term instances (the delay buffer is per-term),
  so this returns a fresh dict on every call.
  """
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
      delay_min_lag=dmin,
      delay_max_lag=dmax,
    ),
    "phase_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock,
      params={"hop_period": _PHASE_PERIOD_PLACEHOLDER},
    ),
  }
