"""Domain-randomization event builders for the Diogenes hop stand (sim-to-real)."""

from dataclasses import dataclass
from typing import Tuple

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..constants import DIOGENES_ACTUATOR_NAMES, JOINT_LIMIT_MARGIN
from .. import mdp as diogenes_mdp
from .entities import (
  actuated_joints_cfg,
  all_links_cfg,
  foot_geom_cfg,
  slider_cfg,
)


@dataclass(frozen=True)
class DomainRandRanges:
  """Tuneable range literals for each domain-randomisation event.

  Each pair ``(lo, hi)`` is the *nominal* uniform range (before any
  ``dr_scale`` widening).  The default values exactly reproduce the
  hand-tuned baseline ranges used during training.
  """

  # PD-gain multiplicative scale: (lo, hi) applied to both kp and kd.
  pd_gains_kp: Tuple[float, float] = (0.89, 1.12)
  pd_gains_kd: Tuple[float, float] = (0.89, 1.12)

  # Pseudo-inertia alpha (fractional mass/inertia perturbation).
  inertia_alpha: Tuple[float, float] = (-0.11, 0.09)

  # Centre-of-mass offset per axis (metres).
  com_offset_x: Tuple[float, float] = (-0.025, 0.025)
  com_offset_y: Tuple[float, float] = (-0.025, 0.025)
  com_offset_z: Tuple[float, float] = (-0.025, 0.025)

  # Joint armature (reflected rotor inertia, kg·m²).
  joint_armature: Tuple[float, float] = (0.016, 0.024)

  # Joint dry friction (frictionloss, N·m).
  joint_friction: Tuple[float, float] = (0.15, 1.60)

  # Foot/ground tangential friction coefficient.
  foot_friction: Tuple[float, float] = (0.3, 1.2)

  # Slider (rail) Coulomb dry friction (Newtons).
  slider_friction: Tuple[float, float] = (1.0, 3.5)

  # Joint encoder bias (rad).
  encoder_bias: Tuple[float, float] = (-0.015, 0.015)

  # Reset start-pose: joint velocity range (m/s or rad/s).
  reset_velocity: Tuple[float, float] = (0.0, 0.0)


# Default ranges instance used by _domain_randomization_events when no
# override is passed.
DEFAULT_DR_RANGES = DomainRandRanges()

# ---------------------------------------------------------------------------
# Per-term enable switches for domain randomization.
#
# Flip any of these to False to drop that single DR term while leaving the rest
# active -- handy for isolating which term destabilizes a policy.
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


def _domain_randomization_events(
  dr_scale: float,
  reset_joint_pose: bool = True,
  ranges: DomainRandRanges = DEFAULT_DR_RANGES,
) -> dict[str, EventTermCfg]:
  """Build the domain-randomization event terms (startup + reset).

  The dict is assembled conditionally: each term is included only if its
  ``ENABLE_*`` module-level switch above is True.

  Args:
    dr_scale: multiplier on every STARTUP randomization range's half-width
      (1.0 = the nominal ranges). Does not affect the reset-pose term.
    reset_joint_pose: include the random LEGAL start-pose reset term.
    ranges: per-term range literals.  Defaults to :data:`DEFAULT_DR_RANGES`
      which reproduces the baseline training values exactly.

  Returns:
    A dict of ``EventTermCfg`` keyed by a short term name, ready to pass as
    ``cfg.events``. Only the enabled terms are present.
  """
  s = dr_scale
  events: dict[str, EventTermCfg] = {}

  # --- PD gains (the single most important actuator DR for sim-to-real).
  if ENABLE_PD_GAINS:
    events["pd_gains"] = EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", actuator_names=DIOGENES_ACTUATOR_NAMES
        ),
        "kp_range": _scale_range(*ranges.pd_gains_kp, s),
        "kd_range": _scale_range(*ranges.pd_gains_kd, s),
        "operation": "scale",
        "distribution": "uniform",
      },
    )

  # --- Mass + inertia, physics-consistent.
  if ENABLE_LINK_INERTIAL:
    events["link_inertial"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": all_links_cfg(),
        "alpha_range": _scale_range(*ranges.inertia_alpha, s),
        "distribution": "uniform",
      },
    )

  # --- Centre-of-mass offset, +-25 mm per axis.
  if ENABLE_COM_OFFSET:
    events["com_offset"] = EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": all_links_cfg(),
        "ranges": {
          0: _scale_range(*ranges.com_offset_x, s),
          1: _scale_range(*ranges.com_offset_y, s),
          2: _scale_range(*ranges.com_offset_z, s),
        },
        "operation": "add",
        "distribution": "uniform",
      },
    )

  # --- Joint armature (reflected rotor inertia).
  if ENABLE_JOINT_ARMATURE:
    events["joint_armature"] = EventTermCfg(
      func=dr.joint_armature,
      mode="startup",
      params={
        "asset_cfg": actuated_joints_cfg(),
        "ranges": _scale_range(*ranges.joint_armature, s),
        "operation": "abs",
        "distribution": "uniform",
      },
    )

  # --- Joint dry friction (frictionloss).
  if ENABLE_JOINT_FRICTION:
    events["joint_friction"] = EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={
        "asset_cfg": actuated_joints_cfg(),
        "ranges": _scale_range(*ranges.joint_friction, s),
        "operation": "abs",
        "distribution": "uniform",
      },
    )

  # --- Foot/ground tangential friction.
  if ENABLE_FOOT_FRICTION:
    events["foot_friction"] = EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": foot_geom_cfg(),
        "ranges": _scale_range(*ranges.foot_friction, s),
        "operation": "abs",
        "distribution": "uniform",
        "shared_random": True,
      },
    )

  # --- Slider (rail) Coulomb dry-friction (Newtons, not N·m).
  if ENABLE_SLIDER_FRICTION:
    events["slider_friction"] = EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={
        "asset_cfg": slider_cfg(),
        "ranges": _scale_range(*ranges.slider_friction, s),
        "operation": "abs",
        "distribution": "uniform",
      },
    )

  # --- Joint encoder bias, +-0.015 rad.
  if ENABLE_ENCODER_BIAS:
    events["encoder_bias"] = EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={
        "asset_cfg": actuated_joints_cfg(),
        "bias_range": _scale_range(*ranges.encoder_bias, s),
      },
    )

  # --- Random LEGAL joint start pose (mode="reset").
  if reset_joint_pose and ENABLE_RESET_JOINT_POSE:
    events["reset_joint_pose"] = EventTermCfg(
      func=diogenes_mdp.reset_joints_uniform_legal,
      mode="reset",
      params={
        "asset_cfg": actuated_joints_cfg(),
        "margin": JOINT_LIMIT_MARGIN,  # lockstep with joint_at_limit
        "safety_eps": 1e-3,
        "velocity_range": ranges.reset_velocity,  # start at rest
      },
    )

  return events
