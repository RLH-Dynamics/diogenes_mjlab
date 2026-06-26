"""Custom MDP terms for the Diogenes periodic-hopping task.

This package replaces the old ``mdp.py`` module. All public names are
re-exported here so existing access paths like ``diogenes_mdp.foo`` and
``from . import mdp as diogenes_mdp; diogenes_mdp.GRAVITY`` continue to work
unchanged.

Sub-modules by concern
----------------------
* trajectories  -- dual_parabola_timing + reference maths
* rewards       -- reward functions
* observations  -- phase clock, slider/proprio observations
* terminations  -- termination functions
* events        -- reset-pose / domain-randomization event functions
"""

from __future__ import annotations

from mjlab.managers.scene_entity_config import SceneEntityCfg

# Re-export GRAVITY so ``diogenes_mdp.GRAVITY`` keeps working (env_cfgs uses it).
from ..constants import GRAVITY  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level singleton (one per process) shared as default arg.
# ---------------------------------------------------------------------------
SLIDER_CFG = SceneEntityCfg("robot", joint_names=("slider",))

# ---------------------------------------------------------------------------
# Trajectory maths.
# ---------------------------------------------------------------------------
from .trajectories import (  # noqa: F401, E402
  dual_parabola_timing,
  dual_parabola_reference,
  dual_parabola_velocity,
  dual_parabola_acceleration,
  spring_timing,
  spring_reference,
  spring_velocity,
  spring_acceleration,
)

# ---------------------------------------------------------------------------
# Observations.
# ---------------------------------------------------------------------------
from .observations import (  # noqa: F401, E402
  _phase,
  phase_clock,
  _height_above_start,
  slider_pos,
  slider_vel,
)

# ---------------------------------------------------------------------------
# Rewards.
# ---------------------------------------------------------------------------
from .rewards import (  # noqa: F401, E402
  FOOT_OFFSET_B,
  DEFAULT_FOOT_REF_XY,
  lateral_contact_force_l2,
  vertical_contact_force_l2,
  foot_slip,
  foot_contact_required,
  foot_contact_phase_dual_parabola,
  foot_contact_phase_spring,
  is_specific_termination,
  slider_dual_parabola_tracking,
  slider_dual_parabola_velocity_tracking,
  slider_dual_parabola_acceleration_tracking,
  slider_spring_tracking,
  slider_spring_velocity_tracking,
  slider_spring_acceleration_tracking,
  slider_sinusoid_tracking,
  slider_sinusoid_velocity_tracking,
  slider_sinusoid_acceleration_tracking,
  foot_xy_position_tracking,
)

# ---------------------------------------------------------------------------
# Terminations.
# ---------------------------------------------------------------------------
from .terminations import (  # noqa: F401, E402
  joint_at_limit,
  foot_contact_phase_wrong_dual_parabola,
  foot_contact_phase_wrong_spring,
  foot_not_in_contact,
)

# ---------------------------------------------------------------------------
# Events.
# ---------------------------------------------------------------------------
from .events import (  # noqa: F401, E402
  reset_joints_uniform_legal,
)
