"""Thin backward-compatibility shim for diogenes_mjlab.env_cfgs.

The real implementation has moved into the ``config`` package. This shim
re-exports every public name so existing imports of the form::

    from diogenes_mjlab.env_cfgs import diogenes_env_cfg
    from diogenes_mjlab import env_cfgs; env_cfgs.diogenes_env_cfg

continue to work unchanged.
"""

from .config import diogenes_env_cfg, TrajectoryType  # noqa: F401
from .config.rewards import TRAJ_T  # noqa: F401
from .constants import SINE_PERIOD  # noqa: F401
