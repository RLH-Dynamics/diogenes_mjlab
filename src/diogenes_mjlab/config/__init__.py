"""Diogenes environment configuration package.

Re-exports the public entry point so ``from .config import diogenes_env_cfg``
and ``from .config.env import diogenes_env_cfg`` both work.
"""

from .env import diogenes_env_cfg, TrajectoryType  # noqa: F401
