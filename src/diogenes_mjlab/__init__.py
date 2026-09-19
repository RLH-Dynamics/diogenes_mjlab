from mjlab.tasks.registry import register_mjlab_task

from .config import diogenes_env_cfg
from .harold_biped.env_cfg import harold_suspended_env_cfg
from .harold_biped.rl_cfg import (
  harold_suspended_ppo_runner_cfg,
  harold_walk_ppo_runner_cfg,
)
from .harold_biped.walk_env_cfg import harold_walk_env_cfg
from .rl_cfg import diogenes_ppo_runner_cfg

# Dual-parabolic (highly dynamic) trajectory -- the original task.
register_mjlab_task(
  task_id="Diogenes-Flat",
  env_cfg=diogenes_env_cfg(trajectory="dual_parabola"),
  play_env_cfg=diogenes_env_cfg(trajectory="dual_parabola", play=True),
  rl_cfg=diogenes_ppo_runner_cfg(),
)

# Sinusoidal (smooth, gentle) trajectory -- easier first sim-to-real target.
# Same rewards/regularizers/terminations as Diogenes-Flat; only the slider
# reference and the phase-clock period differ (see env_cfgs.diogenes_env_cfg).
register_mjlab_task(
  task_id="Diogenes-Flat-Sine",
  env_cfg=diogenes_env_cfg(trajectory="sine"),
  play_env_cfg=diogenes_env_cfg(trajectory="sine", play=True),
  rl_cfg=diogenes_ppo_runner_cfg(),
)

# Spring hop -- dual-parabola free-fall flight arc joined to a Hooke's-law spring
# stance (acceleration grows with compression). Same rewards/regularizers/
# terminations family as Diogenes-Flat; only the slider reference, the derived
# phase-clock period and the spring contact-phase check differ.
register_mjlab_task(
  task_id="Diogenes-Flat-Spring",
  env_cfg=diogenes_env_cfg(trajectory="spring"),
  play_env_cfg=diogenes_env_cfg(trajectory="spring", play=True),
  rl_cfg=diogenes_ppo_runner_cfg(),
)

# Harold biped, torso welded 1 m up: each foot tracks a straight stance line then
# a half-sine swing, legs 180 deg out of phase (see harold_biped/gait.py).
register_mjlab_task(
  task_id="Diogenes-Biped-Suspended",
  env_cfg=harold_suspended_env_cfg(),
  play_env_cfg=harold_suspended_env_cfg(play=True),
  rl_cfg=harold_suspended_ppo_runner_cfg(),
)

# Harold biped walking on flat ground: forward-only velocity commands, no
# reference gait (see harold_biped/walk_env_cfg.py).
register_mjlab_task(
  task_id="Diogenes-Biped-Walk",
  env_cfg=harold_walk_env_cfg(),
  play_env_cfg=harold_walk_env_cfg(play=True),
  rl_cfg=harold_walk_ppo_runner_cfg(),
)
