from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import diogenes_env_cfg
from .rl_cfg import diogenes_ppo_runner_cfg

register_mjlab_task(
    task_id="Diogenes-Flat",
    env_cfg=diogenes_env_cfg(),
    play_env_cfg=diogenes_env_cfg(play=True),
    rl_cfg=diogenes_ppo_runner_cfg(),
)
