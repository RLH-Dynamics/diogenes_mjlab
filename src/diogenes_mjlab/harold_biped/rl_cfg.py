"""RL configuration for the Harold biped tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..rl_cfg import diogenes_ppo_runner_cfg


def harold_suspended_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same PPO setup as the hopping leg, logged under its own experiment name."""
  cfg = diogenes_ppo_runner_cfg()
  cfg.experiment_name = "harold_biped_suspended"
  return cfg


def harold_walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Leg PPO setup with a longer run: balancing and walking take far more
  iterations than tracking a fixed trajectory. Starting value, untuned."""
  cfg = diogenes_ppo_runner_cfg()
  cfg.experiment_name = "harold_biped_walk"
  cfg.max_iterations = 3000
  return cfg
