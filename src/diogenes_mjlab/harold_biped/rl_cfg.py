"""RL configuration for the Harold biped tasks."""

from dataclasses import dataclass

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from ..flags import _env_bool
from ..rl_cfg import diogenes_ppo_runner_cfg

#: rsl-rl data-augmentation function for the walking task's mirror symmetry.
MIRROR_FUNC = "diogenes_mjlab.harold_biped.symmetry:mirror_obs_and_actions"


@dataclass
class SymmetricPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """mjlab's PPO config plus rsl-rl's ``symmetry_cfg`` (None = off)."""

  symmetry_cfg: dict[str, str | bool | float] | None = None


def harold_suspended_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same PPO setup as the hopping leg, logged under its own experiment name."""
  cfg = diogenes_ppo_runner_cfg()
  cfg.experiment_name = "harold_biped_suspended"
  return cfg


def harold_walk_ppo_runner_cfg(symmetry: bool | None = None) -> RslRlOnPolicyRunnerCfg:
  """Leg PPO setup with a longer run (balancing and walking take far more
  iterations than tracking a fixed trajectory) and left/right mirror data
  augmentation (``DIOGENES_SYMMETRY``, default on; see symmetry.py)."""
  if symmetry is None:
    symmetry = _env_bool("DIOGENES_SYMMETRY")
  if symmetry is None:
    symmetry = True
  cfg = diogenes_ppo_runner_cfg()
  cfg.experiment_name = "harold_biped_walk"
  cfg.max_iterations = 3000
  base = vars(cfg.algorithm)
  cfg.algorithm = SymmetricPpoAlgorithmCfg(
    **base,
    symmetry_cfg={
      "use_data_augmentation": True,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.0,
      "data_augmentation_func": MIRROR_FUNC,
    } if symmetry else None,
  )
  return cfg
