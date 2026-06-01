"""RL configuration for the Diogenes task.

Updated for the mjlab 1.4.0 ``RslRlModelCfg`` API: the old ``stochastic`` flag
no longer exists. Actor stochasticity is now expressed via ``distribution_cfg``
(a dict); leaving it ``None`` yields a deterministic model, which is what the
critic uses.

For a zero-action sanity check none of these hyperparameters are exercised, but
they must be valid so the task imports and registers cleanly.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def diogenes_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for the Diogenes task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),
    experiment_name="diogenes",
    num_steps_per_env=24,
    save_interval=50,
    max_iterations=900,
  )
