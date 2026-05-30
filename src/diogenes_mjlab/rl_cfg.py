from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg
)

def diogenes_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      stochastic=True,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),
    experiment_name="diogenes",
    max_iterations=10_000,
  )