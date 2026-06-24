# diogenes-mjlab

MuJoCo/mjlab environment for the Diogenes periodic-hopping leg (sim-to-real).

## Architecture

The package lives under `src/diogenes_mjlab/` and is split into focused modules:

```
constants.py          -- shared physical constants (gravity, joint limits, sensor names, obs noise)
accessors.py          -- thin wrappers that read robot state from the scene
flags.py              -- env-var flag parsing helpers (DIOGENES_* vars)

config/
  entities.py         -- SceneEntityCfg builders (actuated joints, slider, foot geom, ...)
  rewards.py          -- RewardWeights dataclass + reward-term builder (_build_rewards)
  domain_rand.py      -- DomainRandRanges dataclass + DR-event builder (_domain_randomization_events)
  observations.py     -- _actor_terms / _critic_terms (explicit privileged-info split)
  instrumentation.py  -- live metrics and CSV recorder builder
  env.py              -- diogenes_env_cfg() orchestrator (assembles all sub-configs)

mdp/
  trajectories.py     -- dual-parabola and sinusoid trajectory math
  rewards.py          -- custom reward functions (slider tracking, foot xy, slip, ...)
  observations.py     -- custom obs functions (slider pos/vel, phase clock)
  terminations.py     -- joint-at-limit termination
  events.py           -- reset_joints_uniform_legal DR event

monitoring/
  metrics.py          -- live Viser metric terms
  recorder.py         -- per-step CSV recorder terms

tools/                -- offline utilities (check_masses, inspect_joints, export_onnx)
```

### Privileged actor-critic split

The actor group receives `joint_pos`, `joint_vel`, `last_action`, `phase_clock`.
The critic additionally receives `slider_pos` and `slider_vel` (privileged carriage state).
`diogenes_env_cfg()` asserts this invariant at config-build time so the split can never
silently regress.

### Tuning knobs

- `config/rewards.py::RewardWeights` -- frozen dataclass; pass an instance to `_build_rewards()` to override any weight without touching the builder.
- `config/domain_rand.py::DomainRandRanges` -- frozen dataclass; pass an instance to `_domain_randomization_events()` to override any DR range.

### Test safety net

`tests/test_config_snapshot.py` builds all four configs (trajectory x play mode) and
compares them against a golden JSON at `tests/snapshots/diogenes_cfg.json`.
Re-bless after intentional changes: `DIOGENES_BLESS_SNAPSHOT=1 uv run pytest -q tests/test_config_snapshot.py`.
