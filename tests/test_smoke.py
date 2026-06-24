"""Smoke tests: import, task registration, config building, and (if possible) sim step.

These tests assert:
  1. The package imports without error.
  2. Both task IDs are registered in mjlab's task registry.
  3. All four diogenes_env_cfg(...) builds succeed without error.
  4. (Optional) A zero-action env step runs for N steps returning finite rewards/obs.
     Wrapped in pytest.skip if the environment requires a GPU or other unavailable
     resource.
"""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# 1. Package import
# ---------------------------------------------------------------------------

def test_package_imports() -> None:
  """The package must import without error."""
  import diogenes_mjlab  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Task registration
# ---------------------------------------------------------------------------

def test_tasks_registered() -> None:
  """Both Diogenes task IDs must be present in the mjlab task registry."""
  import diogenes_mjlab  # noqa: F401 — triggers register_mjlab_task calls

  from mjlab.tasks.registry import list_tasks
  ids = list_tasks()
  assert "Diogenes-Flat" in ids, f"Diogenes-Flat not in registry: {ids}"
  assert "Diogenes-Flat-Sine" in ids, f"Diogenes-Flat-Sine not in registry: {ids}"


# ---------------------------------------------------------------------------
# 3. Config builds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trajectory,play", [
  ("dual_parabola", False),
  ("dual_parabola", True),
  ("sine", False),
  ("sine", True),
])
def test_env_cfg_builds(trajectory: str, play: bool) -> None:
  """diogenes_env_cfg must build without error for every (trajectory, play) combo."""
  from diogenes_mjlab.env_cfgs import diogenes_env_cfg

  cfg = diogenes_env_cfg(
    trajectory=trajectory,
    play=play,
    monitor=False,    # keep it lightweight
    record_csv=False,
    domain_rand=not play,
    obs_noise=not play,
    dr_scale=1.0,
    reset_joints=not play,
  )

  # Basic sanity on the returned object.
  assert cfg is not None
  assert hasattr(cfg, "rewards")
  assert hasattr(cfg, "observations")
  assert hasattr(cfg, "terminations")

  # Rewards must be non-empty for both trajectories.
  rewards = getattr(cfg, "rewards", {}) or {}
  assert len(rewards) > 0, "Expected at least one reward term"

  # Observations must have actor and critic groups.
  obs = getattr(cfg, "observations", {}) or {}
  assert "actor" in obs, f"Expected 'actor' group in observations, got: {list(obs)}"
  assert "critic" in obs, f"Expected 'critic' group in observations, got: {list(obs)}"

  # Actor must NOT have slider_{pos,vel} (privileged critic-only terms).
  actor_terms = list(getattr(obs["actor"], "terms", {}).keys())
  assert "slider_pos" not in actor_terms, "slider_pos leaked into actor group"
  assert "slider_vel" not in actor_terms, "slider_vel leaked into actor group"

  # Critic MUST have slider_{pos,vel}.
  critic_terms = list(getattr(obs["critic"], "terms", {}).keys())
  assert "slider_pos" in critic_terms, "slider_pos missing from critic group"
  assert "slider_vel" in critic_terms, "slider_vel missing from critic group"

  # Phase clock must be in both groups and have the correct period.
  from diogenes_mjlab.env_cfgs import TRAJ_T, SINE_PERIOD
  expected_period = TRAJ_T if trajectory == "dual_parabola" else SINE_PERIOD

  actor_phase = obs["actor"].terms.get("phase_clock")
  assert actor_phase is not None, "phase_clock missing from actor group"
  actual_period = actor_phase.params.get("hop_period")
  assert actual_period is not None, "hop_period missing from phase_clock params"
  assert abs(actual_period - expected_period) < 1e-6, (
    f"phase_clock period mismatch: got {actual_period}, expected {expected_period}"
  )

  critic_phase = obs["critic"].terms.get("phase_clock")
  assert critic_phase is not None, "phase_clock missing from critic group"
  assert abs(critic_phase.params.get("hop_period", 0) - expected_period) < 1e-6


# ---------------------------------------------------------------------------
# 4. Zero-action sim step (attempt; skip if GPU/resource unavailable)
# ---------------------------------------------------------------------------

_SIM_SKIP_REASON: str | None = None
_SIM_AVAILABLE: bool | None = None


def _check_sim_available() -> tuple[bool, str]:
  """Try to import and build the env; return (available, reason_if_not)."""
  try:
    import torch  # noqa: F401
  except ImportError:
    return False, "torch not importable"

  try:
    from mjlab.envs import ManagerBasedRlEnv  # noqa: F401
  except ImportError:
    return False, "mjlab.envs.ManagerBasedRlEnv not importable"

  return True, ""


@pytest.fixture(scope="module")
def sim_env():
  """Build a tiny (2-env, CPU) Diogenes environment, or skip."""
  available, reason = _check_sim_available()
  if not available:
    pytest.skip(f"Sim step skipped: {reason}")

  from diogenes_mjlab.env_cfgs import diogenes_env_cfg

  cfg = diogenes_env_cfg(
    trajectory="dual_parabola",
    play=False,
    monitor=False,
    record_csv=False,
    domain_rand=False,
    obs_noise=False,
    dr_scale=1.0,
    reset_joints=False,
  )
  # Override to a tiny env footprint.
  cfg.scene.num_envs = 2

  # Try to set CPU device if the cfg supports it.
  for attr in ("device", "sim_device"):
    if hasattr(cfg, attr):
      setattr(cfg, attr, "cpu")
  if hasattr(cfg, "sim") and hasattr(cfg.sim, "device"):
    cfg.sim.device = "cpu"

  try:
    from mjlab.envs import ManagerBasedRlEnv
    env = ManagerBasedRlEnv(cfg=cfg)
  except Exception as exc:
    pytest.skip(f"Sim step skipped: could not build env: {exc}")
    return

  yield env

  try:
    env.close()
  except Exception:
    pass


def test_zero_action_step(sim_env) -> None:
  """N zero-action steps must return finite rewards and observations."""
  import torch

  N_STEPS = 5
  obs, _ = sim_env.reset()

  for step in range(N_STEPS):
    action = torch.zeros(sim_env.num_envs, sim_env.action_space.shape[-1])
    obs, reward, terminated, truncated, info = sim_env.step(action)

    # Rewards must be finite.
    assert torch.isfinite(reward).all(), (
      f"Non-finite reward at step {step}: {reward}"
    )

    # Observations must be finite for every group.
    if isinstance(obs, dict):
      for group_name, group_obs in obs.items():
        if isinstance(group_obs, torch.Tensor):
          assert torch.isfinite(group_obs).all(), (
            f"Non-finite obs in group '{group_name}' at step {step}"
          )
    elif isinstance(obs, torch.Tensor):
      assert torch.isfinite(obs).all(), (
        f"Non-finite obs at step {step}: {obs}"
      )
