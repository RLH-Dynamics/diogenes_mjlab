"""Tests for the Harold biped suspended-gait task (Diogenes-Biped-Suspended)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diogenes_mjlab.harold_biped import gait


def _reference(phase_left) -> tuple[torch.Tensor, torch.Tensor]:
  """(pos, vel) for both feet given the LEFT leg's phase; shape (N, 2, 3)."""
  phase = torch.as_tensor(phase_left, dtype=torch.float64)
  return gait.foot_reference(
    gait.leg_phases(phase), gait.foot_centers(dtype=torch.float64)
  )


# ---------------------------------------------------------------------------
# Gait geometry.
# ---------------------------------------------------------------------------


def test_stance_is_straight_backward_line() -> None:
  phase = torch.linspace(0.0, gait.STANCE_FRACTION - 1e-9, 50, dtype=torch.float64)
  pos, vel = _reference(phase)
  left = pos[:, 0]
  _, cy = gait.FOOT_CENTER_XY["left_foot"]

  # Parallel to the ground and to y (the hip axes are fore/aft, so the outward
  # tilt moves the line sideways and slightly up without tilting it).
  assert torch.allclose(left[:, 2], left[0, 2].expand_as(left[:, 2]))
  assert torch.allclose(left[:, 0], left[0, 0].expand_as(left[:, 0]))
  assert left[0, 1].item() == pytest.approx(cy - gait.STEP_LENGTH / 2, abs=1e-6)
  assert left[-1, 1].item() == pytest.approx(cy + gait.STEP_LENGTH / 2, abs=1e-6)
  # Backward is +y: the foot sweeps monotonically toward +y during stance.
  assert (torch.diff(left[:, 1]) > 0).all()
  assert (vel[:, 0, 2] == 0).all()


def test_swing_peaks_above_stance_midpoint() -> None:
  apex = gait.STANCE_FRACTION + 0.5 * (1.0 - gait.STANCE_FRACTION)
  midpoint = 0.5 * gait.STANCE_FRACTION
  pos, vel = _reference([apex, midpoint])
  _, cy = gait.FOOT_CENTER_XY["left_foot"]
  # Apex is directly above the stance midpoint (same y), SWING_HEIGHT away
  # within the leg's tilted plane, and the foot is moving purely along y there.
  assert pos[0, 0, 1].item() == pytest.approx(cy, abs=1e-9)
  assert pos[1, 0, 1].item() == pytest.approx(cy, abs=1e-9)
  assert torch.linalg.norm(pos[0, 0] - pos[1, 0]).item() == pytest.approx(gait.SWING_HEIGHT, abs=1e-9)
  assert pos[0, 0, 2].item() > pos[1, 0, 2].item()
  assert torch.linalg.norm(vel[0, 0, [0, 2]]).item() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("join", [gait.STANCE_FRACTION, 1.0])
def test_loop_position_is_continuous(join: float) -> None:
  eps = 1e-9
  before, _ = _reference([join - eps])
  after, _ = _reference([(join + eps) % 1.0])
  assert torch.allclose(before, after, atol=1e-6)


def test_legs_are_half_a_cycle_apart() -> None:
  phase = torch.linspace(0.0, 0.999, 37, dtype=torch.float64)
  pos, _ = _reference(phase)
  shifted, _ = _reference(torch.remainder(phase + 0.5, 1.0))
  # Right foot at phase p mirrors the left foot at phase p + 0.5: same (y, z),
  # x reflected across the body's centre plane.
  assert torch.allclose(pos[:, 1, 1:], shifted[:, 0, 1:], atol=1e-9)
  assert torch.allclose(pos[:, 1, 0], -shifted[:, 0, 0], atol=1e-9)


def test_velocity_matches_finite_difference() -> None:
  h = 1e-6
  sf = gait.STANCE_FRACTION
  phase = torch.cat([
    torch.linspace(0.02, sf - 0.02, 20, dtype=torch.float64),
    torch.linspace(sf + 0.02, 0.98, 20, dtype=torch.float64),
  ])
  _, vel = _reference(phase)
  plus, _ = _reference(phase + h)
  minus, _ = _reference(phase - h)
  fd = (plus[:, 0] - minus[:, 0]) / (2 * h * gait.GAIT_PERIOD)
  assert torch.allclose(vel[:, 0], fd, atol=1e-4)


def test_gait_is_reachable_within_joint_limits() -> None:
  from diogenes_mjlab.constants import JOINT_LIMIT_MARGIN
  from diogenes_mjlab.tools.preview_harold_gait import build_model, reference_world, solve_ik
  import mujoco

  model, data = build_model()
  scratch = mujoco.MjData(model)
  lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
  # Every joint must clear the joint_at_limit termination band by 2 deg. (The
  # hips are held at 0 deg, only 5 deg inside their -5..45 deg ranges, so a flat
  # 10 deg margin is not achievable for them.)
  required = JOINT_LIMIT_MARGIN * (upper - lower) + np.radians(2)

  q = data.qpos.copy()
  q, _ = solve_ik(model, scratch, q, reference_world(data, 0.0), iters=200)
  for t in np.linspace(0.0, gait.GAIT_PERIOD, 100, endpoint=False):
    q, residual = solve_ik(model, scratch, q, reference_world(data, t), iters=50)
    assert residual.max() < 1e-3, f"IK residual {residual.max():.4f} m at t={t:.2f}s"
    margin = np.minimum(q - lower, upper - q)
    assert (margin > required).all(), (
      f"too close to a limit at t={t:.2f}s: margins {np.degrees(margin).round(1)} deg"
    )
    # The outward tilt is carried entirely by the hips, and outward means
    # left_hip positive / right_hip negative.
    for site, hip in (("left_foot", "left_hip"), ("right_foot", "right_hip")):
      q_hip = q[model.jnt_qposadr[model.joint(hip).id]]
      expected = gait.HIP_OUTWARD_SIGN[site] * gait.HIP_ABDUCTION
      assert abs(q_hip - expected) < np.radians(0.5), (
        f"{hip} at {np.degrees(q_hip):.2f} deg, expected {np.degrees(expected):.1f}"
      )


def test_loops_tilt_away_from_body() -> None:
  pos, _ = _reference(torch.linspace(0.0, 0.999, 25, dtype=torch.float64))
  for i, name in enumerate(gait.FOOT_SITE_NAMES):
    cx, _ = gait.FOOT_CENTER_XY[name]
    outward = torch.sign(torch.tensor(cx, dtype=torch.float64))
    assert ((pos[:, i, 0] - cx) * outward > 0.05).all(), f"{name} loop not tilted outward"


# ---------------------------------------------------------------------------
# Task config and registration.
# ---------------------------------------------------------------------------


def test_task_registered() -> None:
  import diogenes_mjlab  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  assert "Diogenes-Biped-Suspended" in list_tasks()


@pytest.mark.parametrize("play", [False, True])
def test_env_cfg_builds(play: bool) -> None:
  from diogenes_mjlab.harold_biped.env_cfg import (
    PRIVILEGED_OBS_TERMS,
    harold_suspended_env_cfg,
  )

  cfg = harold_suspended_env_cfg(play=play)
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms
  for name in PRIVILEGED_OBS_TERMS:
    assert name not in actor and name in critic
  assert actor["gait_clock"].params["hop_period"] == gait.GAIT_PERIOD
  assert {"feet_position", "feet_velocity"} <= set(cfg.rewards)
  assert ("reset_joint_pose" in cfg.events) is (not play)


def test_play_errors_follow_flags(monkeypatch) -> None:
  """Play is clean by default; DIOGENES_OBS_NOISE / DIOGENES_DOMAIN_RAND turn the
  sensor noise, delay and DR back on to preview real-robot errors."""
  from diogenes_mjlab.harold_biped.env_cfg import harold_suspended_env_cfg

  clean = harold_suspended_env_cfg(play=True)
  assert not clean.observations["actor"].enable_corruption
  assert "pd_gains" not in clean.events

  monkeypatch.setenv("DIOGENES_OBS_NOISE", "1")
  monkeypatch.setenv("DIOGENES_DOMAIN_RAND", "1")
  noisy = harold_suspended_env_cfg(play=True)
  actor = noisy.observations["actor"]
  assert actor.enable_corruption and actor.terms["joint_pos"].noise is not None
  assert actor.terms["joint_pos"].delay_max_lag == 3
  assert "pd_gains" in noisy.events and noisy.scene.num_envs == 1
  assert not noisy.observations["critic"].enable_corruption


# ---------------------------------------------------------------------------
# Zero-action sim step.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def suspended_env():
  from diogenes_mjlab.harold_biped.env_cfg import harold_suspended_env_cfg
  from mjlab.envs import ManagerBasedRlEnv

  cfg = harold_suspended_env_cfg(
    play=False, domain_rand=True, obs_noise=False, reset_joints=True, obs_history=0
  )
  cfg.scene.num_envs = 2
  cfg.sim.device = "cpu"
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  except TypeError:
    env = ManagerBasedRlEnv(cfg=cfg)
  yield env
  env.close()


def test_zero_action_step(suspended_env) -> None:
  env = suspended_env
  obs, _ = env.reset()
  assert obs["actor"].shape == (2, 20)  # 6 joint pos + 6 vel + 6 last action + 2 clock
  assert obs["critic"].shape == (2, 32)  # actor terms + 6 feet pos + 6 reference

  for _ in range(5):
    action = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
    obs, reward, _, _, _ = env.step(action)
    assert torch.isfinite(reward).all()
    assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()

  # The torso is welded 1 m up: feet sit ~0.61 m above the terrain.
  robot = env.scene["robot"]
  feet_z = robot.data.site_pos_w[:, :, 2] - env.scene.env_origins[:, None, 2]
  assert (feet_z > 0.4).all()
