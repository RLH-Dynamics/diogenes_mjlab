"""Tests for the Harold biped walking task (Diogenes-Biped-Walk)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Standard-frame model.
# ---------------------------------------------------------------------------


def _compiled(standard_frame: bool) -> tuple[mujoco.MjModel, mujoco.MjData]:
  from diogenes_mjlab.harold_biped.harold_biped_constants import get_spec

  model = get_spec(fixed_base=False, standard_frame=standard_frame).compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)  # qpos0: root at the origin, identity orientation
  return model, data


def test_standard_frame_is_a_pure_rotation_of_the_robot() -> None:
  old_m, old_d = _compiled(standard_frame=False)
  new_m, new_d = _compiled(standard_frame=True)
  rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

  assert (old_m.nq, old_m.nv, old_m.nu, old_m.ngeom) == (new_m.nq, new_m.nv, new_m.nu, new_m.ngeom)
  np.testing.assert_allclose(new_d.geom_xpos, old_d.geom_xpos @ rot.T, atol=1e-9)
  np.testing.assert_allclose(new_d.site_xpos, old_d.site_xpos @ rot.T, atol=1e-9)
  np.testing.assert_allclose(new_d.xipos, old_d.xipos @ rot.T, atol=1e-9)
  np.testing.assert_allclose(new_m.body_mass, old_m.body_mass)
  np.testing.assert_allclose(new_m.body_inertia, old_m.body_inertia, rtol=1e-6, atol=1e-12)
  np.testing.assert_allclose(new_m.jnt_range, old_m.jnt_range)
  for i in range(old_m.njnt):
    assert old_m.joint(i).name == new_m.joint(i).name


def test_standard_frame_axes() -> None:
  """+x forward, +y left: left foot at +y, and the knees (behind the hips on a
  bird-kneed robot) at -x relative to the hips."""
  m, d = _compiled(standard_frame=True)
  left, right = d.site("left_foot").xpos, d.site("right_foot").xpos
  assert left[1] > 0.05 and right[1] < -0.05
  for side in ("left", "right"):
    hip = d.xanchor[m.joint(f"{side}_thigh").id]
    knee = d.xanchor[m.joint(f"{side}_calf").id]
    assert knee[0] < hip[0] - 0.05


def test_bno085_site_at_chip_centre_with_chip_axes() -> None:
  """The IMU site sits at the chip's centre with +x left, +y up, +z forward."""
  from diogenes_mjlab.harold_biped.harold_biped_constants import IMU_SITE_NAME

  m, d = _compiled(standard_frame=True)
  mesh = m.mesh("bno085_chip").id
  g = next(i for i in range(m.ngeom) if m.geom_dataid[i] == mesh)
  a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
  verts = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
  site = d.site(IMU_SITE_NAME)
  np.testing.assert_allclose(site.xpos, 0.5 * (verts.min(axis=0) + verts.max(axis=0)), atol=5e-4)
  axes = site.xmat.reshape(3, 3)  # columns: the site's x, y, z in the torso frame
  np.testing.assert_allclose(axes[:, 0], [0, 1, 0], atol=1e-6)  # left
  np.testing.assert_allclose(axes[:, 1], [0, 0, 1], atol=1e-6)  # up
  np.testing.assert_allclose(axes[:, 2], [1, 0, 0], atol=1e-6)  # forward


# ---------------------------------------------------------------------------
# Walking entity: point feet and crouched stance.
# ---------------------------------------------------------------------------


def _walk_model(joint_pos: dict[str, float] | None = None, base_z: float | None = None):
  """The walking entity with a floor, at its init keyframe (optionally reposed)."""
  from diogenes_mjlab.harold_biped.walk_env_cfg import _robot_cfg

  spec = _robot_cfg().build().spec
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=1, conaffinity=1
  )
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  for name, value in (joint_pos or {}).items():
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
  if base_z is not None:
    data.qpos[2] = base_z
  mujoco.mj_forward(model, data)
  return model, data


def _lowest_point(model: mujoco.MjModel, data: mujoco.MjData, geom: str) -> float:
  g = model.geom(geom).id
  kind, radius = model.geom_type[g], model.geom_size[g, 0]
  if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
    return data.geom_xpos[g, 2] - radius
  if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
    half = model.geom_size[g, 1]
    axis_z = data.geom_xmat[g].reshape(3, 3)[2, 2]
    return data.geom_xpos[g, 2] - abs(axis_z) * half - radius
  mesh = model.geom_dataid[g]
  a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
  verts = model.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
  return verts[:, 2].min()


def _floor_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> set[str]:
  floor = model.geom("floor").id
  names = set()
  for c in data.contact[: data.ncon]:
    if floor in (c.geom1, c.geom2):
      names.add(model.geom(c.geom2 if c.geom1 == floor else c.geom1).name)
  return names


def test_point_feet_collide_only_through_tip_and_shank() -> None:
  from diogenes_mjlab.harold_biped.harold_biped_constants import FOOT_TIP_RADIUS

  model, data = _walk_model()
  collidable = {
    model.geom(i).name for i in range(model.ngeom)
    if (model.geom_contype[i] or model.geom_conaffinity[i]) and model.geom(i).name != "floor"
  }
  assert collidable == {"left_foot", "right_foot", "left_shank", "right_shank"}
  for side in ("left", "right"):
    assert model.geom_type[model.geom(f"{side}_foot").id] == mujoco.mjtGeom.mjGEOM_SPHERE
    mesh = model.geom(f"{side}_calf_mesh").id
    assert model.geom_contype[mesh] == 0 and model.geom_conaffinity[mesh] == 0
    # The sphere reaches the calf tip: the farthest mesh vertex from the knee
    # lies on the sphere surface.
    a, n = model.mesh_vertadr[model.geom_dataid[mesh]], model.mesh_vertnum[model.geom_dataid[mesh]]
    verts = model.mesh_vert[a:a + n] @ data.geom_xmat[mesh].reshape(3, 3).T + data.geom_xpos[mesh]
    knee = data.xanchor[model.joint(f"{side}_calf").id]
    tip = verts[np.linalg.norm(verts - knee, axis=1).argmax()]
    assert np.linalg.norm(tip - data.geom_xpos[model.geom(f"{side}_foot").id]) == pytest.approx(
      FOOT_TIP_RADIUS, abs=5e-4
    )


def test_crouch_raises_feet_7_5_cm_and_spawns_clear_of_the_floor() -> None:
  from diogenes_mjlab.constants import JOINT_LIMIT_MARGIN

  model, data = _walk_model()
  zero_m, zero_d = _compiled(standard_frame=True)
  base = data.body("body_assy").xpos
  for site in ("left_foot", "right_foot"):
    rise = (data.site(site).xpos - base)[2] - (zero_d.site(site).xpos - zero_d.body("body_assy").xpos)[2]
    assert rise == pytest.approx(0.075, abs=1e-3)

  lowest = min(_lowest_point(model, data, g) for g in ("left_foot", "right_foot"))
  assert 0.0015 < lowest < 0.0025  # 2 mm clearance
  # Standing on the tips, the shanks are well clear of the ground (~35 mm).
  assert min(_lowest_point(model, data, g) for g in ("left_shank", "right_shank")) > 0.02
  assert data.ncon == 0

  q = data.qpos[7:]
  lower, upper = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
  band = JOINT_LIMIT_MARGIN * (upper - lower)
  assert (np.minimum(q - lower, upper - q) > band + np.radians(2)).all()


def test_flat_calf_touches_ground_with_shank() -> None:
  """Laying the calf nearly flat (the sitting exploit) makes shank contact."""
  from diogenes_mjlab.harold_biped.walk_env_cfg import CROUCH_JOINT_POS

  # Crouched thighs with the knees folded to -74 deg (limit -75) put each calf
  # ~4 deg from horizontal, like the policy's sitting pose.
  pose = {**CROUCH_JOINT_POS, "left_calf": np.radians(-74.0), "right_calf": np.radians(-74.0)}
  colliders = ("left_foot", "right_foot", "left_shank", "right_shank")
  model, data = _walk_model(pose, base_z=1.0)
  drop = min(_lowest_point(model, data, g) for g in colliders)
  model, data = _walk_model(pose, base_z=1.0 - drop + 0.002)  # 2 mm above the floor

  for side in ("left", "right"):
    knee = data.xanchor[model.joint(f"{side}_calf").id]
    tip = data.geom_xpos[model.geom(f"{side}_foot").id]
    pitch = np.degrees(np.arctan2(knee[2] - tip[2], np.linalg.norm((knee - tip)[:2])))
    assert abs(pitch) < 9.0, f"{side} calf pitch {pitch:.1f} deg: test pose not flat"
  first_touch = min(colliders, key=lambda g: _lowest_point(model, data, g))
  assert first_touch.endswith("shank"), f"flat calf touched the floor with {first_touch}"


def test_torso_target_is_7_5_cm_below_standing() -> None:
  from diogenes_mjlab.harold_biped import walk_env_cfg as w

  assert w.STANDING_BASE_HEIGHT - w.TARGET_BASE_HEIGHT == pytest.approx(0.075)
  assert w.FALL_MIN_HEIGHT == pytest.approx(0.25)
  assert w.FALL_MIN_HEIGHT < w.TARGET_BASE_HEIGHT - 0.05


# ---------------------------------------------------------------------------
# Contact schedule.
# ---------------------------------------------------------------------------


def test_schedule_timing_at_1_4_hz() -> None:
  from diogenes_mjlab.harold_biped import gait
  from diogenes_mjlab.harold_biped.mdp import expected_stance

  freq, swing = 1.4, 0.4
  period = 1.0 / freq
  dt = 1e-4
  t = torch.arange(0.0, 10 * period, dt, dtype=torch.float64)
  stance = expected_stance(gait.leg_phases(torch.remainder(t / period, 1.0)), swing)  # [T, 2]

  for foot in range(2):
    s = stance[:, foot].int()
    liftoffs = int(((s[:-1] == 1) & (s[1:] == 0)).sum())
    touchdowns = int(((s[:-1] == 0) & (s[1:] == 1)).sum())
    # One lift-off and one touch-down per cycle, i.e. at the step frequency.
    assert liftoffs == pytest.approx(10, abs=1) and touchdowns == pytest.approx(10, abs=1)
    assert (1 - s.double().mean()).item() == pytest.approx(swing, abs=1e-3)

  both_down = (stance[:, 0] & stance[:, 1]).double().mean().item() * period
  both_up = (~stance[:, 0] & ~stance[:, 1]).double().mean().item()
  assert both_up == 0.0
  # Both feet down (1 - 2 * swing) of the cycle in total: 0.071 s per handover.
  assert both_down / 2 == pytest.approx((1 - 2 * swing) * period / 2, abs=1e-3)
  # The right foot follows the left by half a cycle. Half a period isn't a
  # whole number of samples, so allow one sample of disagreement per edge.
  shift = round(0.5 * period / dt)
  mismatched = int((stance[shift:, 1] != stance[:-shift, 0]).sum())
  assert mismatched <= 2 * 10 + 2


def test_schedule_match_scoring() -> None:
  from diogenes_mjlab.harold_biped.mdp import contact_schedule_match

  swing, margin = 0.4, 0.05
  phase = torch.tensor([0.30, 0.80, 0.02, 0.58, 0.62])
  # Mid-stance: contact matches, airborne doesn't.
  assert contact_schedule_match(torch.tensor([True]), phase[:1], swing, margin).item() == 1.0
  assert contact_schedule_match(torch.tensor([False]), phase[:1], swing, margin).item() == 0.0
  # Mid-swing: airborne matches, contact doesn't.
  assert contact_schedule_match(torch.tensor([False]), phase[1:2], swing, margin).item() == 1.0
  assert contact_schedule_match(torch.tensor([True]), phase[1:2], swing, margin).item() == 0.0
  # Near touch-down (0.02) and either side of lift-off (0.58, 0.62): both states accepted.
  for state in (True, False):
    out = contact_schedule_match(torch.full((3,), state), phase[2:], swing, margin)
    assert torch.equal(out, torch.ones(3))


def test_off_schedule_event_counting() -> None:
  from diogenes_mjlab.harold_biped.mdp import off_schedule_events

  swing, margin = 0.4, 0.1  # lift-off due at phase 0.6, touch-down at 0.0
  phase = torch.tensor([0.02, 0.95, 0.30, 0.55, 0.65, 0.85])
  touchdown = torch.tensor([True, True, True, False, False, False])
  liftoff = torch.tensor([False, False, False, True, True, True])
  counts = off_schedule_events(touchdown, liftoff, phase, swing, margin)
  # On-time touch-downs (0.02, 0.95 wraps to 0.05 away) and lift-offs (0.55,
  # 0.65) are free; a mid-stance touch-down (0.30) and a late lift-off (0.85) count.
  assert counts.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
  both = off_schedule_events(torch.tensor([True]), torch.tensor([True]), torch.tensor([0.3]), swing, margin)
  assert both.item() == 2.0


# ---------------------------------------------------------------------------
# IMU error model.
# ---------------------------------------------------------------------------


def test_gyro_noise_model_holds_bias_and_scale_per_episode() -> None:
  from diogenes_mjlab.harold_biped.mdp import GyroNoiseModelCfg
  from mjlab.utils.noise.noise_cfg import ConstantNoiseCfg

  bias, scale = 0.02, 0.02
  cfg = GyroNoiseModelCfg(
    noise_cfg=ConstantNoiseCfg(bias=0.0),  # no white noise, to see bias and scale alone
    bias_range=(-bias, bias),
    scale_range=(-scale, scale),
  )
  model = cfg.class_type(cfg, num_envs=64, device="cpu")
  rate = torch.full((64, 3), 2.0)

  first = model(rate)
  torch.testing.assert_close(model(rate), first)  # constant within an episode
  assert ((first - rate).abs() <= 2.0 * scale + bias + 1e-6).all()
  torch.testing.assert_close(model(torch.zeros(64, 3)), model.bias)
  assert model.bias.abs().max() <= bias and (model.scale - 1.0).abs().max() <= scale
  assert model.bias.std() > 0.005 and model.scale.std() > 0.005  # differs per env and axis

  model.reset(torch.tensor([0, 1]))  # new episode for envs 0 and 1 only
  after = model(rate)
  assert not torch.equal(after[:2], first[:2])
  torch.testing.assert_close(after[2:], first[2:])


# ---------------------------------------------------------------------------
# Task config and registration.
# ---------------------------------------------------------------------------


def test_task_registered() -> None:
  import diogenes_mjlab  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  assert "Diogenes-Biped-Walk" in list_tasks()


@pytest.mark.parametrize("play", [False, True])
def test_walk_cfg_builds(play: bool) -> None:
  from diogenes_mjlab.harold_biped.walk_env_cfg import (
    PRIVILEGED_OBS_TERMS,
    STEP_FREQUENCY_HZ,
    WALK_SPEED_RANGE,
    harold_walk_env_cfg,
  )

  cfg = harold_walk_env_cfg(play=play)
  ranges = cfg.commands["twist"].ranges
  assert ranges.lin_vel_x == WALK_SPEED_RANGE and ranges.lin_vel_x[0] >= 0.0
  assert ranges.lin_vel_y == (0.0, 0.0) and ranges.ang_vel_z == (0.0, 0.0)
  assert {"time_out", "fell_over", "base_too_low", "joint_at_limit", "shank_contact"} == set(
    cfg.terminations
  )
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms
  for name in PRIVILEGED_OBS_TERMS:
    assert name not in actor and name in critic
  for group in (actor, critic):
    assert group["gait_clock"].params["hop_period"] == pytest.approx(1.0 / STEP_FREQUENCY_HZ)
  assert {"base_height", "contact_schedule", "off_schedule_contacts"} <= set(cfg.rewards)
  assert cfg.rewards["off_schedule_contacts"].weight < 0
  assert "air_time" not in cfg.rewards
  assert ("push_robot" in cfg.events) is (not play)
  assert ("foot_friction" in cfg.events) is (not play)

  # IMU: chip-frame sensor terms on the actor, with the BNO085 error model.
  from diogenes_mjlab.harold_biped.mdp import GyroNoiseModelCfg
  from diogenes_mjlab.harold_biped.walk_env_cfg import (
    IMU_DELAY_HOLD_PROB,
    IMU_GYRO_SENSOR,
    IMU_UP_SENSOR,
  )

  assert list(actor)[:2] == ["imu_ang_vel", "imu_gravity"]
  assert not {"base_ang_vel", "projected_gravity"} & set(actor)
  assert {IMU_GYRO_SENSOR, IMU_UP_SENSOR} <= {s.prefixed_name for s in cfg.scene.sensors}
  assert ("imu_mount_tilt" in cfg.events) is (not play)
  assert isinstance(actor["imu_ang_vel"].noise, GyroNoiseModelCfg) is (not play)
  for term in ("imu_ang_vel", "imu_gravity"):
    assert actor[term].delay_hold_prob == IMU_DELAY_HOLD_PROB
    assert actor[term].delay_max_lag == (0 if play else 3)


def test_play_errors_follow_flags(monkeypatch) -> None:
  """Play is clean by default; DIOGENES_OBS_NOISE / DIOGENES_DOMAIN_RAND turn the
  sensor errors and DR (IMU mount tilt included) back on for a real-robot preview."""
  from diogenes_mjlab.harold_biped.mdp import GyroNoiseModelCfg
  from diogenes_mjlab.harold_biped.walk_env_cfg import harold_walk_env_cfg

  clean = harold_walk_env_cfg(play=True)
  assert not clean.observations["actor"].enable_corruption
  assert "imu_mount_tilt" not in clean.events

  monkeypatch.setenv("DIOGENES_OBS_NOISE", "1")
  monkeypatch.setenv("DIOGENES_DOMAIN_RAND", "1")
  noisy = harold_walk_env_cfg(play=True)
  actor = noisy.observations["actor"]
  assert actor.enable_corruption
  assert isinstance(actor.terms["imu_ang_vel"].noise, GyroNoiseModelCfg)
  assert actor.terms["imu_gravity"].delay_max_lag == 3
  assert "imu_mount_tilt" in noisy.events
  assert "push_robot" not in noisy.events and noisy.scene.num_envs == 1
  # Critic stays clean either way.
  assert not noisy.observations["critic"].enable_corruption


def test_step_frequency_argument_and_env_var(monkeypatch) -> None:
  from diogenes_mjlab.harold_biped.walk_env_cfg import harold_walk_env_cfg

  def periods(cfg) -> tuple[float, float, float]:
    return (
      cfg.observations["actor"].terms["gait_clock"].params["hop_period"],
      1.0 / cfg.rewards["contact_schedule"].params["step_frequency"],
      1.0 / cfg.rewards["off_schedule_contacts"].params["step_frequency"],
    )

  assert periods(harold_walk_env_cfg(step_frequency=2.0)) == pytest.approx((0.5, 0.5, 0.5))
  monkeypatch.setenv("DIOGENES_STEP_FREQ", "1.25")
  assert periods(harold_walk_env_cfg()) == pytest.approx((0.8, 0.8, 0.8))
  assert periods(harold_walk_env_cfg(step_frequency=2.0)) == pytest.approx((0.5, 0.5, 0.5))


# ---------------------------------------------------------------------------
# Zero-action sim steps.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def walk_env():
  from diogenes_mjlab.harold_biped.walk_env_cfg import harold_walk_env_cfg
  from mjlab.envs import ManagerBasedRlEnv

  cfg = harold_walk_env_cfg(
    play=False, domain_rand=True, obs_noise=False, obs_history=0, step_frequency=1.4
  )
  cfg.scene.num_envs = 2
  cfg.sim.device = "cpu"
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  except TypeError:
    env = ManagerBasedRlEnv(cfg=cfg)
  yield env
  env.close()


def test_zero_action_steps(walk_env) -> None:
  env = walk_env
  obs, _ = env.reset()
  # actor: 3 IMU ang vel + 3 IMU gravity + 6 joint pos + 6 joint vel + 6 actions + 3 command + 2 clock
  assert obs["actor"].shape == (2, 29)
  # critic: actor terms + 3 lin vel + 2 foot height + 2 air time + 2 contact + 6 forces
  assert obs["critic"].shape == (2, 44)

  robot = env.scene["robot"]
  for _ in range(10):  # 0.2 s
    action = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
    obs, reward, terminated, _, _ = env.step(action)
    assert torch.isfinite(reward).all()
    assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()

  # Spawned crouched on its foot tips: after 0.2 s both feet touch the ground,
  # the shanks don't, the torso is near the crouched height and no episode ended.
  found = env.scene["feet_ground_contact"].data.found
  assert found.shape == (2, 2) and (found > 0).all()
  shank = env.scene["shank_ground_contact"].data.found
  assert shank.shape == (2, 2) and not (shank > 0).any()
  torso_z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  assert ((torso_z > 0.25) & (torso_z < 0.34)).all()
  assert not terminated.any()
  log = env.extras["log"]
  assert "Metrics/left_foot_touchdown_hz" in log and "Metrics/off_schedule_events_hz" in log


def test_imu_readings_match_bno085_hand_checks(walk_env) -> None:
  """The sim IMU reads like the BNO085 (+x left, +y up, +z forward) in the same
  hand checks to run on the robot. The tolerance covers the +-2 deg mount-tilt DR."""
  from diogenes_mjlab.harold_biped.mdp import imu_angular_velocity, imu_projected_gravity
  from diogenes_mjlab.harold_biped.walk_env_cfg import IMU_GYRO_SENSOR, IMU_UP_SENSOR

  env = walk_env
  env.reset()
  robot = env.scene["robot"]
  n, dev = env.num_envs, env.device

  def place(quat: tuple[float, ...], world_ang_vel: tuple[float, float, float]) -> None:
    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, 1.0], device=dev)
    pose[:, 3:] = torch.tensor(quat, device=dev)
    vel = torch.zeros(n, 6, device=dev)
    vel[:, 3:] = torch.tensor(world_ang_vel, device=dev)
    robot.write_root_link_pose_to_sim(pose)
    robot.write_root_link_velocity_to_sim(vel)
    env.sim.forward()

  def check(actual: torch.Tensor, expected: tuple[float, float, float], what: str) -> None:
    torch.testing.assert_close(
      actual, torch.tensor(expected, device=dev).expand(n, 3), atol=0.08, rtol=0.0, msg=what
    )

  c15, s15, c30 = math.cos(math.radians(15)), math.sin(math.radians(15)), math.cos(math.radians(30))
  # (w, x, y, z) torso orientations in the standard frame (+x forward, +y left).
  for quat, gravity, what in (
    ((1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0), "level"),
    ((c15, 0.0, s15, 0.0), (0.0, -c30, 0.5), "nose down 30 deg"),
    ((c15, -s15, 0.0, 0.0), (0.5, -c30, 0.0), "left side down 30 deg"),
  ):
    place(quat, (0.0, 0.0, 0.0))
    check(imu_projected_gravity(env, IMU_UP_SENSOR), gravity, what)

  for world_rate, chip_rate, what in (
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "turning left"),
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), "tipping nose down"),
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "rolling left side up"),
  ):
    place((1.0, 0.0, 0.0, 0.0), world_rate)
    check(imu_angular_velocity(env, IMU_GYRO_SENSOR), chip_rate, what)
  env.reset()
