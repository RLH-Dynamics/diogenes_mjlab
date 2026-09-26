"""Environment config for the Harold biped walking task.

Registered as ``Diogenes-Biped-Walk``. A velocity-command task in the style of
mjlab's ``tasks/velocity`` (as used for the Unitree G1), adapted to Harold:

  * forward/backward and turning commands (no sideways stepping: the hips
    swing only 5 deg inward), 10% of envs commanded to step in place, the
    command ranges widened by a curriculum,
  * a crouched stance: the default pose bends the knees so the torso walks
    7.5 cm lower than standing straight-legged, with a torso-height reward,
  * a fixed step frequency per training run: a gait clock observation and a
    contact-schedule reward make each foot lift off and touch down once per
    1 / STEP_FREQUENCY_HZ seconds, the right foot half a cycle behind the left,
  * flat ground, no height scan,
  * the RS03 motors modelled as a PD with the robot's gains, a torque-speed
    limit at the battery's voltage (drawn per episode) and targets clamped to
    the joint range, as the control stack clamps them (``actuators.py``),
  * 2 ms timestep x 10 decimation (50 Hz), the leg's observation noise/delay,
    and DR ranges re-centred on the hanging-run logs (``HAROLD_WALK_DR_RANGES``),
  * a BNO085 IMU model: angular velocity and gravity read at the chip's site in
    the chip's own axes, with per-episode gyro bias and scale error, a
    randomized mount tilt and a steadier (held) IMU observation delay,
  * point feet: each foot collides through a sphere at the calf tip; a shank
    capsule touching the ground ends the episode (so the policy can't rest the
    calf flat as a stable support),
  * episodes end when the torso tilts past 45 deg, drops below 0.25 m or a
    shank touches the ground; the outer 5% of each joint's range is penalised
    rather than ending the episode,
  * episodes start from the crouch with the joints jittered, the hips up to
    10 deg splayed outward, so the policy learns to recover from it,
  * left/right mirror symmetry is available to PPO (``symmetry.py``).

The robot is built in the standard frame (+x forward, +y left, +z up; see
``harold_biped_constants._to_standard_frame``) so mjlab's velocity rewards and
command visualisation read correctly.
"""

import math
from dataclasses import replace

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg

from ..config.domain_rand import DEFAULT_DR_RANGES, DomainRandRanges, _scale_range
from ..constants import (
  OBS_DELAY_MAX_LAG,
  OBS_DELAY_MIN_LAG,
  OBS_HISTORY_LENGTH,
  OBS_NOISE_JOINT_POS,
  OBS_NOISE_JOINT_VEL,
)
from ..flags import _env_bool, _env_float, _env_int
from .. import mdp as diogenes_mdp
from . import mdp as harold_mdp
from .actuators import BATTERY_VOLTAGE_RANGE, Rs03ActuatorCfg, randomize_battery_voltage
from .env_cfg import (
  LEG_LINK_NAMES,
  _domain_randomization_events,
  actuators_cfg,
  joints_cfg,
)
from .harold_biped_constants import (
  HAROLD_JOINT_NAMES,
  IMU_SITE_NAME,
  ROOT_BODY_NAME,
  get_harold_biped_cfg,
  get_spec,
)

# ---------------------------------------------------------------------------
# Task constants.
# ---------------------------------------------------------------------------

#: Commanded forward speed (m/s) and yaw rate (rad/s, + = left) at the end of
#: the curriculum. Zero asks for stepping in place (a point-foot biped cannot
#: stand still).
WALK_SPEED_RANGE: tuple[float, float] = (-0.1, 0.3)
YAW_RATE_RANGE: tuple[float, float] = (-0.5, 0.5)

#: Fraction of envs (per command resample) commanded to step in place.
STEP_IN_PLACE_FRACTION: float = 0.1

#: Command curriculum: from each PPO iteration on, (speed range, yaw-rate range).
#: The last stage is the full range; play starts there.
COMMAND_CURRICULUM: tuple[tuple[int, tuple[float, float], tuple[float, float]], ...] = (
  (0, (-0.05, 0.15), (-0.2, 0.2)),
  (400, (-0.1, 0.25), (-0.35, 0.35)),
  (1000, WALK_SPEED_RANGE, YAW_RATE_RANGE),
)

#: Env steps per PPO iteration (rl_cfg num_steps_per_env), to place the stages.
ENV_STEPS_PER_ITERATION: int = 24

#: Default step frequency (Hz): each foot lifts off and touches down once per
#: 1 / STEP_FREQUENCY_HZ seconds. Override per run with DIOGENES_STEP_FREQ.
STEP_FREQUENCY_HZ: float = 1.4

#: Fraction of each leg's cycle spent in the air. At 1.4 Hz: 0.29 s swing,
#: 0.43 s stance, and both feet down for 0.07 s at each handover.
SWING_FRACTION: float = 0.4

#: Half-width (cycle fraction) of the window around each lift-off/touch-down in
#: which either contact state is accepted (36 ms at 1.4 Hz).
CONTACT_TRANSITION_MARGIN: float = 0.05

#: Half-width (cycle fraction) around each scheduled lift-off/touch-down within
#: which an actual lift-off/touch-down is not penalised (71 ms at 1.4 Hz).
CONTACT_EVENT_MARGIN: float = 0.1

#: Crouched default pose (rad): the foot sites 7.5 cm closer to the torso than
#: at the zero pose, directly above their zero-pose ground positions (solved
#: with IK). Joint margins: thigh 42 deg, calf 43 deg, hip 5 deg.
CROUCH_JOINT_POS: dict[str, float] = {
  "left_hip": 0.0,
  "right_hip": 0.0,
  "left_thigh": 0.3105,
  "right_thigh": -0.3105,
  "left_calf": -0.55192,
  "right_calf": -0.55192,
}

#: Spawn height (m) for the crouched pose: foot tip spheres 2 mm above the ground.
CROUCH_SPAWN_HEIGHT: float = 0.3368

#: Torso height (m) the straight-legged robot settles at on the ground
#: (measured: 0.391 m 0.2 s after landing, joints held at zero).
STANDING_BASE_HEIGHT: float = 0.391

#: Walking torso height target: 7.5 cm below standing straight-legged. Holding
#: the crouch with the joints alone sags to ~0.30 m, so the policy has to push
#: up slightly.
TARGET_BASE_HEIGHT: float = STANDING_BASE_HEIGHT - 0.075

#: Foot sites and foot collision geoms, left then right.
FOOT_SITE_NAMES: tuple[str, ...] = ("left_foot", "right_foot")
FOOT_GEOM_NAMES: tuple[str, ...] = ("left_foot", "right_foot")

#: Torso tilt (from upright) that ends the episode.
FALL_TILT_LIMIT: float = math.radians(45.0)

#: Torso height (m) that ends the episode; ~7 cm below the walking target and
#: below the ~0.30 m the crouch sags to with the joints alone.
FALL_MIN_HEIGHT: float = 0.25

#: Shank colliders (one contact sensor column each); any ground contact ends
#: the episode.
SHANK_GEOM_NAMES: tuple[str, ...] = ("left_shank", "right_shank")
SHANK_CONTACT_SENSOR = "shank_ground_contact"

#: Target foot-site height during swing (m). The site sits ~20 mm above the
#: calf tip, so this lifts the tip ~40 mm, like the suspended gait.
SWING_SITE_HEIGHT: float = 0.06

#: Position-action scale (rad per unit action). The hips get half: they only
#: need small sideways corrections, and their inward stop is 5 deg away.
ACTION_SCALE: dict[str, float] = {r".*_hip": 0.25, r".*_thigh": 0.5, r".*_calf": 0.5}

#: Joint-range fraction inside which no penalty applies: the outer 5% at each
#: end is penalised (``dof_pos_limits``) instead of ending the episode.
SOFT_JOINT_LIMIT_FACTOR: float = 0.9

#: Reset offsets (rad) from the crouch, uniform per joint. With DR on, the hips
#: start anywhere from 2 deg inward to 10 deg splayed outward (outward is +
#: for the left hip, - for the right) so the policy learns to bring them back;
#: without DR, a small jitter only.
RESET_JOINT_OFFSETS_DR: dict[str, tuple[float, float]] = {
  "left_hip": (-0.035, 0.175),
  "right_hip": (-0.175, 0.035),
  r".*_thigh": (-0.1, 0.1),
  r".*_calf": (-0.1, 0.1),
}
RESET_JOINT_OFFSETS_PLAY: dict[str, tuple[float, float]] = {r".*": (-0.05, 0.05)}

#: DR ranges for walking, re-centred on the hanging-run logs (2026-09-24/25,
#: replayed in the fixed-base sim): static stiffness matched kp=60 within
#: ~10%; friction fitted ~0.4 N.m (0.9 on the left calf) against the leg's
#: 0.15-1.6; and the joints moved less damped than modelled, which the fit put
#: down to 3-5x the armature -- more likely lag in the motors' own damping --
#: so both the armature range and the low end of kd are widened.
HAROLD_WALK_DR_RANGES = DomainRandRanges(
  pd_gains_kp=(0.85, 1.15),
  pd_gains_kd=(0.5, 1.3),
  joint_armature=(0.015, 0.06),
  joint_friction=(0.1, 1.2),
)

# ---------------------------------------------------------------------------
# BNO085 IMU model.
# ---------------------------------------------------------------------------

#: The actor's IMU observations are read at the BNO085 site (chip centre) in the
#: chip's own axes: +x = robot left, +y = up, +z = forward.
#:   imu_ang_vel: rad/s, like the BNO085 calibrated gyroscope report.
#:   imu_gravity: unit vector pointing DOWN, (0, -1, 0) with the torso level. On
#:     the robot, normalise the BNO085 gravity report (or rotate world-down by the
#:     game rotation vector) and negate it if it reads +g on +y when level.
#: Scene sensor keys (a site-attached sensor is prefixed with its entity).
IMU_GYRO_SENSOR = "robot/imu_gyro"
IMU_UP_SENSOR = "imu_up"

#: Per-step white noise, uniform +-.
OBS_NOISE_IMU_ANG_VEL: float = 0.2  # rad/s
OBS_NOISE_IMU_GRAVITY: float = 0.05  # unit vector (~3 deg)

#: Per-episode, per-axis gyro calibration error: additive bias and scale error.
IMU_GYRO_BIAS: float = 0.02  # rad/s (~1.1 deg/s)
IMU_GYRO_SCALE_ERROR: float = 0.02  # +-2 %

#: Mount tilt / calibration offset: at startup the IMU site is rotated by up to
#: this angle about each axis (DR, widened by dr_scale). It rotates the gyro and
#: gravity readings together, like a real tilted mount.
IMU_MOUNT_TILT: float = math.radians(2.0)

#: IMU observation delay (control steps of 20 ms). Gyro and fused gravity have
#: separate ranges so the gravity estimate can be made slower once the real
#: latency is measured; both currently match the joint encoders.
IMU_GYRO_DELAY_LAG: tuple[int, int] = (OBS_DELAY_MIN_LAG, OBS_DELAY_MAX_LAG)
IMU_GRAVITY_DELAY_LAG: tuple[int, int] = (OBS_DELAY_MIN_LAG, OBS_DELAY_MAX_LAG)
#: Probability of keeping the previous lag each step (mean hold 10 steps = 0.2 s),
#: so latency stays steady instead of jumping between 0 and 60 ms every step.
IMU_DELAY_HOLD_PROB: float = 0.9

#: Critic-only (privileged) observation terms.
PRIVILEGED_OBS_TERMS = (
  "base_lin_vel",
  "foot_height",
  "foot_air_time",
  "foot_contact",
  "foot_contact_forces",
)

FEET_CONTACT_SENSOR = "feet_ground_contact"
FOOT_HEIGHT_SENSOR = "foot_height_scan"

# Smallest "Max" value mjlab's Viser joystick sliders accept (slider min=0.1).
_GUI_MIN_AXIS_MAX = 0.1


class _ForwardVelocityCommand(UniformVelocityCommand):
  """UniformVelocityCommand whose Viser joystick tolerates zero-width axes.

  mjlab's ``create_gui`` starts each axis's "Max" slider at that axis's range
  maximum, but the slider's own minimum is 0.1, so the forward-only ranges
  (lin_vel_y and ang_vel_z capped at 0) trip viser's range assertion and ``play``
  crashes. The GUI is built from ranges widened to at least 0.1; sampled
  commands keep using the real ranges.
  """

  def create_gui(self, *args, **kwargs) -> None:
    ranges = self.cfg.ranges

    def widened(bounds: tuple[float, float]) -> tuple[float, float]:
      return (bounds[0], max(bounds[1], _GUI_MIN_AXIS_MAX))

    self.cfg.ranges = replace(
      ranges,
      lin_vel_x=widened(ranges.lin_vel_x),
      lin_vel_y=widened(ranges.lin_vel_y),
      ang_vel_z=widened(ranges.ang_vel_z),
    )
    try:
      super().create_gui(*args, **kwargs)
    finally:
      self.cfg.ranges = ranges


class _ForwardVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> _ForwardVelocityCommand:
    return _ForwardVelocityCommand(self, env)


def feet_sites_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True)


def torso_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=(ROOT_BODY_NAME,))


def _imu_sensor_cfgs() -> tuple[BuiltinSensorCfg, BuiltinSensorCfg]:
  """Gyro and up-vector (world +z in the chip frame) sensors at the BNO085 site."""
  site = ObjRef(type="site", name=IMU_SITE_NAME, entity="robot")
  gyro = BuiltinSensorCfg(name="imu_gyro", sensor_type="gyro", obj=site)
  up = BuiltinSensorCfg(
    name="imu_up", sensor_type="framezaxis", obj=ObjRef(type="body", name="world"), ref=site
  )
  assert (gyro.prefixed_name, up.prefixed_name) == (IMU_GYRO_SENSOR, IMU_UP_SENSOR)
  return gyro, up


def _walk_spec():
  """The standard-frame, point-foot model without the XML position actuators,
  which the RS03 actuator model replaces."""
  spec = get_spec(fixed_base=False, standard_frame=True, point_feet=True)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


def _robot_cfg() -> EntityCfg:
  """Floating biped in the standard frame with point feet and RS03 actuator
  models, starting crouched."""
  cfg = get_harold_biped_cfg(fixed_base=False, standard_frame=True, point_feet=True)
  cfg.spec_fn = _walk_spec
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      Rs03ActuatorCfg(
        target_names_expr=HAROLD_JOINT_NAMES,
        # Nominal values for play; DR redraws both every episode.
        armature=0.03,
        frictionloss=0.5,
      ),
    ),
    soft_joint_pos_limit_factor=SOFT_JOINT_LIMIT_FACTOR,
  )
  cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, CROUCH_SPAWN_HEIGHT),
    joint_pos=dict(CROUCH_JOINT_POS),
    joint_vel={".*": 0.0},
  )
  return cfg


# ---------------------------------------------------------------------------
# Rewards.
# ---------------------------------------------------------------------------


def _build_rewards(step_frequency: float) -> dict[str, RewardTermCfg]:
  """Velocity-task rewards scaled for Harold's size. Starting weights, untuned."""
  # Loosest on the sagittal joints that make the stride; tight on the hips to
  # limit sideways sway. Centred on the crouched default pose. The same stds
  # apply when the command is ~0, since the robot must keep stepping.
  posture_std = {r".*_hip": 0.15, r".*_thigh": 0.35, r".*_calf": 0.35}
  command_threshold = 0.05

  return {
    # Weights and bands doubled/tightened after the first run of this config
    # (2026-09-26) reached only ~70% of the commanded speed and turn rate with
    # 2.0 / 0.15 m/s and 1.0 / 0.3 rad/s.
    "track_linear_velocity": RewardTermCfg(
      func=vel_mdp.track_linear_velocity,
      weight=4.0,
      params={"command_name": "twist", "std": 0.1},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=vel_mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": 0.2},
    ),
    "upright": RewardTermCfg(
      func=vel_mdp.upright,
      weight=1.0,
      params={"std": math.sqrt(0.2), "asset_cfg": torso_cfg()},
    ),
    "base_height": RewardTermCfg(
      func=harold_mdp.base_height_tracking,
      weight=1.0,
      # Wide enough to still pull a torso sagging 7-8 cm low (a first 50-iteration
      # run with std=0.03 settled at 0.24 m, where that reward is ~0).
      params={"target_height": TARGET_BASE_HEIGHT, "std": 0.08},
    ),
    "contact_schedule": RewardTermCfg(
      func=harold_mdp.feet_contact_schedule,
      weight=1.5,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "step_frequency": step_frequency,
        "swing_fraction": SWING_FRACTION,
        "transition_margin": CONTACT_TRANSITION_MARGIN,
      },
    ),
    "off_schedule_contacts": RewardTermCfg(
      func=harold_mdp.off_schedule_contact_events,
      # Per lift-off/touch-down outside its window. Each costs 2x a full step of
      # the contact_schedule reward, so tapping between scheduled steps loses.
      weight=-3.0,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "step_frequency": step_frequency,
        "swing_fraction": SWING_FRACTION,
        "event_margin": CONTACT_EVENT_MARGIN,
      },
    ),
    "pose": RewardTermCfg(
      func=vel_mdp.variable_posture,
      weight=0.5,
      params={
        "asset_cfg": joints_cfg(),
        "command_name": "twist",
        "std_standing": posture_std,
        "std_walking": posture_std,
        "std_running": posture_std,
        "walking_threshold": command_threshold,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=vel_mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": torso_cfg()},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      # rad past the soft limit (the outer 5% of the range): 2.5 deg into a
      # hip's band costs ~1.3 per step. At -10, ~5% of robots still leaned a
      # hip on its 5 deg inward stop.
      weight=-30.0,
      params={"asset_cfg": joints_cfg()},
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
    "torque": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-0.002,
      params={"asset_cfg": actuators_cfg()},
    ),
    "electrical_power": RewardTermCfg(
      func=envs_mdp.electrical_power_cost,
      weight=-0.0005,
      params={"asset_cfg": joints_cfg()},
    ),
    # No air_time reward: the contact schedule sets when each foot is airborne.
    "foot_clearance": RewardTermCfg(
      func=vel_mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": SWING_SITE_HEIGHT,
        "height_sensor_name": FOOT_HEIGHT_SENSOR,
        "command_name": "twist",
        "command_threshold": command_threshold,
        "asset_cfg": feet_sites_cfg(),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=vel_mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "height_sensor_name": FOOT_HEIGHT_SENSOR,
        "target_height": SWING_SITE_HEIGHT,
        "command_name": "twist",
        "command_threshold": command_threshold,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=vel_mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "command_name": "twist",
        "command_threshold": command_threshold,
        "asset_cfg": feet_sites_cfg(),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=vel_mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "command_name": "twist",
        "command_threshold": command_threshold,
      },
    ),
    "termination_penalty": RewardTermCfg(
      func=envs_mdp.is_terminated,
      weight=-100.0,
    ),
  }


# ---------------------------------------------------------------------------
# Observations.
# ---------------------------------------------------------------------------


def _actor_terms(obs_noise: bool, step_frequency: float) -> dict[str, ObservationTermCfg]:
  """What the real robot can measure: IMU (BNO085 chip frame), encoders, last
  action, command, clock."""

  def noise(n: float) -> UniformNoiseCfg | None:
    return UniformNoiseCfg(n_min=-n, n_max=n) if obs_noise else None

  dmin = OBS_DELAY_MIN_LAG if obs_noise else 0
  dmax = OBS_DELAY_MAX_LAG if obs_noise else 0
  delay = {"delay_min_lag": dmin, "delay_max_lag": dmax}

  def imu_delay(lags: tuple[int, int]) -> dict:
    lo, hi = lags if obs_noise else (0, 0)
    return {"delay_min_lag": lo, "delay_max_lag": hi, "delay_hold_prob": IMU_DELAY_HOLD_PROB}

  gyro_noise = (
    harold_mdp.GyroNoiseModelCfg(
      noise_cfg=UniformNoiseCfg(n_min=-OBS_NOISE_IMU_ANG_VEL, n_max=OBS_NOISE_IMU_ANG_VEL),
      bias_range=(-IMU_GYRO_BIAS, IMU_GYRO_BIAS),
      scale_range=(-IMU_GYRO_SCALE_ERROR, IMU_GYRO_SCALE_ERROR),
    )
    if obs_noise
    else None
  )

  return {
    "imu_ang_vel": ObservationTermCfg(
      func=harold_mdp.imu_angular_velocity,
      params={"sensor_name": IMU_GYRO_SENSOR},
      noise=gyro_noise,
      **imu_delay(IMU_GYRO_DELAY_LAG),
    ),
    "imu_gravity": ObservationTermCfg(
      func=harold_mdp.imu_projected_gravity,
      params={"sensor_name": IMU_UP_SENSOR},
      noise=noise(OBS_NOISE_IMU_GRAVITY),
      **imu_delay(IMU_GRAVITY_DELAY_LAG),
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": joints_cfg()},
      noise=noise(OBS_NOISE_JOINT_POS),
      **delay,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": joints_cfg()},
      noise=noise(OBS_NOISE_JOINT_VEL),
      **delay,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action, **delay),
    "command": ObservationTermCfg(
      func=vel_mdp.generated_commands, params={"command_name": "twist"}
    ),
    "gait_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock, params={"hop_period": 1.0 / step_frequency}
    ),
  }


def _critic_terms(step_frequency: float) -> dict[str, ObservationTermCfg]:
  """Clean actor terms plus privileged base velocity and foot/contact state.

  The critic's IMU terms are the true torso angular velocity and gravity in the
  torso frame, free of the actor's IMU errors (bias, scale, mount tilt).
  """
  return {
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel, params={"asset_cfg": joints_cfg()}
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel, params={"asset_cfg": joints_cfg()}
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "command": ObservationTermCfg(
      func=vel_mdp.generated_commands, params={"command_name": "twist"}
    ),
    "gait_clock": ObservationTermCfg(
      func=diogenes_mdp.phase_clock, params={"hop_period": 1.0 / step_frequency}
    ),
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "foot_height": ObservationTermCfg(
      func=vel_mdp.foot_height, params={"sensor_name": FOOT_HEIGHT_SENSOR}
    ),
    "foot_air_time": ObservationTermCfg(
      func=vel_mdp.foot_air_time, params={"sensor_name": FEET_CONTACT_SENSOR}
    ),
    "foot_contact": ObservationTermCfg(
      func=vel_mdp.foot_contact, params={"sensor_name": FEET_CONTACT_SENSOR}
    ),
    # Heading frame, so the left/right mirror is a fixed map (symmetry.py).
    "foot_contact_forces": ObservationTermCfg(
      func=harold_mdp.foot_contact_forces_heading, params={"sensor_name": FEET_CONTACT_SENSOR}
    ),
  }


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


def harold_walk_env_cfg(
  play: bool = False,
  domain_rand: bool | None = None,
  obs_noise: bool | None = None,
  dr_scale: float | None = None,
  obs_history: int | None = None,
  step_frequency: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the Harold biped forward-walking environment configuration.

  Flags resolve as in ``harold_suspended_env_cfg``: explicit argument, then the
  matching ``DIOGENES_*`` env var, then the default.

  Args:
    play: evaluation overrides (1 env, effectively infinite episode, no pushes;
      DR and observation noise/delay default OFF, but the two flags below turn
      them on to preview the policy with real-robot sensor and model errors).
    domain_rand: startup DR events, including the IMU mount tilt
      (``DIOGENES_DOMAIN_RAND``; default not play).
    obs_noise: actor sensor noise + delay, including the gyro bias and scale
      error (``DIOGENES_OBS_NOISE``; default not play).
    dr_scale: DR range half-width multiplier (``DIOGENES_DR_SCALE``; default 1.0).
    obs_history: past actor timesteps (``DIOGENES_OBS_HISTORY``; default
      OBS_HISTORY_LENGTH).
    step_frequency: steps per second per foot (``DIOGENES_STEP_FREQ``; default
      STEP_FREQUENCY_HZ). A policy only follows the frequency it was trained
      with, so play must use the same value as training.
  """
  if domain_rand is None:
    domain_rand = _env_bool("DIOGENES_DOMAIN_RAND")
  if domain_rand is None:
    domain_rand = not play
  if obs_noise is None:
    obs_noise = _env_bool("DIOGENES_OBS_NOISE")
  if obs_noise is None:
    obs_noise = not play
  if dr_scale is None:
    dr_scale = _env_float("DIOGENES_DR_SCALE")
  if dr_scale is None:
    dr_scale = 1.0
  if obs_history is None:
    obs_history = _env_int("DIOGENES_OBS_HISTORY")
  if obs_history is None:
    obs_history = OBS_HISTORY_LENGTH
  if step_frequency is None:
    step_frequency = _env_float("DIOGENES_STEP_FREQ")
  if step_frequency is None:
    step_frequency = STEP_FREQUENCY_HZ
  assert step_frequency > 0.0, f"step_frequency must be positive, got {step_frequency}"

  # 2 ms physics step * decimation 10 -> 50 Hz control, as for the leg. A drop
  # test peaked at 6 foot contacts / 30 constraint rows, so these leave margin.
  sim_cfg = SimulationCfg(
    njmax=128,
    nconmax=32,
    contact_sensor_maxmatch=64,
    mujoco=MujocoCfg(timestep=0.002),
  )

  feet_contact_cfg = ContactSensorCfg(
    name=FEET_CONTACT_SENSOR,
    primary=ContactMatch(mode="geom", pattern=r"^(left_foot|right_foot)$", entity="robot"),
    # The TerrainEntity plane geom is named "terrain" (see config/env.py).
    secondary=ContactMatch(mode="geom", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  shank_contact_cfg = ContactSensorCfg(
    name=SHANK_CONTACT_SENSOR,
    primary=ContactMatch(mode="geom", pattern=r"^(left_shank|right_shank)$", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="terrain"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )
  foot_height_cfg = TerrainHeightSensorCfg(
    name=FOOT_HEIGHT_SENSOR,
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES),
    pattern=RingPatternCfg.single_ring(radius=0.01, num_samples=4),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
  )
  imu_gyro_cfg, imu_up_cfg = _imu_sensor_cfgs()

  # Walking robots never touch each other: the only collidable geoms are the
  # foot spheres and shank capsules (contype 2 / conaffinity 1), which collide
  # with the terrain only.
  scene_cfg = SceneCfg(
    num_envs=4096,
    env_spacing=2.0,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": _robot_cfg()},
    sensors=(feet_contact_cfg, shank_contact_cfg, foot_height_cfg, imu_gyro_cfg, imu_up_cfg),
  )

  commands = {
    "twist": _ForwardVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 8.0),
      rel_standing_envs=STEP_IN_PLACE_FRACTION,
      heading_command=False,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=WALK_SPEED_RANGE,
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=YAW_RATE_RANGE,
      ),
      viz=UniformVelocityCommandCfg.VizCfg(z_offset=0.5),
    )
  }

  actor_terms = _actor_terms(obs_noise=obs_noise, step_frequency=step_frequency)
  critic_terms = _critic_terms(step_frequency=step_frequency)
  for name in PRIVILEGED_OBS_TERMS:
    assert name not in actor_terms, f"Privileged obs term {name!r} leaked into actor."
    assert name in critic_terms, f"Privileged obs term {name!r} missing from critic."

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      # Follows obs_noise, so DIOGENES_OBS_NOISE=1 also corrupts play observations.
      enable_corruption=obs_noise,
      history_length=obs_history or None,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  events: dict[str, EventTermCfg] = {}
  if domain_rand:
    events.update(
      _domain_randomization_events(
        dr_scale,
        reset_joint_pose=False,
        ranges=HAROLD_WALK_DR_RANGES,
        inertial_body_names=(ROOT_BODY_NAME, *LEG_LINK_NAMES),
      )
    )
    # Battery charge sets the RS03 no-load speed (see actuators.py).
    events["battery_voltage"] = EventTermCfg(
      func=randomize_battery_voltage,
      mode="reset",
      params={"voltage_range": BATTERY_VOLTAGE_RANGE, "asset_cfg": SceneEntityCfg("robot")},
    )
    events["foot_friction"] = EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
        "ranges": DEFAULT_DR_RANGES.foot_friction,
        "operation": "abs",
        "distribution": "uniform",
        "shared_random": True,
      },
    )
    tilt = _scale_range(-IMU_MOUNT_TILT, IMU_MOUNT_TILT, dr_scale)
    events["imu_mount_tilt"] = EventTermCfg(
      func=dr.site_quat,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=(IMU_SITE_NAME,)),
        "roll_range": tilt,
        "pitch_range": tilt,
        "yaw_range": tilt,
        "distribution": "uniform",
      },
    )
  events["reset_base"] = EventTermCfg(
    func=envs_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.02),  # CROUCH_SPAWN_HEIGHT already clears the feet by 2 mm
        "yaw": (-math.pi, math.pi),
      },
      "velocity_range": {},
    },
  )
  events["reset_robot_joints"] = EventTermCfg(
    func=harold_mdp.reset_joints_by_offset_per_joint,
    mode="reset",
    params={
      "offsets": RESET_JOINT_OFFSETS_DR if domain_rand else RESET_JOINT_OFFSETS_PLAY,
      "asset_cfg": joints_cfg(),
    },
  )
  if not play:
    # Gentler than the G1's pushes: Harold is ~9 kg with point feet.
    events["push_robot"] = EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 4.0),
      params={
        "velocity_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-0.3, 0.3),
        },
      },
    )

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": FALL_TILT_LIMIT}
    ),
    "base_too_low": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": FALL_MIN_HEIGHT},
    ),
    "shank_contact": TerminationTermCfg(
      func=vel_mdp.illegal_contact,
      params={"sensor_name": SHANK_CONTACT_SENSOR},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=20.0,
    sim=sim_cfg,
    scene=scene_cfg,
    observations=observations,
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=joints_cfg().joint_names,
        scale=ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands=commands,
    events=events,
    rewards=_build_rewards(step_frequency),
    terminations=terminations,
    curriculum={} if play else {
      "command_ranges": CurriculumTermCfg(
        func=vel_mdp.commands_vel,
        params={
          "command_name": "twist",
          "velocity_stages": [
            {"step": it * ENV_STEPS_PER_ITERATION, "lin_vel_x": vx, "ang_vel_z": wz}
            for it, vx, wz in COMMAND_CURRICULUM
          ],
        },
      ),
    },
  )

  cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
  cfg.viewer.entity_name = "robot"
  cfg.viewer.body_name = ROOT_BODY_NAME
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0
  cfg.viewer.azimuth = 90.0

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)

  return cfg
