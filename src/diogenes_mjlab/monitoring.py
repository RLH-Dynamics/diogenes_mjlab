"""Real-time monitoring + CSV logging for the Diogenes hop stand.

This module adds two cooperating pieces of instrumentation on top of the task
MDP in ``mdp.py``:

1.  **Metric terms** (registered via ``cfg.metrics``). Each is a scalar-per-env
    function of shape ``(num_envs,)`` that mjlab's ``MetricsManager`` accumulates
    and -- crucially -- the Viser viewer auto-plots live in its **Metrics** tab
    (one selectable line per term, see ``viewer/viser/term_plotter.py``). These
    are what you watch in real time while a policy plays.

    Because a Viser plot line is a single scalar, vector quantities (the three
    joint torques, etc.) are exposed as one metric *per component*
    (``torque/hip``, ``torque/thigh``, ``torque/calf``), so every channel gets
    its own toggle and curve. A handful of scalar summaries (e.g. total
    mechanical power, contact-force magnitude) are added for at-a-glance views.

2.  **A CSV recorder** (``DiogenesCsvRecorder``, registered via ``cfg.recorders``).
    It writes one row per control step for a chosen environment, with a column
    for every raw channel you asked to save:

      * foot contact force      (Fx, Fy, Fz, |F|)
      * joint torques           (hip, thigh, calf)        [N*m, joint-space]
      * joint mechanical power  (hip, thigh, calf)        [W = tau * qdot]
      * slider pos / vel / acc  (raw joint-space values)
      * carriage height/vel/acc (sign-corrected: +up; = -slider)
      * joint positions         (hip, thigh, calf)        [rad]
      * joint velocities        (hip, thigh, calf)        [rad/s]

    The file lands in ``logs/diogenes_monitor/<run_tag>_<timestamp>.csv`` by
    default (one file per process), which opens directly in Excel and is easy to
    diff across runs.

Sign / units conventions match ``mdp.py``:
  * ``carriage_height_above_start = -slider_pos`` (the leg_mount body is rotated
    180 deg about X, so +height is -slider). Velocity and acceleration carry the
    same sign flip.
  * Joint torque is taken from ``qfrc_actuator`` (the PD law's joint-space
    generalized force for the position actuators), so power = tau * qdot is the
    mechanical power delivered at each joint in watts. (Note: this is the same
    quantity the ``electrical_power`` reward clamps; here we log it signed and
    unclamped so regeneration shows up.)
  * Foot contact force comes from the ``reduce="netforce"`` contact sensor, i.e.
    the net foot/floor wrench already expressed in the world frame.

All functions are written to be cheap: they index pre-resolved columns and do no
host syncs except inside the recorder (which must copy to host to write the file).
"""

from __future__ import annotations

import csv
import datetime
import os
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# The three actuated leg joints, in a fixed, documented order. Every per-joint
# metric and CSV column follows this order so runs are directly comparable.
JOINT_NAMES: tuple[str, ...] = ("hip", "thigh", "calf")


##
# Small indexing helpers (shared by metrics + recorder).
##


def _resolve_joint_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Resolve and cache the long-tensor joint ids selected by ``asset_cfg``."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  if isinstance(ids, slice):
    n = asset.data.joint_pos.shape[1]
    ids = list(range(*ids.indices(n)))
  return torch.as_tensor(ids, device=env.device, dtype=torch.long)


def _joint_torque(env: ManagerBasedRlEnv, joint_ids: torch.Tensor) -> torch.Tensor:
  """Joint-space actuator torque for ``joint_ids``. Shape (num_envs, J), N*m.

  Uses ``qfrc_actuator`` (the position-actuator PD law mapped into joint space),
  which is the correct quantity for mechanical power ``tau * qdot``.
  """
  asset: Entity = env.scene["robot"]
  return asset.data.qfrc_actuator[:, joint_ids]


def _joint_power(env: ManagerBasedRlEnv, joint_ids: torch.Tensor) -> torch.Tensor:
  """Per-joint mechanical power ``tau * qdot``. Shape (num_envs, J), watts.

  Signed: positive = motor doing work on the load, negative = regeneration.
  """
  asset: Entity = env.scene["robot"]
  tau = asset.data.qfrc_actuator[:, joint_ids]
  qd = asset.data.joint_vel[:, joint_ids]
  return tau * qd


##
# Scalar metric terms (shape (num_envs,)) -> live Viser plots.
##
# Each returns ONE scalar per env. Vector quantities are split into one metric
# per component via the ``component`` argument so each gets its own plot line.


def joint_torque_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  component: int,
) -> torch.Tensor:
  """Signed joint torque for a single joint (``component`` index into asset_cfg)."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  return _joint_torque(env, joint_ids)[:, component]


def joint_power_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  component: int,
) -> torch.Tensor:
  """Signed mechanical power ``tau*qdot`` for a single joint."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  return _joint_power(env, joint_ids)[:, component]


def total_mechanical_power_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Sum of per-joint mechanical power over the actuated joints. Watts."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  return torch.sum(_joint_power(env, joint_ids), dim=1)


def joint_pos_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  component: int,
) -> torch.Tensor:
  """Position of a single joint (rad)."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, joint_ids][:, component]


def joint_vel_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  component: int,
) -> torch.Tensor:
  """Velocity of a single joint (rad/s)."""
  joint_ids = _resolve_joint_ids(env, asset_cfg)
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, joint_ids][:, component]


def carriage_height_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Carriage height above start (m). ``= -slider_pos`` (see mdp.py)."""
  asset: Entity = env.scene[asset_cfg.name]
  return -asset.data.joint_pos[:, asset_cfg.joint_ids][:, -1]


def carriage_vel_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Carriage vertical velocity (m/s, +up). ``= -slider_vel``."""
  asset: Entity = env.scene[asset_cfg.name]
  return -asset.data.joint_vel[:, asset_cfg.joint_ids][:, -1]


def carriage_acc_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Carriage vertical acceleration (m/s^2, +up). ``= -slider_acc``."""
  asset: Entity = env.scene[asset_cfg.name]
  return -asset.data.joint_acc[:, asset_cfg.joint_ids][:, -1]


def slider_pos_metric(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Raw slider joint position (m, joint-space sign)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids][:, -1]


def slider_vel_metric(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Raw slider joint velocity (m/s, joint-space sign)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids][:, -1]


def slider_acc_metric(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Raw slider joint acceleration (m/s^2, joint-space sign)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_acc[:, asset_cfg.joint_ids][:, -1]


def contact_force_component_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  component: int,
) -> torch.Tensor:
  """One world-frame component (0=x,1=y,2=z) of the net foot/ground force (N).

  Reads the ``reduce="netforce"`` contact sensor, summed over its (single) slot.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # [B, N, 3], world frame (netforce).
  assert force is not None, (
    f"Sensor '{sensor_name}' must request the 'force' field for contact metrics."
  )
  return torch.sum(force[..., component], dim=-1)


def contact_force_magnitude_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Magnitude of the net foot/ground contact force (N)."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # [B, N, 3], world frame.
  assert force is not None, (
    f"Sensor '{sensor_name}' must request the 'force' field for contact metrics."
  )
  net = torch.sum(force, dim=1)  # [B, 3]
  return torch.linalg.norm(net, dim=-1)


##
# CSV recorder -- one row per control step for a single environment.
##


class DiogenesCsvRecorder(RecorderTerm):
  """Append one CSV row per control step with all monitored channels.

  Records a single environment (``params["env_idx"]``, default 0). During
  ``play`` there is exactly one env, so the default captures it. During training
  with thousands of envs, env 0 is a representative sample (you usually monitor
  in ``play`` anyway; the metric plots cover the training side).

  Columns (in order):
    step, time_s,
    foot_Fx, foot_Fy, foot_Fz, foot_Fmag,
    tau_hip, tau_thigh, tau_calf,
    P_hip, P_thigh, P_calf, P_total,
    slider_pos, slider_vel, slider_acc,
    carriage_height, carriage_vel, carriage_acc,
    q_hip, q_thigh, q_calf,
    qd_hip, qd_thigh, qd_calf

  ``params``:
    path:    output CSV path. If None, a timestamped file is created under
             ``logs/diogenes_monitor/``.
    env_idx: which environment to record (default 0).
    run_tag: optional label folded into the auto-generated filename. In the
             registered task this is supplied by ``diogenes_env_cfg`` (and can be
             set from the terminal via the ``DIOGENES_CSV_TAG`` env var).
  """

  def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    params = cfg.params or {}
    self._env_idx: int = int(params.get("env_idx", 0))
    self._step: int = 0
    self._dt: float = float(env.step_dt)

    # Resolve joint ids once (actuated leg joints + slider).
    robot: Entity = env.scene["robot"]
    leg_ids, leg_names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    self._leg_ids = torch.as_tensor(leg_ids, device=env.device, dtype=torch.long)
    slider_ids, _ = robot.find_joints(["slider"], preserve_order=True)
    self._slider_id = torch.as_tensor(slider_ids, device=env.device, dtype=torch.long)

    self._sensor_name: str = params["sensor_name"]

    # Resolve output path.
    path = params.get("path")
    if path is None:
      run_tag = params.get("run_tag", "run")
      ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      out_dir = os.path.join("logs", "diogenes_monitor")
      os.makedirs(out_dir, exist_ok=True)
      path = os.path.join(out_dir, f"{run_tag}_{ts}.csv")
    else:
      os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    self._path = path

    self._header = [
      "step",
      "time_s",
      "foot_Fx",
      "foot_Fy",
      "foot_Fz",
      "foot_Fmag",
      "tau_hip",
      "tau_thigh",
      "tau_calf",
      "P_hip",
      "P_thigh",
      "P_calf",
      "P_total",
      "slider_pos",
      "slider_vel",
      "slider_acc",
      "carriage_height",
      "carriage_vel",
      "carriage_acc",
      "q_hip",
      "q_thigh",
      "q_calf",
      "qd_hip",
      "qd_thigh",
      "qd_calf",
    ]
    self._file = open(self._path, "w", newline="")
    self._writer = csv.writer(self._file)
    self._writer.writerow(self._header)
    print(f"[INFO] DiogenesCsvRecorder writing to: {os.path.abspath(self._path)}")

  def record_post_step(self) -> None:
    e = self._env_idx
    robot: Entity = self._env.scene["robot"]
    sensor: ContactSensor = self._env.scene[self._sensor_name]

    # --- Foot contact force (world frame, net). ---
    force = sensor.data.force  # [B, N, 3]
    if force is not None:
      net = torch.sum(force[e], dim=0)  # [3]
      fx, fy, fz = net[0].item(), net[1].item(), net[2].item()
      fmag = float(torch.linalg.norm(net).item())
    else:
      fx = fy = fz = fmag = 0.0

    # --- Joint torque + power (joint-space). ---
    tau = robot.data.qfrc_actuator[e, self._leg_ids]  # [3]
    qd_leg = robot.data.joint_vel[e, self._leg_ids]  # [3]
    q_leg = robot.data.joint_pos[e, self._leg_ids]  # [3]
    power = tau * qd_leg  # [3]

    # --- Slider (raw) + carriage (sign-corrected, +up = -slider). ---
    s_pos = robot.data.joint_pos[e, self._slider_id][-1].item()
    s_vel = robot.data.joint_vel[e, self._slider_id][-1].item()
    s_acc = robot.data.joint_acc[e, self._slider_id][-1].item()

    row = [
      self._step,
      self._step * self._dt,
      fx,
      fy,
      fz,
      fmag,
      tau[0].item(),
      tau[1].item(),
      tau[2].item(),
      power[0].item(),
      power[1].item(),
      power[2].item(),
      float(power.sum().item()),
      s_pos,
      s_vel,
      s_acc,
      -s_pos,
      -s_vel,
      -s_acc,
      q_leg[0].item(),
      q_leg[1].item(),
      q_leg[2].item(),
      qd_leg[0].item(),
      qd_leg[1].item(),
      qd_leg[2].item(),
    ]
    self._writer.writerow(row)
    self._step += 1

  def close(self) -> None:
    try:
      self._file.flush()
      self._file.close()
    except Exception:
      pass
