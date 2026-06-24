"""CSV recorder for the Diogenes hop stand.

The CSV schema is modelled as a single ordered list of ``(column_name, value_fn)``
pairs (``_CSV_SCHEMA`` below). The header row and each data row are both derived
from the SAME list, so they can never desync -- adding a column in one place
automatically adds it in the other.

Column order (preserved verbatim from the original monitoring.py):
  step, time_s,
  foot_Fx, foot_Fy, foot_Fz, foot_Fmag,
  tau_hip, tau_thigh, tau_calf,
  P_hip, P_thigh, P_calf, P_total,
  slider_pos, slider_vel, slider_acc,
  carriage_height, carriage_vel, carriage_acc,
  q_hip, q_thigh, q_calf,
  qd_hip, qd_thigh, qd_calf
"""

from __future__ import annotations

import csv
import datetime
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.sensor import ContactSensor

from ..constants import DIOGENES_ACTUATOR_NAMES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Names of the three actuated leg joints (same order as DIOGENES_ACTUATOR_NAMES).
JOINT_NAMES: tuple[str, ...] = DIOGENES_ACTUATOR_NAMES

# ---------------------------------------------------------------------------
# Type alias for value extractors.
#
# Each extractor receives a named-tuple-like bundle of pre-fetched data and
# returns one Python scalar (int or float).  The bundle fields:
#   robot    -- the robot Entity (already indexed into the env)
#   sensor   -- the ContactSensor (already indexed into the env)
#   e        -- environment index (int)
#   leg_ids  -- torch.Tensor long, resolved actuated joint ids
#   slider_id -- torch.Tensor long, resolved slider joint id
#   step     -- current step counter (int)
#   dt       -- control step duration (float, seconds)
# ---------------------------------------------------------------------------

_ExtractorFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# CSV schema: ONE ordered list of (column_name, extractor_fn) pairs.
#
# Extractors receive keyword arguments matching the fields of _RecordBundle
# (defined in DiogenesCsvRecorder.record_post_step) so they stay independent
# of position and easy to unit-test.
# ---------------------------------------------------------------------------

def _build_csv_schema() -> list[tuple[str, _ExtractorFn]]:
  """Build the ordered (column_name, extractor) list for the CSV."""

  schema: list[tuple[str, _ExtractorFn]] = []

  # --- step / time ---
  schema.append(("step", lambda step, **_: step))
  schema.append(("time_s", lambda step, dt, **_: step * dt))

  # --- foot contact force (world frame, net) ---
  def _net_force(sensor: ContactSensor, e: int) -> torch.Tensor:
    """Net contact force [3] for env e, zeros if no force data."""
    f = sensor.data.force
    if f is not None:
      return torch.sum(f[e], dim=0)
    return torch.zeros(3)

  schema.append(("foot_Fx",   lambda sensor, e, **_: _net_force(sensor, e)[0].item()))
  schema.append(("foot_Fy",   lambda sensor, e, **_: _net_force(sensor, e)[1].item()))
  schema.append(("foot_Fz",   lambda sensor, e, **_: _net_force(sensor, e)[2].item()))
  schema.append(("foot_Fmag", lambda sensor, e, **_: float(
    torch.linalg.norm(_net_force(sensor, e)).item()
  )))

  # --- joint torque (N*m) ---
  for i, jname in enumerate(JOINT_NAMES):
    _i = i  # capture loop variable
    schema.append((
      f"tau_{jname}",
      lambda robot, e, leg_ids, _i=_i, **_: robot.data.qfrc_actuator[e, leg_ids][_i].item(),
    ))

  # --- joint mechanical power (W = tau * qdot) ---
  for i, jname in enumerate(JOINT_NAMES):
    _i = i
    schema.append((
      f"P_{jname}",
      lambda robot, e, leg_ids, _i=_i, **_: (
        robot.data.qfrc_actuator[e, leg_ids][_i] *
        robot.data.joint_vel[e, leg_ids][_i]
      ).item(),
    ))

  schema.append((
    "P_total",
    lambda robot, e, leg_ids, **_: float(
      (robot.data.qfrc_actuator[e, leg_ids] * robot.data.joint_vel[e, leg_ids])
      .sum().item()
    ),
  ))

  # --- slider raw (joint-space sign) ---
  schema.append(("slider_pos", lambda robot, e, slider_id, **_: robot.data.joint_pos[e, slider_id][-1].item()))
  schema.append(("slider_vel", lambda robot, e, slider_id, **_: robot.data.joint_vel[e, slider_id][-1].item()))
  schema.append(("slider_acc", lambda robot, e, slider_id, **_: robot.data.joint_acc[e, slider_id][-1].item()))

  # --- carriage (sign-corrected: +up = -slider) ---
  schema.append(("carriage_height", lambda robot, e, slider_id, **_: -robot.data.joint_pos[e, slider_id][-1].item()))
  schema.append(("carriage_vel",    lambda robot, e, slider_id, **_: -robot.data.joint_vel[e, slider_id][-1].item()))
  schema.append(("carriage_acc",    lambda robot, e, slider_id, **_: -robot.data.joint_acc[e, slider_id][-1].item()))

  # --- joint positions (rad) ---
  for i, jname in enumerate(JOINT_NAMES):
    _i = i
    schema.append((
      f"q_{jname}",
      lambda robot, e, leg_ids, _i=_i, **_: robot.data.joint_pos[e, leg_ids][_i].item(),
    ))

  # --- joint velocities (rad/s) ---
  for i, jname in enumerate(JOINT_NAMES):
    _i = i
    schema.append((
      f"qd_{jname}",
      lambda robot, e, leg_ids, _i=_i, **_: robot.data.joint_vel[e, leg_ids][_i].item(),
    ))

  return schema


# Module-level singleton: built once, shared by all recorder instances.
_CSV_SCHEMA: list[tuple[str, _ExtractorFn]] = _build_csv_schema()

# Derived header (read-only view for the writer).
CSV_HEADER: tuple[str, ...] = tuple(col for col, _ in _CSV_SCHEMA)


class DiogenesCsvRecorder(RecorderTerm):
  """Append one CSV row per control step with all monitored channels.

  Records a single environment (``params["env_idx"]``, default 0). During
  ``play`` there is exactly one env, so the default captures it.

  Columns (in order, derived from ``_CSV_SCHEMA``):
    step, time_s,
    foot_Fx, foot_Fy, foot_Fz, foot_Fmag,
    tau_hip, tau_thigh, tau_calf,
    P_hip, P_thigh, P_calf, P_total,
    slider_pos, slider_vel, slider_acc,
    carriage_height, carriage_vel, carriage_acc,
    q_hip, q_thigh, q_calf,
    qd_hip, qd_thigh, qd_calf

  ``params``:
    path:       output CSV path. If None, a timestamped file is created under
                ``logs/diogenes_monitor/``.
    env_idx:    which environment to record (default 0).
    run_tag:    optional label folded into the auto-generated filename.
    sensor_name: name of the foot/ground ContactSensor (required).
  """

  def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    params = cfg.params or {}
    self._env_idx: int = int(params.get("env_idx", 0))
    self._step: int = 0
    self._dt: float = float(env.step_dt)

    # Resolve joint ids once (actuated leg joints + slider).
    robot: Entity = env.scene["robot"]
    leg_ids, _ = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
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

    self._file = open(self._path, "w", newline="")
    self._writer = csv.writer(self._file)
    # Header is derived from the schema — the single source of truth.
    self._writer.writerow(list(CSV_HEADER))
    print(f"[INFO] DiogenesCsvRecorder writing to: {os.path.abspath(self._path)}")

  def record_post_step(self) -> None:
    e = self._env_idx
    robot: Entity = self._env.scene["robot"]
    sensor: ContactSensor = self._env.scene[self._sensor_name]

    # Bundle of named arguments passed to every extractor.
    kwargs = {
      "robot": robot,
      "sensor": sensor,
      "e": e,
      "leg_ids": self._leg_ids,
      "slider_id": self._slider_id,
      "step": self._step,
      "dt": self._dt,
    }

    # Row is derived from the SAME schema as the header -- can never desync.
    row = [extractor(**kwargs) for _, extractor in _CSV_SCHEMA]
    self._writer.writerow(row)
    self._step += 1

  def close(self) -> None:
    try:
      self._file.flush()
      self._file.close()
    except Exception:
      pass
