"""Monitoring wiring helpers for the Diogenes hop stand.

Builds the metric terms (live Viser plots) and the CSV recorder term
from the ``monitoring`` package. Named ``instrumentation.py`` (not
``monitoring.py``) to avoid shadowing the top-level ``monitoring`` package.
"""

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg

from ..constants import FOOT_CONTACT_SENSOR
from .. import monitoring
from .entities import monitored_joints_cfg, slider_cfg


def _monitoring_metrics() -> dict[str, MetricsTermCfg]:
  """Build the per-step scalar metric terms that the Viser viewer plots live.

  One term per channel so each gets its own toggle/curve in the Metrics tab:
    * contact force        : foot_force/{x,y,z}, foot_force/mag
    * joint torque         : torque/{hip,thigh,calf}
    * joint power          : power/{hip,thigh,calf}, power/total
    * slider (raw)         : slider/{pos,vel,acc}
    * carriage (+up)       : carriage/{height,vel,acc}
    * joint position       : qpos/{hip,thigh,calf}
    * joint velocity       : qvel/{hip,thigh,calf}
  """
  metrics: dict[str, MetricsTermCfg] = {}

  # --- Foot contact force (world frame, net). ---
  for axis, comp in (("x", 0), ("y", 1), ("z", 2)):
    metrics[f"foot_force/{axis}"] = MetricsTermCfg(
      func=monitoring.contact_force_component_metric,
      params={"sensor_name": FOOT_CONTACT_SENSOR, "component": comp},
    )
  metrics["foot_force/mag"] = MetricsTermCfg(
    func=monitoring.contact_force_magnitude_metric,
    params={"sensor_name": FOOT_CONTACT_SENSOR},
  )

  # --- Per-joint torque, power, position, velocity. ---
  for comp, jname in enumerate(monitoring.JOINT_NAMES):
    metrics[f"torque/{jname}"] = MetricsTermCfg(
      func=monitoring.joint_torque_metric,
      params={"asset_cfg": monitored_joints_cfg(), "component": comp},
    )
    metrics[f"power/{jname}"] = MetricsTermCfg(
      func=monitoring.joint_power_metric,
      params={"asset_cfg": monitored_joints_cfg(), "component": comp},
    )
    metrics[f"qpos/{jname}"] = MetricsTermCfg(
      func=monitoring.joint_pos_metric,
      params={"asset_cfg": monitored_joints_cfg(), "component": comp},
    )
    metrics[f"qvel/{jname}"] = MetricsTermCfg(
      func=monitoring.joint_vel_metric,
      params={"asset_cfg": monitored_joints_cfg(), "component": comp},
    )
  metrics["power/total"] = MetricsTermCfg(
    func=monitoring.total_mechanical_power_metric,
    params={"asset_cfg": monitored_joints_cfg()},
  )

  # --- Slider (raw joint-space) ---
  metrics["slider/pos"] = MetricsTermCfg(
    func=monitoring.slider_pos_metric, params={"asset_cfg": slider_cfg()}
  )
  metrics["slider/vel"] = MetricsTermCfg(
    func=monitoring.slider_vel_metric, params={"asset_cfg": slider_cfg()}
  )
  metrics["slider/acc"] = MetricsTermCfg(
    func=monitoring.slider_acc_metric, params={"asset_cfg": slider_cfg()}
  )

  # --- Carriage (sign-corrected: +up = -slider) ---
  metrics["carriage/height"] = MetricsTermCfg(
    func=monitoring.carriage_height_metric, params={"asset_cfg": slider_cfg()}
  )
  metrics["carriage/vel"] = MetricsTermCfg(
    func=monitoring.carriage_vel_metric, params={"asset_cfg": slider_cfg()}
  )
  metrics["carriage/acc"] = MetricsTermCfg(
    func=monitoring.carriage_acc_metric, params={"asset_cfg": slider_cfg()}
  )

  return metrics


def _monitoring_recorder(run_tag: str = "run") -> dict[str, RecorderTermCfg]:
  """Build the CSV recorder term (one row per control step, env 0)."""
  return {
    "csv": RecorderTermCfg(
      func=monitoring.DiogenesCsvRecorder,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "env_idx": 0,
        "run_tag": run_tag,
        # "path": "logs/diogenes_monitor/my_run.csv",  # optional explicit path
      },
    )
  }
