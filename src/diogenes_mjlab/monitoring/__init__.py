"""Real-time monitoring + CSV logging for the Diogenes hop stand.

This package replaces the old ``monitoring.py`` module. All public names are
re-exported here so existing access paths like ``monitoring.JOINT_NAMES``,
``monitoring.DiogenesCsvRecorder``, and ``monitoring.<metric_fn>`` continue to
resolve unchanged.

Sub-modules
-----------
* metrics   -- stateless scalar metric functions (Viser live plots)
* recorder  -- DiogenesCsvRecorder (CSV logging, one row per control step)
"""

from .recorder import (  # noqa: F401
  JOINT_NAMES,
  DiogenesCsvRecorder,
  CSV_HEADER,
)

from .metrics import (  # noqa: F401
  joint_torque_metric,
  joint_power_metric,
  total_mechanical_power_metric,
  joint_pos_metric,
  joint_vel_metric,
  carriage_height_metric,
  carriage_vel_metric,
  carriage_acc_metric,
  slider_pos_metric,
  slider_vel_metric,
  slider_acc_metric,
  contact_force_component_metric,
  contact_force_magnitude_metric,
)
