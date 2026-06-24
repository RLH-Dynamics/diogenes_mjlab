"""SceneEntityCfg factory functions for the Diogenes hop stand.

Each manager term must get its OWN SceneEntityCfg instance, passed via
``params`` so the manager resolves it.
"""

from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..constants import DIOGENES_ACTUATOR_NAMES, FOOT_GEOM_NAME
from .. import monitoring


def slider_cfg() -> SceneEntityCfg:
  """Selects only the unactuated prismatic slider joint."""
  return SceneEntityCfg("robot", joint_names=("slider",))


def actuated_joints_cfg() -> SceneEntityCfg:
  """Selects the three actuated leg joints (excludes the slider)."""
  return SceneEntityCfg("robot", joint_names=DIOGENES_ACTUATOR_NAMES)


def actuators_cfg() -> SceneEntityCfg:
  """Selects the three position actuators (for the torque penalty)."""
  return SceneEntityCfg("robot", actuator_names=DIOGENES_ACTUATOR_NAMES)


def power_joints_cfg() -> SceneEntityCfg:
  """Selects the three actuated joints by NAME for the power penalty.

  ``electrical_power_cost`` resolves its ``asset_cfg`` via ``find_joints`` on the
  joint names, so this must carry ``joint_names`` (not ``actuator_names``).
  """
  return SceneEntityCfg("robot", joint_names=DIOGENES_ACTUATOR_NAMES)


def calf_body_cfg() -> SceneEntityCfg:
  """Selects the calf_assy body (the foot body) for foot-position tracking."""
  return SceneEntityCfg("robot", body_names=("calf_assy",))


def all_links_cfg() -> SceneEntityCfg:
  """Selects the moving leg links for mass / inertia / COM randomization.

  Excludes ``base_link`` (a ~zero-mass anchor) so we only perturb the real
  inertial bodies of the leg.
  """
  return SceneEntityCfg(
    "robot",
    body_names=("leg_mount_assy", "hip_assy", "thigh_assy", "calf_assy"),
  )


def foot_geom_cfg() -> SceneEntityCfg:
  """Selects the named foot collision geom for friction randomization."""
  return SceneEntityCfg("robot", geom_names=(FOOT_GEOM_NAME,))


def monitored_joints_cfg() -> SceneEntityCfg:
  """Selects the three actuated joints, order-preserved, for monitoring.

  Order matches ``monitoring.JOINT_NAMES`` = (hip, thigh, calf) so component
  indices line up with the metric/CSV column names.
  """
  return SceneEntityCfg(
    "robot", joint_names=monitoring.JOINT_NAMES, preserve_order=True
  )
