from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.actuator.xml_actuator import XmlActuatorCfg

_HERE = Path(__file__).parent

# Load the scene (which `include`s harold_biped.xml). MuJoCo resolves both the
# <include> and the compiler's meshdir="assets" relative to this file's
# directory, so no explicit asset loading is needed when compiling from a path.
HAROLD_BIPED_XML = _HERE / "xmls" / "scene.xml"

# Root body of the kinematic tree; carries the freejoint in the exported XML.
ROOT_BODY_NAME = "body_assy"

# The six actuated joints (== XML <position> actuator names), in XML order.
HAROLD_JOINT_NAMES: tuple[str, ...] = (
    "left_hip",
    "left_thigh",
    "left_calf",
    "right_hip",
    "right_thigh",
    "right_calf",
)

# BNO085 IMU site, from the Onshape mate connector "frame_bno085" at the centre
# of the chip, with the chip's own axes: +x = robot left, +y = up, +z = forward.
# It sits on body_assy, so the walking model's standard-frame rotation moves it
# with the torso and leaves its axes pointing the same physical directions.
IMU_SITE_NAME = "bno085"

# Base height (m) for the floating-base (walking) model. At the zero joint pose
# the lowest vertex of the calf (foot) collision meshes sits 0.4097 m below the
# base origin -- ~20 mm below the left_foot/right_foot sites -- so this spawns
# the foot tips ~2 mm above the ground. Spawning lower starts the feet inside
# the floor and the contact solver launches the robot upward.
WALK_BASE_HEIGHT = 0.412

# Base height (m) for the suspended (fixed-base) model. Leaves ~0.6 m of
# clearance under the feet so the legs can swing freely.
SUSPENDED_BASE_HEIGHT = 1.0

# The exported body_assy frame is +x = robot left, +y = robot BACK, +z = up.
# mjlab's velocity-task terms assume +x = forward, +y = left, +z = up, so the
# walking model re-expresses the root body's contents in that standard frame:
# a +90 deg rotation about z (v_standard = R @ v_exported).
_STANDARD_FROM_EXPORTED = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)
_STANDARD_FROM_EXPORTED_QUAT = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])

# Point-foot colliders, in the calf body frame (identical for both calves; the
# knee joint sits at the calf body origin). Derived from the calf mesh: the
# knee-to-tip axis runs 0.2704 m to the farthest vertex.
CALF_BODY_NAMES: dict[str, str] = {"left": "l_calf_assy", "right": "r_calf_assy"}
_CALF_AXIS = np.array([-0.86457, -0.49916, -0.058])
_CALF_LENGTH = 0.2704
# Tip sphere: tangent to the tip vertex, radius of the mesh near the tip.
FOOT_TIP_RADIUS = 0.0205
FOOT_TIP_CENTER = _CALF_AXIS * (_CALF_LENGTH - FOOT_TIP_RADIUS)
# Shank capsule: 3-19 cm along the axis. Its end is 60 mm short of the tip
# sphere's centre and 9.5 mm fatter, so it only reaches the ground before the
# tip when the calf is within ~9 deg of lying flat.
SHANK_RADIUS = 0.030
SHANK_FROMTO = np.concatenate([_CALF_AXIS * 0.03, _CALF_AXIS * 0.19])


def _to_standard_frame(spec: mujoco.MjSpec) -> None:
    """Rotate the root body's frame so +x is forward and +y is left.

    Only the root body's own frame changes: its geoms, sites, child bodies and
    inertia are re-expressed so every part keeps its place relative to the
    others. Joint names, axes (in their own bodies) and limits are unchanged.
    """
    root = spec.body(ROOT_BODY_NAME)
    rot = _STANDARD_FROM_EXPORTED
    for element in [*root.geoms, *root.sites, *root.bodies]:
        assert element.alt.type == mujoco.mjtOrientation.mjORIENTATION_QUAT, (
            f"{element.name!r} uses an alternative orientation; only quat is handled."
        )
        element.pos = rot @ np.asarray(element.pos)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, _STANDARD_FROM_EXPORTED_QUAT, np.asarray(element.quat))
        element.quat = quat

    root.ipos = rot @ np.asarray(root.ipos)
    full = np.asarray(root.fullinertia)
    if np.isnan(full[0]):
        iquat = np.zeros(4)
        mujoco.mju_mulQuat(iquat, _STANDARD_FROM_EXPORTED_QUAT, np.asarray(root.iquat))
        root.iquat = iquat
    else:
        # fullinertia = (ixx, iyy, izz, ixy, ixz, iyz).
        ixx, iyy, izz, ixy, ixz, iyz = full
        tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        t = rot @ tensor @ rot.T
        root.fullinertia = [t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2]]


def _add_point_feet(spec: mujoco.MjSpec) -> None:
    """Replace each calf-mesh foot collider with a tip sphere plus a shank capsule.

    The exported ``<side>_foot`` collider is the whole calf mesh, which lets a
    walking policy rest the calf flat on the ground as a long, stable support.
    Afterwards:
      * ``<side>_calf_mesh``: the old mesh collider, contacts disabled;
      * ``<side>_foot``: a sphere at the calf tip, the only intended contact;
      * ``<side>_shank``: a capsule along the calf, for detecting (and ending
        episodes on) shank-ground contact.
    Both new geoms use the old foot's contact bits (floor only, no
    self-collision) and friction, so foot names used by sensors and DR are kept.
    """
    for side, body_name in CALF_BODY_NAMES.items():
        mesh = spec.geom(f"{side}_foot")
        contype, conaffinity = mesh.contype, mesh.conaffinity
        friction, condim = np.asarray(mesh.friction).copy(), mesh.condim
        mesh.name = f"{side}_calf_mesh"
        mesh.contype = 0
        mesh.conaffinity = 0

        body = spec.body(body_name)
        common = dict(
            contype=contype, conaffinity=conaffinity, group=3,
            friction=friction, condim=condim,
        )
        body.add_geom(
            name=f"{side}_foot",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[FOOT_TIP_RADIUS, 0.0, 0.0],
            pos=FOOT_TIP_CENTER,
            **common,
        )
        body.add_geom(
            name=f"{side}_shank",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[SHANK_RADIUS, 0.0, 0.0],
            fromto=SHANK_FROMTO,
            **common,
        )


def get_spec(
    fixed_base: bool = False, standard_frame: bool = False, point_feet: bool = False
) -> mujoco.MjSpec:
    """Build the biped spec.

    Args:
      fixed_base: If True, delete the root freejoint so the torso is welded to
        the world (task 1: legs track a trajectory while suspended). mjlab then
        places the root body at ``init_state.pos``. If False, keep the
        freejoint (task 2: walking on the ground plane).
      standard_frame: If True, re-express the root body in the standard
        +x forward / +y left / +z up frame (see ``_to_standard_frame``). The
        suspended gait in ``gait.py`` is written in the exported frame, so this
        is only used by the walking task.
      point_feet: If True, collide through a tip sphere and shank capsule
        instead of the full calf mesh (see ``_add_point_feet``). Walking only.
    """
    spec = mujoco.MjSpec.from_file(str(HAROLD_BIPED_XML))
    if standard_frame:
        _to_standard_frame(spec)
    if point_feet:
        _add_point_feet(spec)
    if fixed_base:
        root = spec.body(ROOT_BODY_NAME)
        for joint in list(root.joints):
            spec.delete(joint)
    return spec


# Match the six joints explicitly: a looser "left_.*" would also match the
# left_foot / right_foot sites and trigger mjlab's site-transmission warning.
HAROLD_BIPED_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(target_names_expr=("(left|right)_(hip|thigh|calf)",)),
    ),
)


def _init_state(base_height: float) -> EntityCfg.InitialStateCfg:
    return EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, base_height),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )


def get_harold_biped_cfg(
    fixed_base: bool = False, standard_frame: bool = False, point_feet: bool = False
) -> EntityCfg:
    base_height = SUSPENDED_BASE_HEIGHT if fixed_base else WALK_BASE_HEIGHT
    return EntityCfg(
        spec_fn=lambda: get_spec(
            fixed_base=fixed_base, standard_frame=standard_frame, point_feet=point_feet
        ),
        articulation=HAROLD_BIPED_ARTICULATION,
        init_state=_init_state(base_height),
    )
