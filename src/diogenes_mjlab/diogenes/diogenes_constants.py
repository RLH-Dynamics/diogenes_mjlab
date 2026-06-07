from pathlib import Path

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.actuator.xml_actuator import XmlActuatorCfg

_HERE = Path(__file__).parent

# Load the scene (which `include`s diogenes.xml) so the world gets a ground
# plane for the leg to contact. MuJoCo resolves both the <include> and the
# compiler's meshdir="assets" relative to this file's directory, so no explicit
# asset loading is needed when compiling from a path on disk.
DIOGENES_XML = _HERE / "xmls" / "scene.xml"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(DIOGENES_XML))


DIOGENES_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=("hip", "thigh", "calf")),),
)

DEFAULT_INIT = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "slider": 0.0,
        "hip": 0.0,
        "thigh": -0.0,
        "calf": 0.0,
    },
    joint_vel={".*": 0.0},
)


def get_diogenes_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        articulation=DIOGENES_ARTICULATION,
        init_state=DEFAULT_INIT,
    )
