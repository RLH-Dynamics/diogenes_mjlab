from pathlib import Path

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.actuator.xml_actuator import XmlActuatorCfg

_HERE = Path(__file__).parent

DIOGENES_XML = _HERE / "xmls" / "diogenes.xml"

def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(DIOGENES_XML))

DIOGENES_ARTICULATION = EntityArticulationInfoCfg(
    actuators = (XmlActuatorCfg(target_names_expr=("hip","thigh","calf")),),
)

DEFAULT_INIT = EntityCfg.InitialStateCfg(
    pos=(0.0,0.0,0.0),
    joint_pos={
        "slider": 0.0,
        "hip": 0.0,
        "thigh": 0.0,
        "calf": 0.0,
    },
    joint_vel={".*": 0.0},
)

def get_diogenes_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        articulation=DIOGENES_ARTICULATION,
        init_state=DEFAULT_INIT
    )