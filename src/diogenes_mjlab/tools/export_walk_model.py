# export_walk_model.py  -- run in the TRAINING env
"""Save the walking task's robot as a standalone MuJoCo model for sim-to-sim.

    uv run python src/diogenes_mjlab/tools/export_walk_model.py harold_walk.mjb

Writes the Diogenes-Biped-Walk robot -- standard frame, point feet, the RS03
actuators as plain torque motors, nominal armature and friction, the crouched
init keyframe -- on a floor plane, as a binary .mjb (loads in any MuJoCo of
the same version without mjlab or the meshes), plus a .json beside it with what
the runner needs: joint order, PD gains, torque-speed limit, crouch pose.

diogenes_control/tools/sim2sim_walk.py then runs the control stack's own
policy and observation code against it.
"""

import argparse
import json
from pathlib import Path

import mujoco

from diogenes_mjlab.harold_biped import actuators as rs03
from diogenes_mjlab.harold_biped.harold_biped_constants import HAROLD_JOINT_NAMES, IMU_SITE_NAME
from diogenes_mjlab.harold_biped.walk_env_cfg import (
  CROUCH_JOINT_POS,
  FOOT_GEOM_NAMES,
  _robot_cfg,
)


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("out", type=Path, help="Output .mjb path.")
  args = p.parse_args()

  spec = _robot_cfg().build().spec
  foot = spec.geom(FOOT_GEOM_NAMES[0])
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
    contype=foot.conaffinity, conaffinity=foot.contype, friction=foot.friction,
  )
  spec.option.timestep = 0.002
  model = spec.compile()
  for j in HAROLD_JOINT_NAMES:
    assert model.actuator(j).trnid[0] == model.joint(j).id, f"actuator {j} drives another joint"

  args.out.parent.mkdir(parents=True, exist_ok=True)
  mujoco.mj_saveModel(model, str(args.out), None)
  info = {
    "joint_names": list(HAROLD_JOINT_NAMES),
    "crouch": CROUCH_JOINT_POS,
    "imu_site": IMU_SITE_NAME,
    "keyframe": "init_state",
    "kp": rs03.KP,
    "kd": rs03.KD,
    "peak_torque": rs03.RS03_PEAK_TORQUE,
    "saturation_torque": rs03.RS03_SATURATION_TORQUE,
    "no_load_speed": rs03.no_load_speed(rs03.BATTERY_NOMINAL_VOLTAGE),
    "physics_dt": model.opt.timestep,
  }
  args.out.with_suffix(".json").write_text(json.dumps(info, indent=1))
  print(f"[OK] wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
  main()
