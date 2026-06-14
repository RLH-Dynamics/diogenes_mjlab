"""Standalone joint-zero / axis-orientation inspector for the Diogenes leg.

Opens the STOCK MuJoCo passive viewer on the compiled scene -- no RL env, no
policy, no mjlab machinery -- so you can DRAG each joint and read its angle and
axis directly. This is the right tool for verifying joint zeros and axis
orientations by hand.

Run from the repo root (so the mesh assets resolve):

    python inspect_joints.py
    python inspect_joints.py --xml src/diogenes_mjlab/diogenes/xmls/scene.xml
    python inspect_joints.py --freeze-slider     # weld the carriage so only
                                                 # hip/thigh/calf move

HOW TO INSPECT ONCE THE VIEWER IS OPEN
--------------------------------------
1. Open the LEFT panel (press Tab if it's hidden, or the ">" at top-left).
2. Find the "Joint" section: it has ONE SLIDER PER JOINT (slider, hip, thigh,
   calf). Drag a slider -> that joint moves to that angle, live. The number IS
   the joint value in radians. Set it to 0 to see that joint's ZERO pose.
3. To SEE THE AXIS each joint rotates about: open the right-side "Rendering"
   panel -> "Model Elements" and enable "Joint" (draws a joint marker/axis at
   each joint). You can also enable "Actuator" to confirm wiring. Frames ->
   "Body" draws each body's coordinate frame so you can read axis orientation
   in WORLD coordinates.
4. The viewer starts PAUSED with gravity OFF (see below), so nothing drifts
   while you pose joints. Joints hold exactly where you drag them.

WHY GRAVITY IS DISABLED
-----------------------
With gravity on, an unactuated/zeroed leg would sag the instant you unpause,
fighting your inspection. We zero gravity and keep the sim paused so the model
behaves as a pure kinematic rig: whatever you set the joint sliders to is what
you see. (This script never applies control, so even unpaused the only motion
would be from gravity, which we remove.)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument(
    "--xml",
    default="src/diogenes_mjlab/diogenes/xmls/scene.xml",
    help="Path to the scene/model XML (default: the Diogenes scene).",
  )
  ap.add_argument(
    "--freeze-slider",
    action="store_true",
    help="Pin the prismatic slider at 0 so only the 3 leg joints move "
         "(useful to inspect hip/thigh/calf without the carriage sliding).",
  )
  ap.add_argument(
    "--gravity",
    action="store_true",
    help="Leave gravity ON (default OFF for static posing).",
  )
  args = ap.parse_args()

  xml_path = Path(args.xml)
  if not xml_path.exists():
    raise SystemExit(f"XML not found: {xml_path.resolve()}\n"
                     f"Run from the repo root so meshes under 'assets' resolve.")

  model = mujoco.MjModel.from_xml_path(str(xml_path))

  # Disable gravity for clean static posing unless asked otherwise.
  if not args.gravity:
    model.opt.gravity[:] = 0.0

  # Optionally pin the slider at 0 by clamping its range to a point. This is a
  # soft pin (the joint slider in the GUI will be limited to [0,0]); the leg
  # joints stay fully free.
  if args.freeze_slider:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
    if sid >= 0:
      model.jnt_range[sid] = (0.0, 0.0)
      model.jnt_limited[sid] = 1

  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  # Print the ground-truth table so you can cross-check what you see on screen.
  print("=" * 64)
  print("DIOGENES JOINT / ACTUATOR REFERENCE (drag sliders to verify)")
  print("=" * 64)
  print(f"{'id':>2}  {'joint':<8} {'type':<5} {'axis(local)':<14} "
        f"{'range[lo, hi]':<22}")
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jt = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
    axis = np.array2string(model.jnt_axis[i], precision=2, suppress_small=True)
    rng = model.jnt_range[i]
    print(f"{i:>2}  {name:<8} {jt:<5} {axis:<14} "
          f"[{rng[0]:+.4f}, {rng[1]:+.4f}]")
  print()
  print("All joint zeros (qpos0):",
        np.array2string(model.qpos0, precision=4, suppress_small=True))
  print("Gravity:", "ON" if args.gravity else "OFF (static posing)")
  print("Slider:", "PINNED at 0" if args.freeze_slider else "free")
  print("=" * 64)
  print("Viewer opens PAUSED. Left panel -> 'Joint' sliders to pose joints.")
  print("Right panel -> Rendering -> Model Elements: enable 'Joint' to see "
        "axes,\n  and Frames -> 'Body' to read axis orientation in world frame.")
  print("=" * 64)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    # Enable joint-axis visualization and body frames by default so the user
    # immediately sees each joint's axis without hunting through menus.
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    viewer.sync()

    # Keep the process alive; do NOT step physics (pure kinematic posing).
    # mj_forward is re-run so any GUI joint edits propagate to body poses.
    while viewer.is_running():
      mujoco.mj_forward(model, data)
      viewer.sync()
      time.sleep(0.02)


if __name__ == "__main__":
  main()
