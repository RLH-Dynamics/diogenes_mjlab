"""Mirror the real robot's joint angles onto the suspended Harold model in Viser.

Pairs with ``diogenes_control/tools/stream_joints.py`` on the Pi, which streams
the SIM-frame joint angles (hardware reading * direction sign) the policy would
see. Move a joint by hand and the matching link here should move the same way.

  * The link a joint drives turns YELLOW while that joint is moving, so you can
    see which model link each physical joint is wired to.
  * A link turns RED when its joint reads outside the MJCF joint range. The hip
    and thigh ranges are strongly asymmetric, so a wrong direction sign usually
    shows up as red as soon as the joint moves.
  * Untick "Live" to pose the model with sliders instead, e.g. to find what a
    given sim angle looks like before comparing it with the robot.
  * With `stream_joints.py --imu`, the whole model tilts with the torso as the
    BNO085 reports it (through config.IMU.mount_rotation), with pitch, roll,
    gyro and accuracy in the IMU panel. The heading is zeroed at the first
    packet ("Reset heading" to redo): the IMU's absolute yaw comes from an
    uncalibrated magnetometer and means nothing here, but turns still show.

Standalone: needs only mujoco, viser and numpy (no mjlab/torch), so run it by
path rather than as a package module:

    python src/diogenes_mjlab/tools/live_joint_viewer.py
    # then open http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import viser

XML_PATH = Path(__file__).resolve().parents[1] / "harold_biped" / "xmls" / "scene.xml"
ROOT_BODY_NAME = "body_assy"
# Matches SUSPENDED_BASE_HEIGHT in harold_biped_constants.py.
SUSPENDED_BASE_HEIGHT = 1.0
JOINT_NAMES = (
  "left_hip", "left_thigh", "left_calf",
  "right_hip", "right_thigh", "right_calf",
)
DEFAULT_PORT = 9870

# v_standard = STANDARD_FROM_EXPORTED @ v_exported (harold_biped_constants.py).
# The IMU stream is in the standard base frame; this model is in the exported one.
STANDARD_FROM_EXPORTED = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

MOVING_VEL = 0.3  # rad/s above which a joint counts as "being moved"
STALE_S = 0.5
YELLOW = (255, 200, 0)
RED = (230, 40, 40)


def build_model() -> mujoco.MjModel:
  """The suspended (fixed-base) model, as get_spec(fixed_base=True) builds it."""
  spec = mujoco.MjSpec.from_file(str(XML_PATH))
  root = spec.body(ROOT_BODY_NAME)
  for joint in list(root.joints):
    spec.delete(joint)
  root.pos = [0.0, 0.0, SUSPENDED_BASE_HEIGHT]
  return spec.compile()


def body_meshes(model: mujoco.MjModel) -> dict[int, list[tuple[tuple, np.ndarray, np.ndarray]]]:
  """Visual meshes merged per (body, colour), in each body's own frame."""
  merged: dict[tuple[int, tuple], list[tuple[np.ndarray, np.ndarray]]] = {}
  rot = np.zeros(9)
  for g in range(model.ngeom):
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 2:
      continue
    m = model.geom_dataid[g]
    va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
    fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    verts = model.mesh_vert[va:va + vn] @ rot.reshape(3, 3).T + model.geom_pos[g]
    faces = model.mesh_face[fa:fa + fn]
    mat = model.geom_matid[g]
    rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
    color = tuple(int(255 * c) for c in rgba[:3])
    merged.setdefault((int(model.geom_bodyid[g]), color), []).append((verts, faces))

  out: dict[int, list] = {}
  for (body, color), parts in merged.items():
    offsets = np.cumsum([0] + [len(v) for v, _ in parts[:-1]])
    verts = np.concatenate([v for v, _ in parts]).astype(np.float32)
    faces = np.concatenate([f + o for (_, f), o in zip(parts, offsets)]).astype(np.uint32)
    out.setdefault(body, []).append((color, verts, faces))
  return out


class Receiver(threading.Thread):
  """Keeps the latest packet from the Pi streamer."""

  def __init__(self, port: int):
    super().__init__(daemon=True)
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.bind(("0.0.0.0", port))
    self.lock = threading.Lock()
    self.latest: dict | None = None
    self.sender = ""
    self.received_at = 0.0
    self.count = 0

  def run(self):
    while True:
      data, addr = self.sock.recvfrom(65536)
      try:
        packet = json.loads(data)
      except json.JSONDecodeError:
        continue
      with self.lock:
        self.latest = packet
        self.sender = addr[0]
        self.received_at = time.monotonic()
        self.count += 1

  def snapshot(self) -> tuple[dict | None, float, str, int]:
    with self.lock:
      return self.latest, time.monotonic() - self.received_at, self.sender, self.count


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP port to listen on")
  parser.add_argument("--viser-port", type=int, default=8080)
  args = parser.parse_args()

  model = build_model()
  data = mujoco.MjData(model)
  joint_ids = {n: model.joint(n).id for n in JOINT_NAMES}
  qadr = {n: model.jnt_qposadr[j] for n, j in joint_ids.items()}
  jrange = {n: model.jnt_range[j] for n, j in joint_ids.items()}
  joint_body = {n: int(model.jnt_bodyid[j]) for n, j in joint_ids.items()}

  server = viser.ViserServer(port=args.viser_port, label="Harold live joints")
  server.scene.add_grid("/floor", width=2.0, height=2.0, cell_size=0.1)
  # The exported frame is +x = robot left, +y = robot BACK, so this looks at the
  # robot's front, slightly from its right.
  server.initial_camera.look_at = (0.0, 0.0, SUSPENDED_BASE_HEIGHT - 0.2)
  server.initial_camera.position = (-0.6, -1.1, SUSPENDED_BASE_HEIGHT + 0.1)

  # Every link hangs off one frame at the torso, so the IMU can tilt the lot.
  base_pos = np.array([0.0, 0.0, SUSPENDED_BASE_HEIGHT])
  robot_frame = server.scene.add_frame("/robot", show_axes=False, position=base_pos)
  frames: dict[int, viser.FrameHandle] = {}
  meshes: dict[int, list[tuple[viser.MeshHandle, tuple]]] = {}
  for body, parts in body_meshes(model).items():
    frames[body] = server.scene.add_frame(f"/robot/b{body}", show_axes=False)
    meshes[body] = [
      (server.scene.add_mesh_simple(f"/robot/b{body}/m{i}", v, f, color=c), c)
      for i, (c, v, f) in enumerate(parts)
    ]
  labels = {
    n: server.scene.add_label(f"/robot/labels/{n}", n, font_screen_scale=0.8)
    for n in JOINT_NAMES
  }

  status = server.gui.add_markdown("Waiting for the Pi…")
  live = server.gui.add_checkbox("Live", True, hint="Untick to pose the model by hand")
  show_labels = server.gui.add_checkbox("Joint labels", True)
  sliders = {}
  with server.gui.add_folder("Joints (sim frame, deg)"):
    for n in JOINT_NAMES:
      lo, hi = np.degrees(jrange[n])
      sliders[n] = server.gui.add_slider(
        n, min=-180.0, max=180.0, step=0.1, initial_value=0.0,
        hint=f"MJCF range {lo:+.1f}° … {hi:+.1f}°",
      )
  readout = server.gui.add_markdown("")
  zero = server.gui.add_button("Zero sliders")

  @zero.on_click
  def _(_):
    for s in sliders.values():
      s.value = 0.0

  with server.gui.add_folder("IMU (base: +x fwd, +y left, +z up)"):
    imu_md = server.gui.add_markdown("")
    follow_imu = server.gui.add_checkbox("Tilt model with IMU", True)
    reset_heading = server.gui.add_button("Reset heading")
  heading = {"yaw0": None}

  @reset_heading.on_click
  def _(_):
    heading["yaw0"] = None

  receiver = Receiver(args.port)
  receiver.start()
  print(f"Listening for joint packets on UDP {args.port}; "
        f"open http://localhost:{args.viser_port}")

  body_color: dict[int, tuple | None] = {}
  while True:
    packet, age, sender, count = receiver.snapshot()
    fresh = packet is not None and age < STALE_S
    joints = packet["joints"] if fresh else {}

    missing = [n for n in JOINT_NAMES if n not in joints]
    if live.value and fresh:
      for n in JOINT_NAMES:
        if n in joints:
          sliders[n].value = round(float(np.degrees(joints[n]["sim"])), 1)

    q = {n: np.radians(sliders[n].value) for n in JOINT_NAMES}
    for n in JOINT_NAMES:
      data.qpos[qadr[n]] = q[n]
    mujoco.mj_kinematics(model, data)

    # Per-link highlight: red beats yellow beats the CAD colour.
    wanted: dict[int, tuple | None] = {b: None for b in meshes}
    for n in JOINT_NAMES:
      lo, hi = jrange[n]
      b = joint_body[n]
      if not lo - 1e-3 <= q[n] <= hi + 1e-3:
        wanted[b] = RED
      elif live.value and n in joints and abs(joints[n]["vel"]) > MOVING_VEL:
        wanted[b] = wanted[b] or YELLOW

    # Torso orientation from the IMU, heading zeroed, in the model's frame.
    imu = packet.get("imu") if fresh else None
    robot_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    if imu is not None and follow_imu.value:
      r_wb = np.array(imu["R_world_base"]).reshape(3, 3)
      if heading["yaw0"] is None:
        heading["yaw0"] = float(np.arctan2(r_wb[1, 0], r_wb[0, 0]))
      c, s = np.cos(-heading["yaw0"]), np.sin(-heading["yaw0"])
      r_wb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ r_wb
      r_view = STANDARD_FROM_EXPORTED.T @ r_wb @ STANDARD_FROM_EXPORTED
      mujoco.mju_mat2Quat(robot_wxyz, r_view.ravel())

    with server.atomic():
      robot_frame.wxyz = robot_wxyz
      for b, frame in frames.items():
        frame.position = data.xpos[b] - base_pos
        frame.wxyz = data.xquat[b]
        if body_color.get(b, "unset") != wanted[b]:
          body_color[b] = wanted[b]
          for handle, cad in meshes[b]:
            handle.color = wanted[b] or cad
      for n, label in labels.items():
        label.visible = show_labels.value
        label.position = data.xanchor[joint_ids[n]] - base_pos

    # Status + table.
    if packet is None:
      status.content = f"**Waiting for the Pi** on UDP {args.port}…"
    elif not fresh:
      status.content = f"⚠️ **Stale** — last packet from {sender} {age:.1f} s ago"
    else:
      status.content = f"🟢 **Live** from {sender} · {count} packets"
      if missing:
        status.content += f"\n\n⚠️ Not in stream: {', '.join(missing)}"
    rows = ["| joint | bus:id | hw ° | sim ° | range ° |", "|---|---|--:|--:|---|"]
    for n in JOINT_NAMES:
      lo, hi = np.degrees(jrange[n])
      flag = " 🔴" if not lo - 0.06 <= np.degrees(q[n]) <= hi + 0.06 else ""
      j = joints.get(n)
      bus = f"{j['bus']}:{j['id']}" if j else "–"
      hw = f"{np.degrees(j['hw']):+.1f}" if j else "–"
      rows.append(f"| {n} | {bus} | {hw} | {np.degrees(q[n]):+.1f}{flag} | "
                  f"{lo:+.0f} … {hi:+.0f} |")
    readout.content = "\n".join(rows)

    if imu is None:
      imu_md.content = ("No IMU in the stream (start `stream_joints.py --imu`)."
                        if fresh else "")
    else:
      g, w = np.array(imu["gravity"]), np.array(imu["gyro"])
      pitch = np.degrees(np.arcsin(np.clip(g[0], -1, 1)))
      roll = np.degrees(np.arcsin(np.clip(g[1], -1, 1)))
      imu_md.content = (
        f"| | |\n|---|--:|\n"
        f"| pitch (nose down +) | {pitch:+.1f}° |\n"
        f"| roll (left down +) | {roll:+.1f}° |\n"
        f"| tilt | {imu['tilt_deg']:.1f}° |\n"
        f"| gravity | {g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f} |\n"
        f"| gyro rad/s | {w[0]:+.2f} {w[1]:+.2f} {w[2]:+.2f} |\n"
        f"| accuracy · age | {imu['status']}/3 · {imu['age_ms']:.0f} ms |")

    time.sleep(1 / 30)


if __name__ == "__main__":
  main()
