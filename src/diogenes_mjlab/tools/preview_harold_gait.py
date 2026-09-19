"""Preview the Harold biped suspended gait in the MuJoCo viewer (no RL).

Welds the torso at the suspended height, converts the foot reference loops from
``harold_biped/gait.py`` into joint targets with damped-least-squares IK, and
drives the XML position actuators with them at the 50 Hz control rate. Each
reference loop is drawn as small grey spheres, with the current target on it in
red, so you can see how closely the actuators alone follow the gait.

Run from the repo root:

    uv run python -m diogenes_mjlab.tools.preview_harold_gait
    uv run python -m diogenes_mjlab.tools.preview_harold_gait --headless   # stats only
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from diogenes_mjlab.harold_biped import gait
from diogenes_mjlab.harold_biped.harold_biped_constants import (
  ROOT_BODY_NAME,
  get_harold_biped_cfg,
)

# Joints that move each foot site.
LEG_JOINTS: dict[str, tuple[str, ...]] = {
  "left_foot": ("left_hip", "left_thigh", "left_calf"),
  "right_foot": ("right_hip", "right_thigh", "right_calf"),
}

CONTROL_DT = 0.02  # 50 Hz, matches the env's decimation * timestep


def build_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
  """Compile the suspended biped exactly as the env builds it, at its init pose."""
  model = get_harold_biped_cfg(fixed_base=True).build().spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  mujoco.mj_forward(model, data)
  return model, data


def reference_world(data: mujoco.MjData, t: float) -> np.ndarray:
  """World-frame foot targets at time t, shape (num_feet, 3)."""
  phase = torch.tensor([(t / gait.GAIT_PERIOD) % 1.0], dtype=torch.float64)
  pos_b, _ = gait.foot_reference(
    gait.leg_phases(phase), gait.foot_centers(dtype=torch.float64)
  )
  base = data.body(ROOT_BODY_NAME)
  rot = base.xmat.reshape(3, 3)
  return pos_b[0].numpy() @ rot.T + base.xpos


def solve_ik(
  model: mujoco.MjModel,
  scratch: mujoco.MjData,
  q_init: np.ndarray,
  targets_w: np.ndarray,
  iters: int = 20,
  damping: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
  """Damped-least-squares IK for both feet, clamped to joint limits.

  Args:
    scratch: MjData used only for kinematics (the sim data is left untouched).
    q_init: warm-start qpos, shape (nq,).
    targets_w: world-frame foot targets, shape (num_feet, 3).

  Returns:
    (qpos, residual) with residual the per-foot position error in m.
  """
  scratch.qpos[:] = q_init
  jacp = np.zeros((3, model.nv))
  for _ in range(iters):
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    for i, site in enumerate(gait.FOOT_SITE_NAMES):
      err = targets_w[i] - scratch.site(site).xpos
      mujoco.mj_jacSite(model, scratch, jacp, None, model.site(site).id)
      joint_ids = [model.joint(name).id for name in LEG_JOINTS[site]]
      jac = jacp[:, model.jnt_dofadr[joint_ids]]
      dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), err)
      for jid, step in zip(joint_ids, dq):
        adr = model.jnt_qposadr[jid]
        scratch.qpos[adr] = np.clip(scratch.qpos[adr] + step, *model.jnt_range[jid])
  mujoco.mj_kinematics(model, scratch)
  residual = np.array([
    np.linalg.norm(targets_w[i] - scratch.site(site).xpos)
    for i, site in enumerate(gait.FOOT_SITE_NAMES)
  ])
  return scratch.qpos.copy(), residual


def _add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba) -> None:
  if scene.ngeom >= scene.maxgeom:
    return
  mujoco.mjv_initGeom(
    scene.geoms[scene.ngeom],
    mujoco.mjtGeom.mjGEOM_SPHERE,
    np.array([radius, 0.0, 0.0]),
    np.asarray(pos, dtype=np.float64),
    np.eye(3).flatten(),
    np.asarray(rgba, dtype=np.float32),
  )
  scene.ngeom += 1


def run(duration: float, show_viewer: bool = False) -> dict[str, float | tuple]:
  """Drive the gait for ``duration`` seconds (or until the viewer is closed).

  Returns tracking statistics. The first gait cycle is excluded from the
  actuator tracking error so the start-up transient doesn't dominate it.
  """
  model, data = build_model()
  scratch = mujoco.MjData(model)
  n_substeps = round(CONTROL_DT / model.opt.timestep)
  act_qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]

  # Start the legs on the reference so the preview doesn't open with a lunge.
  q, _ = solve_ik(model, scratch, data.qpos.copy(), reference_world(data, 0.0), iters=200)
  data.qpos[:] = q
  data.ctrl[:] = q[act_qadr]
  mujoco.mj_forward(model, data)

  loop_pts = np.stack([
    reference_world(data, t)
    for t in np.linspace(0.0, gait.GAIT_PERIOD, 64, endpoint=False)
  ]).reshape(-1, 3)

  ik_res_max = 0.0
  track_err: list[np.ndarray] = []
  q_min = np.full(model.nq, np.inf)
  q_max = np.full(model.nq, -np.inf)

  viewer = mujoco.viewer.launch_passive(model, data) if show_viewer else None
  try:
    while (viewer is None and data.time < duration) or (
      viewer is not None and viewer.is_running()
    ):
      wall_start = time.perf_counter()

      q, residual = solve_ik(model, scratch, q, reference_world(data, data.time))
      ik_res_max = max(ik_res_max, float(residual.max()))
      data.ctrl[:] = q[act_qadr]
      for _ in range(n_substeps):
        mujoco.mj_step(model, data)

      targets = reference_world(data, data.time)
      feet = np.stack([data.site(s).xpos for s in gait.FOOT_SITE_NAMES])
      if data.time >= gait.GAIT_PERIOD:
        track_err.append(np.linalg.norm(feet - targets, axis=-1))
      q_min = np.minimum(q_min, data.qpos)
      q_max = np.maximum(q_max, data.qpos)

      if viewer is not None:
        with viewer.lock():
          scene = viewer.user_scn
          scene.ngeom = 0
          for p in loop_pts:
            _add_sphere(scene, p, 0.004, (0.8, 0.8, 0.8, 0.8))
          for p in targets:
            _add_sphere(scene, p, 0.012, (1.0, 0.1, 0.1, 1.0))
        viewer.sync()
        time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - wall_start)))
  finally:
    if viewer is not None:
      viewer.close()

  err = np.array(track_err) if track_err else np.full((1, len(gait.FOOT_SITE_NAMES)), np.nan)
  margin = np.minimum(q_min - model.jnt_range[:, 0], model.jnt_range[:, 1] - q_max)
  stats: dict[str, float | tuple] = {
    "ik_residual_max_mm": 1e3 * ik_res_max,
    "min_limit_margin_deg": float(np.degrees(margin.min())),
  }
  for i, site in enumerate(gait.FOOT_SITE_NAMES):
    stats[f"{site}_track_err_mean_mm"] = 1e3 * float(err[:, i].mean())
    stats[f"{site}_track_err_max_mm"] = 1e3 * float(err[:, i].max())
  for j in range(model.njnt):
    stats[f"{model.joint(j).name}_range_deg"] = (
      round(float(np.degrees(q_min[j])), 1), round(float(np.degrees(q_max[j])), 1)
    )
  return stats


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--headless", action="store_true", help="print stats, no viewer")
  parser.add_argument(
    "--cycles", type=float, default=3.0, help="gait cycles to run with --headless"
  )
  args = parser.parse_args()

  stats = run(args.cycles * gait.GAIT_PERIOD, show_viewer=not args.headless)
  for key, value in stats.items():
    print(f"{key}: {value}")


if __name__ == "__main__":
  main()
