# eval_walk.py  -- run in the TRAINING env
"""Score a Diogenes-Biped-Walk checkpoint on a fixed test set.

    uv run python src/diogenes_mjlab/tools/eval_walk.py \\
        logs/rsl_rl/harold_biped_walk/<run>/model_<N>.pt [--step-freq 1.4]

Every env runs ONE episode of --duration seconds from a fresh reset, with its
command held fixed for the whole episode (no resampling), so an episode either
survives to the end or fails once. Three scenarios, each with observation noise
and delay on and the robot's physical parameters redrawn per env:

  nominal   DR at the training scale (1.0), no pushes
  hard      DR widened to 1.5x, no pushes
  push      DR 1.0, a push every 1-2 s, twice the training push size

The command grid (vx x yaw rate) is shared out evenly over the envs. For each
scenario it reports the survival rate, the episode-mean speed and yaw-rate error
of survivors (after a 2 s settle), lateral drift, the contact-schedule match,
torque (RMS and peak), peak joint speed, mechanical power and the smallest
distance any joint came to its hard stop. Per-command survival and tracking are
printed for the nominal scenario, and everything is written as JSON next to the
checkpoint.

The checkpoint must match the current env code (observation layout).
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

import mjlab.tasks  # noqa: F401  (populate registry)
import diogenes_mjlab  # noqa: F401  (register the Diogenes/Harold tasks)
from diogenes_mjlab.harold_biped import gait
from diogenes_mjlab.harold_biped import mdp as harold_mdp
from diogenes_mjlab.harold_biped.harold_biped_constants import HAROLD_JOINT_NAMES
from diogenes_mjlab.harold_biped.walk_env_cfg import (
  CONTACT_TRANSITION_MARGIN,
  FEET_CONTACT_SENSOR,
  SWING_FRACTION,
  harold_walk_env_cfg,
)
from diogenes_mjlab.mdp.observations import _phase

TASK = "Diogenes-Biped-Walk"
VX_GRID = (-0.1, 0.0, 0.1, 0.2, 0.3)
WZ_GRID = (-0.5, 0.0, 0.5)
SETTLE_S = 2.0

SCENARIOS = {
  "nominal": dict(dr_scale=1.0, push=None),
  "hard": dict(dr_scale=1.5, push=None),
  "push": dict(dr_scale=1.0, push=2.0),
}


def build_env(scenario: dict, num_envs: int, duration: float, step_freq: float, device: str):
  cfg = harold_walk_env_cfg(
    play=False, domain_rand=True, obs_noise=True,
    dr_scale=scenario["dr_scale"], step_frequency=step_freq,
  )
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = duration
  cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
  cfg.commands["twist"].rel_standing_envs = 0.0
  push = cfg.events.pop("push_robot", None)
  if scenario["push"] is not None:
    assert push is not None, "the walk env config no longer defines push_robot"
    push.interval_range_s = (1.0, 2.0)
    push.params["velocity_range"] = {
      k: (lo * scenario["push"], hi * scenario["push"])
      for k, (lo, hi) in push.params["velocity_range"].items()
    }
    cfg.events["push_robot"] = push
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


def command_grid(num_envs: int, device: str) -> torch.Tensor:
  pairs = [(vx, wz) for vx in VX_GRID for wz in WZ_GRID]
  idx = torch.arange(num_envs) % len(pairs)
  cmd = torch.zeros(num_envs, 3)
  cmd[:, 0] = torch.tensor([pairs[i][0] for i in idx])
  cmd[:, 2] = torch.tensor([pairs[i][1] for i in idx])
  return cmd.to(device)


@torch.no_grad()
def run_scenario(env: ManagerBasedRlEnv, policy, agent_cfg, step_freq: float) -> dict:
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  device = env.device
  n = env.num_envs
  robot = env.scene["robot"]
  jids, _ = robot.find_joints(HAROLD_JOINT_NAMES, preserve_order=True)
  aids, _ = robot.find_actuators(HAROLD_JOINT_NAMES, preserve_order=True)
  sensor = env.scene[FEET_CONTACT_SENSOR]

  obs, _ = wrapped.reset()
  twist = env.command_manager.get_term("twist")
  cmd = command_grid(n, device)
  twist.vel_command_b[:] = cmd
  twist.is_standing_env[:] = False
  obs = wrapped.get_observations()

  steps = env.max_episode_length
  settle = int(SETTLE_S / env.step_dt)
  alive = torch.ones(n, dtype=torch.bool, device=device)
  survived = torch.zeros(n, dtype=torch.bool, device=device)
  fail_step = torch.full((n,), steps, device=device)
  z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
  vel_sum, cnt = z(n, 3), z(n)
  match_sum, pow_sum, cnt_all = z(n), z(n), z(n)
  tau_sq, tau_pk, qd_pk = z(n, 6), z(n, 6), z(n, 6)
  lim = robot.data.joint_pos_limits[:, jids]
  rng = lim[..., 1] - lim[..., 0]
  margin_min = torch.ones(n, device=device)
  joint_margin_min = torch.ones(n, 6, device=device)

  for k in range(steps):
    actions = policy(obs)
    obs, _, dones, extras = wrapped.step(actions)
    done = dones.bool()
    timeout = extras["time_outs"].bool()
    # State after the step; for envs that just reset it is the new episode's,
    # so only envs still alive and not done this step are accumulated.
    live = alive & ~done
    q = robot.data.joint_pos[:, jids]
    qd = robot.data.joint_vel[:, jids]
    tau = robot.data.actuator_force[:, aids]
    if k >= settle:
      lv = robot.data.root_link_lin_vel_b
      av = robot.data.root_link_ang_vel_b
      vel_sum[live] += torch.stack([lv[:, 0], lv[:, 1], av[:, 2]], -1)[live]
      cnt[live] += 1
    in_contact = sensor.data.found > 0
    leg_phase = gait.leg_phases(_phase(env, 1.0 / step_freq))
    match = harold_mdp.contact_schedule_match(
      in_contact, leg_phase, SWING_FRACTION, CONTACT_TRANSITION_MARGIN
    ).mean(-1)
    match_sum[live] += match[live]
    pow_sum[live] += (tau * qd).clamp(min=0).sum(-1)[live]
    cnt_all[live] += 1
    tau_sq[live] += tau[live] ** 2
    tau_pk[live] = torch.maximum(tau_pk[live], tau.abs()[live])
    qd_pk[live] = torch.maximum(qd_pk[live], qd.abs()[live])
    m = torch.minimum(q - lim[..., 0], lim[..., 1] - q) / rng
    margin_min[live] = torch.minimum(margin_min[live], m.min(-1).values[live])
    joint_margin_min[live] = torch.minimum(joint_margin_min[live], m[live])

    survived |= alive & done & timeout
    failed = alive & done & ~timeout
    fail_step[failed] = k
    alive &= ~done
    if not alive.any():
      break

  mean_vel = vel_sum / cnt.clamp(min=1).unsqueeze(-1)
  s = survived
  err_vx = (mean_vel[:, 0] - cmd[:, 0]).abs()
  err_wz = (mean_vel[:, 2] - cmd[:, 2]).abs()
  cpu = lambda t: t.float().cpu().numpy()  # noqa: E731
  res = {
    "survival": float(s.float().mean()),
    "mean_fail_time_s": float(fail_step[~s].float().mean() * env.step_dt) if (~s).any() else None,
    "vx_err": float(err_vx[s].mean()) if s.any() else None,
    "wz_err": float(err_wz[s].mean()) if s.any() else None,
    "vy_drift": float(mean_vel[s, 1].abs().mean()) if s.any() else None,
    "schedule_match": float((match_sum / cnt_all.clamp(min=1))[s].mean()) if s.any() else None,
    "power_w": float((pow_sum / cnt_all.clamp(min=1))[s].mean()) if s.any() else None,
    "tau_rms": dict(zip(HAROLD_JOINT_NAMES, cpu(torch.sqrt(tau_sq[s].sum(0) / cnt_all[s].sum())).round(2).tolist())) if s.any() else None,
    "tau_peak_p99": dict(zip(HAROLD_JOINT_NAMES, np.percentile(cpu(tau_pk[s]), 99, axis=0).round(1).tolist())) if s.any() else None,
    "qd_peak_p99": dict(zip(HAROLD_JOINT_NAMES, np.percentile(cpu(qd_pk[s]), 99, axis=0).round(1).tolist())) if s.any() else None,
    "limit_margin_p1": float(np.percentile(cpu(margin_min[s]), 1)) if s.any() else None,
    # Per joint: 1st percentile of the closest approach to a stop (fraction of
    # the range; negative = pushed past it), and the share of robots that did.
    "joint_margin_p1": dict(zip(HAROLD_JOINT_NAMES, np.percentile(cpu(joint_margin_min[s]), 1, axis=0).round(4).tolist())) if s.any() else None,
    "joint_past_stop_frac": dict(zip(HAROLD_JOINT_NAMES, cpu((joint_margin_min[s] < 0).float().mean(0)).round(4).tolist())) if s.any() else None,
    "per_command": {},
  }
  for vx in VX_GRID:
    for wz in WZ_GRID:
      sel = (cmd[:, 0] == vx) & (cmd[:, 2] == wz)
      ss = sel & s
      res["per_command"][f"vx{vx:+.1f}_wz{wz:+.1f}"] = {
        "survival": float(s[sel].float().mean()),
        "vx": float(mean_vel[ss, 0].mean()) if ss.any() else None,
        "wz": float(mean_vel[ss, 2].mean()) if ss.any() else None,
      }
  return res


def score(results: dict) -> float:
  """One number for ranking: survival first, then tracking. Higher is better."""
  surv = (0.35 * results["nominal"]["survival"] + 0.35 * results["hard"]["survival"]
          + 0.3 * results["push"]["survival"])
  n = results["nominal"]
  if n["vx_err"] is None:
    return surv
  track = math.exp(-(n["vx_err"] / 0.05) ** 2) * math.exp(-(n["wz_err"] / 0.15) ** 2)
  return 0.7 * surv + 0.3 * track


def fmt(v, spec=".3f"):
  return "   --" if v is None else format(v, spec)


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("checkpoint", type=Path)
  p.add_argument("--num-envs", type=int, default=1500, help="Per scenario (default: %(default)s).")
  p.add_argument("--duration", type=float, default=20.0)
  p.add_argument("--step-freq", type=float, default=None,
                 help="The training run's DIOGENES_STEP_FREQ (default: the task's).")
  p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  from diogenes_mjlab.harold_biped.walk_env_cfg import STEP_FREQUENCY_HZ
  step_freq = args.step_freq or STEP_FREQUENCY_HZ
  agent_cfg = load_rl_cfg(TASK)
  results = {}
  for name in args.scenarios:
    torch.manual_seed(args.seed)
    env = build_env(SCENARIOS[name], args.num_envs, args.duration, step_freq, args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    results[name] = run_scenario(env, policy, agent_cfg, step_freq)
    env.close()

  print(f"\n{args.checkpoint}")
  print(f"{'scenario':<9} {'surv':>6} {'fail_t':>6} {'vx_err':>7} {'wz_err':>7} {'vy':>6} "
        f"{'sched':>6} {'power':>6} {'margin':>7}")
  for name, r in results.items():
    print(f"{name:<9} {r['survival']:6.3f} {fmt(r['mean_fail_time_s'], '6.1f')} "
          f"{fmt(r['vx_err'], '7.3f')} {fmt(r['wz_err'], '7.3f')} {fmt(r['vy_drift'], '6.3f')} "
          f"{fmt(r['schedule_match'], '6.3f')} {fmt(r['power_w'], '6.1f')} "
          f"{fmt(r['limit_margin_p1'], '7.3f')}")
  if "nominal" in results:
    n = results["nominal"]
    print("\nnominal, per command (survival | achieved vx, wz):")
    for key, c in n["per_command"].items():
      print(f"  {key}: {c['survival']:5.2f} | {fmt(c['vx'], '+.3f')} {fmt(c['wz'], '+.3f')}")
    print(f"torque RMS  {n['tau_rms']}")
    print(f"torque p99 of peak  {n['tau_peak_p99']}")
    print(f"joint speed p99 of peak  {n['qd_peak_p99']}")
    print(f"closest approach to a stop, p1 (fraction of range)  {n['joint_margin_p1']}")
    print(f"share of robots pushed past a stop  {n['joint_past_stop_frac']}")
  if set(SCENARIOS) <= set(results):
    results["score"] = score(results)
    print(f"\nSCORE {results['score']:.3f}")
  out = args.checkpoint.with_suffix(".eval.json")
  out.write_text(json.dumps(results, indent=1))
  print(f"[OK] wrote {out}")


if __name__ == "__main__":
  main()
