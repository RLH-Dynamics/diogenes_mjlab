# export_onnx.py  -- run in the TRAINING env
"""Export a trained .pt checkpoint to deployable ONNX (actor only, obs-normalizer
baked in) with the metadata diogenes_control checks at start-up.

    uv run python src/diogenes_mjlab/tools/export_onnx.py \\
        logs/rsl_rl/harold_biped_suspended/<run>/model_<N>.pt \\
        --task Diogenes-Biped-Suspended \\
        --deploy ../diogenes_control/policy.onnx

On top of mjlab's base metadata (joint_names, default_joint_pos, action_scale,
joint_stiffness/damping, observation_names) this attaches:

    actor_history_length  past steps stacked per actor term (0 = none)
    phase_period          gait-clock period, seconds
    step_dt               control period, seconds
    task_id               the task the layout came from
    command_ranges        velocity-command ranges the policy was trained on,
                          lo,hi for vx, vy, wz (tasks with a "twist" command)

and, for tasks whose actuators are PD models computed in torch (the walking
task's RS03 model), replaces mjlab's joint_stiffness/joint_damping -- read from
MuJoCo actuator parameters, which such actuators don't use -- with their gains.

The control code refuses to run a policy whose metadata disagrees with its
config, so the export MUST use the task the checkpoint was trained on. The gait
clock is taken from the run's own params/env.yaml: a run trained with
DIOGENES_STEP_FREQ is exported with the same clock without setting it again.
"""

import argparse
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import onnx


def run_step_frequency(checkpoint: Path) -> float | None:
    """The gait-clock frequency the checkpoint's run was trained with, from its
    params/env.yaml (None if the run has no gait clock or no saved params)."""
    params = checkpoint.parent / "params" / "env.yaml"
    if not params.exists():
        return None
    periods = {float(p) for p in re.findall(r"hop_period:\s*([0-9.eE+-]+)", params.read_text())}
    if len(periods) > 1:
        raise ValueError(f"{params} has several gait-clock periods: {sorted(periods)}")
    return 1.0 / periods.pop() if periods else None


def actor_layout_metadata(env_cfg, env) -> dict:
    """History length and gait-clock period of the actor group."""
    from diogenes_mjlab.mdp.observations import phase_clock
    actor = env_cfg.observations["actor"]
    periods = {
        term.params.get("hop_period")
        for term in actor.terms.values()
        if term.func is phase_clock
    }
    if len(periods) > 1:
        raise ValueError(f"Actor has several phase-clock periods: {periods}")
    return {
        "actor_history_length": actor.history_length or 0,
        "phase_period": periods.pop() if periods else 0.0,
        "step_dt": env.step_dt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", default="Diogenes-Biped-Suspended",
                        help="Task the checkpoint was trained on.")
    parser.add_argument("--deploy", type=Path, default=None,
                        help="Also copy the exported ONNX here.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    freq = run_step_frequency(args.checkpoint)
    if freq is not None:
        # Read by the task configs when they register, i.e. on import below.
        os.environ["DIOGENES_STEP_FREQ"] = repr(freq)
        print(f"[INFO] Gait clock from the run: {freq:.4f} Hz ({1.0 / freq:.4f} s)")

    from mjlab.actuator.pd_actuator import IdealPdActuator
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import mjlab.tasks            # noqa: F401  (populate registry)
    import diogenes_mjlab         # noqa: F401  (register the Diogenes/Harold tasks)

    # Build the PLAY env exactly like play.py (num_envs forced to 1 inside cfg).
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Same runner the task registered (falls back to base), same load call as play.
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device="cpu")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location="cpu")

    # Export actor -> ONNX next to the checkpoint, then attach metadata.
    policy_dir, filename, onnx_path = runner._get_export_paths(str(args.checkpoint))
    runner.export_policy_to_onnx(str(policy_dir), filename)
    metadata = get_base_metadata(env.unwrapped, "local")
    metadata.update(actor_layout_metadata(env_cfg, env.unwrapped))
    metadata["task_id"] = args.task
    if freq is not None and abs(metadata["phase_period"] - 1.0 / freq) > 1e-6:
        raise RuntimeError(f"Exported clock {metadata['phase_period']} s does not match "
                           f"the run's {1.0 / freq} s.")
    if "twist" in env_cfg.commands:
        r = env_cfg.commands["twist"].ranges
        metadata["command_ranges"] = [*r.lin_vel_x, *r.lin_vel_y, *r.ang_vel_z]
    robot = env.unwrapped.scene["robot"]
    pd = [a for a in robot.actuators if isinstance(a, IdealPdActuator)]
    if pd:
        assert len(pd) == 1 and len(pd[0].target_names) == len(robot.joint_names), (
          "expected one PD actuator group driving every joint")
        order = [pd[0].target_names.index(j) for j in robot.joint_names]
        metadata["joint_stiffness"] = pd[0].default_stiffness[0, order].cpu().tolist()
        metadata["joint_damping"] = pd[0].default_damping[0, order].cpu().tolist()
    attach_metadata_to_onnx(str(onnx_path), metadata)

    m = onnx.load(str(onnx_path))
    width = m.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    print(f"\n[OK] wrote {onnx_path}")
    print(f"     input width = {width}")
    for p in m.metadata_props:
        print(f"     {p.key:<21}= {p.value}")

    if args.deploy is not None:
        shutil.copyfile(onnx_path, args.deploy)
        print(f"[OK] copied to {args.deploy}")
    env.close()


if __name__ == "__main__":
    main()
