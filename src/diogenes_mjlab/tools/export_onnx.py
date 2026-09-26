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

The control code refuses to run a policy whose metadata disagrees with its
config, so the export MUST use the task the checkpoint was trained on.
"""

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import onnx

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab.tasks            # noqa: F401  (populate registry)
import diogenes_mjlab         # noqa: F401  (register the Diogenes/Harold tasks)
from diogenes_mjlab.mdp.observations import phase_clock


def actor_layout_metadata(env_cfg, env) -> dict:
    """History length and gait-clock period of the actor group."""
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
