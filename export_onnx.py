# export_onnx.py  — place at your diogenes_mjlab repo root, run in the TRAINING env
"""Export a trained Diogenes .pt checkpoint to deployable ONNX (actor only,
obs-normalizer baked in), matching control/policy.py's expected 11-dim layout."""

from dataclasses import asdict
from pathlib import Path
import sys

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata, attach_metadata_to_onnx
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab.tasks            # noqa: F401  (populate registry)
import diogenes_mjlab          # noqa: F401  <-- ADJUST to your package import name

TASK_ID = "Diogenes-Flat-Sine"
CKPT    = sys.argv[1] if len(sys.argv) > 1 else "logs/rsl_rl/diogenes/<run>/model_<N>.pt"

def main():
    ckpt = Path(CKPT)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # Build the PLAY env exactly like play.py (num_envs forced to 1 inside cfg).
    env_cfg  = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Same runner the task registered (falls back to base), same load call as play.
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device="cpu")
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cpu")

    # Export actor -> ONNX next to the checkpoint, then attach metadata.
    policy_dir, filename, onnx_path = runner._get_export_paths(str(ckpt))
    runner.export_policy_to_onnx(str(policy_dir), filename)
    attach_metadata_to_onnx(str(onnx_path), get_base_metadata(env.unwrapped, "local"))

    # Verify it matches the deploy layout before you trust it.
    import onnx
    m = onnx.load(str(onnx_path))
    meta = {p.key: p.value for p in m.metadata_props}
    width = m.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    print(f"\n[OK] wrote {onnx_path}")
    print(f"     input width      = {width}   (expect 11)")
    print(f"     observation_names= {meta.get('observation_names')}")
    print(f"     default_joint_pos= {meta.get('default_joint_pos')}  (expect 0,0,0)")
    print(f"     action_scale     = {meta.get('action_scale')}       (expect 1.0)")
    env.close()

if __name__ == "__main__":
    main()