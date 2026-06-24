"""Snapshot test: freeze the Diogenes environment config structure.

Phase 0 safety net — builds all four configs (trajectory x play) and
serializes them into a deterministic JSON dict.  On first run (or when
DIOGENES_BLESS_SNAPSHOT=1) writes the golden file.  On subsequent runs,
loads the golden and asserts deep equality, printing a readable diff on
mismatch.

No GPU, no sim step, no MuJoCo physics — pure Python config objects only.
"""

from __future__ import annotations

import json
import math
import os
import pprint
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
GOLDEN_PATH = SNAPSHOT_DIR / "diogenes_cfg.json"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _safe_name(obj: Any) -> str:
  """Return a stable name for a callable or type."""
  if callable(obj) and hasattr(obj, "__name__"):
    return obj.__name__
  if callable(obj) and hasattr(obj, "__class__"):
    return type(obj).__name__
  return repr(obj)


def _serialize_scene_entity(obj: Any) -> dict:
  """Serialize a SceneEntityCfg to a stable dict."""
  d: dict[str, Any] = {"__type__": "SceneEntityCfg"}
  for attr in ("name", "joint_names", "body_names", "geom_names", "actuator_names"):
    val = getattr(obj, attr, None)
    if val is not None:
      if isinstance(val, (list, tuple)):
        d[attr] = list(val)
      else:
        d[attr] = val
  return d


def _is_scene_entity(obj: Any) -> bool:
  return type(obj).__name__ == "SceneEntityCfg"


def _serialize_noise(obj: Any) -> dict | None:
  """Serialize a noise config (e.g. UniformNoiseCfg)."""
  if obj is None:
    return None
  d: dict[str, Any] = {"__type__": type(obj).__name__}
  for attr in ("n_min", "n_max", "mean", "std", "operation"):
    val = getattr(obj, attr, None)
    if val is not None:
      d[attr] = round(float(val), 8) if isinstance(val, float) else val
  return d


def _serialize_value(val: Any) -> Any:
  """Recursively serialize a parameter value to a JSON-safe form."""
  if val is None:
    return None
  if isinstance(val, bool):
    return val
  if isinstance(val, (int, float)):
    if isinstance(val, float) and not math.isfinite(val):
      return str(val)
    return round(float(val), 8) if isinstance(val, float) else val
  if isinstance(val, str):
    return val
  if isinstance(val, (list, tuple)):
    return [_serialize_value(v) for v in val]
  if isinstance(val, dict):
    return {str(k): _serialize_value(v) for k, v in val.items()}
  if _is_scene_entity(val):
    return _serialize_scene_entity(val)
  if callable(val) and hasattr(val, "__name__"):
    return {"__callable__": val.__name__}
  # Dataclass-like cfg objects: try to iterate their fields
  if hasattr(val, "__dataclass_fields__"):
    return {
      "__type__": type(val).__name__,
      **{k: _serialize_value(getattr(val, k)) for k in val.__dataclass_fields__},
    }
  # Fallback: stable placeholder
  return f"<unserializable:{type(val).__name__}>"


def _serialize_params(params: dict | None) -> dict:
  if not params:
    return {}
  result = {}
  for k, v in params.items():
    result[k] = _serialize_value(v)
  return result


# ---------------------------------------------------------------------------
# Per-manager serializers
# ---------------------------------------------------------------------------

def _serialize_reward_terms(rewards: dict) -> dict:
  out = {}
  for name, term in rewards.items():
    func = getattr(term, "func", None)
    out[name] = {
      "func": _safe_name(func) if func is not None else None,
      "weight": round(float(term.weight), 8) if hasattr(term, "weight") else None,
      "params": _serialize_params(getattr(term, "params", None)),
    }
  return out


def _serialize_obs_group(group: Any) -> dict:
  """Serialize one ObservationGroupCfg."""
  terms = getattr(group, "terms", {}) or {}
  out_terms = {}
  for name, term in terms.items():
    func = getattr(term, "func", None)
    noise_cfg = getattr(term, "noise", None)
    out_terms[name] = {
      "func": _safe_name(func) if func is not None else None,
      "params": _serialize_params(getattr(term, "params", None)),
      "noise": _serialize_noise(noise_cfg),
      "delay_min_lag": getattr(term, "delay_min_lag", 0),
      "delay_max_lag": getattr(term, "delay_max_lag", 0),
    }
  return {
    "term_names_in_order": list(terms.keys()),
    "concatenate_terms": getattr(group, "concatenate_terms", None),
    "enable_corruption": getattr(group, "enable_corruption", None),
    "terms": out_terms,
  }


def _serialize_observations(obs: dict) -> dict:
  return {group_name: _serialize_obs_group(group) for group_name, group in obs.items()}


def _serialize_event_terms(events: dict) -> dict:
  out = {}
  for name, term in events.items():
    func = getattr(term, "func", None)
    out[name] = {
      "func": _safe_name(func) if func is not None else None,
      "mode": getattr(term, "mode", None),
      "params": _serialize_params(getattr(term, "params", None)),
    }
  return out


def _serialize_termination_terms(terminations: dict) -> dict:
  out = {}
  for name, term in terminations.items():
    func = getattr(term, "func", None)
    out[name] = {
      "func": _safe_name(func) if func is not None else None,
      "time_out": getattr(term, "time_out", False),
      "params": _serialize_params(getattr(term, "params", None)),
    }
  return out


def _serialize_metrics_terms(metrics: dict) -> dict:
  out = {}
  for name, term in metrics.items():
    func = getattr(term, "func", None)
    out[name] = {
      "func": _safe_name(func) if func is not None else None,
      "params": _serialize_params(getattr(term, "params", None)),
    }
  return out


def _serialize_recorder_terms(recorders: dict) -> dict:
  out = {}
  for name, term in recorders.items():
    func = getattr(term, "func", None)
    out[name] = {
      "func": _safe_name(func) if func is not None else None,
      "params": _serialize_params(getattr(term, "params", None)),
    }
  return out


def _serialize_scene(scene: Any) -> dict:
  out: dict[str, Any] = {
    "num_envs": getattr(scene, "num_envs", None),
    "env_spacing": getattr(scene, "env_spacing", None),
  }
  # Entity names
  entities = getattr(scene, "entities", None)
  if entities:
    out["entity_names"] = list(entities.keys())
  # Sensors
  sensors = getattr(scene, "sensors", None)
  if sensors:
    if isinstance(sensors, (list, tuple)):
      out["sensor_names"] = [
        getattr(s, "name", type(s).__name__) for s in sensors
      ]
    elif isinstance(sensors, dict):
      out["sensor_names"] = list(sensors.keys())
  return out


def _serialize_sim(sim: Any) -> dict:
  out: dict[str, Any] = {}
  for attr in ("njmax", "nconmax", "decimation"):
    val = getattr(sim, attr, None)
    if val is not None:
      out[attr] = val
  mujoco = getattr(sim, "mujoco", None)
  if mujoco is not None:
    out["mujoco"] = {
      "timestep": getattr(mujoco, "timestep", None),
    }
  return out


def _serialize_actions(actions: dict) -> dict:
  out = {}
  for name, action in actions.items():
    out[name] = {
      "__type__": type(action).__name__,
      "entity_name": getattr(action, "entity_name", None),
      "actuator_names": list(getattr(action, "actuator_names", []) or []),
      "scale": getattr(action, "scale", None),
      "use_default_offset": getattr(action, "use_default_offset", None),
    }
  return out


# ---------------------------------------------------------------------------
# Top-level config serializer
# ---------------------------------------------------------------------------

def serialize_cfg(cfg: Any, trajectory: str, play: bool) -> dict:
  """Serialize a ManagerBasedRlEnvCfg into a JSON-able dict."""
  obs_raw = getattr(cfg, "observations", {}) or {}
  events_raw = getattr(cfg, "events", {}) or {}
  rewards_raw = getattr(cfg, "rewards", {}) or {}
  terminations_raw = getattr(cfg, "terminations", {}) or {}
  metrics_raw = getattr(cfg, "metrics", {}) or {}
  recorders_raw = getattr(cfg, "recorders", {}) or {}
  actions_raw = getattr(cfg, "actions", {}) or {}
  scene = getattr(cfg, "scene", None)
  sim = getattr(cfg, "sim", None)

  return {
    # ---- Identity ----
    "trajectory": trajectory,
    "play": play,
    # ---- Top-level scalars ----
    "decimation": getattr(cfg, "decimation", None),
    "episode_length_s": getattr(cfg, "episode_length_s", None),
    # ---- Sub-configs ----
    "sim": _serialize_sim(sim) if sim is not None else {},
    "scene": _serialize_scene(scene) if scene is not None else {},
    "actions": _serialize_actions(actions_raw),
    "rewards": _serialize_reward_terms(rewards_raw),
    "observations": _serialize_observations(obs_raw),
    "events": _serialize_event_terms(events_raw),
    "terminations": _serialize_termination_terms(terminations_raw),
    "metrics": _serialize_metrics_terms(metrics_raw),
    "recorders": _serialize_recorder_terms(recorders_raw),
  }


# ---------------------------------------------------------------------------
# Build all four configs
# ---------------------------------------------------------------------------

ALL_COMBOS = [
  ("dual_parabola", False),
  ("dual_parabola", True),
  ("sine", False),
  ("sine", True),
]


def _combo_key(trajectory: str, play: bool) -> str:
  return f"{trajectory}__play_{play}"


def build_all_snapshots() -> dict:
  """Build and serialize all four configs. No GPU required."""
  from diogenes_mjlab.env_cfgs import diogenes_env_cfg

  result = {}
  for trajectory, play in ALL_COMBOS:
    cfg = diogenes_env_cfg(
      trajectory=trajectory,
      play=play,
      # Force deterministic flags so snapshots don't depend on env vars.
      monitor=True,
      record_csv=play,
      domain_rand=not play,
      obs_noise=not play,
      dr_scale=1.0,
      reset_joints=not play,
    )
    key = _combo_key(trajectory, play)
    result[key] = serialize_cfg(cfg, trajectory, play)
  return result


# ---------------------------------------------------------------------------
# Diff helper
# ---------------------------------------------------------------------------

def _diff_dicts(old: Any, new: Any, path: str = "") -> list[str]:
  """Return human-readable diff lines between two JSON-deserialized values."""
  lines = []
  if type(old) != type(new):
    lines.append(f"  {path}: type {type(old).__name__} -> {type(new).__name__}")
    return lines
  if isinstance(old, dict):
    all_keys = sorted(set(old) | set(new))
    for k in all_keys:
      subpath = f"{path}.{k}" if path else k
      if k not in old:
        lines.append(f"  ADDED   {subpath}: {new[k]!r}")
      elif k not in new:
        lines.append(f"  REMOVED {subpath}: {old[k]!r}")
      else:
        lines.extend(_diff_dicts(old[k], new[k], subpath))
  elif isinstance(old, list):
    if old != new:
      lines.append(f"  CHANGED {path}:")
      lines.append(f"    was: {old!r}")
      lines.append(f"    now: {new!r}")
  else:
    if old != new:
      lines.append(f"  CHANGED {path}: {old!r} -> {new!r}")
  return lines


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_config_snapshot() -> None:
  """Build all four configs and compare against the golden snapshot.

  On first run (or DIOGENES_BLESS_SNAPSHOT=1), writes the golden.
  On subsequent runs, loads and asserts deep equality.
  """
  bless = os.environ.get("DIOGENES_BLESS_SNAPSHOT", "0").strip().lower() in (
    "1", "true", "yes", "on",
  )

  current = build_all_snapshots()

  SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

  if bless or not GOLDEN_PATH.exists():
    GOLDEN_PATH.write_text(json.dumps(current, indent=2, sort_keys=True))
    print(f"\n[snapshot] Golden written to {GOLDEN_PATH}")
    # Sanity-round-trip: re-read and confirm JSON is valid.
    _ = json.loads(GOLDEN_PATH.read_text())
    return

  golden = json.loads(GOLDEN_PATH.read_text())

  diff_lines = _diff_dicts(golden, current)
  if diff_lines:
    msg = "\n".join([
      "",
      "Config snapshot mismatch!",
      "Re-run with DIOGENES_BLESS_SNAPSHOT=1 to update the golden if this is intentional.",
      "",
      *diff_lines,
    ])
    pytest.fail(msg)
