"""Shared helpers for dev scripts: model compilation, joint/body lookups, inertia reconstruction."""

from pathlib import Path
from typing import Optional

import numpy as np
import mujoco


def load_model_from_xml(xml_path: str | Path) -> mujoco.MjModel:
    """Load and compile a MuJoCo model from XML file.

    Args:
        xml_path: Path to the scene.xml file.

    Returns:
        Compiled MjModel instance.

    Raises:
        SystemExit: If the XML file does not exist.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise SystemExit(
            f"XML not found: {xml_path.resolve()}\n"
            f"Run from the repo root so meshes under 'assets' resolve."
        )
    return mujoco.MjModel.from_xml_path(str(xml_path))


def body_frame_inertia(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    """Reconstruct the full 3x3 inertia tensor of a body in its body frame.

    MuJoCo stores diagonalized inertia (principal moments in model.body_inertia
    and principal-frame orientation as a quaternion in model.body_iquat). This
    function reconstructs the body-frame tensor:
        I_body = R * diag(principal_moments) * R^T
    where R is the rotation matrix derived from body_iquat.

    Args:
        model: Compiled MjModel.
        body_id: ID of the body (index into model.body_*).

    Returns:
        3x3 symmetric inertia tensor in the body frame (kg*m^2).
    """
    Idiag = model.body_inertia[body_id].copy()  # principal moments (Ixx, Iyy, Izz)
    q = model.body_iquat[body_id].copy()        # principal-frame quat (w,x,y,z)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    return R @ np.diag(Idiag) @ R.T


def get_body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Get the name of a body by ID, with fallback to generic name."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def get_joint_name(model: mujoco.MjModel, joint_id: int) -> Optional[str]:
    """Get the name of a joint by ID."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)


def get_joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    """Get the ID of a joint by name.

    Returns:
        Joint ID, or negative if not found.
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)


def validate_inertia(principal_moments: np.ndarray) -> tuple[bool, str]:
    """Validate that principal inertia moments satisfy physical constraints.

    Args:
        principal_moments: 3-element array of principal inertia moments.

    Returns:
        Tuple of (is_valid, reason_string).
    """
    if np.all(principal_moments > 0):
        # Triangle inequality: sum of any two >= the third
        I = principal_moments
        tri = (I[0] <= I[1] + I[2] + 1e-12 and
               I[1] <= I[0] + I[2] + 1e-12 and
               I[2] <= I[0] + I[1] + 1e-12)
        return (tri, "OK" if tri else "*** INVALID ***")
    return (False, "*** INVALID ***")
