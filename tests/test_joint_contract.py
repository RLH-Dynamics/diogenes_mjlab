"""Pin the Harold joint contract the hardware stack was verified against.

diogenes_control's per-joint `direction` signs were checked by hand on the real
robot against the joint meanings recorded here. If this test fails, the sim's
joint conventions (a joint axis, name or range) have changed:

  * A `direction_signature` change means a joint's POSITIVE direction moved.
    The hardware direction signs are no longer known to be right. Re-run the
    live joint-mapping test (diogenes_control/tools/stream_joints.py +
    tools/live_joint_viewer.py), fix config.py, and record the new signature
    there as DIRECTIONS_VERIFIED_SIGNATURE.
  * A `range` change alone leaves the directions valid; the control code derives
    its limits from the contract, so just re-export it.

Either way, re-bless and re-export deliberately:

  DIOGENES_BLESS_SNAPSHOT=1 uv run pytest -q tests/test_joint_contract.py
  uv run python src/diogenes_mjlab/harold_biped/joint_contract.py \\
      --out ../diogenes_control/sim_joint_contract.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from diogenes_mjlab.harold_biped import joint_contract

GOLDEN_PATH = Path(__file__).parent / "snapshots" / "harold_joint_contract.json"
CONTROL_COPY = (
  Path(__file__).resolve().parents[2] / "diogenes_control" / "sim_joint_contract.json"
)


@pytest.fixture(scope="module")
def contract() -> dict:
  return joint_contract.build_contract()


def test_contract_matches_golden(contract):
  if os.environ.get("DIOGENES_BLESS_SNAPSHOT") == "1":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(joint_contract.dumps(contract))
    pytest.skip(f"Blessed {GOLDEN_PATH.name}")

  golden = json.loads(GOLDEN_PATH.read_text())
  assert contract["direction_signature"] == golden["direction_signature"], (
    "A joint's POSITIVE direction changed in the sim. The hardware direction "
    "signs in diogenes_control/config.py were verified against the old meaning "
    "and must be re-checked on the robot. See this file's docstring."
  )
  assert contract == golden, (
    "Joint contract changed (ranges or metadata). See this file's docstring."
  )


def test_contract_covers_every_actuated_joint(contract):
  from diogenes_mjlab.harold_biped.harold_biped_constants import HAROLD_JOINT_NAMES

  assert list(contract["joints"]) == list(HAROLD_JOINT_NAMES)


@pytest.mark.skipif(not CONTROL_COPY.exists(), reason="diogenes_control not checked out alongside")
def test_control_copy_is_current(contract):
  deployed = json.loads(CONTROL_COPY.read_text())
  assert deployed == contract, (
    f"{CONTROL_COPY} is stale. Re-export it (see this file's docstring)."
  )
