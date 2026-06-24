"""Pytest configuration for diogenes_mjlab tests.

Ensures the src layout package is importable without an editable install,
and provides shared fixtures.
"""

import sys
from pathlib import Path

# Make `src/` importable when running pytest from the repo root without
# having done `uv pip install -e .` first.  uv run pytest already installs
# the package, but this guard keeps plain `pytest` invocations working too.
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
  sys.path.insert(0, str(_SRC))
