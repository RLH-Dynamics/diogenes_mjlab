"""Environment-variable parsing helpers for DIOGENES_* flag resolution.

This module centralises all env-var parsing so env_cfgs.py has no direct
``os.environ`` calls. Precedence is always:
    explicit kwarg > env var > play-based default

Functions
---------
_env_bool(name)  : parse a 1/0 boolean env var; returns None if unset.
_env_float(name) : parse a float env var; returns None if unset.

Both raise ``ValueError`` on unrecognised non-empty values so typos fail
loudly rather than silently disabling logging or DR.
"""

import os

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSY: frozenset[str] = frozenset({"0", "false", "no", "off", "n", "f"})


def _env_bool(name: str) -> bool | None:
  """Parse a boolean env var. Returns None if unset/blank, else True/False.

  Unrecognized non-empty values raise, so a typo (``=ture``) fails loudly rather
  than silently disabling logging.
  """
  raw = os.environ.get(name)
  if raw is None:
    return None
  val = raw.strip().lower()
  if val == "":
    return None
  if val in _TRUTHY:
    return True
  if val in _FALSY:
    return False
  raise ValueError(
    f"Environment variable {name}={raw!r} is not a recognized boolean. "
    f"Use one of {sorted(_TRUTHY)} or {sorted(_FALSY)}."
  )


def _env_float(name: str) -> float | None:
  """Parse a float env var. Returns None if unset/blank, else the value.

  Unparseable non-empty values raise so a typo fails loudly.
  """
  raw = os.environ.get(name)
  if raw is None:
    return None
  val = raw.strip()
  if val == "":
    return None
  try:
    return float(val)
  except ValueError as exc:
    raise ValueError(
      f"Environment variable {name}={raw!r} is not a valid float."
    ) from exc
