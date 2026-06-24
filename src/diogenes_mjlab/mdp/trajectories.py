"""Carriage trajectory maths for the Diogenes hop stand.

Dual-parabola (gravity-exact, derived period) and sinusoidal (smooth, free
period) reference functions, plus the timing helper that solves the derived
period from geometry alone.
"""

from __future__ import annotations

import math

import torch

from ..constants import GRAVITY


def dual_parabola_timing(
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> tuple[float, float, float, float, float]:
  """Solve the gravity-exact dual-parabola timing from geometry alone.

  Everything is derived from physics; there is NO free period parameter.

  FLIGHT arc (true free fall at -gravity), amplitude Hf = traj_max - traj_transition:
    * Leaves traj_transition moving UP at v0, rises Hf, falls back to
      traj_transition moving DOWN at v0, with
          v0        = sqrt(2 * gravity * Hf)          (speed at the transition)
          T_flight  = 2 * sqrt(2 * Hf / gravity)      (full up+down flight time)

  RECOVERY arc (constant acceleration +a), amplitude Hr = traj_transition - traj_min:
    * Velocity continuity forces the recovery to ENTER at -v0 (matching the end
      of flight) and LEAVE at +v0 (matching the start of the next flight). A
      constant-accel arc that decelerates from v0 to 0 over a drop of Hr needs
          a         = v0^2 / (2 * Hr) = gravity * Hf / Hr   (the constant accel)
          T_recovery = 2 * v0 / a                            (down+back up time)

  Total period and the phase split:
          T_total     = T_flight + T_recovery
          flight_frac = T_flight / T_total

  Args:
    traj_min, traj_max, traj_transition: trajectory heights, z rel origin (m),
      with traj_max >= traj_transition >= traj_min.
    gravity: free-fall acceleration for the flight arc (m/s^2).

  Returns:
    (T_total, flight_frac, v0, recovery_accel, T_flight) -- seconds / m·s / etc.
  """
  Hf = traj_max - traj_transition  # flight amplitude (>= 0)
  Hr = traj_transition - traj_min  # recovery amplitude (> 0 required)
  assert Hf >= 0.0, "Require traj_max >= traj_transition."
  assert Hr > 0.0, "Require traj_transition > traj_min (a finite recovery dip)."

  v0 = math.sqrt(2.0 * gravity * Hf)
  t_flight = 2.0 * math.sqrt(2.0 * Hf / gravity)
  recovery_accel = gravity * Hf / Hr
  t_recovery = 2.0 * v0 / recovery_accel
  t_total = t_flight + t_recovery
  flight_frac = t_flight / t_total
  return t_total, flight_frac, v0, recovery_accel, t_flight


def dual_parabola_reference(
  phi: torch.Tensor,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> torch.Tensor:
  """Reference carriage height h_ref(phi). Shape == phi.shape. All z's rel origin.

  The cycle (phase phi in [0,1)) is two parabolic arcs meeting at
  ``traj_transition``, parametrized DIRECTLY by the physical motion (see
  ``dual_parabola_timing``):

    * FLIGHT arc, phi in [0, flight_frac]: true free fall at -gravity. Real time
      within the arc is t = phi * T_total, so
          h = traj_transition + v0 * t - 0.5 * gravity * t^2,
      which rises to apex traj_max and returns to traj_transition. Its duration
      is EXACTLY the Earth-gravity ballistic time for amplitude Hf.

    * RECOVERY arc, phi in [flight_frac, 1): constant deceleration/acceleration
      at +recovery_accel. With t' = phi * T_total - T_flight,
          h = traj_transition - v0 * t' + 0.5 * recovery_accel * t'^2,
      which dips to traj_min (at zero velocity) and returns to traj_transition,
      entering at -v0 and leaving at +v0 so velocity is continuous at both joins.

  Because the flight time is fixed by gravity and the recovery time by velocity
  continuity, the whole period T_total is derived, not chosen.

  Args:
    traj_min: lowest carriage height in the cycle (recovery dip), z rel origin.
    traj_max: apex carriage height (flight peak), z rel origin.
    traj_transition: height where flight and recovery meet (the cycle boundary
      level), z rel origin.
    gravity: free-fall acceleration for the flight arc (m/s^2).
  """
  t_total, flight_frac, v0, recovery_accel, t_flight = dual_parabola_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  t = phi * t_total  # real time within the cycle, seconds

  # FLIGHT: free fall from traj_transition, up at v0, accel -gravity.
  flight_h = traj_transition + v0 * t - 0.5 * gravity * torch.square(t)

  # RECOVERY: constant accel from traj_transition, down at v0, accel +recovery_accel.
  tr = t - t_flight
  recovery_h = (
    traj_transition - v0 * tr + 0.5 * recovery_accel * torch.square(tr)
  )

  return torch.where(phi < flight_frac, flight_h, recovery_h)
