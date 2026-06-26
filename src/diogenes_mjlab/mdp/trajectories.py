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


def dual_parabola_velocity(
  phi: torch.Tensor,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> torch.Tensor:
  """Reference carriage velocity v_ref(phi), +up. Shape == phi.shape.

  Exact time-derivative of :func:`dual_parabola_reference`:

    * FLIGHT arc (phi in [0, flight_frac]): free fall, v = v0 - gravity * t.
    * RECOVERY arc (phi in [flight_frac, 1)): constant accel, with
      t' = t - T_flight, v = -v0 + recovery_accel * t'.

  Velocity is continuous at both arc joins by construction (it equals -v0 there).
  """
  t_total, flight_frac, v0, recovery_accel, t_flight = dual_parabola_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  t = phi * t_total  # real time within the cycle, seconds

  # FLIGHT: v = v0 - gravity * t.
  flight_v = v0 - gravity * t

  # RECOVERY: v = -v0 + recovery_accel * t'.
  tr = t - t_flight
  recovery_v = -v0 + recovery_accel * tr

  return torch.where(phi < flight_frac, flight_v, recovery_v)


def dual_parabola_acceleration(
  phi: torch.Tensor,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> torch.Tensor:
  """Reference carriage acceleration a_ref(phi), +up. Shape == phi.shape.

  Exact second time-derivative of :func:`dual_parabola_reference`; piecewise
  constant per arc (with a step at the flight/recovery boundary):

    * FLIGHT arc (phi in [0, flight_frac]): a = -gravity.
    * RECOVERY arc (phi in [flight_frac, 1)): a = +recovery_accel.
  """
  _, flight_frac, _, recovery_accel, _ = dual_parabola_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  flight_a = torch.full_like(phi, -gravity)
  recovery_a = torch.full_like(phi, recovery_accel)

  return torch.where(phi < flight_frac, flight_a, recovery_a)


def spring_timing(
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> tuple[float, float, float, float, float]:
  """Solve the spring-hop timing from geometry alone (no free stiffness knob).

  Identical FLIGHT arc to :func:`dual_parabola_timing`; the recovery (contact)
  arc is replaced by a Hooke's-law spring (simple harmonic motion).

  FLIGHT arc (true free fall at -gravity), amplitude Hf = traj_max - traj_transition:
          v0        = sqrt(2 * gravity * Hf)          (speed at the transition)
          T_flight  = 2 * sqrt(2 * Hf / gravity)      (full up+down flight time)

  CONTACT arc -- SHM with equilibrium AT traj_transition, amplitude
  Hr = traj_transition - traj_min:
    * The carriage ENTERS the transition moving DOWN at -v0 (matching the end of
      flight), swings to the bottom traj_min where it is momentarily at rest, and
      LEAVES at +v0 (matching the start of the next flight). For the bottom of the
      swing to land EXACTLY on traj_min while entering at speed v0, the spring rate
      is fixed by continuity + geometry:
          omega     = v0 / Hr                          (rad/s)
          T_contact = pi / omega = pi * Hr / v0        (half a SHM period)
    The carriage acceleration then follows Hooke's law: a = omega^2 * (transition -
    h), zero at the transition and peaking at v0^2 / Hr (= 2 * gravity * Hf / Hr,
    twice the dual-parabola constant push) at the bottom.

  Total period and the phase split:
          T_total     = T_flight + T_contact
          flight_frac = T_flight / T_total

  Args:
    traj_min, traj_max, traj_transition: trajectory heights, z rel origin (m),
      with traj_max >= traj_transition > traj_min.
    gravity: free-fall acceleration for the flight arc (m/s^2).

  Returns:
    (T_total, flight_frac, v0, omega, T_flight) -- seconds / m·s / rad·s / etc.
  """
  Hf = traj_max - traj_transition  # flight amplitude (>= 0)
  Hr = traj_transition - traj_min  # contact (spring) amplitude (> 0 required)
  assert Hf >= 0.0, "Require traj_max >= traj_transition."
  assert Hr > 0.0, "Require traj_transition > traj_min (a finite spring dip)."

  v0 = math.sqrt(2.0 * gravity * Hf)
  t_flight = 2.0 * math.sqrt(2.0 * Hf / gravity)
  omega = v0 / Hr
  t_contact = math.pi / omega
  t_total = t_flight + t_contact
  flight_frac = t_flight / t_total
  return t_total, flight_frac, v0, omega, t_flight


def spring_reference(
  phi: torch.Tensor,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> torch.Tensor:
  """Reference carriage height h_ref(phi). Shape == phi.shape. All z's rel origin.

  The cycle (phase phi in [0,1)) is a free-fall parabola joined to a spring arc at
  ``traj_transition`` (see :func:`spring_timing`):

    * FLIGHT arc, phi in [0, flight_frac]: true free fall at -gravity, IDENTICAL to
      :func:`dual_parabola_reference`. With t = phi * T_total,
          h = traj_transition + v0 * t - 0.5 * gravity * t^2,
      which rises to apex traj_max and returns to traj_transition.

    * CONTACT arc, phi in [flight_frac, 1): Hooke's-law spring (SHM) with
      equilibrium at traj_transition. With t' = phi * T_total - T_flight,
          h = traj_transition - Hr * sin(omega * t'),
      which dips to traj_min (at zero velocity, omega*t' = pi/2) and returns to
      traj_transition, entering at -v0 and leaving at +v0 so velocity is continuous
      at both joins.

  Because the flight time is fixed by gravity and the spring rate by velocity
  continuity, the whole period T_total is derived, not chosen.

  Args:
    traj_min: lowest carriage height in the cycle (spring dip), z rel origin.
    traj_max: apex carriage height (flight peak), z rel origin.
    traj_transition: height where flight and contact meet (the cycle boundary
      level AND the spring equilibrium), z rel origin.
    gravity: free-fall acceleration for the flight arc (m/s^2).
  """
  t_total, flight_frac, v0, omega, t_flight = spring_timing(
    traj_min, traj_max, traj_transition, gravity
  )
  hr = traj_transition - traj_min  # spring amplitude

  t = phi * t_total  # real time within the cycle, seconds

  # FLIGHT: free fall from traj_transition, up at v0, accel -gravity.
  flight_h = traj_transition + v0 * t - 0.5 * gravity * torch.square(t)

  # CONTACT: SHM about traj_transition, entering down at v0.
  tr = t - t_flight
  contact_h = traj_transition - hr * torch.sin(omega * tr)

  return torch.where(phi < flight_frac, flight_h, contact_h)


def spring_velocity(
  phi: torch.Tensor,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> torch.Tensor:
  """Reference carriage velocity v_ref(phi), +up. Shape == phi.shape.

  Exact time-derivative of :func:`spring_reference`:

    * FLIGHT arc (phi in [0, flight_frac]): free fall, v = v0 - gravity * t.
    * CONTACT arc (phi in [flight_frac, 1)): SHM, with t' = t - T_flight,
      v = -Hr * omega * cos(omega * t') = -v0 * cos(omega * t').

  Velocity is continuous at both arc joins by construction (it equals -v0 there).
  """
  t_total, flight_frac, v0, omega, t_flight = spring_timing(
    traj_min, traj_max, traj_transition, gravity
  )

  t = phi * t_total  # real time within the cycle, seconds

  # FLIGHT: v = v0 - gravity * t.
  flight_v = v0 - gravity * t

  # CONTACT: v = -v0 * cos(omega * t').
  tr = t - t_flight
  contact_v = -v0 * torch.cos(omega * tr)

  return torch.where(phi < flight_frac, flight_v, contact_v)


def spring_acceleration(
  phi: torch.Tensor,
  traj_min: float,
  traj_max: float,
  traj_transition: float,
  gravity: float = GRAVITY,
) -> torch.Tensor:
  """Reference carriage acceleration a_ref(phi), +up. Shape == phi.shape.

  Exact second time-derivative of :func:`spring_reference` (with a step at the
  flight/contact boundary, as in the dual-parabola case):

    * FLIGHT arc (phi in [0, flight_frac]): a = -gravity.
    * CONTACT arc (phi in [flight_frac, 1)): Hooke's law, with t' = t - T_flight,
      a = Hr * omega^2 * sin(omega * t') = (v0^2 / Hr) * sin(omega * t'),
      i.e. a = omega^2 * (traj_transition - h): zero at the transition, peaking at
      v0^2 / Hr at the bottom (omega*t' = pi/2).
  """
  t_total, flight_frac, v0, omega, t_flight = spring_timing(
    traj_min, traj_max, traj_transition, gravity
  )
  hr = traj_transition - traj_min  # spring amplitude

  t = phi * t_total  # real time within the cycle, seconds

  flight_a = torch.full_like(phi, -gravity)

  tr = t - t_flight
  contact_a = hr * (omega**2) * torch.sin(omega * tr)

  return torch.where(phi < flight_frac, flight_a, contact_a)
