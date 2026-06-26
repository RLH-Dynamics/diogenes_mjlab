"""Package-wide physical and task constants for the Diogenes hop stand.

This is a LEAF module: it imports nothing from within the package. All other
modules may safely import from here without creating circular dependencies.

Sign conventions
----------------
The leg hangs from an unactuated prismatic ``slider`` joint. The leg_mount body
carries a 180 deg rotation about X, so the joint's local +Z points along world
-Z. A more negative slider value raises the carriage:

    carriage_height_above_start = -slider_pos   (verified via MuJoCo FK)

All heights (TRAJ_*, OBS_NOISE_SLIDER_*) are in meters relative to the slider
origin.
"""

# ---------------------------------------------------------------------------
# Physical constants.
# ---------------------------------------------------------------------------

# Standard gravity (m/s^2). The dual-parabola flight arc is a TRUE free-fall
# parabola at this acceleration; its duration is fixed by physics.
GRAVITY: float = 9.81

# ---------------------------------------------------------------------------
# Carriage trajectory geometry (shared by both trajectories).
# All heights are z-values RELATIVE TO THE SLIDER ORIGIN.
# Require TRAJ_MAX >= TRAJ_TRANSITION > TRAJ_MIN.
#   TRAJ_MAX        : top of the motion (dual-parabola flight apex; sine peak).
#   TRAJ_MIN        : bottom of the motion (dual-parabola recovery dip; trough).
#   TRAJ_TRANSITION : (dual-parabola only) height where flight and recovery meet.
# ---------------------------------------------------------------------------
TRAJ_MAX: float = 0.45
TRAJ_MIN: float = 0.05
TRAJ_TRANSITION: float = 0.15

# ---------------------------------------------------------------------------
# Sinusoid period (seconds) -- the FREE design parameter for the sine task.
# Peak vertical accel = amp * (2*pi / SINE_PERIOD)^2.  Keep below g for a
# non-ballistic, gentle motion (the gentle, well-behaved first-transfer target).
# ---------------------------------------------------------------------------
SINE_PERIOD: float = 2.0

# ---------------------------------------------------------------------------
# Joint / actuator names.
# ---------------------------------------------------------------------------

# Names of the three actuated leg joints (== XML <position> actuator names).
# Also used as monitoring column headers (hip, thigh, calf order).
DIOGENES_ACTUATOR_NAMES: tuple[str, ...] = ("hip", "thigh", "calf")

# ---------------------------------------------------------------------------
# Inset fraction used by BOTH the joint_at_limit termination AND the
# random-start-pose reset event.  Defining it once keeps the two in lockstep so
# a fresh start can never trip the limit termination on step 0.
# ---------------------------------------------------------------------------
JOINT_LIMIT_MARGIN: float = 0.02

# ---------------------------------------------------------------------------
# Named robot geometries / sensors.
# ---------------------------------------------------------------------------

# Name of the foot collision geom in diogenes.xml (used for friction DR).
FOOT_GEOM_NAME: str = "foot"

# Name of the foot/ground contact sensor (referenced by several reward terms).
FOOT_CONTACT_SENSOR: str = "foot_ground_contact"

# ---------------------------------------------------------------------------
# Contact-phase termination.
# Terminate the episode when the foot/ground contact state is wrong for the
# current hop phase (e.g. foot still touching during the flight arc, or foot
# airborne during stance).  The same name is used as BOTH the termination dict
# key AND the ``term_name`` of the dedicated penalty reward, so the reward fires
# exactly on the step this termination triggers (terminations are computed just
# before rewards each step).
# ---------------------------------------------------------------------------
CONTACT_PHASE_TERM_NAME: str = "contact_phase_violation"

# Set False to disable the contact-phase termination (e.g. during early
# curriculum warm-up or ablation runs).  The DIOGENES_CONTACT_PHASE_TERM env
# var overrides this at runtime without a code change.
CONTACT_PHASE_TERM_ENABLED: bool = False

# Tolerance band (as a fraction of the hop cycle) around each liftoff/landing
# transition where a contact-state mismatch is NOT terminated.  This absorbs the
# finite time the foot needs to leave/meet the ground -- and the grounded reset
# pose at phase ~ 0, which would otherwise terminate the dual-parabola task on
# step 0.  Only clear, sustained violations away from a transition terminate.
CONTACT_PHASE_MARGIN: float = 0.15

# ---------------------------------------------------------------------------
# Observation noise + delay (sim-to-real).
# Applied to the ACTOR proprio terms only; critic stays clean.
# OBS_DELAY_MAX_LAG is the max lag in CONTROL steps (at 50 Hz: lag 3 == 60 ms).
# ---------------------------------------------------------------------------
OBS_DELAY_MIN_LAG: int = 0
OBS_DELAY_MAX_LAG: int = 3

# Rolling history window fed to the ACTOR observation (0 = disabled).
# At 50 Hz, H=5 gives 100 ms of context (~10% of the hop cycle).
OBS_HISTORY_LENGTH: int = 10

# Additive uniform sensor-noise half-widths for each proprio channel (+-h).
OBS_NOISE_JOINT_POS: float = 0.01   # rad
OBS_NOISE_JOINT_VEL: float = 0.5    # rad/s
OBS_NOISE_SLIDER_POS: float = 0.005  # m
OBS_NOISE_SLIDER_VEL: float = 0.05   # m/s
