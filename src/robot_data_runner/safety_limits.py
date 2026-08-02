"""Absolute joint-limit clamping + unit legibility for the SO-101 follower.

This module closes two related hardware-safety gaps found in a domain safety
review (2026-08):

* **S1 — unit mislabel.** ``max_relative_target`` is NOT degrees unless
  ``use_degrees=True``. In lerobot's default normalized mode
  (``MotorNormMode.RANGE_M100_100``), a body joint's ``.pos`` value spans
  -100..100 over its *entire* calibrated range, so ``max_relative_target=5.0``
  is 2.5% of range per control step — which can be a very large real angular
  rate depending on the joint's calibrated span (measured: wrist_roll
  9.00 deg/step = 270 deg/s at 30 Hz on a near-full-turn calibration).
  :func:`describe_clamp` makes this legible regardless of unit mode.
* **S3 — no absolute joint-limit clamp.** ``max_relative_target`` only bounds
  the *per-step delta*; nothing stops a policy that persistently saturates it
  from walking the arm into a joint limit or the table. :func:`clamp_action`
  + :func:`resolve_joint_limits` add a client-side absolute-position clamp,
  independent of (and in addition to) lerobot's server-side relative clamp.

Pure / dependency-light: only ``numpy`` (already a hard dep of this
package) and the stdlib. No ``lerobot`` import at module scope — every
function accepts a duck-typed ``robot`` object (or ``None``) and degrades
to a hardcoded safety floor. Never raises on malformed/missing calibration.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Canonical SO-101 joint order (Feetech servo id 1..6). Matches
# mappers.SO101_CANONICAL_MOTORS and the reference implementation in
# lerobot_isaac_deploy.arm_motor_writer.
SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# FIRST-PASS SAFETY FLOOR — conservative per-joint bounds, ported verbatim
# from lerobot_isaac_deploy.arm_motor_writer (degrees for the 5 arm joints;
# percent-of-jaw-travel for the gripper, which is always MotorNormMode.
# RANGE_0_100 regardless of the follower's ``use_degrees`` setting).
# Calibration-derived limits are always INTERSECTED with this floor — a
# symmetric calibration can never widen the safe range beyond it. The
# elbow_flex -10 deg floor specifically limits negative extent to avoid the
# table.
DEFAULT_JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-10.0, 90.0),  # negative extent limited — avoids the table
    "wrist_flex": (-90.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (0.0, 100.0),  # 0% closed ... 100% open
}

# Internal settle tolerance for ramped_home(); small enough that the ramp
# always takes its final step to *exactly* zero rather than stopping short.
_HOME_TOL_DEG = 1e-6

# Ticks-per-revolution for the sts3215 servo used throughout the SO-101
# (12-bit encoder: 4096 ticks = 360 deg). Matches the reference implementation
# and lerobot's own DEGREES norm-mode conversion (model_resolution_table - 1
# for sts3215 is 4095; using 4096 here matches the reference's approximation
# and is well within the safety margin the floor intersection provides).
_TICKS_PER_REV = 4096.0


class RateCeilingExceeded(RuntimeError):
    """Raised by :func:`check_rate_ceiling` when the worst-joint deg/s implied
    by the clamp settings exceeds the configured ceiling and no override was
    given."""


def _cal_get(cal: Any, name: str) -> Any | None:
    """Best-effort ``cal[name]`` / ``cal.get(name)`` lookup. Never raises."""
    if cal is None:
        return None
    try:
        if hasattr(cal, "get"):
            return cal.get(name)
        return cal[name]
    except Exception:  # noqa: BLE001
        return None


def _cal_span_deg(name: str, robot: Any | None) -> tuple[float, float] | None:
    """Read one joint's calibrated (min_deg, max_deg) from ``robot.calibration``.

    Returns ``None`` when calibration is missing/malformed for this joint
    (caller falls back to the hardcoded floor). Never raises.
    """
    cal = getattr(robot, "calibration", None) if robot is not None else None
    c = _cal_get(cal, name)
    if c is None:
        return None
    range_min = getattr(c, "range_min", None)
    range_max = getattr(c, "range_max", None)
    if range_min is None or range_max is None:
        return None
    try:
        # Approximate tick->degree: 4096 ticks = 360 deg, centered at 0.
        # (Matches lerobot_isaac_deploy.arm_motor_writer.read_joint_limits.)
        lo = float(range_min) / _TICKS_PER_REV * 360.0 - 180.0
        hi = float(range_max) / _TICKS_PER_REV * 360.0 - 180.0
    except (TypeError, ValueError):
        return None
    return lo, hi


def resolve_joint_limits(
    robot: Any | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve per-joint absolute (min_deg, max_deg) limits.

    Uses ``robot.calibration`` (a ``dict[str, MotorCalibration]``-like mapping
    with ``range_min`` / ``range_max`` in raw ticks) when available, converts
    ticks -> degrees, then **intersects** with :data:`DEFAULT_JOINT_LIMITS_DEG`
    so a symmetric or otherwise-wide calibration can never widen the safe
    range. Falls back to the hardcoded floor for any joint whose calibration
    is missing or malformed. Never raises.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        ``(lo, hi)`` — each a ``{joint_name: degrees}`` dict covering all of
        :data:`SO101_JOINT_NAMES`.
    """
    lo = {name: bounds[0] for name, bounds in DEFAULT_JOINT_LIMITS_DEG.items()}
    hi = {name: bounds[1] for name, bounds in DEFAULT_JOINT_LIMITS_DEG.items()}

    cal = getattr(robot, "calibration", None) if robot is not None else None
    if not cal:
        if robot is not None:
            logger.warning(
                "robot has no usable calibration — falling back to the "
                "hardcoded safety floor for all joints: %s",
                DEFAULT_JOINT_LIMITS_DEG,
            )
        return lo, hi

    for name in SO101_JOINT_NAMES:
        span = _cal_span_deg(name, robot)
        if span is None:
            logger.warning(
                "no usable calibration for joint %r — using hardcoded floor %s",
                name,
                DEFAULT_JOINT_LIMITS_DEG[name],
            )
            continue
        cal_lo, cal_hi = span
        # Intersect with the floor — cal-derived limits can only TIGHTEN,
        # never widen, the safe range (a symmetric +/-180 calibration must
        # not defeat the elbow_flex table-avoidance floor, for example).
        lo[name] = max(lo[name], min(cal_lo, cal_hi))
        hi[name] = min(hi[name], max(cal_lo, cal_hi))

    return lo, hi


def clamp_action(
    action: dict[str, float],
    lo: dict[str, float],
    hi: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Clamp ``"<joint>.pos"`` keys of ``action`` to absolute ``[lo, hi]``.

    Unknown keys (not ending in ``.pos``, or whose joint name isn't in
    ``lo``/``hi``) pass through untouched.

    Returns
    -------
    tuple[dict[str, float], list[str]]
        ``(clamped_action, clamped_joints)`` — the clamped dict plus the list
        of joint names whose value was actually changed (in encounter order,
        duplicates removed).
    """
    out: dict[str, float] = {}
    clamped: list[str] = []
    for key, value in action.items():
        if not key.endswith(".pos"):
            out[key] = value
            continue
        joint = key[: -len(".pos")]
        if joint not in lo or joint not in hi:
            out[key] = value
            continue
        lo_v, hi_v = lo[joint], hi[joint]
        if lo_v > hi_v:
            lo_v, hi_v = hi_v, lo_v
        clipped = min(max(value, lo_v), hi_v)
        if clipped != value and joint not in clamped:
            clamped.append(joint)
        out[key] = clipped
    return out, clamped


def _resolve_spans_deg(robot: Any | None = None) -> dict[str, float]:
    """Per-joint calibrated span (max - min), in degrees-equivalent units.

    Falls back to the hardcoded floor's span for any joint without usable
    calibration. Used by :func:`describe_clamp` / :func:`worst_case_rate` to
    convert a normalized-mode (-100..100 / 0..100) per-step delta into a
    legible real-world angle. Never raises.
    """
    spans = {name: (hi - lo) for name, (lo, hi) in DEFAULT_JOINT_LIMITS_DEG.items()}
    for name in SO101_JOINT_NAMES:
        span = _cal_span_deg(name, robot)
        if span is None:
            continue
        cal_lo, cal_hi = span
        spans[name] = abs(cal_hi - cal_lo)
    return spans


def per_joint_rates_deg(
    max_relative_target: float,
    use_degrees: bool,
    robot: Any | None = None,
    rate_hz: float = 30.0,
) -> dict[str, tuple[float, float]]:
    """Per-joint ``(deg_per_step, deg_per_s)`` implied by the clamp settings.

    * Degrees mode (``use_degrees=True``): the 5 arm joints move exactly
      ``max_relative_target`` deg/step (lerobot's ``MotorNormMode.DEGREES`` is
      a direct 1:1 unit — no calibration needed).
    * Normalized mode (or the gripper, which is ALWAYS
      ``MotorNormMode.RANGE_0_100`` regardless of ``use_degrees``): the delta
      is a percent of the joint's calibrated span, so
      ``deg_per_step = span_deg * max_relative_target / full_scale`` where
      ``full_scale`` is 200 for the +/-100 body-joint range and 100 for the
      0..100 gripper range.
    """
    spans = _resolve_spans_deg(robot)
    out: dict[str, tuple[float, float]] = {}
    for name in SO101_JOINT_NAMES:
        if use_degrees and name != "gripper":
            step = float(max_relative_target)
        else:
            full_scale = 100.0 if name == "gripper" else 200.0
            step = spans[name] * float(max_relative_target) / full_scale
        out[name] = (step, step * float(rate_hz))
    return out


def worst_case_rate(
    max_relative_target: float,
    use_degrees: bool,
    robot: Any | None = None,
    rate_hz: float = 30.0,
) -> tuple[str, float]:
    """Return ``(joint_name, deg_per_s)`` for the fastest-moving joint."""
    rates = per_joint_rates_deg(max_relative_target, use_degrees, robot, rate_hz)
    worst_joint = max(rates, key=lambda j: rates[j][1])
    return worst_joint, rates[worst_joint][1]


def check_rate_ceiling(
    max_relative_target: float,
    use_degrees: bool,
    robot: Any | None = None,
    rate_hz: float = 30.0,
    ceiling: float = 90.0,
    allow_fast: bool = False,
) -> tuple[str, float]:
    """Raise :class:`RateCeilingExceeded` if the worst-joint deg/s > ``ceiling``.

    This is the S1 safety-ceiling gate: it prevents a run from silently
    inheriting a dangerous rate (e.g. the old ``max_relative_target=5.0``
    reinterpreted under a unit-mode change) — every real-motor run path
    calls this before entering its control loop. ``allow_fast=True`` is the
    deliberate, explicit override.

    Returns ``(worst_joint, worst_rate_deg_per_s)`` on success (informational
    — callers may still want to log it even when not exceeded).
    """
    worst_joint, worst_rate = worst_case_rate(max_relative_target, use_degrees, robot, rate_hz)
    if worst_rate > ceiling and not allow_fast:
        raise RateCeilingExceeded(
            f"joint {worst_joint!r} implied rate {worst_rate:.1f} deg/s exceeds "
            f"the {ceiling:.1f} deg/s safety ceiling (max_relative_target="
            f"{max_relative_target!r}, use_degrees={use_degrees}, rate_hz={rate_hz}); "
            "pass allow_fast=True (CLI: --allow-fast) to override."
        )
    return worst_joint, worst_rate


def describe_clamp(
    max_relative_target: float,
    use_degrees: bool,
    robot: Any | None = None,
    rate_hz: float = 30.0,
) -> str:
    """Human-readable per-joint deg/step + deg/s table implied by the clamp.

    This is the S1 legibility fix: ``max_relative_target`` is silently NOT
    degrees unless ``use_degrees=True``; this table makes the real implied
    angular rate explicit regardless of which unit mode is active, so the
    mislabel can no longer hide a dangerously fast joint (e.g. a
    near-full-turn wrist_roll calibration in normalized mode).
    """
    rates = per_joint_rates_deg(max_relative_target, use_degrees, robot, rate_hz)
    unit = "DEGREES" if use_degrees else "NORMALIZED (legacy --as-normalized)"
    lines = [
        f"safety clamp: max_relative_target={max_relative_target} "
        f"unit_mode={unit} rate_hz={rate_hz}",
        f"  {'joint':<15}{'deg/step':>12}{'deg/s':>12}",
    ]
    for name in SO101_JOINT_NAMES:
        step, rate = rates[name]
        lines.append(f"  {name:<15}{step:>12.3f}{rate:>12.2f}")
    worst_joint, worst_rate = worst_case_rate(max_relative_target, use_degrees, robot, rate_hz)
    lines.append(f"  worst joint: {worst_joint} @ {worst_rate:.2f} deg/s")
    return "\n".join(lines)


def _extract_joint_pos(obs: dict[str, Any], motor_names: list[str]) -> dict[str, float]:
    """Best-effort ``{joint: obs["<joint>.pos"]}`` extraction; skips + warns on miss."""
    out: dict[str, float] = {}
    for name in motor_names:
        key = f"{name}.pos"
        if key not in obs:
            logger.warning(
                "ramped_home: observation is missing %r — skipping this joint "
                "for the remainder of the ramp",
                key,
            )
            continue
        try:
            out[name] = float(obs[key])
        except (TypeError, ValueError):
            logger.warning(
                "ramped_home: observation[%r]=%r is not a float — skipping this joint",
                key,
                obs[key],
            )
    return out


def ramped_home(
    robot: Any,
    motor_names: list[str],
    max_step_deg: float = 1.0,
    rate_hz: float = 30.0,
    settle_timeout_s: float = 5.0,
) -> None:
    """Ramp every joint in ``motor_names`` toward 0.0 at ``max_step_deg``/tick.

    Replaces the old single-shot "send zero, sleep(0.5), disconnect"
    home-on-exit pattern (S4): that pattern is not a ramp at all — it's one
    ``send_action`` call (itself only bounded by ``max_relative_target``),
    so an arm stopped far from zero can still slam toward it at the full
    per-step rate, then goes limp on ``disconnect()`` (torque disabled by
    default) wherever it lands.

    Each iteration re-reads ``robot.get_observation()`` (never assumes the
    previous target was reached), steps every joint toward 0.0 by at most
    ``max_step_deg``, and terminates when all resolvable joints are within
    tolerance of 0.0 or ``settle_timeout_s`` elapses. Joints missing from an
    observation are skipped (with a one-time warning) rather than raising —
    a partial home is safer than crashing mid-ramp.

    Never raises: ``robot.send_action`` failures are swallowed (best-effort,
    matching the reference ``lerobot_isaac_deploy.arm_motor_writer.ramped_home``
    behaviour) so a single bad write doesn't abort the whole ramp.
    """
    dt = 1.0 / rate_hz
    t_end = time.monotonic() + settle_timeout_s
    warned_missing: set[str] = set()

    while time.monotonic() < t_end:
        try:
            obs = robot.get_observation()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ramped_home: get_observation failed: %s", exc)
            return

        jp = _extract_joint_pos(obs, motor_names)
        missing = [m for m in motor_names if m not in jp]
        for m in missing:
            if m not in warned_missing:
                warned_missing.add(m)

        if not jp:
            logger.warning("ramped_home: no joints resolvable from observation — aborting ramp")
            return

        if all(abs(pos) < _HOME_TOL_DEG for pos in jp.values()):
            return

        target: dict[str, float] = {}
        for name, pos in jp.items():
            delta = -pos
            step = max(-max_step_deg, min(max_step_deg, delta))
            target[f"{name}.pos"] = pos + step

        try:
            robot.send_action(target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ramped_home: send_action failed: %s", exc)

        time.sleep(dt)

    logger.warning(
        "ramped_home: settle_timeout_s=%.1f elapsed before reaching home", settle_timeout_s
    )
