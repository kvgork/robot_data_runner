"""SafetyMonitor — stuck-action watchdog + SIGINT handler.

Sits inside the runner loop and tracks consecutive same-action steps.
Warns once when the threshold is hit. Also installs SIGINT/SIGTERM
handlers that flip the shared `stop_flag`.

Also home to :func:`require_interactive` (S5 fix, 2026-08 domain safety
review): a TTY gate for any run path that drives the arm via a
human-interlocked session (reset/verdict prompts). See its docstring for
why this is needed — short version: catching ``EOFError`` and "assuming
continue" under a detached/non-interactive launch (nohup/setsid/a pipe)
silently deletes the operator interlock AND fabricates a clean-looking
0/N result.

IMPORTANT — no software e-stop: a Python ``signal`` handler cannot
interrupt a blocked serial read/write (PEP 475 retries syscalls on
``EINTR``), so flipping ``stop_flag`` from SIGINT only takes effect at the
next checked boundary (top of a loop, or after a blocking ``input()``
call returns). Ctrl-C is therefore NOT a reliable e-stop; the only
reliable stop is cutting power to the arm. Callers that drive the arm
must print a startup banner saying so (see ``runner._NO_ESTOP_BANNER``).
"""

from __future__ import annotations

import logging
import math
import signal
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class InteractiveRequiredError(RuntimeError):
    """Raised by :func:`require_interactive` when stdin is not a TTY.

    Two distinct refusal modes:

    * ``execute=True`` (this run will drive the arm) — ALWAYS refuses;
      there is no opt-out. A closed-loop hardware session must have an
      operator at the prompt able to type ``abort``.
    * ``execute=False`` (dry-run, no motor writes) — refuses UNLESS the
      caller passes ``allow_non_interactive=True`` (CLI:
      ``--i-am-not-driving-an-arm``). This lets smoke tests / CI exercise
      the episode loop with fakes and no real TTY, while still requiring
      an explicit, named acknowledgement that no arm is being driven.
    """


def require_interactive(
    context: str,
    *,
    execute: bool,
    allow_non_interactive: bool = False,
) -> None:
    """Refuse to proceed with a human-interlocked run when stdin is not a TTY.

    S5 fix: running a closed-loop eval under ``nohup``/``setsid``/a pipe used
    to silently swallow the reset/verdict prompts' ``EOFError`` — episodes
    ran back-to-back with a live torqued arm and no operator, and every
    verdict prompt silently recorded a real FAIL, producing a clean-looking
    ``pc_success=0.0`` indistinguishable from a genuine 0/N. This check runs
    BEFORE any robot/policy setup so a misconfigured detached run fails
    fast, loud, and without side effects (no connect(), no policy load).

    Parameters
    ----------
    context:
        Short human-readable label for the calling run path (used in the
        error message only).
    execute:
        Whether this run will send real motor commands. When True, refusal
        is unconditional — ``allow_non_interactive`` cannot rescue it.
    allow_non_interactive:
        Opt-out for the ``execute=False`` case only (CLI:
        ``--i-am-not-driving-an-arm``).
    """
    if sys.stdin.isatty():
        return
    if execute:
        raise InteractiveRequiredError(
            f"{context}: refusing to start — stdin is not a TTY but "
            "execute=True would drive the arm with no operator present to "
            "reset the workspace or score episodes. This is almost always a "
            "detached/non-interactive launch (nohup / setsid / a pipe) — "
            "never run a hardware eval that way. There is no software "
            "e-stop (Ctrl-C cannot reliably interrupt a blocked serial "
            "read/write); the only reliable stop is cutting power. "
            "--i-am-not-driving-an-arm does NOT apply here — it only "
            "permits a non-interactive DRY run (execute=False)."
        )
    if not allow_non_interactive:
        raise InteractiveRequiredError(
            f"{context}: stdin is not a TTY (non-interactive). No motor "
            "commands will be sent (execute=False), but the reset/verdict "
            "prompts still cannot be answered by a human. Pass "
            "allow_non_interactive=True (CLI: --i-am-not-driving-an-arm) to "
            "confirm this is an intentional unattended dry run (e.g. a "
            "smoke test)."
        )


# ---------------------------------------------------------------------------
# Joint-current / thermal watchdog (2026-09-10)
# ---------------------------------------------------------------------------
#
# The SO-101 has NO force/torque sensing, so the only in-band signal that the arm
# is straining against something is the servo's own current draw. The published
# pattern for exactly this hardware class (arXiv:2401.14403, xArm-6 — same
# "current-readback only" category) is: sample joint current during autonomous
# operation and terminate the episode above a damage threshold. Without it, an RL
# policy exploring against a joint limit, the table, or a jammed object stalls the
# servo indefinitely — the failure mode most likely to end an unattended run in
# smoke rather than a bad reward.
#
# Thresholds are derived from the servos' OWN configured protection registers
# rather than hardcoded, so they track whatever the arm is actually set up with
# (lerobot's SO101Follower.configure() writes Protection_Current=250 and
# Overload_Torque=25 on the gripper specifically, leaving the arm joints at their
# firmware defaults — a hardcoded constant would be wrong for at least one of them).

#: Conservative fallbacks, used only when a servo's protection registers cannot be
#: read. Raw Feetech units (Protection_Current LSB ~= 6.5 mA) and degrees Celsius.
#: Watchdog tuning defaults. Declared ONCE — they were previously duplicated as bare
#: literals in SafetyMonitor.__init__, attach_hardware_watchdog's getattr fallbacks,
#: RunnerConfig and four CLIs, and immediately drifted (the getattr fallback still said
#: 15 after the others moved to 5). test_watchdog_defaults_agree_across_modules guards it.
DEFAULT_HW_EVERY_N_STEPS: int = 5
DEFAULT_HW_BREACH_STEPS: int = 3

FALLBACK_PROTECTION_CURRENT_RAW: int = 250
FALLBACK_MAX_TEMPERATURE_C: int = 70


@dataclass
class HardwareLimits:
    """Per-motor trip points for the current/thermal watchdog.

    Attributes
    ----------
    current_raw:
        Per-motor current trip point in raw servo units.
    temperature_c:
        Per-motor temperature trip point in degrees Celsius.
    resolved_from_hardware:
        True when at least one limit was read off a real servo; False when every
        value is a fallback. Lets a caller log the difference instead of silently
        trusting a guess.
    """

    current_raw: dict[str, float] = field(default_factory=dict)
    temperature_c: dict[str, float] = field(default_factory=dict)
    resolved_from_hardware: bool = False


def resolve_hardware_limits(
    robot: Any,
    *,
    current_frac: float = 0.8,
    temp_margin_c: float = 10.0,
) -> HardwareLimits:
    """Derive watchdog trip points from each servo's own protection registers.

    Parameters
    ----------
    robot:
        Duck-typed connected follower exposing ``robot.bus.motors`` and
        ``robot.bus.read(register, motor)``. ``None`` yields all-fallback limits.
    current_frac:
        Trip at this fraction of the servo's configured ``Protection_Current``, so
        the watchdog fires BEFORE the servo's own protection engages and drops
        torque mid-episode. Default 0.8.
    temp_margin_c:
        Trip this many degrees below the servo's ``Max_Temperature_Limit``. Default
        10 C.

    Returns
    -------
    HardwareLimits

    Notes
    -----
    Never raises. A servo that will not answer a register read gets the fallback
    value, matching this package's "degrade, never crash on hardware quirks"
    discipline (see :mod:`robot_data_runner.safety_limits`).
    """
    limits = HardwareLimits()
    motors = []
    try:
        motors = list(robot.bus.motors)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return limits

    for name in motors:
        prot = None
        maxt = None
        try:
            prot = float(robot.bus.read("Protection_Current", name))
        except Exception:  # noqa: BLE001
            prot = None
        try:
            maxt = float(robot.bus.read("Max_Temperature_Limit", name))
        except Exception:  # noqa: BLE001
            maxt = None

        if prot is not None and prot > 0:
            limits.current_raw[name] = prot * current_frac
            limits.resolved_from_hardware = True
        else:
            limits.current_raw[name] = FALLBACK_PROTECTION_CURRENT_RAW * current_frac

        if maxt is not None and maxt > 0:
            limits.temperature_c[name] = maxt - temp_margin_c
            limits.resolved_from_hardware = True
        else:
            limits.temperature_c[name] = FALLBACK_MAX_TEMPERATURE_C - temp_margin_c

    return limits


class SafetyMonitor:
    """Stuck-action watchdog. One instance per runner loop.

    Parameters
    ----------
    repeat_warn_steps:
        Consecutive ε-identical actions before WARNING is logged.
    epsilon:
        Per-dim tolerance for "same" action (default 1e-4).
    """

    def __init__(
        self,
        repeat_warn_steps: int = 30,
        epsilon: float = 1e-4,
        *,
        hw_limits: HardwareLimits | None = None,
        hw_every_n_steps: int = DEFAULT_HW_EVERY_N_STEPS,
        hw_breach_steps: int = DEFAULT_HW_BREACH_STEPS,
    ) -> None:
        self.repeat_warn_steps = repeat_warn_steps
        self.epsilon = epsilon
        self._last: dict[str, float] | None = None
        self._streak = 0
        self._warned = False
        self.stop_flag = False

        # --- current / thermal watchdog state ---
        self.hw_limits = hw_limits
        self.hw_every_n_steps = max(1, int(hw_every_n_steps))
        self.hw_breach_steps = max(1, int(hw_breach_steps))
        #: Set to a human-readable reason when the watchdog trips. ``None`` while healthy.
        self.fault: str | None = None
        #: Highest current seen per motor this run (raw units) — for post-run reporting.
        self.peak_current_raw: dict[str, float] = {}
        #: Highest temperature seen per motor this run (deg C).
        self.peak_temperature_c: dict[str, float] = {}
        self._hw_step = 0
        self._breach_streak: dict[str, int] = {}
        self._read_failures = 0
        self._read_fail_warned = False
        self._unguarded_warned: set[str] = set()

    # --- public API ---------------------------------------------------------
    def observe(self, action_dict: dict[str, float]) -> None:
        if self._last is not None and self._same(self._last, action_dict):
            self._streak += 1
            if self._streak == self.repeat_warn_steps and not self._warned:
                logger.warning(
                    "policy action unchanged for %d consecutive steps — "
                    "policy may be stuck or returning NaN; consider e-stop",
                    self._streak,
                )
                self._warned = True
        else:
            self._streak = 0
            self._warned = False
        self._last = dict(action_dict)

    def observe_hardware(self, robot: Any) -> None:
        """Sample joint current + temperature and trip ``stop_flag`` on sustained breach.

        Call once per control step, **after** ``robot.send_action`` — never between
        computing an action and writing it. A register read is a blocking serial
        round-trip; placing it before the write would let bus latency delay a motor
        command that has already been decided (found in adversarial review 2026-09-10).

        Batches via ``bus.sync_read`` — one packet per register across all motors
        (2 round-trips), not one per motor-register pair (12). Falls back to per-motor
        reads only if ``sync_read`` is unavailable.

        Sampling self-throttles to every ``hw_every_n_steps`` steps. A breach must
        persist ``hw_breach_steps`` consecutive *samples* before it trips, so a single
        transient spike does not abort an episode; a healthy sample resets that motor's
        streak. When it trips, ``stop_flag`` is set and :attr:`fault` carries the
        reason — the caller stops at the next step boundary exactly as for SIGINT, so
        ramped-home-on-exit still runs.

        Every motor is evaluated on every sample even after one trips, so
        :meth:`hardware_summary` reports a second straining servo rather than only the
        first one found.

        Parameters
        ----------
        robot:
            Duck-typed connected follower exposing ``robot.bus``.

        Notes
        -----
        Never raises. Register reads fail for mundane reasons (busy bus, dropped
        packet); those are counted and warned about once, never propagated, and never
        fabricate a fault. A no-op when :attr:`hw_limits` is ``None``.

        ``normalize=False`` is passed explicitly: lerobot normalises only
        ``Goal_Position`` / ``Present_Position``, so these registers are raw counts
        either way, but stating it keeps the comparison against the raw
        ``Protection_Current`` trip point unit-correct if that list ever grows.
        """
        if self.hw_limits is None or self.stop_flag:
            return

        self._hw_step += 1
        if self._hw_step % self.hw_every_n_steps != 0:
            return

        currents = self._sample_register(robot, "Present_Current")
        temps = self._sample_register(robot, "Present_Temperature")
        if currents is None and temps is None:
            return

        currents = currents or {}
        temps = temps or {}
        reasons: list[str] = []

        for name in set(currents) | set(temps):
            cur = currents.get(name)
            temp = temps.get(name)

            # NaN/Inf must not silently pass the > comparison as "healthy".
            if cur is not None and not math.isfinite(cur):
                cur = None
            if temp is not None and not math.isfinite(temp):
                temp = None

            if cur is not None:
                self.peak_current_raw[name] = max(self.peak_current_raw.get(name, cur), cur)
            if temp is not None:
                self.peak_temperature_c[name] = max(self.peak_temperature_c.get(name, temp), temp)

            cur_lim = self.hw_limits.current_raw.get(name)
            temp_lim = self.hw_limits.temperature_c.get(name)
            if cur_lim is None and temp_lim is None and name not in self._unguarded_warned:
                self._unguarded_warned.add(name)
                logger.warning(
                    "hardware watchdog: motor %r is being read but has NO trip point — "
                    "it is unguarded this run",
                    name,
                )

            reason = None
            if cur is not None and cur_lim is not None and cur > cur_lim:
                reason = (
                    f"{name}: current {cur:.0f} > trip {cur_lim:.0f} (raw units) — "
                    f"servo is straining; likely stalled against a limit, the table, "
                    f"or a jammed object"
                )
            elif temp is not None and temp_lim is not None and temp > temp_lim:
                reason = (
                    f"{name}: temperature {temp:.0f} C > trip {temp_lim:.0f} C — "
                    f"approaching the servo's own thermal cutoff, which would drop "
                    f"torque mid-episode"
                )

            if reason is None:
                self._breach_streak[name] = 0
                continue

            self._breach_streak[name] = self._breach_streak.get(name, 0) + 1
            if self._breach_streak[name] >= self.hw_breach_steps:
                reasons.append(reason)
            else:
                logger.warning(
                    "hardware watchdog breach %d/%d — %s",
                    self._breach_streak[name],
                    self.hw_breach_steps,
                    reason,
                )

        # Evaluate every motor first, THEN trip — so a second straining servo is
        # recorded rather than hidden behind the first one found.
        if reasons:
            self.fault = "; ".join(reasons)
            self.stop_flag = True
            logger.error(
                "HARDWARE WATCHDOG TRIP after %d consecutive samples — %s. "
                "Stopping at next step boundary.",
                self.hw_breach_steps,
                self.fault,
            )

    def _sample_register(self, robot: Any, register: str) -> dict[str, float] | None:
        """Read *register* from every motor in as few bus round-trips as possible."""
        try:
            raw = robot.bus.sync_read(register, normalize=False)
            return {k: float(v) for k, v in raw.items()}
        except Exception:  # noqa: BLE001 — fall back to per-motor reads
            pass
        try:
            motors = list(robot.bus.motors)
        except Exception:  # noqa: BLE001
            self._note_read_failure()
            return None
        out: dict[str, float] = {}
        for name in motors:
            val = self._read_reg(robot, register, name)
            if val is not None:
                out[name] = val
        return out or None

    def hardware_summary(self) -> dict[str, Any]:
        """Peak current/temperature per motor plus any fault, for the run record."""
        return {
            "peak_current_raw": dict(self.peak_current_raw),
            "peak_temperature_c": dict(self.peak_temperature_c),
            "fault": self.fault,
            "limits_resolved_from_hardware": (
                self.hw_limits.resolved_from_hardware if self.hw_limits else None
            ),
        }

    def _read_reg(self, robot: Any, register: str, motor: str) -> float | None:
        try:
            return float(robot.bus.read(register, motor))
        except Exception:  # noqa: BLE001
            self._note_read_failure()
            return None

    def _note_read_failure(self) -> None:
        self._read_failures += 1
        if self._read_failures >= 10 and not self._read_fail_warned:
            logger.warning(
                "hardware watchdog: %d register reads have failed — the current/thermal "
                "guard may not be protecting this run",
                self._read_failures,
            )
            self._read_fail_warned = True

    def install_signal_handlers(self) -> None:
        """Wire SIGINT + SIGTERM to flip ``stop_flag``."""

        def _h(_sig: int, _frame: Any) -> None:
            self.stop_flag = True
            logger.info("signal received — stopping at next step boundary")

        signal.signal(signal.SIGINT, _h)
        signal.signal(signal.SIGTERM, _h)

    # --- helpers ------------------------------------------------------------
    def _same(self, a: dict[str, float], b: dict[str, float]) -> bool:
        if set(a) != set(b):
            return False
        return all(abs(a[k] - b[k]) < self.epsilon for k in a)


def _release_all_torque(robot: Any) -> tuple[list[str], list[str]]:
    """Best-effort ``Torque_Enable = 0`` on every motor. Never raises.

    Returns ``(released, failed)`` motor-name lists. Motors that were already off
    are still written — cheaper than reading first, and a redundant write to an
    already-limp servo is harmless.
    """
    released: list[str] = []
    failed: list[str] = []
    try:
        motors = list(robot.bus.motors)
    except Exception:  # noqa: BLE001 — no reachable bus; nothing we can do
        return released, failed

    for name in motors:
        for attempt in (1, 2, 3):
            try:
                try:
                    robot.bus.write("Torque_Enable", name, 0, normalize=False, num_retry=3)
                except TypeError:
                    # Older/duck-typed buses without the keyword arguments.
                    robot.bus.write("Torque_Enable", name, 0)
                released.append(name)
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    failed.append(name)
    return released, failed


def safe_connect(robot: Any, **kwargs: Any) -> None:
    """``robot.connect(**kwargs)``, releasing torque if it fails part-way.

    Why this exists (observed on real hardware 2026-09-11). lerobot's
    ``MotorsBus.write()`` defaults to ``num_retry=0`` and ``SO101Follower.connect()``
    -> ``configure()`` writes several registers per motor. A single dropped status
    packet therefore aborts the connect mid-sweep::

        ConnectionError: Failed to write 'Torque_Enable' on id_=4 with '1' after 1
        tries. [TxRxResult] There is no status packet!

    Servos visited before the failure are already ENERGISED. Because the exception
    escapes ``connect()``, a caller whose ``disconnect()`` sits in a ``try/finally``
    *after* the connect call never runs it, and the arm is left holding position with
    no live session and nothing watching it. Observed live: servos 1-3 energised,
    4-6 not, all six otherwise healthy.

    This wrapper closes that window: on ANY failure it sweeps ``Torque_Enable = 0``
    across every reachable motor, logs what it managed to release, then re-raises the
    original exception unchanged so the caller still sees why the connect failed.

    ``BaseException`` is caught deliberately, not ``Exception`` — an operator pressing
    Ctrl-C while the arm is energising is exactly when torque most needs dropping.

    Raises
    ------
    BaseException
        Whatever ``robot.connect`` raised, re-raised with its traceback intact. The
        release sweep never raises on its own and never masks the original error.
    """
    try:
        robot.connect(**kwargs)
    except BaseException as exc:
        released, failed = _release_all_torque(robot)
        logger.error(
            "robot.connect() failed part-way (%s: %s). Servos visited before the "
            "failure may have been left ENERGISED, so torque was swept off: "
            "released=%s failed=%s. If any motor is in the failed list, CUT POWER — "
            "it may still be holding.",
            type(exc).__name__,
            exc,
            released or "none",
            failed or "none",
        )
        raise


def attach_hardware_watchdog(monitor: SafetyMonitor, robot: Any, cfg: Any) -> None:
    """Resolve trip points off the connected arm and arm the watchdog on *monitor*.

    Call once, AFTER ``robot.connect()`` — the protection registers can only be read
    from a live bus. A no-op when ``cfg.hardware_watchdog`` is False or absent, so an
    older config object silently leaves the guard disarmed rather than crashing.

    Never raises.
    """
    if not getattr(cfg, "hardware_watchdog", False):
        logger.info("hardware watchdog disabled by config — current/thermal guard OFF")
        return
    try:
        limits = resolve_hardware_limits(robot)
        monitor.hw_limits = limits
        monitor.hw_every_n_steps = max(
            1, int(getattr(cfg, "hw_every_n_steps", DEFAULT_HW_EVERY_N_STEPS))
        )
        monitor.hw_breach_steps = max(
            1, int(getattr(cfg, "hw_breach_steps", DEFAULT_HW_BREACH_STEPS))
        )
        if limits.resolved_from_hardware:
            logger.info(
                "hardware watchdog armed — trip points read from the servos "
                "(current %s, temperature %s C), sampled every %d steps",
                {k: round(v) for k, v in limits.current_raw.items()},
                {k: round(v) for k, v in limits.temperature_c.items()},
                monitor.hw_every_n_steps,
            )
        else:
            logger.warning(
                "hardware watchdog armed on FALLBACK limits — could not read any "
                "servo protection register; guard is a guess, not a measurement"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("hardware watchdog could not be armed (%s) — continuing without it", exc)
