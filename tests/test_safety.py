"""Tests for robot_data_runner.safety.require_interactive (S5 fix).

No hardware. Pure function against a monkeypatched sys.stdin.isatty().
"""

from __future__ import annotations

import sys

import pytest

from robot_data_runner import safety
from robot_data_runner.safety import InteractiveRequiredError, require_interactive


def test_tty_always_proceeds(monkeypatch):
    """A real TTY never refuses, regardless of execute or the opt-out."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    require_interactive("ctx", execute=True)
    require_interactive("ctx", execute=False)
    require_interactive("ctx", execute=False, allow_non_interactive=True)


def test_non_tty_execute_true_always_refuses(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(InteractiveRequiredError):
        require_interactive("ctx", execute=True)


def test_non_tty_execute_true_opt_out_does_not_rescue(monkeypatch):
    """The opt-out flag is documented as execute=False-only; execute=True
    must still refuse even when allow_non_interactive=True is passed."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(InteractiveRequiredError):
        require_interactive("ctx", execute=True, allow_non_interactive=True)


def test_non_tty_dry_run_refuses_by_default(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(InteractiveRequiredError):
        require_interactive("ctx", execute=False)


def test_non_tty_dry_run_proceeds_with_opt_out(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    require_interactive("ctx", execute=False, allow_non_interactive=True)


# ---------------------------------------------------------------------------
# Joint-current / thermal watchdog (2026-09-10)
# ---------------------------------------------------------------------------


class _FakeBus:
    """Duck-typed Feetech bus.

    ``fail`` makes every read raise, as a dropped packet would. ``no_sync`` removes
    ``sync_read`` so the per-motor fallback path is exercised — without this the
    batched and fallback paths could not be told apart in tests.
    """

    def __init__(
        self,
        current=10.0,
        temperature=30.0,
        protection=250.0,
        max_temp=70.0,
        fail=False,
        no_sync=False,
        per_motor_current=None,
    ):
        self.sync_reads = 0
        self.per_motor_current = per_motor_current or {}
        self._no_sync = no_sync
        self.motors = ["shoulder_pan", "gripper"]
        self._vals = {
            "Present_Current": current,
            "Present_Temperature": temperature,
            "Protection_Current": protection,
            "Max_Temperature_Limit": max_temp,
        }
        self.fail = fail
        self.reads = 0

    def read(self, register, motor):
        self.reads += 1
        if self.fail:
            raise RuntimeError("bus timeout")
        return self.per_motor_current.get((register, motor), self._vals[register])

    def sync_read(self, register, motors=None, *, normalize=True, num_retry=0):
        """Batched read — one round-trip for all motors, like the real Feetech bus."""
        if self._no_sync:
            raise AttributeError("this bus has no sync_read")
        self.sync_reads += 1
        if self.fail:
            raise RuntimeError("bus timeout")
        assert normalize is False, "watchdog must read raw counts, not normalized"
        return {
            m: self.per_motor_current.get((register, m), self._vals[register]) for m in self.motors
        }


class _FakeRobot:
    def __init__(self, **kw):
        self.bus = _FakeBus(**kw)


class _Cfg:
    def __init__(self, hardware_watchdog=True, hw_every_n_steps=1, hw_breach_steps=3):
        self.hardware_watchdog = hardware_watchdog
        self.hw_every_n_steps = hw_every_n_steps
        self.hw_breach_steps = hw_breach_steps


def test_limits_derive_from_servo_protection_registers():
    """Trip points must come off the servo, not a hardcoded constant."""
    lim = safety.resolve_hardware_limits(_FakeRobot(protection=300.0, max_temp=80.0))
    assert lim.resolved_from_hardware is True
    assert lim.current_raw["gripper"] == pytest.approx(300.0 * 0.8)
    assert lim.temperature_c["gripper"] == pytest.approx(80.0 - 10.0)


def test_limits_fall_back_when_registers_unreadable():
    """Enumerable motors but unreadable registers -> fallback limits, flagged not-measured.

    The guard must still be armed (a fallback trip point beats none at all), but
    ``resolved_from_hardware`` must be False so the caller can log that the limits are
    a guess rather than a measurement.
    """
    lim = safety.resolve_hardware_limits(_FakeRobot(fail=True))
    assert lim.resolved_from_hardware is False
    expected_cur = safety.FALLBACK_PROTECTION_CURRENT_RAW * 0.8
    expected_temp = safety.FALLBACK_MAX_TEMPERATURE_C - 10.0
    assert set(lim.current_raw) == {"shoulder_pan", "gripper"}
    assert all(v == pytest.approx(expected_cur) for v in lim.current_raw.values())
    assert all(v == pytest.approx(expected_temp) for v in lim.temperature_c.values())


def test_limits_empty_when_motor_list_itself_unavailable():
    """No bus at all -> nothing to guard, and still no exception."""
    lim = safety.resolve_hardware_limits(object())
    assert lim.current_raw == {}
    assert lim.resolved_from_hardware is False


def test_limits_never_raise_on_a_garbage_robot():
    """Contract: degrade, never crash."""
    lim = safety.resolve_hardware_limits(object())
    assert isinstance(lim, safety.HardwareLimits)
    assert lim.resolved_from_hardware is False


def test_watchdog_is_noop_when_not_armed():
    """Existing callers that never set hw_limits must be completely unaffected."""
    r = _FakeRobot(current=99999.0)
    m = safety.SafetyMonitor()
    for _ in range(50):
        m.observe_hardware(r)
    assert m.stop_flag is False
    assert m.fault is None
    assert r.bus.reads == 0, "disarmed watchdog must not touch the serial bus"


def test_healthy_arm_does_not_trip():
    r = _FakeRobot(current=10.0, temperature=30.0)
    m = safety.SafetyMonitor(hw_limits=safety.resolve_hardware_limits(r), hw_every_n_steps=1)
    for _ in range(20):
        m.observe_hardware(r)
    assert m.stop_flag is False and m.fault is None


def test_overcurrent_trips_only_after_consecutive_breaches():
    """A single transient spike must not abort an episode."""
    healthy = _FakeRobot()
    limits = safety.resolve_hardware_limits(healthy)
    hot = _FakeRobot(current=900.0)
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1, hw_breach_steps=3)
    m.observe_hardware(hot)
    assert m.stop_flag is False, "must not trip on the first breached sample"
    m.observe_hardware(hot)
    assert m.stop_flag is False, "must not trip on the second"
    m.observe_hardware(hot)
    assert m.stop_flag is True
    assert "current" in (m.fault or "")


def test_transient_spike_resets_the_streak():
    healthy = _FakeRobot()
    limits = safety.resolve_hardware_limits(healthy)
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1, hw_breach_steps=3)
    m.observe_hardware(_FakeRobot(current=900.0))
    m.observe_hardware(_FakeRobot(current=900.0))
    m.observe_hardware(healthy)  # recovered
    m.observe_hardware(_FakeRobot(current=900.0))
    assert m.stop_flag is False, "streak must reset after a healthy sample"


def test_overtemperature_trips():
    healthy = _FakeRobot()
    limits = safety.resolve_hardware_limits(healthy)
    warm = _FakeRobot(temperature=95.0)
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1, hw_breach_steps=2)
    m.observe_hardware(warm)
    m.observe_hardware(warm)
    assert m.stop_flag is True
    assert "temperature" in (m.fault or "")


def test_sampling_is_throttled():
    """At 30 Hz the guard must not sample every step or it competes with motor writes.

    Asserts the BEHAVIOUR (how many times the bus was sampled) rather than a raw
    register-read count, which is an implementation detail that changes with batching
    or with the number of registers watched.
    """
    r = _FakeRobot()
    m = safety.SafetyMonitor(hw_limits=safety.resolve_hardware_limits(r), hw_every_n_steps=15)
    r.bus.sync_reads = 0
    r.bus.reads = 0
    for _ in range(30):
        m.observe_hardware(r)
    # 30 steps / every-15 = 2 samples; each samples 2 registers in 1 batched call each
    assert r.bus.sync_reads == 4, f"expected 4 batched reads, got {r.bus.sync_reads}"


def test_batched_path_is_used_when_available():
    """sync_read must be preferred: 2 round-trips per sample, not 2 x n_motors."""
    r = _FakeRobot()
    m = safety.SafetyMonitor(hw_limits=safety.resolve_hardware_limits(r), hw_every_n_steps=1)
    r.bus.sync_reads = 0
    r.bus.reads = 0
    m.observe_hardware(r)
    assert r.bus.sync_reads == 2, "one batched call per register"
    assert r.bus.reads == 0, "must not fall back to per-motor reads when sync_read works"


def test_falls_back_to_per_motor_when_sync_read_missing():
    """An older bus without sync_read must still be guarded."""
    r = _FakeRobot(no_sync=True)
    m = safety.SafetyMonitor(hw_limits=safety.resolve_hardware_limits(r), hw_every_n_steps=1)
    r.bus.reads = 0
    m.observe_hardware(r)
    assert r.bus.reads > 0, "fallback path must read per motor"
    assert m.stop_flag is False


def test_read_failures_do_not_raise_or_trip():
    """A flaky bus must not fabricate a fault."""
    healthy = _FakeRobot()
    limits = safety.resolve_hardware_limits(healthy)
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1, hw_breach_steps=2)
    flaky = _FakeRobot(fail=True)
    for _ in range(30):
        m.observe_hardware(flaky)
    assert m.stop_flag is False
    assert m.fault is None


def test_hardware_summary_reports_peaks_and_fault():
    healthy = _FakeRobot()
    limits = safety.resolve_hardware_limits(healthy)
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1, hw_breach_steps=1)
    m.observe_hardware(_FakeRobot(current=900.0, temperature=44.0))
    s = m.hardware_summary()
    assert s["peak_current_raw"]["shoulder_pan"] == 900.0
    assert s["peak_temperature_c"]["shoulder_pan"] == 44.0
    assert s["fault"] is not None
    assert s["limits_resolved_from_hardware"] is True


def test_attach_respects_the_disable_flag():
    m = safety.SafetyMonitor()
    safety.attach_hardware_watchdog(m, _FakeRobot(), _Cfg(hardware_watchdog=False))
    assert m.hw_limits is None


def test_attach_arms_and_carries_config_knobs():
    m = safety.SafetyMonitor()
    safety.attach_hardware_watchdog(m, _FakeRobot(), _Cfg(hw_every_n_steps=7, hw_breach_steps=4))
    assert m.hw_limits is not None
    assert m.hw_every_n_steps == 7
    assert m.hw_breach_steps == 4


def test_attach_never_raises_on_a_bad_robot():
    m = safety.SafetyMonitor()
    safety.attach_hardware_watchdog(m, object(), _Cfg())
    assert m.stop_flag is False


def test_attach_tolerates_a_config_without_the_new_fields():
    """A config object lacking hardware_watchdog entirely must disarm cleanly."""
    m = safety.SafetyMonitor()
    safety.attach_hardware_watchdog(m, _FakeRobot(), object())
    assert m.hw_limits is None


def test_attach_tolerates_a_partial_config():
    """Opted IN but missing the tuning knobs -> arm with defaults, do not crash.

    The previous version of this test passed a bare object(), which lacks
    hardware_watchdog too and therefore early-returned before ever reaching the
    getattr defaults it claimed to cover (found in adversarial review 2026-09-10).
    """

    class _PartialCfg:
        hardware_watchdog = True  # opted in, but no hw_every_n_steps / hw_breach_steps

    m = safety.SafetyMonitor()
    safety.attach_hardware_watchdog(m, _FakeRobot(), _PartialCfg())
    assert m.hw_limits is not None, "must arm when opted in"
    assert m.hw_every_n_steps == safety.DEFAULT_HW_EVERY_N_STEPS
    assert m.hw_breach_steps == 3


def test_watchdog_defaults_agree_across_modules():
    """Guard against the exact drift this refactor fixed.

    The tuning defaults live in safety.py and are consumed by SafetyMonitor.__init__,
    attach_hardware_watchdog's getattr fallbacks, RunnerConfig and four CLIs. When they
    were bare literals they drifted within a single change (the getattr fallback stayed
    at 15 after everything else moved to 5). Assert every copy still agrees.
    """
    import inspect

    from robot_data_runner.config import RunnerConfig

    assert RunnerConfig.hw_every_n_steps == safety.DEFAULT_HW_EVERY_N_STEPS
    assert RunnerConfig.hw_breach_steps == safety.DEFAULT_HW_BREACH_STEPS

    sig = inspect.signature(safety.SafetyMonitor.__init__)
    assert sig.parameters["hw_every_n_steps"].default == safety.DEFAULT_HW_EVERY_N_STEPS
    assert sig.parameters["hw_breach_steps"].default == safety.DEFAULT_HW_BREACH_STEPS

    class _PartialCfg:
        hardware_watchdog = True

    m = safety.SafetyMonitor()
    safety.attach_hardware_watchdog(m, _FakeRobot(), _PartialCfg())
    assert m.hw_every_n_steps == safety.DEFAULT_HW_EVERY_N_STEPS
    assert m.hw_breach_steps == safety.DEFAULT_HW_BREACH_STEPS


# ---------------------------------------------------------------------------
# Findings from the 2026-09-10 adversarial review
# ---------------------------------------------------------------------------


def test_nan_current_does_not_read_as_healthy():
    """A7: NaN > limit is False, so a NaN reading must not silently mean 'fine'.

    Before the fix the comparison simply evaluated False and the motor was treated as
    healthy, so a sensor returning NaN disabled the guard for that joint with no trace.
    """
    healthy = _FakeRobot()
    limits = safety.resolve_hardware_limits(healthy)
    nan_bus = _FakeRobot(current=float("nan"))
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1, hw_breach_steps=2)
    for _ in range(6):
        m.observe_hardware(nan_bus)
    assert m.stop_flag is False, "NaN is unknown, not a breach — must not trip"
    assert "shoulder_pan" not in m.peak_current_raw, "NaN must not enter the peak record"


def test_nan_does_not_corrupt_an_existing_peak():
    """A NaN sample after real readings must leave the recorded peak intact."""
    healthy = _FakeRobot()
    limits = safety.resolve_hardware_limits(healthy)
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1)
    m.observe_hardware(_FakeRobot(current=42.0))
    m.observe_hardware(_FakeRobot(current=float("nan")))
    assert m.peak_current_raw["shoulder_pan"] == 42.0


def test_all_motors_are_evaluated_before_tripping():
    """C2: a second straining servo must be recorded, not hidden behind the first.

    The original loop returned on the first tripped motor and stop_flag then blocked
    every later sample, so a simultaneously-damaged second servo never appeared in the
    run record.
    """
    healthy = _FakeRobot()
    limits = safety.resolve_hardware_limits(healthy)
    both_hot = _FakeRobot(current=900.0)  # every motor over the trip point
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1, hw_breach_steps=1)
    m.observe_hardware(both_hot)
    assert m.stop_flag is True
    summary = m.hardware_summary()
    assert set(summary["peak_current_raw"]) == {"shoulder_pan", "gripper"}, (
        "both motors must appear in the record, not just the first to trip"
    )
    assert "shoulder_pan" in m.fault and "gripper" in m.fault, (
        f"fault must name every breaching motor, got: {m.fault}"
    )


def test_motor_without_a_trip_point_is_warned_about(caplog):
    """A9: an unguarded motor must say so rather than silently no-op."""
    r = _FakeRobot()
    limits = safety.resolve_hardware_limits(r)
    limits.current_raw.pop("gripper")
    limits.temperature_c.pop("gripper")
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1)
    with caplog.at_level("WARNING"):
        m.observe_hardware(r)
    assert any("unguarded" in rec.message for rec in caplog.records), (
        f"expected an unguarded-motor warning, got: {[r.message for r in caplog.records]}"
    )


def test_unguarded_warning_is_emitted_once_per_motor():
    """It must not spam the log every sample for the whole run."""
    r = _FakeRobot()
    limits = safety.resolve_hardware_limits(r)
    limits.current_raw.pop("gripper")
    limits.temperature_c.pop("gripper")
    m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=1)
    for _ in range(10):
        m.observe_hardware(r)
    assert m._unguarded_warned == {"gripper"}


def test_zero_and_negative_tuning_knobs_are_clamped_not_crashing():
    """A12: a mistyped --hw-every-n-steps 0 must not divide-by-zero or disable the guard."""
    r = _FakeRobot()
    limits = safety.resolve_hardware_limits(r)
    for bad in (0, -5):
        m = safety.SafetyMonitor(hw_limits=limits, hw_every_n_steps=bad, hw_breach_steps=bad)
        assert m.hw_every_n_steps >= 1
        assert m.hw_breach_steps >= 1
        m.observe_hardware(r)  # must not raise
        assert m.stop_flag is False


def test_zero_protection_current_still_yields_a_usable_trip_point():
    """A servo reporting Protection_Current=0 must not produce a trip point of 0.

    A 0 trip point would fire on the first non-zero reading (false positive); treating
    it as unreadable and falling back is the safe interpretation.
    """
    lim = safety.resolve_hardware_limits(_FakeRobot(protection=0.0))
    assert lim.current_raw["gripper"] > 0.0


# ---------------------------------------------------------------------------
# safe_connect — partial-torque-on-connect-failure (NEXT_STEPS item 0)
#
# Observed on real hardware 2026-09-11: lerobot's MotorsBus.write() defaults to
# num_retry=0, so one dropped status packet aborts SO101Follower.connect()
# mid-configure. Servos already visited are left ENERGISED, the exception escapes
# connect(), and a caller whose disconnect() sits in a try/finally AFTER the connect
# call never runs it — the arm holds position with no live session.
# ---------------------------------------------------------------------------


class _PartialConnectBus:
    """Bus whose connect enables torque motor-by-motor then fails partway."""

    def __init__(self, fail_at_index=3, release_fails=False):
        self.motors = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self.torque = dict.fromkeys(self.motors, 0)
        self.fail_at_index = fail_at_index
        self.release_fails = release_fails
        self.release_attempts = []

    def write(self, register, motor, value, **kw):
        if register != "Torque_Enable":
            return
        if value == 0:
            self.release_attempts.append(motor)
            if self.release_fails:
                raise RuntimeError("bus timeout on release")
        self.torque[motor] = value

    def read(self, register, motor, **kw):
        if register == "Torque_Enable":
            return self.torque[motor]
        raise KeyError(register)


class _PartialConnectRobot:
    def __init__(self, **kw):
        self.bus = _PartialConnectBus(**kw)
        self.disconnect_called = False

    def connect(self, calibrate=True):
        for i, m in enumerate(self.bus.motors):
            if i == self.bus.fail_at_index:
                raise ConnectionError(
                    f"Failed to write 'Torque_Enable' on id_={i + 1} with '1' after 1 tries. "
                    "[TxRxResult] There is no status packet!"
                )
            self.bus.write("Torque_Enable", m, 1)

    def disconnect(self):
        self.disconnect_called = True


def test_partial_connect_leaves_torque_on_without_the_guard():
    """Characterise the bug: raw connect() energises some servos then raises."""
    r = _PartialConnectRobot()
    with pytest.raises(ConnectionError):
        r.connect()
    assert sum(r.bus.torque.values()) == 3, "expected 3 servos left energised"


def test_safe_connect_releases_torque_on_partial_failure():
    r = _PartialConnectRobot()
    with pytest.raises(ConnectionError):
        safety.safe_connect(r)
    assert sum(r.bus.torque.values()) == 0, (
        f"servos left energised after a failed connect: {r.bus.torque}"
    )


def test_safe_connect_reraises_the_original_error():
    """The release sweep must not mask why the connect failed."""
    r = _PartialConnectRobot()
    with pytest.raises(ConnectionError, match="no status packet"):
        safety.safe_connect(r)


def test_safe_connect_attempts_release_on_every_motor():
    """Release must not stop at the first motor, nor skip ones never energised."""
    r = _PartialConnectRobot()
    with pytest.raises(ConnectionError):
        safety.safe_connect(r)
    assert set(r.bus.release_attempts) == set(r.bus.motors)


def test_safe_connect_survives_a_failing_release():
    """A bus too sick to release must still surface the ORIGINAL connect error."""
    r = _PartialConnectRobot(release_fails=True)
    with pytest.raises(ConnectionError, match="no status packet"):
        safety.safe_connect(r)


def test_safe_connect_is_a_passthrough_on_success():
    class _OK:
        def __init__(self):
            self.connected = False
            self.kwargs = None

        def connect(self, calibrate=True):
            self.connected = True
            self.kwargs = {"calibrate": calibrate}

    r = _OK()
    safety.safe_connect(r, calibrate=False)
    assert r.connected is True
    assert r.kwargs == {"calibrate": False}


def test_safe_connect_never_raises_from_a_garbage_robot():
    """A robot with no bus at all: the original error still wins."""

    class _NoBus:
        def connect(self, calibrate=True):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        safety.safe_connect(_NoBus())


def test_no_entry_point_calls_robot_connect_directly():
    """Canary: every run path must go through safe_connect.

    A bare ``robot.connect()`` re-opens the partial-torque window this wrapper
    exists to close, and does so silently — the arm is left energised only when a
    packet happens to drop, so tests and normal runs would never show it. Counting
    the call sites mechanically beats trusting a cited line number, which goes stale
    the moment either file is next edited.
    """
    import pathlib
    import re

    pkg = pathlib.Path(safety.__file__).parent
    offenders = []
    for path in pkg.glob("*.py"):
        if path.name == "safety.py":
            continue  # safe_connect itself must call robot.connect(**kwargs)
        for n, raw in enumerate(path.read_text().splitlines(), 1):
            # Strip string literals and comments first: a log message that MENTIONS
            # robot.connect() is not a call site. (This canary's first version
            # flagged its own error strings.)
            line = re.sub(r'"[^"]*"|\'[^\']*\'', '""', raw).split("#", 1)[0]
            if re.search(r"(?<!safe_)\brobot\.connect\s*\(", line):
                offenders.append(f"{path.name}:{n}: {raw.strip()}")
    assert not offenders, (
        "these call robot.connect() directly instead of safety.safe_connect():\n  "
        + "\n  ".join(offenders)
    )


def test_safe_connect_is_actually_wired_into_the_run_paths():
    """The mirror of the canary: the guarded call must be present, not just absent."""
    import pathlib

    pkg = pathlib.Path(safety.__file__).parent
    wired = {path.name for path in pkg.glob("*.py") if "safe_connect(robot)" in path.read_text()}
    assert {"runner.py", "episode_runner.py"} <= wired, f"only wired in: {wired}"


# ---------------------------------------------------------------------------
# Non-finite actions (NEXT_STEPS item 2, widened 2026-09-12)
#
# Filed as "_same() can never flag a constant-NaN action". Investigating it found
# two worse siblings: a NaN action passes straight THROUGH clamp_action to
# send_action, and is reported in the clamped list while not being clamped
# (nan != nan makes the "did I change it" check true). NaN must never reach a servo.
# ---------------------------------------------------------------------------

NAN = float("nan")
INF = float("inf")


def test_same_treats_two_nans_as_unchanged():
    """abs(nan-nan) < eps is False, so a constant-NaN stream reset the streak forever."""
    m = safety.SafetyMonitor()
    a = {"shoulder_pan.pos": NAN}
    assert m._same(a, dict(a)) is True


def test_constant_nan_accumulates_a_streak():
    """The stuck-action warning's own text names NaN; it must be reachable."""
    m = safety.SafetyMonitor(repeat_warn_steps=3)
    for _ in range(5):
        m.observe({"shoulder_pan.pos": NAN})
    assert m._streak >= 3, f"streak never accumulated: {m._streak}"


def test_same_still_distinguishes_nan_from_a_real_value():
    m = safety.SafetyMonitor()
    assert m._same({"a.pos": NAN}, {"a.pos": 1.0}) is False
    assert m._same({"a.pos": 1.0}, {"a.pos": NAN}) is False


def test_observe_trips_stop_flag_on_a_non_finite_action():
    """A NaN action is never valid — stop, do not wait 30 steps for a warning."""
    for bad in (NAN, INF, -INF):
        m = safety.SafetyMonitor()
        m.observe({"shoulder_pan.pos": bad})
        assert m.stop_flag is True, f"{bad} did not trip the monitor"
        assert m.fault is not None and "finite" in m.fault.lower()


def test_observe_does_not_trip_on_normal_actions():
    m = safety.SafetyMonitor()
    for v in (0.0, -90.0, 90.0, 1e-9):
        m.observe({"shoulder_pan.pos": v})
    assert m.stop_flag is False
    assert m.fault is None


def test_clamp_action_contains_a_non_finite_value():
    """The absolute joint clamp must not pass NaN through to send_action."""
    from robot_data_runner.safety_limits import clamp_action, resolve_joint_limits

    lo, hi = resolve_joint_limits(None)
    out, clamped = clamp_action({"shoulder_pan.pos": NAN}, lo, hi)
    v = out["shoulder_pan.pos"]
    assert v == v, "NaN survived the clamp"
    assert lo["shoulder_pan"] <= v <= hi["shoulder_pan"]
    assert "shoulder_pan" in clamped


def test_clamp_action_contains_infinities():
    from robot_data_runner.safety_limits import clamp_action, resolve_joint_limits

    lo, hi = resolve_joint_limits(None)
    out, _ = clamp_action({"shoulder_pan.pos": INF, "elbow_flex.pos": -INF}, lo, hi)
    assert out["shoulder_pan.pos"] == hi["shoulder_pan"]
    assert out["elbow_flex.pos"] == lo["elbow_flex"]
