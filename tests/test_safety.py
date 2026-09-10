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
    """Duck-typed Feetech bus. ``fail`` makes every read raise, as a dropped packet would."""

    def __init__(self, current=10.0, temperature=30.0, protection=250.0, max_temp=70.0, fail=False):
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
        return self._vals[register]


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
    """At 30 Hz the guard must not read every step or it saturates the bus."""
    r = _FakeRobot()
    m = safety.SafetyMonitor(hw_limits=safety.resolve_hardware_limits(r), hw_every_n_steps=15)
    r.bus.reads = 0
    for _ in range(30):
        m.observe_hardware(r)
    # 2 samples x 2 motors x 2 registers
    assert r.bus.reads == 8, f"expected 8 reads in 30 steps, got {r.bus.reads}"


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
    """An older config object must disarm cleanly, not crash."""
    m = safety.SafetyMonitor()
    safety.attach_hardware_watchdog(m, _FakeRobot(), object())
    assert m.hw_limits is None
