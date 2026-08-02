"""Tests for robot_data_runner.safety_limits.

No hardware, no lerobot needed — everything is exercised against plain
fakes/stubs. Covers the S1 (unit legibility) + S3 (absolute joint-limit
clamp) + S4 (ramped home) safety fixes.
"""

from __future__ import annotations

import logging

import pytest

from robot_data_runner.safety_limits import (
    DEFAULT_JOINT_LIMITS_DEG,
    RateCeilingExceeded,
    check_rate_ceiling,
    clamp_action,
    describe_clamp,
    per_joint_rates_deg,
    ramped_home,
    resolve_joint_limits,
)


class _FakeCal:
    def __init__(self, range_min, range_max):
        self.range_min = range_min
        self.range_max = range_max


class _FakeRobot:
    def __init__(self, calibration=None):
        self.calibration = calibration


# ---------------------------------------------------------------------------
# resolve_joint_limits — floor fallback + intersection
# ---------------------------------------------------------------------------


def test_resolve_joint_limits_floor_when_robot_none():
    lo, hi = resolve_joint_limits(None)
    for name, (flo, fhi) in DEFAULT_JOINT_LIMITS_DEG.items():
        assert lo[name] == flo
        assert hi[name] == fhi


def test_resolve_joint_limits_floor_when_calibration_missing():
    lo, hi = resolve_joint_limits(_FakeRobot(calibration=None))
    assert lo["elbow_flex"] == -10.0
    assert hi["elbow_flex"] == 90.0
    assert lo["shoulder_pan"] == -90.0
    assert hi["shoulder_pan"] == 90.0


def test_resolve_joint_limits_floor_when_calibration_malformed():
    """A calibration entry with non-numeric range fields never raises."""
    cal = {"elbow_flex": _FakeCal("not-a-number", "also-not")}
    lo, hi = resolve_joint_limits(_FakeRobot(calibration=cal))
    assert lo["elbow_flex"] == -10.0
    assert hi["elbow_flex"] == 90.0


def test_resolve_joint_limits_floor_when_calibration_entry_missing_fields():
    class _NoRangeCal:
        pass

    cal = {"elbow_flex": _NoRangeCal()}
    lo, hi = resolve_joint_limits(_FakeRobot(calibration=cal))
    assert lo["elbow_flex"] == -10.0
    assert hi["elbow_flex"] == 90.0


def test_resolve_joint_limits_symmetric_180_intersected_to_floor():
    """A symmetric +/-180 calibration (ticks 0..4096) never widens past the floor."""
    cal = {name: _FakeCal(0, 4096) for name in DEFAULT_JOINT_LIMITS_DEG}
    lo, hi = resolve_joint_limits(_FakeRobot(calibration=cal))
    for name, (flo, fhi) in DEFAULT_JOINT_LIMITS_DEG.items():
        assert lo[name] == pytest.approx(flo), name
        assert hi[name] == pytest.approx(fhi), name
    # Explicitly re-assert the table-avoidance floor survives the wide calibration.
    assert lo["elbow_flex"] == -10.0
    assert hi["elbow_flex"] == 90.0


def test_resolve_joint_limits_tighter_calibration_is_respected():
    """A calibration NARROWER than the floor should tighten, not widen."""
    # ticks 1024..3072 -> deg = ticks/4096*360-180 -> [-90, 0] for shoulder_pan
    cal = {"shoulder_pan": _FakeCal(1024, 2048)}
    lo, hi = resolve_joint_limits(_FakeRobot(calibration=cal))
    assert lo["shoulder_pan"] == pytest.approx(-90.0)
    assert hi["shoulder_pan"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# clamp_action
# ---------------------------------------------------------------------------


def test_clamp_action_clamps_out_of_range_leaves_in_range_reports_joints():
    lo = {"elbow_flex": -10.0, "shoulder_pan": -90.0}
    hi = {"elbow_flex": 90.0, "shoulder_pan": 90.0}
    action = {
        "elbow_flex.pos": -50.0,  # below lo -> clamp to -10
        "shoulder_pan.pos": 45.0,  # in range -> untouched
    }
    clamped, joints = clamp_action(action, lo, hi)
    assert clamped["elbow_flex.pos"] == -10.0
    assert clamped["shoulder_pan.pos"] == 45.0
    assert joints == ["elbow_flex"]


def test_clamp_action_clamps_above_hi():
    lo = {"shoulder_pan": -90.0}
    hi = {"shoulder_pan": 90.0}
    clamped, joints = clamp_action({"shoulder_pan.pos": 200.0}, lo, hi)
    assert clamped["shoulder_pan.pos"] == 90.0
    assert joints == ["shoulder_pan"]


def test_clamp_action_passes_unknown_keys_through():
    lo = {"shoulder_pan": -90.0}
    hi = {"shoulder_pan": 90.0}
    action = {
        "shoulder_pan.pos": 200.0,
        "unknown_joint.pos": 999.0,  # not in lo/hi
        "some_other_key": "value",  # doesn't end in .pos
    }
    clamped, joints = clamp_action(action, lo, hi)
    assert clamped["unknown_joint.pos"] == 999.0
    assert clamped["some_other_key"] == "value"
    assert joints == ["shoulder_pan"]


def test_clamp_action_no_duplicate_joints_in_report():
    lo = {"shoulder_pan": -90.0}
    hi = {"shoulder_pan": 90.0}
    # Same joint appearing twice (shouldn't happen in practice, but the
    # dedup guarantee must hold regardless).
    clamped, joints = clamp_action({"shoulder_pan.pos": 500.0}, lo, hi)
    assert joints.count("shoulder_pan") == 1


# ---------------------------------------------------------------------------
# describe_clamp / per_joint_rates_deg
# ---------------------------------------------------------------------------


def test_describe_clamp_differs_degrees_vs_normalized():
    text_deg = describe_clamp(5.0, True, robot=None, rate_hz=30.0)
    text_norm = describe_clamp(5.0, False, robot=None, rate_hz=30.0)
    assert text_deg != text_norm
    assert "DEGREES" in text_deg
    assert "NORMALIZED" in text_norm


def test_describe_clamp_degrees_mode_is_uniform_mrt_for_body_joints():
    rates = per_joint_rates_deg(5.0, True, robot=None, rate_hz=30.0)
    for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        step, rate = rates[name]
        assert step == pytest.approx(5.0)
        assert rate == pytest.approx(150.0)


def test_describe_clamp_normalized_wrist_roll_reproduces_9deg_step():
    """A 359.9-deg-span wrist_roll calibration at mrt=5.0 -> ~9.00 deg/step.

    Matches the domain safety review's measured figure (wrist_roll
    9.00 deg/step = 270 deg/s at 30 Hz on a near-full-turn calibration).
    """
    cal = {"wrist_roll": _FakeCal(0, 4095)}  # span = 4095/4096*360 ~= 359.91 deg
    robot = _FakeRobot(calibration=cal)
    rates = per_joint_rates_deg(5.0, False, robot, rate_hz=30.0)
    step, rate = rates["wrist_roll"]
    assert step == pytest.approx(9.0, abs=0.02)
    assert rate == pytest.approx(270.0, abs=1.0)


# ---------------------------------------------------------------------------
# check_rate_ceiling
# ---------------------------------------------------------------------------


def test_check_rate_ceiling_raises_by_default():
    # 5.0 deg/step * 30 Hz = 150 deg/s > 90 deg/s ceiling.
    with pytest.raises(RateCeilingExceeded):
        check_rate_ceiling(5.0, True, robot=None, rate_hz=30.0, ceiling=90.0, allow_fast=False)


def test_check_rate_ceiling_allow_fast_overrides():
    joint, rate = check_rate_ceiling(
        5.0, True, robot=None, rate_hz=30.0, ceiling=90.0, allow_fast=True
    )
    assert rate == pytest.approx(150.0)
    assert joint in DEFAULT_JOINT_LIMITS_DEG


def test_check_rate_ceiling_passes_when_under_ceiling():
    joint, rate = check_rate_ceiling(1.0, True, robot=None, rate_hz=30.0, ceiling=90.0)
    assert rate == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# ramped_home
# ---------------------------------------------------------------------------


class _FakeHomeRobot:
    """Instant-dynamics fake: send_action immediately updates position."""

    def __init__(self, start_pos: dict[str, float]):
        self._pos = dict(start_pos)
        self.send_calls: list[dict[str, float]] = []

    def get_observation(self) -> dict[str, float]:
        return {f"{k}.pos": v for k, v in self._pos.items()}

    def send_action(self, action: dict[str, float]) -> None:
        self.send_calls.append(dict(action))
        for key, value in action.items():
            name = key[: -len(".pos")]
            self._pos[name] = value


def test_ramped_home_reaches_zero_in_expected_steps():
    robot = _FakeHomeRobot({"shoulder_pan": 10.0})
    ramped_home(robot, ["shoulder_pan"], max_step_deg=3.0, rate_hz=200.0, settle_timeout_s=5.0)
    assert robot._pos["shoulder_pan"] == pytest.approx(0.0, abs=1e-6)
    # ceil(10 / 3) = 4 sends to land exactly on 0.
    assert len(robot.send_calls) == 4


def test_ramped_home_never_exceeds_max_step_deg_per_tick():
    robot = _FakeHomeRobot({"shoulder_pan": 10.0})
    ramped_home(robot, ["shoulder_pan"], max_step_deg=3.0, rate_hz=200.0, settle_timeout_s=5.0)
    prev = 10.0
    for call in robot.send_calls:
        new = call["shoulder_pan.pos"]
        assert abs(new - prev) <= 3.0 + 1e-9
        prev = new


def test_ramped_home_multi_joint_converges():
    robot = _FakeHomeRobot({"shoulder_pan": 5.0, "elbow_flex": -7.0})
    ramped_home(
        robot, ["shoulder_pan", "elbow_flex"], max_step_deg=2.0, rate_hz=200.0, settle_timeout_s=5.0
    )
    assert robot._pos["shoulder_pan"] == pytest.approx(0.0, abs=1e-6)
    assert robot._pos["elbow_flex"] == pytest.approx(0.0, abs=1e-6)


def test_ramped_home_skips_missing_joint_and_warns(caplog):
    class _PartialRobot(_FakeHomeRobot):
        def get_observation(self) -> dict[str, float]:
            # elbow_flex never appears in the observation.
            return {f"{k}.pos": v for k, v in self._pos.items() if k != "elbow_flex"}

    robot = _PartialRobot({"shoulder_pan": 6.0, "elbow_flex": 6.0})
    with caplog.at_level(logging.WARNING):
        ramped_home(
            robot,
            ["shoulder_pan", "elbow_flex"],
            max_step_deg=3.0,
            rate_hz=200.0,
            settle_timeout_s=2.0,
        )
    assert robot._pos["shoulder_pan"] == pytest.approx(0.0, abs=1e-6)
    assert robot._pos["elbow_flex"] == 6.0  # untouched — never resolvable
    assert any("elbow_flex" in rec.message for rec in caplog.records)


def test_ramped_home_settle_timeout_does_not_raise():
    """A robot that never reaches home (huge start, tiny timeout) times out cleanly."""
    robot = _FakeHomeRobot({"shoulder_pan": 1000.0})
    # Should return (not raise) once settle_timeout_s elapses.
    ramped_home(robot, ["shoulder_pan"], max_step_deg=0.001, rate_hz=1000.0, settle_timeout_s=0.05)
