"""Tests for robot_data_runner.episode_runner — S5/S7 domain safety fixes.

No hardware. Robot + policy are fakes; ``torch`` is stubbed via
``sys.modules`` for the duration of a test since it is not installed in
this environment (the package already soft-imports it — see
``policy_loader.py`` / ``runner.py``).

Covers:
* S5 — non-interactive stdin + execute=True refuses to start, before any
  robot/policy setup; the --i-am-not-driving-an-arm opt-out cannot rescue
  execute=True (it only applies to an execute=False dry run).
* S7 — a mid-run EOFError (stdin closed) or a SIGINT delivered while
  blocked in a prompt aborts the session and records a distinct
  "aborted" outcome — never a fabricated "fail".
* _aggregate_records — aborted episodes are excluded from the
  success-rate denominator.
"""

from __future__ import annotations

import json
import os
import signal as signal_mod
import sys
import types

import pytest

from robot_data_runner import RunnerConfig
from robot_data_runner.episode_runner import (
    OUTCOME_ABORTED,
    OUTCOME_FAIL,
    OUTCOME_SUCCESS,
    EpisodeRecord,
    _aggregate_records,
    run_episodes,
)
from robot_data_runner.safety import InteractiveRequiredError
from robot_data_runner.task_specs import PromptUserObserverSpec

CANON = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeBus:
    motors = list(CANON)


class _FakeRobot:
    """Minimal SO101Follower stand-in. No calibration attr -> safety_limits
    falls back to its hardcoded floor (exercised elsewhere in
    test_safety_limits.py)."""

    def __init__(self, obs=None):
        self.bus = _FakeBus()
        self._obs = obs if obs is not None else _base_obs()
        self.connected = False
        self.sent_actions: list[dict] = []

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        return dict(self._obs)

    def send_action(self, action_dict):
        self.sent_actions.append(dict(action_dict))


def _base_obs():
    return {f"{m}.pos": 0.0 for m in CANON}


class _FakePolicy:
    def select_action(self, pol_in):  # pragma: no cover - not reached in these tests
        raise AssertionError("select_action should not be called — episodes end on timeout")


class _FakeLoaded:
    def __init__(self):
        self.policy = _FakePolicy()
        self.device = "cpu"
        self.ds_meta = None
        self.preprocessor = None
        self.postprocessor = None


def _install_fake_torch(monkeypatch):
    if "torch" in sys.modules:
        return

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_a):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _NoGrad()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _patch_common(monkeypatch, robot):
    """Wire episode_runner's soft-imported call sites to fakes."""
    import robot_data_runner.episode_runner as er

    monkeypatch.setattr(er, "load_policy", lambda *a, **k: _FakeLoaded())
    monkeypatch.setattr(er, "_build_robot", lambda cfg: robot)  # noqa: ARG005
    monkeypatch.setattr(
        er,
        "resolve_state_mapping",
        lambda policy, ds_meta, motor_names, **k: (list(motor_names), 6),  # noqa: ARG005
    )
    _install_fake_torch(monkeypatch)


def _cfg(tmp_path, **overrides):
    kwargs = dict(
        policy_path=tmp_path / "policy",
        execute=True,
        rate_hz=30.0,
        allow_fast=True,  # sidestep the unrelated S1 rate-ceiling gate
    )
    kwargs.update(overrides)
    return RunnerConfig(**kwargs)


# ---------------------------------------------------------------------------
# S5 — non-interactive refusal
# ---------------------------------------------------------------------------


def test_non_tty_execute_refuses_before_any_setup(monkeypatch, tmp_path):
    """Non-tty stdin + execute=True refuses BEFORE robot/policy setup.

    Deliberately does NOT patch load_policy/_build_robot: if the gate did
    not run first, this would blow up with an unrelated ImportError instead
    of the expected InteractiveRequiredError.
    """
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    cfg = _cfg(tmp_path, execute=True)
    spec = PromptUserObserverSpec(timeout_s=1.0)
    with pytest.raises(InteractiveRequiredError):
        run_episodes(cfg, task_spec=spec, n_episodes=1, output_json=tmp_path / "out.json")


def test_opt_out_does_not_rescue_execute_true(monkeypatch, tmp_path):
    """--i-am-not-driving-an-arm (allow_non_interactive) must NOT rescue execute=True."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    cfg = _cfg(tmp_path, execute=True)
    spec = PromptUserObserverSpec(timeout_s=1.0)
    with pytest.raises(InteractiveRequiredError):
        run_episodes(
            cfg,
            task_spec=spec,
            n_episodes=1,
            output_json=tmp_path / "out.json",
            i_am_not_driving_an_arm=True,
        )


def test_non_tty_dry_run_refuses_without_opt_out(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    cfg = _cfg(tmp_path, execute=False)
    spec = PromptUserObserverSpec(timeout_s=1.0)
    with pytest.raises(InteractiveRequiredError):
        run_episodes(cfg, task_spec=spec, n_episodes=1, output_json=tmp_path / "out.json")


def test_non_tty_dry_run_proceeds_with_opt_out(monkeypatch, tmp_path):
    """execute=False + the named opt-out proceeds past the gate (CI-smoke-testable)."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    robot = _FakeRobot()
    _patch_common(monkeypatch, robot)
    monkeypatch.setattr("builtins.input", lambda *_a: "")

    cfg = _cfg(tmp_path, execute=False)
    spec = PromptUserObserverSpec(timeout_s=0.0)  # ends immediately -> scored fail
    out = run_episodes(
        cfg,
        task_spec=spec,
        n_episodes=1,
        duration_per_episode_s=0.0,
        output_json=tmp_path / "out.json",
        i_am_not_driving_an_arm=True,
    )
    payload = json.loads(out.read_text())
    assert payload["n_episodes"] == 1
    assert payload["n_scored"] == 1
    assert payload["n_aborted"] == 0


# ---------------------------------------------------------------------------
# S7 — mid-run EOFError aborts, never fabricates a fail
# ---------------------------------------------------------------------------


def test_mid_run_eof_aborts_session_not_scored_as_fail(monkeypatch, tmp_path):
    """Episode 0 scores normally; episode 1's reset prompt hits a closed
    stdin (EOFError). The run must abort — episode 1 is "aborted", not
    "fail", and episode 2 is never attempted."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    robot = _FakeRobot()
    _patch_common(monkeypatch, robot)

    # call 1: ep0 reset prompt -> ""   call 2: ep0 verdict prompt -> "y"
    # call 3: ep1 reset prompt -> stdin closed (EOFError)
    responses = iter(["", "y"])

    def _fake_input(_msg=""):
        try:
            return next(responses)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", _fake_input)

    spec = PromptUserObserverSpec(timeout_s=0.0)
    out = run_episodes(
        _cfg(tmp_path, execute=True),
        task_spec=spec,
        n_episodes=3,
        duration_per_episode_s=0.0,
        output_json=tmp_path / "out.json",
    )
    payload = json.loads(out.read_text())
    per_ep = payload["_metadata"]["per_episode"]

    assert len(per_ep) == 2, "episode 2 must never be attempted after the abort"
    assert per_ep[0]["outcome"] == OUTCOME_SUCCESS
    assert per_ep[1]["outcome"] == OUTCOME_ABORTED
    assert per_ep[1]["success"] is False  # aborted is never truthy "success"

    assert payload["n_episodes"] == 2
    assert payload["n_scored"] == 1
    assert payload["n_aborted"] == 1
    assert payload["pc_success"] == 1.0  # only the scored episode counts


def test_mid_run_eof_during_verdict_prompt_aborts_not_fail(monkeypatch, tmp_path):
    """EOFError during the *verdict* prompt (not the reset prompt) must also
    abort rather than fabricate a fail — task_specs.py reuses the same
    KeyboardInterrupt mechanism as the typed "abort" command."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    robot = _FakeRobot()
    _patch_common(monkeypatch, robot)

    # call 1: ep0 reset prompt -> ""   call 2: ep0 verdict prompt -> EOFError
    responses = iter([""])

    def _fake_input(_msg=""):
        try:
            return next(responses)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", _fake_input)

    spec = PromptUserObserverSpec(timeout_s=0.0)
    out = run_episodes(
        _cfg(tmp_path, execute=True),
        task_spec=spec,
        n_episodes=2,
        duration_per_episode_s=0.0,
        output_json=tmp_path / "out.json",
    )
    payload = json.loads(out.read_text())
    per_ep = payload["_metadata"]["per_episode"]

    assert len(per_ep) == 1
    assert per_ep[0]["outcome"] == OUTCOME_ABORTED
    assert payload["n_scored"] == 0
    assert payload["n_aborted"] == 1
    assert payload["pc_success"] == 0.0  # no scored episodes; NOT counted as a fail


def test_sigint_during_reset_prompt_aborts_not_scored(monkeypatch, tmp_path):
    """A real SIGINT delivered while blocked in the reset prompt's input()
    must abort the episode, not silently proceed to run + score it (S7).
    Uses a REAL signal (not a mock) to exercise the actual
    SafetyMonitor.install_signal_handlers() interaction — the whole point
    of the bug is that a blocked input() only returns AFTER the signal, at
    which point stop_flag is already True."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    robot = _FakeRobot()
    _patch_common(monkeypatch, robot)

    def _fake_input(_msg=""):
        # Simulate "operator hits Ctrl-C, then eventually presses ENTER" —
        # the signal handler runs synchronously here (single-threaded);
        # by the time input() "returns", stop_flag is already True.
        os.kill(os.getpid(), signal_mod.SIGINT)
        return ""

    monkeypatch.setattr("builtins.input", _fake_input)

    spec = PromptUserObserverSpec(timeout_s=0.0)
    out = run_episodes(
        _cfg(tmp_path, execute=True),
        task_spec=spec,
        n_episodes=2,
        duration_per_episode_s=0.0,
        output_json=tmp_path / "out.json",
    )
    payload = json.loads(out.read_text())
    per_ep = payload["_metadata"]["per_episode"]

    assert len(per_ep) == 1
    assert per_ep[0]["outcome"] == OUTCOME_ABORTED
    assert payload["n_scored"] == 0
    assert payload["n_aborted"] == 1


# ---------------------------------------------------------------------------
# _aggregate_records — ABORTED excluded from the denominator
# ---------------------------------------------------------------------------


def test_aggregate_excludes_aborted_from_denominator():
    """1 success / 1 fail / 2 aborted -> pc_success = 0.5, NOT 0.25."""
    records = [
        EpisodeRecord(index=0, outcome=OUTCOME_SUCCESS, ep_len_s=5.0, intervention=False),
        EpisodeRecord(index=1, outcome=OUTCOME_FAIL, ep_len_s=5.0, intervention=False),
        EpisodeRecord(index=2, outcome=OUTCOME_ABORTED, ep_len_s=1.0, intervention=False),
        EpisodeRecord(index=3, outcome=OUTCOME_ABORTED, ep_len_s=0.0, intervention=False),
    ]
    agg = _aggregate_records(records)
    assert agg["n_scored"] == 2
    assert agg["n_aborted"] == 2
    assert agg["pc_success"] == 0.5
    assert agg["mean_ep_len"] == 5.0  # only the two scored episodes' 5.0s each


def test_aggregate_all_aborted_is_zero_not_nan():
    records = [
        EpisodeRecord(index=0, outcome=OUTCOME_ABORTED, ep_len_s=0.0, intervention=False),
    ]
    agg = _aggregate_records(records)
    assert agg["n_scored"] == 0
    assert agg["n_aborted"] == 1
    assert agg["pc_success"] == 0.0
    assert agg["mean_ep_len"] == 0.0


def test_aggregate_empty_records():
    agg = _aggregate_records([])
    assert agg["n_scored"] == 0
    assert agg["n_aborted"] == 0
    assert agg["pc_success"] == 0.0


def test_episode_record_success_property():
    assert EpisodeRecord(0, OUTCOME_SUCCESS, 1.0, False).success is True
    assert EpisodeRecord(0, OUTCOME_FAIL, 1.0, False).success is False
    assert EpisodeRecord(0, OUTCOME_ABORTED, 1.0, False).success is False
