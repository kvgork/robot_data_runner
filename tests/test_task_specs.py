"""Smoke tests for the task_specs + episode aggregation logic.

No hardware, no lerobot, no torch required.
"""

from __future__ import annotations

import io
from unittest.mock import patch


def test_prompt_user_observer_timeout():
    from robot_data_runner.task_specs import PromptUserObserverSpec

    spec = PromptUserObserverSpec(timeout_s=2.0)
    assert spec.is_done(1.0, {}) == (False, False)
    assert spec.is_done(2.5, {}) == (True, False)


def test_prompt_user_observer_score_y(monkeypatch):
    from robot_data_runner.task_specs import PromptUserObserverSpec

    spec = PromptUserObserverSpec()
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert spec.on_episode_end(0, False, 5.0) is True


def test_prompt_user_observer_score_n(monkeypatch):
    from robot_data_runner.task_specs import PromptUserObserverSpec

    spec = PromptUserObserverSpec()
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert spec.on_episode_end(0, True, 5.0) is False


def test_gripper_at_target_pose_match():
    from robot_data_runner.task_specs import GripperAtTargetPoseSpec

    target = [0.0, 0.0, 0.0, 0.0, 0.0, 60.0]
    spec = GripperAtTargetPoseSpec(
        target_joint_pos=target,
        tolerance_per_joint=2.0,
        require_gripper_closed=True,
        gripper_closed_value=50.0,
    )
    obs_ok = {
        "shoulder_pan.pos": 0.5,
        "shoulder_lift.pos": -0.3,
        "elbow_flex.pos": 1.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 60.0,
    }
    assert spec.is_done(1.0, obs_ok) == (True, True)

    obs_far = dict(obs_ok)
    obs_far["shoulder_lift.pos"] = 30.0
    assert spec.is_done(1.0, obs_far) == (False, False)

    obs_open = dict(obs_ok)
    obs_open["gripper.pos"] = 10.0
    assert spec.is_done(1.0, obs_open) == (False, False)


def test_make_task_spec_factory():
    from robot_data_runner.task_specs import (
        GripperAtTargetPoseSpec,
        PromptUserObserverSpec,
        make_task_spec,
    )

    assert isinstance(make_task_spec("prompt_user_observer"), PromptUserObserverSpec)
    spec = make_task_spec(
        "gripper_at_target_pose",
        target_joint_pos=[0, 0, 0, 0, 0, 0],
    )
    assert isinstance(spec, GripperAtTargetPoseSpec)

    try:
        make_task_spec("nope")
    except KeyError:
        pass
    else:
        assert False, "expected KeyError for unknown spec"


def test_cli_eval_parser_builds():
    from robot_data_runner.cli_eval import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "--policy-path", "/tmp/p",
        "--n-episodes", "5",
        "--task-spec", "prompt_user_observer",
        "--i-have-read-the-safety-runbook",
    ])
    assert args.n_episodes == 5
    assert args.task_spec == "prompt_user_observer"
    assert args.i_have_read_the_safety_runbook is True
