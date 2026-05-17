"""Smoke: package importable without lerobot/torch installed."""

from __future__ import annotations


def test_top_level_import_ok():
    """Package + sub-modules import without heavy deps."""
    import robot_data_runner

    assert hasattr(robot_data_runner, "RunnerConfig")
    assert hasattr(robot_data_runner, "SafetyMonitor")
    assert hasattr(robot_data_runner, "load_policy")
    assert hasattr(robot_data_runner, "run_policy")


def test_config_dataclass():
    from pathlib import Path

    from robot_data_runner import RunnerConfig

    cfg = RunnerConfig(policy_path=Path("/tmp/x"))
    assert cfg.execute is False  # dry-run default
    assert cfg.rate_hz == 30.0
    assert cfg.max_relative_target == 5.0
    assert isinstance(cfg.cameras, dict) and len(cfg.cameras) == 0


def test_safety_monitor_repeat():
    from robot_data_runner import SafetyMonitor

    sm = SafetyMonitor(repeat_warn_steps=3)
    a = {"j.pos": 1.0}
    b = {"j.pos": 1.0 + 1e-5}
    for _ in range(5):
        sm.observe(a)
    # Streak should have warned at step 3 (1 less, since first observe sets baseline).
    assert sm._streak >= 3
    sm.observe(b)
    # Within epsilon → still treated as same.
    assert sm._streak >= 4
    sm.observe({"j.pos": 5.0})
    assert sm._streak == 0


def test_cli_parses_args():
    """CLI parser builds without instantiating downstream deps."""
    from robot_data_runner.cli import _build_parser, _parse_camera_spec

    parser = _build_parser()
    args = parser.parse_args(
        ["--policy-path", "/tmp/m", "--port", "/dev/ttyACM1", "--execute",
         "--task", "pick and place cube"]
    )
    assert args.policy_path == "/tmp/m"
    assert args.port == "/dev/ttyACM1"
    assert args.execute is True
    assert args.task == "pick and place cube"

    name, dev, w, h = _parse_camera_spec("d435_rgb=/dev/video0,640,480")
    assert name == "d435_rgb" and dev == "/dev/video0" and w == 640 and h == 480


def test_config_task_defaults_none():
    """RunnerConfig.task defaults to None; can be set explicitly."""
    from pathlib import Path

    from robot_data_runner import RunnerConfig

    cfg = RunnerConfig(policy_path=Path("/tmp/x"))
    assert cfg.task is None

    cfg2 = RunnerConfig(policy_path=Path("/tmp/x"), task="pick and place cube")
    assert cfg2.task == "pick and place cube"


def test_mapper_injects_task_when_provided():
    """obs_to_policy_input adds 'task' key only when task arg is non-None."""
    try:
        import numpy as np
        import torch  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("torch + numpy required")

    from robot_data_runner.mappers import obs_to_policy_input

    obs = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 1.0,
        "elbow_flex.pos": 2.0,
        "wrist_flex.pos": 3.0,
        "wrist_roll.pos": 4.0,
        "gripper.pos": 5.0,
        "d435_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    # task=None → no 'task' key
    out_no_task = obs_to_policy_input(obs, "cpu", task=None)
    assert "task" not in out_no_task
    assert "observation.state" in out_no_task
    assert "observation.images.d435_rgb" in out_no_task

    # task="pick and place cube" → 'task' key present
    out_with_task = obs_to_policy_input(obs, "cpu", task="pick and place cube")
    assert out_with_task["task"] == "pick and place cube"
