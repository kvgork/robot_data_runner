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
        ["--policy-path", "/tmp/m", "--port", "/dev/ttyACM1", "--execute"]
    )
    assert args.policy_path == "/tmp/m"
    assert args.port == "/dev/ttyACM1"
    assert args.execute is True

    name, dev, w, h = _parse_camera_spec("d435_rgb=/dev/video0,640,480")
    assert name == "d435_rgb" and dev == "/dev/video0" and w == 640 and h == 480
