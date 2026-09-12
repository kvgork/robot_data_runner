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


def test_mapper_state_uses_motor_order_not_sorted():
    """State MUST be built in the given motor_order, never alphabetical sorted().

    Regression: sorted() alphabetises the SO101 joints to
    [elbow_flex, gripper, shoulder_lift, shoulder_pan, wrist_flex, wrist_roll],
    which permutes the state vs the training layout (servo id 1..6) and yields
    wrong/unsafe actions on the real arm.
    """
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("torch + numpy required")

    from robot_data_runner.mappers import obs_to_policy_input

    canonical = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    # Distinct value per joint == its canonical index, so a permutation shows up.
    obs = {f"{m}.pos": float(i) for i, m in enumerate(canonical)}
    obs["overhead"] = __import__("numpy").zeros((480, 640, 3), dtype="uint8")

    out = obs_to_policy_input(obs, "cpu", motor_order=canonical)
    state = out["observation.state"].squeeze(0).tolist()
    assert state == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], state  # canonical order preserved

    # sorted() path would scramble to [elbow_flex=2, gripper=5, shoulder_lift=1, ...]
    out_sorted = obs_to_policy_input(obs, "cpu", motor_order=None)
    assert out_sorted["observation.state"].squeeze(0).tolist() == [2.0, 5.0, 1.0, 0.0, 3.0, 4.0]


def test_mapper_pads_state_to_expected_dim_with_zeros():
    """state_dim > n_joints right-pads with zeros (the training zero-velocity block)."""
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("torch + numpy required")

    from robot_data_runner.mappers import obs_to_policy_input

    canonical = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    obs = {f"{m}.pos": float(i + 1) for i, m in enumerate(canonical)}

    out = obs_to_policy_input(obs, "cpu", motor_order=canonical, state_dim=12)
    state = out["observation.state"].squeeze(0).tolist()
    assert len(state) == 12
    assert state[:6] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert state[6:] == [0.0] * 6  # zero-velocity pad

    # Refuse to truncate (would silently drop joints).
    import pytest
    with pytest.raises(ValueError):
        obs_to_policy_input(obs, "cpu", motor_order=canonical, state_dim=4)

    # Missing a .pos key for the requested order → KeyError.
    del obs["gripper.pos"]
    with pytest.raises(KeyError):
        obs_to_policy_input(obs, "cpu", motor_order=canonical, state_dim=12)


class _FakeFeat:
    def __init__(self, shape):
        self.shape = shape


class _FakeCfg:
    def __init__(self, state_dim):
        self.input_features = {"observation.state": _FakeFeat((state_dim,))}


class _FakePolicy:
    def __init__(self, state_dim):
        self.config = _FakeCfg(state_dim)


class _FakeMeta:
    def __init__(self, std):
        self.stats = {"observation.state": {"std": std}}


CANON = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def test_resolve_state_mapping_ok_zero_pad():
    """12-dim policy + zero-std velocity block → (canonical, 12)."""
    from robot_data_runner.mappers import resolve_state_mapping

    meta = _FakeMeta([1.0] * 6 + [0.0] * 6)  # pos has std, vel block is 0
    order, dim = resolve_state_mapping(_FakePolicy(12), meta, CANON, require_execute=True)
    assert order == CANON
    assert dim == 12


def test_resolve_state_mapping_refuses_nonzero_pad():
    """Extra dims with real (non-zero) training std must NOT be zero-padded."""
    import pytest

    from robot_data_runner.mappers import resolve_state_mapping

    meta = _FakeMeta([1.0] * 6 + [0.5] * 6)  # pad region carries signal
    with pytest.raises(ValueError, match="not ~0|real signal"):
        resolve_state_mapping(_FakePolicy(12), meta, CANON, require_execute=True)


def test_resolve_state_mapping_refuses_permuted_order():
    """A non-canonical motor order is rejected (would permute state)."""
    import pytest

    from robot_data_runner.mappers import resolve_state_mapping

    permuted = sorted(CANON)  # alphabetical ≠ canonical
    meta = _FakeMeta([1.0] * 6 + [0.0] * 6)
    with pytest.raises(ValueError, match="canonical"):
        resolve_state_mapping(_FakePolicy(12), meta, permuted)


def test_resolve_state_mapping_refuses_execute_when_dim_unknown():
    """require_execute + indeterminable state dim → refuse (no blind send)."""
    import pytest

    from robot_data_runner.mappers import resolve_state_mapping

    class _NoFeat:
        config = None

    with pytest.raises(ValueError, match="refusing --execute"):
        resolve_state_mapping(_NoFeat(), _FakeMeta(None), CANON, require_execute=True)


def test_resolve_state_mapping_refuses_pad_without_stats():
    """12-dim policy but no training stats → refuse to pad blind."""
    import pytest

    from robot_data_runner.mappers import resolve_state_mapping

    with pytest.raises(ValueError, match="no training stats|pad blind"):
        resolve_state_mapping(_FakePolicy(12), _FakeMeta(None), CANON, require_execute=True)
