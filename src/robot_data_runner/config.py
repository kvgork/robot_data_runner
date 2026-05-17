"""RunnerConfig — frozen dataclass for the runner CLI + lib API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunnerConfig:
    """All inputs the runner needs. Constructed from CLI args or in Python.

    Mirrors :class:`robot_data_recorder.config.RecordingConfig` style so the
    two packages feel like siblings.

    Attributes
    ----------
    policy_path:
        Absolute path to a ``pretrained_model/`` directory produced by
        ``lerobot-train`` (lerobot 0.5+ layout: ``model.safetensors`` +
        ``policy_preprocessor.json`` + ``policy_postprocessor.json`` +
        ``config.json``).
    port:
        DYNAMIXEL serial device (e.g. ``/dev/ttyACM0``).
    dataset_root:
        Optional path to the LeRobotDataset that the policy was trained on.
        Needed when the checkpoint config does not embed feature shapes —
        the loader reconstructs them from dataset metadata.
    cameras:
        Mapping ``name -> (device, width, height)``. The ``name`` MUST match
        the ``observation.images.<name>`` key in the policy's input schema.
    rate_hz:
        Control loop frequency.
    duration_s:
        Hard wall-clock cap.
    max_relative_target:
        Per-joint max delta per step (degrees if ``use_degrees`` else
        normalized units). Server-side clip via FeetechMotorsBus.
    use_degrees:
        Bus reads/writes in degrees. Match the units used during training.
    execute:
        If False (default), the loop reads obs and runs the policy but does
        NOT send any motor commands.
    home_on_exit:
        On SIGINT/SIGTERM (or duration cap), send zero-position before
        disconnect. Disable when zero pose collides with workspace.
    repeat_warn_steps:
        Stuck-action watchdog threshold.
    seed:
        Torch seed for non-deterministic policy layers.
    verbose:
        Log every predicted action.
    task:
        Natural-language task instruction passed to the policy each step.
        REQUIRED for VLA policies (SmolVLA / OpenVLA) — without it, the
        preprocessor cannot build ``observation.language.tokens`` and
        ``select_action`` crashes. Match the string used during training
        (look at ``meta/tasks.parquet`` of the source dataset; for
        kvgork/so101-pickplace1 it is ``"pick and place cube"``).
        Ignored by ACT / Diffusion policies.
    """

    policy_path: Path
    port: str = "/dev/ttyACM0"
    dataset_root: Path | None = None
    cameras: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    rate_hz: float = 30.0
    duration_s: float = 60.0
    max_relative_target: float = 5.0
    use_degrees: bool = False
    execute: bool = False
    home_on_exit: bool = False
    repeat_warn_steps: int = 30
    seed: int = 42
    verbose: bool = False
    task: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.policy_path, str):
            self.policy_path = Path(self.policy_path)
        if isinstance(self.dataset_root, str):
            self.dataset_root = Path(self.dataset_root)
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.max_relative_target <= 0:
            raise ValueError("max_relative_target must be positive")
