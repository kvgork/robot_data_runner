"""RunnerConfig — frozen dataclass for the runner CLI + lib API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


def quiet_hf_hub() -> None:
    """Silence per-load huggingface_hub network revalidation + request logs.

    Deploy/eval load a checkpoint whose backbone (e.g. SmolVLM2) is already in
    the local HF cache, but huggingface_hub still HEAD/GETs the hub on every
    load — flooding the log and adding latency. Force offline (env must be set
    BEFORE huggingface_hub imports, so call this at the top of each entrypoint
    main() before any lerobot/transformers import) and drop the httpx logger to
    WARNING. Override by exporting ``HF_HUB_OFFLINE=0``.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for name in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


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
        normalized units). Server-side clip via FeetechMotorsBus. See
        :func:`robot_data_runner.safety_limits.describe_clamp` for the
        resolved per-joint deg/step + deg/s this implies — every run path
        logs that table at startup so the unit mode is never silently
        ambiguous.
    use_degrees:
        Bus reads/writes in degrees for the 5 arm joints (the gripper is
        always ``MotorNormMode.RANGE_0_100`` regardless of this flag — see
        lerobot's ``SOFollower.__init__``). Defaults to ``True``: degrees
        make ``max_relative_target`` legible as an actual angular rate
        (matches lerobot 0.6.0's own ``config_so_follower.py`` default),
        whereas the legacy normalized mode (-100..100 over the *full*
        calibrated range) silently turns a small-looking number like ``5.0``
        into a very large real angular rate on joints with a wide calibrated
        span (measured: wrist_roll 270 deg/s at mrt=5.0, 30 Hz). Set to
        ``False`` (CLI: ``--as-normalized``) only for compatibility with
        checkpoints/recordings captured in the old normalized space.
    max_deg_per_s_ceiling:
        Safety refusal threshold. If the resolved worst-joint deg/s implied
        by ``max_relative_target`` (see ``describe_clamp``) exceeds this and
        ``execute`` is requested, the run refuses to start rather than
        silently inheriting a dangerous rate under a unit-mode change or
        misconfiguration. Override with ``allow_fast=True`` (CLI:
        ``--allow-fast``).
    allow_fast:
        Escape hatch for ``max_deg_per_s_ceiling``. Off by default — the
        ceiling should be raised deliberately, not routinely bypassed.
    execute:
        If False (default), the loop reads obs and runs the policy but does
        NOT send any motor commands.
    home_on_exit:
        On SIGINT/SIGTERM (or duration cap), ramp to zero position (see
        :func:`robot_data_runner.safety_limits.ramped_home`) before
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
    # Degrees by default (S1 fix, 2026-08): legible units + matches lerobot
    # 0.6.0's own config_so_follower.py default. Pass use_degrees=False (CLI:
    # --as-normalized) only for the legacy normalized-percent behaviour.
    use_degrees: bool = True
    max_deg_per_s_ceiling: float = 90.0
    allow_fast: bool = False
    execute: bool = False
    home_on_exit: bool = False
    repeat_warn_steps: int = 30
    # --- joint-current / thermal watchdog (2026-09-10) ---
    # The SO-101 has no force sensing, so servo current is the only in-band signal
    # that the arm is straining. ON by default: trip points are derived from each
    # servo's own protection registers, so it fires strictly BEFORE the servo's own
    # protection drops torque mid-episode. Disable with --no-hardware-watchdog.
    hardware_watchdog: bool = True
    hw_every_n_steps: int = 15
    hw_breach_steps: int = 3
    seed: int = 42
    verbose: bool = False
    task: str | None = None
    # Fixed calibration id for the SO-101 follower. Determines the calibration
    # filename (<HF_LEROBOT_CALIBRATION>/robots/so_follower/<robot_id>.json) so
    # calibration persists and is reused instead of an anonymous `None.json`.
    # None → lerobot default (anonymous) behaviour.
    robot_id: str | None = None
    # If set, run_policy appends each step's action dict to this JSONL file
    # (one {"step", "t", "action"} record per line) — used to capture a
    # dry-run trajectory for later open-loop replay.
    actions_out: Path | None = None
    # If set, run_replay plays this JSONL trajectory back on the arm instead of
    # running the policy (open-loop). Motor writes still gated by `execute`.
    replay_actions: Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.policy_path, str):
            self.policy_path = Path(self.policy_path)
        if isinstance(self.dataset_root, str):
            self.dataset_root = Path(self.dataset_root)
        if isinstance(self.actions_out, str):
            self.actions_out = Path(self.actions_out)
        if isinstance(self.replay_actions, str):
            self.replay_actions = Path(self.replay_actions)
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.max_relative_target <= 0:
            raise ValueError("max_relative_target must be positive")
        if self.max_deg_per_s_ceiling <= 0:
            raise ValueError("max_deg_per_s_ceiling must be positive")
