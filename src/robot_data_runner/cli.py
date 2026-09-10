"""robot-data-run CLI — thin wrapper around :func:`run_policy`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import RunnerConfig, quiet_hf_hub
from .runner import run_policy

logger = logging.getLogger(__name__)


def _parse_camera_spec(spec: str) -> tuple[str, str, int, int]:
    """Parse ``name=device,W,H``. Returns (name, device, w, h)."""
    if "=" not in spec:
        raise ValueError(f"--camera spec must be `name=device,W,H`: {spec!r}")
    name, rhs = spec.split("=", 1)
    bits = [b.strip() for b in rhs.split(",")]
    if len(bits) < 3:
        raise ValueError(f"--camera spec needs device,W,H: {spec!r}")
    try:
        return name.strip(), bits[0], int(bits[1]), int(bits[2])
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"camera W,H must be ints: {spec!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robot-data-run",
        description=(
            "Run a trained LeRobot policy on the real SO-101 follower arm. "
            "Default is DRY-RUN — pass --execute to enable motor writes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--policy-path", required=True, help="pretrained_model/ dir")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--dataset-root", default=None)
    p.add_argument(
        "--camera",
        action="append",
        default=[],
        help="`name=device,W,H` (repeatable). e.g. d435_rgb=/dev/video0,640,480",
    )
    p.add_argument("--rate-hz", type=float, default=30.0)
    p.add_argument("--duration-s", type=float, default=60.0)
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help=(
            "Per-joint max delta per control step. UNITS: DEGREES by default "
            "(e.g. 5.0 = 5 deg/step for the 5 arm joints; the gripper is "
            "always a percent of jaw travel regardless of this flag). Pass "
            "--as-normalized to revert to lerobot's legacy normalized "
            "(-100..100 over the FULL calibrated range) unit space instead — "
            "in that space the same number can imply a much faster real "
            "angular rate on wide-span joints (e.g. wrist_roll). The "
            "resolved per-joint deg/step + deg/s is always logged at startup "
            "(see describe_clamp); a run that would exceed "
            "--max-deg-per-s-ceiling refuses to start unless --allow-fast is "
            "passed."
        ),
    )
    p.add_argument(
        "--as-normalized",
        action="store_true",
        help=(
            "Use lerobot's legacy normalized (-100..100 for the arm joints; "
            "the gripper is always 0..100) unit space instead of degrees. "
            "Only for compatibility with checkpoints/recordings captured "
            "in the old normalized space — degrees (the default) are "
            "recommended for any new run."
        ),
    )
    p.add_argument(
        "--allow-fast",
        action="store_true",
        help=(
            "Override the max-deg-per-s-ceiling startup safety refusal. "
            "Only set this if you have deliberately reviewed the "
            "describe_clamp banner and accept the resolved per-joint rate."
        ),
    )
    p.add_argument(
        "--max-deg-per-s-ceiling",
        type=float,
        default=90.0,
        help="Refuse --execute if the resolved worst-joint rate exceeds this (deg/s).",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="enable real motor writes (default = dry-run)",
    )
    p.add_argument("--home-on-exit", action="store_true")
    p.add_argument(
        "--id",
        dest="robot_id",
        default=None,
        help=(
            "Fixed SO-101 follower calibration id. Calibration is saved to / "
            "loaded from <HF_LEROBOT_CALIBRATION>/robots/so_follower/<id>.json, "
            "so it persists across runs instead of an anonymous None.json."
        ),
    )
    p.add_argument(
        "--actions-out",
        dest="actions_out",
        default=None,
        help=(
            "Record each step's action to this JSONL file (one record per line) "
            "for later open-loop replay with `robot-data-run-replay`. Works in "
            "dry-run too — capture a motors-off trajectory, then replay it."
        ),
    )
    p.add_argument("--repeat-warn-steps", type=int, default=30)
    p.add_argument(
        "--no-hardware-watchdog",
        dest="hardware_watchdog",
        action="store_false",
        default=True,
        help=(
            "Disable the joint-current / thermal watchdog. It is ON by default: the "
            "SO-101 has no force sensing, so servo current is the only in-band signal "
            "that the arm is straining against a limit, the table, or a jammed object. "
            "Trip points are read from each servo's own protection registers, so it "
            "fires before the servo's protection drops torque mid-episode."
        ),
    )
    p.add_argument(
        "--hw-every-n-steps",
        type=int,
        default=5,
        help="Sample joint current/temperature every N control steps (default 5 ~= 6 Hz at 30 Hz).",
    )
    p.add_argument(
        "--hw-breach-steps",
        type=int,
        default=3,
        help="Consecutive breached samples before the watchdog trips (default 3).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--task",
        default=None,
        help=(
            "Natural-language task instruction passed to the policy each step. "
            "Required for VLA policies (SmolVLA, OpenVLA). Must match the "
            "training task string — for kvgork/so101-pickplace1 the training "
            "string is 'pick and place cube'."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    quiet_hf_hub()  # before any lerobot/hf_hub import — stops cache revalidation spam
    args = _build_parser().parse_args(argv)
    cameras: dict[str, tuple[str, int, int]] = {}
    for spec in args.camera:
        name, device, w, h = _parse_camera_spec(spec)
        cameras[name] = (device, w, h)
    cfg = RunnerConfig(
        policy_path=Path(args.policy_path),
        port=args.port,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        cameras=cameras,
        rate_hz=args.rate_hz,
        duration_s=args.duration_s,
        max_relative_target=args.max_relative_target,
        use_degrees=not args.as_normalized,
        max_deg_per_s_ceiling=args.max_deg_per_s_ceiling,
        allow_fast=args.allow_fast,
        execute=args.execute,
        home_on_exit=args.home_on_exit,
        repeat_warn_steps=args.repeat_warn_steps,
        hardware_watchdog=args.hardware_watchdog,
        hw_every_n_steps=args.hw_every_n_steps,
        hw_breach_steps=args.hw_breach_steps,
        seed=args.seed,
        verbose=args.verbose,
        task=args.task,
        robot_id=args.robot_id,
        actions_out=args.actions_out,
    )
    return run_policy(cfg)


if __name__ == "__main__":
    sys.exit(main())
