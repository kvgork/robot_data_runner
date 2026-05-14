"""robot-data-run CLI — thin wrapper around :func:`run_policy`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import RunnerConfig
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
    p.add_argument("--max-relative-target", type=float, default=5.0)
    p.add_argument("--use-degrees", action="store_true")
    p.add_argument(
        "--execute",
        action="store_true",
        help="enable real motor writes (default = dry-run)",
    )
    p.add_argument("--home-on-exit", action="store_true")
    p.add_argument("--repeat-warn-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
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
        use_degrees=args.use_degrees,
        execute=args.execute,
        home_on_exit=args.home_on_exit,
        repeat_warn_steps=args.repeat_warn_steps,
        seed=args.seed,
        verbose=args.verbose,
    )
    return run_policy(cfg)


if __name__ == "__main__":
    sys.exit(main())
