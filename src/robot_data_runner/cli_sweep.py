"""robot-data-run-sweep CLI — move all joints together through a safe band.

Geometric range-of-motion check (no policy, no camera). Drives every joint with
the same smooth sine in lerobot's normalized position space, where -100 ==
calibrated range_min and +100 == range_max. ``--amplitude`` is therefore a
PERCENT of each joint's calibrated range, centered on the midpoint — so
``--amplitude 30`` swings ±30% of range and never approaches the limits.

DRY-RUN by default — pass ``--execute`` for motor writes.

Example (motors on, ±30% of range, 8 s/cycle, 2 cycles)::

    robot-data-run-sweep --port /dev/ttyACM0 --id so101_follower \\
        --amplitude 30 --period-s 8 --cycles 2 \\
        --max-relative-target 5.0 --home-on-exit --execute
"""

from __future__ import annotations

import argparse
import sys

from .config import RunnerConfig, quiet_hf_hub
from .runner import run_sweep


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robot-data-run-sweep",
        description=(
            "Move all SO-101 joints together through a safe reduced band of "
            "their calibrated range (sine). DRY-RUN by default — --execute for "
            "motors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--rate-hz", type=float, default=30.0)
    p.add_argument(
        "--amplitude",
        type=float,
        default=30.0,
        help="Swing as %% of each joint's calibrated range, centered (0–90]. "
        "Default 30 = ±30%%. Smaller = safer.",
    )
    p.add_argument(
        "--period-s",
        type=float,
        default=8.0,
        help="Seconds for one full center→+amp→-amp→center cycle. Default 8.",
    )
    p.add_argument(
        "--cycles", type=float, default=2.0, help="Number of sine cycles. Default 2."
    )
    p.add_argument(
        "--include-gripper",
        action="store_true",
        help="Also sweep the gripper joint (excluded by default).",
    )
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="Per-joint max delta per step (server-side clip). Smaller = safer.",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Enable real motor writes (default = dry-run, logs only).",
    )
    p.add_argument("--home-on-exit", action="store_true")
    p.add_argument(
        "--id",
        dest="robot_id",
        default=None,
        help="Fixed SO-101 follower calibration id (so_follower/<id>.json).",
    )
    p.add_argument("--repeat-warn-steps", type=int, default=10_000,
                   help="Sweep repeats similar values by design; high default "
                        "avoids spurious stuck-action warnings.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    quiet_hf_hub()
    args = _build_parser().parse_args(argv)
    cfg = RunnerConfig(
        policy_path="<sweep>",  # unused by run_sweep; RunnerConfig requires it
        port=args.port,
        rate_hz=args.rate_hz,
        duration_s=1.0,  # unused by sweep (length = period_s*cycles)
        max_relative_target=args.max_relative_target,
        execute=args.execute,
        home_on_exit=args.home_on_exit,
        repeat_warn_steps=args.repeat_warn_steps,
        verbose=args.verbose,
        robot_id=args.robot_id,
    )
    return run_sweep(
        cfg,
        amplitude=args.amplitude,
        period_s=args.period_s,
        cycles=args.cycles,
        exclude_gripper=not args.include_gripper,
    )


if __name__ == "__main__":
    sys.exit(main())
