"""robot-data-run-replay CLI — open-loop replay of a recorded trajectory.

Plays a JSONL action file (captured by ``robot-data-run --actions-out``) back
on the real SO-101 follower. NO policy is loaded — this is pure trajectory
playback. Dry-run by default; pass ``--execute`` to enable motor writes.

Typical flow::

    # 1. capture a motors-OFF dry-run trajectory
    robot-data-run --policy-path ... --camera d435_rgb=/dev/video2,640,480 \\
        --duration-s 60 --actions-out /tmp/dryrun.jsonl

    # 2. replay it on the arm WITH motors on
    robot-data-run-replay --actions /tmp/dryrun.jsonl --port /dev/ttyACM0 \\
        --id so101_follower --max-relative-target 5.0 --execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import RunnerConfig, quiet_hf_hub
from .runner import run_replay


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robot-data-run-replay",
        description=(
            "Open-loop replay of a recorded action trajectory on the real "
            "SO-101 follower. DRY-RUN by default — pass --execute for motors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--actions",
        required=True,
        help="JSONL trajectory file (from `robot-data-run --actions-out`).",
    )
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--rate-hz", type=float, default=30.0)
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="Per-joint max delta per step (server-side clip). Smaller = safer.",
    )
    p.add_argument("--use-degrees", action="store_true")
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
    p.add_argument("--repeat-warn-steps", type=int, default=30)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    quiet_hf_hub()  # no policy load here, but keep lerobot driver imports quiet
    args = _build_parser().parse_args(argv)
    # Validate the actions file up front with a clean message (run_replay also
    # guards, but build the cfg without touching disk).
    if not Path(args.actions).is_file():
        print(
            f"error: --actions file not found: {args.actions}\n"
            "Record one first:\n"
            "  robot-data-run ... --actions-out <FILE>   (motors-off dry run)",
            file=sys.stderr,
        )
        return 2

    cfg = RunnerConfig(
        # policy_path unused by run_replay but RunnerConfig requires it; use the
        # actions file path as a harmless placeholder.
        policy_path=Path(args.actions),
        port=args.port,
        rate_hz=args.rate_hz,
        # duration_s is unused by replay (length = #records) but must be > 0.
        duration_s=1.0,
        max_relative_target=args.max_relative_target,
        use_degrees=args.use_degrees,
        execute=args.execute,
        home_on_exit=args.home_on_exit,
        repeat_warn_steps=args.repeat_warn_steps,
        verbose=args.verbose,
        robot_id=args.robot_id,
        replay_actions=Path(args.actions),
    )
    return run_replay(cfg)


if __name__ == "__main__":
    sys.exit(main())
