"""robot-data-run-eval CLI — closed-loop multi-episode hardware eval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .cli import _parse_camera_spec
from .config import RunnerConfig
from .episode_runner import run_episodes
from .task_specs import make_task_spec

_SAFETY_ACK_PATH = Path.home() / ".config" / "robot-data-runner" / "safety_ack"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robot-data-run-eval",
        description=(
            "Closed-loop N-episode evaluation of a trained LeRobot policy on "
            "the real SO-101. Executes motor writes — first invocation "
            "requires --i-have-read-the-safety-runbook."
        ),
    )
    p.add_argument("--policy-path", required=True)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--camera", action="append", default=[])
    p.add_argument("--rate-hz", type=float, default=30.0)
    p.add_argument("--max-relative-target", type=float, default=3.0,
                   help="default LOWER than open-loop deploy (3 deg vs 5)")
    p.add_argument("--use-degrees", action="store_true")
    p.add_argument("--home-on-exit", action="store_true")
    p.add_argument("--repeat-warn-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--task-spec", default="prompt_user_observer",
        help="prompt_user_observer | gripper_at_target_pose",
    )
    p.add_argument("--target-joint-pos", default=None,
                   help="comma-sep 6 floats (only for gripper_at_target_pose)")
    p.add_argument("--tolerance-per-joint", type=float, default=3.0)
    p.add_argument("--require-gripper-closed", action="store_true")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--duration-per-episode-s", type=float, default=10.0)
    p.add_argument("--reset-prompt",
                   default="match the demo's start pose (arm + object + camera)")
    p.add_argument("--output-json", default="outputs/eval/closed-loop.json")
    p.add_argument("--run-id", default=None)
    p.add_argument("--task-label", default="so101-closed-loop")
    p.add_argument("--n-train-eps", type=int, default=None,
                   help="metadata only — how many training eps the policy saw")
    p.add_argument("--task", default=None,
                   help="Language instruction passed to the policy each step. "
                        "Required for VLA policies (SmolVLA). Match the "
                        "training task string.")
    p.add_argument("--i-have-read-the-safety-runbook", action="store_true",
                   help="one-time consent; stores marker in ~/.config/robot-data-runner/safety_ack")
    return p


def _ensure_safety_ack(consent_flag: bool) -> None:
    if _SAFETY_ACK_PATH.exists():
        return
    if not consent_flag:
        print(
            "First closed-loop eval on this machine.\n"
            "Read docs/runbook/11-closed-loop-eval.md AND docs/runbook/10-deploy-to-hardware.md\n"
            "Then re-run with --i-have-read-the-safety-runbook to consent.",
            file=sys.stderr,
        )
        sys.exit(2)
    _SAFETY_ACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SAFETY_ACK_PATH.write_text("acked")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_parser().parse_args(argv)

    _ensure_safety_ack(args.i_have_read_the_safety_runbook)

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
        duration_s=args.duration_per_episode_s * args.n_episodes + 60,  # generous outer cap
        max_relative_target=args.max_relative_target,
        use_degrees=args.use_degrees,
        execute=True,  # closed-loop requires motor writes
        home_on_exit=args.home_on_exit,
        repeat_warn_steps=args.repeat_warn_steps,
        seed=args.seed,
        verbose=args.verbose,
        task=args.task,
    )

    spec_kwargs: dict = {"timeout_s": args.duration_per_episode_s}
    if args.task_spec == "gripper_at_target_pose":
        if not args.target_joint_pos:
            print("--target-joint-pos required for gripper_at_target_pose", file=sys.stderr)
            return 3
        spec_kwargs["target_joint_pos"] = [float(x) for x in args.target_joint_pos.split(",")]
        spec_kwargs["tolerance_per_joint"] = args.tolerance_per_joint
        spec_kwargs["require_gripper_closed"] = args.require_gripper_closed
    task = make_task_spec(args.task_spec, **spec_kwargs)

    out = run_episodes(
        cfg,
        task_spec=task,
        n_episodes=args.n_episodes,
        duration_per_episode_s=args.duration_per_episode_s,
        reset_prompt=args.reset_prompt,
        output_json=Path(args.output_json),
        run_id=args.run_id,
        task_label=args.task_label,
        n_train_eps=args.n_train_eps,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
