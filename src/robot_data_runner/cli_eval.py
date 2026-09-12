"""robot-data-run-eval CLI — closed-loop multi-episode hardware eval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .cli import _parse_camera_spec
from .config import RunnerConfig, quiet_hf_hub
from .episode_runner import run_episodes
from .safety import InteractiveRequiredError
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
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=3.0,
        help=(
            "Per-joint max delta per control step. UNITS: DEGREES by default "
            "(e.g. 3.0 = 3 deg/step for the 5 arm joints; the gripper is "
            "always a percent of jaw travel regardless of this flag) — "
            "LOWER than open-loop deploy's 5.0 in the same unit space, "
            "tighter for closed-loop autonomy. Pass --as-normalized to "
            "revert to lerobot's legacy normalized (-100..100 over the FULL "
            "calibrated range) unit space instead — in that space the same "
            "number can imply a much faster real angular rate on wide-span "
            "joints (e.g. wrist_roll). The resolved per-joint deg/step + "
            "deg/s is always logged at startup (see describe_clamp); a run "
            "that would exceed --max-deg-per-s-ceiling refuses to start "
            "unless --allow-fast is passed."
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
        help="Refuse to start if the resolved worst-joint rate exceeds this (deg/s).",
    )
    p.add_argument("--home-on-exit", action="store_true")
    p.add_argument(
        "--id",
        dest="robot_id",
        default=None,
        help="Fixed SO-101 follower calibration id (so_follower/<id>.json).",
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
        "--task-spec",
        default="prompt_user_observer",
        help="prompt_user_observer | gripper_at_target_pose",
    )
    p.add_argument(
        "--target-joint-pos",
        default=None,
        help="comma-sep 6 floats (only for gripper_at_target_pose)",
    )
    p.add_argument("--tolerance-per-joint", type=float, default=3.0)
    p.add_argument("--require-gripper-closed", action="store_true")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--duration-per-episode-s", type=float, default=10.0)
    p.add_argument("--reset-prompt", default="match the demo's start pose (arm + object + camera)")
    p.add_argument("--output-json", default="outputs/eval/closed-loop.json")
    p.add_argument("--run-id", default=None)
    p.add_argument("--task-label", default="so101-closed-loop")
    p.add_argument(
        "--n-train-eps",
        type=int,
        default=None,
        help="metadata only — how many training eps the policy saw",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Language instruction passed to the policy each step. "
        "Required for VLA policies (SmolVLA). Match the "
        "training task string.",
    )
    p.add_argument(
        "--save-frames",
        dest="save_frames",
        default=None,
        help=(
            "Write the first observation frame of every episode to this "
            "directory, plus a frames_manifest.json (episode, camera, shape, "
            "dtype, mean pixel value per channel). Off by default. Use this "
            "to diagnose a 0/N result (camera framing, exposure drift, wrong "
            "video node) — see docs/runbook/11-closed-loop-eval.md."
        ),
    )
    p.add_argument(
        "--i-am-not-driving-an-arm",
        dest="i_am_not_driving_an_arm",
        action="store_true",
        help=(
            "Opt-out of the non-interactive stdin refusal for an "
            "execute=False (dry-run) call ONLY — this CLI always sets "
            "execute=True for closed-loop eval, so this flag has NO effect "
            "here; it exists for library callers of run_episodes() (e.g. "
            "smoke tests) that pass execute=False explicitly. It can NEVER "
            "rescue an execute=True run with no operator at a TTY."
        ),
    )
    p.add_argument(
        "--i-have-read-the-safety-runbook",
        action="store_true",
        help="one-time consent; stores marker in ~/.config/robot-data-runner/safety_ack",
    )
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
    quiet_hf_hub()  # stop per-load HF cache revalidation HTTP spam (after basicConfig)
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
        use_degrees=not args.as_normalized,
        max_deg_per_s_ceiling=args.max_deg_per_s_ceiling,
        allow_fast=args.allow_fast,
        execute=True,  # closed-loop requires motor writes
        home_on_exit=args.home_on_exit,
        repeat_warn_steps=args.repeat_warn_steps,
        hardware_watchdog=args.hardware_watchdog,
        hw_every_n_steps=args.hw_every_n_steps,
        hw_breach_steps=args.hw_breach_steps,
        seed=args.seed,
        verbose=args.verbose,
        task=args.task,
        robot_id=args.robot_id,
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

    try:
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
            save_frames_dir=Path(args.save_frames) if args.save_frames else None,
            i_am_not_driving_an_arm=args.i_am_not_driving_an_arm,
        )
    except InteractiveRequiredError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
