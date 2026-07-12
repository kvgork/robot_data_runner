"""run_policy — main control loop.

Reads observations from a connected ``SO101Follower``, runs the policy
forward, optionally sends motor commands, until ``duration_s`` elapses or
SIGINT is received.

Soft-imports lerobot; tests can monkey-patch the imports.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import RunnerConfig
from .mappers import action_to_robot_dict, obs_to_policy_input, resolve_state_mapping
from .policy_loader import LoadedPolicy, load_policy
from .safety import SafetyMonitor

logger = logging.getLogger(__name__)


def _build_robot(cfg: RunnerConfig) -> Any:
    """Construct an SO101Follower from the runner config (soft-import lerobot)."""
    try:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "lerobot >= 0.5 is required to drive an SO-101 follower. "
            "Install with: pip install lerobot"
        ) from exc

    cam_cfgs: dict[str, Any] = {}
    for name, (device, w, h) in cfg.cameras.items():
        cam_cfgs[name] = OpenCVCameraConfig(
            index_or_path=device, width=w, height=h, fps=30
        )

    follower_kwargs: dict[str, Any] = dict(
        port=cfg.port,
        max_relative_target=float(cfg.max_relative_target),
        use_degrees=bool(cfg.use_degrees),
        cameras=cam_cfgs,
    )
    # Fixed calibration id → stable <id>.json calibration (reused across runs).
    # Omitted when None so behaviour matches the anonymous default.
    if getattr(cfg, "robot_id", None):
        follower_kwargs["id"] = cfg.robot_id
    follower_cfg = SO101FollowerConfig(**follower_kwargs)
    return SO101Follower(follower_cfg)


def run_policy(cfg: RunnerConfig, loaded: LoadedPolicy | None = None) -> int:
    """Run the closed-loop policy on the real SO-101.

    Parameters
    ----------
    cfg:
        Fully-populated :class:`RunnerConfig`.
    loaded:
        Optional pre-loaded policy. If omitted, ``load_policy`` is invoked
        from the config.

    Returns
    -------
    int
        0 on clean exit; non-zero on a specific stage failure (matches the
        original deploy.py exit codes).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if loaded is None:
        try:
            loaded = load_policy(cfg.policy_path, cfg.dataset_root, cfg.seed)
        except Exception as exc:  # noqa: BLE001
            logger.error("policy load failed: %s", exc)
            return 2

    try:
        robot = _build_robot(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error("robot setup failed: %s", exc)
        return 3

    monitor = SafetyMonitor(repeat_warn_steps=cfg.repeat_warn_steps)
    monitor.install_signal_handlers()

    try:
        robot.connect()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "robot.connect() failed — is the arm plugged in? %s", exc
        )
        return 4

    motor_names = list(robot.bus.motors)
    # Resolve the state layout (canonical order + expected dim + zero-pad safety)
    # in one audited place. Raises rather than silently permute/mis-pad. The live
    # SO101 obs gives 6 joint positions; training recorded cat(pos[6], vel[6]=zeros)
    # so a 12-dim checkpoint is right-padded with a verified zero block.
    try:
        state_order, expected_state_dim = resolve_state_mapping(
            loaded.policy, loaded.ds_meta, motor_names, require_execute=cfg.execute
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("state mapping unsafe, refusing to run: %s", exc)
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        return 6
    logger.info(
        "state mapping: order=%s expected_state_dim=%s", state_order, expected_state_dim
    )
    dt = 1.0 / cfg.rate_hz
    deadline = time.monotonic() + cfg.duration_s
    rc = 0
    n_steps = 0

    # Optional trajectory capture: one JSON record per step, replayable by
    # run_replay. Opened once; closed in finally.
    actions_fh = None
    if cfg.actions_out is not None:
        cfg.actions_out.parent.mkdir(parents=True, exist_ok=True)
        actions_fh = cfg.actions_out.open("w")
        logger.info("recording actions → %s", cfg.actions_out)

    try:
        import torch  # local import; soft

        while not monitor.stop_flag and time.monotonic() < deadline:
            step_start = time.monotonic()
            try:
                obs = robot.get_observation()
            except Exception as exc:  # noqa: BLE001
                logger.error("get_observation failed: %s", exc)
                rc = 5
                break

            try:
                with torch.no_grad():
                    pol_in = obs_to_policy_input(
                        obs,
                        loaded.device,
                        task=cfg.task,
                        motor_order=motor_names,
                        state_dim=expected_state_dim,
                    )
                    if loaded.preprocessor is not None:
                        pol_in = loaded.preprocessor(pol_in)
                    action = loaded.policy.select_action(pol_in)
                    if loaded.postprocessor is not None:
                        action = loaded.postprocessor(action)
                action_dict = action_to_robot_dict(action, motor_names)
            except Exception as exc:  # noqa: BLE001
                logger.error("policy inference failed: %s", exc)
                rc = 6
                break

            monitor.observe(action_dict)
            if cfg.verbose:
                logger.info(
                    "step %d action=%s",
                    n_steps,
                    {k: round(v, 3) for k, v in action_dict.items()},
                )

            if actions_fh is not None:
                actions_fh.write(
                    json.dumps(
                        {
                            "step": n_steps,
                            "t": round(time.monotonic() - (deadline - cfg.duration_s), 4),
                            "action": {k: float(v) for k, v in action_dict.items()},
                        }
                    )
                    + "\n"
                )

            if cfg.execute:
                try:
                    robot.send_action(action_dict)
                except Exception as exc:  # noqa: BLE001
                    logger.error("send_action failed: %s", exc)
                    rc = 7
                    break
            n_steps += 1

            slack = dt - (time.monotonic() - step_start)
            if slack > 0:
                time.sleep(slack)
    finally:
        if actions_fh is not None:
            actions_fh.close()
        if cfg.home_on_exit and cfg.execute:
            try:
                logger.info("homing on exit (zero position)")
                robot.send_action({f"{m}.pos": 0.0 for m in motor_names})
                time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("home-on-exit failed: %s", exc)
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("robot.disconnect() failed: %s", exc)

    logger.info(
        "runner: %d steps in %.1fs (rc=%d, dry_run=%s)",
        n_steps,
        cfg.duration_s,
        rc,
        not cfg.execute,
    )
    return rc


def run_replay(cfg: RunnerConfig) -> int:
    """Replay a recorded action trajectory open-loop on the real SO-101.

    Reads ``cfg.replay_actions`` (a JSONL file written by ``run_policy`` with
    ``actions_out``) and sends each recorded action dict to the follower at
    ``cfg.rate_hz``. NO policy is loaded — this is pure trajectory playback.

    Safety: dry-run by default (``execute=False`` logs but does not move);
    per-joint ``max_relative_target`` is still enforced server-side by the
    follower; SafetyMonitor handles SIGINT → clean disconnect; ``home_on_exit``
    ramps to zero before disconnecting.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    if cfg.replay_actions is None or not Path(cfg.replay_actions).is_file():
        logger.error("replay_actions file not found: %s", cfg.replay_actions)
        return 2

    records = []
    for line in Path(cfg.replay_actions).read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        logger.error("replay file is empty: %s", cfg.replay_actions)
        return 2
    logger.info("replay: %d recorded steps from %s", len(records), cfg.replay_actions)

    try:
        robot = _build_robot(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error("robot setup failed: %s", exc)
        return 3

    monitor = SafetyMonitor(repeat_warn_steps=cfg.repeat_warn_steps)
    monitor.install_signal_handlers()
    try:
        robot.connect()
    except Exception as exc:  # noqa: BLE001
        logger.error("robot.connect() failed — is the arm plugged in? %s", exc)
        return 4

    motor_names = list(robot.bus.motors)
    dt = 1.0 / cfg.rate_hz
    rc = 0
    n_steps = 0
    try:
        for rec in records:
            if monitor.stop_flag:
                logger.info("stop requested — ending replay")
                break
            step_start = time.monotonic()
            # Keep only known motor .pos keys (drop any extra recorded fields).
            action_dict = {
                k: float(v)
                for k, v in rec.get("action", {}).items()
                if k.endswith(".pos") and k.split(".")[0] in motor_names
            }
            if not action_dict:
                continue
            monitor.observe(action_dict)
            if cfg.verbose:
                logger.info(
                    "replay step %d action=%s",
                    n_steps,
                    {k: round(v, 3) for k, v in action_dict.items()},
                )
            if cfg.execute:
                try:
                    robot.send_action(action_dict)
                except Exception as exc:  # noqa: BLE001
                    logger.error("send_action failed: %s", exc)
                    rc = 7
                    break
            n_steps += 1
            slack = dt - (time.monotonic() - step_start)
            if slack > 0:
                time.sleep(slack)
    finally:
        if cfg.home_on_exit and cfg.execute:
            try:
                logger.info("homing on exit (zero position)")
                robot.send_action({f"{m}.pos": 0.0 for m in motor_names})
                time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("home-on-exit failed: %s", exc)
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("robot.disconnect() failed: %s", exc)

    logger.info(
        "replay: %d/%d steps sent (rc=%d, dry_run=%s)",
        n_steps,
        len(records),
        rc,
        not cfg.execute,
    )
    return rc


def run_sweep(
    cfg: RunnerConfig,
    *,
    amplitude: float = 30.0,
    period_s: float = 8.0,
    cycles: float = 2.0,
    exclude_gripper: bool = True,
) -> int:
    """Move all joints TOGETHER through a safe reduced band of their range.

    Drives every joint with the same smooth sine target in lerobot's normalized
    position space, where -100 == calibrated range_min and +100 == range_max. So
    ``amplitude`` is a PERCENT of each joint's calibrated range, centered on the
    midpoint (amplitude=30 → swings ±30% of range, never near the limits).

    Pure geometric motion — no policy. Dry-run by default; ``execute`` enables
    motor writes. Per-step delta is still clamped by ``max_relative_target``;
    SIGINT → clean disconnect; ``home_on_exit`` ramps to center before exit.
    """
    import math

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    if not (0.0 < amplitude <= 90.0):
        logger.error("amplitude must be in (0, 90] %% of range; got %s", amplitude)
        return 2

    try:
        robot = _build_robot(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error("robot setup failed: %s", exc)
        return 3

    monitor = SafetyMonitor(repeat_warn_steps=cfg.repeat_warn_steps)
    monitor.install_signal_handlers()
    try:
        robot.connect()
    except Exception as exc:  # noqa: BLE001
        logger.error("robot.connect() failed — is the arm plugged in? %s", exc)
        return 4

    motor_names = list(robot.bus.motors)
    sweep_joints = [
        m for m in motor_names if not (exclude_gripper and "gripper" in m.lower())
    ]
    logger.info(
        "sweep: joints=%s amplitude=±%.0f%% period=%.1fs cycles=%.1f execute=%s",
        sweep_joints, amplitude, period_s, cycles, cfg.execute,
    )

    dt = 1.0 / cfg.rate_hz
    total_t = period_s * cycles
    t0 = time.monotonic()
    rc = 0
    n_steps = 0
    try:
        while not monitor.stop_flag and (time.monotonic() - t0) < total_t:
            step_start = time.monotonic()
            t = step_start - t0
            # sin starts at 0 (center) → smooth ramp out from the current pose.
            val = amplitude * math.sin(2.0 * math.pi * t / period_s)
            action_dict = {f"{m}.pos": float(val) for m in sweep_joints}
            monitor.observe(action_dict)
            if cfg.verbose:
                logger.info("sweep step %d val=%.2f", n_steps, val)
            if cfg.execute:
                try:
                    robot.send_action(action_dict)
                except Exception as exc:  # noqa: BLE001
                    logger.error("send_action failed: %s", exc)
                    rc = 7
                    break
            n_steps += 1
            slack = dt - (time.monotonic() - step_start)
            if slack > 0:
                time.sleep(slack)
    finally:
        if cfg.home_on_exit and cfg.execute:
            try:
                logger.info("homing on exit (center, val=0)")
                robot.send_action({f"{m}.pos": 0.0 for m in sweep_joints})
                time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("home-on-exit failed: %s", exc)
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("robot.disconnect() failed: %s", exc)

    logger.info(
        "sweep: %d steps in %.1fs (rc=%d, dry_run=%s)",
        n_steps, total_t, rc, not cfg.execute,
    )
    return rc
