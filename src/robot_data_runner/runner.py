"""run_policy — main control loop.

Reads observations from a connected ``SO101Follower``, runs the policy
forward, optionally sends motor commands, until ``duration_s`` elapses or
SIGINT is received.

Soft-imports lerobot; tests can monkey-patch the imports.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .config import RunnerConfig
from .mappers import action_to_robot_dict, obs_to_policy_input
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

    follower_cfg = SO101FollowerConfig(
        port=cfg.port,
        max_relative_target=float(cfg.max_relative_target),
        use_degrees=bool(cfg.use_degrees),
        cameras=cam_cfgs,
    )
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
    dt = 1.0 / cfg.rate_hz
    deadline = time.monotonic() + cfg.duration_s
    rc = 0
    n_steps = 0

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
                    pol_in = obs_to_policy_input(obs, loaded.device)
                    action = loaded.policy.select_action(pol_in)
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
        "runner: %d steps in %.1fs (rc=%d, dry_run=%s)",
        n_steps,
        cfg.duration_s,
        rc,
        not cfg.execute,
    )
    return rc
