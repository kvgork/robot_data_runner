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

from . import safety_limits
from .config import RunnerConfig
from .mappers import action_to_robot_dict, obs_to_policy_input, resolve_state_mapping
from .policy_loader import LoadedPolicy, load_policy
from .safety import SafetyMonitor, attach_hardware_watchdog

logger = logging.getLogger(__name__)

# S7 fix (domain safety review, 2026-08): printed by every execute=True run
# path via _startup_safety_check (shared by run_policy / run_replay /
# run_sweep / run_episodes). A Python signal handler cannot interrupt a
# blocked serial read/write (PEP 475 retries syscalls on EINTR), so Ctrl-C
# only takes effect at the next checked boundary — it is NOT a reliable
# e-stop. Making this explicit at startup is the mitigation: there is no
# code fix for "signals don't preempt blocking I/O", only an informed
# operator.
_NO_ESTOP_BANNER = (
    "=" * 78
    + "\nSAFETY: there is NO software e-stop.\n"
    "  Ctrl-C only flips a flag checked between control steps — it will NOT\n"
    "  reliably stop a robot already moving mid-command (a blocked serial\n"
    "  read/write is not interrupted; PEP 475 retries syscalls on EINTR).\n"
    "  The ONLY reliable stop is cutting power to the arm.\n"
    "  In a closed-loop eval session, type 'abort' at the verdict prompt for\n"
    "  a clean operator abort (excluded from the success rate) — do not rely\n"
    "  on Ctrl-C alone.\n" + "=" * 78
)


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
        cam_cfgs[name] = OpenCVCameraConfig(index_or_path=device, width=w, height=h, fps=30)

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


def _startup_safety_check(cfg: RunnerConfig, robot: Any) -> int | None:
    """Log the S1 unit-legibility banner; enforce the S1 deg/s safety ceiling.

    Always logs :func:`safety_limits.describe_clamp` plus an explicit line
    naming the active unit mode — this is the loud startup banner required so
    ``max_relative_target``'s unit is never silently ambiguous. When
    ``cfg.execute`` is set, ALSO logs :data:`_NO_ESTOP_BANNER` (S7) — this is
    the single choke point shared by every run path that can drive the arm
    (``run_policy``, ``run_replay``, ``run_sweep``, ``run_episodes``), so the
    no-e-stop warning cannot be skipped by adding a new call site.

    The hard refusal (via :func:`safety_limits.check_rate_ceiling`) only
    fires when ``cfg.execute`` is set (i.e. this run can actually move the
    arm) and the worst-joint implied rate exceeds ``cfg.max_deg_per_s_ceiling``
    — dry runs always proceed so smoke tests / CI aren't blocked by a
    calibration-dependent ceiling. Returns an exit code to abort with, or
    ``None`` to continue.
    """
    banner = safety_limits.describe_clamp(
        cfg.max_relative_target, cfg.use_degrees, robot, cfg.rate_hz
    )
    unit = "DEGREES" if cfg.use_degrees else "NORMALIZED (--as-normalized)"
    logger.info("units in force: %s\n%s", unit, banner)

    if cfg.execute:
        logger.warning(_NO_ESTOP_BANNER)

    if not cfg.execute:
        return None

    try:
        safety_limits.check_rate_ceiling(
            cfg.max_relative_target,
            cfg.use_degrees,
            robot,
            cfg.rate_hz,
            ceiling=cfg.max_deg_per_s_ceiling,
            allow_fast=cfg.allow_fast,
        )
    except safety_limits.RateCeilingExceeded as exc:
        logger.error("refusing to start: %s", exc)
        return 8
    return None


def _clamp_and_warn(
    action_dict: dict[str, float],
    lo: dict[str, float],
    hi: dict[str, float],
    warned: set[str],
) -> dict[str, float]:
    """Absolute joint-limit clamp (S3) shared by every send_action call site.

    Logs a WARNING the first time a given joint is clamped (tracked via
    ``warned``, one set per run/episode) — never per step, which would
    flood the log for a policy that persistently saturates a limit.
    """
    clamped_dict, clamped_joints = safety_limits.clamp_action(action_dict, lo, hi)
    for joint in clamped_joints:
        if joint not in warned:
            logger.warning(
                "joint %r action clamped to absolute safety limit [%.2f, %.2f]",
                joint,
                lo[joint],
                hi[joint],
            )
            warned.add(joint)
    return clamped_dict


def _home_on_exit(robot: Any, motor_names: list[str], cfg: RunnerConfig) -> None:
    """Ramp every joint to zero (S4 fix) — replaces the old single-shot send.

    The old pattern (one ``send_action({..: 0.0})``, ``sleep(0.5)``,
    ``disconnect()``) is not a ramp: it is a single instant goto bounded only
    by ``max_relative_target``, after which ``disconnect()`` disables torque
    by default — the arm goes limp and drops from wherever it stopped, still
    gripping any payload. :func:`safety_limits.ramped_home` steps toward zero
    at a bounded ``max_step_deg`` per tick and re-reads position every
    iteration instead.
    """
    try:
        logger.info("homing on exit (ramped to zero position)")
        safety_limits.ramped_home(robot, motor_names, rate_hz=cfg.rate_hz)
    except Exception as exc:  # noqa: BLE001
        logger.warning("home-on-exit failed: %s", exc)


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
        logger.error("robot.connect() failed — is the arm plugged in? %s", exc)
        return 4

    attach_hardware_watchdog(monitor, robot, cfg)
    motor_names = list(robot.bus.motors)

    rc = _startup_safety_check(cfg, robot)
    if rc is not None:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        return rc
    limits_lo, limits_hi = safety_limits.resolve_joint_limits(robot)
    clamp_warned: set[str] = set()

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
    logger.info("state mapping: order=%s expected_state_dim=%s", state_order, expected_state_dim)
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

            action_dict = _clamp_and_warn(action_dict, limits_lo, limits_hi, clamp_warned)
            monitor.observe(action_dict)
            monitor.observe_hardware(robot)
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
            _home_on_exit(robot, motor_names, cfg)
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
    follower; an additional absolute-position clamp (see
    :mod:`robot_data_runner.safety_limits`) is applied client-side before
    every send; SafetyMonitor handles SIGINT → clean disconnect;
    ``home_on_exit`` ramps to zero (:func:`safety_limits.ramped_home`) before
    disconnecting — it does NOT do an instant goto.

    Unit mode (``cfg.use_degrees``) is NOT overridden here (unlike
    ``run_sweep``) — it defaults to degrees same as ``run_policy``, so a
    trajectory recorded/scored via ``robot-data-run`` and replayed via
    ``robot-data-run-replay`` (e.g. a negative-control comparison) uses the
    identical clamp semantics unless the caller explicitly opts out on both
    sides with ``--as-normalized``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

    attach_hardware_watchdog(monitor, robot, cfg)
    motor_names = list(robot.bus.motors)

    rc = _startup_safety_check(cfg, robot)
    if rc is not None:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        return rc
    limits_lo, limits_hi = safety_limits.resolve_joint_limits(robot)
    clamp_warned: set[str] = set()

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
            action_dict = _clamp_and_warn(action_dict, limits_lo, limits_hi, clamp_warned)
            monitor.observe(action_dict)
            monitor.observe_hardware(robot)
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
            _home_on_exit(robot, motor_names, cfg)
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

    Because this tool's entire semantics depend on the normalized -100..100
    space, it always drives the robot with ``use_degrees=False`` regardless of
    ``cfg.use_degrees`` (see the ``_build_robot`` override below) — degrees
    would silently reinterpret ``amplitude`` as a raw angle instead of a
    percent of range.

    Pure geometric motion — no policy. Dry-run by default; ``execute`` enables
    motor writes. Per-step delta is still clamped by ``max_relative_target``
    server-side, PLUS a client-side absolute joint-limit clamp (see
    :mod:`robot_data_runner.safety_limits`); SIGINT → clean disconnect;
    ``home_on_exit`` ramps to zero (:func:`safety_limits.ramped_home`) before
    exit — it does NOT do an instant goto.
    """
    import math
    from dataclasses import replace

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not (0.0 < amplitude <= 90.0):
        logger.error("amplitude must be in (0, 90] %% of range; got %s", amplitude)
        return 2

    # PINNED — DO NOT "FIX" THIS: run_sweep's --amplitude is a percent of each
    # joint's calibrated range by definition (see the docstring above), which
    # ONLY means what it says in lerobot's normalized (-100..100) unit space.
    # This is unrelated to (and must stay independent of) the S1 default of
    # RunnerConfig.use_degrees=True — flipping that package-wide default must
    # never silently reinterpret --amplitude as a raw degree value here.
    sweep_cfg = replace(cfg, use_degrees=False)

    try:
        robot = _build_robot(sweep_cfg)
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

    attach_hardware_watchdog(monitor, robot, cfg)
    motor_names = list(robot.bus.motors)
    sweep_joints = [m for m in motor_names if not (exclude_gripper and "gripper" in m.lower())]
    logger.info(
        "sweep: joints=%s amplitude=±%.0f%% period=%.1fs cycles=%.1f execute=%s",
        sweep_joints,
        amplitude,
        period_s,
        cycles,
        cfg.execute,
    )

    rc = _startup_safety_check(sweep_cfg, robot)
    if rc is not None:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        return rc
    limits_lo, limits_hi = safety_limits.resolve_joint_limits(robot)
    clamp_warned: set[str] = set()

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
            action_dict = _clamp_and_warn(action_dict, limits_lo, limits_hi, clamp_warned)
            monitor.observe(action_dict)
            monitor.observe_hardware(robot)
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
            _home_on_exit(robot, sweep_joints, cfg)
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("robot.disconnect() failed: %s", exc)

    logger.info(
        "sweep: %d steps in %.1fs (rc=%d, dry_run=%s)",
        n_steps,
        total_t,
        rc,
        not cfg.execute,
    )
    return rc
