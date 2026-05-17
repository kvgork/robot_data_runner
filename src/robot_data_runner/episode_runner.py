"""episode_runner — closed-loop multi-episode eval against the real SO-101.

Each episode:
    1. Prompt the user to reset the workspace.
    2. Loop policy → robot at ``cfg.rate_hz`` until task_spec terminates
       or ``duration_per_episode_s`` elapses.
    3. Score the episode (auto via task_spec.is_done success flag OR
       manual via task_spec.on_episode_end).
    4. Append a row to per-episode records.

After N episodes, aggregate to a JSON matching the dashboard's
``EVAL_SCHEMA`` so the Evaluation tab picks it up.

Soft-imports lerobot + torch (same as :mod:`runner`).
"""

from __future__ import annotations

import json
import logging
import platform
import socket
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import RunnerConfig
from .mappers import action_to_robot_dict, obs_to_policy_input
from .policy_loader import LoadedPolicy, load_policy
from .runner import _build_robot
from .safety import SafetyMonitor
from .task_specs import TaskSpec

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    index: int
    success: bool
    ep_len_s: float
    intervention: bool


def _prompt_reset(idx: int, prompt: str) -> None:
    msg = f"\n[reset] episode {idx}: {prompt}\n  > press ENTER when ready: "
    try:
        input(msg)
    except EOFError:
        logger.warning("stdin closed; assuming continue")


def run_episodes(
    cfg: RunnerConfig,
    *,
    task_spec: TaskSpec,
    n_episodes: int = 10,
    duration_per_episode_s: float = 10.0,
    reset_prompt: str = "match the demo's start pose (arm + object + camera)",
    output_json: Path,
    run_id: str | None = None,
    task_label: str = "so101-closed-loop",
    n_train_eps: int | None = None,
) -> Path:
    """Run a closed-loop multi-episode eval and write the result JSON."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    loaded: LoadedPolicy = load_policy(cfg.policy_path, cfg.dataset_root, cfg.seed)
    robot = _build_robot(cfg)
    monitor = SafetyMonitor(repeat_warn_steps=cfg.repeat_warn_steps)
    monitor.install_signal_handlers()
    robot.connect()

    motor_names = list(robot.bus.motors)
    dt = 1.0 / cfg.rate_hz
    records: list[EpisodeRecord] = []
    interrupted = False

    try:
        import torch

        for idx in range(n_episodes):
            if monitor.stop_flag:
                break
            _prompt_reset(idx, reset_prompt)
            task_spec.start_episode(robot)

            ep_start = time.monotonic()
            success_in_loop = False
            ep_done = False
            while not monitor.stop_flag:
                step_start = time.monotonic()
                t_in_ep = step_start - ep_start
                try:
                    obs = robot.get_observation()
                except Exception as exc:  # noqa: BLE001
                    logger.error("get_observation failed: %s", exc)
                    interrupted = True
                    break

                done, success_flag = task_spec.is_done(t_in_ep, obs)
                if done or t_in_ep >= duration_per_episode_s:
                    success_in_loop = bool(success_flag)
                    ep_done = True
                    break

                try:
                    with torch.no_grad():
                        pol_in = obs_to_policy_input(
                            obs, loaded.device, task=cfg.task
                        )
                        if loaded.preprocessor is not None:
                            pol_in = loaded.preprocessor(pol_in)
                        action = loaded.policy.select_action(pol_in)
                        if loaded.postprocessor is not None:
                            action = loaded.postprocessor(action)
                    action_dict = action_to_robot_dict(action, motor_names)
                except Exception as exc:  # noqa: BLE001
                    logger.error("policy inference failed: %s", exc)
                    interrupted = True
                    break

                monitor.observe(action_dict)
                if cfg.verbose:
                    logger.info(
                        "ep%d step action=%s",
                        idx,
                        {k: round(v, 2) for k, v in action_dict.items()},
                    )
                if cfg.execute:
                    try:
                        robot.send_action(action_dict)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("send_action failed: %s", exc)
                        interrupted = True
                        break

                slack = dt - (time.monotonic() - step_start)
                if slack > 0:
                    time.sleep(slack)

            ep_len = time.monotonic() - ep_start
            try:
                scored = task_spec.on_episode_end(idx, success_in_loop, ep_len)
            except KeyboardInterrupt:
                logger.warning("operator abort during episode %d scoring", idx)
                monitor.stop_flag = True
                break
            records.append(
                EpisodeRecord(
                    index=idx,
                    success=bool(scored),
                    ep_len_s=ep_len,
                    intervention=interrupted,
                )
            )
            logger.info(
                "episode %d: success=%s ep_len=%.1fs",
                idx,
                scored,
                ep_len,
            )
            if interrupted:
                break

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

    # ----- aggregate + write JSON -------------------------------------------
    if records:
        n_succ = sum(1 for r in records if r.success)
        n_intervention = sum(1 for r in records if r.intervention)
        pc_success = n_succ / len(records)
        mean_ep_len = sum(r.ep_len_s for r in records) / len(records)
        intervention_rate = n_intervention / len(records)
    else:
        pc_success = 0.0
        mean_ep_len = 0.0
        intervention_rate = 0.0

    payload = {
        "run_id": run_id or f"closed-loop-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        "task": task_label,
        "ts": datetime.now(UTC).isoformat(),
        "pc_success": pc_success,
        "n_episodes": len(records),
        "intervention_rate": intervention_rate,
        "mean_ep_len": mean_ep_len,
        "_metadata": {
            "source": "closed_loop_hardware",
            "policy_path": str(cfg.policy_path),
            "n_train_eps": n_train_eps,
            "task_spec_type": getattr(task_spec, "type", type(task_spec).__name__),
            "max_relative_target": cfg.max_relative_target,
            "rate_hz": cfg.rate_hz,
            "duration_per_episode_s": duration_per_episode_s,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "per_episode": [asdict(r) for r in records],
        },
    }
    output_json = Path(output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "closed-loop eval: pc_success=%.3f (%d/%d) intervention=%.2f written to %s",
        pc_success,
        sum(1 for r in records if r.success),
        len(records),
        intervention_rate,
        output_json,
    )
    return output_json
