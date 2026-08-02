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

Safety (domain safety review, 2026-08)
---------------------------------------
* **S5** — :func:`require_interactive` refuses to start when stdin is not a
  TTY and ``cfg.execute`` is True: running this under ``nohup``/``setsid``/a
  pipe used to silently swallow the reset/verdict prompts' ``EOFError`` and
  fabricate a real FAIL for every episode. The check runs before ANY
  robot/policy setup so a misconfigured detached run fails fast and loud.
* **S7** — every episode outcome is one of ``"success"``, ``"fail"``, or
  ``"aborted"`` (see :data:`OUTCOME_SUCCESS` et al.). ``"aborted"`` covers a
  SIGINT (Ctrl-C) or closed stdin (EOFError) arriving during a blocking
  reset/verdict prompt or mid-episode — it is excluded from the
  success-rate denominator and reported separately, never scored as a
  policy failure. See :func:`_aggregate_records`.
* **C1** — optional ``save_frames_dir`` writes the first observation frame
  of every episode plus a running manifest (see :mod:`frame_capture`) so a
  0/N result has something to diagnose (framing/exposure/wrong camera).
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

from . import frame_capture, safety_limits
from .config import RunnerConfig
from .mappers import (
    action_to_robot_dict,
    obs_to_policy_input,
    resolve_state_mapping,
)
from .policy_loader import LoadedPolicy, load_policy
from .runner import _build_robot, _clamp_and_warn, _home_on_exit, _startup_safety_check
from .safety import SafetyMonitor, require_interactive
from .task_specs import TaskSpec

logger = logging.getLogger(__name__)

# Episode outcomes. "success"/"fail" are policy verdicts (scored via
# task_spec); "aborted" is an operator-interlock event (SIGINT during a
# blocking prompt, or stdin EOF) — S7 requires it never be scored as a
# policy failure, and it is excluded from the pc_success denominator.
OUTCOME_SUCCESS = "success"
OUTCOME_FAIL = "fail"
OUTCOME_ABORTED = "aborted"


@dataclass
class EpisodeRecord:
    index: int
    outcome: str  # OUTCOME_SUCCESS | OUTCOME_FAIL | OUTCOME_ABORTED
    ep_len_s: float
    intervention: bool

    @property
    def success(self) -> bool:
        """Backward-compat boolean view — True only for a scored success."""
        return self.outcome == OUTCOME_SUCCESS


def _aggregate_records(records: list[EpisodeRecord]) -> dict:
    """Reduce ``records`` to the summary stats written into the result JSON.

    S7: ``OUTCOME_ABORTED`` episodes are excluded from every rate in the
    return value (``pc_success``, ``mean_ep_len``, ``intervention_rate``) —
    an operator abort is not a policy outcome and must not silently drag
    the success rate down (nor up). ``n_aborted`` reports them separately
    so they stay visible.

    Pure function — no I/O, no side effects. Kept standalone (rather than
    inlined in :func:`run_episodes`) so the denominator arithmetic is
    directly unit-testable without driving the whole episode loop.
    """
    scored = [r for r in records if r.outcome != OUTCOME_ABORTED]
    n_aborted = len(records) - len(scored)
    if scored:
        n_succ = sum(1 for r in scored if r.outcome == OUTCOME_SUCCESS)
        n_intervention = sum(1 for r in scored if r.intervention)
        pc_success = n_succ / len(scored)
        mean_ep_len = sum(r.ep_len_s for r in scored) / len(scored)
        intervention_rate = n_intervention / len(scored)
    else:
        n_succ = 0
        pc_success = 0.0
        mean_ep_len = 0.0
        intervention_rate = 0.0
    return {
        "n_succ": n_succ,
        "n_scored": len(scored),
        "n_aborted": n_aborted,
        "pc_success": pc_success,
        "mean_ep_len": mean_ep_len,
        "intervention_rate": intervention_rate,
    }


def _prompt_reset(idx: int, prompt: str) -> None:
    """Blocking reset-confirmation prompt.

    Raises ``KeyboardInterrupt`` (reusing the same operator-abort mechanism
    as ``TaskSpec.on_episode_end``'s typed "abort" — see
    :mod:`task_specs`) when stdin closes (``EOFError``), instead of
    silently treating a closed pipe as "continue" (S5). A detached/
    non-interactive run (nohup/setsid/a pipe) must abort the session, never
    run unattended episodes with a torqued arm and no operator.
    """
    msg = f"\n[reset] episode {idx}: {prompt}\n  > press ENTER when ready: "
    try:
        input(msg)
    except EOFError as exc:
        raise KeyboardInterrupt(
            f"operator abort: stdin closed during reset prompt for episode "
            f"{idx} — refusing to assume continue"
        ) from exc


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
    save_frames_dir: Path | None = None,
    i_am_not_driving_an_arm: bool = False,
) -> Path:
    """Run a closed-loop multi-episode eval and write the result JSON.

    Parameters
    ----------
    save_frames_dir:
        When set (CLI: ``--save-frames DIR``), writes the first observation
        frame of every episode plus ``DIR/frames_manifest.json`` (see
        :mod:`frame_capture`). Off by default.
    i_am_not_driving_an_arm:
        Opt-out for :func:`robot_data_runner.safety.require_interactive`'s
        non-interactive refusal. Only takes effect when ``cfg.execute`` is
        False (a dry run) — it can NEVER rescue ``cfg.execute=True`` (S5).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # S5 gate — runs BEFORE any robot/policy setup so a misconfigured
    # detached run (nohup/setsid/a pipe) fails fast, loud, and with zero
    # side effects (no connect(), no policy load, no torqued arm).
    require_interactive(
        "closed-loop eval (run_episodes)",
        execute=cfg.execute,
        allow_non_interactive=i_am_not_driving_an_arm,
    )

    loaded: LoadedPolicy = load_policy(cfg.policy_path, cfg.dataset_root, cfg.seed)
    robot = _build_robot(cfg)
    monitor = SafetyMonitor(repeat_warn_steps=cfg.repeat_warn_steps)
    monitor.install_signal_handlers()
    robot.connect()

    motor_names = list(robot.bus.motors)

    rc = _startup_safety_check(cfg, robot)
    if rc is not None:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            "refusing closed-loop eval: worst-joint implied rate exceeds "
            "max_deg_per_s_ceiling (see preceding log line); pass "
            "allow_fast=True / --allow-fast to override"
        )
    limits_lo, limits_hi = safety_limits.resolve_joint_limits(robot)

    # Same audited state-layout resolution as run_policy: canonical order +
    # expected dim + verified zero-pad. Closed-loop eval always writes motors
    # (execute forced True upstream), so require_execute=True.
    try:
        state_order, expected_state_dim = resolve_state_mapping(
            loaded.policy, loaded.ds_meta, motor_names, require_execute=cfg.execute
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("state mapping unsafe, refusing closed-loop eval: %s", exc)
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        raise
    logger.info("state mapping: order=%s expected_state_dim=%s", state_order, expected_state_dim)
    dt = 1.0 / cfg.rate_hz
    records: list[EpisodeRecord] = []
    frame_manifest_rows: list[dict] = []
    interrupted = False

    try:
        import torch

        for idx in range(n_episodes):
            if monitor.stop_flag:
                break

            try:
                _prompt_reset(idx, reset_prompt)
            except KeyboardInterrupt:
                logger.warning(
                    "operator abort before episode %d started (reset prompt)", idx
                )
                monitor.stop_flag = True
                records.append(
                    EpisodeRecord(
                        index=idx, outcome=OUTCOME_ABORTED, ep_len_s=0.0, intervention=False
                    )
                )
                break

            if monitor.stop_flag:
                # SIGINT arrived while the reset prompt's input() was
                # blocked (PEP 475 retries syscalls on EINTR, so a signal
                # alone can't interrupt it — the prompt only returns once
                # ENTER is pressed). Treat exactly like an abort: do not
                # run the episode, do not fabricate a verdict (S7).
                logger.warning("operator abort (signal) before episode %d started", idx)
                records.append(
                    EpisodeRecord(
                        index=idx, outcome=OUTCOME_ABORTED, ep_len_s=0.0, intervention=False
                    )
                )
                break

            task_spec.start_episode(robot)

            # Fresh per-episode warn-tracking: "once per joint per episode",
            # not once per whole eval run — a fault that recurs in a later
            # episode should still be surfaced.
            clamp_warned: set[str] = set()

            ep_start = time.monotonic()
            success_in_loop = False
            frame_captured = False
            while not monitor.stop_flag:
                step_start = time.monotonic()
                t_in_ep = step_start - ep_start
                try:
                    obs = robot.get_observation()
                except Exception as exc:  # noqa: BLE001
                    logger.error("get_observation failed: %s", exc)
                    interrupted = True
                    break

                if save_frames_dir is not None and not frame_captured:
                    try:
                        rows = frame_capture.save_episode_frame(obs, save_frames_dir, idx)
                        frame_manifest_rows.extend(rows)
                        frame_capture.write_manifest(save_frames_dir, frame_manifest_rows)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "save-frames: capture failed for episode %d: %s", idx, exc
                        )
                    frame_captured = True

                done, success_flag = task_spec.is_done(t_in_ep, obs)
                if done or t_in_ep >= duration_per_episode_s:
                    success_in_loop = bool(success_flag)
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
                    interrupted = True
                    break

                action_dict = _clamp_and_warn(action_dict, limits_lo, limits_hi, clamp_warned)
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

            if monitor.stop_flag:
                # The while loop exited because its own condition failed
                # (stop_flag flipped by a signal), NOT via one of the
                # explicit `break`s above (done/timeout/get_observation/
                # inference/send_action error). That distinguishes an
                # operator abort mid-episode from a natural or error exit —
                # never call on_episode_end for it (would either block on a
                # second prompt or silently fabricate a verdict); record it
                # distinctly instead (S7).
                logger.warning("operator abort (signal) during episode %d — not scored", idx)
                records.append(
                    EpisodeRecord(
                        index=idx,
                        outcome=OUTCOME_ABORTED,
                        ep_len_s=ep_len,
                        intervention=interrupted,
                    )
                )
                break

            try:
                scored = task_spec.on_episode_end(idx, success_in_loop, ep_len)
            except KeyboardInterrupt:
                logger.warning("operator abort during episode %d scoring", idx)
                monitor.stop_flag = True
                records.append(
                    EpisodeRecord(
                        index=idx,
                        outcome=OUTCOME_ABORTED,
                        ep_len_s=ep_len,
                        intervention=interrupted,
                    )
                )
                break

            records.append(
                EpisodeRecord(
                    index=idx,
                    outcome=OUTCOME_SUCCESS if scored else OUTCOME_FAIL,
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
            _home_on_exit(robot, motor_names, cfg)
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("robot.disconnect() failed: %s", exc)

    # ----- aggregate + write JSON -------------------------------------------
    agg = _aggregate_records(records)

    payload = {
        "run_id": run_id or f"closed-loop-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        "task": task_label,
        "ts": datetime.now(UTC).isoformat(),
        "pc_success": agg["pc_success"],
        "n_episodes": len(records),
        "n_scored": agg["n_scored"],
        "n_aborted": agg["n_aborted"],
        "intervention_rate": agg["intervention_rate"],
        "mean_ep_len": agg["mean_ep_len"],
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
            "per_episode": [{**asdict(r), "success": r.success} for r in records],
        },
    }
    output_json = Path(output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "closed-loop eval: pc_success=%.3f (%d/%d scored, %d aborted) intervention=%.2f "
        "written to %s",
        agg["pc_success"],
        agg["n_succ"],
        agg["n_scored"],
        agg["n_aborted"],
        agg["intervention_rate"],
        output_json,
    )
    return output_json
