# robot-data-runner

**Standalone CLI for running a trained LeRobot policy on the real SO-101
follower arm.** Sibling of `robot-data-recorder`:

| Package                | Direction                                   |
|------------------------|---------------------------------------------|
| `robot-data-recorder`  | SO-101 teleop → `LeRobotDataset` Parquet    |
| **`robot-data-runner`**| **`LeRobotDataset`-trained policy → real SO-101 motors** |

Stacked safety: dry-run default, server-side `max_relative_target` clip,
rate cap, stuck-action watchdog, SIGINT clean exit + optional home-on-exit.

---

## Install

Inside the `lerobot-isaac-training` workspace:

```bash
# One-time clone into a working tree (NO bare repo — same pattern as recorder)
pixi run sync-runner

# Pip-install into your training env
pixi run -e train-policy pip install -e src/robot-data-runner
```

Standalone (outside the workspace):

```bash
git clone https://github.com/kvgork/robot_data_runner.git
cd robot-data-runner
python3 -m pip install -e .
python3 -m pip install lerobot                                # heavy dep
```

---

## CLI Quickstart

```bash
# 1. Pre-flight — load checkpoint, dump expected schema. No hardware needed.
robot-data-run-check \
    --policy-path outputs/.../pretrained_model \
    --dataset-root datasets/<your-hf-user>/so101-pickplace1

# 2. DRY-RUN — connects, reads obs, runs policy, prints actions, NO motor writes.
robot-data-run \
    --policy-path outputs/.../pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/<your-hf-user>/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --duration-s 30 -v

# 3. EXECUTE — real motor writes, tight clamp, home on exit.
robot-data-run \
    --policy-path outputs/.../pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/<your-hf-user>/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --execute \
    --max-relative-target 3.0 \
    --home-on-exit \
    --duration-s 60
```

**The default is DRY-RUN.** Pass `--execute` to enable motor writes. Always
dry-run a fresh checkpoint first.

---

## Public API

```python
from pathlib import Path

from robot_data_runner import (
    RunnerConfig, SafetyMonitor,
    load_policy, run_policy,
    obs_to_policy_input, action_to_robot_dict,
)

cfg = RunnerConfig(
    policy_path=Path("/abs/path/to/pretrained_model"),
    port="/dev/ttyACM0",
    dataset_root=Path("/abs/path/to/datasets/<your-hf-user>/so101-pickplace1"),
    cameras={"d435_rgb": ("/dev/video0", 640, 480)},
    execute=False,           # dry-run
    max_relative_target=3.0,
    duration_s=30.0,
)
rc = run_policy(cfg)
```

---

## Safety Model (six layers)

| Layer | What | Default |
|-------|------|---------|
| 1     | Dry-run gated by `--execute` | OFF (dry-run) |
| 2     | `max_relative_target` server clip | 5 deg / step |
| 3     | Rate cap (`rate_hz`) | 30 Hz |
| 4     | Stuck-action watchdog (`repeat_warn_steps`) | 30 steps ≈ 1 s |
| 5     | SIGINT clean exit + optional `home_on_exit` | home OFF |
| 6     | Physical power switch | — |

See the in-workspace runbook `docs/runbook/10-deploy-to-hardware.md` for the
full safety walkthrough, hardware setup, and troubleshooting.

---

## Schema Glue (what the runner converts)

```
SO101Follower.get_observation()
  → {"shoulder_pan.pos": float, ..., "d435_rgb": ndarray (H,W,3) uint8}

obs_to_policy_input()
  → {"observation.state":            torch (1, 6) float32,
     "observation.images.d435_rgb":  torch (1, 3, H, W) float32 ∈ [0, 1]}

policy.select_action(...)
  → torch tensor shape (1, 6) float32 — joint position targets

action_to_robot_dict()
  → {"shoulder_pan.pos": float, ..., "gripper.pos": float}

SO101Follower.send_action(dict)   # IF --execute
```

Joint order: `shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper`.

---

## Limitations

- No closed-loop success metric on hardware — eval uses open-loop action-MSE.
- Single follower only (bimanual coord = future work).
- Isaac Sim deploy (`Isaac-SO101-PickPlace-v0` rollout) is a separate path.

---

## Related Packages

- `robot-data-recorder` — teleop → dataset (sibling, opposite direction)
- `lerobot-isaac-adapters` — training entrypoint (`lerobot-isaac-train`)
- `lerobot.robots.so_follower` — upstream driver this package wraps

See `docs/runbook/10-deploy-to-hardware.md` in the training workspace for
the full deployment walkthrough including calibration, troubleshooting,
and step-by-step recipe.
