# robot-data-runner — Package Orientation

**Role:** Standalone CLI for closed-loop deployment of a trained LeRobot
policy on the real SO-101 follower arm. Sibling of `robot-data-recorder`
(opposite direction: teleop → dataset).

**Upstream:** [`github.com/kvgork/robot_data_runner`](https://github.com/kvgork/robot_data_runner).
Sibling of `robot_data_recorder` (opposite direction: teleop → dataset).

**Install path in workspace:**
```bash
pixi run sync-runner
pixi run -e train-policy pip install -e src/robot-data-runner
```

---

## Public API

```python
from robot_data_runner import (
    RunnerConfig,       # dataclass — all knobs
    SafetyMonitor,      # stuck-action watchdog + SIGINT handler
    load_policy,        # wraps lerobot.policies.factory.make_policy
    run_policy,         # main control loop (importable, not CLI-only)
    obs_to_policy_input, action_to_robot_dict,  # schema glue
)
```

Console scripts:
- `robot-data-run` — full deploy loop
- `robot-data-run-check` — pre-flight: load + dump expected I/O schema

---

## File Map

| File | Role |
|------|------|
| `src/robot_data_runner/__init__.py` | Public exports + `__version__`. |
| `src/robot_data_runner/config.py` | `RunnerConfig` dataclass. |
| `src/robot_data_runner/policy_loader.py` | `load_policy` (soft-imports lerobot+torch). |
| `src/robot_data_runner/mappers.py` | `obs_to_policy_input` / `action_to_robot_dict`. |
| `src/robot_data_runner/safety.py` | `SafetyMonitor` (stuck-action + SIGINT). |
| `src/robot_data_runner/runner.py` | `run_policy` control loop. |
| `src/robot_data_runner/cli.py` | `robot-data-run` argparse + main. |
| `src/robot_data_runner/check_policy.py` | `robot-data-run-check` preflight. |
| `tests/test_imports.py` | 4 smoke tests — module imports + config defaults + safety streak + CLI parser. |

---

## Coupling

- **Hard deps:** `numpy` (everything else soft-imported).
- **Soft deps:** `lerobot >= 0.5`, `torch`. The package imports without
  either; `load_policy` / `run_policy` raise `ImportError` with an
  actionable install hint when invoked.
- **Sibling deps:** none. Standalone like `robot-data-recorder`.

---

## Safety Layers (six)

Documented in `README.md` + the training-workspace runbook
`docs/runbook/10-deploy-to-hardware.md`. Summary:

1. Dry-run by default (`--execute` to override).
2. `max_relative_target` server-side clip in FeetechMotorsBus.
3. Rate cap (`rate_hz`, default 30 Hz).
4. Stuck-action watchdog (`SafetyMonitor`, default 30-step threshold).
5. SIGINT/SIGTERM clean exit + optional `home_on_exit`.
6. Physical power switch.

---

## Why Standalone

Mirrors the `robot_data_recorder` decision: hardware-bound CLIs that
depend on lerobot's robot drivers stay outside the meta package so:

- Workspaces without an SO-101 don't accidentally pull the dep.
- The package can be released to GitHub / PyPI independently.
- Bug fixes ship per-repo, not gated by a monorepo bump.

The training workspace exposes it through:

```toml
# pixi.toml
sync-runner = "bash scripts/sync/sync_runner.sh"
deploy      = "robot-data-run"
```

---

## Testing

```bash
python3 -m pytest tests/ -q
```

All tests use mocks — no hardware required.

---

## Spinout Status

Phase 0 — scaffold complete. Logic moved from
`lerobot-isaac-adapters.deploy` (which remains as a backward-compat
re-export). All tests pass without lerobot/torch installed.

---

## References

- Training-workspace runbook: `docs/runbook/10-deploy-to-hardware.md`
- Sibling package: [`robot_data_recorder`](https://github.com/kvgork/robot_data_recorder)
- Upstream driver: `lerobot.robots.so_follower.SO101Follower`
- Original implementation (kept as re-export): `lerobot_isaac_adapters.deploy`
