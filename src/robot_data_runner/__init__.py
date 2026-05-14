"""robot_data_runner — closed-loop deployment of trained LeRobot policies on real SO-101.

Standalone counterpart of ``robot_data_recorder``:
  - recorder    teleop → LeRobotDataset
  - runner      LeRobotDataset-trained policy → real arm motors

Public API:
  - :class:`RunnerConfig`   — dataclass capturing CLI + safety knobs.
  - :func:`run_policy`      — main control loop (importable, not CLI-only).
  - :func:`load_policy`     — wrap ``lerobot.policies.factory.make_policy``.
  - :func:`obs_to_policy_input` / :func:`action_to_robot_dict`  — schema glue.
  - :class:`SafetyMonitor`  — stuck-action watchdog + repeat detector.

CLI entrypoints:
  - ``robot-data-run``        full deploy loop
  - ``robot-data-run-check``  pre-flight: load + smoke-print one action

Soft-imports: ``lerobot``, ``torch``. Neither is required to ``import
robot_data_runner`` itself — only at run/check call sites.
"""

from __future__ import annotations

from .config import RunnerConfig
from .policy_loader import load_policy
from .mappers import action_to_robot_dict, obs_to_policy_input
from .runner import run_policy
from .safety import SafetyMonitor

__all__ = [
    "RunnerConfig",
    "SafetyMonitor",
    "load_policy",
    "obs_to_policy_input",
    "action_to_robot_dict",
    "run_policy",
    "__version__",
]

__version__ = "0.1.0"
