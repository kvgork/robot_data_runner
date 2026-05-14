"""SafetyMonitor — stuck-action watchdog + SIGINT handler.

Sits inside the runner loop and tracks consecutive same-action steps.
Warns once when the threshold is hit. Also installs SIGINT/SIGTERM
handlers that flip the shared `stop_flag`.
"""

from __future__ import annotations

import logging
import signal
from typing import Any

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """Stuck-action watchdog. One instance per runner loop.

    Parameters
    ----------
    repeat_warn_steps:
        Consecutive ε-identical actions before WARNING is logged.
    epsilon:
        Per-dim tolerance for "same" action (default 1e-4).
    """

    def __init__(self, repeat_warn_steps: int = 30, epsilon: float = 1e-4) -> None:
        self.repeat_warn_steps = repeat_warn_steps
        self.epsilon = epsilon
        self._last: dict[str, float] | None = None
        self._streak = 0
        self._warned = False
        self.stop_flag = False

    # --- public API ---------------------------------------------------------
    def observe(self, action_dict: dict[str, float]) -> None:
        if self._last is not None and self._same(self._last, action_dict):
            self._streak += 1
            if self._streak == self.repeat_warn_steps and not self._warned:
                logger.warning(
                    "policy action unchanged for %d consecutive steps — "
                    "policy may be stuck or returning NaN; consider e-stop",
                    self._streak,
                )
                self._warned = True
        else:
            self._streak = 0
            self._warned = False
        self._last = dict(action_dict)

    def install_signal_handlers(self) -> None:
        """Wire SIGINT + SIGTERM to flip ``stop_flag``."""

        def _h(_sig: int, _frame: Any) -> None:
            self.stop_flag = True
            logger.info("signal received — stopping at next step boundary")

        signal.signal(signal.SIGINT, _h)
        signal.signal(signal.SIGTERM, _h)

    # --- helpers ------------------------------------------------------------
    def _same(self, a: dict[str, float], b: dict[str, float]) -> bool:
        if set(a) != set(b):
            return False
        return all(abs(a[k] - b[k]) < self.epsilon for k in a)
