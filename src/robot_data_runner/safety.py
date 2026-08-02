"""SafetyMonitor — stuck-action watchdog + SIGINT handler.

Sits inside the runner loop and tracks consecutive same-action steps.
Warns once when the threshold is hit. Also installs SIGINT/SIGTERM
handlers that flip the shared `stop_flag`.

Also home to :func:`require_interactive` (S5 fix, 2026-08 domain safety
review): a TTY gate for any run path that drives the arm via a
human-interlocked session (reset/verdict prompts). See its docstring for
why this is needed — short version: catching ``EOFError`` and "assuming
continue" under a detached/non-interactive launch (nohup/setsid/a pipe)
silently deletes the operator interlock AND fabricates a clean-looking
0/N result.

IMPORTANT — no software e-stop: a Python ``signal`` handler cannot
interrupt a blocked serial read/write (PEP 475 retries syscalls on
``EINTR``), so flipping ``stop_flag`` from SIGINT only takes effect at the
next checked boundary (top of a loop, or after a blocking ``input()``
call returns). Ctrl-C is therefore NOT a reliable e-stop; the only
reliable stop is cutting power to the arm. Callers that drive the arm
must print a startup banner saying so (see ``runner._NO_ESTOP_BANNER``).
"""

from __future__ import annotations

import logging
import signal
import sys
from typing import Any

logger = logging.getLogger(__name__)


class InteractiveRequiredError(RuntimeError):
    """Raised by :func:`require_interactive` when stdin is not a TTY.

    Two distinct refusal modes:

    * ``execute=True`` (this run will drive the arm) — ALWAYS refuses;
      there is no opt-out. A closed-loop hardware session must have an
      operator at the prompt able to type ``abort``.
    * ``execute=False`` (dry-run, no motor writes) — refuses UNLESS the
      caller passes ``allow_non_interactive=True`` (CLI:
      ``--i-am-not-driving-an-arm``). This lets smoke tests / CI exercise
      the episode loop with fakes and no real TTY, while still requiring
      an explicit, named acknowledgement that no arm is being driven.
    """


def require_interactive(
    context: str,
    *,
    execute: bool,
    allow_non_interactive: bool = False,
) -> None:
    """Refuse to proceed with a human-interlocked run when stdin is not a TTY.

    S5 fix: running a closed-loop eval under ``nohup``/``setsid``/a pipe used
    to silently swallow the reset/verdict prompts' ``EOFError`` — episodes
    ran back-to-back with a live torqued arm and no operator, and every
    verdict prompt silently recorded a real FAIL, producing a clean-looking
    ``pc_success=0.0`` indistinguishable from a genuine 0/N. This check runs
    BEFORE any robot/policy setup so a misconfigured detached run fails
    fast, loud, and without side effects (no connect(), no policy load).

    Parameters
    ----------
    context:
        Short human-readable label for the calling run path (used in the
        error message only).
    execute:
        Whether this run will send real motor commands. When True, refusal
        is unconditional — ``allow_non_interactive`` cannot rescue it.
    allow_non_interactive:
        Opt-out for the ``execute=False`` case only (CLI:
        ``--i-am-not-driving-an-arm``).
    """
    if sys.stdin.isatty():
        return
    if execute:
        raise InteractiveRequiredError(
            f"{context}: refusing to start — stdin is not a TTY but "
            "execute=True would drive the arm with no operator present to "
            "reset the workspace or score episodes. This is almost always a "
            "detached/non-interactive launch (nohup / setsid / a pipe) — "
            "never run a hardware eval that way. There is no software "
            "e-stop (Ctrl-C cannot reliably interrupt a blocked serial "
            "read/write); the only reliable stop is cutting power. "
            "--i-am-not-driving-an-arm does NOT apply here — it only "
            "permits a non-interactive DRY run (execute=False)."
        )
    if not allow_non_interactive:
        raise InteractiveRequiredError(
            f"{context}: stdin is not a TTY (non-interactive). No motor "
            "commands will be sent (execute=False), but the reset/verdict "
            "prompts still cannot be answered by a human. Pass "
            "allow_non_interactive=True (CLI: --i-am-not-driving-an-arm) to "
            "confirm this is an intentional unattended dry run (e.g. a "
            "smoke test)."
        )


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
