"""Tests for robot_data_runner.safety.require_interactive (S5 fix).

No hardware. Pure function against a monkeypatched sys.stdin.isatty().
"""

from __future__ import annotations

import sys

import pytest

from robot_data_runner.safety import InteractiveRequiredError, require_interactive


def test_tty_always_proceeds(monkeypatch):
    """A real TTY never refuses, regardless of execute or the opt-out."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    require_interactive("ctx", execute=True)
    require_interactive("ctx", execute=False)
    require_interactive("ctx", execute=False, allow_non_interactive=True)


def test_non_tty_execute_true_always_refuses(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(InteractiveRequiredError):
        require_interactive("ctx", execute=True)


def test_non_tty_execute_true_opt_out_does_not_rescue(monkeypatch):
    """The opt-out flag is documented as execute=False-only; execute=True
    must still refuse even when allow_non_interactive=True is passed."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(InteractiveRequiredError):
        require_interactive("ctx", execute=True, allow_non_interactive=True)


def test_non_tty_dry_run_refuses_by_default(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(InteractiveRequiredError):
        require_interactive("ctx", execute=False)


def test_non_tty_dry_run_proceeds_with_opt_out(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    require_interactive("ctx", execute=False, allow_non_interactive=True)
