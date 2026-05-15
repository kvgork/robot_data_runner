"""task_specs — pluggable success / termination criteria for closed-loop eval.

Two concrete specs ship in this milestone:

* :class:`PromptUserObserverSpec` — never auto-terminates inside the step
  loop except on ``timeout_s``. After each episode the runner asks
  ``"success? (y/n/abort)"`` on stdin. Works on day one with no extra
  hardware.
* :class:`GripperAtTargetPoseSpec` — terminates early when the
  end-effector reaches a target joint pose within tolerance AND the
  gripper is closed. Reads joint state from the observation dict; no
  extra camera needed.

A third (``camera_object_detection``) is reserved for a future commit.

All specs satisfy a tiny protocol; downstream code (``episode_runner``)
only needs ``start_episode`` / ``is_done`` / ``on_episode_end``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TaskSpec(Protocol):
    """Per-episode success/termination contract."""

    type: str

    def start_episode(self, robot: Any) -> None: ...
    def is_done(self, t: float, obs: dict) -> tuple[bool, bool]:
        """Return ``(done, success)``. Called every step at ``rate_hz``.

        - ``done`` flag triggers a clean episode end.
        - ``success`` is recorded only when ``done`` is True; ignored otherwise.
        """

    def on_episode_end(
        self, idx: int, success: bool, ep_len_s: float
    ) -> bool:
        """Finalise scoring after the step loop exits.

        For manual-observer specs this is where stdin prompting happens.
        Returns the final success flag (may differ from the in-loop value
        if the observer overrides).
        """


# ---------------------------------------------------------------------------
# PromptUserObserverSpec
# ---------------------------------------------------------------------------

@dataclass
class PromptUserObserverSpec:
    """Step loop just times out; observer scores y/n on stdin."""

    type: str = "prompt_user_observer"
    timeout_s: float = 10.0
    prompt_text: str = "Episode {idx}: success? (y/n/abort) "

    def start_episode(self, robot: Any) -> None:  # noqa: ARG002
        return None

    def is_done(self, t: float, obs: dict) -> tuple[bool, bool]:  # noqa: ARG002
        if t >= self.timeout_s:
            return True, False
        return False, False

    def on_episode_end(self, idx: int, success: bool, ep_len_s: float) -> bool:  # noqa: ARG002
        text = self.prompt_text.format(idx=idx)
        try:
            ans = input(text).strip().lower()
        except EOFError:
            ans = "n"
        if ans == "abort":
            raise KeyboardInterrupt("operator abort")
        return ans.startswith("y")


# ---------------------------------------------------------------------------
# GripperAtTargetPoseSpec
# ---------------------------------------------------------------------------

# Canonical SO-101 joint order. Matches SO101_JOINT_NAMES in lerobot_isaac_env
# and the dict-key order the SO101Follower bus dispatches actions in.
SO101_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class GripperAtTargetPoseSpec:
    """Success when joint state matches target within tolerance.

    No forward kinematics here — uses joint-space distance because that
    avoids a dependency on the URDF/FK at runtime. The target is a 6-dim
    joint position vector in the **SO101_JOINT_ORDER** (NOT alphabetical):
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper].
    Units match what the bus reports (deg if ``use_degrees``, normalized
    [-100, 100] otherwise).
    """

    target_joint_pos: list[float]
    tolerance_per_joint: float = 3.0
    require_gripper_closed: bool = True
    gripper_closed_value: float = 50.0
    type: str = "gripper_at_target_pose"
    timeout_s: float = 10.0
    joint_order: tuple[str, ...] = field(default_factory=lambda: SO101_JOINT_ORDER)

    def start_episode(self, robot: Any) -> None:  # noqa: ARG002
        return None

    def is_done(self, t: float, obs: dict) -> tuple[bool, bool]:
        if t >= self.timeout_s:
            return True, False
        if len(self.joint_order) != len(self.target_joint_pos):
            return False, False
        for name, tgt in zip(self.joint_order, self.target_joint_pos):
            key = f"{name}.pos"
            if key not in obs:
                return False, False
            if abs(float(obs[key]) - tgt) > self.tolerance_per_joint:
                return False, False
        if self.require_gripper_closed:
            grip = float(obs.get("gripper.pos", 0.0))
            if grip < self.gripper_closed_value:
                return False, False
        return True, True

    def on_episode_end(self, idx: int, success: bool, ep_len_s: float) -> bool:  # noqa: ARG002
        return success


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "prompt_user_observer": PromptUserObserverSpec,
    "gripper_at_target_pose": GripperAtTargetPoseSpec,
}


def make_task_spec(type_: str, **kwargs: Any) -> TaskSpec:
    """Build a task spec by type name. Raises KeyError on unknown type."""
    if type_ not in _REGISTRY:
        raise KeyError(
            f"unknown task spec type {type_!r}; available: {list(_REGISTRY)}"
        )
    return _REGISTRY[type_](**kwargs)
