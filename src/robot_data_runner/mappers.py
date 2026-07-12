"""obs_to_policy_input / action_to_robot_dict — schema glue.

The two boundary conversions between the SO101Follower driver and a
lerobot policy. Pure functions; safe to import without torch / lerobot
at module load time.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Canonical SO-101 joint order (Feetech servo id 1..6). This is the order the
# training dataset recorded observation.state (positions) AND action in, and it
# equals list(robot.bus.motors) insertion order in lerobot's SO101Follower.
# Alphabetical sorted() differs and would permute the state — never use it.
SO101_CANONICAL_MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def obs_to_policy_input(
    obs: dict,
    device: str,
    task: str | None = None,
    motor_order: list[str] | None = None,
    state_dim: int | None = None,
) -> dict:
    """Convert SO101Follower.get_observation() output to a batched policy input.

    ``SO101Follower`` returns a flat dict::

        {"shoulder_pan.pos": float, ..., "<camera_name>": ndarray (H, W, 3) uint8}

    LeRobot policies want::

        {
            "observation.state":              torch.Tensor (1, D) float32,
            "observation.images.<camera>":    torch.Tensor (1, C, H, W) float32 ∈ [0, 1],
            "task":                            str (only when ``task`` is non-None),
        }

    The ``task`` string is the natural-language instruction (e.g.
    "pick and place cube") that VLA policies tokenise via the
    checkpoint's ``tokenizer_processor`` step. Required for SmolVLA
    (without it, ``select_action`` crashes with KeyError on
    ``observation.language.tokens``). Harmless for ACT / Diffusion —
    their preprocessors ignore the field.

    Parameters
    ----------
    motor_order:
        Canonical joint order to build ``observation.state`` in — pass
        ``list(robot.bus.motors)`` (``shoulder_pan, shoulder_lift, elbow_flex,
        wrist_flex, wrist_roll, gripper``, i.e. servo id 1..6). This MUST match
        the order the training dataset recorded state in. When ``None`` the
        code falls back to ``sorted()`` for backward-compat, but note that
        ``sorted()`` alphabetises to ``[elbow_flex, gripper, shoulder_lift,
        shoulder_pan, wrist_flex, wrist_roll]`` — a DIFFERENT order that would
        feed the policy a permuted state and produce wrong (unsafe) actions.
        Always pass ``motor_order`` for a real deploy.
    state_dim:
        The policy's expected ``observation.state`` dimensionality
        (``policy.config.input_features['observation.state'].shape[0]``). The
        SO101 follower exposes only joint positions (6). Training recorded
        state as ``cat(joint_pos[6], joint_vel[6])`` where ``joint_vel`` was a
        zero placeholder (the follower exposes no velocity). When the policy
        expects more dims than the live obs provides, the extra tail is
        right-padded with zeros — reproducing that zero-velocity block exactly.
        ``None`` disables padding (legacy raw behaviour).
    """
    import numpy as np
    import torch

    if motor_order is not None:
        motor_keys = [f"{m}.pos" for m in motor_order]
        missing = [k for k in motor_keys if k not in obs]
        if missing:
            raise KeyError(
                "obs is missing expected .pos keys for the requested motor_order: "
                f"{missing}. obs keys: {sorted(obs)!r}"
            )
    else:
        motor_keys = sorted(k for k in obs if k.endswith(".pos"))
        logger.warning(
            "obs_to_policy_input called WITHOUT motor_order — falling back to "
            "sorted(.pos) which ALPHABETISES joints and permutes observation.state "
            "vs training. Pass motor_order=list(robot.bus.motors) for any real run."
        )
    image_keys = [k for k in obs if not k.endswith(".pos")]

    state = np.array([float(obs[k]) for k in motor_keys], dtype=np.float32)
    if state_dim is not None:
        if state_dim < state.shape[0]:
            raise ValueError(
                f"policy expects observation.state dim {state_dim} but the robot "
                f"provides {state.shape[0]} joint positions; refusing to truncate "
                "(would silently drop joints)."
            )
        if state_dim > state.shape[0]:
            state = np.concatenate(
                [state, np.zeros(state_dim - state.shape[0], dtype=np.float32)]
            )
    out: dict[str, Any] = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
    }
    for k in image_keys:
        arr = np.asarray(obs[k])
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        tensor = (
            torch.from_numpy(arr).unsqueeze(0).to(device).float() / 255.0
        )
        out[f"observation.images.{k}"] = tensor
    if task is not None:
        out["task"] = task
    return out


def action_to_robot_dict(action: Any, motor_names: list[str]) -> dict[str, float]:
    """Convert (1, A) or (A,) policy output → SO101Follower.send_action kwargs."""
    import numpy as np

    arr = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[0] != len(motor_names):
        raise ValueError(
            f"Policy emitted {arr.shape[0]} action dims but robot has "
            f"{len(motor_names)} motors: {motor_names!r}"
        )
    return {f"{m}.pos": float(arr[i]) for i, m in enumerate(motor_names)}


def _train_state_std(ds_meta: Any) -> Any | None:
    """Best-effort read of the training observation.state per-dim std, or None."""
    stats = getattr(ds_meta, "stats", None)
    if not isinstance(stats, dict):
        return None
    st = stats.get("observation.state")
    if not isinstance(st, dict):
        return None
    return st.get("std")


def resolve_state_mapping(
    policy: Any,
    ds_meta: Any,
    motor_names: list[str],
    *,
    require_execute: bool = False,
) -> tuple[list[str], int | None]:
    """Resolve ``(motor_order, state_dim)`` for :func:`obs_to_policy_input`, safely.

    Centralises the state-layout safety checks so every real-hardware call site
    (``run_policy``, ``run_episodes``) shares one audited implementation:

    * **Order:** ``motor_names`` (``list(robot.bus.motors)``) must equal the
      known :data:`SO101_CANONICAL_MOTORS` order, else a future lerobot re-order
      would silently permute the state — raise rather than permute.
    * **Dim:** ``state_dim`` = the policy's expected ``observation.state`` dim
      (from ``policy.config.input_features``).
    * **Zero-pad safety:** the SO-101 exposes only 6 joint positions; training
      recorded ``cat(pos[6], vel[6]=zeros)``. If the policy wants more than 6
      dims, verify the training std of the pad region is ≈0 (i.e. it really was
      a constant zero placeholder) BEFORE letting :func:`obs_to_policy_input`
      zero-pad — otherwise refuse (those dims carry signal the live obs lacks).
    * **Execute gate:** if the dim can't be determined and ``require_execute``,
      refuse (don't fall through to a possibly-wrong 6-dim send).

    Returns ``(motor_names, state_dim)`` (``state_dim`` may be ``None`` only for
    dry-run when the dim is indeterminable).
    """
    import numpy as np

    if list(motor_names) != list(SO101_CANONICAL_MOTORS):
        raise ValueError(
            f"robot motor order {list(motor_names)!r} != expected SO-101 canonical "
            f"{list(SO101_CANONICAL_MOTORS)!r}; refusing to build observation.state "
            "(would permute vs training). Check the lerobot SO101Follower definition."
        )

    feats = getattr(getattr(policy, "config", None), "input_features", None)
    state_dim: int | None = None
    if feats and "observation.state" in feats:
        shp = feats["observation.state"].shape
        state_dim = int(shp[0]) if isinstance(shp, (tuple, list)) else int(shp)

    n_pos = len(motor_names)

    if state_dim is None:
        if require_execute:
            raise ValueError(
                "could not determine the policy's observation.state dim from "
                "input_features; refusing --execute (cannot verify state layout)."
            )
        return list(motor_names), None

    if state_dim < n_pos:
        raise ValueError(
            f"policy expects observation.state dim {state_dim} < {n_pos} live joints; "
            "refusing to truncate (would drop joints)."
        )

    if state_dim > n_pos:
        std = _train_state_std(ds_meta)
        if std is None:
            raise ValueError(
                f"policy expects observation.state dim {state_dim} > {n_pos} live "
                "joints, but no training stats are available to verify the extra dims "
                "are a zero placeholder. Pass --dataset-root so the zero-pad can be "
                "validated (refusing to pad blind)."
            )
        pad_std = np.asarray(std, dtype=float).reshape(-1)[n_pos:state_dim]
        if pad_std.size and float(np.max(np.abs(pad_std))) > 1e-3:
            raise ValueError(
                f"refusing to zero-pad observation.state dims [{n_pos}:{state_dim}]: "
                f"training std there = {pad_std.tolist()} (not ~0), so those dims carry "
                "real signal the live 6-DOF obs cannot supply. This checkpoint needs a "
                "real state provider, not zero-padding."
            )
        logger.info(
            "state pad verified: dims [%d:%d] had ~0 training std -> zero-padding is "
            "distribution-consistent (matches recorded zero-velocity block).",
            n_pos,
            state_dim,
        )

    return list(motor_names), state_dim
