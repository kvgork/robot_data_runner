"""obs_to_policy_input / action_to_robot_dict — schema glue.

The two boundary conversions between the SO101Follower driver and a
lerobot policy. Pure functions; safe to import without torch / lerobot
at module load time.
"""

from __future__ import annotations

from typing import Any


def obs_to_policy_input(obs: dict, device: str, task: str | None = None) -> dict:
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
    """
    import numpy as np
    import torch

    motor_keys = sorted(k for k in obs if k.endswith(".pos"))
    image_keys = [k for k in obs if not k.endswith(".pos")]

    state = np.array([float(obs[k]) for k in motor_keys], dtype=np.float32)
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
