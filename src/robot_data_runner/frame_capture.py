"""frame_capture — per-episode diagnostic frame + manifest dump (C1 fix).

Problem this closes
--------------------
The runner's only capture was ``--actions-out``, a JSONL of actions with NO
images and NO state. If a closed-loop eval comes back 0/N, there was no
artifact to check for the three most likely causes: camera framing,
exposure/brightness drift, or a wrong video node.

``--save-frames DIR`` (wired in :mod:`episode_runner`) writes the FIRST
observation image of every episode as ``DIR/ep<NNN>_<camera>.png`` (or
``.npy``, see below) plus a running ``DIR/frames_manifest.json`` recording,
per captured frame: episode index, camera key, file, format, shape, dtype,
and **mean pixel value per channel**. The per-channel mean is the cheap
numeric signal that would have caught a "brightness 78 vs training 101"
exposure-drift regression — a diff you can `git diff`-style eyeball across
runs without opening a single image.

Backend choice
--------------
Preference order: **PIL (Pillow)** → **imageio** → raw ``.npy`` via
``numpy.save`` (always available — numpy is robot-data-runner's one hard
dependency). PIL is chosen first because it is typically already present
transitively in this workspace's training/eval environments (torch/
matplotlib pull it in) — but it is NOT a declared dependency of this
package, so it is soft-imported here exactly like ``lerobot``/``torch``
elsewhere in this package (see the module docstrings of ``runner.py`` /
``policy_loader.py``). No new hard dependency is added by this module;
if neither PIL nor imageio is importable, frames are still captured (as
``.npy``) and the manifest's ``"format"`` field says so explicitly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WriterFn = Callable[[Path, Any], None]


def _pick_backend() -> tuple[str, _WriterFn | None]:
    """Return ``(backend_name, writer)``. ``writer`` is ``None`` for the npy backend."""
    try:
        from PIL import Image

        def _write_pil(path: Path, arr: Any) -> None:
            Image.fromarray(arr).save(path)

        return "pil_png", _write_pil
    except ImportError:
        pass

    try:
        import imageio.v2 as imageio

        def _write_imageio(path: Path, arr: Any) -> None:
            imageio.imwrite(path, arr)

        return "imageio_png", _write_imageio
    except ImportError:
        pass

    return "npy", None


def save_episode_frame(
    obs: dict[str, Any],
    save_dir: Path,
    episode_idx: int,
) -> list[dict[str, Any]]:
    """Write one frame per camera key found in ``obs``; return manifest rows.

    A "camera key" is any key in ``obs`` that does NOT end in ``.pos``
    (matches the convention used throughout this package, e.g.
    :func:`robot_data_runner.mappers.obs_to_policy_input`). Rows are
    returned, not yet persisted — call :func:`write_manifest` (or merge
    into a running list and call it yourself) to write
    ``frames_manifest.json``.

    Never raises: this is a diagnostics feature and must not crash the eval
    it exists to help diagnose. Any per-camera failure is logged and
    skipped; if the preferred image backend fails to encode a given array
    (e.g. an unexpected dtype/shape), falls back to ``.npy`` for that frame.
    """
    import numpy as np

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    backend, writer = _pick_backend()

    rows: list[dict[str, Any]] = []
    camera_keys = [k for k in obs if not k.endswith(".pos")]
    for cam in camera_keys:
        try:
            arr = np.asarray(obs[cam])
        except Exception as exc:  # noqa: BLE001
            logger.warning("save-frames: could not array-ify obs[%r]: %s", cam, exc)
            continue
        if arr.ndim < 2:
            logger.warning(
                "save-frames: obs[%r] has unexpected shape %s — skipping", cam, arr.shape
            )
            continue

        stem = f"ep{episode_idx:03d}_{cam}"
        format_used = backend
        if backend == "npy" or writer is None:
            path = save_dir / f"{stem}.npy"
            np.save(path, arr)
            format_used = "npy"
        else:
            path = save_dir / f"{stem}.png"
            arr_u8 = arr if arr.dtype == np.uint8 else arr.astype(np.uint8)
            try:
                writer(path, arr_u8)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "save-frames: %s encode failed for %r (%s); falling back to .npy",
                    backend,
                    cam,
                    exc,
                )
                path = save_dir / f"{stem}.npy"
                np.save(path, arr)
                format_used = "npy"

        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            mean_per_channel = [float(np.mean(arr[..., c])) for c in range(arr.shape[-1])]
        else:
            mean_per_channel = [float(np.mean(arr))]

        rows.append(
            {
                "episode": episode_idx,
                "camera": cam,
                "file": path.name,
                "format": format_used,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "mean_per_channel": mean_per_channel,
            }
        )
    return rows


def write_manifest(save_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """(Re)write ``frames_manifest.json`` from the full accumulated ``rows`` list.

    Called after every captured frame (not just once at the end) so a run
    that dies mid-way still leaves a usable, complete-so-far manifest.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "frames_manifest.json"
    manifest_path.write_text(json.dumps({"frames": rows}, indent=2), encoding="utf-8")
    return manifest_path
