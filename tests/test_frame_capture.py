"""Tests for robot_data_runner.frame_capture (C1 fix).

No hardware — a synthetic obs dict with a known, controlled pixel pattern
stands in for a real camera frame. Verifies file naming, manifest content,
and (most importantly) that the mean-pixel-value-per-channel signal is
correct — this is the cheap number that would have caught a prior
brightness-drift regression (session note: "78 vs training 101").
"""

from __future__ import annotations

import json

import numpy as np

from robot_data_runner.frame_capture import save_episode_frame, write_manifest


def _constant_image(r: int, g: int, b: int, h: int = 4, w: int = 4) -> np.ndarray:
    """(H, W, 3) uint8 image with an exact, known mean per channel."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    return arr


def test_save_episode_frame_writes_expected_filename_and_manifest_row(tmp_path):
    obs = {
        "shoulder_pan.pos": 0.0,  # not a camera key — must be skipped
        "d435_rgb": _constant_image(78, 120, 60),
    }
    rows = save_episode_frame(obs, tmp_path, episode_idx=3)

    assert len(rows) == 1
    row = rows[0]
    assert row["episode"] == 3
    assert row["camera"] == "d435_rgb"
    assert row["shape"] == [4, 4, 3]
    assert row["dtype"] == "uint8"
    # exact per-channel means for a constant-value image
    assert row["mean_per_channel"] == [78.0, 120.0, 60.0]

    # the file the row references must actually exist on disk
    written = tmp_path / row["file"]
    assert written.exists()
    assert row["file"].startswith("ep003_d435_rgb")
    # format is one of the documented backends — never silently something else
    assert row["format"] in ("pil_png", "imageio_png", "npy")


def test_save_episode_frame_skips_non_camera_and_malformed_keys(tmp_path):
    obs = {
        "shoulder_pan.pos": 1.0,
        "gripper.pos": 2.0,
    }
    rows = save_episode_frame(obs, tmp_path, episode_idx=0)
    assert rows == []  # no camera keys present -> nothing captured, no crash


def test_save_episode_frame_multi_camera(tmp_path):
    obs = {
        "cam_a": _constant_image(10, 20, 30),
        "cam_b": _constant_image(200, 150, 100),
    }
    rows = save_episode_frame(obs, tmp_path, episode_idx=1)
    by_cam = {r["camera"]: r for r in rows}
    assert set(by_cam) == {"cam_a", "cam_b"}
    assert by_cam["cam_a"]["mean_per_channel"] == [10.0, 20.0, 30.0]
    assert by_cam["cam_b"]["mean_per_channel"] == [200.0, 150.0, 100.0]


def test_write_manifest_persists_all_rows(tmp_path):
    obs1 = {"cam": _constant_image(1, 2, 3)}
    obs2 = {"cam": _constant_image(4, 5, 6)}
    rows = save_episode_frame(obs1, tmp_path, episode_idx=0)
    rows += save_episode_frame(obs2, tmp_path, episode_idx=1)

    manifest_path = write_manifest(tmp_path, rows)
    assert manifest_path == tmp_path / "frames_manifest.json"
    assert manifest_path.exists()

    data = json.loads(manifest_path.read_text())
    assert len(data["frames"]) == 2
    assert data["frames"][0]["episode"] == 0
    assert data["frames"][1]["episode"] == 1
    assert data["frames"][0]["mean_per_channel"] == [1.0, 2.0, 3.0]
    assert data["frames"][1]["mean_per_channel"] == [4.0, 5.0, 6.0]


def test_brightness_drift_is_visible_in_mean_per_channel(tmp_path):
    """Directly demonstrates the motivating scenario: two runs whose only
    difference is exposure/brightness produce a numerically different
    mean_per_channel — no image viewer required to catch it."""
    training_like = _constant_image(101, 101, 101)
    drifted = _constant_image(78, 78, 78)

    rows_train = save_episode_frame({"cam": training_like}, tmp_path / "a", episode_idx=0)
    rows_drift = save_episode_frame({"cam": drifted}, tmp_path / "b", episode_idx=0)

    assert rows_train[0]["mean_per_channel"] == [101.0, 101.0, 101.0]
    assert rows_drift[0]["mean_per_channel"] == [78.0, 78.0, 78.0]
    assert rows_train[0]["mean_per_channel"] != rows_drift[0]["mean_per_channel"]
