from __future__ import annotations

from pathlib import Path

import numpy as np

from .cache import read_cache
from .models import RigSpec


def _topic_key(topic: str) -> str:
    return topic.replace("/", "_").strip("_")


def export_kalibr_npz(cache_path: str | Path, rig: RigSpec, output_path: str | Path) -> Path:
    _, records = read_cache(cache_path)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for camera in rig.cameras:
        rows = sorted(
            (row for row in records if row.camera_id == camera.camera_id),
            key=lambda row: (row.timestamp_ns, row.source_index),
        )
        key = _topic_key(camera.topic)
        timestamps = np.asarray([row.timestamp_ns for row in rows], dtype=np.int64)
        if rows:
            points = np.stack([np.asarray(row.points_xy, dtype=np.float32).reshape(-1, 4, 2) for row in rows])
            confidence = np.stack([np.asarray(row.confidence, dtype=np.float32).reshape(-1, 4) for row in rows])
            # NN-Detector uses BR, BL, TL, TR. The Kalibr adapter consumes TR, TL, BL, BR.
            points = points[:, :, ::-1, :]
            confidence = confidence[:, :, ::-1]
        else:
            points = np.zeros((0, 36, 4, 2), dtype=np.float32)
            confidence = np.zeros((0, 36, 4), dtype=np.float32)
        arrays[f"{key}__timestamps_ns"] = timestamps
        arrays[f"{key}__xy_px"] = points.astype(np.float32, copy=False)
        arrays[f"{key}__conf"] = confidence.astype(np.float32, copy=False)
    np.savez_compressed(output, **arrays)
    return output
