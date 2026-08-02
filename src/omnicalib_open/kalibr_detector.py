from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from .bag_writer import write_full_ros1_bag
from .cache import write_cache
from .models import DetectionRecord, FrameGroup, RigSpec
from .rig import require_kalibr_models
from .sources import all_records, decode_frame


def _docker_mount(source: Path, target: str, *, read_only: bool = False) -> str:
    suffix = ":ro" if read_only else ""
    return f"{source.resolve()}:{target}{suffix}"


def extract_acv_detection_cache(
    *,
    groups: Sequence[FrameGroup],
    rig: RigSpec,
    rig_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    confidence_threshold: float,
    docker_image: str,
    num_processes: int = 0,
    overwrite: bool = False,
) -> Path:
    require_kalibr_models(rig)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bag_path = output / "kalibr_detector_input.bag"
    raw_path = output / "kalibr_raw_detections.npz"
    report_path = output / "kalibr_raw_detections.json"
    cache_path = output / "detections_full.detcache"
    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"cache exists: {cache_path}")
    write_full_ros1_bag(bag_path, groups=groups, rig=rig, overwrite=overwrite)

    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        "ROS_HOME=/tmp/.ros",
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "-v",
        _docker_mount(bag_path, "/input/images.bag", read_only=True),
        "-v",
        _docker_mount(Path(rig_path), "/input/rig.yaml", read_only=True),
        "-v",
        _docker_mount(Path(target_path), "/input/target.yaml", read_only=True),
        "-v",
        _docker_mount(output, "/output"),
        str(docker_image),
        "omnicalib-acv-extract",
        "--bag",
        "/input/images.bag",
        "--rig",
        "/input/rig.yaml",
        "--target",
        "/input/target.yaml",
        "--output",
        "/output/kalibr_raw_detections.npz",
        "--report",
        "/output/kalibr_raw_detections.json",
        "--num-processes",
        str(max(0, int(num_processes))),
    ]
    subprocess.run(command, check=True)

    with np.load(raw_path, allow_pickle=False) as arrays:
        raw_rows = {
            (str(arrays["camera_id"][index]), int(arrays["timestamp_ns"][index])): (
                np.asarray(arrays["xy_px"][index], dtype=np.float32).copy(),
                np.asarray(arrays["confidence"][index], dtype=np.float32).copy(),
            )
            for index in range(len(arrays["camera_id"]))
        }

    records: list[DetectionRecord] = []
    matched = 0
    for frame in sorted(all_records(groups), key=lambda item: (item.timestamp_ns, item.camera_id)):
        image = decode_frame(frame)
        points, confidence = raw_rows.get(
            (frame.camera_id, int(frame.timestamp_ns)),
            (
                np.full((144, 2), np.nan, dtype=np.float32),
                np.zeros(144, dtype=np.float32),
            ),
        )
        if np.count_nonzero(confidence > 0.0) > 0:
            matched += 1
        records.append(
            DetectionRecord(
                camera_id=frame.camera_id,
                source_index=frame.source_index,
                timestamp_ns=frame.timestamp_ns,
                image_height=int(image.shape[0]),
                image_width=int(image.shape[1]),
                points_xy=points,
                confidence=confidence,
            )
        )
    if matched == 0:
        raise RuntimeError("ACV-Detector returned zero observations across all cameras")

    cache = write_cache(
        cache_path,
        records,
        model_path=None,
        confidence_threshold=confidence_threshold,
        camera_ids=rig.camera_ids,
        detector="acv",
        overwrite=overwrite,
    )
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report.update({"matched_source_frames": matched, "cache": str(cache)})
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache
