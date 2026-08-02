#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, "/opt/omnicalib/src")

import aslam_cv_backend as acvb
import kalibr_camera_calibration as kcc
import kalibr_common as kc

from omnicalib_open.corner_order import kalibr_target_to_canonical
from omnicalib_open.rig import load_rig, require_kalibr_models


CAMERA_MODELS = {
    "pinhole-radtan": acvb.DistortedPinhole,
    "pinhole-equi": acvb.EquidistantPinhole,
    "pinhole-fov": acvb.FovPinhole,
    "omni-none": acvb.Omni,
    "omni-radtan": acvb.DistortedOmni,
    "eucm-none": acvb.ExtendedUnified,
    "ds-none": acvb.DoubleSphere,
}


class _GrayscaleDataset:
    def __init__(self, dataset: object):
        self._dataset = dataset
        self.topic = dataset.topic

    def numImages(self) -> int:
        return int(self._dataset.numImages())

    def readDataset(self):
        for timestamp, image in self._dataset.readDataset():
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[2] == 3:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            elif array.ndim == 3 and array.shape[2] == 4:
                array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
            if array.ndim != 2:
                raise ValueError(f"Kalibr detector expects a grayscale image, got {array.shape}")
            yield timestamp, np.ascontiguousarray(array)


def _timestamp_to_ns(stamp: object) -> int:
    for name in ("to_nsec", "toNSec", "getNSec"):
        method = getattr(stamp, name, None)
        if method is not None:
            return int(method())
    method = getattr(stamp, "toSec", None)
    if method is not None:
        return int(round(float(method()) * 1_000_000_000.0))
    return int(getattr(stamp, "secs", 0)) * 1_000_000_000 + int(getattr(stamp, "nsecs", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract ACV-Detector AprilGrid detections")
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--rig", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--num-processes", type=int, default=0)
    args = parser.parse_args()

    if args.num_processes > 0:
        os.environ["OMNICALLIB_TARGET_EXTRACT_PROCESSES"] = str(int(args.num_processes))
    rig = load_rig(args.rig)
    require_kalibr_models(rig)
    target_config = kc.CalibrationTargetParameters(str(args.target.resolve()))
    camera_ids: list[str] = []
    timestamps: list[int] = []
    all_points: list[np.ndarray] = []
    all_confidence: list[np.ndarray] = []
    per_camera: dict[str, dict[str, int]] = {}

    for camera in rig.cameras:
        dataset = _GrayscaleDataset(kc.BagImageDatasetReader(str(args.bag.resolve()), camera.topic))
        geometry = kcc.CameraGeometry(CAMERA_MODELS[camera.model], target_config, dataset, verbose=False)
        target = geometry.ctarget.detector.target()
        grid_rows = int(target.rows())
        grid_cols = int(target.cols())
        if grid_rows != 12 or grid_cols != 12:
            raise ValueError("Datawash currently supports a 6x6 AprilGrid target")
        observations = kc.extractCornersFromDataset(
            dataset,
            geometry.ctarget.detector,
            multithreading=True,
            numProcesses=max(1, int(args.num_processes)) if args.num_processes > 0 else None,
            clearImages=True,
            noTransformation=True,
        )
        for observation in observations:
            points, confidence = kalibr_target_to_canonical(
                observation.getCornersIdx(),
                observation.getCornersImageFrame(),
                grid_rows=grid_rows,
                grid_cols=grid_cols,
            )
            camera_ids.append(camera.camera_id)
            timestamps.append(_timestamp_to_ns(observation.time()))
            all_points.append(points)
            all_confidence.append(confidence)
        per_camera[camera.camera_id] = {
            "images": int(dataset.numImages()),
            "observations": int(len(observations)),
        }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        camera_id=np.asarray(camera_ids, dtype=np.str_),
        timestamp_ns=np.asarray(timestamps, dtype=np.int64),
        xy_px=np.stack(all_points) if all_points else np.zeros((0, 144, 2), dtype=np.float32),
        confidence=np.stack(all_confidence) if all_confidence else np.zeros((0, 144), dtype=np.float32),
    )
    report = {
        "detector": "acv",
        "bag": str(args.bag.resolve()),
        "target": str(args.target.resolve()),
        "per_camera": per_camera,
        "observations": len(timestamps),
    }
    args.report.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not timestamps:
        raise RuntimeError("ACV-Detector returned zero observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
