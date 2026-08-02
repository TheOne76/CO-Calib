from __future__ import annotations

from pathlib import Path

import yaml

from .models import CameraSpec, RigSpec


SUPPORTED_KALIBR_MODELS = frozenset(
    {
        "pinhole-radtan",
        "pinhole-equi",
        "pinhole-fov",
        "omni-none",
        "omni-radtan",
        "eucm-none",
        "ds-none",
    }
)


def load_rig(path: str | Path) -> RigSpec:
    rig_path = Path(path).resolve()
    raw = yaml.safe_load(rig_path.read_text(encoding="utf-8")) or {}
    cameras_raw = raw.get("cameras")
    if not isinstance(cameras_raw, list) or not cameras_raw:
        raise ValueError(f"rig must contain a non-empty cameras list: {rig_path}")

    cameras: list[CameraSpec] = []
    for index, item in enumerate(cameras_raw):
        if not isinstance(item, dict):
            raise ValueError(f"cameras[{index}] must be a mapping")
        camera_id = str(item.get("id", "")).strip()
        topic = str(item.get("topic", "")).strip()
        model = str(item.get("model", "")).strip()
        if not camera_id or not topic:
            raise ValueError(f"cameras[{index}] requires id and topic")
        if model and model not in SUPPORTED_KALIBR_MODELS:
            raise ValueError(
                f"cameras[{index}].model must be one of: {', '.join(sorted(SUPPORTED_KALIBR_MODELS))}"
            )
        cameras.append(
            CameraSpec(
                camera_id=camera_id,
                topic=topic,
                directory=str(item.get("directory", camera_id)).strip(),
                frame_id=str(item.get("frame_id", camera_id)).strip(),
                model=model,
            )
        )

    camera_ids = [camera.camera_id for camera in cameras]
    topics = [camera.topic for camera in cameras]
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("camera ids must be unique")
    if len(set(topics)) != len(topics):
        raise ValueError("camera topics must be unique")

    tolerance_ms = float(raw.get("sync_tolerance_ms", 10.0))
    if tolerance_ms < 0.0:
        raise ValueError("sync_tolerance_ms must be non-negative")
    return RigSpec(cameras=tuple(cameras), sync_tolerance_ns=int(round(tolerance_ms * 1_000_000.0)))


def require_kalibr_models(rig: RigSpec) -> tuple[str, ...]:
    missing = [camera.camera_id for camera in rig.cameras if not camera.model]
    if missing:
        raise ValueError(f"rig cameras require model for calibration: {', '.join(missing)}")
    return rig.kalibr_models
