from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from .models import FrameGroup, FrameRecord, RigSpec


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def detect_source_type(path: str | Path) -> str:
    source = Path(path).resolve()
    if source.is_file() and source.suffix.lower() == ".bag":
        return "ros1_bag"
    if source.is_dir() and (source / "metadata.yaml").is_file():
        return "ros2_bag"
    if source.is_dir():
        return "sequence"
    raise ValueError(f"Unsupported input source: {source}")


def decode_frame(frame: FrameRecord) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(frame.encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode {frame.camera_id} at {frame.timestamp_ns}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported decoded image shape: {image.shape}")


def _message_bytes(data: object) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    return np.asarray(data, dtype=np.uint8).tobytes()


def _header_timestamp_ns(message: object, fallback: int) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback)
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    timestamp = sec * 1_000_000_000 + nanosec
    return timestamp if timestamp > 0 else int(fallback)


def _encode_raw_image(message: object) -> tuple[bytes, str]:
    height = int(getattr(message, "height"))
    width = int(getattr(message, "width"))
    step = int(getattr(message, "step"))
    encoding = str(getattr(message, "encoding", "")).lower()
    raw = np.frombuffer(_message_bytes(getattr(message, "data")), dtype=np.uint8)
    if height <= 0 or width <= 0 or step <= 0 or raw.size < height * step:
        raise ValueError("Invalid sensor_msgs/Image dimensions")
    rows = raw[: height * step].reshape(height, step)
    if encoding in {"mono8", "8uc1"}:
        image = rows[:, :width]
    elif encoding in {"bgr8", "rgb8"}:
        image = rows[:, : width * 3].reshape(height, width, 3)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding in {"bgra8", "rgba8"}:
        image = rows[:, : width * 4].reshape(height, width, 4)
        conversion = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        image = cv2.cvtColor(image, conversion)
    else:
        raise ValueError(f"Unsupported sensor_msgs/Image encoding: {encoding}")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode ROS image as PNG")
    return encoded.tobytes(), "png"


def _read_timestamp_rows(camera_dir: Path) -> list[tuple[int, int, Path]]:
    timestamp_path = camera_dir / "timestamps.csv"
    images_dir = camera_dir / "images"
    if not timestamp_path.is_file() or not images_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and timestamps.csv under {camera_dir}")
    rows: list[tuple[int, int, Path]] = []
    with timestamp_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp_ns", "filename"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{timestamp_path} requires columns: timestamp_ns, filename")
        for fallback_index, item in enumerate(reader):
            source_index = int(item.get("frame_id") or fallback_index)
            timestamp_ns = int(item["timestamp_ns"])
            image_path = images_dir / item["filename"]
            if image_path.suffix.lower() not in _IMAGE_SUFFIXES or not image_path.is_file():
                raise FileNotFoundError(f"Image listed by timestamps.csv is missing: {image_path}")
            rows.append((source_index, timestamp_ns, image_path))
    if not rows:
        raise ValueError(f"No frames listed by {timestamp_path}")
    return rows


def read_sequence(path: str | Path, rig: RigSpec) -> dict[str, list[FrameRecord]]:
    root = Path(path).resolve()
    output: dict[str, list[FrameRecord]] = {}
    for camera in rig.cameras:
        records = []
        for source_index, timestamp_ns, image_path in _read_timestamp_rows(root / camera.directory):
            records.append(
                FrameRecord(
                    camera_id=camera.camera_id,
                    source_index=source_index,
                    timestamp_ns=timestamp_ns,
                    encoded=image_path.read_bytes(),
                    format=image_path.suffix.lower().lstrip("."),
                    source_path=image_path,
                )
            )
        output[camera.camera_id] = records
    return output


def read_bag(path: str | Path, rig: RigSpec) -> dict[str, list[FrameRecord]]:
    bag_path = Path(path).resolve()
    topic_to_camera = {camera.topic: camera.camera_id for camera in rig.cameras}
    output: dict[str, list[FrameRecord]] = {camera.camera_id: [] for camera in rig.cameras}
    with AnyReader([bag_path]) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topic_to_camera]
        available = {connection.topic for connection in connections}
        missing = sorted(set(topic_to_camera) - available)
        if missing:
            raise ValueError(f"Bag is missing configured topics: {', '.join(missing)}")
        for connection, bag_timestamp, raw in reader.messages(connections=connections):
            camera_id = topic_to_camera[connection.topic]
            message = reader.deserialize(raw, connection.msgtype)
            timestamp_ns = _header_timestamp_ns(message, bag_timestamp)
            if connection.msgtype.endswith("/CompressedImage"):
                encoded = _message_bytes(getattr(message, "data"))
                image_format = str(getattr(message, "format", "jpeg")) or "jpeg"
            elif connection.msgtype.endswith("/Image"):
                encoded, image_format = _encode_raw_image(message)
            else:
                raise ValueError(f"Configured topic is not an image topic: {connection.topic} ({connection.msgtype})")
            output[camera_id].append(
                FrameRecord(
                    camera_id=camera_id,
                    source_index=len(output[camera_id]),
                    timestamp_ns=timestamp_ns,
                    encoded=encoded,
                    format=image_format,
                )
            )
    if any(not records for records in output.values()):
        empty = [camera_id for camera_id, records in output.items() if not records]
        raise ValueError(f"No messages found for cameras: {', '.join(empty)}")
    return output


def synchronize(records_by_camera: dict[str, list[FrameRecord]], tolerance_ns: int) -> list[FrameGroup]:
    ordered = sorted(
        (record for records in records_by_camera.values() for record in records),
        key=lambda record: (record.timestamp_ns, record.camera_id),
    )
    open_groups: list[dict[str, FrameRecord]] = []
    finished: list[dict[str, FrameRecord]] = []
    for record in ordered:
        still_open = []
        for group in open_groups:
            earliest = min(item.timestamp_ns for item in group.values())
            if record.timestamp_ns - earliest > int(tolerance_ns):
                finished.append(group)
            else:
                still_open.append(group)
        open_groups = still_open

        candidates = []
        for group in open_groups:
            if record.camera_id in group:
                continue
            center = int(round(sum(item.timestamp_ns for item in group.values()) / len(group)))
            distance = abs(record.timestamp_ns - center)
            if distance <= int(tolerance_ns):
                candidates.append((distance, center, group))
        if candidates:
            min(candidates, key=lambda item: (item[0], item[1]))[2][record.camera_id] = record
        else:
            open_groups.append({record.camera_id: record})
    finished.extend(open_groups)
    finished.sort(key=lambda group: min(item.timestamp_ns for item in group.values()))

    groups = []
    for frame_id, group in enumerate(finished):
        timestamp_ns = int(round(sum(item.timestamp_ns for item in group.values()) / len(group)))
        groups.append(FrameGroup(frame_id=frame_id, timestamp_ns=timestamp_ns, frames=dict(group)))
    return groups


def load_frame_groups(path: str | Path, rig: RigSpec) -> list[FrameGroup]:
    source = Path(path).resolve()
    source_type = detect_source_type(source)
    if source_type == "ros1_bag":
        records = read_bag(source, rig)
    elif source_type == "ros2_bag":
        records = read_bag(source, rig)
    else:
        records = read_sequence(source, rig)
    return synchronize(records, rig.sync_tolerance_ns)


def all_records(groups: Iterable[FrameGroup]) -> list[FrameRecord]:
    return [frame for group in groups for frame in group.frames.values()]
