from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from .models import FrameGroup, RigSpec, SelectedGroup


def _encode_grayscale_png(encoded: bytes, camera_id: str, timestamp_ns: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to decode {camera_id} at {timestamp_ns}")
    ok, payload = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode grayscale {camera_id} at {timestamp_ns}")
    return payload


def write_clean_ros1_bag(
    path: str | Path,
    *,
    groups: Sequence[FrameGroup],
    selected: Sequence[SelectedGroup],
    rig: RigSpec,
    overwrite: bool = False,
) -> Path:
    output = Path(path).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"bag exists: {output}")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    group_by_id = {group.frame_id: group for group in groups}

    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]
    sequence_numbers = {camera.camera_id: 0 for camera in rig.cameras}

    with Writer(output) as writer:
        connections = {
            camera.camera_id: writer.add_connection(
                camera.topic,
                "sensor_msgs/msg/CompressedImage",
                typestore=typestore,
            )
            for camera in rig.cameras
        }
        for selected_group in sorted(selected, key=lambda item: (item.timestamp_ns, item.active_cams)):
            source_group = group_by_id[selected_group.frame_id]
            for camera_id in selected_group.active_cams:
                frame = source_group.frames.get(camera_id)
                if frame is None:
                    raise ValueError(f"Selected frame {selected_group.frame_id} has no data for {camera_id}")
                camera = rig.camera(camera_id)
                timestamp = int(frame.timestamp_ns)
                stamp = Time(sec=timestamp // 1_000_000_000, nanosec=timestamp % 1_000_000_000)
                grayscale = _encode_grayscale_png(frame.encoded, camera_id, timestamp)
                message = CompressedImage(
                    header=Header(
                        seq=sequence_numbers[camera_id],
                        stamp=stamp,
                        frame_id=camera.frame_id,
                    ),
                    format="png",
                    data=grayscale,
                )
                writer.write(
                    connections[camera_id],
                    timestamp,
                    typestore.serialize_ros1(message, "sensor_msgs/msg/CompressedImage"),
                )
                sequence_numbers[camera_id] += 1
    return output


def write_full_ros1_bag(
    path: str | Path,
    *,
    groups: Sequence[FrameGroup],
    rig: RigSpec,
    overwrite: bool = False,
) -> Path:
    selected = [
        SelectedGroup(
            frame_id=group.frame_id,
            timestamp_ns=group.timestamp_ns,
            active_cams=tuple(sorted(group.frames)),
            observations={},
        )
        for group in groups
    ]
    return write_clean_ros1_bag(path, groups=groups, selected=selected, rig=rig, overwrite=overwrite)
