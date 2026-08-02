from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    topic: str
    directory: str
    frame_id: str
    model: str = ""


@dataclass(frozen=True)
class RigSpec:
    cameras: tuple[CameraSpec, ...]
    sync_tolerance_ns: int

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(camera.camera_id for camera in self.cameras)

    def camera(self, camera_id: str) -> CameraSpec:
        return next(camera for camera in self.cameras if camera.camera_id == camera_id)

    @property
    def kalibr_models(self) -> tuple[str, ...]:
        return tuple(camera.model for camera in self.cameras)


@dataclass(frozen=True)
class FrameRecord:
    camera_id: str
    source_index: int
    timestamp_ns: int
    encoded: bytes
    format: str
    source_path: Path | None = None


@dataclass(frozen=True)
class FrameGroup:
    frame_id: int
    timestamp_ns: int
    frames: Mapping[str, FrameRecord]


@dataclass(frozen=True)
class DetectorOutput:
    points_xy: object
    confidence: object


@dataclass(frozen=True)
class DetectionRecord:
    camera_id: str
    source_index: int
    timestamp_ns: int
    image_height: int
    image_width: int
    points_xy: object
    confidence: object


@dataclass(frozen=True)
class ResizePlan:
    original_height: int
    original_width: int
    infer_height: int
    infer_width: int
    scale_factor: float
    scale_x: float
    scale_y: float


@dataclass(frozen=True)
class CameraMask:
    camera_id: str
    image_height: int
    image_width: int
    grid_rows: int
    grid_cols: int
    valid_mask: int
    center_mask: int

    @property
    def n_bits(self) -> int:
        return int(self.grid_rows) * int(self.grid_cols)


@dataclass(frozen=True)
class CameraObservation:
    camera_id: str
    image_path: Path | None
    timestamp_ns: int
    resize: ResizePlan
    mask: CameraMask
    valid_points_xy: object
    coverage_mask: int
    center_hit: bool
    usable: bool
    detected_points: int
    center_points: int = 0
    center_points_fraction: float = 0.0
    center_area_fraction: float = 0.0
    radial_span: float = 0.0
    valid_corner_mask: int = 0
    valid_tag_mask: int = 0
    projective_quality: float = 1.0
    projective_local_shape: float = 1.0
    projective_area_uniformity: float = 1.0
    projective_jacobian_isotropy: float = 1.0
    projective_area_gradient: float = 1.0
    projective_homography_rms_px: float = 0.0
    projective_valid_tags: int = 0
    diagonal_ratio: float = 1.0
    diagonal_valid_tags: int = 0
    board_center_x_norm: float = 0.0
    board_center_y_norm: float = 0.0
    board_center_dist_norm: float = 1.0
    board_diagonal_norm: float = 0.0


@dataclass(frozen=True)
class CandidateGroup:
    frame_id: int
    timestamp_ns: int
    observations: Mapping[str, CameraObservation]


@dataclass(frozen=True)
class SelectedGroup:
    frame_id: int
    timestamp_ns: int
    active_cams: tuple[str, ...]
    observations: Mapping[str, CameraObservation]


@dataclass(frozen=True)
class StageSelectionConfig:
    radial_span: float
    iso: float
    budget: int


@dataclass(frozen=True)
class SelectionConfig:
    anchor: StageSelectionConfig = StageSelectionConfig(radial_span=0.25, iso=0.50, budget=0)
    covisible: StageSelectionConfig = StageSelectionConfig(radial_span=0.00, iso=0.30, budget=100)
    mono_fill: StageSelectionConfig = StageSelectionConfig(radial_span=0.20, iso=0.30, budget=0)


@dataclass(frozen=True)
class DatawashConfig:
    detector_confidence: float = 0.99
    sample_min_detection_points: int = 12
    selection: SelectionConfig = SelectionConfig()
