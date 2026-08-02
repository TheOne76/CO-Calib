from __future__ import annotations

import math

import cv2
import numpy as np

from .models import CameraMask, CameraObservation, ResizePlan

APRILGRID_TAG_ROWS = 6
APRILGRID_TAG_COLS = 6
APRILGRID_TAG_SPACING = 0.3

_CORNER_PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (0, 3, 2, 1),
    (3, 2, 1, 0),
    (2, 1, 0, 3),
    (1, 0, 3, 2),
)


def _aprilgrid_object_corners() -> np.ndarray:
    pitch = 1.0 + float(APRILGRID_TAG_SPACING)
    tags: list[list[tuple[float, float]]] = []
    for row in range(APRILGRID_TAG_ROWS):
        for col in range(APRILGRID_TAG_COLS):
            x0 = float(col) * pitch
            y0 = float(row) * pitch
            tags.append(
                [
                    (x0, y0),
                    (x0 + 1.0, y0),
                    (x0 + 1.0, y0 + 1.0),
                    (x0, y0 + 1.0),
                ]
            )
    return np.asarray(tags, dtype=np.float32)


_APRILGRID_OBJECT_CORNERS = _aprilgrid_object_corners()


class RadialSpanNormalizer:
    """Measure directed center-to-edge board coverage in full-diameter units."""

    def __init__(self, valid_mask: np.ndarray):
        self.valid_mask = np.asarray(valid_mask, dtype=bool)
        if self.valid_mask.ndim != 2 or not np.any(self.valid_mask):
            raise ValueError("radial-span valid mask must be a non-empty 2-D array")
        self.height, self.width = self.valid_mask.shape
        self.cx = 0.5 * float(self.width)
        self.cy = 0.5 * float(self.height)
        yy, xx = np.nonzero(self.valid_mask)
        dx = xx.astype(np.float64) - self.cx
        dy = yy.astype(np.float64) - self.cy
        self.radius = max(float(np.max(np.sqrt(dx * dx + dy * dy))), 1e-6)

    @classmethod
    def from_image(cls, image_bgr: np.ndarray, *, gray_threshold: int) -> "RadialSpanNormalizer":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cls(gray > int(gray_threshold))

    def valid_points(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        if points.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        x = np.rint(points[:, 0]).astype(np.int64)
        y = np.rint(points[:, 1]).astype(np.int64)
        inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        valid = np.zeros_like(inside, dtype=bool)
        valid[inside] = self.valid_mask[y[inside], x[inside]]
        return points[valid]

    def radial_span(self, points_xy: np.ndarray) -> float:
        points = self.valid_points(points_xy)
        if points.shape[0] < 2:
            return 0.0
        vectors = points - np.asarray([[self.cx, self.cy]], dtype=np.float64)
        center_vector = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(center_vector))
        if norm > 1e-6 * self.radius:
            axis = center_vector / norm
        else:
            centered = vectors - np.mean(vectors, axis=0, keepdims=True)
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                axis = vh[0]
            except np.linalg.LinAlgError:
                axis = np.asarray([1.0, 0.0], dtype=np.float64)
        projected = (vectors @ axis) / self.radius
        return float(np.clip(0.5 * (np.max(projected) - np.min(projected)), 0.0, 1.0))

    def center_area_fraction(self, points_xy: np.ndarray, *, radius_fraction: float = 0.20) -> float:
        points = self.valid_points(points_xy)
        if points.size == 0:
            return 0.0
        delta = points - np.asarray([[self.cx, self.cy]], dtype=np.float64)
        radius = np.sqrt(np.sum(delta * delta, axis=1)) / self.radius
        return float(np.mean(radius <= float(radius_fraction)))


def valid_target_masks(points_xy: np.ndarray, confidence: np.ndarray, *, conf_thr: float) -> tuple[int, int]:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    count = min(points.shape[0], confidence.shape[0])
    valid = np.isfinite(points[:count]).all(axis=1) & np.isfinite(confidence[:count])
    valid &= confidence[:count] >= float(conf_thr)
    corner_mask = 0
    for index in np.flatnonzero(valid):
        corner_mask |= 1 << int(index)
    tag_mask = 0
    for tag_index in range(count // 4):
        if ((corner_mask >> (4 * tag_index)) & 0xF) == 0xF:
            tag_mask |= 1 << tag_index
    return int(corner_mask), int(tag_mask)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _quad_area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(points, dtype=np.float32).reshape(-1, 2))))


def _tag_shape_quality(corners_xy: np.ndarray) -> float:
    corners = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
    edge_lengths = [float(np.linalg.norm(corners[(idx + 1) % 4] - corners[idx])) for idx in range(4)]
    if min(edge_lengths) <= 1e-6:
        return 0.0
    aspect = min(edge_lengths) / max(edge_lengths)
    angle_sines = []
    for idx in range(4):
        a = corners[(idx - 1) % 4] - corners[idx]
        b = corners[(idx + 1) % 4] - corners[idx]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        angle_sines.append(abs(_cross2(a, b)) / denom if denom > 1e-9 else 0.0)
    return float(np.clip(aspect * min(angle_sines), 0.0, 1.0))


def _homography_jacobian_stats(homography: np.ndarray, object_corners: np.ndarray) -> tuple[float, float]:
    homography = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    object_points = np.asarray(object_corners, dtype=np.float64).reshape(-1, 2)
    if object_points.size == 0:
        return 0.0, 0.0
    step = max(1, int(len(object_points) // 40))
    isotropy_values: list[float] = []
    det_values: list[float] = []
    for u, v in object_points[::step]:
        a = homography[0, 0] * u + homography[0, 1] * v + homography[0, 2]
        b = homography[1, 0] * u + homography[1, 1] * v + homography[1, 2]
        w = homography[2, 0] * u + homography[2, 1] * v + homography[2, 2]
        if abs(w) <= 1e-12:
            continue
        jacobian = np.asarray(
            [
                [
                    (homography[0, 0] * w - a * homography[2, 0]) / (w * w),
                    (homography[0, 1] * w - a * homography[2, 1]) / (w * w),
                ],
                [
                    (homography[1, 0] * w - b * homography[2, 0]) / (w * w),
                    (homography[1, 1] * w - b * homography[2, 1]) / (w * w),
                ],
            ],
            dtype=np.float64,
        )
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if float(singular_values[0]) <= 1e-9:
            continue
        isotropy_values.append(float(singular_values[-1] / singular_values[0]))
        det_values.append(abs(float(np.linalg.det(jacobian))))
    if not isotropy_values or not det_values:
        return 0.0, 0.0
    area_p90 = float(np.percentile(det_values, 90.0))
    area_gradient = float(np.percentile(det_values, 10.0)) / max(area_p90, 1e-12)
    return (
        float(np.clip(np.percentile(isotropy_values, 10.0), 0.0, 1.0)),
        float(np.clip(area_gradient, 0.0, 1.0)),
    )


def projective_quality_from_points(points_xy: np.ndarray, confidence: np.ndarray, *, conf_thr: float) -> dict[str, float | int]:
    """Pure-image AprilGrid projection health metric; no PnP/intrinsics required.

    The projective homography/isotropy estimate is intentionally point-based:
    every confident finite corner is a correspondence.  Earlier versions gated
    this metric on a minimum number of *complete* AprilTags, which made good
    partially visible boards report ``projective_jacobian_isotropy=0`` even when
    dozens of reliable corners were detected.  Complete-tag statistics are kept
    only for legacy shape/area summaries and ``projective_valid_tags``.
    """
    points = np.asarray(points_xy, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if points.size == 0 or conf.size == 0:
        return _empty_projective_quality()
    try:
        points = points.reshape(APRILGRID_TAG_ROWS * APRILGRID_TAG_COLS, 4, 2)
        conf = conf.reshape(APRILGRID_TAG_ROWS * APRILGRID_TAG_COLS, 4)
    except ValueError:
        return _empty_projective_quality()
    valid_corners = np.isfinite(points).all(axis=2) & (conf >= float(conf_thr))
    valid_point_count = int(np.count_nonzero(valid_corners))
    good_tags = valid_corners.all(axis=1)
    valid_tag_count = int(np.count_nonzero(good_tags))
    if valid_point_count < 16:
        return _empty_projective_quality(valid_tags=valid_tag_count)

    if valid_tag_count > 0:
        image_corners = points[good_tags]
        shape_values = np.asarray([_tag_shape_quality(tag) for tag in image_corners], dtype=np.float64)
        area_values = np.asarray([_quad_area(tag) for tag in image_corners], dtype=np.float64)
        local_shape = float(np.clip(np.percentile(shape_values, 10.0), 0.0, 1.0))
        area_uniformity = float(np.percentile(area_values, 10.0)) / max(float(np.percentile(area_values, 90.0)), 1e-12)
        area_uniformity = float(np.clip(area_uniformity, 0.0, 1.0))
    else:
        # No complete tag exists, but a homography can still be estimated from
        # enough individual ordered AprilGrid corners.  Do not penalize the
        # point-based projective metric with an unavailable tag-level term.
        local_shape = 1.0
        area_uniformity = 1.0

    best_rms = float("inf")
    best_h = None
    best_object = None
    for permutation in _CORNER_PERMUTATIONS:
        obj_all = _APRILGRID_OBJECT_CORNERS[:, permutation, :]
        obj = obj_all[valid_corners].reshape(-1, 2).astype(np.float32, copy=False)
        img = points[valid_corners].reshape(-1, 2).astype(np.float32, copy=False)
        if obj.shape[0] < 4 or img.shape[0] < 4:
            continue
        homography, _ = cv2.findHomography(obj, img, 0)
        if homography is None:
            continue
        projected = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), homography).reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=1))))
        if rms < best_rms:
            best_rms = rms
            best_h = homography
            best_object = obj
    if best_h is None or best_object is None:
        return _empty_projective_quality(valid_tags=valid_tag_count)
    jacobian_isotropy, area_gradient = _homography_jacobian_stats(best_h, best_object)
    quality = min(local_shape, math.sqrt(area_uniformity), jacobian_isotropy, math.sqrt(area_gradient))
    return {
        "projective_quality": float(np.clip(quality, 0.0, 1.0)),
        "projective_local_shape": float(local_shape),
        "projective_area_uniformity": float(area_uniformity),
        "projective_jacobian_isotropy": float(jacobian_isotropy),
        "projective_area_gradient": float(area_gradient),
        "projective_homography_rms_px": float(best_rms),
        "projective_valid_tags": int(valid_tag_count),
    }


def diagonal_ratio_from_points(points_xy: np.ndarray, confidence: np.ndarray, *, conf_thr: float) -> tuple[float, int]:
    """Return the full-board bbox short/long side ratio from all valid detected tag corners.

    A tag contributes only when all four corners are finite and meet the confidence
    threshold. The ratio is computed over the union of all such tag corners as
    min(max_x - min_x, max_y - min_y) / max(max_x - min_x, max_y - min_y).
    """
    points = np.asarray(points_xy, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if points.size == 0 or conf.size == 0:
        return 0.0, 0
    try:
        points = points.reshape(APRILGRID_TAG_ROWS * APRILGRID_TAG_COLS, 4, 2)
        conf = conf.reshape(APRILGRID_TAG_ROWS * APRILGRID_TAG_COLS, 4)
    except ValueError:
        return 0.0, 0
    good_tags = np.isfinite(points).all(axis=(1, 2)) & (conf >= float(conf_thr)).all(axis=1)
    valid_count = int(np.count_nonzero(good_tags))
    if valid_count == 0:
        return 0.0, 0
    corners = points[good_tags].reshape(-1, 2).astype(np.float64, copy=False)
    span = np.max(corners, axis=0) - np.min(corners, axis=0)
    width = float(max(span[0], 0.0))
    height = float(max(span[1], 0.0))
    denom = max(width, height, 1e-9)
    ratio = min(width, height) / denom
    return float(np.clip(ratio, 0.0, 1.0)), valid_count


def board_bbox_stats_from_points(
    points_xy: np.ndarray,
    confidence: np.ndarray,
    *,
    conf_thr: float,
    image_width: int,
    image_height: int,
) -> dict[str, float | int]:
    """Return board-level bbox center/scale metrics from all valid detected tags."""
    points = np.asarray(points_xy, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if points.size == 0 or conf.size == 0:
        return _empty_board_bbox_stats()
    try:
        points = points.reshape(APRILGRID_TAG_ROWS * APRILGRID_TAG_COLS, 4, 2)
        conf = conf.reshape(APRILGRID_TAG_ROWS * APRILGRID_TAG_COLS, 4)
    except ValueError:
        return _empty_board_bbox_stats()
    good_tags = np.isfinite(points).all(axis=(1, 2)) & (conf >= float(conf_thr)).all(axis=1)
    valid_count = int(np.count_nonzero(good_tags))
    if valid_count == 0:
        return _empty_board_bbox_stats(valid_tags=valid_count)
    corners = points[good_tags].reshape(-1, 2).astype(np.float64, copy=False)
    xy_min = np.min(corners, axis=0)
    xy_max = np.max(corners, axis=0)
    center = 0.5 * (xy_min + xy_max)
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    center_x_norm = float(center[0] / width)
    center_y_norm = float(center[1] / height)
    center_dist_norm = math.hypot(
        (float(center[0]) - 0.5 * width) / max(0.5 * width, 1.0),
        (float(center[1]) - 0.5 * height) / max(0.5 * height, 1.0),
    )
    span = xy_max - xy_min
    image_diag = math.hypot(width, height)
    board_diag_norm = math.hypot(float(span[0]), float(span[1])) / max(image_diag, 1.0)
    return {
        "board_center_x_norm": float(np.clip(center_x_norm, 0.0, 1.0)),
        "board_center_y_norm": float(np.clip(center_y_norm, 0.0, 1.0)),
        "board_center_dist_norm": float(center_dist_norm),
        "board_diagonal_norm": float(np.clip(board_diag_norm, 0.0, 10.0)),
        "board_bbox_valid_tags": int(valid_count),
    }


def _empty_projective_quality(*, valid_tags: int = 0) -> dict[str, float | int]:
    return {
        "projective_quality": 0.0,
        "projective_local_shape": 0.0,
        "projective_area_uniformity": 0.0,
        "projective_jacobian_isotropy": 0.0,
        "projective_area_gradient": 0.0,
        "projective_homography_rms_px": float("inf"),
        "projective_valid_tags": int(valid_tags),
    }


def _empty_board_bbox_stats(*, valid_tags: int = 0) -> dict[str, float | int]:
    return {
        "board_center_x_norm": 0.0,
        "board_center_y_norm": 0.0,
        "board_center_dist_norm": 1.0,
        "board_diagonal_norm": 0.0,
        "board_bbox_valid_tags": int(valid_tags),
    }


def bits_to_hex(mask: int, n_bits: int) -> str:
    n_hex = (int(n_bits) + 3) // 4
    return f"{int(mask) & ((1 << int(n_bits)) - 1):0{n_hex}x}"


def popcount(mask: int) -> int:
    return int(mask).bit_count()


def bits_iter(mask: int, n_bits: int):
    current = int(mask)
    while current:
        lsb = current & -current
        idx = lsb.bit_length() - 1
        if idx < int(n_bits):
            yield idx
        current ^= lsb


def resolve_grid(height: int, width: int, grid_cell_px: int) -> tuple[int, int]:
    rows = int(max(1, math.ceil(float(height) / float(grid_cell_px))))
    cols = int(max(1, math.ceil(float(width) / float(grid_cell_px))))
    return rows, cols


def _grid_index(x_px: float, y_px: float, *, height: int, width: int, rows: int, cols: int) -> int | None:
    if width <= 0 or height <= 0:
        return None
    col = int(np.floor(float(x_px) / float(width) * float(cols)))
    row = int(np.floor(float(y_px) / float(height) * float(rows)))
    if row < 0 or row >= int(rows) or col < 0 or col >= int(cols):
        return None
    return int(row) * int(cols) + int(col)


def valid_cells_from_gray_integral(
    image_bgr: np.ndarray,
    *,
    rows: int,
    cols: int,
    gray_thr: int,
    cell_min_frac: float,
) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    binary = (gray > int(gray_thr)).astype(np.uint8)
    integral = cv2.integral(binary, sdepth=cv2.CV_32S)

    mask = 0
    for row in range(int(rows)):
        y0 = int(np.floor(row * height / rows))
        y1 = max(int(np.floor((row + 1) * height / rows)), y0 + 1)
        for col in range(int(cols)):
            x0 = int(np.floor(col * width / cols))
            x1 = max(int(np.floor((col + 1) * width / cols)), x0 + 1)
            area = float((y1 - y0) * (x1 - x0))
            pixel_sum = int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
            frac = float(pixel_sum) / area if area > 0 else 0.0
            if frac >= float(cell_min_frac):
                mask |= 1 << (row * cols + col)
    return int(mask)


def erode_mask(mask: int, rows: int, cols: int, *, n_iter: int) -> int:
    current = int(mask)
    for _ in range(int(n_iter)):
        eroded = 0
        for row in range(int(rows)):
            for col in range(int(cols)):
                idx = row * int(cols) + col
                if not (current & (1 << idx)):
                    continue
                keep = True
                for drow in (-1, 0, 1):
                    for dcol in (-1, 0, 1):
                        if drow == 0 and dcol == 0:
                            continue
                        nrow = row + drow
                        ncol = col + dcol
                        if nrow < 0 or nrow >= int(rows) or ncol < 0 or ncol >= int(cols):
                            keep = False
                            break
                        if not (current & (1 << (nrow * int(cols) + ncol))):
                            keep = False
                            break
                    if not keep:
                        break
                if keep:
                    eroded |= 1 << idx
        current = eroded
    return int(current)


def compute_edge_depth(valid_mask: int, edge_ratio: float) -> int:
    valid_cells = popcount(valid_mask)
    if valid_cells <= 0:
        return 1
    equiv_radius = math.sqrt(float(valid_cells) / math.pi)
    return int(max(1, round(float(edge_ratio) * equiv_radius)))


def build_camera_mask(
    camera_id: str,
    image_bgr: np.ndarray,
    *,
    grid_cell_px: int,
    gray_thr: int,
    cell_min_frac: float,
    edge_ratio: float,
) -> CameraMask:
    height, width = image_bgr.shape[:2]
    rows, cols = resolve_grid(height, width, grid_cell_px)
    valid_mask = valid_cells_from_gray_integral(
        image_bgr,
        rows=rows,
        cols=cols,
        gray_thr=gray_thr,
        cell_min_frac=cell_min_frac,
    )
    if valid_mask == 0:
        raise ValueError(f"Valid lens mask is empty for {camera_id}")
    if edge_ratio > 0.0:
        center_mask = erode_mask(valid_mask, rows, cols, n_iter=compute_edge_depth(valid_mask, edge_ratio))
        if center_mask == 0:
            raise ValueError(f"Center mask is empty for {camera_id}; lower edge_ratio")
    else:
        center_mask = valid_mask
    return CameraMask(
        camera_id=camera_id,
        image_height=int(height),
        image_width=int(width),
        grid_rows=int(rows),
        grid_cols=int(cols),
        valid_mask=int(valid_mask),
        center_mask=int(center_mask),
    )


def coverage_mask_from_points(
    points_xy: np.ndarray,
    confidence: np.ndarray,
    *,
    conf_thr: float,
    mask: CameraMask,
) -> tuple[int, int, int, np.ndarray]:
    coverage_mask = 0
    detected_points = 0
    center_points = 0
    valid_points: list[tuple[float, float]] = []
    for point, conf in zip(np.asarray(points_xy, dtype=np.float32), np.asarray(confidence, dtype=np.float32)):
        if float(conf) < float(conf_thr):
            continue
        idx = _grid_index(
            float(point[0]),
            float(point[1]),
            height=mask.image_height,
            width=mask.image_width,
            rows=mask.grid_rows,
            cols=mask.grid_cols,
        )
        if idx is None:
            continue
        bit = 1 << int(idx)
        if (mask.valid_mask & bit) == 0:
            continue
        coverage_mask |= bit
        detected_points += 1
        if (mask.center_mask & bit) != 0:
            center_points += 1
        valid_points.append((float(point[0]), float(point[1])))
    return (
        int(coverage_mask),
        int(detected_points),
        int(center_points),
        np.asarray(valid_points, dtype=np.float32).reshape(-1, 2),
    )


def evaluate_observation(
    *,
    camera_id: str,
    image_path,
    timestamp_ns: int,
    resize: ResizePlan,
    mask: CameraMask,
    points_xy: np.ndarray,
    confidence: np.ndarray,
    conf_thr: float,
    radial_normalizer: RadialSpanNormalizer | None = None,
) -> CameraObservation:
    coverage_mask, detected_points, center_points, valid_points_xy = coverage_mask_from_points(
        points_xy,
        confidence,
        conf_thr=conf_thr,
        mask=mask,
    )
    projective = projective_quality_from_points(points_xy, confidence, conf_thr=conf_thr)
    diagonal_ratio, diagonal_valid_tags = diagonal_ratio_from_points(points_xy, confidence, conf_thr=conf_thr)
    board_bbox = board_bbox_stats_from_points(
        points_xy,
        confidence,
        conf_thr=conf_thr,
        image_width=mask.image_width,
        image_height=mask.image_height,
    )
    center_hit = bool(int(coverage_mask) & int(mask.center_mask))
    usable = int(coverage_mask) != 0 and center_hit
    center_points_fraction = float(center_points) / float(detected_points) if int(detected_points) > 0 else 0.0
    valid_corner_mask, valid_tag_mask = valid_target_masks(points_xy, confidence, conf_thr=conf_thr)
    radial_span = 0.0 if radial_normalizer is None else radial_normalizer.radial_span(valid_points_xy)
    center_area_fraction = (
        0.0 if radial_normalizer is None else radial_normalizer.center_area_fraction(valid_points_xy)
    )
    return CameraObservation(
        camera_id=camera_id,
        image_path=image_path,
        timestamp_ns=int(timestamp_ns),
        resize=resize,
        mask=mask,
        valid_points_xy=valid_points_xy,
        coverage_mask=int(coverage_mask),
        center_hit=bool(center_hit),
        usable=bool(usable),
        detected_points=int(detected_points),
        center_points=int(center_points),
        center_points_fraction=float(center_points_fraction),
        center_area_fraction=float(center_area_fraction),
        radial_span=float(radial_span),
        valid_corner_mask=int(valid_corner_mask),
        valid_tag_mask=int(valid_tag_mask),
        projective_quality=float(projective["projective_quality"]),
        projective_local_shape=float(projective["projective_local_shape"]),
        projective_area_uniformity=float(projective["projective_area_uniformity"]),
        projective_jacobian_isotropy=float(projective["projective_jacobian_isotropy"]),
        projective_area_gradient=float(projective["projective_area_gradient"]),
        projective_homography_rms_px=float(projective["projective_homography_rms_px"]),
        projective_valid_tags=int(projective["projective_valid_tags"]),
        diagonal_ratio=float(diagonal_ratio),
        diagonal_valid_tags=int(diagonal_valid_tags),
        board_center_x_norm=float(board_bbox["board_center_x_norm"]),
        board_center_y_norm=float(board_bbox["board_center_y_norm"]),
        board_center_dist_norm=float(board_bbox["board_center_dist_norm"]),
        board_diagonal_norm=float(board_bbox["board_diagonal_norm"]),
    )
