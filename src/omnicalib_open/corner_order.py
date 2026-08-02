from __future__ import annotations

import numpy as np


def kalibr_target_to_canonical(
    target_indices: np.ndarray,
    image_points: np.ndarray,
    *,
    grid_rows: int,
    grid_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(grid_rows) != 12 or int(grid_cols) != 12:
        raise ValueError("Datawash currently supports a 6x6 AprilGrid target")
    indices = np.asarray(target_indices, dtype=np.int32).reshape(-1)
    corners = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if indices.size != corners.shape[0]:
        raise ValueError("Kalibr corner indices and image points have different lengths")

    points = np.full((144, 2), np.nan, dtype=np.float32)
    confidence = np.zeros(144, dtype=np.float32)
    tag_cols = int(grid_cols) // 2
    offset_to_model_slot = {(0, 1): 0, (0, 0): 1, (1, 0): 2, (1, 1): 3}
    for target_index, point in zip(indices.tolist(), corners):
        row, col = divmod(int(target_index), int(grid_cols))
        tag_row, row_offset = divmod(row, 2)
        tag_col, col_offset = divmod(col, 2)
        tag_index = tag_row * tag_cols + tag_col
        canonical_index = tag_index * 4 + offset_to_model_slot[(row_offset, col_offset)]
        if not 0 <= canonical_index < 144:
            raise ValueError(f"Kalibr target index {target_index} is outside a 6x6 AprilGrid")
        points[canonical_index] = point
        confidence[canonical_index] = 1.0
    return points, confidence
