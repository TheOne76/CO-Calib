from __future__ import annotations

import cv2
import numpy as np

from .models import ResizePlan


def compute_resize_plan(height: int, width: int, min_side: int, max_side: int) -> ResizePlan:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {height}x{width}")
    lower = float(min_side) / float(min(height, width))
    upper = float(max_side) / float(max(height, width))
    if lower > upper + 1e-9:
        raise ValueError(
            "Cannot satisfy detector side limits while preserving aspect ratio: "
            f"{height}x{width}, min_side={min_side}, max_side={max_side}"
        )
    scale = min(max(1.0, lower), upper)
    infer_height = max(1, int(round(float(height) * scale)))
    infer_width = max(1, int(round(float(width) * scale)))
    scale_x = 0.0 if infer_width <= 1 else float(width - 1) / float(infer_width - 1)
    scale_y = 0.0 if infer_height <= 1 else float(height - 1) / float(infer_height - 1)
    return ResizePlan(height, width, infer_height, infer_width, scale, scale_x, scale_y)


def resize_image(image_bgr: np.ndarray, plan: ResizePlan) -> np.ndarray:
    if image_bgr.shape[:2] == (plan.infer_height, plan.infer_width):
        return image_bgr
    return cv2.resize(image_bgr, (plan.infer_width, plan.infer_height), interpolation=cv2.INTER_LINEAR)


def remap_points_to_original(points_xy: np.ndarray, plan: ResizePlan) -> np.ndarray:
    mapped = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2).copy()
    if mapped.size == 0:
        return mapped
    mapped[:, 0] *= np.float32(plan.scale_x)
    mapped[:, 1] *= np.float32(plan.scale_y)
    return mapped
