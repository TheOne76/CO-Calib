from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .models import DetectorOutput


def normalize_device(device: str) -> str:
    name = str(device).strip().lower()
    if name not in {"auto", "gpu", "cpu"}:
        raise ValueError("device must be 'auto', 'gpu', or 'cpu'")
    return name


def bgr_to_infer_nchw(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32, copy=False)
    chw *= np.float32(1.0 / 255.0)
    return np.ascontiguousarray(chw)[None, ...]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    out = np.empty_like(values, dtype=np.float32)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    out[~positive] = exponent / (1.0 + exponent)
    return out


def _flatten_xy(points: np.ndarray) -> np.ndarray:
    if points.ndim == 4:
        return points.reshape(points.shape[0], -1, 2)
    if points.ndim == 3:
        return points
    raise ValueError(f"Unexpected xy_px shape: {tuple(points.shape)}")


def _flatten_confidence(confidence: np.ndarray) -> np.ndarray:
    if confidence.ndim == 3:
        return confidence.reshape(confidence.shape[0], -1)
    if confidence.ndim == 2:
        return confidence
    raise ValueError(f"Unexpected confidence shape: {tuple(confidence.shape)}")


def _confidence_from_heatmap(logits: np.ndarray, points: np.ndarray, height: int, width: int) -> np.ndarray:
    if logits.ndim == 5:
        logits = logits.reshape(logits.shape[0], -1, logits.shape[-2], logits.shape[-1])
    if logits.ndim != 4:
        raise ValueError(f"Unexpected heatmap_logits shape: {tuple(logits.shape)}")
    heatmap = _sigmoid(logits)
    batch, count, heat_height, heat_width = heatmap.shape
    ix = np.rint(points[:, :, 0] / float(max(width - 1, 1)) * float(max(heat_width - 1, 1))).astype(np.int64)
    iy = np.rint(points[:, :, 1] / float(max(height - 1, 1)) * float(max(heat_height - 1, 1))).astype(np.int64)
    ix = np.clip(ix, 0, heat_width - 1)
    iy = np.clip(iy, 0, heat_height - 1)
    values = heatmap[np.arange(batch)[:, None], np.arange(count)[None, :], iy, ix]
    return np.clip(values, 0.0, 1.0).astype(np.float32)


class NNDetector:
    """ONNX AprilGrid detector used by the public command line and Datawash."""

    def __init__(self, model_path: str | Path, *, sessions: int = 1, device: str = "auto"):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required; create the provided conda environment") from exc

        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"NN-Detector model not found: {self.model_path}")
        self.device = normalize_device(device)
        providers = ["CPUExecutionProvider"]
        cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
        if self.device != "cpu" and cuda_available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "gpu" and not cuda_available:
            warnings.warn(
                "CUDAExecutionProvider is unavailable; NN-Detector is falling back to CPU",
                RuntimeWarning,
                stacklevel=2,
            )
        self._ort = ort
        self._providers = providers
        self._sessions: list[object] = []
        self._input_name = ""
        self._output_names: tuple[str, ...] = ()
        self.ensure_sessions(sessions)

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(self._providers)

    def ensure_sessions(self, count: int) -> None:
        while len(self._sessions) < max(1, int(count)):
            try:
                session = self._ort.InferenceSession(str(self.model_path), providers=self._providers)
            except Exception:
                if "CUDAExecutionProvider" not in self._providers:
                    raise
                warnings.warn(
                    "CUDA initialization failed; NN-Detector is falling back to CPU",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._providers = ["CPUExecutionProvider"]
                session = self._ort.InferenceSession(str(self.model_path), providers=self._providers)
            if not self._input_name:
                self._input_name = str(session.get_inputs()[0].name)
                self._output_names = tuple(str(output.name) for output in session.get_outputs())
                self._providers = list(session.get_providers())
            self._sessions.append(session)

    def detect_batch_nchw(self, tensors: Sequence[np.ndarray], *, session_id: int = 0) -> list[DetectorOutput]:
        if not tensors:
            return []
        shapes = {tuple(tensor.shape[1:]) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError("All images in a detector batch must have the same shape")
        batch = np.ascontiguousarray(np.concatenate(tuple(tensors), axis=0))
        requested = ["xy_px"]
        if "confidence" in self._output_names:
            requested.append("confidence")
        elif "heatmap_logits" in self._output_names:
            requested.append("heatmap_logits")
        else:
            raise KeyError(f"Unsupported model outputs: {self._output_names}")
        session = self._sessions[int(session_id) % len(self._sessions)]
        values = session.run(requested, {self._input_name: batch})
        outputs = dict(zip(requested, values))
        points = _flatten_xy(np.asarray(outputs["xy_px"], dtype=np.float32))
        if "confidence" in outputs:
            confidence = np.clip(
                _flatten_confidence(np.asarray(outputs["confidence"], dtype=np.float32)), 0.0, 1.0
            )
        else:
            confidence = _confidence_from_heatmap(
                np.asarray(outputs["heatmap_logits"], dtype=np.float32),
                points,
                int(batch.shape[2]),
                int(batch.shape[3]),
            )
        return [
            DetectorOutput(points_xy=points[index].copy(), confidence=confidence[index].copy())
            for index in range(points.shape[0])
        ]

    def detect(self, image_bgr: np.ndarray) -> DetectorOutput:
        return self.detect_batch_nchw([bgr_to_infer_nchw(image_bgr)])[0]
