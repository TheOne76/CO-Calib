from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .models import DatawashConfig, SelectionConfig, StageSelectionConfig

T = TypeVar("T")


def _known_values(cls: type[T], raw: dict[str, Any]) -> dict[str, Any]:
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {', '.join(unknown)}")
    return {key: value for key, value in raw.items() if key in known}


def _load_selection(raw: dict[str, Any]) -> SelectionConfig:
    known_stages = {field.name for field in fields(SelectionConfig)}
    unknown = sorted(set(raw) - known_stages)
    if unknown:
        raise ValueError(f"Unknown SelectionConfig fields: {', '.join(unknown)}")
    defaults = SelectionConfig()
    stages: dict[str, StageSelectionConfig] = {}
    for stage_name in sorted(known_stages):
        stage_raw = raw.get(stage_name, {})
        if not isinstance(stage_raw, dict):
            raise ValueError(f"selection.{stage_name} must be a mapping")
        default = getattr(defaults, stage_name)
        values = {field.name: getattr(default, field.name) for field in fields(StageSelectionConfig)}
        values.update(_known_values(StageSelectionConfig, stage_raw))
        stages[stage_name] = StageSelectionConfig(**values)
    return SelectionConfig(**stages)


def load_datawash_config(path: str | Path) -> DatawashConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("datawash config root must be a mapping")
    selection_raw = raw.pop("selection", {})
    if not isinstance(selection_raw, dict):
        raise ValueError("selection must be a mapping")
    selection = _load_selection(selection_raw)
    for stage_name in ("anchor", "covisible", "mono_fill"):
        stage = getattr(selection, stage_name)
        for threshold_name in ("radial_span", "iso"):
            value = float(getattr(stage, threshold_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"selection.{stage_name}.{threshold_name} must be in [0, 1]")
        if int(stage.budget) < 0:
            raise ValueError(f"selection.{stage_name}.budget must be >= 0")
    config = DatawashConfig(selection=selection, **_known_values(DatawashConfig, raw))
    if not 0.0 <= float(config.detector_confidence) <= 1.0:
        raise ValueError("detector_confidence must be in [0, 1]")
    if int(config.sample_min_detection_points) < 1:
        raise ValueError("sample_min_detection_points must be >= 1")
    return config
