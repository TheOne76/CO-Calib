from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .models import CameraMask, CameraObservation, CandidateGroup, SelectedGroup, SelectionConfig
from .quality import popcount


_ANCHOR_CENTER_AREA_FRACTION = 0.05
_GRID_AUTO_SEED_FRACTION = 0.65
_GRID_AUTO_CANDIDATE_FRACTION = 0.25
_GRID_AUTO_TARGET_MAX = 8


def coverage_ratio(mask: int, camera_mask: CameraMask) -> float:
    valid_count = max(popcount(camera_mask.valid_mask), 1)
    return float(popcount(int(mask) & int(camera_mask.valid_mask))) / float(valid_count)


def _within_budget(count: int, budget: int) -> bool:
    return int(budget) == 0 or int(count) < int(budget)


def _quality_ok(
    observation: CameraObservation,
    *,
    min_points: int,
    min_radial_span: float,
    min_iso: float,
) -> bool:
    return (
        int(observation.detected_points) >= int(min_points)
        and float(observation.radial_span) >= float(min_radial_span)
        and float(observation.projective_jacobian_isotropy) >= float(min_iso)
        and int(observation.coverage_mask) != 0
    )


def _observation_score(observation: CameraObservation) -> tuple[float, ...]:
    return (
        float(observation.radial_span),
        float(observation.projective_jacobian_isotropy),
        float(observation.center_area_fraction),
        float(observation.detected_points),
    )


@dataclass
class _State:
    selected: list[SelectedGroup]
    selected_keys: set[tuple[int, tuple[str, ...]]]
    selected_frame_cameras: set[tuple[int, str]]
    covered_by_camera: dict[str, int]
    counts_by_camera: dict[str, int]
    role_by_key: dict[tuple[int, tuple[str, ...]], str]


def _append(state: _State, candidate: CandidateGroup, cameras: Sequence[str], role: str) -> bool:
    active = tuple(sorted(cameras))
    key = (int(candidate.frame_id), active)
    if key in state.selected_keys:
        return False
    if any((int(candidate.frame_id), camera_id) in state.selected_frame_cameras for camera_id in active):
        return False
    state.selected_keys.add(key)
    state.role_by_key[key] = role
    state.selected.append(
        SelectedGroup(candidate.frame_id, candidate.timestamp_ns, active, candidate.observations)
    )
    for camera_id in active:
        state.selected_frame_cameras.add((int(candidate.frame_id), camera_id))
        state.covered_by_camera[camera_id] |= int(candidate.observations[camera_id].coverage_mask)
        state.counts_by_camera[camera_id] += 1
    return True


def _available(state: _State, candidate: CandidateGroup, cameras: Sequence[str]) -> bool:
    return all((int(candidate.frame_id), camera_id) not in state.selected_frame_cameras for camera_id in cameras)


def _new_cells(candidate: CandidateGroup, cameras: Sequence[str], covered: dict[str, int]) -> int:
    return sum(
        popcount(int(candidate.observations[camera_id].coverage_mask) & ~int(covered[camera_id]))
        for camera_id in cameras
    )


def _select_covisible(
    candidates: Sequence[CandidateGroup],
    masks: dict[str, CameraMask],
    config: SelectionConfig,
    state: _State,
    *,
    min_points: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    covisible_radial = float(config.covisible.radial_span)
    covisible_iso = float(config.covisible.iso)
    for first, second in combinations(tuple(masks), 2):
        pair_name = f"{first}-{second}"
        counts[pair_name] = 0
        eligible: list[tuple[CandidateGroup, int, int]] = []
        for candidate in candidates:
            left = candidate.observations[first]
            right = candidate.observations[second]
            if not all(
                _quality_ok(
                    observation,
                    min_points=min_points,
                    min_radial_span=covisible_radial,
                    min_iso=covisible_iso,
                )
                for observation in (left, right)
            ):
                continue
            common_points = popcount(int(left.valid_corner_mask) & int(right.valid_corner_mask))
            if common_points < int(min_points):
                continue
            common_tags = popcount(int(left.valid_tag_mask) & int(right.valid_tag_mask))
            eligible.append((candidate, common_points, common_tags))

        while eligible and _within_budget(counts[pair_name], config.covisible.budget):
            best_index = None
            best_key = None
            for index, (candidate, common_points, common_tags) in enumerate(eligible):
                if not _available(state, candidate, (first, second)):
                    continue
                new_cells = _new_cells(candidate, (first, second), state.covered_by_camera)
                if counts[pair_name] > 0 and new_cells == 0:
                    continue
                left = candidate.observations[first]
                right = candidate.observations[second]
                key = (
                    new_cells,
                    common_points,
                    common_tags,
                    min(left.projective_jacobian_isotropy, right.projective_jacobian_isotropy),
                    min(left.radial_span, right.radial_span),
                    -int(candidate.frame_id),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
            if best_index is None:
                break
            best, _, _ = eligible.pop(best_index)
            if _append(state, best, (first, second), f"covisible:{pair_name}"):
                counts[pair_name] += 1
    return counts


def _select_anchors(
    candidates: Sequence[CandidateGroup],
    masks: dict[str, CameraMask],
    config: SelectionConfig,
    state: _State,
    *,
    min_points: int,
) -> dict[str, int]:
    counts = {camera_id: 0 for camera_id in masks}
    anchor_radial = float(config.anchor.radial_span)
    anchor_iso = float(config.anchor.iso)
    for camera_id in masks:
        eligible = [
            candidate
            for candidate in candidates
            if _quality_ok(
                candidate.observations[camera_id],
                min_points=min_points,
                min_radial_span=anchor_radial,
                min_iso=anchor_iso,
            )
            and float(candidate.observations[camera_id].center_area_fraction) >= _ANCHOR_CENTER_AREA_FRACTION
        ]
        eligible.sort(
            key=lambda candidate: (*_observation_score(candidate.observations[camera_id]), -candidate.frame_id),
            reverse=True,
        )
        for candidate in eligible:
            if not _within_budget(counts[camera_id], config.anchor.budget):
                break
            if _available(state, candidate, (camera_id,)) and _append(state, candidate, (camera_id,), "anchor"):
                counts[camera_id] += 1
    return counts


class _GridCoverage:
    def __init__(self, camera_mask: CameraMask):
        self.valid_mask = int(camera_mask.valid_mask)
        self.counts = np.zeros(camera_mask.n_bits, dtype=np.int32)

    def add(self, footprint: int) -> None:
        active = int(footprint) & self.valid_mask
        while active:
            least_bit = active & -active
            self.counts[least_bit.bit_length() - 1] += 1
            active ^= least_bit

    def score(self, footprint: int, target: int) -> tuple[float, int]:
        active = int(footprint) & self.valid_mask
        score = 0.0
        empty = 0
        while active:
            least_bit = active & -active
            count = int(self.counts[least_bit.bit_length() - 1])
            if count == 0:
                score += 2.0
                empty += 1
            elif count < int(target):
                score += 1.0
            active ^= least_bit
        return score, empty


def _auto_grid_target(seed_footprints: Sequence[int], candidate_footprints: Sequence[int]) -> int:
    seed_mean = float(np.mean([popcount(mask) for mask in seed_footprints])) if seed_footprints else 0.0
    candidate_mean = (
        float(np.mean([popcount(mask) for mask in candidate_footprints])) if candidate_footprints else 0.0
    )
    target = math.ceil(max(1.0, seed_mean * _GRID_AUTO_SEED_FRACTION, candidate_mean * _GRID_AUTO_CANDIDATE_FRACTION))
    return min(int(target), _GRID_AUTO_TARGET_MAX)


def _select_mono_fill(
    candidates: Sequence[CandidateGroup],
    masks: dict[str, CameraMask],
    config: SelectionConfig,
    state: _State,
    *,
    min_points: int,
) -> tuple[dict[str, int], dict[str, int]]:
    added = {camera_id: 0 for camera_id in masks}
    targets: dict[str, int] = {}
    mono_radial = float(config.mono_fill.radial_span)
    mono_iso = float(config.mono_fill.iso)

    for camera_id, camera_mask in masks.items():
        grid = _GridCoverage(camera_mask)
        seed_footprints = [
            int(group.observations[camera_id].coverage_mask)
            for group in state.selected
            if camera_id in group.active_cams
        ]
        for footprint in seed_footprints:
            grid.add(footprint)
        eligible = [
            candidate
            for candidate in candidates
            if _quality_ok(
                candidate.observations[camera_id],
                min_points=min_points,
                min_radial_span=mono_radial,
                min_iso=mono_iso,
            )
        ]
        target = _auto_grid_target(
            seed_footprints,
            [int(candidate.observations[camera_id].coverage_mask) for candidate in eligible],
        )
        targets[camera_id] = target

        while eligible and _within_budget(added[camera_id], config.mono_fill.budget):
            best_index = None
            best_key = None
            for index, candidate in enumerate(eligible):
                if not _available(state, candidate, (camera_id,)):
                    continue
                observation = candidate.observations[camera_id]
                grid_score, empty_cells = grid.score(observation.coverage_mask, target)
                if grid_score <= 0.0:
                    continue
                key = (grid_score, empty_cells, *_observation_score(observation), -candidate.frame_id)
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
            if best_index is None:
                break
            best = eligible.pop(best_index)
            observation = best.observations[camera_id]
            if _append(state, best, (camera_id,), "mono_fill"):
                grid.add(observation.coverage_mask)
                added[camera_id] += 1
    return added, targets


def select_groups(
    candidates: Sequence[CandidateGroup],
    masks: dict[str, CameraMask],
    config: SelectionConfig,
    *,
    sample_min_detection_points: int,
) -> tuple[list[SelectedGroup], dict[str, Any], dict[tuple[int, tuple[str, ...]], str]]:
    state = _State(
        selected=[],
        selected_keys=set(),
        selected_frame_cameras=set(),
        covered_by_camera={camera_id: 0 for camera_id in masks},
        counts_by_camera={camera_id: 0 for camera_id in masks},
        role_by_key={},
    )
    covisible_counts = _select_covisible(
        candidates,
        masks,
        config,
        state,
        min_points=sample_min_detection_points,
    )
    anchor_counts = _select_anchors(
        candidates,
        masks,
        config,
        state,
        min_points=sample_min_detection_points,
    )
    mono_counts, mono_targets = _select_mono_fill(
        candidates,
        masks,
        config,
        state,
        min_points=sample_min_detection_points,
    )
    state.selected.sort(key=lambda group: (group.timestamp_ns, group.active_cams))
    summary = {
        "selected_groups": len(state.selected),
        "per_camera_counts": dict(state.counts_by_camera),
        "per_camera_coverage_ratio": {
            camera_id: coverage_ratio(state.covered_by_camera[camera_id], masks[camera_id]) for camera_id in masks
        },
        "anchor_counts": anchor_counts,
        "covisible_counts": covisible_counts,
        "mono_fill_counts": mono_counts,
        "mono_grid_targets": mono_targets,
        "effective_thresholds": {
            "sample_min_detection_points": int(sample_min_detection_points),
            "anchor_radial_span": float(config.anchor.radial_span),
            "anchor_iso": float(config.anchor.iso),
            "covisible_radial_span": float(config.covisible.radial_span),
            "covisible_iso": float(config.covisible.iso),
            "mono_fill_radial_span": float(config.mono_fill.radial_span),
            "mono_fill_iso": float(config.mono_fill.iso),
        },
        "budgets": {
            "anchor_per_camera": int(config.anchor.budget),
            "covisible_per_edge": int(config.covisible.budget),
            "mono_fill_per_camera": int(config.mono_fill.budget),
        },
    }
    return state.selected, summary, state.role_by_key
