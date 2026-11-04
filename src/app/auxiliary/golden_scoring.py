from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OriginalGridSpikeScores:
    matrix: np.ndarray
    peak_original_indices: np.ndarray
    peak_aligned_indices: np.ndarray
    peak_cell_scores: np.ndarray
    peak_raw_deviations: np.ndarray
    peak_durations_seconds: np.ndarray
    median_offsets: np.ndarray


@dataclass(frozen=True)
class DurationAwareStepOffsets:
    offsets: np.ndarray
    transient_mask: np.ndarray
    event_durations_seconds: np.ndarray


def merge_spike_scores(
    baseline: OriginalGridSpikeScores,
    duration_aware: OriginalGridSpikeScores,
) -> OriginalGridSpikeScores:
    """Keep v0.15 spike scores and add stronger duration-qualified events."""
    if baseline.matrix.shape != duration_aware.matrix.shape:
        raise ValueError("병합할 스파이크 점수 shape가 다릅니다.")
    use_duration = duration_aware.matrix > baseline.matrix

    def choose(base: np.ndarray, duration: np.ndarray) -> np.ndarray:
        return np.where(use_duration, duration, base)

    return OriginalGridSpikeScores(
        matrix=np.maximum(baseline.matrix, duration_aware.matrix).astype(np.float32),
        peak_original_indices=choose(
            baseline.peak_original_indices,
            duration_aware.peak_original_indices,
        ).astype(np.int64),
        peak_aligned_indices=choose(
            baseline.peak_aligned_indices,
            duration_aware.peak_aligned_indices,
        ).astype(np.int64),
        peak_cell_scores=choose(
            baseline.peak_cell_scores,
            duration_aware.peak_cell_scores,
        ).astype(np.float32),
        peak_raw_deviations=choose(
            baseline.peak_raw_deviations,
            duration_aware.peak_raw_deviations,
        ).astype(np.float32),
        peak_durations_seconds=choose(
            baseline.peak_durations_seconds,
            duration_aware.peak_durations_seconds,
        ).astype(np.float32),
        median_offsets=baseline.median_offsets.astype(np.float32),
    )


def _resample_rows(values: np.ndarray, output_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if output_length <= 0:
        raise ValueError("output_length는 1 이상이어야 합니다.")
    if len(values) == output_length:
        return values.copy()
    if len(values) == 1:
        return np.repeat(values, output_length, axis=0).astype(np.float32)
    old_x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, output_length, dtype=np.float32)
    return np.stack([
        np.interp(new_x, old_x, values[:, feature_index])
        for feature_index in range(values.shape[1])
    ], axis=1).astype(np.float32)


def original_grid_spike_scores(
    x_scaled: np.ndarray,
    process_steps: np.ndarray,
    step_offsets: dict[str, list[int]] | dict[int, tuple[int, int]],
    model_golden_scaled: np.ndarray,
    error_scale: np.ndarray,
    top_k: int = 1,
    residual_center: np.ndarray | None = None,
    remove_step_offset: bool = False,
    baseline_offsets: np.ndarray | None = None,
    transient_mask: np.ndarray | None = None,
    event_durations_seconds: np.ndarray | None = None,
) -> OriginalGridSpikeScores:
    """Score short spikes before ProcessStep interpolation can dilute them.

    The aligned golden and error scale are mapped back to each log's original
    ProcessStep length. Each matrix cell is the mean of its largest ``top_k``
    original-grid residual scores. Peak indices retain the strongest single
    point so reports can identify its exact source row.
    """
    if int(top_k) < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    x_scaled = np.asarray(x_scaled, dtype=np.float32)
    process_steps = np.asarray(process_steps, dtype=np.int64)
    golden = np.asarray(model_golden_scaled, dtype=np.float32)
    scale = np.asarray(error_scale, dtype=np.float32)
    center = (
        np.zeros_like(golden, dtype=np.float32)
        if residual_center is None
        else np.asarray(residual_center, dtype=np.float32)
    )
    if x_scaled.ndim != 2 or golden.ndim != 2 or scale.ndim != 2:
        raise ValueError("스파이크 점수 입력은 모두 2차원이어야 합니다.")
    if x_scaled.shape[0] != process_steps.shape[0]:
        raise ValueError("원본 feature와 ProcessStep 길이가 다릅니다.")
    if golden.shape != scale.shape or center.shape != golden.shape or x_scaled.shape[1] != golden.shape[1]:
        raise ValueError("스파이크 점수 feature shape가 일치하지 않습니다.")
    if baseline_offsets is not None:
        baseline_offsets = np.asarray(baseline_offsets, dtype=np.float32)
        expected_shape = (len(step_offsets), x_scaled.shape[1])
        if baseline_offsets.shape != expected_shape:
            raise ValueError(
                f"baseline_offsets shape가 일치하지 않습니다: "
                f"{baseline_offsets.shape} != {expected_shape}"
            )
    if transient_mask is not None:
        transient_mask = np.asarray(transient_mask, dtype=bool)
        if transient_mask.shape != x_scaled.shape:
            raise ValueError("transient_mask shape가 원본 feature와 다릅니다.")
    if event_durations_seconds is not None:
        event_durations_seconds = np.asarray(event_durations_seconds, dtype=np.float32)
        if event_durations_seconds.shape != x_scaled.shape:
            raise ValueError("event_durations_seconds shape가 원본 feature와 다릅니다.")

    score_rows: list[np.ndarray] = []
    original_index_rows: list[np.ndarray] = []
    aligned_index_rows: list[np.ndarray] = []
    peak_score_rows: list[np.ndarray] = []
    peak_raw_deviation_rows: list[np.ndarray] = []
    peak_duration_rows: list[np.ndarray] = []
    offset_rows: list[np.ndarray] = []
    for step_index, step in enumerate(sorted(step_offsets, key=lambda value: int(value))):
        a, b = step_offsets[step]
        a, b = int(a), int(b)
        original_indices = np.flatnonzero(process_steps == int(step))
        if original_indices.size == 0:
            n_features = x_scaled.shape[1]
            score_rows.append(np.zeros(n_features, dtype=np.float32))
            original_index_rows.append(np.full(n_features, -1, dtype=np.int64))
            aligned_index_rows.append(np.full(n_features, -1, dtype=np.int64))
            peak_score_rows.append(np.zeros(n_features, dtype=np.float32))
            peak_raw_deviation_rows.append(np.zeros(n_features, dtype=np.float32))
            peak_duration_rows.append(np.zeros(n_features, dtype=np.float32))
            offset_rows.append(np.zeros(n_features, dtype=np.float32))
            continue

        n_original = int(original_indices.size)
        golden_original = _resample_rows(golden[a:b], n_original)
        scale_original = _resample_rows(scale[a:b], n_original)
        center_original = _resample_rows(center[a:b], n_original)
        residual = x_scaled[original_indices] - golden_original
        if baseline_offsets is not None:
            median_offset = baseline_offsets[step_index].astype(np.float32)
            residual = residual - median_offset[None, :]
        else:
            median_offset = np.median(residual, axis=0).astype(np.float32)
        if remove_step_offset and baseline_offsets is None:
            residual = residual - median_offset[None, :]
        centered_residual = residual - center_original
        signed_cells = centered_residual / np.maximum(
            scale_original,
            1e-6,
        )
        cells = np.abs(signed_cells)
        if transient_mask is not None:
            cells = np.where(transient_mask[original_indices], cells, 0.0)
        n_top = min(int(top_k), n_original)
        top_scores = np.partition(cells, n_original - n_top, axis=0)[-n_top:].mean(axis=0)
        peak_local = np.argmax(cells, axis=0)
        feature_indices = np.arange(cells.shape[1])
        peak_original = original_indices[peak_local]
        if n_original == 1 or b - a <= 1:
            peak_aligned = np.full(cells.shape[1], a, dtype=np.int64)
        else:
            relative = peak_local.astype(np.float64) / float(n_original - 1)
            peak_aligned = a + np.rint(relative * float(b - a - 1)).astype(np.int64)
        score_rows.append(top_scores.astype(np.float32))
        original_index_rows.append(peak_original.astype(np.int64))
        aligned_index_rows.append(peak_aligned.astype(np.int64))
        peak_score_rows.append(cells[peak_local, feature_indices].astype(np.float32))
        peak_raw_deviation_rows.append(
            centered_residual[peak_local, feature_indices].astype(np.float32)
        )
        if event_durations_seconds is None:
            peak_duration_rows.append(np.zeros(cells.shape[1], dtype=np.float32))
        else:
            peak_duration_rows.append(
                event_durations_seconds[
                    peak_original,
                    feature_indices,
                ].astype(np.float32)
            )
        offset_rows.append(median_offset)

    return OriginalGridSpikeScores(
        matrix=np.stack(score_rows, axis=0).astype(np.float32),
        peak_original_indices=np.stack(original_index_rows, axis=0),
        peak_aligned_indices=np.stack(aligned_index_rows, axis=0),
        peak_cell_scores=np.stack(peak_score_rows, axis=0).astype(np.float32),
        peak_raw_deviations=np.stack(peak_raw_deviation_rows, axis=0).astype(np.float32),
        peak_durations_seconds=np.stack(peak_duration_rows, axis=0).astype(np.float32),
        median_offsets=np.stack(offset_rows, axis=0).astype(np.float32),
    )


def _monotonic_elapsed(values: np.ndarray, fallback_dt: float = 0.1) -> tuple[np.ndarray, float]:
    elapsed = np.asarray(values, dtype=np.float64).copy()
    if elapsed.ndim != 1:
        raise ValueError("elapsed_seconds는 1차원이어야 합니다.")
    finite = np.isfinite(elapsed)
    positive = np.diff(elapsed[finite]) if finite.sum() > 1 else np.asarray([], dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    typical_dt = float(np.median(positive)) if positive.size else float(fallback_dt)
    typical_dt = max(typical_dt, 1e-6)
    if not finite.any():
        return np.arange(len(elapsed), dtype=np.float64) * typical_dt, typical_dt
    first = int(np.flatnonzero(finite)[0])
    for index in range(first - 1, -1, -1):
        elapsed[index] = elapsed[index + 1] - typical_dt
    for index in range(first + 1, len(elapsed)):
        if not np.isfinite(elapsed[index]) or elapsed[index] <= elapsed[index - 1]:
            elapsed[index] = elapsed[index - 1] + typical_dt
    return elapsed, typical_dt


def _mark_short_runs(
    candidate: np.ndarray,
    elapsed: np.ndarray,
    typical_dt: float,
    max_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    marked = np.zeros(len(candidate), dtype=bool)
    durations = np.zeros(len(candidate), dtype=np.float32)
    indices = np.flatnonzero(candidate)
    if indices.size == 0:
        return marked, durations
    max_gap = max(typical_dt * 3.0, typical_dt + 1e-6)
    run_start = 0
    for cursor in range(1, len(indices) + 1):
        split = cursor == len(indices)
        if not split:
            previous = int(indices[cursor - 1])
            current = int(indices[cursor])
            split = current != previous + 1 or elapsed[current] - elapsed[previous] > max_gap
        if not split:
            continue
        run = indices[run_start:cursor]
        duration = float(elapsed[int(run[-1])] - elapsed[int(run[0])] + typical_dt)
        if duration <= float(max_seconds) + 1e-9:
            marked[run] = True
            durations[run] = duration
        run_start = cursor
    return marked, durations


def duration_aware_step_offsets(
    x_raw: np.ndarray,
    process_steps: np.ndarray,
    elapsed_seconds: np.ndarray,
    step_offsets: dict[str, list[int]] | dict[int, tuple[int, int]],
    model_golden_raw: np.ndarray,
    normal_offset_center_raw: np.ndarray,
    residual_center_raw: np.ndarray,
    residual_scale_raw: np.ndarray,
    max_transient_seconds: float,
) -> DurationAwareStepOffsets:
    """Separate short original-grid excursions from sustained step offsets.

    A deviation that remains statistically large for no more than
    ``max_transient_seconds`` is marked as a transient and excluded before the
    ProcessStep median offset is recomputed. Actual Date/Time elapsed seconds
    are used, so irregular sampling does not change the classification.
    """
    if float(max_transient_seconds) <= 0.0:
        raise ValueError("max_transient_seconds는 0보다 커야 합니다.")
    x_raw = np.asarray(x_raw, dtype=np.float32)
    process_steps = np.asarray(process_steps, dtype=np.int64)
    elapsed, global_dt = _monotonic_elapsed(elapsed_seconds)
    golden = np.asarray(model_golden_raw, dtype=np.float32)
    normal_offsets = np.asarray(normal_offset_center_raw, dtype=np.float32)
    center = np.asarray(residual_center_raw, dtype=np.float32)
    scale = np.asarray(residual_scale_raw, dtype=np.float32)
    expected_offset_shape = (len(step_offsets), x_raw.shape[1])
    if x_raw.ndim != 2 or x_raw.shape[0] != len(process_steps) or len(elapsed) != len(x_raw):
        raise ValueError("duration-aware offset 입력 길이가 일치하지 않습니다.")
    if golden.shape != center.shape or golden.shape != scale.shape:
        raise ValueError("duration-aware offset 기준 shape가 일치하지 않습니다.")
    if normal_offsets.shape != expected_offset_shape:
        raise ValueError(
            f"normal_offset_center_raw shape가 일치하지 않습니다: "
            f"{normal_offsets.shape} != {expected_offset_shape}"
        )

    offsets: list[np.ndarray] = []
    transient_mask = np.zeros_like(x_raw, dtype=bool)
    event_durations = np.zeros_like(x_raw, dtype=np.float32)
    for step_index, step in enumerate(sorted(step_offsets, key=lambda value: int(value))):
        a, b = (int(value) for value in step_offsets[step])
        original_indices = np.flatnonzero(process_steps == int(step))
        if original_indices.size == 0:
            offsets.append(normal_offsets[step_index].copy())
            continue
        n_original = int(original_indices.size)
        golden_original = _resample_rows(golden[a:b], n_original)
        center_original = _resample_rows(center[a:b], n_original)
        scale_original = _resample_rows(scale[a:b], n_original)
        residual = x_raw[original_indices] - golden_original
        step_elapsed, typical_dt = _monotonic_elapsed(
            elapsed[original_indices],
            fallback_dt=global_dt,
        )
        step_offsets_row = np.zeros(x_raw.shape[1], dtype=np.float32)
        for feature_index in range(x_raw.shape[1]):
            values = residual[:, feature_index]
            expected = (
                normal_offsets[step_index, feature_index]
                + center_original[:, feature_index]
            )
            provisional = float(np.median(values))
            # A short ProcessStep whose whole level moved is a transient by the
            # requested duration rule. Longer steps use their robust median so
            # a sustained offset can coexist with a local spike.
            step_duration = float(step_elapsed[-1] - step_elapsed[0] + typical_dt)
            if step_duration <= float(max_transient_seconds) + 1e-9:
                deviation = values - expected
            else:
                deviation = values - provisional - center_original[:, feature_index]
            excursion_limit = np.maximum(
                3.0 * scale_original[:, feature_index],
                1e-6,
            )
            candidate = np.abs(deviation) >= excursion_limit
            marked, durations = _mark_short_runs(
                candidate,
                step_elapsed,
                typical_dt,
                max_transient_seconds,
            )
            keep = ~marked
            if keep.any():
                refined = float(np.median(values[keep]))
            else:
                refined = float(normal_offsets[step_index, feature_index])
            # Re-evaluate around the refined sustained offset. This second pass
            # catches a local spike that sits on top of a genuine long offset.
            refined_deviation = (
                values
                - refined
                - center_original[:, feature_index]
            )
            candidate = np.abs(refined_deviation) >= excursion_limit
            marked, durations = _mark_short_runs(
                candidate,
                step_elapsed,
                typical_dt,
                max_transient_seconds,
            )
            keep = ~marked
            if keep.any():
                refined = float(np.median(values[keep]))
            else:
                refined = float(normal_offsets[step_index, feature_index])
            step_offsets_row[feature_index] = refined
            transient_mask[original_indices, feature_index] = marked
            event_durations[original_indices, feature_index] = durations
        offsets.append(step_offsets_row)
    return DurationAwareStepOffsets(
        offsets=np.stack(offsets, axis=0).astype(np.float32),
        transient_mask=transient_mask,
        event_durations_seconds=event_durations,
    )


def remove_known_step_offsets(
    residuals: np.ndarray,
    step_offsets: dict[str, list[int]] | dict[int, tuple[int, int]],
    offsets: np.ndarray,
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    expected_shape = (len(step_offsets), residuals.shape[1])
    if offsets.shape != expected_shape:
        raise ValueError(f"offsets shape가 일치하지 않습니다: {offsets.shape} != {expected_shape}")
    corrected = residuals.copy()
    for step_index, step in enumerate(sorted(step_offsets, key=lambda value: int(value))):
        a, b = step_offsets[step]
        corrected[int(a):int(b)] -= offsets[step_index][None, :]
    return corrected.astype(np.float32)


def remove_step_median_offsets(
    residuals: np.ndarray,
    step_offsets: dict[str, list[int]] | dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Remove each ProcessStep-feature median and return those raw offsets."""
    residuals = np.asarray(residuals, dtype=np.float32)
    if residuals.ndim != 2:
        raise ValueError("residuals는 [time, feature] 2차원이어야 합니다.")
    corrected = residuals.copy()
    offsets: list[np.ndarray] = []
    for step in sorted(step_offsets, key=lambda value: int(value)):
        a, b = step_offsets[step]
        a, b = int(a), int(b)
        part = residuals[a:b]
        if len(part) == 0:
            offset = np.zeros(residuals.shape[1], dtype=np.float32)
        else:
            offset = np.median(part, axis=0).astype(np.float32)
            corrected[a:b] = part - offset[None, :]
        offsets.append(offset)
    return corrected.astype(np.float32), np.stack(offsets, axis=0).astype(np.float32)


def fit_detrended_mad_scale(
    detrended_residuals: np.ndarray,
    raw_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a pointwise MAD scale with feature-wide and engineering floors."""
    values = np.asarray(detrended_residuals, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("detrended_residuals는 [sample, time, feature] 3차원이어야 합니다.")
    if float(raw_floor) <= 0.0:
        raise ValueError("raw_floor는 0보다 커야 합니다.")
    center = np.median(values, axis=0).astype(np.float32)
    absolute_deviation = np.abs(values - center[None, :, :])
    pointwise_mad = (1.4826 * np.median(absolute_deviation, axis=0)).astype(np.float32)
    feature_mad = (
        1.4826 * np.median(absolute_deviation, axis=(0, 1))
    ).astype(np.float32)
    scale = np.maximum(pointwise_mad, feature_mad[None, :])
    scale = np.maximum(scale, float(raw_floor)).astype(np.float32)
    return center, scale, feature_mad


def robust_step_feature_scores(
    cell_scores: np.ndarray,
    step_offsets: dict[str, list[int]] | dict[int, tuple[int, int]],
    top_fraction: float = 0.05,
) -> np.ndarray:
    """Aggregate time scores into a ProcessStep x feature matrix.

    Each cell is the larger of the whole-step mean and the mean of the largest
    ``top_fraction`` time scores. This retains sustained shifts while remaining
    sensitive to a short abnormal section inside a long ProcessStep.
    """
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction은 0보다 크고 1 이하이어야 합니다.")
    rows: list[np.ndarray] = []
    for step in sorted(step_offsets, key=lambda value: int(value)):
        a, b = step_offsets[step]
        part = np.asarray(cell_scores[int(a):int(b)], dtype=np.float32)
        if len(part) == 0:
            rows.append(np.zeros(cell_scores.shape[1], dtype=np.float32))
            continue
        mean_score = part.mean(axis=0)
        n_top = max(1, int(math.ceil(len(part) * float(top_fraction))))
        top_score = np.partition(part, len(part) - n_top, axis=0)[-n_top:].mean(axis=0)
        rows.append(np.maximum(mean_score, top_score))
    return np.stack(rows, axis=0).astype(np.float32)


def robust_signed_step_feature_scores(
    signed_cell_scores: np.ndarray,
    step_offsets: dict[str, list[int]] | dict[int, tuple[int, int]],
    top_fraction: float = 0.05,
) -> np.ndarray:
    """Aggregate signed residual scores while retaining their direction.

    For each ProcessStep and feature, compare the signed whole-step mean with
    the signed mean of time points having the largest absolute residuals. The
    candidate with the larger magnitude is retained with its original sign.
    """
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction은 0보다 크고 1 이하이어야 합니다.")
    rows: list[np.ndarray] = []
    for step in sorted(step_offsets, key=lambda value: int(value)):
        a, b = step_offsets[step]
        part = np.asarray(signed_cell_scores[int(a):int(b)], dtype=np.float32)
        if len(part) == 0:
            rows.append(np.zeros(signed_cell_scores.shape[1], dtype=np.float32))
            continue
        mean_score = part.mean(axis=0)
        n_top = max(1, int(math.ceil(len(part) * float(top_fraction))))
        top_indices = np.argpartition(np.abs(part), len(part) - n_top, axis=0)[-n_top:]
        top_values = np.take_along_axis(part, top_indices, axis=0)
        top_score = top_values.mean(axis=0)
        rows.append(np.where(np.abs(top_score) > np.abs(mean_score), top_score, mean_score))
    return np.stack(rows, axis=0).astype(np.float32)


def summarize_cylinder_matrix(matrix: np.ndarray) -> dict[str, float | int]:
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(f"실린더 점수 행렬은 [ProcessStep, 3]이어야 합니다: {matrix.shape}")
    flat_index = int(np.argmax(matrix))
    step_index, feature_index = np.unravel_index(flat_index, matrix.shape)
    step_scores = matrix.max(axis=1)
    feature_scores = matrix.max(axis=0)
    return {
        "score": float(step_scores.max()),
        "step_index": int(step_index),
        "feature_index": int(feature_index),
        "right_score": float(feature_scores[0]),
        "left_score": float(feature_scores[1]),
        "difference_score": float(feature_scores[2]),
    }


def summarize_signed_cylinder_matrix(matrix: np.ndarray) -> dict[str, float | int]:
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(f"실린더 방향성 점수 행렬은 [ProcessStep, 3]이어야 합니다: {matrix.shape}")
    flat_index = int(np.argmax(np.abs(matrix)))
    step_index, feature_index = np.unravel_index(flat_index, matrix.shape)
    return {
        "score": float(matrix[step_index, feature_index]),
        "step_index": int(step_index),
        "feature_index": int(feature_index),
    }


def conformal_upper(values: np.ndarray, false_positive_rate: float) -> tuple[np.ndarray, int]:
    """Finite-sample upper conformal quantile along axis 0."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[0] == 0:
        raise ValueError("임계값 보정 점수가 없습니다.")
    alpha = float(false_positive_rate)
    if not 0.0 < alpha < 1.0:
        raise ValueError("false_positive_rate는 0보다 크고 1보다 작아야 합니다.")
    rank = min(arr.shape[0], max(1, int(math.ceil((arr.shape[0] + 1) * (1.0 - alpha)))))
    return np.sort(arr, axis=0)[rank - 1].astype(np.float32), rank
