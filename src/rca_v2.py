"""Rule-based RCA v2 overlay built on the related-overlay plotting style."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .config import IOConfig
from .related_overlay import (
    ArtifactModel,
    Event,
    Trace,
    _build_ok_value_bands,
    _csv_files,
    _draw_io_change_markers,
    _draw_top_header,
    _family,
    _format_float,
    _io_value_text,
    _load_artifact,
    _load_model_records,
    _lookup_ok_value_mean,
    _mask_spans,
    _merge_traces,
    _model_record_columns,
    _read_search_best,
    _read_search_threshold,
    _resolve_artifacts,
    _safe_filename,
    _step_mask,
    _sub_ci,
    _io_change_markers,
    _load_record_exact,
)
from .sensor_graph import SensorGraph, load_sensor_graph


CONTROL_SENSORS = {
    "h_005_PV",
    "h_0052_PV",
    "h_0053_PV",
}
INNER_SENSOR = "t_002_PV"
PRIMARY_NORMAL_CANDIDATES = [
    "t_003_PV",
    "p_003_PV",
    "p_004_PV",
    "t_004_PV",
]
OUTLET_PULSEVENT_CANDIDATES = [
    "p_003_PV",
    "t_003_PV",
    "p_004_PV",
    "t_004_PV",
]
FRONT_VESSEL = "p_008_PV"
FRONT_FILTER = "h_003_PV"
FRONT_VALVE = "h_004_PV"
FRONT_SUPPLY_LINE = "h_0162_PV"
BACK_CONTROL = "h_0053_PV"
BACK_FILL = "t_001_PV"
BACK_FILTER = "h_001_PV"
BACK_VALVE = "h_002_PV"
BACK_VESSEL = "p_007_PV"
BACK_FILL_OR_VESSEL = [BACK_FILL, BACK_VESSEL]
ABNORMAL_SUPPLY_CANDIDATES = [
    "p_005_PV",
    "p_006_PV",
    "p_006_PV",
    "t_005_PV",
]
ABNORMAL_DOWNSTREAM_PT_FALLBACK = [
    "t_001_PV",
    "p_008_PV",
    "p_007_PV",
    "p_003_PV",
    "t_003_PV",
    "p_004_PV",
    "t_004_PV",
]
FRONT_VALVE_ADJUST_SENSOR = "h_004_PV"
UP_HEATERS = [f"h_022{i}_PV" for i in range(1, 5)]
DOWN_HEATERS = [
    f"h_019{i}_PV" for i in range(1, 5)
] + [
    f"h_019{i}_PV" for i in range(1, 5)
]
VESSEL_HEAT_SENSORS = set(UP_HEATERS + DOWN_HEATERS)
DISPLAY_ROUTE_MAX_NORMAL = 9
DISPLAY_ROUTE_MAX_ABNORMAL = 11
DISPLAY_ROUTE_MIN_NORMAL = 6
DISPLAY_ROUTE_MIN_ABNORMAL = 8
FRONT_LINE_BASE = [FRONT_FILTER, FRONT_VALVE, FRONT_SUPPLY_LINE, FRONT_VESSEL]
BACK_LINE_BASE = [BACK_CONTROL, BACK_FILTER, BACK_VALVE]
FRONT_LINE_SET = set(FRONT_LINE_BASE + UP_HEATERS)
BACK_LINE_SET = set(BACK_LINE_BASE + BACK_FILL_OR_VESSEL + DOWN_HEATERS)


@dataclass
class PeakChoice:
    sensor: str
    step: Optional[int]
    peak_index: int
    level: float


@dataclass
class InnerAccumReference:
    sensor: str
    by_recipe: dict[str, tuple[float, float]]
    global_range: tuple[float, float] | None
    ok_records: int
    ok_bands: dict | None = None
    source: str = ""
    threshold: object = None
    score_kind: str | None = None


def _normalize(value: str) -> str:
    import re

    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def _resolve_trace_name(query: str, traces: dict[str, Trace]) -> str | None:
    aliases = [query]
    for name in aliases:
        if name in traces:
            return name
    wanted = {_normalize(name) for name in aliases}
    for name in traces:
        if _normalize(name) in wanted:
            return name
    return None


def _resolve_many(names: list[str], traces: dict[str, Trace]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        resolved = _resolve_trace_name(name, traces)
        if resolved is not None and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def _ordered_sensors(names: list[str], traces: dict[str, Trace],
                     graph: SensorGraph) -> list[str]:
    sensors = _resolve_many(names, traces)
    sensors.sort(key=lambda s: (int(graph.rank.get(s, 10**9)), s))
    return sensors


def _after_inner_sensors(names: list[str], traces: dict[str, Trace],
                         graph: SensorGraph,
                         inner_sensor: str | None = None) -> list[str]:
    sensors = _resolve_many(names, traces)
    inner = _resolve_trace_name(inner_sensor or INNER_SENSOR, traces) or INNER_SENSOR
    inner_rank = graph.rank.get(inner, graph.rank.get(INNER_SENSOR))
    if inner_rank is None:
        return sensors
    out: list[str] = []
    for sensor in sensors:
        rank = graph.rank.get(sensor)
        if rank is None or int(rank) > int(inner_rank):
            out.append(sensor)
    return out


def _step_for_index(trace: Trace, idx: int) -> Optional[int]:
    step = trace.rec.get("step") if isinstance(trace.rec, dict) else None
    if step is None or len(step) <= int(idx):
        return None
    return int(np.asarray(step).astype(int)[int(idx)])


def _actual_expected_for_gap(
    trace: Trace,
    constant_control_expected: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(trace.actual, dtype=np.float64)
    expected = np.asarray(trace.expected, dtype=np.float64)
    n = min(len(actual), len(expected))
    if n <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    actual = actual[:n]
    expected = expected[:n]
    if constant_control_expected and trace.sensor in CONTROL_SENSORS:
        finite = expected[np.isfinite(expected)]
        if finite.size:
            expected = np.full_like(expected, float(np.nanmean(finite)))
    return actual, expected


def _abs_actual_expected_gap(trace: Trace) -> np.ndarray:
    actual, expected = _actual_expected_for_gap(trace)
    if not actual.size:
        return np.asarray([], dtype=np.float64)
    return np.nan_to_num(
        np.abs(actual - expected),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _evaluated_step_mask(trace: Trace, n: int) -> np.ndarray:
    """Mask samples that are visible/evaluated in tcn-explain style plots."""
    mask = np.ones(int(n), dtype=bool)
    step = trace.rec.get("step") if isinstance(trace.rec, dict) else None
    if step is not None and len(step) >= int(n):
        step_arr = np.asarray(step[:int(n)]).astype(int)
        mask &= ~np.isin(step_arr, [0, 1])
    return mask


def _peak_gap_index(trace: Trace, step: Optional[int] = None) -> int:
    gap = _abs_actual_expected_gap(trace)
    if not gap.size:
        return 0
    valid = _evaluated_step_mask(trace, len(gap))
    if step is not None:
        fallback_idx = (
            int(np.nanargmax(np.where(valid, gap, -np.inf)))
            if valid.any()
            else int(np.nanargmax(gap))
        )
        step_mask = _step_mask(trace.rec, int(step), fallback_idx)
        limited = np.zeros(len(gap), dtype=bool)
        n = min(len(limited), len(step_mask))
        limited[:n] = step_mask[:n]
        valid &= limited
    if not valid.any():
        valid = np.ones(len(gap), dtype=bool)
    local = np.where(valid, gap, -np.inf)
    return int(np.nanargmax(local))


def _peak_choice(names: list[str], traces: dict[str, Trace],
                 step: Optional[int] = None) -> PeakChoice | None:
    best: PeakChoice | None = None
    for sensor in _resolve_many(names, traces):
        trace = traces[sensor]
        gap = _abs_actual_expected_gap(trace)
        if not gap.size:
            continue
        idx = _peak_gap_index(trace, step=step)
        value = float(gap[idx])
        choice = PeakChoice(sensor=sensor, step=_step_for_index(trace, idx),
                            peak_index=idx, level=value)
        if best is None or choice.level > best.level:
            best = choice
    return best


def _peak_choices(names: list[str], traces: dict[str, Trace],
                  step: Optional[int] = None,
                  limit: int = 2) -> list[PeakChoice]:
    choices: list[PeakChoice] = []
    for sensor in _resolve_many(names, traces):
        trace = traces[sensor]
        gap = _abs_actual_expected_gap(trace)
        if not gap.size:
            continue
        idx = _peak_gap_index(trace, step=step)
        choices.append(PeakChoice(
            sensor=sensor,
            step=_step_for_index(trace, idx),
            peak_index=idx,
            level=float(gap[idx]),
        ))
    choices.sort(key=lambda choice: (-choice.level, str(choice.sensor)))
    return choices[:max(0, int(limit))]


def _step_level(trace: Trace, step: Optional[int]) -> float:
    level = np.nan_to_num(np.asarray(trace.level, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
    if not level.size:
        return 0.0
    if step is None:
        return float(np.nanmax(level))
    mask = _step_mask(trace.rec, int(step), int(np.nanargmax(level)))
    return float(np.nanmax(level[mask])) if mask.any() else float(np.nanmax(level))


def _step_signed_gap(trace: Trace, step: Optional[int],
                     constant_control_expected: bool = True) -> tuple[float, str]:
    summary = _step_actual_expected_summary(
        trace, step, constant_control_expected=constant_control_expected)
    if not summary:
        return 0.0, "실제=기대"
    return float(summary["gap"]), str(summary["direction"])


def _step_actual_expected_summary(
    trace: Trace,
    step: Optional[int],
    constant_control_expected: bool = True,
) -> dict[str, float | str] | None:
    actual = np.asarray(trace.actual, dtype=np.float64)
    expected = np.asarray(trace.expected, dtype=np.float64)
    if constant_control_expected and trace.sensor in CONTROL_SENSORS:
        finite = expected[np.isfinite(expected)]
        if finite.size:
            expected = np.full_like(expected, float(np.nanmean(finite)))
    if not actual.size or not expected.size:
        return None
    if step is None:
        mask = np.ones(len(actual), dtype=bool)
    else:
        diff = np.abs(actual - expected)
        finite_diff = np.where(np.isfinite(diff), diff, -np.inf)
        peak_idx = int(np.nanargmax(finite_diff)) if np.isfinite(finite_diff).any() else 0
        mask = _step_mask(trace.rec, int(step), peak_idx)
    valid = mask & np.isfinite(actual) & np.isfinite(expected)
    if not valid.any():
        return None
    actual_mean = float(np.nanmean(actual[valid]))
    expected_mean = float(np.nanmean(expected[valid]))
    gap = actual_mean - expected_mean
    if gap > 0:
        direction = "실제>기대"
    elif gap < 0:
        direction = "실제<기대"
    else:
        direction = "실제=기대"
    return {
        "actual": actual_mean,
        "expected": expected_mean,
        "gap": gap,
        "direction": direction,
    }


def _valve_open(rec: dict, valve: str, step: Optional[int]) -> bool:
    io = np.asarray(rec.get("io", []), dtype=np.float64)
    cols = [str(x) for x in rec.get("io_cols", [])]
    if io.ndim != 2 or valve not in cols:
        return False
    j = cols.index(valve)
    if step is not None and rec.get("step") is not None and len(rec["step"]) == io.shape[0]:
        mask = np.asarray(rec["step"]).astype(int) == int(step)
    else:
        mask = np.ones(io.shape[0], dtype=bool)
    vals = io[mask, j]
    return bool(vals.size and np.nanmax(vals) > 0.0)


def _valve_number_from_col(col: str) -> str:
    name = str(col).strip()
    if name.isdigit():
        return name
    digits = "".join(ch for ch in name if ch.isdigit())
    return digits if len(digits) >= 5 else ""


def _open_valve_numbers(rec: dict, step: Optional[int]) -> list[str]:
    io = np.asarray(rec.get("io", []), dtype=np.float64)
    cols = [str(x) for x in rec.get("io_cols", [])]
    if io.ndim != 2 or not cols:
        return []
    if step is not None and rec.get("step") is not None and len(rec["step"]) == io.shape[0]:
        mask = np.asarray(rec["step"]).astype(int) == int(step)
    else:
        mask = np.ones(io.shape[0], dtype=bool)
    if not mask.any():
        return []
    opened: list[str] = []
    seen: set[str] = set()
    for j, col in enumerate(cols):
        number = _valve_number_from_col(col)
        if not number or number in seen:
            continue
        vals = io[mask, j]
        finite = vals[np.isfinite(vals)]
        if finite.size and float(np.nanmax(finite)) > 0.0:
            opened.append(number)
            seen.add(number)
    return opened


def _open_valves_label(rec: dict, step: Optional[int]) -> str:
    numbers = _open_valve_numbers(rec, step)
    if not numbers:
        return "열림: 없음"
    return "열림: " + ", ".join(numbers)


def _last_finite(values) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return None
    return float(finite[-1])


def _tcn_explain_sensor_accum(trace: Trace, threshold=None,
                              score_kind: str | None = None) -> dict:
    from .recon_explain import (
        _curve_to_view_full,
        _plot_time_filter,
        _sensor_reconstruction_accum_series_from_level,
        _step_accum_series_from_level,
        _uses_step_accum_threshold,
    )

    t_full = int(len(trace.actual))
    plot_keep, _orig_idx, _t, step_plot, _bounds, _spans = _plot_time_filter(
        trace.model, trace.rec, t_full)
    use_step_threshold = (
        threshold is not None and _uses_step_accum_threshold(threshold, score_kind)
    )
    idx = int(trace.sensor_index)
    raw_rec = trace.rec.get("_source_rec") if isinstance(trace.rec, dict) else None
    if raw_rec is None:
        raw_rec = trace.rec
    accum = None
    raw_level_col = None
    source = "plot-level"
    if raw_rec is not None and hasattr(trace.model, "error_level"):
        raw_level = np.asarray(trace.model.error_level(raw_rec), dtype=np.float64)
        if raw_level.ndim == 2 and 0 <= idx < raw_level.shape[1]:
            raw_level_col = np.asarray(raw_level[:, idx], dtype=np.float64)
            raw_step = raw_rec.get("step") if isinstance(raw_rec, dict) else None
            if use_step_threshold:
                raw_curve = _step_accum_series_from_level(
                    raw_level_col, raw_step, threshold_base=1.0)
                source = "tcn-explain-raw-step-excess"
            else:
                raw_curve = _sensor_reconstruction_accum_series_from_level(
                    raw_level_col, raw_step)
                source = "tcn-explain-raw-reconstruction"
            curve_full = _curve_to_view_full(trace.rec, raw_curve)
            if len(curve_full) == t_full:
                accum = np.asarray(curve_full, dtype=np.float64)[plot_keep]
    if accum is None:
        level = np.asarray(trace.level, dtype=np.float64)
        zc = level[plot_keep] if len(level) == t_full else level
        if use_step_threshold:
            accum = _step_accum_series_from_level(zc, step_plot, threshold_base=1.0)
            source = "tcn-explain-plot-step-excess"
        else:
            accum = _sensor_reconstruction_accum_series_from_level(zc, step_plot)
            source = "tcn-explain-plot-reconstruction"
    accum = np.asarray(accum, dtype=np.float64)
    finite_accum = accum[np.isfinite(accum)]
    final = float(finite_accum[-1]) if finite_accum.size else None
    if raw_level_col is None:
        level = np.asarray(trace.level, dtype=np.float64)
        raw_level_col = level[plot_keep] if len(level) == t_full else level
    raw_level_col = np.nan_to_num(
        np.asarray(raw_level_col, dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0)
    plot_level = np.asarray(trace.level, dtype=np.float64)
    plot_level = plot_level[plot_keep] if len(plot_level) == t_full else plot_level
    plot_level = np.nan_to_num(plot_level, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "curve": accum,
        "final_score": 0.0 if final is None else final,
        "source": source,
        "step_reset": bool(use_step_threshold),
        "raw_peak_level": float(np.nanmax(raw_level_col)) if raw_level_col.size else 0.0,
        "plot_peak_level": float(np.nanmax(plot_level)) if plot_level.size else 0.0,
        "level_sum": float(np.maximum(0.0, raw_level_col).sum()) if raw_level_col.size else 0.0,
        "nonzero_count": int(np.count_nonzero(np.abs(raw_level_col) > 1e-12)),
    }


def _sensor_accum_final_from_trace(trace: Trace, threshold=None,
                                   score_kind: str | None = None) -> float:
    summary = _tcn_explain_sensor_accum(trace, threshold=threshold, score_kind=score_kind)
    return float(summary.get("final_score") or 0.0)


def _score_kind_from_search_best(best: dict, threshold=None) -> str | None:
    score_kind = (
        best.get("score_name")
        or (best.get("evaluation_criteria") or {}).get("threshold_field")
        or (best.get("val") or {}).get("score_kind")
    )
    if score_kind is None and isinstance(threshold, dict):
        score_kind = threshold.get("score_kind")
    if score_kind is None:
        stored_threshold_wise = (best.get("val") or {}).get("threshold_wise")
        threshold_wise_from_threshold = (
            str(threshold.get("threshold_wise", "")).lower()
            if isinstance(threshold, dict) else ""
        )
        if str(stored_threshold_wise or "").lower() in {"all", "recipe", "sensor", "step"}:
            mode = str(stored_threshold_wise).lower()
            score_kind = (
                "step_accum_excess_reconstruction_error"
                if mode in {"sensor", "step"}
                else "sum_all_sensor_reconstruction_error"
            )
        elif threshold_wise_from_threshold in {"all", "recipe", "sensor", "step"}:
            score_kind = (
                "step_accum_excess_reconstruction_error"
                if threshold_wise_from_threshold in {"sensor", "step"}
                else "sum_all_sensor_reconstruction_error"
            )
    return None if score_kind is None else str(score_kind)


def _tcn_explain_pt_prediction(model_a, rec: dict, threshold,
                               score_kind: str | None = None,
                               saved_scores: dict | None = None) -> dict:
    """Mirror tcn-explain wafer NG prediction scale for RCA filtering."""
    if threshold is None:
        return {
            "predicted_ng": False,
            "score": None,
            "threshold": None,
            "source": "missing-threshold",
            "score_kind": score_kind,
        }
    from .recon_explain import _prediction_scores, _threshold_for_record

    wid = str(rec.get("wafer_id", ""))
    cutoff = _threshold_for_record(threshold, rec)
    score = None
    source = "tcn-explain-computed"
    if saved_scores and wid in saved_scores:
        try:
            score = float(saved_scores[wid])
            source = "search_best"
        except Exception:
            score = None
    if score is None:
        score = float(_prediction_scores(
            model_a, [rec], threshold=threshold, score_kind=score_kind)[0])
    predicted = bool(cutoff is not None and score >= float(cutoff))
    return {
        "predicted_ng": predicted,
        "score": score,
        "threshold": None if cutoff is None else float(cutoff),
        "source": source,
        "score_kind": score_kind,
    }


def _candidate_ok_dirs(cfg: IOConfig, data_root: Path | None = None) -> list[Path]:
    candidates = [Path(getattr(cfg, "ok_dir", ""))]
    if data_root is not None:
        root = Path(data_root)
        candidates.extend([
            _sub_ci(root.parent, "OK"),
            _sub_ci(root.parent, "ok"),
            _sub_ci(_sub_ci(root.parent, "train"), "OK"),
            _sub_ci(_sub_ci(root.parent, "train"), "ok"),
            _sub_ci(root, "OK"),
            _sub_ci(root, "ok"),
        ])
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate)
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def _load_tcn_explain_ok_records(cfg: IOConfig,
                                 data_root: Path | None = None) -> tuple[list, Path | None]:
    from .data import load_dataset

    for candidate in _candidate_ok_dirs(cfg, data_root):
        if not candidate.is_dir():
            continue
        try:
            records, _io_cols, _sensor_cols, _max_step = load_dataset(candidate, cfg)
        except Exception:
            continue
        if records:
            return records, candidate
    return [], None


def _build_inner_accum_reference(pt_model: ArtifactModel, cfg: IOConfig,
                                 data_root: Path | None = None,
                                 artifacts_dir: Path | None = None,
                                 threshold=None,
                                 score_kind: str | None = None) -> InnerAccumReference:
    from .recon_explain import load_or_build_ok_accum_bands

    ok_records, ok_root = _load_tcn_explain_ok_records(cfg, data_root=data_root)
    if not ok_records:
        return InnerAccumReference(
            INNER_SENSOR, {}, None, 0, source="missing-train-ok",
            threshold=threshold, score_kind=score_kind)
    cache_root = Path(artifacts_dir) if artifacts_dir is not None else Path(getattr(cfg, "artifacts_dir", "."))
    ok_bands = load_or_build_ok_accum_bands(
        pt_model.model,
        ok_records,
        cache_root / "ok_accum_bands.npz",
        threshold=threshold,
        score_kind=score_kind,
    )
    if not ok_bands:
        return InnerAccumReference(
            INNER_SENSOR, {}, None, len(ok_records), None,
            source=str(ok_root or "train-ok"),
            threshold=threshold, score_kind=score_kind)
    return InnerAccumReference(
        INNER_SENSOR, {}, None, len(ok_records), ok_bands,
        source=str(ok_root or "train-ok"),
        threshold=threshold, score_kind=score_kind)


def _inner_ok_stats_from_ok_bands(trace: Trace, ref: InnerAccumReference) -> tuple[dict | None, str]:
    if not ref.ok_bands:
        return None, ref.source or "missing-ok-band"
    from .recon_explain import (
        _lookup_ok_band,
        _lookup_ok_mean,
        _plot_time_filter,
        _uses_step_accum_threshold,
    )

    T = int(len(trace.actual))
    keep, _orig_idx, _t, step_plot, _bounds, spans = _plot_time_filter(
        trace.model, trace.rec, T)
    use_step_threshold = (
        ref.threshold is not None
        and _uses_step_accum_threshold(ref.threshold, ref.score_kind)
    )
    ok_mean = _lookup_ok_mean(
        ref.ok_bands, trace.rec, step_plot, spans,
        sensor_idx=int(trace.sensor_index), step_reset=use_step_threshold)
    ok_mean_final = None
    if ok_mean is not None:
        ok_mean = np.asarray(ok_mean, dtype=np.float64)
        valid_mean = ok_mean[np.isfinite(ok_mean)]
        if valid_mean.size:
            ok_mean_final = float(valid_mean[-1])
    lo, hi = _lookup_ok_band(
        ref.ok_bands, trace.rec, step_plot, spans,
        sensor_idx=int(trace.sensor_index))
    if lo is None or hi is None:
        if ok_mean_final is None:
            return None, "missing-recipe-ok-band"
        return {
            "ok_min": None,
            "ok_max": None,
            "ok_mean": ok_mean_final,
            "step_reset": bool(use_step_threshold),
        }, ref.source or "tcn-explain-ok-mean"
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if len(lo) == len(keep):
        valid = keep & np.isfinite(lo) & np.isfinite(hi)
    else:
        valid = np.isfinite(lo) & np.isfinite(hi)
    if not np.any(valid):
        if ok_mean_final is None:
            return None, "empty-recipe-ok-band"
        return {
            "ok_min": None,
            "ok_max": None,
            "ok_mean": ok_mean_final,
            "step_reset": bool(use_step_threshold),
        }, ref.source or "tcn-explain-ok-mean"
    return {
        "ok_min": float(_last_finite(lo[valid]) or np.nan),
        "ok_max": float(_last_finite(hi[valid]) or np.nan),
        "ok_mean": ok_mean_final,
        "step_reset": bool(use_step_threshold),
    }, ref.source or "tcn-explain-ok-band"


def _inner_status(trace: Trace | None, ref: InnerAccumReference,
                  recipe: str) -> dict:
    if trace is None:
        return {
            "status": "unknown",
            "final_score": None,
            "ok_min": None,
            "ok_max": None,
            "ok_mean": None,
            "source": "missing-inner-trace",
            "accum_source": None,
            "peak_level": None,
            "raw_peak_level": None,
            "plot_peak_level": None,
            "level_sum": None,
            "nonzero_count": 0,
        }
    accum = _tcn_explain_sensor_accum(
        trace, threshold=ref.threshold, score_kind=ref.score_kind)
    final_score = float(accum.get("final_score") or 0.0)
    signal = {
        "peak_level": accum.get("raw_peak_level", accum.get("plot_peak_level", 0.0)),
        "raw_peak_level": accum.get("raw_peak_level", 0.0),
        "plot_peak_level": accum.get("plot_peak_level", 0.0),
        "level_sum": accum.get("level_sum", 0.0),
        "nonzero_count": accum.get("nonzero_count", 0),
        "accum_source": accum.get("source", ""),
        "step_reset": accum.get("step_reset", False),
    }
    ok_stats, source = _inner_ok_stats_from_ok_bands(trace, ref)
    if ok_stats is None:
        return {
            "status": "unknown",
            "final_score": final_score,
            "ok_min": None,
            "ok_max": None,
            "ok_mean": None,
            "source": "missing-ok-reference",
            **signal,
        }
    ok_min = ok_stats.get("ok_min")
    ok_max = ok_stats.get("ok_max")
    ok_mean = ok_stats.get("ok_mean")
    ok_values = [final_score] + [
        v for v in (ok_min, ok_max, ok_mean)
        if v is not None
    ]
    if not ok_values or not all(np.isfinite(v) for v in ok_values):
        status = "unknown"
        source = f"{source}:non-finite"
    elif (
        abs(float(final_score)) <= 1e-12
        and (ok_min is None or abs(float(ok_min)) <= 1e-12)
        and (ok_max is None or abs(float(ok_max)) <= 1e-12)
        and (ok_mean is None or abs(float(ok_mean)) <= 1e-12)
        and int(signal.get("nonzero_count", 0) or 0) == 0
    ):
        status = "unknown"
        source = f"{source}:no-inner-error-signal"
    elif ok_min is None or ok_max is None:
        status = "unknown"
        source = f"{source}:missing-minmax"
    else:
        upper_hit = (
            final_score >= ok_max
            or np.isclose(final_score, ok_max, rtol=1e-9, atol=1e-12)
        )
        status = (
            "normal"
            if ok_min <= final_score < ok_max and not upper_hit
            else "abnormal"
        )
    return {
        "status": status,
        "final_score": final_score,
        "ok_min": ok_min,
        "ok_max": ok_max,
        "ok_mean": ok_mean,
        "source": source,
        **signal,
    }


def _side_score(side: str, traces: dict[str, Trace], step: Optional[int]) -> float:
    if side == "front":
        names = FRONT_LINE_BASE + UP_HEATERS
    else:
        names = BACK_LINE_BASE + BACK_FILL_OR_VESSEL + DOWN_HEATERS
    vals = [_step_level(traces[s], step) for s in _resolve_many(names, traces)]
    return max(vals or [0.0])


def _line_side_for_sensor(sensor: str | None) -> str | None:
    resolved = str(sensor or "")
    if resolved in FRONT_LINE_SET:
        return "front"
    if resolved in BACK_LINE_SET:
        return "back"
    return None


def _append_unique_sensor(names: list[str], sensor: str | None) -> None:
    if sensor and sensor not in names:
        names.append(sensor)


def _pt_sensors_from_route(route: list[str], traces: dict[str, Trace]) -> list[str]:
    return [
        sensor for sensor in _resolve_many(route, traces)
        if _family(sensor) in {"P", "T"}
    ]


def _peak_choice_sensors(names: list[str], traces: dict[str, Trace],
                         step: Optional[int], limit: int = 2) -> list[str]:
    return [choice.sensor for choice in _peak_choices(names, traces, step=step, limit=limit)]


def _ensure_front_vessel_for_front(route: list[str], side: str) -> None:
    if str(side or "").lower() == "front":
        _append_unique_sensor(route, FRONT_VESSEL)


def _evidence_bundle(side: str, traces: dict[str, Trace],
                     step: Optional[int],
                     include_supply: bool = True) -> dict[str, list[str]]:
    side_l = str(side or "").lower()
    if side_l == "front":
        vessel = _resolve_many([FRONT_VESSEL], traces)
        heater = _peak_choice_sensors(UP_HEATERS, traces, step, limit=2)
    elif side_l == "back":
        vessel = _peak_choice_sensors(BACK_FILL_OR_VESSEL, traces, step, limit=2)
        heater = _peak_choice_sensors(DOWN_HEATERS, traces, step, limit=2)
    else:
        vessel = []
        heater = _peak_choice_sensors(UP_HEATERS + DOWN_HEATERS, traces, step, limit=2)
    return {
        "supply": (
            _peak_choice_sensors(ABNORMAL_SUPPLY_CANDIDATES, traces, step, limit=2)
            if include_supply else []
        ),
        "side_vessel": vessel,
        "heater": heater,
        "outlet_pulsevent": _peak_choice_sensors(
            OUTLET_PULSEVENT_CANDIDATES, traces, step, limit=2),
    }


def _append_evidence_bundle(route: list[str], evidence: dict[str, list[str]]) -> None:
    for key in ("supply", "side_vessel", "heater", "outlet_pulsevent"):
        for sensor in evidence.get(key, []) or []:
            _append_unique_sensor(route, sensor)


def _append_display_sensor(out: list[str], traces: dict[str, Trace],
                           sensor: str | None) -> None:
    resolved = _resolve_trace_name(sensor or "", traces)
    if resolved is not None and resolved not in out:
        out.append(resolved)


def _append_display_many(out: list[str], traces: dict[str, Trace],
                         sensors: list[str], limit: int | None = None) -> None:
    count = 0
    for sensor in sensors or []:
        before = len(out)
        _append_display_sensor(out, traces, sensor)
        if len(out) > before:
            count += 1
        if limit is not None and count >= int(limit):
            break


def _compact_display_route(route: list[str], traces: dict[str, Trace],
                           evidence: dict[str, list[str]],
                           branch: str, side: str,
                           primary_peak: PeakChoice | None,
                           inner_sensor: str | None,
                           step: Optional[int]) -> list[str]:
    """Keep the plotted route focused on sensors that support the final call."""
    selected: list[str] = []
    side_l = str(side or "").lower()
    branch_l = str(branch or "").lower()
    primary_sensor = primary_peak.sensor if primary_peak is not None else None

    if branch_l == "inner-abnormal":
        _append_display_sensor(selected, traces, "h_0052_PV")
        _append_display_many(selected, traces, evidence.get("supply", []), limit=1)
        _append_display_sensor(selected, traces, "h_005_PV")
        _append_display_sensor(selected, traces, inner_sensor or INNER_SENSOR)
        if side_l == "front":
            _append_display_sensor(selected, traces, FRONT_VALVE)
            _append_display_sensor(selected, traces, FRONT_VESSEL)
        elif side_l == "back":
            _append_display_sensor(selected, traces, BACK_CONTROL)
            _append_display_sensor(selected, traces, BACK_VALVE)
            _append_display_many(selected, traces, evidence.get("side_vessel", []), limit=1)
        else:
            _append_display_many(selected, traces, evidence.get("side_vessel", []), limit=1)
        _append_display_many(selected, traces, evidence.get("outlet_pulsevent", []), limit=2)
        _append_display_many(selected, traces, evidence.get("heater", []), limit=2)
        min_count = DISPLAY_ROUTE_MIN_ABNORMAL
        max_count = DISPLAY_ROUTE_MAX_ABNORMAL
    else:
        if side_l == "front":
            _append_display_sensor(selected, traces, FRONT_FILTER)
            _append_display_sensor(selected, traces, FRONT_VALVE)
            _append_display_sensor(selected, traces, FRONT_VESSEL)
        elif side_l == "back":
            _append_display_sensor(selected, traces, BACK_CONTROL)
            _append_display_sensor(selected, traces, BACK_VALVE)
            _append_display_many(selected, traces, evidence.get("side_vessel", []), limit=1)
        _append_display_sensor(selected, traces, primary_sensor)
        _append_display_many(selected, traces, evidence.get("outlet_pulsevent", []), limit=2)
        _append_display_many(selected, traces, evidence.get("heater", []), limit=2)
        min_count = DISPLAY_ROUTE_MIN_NORMAL
        max_count = DISPLAY_ROUTE_MAX_NORMAL

    for sensor in _resolve_many(route, traces):
        if len(selected) >= min_count:
            break
        _append_display_sensor(selected, traces, sensor)
    return selected[:max_count]


def _normal_line_route(side: str, traces: dict[str, Trace],
                       step: Optional[int],
                       primary_peak: PeakChoice | None) -> list[str]:
    route: list[str] = []
    primary_sensor = primary_peak.sensor if primary_peak is not None else None
    primary_side = _line_side_for_sensor(primary_sensor)
    selected_side = primary_side or side
    if selected_side == "front":
        route.extend(FRONT_LINE_BASE)
        _ensure_front_vessel_for_front(route, "front")
        for up in _peak_choices(UP_HEATERS, traces, step=step, limit=2):
            _append_unique_sensor(route, up.sensor)
    else:
        route.extend(BACK_LINE_BASE)
        back_peak = _peak_choice(BACK_FILL_OR_VESSEL, traces, step=step)
        if back_peak:
            _append_unique_sensor(route, back_peak.sensor)
        for down in _peak_choices(DOWN_HEATERS, traces, step=step, limit=2):
            _append_unique_sensor(route, down.sensor)
    if primary_side is None:
        _append_unique_sensor(route, primary_sensor)
    return route


def _abnormal_downstream_pt_sensors(
    traces: dict[str, Trace],
    graph: SensorGraph,
    step: Optional[int],
    limit: int = 3,
) -> list[str]:
    inner_name = _resolve_trace_name(INNER_SENSOR, traces) or INNER_SENSOR
    inner_rank = graph.rank.get(inner_name, graph.rank.get(INNER_SENSOR))
    candidates: list[str] = []
    if inner_rank is not None:
        for sensor in traces:
            if sensor == inner_name or _family(sensor) not in {"P", "T"}:
                continue
            rank = graph.rank.get(sensor)
            if rank is not None and int(rank) > int(inner_rank):
                candidates.append(sensor)
    if not candidates:
        candidates = _resolve_many(ABNORMAL_DOWNSTREAM_PT_FALLBACK, traces)

    scored: list[tuple[float, int, str]] = []
    for sensor in _resolve_many(candidates, traces):
        score = _step_level(traces[sensor], step)
        rank = int(graph.rank.get(sensor, 10**9))
        scored.append((score, rank, sensor))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = [sensor for _score, _rank, sensor in scored[:max(0, int(limit))]]
    chosen.sort(key=lambda sensor: (int(graph.rank.get(sensor, 10**9)), sensor))
    return chosen


def _step_directions(names: list[str], traces: dict[str, Trace],
                     step: Optional[int]) -> dict[str, str]:
    out: dict[str, str] = {}
    for sensor in names:
        if sensor not in traces:
            continue
        _gap, direction = _step_signed_gap(traces[sensor], step)
        out[sensor] = direction
    return out


def _direction_items(names: list[str], traces: dict[str, Trace],
                     step: Optional[int]) -> list[dict]:
    items: list[dict] = []
    for sensor in _resolve_many(names, traces):
        gap, direction = _step_signed_gap(traces[sensor], step)
        items.append({
            "sensor": sensor,
            "family": "T" if sensor in VESSEL_HEAT_SENSORS else _family(sensor),
            "gap": float(gap),
            "direction": direction,
        })
    return items


def _non_equal_items(items: list[dict]) -> list[dict]:
    return [item for item in items if item.get("direction") != "실제=기대"]


def _all_direction(items: list[dict], direction: str, min_count: int = 1) -> bool:
    checked = _non_equal_items(items)
    return len(checked) >= int(min_count) and all(
        item.get("direction") == direction for item in checked)


def _any_family_direction(items: list[dict], family: str, direction: str) -> bool:
    return any(
        item.get("family") == family and item.get("direction") == direction
        for item in items
    )


def _format_direction_items(items: list[dict]) -> str:
    if not items:
        return "-"
    return ", ".join(
        f"{item['sensor']} {item['direction']}"
        for item in items
    )


def _direction_evidence_summary(evidence: dict[str, list[str]],
                                traces: dict[str, Trace],
                                step: Optional[int]) -> tuple[dict[str, list[dict]], str]:
    items = {
        key: _direction_items(names, traces, step)
        for key, names in (evidence or {}).items()
    }
    labels = {
        "supply": "supply",
        "side_vessel": "vessel/side",
        "heater": "heater/control",
        "outlet_pulsevent": "outlet/pulsevent",
    }
    summary = " | ".join(
        f"{labels.get(key, key)}: {_format_direction_items(value)}"
        for key, value in items.items()
    )
    return items, summary


def _all_actual_above_expected(names: list[str], traces: dict[str, Trace],
                               step: Optional[int]) -> bool:
    directions = list(_step_directions(names, traces, step).values())
    return len(directions) >= 2 and all(direction == "실제>기대" for direction in directions)


def _choose_side(rec: dict, traces: dict[str, Trace],
                 step: Optional[int]) -> tuple[str, str, dict]:
    front_open = _valve_open(rec, "9000003", step)
    back_open = _valve_open(rec, "9000004", step)
    front_score = _side_score("front", traces, step)
    back_score = _side_score("back", traces, step)
    if front_open and not back_open:
        side, reason = "front", "9000003 valve open"
    elif back_open and not front_open:
        side, reason = "back", "9000004 valve open"
    elif front_open and back_open:
        side = "front" if front_score >= back_score else "back"
        reason = "both valves open; higher side anomaly selected"
    else:
        side = "front" if front_score >= back_score else "back"
        reason = "no target valve open; higher side anomaly selected"
    return side, reason, {
        "9000003_open": front_open,
        "9000004_open": back_open,
        "front_score": front_score,
        "back_score": back_score,
    }


def _final_recommendation(route_order: list[str], traces: dict[str, Trace],
                          step: Optional[int], side: str, side_reason: str) -> str:
    for i, sensor in enumerate(route_order[:-1]):
        if sensor not in CONTROL_SENSORS or sensor not in traces:
            continue
        nxt = next((s for s in route_order[i + 1:] if s in traces), None)
        if not nxt:
            continue
        control_gap, control_dir = _step_signed_gap(traces[sensor], step)
        next_gap, next_dir = _step_signed_gap(traces[nxt], step)
        if "=" in control_dir or "=" in next_dir:
            continue
        same_direction = (control_gap > 0 and next_gap > 0) or (control_gap < 0 and next_gap < 0)
        if same_direction:
            return (
                f"{sensor}와 {nxt}가 같은 방향({control_dir}, {next_dir})으로 벗어납니다. "
                f"우선 제어변수 {sensor} 설정/제어 응답을 조정 대상으로 확인하세요."
            )
        return (
            f"{sensor}와 {nxt}가 반대 방향({control_dir}, {next_dir})으로 벗어납니다. "
            "제어변수 단독 조정보다는 해당 센서 구간과 연결된 valve/flow 경로를 우선 확인하세요."
        )
    if side in {"front", "back"}:
        valve = "9000003" if side == "front" else "9000004"
        return f"{side} 쪽 이상으로 판단했습니다({side_reason}). 관련 valve {valve}와 해당 라인 센서를 우선 확인하세요."
    return "제어변수-센서 방향성이 뚜렷하지 않습니다. 선택된 센서 경로와 valve 상태를 함께 확인하세요."


def _directional_recommendation(
    route_order: list[str],
    traces: dict[str, Trace],
    step: Optional[int],
    side: str,
    side_reason: str,
    evidence_items: dict[str, list[dict]],
) -> str:
    side_l = str(side or "").lower()
    valve = "9000003" if side_l == "front" else "9000004" if side_l == "back" else "front/back valve"
    supply = evidence_items.get("supply", [])
    side_vessel = evidence_items.get("side_vessel", [])
    outlet = evidence_items.get("outlet_pulsevent", [])
    heater = evidence_items.get("heater", [])
    process_pt = [item for item in side_vessel + outlet if item.get("family") in {"P", "T"}]

    priorities: list[str] = []
    if _all_direction(process_pt, "실제>기대", min_count=2):
        priorities.append(
            f"1순위: {valve} 개도 부족, {side_l or 'selected'} 라인 restriction, 또는 체류 증가를 확인하세요."
        )
        if _any_family_direction(heater, "T", "실제>기대") or _all_direction(heater, "실제>기대", min_count=1):
            priorities.append("2순위: heater/control 출력 과다 또는 제어 응답 지연을 확인하세요.")
        else:
            priorities.append("2순위: upstream supply보다 side vessel/outlet 체류 조건을 우선 확인하세요.")
    elif _all_direction(process_pt, "실제<기대", min_count=2):
        if _all_direction(supply, "실제<기대", min_count=1):
            priorities.append("1순위: supply 압력/유량 부족을 먼저 확인하세요.")
            priorities.append(f"2순위: {valve} 과개방 또는 downstream vent/exhaust 과개방 가능성을 확인하세요.")
        elif not supply:
            priorities.append(f"1순위: {valve} 과개방, flow 과다, 또는 downstream vent/exhaust 과개방 가능성을 확인하세요.")
            priorities.append("2순위: 선택된 downstream 라인의 vessel/outlet/pulsevent 센서를 우선 비교하세요.")
        else:
            priorities.append(f"1순위: {valve} 과개방, flow 과다, 또는 downstream vent/exhaust 과개방 가능성을 확인하세요.")
            priorities.append("2순위: supply 센서가 정상인지 확인해 유입 부족 가능성을 배제하세요.")
    elif (
        _any_family_direction(process_pt, "T", "실제<기대")
        and not _any_family_direction(process_pt, "P", "실제<기대")
    ):
        priorities.append("1순위: 압력보다 온도 저하가 두드러지므로 heater/control 출력 부족을 확인하세요.")
        priorities.append("2순위: 유량 증가로 인한 체류 시간 감소 여부를 함께 확인하세요.")
    elif _any_family_direction(process_pt, "P", "실제<기대"):
        if supply:
            priorities.append(f"1순위: 압력 저하가 보이므로 {valve} 과개방, supply 부족, 또는 vent/exhaust 과개방을 확인하세요.")
        else:
            priorities.append(f"1순위: 압력 저하가 보이므로 {valve} 과개방 또는 downstream vent/exhaust 과개방을 확인하세요.")
        priorities.append("2순위: 온도 방향과 함께 heater 영향 여부를 분리해서 확인하세요.")
    elif _any_family_direction(process_pt, "T", "실제>기대"):
        priorities.append("1순위: 온도 상승이 보이므로 heater/control 과다 또는 체류 증가를 확인하세요.")
        priorities.append(f"2순위: {valve} 개도 부족 또는 라인 restriction 여부를 확인하세요.")
    else:
        if supply:
            priorities.append("1순위: P/T 방향이 혼합되어 단일 원인 단정이 어렵습니다. supply, valve, heater, outlet/pulsevent 묶음을 순서대로 확인하세요.")
        else:
            priorities.append("1순위: P/T 방향이 혼합되어 단일 원인 단정이 어렵습니다. valve, heater, outlet/pulsevent 묶음을 순서대로 확인하세요.")
        priorities.append(f"2순위: {valve}와 선택된 {side_l or 'side'} 라인 sensor를 우선 비교하세요.")

    return " / ".join(priorities)


def _build_decision(rec: dict, traces: dict[str, Trace], graph: SensorGraph,
                    ref: InnerAccumReference) -> dict:
    recipe = str(rec.get("recipe", "unknown") or "unknown")
    inner_name = _resolve_trace_name(INNER_SENSOR, traces)
    inner_trace = traces.get(inner_name) if inner_name else None
    inner = _inner_status(inner_trace, ref, recipe)
    if inner_trace is not None:
        inner_peak = _peak_choice([inner_trace.sensor], traces)
    else:
        inner_peak = None
    route: list[str] = []
    notes: list[str] = []
    inner_status = str(inner.get("status", "unknown") or "unknown")
    branch = (
        "inner-abnormal" if inner_status == "abnormal"
        else "inner-normal" if inner_status == "normal"
        else "inner-unknown"
    )
    side = ""
    side_reason = ""
    side_info: dict = {}
    primary_peak = None
    step: Optional[int] = None
    rca_focus_sensor = inner_trace.sensor if inner_trace is not None else INNER_SENSOR
    rca_focus_basis = "t_002_PV 피크 step"
    abnormal_downstream_pt: list[str] = []
    abnormal_downstream_directions: dict[str, str] = {}
    h_004_adjust_recommended = False
    evidence: dict[str, list[str]] = {}

    if inner_status != "abnormal":
        primary_peak = _peak_choice(PRIMARY_NORMAL_CANDIDATES, traces)
        step = primary_peak.step if primary_peak else None
        if primary_peak is not None:
            rca_focus_sensor = primary_peak.sensor
        if inner_status == "normal":
            rca_focus_basis = (
                "t_002_PV 최종 누적복원오차가 tcn-explain OK band 범위라 "
                "downstream 후보 중 실제-기대 차이가 가장 큰 피크 step을 RCA focus로 사용"
            )
        else:
            rca_focus_basis = (
                "t_002_PV 최종 누적복원오차 판정 신호가 불충분해 "
                "downstream 후보 중 실제-기대 차이가 가장 큰 피크 step을 RCA focus로 사용"
            )
        side, side_reason, side_info = _choose_side(rec, traces, step)
        primary_side = _line_side_for_sensor(
            primary_peak.sensor if primary_peak is not None else None)
        if primary_side is not None and primary_side != side:
            side_reason = f"{side_reason}; focus sensor belongs to {primary_side} branch"
            side = primary_side
        route = _normal_line_route(side, traces, step, primary_peak)
        if inner_status == "normal":
            notes.append(
                "t_002_PV 최종 누적복원오차가 tcn-explain과 같은 OK 누적 band 안에 있어 "
                "t_002_PV의 특정 step을 focus로 쓰지 않습니다. RCA v2는 "
                "downstream vessel/outlet/pulse 후보 중 실제-기대 차이가 가장 큰 피크 step과 valve side rule을 따릅니다. "
                "h_003_PV 라인과 h_0053_PV 라인은 섞지 않고 선택된 한 라인만 표시하며, "
                "front 라인은 p_008_PV를 필수 표시하고, "
                "back 라인은 t_001_PV와 p_007_PV 중 더 이상한 하나만 표시합니다."
            )
        else:
            notes.append(
                "t_002_PV 최종 누적복원오차를 tcn-explain 방식으로 계산했지만 "
                "유효한 판정 신호가 없어 정상/비정상으로 단정하지 않습니다. RCA v2는 "
                "downstream vessel/outlet/pulse 후보 중 실제-기대 차이가 가장 큰 피크 step으로 fallback합니다. "
                "h_003_PV 라인과 h_0053_PV 라인은 섞지 않고 선택된 한 라인만 표시하며, "
                "front 라인은 p_008_PV를 필수 표시하고, "
                "back 라인은 t_001_PV와 p_007_PV 중 더 이상한 하나만 표시합니다."
            )
    else:
        primary_peak = inner_peak
        step = inner_peak.step if inner_peak else None
        if inner_trace is not None:
            rca_focus_sensor = inner_trace.sensor
        rca_focus_basis = (
            "t_002_PV 최종 누적복원오차가 tcn-explain OK band 밖이라 "
            "t_002_PV 실제-기대 차이가 가장 큰 피크 step을 RCA focus로 사용하고 upstream 흐름을 확인"
        )
        supply = _peak_choice(ABNORMAL_SUPPLY_CANDIDATES, traces, step=step)
        route = ["h_0052_PV"]
        if supply:
            route.append(supply.sensor)
        route.extend(["h_005_PV", INNER_SENSOR])
        side, side_reason, side_info = _choose_side(rec, traces, step)
        downstream_route = _normal_line_route(side, traces, step, primary_peak=None)
        abnormal_downstream_pt = _pt_sensors_from_route(downstream_route, traces)
        outlet_pulsevent_peaks = _peak_choices(
            OUTLET_PULSEVENT_CANDIDATES, traces, step=step, limit=2)
        for peak in outlet_pulsevent_peaks:
            _append_unique_sensor(downstream_route, peak.sensor)
            _append_unique_sensor(abnormal_downstream_pt, peak.sensor)
        if not abnormal_downstream_pt:
            abnormal_downstream_pt = _abnormal_downstream_pt_sensors(
                traces, graph, step, limit=3)
            for sensor in abnormal_downstream_pt:
                _append_unique_sensor(downstream_route, sensor)
        abnormal_downstream_directions = _step_directions(
            abnormal_downstream_pt, traces, step)
        h_004_adjust_recommended = _all_actual_above_expected(
            abnormal_downstream_pt, traces, step)
        route.extend(downstream_route)
        notes.append(
            "t_002_PV 최종 누적복원오차가 tcn-explain과 같은 OK 누적 band 밖에 있어 "
            "t_002_PV 실제-기대 차이가 가장 큰 피크 step을 기준으로 비정상을 만든 upstream "
            "inner-block supply/control 흐름을 확인합니다. 추가로 해당 step에서 열린 valve 기준으로 "
            "front/back side를 선택한 뒤 정상 branch와 같은 downstream side route를 표시하고, "
            "outlet/pulsevent 후보 중 실제-기대 차이가 큰 센서 2개까지 함께 표시합니다."
        )

    evidence = _evidence_bundle(
        side, traces, step, include_supply=(branch == "inner-abnormal"))
    _append_evidence_bundle(route, evidence)
    _ensure_front_vessel_for_front(route, side)
    if branch != "inner-abnormal":
        route = _after_inner_sensors(
            route, traces, graph,
            inner_trace.sensor if inner_trace is not None else INNER_SENSOR,
        )
    route_full_order = _ordered_sensors(route, traces, graph)
    display_route = _compact_display_route(
        route, traces, evidence, branch, side, primary_peak,
        inner_trace.sensor if inner_trace is not None else INNER_SENSOR,
        step,
    )
    route_order = _ordered_sensors(display_route, traces, graph)
    evidence_items, evidence_summary = _direction_evidence_summary(
        evidence, traces, step)
    recommendation = _directional_recommendation(
        route_order, traces, step, side, side_reason,
        evidence_items)
    if h_004_adjust_recommended and abnormal_downstream_pt:
        sensors_text = ", ".join(abnormal_downstream_pt)
        recommendation = (
            f"{recommendation} 뒷공정 P/T 센서({sensors_text})가 RCA step에서 "
            f"일관되게 실제>기대 방향입니다. {FRONT_VALVE_ADJUST_SENSOR} 센서/밸브 개도 조정을 우선 확인하세요."
        )
    return {
        "recipe": recipe,
        "branch": branch,
        "step": step,
        "inner": inner,
        "side": side,
        "side_reason": side_reason,
        "side_info": side_info,
        "primary_peak": None if primary_peak is None else primary_peak.__dict__,
        "inner_sensor": inner_trace.sensor if inner_trace is not None else INNER_SENSOR,
        "rca_focus_sensor": rca_focus_sensor,
        "rca_focus_basis": rca_focus_basis,
        "route_requested": route,
        "route_full_order": route_full_order,
        "route_order": route_order,
        "route_display_policy": (
            "conclusion-focused compact route: inner-normal/unknown shows only downstream sensors after t_002; "
            "inner-abnormal keeps upstream supply/control plus selected downstream sensors; full candidate route remains in route_full_order"
        ),
        "abnormal_downstream_pt_sensors": abnormal_downstream_pt,
        "abnormal_downstream_pt_directions": abnormal_downstream_directions,
        "h_004_adjust_recommended": h_004_adjust_recommended,
        "evidence_bundle": evidence,
        "direction_evidence": evidence_items,
        "direction_evidence_summary": evidence_summary,
        "notes": notes,
        "recommendation": recommendation,
    }


def _plot_rca_v2(out_path: Path, rec: dict, traces: dict[str, Trace],
                 decision: dict, ok_value_bands: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.patches import FancyArrowPatch
    from .recon_explain import (
        _draw_step_guides,
        _limit_series,
        _plot_time_filter,
        _setup_font,
        _time_axis_label,
    )

    _setup_font()
    import matplotlib.pyplot as plt

    route_sensors = [s for s in decision.get("route_order", []) if s in traces]
    inner_sensor = str(decision.get("inner_sensor") or "")
    if inner_sensor not in traces:
        resolved_inner = _resolve_trace_name(INNER_SENSOR, traces)
        inner_sensor = resolved_inner or ""
    inner = decision.get("inner", {}) or {}
    inner_status = str(inner.get("status", "unknown") or "unknown")
    if inner_sensor and inner_status != "abnormal":
        route_sensors = [s for s in route_sensors if s != inner_sensor]

    sensors: list[str] = []
    if inner_status != "abnormal" and inner_sensor in traces:
        sensors.append(inner_sensor)
    sensors.extend(s for s in route_sensors if s not in sensors)
    if inner_status == "abnormal" and inner_sensor in traces and inner_sensor not in sensors:
        sensors.append(inner_sensor)
    if not sensors:
        sensors = [next(iter(traces))]
    step = decision.get("step")
    focus_sensor = str(decision.get("rca_focus_sensor") or sensors[0])
    if focus_sensor in traces and focus_sensor not in sensors:
        sensors.append(focus_sensor)
    if focus_sensor not in traces or focus_sensor not in sensors:
        focus_sensor = sensors[0]
    max_rows = max(1, len(sensors))
    focus_trace = traces.get(focus_sensor, traces[sensors[0]])
    fig_h = min(26.5, max(5.8, 2.55 * max_rows + 1.35))
    header_ratio = 0.55
    fig = plt.figure(figsize=(18, fig_h))
    gs = fig.add_gridspec(
        max_rows + 1, 2,
        height_ratios=[header_ratio] + [1.0] * max_rows,
        width_ratios=[0.64, 0.36],
        hspace=0.36,
        wspace=0.02,
    )
    header_left_ax = fig.add_subplot(gs[0, 0])
    header_right_ax = fig.add_subplot(gs[0, 1])
    axes = []
    for idx in range(max_rows):
        axes.append(
            fig.add_subplot(
                gs[idx + 1, :],
                sharex=axes[0] if axes else None,
            )
        )

    route_pairs = {
        (route_sensors[i], route_sensors[i + 1])
        for i in range(max(0, len(route_sensors) - 1))
    }
    arrow_labels: list[tuple[int, str]] = []
    inner_status_labels: list[int] = []
    inner_status_label = {
        "normal": "정상",
        "abnormal": "비정상",
        "unknown": "판정불가",
    }.get(inner_status, "판정불가")
    inner_color = {
        "normal": "#047857",
        "abnormal": "#b91c1c",
        "unknown": "#6b7280",
    }.get(inner_status, "#6b7280")
    inner_status_text = f"t_002 {inner_status_label}"
    for row_idx, (ax, sensor) in enumerate(zip(axes, sensors)):
        trace = traces[sensor]
        t_full = int(len(trace.actual))
        plot_keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter(
            trace.model, trace.rec, t_full)
        if orig_idx.size == 0:
            ax.axis("off")
            continue
        cur = np.asarray(trace.actual, dtype=np.float64)[plot_keep]
        exp_raw = np.asarray(trace.expected, dtype=np.float64)[plot_keep]
        exp = exp_raw.copy()
        if sensor in CONTROL_SENSORS and np.isfinite(exp).any():
            exp = np.full_like(exp, float(np.nanmean(exp[np.isfinite(exp)])))
        ok_limit = _limit_series(trace.model, trace.rec, trace.sensor_index, t_full)[plot_keep]
        band_lo = exp - ok_limit
        band_hi = exp + ok_limit
        ok_mean, ok_mean_source = _lookup_ok_value_mean(ok_value_bands, trace, step_plot, spans)
        finite_vals = np.concatenate([
            cur[np.isfinite(cur)],
            exp[np.isfinite(exp)],
            band_lo[np.isfinite(band_lo)],
            band_hi[np.isfinite(band_hi)],
            np.asarray([], dtype=np.float64) if ok_mean is None else ok_mean[np.isfinite(ok_mean)],
        ])
        ylo, yhi = (float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))) if finite_vals.size else (0.0, 1.0)
        pad = (yhi - ylo) * 0.08 + 1e-6
        mask_full = _step_mask(trace.rec, step, _peak_gap_index(trace, step=step))
        mask_plot = mask_full[plot_keep] if len(mask_full) == t_full else np.zeros_like(cur, dtype=bool)
        step_spans = _mask_spans(mask_plot)
        for start, end in step_spans:
            ax.axvspan(float(start) - 0.5, float(end) + 0.5,
                       color="#f03b20", alpha=0.30 if sensor == focus_sensor else 0.11,
                       lw=0, zorder=0, label="RCA step" if sensor == focus_sensor else None)
        ax.fill_between(t, band_lo, band_hi, color="#9ca3af", alpha=0.16,
                        zorder=1, label="OK 기준 정상 범위" if row_idx == 0 else None)
        io_markers = _io_change_markers(trace.rec, plot_keep, mask_plot)
        _draw_io_change_markers(ax, io_markers, show_legend=(row_idx == 0))
        _draw_step_guides(ax, bounds, spans, label=True)
        ax.plot(t, exp, color="#1f77b4", lw=1.4, ls="--",
                label="기대 평균" if sensor in CONTROL_SENSORS and row_idx == 0 else
                ("기대" if row_idx == 0 else None), zorder=4)
        ax.plot(t, cur, color="#d62728", lw=1.4,
                label="실제" if row_idx == 0 else None, zorder=5)
        if ok_mean is not None and np.isfinite(ok_mean).any():
            label = "전체 OK 평균값" if ok_mean_source == "global" else "같은 recipe OK 평균값"
            ax.plot(t, ok_mean, color="#059669", lw=1.25, ls=(0, (5, 2)),
                    label=label if row_idx == 0 else None, zorder=3)
        gap, direction = _step_signed_gap(trace, step)
        if sensor == inner_sensor:
            inner_status_labels.append(row_idx)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_ylabel(("FOCUS\n" if sensor == focus_sensor else "") + sensor,
                      rotation=0, ha="right", va="center", fontsize=8,
                      fontweight="bold" if sensor == focus_sensor else "normal",
                      color="#b91c1c" if sensor == focus_sensor else "black")
        ax.tick_params(labelsize=8)
        ax.margins(x=0.01)
        if row_idx < len(sensors) - 1:
            next_sensor = sensors[row_idx + 1]
            allow_inner_arrow = sensor != inner_sensor or inner_status == "abnormal"
            if allow_inner_arrow and (sensor, next_sensor) in route_pairs:
                arrow_text = (
                    f"{sensor}: {inner_status_label}"
                    if sensor == inner_sensor and inner_status != "abnormal"
                    else f"{sensor}: {direction}"
                )
                arrow_labels.append((
                    row_idx,
                    arrow_text,
                ))

    if max_rows > 1:
        for ax in axes[:-1]:
            ax.label_outer()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="lower center",
                       bbox_to_anchor=(0.5, 1.03),
                       ncol=min(6, max(1, len(labels))),
                       fontsize=8, frameon=False, borderaxespad=0.0)
    axes[-1].set_xlabel(_time_axis_label(focus_trace.rec))
    step_text = "unknown" if step is None else str(step)
    detail = (
        f"RCA v2 | t_002={inner_status.upper()} | "
        f"final={_format_float(inner.get('final_score'))} | "
        f"OK={_format_float(inner.get('ok_min'))}~{_format_float(inner.get('ok_max'))} | "
        f"OKmean={_format_float(inner.get('ok_mean'))} | "
        f"focus={focus_sensor} | step={step_text} | branch={decision.get('branch')}"
    )
    last_sensor = sensors[-1]
    _last_gap, last_direction = _step_signed_gap(traces[last_sensor], step)
    last_footer_text = (
        f"{last_sensor}: {inner_status_label}"
        if last_sensor == inner_sensor and inner_status != "abnormal"
        else f"{last_sensor}: {last_direction}"
    )
    _draw_top_header(
        header_left_ax, header_right_ax, [],
        wafer_text=str(rec.get("wafer_id", "")),
        detail_text=detail, lines=[])
    fig.tight_layout(rect=(0.18, 0.13, 0.99, 0.99))
    fig.canvas.draw()
    for idx, text in arrow_labels:
        upper = axes[idx].get_position()
        lower = axes[idx + 1].get_position()
        x = (upper.x0 + upper.x1) / 2.0
        y0 = upper.y0 - 0.006
        y1 = lower.y1 + 0.006
        arrow = FancyArrowPatch((x, y0), (x, y1), transform=fig.transFigure,
                                arrowstyle="->", mutation_scale=22,
                                lw=1.7, color="#374151", alpha=0.90)
        fig.add_artist(arrow)
        fig.text(x + 0.010, (y0 + y1) / 2.0, text, ha="left", va="center",
                 fontsize=10.0, fontweight="bold", color="#374151",
                 bbox={"boxstyle": "round,pad=0.25", "fc": "white",
                       "ec": "#9ca3af", "lw": 0.7, "alpha": 0.92})
    for idx in inner_status_labels:
        ax_pos = axes[idx].get_position()
        fig.text(
            ax_pos.x0,
            min(0.985, ax_pos.y1 + 0.008),
            inner_status_text,
            transform=fig.transFigure,
            ha="left", va="bottom",
            fontsize=8.7, fontweight="bold",
            color=inner_color,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white",
                  "ec": inner_color, "lw": 0.9, "alpha": 0.94},
            zorder=12,
        )
    last_pos = axes[-1].get_position()
    fig.text(
        (last_pos.x0 + last_pos.x1) / 2.0,
        max(0.066, last_pos.y0 - 0.047),
        last_footer_text,
        transform=fig.transFigure,
        ha="center", va="top",
        fontsize=10.0, fontweight="bold",
        color="#374151",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white",
              "ec": "#9ca3af", "lw": 0.7, "alpha": 0.92},
    )
    recommendation = str(decision.get("recommendation", "") or "").strip()
    if recommendation:
        wrapped_recommendation = textwrap.fill(
            f"Recommendation: {recommendation}", width=132)
        fig.text(
            0.5, 0.026, wrapped_recommendation,
            transform=fig.transFigure,
            ha="center", va="bottom",
            fontsize=10.5, fontweight="bold",
            color="#111827",
            bbox={"boxstyle": "round,pad=0.42", "fc": "white",
                  "ec": "#374151", "lw": 0.8, "alpha": 0.96},
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_rca_v2_report(path: Path, rec: dict, decision: dict,
                         plot_path: Path) -> None:
    inner = decision.get("inner", {})
    side_info = decision.get("side_info", {}) or {}
    primary = decision.get("primary_peak") or {}
    lines = [
        f"# RCA v2 report - {rec.get('wafer_id', '-')}",
        "",
        "## 판정 요약",
        "",
        f"- Recipe: {rec.get('recipe', '-')}",
        f"- RCA step: {decision.get('step', '-')}",
        f"- RCA focus sensor: {decision.get('rca_focus_sensor', '-')}",
        f"- RCA focus basis: {decision.get('rca_focus_basis', '-')}",
        f"- Branch: {decision.get('branch', '-')}",
        f"- t_002_PV final accumulated score: {_format_float(inner.get('final_score'))}",
        f"- t_002_PV peak error level: {_format_float(inner.get('peak_level'))}",
        f"- t_002_PV raw peak error level: {_format_float(inner.get('raw_peak_level'))}",
        f"- t_002_PV plot peak error level: {_format_float(inner.get('plot_peak_level'))}",
        f"- t_002_PV error level nonzero count: {inner.get('nonzero_count', '-')}",
        f"- t_002_PV accum source: {inner.get('accum_source', '-')}",
        f"- TCN-explain OK band min/max: {_format_float(inner.get('ok_min'))} ~ {_format_float(inner.get('ok_max'))} ({inner.get('source', '-')})",
        f"- TCN-explain same-wafer-axis OK mean final: {_format_float(inner.get('ok_mean'))}",
        f"- t_002_PV status: {inner.get('status', '-')}",
        f"- t_002 diagnostic graph: {decision.get('inner_sensor', INNER_SENSOR)}",
        f"- Primary peak: {primary.get('sensor', '-')} step {primary.get('step', '-')} |actual-expected| {_format_float(primary.get('level'))}",
        f"- Abnormal downstream P/T sensors: {', '.join(decision.get('abnormal_downstream_pt_sensors', []) or []) or '-'}",
        f"- Abnormal downstream P/T directions: {decision.get('abnormal_downstream_pt_directions', {}) or '-'}",
        f"- Direction evidence summary: {decision.get('direction_evidence_summary', '-')}",
        f"- Evidence bundle: {decision.get('evidence_bundle', {}) or '-'}",
        f"- h_004 adjustment recommended: {decision.get('h_004_adjust_recommended', False)}",
        f"- Valve 9000003 open: {side_info.get('9000003_open', '-')}",
        f"- Valve 9000004 open: {side_info.get('9000004_open', '-')}",
        f"- Side: {decision.get('side', '-') or '-'} ({decision.get('side_reason', '-') or '-'})",
        f"- Recommendation: {decision.get('recommendation', '-')}",
        f"- Plot sensor policy: {decision.get('route_display_policy', '-')}",
        f"- Full candidate sensor count: {len(decision.get('route_full_order', []) or [])}",
        f"- Overlay image: {plot_path}",
        "",
        "## Plot sensors",
        "",
    ]
    for sensor in decision.get("route_order", []):
        lines.append(f"- {sensor}")
    lines.extend(["", "## Rule notes", ""])
    for note in decision.get("notes", []):
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_rca_v2_overlays(
    pt_artifacts: Path,
    h_artifacts: Path,
    data_dir: Path,
    out_dir: Path,
    cfg: IOConfig,
    wafer_csv: Optional[Path] = None,
    wafer_id: Optional[str] = None,
    prediction_filter: str = "predicted-ng",
    max_wafers: Optional[int] = None,
    graph_path: Optional[Path] = None,
) -> Path:
    pt_artifacts = _resolve_artifacts(Path(pt_artifacts))
    h_artifacts = _resolve_artifacts(Path(h_artifacts))
    models = [
        _load_artifact("PT", pt_artifacts, cfg),
        _load_artifact("H", h_artifacts, cfg),
    ]
    pt_best = _read_search_best(pt_artifacts)
    pt_threshold = pt_best.get("threshold", pt_best.get("val", {}).get("threshold_raw"))
    pt_score_kind = _score_kind_from_search_best(pt_best, pt_threshold)
    inner_score_kind = "sum_all_sensor_reconstruction_error"
    pt_scores = pt_best.get("scores") or pt_best.get("anomaly_scores") or {}
    if not isinstance(pt_scores, dict):
        pt_scores = {}
    cfg.sensors_order_path = Path(graph_path) if graph_path else cfg.sensors_order_path
    graph = load_sensor_graph(cfg, Path(graph_path) if graph_path else None)
    data_root, files = _csv_files(Path(data_dir), wafer_csv)
    ok_value_bands, ok_value_summary = _build_ok_value_bands(models, cfg, data_root=data_root)
    inner_ref = _build_inner_accum_reference(
        models[0],
        cfg,
        data_root=data_root,
        artifacts_dir=pt_artifacts,
        threshold=pt_threshold,
        score_kind=inner_score_kind,
    )
    if wafer_id:
        files = [p for p in files if Path(p).stem == wafer_id or wafer_id in Path(p).stem]
    if max_wafers is not None:
        files = files[:max(0, int(max_wafers))]
    if not files:
        raise SystemExit("No wafer CSV matched the requested filters.")

    out_dir = Path(out_dir)
    report_dir = out_dir / "rca_v2_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []
    summary = {
        "pt_artifacts": str(pt_artifacts),
        "h_artifacts": str(h_artifacts),
        "data_dir": str(data_root),
        "graph_path": str(Path(graph_path) if graph_path else cfg.sensors_order_path),
        "thresholds": {"PT": pt_threshold, "H": _read_search_threshold(h_artifacts)},
        "pt_score_kind": pt_score_kind,
        "inner_score_kind": inner_score_kind,
        "prediction_score_policy": "same as tcn-explain: search_best score reuse when available, otherwise recon_explain._prediction_scores with saved score_kind",
        "ok_value_mean": ok_value_summary,
        "inner_accum_reference": {
            "sensor": inner_ref.sensor,
            "ok_records": inner_ref.ok_records,
            "by_recipe": inner_ref.by_recipe,
            "global_range": inner_ref.global_range,
            "source": inner_ref.source,
            "score_kind": inner_ref.score_kind,
            "policy": "tcn-explain sensor reconstruction accumulation, final same-wafer plotted value against same-recipe OK min/max",
        },
        "prediction_filter": prediction_filter,
        "outputs": [],
        "reports": [],
    }
    for csv_path in files:
        model_records = _load_model_records(csv_path, data_root, cfg, models)
        rec = model_records.get("PT") or next(iter(model_records.values()))
        pt_rec = model_records.get(models[0].label, rec)
        pred = _tcn_explain_pt_prediction(
            models[0].model, pt_rec, pt_threshold,
            score_kind=pt_score_kind, saved_scores=pt_scores)
        if prediction_filter == "predicted-ng" and not pred["predicted_ng"]:
            skipped.append({
                "wafer_id": rec["wafer_id"],
                "source_path": rec.get("source_path", ""),
                "skip_reason": "pt-predicted-ok",
                "pt_score": pred["score"],
                "pt_threshold": pred["threshold"],
                "pt_score_source": pred["source"],
            })
            continue
        traces, skipped_traces = _merge_traces(models, model_records)
        if not traces:
            continue
        decision = _build_decision(rec, traces, graph, inner_ref)
        step_text = "nostep" if decision.get("step") is None else f"step{decision.get('step')}"
        filename = f"{_safe_filename(rec['wafer_id'])}__rca_v2__{step_text}.png"
        out_path = out_dir / filename
        _plot_rca_v2(out_path, rec, traces, decision, ok_value_bands=ok_value_bands)
        report_path = report_dir / f"{_safe_filename(rec['wafer_id'])}__rca_v2__{step_text}.md"
        _write_rca_v2_report(report_path, rec, decision, out_path)
        summary["outputs"].append(str(out_path))
        summary["reports"].append(str(report_path))
        rows.append({
            "wafer_id": rec["wafer_id"],
            "recipe": rec.get("recipe", ""),
            "source_path": rec.get("source_path", ""),
            "pt_score": "" if pred["score"] is None else round(float(pred["score"]), 6),
            "pt_threshold": "" if pred["threshold"] is None else round(float(pred["threshold"]), 6),
            "pt_score_source": pred.get("source", ""),
            "pt_score_kind": pred.get("score_kind", ""),
            "branch": decision.get("branch", ""),
            "step": "" if decision.get("step") is None else decision.get("step"),
            "rca_focus_sensor": decision.get("rca_focus_sensor", ""),
            "rca_focus_basis": decision.get("rca_focus_basis", ""),
            "inner_status": decision.get("inner", {}).get("status", ""),
            "inner_final_score": decision.get("inner", {}).get("final_score", ""),
            "inner_peak_level": decision.get("inner", {}).get("peak_level", ""),
            "inner_raw_peak_level": decision.get("inner", {}).get("raw_peak_level", ""),
            "inner_plot_peak_level": decision.get("inner", {}).get("plot_peak_level", ""),
            "inner_level_sum": decision.get("inner", {}).get("level_sum", ""),
            "inner_nonzero_count": decision.get("inner", {}).get("nonzero_count", ""),
            "inner_ok_min": decision.get("inner", {}).get("ok_min", ""),
            "inner_ok_max": decision.get("inner", {}).get("ok_max", ""),
            "inner_ok_mean": decision.get("inner", {}).get("ok_mean", ""),
            "inner_source": decision.get("inner", {}).get("source", ""),
            "inner_accum_source": decision.get("inner", {}).get("accum_source", ""),
            "side": decision.get("side", ""),
            "side_reason": decision.get("side_reason", ""),
            "route_order": ";".join(decision.get("route_order", [])),
            "route_full_count": len(decision.get("route_full_order", []) or []),
            "route_display_policy": decision.get("route_display_policy", ""),
            "abnormal_downstream_pt_sensors": ";".join(
                decision.get("abnormal_downstream_pt_sensors", []) or []),
            "abnormal_downstream_pt_directions": json.dumps(
                decision.get("abnormal_downstream_pt_directions", {}) or {},
                ensure_ascii=False),
            "evidence_bundle": json.dumps(
                decision.get("evidence_bundle", {}) or {}, ensure_ascii=False),
            "direction_evidence_summary": decision.get(
                "direction_evidence_summary", ""),
            "h_004_adjust_recommended": decision.get(
                "h_004_adjust_recommended", False),
            "recommendation": decision.get("recommendation", ""),
            "plot_path": str(out_path),
            "report_path": str(report_path),
        })
        if skipped_traces:
            summary.setdefault("skipped_trace_columns", []).extend({
                "wafer_id": rec["wafer_id"],
                **row,
            } for row in skipped_traces)

    csv_path = out_dir / "rca_v2_events.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "wafer_id", "recipe", "source_path", "pt_score", "pt_threshold",
            "pt_score_source", "pt_score_kind",
            "branch", "step", "rca_focus_sensor", "rca_focus_basis",
            "inner_status", "inner_final_score", "inner_peak_level",
            "inner_raw_peak_level", "inner_plot_peak_level",
            "inner_level_sum", "inner_nonzero_count",
            "inner_ok_min", "inner_ok_max", "inner_ok_mean",
            "inner_source", "inner_accum_source", "side", "side_reason",
            "route_order", "route_full_count", "route_display_policy",
            "abnormal_downstream_pt_sensors",
            "abnormal_downstream_pt_directions",
            "evidence_bundle", "direction_evidence_summary",
            "h_004_adjust_recommended",
            "recommendation", "plot_path", "report_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    skipped_path = out_dir / "rca_v2_skipped_wafers.csv"
    with open(skipped_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "wafer_id", "source_path", "skip_reason",
            "pt_score", "pt_threshold", "pt_score_source",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped)
    summary["event_csv"] = str(csv_path)
    summary["skipped_csv"] = str(skipped_path)
    (out_dir / "rca_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"RCA v2 images: {len(summary['outputs'])} -> {out_dir}")
    print(f"RCA v2 event manifest: {csv_path}")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Create RCA v2 rule-based overlays.")
    ap.add_argument("--pt-artifacts", required=True)
    ap.add_argument("--h-artifacts", required=True)
    ap.add_argument("--data-dir", default="/home/yapark/semes/dataset/val")
    ap.add_argument("--wafer-csv", default=None)
    ap.add_argument("--wafer-id", default=None)
    ap.add_argument("--out-dir", default="reports/rca_v2")
    ap.add_argument("--include-all-wafers", action="store_true")
    ap.add_argument("--max-wafers", type=int, default=None)
    ap.add_argument("--graph-path", default=None)
    ap.add_argument("--h-cart-mode", choices=["include", "exclude"], default="include")
    ap.add_argument("--future-h-mode", choices=["input", "predict"], default="input")
    ap.add_argument("--virtual-sensor-mode", choices=["off", "input", "target", "both"],
                    default="off")
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = IOConfig()
    cfg.h_cart_mode = args.h_cart_mode
    cfg.future_h_mode = args.future_h_mode
    cfg.virtual_sensor_mode = args.virtual_sensor_mode
    generate_rca_v2_overlays(
        pt_artifacts=Path(args.pt_artifacts),
        h_artifacts=Path(args.h_artifacts),
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        cfg=cfg,
        wafer_csv=Path(args.wafer_csv) if args.wafer_csv else None,
        wafer_id=args.wafer_id,
        prediction_filter="all" if args.include_all_wafers else "predicted-ng",
        max_wafers=args.max_wafers,
        graph_path=Path(args.graph_path) if args.graph_path else None,
    )


if __name__ == "__main__":
    main()
