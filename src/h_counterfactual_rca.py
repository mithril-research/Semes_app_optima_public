"""Counterfactual RCA for H control-variable adjustment candidates.

This module treats h_005-style H control variables as intervention inputs.
It sweeps whole-wafer deltas, reruns the existing PT TCN model, and reports
which adjustment candidates reduce P/T reconstruction error over the full
process. The focus ProcessStep is still reported as diagnostic context and
matches the slope-valve RCA rule: the ProcessStep with the largest slope in the
PT all-sensor accumulated reconstruction-error curve.
"""
from __future__ import annotations


import argparse
import csv
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .sensor_schema import in_group, paired_zone
from .config import IOConfig
from .related_overlay import (
    _csv_files,
    _load_artifact,
    _load_model_records,
    _load_record_exact,
    _model_record_columns,
    _ok_csv_files,
    _read_search_best,
    _resolve_artifacts,
    _safe_filename,
)
from .rca_slope_valve import (
    StepSlope,
    SUSPECT_VALVE_TOP_K,
    _all_sensor_accumulation,
    _parse_exclude_steps,
    _step_slope_rows,
    _valve_text,
)
from .rca_v2 import _score_kind_from_search_best, _tcn_explain_pt_prediction
from .sensor_graph import load_sensor_graph


def _influential_valve_text(peak, top_k: int = SUSPECT_VALVE_TOP_K) -> str:
    """focus step 의 '영향력이 큰 밸브만' 표시(전체 나열 방지 — slope-valve RCA 와 일관).

    peak.open_valves 는 그래프 우선순위(상류=영향 큼)로 정렬돼 있어 앞쪽 top_k 가 영향력 상위다.
    (기존엔 _valve_text(peak.open_valves) 로 열린 밸브를 전부 나열해 slope-valve RCA 와 어긋났음)
    """
    vals = [str(v) for v in (getattr(peak, "open_valves", None) or []) if str(v)]
    if not vals:
        return "-"
    k = max(1, int(top_k))
    return _valve_text(vals[:k])


DEFAULT_H_CONTROLS = (
    "h_005_PV",
    "h_0052_PV",
    "h_0053_PV",
)
DEFAULT_DELTA_GRID = "-5,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,5"
DEFAULT_STEP_RADIUS = 3
ALL_STEPS_LABEL = "전체 step"
EPS = 1e-12


def _compute_model(model, rec):
    from .recon_explain import _compute

    return _compute(model, rec)


def _compact_name(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def _is_h_control(sensor: str) -> bool:
    return in_group(sensor, "control")


def _is_h_response(sensor: str) -> bool:
    return in_group(sensor, "response")


def _is_pt_sensor(sensor: str) -> bool:
    low = str(sensor).strip().lower()
    return (
        bool(re.match(r"^p(\d+|[_-])", low))
        or low.startswith("p_")
        or bool(re.match(r"^t(\d+|[_-])", low))
        or low.startswith("t_")
    )


def _parse_sensor_list(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_H_CONTROLS)
    items = [x.strip() for x in re.split(r"[,/+\n\r\t]+", str(value)) if x.strip()]
    return items or list(DEFAULT_H_CONTROLS)


def _parse_delta_grid(value: str | None) -> list[float]:
    text = str(value or DEFAULT_DELTA_GRID)
    out = []
    for part in re.split(r"[,/+\s]+", text):
        if not part.strip():
            continue
        try:
            out.append(float(part))
        except Exception:
            raise ValueError(f"Invalid delta value: {part}") from None
    if not out:
        out = [0.0]
    if not any(abs(x) <= EPS for x in out):
        out.append(0.0)
    return sorted(set(float(x) for x in out))


def _format_float(value, digits: int = 3) -> str:
    if value in ("", None):
        return "-"
    try:
        v = float(value)
    except Exception:
        return "-"
    if not np.isfinite(v):
        return "-"
    return f"{v:.{int(digits)}g}"


def _format_pct(value, digits: int = 1) -> str:
    if value in ("", None):
        return "-"
    try:
        v = float(value)
    except Exception:
        return "-"
    if not np.isfinite(v):
        return "-"
    return f"{v:.{int(digits)}f}%"


def _as_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _as_optional_float(value) -> float:
    if value in ("", None):
        return float("nan")
    try:
        v = float(value)
    except Exception:
        return float("nan")
    return float(v) if np.isfinite(v) else float("nan")


def _short_text(value, limit: int = 120) -> str:
    text = " ".join(str(value if value is not None else "-").split())
    if len(text) <= int(limit):
        return text
    return text[:max(1, int(limit) - 3)].rstrip() + "..."


def _is_all_steps(value) -> bool:
    return value in (None, "", "-", ALL_STEPS_LABEL, "전체")


def _step_key(step) -> str | int:
    if _is_all_steps(step):
        return ALL_STEPS_LABEL
    return int(step)


def _step_label(step) -> str:
    if _is_all_steps(step):
        return ALL_STEPS_LABEL
    return str(int(step))


def _record_sensor_index(rec: dict, sensor: str) -> int | None:
    cols = [str(c) for c in rec.get("sensor_cols", [])]
    try:
        return cols.index(str(sensor))
    except ValueError:
        return None


def _step_mask(rec: dict, step: Optional[int], window_steps: int = 0) -> np.ndarray:
    n = int(np.asarray(rec.get("sensors")).shape[0])
    if _is_all_steps(step) or rec.get("step") is None or len(rec.get("step")) != n:
        return np.ones(n, dtype=bool)
    st = np.asarray(rec.get("step")).astype(int)
    return np.abs(st - int(step)) <= max(0, int(window_steps))


def _copy_record_with_deltas(rec: dict, deltas: dict[str, float],
                             step: Optional[int], window_steps: int = 0) -> dict:
    out = dict(rec)
    sensors = np.asarray(rec.get("sensors"), dtype=np.float64).copy()
    for sensor, delta in deltas.items():
        idx = _record_sensor_index(rec, sensor)
        if idx is None:
            continue
        sensor_step = step.get(sensor, None) if isinstance(step, dict) else step
        mask = _step_mask(rec, sensor_step, window_steps)
        sensors[mask, idx] = sensors[mask, idx] + float(delta)
    out["sensors"] = sensors
    return out


def _target_indices(model, predicate) -> list[int]:
    cols = [str(c) for c in getattr(model, "sensor_cols", []) or []]
    return [i for i, name in enumerate(cols) if predicate(name)]


def _score_model(model, rec: dict, predicate, step: Optional[int],
                 window_steps: int = 0) -> dict:
    actual, expected, level = _compute_model(model, rec)
    level = np.nan_to_num(np.asarray(level, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
    indices = _target_indices(model, predicate)
    if not indices:
        return {"score": 0.0, "max_level": 0.0, "sensor_count": 0, "sensors": []}
    mask = _step_mask(rec, step, window_steps)
    if not mask.any():
        mask = np.ones(level.shape[0], dtype=bool)
    sub = level[mask][:, indices]
    return {
        "score": float(np.nansum(sub)),
        "max_level": float(np.nanmax(sub)) if sub.size else 0.0,
        "sensor_count": len(indices),
        "sensors": [str(getattr(model, "sensor_cols", [])[i]) for i in indices],
    }


def _accum_from_model(model, rec: dict, predicate) -> tuple[np.ndarray, list[str]]:
    _actual, _expected, level = _compute_model(model, rec)
    level = np.nan_to_num(np.asarray(level, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
    indices = _target_indices(model, predicate)
    if not indices:
        return np.asarray([], dtype=np.float64), []
    per_t = np.nansum(level[:, indices], axis=1)
    return np.cumsum(np.maximum(0.0, per_t)), [
        str(getattr(model, "sensor_cols", [])[i]) for i in indices
    ]


def _focus_step(pt_model, pt_rec: dict, h_model, h_rec: dict,
                graph, exclude_steps: set[int]) -> dict | None:
    if pt_model is None or pt_rec is None:
        return None
    accum_data = _all_sensor_accumulation(pt_model, pt_rec)
    accum = np.asarray(accum_data.get("accum", []), dtype=np.float64)
    step_plot = np.asarray(accum_data.get("step_plot", []), dtype=int)
    if not accum.size or not step_plot.size:
        return None
    view_rec = accum_data.get("rec") or pt_rec
    rows = _step_slope_rows(accum, step_plot, view_rec, graph, exclude_steps)
    if not rows:
        return None
    peak = rows[0]
    return {
        "source": "PT all-sensor accumulated-error slope",
        "peak": peak,
        "step_rows": rows,
        "sensors": [str(x) for x in accum_data.get("sensor_cols", [])],
        "accum_data": accum_data,
    }


def _candidate_adjust_steps(rec: dict, focus_step: int,
                            step_radius: int,
                            exclude_steps: set[int]) -> list[int]:
    step_arr = rec.get("step") if isinstance(rec, dict) else None
    if step_arr is None:
        return [int(focus_step)]
    try:
        raw_steps = [int(x) for x in np.asarray(step_arr).astype(int).tolist()]
    except Exception:
        return [int(focus_step)]
    if not raw_steps:
        return [int(focus_step)]
    unique = sorted(set(raw_steps))
    radius = int(step_radius)
    if radius >= 0:
        candidates = [
            step for step in unique
            if abs(int(step) - int(focus_step)) <= radius
        ]
    else:
        candidates = list(unique)
    candidates = [
        int(step) for step in candidates
        if int(step) not in set(exclude_steps or set()) or int(step) == int(focus_step)
    ]
    if int(focus_step) not in candidates:
        candidates.append(int(focus_step))
    candidates = sorted(set(candidates), key=lambda step: (abs(int(step) - int(focus_step)), int(step)))
    return candidates


def _step_relation(adjust_step, focus_step) -> str:
    if _is_all_steps(adjust_step):
        return ALL_STEPS_LABEL
    try:
        diff = int(adjust_step) - int(focus_step)
    except Exception:
        return "-"
    if diff == 0:
        return "동일"
    if diff < 0:
        return f"focus {abs(diff)} step 전"
    return f"focus {diff} step 후"


def _model_can_accept_control(model, sensor: str) -> bool:
    input_cols = {str(c) for c in getattr(model, "input_sensor_cols", []) or []}
    target_cols = {str(c) for c in getattr(model, "sensor_cols", []) or []}
    return str(sensor) in input_cols and str(sensor) not in target_cols


def _available_controls(models: dict[str, object], records: dict[str, dict],
                        requested: list[str]) -> list[dict]:
    out = []
    pt_model = models.get("PT")
    pt_rec = records.get("PT")
    for sensor in requested:
        if pt_model is None or pt_rec is None:
            continue
        if _record_sensor_index(pt_rec, sensor) is None:
            continue
        if not _model_can_accept_control(pt_model, sensor):
            continue
        labels = []
        for label, model in models.items():
            rec = records.get(label)
            if rec is None:
                continue
            if _record_sensor_index(rec, sensor) is None:
                continue
            if _model_can_accept_control(model, sensor):
                labels.append(label)
        out.append({"sensor": sensor, "models": labels or ["PT"]})
    return out


def _evaluate_counterfactual(models: dict[str, object], records: dict[str, dict],
                             deltas: dict[str, float], score_step,
                             adjust_steps: dict[str, object],
                             window_steps: int, base_scores: dict,
                             h_weight: float, pt_weight: float,
                             control_penalty: float,
                             delta_scale: float) -> dict:
    rec_pt = records.get("PT")
    model_pt = models.get("PT")

    pt_deltas = {
        sensor: delta for sensor, delta in deltas.items()
        if rec_pt is not None and model_pt is not None
        and _model_can_accept_control(model_pt, sensor)
        and _record_sensor_index(rec_pt, sensor) is not None
    }

    pt_adjust_steps = {
        sensor: adjust_steps.get(sensor, score_step)
        for sensor in pt_deltas
    }
    rec_pt_cf = (
        _copy_record_with_deltas(rec_pt, pt_deltas, pt_adjust_steps, window_steps)
        if rec_pt is not None and pt_deltas else rec_pt
    )

    pt_score = _score_model(model_pt, rec_pt_cf, _is_pt_sensor, score_step, window_steps) if model_pt and rec_pt_cf else {}
    pt_value = float(pt_score.get("score", 0.0) or 0.0)
    pt_base = float(base_scores.get("pt_score", 0.0) or 0.0)
    pt_applicable = bool(pt_deltas) and pt_base > EPS
    pt_norm = pt_value / max(pt_base, EPS) if pt_applicable else 0.0
    penalty = float(control_penalty) * sum(abs(float(x)) for x in deltas.values()) / max(float(delta_scale), EPS)
    objective = (
        (float(pt_weight) * pt_norm if pt_applicable else 0.0)
        + penalty
    )
    baseline_objective = float(pt_weight) if pt_applicable else 0.0
    return {
        "deltas": {k: float(v) for k, v in deltas.items()},
        "score_step": _step_key(score_step),
        "score_scope": ALL_STEPS_LABEL if _is_all_steps(score_step) else "focus step window",
        "adjust_steps": {k: _step_key(v) for k, v in pt_adjust_steps.items()},
        "adjust_step": (
            _step_key(next(iter(pt_adjust_steps.values())))
            if len(set(pt_adjust_steps.values())) == 1 and pt_adjust_steps else ""
        ),
        "pt_score": pt_value,
        "pt_applicable": bool(pt_applicable),
        "pt_improvement_pct": (
            (pt_base - pt_value) / pt_base * 100.0 if pt_applicable else None
        ),
        "objective": objective,
        "objective_improvement_pct": (
            (baseline_objective - objective) / baseline_objective * 100.0
            if baseline_objective > EPS else 0.0
        ),
        "penalty": penalty,
        "mutated_models": ["PT"] if pt_deltas else [],
    }


def _candidate_step_distance(row: dict) -> int:
    if _is_all_steps(row.get("score_step")) or _is_all_steps(row.get("adjust_step")):
        return 0
    try:
        score_step = int(row.get("score_step"))
    except Exception:
        return 10**9
    steps = row.get("adjust_steps") or {}
    if not steps:
        try:
            if _is_all_steps(row.get("adjust_step")):
                return 0
            return abs(int(row.get("adjust_step")) - score_step)
        except Exception:
            return 10**9
    if any(_is_all_steps(step) for step in steps.values()):
        return 0
    return max(abs(int(step) - score_step) for step in steps.values())


def _choose_recommendation(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    baseline = next((row for row in candidates if all(abs(v) <= EPS for v in row["deltas"].values())), None)
    baseline_objective = float(baseline.get("objective", 0.0)) if baseline else float("inf")
    better = [row for row in candidates if float(row.get("objective", np.inf)) < baseline_objective - EPS]
    if not better:
        return min(candidates, key=lambda row: (
            float(row.get("objective", np.inf)),
            sum(abs(v) for v in row["deltas"].values()),
            _candidate_step_distance(row),
        ))
    best = min(better, key=lambda row: float(row.get("objective", np.inf)))
    best_gain = baseline_objective - float(best["objective"])
    target_objective = baseline_objective - 0.9 * best_gain
    near_best = [
        row for row in better
        if float(row.get("objective", np.inf)) <= target_objective + EPS
    ]
    return min(near_best, key=lambda row: (
        sum(abs(v) for v in row["deltas"].values()),
        _candidate_step_distance(row),
        float(row.get("objective", np.inf)),
    ))


def _choose_pt_recommendation(candidates: list[dict]) -> dict | None:
    """Pick the visible single-control recommendation from the P/T gain curve."""
    scored = [
        row for row in candidates
        if row.get("pt_applicable")
        and _as_optional_float(row.get("pt_improvement_pct")) is not None
    ]
    if not scored:
        return _choose_recommendation(candidates)

    baseline = next(
        (row for row in scored if all(abs(v) <= EPS for v in row["deltas"].values())),
        None,
    )
    baseline_gain = float(baseline.get("pt_improvement_pct", 0.0) or 0.0) if baseline else 0.0
    better = [
        row for row in scored
        if float(row.get("pt_improvement_pct", -np.inf)) > baseline_gain + EPS
    ]
    if not better:
        return _choose_recommendation(candidates)

    best_gain = max(float(row.get("pt_improvement_pct", -np.inf)) for row in better)
    target_gain = baseline_gain + 0.9 * (best_gain - baseline_gain)
    near_best = [
        row for row in better
        if float(row.get("pt_improvement_pct", -np.inf)) >= target_gain - EPS
    ]
    return min(near_best, key=lambda row: (
        sum(abs(v) for v in row["deltas"].values()),
        _candidate_step_distance(row),
        -float(row.get("pt_improvement_pct", -np.inf)),
        float(row.get("objective", np.inf)),
    ))


def _load_ok_records_for_controls(h_item, cfg: IOConfig, data_root: Path,
                                  max_records: int) -> list[dict]:
    ok_root, ok_files = _ok_csv_files(cfg, data_root=data_root)
    io_cols, sensor_cols = _model_record_columns(h_item)
    records = []
    for path in ok_files[:max(0, int(max_records))]:
        try:
            records.append(_load_record_exact(path, ok_root, cfg, io_cols, sensor_cols))
        except Exception as e:
            print(f"  ⚠ H counterfactual OK reference skip: {path}: {e}")
    return records


def _recipe_key(rec: dict) -> str:
    return str((rec or {}).get("recipe", "unknown") or "unknown")


def _ok_control_reference(ok_records: list[dict], recipe: str, steps,
                          controls: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    same_recipe = [rec for rec in ok_records if _recipe_key(rec) == str(recipe)]
    if isinstance(steps, (list, tuple, set, np.ndarray)):
        step_values = [None] if any(_is_all_steps(x) for x in steps) else sorted(set(int(x) for x in steps))
    else:
        step_values = [None] if _is_all_steps(steps) else [int(steps)]
    for control in controls:
        out[control] = {}
        for step in step_values:
            values = []
            for rec in same_recipe:
                idx = _record_sensor_index(rec, control)
                if idx is None:
                    continue
                mask = _step_mask(rec, step, 0)
                vals = np.asarray(rec["sensors"], dtype=np.float64)[mask, idx]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    values.extend(float(v) for v in vals)
            step_key = _step_key(step)
            if not values:
                out[control][step_key] = {"count": 0}
                continue
            arr = np.asarray(values, dtype=np.float64)
            out[control][step_key] = {
                "count": int(arr.size),
                "p10": float(np.nanpercentile(arr, 10)),
                "median": float(np.nanmedian(arr)),
                "p90": float(np.nanpercentile(arr, 90)),
            }
    return out


def _control_actual_at_step(rec: dict, control: str, step) -> float | None:
    idx = _record_sensor_index(rec, control)
    if idx is None:
        return None
    mask = _step_mask(rec, step, 0)
    vals = np.asarray(rec["sensors"], dtype=np.float64)[mask, idx]
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return None
    return float(np.nanmedian(vals))


def _summarize_control(control: str, candidates: list[dict], ok_ref: dict,
                       actual_values: dict[object, float | None],
                       focus_step: int) -> dict:
    rec = _choose_pt_recommendation(candidates)
    if rec is None:
        return {
            "control": control,
            "recommended_delta": "",
            "status": "no-candidates",
        }
    delta = float(rec["deltas"].get(control, 0.0))
    adjust_step = rec.get("adjust_steps", {}).get(control, rec.get("adjust_step", ALL_STEPS_LABEL))
    adjust_key = _step_key(adjust_step)
    actual_value = actual_values.get(adjust_key)
    ref_by_step = ok_ref.get(control, {}) if isinstance(ok_ref, dict) else {}
    ref = ref_by_step.get(adjust_key, {}) if isinstance(ref_by_step, dict) else {}
    ok_median = ref.get("median")
    ok_delta = ""
    if actual_value is not None and ok_median is not None:
        ok_delta = float(ok_median) - float(actual_value)
    return {
        "control": control,
        "recommended_step": _step_label(adjust_step),
        "step_relation": _step_relation(adjust_step, focus_step),
        "recommended_delta": delta,
        "recommended_value": (
            float(actual_value) + delta if actual_value is not None else ""
        ),
        "pt_applicable": bool(rec.get("pt_applicable")),
        "pt_improvement_pct": rec.get("pt_improvement_pct"),
        "objective_improvement_pct": rec.get("objective_improvement_pct", 0.0),
        "objective": rec.get("objective", ""),
        "ok_count": ref.get("count", 0),
        "ok_p10": ref.get("p10", ""),
        "ok_median": ok_median if ok_median is not None else "",
        "ok_p90": ref.get("p90", ""),
        "ok_delta_to_median": ok_delta,
        "mutated_models": ",".join(rec.get("mutated_models", [])),
        "status": "ok",
    }


def _setup_plot_font():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Malgun Gothic",
        "AppleGothic",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _write_counterfactual_plot(path: Path, wafer: dict,
                               control_summaries: list[dict],
                               combo: dict | None,
                               single_candidates: dict[str, list[dict]]) -> None:
    plt = _setup_plot_font()

    controls = [str(row.get("control", "-")) for row in control_summaries]
    x = np.arange(len(controls), dtype=np.float64)
    deltas = [_as_float(row.get("recommended_delta")) for row in control_summaries]
    recommended_steps = [row.get("recommended_step", "-") for row in control_summaries]
    xlabels = [
        f"{control}\n{_step_label(step)}" if step not in ("", None, "-") else control
        for control, step in zip(controls, recommended_steps)
    ]
    pt_improve = [
        _as_optional_float(row.get("pt_improvement_pct"))
        if row.get("pt_applicable") else float("nan")
        for row in control_summaries
    ]

    fig = plt.figure(figsize=(18, 13.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.32, 1.2, 1.75], hspace=0.78)
    ax_title = fig.add_subplot(gs[0, :])
    ax_delta = fig.add_subplot(gs[1, :])
    ax_curve = fig.add_subplot(gs[2, :])
    ax_title.axis("off")

    combo_text = "-"
    if combo:
        combo_text = ", ".join(
            f"{name} {delta:+.3g}" for name, delta in sorted(combo.get("deltas", {}).items())
        ) or "-"
    title_lines = (
        f"{wafer.get('wafer_id', '-')} | H 제어변수 counterfactual RCA",
        (
            f"Recipe {wafer.get('recipe', '-')} | 진단 focus step {wafer.get('focus_step', '-')} "
            f"({wafer.get('focus_source', '-')}) | 조정/평가 범위 "
            f"{_short_text(wafer.get('score_scope', wafer.get('candidate_steps', '-')), 80)} | "
            f"Focus open valve {_short_text(wafer.get('open_valves', '-'), 80)}"
        ),
        (
            f"Baseline P,T {_format_float(wafer.get('baseline_pt_score'))} | "
            f"조합 후보: {_short_text(combo_text, 115)}"
        ),
    )
    ax_title.text(0.01, 0.86, title_lines[0], ha="left", va="center", fontsize=14, weight="bold")
    ax_title.text(0.01, 0.52, title_lines[1], ha="left", va="center", fontsize=11.5)
    ax_title.text(0.01, 0.22, title_lines[2], ha="left", va="center", fontsize=11.5)

    if controls:
        colors = ["#2563eb" if value >= 0 else "#dc2626" for value in deltas]
        bar_width = 0.34
        delta_x = x - bar_width / 2.0
        improve_x = x + bar_width / 2.0
        delta_bars = ax_delta.bar(
            delta_x, deltas, width=bar_width, color=colors, alpha=0.84,
            label="추천 delta",
        )
        ax_delta.axhline(0.0, color="#111827", lw=0.9)
        ax_delta.set_xticks(x)
        ax_delta.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=8.5)
        ax_delta.set_ylabel("추천 delta")
        ax_delta.set_title("전체 step 추천 delta와 P/T 예상 개선율", fontsize=12, weight="bold", pad=32)
        ax_delta.grid(axis="y", alpha=0.18)
        finite_deltas = [float(v) for v in deltas if np.isfinite(float(v))]
        if finite_deltas:
            y_min = min(0.0, min(finite_deltas))
            y_max = max(0.0, max(finite_deltas))
            y_span = max(float(y_max - y_min), max(abs(v) for v in finite_deltas), 1.0)
            y_pad = y_span * 0.24
            label_offset = y_span * 0.055
            ax_delta.set_ylim(y_min - y_pad, y_max + y_pad)
        else:
            label_offset = 0.08
        for xi, value in zip(delta_x, deltas):
            ax_delta.text(
                xi, value + (label_offset if value >= 0 else -label_offset),
                f"{value:+.3g}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=9,
            )

        ax_pt = ax_delta.twinx()
        improve_bars = ax_pt.bar(
            improve_x, pt_improve, width=bar_width, color="#f97316", alpha=0.76,
            label="P/T 예상 개선율",
        )
        ax_pt.set_ylabel("P/T 개선율 (%)", color="#111827")
        ax_pt.tick_params(axis="y", labelcolor="#111827")
        finite_pt = [float(v) for v in pt_improve if np.isfinite(float(v))]
        if finite_pt:
            delta_low, delta_high = ax_delta.get_ylim()
            delta_span = max(float(delta_high - delta_low), 1e-9)
            zero_frac = (0.0 - float(delta_low)) / delta_span
            zero_frac = min(max(zero_frac, 0.12), 0.88)

            # Keep the secondary-axis zero line aligned with the delta-axis zero.
            pt_pos = max([v for v in finite_pt if v > 0.0] or [0.0])
            pt_neg = max([-v for v in finite_pt if v < 0.0] or [0.0])
            pt_pad = max(max(pt_pos, pt_neg) * 0.28, 1.2)
            pt_scale = max(
                (pt_pos + pt_pad) / max(1.0 - zero_frac, 1e-9),
                (pt_neg + pt_pad) / max(zero_frac, 1e-9),
                1.0,
            )
            pt_low = -zero_frac * pt_scale
            pt_high = (1.0 - zero_frac) * pt_scale
            ax_pt.set_ylim(pt_low, pt_high)
            pt_label_offset = max(float(pt_high - pt_low) * 0.025, 0.12)
            for xi, value in zip(improve_x, pt_improve):
                if np.isfinite(float(value)):
                    value = float(value)
                    ax_pt.text(
                        xi, value + (pt_label_offset if value >= 0.0 else -pt_label_offset),
                        _format_pct(value),
                        ha="center",
                        va="bottom" if value >= 0.0 else "top",
                        fontsize=8.5,
                        color="#111827",
                        clip_on=False,
                    )
        ax_delta.legend(
            [delta_bars, improve_bars],
            ["추천 delta", "P/T 예상 개선율"],
            loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
            borderaxespad=0.0, fontsize=9, frameon=False,
        )

    recommended_step_by_control = {
        str(row.get("control")): row.get("recommended_step")
        for row in control_summaries
    }
    has_curve = False
    curve_x_values: list[float] = []
    for control, candidates in single_candidates.items():
        recommended_step = recommended_step_by_control.get(str(control))
        if recommended_step not in ("", None, "-") and not _is_all_steps(recommended_step):
            candidates = [
                row for row in candidates
                if int(row.get("adjust_steps", {}).get(control, row.get("adjust_step", -10**9))) == int(recommended_step)
            ]
        ordered = sorted(
            candidates,
            key=lambda row: _as_float(row.get("deltas", {}).get(control, 0.0)),
        )
        xs = [_as_float(row.get("deltas", {}).get(control, 0.0)) for row in ordered]
        ys = [_as_float(row.get("pt_improvement_pct")) for row in ordered]
        if not xs:
            continue
        label = f"{control} @{_step_label(recommended_step)}" if recommended_step not in ("", None, "-") else str(control)
        ax_curve.plot(xs, ys, marker="o", lw=1.5, ms=4, label=label)
        has_curve = True
        curve_x_values.extend(float(v) for v in xs if np.isfinite(float(v)))
        best = _choose_pt_recommendation(ordered)
        if best:
            bx = _as_float(best.get("deltas", {}).get(control, 0.0))
            by = _as_float(best.get("pt_improvement_pct"))
            ax_curve.scatter([bx], [by], s=85, edgecolor="white", zorder=5)
    ax_curve.axhline(0.0, color="#111827", lw=0.9)
    ax_curve.axvline(0.0, color="#6b7280", lw=0.8, ls=":")
    ax_curve.set_xlabel("제어변수 delta")
    ax_curve.set_ylabel("P/T 개선율 (%)")
    ax_curve.set_title("Delta별 P/T 개선율 변화", fontsize=12, weight="bold", pad=12)
    ax_curve.grid(alpha=0.18)
    if has_curve:
        ticks = sorted(set(curve_x_values))
        if ticks:
            ax_curve.set_xticks(ticks)
            ax_curve.set_xticklabels([_format_float(value) for value in ticks], fontsize=8.5)
            if len(ticks) == 1:
                ax_curve.set_xlim(ticks[0] - 1.0, ticks[0] + 1.0)
            else:
                x_pad = max((ticks[-1] - ticks[0]) * 0.035, 0.2)
                ax_curve.set_xlim(ticks[0] - x_pad, ticks[-1] + x_pad)
        ax_curve.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                        fontsize=8, frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(right=0.78, bottom=0.08)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(path: Path, wafer: dict, control_summaries: list[dict],
                           combo: dict | None, single_candidates: dict[str, list[dict]],
                           plot_path: Optional[Path] = None) -> None:
    plot_rel = None
    if plot_path:
        try:
            plot_rel = os.path.relpath(str(plot_path), start=str(path.parent))
        except Exception:
            plot_rel = str(plot_path)
    lines = [
        f"# H 제어변수 Counterfactual RCA - {wafer.get('wafer_id', '-')}",
        "",
        "## 분석 기준",
        "",
        f"- Recipe: {wafer.get('recipe', '-')}",
        f"- 진단 focus step: {wafer.get('focus_step', '-')}",
        f"- Focus 선정 기준: {wafer.get('focus_source', '-')}",
        f"- 조정/평가 범위: {wafer.get('score_scope', wafer.get('candidate_steps', '-'))}",
        f"- 진단 focus step open valve: {wafer.get('open_valves', '-')}",
        f"- Baseline P/T residual score: {_format_float(wafer.get('baseline_pt_score'))}",
        "- Delta 적용 기준: 각 H 제어변수 delta를 wafer 전체 ProcessStep에 동일하게 적용",
        "- 개선율 계산 기준: 전체 step P/T residual 개선만 사용",
        "- Focus step 계산 기준: 기존 기울기 RCA와 동일한 PT 전체 센서 누적복원오차 최대 기울기 step이며, 조정 위치가 아니라 진단 기준입니다.",
        "",
    ]
    if plot_rel:
        lines.extend([
            "## 시각화 요약",
            "",
            f"[이미지 생략: 익명화 배포본]",
            "",
        ])
    lines.extend([
        "## 제어변수별 추천 조정 후보",
        "",
        "|제어변수|적용 범위|구분|추천 delta|조정 후 값|P/T 개선|OK p10~p90|OK median까지 차이|계산 모델|",
        "|---|---|---|---:|---:|---:|---|---:|---|",
    ])
    for row in control_summaries:
        ok_band = "-"
        if row.get("ok_count"):
            ok_band = f"{_format_float(row.get('ok_p10'))}~{_format_float(row.get('ok_p90'))}"
        lines.append(
            "|{control}|{step}|{relation}|{delta}|{value}|{ptimp}|{okband}|{okdelta}|{models}|".format(
                control=row.get("control", "-"),
                step=row.get("recommended_step", "-"),
                relation=row.get("step_relation", "-"),
                delta=_format_float(row.get("recommended_delta")),
                value=_format_float(row.get("recommended_value")),
                ptimp=_format_pct(row.get("pt_improvement_pct")),
                okband=ok_band,
                okdelta=_format_float(row.get("ok_delta_to_median")),
                models=row.get("mutated_models", "-") or "-",
            )
        )
    if combo:
        deltas = ", ".join(f"{k} {v:+.3g}" for k, v in sorted(combo.get("deltas", {}).items()))
        lines.extend([
            "",
            "## 조합 조정 후보",
            "",
            f"- 조정 delta: {deltas or '-'}",
            f"- 조정 적용 범위: {json.dumps(combo.get('adjust_steps', {}), ensure_ascii=False) if combo.get('adjust_steps') else '-'}",
            f"- P/T 개선율: {_format_pct(combo.get('pt_improvement_pct'))}",
            f"- 계산에 사용된 모델: {', '.join(combo.get('mutated_models', [])) or '-'}",
        ])
    lines.extend([
        "",
        "## Delta sweep 상세",
        "",
    ])
    for control, candidates in single_candidates.items():
        lines.extend([
            f"### {control}",
            "",
            "|적용 범위|구분|Delta|P/T 개선|",
            "|---|---|---:|---:|",
        ])
        focus_step = int(wafer.get("focus_step", 0) or 0)
        def _adjust_sort_key(row):
            adjust = row.get("adjust_steps", {}).get(control, row.get("adjust_step", ALL_STEPS_LABEL))
            step_rank = 0 if _is_all_steps(adjust) else 1
            step_value = 0 if _is_all_steps(adjust) else int(adjust)
            return (
                step_rank,
                abs(step_value - focus_step) if not _is_all_steps(adjust) else 0,
                step_value,
                float(next(iter(row.get("deltas", {control: 0.0}).values()))),
            )
        ordered = sorted(
            candidates,
            key=_adjust_sort_key,
        )
        for row in ordered:
            delta = row.get("deltas", {}).get(control, 0.0)
            adjust_step = row.get("adjust_steps", {}).get(control, row.get("adjust_step", ALL_STEPS_LABEL))
            lines.append(
                "|{step}|{relation}|{delta}|{ptimp}|".format(
                    step=_step_label(adjust_step),
                    relation=_step_relation(adjust_step, focus_step),
                    delta=_format_float(delta),
                    ptimp=_format_pct(row.get("pt_improvement_pct")),
                )
            )
        lines.append("")
    lines.extend([
        "## 해석 기준",
        "",
        "- Delta는 모델 기반 조정 후보이며, 설비에 바로 입력할 명령값은 아닙니다.",
        "- 현재 리포트의 개선율은 전체 step P/T residual 개선만 사용합니다.",
        "- Focus step은 기존 기울기 RCA와 동일하게 PT 전체 센서 누적복원오차 기울기가 가장 큰 ProcessStep이며, 조정 위치가 아니라 이상 발생 구간을 보여주는 진단 정보입니다.",
        "- 추천 delta는 해당 제어변수를 wafer 전체 step에서 동일하게 증감했을 때 전체 P/T residual이 가장 줄어든 후보입니다.",
        "- H 모델은 개선율 계산과 focus step 선정에 사용하지 않습니다.",
        "- OK p10~p90은 같은 recipe 정상 wafer의 전체 step 제어변수 참고 범위이며, 실제 적용 전 설비 제한값 확인이 필요합니다.",
        "- 추천 delta는 `P/T 잔차 개선율`과 `OK 정상 범위`를 함께 보고 해석해야 합니다.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _process_wafer(csv_path: Path, data_root: Path, cfg: IOConfig, models_items: list,
                   graph, pt_threshold, pt_score_kind: str, pt_scores: dict,
                   prediction_filter: str, requested_controls: list[str],
                   delta_grid: list[float], exclude_steps: set[int],
                   window_steps: int, step_radius: int,
                   h_weight: float, pt_weight: float,
                   control_penalty: float, combo_top_n: int,
                   ok_records: list[dict], report_dir: Path,
                   plot_dir: Path) -> tuple[dict | None, list[dict], dict | None]:
    model_records = _load_model_records(csv_path, data_root, cfg, models_items)
    pt_item = next(item for item in models_items if item.label == "PT")
    h_item = next(item for item in models_items if item.label == "H")
    models = {"PT": pt_item.model, "H": h_item.model}
    pt_rec = model_records.get("PT")
    h_rec = model_records.get("H")
    rec = pt_rec or h_rec or next(iter(model_records.values()))

    pred = _tcn_explain_pt_prediction(
        pt_item.model, pt_rec, pt_threshold,
        score_kind=pt_score_kind, saved_scores=pt_scores) if pt_rec else {"predicted_ng": True}
    if prediction_filter == "predicted-ng" and not pred.get("predicted_ng", False):
        return None, [], {
            "wafer_id": rec.get("wafer_id", Path(csv_path).stem),
            "source_path": str(csv_path),
            "skip_reason": "pt-predicted-ok",
            "pt_score": pred.get("score"),
            "pt_threshold": pred.get("threshold"),
        }

    focus = _focus_step(pt_item.model, pt_rec, h_item.model, h_rec, graph, exclude_steps)
    if focus is None:
        return None, [], {
            "wafer_id": rec.get("wafer_id", Path(csv_path).stem),
            "source_path": str(csv_path),
            "skip_reason": "no-focus-step",
            "pt_score": pred.get("score"),
            "pt_threshold": pred.get("threshold"),
        }
    peak: StepSlope = focus["peak"]
    focus_step = int(peak.step)
    score_step = None
    candidate_steps = [None]
    controls = _available_controls(models, model_records, requested_controls)
    if not controls:
        return None, [], {
            "wafer_id": rec.get("wafer_id", Path(csv_path).stem),
            "source_path": str(csv_path),
            "skip_reason": "no-h-control-inputs",
            "pt_score": pred.get("score"),
            "pt_threshold": pred.get("threshold"),
        }

    pt_base = _score_model(pt_item.model, pt_rec, _is_pt_sensor, score_step, window_steps) if pt_rec else {}
    h_base = _score_model(h_item.model, h_rec, _is_h_response, score_step, window_steps) if h_rec else {}
    base_scores = {
        "pt_score": float(pt_base.get("score", 0.0) or 0.0),
        "h_response_score": float(h_base.get("score", 0.0) or 0.0),
    }
    base_scores["baseline_objective"] = (
        float(pt_weight) if base_scores["pt_score"] > EPS else 0.0
    )
    delta_scale = max(max(abs(x) for x in delta_grid), 1.0)

    ok_ref = _ok_control_reference(
        ok_records, _recipe_key(rec), candidate_steps, [row["sensor"] for row in controls])
    single_candidates: dict[str, list[dict]] = {}
    control_summaries = []
    for item in controls:
        control = item["sensor"]
        candidates = [
            _evaluate_counterfactual(
                models, model_records, {control: delta}, score_step, {control: adjust_step}, window_steps,
                base_scores, h_weight, pt_weight, control_penalty, delta_scale)
            for adjust_step in candidate_steps
            for delta in delta_grid
        ]
        single_candidates[control] = candidates
        value_rec = h_rec if h_rec is not None and _record_sensor_index(h_rec, control) is not None else pt_rec
        actual_values = {
            _step_key(adjust_step): _control_actual_at_step(value_rec, control, adjust_step)
            for adjust_step in candidate_steps
        }
        summary = _summarize_control(control, candidates, ok_ref, actual_values, focus_step)
        summary["input_models"] = ",".join(item["models"])
        control_summaries.append(summary)

    combo = None
    if combo_top_n > 0 and len(control_summaries) >= 2:
        ranked_controls = [
            row["control"] for row in sorted(
                control_summaries,
                key=lambda row: (
                    -float(row.get("objective_improvement_pct", 0.0) or 0.0),
                    abs(float(row.get("recommended_delta", 0.0) or 0.0)),
                    str(row.get("control", "")),
                ),
            )
            if float(row.get("objective_improvement_pct", 0.0) or 0.0) > 0.0
        ][:max(0, int(combo_top_n))]
        if len(ranked_controls) >= 2:
            combo_candidates = []
            ranked_steps = {
                row["control"]: None
                for row in control_summaries
                if row.get("control") in ranked_controls
            }
            for values in itertools.product(delta_grid, repeat=len(ranked_controls)):
                deltas = {control: float(delta) for control, delta in zip(ranked_controls, values)}
                combo_candidates.append(_evaluate_counterfactual(
                    models, model_records, deltas, score_step, ranked_steps, window_steps,
                    base_scores, h_weight, pt_weight, control_penalty, delta_scale))
            combo = _choose_recommendation(combo_candidates)

    wafer = {
        "wafer_id": rec.get("wafer_id", Path(csv_path).stem),
        "recipe": _recipe_key(rec),
        "source_path": str(csv_path),
        "focus_step": focus_step,
        "candidate_steps": ALL_STEPS_LABEL,
        "score_scope": ALL_STEPS_LABEL,
        "step_radius": int(step_radius),
        "focus_source": focus.get("source"),
        "focus_slope": float(peak.slope),
        "focus_gain": float(peak.gain),
        "open_valves": _influential_valve_text(peak),   # 영향력 큰 밸브만(전체 나열 X — slope-valve RCA 와 일관)
        "pt_score": pred.get("score"),
        "pt_threshold": pred.get("threshold"),
        "baseline_h_response_score": base_scores["h_response_score"],
        "baseline_pt_score": base_scores["pt_score"],
        "h_response_sensors": ", ".join(h_base.get("sensors", [])[:8]) if h_base else "",
        "pt_sensors": ", ".join(pt_base.get("sensors", [])[:8]) if pt_base else "",
    }
    report_path = report_dir / f"{_safe_filename(str(wafer['wafer_id']))}_h_counterfactual.md"
    plot_path = plot_dir / f"{_safe_filename(str(wafer['wafer_id']))}_h_counterfactual.png"
    try:
        _write_counterfactual_plot(plot_path, wafer, control_summaries, combo, single_candidates)
        wafer["plot_path"] = str(plot_path)
    except Exception as e:
        print(f"  ⚠ H counterfactual plot skip: {wafer['wafer_id']}: {e}")
        plot_path = None
        wafer["plot_path"] = ""
    _write_markdown_report(report_path, wafer, control_summaries, combo, single_candidates, plot_path)
    wafer["report_path"] = str(report_path)

    rows = []
    for row in control_summaries:
        out = dict(wafer)
        out.update(row)
        out["combo_deltas"] = ""
        out["combo_adjust_steps"] = ""
        out["combo_pt_improvement_pct"] = ""
        if combo:
            out["combo_deltas"] = json.dumps(combo.get("deltas", {}), ensure_ascii=False)
            out["combo_adjust_steps"] = json.dumps(combo.get("adjust_steps", {}), ensure_ascii=False)
            out["combo_pt_improvement_pct"] = combo.get("pt_improvement_pct", "")
        rows.append(out)
    return wafer, rows, None


def generate_h_counterfactual_rca(
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
    control_sensors: Optional[list[str]] = None,
    delta_grid: Optional[list[float]] = None,
    exclude_steps: set[int] | None = None,
    window_steps: int = 0,
    step_radius: int = DEFAULT_STEP_RADIUS,
    h_weight: float = 1.0,
    pt_weight: float = 1.0,
    control_penalty: float = 0.05,
    combo_top_n: int = 2,
    max_ok_records: int = 120,
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
    pt_scores = pt_best.get("scores") or pt_best.get("anomaly_scores") or {}
    if not isinstance(pt_scores, dict):
        pt_scores = {}
    cfg.sensors_order_path = Path(graph_path) if graph_path else cfg.sensors_order_path
    graph = load_sensor_graph(cfg, Path(graph_path) if graph_path else None)
    data_root, files = _csv_files(Path(data_dir), wafer_csv)
    if wafer_id:
        files = [p for p in files if Path(p).stem == wafer_id or wafer_id in Path(p).stem]
    if max_wafers is not None:
        files = files[:max(0, int(max_wafers))]
    if not files:
        raise SystemExit("No wafer CSV matched the requested filters.")

    controls = list(control_sensors or DEFAULT_H_CONTROLS)
    deltas = list(delta_grid or _parse_delta_grid(DEFAULT_DELTA_GRID))
    exclude_steps = set(exclude_steps or {0})
    out_dir = Path(out_dir)
    report_dir = out_dir / "h_counterfactual_reports"
    plot_dir = out_dir / "h_counterfactual_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    h_item = next(item for item in models if item.label == "H")
    ok_records = _load_ok_records_for_controls(h_item, cfg, data_root, max_ok_records)

    rows = []
    wafers = []
    skipped = []
    for csv_path in files:
        wafer, wafer_rows, skipped_row = _process_wafer(
            csv_path, data_root, cfg, models, graph,
            pt_threshold, pt_score_kind, pt_scores, prediction_filter,
            controls, deltas, exclude_steps, window_steps,
            step_radius, h_weight, pt_weight, control_penalty, combo_top_n,
            ok_records, report_dir, plot_dir)
        if skipped_row:
            skipped.append(skipped_row)
            continue
        if wafer:
            wafers.append(wafer)
        rows.extend(wafer_rows)

    csv_path = out_dir / "h_counterfactual_summary.csv"
    fieldnames = [
        "wafer_id", "recipe", "source_path", "focus_step", "candidate_steps",
        "score_scope", "step_radius", "focus_source",
        "focus_slope", "focus_gain", "open_valves", "pt_score", "pt_threshold",
        "baseline_h_response_score", "baseline_pt_score",
        "h_response_sensors", "pt_sensors", "control", "recommended_delta",
        "recommended_step", "step_relation", "recommended_value",
        "pt_applicable", "pt_improvement_pct",
        "ok_count", "ok_p10", "ok_median", "ok_p90",
        "ok_delta_to_median", "mutated_models",
        "input_models", "status", "combo_deltas", "combo_adjust_steps", "combo_pt_improvement_pct",
        "plot_path", "report_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    skipped_path = out_dir / "h_counterfactual_skipped_wafers.csv"
    with open(skipped_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["wafer_id", "source_path", "skip_reason", "pt_score", "pt_threshold"])
        writer.writeheader()
        writer.writerows(skipped)

    summary = {
        "pt_artifacts": str(pt_artifacts),
        "h_artifacts": str(h_artifacts),
        "data_dir": str(data_root),
        "graph_path": str(Path(graph_path) if graph_path else cfg.sensors_order_path),
        "prediction_filter": prediction_filter,
        "controls": controls,
        "delta_grid": deltas,
        "exclude_steps": sorted(int(x) for x in exclude_steps),
        "window_steps": int(window_steps),
        "step_radius": int(step_radius),
        "adjustment_scope": ALL_STEPS_LABEL,
        "score_scope": ALL_STEPS_LABEL,
        "pt_weight": float(pt_weight),
        "control_penalty": float(control_penalty),
        "combo_top_n": int(combo_top_n),
        "max_ok_records": int(max_ok_records),
        "processed_wafers": len(wafers),
        "rows": len(rows),
        "skipped_wafers": len(skipped),
        "summary_csv": str(csv_path),
        "skipped_csv": str(skipped_path),
        "reports": [row.get("report_path") for row in wafers],
        "plots": [row.get("plot_path") for row in wafers if row.get("plot_path")],
        "method": (
            "Focus step은 기존 slope-valve RCA와 동일하게 PT 전체 센서 누적복원오차 "
            "기울기가 가장 큰 ProcessStep으로 선정합니다. H 제어변수를 counterfactual "
            f"input으로 보고 delta 후보를 wafer {ALL_STEPS_LABEL}에 동일하게 적용해 "
            "sweep하며, 출력 개선율은 기존 PT 모델의 전체 step P/T residual 개선 기준으로 표시합니다. "
            "Focus step은 이상 구간 진단용으로만 사용하고 특정 step 조정은 수행하지 않습니다. "
            "H 모델은 개선율 계산과 focus step 선정에 사용하지 않습니다."
        ),
    }
    (out_dir / "h_counterfactual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"H counterfactual RCA rows: {len(rows)} -> {csv_path}")
    print(f"H counterfactual RCA reports: {len(wafers)} -> {report_dir}")
    print(f"H counterfactual skipped wafers: {len(skipped)} -> {skipped_path}")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Create H-control counterfactual RCA reports.")
    ap.add_argument("--pt-artifacts", required=True)
    ap.add_argument("--h-artifacts", required=True)
    ap.add_argument("--data-dir", default="/home/yapark/semes/dataset/val")
    ap.add_argument("--wafer-csv", default=None)
    ap.add_argument("--wafer-id", default=None)
    ap.add_argument("--out-dir", default="reports/h_counterfactual_rca")
    ap.add_argument("--include-all-wafers", action="store_true")
    ap.add_argument("--max-wafers", type=int, default=None)
    ap.add_argument("--graph-path", default=None)
    ap.add_argument("--h-control-sensors", default=",".join(DEFAULT_H_CONTROLS))
    ap.add_argument("--delta-grid", default=DEFAULT_DELTA_GRID)
    ap.add_argument("--exclude-steps", default="0")
    ap.add_argument("--window-steps", type=int, default=0)
    ap.add_argument("--step-radius", type=int, default=DEFAULT_STEP_RADIUS,
                    help="호환용 옵션. 현재 counterfactual은 전체 step에 동일 delta를 적용하므로 사용하지 않습니다.")
    ap.add_argument("--h-weight", type=float, default=1.0,
                    help="호환용 옵션. 현재 버전은 H 개선을 평가에서 제외하므로 사용하지 않습니다.")
    ap.add_argument("--pt-weight", type=float, default=1.0)
    ap.add_argument("--control-penalty", type=float, default=0.05)
    ap.add_argument("--combo-top-n", type=int, default=2)
    ap.add_argument("--max-ok-records", type=int, default=120)
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
    try:
        delta_grid = _parse_delta_grid(args.delta_grid)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    generate_h_counterfactual_rca(
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
        control_sensors=_parse_sensor_list(args.h_control_sensors),
        delta_grid=delta_grid,
        exclude_steps=_parse_exclude_steps(args.exclude_steps),
        window_steps=args.window_steps,
        step_radius=args.step_radius,
        h_weight=args.h_weight,
        pt_weight=args.pt_weight,
        control_penalty=args.control_penalty,
        combo_top_n=args.combo_top_n,
        max_ok_records=args.max_ok_records,
    )


if __name__ == "__main__":
    main()
