"""RCA from all-sensor accumulated reconstruction-error slope and valve state.

This RCA variant uses the wafer-level accumulated reconstruction-error curve to
choose the focus ProcessStep. It then compares valves opened at that step with
other steps in the same wafer and lists related sensors around the opened valve
path.
"""
from __future__ import annotations


import argparse
import csv
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .sensor_schema import in_group, paired_zone
from .config import IOConfig
from .related_overlay import (
    Trace,
    _build_ok_value_bands,
    _csv_files,
    _family,
    _format_float,
    _load_artifact,
    _load_model_records,
    _load_record_exact,
    _merge_traces,
    _model_record_columns,
    _ok_csv_files,
    _lookup_ok_value_mean,
    _read_search_best,
    _resolve_artifacts,
    _safe_filename,
)
from .rca_v2 import _score_kind_from_search_best, _tcn_explain_pt_prediction
from .sensor_graph import SensorGraph, load_sensor_graph


MAX_OK_COMPARE_RECORDS = 120
CHANGE_CONTRIBUTION_TOP_K = 8
OVERLAY_CONTRIBUTION_TOP_K = 5
SUSPECT_VALVE_HIGH_SLOPE_RATIO = 0.55
SUSPECT_VALVE_TOP_K = 2
FRONTLINE_PROMOTION_VALVE = "9000003"
FRONTLINE_PROMOTION_SENSORS = {"tinnerblockpv", "tscco2supplypv"}
BACKLINE_PROMOTION_VALVE = "9000004"
BACKLINE_PROMOTION_SENSORS = {"tinnerblockpv", "tbackfillpv"}
METERING_EQUIVALENT_VALVES = ("9000006", "9000007", "9000008")
METERING_EQUIVALENT_REPRESENTATIVE = "9000006"
DEFAULT_H_ROLE_TOP_K = 2
h_INFLUENCE_BASE_TOP_K = 3
COUNTERFACTUAL_DELTA_CURVE_COLORS = {
    "h_005pv": "#1f77b4",
    "h_0052pv": "#ff7f0e",
    "h_0053pv": "#2ca02c",
}
COUNTERFACTUAL_DELTA_CURVE_DEFAULT_COLOR = "#1f77b4"


def _display_terms(value) -> str:
    text = str(value if value is not None else "-")
    replacements = (
        ("open valves", "열린 밸브"),
        ("open valve", "열린 밸브"),
        ("same valves", "동일 밸브"),
        ("same valve", "동일 밸브"),
        ("Valves", "밸브"),
        ("valves", "밸브"),
        ("Valve", "밸브"),
        ("valve", "밸브"),
    )
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def _build_slope_valve_report_summary_excel(report_dir: Path, out_dir: Path) -> tuple[Path | None, str]:
    try:
        try:
            from .export_slope_valve_reports_excel import build_excel
        except Exception:
            from export_slope_valve_reports_excel import build_excel
        excel_path = Path(out_dir) / "slope_valve_report_summary.xlsx"
        return build_excel(Path(report_dir), out_path=excel_path), ""
    except Exception as exc:
        return None, str(exc)


def _style_h_control_axis(ax, row: dict) -> None:
    if str((row or {}).get("h_role", "") or "") != "control":
        return
    sensor = _compact_sensor_name(str((row or {}).get("sensor", "") or ""))
    color = COUNTERFACTUAL_DELTA_CURVE_COLORS.get(
        sensor, COUNTERFACTUAL_DELTA_CURVE_DEFAULT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.8)
    ax.tick_params(axis="both", color=color)


@dataclass
class StepSlope:
    step: int
    start_index: int
    end_index: int
    duration: int
    gain: float
    slope: float
    accum_start: float
    accum_end: float
    open_valves: list[str]


def _compact_sensor_name(sensor: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(sensor).lower())


def _h_variable_role(sensor: str) -> str:
    if in_group(sensor, "control"):
        return "control"
    if in_group(sensor, "response"):
        return "response"
    return ""


def _h_role_label(role: str, short: bool = False) -> str:
    if role == "control":
        return "H제어" if short else "H 제어변수"
    if role == "response":
        return "H반응" if short else "H 반응변수"
    return ""


def _parse_exclude_steps(value) -> set[int]:
    if value is None:
        return {0}
    if isinstance(value, (list, tuple, set)):
        parts = value
    else:
        parts = re.split(r"[,/+\s]+", str(value))
    out = set()
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        try:
            out.add(int(float(text)))
        except Exception:
            pass
    return out or {0}


def _step_array(rec: dict, n: int) -> np.ndarray:
    step = rec.get("step") if isinstance(rec, dict) else None
    if step is not None and len(step) >= int(n):
        return np.asarray(step[:int(n)]).astype(int)
    return np.ones(int(n), dtype=int)


def _valve_number_from_col(col: str) -> str:
    name = str(col).strip()
    if name.isdigit():
        return name
    digits = "".join(ch for ch in name if ch.isdigit())
    if len(digits) >= 5:
        return digits
    if "valve" in name.lower():
        return name
    return ""


def _valve_sort_key(graph: SensorGraph, valve: str) -> tuple[int, str]:
    return int(graph.rank.get(str(valve), 10**9)), str(valve)


def _open_valves_at_step(rec: dict, step: Optional[int],
                         graph: SensorGraph | None = None) -> list[dict]:
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
    out: list[dict] = []
    seen: set[str] = set()
    for j, col in enumerate(cols):
        number = _valve_number_from_col(col)
        if not number or number in seen:
            continue
        vals = np.asarray(io[mask, j], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size and float(np.nanmax(finite)) > 0.0:
            out.append({
                "valve": str(number),
                "column": str(col),
                "max_value": float(np.nanmax(finite)),
                "open_fraction": float(np.mean(finite > 0.0)),
            })
            seen.add(str(number))
    if graph is not None:
        out.sort(key=lambda row: _valve_sort_key(graph, str(row["valve"])))
    else:
        out.sort(key=lambda row: str(row["valve"]))
    return out


def _valve_text(valves: list[str] | set[str] | tuple[str, ...]) -> str:
    vals = [str(v) for v in valves if str(v)]
    return ", ".join(vals) if vals else "없음"


def _is_metering_equivalent_valve(valve: str) -> bool:
    return str(valve) in set(METERING_EQUIVALENT_VALVES)


def _normalise_equivalent_valves(valves) -> list[str]:
    out: list[str] = []
    added_metering = False
    for raw in valves or []:
        valve = str(raw)
        if not valve:
            continue
        if _is_metering_equivalent_valve(valve):
            if not added_metering:
                out.append(METERING_EQUIVALENT_REPRESENTATIVE)
                added_metering = True
            continue
        if valve not in out:
            out.append(valve)
    return out


def _metering_equivalent_display(open_valves) -> str:
    open_set = {str(v) for v in open_valves or [] if str(v)}
    opened = [v for v in METERING_EQUIVALENT_VALVES if v in open_set]
    if not opened:
        return METERING_EQUIVALENT_REPRESENTATIVE
    representative = opened[0]
    suffix = opened[1:]
    if suffix:
        return f"{representative}({','.join(suffix)})"
    return representative


def _display_valves_with_equivalents(valves, open_valves=None) -> list[str]:
    out: list[str] = []
    added_metering = False
    for raw in valves or []:
        valve = str(raw)
        if not valve:
            continue
        if _is_metering_equivalent_valve(valve):
            if not added_metering:
                out.append(_metering_equivalent_display(open_valves or valves))
                added_metering = True
            continue
        if valve not in out:
            out.append(valve)
    return out


def _valve_text_or_dash(valves: list[str] | set[str] | tuple[str, ...]) -> str:
    vals = [str(v) for v in valves if str(v)]
    return ", ".join(vals) if vals else "-"


def _valve_text_compact(valves, limit: int = 3) -> str:
    vals = [str(v) for v in (valves or []) if str(v)]
    if not vals:
        return "-"
    limit_i = max(1, int(limit))
    shown = vals[:limit_i]
    suffix = f" +{len(vals) - limit_i}" if len(vals) > limit_i else ""
    return ",".join(shown) + suffix


def _format_measure(value, digits: int = 3) -> str:
    if value in ("", None):
        return "-"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "-"
    av = abs(v)
    if 0.0 < av < 10 ** (-int(digits)):
        return f"{v:.2e}"
    return f"{v:.{int(digits)}f}"


def _plot_time_filter_include_step0(model, rec: dict, n: int):
    """RCA plot axis metadata that keeps only the leading ProcessStep 0 visible.

    Peak selection still uses ``exclude_steps``. This helper only changes the
    display axis so process engineers can see the leading step-0 context without
    showing trailing step-0 padding after the process ends.
    """
    keep = np.ones(int(n), dtype=bool)
    step = rec.get("step")
    step_arr = None
    if step is not None and len(step) == int(n):
        step_arr = np.asarray(step).astype(int)
        nonzero_idx = np.flatnonzero(step_arr != 0)
        if nonzero_idx.size:
            leading_end = int(nonzero_idx[0])
            keep = (step_arr != 0) | (np.arange(int(n)) < leading_end)
    orig_idx = np.flatnonzero(keep)
    t = np.arange(len(orig_idx))
    if step_arr is None or orig_idx.size == 0:
        return keep, orig_idx, t, None, [], []

    step_plot = step_arr[keep]
    bounds = (np.nonzero(np.diff(step_plot) != 0)[0] + 1).tolist()
    spans = []
    start = 0
    for end in bounds + [len(step_plot)]:
        if end > start:
            spans.append((int(step_plot[start]), int(start), int(end - 1)))
        start = end
    return keep, orig_idx, t, step_plot, bounds, spans


def _time_axis_label_include_step0(rec: dict) -> str:
    factor = max(1, int(rec.get("_interp_factor", 1) or 1)) if isinstance(rec, dict) else 1
    if factor <= 1:
        return "time index (leading ProcessStep 0 included)"
    return f"interpolated time index x{factor} (leading ProcessStep 0 included)"


def _all_sensor_accumulation(model, rec: dict) -> dict:
    from .recon_explain import (
        _compute,
        _wafer_decision_accum_series,
        _with_interpolated_explain_arrays,
    )

    actual_raw, expected_raw, level_raw = _compute(model, rec)
    view_rec, _actual, _expected, level = _with_interpolated_explain_arrays(
        model, rec, actual_raw, expected_raw, level_raw)
    level = np.asarray(level, dtype=np.float64)
    keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter_include_step0(
        model, view_rec, level.shape[0])
    if step_plot is None or not len(step_plot) or not np.any(keep):
        return {
            "rec": view_rec,
            "level": level,
            "keep": keep,
            "orig_idx": orig_idx,
            "t": t,
            "step_plot": np.asarray([], dtype=int),
            "bounds": [],
            "spans": [],
            "accum": np.asarray([], dtype=np.float64),
            "sensor_cols": [str(x) for x in getattr(model, "sensor_cols", [])],
        }
    accum_full = _wafer_decision_accum_series(level, view_rec.get("step"))
    return {
        "rec": view_rec,
        "level": level,
        "keep": keep,
        "orig_idx": orig_idx,
        "t": t,
        "step_plot": np.asarray(step_plot, dtype=int),
        "bounds": bounds,
        "spans": spans,
        "accum": np.asarray(accum_full, dtype=np.float64)[keep],
        "sensor_cols": [str(x) for x in getattr(model, "sensor_cols", [])],
    }


def _step_slope_rows(accum: np.ndarray, step_plot: np.ndarray, rec: dict,
                     graph: SensorGraph, exclude_steps: set[int]) -> list[StepSlope]:
    accum = np.asarray(accum, dtype=np.float64)
    step_plot = np.asarray(step_plot, dtype=int)
    if not accum.size or not step_plot.size:
        return []
    rows: list[StepSlope] = []
    for step in sorted(int(x) for x in np.unique(step_plot)):
        if int(step) in exclude_steps:
            continue
        idx = np.flatnonzero(step_plot == int(step))
        if not idx.size:
            continue
        start = int(idx[0])
        end = int(idx[-1])
        prev_val = float(accum[start - 1]) if start > 0 else 0.0
        end_val = float(accum[end])
        gain = max(0.0, end_val - prev_val)
        duration = max(1, int(idx.size))
        open_valves = [row["valve"] for row in _open_valves_at_step(rec, step, graph)]
        rows.append(StepSlope(
            step=int(step),
            start_index=start,
            end_index=end,
            duration=duration,
            gain=float(gain),
            slope=float(gain / duration),
            accum_start=prev_val,
            accum_end=end_val,
            open_valves=open_valves,
        ))
    rows.sort(key=lambda row: (-row.slope, -row.gain, row.step))
    return rows


def _trace_step_mask(trace: Trace, step: Optional[int]) -> np.ndarray:
    n = int(len(trace.level))
    if step is not None and trace.rec.get("step") is not None and len(trace.rec["step"]) == n:
        return np.asarray(trace.rec["step"]).astype(int) == int(step)
    return np.ones(n, dtype=bool)


def _trace_step_level(trace: Trace, step: Optional[int]) -> float:
    level = np.nan_to_num(np.asarray(trace.level, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
    if not level.size:
        return 0.0
    mask = _trace_step_mask(trace, step)
    return float(np.nanmax(level[mask])) if mask.any() else float(np.nanmax(level))


def _flat_mean_expected(expected: np.ndarray) -> np.ndarray:
    out = np.asarray(expected, dtype=np.float64).copy()
    finite = out[np.isfinite(out)]
    if finite.size:
        out[:] = float(np.nanmean(finite))
    return out


def _smooth_ok_mean_for_h_sensor(values: np.ndarray, window: int = 7) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 3:
        return arr.copy()
    finite_count = int(np.isfinite(arr).sum())
    if finite_count < 3:
        return arr.copy()
    win = min(int(window), finite_count)
    if win < 3:
        return arr.copy()
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return arr.copy()
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    kernel = np.ones(win, dtype=np.float64)
    summed = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid, kernel, mode="same")
    out = arr.copy()
    mask = counts > 0
    out[mask] = summed[mask] / counts[mask]
    return out


def _is_h_sensor_for_ok_plot(sensor: str, row: dict | None = None) -> bool:
    if row and row.get("h_role"):
        return True
    if _h_variable_role(sensor):
        return True
    try:
        return _family(sensor) == "H"
    except Exception:
        return False


def _trace_step_summary(trace: Trace, step: Optional[int],
                        flat_expected: bool = False) -> dict:
    actual = np.asarray(trace.actual, dtype=np.float64)
    expected = np.asarray(trace.expected, dtype=np.float64)
    if flat_expected:
        expected = _flat_mean_expected(expected)
    n = min(len(actual), len(expected))
    if n <= 0:
        return {
            "actual": None, "expected": None, "gap": None,
            "gap_abs": None, "direction": "-", "compare_index": None,
        }
    mask = _trace_step_mask(trace, step)[:n]
    valid = mask & np.isfinite(actual[:n]) & np.isfinite(expected[:n])
    if not valid.any():
        return {
            "actual": None, "expected": None, "gap": None,
            "gap_abs": None, "direction": "-", "compare_index": None,
        }
    gap_arr = actual[:n] - expected[:n]
    abs_gap = np.where(valid, np.abs(gap_arr), -np.inf)
    idx = int(np.nanargmax(abs_gap))
    a = float(actual[idx])
    e = float(expected[idx])
    gap = a - e
    if gap > 0:
        direction = "실제>기대"
    elif gap < 0:
        direction = "실제<기대"
    else:
        direction = "실제=기대"
    return {
        "actual": a,
        "expected": e,
        "gap": gap,
        "gap_abs": abs(float(gap)),
        "direction": direction,
        "compare_index": idx,
    }


def _shortest_path(graph: SensorGraph, src: str, dst: str,
                   directed: bool, max_hops: int = 80) -> list[str]:
    from collections import deque

    src, dst = str(src), str(dst)
    if not graph.nodes or src not in graph.nodes or dst not in graph.nodes:
        return []
    q = deque([(src, [src])])
    seen = {src}
    while q:
        cur, path = q.popleft()
        if cur == dst:
            return path
        if len(path) - 1 >= int(max_hops):
            continue
        nxts = set(graph.edges.get(cur, ()))
        if not directed:
            nxts |= set(graph.rev.get(cur, ()))
        for nxt in sorted(nxts, key=lambda name: graph.rank.get(name, 10**9)):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, path + [nxt]))
    return []


def _path_text(path: list[str], limit: int = 10) -> str:
    if not path:
        return ""
    if len(path) <= limit:
        return " -> ".join(path)
    head_n = max(1, limit // 2)
    tail_n = max(1, limit - head_n - 1)
    return " -> ".join(path[:head_n] + ["..."] + path[-tail_n:])


def _nearest_valve_relation(graph: SensorGraph, valves: list[str],
                            sensor: str, max_hops: int = 80) -> dict:
    best = None
    for valve in valves:
        forward = _shortest_path(graph, valve, sensor, directed=True, max_hops=max_hops)
        if forward:
            cand = {
                "valve": valve,
                "relation": "opened-valve-downstream",
                "hop": len(forward) - 1,
                "path": forward,
            }
        else:
            undirected = _shortest_path(graph, valve, sensor, directed=False, max_hops=max_hops)
            if not undirected:
                continue
            cand = {
                "valve": valve,
                "relation": "opened-valve-near",
                "hop": len(undirected) - 1,
                "path": undirected,
            }
        rank = (
            0 if cand["relation"] == "opened-valve-downstream" else 1,
            int(cand["hop"]),
            _valve_sort_key(graph, valve),
        )
        if best is None or rank < best[0]:
            best = (rank, cand)
    if best is not None:
        return best[1]
    return {"valve": "", "relation": "same-step-high-error", "hop": "", "path": []}


def _related_sensors_for_step(
    traces: dict[str, Trace],
    graph: SensorGraph,
    step: Optional[int],
    open_valves: list[str],
    top_k: int = 12,
    min_level: float = 1.0,
    max_hops: int = 80,
) -> list[dict]:
    candidates: list[dict] = []
    for sensor, trace in traces.items():
        level = _trace_step_level(trace, step)
        relation = _nearest_valve_relation(graph, open_valves, sensor, max_hops=max_hops)
        has_graph_relation = bool(relation.get("valve"))
        if level < float(min_level) and not has_graph_relation:
            continue
        summary = _trace_step_summary(trace, step)
        relation_bonus = 0.0
        if relation.get("relation") == "opened-valve-downstream":
            relation_bonus = 0.75
        elif relation.get("relation") == "opened-valve-near":
            relation_bonus = 0.25
        candidates.append({
            "sensor": sensor,
            "family": trace.family,
            "model_label": trace.model_label,
            "step_level": float(level),
            "actual": summary.get("actual"),
            "expected": summary.get("expected"),
            "gap": summary.get("gap"),
            "gap_abs": summary.get("gap_abs"),
            "compare_index": summary.get("compare_index"),
            "direction": summary.get("direction", "-"),
            "valve": relation.get("valve", ""),
            "relation": relation.get("relation", "same-step-high-error"),
            "hop": relation.get("hop", ""),
            "path": relation.get("path", []),
            "rank": int(graph.rank.get(sensor, 10**9)),
            "selection_score": float(level) + relation_bonus,
        })
    if not candidates:
        for sensor, trace in traces.items():
            level = _trace_step_level(trace, step)
            summary = _trace_step_summary(trace, step)
            candidates.append({
                "sensor": sensor,
                "family": trace.family,
                "model_label": trace.model_label,
                "step_level": float(level),
                "actual": summary.get("actual"),
                "expected": summary.get("expected"),
                "gap": summary.get("gap"),
                "gap_abs": summary.get("gap_abs"),
                "compare_index": summary.get("compare_index"),
                "direction": summary.get("direction", "-"),
                "valve": "",
                "relation": "same-step-top-error",
                "hop": "",
                "path": [],
                "rank": int(graph.rank.get(sensor, 10**9)),
                "selection_score": float(level),
            })
    candidates.sort(key=lambda row: (
        -float(row["selection_score"]),
        10**9 if row.get("hop", "") == "" else int(row["hop"]),
        int(row["rank"]),
        str(row["sensor"]),
    ))
    chosen = candidates[:max(0, int(top_k))]
    chosen.sort(key=lambda row: (int(row["rank"]), -float(row["step_level"]), str(row["sensor"])))
    return chosen


def _step_row_dict(row: StepSlope) -> dict:
    return {
        "step": row.step,
        "slope": row.slope,
        "gain": row.gain,
        "duration": row.duration,
        "open_valves": list(row.open_valves),
    }


def _previous_step_valve_comparison(rows: list[StepSlope], peak: StepSlope) -> dict:
    ordered = sorted(rows, key=lambda row: int(row.step))
    prev = None
    for row in ordered:
        if int(row.step) < int(peak.step):
            prev = row
        elif int(row.step) >= int(peak.step):
            break
    if prev is None:
        return {
            "status": "missing-previous-step",
            "previous_step": None,
            "peak_step": int(peak.step),
            "added_valves": [],
            "closed_valves": [],
            "kept_valves": [],
            "inference": "peak step 바로 전 비교 가능한 ProcessStep이 없어 직전 step valve 비교를 생략했습니다.",
        }

    prev_set = set(str(v) for v in prev.open_valves if str(v))
    peak_set = set(str(v) for v in peak.open_valves if str(v))
    added = sorted(peak_set - prev_set)
    closed = sorted(prev_set - peak_set)
    kept = sorted(prev_set & peak_set)
    consecutive = int(prev.step) == int(peak.step) - 1
    ratio = float(peak.slope) / max(float(prev.slope), 1e-12)
    step_label = "바로 전 step" if consecutive else "직전 관찰 step"
    step_note = "" if consecutive else f" 실제 바로 전 step {int(peak.step) - 1}은 비교 대상에 없어"
    if added and ratio >= 1.2:
        inference = (
            f"{step_label} {prev.step} 대비 peak step {peak.step}에서 {_valve_text(added)} valve가 추가로 열리고 "
            "누적오차 기울기도 커졌습니다. 추가 개방 valve의 타이밍과 하류 센서를 우선 확인하세요."
        )
    elif added:
        inference = (
            f"{step_label} {prev.step} 대비 peak step {peak.step}에서 {_valve_text(added)} valve가 추가로 열렸습니다. "
            "기울기 증가는 제한적이므로 추가 valve 자체보다 해당 step 센서 응답을 같이 확인하세요."
        )
    elif closed and ratio >= 1.2:
        inference = (
            f"{step_label} {prev.step} 대비 peak step {peak.step}에서 새로 열린 valve는 없고 "
            f"{_valve_text(closed)} valve가 닫혔지만 이상이 커졌습니다. valve 닫힘 이후 잔류 압력/온도 응답을 확인하세요."
        )
    elif prev_set == peak_set and ratio >= 1.2:
        inference = (
            f"{step_label} {prev.step}과 peak step {peak.step}의 valve 조합은 동일하지만 peak에서 이상이 커졌습니다. "
            "valve 조합보다 같은 valve 상태의 지속시간, 개도 타이밍, 센서 응답 변화를 확인하세요."
        )
    else:
        inference = (
            f"{step_label} {prev.step}과 peak step {peak.step}의 valve 변화만으로는 이상 증가가 뚜렷하게 설명되지 않습니다. "
            "센서 contribution과 동일 recipe OK 비교를 같이 보세요."
        )
    if step_note:
        inference = step_note.strip() + ", " + inference
    return {
        "status": "ok",
        "previous_step": int(prev.step),
        "peak_step": int(peak.step),
        "consecutive": bool(consecutive),
        "previous": _step_row_dict(prev),
        "peak": _step_row_dict(peak),
        "added_valves": added,
        "closed_valves": closed,
        "kept_valves": kept,
        "slope_ratio": ratio,
        "inference": inference,
    }


def _step_transition_text(before: StepSlope | dict | None,
                          after: StepSlope | dict | None) -> dict:
    if before is None or after is None:
        return {
            "added_valves": [],
            "closed_valves": [],
            "kept_valves": [],
            "text": "비교 없음",
        }
    before_valves = before.open_valves if isinstance(before, StepSlope) else before.get("open_valves", [])
    after_valves = after.open_valves if isinstance(after, StepSlope) else after.get("open_valves", [])
    before_set = set(str(v) for v in before_valves if str(v))
    after_set = set(str(v) for v in after_valves if str(v))
    added = sorted(after_set - before_set)
    closed = sorted(before_set - after_set)
    kept = sorted(before_set & after_set)
    return {
        "added_valves": added,
        "closed_valves": closed,
        "kept_valves": kept,
        "text": f"+{_valve_text(added)} / -{_valve_text(closed)}",
    }


def _before_peak_after_context(rows: list[StepSlope], peak: StepSlope) -> dict:
    ordered = sorted(rows, key=lambda row: int(row.step))
    peak_idx = next((i for i, row in enumerate(ordered) if int(row.step) == int(peak.step)), None)
    if peak_idx is None:
        return {
            "status": "missing-peak",
            "rows": [_step_row_dict(peak)],
            "before_to_peak": _step_transition_text(None, None),
            "peak_to_after": _step_transition_text(None, None),
            "summary": "peak 주변 step 확인 불가",
            "inference": "Peak step의 전/후 ProcessStep을 찾지 못해 3구간 비교를 생략했습니다.",
        }

    before = ordered[peak_idx - 1] if peak_idx > 0 else None
    after = ordered[peak_idx + 1] if peak_idx + 1 < len(ordered) else None
    items = []
    for label, row in (("before", before), ("peak", peak), ("after", after)):
        if row is None:
            items.append({
                "label": label,
                "step": None,
                "slope": None,
                "gain": None,
                "open_valves": [],
            })
        else:
            item = _step_row_dict(row)
            item["label"] = label
            items.append(item)

    before_to_peak = _step_transition_text(before, peak)
    peak_to_after = _step_transition_text(peak, after)
    before_text = before_to_peak["text"] if before is not None else "이전 step 없음"
    after_text = peak_to_after["text"] if after is not None else "이후 step 없음"
    summary = f"전->피크 {before_text} | 피크->후 {after_text}"

    if before is not None and after is not None:
        inference = (
            f"Before step {before.step}, peak step {peak.step}, after step {after.step}를 비교했습니다. "
            f"전->피크 변화는 {before_text}, 피크->후 변화는 {after_text}입니다."
        )
    elif before is not None:
        inference = (
            f"Before step {before.step}와 peak step {peak.step}를 비교했습니다. "
            f"전->피크 변화는 {before_text}이고 이후 step은 없습니다."
        )
    elif after is not None:
        inference = (
            f"Peak step {peak.step}와 after step {after.step}를 비교했습니다. "
            f"피크->후 변화는 {after_text}이고 이전 step은 없습니다."
        )
    else:
        inference = "Peak step 단독으로만 관찰되어 전/후 비교 대상이 없습니다."

    return {
        "status": "ok",
        "rows": items,
        "before_to_peak": before_to_peak,
        "peak_to_after": peak_to_after,
        "summary": summary,
        "inference": inference,
    }


def _valve_comparison(rows: list[StepSlope], peak: StepSlope,
                      top_n: int = 5) -> dict:
    focus = set(peak.open_valves)
    others = [row for row in rows if row.step != peak.step]
    exact = [row for row in others if set(row.open_valves) == focus]
    subset = [
        row for row in others
        if set(row.open_valves) and set(row.open_valves).issubset(focus)
        and set(row.open_valves) != focus
    ]
    superset = [
        row for row in others
        if focus and focus.issubset(set(row.open_valves))
        and set(row.open_valves) != focus
    ]
    exact.sort(key=lambda row: (-row.slope, abs(row.step - peak.step), row.step))
    subset.sort(key=lambda row: (
        -len(set(row.open_valves) & focus),
        row.slope,
        abs(row.step - peak.step),
        row.step,
    ))
    superset.sort(key=lambda row: (-row.slope, abs(row.step - peak.step), row.step))
    return {
        "previous": _previous_step_valve_comparison(rows, peak),
        "exact": [_step_row_dict(row) for row in exact[:max(0, int(top_n))]],
        "subset": [_step_row_dict(row) for row in subset[:max(0, int(top_n))]],
        "superset": [_step_row_dict(row) for row in superset[:max(0, int(top_n))]],
    }


def _load_ok_reference_records(pt_item, cfg: IOConfig,
                               data_root: Path | None = None,
                               max_records: int = MAX_OK_COMPARE_RECORDS) -> tuple[list[dict], dict]:
    ok_root, ok_files = _ok_csv_files(cfg, data_root=data_root)
    io_cols, sensor_cols = _model_record_columns(pt_item)
    records: list[dict] = []
    for csv_path in ok_files[:max(0, int(max_records))]:
        try:
            records.append(_load_record_exact(csv_path, ok_root, cfg, io_cols, sensor_cols))
        except Exception as e:
            print(f"  ⚠ OK 유사 wafer 비교 record skip: {csv_path}: {e}")
    return records, {
        "ok_root": str(ok_root),
        "available_ok_files": len(ok_files),
        "loaded_ok_records": len(records),
        "max_records": int(max_records),
    }


def _recipe_key(rec: dict) -> str:
    return str((rec or {}).get("recipe", "unknown") or "unknown")


def _valve_jaccard(a, b) -> float:
    sa = set(str(x) for x in (a or []) if str(x))
    sb = set(str(x) for x in (b or []) if str(x))
    union = sa | sb
    if not union:
        return 1.0
    return float(len(sa & sb) / len(union))


def _ok_similarity_score(peak: StepSlope, ok_row: StepSlope) -> float:
    focus = set(peak.open_valves)
    other = set(ok_row.open_valves)
    jaccard = _valve_jaccard(focus, other)
    step_score = 1.0 / (1.0 + abs(int(peak.step) - int(ok_row.step)))
    count_score = 1.0 / (1.0 + abs(len(focus) - len(other)))
    exact_bonus = 0.7 if focus == other else 0.0
    subset_bonus = 0.35 if other and other.issubset(focus) and other != focus else 0.0
    return float(2.5 * jaccard + 0.7 * step_score + 0.3 * count_score + exact_bonus + subset_bonus)


def _ok_similarity_comparison(
    pt_model,
    ok_records: list[dict],
    rec: dict,
    peak: StepSlope,
    graph: SensorGraph,
    exclude_steps: set[int],
    top_n: int = 5,
) -> dict:
    recipe = _recipe_key(rec)
    recipe_ok = [row for row in ok_records if _recipe_key(row) == recipe]
    if not recipe_ok:
        return {
            "status": "missing-same-recipe-ok",
            "recipe": recipe,
            "peak_slope": float(peak.slope),
            "top_matches": [],
            "inference": "같은 recipe train OK wafer가 없어 OK 유사 wafer 비교는 생략했습니다.",
        }

    candidates: list[dict] = []
    for ok_rec in recipe_ok:
        try:
            accum = _all_sensor_accumulation(pt_model, ok_rec)
            rows = _step_slope_rows(
                accum["accum"], accum["step_plot"], accum["rec"], graph, exclude_steps)
        except Exception as e:
            print(f"  ⚠ OK 유사 wafer 비교 skip: {ok_rec.get('wafer_id', '-')}: {e}")
            continue
        for row in rows:
            focus = set(peak.open_valves)
            other = set(row.open_valves)
            jaccard = _valve_jaccard(focus, other)
            if jaccard <= 0.0 and int(row.step) != int(peak.step):
                continue
            candidates.append({
                "ok_wafer_id": ok_rec.get("wafer_id", ""),
                "ok_source_path": ok_rec.get("source_path", ""),
                "ok_step": int(row.step),
                "ok_open_valves": list(row.open_valves),
                "ok_slope": float(row.slope),
                "ok_gain": float(row.gain),
                "ok_duration": int(row.duration),
                "valve_jaccard": float(jaccard),
                "step_delta": int(abs(int(row.step) - int(peak.step))),
                "peak_added_valves": sorted(focus - other),
                "ok_extra_valves": sorted(other - focus),
                "similarity_score": _ok_similarity_score(peak, row),
            })

    if not candidates:
        return {
            "status": "no-similar-ok-valve-step",
            "recipe": recipe,
            "peak_slope": float(peak.slope),
            "top_matches": [],
            "inference": (
                "같은 recipe OK wafer 안에서 peak step valve 조합과 직접 비교할 유사 valve step이 "
                "충분히 잡히지 않았습니다. 해당 valve 조합 자체가 NG wafer에서 특이한 패턴인지 확인하세요."
            ),
        }

    candidates.sort(key=lambda row: (
        -float(row["similarity_score"]),
        float(row["ok_slope"]),
        int(row["step_delta"]),
        str(row["ok_wafer_id"]),
        int(row["ok_step"]),
    ))
    top = candidates[:max(1, int(top_n))]
    best = top[0]
    peak_slope = float(peak.slope)
    ok_slope = float(best.get("ok_slope", 0.0) or 0.0)
    ratio = peak_slope / max(ok_slope, 1e-12)
    added = list(best.get("peak_added_valves") or [])
    ok_valves = list(best.get("ok_open_valves") or [])
    if added:
        inference = (
            f"같은 recipe OK wafer {best.get('ok_wafer_id')} step {best.get('ok_step')}에서는 "
            f"{_valve_text(ok_valves)} 밸브 조합으로 기울기 {_format_float(ok_slope)}가 관찰됐습니다. "
            f"현재 peak step {peak.step}은 여기에 {_valve_text(added)} 밸브가 추가된 조합이고 "
            f"기울기가 {_format_float(peak_slope)}({ratio:.2f}배)입니다. "
            f"OK 유사 패턴 대비 추가된 {_valve_text(added)} 밸브 또는 그 하류 센서를 우선 의심합니다."
        )
    elif set(ok_valves) == set(peak.open_valves) and ratio >= 1.3:
        inference = (
            f"같은 recipe OK wafer {best.get('ok_wafer_id')} step {best.get('ok_step')}에서도 "
            f"동일 valve 조합({_valve_text(peak.open_valves)})이 열렸고 그때는 기울기 "
            f"{_format_float(ok_slope)}로 괜찮았습니다. 현재 wafer의 해당 step은 "
            f"{_format_float(peak_slope)}로 {ratio:.2f}배 커서, valve 조합 자체보다 "
            "해당 step에서만 달라진 센서 응답, valve timing, 개도 지속시간을 의심합니다."
        )
    elif float(best.get("valve_jaccard", 0.0)) < 0.5:
        inference = (
            f"가장 가까운 OK 비교 step도 valve 유사도 {best.get('valve_jaccard'):.2f}로 낮습니다. "
            "현재 peak step valve 조합이 같은 recipe OK 패턴과 다르므로 valve sequence 특이성을 먼저 확인하세요."
        )
    else:
        inference = (
            f"OK 유사 wafer {best.get('ok_wafer_id')} step {best.get('ok_step')}와 valve 패턴은 유사하지만 "
            f"현재 wafer 기울기가 {_format_float(peak_slope)}, OK 비교 기울기가 {_format_float(ok_slope)}입니다. "
            "동일/유사 valve 조건에서 센서 응답이 더 커진 원인을 관련 센서 contribution 중심으로 확인하세요."
        )
    return {
        "status": "ok",
        "recipe": recipe,
        "peak_slope": float(peak_slope),
        "top_matches": top,
        "inference": inference,
    }


def _step_sensor_contribution(accum_data: dict, step: int,
                              baseline_steps: list[int] | None = None,
                              top_k: int = CHANGE_CONTRIBUTION_TOP_K) -> list[dict]:
    level = np.asarray(accum_data.get("level"), dtype=np.float64)
    keep = np.asarray(accum_data.get("keep"), dtype=bool)
    step_plot = np.asarray(accum_data.get("step_plot"), dtype=int)
    sensor_cols = [str(x) for x in accum_data.get("sensor_cols", [])]
    if level.ndim != 2 or not keep.any() or not step_plot.size:
        return []
    z = np.nan_to_num(level[keep], nan=0.0, posinf=0.0, neginf=0.0)
    if z.shape[1] != len(sensor_cols):
        return []
    curr_mask = step_plot == int(step)
    if not curr_mask.any():
        return []
    curr = np.maximum(0.0, z[curr_mask]).sum(axis=0)
    baseline_vectors = []
    for base_step in baseline_steps or []:
        mask = step_plot == int(base_step)
        if mask.any():
            baseline_vectors.append(np.maximum(0.0, z[mask]).sum(axis=0))
    if baseline_vectors:
        baseline = np.mean(np.vstack(baseline_vectors), axis=0)
    else:
        baseline = np.zeros_like(curr)
    increase = curr - baseline
    positive_total = float(np.maximum(0.0, increase).sum())
    step_total = float(curr.sum())
    rows = []
    for idx, sensor in enumerate(sensor_cols):
        inc = float(increase[idx])
        if inc <= 0.0 and curr[idx] <= 0.0:
            continue
        rows.append({
            "sensor": sensor,
            "family": _family(sensor),
            "current_contribution": float(curr[idx]),
            "baseline_contribution": float(baseline[idx]),
            "increase": inc,
            "increase_positive": max(0.0, inc),
            "increase_share_pct": (
                max(0.0, inc) / positive_total * 100.0
                if positive_total > 1e-12 else 0.0
            ),
            "step_share_pct": (
                float(curr[idx]) / step_total * 100.0
                if step_total > 1e-12 else 0.0
            ),
        })
    rows.sort(key=lambda row: (
        -float(row["increase_positive"]),
        -float(row["current_contribution"]),
        str(row["sensor"]),
    ))
    return rows[:max(0, int(top_k))]


def _change_point_analysis(accum_data: dict, step_rows: list[StepSlope],
                           top_k: int = CHANGE_CONTRIBUTION_TOP_K) -> dict:
    ordered = sorted(step_rows, key=lambda row: int(row.step))
    if not ordered:
        return {
            "status": "missing-step-slopes",
            "inference": "step별 기울기가 없어 센서 기여도 분석을 생략했습니다.",
            "top_sensors": [],
        }
    candidates = []
    for idx, row in enumerate(ordered):
        prev = ordered[max(0, idx - 3):idx]
        baseline = float(np.median([p.slope for p in prev])) if prev else 0.0
        jump = float(row.slope - baseline)
        candidates.append((jump, row.slope, row.gain, idx, row, baseline, [p.step for p in prev]))
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[4].step))
    jump, slope, _gain, idx, change_row, baseline_slope, prev_steps = candidates[0]
    if jump <= 0.0:
        change_row = max(ordered, key=lambda row: (row.slope, row.gain))
        idx = ordered.index(change_row)
        prev = ordered[max(0, idx - 3):idx]
        baseline_slope = float(np.median([p.slope for p in prev])) if prev else 0.0
        jump = float(change_row.slope - baseline_slope)
        prev_steps = [p.step for p in prev]
    top_sensors = _step_sensor_contribution(
        accum_data, change_row.step, baseline_steps=prev_steps[-2:], top_k=top_k)
    shown = ", ".join(
        f"{row['sensor']}(+{_format_float(row['increase_positive'])}, {row['increase_share_pct']:.1f}%)"
        for row in top_sensors[:3]
    ) or "센서 기여도 없음"
    inference = (
        f"센서 기여도 기준 step은 {change_row.step}입니다. 최근 이전 step 기울기 기준 "
        f"{_format_float(baseline_slope)}에서 {_format_float(change_row.slope)}로 "
        f"{_format_float(jump)}만큼 증가했습니다. 증가분 상위 센서는 {shown}입니다."
    )
    return {
        "status": "ok",
        "change_step": int(change_row.step),
        "change_slope": float(change_row.slope),
        "baseline_slope": float(baseline_slope),
        "slope_jump": float(jump),
        "baseline_steps": list(prev_steps),
        "top_sensors": top_sensors,
        "inference": inference,
    }


def _peak_step_contribution_analysis(accum_data: dict, step_rows: list[StepSlope],
                                     peak: StepSlope,
                                     top_k: int = CHANGE_CONTRIBUTION_TOP_K) -> dict:
    ordered = sorted(step_rows, key=lambda row: int(row.step))
    peak_idx = next((i for i, row in enumerate(ordered) if int(row.step) == int(peak.step)), None)
    if peak_idx is None:
        baseline_steps: list[int] = []
        baseline_slope = 0.0
    else:
        prev = ordered[max(0, peak_idx - 3):peak_idx]
        baseline_steps = [row.step for row in prev]
        baseline_slope = float(np.median([row.slope for row in prev])) if prev else 0.0
    top_sensors = _step_sensor_contribution(
        accum_data, peak.step, baseline_steps=baseline_steps[-2:], top_k=top_k)
    slope_jump = float(peak.slope - baseline_slope)
    shown = ", ".join(
        f"{row['sensor']}(+{_format_float(row['increase_positive'])}, {row['increase_share_pct']:.1f}%)"
        for row in top_sensors[:3]
    ) or "센서 기여도 없음"
    return {
        "status": "ok",
        "change_step": int(peak.step),
        "change_slope": float(peak.slope),
        "baseline_slope": float(baseline_slope),
        "slope_jump": float(slope_jump),
        "baseline_steps": list(baseline_steps),
        "top_sensors": top_sensors,
        "inference": (
            f"Peak step {peak.step} 기준 센서 기여도입니다. 최근 이전 step 기울기 기준 "
            f"{_format_float(baseline_slope)}에서 {_format_float(peak.slope)}로 "
            f"{_format_float(slope_jump)}만큼 차이가 나며, 기여도 상위 센서는 {shown}입니다."
        ),
    }


def _contribution_overlay_rows(change_point: dict, traces: dict[str, Trace],
                               graph: SensorGraph, open_valves: list[str],
                               step: Optional[int],
                               limit: int = OVERLAY_CONTRIBUTION_TOP_K) -> list[dict]:
    rows: list[dict] = []
    top_sources = sorted(
        change_point.get("top_sensors", []) or [],
        key=lambda row: (
            -float(row.get("increase_positive", 0.0) or 0.0),
            -float(row.get("current_contribution", 0.0) or 0.0),
            str(row.get("sensor", "")),
        ),
    )[:max(0, int(limit))]
    for source in top_sources:
        sensor = str(source.get("sensor", ""))
        if sensor not in traces:
            continue
        trace = traces[sensor]
        relation = _nearest_valve_relation(graph, open_valves, sensor)
        summary = _trace_step_summary(trace, step)
        row = dict(source)
        row.update({
            "sensor": sensor,
            "family": trace.family,
            "model_label": trace.model_label,
            "rank": int(graph.rank.get(sensor, 10**9)),
            "step_level": _trace_step_level(trace, step),
            "actual": summary.get("actual"),
            "expected": summary.get("expected"),
            "gap": summary.get("gap"),
            "gap_abs": summary.get("gap_abs"),
            "compare_index": summary.get("compare_index"),
            "direction": summary.get("direction", "-"),
            "valve": relation.get("valve", ""),
            "relation": relation.get("relation", "same-step-contribution"),
            "hop": relation.get("hop", ""),
            "path": relation.get("path", []),
        })
        rows.append(row)
    rows.sort(key=lambda row: (int(row.get("rank", 10**9)), str(row.get("sensor", ""))))
    return rows


def _h_role_overlay_rows(traces: dict[str, Trace], graph: SensorGraph,
                         open_valves: list[str], step: Optional[int],
                         top_k_per_role: int = DEFAULT_H_ROLE_TOP_K,
                         response_min_level: float = 1.0) -> list[dict]:
    grouped: dict[str, list[dict]] = {"control": [], "response": []}
    for sensor, trace in traces.items():
        role = _h_variable_role(sensor)
        if role not in grouped:
            continue
        summary = _trace_step_summary(trace, step, flat_expected=True)
        level = _trace_step_level(trace, step)
        if role == "response" and float(level) < float(response_min_level):
            continue
        relation = _nearest_valve_relation(graph, open_valves, sensor)
        grouped[role].append({
            "sensor": sensor,
            "family": trace.family,
            "model_label": trace.model_label,
            "rank": int(graph.rank.get(sensor, 10**9)),
            "step_level": float(level),
            "actual": summary.get("actual"),
            "expected": summary.get("expected"),
            "gap": summary.get("gap"),
            "gap_abs": summary.get("gap_abs"),
            "compare_index": summary.get("compare_index"),
            "direction": summary.get("direction", "-"),
            "valve": relation.get("valve", ""),
            "relation": relation.get("relation", "h-role-sensor"),
            "hop": relation.get("hop", ""),
            "path": relation.get("path", []),
            "current_contribution": float(level),
            "baseline_contribution": 0.0,
            "increase": float(level),
            "increase_positive": max(0.0, float(level)),
            "increase_share_pct": 0.0,
            "step_share_pct": 0.0,
            "h_role": role,
            "h_role_label": _h_role_label(role),
            "h_role_added": True,
        })
    rows: list[dict] = []
    for role, candidates in grouped.items():
        candidates.sort(key=lambda row: (
            -float(row.get("gap_abs", 0.0) or 0.0),
            -float(row.get("step_level", 0.0) or 0.0),
            int(row.get("rank", 10**9)),
            str(row.get("sensor", "")),
        ))
        rows.extend(candidates[:max(0, int(top_k_per_role))])
    return rows


def _merge_h_role_rows(base_rows: list[dict], h_rows: list[dict]) -> list[dict]:
    merged: list[dict] = []
    index: dict[str, int] = {}
    for row in list(base_rows or []):
        sensor = str(row.get("sensor", "") or "")
        if not sensor:
            continue
        index[sensor] = len(merged)
        merged.append(row)
    for row in list(h_rows or []):
        sensor = str(row.get("sensor", "") or "")
        if not sensor:
            continue
        if sensor in index:
            target = merged[index[sensor]]
            for key in (
                "actual", "expected", "gap", "gap_abs", "compare_index",
                "direction", "h_role", "h_role_label", "h_role_added",
            ):
                target[key] = row.get(key)
            continue
        index[sensor] = len(merged)
        merged.append(row)
    merged.sort(key=lambda row: (int(row.get("rank", 10**9)), str(row.get("sensor", ""))))
    return merged


def _trim_overlay_rows_for_h_influence(rows: list[dict],
                                       limit: int = h_INFLUENCE_BASE_TOP_K) -> list[dict]:
    ranked = sorted(list(rows or []), key=lambda row: (
        -float(row.get("increase_positive", 0.0) or 0.0),
        -float(row.get("current_contribution", 0.0) or 0.0),
        int(row.get("rank", 10**9)),
        str(row.get("sensor", "")),
    ))
    chosen = ranked[:max(0, int(limit))]
    chosen.sort(key=lambda row: (int(row.get("rank", 10**9)), str(row.get("sensor", ""))))
    return chosen


def _major_sensor_rows(contribution_rows: list[dict] | None,
                       top_n: int = 3) -> list[dict]:
    rows = list(contribution_rows or [])
    rows.sort(key=lambda row: (
        -float(row.get("increase_positive", 0.0) or 0.0),
        -float(row.get("current_contribution", 0.0) or 0.0),
        str(row.get("sensor", "")),
    ))
    return rows[:max(0, int(top_n))]


def _same_valve_state_run(step_rows: list[StepSlope], peak: StepSlope) -> dict:
    ordered = sorted(step_rows, key=lambda row: int(row.step))
    focus = set(peak.open_valves)
    peak_idx = next((i for i, row in enumerate(ordered) if int(row.step) == int(peak.step)), None)
    if peak_idx is None:
        return {
            "steps": [_step_row_dict(peak)],
            "inference": "동일 valve 상태 연속 step을 확인할 수 없어 peak step만 관찰했습니다.",
        }

    start = peak_idx
    while start > 0:
        prev = ordered[start - 1]
        cur = ordered[start]
        if int(cur.step) - int(prev.step) != 1 or set(prev.open_valves) != focus:
            break
        start -= 1
    end = peak_idx
    while end + 1 < len(ordered):
        cur = ordered[end]
        nxt = ordered[end + 1]
        if int(nxt.step) - int(cur.step) != 1 or set(nxt.open_valves) != focus:
            break
        end += 1

    run = ordered[start:end + 1]
    rows = [_step_row_dict(row) for row in run]
    if len(run) <= 1:
        inference = (
            f"Peak step {peak.step}과 동일한 valve 상태({_valve_text(peak.open_valves)})가 "
            "연속으로 유지된 다른 step은 없습니다. 해당 step 단독 timing 또는 센서 응답을 우선 확인하세요."
        )
    else:
        non_peak = [row for row in run if int(row.step) != int(peak.step)]
        best_other = max(non_peak, key=lambda row: row.slope) if non_peak else None
        if best_other is not None and float(peak.slope) > float(best_other.slope) * 1.25:
            inference = (
                f"동일 valve 상태({_valve_text(peak.open_valves)})가 step {run[0].step}~{run[-1].step}에서 "
                f"연속 유지됐지만, peak step {peak.step}의 기울기({_format_float(peak.slope)})가 "
                f"주변 동일 상태 step 최대값({_format_float(best_other.slope)})보다 큽니다. "
                "valve 상태 자체가 같아도 해당 step에서만 센서 응답이 커진 조건을 의심하세요."
            )
        else:
            inference = (
                f"동일 valve 상태({_valve_text(peak.open_valves)})가 step {run[0].step}~{run[-1].step}에서 "
                "연속 유지되며 기울기도 비슷합니다. 단일 step보다 이 연속 valve 구간 전체를 추적 관찰하세요."
            )
    return {
        "steps": rows,
        "start_step": int(run[0].step),
        "end_step": int(run[-1].step),
        "inference": inference,
    }


def _valve_sensor_observation(contribution_rows: list[dict], peak: StepSlope) -> dict:
    valve_related = [
        row for row in contribution_rows
        if str(row.get("relation", "")).startswith("opened-valve")
    ]
    observed = valve_related or contribution_rows
    shown = []
    for row in observed[:5]:
        relation = str(row.get("relation", "-"))
        valve = str(row.get("valve", "") or "")
        relation_text = f"{valve} {relation}" if valve else relation
        shown.append(
            f"{row.get('sensor')}({_format_float(row.get('increase_positive'))}, "
            f"{row.get('direction', '-')}, {relation_text})"
        )
    if shown:
        inference = (
            f"Peak valve({_valve_text(peak.open_valves)}) 관련 관찰은 "
            + ", ".join(shown)
            + " 순입니다."
        )
    else:
        inference = (
            f"Peak valve({_valve_text(peak.open_valves)})와 직접 연결된 기여도 상위 센서가 충분히 잡히지 않았습니다."
        )
    return {
        "sensors": observed,
        "inference": inference,
    }


def _suspect_valves(peak: StepSlope, comparison: dict, ok_comparison: dict,
                    contribution_rows: list[dict] | None = None,
                    top_k: int = SUSPECT_VALVE_TOP_K) -> dict:
    focus = set(str(v) for v in peak.open_valves if str(v))
    if not focus:
        return {
            "valves": [],
            "basis": "peak step에서 열린 valve가 없어 의심 valve를 특정하지 못했습니다.",
        }

    major_rows = _major_sensor_rows(contribution_rows, top_n=3)
    major_sensor_keys = {
        _compact_sensor_name(str(row.get("sensor", "") or ""))
        for row in major_rows
    }
    promotions = []
    if (
        FRONTLINE_PROMOTION_VALVE in focus
        and FRONTLINE_PROMOTION_SENSORS.issubset(major_sensor_keys)
    ):
        promotions.append({
            "valve": FRONTLINE_PROMOTION_VALVE,
            "reason": "t_005_PV와 t_002_PV가 주요 이상센서",
            "sensors": ("t_005_PV", "t_002_PV"),
        })
    promote_back_valve = (
        BACKLINE_PROMOTION_VALVE in focus
        and BACKLINE_PROMOTION_SENSORS.issubset(major_sensor_keys)
    )
    if promote_back_valve:
        promotions.append({
            "valve": BACKLINE_PROMOTION_VALVE,
            "reason": "t_002_PV와 t_001_PV가 주요 이상센서",
            "sensors": ("t_002_PV", "t_001_PV"),
        })

    candidates: dict[str, dict] = {}
    for row in major_rows:
        valve = str(row.get("valve", "") or "")
        if not valve or valve not in focus:
            continue
        candidate_valve = (
            METERING_EQUIVALENT_REPRESENTATIVE
            if _is_metering_equivalent_valve(valve) else valve
        )
        hop_raw = row.get("hop", "")
        try:
            hop = int(hop_raw)
        except Exception:
            hop = 10**6
        entry = candidates.setdefault(candidate_valve, {
            "valve": candidate_valve,
            "count": 0,
            "hop_sum": 0,
            "contribution_sum": 0.0,
            "sensors": [],
            "actual_valves": [],
        })
        entry["count"] += 1
        entry["hop_sum"] += hop
        entry["contribution_sum"] += float(row.get("increase_positive", 0.0) or 0.0)
        if valve not in entry["actual_valves"]:
            entry["actual_valves"].append(valve)
        sensor = str(row.get("sensor", "") or "")
        if sensor:
            entry["sensors"].append(sensor)
    for promotion in promotions:
        valve = str(promotion["valve"])
        entry = candidates.setdefault(valve, {
            "valve": valve,
            "count": 0,
            "hop_sum": 0,
            "contribution_sum": 0.0,
            "sensors": [],
            "actual_valves": [],
        })
        entry["priority_promoted"] = True
        entry["promotion_reason"] = str(promotion["reason"])
        for sensor in promotion["sensors"]:
            if sensor not in entry["sensors"]:
                entry["sensors"].append(sensor)
    if candidates:
        ranked = sorted(candidates.values(), key=lambda row: (
            0 if row.get("priority_promoted") else 1,
            float(row.get("hop_sum", 0)) / max(1, int(row.get("count", 0))),
            -int(row.get("count", 0)),
            -float(row.get("contribution_sum", 0.0)),
            str(row.get("valve", "")),
        ))
        selected = ranked[:max(1, int(top_k))]
        valves = _normalise_equivalent_valves(str(row["valve"]) for row in selected)
        display_valves = _display_valves_with_equivalents(valves, peak.open_valves)
        sensors = []
        for row in selected:
            for sensor in row.get("sensors") or []:
                if sensor and sensor not in sensors:
                    sensors.append(sensor)
        avg_hop = float(np.mean([
            float(row.get("hop_sum", 0)) / max(1, int(row.get("count", 0)))
            for row in selected
        ]))
        basis = (
            f"주요 센서({', '.join(sensors) or '주요 센서'})와 "
            f"graph상 가장 가까운 open valve 상위 {len(valves)}개"
        )
        if any(row.get("priority_promoted") for row in selected):
            promotion_reasons = []
            for row in selected:
                reason = str(row.get("promotion_reason", "") or "")
                if reason and reason not in promotion_reasons:
                    promotion_reasons.append(reason)
            basis = (
                f"{'; '.join(promotion_reasons)}라 "
                "해당 라인 밸브 우선 승격 | "
                f"{basis}"
            )
        return {
            "valves": valves,
            "display_valves": display_valves,
            "basis": basis,
            "major_sensors": sensors,
            "avg_hop": avg_hop,
        }

    for row in ok_comparison.get("top_matches") or []:
        added = sorted(str(v) for v in row.get("peak_added_valves") or [] if str(v))
        if added:
            valves = _normalise_equivalent_valves(added)
            return {
                "valves": valves,
                "display_valves": _display_valves_with_equivalents(valves, peak.open_valves),
                "basis": (
                    f"같은 recipe OK wafer step {row.get('ok_step')} 대비 "
                    "peak step에서 추가로 열린 valve"
                ),
            }

    for row in comparison.get("subset") or []:
        base = set(str(v) for v in row.get("open_valves") or [] if str(v))
        added = sorted(focus - base)
        if added:
            valves = _normalise_equivalent_valves(added)
            return {
                "valves": valves,
                "display_valves": _display_valves_with_equivalents(valves, peak.open_valves),
                "basis": f"same wafer subset step {row.get('step')} 대비 peak step에서 추가로 열린 valve",
            }

    valves = _normalise_equivalent_valves(sorted(focus))
    return {
        "valves": valves,
        "display_valves": _display_valves_with_equivalents(valves, peak.open_valves),
        "basis": "추가 개방 valve가 따로 특정되지 않아 peak step open valve 전체를 추적",
    }


def _slope_high_threshold(step_rows: list[StepSlope], peak: StepSlope) -> float:
    other = [
        float(row.slope) for row in step_rows
        if int(row.step) != int(peak.step) and np.isfinite(float(row.slope))
    ]
    peak_slope = float(peak.slope)
    if other:
        background = max(float(np.percentile(other, 75)), float(np.median(other)) * 1.25)
    else:
        background = 0.0
    threshold = max(background, peak_slope * SUSPECT_VALVE_HIGH_SLOPE_RATIO)
    if threshold <= 0.0 and peak_slope > 0.0:
        threshold = peak_slope * SUSPECT_VALVE_HIGH_SLOPE_RATIO
    return float(threshold)


def _suspect_valve_open_run(step_rows: list[StepSlope], peak: StepSlope,
                            suspect: dict) -> dict:
    suspect_valves = [str(v) for v in suspect.get("valves") or [] if str(v)]
    display_valves = (
        list(suspect.get("display_valves") or [])
        or _display_valves_with_equivalents(suspect_valves, peak.open_valves)
    )
    if not suspect_valves:
        return {
            "status": "no-suspect-valve",
            "suspect_valves": [],
            "display_valves": [],
            "basis": suspect.get("basis", ""),
            "steps": [],
            "inference": "의심 valve가 특정되지 않아 open 지속 구간과 기울기 지속 여부를 판단하지 않았습니다.",
        }

    ordered = sorted(step_rows, key=lambda row: int(row.step))
    peak_idx = next((i for i, row in enumerate(ordered) if int(row.step) == int(peak.step)), None)
    if peak_idx is None:
        return {
            "status": "missing-peak",
            "suspect_valves": suspect_valves,
            "display_valves": display_valves,
            "basis": suspect.get("basis", ""),
            "steps": [_step_row_dict(peak)],
            "inference": "peak step 위치를 step 순서에서 확인하지 못해 의심 valve 지속 구간을 계산하지 않았습니다.",
        }

    target = set(suspect_valves)
    metering_target = bool(target & set(METERING_EQUIVALENT_VALVES))
    exact_target = target - set(METERING_EQUIVALENT_VALVES)

    def contains_target(row: StepSlope) -> bool:
        opened = set(str(v) for v in row.open_valves if str(v))
        if not exact_target.issubset(opened):
            return False
        if metering_target and not (opened & set(METERING_EQUIVALENT_VALVES)):
            return False
        return True

    if not contains_target(ordered[peak_idx]):
        return {
            "status": "suspect-not-open-at-peak",
            "suspect_valves": suspect_valves,
            "display_valves": display_valves,
            "basis": suspect.get("basis", ""),
            "steps": [_step_row_dict(peak)],
            "inference": (
                f"의심 valve({_valve_text(display_valves)})가 peak step {peak.step}의 open valve 목록에서 "
                "확인되지 않아 지속 구간을 계산하지 않았습니다."
            ),
        }

    start = peak_idx
    while start > 0:
        prev = ordered[start - 1]
        cur = ordered[start]
        if int(cur.step) - int(prev.step) != 1 or not contains_target(prev):
            break
        start -= 1
    end = peak_idx
    while end + 1 < len(ordered):
        cur = ordered[end]
        nxt = ordered[end + 1]
        if int(nxt.step) - int(cur.step) != 1 or not contains_target(nxt):
            break
        end += 1

    run = ordered[start:end + 1]
    threshold = _slope_high_threshold(step_rows, peak)
    rows = []
    for row in run:
        high = float(row.slope) >= threshold and float(row.slope) > 0.0
        item = _step_row_dict(row)
        item.update({
            "status": "high-slope" if high else "not-high-slope",
            "high_slope": bool(high),
            "slope_vs_peak_pct": (
                float(row.slope) / max(float(peak.slope), 1e-12) * 100.0
                if float(peak.slope) > 0.0 else 0.0
            ),
        })
        rows.append(item)

    next_row = ordered[end + 1] if end + 1 < len(ordered) else None
    if next_row is None:
        close_text = "마지막 관찰 step까지 open 상태가 이어짐"
        close_step = None
    elif int(next_row.step) - int(ordered[end].step) == 1:
        close_text = f"step {next_row.step}에서 닫히거나 valve 조합이 변경됨"
        close_step = int(next_row.step)
    else:
        close_text = f"step {ordered[end].step} 이후 다음 연속 step이 없어 닫힘 시점 확인 불가"
        close_step = None

    after_peak = [row for row in rows if int(row.get("step", -1)) > int(peak.step)]
    high_after = [row for row in after_peak if row.get("high_slope")]
    continued_high = bool(high_after)
    if after_peak and continued_high:
        high_steps = ", ".join(str(row.get("step")) for row in high_after[:6])
        slope_text = (
            f"peak 이후에도 step {high_steps}에서 큰 기울기 기준 "
            f"{_format_float(threshold)} 이상이 이어졌습니다."
        )
    elif after_peak:
        slope_text = (
            f"peak 이후에도 valve는 이어졌지만 기울기는 큰 기울기 기준 "
            f"{_format_float(threshold)} 아래로 내려갔습니다."
        )
    else:
        slope_text = "peak 이후 같은 의심 valve open 상태가 연속으로 이어진 step은 없습니다."

    series = ", ".join(
        f"{row.get('step')}:{_format_float(row.get('slope'))}"
        + ("*" if row.get("high_slope") else "")
        for row in rows[:10]
    )
    if len(rows) > 10:
        series += ", ..."

    inference = (
        f"의심 valve({_valve_text(display_valves)})는 step {run[0].step}~{run[-1].step}까지 "
        f"열린 상태로 추적됩니다. {close_text}. {slope_text} "
        f"구간 slope({series})에서 * 표시는 큰 기울기 step입니다."
    )
    return {
        "status": "ok",
        "suspect_valves": suspect_valves,
        "display_valves": display_valves,
        "basis": suspect.get("basis", ""),
        "start_step": int(run[0].step),
        "end_step": int(run[-1].step),
        "close_or_change_after_step": close_step,
        "close_text": close_text,
        "high_slope_threshold": threshold,
        "continued_high_after_peak": continued_high,
        "steps": rows,
        "inference": inference,
    }


def _comparison_inference(peak: StepSlope, comparison: dict) -> str:
    focus = set(peak.open_valves)
    if not focus:
        return (
            f"Step {peak.step}에서 개방 valve가 잡히지 않았습니다. "
            "valve 추가 개방보다는 관련 센서의 실제/기대 방향과 IO 기록 누락 여부를 먼저 확인하세요."
        )

    subset_rows = comparison.get("subset") or []
    if subset_rows:
        base = subset_rows[0]
        base_set = set(base.get("open_valves") or [])
        added = sorted(focus - base_set)
        if added:
            ratio = (
                float(peak.slope) / max(float(base.get("slope", 0.0) or 0.0), 1e-12)
            )
            return (
                f"Step {base.get('step')}에서는 {_valve_text(sorted(base_set))} 밸브가 열렸고 "
                f"누적오차 기울기는 {_format_float(base.get('slope'))}였습니다. "
                f"Peak step {peak.step}에서는 여기에 {_valve_text(added)} 밸브가 추가로 열린 상태에서 "
                f"기울기가 {_format_float(peak.slope)}로 커졌습니다"
                f"({ratio:.2f}배). 따라서 추가 개방된 {_valve_text(added)} 밸브 또는 "
                "그 하류 관련 센서를 1차 문제 후보로 봅니다."
            )

    exact_rows = comparison.get("exact") or []
    if exact_rows:
        best = exact_rows[0]
        ratio = float(peak.slope) / max(float(best.get("slope", 0.0) or 0.0), 1e-12)
        if ratio >= 1.3:
            return (
                f"Step {best.get('step')}에서도 동일 valve 조합({_valve_text(sorted(focus))})이 열렸고 "
                f"그때는 기울기 {_format_float(best.get('slope'))}로 상대적으로 괜찮았습니다. "
                f"해당 peak step {peak.step}에서만 기울기가 커졌으므로, 같은 valve 상태에서도 "
                "그 step의 sensor response, valve timing, 개도 지속시간을 의심하세요."
            )
        return (
            f"동일 valve 조합({_valve_text(sorted(focus))})이 열린 다른 step에서도 "
            f"비슷한 누적오차 기울기가 확인됩니다. 특정 단일 valve보다 해당 valve sequence "
            "또는 그 구간의 공정 조건을 반복 이상 후보로 봅니다."
        )

    superset_rows = comparison.get("superset") or []
    if superset_rows:
        other = superset_rows[0]
        extra_other = sorted(set(other.get("open_valves") or []) - focus)
        return (
            f"Peak step과 완전히 같은 valve 조합 또는 subset 비교 step은 부족합니다. "
            f"다만 step {other.get('step')}에서는 peak valve에 {_valve_text(extra_other)}가 더 열린 "
            "superset 조합이 있어, peak step valve 조합과 추가 valve timing을 같이 비교해야 합니다."
        )

    return (
        f"Peak step {peak.step}의 valve 조합({_valve_text(sorted(focus))})과 직접 비교할 "
        "동일/subset step이 충분하지 않습니다. 개방 valve의 단독 문제 단정보다는 관련 센서 "
        "오차 방향과 같은 recipe 반복 여부를 우선 확인하세요."
    )


def _clean_short_text(value, limit: int = 110) -> str:
    text = " ".join(_display_terms(value).split())
    if len(text) <= int(limit):
        return text
    return text[:max(1, int(limit) - 3)].rstrip() + "..."


def _sensor_names_text(contribution_rows: list[dict], top_n: int = 3) -> str:
    sensors = []
    for row in _major_sensor_rows(contribution_rows, top_n=top_n):
        sensor = str(row.get("sensor", "")).strip()
        if not sensor:
            continue
        role = _h_role_label(str(row.get("h_role", "") or ""), short=True)
        sensors.append(f"{sensor}[{role}]" if role else sensor)
    return ", ".join(sensors) if sensors else "주요 센서"


def _direction_text(contribution_rows: list[dict]) -> str:
    directions = [
        str(row.get("direction", ""))
        for row in _major_sensor_rows(contribution_rows, top_n=3)
        if str(row.get("direction", ""))
    ]
    high_count = sum(1 for item in directions if item == "실제>기대")
    low_count = sum(1 for item in directions if item == "실제<기대")
    if high_count > low_count:
        return "실제>기대"
    if low_count > high_count:
        return "실제<기대"
    return "혼재"


def _actual_expected_text(contribution_rows: list[dict], top_n: int = 3) -> str:
    items = []
    for row in _major_sensor_rows(contribution_rows, top_n=top_n):
        sensor = str(row.get("sensor", "") or "-")
        direction = str(row.get("direction", "-") or "-")
        gap = row.get("gap_abs")
        items.append(f"{sensor} {direction}({_format_measure(gap)})")
    return ", ".join(items) if items else "-"


def _role_direction(rows: list[dict], role: str) -> str:
    directions = [
        str(row.get("direction", ""))
        for row in rows
        if str(row.get("h_role", "")) == role and str(row.get("direction", ""))
    ]
    high_count = sum(1 for item in directions if item == "실제>기대")
    low_count = sum(1 for item in directions if item == "실제<기대")
    if high_count > low_count:
        return "실제>기대"
    if low_count > high_count:
        return "실제<기대"
    if directions:
        return "혼재"
    return ""


def _rows_direction(rows: list[dict], top_n: int = 5) -> str:
    selected = _major_sensor_rows(rows, top_n=top_n)
    directions = [
        str(row.get("direction", ""))
        for row in selected
        if str(row.get("direction", ""))
    ]
    high_count = sum(1 for item in directions if item == "실제>기대")
    low_count = sum(1 for item in directions if item == "실제<기대")
    if high_count > low_count:
        return "실제>기대"
    if low_count > high_count:
        return "실제<기대"
    if directions:
        return "혼재"
    return ""


def _h_role_sensor_text(rows: list[dict], role: str, limit: int = 3) -> str:
    selected = [
        row for row in rows
        if str(row.get("h_role", "")) == role and str(row.get("sensor", ""))
    ]
    selected.sort(key=lambda row: (
        -float(row.get("gap_abs", 0.0) or 0.0),
        -float(row.get("step_level", 0.0) or 0.0),
        int(row.get("rank", 10**9)),
        str(row.get("sensor", "")),
    ))
    sensors = [str(row.get("sensor")) for row in selected[:max(0, int(limit))]]
    return ", ".join(sensors) if sensors else "-"


def _row_sensor_text(rows: list[dict], limit: int = 5) -> str:
    selected = _major_sensor_rows(rows, top_n=limit)
    sensors = [
        str(row.get("sensor", ""))
        for row in selected
        if str(row.get("sensor", ""))
    ]
    return ", ".join(sensors) if sensors else "-"


def _h_role_interpretation(contribution_rows: list[dict], compact: bool = False) -> str:
    h_rows = [row for row in contribution_rows if row.get("h_role")]
    if not h_rows:
        return ""
    if not any(str(row.get("h_role", "")) == "control" for row in h_rows):
        return ""
    response_rows = [
        row for row in contribution_rows
        if str(row.get("h_role", "")) != "control"
    ]
    ctrl_dir = _role_direction(h_rows, "control")
    resp_dir = _rows_direction(response_rows, top_n=5)
    ctrl_sensors = _h_role_sensor_text(h_rows, "control")
    resp_sensors = _row_sensor_text(response_rows)
    if ctrl_dir == "실제>기대" and resp_dir == "실제>기대":
        meaning = "heater 제어 과다 영향 가능"
    elif ctrl_dir == "실제>기대" and resp_dir == "실제<기대":
        meaning = "제어는 높지만 전달/유량/valve 개방 미달 확인"
    elif ctrl_dir == "실제<기대" and resp_dir == "실제<기대":
        meaning = "heater 제어 부족 영향 가능"
    elif ctrl_dir == "실제<기대" and resp_dir == "실제>기대":
        meaning = "heater만으로 설명 어려움, 압력/유량/잔류열 확인"
    elif ctrl_dir and not resp_dir:
        meaning = "H 반응변수 응답 추가 확인 필요"
    elif resp_dir and not ctrl_dir:
        meaning = "H 제어변수와 무관한 vessel heater 반응 가능"
    else:
        meaning = "H 제어/반응 방향 혼재"
    if compact:
        return f"제어 {ctrl_dir or '-'} / P/T+H반응 {resp_dir or '-'} | {meaning}"
    return (
        f"제어({ctrl_sensors}) {ctrl_dir or '-'} / "
        f"P/T+H반응({resp_sensors}) {resp_dir or '-'} | {meaning}"
    )


def _h_influence_decision(h_control_rows: list[dict],
                          response_rows: list[dict],
                          h_response_rows: list[dict]) -> dict:
    ctrl_dir = _role_direction(h_control_rows, "control")
    resp_dir = _rows_direction(response_rows, top_n=5)
    h_resp_dir = _role_direction(h_response_rows, "response")
    influence = (
        (ctrl_dir == "실제>기대" and resp_dir == "실제>기대")
        or (ctrl_dir == "실제<기대" and resp_dir == "실제<기대")
    )
    if influence and ctrl_dir == "실제>기대":
        verdict = "H 제어 과다 영향"
    elif influence:
        verdict = "H 제어 부족 영향"
    elif ctrl_dir or resp_dir:
        verdict = "H 영향 아님"
    else:
        verdict = "H 판단 불가"
    return {
        "is_h_influence": bool(influence),
        "control_direction": ctrl_dir,
        "response_direction": resp_dir,
        "h_response_direction": h_resp_dir,
        "verdict": verdict,
        "control_sensors": _h_role_sensor_text(h_control_rows, "control"),
        "response_sensors": _row_sensor_text(response_rows),
        "h_response_sensors": _h_role_sensor_text(h_response_rows, "response"),
    }


def _primary_check_text(contribution_rows: list[dict], suspect_valve_run: dict) -> str:
    direction = _direction_text(contribution_rows)
    if direction == "실제>기대":
        return "밸브 개방 미달 확인"
    if direction == "실제<기대":
        return "밸브 과개방 확인"
    if suspect_valve_run.get("continued_high_after_peak"):
        return "밸브 개방 지속 구간 + 하류 센서"
    if suspect_valve_run.get("status") == "ok":
        return "밸브 개폐 타이밍 + 하류 센서"
    return "피크 step 공정 라인"


def _previous_summary_text(previous: dict) -> str:
    if (previous or {}).get("status") != "ok":
        return "비교 없음"
    return (
        f"유지 {_valve_text(previous.get('kept_valves') or [])} | "
        f"추가 {_valve_text(previous.get('added_valves') or [])} | "
        f"닫힘 {_valve_text(previous.get('closed_valves') or [])}"
    )


def _ok_summary_text(ok_comparison: dict) -> str:
    matches = (ok_comparison or {}).get("top_matches") or []
    if not matches:
        return "유사 OK step 없음"
    best = matches[0]
    added = [str(v) for v in best.get("peak_added_valves") or [] if str(v)]
    ok_extra = [str(v) for v in best.get("ok_extra_valves") or [] if str(v)]
    wafer = _clean_short_text(best.get("ok_wafer_id", "-"), 34)
    base = f"OK {wafer} s{best.get('ok_step', '-')}"
    peak_slope = ok_comparison.get("peak_slope")
    ok_slope = best.get("ok_slope")
    try:
        peak_slope_f = float(peak_slope)
        ok_slope_f = float(ok_slope)
    except Exception:
        peak_slope_f = np.nan
        ok_slope_f = np.nan
    if np.isfinite(peak_slope_f) and np.isfinite(ok_slope_f) and ok_slope_f > 1e-12:
        slope_text = f"기울기 {peak_slope_f / ok_slope_f:.2f}배 차이"
    elif np.isfinite(peak_slope_f) and np.isfinite(ok_slope_f):
        slope_text = f"기울기 NG {_format_float(peak_slope_f)} / OK {_format_float(ok_slope_f)}"
    else:
        slope_text = "기울기 비교불가"
    if not added and not ok_extra:
        valve_text = "밸브 동일"
    else:
        parts = []
        if added:
            parts.append(f"NG+ {_valve_text_compact(added, 2)}")
        if ok_extra:
            parts.append(f"OK+ {_valve_text_compact(ok_extra, 2)}")
        valve_text = f"밸브 유사({', '.join(parts)})"
    return f"{base} | {slope_text} | {valve_text}"


def _ok_difference_text(ok_comparison: dict, peak: StepSlope) -> str:
    matches = (ok_comparison or {}).get("top_matches") or []
    if not matches:
        return "비교할 유사 OK 패턴 없음"
    best = matches[0]
    peak_set = set(str(v) for v in peak.open_valves if str(v))
    ok_set = set(str(v) for v in best.get("ok_open_valves") or [] if str(v))
    added = sorted(peak_set - ok_set)
    missing = sorted(ok_set - peak_set)
    parts = []
    if added:
        parts.append(f"NG만 open {_valve_text(added)}")
    if missing:
        parts.append(f"OK만 open {_valve_text(missing)}")
    if not parts:
        parts.append("valve 조합 동일")
    ok_slope = float(best.get("ok_slope", 0.0) or 0.0)
    peak_slope = float(peak.slope)
    ratio = peak_slope / max(ok_slope, 1e-12)
    if ratio >= 1.3:
        slope_text = "NG 기울기 큼"
    elif ratio <= 0.77:
        slope_text = "NG 기울기 작음"
    else:
        slope_text = "기울기 유사"
    return f"{'; '.join(parts)} | {slope_text}"


def _before_peak_after_summary_text(step_context: dict) -> str:
    text = (step_context or {}).get("summary")
    return str(text) if text else "전/피크/후 비교 없음"


def _same_valve_summary_text(same_valve_run: dict) -> str:
    if not same_valve_run:
        return "확인 불가"
    start = same_valve_run.get("start_step")
    end = same_valve_run.get("end_step")
    if start is None or end is None:
        return "peak step 단독"
    if int(start) == int(end):
        return f"step {start} 단독"
    return f"step {start}~{end}"


def _suspect_valve_summary_text(suspect_valve_run: dict) -> str:
    if not suspect_valve_run:
        return "확인 불가"
    valves = _valve_text(
        suspect_valve_run.get("display_valves")
        or suspect_valve_run.get("suspect_valves")
        or []
    )
    start = suspect_valve_run.get("start_step")
    end = suspect_valve_run.get("end_step")
    if start is None or end is None:
        interval = "open 구간 불명"
    elif int(start) == int(end):
        interval = f"open step {start}"
    else:
        interval = f"open step {start}~{end}"
    high = "지속" if suspect_valve_run.get("continued_high_after_peak") else "비지속"
    return f"{valves} | {interval} | peak후 기울기 {high}"


def _second_peak_valve_check(step_rows: list[StepSlope], peak: StepSlope,
                             suspect_valve_run: dict) -> dict:
    ordered = sorted(step_rows, key=lambda row: (-float(row.slope), -float(row.gain), int(row.step)))
    second = next((row for row in ordered if int(row.step) != int(peak.step)), None)
    suspect_valves = [str(v) for v in suspect_valve_run.get("suspect_valves") or [] if str(v)]
    display_suspect_valves = (
        list(suspect_valve_run.get("display_valves") or [])
        or _display_valves_with_equivalents(suspect_valves, peak.open_valves)
    )
    if second is None:
        return {
            "status": "missing-second-peak",
            "second_step": None,
            "same_open_valves": False,
            "shares_suspect_valves": False,
            "matched_suspect_valves": [],
            "display_suspect_valves": display_suspect_valves,
            "summary": "2순위 peak 없음",
            "inference": "비교할 두 번째 peak step이 없어 같은 valve 반복 여부를 확인하지 않았습니다.",
        }

    peak_set = set(_normalise_equivalent_valves(peak.open_valves))
    second_set = set(_normalise_equivalent_valves(second.open_valves))
    suspect_set = set(_normalise_equivalent_valves(suspect_valves))
    matched_suspect = sorted(suspect_set & second_set)
    same_open = bool(peak_set) and peak_set == second_set
    shares_suspect = bool(suspect_set) and suspect_set.issubset(second_set)
    shared = sorted(peak_set & second_set)
    if same_open:
        verdict = "동일 valve 조합 반복"
    elif shares_suspect:
        verdict = "같은 의심 valve 반복"
    elif matched_suspect:
        verdict = "의심 valve 일부 반복"
    elif shared:
        verdict = "일부 valve만 공통"
    else:
        verdict = "다른 valve 이상"
    summary = f"step {second.step} | {verdict}"
    second_display = _display_valves_with_equivalents(second.open_valves, second.open_valves)
    matched_display = _display_valves_with_equivalents(matched_suspect, second.open_valves)
    inference = (
        f"2순위 peak step {second.step}의 open valve는 {_valve_text(second_display)}입니다. "
        f"1순위 peak step {peak.step}의 의심 valve({_valve_text(display_suspect_valves)})와 비교하면 "
        f"{verdict}입니다."
    )
    return {
        "status": "ok",
        "second_step": int(second.step),
        "second_slope": float(second.slope),
        "second_gain": float(second.gain),
        "second_open_valves": list(second.open_valves),
        "display_second_open_valves": second_display,
        "peak_open_valves": list(peak.open_valves),
        "suspect_valves": suspect_valves,
        "display_suspect_valves": display_suspect_valves,
        "same_open_valves": bool(same_open),
        "shares_suspect_valves": bool(shares_suspect),
        "matched_suspect_valves": matched_suspect,
        "display_matched_suspect_valves": matched_display,
        "shared_valves": shared,
        "verdict": verdict,
        "summary": summary,
        "inference": inference,
    }


def _second_peak_summary_text(second_peak_check: dict) -> str:
    text = (second_peak_check or {}).get("summary")
    return str(text) if text else "2순위 peak 확인 불가"


def _summary_note_text(label: str) -> str:
    notes = {
        "Recipe": "해당 wafer 레시피명",
        "Peak step": "누적복원오차 기울기가 가장 큰 step",
        "Peak 밸브": "peak step에서 open 상태인 밸브",
        "직전 비교": "바로 전 step 대비 밸브 변화",
        "OK 비교": "유사 OK step과 기울기/밸브 차이",
        "동일 밸브": "같은 밸브 조합이 이어진 구간",
        "의심 밸브": "주요 센서와 가까운 밸브 및 지속 여부",
        "2순위 peak": "두 번째 큰 기울기 step의 밸브 반복 여부",
        "주요 센서": "peak step 기여도 상위 센서",
        "방향": "주요 센서의 실제값과 기대값 비교",
        "1차 확인": "방향성과 밸브 상태 기반 우선 확인 대상",
        "H 역할": "H 제어/반응 센서가 판단에 들어간 경우",
    }
    return notes.get(str(label), "")


def _compact_summary_rows(rec: dict, peak: StepSlope, contribution_rows: list[dict],
                          comparison: dict, ok_comparison: dict,
                          same_valve_run: dict, suspect_valve_run: dict,
                          step_context: dict | None = None,
                          second_peak_check: dict | None = None) -> list[tuple[str, str]]:
    previous = (comparison or {}).get("previous") or {}
    rows = [
        ("Recipe", _clean_short_text(rec.get("recipe", "-"), 80)),
        ("Peak step", str(peak.step)),
        ("Peak 밸브", _valve_text(peak.open_valves)),
        ("직전 비교", _previous_summary_text(previous)),
        ("OK 비교", _ok_summary_text(ok_comparison)),
        ("동일 밸브", _same_valve_summary_text(same_valve_run)),
        ("의심 밸브", _suspect_valve_summary_text(suspect_valve_run)),
        ("2순위 peak", _second_peak_summary_text(second_peak_check or {})),
        ("주요 센서", _sensor_names_text(contribution_rows)),
        ("방향", _direction_text(contribution_rows)),
        ("1차 확인", _primary_check_text(contribution_rows, suspect_valve_run)),
    ]
    h_text = _h_role_interpretation(contribution_rows, compact=True)
    if h_text:
        rows.append(("H 역할", h_text))
    return rows


def _write_report(path: Path, rec: dict, peak: StepSlope, related: list[dict],
                  contribution_rows: list[dict], comparison: dict,
                  ok_comparison: dict, change_point: dict,
                  same_valve_run: dict, valve_observation: dict,
                  suspect_valve_run: dict,
                  step_context: dict,
                  second_peak_check: dict,
                  inference: str, image_path: Path,
                  step_rows: list[StepSlope]) -> None:
    exact_text = comparison.get("exact") or []
    subset_text = comparison.get("subset") or []
    superset_text = comparison.get("superset") or []
    previous_text = comparison.get("previous") or {}
    summary_rows = _compact_summary_rows(
        rec, peak, contribution_rows, comparison, ok_comparison,
        same_valve_run, suspect_valve_run, step_context, second_peak_check)
    lines = [
        f"# Slope-valve RCA report - {rec.get('wafer_id', '-')}",
        "",
        "## 판정 요약",
        "",
        "|항목|내용|비고|",
        "|---|---|---|",
    ]
    for label, value in summary_rows:
        lines.append(
            f"|{label}|{_clean_short_text(value, 160)}|"
            f"{_clean_short_text(_summary_note_text(label), 90)}|"
        )
    lines.extend([
        "",
        f"- Peak 기준: 전체 센서 누적복원오차 그래프에서 step 평균 기울기 최대",
        f"- Overlay image: {image_path}",
        "- Overlay 초록 점선: 같은 recipe train OK wafer의 해당 센서 실제값 평균",
        "",
    ])
    h_rows = [row for row in contribution_rows if row.get("h_role")]
    has_h_control = any(str(row.get("h_role", "")) == "control" for row in h_rows)
    if h_rows and has_h_control:
        lines.extend([
            "## H 제어/반응 변수",
            "",
            f"- 요약: {_h_role_interpretation(contribution_rows)}",
            "",
            "|역할|센서|방향|diff|",
            "|---|---|---|---:|",
        ])
        h_rows_sorted = sorted(
            h_rows,
            key=lambda row: (
                0 if row.get("h_role") == "control" else 1,
                -float(row.get("gap_abs", 0.0) or 0.0),
                int(row.get("rank", 10**9)),
                str(row.get("sensor", "")),
            ),
        )
        for row in h_rows_sorted:
            lines.append(
                "|{role}|{sensor}|{direction}|{gap}|".format(
                    role=_h_role_label(str(row.get("h_role", ""))),
                    sensor=row.get("sensor", "-"),
                    direction=row.get("direction", "-"),
                    gap=_format_measure(row.get("gap_abs")),
                )
            )
        lines.extend([""])
    lines.extend([
        "## 전/피크/후 3구간 비교",
        "",
        "|구간|step|slope|gain|open valves|",
        "|---|---:|---:|---:|---|",
    ])
    for row in (step_context.get("rows") or []):
        label = str(row.get("label", "-"))
        lines.append(
            "|{label}|{step}|{slope}|{gain}|{valves}|".format(
                label=label,
                step="-" if row.get("step") is None else row.get("step"),
                slope=_format_float(row.get("slope")),
                gain=_format_float(row.get("gain")),
                valves=_valve_text(row.get("open_valves") or []),
            )
        )
    lines.extend([
        "",
        f"- 전->피크: {(step_context.get('before_to_peak') or {}).get('text', '비교 없음')}",
        f"- 피크->후: {(step_context.get('peak_to_after') or {}).get('text', '비교 없음')}",
        "",
        "## OK 정상 패턴 대비 차이",
        "",
        f"- 요약: {_ok_difference_text(ok_comparison, peak)}",
        f"- 비교 기준: {_ok_summary_text(ok_comparison)}",
        "",
        "## 2순위 peak valve 반복 확인",
        "",
        "|항목|내용|",
        "|---|---|",
        f"|2순위 peak step|{second_peak_check.get('second_step', '-')}|",
        f"|판정|{second_peak_check.get('verdict', second_peak_check.get('summary', '-'))}|",
        f"|1순위 의심 valve|{_valve_text(second_peak_check.get('display_suspect_valves') or second_peak_check.get('suspect_valves') or [])}|",
        f"|2순위 open valve|{_valve_text(second_peak_check.get('display_second_open_valves') or second_peak_check.get('second_open_valves') or [])}|",
        f"|반복된 의심 valve|{_valve_text(second_peak_check.get('display_matched_suspect_valves') or second_peak_check.get('matched_suspect_valves') or [])}|",
        "",
        f"- {second_peak_check.get('inference', '2순위 peak valve 반복 여부를 확인하지 못했습니다.')}",
        "",
        "## Valve 비교",
        "",
        "### 바로 전 step 비교",
        "",
        "|항목|내용|",
        "|---|---|",
        f"|유지 valve|{_valve_text(previous_text.get('kept_valves') or [])}|",
        f"|추가 open valve|{_valve_text(previous_text.get('added_valves') or [])}|",
        f"|닫힌 valve|{_valve_text(previous_text.get('closed_valves') or [])}|",
        "",
    ])
    if previous_text.get("status") == "ok":
        prev_row = previous_text.get("previous") or {}
        peak_row = previous_text.get("peak") or {}
        lines.extend([
            f"- 이전 step {prev_row.get('step')}: valves={_valve_text(prev_row.get('open_valves') or [])}, "
            f"slope={_format_float(prev_row.get('slope'))}, gain={_format_float(prev_row.get('gain'))}",
            f"- Peak step {peak_row.get('step')}: valves={_valve_text(peak_row.get('open_valves') or [])}, "
            f"slope={_format_float(peak_row.get('slope'))}, gain={_format_float(peak_row.get('gain'))}",
            f"- 유지 valve: {_valve_text(previous_text.get('kept_valves') or [])}",
            f"- 추가 open valve: {_valve_text(previous_text.get('added_valves') or [])}",
            f"- 닫힌 valve: {_valve_text(previous_text.get('closed_valves') or [])}",
            "",
        ])
    else:
        lines.extend(["- 없음", ""])
    lines.extend([
        "### 동일 valve 조합 step",
        "",
    ])
    if exact_text:
        for row in exact_text:
            lines.append(
                f"- Step {row.get('step')}: valves={_valve_text(row.get('open_valves') or [])}, "
                f"slope={_format_float(row.get('slope'))}, gain={_format_float(row.get('gain'))}"
            )
    else:
        lines.append("- 없음")
    lines.extend(["", "### Peak step valve의 subset 조합 step", ""])
    if subset_text:
        for row in subset_text:
            added = sorted(set(peak.open_valves) - set(row.get("open_valves") or []))
            lines.append(
                f"- Step {row.get('step')}: base={_valve_text(row.get('open_valves') or [])}, "
                f"peak에서 추가={_valve_text(added)}, "
                f"slope={_format_float(row.get('slope'))}, gain={_format_float(row.get('gain'))}"
            )
    else:
        lines.append("- 없음")
    lines.extend(["", "### Peak step valve의 superset 조합 step", ""])
    if superset_text:
        for row in superset_text:
            extra = sorted(set(row.get("open_valves") or []) - set(peak.open_valves))
            lines.append(
                f"- Step {row.get('step')}: peak 외 추가={_valve_text(extra)}, "
                f"valves={_valve_text(row.get('open_valves') or [])}, "
                f"slope={_format_float(row.get('slope'))}, gain={_format_float(row.get('gain'))}"
            )
    else:
        lines.append("- 없음")
    lines.extend([
        "",
        "## 동일 valve 상태 연속 step 추적",
        "",
        f"- 요약: {_same_valve_summary_text(same_valve_run)}",
        "",
    ])
    run_steps = same_valve_run.get("steps") or []
    if run_steps:
        for row in run_steps:
            label = "peak" if int(row.get("step", -1)) == int(peak.step) else "same-state"
            lines.append(
                f"- {label} step {row.get('step')}: slope={_format_float(row.get('slope'))}, "
                f"gain={_format_float(row.get('gain'))}, valves={_valve_text(row.get('open_valves') or [])}"
            )
    else:
        lines.append("- 없음")
    lines.extend([
        "",
        "## 의심 valve open 지속 및 기울기 추적",
        "",
        f"- 의심 valve: {_valve_text(suspect_valve_run.get('suspect_valves') or [])}",
        f"- open 구간: {_suspect_valve_summary_text(suspect_valve_run)}",
        "",
        "|step|slope|gain|slope vs peak|status|open valves|",
        "|---:|---:|---:|---:|---|---|",
    ])
    suspect_steps = suspect_valve_run.get("steps") or []
    if suspect_steps:
        for row in suspect_steps:
            label = "peak/high" if int(row.get("step", -1)) == int(peak.step) else row.get("status", "-")
            lines.append(
                "|{step}|{slope}|{gain}|{pct:.1f}%|{status}|{valves}|".format(
                    step=row.get("step", "-"),
                    slope=_format_float(row.get("slope")),
                    gain=_format_float(row.get("gain")),
                    pct=float(row.get("slope_vs_peak_pct", 0.0) or 0.0),
                    status=label,
                    valves=_valve_text(row.get("open_valves") or []),
                )
            )
    else:
        lines.append("|-| - |-|-|-|-|")
    lines.extend([
        "",
        "## OK 유사 wafer 비교",
        "",
        f"- 기준: 같은 recipe train OK wafer 중 peak step valve 조합과 유사한 step 비교",
        f"- 요약: {_ok_summary_text(ok_comparison)}",
        "",
        "|OK wafer|OK step|valves|slope|gain|valve similarity|NG 추가 valve|",
        "|---|---:|---|---:|---:|---:|---|",
    ])
    matches = ok_comparison.get("top_matches") or []
    if matches:
        for row in matches:
            lines.append(
                "|{wafer}|{step}|{valves}|{slope}|{gain}|{sim}|{added}|".format(
                    wafer=row.get("ok_wafer_id", "-"),
                    step=row.get("ok_step", "-"),
                    valves=_valve_text(row.get("ok_open_valves") or []),
                    slope=_format_float(row.get("ok_slope")),
                    gain=_format_float(row.get("ok_gain")),
                    sim=_format_float(row.get("valve_jaccard")),
                    added=_valve_text_or_dash(row.get("peak_added_valves") or []),
                )
            )
    else:
        lines.append("|-| - |없음|-|-|-|-|")
    lines.extend([
        "",
        "## Step별 기울기 상위",
        "",
    ])
    for row in sorted(step_rows, key=lambda item: (-item.slope, -item.gain, item.step))[:10]:
        lines.append(
            f"- Step {row.step}: slope={_format_float(row.slope)}, "
            f"gain={_format_float(row.gain)}, valves={_valve_text(row.open_valves)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_display_terms(line) for line in lines) + "\n", encoding="utf-8")


def _plot_image(path: Path, rec: dict, accum_data: dict, peak: StepSlope,
                contribution_rows: list[dict], traces: dict[str, Trace], inference: str,
                ok_comparison: dict | None = None,
                change_point: dict | None = None,
                same_valve_run: dict | None = None,
                valve_observation: dict | None = None,
                suspect_valve_run: dict | None = None,
                comparison: dict | None = None,
                step_context: dict | None = None,
                second_peak_check: dict | None = None,
                ok_value_bands: dict | None = None) -> None:
    from .recon_explain import _draw_step_guides, _setup_font

    import matplotlib

    matplotlib.use("Agg")
    _setup_font()
    import matplotlib.pyplot as plt

    t = np.asarray(accum_data.get("t"), dtype=np.float64)
    accum = np.asarray(accum_data.get("accum"), dtype=np.float64)
    step_plot = np.asarray(accum_data.get("step_plot"), dtype=int)
    bounds = accum_data.get("bounds") or []
    spans = accum_data.get("spans") or []
    overlay_rows = sorted(
        [row for row in contribution_rows if row.get("sensor") in traces],
        key=lambda row: (int(row.get("rank", 10**9)), str(row.get("sensor", ""))),
    )
    n_overlay = len(overlay_rows)
    summary_rows = _compact_summary_rows(
        rec, peak, overlay_rows, comparison or {}, ok_comparison or {},
        same_valve_run or {}, suspect_valve_run or {}, step_context or {},
        second_peak_check or {},
    )
    detail_lines = len(summary_rows) + 1
    detail_ratio = min(6.1, max(3.1, 0.40 * detail_lines + 0.85))
    fig_h = min(
        30.0,
        max(
            10.2,
            5.85
            + 1.35 * max(1, n_overlay)
            + 0.40 * detail_lines,
        ),
    )
    fig = plt.figure(figsize=(18, fig_h))
    height_ratios = [1.55] + [0.82] * n_overlay + [detail_ratio]
    gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    sensor_axes = [fig.add_subplot(gs[i + 1, 0]) for i in range(n_overlay)]
    ax2 = fig.add_subplot(gs[-1, 0])
    ax2.axis("off")
    second_step = (second_peak_check or {}).get("second_step")

    ax.plot(t, accum, color="#111827", lw=1.8, label="전체 센서 누적복원오차")
    suspect_run = suspect_valve_run or {}
    if step_plot.size and suspect_run.get("start_step") is not None and suspect_run.get("end_step") is not None:
        start_step = int(suspect_run.get("start_step"))
        end_step = int(suspect_run.get("end_step"))
        interval_mask = (step_plot >= start_step) & (step_plot <= end_step)
        interval_idx = np.flatnonzero(interval_mask)
        if interval_idx.size:
            ax.axvspan(float(interval_idx[0]) - 0.5, float(interval_idx[-1]) + 0.5,
                       color="#facc15", alpha=0.12, lw=0,
                       label=f"의심 밸브 open {start_step}-{end_step}")
    if step_plot.size:
        if second_step is not None:
            second_mask = step_plot == int(second_step)
            second_idx = np.flatnonzero(second_mask)
            if second_idx.size:
                ax.axvspan(float(second_idx[0]) - 0.5, float(second_idx[-1]) + 0.5,
                           color="#8b5cf6", alpha=0.13, lw=0,
                           label=f"2nd peak step {int(second_step)}")
        mask = step_plot == int(peak.step)
        idx = np.flatnonzero(mask)
        if idx.size:
            ax.axvspan(float(idx[0]) - 0.5, float(idx[-1]) + 0.5,
                       color="#f97316", alpha=0.22, lw=0,
                       label=f"Peak step {peak.step}")
            ax.scatter([float(idx[-1])], [float(accum[idx[-1]])],
                       s=55, color="#b91c1c", edgecolor="white", zorder=5)
    _draw_step_guides(ax, bounds, spans, label=True)
    ax.set_title(
        f"{rec.get('wafer_id', '-')} | slope-밸브 RCA | "
        f"step {peak.step} slope={_format_float(peak.slope)} | "
        f"밸브={_valve_text(peak.open_valves)}",
        fontsize=13,
    )
    ax.set_xlabel(_time_axis_label_include_step0(accum_data.get("rec") or rec))
    ax.set_ylabel("summed reconstruction error")
    ax.grid(axis="y", alpha=0.18)
    handles, labels = ax.get_legend_handles_labels()
    peak_legend_idx = next(
        (i for i, label in enumerate(labels) if str(label).startswith("Peak step")),
        None,
    )
    if peak_legend_idx is not None and peak_legend_idx != 1:
        peak_handle = handles.pop(peak_legend_idx)
        peak_label = labels.pop(peak_legend_idx)
        insert_at = min(1, len(labels))
        handles.insert(insert_at, peak_handle)
        labels.insert(insert_at, peak_label)
    ax.legend(handles, labels, loc="upper left", ncol=1, fontsize=8.5, frameon=True,
              framealpha=0.92, facecolor="white", edgecolor="#d1d5db")

    for idx, (sensor_ax, row) in enumerate(zip(sensor_axes, overlay_rows)):
        sensor = str(row.get("sensor", ""))
        trace = traces[sensor]
        n = len(trace.actual)
        keep, _orig_idx, tt, step_sensor, bounds_sensor, spans_sensor = _plot_time_filter_include_step0(
            trace.model, trace.rec, n)
        if not np.any(keep):
            sensor_ax.axis("off")
            continue
        actual = np.asarray(trace.actual, dtype=np.float64)[keep]
        expected_full = np.asarray(trace.expected, dtype=np.float64)
        if row.get("h_role"):
            expected_full = _flat_mean_expected(expected_full)
        expected = expected_full[keep]
        ok_mean, ok_mean_source = _lookup_ok_value_mean(
            ok_value_bands, trace, step_sensor, spans_sensor)
        ok_mean_plot = None
        if ok_mean is not None:
            ok_mean_plot = np.asarray(ok_mean, dtype=np.float64)
            if _is_h_sensor_for_ok_plot(sensor, row):
                ok_mean_plot = _smooth_ok_mean_for_h_sensor(ok_mean_plot)
        finite_vals = np.concatenate([
            actual[np.isfinite(actual)],
            expected[np.isfinite(expected)],
            np.asarray([], dtype=np.float64) if ok_mean is None
            else ok_mean_plot[np.isfinite(ok_mean_plot)],
        ])
        if finite_vals.size:
            ylo = float(np.nanmin(finite_vals))
            yhi = float(np.nanmax(finite_vals))
        else:
            ylo, yhi = 0.0, 1.0
        pad = (yhi - ylo) * 0.08 + 1e-6
        if step_sensor is not None and len(step_sensor):
            if second_step is not None:
                second_mask = np.asarray(step_sensor).astype(int) == int(second_step)
                second_idx = np.flatnonzero(second_mask)
                if second_idx.size:
                    sensor_ax.axvspan(float(second_idx[0]) - 0.5,
                                      float(second_idx[-1]) + 0.5,
                                      color="#8b5cf6", alpha=0.10, lw=0, zorder=0)
            focus_mask = np.asarray(step_sensor).astype(int) == int(peak.step)
            focus_idx = np.flatnonzero(focus_mask)
            if focus_idx.size:
                sensor_ax.axvspan(float(focus_idx[0]) - 0.5,
                                  float(focus_idx[-1]) + 0.5,
                                  color="#f97316", alpha=0.14, lw=0, zorder=0)
        sensor_ax.plot(tt, expected, color="#1f77b4", lw=1.15, ls="--",
                       label="기대" if idx == 0 else None, zorder=3)
        if ok_mean_plot is not None and np.isfinite(ok_mean_plot).any():
            ok_label = "같은 recipe OK 평균값"
            if ok_mean_source == "global":
                ok_label = "전체 OK 평균값"
            sensor_ax.plot(tt, ok_mean_plot, color="#86efac", lw=1.05,
                           alpha=0.72, ls=(0, (5, 2)),
                           label=ok_label if idx == 0 else None, zorder=3)
        sensor_ax.plot(tt, actual, color="#d62728", lw=1.2,
                       label="실제" if idx == 0 else None, zorder=4)
        _draw_step_guides(sensor_ax, bounds_sensor, spans_sensor, label=False)
        sensor_ax.set_ylim(ylo - pad, yhi + pad)
        direction = str(row.get("direction", "-"))
        gap = _format_measure(row.get("gap_abs"))
        role = _h_role_label(str(row.get("h_role", "") or ""), short=True)
        sensor_label = f"{sensor} [{role}]" if role else sensor
        sensor_ax.set_ylabel(f"{sensor_label}\ndiff {gap} / {direction}",
                             rotation=0, ha="right", va="center", fontsize=8)
        sensor_ax.tick_params(labelsize=8)
        sensor_ax.grid(axis="y", alpha=0.12)
        _style_h_control_axis(sensor_ax, row)
        if idx < n_overlay - 1:
            sensor_ax.label_outer()
        elif len(tt):
            sensor_ax.set_xlabel(_time_axis_label_include_step0(trace.rec))
    if sensor_axes:
        handles, labels = sensor_axes[0].get_legend_handles_labels()
        if handles:
            sensor_axes[0].legend(handles, labels, loc="upper right",
                                  bbox_to_anchor=(1.0, 1.28),
                                  ncol=3, fontsize=8, frameon=False)

    fig.subplots_adjust(bottom=0.035)
    table_data = [
        [
            label,
            _clean_short_text(value, 108),
            _clean_short_text(_summary_note_text(label), 76),
        ]
        for label, value in summary_rows
    ]
    table = ax2.table(
        cellText=table_data,
        colLabels=["항목", "내용", "비고"],
        cellLoc="left",
        colLoc="center",
        colWidths=[0.13, 0.57, 0.30],
        bbox=[0.01, 0.02, 0.98, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    table.scale(1.0, 1.34)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#111827")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("center")
        else:
            cell.set_facecolor("#f9fafb" if row % 2 else "white")
            if col == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color("#374151")
            else:
                cell.get_text().set_color("#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)


def _decision_for_wafer(rec: dict, traces: dict[str, Trace], graph: SensorGraph,
                        accum_data: dict, pt_model, ok_records: list[dict],
                        related_top_k: int, min_level: float,
                        exclude_steps: set[int],
                        include_h_roles: bool = False,
                        h_role_top_k: int = DEFAULT_H_ROLE_TOP_K) -> dict | None:
    step_rows = _step_slope_rows(
        accum_data["accum"], accum_data["step_plot"],
        accum_data["rec"], graph, exclude_steps)
    if not step_rows:
        return None
    peak = step_rows[0]
    related = _related_sensors_for_step(
        traces, graph, peak.step, list(peak.open_valves),
        top_k=related_top_k, min_level=min_level)
    comparison = _valve_comparison(step_rows, peak)
    step_context = _before_peak_after_context(step_rows, peak)
    ok_comparison = _ok_similarity_comparison(
        pt_model, ok_records, rec, peak, graph, exclude_steps)
    change_point = _peak_step_contribution_analysis(accum_data, step_rows, peak)
    base_contribution_rows = _contribution_overlay_rows(
        change_point, traces, graph, list(peak.open_valves), peak.step)
    contribution_rows = base_contribution_rows
    h_role_rows = []
    h_influence = {
        "is_h_influence": False,
        "control_direction": "",
        "response_direction": "",
        "h_response_direction": "",
        "verdict": "H 옵션 비활성",
        "control_sensors": "-",
        "response_sensors": "-",
        "h_response_sensors": "-",
    }
    if include_h_roles:
        base_without_h_control_rows = [
            row for row in base_contribution_rows
            if _h_variable_role(str(row.get("sensor", "") or "")) != "control"
        ]
        contribution_rows = base_without_h_control_rows
        h_candidates = _h_role_overlay_rows(
            traces, graph, list(peak.open_valves), peak.step,
            top_k_per_role=h_role_top_k,
            response_min_level=min_level)
        h_response_rows = [
            row for row in h_candidates
            if str(row.get("h_role", "")) == "response"
        ]
        h_control_rows = [
            row for row in h_candidates
            if str(row.get("h_role", "")) == "control"
        ]
        response_reference_rows = _merge_h_role_rows(
            base_without_h_control_rows,
            h_response_rows,
        )
        h_role_rows = list(h_response_rows)
        h_influence = _h_influence_decision(
            h_control_rows,
            response_reference_rows,
            h_response_rows,
        )
        if h_influence.get("is_h_influence"):
            h_role_rows = h_control_rows + h_response_rows
            contribution_rows = _merge_h_role_rows(
                _trim_overlay_rows_for_h_influence(base_without_h_control_rows),
                h_role_rows,
            )
        else:
            contribution_rows = response_reference_rows
    same_valve_run = _same_valve_state_run(step_rows, peak)
    valve_observation = _valve_sensor_observation(contribution_rows, peak)
    suspect = _suspect_valves(peak, comparison, ok_comparison, contribution_rows)
    suspect_valve_run = _suspect_valve_open_run(step_rows, peak, suspect)
    second_peak_check = _second_peak_valve_check(step_rows, peak, suspect_valve_run)
    inference = _comparison_inference(peak, comparison)
    return {
        "peak": peak,
        "related": related,
        "contribution_rows": contribution_rows,
        "h_role_rows": h_role_rows,
        "h_influence": h_influence,
        "comparison": comparison,
        "step_context": step_context,
        "ok_comparison": ok_comparison,
        "change_point": change_point,
        "same_valve_run": same_valve_run,
        "valve_observation": valve_observation,
        "suspect_valve_run": suspect_valve_run,
        "second_peak_check": second_peak_check,
        "inference": inference,
        "step_rows": step_rows,
    }


def generate_rca_slope_valve_reports(
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
    related_top_k: int = 12,
    min_level: float = 1.0,
    exclude_steps: set[int] | None = None,
    include_h_roles: bool = False,
    h_role_top_k: int = DEFAULT_H_ROLE_TOP_K,
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
    ok_records, ok_reference_summary = _load_ok_reference_records(
        models[0], cfg, data_root=data_root)
    ok_value_bands, ok_value_summary = _build_ok_value_bands(
        models, cfg, data_root=data_root)
    if wafer_id:
        files = [p for p in files if Path(p).stem == wafer_id or wafer_id in Path(p).stem]
    if max_wafers is not None:
        files = files[:max(0, int(max_wafers))]
    if not files:
        raise SystemExit("No wafer CSV matched the requested filters.")

    exclude_steps = set(exclude_steps or {0})
    out_dir = Path(out_dir)
    image_dir = out_dir / "slope_valve_images"
    report_dir = out_dir / "slope_valve_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped: list[dict] = []
    summary = {
        "pt_artifacts": str(pt_artifacts),
        "h_artifacts": str(h_artifacts),
        "data_dir": str(data_root),
        "graph_path": str(Path(graph_path) if graph_path else cfg.sensors_order_path),
        "prediction_filter": prediction_filter,
        "pt_score_kind": pt_score_kind,
        "peak_step_policy": (
            "ProcessStep with max average slope of PT all-sensor accumulated "
            "reconstruction-error curve; excluded steps are ignored"
        ),
        "plot_step_policy": (
            "RCA plots keep the leading ProcessStep 0 context visible on the time axis "
            "and drop trailing step-0 padding; peak selection still ignores steps "
            "listed in exclude_steps"
        ),
        "exclude_steps": sorted(int(x) for x in exclude_steps),
        "ok_reference": ok_reference_summary,
        "ok_value_mean_reference": ok_value_summary,
        "normal_difference_policy": (
            "compare the peak step with the closest same-recipe OK step and show "
            "only valve differences plus slope direction"
        ),
        "before_peak_after_policy": (
            "compare the observed ProcessStep immediately before peak, the peak "
            "step, and the observed ProcessStep immediately after peak"
        ),
        "second_peak_policy": (
            "take the second-highest accumulated-error slope step and check whether "
            "it repeats the primary peak suspect valve or the same open-valve pattern"
        ),
        "sensor_contribution_policy": (
            "step whose accumulated-error slope most increases against the median "
            "of up to three previous steps; sensor contributions are current step "
            "level sum minus previous-step baseline"
        ),
        "overlay_sensor_policy": (
            f"only top {OVERLAY_CONTRIBUTION_TOP_K} contribution sensors are drawn, "
            "then ordered by sensors_order graph rank; each overlay panel also draws "
            "the same-recipe train OK actual-value mean when available"
        ),
        "h_role_policy": (
            "disabled unless include_h_roles is true; when enabled, h_005* sensors "
            "are evaluated as H control variables and Vessel*Heater* sensors as H response "
            "variables. H response variables are treated like P/T response sensors and "
            f"can be shown only when their peak-step level is >= min_level "
            f"({float(min_level):.3g}). H control variables are added only when H control "
            "direction matches the combined direction of major "
            "P/T response sensors plus H response sensors, meaning an H influence conclusion. "
            f"In that case, base related sensors are trimmed to top "
            f"{h_INFLUENCE_BASE_TOP_K} contribution sensors and top {int(h_role_top_k)} "
            "H sensors per role are shown."
        ),
        "include_h_roles": bool(include_h_roles),
        "h_role_top_k": int(h_role_top_k),
        "suspect_valve_tracking_policy": (
            "prefer valves added against same-recipe OK or same-wafer subset comparison; "
            "track how long those valves remain open and whether slope remains high "
            f"(>= max(other-step 75th percentile/median baseline, {SUSPECT_VALVE_HIGH_SLOPE_RATIO:.2f} * peak slope)); "
            f"if t_005_PV and t_002_PV are both major anomaly sensors and "
            f"{FRONTLINE_PROMOTION_VALVE} is open at peak step, promote {FRONTLINE_PROMOTION_VALVE}; "
            f"if t_002_PV and t_001_PV are both major anomaly sensors and "
            f"{BACKLINE_PROMOTION_VALVE} is open at peak step, promote {BACKLINE_PROMOTION_VALVE}; "
            "treat 9000006/9000007/9000008 as equivalent metering valves and display "
            "the actually opened peers in parentheses"
        ),
        "outputs": [],
        "reports": [],
    }

    for csv_path in files:
        model_records = _load_model_records(csv_path, data_root, cfg, models)
        rec = model_records.get("PT") or next(iter(model_records.values()))
        pt_rec = model_records.get("PT", rec)
        pred = _tcn_explain_pt_prediction(
            models[0].model, pt_rec, pt_threshold,
            score_kind=pt_score_kind, saved_scores=pt_scores)
        if prediction_filter == "predicted-ng" and not pred["predicted_ng"]:
            skipped.append({
                "wafer_id": rec.get("wafer_id", ""),
                "source_path": rec.get("source_path", ""),
                "skip_reason": "pt-predicted-ok",
                "pt_score": pred.get("score"),
                "pt_threshold": pred.get("threshold"),
            })
            continue
        traces, skipped_traces = _merge_traces(models, model_records)
        if not traces:
            skipped.append({
                "wafer_id": rec.get("wafer_id", ""),
                "source_path": rec.get("source_path", ""),
                "skip_reason": "no-merged-traces",
                "pt_score": pred.get("score"),
                "pt_threshold": pred.get("threshold"),
            })
            continue
        accum_data = _all_sensor_accumulation(models[0].model, pt_rec)
        decision = _decision_for_wafer(
            rec, traces, graph, accum_data, models[0].model, ok_records,
            related_top_k=related_top_k,
            min_level=min_level,
            exclude_steps=exclude_steps,
            include_h_roles=include_h_roles,
            h_role_top_k=h_role_top_k,
        )
        if decision is None:
            skipped.append({
                "wafer_id": rec.get("wafer_id", ""),
                "source_path": rec.get("source_path", ""),
                "skip_reason": "no-slope-step",
                "pt_score": pred.get("score"),
                "pt_threshold": pred.get("threshold"),
            })
            continue
        peak: StepSlope = decision["peak"]
        stem = f"{_safe_filename(rec.get('wafer_id', 'wafer'))}__slope_valve__step{peak.step}"
        image_path = image_dir / f"{stem}.png"
        report_path = report_dir / f"{stem}.md"
        _plot_image(
            image_path, rec, accum_data, peak, decision["contribution_rows"],
            traces, decision["inference"], decision["ok_comparison"],
            decision["change_point"], decision["same_valve_run"],
            decision["valve_observation"], decision["suspect_valve_run"],
            decision["comparison"], decision["step_context"],
            decision["second_peak_check"], ok_value_bands)
        _write_report(
            report_path, rec, peak, decision["related"],
            decision["contribution_rows"], decision["comparison"],
            decision["ok_comparison"], decision["change_point"],
            decision["same_valve_run"], decision["valve_observation"],
            decision["suspect_valve_run"], decision["step_context"],
            decision["second_peak_check"],
            decision["inference"], image_path, decision["step_rows"])
        summary["outputs"].append(str(image_path))
        summary["reports"].append(str(report_path))
        rows.append({
            "wafer_id": rec.get("wafer_id", ""),
            "recipe": rec.get("recipe", ""),
            "source_path": rec.get("source_path", ""),
            "pt_score": "" if pred.get("score") is None else round(float(pred["score"]), 6),
            "pt_threshold": "" if pred.get("threshold") is None else round(float(pred["threshold"]), 6),
            "pt_score_source": pred.get("source", ""),
            "pt_score_kind": pred.get("score_kind", ""),
            "peak_step": peak.step,
            "peak_slope": round(float(peak.slope), 6),
            "peak_gain": round(float(peak.gain), 6),
            "peak_duration": peak.duration,
            "peak_open_valves": ";".join(peak.open_valves),
            "related_sensors": ";".join(row.get("sensor", "") for row in decision["related"]),
            "related_sensor_details": json.dumps(decision["related"], ensure_ascii=False, default=str),
            "valve_comparison": json.dumps(decision["comparison"], ensure_ascii=False, default=str),
            "previous_step_valve_comparison": json.dumps(
                decision["comparison"].get("previous") or {}, ensure_ascii=False, default=str),
            "previous_step_valve_inference": (
                (decision["comparison"].get("previous") or {}).get("inference", "")
            ),
            "before_peak_after": json.dumps(
                decision["step_context"], ensure_ascii=False, default=str),
            "before_peak_after_summary": _before_peak_after_summary_text(
                decision["step_context"]),
            "ok_similarity_comparison": json.dumps(
                decision["ok_comparison"], ensure_ascii=False, default=str),
            "ok_similarity_inference": decision["ok_comparison"].get("inference", ""),
            "ok_difference_summary": _ok_difference_text(
                decision["ok_comparison"], peak),
            "contribution_step": decision["change_point"].get("change_step", ""),
            "contribution_slope_jump": decision["change_point"].get("slope_jump", ""),
            "contribution_top_sensors": ";".join(
                row.get("sensor", "") for row in decision["change_point"].get("top_sensors", []) or []),
            "contribution_details": json.dumps(
                decision["change_point"], ensure_ascii=False, default=str),
            "h_role_sensors": ";".join(
                f"{row.get('sensor', '')}:{_h_role_label(str(row.get('h_role', '')))}"
                for row in decision.get("h_role_rows", []) or []),
            "h_role_summary": _h_role_interpretation(decision["contribution_rows"]),
            "h_influence": json.dumps(
                decision.get("h_influence", {}), ensure_ascii=False, default=str),
            "h_influence_verdict": (decision.get("h_influence", {}) or {}).get("verdict", ""),
            "h_role_details": json.dumps(
                decision.get("h_role_rows", []) or [], ensure_ascii=False, default=str),
            "same_valve_state_run": json.dumps(
                decision["same_valve_run"], ensure_ascii=False, default=str),
            "same_valve_state_inference": decision["same_valve_run"].get("inference", ""),
            "suspect_valves": ";".join(
                decision["suspect_valve_run"].get("suspect_valves") or []),
            "suspect_valve_open_run": json.dumps(
                decision["suspect_valve_run"], ensure_ascii=False, default=str),
            "suspect_valve_open_inference": decision["suspect_valve_run"].get("inference", ""),
            "suspect_valve_continued_high_after_peak": decision["suspect_valve_run"].get(
                "continued_high_after_peak", ""),
            "second_peak_valve_check": json.dumps(
                decision["second_peak_check"], ensure_ascii=False, default=str),
            "second_peak_summary": _second_peak_summary_text(
                decision["second_peak_check"]),
            "inference": decision["inference"],
            "plot_path": str(image_path),
            "report_path": str(report_path),
        })
        if skipped_traces:
            summary.setdefault("skipped_trace_columns", []).extend({
                "wafer_id": rec.get("wafer_id", ""),
                **row,
            } for row in skipped_traces)

    csv_path = out_dir / "rca_slope_valve_events.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "wafer_id", "recipe", "source_path",
            "pt_score", "pt_threshold", "pt_score_source", "pt_score_kind",
            "peak_step", "peak_slope", "peak_gain", "peak_duration",
            "peak_open_valves", "related_sensors", "related_sensor_details",
            "valve_comparison", "previous_step_valve_comparison",
            "previous_step_valve_inference", "before_peak_after",
            "before_peak_after_summary", "ok_similarity_comparison",
            "ok_similarity_inference", "ok_difference_summary", "contribution_step",
            "contribution_slope_jump", "contribution_top_sensors",
            "contribution_details", "h_role_sensors", "h_role_summary",
            "h_influence", "h_influence_verdict", "h_role_details",
            "same_valve_state_run",
            "same_valve_state_inference", "suspect_valves",
            "suspect_valve_open_run", "suspect_valve_open_inference",
            "suspect_valve_continued_high_after_peak",
            "second_peak_valve_check", "second_peak_summary",
            "inference", "plot_path", "report_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    skipped_path = out_dir / "rca_slope_valve_skipped_wafers.csv"
    with open(skipped_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["wafer_id", "source_path", "skip_reason", "pt_score", "pt_threshold"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped)
    summary["event_csv"] = str(csv_path)
    summary["skipped_csv"] = str(skipped_path)
    excel_path = None
    if summary["reports"]:
        excel_path, excel_error = _build_slope_valve_report_summary_excel(report_dir, out_dir)
        if excel_path is not None:
            summary["report_summary_excel"] = str(excel_path)
        else:
            summary["report_summary_excel_error"] = excel_error
    (out_dir / "rca_slope_valve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"RCA slope-valve images: {len(summary['outputs'])} -> {image_dir}")
    print(f"RCA slope-valve reports: {len(summary['reports'])} -> {report_dir}")
    print(f"RCA slope-valve event manifest: {csv_path}")
    if excel_path is not None:
        print(f"RCA slope-valve report summary Excel: {excel_path}")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Create slope-valve RCA reports.")
    ap.add_argument("--pt-artifacts", required=True)
    ap.add_argument("--h-artifacts", required=True)
    ap.add_argument("--data-dir", default="/home/yapark/semes/dataset/val")
    ap.add_argument("--wafer-csv", default=None)
    ap.add_argument("--wafer-id", default=None)
    ap.add_argument("--out-dir", default="reports/rca_slope_valve")
    ap.add_argument("--include-all-wafers", action="store_true")
    ap.add_argument("--max-wafers", type=int, default=None)
    ap.add_argument("--graph-path", default=None)
    ap.add_argument("--related-top-k", type=int, default=12)
    ap.add_argument("--min-level", type=float, default=1.0)
    ap.add_argument("--exclude-steps", default="0")
    ap.add_argument("--include-h-roles", action="store_true",
                    help="H 제어변수(h_005*)와 H 반응변수(Vessel*Heater*)를 overlay/리포트에 포함")
    ap.add_argument("--h-role-top-k", type=int, default=DEFAULT_H_ROLE_TOP_K,
                    help="--include-h-roles 사용 시 H 제어/반응 역할별 추가 센서 수")
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
    generate_rca_slope_valve_reports(
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
        related_top_k=args.related_top_k,
        min_level=args.min_level,
        exclude_steps=_parse_exclude_steps(args.exclude_steps),
        include_h_roles=args.include_h_roles,
        h_role_top_k=args.h_role_top_k,
    )


if __name__ == "__main__":
    main()
