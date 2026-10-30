"""Combined P/T/H related-sensor overlay plots from two TCN artifacts.

This module loads two saved ``tcn_forecast.pt`` artifacts, usually one trained
for P/T targets and one trained for H targets. For each selected wafer it finds
large reconstruction events, resolves graph-near and same-step related sensors,
and writes one stacked PNG per event.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .config import IOConfig
from .data import _infer_recipe, _read_csv, add_virtual_sensors
from .sensor_graph import SensorGraph, load_sensor_graph


@dataclass
class ArtifactModel:
    label: str
    artifacts_dir: Path
    detector: object
    model: object


@dataclass
class Trace:
    sensor: str
    model_label: str
    family: str
    model: object
    rec: dict
    sensor_index: int
    actual: np.ndarray
    expected: np.ndarray
    level: np.ndarray
    signed_level: np.ndarray


@dataclass
class Event:
    wafer_id: str
    sensor: str
    step: Optional[int]
    peak_index: int
    peak_level: float
    model_label: str


def _sub_ci(base: Path, name: str) -> Path:
    base = Path(base)
    for candidate in (name, name.upper(), name.lower(), name.capitalize()):
        path = base / candidate
        if path.is_dir():
            return path
    if base.is_dir():
        lname = name.lower()
        for child in base.iterdir():
            if child.is_dir() and child.name.lower() == lname:
                return child
    return base / name


def _resolve_artifacts(path: Path) -> Path:
    path = Path(path)
    # ★ 제어기 앱은 ONNX 로 동작하므로 .onnx 아티팩트도 유효한 아티팩트 폴더로 인정.
    if path.is_file() and path.name in ("tcn_forecast.pt", "tcn_forecast.onnx"):
        return path.parent
    if path.is_file() and path.suffix.lower() in (".pt", ".onnx"):
        return path.parent
    if (path / "tcn_forecast.pt").exists() or (path / "tcn_forecast.onnx").exists():
        return path
    if any(path.glob("*.onnx")) or any(path.glob("*.pt")):
        return path
    raise SystemExit(f"모델 아티팩트(.onnx 또는 .pt)를 찾을 수 없음: {path}")


def _load_artifact(label: str, artifacts_dir: Path, cfg: IOConfig) -> ArtifactModel:
    artifacts_dir = _resolve_artifacts(artifacts_dir)
    local_cfg = IOConfig()
    local_cfg.__dict__.update(cfg.__dict__)
    local_cfg.artifacts_dir = artifacts_dir
    params = _read_search_params(artifacts_dir)
    for key, value in params.items():
        if key in IOConfig.__dataclass_fields__:
            setattr(local_cfg, key, value)
    # ★ 제어기 앱은 onnxruntime 로 동작(torch 미포함). ONNX 모델이 있으면 그걸로 로드하고,
    #   RCA 는 model 인터페이스(predict/actual/error_level/sensor_cols/io_cols…)만 쓰므로
    #   OnnxForecastModel 이 TCNForecastModel 을 그대로 대체한다. onnx 없을 때만 PyTorch 폴백.
    try:
        from .onnx_backend_forecast import OnnxForecastModel
        if OnnxForecastModel.available(artifacts_dir):
            model = OnnxForecastModel.load(artifacts_dir, local_cfg)
            return ArtifactModel(label=label, artifacts_dir=artifacts_dir,
                                 detector=None, model=model)
    except Exception:                                     # noqa: BLE001 (onnx 미가용 → 폴백)
        pass
    from .model_tcn_forecast import TCNForecastDetector
    detector = TCNForecastDetector.load(artifacts_dir, local_cfg)
    return ArtifactModel(label=label, artifacts_dir=artifacts_dir,
                         detector=detector, model=detector.A)


def _read_search_params(artifacts_dir: Path) -> dict:
    best = _read_search_best(artifacts_dir)
    params = best.get("params") or {}
    return params if isinstance(params, dict) else {}


def _read_search_best(artifacts_dir: Path) -> dict:
    for name in ("search_best.json", "best_search.json"):
        path = Path(artifacts_dir) / name
        if not path.exists():
            continue
        try:
            best = json.loads(path.read_text(encoding="utf-8")).get("best", {})
            return best if isinstance(best, dict) else {}
        except Exception:
            return {}
    return {}


def _read_search_threshold(artifacts_dir: Path):
    best = _read_search_best(artifacts_dir)
    return best.get("threshold", best.get("val", {}).get("threshold_raw"))


def _model_record_columns(item: ArtifactModel) -> tuple[list[str], list[str]]:
    io_cols = list(getattr(item.model, "io_cols", []) or [])
    sensor_cols: list[str] = []
    seen: set[str] = set()
    for values in (
        getattr(item.model, "input_sensor_cols", []) or [],
        getattr(item.model, "sensor_cols", []) or [],
    ):
        for value in values:
            key = str(value)
            if key not in seen:
                sensor_cols.append(key)
                seen.add(key)
    return io_cols, sensor_cols


def _load_model_records(csv_path: Path, data_root: Path, cfg: IOConfig,
                        models: list[ArtifactModel]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for item in models:
        io_cols, sensor_cols = _model_record_columns(item)
        records[item.label] = _load_record_exact(csv_path, data_root, cfg,
                                                 io_cols, sensor_cols)
    return records


def _find_step_col(df, cfg: IOConfig):
    step_names = {str(x).strip().lower() for x in getattr(cfg, "step_names", ())}
    for col in df.columns:
        if str(col).strip().lower() in step_names:
            return col
    return None


def _numeric_matrix(df, cols: list[str]) -> np.ndarray:
    out = np.zeros((len(df), len(cols)), dtype=np.float64)
    by_name = {str(c): c for c in df.columns}
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for j, name in enumerate(cols):
        src = _find_column(df, str(name), by_name=by_name, by_lower=by_lower)
        if src is not None:
            out[:, j] = np.nan_to_num(
                pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=np.float64),
                nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _find_column(df, name: str, by_name=None, by_lower=None):
    if by_name is None:
        by_name = {str(c): c for c in df.columns}
    if by_lower is None:
        by_lower = {str(c).strip().lower(): c for c in df.columns}
    src = by_name.get(str(name))
    if src is None:
        src = by_lower.get(str(name).strip().lower())
    return src


def _virtual_dependencies() -> dict[str, list[str]]:
    return {
        "v_001": ["t_002_PV", "t_003_PV"],
        "v_002": ["t_001_PV", "t_003_PV"],
        "v_003": ["t_002_PV", "t_001_PV"],
        "v_004": ["p_008_PV", "p_003_PV"],
        "v_005": ["p_007_PV", "p_003_PV"],
        "v_006": ["p_008_PV", "p_007_PV"],
        "v_007": [
            "h_0211_PV", "h_0212_PV",
            "h_0213_PV", "h_0214_PV",
        ],
        "v_008": [
            "h_0221_PV", "h_0222_PV",
            "h_0223_PV", "h_0224_PV",
        ],
        "v_009": [
            "h_0181_PV", "h_0182_PV",
            "h_0183_PV", "h_0184_PV",
        ],
        "v_010": [
            "h_0191_PV", "h_0192_PV",
            "h_0193_PV", "h_0194_PV",
        ],
    }


def _missing_columns(df, cols: list[str]) -> list[str]:
    by_name = {str(c): c for c in df.columns}
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    return [str(c) for c in cols
            if _find_column(df, str(c), by_name=by_name, by_lower=by_lower) is None]


def _invalid_virtual_columns(raw_df, cols: list[str]) -> list[str]:
    deps = _virtual_dependencies()
    out = []
    for col in cols:
        needed = deps.get(str(col))
        if needed and _missing_columns(raw_df, needed):
            out.append(str(col))
    return out


def _load_record_exact(csv_path: Path, data_root: Path, cfg: IOConfig,
                       io_cols: list[str], sensor_cols: list[str]) -> dict:
    virtual_cfg = IOConfig()
    virtual_cfg.__dict__.update(cfg.__dict__)
    virtual_cfg.virtual_sensor_mode = "both"
    raw_df = _read_csv(csv_path)
    df = add_virtual_sensors(raw_df, virtual_cfg)
    step_col = _find_step_col(df, cfg)
    step = None
    if step_col is not None:
        step = np.nan_to_num(np.asarray(df[step_col], dtype=np.float64), nan=0.0).astype(int)
    missing_sensor_cols = set(_missing_columns(df, sensor_cols))
    missing_sensor_cols.update(_invalid_virtual_columns(raw_df, sensor_cols))
    return {
        "wafer_id": Path(csv_path).stem,
        "recipe": _infer_recipe(Path(csv_path), Path(data_root), cfg),
        "source_path": str(csv_path),
        "io": _numeric_matrix(df, io_cols),
        "io_cols": list(io_cols),
        "sensors": _numeric_matrix(df, sensor_cols),
        "sensor_cols": list(sensor_cols),
        "missing_io_cols": _missing_columns(df, io_cols),
        "missing_sensor_cols": sorted(missing_sensor_cols),
        "step": step,
    }


def _csv_files(data_dir: Path, wafer_csv: Optional[Path]) -> tuple[Path, list[Path]]:
    if wafer_csv is not None:
        path = Path(wafer_csv)
        return path.parent, [path]
    data_dir = Path(data_dir)
    if data_dir.is_file():
        return data_dir.parent, [data_dir]
    # ★ 설비 원본 .log/.tsv(탭구분) 도 직접 읽는다(_read_csv 가 확장자별 구분자 처리).
    files = sorted([p for p in data_dir.rglob("*")
                    if p.suffix.lower() in (".csv", ".log", ".tsv")])
    if not files:
        raise SystemExit(f"No wafer files (.csv/.log/.tsv) found under: {data_dir}")
    return data_dir, files


def _ok_csv_files(cfg: IOConfig, data_root: Path | None = None) -> tuple[Path, list[Path]]:
    candidates: list[Path] = []
    cfg_ok = Path(getattr(cfg, "ok_dir", ""))
    if cfg_ok:
        candidates.append(cfg_ok)
    if data_root is not None:
        root = Path(data_root)
        candidates.extend([
            _sub_ci(root.parent, "OK"),
            _sub_ci(root.parent, "ok"),
            _sub_ci(root, "OK"),
            _sub_ci(root, "ok"),
        ])
    seen: set[str] = set()
    last = cfg_ok
    for candidate in candidates:
        ok_dir = Path(candidate)
        if ok_dir and not ok_dir.exists() and ok_dir.parent.exists():
            ok_dir = _sub_ci(ok_dir.parent, ok_dir.name)
        key = str(ok_dir.resolve()) if ok_dir.exists() else str(ok_dir)
        if key in seen:
            continue
        seen.add(key)
        last = ok_dir
        if ok_dir.is_dir():
            files = sorted([p for p in ok_dir.rglob("*")
                            if p.suffix.lower() in (".csv", ".log", ".tsv")])
            if files:
                return ok_dir, files
    return last, []


def _mean_from_sum_count(sum_arr, count_arr) -> np.ndarray:
    out = np.full(sum_arr.shape, np.nan, dtype=np.float64)
    np.divide(sum_arr, count_arr, out=out, where=count_arr > 0)
    return out


def _build_ok_value_bands(models: list[ArtifactModel], cfg: IOConfig,
                          data_root: Path | None = None,
                          grid: int = 100) -> tuple[dict[str, dict], dict[str, dict]]:
    """Build same-recipe train OK actual-value mean curves per artifact schema."""
    from .recon_explain import _plot_time_filter, _step_fraction_bins

    ok_root, ok_files = _ok_csv_files(cfg, data_root=data_root)
    summaries: dict[str, dict] = {}
    if not ok_files:
        print(f"  ⚠ train OK CSV 없음: OK 평균값 표시 생략 ({ok_root})")
        return {}, summaries

    out: dict[str, dict] = {}
    grid = max(10, int(grid))
    for item in models:
        records = []
        io_cols, sensor_cols = _model_record_columns(item)
        for csv_path in ok_files:
            try:
                records.append(
                    _load_record_exact(csv_path, ok_root, cfg, io_cols, sensor_cols)
                )
            except Exception as e:
                print(f"  ⚠ OK 평균값 record skip({item.label}): {csv_path}: {e}")
        n_sensor = len(getattr(item.model, "sensor_cols", []) or [])
        if not records or n_sensor <= 0:
            summaries[item.label] = {"ok_records": 0, "recipes": []}
            continue
        max_step = max(int(getattr(item.model, "n_steps_global", 0) or 0), 1)
        for rec in records:
            step = rec.get("step")
            if step is not None and len(step):
                max_step = max(max_step, int(np.nanmax(np.asarray(step).astype(int))) + 1)
        recipes = sorted({_recipe for _recipe in (str(r.get("recipe", "unknown") or "unknown") for r in records)})
        recipes = ["__global__"] + [r for r in recipes if r != "__global__"]
        recipe_index = {recipe: idx for idx, recipe in enumerate(recipes)}
        value_sum = np.zeros((len(recipes), n_sensor, max_step, grid), dtype=np.float64)
        value_count = np.zeros((len(recipes), n_sensor, max_step, grid), dtype=np.int32)

        for rec in records:
            try:
                actual = np.asarray(item.model.actual(rec), dtype=np.float64)
                if actual.ndim != 2 or actual.shape[1] != n_sensor:
                    continue
                keep, _orig_idx, _t, step_plot, _bounds, spans = _plot_time_filter(
                    item.model, rec, actual.shape[0])
                if step_plot is None or not len(step_plot) or not np.any(keep):
                    continue
                vals = actual[keep]
                step_arr = np.asarray(step_plot, dtype=int)
                bin_arr = _step_fraction_bins(step_arr, spans, grid)
                valid = (step_arr > 0) & (step_arr < max_step)
                if not valid.any():
                    continue
                flat = step_arr[valid] * grid + bin_arr[valid]
                recipe_ids = [0]
                recipe = str(rec.get("recipe", "unknown") or "unknown")
                if recipe in recipe_index and recipe_index[recipe] != 0:
                    recipe_ids.append(recipe_index[recipe])
                for rid in recipe_ids:
                    for sensor_idx in range(n_sensor):
                        src_vals = vals[valid, sensor_idx]
                        finite = np.isfinite(src_vals)
                        if not finite.any():
                            continue
                        sum_flat = value_sum[rid, sensor_idx].reshape(-1)
                        count_flat = value_count[rid, sensor_idx].reshape(-1)
                        np.add.at(sum_flat, flat[finite], src_vals[finite])
                        np.add.at(count_flat, flat[finite], 1)
            except Exception as e:
                print(f"  ⚠ OK 평균값 계산 skip({item.label}): {rec.get('wafer_id', '-')}: {e}")

        out[item.label] = {
            "recipes": recipes,
            "sensor_cols": [str(x) for x in getattr(item.model, "sensor_cols", [])],
            "grid": grid,
            "value_mean": _mean_from_sum_count(value_sum, value_count),
            "value_count": value_count,
        }
        summaries[item.label] = {
            "ok_records": len(records),
            "recipes": recipes,
            "sensor_count": n_sensor,
            "grid": grid,
        }
        print(f"  · {item.label} OK 평균값 계산: train OK {len(records)}장, recipe {len(recipes) - 1}개")
    return out, summaries


def _ok_value_recipe_index(ok_bands: dict | None, recipe: str) -> tuple[int | None, str]:
    if not ok_bands:
        return None, ""
    recipes = list(ok_bands.get("recipes") or [])
    recipe = str(recipe or "unknown")
    if recipe in recipes:
        return recipes.index(recipe), "same-recipe"
    if "__global__" in recipes:
        return recipes.index("__global__"), "global"
    return None, ""


def _lookup_ok_value_mean(ok_value_bands: dict | None, trace: Trace,
                          step_plot, spans) -> tuple[np.ndarray | None, str]:
    from .recon_explain import _project_ok_grid_to_wafer_axis

    bands = (ok_value_bands or {}).get(str(trace.model_label))
    if not bands or step_plot is None or not len(step_plot):
        return None, ""
    ri, source = _ok_value_recipe_index(bands, trace.rec.get("recipe", "unknown"))
    if ri is None:
        return None, ""
    means = bands.get("value_mean")
    if means is None:
        return None, ""
    sensor_idx = int(trace.sensor_index)
    if sensor_idx < 0 or sensor_idx >= means.shape[1]:
        return None, ""
    curve = _project_ok_grid_to_wafer_axis(
        means[ri, sensor_idx], step_plot, spans, monotonic=False)
    return curve, source


def _family(sensor: str) -> str:
    s = str(sensor).strip()
    low = s.lower()
    for family in ("t", "p", "h", "d"):
        if re.match(r"^" + family + r"(\d+|[_-])", low):
            return family.upper()
    return "IO" if re.fullmatch(r"\d+", s) else "S"


def _normalize_name(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def _resolve_sensor(query: Optional[str], sensors: list[str]) -> Optional[str]:
    if not query:
        return None
    q = str(query).strip()
    qn = _normalize_name(q)
    for name in sensors:
        if str(name).lower() == q.lower() or _normalize_name(name) == qn:
            return name
    m = re.fullmatch(r"([pthPTH])(\d+)", q)
    if m:
        fam, idx_s = m.group(1).upper(), m.group(2)
        idx = int(idx_s) - 1
        family_sensors = [s for s in sensors if _family(s) == fam]
        if 0 <= idx < len(family_sensors):
            return family_sensors[idx]
    for name in sensors:
        if q.lower() in str(name).lower():
            return name
    return None


def _parse_event_families(value=None) -> set[str]:
    if value is None:
        return {"P", "T"}
    if isinstance(value, str):
        parts = re.split(r"[,/+\s]+", value)
    else:
        parts = list(value)
    families = {str(x).strip().upper() for x in parts if str(x).strip()}
    return families or {"P", "T"}


def _artifact_family_allowed(label: str, family: str) -> bool:
    label_u = str(label).upper()
    family_u = str(family).upper()
    if label_u == "PT":
        return family_u in {"P", "T"}
    if label_u == "H":
        return family_u == "H"
    return family_u in {"P", "T", "H"}


def _merge_traces(models: list[ArtifactModel], records: dict[str, dict]) -> tuple[dict[str, Trace], list[dict]]:
    from .recon_explain import _compute, _with_interpolated_explain_arrays

    traces: dict[str, Trace] = {}
    skipped: list[dict] = []
    for item in models:
        rec = records.get(item.label)
        if rec is None:
            continue
        A = item.model
        actual_raw, expected_raw, level_raw = _compute(A, rec)
        view_rec, actual, expected, level = _with_interpolated_explain_arrays(
            A, rec, actual_raw, expected_raw, level_raw)
        signed = np.sign(np.asarray(actual, dtype=np.float64)
                         - np.asarray(expected, dtype=np.float64)) * np.asarray(level, dtype=np.float64)
        for j, sensor in enumerate(getattr(A, "sensor_cols", [])):
            sensor = str(sensor)
            family = _family(sensor)
            if not _artifact_family_allowed(item.label, family):
                continue
            if sensor in set(rec.get("missing_sensor_cols") or []):
                skipped.append({
                    "model": item.label,
                    "sensor": sensor,
                    "reason": "missing-source-column",
                })
                continue
            cur = Trace(sensor=sensor, model_label=item.label, family=family,
                        model=A, rec=view_rec, sensor_index=j,
                        actual=actual[:, j], expected=expected[:, j],
                        level=level[:, j], signed_level=signed[:, j])
            traces[sensor] = cur
    return traces, skipped


def _step_mask(rec: dict, step: Optional[int], peak_index: int,
               half_window: int = 8) -> np.ndarray:
    n = int(rec["sensors"].shape[0])
    mask = np.zeros(n, dtype=bool)
    if step is not None and rec.get("step") is not None and len(rec["step"]) == n:
        mask[np.asarray(rec["step"]).astype(int) == int(step)] = True
    if not mask.any():
        lo = max(0, int(peak_index) - int(half_window))
        hi = min(n, int(peak_index) + int(half_window) + 1)
        mask[lo:hi] = True
    return mask


def _mask_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    spans = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(mask) and mask[i + 1]:
            i += 1
        spans.append((start, i))
        i += 1
    return spans


def _event_candidates(rec: dict, traces: dict[str, Trace], min_level: float,
                      focus_sensor: Optional[str], focus_step: Optional[int],
                      top_events: int, event_families=None) -> list[Event]:
    event_families = _parse_event_families(event_families)
    sensors = [name for name, trace in traces.items() if trace.family in event_families]
    resolved_focus = _resolve_sensor(focus_sensor, sensors)
    if focus_sensor and resolved_focus is None:
        print(f"focus sensor not found in event families {sorted(event_families)}: {focus_sensor}")
        return []
    candidates: list[Event] = []
    for sensor in sensors:
        trace = traces[sensor]
        if resolved_focus is not None and sensor != resolved_focus:
            continue
        level = np.nan_to_num(trace.level, nan=0.0, posinf=0.0, neginf=0.0)
        trace_rec = trace.rec if isinstance(trace.rec, dict) else rec
        if focus_step is not None:
            mask = _step_mask(trace_rec, int(focus_step), int(np.nanargmax(level)))
            if not mask.any():
                continue
            local = np.where(mask, level, -np.inf)
            peak_idx = int(np.nanargmax(local))
            step = int(focus_step)
            peak_level = float(level[peak_idx])
        elif trace_rec.get("step") is not None and len(trace_rec["step"]) == len(level):
            best = None
            for step_value in np.unique(np.asarray(trace_rec["step"]).astype(int)):
                mask = np.asarray(trace_rec["step"]).astype(int) == int(step_value)
                if not mask.any():
                    continue
                local = np.where(mask, level, -np.inf)
                peak_idx = int(np.nanargmax(local))
                peak_level = float(level[peak_idx])
                row = (peak_level, int(step_value), peak_idx)
                if best is None or row[0] > best[0]:
                    best = row
            if best is None:
                continue
            peak_level, step, peak_idx = best
        else:
            peak_idx = int(np.nanargmax(level)) if level.size else 0
            peak_level = float(level[peak_idx]) if level.size else 0.0
            step = None
        if peak_level >= float(min_level) or resolved_focus is not None:
            candidates.append(Event(
                wafer_id=str(trace_rec.get("wafer_id", rec.get("wafer_id", ""))),
                sensor=sensor,
                step=step,
                peak_index=peak_idx,
                peak_level=peak_level,
                model_label=trace.model_label,
            ))
    candidates.sort(key=lambda e: e.peak_level, reverse=True)
    return candidates[:max(1, int(top_events))]


def _undirected_distances(graph: SensorGraph, start: str, max_hops: int) -> dict[str, int]:
    from collections import deque

    if not graph.nodes or start not in graph.nodes:
        return {}
    out = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        depth = out[cur]
        if depth >= max_hops:
            continue
        nxts = set(graph.edges.get(cur, ())) | set(graph.rev.get(cur, ()))
        for nxt in sorted(nxts, key=lambda x: graph.rank.get(x, 10**9)):
            if nxt in out:
                continue
            out[nxt] = depth + 1
            q.append(nxt)
    return out


def _graph_rank_delta(graph: SensorGraph, a: str, b: str) -> int | None:
    if not graph.nodes:
        return None
    rank_a = graph.rank.get(str(a))
    rank_b = graph.rank.get(str(b))
    if rank_a is None or rank_b is None:
        return None
    return abs(int(rank_a) - int(rank_b))


def _shortest_graph_path(graph: SensorGraph, src: str, dst: str,
                         directed: bool = True, max_hops: int = 80) -> list[str]:
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
        for nxt in sorted(nxts, key=lambda x: graph.rank.get(x, 10**9)):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, path + [nxt]))
    return []


def _graph_relation(graph: SensorGraph, focus: str, sensor: str,
                    max_hops: int = 80) -> dict:
    forward = _shortest_graph_path(graph, focus, sensor, directed=True, max_hops=max_hops)
    if forward:
        return {
            "direction": "downstream",
            "path": forward,
            "hop": len(forward) - 1,
            "text": f"focus 센서 기준 downstream 경로 ({len(forward) - 1} hop)",
        }
    backward = _shortest_graph_path(graph, sensor, focus, directed=True, max_hops=max_hops)
    if backward:
        return {
            "direction": "upstream",
            "path": list(reversed(backward)),
            "hop": len(backward) - 1,
            "text": f"focus 센서 기준 upstream 경로 ({len(backward) - 1} hop)",
        }
    undirected = _shortest_graph_path(graph, focus, sensor, directed=False, max_hops=max_hops)
    if undirected:
        return {
            "direction": "adjacent",
            "path": undirected,
            "hop": len(undirected) - 1,
            "text": f"방향성 무시 기준 graph 인접 경로 ({len(undirected) - 1} hop)",
        }
    delta = _graph_rank_delta(graph, focus, sensor)
    if delta is not None:
        return {
            "direction": "order-near",
            "path": [],
            "hop": "",
            "text": f"sensors_order 순서상 거리 {delta}",
        }
    return {
        "direction": "same-step",
        "path": [],
        "hop": "",
        "text": "같은 step residual 관계; graph 경로는 확인되지 않음",
    }


def _path_text(path: list[str], max_nodes: int = 8) -> str:
    if not path:
        return ""
    if len(path) <= max_nodes:
        return " -> ".join(path)
    head = path[:max_nodes // 2]
    tail = path[-max(1, max_nodes // 2):]
    return " -> ".join(head + ["..."] + tail)


def _combined_path_summary(focus: str, related: list[dict],
                           graph: SensorGraph) -> str:
    if not related:
        return "관련 센서가 없어 sensors_order 경로를 종합할 수 없습니다."

    direction_names = {
        "downstream": "downstream",
        "upstream": "upstream",
        "adjacent": "방향 무시 인접",
        "order-near": "sensors_order 순서 근접",
        "same-step": "같은 step residual",
    }
    counts: dict[str, int] = {}
    path_examples: list[str] = []
    no_path: list[str] = []
    for row in related:
        sensor = str(row.get("sensor", ""))
        relation = _graph_relation(graph, focus, sensor)
        direction = str(relation.get("direction", "same-step"))
        counts[direction] = counts.get(direction, 0) + 1
        path = _path_text(relation.get("path") or [], max_nodes=10)
        if path:
            path_examples.append(
                f"{sensor}: {relation.get('text', '-')} `{path}`")
        else:
            no_path.append(f"{sensor}: {relation.get('text', '-')}")

    count_text = ", ".join(
        f"{direction_names.get(k, k)} {v}개"
        for k, v in sorted(counts.items())
    )
    parts = [
        f"focus `{focus}` 기준 관련 센서 {len(related)}개는 {count_text}로 분포합니다."
    ]
    if path_examples:
        shown = path_examples[:4]
        extra = len(path_examples) - len(shown)
        path_text = " / ".join(shown)
        if extra > 0:
            path_text += f" / 외 graph 경로 {extra}개"
        parts.append(f"대표 sensors_order 경로는 {path_text}입니다.")
    if no_path:
        shown = "; ".join(no_path[:3])
        extra = len(no_path) - min(3, len(no_path))
        if extra > 0:
            shown += f"; 외 {extra}개"
        parts.append(f"직접 graph path가 없는 센서는 {shown} 기준으로 포함됐습니다.")
    return " ".join(parts)


def _flow_side(row: dict, focus: str, graph: SensorGraph) -> str:
    side = str(row.get("order_side", "") or "")
    if side in {"before", "after"}:
        return side
    relation = _graph_relation(graph, focus, str(row.get("sensor", "")))
    direction = str(relation.get("direction", ""))
    if direction == "upstream":
        return "before"
    if direction == "downstream":
        return "after"
    return "other"


def _row_flow_sort_key(row: dict) -> tuple:
    if row.get("order_rank", "") != "":
        try:
            return (0, int(row.get("order_rank")))
        except Exception:
            pass
    if row.get("graph_rank_delta", "") != "":
        try:
            return (1, int(row.get("graph_rank_delta")))
        except Exception:
            pass
    return (2, str(row.get("sensor", "")))


def _sensor_level_text(rows: list[dict], limit: int = 5) -> str:
    if not rows:
        return "없음"
    shown = []
    for row in rows[:max(0, int(limit))]:
        shown.append(
            f"{row.get('sensor', '-')}({_format_float(row.get('step_level'))}x)")
    extra = len(rows) - len(shown)
    if extra > 0:
        shown.append(f"외 {extra}개")
    return ", ".join(shown)


def _flow_detail_text(rows: list[dict], event: Event, graph: SensorGraph,
                      limit: int = 5) -> str:
    if not rows:
        return "없음"
    shown = []
    for row in rows[:max(0, int(limit))]:
        sensor = str(row.get("sensor", "-"))
        relation = _graph_relation(graph, event.sensor, sensor)
        relation_text = str(relation.get("text", "-"))
        path = _path_text(relation.get("path") or [])
        detail = (
            f"{sensor}({_format_float(row.get('step_level'))}x, "
            f"{relation_text})"
        )
        if path:
            detail += f", 경로 `{path}`"
        shown.append(detail)
    extra = len(rows) - len(shown)
    if extra > 0:
        shown.append(f"외 {extra}개")
    return "; ".join(shown)


def _flow_story_text(event: Event, before: list[dict],
                     after: list[dict], other: list[dict]) -> str:
    step_text = "unknown" if event.step is None else str(event.step)
    focus_text = f"{event.sensor} 최대 이상({_format_float(event.peak_level)}x)"
    before_text = _sensor_level_text(before, limit=3)
    after_text = _sensor_level_text(after, limit=3)
    other_text = _sensor_level_text(other, limit=3)
    if before and after:
        return (
            f"Step {step_text}에서 앞쪽 {before_text} 센서 영향으로 "
            f"{focus_text} 발생, 그 후 뒤쪽 {after_text} 센서에 같은 step "
            "이상이 이어진 흐름으로 볼 수 있습니다.")
    if before:
        return (
            f"Step {step_text}에서 앞쪽 {before_text} 센서 영향이 "
            f"{focus_text}으로 모인 흐름으로 볼 수 있습니다.")
    if after:
        return (
            f"Step {step_text}에서 {focus_text} 발생 후 뒤쪽 {after_text} "
            "센서에 영향이 이어진 흐름으로 볼 수 있습니다.")
    if other:
        return (
            f"Step {step_text}에서 {focus_text}과 함께 방향 미확정 센서 "
            f"{other_text}가 같이 흔들린 패턴입니다.")
    return (
        f"Step {step_text}에서 {focus_text}만 뚜렷하고, 같은 step에서 "
        "흐름상 같이 설명할 센서는 충분히 잡히지 않았습니다.")


def _flow_groups(event: Event, related: list[dict],
                 graph: SensorGraph) -> tuple[list[dict], list[dict], list[dict]]:
    before = sorted(
        [row for row in related if _flow_side(row, event.sensor, graph) == "before"],
        key=_row_flow_sort_key,
    )
    after = sorted(
        [row for row in related if _flow_side(row, event.sensor, graph) == "after"],
        key=_row_flow_sort_key,
    )
    other = sorted(
        [row for row in related if _flow_side(row, event.sensor, graph) == "other"],
        key=_row_flow_sort_key,
    )
    return before, after, other


def _flow_interpretation_lines(event: Event, related: list[dict],
                               graph: SensorGraph) -> list[str]:
    before, after, other = _flow_groups(event, related, graph)
    before_max = max([float(row.get("step_level", 0.0) or 0.0) for row in before] or [0.0])
    after_max = max([float(row.get("step_level", 0.0) or 0.0) for row in after] or [0.0])
    before_sum = sum(float(row.get("step_level", 0.0) or 0.0) for row in before)
    after_sum = sum(float(row.get("step_level", 0.0) or 0.0) for row in after)

    lines = [
        "## 센서 흐름 종합",
        "",
        f"- 주요 흐름: {_flow_story_text(event, before, after, other)}",
        "",
    ]
    lines.extend(["## 원인/결과 추정", ""])

    if before and after:
        if after_sum > before_sum * 1.25 and after_max >= before_max:
            guess = (
                "뒤쪽 센서들의 동반 이상이 더 강합니다. 현재 정보만 보면 "
                f"{event.sensor}가 upstream 쪽 변화의 원인 후보이거나, "
                "focus 지점 이후로 이상이 전파되는 구간일 가능성이 큽니다.")
        elif before_sum > after_sum * 1.25 and before_max >= after_max:
            guess = (
                "앞쪽 센서들의 동반 이상이 더 강합니다. 현재 정보만 보면 "
                f"{event.sensor}는 앞쪽 공정 변화의 결과로 영향을 받은 센서일 "
                "가능성이 큽니다.")
        else:
            guess = (
                "앞쪽과 뒤쪽 센서가 모두 같이 흔들립니다. 현재 정보만 보면 "
                f"{event.sensor} 단독 원인이라기보다는 공정 흐름 중간의 전파 지점 "
                "또는 공통 원인에 함께 반응한 센서일 가능성이 큽니다.")
    elif after:
        guess = (
            "뒤쪽 센서 위주로 이상이 확인됩니다. 현재 정보만 보면 "
            f"{event.sensor}가 이상 전파의 시작점 또는 upstream 원인 후보일 "
            "가능성이 있습니다.")
    elif before:
        guess = (
            "앞쪽 센서 위주로 이상이 확인됩니다. 현재 정보만 보면 "
            f"{event.sensor}는 앞쪽 이상의 결과로 영향을 받은 센서일 가능성이 있습니다.")
    else:
        guess = (
            "흐름상 동반 센서가 충분히 잡히지 않아 원인/결과 방향을 판단하기 어렵습니다.")

    lines.extend([
        guess,
        "",
        ("이 추정은 sensors_order 순서와 같은 step residual 크기만 사용한 1차 판단입니다. "
         "실제 원인 확정에는 recipe setpoint, valve/IO timing, 반복 wafer 여부를 같이 봐야 합니다."),
        "",
    ])
    return lines


def _same_family_neighbors(sensor: str, traces: dict[str, Trace], limit: int = 3) -> dict[str, str]:
    fam = _family(sensor)
    names = [name for name, trace in traces.items() if trace.family == fam and name != sensor]
    if not names:
        return {}
    names.sort(key=lambda name: _normalize_name(name))
    return {name: "same-family" for name in names[:limit]}


def _parse_related_families(value=None) -> set[str]:
    if value is None:
        return {"P", "T", "H"}
    if isinstance(value, str):
        parts = re.split(r"[,/+\s]+", value)
    else:
        parts = list(value)
    families = {str(x).strip().upper() for x in parts if str(x).strip()}
    return families or {"P", "T", "H"}


def _is_related_trace(trace: Trace, related_families: set[str]) -> bool:
    if trace.family in related_families:
        return True
    return "H" in related_families and str(trace.model_label).upper() == "H"


def _step_level_for_trace(trace: Trace, event: Event) -> float:
    mask = _step_mask(trace.rec, event.step, min(event.peak_index, len(trace.level) - 1))
    if mask.any():
        return float(np.nanmax(trace.level[mask]))
    return float(np.nanmax(trace.level))


def _sensor_order_window(event: Event, traces: dict[str, Trace],
                         graph: SensorGraph, related_families,
                         half_window: int = 3) -> list[dict]:
    related_families = _parse_related_families(related_families)
    if not graph.nodes or event.sensor not in traces or event.sensor not in graph.rank:
        return []
    focus_rank = int(graph.rank[event.sensor])
    candidates = []
    for sensor, trace in traces.items():
        if sensor == event.sensor or not _is_related_trace(trace, related_families):
            continue
        if sensor not in graph.rank:
            continue
        order_rank = int(graph.rank[sensor])
        offset = order_rank - focus_rank
        if offset == 0:
            continue
        side = "before" if offset < 0 else "after"
        candidates.append({
            "sensor": sensor,
            "family": trace.family,
            "model_label": trace.model_label,
            "reason": f"sensors-order-{side}-top-error",
            "hop": "",
            "graph_rank_delta": abs(offset),
            "step_level": _step_level_for_trace(trace, event),
            "order_rank": order_rank,
            "order_offset": offset,
            "order_side": side,
        })
    if not candidates:
        return []

    def anomaly_key(row):
        return (
            -float(row["step_level"]),
            abs(int(row["order_offset"])),
            int(row["order_rank"]),
            str(row["sensor"]),
        )

    before = sorted(
        [row for row in candidates if int(row["order_offset"]) < 0],
        key=anomaly_key,
    )[:max(0, int(half_window))]
    after = sorted(
        [row for row in candidates if int(row["order_offset"]) > 0],
        key=anomaly_key,
    )[:max(0, int(half_window))]
    rows = before + after
    rows.sort(key=lambda row: int(row["order_rank"]))
    return rows


def _related_sensors(event: Event, rec: dict, traces: dict[str, Trace],
                     graph: SensorGraph, max_hops: int, related_top_k: int,
                     min_level: float, related_families=None) -> list[dict]:
    order_window = _sensor_order_window(
        event, traces, graph, related_families, half_window=3)
    if order_window:
        return order_window

    graph_dist = _undirected_distances(graph, event.sensor, max_hops=max_hops)
    related_families = _parse_related_families(related_families)
    rows = []
    for sensor, trace in traces.items():
        if sensor == event.sensor or not _is_related_trace(trace, related_families):
            continue
        step_level = _step_level_for_trace(trace, event)
        if step_level < float(min_level):
            continue
        hop = graph_dist.get(sensor)
        rank_delta = _graph_rank_delta(graph, event.sensor, sensor)
        if hop is not None:
            reason = f"graph-hop-{hop}"
        elif rank_delta is not None:
            reason = f"graph-order-near-{rank_delta}"
        else:
            reason = "same-step-high-error"
        rows.append({
            "sensor": sensor,
            "family": trace.family,
            "model_label": trace.model_label,
            "reason": reason,
            "hop": "" if hop is None else hop,
            "graph_rank_delta": "" if rank_delta is None else rank_delta,
            "step_level": step_level,
        })
    def sort_key(row):
        reason = str(row["reason"])
        graph_rank_delta = row.get("graph_rank_delta", "")
        if reason.startswith("graph-hop-"):
            group = 0
        elif graph_rank_delta != "":
            group = 1
        elif reason == "same-step-high-error":
            group = 2
        else:
            group = 3
        return (
            group,
            -float(row["step_level"]),
            999 if row["hop"] == "" else int(row["hop"]),
            999999 if graph_rank_delta == "" else int(graph_rank_delta),
            row["family"],
            row["sensor"],
        )

    rows.sort(key=sort_key)
    return rows[:max(0, int(related_top_k))]


def _safe_filename(text: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text)).strip("._")
    return out or "sensor"


def _pt_prediction(model_a, rec: dict, threshold, saved_scores: dict | None = None) -> dict:
    if threshold is None:
        return {
            "predicted_ng": False,
            "score": None,
            "threshold": None,
            "source": "missing-threshold",
        }
    from .export_report import _decision_score_from_threshold, _threshold_for_record

    wid = str(rec.get("wafer_id", ""))
    cutoff = _threshold_for_record(threshold, rec)
    score = None
    source = "computed"
    if saved_scores and wid in saved_scores:
        try:
            score = float(saved_scores[wid])
            source = "search_best"
        except Exception:
            score = None
    if score is None:
        score = float(_decision_score_from_threshold(model_a, rec, threshold))
    predicted = bool(cutoff is not None and score >= float(cutoff))
    return {
        "predicted_ng": predicted,
        "score": score,
        "threshold": None if cutoff is None else float(cutoff),
        "source": source,
    }


def _plot_sensor_sequence(event: Event, related: list[dict], traces: dict[str, Trace]) -> list[str]:
    if any("order_rank" in row for row in related):
        before = [
            row for row in related
            if int(row.get("order_offset", 0)) < 0 and row.get("sensor") in traces
        ]
        after = [
            row for row in related
            if int(row.get("order_offset", 0)) > 0 and row.get("sensor") in traces
        ]
        before.sort(key=lambda row: int(row.get("order_rank", 10**9)))
        after.sort(key=lambda row: int(row.get("order_rank", 10**9)))
        sensors = [row["sensor"] for row in before]
        if event.sensor in traces:
            sensors.append(event.sensor)
        sensors.extend(row["sensor"] for row in after)
        return sensors
    sensors = [event.sensor] + [row["sensor"] for row in related]
    return [sensor for sensor in sensors if sensor in traces]


def _io_change_markers(rec: dict, plot_keep: np.ndarray, focus_mask_plot: np.ndarray,
                       max_markers: int = 8, max_labels: int = 2) -> list[dict]:
    io = rec.get("io")
    cols = list(rec.get("io_cols") or [])
    if io is None or not cols:
        return []
    io_arr = np.asarray(io, dtype=np.float64)
    plot_keep = np.asarray(plot_keep, dtype=bool)
    if io_arr.ndim != 2 or io_arr.shape[0] != len(plot_keep):
        return []
    io_plot = io_arr[plot_keep]
    if io_plot.shape[0] < 2 or io_plot.shape[1] == 0:
        return []
    focus_mask_plot = np.asarray(focus_mask_plot, dtype=bool)
    if len(focus_mask_plot) != io_plot.shape[0]:
        focus_mask_plot = np.ones(io_plot.shape[0], dtype=bool)

    delta = np.abs(np.diff(io_plot, axis=0))
    if delta.size == 0:
        return []
    row_mask = focus_mask_plot[1:] | focus_mask_plot[:-1]
    change_rows = np.flatnonzero(row_mask & (np.nanmax(delta, axis=1) > 1e-9))
    markers = []
    for delta_idx in change_rows:
        diffs = np.nan_to_num(delta[delta_idx], nan=0.0, posinf=0.0, neginf=0.0)
        changed = np.flatnonzero(diffs > 1e-9)
        if not changed.size:
            continue
        top_idx = sorted(changed, key=lambda j: (-float(diffs[j]), str(cols[j])))[:max(1, int(max_labels))]
        prev_in_focus = bool(focus_mask_plot[int(delta_idx)])
        next_in_focus = bool(focus_mask_plot[int(delta_idx) + 1])
        changes = []
        for j in top_idx:
            before = float(io_plot[int(delta_idx), int(j)])
            after = float(io_plot[int(delta_idx) + 1, int(j)])
            changes.append({
                "name": str(cols[j]),
                "before": before,
                "after": after,
                "direction": _io_change_direction(before, after),
            })
        markers.append({
            "x": int(delta_idx) + 1,
            "phase": _io_marker_phase(prev_in_focus, next_in_focus),
            "labels": [str(cols[j]) for j in top_idx],
            "changes": changes,
            "count": int(changed.size),
            "score": float(np.sum(diffs[top_idx])),
        })
    markers.sort(key=lambda row: (-float(row["score"]), int(row["x"])))
    markers = markers[:max(0, int(max_markers))]
    markers.sort(key=lambda row: int(row["x"]))
    return markers


def _io_marker_phase(prev_in_focus: bool, next_in_focus: bool) -> str:
    if not prev_in_focus and next_in_focus:
        return "start"
    if prev_in_focus and not next_in_focus:
        return "end"
    if prev_in_focus and next_in_focus:
        return "inside"
    return "nearby"


def _io_marker_phase_label(phase: str) -> str:
    return {
        "start": "Focus step 시작 변경",
        "end": "Focus step 끝 변경",
        "inside": "Focus step 내부 변경",
        "nearby": "Focus step 주변 변경",
    }.get(str(phase), "Focus step IO/Valve 변경")


def _io_value_text(value: float) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.3g}"


def _io_change_direction(before: float, after: float) -> str:
    try:
        prev = float(before)
        cur = float(after)
    except Exception:
        return "변경"
    if prev <= 0.0 < cur:
        return "열림"
    if prev > 0.0 >= cur:
        return "닫힘"
    if cur > prev:
        return "증가"
    if cur < prev:
        return "감소"
    return "변경"


def _io_marker_label(marker: dict) -> str:
    changes = list(marker.get("changes") or [])
    if changes:
        parts = []
        for change in changes:
            before = _io_value_text(change.get("before", ""))
            after = _io_value_text(change.get("after", ""))
            direction = str(change.get("direction", "변경"))
            parts.append(
                f"{change.get('name', '-')} {direction}({before}->{after})")
        extra = int(marker.get("count", len(changes))) - len(changes)
        text = " / ".join(parts)
        if extra > 0:
            text += f" / +{extra}"
        return text
    labels = list(marker.get("labels") or [])
    if not labels:
        return ""
    extra = int(marker.get("count", len(labels))) - len(labels)
    text = "/".join(labels)
    if extra > 0:
        text += f"+{extra}"
    return text


def _draw_io_change_markers(ax, markers: list[dict], show_legend: bool = False) -> None:
    for idx, marker in enumerate(markers):
        x = float(marker.get("x", 0))
        ax.axvline(
            x,
            color="#7c3aed",
            lw=0.8,
            ls=(0, (2, 2)),
            alpha=0.65,
            zorder=3,
            label="IO/Valve change" if idx == 0 and show_legend else None,
        )


def _top_io_marker_lines(markers: list[dict], line_width: int = 82,
                         max_lines: int = 5) -> list[str]:
    if not markers:
        return []
    lines = []
    used = 0
    phase_order = ("start", "end", "inside", "nearby")
    grouped = {
        phase: [m for m in markers if str(m.get("phase", "nearby")) == phase]
        for phase in phase_order
    }
    for phase in phase_order:
        phase_markers = grouped.get(phase) or []
        if not phase_markers:
            continue
        pieces = []
        for marker in phase_markers:
            label = _io_marker_label(marker)
            if label:
                pieces.append(f"x{int(marker.get('x', 0))}: {label}")
        if not pieces:
            continue
        prefix = f"{_io_marker_phase_label(phase)}: "
        cur = prefix
        cur_count = 0
        for piece in pieces:
            token = piece if cur_count == 0 else f", {piece}"
            if len(cur) + len(token) > int(line_width) and cur_count > 0:
                lines.append(cur)
                used += cur_count
                if len(lines) >= int(max_lines):
                    break
                cur = "  " + piece
                cur_count = 1
            else:
                cur += token
                cur_count += 1
        if len(lines) >= int(max_lines):
            break
        if cur_count > 0:
            lines.append(cur)
            used += cur_count
        if len(lines) >= int(max_lines):
            break
    total = sum(
        1 for marker in markers
        if _io_marker_label(marker)
    )
    remaining = max(0, total - used)
    if remaining and lines:
        lines[-1] += f", +{remaining} more"
    return lines


def _draw_top_header(left_ax, right_ax, markers: list[dict], wafer_text: str,
                     detail_text: str = "",
                     lines: list[str] | None = None) -> int:
    lines = _top_io_marker_lines(markers) if lines is None else list(lines)
    left_ax.set_axis_off()
    right_ax.set_axis_off()
    if lines:
        left_ax.text(
            0.0,
            0.52,
            "\n".join(lines),
            transform=left_ax.transAxes,
            ha="left",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="#4c1d95",
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": "white",
                "ec": "#8b5cf6",
                "lw": 0.8,
                "alpha": 0.96,
            },
        )
    title = str(wafer_text or "")
    if title:
        right_ax.text(
            1.0,
            0.64 if detail_text else 0.52,
            title,
            transform=right_ax.transAxes,
            ha="right",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="#111827",
        )
    if detail_text:
        right_ax.text(
            1.0,
            0.30,
            str(detail_text),
            transform=right_ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#374151",
        )
    return len(lines)


def _plot_event(out_path: Path, rec: dict, event: Event, traces: dict[str, Trace],
                related: list[dict], title_suffix: str = "",
                ok_value_bands: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from .recon_explain import (
        _draw_step_guides,
        _limit_series,
        _plot_time_filter,
        _setup_font,
        _time_axis_label,
    )

    _setup_font()
    import matplotlib.pyplot as plt

    sensors = _plot_sensor_sequence(event, related, traces)
    max_rows = max(1, len(sensors))
    top_io_markers: list[dict] = []
    focus_trace = traces.get(event.sensor)
    if focus_trace is not None:
        focus_T = int(len(focus_trace.actual))
        focus_keep, _focus_orig_idx, _focus_t, _focus_step_plot, _focus_bounds, _focus_spans = (
            _plot_time_filter(focus_trace.model, focus_trace.rec, focus_T)
        )
        focus_mask_full = _step_mask(
            focus_trace.rec, event.step, min(event.peak_index, focus_T - 1))
        focus_mask_plot = (
            focus_mask_full[focus_keep]
            if len(focus_mask_full) == focus_T
            else np.zeros(int(np.sum(focus_keep)), dtype=bool)
        )
        top_io_markers = _io_change_markers(
            focus_trace.rec, focus_keep, focus_mask_plot,
            max_markers=10, max_labels=3)
    top_io_marker_lines = _top_io_marker_lines(top_io_markers)
    top_io_lines = len(top_io_marker_lines)
    header_lines = max(2, top_io_lines)
    header_ratio = min(1.05, max(0.44, 0.18 + 0.15 * header_lines))
    fig_h = min(20.0, max(4.2, 2.15 * max_rows + 0.95 + 0.16 * header_lines))
    fig = plt.figure(figsize=(18, fig_h))
    gs = fig.add_gridspec(
        max_rows + 1, 2,
        height_ratios=[header_ratio] + [1.0] * max_rows,
        width_ratios=[0.64, 0.36],
        hspace=0.10,
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
    for row_idx, (ax, sensor) in enumerate(zip(axes, sensors)):
        trace = traces[sensor]
        is_focus = sensor == event.sensor
        t_full = int(len(trace.actual))
        plot_keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter(
            trace.model, trace.rec, t_full)
        if orig_idx.size == 0:
            ax.axis("off")
            continue
        cur = np.asarray(trace.actual, dtype=np.float64)[plot_keep]
        exp = np.asarray(trace.expected, dtype=np.float64)[plot_keep]
        ok_limit = _limit_series(trace.model, trace.rec, trace.sensor_index, t_full)[plot_keep]
        band_lo = exp - ok_limit
        band_hi = exp + ok_limit
        ok_mean, ok_mean_source = _lookup_ok_value_mean(
            ok_value_bands, trace, step_plot, spans)
        finite_vals = np.concatenate([
            cur[np.isfinite(cur)],
            exp[np.isfinite(exp)],
            band_lo[np.isfinite(band_lo)],
            band_hi[np.isfinite(band_hi)],
            np.asarray([], dtype=np.float64) if ok_mean is None
            else ok_mean[np.isfinite(ok_mean)],
        ])
        if finite_vals.size:
            ylo = float(np.nanmin(finite_vals))
            yhi = float(np.nanmax(finite_vals))
        else:
            ylo, yhi = 0.0, 1.0
        pad = (yhi - ylo) * 0.08 + 1e-6
        mask_full = _step_mask(trace.rec, event.step, min(event.peak_index, t_full - 1))
        mask_plot = (mask_full[plot_keep]
                     if len(mask_full) == t_full
                     else np.zeros_like(cur, dtype=bool))
        for start, end in _mask_spans(mask_plot):
            ax.axvspan(float(start) - 0.5, float(end) + 0.5,
                       color="#f03b20", alpha=0.30 if is_focus else 0.11,
                       lw=0, zorder=0,
                       label="focus step" if is_focus else None)
        ax.fill_between(
            t, band_lo, band_hi,
            color="#9ca3af",
            alpha=0.16,
            zorder=1,
            label="OK 기준 정상 범위" if row_idx == 0 else None,
        )
        io_markers = _io_change_markers(trace.rec, plot_keep, mask_plot)
        _draw_io_change_markers(ax, io_markers, show_legend=(row_idx == 0))
        _draw_step_guides(ax, bounds, spans, label=True)
        ax.plot(t, exp, color="#1f77b4", lw=1.4, ls="--", label="기대", zorder=4)
        ax.plot(t, cur, color="#d62728", lw=1.4, label="실제", zorder=5)
        if ok_mean is not None and np.isfinite(ok_mean).any():
            ok_label = "같은 recipe OK 평균값"
            if ok_mean_source == "global":
                ok_label = "전체 OK 평균값"
            ax.plot(
                t, ok_mean,
                color="#059669",
                lw=1.25,
                ls=(0, (5, 2)),
                label=ok_label if row_idx == 0 else None,
                zorder=3,
            )
        if is_focus:
            peak_pos = np.flatnonzero(orig_idx == int(event.peak_index))
            if peak_pos.size:
                p = int(peak_pos[0])
                ax.scatter([t[p]], [cur[p]], s=38, color="#b91c1c",
                           edgecolor="white", lw=0.7, zorder=6)
            ax.text(0.006, 0.90, "FOCUS", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, fontweight="bold",
                    color="#b91c1c",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white",
                          "ec": "#b91c1c", "lw": 0.8, "alpha": 0.92},
                    zorder=7)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_ylabel(("FOCUS\n" if is_focus else "") + sensor,
                      rotation=0, ha="right", va="center", fontsize=8,
                      fontweight="bold" if is_focus else "normal",
                      color="#b91c1c" if is_focus else "black")
        ax.tick_params(labelsize=8)
        ax.margins(x=0.01)
    if max_rows > 1:
        for ax in axes[:-1]:
            ax.label_outer()
    axes[0].legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    step_text = "unknown" if event.step is None else str(event.step)
    interp = max(1, int(traces[event.sensor].rec.get("_interp_factor", 1) or 1))
    axes[-1].set_xlabel(_time_axis_label(traces[event.sensor].rec))
    detail_text = (
        f"focus={event.sensor} | step={step_text} | interp x{interp} {title_suffix}"
    ).strip()
    _draw_top_header(
        header_left_ax, header_right_ax, top_io_markers, wafer_text=str(event.wafer_id),
        detail_text=detail_text, lines=top_io_marker_lines)
    fig.tight_layout(rect=(0.16, 0.03, 0.99, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _format_float(value, digits: int = 3) -> str:
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_event_narrative(report_path: Path, event: Event, rec: dict,
                           traces: dict[str, Trace], related: list[dict],
                           graph: SensorGraph, pred: dict, plot_path: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    step_text = "unknown" if event.step is None else str(event.step)
    pred_label = "NG" if pred.get("predicted_ng") else "OK"
    lines = [
        f"# {event.wafer_id} Step {step_text} 센서 흐름 분석 리포트",
        "",
        "## 요약",
        "",
        (f"- P/T 모델 판정: {pred_label} "
         f"(score {_format_float(pred.get('score'))} / threshold "
         f"{_format_float(pred.get('threshold'))}, 기준={pred.get('source', '-')})"),
        f"- 기준 센서: {event.sensor}",
        f"- 기준 센서 peak 이상점수: {_format_float(event.peak_level)} x OK limit",
        f"- 원본 wafer: {rec.get('source_path', '-')}",
        f"- Overlay 이미지: {plot_path}",
        "",
        "## 이 구간을 표시한 이유",
        "",
        (f"Step {step_text}에서 P/T 복원 모델 기준 {event.sensor}의 실제값과 "
         f"기대값 차이가 OK residual 기준을 크게 벗어났습니다. "
         "센서 흐름 종합은 sensors_order.json 기준으로 focus 앞쪽/뒤쪽을 나눈 뒤, "
         "각 방향에서 같은 step 이상점수가 높은 센서를 focus 전후 흐름으로 정리했습니다."),
        "",
        "- 회색 band: 학습 OK wafer에서 저장된 residual 기준 정상 범위(expected ± OK_limit)",
        "- 초록 점선: 같은 recipe train OK wafer의 해당 센서 실제값 평균",
        "- 보라색 점선: focus step 안에서 값이 바뀐 IO/Valve timing. 변경 IO명과 열림/닫힘 방향은 이미지 맨 위 왼쪽 compact header에 시작/끝/내부 변경으로 나눠 표시하고, wafer 이름과 focus/step 정보는 오른쪽에 표시",
        "",
    ]
    lines.extend(_flow_interpretation_lines(event, related, graph))
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _split_semicolon(value) -> list[str]:
    return [x for x in str(value or "").split(";") if x]


def _top_counter_text(counter: Counter, total: int, label: str,
                      limit: int = 3, min_count: int = 2) -> str:
    items = [
        (str(name), int(count))
        for name, count in counter.most_common()
        if str(name) and str(name) != "-" and int(count) >= int(min_count)
    ]
    if not items:
        return f"{label}: 뚜렷한 반복 없음"
    shown = [
        f"{name} {count}/{total}건"
        for name, count in items[:max(1, int(limit))]
    ]
    extra = len(items) - len(shown)
    if extra > 0:
        shown.append(f"외 {extra}개")
    return f"{label}: " + ", ".join(shown)


def _recipe_common_feature_lines(row: dict, recipe_rows: list[dict]) -> list[str]:
    recipe = str(row.get("recipe") or "-")
    peers = list(recipe_rows)
    total = len(peers)
    wafer_count = len({str(r.get("wafer_id", "")) for r in peers if r.get("wafer_id")})
    lines = [
        "## 같은 recipe 공통 특징",
        "",
        f"- 비교 대상: recipe `{recipe}` overlay event {total}건 / wafer {wafer_count}개",
    ]
    if total < 2:
        lines.extend([
            "- 공통 특징: 같은 recipe에서 비교할 다른 overlay event가 부족해 반복 패턴은 판단 보류입니다.",
            "",
        ])
        return lines

    step_counts = Counter(
        f"Step {r.get('step')}" for r in peers
        if str(r.get("step", "")).strip()
    )
    focus_counts = Counter(str(r.get("focus_sensor", "")) for r in peers)
    family_counts = Counter(str(r.get("focus_family", "")) for r in peers)
    related_counts: Counter = Counter()
    before_counts: Counter = Counter()
    after_counts: Counter = Counter()
    for peer in peers:
        related_counts.update(_split_semicolon(peer.get("related_sensors")))
        before_counts.update(_split_semicolon(peer.get("before_sensors")))
        after_counts.update(_split_semicolon(peer.get("after_sensors")))

    lines.extend([
        f"- {_top_counter_text(step_counts, total, '반복 step')}",
        f"- {_top_counter_text(focus_counts, total, '반복 focus sensor')}",
        f"- {_top_counter_text(family_counts, total, '반복 focus 계열')}",
        f"- {_top_counter_text(related_counts, total, '반복 동반 센서')}",
    ])
    before_text = _top_counter_text(before_counts, total, "반복 upstream 쪽 센서")
    after_text = _top_counter_text(after_counts, total, "반복 downstream 쪽 센서")
    if "뚜렷한 반복 없음" not in before_text:
        lines.append(f"- {before_text}")
    if "뚜렷한 반복 없음" not in after_text:
        lines.append(f"- {after_text}")

    has_common_step = bool(step_counts and step_counts.most_common(1)[0][1] >= 2)
    has_common_focus = bool(focus_counts and focus_counts.most_common(1)[0][1] >= 2)
    has_common_related = bool(related_counts and related_counts.most_common(1)[0][1] >= 2)
    if has_common_step or has_common_focus or has_common_related:
        lines.append(
            "- 해석: 같은 recipe에서 위 항목이 반복되므로 recipe 공통 setpoint, "
            "valve/IO timing, 또는 같은 공정 구간의 제어 조건을 우선 확인하는 것이 좋습니다.")
    else:
        lines.append(
            "- 해석: 같은 recipe 안에서도 반복되는 step/focus/동반 센서가 뚜렷하지 않아, "
            "recipe 공통 조건보다는 wafer별 국소 편차나 개별 장비 상태를 먼저 의심할 수 있습니다.")
    lines.append("")
    return lines


def _append_recipe_common_sections(event_rows: list[dict]) -> dict[str, dict]:
    by_recipe: dict[str, list[dict]] = {}
    for row in event_rows:
        by_recipe.setdefault(str(row.get("recipe") or "-"), []).append(row)

    summary: dict[str, dict] = {}
    for recipe, rows in by_recipe.items():
        total = len(rows)
        step_counts = Counter(
            f"Step {r.get('step')}" for r in rows
            if str(r.get("step", "")).strip()
        )
        focus_counts = Counter(str(r.get("focus_sensor", "")) for r in rows)
        related_counts: Counter = Counter()
        for row in rows:
            related_counts.update(_split_semicolon(row.get("related_sensors")))
        summary[recipe] = {
            "events": total,
            "wafers": len({str(r.get("wafer_id", "")) for r in rows if r.get("wafer_id")}),
            "common_steps": step_counts.most_common(5),
            "common_focus_sensors": focus_counts.most_common(5),
            "common_related_sensors": related_counts.most_common(8),
        }

    for row in event_rows:
        report = Path(row.get("narrative_report", ""))
        if not report.exists():
            continue
        recipe = str(row.get("recipe") or "-")
        lines = _recipe_common_feature_lines(row, by_recipe.get(recipe, []))
        current = report.read_text(encoding="utf-8").rstrip()
        report.write_text(current + "\n\n" + "\n".join(lines), encoding="utf-8")
    return summary


def generate_related_overlays(
    pt_artifacts: Path,
    h_artifacts: Path,
    data_dir: Path,
    out_dir: Path,
    cfg: IOConfig,
    wafer_csv: Optional[Path] = None,
    wafer_id: Optional[str] = None,
    focus_sensor: Optional[str] = None,
    focus_step: Optional[int] = None,
    top_events: int = 3,
    related_top_k: int = 8,
    min_level: float = 1.0,
    related_max_hops: int = 4,
    event_families=None,
    related_families=None,
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
    pt_scores = pt_best.get("scores") or pt_best.get("anomaly_scores") or {}
    if not isinstance(pt_scores, dict):
        pt_scores = {}
    cfg.sensors_order_path = Path(graph_path) if graph_path else cfg.sensors_order_path
    graph = load_sensor_graph(cfg, Path(graph_path) if graph_path else None)
    data_root, files = _csv_files(Path(data_dir), wafer_csv)
    ok_value_bands, ok_value_summary = _build_ok_value_bands(
        models, cfg, data_root=data_root)
    if wafer_id:
        files = [p for p in files if Path(p).stem == wafer_id or wafer_id in Path(p).stem]
    if max_wafers is not None:
        files = files[:max(0, int(max_wafers))]
    if not files:
        raise SystemExit("No wafer CSV matched the requested filters.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    narrative_dir = out_dir / "natural_language_reports"
    narrative_dir.mkdir(parents=True, exist_ok=True)
    event_rows = []
    skipped_rows = []
    summary = {
        "pt_artifacts": str(pt_artifacts),
        "h_artifacts": str(h_artifacts),
        "data_dir": str(data_root),
        "graph_path": str(Path(graph_path) if graph_path else cfg.sensors_order_path),
        "thresholds": {
            "PT": pt_threshold,
            "H": _read_search_threshold(h_artifacts),
        },
        "trace_source_policy": {
            "P": "PT artifact only",
            "T": "PT artifact only",
            "H": "H artifact only",
        },
        "record_policy": "Each artifact is evaluated with its own saved io_cols/input_sensor_cols/sensor_cols order.",
        "ok_value_mean_policy": "Each panel overlays the same-recipe train OK actual-value mean for that sensor as a green dashed line, computed per artifact schema and aligned by ProcessStep plus within-step relative position. If a recipe-specific mean is unavailable, the global OK mean is used when available.",
        "ok_value_mean": ok_value_summary,
        "plot_policy": "Reuse tcn-explain time view: drop ProcessStep 0 and apply configured interpolation. When sensors_order rank is available, choose the 3 highest focus-step anomaly sensors before the focus sensor and the 3 highest after it, then plot them in process order around the focus sensor. Every panel draws the focus-step red span; focus panel span alpha is 0.30 and related panel span alpha is 0.11. Every panel also draws a gray OK normal band from expected +/- OK_limit and a green dashed same-recipe OK actual-value mean line. Purple dotted markers show IO/Valve changes inside the focus step. Changed IO names and open/close directions are summarized in a compact header row at the very top-left of the image, while wafer/focus/step metadata is placed at the top-right to avoid overlap. No automatic 4-sigma spans are drawn.",
        "narrative_policy": "Natural-language reports keep only the main sensor-flow sentence in the sensor-flow section, then infer whether the focus sensor is more likely a cause candidate, result candidate, propagation point, or common-response sensor. After all events are generated, each report appends a same-recipe common-feature summary based on repeated step, focus sensor/family, and related sensors.",
        "event_families": sorted(_parse_event_families(event_families)),
        "related_families": sorted(_parse_related_families(related_families)),
        "prediction_filter": prediction_filter,
        "narrative_dir": str(narrative_dir),
        "narrative_reports": [],
        "skipped_trace_columns": [],
        "outputs": [],
    }

    for csv_path in files:
        model_records = _load_model_records(csv_path, data_root, cfg, models)
        rec = model_records.get("PT") or next(iter(model_records.values()))
        pt_rec = model_records.get(models[0].label, rec)
        pred = _pt_prediction(models[0].model, pt_rec, pt_threshold, pt_scores)
        if prediction_filter == "predicted-ng" and not pred["predicted_ng"]:
            skipped_rows.append({
                "wafer_id": rec["wafer_id"],
                "source_path": rec.get("source_path", ""),
                "pt_score": "" if pred["score"] is None else pred["score"],
                "pt_threshold": "" if pred["threshold"] is None else pred["threshold"],
                "prediction_source": pred["source"],
                "skip_reason": "pt-predicted-ok",
            })
            continue
        traces, skipped_traces = _merge_traces(models, model_records)
        for row in skipped_traces:
            summary["skipped_trace_columns"].append({
                "wafer_id": rec["wafer_id"],
                "source_path": rec.get("source_path", ""),
                **row,
            })
        if not traces:
            continue
        events = _event_candidates(
            rec, traces, min_level=min_level, focus_sensor=focus_sensor,
            focus_step=focus_step, top_events=top_events,
            event_families=event_families)
        for event_idx, event in enumerate(events, start=1):
            related = _related_sensors(
                event, rec, traces, graph, max_hops=related_max_hops,
                related_top_k=related_top_k, min_level=min_level,
                related_families=related_families)
            before, after, other = _flow_groups(event, related, graph)
            flow_story = _flow_story_text(event, before, after, other)
            step_text = "nostep" if event.step is None else f"step{event.step}"
            filename = (
                f"{_safe_filename(event.wafer_id)}__event{event_idx:02d}__"
                f"{_safe_filename(event.sensor)}__{step_text}.png"
            )
            out_path = out_dir / filename
            _plot_event(out_path, rec, event, traces, related,
                        ok_value_bands=ok_value_bands)
            report_name = (
                f"{_safe_filename(event.wafer_id)}__event{event_idx:02d}__"
                f"{_safe_filename(event.sensor)}__{step_text}.md"
            )
            report_path = narrative_dir / report_name
            _write_event_narrative(report_path, event, rec, traces, related,
                                   graph, pred, out_path)
            summary["outputs"].append(str(out_path))
            summary["narrative_reports"].append(str(report_path))
            event_rows.append({
                "wafer_id": event.wafer_id,
                "recipe": rec.get("recipe", ""),
                "source_path": rec.get("source_path", ""),
                "event_rank": event_idx,
                "focus_sensor": event.sensor,
                "focus_family": traces[event.sensor].family,
                "focus_model": event.model_label,
                "step": "" if event.step is None else event.step,
                "peak_index": event.peak_index,
                "peak_level": round(event.peak_level, 6),
                "pt_score": "" if pred["score"] is None else round(float(pred["score"]), 6),
                "pt_threshold": "" if pred["threshold"] is None else round(float(pred["threshold"]), 6),
                "prediction_source": pred["source"],
                "related_sensors": ";".join(row["sensor"] for row in related),
                "related_reasons": ";".join(f"{row['sensor']}={row['reason']}" for row in related),
                "before_sensors": ";".join(row["sensor"] for row in before),
                "after_sensors": ";".join(row["sensor"] for row in after),
                "other_sensors": ";".join(row["sensor"] for row in other),
                "flow_story": flow_story,
                "narrative_report": str(report_path),
                "plot_path": str(out_path),
            })

    summary["recipe_common_features"] = _append_recipe_common_sections(event_rows)

    csv_path = out_dir / "related_overlay_events.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "wafer_id", "recipe", "source_path", "event_rank", "focus_sensor", "focus_family",
            "focus_model", "step", "peak_index", "peak_level", "pt_score",
            "pt_threshold", "prediction_source", "related_sensors", "related_reasons",
            "before_sensors", "after_sensors", "other_sensors", "flow_story",
            "narrative_report", "plot_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_rows)
    skipped_csv_path = out_dir / "related_overlay_skipped_wafers.csv"
    with open(skipped_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "wafer_id", "source_path", "pt_score", "pt_threshold",
            "prediction_source", "skip_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped_rows)
    summary["event_csv"] = str(csv_path)
    summary["skipped_csv"] = str(skipped_csv_path)
    summary["processed_wafers"] = len({row["wafer_id"] for row in event_rows})
    summary["skipped_wafers"] = len(skipped_rows)
    index_lines = [
        "# Related Overlay Natural Language Reports",
        "",
        f"- Event reports: {len(event_rows)}",
        f"- Source graph: {summary['graph_path']}",
        "",
    ]
    if summary.get("recipe_common_features"):
        index_lines.extend(["## Recipe Common Features", ""])
        for recipe, info in sorted(summary["recipe_common_features"].items()):
            steps = ", ".join(
                f"{name} {count}건"
                for name, count in info.get("common_steps", [])[:3]
            ) or "반복 step 없음"
            focus = ", ".join(
                f"{name} {count}건"
                for name, count in info.get("common_focus_sensors", [])[:3]
            ) or "반복 focus 없음"
            index_lines.append(
                f"- `{recipe}`: event {info.get('events', 0)}건 / "
                f"wafer {info.get('wafers', 0)}개, {steps}, {focus}"
            )
        index_lines.append("")
    for row in event_rows:
        report = Path(row["narrative_report"])
        index_lines.append(
            f"- [{row['wafer_id']} event {row['event_rank']} "
            f"{row['focus_sensor']} step {row['step']}]({report.name})"
        )
    (narrative_dir / "index.md").write_text("\n".join(index_lines) + "\n",
                                            encoding="utf-8")
    (out_dir / "related_overlay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"related overlay images: {len(summary['outputs'])} -> {out_dir}")
    print(f"event manifest: {csv_path}")
    print(f"skipped wafer manifest: {skipped_csv_path}")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Create combined P/T/H related-sensor overlay plots.")
    ap.add_argument("--pt-artifacts", required=True,
                    help="Artifacts dir containing the P/T tcn_forecast.pt model.")
    ap.add_argument("--h-artifacts", required=True,
                    help="Artifacts dir containing the H tcn_forecast.pt model.")
    ap.add_argument("--data-dir", default="/home/yapark/semes/dataset/val",
                    help="Directory containing wafer CSV files.")
    ap.add_argument("--wafer-csv", default=None,
                    help="Optional single wafer CSV path.")
    ap.add_argument("--wafer-id", default=None,
                    help="Only process wafer CSVs whose stem contains this value.")
    ap.add_argument("--out-dir", default="reports/related_overlay",
                    help="Output directory for PNG/CSV/JSON files.")
    ap.add_argument("--focus-sensor", default=None,
                    help="Optional focus sensor or alias, for example T4.")
    ap.add_argument("--focus-step", type=int, default=None,
                    help="Optional ProcessStep to highlight for the focus sensor.")
    ap.add_argument("--top-events", type=int, default=3)
    ap.add_argument("--related-top-k", type=int, default=8)
    ap.add_argument("--related-max-hops", type=int, default=4)
    ap.add_argument("--event-families", default="P,T",
                    help="Sensor families allowed to create events. Default: P,T.")
    ap.add_argument("--related-families", default="P,T,H",
                    help="Sensor families allowed in related rows. Default: P,T,H.")
    ap.add_argument("--include-all-wafers", action="store_true",
                    help="Disable the default PT predicted-NG wafer filter.")
    ap.add_argument("--min-level", type=float, default=1.0,
                    help="Minimum |actual-expected|/OK_limit level for auto events.")
    ap.add_argument("--max-wafers", type=int, default=None)
    ap.add_argument("--graph-path", default=None,
                    help="Optional sensor graph JSON path. Defaults to config sensors_order_path.")
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
    generate_related_overlays(
        pt_artifacts=Path(args.pt_artifacts),
        h_artifacts=Path(args.h_artifacts),
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        cfg=cfg,
        wafer_csv=Path(args.wafer_csv) if args.wafer_csv else None,
        wafer_id=args.wafer_id,
        focus_sensor=args.focus_sensor,
        focus_step=args.focus_step,
        top_events=args.top_events,
        related_top_k=args.related_top_k,
        min_level=args.min_level,
        related_max_hops=args.related_max_hops,
        event_families=args.event_families,
        related_families=args.related_families,
        prediction_filter="all" if args.include_all_wafers else "predicted-ng",
        max_wafers=args.max_wafers,
        graph_path=Path(args.graph_path) if args.graph_path else None,
    )


if __name__ == "__main__":
    main()
