"""IO-Recon-TCN CLI.

Primary commands:
  python run.py tcn-search
  python run.py tcn-export
  python run.py recon-explain

Transformer command aliases are kept for operator muscle memory:
  transformer-search/export/explain -> same TCN implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .config import (
    h_SENSOR_PRESETS,
    IOConfig,
    SENSOR_ROLE_PROFILE_H_SPLIT_V1,
    h_sensor_keep_for_config,
    normalize_h_sensor_preset,
    normalize_sensor_role_profile,
)
from .data import load_dataset


PKG_ROOT = Path(__file__).resolve().parent
# 앱(학습탭)에서 '고정 파라미터 학습' 시 단일값 search space 를 주입하기 위한 env 우회(기본 동작 불변)
SEARCH_SPACE_PATH = Path(os.environ.get("IO_SEARCH_SPACE") or (PKG_ROOT / "search_space.json"))
REQUIRED_TCN_SEARCH_SPACE_KEYS = (
    "tcn_past_window",
    "tcn_future_window",
    "tcn_stride",
    "tcn_channels",
    "tcn_blocks",
    "tcn_kernel",
    "tcn_dropout",
    "tcn_lr",
    "tcn_epochs",
    "tcn_batch",
    "tcn_weight_decay",
    "use_step_onehot",
    "raw_error_quantile",
)
IGNORED_TCN_SEARCH_SPACE_KEYS = {"resid_agg", "wafer_agg", "wafer_topk"}


def _load_tcn_search_space(path: Path = SEARCH_SPACE_PATH) -> dict:
    if not path.exists():
        sys.exit(f"search_space.json 없음: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"search_space.json 로드 실패({path}): {e}")
    if not isinstance(data, dict):
        sys.exit(f"search_space.json 형식 오류({path}): dict가 아님")
    out = {}
    valid_keys = set(IOConfig.__dataclass_fields__)
    for key, values in data.items():
        if str(key) in IGNORED_TCN_SEARCH_SPACE_KEYS:
            continue
        if str(key) not in valid_keys:
            sys.exit(f"search_space.json 알 수 없는 항목: {key}")
        if not isinstance(values, list) or not values:
            sys.exit(f"search_space.json 형식 오류: {key} 값은 비어있지 않은 list여야 함")
        out[str(key)] = list(values)
    missing = [k for k in REQUIRED_TCN_SEARCH_SPACE_KEYS if k not in out]
    if missing:
        sys.exit(f"search_space.json 누락 항목: {', '.join(missing)}")
    return out


def _resolve_pkg_path(value, base: Path = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or PKG_ROOT) / path


def _apply_cli_paths(cfg: IOConfig, data_name: str = None,
                     artifacts_name: str = None,
                     artifacts_dir: str = None) -> None:
    if data_name:
        raw = Path(data_name)
        if raw.is_absolute():
            data_base = raw
        elif raw.parts and raw.parts[0].lower() == "data":
            data_base = PKG_ROOT / raw
        else:
            data_base = PKG_ROOT / "data" / raw
        cfg.ok_dir = data_base / "OK"
        cfg.ng_dir = data_base / "NG"
        cfg.val_dir = data_base / "VAL"
    if artifacts_dir:
        cfg.artifacts_dir = _resolve_pkg_path(artifacts_dir)
    elif artifacts_name:
        cfg.artifacts_dir = _resolve_pkg_path(artifacts_name)


def _resolve_report_dir(cfg: IOConfig, report_name: str = None,
                        report_dir: str = None) -> Path:
    if report_dir:
        return _resolve_pkg_path(report_dir)
    name = report_name or "io_rca"
    path = Path(name)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "reports":
        return PKG_ROOT / path
    return Path(cfg.artifacts_dir).parent / "reports" / path


def _resolve_fresh_report_dir(cfg: IOConfig, report_name: str = None,
                              report_dir: str = None) -> Path:
    base = _resolve_report_dir(cfg, report_name, report_dir)
    if not base.exists():
        return base
    for idx in range(1, 1000):
        candidate = base.parent / f"{base.name}_try{idx:02d}"
        if not candidate.exists():
            print(f"report dir exists: {base} -> {candidate}")
            return candidate
    raise RuntimeError(f"사용 가능한 report dir 이름을 찾지 못했습니다: {base}")


def _write_command_record(out_dir: Path, filename: str, command_kind: str,
                          artifacts_dir: Path = None, report_dir: Path = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    lines = [
        f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"cwd: {Path.cwd()}",
        f"command_kind: {command_kind}",
        f"command: {shlex.join(sys.argv)}",
        f"python_command: {shlex.join([sys.executable, *sys.argv])}",
    ]
    if artifacts_dir is not None:
        lines.append(f"artifacts_dir: {Path(artifacts_dir)}")
    if report_dir is not None:
        lines.append(f"report_dir: {Path(report_dir)}")
    for key in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"):
        value = os.environ.get(key)
        if value:
            lines.append(f"env.{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sub_ci(base, name):
    base = Path(base)
    for c in (name, name.upper(), name.capitalize()):
        if (base / c).is_dir():
            return base / c
    if base.is_dir():
        lname = str(name).lower()
        try:
            for child in base.iterdir():
                if child.is_dir() and child.name.lower() == lname:
                    return child
        except OSError:
            pass
    return base / name


def _recipe_manifest_name(recipe: str) -> str:
    name = str(recipe or "unknown").strip() or "unknown"
    safe = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", name)
    return safe or "unknown"


def _write_val_recipe_files(cfg: IOConfig, val_ok, val_ng) -> Path:
    out_dir = Path(cfg.artifacts_dir) / "val_recipes"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["recipe", "wafer_id", "label", "source_path"]
    grouped = {}
    for label, records in (("OK", list(val_ok or [])), ("NG", list(val_ng or []))):
        for r in records:
            recipe = str(r.get("recipe", "unknown") or "unknown")
            key = recipe if recipe and recipe != "unknown" else "unknown"
            grouped.setdefault(key, []).append({
                "recipe": recipe,
                "wafer_id": r.get("wafer_id", ""),
                "label": label,
                "source_path": r.get("source_path", ""),
            })
    for recipe in getattr(cfg, "recipe_names", ()) or ():
        grouped.setdefault(str(recipe), [])
    grouped.setdefault("unknown", [])
    for recipe, rows in sorted(grouped.items(), key=lambda kv: (kv[0] == "unknown", kv[0])):
        path = out_dir / f"{_recipe_manifest_name(recipe)}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"VAL recipe 구분 파일 저장: {out_dir}")
    return out_dir


def _load_forecast_split(cfg: IOConfig, seed: int) -> dict:
    ok, io_cols, sensor_cols, ns_ok = load_dataset(cfg.ok_dir, cfg)
    val_ok_dir = _sub_ci(cfg.val_dir, "ok")
    val_ng_dir = _sub_ci(cfg.val_dir, "ng")
    vok, _, _, ns_vok = load_dataset(val_ok_dir, cfg)
    vng, _, _, ns_vng = load_dataset(val_ng_dir, cfg)
    ngd, _, _, ns_ng = load_dataset(cfg.ng_dir, cfg)
    ng = vng if vng else ngd
    if vok:
        train_ok, val_ok = ok, vok
        split_mode = (f"fixed folders - Train_OK={Path(cfg.ok_dir)}({len(train_ok)}) · "
                      f"Val_OK={val_ok_dir}({len(val_ok)}) · "
                      f"NG={val_ng_dir if vng else Path(cfg.ng_dir)}({len(ng)})")
    else:
        frac = float(getattr(cfg, "train_ok_frac", 0.8))
        idx = np.arange(len(ok))
        np.random.default_rng(seed).shuffle(idx)
        n_val = max(1, int(round(len(ok) * (1 - frac)))) if len(ok) > 1 else 0
        val_ok = [ok[i] for i in idx[:n_val]]
        train_ok = [ok[i] for i in idx[n_val:]]
        split_mode = (f"auto split - train_ok {len(train_ok)} / val_ok {len(val_ok)} "
                      f"from {Path(cfg.ok_dir)} · NG={len(ng)}")
    nstep = max(ns_ok, ns_vok, ns_vng, ns_ng, 1)
    _write_val_recipe_files(cfg, val_ok, ng)
    return dict(train_ok=train_ok, val_ok=val_ok, ng=ng, io_cols=io_cols,
                sensor_cols=sensor_cols, nstep=nstep, ok_all=ok,
                split_mode=split_mode)


def _threshold_candidates_midpoint(p):
    p = np.asarray(p, dtype=float)
    vals = np.unique(p[np.isfinite(p)])
    if vals.size == 0:
        return np.asarray([0.0], dtype=float)
    vals = np.sort(vals)[::-1]
    eps = max(1e-12, float(np.nanmax(np.abs(vals))) * 1e-12)
    cands = [float(vals[0] + eps)]
    cands.extend(float((a + b) / 2.0) for a, b in zip(vals[:-1], vals[1:]))
    cands.append(float(vals[-1] - eps))
    return np.asarray(cands, dtype=float)


def _metrics_at_threshold(y, p, th):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= float(th)).astype(int)
    n_ok = int((y == 0).sum())
    n_ng = int((y == 1).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "correct": tp + tn, "n": n_ok + n_ng, "tp": tp, "tn": tn,
        "fn": fn, "fp": fp, "n_ok": n_ok, "n_ng": n_ng,
        **_rate_metrics(tp, fp, fn),
    }


def _safe_auroc(y, p):
    try:
        from sklearn.metrics import roc_auc_score
        return round(float(roc_auc_score(y, p)), 4)
    except Exception:
        return 0.0


def _rate_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) \
        if (precision + recall) else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
    }


def _search_objective(cfg: IOConfig = None) -> str:
    mode = str(getattr(cfg, "search_objective", "correct") or "correct").lower()
    return "precision" if mode in {"precision", "precision_first"} else "correct"


def _search_objective_label(cfg: IOConfig = None) -> str:
    if _search_objective(cfg) == "precision":
        return "precision → correct wafers → TP → fewer FP → low OK forecast MSE"
    return "correct wafers → OK correct wafers → TP → low OK forecast MSE"


def _search_objective_goal_label(cfg: IOConfig = None) -> str:
    if _search_objective(cfg) == "precision":
        return "validation precision 최대화"
    return "validation OK+NG 정답 wafer 수 최대화"


def _search_objective_tiebreak_label(cfg: IOConfig = None) -> str:
    if _search_objective(cfg) == "precision":
        return "정답 wafer 수 -> 불량 TP -> FP 적음 -> OK forecast MSE 낮은 모델"
    return "OK 맞춘 개수 -> 불량 TP -> OK forecast MSE 낮은 모델"


def _precision_value(metrics: dict) -> float:
    try:
        metrics = metrics or {}
        if metrics.get("precision", None) not in (None, ""):
            return float(metrics.get("precision", 0.0) or 0.0)
        tp = int(metrics.get("tp", 0) or 0)
        fp = int(metrics.get("fp", 0) or 0)
        return float(tp / (tp + fp)) if (tp + fp) else 0.0
    except Exception:
        return 0.0


def _ensure_rate_metrics(metrics: dict) -> dict:
    if not isinstance(metrics, dict):
        return {}
    if all(k in metrics for k in ("precision", "recall", "f1_score")):
        return metrics
    try:
        metrics.update(_rate_metrics(
            int(metrics.get("tp", 0) or 0),
            int(metrics.get("fp", 0) or 0),
            int(metrics.get("fn", 0) or 0),
        ))
    except Exception:
        pass
    return metrics


def _metrics_rank(metrics: dict, cfg: IOConfig = None) -> tuple:
    metrics = metrics or {}
    if _search_objective(cfg) == "precision":
        return (
            _precision_value(metrics),
            int(metrics.get("correct", 0) or 0),
            int(metrics.get("tp", 0) or 0),
            -int(metrics.get("fp", 0) or 0),
        )
    return (
        int(metrics.get("correct", 0) or 0),
        int(metrics.get("tn", 0) or 0),
        int(metrics.get("tp", 0) or 0),
        -int(metrics.get("fp", 0) or 0),
    )


def _threshold_candidate_rank(metrics: dict, threshold: float, cfg: IOConfig = None) -> tuple:
    if _search_objective(cfg) == "precision":
        return (*_metrics_rank(metrics, cfg), float(threshold))
    return (
        int((metrics or {}).get("correct", 0) or 0),
        int((metrics or {}).get("tn", 0) or 0),
        int((metrics or {}).get("tp", 0) or 0),
        float(threshold),
    )


def _result_obj_from_val(val: dict, val_mse, cfg: IOConfig = None) -> list:
    try:
        mse_val = float(val_mse)
    except Exception:
        mse_val = float("inf")
    if _search_objective(cfg) == "precision":
        return [
            _precision_value(val),
            int((val or {}).get("correct", 0) or 0),
            int((val or {}).get("tp", 0) or 0),
            -int((val or {}).get("fp", 0) or 0),
            -mse_val,
        ]
    return [
        int((val or {}).get("correct", 0) or 0),
        int((val or {}).get("tn", 0) or 0),
        int((val or {}).get("tp", 0) or 0),
        -mse_val,
    ]


def _correct_from_scores(y, p, cfg: IOConfig = None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    best = None
    for th in _threshold_candidates_midpoint(p):
        m = _metrics_at_threshold(y, p, th)
        cand = (_threshold_candidate_rank(m, float(th), cfg), float(th), m)
        if best is None or cand[0] > best[0]:
            best = cand
    _, th, metrics = best
    return {**metrics, "threshold": round(float(th), 4),
            "threshold_raw": float(th), "auroc": _safe_auroc(y, p)}


def _correct_from_nonnegative_scores(y, p, cfg: IOConfig = None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    eps = max(1e-12, float(np.nanmax(np.abs(p))) * 1e-12) if p.size else 1e-12
    cands = [float(x) for x in _threshold_candidates_midpoint(p) if float(x) > eps]
    if not cands:
        cands = [_no_event_threshold(p)]
    best = None
    for th in cands:
        m = _metrics_at_threshold(y, p, th)
        cand = (_threshold_candidate_rank(m, float(th), cfg), float(th), m)
        if best is None or cand[0] > best[0]:
            best = cand
    _, th, metrics = best
    return {**metrics, "threshold": round(float(th), 4),
            "threshold_raw": float(th), "auroc": _safe_auroc(y, p)}


def _recipe_key(rec) -> str:
    value = rec.get("recipe", "unknown") if isinstance(rec, dict) else "unknown"
    return str(value or "unknown")


def _score_topk(cfg: IOConfig) -> int:
    try:
        value = int(getattr(cfg, "wafer_topk", 0) or 0)
    except Exception:
        value = 0
    return max(0, value)


def _score_agg_enabled(cfg: IOConfig = None) -> bool:
    return bool(getattr(cfg, "score_agg_enabled", False)) if cfg is not None else False


def _score_resid_agg(cfg: IOConfig = None) -> str:
    if not _score_agg_enabled(cfg):
        return "sum"
    mode = str(getattr(cfg, "resid_agg", "sum") or "sum").strip().lower()
    return mode if mode in {"sum", "p99", "max", "mean"} else "sum"


def _score_wafer_agg(cfg: IOConfig = None) -> str:
    if not _score_agg_enabled(cfg):
        return "sum"
    mode = str(getattr(cfg, "wafer_agg", "sum") or "sum").strip().lower()
    return mode if mode in {"sum", "topk_mean", "max", "mean"} else "sum"


def _score_time_avg(cfg: IOConfig = None) -> bool:
    return bool(getattr(cfg, "score_time_avg", False)) if cfg is not None else False


def _score_final_avg(cfg: IOConfig = None) -> bool:
    return bool(getattr(cfg, "score_final_avg", False)) if cfg is not None else False


def _score_kind_for_cfg(cfg: IOConfig) -> str:
    topk = _score_topk(cfg)
    if _score_agg_enabled(cfg):
        resid = _score_resid_agg(cfg)
        wafer = _score_wafer_agg(cfg)
        topk_part = f"_top{topk}"
        base = f"resid_{resid}_wafer_{wafer}{topk_part}_sensor_reconstruction_error"
        if _score_time_avg(cfg):
            return f"time_avg_{base}"
        if _score_final_avg(cfg):
            return f"final_avg_{base}"
        return base
    if topk > 0:
        base = f"sum_top{topk}_sensor_reconstruction_error"
    else:
        base = "sum_all_sensor_reconstruction_error"
    if _score_time_avg(cfg):
        return f"time_avg_{base}"
    if _score_final_avg(cfg):
        return f"final_avg_{base}"
    return base


def _score_metadata_for_cfg(cfg: IOConfig) -> dict:
    return {
        "score_kind": _score_kind_for_cfg(cfg),
        "score_topk": _score_topk(cfg),
        "score_agg_enabled": _score_agg_enabled(cfg),
        "score_resid_agg": _score_resid_agg(cfg),
        "score_wafer_agg": _score_wafer_agg(cfg),
        "score_time_avg": _score_time_avg(cfg),
        "score_final_avg": _score_final_avg(cfg),
    }


def _per_sensor_reconstruction_scores_from_level(level, resid_agg: str = "sum") -> np.ndarray:
    level = np.asarray(level, dtype=np.float64)
    if level.ndim != 2 or level.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    level = np.nan_to_num(np.maximum(0.0, level), nan=0.0, posinf=0.0, neginf=0.0)
    resid_agg = str(resid_agg or "sum").strip().lower()
    if resid_agg == "p99":
        return np.percentile(level, 99.0, axis=0)
    if resid_agg == "max":
        return np.max(level, axis=0)
    if resid_agg == "mean":
        return np.mean(level, axis=0)
    return np.sum(level, axis=0)


def _aggregate_sensor_scores(per_sensor, wafer_agg: str = "sum", topk: int = 0) -> float:
    per_sensor = np.asarray(per_sensor, dtype=np.float64)
    if per_sensor.size == 0:
        return 0.0
    per_sensor = np.nan_to_num(per_sensor, nan=0.0, posinf=0.0, neginf=0.0)
    wafer_agg = str(wafer_agg or "sum").strip().lower()
    if int(topk or 0) > 0 and int(topk) < per_sensor.size:
        order = np.argsort(per_sensor)[::-1][:int(topk)]
        selected = per_sensor[order]
    else:
        selected = per_sensor
    if selected.size == 0:
        return 0.0
    if wafer_agg in {"topk_mean", "mean"}:
        return float(np.mean(selected))
    if wafer_agg == "max":
        return float(np.max(selected))
    return float(np.sum(selected))


def _score_time_denominator(level, valid_steps=None) -> float:
    if valid_steps is not None:
        if np.isscalar(valid_steps):
            try:
                value = int(valid_steps)
            except Exception:
                value = 0
        else:
            try:
                value = int(np.asarray(valid_steps, dtype=bool).sum())
            except Exception:
                value = 0
    else:
        arr = np.asarray(level)
        value = int(arr.shape[0]) if arr.ndim else 0
    return float(max(1, value))


def _reconstruction_score_from_level(level, cfg: IOConfig = None, valid_steps=None) -> float:
    resid_agg = _score_resid_agg(cfg)
    wafer_agg = _score_wafer_agg(cfg)
    topk = _score_topk(cfg or IOConfig())
    per_sensor = _per_sensor_reconstruction_scores_from_level(level, resid_agg)
    score = _aggregate_sensor_scores(per_sensor, wafer_agg, topk)
    if _score_time_avg(cfg) or _score_final_avg(cfg):
        score /= _score_time_denominator(level, valid_steps=valid_steps)
    return float(score)


def _sum_topk_sensor_reconstruction_score_from_level(level, topk: int = 0) -> float:
    per_sensor = _per_sensor_reconstruction_scores_from_level(level, "sum")
    if per_sensor.size == 0:
        return 0.0
    if int(topk or 0) > 0 and int(topk) < per_sensor.size:
        order = np.argsort(per_sensor)[::-1][:int(topk)]
        return float(per_sensor[order].sum())
    return float(per_sensor.sum())


def _recipe_lookup_key(recipe) -> str:
    return str(recipe or "").strip().lower()


def _parse_threshold_recipe_multipliers(values) -> tuple:
    if not values:
        return ()
    if isinstance(values, dict):
        values = [f"{k}={v}" for k, v in values.items()]
    elif isinstance(values, str):
        values = [values]
    out = []
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            parts = [value]
        else:
            parts = []
            for item in str(value).split(","):
                item = item.strip()
                if not item:
                    continue
                sep = "=" if "=" in item else ":" if ":" in item else None
                if sep is None:
                    raise ValueError(f"recipe multiplier 형식 오류: {item} (예: 29a=1.1)")
                parts.append(item.split(sep, 1))
        for recipe, factor in parts:
            recipe = str(recipe).strip()
            if not recipe:
                raise ValueError("recipe multiplier의 recipe 이름이 비어 있습니다")
            try:
                factor = float(factor)
            except Exception as e:
                raise ValueError(f"{recipe} multiplier 숫자 변환 실패: {factor}") from e
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(f"{recipe} multiplier는 0보다 큰 유한값이어야 합니다: {factor}")
            out.append(f"{recipe}={factor:.12g}")
    return tuple(out)


def _threshold_recipe_multiplier_map(cfg: IOConfig) -> dict:
    table = {}
    for item in _parse_threshold_recipe_multipliers(
            getattr(cfg, "threshold_recipe_multipliers", ())):
        recipe, factor = str(item).split("=", 1)
        table[_recipe_lookup_key(recipe)] = float(factor)
    return table


def _threshold_recipe_multiplier_for(cfg: IOConfig, recipe) -> float:
    return float(_threshold_recipe_multiplier_map(cfg).get(_recipe_lookup_key(recipe), 1.0))


def _display_threshold_recipe_multipliers(cfg: IOConfig) -> dict:
    out = {}
    for item in _parse_threshold_recipe_multipliers(
            getattr(cfg, "threshold_recipe_multipliers", ())):
        recipe, factor = str(item).split("=", 1)
        out[recipe] = float(factor)
    return out


def _append_multiplier_recipes(recipes: list, cfg: IOConfig) -> None:
    for recipe in _display_threshold_recipe_multipliers(cfg):
        if recipe not in recipes:
            recipes.append(recipe)


def _apply_recipe_threshold_multiplier(recipe, th, vc, yy, pp, cfg: IOConfig):
    base_th = float(th)
    factor = _threshold_recipe_multiplier_for(cfg, recipe)
    if abs(factor - 1.0) <= 1e-12:
        return base_th, vc
    adjusted_th = base_th * factor
    adjusted_vc = _metrics_at_threshold(yy, pp, adjusted_th)
    adjusted_vc["threshold"] = round(float(adjusted_th), 4)
    adjusted_vc["threshold_raw"] = float(adjusted_th)
    adjusted_vc["auroc"] = _safe_auroc(yy, pp)
    adjusted_vc["base_threshold"] = round(float(base_th), 4)
    adjusted_vc["base_threshold_raw"] = float(base_th)
    adjusted_vc["threshold_multiplier"] = float(factor)
    return adjusted_th, adjusted_vc


def _threshold_object_from_base_with_recipe_multipliers(base_th: float, cfg: IOConfig,
                                                        records=None,
                                                        score_kind: str = None,
                                                        strategy: str = None) -> dict | float:
    multipliers = _display_threshold_recipe_multipliers(cfg)
    topk = _score_topk(cfg)
    if (not multipliers and topk <= 0 and not _score_agg_enabled(cfg)
            and not _score_time_avg(cfg) and not _score_final_avg(cfg)):
        return float(base_th)
    recipes = []
    for rec in records or []:
        recipe = _recipe_key(rec)
        if recipe not in recipes:
            recipes.append(recipe)
    _append_multiplier_recipes(recipes, cfg)
    threshold_by_recipe = {}
    base_threshold_by_recipe = {}
    for recipe in recipes:
        factor = _threshold_recipe_multiplier_for(cfg, recipe)
        if abs(factor - 1.0) <= 1e-12 and recipe not in multipliers:
            continue
        base_threshold_by_recipe[recipe] = float(base_th)
        threshold_by_recipe[recipe] = float(base_th) * factor
    out = {
        "threshold_wise": "all",
        "threshold": round(float(base_th), 4),
        "threshold_raw": float(base_th),
        "global_threshold": round(float(base_th), 4),
        "global_threshold_raw": float(base_th),
        "threshold_by_recipe": threshold_by_recipe,
        "base_threshold_by_recipe": base_threshold_by_recipe,
        "threshold_recipe_multipliers": multipliers,
        **_score_metadata_for_cfg(cfg),
    }
    if score_kind:
        out["score_kind"] = str(score_kind)
    if strategy:
        out["threshold_strategy"] = str(strategy)
    return out


def _correct_from_nonnegative_scores_all(records, y, p, cfg: IOConfig):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    vc = _correct_from_nonnegative_scores(y, p, cfg)
    if vc is None:
        return None
    base_th = float(vc["threshold_raw"])
    threshold_obj = _threshold_object_from_base_with_recipe_multipliers(
        base_th, cfg, records=records,
        score_kind=_score_kind_for_cfg(cfg),
        strategy="global_midpoint_between_adjacent_sum_all_sensor_reconstruction_scores")
    if not isinstance(threshold_obj, dict):
        return vc
    pred = np.asarray([
        int(score >= float((threshold_obj.get("threshold_by_recipe") or {}).get(
            _recipe_key(rec), base_th)))
        for rec, score in zip(records, p)
    ], dtype=int)
    out = _metrics_from_pred(y, pred, p)
    out.update({
        "threshold": threshold_obj,
        "threshold_raw": threshold_obj,
        "threshold_wise": "all",
        **_score_metadata_for_cfg(cfg),
        "global_threshold": round(base_th, 4),
        "global_threshold_raw": base_th,
        "threshold_recipe_multipliers": threshold_obj.get("threshold_recipe_multipliers"),
        "threshold_strategy": (
            "global_midpoint_between_adjacent_sum_all_sensor_reconstruction_scores"
            "_with_recipe_threshold_multipliers"
        ),
    })
    return out


def _correct_from_scores_by_recipe(records, y, p, cfg: IOConfig):
    records = list(records)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    global_vc = _correct_from_scores(y, p, cfg)
    if global_vc is None:
        return None
    fallback_th = float(global_vc["threshold_raw"])
    recipes = []
    for r in getattr(cfg, "recipe_names", ()) or ():
        rr = str(r)
        if rr not in recipes:
            recipes.append(rr)
    for r in records:
        rr = _recipe_key(r)
        if rr not in recipes:
            recipes.append(rr)
    _append_multiplier_recipes(recipes, cfg)
    threshold_by_recipe, recipe_metrics = {}, {}
    base_threshold_by_recipe = {}
    multipliers = _display_threshold_recipe_multipliers(cfg)
    for recipe in recipes:
        idx = [i for i, r in enumerate(records) if _recipe_key(r) == recipe]
        if not idx:
            if recipe not in multipliers:
                continue
            base_th = fallback_th
            factor = _threshold_recipe_multiplier_for(cfg, recipe)
            threshold_by_recipe[recipe] = base_th * factor
            base_threshold_by_recipe[recipe] = base_th
            recipe_metrics[recipe] = {
                "correct": 0, "n": 0, "tp": 0, "tn": 0, "fn": 0, "fp": 0,
                "n_ok": 0, "n_ng": 0, "threshold": round(base_th * factor, 4),
                "threshold_raw": base_th * factor, "base_threshold": round(base_th, 4),
                "base_threshold_raw": base_th, "threshold_multiplier": factor,
                "auroc": 0.0, "fallback": "no_recipe_records_uses_global_midpoint_threshold",
            }
            continue
        yy, pp = y[idx], p[idx]
        if len(set(yy.tolist())) >= 2:
            vc = _correct_from_scores(yy, pp, cfg)
            th = float(vc["threshold_raw"])
            fallback = None
        else:
            th = fallback_th
            vc = _metrics_at_threshold(yy, pp, th)
            vc["threshold"] = round(float(th), 4)
            vc["threshold_raw"] = float(th)
            vc["auroc"] = 0.0
            fallback = "single_class_recipe_uses_global_midpoint_threshold"
        base_threshold_by_recipe[recipe] = float(th)
        th, vc = _apply_recipe_threshold_multiplier(recipe, th, vc, yy, pp, cfg)
        threshold_by_recipe[recipe] = th
        recipe_metrics[recipe] = {**vc, "fallback": fallback}
    pred = np.asarray([
        int(score >= threshold_by_recipe.get(_recipe_key(rec), fallback_th))
        for rec, score in zip(records, p)
    ], dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_ok = int((y == 0).sum())
    n_ng = int((y == 1).sum())
    out = {
        "correct": tp + tn, "n": n_ok + n_ng, "tp": tp, "tn": tn,
        "fn": fn, "fp": fp, "n_ok": n_ok, "n_ng": n_ng,
        "threshold": {k: round(float(v), 4) for k, v in threshold_by_recipe.items()},
        "threshold_raw": dict(threshold_by_recipe),
        "threshold_by_recipe": dict(threshold_by_recipe),
        "base_threshold_by_recipe": dict(base_threshold_by_recipe),
        "global_threshold": round(fallback_th, 4),
        "global_threshold_raw": fallback_th,
        "recipe_metrics": recipe_metrics,
        "threshold_strategy": "per_recipe_midpoint_between_adjacent_scores",
        **_score_metadata_for_cfg(cfg),
        "auroc": _safe_auroc(y, p),
    }
    if multipliers:
        out["threshold_recipe_multipliers"] = multipliers
        out["threshold_strategy"] += "_with_recipe_threshold_multipliers"
    return out


def _correct_from_nonnegative_scores_by_recipe(records, y, p, cfg: IOConfig):
    records = list(records)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    global_vc = _correct_from_nonnegative_scores(y, p, cfg)
    if global_vc is None:
        return None
    fallback_th = float(global_vc["threshold_raw"])
    recipes = []
    for r in getattr(cfg, "recipe_names", ()) or ():
        rr = str(r)
        if rr not in recipes:
            recipes.append(rr)
    for r in records:
        rr = _recipe_key(r)
        if rr not in recipes:
            recipes.append(rr)
    _append_multiplier_recipes(recipes, cfg)
    threshold_by_recipe, recipe_metrics = {}, {}
    base_threshold_by_recipe = {}
    multipliers = _display_threshold_recipe_multipliers(cfg)
    for recipe in recipes:
        idx = [i for i, r in enumerate(records) if _recipe_key(r) == recipe]
        if not idx:
            if recipe not in multipliers:
                continue
            base_th = fallback_th
            factor = _threshold_recipe_multiplier_for(cfg, recipe)
            threshold_by_recipe[recipe] = base_th * factor
            base_threshold_by_recipe[recipe] = base_th
            recipe_metrics[recipe] = {
                "correct": 0, "n": 0, "tp": 0, "tn": 0, "fn": 0, "fp": 0,
                "n_ok": 0, "n_ng": 0, "threshold": round(base_th * factor, 4),
                "threshold_raw": base_th * factor, "base_threshold": round(base_th, 4),
                "base_threshold_raw": base_th, "threshold_multiplier": factor,
                "auroc": 0.0, "fallback": "no_recipe_records_uses_global_positive_threshold",
            }
            continue
        yy, pp = y[idx], p[idx]
        if len(set(yy.tolist())) >= 2:
            vc = _correct_from_nonnegative_scores(yy, pp, cfg)
            th = float(vc["threshold_raw"])
            fallback = None
        else:
            th = fallback_th
            vc = _metrics_at_threshold(yy, pp, th)
            vc["threshold"] = round(float(th), 4)
            vc["threshold_raw"] = float(th)
            vc["auroc"] = 0.0
            fallback = "single_class_recipe_uses_global_positive_threshold"
        base_threshold_by_recipe[recipe] = float(th)
        th, vc = _apply_recipe_threshold_multiplier(recipe, th, vc, yy, pp, cfg)
        threshold_by_recipe[recipe] = th
        recipe_metrics[recipe] = {**vc, "fallback": fallback}
    pred = np.asarray([
        int(score >= threshold_by_recipe.get(_recipe_key(rec), fallback_th))
        for rec, score in zip(records, p)
    ], dtype=int)
    m = _metrics_from_pred(y, pred, p)
    out = {
        **m,
        "threshold": {k: round(float(v), 4) for k, v in threshold_by_recipe.items()},
        "threshold_raw": dict(threshold_by_recipe),
        "threshold_by_recipe": dict(threshold_by_recipe),
        "base_threshold_by_recipe": dict(base_threshold_by_recipe),
        "global_threshold": round(fallback_th, 4),
        "global_threshold_raw": fallback_th,
        "recipe_metrics": recipe_metrics,
        "threshold_strategy": "per_recipe_midpoint_between_sum_all_sensor_reconstruction_scores",
        **_score_metadata_for_cfg(cfg),
    }
    if multipliers:
        out["threshold_recipe_multipliers"] = multipliers
        out["threshold_strategy"] += "_with_recipe_threshold_multipliers"
    return out


def _threshold_wise(cfg: IOConfig) -> str:
    mode = str(getattr(cfg, "threshold_wise", "recipe") or "recipe").lower()
    return mode if mode in {"all", "recipe", "sensor", "step"} else "recipe"


def _threshold_source(cfg: IOConfig) -> str:
    mode = str(getattr(cfg, "threshold_source", "val-search") or "val-search")
    mode = mode.strip().lower().replace("_", "-")
    return "train-ok-sigma" if mode in {"train-ok-sigma", "train-sigma"} else "val-search"


def _threshold_sigma(cfg: IOConfig) -> float:
    try:
        value = float(getattr(cfg, "threshold_sigma", -1.0))
    except Exception:
        value = -1.0
    return value if np.isfinite(value) else -1.0


def _threshold_sigma_auto(cfg: IOConfig) -> bool:
    return _threshold_sigma(cfg) < 0.0


def _threshold_sigma_fixed(cfg: IOConfig, fallback: float = 3.5) -> float:
    value = _threshold_sigma(cfg)
    return value if value >= 0.0 else float(fallback)


def _threshold_sigma_bound(cfg: IOConfig, name: str) -> float | None:
    raw = getattr(cfg, name, None)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _threshold_sigma_range(cfg: IOConfig) -> tuple[float | None, float | None]:
    return (
        _threshold_sigma_bound(cfg, "threshold_sigma_min"),
        _threshold_sigma_bound(cfg, "threshold_sigma_max"),
    )


def _threshold_sigma_range_enabled(cfg: IOConfig) -> bool:
    mn, mx = _threshold_sigma_range(cfg)
    return mn is not None or mx is not None


def _threshold_sigma_range_metadata(cfg: IOConfig) -> dict:
    mn, mx = _threshold_sigma_range(cfg)
    return {
        "threshold_sigma_min": None if mn is None else float(mn),
        "threshold_sigma_max": None if mx is None else float(mx),
    }


def _format_threshold_sigma_range(cfg: IOConfig) -> str:
    mn, mx = _threshold_sigma_range(cfg)
    lo = "-inf" if mn is None else f"{float(mn):g}"
    hi = "inf" if mx is None else f"{float(mx):g}"
    return f"[{lo}, {hi}]"


def _validate_threshold_sigma_range_for_search(cfg: IOConfig) -> None:
    if _threshold_source(cfg) != "train-ok-sigma":
        return
    mn, mx = _threshold_sigma_range(cfg)
    if mn is not None and mn < 0.0:
        sys.exit("--threshold-sigma-min은 0 이상이어야 합니다.")
    if mx is not None and mx < 0.0:
        sys.exit("--threshold-sigma-max는 0 이상이어야 합니다.")
    if mn is not None and mx is not None and mn > mx:
        sys.exit("--threshold-sigma-min 값이 --threshold-sigma-max 값보다 큽니다.")
    if _threshold_sigma_range_enabled(cfg) and not _threshold_sigma_auto(cfg):
        sys.exit(
            "threshold sigma 범위 옵션 오류: --threshold-sigma-min/--threshold-sigma-max는 "
            "--threshold-sigma -1 자동 sigma 탐색에서만 사용할 수 있습니다. "
            f"현재 --threshold-sigma {_threshold_sigma(cfg):g} 고정값이 지정되어 있어 "
            "범위 탐색을 수행하지 않습니다. 고정값을 쓰려면 min/max 옵션을 제거하고, "
            "범위 탐색을 원하면 --threshold-sigma -1로 실행하세요."
        )


def _stored_threshold_wise(threshold) -> str | None:
    if isinstance(threshold, dict) and threshold.get("threshold_wise"):
        return str(threshold.get("threshold_wise")).lower()
    return None


_THRESHOLD_METADATA_KEYS = {
    "threshold_wise", "score_kind", "score_name", "score_topk",
    "score_agg_enabled", "score_resid_agg", "score_wafer_agg",
    "score_time_avg", "score_final_avg",
    "threshold_strategy", "threshold_recipe_multipliers",
    "base_threshold_by_recipe", "recipe_metrics",
    "global_threshold", "global_threshold_raw", "default", "global",
    "decision_threshold", "decision_rule", "formula", "thresholds",
    "threshold_source", "threshold_sigma", "threshold_sigma_mode",
    "threshold_sigma_requested", "threshold_sigma_candidates", "train_ok_score_mean",
    "train_ok_score_std", "train_ok_score_n", "train_ok_score_stats",
}


def _is_threshold_metadata_key(key) -> bool:
    return str(key or "").strip().lower() in _THRESHOLD_METADATA_KEYS


def _float_threshold_value(value) -> float | None:
    if isinstance(value, bool) or isinstance(value, (dict, list, tuple)):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _saved_threshold_for_record(threshold, rec):
    if threshold is None:
        return None
    if not isinstance(threshold, dict):
        return _float_threshold_value(threshold)
    mode = str(threshold.get("threshold_wise", "")).lower()
    if mode in {"sensor", "step"}:
        return _float_threshold_value(threshold.get("decision_threshold", 1.0))
    recipe = _recipe_key(rec)
    if recipe in threshold and not _is_threshold_metadata_key(recipe):
        value = _float_threshold_value(threshold[recipe])
        if value is not None:
            return value
    wanted = _recipe_lookup_key(recipe)
    for k, v in threshold.items():
        if _is_threshold_metadata_key(k):
            continue
        if _recipe_lookup_key(k) == wanted:
            value = _float_threshold_value(v)
            if value is not None:
                return value
    for key in ("threshold_by_recipe", "threshold_raw", "threshold"):
        table = threshold.get(key)
        if isinstance(table, dict):
            if recipe in table:
                value = _float_threshold_value(table[recipe])
                if value is not None:
                    return value
            for k, v in table.items():
                if _recipe_lookup_key(k) == wanted:
                    value = _float_threshold_value(v)
                    if value is not None:
                        return value
    for key in ("global_threshold_raw", "global_threshold", "threshold_raw",
                "threshold", "default", "global"):
        value = threshold.get(key)
        value = _float_threshold_value(value)
        if value is not None:
            return value
    return None


def _threshold_score_name(cfg: IOConfig = None) -> str:
    return _score_kind_for_cfg(cfg or IOConfig())


def _json_cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _step_accum_feature_scores(model_a, rec, mode: str) -> dict:
    """Return per-wafer step accumulated reconstruction excess scores.

    score(t,s) = max(0, |actual-expected| / OK기준오차 - 1)
    step score = sum score(t,s) within the ProcessStep.
    """
    level = np.asarray(model_a.error_level(rec), dtype=np.float64)
    if level.ndim != 2 or level.size == 0:
        return {}
    excess = np.maximum(0.0, level - 1.0)
    step = rec.get("step")
    if step is not None and len(step) == excess.shape[0]:
        step_arr = np.asarray(step).astype(int)
    else:
        step_arr = np.ones(excess.shape[0], dtype=int)
    recipe = _recipe_key(rec)
    out = {}
    for st in sorted(int(x) for x in np.unique(step_arr) if int(x) != 0):
        mask = step_arr == st
        if not mask.any():
            continue
        vals = excess[mask].sum(axis=0)
        for j, sensor in enumerate(model_a.sensor_cols):
            score = float(vals[j])
            if mode == "sensor":
                key = (recipe, str(sensor))
                out[key] = max(float(out.get(key, 0.0)), score)
            else:
                out[(recipe, str(sensor), int(st))] = score
    return out


def _sum_all_sensor_reconstruction_score(model_a, rec) -> float:
    """Wafer score: sum normalized reconstruction error over selected sensors/time."""
    level = np.asarray(model_a.error_level(rec), dtype=np.float64)
    valid_steps = None
    if hasattr(model_a, "_score_keep_mask"):
        try:
            valid_steps = model_a._score_keep_mask(rec)
        except Exception:
            valid_steps = None
    return _reconstruction_score_from_level(
        level, getattr(model_a, "cfg", None) or IOConfig(),
        valid_steps=valid_steps)


def _score_records_for_threshold(model_a, records) -> np.ndarray:
    return np.asarray([
        _sum_all_sensor_reconstruction_score(model_a, r)
        for r in list(records or [])
    ], dtype=np.float64)


def _train_ok_sigma_threshold_value(scores, cfg: IOConfig,
                                    sigma: float = None) -> tuple[float, dict]:
    vals = np.asarray(scores, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    k = _threshold_sigma_fixed(cfg) if sigma is None else float(sigma)
    if vals.size == 0:
        th = 1e-12
        mean = std = 0.0
    else:
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        th = float(mean + k * std)
        if not np.isfinite(th) or th <= 0.0:
            th = _no_event_threshold(vals)
    return th, {
        "n": int(vals.size),
        "mean": float(mean),
        "std": float(std),
        "sigma": float(k),
        "threshold_raw": float(th),
    }


def _sigma_candidate_values(crossings) -> np.ndarray:
    vals = np.asarray([] if crossings is None else crossings, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals >= 0.0)]
    base = [0.0]
    if vals.size:
        vals = np.unique(vals)
        vals = np.sort(vals)
        if vals[0] > 0.0:
            base.append(float(vals[0] / 2.0))
        base.extend(float((a + b) / 2.0) for a, b in zip(vals[:-1], vals[1:]) if b > a)
        last = float(vals[-1])
        base.append(last + max(1e-9, abs(last) * 1e-9))
    out, seen = [], set()
    for value in sorted(base):
        if not np.isfinite(value) or value < 0.0:
            continue
        key = round(float(value), 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return np.asarray(out or [0.0], dtype=np.float64)


def _auto_sigma_candidates(records, eval_scores, train_ok_records,
                           train_ok_scores, cfg: IOConfig) -> np.ndarray:
    records = list(records or [])
    eval_scores = np.asarray(eval_scores, dtype=np.float64)
    train_ok_records = list(train_ok_records or [])
    train_ok_scores = np.asarray(train_ok_scores, dtype=np.float64)
    mode = _threshold_wise(cfg)
    crossings = []
    if mode == "all":
        _base, stats = _train_ok_sigma_threshold_value(train_ok_scores, cfg, sigma=0.0)
        mean = float(stats.get("mean", 0.0) or 0.0)
        std = float(stats.get("std", 0.0) or 0.0)
        if np.isfinite(std) and std > 1e-12:
            for rec, score in zip(records, eval_scores):
                if not np.isfinite(score):
                    continue
                factor = _threshold_recipe_multiplier_for(cfg, _recipe_key(rec))
                if not np.isfinite(factor) or factor <= 1e-12:
                    factor = 1.0
                crossings.append((float(score) / float(factor) - mean) / std)
        return _sigma_candidate_values(crossings)

    train_by_recipe = {}
    for rec, score in zip(train_ok_records, train_ok_scores):
        if np.isfinite(score):
            train_by_recipe.setdefault(_recipe_key(rec), []).append(float(score))
    _global_base, global_stats = _train_ok_sigma_threshold_value(
        train_ok_scores, cfg, sigma=0.0)
    for rec, score in zip(records, eval_scores):
        if not np.isfinite(score):
            continue
        vals = train_by_recipe.get(_recipe_key(rec), [])
        if vals:
            _base, stats = _train_ok_sigma_threshold_value(vals, cfg, sigma=0.0)
        else:
            stats = global_stats
        mean = float(stats.get("mean", 0.0) or 0.0)
        std = float(stats.get("std", 0.0) or 0.0)
        if not np.isfinite(std) or std <= 1e-12:
            continue
        factor = _threshold_recipe_multiplier_for(cfg, _recipe_key(rec))
        if not np.isfinite(factor) or factor <= 1e-12:
            factor = 1.0
        crossings.append((float(score) / float(factor) - mean) / std)
    return _sigma_candidate_values(crossings)


def _bounded_sigma_candidates(candidates, cfg: IOConfig) -> np.ndarray:
    vals = np.asarray(candidates, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals >= 0.0)]
    mn, mx = _threshold_sigma_range(cfg)
    extras = []
    if mn is not None:
        extras.append(float(mn))
    if mx is not None:
        extras.append(float(mx))
    if extras:
        vals = np.concatenate([vals, np.asarray(extras, dtype=np.float64)])
    if mn is not None:
        vals = vals[vals >= float(mn)]
    if mx is not None:
        vals = vals[vals <= float(mx)]
    vals = np.asarray(sorted({round(float(v), 12): float(v) for v in vals}.values()),
                      dtype=np.float64)
    if vals.size == 0:
        raise ValueError(
            "auto threshold sigma 후보가 지정 범위 안에 없습니다: "
            f"--threshold-sigma-min/--threshold-sigma-max={_format_threshold_sigma_range(cfg)}"
        )
    return vals


def _metrics_for_saved_threshold(records, y, scores, threshold, cfg: IOConfig) -> dict:
    preds = []
    for rec, score in zip(records, scores):
        cutoff = _saved_threshold_for_record(threshold, rec)
        preds.append(0 if cutoff is None else int(float(score) >= float(cutoff)))
    out = _metrics_from_pred(y, np.asarray(preds, dtype=int), scores)
    out["threshold_wise"] = _threshold_wise(cfg)
    out.update(_score_metadata_for_cfg(cfg))
    return out


def _train_ok_sigma_thresholds_for_sigma(records, y, eval_scores, train_ok_records,
                                         train_ok_scores, cfg: IOConfig,
                                         sigma: float,
                                         sigma_mode: str = "fixed",
                                         requested_sigma: float = None) -> dict | None:
    records = list(records)
    y = np.asarray(y, dtype=int)
    eval_scores = np.asarray(eval_scores, dtype=np.float64)
    train_ok_records = list(train_ok_records or [])
    train_ok_scores = np.asarray(train_ok_scores, dtype=np.float64)
    mode = _threshold_wise(cfg)
    if mode not in {"all", "recipe"}:
        raise ValueError("--threshold-source train-ok-sigma는 --threshold-wise all/recipe에서만 지원합니다.")
    sigma = float(sigma)
    requested_sigma = _threshold_sigma(cfg) if requested_sigma is None else float(requested_sigma)
    base_th, global_stats = _train_ok_sigma_threshold_value(
        train_ok_scores, cfg, sigma=sigma)
    strategy = (
        f"train_ok_final_wafer_score_mean_plus_{sigma:g}_sigma"
    )
    if sigma_mode == "auto":
        strategy += "_auto_selected_from_validation"
    if mode == "all":
        threshold_obj = _threshold_object_from_base_with_recipe_multipliers(
            base_th, cfg, records=records,
            score_kind=_score_kind_for_cfg(cfg),
            strategy=strategy)
        if isinstance(threshold_obj, dict):
            threshold_obj.update({
                "threshold_source": "train-ok-sigma",
                "threshold_sigma": float(sigma),
                "threshold_sigma_mode": str(sigma_mode),
                "threshold_sigma_requested": float(requested_sigma),
                **_threshold_sigma_range_metadata(cfg),
            })
        vc = _metrics_for_saved_threshold(records, y, eval_scores, threshold_obj, cfg)
        vc.update({
            "threshold": threshold_obj,
            "threshold_raw": threshold_obj,
            "threshold_wise": "all",
            "threshold_source": "train-ok-sigma",
            "threshold_sigma": float(sigma),
            "threshold_sigma_mode": str(sigma_mode),
            "threshold_sigma_requested": float(requested_sigma),
            **_threshold_sigma_range_metadata(cfg),
            "threshold_strategy": strategy,
            "train_ok_score_mean": global_stats["mean"],
            "train_ok_score_std": global_stats["std"],
            "train_ok_score_n": global_stats["n"],
            "train_ok_score_stats": {"global": global_stats},
            "global_threshold": round(float(base_th), 4),
            "global_threshold_raw": float(base_th),
        })
        return vc

    train_by_recipe = {}
    for rec, score in zip(train_ok_records, train_ok_scores):
        train_by_recipe.setdefault(_recipe_key(rec), []).append(float(score))
    recipes = []
    for recipe in getattr(cfg, "recipe_names", ()) or ():
        if str(recipe) not in recipes:
            recipes.append(str(recipe))
    for rec in list(records) + train_ok_records:
        recipe = _recipe_key(rec)
        if recipe not in recipes:
            recipes.append(recipe)
    _append_multiplier_recipes(recipes, cfg)
    threshold_by_recipe, base_threshold_by_recipe = {}, {}
    recipe_metrics, train_stats = {}, {}
    for recipe in recipes:
        vals = train_by_recipe.get(recipe, [])
        if vals:
            base, stats = _train_ok_sigma_threshold_value(vals, cfg, sigma=sigma)
            fallback = None
        else:
            base, stats = base_th, dict(global_stats)
            fallback = "no_train_ok_recipe_records_uses_global_train_ok_sigma_threshold"
        factor = _threshold_recipe_multiplier_for(cfg, recipe)
        th = float(base) * factor
        base_threshold_by_recipe[recipe] = float(base)
        threshold_by_recipe[recipe] = th
        train_stats[recipe] = {**stats, "threshold_raw": float(th),
                               "base_threshold_raw": float(base),
                               "threshold_multiplier": float(factor)}
        idx = [i for i, r in enumerate(records) if _recipe_key(r) == recipe]
        if idx:
            yy, pp = y[idx], eval_scores[idx]
            m = _metrics_at_threshold(yy, pp, th)
            m["auroc"] = _safe_auroc(yy, pp)
        else:
            m = {
                "correct": 0, "n": 0, "tp": 0, "tn": 0, "fn": 0, "fp": 0,
                "n_ok": 0, "n_ng": 0, **_rate_metrics(0, 0, 0), "auroc": 0.0,
            }
        recipe_metrics[recipe] = {
            **m,
            "threshold": round(float(th), 4),
            "threshold_raw": float(th),
            "base_threshold": round(float(base), 4),
            "base_threshold_raw": float(base),
            "threshold_multiplier": float(factor),
            "fallback": fallback,
        }
    threshold_obj = {
        "threshold_wise": "recipe",
        "threshold": {k: round(float(v), 4) for k, v in threshold_by_recipe.items()},
        "threshold_raw": dict(threshold_by_recipe),
        "threshold_by_recipe": dict(threshold_by_recipe),
        "base_threshold_by_recipe": dict(base_threshold_by_recipe),
        "global_threshold": round(float(base_th), 4),
        "global_threshold_raw": float(base_th),
        "threshold_source": "train-ok-sigma",
        "threshold_sigma": float(sigma),
        "threshold_sigma_mode": str(sigma_mode),
        "threshold_sigma_requested": float(requested_sigma),
        **_threshold_sigma_range_metadata(cfg),
        "threshold_strategy": strategy,
        "threshold_recipe_multipliers": _display_threshold_recipe_multipliers(cfg),
        "train_ok_score_stats": train_stats,
        **_score_metadata_for_cfg(cfg),
    }
    vc = _metrics_for_saved_threshold(records, y, eval_scores, threshold_obj, cfg)
    vc.update({
        "threshold": threshold_obj["threshold"],
        "threshold_raw": dict(threshold_by_recipe),
        "threshold_by_recipe": dict(threshold_by_recipe),
        "base_threshold_by_recipe": dict(base_threshold_by_recipe),
        "global_threshold": round(float(base_th), 4),
        "global_threshold_raw": float(base_th),
        "recipe_metrics": recipe_metrics,
        "threshold_source": "train-ok-sigma",
        "threshold_sigma": float(sigma),
        "threshold_sigma_mode": str(sigma_mode),
        "threshold_sigma_requested": float(requested_sigma),
        **_threshold_sigma_range_metadata(cfg),
        "threshold_strategy": strategy,
        "threshold_recipe_multipliers": _display_threshold_recipe_multipliers(cfg),
        "train_ok_score_mean": global_stats["mean"],
        "train_ok_score_std": global_stats["std"],
        "train_ok_score_n": global_stats["n"],
        "train_ok_score_stats": train_stats,
    })
    return vc


def _train_ok_sigma_thresholds(records, y, eval_scores, train_ok_records,
                               train_ok_scores, cfg: IOConfig) -> dict | None:
    mode = _threshold_wise(cfg)
    if mode not in {"all", "recipe"}:
        raise ValueError("--threshold-source train-ok-sigma는 --threshold-wise all/recipe에서만 지원합니다.")
    requested = _threshold_sigma(cfg)
    if not _threshold_sigma_auto(cfg):
        return _train_ok_sigma_thresholds_for_sigma(
            records, y, eval_scores, train_ok_records, train_ok_scores, cfg,
            sigma=_threshold_sigma_fixed(cfg),
            sigma_mode="fixed",
            requested_sigma=requested)

    best = None
    candidates = _bounded_sigma_candidates(
        _auto_sigma_candidates(
            records, eval_scores, train_ok_records, train_ok_scores, cfg),
        cfg)
    for sigma in candidates:
        vc = _train_ok_sigma_thresholds_for_sigma(
            records, y, eval_scores, train_ok_records, train_ok_scores, cfg,
            sigma=float(sigma),
            sigma_mode="auto",
            requested_sigma=requested)
        cand = (_threshold_candidate_rank(vc, float(sigma), cfg), float(sigma), vc)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return _train_ok_sigma_thresholds_for_sigma(
            records, y, eval_scores, train_ok_records, train_ok_scores, cfg,
            sigma=0.0,
            sigma_mode="auto",
            requested_sigma=requested)
    _rank, selected_sigma, vc = best
    vc["threshold_sigma"] = float(selected_sigma)
    vc["threshold_sigma_mode"] = "auto"
    vc["threshold_sigma_requested"] = float(requested)
    vc.update(_threshold_sigma_range_metadata(cfg))
    vc["threshold_sigma_candidates"] = int(len(candidates))
    if isinstance(vc.get("threshold_raw"), dict):
        vc["threshold_raw"]["threshold_sigma"] = float(selected_sigma)
        vc["threshold_raw"]["threshold_sigma_mode"] = "auto"
        vc["threshold_raw"]["threshold_sigma_requested"] = float(requested)
        vc["threshold_raw"].update(_threshold_sigma_range_metadata(cfg))
    if isinstance(vc.get("threshold"), dict):
        vc["threshold"]["threshold_sigma"] = float(selected_sigma)
        vc["threshold"]["threshold_sigma_mode"] = "auto"
        vc["threshold"]["threshold_sigma_requested"] = float(requested)
        vc["threshold"].update(_threshold_sigma_range_metadata(cfg))
    return vc


def _group_threshold_candidates(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float64)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    cands = _threshold_candidates_midpoint(vals)
    eps = max(1e-12, float(np.nanmax(np.abs(vals))) * 1e-12)
    cands = np.asarray([float(x) for x in cands if float(x) > eps], dtype=np.float64)
    return cands if cands.size else np.asarray([_no_event_threshold(vals)], dtype=np.float64)


def _no_event_threshold(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1e-12
    vmax = float(np.max(vals))
    return float(vmax + max(1e-12, abs(vmax) * 1e-12))


def _threshold_table_from_groups(th_by_group: dict, mode: str,
                                 default_threshold: float) -> dict:
    table = {}
    if mode == "sensor":
        for (recipe, sensor), th in sorted(th_by_group.items(), key=lambda kv: kv[0]):
            table.setdefault(str(recipe), {})[str(sensor)] = float(th)
    else:
        for (recipe, sensor, step), th in sorted(th_by_group.items(), key=lambda kv: kv[0]):
            table.setdefault(str(recipe), {}).setdefault(str(sensor), {})[str(int(step))] = float(th)
    return {
        "threshold_wise": mode,
        "score_kind": "step_accum_excess_reconstruction_error",
        "formula": "sum_by_step(max(0, |actual-expected| / OK_error_limit - 1))",
        "thresholds": table,
        "default_threshold": float(default_threshold),
        "decision_threshold": 1.0,
        "decision_rule": "NG if any sensor/step accumulated score exceeds its threshold",
    }


def _decision_scores_from_group_thresholds(group_rows: list[dict],
                                           th_by_group: dict,
                                           default_threshold: float) -> np.ndarray:
    scores = []
    eps = 1e-12
    for row in group_rows:
        mx = 0.0
        for key, val in row.items():
            th = float(th_by_group.get(key, default_threshold))
            th = max(th, eps)
            mx = max(mx, float(val) / th)
        scores.append(mx)
    return np.asarray(scores, dtype=np.float64)


def _maximize_group_thresholds_without_changing_predictions(
        group_rows: list[dict], th_by_group: dict,
        default_threshold: float) -> dict:
    """Raise each selected threshold as high as possible while keeping predictions."""
    out = dict(th_by_group)
    final_scores = _decision_scores_from_group_thresholds(group_rows, out, default_threshold)
    final_pred = final_scores >= 1.0
    for key in sorted(out):
        vals = np.asarray([float(row.get(key, 0.0)) for row in group_rows], dtype=np.float64)
        current = float(out[key])
        cands = [float(x) for x in _group_threshold_candidates(vals)
                 if current <= float(x) <= float(default_threshold)]
        cands.append(current)
        cands = sorted(set(cands), reverse=True)
        for th in cands:
            trial = dict(out)
            trial[key] = float(th)
            pred = _decision_scores_from_group_thresholds(
                group_rows, trial, default_threshold) >= 1.0
            if np.array_equal(pred, final_pred):
                out[key] = float(th)
                break
    return out


def _metrics_from_pred(y, pred, score_for_auc=None):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    n_ok = int((y == 0).sum())
    n_ng = int((y == 1).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "correct": tp + tn, "n": n_ok + n_ng, "tp": tp, "tn": tn,
        "fn": fn, "fp": fp, "n_ok": n_ok, "n_ng": n_ng,
        **_rate_metrics(tp, fp, fn),
        "auroc": _safe_auroc(y, score_for_auc if score_for_auc is not None else pred),
    }


def _correct_from_step_accum_thresholds(records, y, model_a, cfg: IOConfig, mode: str):
    records = list(records)
    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return None, None
    group_rows = [_step_accum_feature_scores(model_a, r, mode) for r in records]
    groups = sorted({key for row in group_rows for key in row.keys()})
    if not groups:
        return None, None
    vals_by_group = {
        key: np.asarray([float(row.get(key, 0.0)) for row in group_rows], dtype=np.float64)
        for key in groups
    }
    default_threshold = max(_no_event_threshold(v) for v in vals_by_group.values())
    th_by_group = {}
    for key, vals in vals_by_group.items():
        vmax = float(np.nanmax(vals)) if len(vals) else 0.0
        th_by_group[key] = default_threshold if vmax <= 1e-12 else _no_event_threshold(vals)
    pred = np.zeros(len(records), dtype=int)
    current = _metrics_from_pred(y, pred, pred)
    current_obj = _metrics_rank(current, cfg)

    candidates = {
        key: [float(x) for x in _group_threshold_candidates(vals)]
        for key, vals in vals_by_group.items()
    }
    max_iter = max(1, len(groups) * 2)
    for _ in range(max_iter):
        best = None
        for key in groups:
            vals = vals_by_group[key]
            cur_th = float(th_by_group[key])
            for th in candidates[key]:
                if th >= cur_th - 1e-15:
                    continue
                cand_pred = np.logical_or(pred, vals >= float(th)).astype(int)
                m = _metrics_from_pred(y, cand_pred, cand_pred)
                obj = _metrics_rank(m, cfg)
                if obj <= current_obj:
                    continue
                cand_rank = (obj, float(th))
                if best is None or cand_rank > best[0]:
                    best = (cand_rank, obj, key, float(th), cand_pred, m)
        if best is None:
            break
        _, current_obj, key, th, pred, current = best
        th_by_group[key] = th

    th_by_group = _maximize_group_thresholds_without_changing_predictions(
        group_rows, th_by_group, default_threshold)
    decision_scores = _decision_scores_from_group_thresholds(
        group_rows, th_by_group, default_threshold)
    pred = (decision_scores >= 1.0).astype(int)
    metrics = _metrics_from_pred(y, pred, decision_scores)
    threshold_obj = _threshold_table_from_groups(th_by_group, mode, default_threshold)
    metrics.update({
        "threshold": threshold_obj,
        "threshold_raw": threshold_obj,
        "threshold_wise": mode,
        "threshold_strategy": f"greedy_or_{mode}_step_accum_excess_reconstruction_error",
        "global_threshold": 1.0,
        "global_threshold_raw": 1.0,
        "n_threshold_groups": int(len(groups)),
    })
    return metrics, decision_scores


def _evaluate_thresholds(records, y, scalar_scores, model_a, cfg: IOConfig,
                         train_ok_records=None):
    mode = _threshold_wise(cfg)
    if _threshold_source(cfg) == "train-ok-sigma":
        step_scores = np.asarray([_sum_all_sensor_reconstruction_score(model_a, r) for r in records],
                                 dtype=np.float64)
        train_scores = _score_records_for_threshold(model_a, train_ok_records or [])
        vc = _train_ok_sigma_thresholds(
            records, y, step_scores, train_ok_records or [], train_scores, cfg)
        return vc, step_scores
    if mode == "all":
        step_scores = np.asarray([_sum_all_sensor_reconstruction_score(model_a, r) for r in records],
                                 dtype=np.float64)
        vc = _correct_from_nonnegative_scores_all(records, y, step_scores, cfg)
        if vc is not None:
            vc["threshold_wise"] = "all"
            vc["threshold_source"] = _threshold_source(cfg)
            vc.update(_score_metadata_for_cfg(cfg))
            if not str(vc.get("threshold_strategy", "")).strip():
                vc["threshold_strategy"] = (
                    "global_midpoint_between_adjacent_sum_all_sensor_reconstruction_scores"
                )
        return vc, step_scores
    if mode == "recipe":
        step_scores = np.asarray([_sum_all_sensor_reconstruction_score(model_a, r) for r in records],
                                 dtype=np.float64)
        vc = _correct_from_nonnegative_scores_by_recipe(records, y, step_scores, cfg)
        if vc is not None:
            vc["threshold_wise"] = "recipe"
            vc["threshold_source"] = _threshold_source(cfg)
            vc.update(_score_metadata_for_cfg(cfg))
        return vc, step_scores
    vc, decision_scores = _correct_from_step_accum_thresholds(records, y, model_a, cfg, mode)
    if vc is not None:
        vc["threshold_source"] = _threshold_source(cfg)
    return vc, decision_scores


def _tcn_search_space():
    return _load_tcn_search_space()


def _tcn_search_space_for_run(fixed_epochs: int = None) -> dict:
    space = _tcn_search_space()
    if fixed_epochs is not None:
        space["tcn_epochs"] = [int(fixed_epochs)]
    return space


def _receptive(kernel, blocks):
    return 2 * (kernel - 1) * (2 ** blocks - 1) + 1


def _tcn_sample_params(trials: int, seed: int, fixed_epochs: int = None):
    import random
    space = _tcn_search_space_for_run(fixed_epochs)
    rng = random.Random(seed)
    params_all, seen = [], set()
    for _ in range(int(trials)):
        params = None
        for _try in range(200):
            cand = {k: rng.choice(v) for k, v in space.items()}
            if cand["tcn_stride"] and cand["tcn_stride"] > cand["tcn_future_window"]:
                cand["tcn_stride"] = cand["tcn_future_window"]
            if _receptive(cand["tcn_kernel"], cand["tcn_blocks"]) < \
                    cand["tcn_past_window"] + cand["tcn_future_window"]:
                continue
            key = tuple(sorted(cand.items()))
            if key not in seen:
                seen.add(key)
                params = cand
                break
        if params is not None:
            params_all.append(params)
    return params_all


def _load_optuna_for_tcn_search():
    try:
        import optuna
    except Exception as e:
        sys.exit("Optuna Bayesian search를 사용하려면 optuna가 필요합니다. "
                 f"현재 import 실패: {e}\n"
                 "설치 예: pip install optuna")
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass
    return optuna


def _make_tcn_optuna_study(seed: int):
    import warnings
    optuna = _load_optuna_for_tcn_search()
    sampler_kwargs = [
        {"seed": int(seed), "multivariate": True, "group": True},
        {"seed": int(seed), "multivariate": True},
        {"seed": int(seed)},
    ]
    last_error = None
    for kwargs in sampler_kwargs:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
                sampler = optuna.samplers.TPESampler(**kwargs)
            return optuna.create_study(direction="maximize", sampler=sampler)
        except TypeError as e:
            last_error = e
            continue
    raise last_error


def _ordered_tcn_space_keys(space: dict) -> list[str]:
    ordered = [k for k in REQUIRED_TCN_SEARCH_SPACE_KEYS if k in space]
    ordered.extend(k for k in space if k not in ordered)
    return ordered


def _tcn_param_key(params: dict) -> tuple:
    return tuple((str(k), json.dumps(v, sort_keys=True, ensure_ascii=False))
                 for k, v in sorted((params or {}).items()))


def _tcn_params_valid(params: dict) -> bool:
    try:
        stride = int(params.get("tcn_stride") or 0)
        future = int(params["tcn_future_window"])
        if stride and stride > future:
            return False
        return _receptive(int(params["tcn_kernel"]), int(params["tcn_blocks"])) >= \
            int(params["tcn_past_window"]) + int(params["tcn_future_window"])
    except Exception:
        return False


def _suggest_tcn_params_optuna(trial, space: dict) -> dict:
    params = {}
    for key in _ordered_tcn_space_keys(space):
        params[key] = trial.suggest_categorical(key, list(space[key]))
    return params


def _optuna_objective_value(result: dict, cfg: IOConfig = None) -> float | None:
    if not isinstance(result, dict) or result.get("error"):
        return None
    try:
        obj = list(result.get("obj") or _result_obj(result, cfg))
    except Exception:
        return None
    if not obj:
        return None
    weights = [1e12, 1e9, 1e6, 1e3, 1.0, 1e-3, 1e-6]
    value = 0.0
    for idx, item in enumerate(obj):
        try:
            part = float(item)
        except Exception:
            part = -1e12
        if not np.isfinite(part):
            part = -1e12 if part < 0 else 0.0
        weight = weights[idx] if idx < len(weights) else 10.0 ** max(0, 6 - idx)
        value += part * weight
    return float(value)


def _coerce_search_space_value(raw, choices: list):
    if raw in (None, ""):
        return None
    parsed = _parse_progress_value(raw)
    for choice in choices:
        if isinstance(choice, bool):
            if isinstance(parsed, bool) and parsed == choice:
                return choice
            if str(parsed).strip().lower() == str(choice).lower():
                return choice
        elif isinstance(choice, int) and not isinstance(choice, bool):
            try:
                if int(parsed) == int(choice):
                    return choice
            except Exception:
                pass
        elif isinstance(choice, float):
            try:
                if abs(float(parsed) - float(choice)) <= 1e-12:
                    return choice
            except Exception:
                pass
        elif str(parsed) == str(choice):
            return choice
    return None


def _read_optuna_resume_history(path: Path, space: dict, cfg: IOConfig = None):
    path = Path(path)
    if not path.exists():
        return [], set(), 0
    history, seen = [], set()
    max_trial_idx = 0
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                status = str(row.get("status", "") or "").strip().lower()
                if status and status != "ok":
                    continue
                params = {}
                valid = True
                for key, choices in space.items():
                    value = _coerce_search_space_value(row.get(f"param_{key}"), choices)
                    if value is None:
                        valid = False
                        break
                    params[key] = value
                if not valid or not _tcn_params_valid(params):
                    continue
                key = _tcn_param_key(params)
                if key in seen:
                    continue
                try:
                    trial_idx = int(float(row.get("trial_idx", len(history) + 1) or len(history) + 1))
                except Exception:
                    trial_idx = len(history) + 1
                max_trial_idx = max(max_trial_idx, trial_idx)
                val = {
                    "correct": int(float(row.get("trial_tp_plus_tf", 0) or 0)),
                    "tn": int(float(row.get("trial_tn", row.get("trial_tf", 0)) or 0)),
                    "tp": int(float(row.get("trial_tp", 0) or 0)),
                    "fp": int(float(row.get("trial_fp", 0) or 0)),
                    "fn": int(float(row.get("trial_fn", 0) or 0)),
                    "n_ok": int(float(row.get("trial_n_ok", 0) or 0)),
                    "n_ng": int(float(row.get("trial_n_ng", 0) or 0)),
                    "n": int(float(row.get("trial_n_wafers", 0) or 0)),
                    "auroc": float(row.get("trial_auroc", 0.0) or 0.0),
                }
                val = _ensure_rate_metrics(val)
                try:
                    mse = float(row.get("forecast_val_mse", float("inf")) or float("inf"))
                except Exception:
                    mse = float("inf")
                fake_result = {"params": params, "val": val, "forecast_val_mse": mse}
                value = _optuna_objective_value(fake_result, cfg)
                if value is None or not np.isfinite(value):
                    continue
                history.append({"trial_idx": trial_idx, "params": params, "value": value})
                seen.add(key)
    except Exception as e:
        print(f"  [resume] Optuna history 읽기 실패: {e}")
        return [], set(), 0
    return history, seen, max_trial_idx


def _optuna_distributions_for_space(optuna, space: dict) -> dict:
    return {
        key: optuna.distributions.CategoricalDistribution(list(values))
        for key, values in space.items()
    }


class _TcnOptunaScheduler:
    def __init__(self, trials: int, seed: int, fixed_epochs: int,
                 progress_csv: Path, resume: bool, cfg: IOConfig = None):
        self.optuna = _load_optuna_for_tcn_search()
        self.study = _make_tcn_optuna_study(seed)
        self.space = _tcn_search_space_for_run(fixed_epochs)
        self.cfg = cfg
        self.target_total = max(0, int(trials))
        self.completed_history = 0
        self.to_schedule = self.target_total
        self.scheduled = 0
        self.reported = 0
        self.running = {}
        self.seen = set()
        self.exhausted = False
        self.next_trial_idx = 1
        if resume:
            history, seen, max_idx = _read_optuna_resume_history(progress_csv, self.space, cfg)
            distributions = _optuna_distributions_for_space(self.optuna, self.space)
            loaded = 0
            for item in history:
                try:
                    trial = self.optuna.trial.create_trial(
                        params=item["params"],
                        distributions=distributions,
                        value=float(item["value"]),
                        state=self.optuna.trial.TrialState.COMPLETE,
                    )
                    self.study.add_trial(trial)
                    loaded += 1
                except Exception:
                    continue
            self.seen.update(seen)
            self.completed_history = loaded
            self.to_schedule = max(0, self.target_total - self.completed_history)
            self.next_trial_idx = max(max_idx + 1, self.completed_history + 1)

    @property
    def remaining_to_schedule(self) -> int:
        if self.exhausted:
            return 0
        return max(0, self.to_schedule - self.scheduled)

    def has_pending_or_schedulable(self) -> bool:
        return bool(self.running) or self.remaining_to_schedule > 0

    def ask(self) -> tuple[int, dict] | None:
        if self.remaining_to_schedule <= 0:
            return None
        for _ in range(2000):
            trial = self.study.ask()
            params = _suggest_tcn_params_optuna(trial, self.space)
            if not _tcn_params_valid(params):
                self.study.tell(trial, state=self.optuna.trial.TrialState.PRUNED)
                continue
            key = _tcn_param_key(params)
            if key in self.seen:
                self.study.tell(trial, state=self.optuna.trial.TrialState.PRUNED)
                continue
            self.seen.add(key)
            trial_idx = int(self.next_trial_idx)
            self.next_trial_idx += 1
            self.scheduled += 1
            self.running[trial_idx] = trial
            return trial_idx, params
        self.exhausted = True
        print("  [optuna] 더 이상 유효한 신규 parameter 조합을 찾지 못했습니다.")
        return None

    def tell(self, result: dict | None, trial_idx: int = None) -> None:
        if isinstance(result, dict):
            result.setdefault("search_sampler", "optuna_tpe")
            trial_idx = result.get("trial_idx", trial_idx)
        try:
            trial_idx = int(trial_idx)
        except Exception:
            trial_idx = None
        trial = self.running.pop(trial_idx, None) if trial_idx is not None else None
        if trial is None:
            return
        self.reported += 1
        value = _optuna_objective_value(result, self.cfg)
        try:
            if value is None:
                self.study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
            else:
                self.study.tell(trial, float(value))
        except Exception as e:
            print(f"  [optuna] trial 결과 기록 실패: {e}")


def _search_result_with_metadata(r, score_name: str, model_kind: str) -> dict:
    out = dict(r)
    val = out.get("val", {})
    scores = out.get("scores", {})
    out["score_name"] = score_name
    out["model_kind"] = model_kind
    out["anomaly_scores"] = scores
    objective = str(out.get("search_objective") or "correct")
    out["evaluation_criteria"] = {
        "search_objective": objective,
        "primary_metric": "precision" if objective == "precision" else "correct_wafers",
        "primary_metric_value": val.get("precision" if objective == "precision" else "correct"),
        "secondary_metric": "correct_wafers" if objective == "precision" else "ok_correct_wafers",
        "secondary_metric_value": val.get("correct" if objective == "precision" else "tn"),
        "tertiary_metric": "true_positive_wafers",
        "tertiary_metric_value": val.get("tp"),
        "trial_idx": out.get("trial_idx"),
        "threshold": out.get("threshold", val.get("threshold_raw", val.get("threshold"))),
        "threshold_field": score_name,
        "score_direction": "higher_is_more_anomalous",
        "score_agg_enabled": val.get("score_agg_enabled"),
        "score_resid_agg": val.get("score_resid_agg"),
        "score_wafer_agg": val.get("score_wafer_agg"),
        "score_time_avg": val.get("score_time_avg"),
        "score_final_avg": val.get("score_final_avg"),
        "score_topk": val.get("score_topk"),
        "threshold_source": val.get("threshold_source"),
        "threshold_sigma": val.get("threshold_sigma"),
        "threshold_sigma_mode": val.get("threshold_sigma_mode"),
        "threshold_sigma_requested": val.get("threshold_sigma_requested"),
        "threshold_sigma_min": val.get("threshold_sigma_min"),
        "threshold_sigma_max": val.get("threshold_sigma_max"),
        "threshold_sigma_candidates": val.get("threshold_sigma_candidates"),
        "trained_epochs": out.get("trained_epochs"),
        "post_eval_extend_epochs": out.get("post_eval_extend_epochs"),
        "post_eval_max_extends": out.get("post_eval_max_extends"),
        "post_eval_extends_attempted": out.get("post_eval_extends_attempted"),
        "post_eval_extends_accepted": out.get("post_eval_extends_accepted"),
        "n_wafers": val.get("n"),
        "n_ok": val.get("n_ok"),
        "n_ng": val.get("n_ng"),
        "tp": val.get("tp"),
        "tn": val.get("tn"),
        "fp": val.get("fp"),
        "fn": val.get("fn"),
        "precision": val.get("precision"),
        "recall": val.get("recall"),
        "f1_score": val.get("f1_score"),
        "auroc": val.get("auroc"),
    }
    return out


def _write_search_jsons(adir: Path, filenames, best: dict,
                        score_name: str, model_kind: str) -> None:
    adir = Path(adir)
    adir.mkdir(parents=True, exist_ok=True)
    payload = {"best": _search_result_with_metadata(best, score_name, model_kind)}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for name in filenames:
        (adir / name).write_text(text, encoding="utf-8")


def _append_best_progress_csv(path: Path, best: dict, event_idx: int,
                              score_name: str, model_kind: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    val = best.get("val", {})
    row = {
        "trial_idx": best.get("trial_idx"),
        "best_trial_idx": best.get("trial_idx"),
        "best_event": event_idx,
        "model_kind": model_kind,
        "score_name": score_name,
        "search_objective": best.get("search_objective", ""),
        "forecast_val_mse": best.get("forecast_val_mse"),
        "correct_wafers": val.get("correct"),
        "n_wafers": val.get("n"),
        "tp": val.get("tp"),
        "tn": val.get("tn"),
        "fp": val.get("fp"),
        "fn": val.get("fn"),
        "n_ok": val.get("n_ok"),
        "n_ng": val.get("n_ng"),
        "precision": val.get("precision"),
        "recall": val.get("recall"),
        "f1_score": val.get("f1_score"),
        "auroc": val.get("auroc"),
        "n_params": best.get("n_params", ""),
        "trained_epochs": best.get("trained_epochs", ""),
        "post_eval_extend_epochs": best.get("post_eval_extend_epochs", ""),
        "post_eval_extends_attempted": best.get("post_eval_extends_attempted", ""),
        "post_eval_extends_accepted": best.get("post_eval_extends_accepted", ""),
        "threshold": _json_cell(best.get("threshold", val.get("threshold_raw", val.get("threshold")))),
        "threshold_sigma": val.get("threshold_sigma", ""),
        "threshold_sigma_mode": val.get("threshold_sigma_mode", ""),
        "threshold_sigma_min": val.get("threshold_sigma_min", ""),
        "threshold_sigma_max": val.get("threshold_sigma_max", ""),
        "threshold_sigma_candidates": val.get("threshold_sigma_candidates", ""),
    }
    for k, v in best.get("params", {}).items():
        row[f"param_{k}"] = v
    rows, fieldnames = [], list(row.keys())
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for k in reader.fieldnames or []:
                if k not in fieldnames:
                    fieldnames.append(k)
    rows.append({k: row.get(k, "") for k in fieldnames})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_trial_progress_csv(path: Path, r: dict, cur_best: dict,
                               completed_idx: int, total: int, run_id: str,
                               score_name: str, model_kind: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = r or {}
    val = r.get("val", {}) if isinstance(r, dict) else {}
    best_val = cur_best.get("val", {}) if isinstance(cur_best, dict) else {}
    err = r.get("error") if isinstance(r, dict) else "no_result"
    status = "error" if err else ("no_score" if not val else "ok")
    row = {
        "run_id": run_id,
        "completed_idx": completed_idx,
        "total_trials": total,
        "trial_idx": r.get("trial_idx", completed_idx),
        "status": status,
        "error": err or "",
        "model_kind": model_kind,
        "score_name": score_name,
        "search_sampler": r.get("search_sampler", ""),
        "search_objective": r.get("search_objective", ""),
        "forecast_val_mse": r.get("forecast_val_mse", ""),
        "trained_epochs": r.get("trained_epochs", ""),
        "post_eval_extend_epochs": r.get("post_eval_extend_epochs", ""),
        "post_eval_extends_attempted": r.get("post_eval_extends_attempted", ""),
        "post_eval_extends_accepted": r.get("post_eval_extends_accepted", ""),
        "trial_tp_plus_tf": val.get("correct", ""),
        "trial_precision": val.get("precision", ""),
        "trial_recall": val.get("recall", ""),
        "trial_f1_score": val.get("f1_score", ""),
        "trial_tp": val.get("tp", ""),
        "trial_tf": val.get("tn", ""),
        "trial_tn": val.get("tn", ""),
        "trial_fp": val.get("fp", ""),
        "trial_fn": val.get("fn", ""),
        "trial_n_ok": val.get("n_ok", ""),
        "trial_n_ng": val.get("n_ng", ""),
        "trial_n_wafers": val.get("n", ""),
        "trial_auroc": val.get("auroc", ""),
        "trial_threshold": _json_cell(r.get("threshold", val.get("threshold_raw", val.get("threshold", "")))),
        "trial_threshold_sigma": val.get("threshold_sigma", ""),
        "trial_threshold_sigma_mode": val.get("threshold_sigma_mode", ""),
        "trial_threshold_sigma_min": val.get("threshold_sigma_min", ""),
        "trial_threshold_sigma_max": val.get("threshold_sigma_max", ""),
        "trial_threshold_sigma_candidates": val.get("threshold_sigma_candidates", ""),
        "best_trial_idx": cur_best.get("trial_idx", "") if isinstance(cur_best, dict) else "",
        "best_tp_plus_tf": best_val.get("correct", ""),
        "best_precision": best_val.get("precision", ""),
        "best_recall": best_val.get("recall", ""),
        "best_f1_score": best_val.get("f1_score", ""),
        "best_tp": best_val.get("tp", ""),
        "best_tf": best_val.get("tn", ""),
        "best_tn": best_val.get("tn", ""),
        "best_fp": best_val.get("fp", ""),
        "best_fn": best_val.get("fn", ""),
        "best_threshold_sigma": best_val.get("threshold_sigma", ""),
        "best_threshold_sigma_mode": best_val.get("threshold_sigma_mode", ""),
        "best_threshold_sigma_min": best_val.get("threshold_sigma_min", ""),
        "best_threshold_sigma_max": best_val.get("threshold_sigma_max", ""),
        "best_threshold_sigma_candidates": best_val.get("threshold_sigma_candidates", ""),
    }
    for k, v in (r.get("params", {}) if isinstance(r, dict) else {}).items():
        row[f"param_{k}"] = v
    rows, fieldnames = [], list(row.keys())
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for k in reader.fieldnames or []:
                if k not in fieldnames:
                    fieldnames.append(k)
    rows.append({k: row.get(k, "") for k in fieldnames})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_progress_value(value):
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        if all(ch not in text.lower() for ch in (".", "e")):
            return int(text)
        return float(text)
    except Exception:
        return text


def _row_params_match(row: dict, params: dict) -> bool:
    for key, expected in params.items():
        raw = row.get(f"param_{key}")
        if raw in (None, ""):
            return False
        got = _parse_progress_value(raw)
        if isinstance(expected, bool):
            if bool(got) != expected:
                return False
        elif isinstance(expected, int) and not isinstance(expected, bool):
            try:
                if int(got) != int(expected):
                    return False
            except Exception:
                return False
        elif isinstance(expected, float):
            try:
                if abs(float(got) - float(expected)) > 1e-12:
                    return False
            except Exception:
                return False
        elif str(got) != str(expected):
            return False
    return True


def _read_completed_trials(path: Path, params_all: list[dict]) -> set[int]:
    path = Path(path)
    if not path.exists():
        return set()
    completed = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    trial_idx = int(float(row.get("trial_idx", "")))
                except Exception:
                    continue
                if 1 <= trial_idx <= len(params_all) and _row_params_match(row, params_all[trial_idx - 1]):
                    completed.add(trial_idx)
    except Exception as e:
        print(f"  [resume] trial progress 읽기 실패: {e}")
    return completed


def _result_obj(r: dict, cfg: IOConfig = None) -> list:
    val = r.get("val", {}) if isinstance(r, dict) else {}
    mse = r.get("forecast_val_mse", float("inf")) if isinstance(r, dict) else float("inf")
    return _result_obj_from_val(val, mse, cfg)


def _copy_array_or_none(value):
    if value is None:
        return None
    return np.asarray(value).copy()


def _snapshot_tcn_model(model_a) -> dict:
    return {
        "state": {
            k: v.detach().cpu().clone()
            for k, v in model_a.net.state_dict().items()
        },
        "val_mse": getattr(model_a, "val_mse", None),
        "trained_epochs": getattr(model_a, "trained_epochs", None),
        "resid_abs_limit": _copy_array_or_none(getattr(model_a, "resid_abs_limit", None)),
        "resid_std": _copy_array_or_none(getattr(model_a, "resid_std", None)),
        "resid_sigma": _copy_array_or_none(getattr(model_a, "resid_sigma", None)),
        "per_step_limit": bool(getattr(model_a, "_per_step_limit", False)),
        "per_step_sigma": bool(getattr(model_a, "_per_step_sigma", False)),
    }


def _restore_tcn_model(model_a, snapshot: dict) -> None:
    if not snapshot:
        return
    model_a.net.load_state_dict(snapshot["state"])
    model_a.val_mse = snapshot.get("val_mse")
    model_a.trained_epochs = snapshot.get("trained_epochs")
    model_a.resid_abs_limit = _copy_array_or_none(snapshot.get("resid_abs_limit"))
    model_a.resid_std = _copy_array_or_none(snapshot.get("resid_std"))
    model_a.resid_sigma = _copy_array_or_none(snapshot.get("resid_sigma"))
    model_a._per_step_limit = bool(snapshot.get("per_step_limit", False))
    model_a._per_step_sigma = bool(snapshot.get("per_step_sigma", False))
    model_a.net.to(model_a.device)


def _score_tcn_candidate(h, cfg: IOConfig, params: dict, S: dict,
                         eval_recs: list, y: np.ndarray) -> dict | None:
    val_mse = float(getattr(h.A, "val_mse", float("nan")))
    vc, decision_scores = _evaluate_thresholds(
        eval_recs, y, None, h.A, cfg, train_ok_records=S["train_ok"])
    if vc is None:
        return None
    ids = [r["wafer_id"] for r in eval_recs]
    scores = {w: float(pp) for w, pp in zip(ids, decision_scores)}
    n_params = int(sum(p.numel() for p in h.A.net.parameters())) if h.A.net else 0
    return {"params": params,
            "obj": _result_obj_from_val(vc, val_mse, cfg),
            "search_objective": _search_objective(cfg),
            "forecast_val_mse": val_mse,
            "trained_epochs": getattr(h.A, "trained_epochs", ""),
            "n_params": n_params,
            "val": vc,
            "receptive": _receptive(params["tcn_kernel"], params["tcn_blocks"]),
            "need": int(params["tcn_past_window"] + params["tcn_future_window"]),
            "scores": scores,
            "model_arch": getattr(h.A, "arch", {}),
            "input_sensor_cols": list(getattr(h.A, "input_sensor_cols", [])),
            "target_sensor_cols": list(getattr(h.A, "sensor_cols", [])),
            "threshold": vc.get("threshold_raw", vc.get("threshold"))}


def _read_best_event_count(path: Path) -> int:
    if not Path(path).exists():
        return 0
    best_event = 0
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    best_event = max(best_event, int(float(row.get("best_event", 0) or 0)))
                except Exception:
                    pass
    except Exception:
        return 0
    return best_event


def _load_resume_best(adir: Path, params_all: list[dict] | None, model_kind: str,
                      cfg: IOConfig = None) -> dict | None:
    for path in (Path(adir) / "search_best.json", Path(adir) / "best_search.json"):
        if not path.exists():
            continue
        try:
            best = json.loads(path.read_text(encoding="utf-8")).get("best", {})
        except Exception as e:
            print(f"  [resume] {path.name} 읽기 실패: {e}")
            continue
        if best.get("model_kind") != model_kind:
            continue
        params = best.get("params") or {}
        if params_all is not None and not any(p == params for p in params_all):
            print(f"  [resume] {path.name}의 best params가 현재 search space와 달라 무시")
            continue
        if isinstance(best.get("val"), dict):
            best["val"] = _ensure_rate_metrics(best["val"])
        best["search_objective"] = _search_objective(cfg)
        best["obj"] = _result_obj(best, cfg)
        return best
    return None


def _checkpoint_ready(adir: Path) -> bool:
    return (Path(adir) / "tcn_forecast.pt").exists()


def _plot_best_progress(csv_path: Path, out_path: Path) -> None:
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    if not csv_path.exists():
        print(f"  [plot] best progress CSV 없음: {csv_path}")
        return
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("correct_wafers") not in ("", None):
                rows.append(row)
    if not rows:
        print(f"  [plot] 그릴 best progress row 없음: {csv_path}")
        return
    def _num(row, key):
        try:
            return float(row.get(key, ""))
        except Exception:
            return np.nan
    rows.sort(key=lambda row: _num(row, "best_event"))
    xs = [_num(r, "trial_idx") for r in rows]
    total = [_num(r, "correct_wafers") for r in rows]
    tp = [_num(r, "tp") for r in rows]
    tf = [_num(r, "tn") for r in rows]
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(xs, total, marker="o", lw=2.0, label="TP+TF")
    ax.plot(xs, tp, marker="o", lw=1.8, label="TP")
    ax.plot(xs, tf, marker="o", lw=1.8, label="TF")
    ax.set_title("TCN Best Search Progress")
    ax.set_xlabel("Trial index where best was updated")
    ax.set_ylabel("Correct wafer count")
    ax.set_xlim(left=1, right=max(max(xs), 2))
    ymax = max([v for v in total + tp + tf if np.isfinite(v)] + [1])
    ax.set_ylim(bottom=0, top=ymax + 1)
    ax.grid(True, color="#dddddd", lw=0.8, alpha=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] best progress 그래프 저장: {out_path}")


def _eval_tcn_params(cfg, params, S, seed, device=None, trial_idx=None,
                     trial_checkpoint_dir=None):
    from dataclasses import replace
    from .model_tcn_forecast import TCNForecastDetector
    tcfg = replace(cfg, **params)
    h = TCNForecastDetector(tcfg, device=device).fit(
        S["train_ok"], S["io_cols"], S["sensor_cols"], S["nstep"],
        val_records=S["val_ok"])
    eval_recs = list(S["val_ok"]) + list(S["ng"])
    y = np.concatenate([np.zeros(len(S["val_ok"])), np.ones(len(S["ng"]))])
    best = _score_tcn_candidate(h, cfg, params, S, eval_recs, y)
    if best is None:
        return None
    best_snapshot = _snapshot_tcn_model(h.A)
    history = [{
        "extend_index": 0,
        "accepted": True,
        "trained_epochs": best.get("trained_epochs"),
        "forecast_val_mse": best.get("forecast_val_mse"),
        "obj": list(best.get("obj") or []),
        "correct": best.get("val", {}).get("correct"),
        "tn": best.get("val", {}).get("tn"),
        "tp": best.get("val", {}).get("tp"),
        "precision": best.get("val", {}).get("precision"),
    }]

    extend_epochs = max(0, int(getattr(cfg, "post_eval_extend_epochs", 0) or 0))
    max_extends = max(0, int(getattr(cfg, "post_eval_max_extends", 0) or 0))
    accepted_extends = 0
    attempted_extends = 0
    if extend_epochs > 0 and max_extends > 0:
        for extend_idx in range(1, max_extends + 1):
            attempted_extends = extend_idx
            print(f"  [tcn-post] trial {trial_idx or '?'} post-eval extend "
                  f"{extend_idx}/{max_extends}: +{extend_epochs} epochs",
                  flush=True)
            h.A.train_extra_epochs(S["train_ok"], extend_epochs)
            cand = _score_tcn_candidate(h, cfg, params, S, eval_recs, y)
            if cand is None:
                history.append({
                    "extend_index": extend_idx,
                    "accepted": False,
                    "error": "score_unavailable",
                    "trained_epochs": getattr(h.A, "trained_epochs", ""),
                })
                _restore_tcn_model(h.A, best_snapshot)
                break
            improved = tuple(cand.get("obj") or []) > tuple(best.get("obj") or [])
            history.append({
                "extend_index": extend_idx,
                "accepted": bool(improved),
                "trained_epochs": cand.get("trained_epochs"),
                "forecast_val_mse": cand.get("forecast_val_mse"),
                "obj": list(cand.get("obj") or []),
                "correct": cand.get("val", {}).get("correct"),
                "tn": cand.get("val", {}).get("tn"),
                "tp": cand.get("val", {}).get("tp"),
                "precision": cand.get("val", {}).get("precision"),
            })
            if improved:
                accepted_extends = extend_idx
                best = cand
                best_snapshot = _snapshot_tcn_model(h.A)
                print(f"  [tcn-post] extend {extend_idx} accepted: "
                      f"obj={best.get('obj')} val_mse={best.get('forecast_val_mse'):.6f}",
                      flush=True)
            else:
                _restore_tcn_model(h.A, best_snapshot)
                print(f"  [tcn-post] extend {extend_idx} rejected: "
                      f"candidate obj={cand.get('obj')} <= best obj={best.get('obj')}",
                      flush=True)
                break
    else:
        _restore_tcn_model(h.A, best_snapshot)

    _restore_tcn_model(h.A, best_snapshot)
    checkpoint_path = None
    if trial_checkpoint_dir is not None and trial_idx is not None:
        checkpoint_path = Path(trial_checkpoint_dir) / f"trial_{int(trial_idx):06d}.pt"
        h.A.save(checkpoint_path)
    best["checkpoint_path"] = str(checkpoint_path) if checkpoint_path is not None else None
    best["post_eval_extend_epochs"] = extend_epochs
    best["post_eval_max_extends"] = max_extends
    best["post_eval_extends_attempted"] = attempted_extends
    best["post_eval_extends_accepted"] = accepted_extends
    best["post_eval_history"] = history
    best["trained_epochs"] = getattr(h.A, "trained_epochs", best.get("trained_epochs", ""))
    return best


def _tcn_search_worker(gpu_id, local_rank, cfg, seed, task_q, out_q, threads=0,
                       stop_event=None, trial_checkpoint_dir=None):
    import queue
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pid = os.getpid()
    worker_device = None
    cuda_seen = 0
    cuda_name = ""
    try:
        import torch
        cuda_seen = int(torch.cuda.device_count())
        if cuda_seen > 0:
            if cuda_seen == 1:
                worker_device = "cuda:0"
            else:
                worker_device = f"cuda:{int(local_rank) % cuda_seen}"
            torch.cuda.set_device(worker_device)
            cuda_name = torch.cuda.get_device_name(torch.device(worker_device))
    except Exception as e:
        cuda_name = f"cuda init failed: {e}"
    print(f"  [worker pid={pid}] gpu={gpu_id} | "
          f"torch_visible={cuda_seen} device={worker_device or 'cpu'} {cuda_name}",
          flush=True)
    if threads and threads > 0:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(int(threads))
        try:
            import torch
            torch.set_num_threads(int(threads))
        except Exception:
            pass
    try:
        S = _load_forecast_split(cfg, seed)
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                item = task_q.get(timeout=30.0)
            except queue.Empty:
                continue
            if item is None:
                break
            trial_idx, params = item if isinstance(item, tuple) else (None, item)
            if trial_idx is not None:
                print(f"  [worker pid={pid} gpu={gpu_id}] start trial {trial_idx}",
                      flush=True)
            try:
                r = _eval_tcn_params(
                    cfg, params, S, seed, device=worker_device,
                    trial_idx=trial_idx,
                    trial_checkpoint_dir=trial_checkpoint_dir)
                if r is None and trial_idx is not None:
                    r = {"error": "채점 불가(클래스 부족)", "params": params,
                         "trial_idx": trial_idx, "score_unavailable": True}
                if isinstance(r, dict) and trial_idx is not None:
                    r["trial_idx"] = trial_idx
                    r["worker_gpu"] = str(gpu_id)
                    r["worker_device"] = str(worker_device or "cpu")
                    r["worker_pid"] = int(pid)
            except Exception as e:
                r = {"error": str(e), "params": params, "trial_idx": trial_idx,
                     "worker_gpu": str(gpu_id), "worker_device": str(worker_device or "cpu"),
                     "worker_pid": int(pid)}
            out_q.put(r)
            if stop_event is not None and stop_event.is_set():
                break
    finally:
        out_q.put({"_done": str(gpu_id), "worker_pid": int(pid)})


def _fmt_tcn_trial(i, total, r):
    p, vc = r["params"], r["val"]
    rf_ok = "OK" if r["receptive"] >= r["need"] else "SHORT"
    worker = ""
    if r.get("worker_gpu") is not None:
        worker = (f" | gpu={r.get('worker_gpu')} device={r.get('worker_device')} "
                  f"pid={r.get('worker_pid')}")
    post = ""
    if int(r.get("post_eval_extends_attempted", 0) or 0) > 0:
        post = (f" | post_ext accepted={r.get('post_eval_extends_accepted', 0)}/"
                f"{r.get('post_eval_extends_attempted', 0)} "
                f"epochs={r.get('trained_epochs', '')}")
    return (f"  [{i:>3}/{total}] val_mse={r['forecast_val_mse']:.6f} "
            f"(trial {r.get('trial_idx', i)}) | threshold 맞춤 {vc['correct']}/{vc['n']} "
            f"precision={float(vc.get('precision', 0.0)):.3f} "
            f"(양품 {vc['tn']}/{vc['n_ok']} 불량 {vc['tp']}/{vc['n_ng']}) "
            f"AUROC={vc['auroc']:.3f} | w{p['tcn_past_window']} h{p['tcn_future_window']} "
            f"s{p['tcn_stride']} rf{r['receptive']}/{r['need']} {rf_ok} "
            f"ch{p['tcn_channels']} blk{p['tcn_blocks']} k{p['tcn_kernel']} "
            f"q{p['raw_error_quantile']}{post}{worker}")


def _fmt_tcn_best(cur, is_new):
    v = cur["val"]
    trial = cur.get("trial_idx", "?")
    return (f"        -> current best: 맞춤 {v['correct']}/{v['n']} "
            f"precision={float(v.get('precision', 0.0)):.3f} "
            f"(양품 {v['tn']}/{v['n_ok']} 불량 {v['tp']}/{v['n_ng']}) "
            f"| val_mse={cur['forecast_val_mse']:.6f} | best trial {trial}"
            + (" *NEW best" if is_new else ""))


def _is_full_validation_correct(r: dict) -> bool:
    val = r.get("val", {}) if isinstance(r, dict) else {}
    return bool(val and val.get("n") and val.get("correct") == val.get("n"))


def _save_tcn_checkpoint(cfg, best_params, S, best_result: dict = None):
    import contextlib
    import io as _io
    from dataclasses import replace
    from .model_tcn_forecast import TCNForecastDetector
    bcfg = replace(cfg, **best_params)
    with contextlib.redirect_stdout(_io.StringIO()):
        h = TCNForecastDetector(bcfg).fit(
            S["train_ok"], S["io_cols"], S["sensor_cols"], S["nstep"],
            val_records=S["val_ok"])
        accepted = max(0, int((best_result or {}).get("post_eval_extends_accepted", 0) or 0))
        extend_epochs = max(0, int((best_result or {}).get(
            "post_eval_extend_epochs",
            getattr(cfg, "post_eval_extend_epochs", 0)) or 0))
        for _ in range(accepted):
            h.A.train_extra_epochs(S["train_ok"], extend_epochs)
        h.save(bcfg.artifacts_dir)
    return Path(bcfg.artifacts_dir)


def _install_trial_checkpoint(adir: Path, result: dict) -> bool:
    """Install the already-trained trial checkpoint as current best."""
    import shutil
    src = result.get("checkpoint_path") if isinstance(result, dict) else None
    if not src:
        return False
    src = Path(src)
    if not src.exists():
        print(f"        trial checkpoint 없음: {src}")
        return False
    adir = Path(adir)
    adir.mkdir(parents=True, exist_ok=True)
    dst = adir / "tcn_forecast.pt"
    tmp = adir / "tcn_forecast.pt.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    print(f"        checkpoint saved: {dst} (from {src.name})")
    return True


def _save_trial_artifact_bundle(adir: Path, result: dict, completed_idx: int,
                                total: int, run_id: str,
                                score_name: str, model_kind: str,
                                command_record: Path | None = None) -> bool:
    """Save one successful trial as a standalone artifact directory."""
    import shutil
    if not isinstance(result, dict) or result.get("error") or not result.get("val"):
        return False
    trial_idx = int(result.get("trial_idx") or completed_idx)
    out = Path(adir) / f"artifacts{trial_idx:04d}"
    out.mkdir(parents=True, exist_ok=True)

    src = result.get("checkpoint_path")
    if not src or not Path(src).exists():
        print(f"        trial artifact 저장 실패: checkpoint 없음 ({src})")
        return False
    dst = out / "tcn_forecast.pt"
    tmp = out / "tcn_forecast.pt.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    bundle_result = dict(result)
    bundle_result["checkpoint_path"] = str(dst)

    for name in ("search_best_progress.csv", "search_trial_progress.csv",
                 "search_best_progress.png", "search_best.json", "best_search.json"):
        try:
            (out / name).unlink(missing_ok=True)
        except Exception:
            pass
    _write_search_jsons(out, ("search_best.json", "best_search.json"),
                        bundle_result, score_name, model_kind)
    _append_best_progress_csv(out / "search_best_progress.csv",
                              bundle_result, 1, score_name, model_kind)
    _append_trial_progress_csv(out / "search_trial_progress.csv",
                               bundle_result, bundle_result, trial_idx, total, run_id,
                               score_name, model_kind)
    try:
        _plot_best_progress(out / "search_best_progress.csv",
                            out / "search_best_progress.png")
    except Exception as e:
        print(f"        trial artifact best progress plot 실패: {e}")
    if command_record is not None and Path(command_record).exists():
        try:
            shutil.copy2(command_record, out / Path(command_record).name)
        except Exception as e:
            print(f"        trial artifact command copy 실패: {e}")
    print(f"        trial artifact saved: {out}")
    return True


def _cleanup_trial_checkpoint(result: dict) -> None:
    src = result.get("checkpoint_path") if isinstance(result, dict) else None
    if not src:
        return
    try:
        Path(src).unlink(missing_ok=True)
    except Exception:
        pass


def cmd_tcn_search(cfg: IOConfig, trials: int = 12, seed: int = 42,
                   fixed_epochs: int = None, train_best: bool = True,
                   n_workers: int = 0, checkpoint: bool = True,
                   resume: bool = False,
                   stop_on_perfect_val: bool = False,
                   save_trial_artifacts: bool = False) -> None:
    if _threshold_source(cfg) == "train-ok-sigma" and _threshold_wise(cfg) not in {"all", "recipe"}:
        sys.exit("--threshold-source train-ok-sigma는 wafer final score 기준이라 "
                 "--threshold-wise all 또는 recipe와 함께 사용해야 합니다.")
    _validate_threshold_sigma_range_for_search(cfg)
    print(f"data dirs: OK={cfg.ok_dir} | NG={cfg.ng_dir} | VAL={cfg.val_dir}")
    print(f"artifacts dir: {cfg.artifacts_dir}")
    print(f"h_cart_mode: {cfg.h_cart_mode}")
    print(f"file_format: {getattr(cfg, 'file_format', 'auto')}")
    h_keep = h_sensor_keep_for_config(cfg)
    print(f"h_sensor_preset: {getattr(cfg, 'h_sensor_preset', '') or 'all'}"
          f"{f' ({len(h_keep)} sensors)' if h_keep else ''}")
    print(f"sensor_role_profile: {normalize_sensor_role_profile(getattr(cfg, 'sensor_role_profile', ''))}")
    print(f"future_h_mode: {cfg.future_h_mode}")
    print(f"virtual_sensor_mode: {getattr(cfg, 'virtual_sensor_mode', 'off')}")
    print(f"threshold_wise: {_threshold_wise(cfg)}")
    sigma_desc = ""
    if _threshold_source(cfg) == "train-ok-sigma":
        if _threshold_sigma_auto(cfg):
            sigma_desc = " (train OK mean + auto sigma)"
            if _threshold_sigma_range_enabled(cfg):
                sigma_desc += f" range={_format_threshold_sigma_range(cfg)}"
        else:
            sigma_desc = f" (train OK mean + {_threshold_sigma_fixed(cfg):g} sigma)"
    print(f"threshold_source: {_threshold_source(cfg)}{sigma_desc}")
    print(f"score topk sensors: {_score_topk(cfg) or 'all'}")
    print(f"score time-step average: {'ON' if _score_time_avg(cfg) else 'OFF'}")
    print(f"score final-length average: {'ON' if _score_final_avg(cfg) else 'OFF'}")
    if _score_agg_enabled(cfg):
        print(f"score aggregation: resid_agg={_score_resid_agg(cfg)} "
              f"wafer_agg={_score_wafer_agg(cfg)}")
    else:
        print("score aggregation: current default (time sum -> selected sensor sum)")
    print(f"search objective: {_search_objective_label(cfg)}")
    if int(getattr(cfg, "post_eval_max_extends", 0) or 0) > 0:
        print("post-eval extension: "
              f"+{int(getattr(cfg, 'post_eval_extend_epochs', 10) or 0)} epochs × "
              f"max {int(getattr(cfg, 'post_eval_max_extends', 0) or 0)} "
              "(accept only if VAL objective improves)")
    else:
        print("post-eval extension: disabled")
    multipliers = _display_threshold_recipe_multipliers(cfg)
    if multipliers:
        msg = ", ".join(f"{recipe}×{factor:g}" for recipe, factor in multipliers.items())
        print(f"threshold recipe multipliers: {msg}")
    S = _load_forecast_split(cfg, seed)
    if not S["train_ok"] or not S["val_ok"] or not S["ng"]:
        sys.exit("필요: data/OK + data/VAL/OK + data/VAL/NG. "
                 f"현재 train_ok={len(S['train_ok'])} val_ok={len(S['val_ok'])} "
                 f"ng={len(S['ng'])}")
    if not S["io_cols"]:
        print("주의: IO(숫자명) 컬럼 없음. 미래 IO 조건을 사용할 수 없습니다.")

    _tcn_search_space_for_run(fixed_epochs)
    total = max(0, int(trials))
    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 0
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_gpus = [x.strip() for x in cuda_env.split(",") if x.strip()] \
        if cuda_env else [str(i) for i in range(n_gpu)]
    if n_workers and n_workers > 0:
        requested_workers = int(n_workers)
        n_workers = min(requested_workers, max(1, len(visible_gpus))) if n_gpu else requested_workers
    else:
        requested_workers = 0
        n_workers = max(1, len(visible_gpus)) if n_gpu else 1
    print(f"CUDA_VISIBLE_DEVICES={cuda_env or '(unset)'} | "
          f"torch.cuda.device_count={n_gpu} | worker GPU list={visible_gpus or ['cpu']} "
          f"| requested_workers={requested_workers or 'auto'}")

    print(f"TCN forecast search: OK total {len(S['ok_all'])} | "
          f"Train_OK {len(S['train_ok'])} / Val_OK {len(S['val_ok'])} / NG {len(S['ng'])} "
          f"| IO {len(S['io_cols'])} | sensor {len(S['sensor_cols'])} | steps {S['nstep']} "
          f"| trials={total} seed={seed} | GPU워커={n_workers}{f'/{n_gpu}장' if n_gpu else '(CPU)'}")
    print(f"목표: {_search_objective_goal_label(cfg)}")
    print("threshold-wise: all=전체 1개, recipe=레시피별, "
          "sensor=레시피×센서별, step=레시피×센서×스텝별")
    print(f"perfect validation 조기 종료: {'ON' if stop_on_perfect_val else 'OFF'}")
    print(f"all trial artifacts: {'ON' if save_trial_artifacts else 'OFF'}")
    print(f"동률 기준: {_search_objective_tiebreak_label(cfg)}")
    print("ProcessStep 0: 입력에는 사용, loss/score/report/explain 강조 계산에서는 제외")
    print("threshold score: step별 Σ max(0, |actual-expected| / OK기준오차 - 1)")
    print("raw residual 기준: OK residual 분위수 초과분을 사용하며 std 나눗셈은 사용하지 않음")
    print(f"split: {S['split_mode']}\n")

    adir = Path(cfg.artifacts_dir)
    command_record = _write_command_record(
        adir, "tcn_search_command.txt", "tcn-search",
        artifacts_dir=adir)
    trial_ckpt_dir = adir / "_trial_checkpoints"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_progress_csv = adir / "search_trial_progress.csv"
    best_progress_csv = adir / "search_best_progress.csv"
    best_progress_png = adir / "search_best_progress.png"
    print(f"실행 명령 기록: {command_record}")
    print(f"trial 성능 누적 CSV: {trial_progress_csv}")
    print(f"best 갱신 CSV: {best_progress_csv}")
    print(f"best 갱신 그래프: {best_progress_png}\n")

    scheduler = _TcnOptunaScheduler(trials, seed, fixed_epochs,
                                    trial_progress_csv, resume, cfg)
    print("search sampler: Optuna TPE Bayesian")
    results = []
    state = {"ckpt": None, "best_events": 0}
    if resume:
        resume_best = _load_resume_best(adir, None, "tcn_forecast_ok_only", cfg)
        if resume_best:
            results.append(resume_best)
            if checkpoint and _checkpoint_ready(adir):
                state["ckpt"] = resume_best.get("params")
        state["best_events"] = _read_best_event_count(best_progress_csv)
        print(f"resume 활성화: 완료된 Optuna trial {scheduler.completed_history}/{total}개 반영")
        if resume_best:
            v = resume_best.get("val", {})
            print(f"resume best 로드: trial {resume_best.get('trial_idx')} "
                  f"맞춤 {v.get('correct')}/{v.get('n')}")

    remaining_trials = scheduler.remaining_to_schedule
    if resume and remaining_trials:
        print(f"resume next trial: {scheduler.next_trial_idx}/{total}")
    if remaining_trials and n_workers > remaining_trials:
        print(f"  worker 수 조정: 남은 trial {remaining_trials}개라 "
              f"{n_workers} -> {remaining_trials}")
        n_workers = remaining_trials
    if requested_workers and n_gpu and requested_workers > len(visible_gpus):
        print(f"  주의: --n-workers {requested_workers} 요청했지만 visible GPU가 "
              f"{len(visible_gpus)}개라 worker는 {n_workers}개만 사용합니다.")
    trial_checkpoint_enabled = bool(checkpoint or save_trial_artifacts)
    if trial_checkpoint_enabled:
        print(f"trial checkpoint dir: {trial_ckpt_dir}")

    def _report(r, idx):
        display_idx = r.get("trial_idx", idx) if isinstance(r, dict) else idx
        if r is None:
            cur_best = max(results, key=lambda x: tuple(x["obj"])) if results else {}
            _append_trial_progress_csv(trial_progress_csv, {}, cur_best,
                                       display_idx, total, run_id,
                                       _threshold_score_name(cfg),
                                       "tcn_forecast_ok_only")
            print(f"  [{display_idx:>3}/{total}] 채점 불가(클래스 부족)")
            return False
        if "error" in r:
            cur_best = max(results, key=lambda x: tuple(x["obj"])) if results else {}
            _append_trial_progress_csv(trial_progress_csv, r, cur_best, display_idx, total, run_id,
                                       _threshold_score_name(cfg),
                                       "tcn_forecast_ok_only")
            print(f"  [{display_idx:>3}/{total}] 실패: {r['error']}")
            return False
        results.append(r)
        print(_fmt_tcn_trial(display_idx, total, r))
        cur = max(results, key=lambda x: tuple(x["obj"]))
        is_new = cur is results[-1]
        _append_trial_progress_csv(trial_progress_csv, r, cur, display_idx, total, run_id,
                                   _threshold_score_name(cfg),
                                   "tcn_forecast_ok_only")
        print(_fmt_tcn_best(cur, is_new))
        if save_trial_artifacts:
            try:
                _save_trial_artifact_bundle(
                    adir, r, display_idx, total, run_id,
                    _threshold_score_name(cfg), "tcn_forecast_ok_only",
                    command_record=command_record)
            except Exception as e:
                print(f"        trial artifact 저장 실패(서치는 계속): {e}")
        if is_new:
            state["best_events"] += 1
            try:
                _append_best_progress_csv(best_progress_csv,
                                          cur, state["best_events"],
                                          _threshold_score_name(cfg),
                                          "tcn_forecast_ok_only")
                _plot_best_progress(best_progress_csv, best_progress_png)
            except Exception as e:
                print(f"        best progress 저장 실패(서치는 계속): {e}")
            try:
                _write_search_jsons(adir, ("search_best.json", "best_search.json"),
                                    cur, _threshold_score_name(cfg),
                                    "tcn_forecast_ok_only")
            except Exception as e:
                print(f"        search_best 저장 실패(서치는 계속): {e}")
        if checkpoint and is_new:
            try:
                installed = _install_trial_checkpoint(adir, cur)
                if not installed:
                    print("        trial checkpoint 설치 실패: best params로 재학습 저장")
                    _save_tcn_checkpoint(cfg, cur["params"], S, cur)
                state["ckpt"] = cur["params"]
                _write_search_jsons(adir, ("search_best.json", "best_search.json"),
                                    cur, _threshold_score_name(cfg),
                                    "tcn_forecast_ok_only")
                print("        best checkpoint ready: "
                      "tcn_forecast.pt + search_best.json + progress.csv + progress.png")
            except Exception as e:
                print(f"        checkpoint 저장 실패(서치는 계속): {e}")
        if trial_checkpoint_enabled:
            _cleanup_trial_checkpoint(r)
        if stop_on_perfect_val and _is_full_validation_correct(r):
            v = r["val"]
            print("        validation 전체 wafer 정답으로 search 조기 종료: "
                  f"{v['correct']}/{v['n']} "
                  f"(양품 {v['tn']}/{v['n_ok']} 불량 {v['tp']}/{v['n_ng']})")
            return True
        return False

    if n_workers <= 1:
        while scheduler.remaining_to_schedule > 0:
            item = scheduler.ask()
            if item is None:
                break
            i, params = item
            try:
                r = _eval_tcn_params(
                    cfg, params, S, seed, trial_idx=i,
                    trial_checkpoint_dir=trial_ckpt_dir if trial_checkpoint_enabled else None)
                if isinstance(r, dict):
                    r["trial_idx"] = i
            except Exception as e:
                r = {"error": str(e), "params": params, "trial_idx": i}
            scheduler.tell(r, i)
            if _report(r, i):
                break
    else:
        import multiprocessing as mp
        import queue
        cpu_total = os.cpu_count() or n_workers
        per_worker = max(1, cpu_total // max(1, n_workers))
        print(f"  CPU threads/worker={per_worker} (cores {cpu_total} / workers {n_workers})")
        ctx = mp.get_context("spawn")
        task_q = ctx.Queue()
        out_q = ctx.Queue()
        stop_event = ctx.Event()
        initial = 0
        for _ in range(min(n_workers, scheduler.remaining_to_schedule)):
            item = scheduler.ask()
            if item is None:
                break
            task_q.put(item)
            initial += 1
        if initial <= 0:
            print("  실행할 Optuna trial 없음.")
            n_workers = 0
        procs = []
        for w in range(n_workers):
            gpu = visible_gpus[w % len(visible_gpus)] if n_gpu else 0
            pr = ctx.Process(target=_tcn_search_worker,
                             args=(gpu, w, cfg, seed, task_q, out_q,
                                   per_worker, stop_event,
                                   trial_ckpt_dir if trial_checkpoint_enabled else None))
            pr.start()
            procs.append(pr)
        print(f"  dynamic Optuna trial queue: initial={initial} | remaining={scheduler.remaining_to_schedule} "
              f"| workers={n_workers} "
              f"| gpu assignment={[visible_gpus[w % len(visible_gpus)] if n_gpu else 'cpu' for w in range(n_workers)]}")
        done_pids, idx, early_stop = set(), 0, False
        while len(done_pids) < n_workers and scheduler.reported < scheduler.to_schedule:
            try:
                msg = out_q.get(timeout=10)
            except queue.Empty:
                for pr in procs:
                    if pr.pid in done_pids:
                        continue
                    if not pr.is_alive() and pr.exitcode is not None:
                        done_pids.add(pr.pid)
                        if pr.exitcode != 0:
                            print(f"  [worker pid={pr.pid}] 비정상 종료 exitcode={pr.exitcode}")
                if len(done_pids) < n_workers:
                    alive = [pr.pid for pr in procs if pr.pid not in done_pids and pr.is_alive()]
                    print(f"  [search] worker 대기 중: alive={alive} done={len(done_pids)}/{n_workers}",
                          flush=True)
                continue
            if isinstance(msg, dict) and "_done" in msg:
                done_pids.add(int(msg.get("worker_pid", -1)))
                print(f"  [worker pid={msg.get('worker_pid')}] done gpu={msg.get('_done')}",
                      flush=True)
                continue
            idx += 1
            scheduler.tell(msg, msg.get("trial_idx") if isinstance(msg, dict) else None)
            if _report(msg, idx):
                early_stop = True
                stop_event.set()
                break
            if not early_stop and scheduler.remaining_to_schedule > 0:
                item = scheduler.ask()
                if item is not None:
                    task_q.put(item)
        if early_stop:
            for pr in procs:
                if pr.is_alive():
                    pr.terminate()
        else:
            stop_event.set()
            for _ in procs:
                task_q.put(None)
        for pr in procs:
            pr.join()

    if not results:
        sys.exit("성공한 TCN trial 없음.")
    results.sort(key=lambda r: tuple(r["obj"]), reverse=True)
    best = results[0]
    print(f"\n=== TCN TOP 5 ({_search_objective_label(cfg)} 기준) ===")
    for r in results[:5]:
        v = r["val"]
        print(f"  val_mse={r['forecast_val_mse']:.6f} "
              f"| threshold 맞춤 {v['correct']}/{v['n']} "
              f"precision={float(v.get('precision', 0.0)):.3f} "
              f"(양품 {v['tn']}/{v['n_ok']} 불량 {v['tp']}/{v['n_ng']}) "
              f"AUROC={v['auroc']:.3f} | {r['params']}")
    print("\n=== BEST ===")
    v = best["val"]
    print(f"  threshold validation: {v['correct']}/{v['n']} "
          f"precision={float(v.get('precision', 0.0)):.3f} "
          f"(양품 {v['tn']}/{v['n_ok']} 불량 {v['tp']}/{v['n_ng']}) "
          f"threshold={best['threshold']} AUROC={v['auroc']:.3f}")
    print(f"  forecast val MSE: {best['forecast_val_mse']:.6f}")
    print("  config.py 반영 후보:")
    for k, val in best["params"].items():
        print(f"      {k:26s}= {val!r}")
    _write_search_jsons(adir, ("search_best.json", "best_search.json"),
                        best, _threshold_score_name(cfg),
                        "tcn_forecast_ok_only")
    print(f"\n  search best 저장: {adir / 'search_best.json'} (+ best_search.json)")
    if not train_best:
        print("  (--no-train: 최종 best 학습 생략)")
        return
    if checkpoint and state["ckpt"] == best["params"]:
        print(f"  최종 best는 이미 checkpoint로 저장됨: {adir / 'tcn_forecast.pt'}")
        return
    _save_tcn_checkpoint(cfg, best["params"], S, best)
    print(f"  저장: {adir / 'tcn_forecast.pt'}")


def cmd_tcn_export(cfg: IOConfig, report_dir: Path = None) -> None:
    from .export_report import build_report, run_existing_rca
    from .model_tcn_forecast import TCNForecastDetector
    adir = Path(cfg.artifacts_dir)
    print(f"data dirs: OK={cfg.ok_dir} | NG={cfg.ng_dir} | VAL={cfg.val_dir}")
    print(f"artifacts dir: {adir}")
    print(f"h_cart_mode: {cfg.h_cart_mode}")
    print(f"file_format: {getattr(cfg, 'file_format', 'auto')}")
    print(f"sensor_role_profile: {normalize_sensor_role_profile(getattr(cfg, 'sensor_role_profile', ''))}")
    print(f"future_h_mode: {cfg.future_h_mode}")
    print(f"virtual_sensor_mode: {getattr(cfg, 'virtual_sensor_mode', 'off')}")
    mp = adir / "tcn_forecast.pt"
    if not mp.exists():
        sys.exit(f"TCN forecast 모델 없음: 먼저 tcn-search 를 실행하세요. ({mp})")
    h = TCNForecastDetector.load(adir, cfg)

    val_ok_dir = _sub_ci(cfg.val_dir, "ok")
    val_ng_dir = _sub_ci(cfg.val_dir, "ng")
    ok, _, _, _ = load_dataset(val_ok_dir, cfg)
    ng, _, _, _ = load_dataset(val_ng_dir, cfg)
    if ok and ng:
        print(f"평가 대상: validation holdout - {val_ok_dir} {len(ok)} + {val_ng_dir} {len(ng)}")
    else:
        ok, _, _, _ = load_dataset(cfg.ok_dir, cfg)
        ng, _, _, _ = load_dataset(cfg.ng_dir, cfg)
        print(f"주의: VAL/OK 또는 VAL/NG 없음. 폴백: {cfg.ok_dir} {len(ok)} + {cfg.ng_dir} {len(ng)}")
    if not ok or not ng:
        sys.exit("평가 데이터 필요: data/VAL/OK + data/VAL/NG")
    _write_val_recipe_files(cfg, ok, ng)

    score_map = None
    thr = 1.0
    sb = next((p for p in (adir / "search_best.json", adir / "best_search.json")
               if p.exists()), None)
    if sb is not None:
        try:
            best = json.loads(sb.read_text(encoding="utf-8")).get("best", {})
            if best.get("model_kind") == "tcn_forecast_ok_only":
                sc = best.get("scores") or best.get("anomaly_scores")
                th = best.get("threshold", best.get("val", {}).get("threshold_raw",
                              best.get("val", {}).get("threshold")))
                eval_ids = {r["wafer_id"] for r in (ok + ng)}
                if sc and th is not None and eval_ids.issubset(set(sc)):
                    score_map = sc
                    eval_recs = list(ok) + list(ng)
                    y = np.concatenate([np.zeros(len(ok)), np.ones(len(ng))])
                    p = np.asarray([float(score_map[r["wafer_id"]]) for r in eval_recs])
                    stored_mode = _stored_threshold_wise(th)
                    if stored_mode in {"sensor", "step"}:
                        v = _metrics_at_threshold(y, p, 1.0)
                        v["auroc"] = _safe_auroc(y, p)
                        v["threshold_wise"] = stored_mode
                        thr = th
                    elif isinstance(th, dict):
                        preds = []
                        for rec, score in zip(eval_recs, p):
                            cutoff = _saved_threshold_for_record(th, rec)
                            preds.append(0 if cutoff is None else int(float(score) >= float(cutoff)))
                        v = _metrics_from_pred(y, np.asarray(preds, dtype=int), p)
                        v["threshold_wise"] = stored_mode or _threshold_wise(cfg)
                        thr = th
                    else:
                        v = _correct_from_scores(y, p, cfg)
                        thr = float(th)
                        if v is None:
                            v = best.get("val", {})
                    print(f"서치 best 점수 재사용: 맞춤 {v.get('correct')}/{v.get('n')} "
                          f"(양품 {v.get('tn')}/{v.get('n_ok')} 불량 {v.get('tp')}/{v.get('n_ng')}) "
                          f"threshold={thr}")
                else:
                    print("주의: search_best 점수가 현재 평가셋과 불일치해 점수 재계산")
            else:
                print(f"주의: {sb.name}은 TCN forecast 결과가 아니라 점수 재계산")
        except Exception as e:
            print(f"주의: search_best 읽기 실패, 점수 재계산: {e}")
    else:
        print("주의: search_best.json 없음. tcn-search 후 export 권장.")

    rep = Path(report_dir) if report_dir is not None else _resolve_report_dir(cfg)
    print(f"리포트 계약 생성 -> {rep}")
    build_report(h, ok, ng, rep, cfg, threshold=thr, score_map=score_map)
    print("기존 RCA 도구 실행(wafer_vs_baseline + 자연어 RCA + heatmap v3)...")
    run_existing_rca(rep, cfg)
    print(f"\n완료: {rep}")


def cmd_plot_best_progress(cfg: IOConfig) -> None:
    adir = Path(cfg.artifacts_dir)
    _plot_best_progress(adir / "search_best_progress.csv",
                        adir / "search_best_progress.png")


def cmd_convert_onnx(cfg: IOConfig, opset: int = 17, verify_export: bool = True) -> None:
    adir = Path(cfg.artifacts_dir)
    pt = adir / "tcn_forecast.pt"
    if not pt.exists():
        sys.exit(f"TCN forecast 모델 없음: 먼저 tcn-search 를 실행하세요. ({pt})")
    from .export_onnx_forecast import export, verify
    print(f"artifacts dir: {adir}")
    print(f"source checkpoint: {pt}")
    onnx_path = export(pt_path=pt, out_dir=adir, opset=opset)
    if verify_export:
        import importlib.util as _ilu
        if _ilu.find_spec("onnxruntime") is None:
            print("ONNX 검증 생략: onnxruntime 없음")
        else:
            verify(pt_path=pt, out_dir=adir)
    print("ONNX artifacts saved:")
    for name in ("tcn_forecast.onnx", "tcn_forecast.npz", "recon_meta.npz"):
        p = adir / name
        print(f"  - {p}{'' if p.exists() else ' (missing)'}")
    print(f"converted: {onnx_path}")


def _trial_artifact_dirs(adir: Path) -> list[Path]:
    adir = Path(adir)
    if adir.is_dir() and re.fullmatch(r"artifacts\d+", adir.name):
        return [adir]
    items = [
        p for p in adir.iterdir()
        if p.is_dir() and re.fullmatch(r"artifacts\d+", p.name)
    ] if adir.exists() else []
    return sorted(items, key=lambda p: (int(re.search(r"\d+", p.name).group()), p.name))


def _resolve_single_trial_artifact_dir(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PKG_ROOT / p
    return p


def _read_trial_artifact_best(path: Path) -> tuple[dict, Path]:
    for name in ("search_best.json", "best_search.json"):
        candidate = Path(path) / name
        if not candidate.exists():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        best = data.get("best", data)
        if not isinstance(best, dict):
            raise ValueError(f"{candidate.name} best payload is not a dict")
        return best, candidate
    raise FileNotFoundError("search_best.json 또는 best_search.json 없음")


def _json_summary_cell(value) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _trial_artifact_summary_row(path: Path) -> dict:
    row = {
        "artifact_dir": str(path),
        "artifact_name": path.name,
        "status": "ok",
        "error": "",
        "source_json": "",
        "trial_idx": "",
        "correct": "",
        "tp": "",
        "tn": "",
        "fp": "",
        "fn": "",
        "n": "",
        "n_ok": "",
        "n_ng": "",
        "precision": "",
        "recall": "",
        "f1_score": "",
        "auroc": "",
        "val_mse": "",
        "n_params": "",
        "threshold": "",
        "threshold_wise": "",
        "search_objective": "",
        "trained_epochs": "",
        "params_json": "",
    }
    try:
        best, source = _read_trial_artifact_best(path)
        val = best.get("val") if isinstance(best.get("val"), dict) else {}
        eval_criteria = (
            best.get("evaluation_criteria")
            if isinstance(best.get("evaluation_criteria"), dict)
            else {}
        )
        row.update({
            "source_json": str(source),
            "trial_idx": best.get("trial_idx", ""),
            "correct": val.get("correct", eval_criteria.get("correct_wafers", "")),
            "tp": val.get("tp", eval_criteria.get("tp", "")),
            "tn": val.get("tn", eval_criteria.get("tn", "")),
            "fp": val.get("fp", eval_criteria.get("fp", "")),
            "fn": val.get("fn", eval_criteria.get("fn", "")),
            "n": val.get("n", eval_criteria.get("n_wafers", "")),
            "n_ok": val.get("n_ok", eval_criteria.get("n_ok", "")),
            "n_ng": val.get("n_ng", eval_criteria.get("n_ng", "")),
            "precision": val.get("precision", eval_criteria.get("precision", "")),
            "recall": val.get("recall", eval_criteria.get("recall", "")),
            "f1_score": val.get("f1_score", eval_criteria.get("f1_score", "")),
            "auroc": val.get("auroc", eval_criteria.get("auroc", "")),
            "val_mse": best.get("forecast_val_mse", ""),
            "n_params": best.get("n_params", eval_criteria.get("n_params", "")),
            "threshold": _json_summary_cell(
                best.get("threshold", val.get("threshold_raw", val.get("threshold", "")))
            ),
            "threshold_wise": val.get("threshold_wise", ""),
            "search_objective": best.get("search_objective", ""),
            "trained_epochs": best.get("trained_epochs", ""),
            "params_json": _json_summary_cell(best.get("params", {})),
        })
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def cmd_summarize_trial_artifacts(cfg: IOConfig, out_csv: str | None = None) -> None:
    adir = Path(cfg.artifacts_dir)
    artifact_dirs = _trial_artifact_dirs(adir)
    if not artifact_dirs:
        sys.exit(f"trial artifact 폴더 없음: {adir / 'artifacts0001'} 형식")
    out = Path(out_csv) if out_csv else (adir / "trial_artifacts_summary.csv")
    if not out.is_absolute():
        out = adir / out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [_trial_artifact_summary_row(path) for path in artifact_dirs]
    fields = [
        "artifact_dir", "artifact_name", "status", "error", "source_json",
        "trial_idx", "correct", "tp", "tn", "fp", "fn", "n", "n_ok", "n_ng",
        "precision", "recall", "f1_score", "auroc", "val_mse", "n_params",
        "threshold", "threshold_wise", "search_objective", "trained_epochs",
        "params_json",
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)
    ok = sum(1 for row in rows if row.get("status") == "ok")
    err = len(rows) - ok
    print(f"trial artifact summary saved: {out}")
    print(f"artifacts: {len(rows)} | ok={ok} | error={err}")


def _target_data_dirs_for_prediction(cfg: IOConfig) -> list[tuple[Path, str | None]]:
    from .data import _dataset_files

    val_ok = _sub_ci(cfg.val_dir, "ok")
    val_ng = _sub_ci(cfg.val_dir, "ng")
    out: list[tuple[Path, str | None]] = []
    if val_ok.exists() and _dataset_files(val_ok, cfg):
        out.append((val_ok, "OK"))
    if val_ng.exists() and _dataset_files(val_ng, cfg):
        out.append((val_ng, "NG"))
    if out:
        return out
    if Path(cfg.ng_dir).exists() and _dataset_files(Path(cfg.ng_dir), cfg):
        return [(Path(cfg.ng_dir), "NG")]
    return [(Path(cfg.ok_dir).parent, None)]


def _read_data_file_timestamp(path: Path) -> datetime:
    import pandas as pd
    from .data import _file_format_for_path, _read_csv

    fmt = _file_format_for_path(path, IOConfig(file_format="auto"))
    kwargs = {"sep": "\t"} if fmt == "log" else {}
    raw = _read_csv(path, **kwargs)
    if raw.empty:
        raise ValueError("empty file")
    col_by_lower = {str(c).strip().lower(): c for c in raw.columns}
    date_col = col_by_lower.get("date")
    time_col = col_by_lower.get("time")
    if date_col is None or time_col is None:
        raise ValueError("Date/Time column missing")
    date_value = str(raw.iloc[-1][date_col]).strip()
    time_value = str(raw.iloc[-1][time_col]).strip()
    if not date_value or not time_value or date_value.lower() == "nan" or time_value.lower() == "nan":
        raise ValueError("last row Date/Time is empty")
    return pd.to_datetime(
        f"{date_value} {time_value}",
        format="%Y/%m/%d %H:%M:%S.%f",
        errors="raise",
    ).to_pydatetime()


def _load_prediction_records_for_model(cfg: IOConfig, A, warnings: list[dict]) -> list[dict]:
    import pandas as pd
    from .data import (
        _dataset_files,
        _infer_recipe,
        _read_data_file,
        add_virtual_sensors,
    )

    records: list[dict] = []
    seen: set[Path] = set()
    required_sensors = list(dict.fromkeys(
        list(getattr(A, "input_sensor_cols", []) or [])
        + list(getattr(A, "sensor_cols", []) or [])
    ))
    io_cols = list(getattr(A, "io_cols", []) or [])
    step_names = {str(x).strip().lower() for x in getattr(cfg, "step_names", ()) or ()}
    for folder, label in _target_data_dirs_for_prediction(cfg):
        folder = Path(folder)
        files = _dataset_files(folder, cfg)
        for path in files:
            path = Path(path)
            if path in seen:
                continue
            seen.add(path)
            try:
                timestamp = _read_data_file_timestamp(path)
            except Exception as exc:
                warnings.append({
                    "artifact": "",
                    "source_path": str(path),
                    "stage": "timestamp",
                    "warning": str(exc),
                })
                print(f"  ⚠ timestamp 읽기 실패: {path} ({exc})")
                continue
            try:
                df = add_virtual_sensors(_read_data_file(path, cfg), cfg)
            except Exception as exc:
                warnings.append({
                    "artifact": "",
                    "source_path": str(path),
                    "stage": "load_log",
                    "warning": str(exc),
                })
                print(f"  ⚠ log/data 읽기 실패: {path} ({exc})")
                continue
            T = len(df)
            if T <= 0:
                warnings.append({
                    "artifact": "",
                    "source_path": str(path),
                    "stage": "load_log",
                    "warning": "empty converted dataframe",
                })
                print(f"  ⚠ 빈 데이터: {path}")
                continue
            io = np.zeros((T, len(io_cols)), dtype=np.float64)
            for j, col in enumerate(io_cols):
                if col in df.columns:
                    io[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
            sensors = np.zeros((T, len(required_sensors)), dtype=np.float64)
            for j, col in enumerate(required_sensors):
                if col in df.columns:
                    sensors[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
            step = None
            for col in df.columns:
                if str(col).strip().lower() in step_names:
                    step = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).to_numpy()
                    break
            records.append({
                "wafer_id": path.stem,
                "recipe": _infer_recipe(path, folder, cfg),
                "source_path": str(path),
                "timestamp": timestamp,
                "label": label or "BLIND",
                "io": io,
                "io_cols": io_cols,
                "sensors": sensors,
                "sensor_cols": required_sensors,
                "step": step,
            })
    return records


def _write_trial_prediction_time_plot(rows: list[dict], png_path: Path) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["timestamp"])
    xs = [r["timestamp"] for r in rows]
    ys = [float(r["anomaly_score"]) for r in rows]
    max_y = max([0.0] + [v for v in ys if np.isfinite(v)])
    y_top = max_y if max_y > 2.0 else 2.0
    fig, ax = plt.subplots(figsize=(max(9, 0.18 * len(rows) + 4), 5))
    preferred_markers = {
        "29a": "o",
        "336": "s",
        "521": "^",
    }
    marker_cycle = ["D", "v", "P", "X", "*", "<", ">", "h", "8"]
    recipe_markers: dict[str, str] = {}
    for row in rows:
        recipe = str(row.get("recipe", "") or "unknown")
        key = recipe.lower()
        if recipe in recipe_markers:
            continue
        if key in preferred_markers:
            recipe_markers[recipe] = preferred_markers[key]
        else:
            recipe_markers[recipe] = marker_cycle[len(recipe_markers) % len(marker_cycle)]
    for recipe, marker in recipe_markers.items():
        group = [r for r in rows if str(r.get("recipe", "") or "unknown") == recipe]
        gx = [r["timestamp"] for r in group]
        gy = [float(r["anomaly_score"]) for r in group]
        gc = ["#d62728" if float(v) >= 1.0 else "#1f77b4" for v in gy]
        ax.scatter(gx, gy, c=gc, marker=marker, s=32, alpha=0.85,
                   edgecolors="none", label=str(recipe))
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.75,
               label="threshold")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.set_ylim(0, y_top if y_top > 0 else 2.0)
    ax.set_xlabel("time")
    ax.set_ylabel("anomaly score")
    ax.set_title("예측값 시간순", fontsize=12)
    ax.yaxis.grid(True, alpha=0.2)
    ax.xaxis.grid(False)
    ax.legend(fontsize=8)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_trial_prediction_time_csv(rows: list[dict], csv_path: Path) -> None:
    fields = [
        "artifact", "timestamp", "date", "time", "wafer_id", "recipe", "label",
        "source_path", "raw_anomaly_score", "threshold_raw", "anomaly_score",
        "pred",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["timestamp"]):
            ts = row["timestamp"]
            out = dict(row)
            out["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            out["date"] = ts.strftime("%Y-%m-%d")
            out["time"] = ts.strftime("%H:%M")
            writer.writerow({k: out.get(k, "") for k in fields})


def _write_trial_prediction_warnings(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    fields = ["artifact", "source_path", "stage", "warning"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)


def _artifact_threshold_and_cfg(base_cfg: IOConfig, artifact_dir: Path):
    best, _source = _read_trial_artifact_best(artifact_dir)
    best_params = best.get("params") if isinstance(best.get("params"), dict) else {}
    fields = set(IOConfig().__dataclass_fields__)
    keep = {
        k: v for k, v in best_params.items()
        if k in fields and k not in {"ok_dir", "ng_dir", "val_dir", "artifacts_dir", "file_format"}
    }
    artifact_cfg = replace(base_cfg, **keep)
    artifact_cfg.artifacts_dir = Path(artifact_dir)
    artifact_cfg.file_format = base_cfg.file_format
    threshold = best.get("threshold", (best.get("val") or {}).get("threshold_raw"))
    score_kind = (
        best.get("score_name")
        or ((best.get("evaluation_criteria") or {}).get("threshold_field"))
        or ((best.get("val") or {}).get("score_kind"))
    )
    if score_kind is None and isinstance(threshold, dict):
        score_kind = threshold.get("score_kind")
    stored_threshold_wise = (best.get("val") or {}).get("threshold_wise")
    threshold_wise_from_threshold = (
        str(threshold.get("threshold_wise", "")).lower()
        if isinstance(threshold, dict) else ""
    )
    if score_kind is None and (
            str(stored_threshold_wise or "").lower() in {"all", "recipe", "sensor", "step"}
            or threshold_wise_from_threshold in {"all", "recipe", "sensor", "step"}):
        stored_mode = str(stored_threshold_wise or threshold_wise_from_threshold).lower()
        score_kind = (
            "step_accum_excess_reconstruction_error"
            if stored_mode in {"sensor", "step"}
            else "sum_all_sensor_reconstruction_error"
        )
    return artifact_cfg, threshold, score_kind, best


def cmd_plot_trial_artifact_predictions(cfg: IOConfig, out_dir: str | None = None,
                                        artifact_dir: str | None = None) -> None:
    from .model_tcn_forecast import TCNForecastDetector
    from .recon_explain import (
        _apply_score_agg_metadata_to_cfg,
        _display_score_threshold,
        _prediction_scores,
        _score_kind_from_cfg,
        _threshold_for_record,
        _threshold_with_explain_recipe_multipliers,
        _threshold_with_score_metadata,
        _uses_sum_reconstruction_threshold,
    )

    adir = Path(cfg.artifacts_dir)
    single_artifact_dir = _resolve_single_trial_artifact_dir(artifact_dir)
    artifact_dirs = [single_artifact_dir] if single_artifact_dir else _trial_artifact_dirs(adir)
    if not artifact_dirs:
        sys.exit(f"trial artifact 폴더 없음: {adir / 'artifacts0001'} 형식")
    if single_artifact_dir and not single_artifact_dir.is_dir():
        sys.exit(f"trial artifact 폴더 없음: {single_artifact_dir}")
    out = Path(out_dir) if out_dir else (adir / "trial_predictions")
    if not out.is_absolute():
        out = PKG_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    warnings: list[dict] = []
    ok_count = 0
    skipped_count = 0
    for artifact_dir in artifact_dirs:
        artifact = artifact_dir.name
        csv_path = out / f"{artifact}_예측값_시간순_원본.csv"
        png_path = out / f"{artifact}_예측값_시간순.png"
        if csv_path.exists() and png_path.exists():
            skipped_count += 1
            print(f"[{artifact}] existing prediction outputs found, skip: {png_path}")
            continue
        try:
            artifact_cfg, threshold, score_kind, best = _artifact_threshold_and_cfg(cfg, artifact_dir)
            if not (artifact_dir / "tcn_forecast.pt").exists():
                raise FileNotFoundError(str(artifact_dir / "tcn_forecast.pt"))
            val_payload = best.get("val") if isinstance(best.get("val"), dict) else {}
            eval_payload = (
                best.get("evaluation_criteria")
                if isinstance(best.get("evaluation_criteria"), dict) else {}
            )
            _apply_score_agg_metadata_to_cfg(artifact_cfg, threshold, val_payload, eval_payload, best)
            A = TCNForecastDetector.load(artifact_dir, artifact_cfg).A
            threshold_for_scores = (
                _threshold_with_score_metadata(threshold, getattr(A, "cfg", artifact_cfg))
                if _uses_sum_reconstruction_threshold(threshold, score_kind)
                else threshold
            )
            threshold_for_scores, _multipliers = _threshold_with_explain_recipe_multipliers(
                threshold_for_scores,
                tuple(getattr(cfg, "threshold_recipe_multipliers", ()) or ()),
            )
            score_kind_for_run = (
                _score_kind_from_cfg(getattr(A, "cfg", artifact_cfg))
                if _uses_sum_reconstruction_threshold(threshold, score_kind)
                else score_kind
            )
            load_warnings: list[dict] = []
            records = _load_prediction_records_for_model(cfg, A, load_warnings)
            for row in load_warnings:
                row["artifact"] = artifact
            warnings.extend(load_warnings)
            if not records:
                raise ValueError("예측 대상 log/csv 파일 없음")
            scores = _prediction_scores(
                A, records, threshold=threshold_for_scores,
                score_kind=score_kind_for_run,
            )
            rows = []
            for rec, raw_score in zip(records, scores):
                raw_score = float(raw_score)
                display_score, display_threshold = _display_score_threshold(
                    raw_score, threshold_for_scores, rec)
                raw_threshold = _threshold_for_record(threshold_for_scores, rec)
                pred = ""
                if raw_threshold is not None:
                    pred = "NG" if raw_score >= float(raw_threshold) else "OK"
                rows.append({
                    "artifact": artifact,
                    "timestamp": rec["timestamp"],
                    "wafer_id": rec["wafer_id"],
                    "recipe": rec.get("recipe", ""),
                    "label": rec.get("label", "BLIND"),
                    "source_path": rec.get("source_path", ""),
                    "raw_anomaly_score": f"{raw_score:.10g}",
                    "threshold_raw": "" if raw_threshold is None else f"{float(raw_threshold):.10g}",
                    "anomaly_score": float(display_score),
                    "pred": pred,
                })
            _write_trial_prediction_time_csv(rows, csv_path)
            _write_trial_prediction_time_plot(rows, png_path)
            ok_count += 1
            print(f"[{artifact}] prediction time plot saved: {png_path}")
        except Exception as exc:
            warnings.append({
                "artifact": artifact,
                "source_path": str(artifact_dir),
                "stage": "artifact",
                "warning": str(exc),
            })
            print(f"  ⚠ artifact 처리 실패: {artifact_dir} ({exc})")
            continue

    warn_csv = out / "예측값_시간순_경고.csv"
    _write_trial_prediction_warnings(warnings, warn_csv)
    if warnings:
        print(f"warnings saved: {warn_csv}")
    print(
        f"trial artifact prediction plots: "
        f"ok={ok_count}/{len(artifact_dirs)} skipped={skipped_count} | out={out}"
    )


def _require_h_cart_mode(ap, args) -> None:
    needs = {"tcn-search", "tcn-export", "recon-explain", "tcn-explain",
             "transformer-search", "transformer-export", "transformer-explain"}
    if args.cmd in needs and not args.h_cart_mode:
        ap.error("--h-cart-mode {include,exclude} 는 필수입니다. "
                 "Cart 계열 H 센서 포함/미포함 실험을 명시해야 합니다.")


def main():
    ap = argparse.ArgumentParser(description="IO-Recon-TCN OK-only future forecaster")
    ap.add_argument("cmd", choices=["tcn-search", "tcn-export", "plot-best-progress",
                                    "convert-onnx", "summarize-trial-artifacts",
                                    "plot-trial-artifact-predictions",
                                    "recon-explain", "tcn-explain",
                                    "transformer-search", "transformer-export",
                                    "transformer-explain"])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("-n-workers", "--n-workers", type=int, default=0)
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-trial-artifacts", action="store_true",
                    help=("tcn-search 성공 trial마다 artifacts0001, artifacts0002, ... "
                          "폴더에 독립 실행 가능한 artifact bundle 저장."))
    ap.add_argument("--onnx-opset", type=int, default=17,
                    help="convert-onnx에서 사용할 ONNX opset version.")
    ap.add_argument("--no-onnx-verify", action="store_true",
                    help="convert-onnx 후 onnxruntime 검증 생략.")
    ap.add_argument("--trial-summary-csv", default=None,
                    help=("summarize-trial-artifacts 출력 CSV 경로. "
                          "기본=--artifacts-dir/trial_artifacts_summary.csv"))
    ap.add_argument("--trial-pred-out-dir", default=None,
                    help=("plot-trial-artifact-predictions 출력 디렉토리. "
                          "기본=--artifacts-dir/trial_predictions"))
    ap.add_argument("--trial-artifact-dir", default=None,
                    help=("plot-trial-artifact-predictions에서 처리할 단일 artifact 폴더. "
                          "예: artifacts/artifacts0005"))
    ap.add_argument("--stop-on-perfect-val", action="store_true",
                    help="validation OK+NG를 전부 맞춘 trial이 나오면 search를 조기 종료.")
    ap.add_argument("--search-objective", choices=["correct", "precision"],
                    default="correct",
                    help=("tcn-search best 선정 기준. correct=1순위 맞춘 개수, 2순위 OK 맞춘 개수, "
                          "precision=1순위 precision, 2순위 맞춘 개수."))
    ap.add_argument("--post-eval-extend-epochs", type=int, default=10,
                    help=("tcn-search trial 기본 학습/early-stop 종료 후 VAL 판정 성능이 "
                          "개선되는지 확인할 추가 학습 epoch 단위."))
    ap.add_argument("--post-eval-max-extends", type=int, default=0,
                    help=("tcn-search trial별 post-eval 추가 학습 최대 반복 횟수. "
                          "0이면 기존처럼 추가 학습 없음."))
    ap.add_argument("--data-name", type=str, default=None,
                    help=("레시피별 데이터 폴더명. 예: 29a 또는 data/29a -> "
                          "OK=data/29a/OK, NG=data/29a/NG, VAL=data/29a/VAL"))
    ap.add_argument("--file-format", choices=["auto", "csv", "log"], default="auto",
                    help=("입력 wafer 파일 형식. auto=파일 확장자별 자동 판별(.csv/.log 혼합 가능), "
                          "csv=기존 CSV 파일만, log=탭 구분 .log 파일만 "
                          "WITH_IO=True, ONLY_HTP=True 규칙으로 메모리 변환해 사용."))
    ap.add_argument("--artifacts-name", type=str, default=None)
    ap.add_argument("--artifacts-dir", type=str, default=None)
    ap.add_argument("--report-name", type=str, default=None)
    ap.add_argument("--report-dir", type=str, default=None)
    ap.add_argument("--h-cart-mode", choices=["include", "exclude"], default=None,
                    help="필수. Cart 이름이 들어간 H sensor 입력 포함/미포함 ablation.")
    ap.add_argument("--h-sensor-preset",
                    choices=["all", "heater-block3", "heater_block3", "heater-12", "heater_12"],
                    default="all",
                    help=("H sensor 입력/복원 대상 preset. all=기존 전체 H, "
                          "heater-block3=h_005 계열 3개만, "
                          "heater-12=리저버/베셀/히터블럭 12개만."))
    ap.add_argument("--sensor-role-profile",
                    choices=["original", "h-split-v1", "h_split_v1"],
                    default="original",
                    help=("센서 입출력 역할 프로파일. original=기존 방식. "
                          "h-split-v1=미래 입력은 IO+h_005 3종+Cart 10종, "
                          "복원 대상은 P/T+h_010+Vessel heater 8종."))
    ap.add_argument("--cylinder-mode", choices=["off", "input", "predict"], default=None,
                    help="실린더 센서 역할: 입력 전용 또는 예측 대상")
    ap.add_argument("--cylinder-diff", action="store_true", default=None,
                    help="raw 오른쪽-왼쪽 v_011 생성")
    ap.add_argument("--future-h-mode", choices=["input", "predict"], default="input",
                    help=("input=현재 방식: 미래 H를 입력으로 사용하고 T/P만 예측. "
                          "predict=미래 H를 입력하지 않고 T/P/H를 예측."))
    ap.add_argument("--virtual-sensor-mode", choices=["off", "input", "target", "both"],
                    default="off",
                    help=("가상 센서 사용 모드. off=사용 안함. 활성화 시 10개 가상 센서는 항상 "
                          "예측 target에 포함되며, input/both는 과거 입력에도 포함."))
    ap.add_argument("--virtual-sensors", action="store_true",
                    help="가상 센서를 input+target 모두에 사용. --virtual-sensor-mode both와 동일.")
    ap.add_argument("--topk", type=int, default=None,
                    help=("reconstruction error score에 포함할 상위 센서 수. "
                          "기본=0(전체 센서), 5=상위 5개 센서만, 10=상위 10개 센서만."))
    ap.add_argument("--resid-agg", choices=["sum", "p99", "max", "mean"],
                    default=None,
                    help=("센서별 residual 시간축 집계 방식. 지정하지 않으면 현재 방식(sum)을 유지."))
    ap.add_argument("--wafer-agg", choices=["sum", "topk_mean", "max", "mean"],
                    default=None,
                    help=("센서별 score를 wafer score로 합치는 방식. 지정하지 않으면 현재 방식(sum)을 유지."))
    ap.add_argument("--score-time-avg", action="store_true",
                    help=("누적 reconstruction score를 유효 time-step 수로 나눠 평가/시각화. "
                          "긴 recipe의 누적 길이 영향을 줄임."))
    ap.add_argument("--score-final-avg", action="store_true",
                    help=("누적 reconstruction score를 wafer 전체 유효 time-step 수로 나눠 "
                          "평가/시각화. 초반 그래프 과대 표시를 줄임."))
    ap.add_argument("--ng-only", action="store_true")
    ap.add_argument("--prediction-order", choices=["score", "recipe"], default="score")
    ap.add_argument("--pred-batch", type=int, default=1,
                    help=("tcn-explain prediction score 계산 시 GPU forward에 한 번에 넣을 "
                          "forecast window batch 크기. 1=저장된 tcn_batch 사용."))
    ap.add_argument("--raw-error-level", type=float, default=None,
                    help="recon-explain 음영 기준 배수. 기본=1.0, 즉 OK residual q 기준 초과.")
    ap.add_argument("--threshold-wise", choices=["all", "recipe", "sensor", "step"],
                    default="recipe",
                    help=("threshold 지정 단위. all=전체 공통, recipe=레시피별, "
                          "sensor=레시피×센서별, step=레시피×센서×ProcessStep별."))
    ap.add_argument("--threshold-source", choices=["val-search", "train-ok-sigma"],
                    default="val-search",
                    help=("threshold 산정 기준. val-search=기존처럼 validation 성능 기준 탐색, "
                          "train-ok-sigma=train OK wafer final score 평균+kσ로 고정 후 val 성능 평가."))
    ap.add_argument("--threshold-sigma", type=float, default=-1.0,
                    help=("--threshold-source train-ok-sigma에서 사용할 k값. "
                          "기본=-1(auto: validation 성능 기준으로 공통 k 탐색)."))
    ap.add_argument("--threshold-sigma-min", type=float, default=None,
                    help=("--threshold-sigma -1 자동 탐색 시 사용할 sigma 하한. "
                          "고정 sigma 값과 함께 쓰면 에러."))
    ap.add_argument("--threshold-sigma-max", type=float, default=None,
                    help=("--threshold-sigma -1 자동 탐색 시 사용할 sigma 상한. "
                          "고정 sigma 값과 함께 쓰면 에러."))
    ap.add_argument("--threshold-recipe-multiplier", "--threshold-recipe-scale",
                    action="append", default=None, metavar="RECIPE=FACTOR",
                    help=("recipe별 threshold 배율. 여러 번 지정하거나 comma로 연결 가능. "
                          "예: --threshold-recipe-multiplier 29a=1.1"))
    ap.add_argument("--interpolation", type=int, default=1,
                    help=("recon/tcn-explain 출력 해상도 배율. 기본=1. "
                          "2면 time resolution 2배, 3이면 3배로 센서/예측/RCA를 보간."))
    ap.add_argument("--suppress-isolated-step-transition-shading", "-ss",
                    action="store_true",
                    help=("tcn-explain PNG에서 ProcessStep 경계 전후에 고립된 "
                          "4σ 적색 음영만 표시하지 않음. 점수/CSV/판정은 유지."))
    ap.add_argument("--step-transition-shading-window", "-sw", type=int, default=2,
                    help=("고립 step 경계 적색 음영 제거 후보 범위. "
                          "ProcessStep 경계 전후 N point 안의 peak를 후보로 봄."))
    ap.add_argument("--step-transition-context-window", "-sc", type=int, default=8,
                    help=("고립 step 경계 적색 음영 제거 주변 확인 범위. "
                          "전후 N point 안에서 경계 후보 밖 4σ 초과가 있으면 음영 유지."))
    ap.add_argument("--smooth-step-boundary-prediction", "-sp",
                    action="store_true",
                    help=("tcn-explain PNG 상단 그래프에서 ProcessStep 경계 주변 "
                          "예측선 spike를 표시용으로 완화. 원본 예측/점수/CSV는 유지."))
    ap.add_argument("--step-boundary-smooth-window", "-bw", type=int, default=2,
                    help=("ProcessStep 경계 예측선 표시 보정 범위. "
                          "경계 전후 N point를 직전/직후 값으로 연결."))
    ap.add_argument("--error-accum-mode", choices=["sensor", "step"], default="sensor",
                    help=("tcn-explain 웨이퍼별 센서 그래프 하단 누적 방식. "
                          "sensor=센서별 전체 시간축 누적(기본, step 변경 시 리셋 없음), "
                          "step=ProcessStep별 누적(기존 방식, step 변경 시 리셋)."))
    ap.add_argument("--ok-mean-exclude-over-threshold",
                    dest="ok_mean_exclude_over_threshold",
                    action="store_true", default=True,
                    help=("tcn/recon-explain에서 같은 recipe train OK 평균/σ 계산 시 "
                          "threshold 이상 anomaly score를 가진 train OK wafer를 제외."))
    ap.add_argument("--ok-mean-include-over-threshold",
                    dest="ok_mean_exclude_over_threshold",
                    action="store_false",
                    help=("tcn/recon-explain에서 같은 recipe train OK 평균/σ 계산 시 "
                          "threshold 이상 train OK wafer도 포함."))
    args = ap.parse_args()
    _require_h_cart_mode(ap, args)

    cfg = IOConfig()
    if args.h_cart_mode:
        cfg.h_cart_mode = args.h_cart_mode
    cfg.file_format = args.file_format
    cfg.h_sensor_preset = normalize_h_sensor_preset(args.h_sensor_preset)
    cfg.sensor_role_profile = normalize_sensor_role_profile(args.sensor_role_profile)
    if cfg.sensor_role_profile == SENSOR_ROLE_PROFILE_H_SPLIT_V1:
        if cfg.h_cart_mode == "exclude":
            ap.error("--sensor-role-profile h-split-v1은 Cart류 10종을 미래 입력으로 "
                     "사용하므로 --h-cart-mode exclude와 함께 사용할 수 없습니다.")
        if cfg.h_sensor_preset:
            ap.error("--sensor-role-profile h-split-v1은 H 역할을 직접 나누므로 "
                     "--h-sensor-preset all과 함께 사용해야 합니다.")
        if args.future_h_mode != "input":
            ap.error("--sensor-role-profile h-split-v1은 미래 H 입력 프로파일이므로 "
                     "--future-h-mode input과 함께 사용해야 합니다.")
    if cfg.h_sensor_preset:
        cfg.h_sensor_keep = tuple(h_SENSOR_PRESETS[cfg.h_sensor_preset])
    cfg.cylinder_mode = args.cylinder_mode or "off"
    cfg.cylinder_diff = bool(args.cylinder_diff)
    cfg.cylinder_mode_request = args.cylinder_mode
    cfg.cylinder_diff_request = args.cylinder_diff
    if cfg.cylinder_diff and args.cmd == "tcn-search" and cfg.cylinder_mode == "off":
        ap.error("--cylinder-diff는 --cylinder-mode input 또는 predict가 필요합니다.")
    cfg.future_h_mode = args.future_h_mode
    cfg.virtual_sensor_mode = "both" if args.virtual_sensors else args.virtual_sensor_mode
    if args.raw_error_level is not None:
        cfg.raw_error_level = float(args.raw_error_level)
    cfg.threshold_wise = args.threshold_wise
    cfg.threshold_source = args.threshold_source
    cfg.threshold_sigma = float(args.threshold_sigma)
    cfg.threshold_sigma_min = args.threshold_sigma_min
    cfg.threshold_sigma_max = args.threshold_sigma_max
    cfg.wafer_topk = max(0, int(args.topk if args.topk is not None else 0))
    if args.score_time_avg and args.score_final_avg:
        ap.error("--score-time-avg와 --score-final-avg는 동시에 사용할 수 없습니다.")
    cfg.score_time_avg = bool(args.score_time_avg)
    cfg.score_final_avg = bool(args.score_final_avg)
    if args.resid_agg is not None or args.wafer_agg is not None:
        cfg.score_agg_enabled = True
        cfg.resid_agg = str(args.resid_agg or "sum")
        cfg.wafer_agg = str(args.wafer_agg or "sum")
    else:
        cfg.score_agg_enabled = False
    cfg.search_objective = args.search_objective
    cfg.post_eval_extend_epochs = max(0, int(args.post_eval_extend_epochs))
    cfg.post_eval_max_extends = max(0, int(args.post_eval_max_extends))
    cfg.pred_batch = (
        max(1, int(args.pred_batch))
        if args.cmd in ("recon-explain", "tcn-explain", "transformer-explain")
        else 1
    )
    try:
        cfg.threshold_recipe_multipliers = _parse_threshold_recipe_multipliers(
            args.threshold_recipe_multiplier)
    except ValueError as e:
        ap.error(str(e))
    cfg.explain_interpolation = max(1, int(args.interpolation))
    cfg.suppress_isolated_step_transition_shading = bool(
        args.suppress_isolated_step_transition_shading)
    cfg.step_transition_shading_window = max(0, int(args.step_transition_shading_window))
    cfg.step_transition_context_window = max(0, int(args.step_transition_context_window))
    cfg.smooth_step_boundary_prediction = bool(args.smooth_step_boundary_prediction)
    cfg.step_boundary_smooth_window = max(0, int(args.step_boundary_smooth_window))
    cfg.error_accum_mode = args.error_accum_mode
    _apply_cli_paths(cfg, data_name=args.data_name,
                     artifacts_name=args.artifacts_name,
                     artifacts_dir=args.artifacts_dir)
    cfg.ensure_dirs()

    if args.cmd in ("tcn-search", "transformer-search"):
        cmd_tcn_search(cfg, trials=args.trials, seed=args.seed,
                       fixed_epochs=args.epochs,
                       train_best=not args.no_train,
                       n_workers=args.n_workers,
                       checkpoint=not args.no_checkpoint,
                       resume=args.resume,
                       stop_on_perfect_val=args.stop_on_perfect_val,
                       save_trial_artifacts=args.save_trial_artifacts)
        return
    if args.cmd in ("tcn-export", "transformer-export"):
        report_dir = _resolve_report_dir(cfg, args.report_name, args.report_dir)
        cmd_tcn_export(cfg, report_dir=report_dir)
        return
    if args.cmd == "plot-best-progress":
        cmd_plot_best_progress(cfg)
        return
    if args.cmd == "convert-onnx":
        cmd_convert_onnx(cfg, opset=args.onnx_opset,
                         verify_export=not args.no_onnx_verify)
        return
    if args.cmd == "summarize-trial-artifacts":
        cmd_summarize_trial_artifacts(cfg, out_csv=args.trial_summary_csv)
        return
    if args.cmd == "plot-trial-artifact-predictions":
        cmd_plot_trial_artifact_predictions(
            cfg, out_dir=args.trial_pred_out_dir,
            artifact_dir=args.trial_artifact_dir)
        return
    if args.cmd in ("recon-explain", "tcn-explain", "transformer-explain"):
        from .recon_explain import cmd_recon_explain
        report_dir = _resolve_fresh_report_dir(cfg, args.report_name or "recon", args.report_dir)
        _write_command_record(
            report_dir, "tcn_explain_command.txt", "tcn-explain",
            artifacts_dir=cfg.artifacts_dir, report_dir=report_dir)
        cmd_recon_explain(cfg, top_k=args.topk, ng_only=args.ng_only,
                          report_dir=report_dir,
                          prediction_order=args.prediction_order,
                          ok_mean_exclude_over_threshold=args.ok_mean_exclude_over_threshold)
        return


if __name__ == "__main__":
    main()
