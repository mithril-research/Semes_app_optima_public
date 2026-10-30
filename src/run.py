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
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .config import IOConfig
from .data import load_dataset


PKG_ROOT = Path(__file__).resolve().parent
SEARCH_SPACE_PATH = PKG_ROOT / "search_space.json"
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
    return {"correct": tp + tn, "n": n_ok + n_ng, "tp": tp, "tn": tn,
            "fn": fn, "fp": fp, "n_ok": n_ok, "n_ng": n_ng}


def _safe_auroc(y, p):
    try:
        from sklearn.metrics import roc_auc_score
        return round(float(roc_auc_score(y, p)), 4)
    except Exception:
        return 0.0


def _correct_from_scores(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    best = None
    for th in _threshold_candidates_midpoint(p):
        m = _metrics_at_threshold(y, p, th)
        cand = (m["correct"], m["tp"], float(th), m)
        if best is None or cand[:3] > best[:3]:
            best = cand
    _, _, th, metrics = best
    return {**metrics, "threshold": round(float(th), 4),
            "threshold_raw": float(th), "auroc": _safe_auroc(y, p)}


def _correct_from_nonnegative_scores(y, p):
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
        cand = (m["correct"], m["tp"], float(th), m)
        if best is None or cand[:3] > best[:3]:
            best = cand
    _, _, th, metrics = best
    return {**metrics, "threshold": round(float(th), 4),
            "threshold_raw": float(th), "auroc": _safe_auroc(y, p)}


def _recipe_key(rec) -> str:
    value = rec.get("recipe", "unknown") if isinstance(rec, dict) else "unknown"
    return str(value or "unknown")


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
    if not multipliers:
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
    }
    if score_kind:
        out["score_kind"] = str(score_kind)
    if strategy:
        out["threshold_strategy"] = str(strategy)
    return out


def _correct_from_nonnegative_scores_all(records, y, p, cfg: IOConfig):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    vc = _correct_from_nonnegative_scores(y, p)
    if vc is None:
        return None
    base_th = float(vc["threshold_raw"])
    threshold_obj = _threshold_object_from_base_with_recipe_multipliers(
        base_th, cfg, records=records,
        score_kind="sum_all_sensor_reconstruction_error",
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
        "score_kind": "sum_all_sensor_reconstruction_error",
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
    global_vc = _correct_from_scores(y, p)
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
            vc = _correct_from_scores(yy, pp)
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
    global_vc = _correct_from_nonnegative_scores(y, p)
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
            vc = _correct_from_nonnegative_scores(yy, pp)
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
    }
    if multipliers:
        out["threshold_recipe_multipliers"] = multipliers
        out["threshold_strategy"] += "_with_recipe_threshold_multipliers"
    return out


def _threshold_wise(cfg: IOConfig) -> str:
    mode = str(getattr(cfg, "threshold_wise", "recipe") or "recipe").lower()
    return mode if mode in {"all", "recipe", "sensor", "step"} else "recipe"


def _stored_threshold_wise(threshold) -> str | None:
    if isinstance(threshold, dict) and threshold.get("threshold_wise"):
        return str(threshold.get("threshold_wise")).lower()
    return None


def _saved_threshold_for_record(threshold, rec):
    if threshold is None:
        return None
    if not isinstance(threshold, dict):
        return float(threshold)
    mode = str(threshold.get("threshold_wise", "")).lower()
    if mode in {"sensor", "step"}:
        return float(threshold.get("decision_threshold", 1.0))
    recipe = _recipe_key(rec)
    for key in ("threshold_by_recipe", "threshold_raw", "threshold"):
        table = threshold.get(key)
        if isinstance(table, dict):
            if recipe in table:
                return float(table[recipe])
            wanted = _recipe_lookup_key(recipe)
            for k, v in table.items():
                if _recipe_lookup_key(k) == wanted:
                    return float(v)
    for key in ("global_threshold_raw", "global_threshold", "threshold_raw",
                "threshold", "default", "global"):
        value = threshold.get(key)
        if value is not None and not isinstance(value, dict):
            return float(value)
    return None


def _threshold_score_name() -> str:
    return "sum_all_sensor_reconstruction_error"


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
    """Wafer score: sum normalized reconstruction error over every sensor/time."""
    level = np.asarray(model_a.error_level(rec), dtype=np.float64)
    if level.ndim != 2 or level.size == 0:
        return 0.0
    level = np.nan_to_num(level, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.maximum(0.0, level).sum())


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
    current_obj = (current["correct"], current["tp"], -current["fp"])

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
                obj = (m["correct"], m["tp"], -m["fp"])
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


def _evaluate_thresholds(records, y, scalar_scores, model_a, cfg: IOConfig):
    mode = _threshold_wise(cfg)
    if mode == "all":
        step_scores = np.asarray([_sum_all_sensor_reconstruction_score(model_a, r) for r in records],
                                 dtype=np.float64)
        vc = _correct_from_nonnegative_scores_all(records, y, step_scores, cfg)
        if vc is not None:
            vc["threshold_wise"] = "all"
            vc["score_kind"] = "sum_all_sensor_reconstruction_error"
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
            vc["score_kind"] = "sum_all_sensor_reconstruction_error"
        return vc, step_scores
    return _correct_from_step_accum_thresholds(records, y, model_a, cfg, mode)


def _tcn_search_space():
    return _load_tcn_search_space()


def _receptive(kernel, blocks):
    return 2 * (kernel - 1) * (2 ** blocks - 1) + 1


def _tcn_sample_params(trials: int, seed: int, fixed_epochs: int = None):
    import random
    space = _tcn_search_space()
    if fixed_epochs is not None:
        space["tcn_epochs"] = [int(fixed_epochs)]
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


def _search_result_with_metadata(r, score_name: str, model_kind: str) -> dict:
    out = dict(r)
    val = out.get("val", {})
    scores = out.get("scores", {})
    out["score_name"] = score_name
    out["model_kind"] = model_kind
    out["anomaly_scores"] = scores
    out["evaluation_criteria"] = {
        "primary_metric": "correct_wafers",
        "primary_metric_value": val.get("correct"),
        "trial_idx": out.get("trial_idx"),
        "threshold": out.get("threshold", val.get("threshold_raw", val.get("threshold"))),
        "threshold_field": score_name,
        "score_direction": "higher_is_more_anomalous",
        "n_wafers": val.get("n"),
        "n_ok": val.get("n_ok"),
        "n_ng": val.get("n_ng"),
        "tp": val.get("tp"),
        "tn": val.get("tn"),
        "fp": val.get("fp"),
        "fn": val.get("fn"),
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
        "forecast_val_mse": best.get("forecast_val_mse"),
        "correct_wafers": val.get("correct"),
        "n_wafers": val.get("n"),
        "tp": val.get("tp"),
        "tn": val.get("tn"),
        "fp": val.get("fp"),
        "fn": val.get("fn"),
        "n_ok": val.get("n_ok"),
        "n_ng": val.get("n_ng"),
        "auroc": val.get("auroc"),
        "threshold": _json_cell(best.get("threshold", val.get("threshold_raw", val.get("threshold")))),
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
        "forecast_val_mse": r.get("forecast_val_mse", ""),
        "trial_tp_plus_tf": val.get("correct", ""),
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
        "best_trial_idx": cur_best.get("trial_idx", "") if isinstance(cur_best, dict) else "",
        "best_tp_plus_tf": best_val.get("correct", ""),
        "best_tp": best_val.get("tp", ""),
        "best_tf": best_val.get("tn", ""),
        "best_tn": best_val.get("tn", ""),
        "best_fp": best_val.get("fp", ""),
        "best_fn": best_val.get("fn", ""),
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


def _result_obj(r: dict) -> list:
    val = r.get("val", {}) if isinstance(r, dict) else {}
    mse = r.get("forecast_val_mse", float("inf")) if isinstance(r, dict) else float("inf")
    try:
        mse_val = float(mse)
    except Exception:
        mse_val = float("inf")
    return [int(val.get("correct", 0) or 0), int(val.get("tp", 0) or 0),
            -mse_val]


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


def _load_resume_best(adir: Path, params_all: list[dict], model_kind: str) -> dict | None:
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
        if not any(p == params for p in params_all):
            print(f"  [resume] {path.name}의 best params가 현재 search space와 달라 무시")
            continue
        best["obj"] = _result_obj(best)
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
    val_mse = float(getattr(h.A, "val_mse", float("nan")))
    eval_recs = list(S["val_ok"]) + list(S["ng"])
    y = np.concatenate([np.zeros(len(S["val_ok"])), np.ones(len(S["ng"]))])
    vc, decision_scores = _evaluate_thresholds(eval_recs, y, None, h.A, cfg)
    if vc is None:
        return None
    ids = [r["wafer_id"] for r in eval_recs]
    scores = {w: float(pp) for w, pp in zip(ids, decision_scores)}
    checkpoint_path = None
    if trial_checkpoint_dir is not None and trial_idx is not None:
        checkpoint_path = Path(trial_checkpoint_dir) / f"trial_{int(trial_idx):06d}.pt"
        h.A.save(checkpoint_path)
    return {"params": params,
            "obj": [vc["correct"], vc["tp"], -val_mse],
            "forecast_val_mse": val_mse,
            "val": vc,
            "receptive": _receptive(params["tcn_kernel"], params["tcn_blocks"]),
            "need": int(params["tcn_past_window"] + params["tcn_future_window"]),
            "scores": scores,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "threshold": vc.get("threshold_raw", vc.get("threshold"))}


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
                item = task_q.get(timeout=1.0)
            except queue.Empty:
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
    return (f"  [{i:>3}/{total}] val_mse={r['forecast_val_mse']:.6f} "
            f"(trial {r.get('trial_idx', i)}) | threshold 맞춤 {vc['correct']}/{vc['n']} "
            f"(양품 {vc['tn']}/{vc['n_ok']} 불량 {vc['tp']}/{vc['n_ng']}) "
            f"AUROC={vc['auroc']:.3f} | w{p['tcn_past_window']} h{p['tcn_future_window']} "
            f"s{p['tcn_stride']} rf{r['receptive']}/{r['need']} {rf_ok} "
            f"ch{p['tcn_channels']} blk{p['tcn_blocks']} k{p['tcn_kernel']} "
            f"q{p['raw_error_quantile']}{worker}")


def _fmt_tcn_best(cur, is_new):
    v = cur["val"]
    return (f"        -> current best: 맞춤 {v['correct']}/{v['n']} "
            f"(양품 {v['tn']}/{v['n_ok']} 불량 {v['tp']}/{v['n_ng']}) "
            f"| val_mse={cur['forecast_val_mse']:.6f}"
            + (" *NEW best" if is_new else ""))


def _is_full_validation_correct(r: dict) -> bool:
    val = r.get("val", {}) if isinstance(r, dict) else {}
    return bool(val and val.get("n") and val.get("correct") == val.get("n"))


def _save_tcn_checkpoint(cfg, best_params, S):
    import contextlib
    import io as _io
    from dataclasses import replace
    from .model_tcn_forecast import TCNForecastDetector
    bcfg = replace(cfg, **best_params)
    with contextlib.redirect_stdout(_io.StringIO()):
        h = TCNForecastDetector(bcfg).fit(
            S["train_ok"], S["io_cols"], S["sensor_cols"], S["nstep"],
            val_records=S["val_ok"])
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
                   stop_on_perfect_val: bool = False) -> None:
    print(f"data dirs: OK={cfg.ok_dir} | NG={cfg.ng_dir} | VAL={cfg.val_dir}")
    print(f"artifacts dir: {cfg.artifacts_dir}")
    print(f"h_cart_mode: {cfg.h_cart_mode}")
    print(f"future_h_mode: {cfg.future_h_mode}")
    print(f"virtual_sensor_mode: {getattr(cfg, 'virtual_sensor_mode', 'off')}")
    print(f"threshold_wise: {_threshold_wise(cfg)}")
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

    params_all = _tcn_sample_params(trials, seed, fixed_epochs)
    total = len(params_all)
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
    print("목표: validation OK+NG 맞춘 wafer 수 최대화")
    print("threshold-wise: all=전체 1개, recipe=레시피별, "
          "sensor=레시피×센서별, step=레시피×센서×스텝별")
    print(f"perfect validation 조기 종료: {'ON' if stop_on_perfect_val else 'OFF'}")
    print("동률 기준: 불량 TP -> OK forecast MSE 낮은 모델")
    print("ProcessStep 0: 입력에는 사용, loss/score/report/explain 강조 계산에서는 제외")
    print("threshold score: step별 Σ max(0, |actual-expected| / OK기준오차 - 1)")
    print("raw residual 기준: OK residual 분위수 초과분을 사용하며 std 나눗셈은 사용하지 않음")
    print(f"split: {S['split_mode']}\n")

    adir = Path(cfg.artifacts_dir)
    trial_ckpt_dir = adir / "_trial_checkpoints"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_progress_csv = adir / "search_trial_progress.csv"
    best_progress_csv = adir / "search_best_progress.csv"
    best_progress_png = adir / "search_best_progress.png"
    print(f"trial 성능 누적 CSV: {trial_progress_csv}")
    print(f"best 갱신 CSV: {best_progress_csv}")
    print(f"best 갱신 그래프: {best_progress_png}\n")

    results = []
    state = {"ckpt": None, "best_events": 0}
    completed_trials = set()
    if resume:
        completed_trials = _read_completed_trials(trial_progress_csv, params_all)
        resume_best = _load_resume_best(adir, params_all, "tcn_forecast_ok_only")
        if resume_best:
            results.append(resume_best)
            if checkpoint and _checkpoint_ready(adir):
                state["ckpt"] = resume_best.get("params")
        state["best_events"] = _read_best_event_count(best_progress_csv)
        print(f"resume 활성화: 완료된 trial {len(completed_trials)}/{total}개 skip")
        if resume_best:
            v = resume_best.get("val", {})
            print(f"resume best 로드: trial {resume_best.get('trial_idx')} "
                  f"맞춤 {v.get('correct')}/{v.get('n')}")

    params_to_run = [(i, p) for i, p in enumerate(params_all, 1)
                     if i not in completed_trials]
    if resume and params_to_run:
        print(f"resume next trial: {params_to_run[0][0]}/{total}")
    if params_to_run and n_workers > len(params_to_run):
        print(f"  worker 수 조정: 남은 trial {len(params_to_run)}개라 "
              f"{n_workers} -> {len(params_to_run)}")
        n_workers = len(params_to_run)
    if requested_workers and n_gpu and requested_workers > len(visible_gpus):
        print(f"  주의: --n-workers {requested_workers} 요청했지만 visible GPU가 "
              f"{len(visible_gpus)}개라 worker는 {n_workers}개만 사용합니다.")
    if checkpoint:
        print(f"trial checkpoint dir: {trial_ckpt_dir}")

    def _report(r, idx):
        display_idx = r.get("trial_idx", idx) if isinstance(r, dict) else idx
        if r is None:
            cur_best = max(results, key=lambda x: tuple(x["obj"])) if results else {}
            _append_trial_progress_csv(trial_progress_csv, {}, cur_best,
                                       display_idx, total, run_id,
                                       _threshold_score_name(),
                                       "tcn_forecast_ok_only")
            print(f"  [{display_idx:>3}/{total}] 채점 불가(클래스 부족)")
            return False
        if "error" in r:
            cur_best = max(results, key=lambda x: tuple(x["obj"])) if results else {}
            _append_trial_progress_csv(trial_progress_csv, r, cur_best, display_idx, total, run_id,
                                       _threshold_score_name(),
                                       "tcn_forecast_ok_only")
            print(f"  [{display_idx:>3}/{total}] 실패: {r['error']}")
            return False
        results.append(r)
        print(_fmt_tcn_trial(display_idx, total, r))
        cur = max(results, key=lambda x: tuple(x["obj"]))
        is_new = cur is results[-1]
        _append_trial_progress_csv(trial_progress_csv, r, cur, display_idx, total, run_id,
                                   _threshold_score_name(),
                                   "tcn_forecast_ok_only")
        print(_fmt_tcn_best(cur, is_new))
        if is_new:
            state["best_events"] += 1
            try:
                _append_best_progress_csv(best_progress_csv,
                                          cur, state["best_events"],
                                          _threshold_score_name(),
                                          "tcn_forecast_ok_only")
                _plot_best_progress(best_progress_csv, best_progress_png)
            except Exception as e:
                print(f"        best progress 저장 실패(서치는 계속): {e}")
            try:
                _write_search_jsons(adir, ("search_best.json", "best_search.json"),
                                    cur, _threshold_score_name(),
                                    "tcn_forecast_ok_only")
            except Exception as e:
                print(f"        search_best 저장 실패(서치는 계속): {e}")
        if checkpoint and is_new:
            try:
                installed = _install_trial_checkpoint(adir, cur)
                if not installed:
                    print("        trial checkpoint 설치 실패: best params로 재학습 저장")
                    _save_tcn_checkpoint(cfg, cur["params"], S)
                state["ckpt"] = cur["params"]
                _write_search_jsons(adir, ("search_best.json", "best_search.json"),
                                    cur, _threshold_score_name(),
                                    "tcn_forecast_ok_only")
                print("        best checkpoint ready: "
                      "tcn_forecast.pt + search_best.json + progress.csv + progress.png")
            except Exception as e:
                print(f"        checkpoint 저장 실패(서치는 계속): {e}")
        if checkpoint:
            _cleanup_trial_checkpoint(r)
        if stop_on_perfect_val and _is_full_validation_correct(r):
            v = r["val"]
            print("        validation 전체 wafer 정답으로 search 조기 종료: "
                  f"{v['correct']}/{v['n']} "
                  f"(양품 {v['tn']}/{v['n_ok']} 불량 {v['tp']}/{v['n_ng']})")
            return True
        return False

    if n_workers <= 1:
        for i, params in params_to_run:
            try:
                r = _eval_tcn_params(
                    cfg, params, S, seed, trial_idx=i,
                    trial_checkpoint_dir=trial_ckpt_dir if checkpoint else None)
                if isinstance(r, dict):
                    r["trial_idx"] = i
            except Exception as e:
                r = {"error": str(e), "params": params, "trial_idx": i}
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
        for item in params_to_run:
            task_q.put(item)
        out_q = ctx.Queue()
        stop_event = ctx.Event()
        procs = []
        for w in range(n_workers):
            gpu = visible_gpus[w % len(visible_gpus)] if n_gpu else 0
            pr = ctx.Process(target=_tcn_search_worker,
                             args=(gpu, w, cfg, seed, task_q, out_q,
                                   per_worker, stop_event,
                                   trial_ckpt_dir if checkpoint else None))
            pr.start()
            procs.append(pr)
        print(f"  dynamic trial queue: pending={len(params_to_run)} | workers={n_workers} "
              f"| gpu assignment={[visible_gpus[w % len(visible_gpus)] if n_gpu else 'cpu' for w in range(n_workers)]}")
        done_pids, idx, early_stop = set(), 0, False
        while len(done_pids) < n_workers:
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
            if _report(msg, idx):
                early_stop = True
                stop_event.set()
                break
        if early_stop:
            for pr in procs:
                if pr.is_alive():
                    pr.terminate()
        for pr in procs:
            pr.join()

    if not results:
        sys.exit("성공한 TCN trial 없음.")
    results.sort(key=lambda r: tuple(r["obj"]), reverse=True)
    best = results[0]
    print("\n=== TCN TOP 5 (correct wafers 기준) ===")
    for r in results[:5]:
        v = r["val"]
        print(f"  val_mse={r['forecast_val_mse']:.6f} "
              f"| threshold 맞춤 {v['correct']}/{v['n']} "
              f"(양품 {v['tn']}/{v['n_ok']} 불량 {v['tp']}/{v['n_ng']}) "
              f"AUROC={v['auroc']:.3f} | {r['params']}")
    print("\n=== BEST ===")
    v = best["val"]
    print(f"  threshold validation: {v['correct']}/{v['n']} "
          f"(양품 {v['tn']}/{v['n_ok']} 불량 {v['tp']}/{v['n_ng']}) "
          f"threshold={best['threshold']} AUROC={v['auroc']:.3f}")
    print(f"  forecast val MSE: {best['forecast_val_mse']:.6f}")
    print("  config.py 반영 후보:")
    for k, val in best["params"].items():
        print(f"      {k:26s}= {val!r}")
    _write_search_jsons(adir, ("search_best.json", "best_search.json"),
                        best, _threshold_score_name(),
                        "tcn_forecast_ok_only")
    print(f"\n  search best 저장: {adir / 'search_best.json'} (+ best_search.json)")
    if not train_best:
        print("  (--no-train: 최종 best 학습 생략)")
        return
    if checkpoint and state["ckpt"] == best["params"]:
        print(f"  최종 best는 이미 checkpoint로 저장됨: {adir / 'tcn_forecast.pt'}")
        return
    _save_tcn_checkpoint(cfg, best["params"], S)
    print(f"  저장: {adir / 'tcn_forecast.pt'}")


def cmd_tcn_export(cfg: IOConfig, report_dir: Path = None) -> None:
    from .export_report import build_report, run_existing_rca
    from .model_tcn_forecast import TCNForecastDetector
    adir = Path(cfg.artifacts_dir)
    print(f"data dirs: OK={cfg.ok_dir} | NG={cfg.ng_dir} | VAL={cfg.val_dir}")
    print(f"artifacts dir: {adir}")
    print(f"h_cart_mode: {cfg.h_cart_mode}")
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
                        v = _correct_from_scores(y, p)
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


def _require_h_cart_mode(ap, args) -> None:
    needs = {"tcn-search", "tcn-export", "recon-explain", "tcn-explain",
             "transformer-search", "transformer-export", "transformer-explain"}
    if args.cmd in needs and not args.h_cart_mode:
        ap.error("--h-cart-mode {include,exclude} 는 필수입니다. "
                 "Cart 계열 H 센서 포함/미포함 실험을 명시해야 합니다.")


def main():
    ap = argparse.ArgumentParser(description="IO-Recon-TCN OK-only future forecaster")
    ap.add_argument("cmd", choices=["tcn-search", "tcn-export", "plot-best-progress",
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
    ap.add_argument("--stop-on-perfect-val", action="store_true",
                    help="validation OK+NG를 전부 맞춘 trial이 나오면 search를 조기 종료.")
    ap.add_argument("--data-name", type=str, default=None,
                    help=("레시피별 데이터 폴더명. 예: 29a 또는 data/29a -> "
                          "OK=data/29a/OK, NG=data/29a/NG, VAL=data/29a/VAL"))
    ap.add_argument("--artifacts-name", type=str, default=None)
    ap.add_argument("--artifacts-dir", type=str, default=None)
    ap.add_argument("--report-name", type=str, default=None)
    ap.add_argument("--report-dir", type=str, default=None)
    ap.add_argument("--h-cart-mode", choices=["include", "exclude"], default=None,
                    help="필수. Cart 이름이 들어간 H sensor 입력 포함/미포함 ablation.")
    ap.add_argument("--future-h-mode", choices=["input", "predict"], default="input",
                    help=("input=현재 방식: 미래 H를 입력으로 사용하고 T/P만 예측. "
                          "predict=미래 H를 입력하지 않고 T/P/H를 예측."))
    ap.add_argument("--virtual-sensor-mode", choices=["off", "input", "target", "both"],
                    default="off",
                    help=("가상 센서 사용 모드. off=사용 안함. 활성화 시 10개 가상 센서는 항상 "
                          "예측 target에 포함되며, input/both는 과거 입력에도 포함."))
    ap.add_argument("--virtual-sensors", action="store_true",
                    help="가상 센서를 input+target 모두에 사용. --virtual-sensor-mode both와 동일.")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--ng-only", action="store_true")
    ap.add_argument("--prediction-order", choices=["score", "recipe"], default="score")
    ap.add_argument("--raw-error-level", type=float, default=None,
                    help="recon-explain 음영 기준 배수. 기본=1.0, 즉 OK residual q 기준 초과.")
    ap.add_argument("--threshold-wise", choices=["all", "recipe", "sensor", "step"],
                    default="recipe",
                    help=("threshold 지정 단위. all=전체 공통, recipe=레시피별, "
                          "sensor=레시피×센서별, step=레시피×센서×ProcessStep별."))
    ap.add_argument("--threshold-recipe-multiplier", "--threshold-recipe-scale",
                    action="append", default=None, metavar="RECIPE=FACTOR",
                    help=("recipe별 threshold 배율. 여러 번 지정하거나 comma로 연결 가능. "
                          "예: --threshold-recipe-multiplier 29a=1.1"))
    ap.add_argument("--interpolation", type=int, default=1,
                    help=("recon/tcn-explain 출력 해상도 배율. 기본=1. "
                          "2면 time resolution 2배, 3이면 3배로 센서/예측/RCA를 보간."))
    ap.add_argument("--error-accum-mode", choices=["sensor", "step"], default="sensor",
                    help=("tcn-explain 웨이퍼별 센서 그래프 하단 누적 방식. "
                          "sensor=센서별 전체 시간축 누적(기본, step 변경 시 리셋 없음), "
                          "step=ProcessStep별 누적(기존 방식, step 변경 시 리셋)."))
    ap.add_argument("--ok-mean-exclude-over-threshold", action="store_true",
                    help=("tcn/recon-explain에서 같은 recipe train OK 평균/σ 계산 시 "
                          "threshold 이상 anomaly score를 가진 train OK wafer를 제외."))
    args = ap.parse_args()
    _require_h_cart_mode(ap, args)

    cfg = IOConfig()
    if args.h_cart_mode:
        cfg.h_cart_mode = args.h_cart_mode
    cfg.future_h_mode = args.future_h_mode
    cfg.virtual_sensor_mode = "both" if args.virtual_sensors else args.virtual_sensor_mode
    if args.raw_error_level is not None:
        cfg.raw_error_level = float(args.raw_error_level)
    cfg.threshold_wise = args.threshold_wise
    try:
        cfg.threshold_recipe_multipliers = _parse_threshold_recipe_multipliers(
            args.threshold_recipe_multiplier)
    except ValueError as e:
        ap.error(str(e))
    cfg.explain_interpolation = max(1, int(args.interpolation))
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
                       stop_on_perfect_val=args.stop_on_perfect_val)
        return
    if args.cmd in ("tcn-export", "transformer-export"):
        report_dir = _resolve_report_dir(cfg, args.report_name, args.report_dir)
        cmd_tcn_export(cfg, report_dir=report_dir)
        return
    if args.cmd == "plot-best-progress":
        cmd_plot_best_progress(cfg)
        return
    if args.cmd in ("recon-explain", "tcn-explain", "transformer-explain"):
        from .recon_explain import cmd_recon_explain
        report_dir = _resolve_report_dir(cfg, args.report_name or "recon", args.report_dir)
        cmd_recon_explain(cfg, top_k=args.topk, ng_only=args.ng_only,
                          report_dir=report_dir,
                          prediction_order=args.prediction_order,
                          ok_mean_exclude_over_threshold=args.ok_mean_exclude_over_threshold)
        return


if __name__ == "__main__":
    main()
