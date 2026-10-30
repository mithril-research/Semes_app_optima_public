"""Aggregate hyperparameter search results from artifacts* folders.

Supported best files:
  - search_best.json / best_search.json: this repo's format
  - recon_search_best.json: colleague format

Recommendations are based on trial-level progress CSVs when available. Each
trial is scored relative to other trials inside the same artifact folder first,
then those within-artifact effects are aggregated by hyperparameter value. This
keeps different training methods, sensor sets, or input/output schemas from
being mistaken for a hyperparameter effect.

Parameter aliases are normalized so they can be analyzed together:
  - recon_past_len      -> tcn_past_window
  - recon_hrizon        -> tcn_future_window
  - recon_horizon       -> tcn_future_window
  - recon_stride        -> tcn_stride

Usage:
  python analyze_hyperparams.py --root . --out reports/hyperparam_analysis
  python analyze_hyperparams.py --root /path/to/experiments --no-recursive
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BEST_FILE_PRIORITY = ("recon_search_best.json", "search_best.json", "best_search.json")
TRIAL_FILE_PRIORITY = (
    "search_trial_progress.csv",
    "recon_search_trial_progress.csv",
    "recon_trial_progress.csv",
    "recon_search_progress.csv",
    "trial_progress.csv",
)

PARAM_ALIASES = {
    "tcn_past_window": ("tcn_past_window", "recon_past_len"),
    "tcn_future_window": ("tcn_future_window", "recon_hrizon", "recon_horizon"),
    "tcn_stride": ("tcn_stride", "recon_stride"),
}

ONE_SIDED_PARAMS = ("raw_error_quantile", "per_step_std")

CORE_PARAM_ORDER = (
    "tcn_past_window",
    "tcn_future_window",
    "tcn_stride",
    "tcn_channels",
    "tcn_blocks",
    "tcn_kernel",
    "tcn_dropout",
    "tcn_lr",
    "tcn_batch",
    "tcn_weight_decay",
    "raw_error_quantile",
    "per_step_std",
    "h_cart_mode",
    "future_h_mode",
    "virtual_sensor_mode",
)

HEATMAP_PAIRS = (
    ("tcn_past_window", "tcn_future_window"),
    ("tcn_stride", "tcn_future_window"),
    ("tcn_channels", "tcn_blocks"),
    ("tcn_kernel", "tcn_blocks"),
    ("raw_error_quantile", "per_step_std"),
)


def _json_default(obj: Any):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _is_artifacts_dir(path: Path) -> bool:
    name = path.name
    return name.startswith("artifacts") and not name.startswith("_artifacts")


def find_artifact_dirs(root: Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    dirs = []
    if root.is_dir() and _is_artifacts_dir(root):
        dirs.append(root)
    it = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        if p.is_dir() and _is_artifacts_dir(p):
            dirs.append(p)
    return sorted(set(dirs), key=lambda p: str(p))


def _pick_best_file(artifact_dir: Path) -> Path | None:
    for name in BEST_FILE_PRIORITY:
        p = artifact_dir / name
        if p.exists():
            return p
    return None


def _pick_trial_file(artifact_dir: Path) -> Path | None:
    for name in TRIAL_FILE_PRIORITY:
        p = artifact_dir / name
        if p.exists():
            return p
    return None


def _source_type(best_file: Path) -> str:
    return "colleague" if best_file.name == "recon_search_best.json" else "ours"


def _source_type_for_artifact(artifact_dir: Path, trial_file: Path | None = None) -> str:
    best_file = _pick_best_file(artifact_dir)
    if best_file is not None:
        return _source_type(best_file)
    if trial_file is not None and trial_file.name.startswith("recon"):
        return "colleague"
    return "ours"


def _get_nested(d: dict, path: tuple[str, ...], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _first_present(d: dict, keys, default=None):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
        if pd.isna(value):
            return True
    except Exception:
        pass
    return isinstance(value, str) and value.strip() == ""


def _first_nonempty(d, keys, default=None):
    for key in keys:
        try:
            value = d.get(key, None)
        except AttributeError:
            value = d[key] if key in d else None
        if not _is_missing(value):
            return value
    return default


def _normalize_params(params: dict) -> dict:
    out = {}
    used = set()
    for std_key, aliases in PARAM_ALIASES.items():
        value = _first_present(params, aliases)
        if value is not None:
            out[std_key] = value
        used.update(aliases)

    for key, value in params.items():
        if key in used:
            continue
        out[key] = value

    for key in ONE_SIDED_PARAMS:
        out.setdefault(key, None)
    return out


def _safe_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


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


def _metric_from_best(best: dict, key: str, default=None):
    val = best.get("val") or {}
    ev = best.get("evaluation_criteria") or {}
    aliases = {
        "correct": (("val", "correct"), ("evaluation_criteria", "primary_metric_value")),
        "n": (("val", "n"), ("evaluation_criteria", "n_wafers")),
        "tp": (("val", "tp"), ("evaluation_criteria", "tp")),
        "tn": (("val", "tn"), ("evaluation_criteria", "tn")),
        "fp": (("val", "fp"), ("evaluation_criteria", "fp")),
        "fn": (("val", "fn"), ("evaluation_criteria", "fn")),
        "n_ok": (("val", "n_ok"), ("evaluation_criteria", "n_ok")),
        "n_ng": (("val", "n_ng"), ("evaluation_criteria", "n_ng")),
        "auroc": (("val", "auroc"), ("evaluation_criteria", "auroc")),
    }
    for path in aliases.get(key, ()):
        value = _get_nested(best, path)
        if value is not None:
            return value
    if key in val:
        return val[key]
    if key in ev:
        return ev[key]
    return default


def read_best_record(artifact_dir: Path) -> dict | None:
    best_file = _pick_best_file(artifact_dir)
    if best_file is None:
        return None
    try:
        doc = json.loads(best_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "artifact_dir": str(artifact_dir),
            "artifact_name": artifact_dir.name,
            "best_file": str(best_file),
            "source_type": _source_type(best_file),
            "read_error": str(exc),
        }

    best = doc.get("best", doc)
    params = _normalize_params(best.get("params") or {})
    val_mse = best.get("forecast_val_mse", best.get("val_mse"))
    threshold = best.get("threshold")
    if threshold is None:
        threshold = _get_nested(best, ("val", "threshold_raw"),
                                _get_nested(best, ("val", "threshold")))

    row = {
        "artifact_dir": str(artifact_dir),
        "artifact_name": artifact_dir.name,
        "best_file": str(best_file),
        "best_file_name": best_file.name,
        "source_type": _source_type(best_file),
        "model_kind": best.get("model_kind", ""),
        "score_name": best.get("score_name", ""),
        "trial_idx": best.get("trial_idx", _get_nested(best, ("evaluation_criteria", "trial_idx"), "")),
        "correct": _safe_int(_metric_from_best(best, "correct", 0)),
        "n": _safe_int(_metric_from_best(best, "n", 0)),
        "tp": _safe_int(_metric_from_best(best, "tp", 0)),
        "tn": _safe_int(_metric_from_best(best, "tn", 0)),
        "fp": _safe_int(_metric_from_best(best, "fp", 0)),
        "fn": _safe_int(_metric_from_best(best, "fn", 0)),
        "n_ok": _safe_int(_metric_from_best(best, "n_ok", 0)),
        "n_ng": _safe_int(_metric_from_best(best, "n_ng", 0)),
        "auroc": _safe_float(_metric_from_best(best, "auroc", np.nan)),
        "forecast_val_mse": _safe_float(val_mse),
        "threshold_json": json.dumps(threshold, ensure_ascii=False, default=_json_default),
    }
    row.update(params)
    return row


def collect_best_records(root: Path, recursive: bool = True) -> pd.DataFrame:
    rows = []
    for artifact_dir in find_artifact_dirs(root, recursive=recursive):
        rec = read_best_record(artifact_dir)
        if rec is not None:
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for key in CORE_PARAM_ORDER:
        if key not in df.columns:
            df[key] = np.nan
    return df


def _trial_params_from_row(row) -> dict:
    direct_keys = set(CORE_PARAM_ORDER) | set(ONE_SIDED_PARAMS)
    for aliases in PARAM_ALIASES.values():
        direct_keys.update(aliases)
    params = {}
    for col, value in row.items():
        if _is_missing(value):
            continue
        key = None
        if str(col).startswith("param_"):
            key = str(col)[6:]
        elif col in direct_keys:
            key = str(col)
        if key:
            params[key] = _parse_progress_value(value)
    return _normalize_params(params)


def read_trial_records(artifact_dir: Path) -> list[dict]:
    trial_file = _pick_trial_file(artifact_dir)
    if trial_file is None:
        return []
    try:
        progress = pd.read_csv(trial_file)
    except Exception as exc:
        return [{
            "artifact_dir": str(artifact_dir),
            "artifact_name": artifact_dir.name,
            "trial_file": str(trial_file),
            "trial_file_name": trial_file.name,
            "source_type": _source_type_for_artifact(artifact_dir, trial_file),
            "read_error": str(exc),
        }]

    rows = []
    for row_no, row in progress.iterrows():
        status = str(_first_nonempty(row, ("status",), "ok")).strip().lower()
        if status in {"error", "failed", "fail", "no_score"}:
            continue

        params = _trial_params_from_row(row)
        correct = _safe_float(_first_nonempty(
            row, ("trial_tp_plus_tf", "trial_correct", "correct", "tp_plus_tf", "n_correct")
        ))
        tp = _safe_float(_first_nonempty(row, ("trial_tp", "tp")))
        tn = _safe_float(_first_nonempty(row, ("trial_tf", "trial_tn", "tf", "tn")))
        if not np.isfinite(correct) and np.isfinite(tp) and np.isfinite(tn):
            correct = tp + tn
        n_ok = _safe_float(_first_nonempty(row, ("trial_n_ok", "n_ok")))
        n_ng = _safe_float(_first_nonempty(row, ("trial_n_ng", "n_ng")))
        n = _safe_float(_first_nonempty(row, ("trial_n_wafers", "n_wafers", "n", "total")))
        if not np.isfinite(n) and np.isfinite(n_ok) and np.isfinite(n_ng):
            n = n_ok + n_ng

        rec = {
            "artifact_dir": str(artifact_dir),
            "artifact_name": artifact_dir.name,
            "trial_file": str(trial_file),
            "trial_file_name": trial_file.name,
            "source_type": _source_type_for_artifact(artifact_dir, trial_file),
            "row_no": int(row_no),
            "trial_idx": _first_nonempty(row, ("trial_idx", "completed_idx", "trial"), row_no + 1),
            "status": status,
            "model_kind": _first_nonempty(row, ("model_kind",), ""),
            "score_name": _first_nonempty(row, ("score_name",), ""),
            "correct": correct,
            "n": n,
            "tp": tp,
            "tn": tn,
            "fp": _safe_float(_first_nonempty(row, ("trial_fp", "fp"))),
            "fn": _safe_float(_first_nonempty(row, ("trial_fn", "fn"))),
            "n_ok": n_ok,
            "n_ng": n_ng,
            "auroc": _safe_float(_first_nonempty(row, ("trial_auroc", "auroc"))),
            "forecast_val_mse": _safe_float(_first_nonempty(
                row, ("forecast_val_mse", "val_mse", "trial_val_mse", "recon_val_mse")
            )),
        }
        rec.update(params)
        rows.append(rec)
    return rows


def collect_trial_records(root: Path, recursive: bool = True) -> pd.DataFrame:
    rows = []
    for artifact_dir in find_artifact_dirs(root, recursive=recursive):
        rows.extend(read_trial_records(artifact_dir))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for key in CORE_PARAM_ORDER:
        if key not in df.columns:
            df[key] = np.nan
    return df


def _rank01(series: pd.Series, high_good: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if valid.empty:
        return out
    if len(valid) == 1 or valid.nunique(dropna=True) <= 1:
        out.loc[valid.index] = 0.5
        return out
    rank = valid.rank(method="average", ascending=True if high_good else False)
    out.loc[valid.index] = (rank - 1.0) / (len(valid) - 1.0)
    return out


def add_within_artifact_scores(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return trials
    out = trials.copy()
    for col in ("correct", "tp", "forecast_val_mse"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["correct_percentile"] = np.nan
    out["tp_percentile"] = np.nan
    out["mse_percentile_good"] = np.nan
    out["correct_delta"] = np.nan
    out["tp_delta"] = np.nan
    out["mse_delta"] = np.nan

    for _, g in out.groupby("artifact_dir", dropna=False):
        idx = g.index
        out.loc[idx, "correct_percentile"] = _rank01(g["correct"], high_good=True)
        out.loc[idx, "tp_percentile"] = _rank01(g["tp"], high_good=True)
        out.loc[idx, "mse_percentile_good"] = _rank01(g["forecast_val_mse"], high_good=False)

        correct_mean = pd.to_numeric(g["correct"], errors="coerce").mean()
        tp_mean = pd.to_numeric(g["tp"], errors="coerce").mean()
        mse_mean = pd.to_numeric(g["forecast_val_mse"], errors="coerce").mean()
        out.loc[idx, "correct_delta"] = out.loc[idx, "correct"] - correct_mean
        out.loc[idx, "tp_delta"] = out.loc[idx, "tp"] - tp_mean
        out.loc[idx, "mse_delta"] = mse_mean - out.loc[idx, "forecast_val_mse"]

    weights = {
        "correct_percentile": 0.65,
        "tp_percentile": 0.25,
        "mse_percentile_good": 0.10,
    }
    score = pd.Series(0.0, index=out.index, dtype=float)
    weight_sum = pd.Series(0.0, index=out.index, dtype=float)
    for col, weight in weights.items():
        valid = pd.to_numeric(out[col], errors="coerce")
        mask = valid.notna()
        score.loc[mask] += valid.loc[mask] * weight
        weight_sum.loc[mask] += weight
    out["relative_score"] = np.where(weight_sum > 0, score / weight_sum, np.nan)
    return out


def _sort_best(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_mse_sort"] = pd.to_numeric(out.get("forecast_val_mse"), errors="coerce")
    out["_mse_sort"] = out["_mse_sort"].fillna(np.inf)
    return out.sort_values(
        ["correct", "tp", "_mse_sort"],
        ascending=[False, False, True],
    ).drop(columns=["_mse_sort"], errors="ignore")


def _param_columns(df: pd.DataFrame) -> list[str]:
    fixed = {
        "artifact_dir", "artifact_name", "best_file", "best_file_name", "source_type",
        "model_kind", "score_name", "trial_idx", "correct", "n", "tp", "tn",
        "fp", "fn", "n_ok", "n_ng", "auroc", "forecast_val_mse",
        "threshold_json", "read_error", "trial_file", "trial_file_name", "row_no",
        "status", "correct_percentile", "tp_percentile", "mse_percentile_good",
        "correct_delta", "tp_delta", "mse_delta", "relative_score",
    }
    cols = [c for c in CORE_PARAM_ORDER if c in df.columns]
    extras = sorted(c for c in df.columns if c not in fixed and c not in cols)
    return [c for c in cols + extras if c in df.columns]


def _write_best_only_recommendations(df: pd.DataFrame, out_dir: Path) -> None:
    param_cols = _param_columns(df)
    rows = []
    best_params = {}
    for param in param_cols:
        valid = df[[param, "correct", "tp", "forecast_val_mse"]].dropna(subset=[param])
        valid = valid[valid[param].astype(str) != ""]
        if valid.empty:
            continue
        grouped = []
        for value, g in valid.groupby(param, dropna=True):
            artifact_count = int(df.loc[g.index, "artifact_dir"].nunique()) if "artifact_dir" in df.columns else int(len(g))
            grouped.append({
                "param": param,
                "value": value,
                "trial_count": int(len(g)),
                "artifact_count": artifact_count,
                "evidence_source": "artifact_best_only_confounded",
                "warning": "best-only values are confounded by training method/sensor setup",
                "mean_correct": float(pd.to_numeric(g["correct"], errors="coerce").mean()),
                "mean_tp": float(pd.to_numeric(g["tp"], errors="coerce").mean()),
                "mean_forecast_val_mse": float(pd.to_numeric(g["forecast_val_mse"], errors="coerce").mean()),
                "best_correct": int(pd.to_numeric(g["correct"], errors="coerce").max()),
                "best_tp": int(pd.to_numeric(g["tp"], errors="coerce").max()),
            })
        tmp = pd.DataFrame(grouped)
        tmp["_mse_sort"] = tmp["mean_forecast_val_mse"].fillna(np.inf)
        tmp = tmp.sort_values(
            ["mean_correct", "mean_tp", "_mse_sort", "trial_count"],
            ascending=[False, False, True, False],
        ).drop(columns=["_mse_sort"])
        rows.extend(tmp.to_dict("records"))
        pick = tmp.iloc[0].to_dict()
        best_params[param] = {
            "value": pick["value"],
            "evidence_source": "artifact_best_only_confounded",
            "warning": "trial progress CSV was not found; recommendation may reflect training method/sensor setup.",
            "artifact_count": int(pick.get("artifact_count", 0)),
            "trial_count": int(pick.get("trial_count", 0)),
            "mean_correct": _safe_float(pick.get("mean_correct")),
            "mean_tp": _safe_float(pick.get("mean_tp")),
            "mean_forecast_val_mse": _safe_float(pick.get("mean_forecast_val_mse")),
        }

    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(out_dir / "hyperparam_recommendations.csv", index=False, encoding="utf-8-sig")
    (out_dir / "recommended_params.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def write_recommendations(best_df: pd.DataFrame, trial_df: pd.DataFrame,
                          out_dir: Path, min_artifacts: int = 2) -> str:
    if trial_df.empty or "relative_score" not in trial_df.columns:
        _write_best_only_recommendations(best_df, out_dir)
        return "artifact_best_only_confounded"

    param_cols = _param_columns(trial_df)
    rows = []
    best_params = {}
    for param in param_cols:
        needed = [
            param, "artifact_dir", "relative_score", "correct_percentile",
            "tp_percentile", "mse_percentile_good", "correct_delta",
            "tp_delta", "mse_delta", "correct", "tp", "forecast_val_mse",
        ]
        valid = trial_df[[c for c in needed if c in trial_df.columns]].dropna(subset=[param])
        valid = valid[valid[param].astype(str) != ""]
        valid = valid[pd.to_numeric(valid["relative_score"], errors="coerce").notna()]
        if valid.empty:
            continue

        grouped = []
        for value, g in valid.groupby(param, dropna=True):
            artifact_count = int(g["artifact_dir"].nunique())
            grouped.append({
                "param": param,
                "value": value,
                "trial_count": int(len(g)),
                "artifact_count": artifact_count,
                "low_support": bool(artifact_count < min_artifacts),
                "evidence_source": "within_artifact_trials",
                "mean_relative_score": float(pd.to_numeric(g["relative_score"], errors="coerce").mean()),
                "median_relative_score": float(pd.to_numeric(g["relative_score"], errors="coerce").median()),
                "mean_correct_percentile": float(pd.to_numeric(g["correct_percentile"], errors="coerce").mean()),
                "mean_tp_percentile": float(pd.to_numeric(g["tp_percentile"], errors="coerce").mean()),
                "mean_mse_percentile_good": float(pd.to_numeric(g["mse_percentile_good"], errors="coerce").mean()),
                "mean_correct_delta": float(pd.to_numeric(g["correct_delta"], errors="coerce").mean()),
                "mean_tp_delta": float(pd.to_numeric(g["tp_delta"], errors="coerce").mean()),
                "mean_mse_delta": float(pd.to_numeric(g["mse_delta"], errors="coerce").mean()),
                "mean_correct": float(pd.to_numeric(g["correct"], errors="coerce").mean()),
                "mean_tp": float(pd.to_numeric(g["tp"], errors="coerce").mean()),
                "mean_forecast_val_mse": float(pd.to_numeric(g["forecast_val_mse"], errors="coerce").mean()),
            })
        tmp = pd.DataFrame(grouped)
        tmp["_support_ok"] = tmp["artifact_count"] >= min_artifacts
        if not tmp["_support_ok"].any():
            tmp["_support_ok"] = True
        tmp = tmp.sort_values(
            [
                "_support_ok", "mean_relative_score", "artifact_count",
                "trial_count", "mean_correct_delta", "mean_tp_delta",
                "mean_mse_delta",
            ],
            ascending=[False, False, False, False, False, False, False],
        ).drop(columns=["_support_ok"])
        rows.extend(tmp.to_dict("records"))

        pick = tmp.iloc[0].to_dict()
        best_params[param] = {
            "value": pick["value"],
            "evidence_source": "within_artifact_trials",
            "artifact_count": int(pick["artifact_count"]),
            "trial_count": int(pick["trial_count"]),
            "low_support": bool(pick["low_support"]),
            "mean_relative_score": _safe_float(pick["mean_relative_score"]),
            "mean_correct_delta": _safe_float(pick["mean_correct_delta"]),
            "mean_tp_delta": _safe_float(pick["mean_tp_delta"]),
            "mean_mse_delta": _safe_float(pick["mean_mse_delta"]),
        }

    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(out_dir / "hyperparam_recommendations.csv", index=False, encoding="utf-8-sig")
    (out_dir / "recommended_params.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return "within_artifact_trials"


def _format_value(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def plot_artifact_ranking(df: pd.DataFrame, out_path: Path) -> None:
    d = _sort_best(df).head(40).iloc[::-1]
    if d.empty:
        return
    labels = d["artifact_name"].astype(str).tolist()
    y = np.arange(len(d))
    correct = pd.to_numeric(d["correct"], errors="coerce").fillna(0)
    tp = pd.to_numeric(d["tp"], errors="coerce").fillna(0)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(d) + 1.2)))
    ax.barh(y, correct, color="#4c78a8", label="correct")
    ax.scatter(tp, y, color="#d62728", s=24, label="TP")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("count")
    ax.set_title("Best performance by artifact (top 40)")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_param_frequency(df: pd.DataFrame, out_path: Path) -> None:
    param_cols = _param_columns(df)
    cards = []
    for param in param_cols:
        vc = df[param].dropna().astype(str)
        vc = vc[vc != ""]
        if vc.empty:
            continue
        counts = vc.value_counts().head(8)
        cards.append((param, counts))
    if not cards:
        return
    ncol = 3
    nrow = math.ceil(len(cards) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow), squeeze=False)
    for i, (param, counts) in enumerate(cards):
        ax = axes[i // ncol][i % ncol]
        counts.iloc[::-1].plot.barh(ax=ax, color="#59a14f")
        ax.set_title(param, fontsize=10)
        ax.set_xlabel("best count")
        ax.tick_params(labelsize=8)
    for i in range(len(cards), nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    fig.suptitle("Best hyperparameter value frequency", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_performance_by_param(df: pd.DataFrame, out_path: Path) -> None:
    param_cols = _param_columns(df)
    cards = []
    for param in param_cols:
        valid = df[[param, "correct", "tp", "forecast_val_mse"]].dropna(subset=[param])
        valid = valid[valid[param].astype(str) != ""]
        if valid[param].nunique(dropna=True) <= 1:
            continue
        g = valid.groupby(param).agg(
            mean_correct=("correct", "mean"),
            mean_tp=("tp", "mean"),
            mean_mse=("forecast_val_mse", "mean"),
            count=("correct", "size"),
        ).reset_index()
        g["_mse_sort"] = g["mean_mse"].fillna(np.inf)
        g = g.sort_values(["mean_correct", "mean_tp", "_mse_sort"],
                          ascending=[False, False, True]).drop(columns=["_mse_sort"])
        cards.append((param, g.head(10)))
    if not cards:
        return
    ncol = 2
    nrow = math.ceil(len(cards) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 3.7 * nrow), squeeze=False)
    for i, (param, g) in enumerate(cards):
        ax = axes[i // ncol][i % ncol]
        labels = [_format_value(v) for v in g[param]]
        x = np.arange(len(g))
        ax.bar(x - 0.18, g["mean_correct"], width=0.36, color="#4c78a8", label="mean correct")
        ax.bar(x + 0.18, g["mean_tp"], width=0.36, color="#f58518", label="mean TP")
        for xi, cnt in zip(x, g["count"]):
            ax.text(xi, max(g["mean_correct"].max(), g["mean_tp"].max()) * 1.01,
                    f"n={int(cnt)}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(param, fontsize=10)
        ax.grid(True, axis="y", alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8)
    for i in range(len(cards), nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    fig.suptitle("Best-only mean performance by hyperparameter value (confounded)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_method_adjusted_by_param(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "relative_score" not in df.columns:
        return
    param_cols = _param_columns(df)
    cards = []
    for param in param_cols:
        valid = df[[param, "relative_score", "artifact_dir"]].dropna(subset=[param, "relative_score"])
        valid = valid[valid[param].astype(str) != ""]
        if valid[param].nunique(dropna=True) <= 1:
            continue
        g = valid.groupby(param).agg(
            mean_relative_score=("relative_score", "mean"),
            trial_count=("relative_score", "size"),
            artifact_count=("artifact_dir", "nunique"),
        ).reset_index()
        g = g.sort_values(
            ["mean_relative_score", "artifact_count", "trial_count"],
            ascending=[False, False, False],
        )
        cards.append((param, g.head(10)))
    if not cards:
        return
    ncol = 2
    nrow = math.ceil(len(cards) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 3.7 * nrow), squeeze=False)
    for i, (param, g) in enumerate(cards):
        ax = axes[i // ncol][i % ncol]
        labels = [_format_value(v) for v in g[param]]
        x = np.arange(len(g))
        ax.bar(x, g["mean_relative_score"], width=0.6, color="#4c78a8")
        for xi, cnt, acnt in zip(x, g["trial_count"], g["artifact_count"]):
            ax.text(xi, min(1.03, g["mean_relative_score"].iloc[xi] + 0.03),
                    f"n={int(cnt)} / a={int(acnt)}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(param, fontsize=10)
        ax.set_ylabel("within-artifact score")
        ax.grid(True, axis="y", alpha=0.2)
    for i in range(len(cards), nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    fig.suptitle("Method-adjusted hyperparameter effect", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, x_param: str, y_param: str, out_path: Path,
                 value_col: str = "correct", value_label: str = "mean correct") -> None:
    if x_param not in df.columns or y_param not in df.columns:
        return
    if value_col not in df.columns:
        return
    valid = df[[x_param, y_param, value_col]].dropna(subset=[x_param, y_param, value_col])
    valid = valid[(valid[x_param].astype(str) != "") & (valid[y_param].astype(str) != "")]
    if valid.empty or valid[x_param].nunique() < 2 or valid[y_param].nunique() < 2:
        return
    pivot = valid.pivot_table(index=y_param, columns=x_param, values=value_col, aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(pivot.columns) + 2),
                                    max(4, 0.45 * len(pivot.index) + 2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_format_value(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_format_value(v) for v in pivot.index])
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"{value_label}: {y_param} x {x_param}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                fmt = ".2f" if value_col == "relative_score" else ".1f"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_plots(best_df: pd.DataFrame, out_dir: Path, trial_df: pd.DataFrame | None = None) -> None:
    plot_artifact_ranking(best_df, out_dir / "artifact_best_ranking.png")
    plot_param_frequency(best_df, out_dir / "best_param_frequency.png")
    plot_performance_by_param(best_df, out_dir / "performance_by_param_best_only.png")
    value_df = best_df
    value_col = "correct"
    value_label = "best-only mean correct"
    if trial_df is not None and not trial_df.empty and "relative_score" in trial_df.columns:
        plot_method_adjusted_by_param(trial_df, out_dir / "method_adjusted_performance_by_param.png")
        value_df = trial_df
        value_col = "relative_score"
        value_label = "method-adjusted mean score"
    for x_param, y_param in HEATMAP_PAIRS:
        plot_heatmap(value_df, x_param, y_param,
                     out_dir / f"heatmap_{y_param}_x_{x_param}.png",
                     value_col=value_col, value_label=value_label)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate hyperparameter search results from artifacts* folders.")
    ap.add_argument("--root", default=".", help="Root folder to scan.")
    ap.add_argument("--out", default=None,
                    help="Output folder. Default: <root>/reports/hyperparam_analysis")
    ap.add_argument("--no-recursive", action="store_true",
                    help="Only scan direct child folders under --root.")
    ap.add_argument("--min-artifacts-for-recommendation", type=int, default=2,
                    help="Prefer values observed in at least this many artifact folders.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else root / "reports" / "hyperparam_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_best_records(root, recursive=not args.no_recursive)
    if df.empty:
        print(f"No artifacts* best JSON found under {root}")
        return

    df = _sort_best(df)
    df.to_csv(out_dir / "hyperparam_summary.csv", index=False, encoding="utf-8-sig")
    trials = collect_trial_records(root, recursive=not args.no_recursive)
    if not trials.empty:
        trials = add_within_artifact_scores(trials)
        trials.to_csv(out_dir / "hyperparam_trials.csv", index=False, encoding="utf-8-sig")

    recommendation_method = write_recommendations(
        df, trials, out_dir,
        min_artifacts=max(1, int(args.min_artifacts_for_recommendation)),
    )
    write_plots(df, out_dir, trials)

    outputs = [
        "hyperparam_summary.csv",
        "hyperparam_recommendations.csv",
        "recommended_params.json",
        "artifact_best_ranking.png",
        "best_param_frequency.png",
        "performance_by_param_best_only.png",
        "heatmap_*.png",
    ]
    if not trials.empty:
        outputs.extend([
            "hyperparam_trials.csv",
            "method_adjusted_performance_by_param.png",
        ])

    overview = {
        "root": str(root),
        "n_artifacts": int(len(df)),
        "n_trial_rows": int(len(trials)) if not trials.empty else 0,
        "n_artifacts_with_trials": int(trials["artifact_dir"].nunique()) if not trials.empty else 0,
        "sources": df["source_type"].value_counts(dropna=False).to_dict(),
        "recommendation_method": recommendation_method,
        "recommendation_note": (
            "Trial-level recommendations are computed from within-artifact relative scores, "
            "so each training method/sensor setup is baseline-adjusted before aggregation."
            if recommendation_method == "within_artifact_trials"
            else "Trial progress CSVs were not found; recommendations are best-only and confounded by training method/sensor setup."
        ),
        "best_artifact": df.iloc[0]["artifact_name"],
        "best_correct": int(df.iloc[0].get("correct", 0)),
        "best_tp": int(df.iloc[0].get("tp", 0)),
        "best_forecast_val_mse": _safe_float(df.iloc[0].get("forecast_val_mse")),
        "outputs": outputs,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"Artifacts analyzed: {len(df)}")
    print(f"Trial rows analyzed: {overview['n_trial_rows']} "
          f"| artifacts with trials={overview['n_artifacts_with_trials']}")
    print(f"Recommendation method: {recommendation_method}")
    print(f"Output: {out_dir}")
    print(f"Best: {overview['best_artifact']} | correct={overview['best_correct']} "
          f"| TP={overview['best_tp']} | val_mse={overview['best_forecast_val_mse']}")


if __name__ == "__main__":
    main()
