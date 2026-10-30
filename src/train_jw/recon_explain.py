"""IO-Recon-TCN 설명력 자료 — 예상 vs 실제 + 잔차 히트맵 + IO 인과 내러티브.

논리(요구 #6):
  "IO 가 이전 단계 대비 *이만큼 변하라*(기대 Δ)고 명령했는데, 실제 센서는 안 변했다
   (실제 Δ≈0). 그래서 불량이다."  → IO 명령 대비 무응답을 수치·그림으로 주장.

산출(요구 #7):
  · 01_이상점수_그래프/<wafer>.png — 상위 이탈 센서별: 실제(빨강) vs 모델 기대(파랑).
  · 01_이상점수_그래프/<wafer>/00_웨이퍼점수.png — 전체 센서 합산 누적 이상치 점수.
  · 02_히트맵/<wafer>.png — raw error level 스텝×센서 히트맵.
  · 04_웨이퍼요약/_웨이퍼_설명요약.json — wafer 별 한 줄 인과 내러티브 + 수치.

저장된 TCN forecast best 모델(artifacts/tcn_forecast.pt)을 불러와
val/ok·val/ng 에 대해 생성.

실행:
  python run.py recon-explain                         # val/ok·val/ng → reports/recon/
  python run.py recon-explain --ng-only               # 불량만
  python run.py recon-explain --topk 8                # wafer score에 포함할 상위 센서 수
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import IOConfig
from .data import load_dataset
from .export_report import _step_map
from .model_tcn_forecast import TCNForecastDetector
from .sensor_graph import graph_adjust_abs_z, graph_sensor_rank, is_pseudo_node, load_sensor_graph


def _setup_font() -> None:
    """Use a Korean-capable font when available; keep plotting usable on headless Linux."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    import matplotlib.font_manager as fm

    candidates = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",
        "AppleGothic",
        "DejaVu Sans",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _sub_ci(base, name):
    """Return an existing child directory with case-insensitive name matching."""
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


def _actual(A, rec) -> np.ndarray:
    if hasattr(A, "actual"):
        return np.asarray(A.actual(rec), dtype=np.float64)
    return np.asarray(rec["sensors"], dtype=np.float64)


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
    """Find midpoint threshold maximizing correct wafers; tie-break by NG TP."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    best = None
    for th in _threshold_candidates_midpoint(p):
        m = _metrics_at_threshold(y, p, th)
        cand = (m["correct"], m["tp"], float(th), m)
        if best is None or cand[:2] > best[:2]:
            best = cand
    _, _, th, metrics = best
    return {**metrics,
            "threshold": round(float(th), 4), "threshold_raw": float(th),
            "auroc": _safe_auroc(y, p)}


def _recipe_key(rec) -> str:
    return str(rec.get("recipe", "unknown") or "unknown")


def _recipe_lookup_key(recipe) -> str:
    return str(recipe or "").strip().lower()


def _lookup_recipe_value(table, recipe):
    if not isinstance(table, dict):
        return None
    recipe = str(recipe or "unknown")
    if recipe in table:
        return table[recipe]
    wanted = _recipe_lookup_key(recipe)
    for key, value in table.items():
        if _recipe_lookup_key(key) == wanted:
            return value
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


def _lookup_bare_recipe_threshold(table, recipe) -> float | None:
    if not isinstance(table, dict):
        return None
    recipe = str(recipe or "unknown")
    if recipe in table and not _is_threshold_metadata_key(recipe):
        value = _float_threshold_value(table[recipe])
        if value is not None:
            return value
    wanted = _recipe_lookup_key(recipe)
    for key, value in table.items():
        if _is_threshold_metadata_key(key):
            continue
        if _recipe_lookup_key(key) == wanted:
            value = _float_threshold_value(value)
            if value is not None:
                return value
    return None


def _threshold_recipe_multiplier(threshold, recipe) -> float:
    if not isinstance(threshold, dict):
        return 1.0
    value = _lookup_recipe_value(threshold.get("threshold_recipe_multipliers") or {}, recipe)
    try:
        value = 1.0 if value is None else float(value)
    except Exception:
        value = 1.0
    return value if np.isfinite(value) and value > 0 else 1.0


def _correct_from_scores_by_recipe(records, y, p, cfg):
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
    threshold_by_recipe, recipe_metrics = {}, {}
    for recipe in recipes:
        idx = [i for i, r in enumerate(records) if _recipe_key(r) == recipe]
        if not idx:
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
    return {
        "correct": tp + tn, "n": n_ok + n_ng, "tp": tp, "tn": tn,
        "fn": fn, "fp": fp, "n_ok": n_ok, "n_ng": n_ng,
        "threshold": {k: round(float(v), 4) for k, v in threshold_by_recipe.items()},
        "threshold_raw": dict(threshold_by_recipe),
        "threshold_by_recipe": dict(threshold_by_recipe),
        "global_threshold": round(fallback_th, 4),
        "global_threshold_raw": fallback_th,
        "recipe_metrics": recipe_metrics,
        "threshold_strategy": "per_recipe_midpoint_between_adjacent_scores",
        "auroc": _safe_auroc(y, p),
    }


def _metrics_from_predictions(y, pred, score_for_auc=None):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    n_ok = int((y == 0).sum())
    n_ng = int((y == 1).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"correct": tp + tn, "n": n_ok + n_ng, "tp": tp, "tn": tn,
            "fn": fn, "fp": fp, "n_ok": n_ok, "n_ng": n_ng,
            "auroc": _safe_auroc(y, score_for_auc if score_for_auc is not None else pred)}


def _threshold_for_record(threshold, rec):
    if threshold is None:
        return None
    if isinstance(threshold, dict):
        if str(threshold.get("threshold_wise", "")).lower() in {"sensor", "step"}:
            return float(threshold.get("decision_threshold", 1.0))
        recipe = _recipe_key(rec)
        value = _lookup_bare_recipe_threshold(threshold, recipe)
        if value is not None:
            return value
        for key in ("threshold_by_recipe", "threshold_raw", "threshold"):
            value = _lookup_recipe_value(threshold.get(key), recipe)
            value = _float_threshold_value(value)
            if value is not None:
                return value
        for key in ("default", "global", "global_threshold_raw", "global_threshold"):
            value = _float_threshold_value(threshold.get(key))
            if value is not None:
                return value
        for key in ("threshold_raw", "threshold"):
            value = _float_threshold_value(threshold.get(key))
            if value is not None:
                return value
        return None
    return float(threshold)


def _display_scale_threshold_for_record(threshold, rec) -> float | None:
    if threshold is None:
        return None
    if not isinstance(threshold, dict):
        try:
            value = float(threshold)
        except Exception:
            return None
        return value if np.isfinite(value) and value > 1e-12 else None
    recipe = _recipe_key(rec)
    if bool(threshold.get("display_scale_uses_adjusted_threshold", False)):
        adjusted = _threshold_for_record(threshold, rec)
        try:
            adjusted = float(adjusted)
            if np.isfinite(adjusted) and adjusted > 1e-12:
                return adjusted
        except Exception:
            pass
    base = _lookup_recipe_value(threshold.get("base_threshold_by_recipe"), recipe)
    if base is not None:
        try:
            base = float(base)
            if np.isfinite(base) and base > 1e-12:
                return base
        except Exception:
            pass
    if _threshold_mode(threshold) == "all":
        return _all_threshold_value(threshold)
    factor = _threshold_recipe_multiplier(threshold, recipe)
    adjusted = _threshold_for_record(threshold, rec)
    if abs(factor - 1.0) > 1e-12 and adjusted is not None:
        base = float(adjusted) / factor
        return base if np.isfinite(base) and base > 1e-12 else None
    return adjusted


def _display_threshold_for_record(threshold, rec) -> float | None:
    decision = _threshold_for_record(threshold, rec)
    scale = _display_scale_threshold_for_record(threshold, rec)
    try:
        decision = None if decision is None else float(decision)
        scale = None if scale is None else float(scale)
    except Exception:
        return None
    if decision is None or scale is None or not np.isfinite(decision) \
            or not np.isfinite(scale) or scale <= 1e-12:
        return None
    return decision / scale


def _format_threshold(threshold) -> str:
    if threshold is None:
        return "—"
    if isinstance(threshold, dict):
        mode = str(threshold.get("threshold_wise", "")).lower()
        if mode in {"sensor", "step"}:
            table = threshold.get("thresholds") or {}
            n = 0
            if mode == "sensor":
                n = sum(len(v) for v in table.values() if isinstance(v, dict))
            else:
                for sensors in table.values():
                    if isinstance(sensors, dict):
                        n += sum(len(steps) for steps in sensors.values()
                                 if isinstance(steps, dict))
            return f"{mode} thresholds {n}개 (decision ≥ {threshold.get('decision_threshold', 1.0)})"
        table = threshold.get("threshold_by_recipe") or threshold.get("threshold_raw") or threshold
        pieces = []
        if isinstance(table, dict):
            for key, value in sorted(table.items()):
                if _is_threshold_metadata_key(key):
                    continue
                value = _float_threshold_value(value)
                if value is not None:
                    pieces.append(f"{key}:{value:.4f}")
        return "; ".join(pieces) if pieces else "—"
    value = _float_threshold_value(threshold)
    return "—" if value is None else f"{value:.4f}"


def _prev_baseline(actual_col: np.ndarray, step, di: int, half: int = 8) -> float:
    """피크 시점 di '이전 단계'의 실제값 기준선 (요구 #6 '이전 단계 대비').

    step 있으면 직전 process step 의 평균, 없으면 피크 직전 윈도우 평균.
    """
    if step is not None and len(step):
        s = np.asarray(step).astype(int)
        cur = int(s[di])
        prev = np.where(s == cur - 1)[0]
        if prev.size:
            return float(actual_col[prev].mean())
    a = max(0, di - half)
    return float(actual_col[a:di].mean()) if di > a else float(actual_col[max(0, di - 1)])


def _classify(exp_delta: float, act_delta: float, level_peak: float,
              sig: float, eps: float = 0.3) -> str:
    """응답 분류.

    level_peak < sig  → '정상'. 여기서 level은 std z가 아니라
    |실제-기대| / OK raw residual 분위수 기준이다.
    level_peak ≥ sig  → 실제 이탈. IO 명령(기대Δ) 대비 어떻게 어긋났나로 세분:
        IO무관이탈 / 역방향응답 / 무응답 / 약응답 / 오프셋이탈 / 과응답
    """
    if level_peak < sig:
        return "정상"                            # 잔차 작음 = 이탈 아님 (IO 응답 양호)
    # 이탈(level≥sig) 인 센서만 아래로 — '정상응답' 으로 안 빠짐
    if abs(exp_delta) < eps:
        return "IO무관이탈"                       # IO 변화 미미한데 센서가 이탈
    ratio = act_delta / exp_delta               # 부호 포함 비율
    if ratio < -0.1:
        return "역방향응답"                      # 명령과 반대로 움직임
    if ratio < 0.4:
        return "무응답"                          # 거의 안 움직임(<40%)
    if ratio < 0.75:
        return "약응답"
    if ratio <= 1.3:
        return "오프셋이탈"                      # 변화량은 따라갔으나 절대값(기준선) 어긋나 잔차 큼
    return "과응답"


def _rank_io_scores(A, scores, top_k: int = 2):
    n_io = min(len(A.io_cols), len(scores))
    if n_io <= 0:
        return []
    vals = np.asarray(scores[:n_io], dtype=np.float64)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmax(vals)) <= 0:
        return []
    order = np.argsort(vals)[::-1]
    out = []
    for j in order:
        v = float(vals[int(j)])
        if v <= 0:
            continue
        out.append((A.io_cols[int(j)], int(j), v))
        if len(out) >= int(top_k):
            break
    return out


def _io_drivers(A, rec, sensor_idx: int, at_t: Optional[int] = None, top_k: int = 2):
    """이 센서 이탈 시점과 가장 관련 있어 보이는 IO 채널 top-k.

    모델에 gradient 기반 io_attribution이 있으면 그 값을 쓰고,
    없으면 피크 시점 주변의 IO
    transition 크기와 open 상태를 함께 보아 가장 관련 있어 보이는 IO를 고른다.
    """
    source_rec = rec.get("_source_rec") if isinstance(rec, dict) else None
    if source_rec is not None:
        factor = max(1, int(rec.get("_interp_factor", 1) or 1))
        raw_t = None if at_t is None else int(round(float(at_t) / float(factor)))
        return _io_drivers(A, source_rec, sensor_idx, at_t=raw_t, top_k=top_k)

    if not A.io_cols:
        return []
    if hasattr(A, "io_attribution"):
        attr = A.io_attribution(rec, sensor_idx, at_t=at_t)
        return _rank_io_scores(A, attr, top_k=top_k)

    io_arr = np.asarray(rec.get("io", np.zeros((0, 0))), dtype=np.float64)
    if io_arr.ndim != 2 or io_arr.shape[1] == 0:
        return []
    T = io_arr.shape[0]
    t0 = int(at_t) if at_t is not None else T // 2
    half = max(2, int(getattr(A, "future_window", 20)))
    lo, hi = max(0, t0 - half), min(T, t0 + half + 1)
    dio = np.abs(np.diff(io_arr, axis=0, prepend=io_arr[:1]))
    local_transition = dio[lo:hi].max(axis=0) if hi > lo else np.zeros(io_arr.shape[1])
    local_state = np.abs(io_arr[lo:hi]).mean(axis=0) if hi > lo else np.zeros(io_arr.shape[1])
    score = local_transition + 0.1 * local_state
    if not np.isfinite(score).any() or float(np.nanmax(score)) <= 0:
        score = dio.max(axis=0)
    if not np.isfinite(score).any() or float(np.nanmax(score)) <= 0:
        return []
    return _rank_io_scores(A, score, top_k=top_k)


def _io_driver(A, rec, sensor_idx: int, at_t: Optional[int] = None):
    drivers = _io_drivers(A, rec, sensor_idx, at_t=at_t, top_k=1)
    if not drivers:
        return None, None
    return drivers[0][0], drivers[0][1]


def _plot_io_drivers(ax, t, io_plot, drivers, fontsize: int = 8) -> None:
    if not drivers:
        return
    io_plot = np.asarray(io_plot, dtype=np.float64)
    if io_plot.ndim != 2 or io_plot.shape[1] == 0:
        return
    if io_plot.shape[0] != len(t):
        return
    ax2 = ax.twinx()
    styles = [
        {"lw": 1.15, "alpha": 0.58, "ls": "-"},
        {"lw": 0.95, "alpha": 0.30, "ls": "--"},
    ]
    shown = []
    for rank, (io_name, io_j, _) in enumerate(drivers[:2], 1):
        if io_j is None or int(io_j) >= io_plot.shape[1]:
            continue
        st = styles[min(rank - 1, len(styles) - 1)]
        ax2.step(t, io_plot[:, int(io_j)], where="pre", color="#9467bd",
                 lw=st["lw"], alpha=st["alpha"], ls=st["ls"], zorder=1)
        shown.append(f"{rank}:{io_name}")
    if not shown:
        return
    ax2.set_ylabel("IO[" + " / ".join(shown) + "]", color="#9467bd", fontsize=fontsize)
    ax2.tick_params(axis="y", labelcolor="#9467bd", labelsize=max(7, fontsize - 1))


def sensor_finding(A, rec, sensor_idx: int, actual=None, expected=None, z=None) -> Dict:
    """한 센서의 인과 소견: 기대 Δ vs 실제 Δ + IO 드라이버 (요구 #6 핵심 수치).

    기준점 = 이전 단계의 *실제* 값으로 통일 → 잔차 = 기대Δ − 실제Δ (= z 와 일치).
    """
    if actual is None or expected is None or z is None:
        actual_full = _actual(A, rec)
        expected_full = A.predict(rec)
        z_full = (A.error_level(rec) if hasattr(A, "error_level")
                  else np.abs(A.z_residual(rec)))
    else:
        actual_full = np.asarray(actual, dtype=np.float64)
        expected_full = np.asarray(expected, dtype=np.float64)
        z_full = np.asarray(z, dtype=np.float64)
    actual = actual_full[:, sensor_idx]                # (T,) 실측
    expected = expected_full[:, sensor_idx]            # (T,) 모델 기대
    z = z_full[:, sensor_idx]
    step = rec.get("step")
    di = int(np.argmax(z))
    base = _prev_baseline(actual, step, di)            # 이전 단계 실제 기준선
    exp_delta = float(expected[di] - base)             # IO 명령상 기대된 변화
    act_delta = float(actual[di] - base)               # 실제 변화
    peak_step = int(step[di]) if step is not None and len(step) else None
    io_name, _ = _io_driver(A, rec, sensor_idx, at_t=di)   # 피크 시점 정렬
    follow = (act_delta / exp_delta) if abs(exp_delta) > 1e-6 else None
    sig = float(getattr(A.cfg, "raw_error_level", 1.0))         # raw error level 기준
    return {
        "sensor": A.sensor_cols[sensor_idx],
        "peak_t": di, "peak_step": peak_step,
        "peak_t_original": round(_source_time_for_rec(rec, di), 6),
        "interpolation": max(1, int(rec.get("_interp_factor", 1) or 1)),
        "z": round(float(z[di]), 2),
        "error_level": round(float(z[di]), 2),
        "expected_delta": round(exp_delta, 3),
        "actual_delta": round(act_delta, 3),
        "follow_ratio": (round(follow, 2) if follow is not None else None),
        "response": _classify(exp_delta, act_delta, float(z[di]), sig),
        "io_driver": io_name,
        "dir": ("상승" if exp_delta >= 0 else "하강"),
    }


def narrate_recon(A, rec, top_k: int = 3, pre=None) -> Dict:
    """wafer 인과 내러티브 — 상위 이탈 센서들의 'IO 명령 대비 무응답' 서술."""
    if pre is None:
        actual, expected, z = _compute(A, rec)
    else:
        actual, expected, z = pre
    order = [int(j) for j in np.argsort(z.sum(axis=0))[::-1][:top_k]]
    findings = [sensor_finding(A, rec, j, actual=actual, expected=expected, z=z)
                for j in order]
    lines = []
    for f in findings:
        io = f["io_driver"] or "관련 IO"
        if f["response"] == "IO무관이탈":
            # IO 변화가 미미한 구간 → 비율은 무의미(헛수치 방지), 잔차만 보고
            lines.append(f"{f['sensor']}: 스텝 {f['peak_step']}에서 IO 변화 미미 "
                         f"(기대Δ {f['expected_delta']:+.2f}) — IO 무관 이탈 "
                         f"기준 {f['error_level']}배")
        else:
            pct = (f"(기대의 {int(f['follow_ratio'] * 100)}%)"
                   if f["follow_ratio"] is not None else "")
            lines.append(
                f"{f['sensor']}: 스텝 {f['peak_step']}에서 IO[{io}]가 {f['dir']} "
                f"{f['expected_delta']:+.2f} 명령 → 실제 {f['actual_delta']:+.2f} {pct} "
                f"[{f['response']}] 기준 {f['error_level']}배")
    return {"wafer_id": rec["wafer_id"], "findings": findings,
            "narrative": " / ".join(lines) if lines else "특이 센서 없음"}


def _compute(A, rec):
    """wafer 1회 계산 → (actual, expected, z). predict 1번만(효율)."""
    actual = _actual(A, rec)
    expected = A.predict(rec)
    if hasattr(A, "error_level"):
        z = A.error_level(rec)
    else:
        res = actual - expected
        rs = A.resid_std
        if (getattr(A, "_per_step", False) and rec.get("step") is not None
                and getattr(rs, "ndim", 1) == 2):
            st = np.clip(np.asarray(rec["step"]).astype(int), 0, rs.shape[0] - 1)
            z = np.abs(res / rs[st])
        else:
            z = np.abs(res / rs)
    return actual, expected, z


def _explain_interp_factor(cfg) -> int:
    try:
        return max(1, int(getattr(cfg, "explain_interpolation", 1) or 1))
    except Exception:
        return 1


def _interp_grid(n: int, factor: int) -> np.ndarray:
    n = int(n)
    factor = max(1, int(factor))
    if n <= 1 or factor <= 1:
        return np.arange(max(n, 0), dtype=np.float64)
    return np.linspace(0.0, float(n - 1), int((n - 1) * factor + 1))


def _interp_continuous(arr, factor: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        return arr
    n = int(arr.shape[0])
    factor = max(1, int(factor))
    if n <= 1 or factor <= 1:
        return arr.copy()
    old_x = np.arange(n, dtype=np.float64)
    new_x = _interp_grid(n, factor)
    flat = arr.reshape(n, -1)
    out = np.empty((len(new_x), flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        finite = np.isfinite(col)
        if finite.sum() == 0:
            out[:, j] = 0.0
        elif finite.sum() == 1:
            out[:, j] = float(col[finite][0])
        else:
            out[:, j] = np.interp(new_x, old_x[finite], col[finite])
    return out.reshape((len(new_x),) + arr.shape[1:])


def _interp_hold(arr, factor: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr
    n = int(arr.shape[0])
    factor = max(1, int(factor))
    if n <= 1 or factor <= 1:
        return arr.copy()
    new_x = _interp_grid(n, factor)
    idx = np.floor(new_x + 1e-12).astype(int)
    idx = np.clip(idx, 0, n - 1)
    return arr[idx].copy()


def _with_interpolated_explain_arrays(A, rec, actual, expected, z):
    """Return rec/arrays for high-resolution explain outputs.

    Continuous values are linearly interpolated. Discrete actuator/step states
    are held until the next original time point so IO open/close and ProcessStep
    boundaries do not become artificial ramps.
    """
    factor = _explain_interp_factor(A.cfg)
    if factor <= 1 or actual.shape[0] <= 1:
        return rec, actual, expected, z

    rec_i = dict(rec)
    rec_i["_source_rec"] = rec
    rec_i["_interp_factor"] = factor
    rec_i["_interp_source_t"] = _interp_grid(actual.shape[0], factor)
    if rec.get("sensors") is not None:
        rec_i["sensors"] = _interp_continuous(rec["sensors"], factor)
    if rec.get("io") is not None:
        rec_i["io"] = _interp_hold(rec["io"], factor)
    if rec.get("step") is not None:
        rec_i["step"] = _interp_hold(rec["step"], factor).astype(np.asarray(rec["step"]).dtype)
    return (
        rec_i,
        _interp_continuous(actual, factor),
        _interp_continuous(expected, factor),
        _interp_continuous(z, factor),
    )


def _source_time_for_rec(rec, t_index: int) -> float:
    factor = max(1, int(rec.get("_interp_factor", 1) or 1)) if isinstance(rec, dict) else 1
    if factor <= 1:
        return float(t_index)
    grid = rec.get("_interp_source_t") if isinstance(rec, dict) else None
    if grid is not None and 0 <= int(t_index) < len(grid):
        return float(np.asarray(grid, dtype=np.float64)[int(t_index)])
    return float(t_index) / float(factor)


def _time_axis_label(rec) -> str:
    factor = max(1, int(rec.get("_interp_factor", 1) or 1)) if isinstance(rec, dict) else 1
    if factor <= 1:
        return "Time point (ProcessStep 0 excluded)"
    return f"Interpolated time point x{factor} (ProcessStep 0 excluded)"


def _threshold_mode(threshold) -> str:
    if isinstance(threshold, dict) and threshold.get("threshold_wise"):
        return str(threshold.get("threshold_wise")).lower()
    if isinstance(threshold, dict):
        return "recipe"
    return "all"


def _all_threshold_value(threshold) -> float | None:
    """Return the scalar wafer-score threshold only for threshold-wise all."""
    if threshold is None:
        return None
    if isinstance(threshold, dict):
        if _threshold_mode(threshold) != "all":
            return None
        for key in (
                "threshold_raw", "threshold",
                "global_threshold_raw", "global_threshold",
                "default", "global"):
            if key not in threshold:
                continue
            value = threshold.get(key)
            if isinstance(value, dict):
                continue
            try:
                value = float(value)
            except Exception:
                continue
            return value if np.isfinite(value) else None
        return None
    try:
        value = float(threshold)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _parse_threshold_recipe_multiplier_overrides(values) -> dict[str, float]:
    if not values:
        return {}
    if isinstance(values, dict):
        iterable = [f"{k}={v}" for k, v in values.items()]
    elif isinstance(values, str):
        iterable = [values]
    else:
        iterable = list(values)
    out = {}
    for value in iterable:
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
                    continue
                parts.append(item.split(sep, 1))
        for recipe, factor in parts:
            recipe = str(recipe).strip()
            if not recipe:
                continue
            try:
                factor = float(factor)
            except Exception:
                continue
            if np.isfinite(factor) and factor > 0:
                out[recipe] = factor
    return out


def _base_threshold_for_recipe_before_multiplier(threshold, recipe: str) -> float | None:
    rec = {"recipe": recipe}
    if not isinstance(threshold, dict):
        try:
            value = float(threshold)
        except Exception:
            return None
        return value if np.isfinite(value) and value > 1e-12 else None
    base = _lookup_recipe_value(threshold.get("base_threshold_by_recipe"), recipe)
    if base is not None:
        base = _float_threshold_value(base)
        if base is not None and base > 1e-12:
            return base
    current = _threshold_for_record(threshold, rec)
    current = _float_threshold_value(current)
    if current is None or current <= 1e-12:
        return None
    factor = _threshold_recipe_multiplier(threshold, recipe)
    if abs(factor - 1.0) > 1e-12:
        base = current / factor
        return base if np.isfinite(base) and base > 1e-12 else current
    return current


def _threshold_with_explain_recipe_multipliers(threshold, multipliers):
    """Apply tcn-explain CLI recipe multipliers without rewriting search artifacts."""
    overrides = _parse_threshold_recipe_multiplier_overrides(multipliers)
    if not overrides or threshold is None:
        return threshold, {}
    if isinstance(threshold, dict) and _threshold_mode(threshold) in {"sensor", "step"}:
        return threshold, {}

    import copy as _copy
    out = _copy.deepcopy(threshold) if isinstance(threshold, dict) else {
        "threshold_wise": "all",
        "threshold": round(float(threshold), 4),
        "threshold_raw": float(threshold),
        "global_threshold": round(float(threshold), 4),
        "global_threshold_raw": float(threshold),
    }
    out.setdefault("threshold_wise", _threshold_mode(out))
    threshold_by_recipe = dict(out.get("threshold_by_recipe") or {})
    base_by_recipe = dict(out.get("base_threshold_by_recipe") or {})
    merged_multipliers = dict(out.get("threshold_recipe_multipliers") or {})
    explain_multipliers = {}

    for recipe, factor in overrides.items():
        base = _base_threshold_for_recipe_before_multiplier(out, recipe)
        if base is None:
            continue
        adjusted = float(base) * float(factor)
        threshold_by_recipe[recipe] = adjusted
        base_by_recipe[recipe] = float(base)
        merged_multipliers[recipe] = float(factor)
        explain_multipliers[recipe] = float(factor)

    if not explain_multipliers:
        return threshold, {}
    out["threshold_by_recipe"] = threshold_by_recipe
    out["base_threshold_by_recipe"] = base_by_recipe
    out["threshold_recipe_multipliers"] = merged_multipliers
    out["explain_threshold_recipe_multipliers"] = explain_multipliers
    out["display_scale_uses_adjusted_threshold"] = True
    strategy = str(out.get("threshold_strategy") or _threshold_mode(out))
    if "explain_recipe_threshold_multipliers" not in strategy:
        out["threshold_strategy"] = strategy + "_with_explain_recipe_threshold_multipliers"
    return out, explain_multipliers


def _sensor_step_threshold(threshold, rec, sensor: str, step_value) -> float | None:
    if threshold is None:
        return None
    mode = _threshold_mode(threshold)
    recipe = _recipe_key(rec)
    if mode == "step":
        table = threshold.get("thresholds") or {}
        th = (((table.get(recipe) or {}).get(str(sensor)) or {}).get(str(int(step_value))))
        if th is None:
            th = threshold.get("default_threshold")
        return None if th is None else float(th)
    if mode == "sensor":
        table = threshold.get("thresholds") or {}
        th = (table.get(recipe) or {}).get(str(sensor))
        if th is None:
            th = threshold.get("default_threshold")
        return None if th is None else float(th)
    if mode == "recipe":
        return _threshold_for_record(threshold, rec)
    return float(threshold)


def _step_accum_series_from_level(level_col, step, threshold_base: float = 1.0):
    level_col = np.asarray(level_col, dtype=np.float64)
    excess = np.maximum(0.0, level_col - float(threshold_base))
    if step is not None and len(step) == len(excess):
        step_arr = np.asarray(step).astype(int)
    else:
        step_arr = np.ones(len(excess), dtype=int)
    out = np.zeros(len(excess), dtype=np.float64)
    last_step = None
    running = 0.0
    for i, st in enumerate(step_arr):
        st = int(st)
        if st == 0:
            running = 0.0
            last_step = st
            out[i] = 0.0
            continue
        if last_step is None or st != last_step:
            running = 0.0
        running += float(excess[i])
        out[i] = running
        last_step = st
    return out


def _sensor_accum_series_from_level(level_col, step, threshold_base: float = 1.0):
    level_col = np.asarray(level_col, dtype=np.float64)
    excess = np.maximum(0.0, level_col - float(threshold_base))
    if step is not None and len(step) == len(excess):
        keep = np.asarray(step).astype(int) != 0
    else:
        keep = np.ones(len(excess), dtype=bool)
    out = np.zeros(len(excess), dtype=np.float64)
    running = 0.0
    for i, ok in enumerate(keep):
        if not ok:
            out[i] = 0.0
            continue
        running += float(excess[i])
        out[i] = running
    return out


def _sensor_reconstruction_accum_series_from_level(level_col, step):
    return _sensor_accum_series_from_level(level_col, step, threshold_base=0.0)


def _error_accum_mode(cfg) -> str:
    mode = str(getattr(cfg, "error_accum_mode", "sensor") or "sensor").lower()
    return mode if mode in {"sensor", "step"} else "sensor"


def _accum_series_from_level(level_col, step, cfg, threshold_base: float = 1.0):
    if _error_accum_mode(cfg) == "step":
        return _step_accum_series_from_level(level_col, step, threshold_base=threshold_base)
    return _sensor_accum_series_from_level(level_col, step, threshold_base=threshold_base)


def _threshold_series_for_sensor(threshold, rec, sensor: str, step_plot, n: int):
    if threshold is None:
        return None
    if step_plot is None or len(step_plot) != n:
        th = _sensor_step_threshold(threshold, rec, sensor, 1)
        return None if th is None else np.full(n, float(th), dtype=np.float64)
    vals = []
    for st in np.asarray(step_plot).astype(int):
        th = _sensor_step_threshold(threshold, rec, sensor, int(st))
        vals.append(np.nan if th is None else float(th))
    arr = np.asarray(vals, dtype=np.float64)
    return None if np.isnan(arr).all() else arr


def _decision_scores_from_threshold(A, records, threshold) -> np.ndarray:
    mode = _threshold_mode(threshold)
    out = []
    records = list(records or [])
    levels = _error_levels_for_records(A, records)
    for rec, level in zip(records, levels):
        step = rec.get("step")
        step_arr = np.asarray(step).astype(int) if step is not None and len(step) == level.shape[0] \
            else np.ones(level.shape[0], dtype=int)
        excess = np.maximum(0.0, np.asarray(level, dtype=np.float64) - 1.0)
        best = 0.0
        for st in sorted(int(x) for x in np.unique(step_arr) if int(x) != 0):
            mask = step_arr == st
            vals = excess[mask].sum(axis=0)
            for j, sensor in enumerate(A.sensor_cols):
                score = float(vals[j])
                th = _sensor_step_threshold(threshold, rec, str(sensor), st)
                if th is None:
                    continue
                if mode in {"sensor", "step"}:
                    best = max(best, score / max(float(th), 1e-12))
                else:
                    best = max(best, score)
        out.append(best)
    return np.asarray(out, dtype=np.float64)


def _sum_all_sensor_reconstruction_score_from_level(level) -> float:
    level = np.asarray(level, dtype=np.float64)
    if level.ndim != 2 or level.size == 0:
        return 0.0
    level = np.nan_to_num(level, nan=0.0, posinf=0.0, neginf=0.0)
    per_sensor = np.maximum(0.0, level).sum(axis=0)
    return float(per_sensor.sum())


def _score_agg_enabled_from_cfg(cfg) -> bool:
    return bool(getattr(cfg, "score_agg_enabled", False)) if cfg is not None else False


def _score_resid_agg_from_cfg(cfg) -> str:
    if not _score_agg_enabled_from_cfg(cfg):
        return "sum"
    mode = str(getattr(cfg, "resid_agg", "sum") or "sum").strip().lower()
    return mode if mode in {"sum", "p99", "max", "mean"} else "sum"


def _score_wafer_agg_from_cfg(cfg) -> str:
    if not _score_agg_enabled_from_cfg(cfg):
        return "sum"
    mode = str(getattr(cfg, "wafer_agg", "sum") or "sum").strip().lower()
    return mode if mode in {"sum", "topk_mean", "max", "mean"} else "sum"


def _score_topk_from_cfg(cfg) -> int:
    try:
        value = int(getattr(cfg, "wafer_topk", 0) or 0)
    except Exception:
        value = 0
    return max(0, value)


def _score_time_avg_from_cfg(cfg) -> bool:
    return bool(getattr(cfg, "score_time_avg", False)) if cfg is not None else False


def _score_final_avg_from_cfg(cfg) -> bool:
    return bool(getattr(cfg, "score_final_avg", False)) if cfg is not None else False


def _score_kind_from_cfg(cfg) -> str:
    topk = _score_topk_from_cfg(cfg)
    if _score_agg_enabled_from_cfg(cfg):
        base = (
            f"resid_{_score_resid_agg_from_cfg(cfg)}"
            f"_wafer_{_score_wafer_agg_from_cfg(cfg)}"
            f"_top{topk}_sensor_reconstruction_error"
        )
        if _score_time_avg_from_cfg(cfg):
            return f"time_avg_{base}"
        if _score_final_avg_from_cfg(cfg):
            return f"final_avg_{base}"
        return base
    if topk > 0:
        base = f"sum_top{topk}_sensor_reconstruction_error"
    else:
        base = "sum_all_sensor_reconstruction_error"
    if _score_time_avg_from_cfg(cfg):
        return f"time_avg_{base}"
    if _score_final_avg_from_cfg(cfg):
        return f"final_avg_{base}"
    return base


def _score_metadata_for_cfg(cfg) -> dict:
    return {
        "score_kind": _score_kind_from_cfg(cfg),
        "score_topk": _score_topk_from_cfg(cfg),
        "score_agg_enabled": _score_agg_enabled_from_cfg(cfg),
        "score_resid_agg": _score_resid_agg_from_cfg(cfg),
        "score_wafer_agg": _score_wafer_agg_from_cfg(cfg),
        "score_time_avg": _score_time_avg_from_cfg(cfg),
        "score_final_avg": _score_final_avg_from_cfg(cfg),
    }


def _bool_from_payload(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _score_topk_from_threshold(threshold, cfg=None) -> int:
    if isinstance(threshold, dict):
        if "score_topk" not in threshold:
            return _score_topk_from_cfg(cfg)
        try:
            return max(0, int(threshold.get("score_topk", 0) or 0))
        except Exception:
            return _score_topk_from_cfg(cfg)
    return _score_topk_from_cfg(cfg)


def _score_topk_from_score_kind(score_kind) -> int | None:
    kind = str(score_kind or "").strip().lower()
    if kind.startswith("time_avg_"):
        kind = kind[len("time_avg_"):]
    if kind.startswith("final_avg_"):
        kind = kind[len("final_avg_"):]
    prefix = "sum_top"
    suffix = "_sensor_reconstruction_error"
    if kind.startswith(prefix) and suffix in kind:
        value = kind[len(prefix):kind.find(suffix)]
        try:
            return max(0, int(value))
        except Exception:
            return None
    if "sum_all_sensor_reconstruction_error" in kind:
        return 0
    marker = "_top"
    if kind.startswith("resid_") and marker in kind and suffix in kind:
        start = kind.rfind(marker) + len(marker)
        end = kind.find(suffix, start)
        value = kind[start:end]
        if value == "all":
            return 0
        try:
            return max(0, int(value))
        except Exception:
            return None
    return None


def _score_agg_enabled_from_threshold(threshold=None, cfg=None) -> bool:
    if isinstance(threshold, dict):
        if "score_agg_enabled" in threshold:
            return _bool_from_payload(threshold.get("score_agg_enabled"))
        if threshold.get("score_resid_agg") or threshold.get("score_wafer_agg"):
            return True
    return _score_agg_enabled_from_cfg(cfg)


def _score_resid_agg_from_threshold(threshold=None, cfg=None) -> str:
    if isinstance(threshold, dict) and threshold.get("score_resid_agg"):
        mode = str(threshold.get("score_resid_agg") or "sum").strip().lower()
        return mode if mode in {"sum", "p99", "max", "mean"} else "sum"
    return _score_resid_agg_from_cfg(cfg)


def _score_wafer_agg_from_threshold(threshold=None, cfg=None) -> str:
    if isinstance(threshold, dict) and threshold.get("score_wafer_agg"):
        mode = str(threshold.get("score_wafer_agg") or "sum").strip().lower()
        return mode if mode in {"sum", "topk_mean", "max", "mean"} else "sum"
    return _score_wafer_agg_from_cfg(cfg)


def _score_time_avg_from_score_kind(score_kind) -> bool | None:
    kind = str(score_kind or "").strip().lower()
    if kind.startswith("time_avg_"):
        return True
    if "sensor_reconstruction_error" in kind:
        return False
    return None


def _score_final_avg_from_score_kind(score_kind) -> bool | None:
    kind = str(score_kind or "").strip().lower()
    if kind.startswith("final_avg_"):
        return True
    if "sensor_reconstruction_error" in kind:
        return False
    return None


def _score_time_avg_from_threshold(threshold=None, cfg=None) -> bool:
    if isinstance(threshold, dict):
        if "score_time_avg" in threshold:
            return _bool_from_payload(threshold.get("score_time_avg"))
        parsed = _score_time_avg_from_score_kind(
            threshold.get("score_kind") or threshold.get("score_name"))
        if parsed is not None:
            return bool(parsed)
    return _score_time_avg_from_cfg(cfg)


def _score_final_avg_from_threshold(threshold=None, cfg=None) -> bool:
    if isinstance(threshold, dict):
        if "score_final_avg" in threshold:
            return _bool_from_payload(threshold.get("score_final_avg"))
        parsed = _score_final_avg_from_score_kind(
            threshold.get("score_kind") or threshold.get("score_name"))
        if parsed is not None:
            return bool(parsed)
    return _score_final_avg_from_cfg(cfg)


def _score_topk_from_saved_metadata(threshold=None, score_kind=None, *payloads) -> int | None:
    for obj in (threshold, *payloads):
        if isinstance(obj, dict) and "score_topk" in obj:
            try:
                return max(0, int(obj.get("score_topk", 0) or 0))
            except Exception:
                pass
    for obj in (threshold, *payloads):
        if isinstance(obj, dict):
            parsed = _score_topk_from_score_kind(obj.get("score_kind") or obj.get("score_name"))
            if parsed is not None:
                return parsed
    return _score_topk_from_score_kind(score_kind)


def _score_agg_metadata_from_score_kind(score_kind) -> dict:
    kind = str(score_kind or "").strip().lower()
    score_time_avg = False
    score_final_avg = False
    if kind.startswith("time_avg_"):
        score_time_avg = True
        kind = kind[len("time_avg_"):]
    if kind.startswith("final_avg_"):
        score_final_avg = True
        kind = kind[len("final_avg_"):]
    suffix = "_sensor_reconstruction_error"
    if kind.startswith("resid_") and "_wafer_" in kind and suffix in kind:
        wafer_pos = kind.find("_wafer_")
        top_pos = kind.rfind("_top")
        if top_pos > wafer_pos:
            resid = kind[len("resid_"):wafer_pos]
            wafer = kind[wafer_pos + len("_wafer_"):top_pos]
            if resid in {"sum", "p99", "max", "mean"} \
                    and wafer in {"sum", "topk_mean", "max", "mean"}:
                return {
                    "score_agg_enabled": True,
                    "score_resid_agg": resid,
                    "score_wafer_agg": wafer,
                    "score_time_avg": score_time_avg,
                    "score_final_avg": score_final_avg,
                }
    if kind.startswith("sum_top") or "sum_all_sensor_reconstruction_error" in kind:
        return {
            "score_agg_enabled": False,
            "score_resid_agg": "sum",
            "score_wafer_agg": "sum",
            "score_time_avg": score_time_avg,
            "score_final_avg": score_final_avg,
        }
    return {}


def _score_metadata_from_saved(threshold=None, score_kind=None, *payloads) -> dict:
    meta = {}
    parsed = _score_agg_metadata_from_score_kind(score_kind)
    if parsed:
        meta.update(parsed)
    topk = _score_topk_from_saved_metadata(threshold, score_kind, *payloads)
    if topk is not None:
        meta["score_topk"] = int(topk)
    for obj in (threshold, *payloads):
        if not isinstance(obj, dict):
            continue
        kind = obj.get("score_kind") or obj.get("score_name")
        parsed_obj = _score_agg_metadata_from_score_kind(kind)
        for key, value in parsed_obj.items():
            meta.setdefault(key, value)
        if "score_topk" in obj:
            try:
                meta["score_topk"] = max(0, int(obj.get("score_topk", 0) or 0))
            except Exception:
                pass
        if "score_agg_enabled" in obj:
            meta["score_agg_enabled"] = _bool_from_payload(obj.get("score_agg_enabled"))
        if obj.get("score_resid_agg"):
            mode = str(obj.get("score_resid_agg")).strip().lower()
            if mode in {"sum", "p99", "max", "mean"}:
                meta["score_resid_agg"] = mode
        if obj.get("score_wafer_agg"):
            mode = str(obj.get("score_wafer_agg")).strip().lower()
            if mode in {"sum", "topk_mean", "max", "mean"}:
                meta["score_wafer_agg"] = mode
        if "score_time_avg" in obj:
            meta["score_time_avg"] = _bool_from_payload(obj.get("score_time_avg"))
        if "score_final_avg" in obj:
            meta["score_final_avg"] = _bool_from_payload(obj.get("score_final_avg"))
    if meta.get("score_agg_enabled") is False:
        meta.setdefault("score_resid_agg", "sum")
        meta.setdefault("score_wafer_agg", "sum")
    meta.setdefault("score_time_avg", False)
    meta.setdefault("score_final_avg", False)
    return meta


def _score_metadata_matches(saved: dict, current: dict) -> bool:
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    required = (
        "score_topk", "score_agg_enabled", "score_resid_agg",
        "score_wafer_agg", "score_time_avg", "score_final_avg",
    )
    if any(key not in saved for key in required):
        return False
    try:
        if int(saved["score_topk"]) != int(current["score_topk"]):
            return False
    except Exception:
        return False
    if bool(saved["score_agg_enabled"]) != bool(current["score_agg_enabled"]):
        return False
    return (
        str(saved["score_resid_agg"]).strip().lower()
        == str(current["score_resid_agg"]).strip().lower()
        and str(saved["score_wafer_agg"]).strip().lower()
        == str(current["score_wafer_agg"]).strip().lower()
        and bool(saved["score_time_avg"]) == bool(current["score_time_avg"])
        and bool(saved["score_final_avg"]) == bool(current["score_final_avg"])
    )


def _threshold_with_score_metadata(threshold, cfg):
    if not isinstance(threshold, dict):
        return threshold
    out = dict(threshold)
    out.update(_score_metadata_for_cfg(cfg))
    return out


def _format_score_metadata(meta: dict | None) -> str:
    if not isinstance(meta, dict):
        return "unknown"
    topk = meta.get("score_topk", "unknown")
    topk_label = "all" if str(topk) == "0" else str(topk)
    return (
        f"topk={topk_label}, "
        f"resid_agg={meta.get('score_resid_agg', 'unknown')}, "
        f"wafer_agg={meta.get('score_wafer_agg', 'unknown')}, "
        f"score_time_avg={bool(meta.get('score_time_avg', False))}, "
        f"score_final_avg={bool(meta.get('score_final_avg', False))}"
    )


def _apply_score_agg_metadata_to_cfg(cfg, threshold=None, *payloads) -> None:
    for obj in (threshold, *payloads):
        if not isinstance(obj, dict):
            continue
        has_agg = (
            "score_agg_enabled" in obj
            or "score_resid_agg" in obj
            or "score_wafer_agg" in obj
            or "score_time_avg" in obj
            or "score_final_avg" in obj
        )
        if not has_agg:
            continue
        cfg.score_agg_enabled = _score_agg_enabled_from_threshold(obj, cfg)
        cfg.resid_agg = _score_resid_agg_from_threshold(obj, cfg)
        cfg.wafer_agg = _score_wafer_agg_from_threshold(obj, cfg)
        cfg.score_time_avg = _score_time_avg_from_threshold(obj, cfg)
        cfg.score_final_avg = _score_final_avg_from_threshold(obj, cfg)
        return


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


def _topk_sensor_indices_from_level(level, topk: int = 0, resid_agg: str = "sum") -> np.ndarray:
    level = np.asarray(level, dtype=np.float64)
    if level.ndim != 2 or level.shape[1] == 0:
        return np.asarray([], dtype=int)
    n_sensor = int(level.shape[1])
    topk = max(0, int(topk or 0))
    if topk <= 0 or topk >= n_sensor:
        return np.arange(n_sensor, dtype=int)
    per_sensor = _per_sensor_reconstruction_scores_from_level(level, resid_agg)
    return np.asarray(np.argsort(per_sensor)[::-1][:topk], dtype=int)


def _aggregate_sensor_scores(per_sensor, wafer_agg: str = "sum", topk: int = 0) -> float:
    per_sensor = np.asarray(per_sensor, dtype=np.float64)
    if per_sensor.size == 0:
        return 0.0
    per_sensor = np.nan_to_num(per_sensor, nan=0.0, posinf=0.0, neginf=0.0)
    if int(topk or 0) > 0 and int(topk) < per_sensor.size:
        order = np.argsort(per_sensor)[::-1][:int(topk)]
        selected = per_sensor[order]
    else:
        selected = per_sensor
    if selected.size == 0:
        return 0.0
    wafer_agg = str(wafer_agg or "sum").strip().lower()
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


def _score_keep_mask_for_record(A, rec, fallback_len: int = 0):
    if hasattr(A, "_score_keep_mask"):
        try:
            keep = A._score_keep_mask(rec)
            if keep is not None:
                return keep
        except Exception:
            pass
    if not isinstance(rec, dict):
        return np.ones(max(0, int(fallback_len or 0)), dtype=bool)
    step = rec.get("step")
    if step is None:
        return np.ones(max(0, int(fallback_len or 0)), dtype=bool)
    keep = np.ones(len(step), dtype=bool)
    cfg = getattr(A, "cfg", None)
    if bool(getattr(cfg, "score_exclude_boundary_step0", True)):
        keep[np.asarray(step).astype(int) == 0] = False
    return keep


def _score_time_counts(length: int, step=None, cfg=None) -> np.ndarray:
    n = max(0, int(length or 0))
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    keep = np.ones(n, dtype=bool)
    if step is not None and len(step) == n and bool(
            getattr(cfg, "score_exclude_boundary_step0", True)):
        keep[np.asarray(step).astype(int) == 0] = False
    counts = np.cumsum(keep.astype(np.float64))
    counts[counts < 1.0] = 1.0
    return counts


def _score_final_counts(length: int, step=None, cfg=None) -> np.ndarray:
    counts = _score_time_counts(length, step=step, cfg=cfg)
    if counts.size == 0:
        return counts
    final = float(counts[-1]) if np.isfinite(counts[-1]) else 1.0
    final = max(1.0, final)
    return np.full_like(counts, final, dtype=np.float64)


def _score_average_curve(curve, step=None, cfg=None,
                         score_time_avg: bool = False,
                         score_final_avg: bool = False) -> np.ndarray:
    arr = np.asarray(curve, dtype=np.float64)
    if not score_time_avg and not score_final_avg:
        return arr.copy()
    counts_fn = _score_final_counts if score_final_avg else _score_time_counts
    if arr.ndim == 1:
        denom = counts_fn(arr.shape[0], step=step, cfg=cfg)
        return arr / denom if denom.size else arr.copy()
    if arr.ndim == 2:
        denom = counts_fn(arr.shape[1], step=step, cfg=cfg)
        return arr / denom[None, :] if denom.size else arr.copy()
    return arr.copy()


def _reconstruction_score_from_level(level, topk: int = 0,
                                     resid_agg: str = "sum",
                                     wafer_agg: str = "sum",
                                     score_time_avg: bool = False,
                                     score_final_avg: bool = False,
                                     valid_steps=None) -> float:
    per_sensor = _per_sensor_reconstruction_scores_from_level(level, resid_agg)
    score = _aggregate_sensor_scores(per_sensor, wafer_agg, topk)
    if score_time_avg or score_final_avg:
        score /= _score_time_denominator(level, valid_steps=valid_steps)
    return float(score)


def _sum_topk_sensor_reconstruction_score_from_level(level, topk: int = 0) -> float:
    return _reconstruction_score_from_level(level, topk, "sum", "sum")


def _error_levels_for_records(A, records) -> list[np.ndarray]:
    records = list(records or [])
    cfg = getattr(A, "cfg", None)
    try:
        pred_batch = int(getattr(cfg, "pred_batch", 1) or 1)
    except Exception:
        pred_batch = 1
    if pred_batch > 1 and hasattr(A, "error_level_batch"):
        try:
            return list(A.error_level_batch(records, pred_batch=pred_batch))
        except Exception as e:
            print(f"  ⚠ pred-batch={pred_batch} batch 예측 실패, wafer별 예측으로 대체: {e}")
    return [
        A.error_level(rec) if hasattr(A, "error_level") else np.abs(A.z_residual(rec))
        for rec in records
    ]


def _sum_all_sensor_reconstruction_scores(A, records, topk: int = None,
                                          resid_agg: str = None,
                                          wafer_agg: str = None,
                                          score_time_avg: bool = None,
                                          score_final_avg: bool = None) -> np.ndarray:
    scores = []
    records = list(records or [])
    cfg = getattr(A, "cfg", None)
    if topk is None:
        topk = _score_topk_from_cfg(cfg)
    if resid_agg is None:
        resid_agg = _score_resid_agg_from_cfg(cfg)
    if wafer_agg is None:
        wafer_agg = _score_wafer_agg_from_cfg(cfg)
    if score_time_avg is None:
        score_time_avg = _score_time_avg_from_cfg(cfg)
    if score_final_avg is None:
        score_final_avg = _score_final_avg_from_cfg(cfg)
    for level in _error_levels_for_records(A, records):
        rec = records[len(scores)]
        valid_steps = _score_keep_mask_for_record(A, rec, fallback_len=level.shape[0])
        scores.append(_reconstruction_score_from_level(
            level, topk, resid_agg, wafer_agg,
            score_time_avg=score_time_avg,
            score_final_avg=score_final_avg,
            valid_steps=valid_steps))
    return np.asarray(scores, dtype=np.float64)


def _threshold_score_kind(threshold, explicit_score_kind=None) -> str:
    if explicit_score_kind:
        return str(explicit_score_kind)
    if isinstance(threshold, dict):
        if threshold.get("score_kind"):
            return str(threshold.get("score_kind"))
        mode = str(threshold.get("threshold_wise", "")).lower()
        if mode in {"all", "recipe"}:
            return "sum_all_sensor_reconstruction_error"
        if mode in {"sensor", "step"}:
            return "step_accum_excess_reconstruction_error"
    return ""


def _uses_sum_reconstruction_threshold(threshold, explicit_score_kind=None) -> bool:
    kind = _threshold_score_kind(threshold, explicit_score_kind).lower()
    if kind:
        return "sensor_reconstruction_error" in kind and "step_accum" not in kind
    if isinstance(threshold, dict):
        return str(threshold.get("threshold_wise", "")).lower() in {"all", "recipe"}
    return threshold is not None


def _uses_step_accum_threshold(threshold, explicit_score_kind=None) -> bool:
    kind = _threshold_score_kind(threshold, explicit_score_kind).lower()
    if kind:
        return "step_accum" in kind
    if isinstance(threshold, dict):
        return str(threshold.get("threshold_wise", "")).lower() in {"sensor", "step"}
    return False


def _prediction_scores(A, records, threshold=None, score_kind=None) -> np.ndarray:
    """Return scores on the same scale as the saved threshold.

    Older artifacts used TCNForecastModel.score_records() thresholds. Newer
    threshold-wise artifacts can use step accumulated scores. Mixing those two
    scales makes blind wafers look massively over threshold.
    """
    if threshold is not None and _uses_sum_reconstruction_threshold(threshold, score_kind):
        topk = _score_topk_from_threshold(threshold, getattr(A, "cfg", None))
        resid_agg = _score_resid_agg_from_threshold(threshold, getattr(A, "cfg", None))
        wafer_agg = _score_wafer_agg_from_threshold(threshold, getattr(A, "cfg", None))
        score_time_avg = _score_time_avg_from_threshold(threshold, getattr(A, "cfg", None))
        score_final_avg = _score_final_avg_from_threshold(threshold, getattr(A, "cfg", None))
        return _sum_all_sensor_reconstruction_scores(
            A, records, topk=topk, resid_agg=resid_agg, wafer_agg=wafer_agg,
            score_time_avg=score_time_avg,
            score_final_avg=score_final_avg)
    if threshold is not None and _uses_step_accum_threshold(threshold, score_kind):
        return _decision_scores_from_threshold(A, records, threshold)
    return np.asarray(A.score_records(records), dtype=np.float64)


def _decision_threshold_for_record(threshold, rec) -> float | None:
    if threshold is None:
        return None
    return _threshold_for_record(threshold, rec)


def _filter_ok_records_below_threshold(A, ok_records, threshold=None,
                                       score_kind=None) -> tuple[list, list]:
    ok_records = list(ok_records or [])
    if not ok_records or threshold is None:
        return ok_records, []
    try:
        scores = _prediction_scores(A, ok_records, threshold=threshold,
                                    score_kind=score_kind)
    except Exception as e:
        print(f"  ⚠ train OK threshold 필터 점수 계산 실패: {e}")
        return ok_records, []
    kept, dropped = [], []
    for rec, score in zip(ok_records, np.asarray(scores, dtype=np.float64)):
        cutoff = _decision_threshold_for_record(threshold, rec)
        if cutoff is None:
            kept.append(rec)
            continue
        if np.isfinite(score) and float(score) < float(cutoff):
            kept.append(rec)
        else:
            dropped.append({
                "wafer_id": str(rec.get("wafer_id", "")),
                "recipe": _recipe_key(rec),
                "score": float(score) if np.isfinite(score) else float("nan"),
                "threshold": float(cutoff),
            })
    return kept, dropped


def _sensor_order(z, sig):
    """★ 정렬: 문제구간(z≥x) 점 수 → 피크 x 내림차순.
    → '상위 센서 = 하이라이트 있는 센서' 보장(합산 정렬의 모순 제거)."""
    n_flag = (z >= sig).sum(axis=0)
    peak = z.max(axis=0)
    return sorted(range(z.shape[1]),
                  key=lambda j: (int(n_flag[j]), float(peak[j])), reverse=True)


def _flow_sensor_order(A, indices=None):
    """Return sensor indices in process-flow order from sensors_order.json."""
    src = list(range(len(A.sensor_cols))) if indices is None else [int(i) for i in indices]
    rank = graph_sensor_rank(A.sensor_cols, A.cfg)
    fallback = len(rank) + len(src)
    return sorted(src, key=lambda j: (
        int(rank.get(str(A.sensor_cols[j]), fallback + j)),
        str(A.sensor_cols[j]),
    ))


def _sensor_family(A, name: str) -> str:
    text = str(name)
    if hasattr(A, "_is_h_sensor") and A._is_h_sensor(text):
        return "H"
    if text.startswith("p_"):
        return "P"
    if text.startswith("t_"):
        return "T"
    if text.startswith("d_"):
        return "D"
    return "OTHER"


def _score_from_level(A, level: np.ndarray, rec=None) -> float:
    """Mirror TCNForecastModel._score_record for already computed error_level."""
    cfg = getattr(A, "cfg", None)
    valid_steps = (
        _score_keep_mask_for_record(A, rec, fallback_len=np.asarray(level).shape[0])
        if rec is not None else None
    )
    return _reconstruction_score_from_level(
        level, _score_topk_from_cfg(cfg),
        _score_resid_agg_from_cfg(cfg), _score_wafer_agg_from_cfg(cfg),
        score_time_avg=_score_time_avg_from_cfg(cfg),
        score_final_avg=_score_final_avg_from_cfg(cfg),
        valid_steps=valid_steps)


def _limit_series(A, rec, sensor_idx: int, n: int) -> np.ndarray:
    """Return raw-unit OK residual tolerance for one sensor.

    This is not a standard deviation. It is the empirical OK residual quantile
    saved by the TCN model, optionally per ProcessStep.
    """
    if hasattr(A, "_limit_for_record"):
        lim = A._limit_for_record(rec)[:, sensor_idx]
    else:
        rs = np.asarray(getattr(A, "resid_std", 1.0), dtype=np.float64)
        if (getattr(A, "_per_step", False) and rec.get("step") is not None
                and getattr(rs, "ndim", 0) == 2):
            st = np.clip(np.asarray(rec["step"]).astype(int), 0, rs.shape[0] - 1)
            lim = rs[st, sensor_idx]
        elif getattr(rs, "ndim", 0) == 1 and len(rs) > sensor_idx:
            lim = np.full(n, float(rs[sensor_idx]), dtype=np.float64)
        else:
            lim = np.full(n, float(np.asarray(rs).reshape(-1)[0]), dtype=np.float64)
    lim = np.nan_to_num(lim, nan=1.0, posinf=1.0, neginf=1.0)
    lim[lim < 1e-12] = 1.0
    return lim


def _sigma_series(A, rec, sensor_idx: int, n: int) -> np.ndarray:
    """Return raw-unit OK residual standard deviation for one sensor."""
    if hasattr(A, "_sigma_for_record"):
        sig = A._sigma_for_record(rec)[:, sensor_idx]
    else:
        rs = np.asarray(getattr(A, "resid_sigma", getattr(A, "resid_std", 1.0)),
                        dtype=np.float64)
        if (getattr(A, "_per_step_sigma", False) and rec.get("step") is not None
                and getattr(rs, "ndim", 0) == 2):
            st = np.clip(np.asarray(rec["step"]).astype(int), 0, rs.shape[0] - 1)
            sig = rs[st, sensor_idx]
        elif getattr(rs, "ndim", 0) == 1 and len(rs) > sensor_idx:
            sig = np.full(n, float(rs[sensor_idx]), dtype=np.float64)
        else:
            sig = np.full(n, float(np.asarray(rs).reshape(-1)[0]), dtype=np.float64)
    sig = np.nan_to_num(sig, nan=1.0, posinf=1.0, neginf=1.0)
    sig[sig < 1e-12] = 1.0
    return sig


def _large_residual_mask(cur, exp, sigma_raw, threshold_sigma: float) -> np.ndarray:
    resid_sigma = np.abs(np.asarray(cur) - np.asarray(exp)) / np.asarray(sigma_raw)
    resid_sigma = np.nan_to_num(resid_sigma, nan=0.0, posinf=float(threshold_sigma), neginf=0.0)
    return resid_sigma, resid_sigma >= float(threshold_sigma)


def _expanded_highlight_mask(mask, score, min_width: int):
    """Expand short display highlights while preserving true event peaks.

    The anomaly decision still uses the original mask. Only the green plot
    shading is widened so one- or two-point spikes remain visible in reports.
    """
    mask = np.asarray(mask, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    n = int(mask.size)
    display = mask.copy()
    events = []
    if n == 0:
        return display, events
    width = max(1, int(min_width))
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        end = i
        local = score[start:end]
        peak = start + int(np.nanargmax(local)) if local.size else start
        d_start, d_end = start, end
        if end - start < width:
            left = width // 2
            d_start = max(0, peak - left)
            d_end = min(n, d_start + width)
            d_start = max(0, d_end - width)
            display[d_start:d_end] = True
        events.append({
            "start": start,
            "end": end,
            "display_start": d_start,
            "display_end": d_end,
            "peak": peak,
            "peak_score": float(score[peak]) if 0 <= peak < n else 0.0,
        })
    return display, events


def _event_peak_indices(events, top_k: int):
    ranked = sorted(events, key=lambda e: float(e.get("peak_score", 0.0)), reverse=True)
    return [int(e["peak"]) for e in ranked[:max(0, int(top_k))]]


def _event_band_alpha(event, threshold: float,
                      base: float = 0.10, max_alpha: float = 0.42) -> float:
    peak = float(event.get("peak_score", 0.0))
    over = max(0.0, peak - float(threshold))
    return float(min(max_alpha, base + 0.06 * over))


def _step_boundary_distance(idx: int, boundary: int) -> int:
    """Distance from a point index to the boundary between boundary-1 and boundary."""
    idx = int(idx)
    boundary = int(boundary)
    return int(min(abs(idx - boundary), abs(idx - (boundary - 1))))


def _nearest_step_boundary(idx: int, bounds, window: int) -> int | None:
    if not bounds:
        return None
    idx = int(idx)
    window = max(0, int(window))
    candidates = [
        (int(b), _step_boundary_distance(idx, int(b)))
        for b in bounds
    ]
    boundary, dist = min(candidates, key=lambda item: item[1])
    return boundary if dist <= window else None


def _step_boundary_zone(n: int, boundary: int, window: int) -> np.ndarray:
    n = max(0, int(n))
    if n == 0:
        return np.zeros(0, dtype=bool)
    idx = np.arange(n, dtype=int)
    boundary = int(boundary)
    window = max(0, int(window))
    dist = np.minimum(np.abs(idx - boundary), np.abs(idx - (boundary - 1)))
    return dist <= window


def _filter_isolated_step_transition_events(events, mask, bounds, cfg):
    """Hide only isolated residual highlights centered on a ProcessStep change.

    This is a display-only filter. A boundary event is hidden only when no other
    threshold-exceeding point exists around it outside the small transition zone.
    """
    if not bool(getattr(cfg, "suppress_isolated_step_transition_shading", False)):
        return events
    if not events or not bounds:
        return events
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.size)
    if n == 0:
        return events

    try:
        transition_window = max(0, int(getattr(cfg, "step_transition_shading_window", 2)))
    except Exception:
        transition_window = 2
    try:
        context_window = max(0, int(getattr(cfg, "step_transition_context_window", 8)))
    except Exception:
        context_window = 8
    kept = []
    for ev in events:
        peak = int(ev.get("peak", ev.get("start", 0)))
        boundary = _nearest_step_boundary(peak, bounds, transition_window)
        if boundary is None:
            kept.append(ev)
            continue
        lo = max(0, peak - context_window)
        hi = min(n, peak + context_window + 1)
        zone = _step_boundary_zone(n, boundary, transition_window)
        context_idx = np.arange(lo, hi, dtype=int)
        outside_transition = context_idx[~zone[context_idx]]
        if outside_transition.size and mask[outside_transition].any():
            kept.append(ev)
    return kept


def _draw_sigma_event_bands(ax, events, threshold: float, color="#d62728") -> None:
    """Draw full-height x-axis bands; darker red means larger sigma excess."""
    labeled = False
    for ev in events:
        start = float(ev.get("display_start", ev.get("start", 0))) - 0.5
        end = float(ev.get("display_end", ev.get("end", 0))) - 0.5
        if end <= start:
            continue
        ax.axvspan(
            start, end,
            color=color,
            alpha=_event_band_alpha(ev, threshold),
            lw=0,
            zorder=1,
            label=("4σ 초과 구간(진할수록 σ 큼)" if not labeled else None),
        )
        labeled = True


def _mark_event_peaks(ax, t, y, events, top_k: int, color="#b2182b") -> None:
    """Mark true event peaks on an axis using the same event list as the top plot."""
    peaks = _event_peak_indices(events, top_k)
    for n, ti in enumerate(peaks):
        if ti < 0 or ti >= len(y):
            continue
        label = "4σ 이벤트 peak" if n == 0 else None
        ax.scatter([t[ti]], [y[ti]], s=34, color=color, edgecolor="white",
                   lw=0.7, zorder=6, label=label)


def _diagonalize_event_rises(series, events):
    """Spread visual cumulative jumps across the red event band.

    The anomaly score and threshold logic keep using the original cumulative
    values. This only changes the plotted line so a one-point reconstruction
    error spike appears as a diagonal rise over the expanded red band.
    """
    y = np.asarray(series, dtype=np.float64).copy()
    n = int(y.size)
    if n == 0:
        return y
    for ev in sorted(events or [], key=lambda e: int(e.get("display_start", e.get("start", 0)))):
        start = max(0, min(n, int(ev.get("display_start", ev.get("start", 0)))))
        end = max(0, min(n, int(ev.get("display_end", ev.get("end", start)))))
        if end - start < 2:
            continue
        before = y[start - 1] if start > 0 else y[start]
        after = y[end - 1]
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        if after < before:
            continue
        y[start:end] = np.linspace(float(before), float(after), end - start + 1)[1:]
    return y


def _smooth_step_boundary_prediction_for_display(series, bounds, cfg):
    """Return a display-only expected line with ProcessStep boundary spikes bridged."""
    y = np.asarray(series, dtype=np.float64).copy()
    if not bool(getattr(cfg, "smooth_step_boundary_prediction", False)):
        return y
    if y.size < 4 or not bounds:
        return y
    try:
        window = max(0, int(getattr(cfg, "step_boundary_smooth_window", 2)))
    except Exception:
        window = 2
    if window <= 0:
        return y
    n = int(y.size)
    for boundary in bounds:
        boundary = int(boundary)
        start = max(0, boundary - window)
        end = min(n - 1, boundary + window - 1)
        left = start - 1
        right = end + 1
        if left < 0 or right >= n or right <= left + 1:
            continue
        if not (np.isfinite(y[left]) and np.isfinite(y[right])):
            continue
        bridge = np.linspace(float(y[left]), float(y[right]), right - left + 1)
        y[start:end + 1] = bridge[start - left:end - left + 1]
    return y


def _explain_keep_mask(A, rec, n: int) -> np.ndarray:
    if hasattr(A, "_score_keep_mask"):
        keep = np.asarray(A._score_keep_mask(rec), dtype=bool)
        if keep.shape[0] == n:
            return keep
    step = rec.get("step")
    if step is not None and len(step) == n:
        return np.asarray(step).astype(int) != 0
    return np.ones(n, dtype=bool)


def _plot_time_filter(A, rec, n: int):
    """Return filtered plot axis metadata.

    ProcessStep 0 is input context only. Explanation plots remove it entirely
    from the time axis so boundary bands and residual highlights cannot point
    to the non-evaluated leading/trailing padding steps.
    """
    keep = _explain_keep_mask(A, rec, n).copy()
    step = rec.get("step")
    step_arr = None
    if step is not None and len(step) == n:
        step_arr = np.asarray(step).astype(int)
        keep &= step_arr != 0
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


def _draw_step_guides(ax, bounds, spans, label: bool = True) -> None:
    """Draw ProcessStep boundaries and optional step labels on an axis."""
    for bx in bounds:
        ax.axvline(float(bx) - 0.5, color="#e8e8e8", lw=0.6, zorder=0)
    if not label or not spans:
        return
    trans = ax.get_xaxis_transform()
    for step, start, end in spans:
        mid = (float(start) + float(end)) / 2.0
        ax.text(mid, 0.985, f"{step}", transform=trans,
                ha="center", va="top", fontsize=7, color="#666666",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
                clip_on=True)


def _step_fraction_bins(step_plot, spans, grid: int) -> np.ndarray:
    """Map each plotted point to a fixed within-step grid bin."""
    grid = max(2, int(grid))
    step_plot = np.asarray(step_plot, dtype=int)
    bins = np.zeros(len(step_plot), dtype=int)
    for _step, start, end in spans or []:
        start, end = int(start), int(end)
        width = max(1, end - start)
        if end >= start:
            bins[start:end + 1] = np.rint(
                np.linspace(0, grid - 1, end - start + 1)
            ).astype(int)
    return np.clip(bins, 0, grid - 1)


def _step_fraction_positions(step_plot, spans, grid: int) -> np.ndarray:
    """Map each plotted point to a continuous within-step grid position."""
    grid = max(2, int(grid))
    step_plot = np.asarray(step_plot, dtype=int)
    pos = np.zeros(len(step_plot), dtype=np.float64)
    for _step, start, end in spans or []:
        start, end = int(start), int(end)
        if end >= start:
            pos[start:end + 1] = np.linspace(0.0, float(grid - 1), end - start + 1)
    return np.clip(pos, 0.0, float(grid - 1))


def _array_digest(arr) -> str:
    arr = np.asarray(arr)
    if arr.size == 0:
        return "empty"
    data = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha1(data).hexdigest()


def _ok_records_digest(records) -> str:
    rows = [
        f"{_recipe_key(r)}\t{str(r.get('wafer_id', ''))}"
        for r in (records or [])
        if isinstance(r, dict)
    ]
    rows.sort()
    return hashlib.sha1("\n".join(rows).encode("utf-8")).hexdigest()


def _ok_band_signature(A, ok_records, grid: int,
                       threshold=None, score_kind=None,
                       exclude_ok_over_threshold: bool = False) -> str:
    recipes = sorted({_recipe_key(r) for r in ok_records} | {"__global__"})
    cfg = getattr(A, "cfg", None)
    payload = {
        "version": 23,
        "sensors": [str(x) for x in A.sensor_cols],
        "n_steps_global": int(getattr(A, "n_steps_global", 0) or 0),
        "grid": int(grid),
        "n_ok": len(ok_records),
        "ok_records_sha1": _ok_records_digest(ok_records),
        "recipes": recipes,
        "resid_abs_limit_sha1": _array_digest(getattr(A, "resid_abs_limit", [])),
        "arch": getattr(A, "arch", {}),
        "score_curve": "interpolated_selected_sensor_progress_scaled_to_prediction_score",
        "score_topk": _score_topk_from_threshold(threshold, cfg),
        "score_resid_agg": _score_resid_agg_from_threshold(threshold, cfg),
        "score_wafer_agg": _score_wafer_agg_from_threshold(threshold, cfg),
        "score_time_avg": _score_time_avg_from_threshold(threshold, cfg),
        "score_final_avg": _score_final_avg_from_threshold(threshold, cfg),
        "score_band": "same_recipe_train_ok_mean_with_min_max_band",
        "sensor_curves": ["sensor_cumulative_excess", "step_cumulative_excess"],
        "threshold_mode": _threshold_mode(threshold) if threshold is not None else "",
        "threshold": threshold,
        "score_kind": "" if score_kind is None else str(score_kind),
        "exclude_ok_over_threshold": bool(exclude_ok_over_threshold),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _ok_cache_signature_payload(data) -> dict | None:
    try:
        return json.loads(str(data["signature"].item()))
    except Exception:
        return None


def _json_equiv(a, b) -> bool:
    try:
        return json.dumps(a, ensure_ascii=False, sort_keys=True, default=str) == \
            json.dumps(b, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return False


def _ok_cache_compatible_without_records(A, data, grid: int,
                                         threshold=None, score_kind=None,
                                         exclude_ok_over_threshold: bool = False) -> bool:
    """Validate an existing OK average cache when train OK CSVs are unavailable."""
    payload = _ok_cache_signature_payload(data)
    required = {
        "sensor_mean", "sensor_step_mean", "total_mean",
        "sensor_min", "sensor_max", "sensor_step_min", "sensor_step_max",
        "total_min", "total_max",
    }
    if not payload or not required.issubset(set(getattr(data, "files", []))):
        return False
    if int(payload.get("version", 0) or 0) < 23:
        return False
    if payload.get("sensors") != [str(x) for x in A.sensor_cols]:
        return False
    if int(payload.get("n_steps_global", -1)) != int(getattr(A, "n_steps_global", 0) or 0):
        return False
    if int(payload.get("grid", -1)) != max(10, int(grid)):
        return False
    if payload.get("resid_abs_limit_sha1") != _array_digest(getattr(A, "resid_abs_limit", [])):
        return False
    if not _json_equiv(payload.get("arch", {}), getattr(A, "arch", {})):
        return False
    cfg = getattr(A, "cfg", None)
    if int(payload.get("score_topk", 0) or 0) != _score_topk_from_threshold(threshold, cfg):
        return False
    if payload.get("score_resid_agg", "sum") != _score_resid_agg_from_threshold(threshold, cfg):
        return False
    if payload.get("score_wafer_agg", "sum") != _score_wafer_agg_from_threshold(threshold, cfg):
        return False
    if bool(payload.get("score_time_avg", False)) != _score_time_avg_from_threshold(threshold, cfg):
        return False
    if bool(payload.get("score_final_avg", False)) != _score_final_avg_from_threshold(threshold, cfg):
        return False
    if payload.get("threshold_mode", "") != (
            _threshold_mode(threshold) if threshold is not None else ""):
        return False
    if not _json_equiv(payload.get("threshold"), threshold):
        return False
    return (
        str(payload.get("score_kind", "")) == ("" if score_kind is None else str(score_kind))
        and bool(payload.get("exclude_ok_over_threshold", False))
        == bool(exclude_ok_over_threshold)
    )


def _cache_to_ok_bands(data) -> dict:
    out = {
        "recipes": [str(x) for x in data["recipes"].tolist()],
        "sensor_cols": [str(x) for x in data["sensor_cols"].tolist()],
        "grid": int(data["grid"].item()),
        "sensor_mean": data["sensor_mean"],
        "sensor_step_mean": data["sensor_step_mean"],
        "total_mean": data["total_mean"],
    }
    if "total_std" in data.files:
        out["total_std"] = data["total_std"]
    for key in ("total_final_min", "total_final_mean", "total_final_max"):
        if key in data.files:
            out[key] = data[key]
    for key in ("sensor_min", "sensor_max", "sensor_step_min", "sensor_step_max",
                "total_min", "total_max"):
        if key in data.files:
            out[key] = data[key]
    return out


def _add_binned_curve_mean(sum_flat, count_flat, flat, values, sumsq_flat=None) -> None:
    """Add one wafer's curve as per-bin means so long steps do not overweight it."""
    flat = np.asarray(flat, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    valid = (
        np.isfinite(values)
        & (flat >= 0)
        & (flat < int(sum_flat.size))
    )
    if not valid.any():
        return
    uniq, inv = np.unique(flat[valid], return_inverse=True)
    sums = np.zeros(len(uniq), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(sums, inv, values[valid])
    np.add.at(counts, inv, 1.0)
    means = sums / np.maximum(counts, 1.0)
    np.add.at(sum_flat, uniq, means)
    if sumsq_flat is not None:
        np.add.at(sumsq_flat, uniq, means ** 2)
    np.add.at(count_flat, uniq, 1)


def _mean_from_sum_count(sum_arr, count_arr) -> np.ndarray:
    out = np.full(sum_arr.shape, np.nan, dtype=np.float64)
    np.divide(sum_arr, count_arr, out=out, where=count_arr > 0)
    return out


def _std_from_sum_sumsq_count(sum_arr, sumsq_arr, count_arr) -> np.ndarray:
    mean = _mean_from_sum_count(sum_arr, count_arr)
    second = _mean_from_sum_count(sumsq_arr, count_arr)
    var = second - mean ** 2
    var = np.where((count_arr > 0) & np.isfinite(var), np.maximum(var, 0.0), np.nan)
    return np.sqrt(var)


def _add_grid_stats(sum_arr, count_arr, min_arr, max_arr, values,
                    sumsq_arr=None) -> None:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return
    sum_arr[valid] += values[valid]
    count_arr[valid] += 1
    min_arr[valid] = np.minimum(min_arr[valid], values[valid])
    max_arr[valid] = np.maximum(max_arr[valid], values[valid])
    if sumsq_arr is not None:
        sumsq_arr[valid] += values[valid] ** 2


def _scale_curve_to_final_score(curve, final_score) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    try:
        final = float(final_score)
    except Exception:
        return curve
    if not np.isfinite(final):
        return curve
    finite = curve[np.isfinite(curve) & (np.abs(curve) > 1e-12)]
    last = float(finite[-1]) if finite.size else 0.0
    if abs(last) <= 1e-12:
        return np.zeros_like(curve) if abs(final) <= 1e-12 else curve
    return curve * (final / last)


def _sensor_score_progress_curves_from_level(level, step,
                                             resid_agg: str = "sum",
                                             score_time_avg: bool = False,
                                             score_final_avg: bool = False,
                                             cfg=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-sensor progress curves whose final values match score aggregation."""
    level = np.asarray(level, dtype=np.float64)
    if level.ndim != 2 or level.size == 0:
        n = level.shape[0] if level.ndim else 0
        return np.zeros((0, n), dtype=np.float64), np.zeros(0, dtype=np.float64)
    n_sensor = int(level.shape[1])
    per_sensor_final = _per_sensor_reconstruction_scores_from_level(level, resid_agg)
    curves = np.vstack([
        _scale_curve_to_final_score(
            _sensor_reconstruction_accum_series_from_level(level[:, j], step),
            per_sensor_final[j] if j < per_sensor_final.size else 0.0,
        )
        for j in range(n_sensor)
    ])
    if score_time_avg or score_final_avg:
        curves = _score_average_curve(
            curves, step=step, cfg=cfg,
            score_time_avg=score_time_avg,
            score_final_avg=score_final_avg)
        per_sensor_final = (
            curves[:, -1].copy()
            if curves.ndim == 2 and curves.shape[1] > 0
            else np.zeros(n_sensor, dtype=np.float64)
        )
    return curves, per_sensor_final


def _wafer_decision_accum_series(level, step, sensor_indices=None,
                                 resid_agg: str = "sum",
                                 wafer_agg: str = "sum",
                                 score_time_avg: bool = False,
                                 score_final_avg: bool = False,
                                 cfg=None,
                                 final_score=None) -> np.ndarray:
    """Wafer score progress curve on the same scale as threshold decisions."""
    level = np.asarray(level, dtype=np.float64)
    if level.ndim != 2 or level.size == 0:
        return np.zeros(level.shape[0] if level.ndim else 0, dtype=np.float64)
    if sensor_indices is None:
        sensor_indices = np.arange(level.shape[1], dtype=int)
    sensor_indices = np.asarray(sensor_indices, dtype=int)
    sensor_indices = sensor_indices[(sensor_indices >= 0) & (sensor_indices < level.shape[1])]
    if sensor_indices.size == 0:
        return np.zeros(level.shape[0], dtype=np.float64)
    curves, _per_sensor = _sensor_score_progress_curves_from_level(
        level, step, resid_agg=resid_agg,
        score_time_avg=score_time_avg,
        score_final_avg=score_final_avg, cfg=cfg)
    selected = curves[sensor_indices]
    if selected.size == 0:
        out = np.zeros(level.shape[0], dtype=np.float64)
    else:
        wafer_agg = str(wafer_agg or "sum").strip().lower()
        if wafer_agg in {"topk_mean", "mean"}:
            out = np.mean(selected, axis=0)
        elif wafer_agg == "max":
            out = np.max(selected, axis=0)
        else:
            out = np.sum(selected, axis=0)
    if final_score is not None:
        out = _scale_curve_to_final_score(out, final_score)
    return out


def _nondecreasing_finite_curve(values) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    last = 0.0
    for i, value in enumerate(out):
        if not np.isfinite(value):
            out[i] = last
        else:
            last = max(last, float(value))
            out[i] = last
    return out


def _canonical_ok_grid(src, *, monotonic: bool = True) -> np.ndarray:
    """Fill sparse recipe OK bins before projecting them to a target wafer axis."""
    grid = np.asarray(src, dtype=np.float64).copy()
    if grid.ndim != 2 or not np.isfinite(grid).any():
        return grid
    flat = grid.reshape(-1)
    if monotonic:
        flat = _nondecreasing_finite_curve(flat)
    else:
        finite_idx = np.flatnonzero(np.isfinite(flat))
        if finite_idx.size:
            x = np.arange(flat.size, dtype=np.float64)
            flat = np.interp(x, finite_idx.astype(np.float64), flat[finite_idx])
    return flat.reshape(grid.shape)


def _project_ok_grid_to_wafer_axis(src, step_plot, spans,
                                   *, monotonic: bool = True) -> np.ndarray | None:
    """Stretch/shrink one canonical recipe OK grid onto this wafer's time axis."""
    if step_plot is None or not len(step_plot):
        return None
    src = _canonical_ok_grid(src, monotonic=monotonic)
    if src.ndim != 2 or src.shape[1] == 0 or not np.isfinite(src).any():
        return None
    grid = int(src.shape[1])
    step_arr = np.asarray(step_plot, dtype=int)
    pos_arr = _step_fraction_positions(step_arr, spans, grid)
    out = np.full(len(step_arr), np.nan, dtype=np.float64)
    valid = (step_arr >= 0) & (step_arr < src.shape[0])
    if not valid.any():
        return None
    grid_x = np.arange(grid, dtype=np.float64)
    for st in np.unique(step_arr[valid]):
        idx = np.flatnonzero(valid & (step_arr == st))
        row = src[int(st)]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        if finite.sum() == 1:
            out[idx] = float(row[finite][0])
        else:
            out[idx] = np.interp(pos_arr[idx], grid_x[finite], row[finite])
    return out if np.isfinite(out).any() else None


def _curve_to_step_grid(curve, spans, n_steps: int, grid: int,
                        *, monotonic_within_step: bool = True) -> np.ndarray:
    """Interpolate one wafer's curve to a dense ProcessStep x within-step grid."""
    curve = np.asarray(curve, dtype=np.float64)
    n_steps = max(1, int(n_steps))
    grid = max(2, int(grid))
    out = np.full((n_steps, grid), np.nan, dtype=np.float64)
    target_x = np.arange(grid, dtype=np.float64)
    for step, start, end in spans or []:
        st = int(step)
        if st <= 0 or st >= n_steps:
            continue
        start = max(0, int(start))
        end = min(int(end), len(curve) - 1)
        if end < start:
            continue
        vals = np.asarray(curve[start:end + 1], dtype=np.float64)
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        if finite.sum() == 1:
            row = np.full(grid, float(vals[finite][0]), dtype=np.float64)
        else:
            src_x = np.linspace(0.0, float(grid - 1), len(vals), dtype=np.float64)
            row = np.interp(target_x, src_x[finite], vals[finite])
        if monotonic_within_step:
            row = np.maximum.accumulate(row)
        out[st] = row
    return out


def _nondecreasing_grid_curves(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).copy()
    if arr.ndim < 2 or not np.isfinite(arr).any():
        return arr
    flat_len = int(arr.shape[-2] * arr.shape[-1])
    rows = arr.reshape(-1, flat_len)
    for i in range(rows.shape[0]):
        rows[i] = _nondecreasing_finite_curve(rows[i])
    return rows.reshape(arr.shape)


def _ensure_band_contains_mean(lo, mean, hi) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, dtype=np.float64).copy()
    mean = np.asarray(mean, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64).copy()
    valid_mean = np.isfinite(mean)
    valid_lo = valid_mean & np.isfinite(lo)
    valid_hi = valid_mean & np.isfinite(hi)
    lo[valid_lo] = np.minimum(lo[valid_lo], mean[valid_lo])
    hi[valid_hi] = np.maximum(hi[valid_hi], mean[valid_hi])
    return lo, hi


def _curve_to_view_full(rec, raw_curve) -> np.ndarray:
    curve = np.asarray(raw_curve, dtype=np.float64)
    factor = max(1, int(rec.get("_interp_factor", 1) or 1)) if isinstance(rec, dict) else 1
    if factor <= 1 or curve.size <= 1:
        return curve.copy()
    return _interp_continuous(curve[:, None], factor)[:, 0]


def _compute_ok_accum_bands(A, ok_records, grid: int = 100,
                            threshold=None, score_kind=None) -> dict:
    """Build recipe-wise OK mean curves for sensor and wafer accumulated scores.

    The alignment unit is ProcessStep plus within-step relative position. This
    avoids comparing raw time indices when wafers have different step lengths.
    """
    grid = max(10, int(grid))
    n_sensor = len(A.sensor_cols)
    n_steps = max(int(getattr(A, "n_steps_global", 0) or 0), 1)
    cfg = getattr(A, "cfg", None)
    score_topk = _score_topk_from_threshold(threshold, cfg)
    score_resid_agg = _score_resid_agg_from_threshold(threshold, cfg)
    score_wafer_agg = _score_wafer_agg_from_threshold(threshold, cfg)
    score_time_avg = _score_time_avg_from_threshold(threshold, cfg)
    score_final_avg = _score_final_avg_from_threshold(threshold, cfg)
    ok_final_scores = None
    if threshold is None or _uses_sum_reconstruction_threshold(threshold, score_kind):
        try:
            ok_final_scores = np.asarray(
                _prediction_scores(A, ok_records, threshold=threshold,
                                   score_kind=score_kind),
                dtype=np.float64)
        except Exception as e:
            ok_final_scores = None
            print(f"  ⚠ OK 누적 band 최종 score 계산 실패: {e}")
    recipes = sorted({_recipe_key(r) for r in ok_records})
    recipes = ["__global__"] + [r for r in recipes if r != "__global__"]
    ridx = {r: i for i, r in enumerate(recipes)}
    shape_s = (len(recipes), n_sensor, n_steps, grid)
    shape_t = (len(recipes), n_steps, grid)
    s_min = np.full(shape_s, np.inf, dtype=np.float64)
    s_max = np.full(shape_s, -np.inf, dtype=np.float64)
    s_step_min = np.full(shape_s, np.inf, dtype=np.float64)
    s_step_max = np.full(shape_s, -np.inf, dtype=np.float64)
    t_min = np.full(shape_t, np.inf, dtype=np.float64)
    t_max = np.full(shape_t, -np.inf, dtype=np.float64)
    s_sum = np.zeros(shape_s, dtype=np.float64)
    s_count = np.zeros(shape_s, dtype=np.int32)
    s_step_sum = np.zeros(shape_s, dtype=np.float64)
    s_step_count = np.zeros(shape_s, dtype=np.int32)
    t_sum = np.zeros(shape_t, dtype=np.float64)
    t_sumsq = np.zeros(shape_t, dtype=np.float64)
    t_count = np.zeros(shape_t, dtype=np.int32)
    total_final_min = np.full(len(recipes), np.inf, dtype=np.float64)
    total_final_max = np.full(len(recipes), -np.inf, dtype=np.float64)
    total_final_sum = np.zeros(len(recipes), dtype=np.float64)
    total_final_count = np.zeros(len(recipes), dtype=np.int32)

    for idx, rec in enumerate(ok_records, 1):
        try:
            level = A.error_level(rec) if hasattr(A, "error_level") else np.abs(A.z_residual(rec))
            level = np.asarray(level, dtype=np.float64)
            keep, _orig_idx, _t, step_plot, _bounds, spans = _plot_time_filter(A, rec, level.shape[0])
            if step_plot is None or not len(step_plot) or not keep.any():
                continue
            z = level[keep]
            if z.ndim != 2 or z.shape[1] != n_sensor:
                continue
            step_arr = np.asarray(step_plot, dtype=int)
            if not np.any((step_arr > 0) & (step_arr < n_steps)):
                continue
            full_step = rec.get("step") if isinstance(rec, dict) else None
            if full_step is not None and len(full_step) != level.shape[0]:
                full_step = None
            sensor_progress_full, _per_sensor_final = _sensor_score_progress_curves_from_level(
                level, full_step, resid_agg=score_resid_agg,
                score_time_avg=score_time_avg,
                score_final_avg=score_final_avg,
                cfg=cfg)
            sensor_accum = sensor_progress_full[:, keep]
            sensor_step_accum = np.vstack([
                _step_accum_series_from_level(z[:, j], step_arr, threshold_base=1.0)
                for j in range(n_sensor)
            ])
            score_sensor_indices = _topk_sensor_indices_from_level(
                level, score_topk, resid_agg=score_resid_agg)
            final_score = (
                float(ok_final_scores[idx - 1])
                if ok_final_scores is not None
                and idx - 1 < len(ok_final_scores)
                and np.isfinite(ok_final_scores[idx - 1])
                else None
            )
            total_curve_full = _wafer_decision_accum_series(
                level, full_step,
                sensor_indices=score_sensor_indices,
                resid_agg=score_resid_agg,
                wafer_agg=score_wafer_agg,
                score_time_avg=score_time_avg,
                score_final_avg=score_final_avg,
                cfg=cfg,
                final_score=final_score)
            total_curve = total_curve_full[keep]
            total_grid = _curve_to_step_grid(
                total_curve, spans, n_steps, grid, monotonic_within_step=True)
            sensor_grids = [
                _curve_to_step_grid(sensor_accum[j], spans, n_steps, grid,
                                    monotonic_within_step=True)
                for j in range(n_sensor)
            ]
            sensor_step_grids = [
                _curve_to_step_grid(sensor_step_accum[j], spans, n_steps, grid,
                                    monotonic_within_step=True)
                for j in range(n_sensor)
            ]
            recipe_ids = [0]
            recipe = _recipe_key(rec)
            if recipe in ridx and ridx[recipe] != 0:
                recipe_ids.append(ridx[recipe])
            for ri in recipe_ids:
                if final_score is not None and np.isfinite(final_score):
                    total_final_min[ri] = min(total_final_min[ri], float(final_score))
                    total_final_max[ri] = max(total_final_max[ri], float(final_score))
                    total_final_sum[ri] += float(final_score)
                    total_final_count[ri] += 1
                _add_grid_stats(t_sum[ri], t_count[ri], t_min[ri], t_max[ri],
                                total_grid, sumsq_arr=t_sumsq[ri])
                for j in range(n_sensor):
                    _add_grid_stats(s_sum[ri, j], s_count[ri, j],
                                    s_min[ri, j], s_max[ri, j],
                                    sensor_grids[j])
                    _add_grid_stats(s_step_sum[ri, j], s_step_count[ri, j],
                                    s_step_min[ri, j], s_step_max[ri, j],
                                    sensor_step_grids[j])
        except Exception as e:
            print(f"  ⚠ OK 누적 band 계산 skip: {rec.get('wafer_id', idx)}: {e}")

    for arr, fill in (
            (s_min, np.nan), (s_max, np.nan),
            (s_step_min, np.nan), (s_step_max, np.nan),
            (t_min, np.nan), (t_max, np.nan)):
        arr[~np.isfinite(arr)] = fill
    sensor_mean = _mean_from_sum_count(s_sum, s_count)
    sensor_step_mean = _mean_from_sum_count(s_step_sum, s_step_count)
    total_mean = _mean_from_sum_count(t_sum, t_count)
    total_std = _std_from_sum_sumsq_count(t_sum, t_sumsq, t_count)
    total_final_mean = np.full(len(recipes), np.nan, dtype=np.float64)
    np.divide(total_final_sum, total_final_count, out=total_final_mean,
              where=total_final_count > 0)
    total_final_min[~np.isfinite(total_final_min)] = np.nan
    total_final_max[~np.isfinite(total_final_max)] = np.nan
    sensor_mean = _nondecreasing_grid_curves(sensor_mean)
    sensor_min = _nondecreasing_grid_curves(s_min)
    sensor_max = _nondecreasing_grid_curves(s_max)
    total_mean = _nondecreasing_grid_curves(total_mean)
    total_min = _nondecreasing_grid_curves(t_min)
    total_max = _nondecreasing_grid_curves(t_max)
    sensor_min, sensor_max = _ensure_band_contains_mean(
        sensor_min, sensor_mean, sensor_max)
    total_min, total_max = _ensure_band_contains_mean(
        total_min, total_mean, total_max)
    return {
        "recipes": recipes,
        "sensor_cols": [str(x) for x in A.sensor_cols],
        "grid": grid,
        "sensor_min": sensor_min,
        "sensor_max": sensor_max,
        "sensor_step_min": s_step_min,
        "sensor_step_max": s_step_max,
        "total_min": total_min,
        "total_max": total_max,
        "sensor_mean": sensor_mean,
        "sensor_step_mean": sensor_step_mean,
        "total_mean": total_mean,
        "total_std": total_std,
        "total_final_min": total_final_min,
        "total_final_mean": total_final_mean,
        "total_final_max": total_final_max,
    }


def load_or_build_ok_accum_bands(A, ok_records, cache_path: Path, grid: int = 100,
                                 threshold=None, score_kind=None,
                                 exclude_ok_over_threshold: bool = False) -> dict | None:
    cache_path = Path(cache_path)
    ok_records = list(ok_records or [])
    had_ok_records = bool(ok_records)
    cfg = getattr(A, "cfg", None)
    print(f"  · train OK 평균 필터: "
          f"{'ON' if exclude_ok_over_threshold else 'OFF'} "
          f"(score topk={_score_topk_from_threshold(threshold, cfg) or 'all'}, "
          f"resid_agg={_score_resid_agg_from_threshold(threshold, cfg)}, "
          f"wafer_agg={_score_wafer_agg_from_threshold(threshold, cfg)})")
    if exclude_ok_over_threshold and ok_records:
        ok_records, dropped = _filter_ok_records_below_threshold(
            A, ok_records, threshold=threshold, score_kind=score_kind)
        if dropped:
            sample = ", ".join(
                f"{d['wafer_id']}({d['score']:.4g}>={d['threshold']:.4g})"
                for d in dropped[:5])
            more = "" if len(dropped) <= 5 else f" 외 {len(dropped) - 5}장"
            print(f"  · train OK 평균 필터: threshold 이상 {len(dropped)}장 제외 "
                  f"({sample}{more})")
        else:
            print("  · train OK 평균 필터: threshold 이상 웨이퍼 없음")
    signature = _ok_band_signature(
        A, ok_records, grid, threshold=threshold, score_kind=score_kind,
        exclude_ok_over_threshold=exclude_ok_over_threshold) if ok_records else None
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            if (signature is not None and str(data["signature"].item()) == signature):
                print(f"  · OK 누적 점수 band 캐시 재사용: {cache_path}")
                return _cache_to_ok_bands(data)
            if not had_ok_records and not ok_records and _ok_cache_compatible_without_records(
                    A, data, grid, threshold=threshold, score_kind=score_kind,
                    exclude_ok_over_threshold=exclude_ok_over_threshold):
                print(f"  · OK 누적 점수 band 캐시 재사용(train OK 없음): {cache_path}")
                return _cache_to_ok_bands(data)
        except Exception as e:
            print(f"  ⚠ OK 누적 점수 band 캐시 로드 실패, 재계산: {e}")
    if not ok_records:
        print(f"  ⚠ train OK 없음 또는 필터 후 0장: OK 평균선 생략"
              f"{' (호환 캐시 없음)' if cache_path.exists() else ''}")
        return None
    print(f"  · OK 누적 점수 band 계산 중: train OK {len(ok_records)}장")
    bands = _compute_ok_accum_bands(
        A, ok_records, grid=grid, threshold=threshold, score_kind=score_kind)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        signature=np.asarray(signature),
        recipes=np.asarray(bands["recipes"]),
        sensor_cols=np.asarray(bands["sensor_cols"]),
        grid=np.asarray(int(bands["grid"])),
        sensor_min=bands["sensor_min"],
        sensor_max=bands["sensor_max"],
        sensor_step_min=bands["sensor_step_min"],
        sensor_step_max=bands["sensor_step_max"],
        total_min=bands["total_min"],
        total_max=bands["total_max"],
        sensor_mean=bands["sensor_mean"],
        sensor_step_mean=bands["sensor_step_mean"],
        total_mean=bands["total_mean"],
        total_std=bands["total_std"],
        total_final_min=bands["total_final_min"],
        total_final_mean=bands["total_final_mean"],
        total_final_max=bands["total_final_max"],
    )
    print(f"  · OK 누적 점수 band 캐시 저장: {cache_path}")
    return bands


def _ok_band_recipe_index(ok_bands: dict | None, recipe: str) -> int | None:
    if not ok_bands:
        return None
    recipes = ok_bands.get("recipes") or []
    recipe = str(recipe or "unknown")
    if recipe in recipes:
        return recipes.index(recipe)
    return recipes.index("__global__") if "__global__" in recipes else None


def _lookup_ok_band(ok_bands: dict | None, rec, step_plot, spans,
                    sensor_idx: int | None = None,
                    step_reset: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not ok_bands or step_plot is None or not len(step_plot):
        return None, None
    ri = _ok_band_exact_recipe_index(ok_bands, _recipe_key(rec))
    if ri is None:
        return None, None
    if sensor_idx is None:
        lo_src = ok_bands.get("total_min")
        hi_src = ok_bands.get("total_max")
        if lo_src is None or hi_src is None:
            return None, None
        lo_src = lo_src[ri]
        hi_src = hi_src[ri]
        monotonic = True
    else:
        lo_key = "sensor_step_min" if step_reset else "sensor_min"
        hi_key = "sensor_step_max" if step_reset else "sensor_max"
        lo_src = ok_bands.get(lo_key)
        hi_src = ok_bands.get(hi_key)
        if lo_src is None or hi_src is None:
            return None, None
        lo_src = lo_src[ri, int(sensor_idx)]
        hi_src = hi_src[ri, int(sensor_idx)]
        monotonic = not step_reset
    lo = _project_ok_grid_to_wafer_axis(lo_src, step_plot, spans, monotonic=monotonic)
    hi = _project_ok_grid_to_wafer_axis(hi_src, step_plot, spans, monotonic=monotonic)
    return lo, hi


def _ok_band_exact_recipe_index(ok_bands: dict | None, recipe: str) -> int | None:
    if not ok_bands:
        return None
    recipes = ok_bands.get("recipes") or []
    recipe = str(recipe or "unknown")
    return recipes.index(recipe) if recipe in recipes else None


def _lookup_ok_mean(ok_bands: dict | None, rec, step_plot, spans,
                    sensor_idx: int | None = None,
                    step_reset: bool = False) -> np.ndarray | None:
    """Return same-recipe train OK mean curve aligned to this wafer's plot axis."""
    if not ok_bands or step_plot is None or not len(step_plot):
        return None
    ri = _ok_band_exact_recipe_index(ok_bands, _recipe_key(rec))
    if ri is None:
        return None
    if sensor_idx is None:
        src = ok_bands.get("total_mean")
        if src is None:
            return None
        src = src[ri]
    else:
        key = "sensor_step_mean" if step_reset else "sensor_mean"
        src = ok_bands.get(key)
        if src is None:
            return None
        src = src[ri, int(sensor_idx)]
    return _project_ok_grid_to_wafer_axis(src, step_plot, spans,
                                          monotonic=not step_reset)


def _lookup_ok_total_final_stats(ok_bands: dict | None, rec) -> tuple[float | None, float | None, float | None]:
    if not ok_bands:
        return None, None, None
    ri = _ok_band_exact_recipe_index(ok_bands, _recipe_key(rec))
    if ri is None:
        return None, None, None
    out = []
    for key in ("total_final_min", "total_final_mean", "total_final_max"):
        arr = ok_bands.get(key)
        if arr is None:
            out.append(None)
            continue
        try:
            value = float(np.asarray(arr, dtype=np.float64)[ri])
        except Exception:
            value = float("nan")
        out.append(value if np.isfinite(value) else None)
    return tuple(out)


def _lookup_ok_total_std(ok_bands: dict | None, rec, step_plot, spans) -> np.ndarray | None:
    if not ok_bands or step_plot is None or not len(step_plot):
        return None
    ri = _ok_band_exact_recipe_index(ok_bands, _recipe_key(rec))
    if ri is None:
        return None
    src = ok_bands.get("total_std")
    if src is None:
        return None
    src = src[ri]
    return _project_ok_grid_to_wafer_axis(src, step_plot, spans, monotonic=True)


def _ratio_to_ok_max(values, ok_min, ok_max) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Convert a cumulative series to % where recipe OK max is 100."""
    values = np.asarray(values, dtype=np.float64)
    if ok_max is None:
        return values, None, None
    hi = np.asarray(ok_max, dtype=np.float64)
    lo = np.asarray(ok_min, dtype=np.float64) if ok_min is not None else np.zeros_like(hi)
    den = hi.copy()
    valid = np.isfinite(den) & (den > 1e-12)
    fallback_vals = den[valid]
    fallback = float(np.nanmedian(fallback_vals)) if fallback_vals.size else 1.0
    if not np.isfinite(fallback) or fallback <= 1e-12:
        fallback = 1.0
    den[~valid] = fallback
    pct = values / np.maximum(den, 1e-12) * 100.0
    lo_pct = lo / np.maximum(den, 1e-12) * 100.0
    hi_pct = hi / np.maximum(den, 1e-12) * 100.0
    lo_pct[~np.isfinite(lo_pct)] = np.nan
    hi_pct[~np.isfinite(hi_pct)] = np.nan
    return pct, lo_pct, hi_pct


def _progress_ratio_to_threshold(accum, final_score, threshold_value) -> np.ndarray:
    """Map cumulative anomaly progression to the final wafer score/threshold %.

    The cumulative sum controls where the score grows; the final value matches
    the actual wafer-level decision score ratio.
    """
    accum = np.asarray(accum, dtype=np.float64)
    try:
        th = float(threshold_value)
        score = float(final_score)
    except Exception:
        return accum
    if not np.isfinite(th) or th <= 1e-12:
        return accum
    final_pct = score / th * 100.0
    last = float(accum[-1]) if accum.size else 0.0
    if not np.isfinite(last) or last <= 1e-12:
        return np.zeros_like(accum)
    return accum / last * final_pct


def _copy_optional_array(value):
    if value is None:
        return None
    try:
        return np.asarray(value).copy()
    except Exception:
        return value


def _fit_missing_sigma_preserve_score_limits(A, ok_records) -> bool:
    """Fill legacy models' sigma bands without changing anomaly score limits.

    Older checkpoints may not contain resid_sigma. Reusing _fit_error_limits()
    directly is unsafe here because it also overwrites resid_abs_limit, which is
    the denominator used for anomaly scoring and threshold comparison.
    """
    if getattr(A, "resid_sigma", None) is not None:
        return False
    if not ok_records or not hasattr(A, "_fit_error_limits"):
        return False
    old_limit = _copy_optional_array(getattr(A, "resid_abs_limit", None))
    old_std = _copy_optional_array(getattr(A, "resid_std", None))
    old_per_step_limit = bool(getattr(A, "_per_step_limit", False))
    try:
        A._fit_error_limits(ok_records)
        new_sigma = _copy_optional_array(getattr(A, "resid_sigma", None))
        new_per_step_sigma = bool(getattr(A, "_per_step_sigma", False))
    finally:
        A.resid_abs_limit = old_limit
        A.resid_std = old_std
        A._per_step_limit = old_per_step_limit
    if new_sigma is not None:
        A.resid_sigma = new_sigma
        A._per_step_sigma = new_per_step_sigma
        return True
    return False


def _csv_float(value) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    return f"{value:.10g}" if np.isfinite(value) else ""


def _display_score_threshold(score, threshold, rec) -> tuple[float, float | None]:
    """Scale a score by the unmultiplied base threshold for display only."""
    try:
        score = float(score)
    except Exception:
        score = float("nan")
    scale = _display_scale_threshold_for_record(threshold, rec)
    display_threshold = _display_threshold_for_record(threshold, rec)
    try:
        scale = None if scale is None else float(scale)
        display_threshold = None if display_threshold is None else float(display_threshold)
    except Exception:
        scale, display_threshold = None, None
    if scale is None or not np.isfinite(scale) or scale <= 1e-12:
        return score, None
    return score / scale, display_threshold


_DISPLAY_SCORE_CAP = 3.0
_PREDICTION_PLOT_COMPRESS_TRIGGER = 2.0
_PREDICTION_PLOT_COMPRESS_BASE = 1.0
_PREDICTION_PLOT_COMPRESS_TOP = 2.0


def _cap_display_values(values, cap: float = _DISPLAY_SCORE_CAP):
    arr = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    arr[finite] = np.minimum(arr[finite], float(cap))
    return arr


def _cap_display_value(value, cap: float = _DISPLAY_SCORE_CAP) -> float:
    try:
        value = float(value)
    except Exception:
        return value
    if not np.isfinite(value):
        return value
    return min(value, float(cap))


def _has_display_cap_excess(*values, cap: float = _DISPLAY_SCORE_CAP) -> bool:
    for value in values:
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.nanmax(finite)) > float(cap):
            return True
    return False


def _prediction_plot_transform_values(values, threshold_values=None):
    """Compress only prediction PNG display values above threshold when needed."""
    converted = []
    for value in values:
        try:
            v = float(value)
        except Exception:
            v = float("nan")
        converted.append(v if np.isfinite(v) else float("nan"))
    values = converted
    finite = [v for v in values if np.isfinite(v)]
    max_score = max(finite) if finite else 0.0
    compress = max_score > _PREDICTION_PLOT_COMPRESS_TRIGGER
    denom = max(max_score - _PREDICTION_PLOT_COMPRESS_BASE, 1e-12)

    def _transform(value):
        if value is None:
            return None
        try:
            v = float(value)
        except Exception:
            return value
        if not np.isfinite(v):
            return v
        if not compress or v <= _PREDICTION_PLOT_COMPRESS_BASE:
            return v
        scaled = _PREDICTION_PLOT_COMPRESS_BASE + (
            (v - _PREDICTION_PLOT_COMPRESS_BASE) / denom
        ) * (_PREDICTION_PLOT_COMPRESS_TOP - _PREDICTION_PLOT_COMPRESS_BASE)
        return min(_PREDICTION_PLOT_COMPRESS_TOP, scaled)

    transformed_values = [_transform(v) for v in values]
    transformed_thresholds = None
    if threshold_values is not None:
        transformed_thresholds = [_transform(v) for v in threshold_values]
    return transformed_values, transformed_thresholds, compress


def _ratio_percent(numerator, denominator) -> float | str:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except Exception:
        return ""
    if (not np.isfinite(numerator)) or (not np.isfinite(denominator)) \
            or abs(denominator) <= 1e-12:
        return ""
    return round(float(numerator) / float(denominator) * 100.0, 3)


def _mean_display_score(rows, score_index: int) -> float | None:
    vals = []
    for row in rows:
        try:
            value = float(row[score_index])
        except Exception:
            continue
        if np.isfinite(value):
            vals.append(value)
    if not vals:
        return None
    mean_value = float(np.mean(vals))
    return mean_value if np.isfinite(mean_value) and abs(mean_value) > 1e-12 else None


def _score_stats(values) -> dict | None:
    vals = []
    for value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value):
            vals.append(value)
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(mean) or not np.isfinite(std):
        return None
    return {"mean": mean, "std": std, "n": int(arr.size)}


def _train_ok_score_stats_from_scores(records, scores, threshold) -> dict | None:
    records = list(records or [])
    if not records or not scores:
        return None
    global_vals = []
    by_recipe_vals = {}
    for rec in records:
        wid = str(rec.get("wafer_id", ""))
        if wid not in scores:
            continue
        display_score, _display_threshold = _display_score_threshold(
            float(scores[wid]), threshold, rec)
        if not np.isfinite(float(display_score)):
            continue
        global_vals.append(float(display_score))
        by_recipe_vals.setdefault(_recipe_key(rec), []).append(float(display_score))
    global_stats = _score_stats(global_vals)
    by_recipe = {
        recipe: stats
        for recipe, vals in by_recipe_vals.items()
        for stats in [_score_stats(vals)]
        if stats is not None
    }
    if global_stats is None and not by_recipe:
        return None
    return {"global": global_stats, "by_recipe": by_recipe}


def _build_train_ok_score_reference(A, records, threshold, score_kind) -> tuple[dict, dict | None]:
    records = list(records or [])
    if not records:
        return {}, None
    try:
        values = _prediction_scores(A, records, threshold, score_kind)
        scores = {r["wafer_id"]: float(v) for r, v in zip(records, values)}
        return scores, _train_ok_score_stats_from_scores(records, scores, threshold)
    except Exception as e:
        print(f"  ⚠ train OK sigma 기준 계산 실패: {e}")
        return {}, None


def _sigma_level(display_score, rec, train_score_stats) -> float | str:
    if not isinstance(train_score_stats, dict):
        return ""
    recipe_stats = (train_score_stats.get("by_recipe") or {}).get(_recipe_key(rec))
    global_stats = train_score_stats.get("global")
    stats = recipe_stats if isinstance(recipe_stats, dict) else None
    if not stats or float(stats.get("std", 0.0) or 0.0) <= 1e-12:
        stats = global_stats if isinstance(global_stats, dict) else None
    if not stats:
        return ""
    try:
        score = float(display_score)
        mean = float(stats.get("mean"))
        std = float(stats.get("std"))
    except Exception:
        return ""
    if (not np.isfinite(score)) or (not np.isfinite(mean)) \
            or (not np.isfinite(std)) or std <= 1e-12:
        return ""
    return round((score - mean) / std, 3)


def _display_threshold_json_value(threshold):
    if threshold is None:
        return None
    if not isinstance(threshold, dict):
        return 1.0
    mode = _threshold_mode(threshold)
    if mode not in {"all", "recipe"}:
        return threshold
    out = dict(threshold)
    recipes = set()
    for key in ("threshold_by_recipe", "threshold_raw", "threshold",
                "base_threshold_by_recipe", "threshold_recipe_multipliers"):
        table = threshold.get(key)
        if isinstance(table, dict):
            recipes.update(str(k) for k in table)
    bare_recipe_map = mode == "recipe" and not recipes
    if bare_recipe_map:
        recipes.update(
            str(k) for k, v in threshold.items()
            if not _is_threshold_metadata_key(k)
            and _float_threshold_value(v) is not None
        )
    display_map = {}
    for recipe in sorted(recipes):
        rec = {"recipe": recipe}
        display_th = _display_threshold_for_record(threshold, rec)
        if display_th is not None:
            display_map[recipe] = float(display_th)
    if mode == "all":
        out["threshold"] = 1.0
        out["threshold_raw"] = 1.0
        out["global_threshold"] = 1.0
        out["global_threshold_raw"] = 1.0
        if display_map:
            out["threshold_by_recipe"] = display_map
        out["base_threshold_by_recipe"] = {
            recipe: 1.0 for recipe in display_map
        }
    else:
        if bare_recipe_map:
            out = {}
        out["threshold"] = {k: round(float(v), 4) for k, v in display_map.items()}
        out["threshold_raw"] = display_map
        out["threshold_by_recipe"] = display_map
        out["base_threshold_by_recipe"] = {
            recipe: 1.0 for recipe in display_map
        }
        out["global_threshold"] = 1.0
        out["global_threshold_raw"] = 1.0
    return out


def _display_summary_json(summary: dict, best_params):
    doc = {k: v for k, v in summary.items() if k not in {"params", "raw_scores"}}
    if doc.get("threshold") is not None:
        doc["threshold"] = _display_threshold_json_value(doc.get("threshold"))
    vc = dict(doc.get("vc") or {})
    if vc.get("threshold") is not None:
        vc["threshold"] = _display_threshold_json_value(vc.get("threshold"))
    if vc.get("threshold_raw") is not None:
        vc["threshold_raw"] = _display_threshold_json_value(vc.get("threshold_raw"))
    doc["vc"] = vc
    doc["params"] = best_params
    return doc


def _write_wafer_accum_score_csv(csv_path, rec, orig_idx, step_plot,
                                 current, ok_mean, ok_min, ok_max,
                                 threshold_value, scale_value=None) -> None:
    import csv as _csv
    csv_path = Path(csv_path)
    fields = [
        "plot_x",
        "source_time",
        "process_step",
        "current_wafer_anomaly_score",
        "current_wafer_threshold_ratio",
        "ok_mean_anomaly_score",
        "ok_mean_threshold_ratio",
        "ok_min_anomaly_score",
        "ok_min_threshold_ratio",
        "ok_max_anomaly_score",
        "ok_max_threshold_ratio",
        "threshold",
        "threshold_ratio",
    ]
    current = np.asarray(current, dtype=np.float64)
    ok_mean = None if ok_mean is None else np.asarray(ok_mean, dtype=np.float64)
    ok_min = None if ok_min is None else np.asarray(ok_min, dtype=np.float64)
    ok_max = None if ok_max is None else np.asarray(ok_max, dtype=np.float64)
    threshold = np.nan if threshold_value is None else float(threshold_value)
    scale = np.nan if scale_value is None else float(scale_value)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = threshold if np.isfinite(threshold) and threshold > 1e-12 else np.nan
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = _csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(current)):
            mean = np.nan if ok_mean is None or i >= len(ok_mean) else float(ok_mean[i])
            lo = np.nan if ok_min is None or i >= len(ok_min) else float(ok_min[i])
            hi = np.nan if ok_max is None or i >= len(ok_max) else float(ok_max[i])
            cur_out = current[i] / scale if np.isfinite(scale) else current[i]
            mean_out = mean / scale if np.isfinite(mean) and np.isfinite(scale) else mean
            lo_out = max(0.0, lo) / scale if np.isfinite(lo) and np.isfinite(scale) else lo
            hi_out = hi / scale if np.isfinite(hi) and np.isfinite(scale) else hi
            th_out = threshold / scale if np.isfinite(threshold) and np.isfinite(scale) else threshold
            src_idx = int(orig_idx[i]) if i < len(orig_idx) else i
            st = int(step_plot[i]) if step_plot is not None and i < len(step_plot) else ""
            writer.writerow({
                "plot_x": i,
                "source_time": _csv_float(_source_time_for_rec(rec, src_idx)),
                "process_step": st,
                "current_wafer_anomaly_score": _csv_float(cur_out),
                "current_wafer_threshold_ratio": _csv_float(cur_out),
                "ok_mean_anomaly_score": _csv_float(mean_out),
                "ok_mean_threshold_ratio": _csv_float(mean_out),
                "ok_min_anomaly_score": _csv_float(max(0.0, lo_out) if np.isfinite(lo_out) else np.nan),
                "ok_min_threshold_ratio": _csv_float(
                    max(0.0, lo_out) if np.isfinite(lo_out) else np.nan),
                "ok_max_anomaly_score": _csv_float(hi_out),
                "ok_max_threshold_ratio": _csv_float(hi_out),
                "threshold": _csv_float(th_out),
                "threshold_ratio": _csv_float(th_out),
            })


def plot_recon_sensor(A, rec, j, actual, expected, z, out_path,
                      threshold=None, ok_bands=None,
                      raw_rec=None, raw_z=None,
                      include_accum: bool = True,
                      sensor_contribution_scale: float = 1.0) -> None:
    """센서 1개 고해상도 2단 그림: (위) 실제 vs 기대, (아래) 누적 이상치 점수."""
    import matplotlib
    matplotlib.use("Agg")
    _setup_font()
    import matplotlib.pyplot as plt
    t_full = actual.shape[0]
    plot_keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter(A, rec, t_full)
    if orig_idx.size == 0:
        return
    cur, exp, zc = actual[plot_keep, j], expected[plot_keep, j], z[plot_keep, j]
    T = len(cur)
    band_sigma = float(getattr(A.cfg, "explain_sigma", 4.0))
    large_sigma = float(getattr(A.cfg, "large_residual_sigma", band_sigma))
    min_highlight_width = int(getattr(A.cfg, "highlight_min_width", 10))

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(20, 7.5),
                                   gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
    sigma_raw = _sigma_series(A, rec, j, t_full)[plot_keep]
    ok_limit = _limit_series(A, rec, j, t_full)[plot_keep]
    resid_sigma, large_mask = _large_residual_mask(cur, exp, sigma_raw, large_sigma)
    _, large_events = _expanded_highlight_mask(large_mask, resid_sigma, min_highlight_width)
    large_events = _filter_isolated_step_transition_events(
        large_events, large_mask, bounds, A.cfg)
    exp_display = _smooth_step_boundary_prediction_for_display(exp, bounds, A.cfg)
    band_lo = exp_display - ok_limit
    band_hi = exp_display + ok_limit
    ylo = float(min(cur.min(), exp_display.min(), band_lo.min()))
    yhi = float(max(cur.max(), exp_display.max(), band_hi.max()))
    pad = (yhi - ylo) * 0.12 + 1e-6
    axT.fill_between(t, band_lo, band_hi, color="#87CEEB", alpha=0.24,
                     zorder=0, label="OK 기준 정상 범위")
    _draw_sigma_event_bands(axT, large_events, large_sigma, color="#d62728")
    _draw_step_guides(axT, bounds, spans, label=True)
    axT.plot(t, exp_display, color="#1f77b4", lw=1.6, ls="--", label="기대", zorder=4)
    axT.plot(t, cur, color="#d62728", lw=1.6, label="실제", zorder=5)
    axT.set_ylim(ylo - pad, yhi + pad)
    axT.set_title(f"{rec['wafer_id']} · {A.sensor_cols[j]} "
                  f"(ProcessStep 0 제외, interpolation x{max(1, int(rec.get('_interp_factor', 1) or 1))})",
                  fontsize=13)
    axT.legend(fontsize=9, loc="upper left")
    axT.tick_params(labelsize=9)
    # (아래) 정상 범위 초과분 누적. 같은 recipe train OK 평균과 비교한다.
    threshold_mode = _threshold_mode(threshold) if threshold is not None else ""
    use_step_threshold = threshold_mode in {"sensor", "step"}
    cfg = getattr(A, "cfg", None)
    score_time_avg = (
        False if use_step_threshold else _score_time_avg_from_threshold(threshold, cfg)
    )
    score_final_avg = (
        False if use_step_threshold else _score_final_avg_from_threshold(threshold, cfg)
    )
    if raw_z is not None and raw_rec is not None:
        raw_step = raw_rec.get("step") if isinstance(raw_rec, dict) else None
        if use_step_threshold:
            raw_curve = _step_accum_series_from_level(
                np.asarray(raw_z, dtype=np.float64)[:, j], raw_step, threshold_base=1.0)
        else:
            raw_curves, _per_sensor_final = _sensor_score_progress_curves_from_level(
                np.asarray(raw_z, dtype=np.float64), raw_step,
                resid_agg=_score_resid_agg_from_cfg(cfg),
                score_time_avg=score_time_avg,
                score_final_avg=score_final_avg,
                cfg=cfg)
            raw_curve = raw_curves[int(j)] if int(j) < raw_curves.shape[0] else None
        curve_full = _curve_to_view_full(rec, raw_curve) if raw_curve is not None else None
        accum = (
            curve_full[plot_keep]
            if curve_full is not None and len(curve_full) == t_full
            else None
        )
    else:
        accum = None
    if accum is None:
        if use_step_threshold:
            accum = _step_accum_series_from_level(zc, step_plot, threshold_base=1.0)
        else:
            sensor_curves, _per_sensor_final = _sensor_score_progress_curves_from_level(
                z, rec.get("step") if isinstance(rec, dict) else step_plot,
                resid_agg=_score_resid_agg_from_cfg(cfg),
                score_time_avg=score_time_avg,
                score_final_avg=score_final_avg,
                cfg=cfg)
            accum = sensor_curves[int(j), plot_keep] if int(j) < sensor_curves.shape[0] else None
            if accum is None:
                accum = _sensor_reconstruction_accum_series_from_level(zc, step_plot)
    ok_mean = _lookup_ok_mean(
        ok_bands, rec, step_plot, spans, sensor_idx=j,
        step_reset=use_step_threshold)
    ok_min, ok_max = _lookup_ok_band(
        ok_bands, rec, step_plot, spans, sensor_idx=j,
        step_reset=use_step_threshold)
    try:
        contribution_scale = float(sensor_contribution_scale)
    except Exception:
        contribution_scale = 1.0
    if (not use_step_threshold and np.isfinite(contribution_scale)
            and contribution_scale > 0 and abs(contribution_scale - 1.0) > 1e-12):
        accum = np.asarray(accum, dtype=np.float64) * contribution_scale
        if ok_mean is not None:
            ok_mean = np.asarray(ok_mean, dtype=np.float64) * contribution_scale
        if ok_min is not None:
            ok_min = np.asarray(ok_min, dtype=np.float64) * contribution_scale
        if ok_max is not None:
            ok_max = np.asarray(ok_max, dtype=np.float64) * contribution_scale
    cap_bottom_display = False
    if threshold_mode == "all":
        scale = _display_scale_threshold_for_record(threshold, rec)
        try:
            scale = None if scale is None else float(scale)
        except Exception:
            scale = None
        if scale is not None and np.isfinite(scale) and scale > 1e-12:
            accum = np.asarray(accum, dtype=np.float64) / scale
            if ok_mean is not None:
                ok_mean = np.asarray(ok_mean, dtype=np.float64) / scale
            if ok_min is not None:
                ok_min = np.asarray(ok_min, dtype=np.float64) / scale
            if ok_max is not None:
                ok_max = np.asarray(ok_max, dtype=np.float64) / scale
            cap_bottom_display = _has_display_cap_excess(
                accum, ok_mean, ok_min, ok_max)
    accum_for_plot = _cap_display_values(accum) if cap_bottom_display else accum
    ok_mean_for_plot = (
        _cap_display_values(ok_mean) if cap_bottom_display and ok_mean is not None else ok_mean
    )
    ok_min_for_plot = (
        _cap_display_values(ok_min) if cap_bottom_display and ok_min is not None else ok_min
    )
    ok_max_for_plot = (
        _cap_display_values(ok_max) if cap_bottom_display and ok_max is not None else ok_max
    )
    if ok_min_for_plot is not None and ok_max_for_plot is not None:
        lo = np.maximum(0.0, np.asarray(ok_min_for_plot, dtype=np.float64))
        hi = np.asarray(ok_max_for_plot, dtype=np.float64)
        valid_band = np.isfinite(lo) & np.isfinite(hi)
        if valid_band.any():
            axB.fill_between(t, lo, hi, where=valid_band, color="#87CEEB",
                             alpha=0.24, label="같은 recipe OK min~max", zorder=1)
    if ok_mean_for_plot is not None:
        axB.plot(t, ok_mean_for_plot, color="#1f77b4", lw=1.25,
                 label="같은 recipe OK 평균", zorder=2)
    if include_accum:
        axB.plot(t, accum_for_plot, color="#333333", lw=1.25,
                 label="현재 wafer 누적 이상치 점수", zorder=3)
    th_series = None
    if use_step_threshold:
        th_series = _threshold_series_for_sensor(
            threshold, rec, str(A.sensor_cols[j]), step_plot, T)
        if th_series is not None and np.isfinite(th_series).any():
            if threshold_mode == "step":
                axB.step(t, th_series, where="post", color="#d62728", ls="--",
                         lw=1.1, label="threshold", zorder=4)
            else:
                finite_th = th_series[np.isfinite(th_series)]
                if finite_th.size:
                    axB.axhline(float(finite_th[0]), color="#d62728", ls="--",
                                lw=1.1, label="threshold", zorder=4)
    _draw_step_guides(axB, bounds, spans, label=False)
    axB.set_ylabel("누적 복원오차 점수", fontsize=10)
    axB.set_xlabel(_time_axis_label(rec), fontsize=10)
    vals_for_ylim = []
    if include_accum:
        vals_for_ylim.append(np.asarray(accum_for_plot)[np.isfinite(accum_for_plot)])
    if ok_mean_for_plot is not None:
        vals_for_ylim.append(ok_mean_for_plot[np.isfinite(ok_mean_for_plot)])
    if ok_max_for_plot is not None:
        ok_max_arr = np.asarray(ok_max_for_plot, dtype=np.float64)
        vals_for_ylim.append(ok_max_arr[np.isfinite(ok_max_arr)])
    if th_series is not None:
        vals_for_ylim.append(th_series[np.isfinite(th_series)])
    finite_vals = np.concatenate([v for v in vals_for_ylim if len(v)]) if vals_for_ylim else np.array([])
    ymax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    if cap_bottom_display:
        axB.set_ylim(0, _DISPLAY_SCORE_CAP)
    else:
        axB.set_ylim(0, ymax * 1.08 + 1e-9 if ymax > 0 else 1.0)
    axB.legend(fontsize=9, loc="upper left"); axB.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wafer_accum_score(A, rec, z, out_path, ok_bands=None,
                           threshold=None, final_score=None,
                           raw_rec=None, raw_z=None, sensor_indices=None,
                           csv_path=None) -> None:
    """Plot all-sensor accumulated anomaly score for one wafer."""
    import matplotlib
    matplotlib.use("Agg")
    _setup_font()
    import matplotlib.pyplot as plt
    z = np.asarray(z, dtype=np.float64)
    plot_keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter(A, rec, z.shape[0])
    if step_plot is None or not len(step_plot) or not plot_keep.any():
        return
    zc = z[plot_keep]
    if raw_z is not None and raw_rec is not None:
        raw_sensor_indices = sensor_indices
        cfg = getattr(A, "cfg", None)
        resid_agg = _score_resid_agg_from_cfg(cfg)
        wafer_agg = _score_wafer_agg_from_cfg(cfg)
        score_time_avg = _score_time_avg_from_threshold(threshold, cfg)
        score_final_avg = _score_final_avg_from_threshold(threshold, cfg)
        if raw_sensor_indices is None:
            raw_sensor_indices = _topk_sensor_indices_from_level(
                np.asarray(raw_z, dtype=np.float64),
                _score_topk_from_cfg(cfg),
                resid_agg=resid_agg)
        raw_curve = _wafer_decision_accum_series(
            np.asarray(raw_z, dtype=np.float64),
            raw_rec.get("step") if isinstance(raw_rec, dict) else None,
            sensor_indices=raw_sensor_indices,
            resid_agg=resid_agg,
            wafer_agg=wafer_agg,
            score_time_avg=score_time_avg,
            score_final_avg=score_final_avg,
            cfg=cfg,
            final_score=final_score)
        curve_full = _curve_to_view_full(rec, raw_curve)
        accum = curve_full[plot_keep] if len(curve_full) == z.shape[0] else None
    else:
        accum = None
    if accum is None:
        if sensor_indices is None:
            cfg = getattr(A, "cfg", None)
            sensor_indices = _topk_sensor_indices_from_level(
                zc, _score_topk_from_cfg(cfg),
                resid_agg=_score_resid_agg_from_cfg(cfg))
        cfg = getattr(A, "cfg", None)
        accum = _wafer_decision_accum_series(
            zc, step_plot, sensor_indices=sensor_indices,
            resid_agg=_score_resid_agg_from_cfg(cfg),
            wafer_agg=_score_wafer_agg_from_cfg(cfg),
            score_time_avg=_score_time_avg_from_threshold(threshold, cfg),
            score_final_avg=_score_final_avg_from_threshold(threshold, cfg),
            cfg=cfg,
            final_score=final_score)
    plot_vals = np.asarray(accum, dtype=np.float64)
    y_label = "anomaly score"
    line_label = "현재 wafer anomaly score"
    threshold_value = _decision_threshold_for_record(threshold, rec)
    scale_value = _display_scale_threshold_for_record(threshold, rec)
    threshold_line = _display_threshold_for_record(threshold, rec)
    scale = (
        float(scale_value)
        if scale_value is not None
        and np.isfinite(float(scale_value))
        and float(scale_value) > 1e-12
        else None
    )
    ok_mean_plot = _lookup_ok_mean(ok_bands, rec, step_plot, spans, sensor_idx=None)
    ok_min_plot, ok_max_plot = _lookup_ok_band(ok_bands, rec, step_plot, spans, sensor_idx=None)
    plot_vals_raw = np.asarray(plot_vals, dtype=np.float64)
    ok_mean_raw = None if ok_mean_plot is None else np.asarray(ok_mean_plot, dtype=np.float64)
    ok_min_raw = None if ok_min_plot is None else np.asarray(ok_min_plot, dtype=np.float64)
    ok_max_raw = None if ok_max_plot is None else np.asarray(ok_max_plot, dtype=np.float64)
    if scale is not None:
        plot_vals_plot = plot_vals_raw / scale
        ok_mean_display = None if ok_mean_raw is None else ok_mean_raw / scale
        ok_min_display = None if ok_min_raw is None else ok_min_raw / scale
        ok_max_display = None if ok_max_raw is None else ok_max_raw / scale
    else:
        plot_vals_plot = plot_vals_raw
        ok_mean_display = ok_mean_raw
        ok_min_display = ok_min_raw
        ok_max_display = ok_max_raw
    final_min, final_mean, final_max = _lookup_ok_total_final_stats(ok_bands, rec)
    if scale is not None:
        final_min = None if final_min is None else final_min / scale
        final_mean = None if final_mean is None else final_mean / scale
        final_max = None if final_max is None else final_max / scale
    if ok_mean_display is not None and final_mean is not None:
        ok_mean_display = _scale_curve_to_final_score(ok_mean_display, final_mean)
    if ok_min_display is not None and final_min is not None:
        ok_min_display = _scale_curve_to_final_score(ok_min_display, final_min)
    if ok_max_display is not None and final_max is not None:
        ok_max_display = _scale_curve_to_final_score(ok_max_display, final_max)
    if ok_mean_display is not None and ok_min_display is not None:
        ok_min_display = np.minimum(ok_min_display, ok_mean_display)
    if ok_mean_display is not None and ok_max_display is not None:
        ok_max_display = np.maximum(ok_max_display, ok_mean_display)
    if final_score is not None:
        display_final_score, _display_threshold = _display_score_threshold(
            final_score, threshold, rec)
        if np.isfinite(float(display_final_score)):
            plot_vals_plot = _scale_curve_to_final_score(
                plot_vals_plot, float(display_final_score))
    cap_wafer_display = scale is not None and _has_display_cap_excess(
        plot_vals_plot, ok_mean_display, ok_min_display, ok_max_display)
    plot_vals_for_plot = (
        _cap_display_values(plot_vals_plot) if cap_wafer_display else plot_vals_plot
    )
    ok_mean_for_plot = (
        _cap_display_values(ok_mean_display)
        if cap_wafer_display and ok_mean_display is not None else ok_mean_display
    )
    ok_min_for_plot = (
        _cap_display_values(ok_min_display)
        if cap_wafer_display and ok_min_display is not None else ok_min_display
    )
    ok_max_for_plot = (
        _cap_display_values(ok_max_display)
        if cap_wafer_display and ok_max_display is not None else ok_max_display
    )
    threshold_line_for_plot = (
        _cap_display_value(threshold_line)
        if cap_wafer_display and threshold_line is not None else threshold_line
    )

    fig, ax = plt.subplots(figsize=(20, 4.2))
    if (ok_min_for_plot is not None and ok_max_for_plot is not None
            and np.isfinite(ok_min_for_plot).any() and np.isfinite(ok_max_for_plot).any()):
        lo = np.maximum(0.0, ok_min_for_plot)
        hi = ok_max_for_plot
        valid_band = np.isfinite(lo) & np.isfinite(hi)
        if valid_band.any():
            ax.fill_between(t, lo, hi, where=valid_band, color="#87CEEB",
                            alpha=0.24, label="같은 recipe OK min~max", zorder=1)
    if ok_mean_for_plot is not None and np.isfinite(ok_mean_for_plot).any():
        ax.plot(t, ok_mean_for_plot, color="#1f77b4", lw=1.45,
                label="같은 recipe OK 평균 복원오차", zorder=2)
    ax.plot(t, plot_vals_for_plot, color="#333333", lw=1.45,
            label=line_label, zorder=3)
    if threshold_line_for_plot is not None:
        ax.axhline(float(threshold_line_for_plot), color="#d62728", ls="--", lw=1.2,
                   label="threshold", zorder=4)
    _draw_step_guides(ax, bounds, spans, label=True)
    vals_for_ylim = [plot_vals_for_plot[np.isfinite(plot_vals_for_plot)]]
    if ok_mean_for_plot is not None:
        vals_for_ylim.append(ok_mean_for_plot[np.isfinite(ok_mean_for_plot)])
    if ok_max_for_plot is not None:
        vals_for_ylim.append(ok_max_for_plot[np.isfinite(ok_max_for_plot)])
    if threshold_line_for_plot is not None:
        vals_for_ylim.append(np.asarray([float(threshold_line_for_plot)], dtype=np.float64))
    finite_vals = np.concatenate([v for v in vals_for_ylim if len(v)]) if vals_for_ylim else np.array([])
    ymax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    if cap_wafer_display:
        ax.set_ylim(0, _DISPLAY_SCORE_CAP)
    else:
        ax.set_ylim(0, ymax * 1.08 + 1e-9 if ymax > 0 else 1.0)
    ax.set_title(f"{rec['wafer_id']} · wafer anomaly score "
                 f"(ProcessStep 0 제외, interpolation x{max(1, int(rec.get('_interp_factor', 1) or 1))})",
                 fontsize=13)
    ax.set_ylabel(y_label)
    ax.set_xlabel(_time_axis_label(rec))
    ax.yaxis.grid(True, alpha=0.18)
    ax.xaxis.grid(False)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _write_wafer_accum_score_csv(
        Path(csv_path) if csv_path is not None else Path(out_path).with_suffix(".csv"),
        rec, orig_idx, step_plot,
        plot_vals_plot, ok_mean_display, ok_min_display, ok_max_display,
        threshold_line, 1.0)


def plot_recon_overlay(A, rec, out_path, top_k=None, pre=None, threshold=None) -> None:
    """전 센서(기본) 또는 상위 top_k: 실제(빨강) vs 기대(파랑) + 오차 음영.
    top_k=None → 모든 센서를 공정 흐름 순서로 그림. top_k는 이탈 큰 센서를 고른 뒤 흐름 순서로 배치."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    actual, expected, z = pre if pre is not None else _compute(A, rec)
    sensors = list(A.sensor_cols)
    t_full = actual.shape[0]
    plot_keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter(A, rec, t_full)
    if orig_idx.size == 0:
        return
    T = len(t)

    sig0 = float(getattr(A.cfg, "raw_error_level", 1.0))
    important = _sensor_order(z[plot_keep], sig0)        # 선별은 Step 0 제외 이탈 크기 기준
    if top_k and top_k > 0:
        order = _flow_sensor_order(A, important[:top_k])
    else:
        order = _flow_sensor_order(A)
    if not order:
        return

    ncol = 2
    nrow = (len(order) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(9 * ncol, 3.0 * nrow), squeeze=False)
    sig = float(getattr(A.cfg, "raw_error_level", 1.0))
    band_sigma = float(getattr(A.cfg, "explain_sigma", 4.0))
    large_sigma = float(getattr(A.cfg, "large_residual_sigma", band_sigma))
    min_highlight_width = int(getattr(A.cfg, "highlight_min_width", 10))

    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        if idx >= len(order):
            ax.axis("off"); continue
        j = order[idx]
        cur, exp = actual[plot_keep, j], expected[plot_keep, j]
        zc = z[plot_keep, j]
        sigma_raw = _sigma_series(A, rec, j, t_full)[plot_keep]
        ok_limit = _limit_series(A, rec, j, t_full)[plot_keep]
        resid_sigma, large_mask = _large_residual_mask(cur, exp, sigma_raw, large_sigma)
        _, large_events = _expanded_highlight_mask(large_mask, resid_sigma, min_highlight_width)
        large_events = _filter_isolated_step_transition_events(
            large_events, large_mask, bounds, A.cfg)
        exp_display = _smooth_step_boundary_prediction_for_display(exp, bounds, A.cfg)
        band_lo = exp_display - ok_limit
        band_hi = exp_display + ok_limit
        ylo = float(min(cur.min(), exp_display.min(), band_lo.min()))
        yhi = float(max(cur.max(), exp_display.max(), band_hi.max()))
        pad = (yhi - ylo) * 0.08 + 1e-6
        ax.fill_between(t, band_lo, band_hi, color="#87CEEB", alpha=0.22,
                        zorder=0, label="OK 기준 정상 범위")
        _draw_sigma_event_bands(ax, large_events, large_sigma, color="#d62728")
        _draw_step_guides(ax, bounds, spans, label=True)
        ax.plot(t, exp_display, color="#1f77b4", lw=1.4, ls="--", label="기대", zorder=4)
        ax.plot(t, cur, color="#d62728", lw=1.4, label="실제", zorder=5)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(f"{sensors[j]}", fontsize=9.5)
        ax.set_xlabel(_time_axis_label(rec), fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{rec['wafer_id']} — 하늘색=OK 기준 정상 범위 · "
                 f"빨간 음영=4σ 초과 x축 구간(진할수록 σ 큼) · "
                 f"빨강선=실제 · 파랑점선=기대 · "
                 f"ProcessStep 0 제외 · interpolation x{max(1, int(rec.get('_interp_factor', 1) or 1))}",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_recon_heatmap(A, rec, out_path, vmax: float = 5.0, z_pre=None) -> None:
    """error level 히트맵 — 가로=스텝, 세로=센서. 흰색(작음)→진한 빨강(큼), 스케일 0~vmax 고정."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()
    z = np.asarray(z_pre, dtype=np.float64) if z_pre is not None else (
        A.error_level(rec) if hasattr(A, "error_level") else np.abs(A.z_residual(rec)))
    ns = max(int(A.n_steps_global), 1)
    M = _step_map(z, rec.get("step"), ns).T            # (F, ns) → 행=센서, 열=스텝
    step = rec.get("step")
    if step is not None and len(step):
        present_steps = sorted({
            int(s) for s in np.asarray(step).astype(int)
            if int(s) != 0 and 0 <= int(s) < ns
        })
    else:
        present_steps = list(range(1, ns)) if ns > 1 else list(range(ns))
    if present_steps:
        M = M[:, present_steps]
        xlabels = [str(s) for s in present_steps]
    else:
        return
    sensors = list(A.sensor_cols)
    order = _flow_sensor_order(A)
    if order:
        M = M[order]
        sensors = [sensors[j] for j in order]
    n_steps_plot = len(xlabels)
    fig, ax = plt.subplots(figsize=(min(0.5 * n_steps_plot + 3, 24), 0.4 * len(sensors) + 2))
    im = ax.imshow(M, aspect="auto", cmap="Reds", origin="upper",
                   vmin=0.0, vmax=float(vmax))          # 흰→빨강, 0~5 고정(모든 wafer 동일)
    ax.set_xticks(range(n_steps_plot)); ax.set_xticklabels(xlabels)
    ax.set_xlabel("process step (0 excluded)")
    ax.set_yticks(range(len(sensors))); ax.set_yticklabels(sensors, fontsize=7)
    ax.set_ylabel("sensor")
    factor = max(1, int(rec.get("_interp_factor", 1) or 1)) if isinstance(rec, dict) else 1
    interp_note = f", interpolation x{factor}" if factor > 1 else ""
    ax.set_title(f"{rec['wafer_id']} — error level "
                 f"(ProcessStep 0 제외{interp_note}, 진할수록 IO 기대 대비 큰 이탈, 스케일 0~{vmax:.0f})",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="error level (스텝평균)")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_sensor_findings(A, rec, actual, expected, z, order, top_n: int = 12) -> list[Dict]:
    """Build wafer-level sensor finding rows from already computed arrays."""
    rows = []
    t_full = actual.shape[0]
    step = rec.get("step")
    sig = float(getattr(A.cfg, "raw_error_level", 1.0))
    rank = graph_sensor_rank(A.sensor_cols, A.cfg)
    for pos, j in enumerate([int(x) for x in order[:max(1, int(top_n))]], 1):
        zc = np.asarray(z[:, j], dtype=np.float64)
        if zc.size == 0:
            continue
        ti = int(np.nanargmax(zc))
        raw = float(actual[ti, j])
        exp = float(expected[ti, j])
        raw_abs = abs(raw - exp)
        ok_lim = float(_limit_series(A, rec, j, t_full)[ti])
        sigma_raw = float(_sigma_series(A, rec, j, t_full)[ti])
        resid_sigma = raw_abs / max(sigma_raw, 1e-12)
        peak_step = int(step[ti]) if step is not None and len(step) == t_full else None
        drivers = _io_drivers(A, rec, j, at_t=ti, top_k=2)
        row = {
            "wafer_id": rec["wafer_id"],
            "recipe": _recipe_key(rec),
            "rank": pos,
            "sensor": A.sensor_cols[j],
            "family": _sensor_family(A, A.sensor_cols[j]),
            "flow_rank": int(rank.get(str(A.sensor_cols[j]), 999999)),
            "peak_t": ti,
            "peak_t_original": round(_source_time_for_rec(rec, ti), 6),
            "interpolation": max(1, int(rec.get("_interp_factor", 1) or 1)),
            "peak_step": peak_step,
            "error_level": round(float(zc[ti]), 4),
            "residual_sigma": round(float(resid_sigma), 4),
            "raw_value": round(raw, 6),
            "expected_value": round(exp, 6),
            "raw_abs_error": round(float(raw_abs), 6),
            "ok_limit": round(ok_lim, 6),
            "direction": "높음" if raw > exp else "낮음",
            "over_raw_error_level": int(float(zc[ti]) >= sig),
            "io_top1": drivers[0][0] if len(drivers) >= 1 else "",
            "io_top2": drivers[1][0] if len(drivers) >= 2 else "",
        }
        rows.append(row)
    return rows


def _short_label(text, max_len: int = 32) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    if "_" in text:
        parts = text.split("_")
        lines, cur = [], ""
        for part in parts:
            piece = part if not cur else "_" + part
            if cur and len(cur) + len(piece) > max_len:
                lines.append(cur)
                cur = part
            else:
                cur += piece
        if cur:
            lines.append(cur)
        if len(lines) <= 2:
            return "\n".join(lines)
    return text[:max_len - 3] + "..."


def _safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _directed_graph_path(A, src: str, dst: str, max_hops: int = 80) -> list[str]:
    """Shortest directed sensor graph path between two nodes, keeping concrete IO nodes."""
    src, dst = str(src), str(dst)
    if not src or not dst or src == dst:
        return [src] if src else []
    graph = load_sensor_graph(A.cfg)
    if not graph.nodes or src not in graph.nodes or dst not in graph.nodes:
        return []
    q = deque([(src, [src])])
    seen = {src}
    while q:
        cur, path = q.popleft()
        if len(path) > max_hops:
            continue
        for nxt in sorted(graph.edges.get(cur, ()),
                          key=lambda n: graph.rank.get(str(n), 10**9)):
            if nxt in seen:
                continue
            npath = path + [nxt]
            if nxt == dst:
                return [n for n in npath if not is_pseudo_node(n)]
            seen.add(nxt)
            q.append((nxt, npath))
    return []


def _sensor_row_by_name(sensor_rows: list[Dict]) -> dict[str, Dict]:
    return {str(r.get("sensor")): r for r in sensor_rows if r.get("sensor")}


def _causal_row_from_sensor(rec, row: Dict, role: str, path_rank: int,
                            path_id: int, evidence: str) -> Dict:
    step_value = row.get("peak_step", "")
    if step_value is None or step_value == "":
        step_value = row.get("peak_t", "")
    return {
        "wafer_id": rec["wafer_id"],
        "recipe": _recipe_key(rec),
        "path_id": path_id,
        "path_rank": path_rank,
        "role": role,
        "sensor": row.get("sensor", ""),
        "step": step_value,
        "peak_t": row.get("peak_t", ""),
        "peak_t_original": row.get("peak_t_original", ""),
        "interpolation": row.get("interpolation", ""),
        "flow_rank": row.get("flow_rank", ""),
        "error_level": row.get("error_level", ""),
        "residual_sigma": row.get("residual_sigma", ""),
        "raw_value": row.get("raw_value", ""),
        "expected_value": row.get("expected_value", ""),
        "raw_abs_error": row.get("raw_abs_error", ""),
        "direction": row.get("direction", ""),
        "io_top1": row.get("io_top1", ""),
        "io_top2": row.get("io_top2", ""),
        "evidence": evidence,
    }


def _causal_step_value(row: Dict, default: int = 10**9) -> int:
    step = _safe_int(row.get("peak_step"), None)
    if step is not None:
        return step
    return _safe_int(row.get("peak_t"), default)


def build_causal_path_rows(A, rec, sensor_rows: list[Dict],
                           max_contributors: int = 6) -> list[Dict]:
    """Build one representative flow-aware RCA path for a wafer.

    The path is an explanation aid, not a causal proof: it prioritizes sensors
    that are earlier in process-flow order, earlier in ProcessStep, and still
    have high model-vs-raw error.
    """
    candidates_base = [
        r for r in sensor_rows
        if r.get("sensor")
        and _causal_step_value(r, None) is not None
    ]
    valid = [
        r for r in candidates_base
        if _safe_int(r.get("flow_rank"), 999999) < 999999
    ] or candidates_base
    if not valid:
        return []

    max_err = max(_safe_float(r.get("error_level")) for r in valid)
    root_floor = max(1.0, max_err * 0.35)
    root_pool = [r for r in valid if _safe_float(r.get("error_level")) >= root_floor]
    root = min(
        root_pool or valid,
        key=lambda r: (
            _causal_step_value(r),
            _safe_int(r.get("flow_rank"), 999999),
            -_safe_float(r.get("error_level")),
        ),
    )

    root_rank = _safe_int(root.get("flow_rank"), 999999)
    root_step = _causal_step_value(root)

    result_pool = [
        r for r in valid
        if r is not root
        and _causal_step_value(r) >= root_step
        and _safe_int(r.get("flow_rank"), 999999) >= root_rank
    ]
    if not result_pool:
        result_pool = [r for r in valid if r is not root]
    result = max(
        result_pool or [root],
        key=lambda r: (
            _safe_float(r.get("error_level")),
            _safe_float(r.get("residual_sigma")),
            _safe_int(r.get("flow_rank"), 999999),
            _causal_step_value(r),
        ),
    )
    result_rank = _safe_int(result.get("flow_rank"), root_rank)
    result_step = _causal_step_value(result, root_step)
    lo_rank, hi_rank = sorted((root_rank, result_rank))
    lo_step, hi_step = sorted((root_step, result_step))
    graph_path = _directed_graph_path(A, root.get("sensor", ""), result.get("sensor", ""))
    graph_nodes = set(graph_path)
    graph_dist = max(0, len(graph_path) - 1) if graph_path else abs(result_rank - root_rank)
    n_contrib = min(
        int(max_contributors),
        max(1, int(np.ceil(max(graph_dist, 1) / 3.0))),
    )

    candidates = []
    for r in valid:
        name = str(r.get("sensor"))
        if name in {root.get("sensor"), result.get("sensor")}:
            continue
        fr = _safe_int(r.get("flow_rank"), 999999)
        st = _causal_step_value(r)
        in_rank = lo_rank <= fr <= hi_rank
        in_time = (lo_step - 1) <= st <= (hi_step + 2)
        on_path = name in graph_nodes
        if not (on_path or (in_rank and in_time)):
            continue
        candidates.append((on_path, r))

    candidates = sorted(
        candidates,
        key=lambda item: (
            int(item[0]),
            _safe_float(item[1].get("error_level")),
            _safe_float(item[1].get("residual_sigma")),
            -abs(_safe_int(item[1].get("peak_step"), result_step) - root_step),
        ),
        reverse=True,
    )
    contributors = [r for _, r in candidates[:n_contrib]]
    if not contributors and len(valid) > 2:
        fallback = [
            r for r in valid
            if r not in (root, result)
        ]
        contributors = sorted(
            fallback,
            key=lambda r: (
                _safe_float(r.get("error_level")),
                _safe_float(r.get("residual_sigma")),
            ),
            reverse=True,
        )[:min(2, max_contributors)]
    sequence = [root] + sorted(
        contributors,
        key=lambda r: (_causal_step_value(r),
                       _safe_int(r.get("flow_rank"), 999999)),
    )
    if result.get("sensor") != root.get("sensor"):
        sequence.append(result)

    out = []
    seen = set()
    for idx, row in enumerate(sequence, 1):
        name = str(row.get("sensor"))
        if name in seen:
            continue
        seen.add(name)
        if row is root and row is result:
            role = "root_and_result"
            evidence = "최대 이탈 센서이며, 앞선 원인 후보가 별도로 분리되지 않음"
        elif row is root:
            role = "root_candidate"
            evidence = "결과 센서보다 flow/step상 앞선 고이탈 센서"
        elif row is result:
            role = "result_sensor"
            evidence = "이 wafer에서 error level이 가장 큰 결과 센서"
        else:
            role = "contributor"
            evidence = "원인 후보와 결과 센서 사이의 flow/step 경로상 고이탈 센서"
        out.append(_causal_row_from_sensor(rec, row, role, idx, 1, evidence))
    return out


def _representative_rca_text(causal_rows: list[Dict]) -> str:
    if not causal_rows:
        return "-"
    root = next((r for r in causal_rows if r["role"] in ("root_candidate", "root_and_result")),
                causal_rows[0])
    result = next((r for r in reversed(causal_rows)
                   if r["role"] in ("result_sensor", "root_and_result")),
                  causal_rows[-1])
    if root["sensor"] == result["sensor"]:
        return f"{root['sensor']}(Step {root['step']})"
    contrib = [r for r in causal_rows if r["role"] == "contributor"]
    mid = f" + 기여 {len(contrib)}개" if contrib else ""
    return f"{root['sensor']}(Step {root['step']}) -> {result['sensor']}(Step {result['step']}){mid}"


def plot_causal_path(A, rec, causal_rows: list[Dict], out_path) -> None:
    if not causal_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    _setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [r for r in causal_rows if r.get("sensor")]
    if not rows:
        return
    rank = graph_sensor_rank(A.sensor_cols, A.cfg)
    rows = sorted(rows, key=lambda r: _safe_int(r.get("path_rank"), 10**9))
    y_sensors = sorted(
        {str(r["sensor"]) for r in rows},
        key=lambda s: (rank.get(s, 999999), s),
    )
    y_map = {s: i for i, s in enumerate(y_sensors)}
    steps = [_safe_int(r.get("step"), 0) for r in rows]
    xmin, xmax = min(steps), max(steps)
    if xmin == xmax:
        xmin -= 1
        xmax += 1

    fig_h = max(4.5, 0.45 * len(y_sensors) + 2.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    role_style = {
        "root_candidate": ("#d62728", "o", "원인 후보"),
        "root_and_result": ("#d62728", "*", "원인/결과 동일"),
        "contributor": ("#ff7f0e", "s", "기여 센서"),
        "result_sensor": ("#1f77b4", "D", "결과 센서"),
    }

    for a, b in zip(rows[:-1], rows[1:]):
        ax.annotate(
            "",
            xy=(_safe_int(b.get("step"), 0), y_map[str(b["sensor"])]),
            xytext=(_safe_int(a.get("step"), 0), y_map[str(a["sensor"])]),
            arrowprops=dict(arrowstyle="->", color="#777777", lw=1.0,
                            alpha=0.65, shrinkA=9, shrinkB=9),
            zorder=1,
        )

    for r in rows:
        role = str(r.get("role", "contributor"))
        color, marker, _ = role_style.get(role, role_style["contributor"])
        x = _safe_int(r.get("step"), 0)
        y = y_map[str(r["sensor"])]
        err = _safe_float(r.get("error_level"))
        size = min(420, 80 + err * 18.0)
        ax.scatter([x], [y], s=size, marker=marker, color=color,
                   edgecolor="white", linewidth=0.9, zorder=3, alpha=0.92)
        t_src = r.get("peak_t_original", "")
        t_note = ""
        try:
            if t_src != "":
                t_note = f", t {float(t_src):.2f}"
        except Exception:
            t_note = ""
        label = (f"{_short_label(r['sensor'], 24)}\n"
                 f"err {err:.1f}, 실제 {r.get('direction', '-')}{t_note}")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6),
                    ha="left", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=color, alpha=0.88),
                    zorder=4)
        ios = [v for v in (r.get("io_top1"), r.get("io_top2")) if v]
        if ios:
            ax.annotate("IO: " + " / ".join(ios), (x, y),
                        textcoords="offset points", xytext=(8, -18),
                        ha="left", va="top", fontsize=7, color="#6a3d9a",
                        zorder=4)

    ax.set_xlim(xmin - 0.8, xmax + 1.2)
    ax.set_yticks(range(len(y_sensors)))
    ax.set_yticklabels([_short_label(s, 34) for s in y_sensors], fontsize=8)
    ax.set_xlabel("ProcessStep")
    ax.set_ylabel("Sensor flow order")
    ax.yaxis.grid(True, alpha=0.18)
    ax.xaxis.grid(False)
    ax.set_title(f"{rec['wafer_id']} - flow-aware RCA path", fontsize=13, pad=12)
    handles = [
        Line2D([0], [0], marker=marker, color="w", label=label,
               markerfacecolor=color, markeredgecolor="white", markersize=9)
        for color, marker, label in role_style.values()
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_step_summary(A, rec, actual, expected, z) -> list[Dict]:
    step = rec.get("step")
    if step is None or len(step) != z.shape[0]:
        return []
    s = np.asarray(step).astype(int)
    rows = []
    for st in sorted(int(x) for x in np.unique(s) if int(x) != 0):
        mask = s == st
        if not mask.any():
            continue
        local = np.asarray(z[mask], dtype=np.float64)
        if local.size == 0:
            continue
        local_flat = int(np.nanargmax(local))
        local_t, j = np.unravel_index(local_flat, local.shape)
        ti = int(np.flatnonzero(mask)[local_t])
        j = int(j)
        raw = float(actual[ti, j])
        exp = float(expected[ti, j])
        rows.append({
            "wafer_id": rec["wafer_id"],
            "recipe": _recipe_key(rec),
            "step": st,
            "max_error_level": round(float(np.nanmax(local)), 4),
            "mean_error_level": round(float(np.nanmean(local)), 4),
            "max_sensor": A.sensor_cols[j],
            "family": _sensor_family(A, A.sensor_cols[j]),
            "peak_t": ti,
            "peak_t_original": round(_source_time_for_rec(rec, ti), 6),
            "interpolation": max(1, int(rec.get("_interp_factor", 1) or 1)),
            "raw_value": round(raw, 6),
            "expected_value": round(exp, 6),
            "raw_abs_error": round(abs(raw - exp), 6),
            "io_top1": "",
            "io_top2": "",
        })
    top_idx = sorted(range(len(rows)), key=lambda i: float(rows[i]["max_error_level"]),
                     reverse=True)[:5]
    for idx in top_idx:
        r = rows[idx]
        try:
            j = A.sensor_cols.index(r["max_sensor"])
        except ValueError:
            continue
        drivers = _io_drivers(A, rec, int(j), at_t=int(r["peak_t"]), top_k=2)
        r["io_top1"] = drivers[0][0] if len(drivers) >= 1 else ""
        r["io_top2"] = drivers[1][0] if len(drivers) >= 2 else ""
    return rows


def build_threshold_exceedances(A, rec, actual, expected, z, threshold,
                                label=None) -> list[Dict]:
    """List every sensor x ProcessStep group whose accumulated score exceeds threshold.

    The rows intentionally use the original, non-interpolated arrays so this CSV
    matches the same resolution used for wafer scoring and threshold decisions.
    """
    if threshold is None:
        return []
    step = rec.get("step")
    if step is None or len(step) != z.shape[0]:
        step_arr = np.ones(z.shape[0], dtype=int)
    else:
        step_arr = np.asarray(step).astype(int)
    level = np.asarray(z, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    excess = np.maximum(0.0, level - 1.0)
    mode = _threshold_mode(threshold)
    label_name = "BLIND" if label is None else ("NG" if int(label) == 1 else "OK")
    rows = []
    for st in sorted(int(x) for x in np.unique(step_arr) if int(x) != 0):
        mask = step_arr == st
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        step_scores = excess[mask].sum(axis=0)
        for j, sensor in enumerate(A.sensor_cols):
            th = _sensor_step_threshold(threshold, rec, str(sensor), st)
            if th is None:
                continue
            score = float(step_scores[int(j)])
            if not np.isfinite(score) or score < float(th):
                continue
            local_level = level[mask, int(j)]
            local_peak = int(np.nanargmax(local_level)) if local_level.size else 0
            ti = int(idx[local_peak])
            raw = float(actual[ti, int(j)])
            exp = float(expected[ti, int(j)])
            drivers = _io_drivers(A, rec, int(j), at_t=ti, top_k=2)
            rows.append({
                "wafer_id": rec["wafer_id"],
                "recipe": _recipe_key(rec),
                "label": label_name,
                "threshold_wise": mode,
                "sensor": str(sensor),
                "family": _sensor_family(A, str(sensor)),
                "step": int(st),
                "step_accum_score": round(score, 6),
                "threshold": round(float(th), 6),
                "threshold_ratio": round(score / max(float(th), 1e-12), 6),
                "peak_t": ti,
                "peak_error_level": round(float(level[ti, int(j)]), 6),
                "raw_value": round(raw, 6),
                "expected_value": round(exp, 6),
                "raw_abs_error": round(abs(raw - exp), 6),
                "direction": "높음" if raw > exp else "낮음",
                "io_top1": drivers[0][0] if len(drivers) >= 1 else "",
                "io_top2": drivers[1][0] if len(drivers) >= 2 else "",
            })
    return sorted(rows, key=lambda r: (
        -float(r["threshold_ratio"]),
        -float(r["step_accum_score"]),
        int(r["step"]),
        str(r["sensor"]),
    ))


def _wrap_step_label(value, width=18, max_lines=3) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    parts = text.split("_")
    lines, cur = [], ""
    for part in parts:
        piece = part if not cur else "_" + part
        if cur and len(cur) + len(piece) > width:
            lines.append(cur)
            cur = part
        else:
            cur += piece
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        keep = lines[:max_lines - 1]
        tail = "_".join(lines[max_lines - 1:])
        if len(tail) > width + 4:
            tail = tail[:width + 1] + "..."
        lines = keep + [tail]
    return "\n".join(lines)


def _annotate_step_peaks(ax, step_rows: list[Dict], max_count: int = 4) -> None:
    peaks = sorted(step_rows, key=lambda x: float(x["max_error_level"]),
                   reverse=True)[:max_count]
    if not peaks:
        return

    y_values = [float(r["max_error_level"]) for r in step_rows]
    y_values += [float(r.get("mean_error_level", 0.0)) for r in step_rows]
    data_min = min(y_values)
    data_max = max(y_values)
    span = max(data_max - data_min, abs(data_max) * 0.15, 1.0)
    bottom = min(0.0, data_min - span * 0.08)
    top = data_max + span * 0.68
    ax.set_ylim(bottom, top)

    steps = [float(r["step"]) for r in step_rows]
    x_span = max(max(steps) - min(steps), 1.0)
    # Adjacent process steps are visually dense in recipe-specific summaries.
    # Keep labels at different heights when their step distance is <= 2.
    close_step = max(2.05, x_span * 0.045)
    placed: list[tuple[float, int]] = []
    y_levels = [0.10, 0.24, 0.38, 0.52]

    for r in sorted(peaks, key=lambda x: float(x["step"])):
        x = float(r["step"])
        y = float(r["max_error_level"])
        used_levels = {
            level for prev_x, level in placed
            if abs(prev_x - x) <= close_step
        }
        level = next((i for i in range(len(y_levels)) if i not in used_levels),
                     len(placed) % len(y_levels))
        placed.append((x, level))
        text_y = data_max + span * y_levels[level]
        ax.annotate(
            _wrap_step_label(r["max_sensor"]),
            xy=(x, y),
            xytext=(x, text_y),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=6.5,
            linespacing=0.9,
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6,
                            alpha=0.75, shrinkA=0, shrinkB=3),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#cccccc", alpha=0.88),
            clip_on=True,
        )


def plot_step_summary(step_rows: list[Dict], out_path) -> None:
    if not step_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    _setup_font()
    import matplotlib.pyplot as plt
    steps = [r["step"] for r in step_rows]
    max_v = [float(r["max_error_level"]) for r in step_rows]
    mean_v = [float(r["mean_error_level"]) for r in step_rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(steps) + 2), 4.2))
    ax.plot(steps, max_v, color="#d62728", marker="o", lw=1.4, label="step max error level")
    ax.plot(steps, mean_v, color="#333333", marker=".", lw=1.1, label="step mean error level")
    _annotate_step_peaks(ax, step_rows, max_count=5)
    ax.set_xlabel("ProcessStep")
    ax.set_ylabel("error level")
    ax.set_title(f"{step_rows[0]['wafer_id']} - ProcessStep별 error 요약", pad=14)
    ax.yaxis.grid(True, alpha=0.2)
    ax.xaxis.grid(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_wafer_summary(A, rec, label, actual, expected, z, sensor_rows, step_rows,
                       score, threshold, out_path, causal_rows=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    _setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    top = sensor_rows[:8]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.25], width_ratios=[1.15, 1.0])
    ax_info = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_step = fig.add_subplot(gs[1, 0])
    ax_tbl = fig.add_subplot(gs[1, 1])

    th = _threshold_for_record(threshold, rec)
    pred = "NG" if th is not None and float(score) >= float(th) else ("-" if th is None else "OK")
    truth = "BLIND" if label is None else ("NG" if int(label) == 1 else "OK")
    display_score, display_threshold = _display_score_threshold(score, threshold, rec)

    info = [
        ["wafer", rec["wafer_id"]],
        ["recipe", _recipe_key(rec)],
        ["actual / pred", f"{truth} / {pred}"],
        ["score", f"{float(display_score):.4f}"],
        ["threshold", "-" if display_threshold is None else f"{float(display_threshold):.4f}"],
        ["target sensors", f"{len(A.sensor_cols)} ({', '.join(sorted({_sensor_family(A, s) for s in A.sensor_cols}))})"],
        ["대표 RCA", _representative_rca_text(causal_rows or [])],
    ]
    ax_info.axis("off")
    tbl = ax_info.table(cellText=info, colLabels=["항목", "값"],
                        cellLoc="left", colWidths=[0.22, 0.78],
                        bbox=[0.0, 0.03, 1.0, 0.88])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.PAD = 0.025 if c == 1 else 0.035
        cell.get_text().set_ha("left")
        cell.get_text().set_wrap(True)
        if (r, c) == (1, 1):
            cell.get_text().set_wrap(False)
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold", color="#222222")
    ax_info.set_title("웨이퍼 판단 요약", fontsize=13, y=0.96, pad=0)

    if top:
        names = [r["sensor"] for r in top][::-1]
        vals = [float(r["error_level"]) for r in top][::-1]
        fam_colors = {"P": "#1f77b4", "T": "#d62728", "H": "#ff7f0e", "D": "#9467bd"}
        cols = [fam_colors.get(r["family"], "#7f7f7f") for r in top][::-1]
        ax_bar.barh(range(len(names)), vals, color=cols)
        ax_bar.set_yticks(range(len(names)))
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.set_xlabel("peak error level")
        ax_bar.set_title("중요 센서 top", fontsize=13)
        handles = [Patch(color=c, label=f) for f, c in fam_colors.items()
                   if any(r["family"] == f for r in top)]
        if handles:
            ax_bar.legend(handles=handles, fontsize=7, loc="lower right")
    else:
        ax_bar.axis("off")

    if step_rows:
        steps = [r["step"] for r in step_rows]
        max_v = [float(r["max_error_level"]) for r in step_rows]
        mean_v = [float(r["mean_error_level"]) for r in step_rows]
        ax_step.plot(steps, max_v, color="#d62728", marker="o", lw=1.4, label="max")
        ax_step.plot(steps, mean_v, color="#333333", marker=".", lw=1.1, label="mean")
        _annotate_step_peaks(ax_step, step_rows, max_count=4)
        ax_step.set_xlabel("ProcessStep")
        ax_step.set_ylabel("error level")
        ax_step.set_title("ProcessStep별 이상 집중도", fontsize=13, pad=14)
        ax_step.yaxis.grid(True, alpha=0.2)
        ax_step.xaxis.grid(False)
        ax_step.legend(fontsize=8)
    else:
        ax_step.axis("off")

    table_rows = []
    for r in top[:6]:
        table_rows.append([
            r["rank"], r["family"], r["sensor"], r["peak_step"],
            f"{float(r['error_level']):.2f}",
            f"{float(r['raw_abs_error']):.3g}",
            r["direction"],
            " / ".join([x for x in (r.get("io_top1"), r.get("io_top2")) if x]),
        ])
    ax_tbl.axis("off")
    if table_rows:
        t2 = ax_tbl.table(cellText=table_rows,
                          colLabels=["#", "군", "센서", "Step", "level", "|err|", "실제", "IO top2"],
                          cellLoc="left",
                          colWidths=[0.06, 0.07, 0.30, 0.08, 0.10, 0.10, 0.08, 0.21],
                          bbox=[0.0, 0.03, 1.0, 0.88])
        t2.auto_set_font_size(False)
        t2.set_fontsize(8)
        t2.scale(1, 1.35)
        for c in range(8):
            cell = t2[0, c]
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold", color="#222222")
    ax_tbl.set_title("중요 센서 상세", fontsize=13, y=0.96, pad=0)

    fig.suptitle(f"{rec['wafer_id']} - TCN explain summary", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_sensor_findings(rows: list[Dict], csv_path, md_path) -> None:
    import csv as _csv
    if not rows:
        return
    fields = ["wafer_id", "recipe", "rank", "sensor", "family", "flow_rank",
              "peak_step", "peak_t", "peak_t_original", "interpolation",
              "error_level", "residual_sigma",
              "raw_value", "expected_value", "raw_abs_error", "ok_limit",
              "direction", "over_raw_error_level", "io_top1", "io_top2"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    lines = ["# Sensor Findings", ""]
    cur = None
    for r in rows:
        if r["wafer_id"] != cur:
            cur = r["wafer_id"]
            lines.extend(["", f"## {cur} ({r.get('recipe', 'unknown')})"])
        io = " / ".join([x for x in (r.get("io_top1"), r.get("io_top2")) if x]) or "-"
        lines.append(
            f"- #{r['rank']} {r['sensor']} (Step {r['peak_step']}): "
            f"actual {r['direction']}, error level {r['error_level']}, "
            f"|err| {r['raw_abs_error']} / OK {r['ok_limit']}, IO {io}"
        )
    Path(md_path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _remove_file_if_exists(path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def build_topk_sensor_selection_rows(A, rec, label, level,
                                     selected_indices, topk: int,
                                     resid_agg: str, wafer_agg: str) -> list[Dict]:
    """Return per-sensor rows documenting which sensors fed the wafer score."""
    try:
        topk = max(0, int(topk or 0))
    except Exception:
        topk = 0
    if topk <= 0:
        return []
    level = np.asarray(level, dtype=np.float64)
    per_sensor = _per_sensor_reconstruction_scores_from_level(level, resid_agg)
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_indices = selected_indices[
        (selected_indices >= 0) & (selected_indices < len(A.sensor_cols))
    ]
    rank_by_sensor = {int(j): rank + 1 for rank, j in enumerate(selected_indices.tolist())}
    selected_count = int(len(selected_indices))
    if str(wafer_agg or "").strip().lower() in {"topk_mean", "mean"} and selected_count > 0:
        contribution_factor = 1.0 / float(selected_count)
    elif str(wafer_agg or "").strip().lower() == "sum":
        contribution_factor = 1.0
    else:
        contribution_factor = float("nan")
    rows = []
    for j, sensor in enumerate(A.sensor_cols):
        score = float(per_sensor[j]) if j < len(per_sensor) and np.isfinite(per_sensor[j]) else 0.0
        used = j in rank_by_sensor
        contribution = score * contribution_factor if used and np.isfinite(contribution_factor) else ""
        rows.append({
            "wafer_id": rec.get("wafer_id", ""),
            "recipe": _recipe_key(rec),
            "label": "BLIND" if label is None else ("NG" if int(label) == 1 else "OK"),
            "requested_topk": topk,
            "selected_count": selected_count,
            "resid_agg": resid_agg,
            "wafer_agg": wafer_agg,
            "sensor_index": int(j),
            "sensor": str(sensor),
            "used_in_score": int(used),
            "topk_rank": rank_by_sensor.get(j, ""),
            "sensor_score": score,
            "contribution_factor": (
                "" if not used or not np.isfinite(contribution_factor)
                else contribution_factor
            ),
            "contribution_score": contribution,
        })
    return rows


def write_topk_sensor_selection(rows: list[Dict], csv_path, json_path) -> None:
    import csv as _csv
    if not rows:
        _remove_file_if_exists(csv_path)
        _remove_file_if_exists(json_path)
        return
    fields = [
        "wafer_id", "recipe", "label", "requested_topk", "selected_count",
        "resid_agg", "wafer_agg", "sensor_index", "sensor",
        "used_in_score", "topk_rank", "sensor_score",
        "contribution_factor", "contribution_score",
    ]
    rows = sorted(rows, key=lambda r: (
        str(r.get("wafer_id", "")),
        -int(r.get("used_in_score", 0) or 0),
        int(r.get("topk_rank", 10**9) or 10**9),
        -float(r.get("sensor_score", 0.0) or 0.0),
        int(r.get("sensor_index", 0) or 0),
    ))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    by_wafer = {}
    for r in rows:
        wid = str(r.get("wafer_id", ""))
        doc = by_wafer.setdefault(wid, {
            "wafer_id": wid,
            "recipe": r.get("recipe", ""),
            "label": r.get("label", ""),
            "requested_topk": r.get("requested_topk", ""),
            "selected_count": r.get("selected_count", ""),
            "resid_agg": r.get("resid_agg", ""),
            "wafer_agg": r.get("wafer_agg", ""),
            "used_sensors": [],
            "unused_sensors": [],
        })
        item = {
            "sensor_index": r.get("sensor_index"),
            "sensor": r.get("sensor"),
            "sensor_score": r.get("sensor_score"),
        }
        if int(r.get("used_in_score", 0) or 0):
            item["topk_rank"] = r.get("topk_rank")
            item["contribution_factor"] = r.get("contribution_factor")
            item["contribution_score"] = r.get("contribution_score")
            doc["used_sensors"].append(item)
        else:
            doc["unused_sensors"].append(item)
    Path(json_path).write_text(
        json.dumps(list(by_wafer.values()), ensure_ascii=False, indent=2,
                   default=_json_default),
        encoding="utf-8")


def write_step_summary(rows: list[Dict], csv_path) -> None:
    import csv as _csv
    if not rows:
        return
    fields = ["wafer_id", "recipe", "step", "max_error_level", "mean_error_level",
              "max_sensor", "family", "peak_t", "peak_t_original", "interpolation",
              "raw_value", "expected_value",
              "raw_abs_error", "io_top1", "io_top2"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_threshold_exceedances(rows: list[Dict], csv_path) -> None:
    import csv as _csv
    if not rows:
        _remove_file_if_exists(csv_path)
        return
    fields = ["wafer_id", "recipe", "label", "threshold_wise", "sensor", "family",
              "step", "step_accum_score", "threshold", "threshold_ratio",
              "peak_t", "peak_error_level", "raw_value", "expected_value",
              "raw_abs_error", "direction", "io_top1", "io_top2"]
    rows = sorted(rows, key=lambda r: (
        str(r.get("wafer_id", "")),
        -float(r.get("threshold_ratio", 0.0) or 0.0),
        -float(r.get("step_accum_score", 0.0) or 0.0),
        int(r.get("step", 0) or 0),
        str(r.get("sensor", "")),
    ))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_threshold_exceedances_by_step(rows: list[Dict], targets, csv_path) -> None:
    import csv as _csv
    if not rows:
        _remove_file_if_exists(csv_path)
        return
    max_step = 0
    for rec, _label in targets:
        step = rec.get("step")
        if step is None or not len(step):
            continue
        vals = [int(x) for x in np.asarray(step).astype(int) if int(x) != 0]
        if vals:
            max_step = max(max_step, max(vals))
    for r in rows:
        try:
            max_step = max(max_step, int(r.get("step", 0) or 0))
        except Exception:
            pass

    step_cols = [str(i) for i in range(1, max_step + 1)]
    fields = ["wafer_id"] + step_cols + ["n_exceedances"]
    by_wafer: dict[str, dict[int, list[str]]] = {}
    for r in rows:
        wid = str(r.get("wafer_id", ""))
        sensor = str(r.get("sensor", ""))
        if not wid or not sensor:
            continue
        try:
            st = int(r.get("step", 0) or 0)
        except Exception:
            continue
        if st <= 0:
            continue
        bucket = by_wafer.setdefault(wid, {}).setdefault(st, [])
        if sensor not in bucket:
            bucket.append(sensor)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec, _label in targets:
            wid = str(rec["wafer_id"])
            step_map = by_wafer.get(wid, {})
            out = {"wafer_id": wid}
            n_exceedances = 0
            for col in step_cols:
                sensors = step_map.get(int(col), [])
                n_exceedances += len(sensors)
                out[col] = " | ".join(sensors) if sensors else "-"
            out["n_exceedances"] = n_exceedances
            w.writerow(out)


def write_causal_paths(rows: list[Dict], csv_path, md_path) -> None:
    import csv as _csv
    if not rows:
        return
    fields = ["wafer_id", "recipe", "path_id", "path_rank", "role", "sensor",
              "step", "peak_t", "peak_t_original", "interpolation",
              "flow_rank", "error_level", "residual_sigma",
              "raw_value", "expected_value", "raw_abs_error", "direction",
              "io_top1", "io_top2", "evidence"]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    lines = ["# Flow-aware RCA paths", ""]
    cur = None
    for r in rows:
        if r["wafer_id"] != cur:
            cur = r["wafer_id"]
            lines.extend(["", f"## {cur} ({r.get('recipe', 'unknown')})"])
        io = " / ".join([x for x in (r.get("io_top1"), r.get("io_top2")) if x]) or "-"
        lines.append(
            f"- {r['path_rank']}. `{r['role']}` {r['sensor']} "
            f"(Step {r['step']}, err {r['error_level']}, 실제 {r['direction']}, IO {io})"
        )
        lines.append(f"  - 근거: {r.get('evidence', '-')}")
    Path(md_path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _safe_write_report_output(label: str, writer, *args) -> bool:
    try:
        writer(*args)
        return True
    except Exception as e:
        print(f"  ⚠ {label} 생성 실패: {e}")
        return False


def perf_summary(A, val_labeled, cfg, threshold=None, params=None, scores=None,
                 score_kind=None) -> Dict:
    """검증셋 성능 요약: 복원 MSE·threshold·정답수·오탐/미검출·AUROC·점수분포.

    scores 주면(서치가 저장한 wafer별 점수) 재채점 안 하고 그 점수로 평가 →
    search 결과와 100% 일치(모델 재학습/비결정성 무관).
    """
    recs = [r for r, _ in val_labeled]
    y = [lab for _, lab in val_labeled]
    if scores is not None:                              # ★ 서치 저장 점수 재사용
        p = [float(scores.get(r["wafer_id"], 0.0)) for r in recs]
    else:
        p = [float(x) for x in _prediction_scores(A, recs, threshold, score_kind)]
    display_p = [
        _display_score_threshold(sc, threshold, rec)[0]
        for rec, sc in zip(recs, p)
    ]
    if threshold is not None:
        preds = []
        for rec, sc in zip(recs, p):
            th = _threshold_for_record(threshold, rec)
            preds.append(0 if th is None else int(float(sc) >= float(th)))
        vc = _metrics_from_predictions(y, preds, p)
        vc["threshold"] = threshold
        vc["threshold_raw"] = threshold
        vc["threshold_wise"] = _threshold_mode(threshold)
        thr = threshold
    else:
        vc = _correct_from_scores_by_recipe(recs, y, p, cfg)
        thr = (vc["threshold_by_recipe"] if vc else threshold)
    # 검증 OK 복원 MSE (표준화 단위; 학습 val_mse 와 동일 척도)
    mses = []
    for r, lab in val_labeled:
        if lab == 0:
            res = (_actual(A, r) - A.predict(r)) / A.sen_std
            mses.append(float(np.mean(res ** 2)))
    val_mse = float(np.mean(mses)) if mses else float("nan")
    yv = np.asarray(y); pv = np.asarray(display_p)
    okv, ngv = pv[yv == 0], pv[yv == 1]
    rng = lambda a: (float(a.min()), float(a.mean()), float(a.max())) if a.size else None
    return {"vc": vc, "threshold": thr, "val_mse": val_mse,
            "ok_score": rng(okv), "ng_score": rng(ngv),
            "n_params": int(sum(q.numel() for q in A.net.parameters())) if A.net else 0,
            "window": (getattr(A, "w", getattr(A, "past_window", None)),
                       getattr(A, "h", getattr(A, "future_window", None)),
                       getattr(A, "stride", None)),
            "arch": getattr(A, "arch", {}),
            "params": params,
            "scores": {recs[i]["wafer_id"]: display_p[i] for i in range(len(recs))},
            "raw_scores": {recs[i]["wafer_id"]: p[i] for i in range(len(recs))}}


def _format_display_threshold(threshold) -> str:
    value = _display_threshold_json_value(threshold)
    if value is None:
        return "—"
    if not isinstance(value, dict):
        return f"{float(value):.4f}"
    mode = str(value.get("threshold_wise", "")).lower()
    pieces = []
    global_value = value.get("global_threshold_raw", value.get("global_threshold"))
    if global_value is not None:
        try:
            pieces.append(f"global:{float(global_value):.4f}")
        except Exception:
            pass
    table = value.get("threshold_by_recipe")
    if isinstance(table, dict):
        for recipe, th in sorted(table.items()):
            try:
                pieces.append(f"{recipe}:{float(th):.4f}")
            except Exception:
                pass
    if pieces:
        return "; ".join(pieces)
    if mode in {"all", "recipe"}:
        return "1.0000"
    return _format_threshold(value)


def plot_perf_table(summary: Dict, cfg, out_path) -> None:
    """성능 요약을 표 이미지로."""
    _setup_font()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vc = summary["vc"]; thr = summary["threshold"]; pr = summary.get("params") or {}
    arch_info = summary.get("arch") or {}
    f3 = lambda t: "{:.3f} / {:.3f} / {:.3f}".format(*t) if t else "—"
    arch = (
        f"ch{pr.get('tcn_channels', arch_info.get('channels', '?'))} "
        f"blk{pr.get('tcn_blocks', arch_info.get('blocks', '?'))} "
        f"k{pr.get('tcn_kernel', arch_info.get('kernel', '?'))} "
        f"drop{pr.get('tcn_dropout', arch_info.get('dropout', '?'))} "
        f"cart={arch_info.get('h_cart_mode', getattr(cfg, 'h_cart_mode', '?'))} "
        f"virtual={arch_info.get('virtual_sensor_mode', getattr(cfg, 'virtual_sensor_mode', 'off'))}"
    )
    shading_note = (
        f"하늘색=기대±OK 기준 오차, "
        f"빨간 x축 band=|실제-예측|≥{getattr(cfg, 'large_residual_sigma', 4.0)}σ "
        f"(진할수록 σ 큼)"
    )
    display_filters = []
    if bool(getattr(cfg, "suppress_isolated_step_transition_shading", False)):
        display_filters.append(
            "고립 step 경계 음영 제외"
            f"(경계±{getattr(cfg, 'step_transition_shading_window', 2)}, "
            f"주변±{getattr(cfg, 'step_transition_context_window', 8)})"
        )
    if bool(getattr(cfg, "smooth_step_boundary_prediction", False)):
        display_filters.append(
            "step 경계 예측선 표시 보정"
            f"(경계±{getattr(cfg, 'step_boundary_smooth_window', 2)})"
        )
    if display_filters:
        shading_note += " · 표시보정=" + ", ".join(display_filters)
    rows = [
        ("검증 OK 복원 MSE (표준화)", f"{summary['val_mse']:.4f}"),
        ("threshold (≥ → 불량)", _format_display_threshold(thr)),
        ("정답수", f"{vc['correct']}/{vc['n']}" if vc else "—"),
        ("양품 통과 (TN)", f"{vc['tn']}/{vc['n_ok']}" if vc else "—"),
        ("불량 검출 (TP)", f"{vc['tp']}/{vc['n_ng']}" if vc else "—"),
        ("오탐 FP / 미검출 FN", f"{vc['fp']} / {vc['fn']}" if vc else "—"),
        ("AUROC", f"{vc['auroc']}" if vc else "—"),
        ("OK 점수 min/평균/max", f3(summary["ok_score"])),
        ("NG 점수 min/평균/max", f3(summary["ng_score"])),
        ("모델 파라미터 수", f"{summary['n_params']:,}"),
        ("윈도우 past/future/stride", "{}/{}/{}".format(*summary["window"])),
        ("아키텍처", arch),
        ("음영 기준", shading_note),
        ("하단 누적 방식", "센서별 누적(스텝 변경 시 리셋 없음)"
                  if _error_accum_mode(cfg) == "sensor"
                  else "ProcessStep별 누적(스텝 변경 시 리셋)"),
        ("threshold 방식", str(getattr(cfg, "threshold_wise", "recipe"))),
        ("설명 해상도", f"interpolation x{getattr(cfg, 'explain_interpolation', 1)} "
                   "(센서/예측/오차 선형 보간, IO/Step 상태 유지)"),
        ("센서군", str(getattr(cfg, "sensor_keep", None) or getattr(cfg, "sensor_keep_prefixes", None) or "전체")),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 0.46 * len(rows) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["항목", "값"], loc="center", cellLoc="left",
                   colWidths=[0.62, 0.38])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for c in range(2):                                  # 헤더 강조
        cell = tbl[0, c]; cell.set_facecolor("#1f77b4"); cell.set_text_props(color="white")
    ttl = f"IO-Recon-TCN 설명 모델 성능 요약 (검증 {vc['n']}장)" if vc else "IO-Recon-TCN 설명 모델 성능 요약"
    ax.set_title(ttl, fontsize=12, pad=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def _sort_prediction_rows(rows, prediction_order: str, score_index: int):
    if str(prediction_order).lower() == "recipe":
        recipe_order = {str(r): i for i, r in enumerate(getattr(IOConfig(), "recipe_names", ()) or ())}
        return sorted(
            rows,
            key=lambda t: (recipe_order.get(_recipe_key(t[0]), 10_000),
                           _recipe_key(t[0]), -float(t[score_index]), t[1]))
    return sorted(rows, key=lambda t: (-float(t[score_index]), t[1]))


def _draw_display_thresholds(ax, xs, display_thresholds,
                             color="#2ca02c", label="threshold"):
    pairs = [
        (float(x), float(th))
        for x, th in zip(xs, display_thresholds)
        if th is not None and np.isfinite(float(th))
    ]
    if not pairs:
        return []
    unique = sorted({round(th, 10) for _x, th in pairs})
    if len(unique) == 1:
        ax.axhline(unique[0], color=color, ls="--", lw=1.0, alpha=0.65,
                   label=label)
    else:
        first = True
        for x, th in pairs:
            ax.hlines(th, x - 0.45, x + 0.45, color=color,
                      linestyles="--", lw=1.1, alpha=0.75,
                      label=label if first else None)
            first = False
    return [th for _x, th in pairs]


def _write_threshold_ratio_plot(rows, threshold, png_path, prediction_order: str,
                                score_index: int, title_prefix: str) -> None:
    _setup_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda t: str(t[1]))
    xs = list(range(len(rows)))
    scaled_scores = []
    display_thresholds = []
    pred_cols = []
    for row in rows:
        rec = row[0]
        sc = float(row[score_index])
        display_score, display_threshold = _display_score_threshold(sc, threshold, rec)
        scaled_scores.append(display_score)
        display_thresholds.append(display_threshold)
        if display_threshold is None:
            pred_cols.append("#9e9e9e")
        elif display_score >= display_threshold:
            pred_cols.append("#d62728")
        else:
            pred_cols.append("#1f77b4")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.26 * len(rows) + 2), 5))
    yvals_raw = [0.0 if not np.isfinite(v) else float(v) for v in scaled_scores]
    yvals, threshold_lines, compressed_display = _prediction_plot_transform_values(
        yvals_raw, display_thresholds)
    ax.bar(xs, yvals, color=pred_cols)
    finite_vals = [v for v in yvals if np.isfinite(v)]
    drawn_thresholds = _draw_display_thresholds(ax, xs, threshold_lines)
    ymax = max([1.0] + finite_vals + drawn_thresholds) if finite_vals or drawn_thresholds else 1.0
    ax.set_xticks(xs)
    ax.set_xticklabels([t[1] for t in rows], rotation=90, fontsize=6)
    order_label = "wafer 이름순"
    ax.set_ylabel("anomaly score")
    ax.set_xlabel(f"wafer ({order_label})")
    ax.set_title(f"{title_prefix} ({order_label})", fontsize=12)
    ax.set_ylim(0, _PREDICTION_PLOT_COMPRESS_TOP
                if compressed_display else ymax * 1.12 + 1e-9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_wafer_scores(val_labeled, scores, threshold, csv_path, png_path,
                       prediction_order: str = "score",
                       train_score_stats=None, ratio_png_path=None) -> None:
    """웨이퍼별 anomaly score → CSV + 정렬 막대그래프(빨강 NG / 파랑 OK + threshold 선)."""
    import csv as _csv
    rows = []
    for r, lab in val_labeled:
        raw_score = float(scores.get(r["wafer_id"], 0.0))
        display_score, display_threshold = _display_score_threshold(raw_score, threshold, r)
        rows.append((r, r["wafer_id"], lab, raw_score, display_score, display_threshold))
    rows = _sort_prediction_rows(rows, prediction_order, score_index=4)
    wafer_mean_score = _mean_display_score(rows, score_index=4)
    vals = [t[4] for t in rows]
    display_thresholds = [row[5] for row in rows]
    vals_display, threshold_plot_values, compressed_display = (
        _prediction_plot_transform_values(vals, display_thresholds)
    )

    eval_pairs = []
    for rec, _wid, lab, sc, _display_score, _display_threshold in rows:
        th = _threshold_for_record(threshold, rec)
        if th is None:
            continue
        eval_pairs.append((int(lab), int(sc >= th)))
    if eval_pairs:
        tp = sum(1 for y, p in eval_pairs if y == 1 and p == 1)
        fp = sum(1 for y, p in eval_pairs if y == 0 and p == 1)
        fn = sum(1 for y, p in eval_pairs if y == 1 and p == 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metric_cols = (round(precision, 6), round(recall, 6), round(f1_score, 6))
    else:
        metric_cols = ("", "", "")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["wafer_id", "recipe", "label", "anomaly_score",
                    "anomaly_score_display",
                    "threshold", "pred", "correct",
                    "precision", "recall", "f1_score",
                    "threshold_ratio_percent",
                    "wafer_mean_ratio_percent",
                    "train_ok_sigma"])
        for (rec, wid, lab, sc, display_score, display_threshold), display_plot_score in zip(rows, vals_display):
            th = _threshold_for_record(threshold, rec)
            if th is not None:
                pred = int(sc >= th)
                w.writerow([wid, _recipe_key(rec), "NG" if lab == 1 else "OK",
                            round(float(display_score), 5),
                            round(float(display_plot_score), 5),
                            "" if display_threshold is None else round(float(display_threshold), 5),
                            "NG" if pred else "OK", int(pred == lab),
                            *metric_cols,
                            _ratio_percent(display_score, display_threshold),
                            _ratio_percent(display_score, wafer_mean_score),
                            _sigma_level(display_score, rec, train_score_stats)])
            else:
                w.writerow([wid, _recipe_key(rec), "NG" if lab == 1 else "OK",
                            round(float(display_score), 5),
                            round(float(display_plot_score), 5),
                            "", "", "",
                            *metric_cols, "", _ratio_percent(display_score, wafer_mean_score),
                            _sigma_level(display_score, rec, train_score_stats)])
    _setup_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    xs = list(range(len(rows)))
    vals_plot = vals_display
    cols = ["#d62728" if t[2] == 1 else "#1f77b4" for t in rows]
    fig, ax = plt.subplots(figsize=(max(7, 0.26 * len(rows) + 2), 5))
    ax.bar(xs, vals_plot, color=cols)
    recipe_spans = []
    if str(prediction_order).lower() == "recipe" and rows:
        prev = _recipe_key(rows[0][0])
        group_start = 0
        ymax = max(vals_plot) if vals_plot else 1.0
        for i, row in enumerate(rows + [(None, "", 0, 0.0, 0.0, None)]):
            recipe = _recipe_key(row[0]) if row[0] is not None else None
            if recipe != prev:
                recipe_spans.append((prev, group_start, i - 1))
                ax.axvline(i - 0.5, color="#999999", lw=0.8, alpha=0.6)
                ax.text((group_start + i - 1) / 2, ymax * 1.01, prev,
                        ha="center", va="bottom", fontsize=8, color="#444444",
                        clip_on=False)
                group_start = i
                prev = recipe
    if threshold is not None:
        if str(prediction_order).lower() == "recipe" and recipe_spans:
            for _recipe, start, end in recipe_spans:
                display_threshold = rows[start][5]
                if display_threshold is None:
                    continue
                ax.hlines(float(threshold_plot_values[start]),
                          start - 0.45, end + 0.45,
                          color="#2ca02c", linestyles="--", lw=1.2, alpha=0.8)
        else:
            if any(row[5] is not None for row in rows):
                _draw_display_thresholds(ax, xs, threshold_plot_values)
    ax.set_xticks(xs); ax.set_xticklabels([t[1] for t in rows], rotation=90, fontsize=6)
    order_label = "recipe별 묶음 + recipe 내 점수 내림차순" \
        if str(prediction_order).lower() == "recipe" else "점수 내림차순"
    ax.set_ylabel("anomaly score"); ax.set_xlabel(f"wafer ({order_label})")
    ax.set_title(f"웨이퍼별 anomaly score ({order_label}, 빨강=NG · 파랑=OK)", fontsize=12)
    h = [Patch(color="#d62728", label="NG(불량)"), Patch(color="#1f77b4", label="OK(양품)")]
    if threshold is not None:
        h.append(plt.Line2D([], [], color="#2ca02c", ls="--",
                            label="threshold"))
    ax.legend(handles=h, fontsize=8)
    if compressed_display:
        ax.set_ylim(0, _PREDICTION_PLOT_COMPRESS_TOP)
    fig.tight_layout(); fig.savefig(png_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    ratio_path = (
        Path(ratio_png_path)
        if ratio_png_path is not None
        else Path(png_path).with_name(f"{Path(png_path).stem}_threshold_ratio.png")
    )
    _write_threshold_ratio_plot(
        rows, threshold, ratio_path, prediction_order, score_index=3,
        title_prefix="웨이퍼별 anomaly score")


def write_blind_predictions(records, scores, threshold, csv_path, png_path,
                            summary_path=None, prediction_order: str = "score",
                            train_score_stats=None, ratio_png_path=None) -> None:
    """라벨 없는 blind wafer별 anomaly score/prediction → CSV + 그래프."""
    import csv as _csv
    import json
    rows = []
    for r in records:
        raw_score = float(scores.get(r["wafer_id"], 0.0))
        display_score, display_threshold = _display_score_threshold(raw_score, threshold, r)
        rows.append((r, r["wafer_id"], raw_score, display_score, display_threshold))
    rows = _sort_prediction_rows(rows, prediction_order, score_index=3)
    wafer_mean_score = _mean_display_score(rows, score_index=3)
    vals = [t[3] for t in rows]
    display_thresholds = [row[4] for row in rows]
    vals_display, threshold_plot_values, compressed_display = (
        _prediction_plot_transform_values(vals, display_thresholds)
    )

    pred_counts = {"NG": 0, "OK": 0, "UNKNOWN": 0}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["wafer_id", "recipe", "label", "anomaly_score",
                    "anomaly_score_display",
                    "threshold", "pred", "correct",
                    "precision", "recall", "f1_score",
                    "threshold_ratio_percent",
                    "wafer_mean_ratio_percent",
                    "train_ok_sigma"])
        for (rec, wid, sc, display_score, display_threshold), display_plot_score in zip(rows, vals_display):
            th = _threshold_for_record(threshold, rec)
            if th is None:
                pred = ""
                pred_counts["UNKNOWN"] += 1
                th_out = ""
            else:
                pred = "NG" if sc >= float(th) else "OK"
                pred_counts[pred] += 1
                th_out = "" if display_threshold is None else round(float(display_threshold), 5)
            w.writerow([wid, _recipe_key(rec), "BLIND", round(float(display_score), 5),
                        round(float(display_plot_score), 5),
                        th_out, pred, "", "", "", "",
                        _ratio_percent(display_score, display_threshold),
                        _ratio_percent(display_score, wafer_mean_score),
                        _sigma_level(display_score, rec, train_score_stats)])

    if summary_path is not None and rows:
        doc = {
            "mode": "blind_prediction",
            "n_wafers": len(rows),
            "pred_counts": pred_counts,
            "threshold": (
                _display_threshold_json_value(threshold)
                if any(row[4] is not None for row in rows) else None
            ),
            "recipes": sorted({_recipe_key(r) for r, _, _, _, _ in rows}),
        }
        Path(summary_path).write_text(
            json.dumps(doc, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8")
    elif summary_path is not None:
        _remove_file_if_exists(summary_path)

    _setup_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    xs = list(range(len(rows)))
    vals_plot = vals_display
    pred_by_row = []
    for rec, _wid, sc, _display_score, _display_threshold in rows:
        th = _threshold_for_record(threshold, rec)
        pred_by_row.append(None if th is None else int(sc >= float(th)))
    cols = ["#d62728" if p == 1 else "#1f77b4" if p == 0 else "#9e9e9e"
            for p in pred_by_row]
    fig, ax = plt.subplots(figsize=(max(7, 0.26 * len(rows) + 2), 5))
    ax.bar(xs, vals_plot, color=cols)
    recipe_spans = []
    if str(prediction_order).lower() == "recipe" and rows:
        prev = _recipe_key(rows[0][0])
        group_start = 0
        ymax = max(vals_plot) if vals_plot else 1.0
        for i, row in enumerate(rows + [(None, "", 0.0, 0.0, None)]):
            recipe = _recipe_key(row[0]) if row[0] is not None else None
            if recipe != prev:
                recipe_spans.append((prev, group_start, i - 1))
                ax.axvline(i - 0.5, color="#999999", lw=0.8, alpha=0.6)
                ax.text((group_start + i - 1) / 2, ymax * 1.01, prev,
                        ha="center", va="bottom", fontsize=8, color="#444444",
                        clip_on=False)
                group_start = i
                prev = recipe
    if threshold is not None:
        if str(prediction_order).lower() == "recipe" and recipe_spans:
            for _recipe, start, end in recipe_spans:
                display_threshold = rows[start][4]
                if display_threshold is not None:
                    ax.hlines(float(threshold_plot_values[start]),
                              start - 0.45, end + 0.45,
                              color="#2ca02c", linestyles="--", lw=1.2, alpha=0.8)
        else:
            if any(row[4] is not None for row in rows):
                _draw_display_thresholds(ax, xs, threshold_plot_values)
    ax.set_xticks(xs)
    ax.set_xticklabels([t[1] for t in rows], rotation=90, fontsize=6)
    order_label = "recipe별 묶음 + recipe 내 점수 내림차순" \
        if str(prediction_order).lower() == "recipe" else "점수 내림차순"
    ax.set_ylabel("anomaly score")
    ax.set_xlabel(f"blind wafer ({order_label})")
    ax.set_title(f"Blind wafer prediction ({order_label}, 빨강=예측 NG · 파랑=예측 OK)",
                 fontsize=12)
    handles = [Patch(color="#d62728", label="pred NG"),
               Patch(color="#1f77b4", label="pred OK"),
               Patch(color="#9e9e9e", label="threshold 없음")]
    if threshold is not None:
        handles.append(plt.Line2D([], [], color="#2ca02c", ls="--",
                                  label="threshold"))
    ax.legend(handles=handles, fontsize=8)
    if compressed_display:
        ax.set_ylim(0, _PREDICTION_PLOT_COMPRESS_TOP)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    ratio_path = (
        Path(ratio_png_path)
        if ratio_png_path is not None
        else Path(png_path).with_name(f"{Path(png_path).stem}_threshold_ratio.png")
    )
    _write_threshold_ratio_plot(
        rows, threshold, ratio_path, prediction_order, score_index=2,
        title_prefix="Blind wafer prediction")


def write_train_ok_predictions(records, scores, threshold, csv_path, png_path,
                               prediction_order: str = "score",
                               train_score_stats=None) -> None:
    """Train OK wafer별 anomaly score → 참고용 CSV + 그래프."""
    import csv as _csv
    rows = []
    for r in records:
        raw_score = float(scores.get(r["wafer_id"], 0.0))
        display_score, display_threshold = _display_score_threshold(raw_score, threshold, r)
        rows.append((r, r["wafer_id"], raw_score, display_score, display_threshold))
    rows = _sort_prediction_rows(rows, prediction_order, score_index=3)
    wafer_mean_score = _mean_display_score(rows, score_index=3)
    vals = [t[3] for t in rows]
    display_thresholds = [row[4] for row in rows]
    vals_display, threshold_plot_values, compressed_display = (
        _prediction_plot_transform_values(vals, display_thresholds)
    )

    pred_counts = {"NG": 0, "OK": 0, "UNKNOWN": 0}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["wafer_id", "recipe", "label", "anomaly_score",
                    "anomaly_score_display",
                    "threshold", "pred", "over_threshold",
                    "threshold_ratio_percent",
                    "wafer_mean_ratio_percent",
                    "train_ok_sigma"])
        for (rec, wid, sc, display_score, display_threshold), display_plot_score in zip(rows, vals_display):
            th = _threshold_for_record(threshold, rec)
            if th is None:
                pred = ""
                over_threshold = ""
                pred_counts["UNKNOWN"] += 1
                th_out = ""
            else:
                over_threshold = int(sc >= float(th))
                pred = "NG" if over_threshold else "OK"
                pred_counts[pred] += 1
                th_out = "" if display_threshold is None else round(float(display_threshold), 5)
            w.writerow([wid, _recipe_key(rec), "TRAIN_OK",
                        round(float(display_score), 5),
                        round(float(display_plot_score), 5),
                        th_out, pred, over_threshold,
                        _ratio_percent(display_score, display_threshold),
                        _ratio_percent(display_score, wafer_mean_score),
                        _sigma_level(display_score, rec, train_score_stats)])

    import matplotlib
    _setup_font()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    xs = list(range(len(rows)))
    vals_plot = vals_display
    pred_by_row = []
    for rec, _wid, sc, _display_score, _display_threshold in rows:
        th = _threshold_for_record(threshold, rec)
        pred_by_row.append(None if th is None else int(sc >= float(th)))
    cols = ["#d62728" if p == 1 else "#1f77b4" if p == 0 else "#9e9e9e"
            for p in pred_by_row]
    fig, ax = plt.subplots(figsize=(max(7, 0.26 * len(rows) + 2), 5))
    ax.bar(xs, vals_plot, color=cols)
    recipe_spans = []
    if str(prediction_order).lower() == "recipe" and rows:
        prev = _recipe_key(rows[0][0])
        group_start = 0
        ymax = max(vals_plot) if vals_plot else 1.0
        for i, row in enumerate(rows + [(None, "", 0.0, 0.0, None)]):
            recipe = _recipe_key(row[0]) if row[0] is not None else None
            if recipe != prev:
                recipe_spans.append((prev, group_start, i - 1))
                ax.axvline(i - 0.5, color="#999999", lw=0.8, alpha=0.6)
                ax.text((group_start + i - 1) / 2, ymax * 1.01, prev,
                        ha="center", va="bottom", fontsize=8, color="#444444",
                        clip_on=False)
                group_start = i
                prev = recipe
    if threshold is not None:
        if str(prediction_order).lower() == "recipe" and recipe_spans:
            for _recipe, start, end in recipe_spans:
                display_threshold = rows[start][4]
                if display_threshold is not None:
                    ax.hlines(float(threshold_plot_values[start]),
                              start - 0.45, end + 0.45,
                              color="#2ca02c", linestyles="--", lw=1.2, alpha=0.8)
        else:
            if any(row[4] is not None for row in rows):
                _draw_display_thresholds(ax, xs, threshold_plot_values)
    ax.set_xticks(xs)
    ax.set_xticklabels([t[1] for t in rows], rotation=90, fontsize=6)
    order_label = "recipe별 묶음 + recipe 내 점수 내림차순" \
        if str(prediction_order).lower() == "recipe" else "점수 내림차순"
    ax.set_ylabel("anomaly score")
    ax.set_xlabel(f"train OK wafer ({order_label})")
    ax.set_title(f"Train OK wafer anomaly score ({order_label}, 빨강=threshold 이상)",
                 fontsize=12)
    handles = [
        Patch(color="#d62728", label=f"over threshold ({pred_counts['NG']})"),
        Patch(color="#1f77b4", label=f"below threshold ({pred_counts['OK']})"),
    ]
    if pred_counts["UNKNOWN"]:
        handles.append(Patch(color="#9e9e9e", label=f"unknown ({pred_counts['UNKNOWN']})"))
    if threshold is not None:
        handles.append(plt.Line2D([], [], color="#2ca02c", ls="--", label="threshold"))
    ax.legend(handles=handles, fontsize=8)
    if compressed_display:
        ax.set_ylim(0, _PREDICTION_PLOT_COMPRESS_TOP)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _train_prediction_artifact_paths(artifacts_dir: Path) -> tuple[Path, Path, Path]:
    base = Path(artifacts_dir)
    return (
        base / "train_prediction.csv",
        base / "train_prediction.png",
        base / "train_prediction_meta.json",
    )


def _train_prediction_meta_payload(A, records, threshold, score_kind,
                                   prediction_order: str) -> dict:
    records = list(records or [])
    return {
        "version": 6,
        "sensors": [str(x) for x in A.sensor_cols],
        "n_steps_global": int(getattr(A, "n_steps_global", 0) or 0),
        "arch": getattr(A, "arch", {}),
        "score_kind": "" if score_kind is None else str(score_kind),
        "score_topk": _score_topk_from_cfg(getattr(A, "cfg", None)),
        "score_agg_enabled": _score_agg_enabled_from_cfg(getattr(A, "cfg", None)),
        "score_resid_agg": _score_resid_agg_from_cfg(getattr(A, "cfg", None)),
        "score_wafer_agg": _score_wafer_agg_from_cfg(getattr(A, "cfg", None)),
        "score_time_avg": _score_time_avg_from_cfg(getattr(A, "cfg", None)),
        "score_final_avg": _score_final_avg_from_cfg(getattr(A, "cfg", None)),
        "threshold_mode": _threshold_mode(threshold) if threshold is not None else "",
        "threshold": threshold,
        "prediction_order": str(prediction_order or "score"),
        "n_train_ok": len(records),
        "train_ok_sha1": _ok_records_digest(records),
    }


def _train_prediction_meta_matches(meta: dict | None, expected: dict,
                                   require_records_match: bool) -> bool:
    if not isinstance(meta, dict):
        return False
    for key in ("version", "sensors", "n_steps_global", "arch",
                "score_kind", "score_topk", "score_agg_enabled",
                "score_resid_agg", "score_wafer_agg", "score_time_avg",
                "score_final_avg", "threshold_mode",
                "prediction_order"):
        if not _json_equiv(meta.get(key), expected.get(key)):
            return False
    if not _json_equiv(meta.get("threshold"), expected.get("threshold")):
        return False
    if require_records_match:
        for key in ("n_train_ok", "train_ok_sha1"):
            if not _json_equiv(meta.get(key), expected.get(key)):
                return False
    return True


def _read_json_file(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _copy_prediction_csv_with_display_column(src_csv: Path, dst_csv: Path) -> None:
    import csv as _csv
    import shutil
    try:
        with Path(src_csv).open(newline="", encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if not fieldnames or "anomaly_score" not in fieldnames:
            shutil.copy2(src_csv, dst_csv)
            return
        if "anomaly_score_capped" in fieldnames:
            fieldnames.remove("anomaly_score_capped")
            for row in rows:
                row.pop("anomaly_score_capped", None)
        if "anomaly_score_display" not in fieldnames:
            idx = fieldnames.index("anomaly_score") + 1
            fieldnames.insert(idx, "anomaly_score_display")
        values = [row.get("anomaly_score") for row in rows]
        thresholds = [
            row.get("threshold") if str(row.get("threshold", "")).strip() else None
            for row in rows
        ]
        display_values, _threshold_values, _compressed = (
            _prediction_plot_transform_values(values, thresholds)
        )
        for row, display_value in zip(rows, display_values):
            row["anomaly_score_display"] = _csv_float(display_value)
        with Path(dst_csv).open("w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception:
        shutil.copy2(src_csv, dst_csv)


def _copy_train_prediction_artifacts(src_csv: Path, src_png: Path,
                                     dst_csv: Path, dst_png: Path) -> None:
    import shutil
    Path(dst_csv).parent.mkdir(parents=True, exist_ok=True)
    _copy_prediction_csv_with_display_column(src_csv, dst_csv)
    shutil.copy2(src_png, dst_png)


def ensure_train_ok_prediction_outputs(A, records, threshold, score_kind,
                                       artifacts_dir: Path, report_dir: Path,
                                       prediction_order: str = "score",
                                       precomputed_scores=None,
                                       train_score_stats=None) -> bool:
    records = list(records or [])
    art_csv, art_png, art_meta = _train_prediction_artifact_paths(artifacts_dir)
    out_csv = Path(report_dir) / "10_예측값_학습셋_원본.csv"
    out_png = Path(report_dir) / "08_예측값_학습셋.png"
    artifacts_ready = art_csv.exists() and art_png.exists()
    expected_meta = _train_prediction_meta_payload(
        A, records, threshold, score_kind, prediction_order)

    if artifacts_ready:
        meta = _read_json_file(art_meta)
        if (meta is not None
                and _train_prediction_meta_matches(
                    meta, expected_meta, require_records_match=bool(records))):
            _copy_train_prediction_artifacts(art_csv, art_png, out_csv, out_png)
            print(f"[train OK anomaly score] artifacts 재사용 → "
                  f"{out_csv} · {out_png}")
            return True
        if not records:
            _copy_train_prediction_artifacts(art_csv, art_png, out_csv, out_png)
            print(f"[train OK anomaly score] artifacts metadata 불일치, "
                  f"train OK를 로드할 수 없어 기존 파일 복사 → {out_csv} · {out_png}")
            return True

    if not records:
        print("  · train OK prediction 생략: artifacts 없음/불완전, train OK 로드 불가")
        return False

    try:
        if precomputed_scores and all(r.get("wafer_id") in precomputed_scores
                                      for r in records):
            train_score_by_id = {
                r["wafer_id"]: float(precomputed_scores[r["wafer_id"]])
                for r in records
            }
        else:
            p_train = _prediction_scores(A, records, threshold, score_kind)
            train_score_by_id = {
                r["wafer_id"]: float(v) for r, v in zip(records, p_train)
            }
        if train_score_stats is None:
            train_score_stats = _train_ok_score_stats_from_scores(
                records, train_score_by_id, threshold)
        art_csv.parent.mkdir(parents=True, exist_ok=True)
        write_train_ok_predictions(
            records, train_score_by_id, threshold, art_csv, art_png,
            prediction_order=prediction_order,
            train_score_stats=train_score_stats)
        art_meta.write_text(
            json.dumps(expected_meta, ensure_ascii=False, indent=2,
                       default=_json_default),
            encoding="utf-8")
        _copy_train_prediction_artifacts(art_csv, art_png, out_csv, out_png)
        print(f"[train OK anomaly score] {len(records)}장 계산/저장 → "
              f"{art_csv} · {art_png} / report 복사 → {out_csv} · {out_png}")
        return True
    except Exception as e:
        if artifacts_ready:
            try:
                _copy_train_prediction_artifacts(art_csv, art_png, out_csv, out_png)
                print(f"  ⚠ train OK prediction 재계산 실패({e}); "
                      f"기존 artifacts 복사 → {out_csv} · {out_png}")
                return True
            except Exception:
                pass
        print(f"  ⚠ train OK prediction 생성 실패: {e}")
        return False


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _export_onnx_forecast_for_report(cfg: IOConfig, report_dir: Path) -> None:
    """Export the trained TCN checkpoint to ONNX artifacts inside report_dir."""
    pt = Path(cfg.artifacts_dir) / "tcn_forecast.pt"
    try:
        from .export_onnx_forecast import export, verify
        onnx_path = export(pt_path=pt, out_dir=report_dir,
                           art_dir=cfg.artifacts_dir)
    except SystemExit as e:
        print(f"  ⚠ ONNX 변환 생략: {e}")
        return
    except Exception as e:
        print(f"  ⚠ ONNX 변환 실패: {e}")
        return

    try:
        import importlib.util as _ilu
        if _ilu.find_spec("onnxruntime") is None:
            print("  ⚠ ONNX 검증 생략: onnxruntime 없음")
        else:
            delta = verify(pt_path=pt, out_dir=report_dir,
                           art_dir=cfg.artifacts_dir)
            if float(delta) >= 1e-4:
                print(f"  ⚠ ONNX 검증 차이 큼: {delta:.3e}")
    except Exception as e:
        print(f"  ⚠ ONNX 검증 실패: {e}")
    print(f"[ONNX export] → {onnx_path}")


def cmd_recon_explain(cfg: IOConfig, top_k=None, ng_only: bool = False,
                      report_dir: Path = None, prediction_order: str = "score",
                      ok_mean_exclude_over_threshold: bool = True) -> None:
    import json
    from dataclasses import replace
    adir = Path(cfg.artifacts_dir)
    apath = adir / "tcn_forecast.pt"
    if not apath.exists():
        sys.exit(f"저장된 TCN forecast 모델 없음: {apath} — 먼저 tcn-search 하세요.")

    # ★ search_best.json: best params + 저장된 wafer별 점수(재사용 → search 와 동일).
    thr, best_params, best_scores, best_score_kind = None, None, None, None
    saved_score_topk = None
    saved_score_payloads = ()
    cli_score_agg_enabled = _score_agg_enabled_from_cfg(cfg)
    cli_resid_agg = getattr(cfg, "resid_agg", None)
    cli_wafer_agg = getattr(cfg, "wafer_agg", None)
    cli_score_time_avg = _score_time_avg_from_cfg(cfg)
    cli_score_final_avg = _score_final_avg_from_cfg(cfg)
    cli_score_override = (
        top_k is not None
        or cli_score_agg_enabled
        or cli_score_time_avg
        or cli_score_final_avg
    )
    cli_threshold_recipe_multipliers = tuple(
        getattr(cfg, "threshold_recipe_multipliers", ()) or ())
    sb = next((p for p in (adir / "search_best.json", adir / "best_search.json")
               if p.exists()), None)
    if sb is not None:
        try:
            b = json.loads(sb.read_text(encoding="utf-8")).get("best", {})
            best_params = b.get("params")
            best_scores = b.get("scores") or b.get("anomaly_scores")
            best_score_kind = (
                b.get("score_name")
                or (b.get("evaluation_criteria") or {}).get("threshold_field")
                or (b.get("val") or {}).get("score_kind")
            )
            thr = b.get("threshold", b.get("val", {}).get("threshold_raw"))
            if best_score_kind is None and isinstance(thr, dict):
                best_score_kind = thr.get("score_kind")
            saved_score_topk = _score_topk_from_saved_metadata(
                thr, best_score_kind, b.get("val"), b.get("evaluation_criteria"), b)
            saved_score_payloads = (b.get("val"), b.get("evaluation_criteria"), b)
            stored_threshold_wise = (b.get("val") or {}).get("threshold_wise")
            threshold_wise_from_threshold = (
                str(thr.get("threshold_wise", "")).lower()
                if isinstance(thr, dict) else ""
            )
            if best_score_kind is None and (
                    str(stored_threshold_wise or "").lower() in {
                        "all", "recipe", "sensor", "step"}
                    or threshold_wise_from_threshold in {
                        "all", "recipe", "sensor", "step"}):
                stored_mode = str(stored_threshold_wise or threshold_wise_from_threshold).lower()
                best_score_kind = (
                    "step_accum_excess_reconstruction_error"
                    if stored_mode in {"sensor", "step"}
                    else "sum_all_sensor_reconstruction_error"
                )
            if stored_threshold_wise:
                cfg.threshold_wise = str(stored_threshold_wise)
            if isinstance(thr, dict) and thr.get("threshold_wise"):
                cfg.threshold_wise = str(thr.get("threshold_wise"))
            if best_params:
                fields = set(IOConfig().__dataclass_fields__)
                cfg = replace(cfg, **{k: v for k, v in best_params.items() if k in fields})
                if cli_score_agg_enabled:
                    cfg.score_agg_enabled = True
                    cfg.resid_agg = cli_resid_agg
                    cfg.wafer_agg = cli_wafer_agg
                if cli_score_time_avg:
                    cfg.score_time_avg = True
                if cli_score_final_avg:
                    cfg.score_final_avg = True
        except Exception as e:
            print(f"  ⚠ search_best 적용 실패(기본값으로 채점): {e}")
    else:
        print("  ⚠ search_best.json 없음: threshold/저장 score 없이 모델로 재채점")

    if top_k is not None:
        cfg.wafer_topk = max(0, int(top_k or 0))
    elif saved_score_topk is not None:
        cfg.wafer_topk = int(saved_score_topk)
    if not cli_score_time_avg and not cli_score_final_avg and (
            not _score_agg_enabled_from_cfg(cfg)
            or not _score_time_avg_from_cfg(cfg)
            or not _score_final_avg_from_cfg(cfg)):
        _apply_score_agg_metadata_to_cfg(cfg, thr, *saved_score_payloads)

    A = TCNForecastDetector.load(adir, cfg).A
    run_threshold_for_scores = (
        _threshold_with_score_metadata(thr, getattr(A, "cfg", cfg))
        if cli_score_override and _uses_sum_reconstruction_threshold(thr, best_score_kind)
        else thr
    )
    run_threshold_for_scores, explain_threshold_multipliers = (
        _threshold_with_explain_recipe_multipliers(
            run_threshold_for_scores, cli_threshold_recipe_multipliers)
    )
    run_score_kind = (
        _score_kind_from_cfg(getattr(A, "cfg", cfg))
        if cli_score_override and _uses_sum_reconstruction_threshold(thr, best_score_kind)
        else best_score_kind
    )
    current_score_meta = _score_metadata_for_cfg(getattr(A, "cfg", cfg))
    saved_score_meta = _score_metadata_from_saved(
        thr, best_score_kind, *saved_score_payloads)
    score_topk = _score_topk_from_cfg(getattr(A, "cfg", cfg))
    top_k = score_topk
    try:
        pred_batch = int(getattr(getattr(A, "cfg", cfg), "pred_batch", 1) or 1)
    except Exception:
        pred_batch = 1
    print(f"  · prediction window batch: {pred_batch if pred_batch > 1 else 'model tcn_batch'}")
    print(f"  · reconstruction score topk sensors: {score_topk or 'all'}")
    print("  · reconstruction score time-step average: "
          f"{'ON' if _score_time_avg_from_cfg(getattr(A, 'cfg', cfg)) else 'OFF'}")
    print("  · reconstruction score final-length average: "
          f"{'ON' if _score_final_avg_from_cfg(getattr(A, 'cfg', cfg)) else 'OFF'}")
    if _score_agg_enabled_from_cfg(getattr(A, "cfg", cfg)):
        print(f"  · reconstruction score aggregation: "
              f"resid_agg={_score_resid_agg_from_cfg(getattr(A, 'cfg', cfg))} "
              f"wafer_agg={_score_wafer_agg_from_cfg(getattr(A, 'cfg', cfg))}")
    else:
        print("  · reconstruction score aggregation: current default "
              "(time sum -> selected sensor sum)")
    if explain_threshold_multipliers:
        pairs = ", ".join(
            f"{recipe}={factor:g}"
            for recipe, factor in sorted(explain_threshold_multipliers.items())
        )
        print("  · explain threshold recipe multiplier 적용: "
              f"{pairs} (저장 score 재사용 가능, 판정 threshold는 explain 기준)")
    ref_ok_records = None
    if getattr(A, "resid_sigma", None) is None:
        try:
            ref_ok_records, _, _, _ = load_dataset(cfg.ok_dir, cfg)
            if ref_ok_records and hasattr(A, "_fit_error_limits"):
                print("  · 기존 모델에 residual std가 없어 train OK 기준으로 설명용 std 재계산")
                if _fit_missing_sigma_preserve_score_limits(A, ref_ok_records):
                    print("  · 설명용 residual std만 보강; anomaly score용 OK 기준 오차는 checkpoint 값 유지")
            else:
                print("  ⚠ 설명용 residual std 없음: 기존 residual 기준으로 4σ band를 대체")
        except Exception as e:
            print(f"  ⚠ 설명용 residual std 재계산 실패: {e}")
    if ref_ok_records is None:
        try:
            ref_ok_records, _, _, _ = load_dataset(cfg.ok_dir, cfg)
        except Exception as e:
            ref_ok_records = []
            print(f"  ⚠ train OK 로드 실패: OK 누적 점수 band 생략: {e}")
    try:
        ok_band_cache = load_or_build_ok_accum_bands(
            A, ref_ok_records or [], Path(cfg.artifacts_dir) / "ok_accum_bands.npz",
            threshold=run_threshold_for_scores, score_kind=run_score_kind,
            exclude_ok_over_threshold=ok_mean_exclude_over_threshold)
    except Exception as e:
        ok_band_cache = None
        print(f"  ⚠ OK 누적 점수 band 생성 실패: {e}")

    blind_mode = False
    vok, _, _, _ = load_dataset(_sub_ci(cfg.val_dir, "ok"), cfg)
    vng, _, _, _ = load_dataset(_sub_ci(cfg.val_dir, "ng"), cfg)
    targets = [(r, 1) for r in vng] + ([] if ng_only else [(r, 0) for r in vok])
    if not targets:
        tg, _, _, _ = load_dataset(cfg.ng_dir, cfg)
        targets = [(r, 1) for r in tg]
    if not targets:
        blind_base = Path(cfg.ok_dir).parent
        blind, _, _, _ = load_dataset(blind_base, cfg)
        targets = [(r, None) for r in blind]
        blind_mode = bool(targets)
    if not targets:
        sys.exit("대상 wafer 없음: val/ok·val/ng, data/ng, 또는 --data-name blind CSV 필요.")

    out = Path(report_dir) if report_dir is not None else Path(cfg.artifacts_dir).parent / "reports" / "recon"
    score_dir = out / "01_이상점수_그래프"
    heatmap_dir = out / "02_히트맵"
    rca_dir = out / "03_RCA경로"
    summary_dir = out / "04_웨이퍼요약"
    step_dir = out / "05_스텝요약"
    score_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    rca_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    pred_png = out / "06_예측값.png"
    pred_name_png = out / "07_예측값_이름순.png"
    pred_csv = out / "09_예측값_원본.csv"
    blind_summary_json = out / "15_블라인드_판정요약.json"
    if not blind_mode:
        _remove_file_if_exists(blind_summary_json)
    _setup_font()

    # ── 성능표 ──
    val_labeled = [(r, 0) for r in vok] + [(r, 1) for r in vng]
    run_threshold = run_threshold_for_scores
    score_by_id = {}
    train_score_by_id = {}
    train_score_stats = None
    # 서치 저장 점수가 검증셋 전체를 덮으면 그대로 재사용 → search 결과와 100% 일치
    use_scores = None
    if val_labeled and best_scores:
        val_ids = {r["wafer_id"] for r, _ in val_labeled}
        score_ids = set(best_scores)
        missing_ids = sorted(val_ids - score_ids)
        if not missing_ids:
            if cli_score_override and not _score_metadata_matches(
                    saved_score_meta, current_score_meta):
                print("  · search 저장 점수 미적용: 현재 score 옵션과 저장 score metadata 불일치 "
                      f"(saved: {_format_score_metadata(saved_score_meta)} / "
                      f"current: {_format_score_metadata(current_score_meta)}) → 모델로 재채점")
            else:
                use_scores = best_scores
                if explain_threshold_multipliers:
                    print("  · search 저장 점수 재사용(재채점 안 함) → "
                          "score는 동일, recipe multiplier 적용 threshold로 판정 재계산")
                else:
                    print("  · search 저장 점수 재사용(재채점 안 함) → search 결과와 동일 보장")
        else:
            sample = ", ".join(missing_ids[:5])
            more = "" if len(missing_ids) <= 5 else f" 외 {len(missing_ids) - 5}개"
            print(f"  · search 저장 점수 미적용: 검증 wafer {len(missing_ids)}개 점수 없음 "
                  f"({sample}{more}) → 모델로 재채점")
    elif val_labeled:
        print("  · search 저장 점수 미적용(검증셋 불일치) → 모델로 재채점")
    else:
        print("  · 라벨 검증셋 없음: blind wafer를 모델로 prediction")
    if val_labeled and len({lab for _, lab in val_labeled}) == 2:
        try:
            summ = perf_summary(A, val_labeled, cfg, threshold=run_threshold,
                                params=best_params, scores=use_scores,
                                score_kind=run_score_kind)
            plot_perf_table(summ, cfg, summary_dir / "_성능요약표.png")
            raw_score_map = dict(summ.get("raw_scores") or summ.get("scores") or {})
            (summary_dir / "_성능요약.json").write_text(
                json.dumps(_display_summary_json(summ, best_params),
                           ensure_ascii=False, indent=2,
                           default=_json_default), encoding="utf-8")
            v = summ["vc"]
            print(f"[성능표] 정답 {v['correct']}/{v['n']} · 불량 {v['tp']}/{v['n_ng']} · "
                  f"AUROC {v['auroc']} · thr {'1.0000' if summ['threshold'] is not None else '—'} · "
                  f"OK복원MSE {summ['val_mse']:.4f} → {summary_dir/'_성능요약표.png'}")
            run_threshold = summ["threshold"]
            score_by_id = raw_score_map
            train_score_by_id, train_score_stats = _build_train_ok_score_reference(
                A, ref_ok_records or [], run_threshold, run_score_kind)
            write_wafer_scores(val_labeled, raw_score_map, summ["threshold"],
                               pred_csv, pred_png,
                               prediction_order=prediction_order,
                               train_score_stats=train_score_stats,
                               ratio_png_path=pred_name_png)
            print(f"[웨이퍼별 anomaly score] → {pred_csv} · {pred_png}")
        except Exception as e:
            print(f"  ⚠ 성능표/웨이퍼점수 생성 실패: {e}")
    elif targets:
        recs = [r for r, _ in targets]
        try:
            p = _prediction_scores(A, recs, run_threshold, run_score_kind)
            score_by_id = {r["wafer_id"]: float(v) for r, v in zip(recs, p)}
            train_score_by_id, train_score_stats = _build_train_ok_score_reference(
                A, ref_ok_records or [], run_threshold, run_score_kind)
            if blind_mode:
                write_blind_predictions(
                    recs, score_by_id, run_threshold,
                    pred_csv, pred_png,
                    summary_path=blind_summary_json,
                    prediction_order=prediction_order,
                    train_score_stats=train_score_stats,
                    ratio_png_path=pred_name_png)
                print(f"[blind prediction] {len(recs)}장 → "
                      f"{pred_csv} · {pred_png}")
            else:
                write_wafer_scores(targets, score_by_id, run_threshold,
                                   pred_csv, pred_png,
                                   prediction_order=prediction_order,
                                   train_score_stats=train_score_stats,
                                   ratio_png_path=pred_name_png)
                print(f"[웨이퍼별 anomaly score] → {pred_csv} · {pred_png}")
        except Exception as e:
            print(f"  ⚠ prediction 생성 실패: {e}")

    ensure_train_ok_prediction_outputs(
        A, ref_ok_records or [], run_threshold, run_score_kind,
        Path(cfg.artifacts_dir), out, prediction_order=prediction_order,
        precomputed_scores=train_score_by_id,
        train_score_stats=train_score_stats)

    sig = float(getattr(A.cfg, "raw_error_level", 1.0))
    interp_factor = _explain_interp_factor(A.cfg)
    step_accum_threshold = (
        run_threshold if _uses_step_accum_threshold(run_threshold, run_score_kind) else None
    )
    score_mode_label = (
        "sum_all_sensor_reconstruction_error"
        if _uses_sum_reconstruction_threshold(run_threshold, run_score_kind)
        else "step_accum_excess_reconstruction_error"
        if _uses_step_accum_threshold(run_threshold, run_score_kind)
        else "model.score_records"
    )
    print(f"  · prediction score mode: {score_mode_label}"
          f"{'' if run_score_kind is None else f' (score_kind={run_score_kind})'}")
    if run_threshold is not None and step_accum_threshold is None:
        print("  · saved threshold는 wafer score 기준이므로 sensor/step 누적 threshold 시각화는 생략")
    print(f"[recon-explain] TCN 모델 {apath} → {len(targets)} wafer "
          f"(합본 + 센서별 고해상도, interpolation x{interp_factor}) → {out}")
    narr = []
    all_sensor_rows = []
    all_step_rows = []
    all_causal_rows = []
    all_threshold_rows = []
    topk_selection_rows = []
    for i, (r, lab) in enumerate(targets, 1):
        wid = r["wafer_id"]
        try:
            actual_raw, expected_raw, z_raw = _compute(A, r)  # wafer 1회 계산 후 재사용
            threshold_rows = build_threshold_exceedances(
                A, r, actual_raw, expected_raw, z_raw, step_accum_threshold, label=lab)
            all_threshold_rows.extend(threshold_rows)
            r_view, actual, expected, z = _with_interpolated_explain_arrays(
                A, r, actual_raw, expected_raw, z_raw)
            score_sensor_indices = _topk_sensor_indices_from_level(
                z_raw, score_topk, resid_agg=_score_resid_agg_from_cfg(getattr(A, "cfg", None)))
            score_sensor_set = {int(x) for x in score_sensor_indices.tolist()}
            score_wafer_agg = _score_wafer_agg_from_cfg(getattr(A, "cfg", None))
            score_resid_agg = _score_resid_agg_from_cfg(getattr(A, "cfg", None))
            if int(score_topk or 0) > 0:
                topk_selection_rows.extend(build_topk_sensor_selection_rows(
                    A, r, lab, z_raw, score_sensor_indices, score_topk,
                    score_resid_agg, score_wafer_agg))
            score_sensor_count = max(1, int(score_sensor_indices.size))
            sensor_contribution_scale = (
                1.0 / float(score_sensor_count)
                if score_wafer_agg in {"topk_mean", "mean"}
                else 1.0
            )
            order = _sensor_order(z, sig)
            sel = _flow_sensor_order(A, order)
            finding_top_n = min(len(order), max(20, int(top_k or 0)))
            sensor_rows = build_sensor_findings(
                A, r_view, actual, expected, z, order, top_n=finding_top_n)
            step_rows = build_step_summary(A, r_view, actual, expected, z)
            causal_rows = build_causal_path_rows(A, r_view, sensor_rows)
            all_sensor_rows.extend(sensor_rows)
            all_step_rows.extend(step_rows)
            all_causal_rows.extend(causal_rows)
            score = float(score_by_id.get(wid, _score_from_level(A, z_raw, r)))
            plot_wafer_summary(A, r_view, lab, actual, expected, z,
                               sensor_rows, step_rows, score, run_threshold,
                               summary_dir / f"{wid}.png",
                               causal_rows=causal_rows)
            plot_step_summary(step_rows, step_dir / f"{wid}.png")
            plot_causal_path(A, r_view, causal_rows, rca_dir / f"{wid}.png")
            # 합본(전 센서 한 장)
            plot_recon_overlay(A, r_view, score_dir / f"{wid}.png",
                               top_k=top_k, pre=(actual, expected, z),
                               threshold=run_threshold)
            # ★ 센서별 개별 고해상도 (per-wafer 폴더)
            sdir = score_dir / wid; sdir.mkdir(parents=True, exist_ok=True)
            plot_wafer_accum_score(A, r_view, z, sdir / "00_웨이퍼점수.png",
                                   ok_bands=ok_band_cache,
                                   threshold=run_threshold,
                                   final_score=score,
                                   raw_rec=r, raw_z=z_raw,
                                   sensor_indices=score_sensor_indices,
                                   csv_path=sdir / "00_웨이퍼점수_원본.csv")
            for rank, j in enumerate(sel, 1):
                sensor_plot_threshold = (
                    step_accum_threshold
                    if step_accum_threshold is not None
                    else run_threshold
                    if run_threshold is not None and _threshold_mode(run_threshold) == "all"
                    else None
                )
                plot_recon_sensor(A, r_view, j, actual, expected, z,
                                  sdir / f"{rank:02d}_{A.sensor_cols[j]}.png",
                                  threshold=sensor_plot_threshold, ok_bands=ok_band_cache,
                                  raw_rec=r, raw_z=z_raw,
                                  include_accum=(int(j) in score_sensor_set),
                                  sensor_contribution_scale=sensor_contribution_scale)
            plot_recon_heatmap(A, r_view, heatmap_dir / f"{wid}.png", z_pre=z)
            n = narrate_recon(A, r_view, top_k=3, pre=(actual, expected, z))
            n["label"] = "BLIND" if lab is None else lab
            narr.append(n)
            tag = "BLIND" if lab is None else ("NG" if lab == 1 else "OK")
            print(f"  [{i:>3}/{len(targets)}] [{tag}] {wid} (센서 {len(sel)}장) — "
                  f"{n['narrative'][:80]}")
        except Exception as e:
            print(f"  [{i:>3}/{len(targets)}] {wid} 실패: {e}")
    (summary_dir / "_웨이퍼_설명요약.json").write_text(
        json.dumps(narr, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[report rows] "
          f"sensor_findings={len(all_sensor_rows)} · "
          f"step_summary={len(all_step_rows)} · "
          f"rca_paths={len(all_causal_rows)} · "
          f"threshold_exceedances={len(all_threshold_rows)} · "
          f"topk_sensor_selection={len(topk_selection_rows)}")
    _safe_write_report_output(
        "센서이상부분", write_sensor_findings, all_sensor_rows,
        out / "11_센서이상부분.csv", out / "12_센서이상부분.md")
    _safe_write_report_output(
        "스텝요약", write_step_summary, all_step_rows,
        step_dir / "_스텝_점수.csv")
    _safe_write_report_output(
        "RCA경로", write_causal_paths, all_causal_rows,
        rca_dir / "_RCA경로.csv", rca_dir / "_RCA경로.md")
    _safe_write_report_output(
        "임계값 초과", write_threshold_exceedances, all_threshold_rows,
        out / "13_임계값_초과.csv")
    _safe_write_report_output(
        "임계값 초과 스텝별", write_threshold_exceedances_by_step,
        all_threshold_rows, targets, out / "14_임계값_초과_스텝별.csv")
    _safe_write_report_output(
        "Topk 센서", write_topk_sensor_selection, topk_selection_rows,
        out / "16_Topk_센서.csv", out / "17_Topk_센서.json")
    _export_onnx_forecast_for_report(cfg, out)
    topk_files_msg = (
        " · 16_Topk_센서.csv/json"
        if topk_selection_rows else ""
    )
    print(f"완료: {out} (01_이상점수_그래프/<wafer>.png 합본 · "
          f"01_이상점수_그래프/<wafer>/ 센서별 · 02_히트맵/ · "
          f"03_RCA경로/ · 04_웨이퍼요약/ · 05_스텝요약/ · "
          f"09_예측값_원본.csv{topk_files_msg})")
