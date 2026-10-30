"""IO-Recon-TCN 설명력 자료 — 예상 vs 실제 + 잔차 히트맵 + IO 인과 내러티브.

논리(요구 #6):
  "IO 가 이전 단계 대비 *이만큼 변하라*(기대 Δ)고 명령했는데, 실제 센서는 안 변했다
   (실제 Δ≈0). 그래서 불량이다."  → IO 명령 대비 무응답을 수치·그림으로 주장.

산출(요구 #7):
  · <wafer>_overlay.png  — 상위 이탈 센서별: 실제(빨강) vs 모델 기대(파랑).
  · overlay/<wafer>/00_wafer_accum_score.png — 전체 센서 합산 누적 이상치 점수.
  · <wafer>_heatmap.png  — raw error level 스텝×센서 히트맵.
  · recon_explain.json   — wafer 별 한 줄 인과 내러티브 + 수치.

저장된 TCN forecast best 모델(artifacts/tcn_forecast.pt)을 불러와
val/ok·val/ng 에 대해 생성.

실행:
  python run.py recon-explain                         # val/ok·val/ng → reports/recon/
  python run.py recon-explain --ng-only               # 불량만
  python run.py recon-explain --topk 8                # 패널 센서 수
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
# TCNForecastDetector(=torch)는 CLI(cmd_recon_explain)에서만 사용 → 지연 import(앱 torch-free)
from .sensor_graph import graph_adjust_abs_z, graph_sensor_rank, is_pseudo_node, load_sensor_graph


def _setup_font() -> None:
    """Use a Korean-capable font when available; keep plotting usable on headless Linux."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    import matplotlib.font_manager as fm

    from pathlib import Path as _P                        # 번들 폰트 등록(exe/헤드리스 한글)
    _cand = _P(__file__).resolve().parent / "assets" / "NanumGothic.ttf"
    try:
        if _cand.exists():
            fm.fontManager.addfont(str(_cand))
    except Exception:                                    # noqa: BLE001
        pass
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
    for key, value in table.items():                     # 대소문자 무시 정확 일치
        if _recipe_lookup_key(key) == wanted:
            return value
    # ★ 접두 매칭: 테이블 키(짧은 '29a')가 레시피 풀네임('29aece38…')의 접두인 경우.
    #   서버 _recipe_threshold 와 동일 규칙 → 플롯 결정선이 실제 판정 임계와 일치(불일치 버그 수정).
    best_v, best_len = None, -1
    for key, value in table.items():
        kl = _recipe_lookup_key(key)
        if kl and wanted.startswith(kl) and len(kl) > best_len:
            best_v, best_len = value, len(kl)
    return best_v


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
        for key in ("threshold_by_recipe", "threshold_raw", "threshold"):
            value = _lookup_recipe_value(threshold.get(key), recipe)
            if value is not None:
                return float(value)
        for key in ("default", "global", "global_threshold_raw", "global_threshold"):
            value = threshold.get(key)
            if value is not None and not isinstance(value, dict):
                return float(value)
        for key in ("threshold_raw", "threshold"):
            value = threshold.get(key)
            if value is not None and not isinstance(value, dict):
                return float(value)
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
        return "; ".join(f"{k}:{float(v):.4f}" for k, v in sorted(table.items()))
    return f"{float(threshold):.4f}"


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


def _score_norm_denominator(A, n_valid) -> float:
    """타임포인트 정규화 모델의 누적 분모.

    ★ 왜 필요한가 — --score-time-avg/--score-final-avg 로 학습한 모델은 OK 밴드
      (ok_accum_bands.npz)가 '누적 ÷ 유효 타임포인트 수' 스케일로 저장된다
      (train_jw _score_average_curve). 앱이 raw 누적을 그 밴드 상한으로 나누면
      비율이 타임포인트 수(수백)배로 부풀어 센서 점수가 전부 2.0 이 됐다.
      웨이퍼 점수는 백엔드(_score_record)가 이미 나누고 있으므로, 센서 점수와
      누적 곡선도 같은 분모로 맞춘다. 일반 모델(score_norm=none)은 1.
    """
    if str(getattr(A, "score_norm", "none") or "none").lower() in ("time", "final"):
        return float(max(1, int(n_valid)))
    return 1.0


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
    for rec in records:
        level = A.error_level(rec) if hasattr(A, "error_level") else np.abs(A.z_residual(rec))
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
    return float(np.maximum(0.0, level).sum())


def _sum_all_sensor_reconstruction_scores(A, records) -> np.ndarray:
    scores = []
    for rec in records:
        level = A.error_level(rec) if hasattr(A, "error_level") else np.abs(A.z_residual(rec))
        scores.append(_sum_all_sensor_reconstruction_score_from_level(level))
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
        return "sum_all_sensor_reconstruction_error" in kind
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
        return _sum_all_sensor_reconstruction_scores(A, records)
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


def _score_from_level(A, level: np.ndarray) -> float:
    """Mirror TCNForecastModel._score_record for already computed error_level."""
    return _sum_all_sensor_reconstruction_score_from_level(level)


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


def _close_mask_gaps(mask, max_gap: int):
    """마스크의 짧은 틈(<max_gap)을 메워 인접 구간을 하나로 병합."""
    mask = np.asarray(mask, dtype=bool)
    if max_gap <= 0 or not mask.any():
        return mask
    out = mask.copy()
    n = mask.size
    i = 0
    while i < n:
        if mask[i]:
            i += 1; continue
        j = i
        while j < n and not mask[j]:
            j += 1
        if 0 < i and j < n and (j - i) < max_gap:   # 양쪽이 True 인 짧은 틈
            out[i:j] = True
        i = j
    return out


def _slope_highlight_events(curve, min_width: int = 10, k: float = 2.5):
    """누적 이상치 곡선의 '기울기가 크게 변하는(급상승)' 구간을 이벤트로 반환.

    누적 곡선은 비감소이므로 기울기(증가율)가 곧 그 시점의 이상 기여도다.
    임계는 (중앙값+k·MAD, 상위 90% 분위, 최대기울기의 40%) 중 큰 값 → 정말로
    가파른 구간만 남긴다. 짧은 틈은 메워 하나의 구간으로 병합.
    이벤트의 peak_score = 그 구간의 최대 기울기(진할수록 큼).
    """
    curve = np.nan_to_num(np.asarray(curve, dtype=np.float64))
    n = int(curve.size)
    if n < 3:
        return [], 0.0
    slope = np.gradient(curve)
    slope = np.clip(slope, 0.0, None)          # 누적=비감소 → 양의 기울기만
    pos = slope[slope > 1e-12]
    if pos.size == 0:
        return [], 0.0
    med = float(np.median(pos))
    # 가장 가파른 상위 구간(상위 15%)만, 단 중앙값 대비 뚜렷할 때만.
    #   정상(거의 일정한 기울기) 곡선은 상위분위≈중앙값 → 게이트에 걸려 거의 안 뜬다.
    q = float(np.quantile(pos, 0.85))
    thr = max(q, 1.30 * med)
    mask = slope >= thr
    if not mask.any():
        return [], float(thr)
    mask = _close_mask_gaps(mask, max_gap=max(2 * min_width, 20))
    _, events = _expanded_highlight_mask(mask, slope, min_width)
    return events, float(thr)


def _event_peak_indices(events, top_k: int):
    ranked = sorted(events, key=lambda e: float(e.get("peak_score", 0.0)), reverse=True)
    return [int(e["peak"]) for e in ranked[:max(0, int(top_k))]]


def _event_band_alpha(event, threshold: float,
                      base: float = 0.10, max_alpha: float = 0.42) -> float:
    peak = float(event.get("peak_score", 0.0))
    over = max(0.0, peak - float(threshold))
    return float(min(max_alpha, base + 0.06 * over))


def _draw_sigma_event_bands(ax, events, threshold: float, color="#d62728",
                            label="4σ 초과 구간(진할수록 σ 큼)",
                            base: float = 0.10, max_alpha: float = 0.42,
                            relative_alpha: bool = False) -> None:
    """Draw full-height x-axis bands; darker means larger excess over threshold.

    relative_alpha=True 이면 이벤트 집합 내 최대 peak 대비 상대값으로 진하기를 정한다
    (기울기처럼 절대값이 작은 지표에 적합).
    """
    peak_max = max((float(ev.get("peak_score", 0.0)) for ev in events), default=0.0)
    span = max(1e-12, peak_max - float(threshold))
    labeled = False
    for ev in events:
        start = float(ev.get("display_start", ev.get("start", 0))) - 0.5
        end = float(ev.get("display_end", ev.get("end", 0))) - 0.5
        if end <= start:
            continue
        if relative_alpha:
            frac = min(1.0, max(0.0, (float(ev.get("peak_score", 0.0)) - float(threshold)) / span))
            alpha = base + (max_alpha - base) * frac
        else:
            alpha = _event_band_alpha(ev, threshold, base=base, max_alpha=max_alpha)
        ax.axvspan(
            start, end,
            color=color,
            alpha=alpha,
            lw=0,
            zorder=1,
            label=(label if not labeled else None),
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


def _ok_band_signature(A, ok_records, grid: int,
                       threshold=None, score_kind=None,
                       exclude_ok_over_threshold: bool = False) -> str:
    recipes = sorted({_recipe_key(r) for r in ok_records} | {"__global__"})
    payload = {
        "version": 14,
        "sensors": [str(x) for x in A.sensor_cols],
        "n_steps_global": int(getattr(A, "n_steps_global", 0) or 0),
        "grid": int(grid),
        "n_ok": len(ok_records),
        "recipes": recipes,
        "resid_abs_limit_sha1": _array_digest(getattr(A, "resid_abs_limit", [])),
        "arch": getattr(A, "arch", {}),
        "score_curve": "sum_of_per_sensor_cumulative_reconstruction_error",
        "score_band": "same_recipe_train_ok_mean_plus_minus_2sigma",
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
    required = {"sensor_mean", "sensor_step_mean", "total_mean", "total_std"}
    if not payload or not required.issubset(set(getattr(data, "files", []))):
        return False
    if int(payload.get("version", 0) or 0) < 13:
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
    """npz(NpzFile) 또는 이미 로드된 dict 어느 쪽이든 OK 밴드 dict 로 변환.

    일부 파생 키(sensor_step_mean/total_std)나 min/max 가 없는 구버전 npz 도
    가능한 키만 실어서 최대한 살린다(밴드/평균이 있으면 표시됨).
    """
    keys = set(getattr(data, "files", None) or data.keys())
    if not {"recipes", "sensor_cols", "grid"}.issubset(keys):
        raise KeyError("recipes/sensor_cols/grid")
    out = {
        "recipes": [str(x) for x in np.asarray(data["recipes"]).tolist()],
        "sensor_cols": [str(x) for x in np.asarray(data["sensor_cols"]).tolist()],
        "grid": int(np.asarray(data["grid"]).item()),
    }
    for key in ("sensor_mean", "sensor_step_mean", "total_mean", "total_std",
                "sensor_min", "sensor_max", "total_min", "total_max"):
        if key in keys:
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


def _wafer_decision_accum_series(level, step) -> np.ndarray:
    """Wafer accumulated score: sum of per-sensor normalized reconstruction error."""
    level = np.asarray(level, dtype=np.float64)
    if level.ndim != 2 or level.size == 0:
        return np.zeros(level.shape[0] if level.ndim else 0, dtype=np.float64)
    curves = [
        _sensor_reconstruction_accum_series_from_level(level[:, j], step)
        for j in range(level.shape[1])
    ]
    if not curves:
        return np.zeros(level.shape[0], dtype=np.float64)
    return np.sum(np.vstack(curves), axis=0)


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


def _smooth_curve(y, frac: float = 0.04, min_win: int = 5) -> np.ndarray:
    """NaN-안전 이동평균 스무딩(가장자리 보정). OK 평균 곡선 표현용.

    OK 평균은 여러 정상 웨이퍼 누적곡선의 앙상블 평균이라 본래 매끈하지만,
    그리드→시간축 투영에서 생기는 미세한 꺾임을 부드럽게 다듬는다.
    """
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    if n < max(3, int(min_win)):
        return y
    win = max(int(min_win), int(round(n * float(frac))))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win < 3:
        return y
    finite = np.isfinite(y)
    yy = np.where(finite, y, 0.0)
    kernel = np.ones(win, dtype=np.float64) / win
    num = np.convolve(yy, kernel, mode="same")
    den = np.convolve(finite.astype(np.float64), kernel, mode="same")
    out = np.where(den > 1e-9, num / den, np.nan)
    out[~finite] = np.nan          # 원래 NaN 위치는 유지
    return out


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
    recipes = sorted({_recipe_key(r) for r in ok_records})
    recipes = ["__global__"] + [r for r in recipes if r != "__global__"]
    ridx = {r: i for i, r in enumerate(recipes)}
    shape_s = (len(recipes), n_sensor, n_steps, grid)
    shape_t = (len(recipes), n_steps, grid)
    s_min = np.full(shape_s, np.inf, dtype=np.float64)
    s_max = np.full(shape_s, -np.inf, dtype=np.float64)
    t_min = np.full(shape_t, np.inf, dtype=np.float64)
    t_max = np.full(shape_t, -np.inf, dtype=np.float64)
    s_sum = np.zeros(shape_s, dtype=np.float64)
    s_count = np.zeros(shape_s, dtype=np.int32)
    s_step_sum = np.zeros(shape_s, dtype=np.float64)
    s_step_count = np.zeros(shape_s, dtype=np.int32)
    t_sum = np.zeros(shape_t, dtype=np.float64)
    t_sumsq = np.zeros(shape_t, dtype=np.float64)
    t_count = np.zeros(shape_t, dtype=np.int32)

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
            bin_arr = _step_fraction_bins(step_arr, spans, grid)
            valid = (step_arr > 0) & (step_arr < n_steps)
            if not valid.any():
                continue
            flat = step_arr[valid] * grid + bin_arr[valid]
            sensor_accum = np.vstack([
                _sensor_reconstruction_accum_series_from_level(z[:, j], step_arr)
                for j in range(n_sensor)
            ])
            sensor_step_accum = np.vstack([
                _step_accum_series_from_level(z[:, j], step_arr, threshold_base=1.0)
                for j in range(n_sensor)
            ])
            total_curve = np.sum(sensor_accum, axis=0)
            recipe_ids = [0]
            recipe = _recipe_key(rec)
            if recipe in ridx and ridx[recipe] != 0:
                recipe_ids.append(ridx[recipe])
            for ri in recipe_ids:
                t_flat_min = t_min[ri].reshape(-1)
                t_flat_max = t_max[ri].reshape(-1)
                vals_t = total_curve[valid]
                np.minimum.at(t_flat_min, flat, vals_t)
                np.maximum.at(t_flat_max, flat, vals_t)
                _add_binned_curve_mean(t_sum[ri].reshape(-1),
                                       t_count[ri].reshape(-1),
                                       flat, vals_t,
                                       sumsq_flat=t_sumsq[ri].reshape(-1))
                for j in range(n_sensor):
                    s_flat_min = s_min[ri, j].reshape(-1)
                    s_flat_max = s_max[ri, j].reshape(-1)
                    vals_s = sensor_accum[j, valid]
                    np.minimum.at(s_flat_min, flat, vals_s)
                    np.maximum.at(s_flat_max, flat, vals_s)
                    _add_binned_curve_mean(s_sum[ri, j].reshape(-1),
                                           s_count[ri, j].reshape(-1),
                                           flat, vals_s)
                    _add_binned_curve_mean(s_step_sum[ri, j].reshape(-1),
                                           s_step_count[ri, j].reshape(-1),
                                           flat, sensor_step_accum[j, valid])
        except Exception as e:
            print(f"  ⚠ OK 누적 band 계산 skip: {rec.get('wafer_id', idx)}: {e}")

    for arr, fill in ((s_min, np.nan), (s_max, np.nan), (t_min, np.nan), (t_max, np.nan)):
        arr[~np.isfinite(arr)] = fill
    sensor_mean = _mean_from_sum_count(s_sum, s_count)
    sensor_step_mean = _mean_from_sum_count(s_step_sum, s_step_count)
    total_mean = _mean_from_sum_count(t_sum, t_count)
    total_std = _std_from_sum_sumsq_count(t_sum, t_sumsq, t_count)
    return {
        "recipes": recipes,
        "sensor_cols": [str(x) for x in A.sensor_cols],
        "grid": grid,
        "sensor_min": s_min,
        "sensor_max": s_max,
        "total_min": t_min,
        "total_max": t_max,
        "sensor_mean": sensor_mean,
        "sensor_step_mean": sensor_step_mean,
        "total_mean": total_mean,
        "total_std": total_std,
    }


def load_or_build_ok_accum_bands(A, ok_records, cache_path: Path, grid: int = 100,
                                 threshold=None, score_kind=None,
                                 exclude_ok_over_threshold: bool = False) -> dict | None:
    cache_path = Path(cache_path)
    ok_records = list(ok_records or [])
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
            if not ok_records and _ok_cache_compatible_without_records(
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
        total_min=bands["total_min"],
        total_max=bands["total_max"],
        sensor_mean=bands["sensor_mean"],
        sensor_step_mean=bands["sensor_step_mean"],
        total_mean=bands["total_mean"],
        total_std=bands["total_std"],
    )
    print(f"  · OK 누적 점수 band 캐시 저장: {cache_path}")
    return bands


_OK_FALLBACK_WARNED: set = set()                          # 레시피별 1회만 경고(로그 스팸 방지)


def _ok_band_recipe_index(ok_bands: dict | None, recipe: str) -> int | None:
    if not ok_bands:
        return None
    recipes = ok_bands.get("recipes") or []
    recipe = str(recipe or "unknown")
    if recipe in recipes:
        return recipes.index(recipe)
    if "__global__" in recipes:
        # 레시피 전용 OK 밴드가 없어 전체(global) 혼합 밴드로 폴백 → 정상범위/OK평균이
        # '이 레시피' 기준이 아님. 조용히 오도하지 않도록 레시피당 1회 경고를 남긴다.
        if recipe not in _OK_FALLBACK_WARNED:
            _OK_FALLBACK_WARNED.add(recipe)
            print(f"[ok-band] recipe '{recipe}' 전용 정상범위 없음 → 전체(global) 밴드로 대체 "
                  f"(정상범위/OK평균은 레시피 기준 아님)")
        return recipes.index("__global__")
    return None


def _lookup_ok_band(ok_bands: dict | None, rec, step_plot, spans,
                    sensor_idx: int | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not ok_bands or step_plot is None or not len(step_plot):
        return None, None
    ri = _ok_band_recipe_index(ok_bands, _recipe_key(rec))
    if ri is None:
        return None, None
    grid = int(ok_bands.get("grid", 100))
    step_arr = np.asarray(step_plot, dtype=int)
    bin_arr = _step_fraction_bins(step_arr, spans, grid)
    if sensor_idx is None:
        lo_all, hi_all = ok_bands.get("total_min"), ok_bands.get("total_max")
        if lo_all is None or hi_all is None:
            return None, None
        lo_src, hi_src = lo_all[ri], hi_all[ri]
    else:
        lo_all, hi_all = ok_bands.get("sensor_min"), ok_bands.get("sensor_max")
        if lo_all is None or hi_all is None:
            return None, None
        lo_src = lo_all[ri, int(sensor_idx)]
        hi_src = hi_all[ri, int(sensor_idx)]
    lo = np.full(len(step_arr), np.nan, dtype=np.float64)
    hi = np.full(len(step_arr), np.nan, dtype=np.float64)
    valid = (step_arr >= 0) & (step_arr < lo_src.shape[0])
    idx = np.flatnonzero(valid)
    if idx.size:
        lo[idx] = lo_src[step_arr[idx], bin_arr[idx]]
        hi[idx] = hi_src[step_arr[idx], bin_arr[idx]]
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
    ri = _ok_band_recipe_index(ok_bands, _recipe_key(rec))   # __global__ 폴백 포함
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
    out = _project_ok_grid_to_wafer_axis(src, step_plot, spans,
                                         monotonic=not step_reset)
    # 누적(비-스텝리셋) OK 평균만 스무딩 표현. 스텝리셋(톱니)은 경계 보존 위해 원본.
    if out is not None and not step_reset:
        out = _smooth_curve(out)
    return out


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
                                 current, ok_mean, ok_sigma,
                                 threshold_value, scale_value=None,
                                 ok_lo=None, ok_hi=None) -> None:
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
    ok_lo = None if ok_lo is None else np.asarray(ok_lo, dtype=np.float64)
    ok_hi = None if ok_hi is None else np.asarray(ok_hi, dtype=np.float64)
    threshold = np.nan if threshold_value is None else float(threshold_value)
    scale = np.nan if scale_value is None else float(scale_value)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = threshold if np.isfinite(threshold) and threshold > 1e-12 else np.nan
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = _csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(current)):
            mean = np.nan if ok_mean is None or i >= len(ok_mean) else float(ok_mean[i])
            lo = np.nan if ok_lo is None or i >= len(ok_lo) else float(ok_lo[i])
            hi = np.nan if ok_hi is None or i >= len(ok_hi) else float(ok_hi[i])
            cur_out = current[i] / scale if np.isfinite(scale) else current[i]
            mean_out = mean / scale if np.isfinite(mean) and np.isfinite(scale) else mean
            lo_out = lo / scale if np.isfinite(lo) and np.isfinite(scale) else lo
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
                "ok_min_anomaly_score": _csv_float(lo_out),
                "ok_min_threshold_ratio": _csv_float(lo_out),
                "ok_max_anomaly_score": _csv_float(hi_out),
                "ok_max_threshold_ratio": _csv_float(hi_out),
                "threshold": _csv_float(th_out),
                "threshold_ratio": _csv_float(th_out),
            })


def plot_recon_sensor_overlay(A, rec, j, actual, expected, z, out_path,
                              ok_bands=None) -> None:
    """센서 1개 '실제 vs 기대' 단일 패널(누적 오차 없음). 요구사항 1/2 용."""
    import matplotlib
    matplotlib.use("Agg")
    _setup_font()
    import matplotlib.pyplot as plt
    t_full = actual.shape[0]
    plot_keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter(A, rec, t_full)
    if orig_idx.size == 0:
        return
    cur, exp = actual[plot_keep, j], expected[plot_keep, j]
    ok_limit = _limit_series(A, rec, j, t_full)[plot_keep]
    band_lo, band_hi = exp - ok_limit, exp + ok_limit
    ylo = float(min(cur.min(), exp.min(), band_lo.min()))
    yhi = float(max(cur.max(), exp.max(), band_hi.max()))
    pad = (yhi - ylo) * 0.12 + 1e-6
    fig, ax = plt.subplots(1, 1, figsize=(20, 5.0))
    ax.fill_between(t, band_lo, band_hi, color="#87CEEB", alpha=0.24,
                    zorder=0, label="OK 기준 정상 범위")
    _draw_step_guides(ax, bounds, spans, label=True)
    ax.plot(t, exp, color="#1f77b4", lw=1.6, ls="--", label="기대", zorder=4)
    ax.plot(t, cur, color="#d62728", lw=1.6, label="실제", zorder=5)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_title(f"{rec['wafer_id']} · {A.sensor_cols[j]} (실제 vs 기대)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def sensor_scores(A, rec, z, ok_bands=None, threshold=None, scale=None,
                  return_detail: bool = False):
    """센서별 이상점수. r = 그 센서 누적 이상치 ÷ 그 센서 정상(OK) 상한.

    score = min(r, 2.0)  — r 을 그대로 쓰고 2.0 에서만 자른다.
    · 1.0 = 그 센서의 OK 상한. 1.07 이면 상한의 1.07배로 언제나 같은 뜻이다.
    · 2.0 은 '상한의 2배 이상'이라는 뜻일 뿐, 더 큰 값은 ratios 로 확인한다.
    · return_detail=True 면 (scores, detail) 을 돌려준다. detail 은 센서마다
      {ratio, peak, dur} — 자르기 전 원값, 순간 최대 이탈 배수, 정상폭을 넘은 시간 비율.
      점수(누적÷상한)만으로는 '오래 조금 벗어남'과 '잠깐 크게 튐'이 구분되지 않아
      화면이 함께 보여 준다.
    · OK 상한이 없는(NaN) 센서는 '있는 센서 상한의 중앙값' 폴백, 그것도 없으면 0.
    반환: 센서 원래 순서 점수(0~2, 소수 3자리). 1.0 = 정상 상한 경계.
    """
    z = np.asarray(z, dtype=np.float64)
    S = int(z.shape[1]) if z.ndim == 2 else 0
    if S == 0:
        return []
    plot_keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter(A, rec, z.shape[0])
    if orig_idx.size == 0:
        return [0.0] * S
    zc = z[plot_keep]                                     # ProcessStep 0 등 제외(다른 곡선과 동일)
    _den = _score_norm_denominator(A, zc.shape[0])        # 타임포인트 정규화 모델 → 밴드와 동일 단위
    finals, refs = [], []
    for j in range(S):
        accum = np.asarray(_sensor_reconstruction_accum_series_from_level(zc[:, j], step_plot),
                           dtype=np.float64)
        finals.append((float(accum[-1]) / _den) if accum.size else 0.0)
        _lo, hi = _lookup_ok_band(ok_bands, rec, step_plot, spans, sensor_idx=j)
        ref = None
        if hi is not None:
            hi = np.asarray(hi, dtype=np.float64)
            fin = hi[np.isfinite(hi)]
            if fin.size:                                  # 상한 곡선의 '마지막 유효값'(=누적 최종 상한)
                ref = float(hi[-1]) if np.isfinite(hi[-1]) else float(fin[-1])
        refs.append(ref if (ref is not None and ref > 1e-9) else None)
    have = [r for r in refs if r is not None]
    ref_fb = float(np.median(have)) if have else None      # 상한 없는 센서 → 중앙값 폴백
    if ref_fb is None:
        # ★ OK 상한이 '하나도' 없는 모델 폴더(밴드 파일 누락 — 예: 다른 PC 에서
        #   복사해 온 모델) → 예전엔 전부 0.000 으로 깔려 "센서 점수가 다 0" 이
        #   됐다. 절대 비교는 불가능하지만 순서라도 보이도록, 그 웨이퍼 누적의
        #   중앙값을 기준으로 한 '상대 점수'로 폴백하고 로그를 남긴다.
        pos = [f for f in finals if f > 1e-9]
        if pos:
            ref_fb = float(np.median(pos))
            print("[sensor_scores] ⚠ OK 밴드 없음 — 웨이퍼 내 상대 점수로 폴백"
                  "(모델 폴더에 OK 밴드 파일을 넣으면 절대 점수로 복귀)", flush=True)
    ratios = []
    for j in range(S):
        ref = refs[j] if refs[j] is not None else ref_fb
        ratios.append(0.0 if (ref is None or ref <= 1e-9) else finals[j] / ref)
    # ★ 점수 = r 그대로, 2.0 에서만 자른다(사용자 결정).
    #
    #   예전에는 1 을 넘는 센서들을 그 웨이퍼 안에서 다시 정규화했다
    #   (가장 큰 r = 2.00, 나머지는 1~2 사이). 그래서 2.00 은 '두 배 나쁘다'가 아니라
    #   '이 웨이퍼에서 제일 나쁘다'는 뜻이었고, 웨이퍼가 바뀌면 같은 2.00 이 전혀 다른
    #   크기를 가리켰다. 1.0 인 센서와 2.0 인 센서가 그래프에서 비슷해 보이는 이유도
    #   이것이다 — 2.0 이 절대 크기가 아니었다.
    #
    #   이제 r 을 그대로 쓴다. 1.07 은 상한의 1.07배, 1.8 은 1.8배로 항상 같은 뜻이고
    #   웨이퍼끼리도 비교된다. 2.0 을 넘는 것만 2.0 으로 자른다(막대 길이 상한).
    #   잘린 센서의 실제 크기는 ratios 로 함께 돌려주어 화면이 원값을 보여 준다.
    scores = [round(min(r, 2.0), 3) for r in ratios]
    if return_detail:
        detail = []
        for j in range(S):
            col = zc[:, j] if zc.size else np.asarray([])
            fin = col[np.isfinite(col)] if col.size else col
            detail.append({
                "ratio": round(float(ratios[j]), 3),
                # 순간 최대 이탈 배수 — 이탈 배수 뷰의 세로 최댓값과 같은 값
                "peak": (round(float(np.max(fin)), 2) if fin.size else 0.0),
                # 정상폭(1배)을 넘은 시간 비율 — '얼마나 오래' 벗어났나
                "dur": (round(float(np.mean(fin > 1.0)), 4) if fin.size else 0.0),
            })
        return scores, detail
    return scores


def _step_marks(step_plot, stride, n):
    """ProcessStep 구간 시작점 목록 [{i, step}] (다운샘플 인덱스) — 차트 구분선·중앙 라벨용.

    첫 원소는 i=0(첫 구간의 스텝). i>0 인 원소 위치에 세로 구분선을 긋고,
    각 구간의 스텝 번호는 구간 중앙에 표시한다."""
    marks = []
    if step_plot is None or not len(step_plot):
        return marks
    sp = np.asarray(step_plot)
    n_ds = (n + stride - 1) // stride
    marks.append({"i": 0, "step": int(sp[0])})
    for b in (np.nonzero(np.diff(sp) != 0)[0] + 1):      # 값이 바뀌는 지점(원본 인덱스)
        di = int(b) // stride
        if 0 < di < n_ds:
            marks.append({"i": di, "step": int(sp[int(b)])})
    return marks


def sensor_series(A, rec, j, actual, expected, z):
    """센서 j 의 인터랙티브 차트용 시계열(overlay + 문제구간 슬라이더용). 데이터만 반환."""
    t_full = actual.shape[0]
    plot_keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter(A, rec, t_full)
    if orig_idx.size == 0:
        return None
    cur = actual[plot_keep, j].astype(float)
    exp = expected[plot_keep, j].astype(float)
    zc = z[plot_keep, j].astype(float)
    ok_limit = _limit_series(A, rec, j, t_full)[plot_keep].astype(float)
    diff = np.abs(cur - exp)                              # raw ↔ 기대 차이(슬라이더 기준)
    # 누적 곡선은 정적 per-sensor PNG(plot_recon_sensor)의 아래 패널과 '동일한' 정의를
    # 쓴다 → 두 화면의 '누적 이상치'가 일치. (기존 ad-hoc cumsum(clip(z-sig)) 은 PNG와 불일치)
    cum = np.asarray(_sensor_reconstruction_accum_series_from_level(zc, step_plot),
                     dtype=np.float64)
    n = len(cur)
    stride = max(1, n // 2000)                            # 과대 데이터 다운샘플(표시용)

    def r(a, nd=4):
        return [round(float(x), nd) for x in a[::stride]]
    return {
        "sensor": str(A.sensor_cols[j]),
        "t": [int(x) for x in t[::stride]],
        "actual": r(cur), "expected": r(exp),
        "band_lo": r(exp - ok_limit), "band_hi": r(exp + ok_limit),
        "diff": r(diff), "cum": r(cum),
        "diff_max": round(float(diff.max()) if diff.size else 0.0, 4),
        "steps": _step_marks(step_plot, stride, n),
        # ★ 스텝 배열(다운샘플 동일 간격) — 화면이 '문제 스텝' 요약을 만들 때 쓴다.
        #   steps(마크)는 경계 표시용이라 시점별 소속 스텝을 알 수 없어 따로 내려준다.
        "step_at": [int(x) for x in np.asarray(step_plot)[::stride]],
        "n": n,
    }


def _final_score_curve(curve, final_score):
    """Rescale display only; never modify the residual or the caller's array."""
    result = np.asarray(curve, dtype=np.float64).copy()
    if final_score is None or not result.size:
        return result
    target = float(final_score)
    if not np.isfinite(target) or target < 0:
        return result
    if np.isfinite(result[-1]) and result[-1] > 1e-12:
        result *= target / result[-1]
    else:
        # A zero-residual curve has no progression to preserve. Mark the endpoint only.
        result[-1] = target
    return result


def wafer_series(A, rec, z, ok_bands=None, threshold=None, raw_rec=None, raw_z=None, final_score=None):
    """'웨이퍼 복원오차 누적'(종합) 인터랙티브 차트용 시계열.

    plot_wafer_accum_score 와 '동일한' 곡선을 데이터로 반환한다(그림을 새로 그리지 않음):
      · band_lo/band_hi = 정상(OK) 범위(최소~최대)   · ok_mean = 같은 recipe OK 평균
      · cum            = 현재 wafer 누적 복원오차     · threshold = 판정선(보통 1.0)
    diff = 정상범위(band_hi) 초과분 → 문제구간 슬라이더 기준(초과가 없으면 0).
    """
    z = np.asarray(z, dtype=np.float64)
    plot_keep, orig_idx, t, step_plot, bounds, spans = _plot_time_filter(A, rec, z.shape[0])
    if step_plot is None or not len(step_plot) or not plot_keep.any():
        return None
    zc = z[plot_keep]
    if raw_z is not None and raw_rec is not None:
        raw_curve = _wafer_decision_accum_series(
            np.asarray(raw_z, dtype=np.float64),
            raw_rec.get("step") if isinstance(raw_rec, dict) else None)
        curve_full = _curve_to_view_full(rec, raw_curve)
        accum = curve_full[plot_keep] if len(curve_full) == z.shape[0] else None
    else:
        accum = None
    if accum is None:
        accum = _wafer_decision_accum_series(zc, step_plot)
    plot_vals_raw = (np.asarray(accum, dtype=np.float64)
                     / _score_norm_denominator(A, zc.shape[0]))   # 밴드와 동일 단위
    scale_value = _display_scale_threshold_for_record(threshold, rec)
    threshold_line = _display_threshold_for_record(threshold, rec)
    scale = (float(scale_value) if scale_value is not None
             and np.isfinite(float(scale_value)) and float(scale_value) > 1e-12 else None)
    ok_mean_plot = _lookup_ok_mean(ok_bands, rec, step_plot, spans, sensor_idx=None)
    ok_lo_plot, ok_hi_plot = _lookup_ok_band(ok_bands, rec, step_plot, spans, sensor_idx=None)

    def _sc(a):
        if a is None:
            return None
        a = np.asarray(a, dtype=np.float64)
        return a / scale if scale is not None else a
    cum = _sc(_final_score_curve(plot_vals_raw, final_score))
    ok_mean = _sc(ok_mean_plot)
    ok_lo = _sc(ok_lo_plot)
    ok_hi = _sc(ok_hi_plot)
    # 문제구간 = 정상범위(band_hi) 초과분. 밴드가 정의되지 않은(NaN) 지점은 '초과 판단 불가'
    # 이므로 diff=NaN 으로 두어 슬라이더 하이라이트에서 제외한다(0 으로 두면 그 지점이 곧
    # cum 전체가 되어 정상 구간을 '가장 이상함'으로 오인 하이라이트하는 버그가 생김).
    if ok_hi is not None and np.isfinite(ok_hi).any():
        finite = np.isfinite(ok_hi)
        diff = np.full(len(cum), np.nan, dtype=np.float64)
        diff[finite] = np.clip(cum[finite] - ok_hi[finite], 0, None)
    else:
        diff = np.concatenate([[0.0], np.clip(np.diff(cum), 0, None)]) if len(cum) else np.zeros(0)
    n = len(cum)
    stride = max(1, n // 2000)
    indices = np.unique(np.r_[np.arange(0, n, stride), n - 1]).astype(int)

    def r(a, nd=4):
        if a is None:
            return None
        return [(round(float(x), nd) if np.isfinite(x) else None) for x in a[indices]]
    return {
        "t": [int(x) for x in np.asarray(t)[indices]],
        "cum": r(cum), "ok_mean": r(ok_mean), "band_lo": r(ok_lo), "band_hi": r(ok_hi),
        "threshold": (round(float(threshold_line), 4) if threshold_line is not None else None),
        "diff": r(diff),
        "diff_max": round(float(np.nanmax(diff)) if np.isfinite(diff).any() and diff.size else 0.0, 4),
        "steps": _step_marks(step_plot, stride, n),
        # ★ 스텝 배열 — 화면의 '문제 스텝 요약'이 시점별 소속 스텝을 알아야 한다.
        "step_at": [int(x) for x in np.asarray(step_plot)[indices]],
        "n": n,
    }


def plot_recon_sensor(A, rec, j, actual, expected, z, out_path,
                      threshold=None, ok_bands=None,
                      raw_rec=None, raw_z=None) -> None:
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
    min_highlight_width = int(getattr(A.cfg, "highlight_min_width", 10))

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(20, 7.5),
                                   gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
    ok_limit = _limit_series(A, rec, j, t_full)[plot_keep]
    band_lo = exp - ok_limit
    band_hi = exp + ok_limit
    ylo = float(min(cur.min(), exp.min(), band_lo.min()))
    yhi = float(max(cur.max(), exp.max(), band_hi.max()))
    pad = (yhi - ylo) * 0.12 + 1e-6
    axT.fill_between(t, band_lo, band_hi, color="#87CEEB", alpha=0.24,
                     zorder=0, label="OK 기준 정상 범위")
    _draw_step_guides(axT, bounds, spans, label=True)
    axT.plot(t, exp, color="#1f77b4", lw=1.6, ls="--", label="기대", zorder=4)
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
    if raw_z is not None and raw_rec is not None:
        raw_step = raw_rec.get("step") if isinstance(raw_rec, dict) else None
        if use_step_threshold:
            raw_curve = _step_accum_series_from_level(
                np.asarray(raw_z, dtype=np.float64)[:, j], raw_step, threshold_base=1.0)
        else:
            raw_curve = _sensor_reconstruction_accum_series_from_level(
                np.asarray(raw_z, dtype=np.float64)[:, j], raw_step)
        curve_full = _curve_to_view_full(rec, raw_curve)
        accum = curve_full[plot_keep] if len(curve_full) == t_full else None
    else:
        accum = None
    if accum is None:
        if use_step_threshold:
            accum = _step_accum_series_from_level(zc, step_plot, threshold_base=1.0)
        else:
            accum = _sensor_reconstruction_accum_series_from_level(zc, step_plot)
    ok_mean = _lookup_ok_mean(
        ok_bands, rec, step_plot, spans, sensor_idx=j,
        step_reset=use_step_threshold)
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
    # ★ 하이라이트 = '누적 이상치 곡선의 기울기가 크게 변하는(급상승)' 구간.
    #   위(실제vs기대)·아래(누적) 두 그래프에 동일하게 표시.
    slope_events, slope_thr = _slope_highlight_events(accum, min_highlight_width)
    for _ax in (axT, axB):
        _draw_sigma_event_bands(_ax, slope_events, slope_thr, color="#d62728",
                                label="이상치 급상승 구간(기울기 급변)",
                                relative_alpha=True)
    if ok_mean is not None:
        axB.plot(t, ok_mean, color="#1f77b4", lw=1.25,
                 label="같은 recipe OK 평균", zorder=2)
    axB.plot(t, accum, color="#333333", lw=1.25,
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
    vals_for_ylim = [np.asarray(accum)[np.isfinite(accum)]]
    if ok_mean is not None:
        vals_for_ylim.append(ok_mean[np.isfinite(ok_mean)])
    if th_series is not None:
        vals_for_ylim.append(th_series[np.isfinite(th_series)])
    finite_vals = np.concatenate([v for v in vals_for_ylim if len(v)]) if vals_for_ylim else np.array([])
    ymax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    axB.set_ylim(0, ymax * 1.08 + 1e-9 if ymax > 0 else 1.0)
    axB.legend(fontsize=9, loc="upper left"); axB.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wafer_accum_score(A, rec, z, out_path, ok_bands=None,
                           threshold=None, final_score=None,
                           raw_rec=None, raw_z=None, apply_final_score=False) -> None:
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
        raw_curve = _wafer_decision_accum_series(
            np.asarray(raw_z, dtype=np.float64),
            raw_rec.get("step") if isinstance(raw_rec, dict) else None)
        curve_full = _curve_to_view_full(rec, raw_curve)
        accum = curve_full[plot_keep] if len(curve_full) == z.shape[0] else None
    else:
        accum = None
    if accum is None:
        accum = _wafer_decision_accum_series(zc, step_plot)
    plot_vals = (np.asarray(accum, dtype=np.float64)
                 / _score_norm_denominator(A, zc.shape[0]))       # 밴드와 동일 단위
    y_label = "summed reconstruction error"
    line_label = "현재 wafer 복원오차 합산"
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
    # ★ 정상 범위 = OK '최소~최대'(min~max). ±2σ 대신(2σ가 임계 1 을 넘어감).
    #   _lookup_ok_band 는 __global__ 폴백 있어 레시피 불일치여도 밴드가 나온다.
    ok_lo_plot, ok_hi_plot = _lookup_ok_band(ok_bands, rec, step_plot, spans, sensor_idx=None)
    plot_vals_raw = _final_score_curve(plot_vals, final_score if apply_final_score else None)
    ok_mean_raw = None if ok_mean_plot is None else np.asarray(ok_mean_plot, dtype=np.float64)
    ok_lo_raw = None if ok_lo_plot is None else np.asarray(ok_lo_plot, dtype=np.float64)
    ok_hi_raw = None if ok_hi_plot is None else np.asarray(ok_hi_plot, dtype=np.float64)
    if scale is not None:
        plot_vals_plot = plot_vals_raw / scale
        ok_mean_display = None if ok_mean_raw is None else ok_mean_raw / scale
        ok_lo_display = None if ok_lo_raw is None else ok_lo_raw / scale
        ok_hi_display = None if ok_hi_raw is None else ok_hi_raw / scale
    else:
        plot_vals_plot = plot_vals_raw
        ok_mean_display = ok_mean_raw
        ok_lo_display = ok_lo_raw
        ok_hi_display = ok_hi_raw

    fig, ax = plt.subplots(figsize=(20, 4.2))
    if (ok_lo_display is not None and ok_hi_display is not None
            and np.isfinite(ok_lo_display).any() and np.isfinite(ok_hi_display).any()):
        valid_band = np.isfinite(ok_lo_display) & np.isfinite(ok_hi_display)
        if valid_band.any():
            ax.fill_between(t, ok_lo_display, ok_hi_display, where=valid_band,
                            color="#87CEEB", alpha=0.24,
                            label="정상(OK) 범위 (최소~최대)", zorder=1)
    if ok_mean_display is not None and np.isfinite(ok_mean_display).any():
        ax.plot(t, ok_mean_display, color="#1f77b4", lw=1.45,
                label="같은 recipe OK 평균 복원오차", zorder=2)
    ax.plot(t, plot_vals_plot, color="#333333", lw=1.45,
            label=line_label, zorder=3)
    if threshold_line is not None:
        ax.axhline(float(threshold_line), color="#d62728", ls="--", lw=1.2,
                   label="threshold", zorder=4)
    _draw_step_guides(ax, bounds, spans, label=True)
    vals_for_ylim = [plot_vals_plot[np.isfinite(plot_vals_plot)]]
    if ok_mean_display is not None:
        vals_for_ylim.append(ok_mean_display[np.isfinite(ok_mean_display)])
    if ok_hi_display is not None:
        vals_for_ylim.append(ok_hi_display[np.isfinite(ok_hi_display)])
    if threshold_line is not None:
        vals_for_ylim.append(np.asarray([float(threshold_line)], dtype=np.float64))
    finite_vals = np.concatenate([v for v in vals_for_ylim if len(v)]) if vals_for_ylim else np.array([])
    ymax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    ax.set_ylim(0, ymax * 1.08 + 1e-9 if ymax > 0 else 1.0)
    ax.set_title(f"{rec['wafer_id']} · summed reconstruction error "
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
        Path(out_path).with_suffix(".csv"), rec, orig_idx, step_plot,
        plot_vals_raw, ok_mean_raw, None, threshold_value, scale,
        ok_lo=ok_lo_raw, ok_hi=ok_hi_raw)


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
    min_highlight_width = int(getattr(A.cfg, "highlight_min_width", 10))

    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        if idx >= len(order):
            ax.axis("off"); continue
        j = order[idx]
        cur, exp = actual[plot_keep, j], expected[plot_keep, j]
        zc = z[plot_keep, j]
        ok_limit = _limit_series(A, rec, j, t_full)[plot_keep]
        # 하이라이트 = 이 센서 누적 이상치 곡선의 기울기 급변(급상승) 구간
        accum = _sensor_reconstruction_accum_series_from_level(zc, step_plot)
        slope_events, slope_thr = _slope_highlight_events(accum, min_highlight_width)
        band_lo = exp - ok_limit
        band_hi = exp + ok_limit
        ylo = float(min(cur.min(), exp.min(), band_lo.min()))
        yhi = float(max(cur.max(), exp.max(), band_hi.max()))
        pad = (yhi - ylo) * 0.08 + 1e-6
        ax.fill_between(t, band_lo, band_hi, color="#87CEEB", alpha=0.22,
                        zorder=0, label="OK 기준 정상 범위")
        _draw_sigma_event_bands(ax, slope_events, slope_thr, color="#d62728",
                                label="이상치 급상승 구간(기울기 급변)",
                                relative_alpha=True)
        _draw_step_guides(ax, bounds, spans, label=True)
        ax.plot(t, exp, color="#1f77b4", lw=1.4, ls="--", label="기대", zorder=4)
        ax.plot(t, cur, color="#d62728", lw=1.4, label="실제", zorder=5)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(f"{sensors[j]}", fontsize=9.5)
        ax.set_xlabel(_time_axis_label(rec), fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{rec['wafer_id']} — 하늘색=OK 기준 정상 범위 · "
                 f"빨간 음영=이상치 급상승(기울기 급변) 구간(진할수록 큼) · "
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
    return {
        "wafer_id": rec["wafer_id"],
        "recipe": _recipe_key(rec),
        "path_id": path_id,
        "path_rank": path_rank,
        "role": role,
        "sensor": row.get("sensor", ""),
        "step": row.get("peak_step", ""),
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


def build_causal_path_rows(A, rec, sensor_rows: list[Dict],
                           max_contributors: int = 6) -> list[Dict]:
    """Build one representative flow-aware RCA path for a wafer.

    The path is an explanation aid, not a causal proof: it prioritizes sensors
    that are earlier in process-flow order, earlier in ProcessStep, and still
    have high model-vs-raw error.
    """
    valid = [
        r for r in sensor_rows
        if r.get("sensor")
        and _safe_int(r.get("flow_rank"), 999999) < 999999
        and _safe_int(r.get("peak_step"), None) is not None
    ]
    if not valid:
        return []

    max_err = max(_safe_float(r.get("error_level")) for r in valid)
    root_floor = max(1.0, max_err * 0.35)
    root_pool = [r for r in valid if _safe_float(r.get("error_level")) >= root_floor]
    root = min(
        root_pool or valid,
        key=lambda r: (
            _safe_int(r.get("peak_step"), 10**9),
            _safe_int(r.get("flow_rank"), 999999),
            -_safe_float(r.get("error_level")),
        ),
    )

    root_rank = _safe_int(root.get("flow_rank"), 999999)
    root_step = _safe_int(root.get("peak_step"), 10**9)

    result_pool = [
        r for r in valid
        if r is not root
        and _safe_int(r.get("peak_step"), 10**9) >= root_step
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
            _safe_int(r.get("peak_step"), 10**9),
        ),
    )
    result_rank = _safe_int(result.get("flow_rank"), root_rank)
    result_step = _safe_int(result.get("peak_step"), root_step)
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
        st = _safe_int(r.get("peak_step"), 10**9)
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
        key=lambda r: (_safe_int(r.get("peak_step"), 10**9),
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
    import matplotlib; matplotlib.use("Agg")
    _setup_font()
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
        ("음영 기준", f"하늘색=기대±OK 기준 오차, "
                  f"빨간 x축 band=|실제-예측|≥{getattr(cfg, 'large_residual_sigma', 4.0)}σ "
                  f"(진할수록 σ 큼)"),
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
    import matplotlib
    matplotlib.use("Agg")
    _setup_font()
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
    yvals = [0.0 if not np.isfinite(v) else float(v) for v in scaled_scores]
    ax.bar(xs, yvals, color=pred_cols)
    finite_vals = [v for v in yvals if np.isfinite(v)]
    drawn_thresholds = _draw_display_thresholds(ax, xs, display_thresholds)
    ymax = max([1.0] + finite_vals + drawn_thresholds) if finite_vals or drawn_thresholds else 1.0
    ax.set_xticks(xs)
    ax.set_xticklabels([t[1] for t in rows], rotation=90, fontsize=6)
    order_label = "wafer 이름순"
    ax.set_ylabel("anomaly score")
    ax.set_xlabel(f"wafer ({order_label})")
    ax.set_title(f"{title_prefix} ({order_label})", fontsize=12)
    ax.set_ylim(0, ymax * 1.12 + 1e-9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_wafer_scores(val_labeled, scores, threshold, csv_path, png_path,
                       prediction_order: str = "score") -> None:
    """웨이퍼별 anomaly score → CSV + 정렬 막대그래프(빨강 NG / 파랑 OK + threshold 선)."""
    import csv as _csv
    rows = []
    for r, lab in val_labeled:
        raw_score = float(scores.get(r["wafer_id"], 0.0))
        display_score, display_threshold = _display_score_threshold(raw_score, threshold, r)
        rows.append((r, r["wafer_id"], lab, raw_score, display_score, display_threshold))
    rows = _sort_prediction_rows(rows, prediction_order, score_index=4)

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
                    "threshold", "pred", "correct",
                    "precision", "recall", "f1_score",
                    "threshold_ratio_percent"])
        for rec, wid, lab, sc, display_score, display_threshold in rows:
            th = _threshold_for_record(threshold, rec)
            if th is not None:
                pred = int(sc >= th)
                w.writerow([wid, _recipe_key(rec), "NG" if lab == 1 else "OK",
                            round(float(display_score), 5),
                            "" if display_threshold is None else round(float(display_threshold), 5),
                            "NG" if pred else "OK", int(pred == lab),
                            *metric_cols,
                            "" if display_threshold is None else round(float(display_score) * 100.0, 3)])
            else:
                w.writerow([wid, _recipe_key(rec), "NG" if lab == 1 else "OK",
                            round(float(display_score), 5), "", "", "", *metric_cols, ""])
    import matplotlib; matplotlib.use("Agg")
    _setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    xs = list(range(len(rows))); vals = [t[4] for t in rows]
    cols = ["#d62728" if t[2] == 1 else "#1f77b4" for t in rows]
    fig, ax = plt.subplots(figsize=(max(7, 0.26 * len(rows) + 2), 5))
    ax.bar(xs, vals, color=cols)
    recipe_spans = []
    if str(prediction_order).lower() == "recipe" and rows:
        prev = _recipe_key(rows[0][0])
        group_start = 0
        ymax = max(vals) if vals else 1.0
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
                ax.hlines(float(display_threshold), start - 0.45, end + 0.45,
                          color="#2ca02c", linestyles="--", lw=1.2, alpha=0.8)
        else:
            if any(row[5] is not None for row in rows):
                _draw_display_thresholds(ax, xs, [row[5] for row in rows])
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
    fig.tight_layout(); fig.savefig(png_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    ratio_path = Path(png_path).with_name(f"{Path(png_path).stem}_threshold_ratio.png")
    _write_threshold_ratio_plot(
        rows, threshold, ratio_path, prediction_order, score_index=3,
        title_prefix="웨이퍼별 anomaly score")


def write_blind_predictions(records, scores, threshold, csv_path, png_path,
                            summary_path=None, prediction_order: str = "score") -> None:
    """라벨 없는 blind wafer별 anomaly score/prediction → CSV + 그래프."""
    import csv as _csv
    import json
    rows = []
    for r in records:
        raw_score = float(scores.get(r["wafer_id"], 0.0))
        display_score, display_threshold = _display_score_threshold(raw_score, threshold, r)
        rows.append((r, r["wafer_id"], raw_score, display_score, display_threshold))
    rows = _sort_prediction_rows(rows, prediction_order, score_index=3)

    pred_counts = {"NG": 0, "OK": 0, "UNKNOWN": 0}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["wafer_id", "recipe", "label", "anomaly_score",
                    "threshold", "pred", "correct",
                    "precision", "recall", "f1_score",
                    "threshold_ratio_percent"])
        for rec, wid, sc, display_score, display_threshold in rows:
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
                        th_out, pred, "", "", "", "",
                        "" if display_threshold is None else round(float(display_score) * 100.0, 3)])

    if summary_path is not None:
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

    import matplotlib; matplotlib.use("Agg")
    _setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    xs = list(range(len(rows)))
    vals = [t[3] for t in rows]
    pred_by_row = []
    for rec, _wid, sc, _display_score, _display_threshold in rows:
        th = _threshold_for_record(threshold, rec)
        pred_by_row.append(None if th is None else int(sc >= float(th)))
    cols = ["#d62728" if p == 1 else "#1f77b4" if p == 0 else "#9e9e9e"
            for p in pred_by_row]
    fig, ax = plt.subplots(figsize=(max(7, 0.26 * len(rows) + 2), 5))
    ax.bar(xs, vals, color=cols)
    recipe_spans = []
    if str(prediction_order).lower() == "recipe" and rows:
        prev = _recipe_key(rows[0][0])
        group_start = 0
        ymax = max(vals) if vals else 1.0
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
                    ax.hlines(float(display_threshold), start - 0.45, end + 0.45,
                              color="#2ca02c", linestyles="--", lw=1.2, alpha=0.8)
        else:
            if any(row[4] is not None for row in rows):
                _draw_display_thresholds(ax, xs, [row[4] for row in rows])
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
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    ratio_path = Path(png_path).with_name(f"{Path(png_path).stem}_threshold_ratio.png")
    _write_threshold_ratio_plot(
        rows, threshold, ratio_path, prediction_order, score_index=2,
        title_prefix="Blind wafer prediction")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def cmd_recon_explain(cfg: IOConfig, top_k=None, ng_only: bool = False,
                      report_dir: Path = None, prediction_order: str = "score",
                      ok_mean_exclude_over_threshold: bool = False) -> None:
    import json
    from dataclasses import replace
    adir = Path(cfg.artifacts_dir)
    apath = adir / "tcn_forecast.pt"
    if not apath.exists():
        sys.exit(f"저장된 TCN forecast 모델 없음: {apath} — 먼저 tcn-search 하세요.")

    # ★ search_best.json: best params + 저장된 wafer별 점수(재사용 → search 와 동일).
    thr, best_params, best_scores, best_score_kind = None, None, None, None
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
        except Exception as e:
            print(f"  ⚠ search_best 적용 실패(기본값으로 채점): {e}")
    else:
        print("  ⚠ search_best.json 없음: threshold/저장 score 없이 모델로 재채점")

    from .model_tcn_forecast import TCNForecastDetector   # 지연 import(torch)
    A = TCNForecastDetector.load(adir, cfg).A
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
            threshold=thr, score_kind=best_score_kind,
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
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    (out / "heatmap").mkdir(parents=True, exist_ok=True)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    (out / "step_summary").mkdir(parents=True, exist_ok=True)
    (out / "rca_path").mkdir(parents=True, exist_ok=True)
    _setup_font()

    # ── 성능표 ──
    val_labeled = [(r, 0) for r in vok] + [(r, 1) for r in vng]
    run_threshold = thr
    score_by_id = {}
    # 서치 저장 점수가 검증셋 전체를 덮으면 그대로 재사용 → search 결과와 100% 일치
    use_scores = None
    if val_labeled and best_scores:
        val_ids = {r["wafer_id"] for r, _ in val_labeled}
        score_ids = set(best_scores)
        missing_ids = sorted(val_ids - score_ids)
        if not missing_ids:
            use_scores = best_scores
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
            summ = perf_summary(A, val_labeled, cfg, threshold=thr,
                                params=best_params, scores=use_scores,
                                score_kind=best_score_kind)
            plot_perf_table(summ, cfg, out / "perf_summary.png")
            raw_score_map = dict(summ.get("raw_scores") or summ.get("scores") or {})
            (out / "perf_summary.json").write_text(
                json.dumps(_display_summary_json(summ, best_params),
                           ensure_ascii=False, indent=2,
                           default=_json_default), encoding="utf-8")
            v = summ["vc"]
            print(f"[성능표] 정답 {v['correct']}/{v['n']} · 불량 {v['tp']}/{v['n_ng']} · "
                  f"AUROC {v['auroc']} · thr {'1.0000' if summ['threshold'] is not None else '—'} · "
                  f"OK복원MSE {summ['val_mse']:.4f} → {out/'perf_summary.png'}")
            run_threshold = summ["threshold"]
            score_by_id = raw_score_map
            write_wafer_scores(val_labeled, raw_score_map, summ["threshold"],
                               out / "predictions.csv", out / "predictions.png",
                               prediction_order=prediction_order)
            print(f"[웨이퍼별 anomaly score] → {out/'predictions.csv'} · {out/'predictions.png'}")
        except Exception as e:
            print(f"  ⚠ 성능표/웨이퍼점수 생성 실패: {e}")
    elif targets:
        recs = [r for r, _ in targets]
        try:
            p = _prediction_scores(A, recs, run_threshold, best_score_kind)
            score_by_id = {r["wafer_id"]: float(v) for r, v in zip(recs, p)}
            if blind_mode:
                write_blind_predictions(
                    recs, score_by_id, run_threshold,
                    out / "predictions.csv", out / "predictions.png",
                    summary_path=out / "blind_prediction_summary.json",
                    prediction_order=prediction_order)
                print(f"[blind prediction] {len(recs)}장 → "
                      f"{out/'predictions.csv'} · {out/'predictions.png'}")
            else:
                write_wafer_scores(targets, score_by_id, run_threshold,
                                   out / "predictions.csv", out / "predictions.png",
                                   prediction_order=prediction_order)
                print(f"[웨이퍼별 anomaly score] → {out/'predictions.csv'} · {out/'predictions.png'}")
        except Exception as e:
            print(f"  ⚠ prediction 생성 실패: {e}")

    sig = float(getattr(A.cfg, "raw_error_level", 1.0))
    interp_factor = _explain_interp_factor(A.cfg)
    step_accum_threshold = (
        run_threshold if _uses_step_accum_threshold(run_threshold, best_score_kind) else None
    )
    score_mode_label = (
        "sum_all_sensor_reconstruction_error"
        if _uses_sum_reconstruction_threshold(run_threshold, best_score_kind)
        else "step_accum_excess_reconstruction_error"
        if _uses_step_accum_threshold(run_threshold, best_score_kind)
        else "model.score_records"
    )
    print(f"  · prediction score mode: {score_mode_label}"
          f"{'' if best_score_kind is None else f' (saved score_kind={best_score_kind})'}")
    if run_threshold is not None and step_accum_threshold is None:
        print("  · saved threshold는 wafer score 기준이므로 sensor/step 누적 threshold 시각화는 생략")
    print(f"[recon-explain] TCN 모델 {apath} → {len(targets)} wafer "
          f"(합본 + 센서별 고해상도, interpolation x{interp_factor}) → {out}")
    narr = []
    all_sensor_rows = []
    all_step_rows = []
    all_causal_rows = []
    all_threshold_rows = []
    for i, (r, lab) in enumerate(targets, 1):
        wid = r["wafer_id"]
        try:
            actual_raw, expected_raw, z_raw = _compute(A, r)  # wafer 1회 계산 후 재사용
            threshold_rows = build_threshold_exceedances(
                A, r, actual_raw, expected_raw, z_raw, step_accum_threshold, label=lab)
            all_threshold_rows.extend(threshold_rows)
            r_view, actual, expected, z = _with_interpolated_explain_arrays(
                A, r, actual_raw, expected_raw, z_raw)
            order = _sensor_order(z, sig)
            sel = _flow_sensor_order(A, order if not top_k else order[:int(top_k)])
            finding_top_n = min(len(order), max(20, int(top_k or 0)))
            sensor_rows = build_sensor_findings(
                A, r_view, actual, expected, z, order, top_n=finding_top_n)
            step_rows = build_step_summary(A, r_view, actual, expected, z)
            causal_rows = build_causal_path_rows(A, r_view, sensor_rows)
            all_sensor_rows.extend(sensor_rows)
            all_step_rows.extend(step_rows)
            all_causal_rows.extend(causal_rows)
            score = float(score_by_id.get(wid, _score_from_level(A, z_raw)))
            plot_wafer_summary(A, r_view, lab, actual, expected, z,
                               sensor_rows, step_rows, score, run_threshold,
                               out / "summary" / f"{wid}.png",
                               causal_rows=causal_rows)
            plot_step_summary(step_rows, out / "step_summary" / f"{wid}.png")
            plot_causal_path(A, r_view, causal_rows, out / "rca_path" / f"{wid}.png")
            # 합본(전 센서 한 장)
            plot_recon_overlay(A, r_view, out / "overlay" / f"{wid}.png",
                               top_k=top_k, pre=(actual, expected, z),
                               threshold=run_threshold)
            # ★ 센서별 개별 고해상도 (per-wafer 폴더)
            sdir = out / "overlay" / wid; sdir.mkdir(parents=True, exist_ok=True)
            plot_wafer_accum_score(A, r_view, z, sdir / "00_wafer_accum_score.png",
                                   ok_bands=ok_band_cache,
                                   threshold=run_threshold,
                                   final_score=score,
                                   raw_rec=r, raw_z=z_raw)
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
                                  raw_rec=r, raw_z=z_raw)
            plot_recon_heatmap(A, r_view, out / "heatmap" / f"{wid}.png", z_pre=z)
            n = narrate_recon(A, r_view, top_k=3, pre=(actual, expected, z))
            n["label"] = "BLIND" if lab is None else lab
            narr.append(n)
            tag = "BLIND" if lab is None else ("NG" if lab == 1 else "OK")
            print(f"  [{i:>3}/{len(targets)}] [{tag}] {wid} (센서 {len(sel)}장) — "
                  f"{n['narrative'][:80]}")
        except Exception as e:
            print(f"  [{i:>3}/{len(targets)}] {wid} 실패: {e}")
    (out / "recon_explain.json").write_text(
        json.dumps(narr, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sensor_findings(all_sensor_rows, out / "sensor_findings.csv",
                          out / "sensor_findings.md")
    write_step_summary(all_step_rows, out / "step_summary.csv")
    write_causal_paths(all_causal_rows, out / "causal_paths.csv",
                       out / "causal_paths.md")
    write_threshold_exceedances(all_threshold_rows, out / "threshold_exceedances.csv")
    write_threshold_exceedances_by_step(
        all_threshold_rows, targets, out / "threshold_exceedances_by_step.csv")
    print(f"완료: {out} (overlay/<wafer>.png 합본 · overlay/<wafer>/ 센서별 · "
          f"heatmap/ · summary/ · step_summary/ · rca_path/ · sensor_findings.csv · "
          f"step_summary.csv · causal_paths.csv · threshold_exceedances.csv · "
          f"threshold_exceedances_by_step.csv · predictions.csv · perf_summary)")
