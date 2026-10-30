"""IO 모델 결과를 기존 RCA report-dir 계약으로 내보내기.

계약 파일:
  summary.json
  predictions.csv / blind_metrics.json
  err_ok_mean.npy / err_ng_mean.npy / wafer_errors/<wid>.npy
  shap_explanations.json
  counterfactuals.csv

여기서 err/wafer_errors 값은 표준편차 z가 아니라
|실제-기대| / OK raw residual 분위수 기준 error level 이다.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import IOConfig
from .sensor_graph import graph_adjust_abs_z


SEMES = Path("/home/kjw/server/semes")


def _semes_root() -> Path:
    candidates = [
        SEMES,
        Path.home() / "server" / "semes",
        Path("/home/node/semes"),
    ]
    for root in candidates:
        if (root / "scripts").is_dir():
            return root
    return SEMES


def _step_map(z_abs: np.ndarray, step, n_steps: int) -> np.ndarray:
    """(T, F) |z| 를 (n_steps, F) step 평균 맵으로 변환."""
    F = z_abs.shape[1]
    out = np.zeros((n_steps, F), dtype=np.float32)
    if step is None:
        idx = (np.linspace(0, n_steps, len(z_abs), endpoint=False)).astype(int)
        step = np.clip(idx, 0, n_steps - 1)
    for s in range(n_steps):
        m = step == s
        if m.any():
            out[s] = z_abs[m].mean(axis=0)
    return out


def _model_error_level(model_a, rec) -> np.ndarray:
    if hasattr(model_a, "error_level"):
        return np.asarray(model_a.error_level(rec), dtype=np.float32)
    return np.abs(model_a.z_residual(rec)).astype(np.float32)


def _n_steps_for_records(records: list[dict], model_a) -> int:
    smax = [int(r["step"].max()) for r in records if r.get("step") is not None]
    if smax:
        return max(2, max(smax) + 1)
    return max(2, int(getattr(model_a, "n_steps_global", 0) or 8))


def _score_map_from_model(hybrid, ok_records, ng_records) -> tuple[dict | None, dict]:
    """Return explicit score_map when available, otherwise cv_report info."""
    if hasattr(hybrid, "cv_report"):
        cvres = {w["wafer_id"]: w for w in hybrid.cv_report(ok_records, ng_records)}
        return None, cvres
    if hasattr(hybrid, "score_records"):
        recs = list(ng_records) + list(ok_records)
        scores = hybrid.score_records(recs)
        return {r["wafer_id"]: float(s) for r, s in zip(recs, scores)}, {}
    raise TypeError("export 대상 모델은 cv_report 또는 score_records 를 제공해야 합니다.")


def _recipe_key(rec) -> str:
    return str(rec.get("recipe", "unknown") or "unknown")


def _threshold_for_record(threshold, rec) -> float | None:
    if threshold is None:
        return None
    if isinstance(threshold, dict):
        if str(threshold.get("threshold_wise", "")).lower() in {"sensor", "step"}:
            return float(threshold.get("decision_threshold", 1.0))
        table = threshold.get("threshold_by_recipe") or threshold.get("threshold_raw") or threshold
        recipe = _recipe_key(rec)
        if recipe in table:
            return float(table[recipe])
        for key in ("default", "global", "global_threshold_raw", "global_threshold"):
            if key in table:
                return float(table[key])
        return None
    return float(threshold)


def _threshold_mode(threshold) -> str:
    if isinstance(threshold, dict) and threshold.get("threshold_wise"):
        return str(threshold.get("threshold_wise")).lower()
    if isinstance(threshold, dict):
        return "recipe"
    return "all"


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
    return _threshold_for_record(threshold, rec)


def _decision_score_from_threshold(model_a, rec, threshold) -> float:
    level = _model_error_level(model_a, rec).astype(np.float64)
    if level.ndim != 2 or level.size == 0:
        return 0.0
    step = rec.get("step")
    step_arr = np.asarray(step).astype(int) if step is not None and len(step) == level.shape[0] \
        else np.ones(level.shape[0], dtype=int)
    excess = np.maximum(0.0, level - 1.0)
    mode = _threshold_mode(threshold)
    best = 0.0
    for st in sorted(int(x) for x in np.unique(step_arr) if int(x) != 0):
        mask = step_arr == st
        vals = excess[mask].sum(axis=0)
        for j, sensor in enumerate(model_a.sensor_cols):
            score = float(vals[j])
            th = _sensor_step_threshold(threshold, rec, str(sensor), st)
            if th is None:
                continue
            if mode in {"sensor", "step"}:
                best = max(best, score / max(float(th), 1e-12))
            else:
                best = max(best, score)
    return float(best)


def build_report(hybrid, ok_records, ng_records, out_dir: Path,
                 cfg: IOConfig, threshold: float = 0.5,
                 score_map: dict = None) -> Path:
    """IO 하이브리드 결과를 report-dir 계약으로 기록한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wafer_errors").mkdir(exist_ok=True)

    A = hybrid.A
    sensors = list(A.sensor_cols)
    recs = list(ng_records) + list(ok_records)
    labels = {r["wafer_id"]: (1 if i < len(ng_records) else 0)
              for i, r in enumerate(recs)}
    n_steps = _n_steps_for_records(recs, A)

    cvres = {}
    if score_map is None:
        if threshold is not None:
            score_map = {
                r["wafer_id"]: _decision_score_from_threshold(A, r, threshold)
                for r in recs
            }
        else:
            score_map, cvres = _score_map_from_model(hybrid, ok_records, ng_records)

    ok_maps, ng_maps = [], []
    preds = []
    shap_out = {}
    cf_rows = []
    for r in recs:
        wid = r["wafer_id"]
        z = _model_error_level(A, r)
        z = graph_adjust_abs_z(z, sensors, cfg).astype(np.float32)
        smap = _step_map(z, r.get("step"), n_steps)
        np.save(out_dir / "wafer_errors" / f"{wid}.npy", smap)
        (ng_maps if labels[wid] else ok_maps).append(smap)

        score = float(smap.sum(axis=1).max() if len(smap) else 0.0)
        if score_map is not None:
            raw_dp = score_map.get(wid)
            if raw_dp is not None:
                score = float(raw_dp)
        else:
            info = cvres.get(wid, {})
            raw_dp = info.get("defect_prob")
            score = float(info.get("score", score))
        dp = None if raw_dp is None else float(raw_dp)
        row_threshold = _threshold_for_record(threshold, r)
        pl = 1 if (dp is not None and row_threshold is not None and dp >= row_threshold) else 0
        preds.append({
            "wafer_id": wid,
            "recipe": _recipe_key(r),
            "label": labels[wid],
            "anomaly_score": round(score, 6),
            "predicted_label": pl,
            "true_label": labels[wid],
            "defect_prob": None if dp is None else round(dp, 6),
            "threshold": None if row_threshold is None else round(float(row_threshold), 6),
            "correct": int(labels[wid] == pl),
        })

        per_sensor = smap.sum(axis=0)
        order = np.argsort(per_sensor)[::-1]
        shap_out[wid] = {
            "wafer_score": score,
            "base_value": 0.0,
            "sensor_importance": [
                {"sensor_name": sensors[j], "abs_importance": float(per_sensor[j])}
                for j in order[:15]
            ],
            "top_sensors": [sensors[j] for j in order[:10]],
        }

        pred_sensor = A.predict(r)
        actual_sensor = A.actual(r) if hasattr(A, "actual") else r["sensors"]
        for rank, j in enumerate(order[:10], 1):
            di = int(np.argmax(z[:, j]))
            cur = float(actual_sensor[di, j])
            tgt = float(pred_sensor[di, j])
            st = int(r["step"][di]) if r.get("step") is not None else di
            cf_rows.append({
                "wafer_id": wid,
                "rank": rank,
                "sensor": sensors[j],
                "sensor_idx": int(j),
                "step": st,
                "current_value": round(cur, 4),
                "target_value": round(tgt, 4),
                "suggested_change": round(tgt - cur, 4),
                "score_before": round(float(z[:, j].sum()), 4),
                "score_after": 0.0,
                "score_reduction": round(float(per_sensor[j]), 4),
            })

    if ok_maps:
        np.save(out_dir / "err_ok_mean.npy", np.mean(ok_maps, axis=0))
    if ng_maps:
        np.save(out_dir / "err_ng_mean.npy", np.mean(ng_maps, axis=0))

    preds.sort(key=lambda x: x["anomaly_score"], reverse=True)
    for rank, row in enumerate(preds, 1):
        row["rank"] = rank

    ng_pred_ids = [p["wafer_id"] for p in preds
                   if p["predicted_label"] == 1 or p["label"] == 1]
    ok_ids = [p["wafer_id"] for p in preds if p["label"] == 0]

    y = [p["label"] for p in preds]
    yp = [p["predicted_label"] for p in preds]
    tp = sum(1 for a, b in zip(y, yp) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, yp) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yp) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, yp) if a == 0 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    for row in preds:
        row["precision"] = round(float(prec), 6)
        row["recall"] = round(float(rec), 6)
        row["f1_score"] = round(float(f1), 6)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["wafer_id", "recipe", "label", "anomaly_score", "predicted_label",
                      "true_label", "defect_prob", "threshold", "correct", "rank",
                      "precision", "recall", "f1_score"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(preds)

    with open(out_dir / "counterfactuals.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(cf_rows[0].keys()) if cf_rows else ["wafer_id"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(cf_rows)

    json.dump(shap_out, open(out_dir / "shap_explanations.json", "w",
                             encoding="utf-8"),
              ensure_ascii=False, indent=2)

    json.dump({
        "precision": prec,
        "recall": rec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
        "threshold_by_recipe": threshold if isinstance(threshold, dict) else None,
        "n_labeled": len(y),
        "f1": f1,
    }, open(out_dir / "blind_metrics.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    json.dump({
        "sensor_names": sensors,
        "ng_pred_ids": ng_pred_ids,
        "ok_ids": ok_ids,
        "blind_ids": [p["wafer_id"] for p in preds],
        "val_mode": True,
        "io_model": True,
        "model_type": hybrid.__class__.__name__,
        "n_steps": n_steps,
        "n_sensors": len(sensors),
        "threshold": threshold,
        "threshold_by_recipe": threshold if isinstance(threshold, dict) else None,
        "error_unit": "raw_abs_error / OK residual quantile; not residual/std",
        "raw_error_quantile": getattr(cfg, "raw_error_quantile", None),
        "score_exclude_boundary_step0": bool(getattr(cfg, "score_exclude_boundary_step0", True)),
    }, open(out_dir / "summary.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)
    return out_dir


def _existing_csv_dirs(cfg: IOConfig) -> list[Path]:
    dirs = [
        Path(cfg.ng_dir),
        Path(cfg.ok_dir),
        Path(cfg.val_dir),
        Path(cfg.val_dir) / "ok",
        Path(cfg.val_dir) / "ng",
        Path(cfg.val_dir) / "OK",
        Path(cfg.val_dir) / "NG",
    ]
    out = []
    seen = set()
    for d in dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        if d.exists():
            out.append(d)
    return out


def _no_window_kwargs() -> dict:
    """Windows 에서 콘솔창(cmd) 깜빡임 방지."""
    if os.name != "nt":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0
    return {"startupinfo": si,
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


def _run_helper(name: str, cmd: list[str], cwd: Path, timeout: int = 1800) -> bool:
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                       encoding="utf-8", errors="replace", timeout=timeout,
                       **_no_window_kwargs())
    if r.returncode == 0:
        print(f"  [export] {name}: OK")
        return True
    tail = (r.stderr or r.stdout)[-300:]
    print(f"  [export] {name}: 실패 {tail}")
    return False


def _auto_error_vmax(report_dir: Path) -> float:
    """Export된 |z-residual| 맵에서 히트맵 색상 스케일을 자동 산정."""
    arrays = []
    for path in (Path(report_dir) / "err_ok_mean.npy",
                 Path(report_dir) / "err_ng_mean.npy"):
        if path.exists():
            try:
                arrays.append(np.abs(np.load(path)).ravel())
            except Exception:
                pass
    wafer_dir = Path(report_dir) / "wafer_errors"
    if wafer_dir.exists():
        for path in sorted(wafer_dir.glob("*.npy")):
            try:
                arrays.append(np.abs(np.load(path)).ravel())
            except Exception:
                pass

    finite = []
    for arr in arrays:
        arr = arr[np.isfinite(arr)]
        arr = arr[arr > 0]
        if arr.size:
            finite.append(arr)
    if not finite:
        return 10.0
    vals = np.concatenate(finite)
    vmax = float(np.percentile(vals, 99.0))
    return max(vmax * 1.05, 1.0)


def run_existing_rca(report_dir: Path, cfg: IOConfig,
                     baseline: bool = True, replot: bool = True,
                     narrative: bool = True) -> None:
    """기존 SEMES RCA helper를 실행한다."""
    semes = _semes_root()
    py = sys.executable
    csv_dirs = _existing_csv_dirs(cfg)
    error_vmax = _auto_error_vmax(report_dir)
    print(f"  [export] heatmap/anomaly color scale: vmax={error_vmax:.6g}")

    if baseline:
        script = semes / "scripts" / "wafer_vs_baseline.py"
        if not script.exists():
            print(f"  [export] wafer_vs_baseline: 생략({script} 없음)")
        else:
            cmd = [py, str(script),
                   "--report-dir", str(report_dir),
                   "--ok-csv-dir", str(cfg.ok_dir),
                   "--project-root", str(semes),
                   "--anomaly-vmax", f"{error_vmax:.9g}"]
            for d in csv_dirs:
                cmd.extend(["--wafer-csv-dir", str(d)])
            _run_helper("wafer_vs_baseline", cmd, semes)

    if replot:
        script = semes / "scripts" / "replot_heatmaps_v3.py"
        if not script.exists():
            print(f"  [export] replot_heatmaps_v3: 생략({script} 없음)")
        else:
            cmd = [py, str(script),
                   "--report-dir", str(report_dir),
                   "--project-root", str(semes),
                   "--search-all-dirs",
                   "--vmax", f"{error_vmax:.9g}"]
            for d in csv_dirs:
                cmd.extend(["--csv-dir", str(d)])
            _run_helper("replot_heatmaps_v3", cmd, semes)

    if narrative:
        if not semes.exists():
            print(f"  [export] 자연어 RCA(Gemma): 생략({semes} 없음)")
        else:
            code = ("import sys; sys.path.insert(0,'.');"
                    "from src.reporting.val_rca_narrative import generate_report_rca;"
                    f"generate_report_rca(r'{report_dir}')")
            r = subprocess.run([py, "-c", code], cwd=str(semes), capture_output=True,
                               encoding="utf-8", errors="replace", timeout=1800,
                               **_no_window_kwargs())
            print("  [export] 자연어 RCA(Gemma):",
                  "OK" if r.returncode == 0
                  else f"생략/실패 {(r.stderr or r.stdout)[-160:]}")
