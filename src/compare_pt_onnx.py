"""PT(torch) vs ONNX 기대값 비교 — 앱과 '완전히 같은 조건'으로.

왜 이 스크립트인가
------------------
플레인 torch 로 대충 비교하면 모델이 아니라 **입력 구성**이 달라서 기대값이
어긋난다. 실제로 어긋나는 지점은 대부분 이 네 가지다.

  #1 io_keep      : IOConfig 기본은 IO 12개 화이트리스트, 앱은 () (전체 로드)
  #2 IO 재정렬    : IO 는 위치기반 입력 — 앱은 모델 io_cols 이름 순서로 재정렬
  #3 AMP          : torch 가 CUDA 에서 bfloat16 autocast (tcn_amp 기본 True)
  #4 params 주입  : 앱은 search_best.json 의 best.params 를 cfg 에 주입

이 스크립트는 넷 다 앱과 같게 맞춘 뒤 두 백엔드의 기대값(predict)을 비교한다.
그래도 차이가 크면 그때는 .pt ↔ .onnx 짝이 다른 시점의 파일인지 의심하라
(export 시 verify 가 넷 단독 1e-4 이내 일치를 보증한다).

사용법
------
    python -m src.compare_pt_onnx <모델폴더> <데이터폴더> [--limit 5] [--gpu]

  모델폴더: tcn_forecast.pt + tcn_forecast.onnx(+recon_meta.npz) + search_best.json
  --gpu   : torch 를 CUDA 로(기본 CPU). AMP 는 어느 쪽이든 끈다(#3).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"

import json
from dataclasses import replace

import numpy as np

from .config import IOConfig
from .data import load_dataset


def _build_cfg(model_dir: Path) -> IOConfig:
    """#1 + #4 — 앱(ENGINE._build_cfg)과 같은 cfg."""
    cfg = replace(IOConfig(), io_keep=())                 # #1 전체 IO 로드
    sb = model_dir / "search_best.json"
    if sb.exists():
        try:
            root = json.loads(sb.read_text(encoding="utf-8"))
            params = (root.get("best") or {}).get("params") or {}
            fields = set(IOConfig().__dataclass_fields__)
            cfg = replace(cfg, **{k: v for k, v in params.items()
                                  if k in fields and k != "io_keep"})   # #4 params 주입
            if params:
                print(f"[cfg] search_best.json params 주입: "
                      f"{sorted(k for k in params if k in fields and k != 'io_keep')}")
        except Exception as e:                            # noqa: BLE001
            print(f"[cfg] search_best.json 읽기 실패(기본값 사용): {e}")
    return cfg


def _align_io(recs, model_io_cols):
    """#2 — 각 rec 의 IO 를 모델 io_cols 이름 순서로 재정렬(앱 _load_aligned 와 동일)."""
    model_io = [str(c) for c in model_io_cols]
    for rec in recs:
        load_io = [str(c) for c in (rec.get("io_cols") or [])]
        idx = {name: i for i, name in enumerate(load_io)}
        src = np.asarray(rec["io"], dtype=np.float64)
        out = np.zeros((src.shape[0], len(model_io)), dtype=np.float64)
        miss = []
        for j, name in enumerate(model_io):
            si = idx.get(name)
            if si is not None and si < src.shape[1]:
                out[:, j] = src[:, si]
            else:
                miss.append(name)
        rec["io"] = out
        rec["io_cols"] = list(model_io)
        if miss:
            rec.setdefault("_io_missing", miss)
    return recs


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="PT vs ONNX 기대값 비교(앱과 동일 조건)")
    ap.add_argument("model_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--limit", type=int, default=5, help="비교할 웨이퍼 수(기본 5)")
    ap.add_argument("--gpu", action="store_true", help="torch 를 CUDA 로(기본 CPU)")
    a = ap.parse_args()

    model_dir, data_dir = Path(a.model_dir), Path(a.data_dir)
    cfg = _build_cfg(model_dir)

    # ── 두 백엔드 로드 ──
    from .onnx_backend_forecast import OnnxForecastModel
    from .model_tcn_forecast import TCNForecastModel
    O = OnnxForecastModel.load(model_dir, cfg)
    device = "cuda" if a.gpu else "cpu"
    T = TCNForecastModel.load(next(iter(sorted(model_dir.glob("*.pt")))), cfg, device=device)
    T.use_amp = False                                     # #3 AMP 끔 — float32 로 통일
    print(f"[load] onnx={Path(O._onnx_path).name} · torch(.pt, {device}, amp=off)")

    # ── 데이터 — 앱과 같은 경로(load_dataset) + IO 재정렬 ──
    recs, io_cols, sen_cols, _ = load_dataset(data_dir, cfg)
    if not recs:
        sys.exit(f"데이터 없음: {data_dir}")
    recs = recs[: max(1, a.limit)]
    _align_io(recs, O.io_cols)
    miss_io = sorted({m for r in recs for m in r.get("_io_missing", [])})
    if miss_io:
        print(f"[data] ⚠ 폴더에 없는 모델 IO {len(miss_io)}개(0 채움): {miss_io[:10]}")
    miss_sen = [c for c in map(str, O.input_sensor_cols) if c not in set(map(str, sen_cols))]
    if miss_sen:
        print(f"[data] ⚠ 폴더에 없는 모델 입력 센서 {len(miss_sen)}개(0 채움): {miss_sen[:10]}")

    # ── 웨이퍼별 기대값 비교 ──
    scols = [str(c) for c in O.sensor_cols]
    worst = (0.0, "", "")
    print(f"\n{'웨이퍼':<28} {'max|Δ|':>10} {'mean|Δ|':>10}  최악 센서")
    for rec in recs:
        po = O.predict(rec)                               # (T, S) 물리 단위
        pt_ = T.predict(dict(rec))                        # dict 사본 — 캐시 간섭 차단
        d = np.abs(po - pt_)
        mx = float(d.max()); mn = float(d.mean())
        jworst = int(np.unravel_index(np.argmax(d), d.shape)[1])
        print(f"{str(rec['wafer_id'])[:28]:<28} {mx:>10.5f} {mn:>10.6f}  {scols[jworst]}")
        if mx > worst[0]:
            worst = (mx, str(rec["wafer_id"]), scols[jworst])

    print(f"\n최대 차이: {worst[0]:.5f} (웨이퍼 {worst[1]} · 센서 {worst[2]})")
    if worst[0] < 1e-3:
        print("✅ 두 백엔드 기대값 일치(1e-3 미만) — 이전에 본 차이는 입력 구성(#1/#2/#3/#4) 원인")
    else:
        print("⚠ 조건을 맞췄는데도 차이가 남 — .pt 와 .onnx 가 같은 학습 시점의 짝인지 확인하세요\n"
              "  (재변환: python -m src.export_onnx_forecast <모델폴더>)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
