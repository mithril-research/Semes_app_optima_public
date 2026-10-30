# -*- coding: utf-8 -*-
"""[골든 비교] 더미 데이터 생성기 — 설비 2종 × DSV 3종 + 불량조치내역 CSV.

· 센서 수 = 모델 원본 그대로(recon_meta 의 전체 센서+IO 열)
· 웨이퍼 길이 = 3,500~3,900pt 랜덤(스텝 비례 신축) — 실데이터 길이 편차 재현
· 종류: norm(정상) / linA(온도 지수→직선: 형상 불량) / bump(압력 스텝3 봉우리
  추가: 형상 불량) / long(형상 정상·스텝5 길이 +40%: 길이 불량) /
  skip(스텝6 누락: 판정불가 검증용)

사용: python3 make_gv_dummy.py <출력폴더> [모델폴더]
  → <출력폴더>/data/*.log + <출력폴더>/불량조치내역_예시.csv
"""
import datetime as _dt
import sys
from pathlib import Path

import numpy as np

BASE_STEPS = [(0, 300), (1, 400), (2, 500), (3, 500), (4, 500), (5, 800),
              (6, 400), (7, 300)]                      # 합 3,700


def _steps_for(kind, rng):
    scale = rng.uniform(3500.0, 3900.0) / 3700.0       # 전체 길이 랜덤
    steps = []
    for s, n in BASE_STEPS:
        if kind == "skip" and s == 6:
            continue
        m = n * scale * (1.4 if (kind == "long" and s == 5) else 1.0)
        steps.append((s, max(2, int(round(m)))))
    return steps


def gen(outdir, model_dir, plan, seed=7, day0=1, per_day=8):
    z = np.load(str(Path(model_dir) / "recon_meta.npz"), allow_pickle=True)
    sens = [str(s) for s in z["input_sensor_cols"]]
    ios = [str(s) for s in z["io_cols"]]
    rng = np.random.default_rng(seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["Date", "Time", "ProcessStep"] + sens + ios
    iT = sens.index("t_002_PV")
    iP = sens.index("p_008_PV")
    made = []
    for eqp, dsv, kinds in plan:
        rng.shuffle(kinds)
        for k, kind in enumerate(kinds):
            day = day0 + k // per_day
            steps = _steps_for(kind, rng)
            stp = np.concatenate([np.full(n, s) for s, n in steps])
            T = len(stp)
            vals = np.zeros((T, len(sens) + len(ios)))
            for j in range(len(sens)):                 # 배경 파형+잡음
                base = 30.0 + 3 * (j % 7) + stp * 0.35
                vals[:, j] = base + np.sin(np.linspace(0, np.pi, T)) \
                    * (1 + j % 3) + rng.normal(0, 0.05, T)
            m5 = stp == 5
            n5 = int(m5.sum())
            hi, lo = 80.0, 40.0
            y5 = (np.linspace(hi, lo, n5) if kind == "linA"
                  else lo + (hi - lo) * np.exp(-4.0 * np.linspace(0, 1, n5)))
            vals[m5, iT] = y5 + rng.normal(0, 0.08, n5)
            vals[stp < 5, iT] = hi + rng.normal(0, 0.08, int((stp < 5).sum()))
            vals[stp > 5, iT] = vals[m5, iT][-1] \
                + rng.normal(0, 0.08, int((stp > 5).sum()))
            if kind == "bump":                         # 압력 스텝3 봉우리 추가
                m3 = stp == 3
                n3 = int(m3.sum())
                vals[m3, iP] += 6.0 * np.exp(
                    -0.5 * ((np.arange(n3) - n3 / 2) / (n3 / 10)) ** 2)
            date9 = (_dt.date(2026, 3, 1)
                     + _dt.timedelta(days=day - 1)).isoformat()
            stem = f"{eqp}_{dsv}_29aece39_d{day:02d}_k{k:03d}_{kind}"
            with open(out / (stem + ".log"), "w", encoding="utf-8",
                      newline="") as fh:
                fh.write("\t".join(cols) + "\r\n")
                for i in range(T):
                    row = [date9,
                           f"{9 + (k % 6):02d}:{(i // 60) % 60:02d}:{i % 60:02d}",
                           str(int(stp[i]))] + [f"{v:.4f}" for v in vals[i]]
                    fh.write("\t".join(row) + "\r\n")
            made.append((stem, eqp, dsv, kind, date9, T))
    return made


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "gv_dummy")
    model = sys.argv[2] if len(sys.argv) > 2 else \
        "/home/node/io_dev/io_recon_tcn/artifacts"
    plan = [
        ("WUKA", "dsv1", ["norm"] * 30 + ["linA"] * 3 + ["long"] * 2 + ["skip"]),
        ("WUKA", "dsv2", ["norm"] * 26 + ["bump"] * 2 + ["long"] * 2),
        ("WUKA", "dsv3", ["norm"] * 24),
        ("WUKB", "dsv1", ["norm"] * 30 + ["linA"] * 2),
    ]
    made = gen(root / "data", model, plan)
    # 불량조치내역 CSV: WUKA dsv1 의 3/02 보고 — 그 날 dsv1 은 골든 평균 제외
    (root / "불량조치내역_예시.csv").write_text(
        "설비명,dsv,불량원인,발생일자\nWUKA,dsv1,파티클,20260302\n"
        "WUKA,dsv1,얼룩,20260302\n", encoding="utf-8")
    from collections import Counter
    lens = [t for *_x, t in made]
    print(f"생성 {len(made)}장 → {root}/data · 길이 {min(lens)}~{max(lens)}pt")
    print("구성:", dict(Counter((e, d) for _s, e, d, _k, _dt2, _t in made)))
    print("종류:", dict(Counter(k for _s, _e, _d, k, _dt2, _t in made)))


if __name__ == "__main__":
    main()
