#!/usr/bin/env python3
"""배포 환경 시뮬레이션 — 실제 server.py 코드로 항상성(불변식 10종) 검증.

시나리오·불변식 정의: docs/08_배포_시뮬레이션_시나리오.md
    python3 sim_deploy.py          # 전부 실행, 실패 있으면 종료코드 1

실제로 쓰는 것: AutoPilot._scan_folder(자동검사) · _load_aligned(파일 파싱·정렬) ·
_result_for(판정) · _archive_wafer(보관) · 이력 인덱스 · api_overview/scatter/
sensors-all(화면 API). 모델만 결정적 가짜(FakeModel)로 주입한다 — 추론 무게만
빼고 경로는 전부 실코드다.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("IO_NO_BROWSER", "1")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np                                        # noqa: E402

import server as S                                        # noqa: E402
from src.config import IOConfig                  # noqa: E402

FAILS: list = []
T = 40                                                    # 웨이퍼 타임포인트 수


def check(scn: str, inv: str, ok: bool, detail: str = ""):
    tag = f"{scn}/{inv}"
    if ok:
        print(f"  ✅ {tag}")
    else:
        print(f"  ❌ {tag}  {detail}")
        FAILS.append(tag)


# ── 결정적 가짜 모델 (OnnxForecastModel 인터페이스 최소 구현) ──
class FakeModel:
    def __init__(self, name: str, gain: float = 1.0):
        self.name = name
        self.gain = gain                                  # 모델마다 다른 출력(교체 검증용)
        self.io_cols = ["9000002"]
        self.input_sensor_cols = ["t_A", "p_B"]
        self.sensor_cols = ["t_A", "p_B"]
        self.n_steps_global = 3
        self.cfg = replace(IOConfig(), io_keep=())
        self.score_norm = "none"
        self.arch = {"virtual_sensor_mode": "off"}
        self.virtual_sensor_mode = "off"

    # 데이터 접근 — rec["sensors"] 를 이름 매핑으로
    def _cols(self, rec, wanted):
        arr = np.asarray(rec["sensors"], dtype=np.float64)
        idx = {str(c): i for i, c in enumerate(rec.get("sensor_cols") or [])}
        out = np.zeros((arr.shape[0], len(wanted)))
        for j, c in enumerate(wanted):
            if c in idx:
                out[:, j] = arr[:, idx[c]]
        return out

    def actual(self, rec):
        return self._cols(rec, self.sensor_cols)

    def _input_sensors(self, rec):
        return self._cols(rec, self.input_sensor_cols)

    def predict(self, rec):
        return self.actual(rec) * (0.99 * self.gain)      # 기대값(결정적)

    def error_level(self, rec):
        return np.abs(self.actual(rec) - self.predict(rec))

    def z_residual(self, rec):
        return self.error_level(rec)

    def assess(self, rec):
        lv = self.error_level(rec)
        return {"score": float(np.maximum(0.0, lv).sum()), "sensors": []}

    def n_flag_sensors(self, rec):
        return 0

    def predict_batch(self, recs, pause_check=None):
        return None


def make_wafer(folder: Path, stem: str, level: float = 1.0):
    """설비 유입 흉내 — 안정된(오래된 mtime) .csv 한 장."""
    rows = ["Time,ProcessStep,9000002,t_A,p_B"]
    for i in range(T):
        step = 0 if i < 5 else (1 if i < 25 else 2)
        rows.append(f"2026-09-04 00:{i:02d}:00,{step},1,"
                    f"{100 + level * (i % 7):.2f},{50 + level * (i % 5):.2f}")
    p = folder / f"{stem}.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    old = time.time() - 3600
    os.utime(p, (old, old))
    return p


# ── 화면 API 불변식 검사(I1·I4·I5) ──
def apis_ok(scn: str):
    try:
        ov = S.api_overview(rng="all", scale="day")
        sc = S.api_scatter(limit=100000)
        S.api_dispersion(rng="all", scale="day")
        for pt in sc.get("points", []):
            v = pt.get("ratio")
            assert v is not None and np.isfinite(float(v)) and float(v) >= 0, pt
        check(scn, "I1(API 무예외)", True)
        check(scn, "I4(점수 유한)", True)
    except Exception as e:                                # noqa: BLE001
        traceback.print_exc()
        check(scn, "I1/I4", False, f"{type(e).__name__}: {e}")
    # I5 — 모든 아카이브 웨이퍼의 개별 센서 점수 조회
    try:
        bad = []
        for r in S._read_history():
            try:
                out = S.api_archive_sensors_all(r["date"], r["wafer_id"])
                for it in out.get("sensors", []):
                    v = it.get("score")
                    if v is None or not np.isfinite(float(v)) or float(v) < 0:
                        bad.append((r["wafer_id"], it.get("sensor"), v))
            except Exception as e:                        # noqa: BLE001
                bad.append((r["wafer_id"], "EXC", f"{type(e).__name__}: {e}"))
        check(scn, "I5(센서점수 무에러)", not bad, str(bad[:3]))
    except Exception as e:                                # noqa: BLE001
        check(scn, "I5(센서점수 무에러)", False, str(e))


def index_matches_archive(scn: str):
    n_files = sum(1 for _ in S.ARCHIVE_DIR.glob("*/*.npz")) if S.ARCHIVE_DIR.exists() else 0
    n_rows = len(S._read_history())
    check(scn, "I6(인덱스=아카이브)", n_files == n_rows, f"파일 {n_files} vs 행 {n_rows}")


def main() -> int:
    # ── 격리 환경 구성 ──
    root = Path(tempfile.mkdtemp(prefix="sim_deploy_"))
    ch1 = root / "DSV1"; ch1.mkdir()
    S.ARCHIVE_DIR = root / "보관"; S.ARCHIVE_DIR.mkdir()
    S.DATA_DIR = root / "데이터"; S.DATA_DIR.mkdir()
    S.BACKUP_DIR = root / "백업"
    S.APP_CONFIG_FILE = root / "app_config.json"
    S.AUTOPILOT_STATE = S.DATA_DIR / "autopilot.json"
    S._archive_path.__globals__["ARCHIVE_DIR"] = S.ARCHIVE_DIR

    # 엔진: 가짜 모델 주입(로드 경로 생략) — threshold=1(정규화 동일)
    m1 = FakeModel("m1", 1.0)
    S.ENGINE.model = m1
    S.ENGINE.cfg = replace(IOConfig(), io_keep=())
    S.ENGINE.threshold = 1.0
    S.ENGINE.thresholds = {}
    S.ENGINE.threshold_mode = "global"
    S.ENGINE.error = None
    S.ENGINE.backend = "sim"
    S.ENGINE.art = root / "모델m1"
    (root / "모델m1").mkdir(); (root / "모델m1" / "sim.onnx").write_bytes(b"x")
    (root / "모델m2").mkdir(); (root / "모델m2" / "sim.onnx").write_bytes(b"x")
    S._probe_model_dir = lambda *a, **k: (True, "")

    ap = S.AutoPilot()
    S.AUTOPILOT = ap
    ap.enabled = True
    S._HIST_IDX.update(built=True, rows={}, by_stem={}, sorted=None)

    def scan():
        ap._check_model_change()                          # 실제 루프(_scan_once)와 동일 순서
        return ap._scan_folder(ch1, "dsv1") or 0

    hist_n = lambda: len(S._read_history())

    # ══ S1 기본 흐름 ══
    print("── S1 기본 흐름")
    for i in range(3):
        make_wafer(ch1, f"WUK370A_2026090400000{i}")
    n = scan()
    check("S1", "I2(전량 검사)", hist_n() == 3, f"검사 {n} · 이력 {hist_n()}")
    n2 = scan()
    check("S1", "I3(중복 없음)", hist_n() == 3 and (n2 or 0) == 0, f"재스캔 {n2}")
    apis_ok("S1"); index_matches_archive("S1")

    # ══ S2 순차 유입 ══
    print("── S2 순차 유입")
    for batch in range(3):
        for i in range(2):
            make_wafer(ch1, f"WUK370A_202609041{batch}{i:02d}")
        scan()
    check("S2", "I2", hist_n() == 9, f"이력 {hist_n()}")
    check("S2", "I3", scan() == 0)
    apis_ok("S2"); index_matches_archive("S2")

    # ══ S3 자정 롤오버 ══
    print("── S3 자정 롤오버")
    ap.day = "2026-09-03"                                 # 어제였다고 치고
    try:
        ap._check_rollover()
        check("S3", "I1(롤오버 무예외)", True)
    except Exception as e:                                # noqa: BLE001
        check("S3", "I1(롤오버 무예외)", False, str(e))
    make_wafer(ch1, "WUK370A_20260904Z001")
    scan()
    check("S3", "I2(롤오버 후 검사)", hist_n() == 10, f"이력 {hist_n()}")
    apis_ok("S3")

    # ══ S4 이력 전체 삭제 → 재검사 ══
    print("── S4 전체 삭제 → 자동 재검사")
    S.api_archive_clear()
    check("S4", "삭제 직후 이력 0", hist_n() == 0)
    make_wafer(ch1, "WUK370A_20260904NEW1")               # 삭제 직후 유입
    scan()
    check("S4", "I7(전량 재검사)", hist_n() == 11, f"이력 {hist_n()} (기존 10 + 신규 1)")
    apis_ok("S4"); index_matches_archive("S4")

    # ══ S5 모델 교체 ══
    print("── S5 모델 교체")
    before = {r["wafer_id"]: r["model"] for r in S._read_history()}
    S.ENGINE.model = FakeModel("m2", 1.5)
    S.ENGINE.art = root / "모델m2"
    scan()
    after = S._read_history()
    models = {r["model"] for r in after}
    check("S5", "I8(전량 재검사)", len(after) == 11 and len(models) == 1
          and list(models)[0] != list(set(before.values()))[0],
          f"행 {len(after)} · 모델 {models}")
    apis_ok("S5"); index_matches_archive("S5")

    # ══ S6 설정 변경 조합 ══
    print("── S6 설정 변경 조합")
    try:
        class _R:                                          # noqa: N801
            days = 30
        S.api_set_retention(_R())
        _R.days = 60
        S.api_set_retention(_R())
        S._save_display_rule({"bands": {"warn": 2.0, "crit": 3.0}})   # 밴드 변경(모드는 raw 고정)
        apis_ok("S6-band")
        ap.set_enabled(False); ap.set_enabled(True)
        check("S6", "I1(설정 조합 무예외)", True)
    except Exception as e:                                # noqa: BLE001
        traceback.print_exc()
        check("S6", "I1(설정 조합 무예외)", False, str(e))
    apis_ok("S6")

    # ══ S7 off 중 유입 → on 소급 ══
    print("── S7 자동검사 off → on 소급")
    ap.set_enabled(False)
    make_wafer(ch1, "WUK370A_20260904OFF1")
    # off 면 스캔 자체를 안 부른다(루프 게이트) — 여기선 게이트만 확인
    ap.set_enabled(True)
    scan()
    check("S7", "I2(소급 검사)", hist_n() == 12, f"이력 {hist_n()}")
    apis_ok("S7")

    # ══ S8 난동 조합 ══
    print("── S8 난동 조합(삭제→교체→유입→삭제→off→유입→on→교체)")
    try:
        S.api_archive_clear()
        S.ENGINE.model = FakeModel("m3", 0.7); S.ENGINE.art = root / "모델m1"
        make_wafer(ch1, "WUK370A_20260904R1"); scan()
        S.api_archive_clear()
        ap.set_enabled(False)
        make_wafer(ch1, "WUK370A_20260904R2")
        ap.set_enabled(True)
        S.ENGINE.model = FakeModel("m4", 1.2); S.ENGINE.art = root / "모델m2"
        scan()
        check("S8", "I1(난동 무예외)", True)
        check("S8", "I2/I7/I8(전량 최신 모델)", hist_n() == 14
              and {r["model"] for r in S._read_history()} == {S._model_id()},
              f"이력 {hist_n()} · 모델 {set(r['model'] for r in S._read_history())}")
    except Exception as e:                                # noqa: BLE001
        traceback.print_exc()
        check("S8", "I1(난동 무예외)", False, str(e))
    apis_ok("S8"); index_matches_archive("S8")

    # ══ S9 재시작(인덱스 재구축) ══
    print("── S9 재시작")
    snap = {(r["date"], r["wafer_id"]): r["score"] for r in S._read_history()}
    S._HIST_IDX.update(built=False, rows={}, by_stem={}, sorted=None)
    rows = S._read_history()                               # 재구축
    snap2 = {(r["date"], r["wafer_id"]): r["score"] for r in rows}
    check("S9", "I9(재구축 동일)", snap == snap2,
          f"이전 {len(snap)} vs 재구축 {len(snap2)}")
    apis_ok("S9")

    # ══ S10 손상 아카이브 혼입 ══
    print("── S10 손상 파일 혼입")
    bad = S.ARCHIVE_DIR / "2026-09-04" / "BROKEN.npz"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not-a-npz")
    S._HIST_IDX.update(built=False, rows={}, by_stem={}, sorted=None)
    try:
        rows = S._read_history()
        check("S10", "I10(손상 제외·무예외)", len(rows) == len(snap2), f"행 {len(rows)}")
    except Exception as e:                                # noqa: BLE001
        check("S10", "I10(손상 제외·무예외)", False, str(e))
    apis_ok("S10")

    shutil.rmtree(root, ignore_errors=True)
    print(f"\n시나리오 10종 · 실패 {len(FAILS)}건")
    if FAILS:
        print("실패:", ", ".join(FAILS))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
