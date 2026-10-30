# -*- coding: utf-8 -*-
"""레시피별 센서 평균 파형 리포트 (테스트 도구 — 앱과 별개 실행).

목적: 과거값 오염이 복원에 전파되는 문제를 다루기 위한 기준 파형(golden waveform) 추출.
데이터셋 폴더를 모델로 판정해 '확실한 양품'(정규화 점수 ≤ ok-max, 기본 0.8)만 골라,
레시피별 · 센서별 평균 파형(±1σ 밴드)을 단일 HTML 리포트로 만든다.

사용 (앱빌드 폴더에서 — src/ 폴더 옆에 두고):
    python test_avg_waveform.py --data D:\데이터폴더 [--model D:\모델폴더]
                                [--ok-max 0.8] [--out 평균파형_리포트.html]
인자 없이 실행하면 대화형으로 묻는다.

- 판정·파싱은 배포 앱과 동일 코드(src.app.server)를 그대로 사용 — 별도 구현 없음.
- 웨이퍼마다 길이가 달라 '레시피 중앙값 길이'로 선형 리샘플 후 평균한다.
- 외부 라이브러리 없는 인라인 SVG — 폐쇄망에서 브라우저로 바로 열림.
"""
from __future__ import annotations

import argparse

TOOL_VER = "v185"                                  # 실행 중 판 확인용(콘솔·내보내기 표기)
import datetime as dt
import re
import html as _html
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("IO_NO_BROWSER", "1")
os.environ.setdefault("IO_KEEP_ALIVE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ★ 윈도우 콘솔(cp949) 대응 — 판정 코드가 진단 로그에 cp949 에 없는 문자(⚠ 등)를
#   출력하면 UnicodeEncodeError 로 죽는다. 표준출력을 UTF-8(표현 불가 문자는 ?)로 강제.
#   (chcp 65001 콘솔에서는 한글도 정상 표시, cp949 콘솔에서는 일부 글자만 깨져 보이고 죽지 않음)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                     # noqa: BLE001
        pass

import numpy as np


# ── 집계 ────────────────────────────────────────────────────────────
def _resample(col: np.ndarray, n: int) -> np.ndarray:
    """길이 T 파형을 길이 n 으로 선형 보간(모양 보존)."""
    t = len(col)
    if t == n:
        return col.astype(np.float64)
    x_new = np.linspace(0.0, 1.0, n)
    x_old = np.linspace(0.0, 1.0, t)
    return np.interp(x_new, x_old, col.astype(np.float64))


def _parse_multi(v):
    """'dsv1,dsv2' / '29a,521' / '5, 2026-07' → 소문자 토큰 리스트(빈값=필터 없음)."""
    if not v:
        return []
    if isinstance(v, (list, tuple)):                   # JSON 배열 입력(골든 비교 탭)
        v = ",".join(str(x9) for x9 in v)
    out = []
    for t in str(v).replace("월", "").split(","):
        t = t.strip().lower()
        if not t:
            continue
        m9 = re.match(r"^(\d+)\s*[-~]\s*(\d+)$", t)   # "1-8" / "3~5" 범위 확장
        if m9 and int(m9.group(1)) <= int(m9.group(2)) <= int(m9.group(1)) + 60:
            out += [str(k9) for k9 in range(int(m9.group(1)), int(m9.group(2)) + 1)]
        else:
            out.append(t)
    return out


def split_folders(v):
    """폴더 입력 문자열 → 경로 목록. 구분자: 세미콜론/줄바꿈/쉼표. 데이터가 여러 곳에
    나눠 담긴 경우 지원 — 존재하지 않는 경로는 호출부에서 검사."""
    if isinstance(v, (list, tuple)):
        parts = [str(x) for x in v]
    else:
        parts = re.split(r"[;\n,]+", str(v or ""))
    out = []
    for t in parts:
        t = t.strip().strip('"')
        if t and t not in out:
            out.append(t)
    return [Path(t) for t in out]


def _meta_of(S, st):
    """stem → {eqp, dsv, recipe} — select_stems 와 동일 규칙(태깅 후 dict 조회 무비용)."""
    eq = st.split("_", 1)[0].lower() if "_" in st else st.lower()
    ch = ""
    try:
        ch = str((S._FILE_META.get(st) or {}).get("chamber")
                 or S._chamber(st) or "").lower()
    except Exception:                                  # noqa: BLE001
        pass
    try:
        rc = str(S._recipe_display(S._RECIPE_TAGS.get(st, "")) or "").lower()
    except Exception:                                  # noqa: BLE001
        rc = ""
    return {"eqp": eq, "dsv": ch, "recipe": rc}


def select_stems_multi(S, folders, dsv=None, recipes=None, months=None, eqp=None):
    """여러 폴더에 select_stems 적용. 반환: (per=[(folder, sel또는None)], n_all, n_sel, avail합)."""
    per, n_all, n_sel = [], 0, 0
    avail = {}
    for fol in folders:
        selx, na, ns, av = select_stems(S, fol, dsv, recipes, months, eqp)
        per.append((fol, selx))
        n_all += na; n_sel += ns
        for k9, d9 in (av or {}).items():
            tgt = avail.setdefault(k9, {})
            for v9, c9 in d9.items():
                tgt[v9] = tgt.get(v9, 0) + c9
    return per, n_all, n_sel, avail


def load_aligned_multi(S, per):
    """per=[(folder, sel또는None)] → recs 이어붙이기 + rec별 출처 폴더 + 첫 커버리지."""
    recs, src, cov = [], [], None
    for fol, selx in per:
        if selx is not None and not selx:
            continue
        r2, _i9, _s9, c9 = S._load_aligned(fol, stems=selx)
        recs += r2
        src += [str(fol)] * len(r2)
        cov = cov or c9
    return recs, src, cov


def select_stems(S, folder: Path, dsv=None, recipes=None, months=None, eqp=None):
    """필터에 맞는 스템만 선별(판정 전 — 필요한 파일만 읽어 속도 확보).

    dsv:     ['dsv1','1',...]  → 챔버 일치(숫자만 주면 dsvN 으로 해석)
    recipes: ['29a','521',...] → 레시피 표시명 부분일치(하나라도 맞으면)
    months:  ['2026-05','5']   → 끝시각의 YYYY-MM 일치 또는 월 번호 일치
    eqp:     ['WUK370A','A',...] → 설비명 = 파일명 첫 '_' 앞 토큰, 부분일치
             (짧게 'A' 만 줘도 WUK370A 매칭 — 대소문자 무시)
    반환: (선별 스템 리스트 또는 None(필터 없음), 전체 수, 선별 수)
    """
    dsv = [("dsv" + t if t.isdigit() else t) for t in _parse_multi(dsv)]
    recipes = _parse_multi(recipes)
    months = _parse_multi(months)
    eqp = [t.lower() for t in _parse_multi(eqp)]
    try:
        S._convert_logs_in(folder)                    # 챔버(C2)·레시피(J2)·끝시각 태깅
    except Exception:                                 # noqa: BLE001
        pass
    try:
        from io_recon_tcn.data import _data_files, _resolve_existing_dir_ci
        files = _data_files(_resolve_existing_dir_ci(folder))
    except Exception:                                 # noqa: BLE001
        files = [p2 for p2 in Path(folder).iterdir()
                 if p2.suffix.lower() in (".csv", ".log", ".tsv")]
    stems_all = [f.stem for f in files]
    if not (dsv or recipes or months or eqp):
        return None, len(stems_all), len(stems_all), {}
    sel = []
    avail = {"dsv": {}, "recipe": {}, "month": {}, "eqp": {}}   # 값 → 개수(안내용)
    for st in stems_all:
        eq = st.split("_", 1)[0].lower() if "_" in st else st.lower()
        ch = ""
        try:
            ch = str((S._FILE_META.get(st) or {}).get("chamber")
                     or S._chamber(st) or "").lower()
        except Exception:                             # noqa: BLE001
            pass
        try:
            rc = str(S._recipe_display(S._RECIPE_TAGS.get(st, "")) or "").lower()
        except Exception:                             # noqa: BLE001
            rc = ""
        try:
            day = S._end_ts_and_date(st)[1]           # YYYY-MM-DD
        except Exception:                             # noqa: BLE001
            day = ""
        ym = day[:7]
        avail["dsv"][ch or "(없음)"] = avail["dsv"].get(ch or "(없음)", 0) + 1
        avail["recipe"][rc or "(없음)"] = avail["recipe"].get(rc or "(없음)", 0) + 1
        avail["month"][ym or "(없음)"] = avail["month"].get(ym or "(없음)", 0) + 1
        avail["eqp"][eq or "(없음)"] = avail["eqp"].get(eq or "(없음)", 0) + 1
        if eqp and not any(t in eq for t in eqp):
            continue
        if dsv and ch not in dsv:
            continue
        if recipes and not any(t in rc for t in recipes):
            continue
        if months and not any(t == ym or t == day[5:7].lstrip("0")
                              or t.zfill(2) == day[5:7] for t in months):
            continue
        sel.append(st)
    return sel, len(stems_all), len(sel), avail


def _avail_text(avail: dict) -> str:
    """'쓸 수 있는 값' 요약 문자열(필터 미스매치 자가진단용)."""
    def top(d, n=8):
        it = sorted(d.items(), key=lambda x: -x[1])[:n]
        return ", ".join(f"{k}({v}장)" for k, v in it) + (" …" if len(d) > n else "")
    return (f"데이터에 실제로 있는 값 — 설비: [{top(avail.get('eqp', {}))}] · "
            f"DSV: [{top(avail.get('dsv', {}))}] · "
            f"유형: [{top(avail.get('recipe', {}))}] · "
            f"월: [{top(avail.get('month', {}))}]")


def collect(S, folder: Path, ok_max: float, dsv=None, recipes=None, months=None, eqp=None):
    """폴더 전체 판정 → OK(≤ok_max)만 레시피별로 모아 (평균, σ, N) 집계.

    반환: (groups, stats)
      groups[recipe] = {"sensors": [이름...], "n": N, "len": L,
                        "mean": (L,S), "std": (L,S)}
      stats = 판정 카운트(전체/양품/애매/불량/보류)
    """
    folders = split_folders(folder)
    per, n_all, n_sel, avail = select_stems_multi(S, folders, dsv, recipes, months, eqp)
    filtered = any(x[1] is not None for x in per)
    if filtered:
        print(f"[filter] 골든 선정: 전체 {n_all}장 → 필터 통과 {n_sel}장 "
              f"(설비={eqp or '전체'} · DSV={dsv or '전체'} · 유형={recipes or '전체'} · 월={months or '전체'})", flush=True)
        if n_sel == 0:
            raise SystemExit("필터를 통과한 파일이 없습니다.\n  " + _avail_text(avail)
                             + "\n  → 입력값을 위 실제 값에 맞춰 주세요")
    if len(folders) > 1:
        print(f"[load] 데이터 폴더 {len(folders)}곳 병합", flush=True)
    print(f"[load] 파일 읽는 중 {n_sel if filtered else '전체'}장 — 수 분 걸릴 수 있습니다", flush=True)
    recs, _src, _cov = load_aligned_multi(S, per)
    print(f"[load] 읽기 완료 {len(recs)}장 → 추론 시작", flush=True)
    if not recs:
        raise SystemExit(f"읽을 수 있는 웨이퍼가 없습니다: {folder} "
                         "(파싱 실패/최소 길이 미달 — 콘솔의 [load] skip 로그 확인)")
    # ★ 모델-데이터 정합 상시 리포트 — '모델 파트 문제'를 즉시 가려낸다
    print(f"[cov] 센서 커버리지 {_cov['sensor_coverage']*100:.0f}% "
          f"({_cov['n_sensor_expected']-_cov['n_sensor_missing']}/{_cov['n_sensor_expected']}) · "
          f"IO {_cov['io_coverage']*100:.0f}% "
          f"({_cov['n_io_expected']-_cov['n_io_missing']}/{_cov['n_io_expected']})", flush=True)
    if _cov.get("missing_sensors"):
        print(f"[cov] 모델은 기대하지만 데이터에 없는 센서: {_cov['missing_sensors']}", flush=True)
    if _cov.get("missing_io"):
        print(f"[cov] 없는 IO: {_cov['missing_io']}", flush=True)
    if _cov["sensor_coverage"] < 0.8:
        print("[cov] ⚠ 커버리지 80% 미만 — 이 모델은 다른 데이터 구성으로 학습된 것일 수 있습니다"
              "(점수·기대값 신뢰 불가)", flush=True)
    stats = {"total": len(recs), "ok": 0, "gray": 0, "ng": 0, "na": 0}
    # ★ 배치 순전파(앱과 동일 경로): 웨이퍼별 단건 추론 대신 청크로 묶어 GPU/CPU 가속.
    #   5만+ 장 규모 대응 — 1000장 단위로 캐시를 채우며 진행 로그를 찍는다.
    CH = 1000
    for k0 in range(0, len(recs), CH):
        S._prefill_batch(recs[k0:k0 + CH])
        if len(recs) > CH:
            print(f"[batch] 추론 {min(k0 + CH, len(recs))}/{len(recs)}", flush=True)
    by_recipe: dict = {}
    for rec in recs:
        r = S._result_for(rec)
        ratio = r.get("ratio")
        if ratio is None or not np.isfinite(float(ratio)):
            stats["na"] += 1
            continue
        ratio = float(ratio)
        if ratio > 1.0:
            stats["ng"] += 1
            continue
        if ratio > ok_max:                            # 임계 이하지만 '확실한 양품'은 아님
            stats["gray"] += 1
            continue
        stats["ok"] += 1
        recipe = S._recipe_display(r.get("recipe") or "") or "unknown"
        arr = np.asarray(rec["sensors"], dtype=np.float64)          # (T, S)
        cols = [str(c) for c in (rec.get("sensor_cols") or [])]
        g = by_recipe.setdefault(recipe, {"mats": [], "cols": cols, "lens": []})
        if cols != g["cols"]:                          # 열 구성이 다른 파일은 이름 교집합으로 정렬
            keep = [c for c in g["cols"] if c in cols]
            idx = [cols.index(c) for c in keep]
            arr = arr[:, idx]
            gidx = [g["cols"].index(c) for c in keep]
            g["mats"] = [m[:, gidx] for m in g["mats"]]
            g["cols"] = keep
        g["mats"].append(arr.astype(np.float32))      # 보관은 절반 메모리(계산은 float64)
        g["lens"].append(arr.shape[0])

    groups = {}
    for recipe, g in by_recipe.items():
        if not g["mats"]:
            continue
        L = int(np.median(g["lens"]))                 # 공통 길이 = 그 레시피 중앙값
        # ★ 대량(수천 장) 안전: (N,L,S) 전체 스택 대신 합/제곱합 누적으로 평균·σ 계산
        #   — 7000장 × 긴 공정에서도 메모리가 L×S 두 장 분량으로 고정된다. 결과는 동일(ddof=0).
        n = len(g["mats"])
        acc = np.zeros((L, len(g["cols"])), dtype=np.float64)
        acc2 = np.zeros_like(acc)
        for m in g["mats"]:
            rs = np.column_stack([_resample(m[:, j], L) for j in range(m.shape[1])])
            acc += rs
            acc2 += rs * rs
        mean = acc / n
        std = np.sqrt(np.maximum(acc2 / n - mean * mean, 0.0))
        # ★ 센서 순서 = 설치 순서(설비 토폴로지 sensors_order.json 의 graph rank)
        try:
            from io_recon_tcn.sensor_graph import graph_sensor_rank
            rk = graph_sensor_rank(g["cols"], S.ENGINE.cfg)
            order = sorted(range(len(g["cols"])), key=lambda j: rk.get(str(g["cols"][j]), 10**6))
        except Exception:                             # noqa: BLE001 — 그래프 없으면 원래 순서
            order = list(range(len(g["cols"])))
        groups[recipe] = {"sensors": [g["cols"][j] for j in order], "n": n, "len": L,
                          "mean": mean[:, order], "std": std[:, order]}
    return groups, stats


# ── SVG 렌더 (외부 라이브러리 없음 — 폐쇄망) ────────────────────────
def _svg_chart(name: str, mean: np.ndarray, std: np.ndarray, n: int) -> str:
    W, H, pl, pr, pt, pb = 560, 210, 52, 10, 26, 24
    iw, ih = W - pl - pr, H - pt - pb
    lo = float(np.min(mean - std)); hi = float(np.max(mean + std))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    L = len(mean)
    x = lambda i: pl + iw * (i / max(1, L - 1))
    y = lambda v: pt + ih * (1 - (v - lo) / (hi - lo))
    band = ("M" + " L".join(f"{x(i):.1f} {y(mean[i]+std[i]):.1f}" for i in range(L))
            + " L" + " L".join(f"{x(i):.1f} {y(mean[i]-std[i]):.1f}" for i in range(L - 1, -1, -1)) + " Z")
    line = "M" + " L".join(f"{x(i):.1f} {y(mean[i]):.1f}" for i in range(L))
    grid = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        gy = y(v)
        grid += (f'<line x1="{pl}" y1="{gy:.1f}" x2="{W-pr}" y2="{gy:.1f}" stroke="#e5e9f0" stroke-width="1"/>'
                 f'<text x="{pl-6}" y="{gy+4:.1f}" text-anchor="end" font-size="10" fill="#8a94a3">{v:.3g}</text>')
    # ★ 호버용 수치 내장(반올림 6자리) — 마우스 위치의 타임포인트 값(평균·σ)을 툴팁으로.
    import json as _json
    data = _json.dumps({"m": [round(float(v), 6) for v in mean],
                        "s": [round(float(v), 6) for v in std],
                        "pl": pl, "pt": pt, "iw": iw, "ih": ih,
                        "lo": round(lo, 6), "hi": round(hi, 6), "L": L},
                       separators=(",", ":"))
    return (f'<div class="card"><div class="t">{_html.escape(name)}'
            f'<span class="n">N={n} · L={L}</span></div>'
            f'<div class="chwrap" style="position:relative">'
            f'<svg class="ch" width="{W}" height="{H}" data-ch=\'{data}\'>{grid}'
            f'<path d="{band}" fill="#2b7fc4" fill-opacity="0.14"/>'
            f'<path d="{line}" fill="none" stroke="#2456a6" stroke-width="1.8"/>'
            f'<line class="hvx" x1="0" y1="{pt}" x2="0" y2="{pt+ih}" stroke="#c0392b" stroke-width="1" stroke-dasharray="3 3" style="display:none"/>'
            f'<circle class="hvc" r="3.4" fill="#c0392b" stroke="#fff" stroke-width="1.2" style="display:none"/>'
            f'<text x="{pl}" y="{H-6}" font-size="10" fill="#8a94a3">0</text>'
            f'<text x="{W-pr}" y="{H-6}" text-anchor="end" font-size="10" fill="#8a94a3">타임포인트 {L}</text>'
            f'</svg><div class="hvtip" style="display:none"></div></div></div>')


def render_html(groups: dict, stats: dict, folder: Path, ok_max: float, model_name: str,
                wafers=None, wafer_folder=None, wafer_total=0, embed=None) -> str:
    """embed=(metas, payload_dict) 이면 서버 없이 여는 정적 내장 리포트(600장 단위 저장용)."""
    head = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>레시피별 센서 평균 파형</title><style>
body{{font-family:'Malgun Gothic',sans-serif;margin:24px;color:#26303b;background:#f6f8fa}}
h1{{font-size:20px}} h2{{font-size:16px;margin:26px 0 10px;border-bottom:2px solid #2456a6;padding-bottom:4px}}
.meta{{font-size:12.5px;color:#5b6675;margin-bottom:6px}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}}
.card{{background:#fff;border:1px solid #dde3ea;border-radius:8px;padding:8px 10px}}
.card .t{{font-weight:700;font-size:13px;margin-bottom:4px}}
.card .n{{font-weight:400;color:#8a94a3;font-size:11.5px;margin-left:8px}}
.legend{{font-size:12px;color:#5b6675;margin:4px 0 14px}}
.tabs{{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0 4px;border-bottom:2px solid #2456a6;padding-bottom:0}}
.tabs button{{border:1px solid #dde3ea;border-bottom:none;background:#eef2f7;color:#3a4654;
 padding:7px 16px;border-radius:8px 8px 0 0;font-size:13.5px;cursor:pointer;font-family:inherit}}
.tabs button.on{{background:#2456a6;color:#fff;border-color:#2456a6;font-weight:700}}
.tabpane{{display:none}} .tabpane.on{{display:block}}
.hvtip{{position:absolute;pointer-events:none;background:#26303b;color:#fff;font-size:11.5px;
 padding:4px 8px;border-radius:6px;white-space:nowrap;z-index:5;line-height:1.5}}
.wstrip{{display:flex;flex-wrap:wrap;gap:6px;max-height:170px;overflow:auto;padding:8px;
 background:#fff;border:1px solid #dde3ea;border-radius:8px;margin-bottom:10px}}
.wrow{{padding:5px 10px;border:1px solid #dde3ea;border-radius:16px;cursor:pointer;font-size:12px;
 display:flex;gap:7px;align-items:center;font-family:Consolas,monospace;white-space:nowrap;background:#f8fafc}}
.wrow:hover{{background:#f0f5fb}} .wrow.sel{{background:#2456a6;border-color:#2456a6}}
.wrow.sel span{{color:#fff!important}}
.wrow .sc{{font-weight:700}}
.wrow .vd{{font-size:11px;padding:0 6px;border-radius:9px;background:#eef2f7;color:#3a4654}}
.wrow .vd.ng{{background:#fdecea;color:#c0392b}}
.ovgrid{{display:flex;flex-wrap:wrap;gap:12px}}
.ovhead{{font-size:13px;margin:0 0 8px;color:#3a4654}}
.lgd{{display:flex;gap:16px;flex-wrap:wrap;align-items:center;font-size:12px;color:#3a4654;margin:2px 0 10px}}
.lgd svg{{vertical-align:-2px}}
.fline{{display:flex;gap:8px;align-items:flex-start;font-size:12.5px;margin:3px 0}}
.fline b{{width:34px;color:#3a4654;padding-top:3px;flex:none}}
.fline>span{{display:flex;flex-wrap:wrap;gap:4px;max-height:96px;overflow:auto}}
.fline label{{display:inline-flex;gap:3px;align-items:center;cursor:pointer;background:#fff;border:1px solid #dde3ea;
 border-radius:12px;padding:2px 9px;font-size:12px;white-space:nowrap}}
</style></head><body>
<h1>레시피별 센서 평균 파형 (양품 기준선)</h1>
<div class="meta">데이터: {_html.escape(str(folder))} · 모델: {_html.escape(model_name)}
 · 생성 {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<div class="meta">판정: 전체 {stats['total']}장 → <b>집계 사용(양품, 점수≤{ok_max:g}) {stats['ok']}장</b>
 · 애매({ok_max:g}&lt;점수≤1) {stats['gray']}장 제외 · 불량(&gt;1) {stats['ng']}장 제외 · 판정보류 {stats['na']}장 제외</div>
<div class="lgd"><span><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="#2456a6" stroke-width="2.2"/></svg> 평균 파형</span>
<span><svg width="26" height="12"><rect x="1" y="1" width="24" height="10" fill="#2b7fc4" fill-opacity="0.18"/></svg> ±1σ(정상 산포 폭)</span>
<span>가로축 = 타임포인트(레시피 중앙값 길이로 리샘플)</span></div>"""
    # ── 레시피 = 탭 + (선택) 웨이퍼 분석 탭. 레시피 1종·분석 없음이면 탭바 생략 ──
    recipes = sorted(groups)
    has_waf = bool(wafers)
    body = ""
    if len(recipes) > 1 or has_waf:
        body += '<div class="tabs">'
        for i, recipe in enumerate(recipes):
            g = groups[recipe]
            body += (f'<button type="button" class="{"on" if i == 0 else ""}" '
                     f'onclick="showTab({i})">{_html.escape(recipe)} ({g["n"]}장)</button>')
        if has_waf:
            body += (f'<button type="button" onclick="showTab({len(recipes)})">웨이퍼 분석</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 1})">골든 크로스</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 2})">추세 스캔</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 3})">주요 패턴</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 4})">스텝 차이</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 5})">데이터 시각화</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 6})">패턴 인식</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 7})">분석자료</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 8})">데이터 뷰어</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 9})">패턴 탐사</button>')
            body += (f'<button type="button" onclick="showTab({len(recipes) + 10})">골든 비교</button>')
        body += "</div>"
    for i, recipe in enumerate(recipes):
        g = groups[recipe]
        body += f'<div class="tabpane{" on" if i == 0 else ""}" id="pane{i}">'
        body += (f'<h2>{_html.escape(recipe)} <span class="n" style="font-size:12px;color:#8a94a3">'
                 f'양품 {g["n"]}장 · 센서 {len(g["sensors"])}개</span></h2><div class="grid">')
        for j, sname in enumerate(g["sensors"]):
            body += _svg_chart(sname, g["mean"][:, j], g["std"][:, j], g["n"])
        body += "</div></div>"
    if has_waf:                                        # 서버 모드: 분석 탭 UI(데이터는 API 로)
        k = len(recipes)
        body += (f'<div class="tabpane" id="pane{k}">'
                 f'<h2>웨이퍼 분석</h2>'
                 f'<div style="display:flex;gap:8px;align-items:center;margin:0 0 10px;flex-wrap:wrap">'
                 f'<input id="wfolder" placeholder="웨이퍼 데이터 폴더 경로 (여러 곳이면 ; 로 구분)" spellcheck="false" '
                 f'title="예: /data/7월 ; /data/8월 — 세미콜론/쉼표/줄바꿈으로 여러 폴더" '
                 f'style="flex:1;min-width:260px;font-family:Consolas,monospace;font-size:13px;'
                 f'border:1px solid #dde3ea;border-radius:8px;padding:8px 10px">'
                 f'<input id="wfeq" placeholder="설비 (예: WUK370A 또는 A,B,2)" spellcheck="false" '
                 f'title="파일명 앞 설비명 부분일치 · 쉼표로 여러 개 · 비우면 전체" '
                 f'style="width:150px;font-size:12.5px;border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
                 f'<input id="wfdsv" placeholder="DSV (예: 1,2,3)" spellcheck="false" title="비우면 전체 · 쉼표로 여러 개" '
                 f'style="width:110px;font-size:12.5px;border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
                 f'<input id="wfrc" placeholder="유형 (예: 29a,521)" spellcheck="false" title="부분일치 · 쉼표로 여러 개" '
                 f'style="width:130px;font-size:12.5px;border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
                 f'<input id="wfmo" placeholder="월 (예: 5,6,7)" spellcheck="false" title="5 = 모든 해 5월 · 2026-05 형식도 가능 · 쉼표로 여러 개" '
                 f'style="width:110px;font-size:12.5px;border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
                 f'<button id="wscan" onclick="__scan()" style="padding:8px 18px;border:none;border-radius:8px;'
                 f'background:#2456a6;color:#fff;font-size:13.5px;cursor:pointer">불러오기</button>'
                 f'<button id="wmon" onclick="__monthly()" style="padding:8px 14px;border:1px solid #1f6f8b;'
                 f'border-radius:8px;background:#fff;color:#1f6f8b;font-size:13px;cursor:pointer;display:none">'
                 f'월간 원인 분석</button>'
                 f'<button id="wwrn" onclick="__warning()" style="padding:8px 14px;border:1px solid #c0392b;'
                 f'border-radius:8px;background:#fff;color:#c0392b;font-size:13px;cursor:pointer;display:none">'
                 f'주간 조기경보</button>'
                 f'<button id="wfoc" onclick="__focusForm()" style="padding:8px 14px;border:1px solid #7b1fa2;'
                 f'border-radius:8px;background:#fff;color:#7b1fa2;font-size:13px;cursor:pointer;display:none">'
                 f'집중 분석 (보고 전 구간)</button>'
                 f'<button id="wgc" onclick="__gcmpForm()" style="padding:8px 14px;border:1px solid #00695c;'
                 f'border-radius:8px;background:#fff;color:#00695c;font-size:13px;cursor:pointer;display:none">'
                 f'골든 검증 (DSV 비교)</button>'
                 f'<button id="wrsh" onclick="__research()" style="padding:8px 14px;border:1px solid #b3541e;'
                 f'border-radius:8px;background:#fff;color:#b3541e;font-size:13px;cursor:pointer;display:none">'
                 f'딥리서치 시작</button>'
                 f'<button id="wrst" onclick="__research_status()" style="padding:8px 10px;border:1px solid #8a94a3;'
                 f'border-radius:8px;background:#fff;color:#5b6675;font-size:12px;cursor:pointer;display:none">'
                 f'리서치 상태/보고</button>'
                 f'<button id="wcsv" onclick="__dailycsv()" style="padding:8px 14px;border:1px solid #6a4fa3;'
                 f'border-radius:8px;background:#fff;color:#6a4fa3;font-size:13px;cursor:pointer;display:none">'
                 f'일별 판정 CSV</button>'
                 f'<button id="wtrd" onclick="__trend()" style="padding:8px 14px;border:1px solid #2e7d32;'
                 f'border-radius:8px;background:#fff;color:#2e7d32;font-size:13px;cursor:pointer;display:none">'
                 f'기간 추세 분석</button>'
                 f'<button id="wexp" onclick="__export()" style="padding:8px 14px;border:1px solid #2456a6;'
                 f'border-radius:8px;background:#fff;color:#2456a6;font-size:13px;cursor:pointer;display:none">'
                 f'리포트 저장 (600장/파일)</button>'
                 f'<span id="wmsg" style="font-size:12.5px;color:#5b6675"></span></div>'
                 f'<div style="display:flex;gap:8px;align-items:center;margin:0 0 8px;flex-wrap:wrap;font-size:12.5px">'
                 f'저장 골든 겹치기: <input id="wgst" placeholder="저장소 폴더 경로" spellcheck="false" '
                 f'style="width:280px;font-family:Consolas,monospace;font-size:12.5px;'
                 f'border:1px solid #dde3ea;border-radius:8px;padding:6px 8px">'
                 f'<button onclick="__gslist(\'wgst\',\'wgsbox\')" style="padding:6px 14px;border:1px solid #6a4fa3;'
                 f'border-radius:8px;background:#fff;color:#6a4fa3;font-size:12.5px;cursor:pointer">저장 골든 목록</button>'
                 f'<span id="wgsbox"></span></div>'
                 f'<div id="wfilters" style="display:none;margin:0 0 8px">'
                 f'<div class="fline"><b>유형</b> <span id="frc"></span></div>'
                 f'<div class="fline"><b>날짜</b> <span id="fdt"></span></div>'
                 f'<div class="fline"><b>판정</b> <span id="fvd"></span> '
                 f'<span id="fcnt" style="color:#8a94a3;font-size:12px"></span></div>'
                 f'<div class="fline"><b>센서</b> <span id="fsn"></span></div></div>'
                 f'<div class="wstrip" id="wlist" style="display:none"></div>'
                 f'<div class="lgd">'
                 f'<span><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="#2456a6" stroke-width="2.2"/></svg> 기준 평균(양품)</span>'
                 f'<span><svg width="26" height="12"><rect x="1" y="1" width="24" height="10" fill="#2b7fc4" fill-opacity="0.18"/></svg> 정상 산포(±1σ)</span>'
                 f'<span><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="#d64545" stroke-width="2.2"/></svg> 실제 값</span>'
                 f'<span><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="#e08a00" stroke-width="2.2" stroke-dasharray="5 3"/></svg> 모델 기대값(복원)</span>'
                 f'</div>'
                 f'<div class="ovgrid" id="ovgrid"><div style="color:#8a94a3;font-size:13px;padding:20px">'
                 f'폴더를 불러온 뒤 웨이퍼를 선택하세요</div></div></div>')
        import json as _json
        gold_js = {r: {"sensors": g["sensors"], "L": g["len"],
                       "m": [[round(float(v), 5) for v in g["mean"][:, j]] for j in range(len(g["sensors"]))],
                       "s": [[round(float(v), 5) for v in g["std"][:, j]] for j in range(len(g["sensors"]))]}
                   for r, g in groups.items()}
        kx = len(recipes) + 1
        body += (
            f'<div class="tabpane" id="pane{kx}">'
            f'<h2>골든 크로스 <span class="n" style="font-size:12px;color:#8a94a3">'
            f'판정 없이 기간 평균 파형의 이동을 본다</span></h2>'
            f'<div style="display:flex;gap:8px;align-items:center;margin:0 0 10px;flex-wrap:wrap">'
            f'<input id="gxf" placeholder="웨이퍼 데이터 폴더 (여러 곳이면 ; 구분)" spellcheck="false" '
            f'style="flex:1;min-width:260px;font-family:Consolas,monospace;font-size:13px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 10px">'
            f'<input id="gxe" placeholder="설비 (예: A,B)" style="width:110px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
            f'<input id="gxd" placeholder="DSV" style="width:70px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
            f'<input id="gxr" placeholder="유형" style="width:90px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
            f'<input id="gxm" placeholder="월 (예: 6,7)" style="width:90px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
            f'<input id="gxb" value="5" title="묶음 일수" style="width:44px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">일 단위'
            f'<button id="gxgo" onclick="__gcross()" style="padding:8px 18px;border:none;border-radius:8px;'
            f'background:#2456a6;color:#fff;font-size:13.5px;cursor:pointer">그리기</button>'
            f'<span id="gxmsg" style="font-size:12.5px;color:#5b6675"></span></div>'
            f'<div style="display:flex;gap:8px;align-items:center;margin:0 0 10px;flex-wrap:wrap;font-size:12.5px">'
            f'저장 골든: <input id="gxst" placeholder="저장소 폴더 경로 (저장·불러오기 공용)" spellcheck="false" '
            f'style="width:300px;font-family:Consolas,monospace;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:6px 8px">'
            f'<button onclick="__gslist(\'gxst\',\'gxsbox\')" style="padding:6px 14px;border:1px solid #6a4fa3;'
            f'border-radius:8px;background:#fff;color:#6a4fa3;font-size:12.5px;cursor:pointer">저장 골든 목록</button>'
            f'<span id="gxsbox"></span></div>'
            f'<div id="gxout"></div></div>')
        ky = len(recipes) + 2
        body += (
            f'<div class="tabpane" id="pane{ky}">'
            f'<h2>추세 스캔 <span class="n" style="font-size:12px;color:#8a94a3">'
            f'이상 기준 파형 대비 하루씩 누적 비교 — 패턴 형성·심화·개입을 자동 추적</span></h2>'
            f'<div style="display:flex;gap:8px;align-items:center;margin:0 0 8px;flex-wrap:wrap">'
            f'<input id="tsf" placeholder="웨이퍼 데이터 폴더 (여러 곳이면 ; 구분)" spellcheck="false" '
            f'style="flex:1;min-width:260px;font-family:Consolas,monospace;font-size:13px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 10px">'
            f'<input id="tse" placeholder="설비" style="width:90px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
            f'<input id="tsd" placeholder="DSV" style="width:64px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
            f'<input id="tsr" placeholder="유형" style="width:84px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
            f'<input id="tsm" placeholder="월 (예: 1-8)" style="width:90px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">'
            f'<input id="tsn" placeholder="전수" title="일별 상한(비우면 전수조사)" style="width:48px;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 8px">장/일</div>'
            f'<div style="display:flex;gap:8px;align-items:center;margin:0 0 10px;flex-wrap:wrap;font-size:12.5px">'
            f'기준 골든: <input id="tsst" placeholder="저장소 폴더" spellcheck="false" '
            f'style="width:200px;font-family:Consolas,monospace;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:6px 8px">'
            f'<input id="tscache" placeholder="캐시 폴더(선택)" spellcheck="false" '
            f'title="집계 캐시를 둘 경로 — 비우면 저장소 안" '
            f'style="width:170px;font-family:Consolas,monospace;font-size:12.5px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:6px 8px">'
            f'<button onclick="__tslist()" style="padding:6px 14px;border:1px solid #6a4fa3;'
            f'border-radius:8px;background:#fff;color:#6a4fa3;font-size:12.5px;cursor:pointer">기준 선택</button>'
            f'<span id="tspick" style="font-size:12px;color:#5b6675"></span>'
            f'<button id="tsgo" onclick="__tscan()" style="padding:8px 18px;border:none;border-radius:8px;'
            f'background:#7b1fa2;color:#fff;font-size:13.5px;cursor:pointer">추세 스캔 실행</button>'
            f'<button id="tsnr" onclick="__tsnarrate()" style="padding:8px 14px;border:1px solid #1f6f8b;'
            f'border-radius:8px;background:#fff;color:#1f6f8b;font-size:13px;cursor:pointer;display:none">'
            f'LLM 종합 서사 (VLM 검증 포함)</button>'
            f'<span id="tsmsg" style="font-size:12.5px;color:#5b6675"></span></div>'
            f'<div id="tsout"></div></div>')
        kz = len(recipes) + 3
        body += (
            f'<div class="tabpane" id="pane{kz}">'
            f'<h2>주요 패턴 <span class="n" style="font-size:12px;color:#8a94a3">'
            f'추세 스캔에서 여러 번 발생한 같은 문제만 정리 — P/T 센서 한정, 히터류 제외</span></h2>'
            f'<div style="display:flex;gap:8px;align-items:center;margin:0 0 10px;flex-wrap:wrap;font-size:12.5px">'
            f'<button onclick="__mpRender()" style="padding:7px 16px;border:none;border-radius:8px;'
            f'background:#7b1fa2;color:#fff;font-size:13px;cursor:pointer">최신 추세 스캔에서 불러오기</button>'
            f'<label style="cursor:pointer"><input type="checkbox" id="mpall" onchange="__mpRender()" '
            f'style="vertical-align:-2px"> 1회 발생도 포함</label>'
            f'<button onclick="__mpcsv()" style="padding:7px 14px;border:1px solid #6a4fa3;border-radius:8px;'
            f'background:#fff;color:#6a4fa3;font-size:12.5px;cursor:pointer">선택 패턴 CSV 저장</button>'
            f'<button id="mpnr" onclick="__mpNarrate()" style="padding:7px 14px;border:1px solid #1f6f8b;'
            f'border-radius:8px;background:#fff;color:#1f6f8b;font-size:12.5px;cursor:pointer">'
            f'LLM 원인 분석·요약</button>'
            f'<span id="mpmsg" style="font-size:12.5px;color:#5b6675"></span></div>'
            f'<div id="mpout"></div></div>')
        kv = len(recipes) + 4
        _in9 = ('font-size:12.5px;border:1px solid #dde3ea;'
                'border-radius:8px;padding:8px 8px')
        body += (
            f'<div class="tabpane" id="pane{kv}">'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:11.5px;'
            f'margin:0 0 6px;padding:6px 8px;border:1px dashed #dde3ea;border-radius:8px">'
            f'프리셋 <input id="sdpsname" placeholder="이름" style="width:110px;{_in9}">'
            f'<button data-ps="save" data-scope="sd" style="padding:3px 10px;border:1px solid #2456a6;'
            f'border-radius:6px;background:#fff;color:#2456a6;cursor:pointer">저장</button>'
            f'<button data-ps="list" data-scope="sd" style="padding:3px 10px;border:1px solid #dde3ea;'
            f'border-radius:6px;background:#fff;cursor:pointer">목록</button>'
            f'<span id="sdpsbox"></span>'
            f'<button data-ps="load" data-scope="sd" style="padding:3px 10px;border:1px solid #1f6f8b;'
            f'border-radius:6px;background:#fff;color:#1f6f8b;cursor:pointer">불러오기</button>'
            f'<button data-ps="del" data-scope="sd" style="padding:3px 10px;border:1px solid #dde3ea;'
            f'border-radius:6px;background:#fff;color:#c0392b;cursor:pointer">삭제</button>'
            f'<span id="sdpsmsg" style="color:#5b6675"></span></div>'
            f'<h2>스텝 차이 <span class="n" style="font-size:12px;color:#8a94a3">'
            f'통합 골든(정상 구간 평균) 대비 센서×스텝 차이를 원단위(℃/bar)로 본다 — 가상센서 포함</span></h2>'
            f'<div class="card" style="padding:10px 12px;margin:0 0 10px">'
            f'<div style="font-size:13px;font-weight:600;margin:0 0 6px">통합 골든 생성 '
            f'<span style="font-weight:400;color:#8a94a3;font-size:12px">'
            f'정상 구간 폴더(양품만)를 판정 없이 전체 평균 — DSV 통합, 필터는 옵션</span></div>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px">'
            f'<input id="gbf" placeholder="정상 구간 폴더 (여러 곳이면 ; 구분)" spellcheck="false" '
            f'style="flex:1;min-width:260px;font-family:Consolas,monospace;font-size:13px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 10px">'
            f'<input id="gbe" placeholder="설비(옵션)" style="width:96px;{_in9}">'
            f'<input id="gbd" placeholder="DSV(옵션)" style="width:84px;{_in9}">'
            f'<input id="gbr" placeholder="유형(옵션)" style="width:90px;{_in9}">'
            f'<input id="gbm" placeholder="월(옵션)" style="width:80px;{_in9}"></div>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;margin-top:8px">'
            f'저장: <input id="gbn" placeholder="이름" style="width:130px;{_in9}">'
            f'<input id="gbst" placeholder="저장소 폴더" spellcheck="false" '
            f'style="width:240px;font-family:Consolas,monospace;{_in9}">'
            f'<button id="gbgo" onclick="__gbuild()" style="padding:8px 18px;border:none;border-radius:8px;'
            f'background:#2456a6;color:#fff;font-size:13.5px;cursor:pointer">통합 골든 생성</button>'
            f'<span id="gbmsg" style="font-size:12.5px;color:#5b6675"></span></div>'
            f'<div id="gbout"></div></div>'
            f'<div style="display:flex;gap:8px;align-items:center;margin:0 0 8px;flex-wrap:wrap">'
            f'<input id="sdf" placeholder="분석할 웨이퍼 데이터 폴더 (여러 곳이면 ; 구분)" spellcheck="false" '
            f'style="flex:1;min-width:260px;font-family:Consolas,monospace;font-size:13px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 10px">'
            f'<input id="sde" placeholder="설비" style="width:90px;{_in9}">'
            f'<input id="sdv" placeholder="DSV" style="width:64px;{_in9}">'
            f'<input id="sdr" placeholder="유형" style="width:84px;{_in9}">'
            f'<input id="sdm" placeholder="월 (예: 1-8)" style="width:90px;{_in9}"></div>'
            f'<div style="display:flex;gap:8px;align-items:center;margin:0 0 10px;flex-wrap:wrap;font-size:12.5px">'
            f'기준 골든: <input id="sdst" placeholder="저장소 폴더" spellcheck="false" '
            f'style="width:200px;font-family:Consolas,monospace;{_in9}">'
            f'<input id="sdcache" placeholder="캐시 폴더(비우면 저장소 안)" spellcheck="false" '
            f'title="집계 캐시를 둘 경로 — 골든 저장소와 분리하려면 지정" '
            f'style="width:190px;font-family:Consolas,monospace;{_in9}">'
            f'<button onclick="__sdlist()" style="padding:6px 14px;border:1px solid #6a4fa3;'
            f'border-radius:8px;background:#fff;color:#6a4fa3;font-size:12.5px;cursor:pointer">기준 선택</button>'
            f'<span id="sdpick" style="font-size:12px;color:#5b6675"></span>'
            f'<button id="sdgo" onclick="__sdrun()" style="padding:8px 18px;border:none;border-radius:8px;'
            f'background:#1f6f8b;color:#fff;font-size:13.5px;cursor:pointer">스텝 차이 집계</button>'
            f'<span id="sdmsg" style="font-size:12.5px;color:#5b6675"></span></div>'
            f'<div id="sdctl" style="display:none">'
            f'<div style="display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;margin:0 0 8px">'
            f'<div style="flex:1;min-width:280px"><div style="font-size:12px;color:#5b6675;margin-bottom:3px">센서 '
            f'<a href="#" onclick="__sdchk(\'sdsen\',1);return false" style="font-size:11px">전체</a> '
            f'<a href="#" onclick="__sdchk(\'sdsen\',0);return false" style="font-size:11px">해제</a></div>'
            f'<div id="sdsen" style="max-height:130px;overflow:auto;border:1px solid #eef1f5;'
            f'border-radius:8px;padding:6px;font-size:12px;line-height:1.9"></div></div>'
            f'<div style="min-width:150px"><div style="font-size:12px;color:#5b6675;margin-bottom:3px">IO(밸브) 표시 '
            f'<a href="#" onclick="__sdchk(\'sdio\',0);return false" style="font-size:11px">해제</a></div>'
            f'<div id="sdio" style="max-height:130px;overflow:auto;border:1px solid #eef1f5;'
            f'border-radius:8px;padding:6px;font-size:12px;line-height:1.9"></div></div>'
            f'<div style="min-width:220px"><div style="font-size:12px;color:#5b6675;margin-bottom:3px">기간(월&gt;주&gt;일)</div>'
            f'<div id="sddt" style="max-height:130px;overflow:auto;border:1px solid #eef1f5;'
            f'border-radius:8px;padding:6px;font-size:12px;line-height:1.8"></div></div>'
            f'<div><span id="sdmode">'
            f'<button data-m="wafer" onclick="__sdsetmode(this)" style="padding:7px 12px;border:1px solid #2456a6;'
            f'border-radius:8px 0 0 8px;background:#2456a6;color:#fff;font-size:12.5px;cursor:pointer">웨이퍼</button>'
            f'<button data-m="day1" onclick="__sdsetmode(this)" style="padding:7px 12px;border:1px solid #2456a6;'
            f'border-left:none;background:#fff;color:#2456a6;font-size:12.5px;cursor:pointer">하루 평균</button>'
            f'<button data-m="day5" onclick="__sdsetmode(this)" style="padding:7px 12px;border:1px solid #2456a6;'
            f'border-left:none;border-radius:0 8px 8px 0;background:#fff;color:#2456a6;font-size:12.5px;cursor:pointer">5일 평균</button>'
            f'</span>'
            f'<br><span id="sdym" style="display:inline-block;margin-top:6px">'
            f'<button data-y="raw" onclick="__sdsety(this)" style="padding:6px 10px;border:1px solid #1f6f8b;'
            f'border-radius:8px 0 0 8px;background:#1f6f8b;color:#fff;font-size:11.5px;cursor:pointer">원단위</button>'
            f'<button data-y="sig" onclick="__sdsety(this)" style="padding:6px 10px;border:1px solid #1f6f8b;'
            f'border-left:none;border-radius:0 8px 8px 0;background:#fff;color:#1f6f8b;font-size:11.5px;cursor:pointer" '
            f'title="문턱(σ×2) 배수로 정규화 — 스케일 작은 압력도 온도와 같은 눈높이로 보임">σ배</button>'
            f'</span>'
            f'<br><span id="sdpm" style="display:inline-block;margin-top:6px">'
            f'<button data-p="prev" onclick="__sdpmode(this)" style="padding:6px 10px;border:1px solid #7b1fa2;'
            f'border-radius:8px 0 0 8px;background:#7b1fa2;color:#fff;font-size:11.5px;cursor:pointer">이전 대비</button>'
            f'<button data-p="seq" onclick="__sdpmode(this)" style="padding:6px 10px;border:1px solid #7b1fa2;'
            f'border-left:none;border-radius:0 8px 8px 0;background:#fff;color:#7b1fa2;font-size:11.5px;cursor:pointer">순차 비교</button>'
            f'</span>'
            f'<br><button id="sdpb" onclick="__sdpatterns()" style="margin-top:6px;padding:7px 14px;border:1px solid #7b1fa2;'
            f'border-radius:8px;background:#fff;color:#7b1fa2;font-size:12.5px;cursor:pointer">선택 기간 패턴</button>'
            f'<br><button id="sdnr" onclick="__sdnarrate()" style="margin-top:6px;padding:7px 14px;border:1px solid #1f6f8b;'
            f'border-radius:8px;background:#fff;color:#1f6f8b;font-size:12.5px;cursor:pointer">LLM 정리</button></div></div>'
            f'<details style="border:1px solid #f2e3c8;border-radius:8px;padding:8px 10px;margin:0 0 8px;background:#fffdf7">'
            f'<summary style="font-size:12.5px;font-weight:600;cursor:pointer">사건(레이블) 분석 '
            f'<span style="font-weight:400;color:#8a94a3;font-size:11.5px">조치내역 CSV 연결 — 사건일 직전 대비 무엇이 가장 달라졌나</span></summary>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;margin:8px 0 4px">'
            f'<input id="sdev_csv" placeholder="레이블 CSV 경로" spellcheck="false" '
            f'style="flex:1;min-width:260px;font-family:Consolas,monospace;{_in9}">'
            f'사건 창 <input id="sdev_e" value="3" style="width:40px;{_in9}">일 · '
            f'기준 창 <input id="sdev_b" value="14" style="width:44px;{_in9}">일'
            f'<button id="sdevgo" onclick="__sdevents()" style="padding:8px 16px;border:none;border-radius:8px;'
            f'background:#b26a00;color:#fff;font-size:13px;cursor:pointer">사건 분석</button>'
            f'<span id="sdevmsg" style="font-size:12px;color:#5b6675"></span></div>'
            f'<div style="font-size:11.5px;color:#8a94a3;margin:0 0 6px">'
            f'CSV 형식: <b>date,label</b>[,eqp,dsv,memo] · 예) 2026-07-16,현PC 불량,WUK370A,dsv5,교체 조치 '
            f'· 날짜는 2026-07-16 / 2026.7.16 / 7/16 모두 인식 · 엑셀 저장(CP949)·탭 구분도 가능 '
            f'· 예시 파일: 앱 폴더의 event_labels_example.csv</div>'
            f'<div id="sdevout"></div></details>'
            f'<div id="sdpat" style="margin:0 0 8px"></div>'
            f'<div id="sdfind" style="margin:0 0 8px"></div>'
            f'<div id="sdnarr"></div>'
            f'<div id="sdlegend" style="font-size:11.5px;margin:0 0 6px"></div></div>'
            f'<div id="sdcomp"></div>'
            f'<div id="sdout"></div>'
            f'<div style="text-align:center;margin:8px 0">'
            f'<button id="sdmore" onclick="__sdfetch(false)" style="display:none;padding:8px 26px;'
            f'border:1px solid #dde3ea;border-radius:8px;background:#fff;font-size:13px;cursor:pointer">더 보기</button></div>'
            f'</div>')
        kw = len(recipes) + 5
        body += (
            f'<div class="tabpane" id="pane{kw}">'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:11.5px;'
            f'margin:0 0 6px;padding:6px 8px;border:1px dashed #dde3ea;border-radius:8px">'
            f'프리셋 <input id="vzpsname" placeholder="이름" style="width:110px;{_in9}">'
            f'<button data-ps="save" data-scope="vz" style="padding:3px 10px;border:1px solid #2456a6;'
            f'border-radius:6px;background:#fff;color:#2456a6;cursor:pointer">저장</button>'
            f'<button data-ps="list" data-scope="vz" style="padding:3px 10px;border:1px solid #dde3ea;'
            f'border-radius:6px;background:#fff;cursor:pointer">목록</button>'
            f'<span id="vzpsbox"></span>'
            f'<button data-ps="load" data-scope="vz" style="padding:3px 10px;border:1px solid #1f6f8b;'
            f'border-radius:6px;background:#fff;color:#1f6f8b;cursor:pointer">불러오기</button>'
            f'<button data-ps="del" data-scope="vz" style="padding:3px 10px;border:1px solid #dde3ea;'
            f'border-radius:6px;background:#fff;color:#c0392b;cursor:pointer">삭제</button>'
            f'<span id="vzpsmsg" style="color:#5b6675"></span></div>'
            f'<h2>데이터 시각화 <span class="n" style="font-size:12px;color:#8a94a3">'
            f'센서별 원본 파형 + 골든 오버레이(스텝 정렬) — 독립 탭(스텝 차이와 무관)</span></h2>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;margin:0 0 8px">'
            f'<input id="vzf" placeholder="데이터 폴더 (여러 곳이면 ; 구분)" spellcheck="false" '
            f'style="flex:1;min-width:260px;font-family:Consolas,monospace;font-size:13px;'
            f'border:1px solid #dde3ea;border-radius:8px;padding:8px 10px">'
            f'<input id="vze" placeholder="설비" style="width:84px;{_in9}">'
            f'<input id="vzv" placeholder="DSV" style="width:64px;{_in9}">'
            f'<input id="vzr" placeholder="유형" style="width:84px;{_in9}">'
            f'<input id="vzm" placeholder="월 (예: 1-8)" style="width:90px;{_in9}"></div>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;margin:0 0 8px">'
            f'기준 골든: <input id="vzst" placeholder="저장소 폴더" spellcheck="false" '
            f'style="width:240px;font-family:Consolas,monospace;{_in9}">'
            f'<button onclick="__vzglist()" style="padding:6px 14px;border:1px solid #6a4fa3;'
            f'border-radius:8px;background:#fff;color:#6a4fa3;font-size:12.5px;cursor:pointer">기준 선택</button>'
            f'<span id="vzpick" style="font-size:12px;color:#5b6675"></span>'
            f'<button onclick="__vzload(1)" style="padding:8px 16px;border:none;border-radius:8px;'
            f'background:#2456a6;color:#fff;font-size:13px;cursor:pointer">웨이퍼 목록 불러오기</button>'
            f'<span id="vzmode0"><button data-m="overlay" onclick="__vzmode(this)" style="padding:7px 12px;'
            f'border:1px solid #2456a6;border-radius:8px 0 0 8px;background:#2456a6;color:#fff;font-size:12px;cursor:pointer">개별 겹침</button>'
            f'<button data-m="mean" onclick="__vzmode(this)" style="padding:7px 12px;border:1px solid #2456a6;'
            f'border-left:none;border-radius:0 8px 8px 0;background:#fff;color:#2456a6;font-size:12px;cursor:pointer">평균 그래프</button></span>'
            f'평균 단위 <span id="vzgm">'
            f'<button data-g="count" onclick="__vzgmode(this)" style="padding:6px 9px;border:1px solid #1f6f8b;'
            f'border-radius:8px 0 0 8px;background:#fff;color:#1f6f8b;font-size:11.5px;cursor:pointer">장수</button>'
            f'<button data-g="day" onclick="__vzgmode(this)" style="padding:6px 9px;border:1px solid #1f6f8b;'
            f'border-left:none;background:#1f6f8b;color:#fff;font-size:11.5px;cursor:pointer">일</button>'
            f'<button data-g="hour" onclick="__vzgmode(this)" style="padding:6px 9px;border:1px solid #1f6f8b;'
            f'border-left:none;border-radius:0 8px 8px 0;background:#fff;color:#1f6f8b;font-size:11.5px;cursor:pointer">시간</button>'
            f'</span>'
            f'<input id="vzb" placeholder="전체" title="장수=시간순 N장씩 · 일=N일씩 · 시간=N시간씩 묶어 카드 1개 '
            f'(빈칸=선택 기간 전체 1카드). 센서를 복수 체크하면 카드 안에서 오버레이" '
            f'style="width:48px;{_in9}"><span id="vzgu">일</span>'
            f'<label style="cursor:pointer"><input type="checkbox" id="vzg" checked style="vertical-align:-2px"> 골든 표시</label>'
            f'<button onclick="__vzdraw()" style="padding:8px 16px;border:none;border-radius:8px;'
            f'background:#1f6f8b;color:#fff;font-size:13px;cursor:pointer">그래프 그리기</button>'
            f'<span id="vzmsg" style="font-size:12px;color:#5b6675"></span></div>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;margin:0 0 8px">'
            f'리포트 CSV: <input id="vzrpt" placeholder="불량/조치 리포트 CSV 경로 (date,dsv,label)" spellcheck="false" '
            f'style="flex:1;min-width:280px;font-family:Consolas,monospace;{_in9}">'
            f'<button onclick="__vzrptLoad()" style="padding:7px 14px;border:1px solid #e08a00;'
            f'border-radius:8px;background:#fff;color:#b26a00;font-size:12.5px;cursor:pointer">리포트 불러오기</button>'
            f'<span id="vzrptmsg" style="font-size:12px;color:#5b6675"></span>'
            f'<span style="font-size:11.5px;color:#8a94a3">주황=리포트일 · 노랑=최초 리포트 3일 전 '
            f'(예시: 앱 폴더 report_labels_example.csv)</span></div>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;margin:0 0 8px">'
            f'두 기간 비교: <button data-cmp="A" style="padding:6px 12px;border:1px solid #2456a6;'
            f'border-radius:8px;background:#fff;color:#2456a6;cursor:pointer">체크 기간을 A(기준)로</button>'
            f'<button data-cmp="B" style="padding:6px 12px;border:1px solid #c0392b;'
            f'border-radius:8px;background:#fff;color:#c0392b;cursor:pointer">B(대상)로</button>'
            f'<span id="vzcmpchip" style="font-size:11.5px;color:#5b6675"></span>'
            f'<button data-cmp="go" style="padding:7px 16px;border:none;border-radius:8px;'
            f'background:#7b1fa2;color:#fff;font-size:12.5px;cursor:pointer">두 기간 비교</button>'
            f'<span id="vzcmpmsg" style="font-size:12px;color:#5b6675"></span></div>'
            f'<div id="vzcmpout"></div>'
            f'<div style="display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;margin:0 0 8px">'
            f'<div style="flex:1;min-width:260px"><div style="font-size:12px;color:#5b6675;margin-bottom:3px">센서(가상센서 포함) '
            f'<a href="#" onclick="__sdchk(\'vzsen\',1);return false" style="font-size:11px">전체</a> '
            f'<a href="#" onclick="__sdchk(\'vzsen\',0);return false" style="font-size:11px">해제</a></div>'
            f'<div id="vzsen" style="max-height:120px;overflow:auto;border:1px solid #eef1f5;'
            f'border-radius:8px;padding:6px;font-size:12px;line-height:1.9"></div></div>'
            f'<div style="min-width:220px"><div style="font-size:12px;color:#5b6675;margin-bottom:3px">기간(월&gt;주&gt;일)</div>'
            f'<div id="vzdt" style="max-height:120px;overflow:auto;border:1px solid #eef1f5;'
            f'border-radius:8px;padding:6px;font-size:12px;line-height:1.8"></div></div>'
            f'<div style="min-width:140px"><div style="font-size:12px;color:#5b6675;margin-bottom:3px">스텝 '
            f'<a href="#" onclick="__sdchk(\'vzstep\',1);return false" style="font-size:11px">전체</a> '
            f'<a href="#" onclick="__sdchk(\'vzstep\',0);return false" style="font-size:11px">해제</a>'
            f'<span style="color:#8a94a3;font-size:10.5px"> 체크한 스텝만 확대</span></div>'
            f'<div id="vzstep" style="max-height:120px;overflow:auto;border:1px solid #eef1f5;'
            f'border-radius:8px;padding:6px;font-size:12px;line-height:1.9"></div></div>'
            f'<div style="flex:1;min-width:280px"><div style="font-size:12px;color:#5b6675;margin-bottom:3px">웨이퍼 선택 '
            f'<a href="#" onclick="__vzwAll(1);return false" style="font-size:11px">전체</a> '
            f'<a href="#" onclick="__vzwAll(0);return false" style="font-size:11px">해제</a> '
            f'<span id="vzwn" style="color:#8a94a3;font-size:11px"></span></div>'
            f'<div id="vzw" style="height:132px;overflow:auto;border:1px solid #eef1f5;'
            f'border-radius:8px;padding:0;font-size:12px;position:relative"></div></div></div>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;margin:0 0 8px">'
            f'<input id="vzcsv" placeholder="선택 웨이퍼 이름 저장 CSV 경로/파일명 (예: /data/선별목록.csv)" spellcheck="false" '
            f'style="flex:1;min-width:300px;font-family:Consolas,monospace;{_in9}">'
            f'<button onclick="__vzsave()" style="padding:8px 16px;border:1px solid #6a4fa3;border-radius:8px;'
            f'background:#fff;color:#6a4fa3;font-size:12.5px;cursor:pointer">선택 웨이퍼 CSV 저장</button>'
            f'<span id="vzsmsg" style="font-size:12px;color:#5b6675"></span></div>'
            f'<div id="vzout"></div></div>')
        kq = len(recipes) + 6
        # 패턴 규칙 센서 표시명 — P1/P2 같은 기호 대신 실제 센서가 보이게(사용자 지적)
        _pt_sn1, _pt_sn2 = "t_002_PV", "dP_VslF-VslB"
        _pt_st1, _pt_st2 = "5", "2,3,4"
        try:
            from io_recon_tcn import rca_causal as _RCl
            _pr9 = (_RCl.load_physics()[1].get("pattern_rules") or {})
            _pt_sn1 = str((_pr9.get("exp_drop") or {}).get("sensor") or _pt_sn1)
            _pt_sn2 = str((_pr9.get("arches") or {}).get("sensor") or _pt_sn2)
            _pt_st1 = str((_pr9.get("exp_drop") or {}).get("step") or _pt_st1)
            _pt_st2 = ",".join(str(v) for v in ((_pr9.get("arches") or {}).get("steps")
                                                or [2, 3, 4]))
        except Exception:                              # noqa: BLE001
            pass
        _pt_d1 = _pt_sn1[:-3] if _pt_sn1.endswith("_PV") else _pt_sn1
        _pt_d2 = ("dP front-back(F−B)" if _pt_sn2 == "dP_VslF-VslB"
                  else _pt_sn2)
        _pt_sn3, _pt_st3 = "t_003_PV", "4"
        try:
            _pt_sn3 = str((_pr9.get("w_bulge") or {}).get("sensor") or _pt_sn3)
            _pt_st3 = str((_pr9.get("w_bulge") or {}).get("step") or _pt_st3)
        except Exception:                              # noqa: BLE001
            pass
        _pt_d3 = _pt_sn3[:-3] if _pt_sn3.endswith("_PV") else _pt_sn3
        body += (
            f'<div class="tabpane" id="pane{kq}">'
            f'<div style="font-size:12px;color:#5b6675;margin:0 0 6px">'
            f'파형의 모양 자체를 점수화합니다 — {_pt_d1}: 지수 하강 볼록비(현 대비 면적), '
            f'{_pt_d2}: 아치 반복(받침점 현 대비 면적). 임계는 골든 상대(α×골든 점수). '
            f'먼저 [데이터 시각화] 탭에서 웨이퍼 목록을 불러오세요.</div>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:0 0 8px;'
            f'font-size:12.5px">'
            f'{_pt_d1} 스텝 <input id="pt_s1" value="{_pt_st1}" style="width:44px;{_in9}"> '
            f'{_pt_d2} 스텝 <input id="pt_s2" value="{_pt_st2}" style="width:70px;{_in9}"> '
            f'α({_pt_d1}) <input id="pt_a1" value="0.4" style="width:48px;{_in9}"> '
            f'α({_pt_d2}) <input id="pt_a2" value="0.5" style="width:48px;{_in9}"> '
            f'기간 <input id="pt_d0" placeholder="시작 YYYY-MM-DD" style="width:120px;{_in9}">'
            f'~<input id="pt_d1" placeholder="끝(비우면 전체)" style="width:120px;{_in9}"> '
            f'평균 <input id="pt_bd" value="5" style="width:40px;{_in9}">일 '
            f'{_pt_d3}(W) 스텝 <input id="pt_s3" value="{_pt_st3}" style="width:44px;{_in9}"> '
            f'W 임계 여유(면적) <input id="pt_a3" value="0.5" style="width:52px;{_in9}"> '
            f'W 최고점 % <input id="pt_n3" value="80" style="width:44px;{_in9}"> '
            f'<span style="color:#8a94a3">보기:</span> '
            f'<label><input type="checkbox" id="pt_f1" checked> 직선 하강</label> '
            f'<label><input type="checkbox" id="pt_f2" checked> 아치</label> '
            f'<label><input type="checkbox" id="pt_f3" checked> W패턴</label> '
            f'<label><input type="checkbox" id="pt_f4" checked> 온도 계통</label> '
            f'<label><input type="checkbox" id="pt_f5" checked> 압력 계통</label> '
            f'<button onclick="__ptScan()" style="padding:8px 18px;border:0;border-radius:8px;'
            f'background:#2456a6;color:#fff;font-size:13px;cursor:pointer">패턴 스캔</button>'
            f'<button id="ptnbtn" onclick="__ptNarrate()" style="display:none;padding:8px 16px;'
            f'border:1px solid #1f6f8b;border-radius:8px;background:#fff;color:#1f6f8b;'
            f'font-size:12.5px;cursor:pointer">LLM 정리</button>'
            f'<span id="ptmsg" style="font-size:12px;color:#5b6675"></span></div>'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;'
            f'margin:0 0 6px;font-size:12px">불량 내역 CSV '
            f'<input id="pt_mk" placeholder="경로 (날짜,설비,DSV,불량내역)" '
            f'style="width:300px;{_in9}">'
            f'<button id="pt_mkb" onclick="__ptMarks()" style="padding:4px 12px;border:1px solid #e08a00;'
            f'border-radius:6px;background:#fff;color:#b26a00;font-size:12px;cursor:pointer">'
            f'불러오기</button>'
            f'<button onclick="__ptMarksClr()" style="padding:4px 10px;border:1px solid #dde3ea;'
            f'border-radius:6px;background:#fff;font-size:12px;cursor:pointer">지우기</button>'
            f'<span style="color:#8a94a3">— 달력에 \u25b2 표시, 마우스를 올리면 내역</span></div>'
            f'<div id="ptout"></div></div>')
        kr = len(recipes) + 7
        body += (
            f'<div class="tabpane" id="pane{kr}">'
            f'<div style="font-size:12px;color:#5b6675;margin:0 0 6px">'
            f'패턴 인식 보고자료 — 기간·설비·DSV·레시피를 고르고 N일 평균으로 봅니다. '
            f'전부 비우면 전체 기간·전체 설비·전체 DSV·전체 레시피, 1일 평균. '
            f'먼저 [데이터 시각화] 탭에서 웨이퍼 목록을 불러오세요.</div>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:0 0 6px;'
            f'font-size:12.5px">'
            f'기간 <input id="an_d0" placeholder="시작 YYYY-MM-DD" style="width:118px;{_in9}">'
            f'~<input id="an_d1" placeholder="끝(비우면 전체)" style="width:118px;{_in9}"> '
            f'평균 <input id="an_nd" value="1" style="width:36px;{_in9}">일 '
            f'{_pt_d1} 스텝 <input id="an_s1" value="{_pt_st1}" style="width:40px;{_in9}"> '
            f'{_pt_d2} 스텝 <input id="an_s2" value="{_pt_st2}" style="width:64px;{_in9}"> '
            f'{_pt_d3}(W) 스텝 <input id="an_s3" value="{_pt_st3}" style="width:40px;{_in9}"> '
            f'α <input id="an_a1" value="0.4" style="width:44px;{_in9}">'
            f'/<input id="an_a2" value="0.5" style="width:44px;{_in9}">'
            f'/W여유<input id="an_a3" value="0.5" style="width:44px;{_in9}"> '
            f'/W 최고점 %<input id="an_n3" value="80" style="width:40px;{_in9}"> '
            f'<span style="color:#8a94a3">보기:</span> '
            f'<label><input type="checkbox" id="an_f1" checked> 직선 하강</label> '
            f'<label><input type="checkbox" id="an_f2" checked> 아치</label> '
            f'<label><input type="checkbox" id="an_f3" checked> W패턴</label> '
            f'<label><input type="checkbox" id="an_f4" checked> 온도 계통</label> '
            f'<label><input type="checkbox" id="an_f5" checked> 압력 계통</label> '
            f'<button onclick="__anStart()" style="padding:8px 18px;border:0;border-radius:8px;'
            f'background:#2456a6;color:#fff;font-size:13px;cursor:pointer">분석 시작</button>'
            f'<span id="anmsg" style="font-size:12px;color:#5b6675"></span></div>'
            f'<div id="anflt" style="display:none;margin:0 0 6px;font-size:12.5px">'
            f'<div style="margin:0 0 3px">설비 <span id="aneqp"></span></div>'
            f'<div style="margin:0 0 3px">DSV <span id="andsv"></span></div>'
            f'<div style="margin:0 0 3px">레시피 <span id="anrc"></span> '
            f'<button onclick="__anAgg(false)" style="padding:4px 12px;border:1px solid #2456a6;'
            f'border-radius:6px;background:#fff;color:#2456a6;font-size:12px;cursor:pointer">'
            f'달력 갱신</button> '
            f'<button onclick="__anFull()" style="padding:4px 12px;border:1px solid #dde3ea;'
            f'border-radius:6px;background:#fff;font-size:12px;cursor:pointer">기간 전체보기</button> '
            f'<button onclick="__anExpCur()" style="padding:4px 12px;border:1px solid #1f6f8b;'
            f'border-radius:6px;background:#fff;color:#1f6f8b;font-size:12px;cursor:pointer">'
            f'현재 설비 내보내기</button> '
            f'<button onclick="__anExpAll()" style="padding:4px 12px;border:1px solid #1f6f8b;'
            f'border-radius:6px;background:#fff;color:#1f6f8b;font-size:12px;cursor:pointer">'
            f'설비별 일괄 내보내기</button></div></div>'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;'
            f'margin:0 0 6px;font-size:12px">불량 내역 CSV '
            f'<input id="an_mk" placeholder="경로 (날짜,설비,DSV,불량내역)" '
            f'style="width:300px;{_in9}">'
            f'<button id="an_mkb" onclick="__anMarks()" style="padding:4px 12px;border:1px solid #e08a00;'
            f'border-radius:6px;background:#fff;color:#b26a00;font-size:12px;cursor:pointer">'
            f'불러오기</button>'
            f'<button onclick="__anMarksClr()" style="padding:4px 10px;border:1px solid #dde3ea;'
            f'border-radius:6px;background:#fff;font-size:12px;cursor:pointer">지우기</button>'
            f'<span style="color:#8a94a3">— 선택한 설비/DSV 에 해당하는 내역만 \u25b2 표시</span></div>'
            f'<div id="ancal"></div><div id="anout"></div></div>')
        kd = len(recipes) + 8
        body += (
            f'<div class="tabpane" id="pane{kd}">'
            f'<div style="font-size:12px;color:#5b6675;margin:0 0 6px">'
            f'원본을 자유롭게 그려보는 뷰어입니다(골든 불필요·정렬/축소 없음). '
            f'목록에서 웨이퍼 이름을 클릭하면 그 파일의 센서 목록이 나오고, '
            f'체크한 웨이퍼(최대 20장)가 센서별 카드에 겹쳐 그려집니다.</div>'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:0 0 6px;'
            f'font-size:12.5px">폴더 <input id="dv_f" placeholder="데이터 폴더 (;로 여러 개)" '
            f'style="width:340px;{_in9}">'
            f'<button onclick="__dvList()" style="padding:8px 16px;border:0;border-radius:8px;'
            f'background:#2456a6;color:#fff;font-size:13px;cursor:pointer">목록 불러오기</button>'
            f'<span id="dvmsg" style="font-size:12px;color:#5b6675"></span></div>'
            f'<div id="dv_flt" style="font-size:12px;margin:0 0 6px"></div>'
            f'<div style="display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap">'
            f'<div style="min-width:330px">'
            f'<div style="font-size:12px;color:#8a94a3;margin:0 0 3px">웨이퍼 '
            f'<input id="dv_wq" placeholder="검색" style="width:110px;{_in9}"> '
            f'<a href="javascript:void(0)" onclick="__dvWAll(1)" style="font-size:11px">전체</a> '
            f'<a href="javascript:void(0)" onclick="__dvWAll(0)" style="font-size:11px">해제</a></div>'
            f'<div id="dv_w" style="max-height:240px;overflow:auto;border:1px solid #eef1f5;'
            f'border-radius:8px;padding:4px 8px;font-size:12px"></div>'
            f'<div id="dv_pg" style="font-size:12px;margin:3px 0"></div></div>'
            f'<div style="min-width:330px">'
            f'<div style="font-size:12px;color:#8a94a3;margin:0 0 3px">센서 '
            f'<input id="dv_sq" placeholder="검색" style="width:110px;{_in9}"> '
            f'<span id="dv_sfrom" style="font-size:11px"></span></div>'
            f'<div id="dv_s" style="max-height:240px;overflow:auto;border:1px solid #eef1f5;'
            f'border-radius:8px;padding:4px 8px;font-size:12px"></div></div></div>'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:6px 0;'
            f'font-size:12.5px">수식: <input id="dv_fx" placeholder="예) (t_002_PV+t_001_PV)/2" '
            f'style="width:340px;{_in9}"> 센서 넣기: <select id="dv_fxsel" style="{_in9}"></select>'
            f'<button onclick="__dvFxIns()" style="padding:4px 10px;border:1px solid #dde3ea;'
            f'border-radius:6px;background:#fff;font-size:12px;cursor:pointer">삽입</button>'
            f'<button onclick="__dvFxAdd()" style="padding:4px 12px;border:1px solid #6a4fa3;'
            f'border-radius:6px;background:#fff;color:#6a4fa3;font-size:12px;cursor:pointer">수식 추가</button>'
            f'</div><div id="dv_fxl" style="font-size:12px;margin:0 0 6px"></div>'
            f'<button onclick="__dvDraw()" style="padding:8px 18px;border:0;border-radius:8px;'
            f'background:#2456a6;color:#fff;font-size:13px;cursor:pointer">그래프 그리기</button>'
            f'<div id="dv_out" style="margin-top:8px"></div></div>')
        kh = len(recipes) + 9
        body += (
            f'<div class="tabpane" id="pane{kh}">'
            f'<div style="font-size:12px;color:#5b6675;margin:0 0 6px">'
            f'패턴 탐사 데몬(pattern_hunter.py)이 만든 가설-검증 원장을 봅니다. '
            f'가설을 클릭하면 설명과 달력(\u25b2문제 리포트·이상 패턴 일)이 나오고, '
            f'달력의 일/월을 누르면 웨이퍼 목록, 웨이퍼 이름을 누르면 '
            f'해당 센서의 패턴 카드가 나옵니다.</div>'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:0 0 6px;'
            f'font-size:12.5px">탐사 작업 폴더 <input id="hx_out" '
            f'placeholder="pattern_hunter --out 폴더" style="width:340px;{_in9}">'
            f'<button onclick="__hxList()" style="padding:8px 16px;border:0;border-radius:8px;'
            f'background:#2456a6;color:#fff;font-size:13px;cursor:pointer">원장 불러오기</button>'
            f'<span id="hxmsg" style="font-size:12px;color:#5b6675"></span></div>'
            f'<div id="hx_l" style="font-size:12.5px;margin:0 0 8px"></div>'
            f'<div id="hx_d"></div><div id="hx_w" style="margin-top:6px"></div>'
            f'<div id="hx_c" style="margin-top:6px"></div></div>')
        kg = len(recipes) + 10
        _gvh = ('display:flex;gap:8px;align-items:center;cursor:pointer;padding:6px 10px;'
                'border:1px solid #dde3ea;border-radius:8px;background:#f6f8fb;font-size:13px;'
                'font-weight:600;margin:6px 0 4px')
        _gvb = ('font-size:11.5px;font-weight:400;color:#2456a6;border:1px solid #c9d7ee;'
                'border-radius:10px;padding:1px 8px;background:#fff')
        _gvbtn = ('padding:8px 16px;border:0;border-radius:8px;background:#2456a6;color:#fff;'
                  'font-size:13px;cursor:pointer')
        _gvlt = ('padding:4px 12px;border:1px solid #2456a6;border-radius:6px;background:#fff;'
                 'color:#2456a6;font-size:12px;cursor:pointer')
        _gvex = ('padding:4px 12px;border:1px solid #1f6f8b;border-radius:6px;background:#fff;'
                 'color:#1f6f8b;font-size:12px;cursor:pointer')
        body += (
            f'<div class="tabpane" id="pane{kg}">'
            f'<div style="font-size:12px;color:#5b6675;margin:0 0 4px">'
            f'골든 비교 — 정상이 모인 기간의 평균(불량조치내역 CSV 보고분 제외)으로 '
            f'설비 대표·DSV별 골든을 만들고, 다른 기간의 웨이퍼를 스텝별 파형(offset 무시)으로 '
            f'판정합니다. 단계를 위에서 아래로 진행하세요 — 완료된 단계는 배지로 표시됩니다.'
            f'<span id="gvmsg" style="margin-left:10px;color:#2456a6"></span></div>'
            # ── ① 데이터 ──
            f'<div style="{_gvh}" onclick="__gvToggle(1)">① 데이터 불러오기 '
            f'<span id="gv_b1" style="{_gvb}">대기</span>'
            f'<span style="margin-left:auto;font-weight:400;color:#8a94a3;font-size:12px">'
            f'폴더·필터·불량조치내역 CSV</span></div>'
            f'<div id="gv_s1" style="padding:4px 6px 8px">'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;'
            f'font-size:12.5px">폴더 <input id="gv_fol" '
            f'placeholder="데이터 폴더(;로 여러 개)" style="width:300px;{_in9}"> '
            f'설비 <input id="gv_eq" placeholder="전체" style="width:90px;{_in9}"> '
            f'DSV <input id="gv_dv" placeholder="전체" style="width:70px;{_in9}"> '
            f'레시피 <input id="gv_rc" placeholder="전체" style="width:90px;{_in9}"> '
            f'월 <input id="gv_mo" placeholder="전체" style="width:70px;{_in9}"> '
            f'불량조치내역 CSV <input id="gv_csv" placeholder="경로(설비,DSV,불량원인,발생일자)" '
            f'style="width:230px;{_in9}"> '
            f'저장소 <input id="gv_st" placeholder="비우면 데이터 폴더 안 gv_store" '
            f'title="골든 json·판정 캐시·VLM 캐시 보관 폴더" style="width:160px;{_in9}">'
            f'<button onclick="__gvList()" style="{_gvbtn}">웨이퍼 목록 불러오기</button>'
            f'<button onclick="__gvFromViz()" style="{_gvlt}" '
            f'title="데이터 시각화 탭에서 이미 불러온 목록을 그대로 사용">'
            f'데이터 시각화 탭 데이터 사용</button></div>'
            f'<div id="gv_flt" style="display:none;margin:6px 0 0;font-size:12.5px">'
            f'<div style="margin:0 0 3px">센서 <a href="#" onclick="__gvSenAll(1);return false;" '
            f'style="color:#2456a6;font-size:11.5px">[전체]</a> '
            f'<a href="#" onclick="__gvSenAll(0);return false;" style="color:#2456a6;'
            f'font-size:11.5px">[해제]</a> <span id="gv_fsn" '
            f'style="display:inline-block;max-height:72px;overflow:auto"></span></div>'
            f'</div></div>'
            # ── ② 골든 ──
            f'<div style="{_gvh}" onclick="__gvToggle(2)">② 골든 산정 '
            f'<span id="gv_b2" style="{_gvb}">대기</span>'
            f'<span style="margin-left:auto;font-weight:400;color:#8a94a3;font-size:12px">'
            f'정상이 모인 기간 → 설비 대표 + DSV별</span></div>'
            f'<div id="gv_s2" style="display:none;padding:4px 6px 8px">'
            f'<div style="font-size:12.5px;margin:0 0 4px"><b>산정 대상</b> '
            f'<span style="color:#8a94a3">(체크한 설비·DSV·레시피만 골든 산정·판정에 포함)</span></div>'
            f'<div style="margin:0 0 3px;font-size:12.5px">설비 <span id="gv_feq"></span></div>'
            f'<div style="margin:0 0 3px;font-size:12.5px">DSV <span id="gv_fdv"></span></div>'
            f'<div style="margin:0 0 6px;font-size:12.5px">레시피 <span id="gv_frc"></span></div>'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:0 0 4px">'
            f'골든 기간 <input id="gv_d0" placeholder="시작 YYYY-MM-DD" style="width:118px;{_in9}">'
            f'~<input id="gv_d1" placeholder="끝(비우면 전체)" style="width:118px;{_in9}"> '
            f'<button onclick="__gvBuild()" style="{_gvbtn}">골든 산정</button>'
            f'<span id="gv_gsum" style="font-size:12px;color:#5b6675"></span></div>'
            f'<div id="gv_gpd" style="font-size:12px;margin:2px 0"></div>'
            f'<div id="gv_gal" style="margin-top:6px"></div></div>'
            # ── ③ 판정 ──
            f'<div style="{_gvh}" onclick="__gvToggle(3)">③ 판정 실행 '
            f'<span id="gv_b3" style="{_gvb}">대기</span>'
            f'<span style="margin-left:auto;font-weight:400;color:#8a94a3;font-size:12px">'
            f'분석 기간 · 모드 · 민감도</span></div>'
            f'<div id="gv_s3" style="display:none;padding:4px 6px 8px">'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:0 0 4px">'
            f'분석 기간 <input id="gv_e0" placeholder="시작(비우면 전체)" style="width:118px;{_in9}">'
            f'~<input id="gv_e1" placeholder="끝" style="width:118px;{_in9}"> '
            f'모드 <select id="gv_md" style="{_in9}">'
            f'<option value="rule">룰베이스 분석</option>'
            f'<option value="vlm">VLM 분석</option>'
            f'<option value="hybrid">하이브리드(룰베이스 선별 후 VLM)</option></select> '
            f'기준 골든 <select id="gv_gref" style="{_in9}" title="판정 기준으로 쓸 골든 — 다른 쪽은 카드에 회색 가이드로">'
            f'<option value="dsv">챔버(DSV)별 골든</option><option value="rep">설비 통합 골든</option></select> '
            f'민감도 <select id="gv_pre" onchange="__gvPreset()" style="{_in9}">'
            f'<option value="">직접 설정</option><option value="c">보수(k 3 · α 0.10)</option>'
            f'<option value="s">표준(k 2 · α 0.05)</option>'
            f'<option value="m">민감(k 1.5 · α 0.02)</option></select> '
            f'면적 임계 α <input id="gv_a" value="0.05" onchange="__gvLive()" style="width:52px;{_in9}"> '
            f'<button onclick="__gvRun()" style="{_gvbtn}">판정 실행</button>'
            f'<button onclick="__gvCancel()" style="padding:8px 12px;border:1px solid #dde3ea;'
            f'border-radius:8px;background:#fff;font-size:13px;cursor:pointer">중단</button>'
            f'<span id="gv_live" style="font-size:12px;color:#d64545"></span> '
            f'<span id="gv_tm" style="font-size:11px;color:#8a94a3"></span></div>'
            f'<div id="gv_jtg" style="font-size:12.5px;margin:2px 0 4px;display:none">'
            f'<b>판정 대상</b> <span style="color:#8a94a3">골든 산정에 넣지 않은 챔버도 판정할 수 있습니다 — '
            f'챔버 골든이 없으면 설비 통합 골든으로 판정</span><br>'
            f'설비 <span id="gv_jeq"></span> 챔버 <span id="gv_jdv"></span></div>'
            f'<details id="gv_advd" style="font-size:12px;margin:2px 0"><summary style="cursor:pointer;'
            f'color:#5b6675">고급 설정(σ밴드 배수·길이 임계·스무딩·스텝별 α·설비/챔버/센서 반영 — 바꾸면 결과 즉시 갱신)</summary>'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:4px 0">'
            f'σ밴드 배수 k <input id="gv_k" value="2.0" onchange="__gvLive()" style="width:44px;{_in9}"> '
            f'길이 임계 β <input id="gv_b" value="0.3" onchange="__gvLive()" style="width:44px;{_in9}"> '
            f'사전 스무딩 창 <input id="gv_w" value="0" style="width:36px;{_in9}">'
            f'<span style="color:#8a94a3">(스무딩 창은 재실행 필요)</span></div>'
            f'<div id="gv_astp" style="margin:2px 0"></div>'
            f'<div id="gv_lvs" style="margin:2px 0"></div>'
            f'<div id="gv_rules" style="margin:4px 0;font-size:12px">'
            f'<div style="margin:2px 0"><b>W 봉우리(출구 온도)</b> <span id="gv_rw_on" style="color:#8a94a3"></span> — 되떨어짐 깊이 '
            f'중≥ <input id="gv_rw_t1" value="0.5" onchange="__gvLive()" style="width:44px;{_in9}"> ℃ · '
            f'상≥ <input id="gv_rw_t2" value="1.5" onchange="__gvLive()" style="width:44px;{_in9}"> ℃ · '
            f'대상 스텝 <input id="gv_rw_st" value="4" onchange="__gvLive()" style="width:36px;{_in9}"> · '
            f'<label><input type="checkbox" id="gv_rw_cb" onchange="__gvLive()" style="vertical-align:-2px"> 면적 규칙과 결합(큰 쪽)</label></div>'
            f'<div style="margin:2px 0"><b>아치 기울기(전후 차압)</b> <span id="gv_ra_on" style="color:#8a94a3"></span> — '
            f'기울기 = (최고값 − 최저값) ÷ 두 점 사이 포인트 수 <span style="color:#8a94a3">(원본 값 그대로 · 최고값은 최저점 이전, 정상=양수)</span> · '
            f'중≤ <input id="gv_ra_t1" value="0.03" onchange="__gvLive()" style="width:60px;{_in9}"> · '
            f'상≤ <input id="gv_ra_t2" value="0.01" onchange="__gvLive()" style="width:60px;{_in9}"> '
            f'<span style="color:#8a94a3">(이 값 이하 = 덜 내려감 → 불량 · 최고값 = 최저값이면 무조건 상)</span> · '
            f'대상 스텝 <input id="gv_ra_st" value="1,3" onchange="__gvLive()" style="width:48px;{_in9}"> · '
            f'<label><input type="checkbox" id="gv_ra_cb" onchange="__gvLive()" style="vertical-align:-2px"> 면적 규칙과 결합</label>'
            f'<select id="gv_ra_mode" style="display:none"><option value="abs" selected>abs</option><option value="ratio">ratio</option></select>'
            f'<input type="checkbox" id="gv_ra_lr" style="display:none"></div></div>'
            f'<div id="gv_rflt" style="font-size:12.5px;margin:4px 0;display:none">'
            f'<b>반영 설비·챔버</b> <span style="color:#8a94a3">체크된 설비·챔버의 웨이퍼만 집계</span><br>'
            f'설비 <span id="gv_rfeq"></span> 챔버 <span id="gv_rfdv"></span></div>'
            f'<div id="gv_jt" style="font-size:12px;margin:4px 0;display:none"></div></details>'
            f'<details id="gv_vlmd" style="font-size:12px;margin:2px 0"><summary style="cursor:pointer;'
            f'color:#5b6675">VLM 검사 지시(공통·스텝별 — 파형 특징과 검사 방법을 직접 적어 VLM에 전달)</summary>'
            f'<div style="margin:4px 0">공통 지시 <textarea id="gv_vn" rows="2" placeholder="예) 모든 스텝에서 '
            f'골든보다 낙폭이 20% 이상 작으면 불량" style="width:96%;{_in9}"></textarea></div>'
            f'<div id="gv_vns" style="margin:2px 0"></div>'
            f'<a href="#" onclick="__gvSysPrompt();return false;" style="color:#2456a6;font-size:11.5px">'
            f'[시스템 지시 보기]</a><pre id="gv_sysp" style="display:none;white-space:pre-wrap;font-size:11px;'
            f'color:#5b6675;background:#f6f8fb;border:1px solid #eef1f5;border-radius:8px;padding:6px 8px"></pre>'
            f'</details></div>'
            # ── ④ 결과 ──
            f'<div style="{_gvh}" onclick="__gvToggle(4)">④ 결과 '
            f'<span id="gv_b4" style="{_gvb}">대기</span>'
            f'<span style="margin-left:auto;font-weight:400;color:#8a94a3;font-size:12px">'
            f'추세 · 판정표 · 웨이퍼 카드</span></div>'
            f'<div id="gv_s4" style="display:none;padding:4px 6px 8px">'
            f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:0 0 4px;'
            f'font-size:12.5px">집계 <select id="gv_ag" onchange="__gvRender()" style="{_in9}">'
            f'<option value="day">일</option><option value="week">주</option>'
            f'<option value="month">월</option></select> '
            f'<button onclick="__gvResults()" style="{_gvlt}">다시 판정</button>'
            f'<button onclick="__gvRepWave()" style="{_gvlt}" title="현재 판정 기준으로 센서별 하(정상) 평균 파형과 '
            f'중·상(불량) 평균 파형을 그립니다">대표 그래프 생성</button>'
            f'<button onclick="__gvCsvOut()" style="{_gvex}">판정 CSV</button>'
            f'<button onclick="__gvExportFull()" style="{_gvex}">HTML 내보내기</button>'
            f'<button onclick="__gvToAnl()" style="{_gvex}" title="불량 웨이퍼 일자를 분석자료·'
            f'패턴 인식 달력의 ▲마크로 보냅니다">불량 일자를 달력 마크로</button>'
            f'<button onclick="__gvSaveCond()" style="{_gvlt}">분석 조건 저장</button>'
            f'<button onclick="__gvLoadCond()" style="{_gvlt}">조건 불러오기</button>'
            f'<span id="gv_wci" style="font-size:12px;color:#8a94a3;margin-left:6px"></span></div>'
            f'<div id="gv_bars" style="display:none"></div>'
            f'<div id="gv_lvbars" style="margin-top:4px"></div>'
            f'<div id="gv_rep" style="margin-top:6px"></div>'
            f'<div id="gv_rank" style="font-size:12.5px;margin:6px 0"></div>'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;'
            f'margin:4px 0"><label><input type="checkbox" id="gv_ngonly" onchange="__gvWlist(window.__GV.bucket)"> '
            f'불량만</label> 검색 <input id="gv_q" placeholder="웨이퍼·DSV·센서" '
            f'oninput="__gvWlist(window.__GV.bucket)" style="width:160px;{_in9}">'
            f'<span id="gv_lsum" style="color:#5b6675"></span></div>'
            f'<div id="gv_list" style="font-size:12.5px;margin:4px 0;max-height:420px;overflow:auto"></div>'
            f'<div id="gv_cards" style="margin-top:6px"></div></div></div>')
        body += (f"<script>window.__TOOLVER='{TOOL_VER}';</script>")
        body += ("<script>var GOLD=" + _json.dumps(gold_js, separators=(",", ":")) + ";</script>")
        if embed is not None:                          # 정적 내장: 메타+파형을 통째로 담는다
            metas, pay = embed
            body += ("<script>var EMBED={metas:" + _json.dumps(metas, separators=(",", ":"))
                     + ",wafers:" + _json.dumps(pay, separators=(",", ":")) + "};</script>")
    if not groups:
        body = "<p><b>집계할 양품 웨이퍼가 없습니다.</b> ok-max 를 올리거나 데이터를 확인하세요.</p>"
    tab_js = """<script>
function showTab(k){var ps=document.querySelectorAll('.tabpane'),bs=document.querySelectorAll('.tabs button');
 for(var i=0;i<ps.length;i++){ps[i].className='tabpane'+(i===k?' on':'');}
 for(var j=0;j<bs.length;j++){bs[j].className=(j===k?'on':'');}
}
function _fmt(v){var a=Math.abs(v);return v.toFixed(a>=100?2:(a>=1?3:5));}
/* 공용 호버: d={m,s,(w),pl,pt,iw,ih,lo,hi,L} */
function wireZoom(svg,pl,pt,iw,ih,L,onRange){
 /* 드래그로 타임포인트 구간 선택 → onRange(i0,i1) · 더블클릭 → onRange(null) 원복 */
 var ns='http://www.w3.org/2000/svg';
 var sel=document.createElementNS(ns,'rect');
 sel.setAttribute('y',pt);sel.setAttribute('height',ih);
 sel.setAttribute('fill','#2456a6');sel.setAttribute('fill-opacity','0.12');
 sel.setAttribute('stroke','#2456a6');sel.setAttribute('stroke-dasharray','3 2');
 sel.style.display='none';svg.appendChild(sel);
 var x0=null;
 function toI(mx){var i=Math.round((mx-pl)/iw*(L-1));return i<0?0:(i>L-1?L-1:i);}
 svg.addEventListener('mousedown',function(ev){var r=svg.getBoundingClientRect();
  x0=ev.clientX-r.left;sel.style.display='';sel.setAttribute('x',x0);sel.setAttribute('width',0);ev.preventDefault();});
 svg.addEventListener('mousemove',function(ev){if(x0==null)return;
  var r=svg.getBoundingClientRect(),x1=ev.clientX-r.left;
  sel.setAttribute('x',Math.min(x0,x1));sel.setAttribute('width',Math.abs(x1-x0));});
 function up(ev){if(x0==null)return;
  var r=svg.getBoundingClientRect(),x1=ev.clientX-r.left;
  sel.style.display='none';
  var i0=toI(Math.min(x0,x1)),i1=toI(Math.max(x0,x1));x0=null;
  if(i1-i0>=4)onRange(i0,i1);}
 svg.addEventListener('mouseup',up);
 svg.addEventListener('mouseleave',function(){x0=null;sel.style.display='none';});
 svg.addEventListener('dblclick',function(){onRange(null);});}
function wireHover(svg,d){
 var wrap=svg.parentNode,tip=wrap.querySelector('.hvtip'),
     vx=svg.querySelector('.hvx'),c=svg.querySelector('.hvc'),cw=svg.querySelector('.hvw');
 function move(ev){var r=svg.getBoundingClientRect();var mx=ev.clientX-r.left;
  var i=Math.round((mx-d.pl)/d.iw*(d.L-1)); if(i<0)i=0; if(i>d.L-1)i=d.L-1;
  var x=d.pl+d.iw*(d.L>1?i/(d.L-1):0.5);
  function Y(v){return d.pt+d.ih*(1-(v-d.lo)/(d.hi-d.lo));}
  vx.setAttribute('x1',x);vx.setAttribute('x2',x);vx.style.display='';
  c.setAttribute('cx',x);c.setAttribute('cy',Y(d.m[i]));c.style.display='';
  var t='타임포인트 '+((d.x0||0)+i)+'<br>'+(d.bl||'기준 평균')+' '+_fmt(d.m[i])
   +(d.nosig?'':' · σ '+_fmt(d.s[i]));
  if(d.w){var wv=d.w[i];
   if(cw){cw.setAttribute('cx',x);cw.setAttribute('cy',Y(wv));cw.style.display='';}
   var dev=wv-d.m[i], nd=(d.s[i]>1e-12)?(dev/d.s[i]):0;
   t+='<br>실제 '+_fmt(wv)+' · 편차 '+(dev>=0?'+':'')+_fmt(dev)+' ('+nd.toFixed(1)+'σ)';}
  else{t+='<br>범위 '+_fmt(d.m[i]-d.s[i])+' ~ '+_fmt(d.m[i]+d.s[i]);}
  if(d.p&&d.p[i]!=null){t+='<br>모델 기대 '+_fmt(d.p[i])
   +(d.w?' · 실제-기대 '+((d.w[i]-d.p[i])>=0?'+':'')+_fmt(d.w[i]-d.p[i]):'');}
  if(d.ex)for(var e9=0;e9<d.ex.length;e9++)
   t+='<br><span style="color:'+d.ex[e9].color+'">\u25c6</span> '+d.ex[e9].label+' '+_fmt(d.ex[e9].v[i]);
  tip.innerHTML=t;tip.style.display='';
  var tx=x+12; if(tx>r.width-170)tx=x-170;
  tip.style.left=tx+'px'; tip.style.top=(d.pt+4)+'px';}
 function out(){tip.style.display='none';vx.style.display='none';c.style.display='none';if(cw)cw.style.display='none';}
 svg.addEventListener('mousemove',move);svg.addEventListener('mouseleave',out);}
(function(){var svgs=document.querySelectorAll('svg.ch');
 for(var k=0;k<svgs.length;k++){var d=JSON.parse(svgs[k].getAttribute('data-ch'));
  var card=svgs[k].closest('.card');
  if(card&&d.m){var t=card.querySelector('.t'),nm=t?t.childNodes[0].textContent.trim():'';
   card.parentNode.replaceChild(__chart(nm,d.m,d.s,null,null),card);}
  else wireHover(svgs[k],d);}})();
(function(){if(!window.EMBED)return;                 /* 정적 내장 리포트 초기화 */
 var inp=document.getElementById('wfolder');
 if(inp){inp.parentNode.style.display='none';}
 WMETA=EMBED.metas; __mkFilters();
 document.getElementById('wfilters').style.display='';
 document.getElementById('wlist').style.display=''; __applyF();})();
/* ── 웨이퍼 분석 탭(서버 모드): 폴더 스캔 → 전체 목록 + 유형/날짜/판정 필터 ── */
var WMETA=[];
function __export(){
 var msg=document.getElementById('wmsg'), btn=document.getElementById('wexp');
 btn.disabled=true; msg.textContent='리포트 생성 중\u2026 (전량 다시 읽어 저장 -- 콘솔에 진행 표시, 수 분~수십 분)';
 fetch('/api/export',{method:'POST'})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){msg.textContent='저장 완료: '+j.dir+' ('+j.files.length+'개 파일 · '+j.n+'장)';})
 .catch(function(e){msg.textContent='저장 실패: '+e.message;})
 .then(function(){btn.disabled=false;});}
function __gmodal(makeFn){
 /* 그래프 확대 팝업: 가운데 큰 화면, 배경 클릭/X/ESC 로 닫기 */
 var ov=document.createElement('div');
 ov.style.cssText='position:fixed;inset:0;background:rgba(20,28,40,0.55);z-index:9000;'
  +'display:flex;align-items:center;justify-content:center';
 var box=document.createElement('div');
 box.style.cssText='background:#fff;border-radius:14px;padding:14px 18px;max-width:96vw;'
  +'max-height:92vh;overflow:auto;box-shadow:0 8px 40px rgba(0,0,0,0.35);position:relative';
 var x=document.createElement('button');
 x.textContent='\u2715';
 x.style.cssText='position:absolute;top:8px;right:10px;border:none;background:none;'
  +'font-size:16px;color:#5b6675;cursor:pointer';
 function close(){document.body.removeChild(ov);document.removeEventListener('keydown',esc);}
 function esc(ev){if(ev.key==='Escape')close();}
 x.onclick=close;
 ov.addEventListener('mousedown',function(ev){if(ev.target===ov)close();});
 document.addEventListener('keydown',esc);
 box.appendChild(x); box.appendChild(makeFn()); ov.appendChild(box);
 document.body.appendChild(ov);}
var GSAV=[];                                   /* 불러온 저장 골든(문서 배열) */
var GXCOL=['#0e8074','#8e5b3a','#b03a8e','#4a6b1f','#616e7d'];
function __grs(v,L){ if(v.length===L)return v;   /* 선형 리샘플 */
 var o=new Array(L);
 for(var i=0;i<L;i++){var x=(v.length-1)*(L>1?i/(L-1):0);
  var a=Math.floor(x),b=Math.min(v.length-1,a+1),t=x-a;
  o[i]=v[a]*(1-t)+v[b]*t;} return o;}
function __gextras(recipe,sensor,L){
 var out=[];
 for(var k=0;k<GSAV.length;k++){var d=GSAV[k];
  if(d.recipe&&recipe&&d.recipe!==recipe)continue;
  var v=d.mean&&d.mean[sensor]; if(!v)continue;
  out.push({label:d.name,vals:__grs(v,L),color:GXCOL[k%GXCOL.length]});}
 return out;}
function __gslist(inpId,boxId){
 var st=document.getElementById(inpId).value.trim(),box=document.getElementById(boxId);
 if(!st){box.innerHTML='<span style="color:#c0392b;font-size:12px">저장소 폴더를 입력하세요</span>';return;}
 box.innerHTML='<span style="font-size:12px;color:#5b6675">목록 조회 중\u2026</span>';
 fetch('/api/glist?store='+encodeURIComponent(st))
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  if(!j.items.length){box.innerHTML='<span style="font-size:12px;color:#5b6675">저장된 골든이 없습니다</span>';return;}
  var h='<div style="font-size:12px;margin:4px 0">';
  for(var i=0;i<j.items.length;i++){var it=j.items[i];
   h+='<label style="margin-right:12px;white-space:nowrap"><input type="checkbox" class="gsv-'+boxId+'" value="'+it.file+'"> '
    +it.name+' <span style="color:#8a94a3">('+(it.recipe||'전체')+(it.label?' \u00b7 '+it.label:'')+')</span></label>';}
  h+='</div><button data-i="'+inpId+'" data-b="'+boxId+'" onclick="__gsapply(this.dataset.i,this.dataset.b)" style="padding:4px 12px;border:1px solid #2456a6;'
   +'border-radius:7px;background:#fff;color:#2456a6;font-size:12px;cursor:pointer">선택 골든 겹쳐 그리기</button> '
   +'<button onclick="GSAV=[];__grefresh()" style="padding:4px 12px;border:1px solid #8a94a3;'
   +'border-radius:7px;background:#fff;color:#5b6675;font-size:12px;cursor:pointer">겹침 해제</button>';
  box.innerHTML=h;})
 .catch(function(e){box.innerHTML='<span style="color:#c0392b;font-size:12px">'+e.message+'</span>';});}
function __gsapply(inpId,boxId){
 var st=document.getElementById(inpId).value.trim();
 var cbs=document.querySelectorAll('.gsv-'+boxId+':checked');
 var files=[]; for(var i=0;i<cbs.length;i++)files.push(cbs[i].value);
 if(!files.length){alert('불러올 골든을 선택하세요');return;}
 fetch('/api/gload',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({store:st,files:files})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){GSAV=j.items;__grefresh();})
 .catch(function(e){alert('불러오기 실패: '+e.message);});}
function __grefresh(){
 if(window.__GXJ)__gxrender(window.__GXJ);
 if(window.__lastW)__ovDraw(window.__lastW);}
var GXCAT=['#2456a6','#d64545','#2e7d32','#e08a00','#7b1fa2','#00838f','#8e5b3a','#c2185b',
 '#4a6b1f','#5c6bc0','#ad1457','#00695c'];
function __gxall(rc,on){
 window.__GXVIS=window.__GXVIS||{};
 var R=window.__GXJ.recipes[rc], v=window.__GXVIS[rc]=window.__GXVIS[rc]||{b:{},e:{}};
 for(var b=0;b<R.buckets.length;b++)v.b[b]=on;
 var ei=0;
 for(var k=0;k<GSAV.length;k++){if(GSAV[k].recipe&&GSAV[k].recipe!==rc)continue;v.e[ei]=on;ei++;}
 __gxrender(window.__GXJ);}
function __gxrecolor(rc){
 /* 지금 보이는(체크된) 라인들에만 구분되는 색을 새로 배정 — 그라데이션이
    비슷해 구분 안 될 때 사용. [전체 선택] 후 다시 누르면 전체 재배색. */
 window.__GXCOLR=window.__GXCOLR||{};
 var R=window.__GXJ.recipes[rc];
 var vst=(window.__GXVIS||{})[rc]||{b:{},e:{}};
 var co=window.__GXCOLR[rc]={b:{},e:{}};
 var k=0;
 for(var b=0;b<R.buckets.length;b++)if(vst.b[b]!==false)co.b[b]=GXCAT[(k++)%GXCAT.length];
 var ei=0;
 for(var g=0;g<GSAV.length;g++){if(GSAV[g].recipe&&GSAV[g].recipe!==rc)continue;
  if(vst.e[ei]!==false)co.e[ei]=GXCAT[(k++)%GXCAT.length]; ei++;}
 __gxrender(window.__GXJ);}
function __gxsen(el){
 window.__GXSEN=window.__GXSEN||{};
 var v=window.__GXSEN[el.dataset.rc]=window.__GXSEN[el.dataset.rc]||{};
 v[el.dataset.sn]=el.checked;
 if(window.__GXJ)__gxrender(window.__GXJ);}
function __gxsenall(rc,on){
 window.__GXSEN=window.__GXSEN||{};
 var R=window.__GXJ.recipes[rc], v=window.__GXSEN[rc]=window.__GXSEN[rc]||{};
 for(var k=0;k<R.sensors.length;k++)v[R.sensors[k]]=on;
 __gxrender(window.__GXJ);}
function __gxvis(el){
 var rc=el.dataset.rc,k=el.dataset.k,i=parseInt(el.dataset.i,10);
 window.__GXVIS=window.__GXVIS||{};
 var v=window.__GXVIS[rc]=window.__GXVIS[rc]||{b:{},e:{}};
 v[k][i]=el.checked;
 if(window.__GXJ)__gxrender(window.__GXJ);}
function __gxrender(j){
 var out=document.getElementById('gxout');
 out.innerHTML='';
 window.__GXVIS=window.__GXVIS||{};
 var esc7=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 Object.keys(j.recipes).sort().forEach(function(rc){
  var R=j.recipes[rc];
  var vst=window.__GXVIS[rc]||{b:{},e:{}};
  var co=(window.__GXCOLR||{})[rc]||{b:{},e:{}};
  var vis={b:[],e:[],cb:[],ce:[]};
  var sec=document.createElement('div');
  var lg='<span style="margin-right:8px">'
   +'<button data-rc="'+esc7(rc)+'" onclick="__gxall(this.dataset.rc,true)" style="padding:2px 8px;border:1px solid #8a94a3;border-radius:6px;background:#fff;color:#5b6675;font-size:11px;cursor:pointer">전체 선택</button> '
   +'<button data-rc="'+esc7(rc)+'" onclick="__gxall(this.dataset.rc,false)" style="padding:2px 8px;border:1px solid #8a94a3;border-radius:6px;background:#fff;color:#5b6675;font-size:11px;cursor:pointer">전체 해제</button> '
   +'<button data-rc="'+esc7(rc)+'" onclick="__gxrecolor(this.dataset.rc)" title="지금 체크된 라인들에 구분되는 색을 새로 배정" style="padding:2px 8px;border:1px solid #7b1fa2;border-radius:6px;background:#fff;color:#7b1fa2;font-size:11px;cursor:pointer">색상 다시 넣기</button></span>';
  for(var b=0;b<R.buckets.length;b++){
   var onb=vst.b[b]!==false; vis.b[b]=onb;
   var cb=co.b[b]||__gxcolor(b,R.buckets.length); vis.cb[b]=cb;
   lg+='<label style="margin-right:10px;white-space:nowrap;cursor:pointer">'
    +'<input type="checkbox" '+(onb?'checked':'')+' data-rc="'+esc7(rc)+'" data-k="b" data-i="'+b+'" onchange="__gxvis(this)" style="vertical-align:-2px"> '
    +'<span style="display:inline-block;width:14px;height:3px;background:'+cb+';vertical-align:3px"></span> '
    +esc7(R.buckets[b].label)+' ('+R.buckets[b].n+'장)</label>';}
  var ei=0;
  for(var k9=0;k9<GSAV.length;k9++){var dv=GSAV[k9];
   if(dv.recipe&&dv.recipe!==rc)continue;
   var one=vst.e[ei]!==false; vis.e[ei]=one;
   var ce=co.e[ei]||GXCOL[k9%GXCOL.length]; vis.ce[ei]=ce;
   lg+='<label style="margin-right:10px;white-space:nowrap;cursor:pointer">'
    +'<input type="checkbox" '+(one?'checked':'')+' data-rc="'+esc7(rc)+'" data-k="e" data-i="'+ei+'" onchange="__gxvis(this)" style="vertical-align:-2px"> '
    +'<span style="display:inline-block;width:14px;height:0;border-top:3px dashed '+ce+';vertical-align:3px"></span> '
    +esc7(dv.name)+' <span style="color:#8a94a3">(저장 골든)</span></label>';
   ei++;}
  var opt='';
  for(var b2=0;b2<R.buckets.length;b2++)opt+='<option value="'+b2+'">'+esc7(R.buckets[b2].label)+' ('+R.buckets[b2].n+'장)</option>';
  var sst=(window.__GXSEN||{})[rc]||{};
  var sh='<b>센서</b> <button data-rc="'+esc7(rc)+'" onclick="__gxsenall(this.dataset.rc,true)" style="padding:1px 7px;border:1px solid #8a94a3;border-radius:6px;background:#fff;color:#5b6675;font-size:11px;cursor:pointer">전체</button> '
   +'<button data-rc="'+esc7(rc)+'" onclick="__gxsenall(this.dataset.rc,false)" style="padding:1px 7px;border:1px solid #8a94a3;border-radius:6px;background:#fff;color:#5b6675;font-size:11px;cursor:pointer">해제</button> ';
  for(var s9=0;s9<R.sensors.length;s9++){var snm=R.sensors[s9];
   var ons=sst[snm]!==false;
   sh+='<label style="margin-right:8px;white-space:nowrap;cursor:pointer"><input type="checkbox" '+(ons?'checked':'')
    +' data-rc="'+esc7(rc)+'" data-sn="'+esc7(snm)+'" onchange="__gxsen(this)" style="vertical-align:-2px"> '+esc7(snm)+'</label>';}
  sec.innerHTML='<h3 style="margin:14px 0 4px">'+esc7(rc)+'</h3>'
   +'<div style="font-size:11.5px;color:#5b6675;margin-bottom:4px">'+lg+'</div>'
   +'<div style="font-size:12px;margin-bottom:8px">구간 저장: <select id="gxsv-'+esc7(rc)+'" style="font-size:12px;padding:2px 4px">'+opt+'</select> '
   +'이름 <input id="gxsn-'+esc7(rc)+'" placeholder="예: 7월상순_dsv5" style="width:140px;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:3px 6px"> '
   +'<button data-rc="'+esc7(rc)+'" onclick="__gxsave(this.dataset.rc)" style="padding:3px 12px;border:1px solid #00695c;border-radius:7px;background:#fff;color:#00695c;font-size:12px;cursor:pointer">이 구간 저장</button></div>'
   +'<div style="font-size:11.5px;color:#33404f;margin-bottom:8px;display:flex;flex-wrap:wrap;gap:2px 4px;max-height:80px;overflow:auto;border:1px solid #eef1f5;border-radius:8px;padding:6px 8px">'+sh+'</div>';
  var grid=document.createElement('div'); grid.className='grid';
  for(var k2=0;k2<R.sensors.length;k2++){var snx=R.sensors[k2];
   if(sst[snx]===false)continue;
   grid.appendChild(__gxchart(snx,R.len,R.buckets,__gextras(rc,snx,R.len),vis,{orig_len:R.orig_len}));}
  sec.appendChild(grid); out.appendChild(sec);});}
function __gxsave(rc){
 var st=document.getElementById('gxst').value.trim();
 if(!st){alert('저장소 폴더를 먼저 입력하세요 (저장 골든 줄)');return;}
 var selv=document.getElementById('gxsv-'+rc).value;
 var nm=document.getElementById('gxsn-'+rc).value.trim();
 if(!nm){alert('저장 이름을 입력하세요 (예: 6월_dsv5)');return;}
 var R=window.__GXJ.recipes[rc], bk=R.buckets[parseInt(selv,10)];
 fetch('/api/gsave',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({store:st,name:nm,recipe:rc,len:R.len,mean:bk.m,
    meta:{label:bk.label+' ('+bk.n+'장)',source:'gcross'}})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){alert('저장 완료: '+j.path+(j.overwritten?' (덮어씀)':''));})
 .catch(function(e){alert('저장 실패: '+e.message);});}
function __gxcolor(i,n){ if(n<2)return 'hsl(210,70%,45%)';
 var h=210-190*(i/(n-1)); return 'hsl('+h.toFixed(0)+',72%,45%)'; }
function __gxchart(name,L,buckets,extras,vis,opts){
 extras=extras||[]; opts=opts||{};
 function shB(i){return !(vis&&vis.b&&vis.b[i]===false);}
 function shE(i){return !(vis&&vis.e&&vis.e[i]===false);}
 var holder=document.createElement('div');
 function build(a,b){
  var W=opts.W||560,H=opts.H||220,pl=52,pr=10,pt=26,pb=24,iw=W-pl-pr,ih=H-pt-pb,Lr=b-a+1;
  var lo=1/0,hi=-1/0,nb=buckets.length;
  for(var b0=0;b0<nb;b0++){var m=buckets[b0].m[name]; if(!m||!shB(b0))continue;
   for(var i=a;i<=b;i++){if(m[i]<lo)lo=m[i]; if(m[i]>hi)hi=m[i];}}
  for(var e0=0;e0<extras.length;e0++){if(!shE(e0))continue;var ex0=extras[e0].vals;
   for(var i0b=a;i0b<=b;i0b++){if(ex0[i0b]<lo)lo=ex0[i0b]; if(ex0[i0b]>hi)hi=ex0[i0b];}}
  if(!(hi-lo>1e-12)){lo-=1;hi+=1;} var pad=(hi-lo)*0.06; lo-=pad; hi+=pad;
  function X(i){return pl+iw*(Lr>1?(i-a)/(Lr-1):0.5);}
  function Y(v){return pt+ih*(1-(v-lo)/(hi-lo));}
  var grid='';
  for(var g=0;g<5;g++){var v=lo+(hi-lo)*g/4,gy=Y(v);
   grid+='<line x1="'+pl+'" y1="'+gy.toFixed(1)+'" x2="'+(W-pr)+'" y2="'+gy.toFixed(1)+'" stroke="#e5e9f0"/>'
    +'<text x="'+(pl-6)+'" y="'+(gy+4).toFixed(1)+'" text-anchor="end" font-size="10" fill="#8a94a3">'+v.toPrecision(3)+'</text>';}
  var paths='';
  function colB(i){return (vis&&vis.cb&&vis.cb[i])||__gxcolor(i,nb);}
  function colE(i){return (vis&&vis.ce&&vis.ce[i])||extras[i].color;}
  for(var b2=0;b2<nb;b2++){var m2=buckets[b2].m[name]; if(!m2||!shB(b2))continue;
   var d2='M'; for(var i2=a;i2<=b;i2++)d2+=(i2>a?' L':'')+X(i2).toFixed(1)+' '+Y(m2[i2]).toFixed(1);
   paths+='<path d="'+d2+'" fill="none" stroke="'+colB(b2)+'" stroke-width="1.6" stroke-opacity="0.9"/>';}
  for(var e2=0;e2<extras.length;e2++){if(!shE(e2))continue;var ev=extras[e2].vals;
   var de='M'; for(var i3=a;i3<=b;i3++)de+=(i3>a?' L':'')+X(i3).toFixed(1)+' '+Y(ev[i3]).toFixed(1);
   paths+='<path d="'+de+'" fill="none" stroke="'+colE(e2)+'" stroke-width="1.8" stroke-dasharray="6 3"/>';}
  var zoomed=(a>0||b<L-1);
  var html='<div class="card"><div class="t">'+name
   +(zoomed?' <span class="n" style="color:#2456a6">확대 '+a+'~'+b+' (더블클릭=원복)</span>':'')
   +(opts.noexp?'':' <span class="expb" title="크게 보기" style="float:right;cursor:pointer;color:#8a94a3;font-size:14px">\u26f6</span>')+'</div>'
   +'<div class="chwrap" style="position:relative">'
   +'<svg class="gx" width="'+W+'" height="'+H+'">'+grid+paths
   +'<line class="hvx" x1="0" y1="'+pt+'" x2="0" y2="'+(pt+ih)+'" stroke="#8a94a3" stroke-dasharray="3 3" style="display:none"/>'
   +'<text x="'+pl+'" y="'+(H-6)+'" font-size="10" fill="#8a94a3">'+a+'</text>'
   +'<text x="'+(W-pr)+'" y="'+(H-6)+'" text-anchor="end" font-size="10" fill="#8a94a3">타임포인트 '+b+(opts.orig_len&&opts.orig_len>b?' (표시용 리샘플 \u00b7 원본 '+opts.orig_len+')':'')+'</text>'
   +'</svg><div class="hvtip" style="display:none"></div></div></div>';
  holder.innerHTML=html;
  var svg=holder.querySelector('svg'),tip=holder.querySelector('.hvtip'),vx=svg.querySelector('.hvx');
  svg.addEventListener('mousemove',function(ev){
   var r=svg.getBoundingClientRect(),mx=ev.clientX-r.left;
   var i=a+Math.round((mx-pl)/iw*(Lr-1)); if(i<a)i=a; if(i>b)i=b;
   var x=X(i);
   vx.setAttribute('x1',x);vx.setAttribute('x2',x);vx.style.display='';
   var t='타임포인트 '+i;
   for(var b3=0;b3<nb;b3++){var m3=buckets[b3].m[name]; if(!m3||!shB(b3))continue;
    t+='<br><span style="color:'+colB(b3)+'">\u25a0</span> '+buckets[b3].label+' : '+_fmt(m3[i]);}
   for(var e3=0;e3<extras.length;e3++){if(!shE(e3))continue;
    t+='<br><span style="color:'+colE(e3)+'">\u25c6</span> '+extras[e3].label+' : '+_fmt(extras[e3].vals[i]);}
   tip.innerHTML=t;tip.style.display='';
   var tx=x+12; if(tx>r.width-190)tx=x-190;
   tip.style.left=tx+'px'; tip.style.top=(pt+4)+'px';});
  svg.addEventListener('mouseleave',function(){tip.style.display='none';vx.style.display='none';});
  wireZoom(svg,pl,pt,iw,ih,Lr,function(i0,i1){
   if(i0==null)build(0,L-1); else build(a+i0,a+i1);});
  var eb=holder.querySelector('.expb');
  if(eb)eb.addEventListener('click',function(){
   __gmodal(function(){
    return __gxchart(name,L,buckets,extras,vis,
     {W:Math.min(window.innerWidth-140,1160),H:Math.min(window.innerHeight-160,540),noexp:1});});});}
 build(0,L-1);
 return holder;}
function __gcross(){
 var btn=document.getElementById('gxgo'),msg=document.getElementById('gxmsg'),out=document.getElementById('gxout');
 var body={folder:document.getElementById('gxf').value.trim(),
  eqp:document.getElementById('gxe').value.trim(),dsv:document.getElementById('gxd').value.trim(),
  recipe:document.getElementById('gxr').value.trim(),months:document.getElementById('gxm').value.trim(),
  bucket_days:parseInt(document.getElementById('gxb').value,10)||5};
 if(!body.folder){msg.textContent='폴더를 입력하세요';return;}
 btn.disabled=true; msg.textContent='읽는 중\u2026 (판정 없이 평균만 — 장수에 따라 수 분)';
 fetch('/api/gcross',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  msg.textContent='웨이퍼 '+j.n_wafers+'장 \u00b7 '+j.bucket_days+'일 단위';
  window.__GXJ=j; __gxrender(j);})
 .catch(function(e){msg.textContent='실패: '+e.message;})
 .then(function(){btn.disabled=false;});}
function __tslist(){
 var st=document.getElementById('tsst').value.trim(),pk=document.getElementById('tspick');
 if(!st){pk.textContent='저장소 폴더를 입력하세요';return;}
 pk.textContent='조회 중\u2026';
 fetch('/api/glist?store='+encodeURIComponent(st))
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  if(!j.items.length){pk.textContent='저장 골든 없음';return;}
  var h='<select id="tsgf" style="font-size:12px;padding:3px 5px;max-width:280px">';
  for(var i=0;i<j.items.length;i++)h+='<option value="'+j.items[i].file+'">'+j.items[i].name+' ('+(j.items[i].recipe||'전체')+(j.items[i].label?' \u00b7 '+j.items[i].label:'')+')</option>';
  pk.innerHTML=h+'</select>';})
 .catch(function(e){pk.textContent=e.message;});}
function __tscan(){
 var btn=document.getElementById('tsgo'),msg=document.getElementById('tsmsg'),out=document.getElementById('tsout');
 var gf=(document.getElementById('tsgf')||{}).value||'';
 var body={folder:document.getElementById('tsf').value.trim(),
  eqp:document.getElementById('tse').value.trim(),dsv:document.getElementById('tsd').value.trim(),
  recipe:document.getElementById('tsr').value.trim(),months:document.getElementById('tsm').value.trim(),
  per_day:parseInt(document.getElementById('tsn').value,10)||0,
  store:document.getElementById('tsst').value.trim(),gfile:gf,
  cache:(document.getElementById('tscache')||{}).value.trim()||''};
 if(!body.folder){msg.textContent='폴더를 입력하세요';return;}
 if(!gf){msg.textContent='기준 골든을 먼저 선택하세요 ([기준 선택])';return;}
 btn.disabled=true; msg.textContent='접수 중\u2026';
 fetch('/api/trendscan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  msg.textContent='백그라운드 실행 중 ('+j.days+'일 \u00b7 '+j.n_total+'장) — 브라우저를 닫아도 계속됩니다';
  if(window.__tsT)clearInterval(window.__tsT);
  window.__tsT=setInterval(__tspoll,3000);})
 .catch(function(e){msg.textContent='실패: '+e.message;btn.disabled=false;});}
function __tspoll(){
 var btn=document.getElementById('tsgo'),msg=document.getElementById('tsmsg');
 fetch('/api/trendscan_status').then(function(r){return r.json();})
 .then(function(st){
  if(st.state==='running'){msg.textContent='진행: '+(st.progress||'\u2026');return;}
  clearInterval(window.__tsT);window.__tsT=null;btn.disabled=false;
  if(st.state==='error'){msg.textContent='실패: '+st.error;return;}
  if(st.state==='done'&&st.result){
   var j=st.result;window.__TSJ=j;
   msg.textContent=j.dates.length+'일 \u00b7 카드 '+j.cards.length+'건 \u00b7 기준: '+j.ref.name;
   document.getElementById('tsnr').style.display='';__tsrender(j);}})
 .catch(function(e){});}
function __tsrender(j){
 var out=document.getElementById('tsout');
 var esc6=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var h='';
 /* 1) 타임라인 스트립: 센서×날짜 강도 히트맵 */
 var sens=Object.keys(j.strip).sort(function(a,b){
  var ma=Math.max.apply(null,Object.values(j.strip[a])),mb=Math.max.apply(null,Object.values(j.strip[b]));
  return mb-ma;}).slice(0,30);
 h+='<div style="font-size:12px;margin-bottom:4px"><b>타임라인</b> <span style="color:#8a94a3">색 농도=이탈 강도(레벨\u00b7볼록 최대 σ) \u00b7 상위 30센서 \u00b7 셀 호버=값</span></div>';
 h+='<div style="overflow:auto;max-height:420px;border:1px solid #eef1f5;border-radius:8px;padding:6px"><table style="border-collapse:collapse;font-size:10px">';
 h+='<tr><td></td>';
 for(var d=0;d<j.dates.length;d++){var lb=(d%Math.max(1,Math.floor(j.dates.length/14))===0)?j.dates[d].slice(5):'';
  h+='<td style="padding:0 1px;color:#8a94a3;white-space:nowrap">'+lb+'</td>';}
 h+='</tr>';
 for(var si=0;si<sens.length;si++){var sn=sens[si];
  h+='<tr><td style="padding:1px 6px 1px 0;white-space:nowrap;color:#33404f;font-size:11px">'+esc6(sn)+'</td>';
  for(var d2=0;d2<j.dates.length;d2++){var v=j.strip[sn][j.dates[d2]];
   var a=(v==null)?0:Math.min(1,Math.max(0,(v-1.5)/6));
   h+='<td title="'+esc6(sn)+' '+j.dates[d2]+' : '+(v==null?'-':v+'\u03c3')+'" style="width:9px;height:13px;background:rgba(214,69,69,'+a.toFixed(2)+');border:1px solid #f4f6f9"></td>';}
  h+='</tr>';}
 h+='</table></div>';
 /* 1.5) 배관 연관 사슬 */
 if(j.chains&&j.chains.length){
  h+='<div style="font-size:12px;background:#eef4fb;border:1px solid #d3e0ef;border-radius:8px;padding:8px 10px;margin:10px 0">'
   +'<div style="font-weight:600;color:#2456a6;margin-bottom:2px">배관 연관 사슬 (같은 시기 시작 \u00b7 기점=최초 시작)</div>';
  for(var ch=0;ch<j.chains.length;ch++){var cc=j.chains[ch];
   h+='<div style="margin:2px 0">사슬 '+(ch+1)+' ('+esc6(cc.start)+'~): <b>기점 '+esc6(j.cards[cc.origin].sensor)+'</b> \u2192 '+esc6(cc.path.join(' \u2192 '))+'</div>';}
  h+='</div>';}
 /* 1.55) Vessel 내부 추정 소견 */
 if(j.vessel_notes&&j.vessel_notes.length){
  h+='<div style="font-size:12px;background:#fdf6ee;border:1px solid #ecd3b5;border-radius:8px;padding:8px 10px;margin:10px 0">'
   +'<div style="font-weight:600;color:#8a5a1e;margin-bottom:2px">Vessel 내부 추정 (T2·T3 정상 + T4 이상 — 간접 추론)</div>';
  for(var vn=0;vn<j.vessel_notes.length;vn++){var v9=j.vessel_notes[vn];
   h+='<div data-i="'+v9.card_idx+'" onclick="__tsdetail(this.dataset.i)" style="margin:2px 0;cursor:pointer">\u25b8 '+esc6(v9.text)+'</div>';}
  h+='</div>';}
 /* 1.6) 패턴 사례 통계(같은 문제 페어링) */
 if(j.groups&&j.groups.length){
  var multi=j.groups.filter(function(g){return g.n_occur>1;});
  h+='<div style="font-size:12px;background:#f3eefb;border:1px solid #ddcdf0;border-radius:8px;padding:8px 10px;margin:10px 0">'
   +'<div style="font-weight:600;color:#6a4fa3;margin-bottom:4px">패턴 사례 통계 — 같은 문제 페어링'
   +(j.cond?' <span style="font-weight:normal;color:#8a94a3">('+esc6(j.cond)+')</span>':'')
   +' \u00b7 그룹 '+j.groups.length+'개, 다회 발생 '+multi.length+'개</div>';
  for(var gi2=0;gi2<j.groups.length;gi2++){var g=j.groups[gi2];
   var locs=g.loc?(' \u00b7 중심 '+Math.round(g.loc.c/6)+'%'):'';
   var occ=g.occurrences.map(function(o){return o.month.slice(5)+'월('+o.start.slice(5)+'\u301c'+String(o.end).slice(5)+' '+o.state+(o.amp!=null?' '+o.amp+'\u03c3':'')+')';}).join(', ');
   h+='<div data-i="'+g.card_idx[0]+'" onclick="__tsdetail(this.dataset.i)" style="margin:3px 0;cursor:pointer;padding:4px 8px;border-radius:6px;background:'+(g.n_occur>1?'#fff':'transparent')+'">'
    +'<b>'+esc6(g.sensor)+'</b> step'+g.step+' \u00b7 '+esc6(g.kind)+'('+esc6(g.direction||'')+locs+') \u2014 '
    +(g.n_occur>1?'<span style="color:#7b1fa2;font-weight:700">'+g.n_occur+'회 발생</span>':'1회')
    +': '+esc6(occ)+'</div>';}
  h+='</div>';}
 /* 1.7) 전 센서×스텝 통계 요약 */
 if(j.stats&&j.stats.length){
  h+='<details style="margin:10px 0"><summary style="font-size:12.5px;font-weight:600;cursor:pointer;color:#33404f">전 센서\u00d7스텝 통계 요약 ('+j.stats.length+'행 \u2014 모든 스텝을 다 본 수치 증거) <button data-x="1" onclick="event.preventDefault();__tscsv()" style="margin-left:8px;padding:2px 10px;border:1px solid #6a4fa3;border-radius:6px;background:#fff;color:#6a4fa3;font-size:11px;cursor:pointer">CSV 저장</button></summary>'
   +'<div style="overflow:auto;max-height:360px;border:1px solid #eef1f5;border-radius:8px;margin-top:6px"><table style="border-collapse:collapse;font-size:11px;white-space:nowrap">'
   +'<tr style="position:sticky;top:0;background:#f4f6f9">'+['센서','스텝','일수','평균레벨σ','최대레벨σ','최악일','유의일수','유의%','평균발생률','최대발생률','최대볼록σ','t'].map(function(x){return '<th style="padding:3px 8px;text-align:left">'+x+'</th>';}).join('')+'</tr>';
  for(var si=0;si<j.stats.length;si++){var r=j.stats[si];
   var hot=Math.abs(r.max_lv)>=3?'background:#fdf2ef;':'';
   h+='<tr style="'+hot+'">'+[r.sensor,'step'+r.step,r.n_days,r.mean_lv,r.max_lv,r.worst_day,r.sig_days,r.sig_pct+'%',r.ratio_avg,r.ratio_max,r.bump_max,r.t].map(function(x){return '<td style="padding:2px 8px;border-top:1px solid #f0f2f6">'+x+'</td>';}).join('')+'</tr>';}
  h+='</table></div></details>';}
 /* 2) 패턴 카드 */
 h+='<div style="font-size:12px;margin:12px 0 4px"><b>패턴 카드 '+j.cards.length+'건</b> <span style="color:#8a94a3">클릭=기준 대비 형성 전/중/최근 곡선 팝업</span></div>';
 h+='<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:8px">';
 var KC2={'레벨 드리프트':'#d84315','볼록 패턴':'#7b1fa2','급변 이벤트':'#c0392b'};
 for(var c2=0;c2<j.cards.length;c2++){var cd=j.cards[c2];
  var sp='';
  var vals=(cd.series||[]).filter(function(x){return x!=null;});
  if(vals.length){var mx=Math.max.apply(null,vals.map(Math.abs))||1;
   sp=(cd.series||[]).map(function(x){return x==null?'\u00b7':('\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588'[Math.min(7,Math.round(Math.abs(x)/mx*7))]);}).join('');}
  var ev='';
  for(var e2=0;e2<(cd.events||[]).length&&e2<3;e2++){var evd=cd.events[e2];
   ev+='<div style="font-size:11px;color:#c0392b">\u25aa '+evd.date+' '+esc6(evd.kind)+' '+(evd.delta>0?'+':'')+evd.delta+'\u03c3 \u2014 '+esc6(evd.label)+'</div>';}
  var rc9='';
  if(cd.recur&&cd.recur.length)rc9='<span style="float:right;font-size:10.5px;background:#7b1fa2;color:#fff;border-radius:9px;padding:1px 8px" title="과거 같은 서명 패턴">재발 '+cd.recur.map(function(r){return r.period;}).join(', ')+'</span>';
  h+='<div class="tscard" data-i="'+c2+'" onclick="__tsdetail(this.dataset.i)" style="border:1px solid #dde3ea;border-left:4px solid '+(KC2[cd.kind]||'#5b6675')+';border-radius:8px;padding:8px 10px;background:#fff;cursor:pointer">'
   +'<div style="font-size:12.5px;font-weight:600">'+esc6(cd.sensor)+' <span style="color:#8a94a3">step'+cd.step+'</span>'+rc9+'</div>'
   +'<div style="font-size:12px">'+esc6(cd.kind)+' \u00b7 '+esc6(cd.direction)+' \u00b7 '+esc6(cd.start)+'\u2192 \u00b7 상태: <b>'+esc6(cd.state)+'</b>'+(cd.amp!=null?' \u00b7 '+cd.amp+'\u03c3':'')+'</div>'
   +'<div style="font-family:monospace;font-size:12px;color:#5b6675;letter-spacing:1px">'+sp+'</div>'+ev+'</div>';}
 h+='</div>';
 out.innerHTML=h;}
function __tsnarrate(){
 var btn=document.getElementById('tsnr'),out=document.getElementById('tsout');
 btn.disabled=true;
 var box=document.createElement('div'); box.id='tsnarbox';
 var old=document.getElementById('tsnarbox'); if(old)old.parentNode.removeChild(old);
 box.innerHTML='<div style="padding:12px;color:#5b6675">종합 서사 생성 중\u2026 (상위 카드 이미지 VLM 검토 + LLM 서사 + 토론 — 수 분\u301c십수 분, 콘솔 [tsnar] 진행)</div>';
 out.insertBefore(box,out.firstChild);
 fetch('/api/ts_narrate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({top:8,vlm:true})})
 .then(function(r){return r.json().then(function(x){if(!r.ok)throw new Error(x.detail||'실패');return x;});})
 .then(function(x){
  var esc5=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var ex='';
  if(x.vlm&&x.vlm.length){
   ex='<details style="margin-bottom:8px" open><summary style="font-size:12px;color:#1f6f8b;cursor:pointer;font-weight:600">VLM 시각 검토 소견 ('+x.vlm.length+'건)</summary>';
   for(var v=0;v<x.vlm.length;v++)
    ex+='<div style="font-size:12px;margin:4px 0"><b>'+esc5(x.vlm[v].sensor)+' step'+x.vlm[v].step+'</b> ('+esc5(x.vlm[v].kind)+'): '+esc5(x.vlm[v].note)+'</div>';
   ex+='</details>';}
  __renderNar(box,x,'추세 스캔 종합',ex);})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;padding:10px">종합 서사 실패: '+e.message+'</div>';})
 .then(function(){btn.disabled=false;});}
function __sdlist(){
 var st=document.getElementById('sdst').value.trim(),pk=document.getElementById('sdpick');
 if(!st){pk.textContent='저장소 폴더를 입력하세요';return;}
 pk.textContent='조회 중…';
 fetch('/api/glist?store='+encodeURIComponent(st))
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  if(!j.items.length){pk.textContent='저장 골든 없음';return;}
  var h='<select id="sdgf" style="font-size:12px;padding:3px 5px;max-width:280px">';
  for(var i=0;i<j.items.length;i++)h+='<option value="'+j.items[i].file+'">'+j.items[i].name+' ('+(j.items[i].recipe||'전체')+(j.items[i].label?' · '+j.items[i].label:'')+')</option>';
  pk.innerHTML=h+'</select>';})
 .catch(function(e){pk.textContent=e.message;});}
function __sdrun(){
 var msg=document.getElementById('sdmsg'),btn=document.getElementById('sdgo');
 var gf=(document.getElementById('sdgf')||{}).value||'';
 var body={folder:document.getElementById('sdf').value.trim(),
  eqp:document.getElementById('sde').value.trim(),dsv:document.getElementById('sdv').value.trim(),
  recipe:document.getElementById('sdr').value.trim(),months:document.getElementById('sdm').value.trim(),
  store:document.getElementById('sdst').value.trim(),gfile:gf,
  cache:(document.getElementById('sdcache')||{}).value.trim()||''};
 if(!body.folder){msg.textContent='폴더를 입력하세요';return;}
 if(!gf){msg.textContent='기준 골든을 선택하세요 ([기준 선택] 먼저)';return;}
 btn.disabled=true;msg.textContent='집계 시작…';
 fetch('/api/stepdiff',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body)})
 .then(function(r){return r.json().then(function(j){return {ok:r.ok,j:j};});})
 .then(function(x){
  if(!x.ok){btn.disabled=false;msg.textContent='실패: '+((x.j&&x.j.detail)||'');return;}
  msg.textContent=x.j.n_days+'일 · '+x.j.n_wafers+'장 백그라운드 집계 중…';
  if(window.__sdT)clearInterval(window.__sdT);
  window.__sdT=setInterval(__sdpoll,3000);})
 .catch(function(e){btn.disabled=false;msg.textContent='오류: '+e;});}
function __sdpoll(){
 var msg=document.getElementById('sdmsg'),btn=document.getElementById('sdgo');
 fetch('/api/stepdiff_status').then(function(r){return r.json();})
 .then(function(st){
  if(st.state==='running'){msg.textContent='진행: '+(st.progress||'…');return;}
  clearInterval(window.__sdT);window.__sdT=null;btn.disabled=false;
  if(st.state==='error'){msg.textContent='실패: '+st.error;return;}
  if(st.state==='done'&&st.result){
   msg.textContent='웨이퍼 '+st.result.n_wafers+'장 · 센서 '+st.result.sensors.length+'개 · 기준: '+st.result.ref.name;
   __sdinit(st.result);}})
 .catch(function(e){});}
window.__SDMODE='wafer';window.__SDOFF=0;
function __sdtipInit(){
 if(window.__SDTIP)return;
 var tp=document.createElement('div');
 tp.id='sdtip';
 tp.style.cssText='position:fixed;display:none;z-index:9999;background:#1f2733;color:#fff;'
  +'font-size:12px;padding:4px 8px;border-radius:6px;pointer-events:none;max-width:420px';
 document.body.appendChild(tp);window.__SDTIP=tp;
 document.addEventListener('mousemove',function(e){
  var el=e.target,t=el&&el.getAttribute&&el.getAttribute('data-tip');
  if(!t){tp.style.display='none';return;}
  var t2=el.getAttribute('data-tip2');
  tp.textContent=t+(t2?'  ·  '+t2:'');
  tp.style.display='block';
  tp.style.left=Math.min(window.innerWidth-320,e.clientX+14)+'px';
  tp.style.top=(e.clientY+16)+'px';});
 document.addEventListener('click',function(e){
  var el=e.target;
  if(el&&el.getAttribute&&el.getAttribute('data-wid'))
   __sdwave(el.getAttribute('data-wid'),el.getAttribute('data-sn'));});}
function __sdwave(wid,sn){
 fetch('/api/sd_wave',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({id:wid,sensor:sn})})
 .then(function(r){return r.json().then(function(x){if(!r.ok)throw new Error(x.detail||'실패');return x;});})
 .then(function(x){
  __gmodal(function(){
   var wrap=document.createElement('div');
   wrap.className='dlroot';
   var cap=document.createElement('div');
   cap.style.cssText='font-size:13px;font-weight:700;margin:0 0 6px;color:#33404f';
   cap.innerHTML=__dlBtns(x.sensor+'_'+x.id+'_원본','')
    +'<span>'+String(x.sensor+' — '+x.id+' 원본 파형 (원본 '+x.T+'포인트'
    +(x.T!==x.L?' · 표시 '+x.L:'')+')').replace(/</g,'&lt;')+'</span>';
   var lg=document.createElement('div');
   lg.style.cssText='font-size:11.5px;color:#5b6675;margin:0 0 4px';
   lg.textContent='파란 실선=골든(정상 구간 평균'+(x.aligned?' · 스텝 경계 정렬됨':'')+') · 빨강=이 웨이퍼 원본 · 드래그=확대, 더블클릭=원복 · 마우스=값';
   wrap.appendChild(cap);wrap.appendChild(lg);
   wrap.appendChild(__chart(x.sensor+' — '+x.id,
    x.golden||x.wafer.map(function(){return null;}),null,x.wafer,null,[],
    {W:Math.min(window.innerWidth-160,1200),H:Math.min(window.innerHeight-220,520),
     baseLabel:'골든(정상 구간 평균)',wLabel:x.id+' 원본'}));
   return wrap;});})
 .catch(function(e){alert('원본 파형 조회 실패: '+e.message);});}
function __sdchk(boxId,on){var cs=document.getElementById(boxId).querySelectorAll('input[type=checkbox]');
 for(var i=0;i<cs.length;i++)cs[i].checked=!!on; __sdfetch(true);}
function __sdinit(j){
 window.__SDJ=j;window.__SDMODE='wafer';window.__SDOFF=0;
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 /* 센서 체크박스: 기본 = P/T + 가상센서만 체크(히터류는 해제 상태로 존재) */
 var sb=document.getElementById('sdsen'),h='';
 for(var i=0;i<j.sensors.length;i++){var sn=j.sensors[i];
  var isv=/^d[pt]_/i.test(sn), on=isv||__mpIsPT(sn);
  h+='<label style="display:inline-block;margin-right:10px;cursor:pointer'+(isv?';font-weight:700;color:#6a4fa3':'')+'">'
   +'<input type="checkbox" class="sdck" value="'+esc(sn)+'"'+(on?' checked':'')+' onchange="__sdfetch(true)" style="vertical-align:-2px"> '
   +esc(sn)+'</label>';}
 sb.innerHTML=h;
 /* IO(밸브) 표시 체크박스 — 모든 IO, 기본은 32/33/41 만 */
 var ib=document.getElementById('sdio'),hio='';
 var iodef={'9000003':1,'9000004':1,'9000009':1};
 var ion=j.io_disp_names||[];
 for(i=0;i<ion.length;i++){
  hio+='<label style="display:inline-block;margin-right:8px;cursor:pointer">'
   +'<input type="checkbox" class="sdiock" value="'+esc(ion[i])+'"'+(iodef[ion[i]]?' checked':'')
   +' onchange="__sdfetch(true)" style="vertical-align:-2px"> '+esc(ion[i])+'</label>';}
 ib.innerHTML=hio||'<span style="color:#8a94a3">IO 없음</span>';
 /* 날짜 계층: 월 > 주(월내 ISO주) > 일 — 부모 체크 시 하위 일괄 */
 var tree={},d,i2;
 for(i2=0;i2<j.dates.length;i2++){d=j.dates[i2];
  var mo=d.slice(0,7);
  var dt=new Date(d+'T00:00:00');
  var wk=Math.ceil((dt.getDate()+((new Date(d.slice(0,8)+'01T00:00:00')).getDay()))/7);
  if(!tree[mo])tree[mo]={};
  if(!tree[mo][wk])tree[mo][wk]=[];
  tree[mo][wk].push(d);}
 var db=document.getElementById('sddt'),ht='';
 var mos=Object.keys(tree).sort();
 for(i2=0;i2<mos.length;i2++){var mo2=mos[i2];
  ht+='<div><label style="cursor:pointer;font-weight:600"><input type="checkbox" checked class="sdmo" data-mo="'+mo2+'" style="vertical-align:-2px"> '+mo2+'</label>';
  var wks=Object.keys(tree[mo2]).sort(function(a,b){return a-b;});
  for(var wi=0;wi<wks.length;wi++){var wk2=wks[wi],ds=tree[mo2][wk2];
   ht+='<div style="margin-left:16px"><label style="cursor:pointer"><input type="checkbox" checked class="sdwk" data-mo="'+mo2+'" data-wk="'+wk2+'" style="vertical-align:-2px"> '+wk2+'주</label> ';
   for(var di=0;di<ds.length;di++)
    ht+='<label style="cursor:pointer;color:#5b6675"><input type="checkbox" checked class="sdday" data-mo="'+mo2+'" data-wk="'+wk2+'" value="'+ds[di]+'" style="vertical-align:-2px"> '+ds[di].slice(8)+'</label> ';
   ht+='</div>';}
  ht+='</div>';}
 db.innerHTML=ht;
 db.querySelectorAll('.sdmo').forEach(function(c){c.onchange=function(){
  db.querySelectorAll('[data-mo="'+c.getAttribute('data-mo')+'"]').forEach(function(x){x.checked=c.checked;});
  __sdfetch(true);};});
 db.querySelectorAll('.sdwk').forEach(function(c){c.onchange=function(){
  db.querySelectorAll('.sdday[data-mo="'+c.getAttribute('data-mo')+'"][data-wk="'+c.getAttribute('data-wk')+'"]')
   .forEach(function(x){x.checked=c.checked;});
  __sdfetch(true);};});
 db.querySelectorAll('.sdday').forEach(function(c){c.onchange=function(){__sdfetch(true);};});
 document.getElementById('sdctl').style.display='';
 __sdfindings(j);
 __sdfetch(true);}
function __sdfindings(j){
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var fd=j.findings||{},hs=fd.holds||[],js=fd.jumps||[],vs=fd.violations||[],ph=fd.phys||[],box=document.getElementById('sdfind');
 if(!hs.length&&!js.length&&!vs.length&&!ph.length){box.innerHTML='';return;}
 var un=function(sn){return /^d?p/i.test(sn)?'bar':'\u2103';};
 var h='<details open style="border:1px solid #eef1f5;border-radius:8px;padding:8px 10px">'
  +'<summary style="font-size:12.5px;font-weight:600;cursor:pointer">이상 구간 추림 '
  +'<span style="font-weight:400;color:#8a94a3">'+(ph.length?'물리 위반 '+ph.length+'건 \u00b7 ':'')+(vs.length?'\u2605부호 위반 '+vs.length+'건 \u00b7 ':'')+'지속 이탈 '+hs.length+'건 \u00b7 급변 '+js.length+'건 \u00b7 행 클릭=해당 기간만 보기</span></summary>';
 var i,x;
 for(i=0;i<ph.length;i++){x=ph[i];
  var arch=x.kind&&x.kind.indexOf('아치')===0;
  h+='<div class="sdjmp" data-d0="'+x.d0+'" data-d1="'+x.d1+'" style="font-size:12px;padding:2px 4px;cursor:pointer;'
   +'background:'+(arch?'#fdecea':'#fff3e0')+';border-radius:6px;margin:1px 0;font-weight:700;color:'+(arch?'#8b0000':'#8a5a00')+'">'
   +'[물리 위반 \u00b7 '+esc(x.kind)+'] '+esc(x.sensor||'')
   +(x.step!=null?' step'+x.step:'')+(x.io?' (밸브 '+esc(x.io)+')':'')
   +' \u2014 '+x.d0.slice(5)+'~'+x.d1.slice(5)+' ('+x.days+'일)'
   +(x.kind==='밸브 타이밍'?' 열림 시점 골든 대비 '+x.dt_pct+'% 어긋남'
     :' 위반 최대 '+Math.round((x.ratio_peak||0)*100)+'%')
   +(x.depth_g!=null?' \u00b7 골든 아치 깊이 '+x.depth_g:'')
   +(x.ongoing?' \u2605진행 중':'')+'</div>';}
 for(i=0;i<vs.length;i++){x=vs[i];
  if(x.golden_neg){
   h+='<div style="font-size:12px;padding:2px 4px;background:#fff6e5;border-radius:6px;margin:1px 0;color:#8a5a00;font-weight:700">'
    +'[\u26a0\uace8\ub4e0 \uc74c\uc555] '+esc(x.sensor)+' step'+x.step
    +' \u2014 \uae30\uc900 \uace8\ub4e0 \uc790\uccb4\uac00 \uc74c\uc555('+x.g0+'bar) \u2014 \uc815\uc0c1 \uad6c\uac04 \ub370\uc774\ud130/\uc13c\uc11c \ud655\uc778 \ud544\uc694</div>';
   continue;}
  h+='<div class="sdjmp" data-d0="'+x.d0+'" data-d1="'+x.d1+'" style="font-size:12px;padding:2px 4px;cursor:pointer;'
   +'background:#fdecea;border-radius:6px;margin:1px 0;color:#8b0000;font-weight:700">'
   +'[\u2605\ubd80\ud638 \uc704\ubc18] '+esc(x.sensor)+' step'+x.step+' \u2014 '+x.d0.slice(5)+'~'+x.d1.slice(5)
   +' ('+x.days+'\uc77c) \uc6d0\uc2dc \ucd5c\uc800 '+x.min_raw+'bar &lt; 0 \u2014 \uace8\ub4e0\uc740 \uc774 \uc2a4\ud15d \ub0b4\ub0b4 \uc591\uc218(\ucd5c\uc800 +'+(x.g_min!=null?x.g_min:'?')+'bar) \u2014 \uc5ed\ub958 \uc2e0\ud638'
   +(x.ongoing?' \u2605\uc9c4\ud589 \uc911':'')+'</div>';}
 for(i=0;i<Math.min(hs.length,25);i++){x=hs[i];
  h+='<div class="sdjmp" data-d0="'+x.d0+'" data-d1="'+x.d1+'" style="font-size:12px;padding:2px 0;cursor:pointer'
   +(x.virtual?';color:#6a4fa3;font-weight:600':'')+'">'
   +'[지속] '+esc(x.sensor)+' step'+x.step+' \u2014 '+x.d0.slice(5)+'~'+x.d1.slice(5)
   +' ('+x.days+'\uc77c) \ud3c9\uade0 '+(x.mean>0?'+':'')+x.mean+un(x.sensor)
   +' \u00b7 \ucd5c\ub300 '+(x.peak>0?'+':'')+x.peak+un(x.sensor)
   +(x.ongoing?' <span style="color:#c0392b">\u2605\uc9c4\ud589 \uc911</span>':'')+'</div>';}
 for(i=0;i<Math.min(js.length,15);i++){x=js[i];
  h+='<div class="sdjmp" data-d0="'+x.date+'" data-d1="'+x.date+'" style="font-size:12px;padding:2px 0;cursor:pointer'
   +(x.virtual?';color:#6a4fa3;font-weight:600':'')+'">'
   +'[\uae09\ubcc0] '+esc(x.sensor)+' step'+x.step+' \u2014 '+x.date.slice(5)
   +' \uc804\uc77c \ub300\ube44 '+(x.delta>0?'+':'')+x.delta+un(x.sensor)+'</div>';}
 h+='</details>';
 box.innerHTML=h;
 box.querySelectorAll('.sdjmp').forEach(function(el){el.onclick=function(){
  __sdjump(el.getAttribute('data-d0'),el.getAttribute('data-d1'));};});}
function __sdjump(d0,d1){
 var db=document.getElementById('sddt');
 db.querySelectorAll('.sdday').forEach(function(c){c.checked=(c.value>=d0&&c.value<=d1);});
 db.querySelectorAll('.sdmo,.sdwk').forEach(function(c){c.checked=false;});
 __sdfetch(true);}
function __sdsety(btn){
 var m=btn.getAttribute('data-y');
 window.__SDYMODE=m;
 document.getElementById('sdym').querySelectorAll('button').forEach(function(b){
  var on=b.getAttribute('data-y')===m;
  b.style.background=on?'#1f6f8b':'#fff'; b.style.color=on?'#fff':'#1f6f8b';});
 __sdfetch(true);}
function __sdpmode(btn){
 var m=btn.getAttribute('data-p');
 window.__SDPMODE=m;
 document.getElementById('sdpm').querySelectorAll('button').forEach(function(b){
  var on=b.getAttribute('data-p')===m;
  b.style.background=on?'#7b1fa2':'#fff'; b.style.color=on?'#fff':'#7b1fa2';});
 if(document.getElementById('sdpat').innerHTML)__sdpatterns();}
window.__VZMODE='overlay';window.__VZGM='day';
function __vzgmode(btn){
 var g=btn.getAttribute('data-g');window.__VZGM=g;
 document.getElementById('vzgm').querySelectorAll('button').forEach(function(b){
  var on=b.getAttribute('data-g')===g;
  b.style.background=on?'#1f6f8b':'#fff';b.style.color=on?'#fff':'#1f6f8b';});
 document.getElementById('vzgu').textContent=(g==='count')?'장':(g==='hour'?'시간':'일');}
/* ── 두 기간 직접 비교: A(기준) vs B(대상) 평균 곡선 + 센서×스텝 랭킹 ── */
window.__VZCMP={A:null,B:null};
function __vzcmpPick(which){
 var d=[];document.querySelectorAll('#vzdt .vzday:checked').forEach(function(c){d.push(c.value);});
 if(!d.length){document.getElementById('vzcmpmsg').textContent='기간(날짜)을 먼저 체크하세요';return;}
 window.__VZCMP[which]={days:d,d0:d[0],d1:d[d.length-1]};
 var A=window.__VZCMP.A,B=window.__VZCMP.B;
 document.getElementById('vzcmpchip').innerHTML=
  (A?'<b style="color:#2456a6">A</b> '+A.d0+'~'+A.d1+' ('+A.days.length+'일) ':'')
  +(B?' <b style="color:#c0392b">B</b> '+B.d0+'~'+B.d1+' ('+B.days.length+'일)':'');
 document.getElementById('vzcmpmsg').textContent=which+' 지정';}
function __vzcmpRun(){
 var msg=document.getElementById('vzcmpmsg'),out=document.getElementById('vzcmpout');
 var A=window.__VZCMP.A,B=window.__VZCMP.B,sel=__vzsel();
 if(!A||!B){msg.textContent='A·B 기간을 각각 지정하세요';return;}
 if(!sel.sn.length){msg.textContent='센서를 체크하세요';return;}
 out.innerHTML='';msg.textContent='두 기간 로드 중… (각 최대 200장)';
 fetch('/api/viz_cmp',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({sensors:sel.sn,datesA:A.days,datesB:B.days})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(x){
  msg.textContent='A '+x.nA+'장 · B '+x.nB+'장';
  var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var ss=__vzstepsel(),gON=(document.getElementById('vzg')||{}).checked;
  var h='';
  if((x.rank||[]).length){
   h+='<div class="card" style="padding:8px 12px;margin:0 0 8px"><div style="font-size:12.5px;font-weight:700;margin:0 0 4px">'
    +'센서×스텝 차이 랭킹 <span style="font-weight:400;color:#8a94a3;font-size:11.5px">'
    +'B('+esc(x.periodB[0])+'~'+esc(x.periodB[1])+') − A('+esc(x.periodA[0])+'~'+esc(x.periodA[1])+')'
    +' · z=A 산포 대비 · 행 클릭=그 센서만 보기</span></div>';
   for(var i=0;i<Math.min(x.rank.length,20);i++){var rk=x.rank[i];
    h+='<div class="vzcmprow" data-sn="'+esc(rk.sensor)+'" style="font-size:12px;padding:1px 3px;cursor:pointer'
     +(rk.virtual?';color:#6a4fa3;font-weight:600':'')+'">'+(i+1)+'. '+esc(rk.sensor)+' step'+rk.step
     +' — A '+rk.a+' → B '+rk.b+' (Δ '+(rk.delta>0?'+':'')+rk.delta+', z='+rk.z+')</div>';}
   h+='</div>';}
  out.innerHTML=h;
  for(var q=0;q<x.sensors.length;q++){var sn=x.sensors[q];
   var holder=document.createElement('div');holder.className='card';
   holder.style.cssText='padding:8px 12px;margin:0 0 8px';
   var dk='vzcmp_'+sn;
   var rows=[['idx','A','B','골든']];
   for(var r2=0;r2<x.curves[sn].A.length;r2++)
    rows.push([r2,x.curves[sn].A[r2],x.curves[sn].B[r2],x.golden[sn][r2]]);
   window.__DLCSV[dk]=rows;
   var cap=document.createElement('div');cap.style.cssText='font-size:12.5px;font-weight:700;margin:0 0 4px';
   cap.innerHTML=__dlBtns('비교_'+sn,dk)+'<span>'+esc(sn)
    +' — <span style="color:#2456a6">A '+esc(x.periodA[0])+'~'+esc(x.periodA[1])+'('+x.nA+'장)</span>'
    +' vs <span style="color:#c0392b">B '+esc(x.periodB[0])+'~'+esc(x.periodB[1])+'('+x.nB+'장)</span>'
    +(gON?' · 골든(회색)':'')+(ss.all?'':' · step '+ss.list.join(',')+' 확대')+'</span>';
   holder.appendChild(cap);
   var extras=[{label:'B',vals:__vzslice(x.curves[sn].B,x.steps,ss),color:'#c0392b'}];
   if(gON)extras.push({label:'골든',vals:__vzslice(x.golden[sn],x.steps,ss),color:'#b9c2cf'});
   holder.appendChild(__chart(sn,__vzslice(x.curves[sn].A,x.steps,ss),null,null,null,extras,
    {W:Math.min(window.innerWidth-140,1240),H:300,notp:1,
     lg:[{label:'A '+x.periodA[0]+'~'+x.periodA[1],color:'#2456a6'}].concat(
        extras.map(function(e){return {label:(e.label==='B'?('B '+x.periodB[0]+'~'+x.periodB[1]):e.label),
                                      color:e.color};}))}));
   out.appendChild(holder);}
  out.querySelectorAll('.vzcmprow').forEach(function(el){el.onclick=function(){
   var s=el.getAttribute('data-sn');
   document.querySelectorAll('#vzsen .vzck').forEach(function(c){c.checked=(c.value===s);});
   __vzcmpRun();};});})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
/* ── 분석 프리셋: 입력칸+선택 체크+모드를 한 세트로 저장/복원 ── */
function __psCtx(scope){
 var st=(scope==='vz')?(document.getElementById('vzst')||{}).value
                      :(document.getElementById('sdst')||{}).value;
 var ca=(scope==='vz')?'':((document.getElementById('sdcache')||{}).value||'');
 return {scope:scope,store:(st||'').trim(),cache:(ca||'').trim()};}
function __psChkGet(sel){var a=[];document.querySelectorAll(sel).forEach(function(c){
 if(c.checked)a.push(c.value);});return a;}
function __psChkSet(sel,list){
 var s={},i;for(i=0;i<(list||[]).length;i++)s[list[i]]=1;
 document.querySelectorAll(sel).forEach(function(c){c.checked=!!s[c.value];});}
function __psVal(id){var e=document.getElementById(id);return e?e.value:'';}
function __psSet(id,v){var e=document.getElementById(id);if(e&&v!=null)e.value=v;}
function __psCollect(scope){
 if(scope==='vz')return {
  f:{vzf:__psVal('vzf'),vze:__psVal('vze'),vzv:__psVal('vzv'),vzr:__psVal('vzr'),
     vzm:__psVal('vzm'),vzst:__psVal('vzst'),vzb:__psVal('vzb'),vzrpt:__psVal('vzrpt'),
     vzgf:((document.getElementById('vzgf')||{}).value||'')},
  sen:__psChkGet('#vzsen .vzck'),step:__psChkGet('#vzstep .vzst'),
  day:__psChkGet('#vzdt .vzday'),
  waf:Object.keys(window.__VZCHK||{}).slice(0,5000),
  mode:window.__VZMODE||'overlay',gm:window.__VZGM||'day',
  gold:!!(document.getElementById('vzg')||{}).checked};
 return {
  f:{sdf:__psVal('sdf'),sde:__psVal('sde'),sdv:__psVal('sdv'),sdr:__psVal('sdr'),
     sdm:__psVal('sdm'),sdst:__psVal('sdst'),sdcache:__psVal('sdcache'),
     sdev_csv:__psVal('sdev_csv'),sdgf:((document.getElementById('sdgf')||{}).value||'')},
  sen:__psChkGet('#sdsen .sdck'),io:__psChkGet('#sdio .sdiock'),
  day:__psChkGet('#sddt .sdday'),
  mode:window.__SDMODE||'wafer',ymode:window.__SDYMODE||'raw',
  pmode:window.__SDPMODE||'prev'};}
function __psApply(scope,d){
 var k,f=d.f||{};
 for(k in f)__psSet(k,f[k]);
 var pk=function(pid,sid,gf){var e=document.getElementById(pid);
  if(e&&gf)e.innerHTML='<select id="'+sid+'"><option value="'+gf+'">'+gf+'</option></select>';};
 if(scope==='vz'){
  pk('vzpick','vzgf',f.vzgf);
  window.__VZGM=d.gm||'day';
  var gb=document.getElementById('vzgm');
  if(gb)gb.querySelectorAll('button').forEach(function(b){
   var on=b.getAttribute('data-g')===window.__VZGM;
   b.style.background=on?'#1f6f8b':'#fff';b.style.color=on?'#fff':'#1f6f8b';});
  var gu2=document.getElementById('vzgu');
  if(gu2)gu2.textContent=(window.__VZGM==='count')?'장':(window.__VZGM==='hour'?'시간':'일');
  __psChkSet('#vzsen .vzck',d.sen);__psChkSet('#vzstep .vzst',d.step);
  __psChkSet('#vzdt .vzday',d.day);
  if(d.waf&&window.__VZALL&&window.__VZALL.length){
   window.__VZCHK={};for(k=0;k<d.waf.length;k++)window.__VZCHK[d.waf[k]]=1;
   if(window.__vzwRender)__vzwRender();}
  window.__VZMODE=d.mode||'overlay';
  var mb=document.getElementById('vzmode0');
  if(mb)mb.querySelectorAll('button').forEach(function(b){
   var on=b.getAttribute('data-m')===window.__VZMODE;
   b.style.background=on?'#2456a6':'#fff';b.style.color=on?'#fff':'#2456a6';});
  var g=document.getElementById('vzg');if(g)g.checked=(d.gold!==false);
  return;}
 pk('sdpick','sdgf',f.sdgf);
 __psChkSet('#sdsen .sdck',d.sen);__psChkSet('#sdio .sdiock',d.io);
 __psChkSet('#sddt .sdday',d.day);
 window.__SDMODE=d.mode||'wafer';window.__SDYMODE=d.ymode||'raw';
 window.__SDPMODE=d.pmode||'prev';
 var seg=function(id,attr,val,col){var e=document.getElementById(id);if(!e)return;
  e.querySelectorAll('button').forEach(function(b){
   var on=b.getAttribute(attr)===val;
   b.style.background=on?col:'#fff';b.style.color=on?'#fff':col;});};
 seg('sdmode','data-m',window.__SDMODE,'#2456a6');
 seg('sdym','data-y',window.__SDYMODE,'#1f6f8b');
 seg('sdpm','data-p',window.__SDPMODE,'#7b1fa2');}
function __psList(scope){
 var cx=__psCtx(scope),box=document.getElementById(scope==='vz'?'vzpsbox':'sdpsbox');
 fetch('/api/preset_list?scope='+scope+'&store='+encodeURIComponent(cx.store)
  +'&cache='+encodeURIComponent(cx.cache))
 .then(function(r){return r.json();})
 .then(function(j){
  var h='<select id="'+(scope==='vz'?'vzpssel':'sdpssel')+'" style="font-size:11.5px;padding:2px 4px;max-width:180px">';
  for(var i=0;i<j.items.length;i++)h+='<option value="'+j.items[i].name+'">'+j.items[i].name+'</option>';
  h+='</select>';
  box.innerHTML=j.items.length?h:'<span style="color:#8a94a3;font-size:11.5px">저장된 프리셋 없음</span>';
  box.setAttribute('data-path',j.path);})
 .catch(function(e){box.textContent='목록 실패: '+e.message;});}
function __psSave(scope){
 var cx=__psCtx(scope),nm=(__psVal(scope==='vz'?'vzpsname':'sdpsname')||'').trim();
 var msg=document.getElementById(scope==='vz'?'vzpsmsg':'sdpsmsg');
 if(!nm){msg.textContent='프리셋 이름을 입력하세요';return;}
 fetch('/api/preset_save',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({name:nm,scope:scope,store:cx.store,cache:cx.cache,
   data:__psCollect(scope)})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){msg.textContent='저장'+(j.overwritten?'(덮어씀)':'')+': '+j.path;
  __psList(scope);})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __psLoad(scope){
 var cx=__psCtx(scope),sel=document.getElementById(scope==='vz'?'vzpssel':'sdpssel');
 var msg=document.getElementById(scope==='vz'?'vzpsmsg':'sdpsmsg');
 if(!sel||!sel.value){msg.textContent='프리셋을 선택하세요([목록])';return;}
 fetch('/api/preset_load',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({name:sel.value,scope:scope,store:cx.store,cache:cx.cache})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){__psApply(scope,j.data||{});
  msg.textContent='불러왔습니다('+j.name+') — '
   +(scope==='vz'?'[웨이퍼 목록 불러오기] → [그래프 그리기]':'[스텝 차이 집계]')+' 실행하세요';})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __psDel(scope){
 var cx=__psCtx(scope),sel=document.getElementById(scope==='vz'?'vzpssel':'sdpssel');
 var msg=document.getElementById(scope==='vz'?'vzpsmsg':'sdpsmsg');
 if(!sel||!sel.value){msg.textContent='삭제할 프리셋을 선택하세요';return;}
 fetch('/api/preset_del',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({name:sel.value,scope:scope,store:cx.store,cache:cx.cache})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(){msg.textContent='삭제했습니다';__psList(scope);})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
/* ── 리포트 강조: 주황=리포트일, 노랑=최초 리포트 3일 전 ── */
window.__VZRPT=null;
function __vzrptLoad(){
 var el=document.getElementById('vzrpt'),msg=document.getElementById('vzrptmsg');
 if(!el.value.trim()){msg.textContent='리포트 CSV 경로를 입력하세요';return;}
 msg.textContent='불러오는 중…';
 fetch('/api/viz_reports',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({csv:el.value.trim(),dsv:(document.getElementById('vzv')||{}).value||''})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  var o={},y={};
  for(var i=0;i<j.orange.length;i++)o[j.orange[i]]=1;
  for(i=0;i<j.yellow.length;i++)y[j.yellow[i]]=1;
  window.__VZRPT={o:o,y:y,labels:j.labels||{}};
  msg.textContent='리포트 '+j.n+'건(DSV '+j.dsv_used.join(',')+') · 주황 '+j.orange.length
   +'일 · 노랑 '+j.yellow.length+'일'+(j.skipped.length?' · 건너뜀 '+j.skipped.length:'')
   +' — [그래프 그리기]로 반영';
  __vzwRender();})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __vzHL(d0,d1){
 /* 카드 기간(d0~d1)이 리포트일과 겹치면 주황, 노랑 구간과 겹치면 노랑 */
 var R=window.__VZRPT; if(!R)return null;
 /* UTC 기준 순회 — 로컬 자정 + toISOString 은 하루 밀린다(KST 시프트 버그) */
 var a=new Date(d0+'T00:00:00Z'),b=new Date((d1||d0)+'T00:00:00Z'),hit=null,lab=[];
 for(var t=new Date(a);t<=b;t.setUTCDate(t.getUTCDate()+1)){
  var s=t.toISOString().slice(0,10);
  if(R.o[s]){hit='o';if(R.labels[s])lab=lab.concat(R.labels[s].map(function(v){return s.slice(5)+' '+v;}));}
  else if(R.y[s]&&hit!=='o')hit='y';}
 return hit?{k:hit,lab:lab}:null;}
/* ── 그래프 내보내기: SVG→PNG(캔버스) / 곡선→CSV. 폐쇄망: 외부 lib 없음 ── */
window.__DLCSV={};
function __dlSave(blob,name){
 var a=document.createElement('a'),u=URL.createObjectURL(blob);
 a.href=u;a.download=name;document.body.appendChild(a);a.click();
 setTimeout(function(){document.body.removeChild(a);URL.revokeObjectURL(u);},400);}
function __dlDecorate(svg){
 /* 저장용 복제본에 제목(기간/센서)과 레전드(색-센서명)를 그려 넣는다.
    화면 카드에는 HTML 로 이미 있으므로 이미지에만 추가한다. */
 var cl=svg.cloneNode(true),NS='http://www.w3.org/2000/svg';
 var meta=svg.__meta||{},W=parseInt(svg.getAttribute('width'),10)||900;
 var H=parseInt(svg.getAttribute('height'),10)||300;
 var lg=meta.lg||[],ttl=meta.title||'';
 if(!ttl&&!lg.length)return cl;
 var wOf=function(t){return 18+Math.max(34,String(t).length*6.6)+12;};
 var rows=[[]],rw=0;
 for(var i=0;i<lg.length;i++){var w1=wOf(lg[i].label);
  if(rw+w1>W-20&&rows[rows.length-1].length){rows.push([]);rw=0;}
  rows[rows.length-1].push(lg[i]); rw+=w1;}
 if(!lg.length)rows=[];
 var HD=(ttl?22:0)+rows.length*15+(rows.length?6:0);
 cl.setAttribute('height',H+HD);
 var g=document.createElementNS(NS,'g');
 g.setAttribute('transform','translate(0,'+HD+')');
 while(cl.firstChild)g.appendChild(cl.firstChild);
 var bg=document.createElementNS(NS,'rect');
 bg.setAttribute('x','0');bg.setAttribute('y','0');
 bg.setAttribute('width',String(W));bg.setAttribute('height',String(H+HD));
 bg.setAttribute('fill','#ffffff');
 cl.appendChild(bg); cl.appendChild(g);
 if(ttl){var t=document.createElementNS(NS,'text');
  t.setAttribute('x','10');t.setAttribute('y','16');t.setAttribute('font-size','12.5');
  t.setAttribute('font-weight','700');t.setAttribute('fill','#26303b');
  t.textContent=ttl; cl.appendChild(t);}
 var y0=(ttl?22:0)+11;
 for(var r=0;r<rows.length;r++){var x=10;
  for(var q=0;q<rows[r].length;q++){var it=rows[r][q];
   var rc=document.createElementNS(NS,'rect');
   rc.setAttribute('x',String(x));rc.setAttribute('y',String(y0+r*15-3));
   rc.setAttribute('width','14');rc.setAttribute('height','3');
   rc.setAttribute('fill',it.color||'#2456a6'); cl.appendChild(rc);
   var lb=document.createElementNS(NS,'text');
   lb.setAttribute('x',String(x+18));lb.setAttribute('y',String(y0+r*15));
   lb.setAttribute('font-size','11');lb.setAttribute('fill','#55606d');
   lb.textContent=String(it.label); cl.appendChild(lb);
   x+=wOf(it.label);}}
 return cl;}
function __dlPng(svg,name){
 if(!svg){alert('그래프를 찾지 못했습니다');return;}
 var cl=__dlDecorate(svg);
 cl.setAttribute('xmlns','http://www.w3.org/2000/svg');
 cl.setAttribute('font-family','Malgun Gothic,Apple SD Gothic Neo,sans-serif');
 var W=parseInt(cl.getAttribute('width'),10)||900,H=parseInt(cl.getAttribute('height'),10)||300;
 var s=new XMLSerializer().serializeToString(cl);
 var img=new Image(),SC=2;
 img.onload=function(){
  var cv=document.createElement('canvas');cv.width=W*SC;cv.height=H*SC;
  var g=cv.getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,cv.width,cv.height);
  g.setTransform(SC,0,0,SC,0,0);g.drawImage(img,0,0);
  if(cv.toBlob)cv.toBlob(function(bl){__dlSave(bl,name+'.png');});
  else __dlSave(new Blob([s],{type:'image/svg+xml'}),name+'.svg');};
 img.onerror=function(){__dlSave(new Blob([s],{type:'image/svg+xml'}),name+'.svg');};
 img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(s);}
function __dlCsvRows(rows,name){
 var out=String.fromCharCode(0xFEFF);
 for(var i=0;i<rows.length;i++)out+=rows[i].join(',')+String.fromCharCode(13,10);
 __dlSave(new Blob([out],{type:'text/csv;charset=utf-8'}),name+'.csv');}
document.addEventListener('click',function(e){
 var el=e.target;
 if(!el||!el.getAttribute)return;
 var cmp=el.getAttribute('data-cmp');
 if(cmp){if(cmp==='go')__vzcmpRun(); else __vzcmpPick(cmp);return;}
 var ps=el.getAttribute('data-ps');
 if(ps){var sc=el.getAttribute('data-scope')||'sd';
  if(ps==='save')__psSave(sc); else if(ps==='list')__psList(sc);
  else if(ps==='load')__psLoad(sc); else if(ps==='del')__psDel(sc);
  return;}
 var kind=el.getAttribute('data-dl');
 if(!kind)return;
 var nm=el.getAttribute('data-name')||'graph';
 if(kind==='set'){__dlSet(el.getAttribute('data-b'),nm,el);return;}
 if(kind==='png'){
  var box=el.closest('.dlroot')||el.closest('.card');
  __dlPng(box?box.querySelector('svg'):null,nm);}
 else if(kind==='csv'){
  var rows=window.__DLCSV[el.getAttribute('data-key')];
  if(rows)__dlCsvRows(rows,nm); else alert('내보낼 데이터가 없습니다');}});
function __dlBtns(nm,key,bidx){
 /* 카드 헤더에 붙일 버튼 HTML(데이터 속성 방식 — 인라인 인자 금지 규칙 준수) */
 var BS='float:right;margin-left:6px;font-size:11px;padding:2px 8px;border:1px solid #dde3ea;'
  +'border-radius:6px;background:#fff;cursor:pointer';
 var s='<button data-dl="png" data-name="'+nm+'" style="'+BS+'">PNG</button>';
 if(key)s+='<button data-dl="csv" data-name="'+nm+'" data-key="'+key+'" style="'+BS+'">CSV</button>';
 if(bidx!=null)s+='<button data-dl="set" data-b="'+bidx+'" data-name="'+nm+'" title="'
  +'이 카드 + 고정 4종(내부블록 온도 / dP 3종 오버레이 step2~5 / 출구 온도 / '
  +'dP front-back step2~5)" style="'+BS
  +';border-color:#2456a6;color:#2456a6">이미지 저장(세트)</button>';
 return s;}
/* ── 고정 세트 저장: 카드 기간·장수 그대로 정해진 그래프를 항상 함께 내려받는다 ── */
window.__FIXSET=[
 {t:'1_t_002',   s:['t_002_PV']},
 {t:'2_dP_3종오버레이', s:['dP_VslF-VslB','dP_VslF-Out','dP_VslB-Out'], st:[2,3,4,5]},
 {t:'3_t_003',       s:['t_003_PV']},
 {t:'4_dP_front-back',  s:['dP_VslF-VslB'],                            st:[2,3,4,5]}];
function __ssFix(list){                     /* 고정 스텝 선택(없으면 현재 보기 그대로) */
 if(!list||!list.length)return null;
 var st={},i;for(i=0;i<list.length;i++)st[String(list[i])]=1;
 return {all:false,set:st,list:list.slice()};}
function __dlfn(t){return String(t).replace(/[\\\\/:*?"<>|]/g,'-').replace(/\\s+/g,' ').trim();}
function __vzsenAvail(){var a=[];
 document.querySelectorAll('#vzsen .vzck').forEach(function(c){a.push(c.value);});return a;}
function __vzresolve(t,av){
 var i,key=function(x){return String(x).toLowerCase().replace(/[^a-z0-9]/g,'');};
 for(i=0;i<av.length;i++)if(av[i]===t)return av[i];
 for(i=0;i<av.length;i++)if(key(av[i])===key(t))return av[i];
 for(i=0;i<av.length;i++)if(key(av[i]).indexOf(key(t))>=0||key(t).indexOf(key(av[i]))>=0)return av[i];
 return null;}
function __dlSet(b,nm,el){
 var ids=(window.__VZB||{})[b]||[];
 if(!ids.length){alert('이 카드의 웨이퍼 목록을 찾지 못했습니다');return;}
 var av=__vzsenAvail(),map={},need=[],miss=[];
 window.__FIXSET.forEach(function(it){it.s.forEach(function(t){
  if(map[t]!==undefined)return;
  var r=__vzresolve(t,av); map[t]=r||null;
  if(r){if(need.indexOf(r)<0)need.push(r);}else if(miss.indexOf(t)<0)miss.push(t);});});
 var card=el.closest('.card');
 var msg=document.getElementById('vzmsg'),bt0=el.textContent,tot=1+window.__FIXSET.length;
 var prog=function(t){el.textContent=t; if(msg)msg.textContent='세트 저장 '+t;};
 var done=function(t){el.textContent=bt0; if(msg)msg.textContent=t;};
 if(card)__dlPng(card.querySelector('svg'),__dlfn(nm));
 if(!need.length){done('고정 세트 센서를 찾지 못했습니다: '+miss.join(', '));return;}
 prog('1/'+tot+' 저장 · 나머지 준비 중…');
 var rng=(window.__VZBD||{})[b],req;
 if(rng){var dd=[],dt0=new Date(rng[0]+'T00:00:00Z'),dt1=new Date(rng[1]+'T00:00:00Z');
  for(var dq=dt0;dq<=dt1;dq.setUTCDate(dq.getUTCDate()+1))dd.push(dq.toISOString().slice(0,10));
  req={dates:dd,sensors:need,group_mode:'day',group_n:0};}   /* 일 단위=캐시 재사용 */
 else req={ids:ids,sensors:need};
 fetch('/api/viz_buckets',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(req)})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(x){
  var bk=(x.buckets||[])[0]; if(!bk)throw new Error('묶음 평균을 만들지 못했습니다');
  var plab=((window.__VZBL||{})[b])||bk.label;   /* 파일명·제목은 카드 기간 라벨 */
  var ss=__vzstepsel(),gEl=document.getElementById('vzg'),gON=gEl?gEl.checked:false;
  var hid=document.getElementById('vzhid');
  if(!hid){hid=document.createElement('div');hid.id='vzhid';
   hid.style.cssText='position:absolute;left:-9999px;top:0;width:1300px';document.body.appendChild(hid);}
  hid.innerHTML='';
  var jobs=[];
  window.__FIXSET.forEach(function(it){
   var sn=[],q;
   it.s.forEach(function(t){var r=map[t]; if(r&&bk.curves[r]&&sn.indexOf(r)<0)sn.push(r);});
   if(!sn.length)return;
   /* 항목별 고정 스텝(dP 2장=step 2~5) — 데이터에 있는 스텝만 확대하고 그대로 표기 */
   var pres=[];
   if(it.st){var sset={};
    for(q=0;q<x.steps.length;q++)sset[String(x.steps[q])]=1;
    for(q=0;q<it.st.length;q++)if(sset[String(it.st[q])])pres.push(it.st[q]);}
   var sfix=__ssFix(pres),ssu=sfix||ss;
   var extras=[];
   for(q=1;q<sn.length;q++)extras.push({label:sn[q],
    vals:__vzslice(bk.curves[sn[q]],x.steps,ssu),color:GXCAT[q%GXCAT.length]});
   if(gON)for(q=0;q<sn.length;q++)extras.push({label:sn[q]+'(골든)',
    vals:__vzslice(x.golden[sn[q]],x.steps,ssu),color:'#b9c2cf'});
   var zt=(sfix?(' \u00b7 step '+pres.join(',')+' 확대'
                 +(pres.length<it.st.length?' (요청 '+it.st.join(',')+' 중 존재분)':''))
           :(it.st?' \u00b7 step '+it.st.join(',')+' 없음(전체 표시)'
             :(ssu.all?'':' \u00b7 step '+ssu.list.join(',')+' 확대')));
   var lgs=[{label:sn[0],color:GXCAT[0]}].concat(extras.map(function(e){
    return {label:e.label,color:e.color};}));
   var ch=__chart(plab+' \u00b7 '+sn.join(' + ')+zt,
    __vzslice(bk.curves[sn[0]],x.steps,ssu),null,null,null,extras,
    {W:1240,H:280,notp:1,noexp:1,lg:lgs});
   hid.appendChild(ch);
   jobs.push({svg:ch.querySelector('svg'),
              name:__dlfn(plab+'_'+it.t
                          +(sfix?'_step'+pres.join('-')
                            :(it.st?'_step'+it.st.join('-')+'없음':'')))});});
  var i=0;
  (function step(){
   if(i>=jobs.length){done('세트 저장 완료 — 이 카드 1장 + 고정 '+jobs.length+'장'
     +(miss.length?' (찾지 못한 센서: '+miss.join(', ')+')':''));
    hid.innerHTML='';return;}
   __dlPng(jobs[i].svg,jobs[i].name); i++;
   prog((i+1)+'/'+tot+' 저장 중…'); setTimeout(step,250);})();})
 .catch(function(e){done('세트 저장 실패: '+e.message);});}
/* ── 웨이퍼 목록 가상 스크롤: 전체를 메모리에 두고 보이는 행만 렌더 ── */
window.__VZALL=[];window.__VZCHK={};
function __vzwInit(items){
 window.__VZALL=items||[];window.__VZCHK={};
 var box=document.getElementById('vzw');
 box.innerHTML='<div id="vzwspace" style="position:relative"></div>';
 box.onscroll=function(){__vzwRender();};
 __vzwRender();}
function __vzwRender(){
 var box=document.getElementById('vzw'),sp=document.getElementById('vzwspace');
 if(!sp)return;
 var all=window.__VZALL,RH=22,per=3;                 /* 한 줄에 3개 */
 var rows=Math.ceil(all.length/per);
 sp.style.height=(rows*RH)+'px';
 var top=box.scrollTop,h=box.clientHeight||132;
 var r0=Math.max(0,Math.floor(top/RH)-3),r1=Math.min(rows,Math.ceil((top+h)/RH)+3);
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var html='';
 for(var r=r0;r<r1;r++){
  html+='<div style="position:absolute;top:'+(r*RH)+'px;left:0;right:0;height:'+RH+'px;white-space:nowrap">';
  for(var k=0;k<per;k++){var i=r*per+k;if(i>=all.length)break;var it=all[i];
   var h2=__vzHL(it.date,it.date);
   html+='<label style="display:inline-block;width:32%;cursor:pointer;overflow:hidden'
    +(h2?';background:'+(h2.k==='o'?'#ffe9cc':'#fff6bf'):'')+'">'
    +'<input type="checkbox" class="vzwck" data-id="'+esc(it.id)+'"'
    +(window.__VZCHK[it.id]?' checked':'')+' style="vertical-align:-2px"> '
    +esc(it.id)+' <span style="color:#8a94a3;font-size:10.5px">'+esc(it.date).slice(5)+'</span></label>';}
  html+='</div>';}
 sp.innerHTML=html;
 sp.querySelectorAll('.vzwck').forEach(function(c){c.onchange=function(){
  var id=c.getAttribute('data-id');
  if(c.checked)window.__VZCHK[id]=1; else delete window.__VZCHK[id];
  __vzwCount();};});
 __vzwCount();}
function __vzwCount(){
 var n=Object.keys(window.__VZCHK).length;
 document.getElementById('vzwn').textContent='(체크 '+n+' / 전체 '+window.__VZALL.length+'장)';}
function __vzwAll(on){
 window.__VZCHK={};
 if(on)for(var i=0;i<window.__VZALL.length;i++)window.__VZCHK[window.__VZALL[i].id]=1;
 __vzwRender();}
function __vzstepsel(){
 var s={},all=true,n=0;
 document.querySelectorAll('#vzstep .vzst').forEach(function(c){
  n++; if(c.checked)s[c.value]=1; else all=false;});
 return {set:s,all:(all||n===0),list:Object.keys(s)};}
function __vzslice(vals,steps,ss){
 /* 체크한 스텝의 포인트만 남겨 붙인다 — y축이 그 구간에 맞게 자동 확대 */
 if(!steps||ss.all)return vals;
 var out=[];
 for(var i=0;i<vals.length;i++)if(ss.set[String(steps[i])])out.push(vals[i]);
 return out.length?out:vals;}
function __vzglist(){
 var st=document.getElementById('vzst').value.trim(),pk=document.getElementById('vzpick');
 if(!st){pk.textContent='저장소 폴더를 입력하세요';return;}
 pk.textContent='조회 중…';
 fetch('/api/glist?store='+encodeURIComponent(st))
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  if(!j.items.length){pk.textContent='저장 골든 없음';return;}
  var h='<select id="vzgf" style="font-size:12px;padding:3px 5px;max-width:260px">';
  for(var i=0;i<j.items.length;i++)h+='<option value="'+j.items[i].file+'">'+j.items[i].name+' ('+(j.items[i].recipe||'전체')+')</option>';
  pk.innerHTML=h+'</select>';})
 .catch(function(e){pk.textContent=e.message;});}
function __vzmode(btn){
 var m=btn.getAttribute('data-m');window.__VZMODE=m;
 document.getElementById('vzmode0').querySelectorAll('button').forEach(function(b){
  var on=b.getAttribute('data-m')===m;
  b.style.background=on?'#2456a6':'#fff';b.style.color=on?'#fff':'#2456a6';});}
function __vzload(fresh){
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var el=function(id){return document.getElementById(id);};
 var days=[];document.querySelectorAll('#vzdt .vzday:checked').forEach(function(c){days.push(c.value);});
 var msg=el('vzmsg');
 var body={dates:days};
 if(fresh){
  if(!el('vzf').value.trim()){msg.textContent='데이터 폴더를 입력하세요';return;}
  var gf=(el('vzgf')||{}).value||'';
  if(!gf){msg.textContent='기준 골든을 선택하세요 ([기준 선택] 먼저)';return;}
  body={folder:el('vzf').value.trim(),eqp:el('vze').value.trim(),dsv:el('vzv').value.trim(),
   recipe:el('vzr').value.trim(),months:el('vzm').value.trim(),
   store:el('vzst').value.trim(),gfile:gf,dates:[]};
  el('vzdt').innerHTML='';el('vzsen').innerHTML='';}
 msg.textContent='목록 조회…';
 fetch('/api/viz_list',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body)})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  msg.textContent='웨이퍼 '+j.total+'장 · 기준: '+(j.ref||'');
  if(!document.getElementById('vzsen').innerHTML){
   var hs='';
   for(var i=0;i<j.sensors.length;i++){var sn=j.sensors[i],isv=/^d[pt]_/i.test(sn);
    hs+='<label style="display:inline-block;margin-right:10px;cursor:pointer'+(isv?';font-weight:700;color:#6a4fa3':'')+'">'
     +'<input type="checkbox" class="vzck" value="'+esc(sn)+'"'+(isv?' checked':'')+' style="vertical-align:-2px"> '+esc(sn)+'</label>';}
   document.getElementById('vzsen').innerHTML=hs;}
  if(!document.getElementById('vzstep').innerHTML&&(j.steps||[]).length){
   var hst='';
   for(var si=0;si<j.steps.length;si++)
    hst+='<label style="display:inline-block;margin-right:8px;cursor:pointer">'
     +'<input type="checkbox" class="vzst" value="'+j.steps[si]+'" checked style="vertical-align:-2px"> step'+j.steps[si]+'</label>';
   document.getElementById('vzstep').innerHTML=hst;}
  if(!document.getElementById('vzdt').innerHTML){
   var tree={},d,i2;
   for(i2=0;i2<j.dates.length;i2++){d=j.dates[i2];
    var mo=d.slice(0,7);
    var dt=new Date(d+'T00:00:00');
    var wk=Math.ceil((dt.getDate()+((new Date(d.slice(0,8)+'01T00:00:00')).getDay()))/7);
    if(!tree[mo])tree[mo]={};if(!tree[mo][wk])tree[mo][wk]=[];tree[mo][wk].push(d);}
   var ht='',mos=Object.keys(tree).sort();
   for(i2=0;i2<mos.length;i2++){var mo2=mos[i2];
    ht+='<div><label style="cursor:pointer;font-weight:600"><input type="checkbox" checked class="vzmo" data-mo="'+mo2+'" style="vertical-align:-2px"> '+mo2+'</label>';
    var wks=Object.keys(tree[mo2]).sort(function(a,b){return a-b;});
    for(var wi=0;wi<wks.length;wi++){var wk2=wks[wi],ds=tree[mo2][wk2];
     ht+='<div style="margin-left:16px"><label style="cursor:pointer"><input type="checkbox" checked class="vzwk" data-mo="'+mo2+'" data-wk="'+wk2+'" style="vertical-align:-2px"> '+wk2+'주</label> ';
     for(var di=0;di<ds.length;di++)
      ht+='<label style="cursor:pointer;color:#5b6675"><input type="checkbox" checked class="vzday" data-mo="'+mo2+'" data-wk="'+wk2+'" value="'+ds[di]+'" style="vertical-align:-2px"> '+ds[di].slice(8)+'</label> ';
     ht+='</div>';}
    ht+='</div>';}
   var db=document.getElementById('vzdt');db.innerHTML=ht;
   db.querySelectorAll('.vzmo').forEach(function(c){c.onchange=function(){
    db.querySelectorAll('[data-mo="'+c.getAttribute('data-mo')+'"]').forEach(function(x){x.checked=c.checked;});__vzload();};});
   db.querySelectorAll('.vzwk').forEach(function(c){c.onchange=function(){
    db.querySelectorAll('.vzday[data-mo="'+c.getAttribute('data-mo')+'"][data-wk="'+c.getAttribute('data-wk')+'"]')
     .forEach(function(x){x.checked=c.checked;});__vzload();};});
   db.querySelectorAll('.vzday').forEach(function(c){c.onchange=function(){__vzload();};});}
  __vzwInit(j.items);})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __vzsel(){
 var w=Object.keys(window.__VZCHK||{});          /* 화면 렌더와 무관 — 전체 체크 반영 */
 var sn=[];document.querySelectorAll('#vzsen .vzck:checked').forEach(function(c){sn.push(c.value);});
 return {w:w,sn:sn};}
function __vzdraw(){
 var sel=__vzsel(),msg=document.getElementById('vzmsg'),out=document.getElementById('vzout');
 var bd=parseInt(document.getElementById('vzb').value,10)||0;
 var gON=document.getElementById('vzg').checked;
 if(window.__VZMODE==='mean'){
  /* 평균 그래프 = 버킷 카드 스크롤: N일마다 카드 1개, 카드 안 곡선 = 체크 센서들
     (복수면 오버레이) + 골든(토글). 카드 체크 → 그 기간 웨이퍼가 CSV 저장 대상. */
  if(!sel.sn.length){msg.textContent='센서를 체크하세요';return;}
  var days2=[];document.querySelectorAll('#vzdt .vzday:checked').forEach(function(c){days2.push(c.value);});
  var ss=__vzstepsel();
  var gu=(window.__VZGM==='count')?'장':(window.__VZGM==='hour'?'시간':'일');
  out.innerHTML='';msg.textContent='기간 '+(bd>0?bd+gu+' 단위':'전체(글로벌 평균)')
   +' 로드 중… 수십 초 걸릴 수 있습니다';
  fetch('/api/viz_buckets',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({sensors:sel.sn,dates:days2,
    group_mode:(window.__VZGM||'day'),group_n:bd})})
  .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
  .then(function(x){
   msg.textContent='묶음 '+x.buckets.length+'개'
    +((x.group_n||0)>0?' ('+x.group_n+({count:'장',hour:'시간'}[x.group_mode]||'일')+' 단위)':' (전체 평균)')
    +(x.sampled?' · ⚠ 장수가 많아 '+(x.sample_unit||'묶음')+' 최대 '+x.per_bucket
      +'장 표본 평균(카드에 표기)':'')
    +((x.cache&&x.cache.enabled)?' · 캐시 '+x.cache.reused_days+'일 재사용'
      +(x.cache.calc_days?'/'+x.cache.calc_days+'일 계산':''):'')
    +' · 카드 체크=그 기간 웨이퍼를 CSV 저장 대상에 포함';
   window.__VZB={};window.__VZBL={};window.__VZBD={};
   var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
   for(var b=0;b<x.buckets.length;b++){var bk=x.buckets[b];
    window.__VZB[b]=bk.ids;window.__VZBL[b]=bk.label;
    window.__VZBD[b]=((x.group_mode||'day')==='day')?[bk.d0,bk.d1]:null;
    var holder=document.createElement('div');holder.className='card';holder.style.cssText='padding:8px 12px;margin:0 0 8px';
    var hl=__vzHL(bk.d0,bk.d1);
    if(hl){holder.style.border='3px solid '+(hl.k==='o'?'#e08a00':'#f5c518');
     holder.style.background=(hl.k==='o'?'#fff8ee':'#fffdf0');}
    var cap=document.createElement('div');cap.style.cssText='font-size:12.5px;font-weight:700;margin:0 0 4px';
    var dkey='vzb'+b,dnm=(x.sensors.length>1?'센서'+x.sensors.length+'개':x.sensors[0])
     +'_'+bk.d0+((x.group_n||0)>0?'_'+x.group_n+({count:'장',hour:'시간'}[x.group_mode]||'일'):'_전체');
    var rows=[['idx'].concat(x.sensors).concat(gON?x.sensors.map(function(s){return s+'(골든)';}):[])];
    for(var ri=0;ri<(bk.curves[x.sensors[0]]||[]).length;ri++){
     var row=[ri];
     for(q=0;q<x.sensors.length;q++)row.push(bk.curves[x.sensors[q]][ri]);
     if(gON)for(q=0;q<x.sensors.length;q++)row.push(x.golden[x.sensors[q]][ri]);
     rows.push(row);}
    window.__DLCSV[dkey]=rows;
    cap.innerHTML=__dlBtns(dnm,dkey,b)
     +'<label style="cursor:pointer"><input type="checkbox" class="vzbk" data-b="'+b+'" style="vertical-align:-2px"> '
     +esc(bk.label)+'</label> <span style="font-weight:400;color:#8a94a3;font-size:11px">'
     +(bk.n_used>=bk.n_total?bk.n_used+'장 전체 평균':bk.n_total+'장 중 '+bk.n_used+'장 표본 평균')
     +(x.sensors.length>1?' · 센서 '+x.sensors.length+'개 오버레이':'')
     +(ss.all?'':' · step '+ss.list.join(',')+' 확대')+'</span>'
     +(hl?'<span style="font-weight:700;font-size:11.5px;color:'+(hl.k==='o'?'#b26a00':'#8a6d00')+'"> · '
       +(hl.k==='o'?('\u25b2 리포트'+(hl.lab.length?': '+esc(hl.lab.join(' / ')):'')):'\u25b3 리포트 직전(3일 내)')+'</span>':'')
     +(function(pt){if(!pt)return '';
       var h2='';
       var one=function(nm,v,thr,vd){
        h2+=' <span style="font-size:11px;font-weight:'+(vd?'700':'400')
         +';color:'+(vd?'#c0392b':'#3a7d44')+'"> · '+nm+' '+v+' (임계 '+thr+') '
         +(vd?vd:'\u2713')+'</span>';};
       if(pt.p1)one('평균 볼록비',pt.p1.ratio,pt.p1.thr,pt.p1.verdict);
       if(pt.p2)one('평균 아치 면적',pt.p2.area+'·'+pt.p2.n_arch+'개',pt.p2.thr,pt.p2.verdict);
       if(pt.p3)one('평균 W면적',pt.p3.area,pt.p3.thr,pt.p3.verdict);
       return h2;})(bk.pattern);
    holder.appendChild(cap);
    var lg2=document.createElement('div');lg2.style.cssText='font-size:11px;margin:0 0 3px';
    var lgh='';
    for(var q=0;q<x.sensors.length;q++)
     lgh+='<span style="display:inline-block;margin-right:10px"><span style="display:inline-block;width:14px;height:3px;background:'+GXCAT[q%GXCAT.length]+';vertical-align:3px"></span> '+esc(x.sensors[q])+'</span>';
    if(gON)lgh+='<span style="display:inline-block;margin-right:10px"><span style="display:inline-block;width:14px;height:3px;background:#b9c2cf;vertical-align:3px"></span> 골든(회색)</span>';
    lg2.innerHTML=lgh;holder.appendChild(lg2);
    /* base = 첫 센서 실측(파랑 #2456a6 = GXCAT[0] 와 동일), 나머지 센서/골든은 extras */
    var extras=[];
    for(q=1;q<x.sensors.length;q++)
     extras.push({label:x.sensors[q],vals:__vzslice(bk.curves[x.sensors[q]],x.steps,ss),color:GXCAT[q%GXCAT.length]});
    if(gON)for(q=0;q<x.sensors.length;q++)
     extras.push({label:x.sensors[q]+'(골든)',vals:__vzslice(x.golden[x.sensors[q]],x.steps,ss),color:'#b9c2cf'});
    var lgb=[{label:x.sensors[0],color:GXCAT[0]}].concat(extras.map(function(e){
     return {label:e.label,color:e.color};}));
    holder.appendChild(__chart(bk.label,__vzslice(bk.curves[x.sensors[0]],x.steps,ss),null,null,null,extras,
     {W:Math.min(window.innerWidth-140,1240),H:280,notp:1,lg:lgb}));
    out.appendChild(holder);}
   })
  .catch(function(e){msg.textContent='실패: '+e.message;});
  return;}
 var bucket=(window.__VZMODE==='mean'&&bd>0);
 if(!bucket&&!sel.w.length){msg.textContent='웨이퍼를 체크하세요 (또는 평균 단위 N일 입력)';return;}
 if(!sel.sn.length){msg.textContent='센서를 체크하세요';return;}
 var lim=(window.__VZMODE==='mean')?60:20;
 if(!bucket&&sel.w.length>lim){msg.textContent=(window.__VZMODE==='mean'?'평균':'겹침')+' 모드는 최대 '+lim+'장 — 앞 '+lim+'장만 사용';}
 var ids=sel.w.slice(0,lim);
 var days=[];document.querySelectorAll('#vzdt .vzday:checked').forEach(function(c){days.push(c.value);});
 out.innerHTML='';msg.textContent=bucket?('기간 '+bd+'일 단위 평균 로드 중…'):('원본 로드 중… ('+ids.length+'장 × '+sel.sn.length+'센서)');
 var done=0;
 sel.sn.forEach(function(sn){
  fetch('/api/viz_wave',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(bucket?{sensor:sn,bucket_days:bd,dates:days}:{ids:ids,sensor:sn})})
  .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
  .then(function(x){
   done++;if(done===sel.sn.length)msg.textContent='';
   var holder=document.createElement('div');holder.className='card';holder.style.cssText='padding:8px 12px;margin:0 0 8px';
   var cap=document.createElement('div');cap.style.cssText='font-size:12.5px;font-weight:700;margin:0 0 4px';
   var bkt=(x.bucket_days||0)>0;
   var ss2=__vzstepsel();
   var dk2='vzo_'+x.sensor,dn2=x.sensor+'_개별'+x.n+'장';
   var rw2=[['idx',x.sensor+'(골든)'].concat(x.curves.map(function(cv){return cv.id;}))];
   for(var r2=0;r2<x.golden.length;r2++){
    var row2=[r2,x.golden[r2]];
    for(var q2=0;q2<x.curves.length;q2++)row2.push(x.curves[q2].vals[r2]);
    rw2.push(row2);}
   window.__DLCSV[dk2]=rw2;
   var esc9=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
   cap.innerHTML=__dlBtns(dn2,dk2)+'<span>'+esc9(x.sensor
    +(bkt?' — '+x.bucket_days+'일 단위 평균 '+x.curves.length+'개 vs 골든(파랑)'
     :(window.__VZMODE==='mean'?' — 선택 '+x.n+'장 평균 vs 골든':' — 개별 '+x.n+'장 겹침 vs 골든(파랑)'))
    +(ss2.all?'':' · step '+ss2.list.join(',')+' 확대'))+'</span>';
   holder.appendChild(cap);
   var extras=[];
   if(bkt||window.__VZMODE!=='mean'){
    for(var q=0;q<x.curves.length;q++)
     extras.push({label:x.curves[q].id,vals:__vzslice(x.curves[q].vals,x.steps,ss2),color:GXCAT[q%GXCAT.length]});}
   var gON2=document.getElementById('vzg').checked;
   var lgo=[{label:(gON2?(x.sensor+'(골든)'):(extras.length?extras[0].label:x.sensor+'(골든)')),
             color:'#2456a6'}]
    .concat(extras.map(function(e,ei){return (!gON2&&ei===0)?null:{label:e.label,color:e.color};})
            .filter(Boolean));
   holder.appendChild(__chart(x.sensor,
    gON2?__vzslice(x.golden,x.steps,ss2):(extras.length?extras[0].vals:__vzslice(x.golden,x.steps,ss2)),
    null,null,null,extras,
    {W:Math.min(window.innerWidth-140,1240),H:300,notp:1,lg:lgo}));
   out.appendChild(holder);})
  .catch(function(e){done++;msg.textContent=sn+' 실패: '+e.message;});});}
/* ── 패턴 인식 탭 ── */
function __ptScan(){
 var msg=document.getElementById('ptmsg'),out=document.getElementById('ptout');
 var pay={step1:document.getElementById('pt_s1').value.trim(),
  steps2:document.getElementById('pt_s2').value.split(',').map(function(v){return v.trim();}).filter(Boolean),
  alpha1:document.getElementById('pt_a1').value.trim(),
  alpha2:document.getElementById('pt_a2').value.trim(),
  step3:document.getElementById('pt_s3').value.trim(),
  alpha3:document.getElementById('pt_a3').value.trim(),
  peak_pct:document.getElementById('pt_n3').value.trim(),
  d0:document.getElementById('pt_d0').value.trim(),
  d1:document.getElementById('pt_d1').value.trim()};
 msg.textContent='패턴 스캔 중… (기간 전수 — 첫 실행은 오래 걸리고, 이후엔 일별 캐시로 빠릅니다)';
 out.innerHTML='';
 fetch('/api/pattern_scan',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(pay)})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(x){
  msg.textContent=x.n+'장 · 위반 '+x.violations.length+'건'
   +(x.cache&&x.cache.enabled?' · 캐시 '+x.cache.reused_days+'일 재사용'
     +(x.cache.calc_days?'/'+x.cache.calc_days+'일 계산':''):'');
  __ptRender(x);document.getElementById('ptnbtn').style.display='';})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __ptRender(x){
 var out=document.getElementById('ptout');out.innerHTML='';
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 window.__PTX=x;
 if(x.off&&x.off.length){
  var wv=document.createElement('div');
  wv.style.cssText='font-size:12px;color:#b26a00;margin:0 0 6px';
  wv.textContent=x.off.join(' · ');out.appendChild(wv);}
 /* 달력: 일별 불량/정상 개수 — 불량 10장 이상인 날은 빨간 테두리 */
 var byd={},dlist=[];
 x.rows.forEach(function(r){
  var d=r.date; if(!byd[d]){byd[d]={bad:0,ok:0};dlist.push(d);}
  var bad=(r.verdicts||[]).length>0
   && __ptRules(r.verdicts).some(function(ru){return __ptShow(ru);});
  if(bad)byd[d].bad++; else byd[d].ok++;});
 if(dlist.length){
  var calWrap=document.createElement('div');calWrap.className='card';
  calWrap.style.cssText='padding:10px 14px;margin:0 0 8px';
  calWrap.innerHTML=__calHtml(byd,dlist,'pt',__calMarksFor('',null));
  out.appendChild(calWrap);
  var dv=document.createElement('div');dv.id='ptdayout';out.appendChild(dv);}
 /* 파형(선분·면적) 카드 — 골든 답지 먼저, 위반 웨이퍼 순서대로(10장씩) */
 var wv=document.createElement('div');
 wv.innerHTML='<div style="font-size:12.5px;font-weight:700;margin:6px 0 4px">파형으로 보기'
  +' <span style="font-weight:400;color:#8a94a3;font-size:11px">골든 답지 + 위반 웨이퍼 —'
  +' 선분(현)과 음영(면적)·원단위 숫자</span></div>';
 out.appendChild(wv);
 var gv=document.createElement('div');out.appendChild(gv);
 var W2=Math.min(window.innerWidth-150,1240);
 if(x.golden.p1&&__ptShow('p1'))__ptWaveFetch('','p1',true)
  .then(function(w){gv.appendChild(__ptCard(w,W2));}).catch(function(){});
 if(x.golden.p2&&__ptShow('p2'))__ptWaveFetch('','p2',true)
  .then(function(w){gv.appendChild(__ptCard(w,W2));}).catch(function(){});
 if(x.golden.p3&&__ptShow('p3'))__ptWaveFetch('','p3',true)
  .then(function(w){gv.appendChild(__ptCard(w,W2));}).catch(function(){});
 if(x.golden.p4&&__ptShow('p4'))__ptWaveFetch('','p4',true)
  .then(function(w){gv.appendChild(__ptCard(w,W2));}).catch(function(){});
 if(x.golden.p5&&__ptShow('p5'))__ptWaveFetch('','p5',true)
  .then(function(w){gv.appendChild(__ptCard(w,W2));}).catch(function(){});
 /* N일 평균 파형 카드 — 선택 기간을 N일씩 평균내 시간순으로(선분·면적·판정 포함) */
 var nd=parseInt((document.getElementById('pt_bd')||{}).value,10)||0;
 if(nd>0){
  var bv=document.createElement('div');out.appendChild(bv);
  bv.innerHTML='<div style="font-size:12px;color:#5b6675;margin:2px 0 4px">'+nd
   +'일 평균 카드 로드 중…</div>';
  fetch('/api/pattern_buckets',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({ndays:nd,step1:x.step1,steps2:x.steps2,
    alpha1:x.alpha1,alpha2:x.alpha2,
    d0:document.getElementById('pt_d0').value.trim(),
    d1:document.getElementById('pt_d1').value.trim()})})
  .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
  .then(function(bx){
   bv.innerHTML='<div style="font-size:12.5px;font-weight:700;margin:6px 0 4px">'
    +nd+'일 평균 파형 <span style="font-weight:400;color:#8a94a3;font-size:11px">'
    +bx.n_buckets+'묶음 — 스크롤하며 모양 변화를 눈대조'
    +' · 소수 불량은 평균에서 희석됨 — 묶음별 개별 판정 수를 함께 표기</span></div>';
   bx.cards.forEach(function(cd){
    /* 묶음 안 '개별' 판정 집계 — 평균은 희석되므로(소수 불량이면 정상처럼 보임)
       개별 위반 수를 카드 위에 따로 표기해야 놓치지 않는다 */
    var vc={},vcr={},tot=0;
    (window.__PTX.rows||[]).forEach(function(r){
     if(r.date>=cd.d0&&r.date<=cd.d1){tot++;
      (r.verdicts||[]).forEach(function(v){
       var ru=__ptRules([v])[0];
       if(!__ptShow(ru))return;                 /* 보기 체크 해제한 규칙은 집계 제외 */
       vc[v]=(vc[v]||0)+1;vcr[ru]=(vcr[ru]||0)+1;});}});
    var ks=Object.keys(vc),hd=document.createElement('div');
    hd.style.cssText='font-size:12px;font-weight:700;margin:6px 0 2px';
    hd.innerHTML=cd.label
     +(ks.length?' <span style="color:#c0392b">· 개별 판정: '+ks.map(function(k){
        return k+' '+vc[k]+'/'+tot+'장';}).join(' · ')+'</span>'
       :' <span style="color:#3a7d44;font-weight:400">· 개별 위반 없음('+tot+'장)</span>');
    bv.appendChild(hd);
    cd.mats.forEach(function(w){if(!__ptShow(w.rule))return;
     w.id=cd.label;
     var card=__ptCard(w,W2);
     /* 주황(희석 경고)은 '이 카드의 규칙' 개별 위반이 있을 때만 — 다른 규칙 위반이
        아치 카드까지 물들이지 않게(사용자 지적) */
     if(!w.verdict&&vcr[w.rule]){card.style.border='2px solid #e08a00';}
     bv.appendChild(card);});});})
  .catch(function(e3){bv.innerHTML='<div style="font-size:12px;color:#c0392b">'
   +nd+'일 평균 카드 실패: '+e3.message+'</div>';});}
 var vv=document.createElement('div');out.appendChild(vv);
 var jobs=[];
 x.violations.forEach(function(r){__ptRules(r.verdicts).forEach(function(ru){
  if(__ptShow(ru))jobs.push([r.id,ru]);});});
 window.__PTJOBS=jobs;window.__PTPOS=0;
 function batch(){
  var end=Math.min(window.__PTPOS+10,jobs.length),chain=Promise.resolve();
  for(var q=window.__PTPOS;q<end;q++)(function(j){
   chain=chain.then(function(){return __ptWaveFetch(j[0],j[1],false)
    .then(function(w){vv.appendChild(__ptCard(w,W2));}).catch(function(){});});})(jobs[q]);
  chain.then(function(){window.__PTPOS=end;
   var mb=document.getElementById('ptmore');
   if(mb)mb.remove();
   if(end<jobs.length){mb=document.createElement('button');mb.id='ptmore';
    mb.textContent='파형 더 보기 ('+(jobs.length-end)+'장 남음)';
    mb.style.cssText='padding:6px 14px;border:1px solid #dde3ea;border-radius:8px;'
     +'background:#fff;cursor:pointer;font-size:12px;margin:0 0 8px';
    mb.onclick=batch;out.appendChild(mb);}});}
 if(jobs.length)batch();}
function __ptNarrate(){
 var x=window.__PTX,bt=document.getElementById('ptnbtn'),msg=document.getElementById('ptmsg');
 if(!x){alert('먼저 [패턴 스캔]을 실행하세요');return;}
 var bt0=bt.textContent;bt.disabled=true;bt.textContent='정리 중…';
 if(msg)msg.textContent='정리 중… (LLM 사용 가능 시 수십 초)';
 fetch('/api/pattern_narrate',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({rows:x.rows,golden:x.golden,sensors:x.sensors})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  bt.disabled=false;bt.textContent=bt0;if(msg)msg.textContent='';
  var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  __gmodal(function(){
   var d=document.createElement('div');d.id='ptnarbox';
   d.style.cssText='min-width:520px;max-width:76vw';
   var h='<div style="font-size:13px;font-weight:700;margin:0 0 8px">패턴 인식 요약'
    +' <span style="font-weight:400;color:#8a94a3;font-size:11px">총 '+j.n+'장'
    +(j.engine?' · '+esc(j.engine):' · 규칙 요약')+'</span></div>';
   if(j.llm){h+='<div style="font-size:12.5px;white-space:pre-wrap;line-height:1.65">'+esc(j.llm)+'</div>'
    +'<div style="font-size:11px;color:#8a94a3;margin:8px 0 2px">아래는 규칙 기반 원문(검증용)</div>';}
   h+='<div style="font-size:12px;color:#3b4653;white-space:pre-wrap;line-height:1.65">'
    +j.summary.map(function(l){return '- '+esc(l);}).join(String.fromCharCode(10))+'</div>';
   if(j.trend&&j.trend.length){
    h+='<div style="font-size:11.5px;font-weight:700;margin:8px 0 2px">기간 추세</div>'
     +'<div style="font-size:12px;color:#3b4653;white-space:pre-wrap;line-height:1.65">'
     +j.trend.map(function(l){return '- '+esc(l);}).join(String.fromCharCode(10))+'</div>';}
   if(j.golden_cmp&&j.golden_cmp.length){
    h+='<div style="font-size:11.5px;font-weight:700;margin:8px 0 2px">골든 대비</div>'
     +'<div style="font-size:12px;color:#3b4653;white-space:pre-wrap;line-height:1.65">'
     +j.golden_cmp.map(function(l){return '- '+esc(l);}).join(String.fromCharCode(10))+'</div>';}
   d.innerHTML=h;
   return d;});})
 .catch(function(e2){bt.disabled=false;bt.textContent=bt0;
  if(msg)msg.textContent='정리 실패: '+e2.message;});}
function __ptRules(verdicts){
 /* 위반 문구 → 관련 규칙 전부(직선/하강=P1, 아치=P2, W=P3 — 복수면 전부) */
 var t=(verdicts||[]).join(','),r=[];
 if(t.indexOf('직선')>=0||t.indexOf('하강')>=0)r.push('p1');
 if(t.indexOf('아치')>=0)r.push('p2');
 if(t.indexOf('W패턴')>=0)r.push('p3');
 if(t.indexOf('온도 계통')>=0)r.push('p4');
 if(t.indexOf('압력 계통')>=0)r.push('p5');
 return r.length?r:['p1'];}
function __ptShow(rule){
 var el=document.getElementById({p1:'pt_f1',p2:'pt_f2',p3:'pt_f3',p4:'pt_f4',p5:'pt_f5'}[rule]);
 return !el||el.checked;}
document.addEventListener('change',function(e){
 if(e.target&&/^pt_f[12345]$/.test(e.target.id||'')&&window.__PTX)__ptRender(window.__PTX);});
function __ptCard(w,W){
 /* 파형 + 선분(현) + 음영(면적) + 원단위 숫자 — 팝업·인라인 공용 */
 var wrap=document.createElement('div');wrap.className='card dlroot';
 wrap.style.cssText='padding:8px 12px;margin:0 0 8px';
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var nm=w.nums||{},u=nm.unit||'',cap=document.createElement('div');
 cap.style.cssText='font-size:12.5px;font-weight:700;margin:0 0 4px';
 var line;
 if(w.rule==='p4'||w.rule==='p5'){
  var t45=(w.rule==='p4'?'온도 계통 잔차':'압력 계통 잔차');
  line='최대 z '+(nm.maxz==null?'-':nm.maxz)+' · 연속 이탈 '+nm.run+'pt(기준 '
   +nm.min_run+'pt) · 임계 \u00b1'+nm.k+'\u03c3 (\u03c3 중앙값 '+nm.sig+u+')'
   +' · 모델 '+(nm.model||'');
  cap.innerHTML=__dlBtns(__dlfn(w.id+'_'+w.rule))
   +'<span>'+t45+' \u2014 '+esc(w.sensor)+' \u2212 모델 \u2014 '+esc(w.id)
   +(w.verdict?' <b style="color:#c0392b">'+esc(w.verdict)+'</b>':'')
   +' <span style="font-weight:400;color:#5b6675;font-size:11.5px">'+esc(line)+'</span></span>';
  if(w.verdict){wrap.style.border='2px solid #c0392b';wrap.style.background='#fff6f5';}
  wrap.appendChild(cap);
  var mk45=(w.peaks||[]).map(function(i){return {i:i,color:'#d64545'};});
  var zero45=(w.y||[]).map(function(){return 0;});
  var lg45=[{label:'잔차 R(t) ('+u+')',color:'#2456a6'},
            {label:'\u00b1'+nm.k+'\u03c3 임계 밴드',color:'#8a94a3'},
            {label:'빨간 점=이탈(z>'+nm.k+')',color:'#d64545'},
            {label:'회색=판정 제외 스텝(첫/끝)',color:'#8a94a3'}];
  wrap.appendChild(__chart(t45+' \u2014 '+w.id,w.y,null,null,null,
   [{label:'+'+nm.k+'\u03c3',vals:w.bandu,color:'#8a94a3'},
    {label:'-'+nm.k+'\u03c3',vals:w.bandl,color:'#8a94a3'}],
   {W:W||Math.min(window.innerWidth-160,1200),H:300,noexp:1,notp:1,
    fill:{vals:zero45,color:'#2b7fc4',opacity:0.14},marks:mk45,lg:lg45,
    roi:w.roi||null,toplabels:w.stepmarks||[]}));
  return wrap;}
 if(w.rule==='p1'){
  line='현 '+nm.y0+' \u2192 '+nm.y1+' (낙폭 '+nm.drop+u+') · 현-곡선 평균 간격 '
   +(nm.gap==null?'-':nm.gap+u)+' · 볼록비 '+w.score
   +(nm.thr!=null?' (골든 '+nm.g_ratio+' · 임계 '+nm.thr+')':'');}
 else if(w.rule==='p3'){
  line='최저점\u2192끝 상승 '+(nm.rise==null?'-':nm.rise+u)
   +' · 현재 면적 '+(nm.bulge==null?'-':nm.bulge+u)
   +(nm.thr_area!=null?' · 임계 면적 '+nm.thr_area+u+' (초과=의심)':'')
   +(nm.peak_pct!=null?' · 관심구간 최고 '+nm.peak_roi+u+' / 전체 최고 '
     +nm.peak_all+u+' ('+nm.peak_pct+'%, 기준 '+nm.n_pct+'%)':'');}
 else{
  var ar=(nm.areas||[]).join(' + ');
  line='아치 면적('+u+'): '+(ar||'-')+(nm.sum!=null?' = '+nm.sum+u:'')
   +(nm.thr!=null?' (골든 '+nm.g_area+' · 임계 '+nm.thr+')':'');}
 cap.innerHTML=__dlBtns(__dlfn(w.id+'_'+w.sensor+'_'+w.rule))
  +'<span>'+esc(w.sensor)+' — '+esc(w.id)
  +(w.verdict?' <b style="color:#c0392b">'+esc(w.verdict)+'</b>':'')
  +' <span style="font-weight:400;color:#5b6675;font-size:11.5px">'+esc(line)+'</span></span>';
 if(w.verdict){wrap.style.border='2px solid #c0392b';wrap.style.background='#fff6f5';}
 wrap.appendChild(cap);
 var marks=[];
 (w.feet||[]).forEach(function(i){marks.push({i:i,color:'#2456a6'});});
 (w.peaks||[]).forEach(function(i){marks.push({i:i,color:'#d64545'});});
 var chnm=(w.rule==='p1'?'현(첫값\u2194끝값)':(w.rule==='p3'?'최저점\u2192끝 선분':'받침점 현'));
 var lg=[{label:w.sensor,color:'#2456a6'},{label:chnm,color:'#8a94a3'},
         {label:'음영=선분-곡선 면적',color:'#2b7fc4'},
         {label:'회색=관심 스텝 밖',color:'#8a94a3'}];
 if((w.peaks||[]).length)lg.push({label:'정점(빨강)·받침점(파랑)',color:'#d64545'});
 wrap.appendChild(__chart(w.sensor+' — '+w.id,w.y,null,null,null,
  [{label:chnm,vals:w.chord,color:'#8a94a3'}],
  {W:W||Math.min(window.innerWidth-160,1200),H:300,noexp:1,notp:1,
   fill:{vals:w.chord,color:'#2b7fc4',opacity:0.18},marks:marks,lg:lg,
   roi:w.roi||null,toplabels:w.stepmarks||[]}));
 return wrap;}
function __ptWaveFetch(wid,rule,golden){
 var x=window.__PTX||{};
 return fetch('/api/pattern_wave',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({id:wid,rule:rule,golden:!!golden,step1:x.step1,steps2:x.steps2})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});});}
/* ── 분석자료 탭 ── */
/* ── 데이터 뷰어 탭 ── */
window.__DVW=[];window.__DVFX=[];
function __dvList(){
 var msg=document.getElementById('dvmsg');
 msg.textContent='목록 조회 중…';
 fetch('/api/dv_list',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({folder:document.getElementById('dv_f').value.trim()})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  window.__DVW=j.items;window.__DVAV=j.avail||{};window.__DVPG=0;
  msg.textContent=j.total+'장 — 이름 클릭=센서 목록 · 체크=그래프 대상';
  __dvFltRender();__dvWRender();})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __dvFltRender(){
 var box=document.getElementById('dv_flt'),av=window.__DVAV||{};
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var grp=function(key,nm){
  var o=av[key]||{},ks=Object.keys(o).sort();
  if(!ks.length)return '';
  return '<div style="margin:0 0 2px"><span style="color:#8a94a3;margin-right:6px">'+nm+'</span>'
   +ks.map(function(v){return '<label style="margin-right:10px"><input type="checkbox" '
    +'class="dvfl" data-g="'+key+'" value="'+esc(v)+'"> '+esc(v)
    +' <span style="color:#8a94a3">('+o[v]+')</span></label>';}).join('')+'</div>';};
 box.innerHTML=grp('eqp','설비')+grp('dsv','챔버')+grp('recipe','유형');
 box.querySelectorAll('.dvfl').forEach(function(c){c.onchange=function(){
  window.__DVPG=0;__dvWRender();};});}
function __dvFltSel(){
 var out={eqp:[],dsv:[],recipe:[]};
 document.querySelectorAll('#dv_flt .dvfl:checked').forEach(function(c){
  out[c.getAttribute('data-g')].push(c.value);});
 return out;}
function __dvFltPass(it,f){
 /* 그룹 안 OR · 그룹 간 AND · 미선택 그룹=전체 (분석자료 탭과 동일 의미) */
 if(f.eqp.length&&f.eqp.indexOf(it.e||'(없음)')<0&&f.eqp.indexOf(it.e)<0)return false;
 if(f.dsv.length&&f.dsv.indexOf(it.d||'(없음)')<0&&f.dsv.indexOf(it.d)<0)return false;
 if(f.recipe.length&&f.recipe.indexOf(it.r||'(없음)')<0&&f.recipe.indexOf(it.r)<0)return false;
 return true;}
function __dvFiltered(){
 var q=(document.getElementById('dv_wq').value||'').toLowerCase();
 var f=__dvFltSel(),out=[];
 for(var i=0;i<window.__DVW.length;i++){var it=window.__DVW[i];
  if(q&&it.id.toLowerCase().indexOf(q)<0)continue;
  if(!__dvFltPass(it,f))continue;
  out.push(it);}
 return out;}
function __dvWRender(){
 var box=document.getElementById('dv_w'),esc=function(t){return String(t==null?'':t)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var fl=__dvFiltered(),PS=100;
 var np=Math.max(1,Math.ceil(fl.length/PS));
 if((window.__DVPG||0)>=np)window.__DVPG=np-1;
 if((window.__DVPG||0)<0)window.__DVPG=0;
 var p0=(window.__DVPG||0)*PS,page=fl.slice(p0,p0+PS),h='';
 for(var i=0;i<page.length;i++){var it=page[i];
  h+='<div><label><input type="checkbox" class="dvwk" value="'+esc(it.id)+'"'
   +(it._ck?' checked':'')+'> </label><a href="javascript:void(0)" class="dvwn" data-id="'
   +esc(it.id)+'">'+esc(it.id)+'</a> <span style="color:#8a94a3">'+esc(it.ts||it.date)+'</span></div>';}
 box.innerHTML=h||'<span style="color:#8a94a3">조건에 맞는 웨이퍼가 없습니다</span>';
 box.querySelectorAll('.dvwk').forEach(function(c){c.onchange=function(){
  window.__DVW.forEach(function(it){if(it.id===c.value)it._ck=c.checked?1:0;});};});
 var pg=document.getElementById('dv_pg');
 if(pg){var ck=0;window.__DVW.forEach(function(it){if(it._ck)ck++;});
  pg.innerHTML='<button onclick="__dvPage(-1)" style="padding:2px 10px;border:1px solid #dde3ea;'
   +'border-radius:6px;background:#fff;cursor:pointer;font-size:12px">이전</button> '
   +'<span style="margin:0 6px">'+(fl.length?((window.__DVPG||0)+1):0)+'/'+(fl.length?np:0)+'페이지</span>'
   +'<button onclick="__dvPage(1)" style="padding:2px 10px;border:1px solid #dde3ea;'
   +'border-radius:6px;background:#fff;cursor:pointer;font-size:12px">다음</button>'
   +'<span style="color:#8a94a3;margin-left:8px">필터 '+fl.length+'장 / 전체 '
   +window.__DVW.length+'장 · 체크 '+ck+'장</span>';}}
function __dvPage(d){window.__DVPG=(window.__DVPG||0)+d;__dvWRender();}
function __dvWAll(v){
 /* 전체/해제는 '지금 필터에 보이는' 웨이퍼에만 적용(다른 필터 오염 방지) */
 var ids={};__dvFiltered().forEach(function(it){ids[it.id]=1;});
 window.__DVW.forEach(function(it){if(ids[it.id])it._ck=v;});__dvWRender();}
function __dvSensors(id){
 var box=document.getElementById('dv_s');
 box.innerHTML='<span style="color:#8a94a3">읽는 중…</span>';
 fetch('/api/dv_sensors',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({id:id})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  window.__DVS=j.sensors;
  document.getElementById('dv_sfrom').textContent='('+j.id+' · '+j.sensors.length+'개 · '+j.T+'pt)';
  var sel=document.getElementById('dv_fxsel'),esc=function(t){return String(t)
   .replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  sel.innerHTML=j.sensors.map(function(sn){return '<option>'+esc(sn)+'</option>';}).join('');
  __dvSRender();})
 .catch(function(e){box.innerHTML='<span style="color:#c0392b">'+e.message+'</span>';});}
function __dvSRender(){
 var q=(document.getElementById('dv_sq').value||'').toLowerCase();
 var box=document.getElementById('dv_s'),prev={};
 box.querySelectorAll('.dvsk').forEach(function(c){prev[c.value]=c.checked;});
 var esc=function(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var h='';
 (window.__DVS||[]).forEach(function(sn){
  if(q&&sn.toLowerCase().indexOf(q)<0)return;
  var isv=/^d[pt]_/i.test(sn);
  h+='<label style="display:block'+(isv?';color:#6a4fa3;font-weight:700':'')+'">'
   +'<input type="checkbox" class="dvsk" value="'+esc(sn)+'"'
   +(prev[sn]?' checked':'')+'> '+esc(sn)+'</label>';});
 box.innerHTML=h||'<span style="color:#8a94a3">웨이퍼 이름을 클릭하면 센서 목록이 나옵니다</span>';}
function __dvFxIns(){
 var inp=document.getElementById('dv_fx'),sel=document.getElementById('dv_fxsel');
 if(!sel.value)return;
 var p=inp.selectionStart||inp.value.length;
 inp.value=inp.value.slice(0,p)+sel.value+inp.value.slice(p);
 inp.focus();}
function __dvFxAdd(){
 var v=document.getElementById('dv_fx').value.trim();
 if(!v){alert('수식을 입력하세요');return;}
 if(window.__DVFX.indexOf(v)<0)window.__DVFX.push(v);
 document.getElementById('dv_fx').value='';
 __dvFxRender();__dvFxSave();}
function __dvFxRender(){
 var esc=function(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 document.getElementById('dv_fxl').innerHTML=window.__DVFX.map(function(fx,i){
  return '<label style="display:inline-block;margin-right:6px;color:#6a4fa3">'
   +'<input type="checkbox" class="dvfk" value="'+i+'" checked> 수식: '+esc(fx)+'</label>'
   +'<a href="javascript:void(0)" class="dvfd" data-i="'+i+'" style="font-size:11px;'
   +'color:#c0392b;margin-right:12px">[삭제]</a>';}).join('');
 document.querySelectorAll('.dvfd').forEach(function(a){a.onclick=function(){
  window.__DVFX.splice(parseInt(a.getAttribute('data-i'),10),1);
  __dvFxRender();__dvFxSave();};});}
function __dvFxSave(){
 fetch('/api/dv_form_save',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({forms:window.__DVFX})}).catch(function(){});}
function __dvDraw(){
 var ids=[];window.__DVW.forEach(function(it){if(it._ck)ids.push(it.id);});
 var sns=[];document.querySelectorAll('#dv_s .dvsk:checked').forEach(function(c){sns.push(c.value);});
 var fxs=[];document.querySelectorAll('.dvfk:checked').forEach(function(c){
  fxs.push(window.__DVFX[parseInt(c.value,10)]);});
 var out=document.getElementById('dv_out'),msg=document.getElementById('dvmsg');
 if(!ids.length){msg.textContent='웨이퍼를 체크하세요';return;}
 if(!sns.length&&!fxs.length){msg.textContent='센서 또는 수식을 체크하세요';return;}
 if(ids.length>20){msg.textContent='최대 20장 — 앞 20장만 사용';ids=ids.slice(0,20);}
 out.innerHTML='';msg.textContent='로드 중… ('+ids.length+'장 × '+(sns.length+fxs.length)+'개)';
 var jobs=sns.map(function(sn){return {sensor:sn};})
  .concat(fxs.map(function(fx){return {formula:fx};}));
 var done=0;
 jobs.forEach(function(jb,ji){
  var body={ids:ids};for(var k in jb)body[k]=jb[k];
  fetch('/api/dv_wave',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(body)})
  .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
  .then(function(x){
   done++;if(done===jobs.length)msg.textContent='';
   var holder=document.createElement('div');holder.className='card dlroot';
   holder.style.cssText='padding:8px 12px;margin:0 0 8px';
   var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
   var cap=document.createElement('div');
   cap.style.cssText='font-size:12.5px;font-weight:700;margin:0 0 4px';
   var dk='dv_'+ji;
   var rows=[['idx'].concat(x.curves.map(function(cv){return cv.id;}))];
   var Lm=0;x.curves.forEach(function(cv){if(cv.vals.length>Lm)Lm=cv.vals.length;});
   for(var r2=0;r2<Lm;r2++){var row=[r2];
    x.curves.forEach(function(cv){row.push(r2<cv.vals.length?cv.vals[r2]:null);});
    rows.push(row);}
   window.__DLCSV[dk]=rows;
   cap.innerHTML=__dlBtns(__dlfn(x.label),dk)+'<span>'+esc(x.label)
    +' <span style="font-weight:400;color:#8a94a3;font-size:11px">'+x.n+'장 겹침 · 원본 그대로</span></span>';
   holder.appendChild(cap);
   var extras=[];
   for(var q=1;q<x.curves.length;q++)
    extras.push({label:x.curves[q].id,vals:x.curves[q].vals,color:GXCAT[q%GXCAT.length]});
   var lg=[{label:x.curves[0]?x.curves[0].id:'',color:'#2456a6'}]
    .concat(extras.map(function(e){return {label:e.label,color:e.color};}));
   var lgd=document.createElement('div');
   lgd.className='dvlg';
   lgd.style.cssText='font-size:11px;color:#5b6675;margin:0 0 2px;line-height:1.9';
   lgd.innerHTML=lg.map(function(e2){return '<span style="margin-right:10px;white-space:nowrap">'
    +'<span style="display:inline-block;width:14px;height:3px;background:'+e2.color
    +';vertical-align:middle;margin-right:4px"></span>'+esc(e2.label)+'</span>';}).join('');
   holder.appendChild(lgd);
   holder.appendChild(__chart(x.label,x.curves[0]?x.curves[0].vals:[],null,null,null,extras,
    {W:Math.min(window.innerWidth-140,1240),H:300,lg:lg,
     baseLabel:(x.curves[0]?x.curves[0].id:'')}));
   out.appendChild(holder);})
  .catch(function(e){done++;
   var d=document.createElement('div');d.className='card';
   d.style.cssText='padding:8px 12px;margin:0 0 8px;color:#c0392b;font-size:12.5px';
   d.textContent=(jb.formula?('수식: '+jb.formula):jb.sensor)+' — '+e.message;
   out.appendChild(d);});});}
document.addEventListener('click',function(e){
 var a=e.target.closest?e.target.closest('.dvwn'):null;
 if(a){__dvSensors(a.getAttribute('data-id'));}});
document.addEventListener('input',function(e){
 if(e.target&&e.target.id==='dv_wq')__dvWRender();
 if(e.target&&e.target.id==='dv_sq')__dvSRender();});
/* ── 패턴 탐사 탭 — 데몬 원장 열람 ── */
window.__HX=null;window.__HXC=null;
function __hxList(){
 var msg=document.getElementById('hxmsg'),out=document.getElementById('hx_out').value.trim();
 msg.textContent='원장 조회 중…';
 fetch('/api/hunt_list',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({out:out})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  window.__HX={out:out,rows:j.rows};
  msg.textContent='가설 '+j.n+'건 — 확정 '+j.counts.confirmed+' · 검증중 '+j.counts.testing
   +' · 제안 '+j.counts.proposed+' · 기각 '+j.counts.rejected;
  var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var stc={confirmed:'#c0392b',testing:'#e08a00',proposed:'#5b6675',rejected:'#8a94a3'};
  document.getElementById('hx_l').innerHTML=j.rows.map(function(h){
   return '<div class="hxrow" data-id="'+h.id+'" style="cursor:pointer;padding:2px 0;'
    +(h.st==='rejected'?'color:#8a94a3':'')+'">가설 #'+h.id
    +' <b style="color:'+(stc[h.st]||'#5b6675')+'">['+esc(h.status)+']</b> — '+esc(h.desc)
    +' <span style="color:#8a94a3;font-size:11px">'+esc(h.cause)+' · '
    +esc(h.eqp)+'/'+esc(h.dsv)+' '+esc(h.date)+'</span>'
    +(h.reject?' <span style="color:#8a94a3;font-size:11px">— '+esc(h.reject)+'</span>':'')
    +'</div>';}).join('');
  document.getElementById('hx_d').innerHTML='';
  document.getElementById('hx_w').innerHTML='';
  document.getElementById('hx_c').innerHTML='';})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __hxOpen(id){
 var out=(window.__HX||{}).out||document.getElementById('hx_out').value.trim();
 var d=document.getElementById('hx_d');
 d.innerHTML='<div style="font-size:12px;color:#5b6675">가설 #'+id+' 조회 중…</div>';
 fetch('/api/hunt_cal',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({out:out,id:id})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  window.__HXC={h:j.h,days:j.days,out:out,id:id};
  var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var h='<div class="card" style="padding:10px 14px;margin:0 0 8px">'
   +'<div style="font-size:13px;font-weight:700;margin:0 0 6px">가설 #'+id+'</div>'
   +'<div style="font-size:12.5px;line-height:1.7;white-space:pre-wrap">'
   +j.desc.map(function(l){return '- '+esc(l);}).join(String.fromCharCode(10))+'</div>'
   +(j.llm?'<div style="font-size:12px;color:#3b4653;margin:6px 0 0;white-space:pre-wrap;'
     +'border-top:1px solid #eef1f5;padding-top:6px">'+esc(j.llm)+'</div>':'')
   +'</div>';
  var dlist=Object.keys(j.byd);
  h+='<div class="card" style="padding:10px 14px;margin:0 0 8px">'
   +__calHtml(j.byd,dlist,'hx',j.marks||{})+'</div>';
  d.innerHTML=h;
  document.getElementById('hx_w').innerHTML='';
  document.getElementById('hx_c').innerHTML='';})
 .catch(function(e){d.innerHTML='<div style="font-size:12px;color:#c0392b">'+e.message+'</div>';});}
function __hxDays(days,label){
 var c=window.__HXC; if(!c)return;
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var h='<div class="card" style="padding:8px 12px;margin:0 0 8px">'
  +'<div style="font-size:12.5px;font-weight:700;margin:0 0 4px">'+esc(label)+' 웨이퍼 목록'
  +' <span style="font-weight:400;color:#8a94a3;font-size:11px">이름 클릭=패턴 카드</span></div>';
 var nb=0,nk=0;
 days.forEach(function(dd){
  var v=c.days[dd]; if(!v)return;
  nk+=v.ok||0;
  (v.bad||[]).forEach(function(w){nb++;
   h+='<div style="font-size:12px"><a href="javascript:void(0)" class="hxwn" data-id="'
    +esc(w)+'" data-day="'+esc(dd)+'">'+esc(w)+'</a>'
    +' <span style="color:#c0392b;font-size:11px">기준 초과</span>'
    +' <span style="color:#8a94a3;font-size:11px">'+esc(dd)+'</span></div>';});});
 h+='<div style="font-size:11.5px;color:#5b6675;margin-top:4px">기준 초과 '+nb
  +'장 · 기준 안 '+nk+'장</div></div>';
 document.getElementById('hx_w').innerHTML=h;
 document.getElementById('hx_c').innerHTML='';}
function __hxWave(wid,day){
 var c=window.__HXC; if(!c)return;
 var box=document.getElementById('hx_c');
 box.innerHTML='<div style="font-size:12px;color:#5b6675">'+wid+' 조회 중…</div>';
 fetch('/api/hunt_wave',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({out:c.out,id:wid,day:day,h:c.h})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var wrap=document.createElement('div');wrap.className='card dlroot';
  wrap.style.cssText='padding:8px 12px;margin:0 0 8px';
  var line='특징값 '+(j.val==null?'-':j.val)+' / 정상 기준 '+(j.fence==null?'-':j.fence)
   +(j.z==null?'':' (z='+j.z+')');
  var cap=document.createElement('div');
  cap.style.cssText='font-size:12.5px;font-weight:700;margin:0 0 4px';
  cap.innerHTML=__dlBtns(__dlfn(j.id+'_'+j.sensor+'_hx'))
   +'<span>'+esc(j.sensor)+' — '+esc(j.id)
   +(j.beyond?' <b style="color:#c0392b">기준 초과</b>':'')
   +' <span style="font-weight:400;color:#5b6675;font-size:11.5px">'
   +esc(c.h.feature+' · '+line)+'</span></span>';
  if(j.beyond){wrap.style.border='2px solid #c0392b';wrap.style.background='#fff6f5';}
  wrap.appendChild(cap);
  wrap.appendChild(__chart(j.sensor+' — '+j.id,j.y,null,null,null,[],
   {W:Math.min(window.innerWidth-150,1240),H:300,noexp:1,notp:1,
    roi:j.roi||null,toplabels:j.stepmarks||[],baseLabel:j.id,
    lg:[{label:j.sensor,color:'#2456a6'},{label:'회색=패턴 스텝 밖',color:'#8a94a3'}]}));
  box.innerHTML='';box.appendChild(wrap);})
 .catch(function(e){box.innerHTML='<div style="font-size:12px;color:#c0392b">'+e.message+'</div>';});}
document.addEventListener('click',function(e){
 var r=e.target.closest?e.target.closest('.hxrow'):null;
 if(r){__hxOpen(parseInt(r.getAttribute('data-id'),10));return;}
 var w=e.target.closest?e.target.closest('.hxwn'):null;
 if(w){__hxWave(w.getAttribute('data-id'),w.getAttribute('data-day'));return;}
 var d=e.target.closest?e.target.closest('.hxday'):null;
 if(d){var k=d.getAttribute('data-day');__hxDays([k],k);return;}
 var m=e.target.closest?e.target.closest('.hxmon'):null;
 if(m){var mo=m.getAttribute('data-mon'),c=window.__HXC;
  if(!c)return;
  var ds=Object.keys(c.days).filter(function(x){return x.slice(0,7)===mo;}).sort();
  __hxDays(ds,mo+' (월 전체)');return;}});
window.__CALM=[];
function __ptMarks(){__calMarksLoad('pt_mk',function(j){
 if(window.__PTX)__ptRender(window.__PTX);
 var m=document.getElementById('ptmsg');if(m)m.textContent='불량 내역 '+j.n+'건 표시';});}
function __ptMarksClr(){window.__CALM=[];if(window.__PTX)__ptRender(window.__PTX);}
function __anMarks(){__calMarksLoad('an_mk',function(j){
 if(window.__ANAGG)__anAgg(false);
 var m=document.getElementById('anmsg');if(m)m.textContent='불량 내역 '+j.n+'건 표시';});}
function __anMarksClr(){window.__CALM=[];if(window.__ANAGG)__anAgg(false);}
function __calMarksLoad(inpId,after){
 var p=(document.getElementById(inpId)||{value:''}).value.trim();
 if(!p){alert('불량 내역 CSV 경로를 입력하세요');return;}
 fetch('/api/cal_marks',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({csv:p})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){window.__CALM=j.marks;after&&after(j);})
 .catch(function(e){alert('불량 내역 CSV 실패: '+e.message);});}
function __calMarksFor(eqpSel,dsvSel){
 /* 현재 선택(설비/DSV)에 해당하는 마크만 {날짜:[내역]} 로 — 빈 칸 마크는 전체 적용 */
 var out={};
 (window.__CALM||[]).forEach(function(m){
  if(eqpSel&&m.eqp&&m.eqp.indexOf(eqpSel)<0&&eqpSel.indexOf(m.eqp)<0)return;
  if(dsvSel&&dsvSel.length&&m.dsv&&dsvSel.indexOf(m.dsv)<0)return;
  (out[m.date]=out[m.date]||[]).push(m.text
   +(m.eqp||m.dsv?' ['+[m.eqp,m.dsv].filter(Boolean).join(' ')+']':''));});
 return out;}
function __anShow(rule){
 var el=document.getElementById({p1:'an_f1',p2:'an_f2',p3:'an_f3',p4:'an_f4',p5:'an_f5'}[rule]);
 return !el||el.checked;}
function __anPay(extra){
 var g=function(id){return (document.getElementById(id)||{value:''}).value.trim();};
 var eqp=[],dsv=[],rc=[];
 document.querySelectorAll('#aneqp input:checked').forEach(function(c){if(c.value)eqp.push(c.value);});
 document.querySelectorAll('#andsv input:checked').forEach(function(c){dsv.push(c.value);});
 document.querySelectorAll('#anrc input:checked').forEach(function(c){rc.push(c.value);});
 var p={eqp:eqp,dsv:dsv,recipe:rc,
  alpha1:g('an_a1'),alpha2:g('an_a2'),alpha3:g('an_a3'),peak_pct:g('an_n3'),
  d0:g('an_d0'),d1:g('an_d1'),
  ndays:parseInt(g('an_nd'),10)||1};
 for(var k in (extra||{}))p[k]=extra[k];
 return p;}
function __anStart(){
 var msg=document.getElementById('anmsg');
 var g=function(id){return document.getElementById(id).value.trim();};
 msg.textContent='분석 시작 중…';
 fetch('/api/anl_scan',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({d0:g('an_d0'),d1:g('an_d1'),step1:g('an_s1'),
   step3:g('an_s3'),
   steps2:g('an_s2').split(',').map(function(v){return v.trim();}).filter(Boolean)})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  msg.textContent='스캔 중… ('+j.n_wafers+'장/'+j.n_days+'일 — 첫 실행은 오래 걸리고 '
   +'이후엔 일별 캐시로 빠릅니다)';
  var tm=setInterval(function(){
   fetch('/api/anl_status',{method:'POST'}).then(function(r){return r.json();})
   .then(function(st){
    if(st.state==='running'){msg.textContent='스캔 중… '+st.progress;return;}
    clearInterval(tm);
    if(st.state==='error'){msg.textContent='실패: '+st.error;return;}
    msg.textContent=st.progress;
    __anAgg(true);});},3000);})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __anAgg(fill){
 var msg=document.getElementById('anmsg');
 fetch('/api/anl_agg',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(__anPay())})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(x){
  window.__ANAGG=x;
  var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  /* 필터 박스는 매 갱신 재구성 — 선택 보존 + 숫자는 '다른 그룹 선택 적용' 패싯
     (체크박스 숫자 = 지금 조건으로 달력에 나올 장수. 만장 vs 몇백장 어긋남 해소) */
  var mk=function(box,vals,type,name){
   var el=document.getElementById(box),prev={},had=el.querySelectorAll('input').length>0;
   el.querySelectorAll('input:checked').forEach(function(c){prev[c.value]=1;});
   var h='';
   if(type==='radio')h+='<label><input type="radio" name="'+name+'" value=""'
    +((prev['']||!had)?' checked':'')+'> 전체</label> ';
   Object.keys(vals).sort().forEach(function(v){
    h+='<label style="margin-right:8px"><input type="'+type+'" '
     +(type==='radio'?'name="'+name+'" ':'')+'value="'+esc(v)+'"'
     +(prev[v]?' checked':'')+'> '
     +esc(v)+' <span style="color:#8a94a3">('+vals[v]+')</span></label>';});
   el.innerHTML=h;};
  mk('aneqp',x.avail.eqp,'radio','anq');
  mk('andsv',x.avail.dsv,'checkbox','');
  mk('anrc',x.avail.recipe,'checkbox','');
  document.getElementById('anflt').style.display='';
  /* 달력(공용 렌더러) — 보기 체크 반영: 켜진 규칙의 불량만 합산 */
  var byd={},dlist=[];
  Object.keys(x.days).forEach(function(d){
   var c=x.days[d];   /* [b1,b2,b3,bad,ok,b4,b5] */
   var shown=(__anShow('p1')?c[0]:0)+(__anShow('p2')?c[1]:0)+(__anShow('p3')?c[2]:0)
    +(__anShow('p4')?(c[5]||0):0)+(__anShow('p5')?(c[6]||0):0);
   var bad=Math.min(c[3],shown);
   byd[d]={bad:bad,ok:c[3]+c[4]-bad};dlist.push(d);});
  var dsvSel=[];document.querySelectorAll('#andsv input:checked')
   .forEach(function(c){dsvSel.push(c.value);});
  var cal=document.getElementById('ancal');
  cal.innerHTML='<div class="card" style="padding:10px 14px;margin:0 0 8px">'
   +__calHtml(byd,dlist,'an',__calMarksFor(__anEqpSel(),dsvSel))+'</div>';
  if(x.off&&x.off.length)msg.textContent=x.off.join(' · ');})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __anEqpSel(){
 var r=document.querySelector('#aneqp input:checked');
 return r&&r.value?r.value:'';}
function __anRange(d0,d1,label,ndFix){
 var out=document.getElementById('anout'),msg=document.getElementById('anmsg');
 out.innerHTML='';
 var hd=document.createElement('div');hd.className='card';
 hd.style.cssText='padding:8px 12px;margin:0 0 8px';
 hd.innerHTML='<div style="font-size:12.5px;font-weight:700">'
  +String(label).replace(/</g,'&lt;')
  +' <span id="anrmsg" style="font-weight:400;color:#8a94a3;font-size:11px">로드 중…</span></div>';
 var cb=document.createElement('button');cb.textContent='닫기';
 cb.style.cssText='float:right;font-size:11px;padding:2px 8px;border:1px solid #dde3ea;'
  +'border-radius:6px;background:#fff;cursor:pointer';
 cb.onclick=function(){out.innerHTML='';window.__ANDV=null;};
 hd.firstChild.appendChild(cb);
 out.appendChild(hd);
 var cc=document.createElement('div');cc.id='anrcards';out.appendChild(cc);
 fetch('/api/anl_buckets',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(__anPay(ndFix?{d0:d0,d1:d1,ndays:ndFix}:{d0:d0,d1:d1}))})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(bx){
  var cards=document.getElementById('anrcards');if(!cards)return;
  var W2=Math.min(window.innerWidth-150,1240),mats=[];
  bx.cards.forEach(function(cd){cd.mats.forEach(function(w){
   if(!__anShow(w.rule))return;
   w.id=cd.label;mats.push(w);cards.appendChild(__ptCard(w,W2));});});
  window.__ANDV={label:label,mats:mats};
  var m2=document.getElementById('anrmsg');
  if(m2)m2.textContent=bx.n_buckets+'묶음 · 그래프 '+mats.length+'장 · 위반 '
   +mats.filter(function(m){return m.verdict;}).length+'건';})
 .catch(function(e){var m2=document.getElementById('anrmsg');
  if(m2)m2.textContent='실패: '+e.message;else msg.textContent='실패: '+e.message;});}
function __anDayView(d){
 /* 일 클릭: 그날 웨이퍼 '전수'의 개별 패턴 그래프(평균 아님) — 40장씩 페이지 */
 var out=document.getElementById('anout');
 out.innerHTML='';
 var hd=document.createElement('div');hd.className='card';
 hd.style.cssText='padding:8px 12px;margin:0 0 8px';
 hd.innerHTML='<div style="font-size:12.5px;font-weight:700">'+d+' 개별 웨이퍼 그래프'
  +' <span id="anrmsg" style="font-weight:400;color:#8a94a3;font-size:11px">목록 조회…</span></div>';
 var cb=document.createElement('button');cb.textContent='닫기';
 cb.style.cssText='float:right;font-size:11px;padding:2px 8px;border:1px solid #dde3ea;'
  +'border-radius:6px;background:#fff;cursor:pointer';
 cb.onclick=function(){out.innerHTML='';window.__ANDV=null;};
 hd.firstChild.appendChild(cb);out.appendChild(hd);
 var cc=document.createElement('div');cc.id='anrcards';out.appendChild(cc);
 var rules=['p1','p2','p3','p4','p5'].filter(__anShow);
 var pay=__anPay({d0:d,d1:d});
 fetch('/api/anl_ids',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(pay)})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  var ids=j.ids,mats=[],pos=0,W2=Math.min(window.innerWidth-150,1240);
  window.__ANDV={label:d+'_개별',mats:mats};
  function page(){
   if(!document.getElementById('anrcards'))return;
   var m2=document.getElementById('anrmsg');
   if(pos>=ids.length){
    if(m2)m2.textContent=ids.length+'장 · 그래프 '+mats.length+'장 · 위반 '
     +mats.filter(function(m){return m.verdict;}).length+'건';
    return;}
   fetch('/api/pattern_wave_bulk',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ids:ids.slice(pos,pos+40),rules:rules,
     alpha1:pay.alpha1,alpha2:pay.alpha2,alpha3:pay.alpha3})})
   .then(function(r){return r.json().then(function(x){if(!r.ok)throw new Error(x.detail||'실패');return x;});})
   .then(function(x){
    if(!document.getElementById('anrcards'))return;
    x.mats.forEach(function(w){mats.push(w);cc.appendChild(__ptCard(w,W2));});
    pos+=40;
    if(m2)m2.textContent='불러오는 중… '+Math.min(pos,ids.length)+'/'+ids.length+'장';
    setTimeout(page,30);})
   .catch(function(e2){if(m2)m2.textContent='실패: '+e2.message;});}
  page();})
 .catch(function(e){var m2=document.getElementById('anrmsg');
  if(m2)m2.textContent='실패: '+e.message;});}
function __anFull(){
 /* 기간 전체보기: 5일 평균 고정(사용자 결정) */
 var x=window.__ANAGG||{};
 var eq=__anEqpSel();
 __anRange(x.d0||'',x.d1||'',(eq||'전체 설비')+' 기간 전체 5일 평균 ('+(x.d0||'')+'~'+(x.d1||'')+')',5);}
function __anRenderMats(mats,W2){
 /* 카드 재료 배열 → HTML 문자열(오프스크린 렌더) */
 var host=document.createElement('div');
 host.style.cssText='position:absolute;left:-9999px;top:0;width:1200px';
 document.body.appendChild(host);
 var parts=[];
 mats.forEach(function(w){var c=__ptCard(w,W2);host.appendChild(c);
  parts.push(c.outerHTML);host.innerHTML='';});
 document.body.removeChild(host);
 return parts.join(String.fromCharCode(10));}
function __anExportFull(eq,done){
 /* 대화형 HTML: 달력+필터 갱신+일/월 클릭 그래프+전체보기 — 전부 파일 안에서 동작.
    v153: 응답 비대(수백 MB→Failed to fetch)를 막기 위해 월 단위로 나눠 수집. */
 var msg=document.getElementById('anmsg');
 msg.textContent=(eq||'전체 설비')+' 내보내기 — 데이터 수집 중…(캐시 경유)';
 var pay=__anPay({});if(eq)pay.eqp=[eq];
 var ag=window.__ANAGG||{},d0=ag.d0||'',d1=ag.d1||'';
 var mons=[];
 if(d0&&d1){var m0=d0.slice(0,7),m1=d1.slice(0,7),cur=m0;
  while(cur<=m1&&mons.length<600){mons.push(cur);
   var y=parseInt(cur.slice(0,4),10),mm=parseInt(cur.slice(5,7),10)+1;
   if(mm>12){y++;mm=1;}cur=y+'-'+(mm<10?'0':'')+mm;}}
 var jf=function(r){return r.json().then(function(j){
  if(!r.ok)throw new Error(j.detail||'실패');return j;});};
 var post=function(b){return fetch('/api/anl_export',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(jf);};
 var run;
 if(mons.length<=1){run=post(pay);}
 else{
  run=(function(){
   var x={rows:[],day_by:{},five_by:{},viol:{},viol_more:{},d0:d0,d1:d1,n:0};
   var chain=Promise.resolve();
   mons.forEach(function(mo,i){chain=chain.then(function(){
    msg.textContent=(eq||'전체 설비')+' 내보내기 — 수집 '+(i+1)+'/'+(mons.length+1);
    var b0=(mo===mons[0]?d0:mo+'-01'),b1=(mo===mons[mons.length-1]?d1:mo+'-31');
    return post(Object.assign({},pay,{mode:'day',d0:b0,d1:b1}))
     .then(function(j){x.rows=x.rows.concat(j.rows||[]);x.n+=j.n||0;
      Object.keys(j.day_by||{}).forEach(function(dv){
       x.day_by[dv]=(x.day_by[dv]||[]).concat(j.day_by[dv]);});
      Object.keys(j.viol||{}).forEach(function(d){x.viol[d]=j.viol[d];});
      Object.keys(j.viol_more||{}).forEach(function(d){x.viol_more[d]=j.viol_more[d];});})
     .catch(function(e){if(String(e.message).indexOf('조건에 맞는')<0)throw e;});});});
   chain=chain.then(function(){
    msg.textContent=(eq||'전체 설비')+' 내보내기 — 수집 '+(mons.length+1)+'/'+(mons.length+1)+' (5일 평균)';
    return post(Object.assign({},pay,{mode:'five'}))
     .then(function(j){x.five_by=j.five_by||{};});});
   return chain.then(function(){if(!x.rows.length)throw new Error('조건에 맞는 웨이퍼가 없습니다');return x;});
  })();}
 run
 .then(function(x){
  msg.textContent=(eq||'전체 설비')+' 내보내기 — 그래프 렌더 중…';
  setTimeout(function(){
   var W2=1100,DAY={},VIOL={},FIVE={};
   var byRule=function(mats,label){var o={};
    ['p1','p2','p3','p4','p5'].forEach(function(ru){
     var mm=mats.filter(function(w){return w.rule===ru;})
      .map(function(w){if(label)w.id=label;return w;});
     if(mm.length)o[ru]=__anRenderMats(mm,W2);});
    return o;};
   Object.keys(x.day_by).forEach(function(dv){
    x.day_by[dv].forEach(function(cd){
     (DAY[cd.d0]=DAY[cd.d0]||{})[dv]=byRule(cd.mats,cd.label
      +(dv?' [DSV '+dv.replace('dsv','')+']':' [전체]'));});});
   Object.keys(x.five_by).forEach(function(dv){
    FIVE[dv]=x.five_by[dv].map(function(cd){
     return [cd.label+(dv?' [DSV '+dv.replace('dsv','')+']':' [전체]'),
             byRule(cd.mats,cd.label)];});});
   Object.keys(x.viol).forEach(function(d){
    var o={_more:x.viol_more[d]||0};
    ['p1','p2','p3','p4','p5'].forEach(function(ru){
     var lst=x.viol[d].filter(function(w){return w.rule===ru;})
      .map(function(w){return {d:w.dsv||'',r:w.recipe||'',
        h:__anRenderMats([w],W2)};});
     if(lst.length)o[ru]=lst;});
    VIOL[d]=o;});
   var marks=__calMarksFor(eq?eq.toLowerCase():'',null);
   var J=function(o){return JSON.stringify(o).split('<'+'/').join('<'+String.fromCharCode(92)+'/');};
   var css='body{font-family:Malgun Gothic,Apple SD Gothic Neo,sans-serif;margin:20px;'
    +'color:#26303b;background:#fff}'
    +'.card{border:1px solid #e5e9f0;border-radius:12px;padding:8px 12px;margin:0 0 10px;'
    +'background:#fff}h1{font-size:17px}.sub{color:#5b6675;font-size:12.5px}'
    +'button[data-dl],.expb{display:none!important}'
    +'.exbtn{padding:4px 12px;border:1px solid #2456a6;border-radius:6px;background:#fff;'
    +'color:#2456a6;font-size:12px;cursor:pointer;margin-right:6px}';
   var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
   var title='분석자료 — '+(eq||'전체 설비')+' ('+x.d0+'~'+x.d1+')';
   var sc=[];
   sc.push('var ROWS='+J(x.rows)+';');
   sc.push('var DAY='+J(DAY)+',VIOL='+J(VIOL)+',VMORE='+J(x.viol_more)+';');
   sc.push('var FIVE='+J(FIVE)+',MARKS='+J(marks)+';');
   sc.push('var __calHtml='+__calHtml.toString()+';');
   sc.push('function sel(id){var o=[];document.querySelectorAll(id+" input:checked")'
    +'.forEach(function(c){o.push(c.value);});return o;}');
   sc.push('function rsel(){var o=[];document.querySelectorAll("#frule input:checked")'
    +'.forEach(function(c){o.push(c.value);});return o;}');
   sc.push('function agg(){var ds=sel("#fdsv"),rc=sel("#frc"),rs=rsel(),byd={},dl=[];'
    +'ROWS.forEach(function(r){'
    +'if(ds.length&&r[1]&&ds.indexOf(r[1])<0)return;'
    +'if(rc.length&&r[2]&&rc.indexOf(r[2])<0)return;'
    +'var bad=((rs.indexOf("p1")>=0&&r[3])||(rs.indexOf("p2")>=0&&r[4])'
    +'||(rs.indexOf("p3")>=0&&r[5])||(rs.indexOf("p4")>=0&&r[6])'
    +'||(rs.indexOf("p5")>=0&&r[7]))?1:0;'
    +'if(!byd[r[0]]){byd[r[0]]={bad:0,ok:0};dl.push(r[0]);}'
    +'if(bad)byd[r[0]].bad++;else byd[r[0]].ok++;});'
    +'return {byd:byd,dl:dl};}');
   sc.push('function redraw(){var a=agg();'
    +'document.getElementById("excal").innerHTML=__calHtml(a.byd,a.dl,"ex",MARKS);}');
   sc.push('function show(h){document.getElementById("exout").innerHTML=h||'
    +'"<div class=sub>이 날짜의 그래프는 내보내기에 포함되지 않았습니다</div>";'
    +'window.scrollTo(0,document.getElementById("exout").offsetTop-10);}');
   sc.push('function dsel(){return sel("#fdsv");}');
   sc.push('function dkeys(){var ds=dsel();return ds.length?ds:[""];}');
   sc.push('function pickMean(byDsv){if(!byDsv)return "";var rs=rsel(),h="";'
    +'dkeys().forEach(function(dv){var o=byDsv[dv];if(!o)return;'
    +'["p1","p2","p3","p4","p5"].forEach(function(ru){if(rs.indexOf(ru)>=0&&o[ru])h+=o[ru];});});'
    +'return h;}');
   sc.push('function pickViol(o){if(!o)return "";var rs=rsel(),ds=dsel(),rc=sel("#frc"),h="";'
    +'["p1","p2","p3","p4","p5"].forEach(function(ru){if(rs.indexOf(ru)<0||!o[ru])return;'
    +'o[ru].forEach(function(it){'
    +'if(ds.length&&it.d&&ds.indexOf(it.d)<0)return;'
    +'if(rc.length&&it.r&&rc.indexOf(it.r)<0)return;'
    +'h+=it.h;});});'
    +'return h;}');
   sc.push('function dayHtml(k){var h="";'
    +'if(VIOL[k]){var v=pickViol(VIOL[k]);'
    +'if(v)h+="<div style=font-size:12px;font-weight:700;color:#c0392b>"+k+" 위반 웨이퍼"'
    +'+(VIOL[k]._more?" (일부 — 외 "+VIOL[k]._more+"장은 앱에서)":"")+"</div>"+v;}'
    +'var dm=pickMean(DAY[k]);'
    +'if(dm)h+="<div style=font-size:12px;font-weight:700>"+k+" 일평균</div>"+dm;'
    +'return h;}');
   sc.push('function fiveHtml(){var h="";dkeys().forEach(function(dv){'
    +'(FIVE[dv]||[]).forEach(function(p){var rs=rsel(),b="";'
    +'["p1","p2","p3","p4","p5"].forEach(function(ru){if(rs.indexOf(ru)>=0&&p[1][ru])b+=p[1][ru];});'
    +'if(b)h+="<div style=font-size:12px;font-weight:700>"+p[0]+"</div>"+b;});});'
    +'return h;}');
   sc.push('document.addEventListener("click",function(e){'
    +'var d=e.target.closest?e.target.closest(".exday"):null;'
    +'if(d){show(dayHtml(d.getAttribute("data-day")));return;}'
    +'var m=e.target.closest?e.target.closest(".exmon"):null;'
    +'if(m){var mo=m.getAttribute("data-mon"),h="";'
    +'Object.keys(DAY).sort().forEach(function(k){if(k.slice(0,7)===mo)h+=dayHtml(k);});'
    +'show(h);return;}});');
   sc.push('function fillf(){var ds={},rc={};'
    +'ROWS.forEach(function(r){if(r[1])ds[r[1]]=(ds[r[1]]||0)+1;'
    +'if(r[2])rc[r[2]]=(rc[r[2]]||0)+1;});'
    +'var mk=function(box,o){var h="";Object.keys(o).sort().forEach(function(v){'
    +'h+="<label style=margin-right:8px><input type=checkbox value="+String.fromCharCode(39)+v+String.fromCharCode(39)+"> "'
    +'+v+" ("+o[v]+")</label>";});document.getElementById(box).innerHTML=h;};'
    +'mk("fdsv",ds);mk("frc",rc);}');
   sc.push('fillf();redraw();');
   var doc='<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>'
    +esc(title)+'</title><style>'+css+'</style></head><body>'
    +'<h1>'+esc(title)+'</h1>'
    +'<div class="sub">도구 '+window.__TOOLVER+' · 웨이퍼 '+x.n+'장 · 달력의 일/월을 누르면 그래프가 아래에 표시됩니다 · '
    +'일 클릭=일평균+위반 개별(일당 최대 '+(pay.cap||10)+'장) · '
    +'필터(유형·DSV·레시피) 변경 후 [달력 갱신] — 달력과 그래프 모두 반영'
    +'(DSV 미선택=전체 평균)</div>'
    +'<div style="font-size:12.5px;margin:8px 0">'
    +'유형 <span id="frule">'
    +'<label style="margin-right:8px"><input type="checkbox" value="p1" checked> 직선 하강</label>'
    +'<label style="margin-right:8px"><input type="checkbox" value="p2" checked> 아치</label>'
    +'<label style="margin-right:8px"><input type="checkbox" value="p3" checked> W패턴</label>'
    +'<label style="margin-right:8px"><input type="checkbox" value="p4" checked> 온도 계통</label>'
    +'<label style="margin-right:8px"><input type="checkbox" value="p5" checked> 압력 계통</label>'
    +'</span><br>DSV <span id="fdsv"></span><br>'
    +'레시피 <span id="frc"></span><br>'
    +'<button class="exbtn" onclick="redraw()">달력 갱신</button>'
    +'<button class="exbtn" onclick="show(fiveHtml())">기간 전체보기(5일 평균)</button></div>'
    +'<div id="excal"></div><div id="exout"></div>'
    +'<scr'+'ipt>'+sc.join(String.fromCharCode(10))+'</scr'+'ipt></body></html>';
   __dlSave(new Blob([doc],{type:'text/html;charset=utf-8'}),
            __dlfn('분석자료_'+(eq||'전체설비')+'_'+x.d0+'~'+x.d1)+'.html');
   msg.textContent=(eq||'전체 설비')+' 내보내기 완료';
   done&&done();},50);})
 .catch(function(e){msg.textContent='내보내기 실패: '+e.message;done&&done();});}
function __anExpCur(){
 __anExportFull(__anEqpSel()||'',null);}
function __anExpAll(){
 /* 설비별 일괄: 설비마다 기간 전체 N일 평균을 받아 HTML 1개씩 저장(순차) */
 var x=window.__ANAGG;
 if(!x){alert('먼저 [분석 시작]을 실행하세요');return;}
 var eqs=Object.keys((x.avail||{}).eqp||{}).sort();
 if(!eqs.length){alert('설비 정보가 없습니다');return;}
 var msg=document.getElementById('anmsg'),i=0;
 (function next(){
  if(i>=eqs.length){msg.textContent='일괄 내보내기 완료 — '+eqs.length+'개 파일';return;}
  var eq=eqs[i];
  msg.textContent='일괄 내보내기 '+(i+1)+'/'+eqs.length+' — '+eq+' 로드 중…';
  __anExportFull(eq,function(){i++;setTimeout(next,500);});})();}
function __calHtml(byd,dlist,pre,marks){
 /* 달력 HTML(공용): byd={날짜:{bad,ok}}, marks={날짜:[내역...]} — 마우스 올리면 내역 */
 marks=marks||{};
 var all=dlist.slice();Object.keys(marks).forEach(function(d){if(all.indexOf(d)<0)all.push(d);});
 var months=[];all.sort().forEach(function(d){var m=d.slice(0,7);
  if(months.indexOf(m)<0)months.push(m);});
 var ch='<div style="font-size:12.5px;font-weight:700;margin:0 0 6px">달력'
  +' <span style="font-weight:400;color:#8a94a3;font-size:11px">일별 불량/정상 —'
  +' 불량 10장 이상 = 빨간 테두리 · 보기 체크박스 반영</span></div>';
 months.forEach(function(mo){
  var Y=parseInt(mo.slice(0,4),10),M=parseInt(mo.slice(5,7),10);
  var first=new Date(Y,M-1,1).getDay(),ndays=new Date(Y,M,0).getDate();
  ch+='<div class="'+pre+'mon" data-mon="'+mo+'" style="font-size:12px;font-weight:700;'
   +'margin:6px 0 3px;cursor:pointer;text-decoration:underline" title="이 달 전체 그래프 보기">'
   +Y+'년 '+M+'월</div>'
   +'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:4px;max-width:980px">';
  var wd=['일','월','화','수','목','금','토'];
  for(var w=0;w<7;w++)ch+='<div style="font-size:10.5px;color:'
   +(w===0?'#c0392b':'#8a94a3')+';text-align:center">'+wd[w]+'</div>';
  for(var e=0;e<first;e++)ch+='<div></div>';
  for(var dd=1;dd<=ndays;dd++){
   var key=mo+'-'+(dd<10?'0':'')+dd,c=byd[key],mk=marks[key];
   /* 과반 이상 불량=빨강 · 미만=비율 그라데이션 주황(진할수록 비율 높음) */
   var tot=c?(c.bad+c.ok):0,ratio=tot?c.bad/tot:0;
   var red=c&&c.bad>0&&ratio>=0.5;
   var oa=c&&c.bad>0&&!red?(0.35+0.65*Math.min(1,ratio/0.5)):0;
   var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;')
    .replace(/</g,'&lt;').replace(/"/g,'&quot;');};
   var tip=mk?esc('불량 내역: '+mk.join(' / ')):(c?'이 날 그래프 보기':'');
   ch+='<div '+(c?'class="'+pre+'day" data-day="'+key+'" ':'')
    +(tip?'title="'+tip+'" ':'')
    +'style="min-height:50px;border-radius:8px;padding:3px 6px;font-size:11px;'
    +(c?'cursor:pointer;':'')
    +'border:'+(red?'3px solid #d64545'
      :(oa?('2px solid rgba(224,138,0,'+oa.toFixed(2)+')')
        :(mk?'2px solid #e08a00':'1px solid #e5e9f0')))+';'
    +'background:'+(c?(red?'#fff3f2'
      :(oa?('rgba(255,222,180,'+(0.25+0.5*Math.min(1,ratio/0.5)).toFixed(2)+')')
        :'#f8fafc')):'#fdfdfe')+'">'
    +'<div style="color:#8a94a3;font-size:10.5px">'+dd+'</div>'
    +(c?('<div style="color:'+(c.bad?'#c0392b':'#9aa3b0')+';font-weight:'
      +(c.bad?'700':'400')+'">불량 '+c.bad+'</div>'
      +'<div style="color:#5b6675">정상 '+c.ok+'</div>'):'')
    +(mk?'<div style="color:#b26a00;font-weight:700;overflow:hidden;white-space:nowrap;'
      +'text-overflow:ellipsis">\u25b2 '+esc(mk.join(' / '))+'</div>':'')
    +'</div>';}
  ch+='</div>';});
 return ch;}
function __ptRangeView(d0,d1,label){
 var x=window.__PTX; if(!x)return;
 var box=document.getElementById('ptdayout'); if(!box)return;
 var ids=[],rules=['p1','p2','p3','p4','p5'].filter(__ptShow);
 x.rows.forEach(function(r){if(r.date>=d0&&r.date<=d1)ids.push(r.id);});
 if(!ids.length){box.innerHTML='<div style="font-size:12px;color:#5b6675;margin:4px 0">'
  +label+': 웨이퍼 없음</div>';return;}
 box.innerHTML='<div class="card" style="padding:8px 12px;margin:0 0 8px" id="ptdvhead">'
  +'<div style="font-size:12.5px;font-weight:700">'+label+' 전체 그래프'
  +' <span id="ptdvmsg" style="font-weight:400;color:#8a94a3;font-size:11px">'
  +'불러오는 중… 0/'+ids.length+'장</span>'
  +'<button id="ptdvexp" style="float:right;font-size:11px;padding:2px 8px;'
  +'border:1px solid #2456a6;border-radius:6px;background:#fff;color:#2456a6;'
  +'cursor:pointer;display:none">HTML 내보내기</button>'
  +'<button id="ptdvcls" style="float:right;margin-right:6px;font-size:11px;'
  +'padding:2px 8px;border:1px solid #dde3ea;border-radius:6px;background:#fff;'
  +'cursor:pointer">닫기</button></div></div>'
  +'<div id="ptdvcards"></div>';
 document.getElementById('ptdvcls').onclick=function(){box.innerHTML='';window.__PTDV=null;};
 var mats=[],pos=0,W2=Math.min(window.innerWidth-150,1240);
 var cards=document.getElementById('ptdvcards');
 window.__PTDV={label:label,mats:mats};
 function page(){
  if(!document.getElementById('ptdvcards'))return;   /* 닫힘 */
  if(pos>=ids.length){
   document.getElementById('ptdvmsg').textContent=mats.length+'장 그래프 · 위반 '
    +mats.filter(function(m){return m.verdict;}).length+'건';
   var eb=document.getElementById('ptdvexp');eb.style.display='';
   eb.onclick=function(){__ptExportHtml();};
   return;}
  var chunk=ids.slice(pos,pos+40);
  fetch('/api/pattern_wave_bulk',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({ids:chunk,rules:rules})})
  .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
  .then(function(j){
   if(!document.getElementById('ptdvcards'))return;
   j.mats.forEach(function(w){mats.push(w);cards.appendChild(__ptCard(w,W2));});
   pos+=chunk.length;
   document.getElementById('ptdvmsg').textContent='불러오는 중… '+Math.min(pos,ids.length)+'/'+ids.length+'장';
   setTimeout(page,30);})
  .catch(function(e2){var m2=document.getElementById('ptdvmsg');
   if(m2)m2.textContent='실패: '+e2.message;});}
 page();}
function __ptExportHtml(){
 var dv=window.__PTDV; if(!dv||!dv.mats.length){alert('내보낼 그래프가 없습니다');return;}
 __ptExportFrom(dv.mats,'패턴인식_'+dv.label);}
function __ptExportFrom(matsIn,fname){
 /* 자립형 HTML(외부 참조 0 — 폐쇄망/고객 전달용) — 패턴/분석자료 공용 */
 var dv={label:fname,mats:matsIn};
 if(!dv.mats.length){alert('내보낼 그래프가 없습니다');return;}
 var W2=1100,host=document.createElement('div');
 host.style.cssText='position:absolute;left:-9999px;top:0;width:1200px';
 document.body.appendChild(host);
 var nbad=dv.mats.filter(function(m){return m.verdict;}).length;
 var css='body{font-family:Malgun Gothic,Apple SD Gothic Neo,sans-serif;margin:20px;'
  +'color:#26303b;background:#fff}'
  +'.card{border:1px solid #e5e9f0;border-radius:12px;padding:8px 12px;margin:0 0 10px;'
  +'background:#fff}h1{font-size:17px}.sub{color:#5b6675;font-size:12.5px}'
  +'button,.expb{display:none!important}';
 var head='<h1>패턴 인식 — '+String(dv.label).replace(/</g,'&lt;')+'</h1>'
  +'<div class="sub">그래프 '+dv.mats.length+'장 · 위반 '+nbad+'건'
  +' · 선분=기준 · 음영=선분-곡선 면적 · 회색=관심 스텝 밖</div>'
  +'<hr style="border:none;border-top:1px solid #e5e9f0;margin:10px 0">';
 var parts=[],i=0;
 var eb=document.getElementById('ptdvexp'),eb0=eb?eb.textContent:'';
 function step(){
  var end=Math.min(i+25,dv.mats.length);
  for(;i<end;i++){
   var c=__ptCard(dv.mats[i],W2);
   host.appendChild(c);parts.push(c.outerHTML);host.innerHTML='';}
  if(eb)eb.textContent='내보내는 중… '+i+'/'+dv.mats.length;
  if(i<dv.mats.length){setTimeout(step,20);return;}
  document.body.removeChild(host);
  if(eb)eb.textContent=eb0;
  var doc='<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>패턴 인식 '
   +String(dv.label).replace(/</g,'&lt;')+'</title><style>'+css+'</style></head><body>'
   +head+parts.join(String.fromCharCode(10))+'</body></html>';
  __dlSave(new Blob([doc],{type:'text/html;charset=utf-8'}),
           __dlfn(dv.label)+'.html');}
 step();}
document.addEventListener('click',function(e){
 var aD=e.target.closest?e.target.closest('.anday'):null;
 if(aD){__anDayView(aD.getAttribute('data-day'));return;}
 var aM=e.target.closest?e.target.closest('.anmon'):null;
 if(aM){var mo=aM.getAttribute('data-mon');
  __anRange(mo+'-01',mo+'-31',mo+' (월 전체)');return;}
 var dEl=e.target.closest?e.target.closest('.ptday'):null;
 if(dEl){var d=dEl.getAttribute('data-day');__ptRangeView(d,d,d);return;}
 var mEl=e.target.closest?e.target.closest('.ptmon'):null;
 if(mEl){var mo=mEl.getAttribute('data-mon');
  __ptRangeView(mo+'-01',mo+'-31',mo);return;}
 var el=e.target.closest?e.target.closest('.ptrow'):null;
 if(!el)return;
 var wid=el.getAttribute('data-ptw');
 var rules=(el.getAttribute('data-rule')||'p1').split(',');
 Promise.all(rules.map(function(r){return __ptWaveFetch(wid,r,false);}))
 .then(function(ws){
  __gmodal(function(){
   var wrap=document.createElement('div');
   var W=Math.min(window.innerWidth-180,1150);
   ws.forEach(function(w){wrap.appendChild(__ptCard(w,W));});
   return wrap;});})
 .catch(function(e2){alert('원본 파형 조회 실패: '+e2.message);});});
function __vzsave(){
 var sel=__vzsel(),sm=document.getElementById('vzsmsg');
 var p=document.getElementById('vzcsv').value.trim();
 var idset={},i;
 for(i=0;i<sel.w.length;i++)idset[sel.w[i]]=1;
 document.querySelectorAll('.vzbk:checked').forEach(function(c){
  var ids=(window.__VZB||{})[c.getAttribute('data-b')]||[];
  for(var k=0;k<ids.length;k++)idset[ids[k]]=1;});
 sel.w=Object.keys(idset);
 if(!sel.w.length){sm.textContent='웨이퍼 또는 기간 카드를 체크하세요';return;}
 if(!p){sm.textContent='CSV 경로/파일명을 입력하세요';return;}
 sm.textContent='저장 중…';
 fetch('/api/viz_save_ids',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({path:p,ids:sel.w})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(x){sm.textContent='저장 완료: '+x.path+' ('+x.n+'장)';})
 .catch(function(e){sm.textContent='실패: '+e.message;});}
function __sdevents(){
 var el=function(id){return document.getElementById(id);};
 var msg=el('sdevmsg');
 if(!el('sdev_csv').value.trim()){msg.textContent='CSV 경로를 입력하세요';return;}
 msg.textContent='사건 분석 중…';
 fetch('/api/sd_events',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({csv:el('sdev_csv').value.trim(),dsv:(el('sdv')||{}).value||'',
   evt_days:parseInt(el('sdev_e').value,10)||3,base_days:parseInt(el('sdev_b').value,10)||14})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){msg.textContent='사건 '+j.events.length+'건 · DSV '
   +((j.params&&j.params.dsv)||['(전체)']).join(',')
   +(j.skipped.length?' · 건너뜀 '+j.skipped.length+'건':'');
  window.__SDEJ=j;__sdevrender(j);})
 .catch(function(e){msg.textContent='실패: '+e.message;});}
function __sdevrender(j){
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var un=function(sn){return /^d?p/i.test(sn)?'bar':'\u2103';};
 var h='',i,k;
 var lbs=Object.keys(j.common||{});
 for(i=0;i<lbs.length;i++){var cm=j.common[lbs[i]];
  h+='<div style="border:1px solid #e7defc;border-radius:8px;padding:6px 10px;margin:6px 0;background:#fbfaff;font-size:12px">'
   +'<b>\u2018'+esc(lbs[i])+'\u2019 \uacf5\ud1b5 \ud6c4\ubcf4(\uc7ac\ubc1c \uc11c\uba85 \ud6c4\ubcf4)</b> \u2014 ';
  for(k=0;k<cm.length;k++)h+=(k?' \u00b7 ':'')+esc(cm[k].key)+' ('+cm[k].n+'/'+cm[k].total+'\uac74)';
  h+='</div>';}
 for(i=0;i<j.events.length;i++){var ev=j.events[i];
  h+='<div class="card" style="padding:8px 12px;margin:6px 0">'
   +'<div style="font-size:13px;font-weight:700">'+esc(ev.date)+' \u2014 '+esc(ev.label)
   +(ev.dsv?' <span style="font-weight:400;color:#6a4fa3">'+esc(ev.dsv)+'</span>':'')
   +(ev.eqp?' <span style="font-weight:400;color:#8a94a3">'+esc(ev.eqp)+'</span>':'')
   +(ev.memo?' <span style="font-weight:400;color:#8a94a3">\u00b7 '+esc(ev.memo)+'</span>':'')
   +' <span style="font-weight:400;font-size:11px;color:#8a94a3">\uc0ac\uac74 \ucc3d '+ev.evt_d0.slice(5)+'~'+ev.date.slice(5)
   +' \u00b7 \uae30\uc900 \ucc3d '+ev.base_d0.slice(5)+'~'+ev.base_d1.slice(5)+'</span> '
   +'<button class="sdevjmp" data-d0="'+ev.base_d0+'" data-d1="'+ev.date+'" style="font-size:11px;padding:3px 8px;'
   +'border:1px solid #dde3ea;border-radius:6px;background:#fff;cursor:pointer">\uae30\uac04 \ubcf4\uae30</button> '
   +'<button onclick="__sdevnar('+i+')" style="font-size:11px;padding:3px 8px;'
   +'border:1px solid #1f6f8b;border-radius:6px;background:#fff;color:#1f6f8b;cursor:pointer">LLM \uc0ac\uac74 \ubd84\uc11d</button></div>';
  var p;
  for(p=0;p<(ev.phys||[]).length;p++){var ph=ev.phys[p];
   h+='<div style="font-size:12px;color:#8b0000;font-weight:700;background:#fdecea;border-radius:6px;padding:2px 6px;margin:2px 0">'
    +'[\ubb3c\ub9ac \uc704\ubc18 \uacb9\uce68] '+esc(ph.kind)+' '+esc(ph.sensor||'')
    +(ph.io?' (\ubc38\ube0c '+esc(ph.io)+')':'')+' \u2014 '+ph.d0.slice(5)+'~'+ph.d1.slice(5)+'</div>';}
  if(!(ev.cands||[]).length)h+='<div style="font-size:12px;color:#8a94a3">\uc720\uc758\ud55c \ubcc0\ud654 \uc5c6\uc74c(z\u22651.5 \ubbf8\ub2ec)</div>';
  for(k=0;k<(ev.cands||[]).length;k++){var cd=ev.cands[k];
   h+='<div class="sdevrow" data-sn="'+esc(cd.sensor)+'" data-d0="'+ev.base_d0+'" data-d1="'+ev.date+'" '
    +'style="font-size:12px;padding:1px 4px;cursor:pointer'+(cd.virtual?';color:#6a4fa3;font-weight:600':'')+'">'
    +(k+1)+'. '+esc(cd.sensor)+' step'+cd.step+' \u2014 \u0394 '+(cd.delta>0?'+':'')+cd.delta+un(cd.sensor)
    +' (z='+cd.z+(cd.sig?' \u00b7 \uc720\uc758':'')+') \u00b7 '+esc(cd.kind)
    +(cd.onset?' \u00b7 \ubc1c\ud654 '+cd.onset.slice(5):'')
    +(cd.slope!=null?' \u00b7 \uc77c\ub2f9 '+(cd.slope>0?'+':'')+cd.slope:'')
    +'</div>';}
  h+='<div id="sdevnar'+i+'"></div></div>';}
 var out=document.getElementById('sdevout');
 out.innerHTML=h;
 out.querySelectorAll('.sdevrow').forEach(function(el2){el2.onclick=function(){
  __sdevfocus(el2.getAttribute('data-sn'),el2.getAttribute('data-d0'),el2.getAttribute('data-d1'));};});}
function __sdevfocus(sn,d0,d1){
 document.querySelectorAll('#sdsen .sdck').forEach(function(c){c.checked=(c.value===sn);});
 __sdjump(d0,d1);}
function __sdevnar(i){
 var box=document.getElementById('sdevnar'+i);
 box.innerHTML='<div style="padding:8px;color:#5b6675;font-size:12.5px">LLM \uc0ac\uac74 \ubd84\uc11d \uc911\u2026</div>';
 fetch('/api/sd_event_narrate',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({index:i})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'\uc2e4\ud328');return j;});})
 .then(function(x){__renderNar(box,x,'\uc0ac\uac74 \uc6d0\uc778 \ubd84\uc11d','');})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;padding:8px;font-size:12px">\uc2e4\ud328: '+e.message+'</div>';});}
function __sdpatterns(){
 var box=document.getElementById('sdpat'),sel=__sdsel();
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var un=function(sn){return /^d?p/i.test(sn)?'bar':'\u2103';};
 box.innerHTML='<div style="padding:8px;color:#5b6675;font-size:12.5px">선택 기간 패턴 분석 중\u2026</div>';
 fetch('/api/sd_patterns',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({dates:sel.days,sensors:sel.sen,mode:window.__SDPMODE||'prev'})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  var pb='';
  if(j.pressure&&j.pressure.length){
   pb='<div style="border:1px solid #f0dcdc;border-radius:8px;padding:8px 10px;margin:0 0 6px;background:#fffafa">'
    +'<div style="font-size:12.5px;font-weight:700;color:#33404f">\uc74c/\uc591\uc555 \uc0c1\ud0dc <span style="font-weight:400;color:#8a94a3;font-size:11.5px">outlet \uae30\uc900 \u00b7 \uc74c\uc555=outlet\uc774 \ub354 \ub192\uc74c=\uc5ed\ub958</span></div>';
   for(var pi=0;pi<Math.min(j.pressure.length,12);pi++){var pr=j.pressure[pi];
    var neg=pr.n_neg>0;
    pb+='<div style="font-size:12px;padding:1px 0;'+(neg?'color:#8b0000;font-weight:700':'color:#2e7d32')+'">'
     +esc(pr.sensor)+' step'+pr.step+': '+esc(pr.state)
     +' \u2014 \uc74c\uc555 '+pr.n_neg+'/'+pr.n_days+'\uc77c \u00b7 \ucd5c\uc800 '+(pr.min_raw>0?'+':'')+pr.min_raw+'bar</div>';}
   pb+='</div>';}
  if(!j.rows.length&&!j.jumps.length){
   box.innerHTML=pb+'<div style="padding:8px;color:#8a94a3;font-size:12.5px">'+esc(j.period[0])+'~'+esc(j.period[1])+' ('+j.n_days+'일): 문턱을 넘는 패턴 없음</div>';return;}
  if(j.mode==='seq'){
   var hs2=pb+'<details open style="border:1px solid #e7defc;border-radius:8px;padding:8px 10px;background:#fbfaff">'
    +'<summary style="font-size:12.5px;font-weight:600;cursor:pointer">선택일 순차 비교 '
    +'<span style="font-weight:400;color:#8a94a3">'+esc(j.period[0])+'~'+esc(j.period[1])+' ('+j.n_days+'일 체크) \u00b7 인접 선택일 간 변화량</span></summary>';
   var kinds2={},i2,r2;
   for(i2=0;i2<j.rows.length;i2++){r2=j.rows[i2];(kinds2[r2.kind]=kinds2[r2.kind]||[]).push(r2);}
   var order2=['연속 상승(점증)','연속 하락(점증)','급변(단발)','진동(오르내림)','변동'];
   for(var oi2=0;oi2<order2.length;oi2++){var kn2=order2[oi2];
    if(!kinds2[kn2])continue;
    hs2+='<div style="font-size:12px;font-weight:700;margin-top:6px;color:#33404f">'+esc(kn2)+' ('+kinds2[kn2].length+')</div>';
    var rs2=kinds2[kn2];
    for(i2=0;i2<Math.min(rs2.length,12);i2++){r2=rs2[i2];
     var seq='';
     for(var q2=0;q2<Math.min(r2.deltas.length,8);q2++){var d2=r2.deltas[q2];
      seq+=(q2?' \u00b7 ':'')+d2.a.slice(8)+'\u2192'+d2.b.slice(8)
        +(d2.gap>1?'(+'+d2.gap+'\uc77c)':'')+' '+(d2.delta>0?'+':'')+d2.delta;}
     if(r2.deltas.length>8)seq+=' \u2026';
     hs2+='<div style="font-size:12px;padding:1px 0'+(r2.virtual?';color:#6a4fa3;font-weight:600':'')+'">'
      +esc(r2.sensor)+' step'+r2.step+' \u2014 '+seq
      +' <span style="color:#8a94a3">(\ub204\uc801 '+(r2.cum>0?'+':'')+r2.cum+un(r2.sensor)+')</span></div>';}
    if(rs2.length>12)hs2+='<div style="font-size:11px;color:#8a94a3">\u2026 \uc678 '+(rs2.length-12)+'\uac74</div>';}
   hs2+='</details>';
   box.innerHTML=hs2;return;}
  var kinds={},i,r;
  for(i=0;i<j.rows.length;i++){r=j.rows[i];(kinds[r.kind]=kinds[r.kind]||[]).push(r);}
  var h=pb+'<details open style="border:1px solid #e7defc;border-radius:8px;padding:8px 10px;background:#fbfaff">'
   +'<summary style="font-size:12.5px;font-weight:600;cursor:pointer">선택 기간 패턴 '
   +'<span style="font-weight:400;color:#8a94a3">'+esc(j.period[0])+'~'+esc(j.period[1])+' ('+j.n_days+'일) \u00b7 직전 14일과 비교</span></summary>';
  var order=['신규 발생','점증 드리프트','점증 드리프트(문턱 근접)','차이 유지','복귀(해소)','완화 추세'];
  var seen={},names=order.concat(Object.keys(kinds));
  for(var oi=0;oi<names.length;oi++){var kn=names[oi];
   if(seen[kn]||!kinds[kn])continue;seen[kn]=1;
   h+='<div style="font-size:12px;font-weight:700;margin-top:6px;color:#33404f">'+esc(kn)+' ('+kinds[kn].length+')</div>';
   var rs=kinds[kn];
   for(i=0;i<Math.min(rs.length,12);i++){r=rs[i];
    h+='<div style="font-size:12px;padding:1px 0'+(r.virtual?';color:#6a4fa3;font-weight:600':'')+'">'
     +esc(r.sensor)+' step'+r.step+' \u2014 '+r.d0.slice(5)+' 기점, 처음 '+(r.first>0?'+':'')+r.first
     +' \u2192 끝 '+(r.last>0?'+':'')+r.last+un(r.sensor)
     +(Math.abs(r.slope)>=0.005?' (일당 '+(r.slope>0?'+':'')+r.slope+')':'')+'</div>';}
   if(rs.length>12)h+='<div style="font-size:11px;color:#8a94a3">\u2026 외 '+(rs.length-12)+'건</div>';}
  if(j.jumps.length){
   h+='<div style="font-size:12px;font-weight:700;margin-top:6px;color:#33404f">급변 ('+j.jumps.length+')</div>';
   for(i=0;i<Math.min(j.jumps.length,10);i++){var q=j.jumps[i];
    h+='<div style="font-size:12px;padding:1px 0'+(q.virtual?';color:#6a4fa3;font-weight:600':'')+'">'
     +esc(q.sensor)+' step'+q.step+' \u2014 '+q.date.slice(5)+' 전일 대비 '+(q.delta>0?'+':'')+q.delta+un(q.sensor)+'</div>';}}
  h+='</details>';
  box.innerHTML=h;})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;padding:8px;font-size:12.5px">패턴 분석 실패: '+e.message+'</div>';});}
function __sdnarrate(){
 var btn=document.getElementById('sdnr'),box=document.getElementById('sdnarr');
 btn.disabled=true;
 box.innerHTML='<div style="padding:10px;color:#5b6675;font-size:12.5px">LLM \uc815\ub9ac \uc911\u2026 (\ubaa8\ub378 \uc18d\ub3c4\uc5d0 \ub530\ub77c \uc218 \ubd84)</div>';
 fetch('/api/sd_narrate',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({dates:__sdsel().days,sensors:__sdsel().sen,pattern_mode:window.__SDPMODE||'prev'})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'\uc2e4\ud328');return j;});})
 .then(function(x){__renderNar(box,x,'\uc2a4\ud15d \ucc28\uc774 \uc815\ub9ac','');})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;padding:10px">\uc815\ub9ac \uc2e4\ud328: '+e.message+'</div>';})
 .then(function(){btn.disabled=false;});}
function __sdsel(){
 var sen=[],days=[];
 document.getElementById('sdsen').querySelectorAll('.sdck:checked').forEach(function(c){sen.push(c.value);});
 document.getElementById('sddt').querySelectorAll('.sdday:checked').forEach(function(c){days.push(c.value);});
 return {sen:sen,days:days};}
function __sdsetmode(btn){
 var m=btn.getAttribute('data-m');
 window.__SDMODE=m;
 document.getElementById('sdmode').querySelectorAll('button').forEach(function(b){
  var on=b.getAttribute('data-m')===m;
  b.style.background=on?'#2456a6':'#fff'; b.style.color=on?'#fff':'#2456a6';});
 __sdfetch(true);}
function __sdfetch(reset){
 var j=window.__SDJ; if(!j)return;
 if(reset){window.__SDOFF=0;document.getElementById('sdout').innerHTML='';}
 var sel=__sdsel();
 if(!sel.days.length){/* 기간 전부 해제 = 0건(빈 배열을 서버가 '전체'로 해석하는 것 방지) */
  document.getElementById('sdout').innerHTML='<div style="color:#8a94a3;font-size:12.5px;padding:10px">표시할 데이터가 없습니다(기간 체크 확인)</div>';
  document.getElementById('sdlegend').innerHTML='';
  document.getElementById('sdmore').style.display='none';return;}
 /* 범례 + 전 카드 공통 세로축 — 원단위(absmax) 또는 σ배(문턱 배수) */
 var sg=(window.__SDYMODE==='sig'),ymax=sg?3:0.5,i;
 for(i=0;i<sel.sen.length;i++){var sn0=sel.sen[i];
  if(sg){var th0=j.thr[sn0]||[1];var mn0=Math.min.apply(null,th0.map(function(t){return t||1;}));
   ymax=Math.max(ymax,Math.min(30,(j.absmax[sn0]||0)/Math.max(mn0,1e-6)));}
  else ymax=Math.max(ymax,j.absmax[sn0]||0);}
 window.__SDY=ymax;
 var iosel=[];document.querySelectorAll('#sdio .sdiock:checked').forEach(function(c){iosel.push(c.value);});
 window.__SDIOSEL=iosel;
 var lg='';
 for(i=0;i<sel.sen.length;i++)
  lg+='<span style="display:inline-block;margin-right:10px"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:'+GXCAT[i%GXCAT.length]+';vertical-align:-1px"></span> '+sel.sen[i]+'</span>';
 document.getElementById('sdlegend').innerHTML=lg+(sel.sen.length?
  '<span style="color:#8a94a3">세로축 ±'+ymax.toFixed(1)+(sg?'σ배 (문턱=2, 압력·온도 동일 눈높이)':' (원단위 ℃/bar · 골든 대비 차이 · 전 카드 공통)')+(window.__SDIOSEL&&window.__SDIOSEL.length?' · IO 레인: ■열림 ▨부분 ▢닫힘, 빨간 테두리=골든과 다름':'')+'</span>':'');
 var md=window.__SDMODE||'wafer';
 var body=(md==='wafer')?{mode:'wafer',offset:window.__SDOFF,limit:60,dates:sel.days}
  :{mode:'bucket',bucket_days:(md==='day1'?1:5),dates:sel.days};
 fetch('/api/stepdiff_page',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body)})
 .then(function(r){return r.json().then(function(x){if(!r.ok)throw new Error(x.detail||'실패');return x;});})
 .then(function(x){
  var out=document.getElementById('sdout'),h='';
  window.__SDITEMS=window.__SDITEMS||{};
  for(var k=0;k<x.items.length;k++){window.__SDITEMS[String(x.items[k].id).replace(/"/g,'')]=x.items[k];h+=__sdcard(x.items[k],j,sel.sen);}
  out.insertAdjacentHTML('beforeend',h);
  if(reset){var cb9=document.getElementById('sdcomp');if(cb9)cb9.innerHTML='';}
  __sdtipInit();
  if(window.__SDMODE==='wafer'){window.__SDOFF+=x.items.length;
   document.getElementById('sdmore').style.display=(window.__SDOFF<x.total)?'':'none';}
  else document.getElementById('sdmore').style.display='none';
  if(!x.items.length&&reset)out.innerHTML='<div style="color:#8a94a3;font-size:12.5px;padding:10px">표시할 데이터가 없습니다(기간 체크 확인)</div>';})
 .catch(function(e){document.getElementById('sdmsg').textContent='조회 실패: '+e.message;});}
function __sdcard(it,j,sen){
 var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 var iosel=window.__SDIOSEL||[],nio=(it.io?iosel.filter(function(q){return it.io[q];}).length:0);
 var cid=String(it.id).replace(/"/g,'');
 var W=1360,pl=44,pr=8,pt=8,pb=18+nio*12,H=150+nio*12,iw=W-pl-pr,ih=150-pt-18;
 var K=j.steps.length,ymax=window.__SDY||1;
 var y=function(v){return pt+ih/2-(v/ymax)*(ih/2);};
 var s='<svg width="'+W+'" height="'+H+'" style="max-width:100%">';
 var k,i;
 for(k=0;k<K;k++){var x0=pl+iw*k/K,x1=pl+iw*(k+1)/K;
  if(k%2)s+='<rect x="'+x0.toFixed(1)+'" y="'+pt+'" width="'+(x1-x0).toFixed(1)+'" height="'+ih+'" fill="#f4f6fa"/>';
  s+='<text x="'+((x0+x1)/2).toFixed(1)+'" y="'+(H-5)+'" text-anchor="middle" font-size="10" fill="#8a94a3">step'+j.steps[k]+'</text>';}
 s+='<line x1="'+pl+'" y1="'+y(0).toFixed(1)+'" x2="'+(W-pr)+'" y2="'+y(0).toFixed(1)+'" stroke="#b9c2cf" stroke-width="1"/>';
 s+='<text x="'+(pl-4)+'" y="'+(y(ymax)+8).toFixed(1)+'" text-anchor="end" font-size="9" fill="#8a94a3">+'+ymax.toFixed(1)+'</text>';
 s+='<text x="'+(pl-4)+'" y="'+y(0).toFixed(1)+'" text-anchor="end" font-size="9" fill="#8a94a3">0</text>';
 s+='<text x="'+(pl-4)+'" y="'+(y(-ymax)+2).toFixed(1)+'" text-anchor="end" font-size="9" fill="#8a94a3">-'+ymax.toFixed(1)+'</text>';
 var seg=iw/K,bw=Math.max(1.5,Math.min(18,(seg-12)/Math.max(1,sen.length)));
 for(k=0;k<K;k++){
  var xs=pl+seg*k+(seg-bw*sen.length)/2;
  for(i=0;i<sen.length;i++){
   var dv=(it.diff[sen[i]]||[])[k];
   if(dv==null)continue;
   var th=((j.thr||{})[sen[i]]||[])[k];
   var pv=(window.__SDYMODE==='sig')?(dv/Math.max(th||1e-6,1e-6)):dv;
   var yy=y(Math.max(-ymax,Math.min(ymax,pv)));
   var y0=y(0),by=Math.min(yy,y0),bh=Math.max(1,Math.abs(yy-y0));
   var hot=(th!=null&&Math.abs(dv)>th);/* 문턱 초과 = 빨간 테두리 강조 */
   var gs=((j.gstep||{})[sen[i]]||[])[k];
   var vio=((j.sign_rules||{})[sen[i]]==='+'&&gs!=null&&(dv+gs)<0);/* 부호 위반=원시 음수 */
   var tip=esc(sen[i])+' step'+j.steps[k]+': '+(dv>0?'+':'')+dv
    +(vio?' [부호 위반: 원시 '+(dv+gs).toFixed(2)+'bar\u003c0 — 골든은 양수]':'');
   var clickable=(it.n===1&&!it.__nocmp);
   s+='<rect x="'+(xs+i*bw).toFixed(1)+'" y="'+by.toFixed(1)+'" width="'+Math.max(1,bw-1).toFixed(1)+'" height="'+bh.toFixed(1)+'" fill="'+GXCAT[i%GXCAT.length]+'" fill-opacity="0.9"'
    +((hot||vio)?' stroke="#c0392b" stroke-width="'+(vio?'2.6':'1.4')+'"':'')
    +(clickable?' style="cursor:pointer" data-wid="'+cid+'" data-sn="'+esc(sen[i]).replace(/"/g,'&quot;')+'"':'')
    +' data-tip="'+tip.replace(/"/g,'&quot;')+'"'
    +(clickable?' data-tip2="클릭=원본 파형+골든 오버레이"':'')+'/>';}
  }
 /* IO(밸브) ON/OFF 레인 — 체크한 IO 만, 골든과 상태 다르면 빨간 테두리 */
 if(nio){var ro=0;
  for(var qi=0;qi<iosel.length;qi++){var io=iosel[qi];
   if(!it.io||!it.io[io])continue;
   var ly=pt+ih+4+ro*12;ro++;
   s+='<text x="'+(pl-4)+'" y="'+(ly+9)+'" text-anchor="end" font-size="8" fill="#5b6675">'+esc(io)+'</text>';
   for(k=0;k<K;k++){var fr=it.io[io][k];
    if(fr==null)continue;
    var gx0=pl+seg*k+2,gw=seg-4;
    var gfr=((j.io_golden||{})[io]||[])[k];
    var mis=(gfr!=null&&((gfr>0.2&&fr<=0.02)||(gfr<=0.02&&fr>0.2)||Math.abs(fr-gfr)>0.5));
    var fill=fr>0.5?'#455a74':(fr>0?'#9db1c7':'none');
    s+='<rect x="'+gx0.toFixed(1)+'" y="'+ly+'" width="'+gw.toFixed(1)+'" height="10" rx="2" fill="'+fill+'"'
     +(mis?' stroke="#c0392b" stroke-width="2"'
        :(fill==='none'?' stroke="#c8d2de" stroke-width="1"':''))
     +' data-tip="'+(esc(io)+' step'+j.steps[k]+': 열림 '+Math.round(fr*100)+'%'
     +(gfr!=null?' ('+(j.__cmpB?'B':'골든')+' '+Math.round(gfr*100)+'%)':'')
     +(mis?' \u26a0'+(j.__cmpB?'B와 다름':'골든과 다름'):'')).replace(/"/g,'&quot;')+'"/>';}}}
 s+='</svg>';
 return '<div class="card" style="padding:6px 10px;margin:0 0 6px">'
  +'<div style="font-size:11.5px;color:#33404f"><b>'+esc(it.id)+'</b>'
  +(it.n>1?' <span style="color:#8a94a3">('+it.n+'장 평균)</span>':' <span style="color:#8a94a3">'+esc(it.date)+'</span>')
  +(it.__nocmp?'':' <label style="float:right;font-size:11px;color:#5b6675;cursor:pointer">'
    +'<input type="checkbox" class="sdcmpck" data-id="'+cid+'" onchange="__sdcmp()" style="vertical-align:-2px"> 비교 선택</label>')
  +'</div>'+s+'</div>';}
function __sdcmpclear(){
 document.querySelectorAll('.sdcmpck').forEach(function(c){c.checked=false;});
 __sdcmp();}
function __sdcmp(){
 var cks=document.querySelectorAll('.sdcmpck:checked');
 var box=document.getElementById('sdcomp');
 if(!box){box=document.createElement('div');box.id='sdcomp';}
 /* 결과는 '마지막으로 체크한 카드' 바로 아래에 생성 — 위로 스크롤할 필요 없음 */
 var anch=cks.length?cks[cks.length-1].closest('.card'):null;
 if(anch)anch.insertAdjacentElement('afterend',box);
 if(cks.length<2){box.innerHTML=cks.length?'<div style="font-size:12px;color:#8a94a3;padding:4px 2px">비교할 카드를 하나 더 선택하면 이 자리에 A−B 차이 그래프가 생깁니다</div>':'';return;}
 if(cks.length>2){cks[0].checked=false;cks=document.querySelectorAll('.sdcmpck:checked');
  anch=cks[cks.length-1].closest('.card');
  if(anch)anch.insertAdjacentElement('afterend',box);}
 var A=(window.__SDITEMS||{})[cks[0].getAttribute('data-id')];
 var B=(window.__SDITEMS||{})[cks[1].getAttribute('data-id')];
 if(!A||!B){box.innerHTML='';return;}
 var j=window.__SDJ,d={},io={},sn,k;
 for(sn in A.diff){if(!B.diff[sn])continue;
  d[sn]=A.diff[sn].map(function(v,i){return (v==null||B.diff[sn][i]==null)?null:Math.round((v-B.diff[sn][i])*1000)/1000;});}
 if(A.io&&B.io){for(sn in A.io){if(B.io[sn])io[sn]=A.io[sn];}}
 var synth={id:A.id+'  \u2212  '+B.id,date:'',n:1,diff:d,io:io,__nocmp:1};
 var j2={};for(k in j)j2[k]=j[k];
 j2.io_golden=B.io||{};j2.__cmpB=1;j2.gstep=null;j2.sign_rules={};
 var sel=__sdsel();
 box.innerHTML='';
 /* 차이 그래프는 팝업으로 — 두 번째 체크 순간 바로 뜬다 */
 __gmodal(function(){
  var wrap=document.createElement('div');
  wrap.className='dlroot';
  wrap.innerHTML='<div style="font-size:13.5px;font-weight:700;margin:0 0 6px">'
   +__dlBtns('비교_'+String(A.id).replace(/[^0-9A-Za-z가-힣_.-]+/g,'')+'-'+String(B.id).replace(/[^0-9A-Za-z가-힣_.-]+/g,''),'')
   +'그래프 비교 — A \u2212 B '
   +'<span style="font-weight:400;color:#8a94a3;font-size:12px">(A='+A.id+', B='+B.id
   +' \u00b7 IO 레인=A 상태, 빨간 테두리=B와 다름 \u00b7 마우스=값)</span> '
   +'<a href="#" onclick="__sdcmpclear();return false" style="font-size:11.5px">선택 지우기</a></div>'
   +__sdcard(synth,j2,sel.sen);
  return wrap;});}
function __gbuild(){
 var el=function(id){return document.getElementById(id);};
 var b=el('gbgo');
 if(!el('gbf').value.trim()){el('gbmsg').textContent='정상 구간 폴더를 입력하세요';return;}
 b.disabled=true; el('gbmsg').textContent='통합 골든 생성 중 - 전량 읽어 평균(수 분 걸릴 수 있음)...';
 fetch('/api/gbuild',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({folder:el('gbf').value,eqp:el('gbe').value,dsv:el('gbd').value,
   recipe:el('gbr').value,months:el('gbm').value,store:el('gbst').value,name:el('gbn').value})})
 .then(function(r){return r.json().then(function(j){return {ok:r.ok,j:j};});})
 .then(function(x){ b.disabled=false;
  if(!x.ok){ el('gbmsg').textContent='실패: '+((x.j&&x.j.detail)||''); return; }
  el('gbmsg').textContent='';
  var h='<div class="card" style="padding:10px 12px;margin-top:8px"><b>저장 완료</b> - 읽은 웨이퍼 '+x.j.n_read+'장';
  (x.j.items||[]).forEach(function(it){
   h+='<div style="font-size:12.5px;margin-top:4px"><b>'+it.file+'</b> · 유형 '+it.recipe
     +' · '+it.n+'장 · 센서 '+it.n_sensors+'개'
     +(it.n_virtual?' <span style="color:#6a4fa3">(가상센서 '+it.n_virtual+'개 포함)</span>':'')+'</div>';});
  h+='</div>'; el('gbout').innerHTML=h; })
 .catch(function(e){ b.disabled=false; el('gbmsg').textContent='오류: '+e; });
}
function __mpIsPT(sn){ if(/heat/i.test(sn))return false; return /^([pt]|d[pt])_/i.test(sn); }
function __mpRender(){
 var out=document.getElementById('mpout'),msg=document.getElementById('mpmsg');
 function draw(j){
  window.__TSJ=window.__TSJ||j;
  var inc1=document.getElementById('mpall').checked;
  var esc=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var gs=(j.groups||[]).filter(function(g){return __mpIsPT(g.sensor)&&(inc1||g.n_occur>1);});
  msg.textContent=(j.ref?('기준: '+j.ref.name+' · '):'')+'패턴 '+gs.length+'개'
   +(j.cond?' · 조건 '+j.cond:'');
  if(!gs.length){out.innerHTML='<div style="color:#5b6675;padding:10px">해당하는 패턴이 없습니다 (P/T 센서 · '+(inc1?'전체':'다회 발생')+')</div>';return;}
  var h='';
  var KC2={'레벨 드리프트':'#d84315','볼록 패턴':'#7b1fa2','급변 이벤트':'#c0392b'};
  for(var gi=0;gi<gs.length;gi++){var g=gs[gi];
   var locs=g.loc?(' · 중심 '+Math.round(g.loc.c/6)+'%'):'';
   h+='<div style="border:1px solid #dde3ea;border-left:5px solid '+(KC2[g.kind]||'#5b6675')+';border-radius:10px;background:#fff;padding:10px 12px;margin:0 0 10px">'
    +'<div style="font-size:13px;margin-bottom:6px"><input type="checkbox" class="mpsel" data-g="'+gi+'" checked style="vertical-align:-2px"> '
    +'<b>'+esc(g.sensor)+'</b> step'+g.step+' · '+esc(g.kind)+'('+esc(g.direction||'')+locs+') — '
    +'<span style="color:#7b1fa2;font-weight:700">'+g.n_occur+'회 발생</span> · '+esc((g.months||[]).join(', '))+'</div>'
    +'<div style="display:flex;gap:8px;flex-wrap:wrap">';
   for(var oi=0;oi<g.occurrences.length;oi++){var o=g.occurrences[oi];
    var ci=(g.card_idx||[])[oi];
    var cd=(ci!=null&&j.cards)?j.cards[ci]:null;
    var sp='';
    if(cd&&cd.series){var vals=cd.series.filter(function(x){return x!=null;});
     if(vals.length){var mx=Math.max.apply(null,vals.map(Math.abs))||1;
      sp='<div style="font-family:monospace;font-size:11px;color:#5b6675;letter-spacing:1px">'
       +cd.series.map(function(x){return x==null?'':('\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588'[Math.min(7,Math.round(Math.abs(x)/mx*7))]);}).join('').slice(0,80)+'</div>';}}
    h+='<div data-i="'+(ci!=null?ci:'')+'" onclick="if(this.dataset.i)__tsdetail(this.dataset.i)" '
     +'style="border:1px solid #eef1f5;border-radius:8px;padding:6px 10px;min-width:170px;cursor:pointer;background:#fbfcfe">'
     +'<div style="font-size:12px;font-weight:600">'+(oi+1)+'회차 · '+esc(o.start)+'\u301c'+esc(String(o.end))+'</div>'
     +'<div style="font-size:11.5px">상태 '+esc(o.state)+(o.amp!=null?' · '+o.amp+'\u03c3':'')+'</div>'+sp+'</div>';}
   h+='</div></div>';}
  out.innerHTML=h;
  window.__MPGS=gs;}
 if(window.__TSJ){draw(window.__TSJ);return;}
 msg.textContent='추세 스캔 결과 조회 중\u2026';
 fetch('/api/trendscan_status').then(function(r){return r.json();}).then(function(st){
  if(st.state==='done'&&st.result){draw(st.result);}
  else{msg.textContent='추세 스캔 결과가 없습니다 — 추세 스캔 탭에서 먼저 실행하세요';}})
 .catch(function(e){msg.textContent='조회 실패: '+e.message;});}
function __mpNarrate(){
 var gs=window.__MPGS;
 if(!gs||!gs.length){alert('먼저 [최신 추세 스캔에서 불러오기]를 누르세요');return;}
 var sel=document.querySelectorAll('.mpsel');
 var pick=[];
 for(var i=0;i<sel.length;i++){if(sel[i].checked)pick.push(gs[parseInt(sel[i].dataset.g,10)]);}
 if(!pick.length){alert('분석할 그룹을 체크하세요');return;}
 var btn=document.getElementById('mpnr'),out=document.getElementById('mpout');
 btn.disabled=true;
 var old=document.getElementById('mpnarbox'); if(old)old.parentNode.removeChild(old);
 var box=document.createElement('div'); box.id='mpnarbox';
 box.innerHTML='<div style="padding:12px;color:#5b6675">주요 패턴 '+pick.length+'개 그룹 원인 분석 중\u2026 (LLM — 수 분, 비판자 설정 시 토론 포함)</div>';
 out.insertBefore(box,out.firstChild);
 fetch('/api/mp_narrate',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({groups:pick})})
 .then(function(r){return r.json().then(function(x){if(!r.ok)throw new Error(x.detail||'실패');return x;});})
 .then(function(x){__renderNar(box,x,'주요 패턴 분석','');})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;padding:10px">분석 실패: '+e.message+'</div>';})
 .then(function(){btn.disabled=false;});}
function __mpcsv(){
 var gs=window.__MPGS; if(!gs||!gs.length){alert('먼저 [최신 추세 스캔에서 불러오기]를 누르세요');return;}
 var sel=document.querySelectorAll('.mpsel');
 var rows=[['sensor','step','kind','direction','n_occur','months','occurrence','start','end','state','amp_sigma']];
 for(var i=0;i<sel.length;i++){if(!sel[i].checked)continue;
  var g=gs[parseInt(sel[i].dataset.g,10)];
  for(var oi=0;oi<g.occurrences.length;oi++){var o=g.occurrences[oi];
   rows.push([g.sensor,g.step,g.kind,g.direction||'',g.n_occur,(g.months||[]).join('|'),oi+1,o.start,o.end,o.state,o.amp==null?'':o.amp]);}}
 var csv='\\ufeff'+rows.map(function(r){return r.join(',');}).join('\\r\\n');
 var a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
 a.download='main_patterns.csv'; a.click();}
function __tscsv(){
 var j=window.__TSJ; if(!j||!j.stats)return;
 var rows=[];
 if(j.groups){rows.push(['[pattern_groups]']);
  rows.push(['sensor','step','kind','direction','n_occur','months','occurrences']);
  for(var gi=0;gi<j.groups.length;gi++){var g=j.groups[gi];
   rows.push([g.sensor,g.step,g.kind,g.direction||'',g.n_occur,g.months.join('|'),
    g.occurrences.map(function(o){return o.start+'~'+o.end+'('+o.state+')';}).join('|')]);}
  rows.push([''],['[step_stats]']);}
 rows.push(['sensor','step','n_days','mean_level_sigma','max_level_sigma','worst_day','sig_days','sig_pct','ratio_avg','ratio_max','bump_max_sigma','t']);
 for(var i=0;i<j.stats.length;i++){var r=j.stats[i];
  rows.push([r.sensor,r.step,r.n_days,r.mean_lv,r.max_lv,r.worst_day,r.sig_days,r.sig_pct,r.ratio_avg,r.ratio_max,r.bump_max,r.t]);}
 var csv='\\ufeff'+rows.map(function(r){return r.join(',');}).join('\\r\\n');
 var a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
 a.download='trendscan_stats_'+(j.dates[0]||'')+'_'+(j.dates[j.dates.length-1]||'')+'.csv';
 a.click();}
function __tsdetail(i){
 var j=window.__TSJ; if(!j)return;
 var cd=j.cards[parseInt(i,10)];
 var ds=(cd.series||[]).map(function(v,k){return v==null?null:j.dates[k];}).filter(Boolean);
 var pick=[ds[0],ds[Math.floor(ds.length/2)],ds[ds.length-1]].filter(function(v,k,arr){return v&&arr.indexOf(v)===k;});
 fetch('/api/ts_curves',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({sensor:cd.sensor,dates:pick})})
 .then(function(r){return r.json().then(function(x){if(!r.ok)throw new Error(x.detail||'실패');return x;});})
 .then(function(x){
  var extras=[];var ci=0;
  Object.keys(x.curves).sort().forEach(function(d){if(x.curves[d]){extras.push({label:d,vals:x.curves[d],color:GXCAT[(ci++)%GXCAT.length]});}});
  __gmodal(function(){
   return __chart(cd.sensor+' \u2014 기준('+x.ref_name+') vs 일평균',x.ref,null,null,null,extras,
    {W:Math.min(window.innerWidth-140,1160),H:Math.min(window.innerHeight-160,540),noexp:1});});})
 .catch(function(e){alert('상세 실패: '+e.message);});}
function __renderNar(box,j,label,extra){
 var esc9=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
 box.innerHTML='<div style="width:100%;border:1px solid #c9d6e8;border-radius:10px;background:#fbfcfe;padding:14px 18px">'
  +'<div style="font-size:11.5px;color:#8a94a3;margin-bottom:6px">'+label+' \u00b7 생성: '+esc9(j.engine)+'</div>'
  +(extra||'')
  +'<div style="white-space:pre-wrap;font-size:13px;line-height:1.65">'+esc9(j.narrative)+'</div>'
  +(j.debate?('<details style="margin-top:8px"><summary style="font-size:11.5px;color:#8a94a3;cursor:pointer">비판자 토론</summary>'
    +'<pre style="font-size:10.5px;white-space:pre-wrap;background:#f7f2ee;padding:8px;border-radius:6px">'+esc9(j.debate)+'</pre></details>'):'')
  +'<details style="margin-top:4px"><summary style="font-size:11.5px;color:#8a94a3;cursor:pointer">투입 프롬프트</summary>'
  +'<pre style="font-size:10.5px;white-space:pre-wrap;background:#f2f4f8;padding:8px;border-radius:6px">[system]'
  +esc9(j.prompt_system)+'\\n\\n[user]'+esc9(j.prompt_user)+'</pre></details></div>';}
function __focusForm(){
 var box=document.getElementById('ovgrid');
 box.innerHTML='<div style="width:100%;border:1px solid #d3b8e0;border-radius:10px;background:#fdfbfe;padding:14px 18px">'
  +'<b>집중 분석 — 불량 보고 전 구간의 웨이퍼 단위 전조 탐색</b>'
  +'<div style="font-size:12px;color:#5b6675;margin:4px 0 8px">보고일 이후는 조치가 섞여 증거로 안 씁니다. 보고 전 며칠을 웨이퍼 한 장 한 장 스텝별로 골든과 트렌드 비교합니다.</div>'
  +'<div style="margin:6px 0">보고 시작일 <input id="fcd" placeholder="2026-07-16" style="width:110px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"> '
  +'며칠 전까지 <input id="fcn" value="3" style="width:44px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px">일</div>'
  +'<div style="margin:6px 0;font-size:12.5px">골든 먼저 검증(선택): 대상 DSV <input id="fgt" placeholder="5" style="width:48px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"> '
  +'비교 DSV <input id="fgo" placeholder="1,2,3,6" style="width:100px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"> '
  +'유형 <input id="fgr" placeholder="분석 유형" style="width:90px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"> '
  +'폴더 <input id="fgf" placeholder="비우면 서버 골든 폴더 · ; 다중" style="width:230px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"> '
  +'<span style="color:#8a94a3">DSV 비우면 검증 생략</span></div>'
  +'<textarea id="fcr" placeholder="불량 보고서 내용(선택) 예: 7/16~7/30 현PC 불량" style="width:70%;height:52px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"></textarea>'
  +'<div style="margin-top:6px"><button onclick="__focusRun()" style="padding:7px 16px;border:none;border-radius:8px;background:#7b1fa2;color:#fff;cursor:pointer">분석 실행</button></div></div>';}
function __focusRun(){
 var box=document.getElementById('ovgrid');
 var body={date:document.getElementById('fcd').value.trim(),
  days_before:parseInt(document.getElementById('fcn').value,10)||3,
  golden_target:document.getElementById('fgt').value.trim(),
  golden_others:document.getElementById('fgo').value.trim(),
  golden_recipe:document.getElementById('fgr').value.trim(),
  golden_folders:document.getElementById('fgf').value.trim(),
  report:document.getElementById('fcr').value};
 box.innerHTML='<div style="padding:12px;color:#5b6675">집중 분석 중\u2026 (해당 구간 웨이퍼 전량 재판독 + 스텝별 트렌드 판정 + LLM — 수 분)</div>';
 fetch('/api/focus',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  var esc8=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var ex='';
  if(j.grid&&j.grid.sensors.length){
   var KC={'레벨높음':'216,67,21','레벨낮음':'21,101,192','기울기상이':'239,108,0','출렁임':'123,31,162','모양상이':'78,52,46'};
   var gh='<div style="font-size:12px;margin:0 0 10px;overflow:auto"><div style="font-weight:600;margin-bottom:4px">스텝\u00d7센서 트렌드 이상 지도 <span style="font-weight:normal;color:#8a94a3">(색=유형 \u00b7 농도=발생률 \u00b7 \u2197=악화 진행 \u00b7 셀: 최성일 발생률과 일자 추이)</span></div>';
   gh+='<div style="margin-bottom:6px">'+Object.keys(KC).map(function(k){return '<span style="display:inline-block;width:11px;height:11px;background:rgb('+KC[k]+');border-radius:3px;vertical-align:-1px"></span> '+k;}).join(' \u00a0 ')
    +' \u00a0 <span style="display:inline-block;width:11px;height:11px;border:2px dashed #6b7686;border-radius:3px;vertical-align:-1px"></span> M=\ubaa8\ub378 \ubcf5\uc6d0 \uae30\uc900(H \uc13c\uc11c\ub958\ub294 \ubaa8\ub378 \ucd9c\ub825 \uc5c6\uc74c)</div>';
   gh+='<table style="border-collapse:collapse;font-size:11px"><tr><td></td>'
    +j.grid.steps.map(function(st){return '<td style="padding:3px 6px;color:#5b6675;text-align:center">step'+st+'</td>';}).join('')+'</tr>';
   for(var gi=0;gi<j.grid.sensors.length;gi++){var sn=j.grid.sensors[gi];
    gh+='<tr><td style="padding:2px 8px 2px 0;white-space:nowrap;color:#33404f">'+esc8(sn)+'</td>';
    for(var gk=0;gk<j.grid.steps.length;gk++){var ce=(j.grid.cells[sn]||{})[String(j.grid.steps[gk])];
     if(!ce){gh+='<td style="border:1px solid #eef1f5;min-width:74px"></td>';continue;}
     var op=(0.18+0.72*ce.peak).toFixed(2);
     var kk=ce.kind.replace('\ubaa8\ub378\u00b7','');var isMod=(kk!==ce.kind);
     var txt=(isMod?'M ':'')+Math.round(ce.peak*100)+'%'+(ce.rising?' \u2197':'');
     var ser=ce.series.map(function(v){return Math.round(v*100);}).join('\u2192');
     gh+='<td style="border:'+(isMod?'2px dashed #6b7686':'1px solid #eef1f5')+';min-width:74px;padding:2px 5px;background:rgba('+(KC[kk]||'90,98,110')+','+op+');color:'+(ce.peak>0.55?'#fff':'#26313d')+'" title="'+esc8(ce.kind)+'">'
      +'<div style="font-weight:600">'+txt+'</div><div style="font-size:10px;opacity:.85">'+ser+'</div></td>';}
    gh+='</tr>';}
   gh+='</table></div>';
   ex+=gh;}
  if(j.golden_check){
   ex+=j.golden_check.valid
    ?'<div style="font-size:12px;background:#eef8ef;border:1px solid #bfe3c2;border-radius:8px;padding:8px 10px;margin-bottom:8px;color:#2e7d32">골든 유효성: 통과 — 비교 DSV 들과 모양 일치</div>'
    :'<div style="font-size:12px;background:#fdefec;border:1px solid #efc4ba;border-radius:8px;padding:8px 10px;margin-bottom:8px;color:#c0392b">골든 유효성 의심 — 모양 불일치 센서: '+esc8(j.golden_check.bad_sensors.join(', '))+' (해당 센서 판정은 신뢰도 낮춰 해석)</div>';}
  if(j.progression&&j.progression.length){
   ex+='<div style="font-size:12px;background:#f6effa;border:1px solid #e3cfee;border-radius:8px;padding:8px 10px;margin-bottom:8px">'
    +'<div style="color:#7b1fa2;font-weight:600;margin-bottom:2px">엔진: 악화 진행 (보고일로 갈수록 발생 비율 증가)</div>'
    +j.progression.map(function(x){return '<div>'+esc8(x)+'</div>';}).join('')+'</div>';}
  ex+='<div style="font-size:11.5px;color:#8a94a3;margin-bottom:6px">분석 창 '+esc8((j.window||[]).join(', '))+' \u00b7 '+j.n_wafers+'장</div>';
  __renderNar(box,j,'집중 분석',ex);})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;padding:10px">집중 분석 실패: '+e.message+'</div>';});}
function __gcmpForm(){
 var box=document.getElementById('ovgrid');
 box.innerHTML='<div style="width:100%;border:1px solid #b2d8d1;border-radius:10px;background:#fbfefd;padding:14px 18px">'
  +'<b>골든 검증 — 다른 DSV 골든과 비교해 현재 골든의 건전성 판정</b>'
  +'<div style="font-size:12px;color:#5b6675;margin:4px 0 8px">골든 데이터 폴더를 DSV별로 다시 집계하므로 수 분\u301c수십 분 걸립니다.</div>'
  +'<div style="margin:6px 0">대상 DSV <input id="gct" placeholder="5" style="width:56px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"> '
  +'비교 DSV <input id="gco" placeholder="1,2,3" style="width:110px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"> '
  +'유형 <input id="gcr" placeholder="예: 29a (비우면 전체)" style="width:140px;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"></div>'
  +'<div style="margin:6px 0">데이터 폴더 <input id="gcf" placeholder="비우면 서버 골든 폴더 · 여러 곳이면 ; 구분" style="width:60%;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px">'
  +'<div style="font-size:11.5px;color:#8a94a3;margin-top:2px">비교할 DSV 데이터가 서버 골든 폴더에 없으면, 대상+비교 DSV 데이터가 담긴 폴더(들)를 지정하세요</div></div>'
  +'<div style="margin-top:6px"><button onclick="__gcmpRun()" style="padding:7px 16px;border:none;border-radius:8px;background:#00695c;color:#fff;cursor:pointer">비교 실행</button></div></div>';}
function __gcmpRun(){
 var box=document.getElementById('ovgrid');
 var body={target:document.getElementById('gct').value.trim(),others:document.getElementById('gco').value.trim(),
  recipe:document.getElementById('gcr').value.trim(),folders:document.getElementById('gcf').value.trim()};
 box.innerHTML='<div style="padding:12px;color:#5b6675">골든 비교 중\u2026 (DSV별 골든 재집계 — 수 분\u301c수십 분, 콘솔 [gcmp] 진행 표시)</div>';
 fetch('/api/golden_compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  var esc8=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var ex='<details style="margin-bottom:6px"><summary style="font-size:11.5px;color:#8a94a3;cursor:pointer">비교 표(유의 차이만)</summary>'
   +'<pre style="font-size:10.5px;white-space:pre-wrap;background:#f2f4f8;padding:8px;border-radius:6px">'+esc8(j.table)+'</pre></details>';
  __renderNar(box,j,'골든 검증',ex);})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;padding:10px">골든 비교 실패: '+e.message+'</div>';});}
function __monthly(){__period('/api/monthly',{},'월간 원인 분석');}
function __warning(){__period('/api/warning',{days:14},'주간 조기경보(최근 14일)');}
function __period(url,body,label){
 var box=document.getElementById('ovgrid');
 box.innerHTML='<div style="padding:12px;color:#5b6675">'+label+' 분석 중\u2026 '
  +'(검사 데이터만으로 규칙 엔진 탐지 + 절차 프롬프트 — 수 분 걸립니다. 비판자 설정 시 토론 포함)</div>';
 fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(body)})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  var esc4=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var ef=j.engine_findings||{}, efh='';
  var part=function(t2,arr,f2){if(arr&&arr.length)efh+='<div style="margin:2px 0"><b>'+t2+'</b> '+arr.map(f2).join(' · ')+'</div>';};
  part('드리프트:',ef.drifts,function(d){return esc4(d.sensor)+'('+esc4(d.direction)+' '+d.amplitude_sigma+'\u03c3, '+esc4(d.start)+'\u2192)';});
  part('보상쌍:',ef.compensation_pairs,function(d){return esc4(d.pair.join('\u2194'));});
  part('헌팅:',ef.hunting,function(d){return esc4(d.sensor)+'(\u03c3 '+d.early_std+'\u2192'+d.late_std+')';});
  part('급변:',ef.jumps,function(d){return esc4(d.sensor)+'('+esc4(d.date)+', '+d.jump_sigma+'\u03c3)';});
  if(ef.ng_inflection)efh+='<div style="margin:2px 0"><b>불량률 변곡일:</b> '+esc4(ef.ng_inflection)+'</div>';
  box.innerHTML='<div style="width:100%;border:1px solid #c9d6e8;border-radius:10px;background:#fbfcfe;padding:14px 18px">'
   +'<div style="font-size:11.5px;color:#8a94a3;margin-bottom:6px">생성: '+esc4(j.engine)+'</div>'
   +(efh?('<div style="font-size:12px;background:#f2f6fb;border:1px solid #dbe5f1;border-radius:8px;padding:8px 10px;margin-bottom:8px">'
     +'<div style="color:#2456a6;font-weight:600;margin-bottom:2px">규칙 엔진 사전 탐지(검사 데이터만)</div>'+efh+'</div>'):'')
   +'<div style="white-space:pre-wrap;font-size:13px;line-height:1.65">'+esc4(j.narrative)+'</div>'
   +(j.debate?('<details style="margin-top:8px"><summary style="font-size:11.5px;color:#8a94a3;cursor:pointer">비판자 토론</summary>'
     +'<pre style="font-size:10.5px;white-space:pre-wrap;background:#f7f2ee;padding:8px;border-radius:6px">'+esc4(j.debate)+'</pre></details>'):'')
   +'<details style="margin-top:4px"><summary style="font-size:11.5px;color:#8a94a3;cursor:pointer">투입 프롬프트</summary>'
   +'<pre style="font-size:10.5px;white-space:pre-wrap;background:#f2f4f8;padding:8px;border-radius:6px">[system]'
   +esc4(j.prompt_system)+'</pre></details></div>';})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;padding:10px">'+label+' 실패: '+e.message+'</div>';});}
function __research(){
 var msg=document.getElementById('wmsg');
 if(!confirm('딥리서치를 시작합니다. 스캔된 전체 데이터로 장기(최대 7일) 탐색 세션을 엽니다. (정체되면 조기 종료, GPU 를 계속 사용)'))return;
 msg.textContent='딥리서치 세션 여는 중\u2026';
 fetch('/api/research',{method:'POST'}).then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){msg.textContent='딥리서치 진행 중 (세션 '+j.sid.slice(0,8)+'\u2026) — [리서치 상태/보고]로 확인. 코퍼스: '+j.corpus;})
 .catch(function(e){msg.textContent='리서치 시작 실패: '+e.message;});}
function __research_status(quiet){
 var box=document.getElementById('ovgrid'), msg=document.getElementById('wmsg');
 if(!quiet)msg.textContent='상태 조회 중\u2026';
 if(window.__rstT){clearInterval(window.__rstT);window.__rstT=null;}
 window.__rstT=setInterval(function(){
  var el=document.getElementById('rstbox');
  if(!el){clearInterval(window.__rstT);window.__rstT=null;return;}
  __research_status(true);},15000);
 fetch('/api/research_status').then(function(r){return r.json();})
 .then(function(st){
  if(!quiet)msg.textContent='';
  var rd0=document.getElementById('rdir');
  if(quiet&&rd0&&(rd0.value||document.activeElement===rd0))return;
  var se=st.session||{};
  var h='<div id="rstbox" style="width:100%"><h3 style="margin:4px 0 8px">딥리서치 상태'
   +' <span style="font-size:11px;color:#8a94a3;font-weight:normal">(15초마다 자동 갱신)</span></h3>'
   +'<div style="font-size:13px;margin-bottom:8px">상태: <b>'+st.status+'</b>'
   +(st.error?' · 오류: '+st.error:'')
   +(se.round!=null?' · 라운드 '+se.round:'')
   +(se.confidence!=null?' · 확신도 '+Number(se.confidence).toFixed(2):'')
   +(se.best?'<br>현재 최선: '+se.best:'')+'</div>';
  if(se.hypotheses&&se.hypotheses.length){
   h+='<div style="font-size:12.5px;margin:6px 0"><b>가설 장부('+se.hypotheses.length+'건)</b></div>'
    +'<table style="border-collapse:collapse;font-size:12px">';
   for(var i=0;i<Math.min(se.hypotheses.length,20);i++){var hp=se.hypotheses[i];
    h+='<tr><td style="padding:2px 8px;color:#5b6675">'+hp.status+'</td>'
     +'<td style="padding:2px 8px">'+hp.claim+'</td>'
     +'<td style="padding:2px 8px;text-align:right">'+(hp.score!=null?hp.score:'')+'</td></tr>';}
   h+='</table>';}
  if(st.status==='paused'){
   h+='<div style="border:1px solid #e0b070;background:#fdf8ef;border-radius:8px;padding:10px 14px;margin:8px 0">'
    +'<b>체크포인트 — 방향 확인</b><div style="font-size:12.5px;margin:4px 0 8px">'
    +'엔진이 잠시 멈추고 확인을 기다립니다. 현재 최선 가설과 장부를 보고 계속할지, 방향을 줄지 정해주세요.</div>'
    +'<input id="rdir" placeholder="방향 지시(선택): 예) Vessel 상부와 배기 상관에 집중해봐" '
    +'style="width:60%;font-size:12.5px;border:1px solid #dde3ea;border-radius:7px;padding:6px 8px"> '
    +'<button data-a="redirect" onclick="__ranswer(this.dataset.a)" style="padding:6px 12px;border:1px solid #b3541e;border-radius:7px;background:#fff;color:#b3541e;cursor:pointer">지시 후 계속</button> '
    +'<button data-a="continue" onclick="__ranswer(this.dataset.a)" style="padding:6px 12px;border:1px solid #2e7d32;border-radius:7px;background:#fff;color:#2e7d32;cursor:pointer">그대로 계속</button> '
    +'<button data-a="finish" onclick="__ranswer(this.dataset.a)" style="padding:6px 12px;border:1px solid #c0392b;border-radius:7px;background:#fff;color:#c0392b;cursor:pointer">종료 후 보고</button></div>';}
  h+='<div style="margin-top:10px"><button onclick="__research_report()" style="padding:6px 14px;'
   +'border:1px solid #2456a6;border-radius:7px;background:#fff;color:#2456a6;cursor:pointer">'
   +'중간/최종 보고 가져오기</button></div></div>';
  box.innerHTML=h;})
 .catch(function(e){msg.textContent='상태 실패: '+e.message;});}
function __ranswer(act){
 var t=(document.getElementById('rdir')||{}).value||'';
 fetch('/api/research_answer',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({action:act,text:t})})
 .then(function(r){return r.json();})
 .then(function(){__research_status();})
 .catch(function(e){document.getElementById('wmsg').textContent='응답 실패: '+e.message;});}
function __research_report(){
 var box=document.getElementById('ovgrid');
 box.innerHTML='<div style="padding:12px;color:#5b6675">보고 생성 중\u2026 (LLM 호출 — 수 분 걸릴 수 있음)</div>';
 fetch('/api/research_report').then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(rep){
  var esc3=function(t){return String(t==null?'':t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  var h='<div style="width:100%;border:1px solid #c9d6e8;border-radius:10px;background:#fbfcfe;padding:14px 18px">'
   +'<h3 style="margin:0 0 8px">딥리서치 보고 (확신도 '+Number(rep.confidence||0).toFixed(2)+')</h3>'
   +'<div style="white-space:pre-wrap;font-size:13px;line-height:1.6">'+esc3(rep.conclusion)+'</div>';
  var sec=function(t,arr){if(!arr||!arr.length)return '';
   return '<h4 style="margin:12px 0 4px">'+t+'</h4><ul style="font-size:12.5px;line-height:1.6">'
    +arr.map(function(x){return '<li>'+esc3(typeof x==='string'?x:JSON.stringify(x)).slice(0,500)+'</li>';}).join('')+'</ul>';};
  h+=sec('핵심 증거',rep.key_evidence)+sec('반증/약점',rep.counter_evidence)
   +sec('기각된 가설',(rep.rejected_alternatives||[]).map(function(a){return (a.claim||'')+' — '+(a.why||'');}))
   +sec('미해결 질문',rep.unresolved_questions)
   +sec('결론이 바뀔 조건',rep.what_would_change_conclusion);
  if(rep.debate){h+='<details style="margin-top:8px"><summary style="font-size:11.5px;color:#8a94a3;cursor:pointer">비판자 토론</summary>'
   +'<pre style="font-size:10.5px;white-space:pre-wrap;background:#f7f2ee;padding:8px;border-radius:6px">'+esc3(rep.debate)+'</pre>'
   +(rep.debated_narrative?('<div style="font-size:12px;margin-top:4px"><b>토론 반영 서술</b><div style="white-space:pre-wrap">'+esc3(rep.debated_narrative)+'</div></div>'):'')
   +'</details>';}
  var stt=rep.stats||{};
  h+='<div style="font-size:11.5px;color:#8a94a3;margin-top:8px">라운드 '+ (stt.rounds||'?')
   +' · 가설 '+(stt.hypotheses_explored||'?')+'(기각 '+(stt.hypotheses_rejected||'?')+') · 도구 '
   +(stt.tool_calls||'?')+'회 · 경과 '+Math.round((stt.elapsed_seconds||0)/60)+'분</div></div>';
  box.innerHTML=h;})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;padding:10px">보고 실패: '+e.message+'</div>';});}
function __dailycsv(){
 var msg=document.getElementById('wmsg');
 msg.textContent='CSV 생성 중\u2026';
 fetch('/api/daily_csv',{method:'POST'}).then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){msg.textContent='저장: '+j.dir+' ('+j.n_days+'일 × '+j.n_sensors+'센서, '+j.files.join(', ')+')';})
 .catch(function(e){msg.textContent='CSV 실패: '+e.message;});}
function __trend(){
 var box=document.getElementById('ovgrid'), msg=document.getElementById('wmsg');
 msg.textContent='기간 추세 집계 중\u2026';
 fetch('/api/trend').then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  msg.textContent='';
  var h='<div style="width:100%"><h3 style="margin:4px 0 8px">기간 추세 — 일별 불량률과 선행 센서</h3>';
  if(j.split) h+='<div style="font-size:13px;margin-bottom:8px">불량률 변곡일: <b style="color:#c0392b">'+j.split+'</b></div>';
  h+='<table style="border-collapse:collapse;font-size:12px;margin-bottom:12px"><tr>'
   +'<th style="padding:2px 10px;color:#5b6675">날짜</th><th style="padding:2px 10px;color:#5b6675">검사</th>'
   +'<th style="padding:2px 10px;color:#5b6675">불량</th><th style="padding:2px 10px;color:#5b6675">불량률</th>'
   +'<th style="padding:2px 10px"></th></tr>';
  for(var i=0;i<j.daily.length;i++){var d=j.daily[i];
   var bar=Math.round(d.rate*160);
   h+='<tr'+(j.split&&d.date>=j.split?' style="background:#fdf3f2"':'')+'>'
    +'<td style="padding:2px 10px;font-family:Consolas,monospace">'+d.date+'</td>'
    +'<td style="padding:2px 10px;text-align:right">'+d.n+'</td>'
    +'<td style="padding:2px 10px;text-align:right;color:#c0392b">'+d.ng+'</td>'
    +'<td style="padding:2px 10px;text-align:right">'+(d.rate*100).toFixed(1)+'%</td>'
    +'<td style="padding:2px 4px"><div style="height:9px;width:'+bar+'px;background:#d64545;border-radius:3px"></div></td></tr>';}
  h+='</table>';
  if(j.ranking&&j.ranking.length){
   h+='<div style="font-size:13px;margin:4px 0"><b>변곡 전후로 이탈이 커진 센서</b> — '+j.note+'</div>';
   h+='<table style="border-collapse:collapse;font-size:12px"><tr>'
    +'<th style="padding:2px 10px;color:#5b6675">센서</th><th style="padding:2px 10px;color:#5b6675">이전 평균σ</th>'
    +'<th style="padding:2px 10px;color:#5b6675">이후 평균σ</th><th style="padding:2px 10px;color:#5b6675">증가</th></tr>';
   for(var k=0;k<j.ranking.length;k++){var r2=j.ranking[k];
    h+='<tr><td style="padding:2px 10px;font-family:Consolas,monospace">'+r2.sensor+'</td>'
     +'<td style="padding:2px 10px;text-align:right">'+r2.pre.toFixed(2)+'</td>'
     +'<td style="padding:2px 10px;text-align:right">'+r2.post.toFixed(2)+'</td>'
     +'<td style="padding:2px 10px;text-align:right;font-weight:700;color:'+(r2.delta>0?'#c0392b':'#26303b')+'">'
     +(r2.delta>0?'+':'')+r2.delta.toFixed(2)+'</td></tr>';}
   h+='</table>';}
  h+='</div>';
  box.innerHTML=h;})
 .catch(function(e){msg.textContent='추세 실패: '+e.message;});}
function __scan(){
 var f=document.getElementById('wfolder').value.trim();
 var msg=document.getElementById('wmsg'), btn=document.getElementById('wscan');
 if(!f){msg.textContent='폴더 경로를 입력하세요';return;}
 btn.disabled=true; msg.textContent='판정 중\u2026 (장수가 많으면 수 분 걸립니다)';
 var fd=(document.getElementById('wfdsv')||{}).value||'';
 var fr=(document.getElementById('wfrc')||{}).value||'';
 var fm=(document.getElementById('wfmo')||{}).value||'';
 var fe=(document.getElementById('wfeq')||{}).value||'';
 fetch('/api/scan',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({folder:f,dsv:fd,recipe:fr,months:fm,eqp:fe})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){__scanShow(j);})
 .catch(function(e){msg.textContent='실패: '+e.message;})
 .then(function(){btn.disabled=false;});}
function __scanShow(j){
 var msg=document.getElementById('wmsg');
 WMETA=j.wafers;
 var ftxt=(j.n_selected<j.n_all_files)?(' · 필터: '+j.n_all_files+'장 중 '+j.n_selected+'장 선별'):'';
 msg.textContent='전체 '+WMETA.length+'장 (정상 '+j.n_ok+' · 불량 '+j.n_ng+' · 보류 '+j.n_na+')'+ftxt;
 __mkFilters(); document.getElementById('wfilters').style.display='';
 document.getElementById('wlist').style.display='';
 ['wexp','wtrd','wcsv','wmon','wwrn','wfoc','wgc','wrsh','wrst'].forEach(function(id){
  var el=document.getElementById(id); if(el)el.style.display='';});
 __applyF();}
function __restoreState(){
 /* 새로고침 복원: 서버 메모리/디스크의 마지막 결과 자동 로드 + 입력값 프리필 */
 fetch('/api/ui_state').then(function(r){return r.json();}).then(function(st){
  var pf=function(id,v){var e=document.getElementById(id);if(e&&!e.value&&v)e.value=v;};
  var pk=function(pid,sid,gf){var e=document.getElementById(pid);
   if(e&&gf&&!e.querySelector('select'))e.innerHTML='<select id="'+sid+'"><option value="'+gf+'">'+gf+'</option></select>';};
  if(st.sd_form){pf('sdf',st.sd_form.folder);pf('sde',st.sd_form.eqp);pf('sdv',st.sd_form.dsv);
   pf('sdr',st.sd_form.recipe);pf('sdm',st.sd_form.months);pf('sdst',st.sd_form.store);
   pf('gbst',st.sd_form.store);pf('sdcache',st.sd_form.cache);
   pk('sdpick','sdgf',st.sd_form.gfile);}
  if(st.ts_form){pf('tsf',st.ts_form.folder);pf('tse',st.ts_form.eqp);pf('tsd',st.ts_form.dsv);
   pf('tsr',st.ts_form.recipe);pf('tsm',st.ts_form.months);pf('tsst',st.ts_form.store);
   pf('tscache',st.ts_form.cache);pk('tspick','tsgf',st.ts_form.gfile);}
  if(st.sde_form){pf('sdev_csv',st.sde_form.csv);}
  if(st.vzrpt_form){pf('vzrpt',st.vzrpt_form.csv);}
  if(st.vz_form){pf('vzf',st.vz_form.folder);pf('vze',st.vz_form.eqp);pf('vzv',st.vz_form.dsv);
   pf('vzr',st.vz_form.recipe);pf('vzm',st.vz_form.months);pf('vzst',st.vz_form.store);
   pk('vzpick','vzgf',st.vz_form.gfile);}
  if(st.dv_form){pf('dv_f',st.dv_form.folder);}
  if(st.dv_forms&&st.dv_forms.length){window.__DVFX=st.dv_forms;__dvFxRender();}
  if(st.sde&&st.sde.events){window.__SDEJ=st.sde;__sdevrender(st.sde);}
  if(st.scan&&st.scan.wafers&&st.scan.wafers.length){
   if(st.folder){var f=document.getElementById('wfolder'); if(f&&!f.value)f.value=st.folder;}
   __scanShow(st.scan);
   document.getElementById('wmsg').textContent+=' · (복원됨)';}
  if(st.gx&&st.gx.recipes){window.__GXJ=st.gx;
   document.getElementById('gxmsg').textContent='웨이퍼 '+st.gx.n_wafers+'장 · '+st.gx.bucket_days+'일 단위 · (복원됨)';
   __gxrender(st.gx);}
  if(st.tscan_done){
   fetch('/api/trendscan_status').then(function(r2){return r2.json();}).then(function(ts){
    if(ts.state==='done'&&ts.result){window.__TSJ=ts.result;
     document.getElementById('tsmsg').textContent=ts.result.dates.length+'일 · 카드 '+ts.result.cards.length+'건 · 기준: '+ts.result.ref.name+' · (복원됨)';
     document.getElementById('tsnr').style.display='';
     __tsrender(ts.result);}
    else if(ts.state==='running'){
     document.getElementById('tsmsg').textContent='진행: '+(ts.progress||'…')+' · (백그라운드 계속 중)';
     document.getElementById('tsgo').disabled=true;
     if(window.__tsT)clearInterval(window.__tsT);
     window.__tsT=setInterval(__tspoll,3000);}});}
  if(st.sd_done){
   fetch('/api/stepdiff_status').then(function(r3){return r3.json();}).then(function(sd){
    if(sd.state==='done'&&sd.result){
     document.getElementById('sdmsg').textContent='웨이퍼 '+sd.result.n_wafers+'장 · 기준: '+sd.result.ref.name+' · (복원됨)';
     __sdinit(sd.result);}
    else if(sd.state==='running'){
     document.getElementById('sdmsg').textContent='진행: '+(sd.progress||'…')+' · (백그라운드 계속 중)';
     document.getElementById('sdgo').disabled=true;
     if(window.__sdT)clearInterval(window.__sdT);
     window.__sdT=setInterval(__sdpoll,3000);}});}
 }).catch(function(e){});}
__restoreState();
function __uniq(arr){var o={},r=[];for(var i=0;i<arr.length;i++){if(!o[arr[i]]){o[arr[i]]=1;r.push(arr[i]);}}return r.sort();}
function __mkChecks(el,vals,cls){el.innerHTML='';
 for(var i=0;i<vals.length;i++){var lb=document.createElement('label');
  lb.innerHTML='<input type="checkbox" class="'+cls+'" value="'+vals[i]+'" checked> '+vals[i];
  el.appendChild(lb);}
 var all=document.createElement('label'); all.style.background='#eef2f7';
 all.innerHTML='<input type="checkbox" checked> 전체';
 all.firstChild.onchange=function(){var cs=el.querySelectorAll('.'+cls);
  for(var i=0;i<cs.length;i++)cs[i].checked=this.checked; __applyF();};
 el.insertBefore(all, el.firstChild);
 var cs=el.querySelectorAll('.'+cls);
 for(var i=0;i<cs.length;i++)cs[i].onchange=__applyF;}
function __sensorList(){var seen={},out=[];
 for(var r in GOLD){var sn=GOLD[r].sensors;
  for(var i=0;i<sn.length;i++)if(!seen[sn[i]]){seen[sn[i]]=1;out.push(sn[i]);}}
 return out;}                               /* 설치 순서(골든 정렬) 유지 */
function __mkFilters(){
 __mkChecks(document.getElementById('frc'), __uniq(WMETA.map(function(w){return w.recipe;})), 'frc');
 __mkChecks(document.getElementById('fdt'), __uniq(WMETA.map(function(w){return w.date||'-';})), 'fdt');
 __mkChecks(document.getElementById('fvd'), ['정상','불량','보류'], 'fvd');
 var el=document.getElementById('fsn');
 if(!el.__done){el.__done=true;
  __mkChecksOrdered(el, __sensorList(), 'fsn', function(){if(window.__lastW)__ovDraw(window.__lastW);});}}
function __mkChecksOrdered(el,vals,cls,onch){__mkChecks(el,vals,cls);
 var cs=el.querySelectorAll('.'+cls);
 for(var i=0;i<cs.length;i++)cs[i].onchange=onch;
 el.querySelector('label input').onchange=function(){var c2=el.querySelectorAll('.'+cls);
  for(var i=0;i<c2.length;i++)c2[i].checked=this.checked; onch();};}
function __checked(cls){var r={},cs=document.querySelectorAll('.'+cls);
 for(var i=0;i<cs.length;i++)if(cs[i].checked)r[cs[i].value]=1; return r;}
function __vd(w){return (w.ratio==null)?'보류':((w.verdict==='NG'||w.ratio>1)?'불량':'정상');}
function __applyF(){
 var rc=__checked('frc'), dt=__checked('fdt'), vd=__checked('fvd');
 var L=document.getElementById('wlist'); L.innerHTML='';
 var n=0;
 for(var i=0;i<WMETA.length;i++)(function(w){
  if(!rc[w.recipe]||!dt[w.date||'-']||!vd[__vd(w)])return;
  n++;
  var r=document.createElement('div');r.className='wrow';r.__i=w.i;
  var ng=(__vd(w)==='불량');
  r.innerHTML='<span>'+w.wid+'</span><span class="vd'+(ng?' ng':'')+'">'+__vd(w)+'</span>'
   +'<span style="color:#8a94a3;font-size:11px">'+w.recipe+'</span>'
   +'<span class="sc" style="color:'+(ng?'#c0392b':'#26303b')+'">'+(w.ratio!=null?w.ratio.toFixed(2):'-')+'</span>';
  r.onclick=function(){var rs=document.querySelectorAll('.wrow');
   for(var j2=0;j2<rs.length;j2++)rs[j2].className='wrow';
   r.className='wrow sel';
   if(window.EMBED){var we=EMBED.wafers[String(w.i)];we.__i=null;__ovDraw(we);return;}
   fetch('/api/wafer?i='+w.i).then(function(x){return x.json();})
   .then(function(wd){wd.__i=w.i;__ovDraw(wd);})
   .catch(function(e){document.getElementById('wmsg').textContent='파형 로드 실패: '+e.message;});};
  L.appendChild(r);})(WMETA[i]);
 document.getElementById('fcnt').textContent='표시 '+n+' / 전체 '+WMETA.length+'장';}
function __chart(name,m,s,w,pp,extras,opts){
 extras=extras||[]; opts=opts||{};
 var holder=document.createElement('div');
 function build(a,b){
  var mm=m.slice(a,b+1),ss=s?s.slice(a,b+1):null,ww=w?w.slice(a,b+1):null,qq=pp?pp.slice(a,b+1):null;
  var exs=extras.map(function(e){return {label:e.label,color:e.color,dash:e.dash,width:e.width,v:e.vals.slice(a,b+1)};});
  var fl=opts.fill?{v:opts.fill.vals.slice(a,b+1),color:opts.fill.color||'#2b7fc4',
                    op:opts.fill.opacity||0.18}:null;
  var roi=opts.roi?[Math.max(0,opts.roi[0]-a),Math.min(b,opts.roi[1])-a]:null;
  var shd=(opts.shade||[]).filter(function(k){return k.i1>=a&&k.i0<=b;})
    .map(function(k){return {i0:Math.max(0,k.i0-a),i1:Math.min(b,k.i1)-a,color:k.color||'#d64545'};});
  /* 확대 시 현재 스텝 유지: 구간 시작(a) 이전의 마지막 라벨을 좌측 끝에 붙인다 */
  var tlsAll=(opts.toplabels||[]).slice().sort(function(p,q){return p.i-q.i;});
  var tls=tlsAll.filter(function(k){return k.i>=a&&k.i<=b;})
    .map(function(k){return {i:k.i-a,label:k.label};});
  if(a>0){var prev=null;for(var t0=0;t0<tlsAll.length;t0++){if(tlsAll[t0].i<a)prev=tlsAll[t0];}
   if(prev&&!(tls.length&&tls[0].i===0))tls.unshift({i:0,label:prev.label});}
  var mks=(opts.marks||[]).filter(function(k){return k.i>=a&&k.i<=b;})
    .map(function(k){return {i:k.i-a,color:k.color||'#d64545'};});
  var W=opts.W||560,H=opts.H||210,pl=52,pr=10,pt=26,pb=24,iw=W-pl-pr,ih=H-pt-pb,L=mm.length;
  var FIN=function(v){return v!=null&&isFinite(v);};
  var lo=1/0,hi=-1/0;
  for(var i=0;i<L;i++){
   if(FIN(mm[i])){var sv=(ss&&FIN(ss[i]))?ss[i]:0,a2=mm[i]-sv,b2=mm[i]+sv;
    if(a2<lo)lo=a2; if(b2>hi)hi=b2;}
   if(ww&&FIN(ww[i])){if(ww[i]<lo)lo=ww[i]; if(ww[i]>hi)hi=ww[i];}
   if(qq&&FIN(qq[i])){if(qq[i]<lo)lo=qq[i]; if(qq[i]>hi)hi=qq[i];}
   for(var e1=0;e1<exs.length;e1++){var xv=exs[e1].v[i];
    if(FIN(xv)){if(xv<lo)lo=xv; if(xv>hi)hi=xv;}}
   if(fl&&FIN(fl.v[i])){if(fl.v[i]<lo)lo=fl.v[i]; if(fl.v[i]>hi)hi=fl.v[i];}}
  if(!(lo<hi)&&!(lo===hi)){lo=0;hi=1;}
  if(!(hi-lo>1e-12)){lo-=1;hi+=1;} var pad=(hi-lo)*0.06; lo-=pad; hi+=pad;
  function X(i){return pl+iw*(L>1?i/(L-1):0.5);}
  function Y(v){return pt+ih*(1-(v-lo)/(hi-lo));}
  /* 표시용 감축: 폭보다 점이 많으면 픽셀 칸마다 min/max 두 점만 그린다.
     스파이크는 보존되고(극값 유지) 경로점은 폭 수준으로 줄어 렌더가 빨라진다.
     확대(build(a,b))하면 구간이 좁아져 자동으로 원본 해상도가 드러난다. */
  /* 버킷당 min/max 2점을 뽑으므로 버킷 수는 목표 점수의 절반(≈픽셀당 1.2점) */
  var DS=Math.max(150,Math.min(Math.floor(iw/2),opts.dsmax||700));
  function path(arr){
   /* 결측(null/NaN)은 선을 끊는다 — 0 으로 찍으면 없던 급강하가 보인다 */
   var d2='',gap=true;
   function put(i){d2+=(gap?(gap=false,'M'):' L')+X(i).toFixed(1)+' '+Y(arr[i]).toFixed(1);}
   if(L<=DS){for(var i=0;i<L;i++){if(FIN(arr[i]))put(i);else gap=true;}return d2||'M';}
   var step=L/DS;
   for(var k=0;k<DS;k++){
    var i0=Math.floor(k*step),i1=Math.min(L-1,Math.floor((k+1)*step)-1);
    if(i1<i0)i1=i0;
    var mn=-1,mx=-1;
    for(var j=i0;j<=i1;j++){if(!FIN(arr[j]))continue;
     if(mn<0||arr[j]<arr[mn])mn=j; if(mx<0||arr[j]>arr[mx])mx=j;}
    if(mn<0){gap=true;continue;}
    var aI=Math.min(mn,mx),bI=Math.max(mn,mx);
    put(aI); if(bI!==aI)put(bI);}
   return d2||'M';}
  var band='';
  if(ss){var seg=[],cur=[];
   for(var i=0;i<L;i++){if(FIN(mm[i])&&FIN(ss[i]))cur.push(i);
    else if(cur.length){seg.push(cur);cur=[];}}
   if(cur.length)seg.push(cur);
   for(var q3=0;q3<seg.length;q3++){var ix=seg[q3];
    band+='M'+X(ix[0]).toFixed(1)+' '+Y(mm[ix[0]]+ss[ix[0]]).toFixed(1);
    for(var p3=1;p3<ix.length;p3++)band+=' L'+X(ix[p3]).toFixed(1)+' '+Y(mm[ix[p3]]+ss[ix[p3]]).toFixed(1);
    for(var p4=ix.length-1;p4>=0;p4--)band+=' L'+X(ix[p4]).toFixed(1)+' '+Y(mm[ix[p4]]-ss[ix[p4]]).toFixed(1);
    band+=' Z';}}
  /* 기준선↔fill.vals 사이 음영(둘 다 유효한 연속 구간별 폴리곤) — '면적'을 눈으로 */
  var fillp='';
  if(fl){var runs=[],cur=null;
   for(var f1=0;f1<L;f1++){
    if(FIN(mm[f1])&&FIN(fl.v[f1])){if(!cur)cur=[f1,f1];cur[1]=f1;}
    else if(cur){runs.push(cur);cur=null;}}
   if(cur)runs.push(cur);
   for(var r1=0;r1<runs.length;r1++){var a1=runs[r1][0],b1=runs[r1][1];
    if(b1-a1<2)continue;
    var d3='M';
    for(var p1=a1;p1<=b1;p1++)d3+=(p1>a1?' L':'')+X(p1).toFixed(1)+' '+Y(mm[p1]).toFixed(1);
    for(var p2=b1;p2>=a1;p2--)d3+=' L'+X(p2).toFixed(1)+' '+Y(fl.v[p2]).toFixed(1);
    fillp+='<path d="'+d3+' Z" fill="'+fl.color+'" fill-opacity="'+fl.op+'" stroke="none"/>';}}
  /* 관심 스텝(roi) 밖은 회색 필터 — '여긴 판정 대상이 아니다' */
  var roip='';
  if(roi&&roi[1]>=roi[0]){
   if(roi[0]>0)roip+='<rect x="'+pl+'" y="'+pt+'" width="'+(X(roi[0])-pl).toFixed(1)
    +'" height="'+ih+'" fill="#8a94a3" fill-opacity="0.13"/>';
   if(roi[1]<L-1)roip+='<rect x="'+X(roi[1]).toFixed(1)+'" y="'+pt+'" width="'
    +((W-pr)-X(roi[1])).toFixed(1)+'" height="'+ih+'" fill="#8a94a3" fill-opacity="0.13"/>';}
  var shp='';
  for(var s1=0;s1<shd.length;s1++)
   shp+='<rect x="'+X(shd[s1].i0).toFixed(1)+'" y="'+pt+'" width="'+(X(shd[s1].i1)-X(shd[s1].i0)).toFixed(1)
    +'" height="'+ih+'" fill="'+shd[s1].color+'" fill-opacity="0.13"/>';
  var tlp='';
  for(var t1=0;t1<tls.length;t1++)
   tlp+='<text x="'+(X(tls[t1].i)+3).toFixed(1)+'" y="'+(pt+11)+'" font-size="9.5" '
    +'fill="#8a94a3">'+String(tls[t1].label).replace(/</g,'&lt;')+'</text>'
    +(tls[t1].i>0?'<line x1="'+X(tls[t1].i).toFixed(1)+'" y1="'+pt+'" x2="'+X(tls[t1].i).toFixed(1)
      +'" y2="'+(pt+ih)+'" stroke="#c9d1dc" stroke-dasharray="2 3"/>':'');
  var markp='';
  for(var k1=0;k1<mks.length;k1++){var ki=mks[k1].i;
   if(ki>=0&&ki<L&&FIN(mm[ki]))
    markp+='<circle cx="'+X(ki).toFixed(1)+'" cy="'+Y(mm[ki]).toFixed(1)
     +'" r="3.6" fill="'+mks[k1].color+'" stroke="#fff" stroke-width="1.2"/>';}
  var grid='';
  for(var g=0;g<5;g++){var v=lo+(hi-lo)*g/4,gy=Y(v);
   grid+='<line x1="'+pl+'" y1="'+gy.toFixed(1)+'" x2="'+(W-pr)+'" y2="'+gy.toFixed(1)+'" stroke="#e5e9f0"/>'
    +'<text x="'+(pl-6)+'" y="'+(gy+4).toFixed(1)+'" text-anchor="end" font-size="10" fill="#8a94a3">'+v.toPrecision(3)+'</text>';}
  var zoomed=(a>0||b<m.length-1);
  var html='<div class="card"><div class="t">'+name+((w||opts.noBaseTag)?'':' <span class="n">기준만</span>')
   +(zoomed?' <span class="n" style="color:#2456a6">확대 '+a+'~'+b+' (더블클릭=원복)</span>':'')
   +(opts.noexp?'':' <span class="expb" title="크게 보기" style="float:right;cursor:pointer;color:#8a94a3;font-size:14px">\u26f6</span>')+'</div>'
   +'<div class="chwrap" style="position:relative">'
   +'<svg class="ov" width="'+W+'" height="'+H+'">'+grid
   +roip+shp+tlp
   +(band?'<path d="'+band+'" fill="#2b7fc4" fill-opacity="0.14"/>':'')
   +fillp
   +'<path d="'+path(mm)+'" fill="none" stroke="'+(opts.baseColor||'#2456a6')+'" stroke-width="'+(opts.baseWidth||1.8)+'"/>'
   +(qq?'<path d="'+path(qq)+'" fill="none" stroke="#e08a00" stroke-width="1.5" stroke-dasharray="5 3"/>':'')
   +exs.map(function(e){return '<path d="'+path(e.v)+'" fill="none" stroke="'+e.color+'" stroke-width="'+(e.width||1.7)+'"'+(e.dash==='none'?'':' stroke-dasharray="'+(e.dash||'6 3')+'"')+'/>';}).join('')
   +(ww?'<path d="'+path(ww)+'" fill="none" stroke="#d64545" stroke-width="1.6"/>':'')
   +'<line class="hvx" x1="0" y1="'+pt+'" x2="0" y2="'+(pt+ih)+'" stroke="#8a94a3" stroke-dasharray="3 3" style="display:none"/>'
   +'<circle class="hvc" r="3.2" fill="#2456a6" stroke="#fff" stroke-width="1.1" style="display:none"/>'
   +(ww?'<circle class="hvw" r="3.2" fill="#d64545" stroke="#fff" stroke-width="1.1" style="display:none"/>':'')
   +markp
   +'<text x="'+pl+'" y="'+(H-6)+'" font-size="10" fill="#8a94a3">'+a+'</text>'
   +(opts.notp?'':'<text x="'+(W-pr)+'" y="'+(H-6)+'" text-anchor="end" font-size="10" fill="#8a94a3">타임포인트 '+b+'</text>')
   +'</svg><div class="hvtip" style="display:none"></div></div></div>';
  holder.innerHTML=html;
  var svg=holder.querySelector('svg');
  /* 내보내기(PNG)용 제목·레전드 — 화면은 HTML 로, 이미지는 svg 안에 그려 넣는다 */
  var lgm=(opts.lg&&opts.lg.length)?opts.lg.slice():[{label:(opts.baseLabel||'기준'),color:(opts.baseColor||'#2456a6')}]
   .concat(ww?[{label:(opts.wLabel||'웨이퍼'),color:'#d64545'}]:[])
   .concat(qq?[{label:(opts.pLabel||'이전'),color:'#e08a00'}]:[])
   .concat(exs.map(function(e){return {label:e.label,color:e.color};}));
  svg.__meta={title:String(name||''),lg:lgm};
  wireHover(svg,{m:mm,s:ss||mm.map(function(){return 0;}),w:ww||null,p:qq||null,ex:exs,
   bl:(opts.baseLabel||null),nosig:!s,
   pl:pl,pt:pt,iw:iw,ih:ih,lo:lo,hi:hi,L:L,x0:a});
  wireZoom(svg,pl,pt,iw,ih,L,function(i0,i1){
   if(i0==null)build(0,m.length-1); else build(a+i0,a+i1);});
  var eb=holder.querySelector('.expb');
  if(eb)eb.addEventListener('click',function(){
   __gmodal(function(){
    /* 옵션(스텝 라벨·관심구간·음영·마크·레전드) 유지한 채 크게 */
    var o2={};Object.keys(opts).forEach(function(k){o2[k]=opts[k];});
    o2.W=Math.min(window.innerWidth-140,1160);o2.H=Math.min(window.innerHeight-160,540);o2.noexp=1;
    return __chart(name,m,s,w,pp,extras,o2);});});}
 build(0,m.length-1);
 return holder;}
function __heatCol(z){ if(z==null)return '#f4f6f9';
 var t=Math.min(1,Math.max(0,(z-1)/5));               /* z=1 흰색 -> z>=6 진빨강 */
 if(z<=1) return '#f4f6f9';
 var r=255, gg=Math.round(235-165*t), bb=Math.round(220-190*t);
 return 'rgb('+r+','+gg+','+bb+')'; }
function __cellCol(zg,zr){                        /* 4상태 + 강도 계조(3σ 옅음 → 10σ+ 진함) */
 var G=(zg!=null&&zg>=3), R=(zr!=null&&zr>=3);
 var z=Math.max(zg||0, zr||0);
 var a=0.25+0.6*Math.min(1,(z-3)/7);              /* 3σ=0.25 → 10σ+=0.85 */
 if(G&&R) return 'rgba(214,69,69,'+a.toFixed(2)+')';    /* 빨강: 발화 */
 if(G&&!R) return 'rgba(224,138,0,'+a.toFixed(2)+')';   /* 주황: 전파/지속 */
 if(!G&&R) return 'rgba(200,170,30,'+a.toFixed(2)+')';  /* 노랑: 조기 */
 return '#f4f6f9';}
function __segline(h){
 /* 센서명→구간 근사 매핑으로 '문제 구간' 헤드라인 생성(지도 데이터 기반) */
 var SEG=[["Reservoir",/^reservoir/i],["충전 미터링",/(slowfill|quickfill)/i],
  ["공급 라인",/(leakdetect|h_0052?_|h_0161|p_005|h_015|t_005|t_002|h_005_)/i],
  ["Front 상부 유입",/(h_003|h_004|h_0162|p_008|h_0053)/i],
  ["Back 하부 유입",/(t_001|h_001|h_002|p_007)/i],
  ["Vessel 상부",/^vesselup/i],["Vessel 하부",/^vesseldown/i],
  ["챔버 주변",/^d_/i],["배기/벤트",/(h_006|p_003|t_003|pulsevent|slowvent|_press$)/i]];
 var agg={};
 for(var r=0;r<h.sensors.length;r++){
  var mz=0; for(var c=0;c<h.steps.length;c++){var v=h.maxz[r][c]; if(v!=null&&v>mz)mz=v;}
  if(mz<3)continue;
  var seg="기타";
  for(var k2=0;k2<SEG.length;k2++)if(SEG[k2][1].test(h.sensors[r])){seg=SEG[k2][0];break;}
  var a2=agg[seg]||(agg[seg]={max:0,n:0}); a2.n++; if(mz>a2.max)a2.max=mz;}
 var ks=Object.keys(agg).sort(function(x,y){return agg[y].max-agg[x].max;});
 if(!ks.length)return '';
 return '<div style="font-size:12.5px;margin:0 0 6px"><b>문제 구간(심각도순):</b> '
  +ks.slice(0,4).map(function(k3){return k3+' <span style="color:#c0392b">'+agg[k3].max.toFixed(0)
   +'σ</span>(센서 '+agg[k3].n+')';}).join(' &nbsp;→&nbsp; ')+'</div>';}
function __heatEl(h){
 var d=document.createElement('div'); d.style.cssText='width:100%;overflow-x:auto;margin:0 0 10px';
 var html=__segline(h)
  +'<div style="font-size:12.5px;color:#3a4654;margin:2px 0 6px"><b>스텝 × 센서 이탈 지도</b>'
  +' — 칸 = 골든 대비 최대 편차(σ). 색: <span style="background:#f3b6b6;padding:0 6px;border-radius:3px">발화 후보(골든·잔차 모두 3σ↑)</span>'
  +' <span style="background:#f8d5a8;padding:0 6px;border-radius:3px">오염 전파(골든만)</span>'
  +' <span style="background:#f5eec0;padding:0 6px;border-radius:3px">조기 신호(잔차만)</span>'
  +' · 칸에 마우스를 올리면 골든/잔차 수치</div>'
  +'<div style="font-size:12.5px;color:#3a4654;margin:0 0 6px">';
 if(h.onset_r&&h.onset_r.length){var f=h.onset_r[0];
  html+='<b style="color:#c0392b">발화 후보(모델이 처음 놀란 지점): '+f.sensor+' (스텝 '+f.step+', tp '+f.tp+')</b> &nbsp;·&nbsp; ';}
 if(h.onset&&h.onset.length){var o=h.onset[0];
  html+='골든 최초 이탈: '+o.sensor+' (스텝 '+o.step+', tp '+o.tp+', '+o.z+'σ)';
  if(h.onset.length>1){html+=' → '+h.onset.slice(1,4).map(function(x){return x.sensor+'(tp '+x.tp+')';}).join(' → ');}}
 if(!(h.onset&&h.onset.length)&&!(h.onset_r&&h.onset_r.length))
  html+='<span style="color:#2e9e73">3σ 지속 이탈 없음</span>';
 html+='</div><table style="border-collapse:collapse;font-size:11.5px">';
 html+='<tr><th style="text-align:left;padding:2px 8px;color:#5b6675">센서\\스텝</th>';
 for(var c=0;c<h.steps.length;c++)html+='<th style="padding:2px 8px;color:#5b6675">'+h.steps[c]+'</th>';
 html+='</tr>';
 for(var r2=0;r2<h.sensors.length;r2++){
  html+='<tr><td style="padding:2px 8px;font-family:Consolas,monospace;white-space:nowrap">'+h.sensors[r2]+'</td>';
  for(var c2=0;c2<h.steps.length;c2++){var z=h.maxz[r2][c2];
   var zr=(h.maxzr&&h.maxzr[r2])?h.maxzr[r2][c2]:null;
   html+='<td title="골든 '+(z==null?'-':z.toFixed(1))+'σ · 잔차 '+(zr==null?'-':zr.toFixed(1))+'σ" '
    +'style="padding:2px 8px;text-align:center;background:'+__cellCol(z,zr)+';'
    +(z!=null&&z>=3?'font-weight:700;color:#7a1010;':'color:#5b6675;')+'border:1px solid #fff">'
    +(z==null?'-':z.toFixed(1))+'</td>';}
  html+='</tr>';}
 html+='</table>';
 if(h.cart&&h.cart.names.length){
  var mx=1e-9;
  for(var r3=0;r3<h.cart.dev.length;r3++)for(var c3=0;c3<h.cart.dev[r3].length;c3++){
   var v=h.cart.dev[r3][c3]; if(v!=null&&Math.abs(v)>mx)mx=Math.abs(v);}
  html+='<div style="font-size:12.5px;color:#3a4654;margin:12px 0 6px"><b>카트리지 목표추종</b>'
   +' — 칸 = 그 스텝에서 실측(Cart) − 목표(Heater) 최대편차(℃).'
   +' 파랑 = 목표 미달(가열 부족), 빨강 = 초과. ※ 존 실온이 아니라 카트리지 거동 기준.</div>';
  html+='<table style="border-collapse:collapse;font-size:11.5px">';
  html+='<tr><th style="text-align:left;padding:2px 8px;color:#5b6675">카트리지\\스텝</th>';
  for(var c4=0;c4<h.steps.length;c4++)html+='<th style="padding:2px 8px;color:#5b6675">'+h.steps[c4]+'</th>';
  html+='</tr>';
  for(var r4=0;r4<h.cart.names.length;r4++){
   html+='<tr><td style="padding:2px 8px;font-family:Consolas,monospace;white-space:nowrap">'+h.cart.names[r4]+'</td>';
   for(var c5=0;c5<h.steps.length;c5++){var v2=h.cart.dev[r4][c5];
    var bg='#f4f6f9', fw='';
    if(v2!=null&&Math.abs(v2)>0.5){var t2=Math.min(1,Math.abs(v2)/Math.max(5,mx));
     bg=(v2<0)?('rgba(36,86,166,'+(0.12+0.45*t2).toFixed(2)+')')
              :('rgba(214,69,69,'+(0.12+0.45*t2).toFixed(2)+')');
     if(Math.abs(v2)>=3)fw='font-weight:700;';}
    html+='<td style="padding:2px 8px;text-align:center;background:'+bg+';'+fw+'border:1px solid #fff">'
     +(v2==null?'-':(v2>0?'+':'')+v2.toFixed(1))+'</td>';}
   html+='</tr>';}
  html+='</table>';}
 d.innerHTML=html; return d;}
function __rcaExplain(i, btn){
 var box=document.getElementById('rcabox'); if(!box)return;
 btn.disabled=true;
 box.innerHTML='<div style="color:#5b6675;font-size:12.5px;padding:8px 0">'
  +'RCA 분석 중\u2026 (규칙 엔진 + LLM 서술 — 수십 초 걸릴 수 있습니다)</div>';
 var md=(document.getElementById('rcamode')||{}).value||'auto';
 var lv=(document.getElementById('rcalevel')||{}).value||null;
 fetch('/api/rca_explain',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({i:i,mode:md,level:lv})})
 .then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});})
 .then(function(j){
  var esc2=function(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;');};
  box.innerHTML='<div style="border:1px solid #c9d6e8;border-radius:10px;background:#fbfcfe;'
   +'padding:12px 16px;margin:6px 0 12px">'
   +'<div style="font-size:11.5px;color:#8a94a3;margin-bottom:6px">생성: '+esc2(j.engine)+'</div>'
   +'<div style="white-space:pre-wrap;font-size:13px;line-height:1.65">'+esc2(j.narrative)+'</div>'
   +'<details style="margin-top:8px"><summary style="font-size:11.5px;color:#8a94a3;cursor:pointer">'
   +'증거 패킷(JSON)</summary><pre style="font-size:10.5px;overflow-x:auto;background:#f2f4f8;'
   +'padding:8px;border-radius:6px">'+esc2(JSON.stringify(j.packet,null,1))+'</pre></details>'
   +(j.debate?('<details style="margin-top:4px"><summary style="font-size:11.5px;color:#8a94a3;cursor:pointer">'
   +'비판자 토론(공격 내용)</summary><pre style="font-size:10.5px;overflow-x:auto;background:#f7f2ee;'
   +'padding:8px;border-radius:6px;white-space:pre-wrap">'+esc2(j.debate)+'</pre></details>'):'')
   +'<details style="margin-top:4px"><summary style="font-size:11.5px;color:#8a94a3;cursor:pointer">'
   +'LLM 프롬프트</summary><pre style="font-size:10.5px;overflow-x:auto;background:#f2f4f8;'
   +'padding:8px;border-radius:6px;white-space:pre-wrap">[system]\\n'+esc2(j.prompt_system||'')
   +'\\n\\n[user]\\n'+esc2(j.prompt_user||'')+'</pre></details></div>';})
 .catch(function(e){box.innerHTML='<div style="color:#c0392b;font-size:12.5px;padding:8px 0">RCA 실패: '+e.message+'</div>';})
 .then(function(){btn.disabled=false;});}
function __ovDraw(w){window.__lastW=w;
 var G=document.getElementById('ovgrid');G.innerHTML='';
 var snOK=null; var fs=document.querySelectorAll('.fsn');
 if(fs.length){snOK={}; for(var q=0;q<fs.length;q++)if(fs[q].checked)snOK[fs[q].value]=1;}
 var g=GOLD[w.recipe];
 var head=document.createElement('div');head.className='ovhead';head.style.width='100%';
 head.innerHTML='<b>'+w.wid+'</b> · 유형 '+w.recipe+' · 점수 '+(w.ratio!=null?w.ratio.toFixed(2):'-')
  +(g?'':' · <span style="color:#c0392b">이 유형의 기준 파형 없음(실제 값만 표시)</span>')
  +((g&&w.__i!=null)?' &nbsp;<select id="rcamode" style="font-size:12px;padding:3px 5px">'
    +'<option value="auto">자동</option><option value="fast">빠른 설명(Qwen 직결)</option>'
    +'<option value="deep">깊은 검증(deep-thinking)</option></select>'
    +' <select id="rcalevel" title="깊은 검증의 사고 깊이" style="font-size:12px;padding:3px 5px">'
    +'<option value="low">low</option><option value="medium" selected>medium</option>'
    +'<option value="high">high</option><option value="research">research</option>'
    +'<option value="wide">wide(광탐색·시드 없음)</option></select>'
    +' <button onclick="__rcaExplain('+w.__i+',this)" style="padding:4px 12px;'
    +'border:1px solid #2456a6;border-radius:7px;background:#fff;color:#2456a6;font-size:12px;'
    +'cursor:pointer">RCA 설명 생성</button>':'');
 G.appendChild(head);
 var rbox=document.createElement('div'); rbox.id='rcabox'; rbox.style.cssText='width:100%';
 G.appendChild(rbox);
 if(w.heat&&w.heat.sensors.length){G.appendChild(__heatEl(w.heat));}
 var names=g?g.sensors:Object.keys(w.data);
 for(var j=0;j<names.length;j++){var nm=names[j];
  if(snOK&&!snOK[nm])continue;                 /* 선택한 센서만 표시 */
  var wv=w.data[nm]||null, pv=(w.pred&&w.pred[nm])||null;
  var Lc=g?g.m[j].length:(wv?wv.length:0);
  var exx=Lc?__gextras(w.recipe,nm,Lc):[];
  if(g){G.appendChild(__chart(nm,g.m[j],g.s[j],wv,pv,exx));}
  else if(wv){G.appendChild(__chart(nm,wv,null,null,pv,exx));}}}

/* ── 골든 비교 탭 v2 — 4단계(데이터→골든→판정→결과) ── */
window.__GV={rows:null,series:null,lastOp:'',bucket:null,sortk:'ng',sortd:-1,gal:null};
function __gvJf(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||'실패');return j;});}
function __gvPost(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify(b||{})}).then(__gvJf);}
function __gvEsc(v){return String(v==null?'':v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');}
function __gvSplit(id){var v=document.getElementById(id).value.trim();
 return v?v.split(/[;,\\n]+/).map(function(t){return t.trim();}).filter(Boolean):null;}
function __gvChk(cls){var out=[];document.querySelectorAll('input.'+cls+':checked')
 .forEach(function(c){out.push(c.value);});return out;}
function __gvSenAll(on){document.querySelectorAll('input.gvcsn').forEach(function(c){c.checked=!!on;});}
function __gvFacet(el,vals,cls,checked){var h='';
 vals.forEach(function(v){h+='<label style="margin-right:8px;cursor:pointer">'
  +'<input type="checkbox" class="'+cls+'" value="'+__gvEsc(v)+'"'+(checked?' checked':'')
  +' style="vertical-align:-2px"> '+__gvEsc(v)+'</label>';});
 el.innerHTML=h||'<span style="color:#8a94a3">(없음)</span>';}
/* 단계 섹션: 열기/닫기 · 완료 배지 */
function __gvToggle(n,force){var el=document.getElementById('gv_s'+n);if(!el)return;
 var open=(force==null)?(el.style.display==='none'):!!force;el.style.display=open?'':'none';}
function __gvStep(n,txt,done){var b=document.getElementById('gv_b'+n);if(!b)return;
 b.textContent=txt;b.style.background=done?'#2456a6':'#fff';b.style.color=done?'#fff':'#2456a6';}
function __gvGoto(n){[1,2,3,4].forEach(function(k){__gvToggle(k,k===n);});}
function __gvMsg(t){document.getElementById('gvmsg').textContent=t||'';}
/* ① 데이터 */
function __gvKeys(v){return Array.isArray(v)?v:Object.keys(v||{});}
/* 안전 숫자 포맷: undefined/NaN 이 와도 화면이 죽지 않게('—') — "cannot read properties of undefined (reading toFixed)" 대책 */
function __gvF(v,d){var x=(typeof v==='number')?v:Number(v);return (v==null||!isFinite(x))?'—':x.toFixed(d);}
window.addEventListener('error',function(e){try{var m=document.getElementById('gvmsg');if(m)m.textContent='화면 오류: '+(e.message||'')+' ('+String(e.filename||'').split('/').pop()+':'+e.lineno+')';}catch(_){}});
function __gvFacetSel(){return {eqp:__gvChk('gvceq'),dsv:__gvChk('gvcdv'),recipe:__gvChk('gvcrc')};}
/* 판정 대상(③) — 골든 산정 대상(②)과 독립. 체크박스가 없으면(구 흐름) 산정 대상을 따름 */
function __gvJudgeSel(){if(!document.querySelectorAll('input.gvjeq').length)return __gvFacetSel();
 return {eqp:__gvChk('gvjeq'),dsv:__gvChk('gvjdv'),recipe:__gvChk('gvcrc')};}
/* 강도 레벨(θ 배수): 스텝별 [중 이상, 상 이상] — 화면 전용(캐시·판정 불변), 기본 0.5/1.0 */
function __gvLv(){var o={};document.querySelectorAll('#gv_lvs .gvl1').forEach(function(x){var st=x.getAttribute('data-st');
 var b=document.querySelector('#gv_lvs .gvl2[data-st="'+st+'"]');var t1=parseFloat(x.value),t2=parseFloat(b?b.value:'');
 o[st]=[isFinite(t1)?t1:0.5,isFinite(t2)?t2:1.0];});return o;}
function __gvLvW(){var e=document.getElementById('gv_lvw');var v=e?parseFloat(e.value):3;return isFinite(v)?v:3;}
function __gvLvHtml(steps,keep){keep=keep||__gvLv();var w=__gvLvW();
 var h='강도 레벨(강도비 A/θ 의 스텝별 기준 — <b>하=정상 · 중/상=불량</b>): ';
 (steps||[]).forEach(function(st){var v=keep[String(st)]||[0.5,1.0];
  h+='<span style="margin-right:8px;white-space:nowrap">step '+st+' 중≥<input class="gvl1" data-st="'+st+'" value="'+v[0]+'" onchange="__gvLvChange()" style="width:40px;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:2px 4px">'
   +' 상≥<input class="gvl2" data-st="'+st+'" value="'+v[1]+'" onchange="__gvLvChange()" style="width:40px;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:2px 4px"></span>';});
 return h+' <a href="#" onclick="__gvLvAll(0.5,1.0);return false;" style="color:#2456a6">[기본 0.5/1.0]</a>'
  +' · 강도 지수 가중 상 <input id="gv_lvw" value="'+w+'" onchange="__gvLvChange()" style="width:36px;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:2px 4px"> (하 0 · 중 1 · 상 가중 — 칸 평균이 선)';}
function __gvLvAll(a,b){document.querySelectorAll('#gv_lvs .gvl1').forEach(function(x){x.value=a;});document.querySelectorAll('#gv_lvs .gvl2').forEach(function(x){x.value=b;});__gvLvChange();}
function __gvLvChange(){__gvLive();}
/* 모양 규칙(W 봉우리·아치 기울기) 파라미터 — 캐시 밖(즉시 재판정) */
function __gvRaMode(){/* v182: 절대 기울기 단일 모드 — 저장 조건에 옛 비 모드 값(양수)이 남아 있으면 기본값으로 */
 var t1=document.getElementById('gv_ra_t1'),t2=document.getElementById('gv_ra_t2');
 if(t1&&t2&&(parseFloat(t1.value)<0||parseFloat(t2.value)<0)){t1.value='0.03';t2.value='0.01';}}
function __gvRules(){var g=function(id,d){var e=document.getElementById(id);var v=e?parseFloat(e.value):NaN;return isFinite(v)?v:d;};
 var ck=function(id){var e=document.getElementById(id);return !!(e&&e.checked);};
 var ram=(document.getElementById('gv_ra_mode')||{value:'abs'}).value,abs=(ram==='abs');
 var ste=(document.getElementById('gv_ra_st')||{value:'1,3'}).value.split(/[ ,]+/).map(function(x){return parseInt(x,10);}).filter(function(x){return isFinite(x);});
 return {w:{t1:g('gv_rw_t1',0.5),t2:g('gv_rw_t2',1.5),step:g('gv_rw_st',4),combine:ck('gv_rw_cb')},
  arch:{mode:ram,t1:g('gv_ra_t1',abs?-0.3:0.6),t2:g('gv_ra_t2',abs?-0.1:0.3),steps:(ste.length?ste:[1,3]),lr_fix:ck('gv_ra_lr'),combine:ck('gv_ra_cb')}};}
function __gvRulesMark(sens){var w=[],a=[];(sens||[]).forEach(function(sn){var t=sn.toLowerCase();
 if(t.indexOf('t_')===0&&t.indexOf('outlet')>=0)w.push(sn);else if(t.indexOf('dp_')===0&&(t.indexOf('vslf-vslb')>=0||/front.*back/.test(t)))a.push(sn);});
 var ew=document.getElementById('gv_rw_on'),ea=document.getElementById('gv_ra_on');
 if(ew)ew.textContent=w.length?'[적용: '+w.join(', ')+']':'[해당 센서 없음]';
 if(ea)ea.textContent=a.length?'[적용: '+a.join(', ')+']':'[해당 센서 없음]';}
function __gvRuleTag(r){var out=[];Object.keys(r.sensors||{}).forEach(function(sn){(r.sensors[sn]||[]).forEach(function(b){
 if(b.rule==='wprom')out.push('W '+__gvF(Number(b.val),2)+'℃'+(b.flag?'↑':''));else if(b.rule==='arch')out.push('아치 기울기 '+(b.mode==='ratio'?__gvF(Number(b.val),2)+'×':__gvF(Number(b.val),4))+'(s'+b.step+')');});});
 return out.length?' · '+out.slice(0,2).join(' '):'';}
/* 웨이퍼 강도: 스텝별 ir(A/θ 최대) → 레벨 0 하/1 중/2 상, 웨이퍼 레벨=최대 */
function __gvLevel(r,lv){lv=lv||__gvLv();var out={lv:0,maxr:0,steps:{},hi:[]};var ir=r.ir||{};
 Object.keys(ir).forEach(function(st){var v=ir[st],t=lv[String(st)]||[0.5,1.0];var l=v>=t[1]?2:(v>=t[0]?1:0);
  out.steps[st]=l;if(l>out.lv)out.lv=l;if(v>out.maxr)out.maxr=v;if(l===2)out.hi.push(st);});
 if(r.lv!=null)out.lv=r.lv;                     /* 서버 판정 레벨 우선(판정과 표시 일치) */
 return out;}
function __gvLvName(l){return l===2?'상':(l===1?'중':'하');}
function __gvAfterList(j){
 __gvStep(1,'웨이퍼 '+j.total+'장'+(j.csv_n?' · CSV '+j.csv_n+'건':''),true);
 var dg=j.diag||{},dgs='';
 if(dg.n_seen!=null)dgs='파일 '+dg.n_seen+'장 중 '+j.total+'장 사용'
  +(dg.n_untagged?' · 날짜 태그 없음 '+dg.n_untagged+'장 제외(웨이퍼 분석 탭에서 폴더를 한 번 불러오면 태깅됩니다)':'')
  +(dg.n_collision?' · 같은 파일명 '+dg.n_collision+'건은 폴더명으로 구분':'')
  +((dg.folders||[]).length>1?' · 폴더별 '+dg.folders.map(function(f){return f.folder.split(/[\\/]/).pop()+' '+f.n;}).join(', '):'');
 window.__GV.diag=dgs;
 document.getElementById('gv_flt').style.display='';
 var av=j.avail||{};
 __gvFacet(document.getElementById('gv_feq'),__gvKeys(av.eqp),'gvceq',true);
 __gvFacet(document.getElementById('gv_fdv'),__gvKeys(av.dsv),'gvcdv',true);
 __gvFacet(document.getElementById('gv_frc'),__gvKeys(av.recipe),'gvcrc',true);
 __gvFacet(document.getElementById('gv_fsn'),j.sensors||[],'gvcsn',true);
 __gvFacet(document.getElementById('gv_jeq'),__gvKeys(av.eqp),'gvjeq',true);
 __gvFacet(document.getElementById('gv_jdv'),__gvKeys(av.dsv),'gvjdv',true);
 document.getElementById('gv_jtg').style.display='';
 var gpd=document.getElementById('gv_gpd'),gh='';
 __gvKeys(av.dsv).slice().forEach(function(dv){
  gh+='<span style="margin-right:14px;white-space:nowrap">'+__gvEsc(dv)
   +' 골든 기간 <input class="gvgp0" data-dsv="'+__gvEsc(dv)+'" placeholder="비우면 공통" '
   +'style="width:104px;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:2px 6px">'
   +'~<input class="gvgp1" data-dsv="'+__gvEsc(dv)+'" placeholder="끝" '
   +'style="width:104px;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:2px 6px"></span>';});
 gpd.innerHTML=gh;
 if(j.dates&&j.dates.length){document.getElementById('gv_d0').value=j.dates[0];
  document.getElementById('gv_d1').value=j.dates[j.dates.length-1];}
 __gvMsg('목록 준비 — ② 골든 기간을 정하고 산정하세요'+(dgs?' · '+dgs:''));__gvGoto(2);}
function __gvList(){
 window.__GV.userActed=true;__gvMsg('목록 불러오는 중…');
 __gvPost('/api/gv_list',{folder:document.getElementById('gv_fol').value.trim(),
  eqp:__gvSplit('gv_eq'),dsv:__gvSplit('gv_dv'),recipe:__gvSplit('gv_rc'),
  months:__gvSplit('gv_mo'),csv:document.getElementById('gv_csv').value.trim(),
  store:document.getElementById('gv_st').value.trim()})
 .then(__gvAfterList).catch(function(e){__gvMsg(e.message);});}
function __gvFromViz(){
 __gvMsg('데이터 시각화 탭 목록 가져오는 중…');
 __gvPost('/api/gv_list',{from_viz:true,csv:document.getElementById('gv_csv').value.trim()})
 .then(function(j){if(j.folder)document.getElementById('gv_fol').value=j.folder;__gvAfterList(j);})
 .catch(function(e){__gvMsg(e.message);});}
/* ② 골든 */
function __gvPerDsv(){var pd={};
 document.querySelectorAll('#gv_gpd .gvgp0').forEach(function(a){
  var dv=a.getAttribute('data-dsv');
  var b=document.querySelector('#gv_gpd .gvgp1[data-dsv="'+dv.replace(/"/g,'\\"')+'"]');
  if(a.value.trim()||(b&&b.value.trim()))pd[dv]={d0:a.value.trim(),d1:b?b.value.trim():''};});
 return pd;}
function __gvBuild(restore){
 window.__GV.userActed=true;window.__GV.lastOp='build';__gvMsg('골든 산정 중…');
 __gvPost('/api/gv_build',{d0:document.getElementById('gv_d0').value.trim(),
  d1:document.getElementById('gv_d1').value.trim(),sensors:__gvChk('gvcsn'),
  per_dsv:__gvPerDsv(),restore_ids:restore||[],facet:__gvFacetSel()})
 .then(function(){__gvPoll();}).catch(function(e){__gvMsg(e.message);});}
function __gvGallery(){
 var host=document.getElementById('gv_gal');host.innerHTML='';
 __gvPost('/api/gv_golden',{}).then(function(j){window.__GV.gal=j;
  var h='';
  if(j.auto_excluded&&j.auto_excluded.length){
   h+='<div style="font-size:12px;margin:4px 0;padding:6px 8px;border:1px solid #f1d9a8;'
    +'border-radius:8px;background:#fff9ec">자체 정화로 평균에서 제외한 웨이퍼 '+j.auto_excluded.length
    +'장(1차 골든 대비 이탈 상위) — 정상이라고 확신하면 체크 후 <button onclick="__gvRestore()" '
    +'style="padding:2px 8px;border:1px solid #b26a00;border-radius:6px;background:#fff;color:#b26a00;'
    +'font-size:11.5px;cursor:pointer">복원해서 재산정</button><br>';
   j.auto_excluded.forEach(function(id){h+='<label style="margin-right:8px;white-space:nowrap">'
    +'<input type="checkbox" class="gvrs" value="'+__gvEsc(id)+'" style="vertical-align:-2px"> '
    +__gvEsc(id)+'</label>';});
   h+='</div>';}
  host.innerHTML=h;
  var dsvCol=['#2e9e6b','#c0392b','#8e44ad','#d68910','#16a085','#e84393','#6c5ce7','#00897b','#8d6e63','#546e7a'];
  (j.groups||[]).forEach(function(g,gi){
   var box=document.createElement('div');
   var lg='<b>'+__gvEsc(g.eqp)+(g.recipe&&g.recipe!=='unknown'?' / '+__gvEsc(g.recipe):'')+'</b> — 설비 대표 '
    +g.rep_n+'장(파랑+밴드) · 표시: '+g.dsvs.map(function(d,i){var c=dsvCol[i%dsvCol.length];
     return '<label style="margin-right:6px;cursor:pointer;white-space:nowrap"><input type="checkbox" class="gvgl" data-g="'+gi+'" data-i="'+i
     +'" checked style="vertical-align:-2px"> <span style="display:inline-block;width:10px;height:10px;background:'+c+';vertical-align:-1px;border-radius:2px"></span> '
     +__gvEsc(d.dsv)+' '+d.n+'장'+(d.small?'[소표본]':'')+'</label>';}).join('')
    +((g.dsvs||[]).length===1?'<br><span style="color:#b26a00">⚠ 이 그룹에 DSV가 하나뿐 — 설비 통합 골든 = 이 챔버 골든. 다른 DSV가 다른 레시피 블록으로 갈라졌다면 레시피 태그를, 아니면 산정 대상 체크를 확인하세요</span>':'')
    +((g.missing||[]).length?'<br><span style="color:#b26a00">골든 없음: '+g.missing.map(function(m){return __gvEsc(m.dsv)+'('+__gvEsc(m.reason)+')';}).join(' · ')
     +' — 골든 기간·DSV별 기간을 넓히거나 산정 대상을 확인하세요</span>':'');
   box.innerHTML='<div style="font-size:12.5px;margin:6px 0 2px">'+lg+'</div>';
   var G=document.createElement('div');G.className='grid';box.appendChild(G);
   var tls=(g.canon||[]).map(function(r){return {i:r[1],label:'step '+r[0]};});
   function draw(){G.innerHTML='';var on=[];box.querySelectorAll('.gvgl').forEach(function(c){if(c.checked)on.push(parseInt(c.getAttribute('data-i'),10));});
    (g.sensors||[]).forEach(function(sn){
     var ex=g.dsvs.map(function(d,i){return {label:d.dsv,vals:d.mean[sn],color:dsvCol[i%dsvCol.length],dash:'none',width:1.4,_i:i};})
      .filter(function(e){return on.indexOf(e._i)>=0;});
     G.appendChild(__chart(sn,g.rep_mean[sn],g.rep_std[sn],null,null,ex,{toplabels:tls,baseLabel:'설비 대표',W:520,H:200}));});}
   draw();box.querySelectorAll('.gvgl').forEach(function(c){c.addEventListener('change',draw);});
   host.appendChild(box);});
 }).catch(function(e){host.innerHTML='<span style="color:#d64545;font-size:12px">'+__gvEsc(e.message)+'</span>';});}
function __gvRestore(){var ids=[];document.querySelectorAll('#gv_gal .gvrs:checked').forEach(function(c){ids.push(c.value);});
 if(!ids.length){__gvMsg('복원할 웨이퍼를 체크하세요');return;}__gvBuild(ids);}
/* ③ 판정 */
function __gvPreset(){var v=document.getElementById('gv_pre').value;
 var m={c:[3,0.1],s:[2,0.05],m:[1.5,0.02]}[v];if(!m)return;
 document.getElementById('gv_k').value=m[0];document.getElementById('gv_a').value=m[1];__gvLive();}
function __gvLive(){if(window.__GV.rows&&!window.__GV.running){__gvResults();__gvPost('/api/gv_state',{live:__gvCond()}).catch(function(){});}}
function __gvVlmNotes(){var o={common:(document.getElementById('gv_vn')||{value:''}).value.trim(),steps:{}};
 document.querySelectorAll('#gv_vns .gvvn').forEach(function(x){if(x.value.trim())o.steps[x.getAttribute('data-st')]=x.value.trim();});return o;}
function __gvSysPrompt(){var pre=document.getElementById('gv_sysp');if(pre.style.display!=='none'){pre.style.display='none';return;}
 fetch('/api/gv_vlm_prompt').then(__gvJf).then(function(j){pre.textContent=j.system;pre.style.display='';}).catch(function(e){__gvMsg(e.message);});}
function __gvAstp(){var o={};document.querySelectorAll('#gv_astp .gvas').forEach(function(x){
 if(x.value.trim())o[x.getAttribute('data-st')]=parseFloat(x.value);});return o;}
/* 판정 센서·민감도 표: [점수 반영][센서 α][센서×스텝 α(펼침)] — 전부 캐시 밖(즉시 재판정) */
function __gvJtState(){var st={sel:[],all:[],asn:{},acell:{}};
 document.querySelectorAll('#gv_jt .gvjs').forEach(function(c){st.all.push(c.value);if(c.checked)st.sel.push(c.value);});
 document.querySelectorAll('#gv_jt .gvja').forEach(function(x){if(x.value.trim())st.asn[x.getAttribute('data-sn')]=parseFloat(x.value);});
 document.querySelectorAll('#gv_jt .gvjc').forEach(function(x){if(x.value.trim())st.acell[x.getAttribute('data-sn')+'|'+x.getAttribute('data-st')]=parseFloat(x.value);});
 return st;}
function __gvJtRender(sensors,steps,keep){
 var host=document.getElementById('gv_jt');if(!host)return;
 keep=keep||__gvJtState();var hadRows=document.querySelectorAll('#gv_jt .gvjs').length>0;
 /* known=이 상태가 알던 센서 목록 — 모르는(새) 센서는 체크, 알던 센서는 저장된 체크 그대로(재실행·복원 공통) */
 var known=(keep.all&&keep.all.length)?keep.all:(hadRows?__gvJtState().all:null);
 if(!sensors||!sensors.length){host.style.display='none';host.innerHTML='';return;}
 var h='<b>반영 센서·민감도</b> <span style="color:#8a94a3">체크 해제한 센서는 불량 계산에서 빠지고 막대그래프가 바로 다시 그려집니다 · 센서 α 비우면 공통 α · [스텝별]을 열면 센서×스텝 α(우선순위: 센서×스텝 &gt; 센서 &gt; 스텝 &gt; 공통)</span>'
  +' <a href="#" onclick="__gvJtPre(\\'t_\\',0.15);return false;" style="color:#2456a6">[T계열 둔감 0.15]</a>'
  +' <a href="#" onclick="__gvJtPre(\\'p_\\',0.02);return false;" style="color:#2456a6">[P계열 민감 0.02]</a>'
  +' <a href="#" onclick="__gvJtPre(\\'\\',null);return false;" style="color:#2456a6">[센서 α 전부 비움]</a>'
  +'<table style="border-collapse:collapse;margin-top:3px"><tr><th style="border:1px solid #dde3ea;padding:2px 6px">반영</th><th style="border:1px solid #dde3ea;padding:2px 6px">센서</th><th style="border:1px solid #dde3ea;padding:2px 6px">센서 α</th><th style="border:1px solid #dde3ea;padding:2px 6px">센서×스텝 α</th></tr>';
 sensors.forEach(function(sn){var on=!known?true:(keep.sel.indexOf(sn)>=0||known.indexOf(sn)<0);
  h+='<tr><td style="border:1px solid #eef1f5;padding:2px 6px;text-align:center"><input type="checkbox" class="gvjs" value="'+__gvEsc(sn)+'"'+(on?' checked':'')+' onchange="__gvLive()"></td>'
   +'<td style="border:1px solid #eef1f5;padding:2px 6px">'+__gvEsc(sn)+'</td>'
   +'<td style="border:1px solid #eef1f5;padding:2px 6px"><input class="gvja" data-sn="'+__gvEsc(sn)+'" value="'+(keep.asn[sn]!=null?keep.asn[sn]:'')+'" placeholder="공통" onchange="__gvLive()" style="width:52px;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:2px 5px"></td>'
   +'<td style="border:1px solid #eef1f5;padding:2px 6px"><a href="#" onclick="var e=this.nextElementSibling;e.style.display=(e.style.display===\\'none\\'?\\'\\':\\'none\\');return false;" style="color:#2456a6">[스텝별]</a><span style="display:none">';
  (steps||[]).forEach(function(st){var v=keep.acell[sn+'|'+st];
   h+=' s'+st+'<input class="gvjc" data-sn="'+__gvEsc(sn)+'" data-st="'+st+'" value="'+(v!=null?v:'')+'" onchange="__gvLive()" style="width:44px;font-size:11px;border:1px solid #dde3ea;border-radius:5px;padding:1px 4px">';});
  h+='</span></td></tr>';});
 host.innerHTML=h+'</table>';host.style.display='';}
function __gvJtPre(prefix,val){document.querySelectorAll('#gv_jt .gvja').forEach(function(x){
 if(!prefix||x.getAttribute('data-sn').indexOf(prefix)===0)x.value=(val==null?'':val);});__gvLive();}
function __gvParams(){var jt=__gvJtState();return {k:parseFloat(document.getElementById('gv_k').value)||2.0,
 alpha:parseFloat(document.getElementById('gv_a').value)||0.05,
 beta:parseFloat(document.getElementById('gv_b').value)||0.3,
 agg:document.getElementById('gv_ag').value,
 mode:document.getElementById('gv_md').value,
 alpha_steps:__gvAstp(),alpha_sensor:jt.asn,alpha_cell:jt.acell,levels:__gvLv(),rules:__gvRules(),
 sensor_sel:(document.querySelectorAll('#gv_jt .gvjs').length?jt.sel:null)};}
function __gvRun(){
 window.__GV.userActed=true;window.__GV.lastOp='run';__gvMsg('판정 진행 준비 중…');
 var p=__gvParams();
 __gvPost('/api/gv_run',{d0:document.getElementById('gv_e0').value.trim(),
  d1:document.getElementById('gv_e1').value.trim(),sensors:__gvChk('gvcsn'),
  w:parseInt(document.getElementById('gv_w').value)||0,
  k:p.k,alpha:p.alpha,alpha_steps:p.alpha_steps,alpha_sensor:p.alpha_sensor,alpha_cell:p.alpha_cell,
  sensor_sel:p.sensor_sel,mode:p.mode,facet:__gvJudgeSel(),vlm_notes:__gvVlmNotes(),
  gref:document.getElementById('gv_gref').value})
 .then(function(){__gvSaveCond(true);__gvPoll();}).catch(function(e){__gvMsg(e.message);});}
function __gvCancel(){__gvPost('/api/gv_cancel',{});}
function __gvPoll(){
 fetch('/api/gv_status').then(__gvJf).then(function(j){
  window.__GV.running=(j.state==='running');
  if(j.status_warn)__gvMsg('상태 조회 경고: '+j.status_warn);
  if(j.state==='running'){__gvMsg(j.progress||'진행 중…');setTimeout(__gvPoll,1500);return;}
  if(j.state==='error'){__gvMsg('오류: '+(j.error||''));return;}
  if(j.steps&&j.steps.length){var ah=document.getElementById('gv_astp');
   var keep={};ah.querySelectorAll('.gvas').forEach(function(x){keep[x.getAttribute('data-st')]=x.value;});
   var h2='스텝별 면적 임계 α(비우면 공통): ';
   j.steps.forEach(function(st){h2+='<span style="margin-right:8px;white-space:nowrap">step '+st
    +' <input class="gvas" data-st="'+st+'" value="'+(keep[String(st)]||'')+'" onchange="__gvLive()"'
    +' style="width:48px;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:2px 5px"></span>';});
   ah.innerHTML=h2;document.getElementById('gv_lvs').innerHTML=__gvLvHtml(j.steps);
   if(j.sensors&&j.sensors.length)window.__GV.sens=j.sensors;
   __gvRulesMark(window.__GV.sens||__gvChk('gvcsn'));
   __gvJtRender(window.__GV.sens||__gvChk('gvcsn'),j.steps);
   var vh=document.getElementById('gv_vns');if(vh){var keepv={};vh.querySelectorAll('.gvvn').forEach(function(x){keepv[x.getAttribute('data-st')]=x.value;});
    var h3='';j.steps.forEach(function(st){h3+='<div style="margin:2px 0">step '+st+' <textarea class="gvvn" data-st="'+st+'" rows="1" '
     +'placeholder="예) 곡선 하향하며 끝값이 시작값보다 낮아야 함 — 직선화되거나 낙폭이 작으면 불량" '
     +'style="width:88%;vertical-align:middle;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:2px 6px">'+__gvEsc(keepv[String(st)]||'')+'</textarea></div>';});
    vh.innerHTML=h3;}}
  if(j.build){var g=j.build,ss=[];
   (g.goldens||[]).forEach(function(x){ss.push((x.dsv||'설비 대표')+' '+x.n+'장'+(x.small?' [소표본 골든(N<10) — 참고용]':''));});
   document.getElementById('gv_gsum').textContent='골든 '+(g.goldens||[]).length+'개(보고분 제외 '+g.excluded_n+'장): '+ss.join(' · ');
   __gvStep(2,'골든 '+(g.goldens||[]).length+'개',true);}
  if(window.__GV.lastOp==='run'){__gvMsg('판정 완료');__gvResults();}
  else if(window.__GV.lastOp==='build'){__gvMsg('골든 산정 완료 — 아래 골든 카드를 확인하고 ③ 판정으로');__gvGallery();__gvToggle(3,true);}
  else{__gvMsg('');}
 }).catch(function(){setTimeout(__gvPoll,3000);});}
/* ④ 결과 */
function __gvRfSel(){var o={eqp:[],dsv:[]};
 document.querySelectorAll('input.gvreq:checked').forEach(function(c){o.eqp.push(c.value);});
 document.querySelectorAll('input.gvrdv:checked').forEach(function(c){o.dsv.push(c.value);});return o;}
function __gvRfRows(){var rows=window.__GV.rows||[],f=__gvRfSel();
 var ne=document.querySelectorAll('input.gvreq').length,nd=document.querySelectorAll('input.gvrdv').length;
 return rows.filter(function(r){var g=(r.gkey||'').split('|');
  if(ne&&f.eqp.indexOf(r.eqp||g[0]||'(없음)')<0)return false;if(nd&&f.dsv.indexOf(r.dsv||g[2]||'(없음)')<0)return false;return true;});}
function __gvRfRender(){
 var rows=window.__GV.rows||[],eqs={},dvs={};
 rows.forEach(function(r){var g=(r.gkey||'').split('|');eqs[r.eqp||g[0]||'(없음)']=1;dvs[r.dsv||g[2]||'(없음)']=1;});
 var keep={};document.querySelectorAll('input.gvreq,input.gvrdv').forEach(function(c){keep[c.className+'|'+c.value]=c.checked;});
 function mk(el,keys,cls){var h='';keys.sort().forEach(function(k){var kk=cls+'|'+k;
  h+='<label style="margin-right:8px;cursor:pointer"><input type="checkbox" class="'+cls+'" value="'+__gvEsc(k)+'"'+((kk in keep)?(keep[kk]?' checked':''):' checked')
   +' onchange="__gvRender()" style="vertical-align:-2px"> '+__gvEsc(k)+'</label>';});el.innerHTML=h;}
 mk(document.getElementById('gv_rfeq'),Object.keys(eqs),'gvreq');mk(document.getElementById('gv_rfdv'),Object.keys(dvs),'gvrdv');
 document.getElementById('gv_rflt').style.display=rows.length?'':'none';}
function __gvRender(){
 var rows=__gvRfRows(),ag=document.getElementById('gv_ag').value,bk={};
 rows.forEach(function(r){var b=(ag==='month')?r.date.slice(0,7):(ag==='week'?__gvWk(r.date):r.date);
  var c=bk[b]||(bk[b]=[0,0]);c[r.ng?1:0]+=1;});
 var series=Object.keys(bk).sort().map(function(b){return {b:b,ok:bk[b][0],ng:bk[b][1]};});
 window.__GV.series=series;window.__GV.frows=rows;
 var nNg=rows.filter(function(r){return r.ng;}).length,all=(window.__GV.rows||[]).length;
 __gvMsg('웨이퍼 '+rows.length+'장 중 불량 '+nNg+'장'+(rows.length!==all?' (전체 '+all+'장 중 필터)':''));
 document.getElementById('gv_live').textContent='현재 설정: 불량 '+nNg+'/'+rows.length+'장';
 __gvStep(4,'불량 '+nNg+'/'+rows.length+'장',true);
 var lk={},lvt=__gvLv(),lw=__gvLvW();rows.forEach(function(r){var b=(ag==='month')?r.date.slice(0,7):(ag==='week'?__gvWk(r.date):r.date);
  var L=__gvLevel(r,lvt);var c=lk[b]||(lk[b]={n:[0,0,0],sc:0});c.n[L.lv]+=1;c.sc+=(L.lv===2?lw:(L.lv===1?1:0));});
 /* 선=강도 지수: 하 0 · 중 1 · 상 가중(기본 3) 의 칸 평균 — 상이 많을수록 급격히 오름 */
 var lvseries=Object.keys(lk).sort().map(function(b){var t=lk[b].n[0]+lk[b].n[1]+lk[b].n[2];return {b:b,n:lk[b].n,mean:t?lk[b].sc/t:0};});
 window.__GV.lvseries=lvseries;__gvBarsLv(document.getElementById('gv_lvbars'),lvseries);
 var sr={},rk={};rows.forEach(function(r){if(!r.ng)return;Object.keys(r.sensors||{}).forEach(function(s){sr[s]=(sr[s]||0)+1;
  r.sensors[s].forEach(function(b){var k=s+' step'+b.step;if(!(k in rk)||b.A>rk[k])rk[k]=b.A;});});});
 var h='';var srk=Object.keys(sr).sort(function(a,b){return sr[b]-sr[a];});
 if(srk.length){h+='<b>센서별 불량</b> ';srk.forEach(function(s){h+='<span style="margin-right:10px">'+__gvEsc(s)+' <span style="color:#d64545">'+sr[s]+'장</span></span>';});h+='<br>';}
 var rkk=Object.keys(rk).sort(function(a,b){return rk[b]-rk[a];}).slice(0,20);
 if(rkk.length){h+='<b>초과면적 상위</b> ';rkk.forEach(function(k){h+='<span style="margin-right:10px">'+__gvEsc(k)+' <span style="color:#d64545">'+__gvF(rk[k],4)+'</span></span>';});}
 document.getElementById('gv_rank').innerHTML=h;
 __gvWlist(window.__GV.bucket);}
function __gvResults(){
 var t0=Date.now();
 __gvPost('/api/gv_results',__gvParams()).then(function(j){
  var t1=Date.now();window.__GV.rows=j.wafers;
  __gvStep(3,'완료 · '+(document.getElementById('gv_md').value==='rule'?'룰베이스':document.getElementById('gv_md').value==='vlm'?'VLM':'하이브리드'),true);
  __gvToggle(4,true);
  __gvRfRender();__gvRender();__gvWaveInfo();
  var t2=Date.now();window.__GV.lastTiming='판정 '+((j.t_verdict||0)).toFixed(2)+'초 · 전송 '+((t1-t0)/1000).toFixed(2)+'초 · 그리기 '+((t2-t1)/1000).toFixed(2)+'초';
  var tm=document.getElementById('gv_tm');if(tm)tm.textContent='('+window.__GV.lastTiming+')';
  var sm=j.sensor_missing||{};var smk=Object.keys(sm);
  if(smk.length){var el=document.getElementById('gv_rank');el.innerHTML+='<div style="color:#b26a00">센서 결측(해당 웨이퍼는 그 센서만 판정불가): '
   +smk.map(function(k){return __gvEsc(k)+' '+sm[k]+'장';}).join(' · ')+' — 가상센서는 두 원 센서가 모두 있어야 계산됩니다</div>';}
 }).catch(function(e){__gvMsg(e.message);});}
function __gvWk(d){var t=new Date(d+'T00:00:00');
 t.setDate(t.getDate()-((t.getDay()+6)%7));return t.toISOString().slice(0,10);}
function __gvSort(k){var g=window.__GV;if(g.sortk===k)g.sortd=-g.sortd;else{g.sortk=k;g.sortd=(k==='id'||k==='date')?1:-1;}
 __gvWlist(g.bucket);}
function __gvWlist(bucket){
 var g=window.__GV,rows=g.frows||g.rows||[],el=document.getElementById('gv_list');g.bucket=bucket;
 var ag=document.getElementById('gv_ag').value,q=(document.getElementById('gv_q').value||'').toLowerCase();
 var ngOnly=document.getElementById('gv_ngonly').checked;
 var sel=rows.filter(function(r){
  if(bucket!=null){var b=(ag==='month')?r.date.slice(0,7):(ag==='week'?__gvWk(r.date):r.date);if(b!==bucket)return false;}
  if(ngOnly&&!r.ng)return false;
  if(q){var t=(r.id+' '+r.gkey+' '+Object.keys(r.sensors||{}).join(' ')).toLowerCase();if(t.indexOf(q)<0)return false;}
  return true;});
 var maxA=function(r){var m=0;Object.keys(r.sensors||{}).forEach(function(s){r.sensors[s].forEach(function(b){if(b.A>m)m=b.A;});});return m;};
 var lvt=__gvLv();
 var kf={ng:function(r){return (r.ng?1:0)*10+(r.len_ng&&r.len_ng.length?1:0);},date:function(r){return r.date+r.id;},
  id:function(r){return r.id;},A:maxA,dsv:function(r){return r.gkey;},lv:function(r){var L=__gvLevel(r,lvt);return L.lv*100+L.maxr;}}[g.sortk]||function(r){return 0;};
 sel.sort(function(a,b){var x=kf(a),y=kf(b);return (x<y?-1:x>y?1:0)*g.sortd;});
 document.getElementById('gv_lsum').textContent=(bucket?bucket+' · ':'')+sel.length+'장 (불량 '
  +sel.filter(function(r){return r.ng;}).length+')'+(bucket?' — 막대 더블클릭=전체':'');
 var th=function(k,t){return '<th style="position:sticky;top:0;background:#f6f8fb;border:1px solid #dde3ea;'
  +'padding:3px 8px;text-align:left;cursor:pointer;white-space:nowrap" onclick="__gvSort(\\''+k+'\\')">'+t
  +(g.sortk===k?(g.sortd>0?' ▲':' ▼'):'')+'</th>';};
 var h='<table style="border-collapse:collapse;width:100%"><tr>'+th('ng','판정')+th('date','날짜')+th('id','웨이퍼')
  +th('dsv','설비|레시피|DSV')+'<th style="position:sticky;top:0;background:#f6f8fb;border:1px solid #dde3ea;padding:3px 8px;text-align:left">불량 센서(스텝)</th>'
  +th('A','최대 A')+th('lv','강도')+'<th style="position:sticky;top:0;background:#f6f8fb;border:1px solid #dde3ea;padding:3px 8px;text-align:left">길이</th>'
  +'<th style="position:sticky;top:0;background:#f6f8fb;border:1px solid #dde3ea;padding:3px 8px;text-align:left">VLM 근거</th></tr>';
 var td='border:1px solid #eef1f5;padding:2px 8px;vertical-align:top';
 sel.slice(0,400).forEach(function(r){
  var ss=Object.keys(r.sensors||{}).map(function(s){return __gvEsc(s)+'('+r.sensors[s].map(function(b){return 'step'+b.step;}).join('/')+')';}).join(' ');
  var ln=(r.len_ng||[]).map(function(l){return 'step'+l.step+' ×'+l.ratio;}).join(' ');
  h+='<tr class="gvw" data-id="'+__gvEsc(r.id)+'" style="cursor:pointer'+(r.ng?';background:#fff5f5':'')+'">'
   +'<td style="'+td+';color:'+(r.ng?'#d64545':'#5b6675')+';font-weight:600">'+(r.ng?'불량':(r.na?'일부 판정불가':'정상'))+'</td>'
   +'<td style="'+td+'">'+r.date+'</td><td style="'+td+'">'+__gvEsc(r.id)+'</td>'
   +'<td style="'+td+'">'+__gvEsc(r.gkey)+'</td><td style="'+td+'">'+ss+'</td>'
   +'<td style="'+td+'">'+(maxA(r)?__gvF(maxA(r),4):'')+'</td>'
   +(function(){var L=__gvLevel(r,lvt);return '<td style="'+td+';color:'+(L.lv===2?'#d64545':(L.lv===1?'#b26a00':'#5b6675'))+'">'+__gvLvName(L.lv)+(__gvRuleTag(r)?__gvEsc(__gvRuleTag(r)):' '+__gvF(L.maxr,2)+(L.hi.length?' (step '+L.hi.join('/')+')':''))+'</td>';})()
   +'<td style="'+td+'">'+ln+'</td>'
   +'<td style="'+td+';max-width:260px">'+__gvEsc(r.vlm&&r.vlm.reason?r.vlm.reason:'')+'</td></tr>';});
 h+='</table>'+(sel.length>400?'<div style="color:#8a94a3">… 외 '+(sel.length-400)+'장 — 검색/불량만으로 좁히세요</div>':'');
 el.innerHTML=h;
 el.querySelectorAll('.gvw').forEach(function(x){x.addEventListener('click',function(){__gvCards(x.getAttribute('data-id'));});});}
function __gvBars(el,series,z0,z1){
 if(!series||!series.length){el.innerHTML='';return;}
 var full=series;
 if(z0!=null){series=series.slice(z0,z1+1);if(!series.length)series=full;}
 var W=Math.max(900,Math.min(1400,series.length*20+90)),H=320,pl=48,pr=54,pt=34,pb=38;
 var iw=W-pl-pr,ih=H-pt-pb,mx=1;
 series.forEach(function(s){if(s.ok+s.ng>mx)mx=s.ok+s.ng;});
 mx=Math.ceil(mx*1.25);                        /* 최고 막대 위 숫자 두 줄 자리(헤드룸) */
 /* 강도 표기: 칸별 강도비 중앙값(A/θ) — 오른쪽 축 선 + 막대 위 파란 숫자 */
 var lvm={};(window.__GV.lvseries||[]).forEach(function(q){lvm[q.b]=q.mean;});
 var mr=0.1;series.forEach(function(s){if(lvm[s.b]!=null&&lvm[s.b]>mr)mr=lvm[s.b];});mr=Math.max(1.2,mr*1.15);
 var hasLv=series.some(function(s){return lvm[s.b]!=null;}),pts=[];
 var bw=Math.max(2,Math.min(40,iw/series.length-3)),svg='';
 for(var g=0;g<=4;g++){var v=Math.round(mx*g/4),gy=pt+ih*(1-g/4);
  svg+='<line x1="'+pl+'" y1="'+gy+'" x2="'+(W-pr)+'" y2="'+gy+'" stroke="#e5e9f0"/>'
   +'<text x="'+(pl-5)+'" y="'+(gy+4)+'" text-anchor="end" font-size="10" fill="#8a94a3">'+v+'</text>'
   +(hasLv?'<text x="'+(W-pr+5)+'" y="'+(gy+4)+'" font-size="10" fill="#2456a6">'+__gvF((mr*g/4),2)+'</text>':'');}
 var lb=Math.max(1,Math.floor(series.length/12));
 series.forEach(function(s,i){
  var x=pl+iw*(i+0.5)/series.length-bw/2;
  var hOk=ih*s.ok/mx,hNg=ih*s.ng/mx;
  if(lvm[s.b]!=null)pts.push([x+bw/2,pt+ih*(1-lvm[s.b]/mr),lvm[s.b]]);
  svg+='<rect class="gvb" data-b="'+s.b+'" x="'+__gvF(x,1)+'" y="'+__gvF((pt+ih-hOk),1)
   +'" width="'+__gvF(bw,1)+'" height="'+__gvF(hOk,1)+'" fill="#6d9dc5" style="cursor:pointer"><title>'+s.b+' 정상 '+s.ok+' / 불량 '+s.ng+'</title></rect>'
   +'<rect class="gvb" data-b="'+s.b+'" x="'+__gvF(x,1)+'" y="'+__gvF((pt+ih-hOk-hNg),1)
   +'" width="'+__gvF(bw,1)+'" height="'+__gvF(hNg,1)+'" fill="#d64545" style="cursor:pointer"><title>'+s.b+' 정상 '+s.ok+' / 불량 '+s.ng+'</title></rect>';
  if(i%lb===0)svg+='<text x="'+__gvF((x+bw/2),1)+'" y="'+(H-18)+'" text-anchor="middle" font-size="9" fill="#8a94a3">'+s.b.slice(5)+'</text>';
  /* 막대별 정상/불량 숫자 — 폭이 좁으면(일별 장기) 불량 수만, 더 좁으면 생략(툴팁) */
  var yTop=pt+ih-hOk-hNg-3;
  if(bw>=16)svg+='<text x="'+__gvF((x+bw/2),1)+'" y="'+__gvF(yTop,1)+'" text-anchor="middle" font-size="9" fill="#2b3440">'+s.ok+'<tspan fill="#d64545">'+(s.ng?' / '+s.ng:'')+'</tspan></text>'
   +(lvm[s.b]!=null?'<text x="'+__gvF((x+bw/2),1)+'" y="'+__gvF((yTop-10),1)+'" text-anchor="middle" font-size="8.5" fill="#2456a6">'+__gvF(lvm[s.b],2)+'</text>':'');
  else if(bw>=7&&s.ng)svg+='<text x="'+__gvF((x+bw/2),1)+'" y="'+__gvF(yTop,1)+'" text-anchor="middle" font-size="8" fill="#d64545">'+s.ng+'</text>';});
 if(pts.length>1)svg+='<polyline fill="none" stroke="#2456a6" stroke-width="1.4" stroke-opacity="0.85" points="'+pts.map(function(q){return __gvF(q[0],1)+','+__gvF(q[1],1);}).join(' ')+'"/>';
 pts.forEach(function(q){svg+='<circle cx="'+__gvF(q[0],1)+'" cy="'+__gvF(q[1],1)+'" r="2" fill="#2456a6"><title>강도비 중앙값 '+__gvF(q[2],2)+'</title></circle>';});
 /* 레전드는 상단 여백 띠(막대와 겹치지 않음) */
 svg+='<rect x="'+(W-150)+'" y="6" width="9" height="9" fill="#6d9dc5"/><text x="'+(W-138)+'" y="15" font-size="10" fill="#5b6675">정상</text>'
  +'<rect x="'+(W-100)+'" y="6" width="9" height="9" fill="#d64545"/><text x="'+(W-88)+'" y="15" font-size="10" fill="#5b6675">불량</text>'
  +'<text x="'+pl+'" y="15" font-size="10" fill="#5b6675">막대 위 숫자 = 정상 / <tspan fill="#d64545">불량</tspan>'+(hasLv?' · <tspan fill="#2456a6">파란 숫자·선 = 강도비 중앙값(A/θ, 오른쪽 축)</tspan>':'')+'</text>';
 var zoomed=(z0!=null&&(z0>0||z1<full.length-1));
 el.innerHTML='<div class="card" style="display:inline-block">'
  +(zoomed?'<div style="font-size:11px;color:#2456a6;margin:0 0 2px">확대 '+series[0].b+'~'+series[series.length-1].b+' (더블클릭=원복)</div>':'')
  +'<svg width="'+W+'" height="'+H+'" style="cursor:crosshair"></svg></div>';
 var sv=el.querySelector('svg');sv.innerHTML=svg;
 el.querySelectorAll('.gvb').forEach(function(x){x.addEventListener('click',function(){__gvWlist(x.getAttribute('data-b'));});});
 (function(){var x0=null,ns='http://www.w3.org/2000/svg';
  var selr=document.createElementNS(ns,'rect');
  selr.setAttribute('y',pt);selr.setAttribute('height',ih);
  selr.setAttribute('fill','#2456a6');selr.setAttribute('fill-opacity','0.12');
  selr.setAttribute('stroke','#2456a6');selr.setAttribute('stroke-dasharray','3 2');
  selr.style.display='none';sv.appendChild(selr);
  function toI(mx){var i=Math.floor((mx-pl)/iw*series.length);return i<0?0:(i>series.length-1?series.length-1:i);}
  sv.addEventListener('mousedown',function(e){x0=e.offsetX;selr.style.display='';selr.setAttribute('x',x0);selr.setAttribute('width',0);});
  sv.addEventListener('mousemove',function(e){if(x0==null)return;var a=Math.min(x0,e.offsetX);selr.setAttribute('x',a);selr.setAttribute('width',Math.abs(e.offsetX-x0));});
  sv.addEventListener('mouseup',function(e){if(x0==null)return;var i0=toI(Math.min(x0,e.offsetX)),i1=toI(Math.max(x0,e.offsetX));x0=null;selr.style.display='none';
   if(i1-i0<1)return;var b0=(z0!=null?z0:0);__gvBars(el,full,b0+i0,b0+i1);});
  sv.addEventListener('dblclick',function(){__gvBars(el,full);__gvWlist(null);});
 })();}
/* 강도 추세: 하/중/상 적층 막대 + 평균 강도비(A/θ) 선 — 서서히 세지는 추세를 보는 그래프 */
function __gvBarsLv(el,series,z0,z1){
 if(!series||!series.length){el.innerHTML='';return;}
 var full=series;if(z0!=null){series=series.slice(z0,z1+1);if(!series.length)series=full;}
 var W=Math.max(900,Math.min(1400,series.length*20+90)),H=320,pl=48,pr=54,pt=34,pb=38;
 var iw=W-pl-pr,ih=H-pt-pb,mx=1,mr=0.1,lw=__gvLvW();
 series.forEach(function(s){var t=s.n[0]+s.n[1]+s.n[2];if(t>mx)mx=t;if(s.mean>mr)mr=s.mean;});
 mx=Math.ceil(mx*1.25);mr=Math.max(Math.min(1,lw),mr*1.15);
 var bw=Math.max(2,Math.min(40,iw/series.length-3)),svg='',cols=['#9fb7cc','#f0a63a','#d64545'],names=['하','중','상'];
 var nt=(mx<=4?mx:4);                          /* 장수가 적으면 정수 눈금만(값과 위치 일치) */
 for(var g=0;g<=nt;g++){var v=Math.round(mx*g/nt),gy=pt+ih*(1-g/nt);
  svg+='<line x1="'+pl+'" y1="'+gy+'" x2="'+(W-pr)+'" y2="'+gy+'" stroke="#e5e9f0"/>'
   +'<text x="'+(pl-5)+'" y="'+(gy+4)+'" text-anchor="end" font-size="10" fill="#8a94a3">'+v+'</text>'
   +'<text x="'+(W-pr+5)+'" y="'+(gy+4)+'" font-size="10" fill="#2456a6">'+__gvF((mr*g/nt),2)+'</text>';}
 var lb=Math.max(1,Math.floor(series.length/12)),pts=[];
 series.forEach(function(s,i){
  var x=pl+iw*(i+0.5)/series.length-bw/2,y0=pt+ih,tip=s.b+' 하 '+s.n[0]+' / 중 '+s.n[1]+' / 상 '+s.n[2]+' · 강도비 중앙값 '+__gvF(s.mean,2);
  for(var l=0;l<3;l++){var hh=ih*s.n[l]/mx;y0-=hh;
   svg+='<rect class="gvlb" data-b="'+s.b+'" x="'+__gvF(x,1)+'" y="'+__gvF(y0,1)+'" width="'+__gvF(bw,1)+'" height="'+__gvF(hh,1)+'" fill="'+cols[l]+'" style="cursor:pointer"><title>'+tip+'</title></rect>';}
  if(i%lb===0)svg+='<text x="'+__gvF((x+bw/2),1)+'" y="'+(H-18)+'" text-anchor="middle" font-size="9" fill="#8a94a3">'+s.b.slice(5)+'</text>';
  if(bw>=30)svg+='<text x="'+__gvF((x+bw/2),1)+'" y="'+__gvF((y0-3),1)+'" text-anchor="middle" font-size="9" fill="#5b6675">'+s.n[0]+((s.n[1]||s.n[2])?' / <tspan fill="#b26a00">'+s.n[1]+'</tspan>/<tspan fill="#d64545">'+s.n[2]+'</tspan>':'')+'</text>';
  else if(bw>=7&&(s.n[1]||s.n[2]))svg+='<text x="'+__gvF((x+bw/2),1)+'" y="'+__gvF((y0-3),1)+'" text-anchor="middle" font-size="8" fill="#d64545">'+(s.n[1]+s.n[2])+'</text>';
  pts.push([(x+bw/2),pt+ih*(1-s.mean/mr)]);});
 if(pts.length>1)svg+='<polyline fill="none" stroke="#2456a6" stroke-width="1.6" points="'+pts.map(function(q){return __gvF(q[0],1)+','+__gvF(q[1],1);}).join(' ')+'"/>';
 pts.forEach(function(q){svg+='<circle cx="'+__gvF(q[0],1)+'" cy="'+__gvF(q[1],1)+'" r="2.2" fill="#2456a6"/>';});
 var lx=W-430;var nm2=['하(정상)','중(불량)','상(불량)'];nm2.forEach(function(nm,l){svg+='<rect x="'+(lx+l*62)+'" y="6" width="9" height="9" fill="'+cols[l]+'"/><text x="'+(lx+l*62+12)+'" y="15" font-size="10" fill="#5b6675">'+nm+'</text>';});
 svg+='<line x1="'+(lx+190)+'" y1="10" x2="'+(lx+208)+'" y2="10" stroke="#2456a6" stroke-width="1.6"/><text x="'+(lx+212)+'" y="15" font-size="10" fill="#2456a6">강도 지수(하0·중1·상'+lw+' 평균, 오른쪽 축)</text>'
  +'<text x="'+pl+'" y="15" font-size="10" fill="#5b6675">막대 위 숫자 = 하 / <tspan fill="#b26a00">중</tspan>/<tspan fill="#d64545">상</tspan></text>';
 var zoomed=(z0!=null&&(z0>0||z1<full.length-1));
 el.innerHTML='<div class="card" style="display:inline-block">'
  +(zoomed?'<div style="font-size:11px;color:#2456a6;margin:0 0 2px">확대 '+series[0].b+'~'+series[series.length-1].b+' (더블클릭=원복)</div>':'')
  +'<svg width="'+W+'" height="'+H+'" style="cursor:crosshair"></svg></div>';
 var sv=el.querySelector('svg');sv.innerHTML=svg;
 el.querySelectorAll('.gvlb').forEach(function(x){x.addEventListener('click',function(){__gvWlist(x.getAttribute('data-b'));});});
 (function(){var x0=null,ns='http://www.w3.org/2000/svg';
  var selr=document.createElementNS(ns,'rect');selr.setAttribute('y',pt);selr.setAttribute('height',ih);
  selr.setAttribute('fill','#2456a6');selr.setAttribute('fill-opacity','0.12');selr.setAttribute('stroke','#2456a6');selr.setAttribute('stroke-dasharray','3 2');
  selr.style.display='none';sv.appendChild(selr);
  function toI(mx){var i=Math.floor((mx-pl)/iw*series.length);return i<0?0:(i>series.length-1?series.length-1:i);}
  sv.addEventListener('mousedown',function(e){x0=e.offsetX;selr.style.display='';selr.setAttribute('x',x0);selr.setAttribute('width',0);});
  sv.addEventListener('mousemove',function(e){if(x0==null)return;var a=Math.min(x0,e.offsetX);selr.setAttribute('x',a);selr.setAttribute('width',Math.abs(e.offsetX-x0));});
  sv.addEventListener('mouseup',function(e){if(x0==null)return;var i0=toI(Math.min(x0,e.offsetX)),i1=toI(Math.max(x0,e.offsetX));x0=null;selr.style.display='none';
   if(i1-i0<1)return;var b0=(z0!=null?z0:0);__gvBarsLv(el,full,b0+i0,b0+i1);});
  sv.addEventListener('dblclick',function(){__gvBarsLv(el,full);__gvWlist(null);});
 })();}
/* 대표 그래프: 현재 판정 기준으로 센서별 하(정상) 평균 vs 중·상(불량) 평균 파형 */
function __gvRepWave(){
 var rows=window.__GV.frows||window.__GV.rows||[];if(!rows.length){__gvMsg('먼저 판정을 실행하세요');return;}
 var host=document.getElementById('gv_rep');host.innerHTML='<div style="font-size:12.5px;color:#2456a6">대표 그래프 생성 중 — 파형 캐시에서 전체·월별 하/중/상 평균을 계산합니다…</div>';
 var p=__gvParams();p.ids=rows.map(function(r){return r.id;});
 var pa=Object.assign({},p,{by:'all'}),pm=Object.assign({},p,{by:'month'});
 Promise.all([__gvPost('/api/gv_repwave',pa),__gvPost('/api/gv_repwave',pm)]).then(function(res){
  window.__GV.rep={all:res[0],month:res[1]};
  if(!window.__GV.repSel){var sel={'전체':true};var mb=res[1].buckets||[];if(mb.length)sel[mb[mb.length-1].b]=true;window.__GV.repSel=sel;}
  __gvRepRender();__gvMsg('대표 그래프 생성 완료'+(res[0].n_raw?' (원본 읽기 '+res[0].n_raw+'장)':''));
 }).catch(function(e){host.innerHTML='';__gvMsg(e.message);});}
function __gvWaveInfo(){fetch('/api/gv_wave_info').then(__gvJf).then(function(j){var e=document.getElementById('gv_wci');if(!e)return;
 e.innerHTML='파형 캐시 '+__gvF((j.bytes/1e9),2)+' GB ('+j.files+'파일) <a href="#" onclick="__gvWaveClear();return false;" style="color:#2456a6">[비우기]</a>';}).catch(function(){});}
function __gvWaveClear(){__gvPost('/api/gv_wave_clear',{}).then(function(j){__gvMsg('파형 캐시 '+j.removed+'파일 비움 — 다음 판정 때 다시 기록됩니다');__gvWaveInfo();}).catch(function(e){__gvMsg(e.message);});}
function __gvRepToggle(b){var sel=window.__GV.repSel||{};sel[b]=!sel[b];window.__GV.repSel=sel;__gvRepRender();}
function __gvRepRender(){
 var R=window.__GV.rep;if(!R)return;var host=document.getElementById('gv_rep');host.innerHTML='';
 var ja=R.all,jm=R.month,sel=window.__GV.repSel||{};
 var tls=(ja.steps||[]).map(function(r){return {i:r[1],label:'step '+r[0]};});
 var list=[{b:'전체',d:(ja.buckets||[])[0]}].concat((jm.buckets||[]).map(function(x){return {b:x.b,d:x};})).filter(function(x){return !!x.d;});
 var h=document.createElement('div');h.style.cssText='font-size:12.5px;margin:4px 0';
 h.innerHTML='<b>대표 그래프</b> — 현재 판정 기준(강도 레벨)으로 나눈 하/중/상 평균 파형 · 기준 그룹 '+__gvEsc(ja.group)+' · 웨이퍼 '+ja.n_wafers+'장'
  +' · <span style="color:#2b7fc4">하(정상)</span> · <span style="color:#e08a00">중</span> · <span style="color:#d64545">상</span> · 가는 선=설비 통합 골든';
 host.appendChild(h);
 var tg=document.createElement('div');tg.style.cssText='font-size:12.5px;margin:2px 0 6px';
 tg.innerHTML='달 선택: '+list.map(function(x){var on=!!sel[x.b];return '<button type="button" class="gvmt" data-b="'+__gvEsc(x.b)+'" onclick="__gvRepToggle(this.getAttribute(\\'data-b\\'))" style="margin:0 4px 4px 0;padding:3px 10px;border:1px solid '+(on?'#2456a6':'#dde3ea')+';border-radius:14px;background:'+(on?'#2456a6':'#fff')+';color:'+(on?'#fff':'#2b3440')+';font-size:12px;cursor:pointer">'+__gvEsc(x.b)+' <span style="opacity:.8">'+x.d.n+'장</span></button>';}).join('');
 host.appendChild(tg);
 var LV=[['lo','하(정상) 평균','#2b7fc4'],['mid','중 평균','#e08a00'],['hi','상 평균','#d64545']];
 list.forEach(function(x){if(!sel[x.b])return;var d=x.d;
  var blk=document.createElement('div');blk.className='gvrepblk';blk.style.cssText='border:1px solid #e5e9f0;border-radius:10px;padding:6px 8px;margin:0 0 8px';
  var hd=document.createElement('div');hd.style.cssText='font-size:13px;font-weight:600;margin:0 0 4px';hd.textContent=x.b+' — '+d.n+'장';blk.appendChild(hd);
  /* 참조 행: 챔버 진입 직전 상태(참조 4센서, 웨이퍼 레벨 기준) — 하·중·상 3선 + 골든 */
  var refs=Object.keys(d.ref||{});
  if(refs.length){var rh=document.createElement('div');rh.style.cssText='font-size:12px;color:#5b6675;margin:2px 0';rh.textContent='챔버 진입 직전 상태(참조 4센서, 웨이퍼 레벨 기준)';blk.appendChild(rh);
   var rr=document.createElement('div');rr.className='gvrefrow';rr.style.cssText='display:flex;flex-wrap:nowrap;gap:6px;align-items:flex-start;margin:0 0 6px;overflow-x:auto';
   refs.forEach(function(sn){var q=d.ref[sn],cell=document.createElement('div');cell.style.cssText='flex:0 0 auto';
    var hasv=function(v){return !!(v&&v.some(function(x){return x!=null;}));};
    if(!hasv(q.lo)&&!hasv(q.mid)&&!hasv(q.hi)){cell.innerHTML='<div class="card" style="width:360px;height:140px;display:flex;align-items:center;justify-content:center;color:#8a94a3;font-size:12px">'+__gvEsc(sn)+' — 없음</div>';rr.appendChild(cell);return;}
    var base=q.g||q.lo||q.mid||q.hi,ex=[];
    LV.forEach(function(l){if(q[l[0]]&&(q.g||l[0]!=='lo'||q.lo!==base))ex.push({label:l[1]+' '+(q[l[0]+'_n']||0)+'장',vals:q[l[0]],color:l[2],dash:'none',width:1.6});});
    cell.appendChild(__chart(sn,base,null,null,null,ex,{toplabels:tls,baseLabel:(q.g?'설비 통합 골든':'하(정상) 평균'),W:360,H:150,noBaseTag:true,baseColor:(q.g?'#aab3c0':'#2b7fc4'),baseWidth:(q.g?1.1:1.6)}));
    rr.appendChild(cell);});
   blk.appendChild(rr);}
  /* 센서마다 한 줄: [하 평균][중 평균][상 평균] — 같은 골든을 기준선으로 깔아 세 칸의 세로축이 같아진다 */
  Object.keys(d.sensors||{}).forEach(function(sn){var q=d.sensors[sn];
   var row=document.createElement('div');row.className='gvreprow';row.style.cssText='display:flex;flex-wrap:nowrap;gap:6px;align-items:flex-start;margin:2px 0 8px;overflow-x:auto';
   LV.forEach(function(l){var v=q[l[0]],n=q[l[0]+'_n']||0,cell=document.createElement('div');cell.style.cssText='flex:0 0 auto';
    if(!v){cell.innerHTML='<div class="card" style="width:470px;height:196px;display:flex;align-items:center;justify-content:center;color:#8a94a3;font-size:12.5px">'+__gvEsc(sn)+' — '+l[1]+': 해당 웨이퍼 없음</div>';row.appendChild(cell);return;}
    var base=q.g||v,ex=[];if(q.g)ex.push({label:l[1],vals:v,color:l[2],dash:'none',width:2});
    cell.appendChild(__chart(sn+' — '+l[1]+' · '+n+'장',base,null,null,null,ex,{toplabels:tls,baseLabel:(q.g?'설비 통합 골든':l[1]),W:470,H:210,noBaseTag:true,baseColor:(q.g?'#aab3c0':l[2]),baseWidth:(q.g?1.2:2)}));
    row.appendChild(cell);});
   blk.appendChild(row);});
  host.appendChild(blk);});}
/* 카드: 불량 센서 먼저 · 정상 센서 접기 · 스텝별 A/α · VLM 이미지 */
function __gvCardEl(x,row){
 var wid=x.id,box=document.createElement('div');
 var bad0=!!(row.sensors&&Object.keys(row.sensors).length);
 var head=document.createElement('div');head.style.cssText='font-size:12.5px;margin:4px 0';
 var lb=[];(x.lens||[]).forEach(function(l){
  if(l.actual==null){lb.push('step '+l.step+': 판정불가(스텝 구성 다름)');return;}
  var pc=Math.round((l.actual/Math.max(1,l.golden)-1)*100);
  lb.push('step '+l.step+': 기준 '+l.golden+' / 실제 '+l.actual+' ('+(pc>=0?'+':'')+pc+'%)');});
 head.innerHTML='<b id="gvc_'+__gvEsc(wid).replace(/[^A-Za-z0-9_-]/g,'_')+'">'+__gvEsc(wid)+'</b> — '
  +(row.ng?'<span style="color:#d64545">불량</span>':'정상')
  +' · 판정 기준: '+__gvEsc(x.main_label||'')+' '+x.golden.n+'장'+(x.golden.small?' [소표본 골든(N&lt;10) — 참고용]':'')
  +(x.guide_label?' · 가이드(회색): '+__gvEsc(x.guide_label)+' '+x.guide_n+'장':'')+(bad0?' · <span style="color:#d64545">빨강 음영=불량 스텝</span>':'')
  +(x.same_set?'<br><span style="color:#b26a00">⚠ 설비 통합 골든이 이 챔버와 같은 웨이퍼 집합으로 산정됨(같은 설비·레시피 그룹에 다른 DSV 없음 — 갤러리에서 레시피 분리/산정 대상을 확인)</span>':'')
  +(row.vlm&&row.vlm.notes&&row.vlm.notes.length?'<div style="margin:4px 0;padding:6px 8px;border:1px solid #eef1f5;border-radius:8px;background:#f9fafc;white-space:pre-wrap">VLM 모양 서술\\n'+row.vlm.notes.map(function(t){return __gvEsc(t).replace(/ \\/ /g,'\\n  ');}).join('\\n')+'</div>':(row.vlm&&row.vlm.reason?'<br>VLM: '+__gvEsc(row.vlm.reason):''))
  +'<br><span style="color:#8a94a3">스텝 길이 — '+lb.join(' · ')+'</span>'
  +(function(){var t=[];Object.keys(row.sensors||{}).forEach(function(sn){(row.sensors[sn]||[]).forEach(function(b){
    if(b.rule==='wprom')t.push(__gvEsc(sn)+' W 깊이 '+__gvF(Number(b.val),2)+'℃ · 너비 '+__gvF(Number(b.width),1)+'%'+(b.flag?' · 봉우리가 최저점 이전 최고값보다 높음':''));
    else if(b.rule==='arch')t.push(__gvEsc(sn)+(b.mode!=='ratio'?' 아치 기울기 '+__gvF(Number(b.val),4)+'(스텝 '+b.step+')'+(b.gslope!=null?' · 골든 '+__gvF(Number(b.gslope),4):'')+(b.same?' · 최고값=최저값(평탄)':''):' 아치 기울기 비 '+__gvF(Number(b.val),2)+'(스텝 '+b.step+') · 골든 대비')+(b.flip&&!b.same?' · 부호 반전(올라감)':''));});});
   return t.length?'<br><span style="color:#b26a00">'+t.join(' · ')+'</span>':'';})()
  +(function(){var ir=row.ir||{};var ks=Object.keys(ir).sort(function(a,b){return a-b;});if(!ks.length)return '';var L=__gvLevel(row);
    return '<br><span style="color:#5b6675">강도(A/θ) — '+ks.map(function(st){var l=L.steps[st];return 'step '+st+' <span style="color:'+(l===2?'#d64545':(l===1?'#b26a00':'#5b6675'))+'">'+__gvLvName(l)+' '+__gvF(Number(ir[st]),2)+'</span>';}).join(' · ')+'</span>';})();
 box.appendChild(head);
 var tls=(x.steps||[]).map(function(r){return {i:r[1],label:'step '+r[0]};});
 var names=Object.keys(x.sensors||{});
 var isNg=function(sn){return !!(row.sensors&&row.sensors[sn]);};
 names.sort(function(a,b){return (isNg(b)?1:0)-(isNg(a)?1:0);});
 var alpha=parseFloat((document.getElementById('gv_a')||{value:'0.05'}).value)||0.05;
 var astp=__gvAstp();
 var stepPos={};(x.steps||[]).forEach(function(r){stepPos[r[0]]=[r[1],r[1]+r[2]-1];});
 var mainL=x.main_label||'기준 골든',guideL=x.guide_label||'가이드 골든';
 function card(sn){
  var d=x.sensors[sn];var mg=(d.g||[]).map(function(v){return v==null?null:v;});
  var bad=(row.sensors&&row.sensors[sn])||[];
  var shade=bad.map(function(b){if(b.rule==='wprom'&&b.pk!=null&&b.valley!=null)return {i0:Math.min(b.pk,b.valley),i1:Math.max(b.pk,b.valley),color:'#e08a00'};
   var p=stepPos[b.step];return p?{i0:p[0],i1:p[1],color:'#d64545'}:null;}).filter(Boolean);
  var ex=[];if(d.ge)ex.push({label:guideL,vals:d.ge,color:'#b9c2cf',dash:'none',width:1.2});
  var wrap=document.createElement('div');wrap.style.cssText='display:inline-block;vertical-align:top';
  var c=__chart(sn+(isNg(sn)?' ⚠':'')+' — 원본 스케일',mg,d.gs||null,d.w,null,ex,{toplabels:tls,shade:shade,baseLabel:mainL,wLabel:'실제'});
  wrap.appendChild(c);
  if(d.wp&&d.gp){var ex2=[];if(d.gep)ex2.push({label:guideL,vals:d.gep,color:'#b9c2cf',dash:'none',width:1.2});
   var c2=__chart(sn+' — 패턴 비교(스텝별 offset 제거, 판정 공간)',d.gp,d.gs||null,d.wp,null,ex2,{toplabels:tls,shade:shade,baseLabel:mainL,wLabel:'실제'});
   wrap.appendChild(c2);}
  var c=wrap;
  var av=(x.A&&x.A[sn])||[];
  var jt=__gvJtState();
  if(av.length){var t=document.createElement('div');t.style.cssText='font-size:11px;color:#5b6675;margin:0 0 6px 6px';
   var lvt2=__gvLv();
   t.innerHTML='스텝별 A/불량 기준(θ×중≥): '+av.map(function(p){var th=(jt.acell[sn+'|'+p[0]]!=null?jt.acell[sn+'|'+p[0]]:(jt.asn[sn]!=null?jt.asn[sn]:(astp[String(p[0])]!=null?astp[String(p[0])]:alpha)));
    var t1=(lvt2[String(p[0])]||[0.5,1])[0],te=th*t1;var bad=p[1]!=null&&p[1]>=te;return '<span style="margin-right:6px;'+(bad?'color:#d64545;font-weight:600':'')+'">s'+p[0]+' '
     +(p[1]==null?'불가':__gvF(p[1],3))+'/'+__gvF(te,3)+'</span>';}).join('');c.appendChild(t);}
  var vs=row.vlm&&row.vlm.steps&&row.vlm.steps[sn];
  if(vs&&vs.length){vs.forEach(function(st){var im=document.createElement('img');im.style.cssText='max-width:520px;display:block;margin:0 0 6px 6px;border:1px solid #eef1f5';
   im.alt='VLM step '+st;__gvPost('/api/gv_vlm_png',{id:wid,sensor:sn,step:st}).then(function(j){im.src='data:image/png;base64,'+j.png;}).catch(function(){});
   c.appendChild(im);});}
  return c;}
 var G=document.createElement('div');G.className='grid';box.appendChild(G);
 var ngn=names.filter(isNg),okn=names.filter(function(n){return !isNg(n);});
 ngn.forEach(function(sn){G.appendChild(card(sn));});
 if(okn.length){
  if(!ngn.length){okn.forEach(function(sn){G.appendChild(card(sn));});}
  else{var b=document.createElement('button');b.style.cssText='padding:4px 12px;border:1px solid #dde3ea;border-radius:6px;background:#fff;font-size:12px;cursor:pointer;margin:4px 0';
   b.textContent='정상 센서 '+okn.length+'개 펼치기';var G2=document.createElement('div');G2.className='grid';G2.style.display='none';
   b.addEventListener('click',function(){if(G2.style.display==='none'){G2.style.display='';if(!G2.children.length)okn.forEach(function(sn){G2.appendChild(card(sn));});b.textContent='정상 센서 '+okn.length+'개 접기';}
    else{G2.style.display='none';b.textContent='정상 센서 '+okn.length+'개 펼치기';}});
   box.appendChild(b);box.appendChild(G2);}}
 return box;}
function __gvCards(wid){
 var host=document.getElementById('gv_cards');
 host.innerHTML='<span style="font-size:12px;color:#5b6675">카드 준비 중…</span>';
 __gvPost('/api/gv_wafer',{id:wid,k:__gvParams().k}).then(function(x){
  host.innerHTML='';
  var row=(window.__GV.rows||[]).filter(function(r){return r.id===wid;})[0]||{};
  host.appendChild(__gvCardEl(x,row));host.scrollIntoView({behavior:'smooth',block:'start'});
 }).catch(function(e){host.innerHTML='<span style="color:#d64545;font-size:12px">'+__gvEsc(e.message)+'</span>';});}
/* 결과 → 분석자료/패턴 인식 달력 마크 */
function __gvToAnl(){var rows=(window.__GV.frows||window.__GV.rows||[]).filter(function(r){return r.ng;});
 if(!rows.length){__gvMsg('불량 웨이퍼가 없습니다');return;}
 window.__CALM=rows.map(function(r){var gk=(r.gkey||'').split('|');
  return {date:r.date,eqp:gk[0]||'',dsv:gk[2]||'',text:'골든 비교 불량 '+Object.keys(r.sensors||{}).join('/')};});
 if(window.__ANAGG)__anAgg(false);if(window.__PTX)__ptRender(window.__PTX);
 __gvMsg('불량 '+rows.length+'장 일자를 달력 마크로 보냈습니다(분석자료·패턴 인식 탭)');}
/* 조건 저장/복원(서버 상태 — 재접속·재시작 유지) */
function __gvCond(){var o={};['gv_fol','gv_eq','gv_dv','gv_rc','gv_mo','gv_csv','gv_st','gv_d0','gv_d1','gv_e0','gv_e1','gv_md','gv_gref','gv_pre','gv_a','gv_k','gv_b','gv_w','gv_ag','gv_rw_t1','gv_rw_t2','gv_rw_st','gv_ra_mode','gv_ra_t1','gv_ra_t2','gv_ra_st']
 .forEach(function(id){var e=document.getElementById(id);if(e)o[id]=e.value;});o.per_dsv=__gvPerDsv();o.alpha_steps=__gvAstp();o.sensors=__gvChk('gvcsn');o.vlm_notes=__gvVlmNotes();o.jt=__gvJtState();
 if(document.querySelectorAll('input.gvjeq').length)o.jsel={eqp:__gvChk('gvjeq'),dsv:__gvChk('gvjdv')};
 if(document.querySelectorAll('#gv_lvs .gvl1').length)o.levels=__gvLv();
 ['gv_rw_cb','gv_ra_lr','gv_ra_cb'].forEach(function(id){var e=document.getElementById(id);if(e)o[id]=!!e.checked;});return o;}
function __gvSaveCond(quiet){var c=__gvCond();__gvPost('/api/gv_state',{save:c,live:c}).then(function(){if(!quiet)__gvMsg('분석 조건 저장됨');}).catch(function(e){if(!quiet)__gvMsg(e.message);});}
function __gvApplyCond(o){if(!o)return;Object.keys(o).forEach(function(id){var e=document.getElementById(id);if(e&&typeof o[id]==='string')e.value=o[id];});
 if(o.sensors&&o.sensors.length){document.querySelectorAll('input.gvcsn').forEach(function(c){c.checked=o.sensors.indexOf(c.value)>=0;});}
 if(o.per_dsv){Object.keys(o.per_dsv).forEach(function(dv){var a=document.querySelector('#gv_gpd .gvgp0[data-dsv="'+dv.replace(/"/g,'\\"')+'"]'),b=document.querySelector('#gv_gpd .gvgp1[data-dsv="'+dv.replace(/"/g,'\\"')+'"]');
  if(a)a.value=o.per_dsv[dv].d0||'';if(b)b.value=o.per_dsv[dv].d1||'';});}
 if(o.alpha_steps){document.querySelectorAll('#gv_astp .gvas').forEach(function(x){var v=o.alpha_steps[x.getAttribute('data-st')];x.value=(v==null?'':v);});}
 if(o.jsel){document.querySelectorAll('input.gvjeq').forEach(function(c){c.checked=(o.jsel.eqp||[]).indexOf(c.value)>=0;});
  document.querySelectorAll('input.gvjdv').forEach(function(c){c.checked=(o.jsel.dsv||[]).indexOf(c.value)>=0;});}
 ['gv_rw_cb','gv_ra_lr','gv_ra_cb'].forEach(function(id){var e=document.getElementById(id);if(e&&o[id]!=null)e.checked=!!o[id];});
 if(o.gv_ra_mode){window.__GV.raModePrev=o.gv_ra_mode;__gvRaMode();}
 if(o.levels){document.querySelectorAll('#gv_lvs .gvl1').forEach(function(x){var v=o.levels[x.getAttribute('data-st')];if(v)x.value=v[0];});
  document.querySelectorAll('#gv_lvs .gvl2').forEach(function(x){var v=o.levels[x.getAttribute('data-st')];if(v)x.value=v[1];});}
 if(o.jt&&document.querySelectorAll('#gv_jt .gvjs').length){__gvJtRender(window.__GV.sens||__gvChk('gvcsn'),(function(){var st=[];document.querySelectorAll('#gv_astp .gvas').forEach(function(x){st.push(parseInt(x.getAttribute('data-st'),10));});return st;})(),o.jt);}
 if(o.vlm_notes){var vn=document.getElementById('gv_vn');if(vn)vn.value=o.vlm_notes.common||'';
  document.querySelectorAll('#gv_vns .gvvn').forEach(function(x){x.value=(o.vlm_notes.steps||{})[x.getAttribute('data-st')]||'';});}}
function __gvLoadCond(){fetch('/api/gv_state').then(__gvJf).then(function(j){__gvApplyCond(j.cond);__gvMsg(j.cond?'조건 불러옴':'저장된 조건이 없습니다');if(j.cond)__gvLive();}).catch(function(e){__gvMsg(e.message);});}
function __gvRestoreSession(){
 fetch('/api/gv_state').then(__gvJf).then(function(j){
  if(window.__GV.userActed)return;j.cond=j.live||j.cond;            /* 사용자가 이미 시작 — 늦은 복원 응답 무시 */
  if(j.restoring){__gvMsg('이전 세션 복원 중(저장된 골든·판정 캐시 재적재)…');
   if(j.listed){__gvAfterList(j.listed);if(j.cond)__gvApplyCond(j.cond);__gvToggle(1,true);}
   window.__GV.lastOp='run';__gvPoll();return;}
  if(j.cond)__gvApplyCond(j.cond);
  if(j.listed){__gvAfterList(j.listed);if(j.cond)__gvApplyCond(j.cond);__gvToggle(1,true);}
  if(j.cond){var e0=j.cond.gv_e0||'',e1=j.cond.gv_e1||'';
   __gvMsg('저장된 분석 조건을 불러왔습니다'+((e0||e1)?' — 분석 기간 '+e0+'~'+e1+' (비우면 전체)':'')+' · [조건 불러오기]/직접 수정 가능');}
  if(j.has_build){window.__GV.lastOp='';fetch('/api/gv_status').then(__gvJf).then(function(st){
   if(st.build){var g=st.build,ss=[];(g.goldens||[]).forEach(function(x){ss.push((x.dsv||'설비 대표')+' '+x.n+'장');});
    document.getElementById('gv_gsum').textContent='골든 '+(g.goldens||[]).length+'개(보고분 제외 '+g.excluded_n+'장): '+ss.join(' · ');
    __gvStep(2,'골든 '+(g.goldens||[]).length+'개',true);__gvGallery();}
   if(st.steps&&st.steps.length){var ah=document.getElementById('gv_astp'),h2='스텝별 면적 임계 α(비우면 공통): ';
    st.steps.forEach(function(s2){h2+='<span style="margin-right:8px;white-space:nowrap">step '+s2+' <input class="gvas" data-st="'+s2+'" value="" onchange="__gvLive()" style="width:48px;font-size:12px;border:1px solid #dde3ea;border-radius:6px;padding:2px 5px"></span>';});
    ah.innerHTML=h2;document.getElementById('gv_lvs').innerHTML=__gvLvHtml(st.steps,(j.cond&&j.cond.levels)||null);
    if(st.sensors&&st.sensors.length)window.__GV.sens=st.sensors;
    __gvRulesMark(window.__GV.sens||__gvChk('gvcsn'));
    __gvJtRender(window.__GV.sens||__gvChk('gvcsn'),st.steps,(j.cond&&j.cond.jt)||null);if(j.cond)__gvApplyCond({alpha_steps:j.cond.alpha_steps});}
   if(j.has_scores){__gvToggle(3,true);__gvResults();}});}
 }).catch(function(){});}
document.addEventListener('DOMContentLoaded',function(){
 if(!document.getElementById('gv_fol'))return;
 var done=false;document.querySelectorAll('.tabs button').forEach(function(b){
  if(b.textContent.trim()==='골든 비교')b.addEventListener('click',function(){if(!done){done=true;__gvRestoreSession();}});});});
function __gvCsvOut(){
 var rows=window.__GV.frows||window.__GV.rows||[];if(!rows.length)return;
 var c='id,date,group,verdict,ng_sensors,len_ng,vlm\\n';
 rows.forEach(function(r){
  var ss=Object.keys(r.sensors||{}).map(function(s){return s+'('+r.sensors[s].map(function(b){return 'step'+b.step+':'+b.A;}).join('/')+')';}).join(' ');
  var ln=(r.len_ng||[]).map(function(l){return 'step'+l.step+':'+l.ratio;}).join(' ');
  c+='"'+r.id+'",'+r.date+',"'+(r.gkey||'')+'",'+(r.ng?'NG':'OK')+',"'+ss+'","'+ln+'","'+String(r.vlm&&r.vlm.reason||'').replace(/"/g,"'")+'"\\n';});
 __dlSave(new Blob(['﻿'+c],{type:'text/csv'}),'golden_compare_'+Date.now()+'.csv');}
function __gvExportFull(){
 var rows=window.__GV.frows||window.__GV.rows,series=window.__GV.series;
 if(!rows||!rows.length){__gvMsg('먼저 판정을 실행하세요');return;}
 var ngs=rows.filter(function(r){return r.ng;}).slice(0,300);
 var oks=rows.filter(function(r){return !r.ng;}).slice(0,5);
 var picks=ngs.concat(oks);
 var hid=document.createElement('div');hid.style.cssText='position:absolute;left:-9999px;top:0';
 document.body.appendChild(hid);
 var cardsHtml=[],i=0;
 function next(){
  if(i>=picks.length){finish();return;}
  var r=picks[i];__gvMsg('HTML 내보내기 — 카드 수집 '+(i+1)+'/'+picks.length);
  __gvPost('/api/gv_wafer',{id:r.id,k:__gvParams().k}).then(function(x){
   hid.innerHTML='';hid.appendChild(__gvCardEl(x,r));
   cardsHtml.push('<div style="border-top:1px solid #e5e9f0;margin:12px 0;padding-top:8px">'+hid.innerHTML+'</div>');
   i++;next();
  }).catch(function(){i++;next();});}
 function finish(){
  hid.innerHTML='';__gvBarsLv(hid,window.__GV.lvseries||[]);var barsH=hid.innerHTML;
  var repH=document.getElementById('gv_rep').innerHTML;if(repH)barsH+='<div style="margin-top:6px">'+repH+'</div>';
  document.body.removeChild(hid);
  var p=__gvParams();
  var gsum=document.getElementById('gv_gsum').textContent;
  var nNg=rows.filter(function(r){return r.ng;}).length;
  var tbl='<table style="border-collapse:collapse;font-size:12px"><tr><th style="border:1px solid #dde3ea;padding:2px 8px">웨이퍼</th>'
   +'<th style="border:1px solid #dde3ea;padding:2px 8px">날짜</th><th style="border:1px solid #dde3ea;padding:2px 8px">판정</th>'
   +'<th style="border:1px solid #dde3ea;padding:2px 8px">불량 센서(스텝)</th><th style="border:1px solid #dde3ea;padding:2px 8px">강도</th><th style="border:1px solid #dde3ea;padding:2px 8px">VLM 근거</th></tr>';
  rows.forEach(function(r){var L=__gvLevel(r);
   var ss=Object.keys(r.sensors||{}).map(function(s){return __gvEsc(s)+'('+r.sensors[s].map(function(b){return 'step'+b.step;}).join('/')+')';}).join(' ');
   tbl+='<tr><td style="border:1px solid #dde3ea;padding:2px 8px">'+__gvEsc(r.id)+'</td><td style="border:1px solid #dde3ea;padding:2px 8px">'+r.date+'</td>'
    +'<td style="border:1px solid #dde3ea;padding:2px 8px;color:'+(r.ng?'#d64545':'#5b6675')+'">'+(r.ng?'불량':'정상')+'</td>'
    +'<td style="border:1px solid #dde3ea;padding:2px 8px">'+ss+'</td><td style="border:1px solid #dde3ea;padding:2px 8px">'+__gvLvName(L.lv)+' '+L.maxr.toFixed(2)+'</td><td style="border:1px solid #dde3ea;padding:2px 8px">'+__gvEsc(r.vlm&&r.vlm.reason||'')+'</td></tr>';});
  tbl+='</table>';
  var doc='<!DOCTYPE html><html><head><meta charset="utf-8"><title>골든 비교</title>'
   +'<style>body{font-family:sans-serif;margin:16px;color:#2b3440}.card{display:inline-block;border:1px solid #e5e9f0;border-radius:10px;padding:8px;margin:4px;vertical-align:top}'
   +'.card .t{font-size:12px;font-weight:600;margin:0 0 4px}.grid{display:flex;flex-wrap:wrap}.n{font-weight:400;color:#8a94a3}</style></head><body>'
   +'<h2>골든 비교 결과</h2><div style="font-size:12.5px;color:#5b6675">'+__gvEsc(gsum)
   +'<br>판정 파라미터: σ밴드 배수 k='+p.k+' · 면적 임계 α='+p.alpha+' · 길이 임계 β='+p.beta+' · 모드='+p.mode
   +(Object.keys(p.alpha_steps).length?' · 스텝별 α='+JSON.stringify(p.alpha_steps):'')
   +(Object.keys(p.alpha_sensor||{}).length?' · 센서 α='+JSON.stringify(p.alpha_sensor):'')
   +(Object.keys(p.alpha_cell||{}).length?' · 센서×스텝 α='+JSON.stringify(p.alpha_cell):'')
   +(p.sensor_sel?' · 반영 센서='+p.sensor_sel.join('/'):'')
   +'<br>웨이퍼 '+rows.length+'장 중 불량 '+nNg+'장'+(ngs.length<nNg?' (카드는 불량 '+ngs.length+'장까지 수록)':'')+'</div>'
   +'<h3>추세</h3>'+barsH+'<h3>판정표</h3>'+tbl
   +'<h3>웨이퍼 카드 (불량 전수 + 정상 표본 '+oks.length+'장)</h3>'+cardsHtml.join('')+'</body></html>';
  __dlSave(new Blob([doc],{type:'text/html'}),'golden_compare_'+Date.now()+'.html');
  __gvMsg('HTML 내보내기 완료');}
 next();}
</script>"""
    return head + body + tab_js + "</body></html>"


# ── 가상센서(차분) 6종 — Vessel 내부(직접 측정 불가)를 유입·배출·전후 차분으로 유추 ──
#    예: dP_VslF-Out < 0 이면 배출부 압력이 Vessel Front 보다 높다 = 역류 유추.
_VIRTUAL_DEFS = [
    ("dT_Front-Out",  "t_002_PV",      "t_003_PV"),
    ("dT_Back-Out",   "t_001_PV",        "t_003_PV"),
    ("dT_Front-Back", "t_002_PV",      "t_001_PV"),
    ("dP_VslF-Out",   "p_008_PV",    "p_003_PV"),
    ("dP_VslB-Out",   "p_007_PV", "p_003_PV"),
    ("dP_VslF-VslB",  "p_008_PV",    "p_007_PV"),
    ("dP_Sup-VslF",   "p_006_PV",  "p_008_PV"),
    ("dP_Sup-VslB",   "p_006_PV",  "p_007_PV"),
]

# 물리 부호 규칙(사용자 지식): Vessel 압력은 항상 배출부(outlet)보다 높아야 한다.
# outlet 이 높아 차분이 음수(음압)가 되면 골든 대비와 무관하게 역류 신호.
_VIRTUAL_SIGN = {"dP_VslF-Out": "+", "dP_VslB-Out": "+"}


def _augment_virtual(recs):
    """rec['sensors'] 뒤에 차분 가상센서 열 추가(소스 두 센서가 다 있을 때만·멱등).

    모델은 센서를 rec['sensor_cols'] 이름으로 매핑하므로 추가 열은 추론에 영향 없음.
    """
    for rec in recs:
        cols = [str(c) for c in (rec.get("sensor_cols") or [])]
        if not cols:
            continue
        low = {c.lower(): j for j, c in enumerate(cols)}
        arr = None
        add_c, add_v = [], []
        for nm, pa, pb in _VIRTUAL_DEFS:
            if nm.lower() in low:
                continue
            ja, jb = low.get(pa.lower()), low.get(pb.lower())
            if ja is None or jb is None:
                continue
            if arr is None:
                a0 = np.asarray(rec["sensors"])
                if a0.ndim != 2 or a0.shape[1] < len(cols):
                    break
                arr = a0.astype(np.float64, copy=False)
            add_c.append(nm)
            add_v.append(arr[:, ja] - arr[:, jb])
        if add_c and arr is not None:
            dt0 = np.asarray(rec["sensors"]).dtype
            out_dt = dt0 if dt0.kind == "f" else np.float64
            rec["sensors"] = np.column_stack([arr] + add_v).astype(out_dt)
            rec["sensor_cols"] = cols + add_c


# 계산 시맨틱 버전 — 차이/σ/판정 규칙이 바뀌면 올린다. 저장된 결과·캐시를 무효화해
# 옛 로직으로 계산된 값이 새 버전에서 그대로 복원되는 것을 막는다(v110).
CALC_VER = "v98-stepalign-1"


def _step_runs(step_arr):
    """스텝 배열 → 연속 런 [(step, start, end_excl)] 목록."""
    runs = []
    s0 = 0
    for i in range(1, len(step_arr) + 1):
        if i == len(step_arr) or step_arr[i] != step_arr[s0]:
            runs.append((int(step_arr[s0]), s0, i))
            s0 = i
    return runs


def _warp_to_canon(mat, step_raw, canon_runs, L9):
    """(T,C) 행렬을 캐논 스텝 격자(L9)로 조각 워핑 — 스텝 런별 리샘플.

    같은 스텝의 k번째 런끼리 매칭. 없는 런은 NaN(카운트 제외). 공정 길이가
    제각각이어도 전이(피크) 위치가 격자에 정렬된다 — 골든 뭉개짐 방지의 핵심.
    """
    out = np.full((L9, mat.shape[1]), np.nan)
    occ = {}
    for u, s, e in _step_runs(step_raw):
        occ.setdefault(u, []).append((s, e))
    cnt = {}
    for u, pos, ln in canon_runs:
        k = cnt.get(u, 0)
        cnt[u] = k + 1
        lst = occ.get(u) or []
        if k >= len(lst):
            continue
        s, e = lst[k]
        if e - s < 2 or ln < 1:
            continue
        seg = mat[s:e]
        for j in range(mat.shape[1]):
            out[pos:pos + ln, j] = _resample(seg[:, j], ln)
    return out


def _arch_feat(x):
    """스텝 내 파형의 아치 특징: (base, ext, tail, rho, depth).

    base=시작 레벨(첫 k 중앙값), tail=끝 레벨(마지막 k 중앙값), ext=base 대비 최대 이탈점,
    rho=(tail-ext)/(base-ext) — 1이면 완전 회복(아치), 0이면 회복 없음(포화).
    """
    n = len(x)
    if n < 8:
        return None
    k = max(2, n // 10)
    base = float(np.median(x[:k]))
    tail = float(np.median(x[-k:]))
    j = int(np.argmax(np.abs(x - base)))
    ext = float(x[j])
    depth = abs(ext - base)
    if depth < 1e-9:
        return (base, ext, tail, 1.0, 0.0)
    rho = float((tail - ext) / (base - ext))
    return (base, ext, tail, rho, depth)


def _formula_tokens(expr, cols):
    """수식에서 센서명을 @k 변수로 치환 — '알려진 이름 우선(길이 내림차순)' 방식.

    센서명에 하이픈·숫자가 포함되어(차압류·Cart1-4류) 빼기 연산자와 충돌하므로,
    일반 토크나이저 대신 그 파일의 센서명 사전을 긴 것부터 통째로 치환한다.
    반환: (치환식, 사용 센서명 목록, [(미해결 토큰, 후보들), ...])
    """
    import re as _re
    used, out = [], str(expr or "")
    low_map = {}
    for c in cols:
        low_map.setdefault(str(c).lower(), str(c))
        low_map.setdefault(_re.sub(r"[^a-z0-9]", "", str(c).lower()), str(c))
    for c in sorted((str(c) for c in cols), key=len, reverse=True):
        if c and c in out:
            if c not in used:
                used.append(c)
            out = out.replace(c, f" @{used.index(c)} ")
    # 잔여 식별자: 대소문자/유사명 해석 시도
    unresolved = []
    def _sub(m):
        t = m.group(0)
        key = t.lower()
        c = low_map.get(key) or low_map.get(_re.sub(r"[^a-z0-9]", "", key))
        if c is None:                                  # 부분 포함 후보 탐색
            kk = _re.sub(r"[^a-z0-9]", "", key)
            cand = [str(x) for x in cols
                    if kk and (kk in _re.sub(r"[^a-z0-9]", "", str(x).lower())
                               or _re.sub(r"[^a-z0-9]", "", str(x).lower()) in kk)]
            if not cand:                               # 조각 단위(t_003 → outlet 등)
                parts = [p for p in _re.split(r"[^a-z0-9]+", key) if len(p) >= 2]
                if parts:
                    cand = [str(x) for x in cols
                            if all(p in _re.sub(r"[^a-z0-9]", "",
                                                str(x).lower()) for p in parts)]
            if len(cand) == 1:
                c = cand[0]
            else:
                unresolved.append((t, cand[:5]))
                return t
        if c not in used:
            used.append(c)
        return f" @{used.index(c)} "
    out = _re.sub(r"[A-Za-z_][A-Za-z0-9_\-]*", _sub, out)
    return out, used, unresolved


def _formula_eval(expr, cols, arr):
    """사칙연산 수식 평가(ast 화이트리스트 — eval 금지). 실패 시 (None, 사유)."""
    import ast as _ast
    sub, used, unresolved = _formula_tokens(expr, cols)
    if unresolved:
        t0, cand0 = unresolved[0]
        hint = (f" — 후보: {', '.join(cand0)}" if cand0 else "")
        return None, f"'{t0}' 센서를 찾지 못했습니다{hint}"
    sub = sub.replace("@", "v")
    try:
        tree = _ast.parse(sub.strip(), mode="eval")
    except SyntaxError:
        return None, "수식 문법 오류 — 사칙연산(+ - * /)과 괄호만 지원합니다"
    cols_l = [str(c) for c in cols]
    vecs = {f"v{k}": np.asarray(arr[:, cols_l.index(c)], dtype=np.float64)
            for k, c in enumerate(used)}

    def _ev(n):
        if isinstance(n, _ast.Expression):
            return _ev(n.body)
        if isinstance(n, _ast.BinOp) and isinstance(
                n.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div)):
            a, b = _ev(n.left), _ev(n.right)
            if isinstance(n.op, _ast.Add):
                return a + b
            if isinstance(n.op, _ast.Sub):
                return a - b
            if isinstance(n.op, _ast.Mult):
                return a * b
            return a / b
        if isinstance(n, _ast.UnaryOp) and isinstance(n.op, (_ast.USub, _ast.UAdd)):
            v = _ev(n.operand)
            return -v if isinstance(n.op, _ast.USub) else v
        if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, _ast.Name) and n.id in vecs:
            return vecs[n.id]
        raise ValueError("허용되지 않는 구문")
    try:
        with np.errstate(all="ignore"):
            y = _ev(tree)
    except ValueError:
        return None, "허용되지 않는 구문 — 사칙연산(+ - * /)과 괄호만 지원합니다"
    except ZeroDivisionError:
        return None, "0 으로 나누는 상수식입니다"
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 0:                                    # 상수식 → 길이 맞춰 확장
        y = np.full(arr.shape[0], float(y))
    y[~np.isfinite(y)] = np.nan
    return y, None


def _sensor_resolve(name, cols):
    """단일 센서명 유사 해석(계통 규칙용) — _formula_tokens 재사용. 실패 시 None."""
    try:
        _sub, used, unres = _formula_tokens(str(name or ""), cols)
        return used[0] if (len(used) == 1 and not unres) else None
    except Exception:                                  # noqa: BLE001
        return None


def _bal_steps_use(st):
    """계통 잔차 대상 스텝 — 첫/마지막 스텝 제외(이전공정 이월 관례)."""
    uniq = []
    for v in np.asarray(st, dtype=int):
        if not uniq or uniq[-1] != int(v):
            uniq.append(int(v))
    seen = []
    for v in uniq:
        if v not in seen:
            seen.append(v)
    return seen[1:-1] if len(seen) > 2 else seen


def _bal_fit(res, causes, st):
    """스텝별 릿지 최소제곱: res ≈ Σw·cause + b.

    골든 평균 곡선에서 호출 — w 는 유효 유량비의 프록시(정상 공정의 분배 지문).
    스텝 내 곡선이 평평하면 병립(collinear)하므로 소량 릿지로 안정화.
    반환 {step: {"w":[...], "b":f, "sig":f}} — sig=적합 잔차 RMS(바닥값 있음).
    """
    st = np.asarray(st, dtype=int)
    res = np.asarray(res, dtype=np.float64)
    fit = {}
    for sp in _bal_steps_use(st):
        m = st == sp
        X = np.column_stack([np.asarray(c, dtype=np.float64)[m] for c in causes]
                            + [np.ones(int(m.sum()))])
        y = res[m]
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < max(8, X.shape[1] + 2):
            continue
        Xo, yo = X[ok], y[ok]
        A = Xo.T @ Xo
        A += (1e-6 * max(float(np.trace(A)), 1e-12)) * np.eye(A.shape[0])
        try:
            wb = np.linalg.solve(A, Xo.T @ yo)
        except np.linalg.LinAlgError:
            continue
        r = yo - Xo @ wb
        rng = float(np.nanmax(yo) - np.nanmin(yo)) if len(yo) else 0.0
        sig = max(float(np.sqrt(np.mean(r * r))), 0.01 * abs(rng), 1e-9)
        fit[int(sp)] = {"w": [float(v) for v in wb[:-1]],
                        "b": float(wb[-1]), "sig": sig}
    return fit


def _bal_eval(res, causes, st, fit, k):
    """잔차 곡선 평가 — (R, z, band(+kσ 스텝별), 최장 연속 이탈 pt, 최대 z).

    학습 안 된 스텝은 nan(판정 제외). z=|R|/σ_s, 이탈=z>k.
    """
    st = np.asarray(st, dtype=int)
    res = np.asarray(res, dtype=np.float64)
    n = len(res)
    R = np.full(n, np.nan)
    z = np.full(n, np.nan)
    band = np.full(n, np.nan)
    for sp, f in fit.items():
        m = st == int(sp)
        if not m.any():
            continue
        pred = f["b"] * np.ones(int(m.sum()))
        for w, c in zip(f["w"], causes):
            pred = pred + w * np.asarray(c, dtype=np.float64)[m]
        R[m] = res[m] - pred
        z[m] = np.abs(R[m]) / f["sig"]
        band[m] = k * f["sig"]
    run = best = 0
    for v in z:
        run = run + 1 if (np.isfinite(v) and v > k) else 0
        best = max(best, run)
    mz = float(np.nanmax(z)) if np.isfinite(z).any() else float("nan")
    return R, z, band, int(best), mz


def _pt_seg(y9, st9, steps_set9):
    """스텝 배열에서 지정 스텝 구간(연속 병합)을 잘라 반환 — 없으면 None."""
    m9 = np.isin(np.asarray(st9, dtype=int), sorted(steps_set9))
    if not m9.any():
        return None
    i9 = np.where(m9)[0]
    return np.asarray(y9, dtype=np.float64)[i9[0]:i9[-1] + 1]


def _wpeak(pseg9, seg9, imin9):
    """W 피크 조건 재료(v151): (전체 최고, 관심구간 최고, raw 비율).

    전체 최고 = 직전 스텝+관심 스텝 통틀어 최고, 관심구간 = 최저점→끝.
    비율은 센서값 그대로(사용자 확정) — 최고가 0 이하면 None(조건 미적용).
    """
    seg9 = np.asarray(seg9, dtype=np.float64)
    allv9 = (np.concatenate([np.asarray(pseg9, dtype=np.float64), seg9])
             if pseg9 is not None and len(pseg9) else seg9)
    with np.errstate(all="ignore"):
        pa9 = float(np.nanmax(allv9))
        pr9 = float(np.nanmax(seg9[int(imin9):]))
    if not (np.isfinite(pa9) and np.isfinite(pr9)) or pa9 <= 1e-12:
        return None
    return {"peak_all": pa9, "peak_roi": pr9, "ratio": pr9 / pa9}


def _wprev_seg(col9, st9, step3):
    """관심 스텝 '직전' 스텝 구간(전체 최고 계산용) — 없으면 None."""
    st9 = np.asarray(st9, dtype=int)
    prevs9 = [int(v9) for v9 in set(st9.tolist()) if int(v9) < int(step3)]
    if not prevs9:
        return None
    return _pt_seg(col9, st9, {max(prevs9)})


def _pt_rows_core(S9, by_f9, sn1, sn2, sn3, step1, steps2, step3, gref9):
    """하루 전수 점수 계산 코어(13열) — 앱 직렬·병렬 워커 공용(단일 소스).

    gref9 는 JSON 왕복 가능한 dict(계통 fit 의 스텝 키가 문자열이어도 안전 —
    _bal_eval 이 int(sp) 캐스팅).
    """
    g49, g59 = gref9.get("p4"), gref9.get("p5")
    out9 = {}
    for f9, sts9 in by_f9.items():
        for c09 in range(0, len(sts9), 100):
            recs9, _a, _b, _c = S9._load_aligned(Path(f9), stems=sts9[c09:c09 + 100])
            for rec9 in recs9:
                wid9 = str(rec9["wafer_id"])
                cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
                st9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
                row9 = [np.nan] * 14
                arr9 = np.asarray(rec9["sensors"], dtype=np.float64)
                if sn1 in cols9 and gref9.get("p1"):
                    seg9 = _pt_seg(arr9[:, cols9.index(sn1)], st9, {int(step1)})
                    r9 = _shape_convexity(seg9) if seg9 is not None else None
                    if r9:
                        row9[0:3] = [r9["ratio"], r9["drop"], r9["early"]]
                        row9[5] = (float(np.nanmean(seg9))
                                   - gref9["p1"]["gmean"])
                if sn2 in cols9 and gref9.get("p2"):
                    a9 = _shape_steparea(arr9[:, cols9.index(sn2)], st9,
                                         [int(v) for v in steps2])
                    if a9:
                        row9[3:5] = [float(len(a9["areas"])), a9["total"]]
                        seg9 = _pt_seg(arr9[:, cols9.index(sn2)], st9,
                                       {int(v) for v in steps2})
                        row9[6] = (float(np.nanmean(seg9))
                                   - gref9["p2"]["gmean"])
                if sn3 in cols9 and gref9.get("p3"):
                    col39 = arr9[:, cols9.index(sn3)]
                    seg9 = _pt_seg(col39, st9, {int(step3)})
                    w9 = _shape_wbulge(seg9) if seg9 is not None else None
                    if w9:
                        row9[7] = w9["bulge"]          # W 면적(원단위) — 판정 기준
                        row9[8] = (float(np.nanmean(seg9))
                                   - gref9["p3"]["gmean"])
                        pk9 = _wpeak(_wprev_seg(col39, st9, int(step3)),
                                     seg9, w9["imin"])
                        if pk9:
                            row9[13] = pk9["ratio"]    # W 피크 비율(raw, v151)
                for off9, gb9 in ((9, g49), (11, g59)):
                    if not gb9:
                        continue
                    need9 = [gb9["sensor"]] + gb9["causes"]
                    if any(c9 not in cols9 for c9 in need9):
                        continue
                    _r9, _z9, _bd9, run9, mz9 = _bal_eval(
                        arr9[:, cols9.index(gb9["sensor"])],
                        [arr9[:, cols9.index(c9)] for c9 in gb9["causes"]],
                        st9, gb9["fit"], gb9["k"])
                    row9[off9] = mz9                   # 최대 z
                    row9[off9 + 1] = float(run9)       # 최장 연속 이탈 pt
                out9[wid9] = row9[:14]
            del recs9
    return out9


_PTW = {}                                              # 스캔 병렬 워커 상태


def _ptw_init(root9, model_dir9, cache9):
    import contextlib
    import io as _io
    try:
        os.nice(10)                                    # 스캔 워커는 화면 조작에 양보
    except Exception:                                  # noqa: BLE001
        pass
    with contextlib.redirect_stdout(_io.StringIO()):
        if root9 and root9 not in sys.path:
            sys.path.insert(0, root9)
        from io_recon_tcn.app import server as S9
        _install_virtual(S9)
        _install_parse_cache(S9, store_hint=cache9)
        if model_dir9:
            S9.ENGINE.select(model_dir9)
        S9.ENGINE.load()
        _PTW["S"] = S9


def _ptw_day(args9):
    day9, by_f9, sn9, cfg9, gref9 = args9
    rows9 = _pt_rows_core(_PTW["S"], by_f9, sn9[0], sn9[1], sn9[2],
                          cfg9[0], cfg9[1], cfg9[2], gref9)
    return day9, rows9


# ══════════════════════════════════════════════════════════════════════
# 골든 비교 — 모듈 수준 코어(v179): 프로세스 워커에서도 쓰도록 serve() 밖으로.
#   GV_KGRID·_gv_metrics_core·_gv_sensor_mat_core·_gv_pick_golden_core 는
#   serve() 안의 동명 클로저가 그대로 위임한다(직렬 경로와 병렬 경로 코드 단일).
# ══════════════════════════════════════════════════════════════════════
GV_KGRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
GV_WAVE_VER = "w1"                                     # 정합 파형 캐시 형식 버전
GV_FEAT_VER = "f5"                                     # f5: 아치 특징은 원본 스텝 구간 원본 값 그대로(리샘플·스무딩 없음)                                     # 형상 특징(F) 버전 — A 캐시 키에는 넣지 않음
GV_REF_SENSORS = ["t_001_PV", "t_002_PV",  # 월별 참조 행(챔버 진입 직전 상태) 고정 4종
                  "p_008_PV", "p_007_PV"]


def _install_listing_cache(S9):
    """★대용량 가속 — load_dataset 이 호출될 때마다 폴더 전체를 rglob 하던 것을 메모이즈.

    5만 장 폴더에서 100장 묶음마다(대표 그래프·카드는 1장마다) 전량 재탐색하던 O(n²)
    비용 제거. 무효화 = 디렉터리(서브디렉터리 포함) mtime 서명 + 2초 TTL. 파일 내용
    변경은 파싱 캐시(mtime/size)와 일별 서명이 따로 잡으므로 목록은 '존재'만 책임.
    data 모듈 전역을 바꿔야 load_dataset 이 본다(서버 바인딩 교체는 무효).
    """
    try:
        D9 = sys.modules[S9.load_dataset.__module__]
        if getattr(D9._data_files, "_memo", False):
            return D9._data_files
        orig9 = D9._data_files
        memo9 = {}

        def _dsig9(folder9):
            sig9 = []
            try:
                sig9.append((str(folder9), folder9.stat().st_mtime_ns))
                for d9 in folder9.rglob("*"):
                    if d9.is_dir():
                        sig9.append((str(d9), d9.stat().st_mtime_ns))
            except Exception:                          # noqa: BLE001
                return None
            return tuple(sig9)

        def _memo9(folder9):
            key9 = str(Path(folder9).resolve())
            ent9 = memo9.get(key9)
            now9 = time.time()
            if ent9 is not None:
                if now9 - ent9["t"] < 2.0:
                    return list(ent9["files"])
                ds9 = _dsig9(Path(folder9))
                if ds9 is not None and ds9 == ent9["dsig"]:
                    ent9["t"] = now9
                    return list(ent9["files"])
            files9 = orig9(folder9)
            memo9[key9] = {"files": list(files9), "dsig": _dsig9(Path(folder9)), "t": now9}
            return list(files9)
        _memo9._memo = True
        _memo9.clear = memo9.clear
        D9._data_files = _memo9
        try:
            S9._data_files = _memo9
        except Exception:                              # noqa: BLE001
            pass
        return _memo9
    except Exception as e9:                            # noqa: BLE001
        print(f"[cache] 목록 캐시를 켜지 못했습니다(기존 방식 진행): {e9}", flush=True)
        return None


def _gv_sensor_mat_core(rec9, sens9, miss9=None):
    """레코드 → 요청 센서 순서의 (T,S) 행렬. 없는 센서 열은 NaN.
    miss9 dict 를 주면 결측 센서를 센다(스캔 전용) — 카드·대표 그래프 호출은 세지 않음."""
    cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
    if not cols9 or not sens9:
        return None, None
    src9 = np.asarray(rec9["sensors"], dtype=np.float64)
    T9 = src9.shape[0]
    out9 = np.full((T9, len(sens9)), np.nan)
    any9 = False
    for j9, s9 in enumerate(sens9):
        if s9 in cols9:
            out9[:, j9] = src9[:, cols9.index(s9)]
            any9 = True
        elif miss9 is not None:
            miss9[s9] = miss9.get(s9, 0) + 1
    if not any9:
        return None, None
    stw9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
    return out9, stw9


def _gv_pick_golden_core(meta9, goldens9, gref9):
    """웨이퍼 메타 → (판정 기준 골든, 가이드 골든). gref 'dsv'=챔버별 우선, 'rep'=설비 통합 우선."""
    gd9 = ge9 = None
    for g9 in goldens9 or []:
        if g9["eqp"] != meta9["eqp"] or g9["recipe"] != meta9["recipe"]:
            continue
        if g9["dsv"] is None:
            ge9 = g9
        elif g9["dsv"] == meta9["dsv"]:
            gd9 = g9
    if (gref9 or "dsv") == "rep":
        return (ge9 or gd9), (gd9 if (ge9 is not None and gd9 is not None) else None)
    return (gd9 or ge9), (ge9 if (gd9 is not None and ge9 is not None) else None)


def _gv_metrics_core(wv9, g9, sens9, w9=0, noise_sigma=False):
    """캐논 정렬 웨이퍼 → 스텝×센서별 k격자 초과면적 A_k[9].
    A_k = mean(max(0, |x'−g'| − k·σ)) / amp_g (offset=스텝 내 평균 차감). 본문은 v178 과 동일."""
    canon9 = g9["canon"]
    nS9, nR9 = len(sens9), len(canon9)
    A9 = np.full((nS9, nR9, len(GV_KGRID)), np.nan)
    for j9, s9 in enumerate(sens9):
        gm9 = np.asarray([np.nan if v9 is None else v9
                          for v9 in g9["mean"][s9]], dtype=np.float64)
        gs9 = np.asarray(g9["std"][s9], dtype=np.float64)
        amp_all9 = float(np.nanmax(gm9) - np.nanmin(gm9)) or 1.0
        for r9, (_u9, p9, l9) in enumerate(canon9):
            x9 = wv9[p9:p9 + l9, j9]
            gg9 = gm9[p9:p9 + l9]
            ss9 = gs9[p9:p9 + l9]
            fin9 = np.isfinite(x9) & np.isfinite(gg9)
            if fin9.sum() < max(4, l9 // 3):
                continue
            x29 = x9[fin9]
            g29 = gg9[fin9]
            s29 = ss9[fin9]
            if w9 and w9 > 1:
                k9 = np.ones(int(w9)) / float(w9)
                x29 = np.convolve(x29, k9, mode="same")
            d9 = np.abs((x29 - np.mean(x29)) - (g29 - np.mean(g29)))
            if noise_sigma or float(np.max(s29)) <= 1e-12:
                dd9 = np.diff(x29)
                s29 = np.full_like(s29, float(
                    np.median(np.abs(dd9 - np.median(dd9)))
                    / 0.9539) if len(dd9) else 0.0)
            amp9 = max(float(np.nanmax(g29) - np.nanmin(g29)),
                       amp_all9 * 0.05, 1e-9)
            for q9, kk9 in enumerate(GV_KGRID):
                A9[j9, r9, q9] = float(np.mean(
                    np.maximum(0.0, d9 - kk9 * s29))) / amp9
    return A9


def _gv_lenratio(stw9, g9):
    """스텝 런 길이 / 골든 원 길이 중앙값 — 캐논 런 순서대로(없으면 NaN)."""
    canon9 = g9["canon"]
    occ9 = {}
    for u9, s09, e09 in _step_runs(stw9):
        occ9.setdefault(u9, []).append(e09 - s09)
    LR9 = np.full(len(canon9), np.nan)
    cnt9 = {}
    for r9, (u9, _p9, _l9) in enumerate(canon9):
        k9 = cnt9.get(u9, 0)
        cnt9[u9] = k9 + 1
        lst9 = occ9.get(u9) or []
        if k9 < len(lst9):
            LR9[r9] = lst9[k9] / max(1, g9["raw_len_med"][r9])
    return LR9


# ── 형상 특징(캐논 격자 위, 센서 공통 계산 · 규칙 활성은 판정 시 센서 이름으로) ──
def _gv_smooth(y9, k9):
    k9 = int(max(1, k9))
    if k9 <= 1 or len(y9) < k9:
        return np.asarray(y9, dtype=np.float64)
    ker9 = np.ones(k9)
    num9 = np.convolve(np.nan_to_num(y9, nan=0.0), ker9, mode="same")
    den9 = np.convolve(np.isfinite(y9).astype(float), ker9, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out9 = num9 / den9
    out9[den9 <= 0] = np.nan
    return out9


def _gv_noise_sig(y9):
    """시간 차분 MAD 노이즈 σ(판정 _gv_metrics 와 같은 정의)."""
    y9 = np.asarray(y9, dtype=np.float64)
    y9 = y9[np.isfinite(y9)]
    if len(y9) < 3:
        return 0.0
    dd9 = np.diff(y9)
    return float(np.median(np.abs(dd9 - np.median(dd9))) / 0.9539)


def _gv_peaks_prom(v9, prom_min9):
    """국소 극대와 prominence(좌우로 더 높은 점을 만나기 전 구간 최저 대비 돌출) — _shape_arches 와 같은 정의."""
    n9 = len(v9)
    out9 = []
    i9 = 1
    while i9 < n9 - 1:
        if v9[i9] >= v9[i9 - 1] and v9[i9] >= v9[i9 + 1]:
            j9 = i9
            while j9 + 1 < n9 - 1 and v9[j9 + 1] == v9[i9]:
                j9 += 1
            p9 = (i9 + j9) // 2
            lo_l9 = v9[:p9 + 1]
            lo_r9 = v9[p9:]
            hl9 = np.where(lo_l9 > v9[p9])[0]
            hr9 = np.where(lo_r9 > v9[p9])[0]
            lmin9 = float(np.min(lo_l9[hl9[-1]:] if len(hl9) else lo_l9))
            rmin9 = float(np.min(lo_r9[:hr9[0] + 1] if len(hr9) else lo_r9))
            prom9 = float(v9[p9]) - max(lmin9, rmin9)
            if prom9 >= prom_min9:
                out9.append((p9, prom9))
            i9 = j9 + 1
        else:
            i9 += 1
    return out9


W_FEAT = ["depth", "width", "flag", "pk", "valley", "nb", "imin"]     # 봉우리(되떨어짐) 특징
A_FEAT = ["s", "imin", "drop", "y0", "ymax", "imax", "flat"]           # 아치 특징(+최고값·위치·평탄 플래그, f2)


def _gv_fcols(canon9):
    """F 열 이름: 캐논 런 r 마다 w.r{r}.<k> 7열 + a.r{r}.<k> 4열."""
    cols9 = []
    for r9 in range(len(canon9)):
        cols9 += [f"w.r{r9}.{k9}" for k9 in W_FEAT]
    for r9 in range(len(canon9)):
        cols9 += [f"a.r{r9}.{k9}" for k9 in A_FEAT]
    return cols9


def _shape_wprom(y9, p9, l9, p59, l59, noise9):
    """W 봉우리: 런(p9,l9) 최저점 이후 구간에서 '되떨어짐 깊이(prominence)' 최대 봉우리.
    단조 급상승·플래토 뒤 상승은 오른쪽에 더 높은 점이 곧 있어 prom≈0 → 미검출.
    반환 [depth, width%, flag(봉우리>최저 이전 최고), pk, valley, n_bumps, imin] (격자 절대 인덱스)."""
    nan9 = [np.nan] * len(W_FEAT)
    if l9 < 12:
        return nan9
    seg9 = np.asarray(y9[p9:p9 + l9], dtype=np.float64)
    if np.isfinite(seg9).sum() < max(12, l9 // 4):
        return nan9
    head9 = np.asarray(y9[p59:p59 + max(2, int(0.02 * l59))], dtype=np.float64) if l59 else np.zeros(0)
    ys9 = _gv_smooth(np.concatenate([seg9, head9]), max(5, l9 // 100))
    lo9 = int(0.05 * l9)
    core9 = ys9[lo9:l9]
    if not np.isfinite(core9).any():
        return nan9
    imin9 = lo9 + int(np.nanargmin(core9))
    if imin9 >= l9 - 8:
        return [0.0, 0.0, 0.0, float(p9 + imin9), float(p9 + imin9), 0.0, float(p9 + imin9)]
    win9 = ys9[imin9:]
    fin9 = np.isfinite(win9)
    if fin9.sum() < 8:
        return nan9
    win9 = np.where(fin9, win9, np.nanmin(win9))
    pk9 = _gv_peaks_prom(win9, max(4.0 * float(noise9), 1e-9))
    best9 = None
    nb9 = 0
    for pp9, prom9 in pk9:
        if pp9 >= l9 - imin9:                          # 꼭대기는 런 안에 있어야(스텝 5 머리는 확인용)
            continue
        thr9 = win9[pp9] - prom9 / 2.0
        a9 = pp9
        while a9 - 1 >= 0 and win9[a9 - 1] > thr9:
            a9 -= 1
        b9 = pp9
        while b9 + 1 < len(win9) and win9[b9 + 1] > thr9:
            b9 += 1
        width9 = 100.0 * (b9 - a9 + 1) / float(l9)
        if width9 < 2.0:                               # 한 점 튀는 스파이크 제외
            continue
        nb9 += 1
        if best9 is None or prom9 > best9[1]:
            best9 = (pp9, prom9, width9)
    if best9 is None:
        return [0.0, 0.0, 0.0, float(p9 + imin9), float(p9 + imin9), 0.0, float(p9 + imin9)]
    pp9, prom9, width9 = best9
    hr9 = np.where(win9[pp9:] > win9[pp9])[0]
    seg_r9 = win9[pp9:(pp9 + hr9[0] + 1) if len(hr9) else len(win9)]
    valley9 = pp9 + int(np.argmin(seg_r9))
    pre9 = ys9[:imin9]
    pre_max9 = float(np.nanmax(pre9)) if np.isfinite(pre9).any() else -np.inf
    flag9 = 1.0 if win9[pp9] > pre_max9 else 0.0
    return [float(prom9), float(width9), flag9, float(p9 + imin9 + pp9),
            float(p9 + imin9 + valley9), float(nb9), float(p9 + imin9)]


def _shape_archslope(y9, p9, l9, noise9):
    """아치 특징: 런 첫점(첫 k점 중앙값)→최하점 첫 도달점 기울기 s(골든 비 모드용), 최고값·위치(ymax, imax),
    평탄 플래그(최고−최저 ≤ 4σ). 반환 [s, imin, drop, y0, ymax, imax, flat]."""
    nan9 = [np.nan] * len(A_FEAT)
    if l9 < 8:
        return nan9
    seg9 = np.asarray(y9[p9:p9 + l9], dtype=np.float64)
    if np.isfinite(seg9).sum() < max(8, l9 // 4):
        return nan9
    k9 = min(5, max(1, l9 // 10))
    y09 = float(np.nanmedian(seg9[:k9]))
    ys9 = seg9                                         # ★원본 값 그대로(스무딩·가공 없음)
    ymin9 = float(np.nanmin(ys9))
    imin_g9 = int(np.nanargmin(ys9))
    # 최고값은 최저점 '이전' 구간에서(하강 구간의 기울기) — 전체에서 찾으면 회복 끝 노이즈가 최고점이 돼 부호가 뒤집힘
    pre9 = ys9[:imin_g9 + 1]
    imax9 = int(np.nanargmax(pre9)) if np.isfinite(pre9).any() else 0
    ymax9 = float(pre9[imax9]) if np.isfinite(pre9).any() else ymin9
    flat9 = 1.0 if (ymax9 - ymin9) <= 4.0 * float(noise9) else 0.0
    drop9 = y09 - ymin9
    if not np.isfinite(drop9) or drop9 <= 4.0 * float(noise9):
        return [0.0, float(imin_g9), float(drop9 if np.isfinite(drop9) else 0.0), y09, ymax9, float(imax9), flat9]
    reach9 = np.where(ys9 <= ymin9 + 0.1 * drop9)[0]
    reach9 = reach9[reach9 >= 1]
    imin9 = int(reach9[0]) if len(reach9) else imin_g9
    s9 = (float(ys9[imin9]) - y09) / float(imin9) if imin9 > 0 else 0.0
    return [float(s9), float(imin9), float(drop9), y09, ymax9, float(imax9), flat9]


def _gv_feats_one(wv9, canon9, j9=None, raw9=None, stw9=None):
    """정합 파형(L,S) 또는 (L,) → 센서별 F 행 (nS, nFeat). 노이즈 σ는 센서 전체 시간차분 MAD.
    raw9(T,S)·stw9(T) 를 주면 아치 특징은 ★원본 스텝 구간(리샘플 없음, 원본 포인트 단위)에서 계산한다(f4).
    골든(원본 없음)은 캐논 격자에서 계산."""
    if wv9.ndim == 1:
        wv9 = wv9[:, None]
    nS9 = wv9.shape[1]
    nF9 = len(canon9) * (len(W_FEAT) + len(A_FEAT))
    F9 = np.full((nS9, nF9), np.nan, dtype=np.float32)
    nR9 = len(canon9)
    runs9 = None
    if raw9 is not None and stw9 is not None:
        occ9 = {}
        for u9, s09, e09 in _step_runs(stw9):
            occ9.setdefault(int(u9), []).append((int(s09), int(e09)))
        runs9, cnt9 = [], {}
        for u9, _p9, _l9 in canon9:                    # 캐논 런 r ↔ 원본의 같은 스텝 k번째 런
            k9 = cnt9.get(int(u9), 0); cnt9[int(u9)] = k9 + 1
            lst9 = occ9.get(int(u9)) or []
            runs9.append(lst9[k9] if k9 < len(lst9) else None)
    for jj9 in range(nS9):
        y9 = wv9[:, jj9]
        if not np.isfinite(y9).any():
            continue
        noise9 = _gv_noise_sig(y9)
        row9 = []
        for r9, (_u9, p9, l9) in enumerate(canon9):
            if r9 + 1 < nR9:
                _u59, p59, l59 = canon9[r9 + 1]
            else:
                p59, l59 = p9 + l9, 0
            row9 += _shape_wprom(y9, int(p9), int(l9), int(p59), int(l59), noise9)
        if runs9 is not None:
            yr9 = np.asarray(raw9[:, jj9], dtype=np.float64)
            noise_r9 = _gv_noise_sig(yr9)
            for r9 in range(nR9):
                rr9 = runs9[r9]
                if rr9 is None:
                    row9 += [np.nan] * len(A_FEAT)
                else:
                    row9 += _shape_archslope(yr9, rr9[0], rr9[1] - rr9[0], noise_r9)
        else:
            for r9, (_u9, p9, l9) in enumerate(canon9):
                row9 += _shape_archslope(y9, int(p9), int(l9), noise9)
        F9[jj9, :] = np.asarray(row9, dtype=np.float32)
    return F9


# ── 정합 파형 캐시: <cache_root>/gv_wave/<canon_sig>/<day>.npz ──
def _gv_canon_sig(canon9, L9):
    c9 = [[int(u9), int(p9), int(l9)] for u9, p9, l9 in canon9]
    return hashlib.md5(f"{json.dumps(c9)}|{int(L9)}|{GV_WAVE_VER}".encode()).hexdigest()[:12]


def _gv_wave_save(dir9, day9, ids9, fks9, sens9, W9, canon9, L9):
    """일별 파형 npz 저장(기존 파일과 병합: 같은 id 는 교체, 센서는 합집합·없는 열 NaN). tmp→replace."""
    try:
        dir9 = Path(dir9)
        dir9.mkdir(parents=True, exist_ok=True)
        fp9 = dir9 / f"{day9}.npz"
        ids9 = [str(x9) for x9 in ids9]
        fks9 = [str(x9) for x9 in fks9]
        sens9 = [str(x9) for x9 in sens9]
        W9 = np.asarray(W9, dtype=np.float32)
        if fp9.is_file():
            try:
                with np.load(fp9, allow_pickle=True) as z9:
                    if str(z9["ver"]) == GV_WAVE_VER and int(z9["L"]) == int(L9):
                        o_ids9 = [str(v9) for v9 in z9["ids"]]
                        o_fk9 = [str(v9) for v9 in z9["fk"]]
                        o_sens9 = [str(v9) for v9 in z9["sens"]]
                        o_W9 = z9["W"]
                        u_sens9 = list(o_sens9) + [s9 for s9 in sens9 if s9 not in o_sens9]
                        keep9 = [i9 for i9, w9 in enumerate(o_ids9) if w9 not in set(ids9)]
                        n9 = len(keep9) + len(ids9)
                        U9 = np.full((n9, len(u_sens9), int(L9)), np.nan, dtype=np.float32)
                        for k9, i9 in enumerate(keep9):
                            for jj9, s9 in enumerate(o_sens9):
                                U9[k9, u_sens9.index(s9), :] = o_W9[i9, jj9, :]
                        for k9 in range(len(ids9)):
                            for jj9, s9 in enumerate(sens9):
                                U9[len(keep9) + k9, u_sens9.index(s9), :] = W9[k9, jj9, :]
                        ids9 = [o_ids9[i9] for i9 in keep9] + ids9
                        fks9 = [o_fk9[i9] for i9 in keep9] + fks9
                        sens9, W9 = u_sens9, U9
            except Exception:                          # noqa: BLE001
                pass
        tmp9 = fp9.with_name(f"{day9}.tmp.npz")      # numpy 는 .npz 로 끝나지 않으면 .npz 를 덧붙인다
        np.savez_compressed(tmp9, ids=np.array(ids9), fk=np.array(fks9), sens=np.array(sens9),
                            W=W9, canon=np.array([[int(u9), int(p9), int(l9)] for u9, p9, l9 in canon9]),
                            L=int(L9), ver=GV_WAVE_VER)
        os.replace(tmp9, fp9)
        return True
    except Exception as e9:                            # noqa: BLE001
        print(f"[gv] 파형 캐시 저장 실패({day9}): {e9}", flush=True)
        return False


def _gv_wave_load(dir9, day9, want_ids9, want_sens9, fk_of9=None):
    """일별 파형 읽기 → (W (n_hit, nS_want, L) float32, hit_ids, miss_ids).
    fk_of9={id: 'mtime|size'} 를 주면 파일이 바뀐 웨이퍼는 miss 로. 없는 센서 열은 NaN."""
    fp9 = Path(dir9) / f"{day9}.npz"
    want_ids9 = [str(x9) for x9 in want_ids9]
    if not fp9.is_file():
        return None, [], list(want_ids9)
    try:
        with np.load(fp9, allow_pickle=True) as z9:
            if str(z9["ver"]) != GV_WAVE_VER:
                return None, [], list(want_ids9)
            ids9 = [str(v9) for v9 in z9["ids"]]
            fks9 = [str(v9) for v9 in z9["fk"]]
            sens9 = [str(v9) for v9 in z9["sens"]]
            pos9 = {w9: i9 for i9, w9 in enumerate(ids9)}
            hit9, miss9 = [], []
            for w9 in want_ids9:
                i9 = pos9.get(w9)
                if i9 is None or (fk_of9 is not None and fk_of9.get(w9) != fks9[i9]):
                    miss9.append(w9)
                else:
                    hit9.append(w9)
            if not hit9:
                return None, [], miss9
            rows9 = [pos9[w9] for w9 in hit9]
            Wz9 = z9["W"]
            L9 = int(Wz9.shape[2])
            out9 = np.full((len(rows9), len(want_sens9), L9), np.nan, dtype=np.float32)
            for jj9, s9 in enumerate(want_sens9):
                if s9 in sens9:
                    out9[:, jj9, :] = Wz9[rows9, sens9.index(s9), :]
            return out9, hit9, miss9
    except Exception:                                  # noqa: BLE001
        return None, [], list(want_ids9)


def _gv_day_core(S9, day9, items9, calc_sens9, w9, goldens9, gref9, wave_root9=None,
                 ref_sens9=None, fk_of9=None):
    """하루치 판정 계산(직렬·병렬 공통): items9=[(wid, file, stem, meta)] 를 sorted(wid) 순으로.
    반환 {A, LR, F, ids, gkey, miss, n}. wave_root9 가 있으면 캐논별 파형 npz 도 기록(판정 센서 ∪ 참조)."""
    miss9 = {}
    by_f9 = {}
    back9 = {}
    meta9 = {}
    for wid9, f9, st9, m9 in items9:
        if not f9:
            continue
        by_f9.setdefault(f9, []).append(st9)
        back9.setdefault(f9, {})[str(st9)] = wid9
        meta9[wid9] = m9
    ref9 = [s9 for s9 in (ref_sens9 or []) if s9 not in calc_sens9]
    load_sens9 = list(calc_sens9) + ref9
    jobs9 = []
    for f9, sts9 in by_f9.items():
        for c09 in range(0, len(sts9), 100):
            jobs9.append((f9, sts9[c09:c09 + 100]))
    rowA9, rowL9, rowF9, rid9, rgk9 = [], [], [], [], []
    wave9 = {}                                         # canon_sig → {ids, fks, W rows, canon, L}
    for f9, sts9 in jobs9:
        recs9 = S9._load_aligned(Path(f9), stems=sts9)[0]
        try:
            _augment_virtual(recs9)
        except Exception:                              # noqa: BLE001
            pass
        for rec9 in recs9:
            stem9 = str(rec9.get("wafer_id") or "")
            wid9 = back9[f9].get(stem9, stem9)
            m9 = meta9.get(wid9)
            if m9 is None:
                continue
            g9, _ge9 = _gv_pick_golden_core(m9, goldens9, gref9)
            if g9 is None:
                continue
            arr9, stw9 = _gv_sensor_mat_core(rec9, load_sens9, miss9)
            if arr9 is None:
                continue
            if not np.isfinite(arr9[:, :len(calc_sens9)]).any():
                continue
            canon9 = [(u9, p9, l9) for u9, p9, l9 in g9["canon"]]
            L9 = int(g9["L"])
            wv9 = _warp_to_canon(arr9, stw9, canon9, L9)
            A9 = _gv_metrics_core(wv9[:, :len(calc_sens9)], g9, calc_sens9, int(w9 or 0))
            LR9 = _gv_lenratio(stw9, g9)
            F9 = _gv_feats_one(wv9[:, :len(calc_sens9)], canon9, raw9=arr9[:, :len(calc_sens9)], stw9=stw9)
            rowA9.append(A9); rowL9.append(LR9); rowF9.append(F9)
            rid9.append(wid9)
            rgk9.append(f"{g9['eqp']}|{g9['recipe']}|{g9['dsv'] or ''}")
            if wave_root9:
                cs9 = _gv_canon_sig(canon9, L9)
                e9 = wave9.setdefault(cs9, {"ids": [], "fks": [], "W": [], "canon": canon9, "L": L9})
                e9["ids"].append(wid9)
                e9["fks"].append((fk_of9 or {}).get(wid9, ""))
                e9["W"].append(wv9.T.astype(np.float32))   # (S, L)
        del recs9
    if wave_root9:
        for cs9, e9 in wave9.items():
            _gv_wave_save(Path(wave_root9) / cs9, day9, e9["ids"], e9["fks"], load_sens9,
                          np.stack(e9["W"]) if e9["W"] else np.zeros((0, len(load_sens9), e9["L"]), np.float32),
                          e9["canon"], e9["L"])
    for s9 in ref9:                                    # 참조 센서 결측은 판정 통계에 넣지 않는다
        miss9.pop(s9, None)
    nF9 = rowF9[0].shape[1] if rowF9 else 0
    return {"A": (np.stack(rowA9) if rowA9 else np.zeros((0, len(calc_sens9), 0, len(GV_KGRID)))),
            "LR": (np.stack(rowL9) if rowL9 else np.zeros((0, 0))),
            "F": (np.stack(rowF9) if rowF9 else np.zeros((0, len(calc_sens9), nF9), np.float32)),
            "ids": rid9, "gkey": rgk9, "miss": miss9, "n": len(rid9)}


_GVW = {}


def _gvw_init(root9, model_dir9, cache9, goldens9, gref9):
    """골든 비교 병렬 워커 부트스트랩 — 모델(ONNX 세션·GPU)은 올리지 않는다.
    로더가 모델에서 쓰는 것은 IO/센서 열 순서뿐이라 recon_meta.npz 만 읽어 가벼운 객체로 대신한다
    (메타가 없을 때만 전체 로드로 폴백). 목록 캐시 설치, 로더 스레드 1(과구독 방지)."""
    import contextlib
    import io as _io
    from types import SimpleNamespace
    try:
        os.nice(10)
    except Exception:                                  # noqa: BLE001
        pass
    os.environ["IO_LOAD_WORKERS"] = "1"
    os.environ.setdefault("IO_ORT_PROVIDER", "cpu")    # 폴백 로드가 일어나도 GPU 는 건드리지 않는다
    with contextlib.redirect_stdout(_io.StringIO()):
        if root9 and root9 not in sys.path:
            sys.path.insert(0, root9)
        from io_recon_tcn.app import server as S9
        _install_virtual(S9)
        _install_parse_cache(S9, store_hint=cache9)
        try:
            _install_listing_cache(S9)
        except Exception:                              # noqa: BLE001
            pass
        E9 = S9.ENGINE
        if model_dir9:
            E9.select(model_dir9)
        ok9 = False
        try:
            art9 = Path(model_dir9) if model_dir9 else S9._resolve_art()
            meta9 = None
            for cand9 in (art9 / "recon_meta.npz", *sorted(art9.glob("*.npz"))):
                if cand9.is_file():
                    with np.load(cand9, allow_pickle=True) as z9:
                        if "io_cols" in z9 and "input_sensor_cols" in z9:
                            meta9 = {k9: [str(c9) for c9 in z9[k9].tolist()]
                                     for k9 in ("io_cols", "input_sensor_cols", "sensor_cols") if k9 in z9}
                            break
            if meta9:
                E9.art = art9
                E9.cfg = E9._build_cfg()
                E9.model = SimpleNamespace(io_cols=meta9["io_cols"],
                                           input_sensor_cols=meta9["input_sensor_cols"],
                                           sensor_cols=meta9.get("sensor_cols", meta9["input_sensor_cols"]))
                ok9 = True
        except Exception:                              # noqa: BLE001
            ok9 = False
        if not ok9:
            E9.load()                                  # 폴백: 전체 로드(CPU)
        _PTW["S"] = S9
    _GVW.update(S=S9, goldens=goldens9, gref=gref9)


def _gvw_day(args9):
    day9, items9, calc_sens9, w9, wave_root9, ref_sens9, fk_of9 = args9
    out9 = _gv_day_core(_GVW["S"], day9, items9, calc_sens9, w9, _GVW["goldens"], _GVW["gref"],
                        wave_root9, ref_sens9, fk_of9)
    return day9, out9


def _shape_steparea(y, st, steps):
    """아치 규칙 v145(사용자 지정): 지정 스텝 '각각'의 첫점↔끝점을 잇는 선분과
    그래프 사이 |면적|(위아래 무관, 평균 |편차| — 원단위)을 구해 합한다.

    예) 스텝 3,4,5 → 스텝3 첫점-끝점 선분 vs 그래프 면적 + 스텝4 … + 스텝5 ….
    정상 아치는 선분에서 불룩해 면적이 크고, 포화/평탄은 면적이 작다.
    반환 {"areas": 스텝별, "total": 합, "segs": [(s0,e0,y0,y1)…]} 또는 None.
    """
    st = np.asarray(st, dtype=int)
    y = np.asarray(y, dtype=np.float64)
    areas, segs = [], []
    for sp in steps:
        m = st == int(sp)
        if m.sum() < 8:
            continue
        i = np.where(m)[0]
        seg = y[i[0]:i[-1] + 1]
        if np.isfinite(seg).sum() < 8:
            continue
        k = min(5, max(1, len(seg) // 10))
        y0 = float(np.nanmedian(seg[:k]))
        y1 = float(np.nanmedian(seg[-k:]))
        ch = np.linspace(y0, y1, len(seg))
        with np.errstate(all="ignore"):
            area = float(np.nanmean(np.abs(seg - ch)))
        areas.append(area)
        segs.append((int(i[0]), int(i[-1]), y0, y1))
    if not areas:
        return None
    return {"areas": [round(a, 4) for a in areas],
            "total": float(sum(areas)), "segs": segs}


def _shape_convexity(y):
    """패턴 인식 P1 — 현(첫값↔끝값 선분) 대비 볼록비.

    지수 감쇠는 현 아래로 볼록해 면적이 크고, 직선 하강은 0.
    · 끝점은 앞/뒤 5점 중앙값(스파이크 1점에 현이 흔들리지 않게)
    · ratio = 2·평균편차/|낙폭| — 스텝 길이·낙폭에 무관한 무차원 값(0=직선)
    · early = 앞 20% 구간 낙폭 비중(지수면 큼) — 혼합 케이스(전반 지수+후반 직선) 분리용
    반환: {ratio, drop, early, area} 또는 None(구간이 너무 짧음)
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 12:
        return None
    k = min(5, max(1, n // 10))
    y0 = float(np.median(y[:k]))
    y1 = float(np.median(y[-k:]))
    chord = np.linspace(y0, y1, n)
    area = float(np.mean(chord - y))                   # 곡선이 현 아래(볼록)면 양수
    drop = y0 - y1
    ratio = float(2.0 * area / abs(drop)) if abs(drop) > 1e-9 else 0.0
    i20 = max(1, int(round(n * 0.2)))
    early = float((y0 - float(np.median(y[max(0, i20 - 2):i20 + 3]))) / drop) \
        if abs(drop) > 1e-9 else 0.0
    return {"ratio": ratio, "drop": drop, "early": early, "area": area}


def _shape_arches(y, polarity=0, prom_min=0.0):
    """패턴 인식 P2 — 극값(prominence) 기반 아치 검출.

    변곡점(2차 미분) 대신 극값을 쓴다: 잡음 미분 증폭이 없어 재현성이 있다.
    · polarity: +1=위로 볼록(∩) 아치, -1=아래로 볼록(∪), 0=자동(우세 극성)
    · prom_min: 극값으로 인정할 최소 돌출 높이(골든 아치 깊이·잡음 상대로 밖에서 산출)
    · 아치 1개 = 받침점→정점→받침점, 면적 = 받침점끼리 이은 현 대비 |적분|
    반환: {n_arch, total_area(정규화 합), polarity, peaks, feet, areas} 또는 None
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 16:
        return None
    w = max(3, n // 100)
    ker = np.ones(w)
    # 가장자리 보정: 0-패딩 평균은 양끝이 0 쪽으로 눌려 가짜 극값을 만든다
    ys = np.convolve(y, ker, mode="same") / np.convolve(np.ones(n), ker, mode="same")
    if prom_min <= 0:                                  # 폴백: 잡음 σ 추정 × 4
        prom_min = 4.0 * float(np.median(np.abs(np.diff(y)))) + 1e-12

    def _extrema(sign):
        """sign=+1 이면 로컬 최대(정점) 후보, -1 이면 로컬 최소."""
        v = ys * sign
        idx = []
        i = 1
        while i < n - 1:
            if v[i] >= v[i - 1] and v[i] >= v[i + 1]:
                j = i
                while j + 1 < n - 1 and v[j + 1] == v[i]:
                    j += 1                             # 평평한 꼭대기는 묶음
                idx.append((i + j) // 2)
                i = j + 1
            else:
                i += 1
        # prominence: 좌우로 더 높은 극값을 만나기 전 구간의 최저 대비 돌출
        out = []
        for p in idx:
            lo_l = v[:p + 1]
            lo_r = v[p:]
            hl = np.where(lo_l > v[p])[0]
            hr = np.where(lo_r > v[p])[0]
            lmin = float(np.min(lo_l[hl[-1]:] if len(hl) else lo_l))
            rmin = float(np.min(lo_r[:hr[0] + 1] if len(hr) else lo_r))
            prom = v[p] - max(lmin, rmin)
            if prom >= prom_min:
                out.append((p, float(prom)))
        return out

    if polarity == 0:                                  # 우세 극성 자동(골든 학습용)
        up = _extrema(+1)
        dn = _extrema(-1)
        su = sum(pr for _, pr in up)
        sd = sum(pr for _, pr in dn)
        polarity = 1 if su >= sd else -1
    sign = 1 if polarity > 0 else -1
    peaks = _extrema(sign)
    if not peaks:
        return {"n_arch": 0, "total_area": 0.0, "polarity": polarity,
                "peaks": [], "feet": [], "areas": [], "prom_ref": 0.0,
                "prom_min": float(prom_min)}
    v = ys * sign
    feet = []                                          # 정점 사이/양끝의 받침점(최저점)
    ppos = [p for p, _ in peaks]
    bounds = [0] + ppos + [n - 1]
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        if b1 <= b0:
            feet.append(b0)
            continue
        feet.append(int(b0 + np.argmin(v[b0:b1 + 1])))
    areas = []
    for q, p in enumerate(ppos):
        f0, f1 = feet[q], feet[q + 1]
        if f1 - f0 < 3:
            areas.append(0.0)
            continue
        seg = y[f0:f1 + 1] * sign
        ch = np.linspace(seg[0], seg[-1], len(seg))
        # 창 전체 길이로 나눔 — 아치가 창의 일부만 덮으면(포화 후 사수평) 그만큼 감점
        areas.append(float(np.sum(np.maximum(0.0, seg - ch)) / n))
    prom_ref = max(pr for _, pr in peaks)
    total = float(sum(areas) / prom_ref) if prom_ref > 1e-12 else 0.0
    return {"n_arch": len(ppos), "total_area": total, "polarity": polarity,
            "peaks": ppos, "feet": feet, "areas": areas,
            "prom_ref": float(prom_ref), "prom_min": float(prom_min)}


def _shape_wbulge(y):
    """패턴 인식 W패턴 — 최저점→끝점 선분 위로 불룩한 면적(크면 불량, 방향 반대).

    정상: 최저점 이후 끝까지 점진 상승(곡선이 선분 근처/아래).
    불량: 중간에 불룩하게 증가(W 모양) → 선분 위 면적이 커진다.
    ratio = 2·mean(max(0, y−chord)) / |상승폭| — 무차원(0=단조 상승).
    반환: {ratio, rise, imin} 또는 None(구간 짧음/최저점이 끝에 붙음=상승 없음).
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 12:
        return None
    w = max(3, n // 100)
    ker = np.ones(w)
    ys = np.convolve(y, ker, mode="same") / np.convolve(np.ones(n), ker, mode="same")
    imin = int(np.argmin(ys))
    if imin >= n - 8:                                  # 끝에서 최저 = 상승 구간 없음
        return None
    k = min(5, max(1, (n - imin) // 10))
    y0 = float(np.median(y[max(0, imin - 2):imin + 3]))
    y1 = float(np.median(y[-k:]))
    rise = y1 - y0
    seg = y[imin:]
    chord = np.linspace(y0, y1, len(seg))
    bulge = float(np.mean(np.maximum(0.0, seg - chord)))
    ratio = float(2.0 * bulge / abs(rise)) if abs(rise) > 1e-9 else 0.0
    return {"ratio": ratio, "rise": rise, "imin": imin, "bulge": bulge}


def _install_parse_cache(S, store_hint=None):
    """★모든 탭 공통 가속 — .log→배열 파싱 결과(npz)를 재사용.

    배포 검사앱은 항상성 우선으로 파싱 캐시를 강제 OFF(_apply_cache_env 가 env 를
    지움). 이 도구는 분석용이라 다르게 간다: 수치가 동일함이 보장되고(mtime/size
    변경 시 자동 무효) 시작 집계·웨이퍼 분석·골든 생성·시각화·비교·패턴 스캔 등
    모든 로드 경로가 파일당 파싱 비용(전체의 절반가량)을 건너뛴다.
    폴더 = 사용자 '캐시 폴더' 설정(avgwave_state.json sd.cache) > 앱 data 폴더.
    """
    try:
        st9 = {}
        try:
            fp9 = Path(__file__).resolve().parent / "avgwave_state.json"
            st9 = json.loads(fp9.read_text(encoding="utf-8"))
        except Exception:                              # noqa: BLE001
            pass
        c9 = str(((st9.get("sd") or {}).get("cache")) or "").strip()
        root9 = Path(c9) if c9 else Path(store_hint or S.DATA_DIR)
        d9 = root9 / "parse_cache"
        d9.mkdir(parents=True, exist_ok=True)

        def _apply9():
            os.environ["IO_PARSE_CACHE_DIR"] = str(d9)
        S._apply_cache_env = _apply9                   # 검사앱 강제 OFF 를 이 도구에서만 우회
        _apply9()
        print(f"[cache] 파싱 캐시 켜짐(모든 탭 공통): {d9}", flush=True)
    except Exception as e9:                            # noqa: BLE001
        print(f"[cache] 파싱 캐시를 켜지 못했습니다(캐시 없이 진행): {e9}", flush=True)


def _install_meta_cache(S, root):
    """★레시피 태그·끝시각(json)을 앱 폴더가 아니라 <캐시 or 저장소>/meta_cache 로.

    지금까지는 S.DATA_DIR(앱 폴더 안)에 저장돼 버전 교체(zip 새로 풀기)마다 사라져
    '웨이퍼 목록 불러오기'가 30만 장을 매번 처음부터 태깅했다(사용자 지적).
    저장소로 옮기면 점수 캐시처럼 앱과 무관하게 살아남는다. 첫 호출 시 앱 폴더에
    이미 태깅된 json 이 있으면 이관(재태깅 방지).
    """
    try:
        d9 = Path(root) / "meta_cache"
        d9.mkdir(parents=True, exist_ok=True)
        cur_rt = None
        try:
            cur_rt = S._recipe_tags_file()
        except Exception:                              # noqa: BLE001
            pass
        new_rt, new_fm = d9 / "recipe_tags.json", d9 / "file_meta.json"
        if str(cur_rt) == str(new_rt):                 # 이미 적용됨
            return
        # 이관: 새 위치에 없고 옛 위치에 있으면 복사(현 세션 태깅 결과 보존)
        for old9, new9 in ((cur_rt, new_rt),
                           (S._file_meta_path() if cur_rt else None, new_fm)):
            try:
                if old9 and Path(old9).is_file() and not new9.is_file():
                    shutil.copy2(old9, new9)
            except Exception:                          # noqa: BLE001
                pass
        S._recipe_tags_file = lambda: new_rt
        S._file_meta_path = lambda: new_fm
        try:
            S._load_recipe_tags()
            S._load_file_meta()
        except Exception:                              # noqa: BLE001
            pass
        n9 = len(getattr(S, "_RECIPE_TAGS", {}) or {})
        print(f"[cache] 메타 캐시(태그·끝시각) 위치: {d9} — 기존 태그 {n9}건 로드",
              flush=True)
    except Exception as e9:                            # noqa: BLE001
        print(f"[cache] 메타 캐시 재지정 실패(앱 폴더 사용): {e9}", flush=True)


def _install_virtual(S):
    """S._load_aligned 한 곳을 감싸 모든 로드 경로(분석·골든·크로스·추세 스캔)에
    가상센서를 자동 주입 — 파이프라인 전체가 실센서와 동급으로 취급하게 된다."""
    if getattr(S._load_aligned, "_virt", False):
        return
    orig = S._load_aligned

    def _wrapped(folder, stems=None):
        recs, io_cols, sensor_cols, cov = orig(folder, stems=stems)
        _augment_virtual(recs)
        try:
            have = {str(c).lower() for c in sensor_cols}
            for nm, pa, pb in _VIRTUAL_DEFS:
                if nm.lower() not in have and pa.lower() in have and pb.lower() in have:
                    sensor_cols.append(nm)
        except Exception:                              # noqa: BLE001
            pass
        return recs, io_cols, sensor_cols, cov
    _wrapped._virt = True
    S._load_aligned = _wrapped


def _is_ctrl_sensor(sn):
    """판정 제외 센서: Vessel 추종형(Cart/Heater — SV 미상·목표 추종으로 상시 변동)과
    H 라인 히터(h_005 — 자체 설정, 핵심 아님). 곡선은 보관하되 패턴 판정은 안 한다.
    (사용자 결정: 히터류는 제어 여부를 알 수 없어 패턴 분석 불가)"""
    import re as _re2
    return bool(_re2.search(r"(?i)(vessel(up|down)?(cart|heat))|h_005", str(sn)))


def _erode_mask(m, e=4):
    """스텝 마스크의 양끝 e샘플 제거 — 스텝 경계 불연속(계단)과 웨이퍼 길이 편차가
    만드는 리샘플 정렬 오차(±수십 σ 가짜 스파이크)를 판정에서 제외한다.
    (완만-W 저진폭 감사에서 발견: 경계 오차가 전 센서 가짜 카드를 유발)"""
    m2 = m.copy()
    idx = np.where(m)[0]
    if idx.size == 0:
        return m2
    # 연속 구간별 양끝 침식
    brk = np.where(np.diff(idx) > 1)[0]
    starts = [idx[0]] + [idx[b + 1] for b in brk]
    ends = [idx[b] for b in brk] + [idx[-1]]
    for a9, b9 in zip(starts, ends):
        m2[a9:min(a9 + e, b9 + 1)] = False
        m2[max(b9 - e + 1, a9):b9 + 1] = False
    return m2


def _step_trend(w, mu, sg, steps_rs, usteps):
    """웨이퍼 1장의 센서 1개: 스텝별로 골든 대비 트렌드를 4유형 판정.

    레벨(트렌드 같고 값 오프셋)·기울기상이·출렁임(평탄 구간 진동)·모양상이.
    시간축은 이미 골든 길이로 리샘플된 상태(길이 상이 문제 해소).
    반환: [{step, kinds:[..], level, slope, rough, corr}] — 이상 있는 스텝만.
    """
    out = []
    base_floor = max(1e-6, 0.05 * float(np.median(sg)), 1e-3 * float(np.mean(np.abs(mu))))
    # 1차 패스: 스텝별 잔차 통계 수집 — 출렁임은 '그 센서의 다른 스텝' 대비 자기정규화.
    #   (골든 평균은 평활해 절대 기준이 없고, 센서마다 노이즈 규모가 달라 σ 기준도 불안정)
    seg = {}
    for st in usteps:
        m = _erode_mask(steps_rs == st)
        if int(m.sum()) < 6:
            continue
        resid = w[m] - mu[m]
        seg[st] = (w[m], mu[m], float(np.std(np.diff(resid))))
    if not seg:
        return out
    wigs = [v[2] for v in seg.values()]
    # 기준선: 하위 25분위 — 스텝이 2~3개뿐일 때 이상 스텝이 중앙값을 오염시키는 것을 방지.
    wig_base = max(float(np.percentile(wigs, 25)), 1e-9)
    for st, (ws, gs, wig_r) in seg.items():
        n = len(ws)
        sc = max(base_floor, float(np.median(sg[_erode_mask(steps_rs == st)]))
                 if int(_erode_mask(steps_rs == st).sum()) else base_floor)
        level = float(np.mean(ws - gs)) / sc
        x = np.arange(n, dtype=float)
        slw = float(np.polyfit(x, ws, 1)[0])
        slg = float(np.polyfit(x, gs, 1)[0])
        slope = (slw - slg) * n / sc                  # 스텝 전체 구간 기준 기울기 차(σ)
        rough = wig_r / wig_base                      # 잔차 출렁임: 자기 스텝들의 중앙값 대비
        vw, vg = float(np.std(ws)), float(np.std(gs))
        if vw > 1e-9 and vg > 1e-9:
            corr = float(np.corrcoef(ws, gs)[0, 1])
        else:
            corr = 1.0
        kinds = []
        if abs(level) >= 2.0:
            kinds.append("레벨높음" if level > 0 else "레벨낮음")
        if abs(slope) >= 2.0:
            kinds.append("기울기상이")
        if rough >= 2.0 and wig_r > 0.3 * sc:
            kinds.append("출렁임")
        if corr < 0.5 and vg > 0.5 * sc:
            kinds.append("모양상이")
        if kinds:
            out.append({"step": int(st), "kinds": kinds, "level": round(level, 1),
                        "slope": round(slope, 1), "rough": round(rough, 1),
                        "corr": round(corr, 2)})
    out.sort(key=lambda o: o["step"])
    return out


def _bump_feat(resid, sc):
    """스텝 내 잔차의 최대 국소 로브(볼록): (부호 진폭 A/σ, 중심 idx, 반치폭). 없으면 None.

    이동평균 5tp 평활 후 |최대| 지점을 중심으로 반치폭을 잰다 — '특정 구간이 점점
    볼록해지는' 패턴(R5)의 수치화. 전체 레벨 오프셋과 구분하기 위해 잔차에서
    스텝 중앙값을 뺀 국소 성분으로 판정한다.
    """
    n = len(resid)
    if n < 10:
        return None
    # 기준선 = 스텝 '양끝' 레벨 — 볼록의 정의는 "진입/이탈 레벨은 기준과 같은데
    #   중간이 부푼다"이다. 중앙값·최빈값 기준선은 로브가 넓으면(완만한 W 등)
    #   로브 바닥으로 끌려가 평탄부를 가짜 +로브로 만든다(완만-W 감사에서 규명).
    #   스텝 전체가 이동한 경우는 양끝도 함께 움직여 loc≈0 → 레벨 피처가 담당(정확).
    k = max(3, n // 10)
    base_lv = float(np.median(np.concatenate([resid[:k], resid[-k:]])))
    loc = resid - base_lv                             # 국소 성분(양끝 기준 제거)
    k = 5
    sm = np.convolve(loc, np.ones(k) / k, mode="same")
    c = int(np.argmax(np.abs(sm)))
    A = float(sm[c]) / sc
    if abs(A) < 1e-6:
        return None
    half = abs(sm[c]) / 2.0
    a = c
    while a > 0 and abs(sm[a]) > half:
        a -= 1
    b2 = c
    while b2 < n - 1 and abs(sm[b2]) > half:
        b2 += 1
    return {"A": round(A, 2), "c": c, "w": max(3, b2 - a)}


def _spearman(vals):
    idx = [i for i, v in enumerate(vals) if v is not None]
    if len(idx) < 4:
        return 0.0
    vv = [vals[i] for i in idx]

    def _rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k2, i in enumerate(o):
            r[i] = k2
        return r
    ri, rv = _rank(idx), _rank(vv)
    n2 = len(vv)
    mi = sum(ri) / n2; mv = sum(rv) / n2
    cov = sum((a - mi) * (b - mv) for a, b in zip(ri, rv))
    den = (sum((a - mi) ** 2 for a in ri) * sum((b - mv) ** 2 for b in rv)) ** 0.5
    return cov / den if den > 1e-9 else 0.0


def trendscan_cards(feat, dates):
    """일별 피처 시계열 → 패턴 카드(R3·R5·R6).

    feat: {(sensor, step): {date: {"level":σ, "bump":{A,c,w}|None, "ratio":0~1}}}
    판정: 드리프트(level 스피어만≥0.8·진폭≥1.5σ) / 볼록(같은 위치 연속≥3일,
    증가=심화·평탄=유지) / 개입 이벤트(하루 급변 ≥3σ — 소멸/급증).
    """
    cards = []
    for (sn, st), by_d in feat.items():
        lv = [by_d[d]["level"] if d in by_d else None for d in dates]
        bp = [by_d[d].get("bump") if d in by_d else None for d in dates]
        rt = [by_d[d].get("ratio") if d in by_d else None for d in dates]
        sh = [by_d[d].get("shape") if d in by_d else None for d in dates]
        def _jumps(series, kind, up_label, down_label):
            """급변 = |Δ|>=3σ '이면서' 평소 변화율(중앙값)의 2배 이상 —
            가파른 '점진' 드리프트가 매일 이벤트를 찍는 스팸을 막는다(급작성 요건)."""
            ds9 = []
            prev9 = None
            for i9, v9 in enumerate(series):
                if v9 is not None and prev9 is not None:
                    ds9.append((i9, v9 - prev9))
                if v9 is not None:
                    prev9 = v9
            if not ds9:
                return []
            med9 = sorted(abs(d9v) for _i9, d9v in ds9)[len(ds9) // 2]
            out9 = []
            for i9, d9v in ds9:
                if abs(d9v) >= 3.0 and abs(d9v) >= 2.0 * max(med9, 0.5):
                    a_prev = series[i9 - 1] if i9 > 0 and series[i9 - 1] is not None else 0.0
                    a_cur = series[i9]
                    out9.append({"date": dates[i9], "kind": kind,
                                 "delta": round(d9v, 1),
                                 "label": up_label if abs(a_cur) > abs(a_prev) else down_label})
            return out9
        events = _jumps(lv, "레벨 급변", "악화 의심", "개입 추정(정상화)")
        events += _jumps([b3["A"] if b3 else 0.0 for b3 in bp],
                         "볼록 급변", "급증(악화)", "소멸(개입 추정)")
        events.sort(key=lambda e9: e9["date"])
        # 드리프트(레벨 누적 추세) — ★슬라이딩 윈도 국소 탐지:
        #   8개월 스캔에서 '7월 열흘짜리' 드리프트는 전기간 단조성으론 상관≈0 이라
        #   놓친다(실전 규모 시뮬레이션에서 발견). 14일 창을 밀며 창 안에서
        #   단조·진폭 조건을 보고, 겹치는 창을 구간으로 합쳐 카드화한다.
        idxs = [i for i, v in enumerate(lv) if v is not None]
        W = 14
        marked = set()
        for a0 in range(len(idxs) - 3):
            win = idxs[a0:a0 + W]
            if len(win) < 4:
                continue
            wv = [lv[i] for i in win]
            rho = _spearman(wv)
            amp_w = wv[-1] - wv[0]
            if abs(rho) >= 0.8 and abs(amp_w) >= 1.5:
                marked.update(win)
        segs = []
        for i in sorted(marked):
            if segs and i - segs[-1][-1] <= 2:
                segs[-1].append(i)
            else:
                segs.append([i])
        drift_cover = set()
        for seg in segs:
            if len(seg) < 4:
                continue
            # 앞부분 다듬기: 윈도 마킹이 정상 구간까지 물고 들어오므로,
            #   시작을 '처음 유의하게 벌어진 날'로 당긴다(기준=구간 첫 값)
            b0 = lv[seg[0]]
            k0 = next((k for k, i in enumerate(seg)
                       if abs(lv[i] - b0) >= 1.0 or abs(lv[i]) >= 1.5), 0)
            seg = seg[max(0, k0 - 1):]
            if len(seg) < 4:
                continue
            sv = [lv[i] for i in seg]
            peak = max(sv, key=abs)
            ended = seg[-1] < idxs[-1] - 1
            after = [lv[i] for i in idxs if i > seg[-1]][:5]
            returned = ended and after and abs(sum(after) / len(after)) < 1.0
            state = ("복귀(소멸)" if returned
                     else ("심화" if abs(sv[-1]) >= abs(sv[0]) + 1.0 else "유지"))
            drift_cover.update(dates[i] for i in seg)
            # 구간 안의 '레벨 급변(악화)'은 드리프트 자체가 설명 — 이벤트에서 제외
            #   (복귀·정상화 급변은 남긴다). 240일 조용한 시계열에선 중앙값 요건이
            #   무력해져 스팸이 재발했음(실운영 스크린샷에서 발견).
            ev_cd = [e for e in events
                     if not (e["kind"] == "레벨 급변" and "악화" in e["label"]
                             and dates[seg[0]] <= e["date"] <= dates[seg[-1]])]
            i_pk = max(seg, key=lambda i9x: abs(lv[i9x]) if lv[i9x] is not None else 0)
            cards.append({"sensor": sn, "step": st, "kind": "레벨 드리프트",
                          "direction": "상승" if peak > 0 else "하강",
                          "shape": sh[i_pk],
                          "start": dates[seg[0]], "end": dates[seg[-1]],
                          "series": [None if v is None else round(v, 1) for v in lv],
                          "state": state, "amp": round(peak, 1), "events": ev_cd,
                          "ratio_series": rt})
        # 볼록 형성/유지/심화 — ★모든 유효 구간을 각각 카드화(최대 3개, 진폭 순):
        #   '최고 구간 하나만' 방식은 8월 재발 볼록을 버렸다(실전 시뮬레이션에서 발견).
        runs = []
        run = []                                       # [(i, A, c, w)]
        for i, b3 in enumerate(bp):
            # 히스테리시스: 형성 1.5σ, 유지 1.0σ — 경계 진폭 사건이 문턱 주위를
            #   오르내리며 run 이 3일을 못 채우는 실종 방지(완만-W 감사에서 규명)
            thr9 = 1.0 if run else 1.5
            if b3 is None or abs(b3["A"]) < thr9:
                if len(run) >= 3:
                    runs.append(run)
                run = []
                continue
            if run and abs(b3["c"] - run[-1][2]) > max(run[-1][3], 8):
                # ★ 다봉 패턴(W 등)은 최대 로브가 다리 사이를 오가며 중심이 점프한다 —
                #   일간 '모양'이 같으면(코사인≥0.85) 같은 패턴으로 보고 run 을 잇는다
                #   (완만-W 감사: 중심 조건만으로는 5월 발생이 통째로 끊겨 실종됐음)
                cs9 = _shape_cos(sh[i], sh[run[-1][0]])
                if cs9 is None or cs9 < 0.75:
                    if len(run) >= 3:
                        runs.append(run)
                    run = []
            run.append((i, b3["A"], b3["c"], b3["w"]))
        if len(run) >= 3:
            runs.append(run)
        # 같은 위치·간격 2일 이내로 끊긴 run 병합(노이즈로 하루 빠져 쪼개지는 것 방지)
        runs.sort(key=lambda r9: r9[0][0])
        merged = []
        for r9 in runs:
            sim9 = _shape_cos(sh[r9[0][0]], sh[merged[-1][-1][0]]) if merged else None
            if (merged and r9[0][0] - merged[-1][-1][0] <= 2
                    and (abs(r9[0][2] - merged[-1][-1][2]) <= max(merged[-1][-1][3], 8)
                         or (sim9 is not None and sim9 >= 0.75))):
                merged[-1] = merged[-1] + r9
            else:
                merged.append(r9)
        runs = sorted(merged, key=lambda r9: -abs(r9[-1][1]))[:6]   # 상한 6 — 재발 다회 보존
        runs.sort(key=lambda r9: r9[0][0])
        prev_run = None
        for best in runs:
            amps = [x[1] for x in best]
            end_i = best[-1][0]
            vanish = any(e["kind"] == "볼록 급변" and "소멸" in e["label"]
                         and dates[min(end_i + 3, len(dates) - 1)] >= e["date"] > dates[end_i - 1]
                         for e in events)
            state = ("심화" if abs(amps[-1]) >= abs(amps[0]) + 1.0
                     else ("소멸" if vanish else "유지"))
            ser = [None] * len(dates)
            for i, A2, _c2, _w2 in best:
                ser[i] = round(A2, 1)
            i_pk = max(best, key=lambda x9: abs(x9[1]))[0]
            cd9 = {"sensor": sn, "step": st, "kind": "볼록 패턴",
                   "direction": "위로" if amps[-1] > 0 else "아래로",
                   "shape": sh[i_pk],
                   "start": dates[best[0][0]], "end": dates[end_i],
                   "series": ser, "state": state,
                   "amp": round(amps[-1], 1),
                   "loc": {"c": best[-1][2], "w": best[-1][3]},
                   "events": events, "ratio_series": rt}
            # 같은 스캔 안의 앞선 동일 위치 구간 = 스캔 내 재발
            if prev_run is not None and abs(best[-1][2] - prev_run[-1][2]) <= max(prev_run[-1][3], 8):
                cd9["recur"] = [{"period": f"{dates[prev_run[0][0]]}~{dates[prev_run[-1][0]]}",
                                 "state": "동일 스캔 내 재발"}]
            prev_run = best
            cards.append(cd9)
        # 카드 없는 센서라도 개입 이벤트가 크면 단독 카드(드리프트 구간 내 악화 급변 제외)
        events = [e for e in events
                  if not (e["kind"] == "레벨 급변" and "악화" in e["label"]
                          and e["date"] in drift_cover)]
        if events and not any(c4["sensor"] == sn and c4["step"] == st for c4 in cards):
            cards.append({"sensor": sn, "step": st, "kind": "급변 이벤트",
                          "direction": "-", "start": events[0]["date"],
                          "series": [None if v is None else round(v, 1) for v in lv],
                          "state": "-", "amp": None, "events": events,
                          "ratio_series": rt})
    cards.sort(key=lambda c4: (c4["start"], -(abs(c4["amp"]) if c4["amp"] else 0)))
    return cards


def _shape_cos(a, b):
    if not a or not b or len(a) != len(b):
        return None
    num = sum(x * y for x, y in zip(a, b))
    return num                                         # 입력이 L2 정규화라 내적=코사인


def pattern_groups(cards):
    """카드들을 '같은 문제'로 페어링 — 사례 통계(P2·P3).

    기준: 같은 (센서, 스텝, 유형, 방향) + 모양 코사인 ≥0.85 (모양 없으면
    볼록 위치 ±8% 폴백). 발생 간격 무제한(1월↔8월도 묶임).
    """
    groups9 = []
    for ci, cd in enumerate(cards):
        placed = False
        for g9 in groups9:
            r0 = cards[g9["members"][0]]
            if (cd["sensor"], cd["step"], cd["kind"], cd.get("direction")) !=                (r0["sensor"], r0["step"], r0["kind"], r0.get("direction")):
                continue
            cs = _shape_cos(cd.get("shape"), r0.get("shape"))
            if cs is not None:
                if cs < 0.75:
                    continue
            else:                                      # 모양 없음 → 위치 폴백
                la, lb = cd.get("loc"), r0.get("loc")
                if la and lb and abs(la["c"] - lb["c"]) > max(lb["w"], 8):
                    continue
            g9["members"].append(ci)
            placed = True
            break
        if not placed:
            groups9.append({"members": [ci]})
    out = []
    for g9 in groups9:
        ms = [cards[i] for i in g9["members"]]
        occ = [{"start": c9["start"], "end": c9.get("end", c9["start"]),
                "state": c9["state"], "amp": c9.get("amp"),
                "month": c9["start"][:7]} for c9 in ms]
        occ.sort(key=lambda o9: o9["start"])
        r0 = ms[0]
        out.append({"sensor": r0["sensor"], "step": r0["step"], "kind": r0["kind"],
                    "direction": r0.get("direction"), "loc": r0.get("loc"),
                    "n_occur": len(ms), "months": sorted({o9["month"] for o9 in occ}),
                    "occurrences": occ, "card_idx": g9["members"]})
    out.sort(key=lambda g9: (-g9["n_occur"], g9["occurrences"][0]["start"]))
    return out


def _wafer_payload(S, groups, rec):
    """단일 웨이퍼 → 겹침용 payload(실제/기대, 골든 길이 리샘플)."""
    r = S._result_for(rec)
    recipe = S._recipe_display(r.get("recipe") or "") or "unknown"
    try:
        ratio = round(float(r.get("ratio")), 4)
    except (TypeError, ValueError):
        ratio = None
    arr = np.asarray(rec["sensors"], dtype=np.float64)
    cols = [str(c) for c in (rec.get("sensor_cols") or [])]
    pred_arr, pred_cols = None, []
    try:
        pred_arr = np.asarray(S.ENGINE.model.predict(rec), dtype=np.float64)
        pred_cols = [str(c) for c in S.ENGINE.model.sensor_cols]
        if pred_arr.shape != (arr.shape[0], len(pred_cols)):
            pred_arr = None
    except Exception:                                 # noqa: BLE001
        pred_arr = None
    g = groups.get(recipe)
    L = g["len"] if g else arr.shape[0]
    names = g["sensors"] if g else cols
    data, pred = {}, {}
    for sname in names:
        if sname in cols:
            data[sname] = [round(float(v), 4) for v in _resample(arr[:, cols.index(sname)], L)]
        if pred_arr is not None and sname in pred_cols:
            pred[sname] = [round(float(v), 4) for v in _resample(pred_arr[:, pred_cols.index(sname)], L)]
    heat = None
    if g:                                             # ── RCA: 스텝×센서 골든편차(z) 히트맵 + 최초 이탈(onset)
        T = arr.shape[0]
        idx = np.clip(np.round(np.linspace(0, T - 1, L)).astype(int), 0, T - 1)
        try:
            steps_rs = np.asarray(rec.get("step"), dtype=float)[idx].astype(int)
        except Exception:                             # noqa: BLE001 — 스텝 없으면 전체 1구간
            steps_rs = np.zeros(L, dtype=int)
        usteps = sorted(set(int(v) for v in steps_rs))

        def _first_sustained(bad, zarr):
            run = 0
            for t in range(L):
                run = run + 1 if bad[t] else 0
                if run >= 3:
                    return t - 2, float(zarr[t - 2])
            return None, None

        sensors_h, maxz_rows, maxzr_rows, onset, onset_r = [], [], [], [], []
        for j, sname in enumerate(g["sensors"]):
            if sname not in data:
                continue
            w_arr = np.asarray(data[sname], dtype=float)
            mu = g["mean"][:, j]
            sg = g["std"][:, j]
            # σ 하한: σ~0(상수 신호·표본 부족) 구간에서 z 폭주 방지 —
            #   σ 중앙값의 5% 또는 신호 크기의 0.1% 중 큰 값. 표시는 ±99σ 캡.
            floor = max(1e-6, 0.05 * float(np.median(sg)),
                        1e-3 * float(np.mean(np.abs(mu))))
            z = np.clip((w_arr - mu) / np.maximum(sg, floor), -99.0, 99.0)
            sensors_h.append(sname)
            maxz_rows.append([round(float(np.max(np.abs(z[steps_rs == st]))), 1)
                              if np.any(steps_rs == st) else None for st in usteps])
            t0, z0 = _first_sustained(np.abs(z) > 3.0, z)
            if t0 is not None:
                onset.append({"sensor": sname, "tp": int(t0),
                              "step": int(steps_rs[t0]), "z": round(z0, 1)})
            # ── 모델 잔차 z: 실제−기대를 웨이퍼 내 강건 정규화(MAD) —
            #   '모델이 놀란 정도'. 과거 오염을 따라간 구간은 잔차가 작아지므로,
            #   골든z 큼+잔차z 작음 = 오염 전파, 둘 다 큼 = 발화(시작) 후보.
            if sname in pred:
                rr = w_arr - np.asarray(pred[sname], dtype=float)
                med = float(np.median(rr))
                # 노이즈 척도 = 인접차분 MAD(레벨 시프트·이상 구간에 오염되지 않는 추정)
                dif = np.diff(rr)
                mad = float(np.median(np.abs(dif))) * 1.4826 / np.sqrt(2.0) if dif.size else 0.0
                if mad < 1e-9:                        # 차분이 전부 0(상수 잔차)이면 레벨 MAD 폴백
                    mad = float(np.median(np.abs(rr - med))) * 1.4826
                rscale = max(mad, 1e-6, 1e-4 * float(np.mean(np.abs(mu))))
                zr = np.clip((rr - med) / rscale, -99.0, 99.0)
                maxzr_rows.append([round(float(np.max(np.abs(zr[steps_rs == st]))), 1)
                                   if np.any(steps_rs == st) else None for st in usteps])
                t1, z1 = _first_sustained(np.abs(zr) > 3.0, zr)
                if t1 is not None:
                    onset_r.append({"sensor": sname, "tp": int(t1),
                                    "step": int(steps_rs[t1]), "z": round(z1, 1)})
            else:
                maxzr_rows.append([None for _ in usteps])
        onset.sort(key=lambda o: o["tp"])
        onset_r.sort(key=lambda o: o["tp"])
        # ── 카트리지 목표추종: Cart(실측 출력온도) − Heater/Heat(목표) — 존 실온 아님 주의
        import re as _re
        cart_names, cart_rows = [], []
        try:
            steps_o = np.asarray(rec.get("step"), dtype=float).astype(int)
        except Exception:                             # noqa: BLE001
            steps_o = np.zeros(arr.shape[0], dtype=int)
        _colmap = {c.lower(): k for k, c in enumerate(cols)}
        for c in cols:
            m2 = _re.match(r'(?i)^(vessel(?:up|down))cart(\d+)_pv$', c)
            if not m2:
                continue
            zone, num = m2.group(1).lower(), m2.group(2)
            targets = [f"{zone}heater{num}_pv", f"{zone}heat{num}_pv"]
            tk = next((_colmap[t] for t in targets if t in _colmap), None)
            if tk is None:
                continue
            dv = arr[:, _colmap[c.lower()]] - arr[:, tk]
            row = []
            for st in usteps:
                seg = dv[steps_o == st]
                if seg.size == 0:
                    row.append(None)
                else:
                    k2 = int(np.argmax(np.abs(seg)))
                    row.append(round(float(seg[k2]), 1))   # 부호 유지한 최대편차(℃)
            cart_names.append(c)
            cart_rows.append(row)
        heat = {"steps": usteps, "sensors": sensors_h, "maxz": maxz_rows,
                "maxzr": maxzr_rows, "onset": onset, "onset_r": onset_r,
                "cart": ({"names": cart_names, "dev": cart_rows} if cart_names else None)}
    return {"wid": str(rec["wafer_id"]), "recipe": recipe, "ratio": ratio,
            "verdict": r.get("verdict"), "has_gold": bool(g), "data": data, "pred": pred,
            "heat": heat}


_DTE_PROC = {"p": None}


def _find_dte_dir():
    """deep-thinking-engine 설치 폴더 탐색: RCA_DTE_DIR > ~/deep-thinking-engine > $HOME 밑 검색."""
    cand = [os.environ.get("RCA_DTE_DIR", ""),
            os.path.expanduser("~/deep-thinking-engine")]
    for c in cand:
        if c and os.path.isfile(os.path.join(c, "src", "deepthink", "server.py")):
            return c
    home = os.path.expanduser("~")
    try:
        for d1 in sorted(os.listdir(home)):
            c = os.path.join(home, d1)
            if os.path.isfile(os.path.join(c, "src", "deepthink", "server.py")):
                return c
    except OSError:
        pass
    return None


def _dte_alive(host, timeout=3):
    import urllib.request as _ur
    try:
        with _ur.urlopen(host.rstrip("/") + "/openapi.json", timeout=timeout):
            return True
    except Exception:                                  # noqa: BLE001
        return False


def _ensure_dte(required=False):
    """DTE_HOST 가 살아있으면 그대로, 아니면 DTE 서버를 앱이 직접 띄워 연결한다.

    - 위치: RCA_DTE_DIR 또는 자동 탐색 · 설정: RCA_DTE_CONFIG(기본 config/ollama.yaml)
    - 파이썬: RCA_DTE_PYTHON(DTE 전용 venv 가 따로 있으면 지정), 기본은 앱과 같은 인터프리터
    - 포트: RCA_DTE_PORT(기본 8571) · 앱 종료 시 자동 정리
    """
    import subprocess, atexit, time as _t
    host = (os.environ.get("DTE_HOST") or "").rstrip("/")
    if host and _dte_alive(host):
        return host
    dte = _find_dte_dir()
    if not dte:
        if required:
            raise RuntimeError("deep-thinking-engine 폴더를 찾지 못했습니다 — "
                               "RCA_DTE_DIR=/경로 로 지정하세요")
        return None
    port = int(os.environ.get("RCA_DTE_PORT", "8571"))
    host = f"http://127.0.0.1:{port}"
    if not _dte_alive(host):
        cfg = os.environ.get("RCA_DTE_CONFIG") or os.path.join(dte, "config", "ollama.yaml")
        if not os.path.isfile(cfg):
            cfg = ""
        py = os.environ.get("RCA_DTE_PYTHON") or sys.executable
        code = ("import uvicorn; from deepthink.server import create_app; "
                f"uvicorn.run(create_app({cfg!r} or None), host='127.0.0.1', "
                f"port={port}, log_level='warning')")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(dte, "src") + os.pathsep + env.get("PYTHONPATH", "")
        logf = open(os.path.join(dte, "dte_auto.log"), "ab")
        proc = subprocess.Popen([py, "-c", code], cwd=dte, env=env,
                                stdout=logf, stderr=logf)
        _DTE_PROC["p"] = proc
        atexit.register(lambda: proc.poll() is None and proc.terminate())
        for _ in range(40):                            # 최대 20초 대기
            _t.sleep(0.5)
            if _dte_alive(host, timeout=1):
                break
            if proc.poll() is not None:
                msg = (f"DTE 자동 실행 실패(로그: {dte}/dte_auto.log) — deepthink 패키지 "
                       f"의존성이 이 파이썬에 없으면 RCA_DTE_PYTHON=DTE전용venv/bin/python 지정")
                if required:
                    raise RuntimeError(msg)
                print("[dte] " + msg)
                return None
        else:
            if required:
                raise RuntimeError("DTE 기동 대기 시간 초과(20초) — dte_auto.log 확인")
            return None
        print(f"[dte] deep-thinking-engine 자동 실행 — {host} (폴더 {dte})")
    os.environ["DTE_HOST"] = host
    return host


def serve(S, groups, stats, folder, ok_max, model_name):
    """경량 로컬 서버 — 페이지 1장 + 스캔/파형 API. 브라우저 자동 오픈."""
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    import socket
    import threading
    import webbrowser
    import uvicorn

    page = render_html(groups, stats, folder, ok_max, model_name,
                       wafers=True)                    # has_waf=True → 분석 탭 UI 포함
    app = FastAPI()
    WST = {"folder": None, "stems": []}                # 스캔 상태(단일 사용자 도구)

    def _load_by_idx(idxs):
        """스캔 인덱스 목록 → 출처 폴더별로 묶어 로드(다중 폴더 안전). rec 순서는 보장 안 함."""
        srcs9 = WST.get("src") or []
        by_f = {}
        for i9 in idxs:
            f9 = srcs9[i9] if i9 < len(srcs9) else str(WST["folder"])
            by_f.setdefault(f9, []).append(WST["stems"][i9])
        recs9 = []
        for f9, st9 in by_f.items():
            r9, _x, _y, _z = S._load_aligned(Path(f9), stems=st9)
            recs9 += r9
        return recs9

    @app.get("/", response_class=HTMLResponse)
    def index():
        return page

    @app.get("/api/ui_state")
    def api_ui_state():
        """새로고침 복원: 서버 메모리에 남은 마지막 결과들(서버 재시작 시 소멸)."""
        _sd_try_restore()
        _ts_try_restore()
        st9 = _state_load()
        return {"scan": WST.get("scan_out"), "gx": WST.get("gx_out"),
                "tscan_done": bool(WST.get("tscan")), "sd_done": SDJOB["state"] != "idle",
                "sd_form": st9.get("sd"), "ts_form": st9.get("ts"),
                "sde_form": st9.get("sde"), "sde": (SDE if SDE else None),
                "vz_form": st9.get("vz"), "vzrpt_form": st9.get("vzrpt"),
                "dv_form": st9.get("dv"), "dv_forms": st9.get("dvform") or [],
                "folder": str(WST.get("folder") or "")}

    @app.post("/api/scan")
    def api_scan(payload: dict = Body(...)):          # ★ 로컬 pydantic 모델은 future-annotations 와 충돌
        wfs = split_folders(payload.get("folder"))
        if not wfs:
            raise HTTPException(400, "폴더를 입력하세요")
        missing = [str(w9) for w9 in wfs if not w9.is_dir()]
        if missing:
            raise HTTPException(400, "폴더가 없습니다: " + " ; ".join(missing))
        per, n_all, n_sel, avail = select_stems_multi(S, wfs, payload.get("dsv"),
                                                      payload.get("recipe"), payload.get("months"),
                                                      payload.get("eqp"))
        filtered = any(x[1] is not None for x in per)
        if filtered and n_sel == 0:
            raise HTTPException(400, f"필터를 통과한 파일이 없습니다 (전체 {n_all}장). "
                                     + _avail_text(avail))
        if filtered:
            print(f"[scan] 필터: 전체 {n_all}장 → {n_sel}장 판정", flush=True)
        if len(wfs) > 1:
            print(f"[scan] 데이터 폴더 {len(wfs)}곳 병합", flush=True)
        print(f"[scan] 파일 읽는 중 {n_sel if filtered else n_all}장 "
              "— 수 분 걸릴 수 있습니다(다음 로그: [batch] 스캔 추론)", flush=True)
        recs, srcs, _c = load_aligned_multi(S, per)
        print(f"[scan] 읽기 완료 {len(recs)}장 → 추론 시작", flush=True)
        if not recs:
            raise HTTPException(400, f"읽을 수 있는 웨이퍼가 없습니다: {wf}")
        CH = 1000
        for k0 in range(0, len(recs), CH):
            S._prefill_batch(recs[k0:k0 + CH])
            if len(recs) > CH:
                print(f"[batch] 스캔 추론 {min(k0 + CH, len(recs))}/{len(recs)}", flush=True)
        metas, stems = [], []
        trend_rows = []                               # 기간 추세용: (date, ng?, {sensor: maxz})
        n_ok = n_ng = n_na = 0
        for i, rec in enumerate(recs):
            r = S._result_for(rec)
            try:
                ratio = round(float(r.get("ratio")), 4)
            except (TypeError, ValueError):
                ratio = None
            if ratio is None:
                n_na += 1
            elif ratio > 1.0 or r.get("verdict") == "NG":
                n_ng += 1
            else:
                n_ok += 1
            metas.append({"i": i, "wid": str(rec["wafer_id"]),
                          "recipe": S._recipe_display(r.get("recipe") or "") or "unknown",
                          "date": str(r.get("end_ts") or "")[:10] or "-",
                          "ts": str(r.get("end_ts") or ""),
                          "ratio": ratio, "verdict": r.get("verdict")})
            stems.append(str(rec["wafer_id"]))
            # ── 센서별 골든 이탈(maxz) 경량 계산 — 기간 추세 재료
            try:
                rc_disp = metas[-1]["recipe"]
                g2 = groups.get(rc_disp)
                if g2 is not None:
                    arr2 = np.asarray(rec["sensors"], dtype=np.float32)
                    cols2 = [str(c) for c in (rec.get("sensor_cols") or [])]
                    T3 = arr2.shape[0]
                    idx3 = np.clip(np.round(np.linspace(0, T3 - 1, g2["len"])).astype(int), 0, T3 - 1)
                    pred2, pcols2 = None, []
                    try:
                        pred2 = np.asarray(S.ENGINE.model.predict(rec), dtype=np.float64)
                        pcols2 = [str(c) for c in S.ENGINE.model.sensor_cols]
                        if pred2.shape[0] != T3:
                            pred2 = None
                    except Exception:                 # noqa: BLE001
                        pred2 = None
                    zz, gz, rz = {}, {}, {}
                    for j2, sn2 in enumerate(g2["sensors"]):
                        if sn2 not in cols2:
                            continue
                        w2 = arr2[idx3, cols2.index(sn2)].astype(np.float64)
                        mu2 = g2["mean"][:, j2]; sg2 = g2["std"][:, j2]
                        fl2 = max(1e-6, 0.05 * float(np.median(sg2)),
                                  1e-3 * float(np.mean(np.abs(mu2))))
                        zs = np.clip((w2 - mu2) / np.maximum(sg2, fl2), -99, 99)
                        zz[sn2] = round(float(np.max(np.abs(zs))), 2)
                        gz[sn2] = round(float(np.mean(zs)), 2)          # 부호 있는 평균(골든 대비)
                        if pred2 is not None and sn2 in pcols2:
                            rr2 = (arr2[:, cols2.index(sn2)].astype(np.float64)
                                   - pred2[:, pcols2.index(sn2)])[idx3]
                            rz[sn2] = round(float(np.mean(rr2 / np.maximum(sg2, fl2))), 2)
                    trend_rows.append({"date": metas[-1]["date"],
                                       "wid": metas[-1]["wid"],
                                       "recipe": rc_disp,
                                       "ratio": metas[-1]["ratio"],
                                       "ng": (metas[-1]["ratio"] is not None
                                              and metas[-1]["ratio"] > 1.0),
                                       "z": zz, "gz": gz, "rz": rz})
            except Exception:                         # noqa: BLE001 — 추세는 부가 기능
                pass
        WST["folder"], WST["stems"] = wfs[0], stems
        WST["flist"] = wfs
        WST["src"] = srcs                              # stem 별 출처 폴더(다중 폴더 병합용)
        WST["dates"] = [m["date"] for m in metas]      # 정렬 전 = rec/stem 순서와 동일
        WST["trend"] = trend_rows
        # ★ 시간 오름차순(끝시각→이름) — 목록을 내리면 시간 흐름대로 보인다(사용자 요청)
        metas.sort(key=lambda m: (m.get("ts") or "", m["wid"]))
        print(f"[scan] {' ; '.join(str(w9) for w9 in wfs)} — {len(metas)}장 "
              f"(정상 {n_ok} · 불량 {n_ng} · 보류 {n_na})", flush=True)
        out_sc = {"wafers": metas, "n_ok": n_ok, "n_ng": n_ng, "n_na": n_na,
                  "n_all_files": n_all, "n_selected": n_sel}
        WST["scan_out"] = out_sc                      # 새로고침 복원용(서버 메모리)
        return out_sc

    @app.post("/api/export")
    def api_export(chunk: int = 600):
        """스캔된 폴더 전체를 600장 단위 정적 HTML 여러 개로 저장(서버 없이 열림)."""
        if WST["folder"] is None:
            raise HTTPException(400, "먼저 폴더를 불러오세요")
        recs = _load_by_idx(list(range(len(WST["stems"]))))
        if not recs:
            raise HTTPException(400, "웨이퍼를 다시 읽지 못했습니다")
        # 목록과 같은 시간 오름차순으로 정렬
        def _ts_of(rec):
            try:
                return str(S._end_ts_and_date(rec["wafer_id"])[0])
            except Exception:                         # noqa: BLE001
                return ""
        recs.sort(key=lambda rc: (_ts_of(rc), str(rc["wafer_id"])))
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        outdir = Path.cwd() / f"avg_report_{stamp}"
        outdir.mkdir(exist_ok=True)
        files = []
        n_all = len(recs)
        n_parts = (n_all + chunk - 1) // chunk
        for kpart in range(n_parts):
            part = recs[kpart * chunk:(kpart + 1) * chunk]
            metas, pay = [], {}
            for i, rec in enumerate(part):
                w = _wafer_payload(S, groups, rec)
                metas.append({"i": i, "wid": w["wid"], "recipe": w["recipe"],
                              "date": _ts_of(rec)[:10] or "-", "ts": _ts_of(rec),
                              "ratio": w["ratio"], "verdict": w["verdict"]})
                pay[str(i)] = w
                if (i + 1) % 100 == 0:
                    print(f"[export] {kpart+1}/{n_parts}파트 {i+1}/{len(part)}", flush=True)
            page2 = render_html(groups, stats, folder, ok_max, model_name,
                                wafers=True, embed=(metas, pay))
            f = outdir / f"part{kpart+1:02d}_{kpart*chunk+1}-{kpart*chunk+len(part)}.html"
            f.write_text(page2, encoding="utf-8")
            files.append(f.name)
            print(f"[export] 저장 {f.name} ({len(part)}장 · {f.stat().st_size//1024//1024}MB)", flush=True)
        return {"dir": str(outdir), "files": files, "n": n_all}

    @app.post("/api/rca_explain")
    def api_rca_explain(payload: dict = Body(...)):
        i = int(payload.get("i", -1))
        if WST["folder"] is None or not (0 <= i < len(WST["stems"])):
            raise HTTPException(400, "먼저 폴더를 불러오세요")
        recs = _load_by_idx([i])
        if not recs:
            raise HTTPException(404, "웨이퍼 파일을 다시 읽지 못했습니다")
        rec = recs[0]
        w = _wafer_payload(S, groups, rec)
        g = groups.get(w["recipe"])
        if not g:
            raise HTTPException(400, "이 유형의 기준(골든) 파형이 없어 RCA 를 만들 수 없습니다")
        T2 = np.asarray(rec["sensors"], dtype=float).shape[0]
        L2 = g["len"]
        idx2 = np.clip(np.round(np.linspace(0, T2 - 1, L2)).astype(int), 0, T2 - 1)
        try:
            steps_rs = np.asarray(rec.get("step"), dtype=float)[idx2].astype(int)
        except Exception:                             # noqa: BLE001
            steps_rs = np.zeros(L2, dtype=int)
        from io_recon_tcn import rca_causal as RC
        try:
            from io_recon_tcn.sensor_graph import graph_sensor_rank
            ranks = graph_sensor_rank(g["sensors"], S.ENGINE.cfg)
        except Exception:                             # noqa: BLE001
            ranks = {sn: k for k, sn in enumerate(g["sensors"])}
        packet = RC.analyze(w["wid"], w["recipe"], g, w["data"], w["pred"], steps_rs, ranks)
        # DTE(deep-thinking-engine) 검증형 경로용 사건 자료: 원본 웨이퍼 + 골든 통계 + 패킷
        case_files = []
        try:
            src = rec.get("source_path")
            if src and Path(str(src)).exists():
                case_files.append(str(src))
            cdir = Path(tempfile.mkdtemp(prefix="rca_case_"))
            gcsv = cdir / f"golden_{w['recipe'][:20]}.csv"
            with gcsv.open("w", encoding="utf-8") as fh:
                fh.write("tp," + ",".join(f"{sn}_mean,{sn}_std" for sn in g["sensors"]) + "\n")
                for t in range(g["len"]):
                    fh.write(str(t) + "," + ",".join(
                        f"{g['mean'][t, j]:.4f},{g['std'][t, j]:.4f}"
                        for j in range(len(g["sensors"]))) + "\n")
            pjson = cdir / "evidence_packet.json"
            pjson.write_text(json.dumps(packet, ensure_ascii=False, indent=1), encoding="utf-8")
            case_files += [str(gcsv), str(pjson)]
            # 설비 도면(depicted) — unsloth GGUF 는 mmproj 내장이라 ollama 로도 판독됨
            #   (DTE install 스크립트 실측 주석 확인, 2026-09-08). 항상 포함.
            gp = RC.find_equipment_graph()
            if gp:
                case_files.append(gp)
        except Exception:                             # noqa: BLE001 — 자료 없이도 ollama 경로는 동작
            traceback.print_exc()
        mode = str(payload.get("mode") or "auto")
        level = payload.get("level") or None
        if mode in ("auto", "deep"):
            try:
                _ensure_dte(required=(mode == "deep"))
            except RuntimeError as e:
                raise HTTPException(400, str(e))
        nar = RC.narrate(packet, case_files=case_files, mode=mode, thinking_level=level)
        return {"packet": packet, **nar}

    @app.post("/api/daily_csv")
    def api_daily_csv():
        """일별×센서 판정 CSV 3종: vs_golden(와이드)·vs_model(와이드)·detail(롱).
        기호: ▲ = 평균/기대보다 높음(+1σ↑) · ▼ = 낮음(−1σ↓) · 공백 = 정상 범위."""
        rows = WST.get("trend") or []
        if not rows:
            raise HTTPException(400, "먼저 폴더를 불러오세요(스캔 필요)")
        days = {}
        sensors_seen = []
        for r2 in rows:
            d2 = days.setdefault(r2["date"], {"n": 0, "gz": {}, "rz": {}})
            d2["n"] += 1
            for key in ("gz", "rz"):
                for sn3, v3 in (r2.get(key) or {}).items():
                    if sn3 not in sensors_seen:
                        sensors_seen.append(sn3)
                    acc2 = d2[key].setdefault(sn3, [0.0, 0])
                    acc2[0] += v3; acc2[1] += 1
        dates = sorted(k for k in days if k and k != "-")

        def _mark(v):
            if v is None:
                return ""
            if v >= 1.0:
                return f"▲{v:+.1f}"
            if v <= -1.0:
                return f"▼{v:+.1f}"
            return f"{v:+.1f}"

        def _avg(d3, key, sn3):
            a2 = days[d3][key].get(sn3)
            return (a2[0] / a2[1]) if a2 and a2[1] else None

        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        outdir = Path.cwd() / f"sensor_daily_{stamp}"
        outdir.mkdir(exist_ok=True)
        files = []
        for key, fname, label in (("gz", "vs_golden.csv", "정상평균 대비"),
                                  ("rz", "vs_model.csv", "모델 기대값 대비")):
            cols3 = [sn3 for sn3 in sensors_seen
                     if any(days[d3][key].get(sn3) for d3 in dates)]
            lines = [f"# {label} 일별 평균 이탈(σ) — ▲=높은 쪽 이탈 · ▼=낮은 쪽 이탈 · 둘 다 문제 신호 · 무기호=정상 범위",
                     "date,n," + ",".join(cols3)]
            for d3 in dates:
                lines.append(d3 + "," + str(days[d3]["n"]) + ","
                             + ",".join(_mark(_avg(d3, key, sn3)) for sn3 in cols3))
            f4 = outdir / fname
            f4.write_text("\ufeff" + "\n".join(lines), encoding="utf-8")
            files.append(fname)
        # 상세 롱 포맷
        lines = ["\ufeffdate,sensor,n,vs_golden_sigma,golden_verdict,vs_model_sigma,model_verdict"]
        for d3 in dates:
            for sn3 in sensors_seen:
                gv = _avg(d3, "gz", sn3); rv = _avg(d3, "rz", sn3)
                if gv is None and rv is None:
                    continue
                def _v(v):
                    return ("높음" if v >= 1.0 else "낮음" if v <= -1.0 else "정상") if v is not None else ""
                lines.append(f"{d3},{sn3},{days[d3]['n']},"
                             f"{'' if gv is None else round(gv, 2)},{_v(gv)},"
                             f"{'' if rv is None else round(rv, 2)},{_v(rv)}")
        (outdir / "detail.csv").write_text("\n".join(lines), encoding="utf-8")
        files.append("detail.csv")
        print(f"[csv] 일별 판정 저장: {outdir} ({len(dates)}일 × {len(sensors_seen)}센서)", flush=True)
        return {"dir": str(outdir), "files": files,
                "n_days": len(dates), "n_sensors": len(sensors_seen)}

    RST = {"sid": None, "dir": None, "status": "idle", "error": None}

    def _build_corpus() -> Path:
        """딥리서치 코퍼스: 스캔 산출물 전부를 파일로 — DTE 도구가 직접 재계산."""
        rows = WST.get("trend") or []
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        cdir = Path.cwd() / f"research_corpus_{stamp}"
        cdir.mkdir(exist_ok=True)
        # 1) 웨이퍼×센서 통계(롱): 사람이 못 훑는 원자료
        sensors_all = sorted({sn for r2 in rows for sn in r2.get("gz", {})})
        lines = ["wafer_id,date,recipe,score_ratio,ng,sensor,"
                 "golden_mean_sigma,golden_max_sigma,model_resid_sigma"]
        for r2 in rows:
            for sn in sensors_all:
                if sn not in r2.get("gz", {}):
                    continue
                lines.append(f"{r2.get('wid','')},{r2['date']},{r2.get('recipe','')},"
                             f"{r2.get('ratio','')},{int(bool(r2['ng']))},{sn},"
                             f"{r2['gz'].get(sn,'')},{r2['z'].get(sn,'')},{r2.get('rz',{}).get(sn,'')}")
        (cdir / "wafer_sensor_stats.csv").write_text("\n".join(lines), encoding="utf-8")
        # 2) 골든 파형 통계(레시피별)
        for rc4, g4 in groups.items():
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", rc4)[:40]
            with (cdir / f"golden_{safe}.csv").open("w", encoding="utf-8") as fh:
                fh.write("tp," + ",".join(f"{sn}_mean,{sn}_std" for sn in g4["sensors"]) + "\n")
                for t in range(g4["len"]):
                    fh.write(str(t) + "," + ",".join(
                        f"{g4['mean'][t, j]:.4f},{g4['std'][t, j]:.4f}"
                        for j in range(len(g4["sensors"]))) + "\n")
        # 3) 설비 사실 문서
        from io_recon_tcn import rca_causal as RC2
        facts = ["# 설비·데이터 사전",
                 "",
                 RC2.analyze.__doc__ or "",
                 "## 배관 경로",
                 "CO2공급원 -> Reservoir(H1~4,Vent) -> 밸브031 -> p_002 -> h_0052",
                 "-> 충전미터링 -> h_0161 -> p_005 -> 필터 -> t_005 -> h_005",
                 "-> t_002 -> 밸브032 분기 -> [상부: Front계열 -> p_008 -> VesselUp존]",
                 "   [하부: h_0053 -> t_001 -> Back계열 -> p_007 -> VesselDown존]",
                 "-> 웨이퍼 세정 -> Outlet -> t_003 -> PulseVent/SlowVent 배기",
                 "## 센서 의미(사실)",
                 "- VesselUp/h_030N_PV = 카트리지(열원) 출력 온도. VesselUp/DownHeat(er)N_PV = 존 외부 온도이자 목표(SV).",
                 "- 웨이퍼가 겪는 온도에 가까운 것은 Heater(존) 쪽. Cart 는 열원이 무리한 정도.",
                 "- H 라인 히터(필터/밸브/라인)는 자체 고정설정, 피드백 없음.",
                 "- 가스 T/P 만 배관 전파. scCO2: 과열->압력상승->벤트 증가->팽창냉각(배기 저온).",
                 "## 파일 설명",
                 "- wafer_sensor_stats.csv: 웨이퍼×센서. golden_mean_sigma=골든 대비 부호평균(σ),",
                 "  golden_max_sigma=|z|최대, model_resid_sigma=모델 기대 대비 부호평균(σ). ng=1 불량.",
                 "- golden_*.csv: 레시피별 정상 평균/표준편차 파형."]
        (cdir / "FACTS.md").write_text("\n".join(facts), encoding="utf-8")
        gp = RC2.find_equipment_graph()
        if gp:
            import shutil as _sh
            _sh.copy(gp, cdir / "equipment_graph.png")
        return cdir

    @app.post("/api/research")
    def api_research():
        import threading as _th
        import urllib.request as _ur
        if not (WST.get("trend")):
            raise HTTPException(400, "먼저 폴더를 불러오세요(스캔 필요)")
        host = (_ensure_dte() or "").rstrip("/")
        if not host:
            raise HTTPException(400, "DTE 를 찾거나 띄우지 못했습니다 — RCA_DTE_DIR=/경로 지정 "
                                     "(자동 실행 로그: DTE폴더/dte_auto.log)")
        if RST["status"] == "running":
            raise HTTPException(409, "딥리서치가 이미 진행 중입니다")
        cdir = _build_corpus()
        n_w = len(WST["trend"])
        goal = (f"초임계 CO2 세정설비 검사 데이터 {n_w}장(웨이퍼×센서 통계·레시피별 정상 파형·"
                "설비 사전 포함)을 장기 리서치하라. 목표: 사람이 일일이 찾기 어려운 문제와 패턴의 발굴 — "
                "예: 불량 급증의 전조가 되는 센서·조합, 날짜/레시피/챔버에 따른 구조적 차이, "
                "센서 간 지연 상관, 간헐적으로만 나타나는 이상, 서서히 누적되는 드리프트. "
                "질문을 스스로 세우고, 각 패턴은 통계 검증(재현·선행성)을 통과한 것만 보고하라. "
                "산출물: 발견 패턴 목록(각각 증거 수치·재현 조건·엔지니어 관점 의미·확인 방법). "
                "확실치 않으면 확실치 않다고 표시하라.")
        body = json.dumps({"goal": goal,
                           "data_files": [str(p3) for p3 in sorted(cdir.iterdir())],
                           "constraints": ["FACTS.md 의 설비 사실을 전제로 하되 판단은 데이터로"],
                           "thinking_level": "deep_research"}).encode()
        req2 = _ur.Request(host + "/sessions", data=body,
                           headers={"Content-Type": "application/json"})
        with _ur.urlopen(req2, timeout=60) as r2:
            sid = json.loads(r2.read())["session_id"]
        RST.update(sid=sid, dir=str(cdir), status="running", error=None)

        RST["chunk"] = int(os.environ.get("RCA_RESEARCH_CHECKPOINT", "6"))
        RST["resume"] = None                          # 'continue' | 'finish'

        def _runner():
            # ★ GPT 딥리서치식 체크포인트: K 스텝마다 멈춰 사용자 확인/방향 지시를 기다린다.
            try:
                while True:
                    for _k in range(RST["chunk"]):
                        rq = _ur.Request(f"{host}/sessions/{sid}/step", data=b"{}",
                                         headers={"Content-Type": "application/json"})
                        out = json.loads(_ur.urlopen(rq, timeout=3600).read() or b"{}")
                        if out.get("stop") or out.get("stop_reason"):
                            RST["status"] = "done"
                            return
                    RST["status"] = "paused"          # 체크포인트 — 사용자 응답 대기
                    RST["resume"] = None
                    while RST["resume"] is None:
                        import time as _t2
                        _t2.sleep(2)
                    if RST["resume"] == "finish":
                        RST["status"] = "done"
                        return
                    RST["status"] = "running"
            except Exception as e:                     # noqa: BLE001
                RST.update(status="error", error=str(e)[:300])
        _th.Thread(target=_runner, daemon=True).start()
        print(f"[research] 시작 — 코퍼스 {cdir} · 세션 {sid}", flush=True)
        return {"sid": sid, "corpus": str(cdir)}

    @app.get("/api/research_status")
    def api_research_status():
        import urllib.request as _ur
        if not RST["sid"]:
            return {"status": "idle"}
        host = (_ensure_dte() or "").rstrip("/")
        info = {}
        try:
            with _ur.urlopen(f"{host}/sessions/{RST['sid']}", timeout=30) as r2:
                info = json.loads(r2.read())
        except Exception as e:                        # noqa: BLE001
            info = {"fetch_error": str(e)[:120]}
        return {"status": RST["status"], "error": RST["error"], "sid": RST["sid"],
                "corpus": RST["dir"], "session": info}

    @app.post("/api/research_answer")
    def api_research_answer(payload: dict = Body(...)):
        """체크포인트 응답: continue(계속) / finish(종료 후 보고) / redirect(방향 지시 후 계속).
        방향 지시는 다음 리서치 라운드가 참조하도록 코퍼스에 DIRECTION.md 로 추가된다."""
        act = str(payload.get("action") or "continue")
        if RST["status"] != "paused":
            raise HTTPException(400, "체크포인트 대기 상태가 아닙니다")
        txt = str(payload.get("text") or "").strip()
        if act == "redirect" and txt and RST.get("dir"):
            dpath = Path(RST["dir"]) / "DIRECTION.md"
            prev = dpath.read_text(encoding="utf-8") if dpath.exists() else "# 사용자 방향 지시(누적)\n"
            dpath.write_text(prev + f"\n- [{dt.datetime.now().strftime('%m-%d %H:%M')}] {txt}",
                             encoding="utf-8")
            print(f"[research] 방향 지시 추가: {txt[:80]}", flush=True)
        RST["resume"] = "finish" if act == "finish" else "continue"
        return {"ok": True, "action": act}

    @app.get("/api/research_report")
    def api_research_report():
        import urllib.request as _ur
        if not RST["sid"]:
            raise HTTPException(400, "진행 중인 리서치가 없습니다")
        host = (_ensure_dte() or "").rstrip("/")
        with _ur.urlopen(f"{host}/sessions/{RST['sid']}/report", timeout=600) as r2:
            rep = json.loads(r2.read())
        from io_recon_tcn import rca_causal as RC
        rep.update(RC.debate_report(rep))              # RCA_CRITIC_MODEL 설정 시 토론 첨부
        return rep

    def _period_pack(last_days: int = None):
        """trend rows → 일별 집계(daily)·압축 표·규칙 엔진 탐지. 검사 데이터만 사용."""
        rows = WST.get("trend") or []
        if not rows:
            raise HTTPException(400, "먼저 폴더를 불러오세요(스캔 필요)")
        days = {}
        sensors_seen = []
        for r2 in rows:
            d2 = days.setdefault(r2["date"], {"n": 0, "ng": 0, "gz": {}})
            d2["n"] += 1
            d2["ng"] += 1 if r2.get("ng") else 0
            for sn3, v3 in (r2.get("gz") or {}).items():
                if sn3 not in sensors_seen:
                    sensors_seen.append(sn3)
                acc2 = d2["gz"].setdefault(sn3, [0.0, 0])
                acc2[0] += v3; acc2[1] += 1
        days = {k: v for k, v in days.items() if k and k != "-"}
        dates = sorted(days)
        if last_days and len(dates) > last_days:
            keep = set(dates[-last_days:])
            days = {k: v for k, v in days.items() if k in keep}
            dates = sorted(days)
        cols3 = sensors_seen
        from io_recon_tcn import rca_causal as RC
        eng = RC.period_engine(days, cols3)            # 엔진은 일 단위 그대로(정밀)
        # ★ 넓은 기간(70일 초과)은 LLM 표를 '주 단위'로 압축 — 240일×전 센서 CSV 는
        #   컨텍스트(32K)를 넘어 잘리거나 실패한다. 엔진 탐지는 일 단위를 유지하므로
        #   변곡일·드리프트 시작일 정밀도는 그대로다.
        if len(dates) > 70:
            import datetime as _dt9
            wk = {}
            for d3 in dates:
                dd = _dt9.date.fromisoformat(d3)
                w0 = (dd - _dt9.timedelta(days=dd.weekday())).isoformat()
                t9 = wk.setdefault(w0, {"n": 0, "ng": 0, "gz": {}})
                t9["n"] += days[d3]["n"]; t9["ng"] += days[d3]["ng"]
                for sn3, a2 in days[d3]["gz"].items():
                    b2 = t9["gz"].setdefault(sn3, [0.0, 0])
                    b2[0] += a2[0]; b2[1] += a2[1]
            lines = [f"# 주 단위 압축(기간 {len(dates)}일 — 행=주 시작일 월요일)",
                     "week,n,ng," + ",".join(cols3)]
            for w0 in sorted(wk):
                vals = []
                for sn3 in cols3:
                    a2 = wk[w0]["gz"].get(sn3)
                    vals.append(f"{a2[0]/a2[1]:+.1f}" if a2 and a2[1] else "")
                lines.append(f"{w0},{wk[w0]['n']},{wk[w0]['ng']}," + ",".join(vals))
        else:
            lines = ["date,n,ng," + ",".join(cols3)]
            for d3 in dates:
                vals = []
                for sn3 in cols3:
                    a2 = days[d3]["gz"].get(sn3)
                    vals.append(f"{a2[0]/a2[1]:+.1f}" if a2 and a2[1] else "")
                lines.append(f"{d3},{days[d3]['n']},{days[d3]['ng']}," + ",".join(vals))
        return "\n".join(lines), eng, RC, len(dates)

    def _step_agg(recs, window_set):
        """웨이퍼 목록 → 일자별 (sensor, step, kind) 발생 집계 (웨이퍼 단위)."""
        agg = {}                                       # date -> {(sn,st,kind): [cnt, mag_sum]}
        day_n = {}
        for rec in recs:
            r2 = S._result_for(rec)
            dt3 = str(r2.get("end_ts") or "")[:10]
            rc2 = S._recipe_display(r2.get("recipe") or "") or "unknown"
            g3 = groups.get(rc2)
            if g3 is None or dt3 not in window_set:
                continue
            day_n[dt3] = day_n.get(dt3, 0) + 1
            arr3 = np.asarray(rec["sensors"], dtype=np.float64)
            cols3 = [str(c) for c in (rec.get("sensor_cols") or [])]
            T4 = arr3.shape[0]
            idx4 = np.clip(np.round(np.linspace(0, T4 - 1, g3["len"])).astype(int), 0, T4 - 1)
            try:
                steps4 = np.asarray(rec.get("step"), dtype=float)[idx4].astype(int)
            except Exception:                          # noqa: BLE001
                steps4 = np.zeros(g3["len"], dtype=int)
            usteps4 = sorted(set(int(v) for v in steps4))
            # 모델 복원값(있으면) — H 센서류는 모델 출력이 없어 골든 기준만 적용된다
            pred4, pcols4 = None, []
            try:
                pred4 = np.asarray(S.ENGINE.model.predict(rec), dtype=np.float64)
                pcols4 = [str(c) for c in S.ENGINE.model.sensor_cols]
                if pred4.shape[0] != T4:
                    pred4 = None
            except Exception:                          # noqa: BLE001
                pred4 = None
            da = agg.setdefault(dt3, {})

            def _acc(sn9, feats, prefix=""):
                for f4 in feats:
                    if f4["step"] == 0:                # 스텝0 제외(이전 공정 이월)
                        continue
                    for kd in f4["kinds"]:
                        mag = {"레벨높음": f4["level"], "레벨낮음": f4["level"],
                               "기울기상이": f4["slope"], "출렁임": f4["rough"],
                               "모양상이": f4["corr"]}[kd]
                        a4 = da.setdefault((sn9, f4["step"], prefix + kd), [0, 0.0])
                        a4[0] += 1; a4[1] += float(mag)

            for j3, sn4 in enumerate(g3["sensors"]):
                if sn4 not in cols3:
                    continue
                w4 = arr3[idx4, cols3.index(sn4)]
                sg4 = g3["std"][:, j3]
                _acc(sn4, _step_trend(w4, g3["mean"][:, j3], sg4, steps4, usteps4))
                # 모델 복원 기준: 같은 판정기, 기준 파형만 모델 출력 (σ 척도는 골든 공유)
                if pred4 is not None and sn4 in pcols4:
                    p4 = pred4[idx4, pcols4.index(sn4)]
                    _acc(sn4, _step_trend(w4, p4, sg4, steps4, usteps4), prefix="모델·")
        return agg, day_n

    def _step_trend_log(per_day: int = 6, cap: int = 300, max_lines: int = 0):
        """기간 전체에서 일자별 표본(per_day 장)을 뽑아 스텝-트렌드 이상 일지를 만든다.

        월간/조기경보용 — '13~15일 스텝5 트렌드 이상' 같은 스텝 수준 단서를 기간 표에 결합.
        표본이라 비율은 근사치(k/표본수)임을 로그에 명시.
        """
        dates_all = WST.get("dates") or []
        by_day = {}
        for i9, dt9 in enumerate(dates_all):
            if dt9 and dt9 != "-":
                by_day.setdefault(dt9, []).append(i9)
        sel = []
        for dt9 in sorted(by_day):
            sel += by_day[dt9][:per_day]
        if len(sel) > cap:                             # 전체 상한 — 오래 걸림 방지
            stride = len(sel) / cap
            sel = [sel[int(k9 * stride)] for k9 in range(cap)]
        if not sel:
            return ""
        print(f"[step-log] 스텝 트렌드 일지 표본 {len(sel)}장 재판독", flush=True)
        recs9 = _load_by_idx(sel)
        agg9, day_n9 = _step_agg(recs9, set(by_day))
        lines9 = []
        for dt9 in sorted(agg9):
            n9 = day_n9.get(dt9, 0)
            if not n9:
                continue
            # 센서·유형별 대표 스텝 1개(최다 발생)만 — 레벨 이상이 여러 스텝을 점유해
            #   출렁임 같은 다른 유형이 상위에서 밀리는 것을 방지
            best9 = {}
            for (sn9, st9, kd9), (c9, _m9) in agg9[dt9].items():
                cur9 = best9.get((sn9, kd9))
                if cur9 is None or c9 > cur9[1]:
                    best9[(sn9, kd9)] = (st9, c9)
            rows9 = sorted(best9.items(), key=lambda kv: -kv[1][1])[:8]
            top = [f"{sn9} step{st9} {kd9} {c9}/{n9}장"
                   for (sn9, kd9), (st9, c9) in rows9 if c9 >= max(2, n9 // 3)]
            if top:
                lines9.append(dt9 + ": " + " · ".join(top))
        if max_lines and len(lines9) > max_lines:      # 넓은 기간: 균등 추림
            stride9 = len(lines9) / max_lines
            lines9 = [lines9[int(k9 * stride9)] for k9 in range(max_lines)]
        return "\n".join(lines9)

    @app.post("/api/focus")
    def api_focus(payload: dict = Body(...)):
        """불량 보고 전 구간 집중 분석 — 웨이퍼 단위 스텝-트렌드 이상 집계 후 LLM."""
        if WST["folder"] is None:
            raise HTTPException(400, "먼저 폴더를 불러오세요(스캔 필요)")
        rdate = str(payload.get("date") or "").strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", rdate):
            raise HTTPException(400, "보고 시작일을 YYYY-MM-DD 로 입력하세요")
        nb = max(1, min(14, int(payload.get("days_before") or 3)))
        gcheck = None
        g_t = str(payload.get("golden_target") or "").strip()
        g_o = [x.strip() for x in str(payload.get("golden_others") or "").split(",") if x.strip()]
        if g_t and g_o:
            print(f"[focus] 1단계 — 골든 유효성 검증: DSV{g_t} vs {g_o}", flush=True)
            gtable, gvalid, gbad, _g9 = _gcmp_table(
                g_t, g_o,
                gfolders=str(payload.get("golden_folders") or "").strip() or None,
                recipes=str(payload.get("golden_recipe") or "") or None)
            gcheck = {"valid": gvalid, "bad_sensors": gbad, "table": gtable}
            if not gvalid:
                print(f"[focus] ⚠ 골든 유효성 의심: {gbad}", flush=True)
        import datetime as _dt
        d0 = _dt.date.fromisoformat(rdate)
        window = [(d0 - _dt.timedelta(days=k)).isoformat() for k in range(nb, 0, -1)]
        sel = [i for i, dt2 in enumerate(WST.get("dates") or []) if dt2 in window]
        if not sel:
            raise HTTPException(400, f"창 {window[0]}~{window[-1]} 에 해당하는 웨이퍼가 없습니다")
        print(f"[focus] 집중 분석 창 {window[0]}~{window[-1]} — {len(sel)}장 재판독", flush=True)
        recs = _load_by_idx(sel)
        agg, day_n = _step_agg(recs, set(window))
        # 표 구성(일자별 top40, 비율순) + 악화 진행 목록
        lines, ratio_series = [], {}
        for dt3 in window:
            n5 = day_n.get(dt3, 0)
            lines.append(f"## {dt3} (웨이퍼 {n5}장)")
            if not n5:
                continue
            rows5 = sorted(agg.get(dt3, {}).items(), key=lambda kv: -kv[1][0])[:40]
            for (sn4, st4, kd), (c5, ms5) in rows5:
                lines.append(f"{sn4} step{st4} {kd}: {c5}/{n5}장 ({c5/n5*100:.0f}%) 대표값 {ms5/c5:+.1f}")
                ratio_series.setdefault((sn4, st4, kd), {})[dt3] = c5 / n5
        prog = []
        for (sn4, st4, kd), by_d in ratio_series.items():
            seq = [by_d.get(d5, 0.0) for d5 in window if day_n.get(d5)]
            if len(seq) >= 2 and seq[-1] >= seq[0] + 0.2 and seq[-1] >= 0.4:
                prog.append(f"{sn4} step{st4} {kd}: " +
                            " -> ".join(f"{v*100:.0f}%" for v in seq))
        prog.sort()
        # ── 한눈 그리드: 센서×스텝 — 지배 유형·최성일 비율·일자 추이 ──
        days_eff = [d5 for d5 in window if day_n.get(d5)]
        cell = {}
        for (sn4, st4, kd), by_d in ratio_series.items():
            seq = [round(by_d.get(d5, 0.0), 2) for d5 in days_eff]
            peak = max(seq) if seq else 0.0
            cur = cell.get((sn4, st4))
            if cur is None or peak > cur["peak"]:
                cell[(sn4, st4)] = {"kind": kd, "peak": peak, "series": seq,
                                    "rising": len(seq) >= 2 and seq[-1] >= seq[0] + 0.2}
        g_sensors = sorted({k[0] for k in cell},
                           key=lambda sn: -max(v["peak"] for (s9, t9), v in cell.items()
                                               if s9 == sn))[:25]
        g_steps = sorted({k[1] for k in cell})
        grid = {"days": days_eff, "steps": g_steps, "sensors": g_sensors,
                "cells": {sn: {str(st): cell[(sn, st)] for st in g_steps
                               if (sn, st) in cell} for sn in g_sensors}}
        from io_recon_tcn import rca_causal as RC
        report_txt = str(payload.get("report") or "")
        if gcheck is not None:
            verdict_txt = ("유효(비교 DSV 들과 모양 일치)" if gcheck["valid"]
                           else "의심 — 다음 센서의 골든 모양이 비교 DSV 과반과 다름: "
                           + ", ".join(gcheck["bad_sensors"])
                           + " (이 센서들의 골든 대비 판정은 신뢰도를 낮춰 해석하라)")
            report_txt += ("\n\n[사전 골든 유효성 검증: DSV" + g_t + " vs "
                           + ",".join(g_o) + "] " + verdict_txt)
        out = RC.narrate_focus("\n".join(lines), "\n".join(prog[:30]), report_txt)
        out["golden_check"] = gcheck
        out["grid"] = grid
        out["window"] = window
        out["progression"] = prog[:30]
        out["n_wafers"] = sum(day_n.values())
        return out

    def _gcmp_table(target: str, others: list, gfolders=None, recipes=None):
        """DSV별 골든 재집계 → 센서×스텝 비교표 + 유효성 판정.

        gfolders: 비교용 데이터 폴더(문자열, ; 다중 — 비우면 서버 골든 폴더).
        recipes: 대상 레시피 필터(부분일치·쉼표) — 분석 중인 레시피만 검증할 때.
        판정 휴리스틱: 대상 vs 비교군에서 모양 붕괴(corr<0.7)가 '비교군 과반'에서
        재현되는 센서가 1개라도 있으면 '골든 유효성 의심'(비교군끼리는 정상 가정).
        """
        base = str(gfolders or folder)
        gold = {}
        for dv in [target] + others:
            print(f"[gcmp] DSV{dv} 골든 수집 중…", flush=True)
            try:
                gsub, _st5 = collect(S, base, ok_max, dsv=[dv], recipes=recipes)
            except SystemExit as e:
                raise HTTPException(400, f"DSV{dv}: {e}")
            gold[dv] = gsub
        lines = []
        for rc6, gt in gold[target].items():
            for dv in others:
                go = gold[dv].get(rc6)
                if go is None:
                    continue
                common = [sn for sn in gt["sensors"] if sn in go["sensors"]]
                L6 = gt["len"]
                for sn in common:
                    jt = gt["sensors"].index(sn); jo = go["sensors"].index(sn)
                    mt = gt["mean"][:, jt]
                    mo = np.interp(np.linspace(0, 1, L6),
                                   np.linspace(0, 1, go["len"]), go["mean"][:, jo])
                    sc6 = max(1e-6, 0.05 * float(np.median(gt["std"][:, jt])),
                              1e-3 * float(np.mean(np.abs(mt))))
                    off = float(np.mean(mt - mo)) / sc6
                    vt, vo = float(np.std(mt)), float(np.std(mo))
                    # 모양(corr)·출렁임은 유의미한 파형이 있을 때만 판정 —
                    #   평평한 센서는 골든 평균의 잔존 노이즈(σ/√N)로 corr 이 난수가 된다.
                    #   웨이퍼 σ의 절반보다 큰 굴곡이 있어야 '모양'으로 취급.
                    sig_t = float(np.median(gt["std"][:, jt]))
                    has_shape = vt > max(0.5 * sc6, 0.5 * sig_t) and vo > max(0.5 * sc6, 0.5 * sig_t)
                    corr = (float(np.corrcoef(mt, mo)[0, 1]) if has_shape else 1.0)
                    rough = (float(np.std(np.diff(mt)))
                             / max(float(np.std(np.diff(mo))), 0.1 * sc6))
                    if abs(off) >= 1.5 or corr < 0.9 or (has_shape and
                                                         (rough >= 2.0 or rough <= 0.5)):
                        lines.append(f"{rc6} | {sn} | vs DSV{dv}: offset {off:+.1f}sig "
                                     f"corr {corr:.2f} rough x{rough:.1f}")
        table = "\n".join(lines) if lines else "(유의한 차이 없음 — 모든 센서 offset<1.5σ, corr>=0.9)"
        # 유효성 판정: 센서별로 corr<0.7 인 비교 DSV 수 집계
        shape_bad = {}
        for ln in lines:
            m9 = re.search(r"\| (\S+) \| vs DSV\S+: offset \S+ corr ([-\d.]+)", ln)
            if m9 and float(m9.group(2)) < 0.7:
                shape_bad[m9.group(1)] = shape_bad.get(m9.group(1), 0) + 1
        bad_sensors = [sn for sn, c9 in shape_bad.items() if c9 >= max(1, len(others) // 2 + 1)]
        valid = not bad_sensors
        return table, valid, bad_sensors, gold

    @app.post("/api/golden_compare")
    def api_golden_compare(payload: dict = Body(...)):
        """DSV 골든 상호 비교 — 대상 DSV 골든의 건전성 검증(무겁다: 골든 폴더 재판독)."""
        target = str(payload.get("target") or "").strip()
        others = [x.strip() for x in str(payload.get("others") or "").split(",") if x.strip()]
        if not target or not others:
            raise HTTPException(400, "대상 DSV 와 비교 DSV 목록을 입력하세요")
        gf = str(payload.get("folders") or "").strip() or None
        if gf:
            miss = [str(w9) for w9 in split_folders(gf) if not w9.is_dir()]
            if miss:
                raise HTTPException(400, "폴더가 없습니다: " + " ; ".join(miss))
        table, valid, bad_sensors, gold = _gcmp_table(target, others, gfolders=gf,
                                                      recipes=str(payload.get("recipe") or "") or None)
        from io_recon_tcn import rca_causal as RC
        out = RC.narrate_golden_compare(table, target, ",".join(others))
        out["table"] = table
        out["golden_valid"] = valid
        out["bad_sensors"] = bad_sensors
        # DSV별 골든 등록: 저장소 경로가 오면 대상+비교 DSV 전부의 골든을 저장 —
        #   각 파일이 추세 스캔의 '기준 선택' 목록에 뜬다
        sv = str(payload.get("save_store") or "").strip()
        if sv:
            sd = _gstore_dir(sv, create=True)
            saved = []
            for dv, gsub in gold.items():
                for rc6, g6 in gsub.items():
                    name6 = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", f"{rc6}_dsv{dv}")
                    doc6 = {"avgwave_golden": 1, "name": name6, "recipe": rc6,
                            "len": int(g6["len"]),
                            "sensors": list(g6["sensors"]),
                            "mean": {sn: [round(float(v), 3) for v in g6["mean"][:, j6]]
                                     for j6, sn in enumerate(g6["sensors"])},
                            "meta": {"label": f"DSV{dv} 골든 ({g6['n']}장, 검증 저장)",
                                     "source": "golden_compare"},
                            "created": __import__("datetime").datetime.now()
                            .isoformat(timespec="seconds")}
                    (sd / f"golden_{name6}.json").write_text(
                        json.dumps(doc6, ensure_ascii=False), encoding="utf-8")
                    saved.append(name6)
            out["saved"] = saved
            print(f"[gcmp] DSV별 골든 {len(saved)}건 저장: {sd}", flush=True)
        return out

    def _gstore_dir(v, create=False):
        d9 = Path(str(v or "").strip().strip('"'))
        if not str(d9) or str(d9) == ".":
            raise HTTPException(400, "저장소 폴더를 입력하세요")
        if create:
            d9.mkdir(parents=True, exist_ok=True)
        if not d9.is_dir():
            raise HTTPException(400, f"저장소 폴더가 없습니다: {d9}")
        return d9

    TSJOB = {"state": "idle", "progress": "", "result": None, "error": None}

    # ── 재시작 영속화: 마지막 실행 입력 + 결과를 디스크에 보관(앱 교체 시 재계산 방지) ──
    #    상태 파일은 앱 폴더 내부(상대 위치 — 앱과 함께 이동), 결과는 저장소 캐시 폴더.
    STATE_FP = Path(__file__).resolve().parent / "avgwave_state.json"

    def _state_load():
        try:
            return json.loads(STATE_FP.read_text(encoding="utf-8"))
        except Exception:                              # noqa: BLE001
            return {}

    import threading as _thr9
    STATE_LOCK = _thr9.RLock()                         # 읽기-수정-쓰기 직렬화(동시 요청 유실 방지)

    def _state_save(key, val):
        try:
            with STATE_LOCK:
                st9 = _state_load()
                st9[key] = val
                STATE_FP.write_text(json.dumps(st9, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        except Exception as e9:                        # noqa: BLE001 — 못 써도 죽지 않는다
            print(f"[state] 상태 저장 실패(무시): {e9}", flush=True)

    def _pkl_dump(fp9, obj9):
        try:
            import pickle
            fp9.write_bytes(pickle.dumps(obj9, protocol=4))
            print(f"[state] 결과 보관: {fp9}", flush=True)
        except Exception as e9:                        # noqa: BLE001
            print(f"[state] 결과 보관 실패(무시): {e9}", flush=True)

    def _cache_root(cache9, store9):
        """캐시 폴더: 지정하면 그 경로, 비우면 골든 저장소 안(기존 동작)."""
        c9 = str(cache9 or "").strip().strip('"')
        if not c9:
            return Path(store9)
        p9 = Path(c9)
        try:
            p9.mkdir(parents=True, exist_ok=True)
        except Exception as e9:                        # noqa: BLE001
            print(f"[cache] 캐시 폴더를 만들 수 없어 저장소를 사용합니다: {e9}", flush=True)
            return Path(store9)
        return p9

    def _ref_cache_dir(store9, gfile9, with_recipe, cache9=None):
        """저장 골든 파일로 결과 캐시 폴더 경로 재계산(실행 시와 동일 규칙)."""
        import hashlib
        fp9 = Path(store9) / str(gfile9)
        if not fp9.is_file():
            return None
        ref9 = json.loads(fp9.read_text(encoding="utf-8"))
        if ref9.get("avgwave_golden") != 1:
            return None
        h9 = hashlib.md5(json.dumps(ref9.get("mean"), sort_keys=True)
                         .encode()).hexdigest()[:8]
        tag9 = (f"{ref9.get('name')}_{ref9.get('recipe') or 'all'}_{h9}" if with_recipe
                else f"{ref9.get('name')}_{h9}")
        sub9 = "tscan_cache" if with_recipe else "sdiff_cache"
        return (_cache_root(cache9, store9) / sub9
                / re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", tag9))

    def _sd_try_restore():
        if SDJOB["state"] != "idle":
            return
        st9 = _state_load().get("sd") or {}
        try:
            cd9 = _ref_cache_dir(st9.get("store") or "", st9.get("gfile") or "", False,
                                 st9.get("cache"))
            pk9 = (cd9 / "result.pkl") if cd9 else None
            if not pk9 or not pk9.is_file():
                return
            import pickle
            d9 = pickle.loads(pk9.read_bytes())
            if d9.get("calc_ver") != CALC_VER:
                print(f"[sdiff] 저장된 결과가 옛 계산 버전({d9.get('calc_ver') or '표기 없음'}"
                      f" ≠ {CALC_VER}) — 복원하지 않습니다. [스텝 차이 집계]를 다시 "
                      "실행하세요(일별 캐시는 재사용되어 빠릅니다)", flush=True)
                return
            SDD.clear()
            SDD.update({k9: d9[k9] for k9 in ("ids", "dates", "arr", "sensors",
                                              "usteps", "daymu", "days_u", "thr")})
            for k9 in ("io_disp", "io_disp_names", "id2src", "rmean", "steps_ref",
                       "excl_sensors"):
                if k9 in d9:
                    SDD[k9] = d9[k9]
            r9 = d9["result"]
            r9["disk_restored"] = True
            SDJOB.update(state="done", result=r9, error=None, progress="")
            print(f"[sdiff] 디스크 복원 — 웨이퍼 {r9.get('n_wafers')}장 (재계산 없음)", flush=True)
        except Exception as e9:                        # noqa: BLE001
            print(f"[sdiff] 디스크 복원 실패(무시): {e9}", flush=True)

    def _ts_try_restore():
        if TSJOB["state"] != "idle" or WST.get("tscan"):
            return
        st9 = _state_load().get("ts") or {}
        try:
            cd9 = _ref_cache_dir(st9.get("store") or "", st9.get("gfile") or "", True,
                                 st9.get("cache"))
            pk9 = (cd9 / "result.pkl") if cd9 else None
            if not pk9 or not pk9.is_file():
                return
            import pickle
            d9 = pickle.loads(pk9.read_bytes())
            if d9.get("calc_ver") != CALC_VER:
                print(f"[tscan] 저장된 결과가 옛 계산 버전({d9.get('calc_ver') or '표기 없음'}"
                      f" ≠ {CALC_VER}) — 복원하지 않습니다. 다시 실행하세요", flush=True)
                return
            WST["tscan"] = d9["tscan"]
            r9 = d9["result"]
            r9["disk_restored"] = True
            TSJOB.update(state="done", result=r9, error=None, progress="")
            print(f"[tscan] 디스크 복원 — 카드 {len(r9.get('cards') or [])}건 (재계산 없음)", flush=True)
        except Exception as e9:                        # noqa: BLE001
            print(f"[tscan] 디스크 복원 실패(무시): {e9}", flush=True)

    def _tsprog(msg):
        TSJOB["progress"] = msg
        print(f"[tscan] {msg}", flush=True)

    def _stems_by_day(per, per_day=0):
        """로드 전에 날짜 태그로 일자→(폴더,스템) 매핑 — 전수조사용 스트리밍 준비.

        per_day=0 이면 전수(기본), >0 이면 일별 상한. 태그 없는 파일은 제외하고
        수를 보고한다(태그는 select_stems 의 _convert_logs_in 이 생성).
        반환: (date -> [(folder, stem)], 전체 수, 태그 있는 수)
        """
        day_map, n_seen, n_dated = {}, 0, 0
        for fol, selx in per:
            stems9 = selx
            if stems9 is None:
                try:
                    from io_recon_tcn.data import _data_files, _resolve_existing_dir_ci
                    stems9 = [f9.stem for f9 in _data_files(_resolve_existing_dir_ci(fol))]
                except Exception:                      # noqa: BLE001
                    stems9 = [f9.stem for f9 in Path(fol).iterdir()
                              if f9.suffix.lower() in (".csv", ".log", ".tsv")]
            for st9 in stems9:
                n_seen += 1
                try:
                    d9 = S._end_ts_and_date(st9)[1] or ""
                except Exception:                      # noqa: BLE001
                    d9 = ""
                if len(d9) < 10:
                    continue
                n_dated += 1
                lst9 = day_map.setdefault(d9, [])
                if per_day and len(lst9) >= per_day:
                    continue
                lst9.append((str(fol), st9))
        return day_map, n_seen, n_dated

    def _load_day(pairs):
        """하루치 (폴더,스템) 목록 로드 — 폴더별 묶음. 스트리밍 단위(메모리=하루치)."""
        by_f = {}
        for fol9, st9 in pairs:
            by_f.setdefault(fol9, []).append(st9)
        recs = []
        for fol9, st9s in by_f.items():
            r9s, _x9, _y9, _z9 = S._load_aligned(Path(fol9), stems=st9s)
            recs += r9s
        return recs

    @app.post("/api/trendscan")
    def api_trendscan(payload: dict = Body(...)):
        """추세 스캔(R1~R7): 지정 저장 골든 기준으로 기간 전체를 하루씩 누적 비교.

        일평균 파형 + 웨이퍼 발생비율 병행. CNN 기대값은 사용하지 않는다(설계 결정).
        """
        wfs = split_folders(payload.get("folder"))
        if not wfs:
            raise HTTPException(400, "폴더를 입력하세요")
        missing = [str(w9) for w9 in wfs if not w9.is_dir()]
        if missing:
            raise HTTPException(400, "폴더가 없습니다: " + " ; ".join(missing))
        store = _gstore_dir(payload.get("store"))
        gfile = str(payload.get("gfile") or "").strip()
        fp = store / gfile
        if not gfile or not fp.is_file():
            raise HTTPException(400, "기준 골든 파일을 선택하세요")
        ref = json.loads(fp.read_text(encoding="utf-8"))
        if ref.get("avgwave_golden") != 1:
            raise HTTPException(400, "기준 파일이 저장 골든 형식이 아닙니다")
        # ★ 긴 공정(수천 tp) 대응: 분석 길이 상한 600 — 기준·일평균을 같은 길이로
        #   리샘플한다. 안 하면 일평균 캐시가 240일×60센서×3000pt(수 GB)로 폭발.
        #   600pt 면 스텝 형상·볼록 위치가 충분히 보존된다(표시용 리샘플과 동일 방식).
        L_full = int(ref["len"])
        L = min(L_full, 600)
        rmean = {sn: (_resample(np.asarray(v, dtype=np.float64), L)
                      if L != L_full else np.asarray(v, dtype=np.float64))
                 for sn, v in ref["mean"].items()}
        per_day = int(payload.get("per_day") or 0)    # 0 = 전수조사(기본)
        per, n_all, n_sel, avail = select_stems_multi(S, wfs, payload.get("dsv"),
                                                      payload.get("recipe"), payload.get("months"),
                                                      payload.get("eqp"))
        if any(x[1] is not None for x in per) and n_sel == 0:
            raise HTTPException(400, f"필터를 통과한 파일이 없습니다 (전체 {n_all}장). "
                                     + _avail_text(avail))
        import datetime as _dt
        import hashlib
        # ★ 전수조사 스트리밍: 하루치씩 읽어 집계하고 버린다(메모리=하루치).
        #   일별 집계는 저장소의 tscan_cache/ 에 저장 — 재실행 시 바뀐 날만 다시 읽는다.
        day_map, n_seen, n_dated = _stems_by_day(per, per_day)
        if n_dated == 0:
            raise HTTPException(400, f"날짜 태그가 있는 파일이 없습니다(전체 {n_seen}장). "
                                     "웨이퍼 분석 탭에서 이 폴더를 한 번 불러오면 태깅됩니다")
        if n_dated < n_seen:
            print(f"[tscan] 날짜 태그 없는 {n_seen - n_dated}장 제외", flush=True)
        dates = sorted(day_map)
        if len(dates) < 4:
            raise HTTPException(400, f"날짜가 {len(dates)}일뿐입니다 — 추세 스캔은 4일 이상 필요")
        n_total = sum(len(v) for v in day_map.values())
        ref_hash = hashlib.md5(json.dumps(ref.get("mean"), sort_keys=True)
                               .encode()).hexdigest()[:8]
        cache_dir = (_cache_root(payload.get("cache"), store) / "tscan_cache"
                     / re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_",
                              f"{ref.get('name')}_{ref.get('recipe') or 'all'}_{ref_hash}"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[tscan] 기준 '{ref.get('name')}' — {dates[0]}~{dates[-1]} {len(dates)}일 · "
              f"{'전수 ' if not per_day else f'일별 {per_day}장 상한 '}{n_total}장 "
              f"(집계 캐시: {cache_dir})", flush=True)

        if TSJOB["state"] == "running":
            raise HTTPException(409, "추세 스캔이 이미 진행 중입니다 — 완료 후 다시 실행하세요")

        def _tsrun():
            def _sinfo(rec):
                stem = str(rec["wafer_id"])
                rc9 = ""
                try:
                    rc9 = str(S._recipe_display(S._RECIPE_TAGS.get(stem, "")) or "")
                except Exception:                          # noqa: BLE001
                    pass
                if not rc9:
                    try:
                        rc9 = S._recipe_display(S._result_for(rec).get("recipe") or "") or ""
                    except Exception:                      # noqa: BLE001
                        pass
                return rc9
            sensors = [sn for sn in ref.get("sensors") or [] if sn in rmean]
            feat = {}
            daymean_cache = {}
            n_hit = n_calc = 0
            last_steps = None
            for di9, d9 in enumerate(dates):
                pairs = day_map[d9]
                hh = hashlib.md5((";".join(sorted(st for _f, st in pairs))).encode()).hexdigest()[:12]
                cf = cache_dir / f"{d9}.npz"
                day = None
                if cf.is_file():                           # 캐시 재사용(스템 목록 동일할 때만)
                    try:
                        z9 = np.load(cf, allow_pickle=False)
                        meta9 = json.loads(bytes(z9["meta"]).decode("utf-8"))
                        if meta9.get("hash") == hh:
                            day = {"sensors": meta9["sensors"], "n": meta9["n"],
                                   "flags": meta9["flags"], "mu": z9["mu"], "sg": z9["sg"],
                                   "steps": z9["steps"]}
                    except Exception as e9:                # noqa: BLE001
                        if n_calc + n_hit < 3:             # 원인 가시화(스팸 방지 3회 한정)
                            print(f"[tscan] 캐시 판독 실패({d9}): {type(e9).__name__} {e9}",
                                  flush=True)
                        day = None
                if day is None:
                    recs = _load_day(pairs)
                    mats = {}
                    steps_rs = None
                    for rec in recs:
                        if ref.get("recipe"):
                            rc9 = _sinfo(rec)
                            if rc9 and rc9 != ref["recipe"]:
                                continue
                        arr9 = np.asarray(rec["sensors"], dtype=np.float64)
                        cols9 = [str(c) for c in (rec.get("sensor_cols") or [])]
                        T9 = arr9.shape[0]
                        idx9 = np.clip(np.round(np.linspace(0, T9 - 1, L)).astype(int), 0, T9 - 1)
                        if steps_rs is None:
                            try:
                                steps_rs = np.asarray(rec.get("step"), dtype=float)[idx9].astype(int)
                            except Exception:              # noqa: BLE001
                                steps_rs = np.zeros(L, dtype=int)
                        for sn in sensors:
                            if sn in cols9:
                                mats.setdefault(sn, []).append(arr9[idx9, cols9.index(sn)])
                    del recs                               # 스트리밍: 하루치 즉시 반환
                    if not mats:
                        continue
                    usteps9 = sorted(set(int(v) for v in steps_rs))
                    sens_d = sorted(mats)
                    mu_a = np.zeros((len(sens_d), L)); sg_a = np.zeros((len(sens_d), L))
                    flags_d = {}
                    n_w = max(len(v) for v in mats.values())
                    for si9, sn in enumerate(sens_d):
                        M = np.vstack(mats[sn])
                        mu_a[si9] = M.mean(axis=0)
                        sg_a[si9] = M.std(axis=0)
                        fl9 = max(1e-6, 1e-3 * float(np.mean(np.abs(rmean[sn]))))
                        for w9 in mats[sn]:
                            for f9 in _step_trend(w9, rmean[sn], np.maximum(sg_a[si9], fl9),
                                                  steps_rs, usteps9):
                                k9 = f"{sn}|{f9['step']}"
                                flags_d[k9] = flags_d.get(k9, 0) + 1
                    day = {"sensors": sens_d, "n": n_w, "flags": flags_d,
                           "mu": mu_a, "sg": sg_a, "steps": steps_rs}
                    try:                                   # 일별 집계 저장(재실행 가속)
                        meta9 = json.dumps({"hash": hh, "sensors": sens_d, "n": n_w,
                                            "flags": flags_d}, ensure_ascii=False)
                        np.savez_compressed(cf, mu=mu_a.astype(np.float32),
                                        sg=sg_a.astype(np.float32), steps=steps_rs,
                                            meta=np.frombuffer(meta9.encode("utf-8"),
                                                               dtype=np.uint8))
                    except Exception as e9:                # noqa: BLE001
                        print(f"[tscan] 캐시 저장 실패({d9}): {e9}", flush=True)
                    n_calc += 1
                else:
                    n_hit += 1
                if di9 and di9 % 5 == 0:
                    _tsprog(f"{di9}/{len(dates)}일 (캐시 {n_hit} · 신규 {n_calc})")
                # 일별 집계 → 피처
                steps_rs = np.asarray(day["steps"]).astype(int)
                last_steps = [int(v) for v in steps_rs]
                usteps9 = sorted(set(int(v) for v in steps_rs))
                dm = daymean_cache.setdefault(d9, {})
                for si9, sn in enumerate(day["sensors"]):
                    mu_d = np.asarray(day["mu"][si9], dtype=np.float64)
                    sg_d = np.asarray(day["sg"][si9], dtype=np.float64)
                    dm[sn] = [round(float(v), 3) for v in mu_d]
                    if _is_ctrl_sensor(sn):            # Vessel 추종형·H라인 — 판정 제외
                        continue
                    if day["n"] < 3:                   # 1~2장짜리 날: σ 추정 불가 —
                        continue                       #   가짜 폭주 판정 방지(곡선만 보관)
                    sc_all = max(1e-6, float(np.median(sg_d)))
                    ref_sc = 1e-3 * float(np.mean(np.abs(rmean[sn])))
                    for st9 in usteps9:
                        if st9 == 0:                   # 스텝0 = 이전 공정 이월 구간 — 분석 제외(사용자 결정)
                            continue
                        m9 = _erode_mask(steps_rs == st9)
                        if int(m9.sum()) < 6:
                            continue
                        resid = mu_d[m9] - rmean[sn][m9]
                        sc9 = max(1e-6, float(np.median(sg_d[m9])), 0.05 * sc_all, ref_sc)
                        loc_r = resid - float(np.median(resid))
                        shp = _resample(loc_r, 16)
                        nrm = float(np.linalg.norm(shp))
                        # ★ 모양 유효성: 레벨 제거 후 남는 구조가 σ의 절반 미만이면
                        #   '모양 없음'(None) — 평평한 드리프트의 노이즈끼리 코사인을
                        #   비교해 페어링이 거부되는 버그 방지(2만장 페어링 감사에서 발견)
                        has_shape = float(np.std(loc_r)) >= 0.5 * sc9
                        fd = feat.setdefault((sn, st9), {})
                        fd[d9] = {"level": round(float(np.mean(resid)) / sc9, 2),
                                  "bump": _bump_feat(resid, sc9),
                                  "shape": ([round(float(v) / nrm, 3) for v in shp]
                                            if has_shape and nrm > 1e-9 else None),
                                  "ratio": round(day["flags"].get(f"{sn}|{st9}", 0)
                                                 / max(1, day["n"]), 2)}
            _tsprog(f"일별 집계 완료 — 캐시 {n_hit}일 · 신규 {n_calc}일, 패턴 판정 중")
            # ── 전 센서×스텝 통계 요약(카드 유무와 무관 — '전부 봤다'의 수치 증거) ──
            stats_rows = []
            for (sn9, st9), by_d9 in feat.items():
                ls = [(d9x, f9x["level"]) for d9x, f9x in by_d9.items()]
                if not ls:
                    continue
                lv9 = [v for _d, v in ls]
                n9 = len(lv9)
                mean9 = sum(lv9) / n9
                sd9 = (sum((v - mean9) ** 2 for v in lv9) / n9) ** 0.5
                tstat = mean9 / (sd9 / (n9 ** 0.5)) if sd9 > 1e-9 else 0.0
                worst = max(ls, key=lambda x9: abs(x9[1]))
                sig_days = sum(1 for v in lv9 if abs(v) >= 1.5)
                rts = [f9x.get("ratio") or 0.0 for f9x in by_d9.values()]
                bmax = max((abs(f9x["bump"]["A"]) for f9x in by_d9.values()
                            if f9x.get("bump")), default=0.0)
                stats_rows.append({"sensor": sn9, "step": st9, "n_days": n9,
                                   "mean_lv": round(mean9, 2), "max_lv": round(worst[1], 1),
                                   "worst_day": worst[0],
                                   "sig_days": sig_days,
                                   "sig_pct": round(sig_days / n9 * 100, 1),
                                   "t": round(tstat, 1),
                                   "ratio_avg": round(sum(rts) / max(1, len(rts)), 3),
                                   "ratio_max": round(max(rts) if rts else 0.0, 2),
                                   "bump_max": round(bmax, 1)})
            stats_rows.sort(key=lambda r9x: -abs(r9x["max_lv"]))
            cards = trendscan_cards(feat, dates)
            pgroups = pattern_groups(cards)
            # ★ 간접 추론(사용자 규칙): 유입 T2(InnerBlock)·T3(BackFill)가 정상인데
            #   배출 T4(Outlet)만 이상이면 — Vessel 내부(추종 제어라 직접 측정 불가)
            #   문제로 '추정'한다. 근거 카드를 연결해 소견으로 남긴다.
            import datetime as _dtv
            vessel_notes = []
            def _ov(cd1, a9, b9):
                s1 = _dtv.date.fromisoformat(cd1["start"])
                e1 = _dtv.date.fromisoformat(str(cd1.get("end") or cd1["start"]))
                return not (e1 < a9 or s1 > b9)
            for ci9, cd in enumerate(cards):
                if "scco2outlet" not in cd["sensor"].lower():
                    continue
                a9 = _dtv.date.fromisoformat(cd["start"]) - _dtv.timedelta(days=3)
                b9 = _dtv.date.fromisoformat(str(cd.get("end") or cd["start"])) + _dtv.timedelta(days=3)
                inflow_bad = any(("innerblock" in c2["sensor"].lower()
                                  or "backfill" in c2["sensor"].lower())
                                 and _ov(c2, a9, b9) for c2 in cards)
                if not inflow_bad:
                    vessel_notes.append({
                        "card_idx": ci9, "period": f"{cd['start']}~{cd.get('end', cd['start'])}",
                        "text": (f"{cd['start']}~{cd.get('end', cd['start'])} 기간 배출측 "
                                 f"T4({cd['sensor']}) {cd['kind']} 이상인데 유입측 T2(InnerBlock)·"
                                 f"T3(BackFill)는 정상 — 원인 확정은 불가하나 Vessel 내부"
                                 f"(추종 제어로 직접 판정 제외 구간) 문제로 추정")})
            # 배관 연관 사슬(계획 A3): 시작일 ±3일 클러스터 → 배관 구간 순 정렬,
            #   기점 = 최초 시작(최상류 아님 — LeakDetect 교훈)
            from io_recon_tcn import rca_causal as RC
            import datetime as _dt2
            segidx = {seg: i9 for i9, (seg, _p9) in enumerate(RC._SEGMENTS)}
            chains = []
            used = set()
            order = sorted(range(len(cards)), key=lambda i9: cards[i9]["start"])
            for i9 in order:
                if i9 in used:
                    continue
                d0 = _dt2.date.fromisoformat(cards[i9]["start"])
                grp = [i9]; used.add(i9)
                for k9 in order:
                    if k9 in used:
                        continue
                    if abs((_dt2.date.fromisoformat(cards[k9]["start"]) - d0).days) <= 3:
                        grp.append(k9); used.add(k9)
                if len(grp) >= 2:
                    grp.sort(key=lambda x9: (segidx.get(RC.segment_of(cards[x9]["sensor"]), 99),
                                             cards[x9]["start"]))
                    chains.append({"origin": min(grp, key=lambda x9: cards[x9]["start"]),
                                   "members": grp,
                                   "start": cards[min(grp, key=lambda x9: cards[x9]['start'])]["start"],
                                   "path": [f"{cards[x9]['sensor']}(step{cards[x9]['step']}·"
                                            f"{RC.segment_of(cards[x9]['sensor'])})" for x9 in grp]})
            # 타임라인 스트립: 센서별 일 강도(레벨·볼록 절대 최대)
            strip = {}
            for (sn, st9), by_d in feat.items():
                row = strip.setdefault(sn, {})
                for d9, f9 in by_d.items():
                    v9 = max(abs(f9["level"]), abs(f9["bump"]["A"]) if f9["bump"] else 0.0)
                    row[d9] = max(row.get(d9, 0.0), round(v9, 1))
            # 재발 매칭(R10): 저장소의 과거 patterns_*.json 과 서명 비교 → 같은 문제로 묶기
            def _sig(cd):
                g9 = {"sensor": cd["sensor"], "step": cd["step"], "kind": cd["kind"],
                      "dir": cd.get("direction", "")}
                if cd.get("loc"):
                    g9["c_rel"] = round(cd["loc"]["c"] / max(1, L), 3)
                if cd.get("shape"):
                    g9["shape"] = cd["shape"]
                return g9

            def _sig_match(a9, b9):
                if (a9["sensor"], a9["step"], a9["kind"], a9["dir"]) !=                (b9["sensor"], b9["step"], b9["kind"], b9["dir"]):
                    return False
                cs9 = _shape_cos(a9.get("shape"), b9.get("shape"))
                if cs9 is not None and cs9 < 0.75:
                    return False                           # 모양이 다르면 다른 문제
                if "c_rel" in a9 and "c_rel" in b9 and abs(a9["c_rel"] - b9["c_rel"]) > 0.08:
                    return False                           # 볼록 위치가 다르면 다른 문제
                return True
            period = f"{dates[0]}_{dates[-1]}"
            for fp9 in sorted(store.glob("patterns_*.json")):
                try:
                    doc9 = json.loads(fp9.read_text(encoding="utf-8"))
                    if doc9.get("avgwave_pattern") != 1 or doc9.get("period") == period:
                        continue
                    # 기간 겹침 가드: 겹치는 스캔의 자기 패턴을 '재발' 오인 방지
                    p0, p1 = doc9["period"].split("_")
                    if not (p1 < dates[0] or p0 > dates[-1]):
                        continue
                    for old9 in doc9.get("sigs") or []:
                        for cd in cards:
                            if _sig_match(_sig(cd), old9):
                                cd.setdefault("recur", []).append(
                                    {"period": doc9["period"].replace("_", "~"),
                                     "state": old9.get("state", "")})
                except Exception:                          # noqa: BLE001
                    continue
            n_rec = sum(1 for cd in cards if cd.get("recur"))
            if n_rec:
                print(f"[tscan] 과거 패턴 재발 {n_rec}건 매칭", flush=True)
            # 이번 스캔 패턴을 라이브러리에 축적(같은 기간은 덮어씀)
            try:
                sigs = []
                for cd in cards:
                    g9 = _sig(cd)
                    g9["state"] = cd.get("state", "")
                    g9["start"] = cd.get("start", "")
                    sigs.append(g9)
                (store / f"patterns_{period}.json").write_text(
                    json.dumps({"avgwave_pattern": 1, "period": period,
                                "ref": ref.get("name"), "sigs": sigs},
                               ensure_ascii=False), encoding="utf-8")
            except Exception as e:                         # noqa: BLE001
                print(f"[tscan] 패턴 저장 실패(계속 진행): {e}", flush=True)
            WST["tscan"] = {"daymean": daymean_cache, "ref": ref, "dates": dates,
                            "steps": last_steps,
                            "cards": cards, "chains": chains, "stats": stats_rows,
                            "groups": pgroups, "vessel_notes": vessel_notes,
                            "ref_rs": {sn: [round(float(v), 3) for v in rmean[sn]]
                                       for sn in rmean}}   # 분석 길이(≤600) 기준 파형
            print(f"[tscan] 완료 — 패턴 카드 {len(cards)}건 · 사슬 {len(chains)}개", flush=True)
            scan_cond = " · ".join(f"{k9}={payload.get(k9)}" for k9 in
                                   ("eqp", "dsv", "recipe", "months") if payload.get(k9))
            TSJOB["result"] = {"dates": dates, "sensors": sensors, "cards": cards,
                               "strip": strip, "chains": chains, "stats": stats_rows,
                               "groups": pgroups, "vessel_notes": vessel_notes, "cond": scan_cond,
                               "vessel_notes": vessel_notes,
                               "ref": {"name": ref.get("name"), "recipe": ref.get("recipe")}}
            TSJOB["state"] = "done"
            _pkl_dump(cache_dir / "result.pkl",
                      {"calc_ver": CALC_VER, "tscan": WST["tscan"],
                       "result": TSJOB["result"]})
            _tsprog(f"완료 — 패턴 카드 {len(cards)}건 · 사슬 {len(chains)}개")

        TSJOB.update({"state": "running", "result": None, "error": None})
        _state_save("ts", {"folder": str(payload.get("folder") or ""),
                           "eqp": str(payload.get("eqp") or ""),
                           "dsv": str(payload.get("dsv") or ""),
                           "recipe": str(payload.get("recipe") or ""),
                           "months": str(payload.get("months") or ""),
                           "per_day": payload.get("per_day") or 0,
                           "store": str(store), "gfile": gfile,
                           "cache": str(payload.get("cache") or "")})
        _install_parse_cache(S)                        # 캐시 폴더 변경 즉시 반영(전 탭)
        _tsprog(f"시작 — {dates[0]}~{dates[-1]} {len(dates)}일 · {n_total}장")

        def _tsguard():
            try:
                _tsrun()
            except Exception as e9:                    # noqa: BLE001
                import traceback as _tb
                _tb.print_exc()
                TSJOB["state"] = "error"
                TSJOB["error"] = str(e9)
        threading.Thread(target=_tsguard, daemon=True).start()
        return {"started": True, "days": len(dates), "n_total": n_total}

    @app.get("/api/trendscan_status")
    def api_trendscan_status():
        out = {"state": TSJOB["state"], "progress": TSJOB["progress"],
               "error": TSJOB["error"]}
        if TSJOB["state"] == "done":
            out["result"] = TSJOB["result"]
        return out


    @app.post("/api/ts_narrate")
    def api_ts_narrate(payload: dict = Body(...)):
        """추세 스캔 종합: 상위 카드 비교 이미지 → VLM 시각 확인 → LLM 서사(+토론)."""
        ts = WST.get("tscan")
        if not ts:
            raise HTTPException(400, "먼저 추세 스캔을 실행하세요")
        from io_recon_tcn import rca_causal as RC
        cards, chains = ts["cards"], ts.get("chains") or []
        topn = max(1, min(12, int(payload.get("top") or 8)))
        use_vlm = bool(payload.get("vlm", True))
        vlm_notes = []
        if use_vlm:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import tempfile
                for ci, cd in enumerate(cards[:topn]):
                    sn = cd["sensor"]
                    ds = [d for d, v in zip(ts["dates"], cd.get("series") or [])
                          if v is not None]
                    pick = [d for d in [ds[0] if ds else None,
                                        ds[len(ds) // 2] if ds else None,
                                        ds[-1] if ds else None] if d]
                    ref_v = (ts.get("ref_rs") or {}).get(sn) or ts["ref"]["mean"].get(sn)
                    if not ref_v or not pick:
                        continue
                    # ★ 잔차 뷰: 원파형 겹침은 다이내믹 레인지에 저진폭 패턴이 묻혀
                    #   VLM(과 사람 눈)이 못 본다(완만-W 실험 실증). 일평균-기준 잔차를
                    #   그리고 해당 스텝을 음영 처리 — 엔진이 보는 것과 같은 공간.
                    rv = np.asarray(ref_v, dtype=float)
                    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)
                    ax.axhline(0, color="k", lw=1.2, label="REF(=0)")
                    for d9 in dict.fromkeys(pick):
                        cv = ts["daymean"].get(d9, {}).get(sn)
                        if cv and len(cv) == len(rv):
                            ax.plot(np.asarray(cv, dtype=float) - rv, "--", lw=1.2, label=d9)
                    stp9 = ts.get("steps")
                    if stp9 and len(stp9) == len(rv):
                        sarr = np.asarray(stp9)
                        w9 = np.where(sarr == cd["step"])[0]
                        if w9.size:
                            ax.axvspan(int(w9[0]), int(w9[-1]), color="orange", alpha=0.15)
                    ax.set_title(f"{sn} step{cd['step']} residual", fontsize=9)
                    ax.legend(fontsize=7)
                    fp9 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    fig.savefig(fp9.name, bbox_inches="tight"); plt.close(fig)
                    print(f"[tsnar] VLM 확인 {ci+1}/{min(topn, len(cards))}: {sn}", flush=True)
                    note = RC.narrate_pattern(cd, fp9.name)
                    vlm_notes.append({"i": ci, "sensor": sn, "step": cd["step"],
                                      "kind": cd["kind"], "note": note})
                    try:
                        os.unlink(fp9.name)
                    except OSError:
                        pass
            except Exception as e:                     # noqa: BLE001
                print(f"[tsnar] VLM 단계 건너뜀: {e}", flush=True)
        stats_text = ""
        if ts.get("stats"):
            hdr9 = "sensor,step,평균레벨σ,최대레벨σ,최악일,유의일수%,평균발생률,최대볼록σ,t"
            stats_text = hdr9 + "\n" + "\n".join(
                f"{r9x['sensor']},{r9x['step']},{r9x['mean_lv']},{r9x['max_lv']},"
                f"{r9x['worst_day']},{r9x['sig_pct']}%,{r9x['ratio_avg']},"
                f"{r9x['bump_max']},{r9x['t']}" for r9x in ts["stats"][:20])
        cards_text = "\n".join(
            f"{cd['sensor']} step{cd['step']} {cd['kind']}({cd.get('direction','')}) "
            f"{cd['start']}~ 상태 {cd['state']}"
            + (f" 진폭 {cd['amp']}σ" if cd.get("amp") is not None else "")
            + ("".join(f" | {e['date']} {e['kind']} {e['delta']:+.1f}σ {e['label']}"
                       for e in cd.get("events") or []))
            + ((" | ★재발: 과거 " + ", ".join(r9["period"] for r9 in cd["recur"])
                + " 기간에 같은 서명") if cd.get("recur") else "")
            for cd in cards[:30])
        chains_text = "\n".join(
            f"사슬 {k+1} (시작 {ch['start']}, 기점 후보={cards[ch['origin']]['sensor']}): "
            + " -> ".join(ch["path"]) for k, ch in enumerate(chains))
        vlm_text = "\n".join(f"{v['sensor']} step{v['step']} ({v['kind']}): {v['note']}"
                              for v in vlm_notes)
        grp_text = ""
        for g9 in (ts.get("groups") or [])[:10]:
            occ9 = ", ".join(f"{o9['month'][5:]}월({o9['start'][5:]}~{str(o9['end'])[5:]} "
                             f"{o9['state']}{' ' + str(o9['amp']) + 'σ' if o9.get('amp') is not None else ''})"
                             for o9 in g9["occurrences"])
            grp_text += (f"{g9['sensor']} step{g9['step']} {g9['kind']}({g9.get('direction','')}) "
                         f"— {g9['n_occur']}회 발생: {occ9}\n")
        vtxt = "\n".join(v9["text"] for v9 in (ts.get("vessel_notes") or []))
        if vtxt:
            cards_text = "[Vessel 내부 추정 소견 — T2·T3 정상 + T4 이상]\n" + vtxt + "\n\n" + cards_text
        if grp_text:
            cards_text = "[패턴 사례 통계 — 같은 모양 페어링]\n" + grp_text + "\n" + cards_text
        if stats_text:
            cards_text = "[전 센서×스텝 통계 상위 20 — 표 밖 스텝은 유의 이탈 없음]\n" \
                + stats_text + "\n\n[패턴 카드]\n" + cards_text
        out = RC.narrate_trendscan(cards_text, chains_text, vlm_text)
        out["vlm"] = vlm_notes
        return out

    @app.post("/api/mp_narrate")
    def api_mp_narrate(payload: dict = Body(...)):
        """주요 패턴(선택 그룹) LLM 원인 분석·상황 요약."""
        gsel = payload.get("groups") or []
        if not gsel:
            raise HTTPException(400, "분석할 패턴 그룹이 없습니다 — 주요 패턴을 먼저 불러오고 선택하세요")
        ts = WST.get("tscan") or {}
        last_day = (ts.get("dates") or ["?"])[-1]
        import datetime as _dtm
        lines = []
        for g9 in gsel[:15]:
            occ = g9.get("occurrences") or []
            gaps = []
            for a9, b9 in zip(occ, occ[1:]):
                try:
                    gaps.append((_dtm.date.fromisoformat(b9["start"])
                                 - _dtm.date.fromisoformat(a9["start"])).days)
                except Exception:                      # noqa: BLE001
                    pass
            occ_t = " / ".join(
                f"{i9+1}회차 {o9['start']}~{o9.get('end', o9['start'])} 상태 {o9['state']}"
                + (f" 진폭 {o9['amp']}σ" if o9.get("amp") is not None else "")
                for i9, o9 in enumerate(occ))
            active = ""
            try:
                if occ and str(occ[-1].get("end", "")) >= str(last_day):
                    active = " ★현재 활성(스캔 말일까지 진행 중)"
            except Exception:                          # noqa: BLE001
                pass
            lines.append(
                f"{g9['sensor']} step{g9['step']} {g9['kind']}({g9.get('direction', '')}) "
                f"— {g9.get('n_occur', len(occ))}회 발생"
                + (f", 발생 간격 {gaps}일" if gaps else "") + active + "\n  " + occ_t)
        extra = "\n".join(v9["text"] for v9 in (ts.get("vessel_notes") or []))
        from io_recon_tcn import rca_causal as RC
        out = RC.narrate_mainpatterns("\n".join(lines), extra)
        return out

    @app.post("/api/ts_curves")
    def api_ts_curves(payload: dict = Body(...)):
        """추세 스캔 카드 상세: 기준 파형 + 지정 날짜들의 일평균 곡선."""
        ts = WST.get("tscan")
        if not ts:
            raise HTTPException(400, "먼저 추세 스캔을 실행하세요")
        sn = str(payload.get("sensor") or "")
        dts = [d for d in (payload.get("dates") or []) if d in ts["daymean"]]
        if sn not in (ts["ref"].get("mean") or {}):
            raise HTTPException(400, f"기준에 없는 센서: {sn}")
        ref_v = (ts.get("ref_rs") or {}).get(sn) or ts["ref"]["mean"][sn]
        return {"ref": ref_v, "ref_name": ts["ref"].get("name"),
                "curves": {d: ts["daymean"][d].get(sn) for d in dts}}

    @app.post("/api/gsave")
    def api_gsave(payload: dict = Body(...)):
        """골든 그래프 저장: 이름·레시피·센서별 평균 파형을 JSON 으로 저장소에 보관."""
        store = _gstore_dir(payload.get("store"), create=True)
        name = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", str(payload.get("name") or "").strip())
        if not name:
            raise HTTPException(400, "저장 이름을 입력하세요")
        mean = payload.get("mean") or {}
        if not mean:
            raise HTTPException(400, "저장할 파형이 없습니다")
        doc = {"avgwave_golden": 1, "name": name,
               "recipe": str(payload.get("recipe") or ""),
               "len": int(payload.get("len") or 0),
               "sensors": list(mean.keys()), "mean": mean,
               "meta": payload.get("meta") or {},
               "created": __import__("datetime").datetime.now().isoformat(timespec="seconds")}
        fp = store / f"golden_{name}.json"
        existed = fp.exists()
        fp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print(f"[gstore] 저장: {fp}" + (" (덮어씀)" if existed else ""), flush=True)
        return {"ok": True, "path": str(fp), "overwritten": existed}

    @app.get("/api/glist")
    def api_glist(store: str):
        d9 = _gstore_dir(store)
        out = []
        for fp in sorted(d9.glob("*.json")):
            try:
                doc = json.loads(fp.read_text(encoding="utf-8"))
                if doc.get("avgwave_golden") != 1:
                    continue
                out.append({"file": fp.name, "name": doc.get("name"),
                            "recipe": doc.get("recipe"), "len": doc.get("len"),
                            "n_sensors": len(doc.get("sensors") or []),
                            "label": (doc.get("meta") or {}).get("label", ""),
                            "created": doc.get("created", "")})
            except Exception:                          # noqa: BLE001
                continue
        return {"items": out}

    @app.post("/api/gload")
    def api_gload(payload: dict = Body(...)):
        d9 = _gstore_dir(payload.get("store"))
        files = payload.get("files") or []
        docs = []
        for fn in files:
            fp = d9 / str(fn)
            if not fp.is_file():
                raise HTTPException(400, f"파일이 없습니다: {fn}")
            doc = json.loads(fp.read_text(encoding="utf-8"))
            if doc.get("avgwave_golden") == 1:
                docs.append(doc)
        return {"items": docs}

    @app.post("/api/gbuild")
    def api_gbuild(payload: dict = Body(...)):
        """통합 골든 생성: 사용자 지정 '정상 구간 폴더'(양품만) 전체를 판정 없이 평균.

        DSV 구분 없이 통합 평균(레시피가 여럿이면 레시피별로 나눠 저장).
        가상센서(dT_/dP_) 포함 — 로드 계층 주입으로 자동. mean+std+steps 를 저장소에 보관.
        """
        wfs = split_folders(payload.get("folder"))
        if not wfs:
            raise HTTPException(400, "정상 구간 폴더를 입력하세요")
        missing = [str(w9) for w9 in wfs if not w9.is_dir()]
        if missing:
            raise HTTPException(400, "폴더가 없습니다: " + " ; ".join(missing))
        store = _gstore_dir(payload.get("store"), create=True)
        name = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", str(payload.get("name") or "").strip())
        if not name:
            raise HTTPException(400, "저장 이름을 입력하세요")
        per, n_all, n_sel, avail = select_stems_multi(S, wfs, payload.get("dsv"),
                                                      payload.get("recipe"), payload.get("months"),
                                                      payload.get("eqp"))
        if any(x[1] is not None for x in per) and n_sel == 0:
            raise HTTPException(400, f"필터를 통과한 파일이 없습니다 (전체 {n_all}장). "
                                     + _avail_text(avail))
        pairs = []                                     # (폴더, 스템) 전체 목록 — 날짜 태그 불필요
        for fol9, selx in per:
            st9s = selx
            if st9s is None:
                try:
                    from io_recon_tcn.data import _data_files, _resolve_existing_dir_ci
                    st9s = [f9.stem for f9 in _data_files(_resolve_existing_dir_ci(fol9))]
                except Exception:                      # noqa: BLE001
                    st9s = [f9.stem for f9 in Path(fol9).iterdir()
                            if f9.suffix.lower() in (".csv", ".log", ".tsv")]
            pairs += [(str(fol9), s9) for s9 in st9s]
        if not pairs:
            raise HTTPException(400, "읽을 수 있는 파일이 없습니다")
        print(f"[gbuild] 통합 골든 '{name}' — {len(pairs)}장 무판정 평균(폴더 {len(wfs)}곳)",
              flush=True)
        acc = {}          # recipe -> {"L","cols","sum","sum2","n","steps"}
        L_CAP = 600       # 추세 스캔 분석 길이 상한과 동일 — 메모리·형상 보존 균형
        n_read = 0
        for k0 in range(0, len(pairs), 200):           # 200장 청크 스트리밍(메모리 고정)
            recs = _load_day(pairs[k0:k0 + 200])
            for rec in recs:
                rc9 = ""
                try:
                    rc9 = str(S._recipe_display(S._RECIPE_TAGS.get(str(rec["wafer_id"]), "")
                                                or rec.get("recipe") or "") or "")
                except Exception:                      # noqa: BLE001
                    pass
                rc9 = rc9 or "unknown"
                arr9 = np.asarray(rec["sensors"], dtype=np.float64)
                cols9 = [str(c) for c in (rec.get("sensor_cols") or [])]
                if arr9.ndim != 2 or not cols9:
                    continue
                try:
                    stw0 = np.asarray(rec.get("step"), dtype=float).astype(int)
                except Exception:                      # noqa: BLE001
                    stw0 = np.zeros(arr9.shape[0], dtype=int)
                a9 = acc.get(rc9)
                if a9 is None:
                    L9 = min(arr9.shape[0], L_CAP)
                    # 캐논 스텝 격자: 첫 웨이퍼의 런 비율대로 L9 를 배분 —
                    # 이후 모든 웨이퍼를 이 격자에 '스텝 런별 워핑'해 평균한다.
                    runs0 = _step_runs(stw0)
                    T0 = max(1, arr9.shape[0])
                    canon0, pos0 = [], 0
                    for qi0, (u0, s0x, e0x) in enumerate(runs0):
                        ln0 = (L9 - pos0 if qi0 == len(runs0) - 1
                               else max(1, int(round(L9 * (e0x - s0x) / T0))))
                        ln0 = min(ln0, L9 - pos0)
                        if ln0 <= 0:
                            continue
                        canon0.append((u0, pos0, ln0))
                        pos0 += ln0
                    steps0 = np.zeros(L9, dtype=int)
                    for u0, p0, l0 in canon0:
                        steps0[p0:p0 + l0] = u0
                    a9 = acc[rc9] = {"L": L9, "cols": cols9,
                                     "sum": np.zeros((L9, len(cols9))),
                                     "sum2": np.zeros((L9, len(cols9))),
                                     "cnt": np.zeros((L9, len(cols9))),
                                     "n": 0, "steps": [int(v) for v in steps0],
                                     "canon": canon0,
                                     "io_sum": {}, "io_n": {},
                                     "ss": {}}
                try:                                   # IO 평균(밸브 개폐율) — 답지 타이밍
                    iod9 = np.asarray(rec.get("io"), dtype=np.float64)
                    ioc9 = [str(c) for c in (rec.get("io_cols") or [])]
                    if iod9.ndim == 2 and ioc9:
                        iow9 = _warp_to_canon((iod9 > 0.5).astype(np.float64),
                                              stw0, a9["canon"], a9["L"])
                        for j9x, nm9x in enumerate(ioc9):
                            v9x = iow9[:, j9x]
                            if nm9x not in a9["io_sum"]:
                                a9["io_sum"][nm9x] = np.zeros(a9["L"])
                                a9["io_n"][nm9x] = 0
                            fin9x = np.isfinite(v9x)
                            a9["io_sum"][nm9x][fin9x] += v9x[fin9x]
                            a9["io_n"][nm9x] += 1
                except Exception:                      # noqa: BLE001
                    pass
                if cols9 != a9["cols"]:                # 열 불일치 파일은 교집합 정렬
                    keep = [c for c in a9["cols"] if c in cols9]
                    if not keep:
                        continue
                    gi = [a9["cols"].index(c) for c in keep]
                    arr9 = arr9[:, [cols9.index(c) for c in keep]]
                    a9["sum"] = a9["sum"][:, gi]
                    a9["sum2"] = a9["sum2"][:, gi]
                    a9["cols"] = keep
                L9 = a9["L"]
                rs9 = _warp_to_canon(arr9, stw0, a9["canon"], L9)   # 스텝 정렬 워핑
                fin9 = np.isfinite(rs9)
                rs0 = np.where(fin9, rs9, 0.0)
                a9["sum"] += rs0
                a9["sum2"] += rs0 * rs0
                a9["cnt"] += fin9
                a9["n"] += 1
                try:
                    # 스텝 정렬 통계: 웨이퍼 '자기 스텝 평균'을 누적 — 공정 길이가
                    # 제각각이라도 오염되지 않는 σ(포인트별 σ는 전이 어긋남으로 폭발)
                    stw9 = np.asarray(rec.get("step"), dtype=float).astype(int)
                    for u9 in set(int(t9) for t9 in stw9):
                        mku9 = _erode_mask(stw9 == u9)
                        if not mku9.any():
                            continue
                        mv9 = np.asarray(rec["sensors"], dtype=np.float64)[mku9].mean(axis=0)
                        mn9 = np.asarray(rec["sensors"], dtype=np.float64)[mku9].min(axis=0)
                        s9d = a9["ss"].setdefault(int(u9), {})
                        for j9u, c9u in enumerate(cols9):
                            e9u = s9d.setdefault(c9u, [0.0, 0.0, 0, 0.0])
                            e9u[0] += float(mv9[j9u])
                            e9u[1] += float(mv9[j9u]) ** 2
                            e9u[2] += 1
                            e9u[3] += float(mn9[j9u])
                except Exception:                      # noqa: BLE001
                    pass
            n_read += len(recs)
            del recs
            if n_read and n_read % 1000 < 200:
                print(f"[gbuild] 처리 {n_read}/{len(pairs)}", flush=True)
        saved = []
        for rc9, a9 in acc.items():
            if not a9["n"]:
                continue
            c9m = np.maximum(a9["cnt"], 1)
            mean9 = a9["sum"] / c9m
            std9 = np.sqrt(np.maximum(a9["sum2"] / c9m - mean9 * mean9, 0.0))
            hole9 = a9["cnt"] < max(3, a9["n"] * 0.3)   # 표본 부족 위치는 선형 보간
            for j9h in range(mean9.shape[1]):
                bad9 = hole9[:, j9h]
                if bad9.any() and (~bad9).any():
                    xi9 = np.where(~bad9)[0]
                    mean9[bad9, j9h] = np.interp(np.where(bad9)[0], xi9, mean9[xi9, j9h])
                    std9[bad9, j9h] = float(np.median(std9[xi9, j9h]))
            try:                                       # 센서 순서 = 설치 순서(토폴로지)
                from io_recon_tcn.sensor_graph import graph_sensor_rank
                rk = graph_sensor_rank(a9["cols"], S.ENGINE.cfg)
                order = sorted(range(len(a9["cols"])),
                               key=lambda j9: rk.get(str(a9["cols"][j9]), 10 ** 6))
            except Exception:                          # noqa: BLE001
                order = list(range(len(a9["cols"])))
            sens9 = [a9["cols"][j9] for j9 in order]
            nm9 = name if len(acc) == 1 else (
                name + "_" + re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", rc9))
            ci9 = {sn: a9["cols"].index(sn) for sn in sens9}
            doc = {"avgwave_golden": 1, "name": nm9, "recipe": rc9, "len": a9["L"],
                   "sensors": sens9,
                   "mean": {sn: [round(float(v), 4) for v in mean9[:, ci9[sn]]]
                            for sn in sens9},
                   "std": {sn: [round(float(v), 4) for v in std9[:, ci9[sn]]]
                           for sn in sens9},
                   "steps": a9["steps"],
                   "io_mean": {nm9x: [round(float(v9x), 3) for v9x in
                                      a9["io_sum"][nm9x] / max(1, a9["io_n"][nm9x])]
                               for nm9x in a9["io_sum"]},
                   "step_stats": {str(u9x): {c9x: [round(e9x[0] / e9x[2], 4),
                                                   round(max(e9x[1] / e9x[2]
                                                             - (e9x[0] / e9x[2]) ** 2,
                                                             0.0) ** 0.5, 4),
                                                   round(e9x[3] / e9x[2], 4)]
                                             for c9x, e9x in d9x.items() if e9x[2] >= 3}
                                  for u9x, d9x in a9["ss"].items()},
                   "meta": {"label": "통합(무판정 전체 평균)", "unified": 1, "n": a9["n"],
                            "calc_ver": CALC_VER,
                            "folders": [str(w9) for w9 in wfs]},
                   "created": __import__("datetime").datetime.now().isoformat(timespec="seconds")}
            fp = store / f"golden_{nm9}.json"
            fp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            saved.append({"file": fp.name, "recipe": rc9, "n": a9["n"], "len": a9["L"],
                          "n_sensors": len(sens9),
                          "n_virtual": sum(1 for s9 in sens9
                                           if re.match(r"(?i)^d[tp]_", s9))})
            print(f"[gbuild] 저장: {fp} (유형 {rc9} · {a9['n']}장)", flush=True)
        if not saved:
            raise HTTPException(400, "집계된 웨이퍼가 없습니다")
        return {"ok": True, "items": saved, "n_read": n_read}

    SDJOB = {"state": "idle", "progress": "", "result": None, "error": None}
    SDD = {}                                           # 스텝 차이 서버 보관(ids/dates/arr…)

    def _sdprog(msg):
        SDJOB["progress"] = msg
        print(f"[sdiff] {msg}", flush=True)

    @app.post("/api/stepdiff")
    def api_stepdiff(payload: dict = Body(...)):
        """스텝 차이: 통합 골든 대비 모든 웨이퍼×센서×스텝 차이를 원단위로 수치화.

        백그라운드 스트리밍(하루치 로드→집계→버림) + 일별 디스크 캐시. 스텝0 제외·
        스텝 경계 침식(_erode_mask) 적용. 결과는 서버 보관 → /api/stepdiff_page 로 조회.
        """
        wfs = split_folders(payload.get("folder"))
        if not wfs:
            raise HTTPException(400, "폴더를 입력하세요")
        missing = [str(w9) for w9 in wfs if not w9.is_dir()]
        if missing:
            raise HTTPException(400, "폴더가 없습니다: " + " ; ".join(missing))
        store = _gstore_dir(payload.get("store"))
        gfile = str(payload.get("gfile") or "").strip()
        fp = store / gfile
        if not gfile or not fp.is_file():
            raise HTTPException(400, "기준 골든 파일을 선택하세요")
        ref = json.loads(fp.read_text(encoding="utf-8"))
        if ref.get("avgwave_golden") != 1:
            raise HTTPException(400, "기준 파일이 저장 골든 형식이 아닙니다")
        L_full = int(ref["len"])
        L = min(L_full, 600)
        rmean = {sn: (_resample(np.asarray(v, dtype=np.float64), L)
                      if L != L_full else np.asarray(v, dtype=np.float64))
                 for sn, v in ref["mean"].items()}
        steps_ref = None
        if ref.get("steps"):
            sr9 = np.asarray(ref["steps"], dtype=float)
            i9 = np.clip(np.round(np.linspace(0, len(sr9) - 1, L)).astype(int),
                         0, len(sr9) - 1)
            steps_ref = sr9[i9].astype(int)
        per, n_all, n_sel, avail = select_stems_multi(S, wfs, payload.get("dsv"),
                                                      payload.get("recipe"), payload.get("months"),
                                                      payload.get("eqp"))
        if any(x[1] is not None for x in per) and n_sel == 0:
            raise HTTPException(400, f"필터를 통과한 파일이 없습니다 (전체 {n_all}장). "
                                     + _avail_text(avail))
        day_map, n_seen, n_dated = _stems_by_day(per, int(payload.get("per_day") or 0))
        if n_dated == 0:
            raise HTTPException(400, f"날짜 태그가 있는 파일이 없습니다(전체 {n_seen}장). "
                                     "웨이퍼 분석 탭에서 이 폴더를 한 번 불러오면 태깅됩니다")
        dates = sorted(day_map)
        n_total = sum(len(v) for v in day_map.values())
        import hashlib
        ref_hash = hashlib.md5(json.dumps(ref.get("mean"), sort_keys=True)
                               .encode()).hexdigest()[:8]
        cache_dir = (_cache_root(payload.get("cache"), store) / "sdiff_cache"
                     / re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_",
                              f"{ref.get('name')}_{ref_hash}"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        if SDJOB["state"] == "running":
            raise HTTPException(409, "스텝 차이 집계가 이미 진행 중입니다 — 완료 후 다시 실행하세요")
        # ── 물리법칙 문서(사용자 소유) — 골든=답지 상대 판정 파라미터 ──
        from io_recon_tcn import rca_causal as _RCp
        _phys_txt, PH = _RCp.load_physics()
        IO_SEL = sorted(PH.get("io_map") or {})
        _excl_rx = None
        try:
            _excl_rx = re.compile(str(PH.get("exclude_sensors_regex") or ""))
        except Exception:                              # noqa: BLE001
            pass
        # 표시용 IO = 데이터에 실제로 있는 모든 IO(첫 파일에서 수집, 골든 io_mean 과 합집합)
        IO_DISP = []
        try:
            first_pair = day_map[dates[0]][:1]
            rec0 = _load_day(first_pair)
            if rec0:
                IO_DISP = sorted({str(c) for c in (rec0[0].get("io_cols") or [])}
                                 | set((ref.get("io_mean") or {}).keys()))
            del rec0
        except Exception as e9:                        # noqa: BLE001
            print(f"[sdiff] IO 목록 수집 실패(표시 생략): {e9}", flush=True)
        if not IO_DISP:
            IO_DISP = [str(x) for x in (PH.get("io_display")
                                        or ["9000003", "9000004", "9000009"])]
        SDJOB.update(state="running", progress="시작", result=None, error=None)
        _state_save("sd", {"folder": str(payload.get("folder") or ""),
                           "eqp": str(payload.get("eqp") or ""),
                           "dsv": str(payload.get("dsv") or ""),
                           "recipe": str(payload.get("recipe") or ""),
                           "months": str(payload.get("months") or ""),
                           "store": str(store), "gfile": gfile,
                           "cache": str(payload.get("cache") or "")})
        _install_parse_cache(S)                        # 캐시 폴더 변경 즉시 반영(전 탭)
        gv9 = (ref.get("meta") or {}).get("calc_ver")
        if gv9 != CALC_VER:
            print(f"[sdiff] ⚠ 기준 골든이 옛 버전({gv9 or '표기 없음'})입니다 — 동작은 하지만 "
                  "스텝 정렬·물리 규칙 효과를 못 받습니다. [통합 골든 생성]으로 한 번 "
                  "재생성하시길 권합니다", flush=True)
        print(f"[sdiff] 기준 '{ref.get('name')}' — {dates[0]}~{dates[-1]} {len(dates)}일 · "
              f"{n_total}장 (캐시: {cache_dir})", flush=True)

        def _sdrun():
            try:
                # Heater(SV 미상)·추종형: 그래프에는 포함(체크박스 선택 가능),
                # 규칙엔진(추림·패턴·LLM 자동 판정)에서만 제외 — 사용자 결정
                sensors = [sn for sn in (ref.get("sensors") or []) if sn in rmean]
                excl_set = {sn for sn in sensors
                            if _excl_rx and _excl_rx.search(sn)}
                # ── 골든=답지: IO 타이밍·응답·아치 기대치 사전계산 ──
                gio = ref.get("io_mean") or {}
                g_t0 = {}
                for io9 in IO_SEL:
                    if io9 in gio:
                        v9 = np.asarray(gio[io9], dtype=np.float64)
                        if len(v9) != L:
                            i9 = np.clip(np.round(np.linspace(0, len(v9) - 1, L)).astype(int),
                                         0, len(v9) - 1)
                            v9 = v9[i9]
                        on9 = np.where(v9 > 0.5)[0]
                        g_t0[io9] = int(on9[0]) if on9.size else None
                    else:
                        g_t0[io9] = None
                W_resp = max(3, int(L * float((PH.get("io_response") or {})
                                              .get("window_frac", 0.10))))
                W_pre = max(2, W_resp // 3)            # 열림 '직전' 기준(즉시 응답 대응)
                def _resp_prof(x9c, t09c):
                    """자기 열림 시점 정렬 응답 프로파일: x[t0−pre..t0+W] − x[t0−pre].
                    골든·웨이퍼를 각자 자기 t0 로 정렬하면 배경(가압 램프)은 양쪽 동일해
                    잔차에서 상쇄되고, 밸브 기여(점프)만 남는다."""
                    b9c = t09c - W_pre
                    e9c = t09c + W_resp
                    if b9c < 0 or e9c > L - 1:
                        return None                    # 창이 잘리면 판정 생략
                    seg9 = np.asarray(x9c[b9c:e9c + 1], dtype=np.float64)
                    return seg9 - seg9[0]
                ref_std0 = ref.get("std") or {}
                if gio:
                    print("[sdiff] 골든 IO 답지: " + ", ".join(
                        f"{io9}(t0={'%.0f%%' % (g_t0[io9] * 100.0 / L) if g_t0.get(io9) is not None else '미개방'})"
                        for io9 in IO_SEL if io9 in gio), flush=True)
                else:
                    print("[sdiff] ⚠ 골든에 io_mean 없음 — 골든 소스 파일에 IO 채널이 "
                          "없거나 v86 이전 골든. 물리 위반(아치·무반응·타이밍) 비활성",
                          flush=True)
                g_io_disp = {}                         # 표시 IO: 골든 스텝별 열림비율
                if gio and steps_ref is not None:
                    for io9 in IO_DISP:
                        if io9 not in gio:
                            continue
                        v9 = np.asarray(gio[io9], dtype=np.float64)
                        if len(v9) != L:
                            i9 = np.clip(np.round(np.linspace(0, len(v9) - 1, L)).astype(int),
                                         0, len(v9) - 1)
                            v9 = v9[i9]
                        g_io_disp[io9] = [round(float(np.mean(
                            v9[steps_ref == u9] > 0.5)), 2) if (steps_ref == u9).any()
                            else None
                            for u9 in sorted(set(int(t9) for t9 in steps_ref) - {0})]
                g_resp = {}                            # io -> (골든 프로파일, 센서 σ)
                for io9, m9 in (PH.get("io_map") or {}).items():
                    sn9 = m9.get("sensor")
                    t09 = g_t0.get(io9)
                    if sn9 in rmean and t09 is not None:
                        gp9 = _resp_prof(rmean[sn9], t09)
                        if gp9 is None or float(np.max(np.abs(gp9))) < 0.2:
                            continue                   # 골든에 응답 구조가 없으면 비활성
                        sgn9 = 0.5
                        if sn9 in ref_std0:
                            sv9 = np.asarray(ref_std0[sn9], dtype=np.float64)
                            sgn9 = float(np.mean(sv9))
                        g_resp[io9] = (gp9, max(sgn9, 1e-3))
                def _tail_feat(x9):
                    """tail 앵커 아치 특징: (tail, s부호, depth) — tail=스텝 끝(회복) 레벨,
                    depth=끝 레벨 대비 최대 이탈. 딥 시작 시점에 강인(첫 구간 기준 안 씀)."""
                    n9 = len(x9)
                    if n9 < 8:
                        return None
                    k9 = max(2, n9 // 10)
                    tl9 = float(np.median(x9[-k9:]))
                    j9 = int(np.argmax(np.abs(x9 - tl9)))
                    d9x = float(x9[j9] - tl9)
                    return (tl9, 1.0 if d9x >= 0 else -1.0, abs(d9x))
                g_arch = []                            # 아치 규칙별 골든 기대치
                for ar9 in (PH.get("arch_rules") or []):
                    sr9 = ar9.get("sensor")
                    if sr9 not in rmean or steps_ref is None:
                        print(f"[sdiff] 아치 규칙 비활성({sr9}): "
                              + ("골든에 이 센서 없음" if sr9 not in rmean
                                 else "골든에 스텝 정보 없음(v77+ 골든으로 재생성 필요)"),
                              flush=True)
                        continue
                    st_g9 = None
                    t0f9 = g_t0.get(str(ar9.get("open_first")))
                    if t0f9 is not None:
                        st_g9 = int(steps_ref[t0f9])
                    usteps0 = sorted(set(int(v) for v in steps_ref) - {0})
                    best9 = None
                    for u9 in usteps0:
                        mk9 = _erode_mask(steps_ref == u9)
                        if not mk9.any():
                            continue
                        f9 = _tail_feat(rmean[sr9][mk9])
                        if f9 is None:
                            continue
                        if st_g9 == u9:
                            best9 = (u9, f9)
                            break
                        if best9 is None or f9[2] > best9[1][2]:   # 폴백: 최심 스텝
                            best9 = (u9, f9)
                    if best9 is None or best9[0] == 0:
                        print(f"[sdiff] 아치 규칙 비활성({sr9}): "
                              + ("유효 스텝 없음" if best9 is None
                                 else "아치가 스텝0(이월 구간)으로 인식됨"), flush=True)
                        continue
                    u9, (gtl9, gs9, gdp9) = best9
                    mk9 = _erode_mask(steps_ref == u9)
                    sg9 = 0.0
                    e9s = ((ref.get("step_stats") or {}).get(str(int(u9))) or {}).get(sr9)
                    if e9s is not None:
                        sg9 = float(e9s[1])            # 스텝 정렬 σ(정렬 무관)
                    elif sr9 in ref_std0:
                        sv9 = np.asarray(ref_std0[sr9], dtype=np.float64)
                        if len(sv9) != L:
                            i9 = np.clip(np.round(np.linspace(0, len(sv9) - 1, L)).astype(int),
                                         0, len(sv9) - 1)
                            sv9 = sv9[i9]
                        sg9 = float(np.mean(sv9[mk9]))
                    # v98: 골든 '평균 곡선'은 웨이퍼별 밸브/경계 편차로 뭉개져 아치
                    # 기대치가 오염된다(스미어를 딥으로 오인). IO 답지에 열림 순서가
                    # 있으면 곡선 기대치를 아예 쓰지 않고 무조건 자기 정렬 모드.
                    self_mode9 = False
                    t0a9 = g_t0.get(str(ar9.get("open_first")))
                    t0b9 = g_t0.get(str(ar9.get("open_second")))
                    if t0a9 is not None and t0b9 is not None and t0a9 < t0b9:
                        self_mode9 = True
                        print(f"[sdiff] 아치 규칙 자기 정렬 모드({sr9}): "
                              "웨이퍼별 밸브 열림 시점 기준으로 판정(골든 곡선 미사용)",
                              flush=True)
                    elif gdp9 < float(ar9.get("depth_sigma_min", 2.0)) * max(sg9, 1e-6):
                        print(f"[sdiff] 아치 규칙 비활성({sr9}): 골든 step{u9} 깊이 "
                              f"{gdp9:.3f} < 문턱 "
                              f"{float(ar9.get('depth_sigma_min', 2.0)):g}×σ({sg9:.3f}) "
                              "— IO 답지(열림 순서)도 없어 자기 정렬 불가", flush=True)
                        continue
                    g_arch.append({"sensor": sr9, "step": int(u9),
                                   "open_first": str(ar9.get("open_first")),
                                   "open_second": str(ar9.get("open_second")),
                                   "g_tail": round(gtl9, 3), "g_sign": gs9,
                                   "depth_g": round(gdp9, 3),
                                   "sig_g": max(sg9, 1e-6),
                                   "self_mode": self_mode9,
                                   "r_min": float(ar9.get("recover_ratio_min", 0.5)),
                                   "d_sig": float(ar9.get("depth_sigma_min", 2.0))})
                if g_arch:
                    print(f"[sdiff] 아치 기대치: " + " · ".join(
                        f"{a9['sensor']} step{a9['step']} 깊이 {a9['depth_g']}"
                        for a9 in g_arch), flush=True)
                min_rise = float((PH.get("io_response") or {}).get("min_rise_ratio", 0.3))
                tol_t0 = float((PH.get("io_timing") or {}).get("tol_frac", 0.02))
                phys_daily = {}                        # date -> 물리 카운터
                usteps = None
                if steps_ref is not None:
                    usteps = sorted(set(int(v) for v in steps_ref) - {0})   # 스텝0 제외
                # ── 정렬 결함 수정: 공정 길이가 제각각이라 전체 균등 리샘플로는 스텝
                #    경계가 어긋난다(피크 싱크 붕괴). 차이는 '각자 자기 스텝 평균'끼리
                #    비교한다 — 웨이퍼 스텝 평균 − 골든 스텝 평균(경계 어긋남 원리 제거).
                gstep_pre = None
                sstat0 = ref.get("step_stats") or {}
                if steps_ref is not None and usteps:
                    gstep_pre = {}
                    for sn9 in sensors:
                        gm9x = rmean.get(sn9)
                        if gm9x is None:
                            continue
                        row9x = []
                        for u9x in usteps:
                            e9x = (sstat0.get(str(int(u9x))) or {}).get(sn9)
                            if e9x is not None:
                                row9x.append(float(e9x[0]))   # 정렬 통계(오염 없음)
                                continue
                            mk9x = _erode_mask(steps_ref == u9x)
                            if not mk9x.any():
                                mk9x = steps_ref == u9x
                            row9x.append(float(np.mean(gm9x[mk9x]))
                                         if mk9x.any() else np.nan)
                        gstep_pre[sn9] = row9x
                ids_all, dates_all, mats = [], [], []
                iod_mats = []
                n_hit = n_calc = 0
                for di9, d9 in enumerate(dates):
                    pairs = day_map[d9]
                    hh = hashlib.md5((";".join(sorted(st for _f, st in pairs))
                                      + "|" + ",".join(sensors)
                                      + "|" + CALC_VER + "|phys6:" + ",".join(IO_SEL) + "|disp:" + ",".join(IO_DISP)
                                      + "|" + ",".join(a9x["sensor"] for a9x in g_arch)
                                      ).encode()).hexdigest()[:12]
                    cf = cache_dir / f"{d9}.npz"
                    day = None
                    if cf.is_file():
                        try:
                            z9 = np.load(cf, allow_pickle=False)
                            meta9 = json.loads(bytes(z9["meta"]).decode("utf-8"))
                            if meta9.get("hash") == hh and (usteps is None
                                                            or meta9.get("usteps") == usteps):
                                day = {"ids": meta9["ids"], "arr": z9["arr"],
                                       "usteps": meta9["usteps"],
                                       "phys": meta9.get("phys"),
                                       "iod": (z9["iod"] if "iod" in z9.files else None)}
                        except Exception as e9:        # noqa: BLE001
                            if n_hit + n_calc < 3:
                                print(f"[sdiff] 캐시 판독 실패({d9}): "
                                      f"{type(e9).__name__} {e9}", flush=True)
                    if day is None:
                        recs = _load_day(pairs)
                        ids9, rows9, iodr9 = [], [], []
                        pc_arch = [[0, 0, 0, 0.0] for _ in g_arch]   # 판정/포화/미개방/ρ합
                        pc_io = [[0, 0.0, 0, 0] for _ in IO_SEL]     # 열림/t0합/응답판정/무반응
                        us9 = usteps
                        for rec in recs:
                            if ref.get("recipe"):
                                try:
                                    rc9 = str(S._recipe_display(
                                        S._RECIPE_TAGS.get(str(rec["wafer_id"]), "")
                                        or rec.get("recipe") or "") or "")
                                except Exception:      # noqa: BLE001
                                    rc9 = ""
                                if rc9 and rc9 != ref["recipe"]:
                                    continue
                            arr9 = np.asarray(rec["sensors"], dtype=np.float64)
                            cols9 = [str(c) for c in (rec.get("sensor_cols") or [])]
                            T9 = arr9.shape[0]
                            idx9 = np.clip(np.round(np.linspace(0, T9 - 1, L)).astype(int),
                                           0, T9 - 1)
                            try:
                                stw9 = np.asarray(rec.get("step"),
                                                  dtype=float)[idx9].astype(int)
                            except Exception:          # noqa: BLE001
                                stw9 = (steps_ref if steps_ref is not None
                                        else np.zeros(L, dtype=int))
                            if us9 is None:
                                us9 = sorted(set(int(v) for v in stw9) - {0})
                            masks9 = []
                            for u9 in us9:
                                mk9 = _erode_mask(stw9 == u9)
                                masks9.append(mk9 if mk9.any() else None)
                            row9 = np.full((len(sensors), len(us9)), np.nan,
                                           dtype=np.float32)
                            for si9, sn in enumerate(sensors):
                                if sn not in cols9:
                                    continue
                                w9 = arr9[idx9, cols9.index(sn)]
                                gp9 = gstep_pre.get(sn) if gstep_pre else None
                                for ki9, mk9 in enumerate(masks9):
                                    if mk9 is None:
                                        continue
                                    if gp9 is not None and ki9 < len(gp9) \
                                            and np.isfinite(gp9[ki9]):
                                        # 스텝 경계 정렬 무관: 자기 평균 − 골든 자기 평균
                                        row9[si9, ki9] = float(
                                            np.mean(w9[mk9]) - gp9[ki9])
                                    else:               # 폴백(구형 골든: 스텝 정보 없음)
                                        row9[si9, ki9] = float(
                                            np.mean(w9[mk9] - rmean[sn][mk9]))
                            ids9.append(str(rec["wafer_id"]))
                            rows9.append(row9)
                            # 표시용 IO(밸브 ON/OFF): 스텝별 열림비율
                            iodrow9 = np.full((len(IO_DISP), len(us9)), np.nan,
                                              dtype=np.float32)
                            try:
                                iodx9 = rec.get("io")
                                iocx9 = [str(c) for c in (rec.get("io_cols") or [])]
                                if iodx9 is not None and iocx9:
                                    iodx9 = np.asarray(iodx9, dtype=np.float64)
                                    for di_9, io9 in enumerate(IO_DISP):
                                        if io9 not in iocx9 or iodx9.ndim != 2:
                                            continue
                                        v9d = iodx9[idx9, iocx9.index(io9)] > 0.5
                                        for ki9, u9 in enumerate(us9):
                                            mm9d = stw9 == u9
                                            if mm9d.any():
                                                iodrow9[di_9, ki9] = float(
                                                    np.mean(v9d[mm9d]))
                            except Exception:          # noqa: BLE001
                                pass
                            iodr9.append(iodrow9)
                            # ── 물리 위반(골든=답지): IO 타이밍·응답·아치 ──
                            io_t0w = {}
                            try:
                                iod9 = rec.get("io")
                                ioc9 = [str(c) for c in (rec.get("io_cols") or [])]
                                if iod9 is not None and ioc9:
                                    iod9 = np.asarray(iod9, dtype=np.float64)
                                    for qi9, io9 in enumerate(IO_SEL):
                                        if io9 not in ioc9 or iod9.ndim != 2:
                                            continue
                                        von9 = iod9[idx9, ioc9.index(io9)] > 0.5
                                        on9 = np.where(von9)[0]
                                        if not on9.size:
                                            continue
                                        io_t0w[io9] = int(on9[0])
                                        pc_io[qi9][0] += 1
                                        pc_io[qi9][1] += float(on9[0]) / L
                                        mm9 = (PH.get("io_map") or {}).get(io9) or {}
                                        sn9 = mm9.get("sensor")
                                        if sn9 in cols9 and io9 in g_resp:
                                            x9 = arr9[idx9, cols9.index(sn9)]
                                            wp9 = _resp_prof(x9, int(on9[0]))
                                            if wp9 is not None:
                                                gp9, sgn9 = g_resp[io9]
                                                miss9 = float(np.mean(gp9 - wp9))
                                                pc_io[qi9][2] += 1
                                                # 골든 응답 대비 부족분이 유의하면 무반응
                                                if miss9 > max(0.5, 3.0 * sgn9):
                                                    pc_io[qi9][3] += 1
                            except Exception:          # noqa: BLE001
                                pass
                            for ri9, ga9 in enumerate(g_arch):
                                if ga9["sensor"] not in cols9:
                                    continue
                                t0f9 = io_t0w.get(ga9["open_first"])
                                st_w9 = (int(stw9[t0f9]) if t0f9 is not None
                                         else ga9["step"])
                                mk9 = _erode_mask(stw9 == st_w9)
                                if not mk9.any() or int(mk9.sum()) < 8:
                                    continue
                                x9a = arr9[idx9, cols9.index(ga9["sensor"])][mk9]
                                k9a = max(2, len(x9a) // 10)
                                pos9a = np.where(mk9)[0]
                                use_self9 = False
                                if ga9.get("self_mode") and t0f9 is not None:
                                    pre9 = x9a[pos9a < t0f9]
                                    post9 = x9a[pos9a >= t0f9]
                                    # 열림 전 표본이 부족하면(밸브가 스텝 시작 직후 열림)
                                    # 자기 정렬 불가 → 아래 골든 tail 앵커로 폴백
                                    use_self9 = len(pre9) >= 3 and len(post9) >= 6
                                if use_self9:
                                    # ── 자기 정렬: 자기 열림 전 구간 = 기준선 ──
                                    base9 = float(np.median(pre9))
                                    nz9 = 1.4826 * float(np.median(
                                        np.abs(np.diff(pre9)))) if len(pre9) > 3 else 0.1
                                    j9x = int(np.argmax(np.abs(post9 - base9)))
                                    s9a = 1.0 if post9[j9x] >= base9 else -1.0
                                    dev9 = s9a * (post9 - base9)
                                    depth_w9 = float(dev9[j9x])
                                    tail_w9 = float(np.median(dev9[-k9a:]))
                                    pc_arch[ri9][0] += 1
                                    rec9 = (1.0 - max(0.0, tail_w9) / max(depth_w9, 1e-9)
                                            if depth_w9 > 0 else 1.0)
                                    pc_arch[ri9][3] += rec9
                                    # 자기 유의성: 열림 전 안정 노이즈 대비 + 원단위 하한
                                    if depth_w9 < max(ga9["d_sig"] * 4.0 * nz9, 0.2):
                                        continue
                                    if tail_w9 >= ga9["r_min"] * depth_w9:
                                        if io_t0w and ga9["open_second"] not in io_t0w:
                                            pc_arch[ri9][2] += 1   # 두번째 밸브 미개방
                                        else:
                                            pc_arch[ri9][1] += 1   # 아치 붕괴(포화)
                                    continue
                                # ── 기본: 골든 tail(회복 레벨) 앵커 ──
                                s9a = ga9["g_sign"]
                                dev9 = s9a * (x9a - ga9["g_tail"])
                                depth_w9 = float(np.max(dev9))
                                tail_w9 = float(np.median(dev9[-k9a:]))
                                pc_arch[ri9][0] += 1
                                rec9 = (1.0 - max(0.0, tail_w9) / max(depth_w9, 1e-9)
                                        if depth_w9 > 0 else 1.0)
                                pc_arch[ri9][3] += rec9
                                if depth_w9 < 0.5 * ga9["depth_g"]:
                                    continue           # 이탈(딥) 자체가 없으면 별개 문제
                                # 포화 = 스텝 끝에서도 골든 회복 수준으로 안 돌아옴
                                if tail_w9 >= ga9["r_min"] * ga9["depth_g"]:
                                    if io_t0w and ga9["open_second"] not in io_t0w:
                                        pc_arch[ri9][2] += 1   # 두번째 밸브 미개방
                                    else:
                                        pc_arch[ri9][1] += 1   # 아치 붕괴(포화)
                        del recs
                        if us9 is None:
                            continue
                        arrd = (np.stack(rows9) if rows9
                                else np.zeros((0, len(sensors), len(us9)), dtype=np.float32))
                        phys9 = {"arch": pc_arch, "io": pc_io}
                        iodd9 = (np.stack(iodr9) if iodr9
                                 else np.zeros((0, len(IO_DISP), len(us9)),
                                               dtype=np.float32))
                        day = {"ids": ids9, "arr": arrd, "usteps": us9, "phys": phys9,
                               "iod": iodd9}
                        try:
                            meta9 = json.dumps({"hash": hh, "ids": ids9, "usteps": us9,
                                                "phys": phys9},
                                               ensure_ascii=False)
                            np.savez_compressed(cf, arr=arrd, iod=iodd9,
                                                meta=np.frombuffer(meta9.encode("utf-8"),
                                                                   dtype=np.uint8))
                        except Exception as e9:        # noqa: BLE001
                            print(f"[sdiff] 캐시 저장 실패({d9}): {e9}", flush=True)
                        n_calc += 1
                    else:
                        n_hit += 1
                    if usteps is None:
                        usteps = day["usteps"]
                    if day["usteps"] != usteps or not len(day["ids"]):
                        continue                       # 스텝 구성이 다른 날(다른 공정)은 제외
                    ids_all += day["ids"]
                    dates_all += [d9] * len(day["ids"])
                    if day.get("phys"):
                        phys_daily[d9] = day["phys"]
                    mats.append(np.asarray(day["arr"], dtype=np.float32))
                    iod_mats.append(np.asarray(day["iod"], dtype=np.float32)
                                    if day.get("iod") is not None else
                                    np.full((len(day["ids"]), len(IO_DISP),
                                             len(day["usteps"])), np.nan,
                                            dtype=np.float32))
                    if di9 and di9 % 5 == 0:
                        _sdprog(f"{di9}/{len(dates)}일 (캐시 {n_hit} · 신규 {n_calc})")
                if not mats:
                    raise RuntimeError("집계된 웨이퍼가 없습니다(레시피 불일치 또는 파싱 실패)")
                big = np.concatenate(mats, axis=0)     # (N, S, K)
                iod_big = np.concatenate(iod_mats, axis=0) if iod_mats else None
                id2src = {st9: fol9 for prs9 in day_map.values()
                          for fol9, st9 in prs9}       # 원본 파형 재로드용
                SDD.clear()
                SDD.update(ids=ids_all, dates=dates_all, arr=big,
                           sensors=sensors, usteps=usteps,
                           io_disp=iod_big, io_disp_names=IO_DISP,
                           id2src=id2src, rmean=rmean,
                           steps_ref=steps_ref,
                           excl_sensors=sorted(excl_set))
                absmax = {}
                with np.errstate(all="ignore"):
                    for si9, sn in enumerate(sensors):
                        v9 = np.abs(big[:, si9, :])
                        absmax[sn] = round(float(np.nanmax(v9)) if np.isfinite(
                            np.nanmax(v9)) else 0.0, 3)
                # ── 자동 추림: 일 평균 차이 시계열 → 지속 이탈·급변 (원단위) ──
                days_u = sorted(set(dates_all))
                di_map = {d9: i9 for i9, d9 in enumerate(days_u)}
                S_n, K_n = len(sensors), len(usteps)
                sumd = np.zeros((len(days_u), S_n, K_n))
                cntd = np.zeros_like(sumd)
                for i9, d9 in enumerate(dates_all):
                    v9 = big[i9]
                    m9 = np.isfinite(v9)
                    sumd[di_map[d9]][m9] += v9[m9]
                    cntd[di_map[d9]][m9] += 1
                with np.errstate(all="ignore"):
                    daymu = np.where(cntd > 0, sumd / np.maximum(cntd, 1), np.nan)
                SDD["daymu"] = daymu
                SDD["days_u"] = days_u
                # 문턱: 통합 골든 σ의 스텝 평균(있으면) — 없으면 일 시계열 MAD 폴백
                # σ: 스텝 정렬 σ(골든 step_stats — 정렬 무관) 우선. 없으면 일 시계열
                #   MAD. 포인트별 std 는 전이 어긋남으로 폭발하므로 쓰지 않는다(v98).
                sstat = ref.get("step_stats") or {}
                sig = np.zeros((S_n, K_n))
                for si9, sn in enumerate(sensors):
                    for ki9, u9 in enumerate(usteps):
                        e9 = (sstat.get(str(int(u9))) or {}).get(sn)
                        if e9 is not None:
                            sig[si9, ki9] = float(e9[1])
                        else:
                            v9 = daymu[:, si9, ki9]
                            v9 = v9[np.isfinite(v9)]
                            sig[si9, ki9] = (1.4826 * float(np.median(
                                np.abs(v9 - np.median(v9)))) if v9.size else 0.0)
                # 최소 문턱 차등: 압력(bar)은 스케일이 작고 미세 차이가 유의 → 0.01,
                # 온도(℃)는 0.05 — 일괄 0.05 로는 압력 이상이 dT 에 묻힌다(사용자 지적).
                floors = np.array([[0.01 if re.match(r"(?i)^d?p", sn) else 0.05]
                                   for sn in sensors])
                thr = np.maximum(2.0 * sig, floors)
                SDD["thr"] = thr
                # 골든 스텝 평균 — 원시값 복원(diff+gstep)·부호 위반 판정용
                gstep = None
                if steps_ref is not None:
                    gstep = np.zeros((S_n, K_n))
                    gmin = np.zeros((S_n, K_n))        # 골든 스텝 '내내' 부호 확인용 최저값
                    gm9 = [np.asarray(ref["mean"].get(sn), dtype=np.float64)
                           if ref["mean"].get(sn) is not None else None
                           for sn in sensors]
                    for ki9, u9 in enumerate(usteps):
                        mk9 = _erode_mask(steps_ref == u9)
                        if not mk9.any():
                            mk9 = steps_ref == u9
                        for si9, sn in enumerate(sensors):
                            e9x = (sstat0.get(str(int(u9))) or {}).get(sn)
                            if e9x is not None and len(e9x) >= 3:
                                gstep[si9, ki9] = float(e9x[0])
                                gmin[si9, ki9] = float(e9x[2])
                                continue
                            if gm9[si9] is not None:
                                v9 = (_resample(gm9[si9], L)
                                      if len(gm9[si9]) != L else gm9[si9])
                                gstep[si9, ki9] = float(np.mean(v9[mk9]))
                                gmin[si9, ki9] = float(np.min(v9[mk9]))
                else:
                    gmin = None
                SDD["gstep"] = gstep
                violations = []
                sign_sens = set((PH.get("sign_rules") or {}).get("sensors")
                                or _VIRTUAL_SIGN)
                m_sig9 = float((PH.get("sign_rules") or {})
                               .get("golden_pos_margin_sigma", 2.0))
                if gstep is not None:
                    for si9, sn in enumerate(sensors):
                        if sn not in sign_sens:
                            continue
                        for ki9, u9 in enumerate(usteps):
                            # 골든=답지: 골든이 그 스텝 '내내'(스텝 내 최저값 기준) 확실히
                            # 양수일 때만 규칙 활성. 스텝 평균만 보면 전이 구간이 섞인
                            # 스텝(골든도 일부 음수)을 면제 못 해 오탐(사용자 지적).
                            gmn9 = float(gmin[si9, ki9]) if gmin is not None else -1.0
                            if gmn9 <= max(m_sig9 * sig[si9, ki9], 0.05):
                                continue
                            raw9 = gstep[si9, ki9] + daymu[:, si9, ki9]
                            bad9 = np.isfinite(raw9) & (raw9 < 0)
                            i9 = 0
                            while i9 < len(days_u):
                                if not bad9[i9]:
                                    i9 += 1
                                    continue
                                j9 = i9
                                while j9 + 1 < len(days_u) and bad9[j9 + 1]:
                                    j9 += 1
                                seg9 = raw9[i9:j9 + 1]
                                violations.append({
                                    "sensor": sn, "step": int(u9),
                                    "d0": days_u[i9], "d1": days_u[j9],
                                    "days": j9 - i9 + 1,
                                    "min_raw": round(float(np.min(seg9)), 3),
                                    "g_min": round(gmn9, 3),
                                    "ongoing": j9 == len(days_u) - 1,
                                    "virtual": True})
                                i9 = j9 + 1
                    violations.sort(key=lambda x9: (x9.get("golden_neg", False),
                                                    x9["min_raw"] if x9["min_raw"]
                                                    is not None else 0.0))
                holds, jumps = [], []
                for si9, sn in enumerate(sensors):
                    if sn in excl_set:                 # 판정 제외(그래프 표시는 유지)
                        continue
                    for ki9, u9 in enumerate(usteps):
                        v9 = daymu[:, si9, ki9]
                        t9 = thr[si9, ki9]
                        run0 = None
                        for i9 in range(len(days_u) + 1):
                            on9 = (i9 < len(days_u) and np.isfinite(v9[i9])
                                   and abs(v9[i9]) > t9)
                            sgn9 = 0 if not on9 else (1 if v9[i9] > 0 else -1)
                            if run0 is not None and (not on9 or sgn9 != run0[1]):
                                a9, _s9 = run0
                                if i9 - a9 >= 3:       # 연속 3일 이상 같은 부호 이탈
                                    seg9 = v9[a9:i9]
                                    pk9 = float(seg9[np.argmax(np.abs(seg9))])
                                    holds.append({
                                        "sensor": sn, "step": int(u9),
                                        "dir": "+" if _s9 > 0 else "-",
                                        "d0": days_u[a9], "d1": days_u[i9 - 1],
                                        "days": i9 - a9,
                                        "mean": round(float(np.mean(seg9)), 3),
                                        "peak": round(pk9, 3),
                                        "ongoing": i9 == len(days_u),
                                        "score": round(abs(pk9) / max(t9, 1e-9), 1),
                                        "virtual": bool(re.match(r"(?i)^d[tp]_", sn))})
                                run0 = None
                            if on9 and run0 is None:
                                run0 = (i9, sgn9)
                        dv9 = np.diff(v9)
                        fin9 = np.abs(dv9[np.isfinite(dv9)])
                        med9 = float(np.median(fin9)) if fin9.size else 0.0
                        jt9 = max(3.0 * sig[si9, ki9], 2.0 * med9, 0.05)
                        for i9 in range(1, len(days_u)):
                            d9v = v9[i9] - v9[i9 - 1]
                            if np.isfinite(d9v) and abs(d9v) > jt9:
                                jumps.append({
                                    "sensor": sn, "step": int(u9),
                                    "date": days_u[i9],
                                    "delta": round(float(d9v), 3),
                                    "score": round(abs(d9v) / max(sig[si9, ki9], 1e-9), 1),
                                    "virtual": bool(re.match(r"(?i)^d[tp]_", sn))})
                holds.sort(key=lambda x9: (-int(x9["virtual"]), -x9["score"]))
                jumps.sort(key=lambda x9: (-int(x9["virtual"]), -x9["score"]))
                # ── 물리 위반 집계(일별 위반 비율 → 연속 run) ──
                phys_items = []
                days_p = sorted(phys_daily)

                def _p_runs(ratios9, dates9, thr9=0.3, solo9=0.6):
                    out9, i9 = [], 0
                    while i9 < len(dates9):
                        if ratios9[i9] < thr9:
                            i9 += 1
                            continue
                        j9 = i9
                        while j9 + 1 < len(dates9) and ratios9[j9 + 1] >= thr9:
                            j9 += 1
                        pk9 = max(ratios9[i9:j9 + 1])
                        if j9 > i9 or pk9 >= solo9:
                            out9.append((dates9[i9], dates9[j9], j9 - i9 + 1,
                                         round(pk9, 2), j9 == len(dates9) - 1))
                        i9 = j9 + 1
                    return out9
                for ri9, ga9 in enumerate(g_arch):
                    r_sat, r_no2 = [], []
                    for d8 in days_p:
                        c8 = (phys_daily[d8].get("arch") or [])
                        c8 = c8[ri9] if ri9 < len(c8) else [0, 0, 0, 0.0]
                        n8 = max(1, c8[0])
                        r_sat.append(c8[1] / n8)
                        r_no2.append(c8[2] / n8)
                    for d08, d18, nd8, pk8, on8 in _p_runs(r_sat, days_p):
                        phys_items.append({"kind": "아치 붕괴(포화)",
                                           "sensor": ga9["sensor"], "step": ga9["step"],
                                           "d0": d08, "d1": d18, "days": nd8,
                                           "ratio_peak": pk8,
                                           "depth_g": ga9["depth_g"],
                                           "ongoing": on8, "virtual": True})
                    for d08, d18, nd8, pk8, on8 in _p_runs(r_no2, days_p):
                        phys_items.append({"kind": "IO 미개방",
                                           "sensor": ga9["sensor"], "step": ga9["step"],
                                           "io": ga9["open_second"],
                                           "d0": d08, "d1": d18, "days": nd8,
                                           "ratio_peak": pk8, "ongoing": on8,
                                           "virtual": True})
                for qi9, io9 in enumerate(IO_SEL):
                    r_bad, r_dt = [], []
                    for d8 in days_p:
                        c8 = (phys_daily[d8].get("io") or [])
                        c8 = c8[qi9] if qi9 < len(c8) else [0, 0.0, 0, 0]
                        r_bad.append(c8[3] / max(1, c8[2]))
                        t0m8 = (c8[1] / c8[0]) if c8[0] else None
                        gt8 = g_t0.get(io9)
                        r_dt.append(abs(t0m8 - gt8 / L)
                                    if (t0m8 is not None and gt8 is not None) else 0.0)
                    sn_r9 = ((PH.get("io_map") or {}).get(io9) or {}).get("sensor", "")
                    for d08, d18, nd8, pk8, on8 in _p_runs(r_bad, days_p):
                        phys_items.append({"kind": "밸브 무반응", "io": io9,
                                           "sensor": sn_r9,
                                           "d0": d08, "d1": d18, "days": nd8,
                                           "ratio_peak": pk8, "ongoing": on8,
                                           "virtual": False})
                    for d08, d18, nd8, pk8, on8 in _p_runs(
                            r_dt, days_p, thr9=tol_t0, solo9=tol_t0 * 3):
                        phys_items.append({"kind": "밸브 타이밍", "io": io9,
                                           "sensor": sn_r9,
                                           "d0": d08, "d1": d18, "days": nd8,
                                           "dt_pct": round(pk8 * 100, 1),
                                           "ratio_peak": pk8, "ongoing": on8,
                                           "virtual": False})
                _ord9 = {"아치 붕괴(포화)": 0, "IO 미개방": 1, "밸브 무반응": 2,
                         "밸브 타이밍": 3}
                phys_items.sort(key=lambda x9: (_ord9.get(x9["kind"], 9),
                                                -x9.get("ratio_peak", 0)))
                findings = {"holds": holds[:40], "jumps": jumps[:40],
                            "violations": violations[:20],
                            "phys": phys_items[:30]}
                SDJOB["result"] = {"ref": {"name": ref.get("name"),
                                           "recipe": ref.get("recipe") or ""},
                                   "sensors": sensors, "steps": usteps,
                                   "dates": sorted(set(dates_all)),
                                   "n_wafers": len(ids_all), "absmax": absmax,
                                   "thr": {sn: [round(float(v), 3) for v in thr[si9]]
                                           for si9, sn in enumerate(sensors)},
                                   "gstep": ({sn: [round(float(v), 3) for v in gstep[si9]]
                                              for si9, sn in enumerate(sensors)}
                                             if gstep is not None else None),
                                   "io_disp_names": IO_DISP,
                                   "io_golden": g_io_disp,
                                   "sign_rules": {sn9x: "+" for sn9x in sign_sens},
                                   "arch_expect": g_arch,
                                   "findings": findings}
                SDJOB["state"] = "done"
                _pkl_dump(cache_dir / "result.pkl",
                          {"calc_ver": CALC_VER,
                           "ids": SDD["ids"], "dates": SDD["dates"], "arr": SDD["arr"],
                           "sensors": sensors, "usteps": usteps, "daymu": SDD["daymu"],
                           "days_u": SDD["days_u"], "thr": SDD["thr"],
                           "io_disp": SDD.get("io_disp"),
                           "io_disp_names": IO_DISP, "id2src": id2src,
                           "rmean": rmean, "steps_ref": steps_ref,
                           "excl_sensors": SDD["excl_sensors"],
                           "result": SDJOB["result"]})
                print(f"[sdiff] 완료 — 웨이퍼 {len(ids_all)}장 · 센서 {len(sensors)} · "
                      f"스텝 {usteps}", flush=True)
            except Exception as e9:                    # noqa: BLE001
                traceback.print_exc()
                SDJOB.update(state="error", error=str(e9))
        import threading
        threading.Thread(target=_sdrun, daemon=True).start()
        return {"ok": True, "n_days": len(dates), "n_wafers": n_total}

    @app.get("/api/stepdiff_status")
    def api_stepdiff_status():
        return {"state": SDJOB["state"], "progress": SDJOB["progress"],
                "error": SDJOB["error"], "result": SDJOB["result"]}

    @app.post("/api/stepdiff_page")
    def api_stepdiff_page(payload: dict = Body(...)):
        """스텝 차이 페이지 조회 — mode=wafer(오프셋 페이징) | bucket(5일 묶음 전체)."""
        if not SDD:
            raise HTTPException(400, "스텝 차이 집계가 없습니다 — 먼저 실행하세요")
        sel_days = set(payload.get("dates") or [])
        idxs = [i9 for i9, d9 in enumerate(SDD["dates"])
                if not sel_days or d9 in sel_days]
        sensors, usteps, arr = SDD["sensors"], SDD["usteps"], SDD["arr"]

        def _diff_of(mat9):                            # (S,K) → {sn: [k…]} (NaN→None)
            out9 = {}
            for si9, sn in enumerate(sensors):
                out9[sn] = [None if not np.isfinite(v9) else round(float(v9), 3)
                            for v9 in mat9[si9]]
            return out9

        iodn = SDD.get("io_disp_names") or []
        iodA = SDD.get("io_disp")

        def _io_of(mat9):                              # (nio,K) → {io: [k…]}
            if mat9 is None:
                return None
            return {io9: [None if not np.isfinite(v9) else round(float(v9), 2)
                          for v9 in mat9[qi9]]
                    for qi9, io9 in enumerate(iodn)}
        mode = str(payload.get("mode") or "wafer")
        if mode == "bucket":
            bd = max(1, min(31, int(payload.get("bucket_days") or 5)))
            days = sorted(set(SDD["dates"][i9] for i9 in idxs))
            if not days:
                return {"total": 0, "items": []}
            import datetime as _dt
            d0 = _dt.date.fromisoformat(days[0])
            by_bk = {}
            for i9 in idxs:
                bk9 = (_dt.date.fromisoformat(SDD["dates"][i9]) - d0).days // bd
                by_bk.setdefault(bk9, []).append(i9)
            items = []
            for bk9 in sorted(by_bk):
                ii9 = by_bk[bk9]
                s0 = d0 + _dt.timedelta(days=bk9 * bd)
                s1 = s0 + _dt.timedelta(days=bd - 1)
                with np.errstate(all="ignore"):
                    m9 = np.nanmean(arr[ii9], axis=0)
                    mio9 = (np.nanmean(iodA[ii9], axis=0) if iodA is not None else None)
                items.append({"id": (s0.isoformat() if bd == 1
                                     else f"{s0.isoformat()[5:]}~{s1.isoformat()[5:]}"),
                              "date": s0.isoformat(), "n": len(ii9),
                              "diff": _diff_of(m9), "io": _io_of(mio9)})
            return {"total": len(items), "items": items}
        off = max(0, int(payload.get("offset") or 0))
        lim = max(1, min(200, int(payload.get("limit") or 60)))
        items = []
        for i9 in idxs[off:off + lim]:
            items.append({"id": SDD["ids"][i9], "date": SDD["dates"][i9], "n": 1,
                          "diff": _diff_of(arr[i9]),
                          "io": _io_of(iodA[i9] if iodA is not None else None)})
        return {"total": len(idxs), "items": items}

    def _sd_pressure_state(scope_days, sensors_sel=None):
        """요소① outlet 기준 음/양압: 규칙 센서(dP_*-Out)의 원시값(golden+diff)으로
        선택 기간의 음압(역류)/양압(정상 방향) 일수·최저값을 스텝별 집계."""
        if SDD.get("gstep") is None:
            return []
        days_u, daymu, gstep = SDD["days_u"], SDD["daymu"], SDD["gstep"]
        sensors, usteps = SDD["sensors"], SDD["usteps"]
        sset9 = set(sensors_sel) if sensors_sel else None
        pos = {d9: i9 for i9, d9 in enumerate(days_u)}
        idx = [pos[d9] for d9 in scope_days if d9 in pos]
        if not idx:
            return []
        rows = []
        for si9, sn in enumerate(sensors):
            if _VIRTUAL_SIGN.get(sn) != "+":
                continue
            if sset9 is not None and sn not in sset9:
                continue
            for ki9, u9 in enumerate(usteps):
                raw9 = gstep[si9, ki9] + daymu[idx, si9, ki9]
                fin9 = raw9[np.isfinite(raw9)]
                if not fin9.size:
                    continue
                n_neg9 = int(np.sum(fin9 < 0))
                g0_9 = float(gstep[si9, ki9])
                st9_ = ("골든 자체 음압(데이터 확인 필요)" if g0_9 < 0
                        else ("음압(역류)" if n_neg9 else "양압(정상 방향)"))
                rows.append({"sensor": sn, "step": int(u9),
                             "n_days": int(fin9.size), "n_neg": n_neg9,
                             "min_raw": round(float(np.min(fin9)), 3),
                             "mean_raw": round(float(np.mean(fin9)), 3),
                             "g0": round(g0_9, 3), "state": st9_})
        rows.sort(key=lambda x9: (-x9["n_neg"], x9["min_raw"]))
        return rows

    def _sd_patterns(sel_days, mode="prev", sensors_sel=None):
        """sensors_sel: 체크된 센서 목록(None=전체) — 목록 밖 센서는 판정 자체를 생략."""
        """선택한 날짜들의 일 평균 차이 시계열을 센서×스텝별로 패턴 분류.

        mode="prev": 직전 14일과 비교 — 신규 발생/점증 드리프트/유지/복귀/급변.
        mode="seq": 체크한 날들끼리 순차(인접 쌍) 비교 — 13→14, 14→15 … 변화량 나열.
        원단위. 가상센서 우선 정렬.
        """
        days_u = SDD["days_u"]
        daymu = SDD["daymu"]
        thr = SDD["thr"]
        sensors, usteps = SDD["sensors"], SDD["usteps"]
        sel = sorted(set(sel_days) & set(days_u))
        min_d = 2 if mode == "seq" else 3
        if len(sel) < min_d:
            raise HTTPException(400, f"선택 기간이 {len(sel)}일뿐입니다 — {min_d}일 이상 체크하세요")
        pos = {d9: i9 for i9, d9 in enumerate(days_u)}
        idx = [pos[d9] for d9 in sel]
        sset9 = set(sensors_sel) if sensors_sel else None
        excl9 = set(SDD.get("excl_sensors") or [])
        if mode == "seq":
            import datetime as _dtq
            rows = []
            for si9, sn in enumerate(sensors):
                if (sset9 is not None and sn not in sset9) or sn in excl9:
                    continue
                for ki9, u9 in enumerate(usteps):
                    v9 = daymu[idx, si9, ki9]
                    t9 = float(thr[si9, ki9])
                    dl9 = []
                    for a9 in range(len(sel) - 1):
                        x0, x1 = v9[a9], v9[a9 + 1]
                        if not (np.isfinite(x0) and np.isfinite(x1)):
                            continue
                        gap9 = (_dtq.date.fromisoformat(sel[a9 + 1])
                                - _dtq.date.fromisoformat(sel[a9])).days
                        dl9.append({"a": sel[a9], "b": sel[a9 + 1], "gap": gap9,
                                    "delta": round(float(x1 - x0), 3)})
                    if not dl9:
                        continue
                    ds9 = [d9["delta"] for d9 in dl9]
                    mx9 = max(abs(d9) for d9 in ds9)
                    cum9 = float(sum(ds9))
                    if mx9 < t9 * 0.5 and abs(cum9) < t9:
                        continue                       # 유의 변화 없는 조합은 생략
                    sig9 = [d9 for d9 in ds9 if abs(d9) >= t9 * 0.3]
                    if sig9 and all(d9 > 0 for d9 in sig9) and len(sig9) >= 2:
                        kind9 = "연속 상승(점증)"
                    elif sig9 and all(d9 < 0 for d9 in sig9) and len(sig9) >= 2:
                        kind9 = "연속 하락(점증)"
                    elif len([d9 for d9 in ds9 if abs(d9) >= t9]) == 1:
                        kind9 = "급변(단발)"
                    elif (len(sig9) >= 2
                          and any(a9 * b9 < 0 for a9, b9 in zip(sig9, sig9[1:]))):
                        kind9 = "진동(오르내림)"
                    else:
                        kind9 = "변동"
                    rows.append({"sensor": sn, "step": int(u9), "kind": kind9,
                                 "deltas": dl9, "cum": round(cum9, 3),
                                 "maxd": round(mx9, 3),
                                 "score": round(max(mx9, abs(cum9)) / max(t9, 1e-9), 1),
                                 "virtual": bool(re.match(r"(?i)^d[tp]_", sn))})
            rows.sort(key=lambda x9: (-int(x9["virtual"]), -x9["score"]))
            return {"period": [sel[0], sel[-1]], "n_days": len(sel),
                    "mode": "seq", "rows": rows[:60], "jumps": [],
                    "pressure": _sd_pressure_state(sel, sensors_sel)}
        i0 = min(idx)
        pre_idx = list(range(max(0, i0 - 14), i0))     # 비교 기준 = 직전 최대 14일
        rows = []
        for si9, sn in enumerate(sensors):
            if (sset9 is not None and sn not in sset9) or sn in excl9:
                continue
            for ki9, u9 in enumerate(usteps):
                v9 = daymu[idx, si9, ki9]
                fin9 = np.isfinite(v9)
                if fin9.sum() < 3:
                    continue
                t9 = float(thr[si9, ki9])
                vf9 = v9[fin9]
                over9 = float(np.mean(np.abs(vf9) > t9))
                vpre9 = daymu[pre_idx, si9, ki9] if pre_idx else np.array([])
                vpre9 = vpre9[np.isfinite(vpre9)] if vpre9.size else vpre9
                pre_over9 = (float(np.mean(np.abs(vpre9) > t9)) >= 0.3) if vpre9.size >= 3 else None
                rho9 = _spearman([float(x9) if np.isfinite(x9) else None for x9 in v9])
                first9, last9 = float(vf9[0]), float(vf9[-1])
                delta9 = last9 - first9
                pk9 = float(vf9[np.argmax(np.abs(vf9))])
                n_half9 = max(1, len(vf9) // 2)
                h1_over9 = float(np.mean(np.abs(vf9[:n_half9]) > t9))
                h2_over9 = float(np.mean(np.abs(vf9[n_half9:]) > t9))
                kind9 = None
                d0_9 = sel[int(np.where(fin9)[0][0])]
                if over9 >= 0.5:
                    cross9 = np.where(np.abs(v9) > t9)[0]
                    d0_9 = sel[int(cross9[0])] if cross9.size else d0_9
                    if pre_over9 is False and h1_over9 <= h2_over9:
                        # 이전엔 없던 차이 — 기간 내 점증까지 겹치면 함께 표기
                        kind9 = ("신규 발생·점증" if (abs(rho9) >= 0.8
                                 and abs(delta9) >= t9 and abs(last9) > abs(first9))
                                 else "신규 발생")
                    elif abs(rho9) >= 0.8 and abs(delta9) >= t9:
                        kind9 = "점증 드리프트" if abs(last9) > abs(first9) else "완화 추세"
                    elif h1_over9 >= 0.6 and h2_over9 <= 0.2:
                        kind9 = "복귀(해소)"
                    else:
                        kind9 = "차이 유지"
                elif abs(rho9) >= 0.85 and abs(delta9) >= t9 and abs(last9) > t9 * 0.7:
                    kind9 = "점증 드리프트(문턱 근접)"
                if kind9 is None:
                    continue
                slope9 = delta9 / max(1, len(vf9) - 1)
                rows.append({"sensor": sn, "step": int(u9), "kind": kind9,
                             "d0": d0_9, "first": round(first9, 3),
                             "last": round(last9, 3), "peak": round(pk9, 3),
                             "slope": round(slope9, 3), "rho": round(rho9, 2),
                             "score": round(abs(pk9) / max(t9, 1e-9), 1),
                             "virtual": bool(re.match(r"(?i)^d[tp]_", sn))})
        rows.sort(key=lambda x9: (-int(x9["virtual"]), -x9["score"]))
        # 급변(선택 기간 내)
        jsel = [j9 for j9 in ((SDJOB.get("result") or {}).get("findings") or {})
                .get("jumps", []) if j9["date"] in set(sel)
                and (sset9 is None or j9["sensor"] in sset9)]
        return {"period": [sel[0], sel[-1]], "n_days": len(sel),
                "mode": "prev", "rows": rows[:60], "jumps": jsel[:20],
                "pressure": _sd_pressure_state(sel, sensors_sel)}

    @app.post("/api/sd_patterns")
    def api_sd_patterns(payload: dict = Body(...)):
        """선택 기간 패턴 비교 — 체크한 날짜만으로 신규/드리프트/유지/복귀 분류."""
        if not SDD or "daymu" not in SDD:
            raise HTTPException(400, "스텝 차이 집계가 없습니다 — 먼저 실행하세요")
        return _sd_patterns(payload.get("dates") or [],
                            str(payload.get("mode") or "prev"),
                            payload.get("sensors") or None)

    SDE = {}                                       # 사건(레이블) 분석 결과 보관

    def _parse_label_csv(path9, days_hint=None):
        """레이블 CSV 유연 파서: utf-8(-sig)/cp949, 콤마/탭/세미콜론, 헤더 자동 인식.

        형식: date,label[,eqp,dsv,memo] — 날짜 2026-07-16 / 2026.7.16 / 7/16 허용
        (연도 없으면 집계 데이터의 연도로 추정). 반환: (events, skipped 사유 목록).
        """
        fp9 = Path(str(path9 or "").strip().strip('"'))
        if not fp9.is_file():
            raise HTTPException(400, f"레이블 CSV 파일이 없습니다: {fp9}")
        raw9 = fp9.read_bytes()
        txt9 = None
        # UTF-16(메모장 '유니코드'/엑셀 '유니코드 텍스트') 우선 감지 — cp949 가
        # UTF-16 바이트를 깨진 채로 통과시키므로 BOM/NUL 로 먼저 가려낸다.
        if raw9[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw9[:200]:
            try:
                txt9 = raw9.decode("utf-16")
            except Exception:                          # noqa: BLE001
                pass
        if txt9 is None:
            for enc9 in ("utf-8-sig", "cp949", "utf-8"):
                try:
                    txt9 = raw9.decode(enc9)
                    break
                except Exception:                      # noqa: BLE001
                    continue
        if txt9 is None:                              # 최후 폴백 — 절대 실패하지 않는다
            txt9 = raw9.decode("cp949", errors="replace")
            print("[csv] 인코딩 자동 판별 실패 — cp949 대체 디코드(깨진 글자는 \ufffd)",
                  flush=True)
        delim9 = "\t" if txt9.count("\t") > txt9.count(",") else (
            ";" if txt9.count(";") > txt9.count(",") else ",")
        days_src9 = days_hint or (SDD.get("days_u") or [])
        years9 = sorted({d8[:4] for d8 in days_src9})
        import datetime as _dtv

        def _to_date(s9):
            s9 = str(s9).strip().replace(".", "-").replace("/", "-")
            pt9 = [p9 for p9 in s9.split("-") if p9 != ""]
            try:
                if len(pt9) == 1 and pt9[0].isdigit() and len(pt9[0]) == 8:
                    # 20260109 처럼 구분자 없는 8자리(사용자 형식)
                    return _dtv.date(int(pt9[0][:4]), int(pt9[0][4:6]),
                                     int(pt9[0][6:8])).isoformat()
                if len(pt9) == 3:
                    return _dtv.date(int(pt9[0]), int(pt9[1]), int(pt9[2])).isoformat()
                if len(pt9) == 2 and years9:           # 연도 생략(7-16) → 데이터 연도로
                    for y9 in years9:
                        d9c = _dtv.date(int(y9), int(pt9[0]), int(pt9[1])).isoformat()
                        du9 = sorted(days_src9)
                        if du9 and du9[0] <= d9c <= du9[-1]:
                            return d9c
                    return _dtv.date(int(years9[-1]), int(pt9[0]),
                                     int(pt9[1])).isoformat()
            except Exception:                          # noqa: BLE001
                return None
            return None
        events9, skipped9 = [], []
        hdr_map9 = None
        for li9, line9 in enumerate(txt9.splitlines()):
            if not line9.strip():
                continue
            cells9 = [c9.strip().strip('"') for c9 in line9.split(delim9)]
            if hdr_map9 is None:
                low9 = [c9.lower() for c9 in cells9]
                def _find9(*keys9):
                    for k9 in keys9:
                        for i9x, c9 in enumerate(low9):
                            if k9 in c9:
                                return i9x
                    return None
                di9 = _find9("date", "날짜", "일자", "발생일")
                li9x = _find9("label", "레이블", "라벨", "불량", "구분", "조치")
                if di9 is not None and _to_date(cells9[di9]) is None:
                    # 헤더 행 — 컬럼 매핑 확정
                    hdr_map9 = {"date": di9, "label": li9x if li9x is not None else 1,
                                "eqp": _find9("eqp", "설비"), "dsv": _find9("dsv", "챔버"),
                                "memo": _find9("memo", "메모", "비고", "내용")}
                    continue
                hdr_map9 = {"date": 0, "label": 1, "eqp": None, "dsv": None,
                            "memo": 2 if len(cells9) > 2 else None}
            m9 = hdr_map9

            def _cell9(k9):
                i9x = m9.get(k9)
                return cells9[i9x].strip() if (i9x is not None
                                               and i9x < len(cells9)) else ""
            d9v = _to_date(_cell9("date"))
            lb9 = _cell9("label")
            if not d9v or not lb9:
                skipped9.append(f"{li9 + 1}행: 날짜/레이블 인식 실패 — {line9[:60]}")
                continue
            events9.append({"date": d9v, "label": lb9, "eqp": _cell9("eqp"),
                            "dsv": _cell9("dsv"), "memo": _cell9("memo")})
        return events9, skipped9

    # ── 데이터 뷰어 탭: 골든 없이 원본 자유 탐색 + 수식 가상센서(그리기 전용) ──
    DVD = {}

    @app.post("/api/dv_form_save")
    def api_dv_form_save(payload: dict = Body(...)):
        """뷰어 수식 목록 영속(재시작 유지)."""
        forms9 = [str(v9) for v9 in (payload.get("forms") or [])][:50]
        _state_save("dvform", forms9)
        return {"ok": True, "n": len(forms9)}

    # ── 패턴 탐사 탭: 데몬(pattern_hunter) 원장 열람 — 읽기 전용 ──
    HXD = {"out": ""}
    HX_ST = {"proposed": "제안", "testing": "검증중",
             "rejected": "기각", "confirmed": "확정"}
    HX_DIR = {1: "상승", -1: "하락"}

    def _hx_db(out9):
        fp9 = Path(str(out9 or "").strip().strip('"')) / "ledger.sqlite"
        if not fp9.is_file():
            raise HTTPException(400, "원장(ledger.sqlite)이 없습니다 — "
                                     "탐사 작업 폴더를 확인하세요")
        return sqlite3.connect(str(fp9))

    def _hx_desc(h9):
        return (f"{h9['sensor']} step{h9['step']} {h9['feature']} "
                f"{HX_DIR.get(int(h9['direction']), '?')} — 보고 전 "
                f"{h9['window_days']}일 지속 (효과 d={h9['effect']}, "
                f"q={h9['fdr_q']})")

    def _hx_rows(db9):
        cur9 = db9.execute("SELECT * FROM hypothesis ORDER BY id")
        cols9 = [d9[0] for d9 in cur9.description]
        return [dict(zip(cols9, r9)) for r9 in cur9.fetchall()]

    @app.post("/api/hunt_list")
    def api_hunt_list(payload: dict = Body(...)):
        out9 = str(payload.get("out") or "").strip().strip('"')
        db9 = _hx_db(out9)
        HXD["out"] = out9
        rows9 = _hx_rows(db9)
        db9.close()
        _state_save("hx", {"out": out9})
        ordk9 = {"confirmed": 0, "testing": 1, "proposed": 2, "rejected": 3}
        rows9.sort(key=lambda h9: (ordk9.get(h9["status"], 9), -h9["id"]))
        return {"n": len(rows9),
                "counts": {k9: sum(1 for r9 in rows9 if r9["status"] == k9)
                           for k9 in HX_ST},
                "rows": [{"id": h9["id"], "status": HX_ST.get(h9["status"], "?"),
                          "st": h9["status"], "desc": _hx_desc(h9),
                          "cause": h9["anchor_cause"], "eqp": h9["anchor_eqp"],
                          "dsv": h9["anchor_dsv"], "date": h9["anchor_date"],
                          "reject": h9.get("reject_reason") or ""}
                         for h9 in rows9]}

    @app.post("/api/hunt_cal")
    def api_hunt_cal(payload: dict = Body(...)):
        """가설 1건 상세: 설명 + 달력 재료(이상 패턴 일·문제 리포트 마크)."""
        out9 = str(payload.get("out") or HXD["out"]).strip().strip('"')
        db9 = _hx_db(out9)
        hid9 = int(payload.get("id") or 0)
        rows9 = [h9 for h9 in _hx_rows(db9) if h9["id"] == hid9]
        if not rows9:
            db9.close()
            raise HTTPException(404, "가설을 찾지 못했습니다")
        h9 = rows9[0]
        ev9 = json.loads(h9.get("evidence_json") or "{}")
        tr9 = db9.execute("SELECT kind,stat,passed,detail FROM trials WHERE "
                          "hyp_id=? ORDER BY id", (hid9,)).fetchall()
        an9 = db9.execute("SELECT eqp,dsv,cause,date FROM anchors").fetchall()
        db9.close()
        days9 = (ev9.get("days") or {})
        byd9 = {d9: {"bad": len(v9.get("bad") or []), "ok": int(v9.get("ok") or 0)}
                for d9, v9 in days9.items()}
        marks9 = {}
        for eq9, dv9, cz9, dd9 in an9:
            if h9["anchor_eqp"] and eq9 and h9["anchor_eqp"] not in eq9 \
                    and eq9 not in h9["anchor_eqp"]:
                continue
            marks9.setdefault(dd9, []).append(
                cz9 + (f" [{eq9} {dv9}]" if (eq9 or dv9) else ""))
        kmap9 = {"contrast": "대조", "holdout": "홀드아웃", "precede": "선행성",
                 "refute": "반증", "debate": "토론(비판자)"}
        lines9 = [_hx_desc(h9),
                  f"상태 {HX_ST.get(h9['status'], '?')} · 지속률 "
                  f"{h9['persist_ratio']} · 앵커 {h9['anchor_eqp']}/"
                  f"{h9['anchor_dsv']} {h9['anchor_date']} ({h9['anchor_cause']})"]
        if ev9.get("onset"):
            lines9.append(f"이탈 시작 {ev9['onset']} — 보고 {ev9.get('lead_days')}일 전"
                          f" · 유형 {ev9.get('kind', '')}")
        for k9, st9, ps9, dt9x in tr9:
            lines9.append(f"{kmap9.get(k9, k9)}: {'통과' if ps9 else '실패'} — {dt9x}")
        if h9.get("reject_reason"):
            lines9.append("기각 사유: " + h9["reject_reason"])
        return {"h": {"id": h9["id"], "sensor": h9["sensor"],
                      "step": h9["step"], "feature": h9["feature"],
                      "direction": h9["direction"],
                      "fence": ev9.get("fence"), "c_med": ev9.get("c_med"),
                      "c_iqr": ev9.get("c_iqr")},
                "desc": lines9, "llm": h9.get("llm_note") or "",
                "byd": byd9, "marks": marks9,
                "days": {d9: {"bad": v9.get("bad") or [],
                              "ok": int(v9.get("ok") or 0)}
                         for d9, v9 in days9.items()}}

    @app.post("/api/hunt_wave")
    def api_hunt_wave(payload: dict = Body(...)):
        """가설 웨이퍼 카드: 원본 곡선 + 패턴 스텝 구간 + 특징값/정상 기준(z)."""
        out9 = str(payload.get("out") or HXD["out"]).strip().strip('"')
        wid9 = str(payload.get("id") or "")
        day9 = str(payload.get("day") or "")
        hx9 = payload.get("h") or {}
        fp9 = Path(out9) / "features" / f"{day9}.npz"
        if not fp9.is_file():
            raise HTTPException(404, "그날 특징 파일이 없습니다")
        with np.load(fp9, allow_pickle=True) as z9:
            ids9 = [str(v9) for v9 in z9["ids"]]
            if wid9 not in ids9:
                raise HTTPException(404, "웨이퍼를 찾지 못했습니다")
            src9 = str(z9["srcs"][ids9.index(wid9)])
        recs9, _a, _b, _c = S._load_aligned(Path(src9), stems=[wid9])
        if not recs9:
            raise HTTPException(404, "웨이퍼 파일을 읽지 못했습니다")
        rec9 = recs9[0]
        cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
        sn9 = str(hx9.get("sensor") or "")
        st9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
        arrh9 = np.asarray(rec9["sensors"], dtype=np.float64)
        if "\u2212" in sn9:                           # 파생 차분 가설(A−B)
            a9x, b9x = sn9.split("\u2212", 1)
            if a9x not in cols9 or b9x not in cols9:
                raise HTTPException(400, f"이 웨이퍼에 {sn9} 구성 센서 없음")
            y9 = (arrh9[:, cols9.index(a9x)] - arrh9[:, cols9.index(b9x)])
        elif sn9 not in cols9:
            raise HTTPException(400, f"이 웨이퍼에 {sn9} 없음")
        else:
            y9 = arrh9[:, cols9.index(sn9)]
        m9 = st9 == int(hx9.get("step") or 0)
        i9 = np.where(m9)[0]
        roi9 = [int(i9[0]), int(i9[-1])] if len(i9) else None
        marks9 = [{"i": int(s09), "label": f"step {sp9}"}
                  for sp9, s09, e09 in _step_runs(st9)]
        val9 = None
        _FN9 = _feats9 = None
        try:                                           # 번들 루트 / 개발 패키지 둘 다
            from pattern_hunter import _feats as _feats9, F_NAMES as _FN9
        except Exception:                              # noqa: BLE001
            try:
                from io_recon_tcn.pattern_hunter import (_feats as _feats9,
                                                         F_NAMES as _FN9)
            except Exception:                          # noqa: BLE001
                _FN9 = None
        if _FN9 and len(i9) and str(hx9.get("feature")) in _FN9:
            fv9 = _feats9(sys.modules[__name__], y9[i9[0]:i9[-1] + 1])
            val9 = float(fv9[_FN9.index(str(hx9["feature"]))])
        z9v = None
        if val9 is not None and hx9.get("c_med") is not None \
                and hx9.get("c_iqr"):
            z9v = (val9 - float(hx9["c_med"])) / (0.7413 * float(hx9["c_iqr"]))
        beyond9 = (val9 is not None and hx9.get("fence") is not None
                   and ((val9 > float(hx9["fence"]))
                        if int(hx9.get("direction") or 1) > 0
                        else (val9 < float(hx9["fence"]))))
        return {"id": wid9, "sensor": sn9,
                "y": [None if not np.isfinite(v9) else round(float(v9), 4)
                      for v9 in y9],
                "roi": roi9, "stepmarks": marks9,
                "val": (None if val9 is None else round(val9, 4)),
                "fence": hx9.get("fence"),
                "z": (None if z9v is None else round(z9v, 2)),
                "beyond": bool(beyond9)}

    @app.post("/api/dv_list")
    def api_dv_list(payload: dict = Body(...)):
        """뷰어 목록: 폴더(;다중) → 웨이퍼 전량(시간순). 골든 불필요."""
        wfs9 = split_folders(payload.get("folder"))
        if not wfs9:
            raise HTTPException(400, "폴더를 입력하세요")
        miss9 = [str(w9) for w9 in wfs9 if not w9.is_dir()]
        if miss9:
            raise HTTPException(400, "폴더가 없습니다: " + " ; ".join(miss9))
        per9, n_all9, _n, _av = select_stems_multi(S, wfs9, None, None, None, None)
        day_map9, _s, _d = _stems_by_day(per9, 0)
        id2src9 = {st9: f9 for prs9 in day_map9.values() for f9, st9 in prs9}
        id2ts9 = {}
        for st9 in id2src9:
            try:
                id2ts9[st9] = S._end_ts_and_date(st9)[0]
            except Exception:                          # noqa: BLE001
                id2ts9[st9] = ""
        DVD.clear()
        DVD.update(id2src=id2src9,
                   id2date={st9: d9 for d9, prs9 in day_map9.items()
                            for _f9, st9 in prs9},
                   id2ts=id2ts9)
        _state_save("dv", {"folder": str(payload.get("folder") or "")})
        avail9 = {"eqp": {}, "dsv": {}, "recipe": {}}
        items9 = []
        for i9 in id2src9:
            m9 = _meta_of(S, i9)
            for k9 in ("eqp", "dsv", "recipe"):
                v9 = m9[k9] or "(없음)"
                avail9[k9][v9] = avail9[k9].get(v9, 0) + 1
            items9.append({"id": i9, "date": DVD["id2date"].get(i9, ""),
                           "ts": id2ts9.get(i9, ""), "e": m9["eqp"],
                           "d": m9["dsv"], "r": m9["recipe"]})
        items9.sort(key=lambda x9: (x9["ts"] or x9["date"], x9["id"]))
        return {"total": len(items9), "items": items9, "avail": avail9}

    @app.post("/api/dv_sensors")
    def api_dv_sensors(payload: dict = Body(...)):
        """웨이퍼 1장을 열어 '그 파일의' 센서 전체 목록(가상 포함)과 길이."""
        wid9 = str(payload.get("id") or "")
        f9 = (DVD.get("id2src") or {}).get(wid9)
        if not f9:
            raise HTTPException(400, "먼저 [목록 불러오기]를 실행하고 목록의 웨이퍼를 클릭하세요")
        recs9, _a, cols9, _c = S._load_aligned(Path(f9), stems=[wid9])
        if not recs9:
            raise HTTPException(404, "웨이퍼 파일을 읽지 못했습니다")
        T9 = int(np.asarray(recs9[0]["sensors"]).shape[0])
        return {"id": wid9, "sensors": [str(c9) for c9 in cols9], "T": T9}

    @app.post("/api/dv_wave")
    def api_dv_wave(payload: dict = Body(...)):
        """원본 곡선(정렬·다운샘플 없음): 실센서 또는 수식. 웨이퍼별 자기 길이."""
        ids9 = [str(v9) for v9 in (payload.get("ids") or [])][:20]
        if not ids9:
            raise HTTPException(400, "웨이퍼를 체크하세요")
        sn9 = str(payload.get("sensor") or "").strip()
        fx9 = str(payload.get("formula") or "").strip()
        if not sn9 and not fx9:
            raise HTTPException(400, "센서 또는 수식을 지정하세요")
        by_f9 = {}
        for w9 in ids9:
            f9 = (DVD.get("id2src") or {}).get(w9)
            if f9:
                by_f9.setdefault(f9, []).append(w9)
        curves9 = []
        for f9, sts9 in by_f9.items():
            recs9, _a, _b, _c = S._load_aligned(Path(f9), stems=sts9)
            for rec9 in recs9:
                cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
                arr9 = np.asarray(rec9["sensors"], dtype=np.float64)
                if fx9:
                    y9, err9 = _formula_eval(fx9, cols9, arr9)
                    if err9:
                        raise HTTPException(400, err9)
                else:
                    if sn9 not in cols9:
                        y9, err9 = _formula_eval(sn9, cols9, arr9)  # 유사명 해석 겸용
                        if err9:
                            raise HTTPException(400, f"'{sn9}' 없음 — {err9}")
                    else:
                        y9 = arr9[:, cols9.index(sn9)]
                curves9.append({"id": str(rec9["wafer_id"]),
                                "vals": [None if not np.isfinite(v9)
                                         else round(float(v9), 4) for v9 in y9]})
            del recs9
        order9 = {w9: k9 for k9, w9 in enumerate(ids9)}
        curves9.sort(key=lambda c9: order9.get(c9["id"], 999))
        return {"label": ("수식: " + fx9) if fx9 else sn9,
                "curves": curves9, "n": len(curves9)}

    @app.post("/api/cal_marks")
    def api_cal_marks(payload: dict = Body(...)):
        """불량 내역 CSV → 달력 표시용 마크. 형식: 날짜,설비,DSV,불량내역(헤더 유연).

        설비/DSV 칸이 비어 있으면 전체에 적용. 반환 마크는 두 달력 탭이 공용.
        """
        hint9 = sorted(set((VZD.get("id2date") or {}).values())) or None
        events9, skipped9 = _parse_label_csv(payload.get("csv"), days_hint=hint9)
        marks9 = []
        for e9 in events9:
            d9 = str(e9.get("dsv") or "").strip().lower()
            if d9.isdigit():
                d9 = "dsv" + d9
            txt9 = e9["label"] + ((" · " + e9["memo"]) if e9.get("memo") else "")
            marks9.append({"date": e9["date"], "eqp": str(e9.get("eqp") or "").lower(),
                           "dsv": d9, "text": txt9})
        if not marks9:
            raise HTTPException(400, "CSV 에서 불량 내역을 읽지 못했습니다 — "
                                     + " / ".join(skipped9[:3]))
        return {"marks": marks9, "n": len(marks9), "skipped": skipped9[:10]}

    @app.post("/api/sd_events")
    def api_sd_events(payload: dict = Body(...)):
        """레이블(조치내역) 사건 분석 — 사례-교차 대조로 원인 후보 랭킹.

        사건 창(사건일 포함 evt_days) vs 기준 창(그 앞 base_days) 를 daymu 로 자기
        대조. z=|Δ평균|/기준σ 랭킹 + 발화일 역추적(점진/급발) + 물리 위반 첨부.
        """
        if not SDD or SDD.get("daymu") is None:
            raise HTTPException(400, "스텝 차이 집계가 없습니다 — 먼저 실행하세요")
        evd = max(1, min(14, int(payload.get("evt_days") or 3)))
        bad = max(5, min(60, int(payload.get("base_days") or 14)))
        events, skipped = _parse_label_csv(payload.get("csv"))
        # ── DSV 구분: CSV 의 dsv 칸을 '집계에 쓴 DSV 필터'와 맞춘다(v110).
        #    지정이 없으면 스텝 차이 실행 때 쓴 DSV 필터를 자동 사용.
        dsv_in9 = payload.get("dsv")
        if dsv_in9 is None:
            dsv_in9 = ((_state_load().get("sd") or {}).get("dsv") or "")
        dsv_f9 = [("dsv" + t9 if t9.isdigit() else t9.lower())
                  for t9 in _parse_multi(dsv_in9)]

        def _nd9(v9):
            v9 = str(v9 or "").strip().lower()
            return ("dsv" + v9 if v9.isdigit() else v9)
        if dsv_f9:
            keep9 = []
            for e9 in events:
                ed9 = _nd9(e9.get("dsv"))
                if not ed9 or ed9 in dsv_f9:
                    keep9.append(e9)
                else:
                    skipped.append(f"{e9['date']} {e9['label']}: DSV 불일치"
                                   f"({ed9} ≠ {','.join(dsv_f9)})")
            events = keep9
        _state_save("sde", {"csv": str(payload.get("csv") or ""),
                            "evt_days": evd, "base_days": bad})
        days_u = SDD["days_u"]
        daymu = SDD["daymu"]
        thr = SDD["thr"]
        sensors, usteps = SDD["sensors"], SDD["usteps"]
        excl9 = set(SDD.get("excl_sensors") or [])
        pos = {d9: i9 for i9, d9 in enumerate(days_u)}
        fd = (SDJOB.get("result") or {}).get("findings") or {}
        import datetime as _dtv

        def _win(d0_9, d1_9):                          # 달력 창 → 가용 일 인덱스
            return [pos[d9] for d9 in days_u if d0_9 <= d9 <= d1_9]
        out_ev = []
        for ev in events:
            D9 = _dtv.date.fromisoformat(ev["date"])
            if not (days_u[0] <= ev["date"] <= days_u[-1]):
                skipped.append(f"{ev['date']} {ev['label']}: 집계 기간 밖")
                continue
            e_i = _win((D9 - _dtv.timedelta(days=evd - 1)).isoformat(), ev["date"])
            b_i = _win((D9 - _dtv.timedelta(days=evd + bad - 1)).isoformat(),
                       (D9 - _dtv.timedelta(days=evd)).isoformat())
            if len(e_i) < 1 or len(b_i) < 5:
                skipped.append(f"{ev['date']} {ev['label']}: 대조 기간 데이터 부족"
                               f"(사건 {len(e_i)}일·기준 {len(b_i)}일)")
                continue
            cands = []
            for si9, sn in enumerate(sensors):
                if sn in excl9:
                    continue
                for ki9, u9 in enumerate(usteps):
                    vb = daymu[b_i, si9, ki9]
                    ve = daymu[e_i, si9, ki9]
                    vb = vb[np.isfinite(vb)]
                    ve = ve[np.isfinite(ve)]
                    if len(vb) < 5 or len(ve) < 1:
                        continue
                    mb, sb = float(np.mean(vb)), float(np.std(vb))
                    me = float(np.mean(ve))
                    den = max(sb, float(thr[si9, ki9]) / 2.0, 1e-6)
                    z9 = abs(me - mb) / den
                    if z9 < 1.5:
                        continue
                    cands.append({"sensor": sn, "step": int(u9),
                                  "z": round(z9, 1), "delta": round(me - mb, 3),
                                  "base": round(mb, 3), "sig": z9 >= 3.0,
                                  "virtual": bool(re.match(r"(?i)^d[tp]_", sn)),
                                  "_si": si9, "_ki": ki9, "_mb": mb,
                                  "_sb": max(sb, float(thr[si9, ki9]) / 2.0)})
            cands.sort(key=lambda x9: -x9["z"])
            cands = cands[:15]
            # 발화일 역추적: 기준평균±2σb 를 벗어나 사건일까지 유지되기 시작한 날
            look_i = _win((D9 - _dtv.timedelta(days=30)).isoformat(), ev["date"])
            for cd in cands:
                v9 = daymu[look_i, cd["_si"], cd["_ki"]]
                dev9 = np.abs(v9 - cd["_mb"]) > 2.0 * cd["_sb"]
                onset9 = None
                for s9 in range(len(look_i)):
                    seg9 = dev9[s9:]
                    fin9 = np.isfinite(v9[s9:])
                    if fin9.sum() and (seg9 & fin9).sum() >= 0.8 * fin9.sum():
                        onset9 = s9
                        break
                if onset9 is None:
                    cd["onset"] = None
                    cd["kind"] = "당일 인접"
                else:
                    od9 = days_u[look_i[onset9]]
                    nb9 = (D9 - _dtv.date.fromisoformat(od9)).days
                    cd["onset"] = od9
                    if nb9 <= 1:
                        cd["kind"] = "급발(당일)"
                    else:
                        cd["kind"] = "점진"
                        vv9 = v9[onset9:]
                        fin9 = np.isfinite(vv9)
                        if fin9.sum() >= 2:
                            cd["slope"] = round(float(
                                (vv9[fin9][-1] - vv9[fin9][0]) / max(1, nb9)), 3)
                for k9 in ("_si", "_ki", "_mb", "_sb"):
                    cd.pop(k9, None)
            # 물리 위반·부호 위반 교차 첨부
            e_d0 = (D9 - _dtv.timedelta(days=evd - 1)).isoformat()

            def _ovl(a0, a1):
                return not (a1 < e_d0 or a0 > ev["date"])
            phys_hit = ([p9 for p9 in (fd.get("phys") or [])
                         if _ovl(p9["d0"], p9["d1"])]
                        + [dict(v9, kind="부호 위반") for v9 in
                           (fd.get("violations") or []) if _ovl(v9["d0"], v9["d1"])])
            out_ev.append({"date": ev["date"], "label": ev["label"],
                           "eqp": ev["eqp"], "dsv": ev["dsv"], "memo": ev["memo"],
                           "evt_d0": e_d0, "base_d0": days_u[b_i[0]],
                           "base_d1": days_u[b_i[-1]],
                           "cands": cands, "phys": phys_hit[:8]})
        # 같은 레이블 다건 → 공통 후보(2건 이상 등장)
        by_lb = {}
        for ev in out_ev:
            by_lb.setdefault(ev["label"], []).append(ev)
        common = {}
        for lb9, evs9 in by_lb.items():
            if len(evs9) < 2:
                continue
            cnt9 = {}
            for ev in evs9:
                for cd in ev["cands"]:
                    k9 = f"{cd['sensor']}|step{cd['step']}"
                    cnt9[k9] = cnt9.get(k9, 0) + 1
            com9 = [(k9, n9) for k9, n9 in cnt9.items() if n9 >= 2]
            com9.sort(key=lambda x9: -x9[1])
            if com9:
                common[lb9] = [{"key": k9, "n": n9, "total": len(evs9)}
                               for k9, n9 in com9[:8]]
        res9 = {"events": out_ev, "skipped": skipped, "common": common,
                "params": {"evt_days": evd, "base_days": bad,
                           "dsv": dsv_f9 or ["(전체)"]}}
        SDE.clear()
        SDE.update(res9)
        return res9

    @app.post("/api/sd_event_narrate")
    def api_sd_event_narrate(payload: dict = Body(...)):
        """사건 1건의 LLM 원인 분석 — 후보 랭킹·발화·물리 위반·공통 서명 동봉."""
        idx = int(payload.get("index") or 0)
        evs = SDE.get("events") or []
        if not evs or idx >= len(evs):
            raise HTTPException(400, "사건 분석을 먼저 실행하세요")
        ev = evs[idx]
        unit = lambda sn: "bar" if re.match(r"(?i)^d?p", sn) else "℃"
        lines = [f"[사건] {ev['date']} — {ev['label']}"
                 + (f" [{ev['dsv']}]" if ev.get("dsv") else "")
                 + (f" (설비 {ev['eqp']})" if ev.get("eqp") else "")
                 + (f" · 메모: {ev['memo']}" if ev.get("memo") else ""),
                 f"사건 창 {ev['evt_d0']}~{ev['date']} · "
                 f"기준 창 {ev['base_d0']}~{ev['base_d1']} (자기 대조)"]
        if ev.get("phys"):
            lines.append("[물리 위반 — 사건 창과 겹침, 최우선]")
            for p9 in ev["phys"]:
                lines.append(f"{p9['kind']}: {p9.get('sensor') or ''}"
                             + (f" (밸브 {p9['io']})" if p9.get("io") else "")
                             + f" — {p9['d0']}~{p9['d1']}")
        lines.append("[원인 후보 랭킹 — 사건 창 vs 기준 창]")
        for cd in ev["cands"]:
            lines.append(
                f"{cd['sensor']} step{cd['step']}: Δ {cd['delta']:+g}{unit(cd['sensor'])} "
                f"(기준 대비 z={cd['z']}{'·유의' if cd['sig'] else ''}) · {cd['kind']}"
                + (f" · 발화 {cd['onset']}" if cd.get("onset") else "")
                + (f" · 일당 {cd['slope']:+g}" if cd.get("slope") is not None else "")
                + (" [가상센서]" if cd.get("virtual") else ""))
        com9 = (SDE.get("common") or {}).get(ev["label"])
        if com9:
            lines.append(f"[같은 레이블('{ev['label']}') 공통 후보 — 재발 서명 후보]")
            for c9 in com9:
                lines.append(f"{c9['key']} — {c9['total']}건 중 {c9['n']}건에서 상위")
        from io_recon_tcn import rca_causal as RC
        return RC.narrate_event("\n".join(lines))

    @app.post("/api/sd_wave")
    def api_sd_wave(payload: dict = Body(...)):
        """바 클릭 상세: 웨이퍼 원본 파형 + 골든 오버레이(원본 대 원본)."""
        if not SDD or not SDD.get("id2src"):
            raise HTTPException(400, "스텝 차이 집계가 없습니다 — 먼저 실행하세요")
        wid = str(payload.get("id") or "")
        sn = str(payload.get("sensor") or "")
        fol = SDD["id2src"].get(wid)
        if not fol:
            raise HTTPException(400, f"웨이퍼 원본 위치를 모릅니다: {wid}")
        recs, _i9, _s9, _c9 = S._load_aligned(Path(fol), stems=[wid])
        rec = next((r9 for r9 in recs if str(r9["wafer_id"]) == wid), None)
        if rec is None:
            raise HTTPException(400, f"원본 파일을 다시 읽지 못했습니다: {wid}")
        cols = [str(c) for c in (rec.get("sensor_cols") or [])]
        if sn not in cols:
            raise HTTPException(400, f"이 웨이퍼에 없는 센서: {sn}")
        w9 = np.asarray(rec["sensors"], dtype=np.float64)[:, cols.index(sn)]
        T9 = len(w9)
        LD = min(T9, 4000)                             # 원본 해상도 유지(통상 3700급 전체)
        wd = _resample(w9, LD)
        gsrc = (SDD.get("rmean") or {}).get(sn)
        try:
            st9 = np.asarray(rec.get("step"), dtype=float)
            i9 = np.clip(np.round(np.linspace(0, len(st9) - 1, LD)).astype(int),
                         0, len(st9) - 1)
            steps9 = [int(v9) for v9 in st9[i9]]
        except Exception:                              # noqa: BLE001
            steps9 = None
        gd = None
        aligned = False
        sref9 = SDD.get("steps_ref")
        if gsrc is not None:
            gsrc = np.asarray(gsrc, dtype=np.float64)
            if sref9 is not None and steps9 is not None:
                # ── 스텝 경계 정렬 워핑: 골든의 각 스텝 조각을 웨이퍼의 해당 스텝
                #    길이에 맞춰 리샘플해 이어붙인다 — 공정 길이가 달라도 피크 싱크 유지.
                sref9 = np.asarray(sref9)
                gd = np.full(LD, np.nan)
                i0 = 0
                while i0 < LD:
                    u9 = steps9[i0]
                    j0 = i0
                    while j0 + 1 < LD and steps9[j0 + 1] == u9:
                        j0 += 1
                    seg_mask = sref9 == u9
                    if seg_mask.any():
                        gd[i0:j0 + 1] = _resample(gsrc[seg_mask], j0 - i0 + 1)
                    i0 = j0 + 1
                aligned = True
                gap9 = ~np.isfinite(gd)
                if gap9.any():                          # 골든에 없는 스텝은 전역 리샘플로 메움
                    gd[gap9] = _resample(gsrc, LD)[gap9]
            else:
                gd = _resample(gsrc, LD)
        return {"id": wid, "sensor": sn, "T": T9, "L": LD,
                "wafer": [round(float(v9), 4) for v9 in wd],
                "golden": ([None if not np.isfinite(v9) else round(float(v9), 4)
                            for v9 in gd] if gd is not None else None),
                "aligned": aligned,
                "steps": steps9}

    VZD = {}                                       # 데이터 시각화 탭 상태(스텝 차이와 독립)

    @app.post("/api/viz_list")
    def api_viz_list(payload: dict = Body(...)):
        """데이터 시각화(독립 탭): 자체 폴더+필터+골든으로 웨이퍼 목록 구성."""
        if payload.get("folder"):                      # 폴더가 오면 새로 스캔
            wfs = split_folders(payload.get("folder"))
            if not wfs:
                raise HTTPException(400, "폴더를 입력하세요")
            missing = [str(w9) for w9 in wfs if not w9.is_dir()]
            if missing:
                raise HTTPException(400, "폴더가 없습니다: " + " ; ".join(missing))
            store = _gstore_dir(payload.get("store"))
            gfile = str(payload.get("gfile") or "").strip()
            fp = store / gfile
            if not gfile or not fp.is_file():
                raise HTTPException(400, "기준 골든 파일을 선택하세요")
            ref = json.loads(fp.read_text(encoding="utf-8"))
            if ref.get("avgwave_golden") != 1:
                raise HTTPException(400, "기준 파일이 저장 골든 형식이 아닙니다")
            # ★메타(태그·끝시각)·파싱 캐시를 저장소로 — 앱 버전 교체에도 재태깅 없음
            cache_set9 = (_state_load().get("sd") or {}).get("cache")
            _install_meta_cache(S, cache_set9 or store)
            _install_parse_cache(S, store_hint=str(store))
            _install_listing_cache(S)
            per, n_all, n_sel, avail = select_stems_multi(
                S, wfs, payload.get("dsv"), payload.get("recipe"),
                payload.get("months"), payload.get("eqp"))
            if any(x9[1] is not None for x9 in per) and n_sel == 0:
                raise HTTPException(400, f"필터를 통과한 파일이 없습니다 (전체 {n_all}장). "
                                         + _avail_text(avail))
            day_map, n_seen, n_dated = _stems_by_day(per, 0)
            if n_dated == 0:
                raise HTTPException(400, f"날짜 태그가 있는 파일이 없습니다(전체 {n_seen}장). "
                                         "웨이퍼 분석 탭에서 이 폴더를 한 번 불러오면 태깅됩니다")
            id2ts9, n_fb9 = {}, 0
            for d9, prs9 in day_map.items():
                for _f9, st9 in prs9:
                    try:
                        ts9 = str(S._end_ts_and_date(st9)[0] or "")
                    except Exception:                  # noqa: BLE001
                        ts9 = ""
                    if len(ts9) < 16:                  # 시각 없음 → 날짜 00:00 폴백
                        ts9 = f"{d9} 00:00:00"
                        n_fb9 += 1
                    id2ts9[st9] = ts9
            if n_fb9:
                print(f"[viz] 시각 정보 없는 파일 {n_fb9}장 — 날짜 00:00 으로 대체"
                      "(장수/시간 묶음 순서가 날짜 단위로만 정확)", flush=True)
            peekT9 = None                              # 격자 길이는 로드 시 1회 확정
            try:
                for _d9, prs9 in sorted(day_map.items()):
                    if not prs9:
                        continue
                    f9x, st9x = prs9[0]
                    rp9, _a, _b, _c = S._load_aligned(Path(f9x), stems=[st9x])
                    if rp9:
                        peekT9 = int(np.asarray(rp9[0]["sensors"]).shape[0])
                    del rp9
                    break
            except Exception as e9:                    # noqa: BLE001
                print(f"[viz] 원본 길이 확인 실패(골든 길이 사용): {e9}", flush=True)
            VZD.clear()
            VZD.update(
                id2ts=id2ts9, ts_fallback=n_fb9, peek_T=peekT9,
                rmean={sn9: np.asarray(v9, dtype=np.float64)
                       for sn9, v9 in (ref.get("mean") or {}).items()},
                rstd={sn9: np.asarray(v9, dtype=np.float64)
                      for sn9, v9 in (ref.get("std") or {}).items()},
                steps_ref=(np.asarray(ref.get("steps")).astype(int)
                           if ref.get("steps") else None),
                sensors=list(ref.get("sensors") or []),
                id2src={st9: f9 for prs9 in day_map.values() for f9, st9 in prs9},
                id2date={st9: d9 for d9, prs9 in day_map.items() for _f9, st9 in prs9},
                id2meta={st9: _meta_of(S, st9)
                         for prs9 in day_map.values() for _f9, st9 in prs9},
                ref_name=str(ref.get("name") or ""),
                store=str(store), gfile=str(gfile))
            _state_save("vz", {"folder": str(payload.get("folder") or ""),
                               "eqp": str(payload.get("eqp") or ""),
                               "dsv": str(payload.get("dsv") or ""),
                               "recipe": str(payload.get("recipe") or ""),
                               "months": str(payload.get("months") or ""),
                               "store": str(store), "gfile": gfile})
        if not VZD.get("id2date"):
            raise HTTPException(400, "폴더/기준 골든을 입력하고 목록을 불러오세요")
        sel9 = set(payload.get("dates") or [])
        ts_map0 = VZD.get("id2ts") or {}
        out9 = sorted(({"id": i9, "date": d9,
                        "ts": ts_map0.get(i9, d9 + " 00:00:00")}
                       for i9, d9 in VZD["id2date"].items()
                       if not sel9 or d9 in sel9),
                      key=lambda x9: (x9["ts"], x9["id"]))
        sref9 = VZD.get("steps_ref")
        return {"total": len(out9), "items": out9,     # 전량(클라 가상 스크롤)
                "dates": sorted(set(VZD["id2date"].values())),
                "ts_fallback": int(VZD.get("ts_fallback") or 0),
                "sensors": VZD["sensors"], "ref": VZD.get("ref_name"),
                "steps": (sorted({int(v9) for v9 in sref9}) if sref9 is not None else [])}

    @app.post("/api/viz_wave")
    def api_viz_wave(payload: dict = Body(...)):
        """선택 웨이퍼들의 원본 파형을 골든 스텝 격자에 정렬해 반환(겹침·평균 공용).

        모든 곡선을 골든 steps_ref 격자에 스텝 런별 워핑 — 공정 길이가 달라도
        피크 싱크가 맞은 상태로 겹쳐진다(v98 정렬 인프라 재사용).
        """
        if not VZD.get("id2src"):
            raise HTTPException(400, "먼저 [웨이퍼 목록 불러오기]를 실행하세요")
        sn = str(payload.get("sensor") or "")
        bd9 = int(payload.get("bucket_days") or 0)
        ids = [str(x9) for x9 in (payload.get("ids") or [])][:60]
        if bd9 > 0:
            # N일 버킷 평균: 체크한 날짜들의 웨이퍼를 서버가 직접 선별(버킷별 표본 캡)
            import datetime as _dtb
            sel9 = set(payload.get("dates") or [])
            cand9 = sorted(((d9, i9) for i9, d9 in VZD["id2date"].items()
                            if not sel9 or d9 in sel9))
            if not cand9:
                raise HTTPException(400, "기간 내 웨이퍼가 없습니다")
            d0b = _dtb.date.fromisoformat(cand9[0][0])
            by_bk9 = {}
            for d9, i9 in cand9:
                bk9 = (_dtb.date.fromisoformat(d9) - d0b).days // bd9
                by_bk9.setdefault(bk9, []).append(i9)
            n_bk9 = len(by_bk9)
            if n_bk9 > 60:
                raise HTTPException(400, f"버킷이 {n_bk9}개 — 평균 단위를 키우거나 기간을 좁혀 주세요(최대 60)")
            per9 = max(1, min(8, 120 // max(1, n_bk9)))   # 총 로드 ≤120장
            bucket_ids9 = {bk9: v9[:per9] for bk9, v9 in by_bk9.items()}
            ids = [i9 for v9 in bucket_ids9.values() for i9 in v9]
        elif not ids:
            raise HTTPException(400, "웨이퍼를 선택하세요")
        if sn not in (VZD.get("rmean") or {}):
            raise HTTPException(400, f"골든에 없는 센서: {sn}")
        canon_runs9, Lg, steps_t9, golden_t9 = _viz_grid()
        gsrc = np.asarray(golden_t9[sn], dtype=np.float64)
        sref9 = steps_t9
        by_f9 = {}
        d_map9 = VZD["id2date"]
        for wid9 in ids:
            f9 = VZD["id2src"].get(wid9)
            if f9:
                by_f9.setdefault(f9, []).append(wid9)
        curves9 = []
        for f9, st9s in by_f9.items():
            recs9, _a, _b, _c = S._load_aligned(Path(f9), stems=st9s)
            for rec9 in recs9:
                cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
                if sn not in cols9:
                    continue
                arr9 = np.asarray(rec9["sensors"], dtype=np.float64)[:, [cols9.index(sn)]]
                if canon_runs9 is not None:
                    try:
                        stw9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
                        wv9 = _warp_to_canon(arr9, stw9, canon_runs9, Lg)[:, 0]
                    except Exception:                  # noqa: BLE001
                        wv9 = _resample(arr9[:, 0], Lg)
                else:
                    wv9 = _resample(arr9[:, 0], Lg)
                curves9.append({"id": str(rec9["wafer_id"]),
                                "date": d_map9.get(str(rec9["wafer_id"]), ""),
                                "vals": [None if not np.isfinite(v9) else
                                         round(float(v9), 4) for v9 in wv9]})
        if not curves9:
            raise HTTPException(400, "원본 파형을 읽지 못했습니다(폴더 연결 확인)")
        if bd9 > 0:                                    # 버킷별 평균 곡선으로 재구성
            import datetime as _dtb
            cur_by9 = {c9["id"]: c9 for c9 in curves9}
            bcurves9 = []
            for bk9 in sorted(bucket_ids9):
                vs9 = [cur_by9[i9]["vals"] for i9 in bucket_ids9[bk9]
                       if i9 in cur_by9]
                if not vs9:
                    continue
                with np.errstate(all="ignore"):
                    m9b = np.nanmean(np.array(
                        [[np.nan if v9 is None else v9 for v9 in v8] for v8 in vs9],
                        dtype=np.float64), axis=0)
                s0b = d0b + _dtb.timedelta(days=bk9 * bd9)
                s1b = s0b + _dtb.timedelta(days=bd9 - 1)
                bcurves9.append({"id": f"{s0b.isoformat()[5:]}~{s1b.isoformat()[5:]} "
                                       f"({len(vs9)}장)",
                                 "date": s0b.isoformat(),
                                 "vals": [None if not np.isfinite(v9) else
                                          round(float(v9), 4) for v9 in m9b]})
            curves9 = bcurves9
        with np.errstate(all="ignore"):
            M9 = np.array([[np.nan if v9 is None else v9 for v9 in c9["vals"]]
                           for c9 in curves9], dtype=np.float64)
            mean9 = np.nanmean(M9, axis=0)
        return {"sensor": sn, "L": Lg, "bucket_days": bd9,
                "golden": [round(float(v9), 4) for v9 in gsrc],
                "steps": ([int(v9) for v9 in sref9] if sref9 is not None else None),
                "curves": curves9,
                "mean": [None if not np.isfinite(v9) else round(float(v9), 4)
                         for v9 in mean9],
                "n": len(curves9)}

    def _viz_grid(peek_T=None):
        """(peek_T 는 폴백 — 기본은 목록 로드 시 확정한 VZD['peek_T'] 사용)"""
        """시각화 정렬 격자: 골든 스텝 비율을 유지한 채 '원본 해상도'로 확장.

        골든은 생성 시 600pt 로 압축되므로 그 격자에 맞추면 3700pt 원본이 6배
        다운샘플된다(짧은 스파이크 소실). 목표 길이 L_t(원본 길이, 최대 4000)로
        스텝 런 길이를 비례 배분한 캐논 격자를 만들고, 골든도 그 격자로 워핑한다.
        반환: (canon_runs, L_t, steps_t, golden_t{sn:배열}) — 실패 시 (None, Lg, …).
        """
        rm0 = VZD.get("rmean") or {}
        if not rm0:
            return None, 0, None, {}
        Lg = len(next(iter(rm0.values())))
        sref = VZD.get("steps_ref")
        if sref is None:
            return None, Lg, None, {sn: np.asarray(v, dtype=np.float64)
                                    for sn, v in rm0.items()}
        sref = np.asarray(sref)
        runs = _step_runs(sref)
        total = sum(e - s for _u, s, e in runs) or 1
        # 격자 길이는 데이터셋 단위로 고정(요청에 포함된 웨이퍼에 따라 흔들리면
        # 호출마다 곡선 길이가 달라져 비교·저장이 어긋난다)
        pT9 = VZD.get("peek_T") or peek_T or Lg
        L_t = int(min(4000, max(Lg, int(pT9))))
        canon, pos = [], 0
        for i9, (u9, s9, e9) in enumerate(runs):
            ln9 = ((L_t - pos) if i9 == len(runs) - 1
                   else max(1, int(round(L_t * (e9 - s9) / total))))
            ln9 = min(ln9, L_t - pos)
            if ln9 <= 0:
                continue
            canon.append((u9, pos, ln9))
            pos += ln9
        steps_t = np.zeros(L_t, dtype=int)
        for u9, p9, l9 in canon:
            steps_t[p9:p9 + l9] = u9
        gmat = np.column_stack([np.asarray(rm0[sn], dtype=np.float64)
                                for sn in sorted(rm0)])
        gw = _warp_to_canon(gmat, sref, canon, L_t)
        for j9 in range(gw.shape[1]):                  # 결손은 전역 리샘플로 메움
            bad = ~np.isfinite(gw[:, j9])
            if bad.any():
                gw[bad, j9] = _resample(gmat[:, j9], L_t)[bad]
        golden_t = {sn: gw[:, j9] for j9, sn in enumerate(sorted(rm0))}
        return canon, L_t, steps_t, golden_t

    @app.post("/api/viz_multi")
    def api_viz_multi(payload: dict = Body(...)):
        """센서 오버레이: 선택 기간 전체 평균을 센서별 1선씩 — 한 그래프용.

        기간 웨이퍼를 균등 표본(≤60장)으로 한 번만 로드해 모든 센서를 함께 평균.
        """
        if not VZD.get("id2src"):
            raise HTTPException(400, "먼저 [웨이퍼 목록 불러오기]를 실행하세요")
        sens = [str(s9) for s9 in (payload.get("sensors") or [])][:12]
        if not sens:
            raise HTTPException(400, "센서를 선택하세요")
        sel9 = set(payload.get("dates") or [])
        cand9 = sorted(((d9, i9) for i9, d9 in VZD["id2date"].items()
                        if not sel9 or d9 in sel9))
        if not cand9:
            raise HTTPException(400, "기간 내 웨이퍼가 없습니다")
        stp9 = max(1, len(cand9) // 60)                # 균등 표본 ≤60장
        pick9 = [i9 for _d9, i9 in cand9[::stp9]][:60]
        sref9 = VZD.get("steps_ref")
        rm0 = VZD.get("rmean") or {}
        miss9 = [s9 for s9 in sens if s9 not in rm0]
        if miss9:
            raise HTTPException(400, f"골든에 없는 센서: {', '.join(miss9)}")
        canon_runs9, Lg, steps_t9, _g9 = _viz_grid()
        sref9 = steps_t9
        by_f9 = {}
        for wid9 in pick9:
            f9 = VZD["id2src"].get(wid9)
            if f9:
                by_f9.setdefault(f9, []).append(wid9)
        acc9 = {s9: [] for s9 in sens}
        n_w9 = 0
        for f9, st9s in by_f9.items():
            recs9, _a, _b, _c = S._load_aligned(Path(f9), stems=st9s)
            for rec9 in recs9:
                cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
                ii9 = [cols9.index(s9) for s9 in sens if s9 in cols9]
                if len(ii9) != len(sens):
                    continue
                arr9 = np.asarray(rec9["sensors"], dtype=np.float64)[:, ii9]
                if canon_runs9 is not None:
                    try:
                        stw9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
                        wv9 = _warp_to_canon(arr9, stw9, canon_runs9, Lg)
                    except Exception:                  # noqa: BLE001
                        wv9 = np.column_stack([_resample(arr9[:, q9], Lg)
                                               for q9 in range(arr9.shape[1])])
                else:
                    wv9 = np.column_stack([_resample(arr9[:, q9], Lg)
                                           for q9 in range(arr9.shape[1])])
                for q9, s9 in enumerate(sens):
                    acc9[s9].append(wv9[:, q9])
                n_w9 += 1
        if not n_w9:
            raise HTTPException(400, "원본 파형을 읽지 못했습니다(폴더 연결 확인)")
        curves9 = []
        with np.errstate(all="ignore"):
            for s9 in sens:
                m9 = np.nanmean(np.stack(acc9[s9]), axis=0)
                curves9.append({"id": s9,
                                "vals": [None if not np.isfinite(v9) else
                                         round(float(v9), 4) for v9 in m9]})
        d0_9, d1_9 = cand9[0][0], cand9[-1][0]
        return {"curves": curves9, "n": n_w9, "L": Lg,
                "steps": ([int(v9) for v9 in sref9] if sref9 is not None else None),
                "period": [d0_9, d1_9]}

    import hashlib                                     # 일별 캐시 키
    VIZ_PER_DAY = 20                                   # 일별 캐시의 하루 표본 상한(고정)
    #   고정값이라야 묶음 크기를 바꿔도 같은 날이 같은 표본을 써 캐시가 맞는다.
    #   20장 = 5일 묶음이면 카드당 100장 평균(기존 83장과 비슷) · 최초 조회 비용도 동급.

    def _viz_cache_dir(Lg9):
        """<캐시 or 저장소>/viz_cache/<골든·격자·버전 키>/  — 키가 다르면 자연히 분리."""
        store9 = VZD.get("store") or ""
        if not store9:
            return None
        cache9 = (_state_load().get("sd") or {}).get("cache")   # 캐시 폴더 설정 공용
        gfile9 = VZD.get("gfile") or ""
        try:
            fp9 = Path(store9) / gfile9
            sig9 = f"{gfile9}|{fp9.stat().st_mtime_ns}|{fp9.stat().st_size}"
        except Exception:                              # noqa: BLE001
            sig9 = gfile9
        key9 = hashlib.md5(f"{sig9}|L{Lg9}|per{VIZ_PER_DAY}|{CALC_VER}"
                           .encode("utf-8")).hexdigest()[:12]
        try:
            d9 = _cache_root(cache9, store9) / "viz_cache" / key9
            d9.mkdir(parents=True, exist_ok=True)
            return d9
        except Exception as e9:                        # noqa: BLE001
            print(f"[viz] 캐시 폴더를 쓸 수 없어 캐시 없이 진행합니다: {e9}", flush=True)
            return None

    def _viz_day_pick(ids9):
        """하루 표본(고정 규칙) — 묶음 크기를 바꿔도 같은 날은 같은 웨이퍼를 쓴다."""
        if len(ids9) <= VIZ_PER_DAY:
            return list(ids9)
        stp9 = max(1, len(ids9) // VIZ_PER_DAY)
        return list(ids9)[::stp9][:VIZ_PER_DAY]

    def _viz_day_sig(ids9):
        h9 = hashlib.md5()
        for w9 in ids9:
            f9 = (VZD.get("id2src") or {}).get(w9) or ""
            try:
                stt9 = Path(f9).stat()
                h9.update(f"{w9}|{stt9.st_mtime_ns}|{stt9.st_size}".encode("utf-8"))
            except Exception:                          # noqa: BLE001
                h9.update(f"{w9}|?".encode("utf-8"))
        return h9.hexdigest()[:16]

    def _viz_day_calc(ids9, sens9, canon9, Lg9):
        """하루 표본을 실제로 읽어 (합, 개수, 장수) — 파일별 청크 스트리밍."""
        sum9 = np.zeros((Lg9, len(sens9)))
        cnt9 = np.zeros((Lg9, len(sens9)))
        n9 = 0
        by_f9 = {}
        for w9 in ids9:
            f9 = (VZD.get("id2src") or {}).get(w9)
            if f9:
                by_f9.setdefault(f9, []).append(w9)
        for f9, sts9 in by_f9.items():
            for c09 in range(0, len(sts9), 100):
                recs9, _a, _b, _c = S._load_aligned(Path(f9), stems=sts9[c09:c09 + 100])
                for rec9 in recs9:
                    cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
                    ii9 = [cols9.index(s9) for s9 in sens9 if s9 in cols9]
                    if len(ii9) != len(sens9):
                        continue
                    arr9 = np.asarray(rec9["sensors"], dtype=np.float64)[:, ii9]
                    try:
                        stw9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
                        wv9 = (_warp_to_canon(arr9, stw9, canon9, Lg9)
                               if canon9 is not None else
                               np.column_stack([_resample(arr9[:, q9], Lg9)
                                                for q9 in range(arr9.shape[1])]))
                    except Exception:                  # noqa: BLE001
                        wv9 = np.column_stack([_resample(arr9[:, q9], Lg9)
                                               for q9 in range(arr9.shape[1])])
                    fin9 = np.isfinite(wv9)
                    sum9 += np.where(fin9, wv9, 0.0)
                    cnt9 += fin9
                    n9 += 1
                del recs9
        return sum9, cnt9, n9

    def _viz_day_load(cdir9, day9, sens9, sig9, Lg9):
        """캐시에서 (합, 개수, 장수) — 센서가 모두 있고 서명이 같을 때만."""
        if cdir9 is None:
            return None
        fp9 = cdir9 / f"{day9}.npz"
        if not fp9.is_file():
            return None
        try:
            with np.load(fp9, allow_pickle=False) as z9:
                if str(z9["sig"]) != sig9:
                    return None
                cols9 = [str(c9) for c9 in z9["cols"]]
                if any(s9 not in cols9 for s9 in sens9):
                    return None
                ix9 = [cols9.index(s9) for s9 in sens9]
                if int(z9["L"]) != Lg9:
                    return None
                return (np.asarray(z9["s"])[:, ix9], np.asarray(z9["c"])[:, ix9],
                        int(z9["n"]))
        except Exception:                              # noqa: BLE001
            return None

    def _viz_day_save(cdir9, day9, sens9, sig9, Lg9, sum9, cnt9, n9):
        """센서별 합/개수 저장 — 기존 파일의 다른 센서는 보존(합집합)."""
        if cdir9 is None:
            return
        fp9 = cdir9 / f"{day9}.npz"
        cols9, S9, C9 = list(sens9), sum9, cnt9
        try:
            try:
                # 기존 파일 병합 — 실패(손상/구형식/타 프로세스 교체 경쟁)해도
                # 저장을 포기하지 않고 새로 쓴다(캐시 저장소 공유 실행 대비)
                if fp9.is_file():
                    with np.load(fp9, allow_pickle=False) as z9:
                        if str(z9["sig"]) == sig9 and int(z9["L"]) == Lg9:
                            old9 = [str(c9) for c9 in z9["cols"]]
                            add9 = [c9 for c9 in old9 if c9 not in cols9]
                            if add9:
                                ix9 = [old9.index(c9) for c9 in add9]
                                cols9 = cols9 + add9
                                S9 = np.column_stack([sum9,
                                                      np.asarray(z9["s"])[:, ix9]])
                                C9 = np.column_stack([cnt9,
                                                      np.asarray(z9["c"])[:, ix9]])
            except Exception as e9:                    # noqa: BLE001
                cols9, S9, C9 = list(sens9), sum9, cnt9
                print(f"[viz] 기존 캐시 병합 생략(새로 씀) {fp9.name}: {e9}",
                      flush=True)
            tmp9 = fp9.with_name(fp9.stem + f".tmp{os.getpid()}.npz")
            np.savez_compressed(tmp9, s=S9.astype(np.float64), c=C9.astype(np.float64),
                                cols=np.asarray([str(c9) for c9 in cols9]),
                                n=np.int64(n9), L=np.int64(Lg9),
                                sig=np.asarray(sig9))
            os.replace(tmp9, fp9)
        except Exception as e9:                        # noqa: BLE001
            print(f"[viz] 일별 캐시 저장 실패(진행에는 무해) {fp9.name}: {e9}",
                  flush=True)

    @app.post("/api/viz_buckets")
    def api_viz_buckets(payload: dict = Body(...)):
        """평균 그래프(카드 흐름): N일 버킷마다 카드 1개 — 체크 센서들의 평균 곡선.

        버킷별 표본을 공유 로드해 모든 센서를 함께 평균. ids 는 버킷의 '전체' 웨이퍼
        목록(카드 체크 → CSV 저장용). bucket_days=0 이면 선택 기간 전체 1버킷.
        """
        if not VZD.get("id2src"):
            raise HTTPException(400, "먼저 [웨이퍼 목록 불러오기]를 실행하세요")
        sens = [str(s9) for s9 in (payload.get("sensors") or [])][:12]
        if not sens:
            raise HTTPException(400, "센서를 선택하세요")
        rm0 = VZD.get("rmean") or {}
        miss9 = [s9 for s9 in sens if s9 not in rm0]
        if miss9:
            raise HTTPException(400, f"골든에 없는 센서: {', '.join(miss9)}")
        # 묶음 단위: 장수(count) / 일(day) / 시간(hour) — bucket_days 는 day 하위 호환
        gmode9 = str(payload.get("group_mode") or "day")
        gn9 = max(0, int(payload.get("group_n") or payload.get("bucket_days") or 0))
        bd9 = gn9 if gmode9 == "day" else 0
        sel9 = set(payload.get("dates") or [])
        # ids 지정 시 그 웨이퍼들만 1묶음으로(카드 '세트 저장' 이 같은 기간·같은 장수로
        # 고정 센서 그래프를 뽑을 때 사용)
        idsel9 = {str(i9) for i9 in (payload.get("ids") or [])}
        if idsel9:
            gmode9, gn9, bd9 = "day", 0, 0
        ts_map9 = VZD.get("id2ts") or {}
        cand9 = sorted(((d9, i9) for i9, d9 in VZD["id2date"].items()
                        if ((i9 in idsel9) if idsel9
                            else (not sel9 or d9 in sel9))),
                       key=lambda x9: (ts_map9.get(x9[1], x9[0] + " 00:00:00"), x9[1]))
        if not cand9:
            raise HTTPException(400, "기간 내 웨이퍼가 없습니다")
        import datetime as _dtb

        def _ts_of(i9):
            try:
                return _dtb.datetime.fromisoformat(
                    str(ts_map9.get(i9) or "").replace("/", "-")[:19])
            except Exception:                          # noqa: BLE001
                return None
        d0b = _dtb.date.fromisoformat(cand9[0][0])
        by_bk9, bk_range9 = {}, {}
        if gmode9 == "count" and gn9 > 0:
            for k9, (_d9, i9) in enumerate(cand9):
                by_bk9.setdefault(k9 // gn9, []).append(i9)
        elif gmode9 == "hour" and gn9 > 0:
            t0h9 = (_ts_of(cand9[0][1])
                    or _dtb.datetime.combine(d0b, _dtb.time()))
            t0h9 = t0h9.replace(minute=0, second=0, microsecond=0)
            for d9, i9 in cand9:
                t9 = (_ts_of(i9)
                      or _dtb.datetime.combine(_dtb.date.fromisoformat(d9),
                                               _dtb.time()))
                by_bk9.setdefault(int((t9 - t0h9).total_seconds()
                                      // (3600 * gn9)), []).append(i9)
            bk_range9 = {"t0": t0h9, "hours": gn9}
        else:
            for d9, i9 in cand9:
                bk9 = ((_dtb.date.fromisoformat(d9) - d0b).days // gn9) if gn9 else 0
                by_bk9.setdefault(bk9, []).append(i9)
        if len(by_bk9) > 60:
            unit9 = {"count": "장수", "hour": "시간"}.get(gmode9, "일수")
            raise HTTPException(400, f"묶음이 {len(by_bk9)}개 — {unit9}를 키우거나 "
                                     "기간을 좁혀 주세요(최대 60)")
        # 표본 예산: 총 4000장. 장수 모드는 '입력한 N장 전수'를 우선(예산 내에서),
        #   일/시간 모드는 버킷 수로 균등 배분(부족하면 카드에 '표본 평균' 표기).
        n_bk9 = len(by_bk9)
        BUDGET9 = 4000
        if gmode9 == "count" and gn9 > 0:
            per9 = gn9 if n_bk9 * gn9 <= BUDGET9 else max(4, BUDGET9 // n_bk9)
        else:
            per9 = (min(BUDGET9, 2000) if n_bk9 == 1
                    else max(4, min(400, BUDGET9 // n_bk9)))
        pk_ids9 = [i9 for v9 in by_bk9.values() for i9 in v9[:1]]
        canon_runs9, Lg, steps_t9, golden_t9 = _viz_grid()
        sref9 = steps_t9
        # ── 일 단위(전체 포함) 모드: 하루 표본을 고정해 '일별 합/개수'를 캐시한다.
        #    같은 기간 재조회·5일↔10일 변경·센서 추가가 즉시 끝난다(날짜 경계를 지키는
        #    묶음만 조립 가능 — 장수/시간 묶음은 기존 경로 그대로).
        cache_info9 = None
        day_mode9 = (gmode9 == "day") and not idsel9
        if day_mode9:
            day_ids9 = {}
            for d9, i9 in cand9:
                day_ids9.setdefault(d9, []).append(i9)
            day_pick9 = {d9: _viz_day_pick(v9) for d9, v9 in day_ids9.items()}
            bk_days9 = {}
            for d9 in day_ids9:
                bk9 = (((_dtb.date.fromisoformat(d9) - d0b).days // gn9) if gn9 else 0)
                bk_days9.setdefault(bk9, []).append(d9)
            cdir9 = _viz_cache_dir(Lg)
            acc_s9 = {bk9: np.zeros((Lg, len(sens))) for bk9 in by_bk9}
            acc_c9 = {bk9: np.zeros((Lg, len(sens))) for bk9 in by_bk9}
            used9 = {bk9: 0 for bk9 in by_bk9}
            hit9 = miss9_n = 0
            for bk9, ds9 in bk_days9.items():
                for d9 in ds9:
                    ids_d9 = day_pick9[d9]
                    sig_d9 = _viz_day_sig(ids_d9)
                    got9 = _viz_day_load(cdir9, d9, sens, sig_d9, Lg)
                    if got9 is None:
                        sd9, cd9, nd9 = _viz_day_calc(ids_d9, sens, canon_runs9, Lg)
                        _viz_day_save(cdir9, d9, sens, sig_d9, Lg, sd9, cd9, nd9)
                        miss9_n += 1
                    else:
                        sd9, cd9, nd9 = got9
                        hit9 += 1
                    acc_s9[bk9] += sd9
                    acc_c9[bk9] += cd9
                    used9[bk9] += nd9
            sampled9 = any(len(day_pick9[d9]) < len(day_ids9[d9]) for d9 in day_ids9)
            per9 = VIZ_PER_DAY
            cache_info9 = {"reused_days": hit9, "calc_days": miss9_n,
                           "enabled": cdir9 is not None, "per_day": VIZ_PER_DAY}
            print(f"[viz] 일별 캐시 {hit9}일 재사용 · {miss9_n}일 새로 계산"
                  f"{' (캐시 없음)' if cdir9 is None else ''}", flush=True)
        pick9 = {bk9: (v9 if len(v9) <= per9 else v9[::max(1, len(v9) // per9)][:per9])
                 for bk9, v9 in by_bk9.items()}
        if not day_mode9:
            sampled9 = any(len(pick9[bk9]) < len(by_bk9[bk9]) for bk9 in by_bk9)
        bk_of9 = {} if day_mode9 else {i9: bk9 for bk9, v9 in pick9.items() for i9 in v9}
        by_f9 = {}
        for wid9 in bk_of9:
            f9 = VZD["id2src"].get(wid9)
            if f9:
                by_f9.setdefault(f9, []).append(wid9)
        if not day_mode9:
            # 증분 누적(메모리 = 버킷×L×센서 고정) — 수백 장 글로벌 평균도 안전
            acc_s9 = {bk9: np.zeros((Lg, len(sens))) for bk9 in by_bk9}
            acc_c9 = {bk9: np.zeros((Lg, len(sens))) for bk9 in by_bk9}
            used9 = {bk9: 0 for bk9 in by_bk9}
        for f9, st9s in by_f9.items():
            for c0 in range(0, len(st9s), 100):       # 100장 청크 스트리밍
                recs9, _a, _b, _c = S._load_aligned(Path(f9), stems=st9s[c0:c0 + 100])
                for rec9 in recs9:
                    wid9 = str(rec9["wafer_id"])
                    bk9 = bk_of9.get(wid9)
                    if bk9 is None:
                        continue
                    cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
                    ii9 = [cols9.index(s9) for s9 in sens if s9 in cols9]
                    if len(ii9) != len(sens):
                        continue
                    arr9 = np.asarray(rec9["sensors"], dtype=np.float64)[:, ii9]
                    try:
                        stw9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
                        wv9 = (_warp_to_canon(arr9, stw9, canon_runs9, Lg)
                               if canon_runs9 is not None else
                               np.column_stack([_resample(arr9[:, q9], Lg)
                                                for q9 in range(arr9.shape[1])]))
                    except Exception:                  # noqa: BLE001
                        wv9 = np.column_stack([_resample(arr9[:, q9], Lg)
                                               for q9 in range(arr9.shape[1])])
                    fin9 = np.isfinite(wv9)
                    acc_s9[bk9] += np.where(fin9, wv9, 0.0)
                    acc_c9[bk9] += fin9
                    used9[bk9] += 1
                del recs9
        buckets9 = []
        for bk9 in sorted(by_bk9):
            if not used9[bk9]:
                continue
            with np.errstate(all="ignore"):
                m9 = acc_s9[bk9] / np.maximum(acc_c9[bk9], 1)
                m9 = np.where(acc_c9[bk9] > 0, m9, np.nan)
            ids_bk9 = by_bk9[bk9]
            tss9 = [t9 for t9 in (_ts_of(i9) for i9 in ids_bk9) if t9 is not None]
            if gmode9 == "count" and gn9 > 0:
                tA9 = tss9[0] if tss9 else _dtb.datetime.combine(d0b, _dtb.time())
                tB9 = tss9[-1] if tss9 else tA9
                s0b, s1b = tA9.date(), tB9.date()
                label9 = f"{tA9.strftime('%m-%d %H:%M')}~{tB9.strftime('%m-%d %H:%M')}"
            elif gmode9 == "hour" and gn9 > 0:
                tA9 = bk_range9["t0"] + _dtb.timedelta(hours=bk9 * gn9)
                tB9 = tA9 + _dtb.timedelta(hours=gn9)
                s0b = tA9.date()
                s1b = (tB9 - _dtb.timedelta(seconds=1)).date()
                # 끝 시각이 다음 날로 넘어가면 날짜까지 표기(24시간 묶음의 '00시~00시' 방지)
                label9 = (f"{tA9.strftime('%m-%d %H시')}~"
                          + (tB9.strftime('%H시') if tB9.date() == tA9.date()
                             else tB9.strftime('%m-%d %H시')))
            else:
                s0b = ((d0b + _dtb.timedelta(days=bk9 * gn9)) if gn9
                       else _dtb.date.fromisoformat(cand9[0][0]))
                s1b = ((s0b + _dtb.timedelta(days=gn9 - 1)) if gn9
                       else _dtb.date.fromisoformat(cand9[-1][0]))
                label9 = (f"{s0b.isoformat()}~{s1b.isoformat()[5:]}" if gn9
                          else f"전체 기간 {s0b.isoformat()}~{s1b.isoformat()}")
            pat9 = None
            try:
                pat9 = _pt_score_mean(sens, sref9,
                                      {s9: [None if not np.isfinite(v9)
                                            else float(v9) for v9 in m9[:, q9]]
                                       for q9, s9 in enumerate(sens)})
            except Exception as e9:                    # noqa: BLE001
                print(f"[viz] 평균 곡선 패턴 판정 실패(무해): {e9}", flush=True)
            buckets9.append({
                "label": label9,
                "pattern": pat9,
                "t0": (tss9[0].isoformat(sep=" ") if tss9 else ""),
                "t1": (tss9[-1].isoformat(sep=" ") if tss9 else ""),
                "d0": s0b.isoformat(), "d1": s1b.isoformat(),
                "n_total": len(by_bk9[bk9]), "n_used": used9[bk9],
                "ids": by_bk9[bk9],
                "curves": {s9: [None if not np.isfinite(v9) else round(float(v9), 4)
                                for v9 in m9[:, q9]]
                           for q9, s9 in enumerate(sens)}})
        return {"sensors": sens, "L": Lg,
                "steps": ([int(v9) for v9 in sref9] if sref9 is not None else None),
                "golden": {s9: [round(float(v9), 4) for v9 in golden_t9[s9]]
                           for s9 in sens},
                "bucket_days": bd9, "group_mode": gmode9, "group_n": gn9,
                "sampled": bool(sampled9), "per_bucket": int(per9),
                "sample_unit": ("하루" if day_mode9 else "묶음"),
                "cache": cache_info9,
                "buckets": buckets9}

    # ── 패턴 인식 탭: 형상 기하 규칙 2종 (P1 지수하강 볼록비 · P2 아치 반복) ──
    def _pt_rules():
        """규칙 파라미터: physics_rules.md 2부 pattern_rules > 내장 기본값."""
        base9 = {"exp_drop": {"sensor": "t_002_PV", "step": 5,
                              "alpha": 0.4, "min_drop_ratio": 0.2},
                 "arches": {"sensor": "dP_VslF-VslB", "steps": [2, 3, 4],
                            "alpha": 0.5},
                 "w_bulge": {"sensor": "t_003_PV", "step": 4,
                             "delta": 0.5, "min_rise_ratio": 0.2,
                             "peak_pct": 80.0},
                 # 계통 잔차(선형 혼합 모델) — 결과 ≈ Σw·원인 + b, w 는 골든 학습
                 "t_balance": {"result": "t_003_PV",
                               "causes": ["t_002_PV", "t_001"],
                               "k": 3.0, "min_run": 5},
                 "p_balance": {"result": "p_008_PV",
                               "causes": ["p_005_PV",
                                          "p_007_PV"],
                               "k": 3.0, "min_run": 5}}
        try:
            from io_recon_tcn import rca_causal as _RCq
            _t9, PH9 = _RCq.load_physics()
            for k9, v9 in (PH9.get("pattern_rules") or {}).items():
                if k9 in base9 and isinstance(v9, dict):
                    base9[k9].update(v9)
        except Exception:                              # noqa: BLE001
            pass
        return base9

    def _pt_step3():
        cfg9 = VZD.get("pt_cfg") or {}
        return int(cfg9.get("step3") or _pt_rules()["w_bulge"]["step"])

    def _pt_golden(step1, steps2):
        """골든 기준치: 기대 볼록비·아치 수·면적·극성·prominence 문턱. 없으면 사유."""
        rm9 = VZD.get("rmean") or {}
        sr9 = VZD.get("steps_ref")
        out9 = {"p1": None, "p2": None, "off": []}
        R9 = _pt_rules()
        sn1, sn2 = R9["exp_drop"]["sensor"], R9["arches"]["sensor"]
        sn3 = R9["w_bulge"]["sensor"]
        if sr9 is None:
            out9["off"] = ["골든에 스텝 정보가 없습니다(골든 재생성 필요)"]
            return out9
        if sn1 not in rm9:
            out9["off"].append(f"P1 비활성: 골든에 {sn1} 없음")
        else:
            g9 = _pt_seg(rm9[sn1], sr9, {int(step1)})
            r9 = _shape_convexity(g9) if g9 is not None else None
            if r9 is None:
                out9["off"].append(f"P1 비활성: 골든에 step{step1} 구간 없음/짧음")
            elif r9["ratio"] < 0.05:
                out9["off"].append(
                    f"P1 비활성: 골든 자체가 지수형 아님(볼록비 {r9['ratio']:.3f})")
            else:
                r9["gmean"] = float(np.nanmean(g9))
                out9["p1"] = r9
        if sn2 not in rm9:
            out9["off"].append(f"P2 비활성: 골든에 {sn2} 없음")
        else:
            a9 = _shape_steparea(rm9[sn2], sr9, [int(v) for v in steps2])
            if a9 is None or a9["total"] <= 1e-6:
                out9["off"].append("P2 비활성: 골든 지정 스텝 구간 없음/면적 0")
            else:
                g9 = _pt_seg(rm9[sn2], sr9, {int(v) for v in steps2})
                out9["p2"] = {"total_area": a9["total"], "areas": a9["areas"],
                              "n_arch": len(a9["areas"]),   # 구간(스텝) 수
                              "polarity": 0, "prom_ref": 0.0,
                              "gmean": float(np.nanmean(g9))}
        step3 = int(_pt_step3())
        out9["p3"] = None
        if sn3 not in rm9:
            out9["off"].append(f"W패턴 비활성: 골든에 {sn3} 없음")
        else:
            g9 = _pt_seg(rm9[sn3], sr9, {step3})
            w9 = _shape_wbulge(g9) if g9 is not None else None
            if w9 is None:
                out9["off"].append(
                    f"W패턴 비활성: 골든에 step{step3} 상승 구간 없음/짧음")
            else:
                w9["gmean"] = float(np.nanmean(g9))
                out9["p3"] = w9
        # 계통 잔차(p4 온도 / p5 압력): 골든 평균 곡선에서 스텝별 w,b,σ 학습
        cols_g9 = list(rm9)
        for key9, rk9, nm9 in (("p4", "t_balance", "온도 계통"),
                               ("p5", "p_balance", "압력 계통")):
            out9[key9] = None
            cfgb9 = R9.get(rk9) or {}
            rs9 = _sensor_resolve(cfgb9.get("result"), cols_g9)
            if rs9 is None:
                out9["off"].append(
                    f"{nm9} 비활성: 골든에 결과 센서({cfgb9.get('result')}) 없음"
                    " — physics_rules 에서 지정")
                continue
            cs9 = [c9 for c9 in (_sensor_resolve(v9, cols_g9)
                                 for v9 in (cfgb9.get("causes") or []))
                   if c9 is not None and c9 != rs9]
            if not cs9:
                out9["off"].append(f"{nm9} 비활성: 원인 센서를 하나도 찾지 못함")
                continue
            fitb9 = _bal_fit(rm9[rs9], [rm9[c9] for c9 in cs9], sr9)
            if not fitb9:
                out9["off"].append(f"{nm9} 비활성: 학습 가능한 스텝 없음")
                continue
            sdv9 = VZD.get("rstd") or {}
            if rs9 in sdv9:
                sra9 = np.asarray(sr9, dtype=int)
                for sp9, f9 in fitb9.items():
                    mstd9 = sra9 == int(sp9)
                    var9 = float(np.median(
                        np.asarray(sdv9[rs9])[mstd9])) ** 2
                    for w9x, c9 in zip(f9["w"], cs9):
                        if c9 in sdv9:
                            var9 += (w9x ** 2) * float(np.median(
                                np.asarray(sdv9[c9])[mstd9])) ** 2
                    f9["sig"] = max(f9["sig"], float(np.sqrt(var9)))
            k9 = float(cfgb9.get("k") or 3.0)
            _r9, _z9, _b9, run9, mz9 = _bal_eval(
                rm9[rs9], [rm9[c9] for c9 in cs9], sr9, fitb9, k9)
            out9[key9] = {"sensor": rs9, "causes": cs9, "fit": fitb9,
                          "k": k9, "min_run": int(cfgb9.get("min_run") or 5),
                          "self_z": (round(mz9, 3) if np.isfinite(mz9) else None)}
        return out9

    def _pt_cache_dir(step1, steps2):
        store9 = VZD.get("store") or ""
        if not store9:
            return None
        cache9 = (_state_load().get("sd") or {}).get("cache")
        gfile9 = VZD.get("gfile") or ""
        try:
            fp9 = Path(store9) / gfile9
            sig9 = f"{gfile9}|{fp9.stat().st_mtime_ns}|{fp9.stat().st_size}"
        except Exception:                              # noqa: BLE001
            sig9 = gfile9
        key9 = hashlib.md5(
            f"{sig9}|s{step1}|a{','.join(str(v) for v in steps2)}|s3-{_pt_step3()}|{CALC_VER}|o7"
            .encode("utf-8")).hexdigest()[:12]
        try:
            d9 = _cache_root(cache9, store9) / "pattern_cache" / key9
            d9.mkdir(parents=True, exist_ok=True)
            return d9
        except Exception as e9:                        # noqa: BLE001
            print(f"[pattern] 캐시 폴더를 쓸 수 없어 캐시 없이 진행: {e9}", flush=True)
            return None

    def _pt_by_f(ids9):
        by_f9 = {}
        for w9 in ids9:
            f9 = (VZD.get("id2src") or {}).get(w9)
            if f9:
                by_f9.setdefault(str(f9), []).append(w9)
        return by_f9

    def _pt_day_calc(ids9, step1, steps2, gref9):
        """하루 전수 점수 계산(13열) — 모듈 코어에 위임(병렬 워커와 단일 소스)."""
        R9 = _pt_rules()
        return _pt_rows_core(S, _pt_by_f(ids9), R9["exp_drop"]["sensor"],
                             R9["arches"]["sensor"], R9["w_bulge"]["sensor"],
                             step1, steps2, _pt_step3(), gref9)

    def _pt_calc_many(miss9, step1, steps2, gref9, progress9=None):
        """빠진 날들 일괄 계산 — 조건 되면 멀티프로세스(결과는 직렬과 동일).

        miss9: [(day, ids, sig)], 반환 {day: rows}. 게이트: 실제 서버 모듈(S)
        + 2일 이상 + 40장 이상 + 워커 설정>1. 실패 시 직렬 폴백.
        """
        R9 = _pt_rules()
        sn9 = (R9["exp_drop"]["sensor"], R9["arches"]["sensor"],
               R9["w_bulge"]["sensor"])
        cfg9 = (int(step1), [int(v9) for v9 in steps2], int(_pt_step3()))
        n_w9 = sum(len(i9) for _d9, i9, _s9 in miss9)
        try:
            workers9 = int(os.environ.get("AVG_SCAN_WORKERS") or 0) \
                or max(1, min(16, (os.cpu_count() or 2) // 2))
        except Exception:                              # noqa: BLE001
            workers9 = 1
        out_all9 = {}
        real9 = hasattr(S, "__file__")                 # 감사용 가짜 S 면 직렬
        if workers9 > 1 and len(miss9) >= 2 and n_w9 >= 40 and real9:
            try:
                gref_s9 = json.loads(json.dumps(gref9, default=float))
                root9 = str(Path(S.__file__).resolve().parents[2])
                model9 = str(getattr(S.ENGINE, "art", "") or "")
                cache9 = (_state_load().get("sd") or {}).get("cache")
                from concurrent.futures import (ProcessPoolExecutor,
                                                as_completed)
                import multiprocessing as _mp9
                print(f"[pattern] 점수 병렬 계산 — 워커 {workers9}개 · "
                      f"{len(miss9)}일 {n_w9}장", flush=True)
                with ProcessPoolExecutor(
                        max_workers=min(workers9, len(miss9)),
                        mp_context=_mp9.get_context("spawn"),
                        initializer=_ptw_init,
                        initargs=(root9, model9, cache9)) as ex9:
                    futs9 = {ex9.submit(_ptw_day,
                                        (d9, _pt_by_f(i9), sn9, cfg9, gref_s9)): d9
                             for d9, i9, _s9 in miss9}
                    done9 = 0
                    for fu9 in as_completed(futs9):
                        d9, rows9 = fu9.result()
                        out_all9[d9] = rows9
                        done9 += 1
                        if progress9:
                            progress9(done9, len(miss9))
                return out_all9
            except Exception as e9:                    # noqa: BLE001
                print(f"[pattern] 병렬 계산 실패 — 직렬로 폴백: {e9}", flush=True)
        for k9, (d9, i9, _s9) in enumerate(miss9):
            out_all9[d9] = _pt_day_calc(i9, step1, steps2, gref9)
            if progress9:
                progress9(k9 + 1, len(miss9))
        return out_all9

    PT_COLS = ["p1_ratio", "p1_drop", "p1_early", "p2_n", "p2_area"]

    def _pt_day_load(cdir9, day9, sig9):
        if cdir9 is None:
            return None
        fp9 = cdir9 / f"{day9}.npz"
        if not fp9.is_file():
            return None
        try:
            with np.load(fp9, allow_pickle=False) as z9:
                if str(z9["sig"]) != sig9:
                    return None
                ids9 = [str(v9) for v9 in z9["ids"]]
                vals9 = np.asarray(z9["vals"], dtype=np.float64)
                return {i9: vals9[k9].tolist() for k9, i9 in enumerate(ids9)}
        except Exception:                              # noqa: BLE001
            return None

    def _pt_day_save(cdir9, day9, sig9, scores9):
        if cdir9 is None or not scores9:
            return
        try:
            fp9 = cdir9 / f"{day9}.npz"
            ids9 = sorted(scores9)
            tmp9 = fp9.with_name(fp9.stem + f".tmp{os.getpid()}.npz")
            np.savez_compressed(tmp9, ids=np.asarray(ids9),
                                vals=np.asarray([scores9[i9] for i9 in ids9],
                                                dtype=np.float64),
                                sig=np.asarray(sig9))
            os.replace(tmp9, fp9)
        except Exception as e9:                        # noqa: BLE001
            print(f"[pattern] 일별 캐시 저장 실패(무해) {day9}: {e9}", flush=True)

    def _pt_score_mean(sens9, steps_arr9, curves9):
        """묶음 '평균 곡선' 판정 — 카드에 규칙 센서가 있을 때만.

        주의(사용자 고지됨): 소수 불량은 평균에서 희석돼 정상으로 보일 수 있다 —
        검출의 본선은 패턴 인식 탭의 개별 전수 판정이고, 이것은 집단 드리프트용.
        """
        if steps_arr9 is None:
            return None
        cfg9 = VZD.get("pt_cfg") or {}
        R9 = _pt_rules()
        step1 = int(cfg9.get("step1") or R9["exp_drop"]["step"])
        steps2 = [int(v9) for v9 in (cfg9.get("steps2") or R9["arches"]["steps"])]
        a1 = float(cfg9.get("a1") or R9["exp_drop"]["alpha"])
        a2 = float(cfg9.get("a2") or R9["arches"]["alpha"])
        sn1, sn2 = R9["exp_drop"]["sensor"], R9["arches"]["sensor"]
        sn3 = R9["w_bulge"]["sensor"]
        if sn1 not in sens9 and sn2 not in sens9 and sn3 not in sens9:
            return None
        gref9 = _pt_golden(step1, steps2)
        out9 = {}
        st9 = np.asarray(steps_arr9, dtype=int)
        if sn1 in sens9 and gref9.get("p1"):
            y9 = np.asarray([np.nan if v9 is None else v9
                             for v9 in curves9[sn1]], dtype=np.float64)
            seg9 = _pt_seg(y9, st9, {step1})
            r9 = _shape_convexity(seg9) if seg9 is not None else None
            if r9:
                g1 = gref9["p1"]
                thr9 = a1 * g1["ratio"]
                verd9 = ("하강 없음" if abs(r9["drop"]) <
                         R9["exp_drop"]["min_drop_ratio"] * abs(g1["drop"])
                         else ("직선 하강 의심" if r9["ratio"] < thr9 else ""))
                out9["p1"] = {"ratio": round(r9["ratio"], 4),
                              "thr": round(thr9, 4), "verdict": verd9}
        if sn2 in sens9 and gref9.get("p2"):
            y9 = np.asarray([np.nan if v9 is None else v9
                             for v9 in curves9[sn2]], dtype=np.float64)
            g2 = gref9["p2"]
            a9 = _shape_steparea(y9, st9, [int(v9) for v9 in steps2])
            if a9:
                thr9 = a2 * g2["total_area"]
                verd9 = ("아치 소실 의심" if a9["total"] < thr9
                         else ("아치 약함(경계)"
                               if a9["total"] < 1.3 * thr9 else ""))
                out9["p2"] = {"area": round(a9["total"], 4),
                              "n_arch": len(a9["areas"]),
                              "thr": round(thr9, 4), "verdict": verd9}
        if sn3 in sens9 and gref9.get("p3"):
            y9 = np.asarray([np.nan if v9 is None else v9
                             for v9 in curves9[sn3]], dtype=np.float64)
            step39 = int(_pt_step3())
            seg9 = _pt_seg(y9, st9, {step39})
            w9 = _shape_wbulge(seg9) if seg9 is not None else None
            if w9:
                g39 = gref9["p3"]
                a39 = float(cfg9.get("a3") or R9["w_bulge"]["delta"])
                n39 = float(cfg9.get("n3")
                            or R9["w_bulge"].get("peak_pct") or 80.0)
                thr9 = g39["bulge"] + a39
                pk9 = _wpeak(_wprev_seg(y9, st9, step39), seg9, w9["imin"])
                # v151: 100% 일치=단독 / 그 외 면적 AND 피크 N%
                if pk9 and pk9["ratio"] >= 1 - 1e-9:
                    verd39 = "W패턴 의심"
                elif w9["bulge"] > thr9 and (pk9 is None
                                             or pk9["ratio"] >= n39 / 100.0):
                    verd39 = "W패턴 의심"
                else:
                    verd39 = ""
                out9["p3"] = {"area": round(w9["bulge"], 3),
                              "thr": round(thr9, 3),
                              "verdict": verd39}
        return out9 or None

    @app.post("/api/pattern_scan")
    def api_pattern_scan(payload: dict = Body(...)):
        """패턴 인식 스캔: 기간 내 전수 웨이퍼의 형상 점수 + 골든 상대 판정."""
        if not VZD.get("id2src"):
            raise HTTPException(400, "먼저 [데이터 시각화] 탭에서 [웨이퍼 목록 불러오기]를 실행하세요")
        R9 = _pt_rules()
        try:
            step1 = int(payload.get("step1") or R9["exp_drop"]["step"])
            steps2 = [int(v9) for v9 in (payload.get("steps2")
                                         or R9["arches"]["steps"])]
            a1 = float(payload.get("alpha1") or R9["exp_drop"]["alpha"])
            a2 = float(payload.get("alpha2") or R9["arches"]["alpha"])
            step3 = int(payload.get("step3") or R9["w_bulge"]["step"])
            a3 = float(payload.get("alpha3") or R9["w_bulge"]["delta"])
            n3 = float(payload.get("peak_pct")
                       or R9["w_bulge"].get("peak_pct") or 80.0)
        except Exception:
            raise HTTPException(400, "스텝/비율 입력을 확인하세요(숫자)")
        d0 = str(payload.get("d0") or "").strip()
        d1 = str(payload.get("d1") or "").strip()
        gref9 = _pt_golden(step1, steps2)
        if not gref9["p1"] and not gref9["p2"]:
            raise HTTPException(400, "두 규칙 모두 비활성 — " + " / ".join(gref9["off"]))
        day_ids9 = {}
        for i9, dd9 in VZD["id2date"].items():
            if (d0 and dd9 < d0) or (d1 and dd9 > d1):
                continue
            day_ids9.setdefault(dd9, []).append(i9)
        if not day_ids9:
            raise HTTPException(400, "기간 내 웨이퍼가 없습니다")
        VZD["pt_cfg"] = {"step1": step1, "steps2": steps2, "a1": a1, "a2": a2,
                         "step3": step3, "a3": a3, "n3": n3}
        cdir9 = _pt_cache_dir(step1, steps2)
        # α 는 판정에만 쓰이므로: 스텝·기간·데이터가 같으면(=α만 변경)
        # 일별 캐시 로드 루프 자체를 건너뛰고 보관된 점수로 재판정만 한다.
        sigp9 = (f"{cdir9}|{d0}|{d1}|s{step1}|a{steps2}|s3-{step3}"
                 f"|n{len(VZD.get('id2date') or {})}"
                 f"|{min(day_ids9)}..{max(day_ids9)}({len(day_ids9)})")
        if PTS.get("sig") == sigp9 and PTS.get("scores"):
            scores9 = PTS["scores"]
            hit9, miss9 = PTS["n_days"], 0
            print("[pattern] α만 변경 — 일별 캐시 로드 생략, 판정만 재적용",
                  flush=True)
        else:
            hit9 = 0
            scores9 = {}
            missL9 = []
            for dd9 in sorted(day_ids9):
                ids9 = sorted(day_ids9[dd9])
                sig9 = _viz_day_sig(ids9)
                got9 = _pt_day_load(cdir9, dd9, sig9)
                if got9 is None:
                    missL9.append((dd9, ids9, sig9))
                else:
                    hit9 += 1
                    scores9.update(got9)
            calc9 = _pt_calc_many(missL9, step1, steps2, gref9)
            for dd9, ids9, sig9 in missL9:
                got9 = calc9.get(dd9) or {}
                _pt_day_save(cdir9, dd9, sig9, got9)
                scores9.update(got9)
            miss9 = len(missL9)
            PTS.update(sig=sigp9, scores=scores9, n_days=len(day_ids9))
            print(f"[pattern] 일별 캐시 {hit9}일 재사용 · {miss9}일 계산"
                  f"{' (캐시 없음)' if cdir9 is None else ''}", flush=True)
        # 판정(α 는 캐시 밖 — 조절해도 재계산 없음)
        g1, g2, g3 = gref9.get("p1"), gref9.get("p2"), gref9.get("p3")
        thrs9 = _pt_thrs(gref9, a1, a2, a3, n3)
        thr1, thr_d, thr2, thr3 = (thrs9["thr1"], thrs9["thr_d"],
                                   thrs9["thr2"], thrs9["thr3"])
        ts9 = VZD.get("id2ts") or {}
        rows9, viol9 = [], []
        for wid9 in sorted(scores9,
                           key=lambda w9: (ts9.get(w9, ""), w9)):
            v9 = scores9[wid9]
            v9 = list(v9) + [np.nan] * (14 - len(v9))  # 구캐시(5/7/9/13열) 호환
            fin9 = [None if not np.isfinite(x9) else round(float(x9), 4)
                    for x9 in v9]
            verd9 = _pt_verdicts(v9, thrs9)
            row9 = {"id": wid9, "date": VZD["id2date"].get(wid9, ""),
                    "ts": ts9.get(wid9, ""), "p1_ratio": fin9[0],
                    "p1_drop": fin9[1], "p1_early": fin9[2],
                    "p2_n": (None if fin9[3] is None else int(fin9[3])),
                    "p2_area": fin9[4],
                    "p1_off": (fin9[5] if len(fin9) > 5 else None),
                    "p2_off": (fin9[6] if len(fin9) > 6 else None),
                    "p3_area": (fin9[7] if len(fin9) > 7 else None),
                    "p3_off": (fin9[8] if len(fin9) > 8 else None),
                    "p3_peak": (fin9[13] if len(fin9) > 13 else None),
                    "p4_z": (fin9[9] if len(fin9) > 9 else None),
                    "p4_run": (fin9[10] if len(fin9) > 10 else None),
                    "p5_z": (fin9[11] if len(fin9) > 11 else None),
                    "p5_run": (fin9[12] if len(fin9) > 12 else None),
                    "verdicts": verd9}
            rows9.append(row9)
            if verd9:
                viol9.append(row9)
        return {"n": len(rows9), "rows": rows9, "violations": viol9,
                "golden": {"p1": ({"ratio": round(g1["ratio"], 4),
                                   "drop": round(g1["drop"], 3),
                                   "thr": round(thr1, 4)} if g1 else None),
                           "p2": ({"n_arch": g2["n_arch"],
                                   "area": round(g2["total_area"], 4),
                                   "polarity": g2["polarity"],
                                   "thr": round(thr2, 4)} if g2 else None),
                           "p3": ({"area": round(g3["bulge"], 3),
                                   "thr": round(thr3, 3)} if g3 else None),
                           "p4": ({"z": gref9["p4"]["self_z"],
                                   "k": gref9["p4"]["k"],
                                   "min_run": gref9["p4"]["min_run"]}
                                  if gref9.get("p4") else None),
                           "p5": ({"z": gref9["p5"]["self_z"],
                                   "k": gref9["p5"]["k"],
                                   "min_run": gref9["p5"]["min_run"]}
                                  if gref9.get("p5") else None)},
                "off": gref9["off"], "step1": step1, "steps2": steps2,
                "step3": step3, "alpha1": a1, "alpha2": a2, "alpha3": a3,
                "peak_pct": n3,
                "sensors": {"p1": R9["exp_drop"]["sensor"],
                            "p2": R9["arches"]["sensor"],
                            "p3": R9["w_bulge"]["sensor"],
                            "p4": ((gref9["p4"]["sensor"] + " − 모델")
                                   if gref9.get("p4") else None),
                            "p5": ((gref9["p5"]["sensor"] + " − 모델")
                                   if gref9.get("p5") else None)},
                "cache": {"enabled": cdir9 is not None,
                          "reused_days": hit9, "calc_days": miss9}}

    def _pt_thrs(gref9, a1, a2, a3, n3=None):
        """규칙별 임계 묶음(스캔·분석자료 공용) — α·N 은 캐시 밖."""
        R9 = _pt_rules()
        if n3 is None:
            n3 = float(R9["w_bulge"].get("peak_pct") or 80.0)
        g1, g2, g3 = gref9.get("p1"), gref9.get("p2"), gref9.get("p3")
        return {"g1": g1, "g2": g2, "g3": g3,
                "g4": gref9.get("p4"), "g5": gref9.get("p5"),
                "thr1": (a1 * g1["ratio"]) if g1 else None,
                "thr_d": ((R9["exp_drop"]["min_drop_ratio"] * abs(g1["drop"]))
                          if g1 else None),
                "thr2": (a2 * g2["total_area"]) if g2 else None,
                # W패턴: 임계 면적 = 골든 면적 + 여유(원단위) — '초과'가 불량(방향 반대)
                "thr3": (g3["bulge"] + a3) if g3 else None,
                # v151: W 피크 비율 기준(%) — 면적 AND 피크, 100% 일치는 단독 불량
                "n3": float(n3)}

    def _pt_verdicts(v9, thrs9):
        """점수 13열 → 판정 문구 목록(스캔·분석자료 공용)."""
        v9 = list(v9) + [np.nan] * (14 - len(v9))
        verd9 = []
        if thrs9["g1"] and np.isfinite(v9[0]):
            if abs(v9[1]) < thrs9["thr_d"]:
                verd9.append("하강 없음")
            elif v9[0] < thrs9["thr1"]:
                verd9.append("직선 하강 의심")
        if thrs9["g2"] and np.isfinite(v9[4]):
            if v9[4] < thrs9["thr2"]:                  # 면적 기준만(사용자 결정)
                verd9.append("아치 소실 의심")
            elif v9[4] < 1.3 * thrs9["thr2"]:
                verd9.append("아치 약함(경계)")
        if thrs9["g3"] and len(v9) > 7 and np.isfinite(v9[7]):
            # v151: ①피크 100% 일치=면적 무관 불량 ②그 외 면적 AND 피크 N%
            r39 = (v9[13] if len(v9) > 13 and np.isfinite(v9[13]) else None)
            if r39 is not None and r39 >= 1 - 1e-9:
                verd9.append("W패턴 의심")
            elif v9[7] > thrs9["thr3"] and (
                    r39 is None or r39 >= thrs9.get("n3", 80.0) / 100.0):
                verd9.append("W패턴 의심")
        for off9, gk9, nm9 in ((9, "g4", "온도 계통 의심"),
                               (11, "g5", "압력 계통 의심")):
            gb9 = thrs9.get(gk9)
            if gb9 and np.isfinite(v9[off9 + 1]) \
                    and v9[off9 + 1] >= gb9["min_run"]:
                verd9.append(nm9)
        return verd9

    def _pt_mat(seg9, rule9, sn9, gref9, a1c9, a2c9, a3c9=None,
                st_seg9=None, steps29=None, pseg9=None, n3c9=None):
        """구간 곡선 → 카드 재료(현/받침점·원단위 숫자·점수·판정)."""
        seg9 = np.asarray(seg9, dtype=np.float64)
        n9 = len(seg9)
        unit9 = "\u2103" if str(sn9).startswith("T") else "bar"
        out9 = {"sensor": sn9, "rule": rule9,
                "y": [None if not np.isfinite(v9) else round(float(v9), 4)
                      for v9 in seg9]}
        R9 = _pt_rules()
        if rule9 == "p1":
            k9 = min(5, max(1, n9 // 10))
            y0 = float(np.median(seg9[:k9]))
            y1 = float(np.median(seg9[-k9:]))
            out9["chord"] = [round(float(v9), 4)
                             for v9 in np.linspace(y0, y1, n9)]
            r9 = _shape_convexity(seg9)
            out9["score"] = (round(r9["ratio"], 4) if r9 else None)
            g19 = gref9.get("p1")
            thr9 = (a1c9 * g19["ratio"]) if g19 else None
            verd9 = ""
            if r9 and g19:
                if abs(r9["drop"]) < (R9["exp_drop"]["min_drop_ratio"]
                                      * abs(g19["drop"])):
                    verd9 = "하강 없음"
                elif r9["ratio"] < thr9:
                    verd9 = "직선 하강 의심"
            out9["verdict"] = verd9
            out9["nums"] = {"y0": round(y0, 2), "y1": round(y1, 2),
                            "drop": round(y0 - y1, 2),
                            "gap": (round(r9["area"], 3) if r9 else None),
                            "unit": unit9,
                            "g_ratio": (round(g19["ratio"], 4) if g19 else None),
                            "thr": (round(thr9, 4) if thr9 is not None else None)}
        elif rule9 == "p3":
            g39 = gref9.get("p3")
            w9 = _shape_wbulge(seg9)
            thr9 = ((g39["bulge"] + (a3c9 if a3c9 is not None
                                     else _pt_rules()["w_bulge"]["delta"]))
                    if g39 else None)
            n39 = float(n3c9 if n3c9 is not None
                        else _pt_rules()["w_bulge"].get("peak_pct") or 80.0)
            ch9 = [None] * n9
            verd9 = ""
            pk9 = None
            if w9:
                im9 = w9["imin"]
                lin9 = np.linspace(float(np.median(seg9[max(0, im9 - 2):im9 + 3])),
                                   float(np.median(seg9[-max(1, min(5, (n9 - im9) // 10)):])),
                                   n9 - im9)
                for q9, v9 in enumerate(lin9):
                    ch9[im9 + q9] = round(float(v9), 4)
                out9["feet"] = [im9]                    # 최저점 마커
                pk9 = _wpeak(pseg9, seg9, im9)
                # v151: ①피크 100% 일치=단독 불량 ②면적 AND 피크 N%
                if g39 and pk9 and pk9["ratio"] >= 1 - 1e-9:
                    verd9 = "W패턴 의심"
                elif g39 and w9["bulge"] > thr9 and (
                        pk9 is None or pk9["ratio"] >= n39 / 100.0):
                    verd9 = "W패턴 의심"
                if pk9:                                 # 관심구간 최고 마커(빨강)
                    out9["peaks"] = [int(im9 + int(np.nanargmax(
                        np.asarray(seg9[im9:], dtype=np.float64))))]
            out9["chord"] = ch9
            out9["score"] = (round(w9["bulge"], 3) if w9 else None)
            out9["verdict"] = verd9
            out9["nums"] = {"rise": (round(w9["rise"], 2) if w9 else None),
                            "bulge": (round(w9["bulge"], 3) if w9 else None),
                            "thr_area": (round(thr9, 3) if thr9 is not None else None),
                            "unit": unit9,
                            "g_area": (round(g39["bulge"], 3) if g39 else None),
                            "thr": (round(thr9, 3) if thr9 is not None else None),
                            "peak_roi": (round(pk9["peak_roi"], 2) if pk9 else None),
                            "peak_all": (round(pk9["peak_all"], 2) if pk9 else None),
                            "peak_pct": (round(pk9["ratio"] * 100, 1) if pk9 else None),
                            "n_pct": round(n39, 1)}
        else:
            # v145: 스텝별 현(첫점↔끝점) 대비 |면적| — 사용자 지정 방식
            g29 = gref9.get("p2")
            a9 = (_shape_steparea(seg9, st_seg9, steps29 or [])
                  if st_seg9 is not None else None)
            ch9 = [None] * n9
            feet9 = []
            if a9:
                for s09, e09, y09, y19 in a9["segs"]:
                    lin9 = np.linspace(y09, y19, e09 - s09 + 1)
                    for q9, v9 in enumerate(lin9):
                        ch9[s09 + q9] = round(float(v9), 4)
                    feet9 += [s09, e09]
            out9["chord"] = ch9
            out9["feet"] = sorted(set(feet9))
            out9["peaks"] = []
            out9["score"] = (round(a9["total"], 4) if a9 else None)
            out9["n_arch"] = (len(a9["areas"]) if a9 else None)
            thr9 = (a2c9 * g29["total_area"]) if g29 else None
            verd9 = ""
            if a9 and g29:
                if a9["total"] < thr9:                 # 면적 기준만
                    verd9 = "아치 소실 의심"
                elif a9["total"] < 1.3 * thr9:
                    verd9 = "아치 약함(경계)"
            out9["verdict"] = verd9
            out9["nums"] = {"areas": [round(v9, 3) for v9 in
                                      ((a9 or {}).get("areas") or [])],
                            "sum": (round(a9["total"], 3) if a9 else None),
                            "unit": unit9,
                            "g_area": (round(g29["total_area"], 4) if g29 else None),
                            "thr": (round(thr9, 4) if thr9 is not None else None)}
        return out9

    def _pt_ctx(y9, st9, want9):
        """관심 스텝 구간 + 앞뒤 한 스텝(컨텍스트) — (전체곡선, roi a, roi b, 스텝 라벨)."""
        st9 = np.asarray(st9, dtype=int)
        m9 = np.isin(st9, sorted(want9))
        if not m9.any():
            return None
        i9 = np.where(m9)[0]
        a9, b9 = int(i9[0]), int(i9[-1])
        runs9 = _step_runs(st9)
        A9, B9 = a9, b9
        for k9, (sp9, s09, e09) in enumerate(runs9):
            if s09 <= a9 < e09 and k9 > 0:
                A9 = runs9[k9 - 1][1]
            if s09 <= b9 < e09 and k9 + 1 < len(runs9):
                B9 = runs9[k9 + 1][2] - 1
        yex9 = np.asarray(y9, dtype=np.float64)[A9:B9 + 1]
        marks9 = [{"i": max(s09, A9) - A9, "label": f"step {sp9}"}
                  for sp9, s09, e09 in runs9 if not (e09 <= A9 or s09 > B9)]
        return yex9, a9 - A9, b9 - A9, marks9

    def _pt_mat_ctx(y9, st9, want9, rule9, sn9, gref9, a1c9, a2c9, a3c9=None,
                    n3c9=None):
        """컨텍스트 포함 카드 재료 — 관심 스텝 밖은 회색 표시용 roi/stepmarks 동봉."""
        c9 = _pt_ctx(y9, st9, want9)
        if c9 is None:
            return None
        yex9, a9, b9, marks9 = c9
        seg9 = yex9[a9:b9 + 1]
        if not np.isfinite(seg9).any() or len(seg9) < 12:
            return None
        i9x = np.where(np.isin(np.asarray(st9, dtype=int), sorted(want9)))[0]
        st_seg9 = np.asarray(st9, dtype=int)[i9x[0]:i9x[-1] + 1]
        pseg9 = (None if rule9 != "p3"
                 else _wprev_seg(np.asarray(y9, dtype=np.float64), st9,
                                 min(want9)))
        m9 = _pt_mat(seg9, rule9, sn9, gref9, a1c9, a2c9, a3c9,
                     st_seg9=st_seg9, steps29=sorted(int(v9) for v9 in want9),
                     pseg9=pseg9, n3c9=n3c9)
        nex9 = len(yex9)
        m9["y"] = [None if not np.isfinite(v9) else round(float(v9), 4)
                   for v9 in yex9]
        ch9 = m9.get("chord") or []
        m9["chord"] = [None] * a9 + list(ch9) + [None] * (nex9 - b9 - 1)
        for k9 in ("feet", "peaks"):
            if m9.get(k9):
                m9[k9] = [int(v9) + a9 for v9 in m9[k9]]
        m9["roi"] = [a9, b9]
        m9["stepmarks"] = marks9
        return m9

    def _pt_bal_mat(getcol9, st9, key9, gref9):
        """계통 잔차 카드 재료 — 곡선=R(t) 원단위·밴드=±kσ(스텝별)·이탈 표시.

        getcol9(센서명)->배열 or None (웨이퍼 원본·평균 곡선 공용).
        """
        gb9 = gref9.get(key9)
        if not gb9:
            return None
        res9 = getcol9(gb9["sensor"])
        cs9 = [getcol9(c9) for c9 in gb9["causes"]]
        if res9 is None or any(c9 is None for c9 in cs9):
            return None
        st9 = np.asarray(st9, dtype=int)
        R9, z9, bd9, run9, mz9 = _bal_eval(res9, cs9, st9,
                                           gb9["fit"], gb9["k"])
        if not np.isfinite(R9).any():
            return None
        i9 = np.where(np.isfinite(bd9))[0]
        roi9 = [int(i9[0]), int(i9[-1])] if len(i9) else None
        marks9 = [{"i": int(s09), "label": f"step {sp9}"}
                  for sp9, s09, e09 in _step_runs(st9)]
        vio9 = np.where(np.isfinite(z9) & (z9 > gb9["k"]))[0]
        pk9 = [int(v9) for v9 in vio9[::max(1, len(vio9) // 60)]]
        verd9 = ""
        if run9 >= gb9["min_run"]:
            verd9 = "온도 계통 의심" if key9 == "p4" else "압력 계통 의심"
        unit9 = "\u2103" if str(gb9["sensor"]).startswith("T") else "bar"
        sig_m9 = float(np.median([f9["sig"] for f9 in gb9["fit"].values()]))

        def _rnd9(a9):
            return [None if not np.isfinite(v9) else round(float(v9), 4)
                    for v9 in a9]
        return {"sensor": gb9["sensor"], "rule": key9, "y": _rnd9(R9),
                "chord": None, "bandu": _rnd9(bd9), "bandl": _rnd9(-bd9),
                "peaks": pk9, "roi": roi9, "stepmarks": marks9,
                "score": (round(mz9, 2) if np.isfinite(mz9) else None),
                "verdict": verd9, "causes": gb9["causes"],
                "nums": {"maxz": (round(mz9, 2) if np.isfinite(mz9) else None),
                         "run": int(run9), "k": gb9["k"],
                         "min_run": gb9["min_run"],
                         "sig": round(sig_m9, 4), "unit": unit9,
                         "model": " + ".join(
                             f"w{q9 + 1}\u00b7{c9}"
                             for q9, c9 in enumerate(gb9["causes"])) + " + b"}}

    def _pt_bal_sens(gref9, rules9=("p4", "p5")):
        """계통 규칙이 요구하는 센서 목록(결과+원인, 중복 제거)."""
        out9 = []
        for k9 in rules9:
            gb9 = gref9.get(k9)
            if gb9:
                for c9 in [gb9["sensor"]] + gb9["causes"]:
                    if c9 not in out9:
                        out9.append(c9)
        return out9

    @app.post("/api/pattern_buckets")
    def api_pattern_buckets(payload: dict = Body(...)):
        """선택 기간을 N일씩 평균낸 파형 카드 — 규칙 센서 구간 + 선분/면적/판정.

        시각화 탭의 일별 부분합 캐시를 그대로 재사용(같은 골든·격자 키).
        """
        if not VZD.get("id2src"):
            raise HTTPException(400, "먼저 [데이터 시각화] 탭에서 [웨이퍼 목록 불러오기]를 실행하세요")
        R9 = _pt_rules()
        step1 = int(payload.get("step1") or R9["exp_drop"]["step"])
        steps2 = [int(v9) for v9 in (payload.get("steps2") or R9["arches"]["steps"])]
        a1 = float(payload.get("alpha1") or R9["exp_drop"]["alpha"])
        a2 = float(payload.get("alpha2") or R9["arches"]["alpha"])
        nd9 = max(1, int(payload.get("ndays") or 5))
        d0 = str(payload.get("d0") or "").strip()
        d1 = str(payload.get("d1") or "").strip()
        step3 = int(payload.get("step3") or R9["w_bulge"]["step"])
        a3 = float(payload.get("alpha3") or R9["w_bulge"]["delta"])
        n3 = float(payload.get("peak_pct")
                   or R9["w_bulge"].get("peak_pct") or 80.0)
        gref9 = _pt_golden(step1, steps2)
        sn1, sn2 = R9["exp_drop"]["sensor"], R9["arches"]["sensor"]
        sn3 = R9["w_bulge"]["sensor"]
        sens9 = [sn9 for sn9, on9 in ((sn1, gref9.get("p1")), (sn2, gref9.get("p2")),
                                      (sn3, gref9.get("p3")))
                 if on9]
        for c9 in _pt_bal_sens(gref9):
            if c9 not in sens9:
                sens9.append(c9)
        if not sens9:
            raise HTTPException(400, "두 규칙 모두 비활성 — " + " / ".join(gref9["off"]))
        canon9, Lg, steps_t9, _g9 = _viz_grid()
        if steps_t9 is None:
            raise HTTPException(400, "골든에 스텝 정보가 없습니다")
        import datetime as _dtb
        day_ids9 = {}
        for i9, dd9 in VZD["id2date"].items():
            if (d0 and dd9 < d0) or (d1 and dd9 > d1):
                continue
            day_ids9.setdefault(dd9, []).append(i9)
        if not day_ids9:
            raise HTTPException(400, "기간 내 웨이퍼가 없습니다")
        days9 = sorted(day_ids9)
        base9 = _dtb.date.fromisoformat(days9[0])
        bk_days9 = {}
        for dd9 in days9:
            bk9 = (_dtb.date.fromisoformat(dd9) - base9).days // nd9
            bk_days9.setdefault(bk9, []).append(dd9)
        if len(bk_days9) > 60:
            raise HTTPException(400, f"묶음이 {len(bk_days9)}개 — 일수를 키우거나 "
                                     "기간을 좁혀 주세요(최대 60)")
        cdir9 = _viz_cache_dir(Lg)
        cards9 = []
        for bk9 in sorted(bk_days9):
            acc9 = np.zeros((Lg, len(sens9)))
            cnt9 = np.zeros((Lg, len(sens9)))
            used9 = 0
            for dd9 in bk_days9[bk9]:
                pick9 = _viz_day_pick(sorted(day_ids9[dd9]))
                sig9 = _viz_day_sig(pick9)
                got9 = _viz_day_load(cdir9, dd9, sens9, sig9, Lg)
                if got9 is None:
                    sd9, cd9, nd_9 = _viz_day_calc(pick9, sens9, canon9, Lg)
                    _viz_day_save(cdir9, dd9, sens9, sig9, Lg, sd9, cd9, nd_9)
                    got9 = (sd9, cd9, nd_9)
                acc9 += got9[0]
                cnt9 += got9[1]
                used9 += got9[2]
            with np.errstate(all="ignore"):
                m9 = np.where(cnt9 > 0, acc9 / np.maximum(cnt9, 1), np.nan)
            dA9, dB9 = bk_days9[bk9][0], bk_days9[bk9][-1]
            card9 = {"d0": dA9, "d1": dB9, "n_used": used9,
                     "label": f"{dA9}~{dB9[5:]} ({used9}장 평균)", "mats": []}
            for q9, sn9 in enumerate(sens9):
                if sn9 == sn1:
                    rule9 = "p1"
                elif sn9 == sn3:
                    rule9 = "p3"
                elif sn9 == sn2:
                    rule9 = "p2"
                else:
                    continue                           # 계통 전용 센서(아래서 사용)
                want9 = {"p1": {step1}, "p3": {step3}}.get(
                    rule9, {int(v9) for v9 in steps2})
                mat9 = _pt_mat_ctx(m9[:, q9], steps_t9, want9, rule9, sn9,
                                   gref9, a1, a2, a3, n3c9=n3)
                if mat9 is None:
                    continue
                mat9["id"] = card9["label"]
                card9["mats"].append(mat9)
            sl9 = list(sens9)

            def _gc9(sn_q9, _m9=m9, _s9=sl9):
                return _m9[:, _s9.index(sn_q9)] if sn_q9 in _s9 else None
            for key9 in ("p4", "p5"):
                mat9 = _pt_bal_mat(_gc9, steps_t9, key9, gref9)
                if mat9 is not None:
                    mat9["id"] = card9["label"]
                    card9["mats"].append(mat9)
            cards9.append(card9)
        return {"cards": cards9, "ndays": nd9, "off": gref9["off"],
                "n_buckets": len(cards9)}

    # ── 분석자료 탭: 필터×N일 평균×달력 (30만장 대응 — 점수 전역 캐시·집계만 전송) ──
    ANLJOB = {"state": "idle", "progress": "", "error": None}
    ANL = {"scores": {}, "d0": "", "d1": ""}
    PTS = {"sig": "", "scores": None, "n_days": 0}     # 패턴 스캔 보관(α 고속 재판정)

    def _anl_match(meta9, eqp9, dsv9, rc9):
        """select_stems 와 동일 의미: 설비 부분일치·DSV 일치·레시피 부분일치. 빈=전체."""
        if eqp9 and not any(t9 in meta9["eqp"] for t9 in eqp9):
            return False
        if dsv9 and meta9["dsv"] not in dsv9:
            return False
        if rc9 and not any(t9 in meta9["recipe"] for t9 in rc9):
            return False
        return True

    def _anl_filters(payload):
        def _lst9(k9):
            return [str(v9).strip().lower() for v9 in (payload.get(k9) or [])
                    if str(v9).strip()]
        dsv9 = [("dsv" + t9 if t9.isdigit() else t9) for t9 in _lst9("dsv")]
        return _lst9("eqp"), dsv9, _lst9("recipe")

    @app.post("/api/anl_scan")
    def api_anl_scan(payload: dict = Body(...)):
        """분석자료 전수 스캔 — 백그라운드 잡(일별 점수 캐시 재사용·증분)."""
        if not VZD.get("id2src"):
            raise HTTPException(400, "먼저 [데이터 시각화] 탭에서 [웨이퍼 목록 불러오기]를 실행하세요")
        if ANLJOB["state"] == "running":
            raise HTTPException(409, "분석이 이미 진행 중입니다")
        R9 = _pt_rules()
        try:
            step1 = int(payload.get("step1") or R9["exp_drop"]["step"])
            steps2 = [int(v9) for v9 in (payload.get("steps2")
                                         or R9["arches"]["steps"])]
            step3 = int(payload.get("step3") or R9["w_bulge"]["step"])
        except Exception:
            raise HTTPException(400, "스텝 입력을 확인하세요(숫자)")
        d0 = str(payload.get("d0") or "").strip()
        d1 = str(payload.get("d1") or "").strip()
        VZD["pt_cfg"] = dict(VZD.get("pt_cfg") or {},
                             step1=step1, steps2=steps2, step3=step3)
        gref9 = _pt_golden(step1, steps2)
        if not (gref9.get("p1") or gref9.get("p2") or gref9.get("p3")):
            raise HTTPException(400, "세 규칙 모두 비활성 — " + " / ".join(gref9["off"]))
        day_ids9 = {}
        for i9, dd9 in VZD["id2date"].items():
            if (d0 and dd9 < d0) or (d1 and dd9 > d1):
                continue
            day_ids9.setdefault(dd9, []).append(i9)
        if not day_ids9:
            raise HTTPException(400, "기간 내 웨이퍼가 없습니다")
        cdir9 = _pt_cache_dir(step1, steps2)
        siga9 = (f"{cdir9}|{d0}|{d1}|s{step1}|a{steps2}|s3-{step3}"
                 f"|n{len(VZD.get('id2date') or {})}"
                 f"|{min(day_ids9)}..{max(day_ids9)}({len(day_ids9)})")
        if ANL.get("sig") == siga9 and ANL["scores"] \
                and ANLJOB["state"] == "done":
            # α 는 집계(anl_agg)에서만 쓰임 — 점수 그대로, 즉시 완료 처리
            print("[anl] 같은 조건 재실행 — 스캔 생략(보관 점수 재사용)", flush=True)
            ANLJOB.update(state="done",
                          progress=f"완료 — {len(ANL['scores'])}장 · "
                                   f"캐시 재사용 {len(day_ids9)}일/계산 0일",
                          error=None)
            return {"ok": True, "n_days": len(day_ids9),
                    "n_wafers": sum(len(v9) for v9 in day_ids9.values()),
                    "off": gref9["off"]}
        ANL["sig"] = siga9
        ANLJOB.update(state="running", progress="시작…", error=None)
        ANL.update(scores={}, d0=(d0 or min(day_ids9)), d1=(d1 or max(day_ids9)),
                   off=gref9["off"], gref=gref9,
                   cfg={"step1": step1, "steps2": steps2, "step3": step3})

        def _anlrun():
            try:
                days_s9 = sorted(day_ids9)
                hit9 = 0
                missL9 = []
                for k9, dd9 in enumerate(days_s9):
                    ids9 = sorted(day_ids9[dd9])
                    sig9 = _viz_day_sig(ids9)
                    got9 = _pt_day_load(cdir9, dd9, sig9)
                    if got9 is None:
                        missL9.append((dd9, ids9, sig9))
                    else:
                        hit9 += 1
                        ANL["scores"].update(got9)
                    ANLJOB["progress"] = (f"캐시 확인 {k9 + 1}/{len(days_s9)}일 · "
                                          f"재사용 {hit9}일")

                def _pg9(done9, tot9):
                    ANLJOB["progress"] = (f"{hit9 + done9}/{len(days_s9)}일 · "
                                          f"캐시 재사용 {hit9}일 · 계산 {done9}일 · "
                                          f"{len(ANL['scores'])}장")
                calc9 = _pt_calc_many(missL9, step1, steps2, gref9,
                                      progress9=_pg9)
                for dd9, ids9, sig9 in missL9:
                    got9 = calc9.get(dd9) or {}
                    _pt_day_save(cdir9, dd9, sig9, got9)
                    ANL["scores"].update(got9)
                ANLJOB.update(state="done",
                              progress=f"완료 — {len(ANL['scores'])}장 · "
                                       f"캐시 재사용 {hit9}일/계산 {len(missL9)}일")
            except Exception as e9:                    # noqa: BLE001
                ANLJOB.update(state="error", error=str(e9))
        threading.Thread(target=_anlrun, daemon=True).start()
        return {"ok": True, "n_days": len(day_ids9),
                "n_wafers": sum(len(v9) for v9 in day_ids9.values()),
                "off": gref9["off"]}

    @app.post("/api/anl_status")
    def api_anl_status():
        return {"state": ANLJOB["state"], "progress": ANLJOB["progress"],
                "error": ANLJOB["error"]}

    @app.post("/api/anl_agg")
    def api_anl_agg(payload: dict = Body(...)):
        """필터+α 적용 집계 — 달력용. 30만 행은 서버에 두고 카운트만 반환."""
        if not ANL["scores"]:
            raise HTTPException(400, "먼저 [분석 시작]을 실행하세요")
        R9 = _pt_rules()
        a1 = float(payload.get("alpha1") or R9["exp_drop"]["alpha"])
        a2 = float(payload.get("alpha2") or R9["arches"]["alpha"])
        a3 = float(payload.get("alpha3") or R9["w_bulge"]["delta"])
        n3 = float(payload.get("peak_pct")
                   or R9["w_bulge"].get("peak_pct") or 80.0)
        eqp9, dsv9, rc9 = _anl_filters(payload)
        thrs9 = _pt_thrs(ANL["gref"], a1, a2, a3, n3)
        meta9 = VZD.get("id2meta") or {}
        dts9 = VZD.get("id2date") or {}
        days9 = {}
        avail9 = {"eqp": {}, "dsv": {}, "recipe": {}}
        for wid9, v9 in ANL["scores"].items():
            m9 = meta9.get(wid9) or {"eqp": "", "dsv": "", "recipe": ""}
            # 패싯 카운트: 각 그룹 숫자는 '다른 두 그룹의 현재 선택'을 적용해 센다
            # (자기 그룹 선택은 제외 — 체크박스 숫자=지금 조건으로 달력에 나올 장수).
            # 키는 0장이어도 유지(체크해 둔 값이 목록에서 사라지지 않게).
            me9 = (not eqp9) or any(t9 in m9["eqp"] for t9 in eqp9)
            md9 = (not dsv9) or m9["dsv"] in dsv9
            mr9 = (not rc9) or any(t9 in m9["recipe"] for t9 in rc9)
            for k9, ok9 in (("eqp", md9 and mr9), ("dsv", me9 and mr9),
                            ("recipe", me9 and md9)):
                vv9 = m9[k9] or "(없음)"
                avail9[k9][vv9] = avail9[k9].get(vv9, 0) + (1 if ok9 else 0)
            if not (me9 and md9 and mr9):
                continue
            verd9 = _pt_verdicts(v9, thrs9)
            d9 = dts9.get(wid9, "")
            # [b1,b2,b3,bad,ok,b4,b5] — 계통은 뒤에 덧붙임(기존 인덱스 유지)
            c9 = days9.setdefault(d9, [0, 0, 0, 0, 0, 0, 0])
            t9 = ",".join(verd9)
            if verd9:
                c9[3] += 1
                if "직선" in t9 or "하강" in t9:
                    c9[0] += 1
                if "아치" in t9:
                    c9[1] += 1
                if "W패턴" in t9:
                    c9[2] += 1
                if "온도 계통" in t9:
                    c9[5] += 1
                if "압력 계통" in t9:
                    c9[6] += 1
            else:
                c9[4] += 1
        return {"days": days9, "avail": avail9,
                "n": sum(c9[3] + c9[4] for c9 in days9.values()),
                "d0": ANL["d0"], "d1": ANL["d1"], "off": ANL.get("off") or []}

    @app.post("/api/anl_ids")
    def api_anl_ids(payload: dict = Body(...)):
        """달력 일 클릭용 — 필터(설비/DSV/레시피)+기간에 해당하는 웨이퍼 id(시간순)."""
        if not ANL["scores"]:
            raise HTTPException(400, "먼저 [분석 시작]을 실행하세요")
        d0 = str(payload.get("d0") or "").strip()
        d1 = str(payload.get("d1") or "").strip()
        eqp9, dsv9, rc9 = _anl_filters(payload)
        meta9 = VZD.get("id2meta") or {}
        ts9 = VZD.get("id2ts") or {}
        ids9 = []
        for i9, dd9 in VZD["id2date"].items():
            if (d0 and dd9 < d0) or (d1 and dd9 > d1):
                continue
            if i9 not in ANL["scores"]:
                continue
            if not _anl_match(meta9.get(i9) or {"eqp": "", "dsv": "", "recipe": ""},
                              eqp9, dsv9, rc9):
                continue
            ids9.append(i9)
        ids9.sort(key=lambda w9: (ts9.get(w9, ""), w9))
        return {"ids": ids9, "n": len(ids9)}

    @app.post("/api/anl_buckets")
    def api_anl_buckets(payload: dict = Body(...)):
        """필터 적용 N일 평균 카드 — 평균 캐시는 필터 서명으로 분리 저장."""
        if not ANL["scores"]:
            raise HTTPException(400, "먼저 [분석 시작]을 실행하세요")
        cfg9 = ANL["cfg"]
        step1, steps2, step3 = cfg9["step1"], cfg9["steps2"], cfg9["step3"]
        R9 = _pt_rules()
        a1 = float(payload.get("alpha1") or R9["exp_drop"]["alpha"])
        a2 = float(payload.get("alpha2") or R9["arches"]["alpha"])
        a3 = float(payload.get("alpha3") or R9["w_bulge"]["delta"])
        n3 = float(payload.get("peak_pct")
                   or R9["w_bulge"].get("peak_pct") or 80.0)
        nd9 = max(1, int(payload.get("ndays") or 1))
        d0 = str(payload.get("d0") or ANL["d0"]).strip()
        d1 = str(payload.get("d1") or ANL["d1"]).strip()
        eqp9, dsv9, rc9 = _anl_filters(payload)
        fsig9 = hashlib.md5(("|".join(sorted(eqp9)) + "#" + "|".join(sorted(dsv9))
                             + "#" + "|".join(sorted(rc9)))
                            .encode("utf-8")).hexdigest()[:8]
        gref9 = ANL["gref"]
        sn1, sn2 = R9["exp_drop"]["sensor"], R9["arches"]["sensor"]
        sn3 = R9["w_bulge"]["sensor"]
        sens9 = [sn9 for sn9, on9 in ((sn1, gref9.get("p1")), (sn2, gref9.get("p2")),
                                      (sn3, gref9.get("p3"))) if on9]
        for c9 in _pt_bal_sens(gref9):
            if c9 not in sens9:
                sens9.append(c9)
        if not sens9:
            raise HTTPException(400, "활성 규칙이 없습니다")
        canon9, Lg, steps_t9, _g9 = _viz_grid()
        if steps_t9 is None:
            raise HTTPException(400, "골든에 스텝 정보가 없습니다")
        meta9 = VZD.get("id2meta") or {}
        import datetime as _dtb
        day_ids9 = {}
        for i9, dd9 in VZD["id2date"].items():
            if (d0 and dd9 < d0) or (d1 and dd9 > d1):
                continue
            if i9 not in ANL["scores"]:
                continue
            if not _anl_match(meta9.get(i9) or {"eqp": "", "dsv": "", "recipe": ""},
                              eqp9, dsv9, rc9):
                continue
            day_ids9.setdefault(dd9, []).append(i9)
        if not day_ids9:
            raise HTTPException(400, "조건에 맞는 웨이퍼가 없습니다")
        days_s9 = sorted(day_ids9)
        base9 = _dtb.date.fromisoformat(days_s9[0])
        bk_days9 = {}
        for dd9 in days_s9:
            bk9 = (_dtb.date.fromisoformat(dd9) - base9).days // nd9
            bk_days9.setdefault(bk9, []).append(dd9)
        if len(bk_days9) > 1000:
            raise HTTPException(400, f"묶음이 {len(bk_days9)}개 — 평균 일수를 키우거나 "
                                     "기간을 좁혀 주세요(최대 1000)")
        cdir9 = _viz_cache_dir(Lg)
        cards9 = []
        for bk9 in sorted(bk_days9):
            acc9 = np.zeros((Lg, len(sens9)))
            cnt9 = np.zeros((Lg, len(sens9)))
            used9 = 0
            for dd9 in bk_days9[bk9]:
                pick9 = _viz_day_pick(sorted(day_ids9[dd9]))
                sig9 = _viz_day_sig(pick9)
                dkey9 = f"{dd9}_{fsig9}"               # 필터별 캐시 분리(덮어쓰기 방지)
                got9 = _viz_day_load(cdir9, dkey9, sens9, sig9, Lg)
                if got9 is None:
                    sd9, cd9, nd_9 = _viz_day_calc(pick9, sens9, canon9, Lg)
                    _viz_day_save(cdir9, dkey9, sens9, sig9, Lg, sd9, cd9, nd_9)
                    got9 = (sd9, cd9, nd_9)
                acc9 += got9[0]
                cnt9 += got9[1]
                used9 += got9[2]
            with np.errstate(all="ignore"):
                m9 = np.where(cnt9 > 0, acc9 / np.maximum(cnt9, 1), np.nan)
            dA9, dB9 = bk_days9[bk9][0], bk_days9[bk9][-1]
            card9 = {"d0": dA9, "d1": dB9, "n_used": used9,
                     "label": f"{dA9}~{dB9[5:]} ({used9}장 평균)", "mats": []}
            for q9, sn9 in enumerate(sens9):
                if sn9 == sn1:
                    rule9 = "p1"
                elif sn9 == sn3:
                    rule9 = "p3"
                elif sn9 == sn2:
                    rule9 = "p2"
                else:
                    continue                           # 계통 전용 센서(아래서 사용)
                want9 = {"p1": {step1}, "p3": {step3}}.get(
                    rule9, {int(v9) for v9 in steps2})
                mat9 = _pt_mat_ctx(m9[:, q9], steps_t9, want9, rule9, sn9,
                                   gref9, a1, a2, a3, n3c9=n3)
                if mat9 is None:
                    continue
                mat9["id"] = card9["label"]
                card9["mats"].append(mat9)
            sl9 = list(sens9)

            def _gc9(sn_q9, _m9=m9, _s9=sl9):
                return _m9[:, _s9.index(sn_q9)] if sn_q9 in _s9 else None
            for key9 in ("p4", "p5"):
                mat9 = _pt_bal_mat(_gc9, steps_t9, key9, gref9)
                if mat9 is not None:
                    mat9["id"] = card9["label"]
                    card9["mats"].append(mat9)
            cards9.append(card9)
        return {"cards": cards9, "ndays": nd9, "n_buckets": len(cards9),
                "off": ANL.get("off") or []}

    @app.post("/api/anl_export")
    def api_anl_export(payload: dict = Body(...)):
        """대화형 HTML 내보내기 재료 일괄 수집(설비 1개 범위).

        rows(판정 행 — 달력 재계산용) + 일평균 카드 + 5일 평균 카드 +
        일별 위반 개별 카드(cap 장/일). 전부 캐시 경유라 2회차부터 빠르다.
        """
        if not ANL["scores"]:
            raise HTTPException(400, "먼저 [분석 시작]을 실행하세요")
        cap9 = max(0, int(payload.get("cap") or 10))
        R9 = _pt_rules()
        a1 = float(payload.get("alpha1") or R9["exp_drop"]["alpha"])
        a2 = float(payload.get("alpha2") or R9["arches"]["alpha"])
        a3 = float(payload.get("alpha3") or R9["w_bulge"]["delta"])
        n3 = float(payload.get("peak_pct")
                   or R9["w_bulge"].get("peak_pct") or 80.0)
        thrs9 = _pt_thrs(ANL["gref"], a1, a2, a3, n3)
        eqp9, dsv9, rc9 = _anl_filters(payload)
        meta9 = VZD.get("id2meta") or {}
        dts9 = VZD.get("id2date") or {}
        ts9 = VZD.get("id2ts") or {}
        d0 = str(payload.get("d0") or ANL["d0"]).strip()
        d1 = str(payload.get("d1") or ANL["d1"]).strip()
        mode9 = str(payload.get("mode") or "full")     # v153: 대용량 분할 수집
        rows9 = []                                     # [date,dsv,recipe,v1,v2,v3]
        viol_day9 = {}
        for wid9, v9 in ANL["scores"].items():
            dd9 = dts9.get(wid9, "")
            if (d0 and dd9 < d0) or (d1 and dd9 > d1):
                continue
            m9 = meta9.get(wid9) or {"eqp": "", "dsv": "", "recipe": ""}
            if not _anl_match(m9, eqp9, dsv9, rc9):
                continue
            verd9 = _pt_verdicts(v9, thrs9)
            t9 = ",".join(verd9)
            f1 = 1 if ("직선" in t9 or "하강" in t9) else 0
            f2 = 1 if "아치" in t9 else 0
            f3 = 1 if "W패턴" in t9 else 0
            f4 = 1 if "온도 계통" in t9 else 0
            f5 = 1 if "압력 계통" in t9 else 0
            rows9.append([dd9, m9["dsv"], m9["recipe"], f1, f2, f3, f4, f5])
            if verd9:
                viol_day9.setdefault(dd9, []).append((ts9.get(wid9, ""), wid9))
        if not rows9:
            raise HTTPException(400, "조건에 맞는 웨이퍼가 없습니다")
        # DSV 별 평균 카드(체크 필터가 그래프에도 적용되도록) + 전체("")
        dsv_all9 = sorted({r9[1] for r9 in rows9 if r9[1]})
        day_by9, five_by9 = {}, {}
        for dv9 in [""] + dsv_all9:
            pl9 = dict(payload, d0=d0, d1=d1)
            if dv9:
                pl9["dsv"] = [dv9]
            try:
                if mode9 in ("full", "day"):
                    day_by9[dv9] = api_anl_buckets(dict(pl9, ndays=1))["cards"]
                if mode9 in ("full", "five"):
                    five_by9[dv9] = api_anl_buckets(dict(pl9, ndays=5))["cards"]
            except HTTPException:
                day_by9.setdefault(dv9, [])
                five_by9.setdefault(dv9, [])
        viol_mats9 = {}
        n_more9 = {}
        if mode9 == "five":                            # 5일 평균만 — 위반 카드 생략
            viol_day9 = {}
        for dd9, lst9 in sorted(viol_day9.items()):
            lst9.sort()
            pick9 = [w9 for _t9, w9 in lst9[:cap9]]
            if len(lst9) > cap9:
                n_more9[dd9] = len(lst9) - cap9
            if not pick9:
                continue
            bk9 = api_pattern_wave_bulk({"ids": pick9,
                                         "rules": [r9 for r9 in
                                                   ("p1", "p2", "p3", "p4", "p5")],
                                         "alpha1": a1, "alpha2": a2, "alpha3": a3})
            out_m9 = []
            for m9x in bk9["mats"]:
                if not m9x.get("verdict"):
                    continue
                mm9 = meta9.get(m9x["id"]) or {}
                m9x["dsv"] = mm9.get("dsv") or ""
                m9x["recipe"] = mm9.get("recipe") or ""
                out_m9.append(m9x)
            viol_mats9[dd9] = out_m9
        return {"rows": rows9, "d0": d0, "d1": d1,
                "day_by": day_by9, "five_by": five_by9,
                "viol": viol_mats9, "viol_more": n_more9,
                "n": len(rows9)}

    @app.post("/api/pattern_wave_bulk")
    def api_pattern_wave_bulk(payload: dict = Body(...)):
        """달력 일/월 보기·HTML 내보내기용 대량 카드 재료 — 파일 단위 청크 로드.

        클라이언트가 웨이퍼 id 를 페이지(수십 장)로 나눠 호출한다(응답 비대 방지).
        """
        ids9 = [str(v9) for v9 in (payload.get("ids") or [])][:200]
        rules9 = [r9 for r9 in (payload.get("rules") or ["p1", "p2", "p3"])
                  if r9 in ("p1", "p2", "p3", "p4", "p5")]
        if not ids9:
            raise HTTPException(400, "웨이퍼 id 를 주세요")
        R9 = _pt_rules()
        cfg9 = VZD.get("pt_cfg") or {}
        step1 = int(cfg9.get("step1") or R9["exp_drop"]["step"])
        steps2 = [int(v9) for v9 in (cfg9.get("steps2") or R9["arches"]["steps"])]
        step3 = int(_pt_step3())
        a1c9 = float(payload.get("alpha1") or cfg9.get("a1")
                     or R9["exp_drop"]["alpha"])
        a2c9 = float(payload.get("alpha2") or cfg9.get("a2")
                     or R9["arches"]["alpha"])
        a3c9 = float(payload.get("alpha3") or cfg9.get("a3")
                     or R9["w_bulge"]["delta"])
        n3c9 = float(payload.get("peak_pct") or cfg9.get("n3")
                     or R9["w_bulge"].get("peak_pct") or 80.0)
        gref9 = _pt_golden(step1, steps2)
        sn_of9 = {"p1": R9["exp_drop"]["sensor"], "p2": R9["arches"]["sensor"],
                  "p3": R9["w_bulge"]["sensor"]}
        want_of9 = {"p1": {step1}, "p2": {int(v9) for v9 in steps2},
                    "p3": {step3}}
        on9 = {r9 for r9 in rules9 if gref9.get(r9)}
        by_f9 = {}
        for w9 in ids9:
            f9 = (VZD.get("id2src") or {}).get(w9)
            if f9:
                by_f9.setdefault(f9, []).append(w9)
        ts9 = VZD.get("id2ts") or {}
        mats9 = []
        for f9, sts9 in by_f9.items():
            for c09 in range(0, len(sts9), 50):
                recs9, _a, _b, _c = S._load_aligned(Path(f9), stems=sts9[c09:c09 + 50])
                for rec9 in recs9:
                    wid9 = str(rec9["wafer_id"])
                    cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
                    st9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
                    arr9 = np.asarray(rec9["sensors"], dtype=np.float64)
                    for r9 in rules9:
                        if r9 not in on9:
                            continue
                        if r9 in ("p4", "p5"):
                            m9 = _pt_bal_mat(
                                lambda sn_q9: (arr9[:, cols9.index(sn_q9)]
                                               if sn_q9 in cols9 else None),
                                st9, r9, gref9)
                        elif sn_of9[r9] not in cols9:
                            continue
                        else:
                            m9 = _pt_mat_ctx(arr9[:, cols9.index(sn_of9[r9])],
                                             st9, want_of9[r9], r9,
                                             sn_of9[r9], gref9,
                                             a1c9, a2c9, a3c9, n3c9=n3c9)
                        if m9 is None:
                            continue
                        m9["id"] = wid9
                        m9["ts"] = ts9.get(wid9, "")
                        mats9.append(m9)
                del recs9
        mats9.sort(key=lambda m9: (m9.get("ts") or "", m9["id"], m9["rule"]))
        return {"mats": mats9, "n": len(mats9)}

    @app.post("/api/pattern_narrate")
    def api_pattern_narrate(payload: dict = Body(...)):
        """패턴 인식 결과의 고객 보고용 정리.

        1단계(항상): 서버가 결정적 요약을 만든다 — 판정별 연속 날짜 병합
        '3/8 ~ 3/10 이상 패턴 감지(직선 하강 N장) — 확인 요망'. 수치 조작 없음.
        2단계(LLM 가능 시): 같은 사실표를 넘겨 개조식 결론로 다듬는다(추가 사실 금지).
        """
        rows9 = payload.get("rows") or []
        if not rows9:
            raise HTTPException(400, "먼저 [패턴 스캔]을 실행하세요")
        import datetime as _dtp
        by_v9 = {}
        day_n9 = {}
        for r9 in rows9:
            d9 = str(r9.get("date") or "")
            day_n9[d9] = day_n9.get(d9, 0) + 1
            for v9 in (r9.get("verdicts") or []):
                by_v9.setdefault(v9, {}).setdefault(d9, 0)
                by_v9[v9][d9] += 1

        def _md(d9):
            return f"{int(d9[5:7])}/{int(d9[8:10])}"

        lines9 = []
        for v9 in sorted(by_v9):
            ds9 = sorted(by_v9[v9])
            runs9 = []
            for d9 in ds9:
                dt9 = _dtp.date.fromisoformat(d9)
                if runs9 and (dt9 - runs9[-1][1]).days == 1:
                    runs9[-1][1] = dt9
                    runs9[-1][2] += by_v9[v9][d9]
                else:
                    runs9.append([dt9, dt9, by_v9[v9][d9]])
            for a9, b9, n9 in runs9:
                span9 = (_md(a9.isoformat()) if a9 == b9
                         else f"{_md(a9.isoformat())} ~ {_md(b9.isoformat())}")
                lines9.append(f"{span9} 이상 패턴 감지({v9} {n9}장) — 확인 요망")
        if not lines9:
            d_all9 = sorted(day_n9)
            span9 = (f"{_md(d_all9[0])} ~ {_md(d_all9[-1])}"
                     if len(d_all9) > 1 else _md(d_all9[0])) if d_all9 else ""
            lines9 = [f"{span9} 전 기간 이상 패턴 없음".strip()]
        # [골든 대비] — 판정 그룹별 오프셋 평균(원단위)과 패턴 점수 vs 골든
        gold9 = payload.get("golden") or {}
        sens9 = payload.get("sensors") or {}

        def _u(sn9):
            return "\u2103" if str(sn9).startswith("T") else "bar"

        def _grp(rows_g9, key9):
            vs9 = [r9.get(key9) for r9 in rows_g9
                   if isinstance(r9.get(key9), (int, float))]
            return (sum(vs9) / len(vs9)) if vs9 else None

        cmp9 = []
        viol_r9 = [r9 for r9 in rows9 if r9.get("verdicts")]
        norm_r9 = [r9 for r9 in rows9 if not r9.get("verdicts")]
        for tag9, grp9 in (("위반", viol_r9), ("정상", norm_r9)):
            if not grp9:
                continue
            seg_t9 = []
            o19 = _grp(grp9, "p1_off")
            r19 = _grp(grp9, "p1_ratio")
            if o19 is not None and gold9.get("p1"):
                seg_t9.append(f"{sens9.get('p1','P1')} 오프셋 평균 "
                              f"{o19:+.2f}{_u(sens9.get('p1','T'))}"
                              + (f" · 볼록비 {r19:.2f}(골든 {gold9['p1']['ratio']})"
                                 if r19 is not None else ""))
            o29 = _grp(grp9, "p2_off")
            a29 = _grp(grp9, "p2_area")
            if o29 is not None and gold9.get("p2"):
                seg_t9.append(f"{sens9.get('p2','P2')} 오프셋 평균 "
                              f"{o29:+.2f}{_u(sens9.get('p2','P'))}"
                              + (f" · 아치 면적 {a29:.2f}(골든 {gold9['p2']['area']})"
                                 if a29 is not None else ""))
            o39 = _grp(grp9, "p3_off")
            r39 = _grp(grp9, "p3_area")
            if o39 is not None and gold9.get("p3"):
                seg_t9.append(f"{sens9.get('p3','W패턴')} 오프셋 평균 "
                              f"{o39:+.2f}{_u(sens9.get('p3','T'))}"
                              + (f" · W면적 {r39:.2f}(골든 {gold9['p3']['area']})"
                                 if r39 is not None else ""))
            if seg_t9:
                cmp9.append(f"{tag9} {len(grp9)}장: " + " / ".join(seg_t9))
        # [기간 추세] — 규칙별 일평균 점수: 위반 시작일 → 최악일(정도 최대)
        trend9 = []
        rule_map9 = [("직선 하강(볼록비)", "p1_ratio", "p1", "min",
                      lambda r9: "직선" in r9 or "하강" in r9),
                     ("아치(면적)", "p2_area", "p2", "min",
                      lambda r9: "아치" in r9),
                     ("W패턴(면적)", "p3_area", "p3", "max",
                      lambda r9: "W패턴" in r9),
                     ("온도 계통(잔차 z)", "p4_z", "p4", "max",
                      lambda r9: "온도 계통" in r9),
                     ("압력 계통(잔차 z)", "p5_z", "p5", "max",
                      lambda r9: "압력 계통" in r9)]
        for nm9, key9, gk9, mode9, is_v9 in rule_map9:
            if not gold9.get(gk9):
                continue
            daily9 = {}
            vdays9 = []
            for r9 in rows9:
                v9x = r9.get(key9)
                if isinstance(v9x, (int, float)):
                    daily9.setdefault(r9["date"], []).append(v9x)
                if any(is_v9(v9y) for v9y in (r9.get("verdicts") or [])):
                    vdays9.append(r9["date"])
            if not daily9:
                continue
            dm9 = {d9: sum(v9x) / len(v9x) for d9, v9x in daily9.items()}
            gv9 = gold9[gk9].get("ratio", gold9[gk9].get(
                "area", gold9[gk9].get("z")))
            if vdays9:
                on9 = min(vdays9)
                pk9 = (min if mode9 == "min" else max)(dm9, key=dm9.get)
                trend9.append(f"{nm9}: {_md(on9)} 위반 시작 → {_md(pk9)} 최대"
                              f"(일평균 {dm9[pk9]:.3f}, 골든 {gv9})")
            else:
                lo9, hi9 = min(dm9.values()), max(dm9.values())
                trend9.append(f"{nm9}: 기간 내 위반 없음(일평균 {lo9:.3f}~{hi9:.3f}, "
                              f"골든 {gv9})")
        facts9 = ("총 " + str(len(rows9)) + "장 검사\n"
                  + "\n".join("- " + l9 for l9 in lines9)
                  + (("\n[기간 추세]\n" + "\n".join("- " + l9 for l9 in trend9))
                     if trend9 else "")
                  + (("\n[골든 대비]\n" + "\n".join("- " + l9 for l9 in cmp9))
                     if cmp9 else ""))
        llm9 = None
        eng9 = ""
        if not payload.get("no_llm"):
            try:
                from io_recon_tcn import rca_causal as _RCn
                r9 = _RCn.narrate_pattern(facts9)
                if r9.get("engine") != "error":
                    llm9, eng9 = r9.get("narrative"), r9.get("engine", "")
            except Exception as e9:                    # noqa: BLE001
                print(f"[pattern] LLM 정리 불가(결정적 요약만): {e9}", flush=True)
        return {"summary": lines9, "trend": trend9, "golden_cmp": cmp9,
                "n": len(rows9), "llm": llm9, "engine": eng9}

    @app.post("/api/pattern_wave")
    def api_pattern_wave(payload: dict = Body(...)):
        """위반 행 클릭 팝업: 스텝 구간 원본 + 현/받침점 오버레이 재료."""
        wid9 = str(payload.get("id") or "")
        rule9 = str(payload.get("rule") or "p1")
        R9 = _pt_rules()
        step1 = int(payload.get("step1") or R9["exp_drop"]["step"])
        steps2 = [int(v9) for v9 in (payload.get("steps2") or R9["arches"]["steps"])]
        step3 = int(payload.get("step3") or _pt_step3())
        if rule9 in ("p4", "p5"):                      # 계통 잔차 카드
            gref9 = _pt_golden(step1, steps2)
            if payload.get("golden"):
                rm9 = VZD.get("rmean") or {}
                sr9 = VZD.get("steps_ref")
                if sr9 is None:
                    raise HTTPException(400, "골든에 스텝 정보가 없습니다")
                out9 = _pt_bal_mat(lambda sn_q9: rm9.get(sn_q9), sr9,
                                   rule9, gref9)
                wid9 = "골든(정상 구간 평균)"
            else:
                f9 = (VZD.get("id2src") or {}).get(wid9)
                if not f9:
                    raise HTTPException(400, "웨이퍼를 찾지 못했습니다(목록을 다시 불러오세요)")
                recs9, _a, _b, _c = S._load_aligned(Path(f9), stems=[wid9])
                if not recs9:
                    raise HTTPException(404, "웨이퍼 파일을 읽지 못했습니다")
                rec9 = recs9[0]
                cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
                st9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
                arr9 = np.asarray(rec9["sensors"], dtype=np.float64)
                out9 = _pt_bal_mat(
                    lambda sn_q9: (arr9[:, cols9.index(sn_q9)]
                                   if sn_q9 in cols9 else None),
                    st9, rule9, gref9)
            if out9 is None:
                raise HTTPException(400, "계통 규칙 비활성/센서 없음 — "
                                    + " / ".join(gref9["off"]))
            out9["id"] = wid9
            return out9
        sn9 = {"p1": R9["exp_drop"]["sensor"],
               "p3": R9["w_bulge"]["sensor"]}.get(rule9, R9["arches"]["sensor"])
        want9 = {"p1": {step1}, "p3": {step3}}.get(rule9,
                                                   {int(v9) for v9 in steps2})
        if payload.get("golden"):                      # 답지 카드: 골든 곡선 자체
            rm9 = VZD.get("rmean") or {}
            sr9 = VZD.get("steps_ref")
            if sn9 not in rm9 or sr9 is None:
                raise HTTPException(400, f"골든에 {sn9} 없음")
            ysrc9, stsrc9 = rm9[sn9], sr9
            wid9 = "골든(정상 구간 평균)"
        else:
            f9 = (VZD.get("id2src") or {}).get(wid9)
            if not f9:
                raise HTTPException(400, "웨이퍼를 찾지 못했습니다(목록을 다시 불러오세요)")
            recs9, _a, _b, _c = S._load_aligned(Path(f9), stems=[wid9])
            if not recs9:
                raise HTTPException(404, "웨이퍼 파일을 읽지 못했습니다")
            rec9 = recs9[0]
            cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
            if sn9 not in cols9:
                raise HTTPException(400, f"이 웨이퍼에 {sn9} 없음")
            stsrc9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
            ysrc9 = np.asarray(rec9["sensors"], dtype=np.float64)[:, cols9.index(sn9)]
        gref9 = _pt_golden(step1, steps2)
        cfg9 = VZD.get("pt_cfg") or {}
        a1c9 = float(cfg9.get("a1") or R9["exp_drop"]["alpha"])
        a2c9 = float(cfg9.get("a2") or R9["arches"]["alpha"])
        a3c9 = float(cfg9.get("a3") or R9["w_bulge"]["delta"])
        n3c9 = float(cfg9.get("n3") or R9["w_bulge"].get("peak_pct") or 80.0)
        out9 = _pt_mat_ctx(ysrc9, stsrc9, want9, rule9, sn9, gref9, a1c9, a2c9,
                           a3c9, n3c9=n3c9)
        if out9 is None:
            raise HTTPException(400, "해당 스텝 구간이 없습니다")
        out9["id"] = wid9
        return out9

    def _preset_fp(store9, cache9):
        """프리셋 파일: 캐시 폴더 > 골든 저장소 > 앱 폴더 (사용자 관리 폴더 우선)."""
        for c9 in (cache9, store9):
            c9 = str(c9 or "").strip().strip('"')
            if not c9:
                continue
            p9 = Path(c9)
            try:
                p9.mkdir(parents=True, exist_ok=True)
                if p9.is_dir():
                    return p9 / "avgwave_presets.json"
            except Exception:                          # noqa: BLE001
                continue
        return STATE_FP.parent / "avgwave_presets.json"

    def _preset_all(fp9):
        try:
            d9 = json.loads(fp9.read_text(encoding="utf-8"))
            return d9 if isinstance(d9, dict) else {}
        except Exception:                              # noqa: BLE001
            return {}

    @app.post("/api/preset_save")
    def api_preset_save(payload: dict = Body(...)):
        """분석 프리셋 저장(탭 scope 별) — 입력칸·선택 체크·모드를 한 세트로."""
        # 이름: 공백은 보존(사용자 표기 유지), 경로/구분자 위험 문자만 치환
        name = re.sub(r"[^0-9A-Za-z가-힣_.\- ]+", "_",
                      str(payload.get("name") or "").strip())
        if not name:
            raise HTTPException(400, "프리셋 이름을 입력하세요")
        scope = str(payload.get("scope") or "sd")
        fp9 = _preset_fp(payload.get("store"), payload.get("cache"))
        all9 = _preset_all(fp9)
        key9 = f"{scope}:{name}"
        existed = key9 in all9
        all9[key9] = {"scope": scope, "name": name,
                      "saved": __import__("datetime").datetime.now()
                      .isoformat(timespec="seconds"),
                      "data": payload.get("data") or {}}
        try:
            fp9.write_text(json.dumps(all9, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        except Exception as e9:                        # noqa: BLE001
            raise HTTPException(400, f"프리셋 저장 실패: {e9}")
        print(f"[preset] 저장: {fp9} ({key9})" + (" (덮어씀)" if existed else ""),
              flush=True)
        return {"ok": True, "path": str(fp9), "name": name, "overwritten": existed}

    @app.get("/api/preset_list")
    def api_preset_list(scope: str = "sd", store: str = "", cache: str = ""):
        fp9 = _preset_fp(store, cache)
        items = [{"name": v9.get("name"), "saved": v9.get("saved")}
                 for k9, v9 in sorted(_preset_all(fp9).items())
                 if str(v9.get("scope")) == scope]
        return {"path": str(fp9), "items": items}

    @app.post("/api/preset_load")
    def api_preset_load(payload: dict = Body(...)):
        fp9 = _preset_fp(payload.get("store"), payload.get("cache"))
        key9 = f"{payload.get('scope') or 'sd'}:{payload.get('name') or ''}"
        it9 = _preset_all(fp9).get(key9)
        if not it9:
            raise HTTPException(400, f"프리셋이 없습니다: {payload.get('name')}")
        return {"name": it9.get("name"), "saved": it9.get("saved"),
                "data": it9.get("data") or {}}

    @app.post("/api/preset_del")
    def api_preset_del(payload: dict = Body(...)):
        fp9 = _preset_fp(payload.get("store"), payload.get("cache"))
        all9 = _preset_all(fp9)
        key9 = f"{payload.get('scope') or 'sd'}:{payload.get('name') or ''}"
        if key9 not in all9:
            raise HTTPException(400, "프리셋이 없습니다")
        all9.pop(key9)
        try:
            fp9.write_text(json.dumps(all9, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        except Exception as e9:                        # noqa: BLE001
            raise HTTPException(400, f"삭제 실패: {e9}")
        return {"ok": True}

    @app.post("/api/viz_cmp")
    def api_viz_cmp(payload: dict = Body(...)):
        """두 기간 직접 비교: 기간 A·B 각각 평균 곡선 + 센서×스텝 차이 랭킹.

        A(기준)·B(대상) 웨이퍼를 균등 표본으로 스트리밍 로드해 골든 스텝 격자에
        정렬(_warp_to_canon), 센서별 평균 곡선과 z 랭킹(σ는 A 의 웨이퍼 간 산포)을 낸다.
        """
        if not VZD.get("id2src"):
            raise HTTPException(400, "먼저 [웨이퍼 목록 불러오기]를 실행하세요")
        sens = [str(s9) for s9 in (payload.get("sensors") or [])][:12]
        if not sens:
            raise HTTPException(400, "센서를 선택하세요")
        rm0 = VZD.get("rmean") or {}
        miss9 = [s9 for s9 in sens if s9 not in rm0]
        if miss9:
            raise HTTPException(400, f"골든에 없는 센서: {', '.join(miss9)}")
        dmap9 = VZD["id2date"]
        def _pick(dates9, cap9=200):
            sel9 = set(dates9 or [])
            cand9 = sorted((d9, i9) for i9, d9 in dmap9.items() if d9 in sel9)
            if not cand9:
                return [], None, None
            step9 = max(1, len(cand9) // cap9)
            ids9 = [i9 for _d9, i9 in cand9[::step9]][:cap9]
            return ids9, cand9[0][0], cand9[-1][0]
        idsA, a0, a1 = _pick(payload.get("datesA"))
        idsB, b0, b1 = _pick(payload.get("datesB"))
        if not idsA or not idsB:
            raise HTTPException(400, "기간 A·B 를 각각 지정하세요(해당 기간 웨이퍼 없음)")
        canon_runs9, Lg, steps_t9, golden_t9 = _viz_grid()
        sref9 = steps_t9
        usteps9 = (sorted({int(v9) for v9 in sref9} - {0}) if sref9 is not None else [])
        masks9 = ([_erode_mask(np.asarray(sref9) == u9) for u9 in usteps9]
                  if usteps9 else [])

        def _load(ids9):
            by_f9 = {}
            for w9 in ids9:
                f9 = VZD["id2src"].get(w9)
                if f9:
                    by_f9.setdefault(f9, []).append(w9)
            acc9 = np.zeros((Lg, len(sens)))
            cnt9 = np.zeros((Lg, len(sens)))
            stepvals9 = []                             # 웨이퍼별 스텝 평균(σ 산출용)
            n9 = 0
            for f9, st9s in by_f9.items():
                for c0 in range(0, len(st9s), 100):
                    recs9, _a, _b, _c = S._load_aligned(Path(f9),
                                                        stems=st9s[c0:c0 + 100])
                    for rec9 in recs9:
                        cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
                        ii9 = [cols9.index(s9) for s9 in sens if s9 in cols9]
                        if len(ii9) != len(sens):
                            continue
                        arr9 = np.asarray(rec9["sensors"], dtype=np.float64)[:, ii9]
                        try:
                            stw9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
                            wv9 = (_warp_to_canon(arr9, stw9, canon_runs9, Lg)
                                   if canon_runs9 is not None else
                                   np.column_stack([_resample(arr9[:, q9], Lg)
                                                    for q9 in range(arr9.shape[1])]))
                        except Exception:              # noqa: BLE001
                            wv9 = np.column_stack([_resample(arr9[:, q9], Lg)
                                                   for q9 in range(arr9.shape[1])])
                        fin9 = np.isfinite(wv9)
                        acc9 += np.where(fin9, wv9, 0.0)
                        cnt9 += fin9
                        if masks9:
                            with np.errstate(all="ignore"):
                                stepvals9.append(np.array(
                                    [[float(np.nanmean(wv9[mk9, q9]))
                                      for q9 in range(len(sens))]
                                     for mk9 in masks9]))
                        n9 += 1
                    del recs9
            with np.errstate(all="ignore"):
                mean9 = np.where(cnt9 > 0, acc9 / np.maximum(cnt9, 1), np.nan)
            return mean9, (np.stack(stepvals9) if stepvals9 else None), n9
        mA, svA, nA = _load(idsA)
        mB, svB, nB = _load(idsB)
        if not nA or not nB:
            raise HTTPException(400, "원본 파형을 읽지 못했습니다(폴더 연결 확인)")
        excl9 = None
        try:
            from io_recon_tcn import rca_causal as _RCx
            _t9, PHx = _RCx.load_physics()
            excl9 = re.compile(str(PHx.get("exclude_sensors_regex") or ""))
        except Exception:                              # noqa: BLE001
            pass
        rank9 = []
        if svA is not None and svB is not None and usteps9:
            with np.errstate(all="ignore"):
                mAs = np.nanmean(svA, axis=0)          # (K, nsens)
                sAs = np.nanstd(svA, axis=0)
                mBs = np.nanmean(svB, axis=0)
            for q9, sn9 in enumerate(sens):
                if excl9 is not None and excl9.search(sn9):
                    continue                           # 추종형은 랭킹 제외(표시는 유지)
                flo9 = 0.01 if re.match(r"(?i)^d?p", sn9) else 0.05
                for k9, u9 in enumerate(usteps9):
                    a9v, b9v = float(mAs[k9, q9]), float(mBs[k9, q9])
                    if not (np.isfinite(a9v) and np.isfinite(b9v)):
                        continue
                    den9 = max(float(sAs[k9, q9]) if np.isfinite(sAs[k9, q9]) else 0.0,
                               flo9)
                    rank9.append({"sensor": sn9, "step": int(u9),
                                  "a": round(a9v, 3), "b": round(b9v, 3),
                                  "delta": round(b9v - a9v, 3),
                                  "z": round(abs(b9v - a9v) / den9, 1),
                                  "virtual": bool(re.match(r"(?i)^d[tp]_", sn9))})
            rank9.sort(key=lambda x9: -x9["z"])
            rank9 = rank9[:30]
        cur9 = {}
        for q9, sn9 in enumerate(sens):
            cur9[sn9] = {
                "A": [None if not np.isfinite(v9) else round(float(v9), 4)
                      for v9 in mA[:, q9]],
                "B": [None if not np.isfinite(v9) else round(float(v9), 4)
                      for v9 in mB[:, q9]]}
        return {"sensors": sens, "curves": cur9,
                "golden": {s9: [round(float(v9), 4) for v9 in golden_t9[s9]]
                           for s9 in sens},
                "steps": ([int(v9) for v9 in sref9] if sref9 is not None else None),
                "rank": rank9, "nA": nA, "nB": nB,
                "periodA": [a0, a1], "periodB": [b0, b1]}

    @app.post("/api/viz_reports")
    def api_viz_reports(payload: dict = Body(...)):
        """리포트(불량/조치) CSV → 강조 날짜 계산.

        주황 = 리포트된 날(연속이면 전부 주황 유지).
        노랑 = 연속 리포트 묶음의 '최초' 리포트일 기준 3일 전까지(리포트일은 제외).
        dsv 별로 따로 계산 — 화면의 DSV 필터가 있으면 그 DSV 만 사용.
        """
        days_hint = sorted(set((VZD.get("id2date") or {}).values()))
        events, skipped = _parse_label_csv(payload.get("csv"), days_hint=days_hint)
        dsv_f = [("dsv" + t9 if t9.isdigit() else t9.lower())
                 for t9 in _parse_multi(payload.get("dsv"))]
        def _nd(v9):
            v9 = str(v9 or "").strip().lower()
            return ("dsv" + v9 if v9.isdigit() else v9)
        use9 = [e9 for e9 in events
                if not dsv_f or not _nd(e9.get("dsv")) or _nd(e9.get("dsv")) in dsv_f]
        import datetime as _dtr
        by_dsv9 = {}
        for e9 in use9:
            by_dsv9.setdefault(_nd(e9.get("dsv")) or "(전체)", []).append(e9["date"])
        orange9, yellow9 = set(), set()
        for dsv9, ds9 in by_dsv9.items():
            u9 = sorted(set(ds9))
            orange9 |= set(u9)
            runs9, cur9 = [], [u9[0]]
            for p9, c9 in zip(u9, u9[1:]):
                if (_dtr.date.fromisoformat(c9)
                        - _dtr.date.fromisoformat(p9)).days == 1:
                    cur9.append(c9)
                else:
                    runs9.append(cur9); cur9 = [c9]
            runs9.append(cur9)
            for r9 in runs9:                            # 최초 리포트일 기준 3일 전까지
                f9 = _dtr.date.fromisoformat(r9[0])
                for k9 in (1, 2, 3):
                    yellow9.add((f9 - _dtr.timedelta(days=k9)).isoformat())
        yellow9 -= orange9
        labels9 = {}
        for e9 in use9:
            labels9.setdefault(e9["date"], []).append(
                ((_nd(e9.get("dsv")) + " ") if e9.get("dsv") else "") + e9["label"]
                + ((" · " + e9["memo"]) if e9.get("memo") else ""))
        VZD["rpt"] = {"orange": sorted(orange9), "yellow": sorted(yellow9),
                      "labels": labels9}
        _state_save("vzrpt", {"csv": str(payload.get("csv") or "")})
        return {"n": len(use9), "n_all": len(events), "skipped": skipped[:20],
                "orange": sorted(orange9), "yellow": sorted(yellow9),
                "labels": labels9, "dsv_used": dsv_f or ["(전체)"]}

    @app.post("/api/viz_save_ids")
    def api_viz_save_ids(payload: dict = Body(...)):
        """체크된 웨이퍼 이름을 사용자가 지정한 경로/파일명의 CSV 로 저장."""
        ids = [str(x9) for x9 in (payload.get("ids") or [])]
        if not ids:
            raise HTTPException(400, "저장할 웨이퍼를 선택하세요")
        p9 = Path(str(payload.get("path") or "").strip().strip('"'))
        if not str(p9) or str(p9) == ".":
            raise HTTPException(400, "저장할 CSV 경로/파일명을 입력하세요")
        if p9.suffix.lower() != ".csv":
            p9 = p9.with_suffix(".csv")
        try:
            p9.parent.mkdir(parents=True, exist_ok=True)
            d_map9 = VZD.get("id2date") or {}
            with open(p9, "w", encoding="utf-8-sig", newline="") as f9:
                f9.write("wafer_id,date\n")
                for wid9 in ids:
                    f9.write(f"{wid9},{d_map9.get(wid9, '')}\n")
        except Exception as e9:                        # noqa: BLE001
            raise HTTPException(400, f"저장 실패: {e9}")
        print(f"[viz] 웨이퍼 목록 저장: {p9} ({len(ids)}장)", flush=True)
        return {"ok": True, "path": str(p9), "n": len(ids)}

    @app.post("/api/sd_narrate")
    def api_sd_narrate(payload: dict = Body(...)):
        """스텝 차이 자동 추림 목록의 LLM 정리 — 가상센서(역류 등) 소견 우선."""
        res = SDJOB.get("result") or {}
        fd = res.get("findings") or {}
        if not (fd.get("holds") or fd.get("jumps")):
            raise HTTPException(400, "추림된 이상 구간이 없습니다 — 스텝 차이 집계를 먼저 실행하세요")
        pat_lines = []
        sel_dates = payload.get("dates") or []
        sel_sens = payload.get("sensors") or None
        sens_set = set(sel_sens) if sel_sens else None
        pmode = str(payload.get("pattern_mode") or "prev")
        if sel_dates and SDD.get("daymu") is not None and pmode == "seq":
            try:
                pj = _sd_patterns(sel_dates, "seq", sel_sens)
                unit0 = lambda sn: "bar" if re.match(r"(?i)^d?p", sn) else "℃"
                if pj["rows"]:
                    pat_lines.append(f"[선택일 순차 비교 {pj['period'][0]}~{pj['period'][1]} ({pj['n_days']}일 체크)]")
                for r9 in pj["rows"][:20]:
                    seq9 = " · ".join(f"{d9['a'][5:]}→{d9['b'][5:]} {d9['delta']:+g}"
                                      for d9 in r9["deltas"][:8])
                    pat_lines.append(f"{r9['kind']}: {r9['sensor']} step{r9['step']} — {seq9}"
                                     f" (누적 {r9['cum']:+g}{unit0(r9['sensor'])})"
                                     + (" [가상센서]" if r9.get("virtual") else ""))
            except HTTPException:
                pass
        elif sel_dates and SDD.get("daymu") is not None:
            try:
                pj = _sd_patterns(sel_dates, "prev", sel_sens)
                unit0 = lambda sn: "bar" if re.match(r"(?i)^d?p", sn) else "℃"
                if pj["rows"]:
                    pat_lines.append(f"[선택 기간 패턴 {pj['period'][0]}~{pj['period'][1]} ({pj['n_days']}일)]")
                for r9 in pj["rows"][:25]:
                    pat_lines.append(
                        f"{r9['kind']}: {r9['sensor']} step{r9['step']} — {r9['d0']} 기점, "
                        f"처음 {r9['first']:+g}→끝 {r9['last']:+g}{unit0(r9['sensor'])}"
                        f" (일당 {r9['slope']:+g})"
                        + (" [가상센서]" if r9.get("virtual") else ""))
            except HTTPException:
                pass
        last_day = (res.get("dates") or ["?"])[-1]
        unit = lambda sn: "bar" if re.match(r"(?i)^d?p", sn) else "℃"
        # ── 분석 범위 = 사용자가 체크한 날짜만(전체 체크면 전 기간과 동일) ──
        all_days = res.get("dates") or []
        scope = sorted(set(sel_dates) & set(all_days)) if sel_dates else list(all_days)
        scope_set = set(scope)
        narrowed = bool(scope) and len(scope) < len(all_days)

        all_sens = res.get("sensors") or []
        sens_narrowed = bool(sens_set) and len(sens_set & set(all_sens)) < len(all_sens)

        def _in_scope(d0_9, d1_9=None):
            """기간(또는 단일 일자)이 선택 날짜와 하루라도 겹치면 포함."""
            if not scope:
                return True
            if d1_9 is None:
                return d0_9 in scope_set
            return not (d1_9 < scope[0] or d0_9 > scope[-1]) and any(
                d0_9 <= d9 <= d1_9 for d9 in scope)

        def _sen_ok(sn9):
            return sens_set is None or sn9 in sens_set
        lines = ["※ 가상센서(dT_/dP_) = 차분 유추 센서 — 해석 규칙 우선 적용"]
        # [물리 위반] — 설비 동작 방식상 나와서는 안 되는 패턴(최우선)
        ph_rows = [p8 for p8 in (fd.get("phys") or [])
                   if _in_scope(p8["d0"], p8["d1"]) and _sen_ok(p8.get("sensor") or "")]
        if ph_rows:
            lines.append("[물리 위반 — 설비 동작 방식상 나와서는 안 되는 패턴, 최우선]")
            for p9 in ph_rows[:12]:
                base9 = (f"{p9['kind']}: {p9.get('sensor') or ''}"
                         + (f" step{p9['step']}" if p9.get("step") is not None else "")
                         + (f" (밸브 {p9['io']})" if p9.get("io") else "")
                         + f" — {p9['d0']}~{p9['d1']} ({p9['days']}일)")
                if p9["kind"].startswith("아치"):
                    base9 += (f" 위반 웨이퍼 최대 {int(p9['ratio_peak'] * 100)}% · "
                              f"골든은 깊이 {p9.get('depth_g')} 이탈 후 회복(아치)하나 "
                              f"웨이퍼는 미회복(포화)")
                elif p9["kind"] == "밸브 타이밍":
                    base9 += f" 열림 시점 골든 대비 {p9.get('dt_pct')}% 어긋남"
                else:
                    base9 += f" 위반 웨이퍼 최대 {int(p9['ratio_peak'] * 100)}%"
                lines.append(base9 + (" ★진행 중" if p9.get("ongoing") else ""))
        # 요소① — outlet 기준 음/양압(역류 판정)
        try:
            pr_rows = _sd_pressure_state(scope, sel_sens)
        except Exception:                              # noqa: BLE001
            pr_rows = []
        if pr_rows:
            lines.append("[요소① 음/양압 상태 — outlet 기준, 음압=outlet 이 더 높음=역류]")
            for p9 in pr_rows[:12]:
                lines.append(
                    f"{p9['sensor']} step{p9['step']}: {p9['state']} — "
                    f"음압 {p9['n_neg']}/{p9['n_days']}일 · 최저 {p9['min_raw']:+g}bar · "
                    f"평균 {p9['mean_raw']:+g}bar")
        # 요소② — 스텝별 골든 대비 차이(선택 범위 평균, 유의 상위)
        try:
            if SDD.get("daymu") is not None and scope:
                days_u9 = SDD["days_u"]
                pos9 = {d8: i8 for i8, d8 in enumerate(days_u9)}
                idx9 = [pos9[d8] for d8 in scope if d8 in pos9]
                diff_rows = []
                excl8 = set(SDD.get("excl_sensors") or [])
                for si8, sn8 in enumerate(SDD["sensors"]):
                    if (sens_set is not None and sn8 not in sens_set) or sn8 in excl8:
                        continue
                    for ki8, u8 in enumerate(SDD["usteps"]):
                        v8 = SDD["daymu"][idx9, si8, ki8]
                        v8 = v8[np.isfinite(v8)]
                        if not v8.size:
                            continue
                        t8 = float(SDD["thr"][si8, ki8])
                        m8 = float(np.mean(v8))
                        if abs(m8) < t8 * 0.5:
                            continue
                        diff_rows.append((abs(m8) / max(t8, 1e-9), sn8, int(u8), m8))
                diff_rows.sort(reverse=True)
                if diff_rows:
                    lines.append("[요소② 스텝별 골든 대비 차이 — 선택 기간 평균(원단위), 유의 상위]")
                    for sc8, sn8, u8, m8 in diff_rows[:15]:
                        lines.append(f"{sn8} step{u8}: 골든 대비 {m8:+g}"
                                     f"{unit(sn8)} (문턱의 {sc8:.1f}배)")
        except Exception:                              # noqa: BLE001
            pass
        if narrowed or sens_narrowed:
            rng9 = (f"날짜 {scope[0]}~{scope[-1]} ({len(scope)}일 체크)" if narrowed else "전 기간")
            sn9t = (f"센서 {len(sens_set & set(all_sens))}개만 체크" if sens_narrowed else "전 센서")
            lines.insert(0, f"[분석 범위] 사용자 선택 한정 — {rng9} · {sn9t}. "
                            "이 범위 밖 날짜·센서는 언급하지 말 것")
        lines.append("[상세 추림 — 자동 검출 원본(부호 위반·지속·급변)]")
        for v9 in [v8 for v8 in (fd.get("violations") or [])
                   if _in_scope(v8["d0"], v8["d1"]) and _sen_ok(v8["sensor"])][:10]:
            if v9.get("golden_neg"):
                lines.append(
                    f"[⚠골든 음압] {v9['sensor']} step{v9['step']} — 기준 골든 자체가 "
                    f"음압({v9['g0']}bar) — 역류가 아니라 정상 구간 데이터/센서 배선 확인 필요")
                continue
            lines.append(
                f"[★부호 위반] {v9['sensor']} step{v9['step']} — {v9['d0']}~{v9['d1']} "
                f"({v9['days']}일) 원시값 최저 {v9['min_raw']}bar < 0 "
                f"(골든은 이 스텝 내내 양수·최저 {v9.get('g_min', 0):+g}bar — "
                "웨이퍼만 음수 = 역류 신호)"
                + (" ★진행 중" if v9.get("ongoing") else ""))
        for h9 in [h8 for h8 in (fd.get("holds") or [])
                   if _in_scope(h8["d0"], h8["d1"]) and _sen_ok(h8["sensor"])][:25]:
            lines.append(
                f"[지속] {h9['sensor']} step{h9['step']} — {h9['d0']}~{h9['d1']} "
                f"({h9['days']}일) 골든 대비 평균 {h9['mean']:+g}{unit(h9['sensor'])} · "
                f"최대 {h9['peak']:+g}{unit(h9['sensor'])} ({'상승' if h9['dir'] == '+' else '하락'})"
                + (" ★진행 중(말일까지 지속)" if h9.get("ongoing") else "")
                + (" [가상센서]" if h9.get("virtual") else ""))
        for j9 in [j8 for j8 in (fd.get("jumps") or [])
                   if _in_scope(j8["date"]) and _sen_ok(j8["sensor"])][:20]:
            lines.append(
                f"[급변] {j9['sensor']} step{j9['step']} — {j9['date']} 전일 대비 "
                f"{j9['delta']:+g}{unit(j9['sensor'])}"
                + (" [가상센서]" if j9.get("virtual") else ""))
        n_base = 2 + (1 if (narrowed or sens_narrowed) else 0)
        has_pr = any(str(l9).startswith("[요소①") for l9 in lines)
        has_df = any(str(l9).startswith("[요소②") for l9 in lines)
        if len(lines) <= n_base and not (pat_lines or has_pr or has_df or ph_rows):
            raise HTTPException(400, "선택한 날짜 범위에는 추림된 이상이 없습니다 — "
                                     "기간 체크를 넓히거나 [선택 기간 패턴]을 먼저 확인하세요")
        if pat_lines:
            pat_lines[0] = "[요소③ 패턴 변화] " + pat_lines[0]
            lines = lines + [""] + pat_lines
        extra = ((f"분석 범위: 선택 {scope[0]}~{scope[-1]} {len(scope)}일 / " if narrowed else "")
                 + f"집계: 웨이퍼 {res.get('n_wafers')}장 · "
                 f"{(res.get('dates') or ['?'])[0]}~{last_day} · "
                 f"기준 골든 '{(res.get('ref') or {}).get('name')}'(정상 구간 통합 평균)")
        from io_recon_tcn import rca_causal as RC
        out = RC.narrate_stepdiff("\n".join(lines), extra)
        return out

    @app.post("/api/gcross")
    def api_gcross(payload: dict = Body(...)):
        """골든 크로스: 여러 폴더의 웨이퍼를 '판정 없이' N일 단위로 평균해 센서별로 겹쳐 본다.

        골든(양품 선별 평균)이 아니라 그냥 기간 평균 — 시간에 따라 평균 파형이
        어떻게 이동(교차)하는지 육안 확인용.
        """
        wfs = split_folders(payload.get("folder"))
        if not wfs:
            raise HTTPException(400, "폴더를 입력하세요")
        missing = [str(w9) for w9 in wfs if not w9.is_dir()]
        if missing:
            raise HTTPException(400, "폴더가 없습니다: " + " ; ".join(missing))
        bucket = max(1, min(31, int(payload.get("bucket_days") or 5)))
        per, n_all, n_sel, avail = select_stems_multi(S, wfs, payload.get("dsv"),
                                                      payload.get("recipe"), payload.get("months"),
                                                      payload.get("eqp"))
        if any(x[1] is not None for x in per) and n_sel == 0:
            raise HTTPException(400, f"필터를 통과한 파일이 없습니다 (전체 {n_all}장). "
                                     + _avail_text(avail))
        per_day_gx = int(payload.get("per_day") or 0)  # 0 = 전수(기본)
        day_map, n_seen, n_dated = _stems_by_day(per, per_day_gx)
        if n_dated == 0 and n_seen:
            raise HTTPException(400, f"날짜 태그가 있는 파일이 없습니다(전체 {n_seen}장). "
                                     "웨이퍼 분석 탭에서 이 폴더를 한 번 불러오면 태깅됩니다")
        n_total = sum(len(v) for v in day_map.values())
        print(f"[gcross] {len(wfs)}개 폴더 · {bucket}일 단위 — "
              f"{'전수' if not per_day_gx else f'일별 {per_day_gx}장 상한'} {n_total}장 "
              f"({len(day_map)}일, 하루치씩 스트리밍)", flush=True)
        if not n_total:
            raise HTTPException(400, "읽을 수 있는 웨이퍼가 없습니다")
        import datetime as _dt

        def _stem_info(rec):
            stem = str(rec["wafer_id"])
            d9 = rc9 = ""
            try:
                d9 = S._end_ts_and_date(stem)[1] or ""
            except Exception:                          # noqa: BLE001
                pass
            try:
                rc9 = str(S._recipe_display(S._RECIPE_TAGS.get(stem, "")) or "")
            except Exception:                          # noqa: BLE001
                pass
            if not d9 or not rc9:                      # 폴백: 판정 경로(모델 추론 비용 있음)
                try:
                    r9 = S._result_for(rec)
                    d9 = d9 or str(r9.get("end_ts") or "")[:10]
                    rc9 = rc9 or (S._recipe_display(r9.get("recipe") or "") or "")
                except Exception:                      # noqa: BLE001
                    pass
            return d9, rc9 or "unknown"

        # 레시피별 · 버킷별 증분 평균(합/개수 — 메모리 고정)
        acc = {}                                       # rc -> {"L":, "cols":, "sum":{bk:(L,S)}, "n":{bk:int}, "d0":date}
        L_DISP = 240
        def _gx_one(rec, d9):
            _du, rc9 = _stem_info(rec)
            day9 = _dt.date.fromisoformat(d9)
            arr9 = np.asarray(rec["sensors"], dtype=np.float64)
            cols9 = [str(c) for c in (rec.get("sensor_cols") or [])]
            a9 = acc.get(rc9)
            if a9 is None:
                g9 = groups.get(rc9)
                L9 = min(int(g9["len"]) if g9 else arr9.shape[0], L_DISP)
                a9 = acc[rc9] = {"L": L9, "cols": cols9, "sum": {}, "n": {}, "d0": day9}
            if cols9 != a9["cols"]:                    # 열 불일치 파일은 교집합 정렬
                keep = [c for c in a9["cols"] if c in cols9]
                arr9 = arr9[:, [cols9.index(c) for c in keep]]
                for bk9 in a9["sum"]:
                    a9["sum"][bk9] = a9["sum"][bk9][:, [a9["cols"].index(c) for c in keep]]
                a9["cols"] = keep
            if day9 < a9["d0"]:
                shift = (a9["d0"] - day9).days // bucket + 1
                a9["sum"] = {k9 + shift: v9 for k9, v9 in a9["sum"].items()}
                a9["n"] = {k9 + shift: v9 for k9, v9 in a9["n"].items()}
                a9["d0"] = a9["d0"] - _dt.timedelta(days=shift * bucket)
            bk9 = (day9 - a9["d0"]).days // bucket
            L9 = a9["L"]
            rs9 = np.column_stack([_resample(arr9[:, j9], L9)
                                   for j9 in range(arr9.shape[1])])
            if bk9 not in a9["sum"]:
                a9["sum"][bk9] = np.zeros((L9, len(a9["cols"])))
                a9["n"][bk9] = 0
            a9["sum"][bk9] = a9["sum"][bk9][:, :rs9.shape[1]] + rs9[:, :a9["sum"][bk9].shape[1]]                 if rs9.shape[1] != a9["sum"][bk9].shape[1] else a9["sum"][bk9] + rs9
            a9["n"][bk9] += 1
        done9 = 0
        for d9 in sorted(day_map):
            drecs = _load_day(day_map[d9])
            for rec in drecs:
                _gx_one(rec, d9)
            done9 += len(drecs)
            del drecs
            if done9 and done9 % 2000 < len(day_map[d9]):
                print(f"[gcross] 처리 {done9}/{n_total}", flush=True)
        out = {}
        for rc9, a9 in acc.items():
            g9 = groups.get(rc9)
            order = ([sn for sn in (g9["sensors"] if g9 else a9["cols"]) if sn in a9["cols"]]
                     or a9["cols"])
            bks = sorted(a9["sum"])
            buckets = []
            for bk9 in bks:
                s0 = a9["d0"] + _dt.timedelta(days=bk9 * bucket)
                s1 = s0 + _dt.timedelta(days=bucket - 1)
                mean9 = a9["sum"][bk9] / max(1, a9["n"][bk9])
                buckets.append({"label": f"{s0.isoformat()[5:]}~{s1.isoformat()[5:]}",
                                "n": a9["n"][bk9],
                                "m": {sn: [round(float(v), 3)
                                           for v in mean9[:, a9["cols"].index(sn)]]
                                      for sn in order}})
            out[rc9] = {"sensors": order, "len": a9["L"], "buckets": buckets,
                        "orig_len": int(g9["len"]) if g9 else None}
        n_used = sum(sum(a9["n"].values()) for a9 in acc.values())
        print(f"[gcross] 완료 — 레시피 {len(out)}종 · 웨이퍼 {n_used}장", flush=True)
        out_gx = {"recipes": out, "n_wafers": n_used, "bucket_days": bucket}
        WST["gx_out"] = out_gx                        # 새로고침 복원용
        return out_gx

    @app.post("/api/monthly")
    def api_monthly(payload: dict = Body(...)):
        table, eng, RC, n_days = _period_pack()
        wide = n_days > 70
        if wide:                                       # 컨텍스트 절약: 변곡 목록만 유지
            eng.pop("daily_ng_rate", None)
        stl = _step_trend_log(max_lines=40 if wide else 0)
        eng["step_trend_log"] = ("(일자별 표본 웨이퍼의 스텝×센서 트렌드 이상 — 표본 비율 근사)\n"
                                 + stl) if stl else "(표본에서 스텝 트렌드 이상 없음)"
        out = RC.narrate_monthly(table, json.dumps(eng, ensure_ascii=False, indent=1),
                                 horizon="월간")
        out["engine_findings"] = eng
        return out

    @app.post("/api/warning")
    def api_warning(payload: dict = Body(...)):
        last = int(payload.get("days") or 14)
        table, eng, RC, _nd = _period_pack(last_days=last)
        stl = _step_trend_log()
        eng["step_trend_log"] = ("(일자별 표본 웨이퍼의 스텝×센서 트렌드 이상 — 표본 비율 근사)\n"
                                 + stl) if stl else "(표본에서 스텝 트렌드 이상 없음)"
        out = RC.narrate_monthly(table, json.dumps(eng, ensure_ascii=False, indent=1),
                                 horizon=f"주간 조기경보(최근 {last}일 — 진행 중 전조와 예측 중심)")
        out["engine_findings"] = eng
        return out

    @app.get("/api/trend")
    def api_trend():
        rows = WST.get("trend") or []
        if not rows:
            raise HTTPException(400, "먼저 폴더를 불러오세요(스캔 필요)")
        days = {}
        for r2 in rows:
            d2 = days.setdefault(r2["date"], {"n": 0, "ng": 0, "z": {}})
            d2["n"] += 1
            d2["ng"] += 1 if r2["ng"] else 0
            for sn3, zv in r2["z"].items():
                acc2 = d2["z"].setdefault(sn3, [0.0, 0])
                acc2[0] += zv; acc2[1] += 1
        dates = sorted(k for k in days if k and k != "-")
        daily = [{"date": d3, "n": days[d3]["n"], "ng": days[d3]["ng"],
                  "rate": round(days[d3]["ng"] / max(1, days[d3]["n"]), 3)} for d3 in dates]
        # 변곡점: 불량률 증가폭 최대 날(표본 5장 이상)
        split = None; best = 0.0
        for a2 in range(1, len(daily)):
            if daily[a2]["n"] < 5 or daily[a2 - 1]["n"] < 5:
                continue
            jump = daily[a2]["rate"] - daily[a2 - 1]["rate"]
            if jump > best:
                best = jump; split = daily[a2]["date"]
        # 전/후 센서 평균 z 델타 랭킹
        ranking = []
        if split:
            pre = {}; post = {}
            for d3 in dates:
                tgt = post if d3 >= split else pre
                for sn3, (sm, ct) in days[d3]["z"].items():
                    acc3 = tgt.setdefault(sn3, [0.0, 0])
                    acc3[0] += sm; acc3[1] += ct
            for sn3 in set(pre) & set(post):
                pm = pre[sn3][0] / max(1, pre[sn3][1])
                qm = post[sn3][0] / max(1, post[sn3][1])
                ranking.append({"sensor": sn3, "pre": round(pm, 2),
                                "post": round(qm, 2), "delta": round(qm - pm, 2)})
            ranking.sort(key=lambda x: -x["delta"])
        return {"daily": daily, "split": split, "ranking": ranking[:15],
                "note": "delta = 변곡일 이후 평균 이탈(σ) − 이전 평균 — 불량 급증에 앞서/함께 커진 센서 순"}

    @app.get("/api/wafer")
    def api_wafer(i: int):
        if WST["folder"] is None or not (0 <= i < len(WST["stems"])):
            raise HTTPException(400, "먼저 폴더를 불러오세요")
        recs = _load_by_idx([i])
        if not recs:
            raise HTTPException(404, "웨이퍼 파일을 다시 읽지 못했습니다(삭제/이동?)")
        return _wafer_payload(S, groups, recs[0])

    # ── 골든 비교 탭: 기간 평균 골든(설비 대표+DSV별) vs 실웨이퍼 형상 판정 ──
    #   지표: 스텝별 offset 제거 후 σ밴드 초과 면적(k 격자 사전계산) + 길이비 분리.
    #   k·α·β 는 캐시 밖(즉시 재판정) — 사전 스무딩 창 w 만 캐시 키(재스캔 필요).
    GVD = {}                                           # 골든 비교 탭 상태(독립)
    GVJOB = {"state": "idle", "progress": "", "error": None, "cancel": False}
    GVX = {"gen": 0, "lock": threading.Lock()}         # 사용자 요청 세대(복원 경합 가드)
    # GV_KGRID 는 모듈 상수(워커 공유)

    def _gv_load(folder9, stems=None):
        """골든 비교 전용 로더: 원 센서 + 차분 가상센서(dT_/dP_) 열을 붙여 돌려준다.
        반환은 S._load_aligned 와 동일한 4튜플 — 가상센서를 고르면 KeyError 나던 문제."""
        recs9, io9, sen9, cov9 = S._load_aligned(Path(folder9), stems=stems)
        try:
            _augment_virtual(recs9)
        except Exception as e9:                        # noqa: BLE001
            print(f"[gv] 가상센서 계산 실패(원 센서만 사용): {e9}", flush=True)
        return recs9, io9, sen9, cov9

    def _gv_read_anchors(path9):
        try:
            from pattern_hunter import read_anchors as _ra9
        except Exception:                              # noqa: BLE001
            from io_recon_tcn.pattern_hunter import read_anchors as _ra9
        return _ra9(str(path9))

    def _gv_anchor_hit(meta9, date9, anchors9):
        """웨이퍼가 불량조치 CSV 보고분인가 — 설비 부분일치·DSV(있으면) 일치·일자 일치."""
        for a9 in anchors9:
            ae9 = str(a9.get("eqp") or "").lower()
            if ae9 and ae9 not in meta9["eqp"] and meta9["eqp"] not in ae9:
                continue
            ad9 = str(a9.get("dsv") or "").lower()
            if ad9 and ad9 != meta9["dsv"]:
                continue
            if str(a9.get("date") or "")[:10].replace(".", "-").replace("/", "-") \
                    == date9:
                return True
        return False

    def _gv_session_save(part9, val9):
        """마지막 세션(목록/골든/판정 입력)을 앱 상태 파일에 — 서버 재시작 복원용."""
        try:
            st9 = (_state_load().get("gv_session") or {})
            st9[part9] = val9
            _state_save("gv_session", st9)
        except Exception as e9:                        # noqa: BLE001
            print(f"[gv] 세션 저장 실패(무시): {e9}", flush=True)

    def _gv_preempt_restore():
        """자동 복원(세션 재적재) 작업이 돌고 있으면 중단하고 사용자 요청을 우선한다.
        사용자가 직접 시작한 작업이면 건드리지 않는다(409 로 이어짐)."""
        if GVJOB["state"] == "running" and GVD.get("restore_job"):
            GVJOB["cancel"] = True
            for _ in range(75):
                if GVJOB["state"] != "running":
                    break
                time.sleep(0.2)
            GVD["restore_job"] = False

    @app.post("/api/gv_list")
    def api_gv_list(payload: dict = Body(...)):
        GVX["gen"] += 1                                # 진행 중인 자동 복원은 무효
        if GVJOB["state"] == "running":                # 새 목록 = 진행 중 작업 중단(사용자 우선)
            GVJOB["cancel"] = True
            for _ in range(150):
                if GVJOB["state"] != "running":
                    break
                time.sleep(0.2)
            GVD["restore_job"] = False
        with GVX["lock"]:
            out9 = _gv_do_list(payload)
        if not payload.get("from_viz"):
            _gv_session_save("list", {k9: payload.get(k9) for k9 in
                                      ("folder", "eqp", "dsv", "recipe", "months",
                                       "csv", "store")})
            _gv_session_save("build", None); _gv_session_save("run", None)
        return out9

    def _gv_do_list(payload):
        """골든 비교: 폴더+필터로 웨이퍼 목록 구성(+불량조치 CSV) — 분석자료와 독립.

        from_viz=True 면 [데이터 시각화] 탭이 이미 불러온 목록을 그대로 공유.
        """
        if payload.get("from_viz"):
            if not VZD.get("id2src"):
                raise HTTPException(400, "데이터 시각화 탭에 불러온 목록이 없습니다")
            anchors9, csv9 = [], str(payload.get("csv") or "").strip()
            if csv9:
                if not Path(csv9).is_file():
                    raise HTTPException(400, f"불량조치내역 CSV 가 없습니다: {csv9}")
                anchors9 = _gv_read_anchors(csv9)
            GVD.clear()
            GVD.update(id2src=dict(VZD["id2src"]), id2date=dict(VZD["id2date"]),
                       id2meta=dict(VZD["id2meta"]),
                       id2stem={k9: k9 for k9 in VZD["id2src"]},
                       anchors=anchors9, csv=csv9,
                       store=str(VZD.get("store") or ""),
                       sensors=list(VZD.get("sensors") or []))
            if not GVD["store"]:
                GVD["store"] = str(Path(next(iter(GVD["id2src"].values()))).parent / "gv_store")
                Path(GVD["store"]).mkdir(parents=True, exist_ok=True)
            return _gv_list_summary(len(anchors9))
        wfs9 = split_folders(payload.get("folder"))
        if not wfs9:
            raise HTTPException(400, "폴더를 입력하세요")
        missing9 = [str(w9) for w9 in wfs9 if not w9.is_dir()]
        if missing9:
            raise HTTPException(400, "폴더가 없습니다: " + " ; ".join(missing9))
        store_v9 = str(payload.get("store") or "").strip()
        if store_v9:
            store9 = _gstore_dir(store_v9, create=True)
        else:                                          # 미입력 → 상태값, 없으면 데이터 폴더 안
            st_v9 = str((_state_load().get("vz") or {}).get("store") or "")
            if st_v9 and Path(st_v9).is_dir():
                store9 = Path(st_v9)
            else:
                store9 = Path(wfs9[0]) / "gv_store"
                store9.mkdir(parents=True, exist_ok=True)
        cache_set9 = (_state_load().get("sd") or {}).get("cache")
        _install_meta_cache(S, cache_set9 or store9)
        _install_parse_cache(S, store_hint=str(store9))
        lc9 = _install_listing_cache(S)                # 폴더 목록 메모이즈(묶음·웨이퍼마다 전량 rglob 방지)
        if lc9 is not None:
            lc9.clear()                                # 목록 불러오기 시점엔 새로 훑는다
        per9, n_all9, n_sel9, avail9 = select_stems_multi(
            S, wfs9, payload.get("dsv"), payload.get("recipe"),
            payload.get("months"), payload.get("eqp"))
        if any(x9[1] is not None for x9 in per9) and n_sel9 == 0:
            raise HTTPException(400, f"필터를 통과한 파일이 없습니다 (전체 {n_all9}장). "
                                     + _avail_text(avail9))
        day_map9, n_seen9, n_dated9 = _stems_by_day(per9, 0)
        if n_dated9 == 0:
            raise HTTPException(400, f"날짜 태그가 있는 파일이 없습니다(전체 {n_seen9}장). "
                                     "웨이퍼 분석 탭에서 이 폴더를 한 번 불러오면 태깅됩니다")
        anchors9, csv9 = [], str(payload.get("csv") or "").strip()
        if csv9:
            if not Path(csv9).is_file():
                raise HTTPException(400, f"불량조치내역 CSV 가 없습니다: {csv9}")
            try:
                anchors9 = _gv_read_anchors(csv9)
            except Exception as e9:                    # noqa: BLE001
                raise HTTPException(400, f"CSV 판독 실패: {e9}")
        sens9 = []
        for _d9, prs9 in sorted(day_map9.items()):     # 센서 목록: 첫 파일에서 확인
            if not prs9:
                continue
            f9x, st9x = prs9[0]
            try:
                rp9, _a, _b, _c = _gv_load(Path(f9x), stems=[st9x])
                if rp9:
                    sens9 = [str(c9) for c9 in (rp9[0].get("sensor_cols") or [])]
                del rp9
            except Exception:                          # noqa: BLE001
                pass
            break
        # ★폴더가 여러 개면 같은 파일 이름(스템)이 겹칠 수 있다(DSV 폴더별 동일
        #   명명) — 스템을 그대로 키로 쓰면 뒤 폴더가 앞 폴더를 덮어써 수만 장이
        #   한 폴더 분량으로 줄어든다. 충돌 시 '스템@폴더명' 으로 구분.
        cnt9 = {}
        for prs9 in day_map9.values():
            for _f9, st9 in prs9:
                cnt9[st9] = cnt9.get(st9, 0) + 1
        id2src9, id2date9, id2meta9, id2stem9 = {}, {}, {}, {}
        for d9, prs9 in day_map9.items():
            for f9, st9 in prs9:
                wid9 = st9 if cnt9[st9] == 1 else f"{st9}@{Path(f9).name}"
                id2src9[wid9] = f9; id2date9[wid9] = d9
                id2meta9[wid9] = _meta_of(S, st9); id2stem9[wid9] = st9
        n_coll9 = sum(1 for v9 in cnt9.values() if v9 > 1)
        fol_stat9 = []
        for fol9, selx9 in per9:
            n_f9 = sum(1 for f9 in id2src9.values() if f9 == str(fol9))
            fol_stat9.append({"folder": str(fol9), "n": n_f9})
        GVD.clear()
        GVD.update(id2src=id2src9, id2date=id2date9, id2meta=id2meta9,
                   id2stem=id2stem9, anchors=anchors9, csv=csv9,
                   store=str(store9), sensors=sens9,
                   diag={"n_seen": int(n_seen9), "n_dated": int(n_dated9),
                         "n_untagged": int(n_seen9 - n_dated9),
                         "n_collision": int(n_coll9), "folders": fol_stat9})
        GVD["folder"] = str(payload.get("folder") or "")
        return _gv_list_summary(len(anchors9))

    def _gv_list_summary(csv_n9):
        avail9 = {"eqp": {}, "dsv": {}, "recipe": {}}  # 필터 유무와 무관하게 직접 집계
        for i9, m9 in GVD["id2meta"].items():
            for k9, v9 in (("eqp", m9["eqp"]), ("dsv", m9["dsv"]),
                           ("recipe", m9["recipe"])):
                v9 = v9 or "(없음)"
                avail9[k9][v9] = avail9[k9].get(v9, 0) + 1
        return {"total": len(GVD["id2date"]),
                "dates": sorted(set(GVD["id2date"].values())),
                "sensors": list(GVD.get("sensors") or []),
                "folder": GVD.get("folder") or "",
                "diag": GVD.get("diag") or {},
                "avail": {k9: sorted(d9, key=_gv_natkey)
                          for k9, d9 in avail9.items()},
                "csv_n": int(csv_n9)}

    def _gv_natkey(v9):
        """자연 정렬 키 — 'dsv10' 이 'dsv2' 뒤에 오게(숫자 조각은 수치로)."""
        import re as _re9
        return [(0, int(t9)) if t9.isdigit() else (1, t9.lower())
                for t9 in _re9.split(r"(\d+)", str(v9)) if t9 != ""]

    def _gv_canon(runsigs9):
        """웨이퍼별 [(step,len),...] 목록 → 최빈 시퀀스 + 위치별 중앙값 길이 캐논."""
        from collections import Counter
        seqs9 = Counter(tuple(u9 for u9, _l9 in rs9) for rs9 in runsigs9)
        seq9 = list(seqs9.most_common(1)[0][0])
        lens9 = [[] for _ in seq9]
        for rs9 in runsigs9:
            if tuple(u9 for u9, _l9 in rs9) != tuple(seq9):
                continue
            for i9, (_u9, l9) in enumerate(rs9):
                lens9[i9].append(l9)
        med9 = [max(2, int(np.median(v9))) for v9 in lens9]
        total9 = sum(med9)
        scale9 = min(1.0, 4000.0 / max(1, total9))     # 격자 상한 4000pt
        canon9, pos9 = [], 0
        for u9, l9 in zip(seq9, med9):
            ln9 = max(2, int(round(l9 * scale9)))
            canon9.append((int(u9), pos9, ln9))
            pos9 += ln9
        return canon9, pos9, med9

    def _gv_sensor_mat(rec9, sens9):
        """레코드 → 요청 센서 순서의 (T,S) 행렬(모듈 코어 위임 · 결측 통계는 스캔 워커가 집계)."""
        return _gv_sensor_mat_core(rec9, sens9, None)

    def _gv_prefetch(jobs9):
        """[(파일, [스템,...]), ...] → (파일, 스템목록, recs) 를 제출 순서대로 산출.
        로드(파싱·npz 읽기)를 스레드 N개가 앞서 수행 — 소비 순서는 고정이라
        합계·판정 결과가 직렬과 동일하다. IO_GV_WORKERS=1 이면 직렬."""
        # 기본 직렬(1): 파싱이 GIL 바운드라 스레드 선읽기는 실측 오히려 느림
        #   (122장 8.4s→10.9s). 디스크가 느린 환경에서만 IO_GV_WORKERS=2~3 시험.
        try:
            w9 = int(os.environ.get("IO_GV_WORKERS") or 1)
        except (TypeError, ValueError):
            w9 = 1
        w9 = max(1, min(8, w9))
        if w9 == 1 or len(jobs9) <= 1:
            for f9, sts9 in jobs9:
                yield f9, sts9, _gv_load(Path(f9), stems=sts9)[0]
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=w9) as ex9:
            futs9 = []
            it9 = iter(jobs9)
            for _ in range(w9 * 2):                   # 앞서 읽는 깊이 = 워커×2
                try:
                    f9, sts9 = next(it9)
                except StopIteration:
                    break
                futs9.append((f9, sts9, ex9.submit(
                    lambda f9=f9, sts9=sts9: _gv_load(Path(f9), stems=sts9)[0])))
            while futs9:
                f9, sts9, fu9 = futs9.pop(0)
                try:
                    f29, sts29 = next(it9)
                    futs9.append((f29, sts29, ex9.submit(
                        lambda f9=f29, sts9=sts29: _gv_load(Path(f9), stems=sts9)[0])))
                except StopIteration:
                    pass
                yield f9, sts9, fu9.result()

    def _gv_runsig_cache():
        c9 = GVD.get("runsig_cache")
        if c9 is None:
            c9 = {}
            try:
                fp9 = Path(GVD["store"]) / "gv_runsig.json"
                if fp9.is_file():
                    c9 = json.loads(fp9.read_text(encoding="utf-8"))
            except Exception:                          # noqa: BLE001
                pass
            GVD["runsig_cache"] = c9
        return c9

    def _gv_runsig_save():
        try:
            (Path(GVD["store"]) / "gv_runsig.json").write_text(
                json.dumps(GVD.get("runsig_cache") or {}), encoding="utf-8")
        except Exception as e9:                        # noqa: BLE001
            print(f"[gv] 스텝 구성 캐시 저장 실패(무시): {e9}", flush=True)

    def _gv_build_worker(d0, d1, sens_sel9, per_dsv9=None, restore9=None):
        """골든 빌드(백그라운드): CSV 보고분 제외 기간 평균 — 대표+DSV별, mean+σ.

        per_dsv9={dsv:{d0,d1}} — DSV 별 골든 기간(비면 공통 d0/d1). 설비 대표
        골든은 각 DSV 가 자기 기간으로 고른 웨이퍼들의 합집합 평균.
        """
        anchors9 = GVD.get("anchors") or []
        per_dsv9 = per_dsv9 or {}
        keys9 = {}                                     # (eqp,recipe) → [(id, dsv)]
        excluded9 = 0
        for i9, d9 in GVD["id2date"].items():
            m9 = GVD["id2meta"][i9]
            if not _gv_facet_ok(m9):
                continue
            rng9 = per_dsv9.get(m9["dsv"]) or {}
            d0x9 = str(rng9.get("d0") or "").strip() or d0
            d1x9 = str(rng9.get("d1") or "").strip() or d1
            if (d0x9 and d9 < d0x9) or (d1x9 and d9 > d1x9):
                continue
            if anchors9 and _gv_anchor_hit(m9, d9, anchors9):
                excluded9 += 1
                continue
            keys9.setdefault((m9["eqp"], m9["recipe"]), []).append((i9, m9["dsv"]))
        if not keys9:
            raise RuntimeError("기간 내 웨이퍼가 없습니다(전부 CSV 보고분 제외?)")
        goldens9 = []
        auto_ids9 = []
        dsv_counts9 = {}
        n_done9 = 0
        n_tot9 = sum(len(v9) for v9 in keys9.values())
        for (eq9, rc9), items9 in sorted(keys9.items()):
            if GVJOB["cancel"]:
                raise RuntimeError("중단됨")
            # 1패스: 스텝 런 서명 수집(캐논 확정용) — 파일 그룹 스트리밍
            runsig9, by_f9 = {}, {}
            for i9, _dv9 in items9:
                f9 = GVD["id2src"].get(i9)
                if f9:
                    by_f9.setdefault(f9, []).append(i9)
            canon9 = L9 = raw_med9 = None

            def _iter9(label9, only9=None):
                """이 (설비,레시피) 묶음의 레코드 스트리밍(선읽기) — 패스마다 재사용.
                only9 가 주어지면 그 id 들만 읽는다(스텝 구성 캐시 미적중분 등)."""
                st2id9 = GVD.get("id2stem") or {}
                jobs9, backs9 = [], {}
                for f9, ids_f9 in by_f9.items():
                    ids_u9 = [i9 for i9 in ids_f9 if only9 is None or i9 in only9]
                    if not ids_u9:
                        continue
                    back9 = {st2id9.get(i9, i9): i9 for i9 in ids_u9}
                    backs9[f9] = back9
                    sts9 = list(back9)
                    for c09 in range(0, len(sts9), 100):
                        jobs9.append((f9, sts9[c09:c09 + 100]))
                for f9, _sts9, recs9 in _gv_prefetch(jobs9):
                    if GVJOB["cancel"]:
                        raise RuntimeError("중단됨")
                    back9 = backs9[f9]
                    for rec9 in recs9:
                        rid9 = back9.get(str(rec9.get("wafer_id") or ""),
                                         str(rec9.get("wafer_id") or ""))
                        GVJOB["progress"] = f"골든 산정 중… {eq9} {label9}"
                        yield rid9, rec9
                    del recs9

            tp9 = time.time()
            rc_cache9 = _gv_runsig_cache()
            def _fkey9(i9):
                try:
                    stt9 = Path(GVD["id2src"][i9]).stat()
                    return f"{stt9.st_mtime_ns}|{stt9.st_size}"
                except Exception:                      # noqa: BLE001
                    return "?"
            need9 = set()
            for i9, _dv9 in items9:
                ent9 = rc_cache9.get(i9)
                if ent9 and ent9.get("k") == _fkey9(i9):
                    runsig9[i9] = [tuple(x9) for x9 in ent9["r"]]
                else:
                    need9.add(i9)
            n_hit9 = len(runsig9)
            for rid9, rec9 in _iter9("스텝 구성 확인", need9 if n_hit9 else None):
                stw9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
                runsig9[rid9] = [(u9, e9 - s9)
                                 for u9, s9, e9 in _step_runs(stw9)]
                rc_cache9[rid9] = {"k": _fkey9(rid9),
                                   "r": [[int(u9), int(l9)] for u9, l9 in runsig9[rid9]]}
            _gv_runsig_save()
            canon9, L9, raw_med9 = _gv_canon(list(runsig9.values()))
            print(f"[gv] 골든 {eq9}: 1패스(스텝 구성) {len(runsig9)}장 "
                  f"(캐시 {n_hit9}장) {time.time() - tp9:.1f}s", flush=True); tp9 = time.time()

            def _warp9(rec9):
                arr9, stw9 = _gv_sensor_mat(rec9, sens_sel9)
                if arr9 is None:
                    return None
                return _warp_to_canon(arr9, stw9, canon9, L9)

            def _accum9(skip9, label9):
                acc9 = {}                              # dsv|'' → [sum,sumsq,cnt,n,raw_lens]
                for rid9, rec9 in _iter9(label9):
                    if rid9 in skip9:
                        continue
                    wv9 = _warp9(rec9)
                    if wv9 is None:
                        continue
                    dv9 = GVD["id2meta"].get(rid9, {}).get("dsv", "")
                    for gk9 in ("", dv9):              # ''=설비 대표에 항상 누적
                        a9 = acc9.setdefault(
                            gk9, [np.zeros((L9, len(sens_sel9))),
                                  np.zeros((L9, len(sens_sel9))),
                                  np.zeros((L9, len(sens_sel9))), 0, []])
                        fin9 = np.isfinite(wv9)
                        a9[0] += np.where(fin9, wv9, 0.0)
                        a9[1] += np.where(fin9, wv9 * wv9, 0.0)
                        a9[2] += fin9
                        a9[3] += 1
                        a9[4].append([l9 for _u9, l9 in runsig9[rid9]])
                return acc9

            def _gold9(a9, gk9, extra9=None):
                with np.errstate(all="ignore"):
                    mean9 = a9[0] / np.maximum(1, a9[2])
                    var9 = a9[1] / np.maximum(1, a9[2]) - mean9 * mean9
                    std9 = np.sqrt(np.maximum(0.0, var9))
                mean9[a9[2] == 0] = np.nan
                rl9 = [v9 for v9 in a9[4] if len(v9) == len(canon9)]
                rlm9 = ([int(np.median([r9[j9] for r9 in rl9]))
                         for j9 in range(len(canon9))] if rl9 else raw_med9)
                g9 = {"eqp": eq9, "recipe": rc9, "dsv": gk9 or None,
                      "n": int(a9[3]), "L": int(L9),
                      "canon": [[int(u9), int(p9), int(l9)]
                                for u9, p9, l9 in canon9],
                      "raw_len_med": rlm9, "sensors": list(sens_sel9),
                      "mean": {s9: [None if not np.isfinite(v9) else
                                    round(float(v9), 5)
                                    for v9 in mean9[:, j9]]
                               for j9, s9 in enumerate(sens_sel9)},
                      "std": {s9: [round(float(v9), 5)
                                   for v9 in std9[:, j9]]
                              for j9, s9 in enumerate(sens_sel9)},
                      "small": bool(a9[3] < 10)}
                if extra9:
                    g9.update(extra9)
                return g9

            # 2패스: 1차 골든(전량 평균 — 미보고 불량이 섞이면 σ 가 부풀어
            #   판정이 무뎌진다) → 3패스에서 자체 정화로 걸러낸다
            acc_p9 = _accum9(set(), "1차 평균")
            print(f"[gv] 골든 {eq9}: 2패스(1차 평균) {time.time() - tp9:.1f}s",
                  flush=True); tp9 = time.time()
            dsv_counts9[f"{eq9}|{rc9}"] = {gk9: int(a9[3]) for gk9, a9 in acc_p9.items()
                                          if gk9}
            prelim9 = {gk9: _gold9(a9, gk9) for gk9, a9 in acc_p9.items()
                       if a9[3] >= 3}
            # 3패스: 자체 정화 — 1차 골든 대비 웨이퍼별 최대 이탈(A, k=0)을 모아
            #   중앙값+5·MAD 초과를 골든 평균에서 제외(CSV 미보고 불량 방어)
            #   ★속도: 이탈 상위 K 장의 (워핑값·원 길이)를 함께 보관해 두고, 제외
            #   집합이 그 안에 들면 4패스(재평균) 대신 1차 합계에서 기여분을 뺀다.
            #   합·제곱합·개수 뺄셈이라 결과는 재평균과 동일(부동소수 반올림 수준).
            Aw9 = {}                                   # dv → {rid: A}
            import heapq as _hq9
            n_items9 = len(items9)
            K9 = max(50, int(np.ceil(0.05 * n_items9)))
            per_wafer_b9 = 2 * int(L9) * len(sens_sel9) * 8
            K9 = max(10, min(K9, int(300e6 // max(1, per_wafer_b9))))
            keep9 = []                                 # 최소힙 [(A, rid)] — 상위 K 유지
            kept9 = {}                                 # rid → (wv, raw_lens, dv)
            for rid9, rec9 in _iter9("자체 정화 검사"):
                dv9 = GVD["id2meta"].get(rid9, {}).get("dsv", "")
                g9 = prelim9.get(dv9) or prelim9.get("")
                if g9 is None:
                    continue
                wv9 = _warp9(rec9)
                if wv9 is None:
                    continue
                # 자체 노이즈 σ 기반(k=3): 골든 σ 오염 무관 + 저진폭 센서의
                # 노이즈 바닥이 max 를 지배하는 문제 제거(다중 센서 필수 조건)
                A9 = _gv_metrics(wv9, g9, sens_sel9, noise_sigma=True)
                with np.errstate(all="ignore"):
                    a_max9 = float(np.nanmax(A9[..., 6]))
                Aw9.setdefault(dv9, {})[rid9] = a_max9
                if np.isfinite(a_max9):
                    if len(keep9) < K9:
                        _hq9.heappush(keep9, (a_max9, rid9))
                        kept9[rid9] = (wv9, [l9 for _u9, l9 in runsig9[rid9]], dv9)
                    elif a_max9 > keep9[0][0]:
                        _a0, r0 = _hq9.heapreplace(keep9, (a_max9, rid9))
                        kept9.pop(r0, None)
                        kept9[rid9] = (wv9, [l9 for _u9, l9 in runsig9[rid9]], dv9)
            print(f"[gv] 골든 {eq9}: 3패스(자체 정화 검사) {time.time() - tp9:.1f}s",
                  flush=True); tp9 = time.time()
            # 컷은 DSV 그룹별 상대 기준 — 소수 오염이 1차 평균을 밀면 그 에코가
            # 그룹 전원에 얹히므로(중앙값도 함께 상승) 그룹 내 상대 컷만 안전.
            # 그룹이 작으면(10장 미만) 설비 전체 분포로 폴백.
            all_v9 = np.asarray([v9 for d9 in Aw9.values()
                                 for v9 in d9.values() if np.isfinite(v9)])
            auto_ex9 = set()
            for dv9, grp9 in sorted(Aw9.items()):
                vals9 = np.asarray([v9 for v9 in grp9.values()
                                    if np.isfinite(v9)])
                base9 = vals9 if len(vals9) >= 10 else all_v9
                if len(base9) < 10:
                    continue
                med9 = float(np.median(base9))
                mad9 = float(np.median(np.abs(base9 - med9)))
                cut9 = max(med9 + 5.0 * max(mad9, med9 * 0.1),
                           med9 * 2.0, 0.01)
                bad9 = {r9 for r9, v9 in grp9.items()
                        if np.isfinite(v9) and v9 > cut9}
                auto_ex9 |= bad9
                top9 = sorted(grp9.items(), key=lambda x9: -x9[1])[:4]
                print(f"[gv] 자체 정화({eq9}/{dv9 or '-'}): 중앙값 {med9:.4f} "
                      f"· 컷 {cut9:.4f} · 제외 {len(bad9)}장 · 상위 "
                      + " ".join(f"{r9[-18:]}:{v9:.3f}" for r9, v9 in top9),
                      flush=True)
            auto_ex9 -= set(restore9 or [])           # 사용자가 정상 확정한 웨이퍼
            auto_ids9 += sorted(auto_ex9)
            # 4패스 대체: 제외 집합이 보관분 안이면 1차 합계에서 기여분 뺄셈
            acc9 = None
            if auto_ex9 and auto_ex9 <= set(kept9):
                acc9 = acc_p9
                for r9 in auto_ex9:
                    wv9, rl9, dv9 = kept9[r9]
                    fin9 = np.isfinite(wv9)
                    for gk9 in ("", dv9):
                        a9 = acc9.get(gk9)
                        if a9 is None:
                            continue
                        a9[0] -= np.where(fin9, wv9, 0.0)
                        a9[1] -= np.where(fin9, wv9 * wv9, 0.0)
                        a9[2] -= fin9
                        a9[3] -= 1
                        try:
                            a9[4].remove(rl9)
                        except ValueError:
                            pass
                print(f"[gv] 골든 {eq9}: 4패스 생략 — 제외 {len(auto_ex9)}장 기여분 "
                      f"뺄셈으로 재평균(보관 {len(kept9)}장)", flush=True)
            elif auto_ex9:
                acc9 = _accum9(auto_ex9, "정화 반영 재평균")
            del kept9
            print(f"[gv] 골든 {eq9}: 4패스(재평균) {time.time() - tp9:.1f}s"
                  f"{'' if auto_ex9 else ' (생략)'}", flush=True)
            src9 = acc9 if acc9 is not None else None
            if src9 is None:                           # 정화 대상 없음 → 1차 그대로
                for gk9 in sorted(prelim9):
                    goldens9.append(dict(prelim9[gk9], auto_excluded=0))
            else:
                for gk9, a9 in sorted(src9.items()):
                    if a9[3] < 3:
                        continue
                    goldens9.append(_gold9(a9, gk9,
                                           {"auto_excluded": len(auto_ex9)}))
            n_done9 += len(items9)
        sig9 = hashlib.md5(
            f"{d0}|{d1}|{json.dumps(per_dsv9, sort_keys=True)}|"
            f"{len(GVD['id2date'])}|{GVD.get('csv')}|"
            f"{'|'.join(sorted(sens_sel9))}|{excluded9}".encode("utf-8")
        ).hexdigest()[:12]
        build9 = {"avgwave_gv_golden": 1, "d0": d0, "d1": d1, "sig": sig9,
                  "excluded_n": excluded9, "auto_excluded_ids": auto_ids9,
                  "dsv_counts": dsv_counts9, "goldens": goldens9}
        try:
            fp9 = Path(GVD["store"]) / f"gv_golden_{sig9}.json"
            fp9.write_text(json.dumps(build9, ensure_ascii=False),
                           encoding="utf-8")
            build9["file"] = fp9.name
        except Exception as e9:                        # noqa: BLE001
            print(f"[gv] 골든 저장 실패(메모리로만 진행): {e9}", flush=True)
        GVD["build"] = build9

    def _gv_facet_ok(m9):
        """체크된 설비/DSV/레시피만 대상 — 비어 있으면(=전체) 필터 없음."""
        f9 = GVD.get("facet") or {}
        for k9 in ("eqp", "dsv", "recipe"):
            sel9 = {str(v9).lower() for v9 in (f9.get(k9) or [])}
            if sel9 and (m9.get(k9) or "(없음)") not in sel9:
                return False
        return True

    def _gv_set_facet(payload):
        f9 = payload.get("facet") or {}
        GVD["facet"] = {k9: [str(v9) for v9 in (f9.get(k9) or [])]
                        for k9 in ("eqp", "dsv", "recipe")}

    def _gv_pick_golden(meta9):
        """웨이퍼 메타 → (판정 기준 골든, 가이드 골든) — 모듈 코어 위임(GVD build/gref)."""
        return _gv_pick_golden_core(meta9, (GVD.get("build") or {}).get("goldens") or [],
                                    GVD.get("gref") or "dsv")

    def _gv_metrics(wv9, g9, sens9, w9=0, noise_sigma=False):
        return _gv_metrics_core(wv9, g9, sens9, w9, noise_sigma)

    def _gv_cache_dir():
        b9 = GVD.get("build") or {}
        if not b9 or not GVD.get("store"):
            return None
        cache9 = (_state_load().get("sd") or {}).get("cache")
        key9 = hashlib.md5(                            # 센서 집합은 키 밖 — npz 안에 센서명
            f"{b9['sig']}|w{GVD.get('w') or 0}|g{GVD.get('gref') or 'dsv'}|"
            f"sens-any|{CALC_VER}".encode("utf-8")).hexdigest()[:12]
        try:
            d9 = _cache_root(cache9, GVD["store"]) / "gv_cache" / key9
            d9.mkdir(parents=True, exist_ok=True)
            return d9
        except Exception as e9:                        # noqa: BLE001
            print(f"[gv] 캐시 폴더를 쓸 수 없어 캐시 없이 진행합니다: {e9}", flush=True)
            return None

    def _gv_day_sig(ids9):
        h9 = hashlib.md5()
        for w9 in ids9:
            f9 = (GVD.get("id2src") or {}).get(w9) or ""
            try:
                stt9 = Path(f9).stat()
                h9.update(f"{w9}|{stt9.st_mtime_ns}|{stt9.st_size}".encode())
            except Exception:                          # noqa: BLE001
                h9.update(f"{w9}|?".encode())
        return h9.hexdigest()[:16]

    def _gv_wave_root():
        try:
            if not GVD.get("store"):
                return None
            cache9 = (_state_load().get("sd") or {}).get("cache")
            d9 = _cache_root(cache9, GVD["store"]) / "gv_wave"
            d9.mkdir(parents=True, exist_ok=True)
            return d9
        except Exception as e9:                        # noqa: BLE001
            print(f"[gv] 파형 캐시 폴더를 쓸 수 없습니다(캐시 없이 진행): {e9}", flush=True)
            return None

    def _gv_wave_have(wroot9, day9, ids9, sens9, fk_of9):
        """그날 웨이퍼 전부의 정합 파형(요청 센서 ∪ 참조)이 캐시에 있는가 — 캐논 폴더 전체를 합쳐 확인."""
        if wroot9 is None:
            return False
        need9 = set(str(x9) for x9 in ids9)
        want9 = list(sens9) + [s9 for s9 in GV_REF_SENSORS if s9 not in sens9]
        try:
            for sub9 in Path(wroot9).iterdir():
                fp9 = sub9 / f"{day9}.npz"
                if not sub9.is_dir() or not fp9.is_file():
                    continue
                with np.load(fp9, allow_pickle=True) as z9:
                    if str(z9["ver"]) != GV_WAVE_VER:
                        continue
                    sens_c9 = set(str(v9) for v9 in z9["sens"])
                    if not all(s9 in sens_c9 for s9 in want9):
                        continue
                    for w9, fk9 in zip([str(v9) for v9 in z9["ids"]], [str(v9) for v9 in z9["fk"]]):
                        if w9 in need9 and fk_of9.get(w9) == fk9:
                            need9.discard(w9)
                if not need9:
                    return True
        except Exception:                              # noqa: BLE001
            return False
        return not need9

    def _gv_scan_worker(d0, d1):
        """판정 스캔(백그라운드): 일별로 로드→정합→k격자 지표+형상 특징→npz 캐시(+정합 파형 캐시).
        계산이 필요한 날만 모아 프로세스 풀(IO_GV_PROCS)로 돌리고, 결과 병합·저장은 부모가 한다."""
        sens9 = GVD.get("run_sens") or []
        cdir9 = _gv_cache_dir()
        wroot9 = _gv_wave_root()
        day_ids9 = {}
        for i9, dd9 in GVD["id2date"].items():
            if (d0 and dd9 < d0) or (d1 and dd9 > d1):
                continue
            if not _gv_facet_ok(GVD["id2meta"].get(i9) or {}):
                continue
            day_ids9.setdefault(dd9, []).append(i9)
        days9 = sorted(day_ids9)
        scores9 = GVD.setdefault("scores", {})
        GVD["sensor_missing"] = {}
        ts9 = time.time(); n_new9 = 0; n_slice9 = 0
        todo9 = []
        st2id9 = GVD.get("id2stem") or {}
        for di9, day9 in enumerate(days9):
            if GVJOB["cancel"]:
                raise RuntimeError("중단됨")
            ids9 = sorted(day_ids9[day9])
            sig9 = _gv_day_sig(ids9)
            fp9 = (cdir9 / f"{day9}.npz") if cdir9 else None
            if day9 in scores9 and scores9[day9]["sig"] == sig9:
                continue
            fk_of9 = {}
            for w9 in ids9:
                try:
                    stt9 = Path(GVD["id2src"].get(w9) or "").stat()
                    fk_of9[w9] = f"{stt9.st_mtime_ns}|{stt9.st_size}"
                except Exception:                      # noqa: BLE001
                    fk_of9[w9] = "?"
            cached9 = None
            if fp9 is not None and fp9.is_file():      # 디스크 캐시: 센서 부분집합이면 슬라이스
                try:
                    with np.load(fp9, allow_pickle=True) as z9:
                        if str(z9["sig"]) == sig9:
                            csens9 = [str(v9) for v9 in z9["sens"]] if "sens" in z9 else None
                            cached9 = {"sig": sig9, "ids": [str(v9) for v9 in z9["ids"]],
                                       "A": z9["A"], "LR": z9["LR"],
                                       "gkey": [str(v9) for v9 in z9["gkey"]],
                                       "sens": csens9,
                                       "F": (z9["F"] if ("F" in z9 and str(z9["fver"]) == GV_FEAT_VER) else None),
                                       "fcols": ([str(v9) for v9 in z9["fcols"]] if "fcols" in z9 else None)}
                except Exception:                      # noqa: BLE001
                    cached9 = None
            wave_ok9 = _gv_wave_have(wroot9, day9, ids9, sens9, fk_of9) if cached9 else False
            if cached9 and cached9["sens"] is not None and cached9["F"] is not None and wave_ok9 and \
                    all(s9 in cached9["sens"] for s9 in sens9):
                idx9 = [cached9["sens"].index(s9) for s9 in sens9]
                scores9[day9] = {"sig": sig9, "ids": cached9["ids"],
                                 "A": cached9["A"][:, idx9, :, :] if cached9["A"].size else cached9["A"],
                                 "LR": cached9["LR"], "gkey": cached9["gkey"],
                                 "F": cached9["F"][:, idx9, :] if cached9["F"].size else cached9["F"],
                                 "fcols": cached9["fcols"]}
                n_slice9 += 1
                continue
            extra_base9 = None
            if cached9 and cached9["sens"] is not None and cached9["A"].size and \
                    cached9["F"] is not None and wave_ok9:
                have9 = [s9 for s9 in sens9 if s9 in cached9["sens"]]
                need9s = [s9 for s9 in sens9 if s9 not in cached9["sens"]]
                if have9 and need9s:
                    extra_base9 = (cached9, have9, need9s)
            calc_sens9 = extra_base9[2] if extra_base9 else sens9
            items9 = []
            for w9 in ids9:
                f9 = GVD["id2src"].get(w9)
                if not f9:
                    continue
                m9 = GVD["id2meta"].get(w9) or _meta_of(S, w9)
                items9.append((w9, f9, st2id9.get(w9, w9), dict(m9)))
            todo9.append((day9, sig9, fp9, extra_base9, calc_sens9, items9, fk_of9))
        # ── 계산(직렬 또는 프로세스 풀) ──
        n_todo9 = sum(len(t9[5]) for t9 in todo9)
        goldens9 = (GVD.get("build") or {}).get("goldens") or []
        gref9 = GVD.get("gref") or "dsv"
        w_pre9 = int(GVD.get("w") or 0)
        try:
            procs9 = int(os.environ.get("IO_GV_PROCS") or 0) \
                or max(1, min(8, (os.cpu_count() or 2) // 2))
        except Exception:                              # noqa: BLE001
            procs9 = 1
        results9 = {}
        done9 = 0
        t_c9 = time.time()

        def _prog9(extra9=""):
            el9 = time.time() - t_c9
            eta9 = (el9 / done9 * (len(todo9) - done9)) if done9 else 0.0
            GVJOB["progress"] = (f"판정 진행 — {done9}/{len(todo9)}일{extra9}"
                                 + (f" · 남은 ≈ {eta9:.0f}초" if done9 else ""))
        real9 = hasattr(S, "__file__")
        pool_ok9 = False
        if procs9 > 1 and len(todo9) >= 2 and n_todo9 >= 300 and real9:   # 소량은 워커 기동 비용이 더 큼
            try:
                root9 = str(Path(S.__file__).resolve().parents[2])
                model9 = str(getattr(S.ENGINE, "art", "") or "")
                cache9 = (_state_load().get("sd") or {}).get("cache")
                gold_s9 = json.loads(json.dumps(goldens9, default=float))
                from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
                import multiprocessing as _mp9
                nproc9 = min(procs9, len(todo9))
                print(f"[gv] 판정 병렬 계산 — 워커 {nproc9}개 · {len(todo9)}일 {n_todo9}장", flush=True)
                GVJOB["progress"] = f"판정 진행 — 워커 {nproc9}개 기동 중…"
                t_b9 = time.time()
                pend9 = list(todo9)
                with ProcessPoolExecutor(max_workers=nproc9, mp_context=_mp9.get_context("spawn"),
                                         initializer=_gvw_init,
                                         initargs=(root9, model9, cache9, gold_s9, gref9)) as ex9:
                    futs9 = {}
                    first9 = [True]
                    def _submit9():
                        while pend9 and len(futs9) < nproc9 * 2:
                            t9 = pend9.pop(0)
                            fu9 = ex9.submit(_gvw_day, (t9[0], t9[5], t9[4], w_pre9,
                                                       (str(wroot9) if wroot9 else None),
                                                       GV_REF_SENSORS, t9[6]))
                            futs9[fu9] = t9
                    _submit9()
                    while futs9:
                        if GVJOB["cancel"]:
                            pend9.clear()
                            for fu9 in list(futs9):
                                fu9.cancel()
                            raise RuntimeError("중단됨")
                        dn9, _nd9 = wait(list(futs9), timeout=1.0, return_when=FIRST_COMPLETED)
                        for fu9 in dn9:
                            t9 = futs9.pop(fu9)
                            try:
                                d9, out9 = fu9.result()
                                results9[d9] = out9
                                if first9[0]:
                                    first9[0] = False
                                    print(f"[gv] 워커 기동+첫 날짜 완료 {time.time() - t_b9:.1f}s", flush=True)
                            except Exception as e9:    # noqa: BLE001
                                print(f"[gv] {t9[0]} 병렬 계산 실패 — 그 날만 직렬 재계산: {e9}", flush=True)
                                results9[t9[0]] = _gv_day_core(S, t9[0], t9[5], t9[4], w_pre9, goldens9, gref9,
                                                               (str(wroot9) if wroot9 else None),
                                                               GV_REF_SENSORS, t9[6])
                            done9 += 1
                            _prog9(f" · 병렬 {nproc9}")
                        _submit9()
                pool_ok9 = True
            except Exception as e9:                    # noqa: BLE001
                if str(e9) == "중단됨":
                    raise
                print(f"[gv] 병렬 계산 실패 — 직렬로 폴백: {e9}", flush=True)
                results9 = {}
                done9 = 0
        if not pool_ok9:
            for t9 in todo9:
                if GVJOB["cancel"]:
                    raise RuntimeError("중단됨")
                _prog9(f" · {t9[0]}" + (f" · 추가 센서 {len(t9[4])}개만 계산" if t9[3] else ""))
                results9[t9[0]] = _gv_day_core(S, t9[0], t9[5], t9[4], w_pre9, goldens9, gref9,
                                               (str(wroot9) if wroot9 else None), GV_REF_SENSORS, t9[6])
                done9 += 1
        # ── 병합·저장(부모) ──
        for day9, sig9, fp9, extra_base9, calc_sens9, items9, _fk9 in todo9:
            out9 = results9.get(day9)
            if out9 is None:
                continue
            for s9, c9 in (out9.get("miss") or {}).items():
                GVD["sensor_missing"][s9] = GVD["sensor_missing"].get(s9, 0) + int(c9)
            rid9, rgk9 = out9["ids"], out9["gkey"]
            A_new9, LR_new9, F_new9 = out9["A"], out9["LR"], out9["F"]
            save_sens9 = list(calc_sens9)
            fcols9 = None
            if extra_base9 and A_new9.size:              # 캐시 열 + 신규 열 병합(웨이퍼 순서 맞춤)
                cz9, have9, need9s = extra_base9
                pos9 = {w9: i9 for i9, w9 in enumerate(cz9["ids"])}
                if all(w9 in pos9 for w9 in rid9) and len(rid9) == len(cz9["ids"]) and \
                        cz9["F"] is not None and cz9["F"].shape[2] == F_new9.shape[2]:
                    order9 = [pos9[w9] for w9 in rid9]
                    A_all9 = np.concatenate([cz9["A"][order9], A_new9], axis=1)
                    F_all9 = np.concatenate([cz9["F"][order9], F_new9], axis=1)
                    save_sens9 = list(cz9["sens"]) + list(need9s)
                    sel9i = [save_sens9.index(s9) for s9 in sens9]
                    fcols9 = cz9["fcols"]
                    ent9 = {"sig": sig9, "ids": rid9, "A": A_all9[:, sel9i, :, :],
                            "LR": LR_new9, "gkey": rgk9, "F": F_all9[:, sel9i, :], "fcols": fcols9}
                    A_save9, F_save9 = A_all9, F_all9
                else:
                    extra_base9 = None
            if not extra_base9:
                if calc_sens9 != sens9:                # 추가 센서만 계산했는데 병합 불가 → 전량 재계산
                    out9 = _gv_day_core(S, day9, items9, sens9, w_pre9, goldens9, gref9,
                                        (str(wroot9) if wroot9 else None), GV_REF_SENSORS, _fk9)
                    rid9, rgk9 = out9["ids"], out9["gkey"]
                    A_new9, LR_new9, F_new9 = out9["A"], out9["LR"], out9["F"]
                    calc_sens9 = sens9
                ent9 = {"sig": sig9, "ids": rid9, "A": A_new9, "LR": LR_new9, "gkey": rgk9,
                        "F": F_new9, "fcols": None}
                A_save9, F_save9 = A_new9, F_new9
                save_sens9 = list(calc_sens9)
            # F 열 이름: 캐논 런 수는 그날 골든(첫 웨이퍼)의 것 — 길이로 역산
            if ent9.get("fcols") is None:
                nR9 = int(A_save9.shape[2]) if A_save9.ndim == 4 and A_save9.size else 0
                ent9["fcols"] = _gv_fcols([(0, 0, 0)] * nR9) if nR9 else []
            scores9[day9] = ent9; n_new9 += len(rid9)
            if fp9 is not None:
                try:
                    np.savez_compressed(
                        fp9, sig=sig9, ids=np.array(rid9),
                        A=A_save9, LR=ent9["LR"], gkey=np.array(rgk9),
                        sens=np.array(save_sens9), F=F_save9,
                        fcols=np.array(ent9["fcols"]), fver=GV_FEAT_VER)
                except Exception as e9:                # noqa: BLE001
                    print(f"[gv] 일별 캐시 저장 실패({day9}): {e9}", flush=True)
        print(f"[gv] 판정 스캔: 신규 {n_new9}장 · 캐시 슬라이스 {n_slice9}일 · {len(days9)}일 · "
              f"{time.time() - ts9:.1f}s"
              + (f" · 센서 결측 {dict(GVD.get('sensor_missing') or {})}" if GVD.get("sensor_missing") else ""),
              flush=True)
        if GVD.get("mode") in ("vlm", "hybrid"):
            _gv_vlm_phase(d0, d1, GVD["mode"],
                          float(GVD.get("p_k") or 2.0),
                          float(GVD.get("p_alpha") or 0.05))

    # ★잔차 뷰 필수(실측): 겹침 뷰는 VLM 이 직선화·봉우리도 놓친다(gemma3
    #   3/3 오답) — 잔차(대상−기준)+0선+밴드로 바꾸면 3/3 정답. 추세 스캔의
    #   ts_narrate 교훈과 동일 계열.
    #   + σ 정규화·클립 필수(2차 실측): 원단위 잔차는 불량이 크면 밴드가
    #   실낱이 돼 모델이 '확신 없음→OK'로 빠지고, 밴드를 키우면 노이즈를
    #   초과로 환각 — σ 단위(밴드 ±3 고정 가시)+±12 클립에서 6/6 정답.
    # ★VLM 은 '밴드 초과'가 아니라 파형 모양을 말로 비교해 판정(사용자 요구):
    #   골든 모양 서술 → 대상 모양 서술 → 방향·형태·변화량·변곡 비교 → 잔차 확인.
    #   한국어 전용·네 줄 고정 양식. 사용자 공통/스텝별 지시가 뒤에 붙는다.
    _GV_VLM_SYS = (
        "당신은 반도체 공정 센서 파형 검사 전문가입니다. 반드시 한국어로만 답합니다"
        "(영어 단어·문장 금지, 숫자와 'NG'/'OK' 표기만 허용).\n"
        "이미지는 한 스텝 구간입니다. 위 그래프: 파랑 선=정상 기준(골든), 빨강 선="
        "검사 대상 웨이퍼(겹쳐 그림). 아래 그래프: 빨강 선=대상과 골든의 차이(σ 단위), "
        "회색 띠=정상 허용 범위(±3σ). 제목에 골든과 대상의 시작→끝 변화량이 숫자로 "
        "적혀 있으니 변화량(낙폭/상승폭) 비교에 반드시 사용하세요.\n"
        "검사 절차: ① 골든의 모양을 말로 서술(방향: 상승/하강/평탄, 형태: 곡선/직선/"
        "계단/봉우리/골/진동, 시작값 대비 끝값, 변곡 위치). ② 대상의 모양을 같은 "
        "항목으로 서술. ③ 두 모양을 비교 — 방향·형태·변화량·변곡 위치 중 다른 점을 "
        "짚음. ④ 아래 잔차 그래프에서 빨강이 회색 띠 밖에 머무는 구간이 있는지로 "
        "차이가 유의미한지 확인. ⑤ 모양이 실질적으로 다르면 NG, 자잘한 노이즈 외에 "
        "같으면 OK. 확신이 없으면 OK.\n"
        "출력은 아래 네 줄 양식만, 각 줄 한 문장:\n"
        "판정: NG 또는 OK\n골든: (골든 모양 서술)\n"
        "현재: (대상 모양 서술 — 골든과 다른 점 포함)\n근거: (판정 이유 한 문장)")

    def _gv_vlm_prompt(step9):
        """시스템 지시 + 사용자 공통/스텝별 검사 지시(③ 판정 섹션에서 편집)."""
        n9 = GVD.get("vlm_notes") or {}
        txt9 = _GV_VLM_SYS
        c9 = str(n9.get("common") or "").strip()
        if c9:
            txt9 += "\n\n[사용자 공통 검사 지시]\n" + c9
        st9 = str((n9.get("steps") or {}).get(str(step9)) or "").strip()
        if st9:
            txt9 += f"\n\n[이 스텝(step {step9})의 정상 파형 특징과 검사 방법]\n" + st9
        return txt9

    def _gv_kfont():
        """한글 폰트(있으면) — matplotlib 기본 폰트는 한글 글리프가 없다."""
        try:
            from matplotlib import font_manager as _fm9
            for cand9 in ("NanumSquareRound", "NanumSquareRoundR", "NanumGothic",
                          "NanumSquare_ac", "Noto Sans CJK KR", "Noto Sans CJK SC",
                          "Malgun Gothic", "AppleGothic"):
                try:
                    fp9 = _fm9.findfont(_fm9.FontProperties(family=cand9),
                                        fallback_to_default=False)
                    if fp9:
                        return _fm9.FontProperties(fname=fp9)
                except Exception:                      # noqa: BLE001
                    continue
        except Exception:                              # noqa: BLE001
            pass
        return None

    def _gv_ollama():
        try:
            from rca_causal import _ollama_chat as _oc9
        except Exception:                              # noqa: BLE001
            from io_recon_tcn.rca_causal import _ollama_chat as _oc9
        return _oc9

    def _gv_vlm_model():
        host9 = os.environ.get("OLLAMA_HOST",
                               "http://127.0.0.1:11434").rstrip("/")
        m9 = (os.environ.get("GV_VLM_MODEL")
              or os.environ.get("RCA_VLM_MODEL") or "").strip()
        if "gpt-oss" in m9.lower():                    # 텍스트 전용 — 이미지 불가
            print(f"[gv] {m9} 는 이미지를 못 보는 텍스트 모델 — 비전 모델을 "
                  "자동 탐색합니다(GV_VLM_MODEL 로 지정 가능)", flush=True)
            m9 = ""
        names9 = []
        try:
            import urllib.request as _ur9
            with _ur9.urlopen(host9 + "/api/tags", timeout=5) as r9:
                names9 = [str(t9.get("name") or "") for t9 in
                          (json.loads(r9.read().decode()).get("models") or [])]
        except Exception:                              # noqa: BLE001
            pass
        if m9:
            # ★ollama 호출은 등록명과 '정확히' 일치해야 한다 — 짧은 지정
            #   (qwen3.6 등)은 설치 목록에서 부분일치로 전체 이름을 해석
            if names9 and m9 not in names9:
                hit9 = [n9 for n9 in names9 if m9.lower() in n9.lower()]
                if hit9:
                    print(f"[gv] GV_VLM_MODEL '{m9}' → 설치명 "
                          f"'{hit9[0]}' 로 해석", flush=True)
                    return host9, hit9[0]
                print(f"[gv] GV_VLM_MODEL '{m9}' 가 설치 목록에 없음 — "
                      "자동 탐색으로 대체", flush=True)
                m9 = ""
            if m9:
                return host9, m9
        # 미설정/미해석 → 설치 비전 모델 자동. qwen3.6 unsloth GGUF 는
        # mmproj 내장 — 비전 가능(설비 도면 판독 실증 관례)
        for pat9 in ("gemma3", "qwen3-vl", "qwen2.5-vl", "-vl", "vl:",
                     "llava", "gemma4", "qwen3.6", "qwen3_6"):
            for n9 in names9:
                if pat9 in n9.lower():
                    return host9, n9
        m29 = (os.environ.get("RCA_LLM_MODEL") or "").strip()
        if m29 and "gpt-oss" not in m29.lower():       # 마지막 폴백: 주 LLM
            return host9, m29
        GVD["vlm_names"] = names9
        return host9, ""

    def _gv_vlm_cache():
        c9 = GVD.get("vlm_cache")
        if c9 is None:
            c9 = {}
            try:
                fp9 = Path(GVD["store"]) / "gv_vlm_cache.json"
                if fp9.is_file():
                    c9 = json.loads(fp9.read_text(encoding="utf-8"))
            except Exception:                          # noqa: BLE001
                pass
            GVD["vlm_cache"] = c9
        return c9

    def _gv_vlm_cache_save():
        try:
            (Path(GVD["store"]) / "gv_vlm_cache.json").write_text(
                json.dumps(GVD.get("vlm_cache") or {}, ensure_ascii=False),
                encoding="utf-8")
        except Exception as e9:                        # noqa: BLE001
            print(f"[gv] VLM 캐시 저장 실패: {e9}", flush=True)

    def _gv_render_png(x9, g29, s29, title9):
        """스텝 2단 PNG: 위=골든(파랑)·실측(빨강) 겹침(모양 비교), 아래=σ정규화 잔차
        (밴드 ±3σ·클립 ±12σ, 증거). 제목에 시작→끝 변화량 수치(모양 비교 보조).
        호출부가 offset 정렬(평균 차감)한 값을 주므로 겹침은 모양만 비교된다.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        kf9 = _gv_kfont()
        fin9 = np.isfinite(x9) & np.isfinite(g29)
        xs9 = x9[fin9]; gs9 = g29[fin9]
        r9 = x9 - g29
        dd9 = np.diff(x9[np.isfinite(x9)])
        nz9 = (float(np.median(np.abs(dd9 - np.median(dd9)))) / 0.9539
               if len(dd9) else 0.0)
        s_eff9 = np.maximum(np.maximum(s29, nz9), 1e-9)
        rz9 = np.clip(r9 / s_eff9, -12, 12)
        w_in9 = float(min(12.8, max(8.0, len(r9) / 100.0)))
        fig9, (a19, a29) = plt.subplots(2, 1, figsize=(w_in9, 5.6), dpi=120,
                                        gridspec_kw={"height_ratios": [3, 2]})
        ix9 = np.arange(len(g29))
        a19.plot(ix9, g29, color="#3b6fd4", lw=1.8)
        a19.plot(ix9, x9, color="#d64545", lw=1.3)
        if len(gs9) >= 2 and len(xs9) >= 2:
            k9 = max(1, len(gs9) // 20)
            g09, g19 = float(np.mean(gs9[:k9])), float(np.mean(gs9[-k9:]))
            x09, x19 = float(np.mean(xs9[:k9])), float(np.mean(xs9[-k9:]))
            if kf9 is not None:
                ttl9 = (f"{title9}\n골든: 시작 {g09:.2f} → 끝 {g19:.2f} (변화 {g19 - g09:+.2f})"
                        f"   대상: 시작 {x09:.2f} → 끝 {x19:.2f} (변화 {x19 - x09:+.2f})")
            else:
                ttl9 = (f"{title9}\ngolden: start {g09:.2f} -> end {g19:.2f} "
                        f"(change {g19 - g09:+.2f}) | wafer: start {x09:.2f} -> end "
                        f"{x19:.2f} (change {x19 - x09:+.2f})")
        else:
            ttl9 = title9
        if kf9 is not None:
            a19.set_title(ttl9, fontsize=8.5, fontproperties=kf9)
            a19.legend(["골든(기준)", "대상 웨이퍼"], fontsize=7.5, loc="best",
                       prop=kf9)
            a29.set_title("잔차(대상−골든, σ 단위) · 회색 띠=정상 허용 ±3σ",
                          fontsize=8.5, fontproperties=kf9)
        else:
            a19.set_title(ttl9, fontsize=8.5)
            a19.legend(["golden", "wafer"], fontsize=7.5, loc="best")
            a29.set_title("residual (wafer - golden) in sigma units, band +/-3",
                          fontsize=8.5)
        a19.tick_params(labelsize=7)
        a29.axhline(0, color="k", lw=1.0)
        a29.fill_between(ix9, -3, 3, color="#b9c2cf", alpha=0.6, linewidth=0)
        a29.plot(ix9, rz9, color="#d64545", lw=1.2)
        a29.set_ylim(-12.5, 12.5)
        a29.tick_params(labelsize=7)
        fig9.tight_layout()
        fp9 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig9.savefig(fp9.name, bbox_inches="tight")
        plt.close(fig9)
        return fp9.name

    def _gv_vlm_judge_step(wid9, sn9, r9, x29, g29, s29, len_txt9):
        """스텝 1건 VLM 모양 서술 판정(캐시 우선) → (ng, 서술 네 줄)."""
        b9 = GVD.get("build") or {}
        host9, model9 = GVD.get("vlm_hm") or ("", "")
        m9s = re.search(r"step (\d+)", len_txt9)
        step_id9 = int(m9s.group(1)) if m9s else r9
        prm9 = _gv_vlm_prompt(step_id9)
        key9 = hashlib.sha1(
            f"{wid9}|{sn9}|{r9}|{b9.get('sig')}|{model9}|shape1|"
            f"{hashlib.md5(prm9.encode()).hexdigest()[:10]}".encode()).hexdigest()
        cache9 = _gv_vlm_cache()
        if key9 in cache9:
            return bool(cache9[key9]["ng"]), str(cache9[key9].get("reason") or "")
        import base64
        png9 = _gv_render_png(x29, g29, s29, len_txt9)
        try:
            b649 = base64.b64encode(Path(png9).read_bytes()).decode()
            # 비전 경로는 temperature 를 못 박을 수 없어(옵션 포함 시 일부 VL
            # 모델 폭주 — 실측) 판정이 흔들릴 수 있다 → 3표 다수결, 2표 일치
            # 시 조기 종료. 후보(룰 선별)만 오므로 비용 상한은 낮다.
            votes9, descs9 = [], []
            ctx9 = GVD.get("vlm_ctx") or ""
            for _v9 in range(3):
                t09 = time.time()
                GVD["vlm_calls"] = int(GVD.get("vlm_calls") or 0) + 1
                GVJOB["progress"] = (f"판정 진행 — {ctx9} · {sn9} step{step_id9} 표 "
                                     f"{_v9 + 1} · VLM 호출 {GVD['vlm_calls']}회"
                                     f"{GVD.get('vlm_eta') or ''}")
                txt9 = _gv_ollama()(
                    host9, model9, "", prm9 + "\n\n대상: " + len_txt9,
                    int(os.environ.get("GV_VLM_TIMEOUT", "240")), images=[b649])
                raw9 = str(txt9 or "")
                lines9 = [l9.strip() for l9 in raw9.splitlines() if l9.strip()]
                m9v = re.search(r"판정\s*[:：]\s*(NG|OK)", raw9, re.I)
                if m9v:
                    v9 = m9v.group(1).upper() == "NG"
                else:                                  # 양식 이탈 폴백
                    head9 = (lines9[0].upper() if lines9 else "")
                    v9 = head9.startswith("NG") or (" NG" in f" {head9}"
                                                    and "OK" not in head9)
                votes9.append(bool(v9))
                desc9 = "\n".join(l9 for l9 in lines9
                                  if re.match(r"^(판정|골든|현재|근거)\s*[:：]", l9)) \
                    or "\n".join(lines9[:4])
                descs9.append((bool(v9), desc9))
                print(f"[gv] VLM {wid9[-22:]} {sn9} step{step_id9} 표{_v9 + 1}: "
                      f"{'NG' if v9 else 'OK'} ({time.time() - t09:.1f}s) "
                      f"{desc9.replace(chr(10), ' / ')[:160]}", flush=True)
                if len(votes9) == 2 and votes9[0] == votes9[1]:
                    break
            ng9 = votes9.count(True) >= 2
            # 서술은 다수결과 같은 표의 것 — OK 여도 보존(사용자가 모양 서술을 본다)
            reason9 = next((d9 for v9, d9 in descs9 if v9 == ng9), "")
        finally:
            try:
                os.unlink(png9)
            except OSError:
                pass
        cache9[key9] = {"ng": bool(ng9), "reason": reason9}
        return bool(ng9), reason9

    def _gv_canonstep(gk9, r9):
        """gkey 문자열 → 그 골든 캐논의 r9 번째 스텝 id(스텝별 α 조회용)."""
        for g9 in (GVD.get("build") or {}).get("goldens") or []:
            if f"{g9['eqp']}|{g9['recipe']}|{g9['dsv'] or ''}" == gk9:
                return g9["canon"][r9][0]
        return r9

    def _gv_vlm_phase(d0, d1, mode9, k9, alpha9):
        """VLM 판정 단계: vlm=기간 전체(상한), hybrid=룰 NG 후보만."""
        host9, model9 = _gv_vlm_model()
        if not model9:
            raise RuntimeError(
                "비전 모델을 찾지 못했습니다 — GV_VLM_MODEL 환경변수로 지정하세요. "
                f"현재 ollama 설치 모델: {', '.join(GVD.get('vlm_names') or []) or '(조회 실패)'}")
        GVD["vlm_hm"] = (host9, model9)
        print(f"[gv] VLM 판정 모델: {model9} @ {host9}", flush=True)
        scores9 = GVD.get("scores") or {}
        sens9 = GVD.get("run_sens") or []
        cap9 = int(GVD.get("vlm_cap") or 200)
        cand9 = []                                     # (wid, day, {sn:[스텝 idx]})
        for day9 in sorted(scores9):
            ent9 = scores9[day9]
            Ak9 = _gv_interp_k(ent9["A"], k9) if ent9["A"].shape[0] else None
            for w9 in range(len(ent9["ids"])):
                tgt9 = {}
                for j9, s9 in enumerate(sens9):
                    if GVD.get("p_sel") is not None and s9 not in GVD["p_sel"]:
                        continue                       # 점수 미반영 센서는 VLM 도 생략
                    rr9 = []
                    for r9 in range(Ak9.shape[2]):
                        v9 = Ak9[w9, j9, r9]
                        if not np.isfinite(v9):
                            continue
                        a2x9 = _gv_thr(s9, _gv_canonstep(ent9["gkey"][w9], r9),
                                       alpha9, GVD.get("p_astp"), GVD.get("p_asn"),
                                       GVD.get("p_acell"))
                        if mode9 == "vlm" or v9 > a2x9:
                            rr9.append(r9)
                    if rr9:
                        tgt9[s9] = rr9
                if tgt9:
                    cand9.append((ent9["ids"][w9], day9, tgt9))
        if mode9 == "hybrid":
            cand9 = [c9 for c9 in cand9 if c9[2]]
        if len(cand9) > cap9:
            print(f"[gv] VLM 대상 {len(cand9)}장 — 상한 {cap9}장으로 제한"
                  "(불량 의심 우선)", flush=True)
            cand9 = cand9[:cap9]
        vlm9 = GVD.setdefault("vlm", {})
        b9 = GVD.get("build") or {}
        n_pairs9 = sum(len(rr9) for _w9, _d9, tg9 in cand9 for rr9 in tg9.values())
        GVD["vlm_calls"], GVD["vlm_eta"] = 0, ""
        t0v9 = time.time()
        print(f"[gv] VLM 판정 시작 — 대상 웨이퍼 {len(cand9)}장 · (센서,스텝) 쌍 "
              f"{n_pairs9}개 · 호출 최대 {n_pairs9 * 3}회(2표 일치 시 조기 종료) · "
              f"모델 {model9} — 첫 호출은 모델 적재로 수 분 걸릴 수 있음",
              flush=True)
        if mode9 == "vlm" and n_pairs9 > 400:
            print(f"[gv] ★VLM 모드 전수({n_pairs9}쌍)는 매우 느립니다 — 하이브리드"
                  "(룰베이스 선별 후 VLM)를 권장", flush=True)
        done_pairs9 = 0
        for ci9, (wid9, day9, tgt9) in enumerate(cand9):
            if GVJOB["cancel"]:
                raise RuntimeError("중단됨")
            GVD["vlm_ctx"] = f"웨이퍼 {ci9 + 1}/{len(cand9)} ({wid9[-18:]})"
            el9 = time.time() - t0v9
            if done_pairs9:
                rem9 = (n_pairs9 - done_pairs9) * el9 / done_pairs9
                GVD["vlm_eta"] = (f" · 경과 {int(el9 // 60)}분 · 남은 예상 "
                                  f"{int(rem9 // 60)}분")
            GVJOB["progress"] = (f"판정 진행 — {GVD['vlm_ctx']} · VLM 호출 "
                                 f"{GVD['vlm_calls']}회{GVD['vlm_eta']}")
            print(f"[gv] VLM 웨이퍼 {ci9 + 1}/{len(cand9)} {wid9[-22:]} — "
                  f"쌍 {sum(len(v9) for v9 in tgt9.values())}개{GVD['vlm_eta']}",
                  flush=True)
            f9 = GVD["id2src"].get(wid9)
            m9 = GVD["id2meta"].get(wid9) or {}
            g9, _ge9 = _gv_pick_golden(m9)
            if not (f9 and g9):
                continue
            recs9, _a, _b, _c = _gv_load(
                Path(f9), stems=[(GVD.get("id2stem") or {}).get(wid9, wid9)])
            if not recs9:
                continue
            rec9 = recs9[0]
            cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
            stw9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
            canon9 = [(u9, p9, l9) for u9, p9, l9 in g9["canon"]]
            ngs9, reason9, notes9 = {}, "", []
            for s9, rr9 in tgt9.items():
                if s9 not in cols9:
                    continue
                arr9 = np.asarray(rec9["sensors"],
                                  dtype=np.float64)[:, [cols9.index(s9)]]
                wv9 = _warp_to_canon(arr9, stw9, canon9, int(g9["L"]))[:, 0]
                gm9 = np.asarray([np.nan if v9 is None else v9
                                  for v9 in g9["mean"][s9]], dtype=np.float64)
                gs9 = np.asarray(g9["std"][s9], dtype=np.float64)
                for r9 in rr9:
                    u9, p9, l9 = canon9[r9]
                    x29 = wv9[p9:p9 + l9]
                    if not np.isfinite(x29).any():
                        continue
                    raw9 = g9["raw_len_med"][r9]
                    ttl9 = (f"step {u9} | golden {raw9}pt | normalized "
                            f"residual (band +/-3 sigma, clip +/-12)")
                    gseg9 = gm9[p9:p9 + l9]
                    try:
                        ng9, rs9 = _gv_vlm_judge_step(
                            wid9, s9, r9,
                            x29 - np.nanmean(x29) + np.nanmean(gseg9),   # offset 정렬·실레벨
                            gseg9, gs9[p9:p9 + l9], ttl9)
                    except Exception as e9:            # noqa: BLE001 — 타임아웃 등
                        print(f"[gv] VLM 호출 실패 {wid9[-22:]} {s9} step{u9}: {e9} "
                              "— 이 스텝은 건너뜀", flush=True)
                        GVD["vlm_fail"] = int(GVD.get("vlm_fail") or 0) + 1
                        continue
                    if ng9:
                        ngs9.setdefault(s9, []).append(int(u9))
                    if rs9:
                        notes9.append(f"[{s9} step{u9}] " + rs9.replace("\n", " / "))
                        if ng9 and not reason9:
                            reason9 = f"{s9} step{u9}: " + rs9.replace("\n", " / ")
            vlm9[wid9] = {"ng": bool(ngs9), "steps": ngs9, "reason": reason9,
                          "notes": notes9}
            done_pairs9 += sum(len(v9) for v9 in tgt9.values())
            print(f"[gv] VLM 웨이퍼 {wid9[-22:]} 결과: "
                  f"{'NG ' + str(ngs9) if ngs9 else 'OK'}", flush=True)
            del recs9
            if (ci9 + 1) % 20 == 0:
                _gv_vlm_cache_save()
        _gv_vlm_cache_save()

    def _gv_job(fn9, *args9):
        def run9():
            gen09 = GVX["gen"]
            try:
                GVJOB.update(state="running", error=None, cancel=False, progress="")
                fn9(*args9)
                if GVX["gen"] != gen09:                # 작업 중 목록이 바뀜 → 결과 무효
                    raise RuntimeError("작업 중 목록이 바뀌어 결과를 버렸습니다 — 다시 실행하세요")
                GVJOB.update(state="done", progress="완료")
            except Exception as e9:                    # noqa: BLE001
                import traceback
                print("[gv] 작업 실패:\n" + traceback.format_exc(), flush=True)
                GVJOB.update(state="error", error=str(e9))
        threading.Thread(target=run9, daemon=True).start()

    @app.post("/api/gv_build")
    def api_gv_build(payload: dict = Body(...)):
        if not GVD.get("id2date"):
            raise HTTPException(400, "먼저 웨이퍼 목록을 불러오세요")
        _gv_preempt_restore()
        if GVJOB["state"] == "running":
            raise HTTPException(409, "작업이 이미 진행 중입니다")
        sens9 = [str(s9) for s9 in (payload.get("sensors") or [])] \
            or list(GVD.get("sensors") or [])
        if not sens9:
            raise HTTPException(400, "센서를 선택하세요")
        GVD["build_sens"] = sens9
        _gv_set_facet(payload)
        per_dsv9 = {str(k9): {"d0": str((v9 or {}).get("d0") or ""),
                              "d1": str((v9 or {}).get("d1") or "")}
                    for k9, v9 in (payload.get("per_dsv") or {}).items()
                    if (v9 or {}).get("d0") or (v9 or {}).get("d1")}
        GVD["restore_ids"] = [str(x9) for x9 in (payload.get("restore_ids") or [])]
        _gv_session_save("build", dict(payload, sensors=sens9))
        _gv_session_save("run", None)
        _gv_job(_gv_build_worker, str(payload.get("d0") or ""),
                str(payload.get("d1") or ""), sens9, per_dsv9, GVD["restore_ids"])
        return {"ok": True}

    @app.post("/api/gv_run")
    def api_gv_run(payload: dict = Body(...)):
        if not (GVD.get("build") or {}).get("goldens"):
            raise HTTPException(400, "먼저 골든 산정을 실행하세요")
        _gv_preempt_restore()
        if GVJOB["state"] == "running":
            raise HTTPException(409, "작업이 이미 진행 중입니다")
        GVD["run_sens"] = [str(s9) for s9 in (payload.get("sensors") or [])] \
            or list(GVD.get("build_sens") or [])
        gsens9 = set()
        for g9 in (GVD.get("build") or {}).get("goldens") or []:
            gsens9 |= set(g9.get("sensors") or [])
        lack9 = [s9 for s9 in GVD["run_sens"] if s9 not in gsens9]
        if lack9:
            raise HTTPException(400, "골든에 없는 센서: " + ", ".join(lack9[:6])
                                     + " — ② 골든 산정에 그 센서를 포함해 다시 산정한 뒤 판정하세요")
        GVD["w"] = int(payload.get("w") or 0)
        _gv_set_facet(payload)
        GVD["mode"] = str(payload.get("mode") or "rule")
        GVD["gref"] = "rep" if str(payload.get("gref") or "") == "rep" else "dsv"
        vn9 = payload.get("vlm_notes") or {}
        GVD["vlm_notes"] = {"common": str(vn9.get("common") or ""),
                            "steps": {str(k9): str(v9) for k9, v9 in
                                      (vn9.get("steps") or {}).items() if str(v9).strip()}}
        GVD["p_k"] = float(payload.get("k") or 2.0)
        GVD["p_alpha"] = float(payload.get("alpha") or 0.05)
        GVD["p_astp"], GVD["p_asn"], GVD["p_acell"], GVD["p_sel"] = \
            _gv_parse_alpha(payload)
        GVD["p_lv"] = _gv_parse_levels(payload)
        GVD["p_rules"] = _gv_parse_rules(payload)
        GVD["vlm_cap"] = int(payload.get("vlm_cap") or 200)
        GVD["scores"] = {}
        _gv_session_save("run", dict(payload))
        _gv_job(_gv_scan_worker, str(payload.get("d0") or ""),
                str(payload.get("d1") or ""))
        return {"ok": True}

    @app.post("/api/gv_cancel")
    def api_gv_cancel():
        GVJOB["cancel"] = True
        return {"ok": True}

    @app.get("/api/gv_status")
    def api_gv_status():
        try:
            return _gv_status_body()
        except Exception as e9:                        # noqa: BLE001 — 폴링은 절대 500 금지
            import traceback
            print("[gv] 상태 조회 예외(폴링은 계속):\n" + traceback.format_exc(), flush=True)
            return {"state": GVJOB["state"], "progress": GVJOB["progress"],
                    "error": GVJOB["error"], "steps": [], "build": None,
                    "status_warn": str(e9)}

    def _gv_status_body():
        b9 = GVD.get("build") or {}
        steps9 = sorted({int(u9) for g9 in (b9.get("goldens") or [])
                         for u9, _p9, _l9 in g9["canon"]})
        return {"state": GVJOB["state"], "progress": GVJOB["progress"],
                "error": GVJOB["error"], "steps": steps9,
                "build": ({"sig": b9.get("sig"),
                           "excluded_n": b9.get("excluded_n"),
                           "goldens": [{k9: g9.get(k9) for k9 in
                                        ("eqp", "recipe", "dsv", "n", "small",
                                         "auto_excluded")}
                                       for g9 in b9.get("goldens") or []]}
                          if b9 else None),
                "sensors": list(GVD.get("run_sens") or GVD.get("build_sens") or [])}

    def _gv_interp_k(A9, k9):
        """k격자 사전계산 A → 임의 k 선형 보간(마지막 축)."""
        ks9 = GV_KGRID
        if k9 <= ks9[0]:
            return A9[..., 0]
        if k9 >= ks9[-1]:
            return A9[..., -1]
        j9 = int((k9 - ks9[0]) / 0.5)
        t9 = (k9 - ks9[j9]) / 0.5
        return A9[..., j9] * (1 - t9) + A9[..., j9 + 1] * t9

    def _gv_thr(sn9, st9, alpha9, astp9, asn9, acell9):
        """면적 임계 조회 — 센서×스텝 > 센서 > 스텝 > 전역(캐시 밖, 즉시 재판정)."""
        v9 = (acell9 or {}).get(f"{sn9}|{int(st9)}")
        if v9 is None:
            v9 = (asn9 or {}).get(sn9)
        if v9 is None:
            v9 = (astp9 or {}).get(int(st9))
        return float(alpha9 if v9 is None else v9)

    def _gv_parse_alpha(payload):
        """payload 의 α 묶음 → (astp{step}, asn{sn}, acell{'sn|step'}, sel[list|None])."""
        astp9, asn9, acell9 = {}, {}, {}
        for k2, v2 in (payload.get("alpha_steps") or {}).items():
            try:
                astp9[int(k2)] = float(v2)
            except Exception:                          # noqa: BLE001
                pass
        for k2, v2 in (payload.get("alpha_sensor") or {}).items():
            try:
                asn9[str(k2)] = float(v2)
            except Exception:                          # noqa: BLE001
                pass
        for k2, v2 in (payload.get("alpha_cell") or {}).items():
            try:
                sn2, st2 = str(k2).rsplit("|", 1)
                acell9[f"{sn2}|{int(st2)}"] = float(v2)
            except Exception:                          # noqa: BLE001
                pass
        sel9 = payload.get("sensor_sel")
        sel9 = [str(x9) for x9 in sel9] if isinstance(sel9, list) else None
        return astp9, asn9, acell9, sel9

    def _gv_parse_levels(payload):
        """levels={step:[중≥, 상≥]}(임계 θ 배수) → {int step:(t1,t2)}; 기본 (0.5, 1.0)."""
        out9 = {}
        for k9, v9 in (payload.get("levels") or {}).items():
            try:
                t1, t2 = float(v9[0]), float(v9[1])
                out9[int(k9)] = (max(0.0, t1), max(0.0, t2))
            except Exception:                          # noqa: BLE001
                continue
        return out9

    def _gv_rule_auto(sn9):
        """센서 이름으로 모양 규칙 자동 활성: 출구 온도(t_·outlet) → W 봉우리, 전후 차압 → 아치 기울기."""
        t9 = str(sn9).lower()
        if t9.startswith("t_") and "outlet" in t9:
            return "wprom"
        if t9.startswith("dp_") and ("vslf-vslb" in t9 or re.search(r"front.*back", t9)):
            return "arch"
        return None

    def _gv_parse_rules(payload):
        """rules={w:{t1,t2,step,combine}, arch:{t1,t2,steps,lr_fix,combine}, off:[sn], force:{sn:rule}}
        → 정규화 dict(캐시 밖, 즉시 재판정). 기본: W 깊이 중≥0.5·상≥1.5℃ 스텝4, 아치 비 중≤0.6·상≤0.3 스텝1·3."""
        r9 = payload.get("rules") or {}
        w9 = r9.get("w") or {}
        a9 = r9.get("arch") or {}
        def _f9(v9, d9):
            try:
                return float(v9)
            except Exception:                          # noqa: BLE001
                return d9
        try:
            steps9 = [int(x9) for x9 in (a9.get("steps") or [1, 3])]
        except Exception:                              # noqa: BLE001
            steps9 = [1, 3]
        mode9 = "ratio" if str(a9.get("mode") or "") == "ratio" else "abs"
        return {"w": {"t1": _f9(w9.get("t1"), 0.5), "t2": _f9(w9.get("t2"), 1.5),
                      "step": int(_f9(w9.get("step"), 4)), "combine": bool(w9.get("combine"))},
                "arch": {"mode": mode9,
                         "t1": _f9(a9.get("t1"), (0.03 if mode9 == "abs" else 0.6)),
                         "t2": _f9(a9.get("t2"), (0.01 if mode9 == "abs" else 0.3)), "steps": steps9,
                         "lr_fix": (True if a9.get("lr_fix") is None else bool(a9.get("lr_fix"))),
                         "combine": bool(a9.get("combine"))},
                "off": [str(x9) for x9 in (r9.get("off") or [])],
                "force": {str(k9): str(v9) for k9, v9 in (r9.get("force") or {}).items()}}

    def _gv_gmap():
        """gkey → 골든 dict(선형 탐색 제거)."""
        b9 = GVD.get("build") or {}
        if GVD.get("gmap_sig") != b9.get("sig") or "gmap" not in GVD:
            GVD["gmap"] = {f"{g9['eqp']}|{g9['recipe']}|{g9['dsv'] or ''}": g9
                           for g9 in (b9.get("goldens") or [])}
            GVD["gmap_sig"] = b9.get("sig")
            GVD["gfeat"] = {}
        return GVD["gmap"]

    def _gv_gfeat(gk9, sens9):
        """골든 평균 파형의 형상 특징(센서별 F 행) — gkey·센서 집합당 1회 계산."""
        key9 = (gk9, tuple(sens9))
        cache9 = GVD.setdefault("gfeat", {})
        if key9 in cache9:
            return cache9[key9]
        g9 = _gv_gmap().get(gk9)
        if g9 is None:
            cache9[key9] = None
            return None
        canon9 = [(u9, p9, l9) for u9, p9, l9 in g9["canon"]]
        L9 = int(g9["L"])
        M9 = np.full((L9, len(sens9)), np.nan)
        for j9, s9 in enumerate(sens9):
            if s9 in (g9.get("mean") or {}):
                M9[:, j9] = np.asarray([np.nan if v9 is None else v9 for v9 in g9["mean"][s9]], dtype=np.float64)
        F9 = _gv_feats_one(M9, canon9)
        cache9[key9] = {"F": F9, "fidx": {c9: i9 for i9, c9 in enumerate(_gv_fcols(canon9))}, "canon": canon9}
        return cache9[key9]

    def _gv_shape_bads(rule9, rules9, Fw9, fidx9, canon9, gk9, sens9, j9, LRw9):
        """웨이퍼 한 장·센서 하나의 모양 규칙 평가 → [bad dict | None(판정불가)] (lv 0 도 r 갱신용으로 돌려줌)."""
        out9 = []
        steps_of9 = {}
        for r9, (u9, _p9, _l9) in enumerate(canon9):
            steps_of9.setdefault(int(u9), r9)          # 반복 스텝이면 첫 런
        if rule9 == "wprom":
            cfg9 = rules9["w"]
            r9 = steps_of9.get(int(cfg9["step"]))
            if r9 is None or f"w.r{r9}.depth" not in fidx9:
                return out9
            d9 = float(Fw9[fidx9[f"w.r{r9}.depth"]])
            if not np.isfinite(d9):
                return [None]
            t1, t2 = float(cfg9["t1"]), float(cfg9["t2"])
            lv9 = 2 if d9 >= t2 else (1 if d9 >= t1 else 0)
            out9.append({"step": int(cfg9["step"]), "rule": "wprom", "val": round(d9, 3),
                         "width": round(float(Fw9[fidx9[f"w.r{r9}.width"]]), 1),
                         "flag": int(Fw9[fidx9[f"w.r{r9}.flag"]] > 0),
                         "pk": int(Fw9[fidx9[f"w.r{r9}.pk"]]), "valley": int(Fw9[fidx9[f"w.r{r9}.valley"]]),
                         "r": round(d9 / t1, 3) if t1 > 0 else 0.0, "lv": lv9})
            return out9
        if rule9 == "arch":
            cfg9 = rules9["arch"]
            gf9 = _gv_gfeat(gk9, sens9)
            t1, t2 = float(cfg9["t1"]), float(cfg9["t2"])
            for st9 in cfg9["steps"]:
                r9 = steps_of9.get(int(st9))
                if r9 is None or f"a.r{r9}.s" not in fidx9:
                    continue
                sw9 = float(Fw9[fidx9[f"a.r{r9}.s"]])
                if not np.isfinite(sw9):
                    out9.append(None)
                    continue
                sg9 = float(gf9["F"][j9, gf9["fidx"][f"a.r{r9}.s"]]) if gf9 is not None else np.nan
                dg9 = float(gf9["F"][j9, gf9["fidx"][f"a.r{r9}.drop"]]) if gf9 is not None else np.nan
                l_r9 = int(canon9[r9][2])
                g_w9 = _gv_gmap().get(gk9) or {}
                raw_med9 = float((g_w9.get("raw_len_med") or [l_r9] * (r9 + 1))[r9]) if g_w9 else float(l_r9)
                lr_w9 = float(LRw9[r9]) if (LRw9 is not None and r9 < len(LRw9) and np.isfinite(LRw9[r9]) and LRw9[r9] > 0) else 1.0
                raw_len9 = max(1.0, raw_med9 * lr_w9)              # 이 웨이퍼의 원본 스텝 포인트 수
                if cfg9.get("mode") == "ratio":        # 비 모드: 웨이퍼 s(원본 점당, f4) → 캐논 점당으로 환산해 골든과 비교
                    sw9 = sw9 * (raw_len9 / max(1.0, float(l_r9)))
                if cfg9.get("mode") != "ratio":       # 절대 기울기(기본): (최고값−최저값)/거리, 원본 구간
                    # 순수 기울기 = (최고값 − 최저값) ÷ 두 점 사이 원본 포인트 수(최고점이 앞이면 양수=정상)
                    def _mm9(Fx9, fx9, scale9):
                        """scale9: 인덱스 단위 → 원본 포인트 환산 배율(웨이퍼 F=원본 그대로 1.0, 골든=캐논→원본)."""
                        if f"a.r{r9}.ymax" not in fx9:
                            return np.nan, True
                        ymx9 = float(Fx9[fx9[f"a.r{r9}.ymax"]]); imx9 = float(Fx9[fx9[f"a.r{r9}.imax"]])
                        ymn9 = float(Fx9[fx9[f"a.r{r9}.y0"]]) - float(Fx9[fx9[f"a.r{r9}.drop"]])
                        imn9 = float(Fx9[fx9[f"a.r{r9}.imin"]])
                        flt9 = float(Fx9[fx9[f"a.r{r9}.flat"]]) > 0
                        span9 = abs(imn9 - imx9) * scale9
                        if not np.isfinite(ymx9) or flt9 or span9 < 1.0:
                            return 0.0, True
                        return (ymx9 - ymn9) / span9 * (1.0 if imx9 <= imn9 else -1.0), False
                    sp9, same9 = _mm9(Fw9, fidx9, 1.0)
                    sgp9, _sg_same9 = _mm9(gf9["F"][j9], gf9["fidx"], raw_med9 / max(1.0, float(l_r9))) if gf9 is not None else (np.nan, True)
                    if not np.isfinite(sp9):
                        out9.append(None); continue
                    if same9:                          # ★사용자 규칙: 최저값과 첫값이 같으면 무조건 상
                        lv9 = 2
                    else:
                        lv9 = 2 if sp9 <= t2 else (1 if sp9 <= t1 else 0)
                    if lv9 >= 1:
                        rr9 = 1.0 + (t1 - sp9) / max(abs(t1), 1e-6)
                    elif np.isfinite(sgp9) and sgp9 > t1:
                        rr9 = max(0.0, (sgp9 - sp9) / (sgp9 - t1))
                    else:
                        rr9 = 0.0
                    out9.append({"step": int(st9), "rule": "arch", "mode": "abs", "val": round(sp9, 4),
                                 "slope": round(sw9, 5), "gslope": (round(sgp9, 4) if np.isfinite(sgp9) else None),
                                 "flip": int(sp9 < 0), "same": int(same9), "imin": int(Fw9[fidx9[f"a.r{r9}.imin"]]),
                                 "imax": (int(Fw9[fidx9[f"a.r{r9}.imax"]]) if f"a.r{r9}.imax" in fidx9 else None),
                                 "r": round(float(rr9), 3), "lv": lv9})
                    continue
                if not np.isfinite(sg9) or abs(sg9) <= 1e-12 or not np.isfinite(dg9) or dg9 <= 0:
                    continue                           # 골든에 하강이 없으면 규칙 비활성
                flip9 = (sw9 * sg9 < 0)
                ratio9 = 0.0 if (flip9 or sw9 == 0.0) else abs(sw9) / abs(sg9)
                if cfg9["lr_fix"] and LRw9 is not None and r9 < len(LRw9) and np.isfinite(LRw9[r9]) and LRw9[r9] > 0:
                    ratio9 = ratio9 / float(LRw9[r9])
                lv9 = 2 if ratio9 <= t2 else (1 if ratio9 <= t1 else 0)
                out9.append({"step": int(st9), "rule": "arch", "mode": "ratio", "val": round(ratio9, 3),
                             "slope": round(sw9, 5), "gslope": round(sg9, 5), "flip": int(flip9),
                             "imin": int(Fw9[fidx9[f"a.r{r9}.imin"]]),
                             "r": round(t1 / max(ratio9, 1e-6), 3), "lv": lv9})
        return out9

    def _gv_verdict(ent9, k9, alpha9, beta9, astp9=None, asn9=None, acell9=None,
                    sel9=None):
        """일 캐시 항목 → 웨이퍼별 {ng, sensors:{sn:[NG 스텝]}, len:{}, na}.

        astp9={step:α}·asn9={sn:α}·acell9={'sn|step':α} — 민감도(센서×스텝 >
        센서 > 스텝 > 전역). sel9=점수에 반영할 센서(없으면 전부).
        """
        astp9 = astp9 or {}
        lvmap9 = GVD.get("p_lv") or {}                 # 스텝별 (중≥, 상≥) — gv_results 가 매번 갱신
        rules9 = GVD.get("p_rules") or _gv_parse_rules({})
        sens9 = GVD.get("run_sens") or []
        out9 = []
        A9 = ent9["A"]
        if A9.shape[0] == 0:
            return out9
        Ak9 = _gv_interp_k(A9, k9)                     # (n, nS, nR)
        gmap9 = _gv_gmap()
        F9 = ent9.get("F")
        fidx9 = {c9: i9 for i9, c9 in enumerate(ent9.get("fcols") or [])} if F9 is not None else {}
        mode_of9 = {}
        for s9 in sens9:                               # 센서별 규칙·모드(area|shape|max) 1회 결정
            rule9 = rules9["force"].get(s9) or _gv_rule_auto(s9)
            if s9 in rules9["off"] or rule9 not in ("wprom", "arch") or not fidx9:
                mode_of9[s9] = (None, "area")
            else:
                mode_of9[s9] = (rule9, "max" if rules9[rule9 if rule9 == "arch" else "w"]["combine"] else "shape")
        for w9 in range(A9.shape[0]):
            gk9 = ent9["gkey"][w9]
            g_w9 = gmap9.get(gk9)
            canon9 = g_w9["canon"] if g_w9 else None
            ngs9, lens9, na9, ir9, slv9 = {}, [], 0, {}, {}
            for j9, s9 in enumerate(sens9):
                if sel9 is not None and s9 not in sel9:  # 점수 미반영 센서(빈 목록=전부 미반영)
                    continue
                rule9, mode9 = mode_of9.get(s9, (None, "area"))
                bad9 = []; slv9[s9] = 0
                if mode9 in ("area", "max"):
                    for r9 in range(Ak9.shape[2]):
                        v9 = Ak9[w9, j9, r9]
                        if not np.isfinite(v9):
                            na9 += 1
                            continue
                        st9v = (canon9[r9][0] if canon9 else r9)
                        thr9 = _gv_thr(s9, st9v, alpha9, astp9, asn9, acell9)
                        rr9 = float(v9) / thr9 if thr9 > 0 else 0.0   # 강도비(임계 배수)
                        if rr9 > ir9.get(int(st9v), -1.0):
                            ir9[int(st9v)] = round(rr9, 3)
                        t1, t2 = lvmap9.get(int(st9v), (0.5, 1.0))
                        lv9 = 2 if rr9 >= t2 else (1 if rr9 >= t1 else 0)
                        if lv9 >= 1:                   # 중 이상 = 불량(사용자 규칙)
                            bad9.append({"step": st9v, "A": round(float(v9), 4),
                                         "r": round(rr9, 3), "lv": lv9, "rule": "area"})
                            if lv9 > slv9[s9]:
                                slv9[s9] = lv9
                if rule9 and canon9:                   # 모양 규칙(캐시 F 열 + 캐시 밖 임계)
                    for bd9 in _gv_shape_bads(rule9, rules9, F9[w9, j9], fidx9, canon9, gk9, sens9, j9,
                                              (ent9["LR"][w9] if ent9["LR"].ndim == 2 else None)):
                        if bd9 is None:
                            na9 += 1
                            continue
                        st9v = int(bd9["step"])
                        if bd9["r"] > ir9.get(st9v, -1.0):
                            ir9[st9v] = round(float(bd9["r"]), 3)
                        if bd9["lv"] >= 1:
                            bad9.append(bd9)
                            if bd9["lv"] > slv9[s9]:
                                slv9[s9] = bd9["lv"]
                if bad9:
                    ngs9[s9] = bad9
            for r9 in range(ent9["LR"].shape[1] if ent9["LR"].ndim == 2 else 0):
                lr9 = ent9["LR"][w9, r9]
                if np.isfinite(lr9) and abs(lr9 - 1.0) > beta9:
                    lens9.append({"step": (canon9[r9][0] if canon9 else r9),
                                  "ratio": round(float(lr9), 3)})
            out9.append({"id": ent9["ids"][w9], "gkey": gk9,
                         "ng": bool(ngs9), "sensors": ngs9,
                         "len_ng": lens9, "na": na9, "ir": ir9,
                         "lv": (max(slv9.values()) if slv9 else 0), "slv": slv9})
        return out9

    @app.post("/api/gv_results")
    def api_gv_results(payload: dict = Body(...)):
        """캐시에서 즉시 재판정(k·α·β 캐시 밖) → 추세 계열+웨이퍼 표."""
        if GVJOB["state"] == "running" and not GVD.get("restore_job"):
            raise HTTPException(409, "판정이 진행 중입니다 — 완료 후 다시 판정하세요")
        scores9 = dict(GVD.get("scores") or {})          # 스냅샷(작업 스레드 경합 방어)
        if not scores9:
            raise HTTPException(400, "판정 결과가 없습니다 — 먼저 판정을 실행하세요")
        try:
            k9 = float(payload.get("k", 2.0))
            alpha9 = float(payload.get("alpha", 0.05))
            beta9 = float(payload.get("beta", 0.3))
        except Exception:
            raise HTTPException(400, "k/α/β 값을 확인하세요(숫자)")
        agg9 = str(payload.get("agg") or "day")
        astp9, asn9, acell9, sel9 = _gv_parse_alpha(payload)
        GVD["p_lv"] = _gv_parse_levels(payload)
        GVD["p_rules"] = _gv_parse_rules(payload)
        vlm9 = dict(GVD.get("vlm") or {})
        mode9 = str(payload.get("mode") or "rule")
        rows9, buckets9 = [], {}
        t_v9 = time.time()
        for day9 in sorted(scores9):
            for v9 in _gv_verdict(scores9[day9], k9, alpha9, beta9, astp9,
                                  asn9, acell9, sel9):
                if mode9 in ("vlm", "hybrid"):
                    hit9 = vlm9.get(v9["id"])
                    if mode9 == "vlm":
                        v9["ng"] = bool(hit9 and hit9.get("ng"))
                    elif v9["ng"]:                     # hybrid: 룰 NG → VLM 재검
                        v9["ng"] = bool(hit9 is None or hit9.get("ng"))
                    if hit9:
                        v9["vlm"] = hit9
                v9["date"] = day9
                m9x = GVD["id2meta"].get(v9["id"]) or {}
                v9["eqp"] = str(m9x.get("eqp") or "")
                v9["dsv"] = str(m9x.get("dsv") or "")
                rows9.append(v9)
                if agg9 == "week":
                    dt9 = dt.date.fromisoformat(day9)
                    b9 = (dt9 - dt.timedelta(days=dt9.weekday())).isoformat()
                elif agg9 == "month":
                    b9 = day9[:7]
                else:
                    b9 = day9
                c9 = buckets9.setdefault(b9, [0, 0])
                c9[1 if v9["ng"] else 0] += 1
        series9 = [{"b": b9, "ok": c9[0], "ng": c9[1]}
                   for b9, c9 in sorted(buckets9.items())]
        ngs9 = [r9 for r9 in rows9 if r9["ng"]]
        srate9 = {}
        for r9 in ngs9:
            for s9 in r9["sensors"]:
                srate9[s9] = srate9.get(s9, 0) + 1
        rank9 = {}
        for r9 in ngs9:
            for s9, bads9 in r9["sensors"].items():
                for bd9 in bads9:
                    key9 = f"{s9} step{bd9['step']}"
                    rank9[key9] = max(rank9.get(key9, 0.0), float(bd9.get("A", bd9.get("r", 0.0))))
        t_v9 = time.time() - t_v9
        print(f"[gv] 재판정 {len(rows9)}장 {t_v9:.2f}s (규칙: W {(GVD.get('p_rules') or {}).get('w', {}).get('t1')}/"
              f"{(GVD.get('p_rules') or {}).get('w', {}).get('t2')} · 아치 {(GVD.get('p_rules') or {}).get('arch', {}).get('t1')}/"
              f"{(GVD.get('p_rules') or {}).get('arch', {}).get('t2')})", flush=True)
        return {"series": series9, "agg": agg9, "t_verdict": round(t_v9, 3),
                "n": len(rows9), "n_ng": len(ngs9),
                "sensor_missing": dict(GVD.get("sensor_missing") or {}),
                "sensor_rate": sorted(({"sensor": k2, "n": v2}
                                       for k2, v2 in srate9.items()),
                                      key=lambda x9: -x9["n"]),
                "rank": sorted(({"key": k2, "A": round(v2, 4)}
                                for k2, v2 in rank9.items()),
                               key=lambda x9: -x9["A"])[:20],
                "wafers": sorted(rows9, key=lambda r9: (not r9["ng"],
                                                        r9["date"], r9["id"]))}

    def _gv_remap_mats(canon_src9, canon_ref9):
        """스텝 순서가 같은 두 캐논 사이의 런별 선형 보간 행렬 [(p_src,l_src,q_ref,m_ref,M(m×l))]."""
        if [u9 for u9, _p, _l in canon_src9] != [u9 for u9, _p, _l in canon_ref9]:
            return None
        out9 = []
        for (_u9, p9, l9), (_u2, q9, m9) in zip(canon_src9, canon_ref9):
            l9, m9 = int(l9), int(m9)
            if l9 <= 1 or m9 <= 0:
                out9.append((int(p9), l9, int(q9), m9, None)); continue
            x9 = np.linspace(0, l9 - 1, m9)
            i09 = np.floor(x9).astype(int)
            fr9 = x9 - i09
            i19 = np.minimum(i09 + 1, l9 - 1)
            M9 = np.zeros((m9, l9))
            M9[np.arange(m9), i09] += 1 - fr9
            M9[np.arange(m9), i19] += fr9
            out9.append((int(p9), l9, int(q9), m9, M9))
        return out9

    def _gv_remap_apply(W9, mats9, L_ref9):
        """W (n,S,L_src) → (n,S,L_ref). NaN 은 가중 평균에서 제외(전부 NaN 이면 NaN)."""
        n9, S9 = W9.shape[0], W9.shape[1]
        out9 = np.full((n9, S9, int(L_ref9)), np.nan, dtype=np.float32)
        for p9, l9, q9, m9, M9 in mats9:
            if M9 is None:
                continue
            seg9 = W9[:, :, p9:p9 + l9].astype(np.float64)
            fin9 = np.isfinite(seg9)
            num9 = np.where(fin9, seg9, 0.0) @ M9.T
            den9 = fin9.astype(np.float64) @ M9.T
            with np.errstate(invalid="ignore", divide="ignore"):
                v9 = num9 / den9
            v9[den9 <= 1e-9] = np.nan
            out9[:, :, q9:q9 + m9] = v9.astype(np.float32)
        return out9

    @app.post("/api/gv_repwave")
    def api_gv_repwave(payload: dict = Body(...)):
        """대표 그래프: 현재 판정 기준으로 센서별 하/중/상 평균 파형(전량, 파형 캐시 소비).
        by='month' 면 달력 월 단위 버킷. ref=참조 4센서(웨이퍼 레벨 기준) 평균도 함께."""
        if GVJOB["state"] == "running" and not GVD.get("restore_job"):
            raise HTTPException(409, "판정이 진행 중입니다 — 완료 후 다시 시도하세요")
        scores9 = dict(GVD.get("scores") or {})
        if not scores9:
            raise HTTPException(400, "판정 결과가 없습니다 — 먼저 판정을 실행하세요")
        ids9 = {str(i9) for i9 in (payload.get("ids") or [])}
        try:
            k9 = float(payload.get("k", 2.0)); alpha9 = float(payload.get("alpha", 0.05))
            beta9 = float(payload.get("beta", 0.3))
        except Exception:
            raise HTTPException(400, "k/α/β 값을 확인하세요(숫자)")
        astp9, asn9, acell9, sel9 = _gv_parse_alpha(payload)
        GVD["p_lv"] = _gv_parse_levels(payload)
        GVD["p_rules"] = _gv_parse_rules(payload)
        by9 = "month" if str(payload.get("by") or "") == "month" else "all"
        t09 = time.time()
        rows_by_day9 = {}
        n_rows9 = 0
        for day9 in sorted(scores9):
            lst9 = [v9 for v9 in _gv_verdict(scores9[day9], k9, alpha9, beta9, astp9, asn9, acell9, sel9)
                    if not ids9 or v9["id"] in ids9]
            if lst9:
                rows_by_day9[day9] = lst9; n_rows9 += len(lst9)
        if not n_rows9:
            raise HTTPException(400, "대상 웨이퍼가 없습니다")
        sens9 = [s9 for s9 in (GVD.get("run_sens") or []) if sel9 is None or s9 in sel9]
        if not sens9:
            raise HTTPException(400, "반영 센서가 없습니다 — 센서를 체크하세요")
        ref9 = [s9 for s9 in GV_REF_SENSORS if s9 not in sens9]
        want9 = list(sens9) + ref9
        gmap9 = _gv_gmap()
        # 기준 격자 = 대상 웨이퍼가 가장 많은 (설비·레시피) 그룹의 캐논
        cnt9 = {}
        for lst9 in rows_by_day9.values():
            for v9 in lst9:
                g9 = gmap9.get(v9["gkey"])
                if g9:
                    cnt9[(g9["eqp"], g9["recipe"])] = cnt9.get((g9["eqp"], g9["recipe"]), 0) + 1
        if not cnt9:
            raise HTTPException(400, "웨이퍼에 맞는 골든이 없습니다")
        eqp9, rc9 = max(cnt9, key=cnt9.get)
        gref9 = gmap9.get(f"{eqp9}|{rc9}|") or next(g9 for g9 in gmap9.values() if g9["eqp"] == eqp9 and g9["recipe"] == rc9)
        canon_ref9 = [(int(u9), int(p9), int(l9)) for u9, p9, l9 in gref9["canon"]]
        L_ref9 = int(gref9["L"])
        wroot9 = _gv_wave_root()
        nW9, nL9 = len(want9), 3
        acc9 = {}                                      # bucket → {"S": (nW,3,L), "C": (nW,3,L), "n": int}
        def _acc9(b9):
            if b9 not in acc9:
                acc9[b9] = {"S": np.zeros((nW9, nL9, L_ref9)), "C": np.zeros((nW9, nL9, L_ref9)), "n": 0,
                            "ln": np.zeros((nW9, nL9), dtype=int)}
            return acc9[b9]
        n_wafers9 = 0; n_raw9 = 0; n_remap9 = 0; n_skip9 = 0
        mats_cache9 = {}
        for day9, lst9 in rows_by_day9.items():
            # 캐논(골든)별로 묶어 파형 캐시 읽기
            by_cs9 = {}
            for v9 in lst9:
                g9 = gmap9.get(v9["gkey"])
                if not g9:
                    n_skip9 += 1; continue
                canon9 = [(int(u9), int(p9), int(l9)) for u9, p9, l9 in g9["canon"]]
                cs9 = _gv_canon_sig(canon9, int(g9["L"]))
                by_cs9.setdefault(cs9, {"canon": canon9, "L": int(g9["L"]), "rows": []})["rows"].append(v9)
            for cs9, e9 in by_cs9.items():
                rows9 = e9["rows"]
                wids9 = [v9["id"] for v9 in rows9]
                fk_of9 = {}
                for w9 in wids9:
                    try:
                        stt9 = Path(GVD["id2src"].get(w9) or "").stat()
                        fk_of9[w9] = f"{stt9.st_mtime_ns}|{stt9.st_size}"
                    except Exception:                  # noqa: BLE001
                        fk_of9[w9] = "?"
                W9, hit9, miss9 = (None, [], list(wids9))
                if wroot9 is not None:
                    W9, hit9, miss9 = _gv_wave_load(Path(wroot9) / cs9, day9, wids9, want9, fk_of9)
                parts9 = []
                if W9 is not None and len(hit9):
                    parts9.append((hit9, W9))
                if miss9:                              # 캐시 밖 웨이퍼만 원본 읽기(+캐시 기록)
                    st2id9 = GVD.get("id2stem") or {}
                    items9 = [(w9, GVD["id2src"].get(w9), st2id9.get(w9, w9),
                               dict(GVD["id2meta"].get(w9) or _meta_of(S, w9))) for w9 in miss9]
                    raw9 = _gv_day_core(S, day9, [it9 for it9 in items9 if it9[1]], list(sens9), int(GVD.get("w") or 0),
                                        (GVD.get("build") or {}).get("goldens") or [], GVD.get("gref") or "dsv",
                                        (str(wroot9) if wroot9 else None), GV_REF_SENSORS, fk_of9)
                    n_raw9 += raw9["n"]
                    if raw9["n"] and wroot9 is not None:
                        W29, hit29, _m29 = _gv_wave_load(Path(wroot9) / cs9, day9, raw9["ids"], want9, fk_of9)
                        if W29 is not None and len(hit29):
                            parts9.append((hit29, W29))
                if not parts9:
                    n_skip9 += len(wids9); continue
                pos9 = {v9["id"]: v9 for v9 in rows9}
                if e9["canon"] != canon_ref9 or e9["L"] != L_ref9:
                    if cs9 not in mats_cache9:
                        mats_cache9[cs9] = _gv_remap_mats(e9["canon"], canon_ref9)
                    mats9 = mats_cache9[cs9]
                    if mats9 is None:
                        n_skip9 += len(wids9); continue
                for hid9, Wp9 in parts9:
                    if e9["canon"] != canon_ref9 or e9["L"] != L_ref9:
                        Wp9 = _gv_remap_apply(Wp9, mats9, L_ref9); n_remap9 += len(hid9)
                    # 레벨 one-hot: 판정 센서는 센서별 slv, 참조 센서는 웨이퍼 lv
                    lv_m9 = np.zeros((len(hid9), nW9), dtype=int)
                    for i9, w9 in enumerate(hid9):
                        v9 = pos9[w9]
                        slv9 = v9.get("slv") or {}
                        for jj9, s9 in enumerate(want9):
                            lv_m9[i9, jj9] = int(slv9.get(s9, v9.get("lv") or 0)) if s9 in sens9 else int(v9.get("lv") or 0)
                    oh9 = (lv_m9[:, :, None] == np.arange(3)[None, None, :]).astype(np.float64)   # (n,S,3)
                    Wd9 = Wp9.astype(np.float64)
                    fin9 = np.isfinite(Wd9)
                    W09 = np.where(fin9, Wd9, 0.0)
                    bkts9 = {}
                    for i9, w9 in enumerate(hid9):
                        bkts9.setdefault(("all" if by9 == "all" else str(pos9[w9].get("date") or day9)[:7]), []).append(i9)
                    for b9, idx9 in bkts9.items():
                        a9 = _acc9(b9)
                        a9["S"] += np.einsum("nsk,nsl->skl", oh9[idx9], W09[idx9])
                        a9["C"] += np.einsum("nsk,nsl->skl", oh9[idx9], fin9[idx9].astype(np.float64))
                        a9["ln"] += oh9[idx9].sum(axis=0).astype(int)
                        a9["n"] += len(idx9)
                    n_wafers9 += len(hid9)
        if not acc9:
            raise HTTPException(400, "웨이퍼 파형을 읽지 못했습니다")
        def _ser9(arr9):
            return [None if not np.isfinite(v9) else round(float(v9), 4) for v9 in arr9]
        gm9 = {}
        for s9 in want9:
            gv9 = (gref9.get("mean") or {}).get(s9)
            gm9[s9] = gv9
        buckets9 = []
        for b9 in sorted(acc9):
            a9 = acc9[b9]
            with np.errstate(invalid="ignore", divide="ignore"):
                M9 = a9["S"] / np.maximum(a9["C"], 1.0)
            M9[a9["C"] <= 0] = np.nan
            def _grp9(jj9):
                o9 = {}
                for k9, nm9 in enumerate(("lo", "mid", "hi")):
                    n_k9 = int(a9["ln"][jj9, k9])
                    o9[nm9] = _ser9(M9[jj9, k9]) if n_k9 else None
                    o9[nm9 + "_n"] = n_k9
                    o9[nm9 + "_used"] = n_k9
                return o9
            buckets9.append({"b": b9, "n": int(a9["n"]),
                             "sensors": {s9: dict(_grp9(jj9), g=gm9.get(s9)) for jj9, s9 in enumerate(want9) if s9 in sens9},
                             "ref": {s9: dict(_grp9(jj9), g=gm9.get(s9)) for jj9, s9 in enumerate(want9) if s9 in ref9}})
        print(f"[gv] 대표 그래프({by9}): 웨이퍼 {n_wafers9}장 · 원본 읽기 {n_raw9}장 · 타 격자 정합 {n_remap9}장 "
              f"· 제외 {n_skip9}장 · {time.time() - t09:.1f}s", flush=True)
        first9 = buckets9[0] if by9 == "all" else None
        return {"by": by9, "buckets": buckets9,
                "sensors": (first9["sensors"] if first9 else {}), "ref": (first9["ref"] if first9 else {}),
                "group": f"{eqp9} · {rc9}" + (f" 격자(타 설비 {n_remap9}장 정합)" if n_remap9 else ""),
                "L": L_ref9, "steps": [[u9, p9, l9] for u9, p9, l9 in canon_ref9],
                "n_wafers": n_wafers9, "n_raw": n_raw9, "n_group": n_wafers9, "ref_names": ref9}

    @app.get("/api/gv_wave_info")
    def api_gv_wave_info():
        """파형 캐시 용량(바이트)·파일 수."""
        d9 = _gv_wave_root()
        n9 = 0; b9 = 0
        if d9 is not None:
            for fp9 in Path(d9).rglob("*.npz"):
                try:
                    b9 += fp9.stat().st_size; n9 += 1
                except Exception:                      # noqa: BLE001
                    pass
        return {"bytes": b9, "files": n9, "dir": (str(d9) if d9 else "")}

    @app.post("/api/gv_wave_clear")
    def api_gv_wave_clear(payload: dict = Body(...)):
        """파형 캐시 비우기(판정 캐시는 유지 — 다음 판정 때 파형만 다시 기록)."""
        if GVJOB["state"] == "running":
            raise HTTPException(409, "판정이 진행 중입니다 — 완료 후 비우세요")
        d9 = _gv_wave_root()
        n9 = 0
        if d9 is not None:
            for fp9 in list(Path(d9).rglob("*.npz")):
                try:
                    fp9.unlink(); n9 += 1
                except Exception:                      # noqa: BLE001
                    pass
        return {"removed": n9}

    @app.post("/api/gv_golden")
    def api_gv_golden(payload: dict = Body(...)):
        """골든 갤러리: (설비,레시피)별 대표 mean/std + DSV별 mean + 자체 정화 제외 목록."""
        b9 = GVD.get("build") or {}
        if not b9.get("goldens"):
            raise HTTPException(400, "먼저 골든 산정을 실행하세요")
        groups9 = {}
        for g9 in b9["goldens"]:
            k9 = (g9["eqp"], g9["recipe"])
            gr9 = groups9.setdefault(k9, {"eqp": g9["eqp"], "recipe": g9["recipe"],
                                          "sensors": g9["sensors"], "canon": g9["canon"],
                                          "rep_mean": {}, "rep_std": {}, "rep_n": 0,
                                          "dsvs": []})
            if g9["dsv"] is None:
                gr9["rep_mean"] = g9["mean"]; gr9["rep_std"] = g9["std"]
                gr9["rep_n"] = g9["n"]
            else:
                gr9["dsvs"].append({"dsv": g9["dsv"], "n": g9["n"],
                                    "small": g9.get("small"), "mean": g9["mean"]})
        all_dsv9 = sorted({m9.get("dsv") for m9 in GVD["id2meta"].values()
                           if m9.get("dsv")}, key=_gv_natkey)
        for gr9 in groups9.values():
            have9 = {d9["dsv"] for d9 in gr9["dsvs"]}
            cnt9 = (b9.get("dsv_counts") or {}).get(f"{gr9['eqp']}|{gr9['recipe']}", {})
            gr9["dsvs"].sort(key=lambda d9: _gv_natkey(d9["dsv"]))
            gr9["missing"] = [{"dsv": d9, "n": int(cnt9.get(d9, 0)),
                               "reason": ("기간 내 0장" if cnt9.get(d9, 0) == 0
                                          else f"{cnt9.get(d9)}장 — 3장 미만")}
                              for d9 in all_dsv9 if d9 not in have9]
        return {"groups": list(groups9.values()),
                "auto_excluded": list(b9.get("auto_excluded_ids") or []),
                "excluded_n": b9.get("excluded_n"), "sig": b9.get("sig")}

    @app.post("/api/gv_vlm_png")
    def api_gv_vlm_png(payload: dict = Body(...)):
        """카드용: VLM 이 실제로 본 스텝 잔차 이미지(base64) — 판정 투명성."""
        import base64
        wid9 = str(payload.get("id") or ""); sn9 = str(payload.get("sensor") or "")
        st9 = int(payload.get("step") or 0)
        f9 = (GVD.get("id2src") or {}).get(wid9)
        m9 = GVD["id2meta"].get(wid9) if f9 else None
        g9, _ge9 = _gv_pick_golden(m9) if m9 else (None, None)
        if not (f9 and g9 and sn9 in (g9.get("mean") or {})):
            raise HTTPException(404, "이미지를 만들 수 없습니다(웨이퍼/골든/센서)")
        recs9, _a, _b, _c = _gv_load(
            Path(f9), stems=[(GVD.get("id2stem") or {}).get(wid9, wid9)])
        if not recs9:
            raise HTTPException(404, "웨이퍼 파일을 읽지 못했습니다")
        rec9 = recs9[0]
        cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
        if sn9 not in cols9:
            raise HTTPException(404, "센서가 이 웨이퍼에 없습니다")
        stw9 = np.asarray(rec9.get("step"), dtype=float).astype(int)
        canon9 = [(u9, p9, l9) for u9, p9, l9 in g9["canon"]]
        arr9 = np.asarray(rec9["sensors"], dtype=np.float64)[:, [cols9.index(sn9)]]
        wv9 = _warp_to_canon(arr9, stw9, canon9, int(g9["L"]))[:, 0]
        gm9 = np.asarray([np.nan if v9 is None else v9 for v9 in g9["mean"][sn9]],
                         dtype=np.float64)
        gs9 = np.asarray(g9["std"][sn9], dtype=np.float64)
        hit9 = [(r9, u9, p9, l9) for r9, (u9, p9, l9) in enumerate(canon9) if u9 == st9]
        if not hit9:
            raise HTTPException(404, "해당 스텝이 골든 캐논에 없습니다")
        _r9, u9, p9, l9 = hit9[0]
        x29 = wv9[p9:p9 + l9]; g29 = gm9[p9:p9 + l9]
        try:
            png9 = _gv_render_png(x29 - np.nanmean(x29) + np.nanmean(g29), g29,
                                  gs9[p9:p9 + l9],
                                  f"step {u9} | golden {g9['raw_len_med'][_r9]}pt | "
                                  f"normalized residual (band +/-3 sigma, clip +/-12)")
        except Exception as e9:                        # noqa: BLE001
            raise HTTPException(500, f"VLM 이미지 렌더 실패: {e9}")
        try:
            b649 = base64.b64encode(Path(png9).read_bytes()).decode()
        finally:
            try:
                os.unlink(png9)
            except OSError:
                pass
        return {"png": b649}

    @app.get("/api/gv_vlm_prompt")
    def api_gv_vlm_prompt():
        """③ 판정 섹션 '시스템 지시 보기'용 — 고정 시스템 지시 원문."""
        return {"system": _GV_VLM_SYS}

    def _gv_session_restore():
        """서버 재시작 후 첫 접속: 저장 세션으로 목록·골든·판정을 되살린다.

        목록=태깅 캐시로 빠름, 골든=저장 json 로드, 판정=일별 npz·VLM 캐시 적중이라
        수 초. 복원 중이면 True(클라이언트는 gv_status 를 폴링).
        """
        ss9 = _state_load().get("gv_session") or {}
        if GVD.get("id2date") or not ss9.get("list") or GVJOB["state"] == "running":
            return False
        gen09 = GVX["gen"]
        with GVX["lock"]:
            if GVX["gen"] != gen09 or GVD.get("id2date"):   # 그 사이 사용자 목록 요청
                return False
            try:
                _gv_do_list(dict(ss9["list"]))
            except Exception as e9:                    # noqa: BLE001
                print(f"[gv] 세션 복원(목록) 실패: {e9}", flush=True)
                return False
        if GVX["gen"] != gen09:                        # 목록 읽는 동안 사용자 요청 → 중단
            return False
        bld9 = ss9.get("build") or {}
        if bld9:
            try:                                       # 같은 서명의 저장 골든을 찾는다
                import glob as _gl9
                cands9 = sorted(_gl9.glob(str(Path(GVD["store"]) / "gv_golden_*.json")),
                                key=os.path.getmtime)
                for fp9 in reversed(cands9):
                    b9 = json.loads(Path(fp9).read_text(encoding="utf-8"))
                    if b9.get("avgwave_gv_golden") == 1 and b9.get("goldens"):
                        b9["file"] = Path(fp9).name
                        GVD["build"] = b9
                        GVD["build_sens"] = list(bld9.get("sensors") or
                                                 b9["goldens"][0]["sensors"])
                        break
            except Exception as e9:                    # noqa: BLE001
                print(f"[gv] 세션 복원(골든) 실패: {e9}", flush=True)
        run9 = ss9.get("run") or {}
        if GVX["gen"] != gen09:
            return False
        if run9 and GVD.get("build"):
            GVD["run_sens"] = [str(x9) for x9 in (run9.get("sensors") or [])] \
                or list(GVD.get("build_sens") or [])
            GVD["w"] = int(run9.get("w") or 0)
            GVD["mode"] = str(run9.get("mode") or "rule")
            GVD["gref"] = "rep" if str(run9.get("gref") or "") == "rep" else "dsv"
            GVD["p_k"] = float(run9.get("k") or 2.0)
            GVD["p_alpha"] = float(run9.get("alpha") or 0.05)
            GVD["p_astp"], GVD["p_asn"], GVD["p_acell"], GVD["p_sel"] = \
                _gv_parse_alpha(run9)
            GVD["vlm_cap"] = int(run9.get("vlm_cap") or 200)
            vn9 = run9.get("vlm_notes") or {}
            GVD["vlm_notes"] = {"common": str(vn9.get("common") or ""),
                                "steps": {str(k9): str(v9) for k9, v9 in
                                          (vn9.get("steps") or {}).items() if str(v9).strip()}}
            f9 = run9.get("facet") or {}
            GVD["facet"] = {k9: [str(v9) for v9 in (f9.get(k9) or [])]
                            for k9 in ("eqp", "dsv", "recipe")}
            GVD["scores"] = {}
            print("[gv] 서버 재시작 후 세션 복원 — 판정 캐시 재적재(백그라운드)", flush=True)
            GVD["restore_job"] = True
            _gv_job(_gv_scan_worker, str(run9.get("d0") or ""), str(run9.get("d1") or ""))
            return True
        return False

    @app.get("/api/gv_state")
    def api_gv_state_get():
        """재접속 복원: 저장 조건 + 현재 서버 메모리 상태(목록·골든·판정 유무).
        서버가 재시작돼 메모리가 비었으면 저장 세션으로 자동 복원을 시작한다."""
        restoring9 = _gv_session_restore()
        gvst9 = _state_load().get("gv") or {}
        cond9 = gvst9.get("cond")
        live9 = gvst9.get("live")
        listed9 = None
        if GVD.get("id2date"):
            listed9 = _gv_list_summary(len(GVD.get("anchors") or []))
        return {"cond": cond9, "live": live9, "listed": listed9,
                "has_build": bool((GVD.get("build") or {}).get("goldens")),
                "has_scores": bool(GVD.get("scores")),
                "restoring": bool(restoring9 or GVJOB["state"] == "running")}

    @app.post("/api/gv_state")
    def api_gv_state_set(payload: dict = Body(...)):
        with STATE_LOCK:                               # cond/live 부분 갱신이 서로를 지우지 않도록
            gv9 = dict(_state_load().get("gv") or {})
            if payload.get("save") is not None:        # 명시 저장(버튼·판정 실행)
                gv9["cond"] = payload.get("save") or {}
            if payload.get("live") is not None:        # 즉시 재판정 때의 현재 상태(재접속 복원용)
                gv9["live"] = payload.get("live") or {}
            _state_save("gv", gv9)
        return {"ok": True}

    @app.post("/api/gv_wafer")
    def api_gv_wafer(payload: dict = Body(...)):
        """웨이퍼 카드: 실측+DSV 골든+설비 대표 골든(캐논 정합) + 길이 배지."""
        wid9 = str(payload.get("id") or "")
        f9 = (GVD.get("id2src") or {}).get(wid9)
        if not f9:
            raise HTTPException(404, "웨이퍼를 찾을 수 없습니다")
        m9 = GVD["id2meta"].get(wid9) or _meta_of(S, wid9)
        g9, ge9 = _gv_pick_golden(m9)
        if g9 is None:
            raise HTTPException(400, "이 웨이퍼에 맞는 골든이 없습니다(설비/유형 확인)")
        recs9, _a, _b, _c = _gv_load(
            Path(f9), stems=[(GVD.get("id2stem") or {}).get(wid9, wid9)])
        if not recs9:
            raise HTTPException(404, "웨이퍼 파일을 읽지 못했습니다")
        rec9 = recs9[0]
        sens_all9 = GVD.get("run_sens") or GVD.get("build_sens") or []
        cols9 = [str(c9) for c9 in (rec9.get("sensor_cols") or [])]
        sens9 = list(sens_all9)
        arr9, stw9 = _gv_sensor_mat(rec9, sens9)
        if arr9 is None:
            raise HTTPException(400, "이 웨이퍼에 선택한 센서가 하나도 없습니다 — 원 센서명을 "
                                     "확인하세요(가상센서는 두 원 센서가 모두 있어야 계산)")
        canon9 = [(u9, p9, l9) for u9, p9, l9 in g9["canon"]]
        L9c = int(g9["L"])
        wv9 = _warp_to_canon(arr9, stw9, canon9, L9c)
        # ★표시 보정: 스텝 런 구성이 캐논과 다르면 워핑이 통째로 NaN 이 돼
        #   실측 선이 사라진다 — 정합 실패 구간만 전역 리샘플로 메워 항상
        #   그린다(데이터 시각화 탭 관례). 판정 경로에는 영향 없음(표시 전용).
        for j9 in range(wv9.shape[1]):
            bad9 = ~np.isfinite(wv9[:, j9])
            if bad9.any():
                wv9[bad9, j9] = _resample(arr9[:, j9], L9c)[bad9]
        occ9 = {}
        for u9, s09, e09 in _step_runs(stw9):
            occ9.setdefault(u9, []).append(e09 - s09)
        lens9, cnt9 = [], {}
        for r9, (u9, _p9, _l9) in enumerate(canon9):
            kx9 = cnt9.get(u9, 0)
            cnt9[u9] = kx9 + 1
            lst9 = occ9.get(u9) or []
            lens9.append({"step": int(u9),
                          "golden": int(g9["raw_len_med"][r9]),
                          "actual": (int(lst9[kx9]) if kx9 < len(lst9)
                                     else None)})
        def _ser9(gx9, s9):
            if gx9 is None or s9 not in (gx9.get("mean") or {}):
                return None
            return gx9["mean"][s9]
        out_s9 = {}
        def _pat9(arr9):
            """스텝별 평균 차감(판정이 보는 공간) — 결측은 유지."""
            o9 = np.array(arr9, dtype=np.float64)
            for u9, p9, l9 in canon9:
                seg9 = o9[p9:p9 + l9]
                fin9 = np.isfinite(seg9)
                if fin9.sum():
                    o9[p9:p9 + l9] = seg9 - float(np.mean(seg9[fin9]))
            return [None if not np.isfinite(v9) else round(float(v9), 4) for v9 in o9]
        for j9, s9 in enumerate(sens9):
            gm9 = np.asarray([np.nan if v9 is None else v9
                              for v9 in (g9.get("mean") or {}).get(s9) or []],
                             dtype=np.float64)
            gem9 = (np.asarray([np.nan if v9 is None else v9
                                for v9 in (ge9.get("mean") or {}).get(s9) or []],
                               dtype=np.float64) if ge9 is not None else None)
            out_s9[s9] = {
                "w": [None if not np.isfinite(v9) else round(float(v9), 4)
                      for v9 in wv9[:, j9]],
                "g": _ser9(g9, s9),
                "gs": (g9.get("std") or {}).get(s9),
                "ge": (_ser9(ge9, s9) if ge9 is not None else None),
                "wp": _pat9(wv9[:, j9]),
                "gp": (_pat9(gm9) if gm9.size else None),
                "gep": (_pat9(gem9) if gem9 is not None and gem9.size else None)}
        Aout9 = {}
        try:
            kk9 = float(payload.get("k", 2.0))
            rs9 = GVD.get("run_sens") or []
            for ent9 in (GVD.get("scores") or {}).values():
                if wid9 in ent9["ids"]:
                    w9i = ent9["ids"].index(wid9)
                    Ak9 = _gv_interp_k(ent9["A"], kk9)
                    for j9, s9 in enumerate(rs9):
                        Aout9[s9] = [[int(canon9[r9][0]),
                                      (None if not np.isfinite(Ak9[w9i, j9, r9])
                                       else round(float(Ak9[w9i, j9, r9]), 4))]
                                     for r9 in range(Ak9.shape[2])]
                    break
        except Exception:                              # noqa: BLE001 — 수치 없이도 카드는 그린다
            pass
        return {"id": wid9, "L": int(g9["L"]),
                "steps": [[int(u9), int(p9), int(l9)] for u9, p9, l9 in canon9],
                "golden": {"eqp": g9["eqp"], "recipe": g9["recipe"],
                           "dsv": g9["dsv"], "n": g9["n"],
                           "small": g9.get("small")},
                "gref": GVD.get("gref") or "dsv",
                "main_label": (("설비 통합 골든" if (GVD.get("gref") or "dsv") == "rep"
                                else "설비 통합 골든(이 챔버의 골든 없음 — 산정 대상 밖)")
                               if g9["dsv"] is None else f"챔버 골든({g9['dsv']})"),
                "guide_label": (None if ge9 is None else
                                ("설비 통합 골든" if ge9["dsv"] is None
                                 else f"챔버 골든({ge9['dsv']})")),
                "guide_n": (None if ge9 is None else ge9["n"]),
                "same_set": bool(ge9 is not None and ge9["n"] == g9["n"]),
                "lens": lens9, "sensors": out_s9, "A": Aout9}

    bind = os.environ.get("RCA_BIND", "127.0.0.1")     # 원격 접속 시 0.0.0.0
    port = int(os.environ.get("RCA_PORT", "0"))
    if port == 0:
        sock = socket.socket()
        sock.bind((bind, 0))
        port = sock.getsockname()[1]
        sock.close()
    url = f"http://127.0.0.1:{port}"
    if bind != "127.0.0.1":
        import subprocess
        try:
            ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True,
                                timeout=5).stdout.split()[0]
            print(f"[serve] 원격 접속: http://{ip}:{port}  (사무실 PC 브라우저에서)", flush=True)
        except Exception:                             # noqa: BLE001
            pass
    print(f"[serve] 도구 버전: {TOOL_VER} — 파일 교체 후에는 반드시 재시작해야 적용됩니다",
          flush=True)
    print(f"[serve] 분석 화면: {url}  (끝내려면 이 창에서 Ctrl+C)", flush=True)
    # 헤드리스 서버(H200 등)에선 브라우저 자동 열기를 시도하지 않는다 —
    #   xdg-open 부재로 ENOENT(errno 2) 소음이 나던 문제
    if os.environ.get("DISPLAY") and bind == "127.0.0.1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    else:
        print("[serve] (헤드리스/원격 모드 — 브라우저는 직접 열어 접속하세요)", flush=True)
    crit = os.environ.get("RCA_CRITIC_MODEL", "").strip()
    print(f"[llm] 비판자 토론: {'켜짐 — ' + crit if crit else '꺼짐 (RCA_CRITIC_MODEL 미설정 — 토론 없이 진행)'}",
          flush=True)
    threading.Thread(target=_ensure_dte, daemon=True).start()   # DTE 자동 기동(있으면)
    uvicorn.run(app, host=bind, port=port, log_level="warning")


def main():
    ap = argparse.ArgumentParser(description="레시피별 센서 평균 파형 리포트")
    ap.add_argument("--data", help="데이터셋 폴더(.log/.csv)")
    ap.add_argument("--model", help="모델 폴더(생략=앱 기본 모델)")
    ap.add_argument("--ok-max", type=float, default=0.8,
                    help="집계에 넣을 최대 정규화 점수(기본 0.8 — 임계의 80%% 이하만 '확실한 양품')")
    ap.add_argument("--out", help="--static 일 때 출력 HTML 경로")
    ap.add_argument("--dsv", help="골든 선정 DSV 필터(쉼표 다중, 예: 1,2,3 또는 dsv1,dsv2)")
    ap.add_argument("--recipe", help="골든 선정 유형 필터(부분일치·쉼표 다중, 예: 29a,521)")
    ap.add_argument("--months", help="골든 선정 월 필터(쉼표 다중, 예: 5,6,7 또는 2026-05)")
    ap.add_argument("--eqp", help="골든 선정 설비 필터(파일명 접두, 부분일치·쉼표 다중, "
                                  "예: WUK370A,WUK3702 또는 A,B)")
    ap.add_argument("--static", action="store_true",
                    help="서버 없이 '기준 파형만' 정적 HTML 로 저장(웨이퍼 분석 탭 없음)")
    a = ap.parse_args()
    data = a.data or input("데이터셋 폴더 경로: ").strip().strip('"')
    folder = Path(data)
    if not folder.is_dir():
        raise SystemExit(f"폴더가 없습니다: {folder}")

    # ★ 앱 코드(src) 위치 탐색: 스크립트 폴더 → 현재 폴더 → 상위 2단계.
    #   못 찾으면 어디에 둬야 하는지 바로 알려준다("No module named src" 방지).
    _cands = [Path(__file__).resolve().parent, Path.cwd()]
    # ★ parents[1:3] 슬라이스는 Python 3.10+ 전용(3.8/3.9 는 TypeError) — Win7=3.8 호환 유지
    # 상위 3단계까지: app/ 에서 직접 실행하면 루트(src|io_recon_tcn 의 부모)는
    # parents[1](=io_recon_tcn 의 부모)이 아니라 parents[2]다 — v116 에서 i<2 로
    # 좁혔다가 개발 트리 직접 실행이 깨졌던 회귀(E2E 감사가 검출).
    _cands += [p for i, p in enumerate(Path(__file__).resolve().parents) if i < 3]
    # 앱 코드 폴더 이름은 배포본(src) 과 개발본(io_recon_tcn) 두 가지 — 둘 다 인식한다.
    #   (예전엔 배포 시 이름을 바꿔 넣어야 했고, 그 변환을 빠뜨리면 여기서 죽었다)
    _root = _pkg = None
    for _c in _cands:
        for _p in ("src", "io_recon_tcn"):
            if (_c / _p / "app" / "server.py").exists():
                _root, _pkg = _c, _p
                break
        if _root is not None:
            break
    if _root is None:
        _here = Path(__file__).resolve().parent
        _run = ("평균파형_실행.bat" if os.name == "nt" else "./run_ubuntu.sh")
        raise SystemExit(
            "앱 코드 폴더(src)를 찾지 못했습니다.\n"
            f"  현재 위치: {_here}\n"
            "  → test_avg_waveform.py 는 'src' 폴더와 같은 위치에 있어야 합니다.\n"
            "    (압축을 푼 avg_waveform_tool 폴더 안에서 실행하세요)\n"
            f"  → 실행: {_run} --data <데이터셋 폴더>")
    sys.path.insert(0, str(_root))
    if _pkg != "io_recon_tcn":
        # 배포본은 폴더 이름이 src — 같은 import 경로(io_recon_tcn.*)로 쓰려면
        # sys.modules 별칭은 부모 프로세스에만 남아 병렬 로딩 자식이 죽는다.
        # → 디스크에 얇은 별칭 패키지를 만들어 어느 프로세스에서도 찾히게 한다.
        _al = _root / "io_recon_tcn"
        if not (_al / "__init__.py").exists():
            _al.mkdir(exist_ok=True)
            (_al / "__init__.py").write_text(
                "# 자동 생성(배포본 src/ 를 io_recon_tcn 이름으로 읽기 위한 별칭)\n"
                "import os\n"
                "__path__ = [os.path.join(os.path.dirname(os.path.dirname(\n"
                "    os.path.abspath(__file__))), %r)]\n"
                "from .config import IOConfig\n"
                "__all__ = ['IOConfig']\n"
                "def __getattr__(name):\n"
                "    if name in ('TCNForecastModel', 'TCNForecastDetector'):\n"
                "        from . import model_tcn_forecast as _m\n"
                "        return getattr(_m, name)\n"
                "    raise AttributeError(name)\n" % _pkg,
                encoding="utf-8")
            print(f"[setup] 배포 폴더 구조({_pkg}/) 인식 — io_recon_tcn 별칭 생성", flush=True)
    from io_recon_tcn.app import server as S          # 배포 앱과 동일 코드 재사용
    S._LOGGER_ON = True                               # 콘솔 출력 유지(파일 로그 전환 방지)
    _install_virtual(S)                               # ★ 가상센서(차분) 전 로드 경로 주입
    _install_parse_cache(S)                           # ★ 모든 탭 공통 파싱 캐시
    def _use_model(d):
        ok, why = S._probe_model_dir(d)
        if not ok:
            raise SystemExit(f"모델 점검 실패: {why}")
        S.ENGINE.select(d)

    if a.model:
        _use_model(a.model)
    else:
        try:
            S.ENGINE.load()                           # 앱 기본 모델이 있으면 그것부터
        except Exception:                             # noqa: BLE001
            pass
        if not S.ENGINE.ready:                        # 없으면 직접 묻는다(독립 실행 대비)
            mp = input("모델 폴더 경로 (search_best.json / .onnx 가 있는 폴더): ").strip().strip('"')
            if not mp:
                raise SystemExit("모델 폴더가 필요합니다 — 다시 실행해 경로를 입력하거나 --model 로 지정하세요.")
            _use_model(mp)
    if not S.ENGINE.ready:
        raise SystemExit(f"모델 로드 실패: {S.ENGINE.error}")
    model_name = Path(str(S.ENGINE.art)).name
    prov = getattr(S.ENGINE.model, "active_provider", None)
    print(f"[gpu] 추론 프로바이더: {prov or 'CPU'}"
          + ("" if prov and "CUDA" in str(prov) else
             "  (GPU 를 쓰려면 onnxruntime-gpu 설치 + IO_ORT_PROVIDER=cuda)"), flush=True)

    print(f"[1/3] 판정 중… (모델 {model_name} · ok-max {a.ok_max:g})")
    groups, stats = collect(S, folder, a.ok_max,
                            dsv=a.dsv, recipes=a.recipe, months=a.months, eqp=a.eqp)
    print(f"[2/3] 집계: 전체 {stats['total']} → 양품 {stats['ok']} · "
          f"애매 {stats['gray']} · 불량 {stats['ng']} · 보류 {stats['na']}")
    if a.static:                                      # 정적 리포트(기준 파형만) 파일로
        out = Path(a.out or f"avg_report_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.html")
        out.write_text(render_html(groups, stats, folder, a.ok_max, model_name), encoding="utf-8")
        print(f"[3/3] 완료: {out.resolve()}")
        try:
            import webbrowser
            webbrowser.open(out.resolve().as_uri())
        except Exception:                             # noqa: BLE001
            pass
        return
    # 기본: 로컬 서버 모드 — 브라우저에서 웨이퍼 폴더 입력·전체 목록·필터·겹침 분석
    print("[3/3] 분석 화면을 띄웁니다 — 웨이퍼 폴더는 브라우저에서 입력하세요")
    serve(S, groups, stats, folder, a.ok_max, model_name)


if __name__ == "__main__":
    main()
