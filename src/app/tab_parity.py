#!/usr/bin/env python3
"""탭 간 숫자 일치 검사 — 같은 웨이퍼·같은 날짜를 탭마다 같게 보여주는가.

parity_check.py 는 앱↔내보낸 HTML, runtime_parity.py 는 그 둘의 응답을 비교한다.
이 검사는 **앱 안에서 탭끼리** 어긋나는지 본다. 사용자가 Overview 에서 본 숫자와
이력 검색·개별 검사에서 본 숫자가 다르면, 그 자체로 착오가 된다.

    python3 tab_parity.py          # 불일치 있으면 종료코드 1
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILS: list[str] = []


def check(name: str, bad: list, note: str = "") -> None:
    if bad:
        print(f"  ❌ {name} — 불일치 {len(bad)}건 {note}")
        for b in bad[:8]:
            print(f"       {b}")
        if len(bad) > 8:
            print(f"       … 외 {len(bad) - 8}건")
        FAILS.append(name)
    else:
        print(f"  ✅ {name} {note}")


def main() -> int:
    os.environ.setdefault("IO_NO_BROWSER", "1")
    sys.path.insert(0, str(HERE))
    import server as S                                       # noqa: E402
    import runtime_parity as RP                              # noqa: E402

    rows = RP.make_history()
    for r in rows:                                           # 달력은 date 필드로 묶는다
        r["date"] = r["ts"][:10]
    S._read_history = lambda *a, **k: rows                    # type: ignore[assignment]
    S._recipe_decision_by_name = lambda rc: (1.0, 1.0)        # type: ignore[assignment]
    S._history_for_current_model = lambda *a, **k: S._relive_rows(rows)  # type: ignore[assignment]
    S._agg_cached = lambda *a, **k: None                      # type: ignore[assignment]
    S._agg_store = lambda k, p, n, res: res                   # type: ignore[assignment]
    S._archive_path = lambda d, w: Path("/nonexistent")       # type: ignore[assignment]
    now = dt.datetime(2026, 7, 8)
    S.dt.datetime = type("D", (dt.datetime,), {"now": staticmethod(lambda *a: now)})  # type: ignore

    print("탭 간 숫자 일치 — Overview · 이력 검색 · 개별 검사")
    print(f"  합성 이력 {len(rows)}매")

    sc = S.api_scatter(limit=1000000)
    pts = {p["wafer_id"]: p for p in sc.get("points", [])}
    dates = sorted({r["date"] for r in rows})
    month = dates[0][:7]
    cal = S.api_calendar(month=month)
    day_rows: dict[str, dict] = {}
    for d in dates:
        for w in S.api_calendar_day(date=d).get("wafers", []):
            day_rows[w["wafer_id"]] = {**w, "date": d}

    # ① 같은 웨이퍼의 '점수'가 산점도와 이력 검색에서 같은가
    bad = []
    for wid, p in pts.items():
        c = day_rows.get(wid)
        if not c or c.get("ratio") is None:
            continue
        if abs(float(p["ratio"]) - float(c["ratio"])) > 1e-9:
            bad.append(f"{wid}: 산점도 {p['ratio']} · 이력 검색 {c['ratio']}")
    check("웨이퍼 점수 (산점도 = 이력 검색)", bad, f"· 대상 {len(pts)}매")

    # ② 같은 웨이퍼의 '등급'을 같은 기준으로 읽는가
    #    이력 검색 색은 verdict + 0.8/1.0 고정 경계, 산점도는 등급 밴드(warn/crit).
    bands = (S.api_overview(rng="all", scale="day") or {}).get("bands") or {}
    bw, bc = float(bands.get("warn", 1.05)), float(bands.get("crit", 1.2))

    def grade(ratio: float) -> str:
        if ratio < 1:
            return "ok"
        if ratio < bw:
            return "caution"
        if ratio < bc:
            return "warn"
        return "crit"

    def cal_color(ratio: float, verdict: str, band: str = "") -> str:
        # 화면의 scoreColor 와 같은 규칙 — 서버 등급을 그대로 쓰고, 없으면 밴드로 계산.
        g = band or grade(ratio)
        return "good" if g == "ok" else "bad"

    GRADE2COLOR = {"ok": "good", "caution": "bad", "warn": "bad", "crit": "bad"}
    bad = []
    for wid, p in pts.items():
        c = day_rows.get(wid)
        if not c or c.get("ratio") is None:
            continue
        g = p.get("band") or grade(float(p["ratio"]))
        want = GRADE2COLOR.get(g, "good")
        got = cal_color(float(c["ratio"]), str(c.get("verdict") or ""), str(c.get("band") or ""))
        if want != got:
            bad.append(f"{wid}: 점수 {p['ratio']} 등급 {g} → 산점도 {want} · 이력 검색 {got}")
    check("웨이퍼 색 기준 (산점도 등급 = 이력 검색 색)", bad, f"· 대상 {len(pts)}매")

    # ③ 하루의 '불량 수'가 Overview 와 이력 검색에서 같은가
    ov = S.api_overview(rng="all", scale="day")
    bounds = [S._parse_ts(b) for b in ov.get("bounds", [])]
    day_of_cell = [b.strftime("%Y-%m-%d") if b else "" for b in bounds[:-1]]
    ov_day: dict[str, dict] = {}
    for lane in ov.get("lanes", []):
        for i, cell in enumerate(lane.get("cells") or []):
            if i >= len(day_of_cell):
                break
            acc = ov_day.setdefault(day_of_cell[i], {"total": 0, "bad": 0, "crit": 0})
            acc["total"] += cell.get("total") or 0
            acc["bad"] += cell.get("bad") or 0
            acc["crit"] += cell.get("crit") or 0
    bad_cnt, bad_tot = [], []
    for d in cal.get("days", []):
        o = ov_day.get(d["date"])
        if not o:
            continue
        if o["total"] != d["count"]:
            bad_tot.append(f"{d['date']}: Overview {o['total']}매 · 이력 검색 {d['count']}매")
        # ★ 달력의 '불량'은 개요 노출 숫자와 같은 '심각' 기준(2026-09-04 통일)
        if o["crit"] != d["ng"]:
            bad_cnt.append(f"{d['date']}: Overview 심각 {o['crit']} · 이력 검색 불량 {d['ng']}")
    check("하루 처리 매수 (Overview = 이력 검색)", bad_tot, f"· {len(cal.get('days', []))}일")
    check("하루 불량 수 (Overview 심각 = 이력 검색 불량)", bad_cnt, f"· {len(cal.get('days', []))}일")

    # ④ 목록 두 곳(웨이퍼 목록 · 이상 분석 목록)도 같은 점수를 보여주는가
    bad = []
    for r in S._relive_rows(rows):
        wid = r.get("wafer_id")
        p_ = pts.get(wid)
        if not p_:
            continue
        item = S._wt_item(r)
        if item.get("ratio") is None:
            continue
        if abs(float(p_["ratio"]) - float(item["ratio"])) > 1e-9:
            bad.append(f"{wid}: 산점도 {p_['ratio']} · 웨이퍼 목록 {item['ratio']}")
    check("웨이퍼 점수 (산점도 = 웨이퍼 목록)", bad, f"· 대상 {len(pts)}매")

    bad = []
    hr = S.api_history_range(start=dates[0], end=dates[-1])
    for it in (hr.get("wafers") or hr.get("items") or []):
        p_ = pts.get(it.get("wafer_id"))
        if not p_ or it.get("ratio") is None:
            continue
        if abs(float(p_["ratio"]) - float(it["ratio"])) > 1e-9:
            bad.append(f"{it['wafer_id']}: 산점도 {p_['ratio']} · 이상 분석 목록 {it['ratio']}")
    check("웨이퍼 점수 (산점도 = 이상 분석 목록)", bad, f"· 대상 {len(pts)}매")

    # ⑤ 표시 규약 — 점수는 '버림 2자리'(사용자 결정). 탭마다 자릿수·방식이 달라선 안 된다.
    bad = []
    for wid, c in day_rows.items():
        v = c.get("ratio")
        if v is None:
            continue
        if abs(float(v) * 100 - math.floor(float(v) * 100 + 1e-9)) > 1e-9:
            bad.append(f"{wid}: 이력 검색 점수 {v} (2자리 버림이 아님)")
    check("점수 표기 규약 (버림 2자리)", bad, f"· 대상 {len(day_rows)}매")

    print(f"\n검사 7건 · 불일치 {len(FAILS)}건")
    if FAILS:
        print("불일치: " + ", ".join(FAILS))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
