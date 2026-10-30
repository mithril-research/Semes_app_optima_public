#!/usr/bin/env python3
"""런타임 대조 — 같은 데이터를 넣고 '화면에 뜨는 숫자'를 통째로 비교한다.

parity_check.py 는 소스 문자열을 비교한다(공식이 같은 모양인지). 그것만으로는
"실제로 같은 숫자가 나오는가"를 보장하지 못한다. 이 스크립트는 한 벌의 이력을
만들어

    ① 앱(server.py)의 api_overview · api_scatter · api_dispersion 을 직접 호출
    ② 같은 웨이퍼로 내보내기 스냅샷(WAFERS)을 만들어 데모 HTML 의 mock 을 node 로 실행
    ③ 두 응답의 '숫자 필드'를 전부 재귀 비교

한다. 새로 생기는 불일치도 자동으로 걸린다(문자열 비교와 달리 항목을 미리
등록해 둘 필요가 없다).

    python3 runtime_parity.py            # 불일치 있으면 종료코드 1
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOL = 1e-6                                    # 부동소수 허용 오차
FAILS: list[str] = []


# ── 합성 이력 ────────────────────────────────────────────────────────
def make_history(n_days: int = 6) -> list[dict]:
    """재현 가능한 합성 이력. 등급이 골고루 나오도록 점수를 흩는다.

    난수를 쓰지 않는다 — 실패를 재현할 수 있어야 한다.
    """
    rows = []
    base = dt.datetime(2026, 7, 1, 8, 0, 0)
    recipes = ["RCP_ALPHA", "RCP_BETA"]
    # ★ 자정을 넘기는 LOT 을 일부러 섞는다 — 묶음을 '시작 시각'으로만 칸에 넣으면
    #   뒷날 칸이 통째로 비는 버그가 있었다. 합성 데이터에 그 조건이 없으면 대조가
    #   그냥 통과해 버린다(실제로 한 번 놓쳤다).
    k = 0
    for d in range(n_days):
        for ch_i in range(1, 7):
            for w in range(14):
                k += 1
                # 0.30 ~ 4.20 을 결정적으로 훑는다(양품·주의·경고·심각 전부 발생)
                sc = 0.30 + ((k * 37) % 40) * 0.0975
                # w>=12 는 자정을 넘겨 다음 날 새벽까지 이어지게 배치
                #   (base 가 08:00 이라 hours 를 더하면 날짜가 밀린다 → 자정 기준으로 계산)
                _mid = dt.datetime(base.year, base.month, base.day)
                ts = (base + dt.timedelta(days=d, hours=ch_i, minutes=w * 7)
                      if w < 12 else
                      _mid + dt.timedelta(days=d, hours=23, minutes=(w - 12) * 80))
                rows.append({
                    "ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "wafer_id": f"W{k:05d}",
                    "chamber": f"dsv{ch_i}",
                    "recipe": recipes[k % 2],
                    "score": round(sc, 4),
                    "score_raw": round(sc, 4),             # 재계산 경로가 쓰는 원점수
                    "threshold": 1.0,
                    "threshold_raw": 1.0,
                    "n_tp": 900 + (k % 200),
                    # 하루 마지막 묶음은 다음 날로 넘어가도록 같은 LOT 키를 준다
                    "lot": (f"LOT{d:02d}{ch_i}X" if w >= 12 else f"LOT{d:02d}{ch_i}{w // 4}"),
                })
    return rows


# ── 숫자 비교 ────────────────────────────────────────────────────────
def diff(path: str, a, b, out: list) -> None:
    """두 응답을 재귀 비교. 숫자는 허용 오차, 그 외는 동등성."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"{path}.{key}: 앱에 없음 (HTML={b[key]!r})")
            elif key not in b:
                out.append(f"{path}.{key}: HTML 에 없음 (앱={a[key]!r})")
            else:
                diff(f"{path}.{key}", a[key], b[key], out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: 길이 다름 앱={len(a)} HTML={len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(f"{path}[{i}]", x, y, out)
        return
    if isinstance(a, bool) or isinstance(b, bool):
        if bool(a) != bool(b):
            out.append(f"{path}: 앱={a} HTML={b}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(float(a) - float(b)) > TOL:
            out.append(f"{path}: 앱={a} HTML={b}")
        return
    if a is None and b is None:
        return
    if str(a) != str(b):
        out.append(f"{path}: 앱={a!r} HTML={b!r}")


def report(name: str, mismatches: list, ignored: int = 0) -> None:
    tail = f" (표시에 안 쓰는 필드 {ignored}개 제외)" if ignored else ""
    if mismatches:
        print(f"  ❌ {name} — 불일치 {len(mismatches)}건{tail}")
        for m in mismatches[:12]:
            print(f"       {m}")
        if len(mismatches) > 12:
            print(f"       … 외 {len(mismatches) - 12}건")
        FAILS.append(name)
    else:
        print(f"  ✅ {name}{tail}")


# ── 데모 mock 실행 ───────────────────────────────────────────────────
def run_mock(wafers: list, cfg: dict, now_iso: str, calls: list[str]) -> dict:
    """export_html.py 의 mock 을 node 에서 실행해 요청 목록의 응답을 받는다."""
    src = (HERE / "export_html.py").read_text(encoding="utf-8")
    i = src.find("<script>")
    j = src.find("</script>", i)
    js = src[i + len("<script>"):j]
    subs = {
        "__WAFERS__": json.dumps(wafers, ensure_ascii=False),
        "__DETAIL__": "{}",
        "__CFG__": json.dumps(cfg, ensure_ascii=False),
        "__NOW__": json.dumps(now_iso),
        "__SENSORS__": "[]",
        "__RECIPES__": json.dumps(cfg.get("recipes") or [], ensure_ascii=False),
        "__MULT__": "{}", "__MDEF__": "1", "__KNOWN_RECIPES__": "[]",
        "__UISENS__": "0.8", "__RETEN__": "0",
    }
    for a, b in subs.items():
        js = js.replace(a, b)
    js = re.sub(r"__[A-Z_]+__", "null", js)
    # 브라우저 전역 최소 구현 — mock 은 fetch 를 갈아끼우고 배너를 붙인다.
    shim = """
var __RESULTS__={}, __CALLS__=%s;
var document={addEventListener:function(_,f){ },createElement:function(){return {style:{},setAttribute:function(){},appendChild:function(){},addEventListener:function(){},removeAttribute:function(){},querySelectorAll:function(){return [];}};},
  body:{appendChild:function(){},style:{}},querySelectorAll:function(){return [];},querySelector:function(){return null;}};
var window={fetch:function(){ return Promise.reject(new Error('no network')); }};
var localStorage={getItem:function(){return null;},setItem:function(){}};
""" % json.dumps(calls)
    tail = """
(async function(){
  for (var i=0;i<__CALLS__.length;i++){
    var u=__CALLS__[i];
    var r=await window.fetch(u);
    __RESULTS__[u]=await r.json();
  }
  console.log(JSON.stringify(__RESULTS__));
})();
"""
    code = shim + js + tail
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:600])
        return json.loads(r.stdout)
    finally:
        os.unlink(path)


def main() -> int:
    os.environ.setdefault("IO_NO_BROWSER", "1")
    sys.path.insert(0, str(HERE))
    import server as S                                        # noqa: E402

    rows = make_history()
    # ★ _history_for_current_model 은 원래 _relive_rows 로 '현재 기준 재판정'을 거쳐 준다.
    #   그 단계를 건너뛰면 판정·등급이 비어 서버 쪽이 전부 양품으로 나온다(하네스 함정).
    S._read_history = lambda *a, **k: rows                     # type: ignore[assignment]
    S._history_for_current_model = lambda *a, **k: S._relive_rows(rows)   # type: ignore[assignment]
    S._agg_cached = lambda *a, **k: None                       # type: ignore[assignment]
    S._agg_store = lambda k, p, n, res: res                    # type: ignore[assignment]
    S._lot_of = lambda wid, row=None: (row or {}).get("lot") if row else next(  # type: ignore[assignment]
        (r["lot"] for r in rows if r["wafer_id"] == wid), None)

    # 모델이 없으므로 임계를 직접 고정한다 — 두 쪽 다 '임계 1.0' 으로 판정하게.
    #   (이 스크립트는 판정식이 맞는지 보는 게 아니라, 같은 입력에 같은 숫자가
    #    나오는지 보는 것이다. 임계는 두 쪽이 공유하는 상수여야 한다.)
    S._recipe_decision_by_name = lambda rc: (1.0, 1.0)         # type: ignore[assignment]
    S._recipe_base_decision = lambda wid: (                    # type: ignore[assignment]
        next((r["recipe"] for r in rows if r["wafer_id"] == wid), ""), 1.0, 1.0)
    S._recipe_threshold = lambda wid: (                        # type: ignore[assignment]
        next((r["recipe"] for r in rows if r["wafer_id"] == wid), ""), 1.0)

    now = dt.datetime(2026, 7, 8, 0, 0, 0)
    S.dt.datetime = type("D", (dt.datetime,), {"now": staticmethod(lambda *a: now)})  # type: ignore

    print("런타임 대조 — 같은 이력으로 앱과 내보낸 HTML 의 숫자를 비교")
    print(f"  합성 이력 {len(rows)}매 · 챔버 6 · 유형 2 · 6일")

    # ── 앱(서버) 응답 ──
    sc = S.api_scatter(limit=1000000)
    ov = S.api_overview(rng="all", scale="day")
    dp = S.api_dispersion(rng="all", scale="day", basis="excess", group_by="lot")

    # ── 내보내기와 '같은 방식'으로 스냅샷 생성 ──
    wafers = []
    for p in sc.get("points", []):
        ts = str(p.get("ts") or "")
        wafers.append({"wafer_id": p.get("wafer_id"), "chamber": p.get("chamber"),
                       "lot": p.get("lot"), "lot_first": bool(p.get("lot_first")),
                       "recipe": p.get("recipe"), "ratio": p.get("ratio"),
                       "score": p.get("score", p.get("ratio")), "verdict": p.get("verdict"),
                       "band": p.get("band"), "n_tp": p.get("n_tp"),
                       "ratio_x": p.get("ratio_x"), "ratio_base": p.get("ratio_base"),
                       "threshold": p.get("threshold", 1.0),
                       "ts": ts, "end_ts": ts, "date": ts[:10]})
    cfg = {"status": {"model_ready": True}, "now": now.isoformat(),
           "recipes": sorted({str(w["recipe"]) for w in wafers if w.get("recipe")}),
           "rule": S._display_rule()}

    calls = ["/api/overview?range=all&scale=day",
             "/api/scatter?limit=1000000",
             "/api/dispersion?range=all&scale=day&basis=excess&group_by=lot"]
    mock = run_mock(wafers, cfg, now.isoformat(), calls)

    # ── 비교: 화면에 실제로 뜨는 것만 ──
    # 요약 카드
    m = []
    diff("summary", ov.get("summary") or {}, (mock[calls[0]].get("summary") or {}), m)
    # ★ 최근 1시간 카드(데이터 기준)·챔버 실시간 상태도 같은 잣대로 비교
    diff("hour", ov.get("hour") or {}, (mock[calls[0]].get("hour") or {}), m)
    a_h = {c["chamber"]: c for c in (ov.get("hour_chambers") or [])}
    b_h = {c["chamber"]: c for c in (mock[calls[0]].get("hour_chambers") or [])}
    for ch in sorted(set(a_h) | set(b_h)):
        diff(f"hour_ch.{ch}", a_h.get(ch, {}), b_h.get(ch, {}), m)
    report("요약 카드(최근 1시간)·챔버 실시간 상태", m)

    # 챔버별 불량률 추이 — 칸 숫자·불량률
    m = []
    a_l = {l["chamber"]: l for l in (ov.get("lanes") or [])}
    b_l = {l["chamber"]: l for l in (mock[calls[0]].get("lanes") or [])}
    for ch in sorted(set(a_l) | set(b_l)):
        for fld in ("avgDefect",):
            diff(f"{ch}.{fld}", a_l.get(ch, {}).get(fld), b_l.get(ch, {}).get(fld), m)
        ca, cb = a_l.get(ch, {}).get("cells") or [], b_l.get(ch, {}).get("cells") or []
        if len(ca) != len(cb):
            m.append(f"{ch}.cells: 칸 수 앱={len(ca)} HTML={len(cb)}")
            continue
        for i, (x, y) in enumerate(zip(ca, cb)):
            for fld in ("total", "bad", "caution", "warn", "crit", "defectPct"):
                diff(f"{ch}.cells[{i}].{fld}", x.get(fld), y.get(fld), m)
    report("챔버별 불량률 추이 칸(총/불량/주의/경고/심각/불량률·챔버 평균)", m)

    # 챔버 실시간 상태 램프
    m = []
    a_c = {c["chamber"]: c for c in (ov.get("chambers") or [])}
    b_c = {c["chamber"]: c for c in (mock[calls[0]].get("chambers") or [])}
    for ch in sorted(set(a_c) | set(b_c)):
        for fld in ("level", "total", "crit", "warn", "caution"):
            diff(f"{ch}.{fld}", a_c.get(ch, {}).get(fld), b_c.get(ch, {}).get(fld), m)
    report("챔버 실시간 상태(등급·총/심각/경고/주의)", m)

    # 등급 배수
    m = []
    diff("bands", ov.get("bands"), mock[calls[0]].get("bands"), m)
    report("등급 배수(경고·심각 배수)", m)

    # 산점도 — 점마다 점수·등급·판정
    m = []
    pa = {p["wafer_id"]: p for p in (sc.get("points") or [])}
    pb = {p["wafer_id"]: p for p in (mock[calls[1]].get("points") or [])}
    if set(pa) != set(pb):
        m.append(f"웨이퍼 집합 다름 앱={len(pa)} HTML={len(pb)}")
    for wid in sorted(set(pa) & set(pb)):
        for fld in ("ratio", "band", "verdict", "threshold", "lot", "lot_first"):
            diff(f"{wid}.{fld}", pa[wid].get(fld), pb[wid].get(fld), m)
    report(f"산점도 점({len(pa)}개 · 점수·등급·판정·임계·LOT·첫장)", m)

    # 산포 추세 — 칸 값
    m = []
    a_d = {l["chamber"]: l for l in (dp.get("lanes") or [])}
    b_d = {l["chamber"]: l for l in (mock[calls[2]].get("lanes") or [])}
    for ch in sorted(set(a_d) | set(b_d)):
        ca, cb = a_d.get(ch, {}).get("cells") or [], b_d.get(ch, {}).get("cells") or []
        if len(ca) != len(cb):
            m.append(f"{ch}.cells: 칸 수 앱={len(ca)} HTML={len(cb)}")
            continue
        for i, (x, y) in enumerate(zip(ca, cb)):
            for fld in ("n_groups", "ratio", "ratio_mean", "rate", "mean_ex"):
                diff(f"{ch}.cells[{i}].{fld}", x.get(fld), y.get(fld), m)
    report("산포 추세 칸(그룹 수·값·초과율·평균 초과량)", m)

    print(f"\n비교 6개 묶음 · 불일치 {len(FAILS)}건")
    if FAILS:
        print("불일치: " + ", ".join(FAILS))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
