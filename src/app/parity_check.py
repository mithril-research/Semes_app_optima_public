#!/usr/bin/env python3
"""앱(server.py) ↔ 내보낸 HTML(export_html.py) 공식 대조 검사.

왜 필요한가
-----------
같은 숫자를 세 곳에서 따로 구현하고 있다.

    server.py        검사·집계의 원본 계산 (Python)
    static/index.html  화면에서 다시 하는 계산 (JS)   — 등급 보기 재집계 등
    export_html.py     내보낸 HTML 안의 재계산 (JS)   — 서버가 없으므로 전부 재현

한 곳을 고치고 나머지를 안 고치면, 같은 데이터인데 화면마다 다른 숫자가 나온다.
실제로 그런 사고가 반복됐다(챔버 램프가 앱에서는 비율 기준인데 내보낸 HTML 은
'1장이면 빨강'인 옛 규칙으로 남아 있었다). 사람이 기억으로 막을 일이 아니라
기계가 매번 대조해야 한다.

쓰는 법
-------
    python3 parity_check.py          # 전부 검사, 불일치 있으면 종료코드 1

새 공식을 세 곳에 나눠 구현했다면 여기에 대조 항목을 추가한다.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILS: list[str] = []
CHECKS = 0


def _js_fn(src: str, name: str) -> str:
    """소스에서 function name(...){...} 한 덩어리를 중괄호 균형으로 잘라 온다."""
    i = src.find("function " + name + "(")
    if i < 0:
        raise LookupError(f"{name} 을 찾지 못했습니다")
    depth, k = 0, src.index("{", i)
    while True:
        c = src[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
        k += 1


def _run_node(code: str):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:400])
        return json.loads(r.stdout)
    finally:
        os.unlink(path)


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}   {detail}")
        FAILS.append(name)


# ─────────────────────────────────────────────────────────────────────
def check_chamber_lamp(server, export_src: str) -> None:
    """챔버 실시간 상태 램프 — server.py _chamber_level ↔ mock _chLevel"""
    js = _js_fn(export_src, "_chLevel") + """
var out=[];
for (var t of [0,1,3,5,10,19,20,21,25,100,500])
  for (var g of [0,1,2,3,5,25,100])
    for (var m of [0,0.5,0.79,0.80,0.89,0.90,0.95,1.4]) {
      if (g>t) continue; out.push([t,g,m,_chLevel(t,g,m)]); }
console.log(JSON.stringify(out));
"""
    rows = _run_node(js)
    bad = [(t, g, m, server._chamber_level(t, g, m), lv)
           for t, g, m, lv in rows if server._chamber_level(t, g, m) != lv]
    check(f"챔버 램프 5단계 ({len(rows)}조합)", not bad,
          f"불일치 {len(bad)}개 예: {bad[:3]}")


def check_bands(server, export_src: str) -> None:
    """등급 배수 자동 산출 — exp(σ) / exp(2σ) 와 상·하한 클램프"""
    i = export_src.find("out.bands={warn:")
    j = export_src.find("}", i)
    src = export_src[i:j]
    ok = ("Math.exp(mad)" in src and "Math.exp(2*mad)" in src
          and "1.05" in src and ",20)" in src and "1.1" in src and ",40)" in src)
    py = Path(HERE / "server.py").read_text(encoding="utf-8")
    pyok = ('warn = float(np.exp(sig)); crit = float(np.exp(2 * sig))' in py
            and 'min(max(warn, 1.05), 20.0)' in py and 'min(max(crit, 1.1), 40.0)' in py)
    check("등급 배수 exp(σ)·exp(2σ) + 클램프(1.05~20 / 1.1~40)", ok and pyok,
          f"mock={ok} server={pyok}")


def check_dispersion(server, export_src: str) -> None:
    """산포 추세 — 그룹 최소 장수 규칙과 버킷 집계 함수(평균/중앙값)"""
    py = Path(HERE / "server.py").read_text(encoding="utf-8")
    # ① 초과 강도는 1장부터, 흩어짐은 2장부터
    py_need = '_need = 1 if basis in ("excess", "excess2") else max(2, int(min_n))' in py
    js_need = "var need=(basis==='excess'||basis==='excess2')?1:Math.max(2,minN);" in export_src
    check("산포 그룹 최소 장수(초과 강도 1장 · 흩어짐 2장)", py_need and js_need,
          f"server={py_need} mock={js_need}")
    # ② 버킷 칸 집계 = 초과 강도만 평균, 나머지 중앙값
    py_agg = '_agg = np.max if basis in ("excess", "excess2") else np.median' in py
    js_agg = ("var _agg=(basis==='excess'||basis==='excess2')" in export_src
              and "if(a[i]>m) m=a[i]" in export_src)
    check("산포 칸 집계(초과 강도=가장 심한 LOT · 그 외=중앙값)", py_agg and js_agg,
          f"server={py_agg} mock={js_agg}")
    # ③ 초과 기준선 = 심각 배수(임계 1.0 아님)
    py_base = ('_ebase = max(1.0, float(_bands.get("crit") or 1.2))' in py
               and 'ex = np.maximum(0.0, a - float(base))' in py)
    js_base = ("ebase=Math.max(1,(+_cb)||1.2)" in export_src
               and "Math.max(0,v-b)" in export_src)
    check("산포 초과 기준선(위험=심각 배수)", py_base and js_base,
          f"server={py_base} mock={js_base}")
    # ④ LOT 묶음은 구간(t0~t1) — 겹치는 칸·조각에 모두 넣는다(자정 넘긴 LOT 대비)
    py_ovl = ('pts.append((_a, _b, g["ratio"]))' in py
              and 'if a < b1 and b >= b0' in py)
    js_ovl = ("g.t<s1 && g.te>=s0" in export_src and "t1:cur[cur.length-1].t" in export_src)
    check("산포 묶음 배치(구간 겹침 기준)", py_ovl and js_ovl,
          f"server={py_ovl} mock={js_ovl}")
    # ⑤ 같은 LOT 해시라도 '한 번 돌린 구간'마다 나눈다.
    #   통째로 합치면 묶음이 첫 구간 칸에만 놓여, 그 LOT 이 다시 돈 날짜가 전부
    #   '데이터 없음'이 된다(산점도엔 점이 찍히는데 산포 칸만 빈칸 — 실제 사고).
    py_run = ('def _lot_runs(' in py
              and 'for lot, run in _lot_runs(items, gap_min, "lot"):' in py)
    js_run = ("if(g2>gapMin || !same){ runs.push([runLot,run]); run=[]; runLot=lot; }"
              in export_src)
    check("산포 LOT 재실행 분리(같은 해시도 구간마다 별도 묶음)", py_run and js_run,
          f"server={py_run} mock={js_run}")
    # ⑥ 산포 계산과 '첫 장 흐리게'가 같은 묶음을 봐야 한다. 따로 묶으면 계산에서 뺀
    #   장과 화면에서 흐려진 장이 어긋난다(LOT 이 이틀에 걸치면 실제로 어긋났다).
    py_share = py.count('_lot_runs(') >= 3
    check("첫 장 표시 = 산포 묶음(_lot_runs 공용)", py_share,
          f"server 호출 {py.count('_lot_runs(')}회")


def check_index_single_source(index_src: str) -> None:
    """같은 숫자를 화면 안에서 두 잣대로 세고 있지 않은지."""
    # 칸 툴팁 · 클릭 상세 둘 다 '심각' 기준이어야 한다(도움말에 그렇게 적혀 있다).
    tip_crit = "var _tot=c.total||0, _crit=c.crit||0" in index_src
    drill_crit = "var total=c.total||0, crit=c.crit||0" in index_src
    check("불량 수 잣대 통일(칸 툴팁 = 클릭 상세 = 심각 기준)", tip_crit and drill_crit,
          f"툴팁={tip_crit} 클릭상세={drill_crit}")
    # 산점도에는 추세 표시를 두지 않는다(사용자 결정) — 불량률 추이 칸과 재는 것도
    # 기간도 달라 반대로 보일 때 오해만 낳았다. 배지·추세선·계산 함수 모두 없어야 한다.
    leftovers = [k for k in ("_lsqTrend", "점수 추세", "tdir") if k in index_src]
    check("산점도에 추세 표시 없음(배지·추세선·계산 함수)", not leftovers,
          f"남아 있음: {leftovers}")


def check_mock_endpoints(index_src: str, export_src: str) -> None:
    """화면이 부르는 API 를 내보낸 HTML 의 mock 이 전부 흉내내는지.

    mock 은 url.indexOf('/api/xxx')===0 로 '접두' 매칭한다. 따라서 문자열이 그대로
    들어 있는지 보는 것으로는 안 되고(접두 하나가 여러 경로를 받는다), 실제 라우팅과
    같은 방식으로 대조해야 한다.
    """
    called = set(re.findall(r"['\"](/api/[a-z0-9\-_/]+)", index_src))
    called |= set(re.findall(r"base\+'(/[a-z0-9\-_/]+)", index_src))
    # mock 이 등록한 접두 목록을 그대로 뽑는다.
    prefixes = set(re.findall(r"url\.indexOf\('(/[^']+)'\)===0", export_src))
    prefixes |= set(re.findall(r"sub\.indexOf\('(/[^']+)'\)===0", export_src))
    # 정규식으로 받는 경로도 있다(예: /\/api\/archive\/(\d{4}-..)/ ) — 접두로 환산.
    for m in re.findall(r"\\/api\\/([a-z0-9\-_]+)", export_src):
        prefixes.add("/api/" + m)
    # href 다운로드 링크는 브라우저가 서버로 직접 요청하므로 fetch mock 이 가로챌 수 없다.
    # 데모는 이런 링크를 통째로 무력화하고 안내를 띄운다(아래 별도 검사).
    HREF_ONLY = {"/api/predictions"}
    missing = sorted(ep for ep in (called - HREF_ONLY)
                     if not any(ep.startswith(p) for p in prefixes))
    check(f"mock 이 흉내내는 화면 API ({len(called)}종 · mock 접두 {len(prefixes)}개)",
          not missing, f"빠짐: {missing[:8]}")
    check("데모가 href 다운로드 링크를 무력화",
          'querySelectorAll(\'a[href^="/api/"]\')' in export_src)


def main() -> int:
    os.environ.setdefault("IO_NO_BROWSER", "1")
    sys.path.insert(0, str(HERE))
    import server                                            # noqa: E402

    export_src = (HERE / "export_html.py").read_text(encoding="utf-8")
    index_src = (HERE / "static" / "index.html").read_text(encoding="utf-8")

    print("앱 ↔ 내보낸 HTML 공식 대조")
    check_chamber_lamp(server, export_src)
    check_bands(server, export_src)
    check_dispersion(server, export_src)
    print("화면 내부 잣대 일치")
    check_index_single_source(index_src)
    check_mock_endpoints(index_src, export_src)

    print(f"\n검사 {CHECKS}건 · 불일치 {len(FAILS)}건")
    if FAILS:
        print("불일치: " + ", ".join(FAILS))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
