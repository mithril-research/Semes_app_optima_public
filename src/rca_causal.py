# -*- coding: utf-8 -*-
"""RCA 인과 규칙 엔진 + LLM 서술 생성 (우분투/H200 · ollama+Qwen3.6).

설계 원칙(2026-09-08 전략 합의):
  · 추론은 '규칙'이 하고, LLM 은 '작문'만 한다(환각 차단).
  · 규칙은 설비 제어 구조(사용자 교정 반영)를 따른다:
      - H 라인 히터: 자체 고정 설정, 피드백 없음 → H 이탈 = 그 히터 자체 문제(전파 아님)
      - Vessel 캐스케이드: Cart(열원 PV)가 Heater(존 외부온도=목표)를 맞추도록 제어
          Cart↑ & Heater↓ = 열전달 불량/존 열손실  ·  Cart↓ & Heater↓ = 열원 출력 부족
          Cart↑ & Heater↑ = 존 과열
      - 가스 온도(T)·압력(P)만 유로를 따라 전파 — 인과 체인은 T/P 로만 잇는다
      - scCO2: 과열 → 압력↑ → 벤트 개도↑ → 팽창(줄-톰슨) 냉각이 배기측 저온으로 나타남
  · 출력 = 구조화 증거 패킷(JSON) → Qwen 이 공정 엔지니어용 한국어 설명으로 변환.
"""
from __future__ import annotations

from .sensor_schema import in_group, paired_zone

import json
import os
import re
import urllib.request

import numpy as np

Z_SIG = 3.0          # 유의 이탈(σ)
CART_SIG_C = 3.0     # 카트리지-존 갭 유의(℃)


# ── 센서 분류 ────────────────────────────────────────────────────────
def kind_of(name: str) -> str:
    if in_group(name, "upper_cart") or in_group(name, "lower_cart"):
        return "CART"
    if in_group(name, "response"):
        return "ZONE"
    n = str(name).lower()
    if n.startswith("t_"):
        return "T"
    if n.startswith("p_") or n.endswith("_press"):
        return "P"
    return "H"


# ── 설비 구간(유로 순) — "어느 구간이 문제인가" 결론용 ──────────────
_SEGMENTS = []  # Segment assignments use anonymous identifiers below.


def segment_of(name: str) -> str:
    return {'s_012_pv': '기타', 'h_001_pv': 'Back 하부 유입', 'h_002_pv': 'Back 하부 유입', 's_003_press': '배기/벤트', 's_011_pv': '기타', 'd_001_pv': '챔버 주변(배기/실린더)', 'd_002_pv': '챔버 주변(배기/실린더)', 'd_003_pv': '챔버 주변(배기/실린더)', 'd_004_pv': '챔버 주변(배기/실린더)', 'd_005_pv': '챔버 주변(배기/실린더)', 'd_0061_pv': '챔버 주변(배기/실린더)', 'd_0062_pv': '챔버 주변(배기/실린더)', 'h_030n_pv': '기타', 's_010_pv': '기타', 'h_003_pv': 'Front 상부 유입', 'h_004_pv': 'Front 상부 유입', 's_002_press': '배기/벤트', 'h_0052_pv': '공급 라인(예열)', 'h_0053_pv': 'Front 상부 유입', 'h_005_pv': '공급 라인(예열)', 'n_pv': '기타', 'h_006_pv': '배기/벤트', 'p_test_pv': '기타', 'p_002_pv': '공급 라인(예열)', 'p_003_pv': '배기/벤트', 'p_004_pv': '배기/벤트', 'p_006_pv': '공급 라인(예열)', 'p_005_pv': '공급 라인(예열)', 'p_007_pv': 'Back 하부 유입', 'p_008_pv': 'Front 상부 유입', 'p_008_sv': 'Front 상부 유입', 'p_sensor_sv': '기타', 'h_0072_pv': '충전 미터링', 'h_007_pv': '충전 미터링', 'h_009_pv': 'Reservoir(저장조)', 'h_008_pv': 'Reservoir(저장조)', 'h_010_pv': 'Reservoir(저장조)', 'h_0111_pv': 'Reservoir(저장조)', 'h_011_pv': 'Reservoir(저장조)', 'h_012_pv': 'Reservoir(저장조)', 'h_013_pv': '충전 미터링', 'h_0141_pv': '배기/벤트', 'h_015_pv': '공급 라인(예열)', 'h_0161_pv': '공급 라인(예열)', 'h_0162_pv': 'Front 상부 유입', 's_001_press': '배기/벤트', 't_001_pv': 'Back 하부 유입', 't_002_pv': '공급 라인(예열)', 't_test_pv': '기타', 't_003_pv': '배기/벤트', 't_004_pv': '배기/벤트', 't_005_pv': '공급 라인(예열)', 'h_0181_pv': 'Vessel 하부 존', 'h_0182_pv': 'Vessel 하부 존', 'h_0183_pv': 'Vessel 하부 존', 'h_0184_pv': 'Vessel 하부 존', 'h_0191_pv': 'Vessel 하부 존', 'h_0192_pv': 'Vessel 하부 존', 'h_0193_pv': 'Vessel 하부 존', 'h_0194_pv': 'Vessel 하부 존', 'h_0211_pv': 'Vessel 상부 존', 'h_0212_pv': 'Vessel 상부 존', 'h_0213_pv': 'Vessel 상부 존', 'h_0214_pv': 'Vessel 상부 존', 'h_0221_pv': 'Vessel 상부 존', 'h_0222_pv': 'Vessel 상부 존', 'h_0223_pv': 'Vessel 상부 존', 'h_0224_pv': 'Vessel 상부 존', 'th_004_pv': 'Front 상부 유입', 'date': '기타', 'time': '기타', 'processstep': '기타'}.get(str(name).lower(), "기타")


def _sig(v, thr):
    return v is not None and np.isfinite(v) and abs(v) >= thr


def _first_sustained(mask, k=3):
    run = 0
    for t, b in enumerate(mask):
        run = run + 1 if b else 0
        if run >= k:
            return t - k + 1
    return None


# ── 엔진 본체 ────────────────────────────────────────────────────────
def analyze(wid, recipe, g, data, pred, steps_rs, ranks):
    """웨이퍼 1장 → 증거 패킷.

    g: 골든 그룹 {sensors, mean(L,S), std(L,S), n}
    data/pred: {sensor: [L]}  · steps_rs: (L,) int  · ranks: {sensor: 설치순위}
    """
    L = len(steps_rs)
    usteps = sorted(set(int(v) for v in steps_rs))
    sensors = [s for s in g["sensors"] if s in data]
    obs = {}                                           # sensor → 관측 요약
    for s in sensors:
        j = g["sensors"].index(s)
        w = np.asarray(data[s], dtype=float)
        mu, sg = g["mean"][:, j], g["std"][:, j]
        floor = max(1e-6, 0.05 * float(np.median(sg)), 1e-3 * float(np.mean(np.abs(mu))))
        z = np.clip((w - mu) / np.maximum(sg, floor), -99, 99)
        dev_c = w - mu                                 # ℃(원단위) 부호 편차
        per_step = {}
        for st in usteps:
            m = steps_rs == st
            if not np.any(m):
                continue
            k2 = int(np.argmax(np.abs(z[m])))
            per_step[st] = {"z": round(float(z[m][k2]), 1),
                            "dev": round(float(dev_c[m][k2]), 2)}
        onset = _first_sustained(np.abs(z) > Z_SIG)
        zr_on = None
        if s in pred:
            rr = w - np.asarray(pred[s], dtype=float)
            dif = np.diff(rr)
            mad = float(np.median(np.abs(dif))) * 1.4826 / np.sqrt(2.0) if dif.size else 0.0
            if mad < 1e-9:
                mad = float(np.median(np.abs(rr - np.median(rr)))) * 1.4826
            rscale = max(mad, 1e-6, 1e-4 * float(np.mean(np.abs(mu))))
            zr = np.clip((rr - np.median(rr)) / rscale, -99, 99)
            zr_on = _first_sustained(np.abs(zr) > Z_SIG)
        worst = max(per_step.values(), key=lambda x: abs(x["z"])) if per_step else None
        obs[s] = {"kind": kind_of(s), "rank": ranks.get(s, 10**6),
                  "segment": segment_of(s),
                  "per_step": per_step, "worst": worst,
                  "onset": onset, "onset_model": zr_on,
                  "sig": bool(worst and abs(worst["z"]) >= Z_SIG)}

    findings, chains, candidates, side = [], [], [], []

    # R1 ── Vessel 캐스케이드 갭 (최우선 신호)
    for zone in ("up", "down"):
        pairs = []
        for s in sensors:
            zsn, num = paired_zone(s, sensors, "upper" if zone == "up" else "lower")
            if num is None:
                continue
            if zsn:
                pairs.append((s, zsn, num))
        for cart, zsn, num in pairs:
            wc = obs[cart]["worst"]; wz = obs[zsn]["worst"]
            if not wc or not wz:
                continue
            dc, dz = wc["dev"], wz["dev"]
            step = max(obs[cart]["per_step"],
                       key=lambda st: abs(obs[cart]["per_step"][st]["z"]))
            if dc >= CART_SIG_C and dz <= -1.0:
                findings.append({"id": f"R1-{zone}{num}", "type": "열전달_불량",
                                 "part": f"Vessel{zone.capitalize()} 존 {num}번 카트리지",
                                 "step": step,
                                 "evidence": f"열원(Cart{num}) {dc:+.1f}℃ 무리하는데 "
                                             f"존 온도(Heater{num}) {dz:+.1f}℃ 미달",
                                 "onset": obs[cart]["onset"]})
            elif dc <= -CART_SIG_C and dz <= -1.0:
                findings.append({"id": f"R1-{zone}{num}", "type": "열원_출력부족",
                                 "part": f"Vessel{zone.capitalize()} 존 {num}번 카트리지",
                                 "step": step,
                                 "evidence": f"열원 {dc:+.1f}℃ · 존 {dz:+.1f}℃ 동반 미달"
                                             " — 카트리지/전원 출력 부족",
                                 "onset": obs[cart]["onset"]})
            elif dc >= CART_SIG_C and dz >= 1.0:
                findings.append({"id": f"R1-{zone}{num}", "type": "존_과열",
                                 "part": f"Vessel{zone.capitalize()} 존 {num}번",
                                 "step": step,
                                 "evidence": f"열원 {dc:+.1f}℃ · 존 {dz:+.1f}℃ 동반 상승",
                                 "onset": obs[cart]["onset"]})

    # R2 ── T/P 전파 체인(배관 경로 기준).
    #   ★ 기점 선정 규칙 교정(2026-09-08): 예전 '설치순 최상류 유의'는 대부분의 센서가
    #     유의할 때 항상 같은 첫 센서(p_002)만 지목하는 고착이 있었다.
    #     기점 = ①발화 센서(모델 잔차도 이탈) 중 onset 가장 이른 것 →
    #            ②발화가 없으면 골든 onset 가장 이른 것. 설치순은 동률일 때만.
    tp = [s for s in sensors if obs[s]["kind"] in ("T", "P") and obs[s]["sig"]]
    if tp:
        fire = [s for s in tp if obs[s]["onset_model"] is not None]
        pool2 = fire if fire else tp
        def _okey(s):
            o = obs[s]
            t0 = o["onset_model"] if o["onset_model"] is not None else o["onset"]
            return (t0 if t0 is not None else 10**6, o["rank"])
        pool2 = sorted(pool2, key=_okey)
        head = pool2[0]
        rest = sorted([s for s in tp if s != head], key=lambda s: obs[s]["rank"])
        chains.append({
            "rule": "배관 전파(T/P)",
            "origin": head,
            "origin_kind": "발화(모델 잔차도 이탈)" if head in fire else "골든 최초 이탈",
            "origin_dev": obs[head]["worst"],
            "origin_onset": obs[head]["onset_model"] if head in fire else obs[head]["onset"],
            "downstream": [{"sensor": s, "dev": obs[s]["worst"],
                            "onset": obs[s]["onset"]} for s in rest[:6]],
            "note": "가스 온도/압력은 배관을 따라 전파됨 — 가장 먼저 '새로' 벗어난 지점이 기점 후보"})

    # R7 ── Vessel 전후 열수지: 유입(T2·T3)과 배출(T4) 온도로 챔버 내부 판정.
    #   유입 정상·배출 낮음 = 챔버가 열을 못 줌(존 가열 실패/열손실) ·
    #   유입 높음·배출 낮음 = 유입 과열이 원인 아님(챔버 문제 강화) ·
    #   유입 낮음·배출 낮음 = 상류(공급 예열) 문제가 챔버로 이어짐.
    def _dev_of(name_pat):
        for s2 in sensors:
            if re.search(name_pat, s2, re.I) and obs[s2]["worst"]:
                return s2, obs[s2]["worst"]["dev"], obs[s2]["sig"]
        return None, None, False
    t2n, t2d, t2s = _dev_of(r"t_002")
    t3n, t3d, t3s = _dev_of(r"t_001")
    t4n, t4d, t4s = _dev_of(r"t_003")
    if t4n is not None and (t2n is not None or t3n is not None):
        inflow = [(n2, d2) for n2, d2 in ((t2n, t2d), (t3n, t3d)) if n2 is not None]
        in_txt = " · ".join(f"{n2} {d2:+.1f}℃" for n2, d2 in inflow)
        if t4s and t4d < 0:
            if all(abs(d2) < 2.0 for _, d2 in inflow):
                verdictv = "유입은 정상인데 배출이 낮음 — 챔버 내부(존 가열/열손실)가 원인일 가능성"
            elif any(d2 > 2.0 for _, d2 in inflow):
                verdictv = "유입이 높은데도 배출이 낮음 — 유입 과열은 원인이 아니며 챔버 내부 열 문제 강화"
            else:
                verdictv = "유입부터 낮고 배출도 낮음 — 상류(공급 예열) 문제가 챔버까지 이어짐"
            chains.append({"rule": "Vessel 열수지(유입 T2/T3 vs 배출 T4)",
                           "origin": "Vessel(챔버)",
                           "downstream": [{"sensor": t4n, "dev": obs[t4n]["worst"]}],
                           "inflow": in_txt, "note": verdictv})

    # R3 ── H 라인 독립 소견(전파 아님)
    for s in sensors:
        if obs[s]["kind"] == "H" and obs[s]["sig"]:
            w0 = obs[s]["worst"]
            side.append({"sensor": s, "type": "라인히터_이상",
                         "dev": w0, "note": "자체 고정설정 히터 — 국소 점검 항목(전파 아님)"})

    # R4 ── 배기측 저온 정합성: 존 저온/열전달 불량과 배기 T 저온의 인과 연결
    out_t = [s for s in sensors if obs[s]["kind"] == "T"
             and re.search(r"(?i)outlet|pulsevent", s) and obs[s]["sig"]
             and obs[s]["worst"]["dev"] < 0]
    ht_loss = [f for f in findings if f["type"] in ("열전달_불량", "열원_출력부족")]
    if out_t and ht_loss:
        chains.append({"rule": "저온 공정 결과",
                       "origin": ht_loss[0]["part"],
                       "downstream": [{"sensor": s, "dev": obs[s]["worst"]} for s in out_t],
                       "note": "존이 목표 미달 → 웨이퍼가 저온에서 공정 → 배기 가스 온도 하락"})
    overheat = [f for f in findings if f["type"] == "존_과열"]
    if out_t and overheat and not ht_loss:
        chains.append({"rule": "팽창 냉각(scCO2)",
                       "origin": overheat[0]["part"],
                       "downstream": [{"sensor": s, "dev": obs[s]["worst"]} for s in out_t],
                       "note": "과열 → 압력 상승 → 벤트 개도 증가 → 팽창(줄-톰슨) 냉각으로 "
                               "배기측이 오히려 저온 — 벤트 IO(1=열림)·유량으로 확인 필요"})

    # R6 ── 초기 상태(carryover) 오염: 웨이퍼 '시작부터' 이탈(onset≈0)인데
    #   모델 잔차 onset 이 없음(= 모델이 놀라지 않음 = 과거부터 그 상태) → 이번 공정
    #   내부가 아니라 이전 런/대기 상태에서 이월된 초기 조건 문제. 특히 상류(저장조·
    #   공급) 구간이면 확신도 높음. (사용자 해석 규칙화, 2026-09-08)
    early = [s for s in sensors if obs[s]["sig"] and obs[s]["onset"] is not None
             and obs[s]["onset"] <= 2 and obs[s]["onset_model"] is None]
    if early:
        segs = sorted(set(obs[s]["segment"] for s in early))
        findings.append({
            "id": "R6", "type": "초기상태_이월오염",
            "part": " / ".join(segs[:3]) + " 구간",
            "step": 0,
            "evidence": ("웨이퍼 시작 시점부터 이미 기준을 벗어나 있었고 모델도 놀라지 "
                         "않음(과거부터 지속) → 이번 공정이 아니라 이전 런/대기 상태에서 "
                         "이월된 초기 조건 이상. 해당 센서: "
                         + ", ".join(early[:6])),
            "onset": 0})

    # 구간 요약: 구간별 최대 |z| · 유의 센서 수 · 발화/전파 성격 → '어느 구간이 문제' 결론
    seg_sum = {}
    for s2 in sensors:
        o = obs[s2]
        if not o["worst"]:
            continue
        g2 = seg_sum.setdefault(o["segment"], {"max_z": 0.0, "n_sig": 0,
                                               "n_fire": 0, "n_prop": 0, "sensors": []})
        az = abs(o["worst"]["z"])
        g2["max_z"] = max(g2["max_z"], az)
        if o["sig"]:
            g2["n_sig"] += 1
            g2["sensors"].append(s2)
            if o["onset_model"] is not None:
                g2["n_fire"] += 1                     # 모델도 놀란 = 발화 성격
            else:
                g2["n_prop"] += 1                     # 전파/지속 성격
    segments = sorted(
        [{"segment": k, **{kk: (round(vv, 1) if kk == "max_z" else vv)
                           for kk, vv in v.items() if kk != "sensors"},
          "sensors": v["sensors"][:6]} for k, v in seg_sum.items()],
        key=lambda x: -x["max_z"])

    # R5 ── 후보 랭킹: (발화 onset 이른 순, 유형 가중)
    pool = list(findings)
    if chains and chains[0]["rule"].startswith("배관"):
        h = chains[0]
        pool.append({"id": "R2-origin", "type": "유로_기점",
                     "part": f"{h['origin']} 인근 배관 구간(직전 히터/밸브 포함)",
                     "step": None, "evidence": f"{h['origin']} {h['origin_dev']['dev']:+.1f}℃ "
                     f"— 유의 이탈 중 최상류", "onset": obs[h["origin"]]["onset"]})
    pool.sort(key=lambda f: (f.get("onset") if f.get("onset") is not None else 10**6))
    for i, f in enumerate(pool[:3]):
        checks = ["같은 존 다른 카트리지와 비교(국소 vs 전역)",
                  "P4/P5 압력·벤트 IO(1=열림)·유량 흔적 확인",
                  "발화 시점(onset) 선후 관계 확인"]
        candidates.append({"rank": i + 1, "part": f["part"], "type": f["type"],
                           "why": f["evidence"], "onset_tp": f.get("onset"),
                           "checks": checks})

    observations = []
    for s2 in sensors:
        o = obs[s2]
        if not o["sig"] or not o["worst"]:
            continue
        d2 = o["worst"]["dev"]
        direction = "평균보다 높음" if d2 > 0 else "평균보다 낮음"
        st2 = max(o["per_step"], key=lambda k2: abs(o["per_step"][k2]["z"]))
        observations.append(
            f"{s2}({o['segment']}): {d2:+.1f}℃ {direction}, 최대 {abs(o['worst']['z']):.1f}σ"
            f" (스텝 {st2}"
            + (f", 시작 tp {o['onset']}" if o["onset"] is not None else "")
            + (", 모델도 놀람=발화 성격" if o["onset_model"] is not None
               else ", 모델 무반응=지속/전파 성격") + ")")
    flow_context = (
        "배관 경로(요약): CO2 공급원 → Reservoir(저장조, H1~4·Vent) → 밸브 031 → "
        "p_002(P1) → h_0052 예열 → 충전 미터링(Slow/Quick) → h_0161 → "
        "p_005(P2) → 필터 → t_005(T1) → h_005 예열 → t_002(T2) → "
        "밸브 032 분기 → [상부: Front필터/밸브 → p_008(P4) → VesselUp 존(Heater=존온도, "
        "Cart=열원) → 챔버 상부 분사] / [하부: h_0053 → t_001(T3) → Back필터/밸브 → "
        "p_007(P5) → VesselDown 존 → 챔버 하부 분사] → 웨이퍼 세정 → h_006 → "
        "p_003(P6)·t_003(T4) → PulseVent(T5)/SlowVent 배기. "
        "핵심 인과 축: 유입 온도(T2·T3)가 챔버 존(Heater)과 함께 세정 온도를 만들고, "
        "그 결과가 배출 온도(T4·T5)에 나타난다.")
    return {"wafer_id": wid, "recipe": recipe, "steps": usteps,
            "flow_context": flow_context,
            "observations": observations,
            "significant": {s: obs[s]["worst"] for s in sensors if obs[s]["sig"]},
            "segments": segments,
            "findings": findings, "chains": chains,
            "candidates": candidates, "side_findings": side}


# ── 서술 생성 ────────────────────────────────────────────────────────
_SYS = """당신은 반도체 초임계 CO2 세정설비를 오래 다룬 수석 공정 분석 엔지니어이고,
동료 엔지니어에게 보내는 기술 검토 보고를 쓴다.

문체 규칙(엄수):
· 자연스러운 한국어 서술문으로 쓴다. "**원인**" 같은 볼드 남발, 개조식 나열, 번역투를 피한다.
· 용어: '유로'가 아니라 '배관'이라고 쓴다. 센서명은 데이터 컬럼명 그대로 쓰되
  괄호로 위치를 풀어준다. 예: t_002(공급 배관, 예열 직후).
· 결론을 서두르지 말 것. 서로 다른 여러 센서와 설비 구간을 놓고 비교·저울질하는
  사고 과정을 보고서에 드러낸 뒤 결론에 도달하라. 한 센서에 매몰된 단정은 금지 —
  특히 배관 최상류 센서가 늘 원인처럼 보이는 착시를 경계하라(전파·지속 상태일 수 있다).
· Vessel 전후 온도 축(유입 T2·T3 → 챔버 존 Heater → 배출 T4·T5)을 반드시 검토에 포함하라.

사실 규칙(위반 시 보고 무효):
· 센서의 높음/낮음 방향은 observations 배열의 문장을 '그대로' 따른다. dev 부호(+는 높음,
  −는 낮음)를 절대 뒤집지 마라. 서술 전 해당 관측문을 인용하라.
· observations 에 없는 센서(유의 이탈이 아닌 센서)를 원인이나 증상으로 단정하지 마라.
  평균과 같은 센서는 "정상 범위"라고만 말할 수 있다.

당신은 반도체 초임계 CO2 세정설비의 공정 엔지니어에게 보고하는 수석 분석가다.
아래 JSON 증거 패킷'만' 근거로 상세한 한국어 보고를 작성하라. 패킷에 없는 사실을 추측하지 마라.
피상적 요약은 금지 — 각 판단마다 패킷의 수치(σ, ℃, 스텝, onset tp)를 최소 2개씩 인용하라.

형식(마크다운, 각 절을 충실히):
## 결론
(2~4문장: 어느 '구간'의 무엇이, 어떤 이유로, 어떤 결과를 만들었는가. segments 배열의
 순위를 반드시 반영해 구간을 명시하라)
## 문제 구간 진단
(segments 를 표처럼 훑어라: 구간별 max_z·유의 센서 수·발화(n_fire)/전파(n_prop) 성격.
 발화 성격 구간 = 문제가 '시작된' 곳, 전파/지속 성격 구간 = 영향이 '번진' 곳으로 구분)
## 인과 체인
(관측 → 원인 → 결과를 단계별로, 유로 순서를 따라 서술. findings 와 chains 의 수치 인용.
 초기상태_이월오염(R6) finding 이 있으면: 이번 공정이 아니라 이전 런/대기에서 이월된
 초기 조건 문제임을 명확히 하고, 웨이퍼 시작 전 상태 점검을 권고하라)
## 각 후보 검증
(candidates 각각에 대해: 지지 증거 2개 이상 / 반박·약점 1개 이상 / 판정)
## 배제된 가설
(왜 아닌지 수치와 함께)
## 현장 확인 항목
(구체적으로: 어느 구간의 어느 장치를, 무엇으로, 어떤 값 기준으로)

설비 상식:
· H라인 히터는 자체 고정설정(피드백 없음) — 하류 변화에 보상하지 않는다. H 이탈 = 그 히터 자체 문제.
· Vessel 은 캐스케이드: Cart=열원 온도(액추에이터), Heater=존 외부 온도(목표이자 측정).
  **웨이퍼가 실제로 겪는 온도에 가까운 것은 Heater(존) 쪽이다** — 존 판단은 Heater 를 1급 증거로,
  Cart 는 '열원이 얼마나 무리했나'의 보조 증거로 써라.
· 가스 T/P 만 배관을 따라 전파된다. scCO2 는 과열→압력↑→벤트↑→팽창냉각으로 배기측 저온 가능.
· 색 의미: 발화(골든·잔차 모두 이탈)=문제 시작점 후보, 전파(골든만)=이미 벗어난 상태의 지속/하류,
  시작부터 전파 상태(onset≈0)=이전 런에서 이월된 초기 오염."""


def find_equipment_graph() -> str:
    """설비 구조 도면 PNG 경로(있으면). 우선순위: env RCA_GRAPH_PNG → 패키지 동봉본."""
    cand = os.environ.get("RCA_GRAPH_PNG")
    if cand and os.path.exists(cand):
        return cand
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "equipment_graph.png"),
              os.path.join(here, "..", "equipment_graph.png")):
        if os.path.exists(c):
            return c
    return ""


def period_engine(daily: dict, sensors: list) -> dict:
    """일별 σ 시계열에서 4대 전조 패턴을 자동 탐지(정비일지 불필요 — 검사 데이터만).

    daily: {date: {"n":N, "ng":k, "gz":{sensor:[sum,cnt]}}} · sensors: 순서 목록
    R8 드리프트: 단조 누적(스피어만 상관>0.8 & 진폭>=1.5σ)
    R9 보상쌍: 같은 구간 열원 계열이 반대 방향(둘 다 |평균|>=1σ)
    R11 헌팅: 후반 변동폭이 전반의 2배 이상(>=0.5σ)
    급변: 인접일 점프 >=3σ
    """
    dates = sorted(daily)
    def series(sn):
        out = []
        for d in dates:
            a = daily[d]["gz"].get(sn)
            out.append(a[0] / a[1] if a and a[1] else None)
        return out
    def _rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k2, i in enumerate(idx):
            r[i] = k2
        return r
    drifts, jumps, hunts = [], [], []
    ser = {}
    for sn in sensors:
        v = [x for x in series(sn)]
        xs = [(i, x) for i, x in enumerate(v) if x is not None]
        if len(xs) < 5:
            continue
        ii = [i for i, _ in xs]; vv = [x for _, x in xs]
        ser[sn] = (ii, vv)
        # R8 드리프트
        ri, rv = _rank(ii), _rank(vv)
        n2 = len(vv)
        mi = sum(ri) / n2; mv = sum(rv) / n2
        cov = sum((a - mi) * (b - mv) for a, b in zip(ri, rv))
        den = (sum((a - mi) ** 2 for a in ri) * sum((b - mv) ** 2 for b in rv)) ** 0.5
        rho = cov / den if den > 1e-9 else 0.0
        amp = vv[-1] - vv[0]
        if abs(rho) >= 0.8 and abs(amp) >= 1.5:
            drifts.append({"sensor": sn, "start": dates[ii[0]],
                           "direction": "상승" if amp > 0 else "하강",
                           "amplitude_sigma": round(amp, 1), "monotonicity": round(rho, 2)})
        # 급변
        for k2 in range(1, len(vv)):
            if abs(vv[k2] - vv[k2 - 1]) >= 3.0:
                jumps.append({"sensor": sn, "date": dates[ii[k2]],
                              "jump_sigma": round(vv[k2] - vv[k2 - 1], 1)})
                break
        # R11 헌팅(변동폭 확대)
        if n2 >= 6:
            h1 = vv[:n2 // 2]; h2 = vv[n2 // 2:]
            def _std(a):
                m3 = sum(a) / len(a)
                return (sum((x - m3) ** 2 for x in a) / len(a)) ** 0.5
            s1, s2 = _std(h1), _std(h2)
            if s2 >= max(0.5, 2 * s1):
                hunts.append({"sensor": sn, "early_std": round(s1, 2),
                              "late_std": round(s2, 2)})
    # R9 보상쌍: 드리프트/유의 센서 중 같은 접두 구간에서 반대 방향
    pairs = []
    sig = {d2["sensor"]: d2 for d2 in drifts}
    names = list(sig)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            sa, sb = names[a], names[b]
            pa = re.sub(r"\d+_PV$|_PV$", "", sa).lower()[:9]
            pb = re.sub(r"\d+_PV$|_PV$", "", sb).lower()[:9]
            if pa[:6] == pb[:6] and sig[sa]["direction"] != sig[sb]["direction"]:
                pairs.append({"pair": [sa, sb],
                              "note": "같은 구간 반대 방향 — 한쪽 폭주+다른쪽 보상 서명"})
    # 불량률 변곡(자가 탐지) — 증가·감소 '모든' 유의 점프를 나열한다.
    #   단일 최대점만 보면 서서히 좋아졌다 나빠지는 중간 굴곡(예: 7/6 감소→7/15 재증가)을 놓친다.
    rates = [(d, daily[d]["ng"] / max(1, daily[d]["n"]), daily[d]["n"]) for d in dates]
    deltas = []
    for k2 in range(1, len(rates)):
        if rates[k2][2] < 5 or rates[k2 - 1][2] < 5:
            continue
        deltas.append((rates[k2][0], rates[k2][1] - rates[k2 - 1][1]))
    absd = sorted(abs(x[1]) for x in deltas)
    madd = absd[len(absd) // 2] if absd else 0.0
    thr = max(0.12, 3.0 * madd)
    infls = [{"date": d2, "delta": round(v2, 3),
              "direction": "증가" if v2 > 0 else "감소"}
             for d2, v2 in deltas if abs(v2) >= thr]
    infls = sorted(infls, key=lambda x: -abs(x["delta"]))[:8]
    infls.sort(key=lambda x: x["date"])
    split = None; best = 0.0
    for d2, v2 in deltas:
        if v2 > best:
            best = v2; split = d2
    return {"drifts": drifts, "compensation_pairs": pairs, "hunting": hunts,
            "jumps": jumps, "ng_inflection": split, "ng_inflections": infls,
            "daily_ng_rate": [{"date": d, "rate": round(r3, 3), "n": n3}
                              for d, r3, n3 in rates]}


_MONTHLY_SYS = """당신은 반도체 초임계 CO2 세정설비의 수석 공정 분석 엔지니어다.
아래에 '기간 일별×센서 이탈 표'(값=그날 평균 이탈 σ, +는 골든보다 높음, -는 낮음,
|값|>=1 이 유의), 규칙 엔진의 사전 탐지 결과(드리프트/보상쌍/헌팅/급변/불량률 변곡),
설비 사실, 과거 확인 사례집이 주어진다. 표 밖의 수치를 지어내지 마라.
정비일지 같은 외부 정답은 없다 — 검사 데이터만으로 원인을 찾고 미래를 예측해야 한다.

다음 절차를 순서대로 밟고, 각 단계의 결과를 보고서에 명시하라:
1단계(동시 시작 그룹핑): 같은 날짜에 이탈이 시작된 센서들을 시작일별로 묶어라.
2단계(배관 정렬): 각 그룹을 배관 순서(설비 사실의 경로)로 정렬하고, 가장 이른 그룹의
  최상류 지점을 기점 후보로 지목하라.
3단계(보상 쌍 탐지): 같은 구간에서 서로 반대 방향으로 벌어진 열원 쌍(예: 재킷↑+카트리지↓)을
  찾아라 — 한쪽 폭주 + 다른 쪽 보상의 서명이다. 반대 방향 신호를 반증으로 착각하지 마라.
4단계(단계 전환): 점진 이탈 시작일 / 변동폭 확대(헌팅) 시작일 / 급변(붕괴)일을 각각 찾아라.
5단계(물리 결합): 밀폐 구간 T상승→P상승 결합, 압력상승→벤트→배기 냉각을 검토하라.
6단계(자가 대조): 엔진이 탐지한 불량률 변곡일 '목록'(ng_inflections — 증가뿐 아니라 감소도
  있다)과 네가 찾은 전환일들을 대조하라. 좋아졌다 다시 나빠지는 중간 굴곡(일시 회복)은
  부분 조치·간헐 증상의 단서이므로 최종 붕괴일 하나로 뭉개지 말고 각각 해석하라.
  엔진의 드리프트·보상쌍 목록과 네 그룹핑이 서로 지지하는지도 대조하고, 불일치 시 이유를 적어라.
7단계(오컴): 관측 전부를 설명하는 최소 원인을 우선하라. 서로 다른 고장 여러 개가 같은 날
  시작됐다는 가설은 후순위다. 과거 사례집과 서명이 일치하면 그 사례를 인용하라.
  엔진의 step_trend_log(일자별 스텝×센서 트렌드 이상 일지)가 있으면 결론을 '어느 스텝'
  수준까지 좁혀라 — 예: 전환일 직전 며칠 특정 스텝의 출렁임·기울기 이상은 강한 단서다.
  일지의 '모델·' 접두 항목은 모델 복원값 기준: 골든·모델 둘 다 이상인 센서·스텝이 발화 후보다.
8단계(결론): 근본 원인 후보 랭킹(왜 그 순서인지), 확신도(%), 반증 조건, 현장 확인 항목을 내라.
9단계(예측·선제 조치): 지금 진행 중인 드리프트·헌팅이 있으면 현재 단계(발화/헌팅/붕괴 전)를
  판정하고, 추세가 유지될 때 예상되는 결과와 대략의 시점, 붕괴 전에 취할 선제 조치를 제시하라.
  진행 중 전조가 없으면 '현재 전조 없음'을 명시하라.

문체: 자연스러운 한국어 기술 보고. 마크다운 기호(**, ##, 표) 금지 — 평문 문장으로. '유로' 대신 '배관'."""


def load_casebook() -> str:
    """확인된 과거 사례집(casebook.md) — 경험 축적 파일. 없으면 빈 문자열."""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "casebook.md"),
              os.path.join(here, "app", "casebook.md"),
              os.path.join(os.getcwd(), "casebook.md")):
        if os.path.isfile(c):
            try:
                return open(c, encoding="utf-8").read()[:6000]
            except Exception:                          # noqa: BLE001
                pass
    return ""


_PHYS_DEFAULT = {
    "io_map": {
        "9000002": {"role": "supply_open", "sensor": "p_005_PV", "dir": "+"},
        "9000006": {"role": "fill_slow", "sensor": "p_005_PV", "dir": "+"},
        "9000007": {"role": "fill_quick", "sensor": "p_005_PV", "dir": "+"},
        "9000008": {"role": "fill_quick2", "sensor": "p_005_PV", "dir": "+"},
        "9000003": {"role": "front_open", "sensor": "p_008_PV", "dir": "+"},
        "9000004": {"role": "back_open", "sensor": "p_007_PV", "dir": "+"},
    },
    "arch_rules": [
        {"sensor": "dP_VslF-VslB", "open_first": "9000004", "open_second": "9000003",
         "recover_ratio_min": 0.5, "recover_abs_min": 0.3, "depth_sigma_min": 2.0},
    ],
    "sign_rules": {"sensors": ["dP_VslF-Out", "dP_VslB-Out"],
                   "golden_pos_margin_sigma": 2.0},
    "io_response": {"window_frac": 0.10, "min_rise_ratio": 0.3},
  "pattern_rules": {
    "exp_drop": {"sensor": "t_002_PV", "step": 5, "alpha": 0.4,
                 "min_drop_ratio": 0.2},
    "arches": {"sensor": "dP_VslF-VslB", "steps": [2, 3, 4], "alpha": 0.5},
    "w_bulge": {"sensor": "t_003_PV", "step": 4, "delta": 0.5,
                "min_rise_ratio": 0.2, "peak_pct": 80.0},
    "t_balance": {"result": "t_003_PV",
                  "causes": ["t_002_PV", "t_001"],
                  "k": 3.0, "min_run": 5},
    "p_balance": {"result": "p_008_PV",
                  "causes": ["p_005_PV", "p_007_PV"],
                  "k": 3.0, "min_run": 5}},
    "io_timing": {"tol_frac": 0.02},
    "exclude_sensors_regex": "(?i)^h_",
}


def load_physics():
    """물리법칙 문서(physics_rules.md) 로드 — (자유 서술 텍스트, 기계 규칙 dict).

    사용자 소유 문서. json 블록이 없거나 깨져도 죽지 않고 내장 기본값 폴백.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    txt = ""
    for c in (os.path.join(here, "physics_rules.md"),
              os.path.join(here, "app", "physics_rules.md"),
              os.path.join(os.getcwd(), "physics_rules.md")):
        if os.path.isfile(c):
            try:
                txt = open(c, encoding="utf-8").read()
                break
            except Exception:                          # noqa: BLE001
                pass
    cfg = dict(_PHYS_DEFAULT)
    if txt:
        m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.S)
        if m:
            try:
                user_cfg = json.loads(m.group(1))
                cfg.update({k: v for k, v in user_cfg.items() if v is not None})
            except Exception as e:                     # noqa: BLE001
                print(f"[physics] 기계 규칙(json) 파싱 실패 — 내장 기본값 사용: {e}",
                      flush=True)
    # 자유 서술만 프롬프트에(json 블록 제외·길이 캡)
    body = re.sub(r"```json.*?```", "", txt, flags=re.S)[:5000] if txt else ""
    return body, cfg


def narrate_monthly(daily_table: str, engine_findings: str = "", model: str = None,
                    host: str = None, timeout: int = None, horizon: str = "월간") -> dict:
    """기간 압축 표 + 규칙 엔진 탐지 결과 → 원인 분석·예측(내 추론 절차 이식). 비판자 토론 포함."""
    timeout = timeout or int(os.environ.get("RCA_LLM_TIMEOUT", "900"))
    host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    model = model or os.environ.get("RCA_LLM_MODEL")
    if not model:
        try:
            with urllib.request.urlopen(host + "/api/tags", timeout=10) as r:
                names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
            model = next((n for n in names if "qwen" in n.lower()), names[0] if names else None)
        except Exception:                             # noqa: BLE001
            model = "qwen3.6:35b-a3b"
    flow = _flow_facts()
    cb = load_casebook()
    user = ("[분석 범위] " + horizon + " 분석\n\n[설비 사실]\n" + flow
            + "\n\n[일별×센서 이탈 표(σ)]\n" + daily_table
            + "\n\n[규칙 엔진 사전 탐지 결과]\n" + (engine_findings or "(없음)")
            + ("\n\n[과거 확인 사례집]\n" + cb if cb else ""))
    body = json.dumps({"model": model, "stream": False,
                       "options": {"temperature": 0.2, "num_ctx": 32768},
                       "messages": [{"role": "system", "content": _MONTHLY_SYS},
                                    {"role": "user", "content": user}]}).encode()
    try:
        req = urllib.request.Request(host + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = _strip_md((json.loads(r.read().decode()).get("message") or {}).get("content", "").strip())
        if not txt:
            raise RuntimeError("빈 응답")
        db = debate(txt, {"daily_table": daily_table[:8000],
                          "engine_findings": engine_findings[:4000]},
                    host, model, timeout)
        eng = f"ollama:{model}" + (f" + 비판자:{os.environ.get('RCA_CRITIC_MODEL')}"
                                   if db.get("debate") else "")
        return {"narrative": db["narrative"], "engine": eng, "debate": db.get("debate"),
                "prompt_system": _MONTHLY_SYS, "prompt_user": user[:4000] + "…"}
    except Exception as e:                             # noqa: BLE001
        return {"narrative": f"{horizon} 분석 실패: {e}", "engine": "error",
                "prompt_system": _MONTHLY_SYS, "prompt_user": user[:2000]}


def debate_report(report: dict, host: str = None, timeout: int = None) -> dict:
    """딥리서치 최종 보고서에 비판자 토론을 붙인다(RCA_CRITIC_MODEL 설정 시).

    DTE 결론은 도구 검증을 거친 것이므로 결론 자체를 다시 쓰지 않고,
    공격 목록과 본 모델의 대응(방어/양보)을 별도 필드로 돌려준다.
    """
    if not os.environ.get("RCA_CRITIC_MODEL"):
        return {}
    timeout = timeout or int(os.environ.get("RCA_LLM_TIMEOUT", "900"))
    host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    model = os.environ.get("RCA_LLM_MODEL")
    if not model:
        try:
            with urllib.request.urlopen(host + "/api/tags", timeout=10) as r:
                names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
            model = next((n for n in names if "qwen" in n.lower()), names[0] if names else None)
        except Exception:                              # noqa: BLE001
            return {}
    text = ("결론: " + str(report.get("conclusion", "")) +
            "\n확신도: " + str(report.get("confidence", "")) +
            "\n핵심 증거: " + json.dumps(report.get("key_evidence", []), ensure_ascii=False) +
            "\n반증/약점: " + json.dumps(report.get("counter_evidence", []), ensure_ascii=False))
    try:
        db = debate(text, {"goal": report.get("goal", "")}, host, model, timeout)
        if db.get("debate"):
            return {"debate": db["debate"], "debated_narrative": db["narrative"]}
    except Exception:                                  # noqa: BLE001
        pass
    return {}


_FOCUS_SYS = """당신은 반도체 초임계 CO2 세정설비의 수석 공정 분석 엔지니어다.
불량 보고 '이전 며칠'의 웨이퍼 데이터에서 전조를 찾는 집중 분석이다. 가정: 어떤 센서가
점점 드리프트하거나 이상 징후를 키우다가 보고일에 터졌다.

입력 자료:
- 불량 보고서: '터진 날'을 지정하는 용도다. 보고일 이후는 엔지니어가 설비를 만지며
  피드백 제어를 바꿨을 것이므로 골든과 달라도 원인 증거가 아니다. 원인은 보고 '전' 데이터로만 찾아라.
- 일자별 스텝×센서 트렌드 이상 집계: 평균이 아니라 '웨이퍼 단위' 판정의 집계다.
  각 웨이퍼의 각 스텝에서 골든과 트렌드를 비교해 4유형으로 분류했다:
  레벨(트렌드 같고 값이 높/낮음) · 기울기상이(오르내림 방향이 다름) ·
  출렁임(평탄해야 할 곳이 진동) · 모양상이(파형 자체가 다름).
  'k/n장'은 그날 n장 중 k장에서 그 이상이 나왔다는 뜻이다.
- '모델·' 접두 항목은 골든이 아니라 '모델 복원값(입력 조건을 반영한 기대 파형)' 기준 판정이다.
  해석 규칙: 골든 이상 + 모델 정상 = 조건 변화를 모델이 설명(상류에서 전파된 결과일 가능성) /
  골든 이상 + 모델도 이상 = 모델조차 설명 못 하는 이변(발화 기점 후보로 우선하라).
  H 센서류(라인 히터)는 모델 출력이 없어 골든 기준만 존재한다 — 모델 항목 부재를 정상으로 오해하지 마라.
- 시간축 주의: 실제 웨이퍼 길이는 제각각이라 골든 길이(중앙값)로 리샘플한 시간축에서
  비교했다. 스텝 경계 근처의 잔이상은 리샘플 오차일 수 있으니 지속·비율이 높은 것을 우선하라.

절차(각 단계 결과를 보고서에 명시):
1단계(악화 진행 추림): 날짜가 보고일에 가까워질수록 발생 비율이 커지는 센서×스텝을 찾아라.
  엔진의 '악화 진행' 목록과 대조하라.
2단계(유형 해석): 레벨=설정/센서 오프셋, 기울기상이=제어 응답 변화, 출렁임=헌팅(제어 불안정),
  모양상이=공정 이벤트 자체가 다름 — 유형별로 물리 해석을 달아라.
3단계(배관 정렬): 이상 센서들을 배관 순서로 정렬해 최상류 기점 후보를 지목하라.
4단계(서사): 드리프트 시작→징후 확대→폭발의 시간 서사를 날짜·수치 인용으로 구성하라.
5단계(결론): 기점 센서·스텝 랭킹, 확신도(%), 반증 조건, 현장 확인 항목.
6단계(선제 관찰): 다음에 같은 전조가 오면 며칠 전에 잡을 수 있는 관찰 항목을 제안하라.

표 밖의 수치를 지어내지 마라. 문체: 자연스러운 한국어 기술 보고. 마크다운 기호 금지."""


_GCMP_SYS = """당신은 반도체 초임계 CO2 세정설비의 수석 공정 분석 엔지니어다.
같은 레시피라도 챔버(DSV)마다 골든 파형이 다를 수 있다. 지금 분석 중인 DSV 의 골든이
'그 자체로 문제인지'를 다른 DSV 들의 골든과 비교해 판정하는 과제다.

입력 표: 센서×스텝별로 대상 DSV 골든 vs 비교 DSV 골든의
  offset(σ, +면 대상이 높음) · corr(모양 상관) · rough(출렁임 배율) 이 주어진다.

판정 기준:
- 트렌드 같고 offset 만 있음 = 챔버 개성(셋포인트·센서 부착 차이)일 가능성 — 크기가 크면 지적.
- corr 낮음(모양 자체가 다름) = 대상 DSV 의 공정 거동이 다른 것 — 골든 오염(불량 섞임) 또는
  설비 상태 차이. 어느 쪽인지 근거와 함께 판단하라.
- 여러 비교 DSV 가 서로는 비슷한데 대상만 다르면 대상 이상 가능성이 높다. 반대로 비교군도
  제각각이면 챔버 개성으로 본다.
결론: 대상 골든 신뢰 가능 여부, 문제 센서·스텝 목록, 골든 재수집 권고 여부.
표 밖의 수치를 지어내지 마라. 문체: 자연스러운 한국어 기술 보고. 마크다운 기호 금지."""


def _flow_facts() -> str:
    return ("배관 순서: CO2공급원>Reservoir(Jacket/Inner/Cart/CartDown/Vent)>밸브031>"
            "p_002(P1)>h_0052(H6)>충전미터링>h_0161>p_005(P2)>"
            "필터>p_006(P3)>t_005(T1)>h_005(H12)>t_002(T2)>"
            "분기>[상부:Front계열>p_008(P4)>h_022(존SV겸측정)/h_021(열원)]"
            "[하부:h_0053(H16)>t_001(T3)>Back계열>p_007(P5)>VesselDown존]"
            ">세정>Outlet>p_003(P6)>t_003(T4)>PulseVent(P7,T5)/SlowVent 배기.\n"
            "사실: 저장조·유지구간은 밀폐라 T상승=P상승. H라인 히터는 자체 고정설정(무피드백). "
            "Vessel/Reservoir 는 열원(Cart)이 측정점을 목표로 제어되는 구조.")


def _resolve_llm(host=None):
    host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    model = os.environ.get("RCA_LLM_MODEL")
    if not model:
        try:
            with urllib.request.urlopen(host + "/api/tags", timeout=10) as r:
                names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
            model = next((n for n in names if "qwen" in n.lower()), names[0] if names else None)
        except Exception:                              # noqa: BLE001
            model = "qwen3.6:35b-a3b"
    return host, model


def _narrate_table(sys_msg: str, user: str, label: str, packet_hint: dict) -> dict:
    """표 기반 분석 공통 실행기: qwen 호출 + 비판자 토론 + 실패 폴백."""
    timeout = int(os.environ.get("RCA_LLM_TIMEOUT", "900"))
    host, model = _resolve_llm()
    try:
        txt = _ollama_chat(host, model, sys_msg, user, timeout, num_ctx=32768)
        if not txt:
            raise RuntimeError("빈 응답")
        db = debate(txt, packet_hint, host, model, timeout)
        eng = f"ollama:{model}" + (f" + 비판자:{os.environ.get('RCA_CRITIC_MODEL')}"
                                   if db.get("debate") else "")
        return {"narrative": db["narrative"], "engine": eng, "debate": db.get("debate"),
                "prompt_system": sys_msg, "prompt_user": user[:4000] + "…"}
    except Exception as e:                             # noqa: BLE001
        return {"narrative": f"{label} 실패: {e}", "engine": "error",
                "prompt_system": sys_msg, "prompt_user": user[:2000]}


def narrate_focus(daily_table: str, progression: str, report_text: str) -> dict:
    """불량 보고 전 구간 집중 분석(웨이퍼 단위 스텝-트렌드 집계 기반)."""
    flow = _flow_facts()
    cb = load_casebook()
    user = ("[불량 보고서]\n" + (report_text or "(내용 없음 — 날짜만 지정됨)")
            + "\n\n[설비 사실]\n" + flow
            + "\n\n[엔진: 악화 진행 목록(보고일로 갈수록 발생 비율 증가)]\n"
            + (progression or "(없음)")
            + "\n\n[일자별 스텝×센서 트렌드 이상 집계]\n" + daily_table
            + ("\n\n[과거 확인 사례집]\n" + cb if cb else ""))
    return _narrate_table(_FOCUS_SYS, user, "집중 분석",
                          {"daily_table": daily_table[:8000], "progression": progression[:2000]})


def narrate_golden_compare(table: str, target: str, others: str) -> dict:
    """DSV 골든 상호 비교 — 대상 골든 자체의 건전성 판정."""
    flow = _flow_facts()
    user = (f"[대상 DSV] {target}\n[비교 DSV] {others}\n\n[설비 사실]\n" + flow
            + "\n\n[센서×스텝 골든 비교 표]\n" + table)
    return _narrate_table(_GCMP_SYS, user, "골든 비교", {"table": table[:8000]})


_VLMPAT_SYS = """이미지는 기준 파형(실선 REF)과 며칠간 일평균(점선)의 비교 그래프다."""


_TRENDSCAN_SYS = """당신은 반도체 초임계 CO2 세정설비의 수석 공정 분석 엔지니어다.
'이상 기준 파형' 대비 기간 전체를 하루씩 누적 비교한 추세 스캔 결과가 주어진다:
패턴 카드(센서·스텝·유형(레벨 드리프트/볼록 패턴/급변)·시작일·상태(형성/유지/심화/소멸)·
심화 곡선·개입 이벤트), 배관 연관 사슬, VLM(시각 검토) 소견. CNN 모델 기대값은 없다.

전제: Vessel 추종형(Cart/Heater)과 H 라인 히터는 SV(설정값)를 알 수 없고 목표 추종으로
상시 변하므로 패턴 판정에서 제외되어 있다 — 이들 센서를 원인으로 단정하지 마라.
[Vessel 내부 추정 소견]이 있으면 "유입(T2·T3) 정상 + 배출(T4) 이상 = Vessel 내부 문제
추정(직접 측정 불가)"으로 서술하고, 확정이 아니라 추정임을 명시하라.

절차(각 단계 결과를 보고서에 명시):
1단계(타임라인 재구성): 카드들을 시작일 순으로 배열해 형성→심화→개입/소멸의 시간 서사를 써라.
2단계(개입 해석): '소멸(개입 추정)' 이벤트는 엔지니어 조치 가능성과 문제 자체 소강 두 가지로
  해석하고 어느 쪽이 그럴듯한지 근거를 대라. 급변 악화는 문제 심화의 증거다.
3단계(배관 연관): 같은 시기 카드들을 배관 순서로 검토해 기점 후보를 지목하라 — 단, 기점은
  '가장 상류'가 아니라 '가장 먼저 시작한' 것을 우선한다.
4단계(VLM 대조): 시각 검토 소견이 '안 보임/희미'인 카드는 신뢰도를 낮춰라.
5단계(재발 대조): '★재발' 표시 카드는 과거 기간에 같은 서명이 있었다는 뜻이다 —
  같은 문제의 재발로 묶어 서술하고, 과거 사례집에 해당 건이 있으면 인용하라.
  [패턴 사례 통계]의 다회 발생 그룹은 '만성/간헐 반복 문제'로 별도 절에서 다뤄라 —
  발생 월 목록과 심화 추이를 인용해 "N회 반복된 같은 모양의 문제"임을 명시하라.
6단계(결론): 근본 원인 후보 랭킹, 확신도(%), 반증 조건, 현장 확인 항목, 그리고
  '다음에 같은 전조가 오면 며칠 전에 잡을 수 있는가'를 제시하라.
표 밖의 수치를 지어내지 마라. 문체: 자연스러운 한국어 기술 보고. 마크다운 기호 금지."""


def narrate_pattern(card: dict, png_path: str) -> str:
    """패턴 카드 1건 VLM 시각 확인 — 이미지가 실제로 그 패턴을 보여주는지."""
    import base64
    # VLM 은 카드마다 호출되므로 전용 타임아웃(기본 180초) — 애매한 이미지에서
    #   비전 모델이 사고 폭주하면 카드 8장 × 900초 = 2시간을 막아버린다(실측).
    timeout = int(os.environ.get("RCA_VLM_TIMEOUT",
                                 min(180, int(os.environ.get("RCA_LLM_TIMEOUT", "900")))))
    host, model = _resolve_llm()
    model = os.environ.get("RCA_VLM_MODEL") or model
    try:
        b64 = base64.b64encode(open(png_path, "rb").read()).decode()
        # 질문은 단순 관찰형으로 — 주장 메타(σ 수치 등)를 섞으면 작은 VL 모델이
        #   수치를 근거로 착각해 판정을 뒤집는다(실측). 이진 관찰 + 위치 + 추세만 묻는다.
        ask_map = {"볼록 패턴": "점선들이 실선(REF)보다 국소적으로 솟거나 꺼진 구간",
                   "레벨 드리프트": "점선들이 실선(REF)에서 위/아래로 벌어진 정도",
                   "급변 이벤트": "점선들 사이의 큰 차이"}
        tgt = ask_map.get(card["kind"], "점선과 실선의 차이")
        user = (f"{tgt}이 보이는가? '명확/희미/안 보임' 중 하나로 시작해 두 문장 이내:"
                f" 어느 구간(타임포인트 대략)인지, 날짜(범례 순)가 갈수록 커지는지.")
        # ★ 비전 호출은 system 역할과 함께면 일부 모델이 폭주(무한 사고) —
        #   지시문을 user 에 합친다. ctx 는 기본 유지(작으면 사고 분량에 밀려 빈 응답).
        out = _ollama_chat(host, model, "", _VLMPAT_SYS + "\n\n" + user, timeout,
                           images=[b64])
        return out or "(VLM 응답 없음)"
    except Exception as e:                             # noqa: BLE001
        return f"(VLM 확인 실패: {e})"


def narrate_trendscan(cards_text: str, chains_text: str, vlm_text: str) -> dict:
    """추세 스캔 종합 서사(형성→심화→개입 타임라인 + 기점 랭킹). 비판자 토론 포함."""
    flow = _flow_facts()
    cb = load_casebook()
    user = ("[설비 사실]\n" + flow
            + "\n\n[패턴 카드]\n" + cards_text
            + "\n\n[배관 연관 사슬]\n" + (chains_text or "(없음)")
            + "\n\n[VLM 시각 검토 소견]\n" + (vlm_text or "(수행 안 함)")
            + ("\n\n[과거 확인 사례집]\n" + cb if cb else ""))
    return _narrate_table(_TRENDSCAN_SYS, user, "추세 스캔 종합",
                          {"cards": cards_text[:8000], "chains": chains_text[:2000]})


_MAINPAT_SYS = """당신은 반도체 초임계 CO2 세정설비의 수석 공정 분석 엔지니어다.
'주요 패턴' 목록이 주어진다 — 장기 추세 스캔에서 같은 모양으로 반복 발생한 문제들을
그룹으로 묶은 것이다(P/T 센서 한정 · Vessel 추종형과 H 라인 히터는 SV 미상이라 판정
제외됨 — 이들을 원인으로 단정하지 마라). 각 그룹에는 발생 회차별 기간·상태(형성/유지/
심화/소멸)·진폭(σ)·발생 간격(일)이 있다.

절차(각 단계를 보고서에 명시):
1단계(그룹별 진단): 그룹마다 — 반복 주기가 규칙적인가(간격 일수 인용), 회차가 갈수록
  진폭이 커지는가(악화 추세), '소멸(개입 추정)' 후 재발이면 임시조치 반복의 서명이다.
2단계(그룹 간 연관): 같은 시기에 함께 발생하는 그룹들이 있으면 배관 순서(설비 사실)로
  연결해 공통 원인 가능성을 검토하라.
3단계(원인 가설): 그룹 단위로 원인 후보를 랭킹하라 — 주기성이 있으면 주기적 운전 조건
  (PM 주기·원료 로트 교체·계절/외기 등)과의 연동을 우선 검토하라. 과거 사례집은
  '같은 센서·같은 구간'일 때만 인용하고, 다른 센서의 사례를 유추 근거로 쓰지 마라.
4단계(현재 상황 요약): '★현재 활성' 표시가 붙은 그룹만 활성이라 서술하고, 표시가
  없는 그룹은 반드시 '현재 비활성(종료됨)'으로 서술하라 — 활성 여부를 추측하지 마라.
  다음 재발 예상 시기를 간격 통계로 추정하라.
5단계(권고): 그룹별 현장 확인 항목과, 다음 재발을 며칠 전에 잡기 위한 감시 항목.
표 밖의 수치를 지어내지 마라. 문체: 자연스러운 한국어 기술 보고. 마크다운 기호 금지."""


def narrate_mainpatterns(groups_text: str, extra_text: str = "") -> dict:
    """주요 패턴(반복 문제 그룹) 원인 분석·상황 요약. 비판자 토론 포함."""
    flow = _flow_facts()
    cb = load_casebook()
    user = ("[설비 사실]\n" + flow
            + "\n\n[주요 패턴 그룹]\n" + groups_text
            + (("\n\n[참고 소견]\n" + extra_text) if extra_text else "")
            + (("\n\n[과거 확인 사례집]\n" + cb) if cb else ""))
    return _narrate_table(_MAINPAT_SYS, user, "주요 패턴 분석",
                          {"groups": groups_text[:8000]})


_STEPDIFF_SYS = """당신은 반도체 초임계 CO2 세정설비의 수석 공정 분석 엔지니어다.
입력은 통합 골든(정상 구간 전체 평균) 대비 센서×스텝 차이(원단위 ℃/bar)를 전수 집계해
자동으로 추린 목록(지속 이탈·급변)이다. 공정 엔지니어가 바로 조치할 수 있는 서사로 정리하라.

가상센서(차분) 해석 규칙 — 소견이 있으면 반드시 최우선으로 서술:
- dT_Front-Out = t_002 − t_003 (Front 유입−배출 온도차)
- dT_Back-Out = t_001 − t_003 (Back 유입−배출 온도차)
- dT_Front-Back = t_002 − t_001 (상부/하부 유입 온도 균형)
- dP_VslF-Out/dP_VslB-Out = Vessel 압력 − 배출부: ★골든이 답지 — 골든이 그 스텝에서
  확실히 양수인데 웨이퍼가 음수화하면 역류 신호(골든이 음수를 그리는 스텝은 정상 위상,
  면제). 압력은 스케일이 작아 수치가 작아도 유의함을 명시하라.
- [물리 위반] 항목(아치 붕괴·밸브 무반응·IO 미개방·밸브 타이밍)은 설비 동작 방식상
  "나와서는 안 되는 패턴"이 나온 것 — 크기와 무관하게 최우선으로 서술한다.
  아치 붕괴 = dP_VslF-VslB 가 골든처럼 회복(아치)하지 않고 포화(saturation)된 것.
- dP_VslB-Out = p_007 − p_003: 하락·음수화면 Back 쪽 역류 유추
- dP_VslF-VslB = p_008 − p_007: 상부/하부 분사 압력 균형

입력은 세 요소로 구성된다 — 반드시 셋을 결합해 한 사건으로 서술하라:
[요소① 음/양압 상태] outlet 기준 부호(음압=역류) → 방향 판정의 근거
[요소② 스텝별 골든 대비 차이] 크기(원단위)와 유의도(문턱 배수) → 심각도의 근거
[요소③ 패턴 변화] 시간 전개(신규 발생·점증·순차 변화) → 진행 양상의 근거
좋은 예: "스텝4의 dP_VslF-Out 은 7/12 기점 음압 전환(역류, 요소①), 골든 대비 −8bar 로
문턱의 13배(요소②), 이후 열흘간 점증 유지(요소③) — 배출측 압력 상승 원인 점검 필요"
같은 센서·스텝의 세 요소는 반드시 한 문장/한 단락으로 묶는다.

규칙:
1) 목록에 있는 수치·날짜만 인용한다(원단위 그대로, 지어내지 않는다).
2) 사건별 한 문장: "센서 X는 M/D 기점 스텝N에서 골든 대비 약 Δ 하락(지속 n일, 진행 중)" 형식.
3) 가상센서 소견 먼저 → 실센서. 같은 기간·같은 유로 계열의 사건은 묶어
   원인→영향 순서(설비 사실의 유로 방향)로 연결을 시도한다.
4) Cart/Heater 추종형 센서는 원인으로 단정하지 않는다(SV 미상).
5) 마지막에 '요약' 3줄 이내: 무엇이 언제부터 어디서 이상인지 + 권장 확인 항목.
표만 나열하지 말고 문장으로 서술한다."""


def narrate_stepdiff(findings_text: str, extra_text: str = "") -> dict:
    """스텝 차이 자동 추림 목록의 LLM 정리(가상센서 해석 우선). 비판자 토론 포함."""
    flow = _flow_facts()
    phys_txt, _cfg9 = load_physics()
    user = ("[설비 사실]\n" + flow
            + (("\n\n[물리 법칙 — 사용자 문서(항상 우선 참조)]\n" + phys_txt)
               if phys_txt else "")
            + "\n\n[자동 추림 — 통합 골든 대비 원단위 차이]\n" + findings_text
            + (("\n\n[참고]\n" + extra_text) if extra_text else ""))
    return _narrate_table(_STEPDIFF_SYS, user, "스텝 차이 정리",
                          {"findings": findings_text[:8000]})


_PATTERN_SYS = """반도체 설비 검사 결과를 고객 보고서 형식으로 정리한다.

절대 규칙(고객사 제출용):
1) 인공지능 말투 금지 — "분석해 보면", "~로 보입니다", "흥미롭게도", 인사말,
   자기 언급, 이모지 전부 금지. 개조식 결론만 쓴다.
2) 형식: 한 줄 = 한 결론. 예)
   - 3/8 ~ 3/10 이상 패턴 감지(직선 하강 5장) — 확인 요망
   - 3/12 아치 소실 2장 — 확인 요망
   - 그 외 기간 특이사항 없음
3) 입력 표의 날짜·장수·수치만 인용한다. 없는 내용을 만들지 않는다.
   입력에 있는 다섯 지표(직선 하강·아치·W패턴·온도 계통·압력 계통) 외의
   센서·규칙·항목은 언급 금지. 아치는 면적 기준으로만 서술한다(아치 개수 언급 금지).
   온도/압력 계통은 잔차 z 기준으로만 서술한다(수식·모델 내부 언급 금지).
3-1) [기간 추세]가 있으면 규칙별로 한 줄씩 추세를 서술한다. 예)
   - 1/3부터 W패턴 정도가 심해지다가 1/8에 최대치에 다다름 — 확인 요망
4) 원인 분석 금지 — 관찰된 차이(기간·장수·오프셋·패턴 점수)만 서술한다.
   [골든 대비] 절이 있으면 오프셋(℃/bar)과 패턴 점수 차이를 한 줄씩 요약한다.
   마지막 줄은 종합 1줄.
5) 전부 정상이면 "전 기간 이상 패턴 없음" 한 줄로 끝낸다."""


def narrate_pattern(facts_text: str) -> dict:
    """패턴 인식 결과의 고객 보고용 정리 — 담백한 개조식 결론만."""
    return _narrate_table(_PATTERN_SYS, "[패턴 인식 결과]\n" + facts_text,
                          "패턴 정리", {"findings": facts_text[:6000]})


_EVENT_SYS = """당신은 반도체 초임계 CO2 세정설비의 수석 공정 분석 엔지니어다.
입력은 공정 엔지니어가 기록한 불량/조치 레이블(사건)과, 그 사건일 직전 기간을
자기 대조(사례-교차)한 계측 변화 랭킹이다. 레이블은 '결과'이고 목록은 그 전후의
'계측 변화'다 — 목적은 원인 후보를 좁혀 주는 것.

추론 규칙:
1) 발화 시점이 사건일보다 앞서고, 유로상 상류(설비 사실의 배관 순서)인 후보를
   원인 우선으로 놓는다. 사건일 이후에만 변한 것은 결과/조치 흔적일 수 있다.
2) 점진(며칠에 걸친 드리프트)=마모·막힘·오염·센서 드리프트 계열 가설,
   급발(당일 점프)=밸브·제어·레시피 변경 계열 가설로 나눠 서술한다.
3) [물리 위반](아치 붕괴·밸브 무반응·IO 미개방·부호 위반)이 사건 창과 겹치면
   크기와 무관하게 최우선 원인 후보로 다룬다.
4) 같은 레이블의 다른 사건과 공통 후보가 있으면 재발 서명으로 명시한다.
5) 목록에 있는 수치·날짜만 인용한다. 마지막에 '확인 권장' 2~3개(현장에서 볼 것)를
   제시한다. 추종형(Cart/Heater) 단정 금지."""


def narrate_event(event_text: str, extra_text: str = "") -> dict:
    """레이블 사건의 원인 후보 서사. 비판자 토론 포함."""
    flow = _flow_facts()
    phys_txt, _c9 = load_physics()
    user = ("[설비 사실]\n" + flow
            + (("\n\n[물리 법칙 — 사용자 문서]\n" + phys_txt) if phys_txt else "")
            + "\n\n" + event_text
            + (("\n\n[참고]\n" + extra_text) if extra_text else ""))
    return _narrate_table(_EVENT_SYS, user, "사건 원인 분석",
                          {"event": event_text[:8000]})


def narrate_fallback(packet: dict) -> str:
    """LLM 없이도 읽을 수 있는 규칙 기반 요약(간이)."""
    out = ["## 결론(규칙 엔진 요약 — LLM 미사용)"]
    segs = packet.get("segments") or []
    if segs:
        out.append("문제 구간(심각도순): " + " > ".join(
            f"{g['segment']}(max {g['max_z']}σ·유의 {g['n_sig']}·발화 {g['n_fire']}/전파 {g['n_prop']})"
            for g in segs[:4]))
    if packet["candidates"]:
        c = packet["candidates"][0]
        out.append(f"1순위: **{c['part']}** — {c['type']} ({c['why']})")
        for c2 in packet["candidates"][1:]:
            out.append(f"{c2['rank']}순위: {c2['part']} — {c2['type']}")
    else:
        out.append("유의한 원인 후보 없음(3σ 지속 이탈 없음)")
    for ch in packet["chains"]:
        ds = " → ".join(f"{d['sensor']}({d['dev']['dev']:+.1f}℃)" for d in ch["downstream"][:4])
        out.append(f"- [{ch['rule']}] {ch['origin']} → {ds} · {ch['note']}")
    for s in packet["side_findings"][:5]:
        out.append(f"- (별개) {s['sensor']} {s['dev']['dev']:+.1f}℃ — {s['note']}")
    return "\n".join(out)


def narrate_dte(packet: dict, case_files: list, host: str = None, timeout: int = None,
                thinking_level: str = None) -> dict:
    """deep-thinking-engine 세션으로 검증형 RCA.

    규칙 엔진 후보를 '가설 시드'로 Goal 에 넣고, DTE 가 도구 검증
    (보정 p·홀드아웃·선행성 3관문)을 반복한 최종 보고를 받는다.
    사건 폴더 자료(data_files)는 DTE 서버와 같은 호스트의 경로여야 한다."""
    host = (host or os.environ.get("DTE_HOST", "")).rstrip("/")
    if not host:
        raise RuntimeError("DTE_HOST 미설정")
    if not host.startswith("http"):
        host = "http://" + host
    timeout = timeout or int(os.environ.get("DTE_TIMEOUT", "900"))
    wide = (thinking_level == "wide")
    obs_txt = " / ".join(packet.get("observations", [])[:20]) or "관측문 없음"
    if wide:
        # 광탐색: 선입견(후보 시드·판단 지침) 없이 기초 사실+관측만 — 가설은 엔진이 넓게
        goal = (f"초임계 CO2 세정설비 웨이퍼 {packet['wafer_id']} (유형 {packet['recipe']}) 이 "
                f"불량으로 판정됐다. 관측(부호는 사실): {obs_txt}. "
                "가능한 원인 가설을 스스로 폭넓게 세우고(구간·부품·제어·이월오염·측정계 등 서로 다른 "
                "각도로), 각 가설을 데이터로 검증해 결론을 내라. 기각한 가설도 사유와 함께 남겨라. "
                "판단 불가면 판단 불가라고 답하라.")
    else:
        cands = "; ".join(f"{c['rank']}순위 {c['part']}({c['type']}: {c['why']})"
                          for c in packet.get("candidates", [])) or "없음"
        goal = (f"초임계 CO2 세정설비 웨이퍼 {packet['wafer_id']} (유형 {packet['recipe']}) 불량의 "
                f"근본 원인을 규명하라. 규칙 엔진의 사전 후보: {cands}. "
                "각 후보를 선행성·재현성으로 검증하고, 공정 엔지니어가 이해할 인과 체인으로 결론을 내라. "
                "판단 불가면 판단 불가라고 답하라.")
    constraints = [                                    # 설비 '사실'만 — 판단 지침은 넣지 않는다
        "H 라인 히터는 자체 고정설정(피드백 없음) — 하류 변화에 보상하지 않는다",
        "Vessel 은 Cart(열원)가 Heater(존 외부온도=목표)를 맞추는 캐스케이드다",
        "가스 T/P 만 배관을 따라 전파된다",
        "scCO2 는 과열→압력상승→벤트 개도 증가→팽창(줄-톰슨) 냉각으로 배기측 저온이 가능하다",
    ]
    if not wide:
        constraints = constraints  # guided 모드는 기존 그대로

    def _post(path, body=None):
        req = urllib.request.Request(host + path,
                                     data=json.dumps(body or {}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    body_s = {"goal": goal, "data_files": [str(f) for f in case_files],
              "constraints": constraints}
    if thinking_level:                                 # low | medium | high | research
        body_s["thinking_level"] = thinking_level
    sid = _post("/sessions", body_s)["session_id"]
    _post(f"/sessions/{sid}/run")
    req = urllib.request.Request(f"{host}/sessions/{sid}/report")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        rep = json.loads(r.read().decode("utf-8"))
    lines = [f"## 결론 (확신도 {rep.get('confidence', 0):.2f})", rep.get("conclusion", "")]
    if rep.get("key_evidence"):
        lines.append("## 핵심 증거")
        lines += [f"- {e}" if isinstance(e, str) else f"- {json.dumps(e, ensure_ascii=False)}"
                  for e in rep["key_evidence"]]
    if rep.get("rejected_alternatives"):
        lines.append("## 기각된 가설")
        lines += [f"- {a.get('claim')}: {a.get('why')}" if isinstance(a, dict) else f"- {a}"
                  for a in rep["rejected_alternatives"]]
    if rep.get("unresolved_questions"):
        lines.append("## 미해결 질문")
        lines += [f"- {q}" for q in rep["unresolved_questions"]]
    stats = rep.get("stats", {})
    lines.append(f"\n(검증 {stats.get('rounds', '?')}라운드 · 가설 {stats.get('hypotheses_explored', '?')}건 "
                 f"중 기각 {stats.get('hypotheses_rejected', '?')} · 도구 호출 {stats.get('tool_calls', '?')}회)")
    return {"narrative": _strip_md("\n".join(lines)),
            "engine": f"deep-thinking-engine({host})", "dte_report": rep}


def _strip_md(t):
    """LLM 출력의 마크다운 볼드(**)·이탤릭 별표 제거 — 화면은 평문 렌더링이라 기호가 그대로 보임."""
    if not isinstance(t, str):
        return t
    return t.replace("**", "").replace("__", "")


def _ollama_chat(host, model, sys_msg, user_msg, timeout, think=None, images=None,
                 num_ctx=16384):
    """ollama 채팅 1회. 빈 응답이면 thinking 끄고 1회 재시도.

    (Qwen3.6 등 thinking 모델은 긴 과제에서 생각에 토큰을 소진해 content 가
    비는 경우가 있음 — think=false 강제 시 본문이 나옴.)
    """
    msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else [])         + [{"role": "user", "content": user_msg}]
    payload = {"model": model, "stream": False, "messages": msgs}
    if images:
        # ★ 비전 요청에 options(temperature/num_ctx)가 섞이면 일부 VL 모델이
        #   폭주(무한 사고·타임아웃)한다 — 실측 이분법으로 확인. options 생략.
        payload["messages"][-1]["images"] = images
    else:
        payload["options"] = {"temperature": 0.3, "num_ctx": num_ctx}
    if think is not None:
        payload["think"] = think
    req = urllib.request.Request(host + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = _strip_md((json.loads(r.read().decode()).get("message") or {}).get("content", "").strip())
    if not out and think is None:
        # 빈 응답의 주원인은 '사고(thinking) 도중 컨텍스트 소진'이다 — 생각을 끄기
        #   전에 먼저 컨텍스트를 2배로 늘려 사고를 유지한 채 재시도한다(사용자 요구:
        #   생각모드 보존). 그래도 비면 마지막 수단으로 think=off.
        if num_ctx < 65536:
            print(f"[rca] {model} 빈 응답 — 사고 유지, num_ctx {num_ctx}→{num_ctx * 2} 재시도",
                  flush=True)
            out2 = _ollama_chat(host, model, sys_msg, user_msg, timeout, think=None,
                                images=images, num_ctx=min(num_ctx * 2, 65536))
            if out2:
                return out2
        print(f"[rca] {model} 여전히 빈 응답 — 최후 수단 thinking 끄고 재시도", flush=True)
        return _ollama_chat(host, model, sys_msg, user_msg, timeout, think=False,
                            images=images, num_ctx=num_ctx)
    return out


def debate(narrative: str, packet: dict, host: str, main_model: str, timeout: int) -> dict:
    """적대 토론(하네스 방식): 비판자(gpt-oss 등, RCA_CRITIC_MODEL)가 보고를 공격하고,
    주 모델이 반영해 수정한다. 비판자 미설정/실패 시 원문 그대로."""
    critic = os.environ.get("RCA_CRITIC_MODEL", "").strip()
    if not critic:
        return {"narrative": narrative, "debate": None}
    try:
        attack = _ollama_chat(
            host, critic,
            "너는 반도체 설비 데이터 분석 보고의 적대적 검증자다. 증거 패킷과 보고를 대조해 "
            "공격하라: ①수치·부호 인용 오류 ②패킷에 없는 주장(지어냄) ③빠뜨린 대안 가설 "
            "④논리 비약. 각 항목을 근거와 함께 목록으로. 문제없으면 '이의 없음'만 답하라. "
            "이미지는 볼 수 없으니 텍스트 증거만 다뤄라.",
            "증거 패킷:\n" + json.dumps(packet, ensure_ascii=False)[:12000]
            + "\n\n검증 대상 보고:\n" + narrative, timeout)
        if not attack or "이의 없음" in attack[:20]:
            return {"narrative": narrative, "debate": attack or "이의 없음"}
        revised = _ollama_chat(
            host, main_model, _SYS,
            "아래는 당신의 이전 보고와 검증자의 공격이다. 공격 중 타당한 지적만 반영해 "
            "보고를 수정하라(타당하지 않은 공격은 왜 아닌지 한 줄로 반박을 덧붙여라). "
            "증거 패킷 밖의 사실은 여전히 추가 금지.\n\n[이전 보고]\n" + narrative
            + "\n\n[검증자 공격]\n" + attack
            + "\n\n[증거 패킷]\n" + json.dumps(packet, ensure_ascii=False)[:12000], timeout)
        if revised:
            return {"narrative": revised, "debate": attack}
        print("[rca] 비판자 공격은 수신했으나 수정 생성 실패 — 원문+공격만 표시", flush=True)
        return {"narrative": narrative, "debate": attack}
    except Exception as e:                             # noqa: BLE001
        print(f"[rca] 비판자 토론 실패(원문 유지): {e}", flush=True)
    return {"narrative": narrative, "debate": None}


def narrate(packet: dict, model: str = None, host: str = None, timeout: int = None,
            case_files: list = None, mode: str = "auto", thinking_level: str = None) -> dict:
    """엔지니어용 설명 생성.

    mode: auto = DTE(가능하면) → ollama → 규칙 요약
          fast = ollama 직결(작문형)만
          deep = DTE(검증형) 필수 — 실패하면 사유와 함께 규칙 요약
    """
    timeout = timeout or int(os.environ.get("RCA_LLM_TIMEOUT", "600"))  # 첫 호출 콜드로딩 대비
    user_prompt = ("증거 패킷:\n```json\n"
                   + json.dumps(packet, ensure_ascii=False, indent=1) + "\n```")
    if mode in ("auto", "deep") and case_files and os.environ.get("DTE_HOST"):
        try:
            out = narrate_dte(packet, case_files, thinking_level=thinking_level)
            out["prompt_system"] = "(deep-thinking-engine 세션 — Goal/제약/자료는 패킷 참조)"
            out["prompt_user"] = out.get("dte_report", {}).get("goal", "")
            return out
        except Exception as e:                        # noqa: BLE001
            if mode == "deep":
                return {"narrative": narrate_fallback(packet),
                        "engine": f"fallback — 깊은 검증(DTE) 실패: {e}",
                        "prompt_system": _SYS, "prompt_user": user_prompt}
            print(f"[rca] DTE 실패 → ollama 직결로 진행: {e}", flush=True)
    if mode == "deep" and not os.environ.get("DTE_HOST"):
        return {"narrative": narrate_fallback(packet),
                "engine": "fallback — 깊은 검증 불가: DTE_HOST 미설정",
                "prompt_system": _SYS, "prompt_user": user_prompt}
    host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    model = model or os.environ.get("RCA_LLM_MODEL")
    if not model:
        # ★ 태그 자동 감지 — hf.co/... 형태 등 설치 태그가 제각각이라 기본값 고정이 위험.
        #   /api/tags 에서 qwen 이 들어간 모델을 우선 선택(없으면 첫 모델).
        try:
            with urllib.request.urlopen(host + "/api/tags", timeout=10) as r:
                names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
            model = next((n for n in names if "qwen" in n.lower()),
                         names[0] if names else None)
            if model:
                print(f"[rca] RCA_LLM_MODEL 미설정 → 자동 선택: {model}", flush=True)
        except Exception:                             # noqa: BLE001
            model = None
        if not model:
            model = "qwen3.6:35b-a3b"
    user_msg = {"role": "user", "content": user_prompt}
    # ★ 비전: RCA_VISION=1 이면 설비 구조 도면 PNG 를 함께 제공(ollama images 필드 —
    #   Qwen3.6 GGUF 가 이미지 판독을 지원함을 실측 확인, 2026-09-08)
    if os.environ.get("RCA_VISION") == "1":
        gp = find_equipment_graph()
        if gp:
            import base64
            with open(gp, "rb") as fh:
                user_msg["images"] = [base64.b64encode(fh.read()).decode("ascii")]
            user_msg["content"] += ("\n\n첨부 이미지는 이 설비의 유로/센서 구조 도면이다. "
                                    "인과 체인 서술 시 도면의 연결 관계를 참조하라.")
    body = json.dumps({
        "model": model, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": _SYS},
            user_msg],
    }).encode("utf-8")
    try:
        req = urllib.request.Request(host + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read().decode("utf-8"))
        txt = (j.get("message") or {}).get("content", "").strip()
        if txt:
            db = debate(txt, packet, host, model, timeout)
            eng = f"ollama:{model}" + (f" + 비판자:{os.environ.get('RCA_CRITIC_MODEL')}"
                                       if db.get("debate") else "")
            return {"narrative": db["narrative"], "engine": eng,
                    "debate": db.get("debate"),
                    "prompt_system": _SYS, "prompt_user": user_prompt}
        raise RuntimeError("빈 응답")
    except Exception as e:                             # noqa: BLE001 — 폴백으로 항상 결과 제공
        return {"narrative": narrate_fallback(packet),
                "engine": f"fallback(규칙 요약) — LLM 실패: {e}",
                "prompt_system": _SYS, "prompt_user": user_prompt}
