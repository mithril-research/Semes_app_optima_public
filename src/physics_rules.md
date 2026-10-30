# 설비 물리법칙 문서 (physics_rules.md)

이 문서는 분석 시스템이 **항상 참조하는 물리 근거**다. 1부는 자유 서술(사용자 작성·수정),
2부는 기계 규칙(JSON — 앱 규칙엔진이 파싱). 그림·추가 규칙은 이 문서에 이어서 적으면
분석에 반영된다.

판정 철학: **골든(정상 구간 평균)이 답지다.** 절대 규칙("음수면 불량")이 아니라,
골든이 그리는 파형·부호·IO 응답을 기대 거동으로 삼고, 웨이퍼가 그걸 위반하면 검출한다.

## 1부 — 설비 동작 방식 (사용자 서술, 2026-09-13)

1. Reservoir 에서 기체를 목표 온도까지 올린 후 공급한다.
2. 9000002이 열리면 9000006, 9000007, 9000008 까지 연결된 배관으로 유체가 이동한다.
3. 세 IO 중 열리는 밸브의 배관으로 유체가 이동한다.
   (9000006 = SlowFill, 9000007 = QuickFill, 9000008 = QuickFill2)
4. 세 밸브 중 하나라도 열리면 해당 Fill Heater 를 지나 T2, T3 로 나뉘는 두 배관
   전부로 유체가 공급된다.
5. T2 기준으로 Front/Back 분기. Front 라인은 h_003 → Front_valve → h_0162
   → 9000003 IO 밸브까지 도달한다.
6. T2 뒤 분기 배관은 h_0053(H16) → CO2BackFill(T3) → h_001 → h_002 →
   9000004 IO 밸브까지 도달한다. H16 존재로 보아 Back 이 Front 보다 길다고 추정.
7. 9000003/9000004 중 열리는 밸브에 따라 P4(p_008)/P5(p_007)가
   반응한다. Front·Back 모두 Vessel 로 공급된다.
8. 9000004이 열리면 P5 상승. 기체가 Vessel 로 들어가며 Back 기체가 P4 배관까지
   도달하므로 P4 < P5 가 된다.
9. 이후 9000003가 열리면 Front 유체가 P4를 지나므로 P4 상승.
10. 따라서 dP_VslF-VslB(P4−P5) 그래프는 9000004/9000003 개폐를 반영해야 한다.
11. **정상(골든): 스텝 초반 Back 우세로 내려갔다가 Front 열림으로 상쇄되는 아치형.
    불량: 아치가 아니라 회복 없이 포화(saturation)됨.** ← 반드시 검출할 결함.

(여기부터 이어서 작성하세요 — 새 규칙·그림 설명·예외 등)

## 2부 — 기계 규칙 (앱이 파싱, 형식 유지)

```json
{
  "io_map": {
    "9000002": {"role": "supply_open", "sensor": "p_005_PV", "dir": "+"},
    "9000006": {"role": "fill_slow", "sensor": "p_005_PV", "dir": "+"},
    "9000007": {"role": "fill_quick", "sensor": "p_005_PV", "dir": "+"},
    "9000008": {"role": "fill_quick2", "sensor": "p_005_PV", "dir": "+"},
    "9000003": {"role": "front_open", "sensor": "p_008_PV", "dir": "+"},
    "9000004": {"role": "back_open", "sensor": "p_007_PV", "dir": "+"}
  },
  "arch_rules": [
    {"sensor": "dP_VslF-VslB", "open_first": "9000004", "open_second": "9000003",
     "recover_ratio_min": 0.5, "recover_abs_min": 0.3, "depth_sigma_min": 2.0}
  ],
  "sign_rules": {
    "sensors": ["dP_VslF-Out", "dP_VslB-Out"],
    "note": "골든이 그 스텝에서 확실히 양수일 때만 웨이퍼 음수화를 위반으로 판정",
    "golden_pos_margin_sigma": 2.0
  },
  "io_response": {"window_frac": 0.10, "min_rise_ratio": 0.3},
  "io_timing": {"tol_frac": 0.02},
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
                  "k": 3.0, "min_run": 5},
    "w_prom": {"sensor_regex": "(?i)^t_.*outlet", "step": 4,
               "mid_ge_c": 0.5, "high_ge_c": 1.5, "min_width_pct": 2.0,
               "note": "골든 비교 v179: 스텝 4 최저점 이후 '되떨어짐 깊이(prominence)' — 단조 급상승·플래토 뒤 상승은 미검출, 기본은 면적 규칙 대신 모양 규칙만"},
    "arch_slope": {"sensor_regex": "(?i)^dp_.*(vslf-vslb|front.*back)", "steps": [1, 3],
                   "mid_le_ratio": 0.6, "high_le_ratio": 0.3, "lr_fix": true,
                   "note": "골든 비교 v179: 스텝 첫점→최하점 첫 도달점 기울기의 골든 대비 비 — 완만할수록 불량, 길이비 보정 기본"}
  },
  "exclude_sensors_regex": "(?i)heat|vessel(up|down)?cart"
}
```
