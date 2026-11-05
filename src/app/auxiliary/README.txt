보조 모델 통합 및 StepTime PV 설명 (2026-11-02)

수치 루틴 출처
- StepTime-IsolationForest v0.2.1 a48c0bf77f662e9a371df13d77c8cbc7c6e41438
  step_time_iforest/data.py -> step_data.py (tqdm은 독립 로더에서만 지연 import)
  pv_explainer.py의 추론 수치 함수 -> step_pv_explainer.py
  앱 배치/결과 계약/영향구간/UI 연결은 별도 구현했습니다.
- InceptionTime-Golden b89c843b702b167d4094fa201099da5cb5074c51
  data.py, scoring.py, sensors.py -> golden_data.py, golden_scoring.py, sensors.py
  앱은 PyTorch 학습 모듈 대신 ONNX golden 출력을 사용합니다.

기본 배치 경로 (실행한 exe 폴더 기준 / 소스 실행은 src 폴더 기준)
보조모델/스텝시간/DSV1/step_time_meta.json
보조모델/스텝시간/DSV1/pv_explainer_meta.json
보조모델/스텝시간/DSV1/models/*.joblib
보조모델/스텝시간/DSV1/models/pv_explainer/*.joblib
보조모델/스텝시간/DSV1/stats/*.npz
보조모델/스텝시간/DSV1/stats/pv_explainer/*.npz
보조모델/실린더/DSV1/golden_meta.json
보조모델/실린더/DSV1/onnx_manifest.json
보조모델/실린더/DSV1/models/*.onnx
보조모델/실린더/DSV1/stats/*.npz
각각 DSV1~DSV6 동일 구조입니다. metadata에 저장된 상대 파일 경로를 보존하세요.

artifacts는 반드시 만들어야 하는 폴더 이름이 아닙니다.
챔버 폴더에 metadata를 바로 놓는 것이 기본입니다.
이미 DSV1/artifacts 또는 DSV1/release_01 등 한 단계 아래에 넣은 경우,
metadata가 있는 하위 폴더가 딱 하나라면 해당 폴더를 자동 사용합니다.
직접 metadata가 있으면 직접 경로가 우선합니다. 여러 하위 버전은 임의 선택하지 않습니다.
그 경우 app_config.json의 *_by_chamber에 정확한 metadata 폴더를 지정하세요.
스텝시간은 다른 DSV/공용 모델로 대체하지 않습니다. UNKNOWN 챔버는 보조 점수를 적용하지 않습니다.

설정 (app_config.json, 상대경로는 exe 폴더 또는 소스 src 기준)
step_time_artifacts_root: 스텝시간 DSV 폴더들의 상위 폴더
step_time_by_chamber: {"DSV1": "보조모델/스텝시간/DSV1", ...}
cylinder_artifacts_root: 실린더 DSV 폴더들의 상위 폴더
cylinder_by_chamber: {"DSV1": "보조모델/실린더/DSV1", ...}
환경변수 IO_STEP_TIME_ARTIFACTS_ROOT, IO_CYLINDER_ARTIFACTS_ROOT 지원.
기존 공용 step_time_artifacts / IO_STEP_TIME_ARTIFACTS는 더 이상 사용하지 않습니다.
DSV는 폴더 선택용입니다. 모델 내부 그룹은 원본 경로에서 추출한 equipment,
PMCUNITn, recipe로 구성하며 대소문자/구분자를 임의 변경하지 않습니다.

점수 및 설명
TCN: 기존 recipe multiplier/wafer override 적용 후 버림 전 유효 score/threshold.
StepTime: -IsolationForest.score_samples / 그룹 threshold.
실린더: shape/spike/offset 각각의 threshold 대비 비율 중 최댓값에 아래 scaling 계약 적용.
통합: 사용 가능한 위 세 비율 중 최댓값, 판정 기준 1.
PV 기여도는 설명용 초 단위이며 통합 점수에 추가하지 않습니다.
기존 실린더 점수의 양수 임계값 검증은 이번 PV 변경에서 수정하지 않았습니다.

PV 처리
32매 마이크로배치에서 TCN 결과 생성 후 보조 결과를 합치고 history/archive를 저장합니다.
동일 DSV/모델 digest/학습 그룹/스텝 occurrence별로 PV 계산을 묶습니다.
보조 추론이 읽은 DataFrame을 StepTime/실린더/PV에서 공유하며 PV 때문에 재개방하지 않습니다.
TCN의 기존 파싱 캐시와 보조 원본 로더는 별개입니다.
원본 explain subprocess 또는 CSV 생성은 하지 않습니다.
이벤트당 기본 예측 1회 + 센서 수만큼 batch predict.
측정 가능한 상위 3개 후보에 각 4구간 추가 batch predict (grid가 4점 미만이면 점 수).
행마다 상위 센서가 다를 때도 순위 슬롯으로 묶어 호출 수는 웨이퍼 수에 비례하지 않습니다.
PV 모델과 통계는 전체 artifact digest 기준으로 캐시합니다.

불량 대표 스텝은 IF contribution 우선, 동률이면 robust deviation으로 정합니다.
도달시간 차이가 큰 다른 스텝을 대신 표시하지 않습니다.
IF는 같은 ProcessStep의 occurrence 합계입니다. 반복 스텝은 모든 occurrence를
정확히 구분해 선택하도록 표시하며, IF가 특정 occurrence를 지목했다고 주장하지 않습니다.
지연은 양수 기여 순, 단축은 음수 기여 순으로 후보를 표시합니다.
원본 센서 누락은 마지막에 표시하고, 정상 시 기대되는 반응이 없는 센서는 후보로 유지합니다.
대표 영향구간은 관측 지연/단축 방향과 같은 기여가 가장 큰 구간입니다.
같은 방향 구간이 없으면 대표 음영은 표시하지 않습니다.
구간 기여도 합은 전체 궤적 기여도와 다를 수 있습니다.

화면/저장
이상분석의 소요시간 막대 클릭 -> 스텝별 실제 소요시간 / 정상 중앙값 / 정상 IQR 비교 그래프.
PV 목표 도달, 원인 후보, 영향구간 및 기여도 표시는 제거했습니다.
기존 기록에 PV 자료가 있어도 이 화면에는 표시하지 않습니다.
실린더는 하나의 접기/펼치기 항목을 유지합니다.
PV 누락/오류는 IF 점수 및 다른 모델을 무효화하지 않습니다.
score_components.step_time.pv 전체를 기존 NPZ meta_json에 저장합니다.
상세 API는 저장된 전체 PV를 반환하고 산점도 summary는 대표 스텝/후보만 반환합니다.
원본 파일이 없어져도 저장된 상세를 볼 수 있습니다.
현재 모델로 새 점수를 얻으려면 재검사해야 합니다. PV 자료 유무와 관계없이 소요시간 그래프를 표시합니다.

검증 및 배포
scikit-learn/joblib은 학습 버전과 맞아야 합니다. Win7/Python3.8 배포는 sklearn1.3.2.
v0.2.0 형식의 합성 artifact와 v0.2.1 원본 explain 수치 대조 테스트를 제공합니다.
실제 현장 v0.2.0 joblib 및 Windows7 실행/패키징 검증은 별도로 필요합니다.
모델 파일 자체는 코드에 포함되지 않습니다. 기존 offline 묶음으로 추가 패키지 없이 빌드합니다.
SEMES_STEP_REFERENCE=<v0.2.1 checkout> 및 SEMES_GOLDEN_REFERENCE=<golden checkout> 환경변수를
지정하면 tests/test_step_pv_integration.py, tests/test_integrated_scores.py에서 원본 대조 실행.


실린더 scaling 연동 (2026-11-03 김재원 메일 / InceptionTime-Golden v0.20.0)
- cylinder_by_chamber 명시 경로 -> cylinder_artifacts_root/DSVn -> cylinder_artifacts_root 순서.
- 선택 그룹의 golden_meta.json, onnx_manifest.json, stats_path, onnx_path가 모두
  해당 패키지 안에 있어야 선택합니다. 다른 DSV를 탐색하거나 파일을 섞지 않습니다.
- 공용 루트의 metadata는 루트에 직접 있어야 합니다. 루트의 유일한 DSV 하위
  폴더를 공용 모델로 간주하지 않습니다.
- 완전한 패키지가 선택된 이후 scoring/scaling 오류는 다른 모델로 조용히 대체하지 않습니다.
- golden_meta.json.models[group].combined_score_scaling이 없으면 기존 비율 유지.
- 필드가 있으면 method=linear_threshold_anchored, source=train_normal,
  input_score=max_threshold_ratio, threshold_anchor=output_threshold=1,
  denominator=threshold_anchor-lower_anchor > 0, clip_below_zero=true,
  clip_above=false를 검증합니다. null/잘못된 계약은 실린더만 미적용합니다.
- raw_ratio=max(shape_ratio,spike_ratio,offset_ratio)
  ratio=max(0,(raw_ratio-lower_anchor)/denominator). 상단은 자르지 않습니다.
- shape/spike/offset 및 step_sensor_details 비율에는 scaling을 적용하지 않습니다.
- ratio/raw_ratio/scaling_applied/scaling_version을 history/archive에 저장합니다.
  실린더 그룹 툴팁에 정규화 점수와 기존 threshold ratio를 함께 표시합니다.
- artifact_digest에 train-normal-scaling-v1 계산 계약 버전을 붙이고, 우선순위를
  포함한 후보 경로/파일 버전은 routing_version으로 별도 저장합니다.
  후보 파일 변경은 현재 모델과의 일치 여부 안내에 사용합니다. 과거 기록은 삭제하거나
  숨기지 않으며 검사 당시 통합 점수를 유지합니다(아래 16:15 조회 정책 참조).
- 새 scaling 정보는 v0.20.0으로 재학습한 동일 릴리스 metadata/manifest/models/stats를
  함께 배포해야 생깁니다. 코드 교체만으로 기존 모델에 scaling을 만들지 않습니다.
- 추가 pip 의존성/빌드 bat 변경은 없습니다. 합성 ONNX와 회귀 테스트로 검증하며
  실제 현장 모델 및 Windows7 실행 검증은 별도로 필요합니다.


Windows → Ubuntu 보관 결과 호환 (2026-11-03 오후 메일)
- 소스 실행 기본 위치: src/보조모델/{스텝시간,실린더}/DSVn.
  해당 종류 폴더가 없으면 저장소 최상위 보조모델/{스텝시간,실린더}/DSVn을 탐색합니다.
  두 곳 모두 있으면 src 쪽이 우선합니다. 명시 설정/환경변수는 자동 탐색보다 우선합니다.
  Windows exe 옆 보조모델 폴더는 저장소 최상위로 그대로 복사할 수 있습니다.
- 파일 내용과 POSIX 상대경로를 정렬해 canonical digest를 계산합니다.
  실린더 routing은 후보 우선순위 번호와 내용 해시만 사용하고 설치 경로는 제외합니다.
- 새 결과는 version_scheme=portable-v2를 저장합니다.
- 구버전 Windows/Linux 저장본은 현재 패키지로 과거 해시를 재현하여 검증합니다.
  일치하면 보관된 상세/점수 자체를 사용하며 원본 로그 재추론이나 파일 덮어쓰기를 하지 않습니다.
  실린더는 완전한 우선 패키지의 그룹/해시까지 확인합니다. 상위 후보 그룹이 달라
  이전 선택을 증명할 수 없으면 모델 버전 차이 안내를 표시합니다.
  scaling 정보가 있는 모델에 계약 버전 없는 옛 raw 점수를 새 정규화 점수로 오인하지 않습니다.
- 실제 파일 내용 변경, 필수 파일 누락, 계산 계약 차이는 과거 결과의 안내 상태입니다.
  현재 모델 기준 새 점수를 원할 때만 재검사가 필요합니다.
- metadata의 models\xxx.joblib, stats\xxx.npz 등을 OS에 맞게 읽습니다.
  패키지 외부 절대경로/상위 폴더 탈출 경로는 사용하지 않습니다. PV 설명 모델에도 동일 적용.
- Ubuntu 요구사항은 scikit-learn==1.3.2로 고정했습니다. setup을 다시 실행하거나
  해당 venv에서 requirements-ubuntu.txt를 재설치해야 기존 환경에도 반영됩니다.
  다른 sklearn 버전으로 학습한 joblib을 자동 변환하지 않습니다.
- 미적용 사유는 모델별 화면 안내/툴팁과 앱 로그에 남깁니다(반복 동일 사유는 중복 제한).
- 테스트는 Windows 경로 및 legacy 해시를 재현한 합성 artifact의 이관/신규 추론 검증입니다.
  실제 Windows7 실행 및 현장 보관본 복사 검증을 수행한 것은 아닙니다.


과거 NPZ 우선 조회 정책 (2026-11-03 16:15 메일, 위의 구버전 무효화 정책 대체)
- NPZ meta_json.score_components에 저장된 TCN/StepTime/Cylinder와 integrated를
  그대로 표시합니다. top-level ratio_thr 또는 선택 스텝 점수로 당시 TCN/통합 점수를 대체하지 않습니다.
- current_components는 stored_result/current_model_status/current_model_reason만 덧붙이며
  모델 부재·버전 불일치로 available/ratio/상세 자료를 지우지 않습니다.
- current: 검사 당시 저장 결과 / missing: 현재 보조모델 없음 /
  different: 현재 보조모델과 버전 다름 / unknown: 현재 상태 또는 저장 버전 정보 확인 불가.
  실린더 계산 계약이 다른 경우 현재 scaling 기준으로 표기하지 않고 계약 차이를 안내합니다.
- 저장 통합 결과가 있는 경우 그 값을 우선합니다. 통합 결과 자체가 없는 구형 자료만
  저장된 component의 최대 비율로 요약합니다. 신규 모델 추론은 하지 않습니다.
- 상태 확인 실패(접근 권한 등)도 과거 결과를 차단하지 않습니다.
- 원본 NPZ/원본 로그를 변경하지 않습니다. PV, 실제/Golden 파형 등 존재하는 자료만 표시합니다.
  상세 배열이 없으면 '저장된 상세 자료 없음 · 검사 당시 요약 점수만 표시합니다'로 안내합니다.
- 산점도와 이상분석 모두 '검사 당시 통합 점수'를 사용합니다.
  신규 검사 경로는 그대로 현재 모델/필수 파일/계산 계약을 검증하여 새 결과를 저장합니다.


DSV1 실린더 임계값 점검 (2026-11-03 16:51 메일)
저장소 최상위에서 다음 명령을 실행합니다.
  src/app/venv/bin/python -m src.app.check_cylinder_thresholds
  src/app/venv/bin/python -m src.app.check_cylinder_thresholds --root "src/보조모델/실린더/DSV1" --output cylinder_threshold_report.txt
  src/app/venv/bin/python -m src.app.check_cylinder_thresholds --root "<모델 폴더>" --group "<그룹명>" --show-arrays
- 기본 경로는 src/보조모델/실린더/DSV1, 없으면 저장소/보조모델/실린더/DSV1을 찾습니다.
  챔버 폴더의 단일 패키징 하위 폴더 및 직접 공용 루트도 인식합니다.
- golden_meta.json의 전체 그룹별 전체 임계값, NPZ 키, 배열 shape/dtype/min/max,
  0/음수/NaN/inf 개수와 문제 위치를 출력합니다. 배열별 위치는 기본 50개(--limit 변경 가능).
- 셀 임계값 행은 step_offsets를 숫자순 정렬한 ProcessStep, 열은 feature_names입니다.
  detrended_residual_scale_raw의 행은 ProcessStep이 아니라 정렬된 측정 시점입니다.
- Windows식 stats 경로도 지원합니다. 누락된 그룹 파일은 보고하고 다음 그룹을 계속 검사합니다.
- 현재 점수 모드에서 필요 없는 배열의 부재는 오류로 세지 않습니다.
- 종료 코드 0: 문제 없음, 1: 문제 발견, 2: 메타데이터/실행 오류.
- 읽기 전용 진단입니다. 임계값을 보정하거나 모델/NPZ를 덮어쓰거나 추론하지 않습니다.
- 검사 중 임계비율 오류에는 그룹·점수종류·셀 스텝/센서·score·threshold가 추가됩니다.
  기존 미적용 안내/툴팁/로그에서 확인할 수 있습니다.
- 모든 임계값이 정상이어도 특정 원본 웨이퍼의 비유한 센서값 등은 별도 확인이 필요합니다.

실린더 0 임계값 호환 (2026-11-03 17:59 메일)
- 실린더 shape/spike/offset의 전체 및 셀 임계값 0을 지원합니다.
  임계값>0: score/threshold, 임계값=0 및 score<=0: 0,
  임계값=0 및 score>0: 무한대 의미의 불량입니다. TCN/StepTime 검증은 유지합니다.
- 무한대는 JSON/NPZ에 ratio:null, ratio_infinite:true,
  threshold_zero_exceeded:true로 저장합니다. 종류별 상태는 ratio_states에 보존합니다.
  통합 최대 모델은 cylinder이며 verdict=NG입니다. 유한 점수로 대체하지 않습니다.
- 화면에서는 ∞ 및 '임계값 0 초과' 안내를 제공하고 막대/산점도 위치만 축 상한에 제한합니다.
- 재조회는 저장된 상태를 보존합니다. 기존 available:false 결과는 원본 재검사가 필요합니다.
- 진단 도구에서 임계값 0은 정상 계약으로 인정합니다. 음수/비유한 임계값과
  0 이하 residual scale은 계속 오류입니다. 모델 파일은 수정하지 않습니다.
