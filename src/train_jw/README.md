# IO-Recon-TCN

OK wafer only future-forecast anomaly detection with a masked-future TCN.

## 개요

OK 웨이퍼만으로 미래 센서를 예측하는 TCN 기반 이상탐지/설명 도구입니다.
과거 센서 흐름과 미래 IO/H 입력으로 미래 T/P 또는 지정된 target 센서를 예측하고,
예측값과 실제값의 복원오차를 wafer anomaly score로 집계해 OK/NG를 판정합니다.

| 작업 | 커맨드 | 주요 산출물 |
| --- | --- | --- |
| 하이퍼파라미터 탐색 + 학습 | `tcn-search` | `tcn_forecast.pt`, `search_best.json`, search progress |
| 검증/블라인드 예측 리포트 | `tcn-explain` 또는 `recon-explain` | `09_예측값_원본.csv`, `06_예측값.png`, wafer별 overlay/heatmap/RCA |
| 모델/메타 내보내기 | `tcn-export` | 학습된 모델 기반 평가/리포트 산출물 |
| ONNX 변환 | `convert-onnx` | `tcn_forecast.onnx`, `tcn_forecast.npz`, `recon_meta.npz` |
| trial artifact 요약 | `summarize-trial-artifacts` | `trial_artifacts_summary.csv` |
| trial artifact 시간순 예측 | `plot-trial-artifact-predictions` | `artifacts0001_예측값_시간순.png`, `artifacts0001_예측값_시간순_원본.csv` |

`transformer-search`, `transformer-export`, `transformer-explain`은 하위호환 별칭이며
실제로는 TCN 파이프라인을 실행합니다.

## 설치

Python 3.9 이상과 PyTorch가 필요합니다. GPU 학습을 권장하며, PyTorch는 사용하는
CUDA 버전에 맞춰 설치하세요.

```bash
pip install torch numpy pandas scikit-learn matplotlib optuna
```

ONNX 산출물을 함께 생성/검증하려면 아래 패키지도 설치합니다.

```bash
pip install onnx onnxruntime
```

## 데이터 준비

기본 데이터 구조는 아래와 같습니다.

```text
data/
  OK/             # train OK wafer
  NG/             # 필요 시 설명/검증용 NG wafer
  VAL/
    OK/           # validation OK wafer
    NG/           # validation NG wafer
```

레시피별로 분리할 때는 `--data-name 29a`처럼 지정합니다. 이 경우
`data/29a/OK`, `data/29a/NG`, `data/29a/VAL/OK`, `data/29a/VAL/NG`를 사용합니다.

기본 입력 파일 형식은 `auto`입니다. `.csv`는 기존 CSV 방식으로 읽고, `.log`는
기존 `log2csv.py` 기준의 `WITH_IO=True`, `ONLY_HTP=True` 변환을 메모리에서
적용합니다. 같은 실행 안에서 train은 `.log`, VAL은 `.csv`처럼 섞여 있어도
확장자별로 자동 판별합니다. 형식을 강제하려면 `--file-format csv` 또는
`--file-format log`를 지정합니다.

파일명만으로 recipe를 알 수 없으면 `manual_recipes.json`에 wafer 파일명과 recipe를
매핑해 사용합니다.

```json
{
  "VAL_OK_001.csv": "29a"
}
```

## 빠른 실행

가장 기본적인 학습/설명 실행 예시는 아래와 같습니다. `--h-cart-mode`는 필수입니다.

```bash
python run.py tcn-search --h-cart-mode include --trials 300 --n-workers 2
python run.py tcn-explain --h-cart-mode include
```

레시피별 산출물을 분리하려면 데이터/아티팩트/리포트 이름을 함께 지정합니다.

```bash
python run.py tcn-search --h-cart-mode include --data-name 29a --artifacts-name artifacts_29a
python run.py tcn-explain --h-cart-mode include --artifacts-name artifacts_29a --report-name recon_29a
```

입력 형식을 직접 지정하려면 아래처럼 실행합니다. 기본값인 `auto`에서는 `.csv`와
`.log`를 확장자별로 자동 판별합니다.

```bash
python run.py tcn-search --h-cart-mode include --file-format auto
python run.py tcn-explain --h-cart-mode include --file-format log
```

SEQUS log 파일을 하나의 폴더로 복사/이동하며 파일명을 표준화하려면 아래처럼 실행합니다.

```bash
python flatten_logs.py SRC_DIR TGT_DIR
python flatten_logs.py SRC_DIR TGT_DIR --move
```

SEQUS 구조에서 특정 설비명/레시피에 해당하는 log만 골라 복사/이동하려면
아래처럼 실행합니다.

```bash
python copy_recipe_logs.py SRC_DIR TGT_DIR --equipment EQ_NAME --recipe 29a
python copy_recipe_logs.py SRC_DIR TGT_DIR --equipment EQ_NAME --recipe 29a --move
```

이미 평탄화된 `(제어기)_(yyyymmddhh)_(유닛명)_(레시피명)_(기존파일명).log`
파일을 날짜별 폴더로 정리하고 마지막 `Time` 기준으로 번호를 붙이려면 아래처럼 실행합니다.

```bash
python organize_flat_logs.py SRC_DIR TGT_DIR
python organize_flat_logs.py SRC_DIR TGT_DIR --move
```

정리된 log 루트 아래의 가장 하위 날짜 폴더들을 순회하며 `tcn-explain`을 한 번씩
실행하려면 아래처럼 실행합니다. `--` 뒤 인자는 그대로 `run.py tcn-explain`에
전달됩니다.

```bash
python batch_tcn_explain_logs.py ROOT_DIR --report-root reports_batch -- \
  --h-cart-mode include --artifacts-dir artifacts --pred-batch 1024
```

리포트 루트에서 `07_예측값_이름순.png`만 모아 하나의 폴더에 복사하려면
아래처럼 실행합니다.

```bash
python collect_report_predictions.py --report-root reports_batch --tgt-dir prediction_pngs
```

H 센서를 입력/출력 역할로 나누는 현재 profile은 아래처럼 실행합니다.

```bash
python run.py tcn-search --h-cart-mode include --sensor-role-profile h-split-v1
python run.py tcn-explain --h-cart-mode include --sensor-role-profile h-split-v1
```

검색 성능 기준과 threshold 산정 방식을 명시하는 예시는 아래와 같습니다.

```bash
python run.py tcn-search --h-cart-mode include \
  --threshold-wise recipe \
  --threshold-source train-ok-sigma \
  --threshold-sigma 3.5 \
  --search-objective correct
```

모든 search trial을 독립 artifact 폴더로 남기려면 아래 옵션을 추가합니다.

```bash
python run.py tcn-search --h-cart-mode include --save-trial-artifacts
```

저장된 `artifacts0001`, `artifacts0002`, ... 폴더들의 성능을 하나의 CSV로
요약하려면 아래처럼 실행합니다.

```bash
python run.py summarize-trial-artifacts --artifacts-dir artifacts
```

저장된 trial artifact들을 같은 데이터에 대해 순회 예측하고, wafer 종료 시간순 산점도와
원본 CSV를 artifact별로 저장하려면 아래처럼 실행합니다.

```bash
python run.py plot-trial-artifact-predictions \
  --artifacts-dir artifacts \
  --data-name blind_logs \
  --file-format log \
  --trial-pred-out-dir reports/trial_predictions
```

특정 trial artifact 하나만 예측하려면 아래처럼 실행합니다.

```bash
python run.py plot-trial-artifact-predictions \
  --artifacts-dir artifacts \
  --trial-artifact-dir artifacts/artifacts0005 \
  --data-name blind_logs \
  --file-format log \
  --trial-pred-out-dir reports/trial_predictions
```

학습된 artifact 폴더에서 ONNX와 메타 파일만 생성하려면 아래처럼 실행합니다.

```bash
python run.py convert-onnx --artifacts-dir artifacts
```

학습 때 사용한 sensor 역할/profile, `--h-cart-mode`, score 집계 옵션(`--topk`,
`--resid-agg`, `--wafer-agg`)은 explain에서도 동일하게 맞추는 것을 권장합니다.
단, `tcn-explain`은 저장된 모델 schema와 저장 metadata를 우선 사용하며, CLI로
명시한 일부 옵션은 저장값을 덮어씁니다.

## 주요 산출물

`tcn-search`는 기본적으로 `artifacts/` 아래에 결과를 저장합니다.

| 파일 | 내용 |
| --- | --- |
| `tcn_forecast.pt` | 최적 TCN 모델 |
| `search_best.json`, `best_search.json` | best trial 파라미터, threshold, wafer score |
| `search_trial_progress.csv` | trial별 진행/성능 로그 |
| `search_best_progress.csv`, `search_best_progress.png` | best 갱신 이력과 best trial 파라미터 수 |
| `tcn_search_command.txt` | 학습 실행 커맨드 |

`--save-trial-artifacts`를 사용하면 성공한 각 trial도 아래처럼 독립 artifact로 저장됩니다.

```text
artifacts/
  artifacts0001/
    tcn_forecast.pt
    search_best.json
    best_search.json
    search_trial_progress.csv
    search_best_progress.csv
    search_best_progress.png
    tcn_search_command.txt
  artifacts0002/
    ...
```

각 `artifacts0001`, `artifacts0002` 폴더의 `search_best.json`은 해당 trial 자체를
best로 간주한 형태라 다른 앱에서 일반 best artifact처럼 사용할 수 있습니다.

`summarize-trial-artifacts`는 `artifacts0001`, `artifacts0002`, ... 폴더를 순회해
아래 CSV를 생성합니다.

| 파일 | 내용 |
| --- | --- |
| `trial_artifacts_summary.csv` | artifact별 맞춘 개수, TP/TN/FP/FN, precision, recall, f1-score, val_mse, n_params, threshold, params |

`plot-trial-artifact-predictions`는 기본적으로 `artifacts0001`, `artifacts0002`, ...
폴더를 순회하면서 지정 데이터의 wafer 종료 시각 기준 산점도와 CSV를 생성합니다.
`--trial-artifact-dir`를 지정하거나 `--artifacts-dir` 자체가 `artifacts0005` 같은
단일 trial artifact 폴더이면 해당 artifact 하나만 처리합니다.
출력 디렉토리에 `<artifact명>_예측값_시간순.png`와
`<artifact명>_예측값_시간순_원본.csv`가 이미 모두 있으면 해당 artifact는 다시
계산하지 않고 건너뜁니다.
`.log`는 마지막 행의 `Date`와 `Time`을 실제 x축 시간으로 사용하므로 중간에 빈
시간대가 있으면 그래프에도 그만큼 공백이 생깁니다. 불완전한 artifact나 읽기 실패한
log 파일은 건너뛰고 경고 CSV에 기록합니다.

| 파일 | 내용 |
| --- | --- |
| `<artifact명>_예측값_시간순.png` | threshold를 1로 normalize한 시간순 산점도. 레시피별로 marker 모양을 다르게 표시 |
| `<artifact명>_예측값_시간순_원본.csv` | 시간순 wafer별 raw score, threshold, normalize score, 판정 |
| `예측값_시간순_경고.csv` | 처리 실패 artifact/log 경고 목록 |

`convert-onnx`는 지정한 `--artifacts-dir`에 아래 파일을 저장합니다.

| 파일 | 내용 |
| --- | --- |
| `tcn_forecast.onnx` | TCN forecast network ONNX 단일 파일 |
| `tcn_forecast.npz` | sensor schema, normalization, residual 기준 메타 |
| `recon_meta.npz` | 배포/검증 호환용 메타 파일 |

`tcn-explain`/`recon-explain`은 기본적으로 `reports/<report-name>/` 아래에 리포트를 저장합니다.
같은 report 폴더가 이미 있으면 기존 결과를 덮어쓰지 않고 `<report-name>_try01`, `<report-name>_try02` 순서로 새 폴더를 만듭니다.

| 경로 | 내용 |
| --- | --- |
| `01_이상점수_그래프/` | wafer별 전체/센서별 overlay와 anomaly score 그래프 |
| `02_히트맵/` | wafer별 복원오차 heatmap |
| `03_RCA경로/` | RCA 경로 CSV/MD/PNG |
| `04_웨이퍼요약/` | 성능 요약, wafer 설명 요약 |
| `05_스텝요약/` | ProcessStep별 이상 점수 요약 |
| `06_예측값.png`, `07_예측값_이름순.png` | wafer 예측값 그래프 |
| `09_예측값_원본.csv` | wafer별 예측 원본 표 |
| `10_예측값_학습셋_원본.csv` | train OK 참고 예측 표 |
| `tcn_explain_command.txt` | explain 실행 커맨드와 artifacts 경로 |

## Current Version

- Version: `v0.96.0`

## Ver. Explanation

### v0.96.0 - 2026-08-18 12:10:51 KST

- `--score-final-avg` 옵션 추가.
- reconstruction score 누적 곡선을 wafer 전체 유효 time-step 수로 나눠 표시해
  초반 구간이 과대하게 보이는 문제를 줄일 수 있도록 search/explain에 반영.
- `plot-trial-artifact-predictions`는 출력 PNG/CSV가 이미 있는 artifact를 건너뜀.

## Model Contract

- With H sensors: past `T/P/H/IO` plus future `H/IO` predicts future `T/P`.
- With `--sensor-role-profile h-split-v1`: past `T/P/target-H/input-H/IO`
  plus future `h_005/Cart/IO` predicts future
  `T/P/h_010/VesselHeater`.
- Without H sensors: past `T/P/IO` plus future `IO` predicts future `T/P`.
- Future target `T/P` values are masked from the input.
- All `ProcessStep == 0` regions are usable as input context, but are excluded from training loss, score, heatmap, and explanation highlight metrics.
- H sensors whose names include `Cart` are controlled by the required CLI option `--h-cart-mode include|exclude`.

## Commands

```bash
python run.py tcn-search --h-cart-mode include --trials 300 --n-workers 2
python run.py tcn-export --h-cart-mode include
python run.py recon-explain --h-cart-mode include
```

H split role profile:

```bash
python run.py tcn-search --h-cart-mode include --sensor-role-profile h-split-v1
python run.py recon-explain --h-cart-mode include --sensor-role-profile h-split-v1
```

Recipe-specific dataset and output folders follow the previous Transformer CLI:

```bash
python run.py tcn-search --h-cart-mode exclude --data-name 29a --artifacts-name artifacts_29a
python run.py tcn-export --h-cart-mode exclude --artifacts-name artifacts_29a --report-name io_rca_29a
python run.py recon-explain --h-cart-mode exclude --artifacts-name artifacts_29a --report-name recon_29a
```

For operator compatibility, `transformer-search`, `transformer-export`, and
`transformer-explain` are accepted as aliases, but they run the TCN model.

## CLI Options

| 옵션명 | tcn-search | tcn-explain | 설명 | type/choices/action | default |
| --- | --- | --- | --- | --- | --- |
| `--trials` | 사용 | 미사용 | Optuna TPE search trial 개수 | `int` | `12` |
| `--seed` | 사용 | 미사용 | train/val split 및 Optuna sampler seed | `int` | `42` |
| `--epochs` | 사용 | 미사용 | search space의 `tcn_epochs`를 단일 값으로 고정 | `int` | `search_space.json`의 `tcn_epochs` 후보 사용 |
| `--no-train` | 사용 | 미사용 | search 종료 후 최종 best 재학습 저장 생략 | `store_true` |  최종 best를 `tcn_forecast.pt`로 저장. 단 checkpoint가 켜져 있으면 search 중 best가 이미 저장될 수 있음 |
| `--n-workers`, `-n-workers` | 사용 | 미사용 | search 병렬 worker 수<br> GPU가 있으면 visible GPU 수, CPU면 1로 자동 설정 | `int` | `0` |
| `--no-checkpoint` | 사용 | 미사용 | search 중 best trial checkpoint 설치 생략<br>`--no-train`과 같이 쓰면 모델 파일을 저장하지 않음 | `store_true` |  best trial을 `tcn_forecast.pt`로 즉시 설치하고 임시 `_trial_checkpoints/*.pt`는 삭제 |
| `--resume` | 사용 | 미사용 | 기존 search progress를 읽어 Optuna를 warm-start하고 best를 이어감 | `store_true` | 새 search 시작 |
| `--save-trial-artifacts` | 사용 | 미사용 | 성공한 각 trial을 `--artifacts-dir/artifacts0001`, `artifacts0002`, ... 독립 artifact 폴더로 저장<br>`--no-checkpoint`를 같이 써도 trial artifact 저장에 필요한 임시 checkpoint는 생성 후 정리 | `store_true` | best artifact만 기존 방식대로 저장 |
| `--stop-on-perfect-val` | 사용 | 미사용 | VAL 셋에서 OK+NG를 전부 맞추면 search 조기 종료 | `store_true` | 전부 맞춰도 성능 우선순위에 따라 탐색 지속 |
| `--search-objective` | 사용 | 미사용 | best trial 선정 우선순위<br>`correct`는 맞춘 개수 -> OK 맞춘 개수 -> NG 맞춘 개수 -> 낮은 OK forecast MSE, `precision`은 precision -> 맞춘 개수 -> NG 맞춘 개수 -> FP 적음 -> 낮은 OK forecast MSE | `correct, precision` | `correct`  |
| `--post-eval-extend-epochs` | 사용 | 미사용 | trial 기본 학습/early-stop 종료 후 VAL 판정 성능 개선 여부를 확인할 추가 학습 epoch 단위<br>`--post-eval-max-extends 0`이면 무시 | `int` | `10` |
| `--post-eval-max-extends` | 사용 | 미사용 | post-eval 추가 학습 최대 반복 횟수<br>각 반복 후 VAL search objective가 좋아진 경우만 채택하고, 좋아지지 않으면 직전 모델로 복구 | `int` | `0`: 기존처럼 추가 학습 없음 |
| `--data-name` | 사용 | 사용 | recipe별 데이터 폴더 지정 | `str` | `data/OK`, `data/NG`, `data/VAL` |
| `--file-format` | 사용 | 사용 | 입력 wafer 파일 형식<br>`auto`: `.csv`/`.log`를 파일 확장자별로 자동 로드, 혼합 가능<br>`csv`: `.csv` 파일만 로드<br>`log`: `.log` 파일만 `WITH_IO=True`, `ONLY_HTP=True` 기준으로 메모리 변환 후 로드. 필수 컬럼 누락 시 실패 | `auto, csv, log` | `auto` |
| `--artifacts-name` | 사용 | 사용 | artifacts 폴더명 지정<br>`--artifacts-dir`가 있으면 무시 | `str` | `artifacts`  |
| `--artifacts-dir` | 사용 | 사용 | artifacts 절대/상대 경로 직접 지정<br>지정하면 `--artifacts-name` 무시 | `str` |  `--artifacts-name` 또는 기본 `artifacts` |
| `--onnx-opset` | 미사용 | 미사용 | `convert-onnx`에서 사용할 ONNX opset version | `int` | `17` |
| `--no-onnx-verify` | 미사용 | 미사용 | `convert-onnx` 후 onnxruntime 검증 생략 | `store_true` | onnxruntime이 있으면 변환 결과 검증 |
| `--trial-summary-csv` | 미사용 | 미사용 | `summarize-trial-artifacts` 출력 CSV 경로<br>상대 경로면 `--artifacts-dir` 아래에 저장 | `str` | `trial_artifacts_summary.csv` |
| `--trial-pred-out-dir` | 미사용 | 미사용 | `plot-trial-artifact-predictions` 출력 디렉토리<br>artifact별 시간순 예측 PNG/CSV와 경고 CSV 저장 | `str` | `--artifacts-dir/trial_predictions` |
| `--trial-artifact-dir` | 미사용 | 미사용 | `plot-trial-artifact-predictions`에서 처리할 단일 trial artifact 폴더 지정<br>지정하면 `--artifacts-dir` 아래 전체 순회 대신 해당 폴더 하나만 처리 | `str` | 미지정: `--artifacts-dir` 아래 `artifacts0001`, `artifacts0002`, ... 전체 순회 |
| `--report-name` | 미사용 | 사용 | report 폴더명 지정<br>`--report-dir`가 있으면 무시. 같은 폴더가 이미 있으면 `_try01`, `_try02` 순서로 새 폴더 생성 | `str` | `tcn-explain` 기본은 `recon` |
| `--report-dir` | 미사용 | 사용 | report 절대/상대 경로 직접 지정<br>지정하면 `--report-name` 무시. 같은 경로가 이미 있으면 `_try01`, `_try02` 순서로 새 폴더 생성 | `str` | `reports/<report-name>` |
| `--h-cart-mode` | 사용 | 사용 | Cart 이름이 들어간 H sensor 포함/제외<br>`--sensor-role-profile h-split-v1`에서는 `exclude` 에러 | ` include, exclude` | 필수  |
| `--h-sensor-preset` | 사용 | 사용 | H sensor 입력/복원 대상 preset<br>`all`은 preset 미사용. `--sensor-role-profile h-split-v1`에서는 `all`만 허용되고 다른 preset은 에러. `tcn-explain`은 저장된 모델 schema가 우선 | `all, heater-block3, heater-12` | `all` |
| `--sensor-role-profile` | 사용 | 사용 | 센서 입출력 역할 프로파일 선택<br>`original`은 기존 센서 역할 방식. `h-split-v1`은 future input을 IO+h_005 3종+Cart 10종으로, target을 P/T+h_010+Vessel heater 8종으로 사용<br>기존 저장 metadata의 `legacy` 값은 호환 alias로 인식 | `original, h-split-v1` | `original` |
| `--future-h-mode` | 사용 | 사용 | 미래 H를 입력으로 쓸지, 예측 대상으로 넣을지 선택<br>`--sensor-role-profile h-split-v1`에서는 `input`만 허용되고 `predict`는 에러. `tcn-explain`은 저장된 모델 schema가 우선 | `input, predict` | `input` |
| `--virtual-sensor-mode` | 사용 | 사용 | 가상 센서 사용 위치 지정<br>`--virtual-sensors`를 쓰면 `both`로 강제되어 이 값은 무시됨 | `off, input, target, both` | `off` |
| `--virtual-sensors` | 사용 | 사용 | 가상 센서를 input+target 모두에 사용<br> 지정 시 `--virtual-sensor-mode both`와 동일. `--virtual-sensor-mode` 값을 덮어씀 | `store_true` | 미사용 |
| `--topk` | 사용 | 사용 | reconstruction score에 포함할 상위 센서 수<br>CLI로 지정하면 저장 topk를 덮어씀<br>search: 전체 센서<br>explain: search 저장 metadata의 topk, 저장값이 없으면 전체 센서 | `int` | 전체 센서 사용 |
| `--resid-agg` | 사용 | 사용 | 센서별 residual 시간축 집계 방식<br>`--resid-agg` 또는 `--wafer-agg` 중 하나라도 지정하면 score aggregation이 켜지고, 미지정된 짝은 `sum`으로 채움. explain에서는 CLI 지정값이 저장 metadata보다 우선 | `sum, p99, max, mean` | `sum`  |
| `--wafer-agg` | 사용 | 사용 | 센서별 score를 wafer score로 합치는 방식<br> `--resid-agg` 또는 `--wafer-agg` 중 하나라도 지정하면 score aggregation이 켜지고, 미지정된 짝은 `sum`으로 채움. explain에서는 CLI 지정값이 저장 metadata보다 우선 | `sum, topk_mean, max, mean` | `sum` |
| `--score-time-avg` | 사용 | 사용 | reconstruction score 누적값을 유효 time-step 수로 나눠 평가/시각화<br>search에서는 threshold 산정/성능 평가 score에 반영. explain에서는 저장 metadata가 있으면 자동 적용되고, CLI로 지정하면 현재 실행 score 정의에 반영 | `store_true` | 기존 방식: time-step 수로 나누지 않는 누적 score |
| `--score-final-avg` | 사용 | 사용 | reconstruction score 누적값을 wafer 전체 유효 time-step 수로 나눠 평가/시각화<br>search에서는 threshold 산정/성능 평가 score에 반영. explain에서는 저장 metadata가 있으면 자동 적용되고, CLI로 지정하면 현재 실행 score 정의에 반영. `--score-time-avg`와 동시에 사용하면 에러 | `store_true` | 기존 방식: time-step 수로 나누지 않는 누적 score |
| `--ng-only` | 미사용 | 사용 | report overlay/RCA 대상을 NG wafer로 제한 | `store_true` | VAL/OK와 VAL/NG 모두 출력 |
| `--prediction-order` | 미사용 | 사용 | 예측값 그래프/표 정렬 방식.<br>`score`: 점수순 정렬<br>`recipe`: recipe별 묶음 정렬 | `score, recipe` | `score`  |
| `--pred-batch` | 미사용 | 사용 | prediction score 계산 시 GPU forward에 한 번에 넣을 forecast window batch 크기<br>2 이상이면 저장된 `tcn_batch` 대신 해당 크기로 `self.net(Xt)`를 호출. PNG/CSV 저장 단계는 CPU 작업이라 직접 가속하지 않음 | `int` | `1`: 저장된 모델의 `tcn_batch` 사용 |
| `--raw-error-level` | 미사용 | 사용 | explain 음영/이상 구간 판정 배수<br>OK residual quantile 기준 초과를 음영 처리. search 성능 계산에는 사용되지 않음  | `float` | `1.0`|
| `--threshold-wise` | 사용 | 사용 | threshold 단위 지정<br>search에서 threshold 산정 단위로 사용. explain에서는 `search_best.json`/저장 threshold의 `threshold_wise`가 있으면 그 값이 CLI보다 우선 | `all, recipe, sensor, step` | `recipe`  |
| `--threshold-source` | 사용 | 미사용 | threshold 산정 방식<br>`train-ok-sigma`는 train OK 최종 wafer score의 평균+k시그마 기준. `tcn-explain`은 저장된 threshold를 사용 | `val-search, train-ok-sigma` | `val-search`  |
| `--threshold-sigma` | 사용 | 미사용 | `train-ok-sigma`에서 사용할 k값<br>`-1`이면 VAL 성능 기준 공통 k 자동 탐색. `--threshold-source val-search`이면 무시 | `float` | `-1.0`: 자동 sigma 탐색 |
| `--threshold-sigma-min` | 사용 | 미사용 | `--threshold-sigma -1` 자동 탐색에서 사용할 sigma 하한<br>고정 `--threshold-sigma` 값과 같이 쓰면 에러. `--threshold-source val-search`이면 무시 | `float` | 제한 없음 |
| `--threshold-sigma-max` | 사용 | 미사용 | `--threshold-sigma -1` 자동 탐색에서 사용할 sigma 상한<br>고정 `--threshold-sigma` 값과 같이 쓰면 에러. `--threshold-source val-search`이면 무시 | `float` | 제한 없음 |
| `--threshold-recipe-multiplier`, `--threshold-recipe-scale` | 사용 | 사용 | recipe별 threshold 배율 지정<br>search에서는 threshold 산정 후 recipe별 배율 적용. explain에서는 저장 threshold 위에 CLI 배율을 추가 적용하며, 해당 recipe는 조정된 threshold로 판정/normalize되어 threshold가 1.0으로 표시 | `append`, `RECIPE=FACTOR` | 레시피별 배율 없음 |
| `--interpolation` | 미사용 | 사용 | explain 출력 해상도 보간 배율<br>2 이상이면 센서/예측/RCA 표시 time resolution을 배율만큼 보간 | `int` | `1` |
| `--suppress-isolated-step-transition-shading`, `-ss` | 미사용 | 사용 | PNG 상단 그래프에서 ProcessStep 경계 전후에 고립된 4σ 적색 음영만 숨김<br>점수 계산, CSV, threshold 판정은 바꾸지 않음 | `store_true` | 고립된 step 경계 음영도 기존처럼 표시 |
| `--step-transition-shading-window`, `-sw` | 미사용 | 사용 | `--suppress-isolated-step-transition-shading` 사용 시 경계 이벤트 후보 범위<br>ProcessStep 경계 전후 N point 안의 peak를 후보로 봄. suppression 옵션이 꺼져 있으면 무시 | `int` | `2` |
| `--step-transition-context-window`, `-sc` | 미사용 | 사용 | `--suppress-isolated-step-transition-shading` 사용 시 고립 여부를 확인할 주변 범위<br>전후 N point 안에서 경계 후보 밖 4σ 초과가 있으면 음영 유지. suppression 옵션이 꺼져 있으면 무시 | `int` | `8` |
| `--smooth-step-boundary-prediction`, `-sp` | 미사용 | 사용 | PNG 상단 그래프의 ProcessStep 경계 주변 예측선 spike를 표시용으로 완화<br>원본 예측값, 센서 finding, anomaly score, CSV는 바꾸지 않음 | `store_true` | 예측선을 원본 그대로 표시 |
| `--step-boundary-smooth-window`, `-bw` | 미사용 | 사용 | `--smooth-step-boundary-prediction` 사용 시 예측선 표시 보정 범위<br>ProcessStep 경계 전후 N point를 직전/직후 값으로 연결. smoothing 옵션이 꺼져 있으면 무시 | `int` | `2` |
| `--error-accum-mode` | 미사용 | 사용 | 센서별 하단 누적 그래프 방식<br>`sensor`: 센서별 전체 시간축 누적<br>`step`: ProcessStep별로 누적을 리셋 | `sensor, step` | `sensor` |
| `--ok-mean-exclude-over-threshold` | 미사용 | 사용 | train OK 평균/band 계산 시 threshold 이상 train OK wafer 제외<br>`--ok-mean-include-over-threshold`설정 시 해당 옵션 꺼짐 | `store_true` | 기본 설정 |
| `--ok-mean-include-over-threshold` | 미사용 | 사용 | train OK 평균/band 계산 시 threshold 이상 train OK wafer도 포함 | `store_false` | 기본 미설정. `--ok-mean-exclude-over-threshold`와 둘 다 쓰면 CLI에서 뒤에 나온 옵션이 최종 적용 |

## Update Log

### v0.95.1 - 2026-08-18 15:02:11 KST

- `plot-trial-artifact-predictions`에서 `--trial-artifact-dir`로 단일 trial
  artifact 폴더만 지정해 시간순 예측 산점도/CSV를 생성할 수 있도록 지원.
- `--artifacts-dir` 자체가 `artifacts0005` 같은 단일 artifact 폴더인 경우에도
  해당 artifact 하나만 처리.

### v0.95.0 - 2026-08-14 10:32:17 KST

- `--score-time-avg` 옵션 추가.
- reconstruction score 누적값을 유효 time-step 수로 나눠 긴 recipe의 누적 길이
  영향을 줄일 수 있도록 search/explain score와 wafer/sensor 누적 그래프에 반영.

### v0.94.1 - 2026-08-13 10:43:09 KST

- `plot-trial-artifact-predictions` 시간순 산점도에서 레시피별 marker 모양을
  다르게 표시.
- 기본 marker는 `29a=원`, `336=네모`, `521=삼각형`으로 표시하고,
  그 외 레시피는 다른 marker를 순환 적용.

### v0.94.0 - 2026-08-13 10:24:10 KST

- `plot-trial-artifact-predictions` 명령으로 `artifacts0001`, `artifacts0002`, ...
  trial artifact를 지정 데이터에 대해 순회 예측.
- wafer 종료 시각 기준 산점도 PNG와 원본 CSV를 artifact별로 저장하고,
  불완전한 artifact/log 파일은 경고 CSV에 기록.

### v0.93.0 - 2026-08-13 09:40:22 KST

- `summarize-trial-artifacts` 명령으로 `artifacts0001`, `artifacts0002`, ...
  trial artifact 성능을 CSV로 요약.
- 요약 CSV에 correct, TP/TN/FP/FN, precision, recall, f1-score, val_mse,
  n_params, threshold, params 기록.

### v0.92.0 - 2026-08-12 19:10:37 KST

- `--save-trial-artifacts`로 성공한 모든 search trial을 독립 artifact 폴더에 저장.
- `convert-onnx` 명령으로 `tcn-explain` 없이 artifacts 폴더에 ONNX/NPZ 변환물 생성.

### v0.91.0 - 2026-08-06 16:53:45 KST

- `--threshold-sigma -1` 자동 탐색에서 사용할 sigma 하한/상한 옵션 추가.
- 고정 `--threshold-sigma` 값과 sigma 범위 옵션을 같이 쓰면 명확한 에러를 출력하고 중단.

### v0.90.0 - 2026-08-05 13:50:42 KST

- `--file-format auto`를 추가하고 기본값으로 변경.
- `auto`에서는 `.csv`와 `.log`가 섞여 있어도 파일 확장자별로 자동 로드.

### v0.89.1 - 2026-08-05 13:25:59 KST

- `copy_recipe_logs.py`의 recipe 매칭 기준을 log 첫 행 10번째 컬럼에서 `'` 제거 후
  앞 3자리로 수정.
- 출력 파일명에도 전체 recipe가 아니라 앞 3자리 recipe key를 사용.

### v0.89.0 - 2026-08-05 13:09:42 KST

- `root/(yyyymmddhh)-SEQUS/(유닛명)` 구조에서 지정 recipe log만 복사/이동하는
  `copy_recipe_logs.py` 유틸리티 추가.
- 레시피는 log 파일 첫 행의 10번째 컬럼(`df.iloc[0, 9]`) 기준으로 확인.

### v0.88.0 - 2026-07-31 10:21:58 KST

- `report-root/(제어기)/(유닛)/(날짜)` 리포트에서 `07_예측값_이름순.png`만 모으는
  `collect_report_predictions.py` 유틸리티 추가.
- 복사 파일명은 `(제어기)_(유닛)_(날짜)_예측값_이름순.png` 형식으로 저장.

### v0.87.2 - 2026-07-30 16:39:22 KST

- `tcn-explain` report 폴더명이 이미 있으면 `_try01`, `_try02` 순서로 새 폴더에 저장.
- `tcn-search` 시작 로그의 목표/동률 기준 설명을 실제 `--search-objective` 기준과 맞춤.

### v0.87.1 - 2026-07-30 15:04:12 KST

- `--sensor-role-profile`의 기존 방식 선택지 이름을 `legacy`에서 `original`로 변경.
- 기존 checkpoint/search metadata에 저장된 `legacy` 값은 호환 alias로 계속 인식.

### v0.87.0 - 2026-07-30 13:41:43 KST

- `tcn-search` trial 학습 종료 후 VAL 판정 성능 기준으로 10 epoch 단위
  추가 학습을 선택적으로 시도하는 옵션 추가.
- 추가 학습 모델은 search objective가 개선될 때만 채택하고, 악화되면 직전
  모델 상태로 복구.

### v0.86.0 - 2026-07-30 12:57:55 KST

- 정리된 log 루트의 가장 하위 `.log` 폴더를 순회하며 `tcn-explain`을
  폴더별로 실행하는 `batch_tcn_explain_logs.py` 유틸리티 추가.
- 각 폴더별 리포트와 `batch_tcn_explain_summary.csv` 실행 요약을 저장.

### v0.85.0 - 2026-07-30 12:31:09 KST

- 평탄화된 log 파일을 `(제어기)/(유닛명)/(yyyymmdd)` 폴더로 정리하는
  `organize_flat_logs.py` 유틸리티 추가.
- 같은 날짜 폴더 안에서는 각 log 파일 마지막 행의 `Time` 값 기준으로
  `001`부터 번호를 부여.

### v0.84.1 - 2026-07-30 10:06:40 KST

- `--pred-batch`가 실제 GPU forward forecast window batch 크기를 제어하도록 수정.
- 기본값 `1`은 저장된 모델의 `tcn_batch`를 그대로 사용.

### v0.84.0 - 2026-07-30 09:45:13 KST

- `tcn-explain` prediction score 계산에서 여러 wafer의 forecast window를 함께
  GPU forward하는 `--pred-batch` 옵션 추가.
- 기본값 `1`은 기존처럼 wafer별 순차 예측.

### v0.83.0 - 2026-07-28 17:21:24 KST

- ProcessStep 경계 표시 보정 옵션 5개에 short flag를 추가.
- 기존 긴 옵션명은 모두 유지.

### v0.82.0 - 2026-07-28 16:56:39 KST

- `--suppress-isolated-step-transition-shading`의 경계 후보 범위와 주변 확인 범위를
  숫자 옵션으로 조절 가능하게 변경.
- `--smooth-step-boundary-prediction`의 ProcessStep 경계 스무딩 범위를 숫자 옵션으로
  조절 가능하게 변경.

### v0.81.0 - 2026-07-28 16:21:47 KST

- `tcn-explain` PNG에서 ProcessStep 경계에 고립된 4σ 적색 음영만 숨기는
  `--suppress-isolated-step-transition-shading` 옵션 추가.
- `tcn-explain` PNG 상단의 ProcessStep 경계 예측선 spike를 표시용으로 완화하는
  `--smooth-step-boundary-prediction` 옵션 추가.
- `tcn-search`의 `search_best_progress.csv`에 best trial 모델 파라미터 수
  `n_params` 열을 기록.

### v0.80.2 - 2026-07-28 14:28:57 KST

- `flatten_logs.py`가 `PMCUNIT7`부터 `PMCUNIT12`까지의 unit 폴더에 있는 `.log`만
  복사/이동하도록 제한.
- `PMCUNIT1`부터 `PMCUNIT6`까지의 `.log`는 처리 대상에서 제외.

### v0.80.1 - 2026-07-28 14:18:08 KST

- `flatten_logs.py` 실행 중 파일 처리 진행률을 표시하도록 수정.
- `tqdm`이 설치돼 있으면 진행률 bar를 사용하고, 없으면 기존 로그 출력으로 동작.

### v0.80.0 - 2026-07-28 13:49:53 KST

- Windows에서도 독립 실행 가능한 `flatten_logs.py` 유틸리티 추가.
- `src/제어기명/날짜시간-SEQUS/유닛명/*.log` 파일을 하나의 target 폴더로 복사/이동하며
  `제어기명_날짜시간_유닛명_레시피앞3글자_파일명.log` 형식으로 이름 변경.

### v0.79.5 - 2026-07-27 13:19:25 KST

- `clean_stuck_sensors.py`에서 숫자로만 된 컬럼명을 모두 IO 센서로 보고 검사에서 제외.
- `IO_ORDER`에 없는 숫자명 IO 컬럼도 포화/죽음 검사 대상에 포함되지 않도록 보강.

### v0.79.4 - 2026-07-27 12:42:04 KST

- `clean_stuck_sensors.py`의 검사 제외 기준을 `log2csv.py`의 `NO_HTP`, `IO_ORDER` 기준으로 변경.
- Cart 계열 등 `NO_HTP`/`IO_ORDER`에 없는 named numeric 센서는 포화/죽음 검사 대상에 포함.

### v0.79.3 - 2026-07-27 12:38:12 KST

- `clean_stuck_sensors.py`의 검사 대상을 H/T/P 계열 컬럼으로 제한.
- IO 숫자 컬럼과 H/T/P 계열이 아닌 named numeric 컬럼은 포화/죽음 검사에서 제외.

### v0.79.2 - 2026-07-27 12:14:38 KST

- `clean_stuck_sensors.py`가 CSV뿐 아니라 탭 구분 `.log` wafer 파일도 검사하도록 수정.
- 폴더 입력 시 `.csv`와 `.log`를 함께 탐색하고, `.log`는 `ProcessStep` 비숫자 행을 제거한 뒤 검사.

### v0.79.1 - 2026-07-27 11:16:36 KST

- `tcn-search`의 `current best` 출력에 해당 best가 나온 trial 번호를 함께 표시.

### v0.79.0 - 2026-07-24 17:49:05 KST

- `--file-format {csv,log}` 옵션 추가.
- `log` 입력은 기존 `log2csv.py`의 `WITH_IO=True`, `ONLY_HTP=True` 변환 규칙을 메모리에서 적용.
- log 학습/추론 로더에서는 `_336_` prefix를 붙이지 않고 기존 recipe 추론 방식을 유지.
- log 필수 컬럼 누락 시 실행을 중단하도록 처리.

### v0.78.12 - 2026-07-24 10:41:24 KST

- `tcn-search` 병렬 worker가 task queue를 잠깐 비었다고 판단해 조기 종료하던 문제를 수정.
- 모든 trial이 끝났을 때 main process가 sentinel을 보내 worker를 정상 종료하도록 변경.

### v0.78.11 - 2026-07-23 17:23:25 KST

- README의 `Shading Criterion` 설명을 현재 코드 기준으로 수정.
- 하늘색 정상 범위 band는 OK residual quantile 기준, 빨간 x축 band는 OK residual sigma 기준임을 분리해서 명시.

### v0.78.10 - 2026-07-23 16:12:06 KST

- README 상단에 설치, 데이터 준비, 빠른 실행, 주요 산출물 안내를 추가.

### v0.78.9 - 2026-07-23 09:59:42 KST

- `tcn-explain` prediction CSV의 표시용 score 컬럼을 `anomaly_score_capped`에서 `anomaly_score_display`로 변경.
- prediction PNG에서 2 초과 score가 있을 때 적용하는 1~2 구간 표시 압축값을 `anomaly_score_display`에 동일하게 저장.
- 기존 artifacts의 train prediction CSV를 report로 복사할 때도 `anomaly_score_display` 컬럼으로 변환하도록 보정.

### 2026-07-23 09:40:28 KST

- `tcn-explain` now applies CLI `--threshold-recipe-multiplier` values on top
  of saved search thresholds.
- Explain-time recipe multipliers affect prediction decisions, CSV/PNG
  normalization, overlay score normalization, and train OK band filtering.
- For adjusted recipes, the adjusted threshold is used as the display scale, so
  the plotted threshold remains 1.0.

### 2026-07-22 19:50:27 KST

- Updated prediction PNG scaling: when any threshold-normalized wafer score
  exceeds 2, only the score portion above threshold 1 is compressed into the
  1-2 display range.
- CSV raw score columns remain unchanged.

### 2026-07-22 19:28:18 KST

- Fixed RCA path row generation when sensor findings exist but flow rank or
  ProcessStep metadata is incomplete.
- Final report CSV/MD writers now run independently and print row counts, so
  one failed output no longer blocks the remaining report files.
- Empty RCA row sets are no longer forced into placeholder CSV/MD files.

### 2026-07-22 19:25:08 KST

- Fixed RCA path report writing so `03_RCA경로/_RCA경로.csv` is created even
  when no RCA path rows are produced.

### 2026-07-22 19:00:45 KST

- Added automatic ONNX export to `tcn-explain`/`recon-explain`.
- Explain reports now attempt to write `tcn_forecast.onnx`,
  `tcn_forecast.npz`, and `recon_meta.npz` into `report_dir`.
- ONNX output verification runs when `onnxruntime` is installed; missing ONNX
  packages are reported without stopping the rest of explain report generation.

### 2026-07-22 15:34:04 KST

- `tcn-search` now writes the executed command metadata to
  `artifacts_dir/tcn_search_command.txt`.
- `tcn-explain` now writes the executed command metadata and selected
  `artifacts_dir` to `report_dir/tcn_explain_command.txt`.

### 2026-07-22 13:10:23 KST

- Added a README CLI option table for `tcn-search` and `tcn-explain`.
- Documented default behavior and option override/error relationships.

### 2026-07-22 09:59:52 KST

- Fixed `--sensor-role-profile h-split-v1` to require 8 Vessel heater target
  sensors instead of 3.
- Updated the h-split-v1 CLI/README descriptions to match the 8 Vessel heater
  target sensor contract.

### 2026-07-21 17:57:17 KST

- Updated `tcn-search` to use Optuna TPE Bayesian search instead of fixed
  random sampling.
- Search progress CSV now records the sampler as `optuna_tpe`.
- `--resume` can warm-start Optuna from previous successful trial rows when
  their parameters still match the current search space.

### 2026-07-21 13:58:06 KST

- Marked the search space expansion as patch version `v0.77.1`.
- Expanded TCN search candidates while keeping the current fixed
  `tcn_epochs=[80]` and `tcn_batch=[512]` settings unchanged.

### 2026-07-21 13:40:24 KST

- Added `--sensor-role-profile h-split-v1`.
- In this profile, future inputs are IO, the 3 h_005 sensors, and the
  10 sensors whose names contain `Cart`; reconstruction targets are P/T,
  `h_010_PV`, and the 8 Vessel heater sensors.
- Added safety checks for exactly 10 Cart sensors, exactly the 3 named
  h_005 sensors, and exactly 8 Vessel heater target sensors.
- `--sensor-role-profile h-split-v1` now rejects `--h-cart-mode exclude`.

### 2026-07-21 13:00:04 KST

- Marked the current code state as version `v0.76.0`.

### 2026-07-21 10:43:35 KST

- Changed `--threshold-sigma` default to `-1`, which automatically selects a
  common sigma multiplier from validation performance for
  `--threshold-source train-ok-sigma`.
- For `--threshold-wise recipe`, the selected sigma multiplier is shared across
  recipes while each recipe still uses its own train OK mean and sigma.
- Capped threshold-scaled explanation PNG values at 3 for readability; CSV and
  JSON outputs keep the uncapped values.

### 2026-07-16 16:52:42 KST

- Added `--threshold-source train-ok-sigma` to set the decision threshold from
  train OK wafer final anomaly scores: `mean + k * sigma`.
- Added `--threshold-sigma` for train OK sigma thresholds.
- Added H sensor input presets:
  - `--h-sensor-preset heater-block3`
  - `--h-sensor-preset heater-12`
- H sensors excluded by an H preset are removed from past input, future input,
  and reconstruction targets. Without these options, the previous behavior is
  preserved.
- `tcn-explain` uses the sensor schema saved in the trained checkpoint, so
  explanation runs follow the H preset used during training.

### 버저닝 기준
- MAJOR: 0.x.x -> 1.0.0
> 기존 코드, 모델, CLI, 산출물 해석이 깨질 수 있는 변경. 기존 아티팩트와 호환되지 않거나 같은 명령으로 돌려도 의미가 달라지는 경우

- MINOR: 0.77.x -> 0.78.0
> 기존 방식 유지한 채로 새 기능 추가된 경우. 새 CLI 옵션, 센서 입출력 변경, 새 임계값 방식, 새 학습/판정 모드 추가 등

- PATCH: 0.77.1 -> 0.77.2
> 기존 기능 안에서 수정하는 경우. 버그 수정, 서치 스페이스 조정, 문서 수정, 그래프/csv 표시 수정, 기존 옵션 동작 보정 등

## Shading Criterion

The explanation plots use two different OK residual references:

- `OK_limit`: empirical OK residual quantile. This controls the blue normal
  range band.
- `OK_residual_sigma`: empirical OK residual standard deviation. This controls
  the red x-axis event band.

The model saves `OK_limit` in raw sensor units:

```text
OK_limit(sensor, step) = quantile(|actual - expected| on OK wafers)
```

Default quantile is `raw_error_quantile = 99.0`. The blue band is:

```text
expected ± OK_limit
```

The red x-axis band does not use `OK_limit`. It uses sigma-scaled residuals:

```text
residual_sigma = |actual - expected| / OK_residual_sigma
red x-axis band = residual_sigma >= large_residual_sigma
```

Default `large_residual_sigma = 4.0`. Darker red means larger sigma excess.
`raw_error_level` is still used by some explanation tables, such as
`over_raw_error_level`, but it does not directly set the blue band width or the
red x-axis band in the current plots.

When `--suppress-isolated-step-transition-shading` is enabled, isolated red
bands whose peak is only near a `ProcessStep` boundary are hidden in PNG plots.
The boundary candidate window defaults to 2 points
(`--step-transition-shading-window 2`), and the surrounding context check
defaults to 8 points (`--step-transition-context-window 8`).
When `--smooth-step-boundary-prediction` is enabled, the blue expected line is
bridged around `ProcessStep` boundaries in PNG plots. The smoothing window
defaults to 2 points (`--step-boundary-smooth-window 2`). All of these options
are display-only and do not change prediction values, CSV outputs, anomaly
scores, or threshold decisions.
