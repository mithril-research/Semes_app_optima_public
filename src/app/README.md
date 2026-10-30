# src 제어기 UX (FastAPI 초안)

고객사 제어기(HMI)에 들어갈 검사 화면. **데이터가 폴더로 들어오고 → [검사 시작] →
추론 결과 → 상세 분석(overlay + RCA)** 흐름을 그대로 구현했다.

추론은 `src` 의 학습된 recon 모델·함수를 **그대로 재사용**한다(중복 구현 0):
`ReconTCNResponseModel.load` · `load_dataset` · `detect.score_wafer` ·
`recon_explain.narrate_recon` / `plot_recon_overlay` / `plot_recon_heatmap`.

## 실행

### A. 웹 서버(개발/서버 배포)
```bash
cd /home/node/io_dev
uvicorn src.app.server:app --host 0.0.0.0 --port 8000
# 또는
PORT=8000 python -m src.app.server
```
브라우저: http://localhost:8000

### B. 웹 런처(로컬 실행 = exe 와 동일 동작)
```bash
python -m src.app.launcher     # 서버 기동 + 기본 브라우저 자동 오픈
```

### C. Windows 실행 파일(.exe) — 웹 대시보드 런처
`app/BUILD_WINDOWS.md` 참고. 이 `src/` 폴더를 Windows PC 로 복사 후
`app/build_win.bat` 실행 → `dist/IOAnomalyConsole/IOAnomalyConsole.exe`.
exe 실행 시 서버가 뜨고 브라우저로 대시보드가 열립니다.
(Linux→Windows 크로스컴파일 불가 — 빌드 자체는 Windows 에서)

## 검사 폴더 (데이터 인입 경로)
- 기본: `src/data/inbox/` — 여기에 wafer CSV 가 들어온다.
- 변경: 환경변수 `IO_INBOX=/path/to/folder` 로 덮어쓰기.

## 전제(모델 산출물)
`artifacts/recon_A.pt` (학습 가중치) + `artifacts/recon_search_best.json`
(best 하이퍼파라미터 + 판정 임계). 없으면 먼저:
```bash
python -m src.run recon-search --trials 200
```

## 화면 (탭 3개)
- **검사 탭**: 폴더 경로·대기 파일 수 → [검사 시작] → 요약 카드(전체/OK/NG/임계)
  + 결과 테이블(이상점수 막대·판정 배지·핵심 RCA 한 줄). 행 클릭 → 상세.
  상단에 **자동감시 토글**·**마감 백업** 버튼·자동감시 상태줄.
- **추세 탭**: **일자별 이상점수** 라인차트(일 최대·평균 + 판정임계선 + 일 NG 막대)
  + 상승/하락/안정 배지 + 일자별 표. → 이상점수가 **올라가는 추세**인지 한눈에.
- **상세 분석 탭**: 웨이퍼 선택 → 자연어 RCA + ① Overlay(실제 vs 기대, 이상구간 음영)
  + ② Heatmap(센서×시간 |z|σ) + ③ 센서별 응답분류 표(무응답/약응답/과응답/역방향/…).

## 자동화 (AutoPilot — 백그라운드 스레드)
1. **자동 감시**: `inbox` 에 '완성된'(수정 후 `IO_STABLE_SECS`초 정지) 신규 CSV 가
   들어오면 즉시 추론 → 결과/이력에 반영(화면은 4초 주기로 자동 갱신). 토글로 on/off.
2. **일 마감 백업**: 날짜가 바뀌면 그날 `inbox` 파일을 `data/backup/<날짜>/` 로 이동
   (앱이 자정에 꺼져 있었다면 재시작 시 지난 날짜로 catch-up). [마감 백업] 버튼으로 수동 실행도.
3. **30일 보관**: `IO_RETENTION_DAYS`(기본 30)일 지난 `backup/<날짜>` 폴더 자동 삭제.

이력은 `data/history.jsonl`(append-only), 자동감시 상태는 `data/autopilot.json` 에 영속화.

## API
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/status` | 모델·폴더·대기파일 + `autopilot` 상태 |
| POST | `/api/inspect` | 폴더 전체 추론 (`{"folder": null}` → 기본 inbox) |
| GET | `/api/results` | 마지막 검사 결과 |
| GET | `/api/history?days=30` | **일자별 이상점수 집계(추세)** |
| GET | `/api/wafer/{id}` | 상세: 판정·점수·내러티브·findings |
| GET | `/api/wafer/{id}/plot/{overlay\|heatmap}` | PNG (`reports/{overlay,heatmap}/<id>.png`) |
| POST | `/api/report` | **전체 웨이퍼** overlay+heatmap 일괄 생성(백그라운드) |
| GET | `/api/report` | 리포트 진행률(done/total) |
| GET | `/api/predictions.csv` | **predictions.csv 다운로드** (전체 웨이퍼) |
| GET | `/api/autopilot` | 자동감시/백업/보관 상태 |
| POST | `/api/autopilot/toggle` | 자동감시 on/off (`{"enabled": true}`) |
| POST | `/api/autopilot/backup` | 지금 마감 백업 실행 |

## 환경변수
| 변수 | 기본 | 설명 |
|---|---|---|
| `IO_INBOX` | `data/inbox` | 검사 대상 폴더 |
| `IO_BACKUP` | `data/backup` | 마감 백업 폴더 |
| `IO_RETENTION_DAYS` | `30` | 백업 보관 일수 |
| `IO_WATCH_INTERVAL` | `5` | inbox 폴링 주기(초) |
| `IO_STABLE_SECS` | `3` | 파일 '완성' 판단 정지시간(초) |
| `IO_AUTOWATCH` | `1` | 자동감시 기본 on(0=off) |

## 초안 한계 (다음 단계 후보)
- 판정 임계는 `recon_search_best.json` 의 단일 레시피 값을 사용(다중 레시피 미지원).
- `/api/inspect`(수동) 동기 처리 — 대량 wafer 시 진행률 표시 필요(자동감시는 파일 단위라 무관).
- 인증·다중 사용자 동시성은 미구현(단일 콘솔 가정).
- 분류기 B(`clf_B.joblib`) 불량확률은 미표시(현재 recon A-점수 단독 판정).
