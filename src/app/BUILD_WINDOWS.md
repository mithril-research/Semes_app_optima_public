# Windows 실행 파일(.exe) 빌드 — 웹 대시보드 런처

이 폴더(`src/`)를 **Windows PC 로 복사**한 뒤 빌드하면
`IOAnomalyConsole.exe` 가 나온다. exe 를 실행하면 내부 서버가 뜨고
**기본 브라우저로 대시보드가 자동으로 열린다**(별도 브라우저 설정 불필요).
콘솔 창을 닫으면 종료.

> PyInstaller 는 Linux→Windows 크로스컴파일을 못 하므로 **빌드는 Windows 에서** 한다.
> (앱 코드·스크립트는 준비 완료 — Windows 에서 빌드만 1회 실행)

---

## 0. 준비물 (Windows PC)
- Windows 10 (64-bit) / 11
- Python 3.10 ~ 3.12 (64-bit) — https://python.org (설치 시 *Add to PATH* 체크)
- 이 `src/` 폴더 **전체** (특히 아래 파일 포함 필수)
  - `artifacts/tcn_forecast.pt`         ← 학습된 모델 가중치
  - `artifacts/recon_search_best.json` ← 하이퍼파라미터 + 판정 임계
  - `assets/NanumGothic.ttf`       ← 그래프 한글 폰트

---

## 1. 빌드 (원클릭)
`src\app\build_win.bat` 더블클릭.
자동으로: 가상환경 생성 → 패키지 설치 → PyInstaller 빌드.

> **CPU / GPU 선택**: bat 안의
> `pip install torch --index-url https://download.pytorch.org/whl/cu124`
> - GPU(권장 아님, 제어기 PC 에 NVIDIA 없으면 무의미) → `cu124` 유지
> - CPU(호환·소형, 권장) → `cu124` 를 `cpu` 로 수정

## 1-b. 빌드 (수동)
```bat
cd <src 를 복사한 폴더>      REM  src 의 '부모' 폴더
python -m venv venv_build && venv_build\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r src\app\requirements-win.txt
pyinstaller --clean src\app\io_console.spec
```

### 결과물
```
dist\IOAnomalyConsole\IOAnomalyConsole.exe   ← 실행 파일
dist\IOAnomalyConsole\                        ← 이 폴더를 통째로 배포/복사
```

---

## 1-c. 모델 교체 — 재빌드·재설치 없이, 폴더에 넣고 [모델 교체] 클릭
exe 옆에 **`model\` 폴더**(또는 `artifacts\`)를 만들고 웨이트 2개를 넣으면 그걸 사용한다.
```
IOAnomalyConsole.exe
model\
   tcn_forecast.pt                 <- 재학습 모델 웨이트
   recon_search_best.json     <- 임계(레시피별)·파라미터
```
- 프로그램 **실행 중에도** 대시보드 상단 **[🔄 모델 교체]** 버튼을 누르면
  폴더의 웨이트를 다시 읽어 **즉시 교체**된다(프로그램 재시작·재설치 불필요).
- 폴더 탐색 우선순위: `IO_ARTIFACTS`(환경변수) → `model\` → `artifacts\` → 내장(기본).
- 헤더에 현재 모델이 `내장`인지 `외장폴더`인지 표시된다.

> 설비/레시피가 다른 모델을 여러 개 두고, 폴더의 웨이트만 바꿔가며 교체하면 된다.
> (한 번 설치하면 이후엔 파일 교체 + [모델 교체] 클릭이 끝)

## 2. 실행 · 사용 (배포 PC)
1. `IOAnomalyConsole.exe` 실행 → 콘솔 창 + 몇 초 뒤 브라우저에 대시보드.
2. wafer **CSV 를 exe 옆 `data\inbox\`** 에 넣기(첫 실행 시 자동 생성).
   폴더를 바꾸려면 환경변수 `IO_INBOX` 설정.
3. **[검사 시작]** → 결과 테이블 → 행 클릭 → **상세 분석**(overlay + RCA).
4. 종료: 콘솔 창 닫기(Ctrl+C).

overlay/heatmap PNG 는 exe 옆 `reports\ux\` 에 캐시된다.

---

## 3. 문제 해결
| 증상 | 해결 |
|---|---|
| 브라우저 안 열림 | 콘솔에 뜬 주소(예 http://127.0.0.1:8000) 직접 접속 |
| 포트 충돌 | 8000 사용 중이면 빈 포트 자동 선택 — 콘솔 주소 확인 |
| `torch` DLL 크래시 | 빌드 PC 에 맞는 torch 재설치 후 재빌드 |
| exe 가 2GB+ | CUDA torch 포함 시 정상 — CPU 빌드로 축소 |
| 안티바이러스 오탐 | PyInstaller onedir 특성 — 코드사이닝 권장 |
| 모델 없음 오류 | `artifacts\tcn_forecast.pt` 가 복사됐는지 확인 |
| 디버그 | spec 은 `console=True` — 콘솔 로그로 원인 확인 |

---

## 4. 빌드에 쓰이는 파일
```
src/
├── *.py                     # 추론 패키지 전체(모두 필요)
├── artifacts/tcn_forecast.pt              # 모델(필수)
├── artifacts/recon_search_best.json  # 임계/파라미터(필수)
├── assets/NanumGothic.ttf            # 폰트(필수)
└── app/
    ├── launcher.py          # ★exe 진입점: 서버 + 브라우저 자동 오픈
    ├── server.py            # FastAPI 백엔드(추론 API)
    ├── static/index.html    # 대시보드 UI
    ├── io_console.spec      # PyInstaller 빌드 정의
    ├── build_win.bat        # 원클릭 빌드
    ├── requirements-win.txt # 빌드 패키지
    └── BUILD_WINDOWS.md      # (이 문서)
```
