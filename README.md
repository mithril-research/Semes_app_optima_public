> 이 저장소는 센서명이 익명화된 배포본입니다. 온도·압력·히터·보조 센서는 `t_`, `p_`, `h_`, `d_`, `s_` 식별자를 사용합니다. 디지털 IO는 별도의 익명 숫자 ID를 사용합니다. 실제 센서명과의 대응표는 포함하지 않습니다.
>
> 입력 데이터의 컬럼과 모델 메타데이터도 동일한 익명 스키마여야 합니다. 기존 원본 데이터·모델을 그대로 사용하지 말고, 별도로 보관한 대응 관계를 적용하거나 익명 데이터로 다시 학습하세요. `_PV`와 `_SV`는 측정값/설정값 구분을 위해 유지합니다.
>
> 화면 캡처와 PDF·Word·Excel 원본은 제외했습니다. 추출 가능한 문서 본문은 익명화한 `.txt` 파일로 제공합니다. 서식과 이미지 내용은 보존하지 않습니다.

# 챔버 이상 검사 시스템 (ChamberInspect)

웨이퍼 센서 로그를 읽어 복원오차 기반으로 이상 웨이퍼를 판정하는 제어기용 데스크톱 앱입니다.
FastAPI + onnxruntime + WebView2 로 동작하며, PyInstaller 로 단일 폴더 exe 를 만듭니다.

## 빠른 시작 — 내려받고 바로 빌드

이 저장소를 받으면 최상위에 **`앱빌드.bat`** 이 있습니다. 더블클릭하면 끝입니다.

```
git clone https://github.com/mithril-research/Semes_app.git
cd Semes_app
앱빌드.bat          ← 더블클릭
```

빌드가 끝나면 `ChamberInspect\` 폴더가 생깁니다. 그 폴더를 제어기로 옮기고
`ChamberInspect.exe` 를 실행하면 됩니다.

- 빌드 PC 요구사항: **Python 3.10 ~ 3.12 (64-bit)** + 인터넷(패키지 설치)
- 빌드는 `C:\iobuild` 에서 수행됩니다(한글·긴 경로에서 PyInstaller 가 실패하기 때문).
- 제어기에는 **WebView2 런타임**이 필요합니다 → `src\app\IO_Prerequisites_Setup.bat`

## 배치 파일

| 파일 | 용도 |
| --- | --- |
| `앱빌드.bat` | **기본(Win10).** 배포용 앱 빌드 → `ChamberInspect\` |
| `앱빌드_WIN7.bat` | **Windows 7 SP1 / WES7 전용** 빌드 → `ChamberInspect_WIN7\` |
| `진단_실행_WIN7.bat` | 윈도우7용 진단(콘솔 CP949) |
| `앱빌드_콘솔진단.bat` | 콘솔창이 뜨는 진단용 빌드 → `ChamberInspect_Console\` |
| `진단_실행.bat` | 앱이 실행돼도 반응이 없을 때 원인 확인(차단 해제·종료코드·로그) |
| `빠른빌드_증분.bat` | venv 를 재사용하는 증분 빌드 |
| `변환기빌드.bat` / `모델변환.bat` | 학습 결과(.pt) → ONNX 변환기 |
| `개발실행_빌드없이.bat` | 빌드 없이 소스로 바로 실행(개발용) |

## Windows 7 SP1 / WES7

**Win10 빌드와 완전히 별개로 관리합니다.** 파일도 산출물 폴더도 겹치지 않습니다.

| | Win10 | WES7 / Win7 SP1 |
| --- | --- | --- |
| 빌드 | `앱빌드.bat` | `앱빌드_WIN7.bat` |
| 파이썬 | 3.10 ~ 3.12 | **3.8 만** (3.9+ 는 Win8.1 이상 요구) |
| 패키지 | `requirements-win.txt` | `requirements-win7.txt` (cp38 핀) |
| 스펙 | `io_console_webview.spec` | `io_console_win7.spec` |
| 화면 | WebView2 네이티브 창 | 네이티브 창(WebView2 **109 고정버전**) 또는 브라우저 모드 |
| 추론 | onnxruntime | **torch 2.3.1** (기본) 또는 onnxruntime |
| 산출물 | `ChamberInspect\` | `ChamberInspect_WIN7\` |

```
앱빌드_WIN7.bat      ← 더블클릭 (빌드 PC 에 Python 3.8 64-bit 필요)
검증 대상: Windows 7 Professional K 6.1.7601 SP1 (빌드 7601) 64bit — WES7 동일
```

빌드 옵션(환경변수로 조합):

```
set IO_WES7_UI=webview        네이티브 창 (기본)
set IO_WES7_UI=browser        브라우저 모드
set IO_WES7_BACKEND=torch     torch 추론 (.pt 모델) — 기본
set IO_WES7_BACKEND=onnx      onnxruntime 추론 (.onnx 모델, 훨씬 가벼움)
```

### 네이티브 창을 Win7 에서 쓰는 방법

가능합니다. 버전이 정확히 맞아야 합니다.

- pywebview 4.4.1 이 번들한 WebView2 SDK 가 **1.0.1518.46** 이고, 이 `1518` 빌드가
  곧 **Edge 109** — Windows 7 에서 도는 마지막 버전입니다. 그래서 pywebview 를 올리면 안 됩니다.
- 제어기에 **WebView2 고정버전 런타임 109.0.1518.x (x64)** 를 폴더로 넣어야 합니다.
  Evergreen 설치기는 Win7 에 설치되지 않습니다(2023-01 지원 종료).
  `webview2_runtime\` 폴더를 저장소 루트에 두면 빌드가 통째로 번들합니다.
- pywebview 는 레지스트리(`EdgeUpdate\Clients\{F3017226-...}\pv`)로만 런타임을 감지하는데
  고정버전은 그 키를 남기지 않습니다 → `desktop.py` 의 `_force_chromium_if_bundled()` 가
  우회합니다(번들 런타임을 실제로 찾았고 정식 설치가 없을 때만 동작).
- **.NET Framework 4.6.2 이상** 필요(Win7 SP1 은 4.8 까지 설치 가능).

### 빌드 후 자동 검사

`앱빌드_WES7.bat` 은 빌드가 끝나면 `check_win7_compat.py` 로 산출물을 검사합니다.
결과 폴더의 모든 `.exe/.dll/.pyd` 의 **PE 임포트 테이블**을 읽어, Windows 8 이상에서만
존재하는 API 를 정적 임포트하는 파일이 있으면 빌드를 실패로 처리합니다.

Windows 는 DLL 을 열 때 임포트 테이블의 함수를 로드 시점에 전부 해석하므로, 하나라도
없으면 프로세스가 시작조차 못 합니다. 이 검사가 그것을 미리 잡습니다. 따로 돌릴 수도 있습니다:

```
python src\app\check_win7_compat.py ChamberInspect_WES7
```

문자열 검색으로는 판별할 수 없습니다 — `GetProcAddress` 로 동적 조회하는 코드는 문자열이
있어도 Win7 에서 잘 돕니다(torch 의 `VirtualAlloc2`, `SetThreadDescription` 이 그런 경우).

### 왜 onnxruntime 이 아니라 torch 인가

PE 임포트 테이블 실측 결과입니다.

| onnxruntime | Win8+ API 정적 임포트 | Win7 |
| --- | --- | --- |
| 1.19.2 / 1.18.1 / 1.17.3 / 1.16.3 / 1.15.1 | `kernel32!CreateFile2` | ✗ |
| 1.14.1 / 1.12.1 | 없음 | ✓ |

1.15.1 부터 막힙니다. 1.14.1 로 내리면 로드는 되지만, 학습앱이 만드는 `.onnx` 가
IR=10 이라 1.14 가 읽지 못합니다(export 시 onnx 버전을 낮춰 다시 뽑아야 함).
그래서 **torch 를 기본**으로 했습니다. torch 2.3.1 은 휠 안의 DLL/PYD 12개를 전수 검사해
Win8+ 정적 임포트가 하나도 없음을 확인했습니다.

### torch 백엔드를 쓸 때 (★버전 상한 주의)

`torch 2.4.1` 의 `torch_cpu.dll` 은 `kernel32!CreateFile2`(Windows 8+)를 **정적 임포트**해서
Windows 7 에서는 `import torch` 자체가 실패합니다. PE 임포트 테이블 실측 결과:

| torch | Win8+ API 정적 임포트 | Win7 |
| --- | --- | --- |
| 2.4.1 | `kernel32!CreateFile2` | ✗ |
| **2.3.1** | 없음 | ✓ (상한으로 채택) |
| 2.2.2 / 2.1.2 / 2.0.1 / 1.13.1 | 없음 | ✓ |

**모델 폴더에는 `.onnx` 가 아니라 `tcn_forecast.pt` 를 넣으세요.** torch 빌드에는
onnxruntime 이 들어있지 않아서, 폴더에 `.onnx` 가 섞여 있으면 앱이 그쪽을 고르려다 죽습니다.
이를 막기 위해 torch 빌드에는 런타임 훅(`rthook_backend_torch.py`)이 포함되어 앱이 항상
`IO_BACKEND=torch` 로 뜨고 ONNX 경로를 아예 보지 않습니다.

용량 주의: `torch_cpu.dll` 하나가 200MB 이상이라 배포 폴더가 1GB 를 넘을 수 있습니다.

CPU 사용 제한 설정은 torch 에도 적용됩니다. torch 는 환경변수를 무시하고 물리코어 수만큼
스레드를 잡기 때문에, 앱이 모델 로드 시 `torch.set_num_threads()` 로 직접 지정합니다.

빌드 PC 는 Win10 이어도 됩니다. **Python 3.8 로 빌드하기만 하면** 산출물이 WES7 에서 돕니다
(오히려 권장 — Win7 빌드 PC 는 TLS 1.2 문제로 pip 가 PyPI 에 못 붙는 경우가 있습니다).

> 기술 근거: [docs/Win7_호환성_검증결과.md](docs/Win7_호환성_검증결과.md)
> — 버전을 그렇게 고정한 이유(PE 임포트 테이블 실측표).
>
> 성능 자료: [docs/CPU_사용량_기능별_정리.md](docs/CPU_사용량_기능별_정리.md)
> — 기능별 CPU 실측치와 덜어낼 후보 우선순위.
>
> 고객사 공유용 문서: [docs/WES7_배포_사전점검.md](docs/WES7_배포_사전점검.md)
> — 준비 사항·예상 문제·검증 절차를 현장 담당자가 읽을 수 있게 정리했습니다.

### WES7 제어기에서 확인할 것

1. **화면** — 네이티브 창으로 빌드했다면 WebView2 고정버전 런타임 109 와 .NET 4.6.2+.
   브라우저 모드라면 Chrome 109 또는 Firefox 115 ESR (앱이 설치된 브라우저를 찾아 직접
   실행하며, 없으면 콘솔에 경고를 띄웁니다). Internet Explorer 로는 화면이 비어 보입니다.
2. **쓰기 필터(EWF/FBWF)** — 켜져 있으면 검사 이력이 재부팅 때 사라집니다.
   설정 탭 > 저장소 위치 에서 필터가 걸리지 않은 드라이브를 지정하세요.
3. **Visual C++ 2015-2019 재배포 패키지(x64)** 설치 필요.
4. GPU 가속은 쓰지 않습니다(CPU 추론). DirectML 은 DirectX 12 가 필요해 대상이 아닙니다.

### 제약

- 이 구성의 패키지들은 보안 업데이트가 끝난 버전입니다. 폐쇄망 전용으로만 쓰세요.
- `onnxruntime` 공식 지원 OS 는 Windows 10+ 입니다. 1.19.2 가 WES7 에서 실제로
  로드되는지는 **실기 확인이 필요합니다.** 제어기에서 아래 한 줄로 5초면 판별됩니다.
  ```
  ChamberInspect.exe   →  콘솔에 "[onnx] 실행 프로바이더: CPUExecutionProvider" 가 뜨면 정상
  ```

## 우분투 학습앱

같은 소스로 리눅스에서는 **학습앱**이 됩니다(빌드 없이 소스 실행, 브라우저 접속).
학습 탭에서 폴더·필터·GPU 를 고르고 여러 작업을 동시에 돌릴 수 있습니다.

```
git clone https://github.com/mithril-research/Semes_app.git
cd Semes_app
./우분투_학습앱.sh setup      # 최초 1회 — venv + 패키지 + CUDA torch
./우분투_학습앱.sh            # 실행 → http://127.0.0.1:8000
PORT=8010 ./우분투_학습앱.sh  # 포트 변경
```

- `setup` 이 끝나면 **추론 CUDA 사용 가능 여부와 학습(torch) CUDA 여부**를 바로 출력합니다.
  CPU 판(`onnxruntime`)이 GPU 판과 함께 깔려 있으면 자동으로 제거합니다
  (같이 두면 CPU 판이 이겨서 "학습만 CUDA, 추론은 CPU" 가 됩니다).
- 상시 구동은 `src/app/chamberinspect-train.service` 를 참고하세요
  (`KillMode=process` — 서비스를 재시작해도 진행 중인 학습이 죽지 않습니다).
- 학습 결과는 출력 폴더에 `onnx + npz + search_best.json + tcn_forecast.pt` 로 나오며
  그대로 챔버앱의 모델 폴더로 쓸 수 있습니다.
- 데이터·학습 작업 폴더를 앱 밖에 두려면 `storage.json` 또는 설정 탭 > 저장소 위치를 쓰세요.
- 학습 CLI 를 직접 쓰려면: `python3 -m src.train_jw.run tcn-search --h-cart-mode exclude …`
- 제어기에 WebView2 등 전제 프로그램을 깔 때는 `src/app/IO_Prerequisites_Setup.bat` 을 씁니다.

## 폴더 구조

```
앱빌드.bat            윈도우10 빌드 진입점(최상위)
앱빌드_WIN7.bat       윈도우7/WES7 빌드 진입점(최상위)
우분투_학습앱.sh       우분투 학습앱 진입점(최상위)
src/                  앱 소스 (파이썬 패키지)
  app/
    server.py         검사·개요·산점도·RCA·저장소·학습 API
    static/index.html 대시보드 UI
    desktop.py        WebView2 창 진입점 + 시작 실패 진단
    io_console_webview.spec   PyInstaller 스펙
    requirements-win.txt      빌드 패키지 (CPU 추론, 기본)
    requirements-win-gpu.txt  DirectML GPU 추론 (확인된 PC 에서만)
    requirements-win7.txt     윈도우7 패키지 (cp38 핀, onnx 백엔드)
    requirements-win7-torch.txt  윈도우7 torch 백엔드 (torch 2.3.1 상한)
    io_console_win7.spec      윈도우7 빌드 스펙
    launcher.py               브라우저 모드 진입점
    run_ubuntu.sh             우분투 학습앱 실행(본체)
    requirements-ubuntu.txt   리눅스 패키지 (onnxruntime-gpu 등)
    chamberinspect-train.service  상시 구동용 systemd 유닛 예시
  recon_explain.py    복원오차·리포트
  onnx_backend_forecast.py    ONNX 추론 백엔드
  train_jw/           학습 코드(서브패키지) — 우분투 학습앱이 이걸 실행합니다
```

## 문제가 생기면

**앱을 실행해도 아무 반응이 없을 때** — exe 옆에 아래 파일이 남습니다. 이걸 보면 원인이 나옵니다.

- `앱_시작기록.txt` — 어디까지 진행했는지
- `앱_시작오류.txt` — 파이썬 예외와 진행 단계
- `앱_크래시.txt` — 네이티브 크래시(DLL 로드 실패 등)

그래도 모르겠으면 `진단_실행.bat` 을 exe 와 같은 폴더에서 실행하세요.
차단 표시 해제, 중복 실행, 종료코드 해석, 최근 로그를 한 번에 보여줍니다.

**다른 PC 에서 쓰던 설정을 가져왔을 때** — 설정 탭 > `폴더 경로 점검` 에서
이 PC 에 없는 경로를 확인하고 한 번에 정리할 수 있습니다.
앱 폴더 안의 경로는 상대경로로 저장되므로 폴더째 옮겨도 그대로 동작합니다.

**GPU 가속** — 기본은 CPU 추론입니다. GPU 는 설정 탭에서 켜며, 안 걸리는 경우
같은 화면에 이유(패키지 충돌·라이브러리 누락 등)와 해결 명령이 표시됩니다.
