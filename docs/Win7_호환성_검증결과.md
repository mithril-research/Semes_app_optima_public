# Windows 7 / WES7 호환성 검증 결과 (기술 근거)

대상: Windows Embedded Standard 7 (버전 6.1 SP1)
검증일: 2026-08-20
목적: WES7 배포본의 구성 요소를 왜 그 버전으로 고정했는지에 대한 근거 기록

---

## 검증 방법

**PE 임포트 테이블 정적 분석**을 사용했습니다.

Windows는 DLL을 열 때 임포트 테이블에 적힌 함수를 **로드 시점에 전부 해석**합니다.
그 중 하나라도 해당 OS의 DLL에 없으면 프로세스가 시작조차 못 하고 종료됩니다.
창 없는 exe라면 화면에 아무 메시지도 남지 않습니다.

> **문자열 검색으로는 판별할 수 없습니다.**
> `GetProcAddress`로 함수를 실행 중에 찾는 코드는 문자열이 파일에 들어 있어도
> Windows 7에서 정상 동작합니다. 실제로 torch에는 `VirtualAlloc2`,
> `SetThreadDescription`(Windows 10 전용) 문자열이 있지만 동적 조회라 무해합니다.
> 반드시 임포트 테이블을 봐야 구분됩니다.

검사 도구는 저장소에 포함되어 있습니다: `src/app/check_win7_compat.py`
빌드 배치가 빌드 직후 자동 실행하며, 문제가 발견되면 빌드를 실패로 처리합니다.

```
python src\app\check_win7_compat.py ChamberInspect_WES7
```

---

## 1. 파이썬 — 3.8 고정

CPython 공식 문서(`Doc/using/windows.rst`, 3.9 브랜치) 원문:

> "Python 3.9 supports Windows 8.1 and newer.
> If you require Windows 7 support, please install Python 3.8."

3.9 이상은 설치 자체가 되지 않으며, 그 버전으로 빌드한 exe도 Windows 7에서 실행되지 않습니다.
**→ Python 3.8 (64-bit) 로 고정.** 빌드 배치가 3.8이 아니면 진행을 막습니다.

빌드 PC의 OS는 무관합니다. Windows 11에서 Python 3.8로 빌드해도 산출물은 WES7에서 동작합니다.
(오히려 권장 — Windows 7 빌드 PC는 TLS 1.2 문제로 pip가 PyPI에 접속하지 못하는 경우가 있습니다.)

---

## 2. 추론 엔진 — torch 2.3.1 채택

### onnxruntime 검사 결과

| 버전 | Windows 8+ API 정적 임포트 | Windows 7 |
| --- | --- | --- |
| 1.19.2 | `kernel32!CreateFile2` | ✗ |
| 1.18.1 | `kernel32!CreateFile2` | ✗ |
| 1.17.3 | `kernel32!CreateFile2` | ✗ |
| 1.16.3 | `kernel32!CreateFile2` | ✗ |
| 1.15.1 | `kernel32!CreateFile2` | ✗ |
| 1.14.1 | 없음 | ✓ |
| 1.12.1 | 없음 | ✓ |

해당 파일: `onnxruntime.dll`, `onnxruntime_pybind11_state.pyd`
`CreateFile2`는 Windows 8부터 존재하는 API입니다.

**1.15.1부터 사용 불가**입니다. 1.14.1로 내리면 로드는 되지만, 학습앱이 만드는 모델
파일이 **IR 버전 10**(ONNX 1.16 계열)이라 1.14가 읽지 못합니다
(`Unsupported model IR version`). 쓰려면 모델을 낮은 IR로 다시 내보내야 합니다.

### torch 검사 결과

| 버전 | Windows 8+ API 정적 임포트 | Windows 7 |
| --- | --- | --- |
| 2.4.1 (cp38 마지막 버전) | `kernel32!CreateFile2` (`torch_cpu.dll`) | ✗ |
| **2.3.1** | 없음 (DLL/PYD 12개 전수) | **✓ 채택** |
| 2.2.2 | 없음 | ✓ |
| 2.1.2 | 없음 | ✓ |
| 2.0.1 | 없음 | ✓ |
| 1.13.1 | 없음 | ✓ |

> PyTorch 공식 문서는 아직 "Windows 7 and greater"라고 표기하고 있으나,
> 실제 바이너리(2.4.1)와는 맞지 않습니다.

**결론**: 모델 파일을 다시 내보내는 작업 없이 쓸 수 있는 것은 torch 쪽이므로
**torch 2.3.1을 기본 백엔드로 채택**했습니다. 모델 파일은 `.pt` 형식을 사용합니다.

**대가**: `torch_cpu.dll` 하나가 200MB 이상이라 배포 폴더가 1GB에 근접합니다.

---

## 3. 나머지 의존성 — 전부 통과

cp38-win_amd64 휠의 바이너리를 전수 검사했습니다.

| 패키지 | 버전 | 바이너리 수 | 결과 |
| --- | --- | --- | --- |
| numpy | 1.24.4 | 20 | ✓ |
| pandas | 2.0.3 | 46 | ✓ |
| matplotlib | 3.7.5 | 10 | ✓ |
| pydantic | 1.10.26 | 25 | ✓ |
| cffi | 1.17.1 | 1 | ✓ |
| pythonnet | 3.0.3 | 1 | ✓ |
| pywebview | 4.4.1 | 7 | ✓ |
| pyinstaller | 5.13.2 | 4 | ✓ |
| contourpy / kiwisolver / pillow | — | 9 | ✓ |
| **torch** | **2.3.1** | **12** | **✓** |

전부 Windows 8 이상 전용 API의 정적 임포트가 없습니다.

### pydantic을 1.x로 고정한 이유

pydantic 2.x의 `pydantic-core`는 Rust로 빌드되는데, **Rust는 1.78(2024년 5월)부터
최소 지원 대상을 Windows 10으로 올렸습니다.** cp38 휠이 존재하더라도 Windows 7에서
로드가 실패할 위험이 있습니다. pydantic 1.10은 Cython 빌드라 이 문제와 무관합니다.

앱 코드는 `BaseModel`과 `.dict()`만 사용하므로(v1 API) 수정 없이 동작합니다.

### PyInstaller는 문제가 아닙니다

PyInstaller 6.x가 Windows 7을 버렸다는 통설이 있으나, 변경 이력을 확인한 결과
부트로더는 `_WIN32_WINNT=0x0601`(Windows 7)로 빌드되고 있습니다.
다만 다른 핀과의 세대를 맞추기 위해 5.13.2를 사용합니다.

---

## 4. 화면 표시 — WebView2 109

Microsoft 공식 문서(Microsoft Edge supported operating systems):

> "Microsoft Edge support for Windows 7, Windows 8, Windows 8.1 ... ended in Edge version 109." (2023-01)

WebView2는 Edge 런타임이므로, 현재 배포되는 Evergreen 설치기는 WES7에 설치되지 않습니다.

**그런데 버전이 정확히 맞물립니다.** pywebview 4.4.1 휠에 포함된
`Microsoft.Web.WebView2.Core.dll`의 버전을 확인한 결과 **1.0.1518.46** 이며,
이 `1518` 빌드가 곧 **Edge 109** 입니다. Windows 7에서 도는 마지막 버전과 일치합니다.

**→ pywebview는 4.4.1로 고정해야 합니다.** 올리면 Windows 7에 없는 런타임을 요구합니다.

### 추가로 필요했던 처리

pywebview의 `winforms._is_chromium()`은 레지스트리
`SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}\pv`
값 하나로만 WebView2 설치 여부를 판단합니다(소스 확인).

고정버전 런타임은 폴더 복사 방식이라 이 레지스트리 키를 남기지 않으므로,
그대로 두면 무조건 Internet Explorer로 폴백해 화면이 비어 보입니다.
Windows 7은 Evergreen 설치기가 설치되지 않아 정식 등록 자체가 불가능합니다.

**→ `desktop.py`의 `_force_chromium_if_bundled()`가 우회합니다.**
번들 런타임을 실제로 찾았고 정식 설치가 없을 때만 동작하며,
Windows 10의 정식 설치 경로는 건드리지 않습니다.

필요 조건: **WebView2 고정버전 런타임 109.0.1518.x (x64)** + **.NET Framework 4.6.2 이상**

---

## 5. 앱 소스 코드 — 3.8 호환 확인

- 전체 27개 파일을 `ast.parse(feature_version=(3,8))`로 검사 → **전부 통과**
- Python 3.9 이상 전용 표준 기능(`removeprefix`, `zoneinfo`, `dict | dict` 등) **사용 0건**
- Windows API 사용처는 `MessageBoxW`, `SetProcessWorkingSetSize`, `taskkill` 뿐 — 전부 Windows 7에 존재
- 대시보드가 옵셔널 체이닝, `??`, `structuredClone`, `:has()` 등 **최신 문법 미사용**
  → Chrome 109 / Firefox 115 ESR에서 동작 (Internet Explorer는 불가)
- CPU 코어 인식에 쓰는 `GetLogicalProcessorInformation`은 Windows Vista부터 존재

---

## 6. 검증되지 않은 항목

| 항목 | 상태 |
| --- | --- |
| 라이브러리 로딩 | **검증 완료** — 실행 파일 130여 개 정적 전수 검사 |
| 실행 중 동작 | **미검증** — 로딩 이후는 실기 확인 필요 |
| WebView2 109 창 표시 | **미검증** — 런타임 배치 후 확인 필요 |
| 제어기 성능에서의 처리 속도 | **미검증** |

정적 검증은 "프로그램이 시작될 수 있는가"까지를 보장하며,
그 이후의 동작까지 보장하지는 않습니다.

**최종 확인 방법**: 제어기 또는 Windows 7 가상머신에서 앱을 실행하고
콘솔에 `[engine] 모델 로드(torch)` 가 표시되는지 확인합니다.
실패 시 앱 폴더에 `앱_시작기록.txt` / `앱_시작오류.txt` / `앱_크래시.txt` 가 남습니다.
