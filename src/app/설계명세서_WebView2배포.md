# IO 이상탐지 검사 시스템 — WebView2 기반 배포 및 폐쇄망 전제조건 설치 설계 명세서

- 문서 대상: IO 이상탐지 검사 앱(FastAPI 대시보드 + onnxruntime 추론, PyInstaller exe)
- 배포 대상 PC: 고객사 제어기 (Windows 10 LTSC, IE 전용, 외부 인터넷 차단)
- 작성 기준일: 2026-07-20

---

## 1. 요구사항 명세서

### 1.1 개요 (역할, 목적)

#### 1.1.1 요구 사항과 목적
- **역할**: 고객사 제어기에서 웨이퍼 이상탐지 대시보드를 **시스템 브라우저(IE) 없이** 네이티브 창(WebView2, 크로미움)으로 표시한다.
- **목적**:
  1. IE(Trident) 렌더링으로 인한 화면 열화(빨간 박스, 색/탭/드롭다운 미동작)를 제거한다.
  2. 고객사 제어기가 **폐쇄망(외부 인터넷 불가)** 이므로, 실행에 필요한 모든 구성요소를 오프라인으로 확보한다.
  3. 앱 배포본을 가볍게 유지하여(런타임 분리) 앱 업데이트 부담을 줄인다.

#### 1.1.2 설계 범위 (블랙 박스로 간주하는 시스템)
설계 대상은 **앱의 창 표시 방식과 배포/설치 절차**이다. 다음은 블랙 박스로 간주한다.
- pywebview 4.4.1 내부 구현(렌더러 선택 로직은 분석하되 수정하지 않음)
- Microsoft WebView2 런타임, Visual C++ 재배포판, .NET Framework 설치기 내부 동작
- 이상탐지 모델/추론 로직, FastAPI 서버 로직(별도 문서)
- 고객사 제어기의 OS 정책(관리자 권한 부여 여부 등)

### 1.2 동작 시나리오

#### 1.2.1 선행조건 (시작 시 만족해야 하는 조건)
- 빌드 PC(우리측)는 인터넷 접속이 가능하다.
- 제어기는 Windows 10 x64, .NET Framework 4.7.2 이상이 설치되어 있다(대부분 LTSC 기본 포함).
- 제어기에서 **관리자 권한 1회 실행**이 허용된다(고객사 확인 완료).

#### 1.2.2 후행조건 (마쳤을 때 만족해야 하는 조건)
- 제어기에서 앱 exe 실행 시 **WebView2(크로미움)** 로 대시보드가 렌더링된다(빨간 박스 미발생).
- 제어기에 인터넷 연결 없이도 필요한 런타임이 모두 설치/등록되어 있다.
- 앱 배포본에는 180MB WebView2 런타임이 **포함되지 않는다**(린 빌드).

#### 1.2.3 기본 동작 시나리오
1. 빌드 PC에서 `build_webview_click.bat` 실행 → `webview2_runtime` 폴더 없음 감지 → **린 빌드** 산출(`dist\IOAnomalyConsole\`).
2. 빌드 PC에서 `Download_Installers.bat` 실행 → `vc_redist.x64.exe`, `MicrosoftEdgeWebView2RuntimeInstallerX64.exe`를 폴더로 자동 다운로드.
3. `IO_Prerequisites_Setup.bat` + 설치본들을 폴더째 제어기로 복사(USB 등).
4. 제어기에서 `IO_Prerequisites_Setup.bat`를 **관리자로 1회 실행** → VC++ + WebView2(+선택 .NET) 오프라인 설치, WebView2가 레지스트리에 등록됨.
5. 제어기에서 앱 exe 실행 → pywebview가 레지스트리에서 WebView2 감지 → 크로미움으로 대시보드 표시.

### 1.3 제약사항
- 제어기는 외부 인터넷이 차단되므로 **온라인 부트스트래퍼(2MB) 사용 불가** → 오프라인 단독 설치본(195MB) 필수.
- WebView2 **고정버전 런타임(폴더 복사, 미설치)** 은 레지스트리에 등록되지 않아, pywebview가 감지하지 못한다(별도 우회 필요). 본 설계는 **Evergreen 정식 설치**로 전환하여 이 제약을 해소한다.
- Gmail 첨부 25MB 한도로 설치본(합 220MB) 직접 전달 불가 → 링크 기반 다운로드 배치로 전달.
- 제어기 관리자 권한이 불가한 환경으로 바뀌면, 고정버전 번들(무거운 빌드) + 레지스트리 우회 패치로 회귀해야 한다.

---

## 2. 본 설계

### 2.1 요구사항 접수

#### 2.1.1 현상 기록
- 고객사 제어기에서 WebView2 빌드 실행 시 **빨간 박스**(엔진 감지기 경고, IE/Trident 렌더링) 발생. 색/탭/검사모델 드롭다운 미동작.
- WebView2 고정버전 런타임 폴더를 앱에 번들해도 동일 증상.
- `WEBVIEW2_BROWSER_EXECUTABLE_FOLDER` 지정, `gui="edgechromium"` 강제에도 IE로 폴백.
- 다른 PC에서는 `pythonnet(.NET 브릿지) 초기화 실패` 오류 발생.

#### 2.1.2 개선 요청사항
1. 제어기에서 WebView2로 정상 렌더링(빨간 박스 제거).
2. 제어기는 exe/설치만 실행하며, 폐쇄망에서 동작.
3. 앱이 너무 무거워지지 않도록 런타임을 분리하고, 필요한 패키지는 별도 설치 프로그램으로 처리.

### 2.2 요구사항 개발

#### 2.2.1 변경 범위 입력
- `app/desktop.py` — WebView2 진입점(레지스트리 감지/런타임 경로 처리)
- `app/io_console_webview.spec` — PyInstaller 빌드(런타임 번들 조건부)
- `app/build_webview_click.bat` — 린/무거운 빌드 자동 분기
- 신규: `app/IO_Prerequisites_Setup.bat`, `app/Download_Installers.bat`, 배포 가이드

#### 2.2.2 제약 사항 정의
- pywebview는 **레지스트리(`EdgeUpdate\Clients\{F3017226-...}` 의 `pv`)** 로만 WebView2를 감지한다(소스 확인).
- pythonnet 3.x는 Windows 기본 `netfx` 로더 사용, **.NET Framework 4.7.2+** 및 **VC++ 런타임(msvcp140.dll)** 필요.
- 제어기 인터넷 불가 → 모든 설치본은 오프라인 단독본으로 사전 확보.

### 2.3 변경점 설계

#### 2.3.1 변경 전
- 앱에 WebView2 **고정버전 런타임(약 180MB) 번들** + 실행 시 레지스트리 조회를 가로채 '설치됨'으로 응답하는 **우회 패치**(`_force_webview2_detection`)로 강제 크로미움 선택.
- 배포본이 무겁고(≈330MB), 우회 패치라는 비표준 방식에 의존.

#### 2.3.2 변경 후
- 앱은 **린 빌드**(런타임 미포함, ≈150MB). WebView2/VC++는 **전제조건 설치기**로 제어기에 정식 설치.
- WebView2 Evergreen 오프라인 단독본을 **정식 설치 → 레지스트리 등록** → pywebview가 자연히 크로미움 선택. 우회 패치 불필요.
- 인터넷 되는 빌드 PC에서 설치본을 받아 폴더째 폐쇄망 제어기로 전달.

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 앱 배포 크기 | ≈330MB | ≈150MB |
| WebView2 확보 | 앱에 고정버전 번들 | 제어기에 Evergreen 정식 설치 |
| 감지 방식 | 레지스트리 우회 패치 | 레지스트리 정식 등록 |
| 제어기 인터넷 | 불필요 | 불필요(오프라인 단독본) |
| 설치 횟수 | 없음 | 제어기당 1회(관리자) |

---

## 3. 디테일 디자인

### 3.1 설명
전제조건 설치기는 앱 실행에 필요한 3요소(VC++, WebView2, 선택적 .NET)를 조용히(`/silent`, `/quiet`) 설치하고 결과를 검증한다. 앱은 실행 시 시스템에 설치된 WebView2를 사용한다. 다운로드 배치는 검증된 Microsoft 영구 링크로 오프라인 단독 설치본을 확보한다.

- VC++: `vc_redist.x64.exe /install /quiet /norestart`
- WebView2: `MicrosoftEdgeWebView2RuntimeInstallerX64.exe /silent /install`
- 검증: 레지스트리 `pv` 키 존재 및 .NET `Release` 값 확인.

### 3.2 설계도

#### 3.2.1 Sequence Diagram — 배포/실행 흐름
```
[빌드 PC(인터넷 O)]                 [운반]              [제어기(인터넷 X)]
   |                                                          |
   |-- build_webview_click.bat ------> dist\IOAnomalyConsole\ |
   |     (webview2_runtime 없음 → 린 빌드)                    |
   |                                                          |
   |-- Download_Installers.bat                                |
   |     └ curl: aka.ms/.../vc_redist.x64.exe                 |
   |     └ curl: fwlink linkid=2124701 (WebView2 x64)         |
   |          → 폴더에 설치본 2개 확보                         |
   |                                                          |
   |==== (앱 폴더 + 전제조건 폴더) USB 복사 ==================>|
   |                                                          |
   |                          IO_Prerequisites_Setup.bat (관리자)
   |                              ├ 관리자 권한 확인          |
   |                              ├ vc_redist /quiet          |
   |                              ├ WebView2 /silent /install → 레지스트리 pv 등록
   |                              └ 검증(pv, .NET Release)     |
   |                                                          |
   |                          IOAnomalyConsole.exe 실행       |
   |                              └ pywebview → _is_chromium() = True(레지스트리)
   |                                   → edgechromium(WebView2) 렌더링 ✔
```

#### 3.2.2 전제조건 설치기 판정 로직 (의사결정)
```
관리자? --No--> [중단: 관리자로 실행 안내]
   |Yes
vc_redist.x64.exe 존재? --No--> [중단: 파일 없음 안내]
   |Yes → 설치
WebView2 설치본 존재? --No--> [중단: 파일 없음 안내]
   |Yes → 설치
ndp48 존재? --Yes--> .NET 4.8 설치
   |No → 건너뜀(대부분 4.7.2+ 보유)
검증: pv 키 존재? --No--> [결과: WebView2 미검출, 실패 안내]
   |Yes → [결과: Prerequisites OK]
```

#### 3.2.3 UI 화면 구성 (콘솔 출력)
```
 IO Anomaly - Prerequisites Setup
 ------------------------------------------------------------
 [1/3] Visual C++ 2015-2022 Redistributable (x64)...  done.
 [2/3] Microsoft Edge WebView2 Runtime...             done.
 [3/3] .NET 4.8 installer not present - skipped.
 ------------------------------------------------------------
 Verifying...
   [OK] WebView2 Runtime registered.
   [OK] .NET Framework v4 present. Release=461814
 ============================================================
 RESULT: Prerequisites OK. You can now run the IO Anomaly app exe.
 ============================================================
```

---

## 4. 테스트 케이스

> 총 12건 중 네거티브 7건(58%).

### TC-01 (정상) 린 빌드 산출
- **TC설명**: `webview2_runtime` 폴더가 없을 때 린 빌드가 생성되는지
- **사전 조건**: 빌드 PC, Python 3.10~3.12, 저장소 옆에 `webview2_runtime` 없음
- **시나리오**: `build_webview_click.bat` 더블클릭
- **필요 데이터**: 소스 트리, requirements-win.txt
- **기대결과**: 콘솔에 "building LEAN app" 출력, `dist\IOAnomalyConsole\`에 `webview2_runtime` 미포함, 크기 ≈150MB

### TC-02 (정상) 설치본 자동 다운로드
- **TC설명**: 다운로드 배치가 두 설치본을 확보하는지
- **사전 조건**: 인터넷 되는 빌드 PC
- **시나리오**: `Download_Installers.bat` 실행
- **필요 데이터**: MS 영구 링크(내장)
- **기대결과**: `vc_redist.x64.exe`(≈25MB), `MicrosoftEdgeWebView2RuntimeInstallerX64.exe`(≈195MB) 생성, 크기 MB 단위

### TC-03 (정상) 전제조건 설치 및 렌더링
- **TC설명**: 제어기에서 설치 후 앱이 WebView2로 뜨는지
- **사전 조건**: 제어기, 관리자 권한, 설치본 2개 동봉, WebView2 미설치 상태
- **시나리오**: `IO_Prerequisites_Setup.bat` 관리자 실행 → 앱 exe 실행
- **필요 데이터**: 전제조건 폴더, 앱 폴더
- **기대결과**: "RESULT: Prerequisites OK", 앱이 크로미움으로 렌더링(빨간 박스 없음)

### TC-04 (정상) WebView2 기설치 환경 멱등성
- **TC설명**: 이미 WebView2가 있는 PC에서 재실행 시 문제 없는지
- **사전 조건**: WebView2 기설치 PC, 관리자
- **시나리오**: `IO_Prerequisites_Setup.bat` 실행
- **필요 데이터**: 전제조건 폴더
- **기대결과**: 설치기 정상 종료(1638 등 '이미 설치' 코드 허용), "Prerequisites OK", 앱 정상

### TC-05 (정상) 앱 업데이트(전제조건 재실행 불필요)
- **TC설명**: 전제조건 설치 후 앱 폴더만 교체해도 동작하는지
- **사전 조건**: TC-03 완료된 제어기
- **시나리오**: 앱 폴더만 새 린 빌드로 교체 후 exe 실행
- **필요 데이터**: 신규 앱 폴더
- **기대결과**: 전제조건 재설치 없이 WebView2로 정상 렌더링

### TC-06 (네거티브) 관리자 권한 없이 실행
- **TC설명**: 일반 권한으로 설치기 실행 시 차단되는지
- **사전 조건**: 제어기, 일반 사용자 권한
- **시나리오**: `IO_Prerequisites_Setup.bat` 더블클릭(관리자 아님)
- **필요 데이터**: 전제조건 폴더
- **기대결과**: "Please run this file as Administrator" 출력 후 종료(설치 미수행)

### TC-07 (네거티브) WebView2 설치본 누락
- **TC설명**: WebView2 exe가 폴더에 없을 때 안내하는지
- **사전 조건**: `MicrosoftEdgeWebView2RuntimeInstallerX64.exe` 삭제 상태, 관리자
- **시나리오**: `IO_Prerequisites_Setup.bat` 실행
- **필요 데이터**: vc_redist만 존재
- **기대결과**: WebView2 파일 없음 안내 + 다운로드 링크 출력, 비정상 종료(exit 1)

### TC-08 (네거티브) VC++ 재배포판 미설치 시 앱 실행
- **TC설명**: VC++ 런타임이 없을 때 pythonnet 오류를 명확히 안내하는지
- **사전 조건**: 전제조건 설치를 건너뛴(또는 VC++ 미설치) 제어기, WebView2만 설치
- **시나리오**: 앱 exe 실행
- **필요 데이터**: 앱 폴더
- **기대결과**: "pythonnet(.NET 브릿지) 초기화 실패" 안내(창 미개방), 임시 대안(브라우저 접속) 안내

### TC-09 (네거티브) .NET 4.7.2 미만 환경
- **TC설명**: .NET이 낮은 PC에서 실패/안내가 적절한지
- **사전 조건**: .NET 4.6.x PC, WebView2/VC++ 설치
- **시나리오**: 앱 exe 실행
- **필요 데이터**: 앱 폴더
- **기대결과**: .NET 브릿지 초기화 실패 안내(.NET 4.7.2+ 필요 문구), 창 미개방

### TC-10 (네거티브) 폐쇄망에서 온라인 부트스트래퍼 사용
- **TC설명**: 부트스트래퍼(2MB)를 폐쇄망 제어기에서 쓰면 실패하는지(사용 금지 근거)
- **사전 조건**: 인터넷 차단 제어기, 부트스트래퍼만 보유
- **시나리오**: 부트스트래퍼 `/silent /install` 실행
- **필요 데이터**: MicrosoftEdgeWebview2Setup.exe(부트스트래퍼)
- **기대결과**: 다운로드 단계에서 설치 실패 → 오프라인 단독본 필요 확인

### TC-11 (네거티브) 다운로드 배치 인터넷 차단
- **TC설명**: 빌드 PC 인터넷이 없을 때 배치가 안전하게 실패하는지
- **사전 조건**: 인터넷 차단된 PC
- **시나리오**: `Download_Installers.bat` 실행
- **필요 데이터**: 없음
- **기대결과**: `--fail`로 다운로드 실패, 파일 미생성 또는 0/수 KB, 크기 안내로 실패 인지 가능

### TC-12 (네거티브) 고정버전 번들 + 우회 패치 제거 상태 회귀 확인
- **TC설명**: 우회 패치 제거 후 고정버전 번들만으로는 IE 폴백함을 확인(설계 근거)
- **사전 조건**: 우회 패치 제거된 앱, 고정버전 런타임 번들, WebView2 미설치·미등록 PC
- **시나리오**: 앱 exe 실행
- **필요 데이터**: 무거운 빌드(고정버전 포함)
- **기대결과**: 레지스트리 미등록으로 IE 폴백(빨간 박스) → Evergreen 정식 설치가 필요함을 입증

---

## 부록 A. 검증된 다운로드 링크
- VC++ 2015-2022 x64: `https://aka.ms/vs/17/release/vc_redist.x64.exe`
- WebView2 x64 Evergreen 오프라인 단독본: `https://go.microsoft.com/fwlink/?linkid=2124701`
- (선택) .NET Framework 4.8: `https://dotnet.microsoft.com/download/dotnet-framework/net48`

## 부록 B. 관련 파일
- `app/desktop.py`, `app/io_console_webview.spec`, `app/build_webview_click.bat`
- `app/IO_Prerequisites_Setup.bat`, `app/Download_Installers.bat`, `app/배포_가이드_린빌드.txt`
