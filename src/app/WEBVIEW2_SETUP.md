# WebView2 내장 빌드 — IE 전용 / Windows 10 LTSC PC 대응

시스템 브라우저(IE)를 쓰지 않고, **WebView2 크로미움 엔진을 프로그램에 함께 담아**
자체 창으로 대시보드를 띄웁니다. 고객 PC에 **인터넷·관리자권한·브라우저 설치가 필요 없습니다.**

---

## A. 빌드 PC 에서 (Windows, 인터넷 됨)

### 1) WebView2 "고정 버전(Fixed Version)" 런타임 내려받기
- https://developer.microsoft.com/microsoft-edge/webview2/ → **Fixed Version** 섹션
- **x64** 최신 버전 `.cab` 다운로드 (예: `Microsoft.WebView2.FixedVersionRuntime.<버전>.x64.cab`)

### 2) cab 압축 풀기 → 폴더로
명령창에서(예: 다운로드 폴더):
```
mkdir wv2
expand "Microsoft.WebView2.FixedVersionRuntime.<버전>.x64.cab" -F:* wv2
```
- 풀면 `wv2\Microsoft.WebView2.FixedVersionRuntime.<버전>.x64\` 안에
  **`msedgewebview2.exe`** 와 DLL들이 있습니다. 그 폴더가 런타임 본체입니다.

### 3) 앱 빌드
```
src\app\build_win_webview.bat
```
→ `dist\IOAnomalyConsole\IOAnomalyConsole.exe` (폴더형/onedir)

### 4) 런타임을 앱 폴더에 복사  ★필수
2)에서 만든 `msedgewebview2.exe` 가 들어있는 폴더를 통째로 아래로 복사하고
폴더 이름을 **`webview2_runtime`** 으로:
```
dist\IOAnomalyConsole\
├── IOAnomalyConsole.exe
├── _internal\                 (파이썬/torch 등)
└── webview2_runtime\          ← 여기! (안에 msedgewebview2.exe 있어야 함)
      ├── msedgewebview2.exe
      └── ... (DLL 들)
```
> 버전 하위폴더가 한 겹 더 있어도 됩니다(자동 탐색). 요점은 `webview2_runtime` 아래
> 어딘가에 `msedgewebview2.exe` 가 있으면 됩니다.

---

## B. 고객 PC 로 전달
`dist\IOAnomalyConsole\` **폴더 전체**(webview2_runtime 포함)를 압축해 전달합니다.
- 용량이 커서(파이썬+torch+WebView2 약 1GB+) 다운로드 링크(WeTransfer/드라이브/사내서버) 권장.
- 고객은 압축 풀고 `IOAnomalyConsole.exe` 실행 → **IE 없이 자체 창**으로 대시보드가 뜹니다.

---

## C. 확인 / 문제 해결
| 증상 | 확인 |
|---|---|
| 창이 안 열리고 콘솔에 "번들 런타임 못 찾음" | `webview2_runtime\msedgewebview2.exe` 경로 확인 |
| "WebView2 런타임 확인" 오류 | 4)의 런타임 폴더가 exe 옆에 있는지, x64 인지 |
| 창은 뜨는데 흰 화면 | 잠시 대기(엔진 로딩) 후 새로고침. 콘솔 로그 확인 |
| 콘솔 창이 거슬림 | `io_console_webview.spec` 의 `console=True` → `False` 로 바꿔 재빌드 |

## D. 원리 (참고)
- 앱이 시작할 때 `webview2_runtime` 를 찾아 환경변수
  `WEBVIEW2_BROWSER_EXECUTABLE_FOLDER` 로 지정 → WebView2 가 **그 고정버전 엔진**을 사용.
- 따라서 시스템에 WebView2/Edge 가 없어도(=IE만 있어도) 동작합니다.
- 창 렌더링은 크로미움 → 지금 대시보드(최신 웹기술)가 그대로 정상 표시됩니다.
