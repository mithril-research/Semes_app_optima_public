@echo off
chcp 949 >nul
REM  * 한국어판 Windows 7 은 콘솔 코드페이지 949(CP949)가 기본이다.
REM     65001(UTF-8)로 바꾸면 Win7 cmd 에서 출력이 깨지거나 멈추는 문제가 있다.
REM     이 파일도 CP949 로 저장돼 있으니 편집 시 인코딩을 유지할 것.
REM ============================================================
REM  ChamberInspect - Windows 7 SP1 전용 빌드
REM  검증 대상: Windows 7 Professional K  6.1.7601 SP1 (빌드 7601), 64bit
REM  (WES7 - Windows Embedded Standard 7 - 도 같은 커널이라 그대로 사용 가능)
REM  * Win10 용 앱빌드.bat 과 완전히 별도입니다.
REM     - Python 3.8 (Win7 마지막 지원 버전) 필요
REM     - 패키지: src\app\requirements-win7.txt (cp38 핀)
REM     - WebView2 없이 브라우저 모드로 동작
REM     - 결과 폴더: ChamberInspect_WES7 (Win10 산출물을 덮어쓰지 않음)
REM ============================================================
setlocal
cd /d "%~dp0"
set "ROOT=%CD%"
set "SPEC=%~dp0src\app\io_console_win7.spec"
set "OUTDIR=%ROOT%\ChamberInspect_WIN7"
set "BLD=%SystemDrive%\iobuild7"
REM  * 한국어 Windows 는 파이썬/pip 의 기본 인코딩이 CP949 라, UTF-8 로 쓴 한글 주석이
REM     들어간 requirements 파일을 읽다가 "cp949 codec can't decode byte 0xe2" 로 죽는다.
REM     UTF-8 모드를 켜서 로케일과 무관하게 읽도록 한다(requirements 에는 BOM 도 넣어 두었다).
set "PYTHONUTF8=1"

REM ── 추론 백엔드 선택 (기본 onnx) ─────────────────────────────
REM   기본은 onnxruntime 입니다. set IO_WES7_BACKEND=torch 로 실행하면 torch 판을 만듭니다.
if /i "%IO_WES7_BACKEND%"=="torch" (
  set "REQ=%~dp0src\app\requirements-win7-torch.txt"
  set "BACKENDNAME=torch (.pt) - 용량 약 1GB"
) else (
  set "IO_WES7_BACKEND=onnx"
  set "REQ=%~dp0src\app\requirements-win7.txt"
  set "BACKENDNAME=onnxruntime 1.14.1 (.onnx)  *기본 - 가볍습니다"
)
REM ── 화면 방식 선택 (기본 네이티브 창) ────────────────────────
REM   기본은 브라우저 모드입니다. WebView2 109 런타임을 이미 확보한 경우에만
REM   set IO_WES7_UI=webview 로 네이티브 창 빌드를 하세요.
if /i "%IO_WES7_UI%"=="webview" ( set "UINAME=네이티브 창 (WebView2 109 런타임을 직접 준비해야 함)" ) else (
  set "IO_WES7_UI=browser"
  set "UINAME=브라우저 모드 (기본) - Firefox 115 ESR 권장"
)

echo ============================================================
echo    ChamberInspect - Windows 7 SP1 Build (Professional K 6.1.7601)
echo    추론 백엔드: %BACKENDNAME%
echo    화면 방식  : %UINAME%
echo    빌드 작업 폴더(영문): %BLD%
echo ============================================================
echo.

if not exist "%REQ%" goto NOREQ
if not exist "%SPEC%" goto NOSPEC

REM ── Python 3.8 만 허용 (3.9+ 로 빌드하면 Win7 에서 실행되지 않습니다) ──
set "PYCMD="
py -3.8 --version >nul 2>&1 && set "PYCMD=py -3.8"
if not defined PYCMD (
  python -c "import sys;sys.exit(0 if sys.version_info[:2]==(3,8) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD goto NOPY38
echo  [OK] Python: %PYCMD%
%PYCMD% -c "import struct,sys;sys.exit(0 if struct.calcsize('P')*8==64 else 1)" >nul 2>&1
if errorlevel 1 goto NOT64

echo.
echo  [1/3] 이전 빌드 정리...
if exist "%OUTDIR%" rd /s /q "%OUTDIR%"
if exist "%BLD%" rd /s /q "%BLD%"
mkdir "%BLD%"
if errorlevel 1 goto BLDFAIL

echo.
echo  [2/3] 가상환경 + 패키지 설치 (cp38 고정 버전)...
%PYCMD% -m venv "%BLD%\venv"
if errorlevel 1 goto VENVFAIL
set "VPY=%BLD%\venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade "pip<25"
REM  * --only-binary=:all: 를 쓰면 안 된다.
REM     pywebview 가 의존하는 proxy_tools 는 PyPI 에 wheel 이 없고 sdist(tar.gz)만 있어서
REM     "No matching distribution found for proxy-tools" 로 실패한다.
REM     --prefer-binary 만 쓰면 wheel 을 우선하되 없을 때만 sdist 를 쓴다(순수 파이썬이라 안전).
"%VPY%" -m pip install --prefer-binary -r "%REQ%"
if errorlevel 1 goto PIPFAIL

echo.
echo  [3/3] 빌드 (결과를 ChamberInspect_WES7 으로 복사)...
REM  * 빌드 로그를 파일로 남긴다. 화면만 보면 스크롤이 넘어가 원인을 놓친다.
set "LOG=%ROOT%\빌드로그.txt"
echo    (진행 상황은 %LOG% 에 기록됩니다. 몇 분 걸립니다)
"%VPY%" -m PyInstaller --clean --noconfirm --distpath "%BLD%\dist" --workpath "%BLD%\build" "%SPEC%" > "%LOG%" 2>&1
set "PIRC=%ERRORLEVEL%"
echo    PyInstaller 종료코드 = %PIRC%
if not "%PIRC%"=="0" (
  echo.
  echo  ---------------- 빌드 로그 마지막 40줄 ----------------
  powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 40"
  echo  ------------------------------------------------------
  goto BUILDFAIL
)

REM  * 산출물 폴더 이름을 고정하지 않고 직접 찾는다.
REM     스펙의 COLLECT 이름과 여기 적은 이름이 어긋나면 빌드가 멀쩡한데도
REM     "산출물이 없습니다" 로 실패한다(실제로 겪은 문제).
set "BUILT="
for /d %%D in ("%BLD%\dist\*") do if exist "%%~fD\ChamberInspect.exe" set "BUILT=%%~fD"
if not defined BUILT if exist "%BLD%\dist\ChamberInspect.exe" set "BUILT=%BLD%\dist"
if not defined BUILT (
  echo.
  echo  ---------------- 빌드 로그 마지막 40줄 ----------------
  powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 40"
  echo  ------------------------------------------------------
  goto NOEXE
)
echo  결과 폴더: %BUILT%
echo  복사 중... (파일이 수천 개라 잠시 걸립니다)
REM  * xcopy 는 파일을 한 개씩 처리해서, 백신 실시간 검사와 맞물리면 매우 느려진다
REM     (수천 개 파일에 10분 이상). robocopy 는 여러 스레드로 복사해 훨씬 빠르다.
REM     robocopy 는 윈도우 비스타부터 기본 포함이라 별도 설치가 필요 없다.
REM     종료코드 0~7 은 정상(복사함/건너뜀 등), 8 이상이 실패다.
where robocopy >nul 2>&1
if errorlevel 1 goto USEXCOPY
robocopy "%BUILT%" "%OUTDIR%" /E /MT:16 /R:1 /W:1 /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 goto COPYFAIL
goto COPYOK
:USEXCOPY
xcopy /e /i /y /q "%BUILT%" "%OUTDIR%\" >nul
if errorlevel 1 goto COPYFAIL
:COPYOK

REM  * 정리와 검사는 반드시 venv 를 지우기 '전에' 한다.
REM     %VPY% 가 그 venv 안의 python 이라, 먼저 지우면 실행 파일이 없어 조용히 실패한다.
REM     (실제로 그렇게 만들어 놓아 폴더 정리가 동작하지 않았다)
echo.
echo  [정리] 실행 파일만 남기고 지원 파일을 build 폴더로 옮깁니다...
if not exist "%VPY%" (
  echo  [!] 빌드용 파이썬을 찾지 못했습니다: %VPY%
  echo      폴더 정리를 건너뜁니다.
) else (
  "%VPY%" "%~dp0src\app\arrange_dist.py" "%OUTDIR%"
  if errorlevel 1 echo  [!] 폴더 정리 실패 - 앱은 동작하지만 파일이 한 폴더에 함께 있습니다.
)

echo.
echo  [검사] Windows 7 에서 로드 가능한지 확인합니다(PE 임포트 테이블)...
"%VPY%" "%~dp0src\app\check_win7_compat.py" "%OUTDIR%\build"
if errorlevel 1 goto WIN7FAIL

REM  정리-검사가 끝난 뒤에 빌드 작업 폴더(venv 포함)를 지운다.
if exist "%BLD%" rd /s /q "%BLD%"

echo.
echo ============================================================
echo  완료. 앱 폴더:  %OUTDIR%
echo  실행 파일:  ChamberInspect.exe  (콘솔창 없이 뜹니다)
echo  나머지 파일은 build 폴더 안에 있습니다 - 건드리지 마세요
echo  기록은 콘솔 대신 파일로 - 로그\app_날짜.log - 앱_시작오류.txt
echo  콘솔을 보며 진단하려면 빌드 전에:  set IO_WES7_CONSOLE=1
echo.
echo  Windows 7 제어기에서 확인할 것:
echo   1) 브라우저 모드(기본)  : 제어기에 Firefox 115 ESR 설치 (인터넷 익스플로러는 불가)
echo      네이티브 창(선택) : WebView2 고정버전 109 런타임 + .NET 4.6.2 이상이 이미 있을 때만
echo   2) .NET Framework 4.6.2 이상 필요(4.8 권장). Windows 7 은 기본이 3.5.1 이라 별도 설치입니다.
echo      ^(WES7 처럼 쓰기 필터를 쓰는 이미지라면 설정 - 저장소 위치로 쓰기 가능한 경로를 지정^)
echo   3) Visual C++ 2015-2019 재배포 패키지(x64) 필요
echo   4) onnx 판(기본)은 모델 폴더에 .onnx + .npz + search_best.json 을 넣으세요.
echo      * 모델을 한 번 변환해야 합니다: 5.모델 변환.bat 실행
echo         (Windows 7 용 onnxruntime 1.14 는 최신 IR 형식을 못 읽습니다)
echo      torch 판으로 빌드했다면 대신 tcn_forecast.pt 를 넣으세요(용량 약 1GB).
echo ============================================================
goto END

:WIN7FAIL
echo.
echo ============================================================
echo  [X] 위 파일들이 Windows 7 에서 로드되지 않습니다.
echo      빌드는 끝났지만 WES7 제어기에서는 실행되지 않습니다.
echo      표시된 패키지의 버전을 낮춘 뒤 다시 빌드하세요.
echo      (torch 는 2.3.1 이하, onnxruntime 은 위 검사 결과 참고)
echo      앱 폴더는 확인용으로 남겨 둡니다: %OUTDIR%
echo ============================================================
goto END

:NOEXE
echo  [X] 빌드 산출물(ChamberInspect.exe)을 %BLD%\dist 아래에서 찾지 못했습니다.
echo      전체 로그: %LOG%
echo      아래는 작업 폴더에 실제로 만들어진 것들입니다:
dir /b "%BLD%" 2>nul
dir /b "%BLD%\dist" 2>nul
goto END
:NOREQ
echo  [X] 목록 파일이 없습니다: %REQ%
goto END
:NOSPEC
echo  [X] 빌드 스펙이 없습니다: %SPEC%
goto END
:NOPY38
echo  [X] Python 3.8 (64bit) 이 필요합니다.
echo      Windows 7 을 지원하는 마지막 파이썬이라 반드시 3.8 로 빌드해야 합니다.
echo      3.9 이상으로 빌드하면 WES7 에서 실행되지 않습니다.
echo      https://www.python.org/downloads/release/python-3810/
echo      (Windows x86-64 executable installer)
goto END
:NOT64
echo  [X] 64비트 Python 3.8 이 필요합니다(현재 32비트).
goto END
:BLDFAIL
echo  [X] 작업폴더(%BLD%) 생성 실패. 드라이브 쓰기권한을 확인하세요.
goto END
:VENVFAIL
echo  [X] 가상환경 생성 실패.
goto END
:PIPFAIL
echo  [X] 패키지 설치 실패.
echo      Windows 7 빌드 PC 라면 TLS 1.2 미지원으로 pip 가 PyPI 에 못 붙을 수 있습니다.
echo      가능하면 Win10 PC 에서 Python 3.8 로 빌드하세요(산출물은 WES7 에서 동작합니다).
goto END
:BUILDFAIL
echo  [X] 빌드 실패. 전체 내용은 아래 파일에 있습니다:
echo      %LOG%
echo      그 파일을 보내주시면 원인을 짚어드리겠습니다.
goto END
:COPYFAIL
echo  [X] 결과 복사 실패(%OUTDIR%). 디스크 공간/권한 확인.
goto END

:END
echo.
pause
exit /b 0
