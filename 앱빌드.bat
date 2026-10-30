@echo off
chcp 65001 >nul
REM ============================================================
REM  ChamberInspect - App Build (내 PC, Python 3.10~3.12 필요)
REM  ★ 빌드는 짧은 영문 경로(C:\iobuild)에서 수행합니다. 한글/긴 경로에서 돌리면
REM     PyInstaller 가 패키지 dist-info 를 못 찾는 "unable to add ...dist-info" 에러가
REM     나기 때문입니다. 결과 폴더(ChamberInspect)만 여기로 복사합니다.
REM ============================================================
setlocal
cd /d "%~dp0"
set "ROOT=%CD%"
set "REQ=%~dp0src\app\requirements-win.txt"
set "SPEC=%~dp0src\app\io_console_webview.spec"
set "OUTDIR=%ROOT%\ChamberInspect"
set "BLD=%SystemDrive%\iobuild"
REM  ★ 한국어 Windows 에서 pip 가 requirements 를 CP949 로 읽어 실패하는 것 방지
set "PYTHONUTF8=1"

echo ============================================================
echo    ChamberInspect - App Build
echo    빌드 작업 폴더(영문): %BLD%
echo ============================================================
echo.

if not exist "%REQ%" goto NOREQ
if not exist "%SPEC%" goto NOSPEC

set "PYCMD="
py -3.12 --version >nul 2>&1 && set "PYCMD=py -3.12"
if not defined PYCMD ( py -3.11 --version >nul 2>&1 && set "PYCMD=py -3.11" )
if not defined PYCMD ( py -3.10 --version >nul 2>&1 && set "PYCMD=py -3.10" )
if not defined PYCMD ( python -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set "PYCMD=python" )
if not defined PYCMD goto NOPY
echo  [OK] Python: %PYCMD%

echo.
echo  [1/3] 이전 빌드 정리...
if exist "%OUTDIR%" rd /s /q "%OUTDIR%"
if exist "%BLD%" rd /s /q "%BLD%"
mkdir "%BLD%"
if errorlevel 1 goto BLDFAIL

echo.
echo  [2/3] 가상환경 + 패키지 설치 (영문 경로에서)...
%PYCMD% -m venv "%BLD%\venv"
if errorlevel 1 goto VENVFAIL
set "VPY=%BLD%\venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install --prefer-binary -r "%REQ%"
if errorlevel 1 goto PIPFAIL

echo.
echo  [3/3] 빌드 (영문 경로 %BLD% 에서 -^> 결과를 ChamberInspect 으로 복사)...
"%VPY%" -m PyInstaller --clean --noconfirm --distpath "%BLD%\dist" --workpath "%BLD%\build" "%SPEC%"
if errorlevel 1 goto BUILDFAIL

if not exist "%BLD%\dist\ChamberInspect\ChamberInspect.exe" goto NOEXE
echo  결과 복사: %BLD%\dist\ChamberInspect  -^>  %OUTDIR%
echo  복사 중... (파일이 수천 개라 잠시 걸립니다)
REM  ★ xcopy 는 파일 하나씩 처리해 백신 검사와 맞물리면 매우 느리다 → robocopy 멀티스레드
where robocopy >nul 2>&1
if errorlevel 1 goto USEXCOPY
robocopy "%BLD%\dist\ChamberInspect" "%OUTDIR%" /E /MT:16 /R:1 /W:1 /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 goto COPYFAIL
goto COPYOK
:USEXCOPY
xcopy /e /i /y /q "%BLD%\dist\ChamberInspect" "%OUTDIR%\" >nul
if errorlevel 1 goto COPYFAIL
:COPYOK

if exist "%BLD%" rd /s /q "%BLD%"

echo.
echo ============================================================
echo  완료. 앱 폴더:  %OUTDIR%
echo  실행 파일:  ChamberInspect.exe
echo  생성된 'ChamberInspect' 폴더를 제어기로 옮겨 사용하세요.
echo ============================================================
if not exist "%OUTDIR%\ChamberInspect.exe" goto NOEXE2
echo.
echo  (테스트) 프로그램을 실행합니다...
start "" "%OUTDIR%\ChamberInspect.exe"
goto END

:NOEXE
echo  [X] 빌드 산출물(exe)이 %BLD%\dist 에 없습니다. 위 메시지를 확인하세요.
goto END
:NOEXE2
echo  (복사 후 실행 파일이 없습니다. 위 메시지를 확인하세요.)
goto END
:NOREQ
echo  [X] 목록 파일이 없습니다: %REQ%
goto END
:NOSPEC
echo  [X] 빌드 스펙이 없습니다: %SPEC%
goto END
:NOPY
echo  [X] Python 3.10~3.12(64bit) 가 없습니다. 설치 후 다시 실행하세요:
echo      https://www.python.org/downloads/release/python-3129/
goto END
:BLDFAIL
echo  [X] 영문 작업폴더(%BLD%) 생성 실패. C: 드라이브 쓰기권한을 확인하세요.
goto END
:VENVFAIL
echo  [X] 가상환경 생성 실패.
goto END
:PIPFAIL
echo  [X] 패키지 설치 실패(인터넷/pip 확인).
goto END
:BUILDFAIL
echo  [X] 빌드 실패(위 메시지 확인).
goto END
:COPYFAIL
echo  [X] 결과 복사 실패(%OUTDIR%). 디스크 공간/권한 확인.
goto END

:END
echo.
pause
exit /b 0
