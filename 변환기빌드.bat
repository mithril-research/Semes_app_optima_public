@echo off
chcp 65001 >nul
REM ============================================================
REM  모델 변환기 빌드 (.pt -> ONNX 전용 exe)  ·  내 PC (Python 3.10~3.12)
REM  torch 를 포함한 '모델변환.exe' 를 만들어, 제어기에 메인 앱과 함께 둡니다.
REM  새 .pt 모델이 오면 제어기에서 '모델변환.exe' 만 실행 -> onnx 생성 + .pt 삭제.
REM  (메인 앱 exe 는 경량 유지 — 이 변환기 exe 에만 torch 가 들어갑니다)
REM ============================================================
setlocal
cd /d "%~dp0"
set "ROOT=%CD%"
set "REQ=%~dp0모델변환_requirements.txt"
set "SPEC=%~dp0src\app\모델변환.spec"
set "OUTDIR=%ROOT%\모델변환"

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
if exist venv_convbuild rd /s /q venv_convbuild
if exist build_conv rd /s /q build_conv
if exist "%OUTDIR%" rd /s /q "%OUTDIR%"

echo.
echo  [2/3] 가상환경 + 패키지 설치(torch 포함, 최초 1회는 오래 걸립니다)...
%PYCMD% -m venv venv_convbuild
if errorlevel 1 goto VENVFAIL
set "VPY=%ROOT%\venv_convbuild\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r "%REQ%" pyinstaller
if errorlevel 1 goto PIPFAIL

echo.
echo  [3/3] 변환기 빌드 (모델변환\모델변환.exe 생성)...
"%VPY%" -m PyInstaller --clean --noconfirm --distpath "%ROOT%" --workpath "%ROOT%\build_conv" "%SPEC%"
if errorlevel 1 goto BUILDFAIL
if exist build_conv rd /s /q build_conv
if exist venv_convbuild rd /s /q venv_convbuild

echo.
echo ============================================================
echo  완료. 변환기 폴더:  %OUTDIR%
echo  실행 파일:  모델변환.exe
echo  '모델변환' 폴더를 제어기의 앱 폴더 옆(또는 안)에 함께 두세요.
echo  새 .pt 모델이 오면: 모델 폴더/.pt 를 '모델변환.exe' 에 드래그&드롭
echo    -^> .onnx 생성 + 원본 .pt 자동 삭제.
echo ============================================================
if not exist "%OUTDIR%\모델변환.exe" goto NOEXE
goto END

:NOEXE
echo  (실행 파일이 생성되지 않았습니다. 위 메시지를 확인하세요.)
goto END
:NOREQ
echo  [X] 목록 파일이 없습니다: %REQ%
goto END
:NOSPEC
echo  [X] 빌드 스펙이 없습니다: %SPEC%
goto END
:NOPY
echo  [X] Python 3.10~3.12(64bit) 가 필요합니다: https://www.python.org/downloads/release/python-3129/
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

:END
echo.
pause
exit /b 0
