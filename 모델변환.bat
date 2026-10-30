@echo off
chcp 65001 >nul
REM ============================================================
REM  모델 변환 (.pt -> ONNX)  ·  내 PC (Python 3.10~3.12)
REM  학습된 모델을 .pt 로 받으면, exe 가 torch 없이 추론하도록
REM  .onnx(+메타) 로 변환합니다.  변환에는 torch 가 필요하므로
REM  이 스텝은 별도 venv 에 torch 를 설치해 수행합니다(1회).
REM
REM  사용법 1) 이 .bat 위에 '모델 .pt 파일' 또는 '모델 폴더' 를 드래그&드롭
REM  사용법 2) 그냥 실행하면 경로를 물어봅니다
REM  결과물: 같은 폴더에 tcn_forecast.onnx + recon_meta.npz 생성
REM          -> 그 폴더를 제어기 '모델' 폴더로 넣으면 됩니다
REM ============================================================
setlocal
cd /d "%~dp0"
set "ROOT=%CD%"
set "REQ=%~dp0모델변환_requirements.txt"

set "TARGET=%~1"
if "%TARGET%"=="" set /p "TARGET=변환할 .pt 파일 또는 모델 폴더 경로를 입력하세요: "
if "%TARGET%"=="" goto NOTARGET
set "TARGET=%TARGET:"=%"

set "PYCMD="
py -3.12 --version >nul 2>&1 && set "PYCMD=py -3.12"
if not defined PYCMD ( py -3.11 --version >nul 2>&1 && set "PYCMD=py -3.11" )
if not defined PYCMD ( py -3.10 --version >nul 2>&1 && set "PYCMD=py -3.10" )
if not defined PYCMD ( python -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set "PYCMD=python" )
if not defined PYCMD goto NOPY
echo  [OK] Python: %PYCMD%

echo.
echo  [1/2] 변환용 가상환경 준비(torch 포함, 최초 1회는 다소 걸립니다)...
if not exist venv_convert (
  %PYCMD% -m venv venv_convert
  if errorlevel 1 goto VENVFAIL
)
set "VPY=%ROOT%\venv_convert\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip >nul
"%VPY%" -m pip install -r "%REQ%"
if errorlevel 1 goto PIPFAIL

echo.
echo  [2/2] 변환 실행: %TARGET%
"%VPY%" -m src.export_onnx_forecast "%TARGET%"
if errorlevel 1 goto CONVFAIL

echo.
echo ============================================================
echo  변환 완료. 대상 폴더에 tcn_forecast.onnx + recon_meta.npz 생성됨.
echo  이 폴더를 제어기 '모델' 폴더로 넣으면 exe 가 바로 사용합니다.
echo ============================================================
goto END

:NOTARGET
echo  [X] 대상(.pt 또는 폴더)이 지정되지 않았습니다.
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
:CONVFAIL
echo  [X] 변환 실패(위 메시지 확인). .pt 가 올바른 학습 체크포인트인지 확인하세요.
goto END

:END
echo.
pause
exit /b 0
