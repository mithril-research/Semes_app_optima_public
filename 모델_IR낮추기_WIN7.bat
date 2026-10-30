@echo off
chcp 949 >nul
title 모델 변환 (Windows 7 용 IR 낮추기)
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo ============================================================
echo   모델을 Windows 7 용으로 변환 (IR 버전 낮추기)
echo ============================================================
echo.
echo  이 파일이 있는 폴더와 그 아래 모든 하위 폴더를 훑어서
echo  .onnx 파일을 찾아 자동으로 변환합니다.
echo.
echo    예)  모델\
echo           20260801\tcn_forecast.onnx
echo           20260815\tcn_forecast.onnx
echo         -^> 모델 폴더에 이 파일을 두고 실행하면 둘 다 변환됩니다.
echo.
echo  모델 내용은 바뀌지 않습니다. 원본은 .onnx.bak 으로 백업되고,
echo  이미 변환된 파일은 건너뜁니다. 여러 번 실행해도 안전합니다.
echo.

REM  인자로 폴더를 주면 그 폴더를, 안 주면 이 배치가 있는 폴더를 대상으로 한다.
REM  (파일을 끌어다 놓아도 되고, 그냥 더블클릭해도 된다)
set "DIR=%~1"
if "%DIR%"=="" set "DIR=%~dp0"

echo  대상 폴더: %DIR%
echo.

set "PYCMD="
py -3.8 --version >nul 2>&1 && set "PYCMD=py -3.8"
if not defined PYCMD ( py --version >nul 2>&1 && set "PYCMD=py" )
if not defined PYCMD ( python --version >nul 2>&1 && set "PYCMD=python" )
if not defined PYCMD goto NOPY

REM  변환 도구 위치: 이 배치 옆의 src\app, 또는 상위 폴더의 src\app
set "TOOL=%~dp0src\app\onnx_ir_downgrade.py"
if not exist "%TOOL%" set "TOOL=%~dp0..\src\app\onnx_ir_downgrade.py"
if not exist "%TOOL%" goto NOTOOL

%PYCMD% "%TOOL%" "%DIR%"
goto END

:NOPY
echo  [X] 파이썬을 찾지 못했습니다.
echo      이 작업은 빌드용 PC 에서 하세요(제어기에는 파이썬이 없습니다).
goto END
:NOTOOL
echo  [X] 변환 도구를 찾지 못했습니다: src\app\onnx_ir_downgrade.py
echo      배포 폴더 안에서 실행하거나, 이 파일을 모델 폴더로 복사해 쓰는 경우
echo      배포 폴더의 src 폴더도 함께 있어야 합니다.
goto END

:END
echo.
pause
