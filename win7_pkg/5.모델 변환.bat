@echo off
chcp 949 >nul
title 모델 IR 낮추기 (Windows 7 용 onnxruntime 대응)
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo ============================================================
echo   ONNX 모델을 Windows 7 용으로 변환 (IR 버전 낮추기)
echo ============================================================
echo.
echo  Windows 7 에서 쓸 수 있는 onnxruntime 은 1.14 가 상한인데,
echo  학습앱이 만드는 .onnx 는 그보다 새 형식(IR 10)이라 그대로는 못 읽습니다.
echo  이 도구가 형식 표기만 낮춰 줍니다. 모델 내용은 바뀌지 않습니다.
echo  (원본은 .onnx.bak 으로 백업됩니다)
echo.

set "DIR=%~1"
if "%DIR%"=="" set /p DIR=모델 폴더 경로를 입력하세요: 
if "%DIR%"=="" goto :NODIR

set "PYCMD="
py -3.8 --version >nul 2>&1 && set "PYCMD=py -3.8"
if not defined PYCMD ( py --version >nul 2>&1 && set "PYCMD=py" )
if not defined PYCMD ( python --version >nul 2>&1 && set "PYCMD=python" )
if not defined PYCMD goto :NOPY

%PYCMD% "%~dp0src\app\onnx_ir_downgrade.py" "%DIR%"
goto :END

:NODIR
echo  [X] 모델 폴더를 지정하지 않았습니다.
echo      이 배치 파일 위로 모델 폴더를 끌어다 놓아도 됩니다.
goto :END
:NOPY
echo  [X] 파이썬을 찾지 못했습니다. 빌드 PC 에서 실행하세요.
goto :END

:END
echo.
pause
