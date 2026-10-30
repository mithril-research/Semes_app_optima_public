@echo off
chcp 65001 >nul
setlocal
title 챔버 이상 검사 시스템 - 시작 진단
echo ============================================================
echo  앱이 "실행해도 아무 반응이 없을 때" 원인을 찾는 진단 도구
echo  (이 창을 닫지 말고 끝까지 두세요)
echo ============================================================
echo.

set "HERE=%~dp0"
cd /d "%HERE%"

rem ── 1) exe 찾기 ─────────────────────────────────────────────
set "EXE="
for %%F in ("%HERE%*.exe") do if not defined EXE set "EXE=%%~fF"
if not defined EXE for /r "%HERE%" %%F in (*.exe) do if not defined EXE set "EXE=%%~fF"
if not defined EXE (
  echo [X] 이 폴더에서 실행파일(.exe)을 찾지 못했습니다.
  echo     이 파일을 앱 exe 와 같은 폴더에 두고 다시 실행하세요.
  goto :END
)
echo [1] 실행파일 : %EXE%

rem ── 2) 차단 표시(다른 PC에서 받은 파일) 해제 ────────────────
echo [2] 차단 해제 시도(인터넷에서 받은 파일 표시 제거)
powershell -NoProfile -Command "Get-ChildItem -LiteralPath '%HERE%' -Recurse -File | Unblock-File" 2>nul
if errorlevel 1 (echo     - 건너뜀^(PowerShell 사용 불가^)) else (echo     - 완료)

rem ── 3) 이전 시작 기록 정리 ──────────────────────────────────
if exist "%HERE%앱_시작오류.txt" del /q "%HERE%앱_시작오류.txt"
if exist "%HERE%앱_시작기록.txt" del /q "%HERE%앱_시작기록.txt"

rem ── 4) 이미 떠 있는지 확인 ──────────────────────────────────
for %%N in ("%EXE%") do set "EXENAME=%%~nxN"
tasklist /FI "IMAGENAME eq %EXENAME%" 2>nul | find /i "%EXENAME%" >nul
if not errorlevel 1 (
  echo [3] 주의: 같은 앱이 이미 실행 중입니다. 트레이/작업관리자에서 종료 후 다시 시도하세요.
) else (
  echo [3] 중복 실행 없음
)

rem ── 5) 실행하고 종료코드 확인 ───────────────────────────────
echo [4] 실행합니다... 창이 뜰 때까지 최대 2분 걸릴 수 있습니다.
echo.
"%EXE%"
set "RC=%ERRORLEVEL%"
echo.
echo [5] 종료코드 = %RC%
if "%RC%"=="0" echo     - 정상 종료(사용자가 창을 닫음)
if "%RC%"=="1" echo     - 시작 실패. 아래 오류 내용을 확인하세요.
if "%RC%"=="-1073741515" echo     - 필수 DLL 누락^(VC++ 재배포 패키지 미설치 가능성^)
if "%RC%"=="-1073741819" echo     - 프로세스가 비정상 종료^(백신 차단 가능성^)

rem ── 6) 결과 파일 출력 ───────────────────────────────────────
echo.
echo ============================ 시작 기록 ============================
if exist "%HERE%앱_시작기록.txt" (type "%HERE%앱_시작기록.txt") else (echo   (기록 없음 - 앱이 시작 직후에 종료되었습니다^))
echo.
echo ============================ 오류 내용 ============================
if exist "%HERE%앱_시작오류.txt" (type "%HERE%앱_시작오류.txt") else (echo   (오류 파일 없음^))
echo.
echo ============================ 최근 로그 ============================
set "LOGDIR=%HERE%로그"
if exist "%LOGDIR%" (
  for /f "delims=" %%L in ('dir /b /o-d "%LOGDIR%\app_*.log" 2^>nul') do (
    echo --- %%L ^(마지막 40줄^) ---
    powershell -NoProfile -Command "Get-Content -LiteralPath '%LOGDIR%\%%L' -Tail 40"
    goto :LOGDONE
  )
  echo   (로그 파일 없음^)
) else (
  echo   (로그 폴더 없음 - 앱이 로그를 만들기도 전에 죽었습니다^)
)
:LOGDONE

echo.
echo ==================================================================
echo  위 내용을 통째로 복사해서 보내주시면 원인을 바로 잡을 수 있습니다.
echo  ^(앱_시작오류.txt 파일을 그대로 첨부하셔도 됩니다^)
echo ==================================================================
:END
echo.
pause
endlocal
