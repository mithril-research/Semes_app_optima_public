@echo off
chcp 949 >nul
title 챔버 이상 검사 시스템 - 윈도우7 제어기 전제조건 설치
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ★ 괄호 블록 안에서 %VAR% 는 '블록을 읽는 시점'에 값이 정해진다.
REM   설치 직후의 ERRORLEVEL 을 그 안에서 읽으면 이전 값이 나온다.
REM   그래서 설치는 서브루틴(:RUN)으로 빼고, 결과 판정도 블록 밖에서 한다.

echo ============================================================
echo   Windows 7 제어기 전제조건 설치
echo   인터넷 없이 설치합니다. 미리 받아 둔 '설치본' 폴더가 필요합니다.
echo ============================================================
echo.

net session >nul 2>&1
if errorlevel 1 goto NOADMIN
echo  [OK] 관리자 권한 확인

set "SRC=%~dp0설치본"
if not exist "%SRC%" set "SRC=%~dp0"
echo  [i] 설치 파일 위치: %SRC%
echo.
set "FAIL=0"

REM ── 0) Universal C Runtime (KB2999226) : 윈도우7 에서 가장 먼저 ──
REM   파이썬 3.8 이 이 런타임을 요구한다. 윈10 이상은 OS 에 있지만 윈7 은 없어서
REM   "api-ms-win-crt-runtime-l1-1-0.dll 이 없습니다" 오류가 난다.
echo  [0/4] Universal C Runtime (KB2999226)
if exist "%SystemRoot%\System32\ucrtbase.dll" goto UCRTOK
if not exist "%SRC%\Windows6.1-KB2999226-x64.msu" goto NOUCRT
echo      설치 중... 몇 분 걸릴 수 있습니다
wusa.exe "%SRC%\Windows6.1-KB2999226-x64.msu" /quiet /norestart
set "RC=!ERRORLEVEL!"
if "!RC!"=="0" echo      설치 완료
if "!RC!"=="3010" echo      설치 완료 - 재부팅 후 앱을 실행하세요
if "!RC!"=="2359302" echo      이미 설치돼 있습니다
if "!RC!"=="0" goto UCRTDONE
if "!RC!"=="3010" goto UCRTDONE
if "!RC!"=="2359302" goto UCRTDONE
echo      ★실패 ^(코드 !RC!^) - 윈도우 업데이트 서비스가 꺼져 있으면 실패합니다
set "FAIL=1"
goto UCRTDONE
:UCRTOK
echo      이미 설치돼 있습니다 ^(ucrtbase.dll 확인^)
goto UCRTDONE
:NOUCRT
echo      ★파일 없음: Windows6.1-KB2999226-x64.msu
echo        이것이 없으면 앱 실행 시 api-ms-win-crt-runtime-l1-1-0.dll 오류가 납니다.
set "FAIL=1"
:UCRTDONE
echo.

REM ── 1) Visual C++ 2015-2019 재배포 패키지 (필수) ─────────────
echo  [1/4] Visual C++ 2015-2019 재배포 패키지 (x64)
if not exist "%SRC%\vc_redist.x64.exe" goto NOVC
call :RUN "%SRC%\vc_redist.x64.exe" /install /quiet /norestart
goto VCDONE
:NOVC
echo      ★파일 없음: vc_redist.x64.exe
echo        인터넷 되는 PC 에서 윈도우7_설치본_받기.bat 을 먼저 실행하세요.
set "FAIL=1"
:VCDONE
echo.

REM ── 2) Firefox 115 ESR (화면 표시에 필요) ────────────────────
echo  [2/4] Firefox 115 ESR (한국어)
set "FF="
for %%F in ("%SRC%\Firefox*.exe") do if not defined FF set "FF=%%~fF"
if not defined FF goto NOFF
echo      !FF!
call :RUN "!FF!" -ms
goto FFDONE
:NOFF
echo      ★파일 없음: Firefox 설치본
echo        앱 화면을 브라우저로 띄웁니다. 인터넷 익스플로러로는 화면이 나오지 않습니다.
set "FAIL=1"
:FFDONE
echo.

REM ── 3) .NET Framework 4.8 (선택) ─────────────────────────────
echo  [3/4] .NET Framework 4.8   선택 - 브라우저 모드에서는 필요 없습니다
if not exist "%SRC%\ndp48-offline.exe" goto NONET
choice /C YN /N /M "  설치하시겠습니까? (Y/N) "
if errorlevel 2 goto SKIPNET
call :RUN "%SRC%\ndp48-offline.exe" /q /norestart
goto NETDONE
:SKIPNET
echo      건너뜀
goto NETDONE
:NONET
echo      파일 없음 - 건너뜀
:NETDONE
echo.

REM ── 설치 확인 ────────────────────────────────────────────────
echo ------------------------------------------------------------
echo  설치 확인
if exist "%SystemRoot%\System32\ucrtbase.dll" (echo   UCRT         : 설치됨) else (echo   UCRT         : 없음 ★ 앱이 실행되지 않습니다)
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if errorlevel 1 (echo   VC++ 재배포  : 확인 안 됨 ★) else (echo   VC++ 재배포  : 설치됨)
set "FFOK="
if exist "%ProgramFiles%\Mozilla Firefox\firefox.exe" set "FFOK=1"
if exist "%ProgramW6432%\Mozilla Firefox\firefox.exe" set "FFOK=1"
if exist "C:\Program Files (x86)\Mozilla Firefox\firefox.exe" set "FFOK=1"
if defined FFOK (echo   Firefox      : 설치됨) else (echo   Firefox      : 확인 안 됨 ★)
reg query "HKLM\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full" /v Release >nul 2>&1
if errorlevel 1 (echo   .NET 4.x     : 없음  브라우저 모드면 무관) else (call :NETVER)
echo ------------------------------------------------------------
echo.
if "%FAIL%"=="1" goto SOMEFAIL
echo  전제조건 설치가 끝났습니다.
echo  이제 앱 폴더의 ChamberInspect.exe 를 실행하세요.
echo  오류를 보시려면 실행_창유지_WIN7.bat 을 쓰세요.
goto END

:SOMEFAIL
echo  ★ 일부 항목이 설치되지 않았습니다. 위 표시를 확인해 주세요.
goto END

:NOADMIN
echo  [X] 관리자 권한이 필요합니다.
echo      이 파일을 우클릭 - "관리자 권한으로 실행" 으로 다시 실행하세요.
goto END

REM ── 설치 실행 + 결과 해석(블록 밖이라 ERRORLEVEL 이 정확하다) ──
:RUN
"%~1" %2 %3 %4
set "RC=!ERRORLEVEL!"
if "!RC!"=="0" echo      설치 완료
if "!RC!"=="3010" echo      설치 완료 - 재부팅이 필요합니다
if "!RC!"=="1638" echo      이미 같거나 최신 버전이 설치돼 있습니다
if "!RC!"=="0" exit /b
if "!RC!"=="3010" exit /b
if "!RC!"=="1638" exit /b
echo      ★실패 ^(코드 !RC!^)
set "FAIL=1"
exit /b

:NETVER
for /f "tokens=3" %%R in ('reg query "HKLM\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full" /v Release 2^>nul ^| find "Release"') do echo   .NET Release : %%R   394802 이상이면 4.6.2+
exit /b

:END
echo.
pause
