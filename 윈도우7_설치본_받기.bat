@echo off
chcp 949 >nul
REM ============================================================
REM  Windows 7 제어기 준비물 내려받기 (인터넷 되는 PC 에서 실행)
REM  받은 파일을 USB 등으로 제어기에 옮겨 설치하세요.
REM
REM  ★ URL 에 % 를 쓰지 않는다.
REM     cmd 의 call 은 %20 같은 표기를 '인자'로 다시 해석해서 주소를 망가뜨린다
REM     (Firefox%20Setup... -> Firefox0Setup... 이 되어 404).
REM     그래서 퍼센트 인코딩이 없는 download.mozilla.org 주소를 쓴다.
REM ============================================================
setlocal
cd /d "%~dp0"
set "OUT=%~dp0설치본"
if not exist "%OUT%" mkdir "%OUT%"

echo ============================================================
echo   Windows 7 제어기 준비물 다운로드
echo   저장 위치: %OUT%
echo ============================================================
echo.

set "VCURL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
set "FFURL=https://download.mozilla.org/?product=firefox-115.39.0esr-SSL&os=win64&lang=ko"
set "NETURL=https://go.microsoft.com/fwlink/?linkid=2088631"
REM  ★ Universal C Runtime (KB2999226) - 파이썬 3.8 이 요구한다.
REM     윈도우10 이상은 OS 에 포함돼 있지만 윈도우7 은 따로 깔아야 한다.
REM     없으면 실행 시 "api-ms-win-crt-runtime-l1-1-0.dll 이 없습니다" 가 뜬다.
set "UCRTURL=https://download.microsoft.com/download/1/1/5/11565A9A-EA09-4F0A-A57E-520D5D138140/Windows6.1-KB2999226-x64.msu"

echo  [0/4] Universal C Runtime KB2999226 (x64) ... 약 1MB  ^(윈도우7 필수^)
call :GET "%UCRTURL%" "%OUT%\Windows6.1-KB2999226-x64.msu"

echo  [1/4] Visual C++ 2015-2019 재배포 패키지 (x64) ... 약 26MB  ^(필수^)
call :GET "%VCURL%" "%OUT%\vc_redist.x64.exe"

echo  [2/4] Firefox 115.39 ESR 한국어 (64bit) ... 약 59MB  ^(화면 표시에 필요^)
call :GET "%FFURL%" "%OUT%\Firefox_115ESR_ko_x64.exe"

echo  [3/4] .NET Framework 4.8 오프라인 ... 약 121MB  ^(네이티브 창을 쓸 때만 필요^)
choice /C YN /N /M "  받으시겠습니까? (Y/N) "
if errorlevel 2 (echo     - 건너뜀) else (call :GET "%NETURL%" "%OUT%\ndp48-offline.exe")

echo.
echo ============================================================
echo  완료. 폴더: %OUT%
echo.
echo  받은 파일이 0KB 이거나 실패로 나오면, 아래 주소를 브라우저에 직접 붙여넣어
echo  받으셔도 됩니다.
echo    Firefox 115 ESR : https://download.mozilla.org/?product=firefox-115.39.0esr-SSL^&os=win64^&lang=ko
echo    VC++ 재배포      : https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
echo  ※ WebView2 고정버전 런타임 109 는 마이크로소프트가 더 이상 배포하지 않습니다.
echo     기본 구성은 '브라우저 모드 + Firefox 115 ESR' 입니다.
echo ============================================================
goto :END

:GET
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; try{ [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%~1' -OutFile '%~2' -UseBasicParsing; $s=(Get-Item '%~2').Length; Write-Host ('     - 완료 ' + [math]::Round($s/1MB,1) + 'MB') } catch { Write-Host ('     - 실패: ' + $_.Exception.Message) }"
exit /b

:END
echo.
pause
