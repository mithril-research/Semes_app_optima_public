@echo off
REM ============================================================
REM  ChamberInspect - Download offline installers (run on an ONLINE PC)
REM  Downloads the two Microsoft installers into THIS folder, so the
REM  whole folder can be shipped to the (offline) controller PC.
REM
REM  Fix note: URLs are passed to the downloader via environment
REM  variables, never inlined into an IF one-liner. cmd.exe used to
REM  re-parse the '?linkid=2124701' part and abort with
REM  "2124701 was unexpected at this time".
REM  ASCII only. Uses labels and subroutines (no risky one-liners).
REM ============================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "VCURL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
set "WV2URL=https://go.microsoft.com/fwlink/?linkid=2124701"
set "VCOUT=vc_redist.x64.exe"
set "WV2OUT=MicrosoftEdgeWebView2RuntimeInstallerX64.exe"

echo.
echo  ChamberInspect - Download prerequisites
echo  ------------------------------------------------------------

REM --- pick a downloader: curl.exe (Win10 1803+) else PowerShell ---
set "DL=ps"
where curl.exe >nul 2>nul
if errorlevel 1 goto PICKED
set "DL=curl"
:PICKED
echo  downloader : %DL%
echo  target dir : %CD%
echo.

echo  [1/2] %VCOUT%   (about 25 MB)
call :GET "%VCURL%" "%VCOUT%"
echo.
echo  [2/2] %WV2OUT%   (about 195 MB, may take a while)
call :GET "%WV2URL%" "%WV2OUT%"

echo.
echo  ------------------------------------------------------------
echo  Result (size must be MB, not a few KB):
if exist "%VCOUT%"  call :SHOW "%VCOUT%"
if not exist "%VCOUT%"  echo    [X] %VCOUT% missing
if exist "%WV2OUT%" call :SHOW "%WV2OUT%"
if not exist "%WV2OUT%" echo    [X] %WV2OUT% missing
echo.
echo  If both look right, copy THIS whole folder to the controller PC
echo  and run IO_Prerequisites_Setup.bat there as Administrator.
echo.
echo  If a download failed, open these links in a browser and save the
echo  files into this folder manually:
echo    %VCURL%
echo    %WV2URL%
echo ============================================================
pause
exit /b 0

REM ---------- subroutines ----------
:GET
REM  %1 = url   %2 = output file
set "U=%~1"
set "O=%~2"
if "%DL%"=="curl" goto GETCURL
REM  PowerShell reads the URL from the environment ($env:U) so cmd never
REM  re-parses the query string.
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri $env:U -OutFile $env:O -UseBasicParsing"
goto GETDONE
:GETCURL
curl.exe -L --fail --retry 2 -o "%O%" "%U%"
:GETDONE
if errorlevel 1 echo    [X] download failed (exit %errorlevel%)
goto :eof

:SHOW
for %%A in ("%~1") do echo    [OK] %%~nxA   %%~zA bytes
goto :eof
