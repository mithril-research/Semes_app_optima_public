@echo off
REM ============================================================
REM  IO Anomaly - Prerequisites Setup  (run ONCE per PC, as Admin)
REM  Installs what the LEAN app needs so the exe runs standalone:
REM    1) Microsoft Visual C++ 2015-2022 Redistributable (x64)
REM    2) Microsoft Edge WebView2 Runtime (Evergreen)
REM    3) (optional) .NET Framework 4.8, only if you drop its installer here
REM
REM  PUT THESE MICROSOFT INSTALLERS NEXT TO THIS FILE:
REM    vc_redist.x64.exe
REM        https://aka.ms/vs/17/release/vc_redist.x64.exe
REM    MicrosoftEdgeWebView2RuntimeInstallerX64.exe   (Evergreen Standalone)
REM        https://developer.microsoft.com/microsoft-edge/webview2/  (Download section)
REM    ndp48-x86-x64-allos-enu.exe   (OPTIONAL - only if a PC lacks .NET 4.7.2+)
REM        https://dotnet.microsoft.com/download/dotnet-framework/net48
REM
REM  ASCII only. goto-based (no risky if-parens blocks).
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo  IO Anomaly - Prerequisites Setup
echo  ------------------------------------------------------------

REM --- must be Administrator ---
net session >nul 2>&1
if errorlevel 1 goto NOADMIN

REM --- 1) Visual C++ 2015-2022 Redistributable (x64) ---
if not exist "vc_redist.x64.exe" goto NOVC
echo  [1/3] Visual C++ 2015-2022 Redistributable (x64)...
vc_redist.x64.exe /install /quiet /norestart
echo        installer returned %errorlevel% (0 or 3010 = OK, 1638 = newer already there)

REM --- 2) WebView2 Runtime (Evergreen) ---
if not exist "MicrosoftEdgeWebView2RuntimeInstallerX64.exe" goto NOWV2
echo  [2/3] Microsoft Edge WebView2 Runtime...
MicrosoftEdgeWebView2RuntimeInstallerX64.exe /silent /install
echo        installer returned %errorlevel%

REM --- 3) OPTIONAL .NET 4.8 (only if its installer was placed here) ---
if not exist "ndp48-x86-x64-allos-enu.exe" goto SKIPNET
echo  [3/3] .NET Framework 4.8...
ndp48-x86-x64-allos-enu.exe /q /norestart
echo        installer returned %errorlevel% (may require a reboot)
goto VERIFY
:SKIPNET
echo  [3/3] .NET 4.8 installer not present - skipped (most Windows already have 4.7.2+).

:VERIFY
echo.
echo  ------------------------------------------------------------
echo  Verifying...
set "WV2OK="
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1 && set "WV2OK=1"
if not defined WV2OK reg query "HKLM\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1 && set "WV2OK=1"
if not defined WV2OK reg query "HKCU\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1 && set "WV2OK=1"

set "NETREL="
for /f "tokens=3" %%v in ('reg query "HKLM\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full" /v Release 2^>nul ^| find "Release"') do set "NETREL=%%v"

if defined WV2OK echo   [OK] WebView2 Runtime registered.
if not defined WV2OK echo   [X ] WebView2 NOT detected - check the installer output above.
if defined NETREL echo   [OK] .NET Framework v4 present. Release=%NETREL%  need 461810+ for 4.7.2
if not defined NETREL echo   [X ] .NET Framework v4 not found - install .NET 4.8.

echo.
echo ============================================================
if defined WV2OK goto DONEOK
echo  RESULT: WebView2 missing. The app will not render until it is installed.
goto END
:DONEOK
echo  RESULT: Prerequisites OK. You can now run the IO Anomaly app exe.
:END
echo ============================================================
pause
exit /b 0

:NOADMIN
echo  [X] Please run this file as Administrator.
echo      Right-click - "Run as administrator" (우클릭 - 관리자 권한으로 실행).
pause
exit /b 1

:NOVC
echo  [X] vc_redist.x64.exe not found next to this file.
echo      Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
pause
exit /b 1

:NOWV2
echo  [X] MicrosoftEdgeWebView2RuntimeInstallerX64.exe not found next to this file.
echo      Get the Evergreen Standalone Installer from:
echo      https://developer.microsoft.com/microsoft-edge/webview2/
pause
exit /b 1
