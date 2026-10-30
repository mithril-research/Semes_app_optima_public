@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYCMD="
py -3.8 --version >nul 2>&1 && set "PYCMD=py -3.8"
if not defined PYCMD (
  python -c "import sys;sys.exit(0 if sys.version_info[:2]==(3,8) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
  echo Python 3.8 x64 is required.
  pause
  exit /b 1
)
%PYCMD% "%~dp0src\app\win7_build_prepared.py" build %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
