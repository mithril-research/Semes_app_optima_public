@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM 빠른(증분) 빌드: venv 재사용 + PyInstaller 캐시 재사용(--clean 없음). 5분 -> 30~60초.
if not exist "venv_build\Scripts\python.exe" (
  echo [오류] venv_build 가 없습니다. 먼저 앱빌드.bat 를 한 번 실행하세요.
  pause & exit /b 1
)
if exist "챔버이상검사시스템" rd /s /q "챔버이상검사시스템"
echo 증분 빌드 중(venv 재사용, 클린 없음)...
"venv_build\Scripts\python.exe" -m PyInstaller --noconfirm --distpath "%CD%" --workpath "%CD%\build" "src\app\io_console_webview.spec"
if errorlevel 1 ( echo 빌드 실패 & pause & exit /b 1 )
echo 완료: 챔버이상검사시스템\챔버이상검사.exe
pause
