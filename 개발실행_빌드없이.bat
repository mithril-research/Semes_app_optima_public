@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM 빌드 없이 소스에서 바로 실행(반복 확인용, 즉시). 전제: 앱빌드.bat 1회 실행(venv_build 생성).
if not exist "venv_build\Scripts\python.exe" (
  echo [오류] venv_build 가 없습니다. 먼저 앱빌드.bat 를 한 번 실행하세요.
  pause & exit /b 1
)
echo 소스에서 바로 실행합니다(빌드 없음). 코드 바꾸면 창 닫고 다시 실행하면 반영됩니다.
echo (화면 index.html 만 바꿨으면 앱에서 Ctrl+R/F5 새로고침만 해도 반영)
"venv_build\Scripts\python.exe" -m src.app.desktop
pause
