; ============================================================
;  IO 이상탐지 검사 시스템 — Windows 설치 파일 빌드 스크립트
;  (Inno Setup 6 필요: https://jrsoftware.org/isdl.php)
;
;  dist\IOAnomalyConsole\ (build_win.bat 결과)  ->  하나의 setup.exe 로 압축
;  용량이 큰 폴더를 LZMA2 로 압축 → 단일 파일(대개 절반 수준)로 배포/전송 용이.
;
;  빌드:  package_win.bat  (또는  ISCC src\app\installer.iss )
;  결과:  installer_out\IOAnomalyConsole_Setup.exe
; ============================================================

#define AppName "IO 이상탐지 검사 시스템"
#define AppVer "1.0.0"
#define ExeName "IOAnomalyConsole.exe"
#define Publisher "SEMES"

[Setup]
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher={#Publisher}
; ── 관리자 권한 없이(사용자 폴더에) 설치 → IT 승인/관리자 불필요, 앱이 폴더에 쓰기 가능 ──
PrivilegesRequired=lowest
DefaultDirName={localappdata}\IOAnomalyConsole
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\..\installer_out
OutputBaseFilename=IOAnomalyConsole_Setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
; SetupIconFile=..\assets\app.ico     ; 아이콘 있으면 주석 해제

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Files]
; onefile 빌드(기본): 단일 exe 하나만 담는다(소스·부속 파일 없음)
Source: "..\..\dist\IOAnomalyConsole.exe"; DestDir: "{app}"; Flags: ignoreversion
; onedir 빌드(ONEFILE=0)로 만들었다면 위 줄 대신 아래를 사용:
; Source: "..\..\dist\IOAnomalyConsole\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Dirs]
; 검사 대상/모델 교체 폴더를 미리 만들어 둔다(쓰기 가능 위치)
Name: "{app}\data\inbox"
Name: "{app}\models"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#ExeName}"
Name: "{group}\검사 폴더 열기 (data\inbox)"; Filename: "{app}\data\inbox"
Name: "{group}\모델 폴더 열기 (models)"; Filename: "{app}\models"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#ExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 작업:"

[Run]
Filename: "{app}\{#ExeName}"; Description: "지금 실행하기"; Flags: nowait postinstall skipifsilent
