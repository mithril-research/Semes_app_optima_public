# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — 모델 변환기(.pt → ONNX) 별도 실행 파일.

메인 앱(웨이퍼품질검사.exe)은 torch 를 뺀 경량 빌드다. 이 변환기 exe 만
torch/onnx/onnxscript 를 포함해 .pt 를 .onnx(+메타) 로 바꾸고 원본 .pt 를 삭제한다.
제어기에 메인 앱과 '함께' 두고, 새 .pt 모델이 오면 이 exe 만 실행한다.

빌드(Windows, torch 있는 내 PC):
    pip install -r src/app/../모델변환_requirements.txt
    pyinstaller --clean src/app/모델변환.spec
결과: 모델변환/모델변환.exe  (onedir · console)
"""
import os, sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

APP_DIR = os.path.abspath(SPECPATH)
PKG_DIR = os.path.dirname(APP_DIR)                       # src/ (배포본은 src/)
ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, ROOT)

PKG_NAME = os.path.basename(PKG_DIR)                     # 'src' 또는 'src'

datas, binaries, hiddenimports = [], [], []

# 변환에 필요한 무거운 패키지 전체 수집(torch 포함 — 변환기 exe 에만).
for pkg in ("torch", "onnx", "onnxscript", "numpy", "pandas"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
        print(f"[spec] collect_all({pkg}) OK")
    except Exception as _e:                              # noqa: BLE001
        print(f"[spec] collect_all({pkg}) 스킵: {_e}")

# 우리 패키지의 변환 관련 모듈(torch 모델 포함 — 메인 앱과 달리 여기선 필요).
try:
    for _m in collect_submodules(PKG_NAME):
        hiddenimports.append(_m)
except Exception as _e:                                  # noqa: BLE001
    print(f"[spec] collect_submodules({PKG_NAME}) 스킵: {_e}")

hiddenimports += [
    PKG_NAME,
    f"{PKG_NAME}.export_onnx_forecast",
    f"{PKG_NAME}.onnx_backend_forecast",
    f"{PKG_NAME}.model_tcn_forecast",
    f"{PKG_NAME}.tcn",
    f"{PKG_NAME}.config",
    f"{PKG_NAME}.data",
]

a = Analysis(
    [os.path.join(PKG_DIR, "app", "convert_main.py")],   # 변환기 진입점
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["tkinter", "PyQt5", "PySide2", "PySide6", "notebook", "IPython",
              "webview", "pythonnet", "clr_loader"],     # 변환기엔 UI/웹뷰 불필요
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="모델변환",             # 실행 파일: 모델변환.exe
    console=True,                # 변환 진행상황을 콘솔로 보여줌(변환 전용 도구)
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="모델변환")             # 배포 폴더: 모델변환/모델변환.exe
