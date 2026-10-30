# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — IO 이상탐지 검사 콘솔 (Windows .exe, onedir)

실행 방식: 웹 대시보드 런처(launcher.py) — exe 가 서버를 켜고 기본 브라우저를 자동으로 연다.

Windows PC 에서:
    pip install -r src/app/requirements-win.txt
    pyinstaller --clean src/app/io_console.spec
결과: dist/IOAnomalyConsole/IOAnomalyConsole.exe  (폴더 통째로 배포)

경로는 spec 파일 위치(SPECPATH) 기준으로 계산 → 어느 폴더에서 빌드해도 동작.
설치된 torch 종류(CPU/CUDA)가 그대로 exe 에 번들된다.
"""
import os
from PyInstaller.utils.hooks import collect_all

# SPECPATH = 이 spec 이 있는 폴더(src/app)
APP_DIR = os.path.abspath(SPECPATH)
PKG_DIR = os.path.dirname(APP_DIR)                       # src/
ROOT = os.path.dirname(PKG_DIR)                          # repo 루트(src 의 부모)

datas, binaries, hiddenimports = [], [], []

# ── 무거운 서드파티 수집 ──
#   ★ torch 제외(≈450MB↓) — 추론은 onnxruntime(경량)로. matplotlib/pandas/onnxruntime 수집.
for pkg in ("matplotlib", "pandas", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# ── 우리 리소스: 정적 UI · 모델(onnx+메타) · 그래프순서 · 한글 폰트 ──
#   모델 파일은 '있으면' 번들(없어도 빌드 성공 — 사용자가 models/ 로 넣을 수 있음).
datas += [(os.path.join(PKG_DIR, "app", "static"), "src/app/static")]
_opt = [
    (os.path.join(PKG_DIR, "artifacts", "tcn_forecast.onnx"),  "src/artifacts"),
    (os.path.join(PKG_DIR, "artifacts", "recon_meta.npz"),     "src/artifacts"),
    (os.path.join(PKG_DIR, "artifacts", "search_best.json"),   "src/artifacts"),
    (os.path.join(PKG_DIR, "artifacts", "ok_accum_bands.npz"), "src/artifacts"),
    (os.path.join(PKG_DIR, "sensors_order.json"),              "src"),
    (os.path.join(PKG_DIR, "assets", "NanumGothic.ttf"),       "src/assets"),
]
datas += [(src, dst) for src, dst in _opt if os.path.exists(src)]

# ── 동적 import(정적 분석이 못 잡는 것) 강제 포함 ──
#   ★ torch 계열(model_recon_tcn/model_tcn)은 제외 → onnx_backend 만 포함
hiddenimports += [
    "src", "src.app.server", "src.onnx_backend_forecast",
    "src.data", "src.recon_explain", "src.sensor_graph",
    "onnxruntime",
    "uvicorn", "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on",
    "fastapi", "starlette", "pydantic",
]

ICON = os.path.join(PKG_DIR, "assets", "app.ico")       # 없으면 아래에서 무시

a = Analysis(
    [os.path.join(PKG_DIR, "app", "launcher.py")],       # 웹 대시보드 런처
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["tkinter", "PyQt5", "PySide2", "PySide6", "notebook", "IPython",
              # ★ torch 계열 완전 배제(용량↓) — 추론은 onnxruntime
              "torch", "torchvision", "torchaudio",
              "src.model_tcn_forecast"],
    noarchive=False,
)
pyz = PYZ(a.pure)

# ── 배포 형태 선택 ──
#   ONEFILE=1(기본): 단일 IOAnomalyConsole.exe 하나만 생성(폴더 잡동사니·소스 없음).
#                    고객 전달물 = exe + (data\, models\ 는 실행 시 옆에 생성).
#   ONEFILE=0      : onedir(exe + _internal\ 폴더). 첫 실행이 빠름(대량 배포·설치본용).
ONEFILE = os.environ.get("ONEFILE", "1") != "0"

if ONEFILE:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name="IOAnomalyConsole",
        console=True,                                    # 서버 로그 창(닫으면 종료)
        icon=(ICON if os.path.exists(ICON) else None),
        upx=False, runtime_tmpdir=None,
    )
else:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="IOAnomalyConsole",
        console=True,
        icon=(ICON if os.path.exists(ICON) else None),
    )
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
                   name="IOAnomalyConsole")
