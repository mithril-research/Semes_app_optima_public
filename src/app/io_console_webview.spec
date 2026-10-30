# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — IO 이상탐지 (WebView2 네이티브 창 빌드, LTSC/IE 전용 PC 대응)

시스템 브라우저(IE) 없이 자체 창으로 대시보드 표시. pywebview + WebView2 엔진.
onedir(폴더형) 로 빌드 → 실행 빠름 + WebView2 런타임을 폴더로 함께 배치.

빌드(Windows):
    pip install -r src/app/requirements-win.txt   (pywebview, pythonnet 포함)
    pyinstaller --clean src/app/io_console_webview.spec
결과: 웨이퍼품질검사시스템/웨이퍼품질검사.exe   (build_webview_click.bat 이 --distpath 로 dist 생략)
그 후: WebView2 고정버전 런타임을  웨이퍼품질검사시스템/webview2_runtime/  로 복사(선택)
       (WEBVIEW2_SETUP.md 참고)
"""
import os, sys
from PyInstaller.utils.hooks import (collect_all, collect_data_files,
                                     collect_dynamic_libs, collect_submodules)

APP_DIR = os.path.abspath(SPECPATH)
PKG_DIR = os.path.dirname(APP_DIR)                       # src/
ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, ROOT)                                # src 을 spec 실행 중 import 가능하게

datas, binaries, hiddenimports = [], [], []

# ★★★ src 패키지 전체(하위모듈 포함) 수집 → "No module named src.app.server" 방지.
#    torch 계열 모델 모듈은 제외(onnx 로 추론하므로 불필요).
_IO_EXCL = ("src.model_tcn_forecast", "src.model_tcn",
            "src.hybrid", "src.model_windowed",
            "src.model_multibranch", "src._model_select")
try:
    for _m in collect_submodules("src"):
        if not _m.startswith(_IO_EXCL):
            hiddenimports.append(_m)
    print(f"[spec] src 하위모듈 {len(hiddenimports)}개 수집")
except Exception as _e:                                  # noqa: BLE001
    print(f"[spec] collect_submodules(src) 실패: {_e}")

# ★ torch 제외(≈450MB↓) — 추론은 onnxruntime. matplotlib/pandas/onnxruntime + pywebview 수집.
# ★★ pythonnet/clr_loader/cffi 도 함께 수집 → Python.Runtime.dll(.NET 관리형 어셈블리)·
#    clr_loader 설정·_cffi_backend 가 exe 에 포함된다(collect_all 이 data+binary 모두 처리).
#    이게 빠지면 실행 시 "failed to resolve Python.Runtime.Loader.Initialize" 로 창이 안 열린다.
# ★ matplotlib 은 '의도적으로 제외'한다. exe 화면은 전부 numpy 기반 SVG(개요·산점도·누적·센서·
#   상세·배점)이고 matplotlib 은 PNG 리포트에서만 지연 import 된다. matplotlib 을 번들하면
#   PyInstaller 런타임 훅(pyi_rth_mplconfig)이 matplotlib/hooks-contrib 버전 불일치로 부팅 때
#   "No module named _pyi_rth_mplconfig" 를 반복 발생시킨다 → 아예 빼서 원천 차단(부팅 100% 보장).
#   (PNG 리포트가 필요하면 matplotlib 을 3.9 로 고정해 다시 포함하는 별도 빌드로 대응)
# ★ pandas 는 collect_all 에서 제외한다. collect_all('pandas')는 pandas/tests 까지 1461개
#   (깊은 경로) 를 통째로 수집해, 긴 한글 빌드경로에서 Windows 경로한계로
#   "unable to find pandas-x.dist-info" 복사에러를 냈다. 자동 훅(hook-pandas)이 필요한 것만
#   (테스트 제외, ~57개) 수집하므로 pandas 는 자동 훅에 맡긴다(앱의 import pandas 로 번들됨).
from PyInstaller.utils.hooks import copy_metadata as _copy_meta
for pkg in ("onnxruntime", "webview",
            "pythonnet", "clr_loader", "cffi"):
    try:
        binaries += collect_dynamic_libs(pkg)            # 네이티브 DLL(.pyd/.dll)
    except Exception as _e:
        print(f"[spec] dylibs({pkg}) 스킵: {_e}")
    try:
        datas += collect_data_files(pkg)                 # 데이터 파일
    except Exception as _e:
        print(f"[spec] data({pkg}) 스킵: {_e}")
    try:
        hiddenimports += collect_submodules(pkg)         # 서브모듈
    except Exception as _e:
        print(f"[spec] submods({pkg}) 스킵: {_e}")
    # ★ 메타데이터(dist-info)는 '실패해도 무시' — "unable to add X.dist-info" 로 빌드가 죽지 않게.
    #   (런타임에 대개 불필요. 짧은 영문 경로에서 빌드하면 정상 포함됨.)
    try:
        datas += _copy_meta(pkg)
    except Exception as _e:
        print(f"[spec] metadata({pkg}) 스킵(무해): {_e}")

# ★ CUDA 자체포함 빌드용: onnxruntime-gpu[cuda,cudnn] 로 깔린 nvidia pip 패키지의
#   CUDA/cuDNN DLL 을 exe 에 통째로 포함 → 실행 PC 에 CUDA 설치 없이도 CUDA EP 동작.
#   (DirectML/CPU 빌드는 nvidia 패키지가 없으므로 자동 스킵 — 일반 빌드엔 영향 없음)
try:
    import importlib.util as _ilu, glob as _glob
    _nv = _ilu.find_spec("nvidia")
    if _nv and _nv.submodule_search_locations:
        _nvroot = list(_nv.submodule_search_locations)[0]
        _n = 0
        for _dll in _glob.glob(os.path.join(_nvroot, "**", "*.dll"), recursive=True):
            _rel = os.path.join("nvidia", os.path.relpath(os.path.dirname(_dll), _nvroot))
            binaries.append((_dll, _rel)); _n += 1
        print(f"[spec] nvidia CUDA/cuDNN DLL {_n}개 포함(자체포함 CUDA 빌드): {_nvroot}")
    else:
        print("[spec] nvidia 패키지 없음 → CUDA DLL 미포함(일반 빌드)")
except Exception as _e:                                  # noqa: BLE001
    print(f"[spec] nvidia DLL 수집 스킵: {_e}")

# ★★★ pywebview 의 WebView2 로더/.NET 어셈블리(webview/lib) 확실히 포함.
#     WebView2Loader.dll · Microsoft.Web.WebView2.Core.dll · WinForms.dll 이 빠지면
#     edgechromium 초기화가 실패하고 pywebview 가 조용히 IE(mshtml) 로 폴백한다(현재 증상).
try:
    datas += collect_data_files("webview")              # webview/lib/*.dll, *.json 등
    binaries += collect_dynamic_libs("webview")         # 네이티브 WebView2Loader.dll
except Exception as _e:                                  # noqa: BLE001
    print(f"[spec] webview lib 수집 스킵: {_e}")

# ★★★ pywebview 의 WebView2 .NET 어셈블리를 exe 안 'webview/lib' 에 정확히 배치.
#   edgechromium 은 webview/lib/Microsoft.Web.WebView2.*.dll, WebView2Loader.dll 을
#   상대경로로 로드한다. 이게 제 위치에 없으면 초기화 실패 -> 조용히 IE(mshtml)로 폴백.
try:
    import webview as _wv
    _wvlib = os.path.join(os.path.dirname(os.path.abspath(_wv.__file__)), "lib")
    if os.path.isdir(_wvlib):
        _c = 0
        for _dp, _dn, _fn in os.walk(_wvlib):
            for _f in _fn:
                _full = os.path.join(_dp, _f)
                _rel = os.path.relpath(_dp, _wvlib)
                _dest = os.path.join("webview", "lib", _rel) if _rel != "." else os.path.join("webview", "lib")
                datas.append((_full, _dest)); _c += 1
        print(f"[spec] webview/lib WebView2 어셈블리 {_c}개 강제 포함")
    else:
        print(f"[spec] (경고) webview/lib 폴더 없음: {_wvlib}")
except Exception as _e:                                  # noqa: BLE001
    print(f"[spec] webview/lib 강제포함 스킵: {_e}")

# ★★★ WebView2 '고정버전' 런타임을 앱에 함께 번들(제어기 설치 불필요).
#     빌드 전 환경변수 IO_WV2_RUNTIME 에 runtime 폴더(msedgewebview2.exe 있는 곳) 경로 지정하면
#     dist 안 webview2_runtime\ 로 통째로 담긴다. desktop.py 가 그걸 자동으로 찾아 사용한다.
_wv2 = os.environ.get("IO_WV2_RUNTIME")
if _wv2 and os.path.isdir(_wv2):
    _cnt = 0
    for _root, _dirs, _files in os.walk(_wv2):
        for _f in _files:
            _full = os.path.join(_root, _f)
            _rel = os.path.relpath(_root, _wv2)
            _dest = os.path.join("webview2_runtime", _rel) if _rel != "." else "webview2_runtime"
            datas.append((_full, _dest)); _cnt += 1
    print(f"[spec] WebView2 고정버전 런타임 번들: {_wv2} ({_cnt}개 파일)")
else:
    print("[spec] (참고) IO_WV2_RUNTIME 미지정 — 런타임은 dist 옆 webview2_runtime\\ 에 수동 복사 필요")

datas += [
    (os.path.join(PKG_DIR, "app", "static"),                       "src/app/static"),
    (os.path.join(PKG_DIR, "assets", "NanumGothic.ttf"),           "src/assets"),
    # ※ 기본(내장) 모델은 포함하지 않습니다. 실행 후 exe 옆 '모델\' 폴더에
    #   사용자 모델(.onnx+메타 또는 .pt)을 넣으면 그 모델로 동작합니다.
]

hiddenimports += [
    "src", "src.app", "src.app.server",
    "src.onnx_backend_forecast",
    "src.data", "src.config", "src.recon_explain",
    "src.tcn", "src.sensor_graph",
    "onnxruntime",
    "uvicorn", "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on",
    "fastapi", "starlette", "pydantic",
    # ── pywebview + WebView2(.NET) ──
    "webview", "webview.platforms.edgechromium", "webview.platforms.winforms",
    "clr", "System", "System.Windows.Forms",
    # ── pythonnet 3.x 런타임 로더(.NET 부팅) — 이게 있어야 Python.Runtime 초기화됨 ──
    "pythonnet", "clr_loader", "clr_loader.ffi", "clr_loader.netfx",
    "clr_loader.mono", "clr_loader.hostfxr", "clr_loader.wrappers",
    "cffi", "_cffi_backend",
]

ICON = os.path.join(PKG_DIR, "assets", "app.ico")

a = Analysis(
    [os.path.join(PKG_DIR, "app", "desktop.py")],        # WebView2 네이티브 창 진입점
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    # ★ matplotlib 설정 디렉터리를 우리가 직접 지정(자동 훅 pyi_rth_mplconfig 불일치 대비).
    #   "No module named _pyi_rth_mplconfig" 류 부팅 에러 보강.
    runtime_hooks=[os.path.join(PKG_DIR, "app", "rthook_crashlog.py"),   # ★ 먼저: 시작 실패 흔적
                   os.path.join(PKG_DIR, "app", "rthook_mplconfig.py")],
    excludes=["tkinter", "PyQt5", "PySide2", "PySide6", "notebook", "IPython",
              # ★ matplotlib 재포함(자동 훅) — PNG 리포트/RCA 이미지용.
              #   과거 부팅에러 2원인(3.12↔hooks-contrib 불일치, 한글경로 dist-info)은
              #   requirements 고정(matplotlib>=3.8,<3.10 + contrib 2024.x)과 영문 빌드경로로 해소됨.
              #   함수 내부 지연 import 를 PyInstaller 가 정적으로 감지해 자동 번들(과다수집 없음).
              # ★ torch 계열 완전 배제 → 추론은 onnxruntime
              "torch", "torchvision", "torchaudio",
              "src.model_tcn_forecast", "src.model_tcn",
              "src.hybrid", "src.model_windowed",
              "src.model_multibranch", "src._model_select"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ChamberInspect",       # 실행 파일: ChamberInspect.exe (경로 한글 회피 위해 영문)
    # 콘솔창 숨김(로그는 로그\app_<날짜>.log 파일로 저장).
    # ★ 진단 빌드: 환경변수 IO_BUILD_CONSOLE=1 로 빌드하면 콘솔창이 뜨는 exe 가 나온다.
    #   "실행해도 아무 반응이 없다" 를 눈으로 확인할 때 쓴다(앱빌드_콘솔진단.bat).
    console=(os.environ.get("IO_BUILD_CONSOLE", "") == "1"),
    icon=(ICON if os.path.exists(ICON) else None),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="ChamberInspect")       # 배포 폴더: ChamberInspect/ (영문 — 경로 한글 회피)
