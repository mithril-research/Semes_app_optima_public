# -*- mode: python ; coding: utf-8 -*-
r"""PyInstaller spec — 챔버 이상 검사 시스템 / **WES7 · Windows 7 SP1 전용** (onedir)

Win10 용(io_console_webview.spec)과 **별도 파일로 관리한다.** 섞으면 안 되는 이유:

  · 화면 방식이 두 가지다. 환경변수 IO_WES7_UI 로 고른다(기본 webview).
      IO_WES7_UI=webview  네이티브 창. **WebView2 고정버전 런타임 109.0.1518.x 필요**
                          (Edge 109 = Windows 7 마지막 버전. Evergreen 설치기는 Win7 에
                           설치되지 않는다). exe 옆 webview2_runtime\ 폴더에 넣으면
                           desktop.py 가 찾아서 쓰고, pywebview 의 레지스트리 감지도 우회한다.
      IO_WES7_UI=browser  브라우저 모드(launcher.py). 런타임을 넣을 수 없는 현장용.
                          제어기에 Chrome 109 또는 Firefox 115 ESR 필요.

  · 파이썬이 3.8 이다(3.9+ 는 Windows 8.1 이상 요구 — CPython 공식 문서).
    따라서 패키지도 requirements-win7.txt 의 cp38 핀을 쓴다.

  · onedir 고정. onefile 은 실행할 때마다 임시폴더에 수백 MB 를 풀어야 하는데,
    WES7 은 쓰기 필터(EWF/FBWF)나 소용량 SSD 를 쓰는 경우가 많아 위험하다.

빌드:  앱빌드_win7.bat   (Python 3.8.10 64-bit 필요)
결과:  dist/ChamberInspect_WIN7/ChamberInspect.exe
"""
import os

from PyInstaller.utils.hooks import collect_all

APP_DIR = os.path.abspath(SPECPATH)                      # src/app
PKG_DIR = os.path.dirname(APP_DIR)                       # src/
PKG_NAME = os.path.basename(PKG_DIR)                     # 저장소에선 src, 배포본에선 src
ROOT = os.path.dirname(PKG_DIR)

datas, binaries, hiddenimports = [], [], []

UI_MODE = os.environ.get("IO_WES7_UI", "webview").strip().lower()
WEBVIEW = UI_MODE != "browser"
print(f"[spec] WES7 화면 방식: {'네이티브 창(WebView2 109)' if WEBVIEW else '브라우저 모드'}")

# ── 추론 백엔드 선택 (IO_WES7_BACKEND=torch|onnx, 기본 torch) ──
#   onnx  : 가볍다(수십 MB). onnxruntime 1.14.1 은 Win8+ API 정적 임포트가 없어 Win7 에서 로드된다.
#           모델은 IR 8 이어야 한다(모델_IR낮추기_WIN7.bat 로 변환).
#   torch : 무겁다(torch_cpu.dll 만 200MB+). ★ 반드시 2.3.1 이하를 쓸 것 —
#           2.4.1 의 torch_cpu.dll 은 kernel32!CreateFile2(Windows 8+)를 정적 임포트해서
#           Windows 7 에서는 import torch 자체가 실패한다(임포트 테이블 실측 확인).
#           2.3.1 까지는 Win8+ API 정적 임포트가 없다.
# ★ 기본은 onnxruntime 1.14.1 이다(사용자 결정).
#   · Win7 에서 로드되는 마지막 버전이고, torch 판(1GB)보다 수십 배 가볍다.
#   · 모델은 IR 8 이어야 한다 → 모델_IR낮추기_WIN7.bat 로 한 번 변환하면 된다.
#   torch 판이 필요하면 set IO_WES7_BACKEND=torch.
BACKEND = os.environ.get("IO_WES7_BACKEND", "onnx").strip().lower()
TORCH = BACKEND == "torch"
print(f"[spec] WES7 추론 백엔드: {'torch(.pt)' if TORCH else 'onnxruntime(.onnx)'}")

# ── 서드파티 수집 ──
_pkgs = ["matplotlib", "pandas"] + (["torch"] if TORCH else ["onnxruntime"])
if WEBVIEW:
    _pkgs += ["webview", "clr_loader"]                   # pywebview + .NET 로더
for pkg in _pkgs:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# ── torch DLL 을 원래 폴더 구조(torch/lib)로도 담는다 ──
#   PyInstaller 가 DLL 을 루트로 평탄화하면 torch 의 로더가 찾지 못해
#   '지연 로드 DLL 없음'(0xC06D007E)으로 프로세스가 즉사한다. 양쪽에 둔다.
if TORCH:
    try:
        import torch as _t
        _tlib = os.path.join(os.path.dirname(_t.__file__), "lib")
        _n = 0
        if os.path.isdir(_tlib):
            for _f in os.listdir(_tlib):
                if _f.lower().endswith((".dll", ".pyd")):
                    binaries.append((os.path.join(_tlib, _f), os.path.join("torch", "lib")))
                    _n += 1
        print(f"[spec] torch/lib DLL {_n}개를 torch/lib 구조로 추가")
    except Exception as _e:
        print(f"[spec] torch/lib 수집 실패(무시): {_e}")

# ── WebView2 고정버전 런타임(있으면 통째로 번들) ──
#   Evergreen 을 설치할 수 없는 Win7 에서는 이 폴더가 유일한 방법이다.
if WEBVIEW:
    _rt = os.path.join(ROOT, "webview2_runtime")
    if not os.path.isdir(_rt):
        _rt = os.path.join(PKG_DIR, "webview2_runtime")
    if os.path.isdir(_rt):
        _n = 0
        for _dir, _, _files in os.walk(_rt):
            _rel = os.path.relpath(_dir, os.path.dirname(_rt))
            for _f in _files:
                datas.append((os.path.join(_dir, _f), _rel))
                _n += 1
        print(f"[spec] WebView2 고정버전 런타임 {_n}개 파일 번들: {_rt}")
    else:
        print("[spec] ※ webview2_runtime 폴더가 없습니다 — 제어기에 고정버전 런타임을 "
              "직접 넣어야 네이티브 창이 뜹니다(109.0.1518.x).")

# ── Universal C Runtime(UCRT)을 앱 폴더에 함께 담는다 ──
#   ★ 파이썬 3.8 은 UCRT 를 요구한다. Windows 10 이상은 OS 에 들어 있지만
#     Windows 7 에는 없어서 실행 시 "api-ms-win-crt-runtime-l1-1-0.dll 이 없습니다"
#     가 뜬다(실제로 겪은 오류). PyInstaller 는 이 DLL 들을 'OS 구성요소'로 보고
#     번들에서 제외하기 때문에 직접 넣어야 한다.
#   Windows SDK 가 깔린 빌드 PC 라면 redist 폴더에서 가져온다(앱 로컬 배치는 정식 지원 방식).
#   없으면 경고만 하고 넘어간다 - 그 경우 제어기에 KB2999226 을 설치해야 한다.
_ucrt_dirs = [
    r"C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\x64",
    os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                 "Windows Kits", "10", "Redist", "ucrt", "DLLs", "x64"),
]
_ucrt_n = 0
for _d in _ucrt_dirs:
    if os.path.isdir(_d):
        for _f in os.listdir(_d):
            if _f.lower().endswith(".dll"):
                binaries.append((os.path.join(_d, _f), "."))
                _ucrt_n += 1
        break
if _ucrt_n:
    print(f"[spec] UCRT {_ucrt_n}개를 앱 폴더에 포함(윈도우7 에서 KB2999226 없이도 실행)")
else:
    print("[spec] ※ UCRT redist 를 찾지 못했습니다. 윈도우7 제어기에는 KB2999226 을 "
          "설치해야 합니다(윈도우7_전제조건_설치.bat 이 처리합니다).")

# ── 우리 리소스 ──
datas += [(os.path.join(PKG_DIR, "app", "static"), f"{PKG_NAME}/app/static")]
_opt = [
    # torch 빌드에는 .onnx 를 넣지 않는다(있으면 앱이 그쪽을 고르려다 죽는다)
    (os.path.join(PKG_DIR, "artifacts", "tcn_forecast.pt"),    f"{PKG_NAME}/artifacts"),
] + ([] if TORCH else [
    (os.path.join(PKG_DIR, "artifacts", "tcn_forecast.onnx"),  f"{PKG_NAME}/artifacts"),
    (os.path.join(PKG_DIR, "artifacts", "recon_meta.npz"),     f"{PKG_NAME}/artifacts"),
]) + [
    (os.path.join(PKG_DIR, "artifacts", "search_best.json"),   f"{PKG_NAME}/artifacts"),
    (os.path.join(PKG_DIR, "artifacts", "ok_accum_bands.npz"), f"{PKG_NAME}/artifacts"),
    (os.path.join(PKG_DIR, "sensors_order.json"),              PKG_NAME),
    (os.path.join(PKG_DIR, "manual_recipes.json"),             PKG_NAME),
    (os.path.join(PKG_DIR, "assets", "NanumGothic.ttf"),       f"{PKG_NAME}/assets"),
]
datas += [(src, dst) for src, dst in _opt if os.path.exists(src)]

if WEBVIEW:
    hiddenimports += ["webview", "webview.platforms.winforms", "clr_loader", "clr", "cffi"]

hiddenimports += [
    PKG_NAME, f"{PKG_NAME}.app.server", f"{PKG_NAME}.app.cpu_limit",
    f"{PKG_NAME}.onnx_backend_forecast", f"{PKG_NAME}.data",
    f"{PKG_NAME}.recon_explain", f"{PKG_NAME}.sensor_graph",
] + ([f"{PKG_NAME}.model_tcn_forecast", "torch"] if TORCH else ["onnxruntime"]) + [
    "uvicorn", "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on",
    "fastapi", "starlette", "pydantic",
]

ICON = os.path.join(PKG_DIR, "assets", "app.ico")

ENTRY = "desktop.py" if WEBVIEW else "launcher.py"   # 네이티브 창 / 브라우저 모드

a = Analysis(
    [os.path.join(PKG_DIR, "app", ENTRY)],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=([os.path.join(PKG_DIR, "app", "rthook_crashlog.py")]
                   + ([os.path.join(PKG_DIR, "app", "rthook_torch_dll.py"),
                       os.path.join(PKG_DIR, "app", "rthook_backend_torch.py")]
                      if TORCH else [])),
    excludes=(["tkinter", "PyQt5", "PySide2", "PySide6", "notebook", "IPython"]
              + ([] if TORCH else ["torch", "torchvision", "torchaudio",
                                   f"{PKG_NAME}.model_tcn_forecast"])
              + (["torchvision", "torchaudio"] if TORCH else [])
              + ([] if WEBVIEW else ["webview", "pywebview", "clr", "pythonnet", "clr_loader"])),
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,                               # onedir 고정(위 주석 참고)
    name="ChamberInspect",
    # ★ 기본은 콘솔 없이(창 하나만). 사용자 요청.
    #   콘솔을 없애면 오류가 화면에 안 남으므로, 그 대신 아래 세 가지가 받쳐 준다.
    #     · 시작 실패 → 앱_시작오류.txt + '항상 위' 알림창
    #     · 실행 중 출력 → 로그\app_<날짜>.log (launcher 가 시작하자마자 켠다)
    #     · 라이브러리 크래시 → 앱_크래시.txt (런타임 훅)
    #   현장에서 콘솔을 봐야 하면 빌드 전에 set IO_WES7_CONSOLE=1 하면 콘솔 빌드가 나온다.
    console=(os.environ.get("IO_WES7_CONSOLE", "0").strip().lower() in ("1", "true", "y", "yes")),
    icon=(ICON if os.path.exists(ICON) else None),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="ChamberInspect_WIN7")   # ★ 배치(앱빌드_WIN7.bat)가 찾는 이름과 반드시 일치
