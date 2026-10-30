"""IO 이상탐지 검사 콘솔 — 웹 대시보드 실행 런처(.exe 진입점).

exe 를 실행하면:
  1) 로컬 포트(기본 8000, 사용중이면 빈 포트)에 FastAPI 서버 기동
  2) 준비되면 기본 브라우저로 대시보드 자동 오픈
  3) 이 콘솔 창을 닫거나 Ctrl+C 하면 서버 종료

개발 실행:  python -m src.app.launcher
exe 빌드:   app/io_console.spec (Windows, PyInstaller) — build_win.bat 참고
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# 개발 실행(-m 아님) 대비 패키지 경로 보정 (frozen exe 에선 불필요)
if not getattr(sys, "frozen", False) and __package__ in (None, "", "app"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

APP_TITLE = "IO 이상탐지 검사 콘솔"

from src.app.ui_runtime import fresh_ui_url, running_app


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _pick_port(preferred: int = 8000) -> int:
    """선호 포트(8000)가 비어있으면 그걸, 아니면 OS 가 주는 빈 포트."""
    if _port_free(preferred):
        return preferred
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _running_app(port: int = 8000, timeout: float = 1.5, exit_wait: float = 330.0):
    return running_app(port, timeout, exit_wait)


def _wait_ready(url: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:                              # noqa: BLE001
            time.sleep(0.3)
    return False


# ★ 기본 브라우저에 맡기면 안 되는 이유(WES7 / Windows 7)
#   그 PC 의 기본 브라우저는 대개 Internet Explorer 인데, 이 대시보드는 fetch·화살표 함수·
#   템플릿 리터럴을 쓰기 때문에 IE 에서는 빈 화면이 된다. 그래서 크로미움/파이어폭스를
#   직접 찾아서 연다. (윈7 에서 도는 마지막 버전 = Chrome 109 / Firefox 115 ESR)
_MODERN_BROWSERS = [
    ("Chrome",  [r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
                 r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
                 r"%LocalAppData%\Google\Chrome\Application\chrome.exe"]),
    ("Edge",    [r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
                 r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"]),
    ("Firefox", [r"%ProgramFiles%\Mozilla Firefox\firefox.exe",
                 r"%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe"]),
    ("Whale",   [r"%ProgramFiles%\Naver\Naver Whale\Application\whale.exe",
                 r"%ProgramFiles(x86)%\Naver\Naver Whale\Application\whale.exe"]),
]


def _find_modern_browser():
    """설치된 최신 브라우저 경로 찾기. 없으면 None."""
    if os.name != "nt":
        return None
    for name, cands in _MODERN_BROWSERS:
        for c in cands:
            p = os.path.expandvars(c)
            if "%" not in p and os.path.exists(p):
                return name, p
    return None


def _open_browser_when_ready(base: str):
    """서버가 응답하면 브라우저를 연다(백그라운드 스레드).
    IE 로 열려 빈 화면이 되는 것을 막기 위해 최신 브라우저를 우선 사용한다."""
    if not _wait_ready(base + "/api/status"):
        print("[launcher] 서버가 시간 내 준비되지 않았습니다.", file=sys.stderr)
        return
    url = fresh_ui_url(base)
    hit = _find_modern_browser()
    if hit:
        name, path = hit
        print(f"[launcher] 준비 완료 → {name} 로 열기: {url}")
        try:
            import subprocess
            subprocess.Popen([path, url], close_fds=True)
            return
        except Exception as e:                         # noqa: BLE001
            print(f"[launcher] {name} 실행 실패({e}) → 기본 브라우저로 시도")
    else:
        print("[launcher] ※ Chrome/Firefox 등 최신 브라우저를 찾지 못했습니다.\n"
              "   Internet Explorer 로 열리면 화면이 비어 보입니다.\n"
              "   Chrome 109 또는 Firefox 115 ESR 을 설치한 뒤 다시 실행하세요.", flush=True)
    print(f"[launcher] 준비 완료 → 브라우저 열기: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:                             # noqa: BLE001
        print(f"[launcher] 브라우저 자동 열기 실패: {e}\n  직접 접속: {url}")


def _boot_file(name):
    """exe 옆에 쓰고, 권한이 없으면 임시 폴더로 물러난다."""
    try:
        base = (Path(sys.executable).parent if getattr(sys, "frozen", False)
                else Path(__file__).resolve().parents[1])
        f = base / name
        f.parent.mkdir(parents=True, exist_ok=True)
        return f
    except Exception:                                  # noqa: BLE001
        import tempfile
        return Path(tempfile.gettempdir()) / name


def _boot_failed(e):
    """창 없이(콘솔 없이) 실행할 때 시작 실패를 눈에 보이게 남긴다.

    ★ console=False 로 빌드하면 stdout/stderr 가 없어서, 예외가 나도 화면에
      아무것도 안 뜨고 그냥 사라진다("실행해도 반응 없음"). 파일 + 메시지 상자로 알린다.
    """
    import traceback
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    path = _boot_file("앱_시작오류.txt")
    try:
        # 한국어판 메모장에서 한글이 깨지지 않도록 BOM 포함
        path.write_text(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 앱 시작 실패\n"
                        f"실행파일: {sys.executable}\n\n{tb}\n", encoding="utf-8-sig")
    except Exception:                                  # noqa: BLE001
        pass
    try:
        import ctypes
        # 0x10=오류아이콘 · 0x40000=항상 위 · 0x10000=포그라운드로.
        #   ★ 앞 두 개가 없으면 이 창이 다른 창 뒤에 숨어, 사용자에겐 '눌러도 아무 반응 없음'
        #     으로 보인다(실제로 겪은 증상이다). 시작 실패는 반드시 눈에 띄어야 한다.
        ctypes.windll.user32.MessageBoxW(
            0, f"앱을 시작하지 못했습니다.\n\n{type(e).__name__}: {e}\n\n자세한 내용:\n{path}",
            APP_TITLE, 0x10 | 0x40000 | 0x10000)
    except Exception:                                  # noqa: BLE001
        pass


def _selftest():
    """--selftest 로 실행하면 '어디서 죽는지'를 단계별로 찍는다.

    앱이 메시지 없이 사라질 때, 어느 import 에서 죽는지 눈으로 확인하기 위한 모드.
    각 단계를 출력하고 바로 flush 하므로, 마지막에 찍힌 줄이 죽은 지점이다.
    """
    def step(msg):
        print(msg, flush=True)
        try:
            sys.stdout.flush()
        except Exception:                              # noqa: BLE001
            pass
    step("=" * 60)
    step("  자체 점검 (--selftest)")
    step("=" * 60)
    step(f"  파이썬   : {sys.version}")
    step(f"  실행파일 : {sys.executable}")
    step(f"  IO_BACKEND={os.environ.get('IO_BACKEND')}  "
         f"KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK')}")
    # 번들 안에 torch DLL 이 실제로 있는지 먼저 보여준다(0xC06D007E 진단용)
    try:
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
        for sub in ("", os.path.join("torch", "lib"), "_internal"):
            d = os.path.join(base, sub) if sub else base
            if os.path.isdir(d):
                dlls = [f for f in os.listdir(d) if f.lower().endswith(".dll")]
                key = [f for f in dlls if f.lower() in ("torch_cpu.dll", "c10.dll", "fbgemm.dll")]
                step(f"  [DLL] {d}  →  dll {len(dlls)}개" + (f" · 핵심 {key}" if key else ""))
    except Exception as e:                             # noqa: BLE001
        step(f"  [DLL] 확인 실패: {e}")
    step("\n[1] numpy import …")
    import numpy; step(f"    OK  numpy {numpy.__version__}")
    step("[2] torch import …  (여기서 멈추면 torch DLL 문제입니다)")
    try:
        import torch
        step(f"    OK  torch {torch.__version__}  스레드 {torch.get_num_threads()}")
    except Exception as e:                             # noqa: BLE001
        step(f"    실패: {type(e).__name__}: {e}")
        return
    step("[3] 작은 텐서 연산 …")
    x = torch.randn(8, 8); step(f"    OK  합계 {float(x.sum()):.3f}")
    step("[4] 모델 폴더 로드 …")
    try:
        from src.app.server import ENGINE
        ENGINE.load()
        step(f"    OK  backend={ENGINE.backend} ready={ENGINE.ready} 폴더={ENGINE.art}")
        if ENGINE.error:
            step(f"    error={ENGINE.error}")
    except Exception as e:                             # noqa: BLE001
        step(f"    실패: {type(e).__name__}: {e}")
    step("\n점검 끝. 마지막에 찍힌 줄이 문제 지점입니다.")


def main():
    if "--selftest" in sys.argv:
        _selftest()
        return
    # 콘솔 없는(windowed) 빌드에선 stdout/stderr 가 None → 서버의 파일 로거를 먼저 켠다.
    try:
        from src.app.server import _setup_logging
        _setup_logging()
    except Exception:                                  # noqa: BLE001
        pass
    # 즉시 안내(사용자가 '멈춘 것'으로 오해하지 않도록) — flush 로 바로 출력
    print("=" * 60, flush=True)
    print(f"  {APP_TITLE}", flush=True)
    print("  프로그램을 준비하고 있습니다. 잠시만 기다려 주세요…", flush=True)
    print("  (최초 실행은 1~2분 걸릴 수 있습니다. 창을 닫지 마세요.)", flush=True)
    print("=" * 60, flush=True)

    # ★ 이미 떠 있는 앱이 있으면 새로 띄우지 않는다(프로세스가 쌓이지 않게).
    #   화면만 다시 열어 주고 이 프로세스는 바로 끝낸다.
    _pref = int(os.environ.get("PORT") or 8000)
    _hit = _running_app(_pref)
    if _hit:
        _base, _pid = _hit
        print(f"  이미 실행 중인 프로그램이 있습니다(프로세스 {_pid}). 화면만 다시 엽니다.", flush=True)
        print(f"  주소: {_base}", flush=True)
        try:
            _open_browser_when_ready(_base)
        except Exception:                              # noqa: BLE001
            pass
        return

    port = _pick_port(_pref)
    base = f"http://127.0.0.1:{port}"

    # 브라우저 열기는 서버가 뜬 뒤에 → 별도 스레드로 대기
    threading.Thread(target=_open_browser_when_ready, args=(base,),
                     daemon=True).start()

    # 무거운 import(torch 등) — 여기서 수 초~수십 초 걸릴 수 있음
    print("  엔진 로딩 중… (torch/모델)", flush=True)
    import uvicorn
    from src.app.server import app, start_exit_watch
    print(f"  준비 완료. 대시보드 주소: {fresh_ui_url(base)}", flush=True)
    print("  브라우저가 자동으로 열립니다. 안 열리면 위 주소를 직접 입력하세요.", flush=True)
    print("  ※ 브라우저 창을 닫으면 프로그램도 함께 종료됩니다.", flush=True)
    # ★ 브라우저 창을 닫아도 프로세스가 남던 문제 — 화면 연결이 모두 끊기면 종료한다.
    #   검사·리포트·학습·LOT 채우기 중에는 끝날 때까지 기다린다.
    #   화면 없이 자동감시만 돌리려면 IO_KEEP_ALIVE=1 로 실행한다.
    start_exit_watch()
    uvicorn.run(app, host="127.0.0.1", port=port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # ★ 얼린 exe 에서 리포트 병렬 워커가 앱을 재실행하지 않게(필수)
    try:
        main()
    except SystemExit:
        raise
    except BaseException as _e:            # noqa: BLE001 — 창 없는 빌드에서 조용히 죽지 않게
        _boot_failed(_e)
        sys.exit(1)
