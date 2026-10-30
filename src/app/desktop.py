r"""웨이퍼 품질 검사 시스템 — WebView2 네이티브 창 진입점 (LTSC / IE 전용 PC 대응).

시스템 브라우저(IE)를 쓰지 않고 WebView2(크로미움)로 자체 창을 띄워 대시보드 표시.

기본(권장) 배포: 앱은 린 빌드(런타임 미포함)로 배포하고, 제어기에는
IO_Prerequisites_Setup.bat 로 WebView2 Evergreen 런타임을 정식 설치한다.
설치되면 레지스트리에 등록되어 pywebview 가 자동으로 크로미움을 선택한다.

동작:
  1) (무거운 빌드 예외) exe 옆/번들에 webview2_runtime\ (msedgewebview2.exe 포함)이
     있으면 WEBVIEW2_BROWSER_EXECUTABLE_FOLDER 로 지정. 없으면 시스템 설치본 사용.
  2) uvicorn(FastAPI) 서버를 백그라운드 스레드로 기동
  3) pywebview 로 네이티브 창을 열어 http://127.0.0.1:<port> 표시 (gui=edgechromium)

빌드: io_console_webview.spec (build_webview_click.bat) — 배포_가이드_린빌드.txt 참고.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

if not getattr(sys, "frozen", False) and __package__ in (None, "", "app"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

APP_TITLE = "챔버 이상 검사 시스템"

from src.app.ui_runtime import fresh_ui_url, running_app


def _exe_dir() -> Path:
    return (Path(sys.executable).parent if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parents[1])


def _meipass() -> Path:
    return Path(getattr(sys, "_MEIPASS", _exe_dir()))


def _find_webview2_runtime():
    """WebView2 고정버전 런타임 폴더(msedgewebview2.exe 있는 곳) 탐색.
    exe 옆 / 번들 내부의 webview2_runtime 를 '깊이 상관없이' 재귀로 찾는다."""
    roots = [_exe_dir() / "webview2_runtime", _meipass() / "webview2_runtime",
             _exe_dir(), _meipass()]                      # webview2_runtime 없이 넣은 경우도 대비
    for root in roots:
        try:
            if not root.exists():
                continue
            if (root / "msedgewebview2.exe").exists():
                return root
            for hit in root.rglob("msedgewebview2.exe"):  # 몇 겹 깊어도 찾음
                return hit.parent
        except Exception:                                 # noqa: BLE001
            continue
    return None


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _pick_port(pref: int = 8000) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", pref)) != 0:
            return pref
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


_SERVER_ERR: list = []                                   # 서버 스레드에서 죽은 이유(있으면)


def start_server(port: int):
    import uvicorn
    from src.app.server import app
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port,
                         log_level="warning", access_log=False)
    server = uvicorn.Server(cfg)

    def _run():
        # 스레드에서 난 예외는 아무 데도 안 뜨고 조용히 사라진다 → 반드시 붙잡아 둔다.
        try:
            server.run()
        except BaseException as e:                       # noqa: BLE001
            import traceback
            _SERVER_ERR.append("".join(
                traceback.format_exception(type(e), e, e.__traceback__)))

    threading.Thread(target=_run, name="uvicorn", daemon=True).start()
    return server


def _wait_ready(url: str, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _SERVER_ERR:                                  # 서버가 죽었으면 기다릴 이유가 없다
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:                                # noqa: BLE001
            time.sleep(0.3)
    return False


def _fatal(msg: str):
    print(msg, file=sys.stderr, flush=True)
    try:
        import ctypes
        # 0x10=오류아이콘 · 0x40000=항상 위 · 0x10000=포그라운드로(다른 창 뒤에 숨지 않게)
        ctypes.windll.user32.MessageBoxW(0, msg, APP_TITLE, 0x10 | 0x40000 | 0x10000)
    except Exception:                                    # noqa: BLE001
        pass


# ── 시작 실패 진단 ────────────────────────────────────────────────────
# 콘솔 없는(windowed) exe 는 시작 도중 예외가 나면 창도 메시지도 없이 그냥 사라진다
# ("실행해도 아무 반응 없음"의 정체). 아래 두 장치로 반드시 흔적을 남긴다.
#   1) _step(): 어디까지 갔는지 exe 옆 '앱_시작기록.txt' 에 즉시 기록(파일 로거보다 앞)
#   2) main() 전체를 감싸 예외를 '앱_시작오류.txt' + 메시지 상자로 노출
_BOOT_LOG: list = []


def _boot_file(name: str) -> Path:
    """exe 옆에 쓰고, 권한이 없으면 임시 폴더로 물러난다(로그조차 못 남기는 일 방지)."""
    try:
        f = _exe_dir() / name
        f.parent.mkdir(parents=True, exist_ok=True)
        return f
    except Exception:                                    # noqa: BLE001
        import tempfile
        return Path(tempfile.gettempdir()) / name


def _step(msg: str):
    """시작 단계 기록 — 파일에 바로 flush 해서 강제 종료돼도 남게 한다."""
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    _BOOT_LOG.append(line)
    try:
        with open(_boot_file("앱_시작기록.txt"), "a", encoding="utf-8-sig") as fp:
            fp.write(line + "\n")
    except Exception:                                    # noqa: BLE001
        pass
    try:
        print(line, flush=True)
    except Exception:                                    # noqa: BLE001
        pass


def _boot_failed(e: BaseException):
    """시작 실패를 사용자에게 '보이게' 만든다(파일 + 메시지 상자)."""
    import traceback
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    path = _boot_file("앱_시작오류.txt")
    body = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 앱 시작 실패\n"
            f"실행파일: {sys.executable}\n"
            f"폴더    : {_exe_dir()}\n"
            f"파이썬  : {sys.version}\n\n"
            f"--- 진행 단계 ---\n" + "\n".join(_BOOT_LOG) +
            f"\n\n--- 오류 내용 ---\n{tb}\n")
    try:
        # 메모장에서 한글이 깨지지 않도록 BOM 포함(utf-8-sig)
        path.write_text(body, encoding="utf-8-sig")
    except Exception:                                    # noqa: BLE001
        pass
    last = str(e) or e.__class__.__name__
    _fatal(f"앱을 시작하지 못했습니다.\n\n{e.__class__.__name__}: {last}\n\n"
           f"자세한 내용은 아래 파일을 열어 확인하거나 그대로 보내 주세요.\n{path}")


def _force_chromium_if_bundled(webview, runtime_dir):
    r"""고정버전 WebView2 런타임을 번들했을 때 pywebview 가 IE 로 떨어지는 것을 막는다.

    ★ 원인(pywebview 4.4.1 winforms.py `_is_chromium()` 실측):
        레지스트리 SOFTWARE\...\EdgeUpdate\Clients\{F3017226-...}\pv 의 존재로만
        WebView2 를 감지한다. 이 키는 런타임을 '설치'해야 생기므로, 폴더째 복사해 쓰는
        고정버전 런타임은 감지되지 않아 무조건 IE(mshtml) 로 폴백한다.
        WEBVIEW2_BROWSER_EXECUTABLE_FOLDER 를 지정해도 소용없다 — 판단이 먼저 끝난다.

    ★ 왜 WES7 에서 이 경로가 필요한가:
        Microsoft 는 Edge 109(2023-01)로 Windows 7 지원을 끝냈다. 지금 배포되는
        Evergreen 설치기는 Win7 에 설치되지 않으므로 레지스트리 등록 자체가 불가능하다.
        따라서 Win7/WES7 에서 네이티브 창을 쓰려면 '고정버전 109.0.1518.x + 이 우회'가
        유일한 방법이다.

    번들 런타임을 실제로 찾았을 때만 패치한다(윈도우 10 의 정식 설치 경로는 그대로 둔다).
    """
    if not runtime_dir:
        return
    try:
        from webview.platforms import winforms as _wf
    except Exception as e:                               # noqa: BLE001
        print(f"[webview2] 백엔드 모듈 로드 실패: {e}", flush=True)
        return
    if not hasattr(_wf, "_is_chromium"):
        print("[webview2] 이 pywebview 버전에는 _is_chromium 이 없어 우회를 건너뜁니다.", flush=True)
        return
    if _wf._is_chromium():                               # 정식 설치돼 있으면 손대지 않는다
        print("[webview2] WebView2 가 정식 설치돼 있어 우회 불필요", flush=True)
        return
    _wf._is_chromium = lambda: True
    print(f"[webview2] 고정버전 런타임 사용 → 크로미움 강제(레지스트리 미등록 우회): {runtime_dir}",
          flush=True)


def main():
    # 콘솔 없는(windowed) exe 에선 stdout/stderr 가 None → print 가 죽는다.
    # 서버의 파일 로거를 가장 먼저 켜서 모든 로그를 로그\app_<날짜>.log 로 보낸다.
    _step(f"시작 — exe={sys.executable}")
    try:
        from src.app.server import _setup_logging
        _step("server 모듈 로드 완료")
        _setup_logging()
        _step("파일 로거 시작")
    except Exception:                                    # noqa: BLE001
        if getattr(sys, "stdout", None) is None:         # 최소 안전장치
            import io as _io
            sys.stdout = sys.stderr = _io.StringIO()
    print(f"{APP_TITLE} — 준비 중입니다. 잠시만 기다려 주세요…", flush=True)

    rt = _find_webview2_runtime()
    if rt:
        # (선택) 고정버전 런타임을 앱에 함께 번들한 '무거운 빌드' 인 경우에만 사용.
        #  기본은 린 빌드 + 전제조건 설치기로 WebView2 를 정식 설치하는 방식이라
        #  이 분기는 보통 타지 않는다.
        os.environ["WEBVIEW2_BROWSER_EXECUTABLE_FOLDER"] = str(rt)
        print(f"[webview2] 번들 고정버전 런타임 사용: {rt}", flush=True)
    else:
        print("[webview2] 시스템에 설치된 WebView2 런타임 사용 "
              "(전제조건 설치기로 설치됨). 미설치 시 창이 안 열립니다.", flush=True)

    # ★ 이미 떠 있는 앱이 있으면 새 서버를 띄우지 않고 그 화면에 붙는다.
    #   (창을 닫아도 남은 예전 프로세스가 있을 때 앱이 겹쳐 쌓이는 것을 막는다)
    _pref = int(os.environ.get("PORT") or 8000)
    _hit = running_app(_pref)
    _attached = _hit is not None
    port = _pref if _attached else _pick_port(_pref)
    base = f"http://127.0.0.1:{port}"
    if _attached:
        _step(f"이미 실행 중인 프로그램에 연결 — 포트 {port}")
        print("  이미 실행 중인 프로그램이 있어 서버를 새로 띄우지 않습니다.", flush=True)
    else:
        _step(f"포트 {port} 확보 — 서버 기동")
        start_server(port)
    print("  엔진 로딩 중… (torch/모델)", flush=True)
    if not _wait_ready(base + "/api/status"):
        if _SERVER_ERR:                                  # 원인을 알고 있으면 그대로 보여준다
            _boot_failed(RuntimeError("내부 서버 기동 실패\n\n" + _SERVER_ERR[0][-1500:]))
        else:
            _step("서버 준비 시간 초과")
            _boot_failed(RuntimeError(
                "서버가 시간 내 준비되지 않았습니다.\n"
                "백신/방화벽이 127.0.0.1 접속을 막고 있을 수 있습니다."))
        _idle()
        return

    _step("서버 준비 완료 — pywebview 로드")
    try:
        import webview
    except Exception as e:                               # noqa: BLE001
        _fatal(f"pywebview 로드 실패: {e}")
        _idle()
        return
    _force_chromium_if_bundled(webview, rt)
    print(f"[webview2] pywebview {getattr(webview, '__version__', '?')} · "
          f"런타임경로={os.environ.get('WEBVIEW2_BROWSER_EXECUTABLE_FOLDER', '(미설정=시스템 WebView2 필요)')}",
          flush=True)

    _step("창 생성")
    ui_url = fresh_ui_url(base)
    webview.create_window(APP_TITLE, ui_url, width=1480, height=980,
                          min_size=(1120, 720))
    try:
        webview.start(gui="edgechromium")                # WebView2 강제(IE 폴백 방지)
    except Exception as e:                               # noqa: BLE001
        emsg = str(e)
        if "Python.Runtime" in emsg or "clr" in emsg.lower() or "pythonnet" in emsg.lower():
            hint = ("pythonnet(.NET 브릿지) 초기화 실패입니다.\n"
                    " · 이 PC 에 .NET Framework 4.7.2 이상이 설치돼 있어야 합니다.\n"
                    " · 빌드에 Python.Runtime.dll 이 안 담겼으면 io_console_webview.spec 로 재빌드하세요.")
        else:
            hint = ("WebView2 런타임을 못 찾았을 수 있습니다.\n"
                    " · exe 옆 webview2_runtime\\ 폴더(msedgewebview2.exe 포함)를 확인하세요.")
        _fatal(f"창을 열 수 없습니다: {e}\n\n{hint}\n\n"
               f"임시 대안: 브라우저에서 {ui_url} 로 접속해도 동작합니다.")
        _idle()
    else:
        # 창이 정상적으로 닫힘(사용자 종료) — 검사/리포트 도중이라도
        # 자식 프로세스를 정리하고 프로세스를 즉시 완전 종료(메모리·핸들 반환).
        _hard_shutdown(0)


def _hard_shutdown(code: int = 0):
    """창이 닫히면(검사/리포트 도중이라도) 프로세스를 확실히 완전 종료한다.

    문제: 검사 중 창을 닫으면 onnxruntime/numpy 가 C 레벨에서 계속 돌거나,
    리포트 병렬 워커(spawn 자식 프로세스)가 고아로 남아 Task Manager 에 계속 뜨고
    메모리·디스크 핸들이 반환되지 않는다. 정상 종료(atexit·스레드 join)에 기대면
    이 지점에서 멈춘다.

    해결: (1) 자식 프로세스(리포트 병렬 워커 등)를 전부 강제 종료 → 고아 방지,
          (2) os._exit() 로 즉시 종료 → atexit/스레드 join 을 우회하고
          OS 가 모든 메모리·파일 핸들을 즉시 회수한다.
    """
    # 1) 자율감시 스레드 정지 시도(있으면) — 새 작업이 안 뜨도록.
    try:
        from src.app.server import AUTOPILOT
        if AUTOPILOT is not None:
            AUTOPILOT.stop()
    except Exception:                                    # noqa: BLE001
        pass
    # 2) 이 프로세스가 띄운 모든 자식 프로세스(리포트 spawn 워커 등) 강제 종료.
    try:
        import multiprocessing
        for p in multiprocessing.active_children():
            try:
                p.terminate()
            except Exception:                            # noqa: BLE001
                pass
        for p in multiprocessing.active_children():
            try:
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()
            except Exception:                            # noqa: BLE001
                pass
    except Exception:                                    # noqa: BLE001
        pass
    # 3) 남아있는 자식 프로세스 트리까지(Windows) 확실히 정리.
    if os.name == "nt":
        try:
            import subprocess
            CREATE_NO_WINDOW = 0x08000000
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(os.getpid())],
                           creationflags=CREATE_NO_WINDOW, timeout=5)
        except Exception:                                # noqa: BLE001
            pass
    # 4) 즉시 종료(정상 종료 절차를 기다리지 않음).
    sys.stdout.flush() if getattr(sys, "stdout", None) else None
    os._exit(code)


def _idle():
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # ★ 얼린 exe 에서 리포트 병렬 워커가 앱을 재실행하지 않게(필수)
    try:
        main()
    except SystemExit:
        raise
    except BaseException as _e:                          # noqa: BLE001
        # ★ 여기가 없으면 windowed exe 는 '아무 반응 없이' 사라진다.
        _boot_failed(_e)
        sys.exit(1)
