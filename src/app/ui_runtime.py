"""UI 캐시 방지와 실행 중인 서버의 버전/설치 위치 확인(표준 라이브러리만 사용)."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def _fingerprint():
    frozen = getattr(sys, "frozen", False)
    root = Path(sys.executable).resolve().parent if frozen else Path(__file__).resolve().parents[1]
    pkg = Path(getattr(sys, "_MEIPASS", root)) / "src" if frozen else root
    files = [("static/index.html", pkg / "app" / "static" / "index.html")]
    if frozen:
        files.append(("executable", Path(sys.executable)))
    else:
        # 실행 코드만 포함 — 모델/검사 데이터/설정 변경으로 버전이 바뀌지 않는다.
        for folder in (pkg, pkg / "app", pkg / "train_jw"):
            files.extend((str(p.relative_to(pkg)), p) for p in folder.glob("*.py"))
    digest = hashlib.sha256()
    digest.update(os.environ.get("IO_APP_VERSION", "").encode("utf-8"))
    for name, path in sorted(files):
        digest.update(name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    location = os.path.normcase(str(root))
    return {"build_id": digest.hexdigest(),
            "install_id": hashlib.sha256(location.encode("utf-8")).hexdigest()}


# 시작 시 고정: 실행 도중 디스크 파일이 교체돼도 이전 서버가 새 버전으로 위장하지 않는다.
IDENTITY = _fingerprint()
INSTANCE_ID = uuid.uuid4().hex


def fresh_ui_url(base: str) -> str:
    """이전 버전에서 이미 저장된 HTML 캐시도 우회한다(같은 버전 재실행 포함)."""
    return base.rstrip("/") + "/?build=" + IDENTITY["build_id"] + "&open=" + uuid.uuid4().hex


def _probe_app(base: str, timeout: float):
    request = urllib.request.Request(base + "/api/ping-app?probe=" + uuid.uuid4().hex,
                                     headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _port_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) in (errno.ECONNREFUSED, 10061)


def running_app(port: int = 8000, timeout: float = 1.5, exit_wait: float = 330.0):
    """같은 설치·버전만 재사용. 구버전이 종료 중이면 포트가 해제될 때까지 기다린다."""
    base = f"http://127.0.0.1:{port}"
    try:
        data = _probe_app(base, timeout)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("app") != "chamber-inspect":
        return None
    deadline = time.monotonic() + max(0.0, exit_wait)
    announced = False
    while True:
        if all(data.get(key) == value for key, value in IDENTITY.items()):
            return base, int(data.get("pid") or 0)
        if _port_closed(port):
            return None
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"이전 앱의 서버가 아직 종료되지 않았습니다(PID {data.get('pid', '?')}).\n"
                "종료를 기다렸지만 포트가 계속 사용 중입니다. 이전 화면에는 연결하지 않았습니다.\n"
                "진행 중인 검사·학습을 마친 뒤 기존 앱과 서버 프로세스를 종료하고 다시 실행하세요."
            )
        if not announced:
            print(f"[startup] 이전 앱 서버(PID {data.get('pid', '?')}) 종료 대기 중… "
                  "기존 창을 닫았다면 서버 종료 후 자동으로 시작합니다.", flush=True)
            announced = True
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        try:
            candidate = _probe_app(base, timeout)
            if isinstance(candidate, dict) and candidate.get("app") == "chamber-inspect":
                data = candidate
        except (OSError, ValueError):
            pass  # 응답 실패만으로 종료로 판단하지 않는다. 실제 포트 해제를 확인한다.


class FreshUIResponses:
    """HTML과 변경 가능한 API는 저장하지 않는다. 폰트 등 고정 리소스는 기존 정책 유지."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        dynamic = path in ("/", "/static/index.html") or path.startswith("/api/")
        if scope["type"] != "http" or not dynamic:
            await self.app(scope, receive, send)
            return

        async def send_fresh(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.lower() not in (b"cache-control", b"pragma", b"expires")]
                headers.extend([(b"cache-control", b"no-store"),
                                (b"pragma", b"no-cache"), (b"expires", b"0"),
                                (b"x-chamber-instance", INSTANCE_ID.encode("ascii"))])
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_fresh)
