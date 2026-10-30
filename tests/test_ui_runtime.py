"""Run with: python -m unittest discover -s tests -p test_ui_runtime.py"""
import asyncio
import ast
import errno
import io
import json
import tempfile
import threading
import time
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

from src.app import desktop, launcher, ui_runtime as UI


class RuntimeTests(unittest.TestCase):
    def probe(self, payload):
        with patch.object(UI.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())), \
                patch.object(UI, "_port_closed", return_value=False):
            return launcher._running_app(8000, exit_wait=0)

    def test_old_server_closing_is_waited_for(self):
        with patch.object(UI, "_probe_app", side_effect=[{"app": "chamber-inspect", "pid": 123}, OSError("closing")]), \
                patch.object(UI, "_port_closed", side_effect=[False, True]), \
                patch.object(UI.time, "sleep") as sleep:
            self.assertIsNone(UI.running_app(exit_wait=10))
            sleep.assert_called_once()

    def test_unresponsive_old_server_is_not_treated_as_closed(self):
        with patch.object(UI, "_probe_app", side_effect=[{"app": "chamber-inspect", "pid": 123}, OSError("timeout")]), \
                patch.object(UI, "_port_closed", return_value=False), \
                patch.object(UI.time, "sleep"), \
                patch.object(UI.time, "monotonic", side_effect=[0, 0, 0, 2]), \
                self.assertRaisesRegex(RuntimeError, "아직 종료되지"):
            UI.running_app(exit_wait=1)

    def test_busy_port_timeout_is_not_closed(self):
        for code, expected in ((0, False), (errno.ETIMEDOUT, False), (errno.ECONNREFUSED, True), (10061, True)):
            with patch.object(UI.socket, "socket") as socket:
                socket.return_value.__enter__.return_value.connect_ex.return_value = code
                self.assertEqual(UI._port_closed(8000), expected)

    def test_brief_browser_visit_marks_server_as_seen(self):
        tree = ast.parse(Path("src/app/server.py").read_text(encoding="utf-8"))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "api_alive")
        func.decorator_list = []
        loop = types.SimpleNamespace()
        ns = {"_CLIENTS_LOCK": threading.Lock(), "_CLIENTS": {}, "time": time,
              "_exit_watch_loop": loop, "_EXIT_ENABLED": True}
        exec(compile(ast.Module(body=[func], type_ignores=[]), "<api_alive>", "exec"), ns)
        ns["api_alive"]("quick-visit")
        self.assertTrue(loop._seen)
        self.assertIn("quick-visit", ns["_CLIENTS"])

    def test_matching_install_and_build_are_reused(self):
        self.assertEqual(self.probe({"app": "chamber-inspect", "pid": 123, **UI.IDENTITY}),
                         ("http://127.0.0.1:8000", 123))

    def test_old_build_legacy_and_other_install_are_rejected(self):
        base = {"app": "chamber-inspect", "pid": 123, **UI.IDENTITY}
        for payload in ({"app": "chamber-inspect", "pid": 123},
                        {**base, "build_id": "old"}, {**base, "install_id": "elsewhere"}):
            with self.subTest(payload=payload), self.assertRaisesRegex(RuntimeError, "기존 앱과 서버"):
                self.probe(payload)

    def test_other_service_or_closed_port_is_not_reused(self):
        self.assertIsNone(self.probe({"app": "other"}))
        with patch.object(UI.urllib.request, "urlopen", side_effect=OSError("closed")):
            self.assertIsNone(UI.running_app())

    def test_each_open_uses_unique_url(self):
        first = UI.fresh_ui_url("http://127.0.0.1:8000")
        second = UI.fresh_ui_url("http://127.0.0.1:8000")
        self.assertNotEqual(first, second)
        self.assertEqual(parse_qs(urlsplit(first).query)["build"], [UI.IDENTITY["build_id"]])

    def test_source_fingerprint_detects_html_and_code_changes(self):
        snapshot = dict(UI.IDENTITY)
        with tempfile.TemporaryDirectory() as td:
            app = Path(td) / "src" / "app"
            (app / "static").mkdir(parents=True)
            html = app / "static" / "index.html"
            html.write_text("old UI")
            code = app / "server.py"
            code.write_text("old code")
            with patch.object(UI, "__file__", str(app / "ui_runtime.py")), patch.object(UI.sys, "frozen", False, create=True):
                first = UI._fingerprint()
                html.write_text("new UI")
                second = UI._fingerprint()
                code.write_text("new code")
                third = UI._fingerprint()
            self.assertNotEqual(first["build_id"], second["build_id"])
            self.assertNotEqual(second["build_id"], third["build_id"])
            self.assertEqual(first["install_id"], third["install_id"])
        self.assertEqual(UI.IDENTITY, snapshot)  # running process keeps its startup identity

    def test_frozen_build_detects_executable_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            static = root / "_internal" / "src" / "app" / "static"
            static.mkdir(parents=True)
            (static / "index.html").write_text("UI")
            exe = root / "ChamberInspect.exe"
            exe.write_bytes(b"build A")
            with patch.object(UI.sys, "frozen", True, create=True), \
                    patch.object(UI.sys, "_MEIPASS", str(root / "_internal"), create=True), \
                    patch.object(UI.sys, "executable", str(exe)):
                first = UI._fingerprint()
                exe.write_bytes(b"build B")
                self.assertNotEqual(first["build_id"], UI._fingerprint()["build_id"])

    def test_browser_launches_receive_fresh_urls(self):
        for browser in (None, ("Chrome", "chrome.exe"), ("Firefox", "firefox.exe")):
            with self.subTest(browser=browser), \
                    patch.object(launcher, "_wait_ready", return_value=True), \
                    patch.object(launcher, "_find_modern_browser", return_value=browser), \
                    patch.object(launcher.webbrowser, "open") as open_browser, \
                    patch("subprocess.Popen") as popen:
                launcher._open_browser_when_ready("http://127.0.0.1:8000")
                url = popen.call_args[0][0][1] if browser else open_browser.call_args[0][0]
                self.assertIn("open=", url)
                self.assertIn("build=", url)

    def test_desktop_launch_and_reuse_use_fresh_url(self):
        for hit in (None, ("http://127.0.0.1:8000", 123)):
            with self.subTest(hit=hit), ExitStack() as stack:
                webview = MagicMock()
                stack.enter_context(patch.dict("sys.modules", {
                    "src.app.server": types.SimpleNamespace(_setup_logging=lambda: None), "webview": webview}))
                for name in ("_step", "_force_chromium_if_bundled", "_hard_shutdown"):
                    stack.enter_context(patch.object(desktop, name))
                stack.enter_context(patch.object(desktop, "_find_webview2_runtime", return_value=None))
                stack.enter_context(patch.object(desktop, "running_app", return_value=hit))
                stack.enter_context(patch.object(desktop, "_pick_port", return_value=8000))
                stack.enter_context(patch.object(desktop, "_wait_ready", return_value=True))
                start = stack.enter_context(patch.object(desktop, "start_server"))
                desktop.main()
                self.assertEqual(start.call_count, 1 if hit is None else 0)
                self.assertIn("open=", webview.create_window.call_args[0][1])

    def test_desktop_mismatch_does_not_start_or_open(self):
        with patch.dict("sys.modules", {"src.app.server": types.SimpleNamespace(_setup_logging=lambda: None)}), \
                patch.object(desktop, "_step"), \
                patch.object(desktop, "_find_webview2_runtime", return_value=None), \
                patch.object(desktop, "running_app", side_effect=RuntimeError("old build")), \
                patch.object(desktop, "start_server") as start, \
                self.assertRaisesRegex(RuntimeError, "old build"):
            desktop.main()
        start.assert_not_called()


class CacheTests(unittest.TestCase):
    def response(self, path, status=200):
        messages = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"cache-control", b"public, max-age=3600"),
                                    (b"content-type", b"text/html")]})
            await send({"type": "http.response.body", "body": b"new UI"})

        async def send(message):
            messages.append(message)

        async def receive():
            return {"type": "http.request", "body": b""}

        asyncio.run(UI.FreshUIResponses(app)({"type": "http", "path": path}, receive, send))
        return messages

    def test_html_and_api_responses_are_not_stored(self):
        for path in ("/", "/static/index.html", "/api/ping-app", "/api/status"):
            for status in (200, 304, 404):
                with self.subTest(path=path, status=status):
                    result = self.response(path, status)
                    headers = dict(result[0]["headers"])
                    self.assertEqual(headers[b"cache-control"], b"no-store")
                    self.assertEqual(headers[b"pragma"], b"no-cache")
                    self.assertEqual(headers[b"x-chamber-instance"], UI.INSTANCE_ID.encode("ascii"))
                    self.assertEqual(result[0]["status"], status)
                    self.assertEqual(result[1]["body"], b"new UI")
                    self.assertEqual(sum(k == b"cache-control" for k, v in result[0]["headers"]), 1)

    def test_font_cache_is_preserved(self):
        result = self.response("/static/fonts/PretendardVariable.woff2")
        self.assertEqual(dict(result[0]["headers"])[b"cache-control"], b"public, max-age=3600")


if __name__ == "__main__":
    unittest.main()
