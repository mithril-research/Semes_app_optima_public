import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepared_progress', Path(__file__).resolve().parents[1] / 'src/app/win7_build_prepared.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class BuildProgressTests(unittest.TestCase):
    def test_output_log_and_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'build.log'
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                build.run_build([sys.executable, '-u', '-c',
                    'import time; print("phase one", flush=True); time.sleep(.25); print("phase two")'],
                    os.environ.copy(), log, heartbeat_seconds=.05)
            text = output.getvalue()
            self.assertIn('phase one', text)
            self.assertIn('phase two', text)
            self.assertIn('Process running; elapsed', text)
            self.assertIn('finished successfully', text)
            self.assertEqual(log.read_text(), text)

    def test_failure_keeps_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'build.log'
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'exit 7'):
                    build.run_build([sys.executable, '-c',
                        'import sys; print("build error", file=sys.stderr); sys.exit(7)'],
                        os.environ.copy(), log)
            self.assertIn('build error', log.read_text())
            self.assertNotIn('finished successfully', log.read_text())
