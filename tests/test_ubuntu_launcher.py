"""Exercise shell argument handling and the actual environment passed to Python."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'src/app/run_ubuntu.sh'


class UbuntuLauncherTest(unittest.TestCase):
    def test_headless_start_passes_bind_port_and_keeps_server_alive(self):
        with tempfile.TemporaryDirectory(prefix='optima venv ') as d:
            python = Path(d) / 'bin/python'
            python.parent.mkdir()
            python.write_text('#!/usr/bin/env python3\n'
                              'import json, os, sys\n'
                              'print(json.dumps({"args": sys.argv[1:], "cwd": os.getcwd(), '
                              '"env": {k: os.environ.get(k) for k in '
                              '["PORT", "IO_HOST", "IO_KEEP_ALIVE", "MPLBACKEND"]}}))\n')
            python.chmod(0o755)
            env = dict(os.environ, OPTIMA_VENV=d, PORT='18099', IO_HOST='127.0.0.1',
                       IO_KEEP_ALIVE='0')
            result = subprocess.run(['bash', str(SCRIPT)], cwd='/tmp', env=env,
                                    capture_output=True, text=True, check=True)
            out = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(out['args'], ['-m', 'src.app.server'])
            self.assertEqual(out['cwd'], str(ROOT))
            self.assertEqual(out['env'], dict(PORT='18099', IO_HOST='127.0.0.1',
                                             IO_KEEP_ALIVE='1', MPLBACKEND='Agg'))

    def test_unknown_action_fails_instead_of_starting_server(self):
        result = subprocess.run(['bash', str(SCRIPT), 'unknown'], capture_output=True)
        self.assertEqual(result.returncode, 2)

    def test_cpu_settings_follow_server_state_directory(self):
        from src.app import cpu_limit
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, IO_STATE_DIR=d):
            (Path(d) / 'app_config.json').write_text('{"cpu_max": 2}')
            self.assertEqual(cpu_limit._writable_root(), Path(d))
            self.assertEqual(cpu_limit._read_cfg()['cpu_max'], 2)


if __name__ == '__main__':
    unittest.main()
