import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


class OfflinePrepareTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('prepared', Path(__file__).resolve().parents[1] / 'src/app/win7_build_prepared.py')
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m.ROOT = self.root
        self.m.os = SimpleNamespace(name='nt', environ={'SystemDrive': str(self.root)})
        self.m.sys = SimpleNamespace(version_info=(3, 8, 10), executable='python38')
        self.commands = []
        self.m.run = lambda args, env: self.commands.append([str(x) for x in args])
        req = self.root / 'src/app/requirements-win7.txt'
        req.parent.mkdir(parents=True)
        req.write_text('sample==1\n')
        self.wheels = self.root / 'offline/win7/onnx'
        self.wheels.mkdir(parents=True)
        self.package = self.wheels / 'sample-1-py3-none-any.whl'
        self.package.write_bytes(b'test fixture')
        (self.wheels / 'requirements.lock').write_text('sample==1\n')
        (self.wheels / 'manifest.json').write_text(json.dumps(dict(
            backend='onnx', requirements_sha256=hashlib.sha256(req.read_bytes()).hexdigest(),
            packages=[dict(file=self.package.name, sha256=hashlib.sha256(self.package.read_bytes()).hexdigest())])))

    def prepare(self):
        with patch('sys.argv', ['prepare', 'prepare']):
            self.m.main()

    def test_install_uses_only_local_wheels_and_hashes(self):
        self.prepare()
        install = next(c for c in self.commands if 'install' in c)
        for flag in ['--no-index', '--find-links', '--only-binary=:all:', '--require-hashes']:
            self.assertIn(flag, install)
        self.assertNotIn('--upgrade', install)

    def test_corrupt_package_fails_before_install(self):
        self.package.write_bytes(b'broken')
        with self.assertRaisesRegex(RuntimeError, 'corrupt'):
            self.prepare()
        self.assertEqual(self.commands, [])

    def test_missing_manifest_fails_without_online_fallback(self):
        (self.wheels / 'manifest.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'manifest'):
            self.prepare()
        self.assertEqual(self.commands, [])

    def test_changed_requirements_fails_before_install(self):
        (self.root / 'src/app/requirements-win7.txt').write_text('sample==2')
        with self.assertRaisesRegex(RuntimeError, 'Requirements changed'):
            self.prepare()
        self.assertEqual(self.commands, [])

    def test_build_outputs_directly_under_app_even_if_compatibility_fails(self):
        self.m.os.getpid = lambda: 1234
        python = self.root / 'semes_win7_prepared/venv_onnx/Scripts/python.exe'
        python.parent.mkdir(parents=True)
        python.write_bytes(b'fixture')
        output = self.root / 'ChamberInspect_WIN7'
        output.mkdir(parents=True)
        (output / 'previous.txt').write_text('keep previous build')
        def fake_build(command, env, log):
            dist = Path(command[command.index('--distpath') + 1])
            self.assertEqual(dist, self.root)
            self.assertEqual(log.parent, self.root / 'build')
            built = dist / 'ChamberInspect_WIN7'
            built.mkdir()
            (built / 'ChamberInspect.exe').write_bytes(b'new build fixture')
        self.m.run_build = fake_build
        def fake_run(command, env):
            if any(str(x).endswith('check_win7_compat.py') for x in command):
                self.assertEqual(command[-1], output)
                raise RuntimeError('compatibility failure fixture')
        self.m.run = fake_run
        with patch('sys.argv', ['build', 'build']):
            with self.assertRaisesRegex(RuntimeError, 'compatibility failure fixture'):
                self.m.main()
        self.assertTrue((output / 'ChamberInspect.exe').is_file())
        backups = list(self.root.glob('ChamberInspect_WIN7_backup_*/previous.txt'))
        self.assertEqual(len(backups), 1)
