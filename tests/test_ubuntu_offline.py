import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.app import ubuntu_offline as offline


class OfflinePackageTests(unittest.TestCase):
    def bundle(self, root):
        (root / 'wheels').mkdir()
        wheel = root / 'wheels/example.whl'
        wheel.write_bytes(b'fixture')
        lock = root / 'requirements.lock'
        lock.write_text('example==1\n')
        manifest = {'target': offline.target(), 'wheels': {'example.whl': offline.digest(wheel)},
                    'lock_sha256': offline.digest(lock)}
        (root / 'manifest.json').write_text(json.dumps(manifest))

    def test_modified_wheel_rejected_before_creating_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.bundle(root)
            (root / 'wheels/example.whl').write_bytes(b'corrupt')
            destination = root / 'venv'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                offline.install(root, destination)
            self.assertFalse(destination.exists())

    def test_target_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.bundle(root)
            with patch.object(offline, 'target', return_value={'python': 'wrong'}):
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    offline.verify(root)

    def test_extra_or_missing_wheels_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.bundle(root)
            offline.verify(root)
            (root / 'wheels/extra.whl').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'missing or unexpected'):
                offline.verify(root)

    def test_existing_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.bundle(root)
            destination = root / 'venv'
            destination.mkdir()
            sentinel = destination / 'keep'
            sentinel.write_text('existing')
            with self.assertRaisesRegex(ValueError, 'existing environment is preserved'):
                offline.install(root, destination)
            self.assertEqual(sentinel.read_text(), 'existing')
