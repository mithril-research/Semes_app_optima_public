"""CPU boot settings must come from the same file used by the settings API."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.app import cpu_limit as CPU


class CPUConfigTests(unittest.TestCase):
    def test_app_config_overrides_legacy_data_config_at_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '데이터').mkdir()
            (root / '데이터' / 'app_config.json').write_text(json.dumps({'cpu_max': 0}))
            (root / 'app_config.json').write_text(json.dumps({'cpu_max': 1, 'cpu_affinity': True}))
            topo = {'physical': 4, 'logical': 8, 'masks': [3, 12, 48, 192]}
            with patch.object(CPU, '_writable_root', return_value=root), \
                 patch.object(CPU, 'topology', return_value=topo), \
                 patch.object(CPU, '_STATE', {}), patch.dict(os.environ, {}, clear=True):
                b = CPU.apply_early()
                self.assertEqual(b['use_cores'], 1)
                self.assertEqual(b['threads'], 2)
                self.assertTrue(b['affinity'])
                self.assertEqual(os.environ['IO_ORT_THREADS'], '2')
                self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'], '2')

    def test_legacy_custom_data_path_used_only_without_app_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'old-data'
            data.mkdir()
            (root / 'storage.json').write_text(json.dumps({'data': 'old-data'}))
            (data / 'app_config.json').write_text(json.dumps({'cpu_max': 1}))
            with patch.object(CPU, '_writable_root', return_value=root):
                self.assertEqual(CPU._read_cfg()['cpu_max'], 1)
                (root / 'app_config.json').write_text('{}')
                self.assertEqual(CPU._read_cfg(), {})


class WindowsAffinityTests(unittest.TestCase):
    def kernel(self, mask):
        from unittest.mock import Mock
        k = Mock()
        k.GetCurrentProcess.return_value = 0x1234567887654321
        k.SetProcessAffinityMask.return_value = 1
        def read(handle, process, system):
            process._obj.value = mask
            system._obj.value = (1 << 48) - 1
            return 1
        k.GetProcessAffinityMask.side_effect = read
        return k

    def test_64bit_signatures_error_capture_and_readback(self):
        import ctypes
        from ctypes import wintypes
        mask = (1 << 40) | (1 << 41)
        kernel = self.kernel(mask)
        with patch.object(ctypes, 'WinDLL', return_value=kernel, create=True) as loader:
            CPU._win_set_affinity(mask)
        loader.assert_called_with('kernel32', use_last_error=True)
        self.assertIs(kernel.GetCurrentProcess.restype, wintypes.HANDLE)
        self.assertEqual(kernel.SetProcessAffinityMask.argtypes, [wintypes.HANDLE, ctypes.c_size_t])
        handle, requested = kernel.SetProcessAffinityMask.call_args.args
        self.assertEqual(handle, 0x1234567887654321)
        self.assertEqual(requested.value, mask)
        kernel.GetProcessAffinityMask.assert_called_once()

    def test_real_error_is_reported_and_failed_set_does_not_claim_success(self):
        import ctypes
        kernel = self.kernel(3)
        kernel.SetProcessAffinityMask.return_value = 0
        with patch.object(ctypes, 'WinDLL', return_value=kernel, create=True), \
             patch.object(ctypes, 'get_last_error', return_value=6, create=True), \
             self.assertRaises(OSError) as ctx:
            CPU._win_set_affinity(3)
        self.assertEqual(ctx.exception.errno, 6)
        kernel.GetProcessAffinityMask.assert_not_called()

    def test_mismatched_readback_is_not_success(self):
        import ctypes
        kernel = self.kernel(255)
        with patch.object(ctypes, 'WinDLL', return_value=kernel, create=True), \
             self.assertRaisesRegex(RuntimeError, '검증 실패'):
            CPU._win_set_affinity(3)

    def test_readback_failure_preserves_error(self):
        import ctypes
        kernel = self.kernel(3)
        kernel.GetProcessAffinityMask.side_effect = None
        kernel.GetProcessAffinityMask.return_value = 0
        with patch.object(ctypes, 'WinDLL', return_value=kernel, create=True), \
             patch.object(ctypes, 'get_last_error', return_value=5, create=True), \
             self.assertRaises(OSError) as ctx:
            CPU._win_process_mask()
        self.assertEqual(ctx.exception.errno, 5)
