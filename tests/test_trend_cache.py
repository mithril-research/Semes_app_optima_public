import ast
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
from src.app.sensor_differences import TrendSummaryCache, difference_series


class TrendCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'wafer.npz'
        self.path.write_bytes(b'original')
        self.cache = TrendSummaryCache()

    def test_reuse_independent_results_and_model_isolation(self):
        build = Mock(return_value={'values': [1, 2]})
        first = self.cache.get(self.path, 'm1', build)
        first['values'][0] = 99
        self.assertEqual(self.cache.get(self.path, 'm1', build)['values'], [1, 2])
        self.assertEqual(build.call_count, 1)
        self.cache.get(self.path, 'm2', build)
        self.assertEqual(build.call_count, 2)

    def test_atomic_replacement_and_missing_archive(self):
        build = Mock(side_effect=lambda: {'size': self.path.stat().st_size} if self.path.exists() else None)
        self.cache.get(self.path, 'm', build)
        replacement = self.path.with_suffix('.tmp')
        replacement.write_bytes(b'new wafer data')
        replacement.replace(self.path)
        self.assertEqual(self.cache.get(self.path, 'm', build), {'size': 14})
        self.path.unlink()
        self.assertIsNone(self.cache.get(self.path, 'm', build))
        self.path.write_bytes(b'restored')
        self.assertEqual(self.cache.get(self.path, 'm', build), {'size': 8})
        self.assertEqual(build.call_count, 4)

    def test_simultaneous_panels_compute_once(self):
        import time
        def calculate():
            time.sleep(.02)
            return {'values': [1]}
        build = Mock(side_effect=calculate)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.cache.get(self.path, 'm', build), range(8)))
        self.assertEqual(build.call_count, 1)
        self.assertEqual(results, [{'values': [1]}] * 8)

    def test_budget_eviction_and_errors_retry(self):
        cache = TrendSummaryCache(max_bytes=20)
        build = Mock(return_value={'x': '1234567890'})
        for model in ['a', 'b', 'a']:
            cache.get(self.path, model, build)
        self.assertLessEqual(cache.size, 20)
        self.assertEqual(build.call_count, 3)
        error = Mock(side_effect=[ValueError('bad archive'), {'ok': True}])
        with self.assertRaises(ValueError):
            cache.get(self.path, 'error', error)
        self.assertEqual(cache.get(self.path, 'error', error), {'ok': True})

    def test_replacement_during_read_is_not_cached(self):
        def change():
            self.path.write_bytes(b'changed')
            return {'old': True}
        self.cache.get(self.path, 'm', change)
        self.assertEqual(self.cache.get(self.path, 'm', lambda: {'new': True}), {'new': True})

    def test_summary_matches_original_without_reconstruction_arrays(self):
        arch = dict(actual=np.arange(24, dtype=float).reshape(12, 2),
                    step=np.repeat([1, 2, 3], 4),
                    meta={'sensor_cols': ['p_008_PV', 'p_007_PV'],
                          'step_timing': {'version': 2, 'steps': {'1': {'seconds': 2}}}})
        # No expected/z arrays: the former full archive loader would fail here.
        np.savez_compressed(self.path, actual=arch['actual'], step=arch['step'],
                            meta_json=np.frombuffer(json.dumps(arch['meta']).encode(), dtype=np.uint8))
        tree = ast.parse(Path('src/app/server.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_trend_summary')
        ns = dict(np=np, json=json, _archive_path=lambda *a: self.path,
                  _model_id=lambda: 'model', _TREND_SUMMARY_CACHE=self.cache, _ARCHIVE_Q=None,
                  _trend_original_path=lambda meta,wid: None)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<summary>', 'exec'), ns)
        for _ in range(2):
            result = ns['_trend_summary']('2026-10-02', 'w')
            self.assertEqual(result['summary'], difference_series(arch))
            self.assertEqual(result['meta']['step_timing'], arch['meta']['step_timing'])
        self.assertEqual(len(self.cache.entries), 1)


if __name__ == '__main__':
    unittest.main()
