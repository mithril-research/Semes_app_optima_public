"""Trend requests should not reload each wafer or decompress unused arrays."""
import ast
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from src.app.sensor_differences import TrendSummaryCache, difference_series


def functions(*names):
    tree = ast.parse(Path('src/app/server.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
    return compile(ast.Module(body=nodes, type_ignores=[]), '<server-functions>', 'exec')


class HTTPException(Exception):
    def __init__(self, status, detail):
        self.status_code = status


class TrendRequestPerformanceTests(unittest.TestCase):
    def test_recipe_uses_one_summary_per_wafer_even_without_cache(self):
        summary = dict(meta={'step_timing': {'version': 2, 'steps': {'1': {'seconds': 3}}}},
                       summary=dict(series=[], sensor_series=[dict(sensor='P1', step_stats={
                           '1': dict(mean=2, min=1, max=3, std=.5, count=4)})]))
        load = Mock(side_effect=lambda *a, **k: copy.deepcopy(summary))
        ns = dict(HTTPException=HTTPException, re=re, _model_id=lambda: 'm',
                  _history_for_current_model=lambda: [{'wafer_id':str(i)} for i in range(65)],
                  _trend_summary=load, _wafer_step_timing=lambda wid, arch: arch['meta']['step_timing'])
        exec(functions('_scatter_trend_rows', 'api_scatter_step_times', 'api_scatter_differences'), ns)
        req = SimpleNamespace(model_id='m', wafers=[SimpleNamespace(date='2026-10-19',wid=str(i)) for i in range(64)])
        result = ns['api_scatter_step_times'](req)
        self.assertEqual(load.call_count, 64)
        self.assertEqual(len(result['wafers']), 64)
        for row in result['wafers']:
            self.assertEqual(row['peaks'], {'P1': {'1': 3}})
            self.assertEqual(row['sensor_metrics']['mean'], {'P1': {'1': 2}})
            self.assertEqual(row['step_timing']['steps']['1']['seconds'], 3)
            self.assertNotIn('sensor_series', row)
        req.wafers.append(SimpleNamespace(date='2026-10-19',wid='64'))
        with self.assertRaises(HTTPException) as caught:
            ns['api_scatter_step_times'](req)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(load.call_count, 64)
        req.wafers=req.wafers[:1];load.side_effect=lambda *a,**k:None
        missing=ns['api_scatter_step_times'](req)['wafers'][0]
        self.assertTrue(missing['missing'])
        self.assertEqual(missing['peaks'], {})
        self.assertEqual(missing['step_timing']['steps'], {})

    def test_full_sensor_archive_does_not_read_unused_model_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'w.npz'
            meta={'sensor_cols':['P1'], 'trend_sensor_cols':['P1','T1'], 'trend_zero_version':1}
            arrays=dict(meta_json=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8),
                        actual=np.ones((12,1)),rca_insen=np.ones((12,20)),step=np.repeat([1,2],6),
                        trend_actual=np.arange(24,dtype=float).reshape(12,2),trend_step=np.repeat([1,2],6))
            np.savez_compressed(path,**arrays)
            expected=difference_series(dict(meta=meta,**{k:v for k,v in arrays.items() if k!='meta_json'}))
            read=[];original=np.load
            class Reader:
                def __enter__(self):
                    self.data=original(path,allow_pickle=False);self.files=self.data.files;return self
                def __exit__(self,*a):self.data.close()
                def __getitem__(self,key):read.append(key);return self.data[key]
            ns=dict(np=np,json=json,_archive_path=lambda *a:path,_model_id=lambda:'m',
                    _ARCHIVE_Q=None,_TREND_SUMMARY_CACHE=TrendSummaryCache())
            exec(functions('_trend_summary'),ns)
            with patch.object(np,'load',side_effect=lambda *a,**k:Reader()):
                result=ns['_trend_summary']('2026-10-19','w')
            self.assertEqual(result['summary'],expected)
            self.assertNotIn('actual',read)
            self.assertNotIn('rca_insen',read)
            self.assertEqual(read.count('trend_actual'),1)
