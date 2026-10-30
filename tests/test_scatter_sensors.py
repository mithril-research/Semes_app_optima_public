"""Archive batch contract without importing the application/startup threads."""
import ast
from contextlib import nullcontext
from pathlib import Path
import re
import unittest
from unittest.mock import Mock
import numpy as np
from pydantic import BaseModel


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


class ScatterSensorsTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).parents[1] / 'src/app/server.py').read_text())
        names = {'ScatterSensorWaferReq', 'ScatterSensorsReq', 'api_scatter_sensors'}
        nodes = [n for n in tree.body if getattr(n, 'name', '') in names]
        for n in nodes:
            if isinstance(n, ast.FunctionDef):
                n.decorator_list = []
        self.reader = Mock(return_value={'sensors': [
            {'sensor': 't_1', 'ratio': 3.75, 'score': 2.0},
            {'sensor': 'p_1', 'ratio': 0.0},
            {'sensor': 't_missing', 'ratio': None},
            {'sensor': 'p_nan', 'ratio': float('nan')}]})
        self.ns = dict(BaseModel=BaseModel, List=list, re=re, np=np,
                       HTTPException=HTTPException, _model_id=lambda: 'model-a',
                       _history_for_current_model=lambda: [{'wafer_id': 'w1'}, {'wafer_id': 'w2'}],
                       api_archive_sensors_all=self.reader)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<server functions>', 'exec'), self.ns)

    def request(self, wafers=None, model_id='model-a'):
        return self.ns['ScatterSensorsReq'](model_id=model_id, wafers=wafers or [{'date': '2026-09-09', 'wid': 'w1'}])

    def test_unclipped_ratio_zero_and_nonfinite(self):
        result = self.ns['api_scatter_sensors'](self.request())
        self.assertEqual(result['wafers'][0]['scores'], {'t_1': 3.75, 'p_1': 0.0})

    def test_missing_archive_is_gap_and_order_preserved(self):
        self.reader.side_effect = [HTTPException(404, 'missing'), {'sensors': []}]
        result = self.ns['api_scatter_sensors'](self.request([
            {'date': '2026-09-09', 'wid': 'w2'}, {'date': '2026-09-09', 'wid': 'w1'}]))
        self.assertEqual([r['wid'] for r in result['wafers']], ['w2', 'w1'])
        self.assertEqual(result['wafers'][0]['scores'], {})
        self.assertTrue(result['wafers'][0]['missing'])

    def test_model_change_rejected_before_read(self):
        with self.assertRaises(HTTPException) as ctx:
            self.ns['api_scatter_sensors'](self.request(model_id='old'))
        self.assertEqual(ctx.exception.status_code, 409)
        self.reader.assert_not_called()

    def test_batch_bound(self):
        with self.assertRaises(HTTPException) as ctx:
            self.ns['api_scatter_sensors'](self.request([{'date': '2026-09-09', 'wid': 'w1'}]*65))
        self.assertEqual(ctx.exception.status_code, 400)
        self.reader.assert_not_called()

    def test_unknown_wafer_and_invalid_date(self):
        for row in [{'date': '2026-09-09', 'wid': '../unknown'}, {'date': '../bad', 'wid': 'w1'}]:
            with self.assertRaises(HTTPException):
                self.ns['api_scatter_sensors'](self.request([row]))
        self.reader.assert_not_called()

    def test_nonmissing_http_error_propagates(self):
        self.reader.side_effect = HTTPException(500, 'failure')
        with self.assertRaises(HTTPException) as ctx:
            self.ns['api_scatter_sensors'](self.request())
        self.assertEqual(ctx.exception.status_code, 500)


if __name__ == '__main__':
    unittest.main()
