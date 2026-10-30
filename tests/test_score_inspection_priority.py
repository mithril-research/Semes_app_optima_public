"""Score reads remain available while inspection progress is active."""
import ast
from functools import wraps
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock
from types import SimpleNamespace


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code


class ScoreConcurrencyTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path('src/app/server.py').read_text())
        names = {'_track_inspection_request', 'api_scatter_step_scores'}
        nodes = [n for n in tree.body if getattr(n, 'name', '') in names]
        for node in nodes:
            node.decorator_list = []
        self.state = {}
        self.reader = Mock(return_value=None)
        self.ns = dict(STATE=self.state, STATE_LOCK=threading.Lock(),
                       HTTPException=HTTPException, wraps=wraps,
                       ScatterStepScoresReq=object, _model_id=lambda: 'm',
                       _history_for_current_model=lambda: [{'wafer_id': 'w1'}, {'wafer_id': 'w2'}],
                       re=__import__('re'), _load_archive=self.reader)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<concurrent scores>', 'exec'), self.ns)
        self.req = SimpleNamespace(model_id='m', steps=[1], sensors=['P1'], wafers=[
            SimpleNamespace(wid=w, date='2026-09-16') for w in ('w1', 'w2')])

    def test_foreground_background_and_paused_inspection_allow_reads(self):
        for state in ({'progress': {'active': True}},
                      {'progress_bg': {'active': True}},
                      {'progress': {'active': True, 'phase': '일시중단'}},
                      {'score_inspect_requests': 1}):
            self.state.clear(); self.state.update(state)
            self.reader.reset_mock()
            result = self.ns['api_scatter_step_scores'](self.req)
            self.assertEqual(self.reader.call_count, 2)
            self.assertEqual(len(result['wafers']), 2)
            self.assertTrue(all(w['missing'] == '보관 데이터 없음' for w in result['wafers']))

    def test_progress_tracking_cleans_up_errors_without_blocking_scores(self):
        @self.ns['_track_inspection_request']
        def inspect():
            self.assertEqual(self.state['score_inspect_requests'], 1)
            self.assertEqual(len(self.ns['api_scatter_step_scores'](self.req)['wafers']), 2)
            raise ValueError('inspection failed')
        with self.assertRaises(ValueError):
            inspect()
        self.assertEqual(self.state['score_inspect_requests'], 0)

    def test_inspection_start_between_wafers_does_not_interrupt_batch(self):
        def load(*args):
            self.state['progress'] = {'active': True}
            return None
        self.reader.side_effect = load
        result = self.ns['api_scatter_step_scores'](self.req)
        self.assertEqual(self.reader.call_count, 2)
        self.assertEqual([w['wid'] for w in result['wafers']], ['w1', 'w2'])
