import ast
import datetime as dt
from pathlib import Path
import unittest


class CustomOverviewWindowTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path('src/app/server.py').read_text())
        names = {'_parse_ts', '_bucket_bounds', '_range_window', 'api_overview'}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        for fn in functions:
            fn.decorator_list = []
        self.rows = [dict(wafer_id=str(i), ts=t, recipe='29a', chamber='dsv1',
                          verdict='OK', band='ok') for i, t in enumerate([
            '2026-08-31T23:59:59', '2026-09-01T00:00:00',
            '2026-09-10T23:59:59', '2026-09-11T00:00:00', '2026-09-30T12:00:00'])]
        self.ns = dict(dt=dt, Dict=dict, Query=lambda default, **kw: default,
                       _history_for_current_model=lambda: self.rows,
                       _agg_cached=lambda *args: None, _agg_store=lambda *args: args[-1],
                       _recipe_display=lambda r: r, _chamber_key=lambda r: r['chamber'],
                       _display_rule=lambda: {'bands': {}}, CHAMBERS=['dsv1'],
                       chamber_overview=lambda **kw: {'chambers': []},
                       _chamber_level=lambda *args: 0)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<overview>', 'exec'), self.ns)

    def test_custom_month_counts_only_requested_days(self):
        for scale in ['day', 'week', 'month']:
            result = self.ns['api_overview'](rng='custom', scale=scale,
                start='2026-09-01T00:00:00', end='2026-09-10T23:59:59')
            self.assertEqual(result['bounds'][0], '2026-09-01T00:00:00')
            self.assertEqual(result['bounds'][-1], '2026-09-11T00:00:00')
            self.assertEqual(result['summary']['total'], 2)
            self.assertEqual(sum(c['total'] for c in result['lanes'][0]['cells']), 2)
            if scale == 'month':
                self.assertEqual(result['count'], 1)

    def test_partial_months_and_single_second(self):
        result = self.ns['api_overview'](rng='custom', scale='month',
            start='2026-08-31T12:30:00', end='2026-09-10T23:59:59')
        self.assertEqual(result['bounds'], ['2026-08-31T12:30:00',
            '2026-09-01T00:00:00', '2026-09-11T00:00:00'])
        result = self.ns['api_overview'](rng='custom', scale='month',
            start='2026-09-10T23:59:59', end='2026-09-10T23:59:59')
        self.assertEqual(result['summary']['total'], 1)

    def test_normal_calendar_alignment_unchanged(self):
        bounds = self.ns['_bucket_bounds'](dt.datetime(2026, 9, 4), dt.datetime(2026, 9, 10), 'month')
        self.assertEqual(bounds, [dt.datetime(2026, 9, 1), dt.datetime(2026, 10, 1)])

    def test_selected_scores_receive_clipped_window(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        tree = ast.parse(Path('src/app/server.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'api_overview_selection')
        fn.decorator_list = []
        scatter = Mock(return_value={'points': self.rows})
        dispersion = Mock(return_value={})
        self.ns.update(OverviewSelectionReq=SimpleNamespace, ScatterStepScoresReq=SimpleNamespace,
            _model_id=lambda: 'm', api_scatter=scatter, _disp_band=lambda *a: 'ok',
            api_scatter_step_scores=lambda req: {'wafers': [dict(wid=w['wid'], ratio=.5) for w in req.wafers]},
            _dispersion_result=dispersion)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<selection>', 'exec'), self.ns)
        result = self.ns['api_overview_selection'](SimpleNamespace(model_id='m', steps=[1], sensors=['s'],
            rng='custom', scale='month', month_start_day=1, recipe='all', start='2026-09-01T00:00:00',
            end='2026-09-10T23:59:59', chambers='', recipes='', dispersion=True, hold=False, include_first=True))
        self.assertEqual(scatter.call_args.kwargs['end'], '2026-09-11T00:00:00')
        self.assertEqual(result['lanes'][0]['cells'][0]['total'], 2)
        self.assertEqual(len(dispersion.call_args.args[0]), 2)
        self.assertTrue(dispersion.call_args.kwargs['clip'])

    def test_user_month_anchor_and_default(self):
        args = dict(rng='custom', scale='month', start='2026-08-13T00:00:00', end='2026-09-13T23:59:59')
        custom = self.ns['api_overview'](**args, month_start_day=13)
        self.assertEqual(custom['bounds'], ['2026-08-13T00:00:00', '2026-09-14T00:00:00'])
        self.assertEqual(custom['buckets'], ['8/13~9/13'])
        normal = self.ns['api_overview'](**args)
        self.assertEqual(normal['buckets'], ['2026-08', '2026-09'])
        self.assertEqual(normal['summary']['total'], custom['summary']['total'])

    def test_anchor_short_month_and_year_boundary(self):
        bounds = self.ns['_bucket_bounds'](dt.datetime(2026, 1, 31), dt.datetime(2026, 4, 2), 'month', clip=True, month_start_day=31)
        self.assertEqual(bounds, [dt.datetime(2026, 1, 31), dt.datetime(2026, 2, 28), dt.datetime(2026, 3, 31), dt.datetime(2026, 4, 2)])
        bounds = self.ns['_bucket_bounds'](dt.datetime(2025, 12, 20), dt.datetime(2026, 2, 15), 'month', clip=True, month_start_day=13)
        self.assertEqual(bounds, [dt.datetime(2025, 12, 20), dt.datetime(2026, 1, 13), dt.datetime(2026, 2, 13), dt.datetime(2026, 2, 15)])
