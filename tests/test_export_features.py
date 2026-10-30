from typing import Optional
import ast
import datetime as dt
import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.app.export_html import build_export_html


class ExportFeaturesTests(unittest.TestCase):
    def test_curation_export_retains_process_payload_without_recipe_chart(self):
        cfg = dict(export_features=dict(scores=True, detail=False, trends=False, recipe=False),
                   step_times={'w': {'valve_phases': {'version': 2, 'steps': {'process': [6]}},
                                     'sensor_metrics': {'mean': {'T': {'6': 10}}},
                                     'step_timing': {'steps': {'6': {'seconds': 12}}}}})
        html = build_export_html('<script></script>', [], {}, cfg)
        self.assertIn('"valve_phases"', html)
        self.assertIn('"seconds": 12', html)
        self.assertIn('"sensor_metrics"', html)

    def test_independent_trend_features(self):
        for differences, equipment, recipe in itertools.product((False, True), repeat=3):
            cfg = dict(export_features=dict(scores=False, detail=False, trends=False,
                       chamber_means=False, differences=differences, equipment=equipment, recipe=recipe),
                       sensor_differences={'marker': 'CUSTOM_DIFFERENCE_DATA'},
                       step_times={'marker': 'CUSTOM_RECIPE_DATA'})
            html = build_export_html('<script></script>', [], {}, cfg)
            self.assertEqual('CUSTOM_DIFFERENCE_DATA' in html, differences or equipment)
            self.assertEqual('CUSTOM_RECIPE_DATA' in html, recipe)
            for key, value in [('differences', differences), ('equipment', equipment), ('recipe', recipe)]:
                self.assertRegex(html, '"'+key+r'"\s*:\s*'+str(value).lower())

    def test_selected_payloads_only(self):
        source = Path('src/app/static/index.html').read_text()
        for scores, trends, detail in itertools.product((False, True), repeat=3):
            cfg = dict(export_features=dict(scores=scores, trends=trends, detail=detail),
                       sensor_differences={'marker': 'UNIQUE_TRENDS'},
                       scatter_scores={'marker': 'UNIQUE_SCORES'},
                       offline_scores={'marker': 'UNIQUE_TRAIN'})
            html = build_export_html(source, [{'wafer_id': 'UNIQUE_WAFER'}],
                                     {'marker': 'UNIQUE_DETAIL'}, cfg)
            for marker, included in [('UNIQUE_TRENDS', trends), ('UNIQUE_SCORES', scores or detail),
                                     ('UNIQUE_TRAIN', scores or detail), ('UNIQUE_DETAIL', detail)]:
                self.assertEqual(marker in html, included, (scores, trends, detail, marker))
            self.assertIn('UNIQUE_WAFER', html)
            self.assertIn('sensor_differences', cfg)  # caller not mutated

    def test_basic_export_skips_archives_and_score_computation(self):
        # Exercise the actual endpoint body without starting the application's workers.
        tree = ast.parse(Path('src/app/server.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'api_export_html')
        fn.decorator_list = []
        ns = dict(Dict=dict, dt=dt, re=re, json=json, STATIC=Path('src/app/static'),
                  ENGINE=SimpleNamespace(load=lambda: None),
                  api_scatter=lambda **kw: {'points': [{'wafer_id': 'w1', 'ts': '2026-09-28T12:00:00'}]},
                  _lot_of=lambda w: 'lot', _model_id=lambda: 'model', _today=lambda: '2026-09-28',
                  _history_for_current_model=lambda: [], _display_rule=lambda: {},
                  _load_archive=Mock(side_effect=AssertionError('must not load archives')),
                  api_scatter_sensors=Mock(side_effect=AssertionError('must not compute scores')))
        for name in ['api_ui_config', 'api_sensors', 'api_status', 'api_get_retention', 'api_chamber_paths', 'api_models']:
            ns[name] = lambda: {}
        ns['Optional'] = Optional
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<endpoint>', 'exec'), ns)
        response = ns['api_export_html'](include_scores=False, include_trends=False, include_detail=False)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'w1', response.body)
        ns['_load_archive'].assert_not_called()
        ns['api_scatter_sensors'].assert_not_called()


    def test_chamber_detection_exports_means_independently(self):
        import numpy as np
        from unittest.mock import patch
        tree = ast.parse(Path('src/app/server.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'api_export_html')
        fn.decorator_list = []
        archive = dict(actual=np.array([[10., 20.], [30., 40.]]),
                       step=np.array([1, 2]), meta={'sensor_cols': ['p_test', 't_test']})
        ns = dict(Dict=dict, dt=dt, re=re, json=json, STATIC=Path('src/app/static'),
                  ENGINE=SimpleNamespace(load=lambda: None),
                  api_scatter=lambda **kw: {'points': [{'wafer_id': 'w1', 'ts': '2026-09-28T12:00:00'}]},
                  _lot_of=lambda w: 'lot', _model_id=lambda: 'model', _today=lambda: '2026-09-28',
                  _history_for_current_model=lambda: [], _display_rule=lambda: {},
                  _load_archive=Mock(return_value=archive),
                  _trend_summary=Mock(return_value=None),
                  _wafer_step_timing=Mock(return_value={}),
                  api_scatter_sensors=Mock(side_effect=AssertionError('must not compute scores')))
        for name in ['api_ui_config', 'api_sensors', 'api_status', 'api_get_retention', 'api_chamber_paths', 'api_models']:
            ns[name] = lambda: {}
        ns['Optional'] = Optional
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<endpoint>', 'exec'), ns)
        for differences, equipment, recipe in itertools.product((False, True), repeat=3):
            with patch('src.app.export_html.build_export_html', return_value='snapshot') as build:
                ns['api_export_html'](include_scores=False, include_detail=False, include_trends=False,
                    include_differences=differences, include_equipment=equipment, include_recipe=recipe)
                cfg = build.call_args.args[3]
            self.assertEqual(bool(cfg['step_times']), recipe)
            self.assertEqual(bool(cfg['sensor_differences']), differences or equipment)
            if differences or equipment:
                row = cfg['sensor_differences']['2026-09-28/w1']
                self.assertEqual(bool(row['series']), differences)
                self.assertEqual(bool(row['sensor_series']), differences or equipment)
        for trends in (False, True):
            with patch('src.app.export_html.build_export_html', return_value='snapshot') as build:
                ns['api_export_html'](include_scores=False, include_detail=False,
                    include_trends=trends, trend_metrics='std', include_chamber_means=True)
                cfg = build.call_args.args[3]
            row = cfg['sensor_differences']['2026-09-28/w1']
            stats = row['sensor_series'][0]['export_step_stats']['1']
            self.assertEqual(stats['mean'], 10.)
            self.assertEqual(set(stats), {'mean', 'std'} if trends else {'mean'})
            self.assertEqual(cfg['trend_export']['metrics'], ['std'])
            self.assertEqual(bool(cfg['step_times']), trends)
            if trends:
                self.assertEqual(cfg['step_times']['2026-09-28/w1']['sensor_metrics']['mean']['p_test']['1'], 10.)
                self.assertEqual(cfg['step_times']['2026-09-28/w1']['sensor_metrics']['min']['p_test']['2'], 30.)
                self.assertEqual(cfg['step_times']['2026-09-28/w1']['sensor_metrics']['std']['p_test']['1'], 0.)
            if not trends:
                self.assertEqual(row['series'], [])
            cfg['sensor_differences']['test_marker'] = 'DETECTION_MEANS_PRESENT'
            html = build_export_html('<script></script>', [], {}, cfg)
            self.assertIn('DETECTION_MEANS_PRESENT', html)
        archive['meta']['sensor_cols'] = ['s_010_PV', 's_011_PV']
        with patch('src.app.export_html.build_export_html', return_value='snapshot') as build:
            ns['api_export_html'](include_scores=False, include_detail=False, include_trends=False,
                include_differences=True, trend_pairs=json.dumps([['s_010_PV', 's_011_PV']]))
            row = build.call_args.args[3]['sensor_differences']['2026-09-28/w1']
        self.assertEqual(row['series'][-1]['left'], 's_010_PV')
        self.assertEqual(row['series'][-1]['export_step_stats']['1']['mean'], -10.)
        self.assertEqual({s['sensor'] for s in row['sensor_series']}, {'s_010_PV','s_011_PV'})
        from starlette.exceptions import HTTPException
        ns['HTTPException'] = HTTPException
        for bad in ['null', '{}', '[1]', '[["A","A"]]', 'not-json']:
            with self.assertRaises(HTTPException) as caught:
                ns['api_export_html'](trend_pairs=bad)
            self.assertEqual(caught.exception.status_code, 400)
        ns['api_scatter_sensors'].assert_not_called()


class CompactTrendTests(unittest.TestCase):
    def test_individual_steps_and_selected_metrics(self):
        from src.app.sensor_differences import compact_trend_export
        stats = {'0': dict(mean=999,min=-999,max=999,std=99,count=1),
                 '1': dict(mean=2,min=1,max=3,std=1,count=2),
                 '6': dict(mean=100,min=90,max=110,std=5,count=3)}
        row = {'series':[dict(left='A',right='B',step_stats=stats,values=[2,100])],
               'sensor_series':[dict(sensor='A',step_stats=stats)]}
        for size in (1,2,3,4):
            for metrics in itertools.combinations(('mean','min','max','std'),size):
                result=compact_trend_export(row,metrics)
                self.assertEqual(result['process_steps'],[1,6])
                for group in ('series','sensor_series'):
                    compact=result[group][0]
                    self.assertEqual(compact['export_step_stats'],
                                     {k:{m:stats[k][m] for m in metrics} for k in ('1','6')})
                    self.assertNotIn('step_stats',compact)
                    self.assertNotIn('values',compact)
        missing=compact_trend_export({'series':[{'step_stats':{}}]},['min'])
        self.assertEqual(missing['process_steps'],[])
        self.assertEqual(missing['series'][0]['export_step_stats'],{})
        self.assertIn('step_stats',row['series'][0])
