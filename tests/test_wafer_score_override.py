"""Filename policy integration without starting the app's inspection workers."""
import ast
import csv
import io
import json
import math
import tempfile
import threading
import time
import traceback
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
from src.config import IOConfig
from src.wafer_score_override import AUDIT_FIELDS, effective_wafer_score, normalize_override, override_from_npz, artifact_override, apply_selected_override, override_identity

RULE = dict(version=1, match_field='source_filename', match_type='substring',
            case_sensitive=True, patterns=['BAD'], score_mode='threshold_multiplier',
            threshold_multiplier=1.01)
ROOT = Path(__file__).resolve().parents[1]


def functions(path, names, ns):
    tree = ast.parse((ROOT/path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


class OverrideTests(unittest.TestCase):
    def test_filename_only_case_and_windows_paths(self):
        for source, matched in [(r'C:\data\BAD_001.log', True),
                                ('/BAD/fine.log', False), ('/data/bad.log', False),
                                ('/data/BAD.log', True), ('/data/fine.log', False)]:
            result = effective_wafer_score(dict(wafer_id='BAD', source_path=source), 90, 4.264, RULE)
            self.assertEqual(result['score_overridden'], matched, source)
            self.assertEqual(result['model_score_raw'], 90)
            self.assertEqual(result['effective_score_raw'], 4.264*1.01 if matched else 90)
        self.assertTrue(effective_wafer_score(dict(wafer_id='BAD'), 1, 4, RULE)['score_overridden'])
        self.assertTrue(effective_wafer_score(dict(wafer_id='bad'), 1, 4, dict(RULE, case_sensitive=False))['score_overridden'])

    def test_invalid_or_old_artifacts_are_noops(self):
        for rule in (None, {}, dict(RULE, patterns=[]), dict(RULE, version=9),
                     dict(RULE, threshold_multiplier=float('nan'))):
            self.assertFalse(effective_wafer_score(dict(wafer_id='BAD'), 2, 4, rule)['score_overridden'])
        for threshold in (0, None, float('inf'), -1):
            self.assertFalse(effective_wafer_score(dict(wafer_id='BAD'), 2, threshold, RULE)['score_overridden'])
        for raw in ({}, {'wafer_score_override_json': 'broken'}, {'wafer_score_override_json': np.array(['a','b'])}):
            self.assertIsNone(override_from_npz(raw))

    def test_model_json_precedes_standalone_and_selected_chambers_stay_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = folder / 'best_search.json'
            config.write_text(json.dumps({'best': {'wafer_score_override': RULE}}))
            sidecar = folder / 'wafer_score_override.json'
            for value in (None, dict(RULE,patterns=['OTHER']), {'patterns': []}):
                sidecar.write_text(json.dumps(value))
                present, rule = artifact_override(folder, config)
                self.assertTrue(present)
                self.assertEqual(rule, normalize_override(RULE))
                self.assertEqual(effective_wafer_score(dict(source_path=r'C:\logs\BAD.log'),20,4,rule)['override_ratio'],1.01)
            config.write_text(json.dumps({'best': {'wafer_score_override': None}}))
            sidecar.write_text(json.dumps(RULE))
            self.assertEqual(artifact_override(folder, config), (True,None))
            config.write_text(json.dumps({'best': {'threshold': 4}}))
            self.assertEqual(artifact_override(folder, config, include_standalone=False), (False,None))
            self.assertEqual(artifact_override(folder, config), (True,normalize_override(RULE)))
            sidecar.unlink()
            selected = folder / 'selected.json'
            selected.write_text(json.dumps({'best': {'threshold': 4}}))
            config.write_text(json.dumps({'best': {'wafer_score_override': RULE}}))
            self.assertEqual(artifact_override(folder, selected), (False,None))

    def test_selected_total_override_preserves_sensor_values_and_missing_scores(self):
        result = dict(total_score=20., total_threshold=4., total_ratio=5.,
                      sensors=[dict(sensor='P1', ratio=8.)])
        selected = apply_selected_override(result, dict(wafer_id='BAD.log'), RULE)
        self.assertEqual(selected['total_ratio'], 1.01)
        self.assertEqual(selected['total_score'], 4.04)
        self.assertEqual(selected['selected_score_before_override'], 20.)
        self.assertIs(selected['sensors'], result['sensors'])
        self.assertEqual(result['total_ratio'], 5.)
        self.assertIs(apply_selected_override(result, dict(wafer_id='normal'), RULE), result)
        missing = dict(result, total_ratio=None, total_threshold=0.)
        self.assertIs(apply_selected_override(missing, dict(wafer_id='BAD'), RULE), missing)

    def test_legacy_history_identity_and_two_decimal_multipliers(self):
        legacy=dict(patterns=['BAD'],threshold_multiplier=1.01,case_sensitive=True)
        self.assertEqual(override_identity(RULE),legacy)
        self.assertEqual(override_identity(normalize_override(RULE)),legacy)
        self.assertNotEqual(override_identity(dict(version=2,rules=[dict(pattern='A',threshold_multiplier=1.02),dict(pattern='B',threshold_multiplier=1.15)])),legacy)
        ns=self.server()
        for value in (1.01,1.02,1.15,4.1):
            self.assertEqual(ns['_floor2'](value),value)
        self.assertEqual(ns['_floor2'](1.159),1.15)

    def test_v2_order_default_case_and_invalid_rules(self):
        rule = dict(version=2, rules=[dict(pattern='WAFER_A',threshold_multiplier=1.02),
                                     dict(pattern='WAFER_B',threshold_multiplier=1.15),
                                     dict(pattern='DEFAULT'), dict(pattern='WAFER',threshold_multiplier=2.)])
        for name, expected in [('WAFER_A',1.02),('WAFER_B',1.15),('DEFAULT',1.01)]:
            result=effective_wafer_score(dict(source_path='C:/logs/'+name+'.log'),20,4,rule)
            self.assertEqual(result['override_ratio'],expected)
            self.assertEqual(result['matched_pattern'],name)
            self.assertEqual(result['effective_score_raw'],4*expected)
        self.assertFalse(effective_wafer_score(dict(wafer_id='wafer_a'),20,4,rule)['score_overridden'])
        self.assertEqual(effective_wafer_score(dict(wafer_id='wafer_a'),20,4,dict(rule,case_sensitive=False))['override_ratio'],1.02)
        self.assertFalse(effective_wafer_score(dict(source_path='/WAFER_A/normal.log'),20,4,rule)['score_overridden'])
        for invalid in ['', ' ', None, 123]:
            self.assertIsNone(normalize_override(dict(version=2,rules=[dict(pattern=invalid)])))
        for invalid in [0,-1,float('nan'),float('inf'),float('-inf'),None,True,'invalid']:
            self.assertIsNone(normalize_override(dict(version=2,rules=[dict(pattern='A',threshold_multiplier=invalid)])))
        self.assertEqual(normalize_override(normalize_override(rule)),normalize_override(rule))

    def test_v2_artifact_paths_and_history_roundtrip(self):
        rule=dict(version=2,rules=[dict(pattern='A',threshold_multiplier=1.02),dict(pattern='B',threshold_multiplier=1.15)])
        canonical=normalize_override(rule)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);config=folder/'search_best.json'
            config.write_text(json.dumps({'best':{'wafer_score_override':rule}}))
            self.assertEqual(artifact_override(folder)[1],canonical)
            config.unlink();(folder/'wafer_score_override.json').write_text(json.dumps(rule))
            self.assertEqual(artifact_override(folder)[1],canonical)
            np.savez(folder/'meta.npz',wafer_score_override_json=json.dumps(rule))
            with np.load(folder/'meta.npz') as meta:
                self.assertEqual(override_from_npz(meta),canonical)
        ns=self.server();ns['ENGINE'].wafer_score_override=canonical
        for name,ratio in [('A',1.02),('B',1.15)]:
            result=dict(wafer_id=name,**ns['_score_view'](name,90,dict(wafer_id=name,source_path='C:/data/'+name+'.log')))
            restored=json.loads(json.dumps(ns['_hist_entry'](result,'m')))
            ns['_RELIVE_CACHE']=dict(key=None,rows=None)
            live=ns['_relive_rows']([restored])[0]
            self.assertEqual(live['ratio_thr'],ratio)
            self.assertEqual(live['matched_pattern'],name)
            self.assertEqual(live['model_score_raw'],90)

    def test_npz_roundtrip_and_normalization(self):
        buff = io.BytesIO()
        np.savez(buff, wafer_score_override_json=json.dumps(RULE))
        buff.seek(0)
        with np.load(buff, allow_pickle=False) as data:
            rule = override_from_npz(data)
        self.assertEqual(rule, normalize_override(RULE))
        self.assertEqual([r['pattern'] for r in normalize_override(dict(RULE, patterns=' BAD, OK, BAD '))['rules']], ['BAD', 'OK'])

    def server(self):
        engine = SimpleNamespace(wafer_score_override=RULE, ready=True, threshold=4.264,
                                 thresholds={}, using=lambda e: nullcontext())
        bands = dict(warn=1.2, crit=1.5)
        ns = dict(Dict=Dict, List=List, np=np, math=math, ENGINE=engine, STATE={},
                  AUDIT_FIELDS=AUDIT_FIELDS, effective_wafer_score=effective_wafer_score,
                  _recipe_threshold=lambda wid: ('29a', 4.264),
                  _recipe_base_decision=lambda wid: ('29a', 2.132, 4.264),
                  _end_ts_and_date=lambda wid: ('2026-10-23T12:00:00', '2026-10-23'),
                  _lot_of=lambda wid: 'lot', _isnum=lambda v: isinstance(v, (int,float)) and math.isfinite(v),
                  _score_mode=lambda: 'raw', _display_rule=lambda: dict(bands=bands, mode='raw', thr_scale={'all':1}),
                  _model_id=lambda: 'm', _recipe_mult_sig=lambda: (), _disp_sig=lambda r: (),
                  _RELIVE_CACHE=dict(key=None,rows=None), _engine_for_wafer=lambda *a: engine,
                  _recipe_decision_by_name=lambda r: (4.264,4.264), _disp_lookup=lambda *a: 1,
                  _plot_threshold_dict=lambda: {})
        return functions('src/app/server.py', {'_score_view','_floor2','_hist_entry','_relive_rows',
                         '_disp_band','_archive_rec','_waferaccum_threshold','_weighted_view_results'}, ns)

    def test_final_score_and_history_restart_keep_exact_ratio(self):
        ns=self.server()
        rec=dict(wafer_id='w', source_path='/tmp/BAD.log')
        result=dict(wafer_id='w', **ns['_score_view']('w', 0.056, rec))
        self.assertEqual(result['score'], 1.01)
        self.assertEqual(result['verdict'], 'NG')
        self.assertEqual(result['score_raw'], 4.264*1.01)
        history=ns['_hist_entry'](result,'m')
        restored=json.loads(json.dumps(history))
        live=ns['_relive_rows']([restored])[0]
        self.assertEqual(live['ratio_thr'],1.01)
        self.assertEqual(live['model_score_raw'],.056)
        self.assertEqual(live['verdict'],'NG')
        self.assertEqual(live['band'],'caution')
        # Display customization and legacy weighting must not undo the policy.
        ns['_score_mode']=lambda:'weighted'
        result['score_w_raw']=.01
        self.assertEqual(ns['_weighted_view_results']([result])[0]['score'],1.01)

    def test_archive_csv_and_index_rebuild_keep_original_residuals(self):
        ns = self.server()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); day = root/'2026-10-23'; day.mkdir()
            arrays = np.array([[.125, 2.5], [.25, 1.5]], dtype=float)
            rec = dict(wafer_id='w', source_path='/tmp/BAD.log', step=[1, 2])
            result = dict(wafer_id='w', n_flag_sensor=1, **ns['_score_view']('w', 3.25, rec))
            def save(item, timeout):
                path, data = item
                np.savez_compressed(path, **data)
            ns['ENGINE'].model = SimpleNamespace(sensor_cols=['T1', 'P1'])
            ns.update(Path=Path, csv=csv, json=json, time=time, traceback=traceback,
                      RX=SimpleNamespace(_compute=lambda *a: (arrays, arrays*.5, arrays)),
                      _valve_open_map=lambda rec: ({}, 2),
                      _load_app_config=lambda: dict(archive_rca_source=False),
                      _archive_write_path=lambda d,w: day/(w+'.npz'),
                      _archive_queue=lambda: SimpleNamespace(put=save), ARCHIVE_DIR=root,
                      _HIST_IDX=dict(built=False), _HIST_LOCK=threading.Lock(),
                      _natkey=lambda w:w, _score_thr_for=lambda rc:None)
            functions('src/app/server.py', {'_archive_wafer','_hist_index_ensure','_write_predictions_csv'},ns)
            ns['_archive_wafer'](rec,result)
            with np.load(day/'w.npz',allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['app_score_level'],arrays)
                np.testing.assert_array_equal(archive['z'],arrays)
                meta=json.loads(bytes(archive['meta_json']).decode())
            self.assertEqual(meta['effective_score_raw'],4.264*1.01)
            self.assertEqual(meta['model_score_raw'],3.25)
            ns['_hist_index_ensure']()
            restored=list(ns['_HIST_IDX']['rows'].values())
            self.assertEqual(len(restored),1)
            self.assertEqual(ns['_relive_rows'](restored)[0]['ratio_thr'],1.01)
            ns['_write_predictions_csv']([result],root/'predictions.csv',inspected_at='now')
            with (root/'predictions.csv').open(encoding='utf-8-sig') as stream:
                row=next(csv.DictReader(stream))
            self.assertEqual(float(row['effective_score_raw']),4.264*1.01)
            self.assertEqual(row['score_overridden'],'True')
            self.assertEqual(float(row['model_score_raw']),3.25)

    def test_unmatched_display_still_floors_original_score(self):
        ns=self.server()
        v=ns['_score_view']('fine', 2.98765, dict(wafer_id='fine',source_path='/BAD/fine.log'))
        self.assertEqual(v['score'], math.floor(2.98765/4.264*100)/100)
        self.assertEqual(v['score_raw'],2.987)
        self.assertFalse(v['score_overridden'])

    def test_graph_uses_effective_decision_not_base(self):
        ns=self.server()
        result=ns['_score_view']('w',90,dict(wafer_id='w',source_path='/tmp/BAD.log'))
        rec=ns['_archive_rec'](dict(meta=dict(wafer_id='w',recipe='29a',**{k:result[k] for k in AUDIT_FIELDS}),
                                    step=np.array([1,2]),actual=np.zeros((2,1))))
        # Stored policy must survive restart, even without the original source file.
        ns['ENGINE'].wafer_score_override=None
        threshold=ns['_waferaccum_threshold']('w',rec)
        self.assertEqual(threshold['base_threshold_by_recipe']['29a'],4.264)
        self.assertEqual(threshold['threshold_recipe_multipliers']['29a'],1)
        self.assertEqual(rec['effective_score_raw']/threshold['default'],1.01)

    def test_json_priority_and_chamber_rule_isolation(self):
        tree=ast.parse((ROOT/'src/app/server.py').read_text())
        engine=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine')
        cfg=next(n for n in engine.body if isinstance(n,ast.FunctionDef) and n.name=='_build_cfg')
        ns=dict(IOConfig=IOConfig, replace=replace, json=json, normalize_override=normalize_override, artifact_override=artifact_override,
                _find_recipe_thresholds=lambda r:{}, _detect_score_norm=lambda *a:'none',
                _recipe_thr_map=lambda v:{}, _is_num=lambda v:isinstance(v,(int,float)), _find_threshold=lambda r:4)
        exec(compile(ast.Module(body=[cfg],type_ignores=[]),'cfg','exec'),ns)
        # Execute the actual model-load fallback block (without loading the model runtime).
        load=next(n for n in engine.body if isinstance(n,ast.FunctionDef) and n.name=='load')
        fallback=next(n for n in ast.walk(load) if isinstance(n,ast.If) and
                      ast.unparse(n.test)=='not self._override_in_json')
        with tempfile.TemporaryDirectory() as tmp:
            roots=[]
            for i,payload in enumerate((dict(wafer_score_override=RULE), {}, dict(wafer_score_override=None))):
                folder=Path(tmp)/str(i);folder.mkdir()
                (folder/'best_search.json').write_text(json.dumps(dict(best=dict(threshold=4,**payload))))
                (folder/'wafer_score_override.json').write_text('null')
                obj=SimpleNamespace(artifact=None,art=folder, model=SimpleNamespace(wafer_score_override=dict(RULE,patterns=['NPZ'])))
                ns['_build_cfg'](obj)
                ns['self']=obj
                exec(compile(ast.Module(body=[fallback],type_ignores=[]),'fallback','exec'),ns)
                roots.append(obj)
            self.assertEqual([r['pattern'] for r in roots[0].wafer_score_override['rules']],['BAD'])
            self.assertEqual([r['pattern'] for r in roots[1].wafer_score_override['rules']],['NPZ'])
            self.assertIsNone(roots[2].wafer_score_override)
            self.assertFalse(effective_wafer_score(dict(wafer_id='BAD'),1,4,roots[1].wafer_score_override)['score_overridden'])

    def test_curve_scales_only_display_and_keeps_final_sample(self):
        ns=dict(np=np)
        functions('src/recon_explain.py',{'_final_score_curve','wafer_series'},ns)
        curve=np.array([0.,2.,4.]);original=curve.copy()
        np.testing.assert_allclose(ns['_final_score_curve'](curve,8.08),[0,4.04,8.08])
        np.testing.assert_array_equal(curve,original)
        np.testing.assert_array_equal(ns['_final_score_curve'](curve,None),original)
        np.testing.assert_allclose(ns['_final_score_curve'](np.zeros(3),4.04),[0,0,4.04])
        n=4000
        z=np.ones((n,1));steps=np.ones(n,dtype=int)
        ns.update(_plot_time_filter=lambda *a:(np.ones(n,dtype=bool),np.arange(n),np.arange(n),steps,[],[]),
                  _wafer_decision_accum_series=lambda z,s:np.arange(1,n+1,dtype=float),
                  _score_norm_denominator=lambda *a:1, _display_scale_threshold_for_record=lambda *a:4,
                  _display_threshold_for_record=lambda *a:1, _lookup_ok_mean=lambda *a,**kw:None,
                  _lookup_ok_band=lambda *a,**kw:(None,None),_step_marks=lambda *a:[])
        data=ns['wafer_series'](None,{},z,final_score=4.04)
        self.assertEqual(data['cum'][-1],1.01)
        self.assertEqual(data['t'][-1],3999)
        self.assertEqual(len(data['t']),len(data['step_at']))
        np.testing.assert_array_equal(z,np.ones((n,1)))


if __name__=='__main__':
    unittest.main()
