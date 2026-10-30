"""Run with FastAPI, onnx and onnxruntime installed; generated models need no customer data."""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from src.app.chamber_models import discover_chamber_models, artifact_digest, EngineRouter


def make_model(folder, ch, flat=False):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    folder = Path(folder)
    d = folder if flat else folder / ('DSV%d' % ch)
    d.mkdir(parents=True, exist_ok=True)
    prefix = 'DSV%d-' % ch if flat else ''
    model_path = d / (prefix + 'tcn_forecasr.onnx')  # accept user's filename spelling
    # Output first IO channel times chamber number: dynamic batch and time dimensions.
    graph = helper.make_graph([
        helper.make_node('Gather', ['x', 'indices'], ['io'], axis=1),
        helper.make_node('Mul', ['io', 'factor'], ['y'])], 'fixture',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 3, 'time'])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 1, 'time'])],
        [numpy_helper.from_array(np.array([0], dtype=np.int64), 'indices'),
         numpy_helper.from_array(np.array(float(ch), dtype=np.float32), 'factor')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    onnx.save(model, model_path)
    np.savez(d / (prefix + 'recon_meta.npz'),
             io_cols=['IO%d' % ch], input_sensor_cols=['P%d' % ch], sensor_cols=['P%d' % ch],
             target_indices=[0], h_indices=np.array([], dtype=int), n_steps_global=0,
             schema_mode='union', input_sen_mean=[0.], input_sen_std=[1.],
             sen_mean=[0.], sen_std=[1.], resid_abs_limit=[1.], resid_sigma=[1.],
             per_step_limit=False, per_step_sigma=False, w=2, h=2, stride=2,
             n_in=3, future_h_mode='off', virtual_sensor_mode='off', use_step_onehot=False)
    config = {'best': {'params': {'graph_score_enabled': False, 'io_name_regex': r'^IO\d+$', 'sensor_keep_prefixes': ['P']}, 'val': {'threshold': 10. * ch}}}
    (d / (prefix + 'search_best.json')).write_text(json.dumps(config))
    return model_path


class DiscoveryTests(unittest.TestCase):
    def test_layouts_and_digest(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td)/'sub', Path(td)/'moved'
            for ch in range(1, 7):
                make_model(a, ch)
                make_model(b, ch)
            aa, bb = discover_chamber_models(a), discover_chamber_models(b)
            self.assertEqual(list(aa), ['dsv%d' % i for i in range(1,7)])
            self.assertEqual(artifact_digest(aa), artifact_digest(bb))
            old = artifact_digest(bb)
            bb['dsv6'].config.write_text('{"threshold": 999}')
            self.assertNotEqual(old, artifact_digest(bb))

    def test_missing_or_ambiguous_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            make_model(td, 1)
            make_model(td, 2)
            specs = discover_chamber_models(td)
            specs['dsv2'].meta.unlink()
            with self.assertRaisesRegex(ValueError, 'DSV2'):
                discover_chamber_models(td)
            make_model(td, 2)
            (Path(td)/'DSV2'/'other.onnx').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, '하나만'):
                discover_chamber_models(td)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['IO_PROBE_CHILD'] = '1'
        os.environ['IO_ORT_THREADS'] = '1'
        os.environ['IO_ORT_BATCH'] = '2048'
        from src.app import server
        cls.s = server

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.models = self.folder/'models'
        for ch in (1,2):
            make_model(self.models, ch)
        self.config = patch.object(self.s, '_load_app_config', return_value={})
        self.config.start(); self.addCleanup(self.config.stop)
        for name in ('_save_app_config','_save_file_meta','_tag_file_meta'):
            patcher = patch.object(self.s, name)
            patcher.start(); self.addCleanup(patcher.stop)
        font = patch.object(self.s.RX, '_setup_font')
        font.start(); self.addCleanup(font.stop)
        router = EngineRouter(self.s.Engine())
        patcher = patch.object(self.s, 'ENGINE', router)
        patcher.start(); self.addCleanup(patcher.stop)
        info = router.select(self.models)
        self.assertTrue(info['ready'], info)
        self.engine = router
        patcher = patch.object(self.s, '_chamber', side_effect=lambda wid: 'dsv'+str(wid)[0])
        patcher.start(); self.addCleanup(patcher.stop)
        self.s._RELIVE_CACHE['key'] = None
        self.s._RELIVE_CACHE['rows'] = None

    def rec(self, ch, index=0):
        t = 4 + index
        return dict(wafer_id='%d_%d' % (ch,index), chamber='dsv%d' % ch, recipe='x',
                    sensors=np.full((t,1), 5.), sensor_cols=['P%d' % ch],
                    io=np.ones((t,1)), io_cols=['IO%d' % ch], step=np.ones(t,dtype=int))

    def test_sensor_difference_batch_preserves_every_wafer_and_validates(self):
        from src.app.sensor_differences import PAIRS, difference_series
        cols = list(dict.fromkeys(name for pair in PAIRS for name in pair))
        arch = dict(actual=np.tile(np.arange(6), (7,1)), step=np.arange(1,8), meta={'sensor_cols': cols})
        req = self.s.ScatterSensorsReq(model_id=self.s._model_id(),
            wafers=[{'date': '2026-09-08', 'wid': '1_0'}, {'date': '2026-09-08', 'wid': '2_0'}])
        with patch.object(self.s, '_history_for_current_model', return_value=[{'wafer_id':'1_0'}, {'wafer_id':'2_0'}]), \
             patch.object(self.s, '_trend_summary', side_effect=[{'summary': difference_series(arch), 'meta': arch['meta']}, None]) as load:
            result = self.s.api_scatter_differences(req)
            self.assertEqual(len(result['wafers']), 2)
            self.assertEqual(result['wafers'][0]['series'][0]['values'], [-1., -1.])
            self.assertEqual(result['wafers'][0]['date'], '2026-09-08')
            self.assertTrue(result['wafers'][1]['missing'])
            self.assertEqual(load.call_count, 2)
        req.model_id = 'invalid-model'
        with self.assertRaises(self.s.HTTPException) as error:
            self.s.api_scatter_differences(req)
        self.assertEqual(error.exception.status_code, 409)
        req.model_id = self.s._model_id()
        req.wafers = req.wafers * 33
        with self.assertRaises(self.s.HTTPException) as error:
            self.s.api_scatter_differences(req)
        self.assertEqual(error.exception.status_code, 400)

    def test_trend_http_fast_json_gzip_and_validation(self):
        from fastapi.testclient import TestClient
        from src.app.sensor_differences import difference_series
        arch = dict(actual=np.arange(96*52, dtype=float).reshape(52,96),
                    step=np.repeat(np.arange(1,27),2),
                    meta={'sensor_cols':['S'+str(i) for i in range(96)],
                          'step_timing':{'version':2,'steps':{'1':{'seconds':4}}}})
        summary = {'summary':difference_series(arch), 'meta':arch['meta']}
        req = dict(model_id=self.s._model_id(), wafers=[{'date':'2026-10-20','wid':'1_0'}])
        client = TestClient(self.s.app)
        with patch.object(self.s, '_history_for_current_model', return_value=[{'wafer_id':'1_0'}]), \
             patch.object(self.s, '_trend_summary', return_value=summary):
            for url in ('/api/scatter/differences','/api/scatter/step-times'):
                compressed = client.post(url,json=req,headers={'Accept-Encoding':'gzip'})
                plain = client.post(url,json=req,headers={'Accept-Encoding':'identity'})
                self.assertEqual(compressed.status_code,200)
                self.assertEqual(compressed.headers.get('content-encoding'),'gzip')
                self.assertEqual(compressed.json(),plain.json())
                self.assertNotIn('content-encoding',plain.headers)
                self.assertEqual(len(compressed.json()['wafers']),1)
                invalid=dict(req,model_id='stale')
                self.assertEqual(client.post(url,json=invalid).status_code,409)
                invalid=dict(req,wafers=req['wafers']*65)
                self.assertEqual(client.post(url,json=invalid).status_code,400)

    def test_real_onnx_batch_matches_individual_and_uses_one_run_per_chamber(self):
        recs = [self.rec(1,0), self.rec(2,0), self.rec(1,1), self.rec(2,1)]
        expected = [self.engine.for_chamber(r['chamber']).model.predict(copy.deepcopy(r)) for r in recs]
        calls = []
        for ch, eng in self.engine.root.chamber_engines.items():
            orig = eng.model._run_mem_safe
            def counted(x, ch=ch, orig=orig):
                calls.append((ch, len(x)))
                return orig(x)
            eng.model._run_mem_safe = counted
        self.s._prefill_batch(recs)
        self.assertEqual([c[0] for c in calls], ['dsv1','dsv2'])
        for r, want in zip(recs, expected):
            actual = self.engine.for_chamber(r['chamber']).model.predict(r)
            np.testing.assert_allclose(actual, want, rtol=1e-6)
            np.testing.assert_allclose(actual, 1. if r['chamber']=='dsv1' else 2.)
            self.s._assess(r)  # uses cached predictions, no extra ONNX call
        self.assertEqual(len(calls), 2)

    def test_mixed_minmax_and_standard_models_route_and_batch_correctly(self):
        meta = self.models / 'DSV1' / 'recon_meta.npz'
        with np.load(meta, allow_pickle=True) as data:
            values = dict(data)
        values.update(normalization_mode='minmax', normalization_min=0., normalization_max=1.,
                      io_min=[1.], io_max=[2.], io_scale=[1.],
                      input_sen_min=[0.], input_sen_max=[10.], input_sen_scale=[10.],
                      sen_min=[10.], sen_max=[20.], sen_scale=[10.])
        np.savez(meta, **values)
        self.assertTrue(self.engine.reload()['ready'])
        mid = self.s._model_id()
        self.assertTrue(mid.endswith('-minmax-v1'))
        recs = [self.rec(1), self.rec(2), self.rec(1, 1)]
        self.s._prefill_batch(recs)
        for rec in recs:
            eng = self.engine.for_chamber(rec['chamber'])
            np.testing.assert_allclose(eng.model.predict(rec),
                                       10. if rec['chamber'] == 'dsv1' else 2.)
            with self.engine.using(eng):
                self.assertEqual(self.s._model_id(), mid)

    def test_thresholds_and_reloaded_history_remain_per_chamber(self):
        rows = [dict(wafer_id='%d_0'%ch, chamber='dsv%d'%ch, recipe='x',
                     score_raw=15., threshold_raw=10.*ch, n_tp=4) for ch in (1,2)]
        with patch.object(self.s, '_recipe_tag', return_value='x'):
            self.assertEqual(self.s._recipe_base_decision('1_0')[1], 10.)
            self.assertEqual(self.s._recipe_base_decision('2_0')[1], 20.)
        result = self.s._relive_rows(rows)
        self.assertEqual([r['verdict'] for r in result], ['NG','OK'])
        mid = self.s._model_id()
        with self.engine.using(self.engine.for_chamber('dsv2')):
            self.assertEqual(mid, self.s._model_id())
        self.assertTrue(self.engine.reload()['ready'])
        self.assertEqual(mid, self.s._model_id())

    def test_mixed_folder_aligned_with_each_model_schema(self):
        data = self.folder/'data'; data.mkdir()
        for ch in (1,2):
            (data/('%d_0.log'%ch)).write_text('header\nx\ty\tPMCUNIT%d\nx\ty\tPMCUNIT12\n'%(ch+6))
        calls=[]
        def loader(folder,cfg,stems,cancel_check=None,include_trend_source=False):
            recs = [self.rec(int(wid[0])) for wid in sorted(stems)]
            calls.append(set(stems))
            return recs, recs[0]['io_cols'], recs[0]['sensor_cols'], 0
        with patch.object(self.s,'load_dataset',side_effect=loader), \
             patch.object(self.s,'_file_meta',side_effect=lambda wid: {'chamber':'dsv'+wid[0]}), \
             patch.object(self.s,'_recipe_tag',return_value='x'):
            recs, io, sensors, cov = self.s._load_aligned(data)
        self.assertEqual(calls, [{'1_0'},{'2_0'}])
        self.assertEqual([r['chamber'] for r in recs], ['dsv1','dsv2'])
        self.assertEqual(set(cov['chambers']), {'dsv1','dsv2'})
        self.assertEqual(io,['IO1','IO2'])

    def test_missing_chamber_and_invalid_selection_do_not_use_previous_model(self):
        with self.assertRaisesRegex(ValueError,'챔버 모델 없음'):
            self.engine.for_chamber('dsv6')
        old = self.engine.root
        bad = self.folder/'bad'; make_model(bad, 1)
        (bad/'DSV1'/'recon_meta.npz').unlink()
        info = self.engine.select(bad)
        self.assertFalse(info['ready'])
        self.assertIs(self.engine.root, old)

    def test_six_subfolders_inference_and_canonical_rca_files(self):
        from src.app.chamber_models import canonical_artifacts
        bundle=self.folder/'six'
        for ch in range(1,7):
            make_model(bundle,ch)
        self.assertTrue(self.engine.select(bundle)['ready'])
        self.assertEqual(len(self.engine.root.chamber_engines),6)
        recs=[self.rec(ch) for ch in range(1,7)]
        self.s._prefill_batch(recs)
        for rec in recs:
            ch=int(rec['chamber'][-1])
            np.testing.assert_allclose(self.engine.for_chamber(rec['chamber']).model.predict(rec),float(ch))
        spec=self.engine.root.chamber_artifacts['dsv6']
        with canonical_artifacts(spec) as canonical:
            self.assertEqual((canonical/'tcn_forecast.onnx').read_bytes(),spec.onnx.read_bytes())
            self.assertEqual((canonical/'recon_meta.npz').read_bytes(),spec.meta.read_bytes())
            self.assertEqual((canonical/'search_best.json').read_bytes(),spec.config.read_bytes())
        self.assertFalse(canonical.exists())

    def test_prefix_files_are_rejected_without_single_model_fallback(self):
        from src.app.chamber_models import has_chamber_layout
        flat=self.folder/'flat'
        make_model(flat,1,True)
        make_model(flat,2,True)
        self.assertEqual(discover_chamber_models(flat),{})
        self.assertFalse(has_chamber_layout(flat))
        self.assertFalse(self.s._has_model(flat))
        previous=self.engine.root
        self.assertFalse(self.engine.select(flat)['ready'])
        self.assertIs(self.engine.root,previous)

    def test_single_model_folder_still_applies_to_all_chambers(self):
        info=self.engine.select(self.models/'DSV1')
        self.assertTrue(info['ready'])
        self.assertEqual(info['chamber_models'],{})
        self.assertIs(self.engine.for_chamber('dsv6'),self.engine.root)
        rec=self.rec(1)
        self.s._prefill_batch([rec])
        np.testing.assert_allclose(self.engine.model.predict(rec),1.)

    def test_c2_migration_retags_already_processed_logs(self):
        data=self._integration_storage()
        self.s._FILE_META['1_0']={'end_ts':'2026-09-08T00:12:03','chamber':'dsv6','chamber_cell':'C3',
                                    'lot_c':self.s._lot_cell_tag()}
        self.s._convert_logs_in(data,skip_stems={'1_0'})
        self.assertEqual(self.s._FILE_META['1_0']['chamber'],'dsv1')
        self.assertEqual(self.s._FILE_META['1_0']['chamber_cell'],'C2')

    def test_bundle_model_change_hides_old_history(self):
        mid=self.s._model_id()
        rows=[{'model':mid,'wafer_id':'1_0','chamber':'dsv1','score_raw':15.,'recipe':'x'}]
        spec=self.engine.root.chamber_artifacts['dsv2']
        spec.config.write_text(json.dumps({'best':{'val':{'threshold':99.}}}))
        self.assertTrue(self.engine.reload()['ready'])
        with patch.object(self.s,'_read_history',return_value=rows):
            self.assertEqual(self.s._history_for_current_model(),[])

    def test_context_isolation_and_restore(self):
        barrier = threading.Barrier(2)
        values=[]
        def worker(ch):
            with self.engine.pinned(), self.engine.using(self.engine.for_chamber(ch)):
                barrier.wait(timeout=3)
                values.append((ch,self.engine.threshold))
        threads=[threading.Thread(target=worker,args=(ch,)) for ch in ('dsv1','dsv2')]
        for t in threads: t.start()
        for t in threads: t.join(timeout=4)
        self.assertEqual(sorted(values), [('dsv1',10.),('dsv2',20.)])
        self.assertIsNone(self.engine._scope.get())

    def _integration_storage(self):
        paths = {name: self.folder/name for name in ('DATA_DIR','ARCHIVE_DIR','REPORT_DIR')}
        paths.update(AUTOPILOT_STATE=self.folder/'auto.json', NTP_FILE=self.folder/'ntp.json',
                     _FILE_META={}, _RECIPE_TAGS={}, _NTP={},
                     _HIST_IDX={'built':False,'rows':{},'by_stem':{},'sorted':None},
                     STATE={'folder':'', 'results':[], 'recs':{}, 'progress':{}, 'progress_bg':{}},
                     AUTOPILOT=None)
        patcher=patch.multiple(self.s, **paths); patcher.start(); self.addCleanup(patcher.stop)
        for name in ('_wait_if_low_memory','_release_memory'):
            patcher=patch.object(self.s,name); patcher.start(); self.addCleanup(patcher.stop)
        data=self.folder/'logs'; data.mkdir()
        for ch in (1,2):
            for index in (0,1):
                rows=['Date\tTime\tChamber\tProcessStep\tIO1\tIO2\tP1\tP2']
                for t in range(4):
                    unit=12 if t==1 else ch+6
                    rows.append('2026-09-08\t00:12:%02d\tPMCUNIT%d\t1\t1\t1\t5\t5'%(t,unit))
                f=data/('%d_%d.log'%(ch,index)); f.write_text('\n'.join(rows)+'\n')
                os.utime(f,(1,1))
        return data

    def test_model_switch_retains_same_wafer_results_and_archived_details(self):
        import shutil
        data = self._integration_storage()
        other = self.folder / 'other_models'
        shutil.copytree(self.models, other)
        for ch in (1,2):
            config = other / ('DSV%d' % ch) / 'search_best.json'
            payload = json.loads(config.read_text())
            payload['best']['val']['threshold'] = 100. * ch
            config.write_text(json.dumps(payload))
        with patch.object(self.s, '_recipe_tag', return_value='x'), \
             patch.object(self.s, '_probe_model_dir', return_value=(True,'')), \
             patch.object(self.s, '_inspect_cache_enabled', return_value=False):
            self.s.run_inspection(data)
            original = {r['wafer_id']:r['ratio_thr'] for r in self.s._history_for_current_model()}
            first_id = self.s._model_id()
            self.assertEqual(self.s.api_model_select(self.s.SelectReq(dir=str(other)))['restored_results'],0)
            self.s.run_inspection(data)
            second_id = self.s._model_id()
            self.assertNotEqual(first_id, second_id)
            changed = {r['wafer_id']:r['ratio_thr'] for r in self.s._history_for_current_model()}
            self.assertNotEqual(original, changed)
            self.assertEqual(len(list(self.s.ARCHIVE_DIR.rglob('*.npz'))),8)
            self.assertEqual(self.s.api_model_select(self.s.SelectReq(dir=str(self.models)))['restored_results'],4)
            self.assertEqual({r['wafer_id']:r['ratio_thr'] for r in self.s.STATE['results']},original)
            self.assertEqual(self.s._load_archive('2026-09-08','1_0')['meta']['threshold_raw'],10.)
            self.s._HIST_IDX.update(built=False,rows={},by_stem={},sorted=None)
            self.s.STATE.update(results=[],recs={},result_model=None)
            self.assertEqual(len(self.s.api_results()['results']),4)
            self.assertEqual(self.s.api_model_select(self.s.SelectReq(dir=str(other)))['restored_results'],4)
            self.assertEqual(self.s._load_archive('2026-09-08','1_0')['meta']['threshold_raw'],100.)

    def _assert_archives(self):
        archives=list(self.s.ARCHIVE_DIR.rglob('*.npz'))
        self.assertEqual(len(archives),4)
        for p in archives:
            with np.load(p,allow_pickle=False) as z:
                meta=json.loads(bytes(z['meta_json']).decode())
                ch=int(meta['chamber'][-1])
                np.testing.assert_allclose(z['expected'],float(ch))
                self.assertEqual(meta['threshold_raw'],10.*ch)
                self.assertEqual(meta['model'],self.s._model_id())
        # Simulate restart: no in-memory metadata, rebuild history only from archives.
        self.s._FILE_META.clear()
        self.s._HIST_IDX.update(built=False,rows={},by_stem={},sorted=None)
        self.s._RELIVE_CACHE.update(key=None,rows=None)
        rows=self.s._history_for_current_model()
        self.assertEqual(len(rows),4)
        self.assertEqual({r['verdict'] for r in rows if r['chamber']=='dsv1'},{'NG'})
        self.assertEqual({r['verdict'] for r in rows if r['chamber']=='dsv2'},{'OK'})
        # Stored sensor schema still resolves for a direct archived detail request.
        response=self.s.api_archive_sensors_all('2026-09-08','2_0')
        self.assertTrue(response)

    def test_pmcunit7_and_dsv1_filters_inspect_same_logs(self):
        data=self._integration_storage()
        with patch.object(self.s,'_recipe_tag',return_value='x'):
            for chamber in ('PMCUNIT7','DSV1'):
                result=self.s.run_inspection(data,chamber_prefixes={chamber})
                self.assertEqual(result['n'],2,result)
                self.assertEqual({r['chamber'] for r in result['results']},{'dsv1'})

    def test_empty_filter_result_has_current_diagnostics(self):
        data=self._integration_storage()
        self.s.STATE['last_diag']={'failed':99}
        result=self.s.run_inspection(data,chamber_prefixes={'DSV6'})
        self.assertEqual(result['n'],0)
        self.assertEqual(result['diag']['unique'],4)
        self.assertEqual(result['diag']['filtered_excluded'],4)
        self.assertEqual(result['diag']['chambers_found'],['dsv1','dsv2'])
        self.assertEqual(self.s.STATE['last_diag']['failed'],0)
        self.assertEqual(self.s.STATE['last_diag'],result['diag'])

    def test_real_source_metadata_survives_uppercase_extension_and_missing_parent_path(self):
        # setUp mocks tagging to isolate inference tests; exercise the real implementation here.
        import ast
        module=ast.parse(Path(self.s.__file__).read_text())
        node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_tag_file_meta')
        namespace=dict(vars(self.s))
        stored={'wafer': {'chamber':'dsv1', 'chamber_cell':'C2'}}
        namespace['_file_meta']=lambda stem: stored.get(stem,{})
        namespace['_save_file_meta']=lambda stem,**kv: stored.setdefault(stem,{}).update(kv)
        exec(compile(ast.Module(body=[node],type_ignores=[]),self.s.__file__,'exec'),namespace)
        tag=namespace['_tag_file_meta']
        nested=self.folder/'nested'; nested.mkdir()
        source=nested/'wafer.LOG'
        source.write_text('a\tb\tc\nx\ty\tPMCUNIT7\nx\ty\tPMCUNIT12\n')
        tag(nested,'wafer',source_path=source)
        self.assertEqual(stored['wafer']['chamber'],'dsv1')
        tag(self.folder,'wafer')
        self.assertEqual(stored['wafer']['chamber'],'dsv1')

    def test_c2_quoted_tab_fields_and_utf16(self):
        for encoding in ('utf-8-sig','cp949','utf-16'):
            source=self.folder/(encoding+'.LOG')
            source.write_text('a\tb\tc\n"embedded\ttab"\ty\t"PMCUNIT7"\nx\ty\tPMCUNIT12\n',encoding=encoding)
            self.assertEqual(self.s._extract_chamber_from_log(source),'dsv1')

    def test_manual_full_log_to_archive_pipeline(self):
        data=self._integration_storage()
        with patch.object(self.s,'_recipe_tag',return_value='x'):
            result=self.s.run_inspection(data)
        self.assertEqual(result['n'],4,result)
        self._assert_archives()

    def test_app_sensor_scores_use_chamber_reference_and_skip_step_zero(self):
        data = self._integration_storage()
        with patch.object(self.s, '_recipe_tag', return_value='x'):
            self.s.run_inspection(data)
        for ch in (1, 2):
            folder = self.models / ('DSV%d' % ch) / '18_앱점수'
            folder.mkdir()
            meta = dict(process_steps=[0, 1], sensors=['P%d' % ch], score_topk=0,
                        score_resid_agg='sum', score_time_avg=False, score_final_avg=False,
                        additive_exact_match_expected=True)
            np.savez(folder/'step_sensor_scores.npz', process_steps=[0,1],
                     sensors=['P%d' % ch], train_ok_scores=[[[999.],[float(ch)]], [[999.],[float(ch)]]])
            (folder/'metadata.json').write_text(json.dumps(meta))
        for ch in (1, 2):
            response = self.s.api_archive_app_sensor_scores('2026-09-08', '%d_0' % ch)
            self.assertTrue(response['available'], response)
            self.assertFalse(response['legacy_precision'])
            self.assertEqual(response['sensors'][0]['sensor'], 'P%d' % ch)
            self.assertEqual(response['sensors'][0]['threshold_raw'], float(ch))
            self.assertEqual(response['excluded_steps'], [0])

    def test_v2_chambers_keep_independent_multipliers_in_totals_and_reports(self):
        self.test_app_sensor_scores_use_chamber_reference_and_skip_step_zero()
        from src.app.score_report import build_score_report
        for ch,ratio in [(1,1.02),(2,1.15)]:
            engine=self.s.ENGINE.for_chamber('dsv%d'%ch)
            engine.wafer_score_override=dict(version=2,rules=[dict(pattern='_0',threshold_multiplier=ratio)])
        for ch,ratio in [(1,1.02),(2,1.15)]:
            wid='%d_0'%ch;engine=self.s.ENGINE.for_chamber('dsv%d'%ch)
            detail=self.s.api_archive_app_sensor_scores('2026-09-08',wid)
            self.assertEqual(detail['total_ratio'],ratio)
            self.assertEqual(detail['matched_pattern'],'_0')
            scatter=self.s.api_scatter_step_scores(self.s.ScatterStepScoresReq(model_id=self.s._model_id(),steps=[1],sensors=['P%d'%ch],wafers=[dict(wid=wid,date='2026-09-08')]))['wafers'][0]
            self.assertEqual(scatter['ratio'],ratio)
            diagnostic=self.s.api_debug_wafer_score_override(wid)
            self.assertEqual(diagnostic['diagnostic_version'],'basename-override-v2')
            self.assertEqual(diagnostic['wafers'][0]['matched_rule']['threshold_multiplier'],ratio)
            report=build_score_report(wid,self.s._load_archive('2026-09-08',wid),engine.art,{},dict(wafer_score_override=engine.wafer_score_override))
            self.assertEqual(report['selections']['all_steps_sensors']['ratio'],ratio)
            self.assertEqual(report['selections']['all_steps_sensors']['matched_pattern'],'_0')

    def test_selected_override_scatter_detail_and_report_preserve_sensor_scores(self):
        self.test_app_sensor_scores_use_chamber_reference_and_skip_step_zero()
        engine = self.s.ENGINE.for_chamber('dsv1')
        before = self.s.api_archive_app_sensor_scores('2026-09-08', '1_0')
        engine.wafer_score_override = dict(patterns=['1_0'], threshold_multiplier=1.01)
        detail = self.s.api_archive_app_sensor_scores('2026-09-08', '1_0')
        self.assertEqual(detail['total_ratio'], 1.01)
        self.assertEqual(detail['sensors'], before['sensors'])
        req = self.s.ScatterStepScoresReq(model_id=self.s._model_id(), steps=[1], sensors=['P1'],
            wafers=[dict(wid='1_0', date='2026-09-08')])
        scatter = self.s.api_scatter_step_scores(req)['wafers'][0]
        self.assertEqual(scatter['ratio'], 1.01)
        self.assertEqual(scatter['scores']['P1'], before['sensors'][0]['ratio'])
        from src.app.score_report import build_score_report
        report = build_score_report('1_0', self.s._load_archive('2026-09-08', '1_0'),
            engine.art, {}, dict(wafer_score_override=engine.wafer_score_override))
        self.assertEqual(report['selections']['all_steps_sensors']['ratio'], 1.01)
        self.assertTrue(report['selections']['all_steps_sensors']['score_overridden'])
        diagnostic = self.s.api_debug_wafer_score_override('1_0')['wafers'][0]
        self.assertEqual(diagnostic['reason'], 'applied')
        self.assertEqual(diagnostic['expected_ratio'], 1.01)
        self.assertIn('1_0', diagnostic['basename'])
        engine.wafer_score_override = dict(patterns=['NOT_A_MATCH'],threshold_multiplier=1.01)
        self.assertEqual(self.s.api_debug_wafer_score_override('1_0')['wafers'][0]['reason'],'basename_not_matched')
        engine.wafer_score_override = None
        self.assertEqual(self.s.api_debug_wafer_score_override('1_0')['wafers'][0]['reason'],'rule_missing')
        self.assertEqual(self.s.api_scatter_step_scores(req)['wafers'][0]['ratio'], before['total_ratio'])

    def test_score_report_saved_without_plotting_and_matches_selection_api(self):
        self.test_app_sensor_scores_use_chamber_reference_and_skip_step_zero()
        output=Path(self.tmp.name)/'score_reports'
        selection=dict(overview=dict(steps=[1],sensors=['P1']))
        with patch.object(self.s,'_report_root',return_value=output), patch.object(self.s,'_archive_date_of',return_value='2026-09-08'):
            self.s._run_report(['1_0'],score_selections=selection,include_images=False)
            report=json.loads((output/self.s._short_name('1_0')/'score_calculation.json').read_text())
        result=self.s.api_scatter_step_scores(self.s.ScatterStepScoresReq(model_id=self.s._model_id(),steps=[1],sensors=['P1'],wafers=[dict(wid='1_0',date='2026-09-08')]))['wafers'][0]
        audit=report['selections']['overview']
        self.assertEqual(audit['ratio'],result['ratio'])
        self.assertEqual(audit['threshold'],result['threshold'])
        self.assertEqual(self.s.REPORT['done'],1)
        self.assertFalse(self.s.REPORT['running'])
        self.assertFalse(self.s.REPORT['images_enabled'])

    def test_overview_max_endpoint_keeps_detail_and_wafer_sigma(self):
        self.test_app_sensor_scores_use_chamber_reference_and_skip_step_zero()
        for ch in (1,2):
            folder = self.models / ('DSV%d' % ch) / '18_앱점수'
            np.savez(folder/'step_sensor_scores.npz',process_steps=[0,1],sensors=['P%d' % ch],
                train_ok_scores=[[[999.],[float(ch)]],[[999.],[3.*ch]]])
        items = [dict(date='2026-09-08',wid='%d_0' % ch) for ch in (1,2)]
        req = self.s.ScatterMaxSensorsReq(model_id=self.s._model_id(),wafers=items)
        full = self.s.api_scatter_sensors_max(req)['wafers']
        selected = self.s.api_scatter_sensors_max(self.s.ScatterMaxSensorsReq(
            model_id=self.s._model_id(),wafers=items,steps=[1]))['wafers']
        self.assertEqual(full,selected)
        for ch,row in zip((1,2),full):
            detail = self.s.api_archive_app_sensor_scores('2026-09-08','%d_0' % ch)
            sensor = detail['sensors'][0]
            self.assertAlmostEqual(sensor['threshold_raw'], 5.5*ch)
            self.assertAlmostEqual(row['scores']['P%d'%ch],sensor['selected_score']/(3*ch))
            self.assertNotEqual(row['scores']['P%d'%ch],sensor['ratio'])
            total = self.s.api_scatter_step_scores(self.s.ScatterStepScoresReq(
                model_id=self.s._model_id(),wafers=[items[ch-1]],steps=[1]))['wafers'][0]
            self.assertEqual(total['ratio'],detail['total_ratio'])
            self.assertEqual(total['scores']['P%d'%ch],sensor['ratio'])

    def test_selected_scores_and_overview_work_during_inspection(self):
        self.test_app_sensor_scores_use_chamber_reference_and_skip_step_zero()
        req = self.s.ScatterStepScoresReq(model_id=self.s._model_id(), steps=[1], sensors=['P1'],
            wafers=[{'date': '2026-09-08', 'wid': '1_0'}])
        expected = self.s.api_scatter_step_scores(req)['wafers'][0]['ratio']
        self.assertIsNotNone(expected)
        point = dict(wafer_id='1_0', ts='2026-09-08T00:12:03', chamber='dsv1', recipe='x')
        base = dict(bounds=['2026-09-08T00:00:00', '2026-09-09T00:00:00'],
                    scale='day', bands={'warn': 1.05, 'crit': 1.2},
                    lanes=[dict(chamber='dsv1', cells=[])])
        for progress in ('progress', 'progress_bg'):
            with patch.dict(self.s.STATE, {progress: {'active': True}, 'score_inspect_requests': 1}), \
                 patch.object(self.s, 'api_overview', return_value=base), \
                 patch.object(self.s, 'api_scatter', return_value={'points': [point]}):
                self.assertEqual(self.s.api_scatter_step_scores(req)['wafers'][0]['ratio'], expected)
                result = self.s.api_overview_selection(self.s.OverviewSelectionReq(
                    model_id=self.s._model_id(), steps=[1], sensors=['P1'], dispersion=True))
                self.assertEqual(result['lanes'][0]['cells'][0]['total'], 1)
                self.assertIn('dispersion', result)

    def test_scatter_step_scores_use_each_chamber_reference(self):
        self.test_app_sensor_scores_use_chamber_reference_and_skip_step_zero()
        catalog = self.s.api_scatter_steps()
        self.assertEqual(catalog['steps'], [1])
        request = self.s.ScatterStepScoresReq(model_id=self.s._model_id(), steps=[1],
            wafers=[{'date':'2026-09-08','wid':'%d_0' % ch} for ch in (1,2)])
        response = self.s.api_scatter_step_scores(request)
        for ch, row in zip((1,2), response['wafers']):
            self.assertIsNone(row['missing'], row)
            self.assertEqual(row['threshold'], float(ch))
            self.assertAlmostEqual(row['ratio'], row['scores']['P%d' % ch])
        request.sensors = ['P1']
        selected = self.s.api_scatter_step_scores(request)['wafers']
        self.assertIsNone(selected[0]['missing'])
        self.assertIsNotNone(selected[1]['missing'])
        request.sensors = []
        with self.assertRaises(self.s.HTTPException):
            self.s.api_scatter_step_scores(request)
        request.sensors = None
        request.steps = [0]
        with self.assertRaises(self.s.HTTPException):
            self.s.api_scatter_step_scores(request)
        request.steps = [2]
        response = self.s.api_scatter_step_scores(request)
        self.assertTrue(all(row['missing'] and row['ratio'] is None for row in response['wafers']))

    def test_detail_score_selection_matches_overview_and_rejects_empty(self):
        self.test_app_sensor_scores_use_chamber_reference_and_skip_step_zero()
        req = self.s.DetailScoreSelectionReq(steps=[1], sensors=['P1'])
        result = self.s.api_archive_app_sensor_selection('2026-09-08','1_0',req)
        overview = self.s.api_scatter_step_scores(self.s.ScatterStepScoresReq(
            model_id=self.s._model_id(),steps=[1],sensors=['P1'],
            wafers=[{'date':'2026-09-08','wid':'1_0'}]))['wafers'][0]
        self.assertEqual(result['sensors'][0]['ratio'],overview['scores']['P1'])
        self.assertEqual(result['available_steps'],[1])
        self.assertEqual(result['available_sensors'],['P1'])
        req.steps = [0]
        with self.assertRaises(self.s.HTTPException):
            self.s.api_archive_app_sensor_selection('2026-09-08','1_0',req)
        req.steps = [1]
        req.sensors = ['P2']
        self.assertFalse(self.s.api_archive_app_sensor_selection('2026-09-08','1_0',req)['available'])

    def test_detail_selection_get_http_contract(self):
        self.test_app_sensor_scores_use_chamber_reference_and_skip_step_zero()
        from fastapi.testclient import TestClient
        client = TestClient(self.s.app)
        url = '/api/archive/2026-09-08/1_0/sensors-app-score'
        selection = {'steps':[1], 'sensors':['P1']}
        response = client.get(url, params={'selection':json.dumps(selection)})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['applied_selection'],selection)
        self.assertTrue(response.json()['available'])
        self.assertEqual(client.get(url,params={'selection':'bad'}).status_code,400)
        self.assertEqual(client.get(url,params={'selection':json.dumps({'steps':[], 'sensors':['P1']})}).status_code,400)
        self.assertEqual(client.post(url,json=selection).status_code,200)

    def test_selection_boxes_count_threshold_and_exclude_missing(self):
        base = dict(bounds=['2026-09-08T00:00:00','2026-09-09T00:00:00'],
                    bands={'warn':2,'crit':3}, lanes=[dict(chamber='dsv1',cells=[{'total':99}])])
        points = [dict(wafer_id=str(i),ts='2026-09-08T12:00:00',chamber='dsv1') for i in range(3)]
        points.append(dict(wafer_id='end',ts='2026-09-09T00:00:00',chamber='dsv1'))
        req = self.s.OverviewSelectionReq(model_id=self.s._model_id(), steps=[1], sensors=['P1'])
        with patch.object(self.s,'api_overview',return_value=base), patch.object(self.s,'api_scatter',return_value={'points':points}), patch.object(self.s,'api_scatter_step_scores',return_value={'wafers':[{'wid':'0','ratio':.5},{'wid':'1','ratio':1.0},{'wid':'2','ratio':None}]}):
            result = self.s.api_overview_selection(req)
        cell = result['lanes'][0]['cells'][0]
        self.assertEqual((cell['total'],cell['bad'],cell['na'],cell['defectPct']), (2,1,1,50.0))
        self.assertEqual(base['lanes'][0]['cells'][0]['total'],99)

    def test_selected_dispersion_reuses_scores_and_never_falls_back_to_original(self):
        base = dict(bounds=['2026-09-08T00:00:00','2026-09-09T00:00:00'],
                    scale='day', bands={'warn':1.05,'crit':1.2},
                    lanes=[dict(chamber='dsv1',cells=[])])
        points = [dict(wafer_id=str(i),ts='2026-09-08T12:0%d:00' % i,
                       chamber='dsv1',recipe='x',lot='lot-a',score=99,ratio_x=99)
                  for i in range(3)]
        scores = {'wafers':[{'wid':'0','ratio':.5},{'wid':'1','ratio':2.2},
                            {'wid':'2','ratio':None}]}
        req = self.s.OverviewSelectionReq(model_id=self.s._model_id(), steps=[1],
                                         sensors=['P1'], dispersion=True)
        with patch.object(self.s,'api_overview',return_value=base), \
             patch.object(self.s,'api_scatter',return_value={'points':points}), \
             patch.object(self.s,'api_scatter_step_scores',return_value=scores) as calc:
            result = self.s.api_overview_selection(req)
            self.assertEqual(calc.call_count, 1)  # One calculation for both cards and dispersion.
            disp = result['dispersion']
            cell = next(l for l in disp['lanes'] if l['chamber']=='dsv1')['cells'][0]
            self.assertAlmostEqual(cell['ratio'], .5)  # mean(max(0, [.5,2.2]-1.2))
            self.assertEqual(disp['stat']['n_no_score'], 1)
            self.assertEqual(disp['bounds'], base['bounds'])
            self.assertEqual(disp['selection'], result['selection'])
            req.include_first = False
            without_first = self.s.api_overview_selection(req)['dispersion']
            self.assertEqual(without_first['stat']['n_first_excluded'], 1)
            self.assertAlmostEqual(without_first['lanes'][0]['cells'][0]['ratio'], 1.)
            req.steps = [2]
            req.sensors = ['P2']
            scores['wafers'][1]['ratio'] = .8
            changed = self.s.api_overview_selection(req)['dispersion']
            self.assertEqual(changed['lanes'][0]['cells'][0]['ratio'], 0.)
            self.assertEqual(changed['selection']['sensors'], ['P2'])
        self.assertEqual(points[0]['score'], 99)

    def test_dispersion_excludes_only_identified_lot_first(self):
        rows = [dict(wafer_id='1_%d' % i, ts='2026-09-08T12:0%d:00' % i,
                     chamber='dsv1', recipe='x', lot='lot-a', lot_first=False,
                     ratio_thr=v, score=None) for i, v in enumerate((2.2, .5))]
        def calculate():
            return self.s._dispersion_groups(rows, 10, 'iqr', 2, basis='excess',
                                             include_first=False, excess_base=1.2)
        result = calculate()
        self.assertEqual(result['stat']['n_first_excluded'], 0)
        self.assertAlmostEqual(result['groups']['dsv1'][0]['disp'], .5)
        rows[0]['lot_first'] = True
        result = calculate()
        self.assertEqual(result['stat']['n_first_excluded'], 1)
        self.assertEqual(result['groups']['dsv1'][0]['disp'], 0)
        rows[0]['ratio_thr'] = None
        result = calculate()
        self.assertEqual(result['stat']['n_first_excluded'], 0)
        self.assertEqual(result['groups']['dsv1'][0]['n'], 1)

    def test_demo_export_keeps_all_rows_for_manual_period_queries(self):
        points = [dict(wafer_id='w%03d' % i, chamber='dsv1',
                       ts='2026-08-%02dT12:00:00' % (1 + i // 3), ratio=1., recipe='x')
                  for i in range(60)]
        with patch.object(self.s, 'api_scatter', return_value={'points': list(reversed(points))}), \
             patch.object(self.s, '_load_archive', return_value=None) as archives, \
             patch.object(self.s, 'api_scatter_sensors', side_effect=lambda req: {'wafers': [{} for w in req.wafers]}) as scores, \
             patch('src.app.export_html.build_export_html', return_value='<html></html>') as build:
            self.s.api_export_html()
        exported = build.call_args.args[1]
        self.assertEqual([w['wafer_id'] for w in exported], ['w%03d' % i for i in range(60)])
        self.assertTrue(build.call_args.args[3]['export_manual_query'])
        self.assertEqual([c.args[1] for c in archives.call_args_list][-50:],
                         ['w%03d' % i for i in range(10, 60)])
        self.assertEqual(sum(len(c.args[0].wafers) for c in scores.call_args_list), 60)

    def test_demo_exports_selection_matrices_for_all_chambers(self):
        self.test_app_sensor_scores_use_chamber_reference_and_skip_step_zero()
        points = [dict(wafer_id='%d_0' % ch, chamber='dsv%d' % ch,
                       ts='2026-09-08T00:12:03', ratio=1., recipe='x') for ch in (1,2)]
        with patch.object(self.s,'api_scatter',return_value={'points':points}), patch('src.app.export_html.build_export_html',return_value='<html></html>') as build:
            self.s.api_export_html(detail_limit=0)
        data = build.call_args.args[3]['offline_scores']
        self.assertEqual(len(data['models']),2)
        self.assertEqual(len(data['wafers']),2)
        self.assertTrue(all('matrix' in w for w in data['wafers'].values()))

    def test_demo_export_includes_sensor_scores_without_detail(self):
        data = self._integration_storage()
        with patch.object(self.s, '_recipe_tag', return_value='x'):
            self.s.run_inspection(data)
        points = [dict(wafer_id='%d_%d' % (ch, i), chamber='dsv%d' % ch,
                       ts='2026-09-08T00:12:03', ratio=1., recipe='x')
                  for ch in (1, 2) for i in (0, 1)]
        with patch.object(self.s, 'api_scatter', return_value={'points': points}), \
             patch('src.app.export_html.build_export_html', return_value='<html></html>') as build:
            self.s.api_export_html(detail_limit=0)
        self.assertEqual(build.call_args.args[2], {})
        cfg = build.call_args.args[3]
        self.assertEqual(len(cfg['scatter_scores']), 4)
        self.assertEqual(set(cfg['recon_sensors']), {'P1', 'P2'})
        self.assertEqual(set(cfg['scatter_scores']['2026-09-08/1_0']['scores']), {'P1'})
        self.assertEqual(set(cfg['scatter_scores']['2026-09-08/2_0']['scores']), {'P2'})

    def test_scatter_sensor_batch_uses_each_chamber_archive_schema(self):
        data = self._integration_storage()
        with patch.object(self.s, '_recipe_tag', return_value='x'):
            self.s.run_inspection(data)
        self._assert_archives()
        mid = self.s._model_id()
        request = self.s.ScatterSensorsReq(model_id=mid, wafers=[
            {'date': '2026-09-08', 'wid': '1_0'},
            {'date': '2026-09-08', 'wid': '2_0'}])
        result = self.s.api_scatter_sensors(request)
        self.assertEqual(result['model_id'], mid)
        self.assertEqual(set(result['wafers'][0]['scores']), {'P1'})
        self.assertEqual(set(result['wafers'][1]['scores']), {'P2'})
        self.assertIsNone(self.engine._scope.get())

    def test_cleared_chamber_path_does_not_restore_previous_or_default(self):
        config = {'chamber_paths': {'dsv1': str(self.folder)}}
        with patch.object(self.s, '_load_app_config', return_value=config), \
             patch.object(self.s, '_save_app_config') as save:
            self.s._set_chamber_paths({'dsv1': ''})
            save.assert_called_once()
            self.assertEqual(self.s._chamber_paths()['dsv1'], '')
            self.assertEqual(self.s._chamber_paths()['dsv2'], self.s.CHAMBER_PATH_DEFAULTS['dsv2'])
            self.assertNotIn('dsv1', [ch for ch, path in self.s._active_chamber_dirs()])

    def test_auto_off_cancels_prepared_batch_and_survives_restart(self):
        data = self._integration_storage()
        auto = self.s.AutoPilot()
        auto.set_enabled(True)
        self.addCleanup(self.s._ABORT.clear)
        self.addCleanup(self.s._PAUSE.set)
        self.s._ABORT.clear()
        self.s._PAUSE.set()
        calls = []
        def prefill(recs, pause_check=None):
            calls.append(len(recs))
            auto.set_enabled(False)
            self.assertIsNotNone(pause_check)
            pause_check()
        with patch.object(self.s, '_scan_targets', return_value=[(None, data)]), \
             patch.object(self.s, '_prefill_batch', side_effect=prefill), \
             patch.object(self.s, '_result_for') as result:
            with self.assertRaises(self.s.InspectAborted):
                auto._scan_once()
        self.assertEqual(len(calls), 1)
        result.assert_not_called()
        self.assertFalse(self.s.AutoPilot().enabled)
        self.assertIsNone(getattr(self.s._INSPECT_CONTEXT, 'auto', None))

    def test_auto_old_generation_cannot_resume_after_toggle_or_path_change(self):
        self._integration_storage()
        auto = self.s.AutoPilot()
        auto.set_enabled(True)
        self.addCleanup(self.s._ABORT.clear)
        self.addCleanup(self.s._PAUSE.set)
        self.s._ABORT.clear()
        self.s._PAUSE.set()
        for change in ('toggle', 'inbox', 'chamber'):
            visited = []
            def scan(folder, chamber, **kwargs):
                visited.append(folder)
                if change == 'toggle':
                    auto.set_enabled(False)
                    auto.set_enabled(True)
                elif change == 'inbox':
                    self.s.api_set_inbox(self.s.InboxReq(folder=str(self.folder)))
                else:
                    self.s.api_set_chamber_paths(self.s.ChamberPathsReq(paths={}))
                self.s._pause_gate()
            with patch.object(self.s, 'AUTOPILOT', auto), \
                 patch.object(self.s, '_scan_targets', return_value=[(None, self.folder), (None, self.models)]), \
                 patch.object(auto, '_pending_count', return_value=1), \
                 patch.object(auto, '_scan_folder', side_effect=scan):
                with self.assertRaises(self.s.InspectAborted):
                    auto._scan_once()
            self.assertEqual(visited, [self.folder])
            self.assertTrue(auto.enabled)
        # The next scan receives a new token and can run even if foreground was aborted.
        with patch.object(auto, '_scan_once_current') as scan:
            auto._scan_once()
            scan.assert_called_once()

    def test_cancel_callback_is_carried_into_reader_threads(self):
        self._integration_storage()
        auto = self.s.AutoPilot()
        auto.set_enabled(True)
        self.s._INSPECT_CONTEXT.auto = (auto, auto._generation)
        try:
            check = self.s._reader_cancel_check()
        finally:
            self.s._INSPECT_CONTEXT.auto = None
        auto.set_enabled(False)
        cancelled = threading.Event()
        def read():
            try:
                check()
            except self.s.InspectAborted:
                cancelled.set()
        worker = threading.Thread(target=read)
        worker.start()
        worker.join(2)
        self.assertTrue(cancelled.is_set())

    def test_complete_abort_wakes_paused_auto_and_persists_off(self):
        self._integration_storage()
        auto = self.s.AutoPilot()
        auto.set_enabled(True)
        self.addCleanup(self.s._ABORT.clear)
        self.addCleanup(self.s._PAUSE.set)
        entered, stopped = threading.Event(), threading.Event()
        self.s._ABORT.clear()
        self.s._PAUSE.clear()
        def scan():
            self.s._INSPECT_CONTEXT.auto = (auto, auto._generation)
            entered.set()
            try:
                self.s._wait_if_paused(0, 1, 'old', bg=True)
            except self.s.InspectAborted:
                stopped.set()
            finally:
                self.s._INSPECT_CONTEXT.auto = None
        worker = threading.Thread(target=scan, daemon=True)
        worker.start()
        self.assertTrue(entered.wait(2))
        with patch.object(self.s, 'AUTOPILOT', auto):
            self.s.api_inspect_abort()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(stopped.is_set())
        self.assertTrue(self.s._ABORT.is_set())
        self.assertFalse(self.s.AutoPilot().enabled)

    def test_automatic_full_pipeline_batches_and_skips_completed(self):
        data=self._integration_storage()
        auto=self.s.AutoPilot()
        with patch.object(self.s,'_recipe_tag',return_value='x'), \
             patch.object(self.s,'_prefill_batch',wraps=self.s._prefill_batch) as batch:
            self.assertEqual(auto._scan_folder(data),4)
            groups=[{r['chamber'] for r in c.args[0]} for c in batch.call_args_list]
            self.assertTrue(all(len(g)==1 for g in groups),groups)
            self.assertEqual(auto._scan_folder(data),0)
        self._assert_archives()

    def test_settings_http_routes_and_failed_selection(self):
        from fastapi.testclient import TestClient
        with patch.object(self.s,'_list_models',return_value=[]):
            client=TestClient(self.s.app)
            response=client.get('/api/models')
            self.assertEqual(response.status_code,200)
            self.assertEqual(set(response.json()['chamber_models']),{'dsv1','dsv2'})
            response=client.get('/api/sensors')
            self.assertEqual(set(response.json()['input_sensors']),{'P1','P2'})
            self.assertEqual(client.post('/api/model/select',json={'dir':'/missing/path'}).status_code,400)

    def test_log_reads_physical_second_row_all_six_chambers(self):
        for ch in range(1,7):
            log = self.folder/('%d.log'%ch)
            log.write_text('a\tb\tc\nx\ty\t\'PMCUNIT%d\nx\ty\tPMCUNIT12\n'%(ch+6))
            self.assertEqual(self.s._extract_chamber_from_log(log),'dsv%d'%ch)


if __name__ == '__main__':
    unittest.main()
