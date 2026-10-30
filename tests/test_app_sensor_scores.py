import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.app.app_sensor_scores import sensor_scores, score_matrix, AppScoreUnavailable, OfflineScoreExport


class AppSensorScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / '18_앱점수'
        self.folder.mkdir()
        self.meta = dict(process_steps=[0, 1, 2], sensors=['T1', 'P1'],
                         score_topk=0, score_resid_agg='sum', score_time_avg=False,
                         score_final_avg=False, additive_exact_match_expected=True)
        self.train = np.array([[[999,999],[1,2],[1,2]], [[888,888],[2,4],[2,4]]], float)
        self.save()

    def save(self):
        np.savez(self.folder/'step_sensor_scores.npz', train_ok_scores=self.train,
                 process_steps=self.meta['process_steps'], sensors=self.meta['sensors'])
        (self.folder/'metadata.json').write_text(json.dumps(self.meta))

    def test_overview_max_uses_maximum_of_sums_and_keeps_sigma_totals(self):
        self.train[:,1:,0] = [[1,8],[6,2]]
        self.save()
        args = (self.root, [[3,2],[4,2]], [1,2], ['T1','P1'])
        full = sensor_scores(*args, sensor_basis='train_ok_max')
        selected = sensor_scores(*args, sensor_basis='train_ok_max', selected_steps=[1,2], selected_sensors=['T1','P1'])
        self.assertEqual(full, selected)
        self.assertEqual(full['sensors'][0]['threshold_raw'], 9)
        self.assertAlmostEqual(full['sensors'][0]['ratio'], 7/9)
        partial = sensor_scores(*args, sensor_basis='train_ok_max', selected_steps=[1], selected_sensors=['T1'])
        self.assertEqual(partial['sensors'][0]['threshold_raw'], 6)
        self.assertEqual(partial['sensors'][0]['ratio'], .5)
        sigma = sensor_scores(*args)
        self.assertEqual(sigma['sensors'][0]['threshold_raw'], 8.5+3.5*.5)
        for field in ('total_score','total_threshold','total_ratio'):
            self.assertEqual(sigma[field], full[field])

    def test_overview_max_zero_reference_is_not_sigma_or_band_fallback(self):
        self.train[:,:,0] = 0
        self.save()
        result = sensor_scores(self.root, [[3,2]], [1], ['T1','P1'], sensor_basis='train_ok_max')
        self.assertIsNone(result['sensors'][0]['ratio'])
        self.assertEqual(result['sensors'][0]['threshold_raw'], 0)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey gjs')
    def test_export_max_sensors_and_unchanged_detail_sigma(self):
        from src.app.export_html import _MOCK_TMPL
        self.train[:,1:,0] = [[1,8],[6,2]]
        self.save()
        level = [[3,2],[4,2]]
        export = OfflineScoreExport()
        export.add('2026-09-08/w', self.root, level, [1,2], ['T1','P1'])
        cases = []
        for steps in (None,[1],[2],[1,2]):
            for sensors in (None,['T1'],['P1'],['T1','P1']):
                kwargs = dict(selected_steps=steps,selected_sensors=sensors)
                maximum = sensor_scores(self.root,level,[1,2],['T1','P1'],sensor_basis='train_ok_max',**kwargs)
                sigma = sensor_scores(self.root,level,[1,2],['T1','P1'],**kwargs)
                cases.append(dict(steps=steps,sensors=sensors,
                    maximum={r['sensor']:r['ratio'] for r in maximum['sensors']},
                    sigma={r['sensor']:r['ratio'] for r in sigma['sensors']},total=sigma['total_ratio']))
        mock = (_MOCK_TMPL.replace('__WAFERS__','[]').replace('__DETAIL__','{}')
            .replace('__CFG__',json.dumps(dict(model_id='m',offline_scores=export.payload())))
            .replace('<script>','').replace('</script>',''))
        check = '''
const loop=new imports.gi.GLib.MainLoop(null,false);
async function run(){for(var c of CASES){
 var req={model_id:'m',steps:c.steps,sensors:c.sensors,wafers:[{date:'2026-09-08',wid:'w'}]};
 var max=await window.fetch('/api/scatter/sensors-max',{method:'POST',body:JSON.stringify(req)}).then(r=>r.json());
 Object.keys(c.maximum).forEach(s=>{if(Math.abs(max.wafers[0].scores[s]-c.maximum[s])>1e-10)throw Error('max parity');});
 var sel={steps:c.steps||[1,2],sensors:c.sensors||['T1','P1']};
 var detail=await window.fetch('/api/archive/2026-09-08/w/sensors-app-score?selection='+encodeURIComponent(JSON.stringify(sel))).then(r=>r.json());
 detail.sensors.forEach(s=>{if(Math.abs(s.ratio-c.sigma[s.sensor])>1e-10)throw Error('detail sigma changed');});
 if(Math.abs(detail.total_ratio-c.total)>1e-10)throw Error('wafer total changed');
}}
run().then(()=>loop.quit(),e=>{printerr(e.stack);imports.system.exit(1);});loop.run();
'''.replace('CASES',json.dumps(cases))
        script = self.root/'check.js'
        script.write_text('var window={},document={addEventListener:function(){}};\n'+mock+check)
        result = subprocess.run(['gjs',str(script)],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey gjs')
    def test_exported_selected_dispersion_matches_calibrated_scores(self):
        from src.app.export_html import _MOCK_TMPL
        export = OfflineScoreExport()
        levels = [np.array([[8.,10.],[9.,11.]]), np.array([[1.,3.],[4.,2.]])]
        wafers = []
        for i in range(3):
            wid = 'w%d' % i
            wafers.append(dict(wafer_id=wid,date='2026-09-08',ts='2026-09-08T12:0%d:00'%i,
                               chamber='dsv1',recipe='x',lot='lot-a',ratio=99.,score=99.))
            if i < 2:
                export.add('2026-09-08/'+wid,self.root,levels[i],[1,2],['T1','P1'])
        cases = []
        for steps in ([1],[2],[1,2]):
            for sensors in (['T1'],['P1'],['T1','P1']):
                ratios = [sensor_scores(self.root,level,[1,2],['T1','P1'],
                                        selected_steps=steps,selected_sensors=sensors)['total_ratio']
                          for level in levels]
                for include_first in (True,False):
                    selected = ratios if include_first else ratios[1:]
                    cases.append(dict(steps=steps,sensors=sensors,include_first=include_first,
                                      expected=float(np.mean(np.maximum(0,np.array(selected)-1.2)))))
        cfg = dict(model_id='m',now='2026-09-09T00:00:00',offline_scores=export.payload())
        mock = (_MOCK_TMPL.replace('__WAFERS__',json.dumps(wafers))
                .replace('__DETAIL__','{}').replace('__CFG__',json.dumps(cfg)))
        mock = mock.replace('<script>','').replace('</script>','')
        harness = r"""
const loop=new imports.gi.GLib.MainLoop(null,false);
async function run(){
 for(var c of CASES){
  var req=Object.assign({model_id:'m',rng:'all',scale:'day',dispersion:true},c);
  var d=await window.fetch('/api/overview/selection',{method:'POST',body:JSON.stringify(req)}).then(r=>r.json());
  var cell=d.dispersion.lanes[0].cells.find(c=>c.ratio!=null);
  if(!cell||Math.abs(cell.ratio-c.expected)>1e-6)throw Error('dispersion parity '+JSON.stringify({c,cell}));
  if(d.dispersion.stat.n_no_score!==1)throw Error('missing selected score used original score');
  if(d.dispersion.bounds.join()!==d.bounds.join())throw Error('date bounds mismatch');
 }
}
run().then(()=>loop.quit(),e=>{printerr(e.stack);imports.system.exit(1);});loop.run();
""".replace('CASES',json.dumps(cases))
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder)/'check.js'
            script.write_text('var window={},document={addEventListener:function(){}};\n'+mock+harness)
            result = subprocess.run(['gjs',str(script)],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey gjs')
    def test_offline_dispersion_respects_exported_first_wafer_identity(self):
        from src.app.export_html import _MOCK_TMPL
        for first in (False, True):
            wafers = [dict(wafer_id='w%d' % i, date='2026-09-08',
                           ts='2026-09-08T12:0%d:00' % i, chamber='dsv1',
                           recipe='x', lot='lot-a', lot_first=first and i == 0,
                           ratio=v, ratio_x=v, score=v, threshold=1.)
                      for i, v in enumerate((2.2, .5))]
            mock = (_MOCK_TMPL.replace('__WAFERS__', json.dumps(wafers))
                    .replace('__DETAIL__', '{}').replace('__CFG__', '{}')
                    .replace('<script>', '').replace('</script>', ''))
            check = '''
const loop=new imports.gi.GLib.MainLoop(null,false);
window.fetch('/api/dispersion?range=all&scale=day&include_first=0')
 .then(r=>r.json()).then(d=>{
   var cell=d.lanes[0].cells.find(c=>c.ratio!=null);
   if(!cell||Math.abs(cell.ratio-EXPECTED)>1e-6||d.stat.n_first_excluded!==EXCLUDED)
     throw Error(JSON.stringify(d));
 }).then(()=>loop.quit(),e=>{printerr(e.stack);imports.system.exit(1);});loop.run();
'''.replace('EXPECTED', '0' if first else '.5').replace('EXCLUDED', '1' if first else '0')
            with tempfile.TemporaryDirectory() as folder:
                script = Path(folder) / 'check.js'
                script.write_text('var window={},document={addEventListener:function(){}};\n'+mock+check)
                result = subprocess.run(['gjs', str(script)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey gjs')
    def test_html_basename_override_applies_to_both_total_apis_only(self):
        from src.app.export_html import _MOCK_TMPL
        export = OfflineScoreExport()
        rule = dict(version=2,rules=[dict(pattern='BAD.log',threshold_multiplier=1.02),dict(pattern='SECOND.log',threshold_multiplier=1.15),dict(pattern='DEFAULT.log')])
        for wid, source in [('yes', r'C:\logs\BAD.log'), ('no', '/BAD.log/normal.log'), ('second','/SECOND.log'), ('default','/DEFAULT.log')]:
            export.add('2026-09-08/'+wid, self.root, [[30,40],[50,60]], [1,2], ['T1','P1'],
                       record=dict(wafer_id=wid, source_path=source), override=rule)
        self.assertEqual(export.wafers['2026-09-08/yes']['override_ratio'], 1.02)
        self.assertNotIn('override_ratio', export.wafers['2026-09-08/no'])
        mock = (_MOCK_TMPL.replace('__WAFERS__','[]').replace('__DETAIL__','{}')
                .replace('__CFG__',json.dumps(dict(model_id='m',offline_scores=export.payload())))
                .replace('<script>','').replace('</script>',''))
        check = r"""
const loop=new imports.gi.GLib.MainLoop(null,false);
async function run(){for(var steps of [[1],[2],[1,2]])for(var sensors of [['T1'],['P1'],['T1','P1']]){
 var selection={steps:steps,sensors:sensors};
 var request={model_id:'m',steps:steps,sensors:sensors,wafers:[{date:'2026-09-08',wid:'yes'},{date:'2026-09-08',wid:'no'}]};
 var scatter=await window.fetch('/api/scatter/steps',{method:'POST',body:JSON.stringify(request)}).then(r=>r.json());
 if(scatter.wafers[0].ratio!==1.02||scatter.wafers[1].ratio===1.02)throw Error('scatter override');
 if(JSON.stringify(scatter.wafers[0].scores)!==JSON.stringify(scatter.wafers[1].scores))throw Error('sensor scores changed');
 var detail=await window.fetch('/api/archive/2026-09-08/yes/sensors-app-score?selection='+encodeURIComponent(JSON.stringify(selection))).then(r=>r.json());
 if(detail.total_ratio!==1.02||!detail.score_overridden||detail.selected_ratio_before_override===1.02)throw Error('detail override');
 for(var pair of [['second',1.15],['default',1.01]]){var extra=await window.fetch('/api/archive/2026-09-08/'+pair[0]+'/sensors-app-score?selection='+encodeURIComponent(JSON.stringify(selection))).then(r=>r.json());if(extra.total_ratio!==pair[1])throw Error('per wafer multiplier');}
}}
run().then(()=>loop.quit(),e=>{printerr(e.stack);imports.system.exit(1);});loop.run();
"""
        script=self.root/'override.js'
        script.write_text('var window={},document={addEventListener:function(){}};\n'+mock+check)
        result=subprocess.run(['gjs',str(script)],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_zero_step_excluded_from_current_and_reference_and_threshold(self):
        r = sensor_scores(self.root, [[10000,10000],[3.25,6.5],[3.25,6.5]], [0,1,2], ['T1','P1'])
        self.assertEqual(r['process_steps'], [1,2])
        for row in r['sensors']:
            self.assertAlmostEqual(row['ratio'], 1.)
            self.assertEqual(row['verdict'], 'NG')  # >= threshold
        self.assertEqual(r['sensors'][0]['threshold_raw'], 6.5) # mean3 + 3.5*population_std1

    def test_selected_steps_apply_to_reference_and_current(self):
        r = sensor_scores(self.root, [[999,999],[3.25,6.5],[900,900]],
                          [0,1,2], ['T1','P1'], selected_steps=[1])
        self.assertEqual(r['process_steps'], [1])
        self.assertEqual(r['total_score'], 9.75)
        self.assertEqual(r['total_threshold'], 9.75)
        self.assertEqual(r['total_ratio'], 1.)
        self.assertEqual(r['sensors'][0]['ratio'], 1.)
        for steps in ([], [0], [3], [1,3]):
            with self.assertRaises(AppScoreUnavailable):
                sensor_scores(self.root, [[1,1]], [1], ['T1','P1'], selected_steps=steps)

    def test_total_threshold_preserves_sensor_covariance(self):
        self.train[:,1,:] = [[1,9],[9,1]]
        self.save()
        r = sensor_scores(self.root, [[5,5]], [1], ['T1','P1'], selected_steps=[1])
        self.assertEqual(r['total_threshold'], 10.)
        self.assertEqual(r['total_ratio'], 1.)
        self.assertEqual(r['sensors'][0]['threshold_raw'], 19.)

    def test_sensor_and_step_selection_calibrate_same_subset(self):
        r = sensor_scores(self.root, [[3.25,999],[999,999]], [1,2],
                          ['T1','P1'], selected_steps=[1], selected_sensors=['T1'])
        self.assertEqual(r['total_score'], 3.25)
        self.assertEqual(r['total_threshold'], 3.25)
        self.assertEqual(r['total_ratio'], 1.)
        self.assertEqual([v['sensor'] for v in r['sensors']], ['T1'])
        for sensors in ([], ['unknown'], ['T1','unknown']):
            with self.assertRaises(AppScoreUnavailable):
                sensor_scores(self.root, [[1,1]], [1], ['T1','P1'], selected_sensors=sensors)

    def test_offline_statistics_preserve_joint_threshold(self):
        self.train[:,1,:] = [[1,9],[9,1]]
        self.save()
        export = OfflineScoreExport()
        export.add('w',self.root,[[5,5],[4,7]],[1,2],['T1','P1'])
        model = export.models['0']
        for indices in ([0],[1],[0,1],[0,2],[0,1,2,3]):
            flat = self.train[:,1:,:].reshape(2,-1)
            expected = flat[:,indices].sum(axis=1)
            std = np.linalg.norm(np.asarray(model['factor'])[:,indices].sum(axis=1))
            self.assertAlmostEqual(std,expected.std(ddof=0))
            self.assertAlmostEqual(np.asarray(model['mean'])[indices].sum(),expected.mean())
        export.add('missing',self.root/'missing',[[1,1]],[1],['T1','P1'])
        self.assertIn('missing',export.wafers['missing'])

    def test_sensor_order_aligned_by_name(self):
        r = sensor_scores(self.root, [[6,3],[6,3]], [1,2], ['P1','T1'])
        self.assertEqual(r['sensors'][0]['index'], 1)
        self.assertEqual(r['sensors'][0]['selected_score'], 6.)

    def test_topk_masks_before_step_sums_and_time_divides_nonzero_count(self):
        for mode in ('score_time_avg', 'score_final_avg'):
            meta = dict(self.meta, score_topk=1, **{mode:True})
            got = score_matrix([[1000,0],[2,1],[4,1]], [0,1,2], ['T1','P1'], np.array([1,2]), ['T1','P1'], meta)
            np.testing.assert_allclose(got, [[1,0],[2,0]])

    def test_topk_aggregation_matches_export_contract(self):
        level = np.array([[6,4],[0,4]],float)
        for agg, expected in [('sum',1),('mean',1),('max',0),('p99',0)]:
            got = score_matrix(level,[1,2],['T1','P1'],np.array([1,2]),['T1','P1'],dict(self.meta,score_topk=1,score_resid_agg=agg))
            self.assertGreater(got[:,expected].sum(),0)
            self.assertEqual(got[:,1-expected].sum(),0)

    def test_zero_reference_is_unavailable_not_infinite_or_false_ok(self):
        self.train[:] = 0; self.save()
        r = sensor_scores(self.root,[[1,1]], [1],['T1','P1'])
        self.assertIsNone(r['sensors'][0]['ratio'])
        self.assertEqual(r['sensors'][0]['verdict'],'NA')

    def test_missing_artifact_and_schema_fail_explicitly(self):
        with self.assertRaises(AppScoreUnavailable):
            sensor_scores(self.root/'missing',[[1,1]],[1],['T1','P1'])
        with self.assertRaises(AppScoreUnavailable):
            sensor_scores(self.root,[[1,1]],[3],['T1','P1'])
        with self.assertRaises(AppScoreUnavailable):
            sensor_scores(self.root,[[1,1]],[1],['T1','wrong'])
        with self.assertRaises(AppScoreUnavailable):
            sensor_scores(self.root,[[1,1]],[0],['T1','P1'])

    def test_bad_reference_and_metadata_rejected(self):
        self.train[0,1,0] = np.nan;self.save()
        with self.assertRaises(AppScoreUnavailable):
            sensor_scores(self.root,[[1,1]],[1],['T1','P1'])
        self.train[0,1,0]=1; self.meta['score_time_avg']='false';self.save()
        with self.assertRaises(AppScoreUnavailable):
            sensor_scores(self.root,[[1,1]],[1],['T1','P1'])

    def test_nonadditive_scores_still_use_same_matrix_calibration(self):
        self.meta['additive_exact_match_expected']=False;self.save()
        r=sensor_scores(self.root,[[3,6],[3,6]],[1,2],['T1','P1'])
        self.assertFalse(r['additive_exact_match_expected'])
        self.assertAlmostEqual(r['sensors'][0]['ratio'],6./6.5)
        self.assertEqual(r['sigma'],3.5)

if __name__=='__main__':unittest.main()
