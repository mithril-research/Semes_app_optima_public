import unittest
import numpy as np
from src.app.sensor_differences import valve_phase_steps, difference_series, compact_trend_export, PHASE_VALVES

class ValvePhaseTests(unittest.TestCase):
 def archive(self):
  steps=np.repeat(np.arange(1,9),3);io=np.zeros((24,len(PHASE_VALVES)))
  def open_step(st,*valves):
   for v in valves:io[steps==st,PHASE_VALVES.index(v)]=1
  for st in (1,2):open_step(st,'9000002','9000003')
  for st in (3,4):open_step(st,'9000009')
  for st in (5,6):open_step(st,'9000004')
  open_step(7,'9000011','9000012')
  open_step(8,'9000013')
  return dict(meta={'rca_io_cols':list(PHASE_VALVES),'sensor_cols':['p_test']},step=steps,rca_io=io,actual=np.ones((24,1)))
 def test_all_four_phases(self):
  result=valve_phase_steps(self.archive());self.assertEqual(result['status'],'ok')
  self.assertEqual(result['steps'],dict(supply=[1,2],wait=[3,4],process=[5,6],discharge=[7,8]))
 def test_wait_allows_one_mismatch(self):
  a=self.archive();a['rca_io'][6,0]=1
  self.assertEqual(valve_phase_steps(a)['steps']['wait'],[3,4])
  self.assertEqual(valve_phase_steps(a)['steps']['supply'],[1,2])
 def test_discharge_allows_one_mismatch(self):
  a=self.archive();a['rca_io'][18,PHASE_VALVES.index('9000012')]=0
  self.assertEqual(valve_phase_steps(a)['status'],'ok')
 def test_independent_maxima_not_simultaneous(self):
  a=self.archive();a['rca_io'][18:21,PHASE_VALVES.index('9000011')]=[1,0,1]
  a['rca_io'][18:21,PHASE_VALVES.index('9000012')]=[0,1,0]
  self.assertEqual(valve_phase_steps(a)['status'],'unavailable')
 def test_missing_or_unknown_does_not_fall_back(self):
  for missing in ('rca_io',):
   a=self.archive();a.pop(missing);self.assertEqual(valve_phase_steps(a)['status'],'unavailable')
  a=self.archive();a['rca_io'][0,0]=np.nan;self.assertEqual(valve_phase_steps(a)['status'],'unavailable')
  a=self.archive();a['meta']['rca_io_cols'][0]='not_valve';self.assertEqual(valve_phase_steps(a)['status'],'unavailable')
 def test_additional_valve_prevents_only_41(self):
  a=self.archive();a['meta']['rca_io_cols'].append('9000014');a['rca_io']=np.column_stack((a['rca_io'],np.ones(24)))
  self.assertEqual(valve_phase_steps(a)['status'],'unavailable')
 def test_repeated_step_ambiguous(self):
  a=self.archive();a['step'][-1]=1;self.assertEqual(valve_phase_steps(a)['status'],'unavailable')
 def test_original_samples_only(self):
  a=self.archive();a['meta']['interp_factor']=2;a['step']=np.repeat(a['step'],2);a['actual']=np.repeat(a['actual'],2,axis=0);a['rca_io']=np.repeat(a['rca_io'],2,axis=0)
  a['rca_io'][1::2]=.5
  self.assertEqual(valve_phase_steps(a)['steps']['wait'],[3,4])
 def test_export_retains_boundaries(self):
  a=self.archive();summary=difference_series(a);export=compact_trend_export(summary,['mean'])
  self.assertEqual(export['valve_phases'],summary['valve_phases']);self.assertEqual(export['valve_phases']['steps']['discharge'],[7,8])

 def test_after_discharge_is_always_discharge(self):
  a=self.archive();a['rca_io'][21:]=0;a['rca_io'][21:,PHASE_VALVES.index('9000009')]=1
  self.assertEqual(valve_phase_steps(a)['steps']['discharge'],[7,8])
 def test_server_summary_reads_archived_io(self):
  import ast,json,tempfile
  from pathlib import Path
  from src.app.sensor_differences import TrendSummaryCache
  a=self.archive()
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'w.npz'
   np.savez_compressed(path,actual=a['actual'],step=a['step'],rca_io=a['rca_io'],meta_json=np.frombuffer(json.dumps(a['meta']).encode(),dtype=np.uint8))
   tree=ast.parse(Path('src/app/server.py').read_text())
   node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_trend_summary')
   scope=dict(np=np,json=json,_archive_path=lambda *args:path,_ARCHIVE_Q=None,_model_id=lambda:'m',_TREND_SUMMARY_CACHE=TrendSummaryCache(),_trend_original_path=lambda meta,wid:None)
   exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-summary>','exec'),scope)
   self.assertEqual(scope['_trend_summary']('2026-10-14','w')['summary']['valve_phases']['steps']['wait'],[3,4])

 def test_three_allowed_four_rejected(self):
  for phase in ('wait','discharge'):
   for misses in (3,4):
    a=self.archive();a['step']=np.repeat(a['step'],3);a['actual']=np.repeat(a['actual'],3,axis=0);a['rca_io']=np.repeat(a['rca_io'],3,axis=0)
    if phase=='wait':
     for step in (3,4):
      indices=np.flatnonzero(a['step']==step)[:misses];a['rca_io'][indices,0]=1
    else:
     indices=np.flatnonzero(a['step']==7)[:misses];a['rca_io'][indices,PHASE_VALVES.index('9000012')]=0
    result=valve_phase_steps(a)
    self.assertEqual(result['status'],'ok' if misses==3 else 'unavailable',(phase,misses))
    if misses==3:self.assertEqual(result['margin_samples'],3)
 def test_short_step_requires_a_matching_sample(self):
  a=self.archive();a['rca_io'][6:12]=0
  self.assertEqual(valve_phase_steps(a)['status'],'unavailable')
 def test_old_phase_version_not_reused(self):
  self.assertEqual(valve_phase_steps(self.archive())['version'],2)
