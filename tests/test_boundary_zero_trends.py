"""Boundary zero visits must never be combined with each other or internal zero."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
from src.trend_sensors import trend_measurements, attach_trend_measurements, boundary_zero_indices
from src.data import step_timing
from src.app.sensor_differences import difference_series, compact_trend_export, step_sensor_metrics, TrendSummaryCache


class BoundaryZeroTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'Time':[0,2,5,8,10,13,17,21,25],
            'ProcessStep':[0,0,1,1,0,2,2,0,0],
            'p_test':[10,14,30,34,999,50,54,100,120],
            't_test':[1,3,5,7,999,9,11,10,20], '9000009':[0]*9})

    def summary(self, frame=None):
        return difference_series(attach_trend_measurements({'meta':{}},trend_measurements(self.frame() if frame is None else frame)),pairs=[['p_test','t_test']])

    def test_means_difference_timing_and_export_stay_separate(self):
        row=self.summary();stats=row['sensor_series'][0]['step_stats']
        self.assertEqual(stats['zero_start'],dict(mean=12.,min=10.,max=14.,std=2.,count=2))
        self.assertEqual(stats['zero_end'],dict(mean=110.,min=100.,max=120.,std=10.,count=2))
        self.assertNotIn('0',stats)
        self.assertEqual(row['series'][-1]['step_stats']['zero_start']['mean'],10.)
        self.assertEqual(row['series'][-1]['step_stats']['zero_end']['mean'],95.)
        self.assertEqual(row['process_steps'],[1,2])
        timing=step_timing(self.frame())['steps']
        self.assertEqual(timing['zero_start']['seconds'],5.)
        self.assertEqual(timing['zero_end']['seconds'],4.)
        self.assertEqual(timing['1']['seconds'],5.)
        self.assertEqual(step_sensor_metrics(row)['mean']['p_test']['zero_end'],110.)
        export=compact_trend_export(row,['mean','max'])
        self.assertEqual(export['sensor_series'][0]['export_step_stats']['zero_end'],dict(mean=110.,max=120.))
        self.assertEqual(export['process_steps'],[1,2]);self.assertEqual(export['zero_step_version'],1)
        json.dumps(export,allow_nan=False)

    def test_no_invented_boundaries_or_missing_values(self):
        for steps,expected in [([0,0],[]),([1,0,2],[]),([1,0],['zero_end']),([0,1],['zero_start']),([np.nan,0,1],[])]:
            self.assertEqual(list(boundary_zero_indices(steps)),expected)
        f=self.frame();f.loc[:1,'p_test']=np.nan
        self.assertNotIn('zero_start',self.summary(f)['sensor_series'][0]['step_stats'])
        f=self.frame().iloc[:-1]
        self.assertIsNone(step_timing(f)['steps']['zero_end']['seconds'])
        f=self.frame();f.loc[0,'ProcessStep']=np.nan
        self.assertNotIn('zero_start',self.summary(f)['sensor_series'][0]['step_stats'])

    def test_old_full_block_archive_restores_zero_from_original(self):
        tree=ast.parse(Path('src/app/server.py').read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_trend_summary','_trend_original_path','_wafer_step_timing')]
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'w.csv';self.frame().to_csv(source,index=False)
            path=Path(folder)/'w.npz';old=self.frame().query('ProcessStep > 0')
            measurements=trend_measurements(old)
            meta={'trend_sensor_cols':measurements['sensor_cols'],'source_path':str(source),'step_timing':{'version':2,'steps':{'1':{'seconds':5}}}}
            np.savez(path,meta_json=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8),trend_step=measurements['step'],trend_actual=measurements['actual'],step=np.array([1,2]))
            ns=dict(np=np,json=json,Path=Path,_ARCHIVE_Q=None,_model_id=lambda:'m',_archive_path=lambda d,w:path,
                    _find_wafer_source=lambda w:None,_TREND_SUMMARY_CACHE=TrendSummaryCache())
            exec(compile(ast.Module(body=funcs,type_ignores=[]),'<server>','exec'),ns)
            result=ns['_trend_summary']('2026-10-20','w')
            self.assertEqual(result['summary']['sensor_series'][0]['step_stats']['zero_end']['mean'],110.)
            self.assertEqual(result['meta']['step_timing']['steps']['zero_start']['seconds'],5.)
            self.assertEqual(ns['_wafer_step_timing']('w',{'meta':meta})['steps']['zero_end']['seconds'],4.)
            source.unlink();missing=ns['_trend_summary']('2026-10-20','w')
            self.assertEqual(missing['summary']['zero_steps'],[])
            self.assertNotIn('zero_start',ns['_wafer_step_timing']('w',{'meta':meta})['steps'])

    def test_js_zero_selection_math(self):
        html=Path('src/app/static/index.html').read_text()
        code=html[html.index('function differenceStats('):html.index('function differenceSummary(')]
        code+=html[html.index('function durationStepRange('):html.index('// A user-selected reference period')]
        code+=html[html.index('function chamberMeanPhaseName('):html.index('function renderChamberMeanAutoPlot(')]
        code+='''
function assert(v,m){if(!v)throw Error(m);}
var s={step_stats:{zero_start:{mean:12},zero_end:{mean:110},1:{mean:30}}};
assert(differenceStats(s,'zero_start').mean===12,'start mean');
assert(differenceStats(s,'zero_end').mean===110,'end mean');
assert(differenceStats(s,'step:1').mean===30,'positive unchanged');
assert(differenceStats(s,'range:0:00').mean===(12+30+110)/3,'boundary difference average');
assert(differenceStats(s,'range:00:1')===null,'reverse invalid');
assert(differenceStats(s,'range:0:00',null,[1,2])===null,'missing positive invalid');
var r={step_timing:{steps:{zero_start:{seconds:5},zero_end:{seconds:4}}},sensor_metrics:{mean:{P:{zero_start:12,zero_end:110}}},peaks:{P:{zero_start:14,zero_end:120}}};
assert(durationRangeSeconds(r,'zero_start')===5,'start seconds');
assert(durationRangeSeconds(r,'zero_end')===4,'end seconds');
assert(durationSensorValue(r,'P','zero_end','mean')===110,'end mean');
assert(durationSensorValue(r,'P','zero_start','max')===14,'start peak');
assert(durationRangeSeconds({},'zero_end')===null,'missing not zero');
var points=[],rows={};[1,2].forEach(function(ch){for(var i=0;i<3;i++){var wid=ch+'_'+i;points.push({wafer_id:wid,ts:'2026-10-20T00:0'+i+':00',chamber:'dsv'+ch,recipe:'A'});rows['2026-10-20/'+wid]={sensor_series:[{sensor:'P',step_stats:{zero_start:{mean:ch*10},zero_end:{mean:ch*100}}}]};}});
var result=chamberMeanAuto(points,rows,{phases:['zero_end']});
assert(result.ranked.length===1&&result.ranked[0].label==='스텝 00번','zero comparison without valves');
'''
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.js';path.write_text(code)
            result=subprocess.run(['gjs',str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
