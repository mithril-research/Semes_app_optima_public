"""All archived measurement channels, independent groups and exact pair statistics."""
from pathlib import Path
import json
import subprocess
import unittest
import numpy as np
from src.app.sensor_differences import difference_series, step_sensor_metrics, compact_trend_export


class TrendSensorGroupsTests(unittest.TestCase):
    def test_input_channels_keep_statistics_and_valves_are_excluded(self):
        archive = dict(meta={'sensor_cols':['t_test', 'p_test'],
                             'rca_in_cols':['s_010_PV', 's_011_PV', 't_test'],
                             'rca_io_cols':['9000009', 'ValveOther']},
                       actual=np.array([[9,9],[10,20],[14,24],[30,40]],float),
                       rca_insen=np.array([[999,999,999],[2,1,999],[8,3,999],[10,4,999]],float),
                       rca_io=np.array([[0,0],[1,0],[0,1],[1,1]],float),
                       step=np.array([0,1,1,2]))
        row=difference_series(archive,pairs=[['s_010_PV','s_011_PV'],['9000009','ValveOther']])
        sensors={s['sensor']:s['step_stats'] for s in row['sensor_series']}
        self.assertEqual(set(sensors), {'t_test','p_test','s_010_PV','s_011_PV'})
        self.assertEqual(sensors['t_test']['1']['mean'],12) # actual wins duplicate input
        self.assertEqual(sensors['s_010_PV']['1'],dict(mean=5.,min=2.,max=8.,std=3.,count=2))
        self.assertNotIn('9000009', sensors)
        self.assertNotIn('ValveOther', sensors)
        diff=row['series'][-2]['step_stats']['1']
        self.assertEqual((diff['mean'],diff['min'],diff['max'],diff['std']), (3.,1.,5.,2.))
        self.assertEqual(step_sensor_metrics(row)['mean']['s_010_PV']['1'],5.)
        export=compact_trend_export(row,['mean','min','max','std'])
        self.assertEqual(export['series'][-2]['export_step_stats']['1']['std'],2.)
        self.assertEqual(len(export['sensor_series']),4)
        json.dumps(export,allow_nan=False)

    def test_missing_or_malformed_additional_arrays_are_not_fabricated(self):
        archive=dict(meta={'sensor_cols':['s_010_PV'],'rca_io_cols':['Valve']},
                     actual=np.array([[2.],[4.]]),step=np.array([1,1]),rca_io=np.array([[1.]]))
        row=difference_series(archive)
        self.assertEqual([s['sensor'] for s in row['sensor_series']],['s_010_PV'])
        self.assertEqual(row['sensor_series'][0]['step_stats']['1']['mean'],3.)

    def test_group_partition_ranking_and_shared_cache(self):
        s=Path('src/app/static/index.html').read_text()
        js=s[s.index('function scSensorGroup('):s.index('function scSensorColors(')]
        js+=s[s.index('function trendSensorMatches('):s.index('function mountStepTimeTrend(')]
        js+=s[s.index('function chamberMeanPhaseName('):s.index('function renderChamberMeanAutoPlot(')]
        js+=r'''
function ok(v,m){if(!v)throw Error(m);}
['t_test','p_test','T1','P2'].forEach(function(n){ok(trendSensorMatches(n,'tp')&&!trendSensorMatches(n,'other'),'temperature pressure');});
['s_010_PV','s_011_PV','h_Heater','h_004_PV'].forEach(function(n){ok(trendSensorMatches(n,'other')&&!trendSensorMatches(n,'tp'),'other');});
['s_001_Press','s_002_Press','s_003_Press','sample_press'].forEach(function(n){ok(!trendSensorMatches(n,'all')&&!trendSensorMatches(n,'other'),'press sensors excluded');});
ok(!trendSensorMatches('9000009','tp')&&!trendSensorMatches('9000009','other'),'valves excluded');
ok(trendStateKey('timeTrend','tp')==='timeTrend'&&trendStateKey('timeTrend','other')==='timeTrendOther','independent settings');
var points=[],rows={};[1,2,3].forEach(function(ch){[0,1,2].forEach(function(i){var id='w'+ch+i;points.push({wafer_id:id,ts:'2026-10-19T00:00:0'+i,chamber:'dsv'+ch,recipe:'A'});rows['2026-10-19/'+id]={valve_phases:{version:2,status:'ok',steps:{process:[1]}},sensor_series:['p_test','s_010_PV'].map(function(n){return {sensor:n,step_stats:{1:{mean:(ch===3?20:10)+i}}};})};});});
var tp=chamberMeanAuto(points,rows,{sensorGroup:'tp'}),other=chamberMeanAuto(points,rows,{sensorGroup:'other'});
ok(tp.ranked.length===1&&tp.ranked[0].sensor==='p_test','tp ranking');
ok(other.ranked.length===1&&other.ranked[0].sensor==='s_010_PV'&&other.ranked[0].chamber==='dsv3','other ranking');
var cache={a:{series:[{left:'s_010_PV',right:'s_011_PV',value:1}]}};
mergeTrendRow(cache,'a',{series:[{left:'p_a',right:'p_b',value:2}]});ok(cache.a.series.length===2,'preserve other group pairs');
mergeTrendRow(cache,'a',{series:[{left:'s_010_PV',right:'s_011_PV',value:3}]});ok(cache.a.series.length===2&&cache.a.series[0].value===3,'replace same pair');
'''
        result=subprocess.run(['gjs','-c',js],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
