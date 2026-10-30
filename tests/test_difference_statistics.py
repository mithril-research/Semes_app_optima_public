import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from src.app.sensor_differences import difference_series


class DifferenceStatisticsTests(unittest.TestCase):
    def test_signed_population_stats_and_missing_steps(self):
        arch=dict(actual=np.array([[999,0],[1,0],[3,0],[-2,0],[np.nan,0]]),
                  step=np.array([0,1,1,3,3]),meta={'sensor_cols':['p_008','p_007']})
        r=difference_series(arch);s=r['series'][0]['step_stats']
        self.assertEqual(s['1'],dict(mean=2.,min=1.,max=3.,std=1.,count=2))
        self.assertEqual(s['3']['std'],0)
        self.assertNotIn('0',s);self.assertNotIn('2',s)
        self.assertEqual(r['series'][0]['values'],[0.])
        json.dumps(r,allow_nan=False)

    def test_no_synthetic_statistics_for_missing_pair(self):
        r=difference_series(dict(actual=np.ones((2,1)),step=np.array([1,1]),meta={'sensor_cols':['p_008']}))
        self.assertEqual(r['series'][0]['step_stats'],{})

    def test_individual_sensor_stats_use_measurements_and_preserve_names(self):
        arch=dict(actual=np.array([[999.,1.],[2.,10.],[4.,14.],[np.nan,20.]]),
                  step=np.array([0,1,1,2]),meta={'sensor_cols':['t_Test_PV','p_Test_PV']},
                  z=np.full((4,2),9999.))
        rows={r['sensor']:r['step_stats'] for r in difference_series(arch)['sensor_series']}
        self.assertEqual(rows['t_Test_PV']['1'],dict(mean=3.,min=2.,max=4.,std=1.,count=2))
        self.assertEqual(rows['p_Test_PV']['1']['std'],2.)
        self.assertNotIn('0',rows['t_Test_PV'])
        self.assertNotIn('2',rows['t_Test_PV'])
        self.assertEqual(rows['p_Test_PV']['2']['mean'],20.)

    @unittest.skipUnless(shutil.which('gjs'),'SpiderMonkey required')
    def test_equal_step_weights_population_std_and_filtered_summary(self):
        s=Path('src/app/static/index.html').read_text();s=s[s.index('function recipeStepEnds'):s.index('// Physical sensor differences:')]
        s+='''
function assert(v,m){if(!v)throw Error(m);}
var row={step_stats:{'1':{mean:2,min:1,max:3,std:1,count:100},'3':{mean:-2,min:-2,max:-2,std:0,count:1}}};
var phaseRow={step_stats:{}};for(var st=1;st<=26;st++)phaseRow.step_stats[st]={mean:st,min:st,max:st,std:0};
[['phase:pressurize',2.5,1,4],['phase:wait',5,5,5],['phase:process',15,6,24],['phase:discharge',25.5,25,26]].forEach(function(p){var result=differenceStats(phaseRow,p[0],26);assert(result.mean===p[1]&&result.min===p[2]&&result.max===p[3],'phase endpoints '+p[0]);});
assert(differenceStats({export_step_stats:phaseRow.step_stats},'phase:process',26).mean===15,'export phase');
assert(differenceStats(phaseRow,'phase:discharge',22).mean===21.5,'22 step discharge');
assert(differenceStats(phaseRow,'phase:process',22).mean===13,'22 step process');
assert(differenceStats(phaseRow,'phase:discharge',undefined)===null,'unknown recipe end');
assert(differenceStats({step_stats:{'21':{mean:21}}},'phase:discharge',22)===null,'incomplete discharge');
assert(differenceStats({step_stats:{}},'phase:discharge',26)===null,'missing phase');
var r=differenceStats(row,'block:1');
assert(r.mean===0&&r.min===-2&&r.max===3,'step weights or bounds');
assert(Math.abs(r.std-Math.sqrt(4.5))<1e-12,'mixture variance');
assert(differenceStats(row,'step:1').std===1,'single step');
assert(differenceStats(row,'step:2')===null,'missing step');
var a=differenceSummary([1,3,null,NaN]);assert(a.count===2&&a.mean===2&&a.std===1,'summary');
assert(differenceSummary([])===null,'empty');
assert(differenceSummary([5]).std===0,'single wafer');
void 0;
'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'check.js';p.write_text(s)
            r=subprocess.run(['gjs',str(p)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr+r.stdout)

if __name__=='__main__':unittest.main()
