from pathlib import Path
import shutil, subprocess, tempfile, unittest

@unittest.skipUnless(shutil.which('gjs'),'requires SpiderMonkey')
class MovingAverageTests(unittest.TestCase):
    def test_time_window_grouping_missing_first_and_range(self):
        s=Path('src/app/static/index.html').read_text();s=s[s.index('function sensorMovingAverage'):s.index('// Physical sensor differences:')]
        s+='''
function assert(v,m){if(!v)throw Error(m);}
var pts=[0,1,2,3,4].map(function(m){return {ts:'2026-09-18T12:0'+m+':00',chamber:'a',recipe:'r'};});
var r=sensorMovingAverage(pts,[1,3,null,7,9],2);
assert(r[1].value===2&&r[2]===null&&r[3].value===5&&r[4].value===8,'inclusive time window');
pts[1].chamber='b';r=sensorMovingAverage(pts,[1,100,3,5,7],60);assert(r[2].value===2,'chambers mixed');
pts[2].recipe='other';r=sensorMovingAverage(pts,[1,100,300,5,7],60);assert(r[3].value===3,'recipes mixed');
pts[0].lot_first=true;r=sensorMovingAverage(pts,[1,100,300,5,7],60);assert(r[0]===null&&r[3].value===5,'first exclusion');
r=sensorMovingAverage(pts,[1,100,300,5,7],60,true);assert(r[0]===null&&r[3].value===5,'first always excluded even with legacy include flag');
pts.forEach(p=>p.lot_first=true);assert(sensorMovingAverage(pts,[1,2,3,4,5],60).every(x=>x===null),'all first yields no average');pts.forEach(p=>p.lot_first=false);
r=sensorMovingAverage(pts.slice(3),[5,7],60);assert(r[0].value===5&&r[1].value===6,'selected range only');
assert(sensorMovingAverage(pts,[null,null,null,null,null],60).every(x=>x===null),'missing');
pts[4].ts='2026-09-20T12:00:00';r=sensorMovingAverage(pts.slice(3),[5,9],null);assert(r[1].value===7,'full query cumulative window');
r=sensorMovingAverage(pts.slice(3),[5,9],60);assert(r[1].value===9,'custom window still expires');
void 0;
'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.js';p.write_text(s);r=subprocess.run(['gjs',str(p)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr+r.stdout)
