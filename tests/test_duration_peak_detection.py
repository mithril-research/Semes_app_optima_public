from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('gjs'), 'requires gjs')
class DurationPeakDetectionTests(unittest.TestCase):
    def test_reference_fit_and_detection(self):
        html=Path('src/app/static/index.html').read_text()
        code=html[html.index('function durationPeakDetection('):html.index('function mountDurationPeakDetection(')]
        checks='''
function assert(v,m){if(!v)throw Error(m);}
var points=[];for(var i=0;i<20;i++)points.push({x:i+1,y:2*(i+1)+10,p:{ts:'2026-10-01T12:00:00'}});
points.push({x:10,y:40,p:{ts:'2026-10-02T12:00:00'}});
points.push({x:10,y:20,p:{ts:'2026-10-02T12:00:00'}});
points.push({x:30,y:999,p:{ts:'2026-10-02T12:00:00'}});
var options={from:Date.parse('2026-10-01T00:00:00'),to:Date.parse('2026-10-01T23:59:59'),minimum:20,tolerance:10,direction:'both'};
var r=durationPeakDetection(points,options);assert(r.referenceCount===20,'period exclusion');assert(r.predict(10)===30,'duration prediction');assert(r.flagged.length===2,'inclusive threshold high and low');assert(r.outside===1,'no extrapolation');
options.direction='high';assert(durationPeakDetection(points,options).flagged.length===1,'high only');
options.direction='low';var low=durationPeakDetection(points,options);assert(low.flagged.length===1&&low.flagged[0].delta===-10,'low only including boundary');
options.tolerance=11;assert(durationPeakDetection(points,options).flagged.length===0,'threshold change');
options.minimum=21;assert(durationPeakDetection(points,options).error,'insufficient reference');options.minimum=3;
assert(durationPeakDetection(points.map(function(p){return {x:1,y:p.y,p:p.p};}),options).error,'constant duration');
assert(durationPeakDetection([{x:null,y:1,p:{ts:'2026-10-01'}}],options).error,'missing values not zero');
print('PASS');
'''
        with tempfile.NamedTemporaryFile(mode='w',suffix='.js') as f:
            f.write(code+checks);f.flush()
            result=subprocess.run(['gjs',f.name],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
