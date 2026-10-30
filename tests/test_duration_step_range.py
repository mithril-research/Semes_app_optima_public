from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('gjs'), 'requires gjs')
class DurationStepRangeTests(unittest.TestCase):
    def test_range_aggregation_and_missing_data(self):
        html = Path('src/app/static/index.html').read_text()
        code = html[html.index('function durationStepRange('):html.index('// A user-selected reference period')]
        checks = '''
function assert(v,m){if(!v)throw Error(m);}
var row={step_timing:{steps:{1:{seconds:2},2:{seconds:3}}},peaks:{P:{1:8,2:20}},sensor_metrics:{mean:{P:{1:3,2:10}},min:{P:{1:-2,2:4}},std:{P:{1:2,2:4}}}};
assert(durationRangeSeconds(row,'range:1:2')===5,'duration sum');
assert(durationSensorValue(row,'P','range:1:2','max')===20,'range maximum');
assert(durationSensorValue(row,'P','range:1:2','min')===-2,'range minimum');
assert(durationSensorValue(row,'P','range:1:2','mean')===6.5,'equal step mean');
assert(Math.abs(durationSensorValue(row,'P','range:1:2','std')-Math.sqrt(22.25))<1e-12,'equal-step mixture deviation');
assert(durationSensorValue(row,'P','1','max')===8,'single step compatible');
assert(durationRangeSeconds(row,'range:1:3')===null,'missing time not partial sum');
assert(durationSensorValue(row,'P','range:1:3','max')===null,'missing peak not partial max');
assert(durationStepRange('range:3:1')===null&&durationStepRange('00:0')===null,'invalid range');

row.step_timing.steps.zero_start={seconds:4};row.step_timing.steps.zero_end={seconds:6};
row.peaks.P.zero_start=2;row.peaks.P.zero_end=30;
row.sensor_metrics.mean.P.zero_start=1;row.sensor_metrics.mean.P.zero_end=15;
assert(durationRangeSeconds(row,'range:0:00')===15,'all including both boundaries');
assert(durationRangeSeconds(row,'range:0:1')===6,'leading range');
assert(durationRangeSeconds(row,'range:2:00')===9,'trailing range');
assert(durationRangeSeconds(row,'range:00:00')===6,'single trailing');
assert(durationRangeSeconds(row,'range:0:0')===4,'single leading');
assert(durationSensorValue(row,'P','range:0:00','mean')===7.25,'boundary mean');
assert(durationSensorValue(row,'P','range:0:00','max')===30,'boundary max');
assert(durationStepRange('range:00:1')===null&&durationStepRange('range:1:0')===null,'reverse boundaries');
['','000','-1','1.5','1e2'].forEach(function(v){assert(durationStepRange('range:'+v+':00')===null,'invalid token');});
row.process_steps=[1,2,3];assert(durationSensorValue(row,'P','range:0:00','mean')===null,'missing last positive cannot shorten range');
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'check.js'
            path.write_text(code + checks)
            result = subprocess.run(['gjs', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
