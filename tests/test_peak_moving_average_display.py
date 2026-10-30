from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('gjs'), 'requires gjs')
class PeakMovingAverageDisplayTests(unittest.TestCase):
    def test_single_and_coincident_points_have_visible_marks(self):
        html = Path('src/app/static/index.html').read_text()
        code = html[html.index('function peakMovingAverageStroke('):html.index('function durationStepRange(')]
        checks = '''
function esc(v){return String(v);}function sensorChamberDash(){return 'none';}
function assert(v,m){if(!v)throw Error(m);}
var metric=['mean','평균','dsv1','#c05600'];
var group={ch:'dsv1',recipe:'R',points:['10,20']};
assert(peakMovingAverageStroke(group,metric).includes('class="peak-ma-single"'),'singleton visible');
group.points=['10,20','10,20'];assert(peakMovingAverageStroke(group,metric).includes('class="peak-ma-single"'),'coincident points visible');
group.points=['10,20','20,30'];var svg=peakMovingAverageStroke(group,metric);assert(svg.includes('peak-moving-average-line')&&!svg.includes('peak-ma-halo'),'no heavy halo');assert(svg.includes('points="10,20 20,30"'),'unchanged coordinates');
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'check.js'
            path.write_text(code + checks)
            result = subprocess.run(['gjs', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
