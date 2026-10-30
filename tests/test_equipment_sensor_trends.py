"""Verify filename-derived equipment groups remain isolated across chambers."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('gjs'), 'requires gjs')
class EquipmentSensorTrendTests(unittest.TestCase):
    def test_equipment_and_chamber_partition(self):
        html = Path('src/app/static/index.html').read_text()
        functions = html[html.index('function sensorTrendEquipment('):html.index('function mountSensorDifferences(')]
        functions += html[html.index('function sensorMovingAverage('):html.index('// Physical sensor differences:')]
        checks = r'''
function assert(v,m){if(!v)throw Error(m);}
assert(sensorTrendEquipment({wafer_id:'WUK370A_20261008_001'})==='WUK370A','underscore');
assert(sensorTrendEquipment({wafer_id:'WUK370B*20261008_001'})==='WUK370B','asterisk');
assert(sensorTrendEquipment({wafer_id:'TOOL_X*001'})==='TOOL_X','asterisk precedence matches server');
assert(sensorTrendEquipment({wafer_id:'legacy',tool:'known'})==='known','explicit metadata');
assert(sensorTrendEquipment({})==='설비 미상','missing equipment');
var points=[
 {wafer_id:'WUK370B_2',chamber:'dsv1',ts:'2026-10-08T12:02:00'},
 {wafer_id:'WUK370A_1',chamber:'dsv1',ts:'2026-10-08T12:01:00'},
 {wafer_id:'WUK370B*1',chamber:'dsv1',ts:'2026-10-08T12:00:00'},
 {wafer_id:'WUK370A_3',chamber:'dsv2',ts:'2026-10-08T12:03:00'}];
var before=JSON.stringify(points),groups=equipmentTrendGroups(points);
assert(groups.length===2,'one graph group per chamber');
assert(groups[0].chamber==='dsv1'&&groups[0].points.length===3,'all equipment in same chamber');
assert(groups[0].points[0].wafer_id==='WUK370B*1','time order across equipment');
assert(groups[1].chamber==='dsv2'&&groups[1].points.length===1,'chambers stay separate');
assert(equipmentTrendColor('WUK370A')!==equipmentTrendColor('WUK370B'),'example equipment colors distinct');
assert(equipmentTrendColor('WUK370A')===equipmentTrendColor('WUK370A'),'stable color');
assert(JSON.stringify(points)===before,'input order untouched');
assert(equipmentTrendGroups([]).length===0,'empty selection');
var mixed=[0,1,2,3].map(function(i){return {wafer_id:(i%2?'WUK370B':'WUK370A')+'_'+i,chamber:'dsv1',recipe:'r',ts:'2026-10-08T12:0'+i+':00'};});
var averages=sensorMovingAverage(mixed,[10,100,20,200],null,'equipment');
assert(averages[2].value===15&&averages[3].value===150,'moving average separated by equipment');
var pooled=sensorMovingAverage(mixed,[10,100,20,200],null);
assert(pooled[3].value===82.5,'existing pooled mode unchanged');

'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'check.js'
            path.write_text(functions + checks)
            result = subprocess.run(['gjs', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
