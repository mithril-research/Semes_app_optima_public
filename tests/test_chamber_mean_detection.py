"""Exercise the exact browser calculation with SpiderMonkey, including missing data."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('gjs'), 'requires gjs')
class ChamberMeanDetectionTests(unittest.TestCase):
    def test_comparison_rules(self):
        html = Path('src/app/static/index.html').read_text()
        calculate = html[html.index('function chamberMeanPhaseName('):html.index('function mountChamberMeanDetection(')]
        calculate = html[html.index('function recipeStepEnds('):html.index('function differenceStats(')] + calculate
        checks = r'''
function assert(v,m){if(!v)throw Error(m);}
var phasePoints=[],phaseRows={};
[22,26].forEach(function(end){['dsv1','dsv2','dsv3'].forEach(function(ch){var p={wafer_id:ch+'_'+end,ts:'2026-10-01T12:00:00',recipe:'recipe'+end,chamber:ch};phasePoints.push(p);var table={};for(var st=1;st<=end;st++)table[st]={mean:st+(ch==='dsv3'?100:0)};phaseRows['2026-10-01/'+p.wafer_id]={valve_phases:{version:2,status:'ok',steps:{discharge:[end-1,end]}},sensor_series:[{sensor:'p_test',step_stats:table}]};});});
var phaseResult=chamberMeanDifferences(phasePoints,phaseRows,{stepMode:'discharge',minimum:1,mode:'absolute',pressure:75,temperature:75});
assert(phaseResult.found.length===2,'recipe-specific discharge detection');
assert(phaseResult.found.some(r=>r.recipe==='recipe22'&&r.step==='배출'&&r.mean===121.5),'22 steps chamber comparison');
assert(phaseResult.found.some(r=>r.recipe==='recipe26'&&r.step==='배출'&&r.mean===125.5),'26 steps chamber comparison');

var missingPhases=JSON.parse(JSON.stringify(phaseRows));Object.values(missingPhases).forEach(function(r){delete r.valve_phases;});
assert(chamberMeanDifferences(phasePoints,missingPhases,{stepMode:'discharge',minimum:1,mode:'absolute',pressure:75,temperature:75}).phaseMissing===6,'no fixed-range fallback');
var dynamicPoints=[{wafer_id:'a',ts:'2026-10-01',recipe:'same',chamber:'dsv1'},{wafer_id:'b',ts:'2026-10-01',recipe:'same',chamber:'dsv2'}];
var dynamicRows={'2026-10-01/a':{valve_phases:{version:2,status:'ok',steps:{supply:[1,2]}},sensor_series:[{sensor:'p_test',step_stats:{1:{mean:10},2:{mean:20}}}]},'2026-10-01/b':{valve_phases:{version:2,status:'ok',steps:{supply:[1,2,3]}},sensor_series:[{sensor:'p_test',step_stats:{1:{mean:30},2:{mean:40},3:{mean:50}}}]}};
var dynamicResult=chamberMeanDifferences(dynamicPoints,dynamicRows,{stepMode:'supply',minimum:1,mode:'absolute',pressure:1,temperature:1});
assert(dynamicResult.found.length===2&&dynamicResult.found.find(r=>r.chamber==='dsv1').mean===15&&dynamicResult.found.find(r=>r.chamber==='dsv2').mean===40,'compare phase means with wafer-specific boundaries');
var pts=[],rows={};
function add(ch,value,recipe,step,sensor,id){
 var wid=id||String(pts.length),p={ts:'2026-10-01T12:00:00',wafer_id:wid,chamber:ch,recipe:recipe||'A'};
 pts.push(p);var stats={};stats[step===undefined?'1':step]={mean:value};
 rows['2026-10-01/'+wid]={sensor_series:[{sensor:sensor||'p_008_PV',step_stats:stats}]};
}
var opt={minimum:3,mode:'absolute',pressure:40,temperature:5};
for(var ch=1;ch<=4;ch++)for(var w=0;w<3;w++)add('dsv'+ch,ch===4?140:100);
var r=chamberMeanDifferences(pts,rows,opt);
assert(r.found.length===1&&r.found[0].chamber==='dsv4','isolate shifted chamber');
assert(r.found[0].mean===140&&r.found[0].reference===100&&r.found[0].count===3,'means/counts');
assert(r.found[0].chamberMeans.length===4&&r.found[0].chamberMeans.find(function(c){return c.chamber==='dsv1';}).mean===100,'peer chamber means retained');
add('dsv5',999);var detail=chamberMeanDifferences(pts,rows,opt).found[0];
assert(detail.reference===100&&detail.peers===3,'insufficient chamber excluded from reference');
assert(detail.chamberMeans.find(function(c){return c.chamber==='dsv5';}).eligible===false,'excluded chamber visibly marked');
assert(detail.chamberMeans.find(function(c){return c.chamber==='dsv5';}).count===1,'excluded sample count retained');
opt.pressure=40.01;assert(chamberMeanDifferences(pts,rows,opt).found.length===0,'threshold boundary');
opt.mode='percent';opt.pressure=40;assert(chamberMeanDifferences(pts,rows,opt).found.length===1,'percent boundary');
opt.minimum=4;assert(chamberMeanDifferences(pts,rows,opt).compared===0,'minimum sample count');
opt.minimum=3;
var before=chamberMeanDifferences(pts,rows,opt);pts.push(pts[0]);assert(chamberMeanDifferences(pts,rows,opt).wafers===before.wafers,'deduplicate wafer');
add('dsv1',9999,'B');add('dsv1',9999,'A',0);add('dsv1',NaN,'A',2);add('dsv1',9999,'A',1,'p_sensor_SV');
assert(chamberMeanDifferences(pts,rows,opt).found.length===1,'recipe/step/missing/setpoint separation');
pts=[];rows={};opt.minimum=1;opt.mode='absolute';opt.temperature=5;
add('dsv1',0,'A',2,'t_002_PV');add('dsv2',10,'A',2,'t_002_PV');
r=chamberMeanDifferences(pts,rows,opt);assert(r.found.length===2,'two chambers flag both');
opt.mode='percent';r=chamberMeanDifferences(pts,rows,opt);assert(r.zeroReference===1,'zero reference is not infinity');
pts=[];rows={};opt.mode='absolute';opt.pressure=1;
add('dsv1',1);add('dsv1',3);add('dsv2',4);
r=chamberMeanDifferences(pts,rows,opt);assert(r.found.find(function(x){return x.chamber==='dsv1';}).mean===2,'equal wafer weights');
assert(r.found.find(function(x){return x.chamber==='dsv2';}).reference===2,'do not pool wafer counts');
var expected=JSON.stringify(r);Object.keys(rows).forEach(function(k){rows[k].sensor_series.forEach(function(s){s.export_step_stats=s.step_stats;delete s.step_stats;});});
assert(JSON.stringify(chamberMeanDifferences(pts,rows,opt))===expected,'exported means agree');
Object.keys(rows).forEach(function(k){rows[k].sensor_series.forEach(function(s){Object.keys(s.export_step_stats).forEach(function(st){delete s.export_step_stats[st].mean;});});});
assert(chamberMeanDifferences(pts,rows,opt).compared===0,'missing export means not treated as zero');
pts=[];rows={};opt.minimum=1;opt.pressure=1;
// Two wafers per chamber, each with a complete multi-step range.
for(var c=1;c<=2;c++)for(var w=0;w<2;w++){
 add('dsv'+c,0);var k='2026-10-01/'+String(pts.length-1);
 rows[k].sensor_series[0].step_stats={'4':{mean:9999},'5':{mean:10*c+10*w,count:1000},'6':{mean:30*c+10*w,count:1},'7':{mean:9999}};
}
opt.stepFrom=5;opt.stepTo=6;r=chamberMeanDifferences(pts,rows,opt);
assert(r.found.length===2&&r.found.every(function(x){return x.step==='5~6'&&x.count===2;}),'one range result per chamber, count wafers not steps');
assert(r.found.find(function(x){return x.chamber==='dsv1';}).mean===25,'equal step weighting then equal wafer weighting');
assert(r.found.find(function(x){return x.chamber==='dsv2';}).mean===45,'second chamber range mean');
opt.stepFrom=6;opt.stepTo=6;assert(chamberMeanDifferences(pts,rows,opt).found.find(function(x){return x.chamber==='dsv1';}).mean===35,'single step range');
opt.stepFrom=20;opt.stepTo=25;assert(chamberMeanDifferences(pts,rows,opt).compared===0,'absent range');
opt.stepFrom=null;opt.stepTo=null;assert(chamberMeanDifferences(pts,rows,opt).found.length===4,'all mode retains individual step comparisons');
Object.keys(rows).forEach(function(k){rows[k].sensor_series.forEach(function(s){s.export_step_stats=s.step_stats;delete s.step_stats;});});
opt.stepFrom=5;opt.stepTo=6;assert(chamberMeanDifferences(pts,rows,opt).found.length===2,'export range averages');
delete rows['2026-10-01/0'].sensor_series[0].export_step_stats['6'];
r=chamberMeanDifferences(pts,rows,opt);assert(r.found.find(function(x){return x.chamber==='dsv1';}).count===1,'incomplete wafer excluded');
assert(r.found.find(function(x){return x.chamber==='dsv1';}).mean===30,'partial range not averaged');
print('PASS chamber mean detection');
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js') as f:
            f.write(calculate + checks)
            f.flush()
            run = subprocess.run(['gjs', f.name], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
