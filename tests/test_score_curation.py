"""Score curation must preserve group boundaries and use displayed scores."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey')
class ScoreCurationTests(unittest.TestCase):
    def check_js(self, checks):
        source = Path('src/app/static/index.html').read_text()
        source = source[source.index('function detectScoreRegions('):source.index('function mountScoreCuration(')]
        source += '''
function assert(v,m){if(!v)throw Error(m);}
function series(values, extra){return values.map(function(v,i){return Object.assign({wafer_id:'WUK_A_'+i,chamber:'DSV1',recipe:'29a',tool:'WUK',ratio:v,ratio_thr:99,ts:new Date(Date.UTC(2026,0,1,0,i)).toISOString()},extra||{});});}
function detect(rows,opt){return detectScoreRegions(rows,function(p){return p.tool;},opt);}
var normal=Array(20).fill(.5);
'''
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'check.js'
            path.write_text(source + checks + '\nvoid 0;\n')
            result = subprocess.run(['gjs', str(path)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lot_ranking_context_and_missing_metadata(self):
        self.check_js('''
var points=[];[.2,.4,1,3,.5,2,.3].forEach(function(value,l){series(Array(6).fill(value)).forEach(function(p,i){p.lot='L'+l;p.wafer_id='W_'+l+'_'+i;p.ts=new Date(Date.UTC(2026,0,1,0,l*10+i)).toISOString();points.push(p);});});
var before=JSON.stringify(points),r=rankCurationLots(points,p=>p.tool,5);
assert(r.regions.map(x=>x.lot).join()==='L3,L5,L2,L4,L1','top mean order');
var best=r.regions[0],context=scoreRegionSummary(points,best,p=>p.tool);
assert(best.beforeLots.map(x=>x.lot).join()==='L1,L2'&&best.afterLots.map(x=>x.lot).join()==='L4,L5','two on each side');
assert(context.points.length===30&&context.selected.length===6&&context.normal.length===6&&context.after.length===12,'whole LOT contexts');
assert(context.roles.filter(x=>x==='after').length===12,'after not selected');assert(best.normalLot.lot==='L1','skip preceding abnormal LOT');
var distant=points.map(p=>Object.assign({},p,{ratio:p.lot==='L1'?1.5:p.ratio})),far=rankCurationLots(distant,p=>p.tool,1).regions[0];assert(far.normalLot.lot==='L0'&&far.beforeLots.length===3,'reach farther normal LOT including intervening LOTs');
var none=rankCurationLots(points.map(p=>Object.assign({},p,{ratio:p.ratio+1})),p=>p.tool,1).regions[0];assert(!none.normalLot&&!none.beforeLots.length,'no invented normal reference');
assert(JSON.stringify(points)===before,'input immutable');
var copy=points.concat(points,[Object.assign({},points[0],{wafer_id:'unknown',lot:''})]);
var d=rankCurationLots(copy,p=>p.tool,1);assert(d.regions.length===1&&d.duplicates===42&&d.excluded===1,'dedup and unknown');
[0,6,1.5,NaN].forEach(function(n){var rejected=false;try{rankCurationLots(points,p=>p.tool,n);}catch(e){rejected=true;}assert(rejected,'top bounds');});
''')

    def test_reused_lot_name_does_not_extend_context(self):
        self.check_js("""
var points=series([.2,4,.3,.4,.5,.6]);['N','TOP','A','B','A','C'].forEach((lot,i)=>points[i].lot=lot);
var best=rankCurationLots(points,p=>p.tool,1).regions[0],ctx=scoreRegionSummary(points,best,p=>p.tool);
assert(best.afterLots.map(x=>x.lot).join()==='A,B','two subsequent runs');
assert(ctx.points.length===4&&!ctx.points.includes(points[4]),'later reused LOT excluded');
""")

    def test_lot_average_not_peak_and_group_boundaries(self):
        self.check_js('''
var a=series([10,0,0,0],{lot:'A'}),b=series([3,3],{lot:'B'});b.forEach((p,i)=>{p.wafer_id='B'+i;p.ts=new Date(Date.parse(p.ts)+3600000).toISOString();});
var r=rankCurationLots(a.concat(b),p=>p.tool,2);assert(r.regions[0].lot==='B','mean rather than single extreme');
var foreign=series([100],{lot:'A',tool:'OTHER'}),all=rankCurationLots(a.concat(b,foreign),p=>p.tool,5);
assert(all.total===3&&all.regions[0].beforeLots.length===0&&all.regions[0].afterLots.length===0,'same lot separate equipment');
var partial=rankCurationLots(a.slice(0,1),p=>p.tool,1);assert(partial.regions[0].mean===10&&partial.regions[0].count===1,'query subset only');
''')

    def test_adjustable_parameters_and_saved_settings_validation(self):
        self.check_js('''
var defaults=scoreCurationSettings();assert(defaults.baseline===20&&defaults.minRun===5,'defaults');
assert(detect(series(Array(5).fill(.5).concat([.9,.9])),{baseline:5,minRun:2}).regions.length===1,'smaller sample window and run');
var small=series(normal.concat(Array(5).fill(.56)));
assert(!detect(small).regions.length&&detect(small,{relative:.1}).regions.length===1,'relative sensitivity');
var tiny=series(Array(20).fill(0).concat(Array(5).fill(.03)));
assert(!detect(tiny).regions.length&&detect(tiny,{absolute:.02}).regions.length===1,'absolute sensitivity');
var noisy=series(Array.from({length:20},(_,i)=>i%2?.6:.4).concat(Array(5).fill(.9)));
assert(!detect(noisy).regions.length&&detect(noisy,{sigma:1}).regions.length===1,'MAD sensitivity');
var gap=series(normal.concat(Array(5).fill(.9)));gap.slice(20).forEach(p=>p.ts=new Date(Date.parse(p.ts)+7200000).toISOString());
assert(!detect(gap).regions.length&&detect(gap,{gapMs:10800000}).regions.length===1,'gap sensitivity');
['baseline','minRun','sigma','relative','absolute','gapMs'].forEach(function(k){[null,'3',NaN,Infinity].forEach(function(v){var cfg=Object.assign({},defaults);cfg[k]=v;var rejected=false;try{scoreCurationSettings(cfg);}catch(e){rejected=true;}assert(rejected,'reject '+k+' '+v);});});
assert(JSON.stringify(scoreCurationSettings(defaults))===JSON.stringify(defaults),'saved round trip');
''')

    def test_high_low_and_return_to_baseline(self):
        self.check_js('''
var r=detect(series(normal.concat(Array(7).fill(.9),Array(3).fill(.5),Array(5).fill(.2))));
assert(r.regions.length===2,'two separate shifts');
assert(r.regions[0].direction==='high'&&r.regions[0].count===7,'high extent');
assert(r.regions[1].direction==='low'&&r.regions[1].count===5,'low extent');
assert(Math.abs(r.regions[0].mean-.9)<1e-10,'displayed ratio, not stored ratio_thr');
assert(r.regions[0].waferIds[0]==='WUK_A_20','correct boundary');
''')

    def test_noise_short_runs_and_threshold_boundary(self):
        self.check_js('''
assert(!detect(series(normal.concat([.9,.9,.9,.9,.5]))).regions.length,'short run');
assert(!detect(series(Array.from({length:100},(_,i)=>.5+Math.sin(i)*.025))).regions.length,'stable noise');
assert(!detect(series(Array(20).fill(1).concat(Array(5).fill(1.2)))).regions.length,'strict boundary');
var alternating=Array.from({length:30},(_,i)=>i%2?.8:.2);
assert(!detect(series(alternating)).regions.length,'MAD suppresses variable baseline');
''')

    def test_group_isolation_sort_and_duplicates(self):
        self.check_js('''
var a=series(normal.concat(Array(5).fill(.9))), b=series(Array(25).fill(4),{tool:'OTHER'});
var before=JSON.stringify(a);
var r=detect(a.concat(b).reverse());assert(r.regions.length===1&&r.regions[0].equipment==='WUK','equipment isolation');
assert(JSON.stringify(a)===before,'input not mutated');
['recipe','chamber'].forEach(function(k){var x=series(Array(25).fill(4));x.forEach(p=>p[k]='OTHER');assert(detect(a.concat(x)).regions.length===1,k+' isolation');});
assert(detect(a.concat(a)).regions[0].count===5,'deduplicate');
assert(detect(a.concat(a)).duplicates===25,'duplicate report');
''')

    def test_missing_gap_and_minimum_samples(self):
        self.check_js('''
assert(!detect(series(Array(19).fill(.5).concat(Array(5).fill(.9)))).regions.length,'not enough baseline');
var gap=series(normal.concat(Array(5).fill(.9)));gap.slice(20).forEach(p=>p.ts=new Date(Date.parse(p.ts)+7200000).toISOString());
assert(!detect(gap).regions.length,'gap resets comparison');
[null,NaN,Infinity,-1,'0.5'].forEach(function(v){var r=detect(series(normal.concat([v],Array(5).fill(.9))));assert(!r.regions.length&&r.excluded===1,'invalid score breaks sequence');});
assert(detect(series(normal,{recipe:''})).excluded===20,'unknown metadata');
''')

    def test_zero_baseline_and_long_shift(self):
        self.check_js('''
var r=detect(series(Array(20).fill(0).concat(Array(5000).fill(.1))));
assert(r.regions.length===1&&r.regions[0].count===5000,'sustained shift');
assert(r.regions[0].percent===null&&isFinite(r.regions[0].strength),'zero baseline');
assert(!detect(series(Array(20).fill(0).concat(Array(10).fill(.01)))).regions.length,'absolute floor');
var failed=false;try{detect([], {baseline:0});}catch(e){failed=true;}assert(failed,'validate options');
''')


    def test_summary_normal_context_and_selected_statistics(self):
        self.check_js('''
var points=series(normal.concat(Array(6).fill(.9))),region=detect(points).regions[0];
var summary=scoreRegionSummary(points,region,p=>p.tool);
assert(summary.normal.length===8&&summary.selected.length===6,'eight context and six selected');
assert(summary.normal[0].wafer_id==='WUK_A_12','immediate preceding');
assert(summary.points.length===14&&region.count===6,'context does not change region');
assert(summary.selected.every(p=>p.ratio===.9),'no baseline in selected mean');
assert(scoreRegionSummary(points.concat(points),region,p=>p.tool).points.length===14,'deduplicate context');
var other=series(Array(40).fill(.4),{tool:'OTHER'});
assert(scoreRegionSummary(other.concat(points).reverse(),region,p=>p.tool).points.length===14,'group isolation sorted');
''')

    def test_summary_stops_at_nonnormal_missing_and_gap(self):
        self.check_js('''
var points=series(normal.concat(Array(5).fill(1.5))),region=detect(points).regions[0];
[1,1.3,null,NaN].forEach(function(v){var copy=points.map(p=>Object.assign({},p));copy[17].ratio=v;assert(scoreRegionSummary(copy,region,p=>p.tool).normal.length===2,'stop at invalid or abnormal');});
var copy=points.map(p=>Object.assign({},p));copy.slice(20).forEach(p=>p.ts=new Date(Date.parse(p.ts)+7200000).toISOString());
var shifted=Object.assign({},region,{start:copy[20].ts,end:copy[24].ts});
assert(scoreRegionSummary(copy,shifted,p=>p.tool).normal.length===0,'do not cross long gap');
assert(scoreRegionSummary(points.slice(20),region,p=>p.tool).normal.length===0,'no preceding records');
var extended=detect(copy,{gapMs:1440*60000}).regions[0];
assert(extended.gapMs===1440*60000,'region retains configured gap');
assert(scoreRegionSummary(copy,extended,p=>p.tool).normal.length===8,'normal context follows configured gap');
var exact=Object.assign({},extended,{gapMs:Date.parse(copy[20].ts)-Date.parse(copy[19].ts)});
assert(scoreRegionSummary(copy,exact,p=>p.tool).normal.length===8,'equal gap included');
exact.gapMs-=1;assert(scoreRegionSummary(copy,exact,p=>p.tool).normal.length===0,'over gap excluded');
copy[19].ratio=1;assert(scoreRegionSummary(copy,extended,p=>p.tool).normal.length===0,'extended gap never includes abnormal baseline');

''')


    def test_whole_process_duration_and_sensor_means(self):
        self.check_js('''
var row={valve_phases:{version:2,status:'ok',steps:{supply:[1]}},step_timing:{steps:{1:{seconds:4},3:{seconds:8}}},sensor_metrics:{mean:{t2:{1:10,3:20},p2:{1:2,3:4}}}};
var v=curationSupplyValues(row,'t2','p2');assert(v.seconds===12&&v.temperature===15&&v.pressure===3,'whole process includes non-supply steps, equal step means');
var stats=curationSupplyStats([v,{seconds:6},{seconds:null}],'seconds',1);assert(stats.normal.mean===12&&stats.selected.mean===6&&stats.selected.count===1,'exclude missing not zero');
row.step_timing.steps.zero_start={seconds:100};row.step_timing.steps.zero_end={seconds:200};assert(curationSupplyValues(row,'t2','p2').seconds===12,'boundary zeros excluded');
row.step_timing.steps[3].seconds=null;v=curationSupplyValues(row,'t2','p2');assert(v.seconds===null&&v.temperature===15,'missing duration independent of sensors');
row.sensor_metrics.mean.t2[3]=null;assert(curationSupplyValues(row,'t2','p2').temperature===null,'no partial step mean');
row.valve_phases.version=1;assert(curationSupplyValues(row,'t2','p2').steps.length===2,'independent of valve phases');
row.valve_phases={version:2,status:'unavailable',steps:{supply:[1]}};assert(curationSupplyValues(row,'t2','p2').seconds===null,'no guessed phase');
assert(curationSupplyValues(null,'t2','p2').seconds===null,'missing archive');row.valve_phases={version:2,status:'ok',steps:{supply:[1],process:[3]}};assert(curationSupplyValues(row,'t2','p2','supply').seconds===4,'selected supply only');assert(curationSupplyValues(row,'t2','p2','process').seconds===null,'missing selected phase timing');assert(curationSupplyValues(row,'t2','p2','discharge').steps.length===0,'no guessed phase');
''')


    def test_auto_sensor_up_down_stable_and_outlier(self):
        self.check_js('''
function row(a,b,c,d){return {process_steps:[1],valve_phases:{version:2,status:'ok',steps:{supply:[1]}},sensor_metrics:{mean:{t_real:{1:a},p_real:{1:b},Flow:{1:c},Outlier:{1:d}}}};}
var rows=Array.from({length:8},()=>row(100,50,10,10)).concat(Array.from({length:10},(_,i)=>row(110,40,10,i===0?1000:10)));
var found=detectCurationSensors(rows,8);assert(found.tested===4&&found.sensors.length===2,'only sustained changes');
assert(found.sensors.find(x=>x.sensor==='t_real').direction==='high','up');
assert(found.sensors.find(x=>x.sensor==='p_real').direction==='low','down');
assert(!found.sensors.some(x=>x.sensor==='Outlier'),'single outlier rejected');
''')

    def test_auto_sensor_missing_zero_and_variation(self):
        self.check_js('''
function row(v){return {process_steps:[1],valve_phases:{version:2,status:'ok',steps:{supply:[1]}},sensor_metrics:{mean:{sensor:{1:v}}}};}
var rows=Array.from({length:8},()=>row(0)).concat(Array.from({length:5},()=>row(2)));
assert(detectCurationSensors(rows,8).sensors.length===1,'zero baseline works');
assert(!detectCurationSensors(rows.slice(4),4).tested,'minimum normal sample count');
rows[0]={};rows[1]={};assert(!detectCurationSensors(rows,8).tested,'normal coverage');
rows=Array.from({length:8},(_,i)=>row(i%2?90:110)).concat(Array.from({length:5},()=>row(115)));
assert(!detectCurationSensors(rows,8).sensors.length,'within normal variation');
rows=Array.from({length:8},()=>row(100)).concat(Array.from({length:5},()=>row(100.1)));
assert(!detectCurationSensors(rows,8).sensors.length,'relative floor');
rows=Array.from({length:8},()=>row(100)).concat([row(120),row(120),row(120),row(null),row(null)]);
assert(!detectCurationSensors(rows,8).tested,'selected coverage');
''')


if __name__ == '__main__':
    unittest.main()
