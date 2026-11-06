"""Measured full-analysis calculations, REF scope and self-contained exports."""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from src.app.export_html import build_export_html


class FullAnalysisTests(unittest.TestCase):
    def test_offline_export_embeds_full_analysis_module(self):
        source = Path('src/app/static/index.html').read_text()
        html = build_export_html(source, [], {}, {})
        self.assertIn('id="view-full-analysis"', html)
        self.assertNotIn('src="/static/full_analysis.js"', html)
        self.assertIn('w.FullAnalysis=', html)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey')
    def test_offline_metadata_preserves_equipment_and_saved_composite(self):
        wafers = [dict(wafer_id='record', ts='2026-11-05T12:00:00', tool='EQ_EXPLICIT',
                       equipment='EQ_EXPLICIT', chamber='dsv1', recipe='A', ratio=.2,
                       score_components={'integrated': {'ratio': 2., 'stored_result': True}})]
        html = build_export_html('<script></script>', wafers, {}, {'model_id': 'fixture'})
        script = re.search(r'<script>([\s\S]*?)</script>', html).group(1)
        setup = "var window={},document={addEventListener:function(){}};const loop=new imports.gi.GLib.MainLoop(null,false);"
        check = """
window.fetch('/api/scatter?chamber=all').then(r=>r.json()).then(data=>{
 var p=data.points[0];if(p.tool!=='EQ_EXPLICIT'||p.score_components.integrated.ratio!==2)
  throw Error('offline analysis lost equipment scope or saved composite verdict');
 loop.quit();}).catch(e=>{printerr(e.stack);imports.system.exit(1);});loop.run();
"""
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'offline.js';path.write_text(setup+script+check)
            result=subprocess.run(['gjs',str(path)],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey')
    def test_measured_values_scope_sampling_and_normal_ranges(self):
        html = Path('src/app/static/index.html').read_text()
        helpers = html[html.index('var INTEGRATED_SCORE_MODE='):html.index('var SC_SENSOR_VIEW_STATE=')]
        module = Path('src/app/static/full_analysis.js').read_text()
        checks = r'''
var u=window.FullAnalysis.utils;
function ok(v,m){if(!v)throw Error(m);}
var row={valve_phases:{version:2,status:'ok',steps:{process:[0,1,1,-1,2.5]}},
 sensor_metrics:{mean:{T_950:{1:10}},min:{T_950:{1:-2}},std:{T_950:{1:0}}},
 peaks:{T_950:{1:14}},step_timing:{steps:{1:{seconds:0}}}};
ok(u.positiveSteps(row,'process').join()==='1','only actual positive integral steps');
ok(u.positiveSteps({valve_phases:{version:2,status:'missing'}},'process').length===0,'no invented phase mapping');
[['mean',10],['min',-2],['std',0],['max',14]].forEach(x=>ok(u.value(row,'T_950',1,x[0])===x[1],'exact '+x[0]));
ok(u.value(row,'__duration__',1,'max')===0,'zero-second duration remains valid');
ok(u.value(row,'T_950',2,'mean')===null&&u.value(row,'unknown',1,'mean')===null,'missing measurements remain missing');
ok(u.normalRange([1,2,3,4])===null,'minimum five normal wafers');
var nr=u.normalRange([1,2,3,4,5,null,NaN]);ok(nr.count===5&&nr.mean===3&&Math.abs(nr.sd-Math.sqrt(2))<1e-9,'normal ddof zero and finite-only');
var many=Array.from({length:220},(_,i)=>({tool:'EQ_A',chamber:'dsv1',recipe:'A',wafer_id:'w'+i,ts:new Date(1700000000000+i*1000).toISOString(),ratio:i>=200?2:.2}));
var sample=u.thin(many,160,p=>p.ratio>=1);ok(sample.points.length===160&&sample.points.filter(p=>p.ratio>=1).length===20,'sampling keeps every alert');
ok(u.thin(many,160,()=>true).points.length===220,'alerts never dropped to meet target');
var reference={from:many[0].ts,to:many[2].ts,groups:[{equipment:'EQ_A',chambers:['dsv1']}],recipes:['A']};
ok(u.refPoints(many,reference).length===3,'REF includes both endpoints');
ok(!u.refPoints(many.map(p=>Object.assign({},p,{recipe:'B'})),reference).length,'REF recipe scope');
var points=many.slice(0,6),cache={};points.forEach((p,i)=>{cache[JSON.stringify([p.ts.slice(0,10),p.wafer_id])]={valve_phases:{version:2,status:'ok',steps:{supply:[1],process:[1]}},sensor_metrics:{mean:{T_950:{1:i===5?100:10}}}};p.ratio=i===5?2:.2;});
var display={eq:'EQ_A',chambers:['dsv1','dsv2'],equipments:['EQ_B'],current:points,reference:points.slice(0,5),useRef:true,cache:cache};
var rows=u.rowsFor(display,['T_950']);ok(rows.length===2&&rows[0].segments.length===2&&rows[1].comparison,'row per equipment and sensor');
ok(rows[0].segments[1].refFallback,'missing REF chamber uses whole REF with explicit notice');
var prep=u.prepare(rows[0],display,'mean',['supply','process']);var range=prep.normals.get(prep.rangeKey(points[0],1));
ok(range.count===5,'overlapping REF and overlapping phases do not inflate normal population');
ok(prep.out(points[5],1,100)&&!prep.out(points[0],1,10),'measured normal range flags step deviation');
window.FullAnalysis.saveRef({from:points[0].ts,to:points[2].ts,points:points.slice(0,3)},'model');
ok(window.FullAnalysis.getRef().groups[0].equipment==='EQ_A','Overview selection saves REF scope');
window.FullAnalysis.saveRef({from:points[3].ts,to:points[4].ts,points:points.slice(3,5)},'model');
ok(window.FullAnalysis.getRef().from===points[3].ts,'single REF replaced');
ok(storage['semes-full-analysis-ref-v1'],'REF uses session storage');
window.FullAnalysis.clearRef();ok(!window.FullAnalysis.getRef()&&!storage['semes-full-analysis-ref-v1'],'REF clear');
'''
        setup = '''var window={},SC_RULE=null,storage={};var sessionStorage={getItem:k=>storage[k]||null,setItem:(k,v)=>{storage[k]=v;},removeItem:k=>{delete storage[k];}};function sensorTrendEquipment(p){return p.tool;}'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checks.js'
            path.write_text(setup+helpers+module+checks+'\nvoid 0;')
            result = subprocess.run(['gjs', str(path)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
