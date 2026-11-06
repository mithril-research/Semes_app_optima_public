"""Fixture-only browser coverage of measured full analysis and REF controls."""
import argparse
import json
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--firefox', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = (root/'src/app/static/index.html').read_text()
module = (root/'src/app/static/full_analysis.js').read_text()
helpers = source[source.index('var INTEGRATED_SCORE_MODE='):source.index('var SC_SENSOR_VIEW_STATE=')]
helpers += source[source.index('async function fetchTrendBatch('):source.index('// Match the existing server filename convention:')]
helpers += source[source.index('function sensorTrendEquipment('):source.index('function equipmentTrendColor(')]
helpers += source[source.index('var TREND_ROUTE_ORDER='):source.index('function syncTrendScroll(')]
setup = r'''
var SC_RULE=null,checks=0,opened=[],requests=[],metadataCalls=0,fixtureNow=Date.now(),waferHistory=[],rows={},badModel=false;
function ok(v,m){checks++;if(!v)throw Error(m);}
function chamberColor(c){return c==='dsv1'?'#2563eb':'#059669';}
function switchView(){}function openWaferInDetail(id,date){opened.push({id:decodeURIComponent(id),date:date});}
var nativeFetch=window.fetch;
['EQ_A','EQ_B'].forEach(function(eq,e){['dsv1','dsv2'].forEach(function(ch,c){for(var i=0;i<12;i++){
 var p={tool:eq,chamber:ch,recipe:'A',wafer_id:eq+'_'+ch+'_'+i,ts:new Date(fixtureNow-1800000+i*60000).toISOString(),ratio:i===11?2:.2};waferHistory.push(p);
 var means={},mins={},stds={},peaks={};['T_950','P_950','S_950'].forEach(function(name,j){means[name]={};mins[name]={};stds[name]={};peaks[name]={};for(var step=1;step<=5;step++){var v=10+e*20+c*5+step+i*.01+j+(i===11?100:0);means[name][step]=v;mins[name][step]=v-2;peaks[name][step]=v+2;stds[name][step]=1.25;}});
 rows[p.wafer_id]={wid:p.wafer_id,date:p.ts.slice(0,10),sensor_metrics:{mean:means,min:mins,std:stds},peaks:peaks,step_timing:{steps:{1:{seconds:5},2:{seconds:7},3:{seconds:9},4:{seconds:11},5:{seconds:13}}},valve_phases:{version:2,status:i===10?'missing':'ok',steps:{supply:[0,1],wait:[2],process:[3,4],discharge:[5]}}};
}});});
waferHistory.sort((a,b)=>Date.parse(a.ts)-Date.parse(b.ts));
window.fetch=async function(url,opts){
 if(url.startsWith('/api/scatter?')){metadataCalls++;return {ok:true,json:async()=>({model_id:'fixture',points:waferHistory,rule:{bands:{warn:2,crit:3}}})};}
 if(url==='/api/sensors')return {ok:true,json:async()=>({recon_sensors:['T_950','P_950'],input_sensors:['T_950','P_950','S_950','V_950'],io:['V_950']})};
 if(url==='/api/scatter/step-times'){var req=JSON.parse(opts.body);requests.push(req);return {ok:true,json:async()=>({model_id:badModel?'wrong':req.model_id,wafers:req.wafers.map(p=>rows[p.wid])})};}
 return nativeFetch(url,opts);
};
function pause(){return new Promise(r=>setTimeout(r,80));}
async function waitFor(fn){for(var i=0;i<200;i++){if(fn())return;await new Promise(r=>setTimeout(r,25));}throw Error('timeout '+fn);}
function local(t){var d=new Date(t),p=n=>String(n).padStart(2,'0');return d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate())+'T'+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());}
document.getElementById('drillTrendChart')._opts={axisStart:new Date(fixtureNow-3600000).toISOString(),axisEnd:new Date(fixtureNow).toISOString()};
'''
checks = r'''
(async function(){
 await FullAnalysis.init();var app=document.getElementById('fullAnalysisApp');
 function sensorRow(){return app.querySelector('.fa-row[data-sensor="T_950"]');}
 ok(FullAnalysis.state.loaded&&FullAnalysis.state.display.current.length===24,'full analysis loads current equipment measured cohort');
 ok(requests.every(r=>r.wafers.length<=64&&r.model_id==='fixture'),'bounded model-checked wafer batches');
 ok(!Array.from(app.querySelectorAll('.fa-sensors input')).some(c=>c.value==='V_950'),'valve channels excluded from sensor choices');
 app.querySelectorAll('.fa-sensors button')[1].click();app.querySelector('.fa-sensors input[value="T_950"]').click();app.querySelector('.fa-sensors input[value="__duration__"]').click();
 await waitFor(()=>app.querySelectorAll('.fa-row').length===2);sensorRow().scrollIntoView();await waitFor(()=>sensorRow().querySelector('.fa-svg'));
 ok(app.querySelectorAll('.fa-row').length===2,'sensor and duration produce separate rows');
 ok(app.querySelectorAll('.fa-row')[0].querySelectorAll('.fa-stage').length===4,'four actual valve phases');
 ok(!app.querySelector('.fa-point[data-step="0"]'),'numeric zero excluded');
 ok(app.querySelector('.fa-ng-ring')&&app.querySelector('.fa-outlier'),'composite ring and measured step outlier displayed separately');
 ok(app.querySelector('.fa-status').textContent.includes('미확인 2매'),'missing valve phases reported');
 var reference=waferHistory.filter(p=>p.tool==='EQ_B'&&p.chamber==='dsv1').slice(0,6);
 FullAnalysis.saveRef({from:reference[0].ts,to:reference[5].ts,points:reference},'fixture');
 await waitFor(()=>FullAnalysis.state.display.reference.length===6);sensorRow().scrollIntoView();await waitFor(()=>app.querySelector('.fa-point[data-reference="true"]'));
 ok(JSON.parse(sessionStorage.getItem('semes-full-analysis-ref-v1')).groups[0].equipment==='EQ_B','single REF saved to session storage');
 ok(app.querySelector('.fa-ref-label').textContent.includes('EQ_B'),'REF provenance visible');
 ok(app.querySelector('.fa-row-head').textContent.includes('REF 챔버 전체와 비교'),'missing chamber REF fallback explicitly labelled');
 var firstRow=sensorRow(),phase=firstRow.querySelector('.fa-stage[data-phase="supply"]');
 ok(Number(phase.dataset.yMin)<11&&Number(phase.dataset.yMax)>31,'phase Y domain includes selected and REF on one axis');
 ['max','min','std','mean'].forEach(function(stat){var before=requests.length;app.querySelector('.fa-stat[data-stat="'+stat+'"]').click();ok(requests.length===before,'stat switch reuses measurements');});
 app.querySelector('.fa-stat[data-stat="std"]').click();sensorRow().scrollIntoView();await waitFor(()=>sensorRow().querySelector('.fa-point[data-step="1"]'));
 ok(Number(sensorRow().querySelector('.fa-point[data-step="1"]').dataset.value)===1.25,'population std from measured summary');
 app.querySelector('.fa-stat[data-stat="mean"]').click();sensorRow().scrollIntoView();await waitFor(()=>sensorRow().querySelector('.fa-point[data-step="1"]'));
 var phaseCheck=app.querySelector('.fa-phases input[value="wait"]');phaseCheck.click();sensorRow().scrollIntoView();await waitFor(()=>sensorRow().querySelector('.fa-svg'));
 ok(!app.querySelector('.fa-stage[data-phase="wait"]'),'phase toggle removes just selected phase');phaseCheck.click();
 sensorRow().scrollIntoView();await waitFor(()=>app.querySelectorAll('.fa-svg').length===2);await pause();
 var scrollers=app.querySelectorAll('.fa-chart-scroll');scrollers[0].scrollLeft=300;await pause();
 ok(Math.abs(scrollers[0].scrollLeft/(scrollers[0].scrollWidth-scrollers[0].clientWidth)-scrollers[1].scrollLeft/(scrollers[1].scrollWidth-scrollers[1].clientWidth))<.01,'row scrolling synchronized');
 var shown=app.querySelector('.fa-point').closest('[data-fa-wafer]');shown.dispatchEvent(new MouseEvent('click',{bubbles:true}));ok(opened.length===1&&opened[0].id.indexOf('EQ_')===0,'measured point opens wafer detail');
 var search=app.querySelector('.fa-sensor-search');search.value='T_950';search.dispatchEvent(new Event('input'));ok(app.querySelector('.fa-sensors input[value="P_950"]').parentNode.hidden,'sensor search filters choices');search.value='';search.dispatchEvent(new Event('input'));
 var reqBefore=requests.length;app.querySelector('.fa-chambers input[value="dsv2"]').click();await waitFor(()=>FullAnalysis.state.display.current.length===12);ok(requests.length===reqBefore,'chamber filter reuses archive cache');
 app.querySelector('.fa-equipments input[value="EQ_B"]').click();await waitFor(()=>app.querySelectorAll('.fa-row').length===4);ok(FullAnalysis.state.display.current.length===36,'comparison equipment includes its full chamber population');
 ok(app.querySelectorAll('.fa-equipment-group').length===2,'one chart group per selected equipment');
 var groups=Array.from(app.querySelectorAll('.fa-equipment-group'));ok(groups.map(g=>g.dataset.equipment).join()==='EQ_A,EQ_B'&&groups.every(g=>g.querySelectorAll('.fa-row').length===2),'each equipment group contains selected sensors');
 var eqB=groups[1];eqB.scrollIntoView();await waitFor(()=>eqB.querySelector('.fa-svg'));
 var segments=Array.from(eqB.querySelectorAll('.fa-row[data-sensor="T_950"] .fa-stage[data-phase="supply"] .fa-chamber-segment[data-reference="false"]'));
 ok(segments.map(g=>g.dataset.chamber).join()==='dsv1,dsv2'&&Number(segments[0].querySelector('rect').getAttribute('x'))<Number(segments[1].querySelector('rect').getAttribute('x')),'comparison equipment chambers are adjacent left to right within each phase');
 ok(segments.every(g=>Array.from(g.querySelectorAll('[data-fa-wafer] title')).every(t=>t.textContent.includes('EQ_B · '+g.dataset.chamber))),'each chamber contains only its own measured wafers');
 app.querySelector('.fa-equipment-chambers[data-equipment="EQ_B"] input[value="dsv1"]').click();await waitFor(()=>FullAnalysis.state.display.current.length===24);
 eqB=app.querySelector('.fa-equipment-group[data-equipment="EQ_B"]');eqB.scrollIntoView();await waitFor(()=>eqB.querySelector('.fa-svg'));
 ok(Array.from(eqB.querySelectorAll('.fa-chamber-segment[data-reference="false"]')).every(g=>g.dataset.chamber==='dsv2'),'comparison chamber selection filters chart segments');
 app.querySelectorAll('.fa-equipment-chambers[data-equipment="EQ_B"] button')[1].click();await waitFor(()=>app.querySelectorAll('.fa-equipment-group').length===1);ok(FullAnalysis.state.display.current.length===12,'clear all comparison chambers removes equipment cohort and group');
 app.querySelector('.fa-equipment-chambers[data-equipment="EQ_B"] input[value="dsv1"]').click();await waitFor(()=>app.querySelectorAll('.fa-equipment-group').length===2);ok(FullAnalysis.state.display.current.length===24,'comparison chambers can be reselected');
 app.querySelector('.fa-query').click();await waitFor(()=>!app.querySelector('.fa-query').disabled);ok(FullAnalysis.state.equipmentChambers.EQ_B.join()==='dsv1','refresh preserves per-equipment chamber selection');
 app.querySelector('.fa-ref-edit').click();var editor=app.querySelector('.fa-ref-editor');
 editor.querySelector('.fa-r-previous').click();ok(Date.parse(editor.querySelector('.fa-r-to').value)===Date.parse(app.querySelector('.fa-from').value),'previous same length shortcut');
 editor.querySelector('.fa-r-day').click();ok(Date.parse(editor.querySelector('.fa-r-from').value)===Date.parse(app.querySelector('.fa-from').value)-86400000,'day-before REF shortcut');
 editor.querySelector('.fa-r-week').click();ok(Date.parse(editor.querySelector('.fa-r-from').value)===Date.parse(app.querySelector('.fa-from').value)-604800000,'week-before REF shortcut');
 var select=editor.querySelector('.fa-r-equipment');select.value='EQ_A';select.dispatchEvent(new Event('change'));
 editor.querySelectorAll('.fa-ref-chambers button')[1].click();editor.querySelector('.fa-r-save').click();ok(editor.querySelector('.fa-r-status').textContent.includes('챔버'),'REF empty chamber selection rejected');
 editor.querySelector('.fa-ref-chambers input[value="dsv1"]').click();
 var own=waferHistory.filter(p=>p.tool==='EQ_A'&&p.chamber==='dsv1');editor.querySelector('.fa-r-from').value=local(Date.parse(own[0].ts)-1000);editor.querySelector('.fa-r-to').value=local(Date.parse(own[5].ts)+1000);editor.querySelector('.fa-r-save').click();
 await waitFor(()=>FullAnalysis.getRef().groups[0].equipment==='EQ_A'&&editor.hidden);
 ok(FullAnalysis.getRef().groups[0].chambers.join()==='dsv1','manual REF preserves explicit equipment and chamber');
 app.querySelector('.fa-use-ref').click();await waitFor(()=>!FullAnalysis.state.display.useRef);sensorRow().scrollIntoView();await waitFor(()=>sensorRow().querySelector('.fa-svg'));ok(!app.querySelector('.fa-point[data-reference="true"]'),'REF comparison toggle hides reference');app.querySelector('.fa-use-ref').click();await waitFor(()=>FullAnalysis.state.display.useRef);
 app.querySelector('.fa-ref-clear').click();await waitFor(()=>!FullAnalysis.getRef()&&FullAnalysis.state.display.reference.length===0);ok(!sessionStorage.getItem('semes-full-analysis-ref-v1'),'REF clear removes session record');
 sensorRow().scrollIntoView();await waitFor(()=>sensorRow().querySelector('.fa-svg'));sensorRow().querySelector('button').click();var zoom=document.querySelector('.fa-zoom');ok(zoom&&zoom.getAttribute('role')==='dialog'&&zoom.querySelector('svg'),'row zoom opens measured SVG');
 var downloads=[],anchorClick=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){if(this.download)downloads.push({name:this.download,href:this.href});else anchorClick.call(this);};
 zoom.querySelector('.fa-save-image').click();await waitFor(()=>downloads.length);ok(downloads[0].name.endsWith('.png')&&downloads[0].href.startsWith('data:image/png'),'row export creates actual PNG');HTMLAnchorElement.prototype.click=anchorClick;
 zoom.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));ok(!document.querySelector('.fa-zoom'),'zoom closes with escape');
 var beforeInvalid=metadataCalls;app.querySelector('.fa-from').value=local(fixtureNow+86400000);app.querySelector('.fa-query').click();await pause();ok(metadataCalls===beforeInvalid&&app.querySelector('.fa-status').textContent.includes('올바르게'),'invalid period rejected before API work');
 app.querySelector('.fa-from').value=local(fixtureNow-3600000);app.querySelector('.fa-to').value=local(fixtureNow);
 // A new uncached wafer plus a mismatched response must never be silently shown.
 var extra=Object.assign({},own[0],{wafer_id:'EQ_A_new',ts:new Date(fixtureNow-1000).toISOString()});waferHistory.push(extra);rows[extra.wafer_id]=Object.assign({},rows[own[0].wafer_id],{wid:extra.wafer_id,date:extra.ts.slice(0,10)});badModel=true;app.querySelector('.fa-query').click();await waitFor(()=>app.querySelector('.fa-status').textContent.includes('모델이 변경'));ok(!app.querySelector('.fa-query').disabled,'model mismatch unlocks query');
 await nativeFetch('/result',{method:'POST',body:JSON.stringify({ok:true,checks:checks,requests:requests.length})});
})().catch(e=>nativeFetch('/result',{method:'POST',body:JSON.stringify({error:e.message,stack:e.stack,status:document.querySelector('.fa-status')?.textContent})}));
'''
page = '<!doctype html><meta charset="utf-8"><style>:root{--panel:#fff;--line:#dbe2ec;--txt:#172033;--txt-dim:#64748b}body{font-family:sans-serif}button,input,select{font:inherit}.btn{padding:4px 8px}</style><div id="drillTrendChart"></div><div id="view-full-analysis" class="active"><div id="fullAnalysisApp"></div></div><script>'+helpers+setup+'</script><script src="/static/full_analysis.js"></script><script>'+checks+'</script>'
result = {}
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        payload = module if self.path == '/static/full_analysis.js' else page
        self.send_response(200); self.send_header('Content-Type', 'text/javascript; charset=utf-8' if self.path.endswith('.js') else 'text/html; charset=utf-8'); self.end_headers(); self.wfile.write(payload.encode())
    def do_POST(self):
        result.update(json.loads(self.rfile.read(int(self.headers['Content-Length'])))); self.send_response(200); self.end_headers()
server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
with tempfile.TemporaryDirectory(prefix='full-analysis-browser-') as profile:
    with open(Path(profile)/'browser.log','w') as log:
        proc = subprocess.Popen([args.firefox,'--headless','--no-remote','--profile',profile,'http://127.0.0.1:'+str(server.server_port)], stdout=log, stderr=log)
        try:
            for _ in range(240):
                if result: break
                time.sleep(.25)
        finally:
            proc.terminate(); proc.wait(timeout=10)
print(json.dumps(result,ensure_ascii=False)); server.shutdown()
if not result.get('ok'): raise SystemExit(1)
