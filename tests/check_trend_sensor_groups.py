"""Fixture-only browser test: --firefox /path/to/firefox (no production data)."""
import argparse
import json,subprocess,tempfile,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
parser=argparse.ArgumentParser()
parser.add_argument('--firefox',required=True)
args=parser.parse_args()
s=(Path(__file__).resolve().parents[1]/'src/app/static/index.html').read_text()
js=s[s.index('function recipeStepEnds('):s.index('function drawScatter(')]
js+=s[s.index('function scSensorGroup('):s.index('function scSensorColors(')]
js+=s[s.index('function mountTrendToggle('):s.index('function mountSensorScatter(')]
setup=r'''
var SC_SENSOR_VIEW_STATE={model:'fixture'};
function esc(v){return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');}
function scSensorProcessOrder(v){return v.sort();}function scScaleLoad(){return '6h';}function chamberColor(ch){return ['red','blue','green'][Number(ch.slice(-1))-1]||'gray';}
var pts=[],rows={},source={model_id:'fixture'},scatter=document.getElementById('drillTrendChart');scatter._data=source;scatter._opts={scaleMin:'6h'};
[1,2,3].forEach(function(ch){[0,1,2].forEach(function(i){var id='WUK370A_'+ch+i,key='2026-10-19/'+id;pts.push({wafer_id:id,ts:'2026-10-19T00:0'+i+':00',chamber:'dsv'+ch,recipe:'A'});var names=['p_test','t_test','s_010_PV','s_011_PV','d_001_PV','d_003_PV','s_001_Press','s_002_Press','s_003_Press','9000009'];var ss=names.map(function(n){var v=(ch===3?20:10)+i;return {sensor:n,step_stats:{1:{mean:v,min:v-1,max:v+1,std:1,count:3}}};});rows[key]={wid:id,date:'2026-10-19',sensor_series:ss,series:[],valve_phases:{version:2,status:'ok',steps:{process:[1]}},process_steps:[1],step_timing:{steps:{1:{seconds:5+i}}},peaks:{},sensor_metrics:{mean:{},min:{},std:{}}};ss.forEach(function(x){rows[key].peaks[x.sensor]={1:x.step_stats[1].max};['mean','min','std'].forEach(function(m){rows[key].sensor_metrics[m][x.sensor]={1:x.step_stats[1][m]};});});});});
Object.values(rows).forEach(function(row){row.step_timing.steps[2]={seconds:12};Object.keys(row.peaks).forEach(function(sensor){row.peaks[sensor][2]=row.peaks[sensor][1]+5;['mean','min','std'].forEach(function(m){row.sensor_metrics[m][sensor][2]=row.sensor_metrics[m][sensor][1]+2;});});});
Object.values(rows).forEach(function(row){row.zero_step_version=1;row.zero_steps=['zero_start','zero_end'];row.sensor_series.forEach(function(sensor){['zero_start','zero_end'].forEach(function(st,i){var v=sensor.step_stats[1].mean*(i?10:2);sensor.step_stats[st]={mean:v,min:v-1,max:v+1,std:1,count:2};row.peaks[sensor.sensor][st]=v+1;['mean','min','std'].forEach(function(m){row.sensor_metrics[m][sensor.sensor][st]=sensor.step_stats[st][m];});});});row.step_timing.steps.zero_start={seconds:3};row.step_timing.steps.zero_end={seconds:7};});
var nativeFetch=window.fetch,requests=0;
window.fetch=async function(url,opts){if(url.indexOf('/api/scatter/')!==0)return nativeFetch(url,opts);requests++;var body=JSON.parse(opts.body);return {ok:true,json:async function(){return {wafers:body.wafers.map(function(p){var row=JSON.parse(JSON.stringify(rows[p.date+'/'+p.wid]));(body.pairs||[]).forEach(function(pair){row.series.push({left:pair[0],right:pair[1],step_stats:{1:{mean:2,min:1,max:3,std:1,count:3},zero_start:{mean:5,min:4,max:6,std:1,count:2},zero_end:{mean:25,min:24,max:26,std:1,count:2}}});});return row;})};}};};
function ok(v,m){if(!v)throw Error(m);}
async function waitFor(fn){for(var i=0;i<100;i++){if(fn())return;await new Promise(r=>setTimeout(r,25));}throw Error('timeout: '+fn);}
async function click(selector){var el=document.querySelector(selector);ok(el,'missing '+selector);el.click();await new Promise(r=>setTimeout(r,60));}
async function exercise(group){
 ok(document.querySelectorAll('#trendAnalysisCharts > section').length===4,'four features');
 await click('.sensor-difference-pending .trend-load');await waitFor(()=>document.querySelector('.difference-left option'));
 var names=Array.from(document.querySelector('.difference-left').options).map(o=>o.value);ok(names.every(n=>trendSensorMatches(n,group)),'difference selector group');
 if(group==='all'){ok(!document.querySelector('.difference-wafer'),'no initial difference pairs');await click('.difference-pair-add');await waitFor(()=>document.querySelector('.difference-wafer'));}
 var differenceStep=document.querySelector('.difference-step');ok(differenceStep.options[0].value==='zero_start'&&differenceStep.options[differenceStep.options.length-1].value==='zero_end','difference order');differenceStep.value='zero_start';differenceStep.dispatchEvent(new Event('change'));
 if(group==='all')ok(document.querySelector('.difference-wafer'),'zero difference points');
 ok(document.querySelector('.sensor-difference-panel')||document.body.textContent.includes('스텝 0번'),'boundary label');
 await click('.equipment-value-pending .trend-load');await waitFor(()=>document.querySelector('.sensor-value-select input'));
 ok(Array.from(document.querySelectorAll('.sensor-value-select input')).every(c=>trendSensorMatches(c.value,group)),'equipment selector group');
 ok(!document.querySelector('.sensor-value-select input:checked'),'equipment sensors initially empty');await click('.value-sensors-all');ok(document.querySelector('.equipment-fixed-axis'),'equipment plot');
 var equipmentStep=document.querySelector('.equipment-value-panel .difference-step')||Array.from(document.querySelectorAll('.difference-step')).slice(-1)[0];equipmentStep.value='zero_end';equipmentStep.dispatchEvent(new Event('change'));ok(document.querySelector('.equipment-sensor-chart h4').textContent.includes('스텝 00번'),'equipment end-zero label');
 await click('.time-load');await waitFor(()=>document.querySelector('.shared-step-range .relation-step'));ok(!document.querySelector('.peak-sensor-select input:checked'),'recipe sensors initially empty');await click('.peak-sensors-all');
 ok(!document.querySelector('.recipe-chamber-bars'),'stacked bars removed');
 var step=document.querySelector('.relation-step');ok(step.options[0].value==='zero_start'&&step.options[step.options.length-1].value==='zero_end','recipe order');step.value='zero_end';step.dispatchEvent(new Event('change'));
 ok(document.querySelector('.step-peak-detail h4').textContent.includes('스텝 00'),'recipe zero label');
 ok(document.querySelector('.peak-duration-bar').dataset.seconds==='7','end zero timing');
 ok(document.querySelector('.relation-point').dataset.seconds==='7','relationship end zero timing');
 step.value='2';step.dispatchEvent(new Event('change'));
 ok(document.querySelector('.step-peak-detail h4').textContent.includes('스텝 2'),'step selection updates wafer trend');
 document.querySelector('.shared-step-from').value='1';document.querySelector('.shared-step-to').value='2';await click('.shared-step-apply');
 ok(step.value==='range:1:2'&&document.querySelector('.relation-point'),'range selection synchronized');
 ok(document.querySelector('.step-peak-detail h4').textContent.includes('1~2'),'range reflected in wafer trend');
 document.querySelector('.shared-step-from').value='0';document.querySelector('.shared-step-to').value='00';await click('.shared-step-apply');
 ok(step.value==='range:0:00'&&document.querySelector('.peak-duration-bar').dataset.seconds==='27','0 to 00 timing');
 ok(step.options[0].value==='zero_start'&&step.options[step.options.length-1].value==='zero_end','custom keeps boundary order');
 var custom=document.querySelector('.custom-step-from');custom.value='0';document.querySelector('.custom-step-to').value='00';await click('.custom-step-apply');
 ok(differenceStep.value==='range:0:00','difference boundary range accepted');
 ok(differenceStep.options[0].value==='zero_start'&&differenceStep.options[differenceStep.options.length-1].value==='zero_end','difference custom order');
 ok(Array.from(document.querySelector('.relation-sensor').options).every(o=>trendSensorMatches(o.value,group)),'recipe selector group');
 ok(document.querySelector('.relation-point'),'duration relationship plot');
 await click('.chamber-mean-load');await waitFor(()=>document.querySelector('.chamber-mean-sensors input'));ok(!document.querySelector('.chamber-mean-sensors input:checked'),'mean sensors initially empty');await click('.chamber-mean-sensors button');await click('.chamber-mean-run');await waitFor(()=>document.querySelector('.chamber-mean-results svg'));
 ok(document.querySelector('.chamber-mean-phases').textContent.includes('스텝 0번')&&document.querySelector('.chamber-mean-phases').textContent.includes('스텝 00번'),'chamber zero options');
 ok(Array.from(document.querySelectorAll('.chamber-mean-sensors label')).every(l=>trendSensorMatches(l.textContent,group)),'mean selector group');
}
(async function(){syncTrendAnalysis(scatter,pts,{H:240});ok(!document.querySelector('.trend-sensor-groups'),'single unified sensor view');ok(trendSensorArea('d_001_PV')===6&&trendSensorArea('d_003_PV')===6,'D sensors in other area');await exercise('all');var commonBars=Array.from(document.querySelectorAll('.trend-common-scale'));ok(commonBars.length===4,'each trend has a common scale');commonBars.forEach(bar=>{bar.querySelector('.common-ymin').value='-10';bar.querySelector('.common-ymax').value='100';bar.querySelector('.common-scale-apply').click();});ok(Array.from(document.querySelectorAll('.diff-ymin,.peak-ymin')).every(input=>input.value==='-10'),'common minimum applied to all trend sensor graphs');ok(Array.from(document.querySelectorAll('.diff-ymax,.peak-ymax')).every(input=>input.value==='100'),'common maximum applied to all trend sensor graphs');var bad=document.querySelector('.trend-common-scale');bad.querySelector('.common-ymin').value='200';bad.querySelector('.common-scale-apply').click();ok(bad.querySelector('.common-scale-error').textContent,'invalid common range rejected');bad.querySelector('.common-ymin').value='-10';ok(document.querySelector('.difference-left optgroup'),'difference sensors grouped');ok(document.querySelector('.sensor-value-select details.trend-sensor-area'),'equipment sensors grouped');ok(document.querySelector('.chamber-mean-sensors details.trend-sensor-area'),'mean sensors grouped');ok(document.querySelector('.relation-sensor optgroup'),'recipe sensors grouped');var area=document.querySelector('.sensor-value-select details.trend-sensor-area');ok(!area.open,'sensor area collapsed by default');area.querySelector('summary').click();await new Promise(r=>setTimeout(r,60));ok(area.open,'sensor area opens');ok(area.querySelector('summary').textContent.includes('선택'),'selection count displayed');ok(trendSensorArea('p_005_PV')===trendSensorArea('h_005_PV'),'related pressure and heater grouped');ok(trendSensorArea('t_001_PV')===trendSensorArea('h_001_PV'),'back path grouped');var root=document.getElementById('trendAnalysisCharts'),hs=document.createElement('div');root.appendChild(hs);var ss=[1000,1800].map(w=>{var el=document.createElement('div');el.style.cssText='width:200px;overflow:auto';el.innerHTML='<div style="width:'+w+'px;height:10px"></div>';hs.appendChild(el);syncTrendScroll(hs,el);return el;});ss[0].scrollLeft=400;ss[0].dispatchEvent(new Event('scroll'));ok(Math.abs(ss[1].scrollLeft-800)<2,'cross chart proportional scroll');await new Promise(r=>setTimeout(r,60));ok(Math.abs(ss[0].scrollLeft-400)<2,'no scroll feedback loop');hs.remove();var n=requests;scatter._restoreTrendPanels=trendAnalysisLoaded(scatter);syncTrendAnalysis(scatter,pts,{H:240});await waitFor(()=>document.querySelector('.equipment-fixed-axis'));ok(requests===n,'redraw reuses cache');ok(document.querySelector('.sensor-value-select details.trend-sensor-area').open,'area toggle survives redraw');await nativeFetch('/result',{method:'POST',body:JSON.stringify({ok:true,requests:requests})});})().catch(e=>nativeFetch('/result',{method:'POST',body:JSON.stringify({error:e.stack,text:document.body.textContent.slice(-4000)})}));
'''
html='<!doctype html><meta charset="utf-8"><div id="drillTrendChart"><div class="sc-scroll"></div></div><div id="trendAnalysisStatus"></div><div id="trendAnalysisCharts"></div><script>'+js+setup+'</script>'
result={}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers();self.wfile.write(html.encode())
 def do_POST(self):
  result.update(json.loads(self.rfile.read(int(self.headers['Content-Length']))));self.send_response(200);self.end_headers()
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
with tempfile.TemporaryDirectory(prefix='trend-browser-') as profile:
 with open(Path(profile)/'browser.log','w') as log:
  proc=subprocess.Popen([args.firefox,'--headless','--no-remote','--profile',profile,'http://127.0.0.1:'+str(server.server_port)],stdout=log,stderr=log)
  try:
   for i in range(120):
    if result:break
    time.sleep(.25)
  finally:proc.terminate();proc.wait(timeout=10)
print(json.dumps(result,ensure_ascii=False));server.shutdown()
if not result.get('ok'):raise SystemExit(1)
