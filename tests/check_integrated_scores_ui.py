"""Fixture-only browser test: --firefox /path/to/firefox (no production data)."""
import argparse
import json,subprocess,tempfile,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
parser=argparse.ArgumentParser()
parser.add_argument('--firefox',required=True)
args=parser.parse_args()
s=(Path(__file__).resolve().parents[1]/'src/app/static/index.html').read_text()
import re
scripts=re.findall(r'<script(?:\s[^>]*)?>([\s\S]*?)</script>',s)
js='var checks=0;function ok(v,m){checks++;if(!v)throw Error(m);}function esc(s){return String(s).replace(/</g,"&lt;");}function flr2(x){return Math.floor(x*100)/100;}'
js+='new Function('+json.dumps('\n'.join(scripts))+');'
js+=s[s.index("var INTEGRATED_SCORE_MODE="):s.index("var SC_SENSOR_VIEW_STATE=")]
js+=s[s.index("function auxiliarySensorItems("):s.index("var _scUID=")]
setup=r'''
(async function(){
 var infinite={available:true,ratio:null,ratio_infinite:true,threshold_zero_exceeded:true,step_sensor_details:[{step:1,sensor:'C1',label:'C1',ratio:null,ratio_infinite:true,ratios:{shape:null}}]};
 var infPoint=integratedPoint({ratio:999,score_components:{cylinder:infinite}},false);
 ok(infPoint.ratio===Infinity&&infPoint.verdict==='NG','zero threshold wins');
 ok(!infPoint._integrated.missing_models.cylinder,'infinite cylinder available');
 ok(integratedTooltip(infPoint).includes('임계값 0 초과'),'infinite tooltip');
 var infHost=document.createElement('div');document.body.appendChild(infHost);
 renderSensorBars(infHost,'',auxiliarySensorItems({cylinder:infinite}),{appScore:true});
 ok(infHost.textContent.includes('∞'),'infinite bar text');
 ok(infHost.querySelector('button').innerHTML.includes('width:100%'),'infinite bar capped');
 var savedInf=integratedPoint({ratio:2,score_components:{cylinder:infinite,integrated:{ratio:null,ratio_infinite:true,stored_result:true,source_model:'cylinder'}}},false);
 ok(savedInf.ratio===Infinity,'saved infinite restored');
 var p={ratio:.99,ratio_x:.99999,score_components:{step_time:{available:true,ratio:1.12},cylinder:{available:false,reason:'DSV2 모델 없음'}}};
 var merged=integratedPoint(p,false,{warn:1.05,crit:1.2});
 ok(merged.ratio===1.12&&merged.band==='warn','max and existing bands');
 ok(merged._integrated.source_model==='step_time','winner tracked');
 ok(p.ratio===.99,'TCN fields unmodified');
 ok(integratedPoint(Object.assign({},p,{ratio:1.25}),true).ratio===1.25,'selected TCN recombined');
 ok(integratedTooltip(merged).includes('DSV2 모델 없음'),'missing model tooltip');
 ok(auxiliaryMissingText({step_time:'long model error',cylinder:'DSV2 model absent'})==='미적용: 스텝시간, 실린더','short model status');
 ok(auxiliaryMissingDetail({step_time:'long model error'}).includes('long model error'),'full reason retained for tooltip');
 ok(auxiliaryMissingText({})==='','no missing status when all available');
 var historical={tcn:{available:true,ratio:.7,stored_result:true},step_time:{available:true,ratio:1.4,stored_result:true,current_model_status:'missing',current_model_reason:'검사 당시 저장 결과 · 현재 보조모델 없음'},cylinder:{available:true,ratio:2.3,stored_result:true,current_model_status:'different',current_model_reason:'검사 당시 저장 점수 · 현재 계산 계약과 다름'},integrated:{ratio:2.31,source_model:'cylinder',stored_result:true}};
 var historyPoint=integratedPoint({ratio:99,score_components:historical},true);
 ok(historyPoint.ratio===2.31,'saved integrated score preserved despite selected/current TCN');
 ok(historyPoint._integrated.ratios.tcn===.7,'stored TCN used');
 ok(integratedTooltip(historyPoint).includes('검사 당시')&&integratedTooltip(historyPoint).includes('계산 계약과 다름'),'historical basis and contract mismatch visible');
 var summaries=auxiliarySensorItems(historical);ok(summaries.length===2,'stored summaries visible without current models or detail arrays');
 ok(summaries[0]._tooltip.includes('보조모델 없음'),'model status on group tooltip');
 var summaryHost=document.createElement('div');renderAuxiliaryDetail(summaryHost,summaries[0]);ok(summaryHost.textContent.includes('저장된 상세 자료 없음'),'missing arrays explained without synthesis');
 var components={step_time:{available:true,ratio:1.12,step_details:[{step:3,label:'ProcessStep 3 소요시간',actual:10,median:5,q25:4,q75:6,robust_deviation:5,contribution:.2}]},cylinder:{available:true,ratio:1.23,step_sensor_details:[{step:3,sensor:'d_004_PV',label:'C1 / ProcessStep 3',ratio:3.14,ratios:{shape:3.14},raw:[2,3],golden:[1,1]}]}};
 var scaledFixture=JSON.parse(JSON.stringify(components));scaledFixture.cylinder.ratio=2.3;scaledFixture.cylinder.raw_ratio=1.78;scaledFixture.cylinder.scaling_applied=true;
 var scaledItem=auxiliarySensorItems(scaledFixture).find(function(item){return item._kind==='cylinder_group';});
 ok(scaledItem.ratio===2.3,'scaled cylinder group score above one retained');
 ok(scaledItem._children[0].ratio===3.14,'cell diagnostic ratio remains unscaled');
 ok(scaledItem._tooltip.includes('정규화 점수: 2.3')&&scaledItem._tooltip.includes('threshold ratio: 1.78'),'scaled and raw scores both in tooltip');
 ok(integratedPoint({ratio:.5,score_components:scaledFixture},false).ratio===2.3,'integrated uses scaled wafer ratio');
 var items=auxiliarySensorItems(components);ok(items.length===2,'auxiliary items share one list');
 var g=document.createElement('div');document.body.appendChild(g);renderSensorBars(g,'/unused',items,{appScore:true});
 ok(g.querySelectorAll('button').length===2,'unified bars');
 ok(g.textContent.includes('1.23'),'group uses cylinder total ratio');ok(!g.textContent.includes('3.14'),'cell scores initially collapsed');
 var stepToggle=g.querySelector('.step-time-group-toggle');
 ok(!g.querySelector('.step-time-items'),'step list initially collapsed');
 ok(stepToggle.textContent.includes('1.12'),'step group shows total IF ratio');
 stepToggle.click();ok(stepToggle.getAttribute('aria-expanded')==='true','step group expands');
 g.querySelector('.step-time-items button').click();ok(g.querySelector('svg'),'step duration comparison plotted');
 var toggle=g.querySelector('.cylinder-group-toggle');toggle.click();
 ok(toggle.getAttribute('aria-expanded')==='true','group expands');ok(stepToggle.getAttribute('aria-expanded')==='false'&&!g.querySelector('.step-time-items'),'switching groups closes step group');
 ok(g.textContent.includes('3.14'),'above-two cell score retained');
 g.querySelector('.cylinder-items button').click();ok(g.querySelectorAll('polyline').length===2,'raw and fixed golden plotted');
 toggle.click();ok(!g.querySelector('.cylinder-items'),'group collapses');ok(toggle.getAttribute('aria-expanded')==='false','collapsed accessibility state');
 toggle.click();ok(g.querySelector('.cylinder-items button'),'group reopens');
 g.querySelectorAll('button')[0].click();ok(!g.querySelector('.cylinder-items')&&toggle.getAttribute('aria-expanded')==='false','switching to step time resets group');

 g.querySelector('.step-time-items button').click();
 ok(g.textContent.includes('PV 설명 없음'),'old archive explains missing PV');
 stepToggle.click();ok(!g.querySelector('.step-time-items')&&stepToggle.getAttribute('aria-expanded')==='false','step group collapses');
 stepToggle.click();ok(g.querySelector('.step-time-items button'),'step group reopens');
 var event={available:true,step:3,occurrence:1,transition_type:'time',transition_confidence:.8,target_sensor:'p_008_PV',target_value:5,target_direction:'up',normal_reach_seconds:2,actual_reach_seconds:4,reach_delay_seconds:2,predicted_reach_seconds:3.8,duration_seconds:5,explanation_confidence:'high',validation_mae_seconds:.2,target_grid_seconds:[0,1,2,3,4,5],actual_target_trajectory:[0,1,2,3,5,6],candidates:[{sensor:'A',contribution_seconds:1.5,response_status:'expected_response_missing',grid_seconds:[0,1,2,3],actual_trajectory:[0,0,0,0],normal_trajectory:[0,1,2,3],impact_windows:[{start_seconds:0,end_seconds:1,contribution_seconds:1},{start_seconds:1,end_seconds:2,contribution_seconds:-.3}],primary_impact_window:{start_seconds:0,end_seconds:1,contribution_seconds:1}},{sensor:'B',contribution_seconds:-.5,response_status:'accelerated_response',grid_seconds:[0,1,2,3],actual_trajectory:[1,2,3,4],normal_trajectory:[0,1,2,3],impact_windows:[{start_seconds:2,end_seconds:3,contribution_seconds:-.5}],primary_impact_window:{start_seconds:2,end_seconds:3,contribution_seconds:-.5}}]};
 items[0]._children[0]._pv={available:true,primary_step:3,events:[event,{available:false,step:3,occurrence:2,reason:'두 번째 occurrence 모델 없음'},{available:true,step:5,occurrence:1}]};
 renderSensorBars(g,'/unused',items,{appScore:true});g.querySelector('.step-time-group-toggle').click();g.querySelector('.step-time-items button').click();
 ok(g.querySelectorAll('.pv-events button').length===2,'only matching step occurrences');
 ok(g.querySelectorAll('.pv-plot').length===3,'target candidate and impact graphs');
 ok(g.querySelectorAll('.pv-shade').length===3,'shared impact shading');
 ok(g.textContent.includes('정상 시 기대되는 반응이 나타나지 않음'),'unresponsive candidate retained');
 ok(g.textContent.includes('압력 원인 해석 신뢰도가 낮습니다'),'time transition caution');
 ok(g.querySelector('.pv-impact-bar[data-value="1"]').getAttribute('fill')==='#dc5656','delay positive red');
 ok(g.querySelector('.pv-impact-bar[data-value="-0.3"]').getAttribute('fill')==='#3478ce','advance negative blue');
 g.querySelectorAll('.pv-candidates button')[1].click();ok(g.querySelector('.pv-sensor').textContent.includes('B'),'candidate switches plot');
 g.querySelector('.pv-from').value=1;g.querySelector('.pv-to').value=4;g.querySelector('.pv-apply').click();
 ok(Array.from(g.querySelectorAll('.pv-plot')).every(function(x){return x.dataset.xFrom==='1'&&x.dataset.xTo==='4';}),'all elapsed axes synchronized');
 g.querySelectorAll('.pv-events button')[1].click();ok(g.textContent.includes('두 번째 occurrence 모델 없음')&&!g.querySelector('.pv-plot'),'missing occurrence never substitutes another event');

 for(var group of items){
   var fixture=Object.assign({},group),template=group._children[0];
   fixture._children=[.4,3.2,1.1,2.5,.7,4.1,1.8,2.9].map(function(score,i){return Object.assign({},template,{index:'test:'+i,sensor:'항목 '+i,score:score,ratio:score});});
   var before=fixture._children.map(function(it){return it.index;}).join(',');
   var h=document.createElement('div');document.body.appendChild(h);renderAuxiliaryDetail(h,fixture);
   ok(h.querySelectorAll('.aux-top-five button').length===5,'five entries per group');
   ok(h.querySelector('.aux-top-five button').textContent.includes('항목 5'),'highest actual score first, not capped score');
   ok(h.querySelectorAll('.aux-remaining > div:first-child > button').length===0,'remaining rows lazy and collapsed');
   ok(fixture._children.map(function(it){return it.index;}).join(',')===before,'source ordering unchanged');
   var more=h.querySelector('.aux-more');more.open=true;more.dispatchEvent(new Event('toggle'));
   ok(h.querySelectorAll('.aux-remaining > div:first-child > button').length===3,'all remaining entries visible on expand');
   h.querySelector('.aux-remaining button').click();ok(h.querySelector('.aux-remaining svg'),'remaining entry detail still works');
   more.open=false;more.dispatchEvent(new Event('toggle'));more.open=true;more.dispatchEvent(new Event('toggle'));
   ok(h.querySelectorAll('.aux-remaining > div:first-child > button').length===3,'reopening does not duplicate rows');
   fixture._children=fixture._children.slice(0,5);renderAuxiliaryDetail(h,fixture);ok(!h.querySelector('.aux-more'),'no more control for five entries');
   fixture._children=[];renderAuxiliaryDetail(h,fixture);ok(h.textContent.includes('저장된 상세 자료 없음'),'empty group handled');
 }
 await fetch('/result',{method:'POST',body:JSON.stringify({ok:true,checks:checks})});
})().catch(e=>fetch('/result',{method:'POST',body:JSON.stringify({error:e.message,stack:e.stack})}));
'''
html='<!doctype html><meta charset="utf-8"><div id="drillTrendChart"></div><script>'+js+setup+'</script>'
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
