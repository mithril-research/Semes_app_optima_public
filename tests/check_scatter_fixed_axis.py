"""Browser regression for pinned scatter axes, redraws and range coordinates."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument('--firefox', required=True)
args = parser.parse_args()
source = (Path(__file__).resolve().parents[1] / 'src/app/static/index.html').read_text()
js = source[source.index("var INTEGRATED_SCORE_MODE="):source.index('var SC_SENSOR_VIEW_STATE=')]
js += source[source.index('function scatterFixedYAxis('):source.index('function recipeStepEnds(')]
js += source[source.index('function mountScatterDrag('):source.index('function closeDrill(')]
setup = r'''
var checks=0,opened=[],INTEGRATED_SCORE_MODE='original',SC_RULE=null,SC_TRC='__all__',SC_RECIPES=[];
var SC_SCALE_KEYS=['10m','30m','60m','6h','12h','day'],SC_SENSOR_VIEW_STATE={};
function ok(v,m){checks++;if(!v)throw Error(m);}
function esc(v){return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');}
function scMountSteps(){return false;}function clearTrendAnalysis(){}function trendAnalysisLoaded(){return {};}
function syncTrendAnalysis(){}function setBands(){}function refreshAfterRuleChange(){}
function scScaleLoad(){return '10m';}function scScaleSave(){}function scRecipeShapeIndex(){return 0;}
function dispFirstOn(){return true;}function integratedTooltip(){return '';}
function scatterScoreText(p){return Number(p.ratio).toFixed(2);}
function chamberColor(){return '#2563eb';}function mountTrendToggle(){}
function scSensorProcessOrder(names){return names.sort();}
function scSensorGroup(name){return name[0]==='T'?'temperature':'pressure';}
function scSensorColors(){return {T_900:'#ea580c',P_900:'#2563eb'};}
function scSensorScoreKey(){return '';}function scSensorScoreGet(){return null;}function scSensorScorePut(){}
function sensorChamberLegend(){return '';}
function sensorChamberMarker(ch,x,y,r,attrs,title){return '<circle cx="'+x+'" cy="'+y+'" r="'+r+'" '+attrs+'><title>'+esc(title)+'</title></circle>';}
function openWaferInDetail(wid){opened.push(wid);}function curationRangeGroups(){return [];}
function scoreRegionTime(value){return value;}function sensorTrendEquipment(){return 'fixture';}
var nativeFetch=window.fetch;
window.fetch=async function(url,opts){
 if(url==='/api/sensors')return {ok:true,json:async()=>({recon_sensors:['T_900','P_900']})};
 if(url==='/api/scatter/sensors-max')return {ok:true,json:async()=>({wafers:JSON.parse(opts.body).wafers.map(w=>({wid:w.wid,scores:{T_900:5.2,P_900:6.8}}))})};
 return nativeFetch(url,opts);
};
function pause(){return new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));}
async function waitFor(fn){for(var i=0;i<120;i++){if(fn())return;await new Promise(r=>setTimeout(r,25));}throw Error('timeout '+fn);}
function state(box){return Array.from(box.querySelectorAll('.sc-scroll')).map(scroll=>{
 var axis=scroll.querySelector('.sc-fixed-y-axis'),rect=axis.getBoundingClientRect();
 return {scroll:scroll,axis:axis,left:rect.left,top:rect.top,ticks:Array.from(axis.querySelectorAll('text')).map(t=>t.textContent).join('|')};
});}
(async function(){
 var box=document.getElementById('drillTrendChart'),data={model_id:'fixture',rule:{mode:'raw',bands:{warn:2,crit:3}},_sensorSelection:{temperature:['T_900'],pressure:['P_900']},points:[]};
 for(var i=0;i<60;i++)data.points.push({wafer_id:'wafer'+i,ts:new Date(Date.UTC(2026,0,1,i)).toISOString(),ratio:.5+(i%4)*.4,score:1,band:'ok',chamber:'dsv1',recipe:'A'});
 box._fitWidth=false;
 drawScatter(box,data,{scaleMin:'10m'});
 await waitFor(()=>box.querySelectorAll('.sc-fixed-y-axis').length===3);
 var before=state(box),main=before[0].scroll,svg=main.querySelector('svg');
 ok(before.length===3,'overall, temperature and pressure scatter axes rendered');
 ok(Array.from(box.querySelectorAll('.sensor-score-point')).every(p=>Number(p.getAttribute('cy'))===26),'sensor scores above three sit at top Y position');
 ok(before.every(s=>s.scroll.scrollWidth>s.scroll.clientWidth+500),'fixture overflows horizontally');
 var dot=svg.querySelector('g[onclick] path'),dotBefore=dot.getBoundingClientRect().left;
 main.scrollLeft=500;await pause();
 var after=state(box);
 after.forEach((s,i)=>{ok(Math.abs(s.left-before[i].left)<1,'Y axis stays visible while scrolling');ok(Math.abs(s.top-before[i].top)<1,'tick vertical positions unchanged');ok(s.ticks===before[i].ticks,'tick values unchanged');ok(Math.abs(s.scroll.scrollLeft-500)<1,'scatter scrolling remains synchronized');});
 ok(Math.abs(dot.getBoundingClientRect().left-dotBefore+500)<1,'wafer points continue moving horizontally');
 dot.parentNode.dispatchEvent(new MouseEvent('click',{bubbles:true}));ok(opened[0]==='wafer0','point still opens wafer detail');
 var hit=svg.querySelector('rect[fill-opacity=".06"]');
 function clickRange(x){var p=svg.createSVGPoint();p.x=x;p.y=svg.viewBox.baseVal.height-15;p=p.matrixTransform(svg.getScreenCTM());hit.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:p.x,clientY:p.y}));}
 clickRange(650);clickRange(850);
 var saved=box._modelPanels.integrated.chart._curationRangeSelection.ranges[0],W=svg.viewBox.baseVal.width;
 ok(Math.abs(saved[0]-(650-50)/(W-70))<.00001&&Math.abs(saved[1]-(850-50)/(W-70))<.00001,'range selection retains original SVG coordinates after scrolling');
 ok(box.querySelector('.sc-ymax').disabled,'score Y maximum is fixed');
 box._ymax=7;drawScatter(box,data,box._opts);
 await waitFor(()=>box.querySelectorAll('.sc-fixed-y-axis').length===3);await pause();
 ok(state(box)[0].ticks.split('|').includes('3'),'fixed axis ignores previous Y maximum');
 ok(box.querySelector('.sc-scroll').scrollLeft>400,'redraw restores horizontal position');
 var normal=state(box);normal[0].scroll.scrollLeft=900;await pause();state(box).forEach((s,i)=>ok(Math.abs(s.left-normal[i].left)<1,'redrawn axes stay pinned'));
 var sensorY=box.querySelector('.sensor-ymax');ok(sensorY.disabled&&sensorY.value==='3','sensor Y maximum fixed at three');await pause();
 ok(state(box)[1].ticks.split('|').includes('3'),'sensor axis maximum is three');
 box.querySelector('.sc-size').click();await waitFor(()=>box.querySelectorAll('.sc-fixed-y-axis').length===3);await pause();
 ok(state(box).every(s=>s.scroll.scrollWidth<=s.scroll.clientWidth+2),'compact scatter keeps responsive width');
 ok(state(box)[0].ticks.split('|').includes('3'),'compact mode retains fixed Y maximum');


 var integrated=box._modelPanels.integrated.chart;integrated._scatterExpanded=true;
 data.points.forEach(function(p){p.score_components={step_time:{available:true,ratio:1.4},cylinder:{available:true,ratio:5.6}};});
 drawScatter(box,data,box._opts);await pause();
 function panel(mode){return box.querySelector('.scatter-model-panel[data-score-mode="'+mode+'"]');}
 function chart(mode){return box._modelPanels[mode].chart;}
 function shownScore(mode){return chart(mode).querySelector('.sc-scroll g[onclick] title').textContent;}
 ok(box.querySelectorAll('.scatter-model-panel').length===4,'four scatter choices');
 ok(chart('integrated').querySelector('.sc-fit'),'OPTIMA fit-width control retained');
 ok(shownScore('integrated').includes('이상점수 5.60'),'integrated displays maximum model score');
 panel('step_time').querySelector('summary').click();await pause();
 ok(panel('integrated').open&&panel('step_time').open,'multiple panels remain expanded together');
 ok(shownScore('step_time').includes('이상점수 1.40'),'step-time displays its own score');
 ok(chart('step_time').querySelector('.sc-thr').disabled,'component threshold is fixed');
 var scroll=chart('step_time').querySelector('.sc-scroll'),axis=scroll.querySelector('.sc-fixed-y-axis'),left=axis.getBoundingClientRect().left;
 scroll.scrollLeft=600;await pause();ok(Math.abs(axis.getBoundingClientRect().left-left)<1,'component axis pinned');
 panel('cylinder').querySelector('summary').click();panel('tcn').querySelector('summary').click();await pause();
 ok(Array.from(box.querySelectorAll('.scatter-model-panel')).every(p=>p.open),'all four panels expand at once');
 ok(chart('integrated').querySelectorAll('.scatter-range-analyze').length===1,'integrated scatter keeps range analysis');
 ok(['tcn','step_time','cylinder'].every(mode=>!chart(mode).querySelector('.scatter-range-analyze')),'model-only scatters omit range analysis');
 var scale=chart('tcn').querySelector('.sc-scale');scale.value='60m';scale.dispatchEvent(new Event('change'));await pause();
 ok(['integrated','tcn','step_time','cylinder'].every(mode=>chart(mode).querySelector('.sc-scale').value==='60m'),'axis unit change updates all four plots');
 ok(chart('integrated')._curationRangeSelection.ranges.length===1,'shared scale retains integrated selection');
 var sourceScroll=chart('step_time').querySelector('.sc-scroll');sourceScroll.scrollLeft=(sourceScroll.scrollWidth-sourceScroll.clientWidth)/3;await pause();
 function ratio(mode){var el=chart(mode).querySelector('.sc-scroll');return el.scrollLeft/(el.scrollWidth-el.clientWidth);}
 ok(['integrated','tcn','step_time','cylinder'].every(mode=>Math.abs(ratio(mode)-1/3)<.002),'scrolling one model synchronizes all four');
 await new Promise(r=>setTimeout(r,100));ok(Math.abs(ratio('step_time')-1/3)<.002,'synchronized scroll has no feedback loop');
 var sensorScroll=chart('integrated').querySelectorAll('.sc-scroll')[1];sensorScroll.scrollLeft=(sensorScroll.scrollWidth-sensorScroll.clientWidth)*.6;await pause();
 ok(['integrated','tcn','step_time','cylinder'].every(mode=>Math.abs(ratio(mode)-.6)<.002),'sensor scroll also synchronizes model views');
 data.points[0].score_components.step_time={available:false,reason:'missing endpoint'};data.points[data.points.length-1].score_components.step_time={available:false,reason:'missing endpoint'};
 drawScatter(box,data,box._opts);await pause();
 ok(chart('step_time').querySelector('.sc-scroll svg').viewBox.baseVal.width===chart('integrated').querySelector('.sc-scroll svg').viewBox.baseVal.width,'missing model endpoints preserve shared time domain');

 ok(shownScore('cylinder').includes('이상점수 5.60'),'cylinder tooltip retains actual score above cap');
 ok(shownScore('tcn').includes('이상점수 0.50'),'TCN retains original score');
 ok(!chart('tcn').querySelector('.sc-thr').disabled,'TCN controls independent of integrated threshold');
 var capped=chart('cylinder').querySelector('.sc-scroll g[onclick] path'),bounds=capped.getBBox();
 ok(Math.abs(bounds.y+bounds.height/2-26)<.01,'score above three sits at top Y position');
 ok(Array.from(box.querySelectorAll('.sc-ymax')).every(e=>e.value==='3'),'every model axis remains capped at three');
 var titles=Array.from(box.querySelectorAll('.sc-fixed-y-axis text[transform]'));
 ok(titles.every(e=>Number(e.getAttribute('x'))===10),'all score axis titles shifted left');
 chart('cylinder').querySelector('.sc-scroll g[onclick]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 ok(INTEGRATED_SCORE_MODE==='original','component point uses component detail basis');
 chart('integrated').querySelector('.sc-scroll g[onclick]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 ok(INTEGRATED_SCORE_MODE==='integrated','integrated point restores integrated detail basis');
 var siblingSvg=chart('step_time').querySelector('svg');
 chart('cylinder').querySelector('.sc-size').click();await pause();
 ok(chart('step_time').querySelector('svg')===siblingSvg,'one model resize does not rebuild siblings');
 chart('cylinder').querySelector('.sc-size').click();await pause();
 panel('cylinder').querySelector('summary').click();await pause();
 ok(!panel('cylinder').open&&panel('step_time').open&&panel('tcn').open,'closing one leaves others open');
 var sensorScale=chart('integrated').querySelector('.sensor-scale');sensorScale.value='30m';sensorScale.dispatchEvent(new Event('change'));await pause();
 ok(chart('cylinder')._opts.scaleMin==='30m','closed panel receives latest shared scale');

 drawScatter(box,data,box._opts);await pause();
 ok(!panel('cylinder').open&&panel('step_time').open&&panel('tcn').open,'open states survive parent refresh');
 panel('cylinder').querySelector('summary').click();await pause();ok(panel('cylinder').open,'collapsed model reopens independently');
 ok(chart('cylinder').querySelector('.sc-scale').value==='30m','reopened panel renders with shared scale');
 ok(Math.abs(ratio('cylinder')-ratio('step_time'))<.002,'reopened panel joins current scroll position');
 data.points.forEach(function(p){p.score_components.cylinder={available:false,reason:'DSV 모델 없음'};});
 drawScatter(box,data,box._opts);await pause();
 ok(panel('cylinder').textContent.includes('DSV 모델 없음'),'missing model reason visible');
 ok(!chart('cylinder').querySelector('g[onclick]'),'missing scores never become zero dots');
 ok(shownScore('tcn').includes('이상점수 0.50')&&panel('tcn').open,'other models stay visible when one is missing');
 await nativeFetch('/result',{method:'POST',body:JSON.stringify({ok:true,checks:checks})});
})().catch(e=>nativeFetch('/result',{method:'POST',body:JSON.stringify({error:e.message,stack:e.stack})}));
'''
html = '<!doctype html><meta charset="utf-8"><div id="drillTrendChart" style="width:740px"></div><script>' + js + setup + '</script>'
result = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        result.update(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
        self.send_response(200)
        self.end_headers()


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
with tempfile.TemporaryDirectory(prefix='scatter-axis-browser-') as profile:
    with open(Path(profile) / 'browser.log', 'w') as log:
        proc = subprocess.Popen([args.firefox, '--headless', '--no-remote', '--profile', profile,
                                 'http://127.0.0.1:' + str(server.server_port)], stdout=log, stderr=log)
        try:
            for _ in range(120):
                if result:
                    break
                time.sleep(.25)
        finally:
            proc.terminate()
            proc.wait(timeout=10)
print(json.dumps(result, ensure_ascii=False))
server.shutdown()
if not result.get('ok'):
    raise SystemExit(1)
