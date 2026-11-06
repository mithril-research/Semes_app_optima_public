/* Stage-by-step measured trends and session REF comparison. No simulated values. */
(function(w){
'use strict';
var PHASES=[['supply','가압','#2563eb'],['wait','대기','#8b5cf6'],['process','공정','#059669'],['discharge','배출','#d97706']];
var STATS=[['mean','평균'],['max','최대'],['min','최소'],['std','표준편차']],DURATION='__duration__',REF_KEY='semes-full-analysis-ref-v1';
var state={host:null,history:[],catalog:{},model:null,source:null,eq:'',chambers:[],equipments:[],sensors:[],sensorInitialized:false,lastEq:null,phases:PHASES.map(function(p){return p[0];}),stat:'mean',useRef:true,loaded:false,controller:null,timer:null,observer:null,scrollRatio:0,ref:null,display:null};
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
function stamp(p){return String(p.ts||'').replace(' ','T');}
function time(p){return Date.parse(stamp(p));}
function eq(p){return sensorTrendEquipment(p);}
function key(p){return JSON.stringify([p.date||stamp(p).slice(0,10),p.wafer_id]);}
function number(v){return v==null?'자료 없음':Number(v.toPrecision(6)).toString();}
function unique(a){return Array.from(new Set(a));}
function sortPoints(a){return a.slice().sort(function(a,b){return time(a)-time(b)||String(a.wafer_id).localeCompare(String(b.wafer_id));});}
function local(t){var d=new Date(t),pad=function(n){return String(n).padStart(2,'0');};return d.getFullYear()+'-'+pad(d.getMonth()+1)+'-'+pad(d.getDate())+'T'+pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());}
function get(id){return state.host.querySelector('.fa-'+id);}
function status(text){if(state.host)get('status').textContent=text;}
function positiveSteps(row,phase){var v=row&&row.valve_phases;return v&&v.version===2&&v.status==='ok'?unique(((v.steps||{})[phase]||[]).map(Number).filter(function(n){return Number.isInteger(n)&&n>0;})).sort(function(a,b){return a-b;}):[];}
function value(row,sensor,step,stat){
  if(!row)return null;
  var v=sensor===DURATION?((((row.step_timing||{}).steps||{})[step]||{}).seconds):stat==='max'?(((row.peaks||{})[sensor]||{})[step]):(((((row.sensor_metrics||{})[stat]||{})[sensor])||{})[step]);
  return typeof v==='number'&&isFinite(v)?v:null;
}
function score(p){var q=integratedPoint(p,false);return typeof q.ratio==='number'&&!Number.isNaN(q.ratio)?q.ratio:null;}
function normalRange(values){
  var n=0,mean=0,m2=0;values.forEach(function(v){if(typeof v!=='number'||!isFinite(v))return;n++;var d=v-mean;mean+=d/n;m2+=d*(v-mean);});
  if(n<5)return null;var sd=Math.sqrt(Math.max(0,m2/n)),tol=Math.max(sd*3.5,Math.abs(mean)*1e-12,1e-12);return {mean:mean,sd:sd,count:n,lo:mean-tol,hi:mean+tol};
}
function thin(points,limit,alert){
  var ordered=sortPoints(points);if(ordered.length<=limit)return {points:ordered,thinned:false};
  var critical=ordered.filter(alert),rest=ordered.filter(function(p){return !alert(p);}),count=Math.max(0,limit-critical.length),pick=[];
  for(var i=0;i<Math.min(count,rest.length);i++)pick.push(rest[Math.floor(i*rest.length/Math.min(count,rest.length))]);
  var out=sortPoints(critical.concat(pick));return {points:out,thinned:out.length<ordered.length};
}
function validRef(r){return r&&isFinite(Date.parse(r.from))&&isFinite(Date.parse(r.to))&&Date.parse(r.from)<=Date.parse(r.to)&&Array.isArray(r.groups)&&r.groups.length&&r.groups.every(function(g){return typeof g.equipment==='string'&&Array.isArray(g.chambers)&&g.chambers.length;});}
try{var saved=JSON.parse(sessionStorage.getItem(REF_KEY)||'null');if(validRef(saved))state.ref=saved;}catch(e){}
function refPoints(history,r){if(!validRef(r))return [];var a=Date.parse(r.from),b=Date.parse(r.to);return history.filter(function(p){return time(p)>=a&&time(p)<=b&&(!r.recipes||!r.recipes.length||r.recipes.indexOf(p.recipe)>=0)&&r.groups.some(function(g){return g.equipment===eq(p)&&g.chambers.indexOf(p.chamber)>=0;});});}
function saveRef(selection,model){
  if(!selection){clearRef();return;}
  var groups=selection.groups;
  if(!groups){var by=new Map();(selection.points||[]).forEach(function(p){if(!by.has(eq(p)))by.set(eq(p),[]);by.get(eq(p)).push(p.chamber);});groups=Array.from(by).map(function(e){return {equipment:e[0],chambers:unique(e[1])};});}
  var r={from:selection.from,to:selection.to,groups:groups,recipes:selection.recipes||unique((selection.points||[]).map(function(p){return p.recipe;}).filter(Boolean)),model_id:model||state.model};
  if(!validRef(r))return false;state.ref=r;state.useRef=true;try{sessionStorage.setItem(REF_KEY,JSON.stringify(r));}catch(e){}
  refreshRef();if(state.host&&document.getElementById('view-full-analysis').classList.contains('active')&&state.loaded)runQuery(false);return true;
}
function clearRef(){state.ref=null;try{sessionStorage.removeItem(REF_KEY);}catch(e){}refreshRef();if(state.host&&state.loaded&&document.getElementById('view-full-analysis').classList.contains('active'))runQuery(false);}
function refLabel(r){return r.groups.map(function(g){return g.equipment+' · '+g.chambers.join(', ');}).join(' / ')+' · '+r.from.replace('T',' ')+' ~ '+r.to.replace('T',' ');}
function refreshRef(){if(!state.host)return;get('ref-label').textContent=state.ref?refLabel(state.ref):'REF가 없습니다. 종합 산점도에서 평균 비교기간을 클릭하거나 직접 설정하세요.';get('use-ref').checked=state.useRef;get('ref-clear').disabled=!state.ref;}
function css(){if(document.getElementById('full-analysis-style'))return;var s=document.createElement('style');s.id='full-analysis-style';s.textContent='.fa-main{background:var(--panel,#fff);color:var(--txt);border:1px solid var(--line);border-radius:16px;padding:20px}.fa-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0;font-size:12px}.fa-choices{display:flex;gap:8px;flex-wrap:wrap;padding:8px 0;max-height:240px;overflow:auto}.fa-choices label{display:inline-flex;gap:4px;align-items:center}.fa-group{width:100%;border-top:1px solid var(--line);padding-top:8px}.fa-row{border:1px solid var(--line);border-radius:10px;margin:12px 0;padding:10px;min-width:0}.fa-row-head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:12px}.fa-chart-scroll{overflow-x:auto;overflow-y:hidden;min-height:282px;margin-top:8px;padding-bottom:8px}.fa-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;margin:12px 0}.fa-refbar{padding:10px;border:1px dashed var(--accent,#2563eb);border-radius:8px;font-size:12px;margin:12px 0}.fa-ref-editor{margin:10px 0;border:1px solid var(--line);padding:12px;border-radius:8px}.fa-ref-editor[hidden]{display:none!important}.fa-selected{background:var(--accent,#2563eb)!important;color:white!important}.fa-empty{padding:24px;color:var(--txt-dim)}.fa-ref-chambers{display:flex;gap:12px;flex-wrap:wrap}';document.head.appendChild(s);}
function mount(){
  css();state.host=document.getElementById('fullAnalysisApp');
  var overview=document.getElementById('drillTrendChart'),opts=overview&&overview._opts||{},now=Date.now();state.from=opts.axisStart||local(now-86400000);state.to=opts.axisEnd||local(now);
  state.host.innerHTML='<div class="fa-controls"><b>선택기간</b>'+[['24h','최근 24시간'],['3d','최근 3일'],['7d','최근 7일']].map(function(x){return '<button class="btn soft fa-quick" data-range="'+x[0]+'">'+x[1]+'</button>';}).join('')+'<label>시작 <input class="fa-from" type="datetime-local" step="1" required></label><label>종료 <input class="fa-to" type="datetime-local" step="1" required></label><button class="btn primary fa-query">조회</button></div><div class="fa-controls"><label>기준 설비 <select class="fa-equipment"></select></label></div><details open><summary>챔버 선택 (기준 설비)</summary><div class="fa-chambers fa-choices"></div></details><details><summary>비교 설비 선택 (챔버 전체를 한 줄로 표시)</summary><div class="fa-equipments fa-choices"></div></details><details open><summary class="fa-sensor-summary">센서 · 스텝 소요시간 선택</summary><input class="fa-sensor-search" type="search" placeholder="센서 검색" aria-label="전체분석 센서 검색"><div class="fa-sensors fa-choices"></div></details><div class="fa-controls"><b>단계</b><span class="fa-phases"></span><b>통계값</b><span class="fa-statistics"></span></div><div class="fa-refbar"><b>REF</b> <span class="fa-ref-label"></span><div class="fa-controls"><label><input class="fa-use-ref" type="checkbox"> REF와 비교 (단계마다 옆에 표시)</label><button class="btn soft fa-ref-edit">REF 직접 설정</button><button class="btn soft fa-ref-clear">REF 해제</button><button class="btn soft fa-overview">Overview에서 선택</button></div></div><div class="fa-ref-editor" hidden></div><div class="fa-status muted" role="status"></div><div class="fa-legend">'+PHASES.map(function(p){return '<span style="color:'+p[2]+'">● '+p[1]+'</span>';}).join('')+'<span style="color:#dc2626">● 스텝 값 정상 범위 밖</span><span style="color:#b91c1c">◎ 종합 점수 1 이상</span><span>REF: 옅은 배경</span></div><p class="muted" style="font-size:12px">점 하나는 웨이퍼 한 장의 스텝 통계값입니다. 스텝 칸 안에서 시간 순서로 표시하며, 같은 단계의 챔버·REF가 Y축을 공유합니다. 정상 범위는 동일 설비·챔버·레시피·스텝의 정상 판정 실측값 5매 이상에서 평균 ± 3.5σ(ddof=0)로 계산합니다. REF 정상 자료를 포함하며 자료가 부족하면 정상 범위 판정을 표시하지 않습니다.</p><div class="fa-rows"></div>';
  get('from').value=local(Date.parse(state.from));get('to').value=local(Date.parse(state.to));
  get('query').onclick=function(){runQuery(true);};get('equipment').onchange=function(){state.eq=this.value;updateChoices();queueQuery();};
  state.host.querySelectorAll('.fa-quick').forEach(function(b){b.onclick=function(){var days={"24h":1,"3d":3,"7d":7}[b.dataset.range],end=Date.now();get('from').value=local(end-days*86400000);get('to').value=local(end);runQuery(false);};});
  PHASES.forEach(function(p){var label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=true;check.value=p[0];check.onchange=function(){state.phases=PHASES.map(function(p){return p[0];}).filter(function(k){return Array.from(get('phases').querySelectorAll('input:checked')).some(function(c){return c.value===k;});});if(state.display)renderRows();};label.style.color=p[2];label.appendChild(check);label.appendChild(document.createTextNode(p[1]));get('phases').appendChild(label);});
  STATS.forEach(function(pair){var b=document.createElement('button');b.className='btn soft fa-stat';b.dataset.stat=pair[0];b.textContent=pair[1];b.onclick=function(){state.stat=pair[0];if(state.display)renderRows();updateStats();};get('statistics').appendChild(b);});
  get('sensor-search').oninput=filterSensors;get('use-ref').onchange=function(){state.useRef=this.checked;queueQuery();};get('ref-edit').onclick=editRef;get('ref-clear').onclick=clearRef;get('overview').onclick=function(){switchView('overview');};updateStats();refreshRef();
}
function updateStats(){state.host.querySelectorAll('.fa-stat').forEach(function(b){b.classList.toggle('fa-selected',b.dataset.stat===state.stat);b.setAttribute('aria-pressed',String(b.dataset.stat===state.stat));});}
function choices(node,items,selected,change,max,grouped){
  node.innerHTML='';var commands=document.createElement('span');commands.style.width='100%';
  [['전체 선택'+(max?' (최대 '+max+'개)':''),true],['전체 해제',false]].forEach(function(pair){var b=document.createElement('button');b.type='button';b.className='btn soft';b.textContent=pair[0];b.onclick=function(){change(pair[1]?items.slice(0,max||items.length).map(function(x){return x.value;}):[]);};commands.appendChild(b);});node.appendChild(commands);
  var last=null;items.forEach(function(item){
    if(grouped&&item.value!==DURATION){var area=typeof trendSensorArea==='function'?trendSensorArea(item.value):6;if(area!==last){var title=document.createElement('b');title.className='fa-group';title.textContent=typeof TREND_AREAS!=='undefined'?TREND_AREAS[area]:'센서';node.appendChild(title);last=area;}}
    var label=document.createElement('label'),check=document.createElement('input');label.dataset.name=item.label.toLowerCase();check.type='checkbox';check.value=item.value;check.checked=selected.indexOf(item.value)>=0;check.onchange=function(){var checked=Array.from(node.querySelectorAll('input:checked')).map(function(c){return c.value;}),next=selected.filter(function(v){return checked.indexOf(v)>=0;}).concat(checked.filter(function(v){return selected.indexOf(v)<0;}));if(max&&next.length>max){check.checked=false;status('한 번에 최대 '+max+'개까지 선택하세요.');return;}change(next);};label.appendChild(check);label.appendChild(document.createTextNode(item.label));node.appendChild(label);
  });
}
function updateChoices(){
  var equipments=unique(state.history.map(eq)).sort(),select=get('equipment');if(equipments.indexOf(state.eq)<0)state.eq=equipments[0]||'';
  select.innerHTML='';equipments.forEach(function(name){var o=document.createElement('option');o.value=name;o.textContent=name;select.appendChild(o);});select.value=state.eq;
  var chambers=unique(state.history.filter(function(p){return eq(p)===state.eq;}).map(function(p){return p.chamber;}).filter(Boolean)).sort();
  state.chambers=state.lastEq!==state.eq?chambers:state.chambers.filter(function(c){return chambers.indexOf(c)>=0;});state.lastEq=state.eq;state.equipments=state.equipments.filter(function(e){return equipments.indexOf(e)>=0;});
  choices(get('chambers'),chambers.map(function(c){return {value:c,label:c};}),state.chambers,function(v){state.chambers=v;updateChoices();queueQuery();});
  choices(get('equipments'),equipments.map(function(e){return {value:e,label:e};}),state.equipments,function(v){state.equipments=v;updateChoices();queueQuery();},20);
  var io=new Set(state.catalog.io||[]),sensorNames=new Set((state.catalog.recon_sensors||[]).concat(state.catalog.input_sensors||[]));
  if(state.source)Object.keys(state.source._stepTimingRows).forEach(function(k){Object.keys((((state.source._stepTimingRows[k]||{}).sensor_metrics||{}).mean)||{}).forEach(function(name){sensorNames.add(name);});});
  var names=Array.from(sensorNames).filter(function(n){return !io.has(n)&&!/^\d+$/.test(n);});if(typeof trendSensorCompare==='function')names.sort(trendSensorCompare);else names.sort();
  state.availableSensors=names;if(!state.sensorInitialized){state.sensors=names.length?[names[0]]:[DURATION];state.sensorInitialized=true;}
  state.sensors=state.sensors.filter(function(s){return s===DURATION||names.indexOf(s)>=0;});
  choices(get('sensors'),[{value:DURATION,label:'스텝 소요시간 (초)'}].concat(names.map(function(n){return {value:n,label:n};})),state.sensors,function(v){state.sensors=v;updateChoices();queueQuery();},20,true);
  get('sensor-summary').textContent='센서 · 스텝 소요시간 선택 ('+state.sensors.length+'개 / 최대 20개)';filterSensors();
}
function filterSensors(){var query=get('sensor-search').value.trim().toLowerCase();get('sensors').querySelectorAll('label[data-name]').forEach(function(label){label.hidden=!!query&&label.dataset.name.indexOf(query)<0;label.style.display=label.hidden?'none':'inline-flex';});}
function queueQuery(){clearTimeout(state.timer);if(state.loaded)state.timer=setTimeout(function(){runQuery(false);},180);}
async function runQuery(refresh){
  clearTimeout(state.timer);var from=get('from'),to=get('to'),queryFrom=from.value,queryTo=to.value,a=Date.parse(queryFrom),b=Date.parse(queryTo);
  if(!from.validity.valid||!to.validity.valid||!isFinite(a)||!isFinite(b)||a>b){status('시작·종료 일시를 올바르게 입력하세요.');return;}
  if(state.controller)state.controller.abort();var controller=new AbortController();state.controller=controller;var signal=controller.signal;get('query').disabled=true;status('실측 자료 조회 중…');
  function live(){return state.controller===controller&&!signal.aborted;}
  try{
    if(refresh||!state.model){
      var responses=await Promise.all([fetch('/api/scatter?limit=1000000&chamber=all&recipe=all',{signal:signal}),fetch('/api/sensors',{signal:signal})]);
      if(!responses[0].ok||!responses[1].ok)throw Error('조회 조건 또는 센서 목록을 불러오지 못했습니다.');
      var data=await responses[0].json(),catalog=await responses[1].json();if(!live())return;
      state.history=sortPoints((data.points||[]).filter(function(p){return p.wafer_id&&isFinite(time(p));}));state.catalog=catalog;
      if(!state.source||state.model!==data.model_id){state.model=data.model_id;state.source={model_id:state.model,_stepTimingRows:{}};}
      state.rule=data.rule||null;updateChoices();
    }
    var query={eq:state.eq,chambers:state.chambers.slice(),equipments:state.equipments.slice(),useRef:state.useRef,ref:state.ref};
    var current=state.history.filter(function(p){return time(p)>=a&&time(p)<=b&&((eq(p)===query.eq&&query.chambers.indexOf(p.chamber)>=0)||query.equipments.indexOf(eq(p))>=0);}),reference=query.useRef?refPoints(state.history,query.ref):[];
    var all=new Map();current.concat(reference).forEach(function(p){all.set(key(p),p);});var cache=state.source._stepTimingRows;
    var needed=Array.from(all.values()),pending=needed.filter(function(p){var r=cache[key(p)];return !r||r.missing||!r.sensor_metrics||!r.peaks||!r.step_timing||!r.valve_phases||r.valve_phases.version!==2;});
    for(var i=0;i<pending.length;i+=64){
      status('실측 자료 조회 중 '+Math.min(i+64,pending.length)+' / '+pending.length+'매');
      var batch=pending.slice(i,i+64).map(function(p){return {date:p.date||stamp(p).slice(0,10),wid:p.wafer_id};}),result=await fetchTrendBatch(state.source,state.model,'step-times',batch,signal);
      if(!live())return;if(result.model_id!==state.model)throw Error('모델이 변경되었습니다. 조회를 다시 눌러주세요.');
      (result.wafers||[]).forEach(function(row){cache[JSON.stringify([row.date,row.wid])]=row;});
      if(batch.some(function(p){return !cache[JSON.stringify([p.date,p.wid])];}))throw Error('일부 웨이퍼 응답이 누락되었습니다. 다시 조회하세요.');
    }
    if(!live())return;
    state.from=queryFrom;state.to=queryTo;state.loaded=true;state.display={current:current,reference:reference,from:state.from,to:state.to,eq:query.eq,chambers:query.chambers,equipments:query.equipments,cache:cache,ref:query.ref,useRef:query.useRef};
    updateChoices();renderRows();refreshRef();
    var missing=needed.filter(function(p){var r=cache[key(p)];return !r||r.missing;}).length,phasesMissing=needed.filter(function(p){var r=cache[key(p)],v=r&&r.valve_phases;return !v||v.version!==2||v.status!=='ok';}).length;
    status('선택기간 '+current.length+'매 · REF '+reference.length+'매 · 보관 실측 요약 없음 '+missing+'매 · 공정 구간 미확인 '+phasesMissing+'매'+(state.useRef&&state.ref&&!reference.length?' · REF 조건의 현재 모델 자료가 없습니다.':''));
  }catch(e){if(live())status('조회 실패: '+e.message);}finally{if(live())get('query').disabled=false;}
}
function rowsFor(display,sensors){
  var out=[];sensors.forEach(function(sensor){
    if(display.chambers.length)out.push({title:display.eq+' · '+display.chambers.join(', '),sensor:sensor,equipment:display.eq,comparison:false,segments:display.chambers.map(function(ch){var refs=display.reference.filter(function(p){return p.chamber===ch;});return {label:ch,points:display.current.filter(function(p){return eq(p)===display.eq&&p.chamber===ch;}),ref:refs.length?refs:display.reference,refFallback:display.reference.length>0&&!refs.length};})});
    display.equipments.forEach(function(e){out.push({title:e+' · 챔버 전체',sensor:sensor,equipment:e,comparison:true,segments:[{label:'챔버 전체',points:display.current.filter(function(p){return eq(p)===e;}),ref:display.reference}]});});
  });return out;
}
function prepare(row,display,stat,phases){
  var cache=display.cache,sets=[],normalPool=new Map(),normals=new Map(),normalValues=new Map(),phaseSteps={};
  row.segments.forEach(function(sg){sets.push({segment:sg,ref:false,points:sg.points});if(display.useRef&&sg.ref.length)sets.push({segment:sg,ref:true,points:sg.ref});sg.points.concat(display.useRef?sg.ref:[]).forEach(function(p){normalPool.set(key(p),p);});});
  function rangeKey(p,n){return JSON.stringify([eq(p),p.chamber,p.mean_recipe||p.recipe,n]);}
  Array.from(normalPool.values()).forEach(function(p){var s=score(p);if(s==null||s>=1)return;var visited=new Set();phases.forEach(function(phase){positiveSteps(cache[key(p)],phase).forEach(function(n){if(visited.has(n))return;visited.add(n);var v=value(cache[key(p)],row.sensor,n,stat);if(v===null)return;var k=rangeKey(p,n);if(!normalValues.has(k))normalValues.set(k,[]);normalValues.get(k).push(v);});});});
  normalValues.forEach(function(values,k){normals.set(k,normalRange(values));});
  function out(p,n,v){var r=normals.get(rangeKey(p,n));return !!r&&v!==null&&(v<r.lo||v>r.hi);}
  phases.forEach(function(phase){var steps=new Set();sets.forEach(function(set){set.points.forEach(function(p){positiveSteps(cache[key(p)],phase).forEach(function(n){steps.add(n);});});});phaseSteps[phase]=Array.from(steps).sort(function(a,b){return a-b;});});
  sets.forEach(function(set){var sampled=thin(set.points,160,function(p){var s=score(p);return s!==null&&s>=1||phases.some(function(phase){return positiveSteps(cache[key(p)],phase).some(function(n){return out(p,n,value(cache[key(p)],row.sensor,n,stat));});});});set.sample=sampled.points;set.thinned=sampled.thinned;});
  return {sets:sets,steps:phaseSteps,normals:normals,rangeKey:rangeKey,out:out};
}
function scrollSync(el){
  function apply(other){var left=state.scrollRatio*Math.max(0,other.scrollWidth-other.clientWidth);if(Math.abs(other.scrollLeft-left)>1)other.scrollLeft=left;other._faSyncLeft=other.scrollLeft;}
  el.addEventListener('scroll',function(){if(!el.isConnected||!el.clientWidth)return;if(el._faSyncLeft!=null&&Math.abs(el.scrollLeft-el._faSyncLeft)<=1)return;var range=el.scrollWidth-el.clientWidth;if(range<=0)return;state.scrollRatio=el.scrollLeft/range;get('rows').querySelectorAll('.fa-chart-scroll').forEach(function(other){if(other!==el)apply(other);});},{passive:true});apply(el);
}
function renderRows(){
  if(!state.display)return;if(state.observer)state.observer.disconnect();state.observer=null;
  var host=get('rows');host.innerHTML='';var phases=PHASES.filter(function(p){return state.phases.indexOf(p[0])>=0;}),rows=rowsFor(state.display,state.sensors);
  if(!state.sensors.length||!phases.length||!rows.length){host.innerHTML='<div class="fa-empty">센서, 단계, 챔버 또는 설비를 하나 이상 선택하세요.</div>';return;}
  var snapshot=Object.assign({},state.display,{stat:state.stat,phases:phases});
  rows.forEach(function(row){
    row.display=snapshot;var section=document.createElement('section');section.className='fa-row';section.dataset.sensor=row.sensor;section.dataset.equipment=row.equipment;var head=document.createElement('div');head.className='fa-row-head';
    var title=document.createElement('b');title.textContent=row.title+' · '+(row.sensor===DURATION?'스텝 소요시간 (초)':row.sensor+' (원본 단위)');head.appendChild(title);
    row.note=document.createElement('span');row.note.className='muted';row.note.textContent=row.segments.map(function(sg){return sg.label+' '+sg.points.length+'매'+(snapshot.useRef?' / REF '+sg.ref.length+'매':'')+(sg.refFallback?' (REF 챔버 전체와 비교)':'');}).join(' · ');head.appendChild(row.note);
    var zoom=document.createElement('button');zoom.className='btn soft';zoom.textContent='크게 보기 · 이미지 저장';zoom.style.marginLeft='auto';zoom.onclick=function(){drawRow(row);zoomRow(row);};head.appendChild(zoom);section.appendChild(head);
    var plot=document.createElement('div');plot.className='fa-chart-scroll';plot.textContent='실측 통계 준비 중…';section.appendChild(plot);row.plot=plot;plot._faRow=row;host.appendChild(section);
    if(!state.observer&&window.IntersectionObserver)state.observer=new IntersectionObserver(function(entries){entries.forEach(function(entry){if(entry.isIntersecting)drawRow(entry.target._faRow);});},{rootMargin:'300px'});
    if(state.observer)state.observer.observe(plot);else drawRow(row);
  });
}
function drawRow(row){
  if(row.drawn||!row.plot.isConnected)return;row.drawn=true;
  var d=row.display,prepared=prepare(row,d,d.stat,d.phases.map(function(p){return p[0];})),cache=d.cache,H=282,T=48,B=42,AX=48,SW=68,GAP=26,svg='',x=0,points=new Map();row.pointMap=points;
  d.phases.forEach(function(phase,pi){
    var phaseKey=phase[0],steps=prepared.steps[phaseKey],lo=Infinity,hi=-Infinity;
    prepared.sets.forEach(function(set){set.points.forEach(function(p){var own=positiveSteps(cache[key(p)],phaseKey);steps.forEach(function(n){if(own.indexOf(n)<0)return;var v=value(cache[key(p)],row.sensor,n,d.stat);if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v);}});});});
    if(!isFinite(lo)){lo=0;hi=1;}var padding=(hi-lo)*.08||Math.max(Math.abs(lo)*.05,1);lo-=padding;hi+=padding;
    function y(v){return T+(1-(v-lo)/(hi-lo))*(H-T-B);}
    if(pi){svg+='<line x1="'+(x+GAP/2)+'" x2="'+(x+GAP/2)+'" y1="4" y2="'+(H-8)+'" stroke="var(--line,#ccc)"/>';x+=GAP;}
    svg+='<g class="fa-stage" data-phase="'+phaseKey+'" data-y-min="'+lo+'" data-y-max="'+hi+'">';
    var axis=x;for(var tick=0;tick<=4;tick++){var v=lo+(hi-lo)*tick/4;svg+='<text x="'+(axis+AX-5)+'" y="'+(y(v)+4)+'" font-size="11" text-anchor="end" fill="var(--txt-dim,#64748b)">'+esc(number(v))+'</text>';}
    x+=AX;var phaseStart=x;
    if(!steps.length){svg+='<text x="'+x+'" y="100" font-size="12" fill="var(--txt-dim,#64748b)">공정 구간 자료 없음</text>';x+=170;}
    prepared.sets.forEach(function(set,si){
      if(!steps.length)return;if(si)x+=set.ref?12:18;var left=x,width=steps.length*SW;
      svg+='<rect x="'+left+'" y="'+T+'" width="'+width+'" height="'+(H-T-B)+'" fill="'+(set.ref?'var(--panel-2,#eaf0ff)':'transparent')+'"'+(set.ref?' stroke="'+phase[2]+'" stroke-dasharray="4 3"':'')+'/>';
      svg+='<text x="'+(left+4)+'" y="35" font-size="11" font-weight="600" fill="'+phase[2]+'">'+esc(set.segment.label+(set.ref?' REF':''))+'</text>';
      for(var t=0;t<=4;t++){var val=lo+(hi-lo)*t/4;svg+='<line x1="'+left+'" x2="'+(left+width)+'" y1="'+y(val)+'" y2="'+y(val)+'" stroke="var(--line,#e2e8f0)"/>';}
      steps.forEach(function(n,j){
        var sx=left+j*SW;if(j)svg+='<line x1="'+sx+'" x2="'+sx+'" y1="'+T+'" y2="'+(H-B)+'" stroke="var(--line,#e2e8f0)" stroke-dasharray="2 3"/>';
        svg+='<text x="'+(sx+SW/2)+'" y="'+(H-B+17)+'" font-size="11" text-anchor="middle" fill="var(--txt-dim,#64748b)">스텝 '+n+'</text>';
        set.sample.forEach(function(p,i){
          var rowData=cache[key(p)];if(positiveSteps(rowData,phaseKey).indexOf(n)<0)return;var v=value(rowData,row.sensor,n,d.stat);if(v===null)return;
          var cx=sx+4+(set.sample.length>1?i/(set.sample.length-1):.5)*(SW-8),cy=y(v),out=prepared.out(p,n,v),total=score(p),bad=total!==null&&total>=1,range=prepared.normals.get(prepared.rangeKey(p,n)),k=key(p);points.set(k,p);
          var tip=(set.ref?'REF · ':'')+eq(p)+' · '+p.chamber+' · '+p.wafer_id+'\n종료 '+stamp(p)+' · 레시피 '+(p.recipe||'')+'\n'+phase[1]+' · 스텝 '+n+' · '+(row.sensor===DURATION?'실제 소요시간(초)':row.sensor+' '+STATS.find(function(s){return s[0]===d.stat;})[1])+': '+number(v)+'\n정상 범위: '+(range?number(range.lo)+' ~ '+number(range.hi)+' ('+range.count+'매)'+(out?' · 범위 밖':' · 범위 안'):'정상 판정 실측 자료 5매 미만')+'\n종합 점수: '+(total===Infinity?'∞':number(total))+' · 클릭하면 이상 분석';
          svg+='<g data-fa-wafer="'+esc(k)+'" tabindex="0" role="button" aria-label="'+esc(tip)+'" style="cursor:pointer">'+(bad?'<circle class="fa-ng-ring" cx="'+cx+'" cy="'+cy+'" r="5.7" fill="none" stroke="#b91c1c" stroke-width="1.2"/>':'')+'<circle class="fa-point'+(out?' fa-outlier':'')+'" data-reference="'+set.ref+'" data-step="'+n+'" data-value="'+v+'" cx="'+cx+'" cy="'+cy+'" r="'+(out?3.6:2.6)+'" fill="'+(out?'#dc2626':row.comparison?chamberColor(p.chamber):phase[2])+'" fill-opacity="'+(out?1:set.ref?0.45:0.7)+'"/><title>'+esc(tip)+'</title></g>';
        });
      });x=left+width;
    });
    svg+='<text x="'+phaseStart+'" y="15" font-size="13" font-weight="700" fill="'+phase[2]+'">'+phase[1]+' · 단계 공통 Y축</text></g>';
  });
  var width=x+12;svg+='<text x="4" y="'+(H-5)+'" font-size="11" fill="var(--txt-dim,#64748b)">선택기간 '+esc(d.from.replace('T',' ')+' ~ '+d.to.replace('T',' '))+(d.useRef&&d.ref?' · REF '+esc(d.ref.from.replace('T',' ')+' ~ '+d.ref.to.replace('T',' ')):'')+'</text>';
  row.plot.innerHTML='<svg xmlns="http://www.w3.org/2000/svg" class="fa-svg" role="img" aria-label="'+esc(row.title+' '+row.sensor+' 단계별 추이')+'" width="'+width+'" height="'+H+'" viewBox="0 0 '+width+' '+H+'" style="display:block;max-width:none">'+svg+'</svg>';
  if(prepared.sets.some(function(set){return set.thinned;}))row.note.textContent+=' · 기본 160매 간격 표시 (종합 기준 초과·스텝 이탈 웨이퍼는 모두 유지)';
  function open(event){var point=event.target.closest('[data-fa-wafer]');if(!point)return;var p=row.pointMap.get(point.getAttribute('data-fa-wafer'));if(p)openWaferInDetail(encodeURIComponent(p.wafer_id),p.date||stamp(p).slice(0,10),false);}
  row.plot.addEventListener('click',open);row.plot.addEventListener('keydown',function(event){if(event.key==='Enter'||event.key===' '){event.preventDefault();open(event);}});scrollSync(row.plot);
}
function imageSVG(svg){
  var clone=svg.cloneNode(true),src=svg.querySelectorAll('*'),dst=clone.querySelectorAll('*');
  src.forEach(function(node,i){var style=getComputedStyle(node),copy=dst[i];['fill','stroke','opacity','fill-opacity','stroke-width','font-size','font-weight','font-family'].forEach(function(k){var v=style.getPropertyValue(k);if(v)copy.setAttribute(k,v);});copy.removeAttribute('style');copy.removeAttribute('class');copy.removeAttribute('tabindex');copy.removeAttribute('role');});
  var rect=document.createElementNS('http://www.w3.org/2000/svg','rect');rect.setAttribute('width',svg.getAttribute('width'));rect.setAttribute('height',svg.getAttribute('height'));rect.setAttribute('fill',getComputedStyle(state.host).getPropertyValue('--panel').trim()||'#fff');clone.insertBefore(rect,clone.firstChild);clone.setAttribute('xmlns','http://www.w3.org/2000/svg');return new XMLSerializer().serializeToString(clone);
}
function download(row){
  var svg=row.plot.querySelector('svg'),W=Number(svg.getAttribute('width')),H=Number(svg.getAttribute('height')),source=imageSVG(svg),url=URL.createObjectURL(new Blob([source],{type:'image/svg+xml;charset=utf-8'})),img=new Image();
  function save(href,ext){var link=document.createElement('a');link.href=href;link.download=('full_analysis_'+row.title+'_'+row.sensor).replace(/[<>:"/\\|?*\x00-\x1f]/g,'_').slice(0,120)+'.'+ext;document.body.appendChild(link);link.click();link.remove();}
  img.onload=function(){try{var scale=Math.min(3.2,32000/W,32000/H,Math.sqrt(32000000/(W*H))),canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.floor(W*scale));canvas.height=Math.max(1,Math.floor(H*scale));var ctx=canvas.getContext('2d');ctx.drawImage(img,0,0,canvas.width,canvas.height);var png=canvas.toDataURL('image/png');if(png==='data:,')throw Error('canvas');save(png,'png');}catch(e){save(url,'svg');}setTimeout(function(){URL.revokeObjectURL(url);},1000);};img.onerror=function(){save(url,'svg');setTimeout(function(){URL.revokeObjectURL(url);},1000);};img.src=url;
}
function zoomRow(row){
  var prior=document.activeElement,overflow=document.documentElement.style.overflow,dialog=document.createElement('div');dialog.className='fa-zoom';dialog.setAttribute('role','dialog');dialog.setAttribute('aria-modal','true');dialog.setAttribute('aria-label',row.title+' 확대');dialog.style.cssText='position:fixed;inset:16px;z-index:22000;display:flex;flex-direction:column;gap:10px;background:var(--panel,#fff);color:var(--txt);padding:16px;border:1px solid var(--line);border-radius:12px';
  var header=document.createElement('div');header.className='fa-controls';var title=document.createElement('strong');title.textContent=row.title+' · '+row.sensor;header.appendChild(title);var save=document.createElement('button');save.className='btn primary fa-save-image';save.textContent='이미지 저장 (PNG)';save.onclick=function(){download(row);};header.appendChild(save);var close=document.createElement('button');close.className='btn soft';close.textContent='닫기 (Esc)';header.appendChild(close);dialog.appendChild(header);
  var body=document.createElement('div');body.style.cssText='overflow:auto;flex:1;min-height:0';var clone=row.plot.querySelector('svg').cloneNode(true);clone.style.width=Math.round(Number(clone.getAttribute('width'))*1.6)+'px';clone.style.height='auto';body.appendChild(clone);dialog.appendChild(body);document.body.appendChild(dialog);document.documentElement.style.overflow='hidden';
  function end(){dialog.remove();document.documentElement.style.overflow=overflow;if(prior&&prior.isConnected)prior.focus({preventScroll:true});}
  close.onclick=end;dialog.addEventListener('keydown',function(e){if(e.key==='Escape'){e.preventDefault();end();}if(e.key==='Tab'){if(e.shiftKey&&document.activeElement===save){e.preventDefault();close.focus();}else if(!e.shiftKey&&document.activeElement===close){e.preventDefault();save.focus();}}});save.focus();
}
function editRef(){
  var node=get('ref-editor');if(!node.hidden){node.hidden=true;return;}node.hidden=false;
  var r=state.ref,defaultEq=r&&r.groups.length?r.groups[0].equipment:state.eq,from=Date.parse(get('from').value),to=Date.parse(get('to').value),len=to-from,selected=r&&r.groups.length?r.groups[0].chambers.slice():(state.chambers.length?state.chambers.slice():null);
  node.innerHTML='<b>REF 직접 설정</b><div class="fa-controls"><label>설비 <select class="fa-r-equipment"></select></label><label>시작 <input class="fa-r-from" type="datetime-local" step="1" required></label><label>종료 <input class="fa-r-to" type="datetime-local" step="1" required></label></div><div class="fa-ref-chambers"></div><div class="fa-controls"><button class="btn soft fa-r-previous">선택기간 바로 전 (같은 길이)</button><button class="btn soft fa-r-day">하루 전 같은 시간대</button><button class="btn soft fa-r-week">일주일 전 같은 시간대</button><button class="btn primary fa-r-save">REF로 저장</button><span class="fa-r-status" role="status"></span></div>';
  var select=node.querySelector('.fa-r-equipment'),a=node.querySelector('.fa-r-from'),b=node.querySelector('.fa-r-to');unique(state.history.map(eq)).sort().forEach(function(name){var o=document.createElement('option');o.value=name;o.textContent=name;select.appendChild(o);});select.value=defaultEq;
  a.value=r?local(Date.parse(r.from)):local(from-len);b.value=r?local(Date.parse(r.to)):local(from);
  function chambers(){var values=unique(state.history.filter(function(p){return eq(p)===select.value;}).map(function(p){return p.chamber;})).sort();if(selected===null||(selected.length&&selected.every(function(c){return values.indexOf(c)<0;})))selected=values.slice();choices(node.querySelector('.fa-ref-chambers'),values.map(function(c){return {value:c,label:c};}),selected,function(v){selected=v;chambers();});}
  select.onchange=function(){selected=null;chambers();};chambers();
  [['previous',from-len,from],['day',from-86400000,to-86400000],['week',from-604800000,to-604800000]].forEach(function(v){node.querySelector('.fa-r-'+v[0]).onclick=function(){a.value=local(v[1]);b.value=local(v[2]);};});
  node.querySelector('.fa-r-save').onclick=function(){
    var start=Date.parse(a.value),end=Date.parse(b.value),message=node.querySelector('.fa-r-status');if(!a.validity.valid||!b.validity.valid||!isFinite(start)||!isFinite(end)||start>end||!selected.length){message.textContent='기간과 챔버를 올바르게 선택하세요.';return;}
    var proposed={from:a.value,to:b.value,groups:[{equipment:select.value,chambers:selected.slice()}],recipes:[]};if(!refPoints(state.history,proposed).length){message.textContent='이 조건의 현재 모델 웨이퍼가 없습니다.';return;}saveRef(proposed,state.model);node.hidden=true;
  };
}
async function init(){if(state.host){refreshRef();return runQuery(true);}mount();await runQuery(true);}
w.FullAnalysis={init:init,saveRef:saveRef,clearRef:clearRef,getRef:function(){return state.ref;},utils:{positiveSteps:positiveSteps,value:value,normalRange:normalRange,thin:thin,refPoints:refPoints,rowsFor:rowsFor,prepare:prepare},state:state};
})(window);
