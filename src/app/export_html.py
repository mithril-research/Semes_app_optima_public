# -*- coding: utf-8 -*-
"""데모용 HTML 내보내기 — 실제 index.html + '현재 데이터 스냅샷' + 읽기전용 fetch mock.

앱/서버 없이 브라우저에서 열면 현재 대시보드와 동일하게 동작(반응형·토글·필터·상세)한다.
단, 새 검사/저장/삭제 등 변경 동작은 데모 모드로 무효화(읽기전용). 고객사 데모용.

핵심:
  - 대시보드(개요/산점도/웨이퍼목록/달력)는 baked WAFERS 로 JS 에서 재계산 → 토글/필터 그대로 동작.
  - 상세(누적/센서/overlay)는 실제 아카이브 응답을 구워넣고, zone-contrib 만 baked z 로 클라이언트 계산.
"""
import base64
import json
from pathlib import Path

# 대시보드 계산 + 상세/설정 라우팅 mock. WAFERS/DETAIL/CFG 는 아래에서 JSON 으로 주입.
_MOCK_TMPL = r"""
<script>
/* ═══════════ 데모 모드: 읽기전용 fetch mock (실서버 아님) ═══════════ */
(function(){
  window.SEMES_OFFLINE_EXPORT = true;
  var WAFERS = __WAFERS__;
  var DETAIL = __DETAIL__;   // {"date/wid": {meta, wseries, sensorsAll, sensorSeries:{j:...}, z:[[..]], cols:[..]}}
  var CFG = __CFG__;
  window.SEMES_TREND_EXPORT = CFG.trend_export || null;
  window.SEMES_EXPORT_FEATURES = CFG.export_features || {scores:true,trends:true,detail:true,chamber_means:true};
  var RECIPES = CFG.recipes||[];                 // 데이터에 실제로 있는 유형(체크박스용)
  var KNOWN_RECIPES = CFG.known_recipes||RECIPES; // 모델이 아는 레시피(배율 설정용)
  var SENSORS = CFG.sensors||[];
  var RETEN = CFG.reten!=null?CFG.reten:60, UISENS = CFG.uisens!=null?CFG.uisens:0.8;
  var NOW = new Date(CFG.now || Date.now());
  function iso(d){function p(x){return String(x).padStart(2,'0');}
    return d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate())+'T'+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());}
  function dstr(d){return iso(d).slice(0,10);}
  function q(url,key,def){var m=url.match(new RegExp('[?&]'+key+'=([^&]*)'));return m?decodeURIComponent(m[1]):def;}
  // ★ 시각 파싱은 한 곳에서 — 이력의 ts 는 'YYYY-MM-DD HH:MM:SS'(공백 구분)라
  //   브라우저에 따라 Invalid Date 가 된다. 그러면 그 웨이퍼가 조회 구간 계산에서
  //   빠져 앱과 칸 개수가 달라진다(실제로 앱 8칸 · HTML 7칸으로 어긋났다).
  function _T(x){ return new Date(String(x||'').replace(' ','T')); }
  function monthlyBounds(t0,t1,day,clip){
    if(!Number.isInteger(day)||day<1||day>31)day=1;
    function anchor(y,m){var first=new Date(y,m,1);return new Date(first.getFullYear(),first.getMonth(),Math.min(day,new Date(first.getFullYear(),first.getMonth()+1,0).getDate()));}
    var cur=anchor(t0.getFullYear(),t0.getMonth()),out=[],last=new Date(+t1-1000);if(cur>t0)cur=anchor(t0.getFullYear(),t0.getMonth()-1);
    while(cur<t1){if(!(clip&&day!==1&&cur>t0&&cur.getFullYear()===last.getFullYear()&&cur.getMonth()===last.getMonth()&&cur.getDate()===last.getDate()))out.push(cur);cur=anchor(cur.getFullYear(),cur.getMonth()+1);}out.push(cur);return out;
  }
  function inWin(w,t0,t1){var t=_T(w.ts);return t>=t0&&t<t1;}
  function R(data){return Promise.resolve({ok:true,status:200,json:function(){return Promise.resolve(data);}});}

  // zone-contrib: baked z 로 백엔드와 동일 계산(구간 합=0 이면 좌우 확장).
  function zoneContrib(det,a,b){
    var z=det&&det.z, cols=det&&det.cols||SENSORS;
    if(!z||!z.length) return {sensors:[],total:0};
    var T=z.length, S=z[0].length;
    var i0=Math.max(0,Math.min(T-1,Math.round(a*(T-1)))), i1=Math.max(0,Math.min(T-1,Math.round(b*(T-1))));
    if(i1<i0){var t=i0;i0=i1;i1=t;}
    function sum(lo,hi){var per=new Array(S).fill(0),tot=0;
      for(var i=lo;i<=hi;i++)for(var j=0;j<S;j++){var v=z[i][j];if(v>0){per[j]+=v;tot+=v;}} return {per:per,tot:tot};}
    var r=sum(i0,i1);
    if(r.tot<=1e-12){var step=Math.max(2,Math.round(T*0.02)),lo=i0,hi=i1;
      while(r.tot<=1e-12&&(lo>0||hi<T-1)){lo=Math.max(0,lo-step);hi=Math.min(T-1,hi+step);r=sum(lo,hi);}}
    var out=[];for(var j=0;j<S;j++)out.push({index:j,sensor:cols[j]||('S'+j),
      share:r.tot>1e-12?Math.round(1000*r.per[j]/r.tot)/10:0});
    out.sort(function(x,y){return y.share-x.share;});
    return {sensors:out,total:Math.round(r.tot*1000)/1000};
  }


  // ── 산포 추세(서버 /api/dispersion 과 동일 계산) ─────────────────────────
  function _med(a){ if(!a.length) return NaN; var b=a.slice().sort(function(x,y){return x-y;});
    var m=b.length>>1; return b.length%2?b[m]:(b[m-1]+b[m])/2; }
  function _pct(a,q){ if(!a.length) return NaN;                    // numpy 'linear' 보간과 동일
    var b=a.slice().sort(function(x,y){return x-y;}), pos=(b.length-1)*q/100;
    var lo=Math.floor(pos), hi=Math.ceil(pos);
    return lo===hi?b[lo]:b[lo]+(b[hi]-b[lo])*(pos-lo); }
  function _std(a){ if(a.length<2) return 0; var m=a.reduce(function(s,v){return s+v;},0)/a.length;
    return Math.sqrt(a.reduce(function(s,v){return s+(v-m)*(v-m);},0)/(a.length-1)); }
  function _mad(a){ var m=_med(a), d=a.map(function(v){return Math.abs(v-m);});
    var v=_med(d)*1.4826; if(!(v>1e-9)){ var sd=_std(a); v=sd>1e-9?sd:1; } return v; }
  function _metric(a,kind){ if(a.length<2) return NaN;
    if(kind==='range') return Math.max.apply(null,a)-Math.min.apply(null,a);
    if(kind==='p90p50') return _pct(a,90)-_pct(a,50);
    return _pct(a,75)-_pct(a,25); }
  function _marginSigma(vals){                            // 임계까지 몇 σ (로그공간)
    var a=vals.filter(function(v){return isFinite(v)&&v>0;});
    if(a.length<2) return NaN;
    var l=a.map(Math.log), med=_med(l);
    var sig=_med(l.map(function(v){return Math.abs(v-med);}))*1.4826;
    if(!(sig>1e-9)) sig=_std(l);
    if(!(sig>1e-9)) return med<0?99:0;
    return Math.max(-99,Math.min(99,-med/sig));
  }
  // ★ Python 의 round() 와 같은 규칙(절반은 짝수로). JS Math.round 는 절반을 올려
  //   보내므로 1.88375 가 앱 1.8837 / HTML 1.8838 로 갈렸다. 자릿수 규칙도 맞춰야 한다.
  function _pyround(x,nd){
    // Math.round(x*1e4)/1e4 는 곱하는 순간 오차가 생긴다(1.88375*1e4 = 18837.499999999996
    // → 내림). Python 의 round 는 곱하지 않고 '그 double 의 십진 표현'을 반올림하므로
    // 1.8838 이 된다. toFixed 가 같은 방식이라 그대로 쓴다.
    if(x==null||!isFinite(x)) return x;
    return Number(Number(x).toFixed(nd));
  }
  // 초과 강도 = 평균(max(0, r - base)^pw). base 는 '어디부터 초과로 볼 것인가'.
  // 앱과 동일하게 기본은 '심각 배수' — 임계(1.0)부터 세면 칸이 온통 빨개진다.
  function _excess(vals,pw,base){
    var b=(base==null?1:+base);
    var a=vals.filter(function(v){return isFinite(v);});
    if(!a.length) return {d:NaN,rate:0,mex:0};
    var ex=a.map(function(v){return Math.max(0,v-b);});
    var over=ex.filter(function(v){return v>0;});
    var d=ex.reduce(function(s,v){return s+Math.pow(v,pw);},0)/ex.length;
    var mex=over.length?over.reduce(function(s,v){return s+v;},0)/over.length:0;
    return {d:d,rate:over.length/ex.length,mex:mex};
  }
  function _theilSen(xs,ys){ var n=xs.length; if(n<3) return 0;
    if(n>60){ var ix=[],x2=[],y2=[]; for(var k=0;k<60;k++){ var i=Math.round(k*(n-1)/59); x2.push(xs[i]); y2.push(ys[i]); }
      xs=x2; ys=y2; n=xs.length; }
    var sl=[]; for(var i=0;i<n-1;i++) for(var j=i+1;j<n;j++){ var dx=xs[j]-xs[i]; if(dx>0) sl.push((ys[j]-ys[i])/dx); }
    return sl.length?_med(sl):0; }
  function __dispersion(url, source, fixedBounds){
    source=source||WAFERS;
    var rng=q(url,'range','all'), scale=q(url,'scale','month');
    var metric=q(url,'metric','iqr'), gapMin=+q(url,'gap_min','10'), minN=+q(url,'min_n','2');
    var basis=q(url,'basis','excess'), hold=(q(url,'hold','0')==='1');
    var groupBy=(q(url,'group_by','lot')==='gap')?'gap':'lot';   // ★ 앱과 동일: 기본 LOT
    // 초과 기준선 — 앱과 동일하게 기본은 심각 배수(over=thr 면 임계 1.0)
    var ebase=1;
    if(String(q(url,'over','crit')).toLowerCase()!=='thr'){
      var _cb=((RULE.bands||{}).crit);
      ebase=Math.max(1,(+_cb)||1.2);
    }
    var inclFirst=(q(url,'include_first','1')!=='0');            // LOT 첫 장 포함 여부(앱과 동일)
    if(['spread','margin','cpu','excess','excess2'].indexOf(basis)<0) basis='excess';
    var _slq=+q(url,'slices','0')||0;                      // 0 = 자동(앱과 동일 규칙)
    var slices=_slq;
    var chsv=q(url,'chambers',''), rcsv=q(url,'recipes','');
    var rcOne=q(url,'recipe','all');                       // 유형 드롭다운(앱과 동일)
    var t1=NOW,t0;
    if(rng==='custom'){ t0=new Date(q(url,'start',iso(new Date(NOW-7*86400e3)))); t1=new Date(+new Date(q(url,'end',iso(NOW)))+1000); }
    else if(rng==='all'){                              // ★ 앱과 동일: 전체 기간
      var _ts=source.map(function(w){return _T(w.ts).getTime();}).filter(isFinite);
      t0=new Date(_ts.length?Math.min.apply(null,_ts):(NOW-7*86400e3));
      t1=new Date(_ts.length?(Math.max.apply(null,_ts)+86400e3):NOW); }
    else { var h={'1h':1,'6h':6,'12h':12,'24h':24,'7d':168,'30d':720}[rng]||168; t0=new Date(NOW-h*3600e3); }
    if(fixedBounds){t0=_T(fixedBounds[0]);t1=_T(fixedBounds[fixedBounds.length-1]);}
    var chset=(chsv&&chsv!=='__NONE__')?chsv.split(','):null, rcset=(rcsv&&rcsv!=='__NONE__')?rcsv.split(','):null;
    // ★ '__NONE__' = 사용자가 유형을 전부 해제한 상태 → 계산 대상 없음(앱과 동일하게 빈 결과)
    var _noneCh=(chsv==='__NONE__'), _noneRc=(rcsv==='__NONE__');
    var ws=(_noneCh||_noneRc)?[]:source.filter(function(w){ return inWin(w,t0,t1)
      &&(chset?chset.indexOf(w.chamber)>=0:true)&&(rcset?rcset.indexOf(w.recipe)>=0:true)
      // 구간은 전체로 잡고, 계산 대상만 유형으로 거른다(앱과 같은 순서)
      &&((rcOne&&rcOne!=='all')?(String(w.recipe||'')===rcOne):true); });
    // 버킷 경계(개요와 동일)
    if(['day','week','month'].indexOf(scale)<0){ var td=(t1-t0)/86400e3; scale=td<=31?'day':(td<=183?'week':'month'); }
    var bounds=[];
    if(scale==='month')bounds=monthlyBounds(t0,t1,+q(url,'month_start_day','1'),rng==='custom');
    else { var stepD=(scale==='week'?7:1); var b0=new Date(t0); b0.setHours(0,0,0,0);
      if(scale==='week'){ var dow=(b0.getDay()+6)%7; b0=new Date(b0-dow*86400e3); }
      var c2=b0; while(c2<t1){ bounds.push(new Date(c2)); c2=new Date(+c2+stepD*86400e3); } bounds.push(c2); }
    if(rng==='custom'&&!fixedBounds)bounds=[t0].concat(bounds.filter(function(b){return b>t0&&b<t1;}),[t1]);
    if(fixedBounds)bounds=fixedBounds.map(_T);
    else if(bounds.length>401) bounds=bounds.slice(-401);
    var count=bounds.length-1;
    if(!slices){                                           // 자동 — 칸의 자연스러운 하위 단위
      if(scale==='month'){                                 // 월 → 그 달의 일수(하루 = 한 조각)
        var _mx=0;
        for(var _i=0;_i<count;_i++){
          var _d=Math.round((bounds[_i+1]-bounds[_i])/86400e3);
          if(_d>_mx) _mx=_d;
        }
        slices=_mx||30;
      } else if(scale==='week'){ slices=7; }
      else { slices=24; }
    }
    slices=Math.max(4,Math.min(slices,60));
    // 레시피별 기저 제거(전체 구간 기준)
    function _rv(w){ return (w.ratio_x!=null)?+w.ratio_x:(w.ratio!=null?+w.ratio:NaN); }   // 정밀값 우선(서버와 동일 입력)
    var nNoScore=ws.filter(function(w){return !isFinite(_rv(w));}).length;
    ws=ws.filter(function(w){return isFinite(_rv(w));});
    var byRc={}; ws.forEach(function(w){ (byRc[w.recipe||'']=byRc[w.recipe||'']||[]).push(_rv(w)); });
    var base={}; Object.keys(byRc).forEach(function(rc){ base[rc]={m:_med(byRc[rc]),s:_mad(byRc[rc])}; });
    // 챔버별 공정 그룹(간격 > gap_min)
    var perCh={}, nLotKnown=0; ws.forEach(function(w){ var b=base[w.recipe||'']||{m:0,s:1};
      var _lot=(groupBy==='lot'&&w.lot)?String(w.lot):null; if(_lot) nLotKnown++;
      (perCh[w.chamber||'-']=perCh[w.chamber||'-']||[]).push(
        {t:_T(w.ts).getTime(),u:(_rv(w)-b.m)/b.s,r:_rv(w),lot:_lot,first:w.lot_first}); });
    var groups={}, allD=[], gaps=[], nSmall=0, nFirstEx=0;
    Object.keys(perCh).forEach(function(ch){
      var it=perCh[ch].sort(function(a,b){return a.t-b.t;}), cur=[], out=[];
      function close(lot){
        // ★ 첫 장 제외(앱 _dispersion_groups 와 동일 규칙) — 시간순이라 맨 앞이 첫 장.
        //   LOT 을 아는 묶음에만 적용한다(미상은 시간으로 묶은 것뿐이라 첫 장이 아니다).
        if(!inclFirst && lot && cur.length>=1){
          var identified=cur.some(function(x){return typeof x.first==='boolean';});
          if(identified){var before=cur.length;cur=cur.filter(function(x){return !x.first;});nFirstEx+=before-cur.length;}
          else {cur=cur.slice(1);nFirstEx++;} // Compatibility with older snapshots.
        }
        // ★ 필요 장수는 기준마다 다르다(앱과 동일). 초과 강도는 그룹 평균이라 1장으로도
        //   성립한다 — 2장을 일괄 요구하면 2장짜리 LOT 에서 첫 장을 빼는 순간 사라진다.
        var need=(basis==='excess'||basis==='excess2')?1:Math.max(2,minN);
        if(cur.length<need){ if(cur.length) nSmall++; cur=[]; return; }
        var d, rate=null, mex=null;
        if(basis==='margin'||basis==='cpu'){ d=_marginSigma(cur.map(function(x){return x.r;}));
          if(basis==='cpu') d=d/3; }
        else if(basis==='excess'||basis==='excess2'){
          var e=_excess(cur.map(function(x){return x.r;}), basis==='excess2'?2:1, ebase);
          d=e.d; rate=e.rate; mex=e.mex; }
        else d=_metric(cur.map(function(x){return x.u;}),metric);
        // 그룹 단계에서 4자리로 접는 것도 앱과 같아야 한다(server _close 가 round(...,4)).
        // 여기서 원값을 들고 있다가 칸에서 한 번만 접으면 마지막 자리가 어긋난다.
        if(isFinite(d)){ out.push({t0:cur[0].t,t1:cur[cur.length-1].t,n:cur.length,disp:d,lot:(lot||null),
          rate:(rate!=null?_pyround(rate,4):null), mex:(mex!=null?_pyround(mex,4):null)});
          allD.push(d); }
        cur=[]; }
      // ★ 서버(_dispersion_groups)와 같은 규칙: LOT 해시로 묶고, LOT 미상만 시간 간격 폴백.
      //   예전엔 여기서 시간 간격만 썼기 때문에, 내보낸 HTML 의 산포가 앱과 달랐다.
      if(groupBy==='lot'){
        // ★ 서버 _lot_runs 와 같은 규칙 — 두 가지다.
        //   1) 같은 LOT 해시라도 간격이 벌어지면 끊는다('한 번 돌린 LOT'). 통째로 합치면
        //      묶음이 첫 구간 칸에만 놓여 그 LOT 이 다시 돈 날짜가 통째로 빈칸이 된다.
        //   2) LOT 미상 웨이퍼가 붙어 있으면 그 묶음에 흡수한다. 끊어 두면 진짜 첫 장에
        //      태그가 없을 때 두 번째 장이 '첫 장'으로 밀려 표시된다.
        var seq=it.slice().sort(function(a,b){ return a.t-b.t; });
        var runs=[], run=[seq[0]], runLot=seq[0].lot||null;
        for(var i=1;i<seq.length;i++){
          var g2=(seq[i].t-run[run.length-1].t)/60000; gaps.push(g2);
          var lot=seq[i].lot||null;
          var same=(lot===null)||(runLot===null)||(lot===runLot);
          if(g2>gapMin || !same){ runs.push([runLot,run]); run=[]; runLot=lot; }
          else if(runLot===null && lot!==null){ runLot=lot; }
          run.push(seq[i]);
        }
        runs.push([runLot,run]);
        runs.forEach(function(r){ cur=r[1]; close(r[0]); });
        out.sort(function(a,b){ return a.t0-b.t0; });     // 시작시각 순(레인 그리기용)
      } else {
        it.forEach(function(x){ if(cur.length){ var g=(x.t-cur[cur.length-1].t)/60000; gaps.push(g);
          if(g>gapMin) close(); } cur.push(x); });
        close();
      }
      groups[ch]=out;
    });
    var bl=_med(allD.filter(isFinite)); if(!(isFinite(bl)&&bl>1e-9)) bl=1;
    if(basis!=='spread') bl=1;                             // 여유·Cpu·초과강도는 절대 스케일
    // ★ 칸 합치기는 기준에 따라 다르다(앱과 동일). 초과 강도는 '가장 심했던 LOT'(최댓값).
    //   중앙값이면 하나가 크게 넘어도 0 이 되고, 평균이면 긴 구간에서 희석된다
    //   (한 달 칸에서 이틀만 심해도 0.02 로 떨어져 '정상'처럼 읽힌다).
    var _agg=(basis==='excess'||basis==='excess2')
      ? function(a){ var m=-Infinity; for(var i=0;i<a.length;i++) if(a[i]>m) m=a[i]; return a.length?m:NaN; }
      : _med;
    var _mean=function(a){ var t=0; for(var i=0;i<a.length;i++) t+=a[i]; return a.length?t/a.length:NaN; };
    var lanes=[];
    for(var ci=1;ci<=6;ci++){
      // ★ 자릿수는 앱과 같아야 한다(server 는 round(...,6)). 여기서 3자리로 줄여 놓으면
      //   칸 값을 합칠 때 오차가 남아 같은 데이터인데 소수점이 어긋난다.
      // ★ LOT 묶음은 '시작 시각 한 점'이 아니라 구간(t0~t1)이다 — 앱과 같은 규칙.
      //   시작 시각만 보면 자정을 넘긴 LOT 이 뒷날 칸을 비운다(웨이퍼는 있는데 빈 칸).
      var ch='dsv'+ci, gs=(groups[ch]||[]).map(function(g){
        return {t:g.t0, te:(g.t1!=null?g.t1:g.t0), r:_pyround(g.disp/bl,6), n:g.n, rate:g.rate, mex:g.mex}; });
      var cells=[], nHold=0;
      for(var bi=0;bi<count;bi++){
        var s0=+bounds[bi], s1=+bounds[bi+1], span=(s1-s0)||1;
        var inb=gs.filter(function(g){ return g.t<s1 && g.te>=s0; });   // 겹치면 포함
        var sl=[];
        for(var si=0;si<slices;si++){
          var a0=s0+span*si/slices, a1=s0+span*(si+1)/slices;
          var vv=inb.filter(function(g){return g.t<a1 && g.te>=a0;}).map(function(g){return g.r;});
          sl.push(vv.length?_pyround(_agg(vv),6):null);
        }
        var xs=inb.map(function(g){return (Math.max(g.t,s0)-s0)/span;}), ys=inb.map(function(g){return g.r;});
        var rr=inb.filter(function(g){return g.rate!=null;}), mm=inb.filter(function(g){return g.mex!=null;});
        cells.push({n_groups:inb.length, ratio:(ys.length?_pyround(_agg(ys),6):null),
          ratio_mean:(ys.length?_pyround(_mean(ys),6):null),
          rate:(rr.length?_pyround(rr.reduce(function(s,g){return s+g.rate;},0)/rr.length,4):null),
          mean_ex:(mm.length?_pyround(mm.reduce(function(s,g){return s+g.mex;},0)/mm.length,4):null),
          slope:(ys.length>=3?Math.round(10000*_theilSen(xs,ys))/10000:0), slices:sl});
      }
      if(hold){ var last=null;                              // 옵션: 빈 구간을 직전 색으로 이어 칠함
        for(var ci2=0;ci2<cells.length;ci2++){ var s2=cells[ci2].slices;
          for(var k2=0;k2<s2.length;k2++){ if(s2[k2]==null){ if(last!=null){ s2[k2]=last; nHold++; } } else last=s2[k2]; } } }
      lanes.push({chamber:ch,cells:cells,n_groups:gs.length,n_hold:nHold});
    }
    var gs2=gaps.slice().sort(function(a,b){return a-b;});
    // ★ 앱과 같은 통계 키 — 화면이 'LOT 미상 N장' 안내와 'LOT N개' 표기에 쓴다.
    var _lotSet={}; Object.keys(perCh).forEach(function(c){ perCh[c].forEach(function(x){
      if(x.lot) _lotSet[c+'|'+x.lot]=1; }); });
    var stat={n_no_score:nNoScore,n_groups:allD.length,n_small:nSmall,n_wafers:ws.length,
      group_by:groupBy, include_first:inclFirst, n_first_excluded:nFirstEx, excess_base:Math.round(ebase*1000)/1000,
      n_lot_known:nLotKnown, n_lots:Object.keys(_lotSet).length,
      gap_median:gs2.length?Math.round(10*_med(gs2))/10:null,
      gap_p90:gs2.length?Math.round(10*_pct(gs2,90))/10:null};
    if(!stat.n_groups&&gs2.length) stat.suggest_gap=Math.max(gapMin,Math.round(_pct(gs2,90)*1.5));
    return {lanes:lanes,baseline:Math.round(10000*bl)/10000,stat:stat,basis:basis,metric:metric,
      group_by:groupBy, include_first:inclFirst, excess_base:Math.round(ebase*1000)/1000,
      gap_min:gapMin,min_n:minN,slices:slices,hold:(hold?1:0),bounds:bounds.map(iso),scale:scale};
  }


  // ── 표시 판정 규칙(내보낸 화면에서도 조절 가능) ───────────────────────────
  //   모델이 필요 없다: 등급/판정은 이미 계산된 '임계 대비 점수'를 구간으로 나누는 산술이다.
  var RULE=JSON.parse(JSON.stringify(CFG.rule||{mode:'raw',bands:{warn:1.05,crit:1.2},thr_scale:{},thr_abs:{}}));
  function __lookup(tbl,rc){ if(!tbl) return null;
    if(tbl[rc]!=null) return +tbl[rc];
    if(tbl['__all__']!=null) return +tbl['__all__']; return null; }
  function __bandOf(r,b){ if(!(r>=1)) return 'ok';
    if(r<(+b.warn)) return 'caution'; if(r<(+b.crit)) return 'warn'; return 'crit'; }
  function __applyRule(){                                  // 전 웨이퍼의 판정·등급·표시점수 재계산
    var b=RULE.bands||{warn:1.05,crit:1.2}, per=(RULE.mode==='per_tp');
    WAFERS.forEach(function(w){
      var eff;
      if(per){ var thr=__lookup(RULE.thr_abs,w.recipe);
        if(!(thr>0)||w.n_tp==null){ w.band='na'; w.verdict=null; return; }
        eff=(w.score_tp!=null?+w.score_tp:+w.score)/thr;
      } else {
        var base=(w.ratio_base!=null)?+w.ratio_base:((w.ratio_x!=null)?+w.ratio_x:+w.ratio);
        var sc=__lookup(RULE.thr_scale,w.recipe); if(!(sc>0)) sc=1;
        eff=base/sc;
      }
      w.ratio_x=Math.round(eff*1e6)/1e6;
      w.ratio=Math.floor(eff*100)/100;                     // 표시값(2자리 버림 — 앱과 동일)
      w.verdict=(eff>=1)?'NG':'OK';
      w.band=__bandOf(eff,b);
    });
  }
  function __ruleApply(body){
    if(body.reset){ RULE=JSON.parse(JSON.stringify(CFG.rule||{})); __applyRule(); return; }
    if(body.mode==='per_tp'||body.mode==='raw'){
      if(body.mode==='per_tp'&&!(RULE.thr_abs&&Object.keys(RULE.thr_abs).length)){
        __demoToast('타임포인트 정규화는 내보낸 화면에서 임계 정보가 없어 전환할 수 없습니다'); return; }
      RULE.mode=body.mode;
    }
    var key=body.recipe?String(body.recipe):'__all__';
    if(body.thr_scale!=null){ RULE.thr_scale=RULE.thr_scale||{}; RULE.thr_scale[key]=+body.thr_scale; }
    if(body.thr_abs!=null){ RULE.thr_abs=RULE.thr_abs||{}; RULE.thr_abs[key]=+body.thr_abs; }
    if(body.warn!=null||body.crit!=null){ var b=Object.assign({},RULE.bands);
      if(body.warn!=null) b.warn=+body.warn; if(body.crit!=null) b.crit=+body.crit; RULE.bands=b; }
    __applyRule();
  }
  function __ruleAuto(url){                                // 자동 계산도 스냅샷만으로 가능
    var rc=q(url,'recipe','');
    var ws=WAFERS.filter(function(w){ return !rc||w.recipe===rc; });
    var oks=ws.filter(function(w){ return w.verdict==='OK'; });
    var vals=oks.map(function(w){ return (w.ratio_x!=null)?+w.ratio_x:+w.ratio; })
      .filter(function(v){ return isFinite(v)&&v>0; });
    var out={n_ok:oks.length,n_all:ws.length,recipe:rc||'전체',per_tp:null,current:RULE};
    if(vals.length<8){ out.bands=null; out.note='OK 표본이 부족해(8장 미만) 자동 산출 불가'; return out; }
    var l=vals.map(Math.log), med=_med(l);
    var mad=_med(l.map(function(v){return Math.abs(v-med);}))*1.4826;
    if(!(mad>1e-9)) mad=_std(l);
    if(!(mad>1e-9)){ out.bands=null; out.note='분포가 너무 좁아 자동 산출 불가'; return out; }
    out.sigma_log=Math.round(mad*1e4)/1e4;
    out.bands={warn:Math.round(Math.min(Math.max(Math.exp(mad),1.05),20)*1000)/1000,
               crit:Math.round(Math.min(Math.max(Math.exp(2*mad),1.1),40)*1000)/1000};
    return out;
  }
  __applyRule();                                           // 초기 1회(스냅샷 등급과 동일해야 정상)

  var OFF=CFG.offline_scores||{models:{},wafers:{}};
  function selectedScore(key,selection,sensorBasis){
    var w=OFF.wafers[key],m=w&&OFF.models[w.model];
    var fail=function(note){return {available:false,sensors:[],note:note,applied_selection:selection};};
    if(!w||!m)return fail((w&&w.missing)||'내보낸 점수 자료 없음');
    var steps=selection&&selection.steps!=null?selection.steps:m.steps,sensors=selection&&selection.sensors!=null?selection.sensors:m.sensors;
    if(!steps||!steps.length||!sensors||!sensors.length||steps.some(function(k){return k===0||m.steps.indexOf(k)<0;})||sensors.some(function(n){return m.sensors.indexOf(n)<0;}))return fail('이 챔버 기준에 없는 스텝·센서이거나 빈 선택입니다.');
    steps=Array.from(new Set(steps));sensors=Array.from(new Set(sensors));
    function stats(indices){var value=0,mean=0,variance=0;
      indices.forEach(function(i){value+=w.matrix[i];mean+=m.mean[i];});
      m.factor.forEach(function(row){var sum=0;indices.forEach(function(i){sum+=row[i];});variance+=sum*sum;});
      var std=Math.sqrt(Math.max(0,variance)),limit=mean+3.5*std;
      return {value:value,mean:mean,std:std,limit:limit,ratio:limit>1e-12?value/limit:null};
    }
    if(sensorBasis==='train_ok_max'&&(!m.train_scores||!m.train_scores.length))return fail('정상 학습 최댓값 자료가 없습니다. HTML을 다시 내보내세요.');
    var all=[],rows=[];
    m.sensors.forEach(function(name,j){if(sensors.indexOf(name)<0)return;
      var indices=[];m.steps.forEach(function(k,i){if(steps.indexOf(k)>=0)indices.push(i*m.sensors.length+j);});all=all.concat(indices);
      var r=stats(indices);
      if(sensorBasis==='train_ok_max'){
        r.limit=0;m.train_scores.forEach(function(row){var sum=0;indices.forEach(function(i){sum+=row[i];});r.limit=Math.max(r.limit,sum);});
        r.ratio=r.limit>1e-12?r.value/r.limit:null;
      }
      rows.push({sensor:name,index:w.columns.indexOf(name),selected_score:r.value,train_mean:r.mean,train_std:r.std,threshold_raw:r.limit,ratio:r.ratio,score:r.ratio==null?null:Math.min(2,r.ratio),verdict:r.ratio==null?'NA':r.value>=r.limit?'NG':'OK'});
    });
    var total=stats(all),beforeScore=total.value,beforeRatio=total.ratio;
    var overridden=total.ratio!==null&&typeof w.override_ratio==='number'&&isFinite(w.override_ratio)&&w.override_ratio>0;
    if(overridden){total.value=total.limit*w.override_ratio;total.ratio=w.override_ratio;}
    return {score_overridden:overridden,matched_pattern:overridden?w.matched_pattern:null,override_ratio:overridden?w.override_ratio:null,selected_score_before_override:beforeScore,selected_ratio_before_override:beforeRatio,available:true,sensors:rows,process_steps:m.steps.filter(function(k){return steps.indexOf(k)>=0;}),available_steps:m.steps,available_sensors:m.sensors,
      sigma:3.5,threshold:1,train_count:m.train_count,additive_exact_match_expected:m.additive_exact_match_expected,
      total_score:total.value,total_threshold:total.limit,total_ratio:total.ratio,applied_selection:selection};
  }
  function selectedRow(w,selection){var s=selectedScore(w.date+'/'+w.wid,selection),scores={};
    (s.sensors||[]).forEach(function(r){if(r.ratio!=null)scores[r.sensor]=r.ratio;});
    return {wid:w.wid,scores:scores,score:s.total_score,threshold:s.total_threshold,ratio:s.available?s.total_ratio:null,missing:s.available?(s.total_ratio==null?'선택 임계가 0입니다.':null):s.note};
  }

  var realFetch=window.fetch;
  window.fetch=function(url,opts){
    url=String(url); var isPost=opts&&(opts.method||'').toUpperCase()==='POST';
    try{
      if(url.indexOf('/api/status')===0) return R(CFG.status||{model_ready:true});
      // ★ 표시 판정 규칙(밴드·모드·임계): 내보낼 때의 설정을 그대로 돌려준다.
      //   (이게 없으면 화면이 기본값 2/3 으로 떨어져 '내가 보던 화면'과 달라진다)
      if(url.indexOf('/api/display-rule/auto')===0) return R(__ruleAuto(url));
      if(url.indexOf('/api/display-rule')===0){
        if(isPost){                                       // 밴드·임계는 계산만으로 바뀌므로 데모에서도 허용
          var body={}; try{ body=JSON.parse((opts&&opts.body)||'{}'); }catch(e){}
          __ruleApply(body);
        }
        return R(Object.assign({ntp_known:0,model_score_norm:'none'},RULE));
      }
      // ★ 산포 추세: 서버와 같은 계산을 스냅샷(WAFERS)으로 그대로 재현
      if(url.indexOf('/api/dispersion')===0) return R(__dispersion(url));
      if(url.indexOf('/api/progress')===0)
        return R({active:false,done:0,total:0,phase:'',folder:'',bg:{active:false,done:0,total:0,phase:'대기',folder:''}});
      if(url.split('?')[0]==='/api/overview/selection'){
        var req=JSON.parse(opts.body),query={range:req.rng,scale:req.scale,recipe:req.recipe,start:req.start,end:req.end,chambers:req.chambers,recipes:req.recipes,month_start_day:req.month_start_day};
        var queryString=Object.keys(query).filter(function(k){return query[k]!=null;}).map(function(k){return k+'='+encodeURIComponent(query[k]);}).join('&');
        return window.fetch('/api/overview?'+queryString).then(function(r){return r.json();}).then(function(d){
          var selectedRows=[];
          var bounds=d.bounds.map(_T),chs=req.chambers?req.chambers.split(','):null,rcs=req.recipes?req.recipes.split(','):null;
          d.lanes.forEach(function(lane){lane.cells=lane.cells.map(function(){return {total:0,bad:0,na:0,caution:0,warn:0,crit:0,defectPct:0};});
            WAFERS.forEach(function(w){if(String(w.chamber).toLowerCase()!==lane.chamber||(chs&&chs.indexOf(w.chamber)<0)||(rcs&&rcs.indexOf(w.recipe)<0)||(req.recipe&&req.recipe!=='all'&&req.recipe!==w.recipe))return;
              var t=_T(w.ts),idx=-1;for(var i=0;i<bounds.length-1;i++){if(t>=bounds[i]&&t<bounds[i+1]){idx=i;break;}}if(idx<0)return;
              var c=lane.cells[idx],row=selectedRow({date:w.date,wid:w.wafer_id},req),v=row.ratio;
              if(req.dispersion)selectedRows.push(Object.assign({},w,{ratio_x:v,ratio:v}));
              if(v==null){c.na++;return;}c.total++;if(v>=1)c.bad++;if(v>=RULE.bands.crit)c.crit++;else if(v>=RULE.bands.warn)c.warn++;else if(v>=1)c.caution++;
            });
            var total=0,bad=0;lane.cells.forEach(function(c){c.defectPct=c.total?Math.round(1000*c.bad/c.total)/10:0;total+=c.total;bad+=c.bad;});lane.avgDefect=total?Math.round(1000*bad/total)/10:0;
          });d.selection={steps:req.steps,sensors:req.sensors,sigma:3.5};d.model_id=CFG.model_id;
          if(req.dispersion){
            d.dispersion=__dispersion('/api/dispersion?'+queryString+'&basis=excess&include_first='+(req.include_first===false?'0':'1')+'&hold='+(req.hold?'1':'0'),selectedRows,d.bounds);
            d.dispersion.selection=d.selection;
          }
          return R(d);
        });
      }
      if(url.indexOf('/api/overview')===0){
        var rng=q(url,'range','7d'), scale=q(url,'scale','day'), rcp=q(url,'recipe','all');
        // ★ 체크박스 상태(chambers/recipes)를 앱과 똑같이 반영 — 없으면 유형을 해제해도
        //   히트맵만 그대로 남아 산점도와 어긋난다.
        var _ovCh=q(url,'chambers',''), _ovRc=q(url,'recipes','');
        var _ovChSet=(_ovCh&&_ovCh!=='__NONE__')?_ovCh.split(','):null;
        var _ovRcSet=(_ovRc&&_ovRc!=='__NONE__')?_ovRc.split(','):null;
        var _ovNone=(_ovCh==='__NONE__'||_ovRc==='__NONE__');
        var t1=NOW,t0;
        if(rng==='custom'){t0=new Date(q(url,'start',iso(new Date(NOW-7*86400e3))));t1=new Date(+new Date(q(url,'end',iso(NOW)))+1000);}
        // ★ '전체 기간' 을 이 갈래가 모르고 있었다 → 프리셋 표에서 못 찾아 168시간(7일)로
        //   떨어졌다. 내보낸 HTML 에서 전체 기간을 골라도 최근 7일만 보이고 그 앞은
        //   통째로 사라진다(앱은 데이터가 있는 전 기간을 보여준다). 앱과 같은 규칙으로.
        else if(rng==='all'){
          var _at=WAFERS.map(function(w){return _T(w.ts).getTime();}).filter(isFinite);
          t0=new Date(_at.length?Math.min.apply(null,_at):(NOW-7*86400e3));
          t1=new Date(_at.length?(Math.max.apply(null,_at)+86400e3):NOW); }
        else {var h={'1h':1,'6h':6,'12h':12,'24h':24,'7d':168,'30d':720}[rng]||168;t0=new Date(NOW-h*3600e3);}
        if(['day','week','month'].indexOf(scale)<0){var td=(t1-t0)/86400e3; scale=td<=31?'day':(td<=183?'week':'month');}
        var bounds=[];
        if(scale==='month')bounds=monthlyBounds(t0,t1,+q(url,'month_start_day','1'),rng==='custom');
        else { var stepD=(scale==='week'?7:1); var base0=new Date(t0); base0.setHours(0,0,0,0);
          if(scale==='week'){ var dow=(base0.getDay()+6)%7; base0=new Date(base0-dow*86400e3); }
          var cur2=base0; while(cur2<t1){ bounds.push(new Date(cur2)); cur2=new Date(+cur2+stepD*86400e3); } bounds.push(cur2); }
        if(rng==='custom')bounds=[t0].concat(bounds.filter(function(b){return b>t0&&b<t1;}),[t1]);
        if(bounds.length>401)bounds=bounds.slice(-401);
        var count=bounds.length-1; function p(x){return String(x).padStart(2,'0');}
        var buckets=bounds.slice(0,count).map(function(bs,i){ if(scale==='month'&&+q(url,'month_start_day','1')!==1){var last=new Date(+bounds[i+1]-1000);return (bs.getMonth()+1)+'/'+bs.getDate()+'~'+(last.getMonth()+1)+'/'+last.getDate();}return scale==='month'?(bs.getFullYear()+'-'+p(bs.getMonth()+1)):(p(bs.getMonth()+1)+'-'+p(bs.getDate())); });
        var ws=_ovNone?[]:WAFERS.filter(function(w){ return inWin(w,bounds[0],bounds[bounds.length-1])
          &&(rcp==='all'||w.recipe===rcp)
          &&(_ovChSet?_ovChSet.indexOf(w.chamber)>=0:true)
          &&(_ovRcSet?_ovRcSet.indexOf(w.recipe)>=0:true); });
        var lanes=[],chs=[];
        function _bd(w){ return w.band || (w.verdict==='NG'?'caution':'ok'); }   // 스냅샷의 등급 사용
        // server.py _chamber_level 과 동일 — 비율 기준(기본 5% / 2%), 표본 20매 미만이면
        // 건수(2장 이상)로 판단, 1장이면 경고까지, 불량 0 이면 임계 근접도로 예열 표시.
        function _chLevel(total,ng,maxRatio){
          if(!(total>0)) return 'none';
          var rate=100*ng/total;
          if(total>=20){ if(rate>=5) return 'defect'; if(rate>=2) return 'warn3'; }
          else { if(ng>=2) return 'defect'; }
          if(ng>=1) return 'warn3';
          var r=+maxRatio||0;
          if(r>=0.90) return 'warn2';
          if(r>=0.80) return 'warn1';
          return 'normal';
        }
        function _cnt(g,k){ return g.filter(function(w){return _bd(w)===k;}).length; }
        // 챔버 문자열은 소문자로 맞춰 비교한다(앱 _chamber_key 와 동일) —
        // 'DSV1' 처럼 대문자로 적힌 기록이 격자에서 통째로 빠지던 문제.
        function _chk(w){ return String(w.chamber||'').trim().toLowerCase(); }
        for(var ci=1;ci<=6;ci++){var ch='dsv'+ci,cells=[],ct=0,cb=0,cw=0,cc=0;
          for(var i=0;i<count;i++){var s0=bounds[i],s1=bounds[i+1];
            var g=ws.filter(function(w){return _chk(w)===ch&&inWin(w,s0,s1);});
            var bad=g.filter(function(w){return w.verdict==='NG';}).length;
            cells.push({total:g.length,bad:bad,caution:_cnt(g,'caution'),warn:_cnt(g,'warn'),crit:_cnt(g,'crit'),
              defectPct:g.length?Math.round(1000*bad/g.length)/10:0});
            ct+=g.length;cb+=bad;cw+=_cnt(g,'warn');cc+=_cnt(g,'crit');}
          lanes.push({chamber:ch,avgDefect:ct?Math.round(1000*cb/ct)/10:0,cells:cells});
          // ★ 램프는 앱(server.py _chamber_level)과 '같은 규칙'이어야 한다.
          //   예전에는 여기만 옛 규칙(심각 1장이면 무조건 빨강)이 남아 있어서,
          //   같은 데이터인데 앱은 정상인 챔버가 내보낸 HTML 에서는 빨강으로 떴다.
          var _chw=ws.filter(function(w){return _chk(w)===ch;});
          var _mx=0; _chw.forEach(function(w){ var r=(w.ratio_x!=null?+w.ratio_x:+w.ratio);
            if(isFinite(r)&&r>_mx) _mx=r; });
          chs.push({chamber:ch,level:_chLevel(ct,cc,_mx),ng:cb,total:ct,max_ratio:Math.round(_mx*1000)/1000,
            caution:_cnt(_chw,'caution'),warn:cw,crit:cc});}
        // ── ★ '데이터 기준 최근 1시간' 블록 — 앱(api_overview.hour)과 동일 규칙 ──
        //   닻 = (체크박스 필터 적용된) 전체 기록의 최신 ts. 조회기간과 무관.
        var _hrows=_ovNone?[]:WAFERS.filter(function(w){
          return (_ovChSet?_ovChSet.indexOf(w.chamber)>=0:true)
              &&(_ovRcSet?_ovRcSet.indexOf(w.recipe)>=0:true); });
        var hourBlk={total:0,good:0,bad:0,caution:0,warn:0,crit:0,na:0,unassigned:0,
                     goodRate:0,badRate:0,warnRate:0,critRate:0,anchor:null};
        var hourChs=[]; var _hc={};
        ['dsv1','dsv2','dsv3','dsv4','dsv5','dsv6'].forEach(function(c){
          _hc[c]={chamber:c,total:0,ng:0,max_ratio:0,caution:0,warn:0,crit:0,level:0}; });
        if(true){
          // ★ 벽시계(NOW) 기준 — 앱과 동일(과거 데이터는 최근 1시간에 안 잡힘)
          var _anc=NOW.getTime(), _h0=_anc-3600e3;
          var _ad=new Date(_anc);
          hourBlk.anchor=_ad.getFullYear()+'-'+p(_ad.getMonth()+1)+'-'+p(_ad.getDate())
            +'T'+p(_ad.getHours())+':'+p(_ad.getMinutes())+':'+p(_ad.getSeconds());
          var _hws=_hrows.filter(function(w){
            var t=_T(w.ts).getTime();
            return isFinite(t)&&t>=_h0&&t<=_anc&&(rcp==='all'||w.recipe===rcp); });
          _hws.forEach(function(w){
            if(_bd(w)==='na'){ hourBlk.na++; return; }
            var ng=(w.verdict==='NG'), bd=_bd(w);
            hourBlk.total++; if(ng) hourBlk.bad++; else hourBlk.good++;
            if(bd==='caution')hourBlk.caution++; if(bd==='warn')hourBlk.warn++; if(bd==='crit')hourBlk.crit++;
            var c=_chk(w);
            if(!_hc[c]){ hourBlk.unassigned++; return; }
            var b2=_hc[c]; b2.total++; if(ng)b2.ng++;
            if(bd==='caution')b2.caution++; if(bd==='warn')b2.warn++; if(bd==='crit')b2.crit++;
            var r=(w.ratio_x!=null?+w.ratio_x:+w.ratio);
            if(isFinite(r)&&r>b2.max_ratio) b2.max_ratio=r;
          });
          if(hourBlk.total){
            hourBlk.goodRate=Math.round(1000*hourBlk.good/hourBlk.total)/10;
            hourBlk.badRate=Math.round(1000*hourBlk.bad/hourBlk.total)/10;
            hourBlk.warnRate=Math.round(1000*hourBlk.warn/hourBlk.total)/10;
            hourBlk.critRate=Math.round(1000*hourBlk.crit/hourBlk.total)/10;
          }
        }
        ['dsv1','dsv2','dsv3','dsv4','dsv5','dsv6'].forEach(function(c){
          var b2=_hc[c]; b2.level=_chLevel(b2.total,b2.ng,b2.max_ratio);
          b2.max_ratio=Math.round(b2.max_ratio*1000)/1000; hourChs.push(b2); });
        var tot=ws.length,bad=ws.filter(function(w){return w.verdict==='NG';}).length;
        var nca=_cnt(ws,'caution'),nwa=_cnt(ws,'warn'),ncr=_cnt(ws,'crit');
        return R({range:rng,scale:scale,recipe:rcp,base:iso(bounds[0]),bounds:bounds.map(iso),count:count,now:iso(NOW),
          summary:{total:tot,good:tot-bad,bad:bad,na:0,caution:nca,warn:nwa,crit:ncr,
            thr_from_history:0,                            // 스냅샷은 현재 임계로 이미 판정됨
            // 격자(dsv1~6)에 자리가 없어 칸에 못 들어간 수 — 앱과 같은 기준으로 센다
            unassigned:ws.filter(function(w){
              var c=String(w.chamber||'').trim().toLowerCase();
              return ['dsv1','dsv2','dsv3','dsv4','dsv5','dsv6'].indexOf(c)<0; }).length,
            goodRate:tot?Math.round(1000*(tot-bad)/tot)/10:0,badRate:tot?Math.round(1000*bad/tot)/10:0,
            warnRate:tot?Math.round(1000*nwa/tot)/10:0,critRate:tot?Math.round(1000*ncr/tot)/10:0},
          bands:RULE.bands,
          chambers:chs,
          hour:hourBlk, hour_chambers:hourChs,
          buckets:buckets,lanes:lanes,recipes:RECIPES});
      }
      if(url.indexOf('/api/wafer-list')===0){
        var chsel=q(url,'chamber','all'),rcp=q(url,'recipe','all'),page=+q(url,'page','1');
        var defo=q(url,'defects','0')==='1'||q(url,'defects_only','false')==='true';
        var chsv=q(url,'chambers',''),rcsv=q(url,'recipes',''),qq=q(url,'q','').toLowerCase();
        if(chsv==='__NONE__'||rcsv==='__NONE__') return R({wafers:[],total:0,ng_total:0,page:1,pages:1,page_size:30,defects_only:false,recipes:RECIPES,span:''});
        var chset=chsv?chsv.split(','):null, rcset=rcsv?rcsv.split(','):null;
        var rng=q(url,'range','7d'),t1=NOW,t0;
        if(rng==='custom'){t0=new Date(q(url,'start',iso(new Date(NOW-7*86400e3))));t1=new Date(q(url,'end',iso(NOW)));}
        else if(rng==='all'){var times=WAFERS.map(function(w){return _T(w.ts).getTime();}).filter(isFinite);t0=new Date(times.length?Math.min.apply(null,times):NOW);t1=new Date(times.length?Math.max.apply(null,times)+1:NOW);}
        else {var h={'1h':1,'6h':6,'12h':12,'24h':24,'7d':168,'30d':720}[rng]||168;t0=new Date(NOW-h*3600e3);}
        var ws=WAFERS.filter(function(w){return inWin(w,t0,t1)
          &&(chsel==='all'||w.chamber===chsel)&&(chset?chset.indexOf(w.chamber)>=0:true)
          &&(rcset?rcset.indexOf(w.recipe)>=0:(rcp==='all'||w.recipe===rcp))
          &&(!defo||w.verdict==='NG')&&(!qq||(w.wafer_id||'').toLowerCase().indexOf(qq)>=0);});
        var ps=30,pages=Math.max(1,Math.ceil(ws.length/ps));
        return R({bands:RULE.bands,mode:RULE.mode,
          wafers:ws.slice((page-1)*ps,page*ps),total:ws.length,
          ng_total:ws.filter(function(w){return w.verdict==='NG';}).length,
          page:Math.min(page,pages),pages:pages,page_size:ps,defects_only:defo,recipes:RECIPES,
          span:(ws.length?{from:ws[0].ts,to:ws[ws.length-1].ts}:'')});
      }
      if(url.split('?')[0]==='/api/scatter/differences'){
        var request=JSON.parse((opts&&opts.body)||'{}');
        return R({model_id:CFG.model_id,wafers:(request.wafers||[]).map(function(w){
          return Object.assign({wid:w.wid,date:w.date,series:[],missing:'보관된 센서 차이 자료 없음'},
            (CFG.sensor_differences||{})[w.date+'/'+w.wid]||{});
        })});
      }
      if(url.split('?')[0]==='/api/scatter/step-times'){
        var req=JSON.parse((opts&&opts.body)||'{}');return R({model_id:CFG.model_id,wafers:(req.wafers||[]).map(function(w){return Object.assign({wid:w.wid,date:w.date,step_timing:{steps:{},missing:'내보낸 시간 자료 없음'},peaks:{}},(CFG.step_times||{})[w.date+'/'+w.wid]||{});})});
      }
      if(url.split('?')[0]==='/api/scatter/sensors-max'){
        var request=JSON.parse((opts&&opts.body)||'{}');
        return R({model_id:CFG.model_id,sensor_basis:'train_ok_max',wafers:(request.wafers||[]).map(function(w){
          var result=selectedScore(w.date+'/'+w.wid,request,'train_ok_max'),scores={},missing=[];
          (result.sensors||[]).forEach(function(sensor){if(sensor.ratio!=null)scores[sensor.sensor]=sensor.ratio;else missing.push(sensor.sensor);});
          return {wid:w.wid,scores:scores,missing:result.available?(missing.length?'정상 학습 최댓값이 0인 센서: '+missing.join(', '):null):result.note};
        })});
      }
      if(url.split('?')[0]==='/api/scatter/sensors'){
        var request=JSON.parse((opts&&opts.body)||'{}');
        return R({model_id:CFG.model_id,wafers:(request.wafers||[]).map(function(w){
          var key=w.date+'/'+w.wid, saved=(CFG.scatter_scores||{})[key];
          if(saved)return saved;
          var detail=DETAIL[key], scores={};
          ((detail&&detail.sensorsAll&&detail.sensorsAll.sensors)||[]).forEach(function(sensor){
            if(sensor.ratio!=null&&isFinite(Number(sensor.ratio)))scores[sensor.sensor]=Number(sensor.ratio);
          });
          return {wid:w.wid,scores:scores,missing:Object.keys(scores).length?null:'보관 데이터 없음'};
        })});
      }
      if(url.split('?')[0]==='/api/scatter/steps'){
        if(isPost){var req=JSON.parse(opts.body);return R({model_id:CFG.model_id,steps:req.steps,sigma:3.5,wafers:req.wafers.map(function(w){return selectedRow(w,req);})});}
        var steps=[],sensors=[];Object.keys(OFF.models).forEach(function(k){steps=steps.concat(OFF.models[k].steps);sensors=sensors.concat(OFF.models[k].sensors);});
        return R({model_id:CFG.model_id,steps:Array.from(new Set(steps)).sort(function(a,b){return a-b;}),sensors:Array.from(new Set(sensors)),issues:steps.length?[]:['학습 앱점수 자료 없이 내보낸 HTML입니다.']});
      }
      if(url.indexOf('/api/scatter')===0){
        var ch=q(url,'chamber','all'),rcp=q(url,'recipe','all');
        var chsv=q(url,'chambers',''),rcsv=q(url,'recipes','');
        if(chsv==='__NONE__'||rcsv==='__NONE__') return R({points:[],trend:{slope:0,dir:'flat',y0:null,y1:null}});
        var chset=chsv?chsv.split(','):null, rcset=rcsv?rcsv.split(','):null;
        var s2=q(url,'start',''),e2=q(url,'end','');
        var ws=WAFERS.filter(function(w){return (chset?chset.indexOf(w.chamber)>=0:(ch==='all'||w.chamber===ch))
          &&(rcset?rcset.indexOf(w.recipe)>=0:(rcp==='all'||w.recipe===rcp));});
        if(s2&&e2){var t0=new Date(s2),t1=new Date(e2);ws=ws.filter(function(w){return inWin(w,t0,t1);});}
        ws=ws.slice().sort(function(a,b){return a.ts<b.ts?-1:1;});
        // ★ lot·lot_first 도 그대로 넘긴다 — 산점도가 'LOT 첫 장'을 흐리게 그리는 데 쓴다.
        //   (예전엔 빠져 있어 내보낸 HTML 에서만 첫 장 표시가 안 됐다)
        var pts=ws.map(function(w){return {ts:w.ts,wafer_id:w.wafer_id,chamber:w.chamber,recipe:w.recipe,mean_recipe:w.mean_recipe,
          ratio:w.ratio,score:w.score,threshold:(w.threshold!=null?w.threshold:1.0),verdict:w.verdict,
          band:w.band,n_tp:w.n_tp,lot:w.lot,lot_first:!!w.lot_first};});
        var n=pts.length,tr={slope:0,dir:'flat',y0:null,y1:null};
        if(n>=2){var sx=0,sy=0,sxx=0,sxy=0;for(var i=0;i<n;i++){sx+=i;sy+=pts[i].ratio;sxx+=i*i;sxy+=i*pts[i].ratio;}
          var den=n*sxx-sx*sx||1,aa=(n*sxy-sx*sy)/den,bb=(sy-aa*sx)/n;
          tr={slope:aa,dir:Math.abs(aa)<1e-4?'flat':(aa>0?'up':'down'),y0:bb,y1:aa*(n-1)+bb};}
        return R({model_id:CFG.model_id,points:pts,trend:tr,n:n,n_na:0,n_est:0,na_reason:'',
          rule:RULE});
      }
      if(url.indexOf('/api/history/range')===0){var s=q(url,'start',dstr(NOW)),e=q(url,'end',dstr(NOW));
        return R({wafers:WAFERS.filter(function(w){return w.date>=s&&w.date<=e;})});}
      if(url.indexOf('/api/calendar/day/')===0){var d=url.split('/api/calendar/day/')[1].slice(0,10);
        // ★ has_archive: 상세 포함 웨이퍼(DETAIL)만 행 펼치기 가능 — 앱과 같은 동작
        return R({date:d,wafers:WAFERS.filter(function(w){return w.date===d;})
          .map(function(w){return Object.assign({},w,{has_archive:!!DETAIL[d+'/'+w.wafer_id]});})});}
      if(url.indexOf('/api/calendar')===0){var m=q(url,'month',dstr(NOW).slice(0,7));var by={};
        WAFERS.forEach(function(w){if((w.date||'').indexOf(m)===0){(by[w.date]=by[w.date]||[]).push(w);}});
        var days=Object.keys(by).sort().map(function(d){var g=by[d],ng=g.filter(function(w){return w.verdict==='NG';}).length;
          var avg=g.reduce(function(s,w){return s+w.ratio;},0)/g.length;
          return {date:d,count:g.length,ng:ng,avg:Math.round(avg*1000)/1000,max:Math.max.apply(null,g.map(function(w){return w.ratio;}))};});
        return R({month:m,days:days});}
      var am=url.match(/\/api\/archive\/(\d{4}-\d{2}-\d{2})\/([^\/?]+)(\/[^?]*)?(\?.*)?$/);
      if(am){
        var date=am[1],wid=decodeURIComponent(am[2]),sub=am[3]||'',key=date+'/'+wid;
        if(sub.indexOf('/sensors-app-score')===0 && OFF.wafers[key]){
          var selection=isPost?JSON.parse(opts.body):JSON.parse(q(url,'selection','null'));
          return R(selectedScore(key,selection));
        }
        var det=DETAIL[key];
        if(!det) return R({__demo_missing__:true});     // 미포함 웨이퍼(데모 범위 밖)
        // ★ batch 우선 매칭 — 앱은 /zone-contrib-batch 를 쓴다. 이걸 처리 못 해
        //   데모에서는 센서 음영 색이 영영 오지 않았다(민감도를 바꿔도 무반응으로 보임).
        if(sub.indexOf('/zone-contrib-batch')===0){
          var _wins=[];
          try{ _wins=(JSON.parse((opts&&opts.body)||'{}').windows)||[]; }catch(e){ _wins=[]; }
          // 서버와 같은 형식: zones[].sensors[{index,sensor,share(%)}] + total
          return R({zones:_wins.map(function(w){
            var r0=zoneContrib(det,+w[0],+w[1]);
            return {sensors:(r0.sensors||[]), total:r0.total||0};
          })});}
        if(sub.indexOf('/zone-contrib')===0){var a=+q(url,'a','0'),b=+q(url,'b','1');return R(zoneContrib(det,a,b));}
        if(sub.indexOf('/sensors-app-score')===0 && ((opts && opts.method==='POST') || url.indexOf('selection=')>=0)) return R({available:false,sensors:[],note:'선택 점수 재계산은 실행 중인 앱에서 사용하세요.'});
        if(sub.indexOf('/sensors-app-score')===0) return R(det.appSensorScores||{available:false,sensors:[],note:'앱점수 자료를 포함하여 데모 HTML을 다시 생성하세요.'});
        if(sub.indexOf('/sensors-all')===0) return R(det.sensorsAll||{sensors:[],threshold:1.0});
        if(sub.indexOf('/sensor-series/')===0){var j=sub.split('/sensor-series/')[1];
          return R((det.sensorSeries&&det.sensorSeries[j])||{});}
        if(sub.indexOf('/wafer-series')===0) return R(det.wseries||{});
        return R(det.meta||{wafer_id:wid,date:date});
      }
      if(url.indexOf('/api/ui-config')===0){
        if(isPost){try{var ub=JSON.parse(opts.body);if(ub.overlay_sensitivity!=null)UISENS=Math.min(1,Math.max(0,+ub.overlay_sensitivity));}catch(e){}}
        return R({overlay_sensitivity:UISENS});}
      if(url.indexOf('/api/archive-retention')===0){
        if(isPost){try{var rb=JSON.parse(opts.body),rd=Number(rb.days);
          RETEN=rb.unlimited===true?0:((Number.isInteger(rd)&&rd>0&&rd<=60)?rd:60);}catch(e){}}
        return R({days:RETEN,unlimited:RETEN===0});}
      if(url.indexOf('/api/chamber-paths')===0) return R(CFG.chamber_paths||{paths:{},exists:{}});
      if(url.indexOf('/api/models')===0) return R(CFG.models||{models:[],current_dir:'',default_dir:''});
      if(url.indexOf('/api/sensors')===0) return R({input_sensors:SENSORS,recon_sensors:CFG.recon_sensors||SENSORS,io:CFG.io||[]});
      // ── 데모에 없는 '설정·작업' API ──
      //   서버가 있어야 뜻이 있는 것들이다(장비 설정 읽기·폴더 탐색·LOT 채우기).
      //   catch-all 로 {} 만 돌려주면 화면이 '고장난 것처럼' 보인다(폴더 창이 텅 비고,
      //   설정 화면이 빈칸이 된다). 데모임을 분명히 알리거나, 화면이 깨지지 않을
      //   최소 형태를 돌려준다.
      if(url.indexOf('/api/list-dirs')===0)
        return R({path:'',parent:null,dirs:[],drives:[],error:'데모(읽기전용)에서는 폴더를 탐색할 수 없습니다'});
      if(url.indexOf('/api/lot-path-count')===0) return R({count:null,demo:true});
      if(url.indexOf('/api/cpu-limit')===0){
        if(isPost){ __demoToast('데모 모드에서는 설정을 바꿀 수 없습니다'); return R({demo:true}); }
        return R({demo:true,physical:null,logical:null,saved:{reserve:0,max_cores:0,affinity:false},
                  affinity_note:'데모(읽기전용)',pool_workers:null,restart_required:false}); }
      if(url.indexOf('/api/export-html')===0){
        __demoToast('이미 내보낸 HTML 입니다 — 여기서 다시 내보낼 수는 없습니다');
        return R({demo:true}); }
      // 화면 생존 신호 — 데모(단일 HTML)에는 종료할 서버가 없다. 조용히 받아만 준다.
      if(url.indexOf('/api/alive')===0||url.indexOf('/api/bye')===0) return R({ok:true,demo:true});
      // 저장소·폴더 경로·학습 설정군 — 데모에는 대상 파일시스템이 없다.
      if(url.indexOf('/api/storage')===0||url.indexOf('/api/paths')===0
         ||url.indexOf('/api/set-inbox')===0){
        if(isPost){ __demoToast('데모 모드에서는 경로·저장소를 바꿀 수 없습니다'); return R({demo:true,ok:false}); }
        return R({demo:true,paths:{},exists:{},ok:true,note:'데모(읽기전용)'}); }
      if(url.indexOf('/api/train')===0){
        if(isPost){ __demoToast('데모 모드에서는 학습을 실행할 수 없습니다'); return R({demo:true,ok:false}); }
        return R({demo:true,running:false,params:{},best:null}); }
      if(url.indexOf('/api/rca')===0){
        __demoToast('데모 모드에서는 RCA 를 생성할 수 없습니다 (원본 .log 가 필요합니다)');
        return R({demo:true}); }
      if(url.indexOf('/api/scan-recipes')===0){
        __demoToast('데모 모드에서는 폴더를 훑을 수 없습니다'); return R({demo:true,recipes:[]}); }
      if(url.indexOf('/api/lot-paths')===0){
        if(isPost){ __demoToast('데모 모드에서는 경로를 저장할 수 없습니다'); return R({demo:true}); }
        return R({paths:[],demo:true}); }
      if(url.indexOf('/api/ort-batch')===0){
        if(isPost){ __demoToast('데모 모드에서는 설정을 바꿀 수 없습니다'); return R({demo:true}); }
        return R({batch:0,demo:true}); }
      // ── 변경 동작(읽기전용 데모): 무효화 ──
      if(url.indexOf('/api/inspect')===0||url.indexOf('/api/model/')===0
         ||url.indexOf('/api/archive/clear')===0){ __demoToast('데모 모드에서는 검사/변경이 동작하지 않습니다'); return R({demo:true,n:0,results:[]}); }
      if(url.indexOf('/api/report')===0) return R({running:false,done:0,total:0,errors:0});
      if(url.indexOf('/api/autopilot')===0) return R((CFG.status&&CFG.status.autopilot)||{enabled:false});
      return R({});
    }catch(err){ return R({}); }
  };
  function __demoToast(msg){var t=document.getElementById('__demoToast');
    if(!t){t=document.createElement('div');t.id='__demoToast';
      t.style.cssText='position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:99999;background:#1f2937;color:#fbbf24;font-size:13px;font-weight:700;padding:9px 16px;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.3)';document.body.appendChild(t);}
    t.textContent=msg;t.style.opacity='1';clearTimeout(t._h);t._h=setTimeout(function(){t.style.opacity='0';},2200);}
  document.addEventListener('DOMContentLoaded',function(){
    if(!window.SEMES_EXPORT_FEATURES.detail){var tab=document.querySelector('.gnb [data-view="detail"]');if(tab)tab.style.display='none';}
    var b=document.createElement('div');
    b.style.cssText='position:fixed;top:0;left:0;right:0;z-index:9999;background:#0f766e;color:#fff;font-size:12px;font-weight:700;text-align:center;padding:4px 0';
    b.textContent='데모(읽기전용) — 포함: 기본 화면'+(window.SEMES_EXPORT_FEATURES.scores?' · 센서/스텝 점수':'')+(window.SEMES_EXPORT_FEATURES.trends?' · 센서값 추이':'')+(window.SEMES_EXPORT_FEATURES.chamber_means?' · 챔버별 스텝 평균 차이 감지':'')+(window.SEMES_EXPORT_FEATURES.detail?' · 이상분석 상세':'');
    document.body.appendChild(b); document.body.style.paddingTop='24px';

    // ★ href 다운로드 링크(결과 CSV 등)는 fetch mock 이 가로챌 수 없다 — 브라우저가
    //   서버로 직접 요청하므로 서버 없는 데모에서는 '찾을 수 없음'이 뜬다. 안내로 바꾼다.
    try{
      document.querySelectorAll('a[href^="/api/"]').forEach(function(a){
        a.removeAttribute('download'); a.removeAttribute('href');
        a.style.opacity='0.55'; a.style.cursor='not-allowed';
        a.title='데모(읽기전용)에서는 내려받을 수 없습니다';
        a.addEventListener('click',function(e){ e.preventDefault();
          __demoToast('데모 모드에서는 파일을 내려받을 수 없습니다'); });
      });
    }catch(e){}
  });
})();
</script>
"""


def _inline_font(html: str, static_dir) -> str:
    """글꼴 파일을 HTML 안에 직접 박아 넣는다(파일 하나로 다녀야 하므로).

    ★ 왜 필요한가
      앱에서는 /static/fonts/... 로 글꼴을 받아오는데, 내보낸 HTML 은 서버 없이
      파일 하나로 열리므로 그 경로가 없다. 그대로 두면 글꼴을 못 읽어 다른 글꼴로
      나온다(깨지지는 않지만 앱 화면과 달라 보인다 — 고객사 데모용이라 이게 걸린다).
      그래서 글꼴을 글자로 바꿔 HTML 안에 넣는다. 대신 파일이 ~2.7MB 커진다.

    글꼴 파일이 없으면 @font-face 를 아예 지운다. 없는 경로를 붙들고 있어 봐야
    브라우저가 헛되이 찾을 뿐이고, 어차피 폴백 글꼴로 잘 나온다.
    """
    ref = "url('/static/fonts/PretendardVariable.woff2')"
    if ref not in html:
        return html
    f = Path(static_dir) / "fonts" / "PretendardVariable.woff2"
    try:
        b64 = base64.b64encode(f.read_bytes()).decode("ascii")
    except Exception:                                     # noqa: BLE001 — 글꼴 없으면 폴백
        import re as _re
        return _re.sub(r"@font-face\s*\{[^}]*\}", "", html, count=1)
    # 같은 url() 이 두 번(woff2-variations / woff2) 나오는데 둘 다 같은 파일이라
    # 데이터를 두 벌 넣으면 5MB 가 된다. 하나만 남기고 형식은 woff2 로 통일한다.
    html = html.replace(
        ref + " format('woff2-variations'),\n        " + ref + " format('woff2')",
        "url('data:font/woff2;base64," + b64 + "') format('woff2')")
    if ref in html:                                       # 배치가 달라졌을 때의 안전망
        html = html.replace(ref, "url('data:font/woff2;base64," + b64 + "')")
    return html


def build_export_html(index_html: str, wafers: list, detail: dict, cfg: dict,
                      static_dir=None) -> str:
    """index.html 원문 + 스냅샷(WAFERS/DETAIL/CFG) → 단일 데모 HTML 문자열."""
    cfg = dict(cfg)
    features = dict(cfg.get("export_features") or {"scores": True, "trends": True, "detail": True})
    features.setdefault("chamber_means", features.get("trends", True))
    features = {key: bool(features.get(key, True)) for key in ("scores", "trends", "detail", "chamber_means")}
    original = cfg.get("export_features") or {}
    for key in ("differences", "equipment", "recipe"):
        features[key] = bool(original.get(key, features["trends"]))
    features["trends"] = any(features[key] for key in ("differences", "equipment", "recipe"))
    features["scores"] = features["scores"] or features["detail"]
    cfg["export_features"] = features
    if not features["detail"]:
        detail = {}
    if not (features["differences"] or features["equipment"] or features["chamber_means"]):
        cfg.pop("sensor_differences", None)
    if not (features["recipe"] or features["scores"]):
        cfg.pop("step_times", None)
    if not features["scores"]:
        cfg.pop("scatter_scores", None)
        cfg.pop("offline_scores", None)
    mock = (_MOCK_TMPL
            .replace("__WAFERS__", json.dumps(wafers, ensure_ascii=False))
            .replace("__DETAIL__", json.dumps(detail, ensure_ascii=False))
            .replace("__CFG__", json.dumps(cfg, ensure_ascii=False)))
    idx = index_html.find("<script>")
    if idx < 0:
        raise RuntimeError("index.html 에 <script> 태그 없음")
    if cfg.get("export_manual_query"):
        # Export-only manual queries; the live app retains its existing behavior.
        index_html = index_html.replace(
            'async function render(opts){',
            "var exportQueryStarted=false;\n"
            "async function render(opts){\n"
            "  if(!exportQueryStarted)return;")
        index_html = index_html.replace(
            "  curRange='custom';",
            "  exportQueryStarted=true;\n  curRange='custom';")
        index_html = index_html.replace(
            "  curRange='7d';curScale='day';",
            "  exportQueryStarted=true;\n  curRange='7d';curScale='day';")
        # With no dates selected, Query uses the original recent-seven-day range.
        # A partially filled period still uses the existing validation.
        index_html = index_html.replace(
            "  if(!sd||!ed){",
            "  if(!sd&&!ed){document.getElementById('resetFilter').click();return;}\n"
            "  if(!sd||!ed){")
        idx = index_html.find("<script>")
    out = index_html[:idx] + mock + index_html[idx:]
    out = out.replace("<title>", "<title>[데모] ", 1)
    if static_dir is not None:                            # 글꼴을 파일 안에 포함(서버 없이 열리므로)
        out = _inline_font(out, static_dir)
    return out
