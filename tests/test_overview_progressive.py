"""Saved-score cards must remain usable while selected scores are pending."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class OverviewProgressiveTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey gjs')
    def test_preview_completion_failure_and_stale_response(self):
        source = Path('src/app/static/index.html').read_text()
        render = source[source.index('var OVERVIEW_SCORE_ABORT='):source.index('function overviewScoreStatus(')]
        harness = r'''
const loop = new imports.gi.GLib.MainLoop(null, false);
var inspectionBusy=false;
var AbortController=class {constructor(){this.signal={aborted:false};}abort(){this.signal.aborted=true;}};
var OVPROG=null;
var dispersionToggle={checked:false};
var window={},document={getElementById:id=>id==='dispToggle'?dispersionToggle:{}};
var curRange='7d',curScale='day',curRecipe='all',CHK_CH=[],CHK_RC=[];
var rangeConf={'7d':{label:'7 days'}},SC_SENSOR_VIEW_STATE=null;
function _chkCsv(){return '';}
function dispFirstOn(){return true;}
var paints=[],messages=[],pending=[];
async function paintOverview(d,qs,id,light,status){paints.push(d.kind);messages.push(status);window._overviewPaintedQuery=qs;}
function overviewScoreStatus(s){messages.push(s);}
function pollProgress(){}
var lastArgs=null;
async function fetch(url,opts){
 if(url==='/api/progress')return {ok:true,json:async()=>({active:inspectionBusy})};
 if(url.indexOf('/api/overview?')===0)return {ok:true,json:async()=>({kind:'saved'})};
 if(url==='/api/scatter/steps')return {ok:true,json:async()=>({model_id:'m',steps:[1],sensors:['P1']})};
 lastArgs=JSON.parse(opts.body);
 return await new Promise(resolve=>pending.push(resolve));
}
function assert(v,msg){if(!v)throw Error(msg);}
async function untilPending(n){for(var i=0;i<50&&pending.length<n;i++)await Promise.resolve();assert(pending.length===n,'selection request missing');}
async function run(){
 await render();
 assert(pending.length===0,'initial view recalculated selected scores');
 assert(SC_SENSOR_VIEW_STATE.original&&paints.every(v=>v==='saved'),'initial view not saved scores');
 assert(!messages.some(s=>s.indexOf('선택 조건 계산 중')>=0),'initial saved view mislabeled');
 paints=[];messages=[];SC_SENSOR_VIEW_STATE.original=false;
 var first=render();await untilPending(1);
 assert(paints.join()==='saved','saved cards not painted before calculation');
 assert(messages.some(s=>s.indexOf('계산 중')>=0),'preview not labeled');
 pending.shift()({ok:true,json:async()=>({kind:'selected'})});await first;
 assert(paints.join()==='saved,selected','final cards missing');
 assert(messages[messages.length-1]==='','pending label not cleared');
 paints=[];var fail=render();await untilPending(1);
 pending.shift()({ok:false,json:async()=>({detail:'test failure'})});await fail;
 assert(paints.join()==='saved','failure removed preview');
 assert(messages[messages.length-1].indexOf('계산 실패')>=0,'failure not labeled');
 paints=[];var older=render();await untilPending(1);var newer=render();await untilPending(2);
 pending[1]({ok:true,json:async()=>({kind:'new'})});await newer;
 pending[0]({ok:true,json:async()=>({kind:'old'})});await older;
 assert(paints.join()==='saved,saved,new','stale response overwrote latest cards');
 pending=[];paints=[];dispersionToggle.checked=true;
 var savedState=SC_SENSOR_VIEW_STATE;var disp=render();await untilPending(1);
 assert(lastArgs.dispersion&&lastArgs.steps[0]===1&&lastArgs.sensors[0]==='P1','dispersion lost selected score request');
 pending.shift()({ok:true,json:async()=>({kind:'selected-dispersion'})});await disp;
 assert(paints.join()==='saved,selected-dispersion','selected dispersion missing');
 assert(SC_SENSOR_VIEW_STATE===savedState&&!savedState.original,'dispersion lost selected settings');
 dispersionToggle.checked=false;
 dispersionToggle.checked=true;
 pending=[];paints=[];inspectionBusy=true;
 var concurrent=render();await untilPending(1);
 assert(paints.join()==='saved','saved cards missing during inspection');
 pending.shift()({ok:true,json:async()=>({kind:'concurrent-selected'})});await concurrent;
 assert(paints.join()==='saved,concurrent-selected','inspection blocked selected scores');
 inspectionBusy=false;
 pending=[];paints=[];
 var preserved=render({preserve:true,light:true});await untilPending(1);
 assert(paints.length===0,'option change repainted saved-score preview');
 assert(messages[messages.length-1].indexOf('현재 화면 유지')>=0,'preserved view not labeled');
 pending.shift()({ok:true,json:async()=>({kind:'preserved-dispersion'})});await preserved;
 assert(paints.join()==='preserved-dispersion','option result missing');
 paints=[];var kept=render({preserve:true,light:true});await untilPending(1);
 pending.shift()({ok:false,json:async()=>({detail:'test failure'})});await kept;
 assert(paints.length===0,'failed option update replaced previous view');
 assert(messages[messages.length-1].indexOf('현재 화면 유지')>=0,'failure label misleading');
 pending=[];paints=[];SC_SENSOR_VIEW_STATE.original=true;await render();
 assert(pending.length===0&&paints.every(v=>v==='saved'),'original mode recalculated scores');
}
run().then(()=>loop.quit(),e=>{printerr(e.stack);imports.system.exit(1);});loop.run();
'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'check.js'
            path.write_text(render + '\n' + harness)
            result = subprocess.run(['gjs', str(path)], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
