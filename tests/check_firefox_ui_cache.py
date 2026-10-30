"""Browser regression check; uses a temporary profile and localhost fixture, not production data."""
import argparse
import json, subprocess, tempfile, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
 parser=argparse.ArgumentParser(description='Verify API caching, server restart and bfcache with a real Firefox binary.')
 parser.add_argument('--firefox',required=True,help='Path to Firefox executable')
 args=parser.parse_args()
 html=(Path(__file__).resolve().parents[1]/'src/app/static/index.html').read_text(encoding='utf-8')
 start=html.index('(function installFreshAppRequests()')
 guard=html[start:html.index('\n})();',start)+len('\n})();')]
 state={'cache':0,'instance':'one','results':{},'history_requests':0}
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args): pass
  def reply(self,body,cache='no-store',instance=False):
   data=body.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Cache-Control',cache)
   if instance:self.send_header('X-Chamber-Instance',state['instance'])
   self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  def do_POST(self):
   body=self.rfile.read(int(self.headers.get('Content-Length','0'))).decode()
   state['results'][self.path]=body
   self.reply('ok')
  def do_GET(self):
   if self.path.startswith('/api/cache'):
    state['cache']+=1;self.reply(str(state['cache']),'public, max-age=3600',True);return
   if self.path.startswith('/api/restart'):
    state['instance']='two';self.reply('restarted',instance=True);return
   report="function report(k,v){return fetch('/record/'+k,{method:'POST',body:String(v),keepalive:true});}"
   if self.path.startswith('/away'):
    self.reply('<script>setTimeout(function(){history.back();},150);</script>');return
   if self.path.startswith('/history'):
    state['history_requests']+=1
    before="window.addEventListener('pageshow',function(e){if(e.persisted)navigator.sendBeacon('/record/persisted','true');});"
    after="report('history-reload',location.search);" if 'open=' in self.path else "setTimeout(function(){location.assign('/away');},150);"
    self.reply('<!doctype html><html><body>history<script>'+report+before+guard+after+'</script></body></html>','public, max-age=3600');return
   if 'open=' in self.path:
    js=guard+report+"report('restart-reload',location.search).then(function(){location.assign('/history');});"
   else:
    js=report+"(async function(){await fetch('/api/cache');"+guard+"var value=await fetch('/api/cache').then(r=>r.text());if(value!=='2')throw Error('stale API '+value);await report('cache','fresh');await fetch('/api/restart');})().catch(e=>report('error',e.stack));"
   self.reply('<!doctype html><html><body>test<script>'+js+'</script></body></html>')
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
 threading.Thread(target=server.serve_forever,daemon=True).start()
 with tempfile.TemporaryDirectory(prefix='semes-firefox-profile-') as profile:
  log=(Path(profile)/'browser.log').open('w')
  proc=subprocess.Popen([args.firefox,'--headless','--no-remote','--profile',profile,f'http://127.0.0.1:{server.server_port}/'],stdout=log,stderr=log)
  try:
   end=time.monotonic()+40
   while time.monotonic()<end and '/record/history-reload' not in state['results'] and '/record/error' not in state['results']:
    if proc.poll() is not None:break
    time.sleep(.2)
   print(json.dumps(state,ensure_ascii=False))
   assert state['results'].get('/record/cache')=='fresh',state
   assert '/record/restart-reload' in state['results'],state
   assert state['results'].get('/record/persisted')=='true',state
   assert '/record/history-reload' in state['results'] and state['history_requests']>=2,state
   print('PASS: browser: cached API bypass, server-instance reload, real bfcache restore reload')
  finally:
   proc.terminate()
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
   log.close();server.shutdown();server.server_close()

if __name__=="__main__":
 main()
