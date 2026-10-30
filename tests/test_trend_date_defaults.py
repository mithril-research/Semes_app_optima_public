import json
from pathlib import Path
import re
import subprocess
import unittest

class TrendDateTests(unittest.TestCase):
 def test_date_defaults_and_explicit_times(self):
  source=Path('src/app/static/index.html').read_text()
  fn=source[source.index('function trendDateTimeValue('):source.index('function trendDateTimeInput(')]
  checks='''
function check(a,b){if(a!==b)throw Error(a+' != '+b);}
check(trendDateTimeValue('2026-10-14','',false),'2026-10-14T00:00:00');
check(trendDateTimeValue('2026-10-14','',true),'2026-10-14T23:59:59');
check(trendDateTimeValue('2026-10-14','11:30',false),'2026-10-14T11:30:00');
check(trendDateTimeValue('2026-10-14','11:30',true),'2026-10-14T11:30:59');
check(trendDateTimeValue('2026-10-14','11:30:12',true),'2026-10-14T11:30:12');
check(trendDateTimeValue('','',false),'');
'''
  r=subprocess.run(['gjs','-c',fn+checks],capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr)

 def test_all_trend_ranges_use_optional_time_controls(self):
  source=Path('src/app/static/index.html').read_text()
  for selector in ('recipe-from','recipe-to','difference-from','difference-to','chamber-mean-from','chamber-mean-to','peak-detect-from','peak-detect-to'):
   self.assertRegex(source,r"trendDateTimeInput\([^;\n]*querySelector\('\."+selector+r"'\),")
  self.assertIn("(host._dateFields||{})[name]", source)
