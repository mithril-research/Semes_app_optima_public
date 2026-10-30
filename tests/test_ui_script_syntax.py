"""Parse the entire UI script: duplicate bindings can disable every tab."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
import tempfile


class UIScriptSyntaxTests(unittest.TestCase):
    def test_removed_overview_features_are_not_exported(self):
        from src.app.export_html import build_export_html
        source = (Path(__file__).resolve().parents[1] / 'src/app/static/index.html').read_text()
        exported = build_export_html(source, [], {}, {'export_manual_query': True})
        for html in (source, exported):
            self.assertNotIn('id="meanShiftPanel"', html)
            self.assertNotIn('id="wtPanel"', html)
            self.assertNotIn('id="wtBody"', html)
            self.assertNotIn('/api/scatter/means', html)
            self.assertIn('function mountSensorDifferences', html)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey gjs')
    def test_export_keeps_seven_day_manual_query_and_parses(self):
        from src.app.export_html import build_export_html
        source = (Path(__file__).resolve().parents[1] / 'src/app/static/index.html').read_text()
        exported = build_export_html(source, [], {}, {'export_manual_query': True})
        self.assertIn("curRange='7d'", source)
        self.assertGreaterEqual(exported.count("curRange='7d'"), 2)
        self.assertNotIn('id="exportRecent50"', exported)
        self.assertNotIn('exportRecentQuery', exported)
        self.assertNotIn('#view-overview .custom-range{display:none}', exported)
        self.assertIn('if(!exportQueryStarted)return;', exported)
        self.assertIn("if(!sd&&!ed){document.getElementById('resetFilter').click();return;}", exported)
        for script in re.findall(r'<script\b[^>]*>(.*?)</script>', exported, re.S | re.I):
            with tempfile.TemporaryDirectory() as folder:
                check = Path(folder) / 'check.js'
                check.write_text('new Function(' + json.dumps(script) + ');')
                result = subprocess.run(['gjs', str(check)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey gjs')
    def test_complete_inline_scripts_parse(self):
        source = (Path(__file__).resolve().parents[1] / 'src/app/static/index.html').read_text()
        scripts = re.findall(r'<script\b[^>]*>(.*?)</script>', source, re.S | re.I)
        self.assertTrue(scripts)
        for script in scripts:
            with tempfile.TemporaryDirectory() as folder:
                check = Path(folder) / 'check.js'
                check.write_text('new Function(' + json.dumps(script) + ');')
                result = subprocess.run(['gjs', str(check)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey gjs')
    def test_detail_basis_cancels_stale_totals_and_preserves_sensor_selection(self):
        source = (Path(__file__).resolve().parents[1] / 'src/app/static/index.html').read_text()
        begin = source.index('var DETAIL_SELECTION=null,')
        end = source.index('async function initDetailScoreControls()', begin)
        script = source[begin:end] + """
var DETAIL_SCORE_SEQ=7;
DETAIL_SELECTION={model:'m',steps:[1,26],sensors:['P1'],originalTotal:false};
DETAIL_LIST_SEQ=4;DETAIL_LIST_BUSY=true;
setDetailTotalBasis(true);
if(DETAIL_LIST_SEQ!==5||DETAIL_SCORE_SEQ!==8||DETAIL_LIST_BUSY||!DETAIL_TOTAL_DIRTY)
  throw Error('pending selected-score response not invalidated');
if(!DETAIL_ORIGINAL_TOTAL||!DETAIL_SELECTION.originalTotal||DETAIL_SELECTION.steps.join()!=='1,26'||DETAIL_SELECTION.sensors.join()!=='P1')
  throw Error('whole-wafer basis changed sensor selection');
setDetailTotalBasis(true);
if(DETAIL_LIST_SEQ!==5)throw Error('unchanged basis cancels a valid request');
setDetailTotalBasis(false);
if(DETAIL_ORIGINAL_TOTAL||DETAIL_SELECTION.originalTotal||DETAIL_LIST_SEQ!==6)
  throw Error('explicit apply cannot restore selected mode');
"""
        with tempfile.TemporaryDirectory() as folder:
            check=Path(folder)/'basis.js';check.write_text(script)
            result=subprocess.run(['gjs',str(check)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(shutil.which('gjs'), 'requires SpiderMonkey gjs')
    def test_scatter_tooltip_does_not_truncate_server_ratio_twice(self):
        source = (Path(__file__).resolve().parents[1] / 'src/app/static/index.html').read_text()
        start=source.index('function scatterScoreText(')
        end=source.index('function drawScatter(',start)
        floor=re.search(r'function flr2\(x\).*',source).group(0)
        script=floor+'\n'+source[start:end]+"""
for(var n of [4.1,2.28,1.13,0,2.27]){
 if(scatterScoreText({ratio:n},false)!==n.toFixed(2))throw Error('double truncation '+n);
}
if(scatterScoreText({ratio:2.279},true)!=='2.27')throw Error('selection was rounded up');
if(scatterScoreText({ratio:4.109},true)!=='4.10')throw Error('selection truncation');
if(scatterScoreText({ratio:null},false)!=='-')throw Error('missing ratio');
"""
        with tempfile.TemporaryDirectory() as folder:
            check=Path(folder)/'tooltip.js';check.write_text(script)
            result=subprocess.run(['gjs',str(check)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
