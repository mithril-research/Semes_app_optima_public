from pathlib import Path
import shutil,subprocess,tempfile,unittest

@unittest.skipUnless(shutil.which('gjs'),'requires SpiderMonkey')
class CustomStepRangeTests(unittest.TestCase):
    def test_inclusive_range_gaps_zero_invalid_and_single(self):
        s=Path('src/app/static/index.html').read_text();s=s[s.index('function customStepRange'):s.index('// Range controls expand')]
        s+='''
function assert(v,m){if(!v)throw Error(m);}
assert(customStepRange([1,2,4,5,26],2,5).join()==='2,4,5','inclusive gaps');
assert(customStepRange([1,2,26],26,26).join()==='26','single');
assert(customStepRange([1,2],1,99).join()==='1,2','only available');
[['',5],[0,5],[5,2],[1.5,5],['x',5],[30,40]].forEach(function(v){var failed=false;try{customStepRange([1,2,5],v[0],v[1]);}catch(e){failed=true;}assert(failed,'invalid '+v);});
void 0;
'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'check.js';p.write_text(s);r=subprocess.run(['gjs',str(p)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr+r.stdout)
