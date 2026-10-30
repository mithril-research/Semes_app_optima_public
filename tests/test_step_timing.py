import unittest
import pandas as pd
from src.data import step_timing


class StepTimingTests(unittest.TestCase):
    def test_numeric_and_repeated_steps(self):
        result=step_timing(pd.DataFrame({'Time':[0,1,2,3,5,9,10,14],
                                        'ProcessStep':[0,0,1,1,2,2,1,1]}))
        self.assertEqual(result['steps']['zero_start']['seconds'],2.)
        self.assertEqual({k:v for k,v in result['steps'].items() if k!='zero_start'},{'1':{'seconds':7.0,'visits':2,'missing':None},'2':{'seconds':5.0,'visits':1,'missing':None}})

    def test_midnight_and_fractional_clock(self):
        result=step_timing(pd.DataFrame({'Time':['23:59:59.5','00:00:00.5','00:00:03.5'],
                                        'ProcessStep':[1,1,1]}))
        self.assertEqual(result['steps']['1']['seconds'],4)

    def test_missing_reverse_single_not_zero(self):
        for times in ([5,4],[5],['bad','00:00:02']):
            result=step_timing(pd.DataFrame({'Time':times,'ProcessStep':[1]*len(times)}))
            self.assertIsNone(result['steps']['1']['seconds'])
        self.assertTrue(step_timing(pd.DataFrame({'ProcessStep':[1]}))['missing'])

    def test_repeated_step_with_invalid_visit(self):
        result=step_timing(pd.DataFrame({'Time':[0,2,5,6,9],'ProcessStep':[1,1,2,2,1]}))
        self.assertIsNone(result['steps']['1']['seconds'])
        self.assertEqual(result['steps']['2']['seconds'],4)

    def test_archive_summary_and_missing_source(self):
        import ast
        from pathlib import Path
        from unittest.mock import Mock
        tree=ast.parse(Path('src/app/server.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_wafer_step_timing')
        find=Mock(return_value=None)
        ns={'_find_wafer_source':find,'Path':Path}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<timing helper>','exec'),ns)
        saved={'version':3,'steps':{'1':{'seconds':5,'visits':1}}}
        self.assertEqual(ns['_wafer_step_timing']('w',{'meta':{'step_timing':saved}}),saved)
        find.assert_not_called()
        missing=ns['_wafer_step_timing']('w',{'meta':{}})
        self.assertEqual(missing['steps'],{})
        self.assertIn('재검사',missing['missing'])


    def test_single_row_step_uses_next_boundary(self):
        result=step_timing(pd.DataFrame({'Time':[0,2,5,8], 'ProcessStep':[1,2,3,3]}))
        self.assertEqual([result['steps'][str(s)]['seconds'] for s in (1,2,3)],[2,3,3])

    def test_invalid_summary_is_recomputed_from_original(self):
        import ast, tempfile
        from pathlib import Path
        from unittest.mock import Mock
        tree=ast.parse(Path('src/app/server.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_wafer_step_timing')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'w.csv';p.write_text('Time,ProcessStep\n0,1\n3,2\n6,2\n')
            from src.app.sensor_differences import TrendSummaryCache
            ns={'_find_wafer_source':Mock(return_value=p),'Path':Path,
                '_TREND_SUMMARY_CACHE':TrendSummaryCache(), '_model_id':lambda:'test'}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'<timing helper>','exec'),ns)
            result=ns['_wafer_step_timing']('w',{'meta':{'step_timing':{'steps':{'1':{'seconds':None}}}}})
            self.assertEqual(result['version'],3)
            self.assertEqual(result['steps']['1']['seconds'],3)
            from unittest.mock import patch
            with patch('src.data._read_csv', side_effect=AssertionError('duplicate CSV read')):
                self.assertEqual(ns['_wafer_step_timing']('w',{'meta':{}}), result)
            p.write_text('Time,ProcessStep\n0,1\n5,2\n9,2\n')
            updated=ns['_wafer_step_timing']('w',{'meta':{}})
            self.assertEqual(updated['steps']['1']['seconds'],5)


    def test_parse_cache_preserves_clock_summary(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.data import _parsed_frame
        from src.config import IOConfig
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'w.csv';cache=root/'cache';cache.mkdir()
            source.write_text('Time,ProcessStep,p_test\n12:00:00,1,1\n12:00:03,2,2\n12:00:08,2,3\n')
            with patch('src.data._parse_cache_dir',return_value=cache),patch('src.data.add_virtual_sensors',side_effect=lambda df,cfg:df):
                first=_parsed_frame(source,IOConfig())
                with patch('src.data._read_csv',side_effect=AssertionError('must read cache')):
                    second=_parsed_frame(source,IOConfig())
                self.assertEqual(first.attrs['step_timing'],second.attrs['step_timing'])
                self.assertEqual(second.attrs['step_timing']['steps']['1']['seconds'],3)
                self.assertEqual(second.attrs['step_timing']['steps']['2']['seconds'],5)
