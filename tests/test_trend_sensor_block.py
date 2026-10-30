import ast
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.trend_sensors import sensor_block_columns, trend_measurements, attach_trend_measurements
from src.app.sensor_differences import difference_series, TrendSummaryCache


class TrendSensorBlockTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'Date':['2026-10-19']*3, 'Time':[0,1,2], 'ProcessStep':[0,1,1],
            'h_011_PV':[99,2,4], 't_test':[99,10,20], 'h_004_PV':[99,7,9],
            'd_Gauge':[99,3,5], 's_001_Press':[99,5,9], '9000009':[1,0,1],
            's_012_PV':[99,90,90]})

    def test_column_boundaries_not_sensor_whitelist(self):
        f=self.frame();columns=sensor_block_columns(f.columns)
        self.assertEqual(columns,['h_011_PV','t_test','h_004_PV','d_Gauge','s_001_Press'])
        self.assertEqual(sensor_block_columns(['ProcessStep','9000009','t_test']),[])
        self.assertEqual(sensor_block_columns(['ProcessStep','t_test']),[])
        measurements=trend_measurements(f)
        self.assertEqual(measurements['step'].tolist(),[0,1,1])
        archive=dict(meta={'sensor_cols':['t_test'],'interp_factor':4},actual=np.array([[999.]]),step=np.array([1]))
        row=difference_series(attach_trend_measurements(archive,measurements),pairs=[['h_004_PV','d_Gauge']])
        self.assertEqual(row['sensor_scope'],'full_block')
        self.assertEqual([r['sensor'] for r in row['sensor_series']],columns)
        self.assertEqual(row['sensor_series'][1]['step_stats']['1']['mean'],15.)
        self.assertEqual(row['series'][-1]['step_stats']['1']['mean'],4.)
        self.assertEqual(row['valve_phases']['status'],'unavailable')

    def test_missing_values_remain_missing(self):
        f=self.frame();f.loc[1:,'d_Gauge']=np.nan
        row=difference_series(attach_trend_measurements({'meta':{},'step':np.array([1])},trend_measurements(f)))
        gauge=next(s for s in row['sensor_series'] if s['sensor']=='d_Gauge')
        self.assertNotIn('1',gauge['step_stats'])
        self.assertEqual(gauge['step_stats']['zero_start']['mean'],99.)
        json.dumps(row,allow_nan=False)

    def test_legacy_archive_reads_original_and_invalidates_source_cache(self):
        tree=ast.parse(Path('src/app/server.py').read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_trend_summary','_trend_original_path')]
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'w.csv';self.frame().to_csv(source,index=False)
            archive=Path(td)/'w.npz'
            meta={'sensor_cols':['t_test'],'source_path':str(source)}
            np.savez(archive,meta_json=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8),actual=np.array([[5.],[5.],[5.]]),step=np.array([0,1,1]))
            ns=dict(np=np,json=json,Path=Path,_ARCHIVE_Q=None,_model_id=lambda:'model',
                    _archive_path=lambda d,w:archive,_find_wafer_source=lambda w:None,
                    _TREND_SUMMARY_CACHE=TrendSummaryCache())
            exec(compile(ast.Module(body=funcs,type_ignores=[]),'<trend>','exec'),ns)
            query=lambda:ns['_trend_summary']('2026-10-19','w')['summary']
            first=query();self.assertEqual(first['sensor_scope'],'full_block')
            self.assertEqual(first['sensor_series'][0]['step_stats']['1']['mean'],3.)
            f=self.frame();f['h_011_PV']=[99,200,400];f.to_csv(source,index=False)
            self.assertEqual(query()['sensor_series'][0]['step_stats']['1']['mean'],300.)
            source.unlink()
            self.assertEqual(query()['sensor_scope'],'archive_subset')

    def test_loader_preserves_block_without_changing_inference_inputs(self):
        from src.data import load_dataset
        from src.config import IOConfig
        with tempfile.TemporaryDirectory() as td:
            self.frame().to_csv(Path(td)/'wafer.csv',index=False)
            cfg=IOConfig(sensor_keep=('t_test',),h_cart_mode='exclude')
            before=load_dataset(Path(td),cfg)[0][0]
            after=load_dataset(Path(td),cfg,include_trend_source=True)[0][0]
            np.testing.assert_array_equal(before['sensors'],after['sensors'])
            np.testing.assert_array_equal(before['io'],after['io'])
            self.assertNotIn('trend_measurements',before)
            self.assertIn('d_Gauge',after['trend_measurements']['sensor_cols'])
            self.assertNotIn('s_012_PV',after['trend_measurements']['sensor_cols'])

    def test_archive_round_trip_preserves_full_block_without_source(self):
        from types import SimpleNamespace
        from queue import Queue
        tree=ast.parse(Path('src/app/server.py').read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_archive_wafer','_load_archive')]
        for fn in funcs:fn.decorator_list=[]
        measurements=trend_measurements(self.frame())
        rec={'step':np.array([1,1]),'trend_measurements':measurements,'source_path':'gone.log'}
        model=SimpleNamespace(sensor_cols=['t_test'])
        q=Queue()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'w.npz'
            ns=dict(np=np,json=json,Dict=dict,ENGINE=SimpleNamespace(model=model),
                    RX=SimpleNamespace(_compute=lambda a,r:(np.ones((2,1)),)*3),
                    _load_app_config=lambda:{'archive_rca_source':False},
                    _lot_of=lambda w:'lot',_end_ts_and_date=lambda w:('2026-10-19T00:00:00','2026-10-19'),
                    _valve_open_map=lambda r:({},2),_model_id=lambda:'m',_archive_path=lambda d,w:path,
                    _archive_write_path=lambda d,w:path,
                    _archive_queue=lambda:q,traceback=__import__('traceback'))
            exec(compile(ast.Module(body=funcs,type_ignores=[]),'<archive>','exec'),ns)
            ns['_archive_wafer'](rec,{'wafer_id':'w'})
            self.assertEqual(q.qsize(),1)
            _,arrays=q.get_nowait();np.savez(path,**arrays)
            loaded=ns['_load_archive']('2026-10-19','w')
            self.assertEqual(loaded['meta']['trend_sensor_cols'],measurements['sensor_cols'])
            self.assertEqual(difference_series(loaded)['sensor_scope'],'full_block')
            self.assertNotIn('trend_measurements',rec)
