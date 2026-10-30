"""Cylinder preprocessing, artifact contracts and torch/ONNX input parity."""
import copy
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd

from src.config import IOConfig
from src.auxiliary_sensors import (CYLINDER_SENSORS, CYLINDER_DIFF_COLUMN, AuxiliarySensorError,
    add_auxiliary_sensors, auxiliary_sensor_metadata, auxiliary_npz_metadata, configure_auxiliary_model)
from src import data
from src.onnx_backend_forecast import OnnxForecastModel


class CylinderDataTests(unittest.TestCase):
    def setUp(self):
        self.cfg=IOConfig(cylinder_mode='input',cylinder_diff=True)
        self.frame=pd.DataFrame({'ProcessStep':[0,1,1], 'p_test':[2.,3.,4.],
                                'd_004_PV':[8.,9.,10.], 'd_001_PV':[3.,4.,5.],
                                'v_011':[999.,999.,999.]})

    def test_raw_signed_difference_and_whitelist_both_loaders(self):
        for package in ('src','src.train_jw'):
            config=importlib.import_module(package+'.config').IOConfig(cylinder_mode='input',cylinder_diff=True)
            loader=importlib.import_module(package+'.data')
            result=loader._add_auxiliary_sensors(self.frame,config)
            np.testing.assert_array_equal(result['v_011'],[5.,5.,5.])
            self.assertEqual(self.frame['v_011'].iloc[0],999)
            self.assertTrue(set(CYLINDER_SENSORS+(CYLINDER_DIFF_COLUMN,))<=set(loader.split_columns(result,config)[1]))

    def test_missing_invalid_nan_infinity_fail_instead_of_zero(self):
        for name in CYLINDER_SENSORS:
            bad=self.frame.drop(columns=[name])
            with self.assertRaises(AuxiliarySensorError):add_auxiliary_sensors(bad,self.cfg)
            for value in ('oops',None,np.nan,np.inf):
                bad=self.frame.astype(object);bad.loc[1,name]=value
                with self.assertRaises(AuxiliarySensorError):add_auxiliary_sensors(bad,self.cfg)
        bad=self.frame.copy();bad['d_004_PV']=1e308;bad['d_001_PV']=-1e308
        with self.assertRaises(AuxiliarySensorError):add_auxiliary_sensors(bad,self.cfg)

    def test_cache_and_schema_are_separate_between_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'w.csv';self.frame.to_csv(path,index=False)
            with patch.dict(os.environ,IO_PARSE_CACHE_DIR=str(root/'cache'),IO_LOAD_WORKERS='1'):
                off=IOConfig();_,_,old,_=data.load_dataset(root,off)
                self.assertNotIn('v_011',old)
                records,_,active,_=data.load_dataset(root,self.cfg)
                self.assertIn('v_011',active)
                np.testing.assert_array_equal(records[0]['sensors'][:,active.index('v_011')],5)
                _,_,again,_=data.load_dataset(root,off)
                self.assertEqual(old,again)
                # Invalid data in a later row must abort full-file reading, even if the schema sample was valid.
                bad=pd.concat([self.frame]*3,ignore_index=True).astype(object);bad.loc[8,'d_001_PV']='bad'
                bad.to_csv(path,index=False)
                self.cfg.schema_infer_rows=2
                for workers in ('1','2'):
                    with patch.dict(os.environ,IO_LOAD_WORKERS=workers):
                        with self.assertRaises(AuxiliarySensorError):data.load_dataset(root,self.cfg)

    def test_artifact_roles_indices_and_definition_validate(self):
        def model(mode='input'):
            cfg=IOConfig(cylinder_mode=mode,cylinder_diff=True)
            return SimpleNamespace(cfg=cfg,arch=auxiliary_sensor_metadata(cfg),
                input_sensor_cols=['p_test','h_test',*CYLINDER_SENSORS,CYLINDER_DIFF_COLUMN],
                sensor_cols=['p_test']+(list(CYLINDER_SENSORS)+[CYLINDER_DIFF_COLUMN] if mode=='predict' else []),
                h_indices=[1],future_h_mode='input')
        m=model();configure_auxiliary_model(m,{})
        self.assertEqual(m.future_input_sensor_indices,[1,2,3,4])
        meta=auxiliary_npz_metadata(m)
        for change in ({'future_context_indices':[2,3]}, {'future_input_sensor_indices':[1,4,3,2]},
                       {'auxiliary_sensor_cols':['bad']}, {'future_h_indices':[99]},
                       {'future_context_indices':[2,3,4.5]},
                       {'cylinder_diff_definition_json':json.dumps(dict(column='v_011',minuend='d_001_PV'))}):
            with self.assertRaises(AuxiliarySensorError):configure_auxiliary_model(model(),dict(meta,**change))
        predicted=model('predict');configure_auxiliary_model(predicted,{})
        self.assertEqual(predicted.future_input_sensor_indices,[1])
        missing=model();missing.input_sensor_cols.remove('v_011')
        with self.assertRaises(AuxiliarySensorError):configure_auxiliary_model(missing,{})
        old=model();old.arch={};old.input_sensor_cols=['p_test','h_test'];old.sensor_cols=['p_test'];configure_auxiliary_model(old,{})
        self.assertEqual(old.cfg.cylinder_mode,'off');self.assertFalse(old.cfg.cylinder_diff)
        self.assertEqual(old.future_input_sensor_indices,[1])


class CylinderInferenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(13)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        env=patch.dict(os.environ,IO_ORT_THREADS='1',IO_ORT_PROVIDER='cpu');env.start();self.addCleanup(env.stop)

    def make_model(self, package, mode):
        mod=importlib.import_module(package+'.model_tcn_forecast')
        cfg=importlib.import_module(package+'.config').IOConfig(cylinder_mode=mode,cylinder_diff=True)
        cfg.h_cart_mode='exclude';cfg.use_step_onehot=True;cfg.score_final_avg=True
        m=mod.TCNForecastModel(cfg,device='cpu');m.io_cols=['9000001'];m.n_steps_global=2
        m._setup_sensor_schema(['p_test','h_test',*CYLINDER_SENSORS,'v_011'])
        m.w=2;m.h=2;m.stride=1
        m.input_sen_mean=np.arange(len(m.input_sensor_cols),dtype=float)
        m.input_sen_std=np.arange(1,len(m.input_sensor_cols)+1,dtype=float)
        m.sen_mean=np.zeros(len(m.sensor_cols));m.sen_std=np.ones(len(m.sensor_cols))
        m.resid_abs_limit=np.full(len(m.sensor_cols),2.);m.resid_std=m.resid_abs_limit;m.resid_sigma=np.ones(len(m.sensor_cols))
        m._per_step_limit=False;m._per_step_sigma=False
        m.arch=dict(past_window=2,future_window=2,stride=1,channels=4,blocks=1,kernel=2,dropout=0.,
                    future_h_mode='input',virtual_sensor_mode='off',h_cart_mode='exclude',**auxiliary_sensor_metadata(cfg))
        m.net=mod.IOTCN(m.n_in(),len(m.sensor_cols),4,1,2,0.).to('cpu');m.net.eval()
        rec=dict(wafer_id='w',io=np.arange(5,dtype=float).reshape(-1,1),sensor_cols=list(m.input_sensor_cols),
                 sensors=np.array([[2+i,4+i,8+i,3+i,5] for i in range(5)],dtype=float),step=np.array([0,1,1,1,0]))
        return m,rec

    def test_both_exporters_roundtrip_and_inference_parity(self):
        for package in ('src','src.train_jw'):
            for mode in ('input','predict'):
                with self.subTest(package=package,mode=mode):
                    m,rec=self.make_model(package,mode)
                    folder=self.root/(package.replace('.','_')+mode);folder.mkdir();pt=folder/'tcn_forecast.pt';m.save(pt)
                    (folder/'search_best.json').write_text(json.dumps({'best':{'wafer_score_override':dict(version=2,rules=[dict(pattern='w',threshold_multiplier=1.01),dict(pattern='second',threshold_multiplier=1.15)])}}))
                    (folder/'wafer_score_override.json').write_text(json.dumps(dict(patterns=['OTHER'], threshold_multiplier=1.01)))
                    exporter=importlib.import_module(package+'.export_onnx_forecast')
                    # Use the stable legacy ONNX exporter for the numerical fixture under new torch releases.
                    import torch
                    export=torch.onnx.export
                    def legacy_export(*args,**kwargs):return export(*args,**dict(kwargs,dynamo=False))
                    with patch.object(torch.onnx,'export',side_effect=legacy_export):exporter.export(pt,folder)
                    app=OnnxForecastModel.load(folder,IOConfig())
                    self.assertEqual(app.cfg.cylinder_mode,mode)
                    self.assertTrue(app.cfg.cylinder_diff)
                    self.assertEqual(app.wafer_score_override["rules"][0]["threshold_multiplier"],1.01)
                    # Prove the exported NPZ preserves the rule without the source sidecar.
                    (folder/"wafer_score_override.json").unlink()
                    x=m._std_input_sensor(m._input_sensors(rec));y=m._std_target_sensor(m.actual(rec))
                    expected=m._block(rec['io'],x,y,rec['step'],m._score_keep_mask(rec),2,5)[0]
                    actual=app._block_x(rec['io'],app._std_input_sensor(app._input_sensors(rec)),rec['step'],2,5)
                    np.testing.assert_allclose(actual,expected,rtol=0,atol=0)
                    for name in (*CYLINDER_SENSORS,'v_011'):
                        index=app.input_sensor_cols.index(name)
                        np.testing.assert_allclose(actual[1+index,2:],x[2:4,index] if mode=='input' else 0)
                    np.testing.assert_allclose(app.predict(copy.deepcopy(rec)),m.predict(copy.deepcopy(rec)),rtol=1e-5,atol=1e-6)
                    np.testing.assert_allclose(app.residual(copy.deepcopy(rec)),m.residual(copy.deepcopy(rec)),rtol=1e-5,atol=1e-6)
                    app.score_norm='final'
                    expected_score=m._score_record(copy.deepcopy(rec))
                    self.assertAlmostEqual(app._score_record(copy.deepcopy(rec)),expected_score,places=5)
                    # Exercise the actual Engine contract and aligned data loader as well.
                    from src.app import server
                    from src.app.chamber_models import EngineRouter
                    (folder/'search_best.json').write_text(json.dumps({'best':{'val':{'threshold':4.,'score_final_avg':True}}}))
                    (folder/'wafer_score_override.json').write_text('null')
                    engine=server.Engine();engine.selected=folder
                    with patch.object(server,'_probe_model_dir',return_value=(True,'')):
                        engine.load()
                    self.assertTrue(engine.ready,engine.error)
                    self.assertEqual(engine.cfg.cylinder_mode,mode)
                    self.assertTrue(engine.cfg.cylinder_diff)
                    self.assertEqual(engine.score_norm,'final')
                    self.assertEqual([r['pattern'] for r in engine.wafer_score_override['rules']],['w','second'])
                    from src.wafer_score_override import effective_wafer_score
                    corrected = effective_wafer_score(rec, expected_score, engine.threshold, engine.wafer_score_override)
                    self.assertTrue(corrected['score_overridden'])
                    self.assertEqual(corrected['override_ratio'],1.01)
                    logs=folder/'logs';logs.mkdir()
                    frame=pd.DataFrame(rec['sensors'],columns=rec['sensor_cols'])
                    frame['v_011']=999;frame['ProcessStep']=rec['step'];frame['9000001']=rec['io'][:,0]
                    frame.to_csv(logs/'w.csv',index=False)
                    with patch.object(server,'ENGINE',EngineRouter(engine)), patch.object(server,'_load_app_config',return_value={}), patch.object(server,'_tag_file_meta'), patch.object(server,'_apply_cache_env'), patch.object(server,'_reader_cancel_check',return_value=None), patch.object(server,'_recipe_threshold',return_value=('recipe',4.)):
                        loaded,_,columns,coverage=server._load_aligned(logs)
                    self.assertEqual(coverage['n_auxiliary_missing'],0)
                    self.assertEqual(coverage['sensor_coverage'],1.)
                    np.testing.assert_array_equal(loaded[0]['sensors'][:,columns.index('v_011')],5.)
                    bad=copy.deepcopy(rec);bad['sensor_cols'][-1]='missing'
                    with self.assertRaises(AuxiliarySensorError):app.predict(bad)
                    # Metadata are the same contract in automatic and embedded conversion paths.
                    with np.load(folder/'tcn_forecast.npz',allow_pickle=True) as meta:
                        self.assertIn('cylinder_diff_definition_json',meta)
                        self.assertEqual(list(meta['future_context_indices']),[2,3,4] if mode=='input' else [])


if __name__=='__main__':unittest.main()
