"""Exercise real archive/index/switch helpers with isolated model and disk fixtures."""
import ast
import datetime as dt
import hashlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.wafer_score_override import AUDIT_FIELDS, normalize_override, override_identity
from src.auxiliary_sensors import auxiliary_sensor_columns


class ModelHistoryRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.mid='m@A';self.q=queue.Queue()
        root=SimpleNamespace(art=self.base/'A');root.art.mkdir()
        self.engine=SimpleNamespace(root=root,model=SimpleNamespace(sensor_cols=['p_test']),_switch_lock=threading.RLock())
        self.state=dict(results=[],recs={},progress={},progress_bg={})
        self.ns=dict(AUDIT_FIELDS=AUDIT_FIELDS,Dict=dict,List=list,Path=Path,np=np,json=json,os=os,hashlib=hashlib,time=time,dt=dt,shutil=shutil,
            ENGINE=self.engine,STATE=self.state,STATE_LOCK=threading.Lock(),_HIST_LOCK=threading.Lock(),
            ARCHIVE_DIR=self.base/'archive',_model_id=lambda:self.mid,_legacy_model_id=lambda:'m@legacy',
            _safe=lambda w:w,_lot_of=lambda w:'lot',_model_archive_path=None,
            _end_ts_and_date=lambda w:('2026-10-19T10:00:00','2026-10-19'),
            _HIST_IDX=dict(built=False,rows={},by_stem={},sorted=None),
            _load_ntp=lambda:None,_flush_ntp=lambda:None,_NTP={},_NTP_LOCK=threading.Lock(),
            _isnum=lambda v:isinstance(v,(int,float)),_relive_rows=lambda rows:rows,
            _RELIVE_CACHE={'key':None},_AGG_CACHE={},
            RX=SimpleNamespace(_compute=lambda a,r:(np.full((2,1),r['value']),)*3),
            _valve_open_map=lambda r:({},2),_load_app_config=lambda:{'archive_rca_source':False},
            _archive_queue=lambda:self.q,_ARCHIVE_Q=None,traceback=traceback,_archive_retention_days=lambda:0)
        names=['_archive_model_matches','_model_archive_path','_archive_write_path','_archive_path',
               '_atomic_savez','_archive_wafer','_load_archive','_hist_entry','_append_history',
               '_hist_index_ensure','_read_history','_history_for_current_model','_restore_model_results',
               '_archive_date_of','_prune_archive','_prune_history']
        self.tree=ast.parse(Path('src/app/server.py').read_text())
        funcs=[n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        for fn in funcs:fn.decorator_list=[]
        exec(compile(ast.Module(body=funcs,type_ignores=[]),'<actual-history>','exec'),self.ns)

    def save(self,value,wid='wafer'):
        result={'wafer_id':wid,'score':value,'score_raw':value,'threshold':1,'verdict':'NG'}
        self.ns['_append_history']([result])
        self.ns['_archive_wafer']({'value':value,'step':np.array([1,1])},result)
        path,arrays=self.q.get_nowait();self.ns['_atomic_savez'](path,arrays)
        return path

    def restore(self,mid):
        self.mid=mid
        self.ns['_restore_model_results'](force=True)
        return [r['score'] for r in self.state['results']]

    def test_a_b_a_same_wafer_persists_scores_arrays_and_restart(self):
        a=self.save(2.28);self.mid='m@B';b=self.save(8.71)
        self.assertNotEqual(a,b);self.assertTrue(a.exists() and b.exists())
        self.assertEqual(self.restore('m@A'),[2.28])
        np.testing.assert_allclose(self.ns['_load_archive']('2026-10-19','wafer')['z'],2.28,atol=.01)
        self.assertEqual(self.restore('m@B'),[8.71])
        np.testing.assert_allclose(self.ns['_load_archive']('2026-10-19','wafer')['z'],8.71,atol=.01)
        self.ns['_HIST_IDX'].update(built=False,rows={},by_stem={},sorted=None)
        self.state.clear();self.state.update(results=[],recs={},progress={},progress_bg={})
        self.assertEqual(self.restore('m@A'),[2.28])
        self.assertEqual(self.restore('m@B'),[8.71])
        self.assertEqual(self.ns['_HIST_IDX']['by_stem']['wafer'],{'m@A','m@B'})
        self.assertEqual(self.ns['_archive_date_of']('wafer'),'2026-10-19')
        self.assertEqual(self.restore('m@C'),[])
        self.assertIsNone(self.ns['_load_archive']('2026-10-19','wafer'))

    def test_legacy_flat_archive_is_preserved_and_only_matches_owner(self):
        a=self.save(2);flat=a.parent.parent/a.name;a.replace(flat)
        self.mid='m@B';self.save(7)
        self.ns['_HIST_IDX'].update(built=False,rows={},by_stem={},sorted=None)
        self.assertEqual(self.restore('m@A'),[2])
        self.assertEqual(self.ns['_archive_path']('2026-10-19','wafer'),flat)
        self.assertEqual(self.restore('m@B'),[7])
        self.mid='m@C';self.assertFalse(self.ns['_archive_path']('2026-10-19','wafer').exists())
        self.mid='m@A';self.save(3)
        self.ns['_HIST_IDX'].update(built=False,rows={},by_stem={},sorted=None)
        self.assertEqual(self.restore('m@A'),[3])
        self.assertTrue(flat.exists())
        self.assertEqual(self.restore('m@B'),[7])

    def test_same_model_reinspection_updates_only_that_model(self):
        self.save(1);self.mid='m@B';self.save(9);self.mid='m@A';self.save(2)
        self.assertEqual(len(list(self.ns['ARCHIVE_DIR'].rglob('*.npz'))),2)
        self.assertEqual(self.restore('m@A'),[2]);self.assertEqual(self.restore('m@B'),[9])

    def test_retention_prunes_all_models_without_tuple_unpack_error(self):
        self.save(1);self.mid='m@B';self.save(9)
        self.ns['_archive_retention_days']=lambda:1
        self.ns['dt']=SimpleNamespace(date=SimpleNamespace(today=lambda:dt.date(2026,10,25),fromisoformat=dt.date.fromisoformat),timedelta=dt.timedelta)
        self.assertEqual(self.ns['_prune_archive'](),1)
        self.assertEqual(self.ns['_prune_history'](),2)
        self.assertEqual(self.ns['_HIST_IDX']['by_stem'],{})

    def test_model_select_restores_and_failure_does_not_clear_current_results(self):
        from starlette.exceptions import HTTPException
        self.save(2);self.mid='m@B';self.save(9);self.restore('m@A')
        bdir=self.base/'B';bdir.mkdir()
        def select(path):
            self.mid='m@'+Path(path).name;self.engine.root.art=Path(path)
            return {'ready':True,'backend':'fixture'}
        self.engine.select=select
        ns=self.ns;ns.update(SelectReq=object,HTTPException=HTTPException,_has_model=lambda p:True,
                            _probe_model_dir=lambda p:(True,''),_add_custom_dir=lambda p:None,
                            _invalidate_ok_bands=lambda:None,AUTOPILOT=None)
        fn=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='api_model_select');fn.decorator_list=[]
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<select>','exec'),ns)
        self.assertEqual(ns['api_model_select'](SimpleNamespace(dir=str(bdir)))['restored_results'],1)
        self.assertEqual(self.state['results'][0]['score'],9)
        ns['_probe_model_dir']=lambda p:(False,'bad model')
        with self.assertRaises(HTTPException):ns['api_model_select'](SimpleNamespace(dir=str(self.base/'A')))
        self.assertEqual(self.state['results'][0]['score'],9)

    def test_single_model_identity_includes_weights_and_survives_move(self):
        fn=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='_model_id')
        a=self.base/'one';b=self.base/'two';a.mkdir();b.mkdir()
        (a/'net.onnx').write_bytes(b'weights A');(b/'net.onnx').write_bytes(b'weights B')
        ns=dict(Path=Path,hashlib=hashlib,json=json,normalize_override=normalize_override,override_identity=override_identity,auxiliary_sensor_columns=auxiliary_sensor_columns,_legacy_model_id=lambda:'m@same-meta')
        def identify(folder, override=None):
            ns['ENGINE']=SimpleNamespace(root=SimpleNamespace(bundle_id=None,art=folder,wafer_score_override=override,model=SimpleNamespace(_onnx_path=folder/'net.onnx')))
            return ns['_model_id']()
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<identity>','exec'),ns)
        first=identify(a);self.assertNotEqual(first,identify(b))
        shutil.copyfile(a/'net.onnx',b/'net.onnx');self.assertEqual(first,identify(b))
        rule=dict(patterns=['BAD'],threshold_multiplier=1.01)
        self.assertNotEqual(first,identify(a,rule))
        self.assertEqual(identify(a,rule),identify(b,rule))

    def test_restored_results_are_not_hidden_by_previous_model_filters(self):
        html=Path('src/app/static/index.html').read_text()
        code=html[html.index('function showRestoredModelResults('):html.index('async function applyModel(')]
        js=r"""
var SC_SENSOR_VIEW_STATE={steps:[99]},CHK_CH={dsv1:false},CHK_RC={old:false},_checksSig='old',curRecipe='old';
var customStart=null,customEnd=null,curRange='7d',curScale='day',closed=0;
var nodes={};var document={getElementById:function(id){return nodes[id]||(nodes[id]={});},querySelectorAll:function(){return [];}};
function closeDrill(){closed++;}function applyScaleAvailability(range){if(range!=='custom')throw Error('range');}
"""+code+r"""
showRestoredModelResults({restored_range:{start:'2025-01-01',end:'2026-10-19'}});
if(curRange!=='custom'||curScale!=='month'||curRecipe!=='all'||closed!==1)throw Error('restored period');
if(Object.keys(CHK_CH).length||Object.keys(CHK_RC).length||SC_SENSOR_VIEW_STATE!==null)throw Error('stale filters');
if(nodes.startDate.value!=='2025-01-01'||nodes.endDate.value!=='2026-10-19'||customEnd.getHours()!==23||customEnd.getSeconds()!==59)throw Error('date boundaries');
"""
        result=subprocess.run(['gjs','-c',js],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
