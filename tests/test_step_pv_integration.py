"""Synthetic artifact tests; production v0.2.0 artifacts/Win7 are not available here."""
import ast
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from src.app import integrated_scores as scoring
from src.app.auxiliary import step_pv_explainer as pv
from test_integrated_scores import sample, make_step


def pv_fixture(root):
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor
    grid=np.linspace(0,4,8,dtype=np.float32)
    refs=np.array([grid,grid*.5],np.float32)
    rng=np.random.RandomState(7)
    x=rng.uniform(0,7,(80,17)).astype(np.float32)
    y=2+x[:,:8].mean(1)*.7-x[:,8:16].mean(1)*.5
    model=ExtraTreesRegressor(n_estimators=8,random_state=6,n_jobs=1).fit(x,y)
    joblib.dump(model,root/'pv.joblib')
    np.savez(root/'pv.npz',sensors=np.array(['P_902','T_902']),grid_seconds=grid,
        reference_trajectories=refs,feature_reference=np.r_[refs.ravel(),0],
        normal_changes=np.array([4,2]),normal_response_times=np.array([2,2]))
    info=dict(group_key='all',step=1,occurrence=1,selected_sensors=['P_902','T_902'],grid_points=8,
        target_value=4,direction='up',normal_reach_median_seconds=4,normal_duration_median_seconds=5,
        transition_type='pressure',transition_confidence=.9,explanation_confidence='high',
        validation_mae_seconds=.1,validation_r2=.8,model_path='pv.joblib',stats_path='pv.npz')
    (root/'pv_explainer_meta.json').write_text(json.dumps(dict(version='0.2.0',target_sensor='p_008_PV',models={'event':info})))
    return model,info


def frame_sample(i=0):
    s=sample();s.path=Path('WUK370B_PMCUNIT8_29a_%d.csv'%i)
    s.frame['p_008_PV']=np.arange(12,dtype=float)*.5
    s.frame['P_902']=np.arange(12,dtype=float)*.3
    s.frame['T_902']=0.0 # expected response missing must stay a candidate
    return s


@pytest.mark.parametrize('chamber',['DSV1','DSV2','DSV3','DSV4','DSV5','DSV6'])
def test_direct_chamber_paths(tmp_path,chamber):
    a,b=scoring.paths(tmp_path,{},chamber)
    assert a==tmp_path/'보조모델/스텝시간'/chamber
    assert b==tmp_path/'보조모델/실린더'/chamber
    assert scoring.paths(tmp_path,{},'UNKNOWN')==(None,None)


def test_arbitrary_packaging_folder_and_explicit_override(tmp_path):
    root=tmp_path/'보조모델/스텝시간/DSV2/release_02';root.mkdir(parents=True)
    (root/'step_time_meta.json').write_text('{}')
    assert scoring.paths(tmp_path,{},'DSV2')[0]==root
    custom=tmp_path/'custom';custom.mkdir();(custom/'step_time_meta.json').write_text('{}')
    assert scoring.paths(tmp_path,{'step_time_by_chamber':{'DSV2':'custom'}},'DSV2')[0]==custom
    assert scoring.paths(tmp_path,{'step_time_artifacts':'custom'},'DSV1')[0]!=custom
    # Direct metadata wins; do not silently pick one of two releases.
    (root.parent/'step_time_meta.json').write_text('{}')
    assert scoring.paths(tmp_path,{},'DSV2')[0]==root.parent


def test_pv_parity_batch_and_missing_response(tmp_path):
    root=tmp_path/'model';make_step(root);model,info=pv_fixture(root)
    samples=[frame_sample(i) for i in range(3)]
    scores=[scoring.step_time_score(root,s) for s in samples]
    originals=[sc['ratio'] for sc in scores]
    bundle=scoring.load_artifacts(root,'step_time')
    pv.explain_batch(root,'all',list(zip(samples,scores)),bundle,threading.RLock())
    for s,score,ratio in zip(samples,scores,originals):
        assert score['ratio']==ratio
        event=next(e for e in score['pv']['events'] if e['step']==1)
        assert event['available'] and event['actual_reach_seconds'] is None
        candidates={c['sensor']:c for c in event['candidates']}
        assert candidates['T_902']['response_status']=='expected_response_missing'
        assert not candidates['T_902']['sensor_missing']
        cache={};pv._extract_frame_features(s,s.frame,{'step-1__occ-1':dict(target_value=4,direction='up',grid_seconds=np.linspace(0,4,8))},'p_008_PV',info['selected_sensors'],cache)
        with np.load(root/'pv.npz') as z:stats={k:z[k] for k in z.files}
        x=pv._rows_to_matrix([cache[(str(s.path),'step-1__occ-1')]],info['selected_sensors'],stats['reference_trajectories'],8,0)
        predicted=float(model.predict(x)[0])
        assert event['predicted_reach_seconds']==pytest.approx(predicted)
        for j,sensor in enumerate(info['selected_sensors']):
            replaced=x.copy();replaced[:,j*8:(j+1)*8]=stats['reference_trajectories'][j]
            assert candidates[sensor]['contribution_seconds']==pytest.approx(predicted-model.predict(replaced)[0])
            assert len(candidates[sensor]['impact_windows'])==4
            for k,w in enumerate(candidates[sensor]['impact_windows']):
                block=np.array_split(np.arange(8),4)[k];r=x.copy();r[:,j*8+block]=stats['reference_trajectories'][j,block]
                assert w['contribution_seconds']==pytest.approx(predicted-model.predict(r)[0])
        assert next(e for e in score['pv']['events'] if e['step']==2)['available'] is False
    # IF most-contributing step, not largest pressure delay.
    scores[0]['ratio']=2;scores[0]['step_details'][1]['contribution']=100
    pv.explain_batch(root,'all',[(samples[0],scores[0])],bundle,threading.RLock())
    assert scores[0]['pv']['primary_step']==2
    assert scores[0]['pv'].get('primary_sensor') is None
    lightweight=scoring.summary(scoring.combine({'available':True,'ratio':.5},scores[0],scoring.unavailable('x')))
    assert 'events' not in lightweight['step_time']['pv']
    assert 'step_details' not in lightweight['step_time']
    assert scores[0]['pv']['events'] # summary did not mutate archive payload


class CountingModel:
    def __init__(self):self.calls=0
    def predict(self,x):
        self.calls+=1
        return x[:,:8].mean(1)*2-x[:,8:16].mean(1)


@pytest.mark.parametrize('n',[1,1122])
def test_predict_calls_do_not_scale_with_wafers(n):
    grid=np.linspace(0,4,8,dtype=np.float32);refs=np.zeros((2,8),np.float32)
    stats=dict(sensors=np.array(['A','B']),grid_seconds=grid,reference_trajectories=refs,
               normal_changes=np.array([4,4]),normal_response_times=np.array([2,2]),feature_reference=np.zeros(17))
    info=dict(step=3,occurrence=1,normal_reach_median_seconds=4,transition_type='pressure',target_value=1,direction='up')
    rows=[]
    for i in range(n):
        rows.append((None,None,dict(target_initial=0,target_reach=6 if i%2==0 else 2,duration=7,
            trajectories={'A':np.ones(8,dtype=np.float32),'B':np.ones(8,dtype=np.float32)*2},
            target_series=(np.array([0,7]),np.array([0,3])))))
    model=CountingModel();events=pv._event_predictions(model,stats,info,'target',rows)
    assert model.calls==1+2+2*4
    assert events[0]['candidates'][0]['sensor']=='A'
    if n>1:assert events[1]['candidates'][0]['sensor']=='B'
    for i,event in enumerate(events):
        c=event['candidates'][0];w=c['primary_impact_window']
        assert w and (w['contribution_seconds']>0 if i%2==0 else w['contribution_seconds']<0)


def test_missing_pv_and_one_read_per_wafer(tmp_path,monkeypatch):
    root=tmp_path/'보조모델/스텝시간/DSV2';make_step(root)
    s=frame_sample();reader=Mock(return_value=s.frame);monkeypatch.setattr(scoring.sd,'read_process_file',reader)
    results=scoring.inspect_many([s.path]*3,[dict(chamber='DSV2',ratio_thr=.4)]*3,tmp_path,{})
    assert reader.call_count==3
    assert all(r['step_time']['available'] and not r['step_time']['pv']['available'] for r in results)
    before=results[0]['integrated']['ratio'];pv_fixture(root);reader.reset_mock()
    results=scoring.inspect_many([s.path]*3,[dict(chamber='DSV2',ratio_thr=.4)]*3,tmp_path,{})
    assert reader.call_count==3 and results[0]['step_time']['pv']['available']
    assert results[0]['integrated']['ratio']==before
    assert results[0]['step_time']['artifact_digest']!=scoring.archived({'score_components':{}}).get('step_time',{}).get('artifact_digest')
    # Missing candidate values don't hide other sensors or invalidate IF.
    s.frame=s.frame.drop(columns=['P_902']);reader.return_value=s.frame
    result=scoring.inspect(s.path,dict(chamber='DSV2',ratio_thr=.4),tmp_path,{})
    event=result['step_time']['pv']['events'][0]
    assert event['candidates'][-1]['sensor_missing']
    assert result['step_time']['available']


def test_microbatch_server_integration():
    tree=ast.parse(Path('src/app/server.py').read_text());node=next(n for n in tree.body if getattr(n,'name','')=='_results_with_aux')
    many=Mock(side_effect=lambda sources,results,*a,**kw:[{'from':s} for s in sources])
    def result(rec,auxiliary):
        assert auxiliary is False
        if rec['source_path']==2:raise ValueError('one bad wafer')
        return {'_aux_input':{},'chamber':None}
    ns=dict(_pause_gate=lambda:None,_result_for=result,ISCORE=SimpleNamespace(inspect_many=many),_WRITABLE=Path('.'),_load_app_config=lambda:{})
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<batch>','exec'),ns)
    out=list(ns['_results_with_aux']([dict(source_path=i) for i in range(65)],'DSV2'))
    assert len(out)==65 and out[2][2] is not None
    assert [len(c.args[0]) for c in many.call_args_list]==[31,32,1]
    assert all(r['chamber']=='DSV2' and '_aux_input' not in r and 'score_components' in r for _,r,e in out if e is None)


def test_occurrences_and_boundary():
    s=frame_sample();s.frame['ProcessStep']=[0,1,1,2,2,1,1,2,2,0,0,0]
    runs,_=scoring.sd.extract_step_runs(s.frame)
    assert [(r.step,r.occurrence) for r in runs]==[(1,1),(2,1),(1,2),(2,2)]
    ts=scoring.sd.parse_timestamps(s.frame)
    target=pv._run_series(s.frame,ts,runs[0],'p_008_PV')
    assert target[0][-1]==2 # next step boundary included for reach
    sensor=pv._resample_sensor(s.frame,ts,runs[0],'p_008_PV',np.array([0,1,2]))
    assert sensor[-1]==s.frame['p_008_PV'].iloc[2] # boundary excluded for candidate


def test_against_original_v021_explain(tmp_path,monkeypatch):
    import os,sys,importlib,csv
    reference=os.environ.get('SEMES_STEP_REFERENCE')
    if not reference:pytest.skip('set SEMES_STEP_REFERENCE to v0.2.1 checkout')
    monkeypatch.syspath_prepend(reference)
    upstream=importlib.import_module('step_time_iforest.pv_explainer')
    root=tmp_path/'model';make_step(root);_,info=pv_fixture(root)
    samples=[frame_sample(0),frame_sample(1)]
    # One delayed reach and one accelerated reach, both within step/boundary.
    samples[0].frame['p_008_PV']=np.arange(12,dtype=float)*.7
    samples[1].frame['p_008_PV']=np.arange(12,dtype=float)*1.5
    cache={}
    specs={'step-1__occ-1':dict(target_value=4,direction='up',grid_seconds=np.linspace(0,4,8))}
    for i,s in enumerate(samples):
        s.log_id=str(i);s.end_timestamp=__import__('pandas').Timestamp('2026-11-02')
        upstream._extract_frame_features(s,s.frame,specs,'p_008_PV',info['selected_sensors'],cache)
    report=tmp_path/'report';report.mkdir()
    upstream.explain_pv_explainer({'all':samples},root,report,feature_cache=cache)
    summaries=list(csv.DictReader((report/'pv_step_summary.csv').open(encoding='utf-8-sig')))
    influences=list(csv.DictReader((report/'pv_sensor_influences.csv').open(encoding='utf-8-sig')))
    scores=[scoring.step_time_score(root,s) for s in samples]
    pv.explain_batch(root,'all',list(zip(samples,scores)),scoring.load_artifacts(root,'step_time'),threading.RLock())
    for i,score in enumerate(scores):
        event=next(e for e in score['pv']['events'] if e['step']==1)
        reference_row=next(r for r in summaries if r['log_id']==str(i))
        for name in ['actual_reach_seconds','reach_delay_seconds','predicted_reach_seconds']:
            assert event[name]==pytest.approx(float(reference_row[name]))
        for c in event['candidates']:
            r=next(r for r in influences if r['log_id']==str(i) and r['sensor']==c['sensor'])
            assert c['contribution_seconds']==pytest.approx(float(r['reach_time_contribution_seconds']))
            assert c['response_status']==r['response_status']
        assert event['candidates'][0]['sensor']==reference_row['top_sensor']


def test_failed_pv_preserves_if_and_other_event(tmp_path):
    root=tmp_path/'m';make_step(root);pv_fixture(root)
    s=frame_sample();score=scoring.step_time_score(root,s);before=score['ratio']
    (root/'pv.joblib').write_text('corrupt')
    pv.explain_batch(root,'all',[(s,score)],scoring.load_artifacts(root,'step_time'),threading.RLock())
    assert score['available'] and score['ratio']==before
    assert not score['pv']['available']
    assert 'PV 설명 실패' in score['pv']['events'][0]['reason']


def test_missing_chamber_cannot_use_shared_model(tmp_path):
    common=tmp_path/'보조모델/스텝시간';make_step(common)
    result=scoring.inspect(None,dict(chamber='DSV2',ratio_thr=.8),tmp_path,{'step_time_artifacts':str(common)})
    assert not result['step_time']['available'] and result['integrated']['ratio']==.8
    assert 'DSV2' in result['step_time']['reason']


def test_v021_numerical_functions_are_unmodified():
    import os
    reference=os.environ.get('SEMES_STEP_REFERENCE')
    if not reference:pytest.skip('reference checkout unavailable')
    original=ast.parse((Path(reference)/'step_time_iforest/pv_explainer.py').read_text())
    actual=ast.parse(Path('src/app/auxiliary/step_pv_explainer.py').read_text())
    names={'_extract_frame_features','_rows_to_matrix','_batch_counterfactual_predictions','_response_status','_first_crossing','_run_series','_resample_sensor'}
    for name in names:
        a=next(n for n in actual.body if getattr(n,'name','')==name)
        b=next(n for n in original.body if getattr(n,'name','')==name)
        assert ast.dump(a)==ast.dump(b)
    original=ast.parse((Path(reference)/'step_time_iforest/data.py').read_text())
    actual=ast.parse(Path('src/app/auxiliary/step_data.py').read_text())
    for name in ['extract_step_runs','extract_step_durations']:
        assert ast.dump(next(n for n in actual.body if getattr(n,'name','')==name))==ast.dump(next(n for n in original.body if getattr(n,'name','')==name))


def test_archive_api_returns_saved_pv_without_source(tmp_path):
    root=tmp_path/'보조모델/스텝시간/DSV2';make_step(root);pv_fixture(root)
    s=frame_sample();score=scoring.step_time_score(root,s)
    pv.explain_batch(root,'all',[(s,score)],scoring.load_artifacts(root,'step_time'),threading.RLock())
    meta=dict(chamber='DSV2',ratio_thr=.8,score_components=scoring.combine({'available':True,'ratio':.8},score,scoring.unavailable('no cylinder')))
    # Same metadata storage format used by _archive_wafer; strictly JSON-safe.
    archive=tmp_path/'wafer.npz';np.savez_compressed(archive,meta_json=json.dumps(meta,allow_nan=False))
    with np.load(archive,allow_pickle=False) as z:stored=json.loads(str(z['meta_json']))
    tree=ast.parse(Path('src/app/server.py').read_text());node=next(n for n in tree.body if getattr(n,'name','')=='api_archive_score_components');node.decorator_list=[]
    ns=dict(_load_archive=lambda *a:{'meta':stored},ISCORE=scoring,_WRITABLE=tmp_path,_load_app_config=lambda:{})
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<api>','exec'),ns)
    result=ns['api_archive_score_components']('2026-11-02','missing-original-log')
    assert result['step_time']['pv']['events']==score['pv']['events']
    # A changed PV model is advisory; saved events remain historical facts.
    metadata=root/'pv_explainer_meta.json';metadata.write_text(metadata.read_text()+' ')
    result=ns['api_archive_score_components']('2026-11-02','missing-original-log')
    assert result['step_time']['available'] and result['step_time']['current_model_status']=='different'
    assert result['step_time']['pv']['events']==score['pv']['events']


def test_pv_cancellation_is_not_swallowed(tmp_path):
    class Stop(BaseException):pass
    root=tmp_path/'m';make_step(root);pv_fixture(root);s=frame_sample();score=scoring.step_time_score(root,s)
    def stop():raise Stop()
    with pytest.raises(Stop):pv.explain_batch(root,'all',[(s,score)],scoring.load_artifacts(root,'step_time'),threading.RLock(),stop)
