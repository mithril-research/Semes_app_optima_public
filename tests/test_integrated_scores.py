import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.app import integrated_scores as scoring


def sample():
    return SimpleNamespace(equipment="WUK370A", chamber="PMCUNIT7", recipe="29a",
        frame=pd.DataFrame({"Date":["2026-10-30"]*12,
            "Time":["00:00:%02d"%i for i in range(12)], "ProcessStep":[0]+[1]*5+[2]*5+[0],
            "d_004_PV":np.arange(12, dtype=float), "d_001_PV":np.arange(12, dtype=float)-1}))


def make_step(root):
    import joblib
    from sklearn.ensemble import IsolationForest
    root.mkdir(parents=True)
    train=np.asarray([[4+i*.01,5-i*.01] for i in range(30)],dtype=np.float32)
    model=IsolationForest(n_estimators=10, random_state=4).fit(train)
    joblib.dump(model,root/"model.joblib")
    np.savez(root/"stats.npz",normal_median=np.median(train,axis=0),
        normal_scale=np.ones(2),normal_q25=np.percentile(train,25,axis=0),normal_q75=np.percentile(train,75,axis=0))
    (root/"step_time_meta.json").write_text(json.dumps({"group_by":"all","models":{"all":{
        "steps":[1,2],"threshold":.6,"model_path":"model.joblib","stats_path":"stats.npz"}}}))
    return model


def make_cylinder(root,mode="legacy"):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    root.mkdir(parents=True)
    golden=np.zeros((1,8,3),np.float32)
    graph=helper.make_graph([
        helper.make_node("Constant",[],["golden"],value=numpy_helper.from_array(golden)),
        helper.make_node("Identity",["x"],["reconstruction"])],"golden",
        [helper.make_tensor_value_info("x",TensorProto.FLOAT,[1,8,3])],
        [helper.make_tensor_value_info("reconstruction",TensorProto.FLOAT,[1,8,3]),
         helper.make_tensor_value_info("golden",TensorProto.FLOAT,[1,8,3])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)]);model.ir_version=8
    onnx.save(model,str(root/"model.onnx"))
    np.savez(root/"stats.npz",model_golden_scaled=golden[0],model_golden_raw=golden[0],
        error_p95=np.ones((8,3),np.float32),step_feature_thresholds=np.full((2,3),2,np.float32),
        spike_step_feature_thresholds=np.full((2,3),3,np.float32),
        offset_step_feature_thresholds=np.full((2,3),4,np.float32),
        detrended_residual_center_raw=np.zeros((8,3),np.float32),
        detrended_residual_scale_raw=np.ones((8,3),np.float32),
        normal_offset_center_raw=np.zeros((2,3),np.float32))
    info={"model_path":"unused.pt","stats_path":"stats.npz","step_lengths":{"1":4,"2":4},
        "step_offsets":{"1":[0,4],"2":[4,8]},"standardizer":{"mean":[0,0,0],"std":[1,1,1]},
        "threshold":2,"spike_threshold":3,"offset_threshold":4,"spike_topk":1,"cylinder_score_mode":mode}
    (root/"golden_meta.json").write_text(json.dumps({"group_by":"all","feature_profile":"cylinder",
        "feature_names":scoring.gd.feature_names(False,"cylinder"),"models":{"all":info}}))
    (root/"onnx_manifest.json").write_text(json.dumps({"models":{"all":{
        "onnx_path":"model.onnx","input_name":"x","platform_score_output":"golden"}}}))


def test_combination_precision_missing_and_override():
    tcn=scoring.tcn_component({"score_raw":999,"threshold_raw":2,"ratio_thr":1.019999})
    c=scoring.combine(tcn,scoring.unavailable("missing"),{"available":True,"ratio":.9})
    assert c["integrated"]["ratio"]==1.019999
    assert c["integrated"]["source_model"]=="tcn"
    assert c["integrated"]["missing_models"]=={"step_time":"missing"}
    assert scoring.archived({"ratio_thr":.4})["integrated"]["available_models"]==["tcn"]


def test_step_matches_training_explain(tmp_path):
    model=make_step(tmp_path/"steps")
    s=sample()
    result=scoring.step_time_score(tmp_path/"steps",s)
    durations,_=scoring.sd.extract_step_durations(s.frame)
    x=np.asarray([[durations[1],durations[2]]],np.float32)
    assert result["score"]==pytest.approx(float(-model.score_samples(x)[0]))
    median=np.load(tmp_path/"steps/stats.npz")["normal_median"]
    for i,d in enumerate(result["step_details"]):
        counter=x.copy();counter[0,i]=median[i]
        assert d["contribution"]==pytest.approx(result["score"]+model.score_samples(counter)[0])
    s.frame=s.frame[s.frame.ProcessStep!=2]
    with pytest.raises(ValueError,match="스텝 구성"):scoring.step_time_score(tmp_path/"steps",s)


@pytest.mark.parametrize("mode",["legacy","offset_aware_mad","duration_aware_offset_mad","duration_classified_offset_mad"])
def test_cylinder_fixed_golden_modes(tmp_path,mode):
    root=tmp_path/mode;make_cylinder(root,mode)
    result=scoring.cylinder_score(root,sample())
    assert result["available"] and result["ratio"]>0  # reconstruction is exact input: would yield zero.
    assert len(result["step_sensor_details"])==6
    assert result["ratio"]==max(result[k] for k in ("shape_ratio","spike_ratio","offset_ratio") if k in result)
    assert result["step_sensor_details"][0]["golden"]==[0]*4


def test_no_chamber_fallback_and_digest_change(tmp_path):
    step,cy=scoring.paths(tmp_path,{},"DSV2")
    assert cy==tmp_path/"보조모델/실린더/DSV2"
    assert scoring.paths(tmp_path,{},"UNKNOWN")[1] is None
    result=scoring.inspect(None,{"chamber":"DSV2","ratio_thr":.8},tmp_path,{})
    assert result["integrated"]["ratio"]==.8
    assert not result["cylinder"]["available"]
    root=tmp_path/"step";make_step(root)
    before=scoring.artifact_digest(root)
    meta=root/"step_time_meta.json";meta.write_text(meta.read_text()+" ")
    assert scoring.artifact_digest(root)!=before


@pytest.mark.parametrize("mode",["legacy","offset_aware_mad","duration_aware_offset_mad","duration_classified_offset_mad"])
def test_upstream_explain_parity(tmp_path,mode):
    """Run the original explain numerical branch, not a reimplementation."""
    import os, sys, ast, inspect, importlib
    upstream=os.environ.get("SEMES_GOLDEN_REFERENCE")
    if not upstream:
        pytest.skip("set SEMES_GOLDEN_REFERENCE to the reviewed upstream checkout")
    sys.path.insert(0,upstream)
    reference=importlib.import_module("inception_golden.explain")
    root=tmp_path/mode;make_cylinder(root,mode)
    result=scoring.cylinder_score(root,sample())
    meta=json.loads((root/"golden_meta.json").read_text());info=meta["models"]["all"]
    with np.load(root/"stats.npz") as z: stats=dict(z)
    std=reference.Standardizer.from_json(info["standardizer"])
    raw=reference.align_sample(sample(),{int(k):v for k,v in info["step_lengths"].items()},False,"cylinder")
    x=std.transform(raw);golden=stats["model_golden_scaled"]
    tree=ast.parse(inspect.getsource(reference.explain))
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=="is_cylinder")
    env=dict(vars(reference),sample=sample(),info=info,stats=stats,standardizer=std,raw=raw,x=x,recon_scaled=golden,
        cell=np.abs(x-golden)/np.maximum(stats["error_p95"],1e-6),is_cylinder=True,steps=["1","2"],
        feature_names=meta["feature_names"],key="all",report_dir=tmp_path,
        threshold_files_written={"all"},spike_threshold_files_written={"all"},offset_threshold_files_written={"all"},
        offset_matrix=None,spike_matrix=None,spike_score=float("nan"),offset_score=float("nan"))
    exec(compile(ast.Module(body=[branch],type_ignores=[]),"<upstream-explain>","exec"),env)
    assert result["shape_ratio"]==pytest.approx(env["score"]/env["threshold"],rel=1e-6)
    assert result["spike_ratio"]==pytest.approx(env["spike_score"]/env["spike_threshold"],rel=1e-6)
    if mode!="legacy":
        assert result["offset_ratio"]==pytest.approx(env["offset_score"]/env["offset_threshold"],rel=1e-6)


def test_server_result_keeps_unrounded_tcn_component(tmp_path):
    import ast, math
    from src.wafer_score_override import AUDIT_FIELDS
    tree=ast.parse(Path("src/app/server.py").read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_result_for")
    node.decorator_list=[]
    ns=dict(Dict=dict,np=np,math=math,ISCORE=scoring,AUDIT_FIELDS=AUDIT_FIELDS,_WRITABLE=tmp_path,
        _assess=lambda rec:{"score":.123456789,"n_flag_sensor":0},
        _score_view=lambda *a:{"score":.99,"threshold":1,"recipe":"29a","verdict":"OK",
                "effective_score_raw":.123456789,"score_raw":.123,"threshold_raw":.124},
        _weighted_score_for_rec=lambda rec:None,_chamber=lambda wid:"DSV1",
        _lot_of=lambda wid:"lot",_end_ts_and_date=lambda wid:("2026-10-30", "2026-10-30"),
        _ntp_from_rec=lambda rec:12,_recipe_threshold=lambda wid:("29a",.12456789),
        _load_app_config=lambda:{})
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<server-result>","exec"),ns)
    result=ns["_result_for"]({"wafer_id":"wafer"})
    assert result["score"]==.99
    assert result["score_components"]["tcn"]["ratio"]==.123456789/.12456789
    assert result["score_components"]["integrated"]["available_models"]==["tcn"]


def test_artifact_replacement_retains_stored_component():
    original={"ratio_thr":.4,"score_components":{"step_time":{"available":True,"ratio":2,"artifact_digest":"old"}}}
    c=scoring.current_components(original,{"step_time":"new"})
    assert c["integrated"]["ratio"]==2
    assert c["step_time"]["available"] and c["step_time"]["current_model_status"]=="missing"
    assert original["score_components"]["step_time"]["ratio"]==2


def scaling_info(**overrides):
    value = dict(method="linear_threshold_anchored", source="train_normal",
                 input_score="max_threshold_ratio", lower_anchor=.4, threshold_anchor=1.,
                 denominator=.6, output_threshold=1., clip_below_zero=True,
                 clip_above=False, version="v1")
    value.update(overrides)
    return {"combined_score_scaling": value}


@pytest.mark.parametrize("raw,expected",[(.4,0),(.1,0),(1,1),(1.6,2),(2.8,4)])
def test_cylinder_scaling_anchors(raw, expected):
    value, applied = scoring.scale_cylinder_ratio(raw, scaling_info())
    assert applied and value == pytest.approx(expected)


@pytest.mark.parametrize("overrides",[
    {"denominator":0},{"denominator":.7},{"lower_anchor":float("nan")},
    {"clip_above":True},{"threshold_anchor":2},{"output_threshold":2},
    {"clip_below_zero":False},{"source":"target"},{"input_score":"shape"},
])
def test_cylinder_scaling_invalid_contract(overrides):
    with pytest.raises(ValueError):
        scoring.scale_cylinder_ratio(1.2, scaling_info(**overrides))


def test_cylinder_scaling_absent_only_is_legacy():
    assert scoring.scale_cylinder_ratio(1.23456789,{}) == (1.23456789,False)
    with pytest.raises(ValueError):
        scoring.scale_cylinder_ratio(1,{"combined_score_scaling":None})


def test_cylinder_scaling_once_and_cell_scores_unchanged(tmp_path):
    root=tmp_path/"cylinder";make_cylinder(root)
    original=scoring.cylinder_score(root,sample())
    path=root/"golden_meta.json";meta=json.loads(path.read_text())
    meta["models"]["all"].update(scaling_info());path.write_text(json.dumps(meta))
    scaled=scoring.cylinder_score(root,sample())
    assert scaled["ratio"] == pytest.approx((original["ratio"]-.4)/.6)
    assert scaled["raw_ratio"] == original["ratio"]
    assert scaled["step_sensor_details"] == original["step_sensor_details"]
    assert scaled["shape_ratio"] == original["shape_ratio"]
    assert scaled["spike_ratio"] == original["spike_ratio"]
    assert scaled["scaling_applied"] and scaled["scaling_version"]=="v1"
    assert scaled["artifact_digest"].endswith(":"+scoring.CYLINDER_RATIO_CONTRACT)


def test_cylinder_package_priority_and_complete_sets(tmp_path):
    parent=tmp_path/"보조모델/실린더";make_cylinder(parent)
    dsv=parent/"DSV2";make_cylinder(dsv)
    explicit=tmp_path/"custom";make_cylinder(explicit)
    config={"cylinder_by_chamber":{"DSV2":"custom"}}
    candidates=scoring.cylinder_candidates(tmp_path,config,"DSV2")
    assert scoring.cylinder_package(candidates,sample())==explicit
    (explicit/"model.onnx").unlink()
    assert scoring.cylinder_package(candidates,sample())==dsv
    (dsv/"stats.npz").unlink()
    assert scoring.cylinder_package(candidates,sample())==parent
    (parent/"onnx_manifest.json").unlink()
    with pytest.raises(ValueError,match="완전한"):
        scoring.cylinder_package(candidates,sample())


def test_cylinder_never_uses_other_chamber(tmp_path):
    parent=tmp_path/"보조모델/실린더";make_cylinder(parent/"DSV1")
    candidates=scoring.cylinder_candidates(tmp_path,{},"DSV2")
    assert parent/"DSV1" not in candidates
    with pytest.raises(ValueError):
        scoring.cylinder_package(candidates,sample())
    assert not scoring.cylinder_candidates(tmp_path,{},"UNKNOWN")


def test_cylinder_group_specific_fallback_and_package_escape(tmp_path):
    parent=tmp_path/"보조모델/실린더";make_cylinder(parent)
    dsv=parent/"DSV2";make_cylinder(dsv)
    path=dsv/"golden_meta.json";meta=json.loads(path.read_text())
    info=meta["models"].pop("all");meta["group_by"]="recipe"
    meta["models"]["recipe-OTHER"]=info;path.write_text(json.dumps(meta))
    candidates=scoring.cylinder_candidates(tmp_path,{},"DSV2")
    assert scoring.cylinder_package(candidates,sample())==parent
    meta["group_by"]="all";meta["models"]={"all":dict(info,stats_path="../stats.npz")}
    path.write_text(json.dumps(meta))
    assert scoring.cylinder_package(candidates,sample())==parent


def test_scaling_error_isolated_and_history_invalidated(tmp_path,monkeypatch):
    parent=tmp_path/"보조모델/실린더";make_cylinder(parent)
    make_step(tmp_path/"보조모델/스텝시간/DSV2")
    monkeypatch.setattr(scoring.sd,"read_process_file",lambda path:sample().frame)
    result={"chamber":"DSV2","ratio_thr":.8}
    source=Path("WUK370A_PMCUNIT8_29a.csv")
    components=scoring.inspect(source,result,tmp_path,{})
    assert components["cylinder"]["available"]
    saved=dict(result,score_components=components)
    versions=scoring.versions(tmp_path,{})["DSV2"]
    assert scoring.current_components(saved,versions)["cylinder"]["available"]
    old=json.loads(json.dumps(saved))
    old["score_components"]["cylinder"].pop("routing_version")
    old["score_components"]["cylinder"]["artifact_digest"]=scoring.artifact_digest(parent)
    assert scoring.current_components(old,versions)["cylinder"]["available"]
    path=parent/"golden_meta.json";meta=json.loads(path.read_text())
    meta["models"]["all"].update(scaling_info(denominator=0));path.write_text(json.dumps(meta))
    failed=scoring.inspect(source,result,tmp_path,{})
    assert not failed["cylinder"]["available"]
    assert failed["tcn"]["available"] and failed["step_time"]["available"]
    assert "scaling" in failed["cylinder"]["reason"]
    assert scoring.current_components(saved,scoring.versions(tmp_path,{})["DSV2"])["cylinder"]["current_model_status"]=="different"


def test_archive_endpoint_uses_same_contract_check(tmp_path):
    import ast
    tree=ast.parse(Path("src/app/server.py").read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="api_archive_score_components")
    node.decorator_list=[]
    meta={"chamber":"DSV2","ratio_thr":.5,"score_components":{"cylinder":{
        "available":True,"ratio":2,"artifact_digest":"old"}}}
    ns=dict(ISCORE=scoring,_WRITABLE=tmp_path,_load_app_config=lambda:{},
            _load_archive=lambda *args:{"meta":meta})
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<archive>", "exec"),ns)
    response=ns["api_archive_score_components"]("2026-11-03","wafer")
    assert response["cylinder"]["available"] and response["integrated"]["ratio"]==2
    assert response["cylinder"]["current_model_status"]=="missing"


def test_new_chamber_package_invalidates_shared_routing(tmp_path, monkeypatch):
    parent=tmp_path/"보조모델/실린더";make_cylinder(parent)
    monkeypatch.setattr(scoring.sd,"read_process_file",lambda path:sample().frame)
    record={"chamber":"DSV2","ratio_thr":.8}
    record["score_components"]=scoring.inspect(Path("WUK370A_PMCUNIT8_29a.csv"),record,tmp_path,{})
    assert record["score_components"]["cylinder"]["available"]
    make_cylinder(parent/"DSV2")
    current=scoring.current_components(record,scoring.versions(tmp_path,{})["DSV2"])
    assert current["cylinder"]["available"] and current["cylinder"]["current_model_status"]=="different"
    assert current["tcn"]["available"]


def test_historical_npz_is_fact_not_current_inference(tmp_path,monkeypatch):
    import copy
    saved={"tcn":{"available":True,"ratio":.7},
           "step_time":{"available":True,"ratio":1.4,"score":.7,"threshold":.5,"pv":{"events":[{"saved":True}]}},
           "cylinder":{"available":True,"ratio":2.3,"raw_ratio":1.8,"artifact_digest":"old","step_sensor_details":[{"raw":[1,2],"golden":[2,3]}]},
           "integrated":{"ratio":2.31,"source_model":"cylinder","available_models":["tcn","step_time","cylinder"],"missing_models":{}}}
    meta={"ratio_thr":99,"score_components":saved}
    archive=tmp_path/"old.npz";np.savez_compressed(archive,meta_json=json.dumps(meta))
    before=archive.read_bytes()
    with np.load(archive,allow_pickle=False) as z:record=json.loads(str(z["meta_json"]))
    untouched=copy.deepcopy(record)
    monkeypatch.setattr(scoring,"inspect",lambda *args:pytest.fail("archive must not infer"))
    for current in ({},scoring.versions(tmp_path,{})["DSV2"],{"step_time":"new","cylinder":"new","legacy_step_time":["new"],"legacy_cylinder":[{"models":{"all":True},"digests":["new"]}]}):
        result=scoring.current_components(record,current)
        assert result["integrated"]["ratio"]==2.31 and result["tcn"]["ratio"]==.7
        assert result["step_time"]["pv"]==saved["step_time"]["pv"]
        assert result["cylinder"]["step_sensor_details"]==saved["cylinder"]["step_sensor_details"]
        assert result["step_time"]["available"] and result["cylinder"]["available"]
    assert archive.read_bytes()==before and record==untouched


def test_no_components_and_status_failure_do_not_infer(tmp_path,monkeypatch):
    empty=scoring.current_components({"ratio_thr":.5},{})
    assert not empty["step_time"]["available"] and "과거 이력" in empty["step_time"]["reason"]
    monkeypatch.setattr(scoring,"versions",lambda *args:(_ for _ in ()).throw(PermissionError("blocked")))
    assert scoring.status_versions(tmp_path,{})=={}


@pytest.mark.parametrize("kind", ["shape", "spike", "offset"])
def test_cylinder_zero_semantics(kind):
    assert scoring.cylinder_ratio(0, 0, "group", kind) == 0
    assert scoring.cylinder_ratio(-1, 0, "group", kind) == 0
    assert np.isposinf(scoring.cylinder_ratio(1, 0, "group", kind))
    assert scoring.cylinder_ratio(6, 2, "group", kind) == 3
    for score, threshold in [(1, -1), (np.nan, 1), (1, np.inf)]:
        with pytest.raises(ValueError, match="group"):
            scoring.cylinder_ratio(score, threshold, "group", kind, 2, "C_DIFF")
    with pytest.raises(ValueError):
        scoring.ratio(1, 0)


@pytest.mark.parametrize("kind,key,cellkey", [
    ("shape", "threshold", "step_feature_thresholds"),
    ("spike", "spike_threshold", "spike_step_feature_thresholds"),
    ("offset", "offset_threshold", "offset_step_feature_thresholds")])
def test_zero_threshold_inference_and_archive(tmp_path, kind, key, cellkey):
    root=tmp_path/"model";make_cylinder(root,"offset_aware_mad")
    meta=json.loads((root/"golden_meta.json").read_text())
    meta["models"]["all"][key]=0
    (root/"golden_meta.json").write_text(json.dumps(meta))
    with np.load(root/"stats.npz",allow_pickle=False) as z:arrays=dict(z)
    arrays[cellkey][:]=0
    np.savez(root/"stats.npz",**arrays)
    result=scoring.cylinder_score(root,sample())
    assert result["available"] and result["ratio_infinite"]
    assert result["ratio"] is None and result["threshold_zero_exceeded"]
    assert result["ratio_states"][kind]["ratio_infinite"]
    assert any(d["ratio_infinite"] for d in result["step_sensor_details"])
    combined=scoring.combine({"available":True,"ratio":999},scoring.unavailable("missing"),result)
    assert combined["integrated"]["source_model"]=="cylinder"
    assert combined["integrated"]["verdict"]=="NG"
    payload=json.dumps({"score_components":combined},allow_nan=False)
    np.savez(tmp_path/"archive.npz",meta_json=np.array(payload))
    with np.load(tmp_path/"archive.npz",allow_pickle=False) as z:
        restored=scoring.current_components(json.loads(str(z["meta_json"])), {})
    assert restored["integrated"]["ratio_infinite"]
    assert restored["cylinder"]["step_sensor_details"]==result["step_sensor_details"]
    json.dumps(restored,allow_nan=False)


def test_zero_threshold_scaling_contract():
    info={"combined_score_scaling":{"method":"linear_threshold_anchored","source":"train_normal",
        "input_score":"max_threshold_ratio","lower_anchor":.2,"threshold_anchor":1.,
        "denominator":.8,"output_threshold":1.,"clip_above":False,"clip_below_zero":True}}
    assert scoring.scale_cylinder_ratio(float("inf"),info)==(float("inf"),True)
    info["combined_score_scaling"]["denominator"]=0
    with pytest.raises(ValueError):scoring.scale_cylinder_ratio(float("inf"),info)
