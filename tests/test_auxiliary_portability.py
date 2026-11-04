"""Cross-installation fixtures, including exact pre-portability Windows hashes."""
import hashlib
import json
import shutil
from pathlib import Path, PureWindowsPath
import pytest
from src.app import integrated_scores as scoring
from src.app.auxiliary.artifact_paths import artifact_file
from test_integrated_scores import sample, make_step, make_cylinder, scaling_info


def nested_paths(root, kind):
    filename = "step_time_meta.json" if kind == "step_time" else "golden_meta.json"
    meta = json.loads((root/filename).read_text())
    info = meta["models"]["all"]
    for key, folder in (("stats_path","stats"),("model_path","models")):
        old = root/info[key]
        if old.exists():
            (root/folder).mkdir(exist_ok=True)
            old.rename(root/folder/old.name)
            info[key] = folder + "\\" + old.name
    if kind == "cylinder":
        (root/"models").mkdir(exist_ok=True)
        (root/"model.onnx").rename(root/"models/model.onnx")
        manifest=json.loads((root/"onnx_manifest.json").read_text())
        manifest["models"]["all"]["onnx_path"]="models\\model.onnx"
        (root/"onnx_manifest.json").write_text(json.dumps(manifest))
    (root/filename).write_text(json.dumps(meta))


def old_windows_hash(root):
    files=sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in
                  (".json",".npz",".joblib",".onnx")),key=lambda p:PureWindowsPath(p.relative_to(root).as_posix()))
    h=hashlib.sha256()
    for p in files:
        h.update(str(PureWindowsPath(p.relative_to(root).as_posix())).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def fixture(root):
    step=root/"보조모델/스텝시간/DSV2";cylinder=root/"보조모델/실린더/DSV2"
    make_step(step);make_cylinder(cylinder)
    nested_paths(step,"step_time");nested_paths(cylinder,"cylinder")
    return step,cylinder


def test_paths_and_portable_versions_across_installations(tmp_path):
    one=tmp_path/"Windows_install";two=tmp_path/"Ubuntu_install"
    step,cylinder=fixture(one);shutil.copytree(one,two)
    assert scoring.artifact_digest(step)!=old_windows_hash(step)
    assert scoring.artifact_digests(step)["windows"]==old_windows_hash(step)
    a=scoring.versions(one,{})["DSV2"];b=scoring.versions(two,{})["DSV2"]
    assert a["step_time"]==b["step_time"] and a["cylinder"]==b["cylinder"]
    assert scoring.step_time_score(step,sample())["ratio"]==scoring.step_time_score(two/"보조모델/스텝시간/DSV2",sample())["ratio"]
    assert scoring.cylinder_score(cylinder,sample())["ratio"]==scoring.cylinder_score(two/"보조모델/실린더/DSV2",sample())["ratio"]


def test_legacy_windows_archive_reused_without_inference(tmp_path,monkeypatch):
    old=tmp_path/"win";new=tmp_path/"linux";step,cyl=fixture(old)
    st=scoring.step_time_score(step,sample());cy=scoring.cylinder_score(cyl,sample())
    st.pop("version_scheme");cy.pop("version_scheme")
    st["artifact_digest"]=old_windows_hash(step)
    cy["artifact_digest"]=old_windows_hash(cyl)+":"+scoring.CYLINDER_RATIO_CONTRACT
    cy["routing_version"]="old-hash-containing-Windows-installation"
    record={"chamber":"DSV2","ratio_thr":.4,"score_components":scoring.combine(scoring.tcn_component({"ratio_thr":.4}),st,cy)}
    shutil.copytree(old,new)
    monkeypatch.setattr(scoring,"step_time_score",lambda *args:pytest.fail("must not reinfer archive"))
    monkeypatch.setattr(scoring,"cylinder_score",lambda *args:pytest.fail("must not reinfer archive"))
    restored=scoring.current_components(record,scoring.versions(new,{})["DSV2"])
    assert restored["step_time"]["available"] and restored["cylinder"]["available"]
    assert restored["step_time"]["step_details"]==st["step_details"]
    assert restored["cylinder"]["step_sensor_details"]==cy["step_sensor_details"]
    assert restored["integrated"]["ratio"]==record["score_components"]["integrated"]["ratio"]
    (new/"보조모델/스텝시간/DSV2/models/model.joblib").write_bytes(b"changed")
    (new/"보조모델/실린더/DSV2/models/model.onnx").write_bytes(b"changed")
    changed=scoring.current_components(record,scoring.versions(new,{})["DSV2"])
    assert changed["step_time"]["available"] and changed["cylinder"]["available"]
    assert changed["step_time"]["current_model_status"]=="different" and changed["cylinder"]["current_model_status"]=="different"
    assert changed["tcn"]["available"]


def test_legacy_pre_scaling_only_allowed_for_unscaled_artifact(tmp_path):
    step,cyl=fixture(tmp_path)
    meta_path=cyl/"golden_meta.json";meta=json.loads(meta_path.read_text())
    meta["models"]["all"].update(scaling_info());meta_path.write_text(json.dumps(meta))
    record={"ratio_thr":.4,"score_components":{"cylinder":{"available":True,"ratio":2,
        "artifact_digest":old_windows_hash(cyl),"group":"all"}}}
    assert scoring.current_components(record,scoring.versions(tmp_path,{})["DSV2"])["cylinder"]["current_model_status"]=="different"
    record["score_components"]["cylinder"]["artifact_digest"]+=":"+scoring.CYLINDER_RATIO_CONTRACT
    assert scoring.current_components(record,scoring.versions(tmp_path,{})["DSV2"])["cylinder"]["available"]


def test_new_inference_and_archive_portability(tmp_path,monkeypatch):
    one=tmp_path/"win";two=tmp_path/"ubuntu";fixture(one);shutil.copytree(one,two)
    monkeypatch.setattr(scoring.sd,"read_process_file",lambda path:sample().frame)
    source=Path("WUK370A_PMCUNIT8_29a.csv");base={"chamber":"DSV2","ratio_thr":.4}
    first=scoring.inspect(source,base,one,{})
    second=scoring.inspect(source,base,two,{})
    assert first==second
    saved=dict(base,score_components=first)
    restored=scoring.current_components(saved,scoring.versions(two,{})["DSV2"])
    for kind in ("step_time","cylinder"):
        assert restored[kind]["ratio"]==first[kind]["ratio"] and restored[kind]["current_model_status"]=="current"


def test_default_source_folder_discovery_and_precedence(tmp_path):
    repo=tmp_path/"repo";src=repo/"src";src.mkdir(parents=True)
    step,cyl=fixture(repo)
    assert scoring.paths(src,{},"DSV2")[0]==step
    assert scoring.cylinder_candidates(src,{},"DSV2")[0]==cyl
    direct=src/"보조모델/스텝시간/DSV2";make_step(direct)
    assert scoring.paths(src,{},"DSV2")[0]==direct
    assert scoring.paths(src,{"step_time_by_chamber":{"DSV2":"..\\보조모델\\스텝시간\\DSV2"}},"DSV2")[0].resolve()==step.resolve()


@pytest.mark.parametrize("value",["../outside.npz","..\\outside.npz","C:\\model\\x.npz","/tmp/x.npz"])
def test_metadata_cannot_escape_package(tmp_path,value):
    with pytest.raises(ValueError):
        artifact_file(tmp_path,value)


def test_ubuntu_sklearn_pin():
    text=Path("src/app/requirements-ubuntu.txt").read_text()
    assert "\nscikit-learn==1.3.2\n" in text
    assert "scikit-learn>=" not in text


def test_pv_windows_metadata_paths(tmp_path):
    import threading
    from test_step_pv_integration import pv_fixture
    from src.app.auxiliary import step_pv_explainer as pv
    model,info=pv_fixture(tmp_path)
    (tmp_path/"models").mkdir();(tmp_path/"stats").mkdir()
    (tmp_path/"pv.joblib").rename(tmp_path/"models/pv.joblib")
    (tmp_path/"pv.npz").rename(tmp_path/"stats/pv.npz")
    info["model_path"]="models\\pv.joblib";info["stats_path"]="stats\\pv.npz"
    loaded,stats=pv._load_event(tmp_path,info,{},threading.RLock())
    assert loaded.n_features_in_==model.n_features_in_ and "sensors" in stats


def test_legacy_unscaled_archive(tmp_path):
    step,cyl=fixture(tmp_path)
    record={"ratio_thr":.4,"score_components":{"cylinder":{"available":True,"ratio":2,
        "artifact_digest":old_windows_hash(cyl),"group":"all"}}}
    assert scoring.current_components(record,scoring.versions(tmp_path,{})["DSV2"])["cylinder"]["available"]
