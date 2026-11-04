import json
import numpy as np
import pytest
from src.app.check_cylinder_thresholds import inspect_array, inspect_package, main
from src.app import integrated_scores as scoring
from test_integrated_scores import make_cylinder, sample


def test_counts_and_indices():
    summary=inspect_array(np.array([[0.,-1.,np.nan],[np.inf,2.,3.]]),2)
    assert summary["zero"]==1 and summary["negative"]==1 and summary["nonfinite"]==2
    assert summary["min"]==-1 and summary["max"]==3 and summary["bad_count"]==4
    assert summary["bad"][0][0]==(0,0) and len(summary["bad"])==2
    assert inspect_array(np.array(0.))["bad_count"]==1


def test_full_group_report_is_read_only(tmp_path):
    root=tmp_path/"model";make_cylinder(root)
    report,issues=inspect_package(root)
    assert issues==0 and "GROUP: all" in report
    (root/"stats").mkdir();(root/"stats.npz").rename(root/"stats/good.npz")
    meta=json.loads((root/"golden_meta.json").read_text())
    info=meta["models"]["all"];info["stats_path"]="stats\\good.npz"
    meta["models"]["broken"]=dict(info,stats_path="missing.npz")
    (root/"golden_meta.json").write_text(json.dumps(meta))
    with np.load(root/"stats/good.npz",allow_pickle=False) as z:arrays=dict(z)
    arrays["spike_step_feature_thresholds"][1,2]=-1
    np.savez(root/"stats/good.npz",**arrays)
    before={p:p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report,issues=inspect_package(root,show_arrays=True)
    assert issues==2
    assert "스텝 2" in report and scoring.gd.feature_names(False,"cylinder")[2] in report
    assert "GROUP: broken" in report and "그룹 검사 실패" in report
    assert all(p.read_bytes()==content for p,content in before.items())
    output=tmp_path/"report.txt"
    assert main(["--root",str(root),"--group","all","--output",str(output)])==1
    assert output.read_text(encoding="utf-8")
    assert main(["--root",str(tmp_path/"absent")])==2


def test_invalid_cell_error_identifies_group_step_sensor(tmp_path):
    root=tmp_path/"model";make_cylinder(root)
    with np.load(root/"stats.npz",allow_pickle=False) as z:arrays=dict(z)
    arrays["spike_step_feature_thresholds"][1,2]=-1
    np.savez(root/"stats.npz",**arrays)
    with pytest.raises(ValueError) as error:
        scoring.cylinder_score(root,sample())
    message=str(error.value)
    assert "all" in message and "spike_step_feature_thresholds" in message
    assert "스텝 2" in message and "threshold=-1.0" in message


def test_missing_optional_arrays_not_required_for_legacy(tmp_path):
    root=tmp_path/"model";make_cylinder(root)
    with np.load(root/"stats.npz",allow_pickle=False) as z:arrays=dict(z)
    arrays.pop("detrended_residual_scale_raw");arrays.pop("offset_step_feature_thresholds")
    np.savez(root/"stats.npz",**arrays)
    assert inspect_package(root)[1]==0
