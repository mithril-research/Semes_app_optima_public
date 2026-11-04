"""TCN + StepTime IsolationForest + chamber-specific Golden ONNX inference.

Missing/incompatible components are unavailable, never synthesized. Original TCN
fields remain untouched. Artifact folders are versioned as a whole.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import threading
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
import numpy as np
from src import sensor_names as SN
from .auxiliary import golden_data as gd, golden_scoring as gs, step_data as sd
from .auxiliary.artifact_paths import artifact_file

_LOCK = threading.RLock()
_CACHE = {}
_REPORTED = set()
VERSION_SCHEME = "portable-v2"
CYLINDER_RATIO_CONTRACT = "train-normal-scaling-v2-zero-threshold"



def report_unavailable(kind, reason):
    key = (kind, str(reason))
    with _LOCK:
        if key in _REPORTED:
            return
        if len(_REPORTED) >= 256:
            _REPORTED.clear()
        _REPORTED.add(key)
    print("[보조모델 미적용] %s: %s" % key)


def unavailable(reason):
    return {"available": False, "ratio": None, "reason": str(reason)}


def ratio(score, threshold):
    score, threshold = float(score), float(threshold)
    if not np.isfinite(score) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("유효한 양수 임계값이 필요합니다")
    return score / threshold


def safe_ratio(value):
    infinite = bool(np.isposinf(value))
    return {"ratio": None if infinite else float(value), "ratio_infinite": infinite,
            "threshold_zero_exceeded": infinite}


def combine(tcn, step_time, cylinder):
    components = {"tcn": tcn, "step_time": step_time, "cylinder": cylinder}
    valid = {k: float(v["ratio"]) for k, v in components.items()
             if v.get("available") and v.get("ratio") is not None and np.isfinite(v["ratio"])}
    if cylinder.get("available") and cylinder.get("ratio_infinite"):
        valid["cylinder"] = float("inf")
    winner = max(valid, key=valid.get) if valid else None
    return dict(components, integrated={
        **(safe_ratio(valid[winner]) if winner else {"ratio": None}), "source_model": winner,
        "verdict": "NG" if winner and valid[winner] >= 1 else "OK" if winner else None,
        "available_models": list(valid),
        "missing_models": {k: v.get("reason") or "점수 없음" for k, v in components.items() if k not in valid}})


def tcn_component(result):
    try:
        # ratio_thr already includes recipe multipliers/wafer overrides.
        value = result.get("ratio_thr")
        if value is None:
            value = ratio(result["score_raw"], result["threshold_raw"])
        if not np.isfinite(float(value)):
            raise ValueError("TCN 점수 없음")
        return {"available": True, "reason": "", "score": result.get("score_raw"),
                "threshold": result.get("threshold_raw"), "ratio": float(value)}
    except (ValueError, TypeError, KeyError) as exc:
        return unavailable(exc)


def artifact_directory(base, filename):
    """Metadata location defines the artifact root, not a literal folder name."""
    base = Path(base)
    if (base / filename).is_file():
        return base
    # One optional packaging directory; never search other DSV siblings.
    if base.is_dir():
        matches = sorted(p.parent for p in base.glob("*/" + filename) if p.is_file())
        if len(matches) == 1:
            return matches[0]
    return base



def default_aux_parent(root, folder):
    """Source: src/보조모델 first, repository-level copy second. Exe: beside exe."""
    base = Path(root).resolve() / "보조모델" / folder
    copied = Path(root).resolve().parent / "보조모델" / folder
    return base if base.exists() or not copied.exists() else copied


def paths(root, config, chamber):
    root = Path(root)
    chamber = str(chamber or "").upper()
    if chamber not in {"DSV%d" % i for i in range(1, 7)}:
        return None, None
    def resolve(value):
        p = Path(str(value).replace("\\", "/"))
        return p if p.is_absolute() else root / p
    found = []
    for kind, folder, filename, env in (
        ("step_time", "스텝시간", "step_time_meta.json", "IO_STEP_TIME_ARTIFACTS_ROOT"),
        ("cylinder", "실린더", "golden_meta.json", "IO_CYLINDER_ARTIFACTS_ROOT"),
    ):
        parent = config.get(kind + "_artifacts_root") or os.getenv(env) or str(default_aux_parent(root, folder))
        value = (config.get(kind + "_by_chamber") or {}).get(chamber) or str(Path(parent) / chamber)
        found.append(artifact_directory(resolve(value), filename))
    return tuple(found)



def cylinder_candidates(root, config, chamber):
    """Ordered whole packages; shared root must never resolve to another DSV."""
    chamber = str(chamber or "").upper()
    if chamber not in {"DSV%d" % i for i in range(1, 7)}:
        return []
    root = Path(root)
    def resolve(value):
        p = Path(str(value).replace("\\", "/"))
        return p if p.is_absolute() else root / p
    parent = resolve(config.get("cylinder_artifacts_root") or
                     os.getenv("IO_CYLINDER_ARTIFACTS_ROOT") or str(default_aux_parent(root, "실린더")))
    explicit = (config.get("cylinder_by_chamber") or {}).get(chamber)
    candidates = []
    for base in ([resolve(explicit)] if explicit else []) + [parent / chamber]:
        candidate = artifact_directory(base, "golden_meta.json")
        if candidate not in candidates:
            candidates.append(candidate)
    # Deliberately no artifact_directory(parent): a sole DSV child is NOT shared.
    if parent not in candidates:
        candidates.append(parent)
    return candidates


def cylinder_sample(meta, sample):
    result = SimpleNamespace(**vars(sample))
    if getattr(result, "path", None):
        result.equipment = gd.infer_equipment(result.path, meta.get("equipment_regex"))
        result.chamber = gd.infer_chamber(result.path, meta.get("chamber_regex") or r"PMCUNIT\d+")
    return result


def cylinder_package(candidates, sample):
    failures = []
    for directory in candidates:
        try:
            bundle = load_artifacts(directory, "cylinder")
            key = group_key(bundle["meta"], cylinder_sample(bundle["meta"], sample))
            info = bundle["meta"]["models"][key]
            manifest = json.loads((directory / "onnx_manifest.json").read_text(encoding="utf-8-sig"))["models"][key]
            # Do not combine files from separate releases, even through ../ paths.
            for name in (info["stats_path"], manifest["onnx_path"]):
                path = artifact_file(directory, name)
                path.relative_to(directory.resolve())
                if not path.is_file():
                    raise ValueError("필수 파일 없음: " + str(path))
            return directory
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            failures.append(str(directory) + ": " + str(exc))
    raise ValueError("완전한 실린더 모델 세트 없음 · " + " / ".join(failures))


def component_version(root, kind):
    digest = artifact_digest(root)
    return digest + ":" + CYLINDER_RATIO_CONTRACT if kind == "cylinder" else digest


def cylinder_routing_version(candidates):
    # Include ordered search locations: adding/removing a higher priority package
    # must invalidate old shared-model scores, even with unchanged shared files.
    signature = [("priority-%d" % i, component_version(p, "cylinder")) for i, p in enumerate(candidates)]
    return VERSION_SCHEME + ":" + hashlib.sha256(json.dumps(signature).encode()).hexdigest() + ":" + CYLINDER_RATIO_CONTRACT


def scale_cylinder_ratio(raw_ratio, info):
    if "combined_score_scaling" not in info:
        return raw_ratio, False
    scaling = info["combined_score_scaling"]
    if not isinstance(scaling, dict) or scaling.get("method") != "linear_threshold_anchored":
        raise ValueError("지원하지 않는 실린더 scaling 방식")
    if scaling.get("source") != "train_normal" or scaling.get("input_score") != "max_threshold_ratio":
        raise ValueError("실린더 scaling 학습 기준 계약 오류")
    lower, anchor, denominator, output = (float(scaling[k]) for k in
        ("lower_anchor", "threshold_anchor", "denominator", "output_threshold"))
    if not all(np.isfinite(v) for v in (lower, anchor, denominator, output)):
        raise ValueError("실린더 scaling 값이 유한하지 않음")
    if denominator <= 0 or not np.isclose(denominator, anchor-lower, rtol=1e-6, atol=1e-8):
        raise ValueError("실린더 scaling anchor/denominator 불일치")
    if not np.isclose(anchor, 1.0) or not np.isclose(output, 1.0):
        raise ValueError("통합 점수 판정 기준 1 계약 불일치")
    if scaling.get("clip_above") is not False or scaling.get("clip_below_zero") is not True:
        raise ValueError("실린더 clipping 계약 오류")
    if np.isposinf(raw_ratio):
        return float("inf"), True
    scaled = max(0.0, (float(raw_ratio)-lower)/denominator)
    if not np.isfinite(scaled):
        raise ValueError("실린더 scaling 결과가 유한하지 않음")
    return scaled, True


def artifact_digests(root):
    """Canonical POSIX version and exact legacy Linux/Windows hash variants."""
    if root is None or not root.is_dir():
        return {"canonical": "missing", "posix": "missing", "windows": "missing"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".json", ".npz", ".joblib", ".onnx")]
    canonical = sorted(files, key=lambda p: p.relative_to(root).as_posix())
    signature = tuple((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in canonical)
    cache_key = ("portable-digests", str(root))
    with _LOCK:
        cached = _CACHE.get(cache_key)
        if cached and cached[0] == signature:
            return cached[1]
    result = {}
    orders = {"canonical": canonical, "posix": sorted(files, key=lambda p: p.relative_to(root).parts),
              "windows": sorted(files, key=lambda p: PureWindowsPath(p.relative_to(root).as_posix()))}
    for kind, ordered in orders.items():
        if kind == "posix" and ordered == canonical:
            result[kind] = result["canonical"]
            continue
        h = hashlib.sha256()
        for p in ordered:
            name = p.relative_to(root).as_posix()
            if kind == "windows":
                name = name.replace("/", "\\")
            h.update(name.encode())
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        result[kind] = h.hexdigest()
    with _LOCK:
        _CACHE[cache_key] = (signature, result)
    return result


def artifact_digest(root):
    return artifact_digests(root)["canonical"]


def load_artifacts(root, kind):
    filename = "step_time_meta.json" if kind == "step_time" else "golden_meta.json"
    if root is None or not (root / filename).is_file():
        raise ValueError(("스텝 시간" if kind == "step_time" else "실린더") + " 모델 없음: " + (str(root / filename) if root is not None else "DSV 챔버 미확인"))
    digest = artifact_digest(root)
    key = (kind, str(root), digest + ":" + SN.mapping_signature())
    with _LOCK:
        if key not in _CACHE:
            meta = SN.normalize(json.loads((root / filename).read_text(encoding="utf-8-sig")))
            _CACHE[key] = {"meta": meta, "groups": {}, "digest": digest}
            stale = [k for k in _CACHE if len(k) == 3 and k[:2] == key[:2] and k != key]
            for old in stale:
                del _CACHE[old]
        return _CACHE[key]


def group_key(meta, sample):
    group = meta.get("group_by")
    if not group:
        key = "equipment" if meta.get("scope") == "equipment" else sample.chamber
    elif group == "all":
        key = "all"
    else:
        names = {"equipment": ["equipment"], "chamber": ["chamber"], "recipe": ["recipe"],
                 "equipment_recipe": ["equipment", "recipe"], "chamber_recipe": ["chamber", "recipe"],
                 "equipment_chamber_recipe": ["equipment", "chamber", "recipe"]}[group]
        key = "__".join(name + "-" + (re.sub(r"[^0-9A-Za-z_.-]+", "-", str(getattr(sample, name))).strip("-") or "unknown") for name in names)
    if key not in meta["models"]:
        raise ValueError("일치하는 모델 그룹 없음: " + key + " · 모델 그룹 예: " + ", ".join(list(meta["models"])[:5]))
    return key


def step_time_score(root, sample):
    bundle = load_artifacts(root, "step_time")
    meta = bundle["meta"]
    sample = SimpleNamespace(**vars(sample))
    if getattr(sample, "path", None):
        sample.equipment = sd.infer_equipment(sample.path, meta.get("equipment_regex"))
        sample.chamber = sd.infer_chamber(sample.path, meta.get("chamber_regex") or r"PMCUNIT\d+")
    key = group_key(meta, sample)
    info = meta["models"][key]
    durations, _ = sd.extract_step_durations(sample.frame)
    steps = [int(v) for v in info["steps"]]
    if set(steps) != set(durations):
        raise ValueError("스텝 구성 불일치: 누락=%s 추가=%s" % (sorted(set(steps)-set(durations)), sorted(set(durations)-set(steps))))
    with _LOCK:
        if key not in bundle["groups"]:
            import joblib
            import warnings
            from sklearn.exceptions import InconsistentVersionWarning
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                model = joblib.load(artifact_file(root, info["model_path"]))
            with np.load(artifact_file(root, info["stats_path"]), allow_pickle=False) as z:
                stats = {k: z[k] for k in z.files}
            bundle["groups"][key] = (model, stats)
        model, stats = bundle["groups"][key]
    x = np.asarray([[durations[k] for k in steps]], dtype=np.float32)
    score = float(-model.score_samples(x)[0])
    threshold = float(info["threshold"])
    total_ratio = ratio(score, threshold)
    median = np.asarray(stats["normal_median"], dtype=np.float32)
    scale = np.asarray(stats["normal_scale"], dtype=np.float32)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(median)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("정상 시간 scale 오류")
    counter = np.repeat(x, len(steps), axis=0)
    counter[np.arange(len(steps)), np.arange(len(steps))] = median
    contribution = score + model.score_samples(counter)
    details = []
    for i, step in enumerate(steps):
        dev = float(abs(x[0, i]-median[i])/scale[i])
        details.append({"step": step, "actual": float(x[0, i]), "median": float(median[i]),
                        "q25": float(stats["normal_q25"][i]), "q75": float(stats["normal_q75"][i]),
                        "robust_deviation": dev, "contribution": float(contribution[i]),
                        "ratio": dev, "label": "ProcessStep %s 소요시간" % step})
    return {"available": True, "reason": "", "score": score, "threshold": threshold, "ratio": total_ratio,
            "step_details": details, "artifact_digest": bundle["digest"], "version_scheme": VERSION_SCHEME, "group": key}



def cylinder_ratio(score, threshold, group, kind, step=None, sensor=None):
    try:
        score, threshold = float(score), float(threshold)
        if not np.isfinite(score) or not np.isfinite(threshold) or threshold < 0:
            raise ValueError("유한 점수와 0 이상 임계값이 필요합니다")
        if threshold == 0:
            return float("inf") if score > 0 else 0.0
        value = score / threshold
        if not np.isfinite(value):
            raise ValueError("점수 비율 범위 초과")
        return value
    except (ValueError, TypeError, OverflowError) as exc:
        where = "실린더 그룹 %s · %s · %s" % (group, kind, "전체" if step is None else "셀")
        if step is not None:
            where += " · 스텝 %s · 센서 %s" % (step, sensor)
        raise ValueError("%s (score=%s, threshold=%s): %s" % (where, score, threshold, exc)) from exc


def cylinder_score(root, sample):
    bundle = load_artifacts(root, "cylinder")
    meta = bundle["meta"]
    if meta.get("feature_profile", "cylinder") != "cylinder":
        raise ValueError("실린더 전용 artifact가 아닙니다")
    sample = cylinder_sample(meta, sample)
    key = group_key(meta, sample)
    info = meta["models"][key]
    manifest = json.loads((root / "onnx_manifest.json").read_text(encoding="utf-8-sig"))["models"][key]
    if manifest.get("platform_score_output") != "golden":
        raise ValueError("ONNX golden 출력 계약 없음")
    with _LOCK:
        if key not in bundle["groups"]:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(artifact_file(root, manifest["onnx_path"])), sess_options=options, providers=["CPUExecutionProvider"])
            with np.load(artifact_file(root, info["stats_path"]), allow_pickle=False) as z:
                stats = {k: z[k] for k in z.files}
            bundle["groups"][key] = (session, stats)
        session, stats = bundle["groups"][key]
    features = list(meta["feature_names"])
    if features != gd.feature_names(False, "cylinder"):
        raise ValueError("지원하지 않는 실린더 feature 순서: " + str(features))
    lengths = {int(k): int(v) for k, v in info["step_lengths"].items()}
    if 0 in lengths or set(lengths) != set(int(v) for v in sample.frame.ProcessStep if int(v) > 0):
        raise ValueError("실린더 모델 스텝 구성 불일치")
    std = gd.Standardizer.from_json(info["standardizer"])
    raw = gd.align_sample(sample, lengths, False, "cylinder")
    x = std.transform(raw).astype(np.float32)
    golden = session.run(["golden"], {manifest.get("input_name", "x"): x[None]})[0][0]
    if golden.shape != x.shape or not np.all(np.isfinite(golden)):
        raise ValueError("ONNX golden 출력 오류")
    offsets = info["step_offsets"]
    steps = sorted(offsets, key=int)
    mode = info.get("cylinder_score_mode", "legacy")
    original, original_steps = gd.original_feature_sequence(sample, False, "cylinder")
    if "model_golden_scaled" in stats and not np.allclose(golden, stats["model_golden_scaled"], rtol=1e-6, atol=1e-6):
        raise ValueError("ONNX와 stats의 고정 golden 불일치: artifact 폴더를 함께 교체하세요")
    golden_raw = np.asarray(stats.get("model_golden_raw", std.inverse(golden)), dtype=np.float32)
    offset_matrix = None
    duration = None
    if mode in ("offset_aware_mad", "duration_aware_offset_mad", "duration_classified_offset_mad"):
        center = np.asarray(stats["detrended_residual_center_raw"], dtype=np.float32)
        scale = np.asarray(stats["detrended_residual_scale_raw"], dtype=np.float32)
        normal_offsets = np.asarray(stats["normal_offset_center_raw"], dtype=np.float32)
        if mode.startswith("duration_"):
            duration = gs.duration_aware_step_offsets(original, original_steps, gd.parse_timestamp(sample.frame),
                        offsets, golden_raw, normal_offsets, center, scale, float(info.get("spike_max_seconds", 1.0)))
        if mode == "duration_aware_offset_mad":
            sample_offsets = duration.offsets
            corrected = gs.remove_known_step_offsets(raw-golden_raw, offsets, sample_offsets)
        else:
            corrected, sample_offsets = gs.remove_step_median_offsets(raw-golden_raw, offsets)
        cell = np.abs((corrected-center)/scale)
        offset_matrix = np.abs((duration.offsets if mode == "duration_classified_offset_mad" else sample_offsets)-normal_offsets).astype(np.float32)
    elif mode == "legacy":
        cell = np.abs(x-golden)/np.maximum(np.asarray(stats["error_p95"], dtype=np.float32), 1e-6)
    else:
        raise ValueError("지원하지 않는 실린더 점수 모드: " + mode)
    matrices = {"shape": gs.robust_step_feature_scores(cell, offsets, float(info.get("step_top_fraction", .05)))}
    thresholds = {"shape": info["threshold"]}
    if offset_matrix is not None:
        matrices["offset"], thresholds["offset"] = offset_matrix, info["offset_threshold"]
    topk = int(info.get("spike_topk", 0))
    if topk > 0 and "spike_threshold" in info:
        if mode == "legacy":
            spike = gs.original_grid_spike_scores(std.transform(original), original_steps, offsets, golden, stats["error_p95"], topk)
        else:
            kwargs = dict(residual_center=center)
            if mode.startswith("duration_"):
                kwargs.update(baseline_offsets=duration.offsets, transient_mask=duration.transient_mask,
                              event_durations_seconds=duration.event_durations_seconds)
            else:
                kwargs["remove_step_offset"] = True
            spike = gs.original_grid_spike_scores(original, original_steps, offsets, golden_raw, scale, topk, **kwargs)
            if mode == "duration_classified_offset_mad":
                baseline = gs.original_grid_spike_scores(original, original_steps, offsets, golden_raw, scale, topk,
                           residual_center=center, remove_step_offset=True)
                spike = gs.merge_spike_scores(baseline, spike)
        matrices["spike"], thresholds["spike"] = spike.matrix, info["spike_threshold"]
    ratios = {name: cylinder_ratio(float(matrix.max()), thresholds[name], key, name) for name, matrix in matrices.items()}
    details = []
    labels = ["C1", "C2", "C_DIFF"]
    for i, step in enumerate(steps):
        a, b = map(int, offsets[step])
        for j, name in enumerate(features):
            cell_ratios = {}
            for kind, matrix in matrices.items():
                tk = {"shape": "step_feature_thresholds", "spike": "spike_step_feature_thresholds", "offset": "offset_step_feature_thresholds"}[kind]
                if tk not in stats:
                    raise ValueError("셀 임계값 누락: " + tk)
                cell_ratios[kind] = cylinder_ratio(matrix[i, j], stats[tk][i, j], key, tk, step, labels[j]+" ("+name+")")
            details.append({"step": int(step), "sensor": name, "label": labels[j]+" / ProcessStep "+step,
                            **safe_ratio(max(cell_ratios.values())),
                            "ratios": {k: safe_ratio(v)["ratio"] for k, v in cell_ratios.items()},
                            "ratio_states": {k: safe_ratio(v) for k, v in cell_ratios.items()},
                            "raw": raw[a:b, j].tolist(), "golden": golden_raw[a:b, j].tolist()})
    raw_ratio = max(ratios.values())
    scaled_ratio, applied = scale_cylinder_ratio(raw_ratio, info)
    return dict(available=True, reason="", **safe_ratio(scaled_ratio),
                raw_ratio=safe_ratio(raw_ratio)["ratio"], raw_ratio_infinite=bool(np.isposinf(raw_ratio)),
                ratio_states={k: safe_ratio(v) for k, v in ratios.items()},
                scaling_applied=applied, scaling_version=(info.get("combined_score_scaling") or {}).get("version"),
                **{k+"_ratio": safe_ratio(v)["ratio"] for k, v in ratios.items()}, step_sensor_details=details,
                artifact_digest=component_version(root, "cylinder"), version_scheme=VERSION_SCHEME, group=key)


def inspect_many(sources, results, root, config, checkpoint=None):
    """Share each parsed raw frame; PV predicts once per chamber/group/event batch."""
    from .auxiliary.step_pv_explainer import explain_batch
    if len(sources) != len(results):
        raise ValueError("source/result count mismatch")
    outputs, pv_groups, routing_versions = [], {}, {}
    for source, result in zip(sources, results):
        if checkpoint:
            checkpoint()
        chamber = str(result.get("chamber") or "").upper()
        step_root, cylinder_root = paths(root, config, chamber)
        values, sample = {}, None
        for kind, directory, fn in (("step_time", step_root, step_time_score), ("cylinder", cylinder_root, cylinder_score)):
            try:
                if kind == "step_time":
                    bundle = load_artifacts(directory, kind)
                if sample is None:
                    if source is None:
                        raise ValueError("원본 파일 없음")
                    path = Path(source)
                    frame = sd.read_process_file(path)
                    sample = SimpleNamespace(path=path, frame=frame, equipment=sd.infer_equipment(path),
                              chamber=sd.infer_chamber(path), recipe=sd.infer_recipe(path, frame))
                if kind == "cylinder":
                    candidates = cylinder_candidates(root, config, chamber)
                    directory = cylinder_package(candidates, sample)
                values[kind] = fn(directory, sample)
                if kind == "cylinder":
                    if chamber not in routing_versions:
                        routing_versions[chamber] = cylinder_routing_version(candidates)
                    values[kind]["routing_version"] = routing_versions[chamber]
                if kind == "step_time":
                    pv_key = (chamber, str(directory), bundle["digest"], values[kind]["group"])
                    pv_groups.setdefault(pv_key, {"bundle": bundle, "rows": []})["rows"].append((sample, values[kind]))
            except Exception as exc:
                values[kind] = unavailable(("%s " % chamber) + str(exc))
                report_unavailable(kind, "%s: %s" % (chamber, exc))
        outputs.append(combine(tcn_component(result), values["step_time"], values["cylinder"]))
    for (_, directory, _, group), entry in pv_groups.items():
        if checkpoint:
            checkpoint()
        explain_batch(Path(directory), group, entry["rows"], entry["bundle"], _LOCK, checkpoint)
    return outputs


def inspect(source, result, root, config):
    return inspect_many([source], [result], root, config)[0]


def archived(result):
    saved = result.get("score_components") or {}
    def stored(name, fallback):
        value = saved.get(name)
        if not isinstance(value, dict) or not value:
            return fallback
        value = dict(value)
        value["stored_result"] = bool(value.get("available") or value.get("ratio") is not None or value.get("step_details") or value.get("step_sensor_details"))
        value.setdefault("available", value.get("ratio") is not None)
        return value
    tcn = stored("tcn", tcn_component(result))
    components = combine(tcn,
        stored("step_time", unavailable("과거 이력: 스텝 시간 점수 없음")),
        stored("cylinder", unavailable("과거 이력: 실린더 점수 없음")))
    integrated = saved.get("integrated")
    if isinstance(integrated, dict) and (integrated.get("ratio") is not None or integrated.get("ratio_infinite")):
        components["integrated"] = dict(integrated)
    components["integrated"]["stored_result"] = True
    components["integrated"]["score_basis"] = "inspection_saved"
    return components


def summary(components):
    # Scatter/history never carry raw/golden waveforms; detail reads the archive.
    result = {}
    for name, component in components.items():
        result[name] = {k: v for k, v in component.items() if k not in ("step_details", "step_sensor_details", "pv")}
        if name == "step_time" and component.get("pv"):
            pv = component["pv"]
            result[name]["pv"] = {k: pv.get(k) for k in ("available", "reason", "target_sensor", "primary_step", "primary_occurrence", "primary_sensor")}
    return result


def legacy_cylinder_packages(candidates):
    packages = []
    for directory in candidates:
        try:
            meta = load_artifacts(directory, "cylinder")["meta"]
            manifest = json.loads((directory / "onnx_manifest.json").read_text(encoding="utf-8-sig"))["models"]
            models = {}
            for key, info in meta["models"].items():
                try:
                    if all(artifact_file(directory, name).is_file() for name in
                           (info["stats_path"], manifest[key]["onnx_path"])):
                        models[key] = "combined_score_scaling" in info
                except (ValueError, KeyError, TypeError):
                    pass
            if models:
                packages.append({"digests": sorted(set(artifact_digests(directory).values())), "models": models})
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    return packages


def versions(root, config):
    result = {}
    for ch in ("DSV1", "DSV2", "DSV3", "DSV4", "DSV5", "DSV6"):
        step = paths(root, config, ch)[0]
        candidates = cylinder_candidates(root, config, ch)
        result[ch] = {
            "step_time": component_version(step, "step_time"),
            "cylinder": cylinder_routing_version(candidates),
            "legacy_step_time": sorted(set(artifact_digests(step).values()) - {"missing"}),
            "legacy_cylinder": legacy_cylinder_packages(candidates),
            "locations": {"step_time": str(step), "cylinder": " / ".join(map(str, candidates))},
        }
    return result


def legacy_component_matches(kind, component, current):
    if component.get("version_scheme"):
        return False
    saved = component.get("artifact_digest")
    if kind == "step_time":
        return saved in current.get("legacy_step_time", [])
    # Legacy routing contains an unknowable Windows installation directory.
    # Instead prove the entire selected package's content hash and score contract.
    # Conservatively reject ambiguous higher-priority packages.
    packages = current.get("legacy_cylinder", [])
    if not packages:
        return False
    package = packages[0]
    group = component.get("group")
    if group not in package["models"]:
        return False
    for digest in package["digests"]:
        if saved == digest + ":" + CYLINDER_RATIO_CONTRACT:
            return True
        if saved == digest and not package["models"][group]:
            return True
    return False



def status_versions(root, config):
    try:
        return versions(root, config)
    except Exception as exc:
        report_unavailable("상태 확인", exc)
        return {}


def current_components(result, current):
    """Historical facts remain visible; current artifacts supply advisory status only."""
    components = archived(result)
    for kind in ("step_time", "cylinder"):
        c = components[kind]
        if not c.get("stored_result"):
            continue
        saved_version = c.get("routing_version") if kind == "cylinder" else c.get("artifact_digest")
        valid = bool(saved_version) and saved_version == current.get(kind)
        if kind == "cylinder":
            valid = valid and str(c.get("artifact_digest", "")).endswith(":" + CYLINDER_RATIO_CONTRACT)
        valid = valid or legacy_component_matches(kind, c, current)
        if valid:
            status, reason = "current", "검사 당시 저장 결과"
        elif not current:
            status, reason = "unknown", "검사 당시 저장 결과 · 현재 모델 상태 확인 불가"
        elif not current.get("legacy_" + kind):
            status, reason = "missing", "검사 당시 저장 결과 · 현재 보조모델 없음"
        elif not c.get("artifact_digest"):
            status, reason = "unknown", "검사 당시 저장 결과 · 저장 모델 버전 정보 없음"
        else:
            status, reason = "different", "검사 당시 저장 결과 · 현재 보조모델과 버전 다름"
        if kind == "cylinder" and status == "different" and not str(c.get("artifact_digest", "")).endswith(":" + CYLINDER_RATIO_CONTRACT):
            reason = "검사 당시 저장 점수 · 현재 계산 계약과 다름"
        c["current_model_status"] = status
        c["current_model_reason"] = reason
        if status != "current":
            report_unavailable(kind, reason)
    return components
