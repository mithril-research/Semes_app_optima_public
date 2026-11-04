"""Inference numerical routines from StepTime v0.2.1 a48c0bf.

Original feature/batch counterfactual functions retained; no training, CSV writes,
subprocess calls or log reopens. App orchestration follows below.
"""
from __future__ import annotations
from .artifact_paths import artifact_file
import numpy as np
from src import sensor_names as SN
import pandas as pd
from .step_data import StepRun, StepTimeSample, extract_step_runs, parse_timestamps

def _event_key(step: int, occurrence: int) -> str:
    return f"step-{int(step)}__occ-{int(occurrence)}"

def _numeric_series(frame: pd.DataFrame, name: str, indices: list[int]) -> np.ndarray | None:
    if name not in frame.columns:
        return None
    values = pd.to_numeric(frame.iloc[indices][name], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    if not finite.all():
        values = pd.Series(values).interpolate(limit_direction="both").to_numpy(np.float64)
    return values if np.isfinite(values).all() else None

def _run_indices(run: StepRun, include_boundary: bool = True) -> list[int]:
    indices = list(range(int(run.start_index), int(run.end_index) + 1))
    if include_boundary and run.boundary_index is not None:
        indices.append(int(run.boundary_index))
    return indices

def _run_series(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    run: StepRun,
    sensor: str,
    *,
    include_boundary: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    indices = _run_indices(run, include_boundary=include_boundary)
    values = _numeric_series(frame, sensor, indices)
    if values is None:
        return None
    start = timestamps.iloc[int(run.start_index)]
    elapsed = (timestamps.iloc[indices] - start).dt.total_seconds().to_numpy(np.float64)
    finite = np.isfinite(elapsed) & np.isfinite(values)
    elapsed, values = elapsed[finite], values[finite]
    if not len(elapsed):
        return None
    order = np.argsort(elapsed, kind="stable")
    elapsed, values = elapsed[order], values[order]
    unique, positions = np.unique(elapsed, return_index=True)
    return unique, values[positions]

def _first_crossing(
    elapsed: np.ndarray,
    values: np.ndarray,
    target: float,
    direction: str,
) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    elapsed = np.asarray(elapsed, dtype=np.float64)
    condition = values >= target if direction == "up" else values <= target
    hit = np.flatnonzero(condition)
    if not len(hit):
        return None
    index = int(hit[0])
    if index == 0:
        return float(elapsed[0])
    a, b = float(values[index - 1]), float(values[index])
    ta, tb = float(elapsed[index - 1]), float(elapsed[index])
    if not np.isfinite(a + b + ta + tb) or b == a:
        return tb
    fraction = (float(target) - a) / (b - a)
    return float(ta + np.clip(fraction, 0.0, 1.0) * (tb - ta))

def _resample_sensor(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    run: StepRun,
    sensor: str,
    grid: np.ndarray,
) -> np.ndarray | None:
    # The boundary row already belongs to the next ProcessStep.  It is valid for
    # detecting the target crossing, but using it as a candidate feature would
    # let a downstream step-change response masquerade as a cause.
    series = _run_series(
        frame, timestamps, run, sensor, include_boundary=False)
    if series is None:
        return None
    elapsed, values = series
    return np.interp(grid, elapsed, values, left=values[0], right=values[-1]).astype(np.float32)

def _extract_frame_features(
    sample: StepTimeSample,
    frame: pd.DataFrame,
    specs: dict[str, dict],
    target_sensor: str,
    sensors: list[str],
    cache: dict[tuple[str, str], dict],
) -> None:
    """Store compact PV features from an already parsed process frame."""
    if target_sensor not in frame.columns:
        return
    timestamps = parse_timestamps(frame)
    runs, _ = extract_step_runs(frame)
    for run in runs:
        event = _event_key(run.step, run.occurrence)
        if event not in specs:
            continue
        spec = specs[event]
        grid = np.asarray(spec["grid_seconds"], dtype=np.float64)
        target_series = _run_series(frame, timestamps, run, target_sensor)
        if target_series is None:
            continue
        target_elapsed, target_values = target_series
        trajectories = {
            sensor: _resample_sensor(frame, timestamps, run, sensor, grid)
            for sensor in sensors
        }
        cache[(str(sample.path), event)] = {
            "sample": sample,
            "duration": float(run.duration_seconds),
            "target_initial": float(target_values[0]),
            "target_reach": _first_crossing(
                target_elapsed,
                target_values,
                float(spec["target_value"]),
                str(spec["direction"]),
            ),
            "trajectories": trajectories,
        }

def _response_time(trajectory: np.ndarray, grid: np.ndarray, normal_change: float) -> float | None:
    if not np.isfinite(normal_change) or abs(normal_change) <= 1e-12:
        return None
    target = float(trajectory[0] + 0.5 * normal_change)
    return _first_crossing(
        np.asarray(grid, dtype=np.float64),
        np.asarray(trajectory, dtype=np.float64),
        target,
        "up" if normal_change > 0.0 else "down",
    )

def _rows_to_matrix(
    rows: list[dict],
    sensors: list[str],
    references: np.ndarray,
    points: int,
    target_initial_reference: float,
) -> np.ndarray:
    matrix = []
    for row in rows:
        blocks = []
        for index, sensor in enumerate(sensors):
            value = row["trajectories"].get(sensor)
            if value is None or len(value) != points or not np.isfinite(value).all():
                value = references[index]
            blocks.append(np.asarray(value, dtype=np.float32))
        initial = float(row.get("target_initial", target_initial_reference))
        if not np.isfinite(initial):
            initial = float(target_initial_reference)
        matrix.append(np.concatenate([*blocks, np.asarray([initial], dtype=np.float32)]))
    width = len(sensors) * points + 1
    return np.asarray(matrix, dtype=np.float32).reshape(len(rows), width)

def _batch_counterfactual_predictions(
    model,
    matrix: np.ndarray,
    references: np.ndarray,
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict all wafers in one call, then one batch call per sensor.

    The previous implementation called ``predict`` once per wafer and sensor,
    which multiplied joblib/ExtraTrees scheduling overhead by hundreds of
    thousands on a large blind set.  This keeps the call count at
    ``1 + n_sensors`` for an event regardless of wafer count.
    """
    predicted = np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)
    counterfactual = np.empty((len(matrix), len(references)), dtype=np.float64)
    for sensor_index, reference in enumerate(references):
        start = sensor_index * int(points)
        end = start + int(points)
        replaced = matrix.copy()
        replaced[:, start:end] = reference
        counterfactual[:, sensor_index] = np.asarray(
            model.predict(replaced), dtype=np.float64).reshape(-1)
    contributions = predicted[:, None] - counterfactual
    return predicted, counterfactual, contributions

def _response_status(
    current: np.ndarray,
    grid: np.ndarray,
    normal_change: float,
    normal_time: float,
) -> tuple[str, float | None, float | None]:
    if not np.isfinite(normal_change) or abs(normal_change) <= 1e-12 or not np.isfinite(normal_time):
        return "normal_pattern_stable", None, None
    current_time = _response_time(current, grid, normal_change)
    if current_time is None:
        return "expected_response_missing", None, None
    delay = float(current_time - normal_time)
    tolerance = max(float(np.median(np.diff(grid))) if len(grid) > 1 else 0.1, 1e-6)
    if delay > tolerance:
        status = "delayed_response"
    elif delay < -tolerance:
        status = "accelerated_response"
    else:
        status = "normal_response_timing"
    return status, float(current_time), delay


def _finite(value):
    return float(value) if value is not None and np.isfinite(float(value)) else None


def _load_event(root, info, bundle, lock):
    import joblib
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning
    key = (info['model_path'], info['stats_path'])
    with lock:
        cache = bundle.setdefault('pv_models', {})
        if key not in cache:
            with warnings.catch_warnings():
                warnings.simplefilter('error', InconsistentVersionWarning)
                model = joblib.load(artifact_file(root, info['model_path']))
            # Avoid ExtraTrees multiplying workers inside the inspection pipeline.
            if hasattr(model, 'n_jobs'):
                model.n_jobs = 1
            with np.load(artifact_file(root, info['stats_path']), allow_pickle=False) as z:
                stats = {name: z[name] for name in z.files}
            cache[key] = model, stats
        return cache[key]


def _event_predictions(model, stats, info, target_sensor, rows, checkpoint=None):
    """Return rich event records. Predict count: 1+S+min(3,S)*min(4,G)."""
    selected = [str(v) for v in stats['sensors'].tolist()]
    grid = np.asarray(stats['grid_seconds'], dtype=np.float32)
    refs = np.asarray(stats['reference_trajectories'], dtype=np.float32)
    normal_changes = np.asarray(stats['normal_changes'], dtype=np.float32)
    normal_times = np.asarray(stats['normal_response_times'], dtype=np.float32)
    if not selected or not len(grid) or refs.shape != (len(selected), len(grid)) or not np.isfinite(refs).all():
        raise ValueError('PV reference/grid 구성 오류')
    features = [row[2] for row in rows]
    matrix = _rows_to_matrix(features, selected, refs, len(grid), float(stats['feature_reference'][-1]))
    if checkpoint:
        checkpoint()
    predicted, counter, contributions = _batch_counterfactual_predictions(model, matrix, refs, len(grid))
    if not np.isfinite(predicted).all() or not np.isfinite(counter).all():
        raise ValueError('PV 모델 예측값 오류')
    events, rankings = [], []
    for row_index, (sample, run, row) in enumerate(rows):
        normal = float(info['normal_reach_median_seconds'])
        actual = _finite(row.get('target_reach'))
        delay = actual-normal if actual is not None else None
        direction = -1 if delay is not None and delay < 0 else 1
        candidates = []
        for j, sensor in enumerate(selected):
            current = row['trajectories'].get(sensor)
            missing = current is None or len(current) != len(grid) or not np.isfinite(current).all()
            status, response, response_delay = _response_status(refs[j] if missing else current, grid, float(normal_changes[j]), float(normal_times[j]))
            candidates.append(dict(sensor=sensor, contribution_seconds=0.0 if missing else float(contributions[row_index,j]),
                counterfactual_reach_seconds=float(counter[row_index,j]), response_status='sensor_missing' if missing else status,
                normal_response_seconds=_finite(normal_times[j]), actual_response_seconds=None if missing else response,
                response_delay_seconds=None if missing else response_delay, sensor_missing=bool(missing),
                grid_seconds=grid.tolist(), actual_trajectory=None if missing else current.tolist(),
                normal_trajectory=refs[j].tolist(), impact_windows=[], primary_impact_window=None))
        # Missing sensors are never promoted over measured sensors; a missing expected response IS measured.
        order = sorted(range(len(selected)), key=lambda j:(candidates[j]['sensor_missing'], -direction*candidates[j]['contribution_seconds']))
        rankings.append([j for j in order if not candidates[j]['sensor_missing']][:3])
        elapsed, actual_values = row['target_series']
        target_grid = np.linspace(0.0, max(float(elapsed[-1]), float(grid[-1])), max(2, len(grid)))
        events.append(dict(available=True, reason='', step=int(info['step']), occurrence=int(info['occurrence']),
            transition_type=info['transition_type'], transition_confidence=_finite(info.get('transition_confidence')),
            target_sensor=target_sensor, target_value=float(info['target_value']), target_direction=info['direction'],
            normal_reach_seconds=normal, actual_reach_seconds=actual, reach_delay_seconds=delay,
            predicted_reach_seconds=float(predicted[row_index]), duration_seconds=row['duration'],
            normal_duration_seconds=info.get('normal_duration_median_seconds'),
            explanation_confidence=info.get('explanation_confidence'), validation_mae_seconds=_finite(info.get('validation_mae_seconds')),
            target_grid_seconds=target_grid.tolist(), actual_target_trajectory=np.interp(target_grid,elapsed,actual_values).tolist(),
            candidates=candidates, _order=order, _direction=direction))
    # Rank-slot batching allows different top sensors per wafer without N-dependent model calls.
    for rank in range(min(3,len(selected))):
        indices = [i for i, order in enumerate(rankings) if len(order)>rank]
        if not indices:
            continue
        for block in np.array_split(np.arange(len(grid)), min(4,len(grid))):
            if checkpoint:
                checkpoint()
            replaced = matrix[indices].copy()
            for local,i in enumerate(indices):
                j=rankings[i][rank]
                replaced[local, j*len(grid)+block] = refs[j,block]
            after = np.asarray(model.predict(replaced),dtype=float).reshape(-1)
            if not np.isfinite(after).all():
                raise ValueError('PV 시간구간 예측값 오류')
            for local,i in enumerate(indices):
                # Interpolation segments are visual boundaries; each grid feature is replaced exactly once.
                end = int(block[-1])+1
                events[i]['candidates'][rankings[i][rank]]['impact_windows'].append(dict(
                    start_seconds=float(grid[int(block[0])]),
                    end_seconds=float(grid[end] if end<len(grid) else grid[-1]),
                    contribution_seconds=float(predicted[i]-after[local]),
                    counterfactual_reach_seconds=float(after[local])))
    for event in events:
        direction = event.pop('_direction')
        for candidate in event['candidates']:
            windows=candidate['impact_windows']
            # Only label a primary interval that agrees with the observed delay/advance.
            matching=[w for w in windows if direction*w['contribution_seconds']>0]
            if matching:
                candidate['primary_impact_window']=max(matching,key=lambda w:direction*w['contribution_seconds'])
        event['candidates']=[event['candidates'][j] for j in event.pop('_order')]
    return events


def explain_batch(root, group, samples_and_scores, bundle, lock, checkpoint=None):
    """PV errors affect explanation only. IF and the integrated verdict stay unchanged."""
    import json
    for _, score in samples_and_scores:
        score['pv'] = dict(available=False, reason='PV 설명 모델 없음', events=[])
    try:
        path=root/'pv_explainer_meta.json'
        if not path.is_file():
            return
        with lock:
            if 'pv_meta' not in bundle:
                bundle['pv_meta']=SN.normalize(json.loads(path.read_text(encoding='utf-8-sig')))
            meta=bundle['pv_meta']
        target=str(meta['target_sensor'])
        infos={_event_key(int(info['step']),int(info['occurrence'])):info for info in meta.get('models',{}).values() if info.get('group_key')==group}
        contexts=[]
        for sample, score in samples_and_scores:
            pv=score['pv'];pv.update(target_sensor=target, reason='해당 그룹의 PV 설명 모델 없음' if not infos else '')
            try:
                runs,_=extract_step_runs(sample.frame)
                timestamps=parse_timestamps(sample.frame)
                cache={}
                specs={event:dict(target_value=info['target_value'],direction=info['direction'],grid_seconds=np.linspace(0,float(info['normal_reach_median_seconds']),int(info['grid_points'])).tolist()) for event,info in infos.items()}
                sensors=sorted({sensor for info in infos.values() for sensor in info['selected_sensors']})
                _extract_frame_features(sample,sample.frame,specs,target,sensors,cache)
                contexts.append((sample,score,runs,timestamps,cache))
            except Exception as exc:
                pv['reason']='PV 특징 계산 실패: '+str(exc)
        for event,info in infos.items():
            if checkpoint:
                checkpoint()
            rows=[]; scores=[]
            for sample,score,runs,timestamps,cache in contexts:
                row=cache.get((str(sample.path),event))
                run=next((r for r in runs if _event_key(r.step,r.occurrence)==event),None)
                if row is not None and run is not None:
                    row['target_series']=_run_series(sample.frame,timestamps,run,target)
                    rows.append((sample,run,row));scores.append(score)
            if not rows:
                continue
            try:
                model,stats=_load_event(root,info,bundle,lock)
                events=_event_predictions(model,stats,info,target,rows,checkpoint)
                for score,value in zip(scores,events):
                    score['pv']['events'].append(value)
            except Exception as exc:
                for score in scores:
                    score['pv']['events'].append(dict(available=False,step=int(info['step']),occurrence=int(info['occurrence']),reason='PV 설명 실패: '+str(exc)))
        for sample,score,runs,_,_ in contexts:
            pv=score['pv'];found={(e['step'],e['occurrence']) for e in pv['events']}
            for run in runs:
                if (run.step,run.occurrence) not in found:
                    pv['events'].append(dict(available=False,step=run.step,occurrence=run.occurrence,
                        reason='해당 스텝/occurrence PV 모델 또는 목표 센서값 없음'))
            pv['events'].sort(key=lambda e:(e['step'],e['occurrence']))
            pv['available']=any(e.get('available') for e in pv['events'])
            if not pv['available'] and not pv['reason']:
                pv['reason']='사용 가능한 PV 이벤트 없음'
            # IF attribution takes precedence over PV delay; never substitute another step.
            if score.get('ratio',0)>=1 and score.get('step_details'):
                primary=max(score['step_details'],key=lambda d:(d['contribution'],d['robust_deviation']))
                pv['primary_step']=primary['step']
                matches=[e for e in pv['events'] if e['step']==primary['step']]
                # IF is step-aggregated. Preserve every occurrence; do not invent per-occurrence IF attribution.
                pv['primary_occurrence']=matches[0]['occurrence'] if len(matches)==1 else None
                if len(matches)==1 and matches[0].get('available'):
                    candidates=[c for c in matches[0]['candidates'] if not c['sensor_missing']]
                    pv['primary_sensor']=candidates[0]['sensor'] if candidates else None
    except Exception as exc:
        for _,score in samples_and_scores:
            score['pv']=dict(available=False,reason='PV 설명 실패: '+str(exc),events=[])
