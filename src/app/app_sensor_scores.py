"""Train-OK ProcessStep × sensor calibration for anomaly-detail bars.

Matrix rules match IO-Recon-TCN 98afe91 _app_score_matrix_from_level.
This does not change the stored wafer verdict or the overview sensor ratios.
"""
import json
from pathlib import Path
import numpy as np
from src.wafer_score_override import effective_wafer_score


class AppScoreUnavailable(ValueError):
    pass


def load_reference(model_dir):
    folder = Path(model_dir) / '18_앱점수'
    try:
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8-sig'))
        with np.load(folder / 'step_sensor_scores.npz', allow_pickle=True) as data:
            train = np.asarray(data['train_ok_scores'], dtype=np.float64)
            steps = np.asarray(data['process_steps'])
            sensors = [str(x) for x in data['sensors'].tolist()]
    except (OSError, KeyError, ValueError) as exc:
        raise AppScoreUnavailable('모델 폴더의 18_앱점수/step_sensor_scores.npz와 metadata.json을 확인하세요.') from exc
    if steps.ndim != 1 or not np.isfinite(steps).all() or not np.equal(steps, steps.astype(int)).all():
        raise AppScoreUnavailable('앱점수 ProcessStep 목록이 올바르지 않습니다.')
    steps = steps.astype(int)
    if len(set(steps.tolist())) != len(steps) or len(set(sensors)) != len(sensors):
        raise AppScoreUnavailable('앱점수 스텝 또는 센서가 중복됩니다.')
    if train.ndim != 3 or train.shape[1:] != (len(steps), len(sensors)) or train.shape[0] < 2:
        raise AppScoreUnavailable('앱점수 행렬 크기 또는 학습 OK 표본 수(최소 2장)를 확인하세요.')
    if not np.isfinite(train).all() or (train < 0).any():
        raise AppScoreUnavailable('학습 OK 점수에 유효하지 않은 값이 있습니다.')
    if list(meta.get('process_steps', [])) != steps.tolist() or list(meta.get('sensors', [])) != sensors:
        raise AppScoreUnavailable('metadata.json과 NPZ의 스텝/센서 순서가 다릅니다.')
    for key in ('score_topk', 'score_resid_agg', 'score_time_avg', 'score_final_avg'):
        if key not in meta:
            raise AppScoreUnavailable('앱점수 계산 메타 누락: ' + key)
    for key in ('score_time_avg', 'score_final_avg'):
        if not isinstance(meta[key], bool):
            raise AppScoreUnavailable('앱점수 평균 설정이 올바르지 않습니다: ' + key)
    if meta['score_resid_agg'] not in ('sum', 'mean', 'max', 'p99'):
        raise AppScoreUnavailable('지원하지 않는 센서 집계 방식입니다.')
    topk = meta['score_topk']
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 0:
        raise AppScoreUnavailable('앱점수 top-k 설정이 올바르지 않습니다.')
    keep = steps != 0
    if not keep.any():
        raise AppScoreUnavailable('ProcessStep 0을 제외한 학습 스텝이 없습니다.')
    return train[:, keep, :], steps[keep], sensors, meta


def score_matrix(level, step, columns, steps, sensors, meta):
    level = np.asarray(level, dtype=np.float64)
    step = np.asarray(step)
    if level.ndim != 2 or step.shape != (level.shape[0],) or len(columns) != level.shape[1]:
        raise AppScoreUnavailable('현재 웨이퍼의 점수·스텝·센서 배열 크기가 다릅니다.')
    if len(set(columns)) != len(columns) or set(columns) != set(sensors):
        raise AppScoreUnavailable('검사 당시 센서와 앱점수 기준 센서가 다릅니다. 같은 모델의 자료가 필요합니다.')
    if not np.isfinite(step).all() or not np.equal(step, step.astype(int)).all():
        raise AppScoreUnavailable('웨이퍼 ProcessStep 값이 올바르지 않습니다.')
    step = step.astype(int)
    keep = step != 0
    if not keep.any():
        raise AppScoreUnavailable('ProcessStep 0을 제외한 검사 구간이 없습니다.')
    if set(step[keep].tolist()) - set(steps.tolist()):
        raise AppScoreUnavailable('학습 앱점수 기준에 없는 ProcessStep이 있습니다.')
    level = level[:, [columns.index(s) for s in sensors]].copy()
    level[~keep] = 0.0
    if not np.isfinite(level).all():
        raise AppScoreUnavailable('보관 점수에 무한대/결측값이 있습니다. 재검사가 필요합니다.')
    level = np.maximum(0.0, level)
    topk = meta['score_topk']
    if 0 < topk < len(sensors):
        agg = meta['score_resid_agg']
        if agg == 'p99':
            per = np.percentile(level, 99.0, axis=0)
        elif agg == 'max':
            per = level.max(axis=0)
        elif agg == 'mean':
            per = level.mean(axis=0)
        else:
            per = level.sum(axis=0)
        selected = np.argsort(per)[::-1][:topk]
        mask = np.ones(len(sensors), dtype=bool)
        mask[selected] = False
        level[:, mask] = 0.0
    denom = max(1, int(keep.sum())) if meta['score_time_avg'] or meta['score_final_avg'] else 1
    return np.stack([level[step == st].sum(axis=0) / denom for st in steps])


def sensor_scores(model_dir, level, step, columns, sigma=3.5, selected_steps=None, reference=None, selected_sensors=None, sensor_basis="sigma"):
    train, steps, sensors, meta = reference if reference is not None else load_reference(model_dir)
    current = score_matrix(level, step, columns, steps, sensors, meta)
    mask = np.ones(len(steps), dtype=bool)
    if selected_steps is not None:
        chosen = set(selected_steps) - {0}
        if not chosen:
            raise AppScoreUnavailable('ProcessStep 0을 제외한 스텝을 하나 이상 선택하세요.')
        if chosen - set(steps.tolist()):
            raise AppScoreUnavailable('이 챔버의 학습 기준에 없는 선택 스텝이 있습니다.')
        mask = np.isin(steps, list(chosen))
    sensor_mask = np.ones(len(sensors), dtype=bool)
    if selected_sensors is not None:
        chosen_sensors = set(selected_sensors)
        if not chosen_sensors:
            raise AppScoreUnavailable('센서를 하나 이상 선택하세요.')
        if chosen_sensors - set(sensors):
            raise AppScoreUnavailable('이 챔버의 학습 기준에 없는 선택 센서가 있습니다.')
        sensor_mask = np.isin(sensors, list(chosen_sensors))
    # Build top-k/time-normalized matrices first, then apply the same mask to both sides.
    samples = train[:, mask, :].sum(axis=1)
    mean, std = samples.mean(axis=0), samples.std(axis=0, ddof=0)
    if sensor_basis not in ('sigma', 'train_ok_max'):
        raise AppScoreUnavailable('지원하지 않는 센서 점수 기준입니다.')
    limits = samples.max(axis=0) if sensor_basis == 'train_ok_max' else mean + float(sigma) * std
    values = current[mask].sum(axis=0)
    rows = []
    for j, name in enumerate(sensors):
        if not sensor_mask[j]:
            continue
        limit, value = float(limits[j]), float(values[j])
        ratio = value / limit if limit > 1e-12 else None
        rows.append(dict(index=columns.index(name), sensor=name,
                         score=min(ratio, 2.0) if ratio is not None else None,
                         ratio=ratio, selected_score=value, threshold_raw=limit,
                         train_mean=float(mean[j]), train_std=float(std[j]),
                         verdict=('NG' if value >= limit else 'OK') if limit > 1e-12 else 'NA'))
    total_samples = samples[:, sensor_mask].sum(axis=1)
    total_limit = float(total_samples.mean() + float(sigma) * total_samples.std(ddof=0))
    total_value = float(values[sensor_mask].sum())
    total_ratio = total_value / total_limit if total_limit > 1e-12 else None
    return dict(total_score=total_value, total_threshold=total_limit, total_ratio=total_ratio,
                available=True, method='train_ok_step_sensor_sigma', sensor_basis=sensor_basis, sigma=float(sigma),
                std_ddof=0, train_count=len(train), process_steps=steps[mask].tolist(),
                excluded_steps=[0], sensors=rows, threshold=1.0,
                additive_exact_match_expected=bool(meta.get('additive_exact_match_expected', False)))


class OfflineScoreExport:
    """Sufficient statistics preserve arbitrary step/sensor sum calibration offline."""
    def __init__(self):
        self.references = {}
        self.models = {}
        self.wafers = {}

    def add(self, key, model_dir, level, step, columns, record=None, override=None):
        model_key = str(model_dir)
        try:
            if model_key not in self.references:
                self.references[model_key] = load_reference(model_dir)
            reference = self.references[model_key]
            train, steps, sensors, meta = reference
            model_id = str(list(self.references).index(model_key))
            if model_id not in self.models:
                flat = train.reshape(len(train), -1)
                mean = flat.mean(axis=0)
                centered = flat - mean
                self.models[model_id] = dict(steps=steps.tolist(), sensors=sensors,
                    train_scores=flat.tolist(), mean=mean.tolist(), factor=(np.linalg.qr(centered, mode="r") / np.sqrt(len(train))).tolist(),
                    train_count=len(train), additive_exact_match_expected=bool(meta.get('additive_exact_match_expected', False)))
            matrix = score_matrix(level, step, columns, steps, sensors, meta)
            self.wafers[key] = dict(model=model_id, matrix=matrix.reshape(-1).tolist(), columns=list(columns))
            audit = effective_wafer_score(record or {}, 0., 1., override)
            if audit["score_overridden"]:
                self.wafers[key]["override_ratio"] = audit["override_ratio"]
                self.wafers[key]["matched_pattern"] = audit["matched_pattern"]
        except (AppScoreUnavailable, ValueError, KeyError, OSError) as exc:
            self.wafers[key] = dict(missing=str(exc))

    def payload(self):
        return dict(models=self.models, wafers=self.wafers)
