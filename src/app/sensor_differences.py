"""Signed sensor differences: per-step statistics and legacy block means."""
import re
import numpy as np

PAIRS = (
    ('p_008_PV', 'p_007_PV'),
    ('p_008_PV', 'p_003_PV'),
    ('p_007_PV', 'p_003_PV'),
    ('t_002_PV', 't_001_PV'),
    ('t_002_PV', 't_003_PV'),
    ('t_001_PV', 't_003_PV'),
)


# Explicit aliases only: never confuse PV measurements with SV setpoints.
SENSOR_ALIASES = {'p_008_PV': ('p_008_PV', 'p_008'), 'p_007_PV': ('p_007_PV', 'p_007'), 'p_003_PV': ('p_003_PV', 'p_003'), 't_002_PV': ('t_002_PV', 't_002'), 't_001_PV': ('t_001_PV', 't_001'), 't_003_PV': ('t_003_PV', 't_003')}


def _sensor_key(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


VALVE_PHASE_VERSION = 2
VALVE_PHASE_MARGIN = 3
PHASE_VALVES = ('9000002', '9000003', '9000004', '9000005', '9000006', '9000007',
                '9000008', '9000009', '9000010', '9000011', '9000012', '9000013')


def valve_phase_steps(archive):
    """Detect phase boundaries from simultaneous recorded valve IO, per wafer.

    Allow up to three non-matching original samples per step, with at least
    one matching sample required. Missing/invalid IO is not inferred as closed.
    Only known valve IO (plus additional recorded 900xxxx channels) is checked,
    never temperatures whose names contain 'Valve'. Unknowns are not closed.
    """
    def missing(reason):
        return dict(version=VALVE_PHASE_VERSION, status='unavailable', reason=reason, steps={})
    meta = archive.get('meta', {})
    names = [str(x).strip() for x in meta.get('rca_io_cols', [])]
    source = archive.get('rca_io')
    if source is None:
        return missing('보관된 밸브 IO 없음')
    io = np.asarray(source, dtype=float)
    step = np.asarray(archive.get('step'))
    if io.ndim != 2 or io.shape != (len(step) if step.ndim == 1 else -1, len(names)):
        return missing('밸브 IO·스텝 길이 불일치')
    indices = {}
    for i, name in enumerate(names):
        matches = re.findall(r'(?<!\d)(900\d{4})(?!\d)', name)
        if len(matches) == 1:
            if matches[0] in indices:
                return missing('밸브 IO 중복: ' + matches[0])
            indices[matches[0]] = i
    absent = [v for v in PHASE_VALVES if v not in indices]
    if absent:
        return missing('대기 단독 개방 확인용 밸브 IO 누락: ' + ', '.join(absent))
    factor = max(1, int(meta.get('interp_factor', 1) or 1))
    io = io[::factor];step = step[::factor]
    positive = np.isfinite(step) & (step > 0) & (step == np.floor(step))
    io = io[positive];step = step[positive].astype(int)
    if not len(step):
        return missing('유효 공정 스텝 없음')
    order = [int(step[0])]
    for a, b in zip(step[:-1], step[1:]):
        if a != b:
            order.append(int(b))
    if len(order) != len(set(order)):
        return missing('동일 스텝 재등장으로 구간 경계 불명확')
    values = io[:, list(indices.values())]
    if not np.all(np.isfinite(values) & (values >= 0)):
        return missing('밸브 IO 누락/비정상 값으로 구간 확인 불가')
    # Check co-occurrence at the same sample, not independent maxima per step.
    only_wait = (io[:, indices['9000009']] > 0) & ((values > 0).sum(axis=1) == 1)
    both_discharge = (io[:, indices['9000011']] > 0) & (io[:, indices['9000012']] > 0)
    def qualifies(condition, st):
        samples = condition[step == st]
        matched = int(np.count_nonzero(samples))
        return matched > 0 and len(samples) - matched <= VALVE_PHASE_MARGIN
    waiting = [i for i, st in enumerate(order) if qualifies(only_wait, st)]
    discharge = [i for i, st in enumerate(order) if qualifies(both_discharge, st)]
    if not waiting or not discharge:
        return missing('9000009 단독 개방 또는 9000011·9000012 동시 개방 스텝 없음')
    begin = waiting[0];end = discharge[0]
    if end <= begin:
        return missing('대기·배출 경계 순서가 맞지 않거나 한 스텝에 경계가 겹침')
    waiting = [i for i in waiting if i < end]
    last_wait = waiting[-1]
    if waiting != list(range(begin, last_wait + 1)):
        return missing('대기 스텝이 불연속으로 재등장하여 구간 불명확')
    return dict(version=VALVE_PHASE_VERSION,status='ok',reason='',margin_samples=VALVE_PHASE_MARGIN,wait_start=order[begin],
                discharge_start=order[end],steps=dict(supply=order[:begin],wait=order[begin:last_wait+1],
                process=order[last_wait+1:end],discharge=order[end:]))


def difference_series(archive, pairs=None):
    """Keep wafer boundaries, the last short block and the sign of A - B.

    Read archived physical measurements, never residuals or model scores.
    Average within each positive ProcessStep first, then give each step equal
    weight in fixed bins 1–5, 6–10, etc. Missing steps do not shift the bins.
    """
    original_archive = archive
    full_block = archive.get('trend_actual') is not None and archive.get('trend_step') is not None
    if full_block:
        archive = dict(actual=archive['trend_actual'], step=archive['trend_step'],
                       meta={'sensor_cols': archive['meta'].get('trend_sensor_cols', []), 'interp_factor': 1})
    meta = archive['meta']
    actual = np.asarray(archive['actual'], dtype=float)
    if actual.ndim != 2:
        raise ValueError('센서 실측 배열 형식 오류')
    factor = max(1, int(meta.get('interp_factor', 1) or 1))
    step = np.asarray(archive.get('step'))
    if step.shape != (len(actual),) or not np.issubdtype(step.dtype, np.number):
        raise ValueError('공정 스텝 정보가 없거나 센서값 길이와 다릅니다.')
    if not np.isfinite(step).all() or not np.equal(step, np.floor(step)).all():
        raise ValueError('공정 스텝 번호가 올바르지 않습니다.')
    step = step[::factor].astype(int)
    present = sorted(int(v) for v in np.unique(step) if v > 0)
    step_indices = {st: np.flatnonzero(step == st) for st in present}
    from src.trend_sensors import boundary_zero_indices
    zero_indices = boundary_zero_indices(step)
    step_indices.update(zero_indices)
    stat_steps = present + list(zero_indices)
    max_step = max(present, default=0)
    starts = list(range(1, max_step + 1, 5))
    columns = {}
    labels = {}
    for array, names in ((actual, meta.get('sensor_cols', [])),
                         (archive.get('rca_insen'), meta.get('rca_in_cols', []))):
        if array is None:
            continue
        array = np.asarray(array, dtype=float)
        if array.ndim != 2 or array.shape[0] != len(actual) or array.shape[1] != len(names):
            continue
        for j, name in enumerate(names):
            if re.fullmatch(r'\d+', str(name).strip()):
                continue
            columns.setdefault(_sensor_key(name), array[::factor, j])
            labels.setdefault(_sensor_key(name), str(name))
    sensor_series = []
    canonical = {_sensor_key(alias): name for name, aliases in SENSOR_ALIASES.items() for alias in aliases}
    # Reduce all sensor rows together, once per step, rather than calling
    # mean/std/min/max separately for every sensor × step cell.
    keys = list(columns)
    sensor_series = [dict(sensor=canonical.get(key, labels[key]), step_stats={}) for key in keys]
    if keys:
        for st in stat_steps:
            block = np.asarray([columns[key][step_indices[st]] for key in keys], dtype=float)
            finite = np.isfinite(block)
            complete = finite.all(axis=1)
            if complete.any():
                values = block[complete]
                means, lows, highs = values.mean(axis=1), values.min(axis=1), values.max(axis=1)
                deviations = values.std(axis=1, ddof=0)
                for j, mean, low, high, std in zip(np.flatnonzero(complete), means, lows, highs, deviations):
                    sensor_series[j]['step_stats'][str(st)] = dict(
                        mean=float(mean), min=float(low), max=float(high), std=float(std), count=int(values.shape[1]))
            # Preserve exact finite-only semantics for incomplete channels.
            for j in np.flatnonzero(~complete):
                values = block[j, finite[j]]
                if values.size:
                    sensor_series[j]['step_stats'][str(st)] = dict(
                        mean=float(values.mean()), min=float(values.min()), max=float(values.max()),
                        std=float(values.std(ddof=0)), count=int(values.size))
    matched = {}
    for name, aliases in SENSOR_ALIASES.items():
        for alias in aliases:
            if _sensor_key(alias) in columns:
                matched[name] = columns[_sensor_key(alias)]
                break
    for key, values in columns.items():
        matched.setdefault(canonical.get(key, labels[key]), values)
    columns = matched
    result = dict(step_max=(starts[-1] + 4 if starts else 0), block_size=5,
                  aggregation='process_step_mean', excluded_steps=[0],  # unsplit zero remains excluded from numeric ranges
                  zero_steps=list(zero_indices), zero_step_version=original_archive['meta'].get('trend_zero_version', 0),
                  sensor_scope='full_block' if full_block else 'archive_subset',
                  valve_phases=valve_phase_steps(original_archive), process_steps=present, x=[i + 2 for i in starts],
                  series=[], sensor_series=sensor_series, missing=None, stats_version=2)
    pairs = list(PAIRS) + [tuple(pair) for pair in (pairs or []) if tuple(pair) not in PAIRS]
    for left, right in pairs:
        row = dict(left=left, right=right, values=[], step_stats={}, missing=None)
        absent = [name for name in (left, right) if name not in columns]
        if absent:
            row['missing'] = '보관된 센서 없음: ' + ', '.join(absent)
            row['values'] = [None] * len(starts)
        else:
            diff = columns[left] - columns[right]
            means = {}
            for st in stat_steps:
                block = diff[step_indices[st]]
                block = block[np.isfinite(block)]
                if block.size:
                    means[st] = float(block.mean())
                    row['step_stats'][str(st)] = dict(mean=means[st], min=float(block.min()),
                        max=float(block.max()), std=float(block.std(ddof=0)), count=int(block.size))
            for i in starts:
                values = [means[st] for st in range(i, i + 5) if st in means]
                row['values'].append(float(np.mean(values)) if values else None)
            if not any(v is not None for v in row['values']):
                row['missing'] = '유효한 센서값 없음'
        result['series'].append(row)
    return result


def step_sensor_metrics(row):
    """Per-step measured statistics for the combined sensor and duration views."""
    return {metric: {sensor['sensor']: {step: stat[metric]
                    for step, stat in sensor.get('step_stats', {}).items()
                    if metric in stat}
                    for sensor in row.get('sensor_series', [])}
            for metric in ('mean', 'min', 'std')}


def compact_trend_export(row, metrics):
    """Keep positive steps and separate boundary-zero statistics in HTML exports."""
    def compact(series):
        stats = {str(key): {m: value[m] for m in metrics if m in value}
                 for key, value in series.get('step_stats', {}).items() if key in ('zero_start', 'zero_end') or str(key).isdigit() and int(key) > 0}
        return {**{key: series[key] for key in ('sensor', 'left', 'right', 'missing') if key in series},
                'export_step_stats': stats}
    return {'missing': row.get('missing'), 'sensor_scope': row.get('sensor_scope'), 'valve_phases': row.get('valve_phases'),
            'process_steps': sorted({int(key) for group in ('series', 'sensor_series')
                                     for series in row.get(group, [])
                                     for key in series.get('step_stats', {}) if str(key).isdigit() and int(key) > 0}),
            'zero_steps': row.get('zero_steps', []), 'zero_step_version': row.get('zero_step_version', 0),
            'series': [compact(s) for s in row.get('series', [])],
            'sensor_series': [compact(s) for s in row.get('sensor_series', [])]}


class TrendSummaryCache:
    """Bounded JSON summaries, invalidated by atomic archive replacement.

    Never retain full measurement arrays. Striped locks coalesce concurrent
    requests for the same archive; callers receive independent dictionaries.
    """
    def __init__(self, max_bytes=32 * 1024 * 1024):
        from collections import OrderedDict
        import threading
        self.entries = OrderedDict()
        self.size = 0
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.work_locks = [threading.Lock() for _ in range(16)]

    @staticmethod
    def signature(path):
        try:
            st = path.stat()
            return (str(path.resolve()), st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino)
        except OSError:
            return None

    def get(self, path, model, build):
        import json
        with self.work_locks[hash((str(path), model)) % len(self.work_locks)]:
            before = self.signature(path)
            key = (model, before)
            if before is not None:
                with self.lock:
                    data = self.entries.get(key)
                    if data is not None:
                        self.entries.move_to_end(key)
                if data is not None:
                    return json.loads(data)
            result = build()
            # Missing/error results and data replaced during the read are retried.
            if result is not None and before is not None and before == self.signature(path):
                data = json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')
                if len(data) <= self.max_bytes:
                    with self.lock:
                        old = self.entries.pop(key, None)
                        self.size -= len(old) if old is not None else 0
                        self.entries[key] = data
                        self.size += len(data)
                        while self.size > self.max_bytes:
                            _, old = self.entries.popitem(last=False)
                            self.size -= len(old)
            return result
