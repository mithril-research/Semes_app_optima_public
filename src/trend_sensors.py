"""Physical sensor block in original log column order, independent of model inputs."""
import re
import numpy as np

STEP_NAMES = {'processstep', 'process_step', 'process step', 'step', 'phase'}
META_NAMES = STEP_NAMES | {'date', 'time', 'datetime', 'timestamp', 'index', 'linenumber'}


def sensor_block_columns(names):
    """After process metadata and before the first numeric valve/IO column.

    Do not use 'Valve' in a name as a boundary: h_004_PV is an analog
    measurement. No identifiable IO boundary means the original range is unknown.
    """
    names = list(names)
    steps = [i for i, name in enumerate(names) if str(name).strip().lower() in STEP_NAMES]
    if not steps:
        return []
    start = steps[0] + 1
    end = next((i for i in range(start, len(names)) if re.fullmatch(r'\d+', str(names[i]).strip())), None)
    if end is None:
        return []
    return [name for name in names[start:end]
            if str(name).strip().lower() not in META_NAMES and not str(name).startswith('Unnamed')]


def trend_measurements(frame):
    """Keep missing measurements as NaN; never feed these arrays to inference."""
    import pandas as pd
    columns = sensor_block_columns(frame.columns)
    if not columns:
        return None
    step_col = next(c for c in frame.columns if str(c).strip().lower() in STEP_NAMES)
    step = pd.to_numeric(frame[step_col], errors='coerce').to_numpy(dtype=float)
    # Invalid rows remain separators, so they cannot turn an internal zero into a boundary.
    valid = np.isfinite(step) & (step == np.floor(step)) & (step >= 0) & (step <= np.iinfo(np.int32).max)
    return {'sensor_cols': [str(c) for c in columns],
            'actual': frame.loc[:, columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32),
            'step': np.where(valid, step, -1).astype(np.int32)}


def attach_trend_measurements(archive, measurements):
    if measurements is None:
        return archive
    return dict(archive, trend_actual=measurements['actual'], trend_step=measurements['step'],
                meta=dict(archive['meta'], trend_sensor_cols=measurements['sensor_cols'], trend_zero_version=1))


def boundary_zero_indices(step):
    """Only the leading/trailing contiguous zero runs; internal zero is neither.

    A wafer with no positive process step has no unambiguous start/end pair.
    """
    step = np.asarray(step)
    if not np.any(np.isfinite(step) & (step > 0) & (step == np.floor(step))):
        return {}
    result = {}
    if len(step) and step[0] == 0:
        end = 0
        while end < len(step) and step[end] == 0:
            end += 1
        result['zero_start'] = np.arange(end)
    if len(step) and step[-1] == 0:
        start = len(step) - 1
        while start >= 0 and step[start] == 0:
            start -= 1
        result['zero_end'] = np.arange(start + 1, len(step))
    return result
