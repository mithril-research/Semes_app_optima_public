"""Anonymous sensor IDs and optional installation-local legacy name translation.

The private mapping is never part of the source distribution. Numerical arrays,
channel order and model weights are not changed by translation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading

_LOCK = threading.RLock()
_CACHE = None
_META = {'date', 'time', 'processstep', 'process_step', 'process step', 'step',
         'phase', 'recipe', 'lot', 'chamber', 'datetime', 'timestamp', 'index', 'linenumber'}
_ID = re.compile(r'^(?:(?:LP|P|T|H|D|S|C|V|IO)[_-]?\d+(?:-\d+)?|S_[0-9a-f]{12}|'
                 r'(?:VT|VP|VC)\d+_[A-Za-z0-9_]+|del_[PT]\d+|[pthds]_\d+(?:-\d+)?_(?:PV|SV|Press)|900\d{4})$', re.I)
_NAME_LISTS = {'sensor_cols', 'input_sensor_cols', 'io_cols', 'sensor_names',
               'feature_names', 'sensors', 'cols', 'recon_sensors', 'input_sensors',
               'rca_in_cols', 'rca_io_cols', 'trend_sensor_cols', 'valve_cols',
               'context_sensor_cols', 'auxiliary_sensor_cols', 'auxiliary_target_cols',
               'target_sensor_names', 'active_control_sensor_names'}
_NAME_FIELDS = {'sensor', 'target_sensor', 'primary_sensor', 'sensor_name', 'left', 'right'}


def _mapping():
    global _CACHE
    configured = os.environ.get('IO_SENSOR_NAME_MAP')
    default_root = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    root = Path(os.environ.get('IO_STATE_DIR') or default_root)
    path = Path(configured).expanduser() if configured else root / 'sensor_name_map.private.json'
    try:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        if configured:
            raise ValueError('Private sensor mapping file is missing') from None
        return {}, None
    with _LOCK:
        if _CACHE and _CACHE[0] == key:
            return _CACHE[1:]
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            names = data.get('names', data)
            if not isinstance(names, dict) or not all(
                    isinstance(k, str) and k and isinstance(v, str) and _ID.fullmatch(v)
                    for k, v in names.items()):
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise ValueError('Invalid private sensor mapping; use anonymous IDs as values') from None
        pattern = re.compile('|'.join(re.escape(k) for k in sorted(names, key=len, reverse=True))) if names else None
        _CACHE = (key, names, pattern)
        return names, pattern


def name(value):
    """Stable anonymous ID for an external sensor column, without an inverse map."""
    text = str(value)
    names, _ = _mapping()
    if text in names:
        return names[text]
    if text.lower() in _META or text.isdecimal() or _ID.fullmatch(text) or text.startswith('Unnamed'):
        return text
    return 'S_' + hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


def normalize(value, field=None):
    """Translate metadata/labels, preserving arrays, values and insertion order."""
    if isinstance(value, dict):
        return {redact_text(str(k)): normalize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [name(v) if field in _NAME_LISTS and isinstance(v, str) else normalize(v, field)
                 for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, str):
        return name(value) if field in _NAME_FIELDS else redact_text(value)
    # Only string arrays in metadata are translated; numerical tensors stay intact.
    if field in _NAME_LISTS and hasattr(value, 'dtype') and value.dtype.kind in 'UOS':
        import numpy as np
        return np.asarray([name(v) for v in value.tolist()])
    return value


def redact_text(value):
    names, pattern = _mapping()
    return pattern.sub(lambda match: names[match.group()], value) if pattern else value


def frame(data):
    """Rename input columns only. Ambiguous aliases fail before model inference."""
    columns = [name(c) for c in data.columns]
    if len(columns) != len(set(columns)):
        raise ValueError('Duplicate anonymous sensor IDs in input columns')
    result = data.copy(deep=False)
    result.columns = columns
    return result


class MetadataNPZ:
    """Lazy NPZ adapter; only named metadata is normalized, never numeric arrays."""
    def __init__(self, data):
        self.data = data
        self.files = data.files

    def __getitem__(self, key):
        return normalize(self.data[key], key)

    def get(self, key, default=None):
        return self[key] if key in self.data else default

    def __contains__(self, key):
        return key in self.data

    def close(self):
        self.data.close()


def mapping_signature():
    names, _ = _mapping()
    return hashlib.sha256(json.dumps(names, sort_keys=True).encode()).hexdigest()
