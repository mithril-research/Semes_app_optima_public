"""Translation occurs at input boundaries; predictions and channel order stay fixed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import sensor_names as sn


@pytest.fixture
def private_names(tmp_path, monkeypatch):
    path = tmp_path / 'sensor_name_map.private.json'
    path.write_text(json.dumps({'names': {'VendorChannelAlpha': 'p_008_PV',
                                         'VendorChannelBeta': 'h_0053_PV'}}))
    monkeypatch.setenv('IO_SENSOR_NAME_MAP', str(path))
    return path


def test_frame_translation_preserves_measurements_and_metadata(private_names):
    original = pd.DataFrame({'Date': ['2026-11-04'], 'ProcessStep': [1],
                             'VendorChannelAlpha': [2.5], '9000009': [1]})
    original.attrs['n_bad_lines'] = 2
    actual = sn.frame(original)
    assert actual.columns.tolist() == ['Date', 'ProcessStep', 'p_008_PV', '9000009']
    np.testing.assert_array_equal(actual.values, original.values)
    assert actual.attrs == original.attrs
    assert original.columns[2] == 'VendorChannelAlpha'


def test_nested_metadata_reports_and_arrays(private_names):
    numeric = np.asarray([[1., 2.]])
    result = sn.normalize({'sensor_cols': ['VendorChannelAlpha', 'VendorChannelBeta'],
                           'sensor': 'VendorChannelAlpha', 'actual': numeric,
                           'note': '[VendorChannelAlpha] exceeds threshold',
                           'per_sensor': {'VendorChannelBeta': 3.5}})
    assert result['sensor_cols'] == ['p_008_PV', 'h_0053_PV']
    assert result['sensor'] == 'p_008_PV'
    assert 'VendorChannel' not in json.dumps({k: v for k, v in result.items() if k != 'actual'})
    assert result['actual'] is numeric
    assert result['per_sensor']['h_0053_PV'] == 3.5


def test_unknown_sensor_is_stable_and_anonymous(monkeypatch, tmp_path):
    monkeypatch.delenv('IO_SENSOR_NAME_MAP', raising=False)
    monkeypatch.setenv('IO_STATE_DIR', str(tmp_path))
    value = sn.name('UnregisteredVendorChannel')
    assert value.startswith('S_') and value == sn.name('UnregisteredVendorChannel')
    assert sn.name(value) == value
    assert sn.name('p_008_PV') == 'p_008_PV'


def test_collisions_fail_before_inference(private_names):
    with pytest.raises(ValueError, match='Duplicate anonymous'):
        sn.frame(pd.DataFrame([[1, 2]], columns=['VendorChannelAlpha', 'p_008_PV']))


def test_npz_adapter_preserves_numeric_arrays_and_order(private_names, tmp_path):
    path = tmp_path / 'meta.npz'
    weights = np.asarray([3., 1.])
    np.savez(path, sensor_cols=['VendorChannelBeta', 'VendorChannelAlpha'], weights=weights)
    data = sn.MetadataNPZ(np.load(path, allow_pickle=False))
    try:
        assert data['sensor_cols'].tolist() == ['h_0053_PV', 'p_008_PV']
        np.testing.assert_array_equal(data['weights'], weights)
    finally:
        data.close()


def test_missing_or_invalid_private_mapping_is_not_silently_ignored(monkeypatch, tmp_path):
    path = tmp_path / 'missing.json'
    monkeypatch.setenv('IO_SENSOR_NAME_MAP', str(path))
    with pytest.raises(ValueError, match='missing'):
        sn.name('p_008_PV')
    path.write_text('{"names":{"a":"NotAnAnonymousId"}}')
    with pytest.raises(ValueError, match='Invalid private'):
        sn.name('p_008_PV')


def test_legacy_onnx_metadata_keeps_inference_identical(private_names, tmp_path):
    from test_chamber_models import make_model
    from src.config import IOConfig
    from src.onnx_backend_forecast import OnnxForecastModel
    make_model(tmp_path, 1)
    folder = tmp_path / 'DSV1'
    before = OnnxForecastModel.load(folder, IOConfig())
    meta = folder / 'recon_meta.npz'
    with np.load(meta, allow_pickle=False) as data:
        values = dict(data)
    values['sensor_cols'] = np.asarray(['VendorChannelAlpha'])
    values['input_sensor_cols'] = np.asarray(['VendorChannelAlpha'])
    np.savez(meta, **values)
    after = OnnxForecastModel.load(folder, IOConfig())
    assert after.sensor_cols == ['p_008_PV']
    x = np.arange(18, dtype=np.float32).reshape(1, 3, 6)
    expected = before.sess.run(None, {before.sess.get_inputs()[0].name: x})
    actual = after.sess.run(None, {after.sess.get_inputs()[0].name: x})
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


def test_offline_html_does_not_embed_legacy_sensor_names(private_names):
    from src.app.export_html import build_export_html
    source = (Path(__file__).resolve().parents[1] / 'src/app/static/index.html').read_text()
    result = build_export_html(source, [], {'date/wafer': {'cols': ['VendorChannelAlpha']}},
                               {'sensors': ['VendorChannelAlpha'], 'io': []})
    assert 'VendorChannelAlpha' not in result
    assert 'p_008_PV' in result
