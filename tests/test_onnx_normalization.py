"""Real ONNX inference against independent column-wise normalization examples."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from src.config import IOConfig
from src.onnx_backend_forecast import OnnxForecastModel


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        env = patch.dict(os.environ, IO_ORT_THREADS='1', IO_ORT_PROVIDER='cpu')
        env.start(); self.addCleanup(env.stop)
        graph = helper.make_graph([
            helper.make_node('Gather', ['x', 'indices'], ['y'], axis=1)], 'normalization',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 7, 'time'])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 2, 'time'])],
            [numpy_helper.from_array(np.array([0, 1], dtype=np.int64), 'indices')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        model.ir_version = 8
        onnx.save(model, self.path/'tcn_forecast.onnx')
        self.meta = dict(io_cols=['IO1', 'IO2'], input_sensor_cols=['T1', 'H1'],
                         sensor_cols=['T1', 'P1'], target_indices=[0], h_indices=[1],
                         n_steps_global=2, schema_mode='union', input_sen_mean=[2., 20.],
                         input_sen_std=[2., 10.], sen_mean=[10., 100.], sen_std=[2., 5.],
                         resid_abs_limit=[1., 1.], resid_sigma=[1., 1.],
                         per_step_limit=False, per_step_sigma=False, w=2, h=2, stride=2,
                         n_in=7, future_h_mode='input', virtual_sensor_mode='off', use_step_onehot=True)
        self.rec = dict(io=np.array([[4., 4.], [6., 20.], [3., -10.]]),
                        sensors=np.array([[4., 4.], [5., 10.], [3., -10.]]),
                        sensor_cols=['T1', 'H1'], step=np.array([0, 1, 1]), wafer_id='fixture')

    def minmax(self, low=0., high=1.):
        self.meta.update(normalization_mode='minmax', normalization_min=low, normalization_max=high,
                         io_min=[4., 0.], io_max=[5., 10.], io_scale=[1., 10.],
                         input_sen_min=[4., 0.], input_sen_max=[5., 10.], input_sen_scale=[1., 10.],
                         sen_min=[-10., 100.], sen_max=[10., 150.], sen_scale=[20., 50.])

    def load(self):
        np.savez(self.path/'tcn_forecast.npz', **self.meta)
        return OnnxForecastModel.load(self.path, IOConfig())

    def test_override_metadata_does_not_change_model_or_sensor_scores(self):
        from src.wafer_score_override import effective_wafer_score
        original = self.load()
        self.assertIsNone(original.wafer_score_override)
        rule = dict(patterns=['fixture'], threshold_multiplier=1.01)
        self.meta['wafer_score_override_json'] = json.dumps(rule)
        updated = self.load()
        self.assertEqual([r['pattern'] for r in updated.wafer_score_override['rules']], ['fixture'])
        for method in ('predict', 'residual', 'error_level', 'z_residual', '_score_record'):
            before = getattr(original, method)(copy.deepcopy(self.rec))
            after = getattr(updated, method)(copy.deepcopy(self.rec))
            np.testing.assert_array_equal(before, after)
        score = updated._score_record(copy.deepcopy(self.rec))
        result = effective_wafer_score(self.rec, score, 4.264, updated.wafer_score_override)
        self.assertEqual(result['effective_score_raw'], 4.264*1.01)
        self.assertEqual(result['model_score_raw'], score)

    def test_legacy_and_explicit_standard_are_unchanged(self):
        for mode in (None, 'standard'):
            if mode: self.meta['normalization_mode'] = mode
            model = self.load()
            np.testing.assert_allclose(model._normalize_io(self.rec['io']), self.rec['io'])
            np.testing.assert_allclose(model._std_input_sensor(self.rec['sensors']),
                                       [[1., -1.6], [1.5, -1.], [.5, -3.]])
            np.testing.assert_allclose(model.predict(copy.deepcopy(self.rec)),
                                       [[18., 120.], [22., 200.], [16., 50.]])

    def test_columnwise_input_and_unclipped_output(self):
        self.minmax(); model = self.load()
        np.testing.assert_allclose(model._normalize_io(self.rec['io']), [[0., .4], [2., 2.], [-1., -1.]])
        np.testing.assert_allclose(model._std_input_sensor(self.rec['sensors']), [[0., .4], [1., 1.], [-1., -1.]])
        np.testing.assert_allclose(model.predict(self.rec), [[-10., 120.], [30., 200.], [-30., 50.]], atol=1e-5)
        np.testing.assert_array_equal(self.rec['io'][0], [4., 4.])

    def test_custom_range_roundtrip_and_batch(self):
        self.minmax(-2., 3.); model = self.load()
        records = [copy.deepcopy(self.rec), copy.deepcopy(self.rec)]
        records[1]['io'] = records[1]['io'] + 1
        expected = [model.predict(copy.deepcopy(r)) for r in records]
        model.predict_batch(records)
        for rec, want in zip(records, expected):
            with patch.object(model, '_run_mem_safe', side_effect=AssertionError('cache not used')):
                np.testing.assert_allclose(model.predict(rec), want, atol=1e-5)
        raw = np.array([[-10., 125.], [30., 50.]])
        np.testing.assert_allclose(model._inverse_target(model._std_target_sensor(raw)), raw)

    def test_masks_onehot_future_h_and_padding_are_not_scaled(self):
        self.minmax(-2., 3.); model = self.load()
        x = model._block_x(model._normalize_io(self.rec['io']),
                           model._std_input_sensor(self.rec['sensors']), self.rec['step'], 1, 3)
        np.testing.assert_array_equal(x[:, 0], 0.)  # padding stays zero, not normalized raw zero
        np.testing.assert_array_equal(x[4], [0., 1., 0., 0.])  # past mask
        np.testing.assert_array_equal(x[5:], [[0., 1., 0., 0.], [0., 0., 1., 1.]])
        np.testing.assert_allclose(x[3, 2:], [3., -7.])  # future H, still unbounded
        np.testing.assert_array_equal(x[2, 2:], 0.)  # target future is masked

    def test_minmax_does_not_require_unused_standard_statistics(self):
        self.minmax()
        for name in ('input_sen_mean', 'input_sen_std', 'sen_mean', 'sen_std'):del self.meta[name]
        np.testing.assert_allclose(self.load().predict(self.rec)[0], [-10., 120.], atol=1e-5)

    def test_bad_scale_uses_training_fallback(self):
        self.minmax(); self.meta['io_scale'] = [0., np.nan]
        model = self.load()
        np.testing.assert_array_equal(model.io_scale, [1., 1.])
        np.testing.assert_allclose(model._normalize_io([[6., 4.]]), [[2., 4.]])

    def test_missing_or_malformed_metadata_rejected(self):
        for key, value in [('normalization_mode', 'unknown'), ('normalization_max', 0.),
                           ('normalization_min', np.nan), ('io_min', [4.]), ('sen_max', [-20., 150.])]:
            self.minmax(); self.meta[key] = value
            with self.assertRaises(ValueError, msg=key):self.load()
        self.minmax(); del self.meta['input_sen_scale']
        with self.assertRaisesRegex(ValueError, 'input_sen_scale'):self.load()

    def test_matching_metadata_preferred_and_unrelated_npz_rejected(self):
        self.minmax(); model = self.load()
        (self.path/'recon_meta.npz').write_bytes(b'stale metadata')
        self.assertEqual(OnnxForecastModel.find_onnx(self.path)[1].name, 'tcn_forecast.npz')
        (self.path/'tcn_forecast.npz').unlink(); (self.path/'recon_meta.npz').unlink()
        np.savez(self.path/'ok_accum_bands.npz', x=[1])
        self.assertFalse(OnnxForecastModel.available(self.path))


if __name__ == '__main__':unittest.main()
