import json
import unittest
import numpy as np
from src.app.sensor_differences import PAIRS, difference_series, step_sensor_metrics


class SensorDifferenceTests(unittest.TestCase):
    def test_duration_relationship_mean_and_min_use_step_measurements(self):
        archive = dict(actual=np.array([[999.], [-2.], [8.], [20.], [float('nan')]]),
                       step=np.array([0, 1, 1, 2, 3]), meta={'sensor_cols': ['p_test']})
        row = difference_series(archive)
        metrics = step_sensor_metrics(row)
        self.assertEqual(metrics['mean']['p_test'], {'1': 3., '2': 20., 'zero_start':999.})
        self.assertEqual(metrics['min']['p_test'], {'1': -2., '2': 20., 'zero_start':999.})
        self.assertEqual(row['sensor_series'][0]['step_stats']['1']['max'], 8.)
        self.assertEqual(metrics['std']['p_test'], {'1': 5., '2': 0., 'zero_start':0.})
        self.assertEqual(step_sensor_metrics({}), {'mean': {}, 'min': {}, 'std': {}})

    def test_custom_pair_uses_aligned_raw_samples_and_reversed_sign(self):
        archive = dict(actual=np.array([[999., 0.], [10., 8.], [20., 30.], [np.nan, 5.]]),
                       step=np.array([0, 1, 1, 1]), meta={'sensor_cols': ['p_customA', 'p_customB']})
        result = difference_series(archive, pairs=[['p_customA', 'p_customB'], ['p_customB', 'p_customA']])
        a, b = result['series'][-2:]
        self.assertEqual(a['step_stats']['1'], dict(mean=-4., min=-10., max=2., std=6., count=2))
        self.assertEqual(b['step_stats']['1'], dict(mean=4., min=-2., max=10., std=6., count=2))
        missing = difference_series(archive, pairs=[['p_customA', 'p_missing']])['series'][-1]
        self.assertIsNotNone(missing['missing'])
        self.assertEqual(missing['step_stats'], {})

    def archive(self, n=12):
        names = list(dict.fromkeys(x for pair in PAIRS for x in pair))
        actual = np.column_stack([np.arange(n, dtype=float) * (j + 1) + 10 * j for j in range(6)])
        return dict(actual=actual, step=np.arange(1,n+1), meta={'sensor_cols': names})

    def test_six_signed_differences_five_steps_and_tail(self):
        arch = self.archive()
        result = difference_series(arch)
        self.assertEqual(result['x'], [3, 8, 13])
        self.assertEqual(len(result['series']), 6)
        for pair, row in zip(PAIRS, result['series']):
            names = arch['meta']['sensor_cols']
            raw = arch['actual'][:, names.index(pair[0])] - arch['actual'][:, names.index(pair[1])]
            np.testing.assert_allclose(row['values'], [raw[:5].mean(), raw[5:10].mean(), raw[10:].mean()])
            self.assertTrue(all(v < 0 for v in row['values']))
        self.assertEqual(arch['actual'].shape, (12, 6))

    def test_interpolation_does_not_change_process_step_means(self):
        arch = self.archive(6)
        expected = difference_series(arch)
        arch['actual'] = np.column_stack([np.interp(np.arange(11)/2, np.arange(6), col) for col in arch['actual'].T])
        arch['step'] = np.repeat(arch['step'], 2)[:-1]
        arch['meta']['interp_factor'] = 2
        self.assertEqual(difference_series(arch), expected)

    def test_input_sensor_fallback_case_insensitive_and_missing_values(self):
        arch = self.archive(7)
        arch['rca_insen'] = arch['actual'].copy()
        arch['meta']['rca_in_cols'] = [n.upper() for n in arch['meta']['sensor_cols']]
        arch['actual'] = arch['actual'][:, :1]
        arch['meta']['sensor_cols'] = ['p_008_PV']
        arch['actual'][:5, 0] = np.nan
        result = difference_series(arch)
        self.assertIsNone(result['series'][0]['values'][0])
        self.assertIsNotNone(result['series'][0]['values'][1])
        json.dumps(result, allow_nan=False)
        arch.pop('rca_insen')
        result = difference_series(arch)
        self.assertTrue(result['series'][0]['missing'])
        self.assertEqual(result['series'][0]['values'], [None, None])

    def test_short_wafer_keeps_one_point_and_no_cross_wafer_average(self):
        result = difference_series(self.archive(2))
        self.assertEqual(result['x'], [3])
        self.assertEqual(len(result['series'][0]['values']), 1)

    def test_unequal_step_durations_receive_equal_weight_and_zero_is_excluded(self):
        arch = self.archive(9)
        arch['step'] = np.array([0,1,1,1,1,2,3,4,5])
        arch['actual'][:,0] = [9999,10,10,10,10,20,30,40,50]
        arch['actual'][:,1] = 0
        result = difference_series(arch)
        self.assertEqual(result['series'][0]['values'], [30.])
        self.assertEqual(result['process_steps'], [1,2,3,4,5])

    def test_missing_steps_keep_fixed_bins_and_empty_bins_are_gaps(self):
        arch = self.archive(4)
        arch['step'] = np.array([1,5,11,12])
        result = difference_series(arch)
        values = result['series'][0]['values']
        self.assertEqual(result['x'], [3,8,13])
        self.assertIsNone(values[1])
        self.assertEqual(values[0], -10.5)
        self.assertEqual(values[2], -12.5)
        self.assertEqual(result['step_max'], 15)

    def test_missing_or_invalid_process_steps_never_fall_back_to_timepoints(self):
        arch = self.archive()
        for step in (None, [1], [1.5]*12):
            arch['step'] = step
            with self.assertRaises(ValueError):
                difference_series(arch)

    def test_real_pv_names_ignore_case_and_separators(self):
        arch = self.archive()
        expected = difference_series(arch)
        arch['meta']['sensor_cols'] = [name.upper().replace('_', ' ') for name in arch['meta']['sensor_cols']]
        self.assertEqual(difference_series(arch), expected)

    def test_short_aliases_remain_supported_but_sv_is_not_a_measurement(self):
        arch = self.archive()
        expected = difference_series(arch)
        arch['meta']['sensor_cols'] = ['p_008', 'p_007', 'p_003',
                                     't_002', 't_001', 't_003']
        self.assertEqual(difference_series(arch), expected)
        arch['meta']['sensor_cols'][0] = 'p_008_SV'
        result = difference_series(arch)
        self.assertTrue(result['series'][0]['missing'])
        self.assertEqual(result['series'][0]['values'], [None]*3)
