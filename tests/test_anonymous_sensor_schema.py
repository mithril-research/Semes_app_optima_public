"""Regression checks for the anonymized deployment schema."""
import importlib
import json
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd

from src.config import IOConfig
from src import data
from src.sensor_schema import in_group, paired_zone
from src.sensor_graph import SensorGraph
from src.rca_causal import kind_of, segment_of
from src.app.sensor_differences import PAIRS, difference_series, valve_phase_steps, PHASE_VALVES


class AnonymousSensorSchemaTests(unittest.TestCase):
    def test_cart_exclusion_preserves_temperature_pressure_and_control(self):
        names = ['t_002_PV', 'p_008_PV', 'h_0211_PV', 'h_008_PV', 'h_005_PV', '9000002']
        frame = pd.DataFrame({n: [1., 2.] for n in names})
        frame['ProcessStep'] = [1, 1]
        io, sensors, step = data.split_columns(frame, IOConfig(h_cart_mode='exclude'))
        self.assertEqual(io, ['9000002'])
        self.assertEqual(sensors, ['t_002_PV', 'p_008_PV', 'h_005_PV'])
        self.assertEqual(step, 'ProcessStep')
        self.assertFalse(in_group('t_002_PV', 'cart'))
        self.assertTrue(in_group('H_0211_PV', 'cart'))

    def test_model_input_order_keeps_primary_then_additional_channels(self):
        names = ['h_008_PV', 'p_008_PV', 'h_005_PV', 'h_0221_PV', 'v_001', 'v_008']
        frame = pd.DataFrame({name: [1., 2.] for name in names})
        _, selected, _ = data.split_columns(frame, IOConfig(h_cart_mode='include', virtual_sensor_mode='both'))
        self.assertEqual(selected, ['p_008_PV', 'h_005_PV', 'h_0221_PV', 'h_008_PV', 'v_008', 'v_001'])

    def test_virtual_differences_and_group_averages_use_anonymous_inputs(self):
        frame = pd.DataFrame({'t_002_PV': [10., 20.], 't_003_PV': [4., 7.],
                              'p_008_PV': [8., 14.], 'p_003_PV': [2., 3.],
                              **{f'h_021{i}_PV': [float(i), float(i + 4)] for i in range(1, 5)}})
        result = data.add_virtual_sensors(frame, IOConfig(virtual_sensor_mode='both'))
        np.testing.assert_array_equal(result['v_001'], [6., 13.])
        np.testing.assert_array_equal(result['v_004'], [6., 11.])
        np.testing.assert_array_equal(result['v_007'], [2.5, 6.5])

    def test_graph_range_expansion_and_cascade_pairing(self):
        graph = SensorGraph.from_json(Path('src/sensors_order.json'))
        self.assertTrue({'h_0211_PV', 'h_0214_PV', '9000015', '9000018'} <= graph.nodes)
        self.assertNotIn('h_0211-4_PV', graph.nodes)
        self.assertEqual(paired_zone('h_0212_PV', ['h_0221_PV', 'h_0222_PV'], 'upper'), ('h_0222_PV', '2'))
        self.assertEqual(paired_zone('h_0183_PV', ['h_0193_PV'], 'lower'), ('h_0193_PV', '3'))
        self.assertEqual(kind_of('h_0211_PV'), 'CART')
        self.assertEqual(kind_of('h_0191_PV'), 'ZONE')
        self.assertEqual(kind_of('t_002_PV'), 'T')
        self.assertEqual(segment_of('t_002_PV'), '공급 라인(예열)')

    def test_training_and_inspection_groups_and_log_conversion_agree(self):
        self.assertEqual(Path('src/sensor_schema.py').read_text(), Path('src/train_jw/sensor_schema.py').read_text())
        # Import preprocessing independently of the optional GPU training package initializer.
        package = types.ModuleType('_anonymous_training_fixture')
        package.__path__ = [str(Path('src/train_jw').resolve())]
        sys.modules[package.__name__] = package
        try:
            conversion = importlib.import_module(package.__name__ + '.log2csv')
            training = importlib.import_module(package.__name__ + '.data')
            config = importlib.import_module(package.__name__ + '.config')
            columns = conversion.ALL_ORDER + conversion.IO_ORDER
            frame = pd.DataFrame({name: [1., 2.] for name in columns})
            frame['ProcessStep'] = [1, 2]
            converted = conversion.convert_log_dataframe(frame)
            self.assertEqual(len(converted), 2)
            _, train_sensors, _ = training.split_columns(converted, config.IOConfig(h_cart_mode='exclude'))
            _, app_sensors, _ = data.split_columns(converted, IOConfig(h_cart_mode='exclude'))
            self.assertEqual(train_sensors, app_sensors)
            self.assertTrue(all(not in_group(n, 'cart') for n in train_sensors))
            self.assertEqual(len(set(columns)), len(columns))
        finally:
            for name in list(sys.modules):
                if name.startswith(package.__name__):
                    del sys.modules[name]

    def test_pv_and_sv_are_not_merged(self):
        left, right = PAIRS[0]
        names = [left.replace('_PV', '_SV'), right]
        archive = dict(actual=np.array([[10., 2.], [20., 3.]]), step=np.array([1, 1]), meta={'sensor_cols': names})
        row = difference_series(archive)['series'][0]
        self.assertIsNotNone(row['missing'])

    def test_anonymous_io_phase_detection(self):
        values = np.zeros((4, len(PHASE_VALVES)))
        values[1, PHASE_VALVES.index('9000009')] = 1
        values[3, PHASE_VALVES.index('9000011')] = 1
        values[3, PHASE_VALVES.index('9000012')] = 1
        archive = dict(rca_io=values, step=np.array([1, 2, 3, 4]), meta={'rca_io_cols': list(PHASE_VALVES)})
        result = valve_phase_steps(archive)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['steps']['process'], [3])


if __name__ == '__main__':
    unittest.main()
