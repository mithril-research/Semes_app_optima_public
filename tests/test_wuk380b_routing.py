"""Equipment-specific routing without starting the production server."""
import ast
import csv
from pathlib import Path
import re
import tempfile
import unittest
from typing import Optional

SOURCE = Path(__file__).resolve().parents[1] / 'src/app/server.py'
NAMES = {'_is_wuk380b', '_chamber_cell_tag', '_chamber_from_token',
         '_extract_chamber_from_log', '_chamber', '_match_chamber', '_chamber_mapping_stale'}


class RoutingTest(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
        self.meta = {}
        self.ns = dict(re=re, csv=csv, Path=Path, Optional=Optional, _WUK380B_OVERRIDE=False,
                       _clean_recipe=lambda x: str(x or '').strip(" '\""),
                       _file_meta=lambda _: self.meta)
        exec(compile(tree, str(SOURCE), 'exec'), self.ns)

    def test_mapping_and_other_equipment(self):
        parse = self.ns['_chamber_from_token']
        for dsv in range(1, 7):
            self.assertEqual(parse(f'PMCUNIT{2*dsv}', '52__WUK380B_wafer'), f'dsv{dsv}')
            for equipment in ['WUK370A', 'WUK380B2', 'WUK380BA', 'OTHER']:
                self.assertEqual(parse(f'PMCUNIT{6+dsv}', equipment), f'dsv{dsv}')
            self.assertEqual(parse(f'DSV{dsv}', 'wuk380b'), f'dsv{dsv}')
        for invalid in [0, 1, 3, 5, 7, 9, 11, 14, 107, 112]:
            self.assertIsNone(parse(f'PMCUNIT{invalid}', 'WUK380B'))
        self.assertIsNone(parse('PMCUNIT107', 'OTHER'))

    def test_physical_c2_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            for encoding in ['utf-8-sig', 'cp949', 'utf-16']:
                for equipment, unit, expected in [('WUK380B', 4, 'dsv2'), ('WUK370A', 8, 'dsv2')]:
                    path = Path(directory) / f'{equipment}_PMCUNIT12.log'
                    path.write_text(f'헤더\nA\tB\tPMCUNIT{unit}\nA\tB\tPMCUNIT12\n', encoding=encoding)
                    self.assertEqual(self.ns['_extract_chamber_from_log'](path), expected)

    def test_stale_metadata_and_version(self):
        self.meta.update(chamber='dsv2', chamber_cell='C2')
        self.assertEqual(self.ns['_chamber']('WUK380B_PMCUNIT8'), 'dsv4')
        self.assertEqual(self.ns['_chamber']('WUK370A_PMCUNIT8'), 'dsv2')
        self.meta['chamber_cell'] = self.ns['_chamber_cell_tag']('WUK380B')
        self.assertEqual(self.ns['_chamber']('WUK380B_PMCUNIT8'), 'dsv2')
        self.assertEqual(self.ns['_chamber_cell_tag']('WUK370A'), 'C2')

    def test_filter_aliases(self):
        match = self.ns['_match_chamber']
        self.assertTrue(match('dsv1', ['PMCUNIT2'], 'WUK380B'))
        self.assertFalse(match('dsv1', ['PMCUNIT7'], 'WUK380B'))
        self.assertTrue(match('dsv1', ['PMCUNIT7'], 'WUK370A'))
        self.assertFalse(match('dsv1', ['PMCUNIT2'], 'WUK370A'))

    def test_app_override_and_disable_invalidate_cached_mapping(self):
        self.ns['_WUK380B_OVERRIDE'] = True
        for number in range(1, 7):
            self.assertEqual(self.ns['_chamber_from_token']('PMCUNIT%d' % (number * 2), 'OTHER'), 'dsv%d' % number)
        self.assertTrue(self.ns['_match_chamber']('dsv1', ['PMCUNIT2'], 'OTHER'))
        self.meta.update(chamber='dsv4', chamber_cell=self.ns['_chamber_cell_tag']('OTHER'))
        self.assertEqual(self.ns['_chamber']('OTHER_PMCUNIT8'), 'dsv4')
        self.ns['_WUK380B_OVERRIDE'] = False
        self.assertTrue(self.ns['_chamber_mapping_stale']('OTHER_PMCUNIT8'))
        self.assertEqual(self.ns['_chamber']('OTHER_PMCUNIT8'), 'dsv2')
        self.assertEqual(self.ns['_chamber_from_token']('PMCUNIT8', 'WUK380B'), 'dsv4')

    def test_setting_persists_and_legacy_path_save_preserves_it(self):
        from typing import Dict
        tree = ast.parse(SOURCE.read_text())
        names = {'_load_app_config', '_save_app_config', '_set_chamber_paths'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        startup = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == '_WUK380B_OVERRIDE' for t in n.targets))
        with tempfile.TemporaryDirectory() as directory:
            ns = dict(Dict=Dict, APP_CONFIG_FILE=Path(directory)/'app.json',
                      CHAMBERS=['dsv1'], _path_store=lambda value:value,
                      HTTPException=lambda code,message:RuntimeError(message))
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
            ns['_set_chamber_paths']({'dsv1':'D:/logs'}, True)
            self.assertTrue(ns['_load_app_config']()['wuk380b_mapping'])
            ns['_set_chamber_paths']({'dsv1':'D:/new'})
            exec(compile(ast.Module(body=[startup], type_ignores=[]), str(SOURCE), 'exec'), ns)
            self.assertTrue(ns['_WUK380B_OVERRIDE'])
            self.assertEqual(ns['_load_app_config']()['chamber_paths']['dsv1'], 'D:/new')
            ns['_set_chamber_paths']({}, False)
            exec(compile(ast.Module(body=[startup], type_ignores=[]), str(SOURCE), 'exec'), ns)
            self.assertFalse(ns['_WUK380B_OVERRIDE'])


if __name__ == '__main__':
    unittest.main()
