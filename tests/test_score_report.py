import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.app.score_report import build_score_report, write_score_report


class ScoreReportTests(unittest.TestCase):
    def test_records_selected_calculation_and_writes_readable_and_raw_reports(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/'18_앱점수';folder.mkdir()
            meta=dict(process_steps=[0,21,26],sensors=['T1','P1'],score_topk=0,
                      score_resid_agg='sum',score_time_avg=False,score_final_avg=False)
            (folder/'metadata.json').write_text(json.dumps(meta))
            np.savez(folder/'step_sensor_scores.npz',process_steps=[0,21,26],sensors=['T1','P1'],
                     train_ok_scores=[[[999,999],[1,2],[3,4]],[[999,999],[3,4],[5,6]]])
            arch=dict(meta=dict(sensor_cols=['T1','P1']),step=[0,21,26],
                      z=[[999,999],[2,3],[4,5]],app_score_level=[[999,999],[2,3],[4,5]])
            report=build_score_report('w',arch,root,dict(score_raw=14,ratio_thr=1.4,verdict='NG'),
                        dict(model_threshold=10,score_norm='none'),
                        dict(overview=dict(steps=[21],sensors=['T1']),detail=dict(steps=[26],sensors=['P1'])))
            self.assertEqual(report['original']['positive_error_sum'],14)
            all_score=report['selections']['all_steps_sensors']
            self.assertEqual(all_score['train_wafer_sums'],[10,18])
            self.assertEqual(all_score['threshold'],28)
            partial=report['selections']['overview']
            self.assertEqual(partial['matrix'],[[2]])
            self.assertEqual(partial['train_mean'],2)
            self.assertEqual(partial['train_std'],1)
            self.assertEqual(partial['threshold'],5.5)
            self.assertEqual(partial['ratio'],2/5.5)
            self.assertEqual(report['selections']['detail']['steps'],[26])
            dest=Path(root)/'reports';write_score_report(dest,report)
            self.assertEqual(json.loads((dest/'score_calculation.json').read_text())['selections']['overview']['ratio'],2/5.5)
            text=(dest/'score_calculation.md').read_text()
            self.assertIn('2.0 + 3.5 × 1.0 = 5.5',text)
            self.assertIn('오버뷰 적용 선택 점수',text)

    def test_original_detail_total_does_not_recompute_selected_total(self):
        report=build_score_report('w',None,'/nonexistent',
            dict(score=2.28,threshold=1,ratio_thr=2.289,verdict='NG'),{},
            dict(detail=dict(steps=[1],sensors=['P1'],total_basis='original')))
        detail=report['selections']['detail']
        self.assertEqual(detail['total_basis'],'original')
        self.assertEqual(detail['ratio'],2.289)
        self.assertEqual(detail['verdict'],'NG')
        self.assertIn('3.5σ',detail['sensor_score_basis'])
        with tempfile.TemporaryDirectory() as root:
            write_score_report(root,report)
            text=(Path(root)/'score_calculation.md').read_text()
            self.assertIn('전체 이상점수 기준: 기존 점수',text)

    def test_missing_archive_is_reported_instead_of_fabricated_zero(self):
        report=build_score_report('missing',None,'/nonexistent',{}, {})
        self.assertIn('missing',report)
        self.assertEqual(report['selections'],{})

    def test_original_time_divisor_excludes_step_zero(self):
        arch=dict(meta={},step=[0,21,21,26],z=[[100],[2],[3],[4]])
        report=build_score_report('w',arch,'/nonexistent',{},dict(score_norm='time'))
        self.assertEqual(report['original']['normalization_denominator'],3)
        self.assertEqual(report['original']['reconstructed_raw_score'],3)
        self.assertIn('selection_missing',report)
