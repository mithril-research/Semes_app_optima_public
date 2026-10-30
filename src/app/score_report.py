"""Auditable wafer score reports; uses the same selection calculator as the UI."""
import json
from pathlib import Path
from src.wafer_score_override import apply_selected_override
import numpy as np
from src.app.app_sensor_scores import load_reference, score_matrix, sensor_scores


def build_score_report(wid, archive, model_dir, history, original_context, selections=None, reference=None):
    report = dict(wafer_id=wid, recipe=history.get('recipe'), chamber=history.get('chamber'),
                  original=dict(original_context, recorded=history), selections={})
    requested = dict(all_steps_sensors=dict(steps=None, sensors=None))
    requested.update(selections or {})
    # Preserve the applied whole-wafer basis independently of sensor calibration.
    for name, selection in list(requested.items()):
        if selection.get('total_basis') == 'original':
            report['selections'][name] = dict(total_basis='original',steps=selection.get('steps'),
                sensors=selection.get('sensors'), score=history.get('score'), threshold=history.get('threshold'),
                ratio=history.get('ratio_thr'), verdict=history.get('verdict'),
                formula='기존 검사 점수에 현재 모델 임계값과 표시 설정 적용 (original 항목 참조)',
                sensor_score_basis='선택 스텝·센서의 학습 정상 평균+3.5σ, 기존 센서 계산 유지')
            del requested[name]

    if archive is None:
        report['missing'] = '보관 데이터 없음: 계산 과정을 복원할 수 없습니다.'
        return report
    level = archive.get('app_score_level')
    level = np.asarray(archive['z'] if level is None else level, dtype=float)
    step = np.asarray(archive['step'])
    keep = step != 0
    valid = keep if original_context.get('exclude_step_zero', True) else np.ones(len(step), bool)
    total = float(np.maximum(0, np.nan_to_num(level[valid], nan=0., posinf=0., neginf=0.)).sum())
    denom = max(1, int(valid.sum())) if original_context.get('score_norm') in ('time', 'final') else 1
    report['original'].update(positive_error_sum=total, normalization_denominator=denom,
                              reconstructed_raw_score=total/denom,
                              error_definition='max(0, abs(actual - prediction) / resid_abs_limit)',
                              precision_note='보관 오차로 재계산한 값입니다. 검사 당시 저장 점수와 저장 정밀도로 인한 차이가 있을 수 있습니다.')
    try:
        ref = reference if reference is not None else load_reference(model_dir)
    except (ValueError, KeyError, OSError) as exc:
        report['selection_missing'] = str(exc)
        return report
    train, steps, sensors, meta = ref
    columns = archive['meta'].get('sensor_cols') or sensors
    for name, selection in requested.items():
        try:
            result = sensor_scores(model_dir, level, step, columns, reference=ref,
                                   selected_steps=selection.get('steps'), selected_sensors=selection.get('sensors'))
            result = apply_selected_override(result, dict(archive['meta'], wafer_id=wid), original_context.get('wafer_score_override'))
            chosen_steps = result['process_steps']
            chosen_sensors = [r['sensor'] for r in result['sensors']]
            sm, cm = np.isin(steps, chosen_steps), np.isin(sensors, chosen_sensors)
            values = train[:,sm,:][:,:,cm].sum(axis=(1,2))
            matrix = score_matrix(level, step, columns, steps, sensors, meta)[sm][:,cm]
            ratio = result['total_ratio']
            report['selections'][name] = dict(
                score_overridden=bool(result.get('score_overridden')),
                selected_score_before_override=result.get('selected_score_before_override',result['total_score']),
                selected_ratio_before_override=result.get('selected_ratio_before_override',ratio),
                override_ratio=result.get('override_ratio'), matched_pattern=result.get('matched_pattern'),
                steps=chosen_steps, sensors=chosen_sensors, excluded_steps=[0],
                score_settings={k:meta.get(k) for k in ('score_topk','score_resid_agg','score_time_avg','score_final_avg')},
                matrix=matrix.tolist(), train_wafer_sums=values.tolist(), train_count=len(values),
                train_mean=float(values.mean()), train_std=float(values.std(ddof=0)), sigma=3.5, ddof=0,
                score=result['total_score'], threshold=result['total_threshold'], ratio=ratio,
                verdict='NA' if ratio is None else ('NG' if ratio >= 1 else 'OK'),
                formula='train_sum[i] = sum(train_ok_scores[i, selected_steps, selected_sensors]); threshold = mean(train_sum) + 3.5 * std(train_sum, ddof=0); ratio = selected_score / threshold; filename override match => final ratio = matched filename rule threshold_multiplier (sensor scores unchanged)',
                threshold_note='레시피 구분 없이 정상 학습 표본 전체 사용. 레시피 배율 미적용. 셀별 임계값의 합이 아닌 웨이퍼별 선택 합계의 분포에서 계산.')
        except (ValueError, KeyError, OSError) as exc:
            report['selections'][name] = dict(selection=selection, missing=str(exc))
    return report


def write_score_report(folder, report):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'score_calculation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    original = report['original']
    recorded = original.get('recorded') or {}
    lines = ['# 전체 이상점수 계산 리포트', '', '웨이퍼: '+str(report['wafer_id']), '',
             '## 기존 점수', '',
             '- 오차 정의: max(0, |실측 − 예측| / 센서 허용오차)',
             '- 보관 오차 합계: '+str(original.get('positive_error_sum')),
             '- 시간 정규화 분모: '+str(original.get('normalization_denominator')),
             '- 보관 오차 재계산 원점수: '+str(original.get('reconstructed_raw_score')),
             '- 검사 이력 원점수: '+str(recorded.get('score_raw')),
             '- 현재 모델/레시피 임계값: '+str(original.get('model_threshold')),
             '- 현재 표시 점수(임계 대비): '+str(recorded.get('ratio_thr')),
             '- 현재 판정: '+str(recorded.get('verdict')),
             '- 표시 계산: 원점수 / 모델·레시피 임계값 / 표시 배율 (시간 정규화 표시 모드는 원점수 / 시점 수 / 표시 임계값)',
             '- 표시 설정: '+json.dumps(original.get('display_rule'), ensure_ascii=False), '',
             original.get('precision_note',''), '']
    for field in ('missing','selection_missing'):
        if report.get(field): lines.extend(['계산 불가: '+report[field], ''])
    labels = dict(all_steps_sensors='전체 스텝·센서 선택 점수', overview='오버뷰 적용 선택 점수', detail='이상분석 적용 전체 점수')
    for name, data in report['selections'].items():
        lines.extend(['## '+labels.get(name,name), ''])
        if data.get('total_basis') == 'original':
            lines.extend(['- 전체 이상점수 기준: 기존 점수', '- 최종 점수: '+str(data['ratio']),
                          '- 판정: '+str(data['verdict']), '- 계산: '+data['formula'],
                          '- 센서별 점수: '+data['sensor_score_basis'], ''])
            continue
        if data.get('missing'):
            lines.extend(['계산 불가: '+data['missing'], '']); continue
        lines.extend(['- 선택 스텝: '+', '.join(map(str,data['steps'])),
                      '- 선택 센서: '+', '.join(data['sensors']),
                      '- 정상 학습 표본 수: '+str(data['train_count']),
                      '- 학습 평균: '+str(data['train_mean']), '- 학습 표준편차(ddof=0): '+str(data['train_std']),
                      '- 선택 임계값 = '+str(data['train_mean'])+' + 3.5 × '+str(data['train_std'])+' = '+str(data['threshold']),
                      '- 현재 선택 오차 합계: '+str(data['score']),
                      '- 최종 점수 = '+str(data['score'])+' / '+str(data['threshold'])+' = '+str(data['ratio']),
                      '- 판정: '+data['verdict']+' (1 이상 NG)', '', data['threshold_note'], '',
                      '스텝×센서 합계 행렬과 정상 학습 웨이퍼별 합계는 score_calculation.json에 저장됩니다.', ''])
    (folder/'score_calculation.md').write_text('\n'.join(lines), encoding='utf-8')
