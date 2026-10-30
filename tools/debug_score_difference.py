"""Read-only score comparison against a running Semes app (Python standard library)."""
import argparse
import json
import math
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import urlopen


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def divide(a, b):
    a, b = number(a), number(b)
    return a / b if a is not None and b is not None and b > 0 else None


def delta(a, b):
    a, b = number(a), number(b)
    return a - b if a is not None and b is not None else None


def compare(saved, selected):
    old_score = saved.get('score_raw')
    old_limit = saved.get('threshold_raw')
    new_score = selected.get('total_score')
    new_limit = selected.get('total_threshold')
    return {
        'available': selected.get('available', False),
        'missing_reason': selected.get('note'),
        'stored': {
            'raw_score': old_score, 'raw_threshold': old_limit,
            'display_score': saved.get('score'),
            'ratio_reconstructed_from_stored_raw': divide(old_score, old_limit),
            'verdict': saved.get('verdict'),
        },
        'all_steps_all_sensors': {
            'raw_score': new_score, 'raw_threshold': new_limit,
            'ratio': selected.get('total_ratio'),
            'steps': selected.get('process_steps'),
            'sensors': [s.get('sensor') for s in selected.get('sensors', [])],
            'sigma': selected.get('sigma'), 'std_ddof': selected.get('std_ddof'),
            'train_count': selected.get('train_count'),
            'legacy_precision': selected.get('legacy_precision'),
        },
        'selected_minus_stored': {
            'raw_score': delta(new_score, old_score),
            'raw_threshold': delta(new_limit, old_limit),
            'ratio_approximate': delta(selected.get('total_ratio'), divide(old_score, old_limit)),
        },
        'cross_comparison_approximate': {
            'selected_score_divided_by_stored_threshold': divide(new_score, old_limit),
            'stored_score_divided_by_selected_threshold': divide(old_score, new_limit),
        },
        'limitations': [
            'Stored raw score and threshold are truncated to 3 decimal places; display score to 2.',
            'Small differences within that precision cannot establish a computation mismatch.',
            'Selected scores are recomputed from archived errors, not fresh model inference.',
            'This report cannot prove float16/float32 rounding caused the observed difference.',
            'The model currently loaded in the app must match the model used for inspection.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--date', required=True, help='Archive date: YYYY-MM-DD')
    parser.add_argument('--wafer', required=True, help='Exact wafer ID, without file extension')
    args = parser.parse_args()
    if urlsplit(args.base_url).scheme not in ('http', 'https'):
        parser.error('--base-url must use http or https')
    base = args.base_url.rstrip('/') + '/api/archive/' + quote(args.date, safe='') + '/' + quote(args.wafer, safe='')

    def get(url):
        with urlopen(url, timeout=180) as response:
            return json.load(response)

    try:
        saved = get(base)
        selected = get(base + '/sensors-app-score')
        report = compare(saved, selected)
        report.update(date=args.date, wafer=args.wafer, archive_metadata=saved)
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        return 0 if report['available'] else 2
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        print('Diagnostic failed: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
