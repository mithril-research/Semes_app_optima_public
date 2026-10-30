"""Artifact-directed final-score policy; residuals and sensor scores stay intact."""
import json
import math
from pathlib import Path, PurePosixPath

AUDIT_FIELDS = ('model_score_raw', 'effective_score_raw', 'score_overridden',
                'override_ratio', 'matched_pattern', 'source_path')


def normalize_override(value):
    """Normalize legacy global multipliers and ordered per-filename v2 rules."""
    if not isinstance(value, dict):
        return None
    version = value.get('version', 2 if 'rules' in value else 1)
    if (version not in (1, 2) or
            value.get('match_field', 'source_filename') != 'source_filename' or
            value.get('match_type', 'substring') != 'substring' or
            value.get('score_mode', 'threshold_multiplier') != 'threshold_multiplier'):
        return None
    if version == 1:
        patterns = value.get('patterns')
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, (list, tuple)):
            return None
        patterns = list(dict.fromkeys(part.strip() for item in patterns if isinstance(item, str)
                                     for part in item.split(',') if part.strip()))
        items = [dict(pattern=p, threshold_multiplier=value.get('threshold_multiplier', 1.01)) for p in patterns]
    else:
        items = value.get('rules')
    if not isinstance(items, (list, tuple)) or not items:
        return None
    rules = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('pattern'), str) or not item['pattern'].strip():
            return None
        try:
            raw = item.get('threshold_multiplier', 1.01)
            if isinstance(raw, bool):
                return None
            multiplier = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(multiplier) or multiplier <= 0:
            return None
        rules.append(dict(pattern=item['pattern'].strip(), threshold_multiplier=multiplier))
    return dict(version=2, rules=rules, case_sensitive=bool(value.get('case_sensitive', True)))


def override_from_npz(data):
    try:
        raw = data.get('wafer_score_override_json')
        if raw is None:
            return None
        if hasattr(raw, 'item'):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        return normalize_override(json.loads(str(raw)))
    except (ValueError, TypeError, UnicodeError):
        return None


def effective_wafer_score(rec, model_score, threshold, rule):
    raw = float(model_score)
    result = dict(model_score_raw=raw, effective_score_raw=raw, score_overridden=False)
    rule = normalize_override(rule)
    if not rule:
        return result
    source = str(rec.get('source_path') or '').strip()
    filename = PurePosixPath(source.replace('\\', '/')).name if source else str(rec.get('wafer_id') or '')
    matched = next((r for r in rule['rules'] if
                    (r['pattern'] if rule['case_sensitive'] else r['pattern'].casefold()) in
                    (filename if rule['case_sensitive'] else filename.casefold())), None)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return result
    if not math.isfinite(threshold) or threshold <= 1e-12 or matched is None:
        return result
    score = threshold * matched['threshold_multiplier']
    if not math.isfinite(score):
        return result
    result.update(effective_score_raw=score, score_overridden=True,
                  override_ratio=matched['threshold_multiplier'], matched_pattern=matched['pattern'])
    return result


def artifact_override(folder, config_path=None, include_standalone=True):
    """Preserve 6148ae8's selected model JSON authority, including explicit null.

    The runtime can defer the standalone fallback until after the model NPZ.
    A sibling chamber's config must never replace the selected configuration.
    """
    folder = Path(folder)
    configs = [Path(config_path)] if config_path is not None else [folder / n for n in ("search_best.json", "best_search.json")]
    for path in configs:
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        payload = document.get("best", document) if isinstance(document, dict) else {}
        if isinstance(payload, dict) and "wafer_score_override" in payload:
            # An explicit config entry is authoritative, as in 6148ae8.
            return True, normalize_override(payload["wafer_score_override"])
    standalone = folder / "wafer_score_override.json"
    if include_standalone and standalone.is_file():
        value = json.loads(standalone.read_text(encoding="utf-8-sig"))
        rule = normalize_override(value)
        if value is not None and rule is None:
            raise ValueError("잘못된 wafer_score_override.json 보정 규칙")
        return True, rule
    return False, None


def apply_selected_override(result, rec, rule):
    """Override only the selected wafer total; retain sensor/error measurements."""
    score, threshold = result.get("total_score"), result.get("total_threshold")
    if score is None or threshold is None or result.get("total_ratio") is None:
        return result
    audit = effective_wafer_score(rec, score, threshold, rule)
    if not audit["score_overridden"]:
        return result
    return dict(result, selected_score_before_override=score,
                selected_ratio_before_override=result["total_ratio"],
                score_overridden=True, override_ratio=audit["override_ratio"], matched_pattern=audit["matched_pattern"],
                total_score=audit["effective_score_raw"], total_ratio=audit["override_ratio"])


def override_identity(value):
    """Keep existing v1 model/history keys stable across canonical-format changes."""
    rule = normalize_override(value)
    if rule is None:
        return None
    multipliers = {r['threshold_multiplier'] for r in rule['rules']}
    if len(multipliers) == 1:
        return dict(patterns=list(dict.fromkeys(r['pattern'] for r in rule['rules'])),
                    threshold_multiplier=rule['rules'][0]['threshold_multiplier'],
                    case_sensitive=rule['case_sensitive'])
    return rule
