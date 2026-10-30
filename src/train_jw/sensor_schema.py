"""Anonymous sensor groups shared by training and inspection.

Identifiers are deployment-local. No physical-name reverse mapping is shipped.
"""
import re

GROUP_PREFIXES = {'cart': ('h_008', 'h_009', 'h_021', 'h_018'), 'control': ('h_005',), 'response': ('h_022', 'h_019'), 'upper_cart': ('h_021',), 'lower_cart': ('h_018',), 'upper_zone': ('h_022',), 'lower_zone': ('h_019',)}

def compact(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

def in_group(name, group):
    key = compact(name)
    return any(key.startswith(compact(prefix)) for prefix in GROUP_PREFIXES.get(group, ()))

def paired_zone(name, sensors, side):
    prefix = GROUP_PREFIXES[side + "_cart"][0]
    match = re.fullmatch(re.escape(prefix) + r"(\d+)_pv", str(name), re.IGNORECASE)
    if not match:
        return None, None
    number = match.group(1)
    candidates = {compact(p + number + "_PV") for p in GROUP_PREFIXES[side + "_zone"]}
    return next((s for s in sensors if compact(s) in candidates), None), number
