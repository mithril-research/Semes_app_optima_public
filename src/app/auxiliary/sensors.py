"""Sensor constants copied into this repo intentionally.

The source project uses the same SEMES sensor names, but this repository should
stay runnable on its own. Do not import from IO-Recon-TCN at runtime.
"""

from __future__ import annotations

SENSORS = [
    ("h_0111_PV", "H1"), ("h_010_PV", "H2"),
    ("h_008_PV", "H3"), ("h_009_PV", "H4"),
    ("h_012_PV", "H5"), ("p_002_PV", "P1"),
    ("h_0052_PV", "H6"), ("h_013_PV", "H7"),
    ("h_007_PV", "H8"), ("h_0072_PV", "H9"),
    ("h_0161_PV", "H10"), ("p_005_PV", "P2"),
    ("h_015_PV", "H11"), ("p_006_PV", "P3"),
    ("t_005_PV", "T1"), ("h_005_PV", "H12"),
    ("t_002_PV", "T2"), ("h_003_PV", "H13"),
    ("h_004_PV", "H14"), ("h_0162_PV", "H15"),
    ("p_008_PV", "P4"), ("h_0053_PV", "H16"),
    ("t_001_PV", "T3"), ("h_001_PV", "H17"),
    ("h_002_PV", "H18"), ("p_007_PV", "P5"),
    ("h_0211_PV", "H19"), ("h_0212_PV", "H20"),
    ("h_0213_PV", "H21"), ("h_0214_PV", "H22"),
    ("h_0221_PV", "H23"), ("h_0222_PV", "H24"),
    ("h_0223_PV", "H25"), ("h_0224_PV", "H26"),
    ("d_002_PV", "E1"), ("d_005_PV", "E2"),
    ("d_003_PV", "E3"), ("d_001_PV", "C1"),
    ("d_004_PV", "C2"), ("d_0061_PV", "S1"),
    ("d_0062_PV", "S2"), ("h_0181_PV", "H27"),
    ("h_0182_PV", "H28"), ("h_0183_PV", "H29"),
    ("h_0184_PV", "H30"), ("h_0191_PV", "H31"),
    ("h_0192_PV", "H32"), ("h_0193_PV", "H33"),
    ("h_0194_PV", "H34"), ("h_006_PV", "H35"),
    ("p_003_PV", "P6"), ("t_003_PV", "T4"),
    ("h_0141_PV", "H36"), ("p_004_PV", "P7"),
    ("t_004_PV", "T5"), ("s_001_Press", "LP1"),
    ("s_002_Press", "LP2"), ("s_003_Press", "LP3"),
]

TAG_BY_NAME = {name: tag for name, tag in SENSORS}
NAME_BY_TAG = {tag: name for name, tag in SENSORS}

P_SENSORS = [
    "p_002_PV",
    "p_005_PV",
    "p_006_PV",
    "p_008_PV",
    "p_007_PV",
    "p_003_PV",
    "p_004_PV",
]

T_SENSORS = [
    "t_005_PV",
    "t_002_PV",
    "t_001_PV",
    "t_003_PV",
    "t_004_PV",
]

# Neighboring process differences requested for the first golden-pattern model.
VIRTUAL_PAIRS = [
    ("v_101", "t_005_PV", "t_002_PV"),
    ("v_102", "t_002_PV", "t_001_PV"),
    ("v_103", "t_002_PV", "t_003_PV"),
    ("v_104", "t_001_PV", "t_003_PV"),
    ("v_105", "t_003_PV", "t_004_PV"),
    ("v_106", "p_002_PV", "p_005_PV"),
    ("v_107", "p_005_PV", "p_006_PV"),
    ("v_108", "p_006_PV", "p_008_PV"),
    ("v_109", "p_006_PV", "p_007_PV"),
    ("v_110", "p_008_PV", "p_003_PV"),
    ("v_111", "p_007_PV", "p_003_PV"),
    ("v_112", "p_003_PV", "p_004_PV"),
]

CYLINDER_RIGHT = "d_004_PV"
CYLINDER_LEFT = "d_001_PV"
CYLINDER_DIFF = "v_113"
CYLINDER_FEATURES = [CYLINDER_RIGHT, CYLINDER_LEFT, CYLINDER_DIFF]


def short_name(name: str) -> str:
    """Return a plot-friendly label such as ``[P4] p_008_PV``."""
    if name in TAG_BY_NAME:
        return f"[{TAG_BY_NAME[name]}] {name}"
    if name == CYLINDER_DIFF:
        return "[C2-C1] Right - Left cylinder"
    return name
