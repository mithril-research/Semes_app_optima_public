"""Configuration for IO-Recon-TCN.

The model learns from OK wafers only:
  future_h_mode=input:   past T/P/H/IO + future H/IO -> future T/P
  future_h_mode=predict: past T/P/H/IO + future IO   -> future T/P/H
  virtual_sensor_mode=both adds 10 derived sensors to past input and future target.

When no H columns are present it falls back to:
  past T/P/IO + future IO -> future T/P
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from .auxiliary_sensors import (CYLINDER_SENSORS, CYLINDER_DIFF_COLUMN, auxiliary_sensor_columns,
                                auxiliary_sensor_metadata, restore_context_modes)

PKG_ROOT = Path(__file__).resolve().parent
MANUAL_RECIPES_PATH = PKG_ROOT / "manual_recipes.json"

# VAL 파일명에 recipe가 없을 때 수동 매핑한다.
# manual_recipes.json 예:
# {
#   "VAL_OK_001.csv": "29a",
#   "VAL_NG_002": "336"
# }
def _load_manual_recipes(path: Path = MANUAL_RECIPES_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"manual_recipes.json 로드 실패({path}): {e}")
        return {}
    if not isinstance(data, dict):
        print(f"manual_recipes.json 형식 오류({path}): dict가 아니어서 무시")
        return {}
    return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}


MANUAL_RECIPE_BY_FILENAME = _load_manual_recipes()


h_SENSOR_PRESETS = {
    "heater_block3": (
        "h_005_PV", "h_0052_PV", "h_0053_PV",
    ),
    "heater_12": (
        "h_010_PV",
        "h_005_PV", "h_0052_PV", "h_0053_PV",
        "h_0221_PV", "h_0222_PV",
        "h_0223_PV", "h_0224_PV",
        "h_0191_PV", "h_0192_PV",
        "h_0193_PV", "h_0194_PV",
    ),
}


SENSOR_ROLE_PROFILE_ORIGINAL = "original"
# Backward-compatible alias for checkpoints/search metadata saved before v0.87.1.
SENSOR_ROLE_PROFILE_LEGACY = SENSOR_ROLE_PROFILE_ORIGINAL
SENSOR_ROLE_PROFILE_H_SPLIT_V1 = "h_split_v1"
SENSOR_ROLE_PROFILES = (
    SENSOR_ROLE_PROFILE_ORIGINAL,
    SENSOR_ROLE_PROFILE_H_SPLIT_V1,
)

h_SPLIT_HEATER_BLOCK_SENSORS = (
    "h_005_PV",
    "h_0052_PV",
    "h_0053_PV",
)
h_SPLIT_HEATER_BLOCK_COUNT = 3
h_SPLIT_CART_COUNT = 10
h_SPLIT_RESERVOIR_TARGET_SENSORS = (
    "h_010_PV",
)
h_SPLIT_VESSEL_HEATER_COUNT = 8


def normalize_sensor_role_profile(value) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"", "original", "legacy", "default", "none"}:
        return SENSOR_ROLE_PROFILE_ORIGINAL
    if text in {"h_split", "hsplit", "h_split_v1", "hsplit_v1"}:
        return SENSOR_ROLE_PROFILE_H_SPLIT_V1
    return text


def normalize_h_sensor_preset(value) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"", "all", "none"}:
        return ""
    if text in {"heater_block_3", "heater_blocks", "h_0053"}:
        return "heater_block3"
    if text in {"heater12", "heater_12pv"}:
        return "heater_12"
    return text


def h_sensor_keep_for_config(cfg) -> tuple:
    if normalize_sensor_role_profile(getattr(cfg, "sensor_role_profile", "")) == SENSOR_ROLE_PROFILE_H_SPLIT_V1:
        return tuple()
    preset = normalize_h_sensor_preset(getattr(cfg, "h_sensor_preset", ""))
    if preset:
        return tuple(h_SENSOR_PRESETS.get(preset, ()))
    return tuple(str(x).strip() for x in getattr(cfg, "h_sensor_keep", ()) or ()
                 if str(x).strip())


@dataclass
class IOConfig:
    cylinder_mode: str = "off"
    cylinder_diff: bool = False
    cylinder_mode_request: str = None
    cylinder_diff_request: bool = None

    # Data paths.
    ok_dir: Path = PKG_ROOT / "data" / "OK"
    ng_dir: Path = PKG_ROOT / "data" / "NG"
    val_dir: Path = PKG_ROOT / "data" / "VAL"
    artifacts_dir: Path = PKG_ROOT / "artifacts"
    file_format: str = "auto"

    # Column identification.
    io_name_regex: str = r"^\d+$"
    step_names: tuple = ("process step", "process_step", "processstep", "step", "phase")
    time_names: tuple = ("date", "time", "datetime", "timestamp", "index", "linenumber")

    # IO sensors used as actuator commands.
    io_keep: tuple = (
        "9000002", "9000003", "9000004", "9000005", "9000006", "9000007",
        "9000008", "9000009", "9000010", "9000011", "9000012", "9000013",
    )

    # Continuous sensor schema.
    sensor_keep: tuple = ()
    sensor_keep_prefixes: tuple = (
        "p_", "t_", "h_005", "h_022", "h_019",
    )
    target_sensor_prefixes: tuple = ("p_", "t_")
    active_control_sensor_prefixes: tuple = (
        "h_", "h_022", "h_019",
    )
    active_control_sensor_keywords: tuple = ()
    active_control_sensor_regex: str = r"^h_[0-9]+(?:_PV)?$"
    active_control_sensor_names: tuple = (
        "v_008", "v_010",
        "h_0111_PV", "h_010_PV", "h_008_PV",
        "h_009_PV", "h_012_PV", "h_0052_PV",
        "h_013_PV", "h_007_PV", "h_0072_PV",
        "h_0161_PV", "h_015_PV", "h_005_PV", "h_003_PV",
        "h_004_PV", "h_0162_PV", "h_0053_PV", "h_001_PV",
        "h_002_PV", "h_0211_PV", "h_0212_PV",
        "h_0213_PV", "h_0214_PV", "h_0221_PV",
        "h_0222_PV", "h_0223_PV", "h_0224_PV",
        "h_0181_PV", "h_0182_PV", "h_0183_PV",
        "h_0184_PV", "h_0191_PV", "h_0192_PV",
        "h_0193_PV", "h_0194_PV", "h_006_PV",
        "h_0141_PV",
    )
    sensor_exclude: tuple = ()
    schema_infer_rows: int = 512
    virtual_sensor_mode: str = "off"  # off/input/target/both. Enabled modes always predict virtual sensors.
    virtual_sensor_names: tuple = (
        "v_001", "v_002", "v_003",
        "v_004", "v_005", "v_006",
        "v_007", "v_008", "v_009", "v_010",
    )

    # H cartridge ablation. Must be set by CLI: include or exclude.
    h_cart_mode: str = ""
    h_sensor_preset: str = ""
    h_sensor_keep: tuple = ()
    sensor_role_profile: str = SENSOR_ROLE_PROFILE_ORIGINAL

    # Recipe handling.
    recipe_names: tuple = ("29a", "521", "336", "232")
    manual_recipe_by_filename: dict = field(
        default_factory=lambda: dict(MANUAL_RECIPE_BY_FILENAME))
    train_ok_frac: float = 0.8

    # TCN future forecast.
    tcn_past_window: int = 64
    tcn_future_window: int = 64
    tcn_stride: int = 0
    tcn_channels: int = 64
    tcn_blocks: int = 5
    tcn_kernel: int = 3
    tcn_dropout: float = 0.1
    tcn_epochs: int = 60
    tcn_lr: float = 1e-3
    tcn_batch: int = 256
    tcn_weight_decay: float = 1e-4
    tcn_val_frac: float = 0.15
    tcn_patience: int = 8
    post_eval_extend_epochs: int = 10
    post_eval_max_extends: int = 0
    tcn_amp: bool = True
    tcn_cache_train_on_gpu: bool = True
    tcn_cache_max_gb: float = 24.0
    tcn_cudnn_benchmark: bool = True
    use_step_onehot: bool = True
    future_h_mode: str = "input"  # input: future H is known input, predict: future H is target
    tcn_score_percentile: float = 99.0
    tcn_score_max_windows_per_wafer: int = 512
    pred_batch: int = 1

    # Scoring and visual explanation.
    resid_agg: str = "p99"
    wafer_agg: str = "topk_mean"
    score_agg_enabled: bool = False
    wafer_topk: int = 0
    score_time_avg: bool = False
    score_final_avg: bool = False
    threshold_wise: str = "recipe"  # all/recipe/sensor/step
    threshold_source: str = "val-search"  # val-search/train-ok-sigma
    threshold_sigma: float = -1.0
    threshold_sigma_min: float | None = None
    threshold_sigma_max: float | None = None
    threshold_recipe_multipliers: tuple = ()
    search_objective: str = "correct"  # correct/precision
    score_exclude_boundary_step0: bool = True
    raw_error_quantile: float = 99.0
    raw_error_level: float = 1.0
    raw_error_min: float = 1e-9
    explain_sigma: float = 4.0
    large_residual_sigma: float = 4.0
    explain_interpolation: int = 1
    suppress_isolated_step_transition_shading: bool = False
    step_transition_shading_window: int = 2
    step_transition_context_window: int = 8
    smooth_step_boundary_prediction: bool = False
    step_boundary_smooth_window: int = 2
    error_accum_mode: str = "sensor"  # sensor: no reset across steps, step: reset per ProcessStep
    rca_top_sensors: int = 5
    rca_points_per_sensor: int = 2

    # Optional graph score correction retained for RCA/report compatibility.
    sensor_causal_order: tuple = ()
    sensors_order_path: Path = PKG_ROOT / "sensors_order.json"
    graph_score_enabled: bool = True
    graph_score_alpha: float = 0.25
    graph_score_max_hops: int = 4

    def ensure_dirs(self) -> None:
        for d in (self.ok_dir, self.ng_dir, self.val_dir, self.artifacts_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
