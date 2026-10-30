"""Cylinder input contract shared with IO-Recon-TCN exports."""
import json
import numpy as np

CYLINDER_SENSORS = ("d_004_PV", "d_001_PV")
CYLINDER_DIFF_COLUMN = "v_011"
STEP_DURATION_COLUMN = "step_duration_seconds"


def auxiliary_sensor_columns(cfg, role=None) -> tuple:
    if str(getattr(cfg, "step_time_mode", "off") or "off") != "off":
        raise ValueError("이 앱 추론 경로는 step_time_mode 입력 모델을 지원하지 않습니다.")
    columns = []
    cylinder_names = CYLINDER_SENSORS
    if bool(getattr(cfg, "cylinder_diff", False)):
        if str(getattr(cfg, "cylinder_mode", "off") or "off") == "off":
            raise ValueError("--cylinder-diff requires --cylinder-mode input or predict")
        cylinder_names += (CYLINDER_DIFF_COLUMN,)
    for key, names in (("cylinder_mode", cylinder_names),
                       ("step_time_mode", (STEP_DURATION_COLUMN,))):
        mode = str(getattr(cfg, key, "off") or "off")
        if mode not in {"off", "input", "predict"}:
            raise ValueError(f"--{key.replace('_', '-')} supports off/input/predict, got {mode!r}")
        if mode != "off" and (role is None or role == mode):
            columns.extend(names)
    return tuple(columns)


def auxiliary_sensor_metadata(cfg) -> dict:
    return {
        "cylinder_mode": str(getattr(cfg, "cylinder_mode", "off") or "off"),
        "cylinder_diff": bool(getattr(cfg, "cylinder_diff", False)),
        "step_time_mode": str(getattr(cfg, "step_time_mode", "off") or "off"),
        "auxiliary_sensor_cols": list(auxiliary_sensor_columns(cfg)),
        "context_sensor_cols": list(auxiliary_sensor_columns(cfg, "input")),
        "auxiliary_target_cols": list(auxiliary_sensor_columns(cfg, "predict")),
        "cylinder_diff_definition": {
            "column": CYLINDER_DIFF_COLUMN,
            "operation": "subtract",
            "minuend": CYLINDER_SENSORS[0],
            "subtrahend": CYLINDER_SENSORS[1],
            "calculation_scale": "raw_before_normalization",
            "absolute": False,
            "role": "same_as_cylinder_mode",
            "preserve_source_sensors": True,
        },
        "step_time_definition": {
            "column": STEP_DURATION_COLUMN,
            "unit": "seconds",
            "timestamps": ["Date", "Time"],
            "segment": "consecutive_equal_ProcessStep",
            "duration": "next_segment_start_minus_current_start",
            "last_segment": "last_log_timestamp_minus_current_start",
            "broadcast": "constant_for_each_row_in_segment",
            "requires_completed_log": True,
            "predict_past_mask": "hide_duration_of_segment_containing_forecast_origin",
        },
    }


def restore_context_modes(cfg, arch: dict) -> None:
    # 앱 개발 주의: 이 값이 없으면 기존 모델(off)이다. 입력 차원을 바꾸는 옵션이므로
    # 추론 시 임의로 켜거나 끄지 말고 checkpoint/model_arch에 저장된 값을 사용한다.
    for key in ("cylinder_mode", "step_time_mode"):
        saved = str(arch.get(key, "off") or "off")
        requested = getattr(cfg, key + "_request", None)
        if requested is not None and requested != saved:
            raise ValueError(
                f"--{key.replace('_', '-')}={requested} conflicts with trained model "
                f"({saved}). Use the saved mode or train a new model."
            )
        setattr(cfg, key, saved)
    # 앱 개발 주의: cylinder_diff가 true일 때만 raw 오른쪽-왼쪽 채널을 생성한다.
    # 구모델은 false로 처리하며 정규화된 두 입력을 빼서 만들면 안 된다.
    saved_diff = bool(arch.get("cylinder_diff", False))
    requested_diff = getattr(cfg, "cylinder_diff_request", None)
    if requested_diff is not None and requested_diff != saved_diff:
        raise ValueError(
            f"--cylinder-diff={requested_diff} conflicts with trained model "
            f"({saved_diff}). Use the saved setting or train a new model."
        )
    cfg.cylinder_diff = saved_diff
    auxiliary_sensor_columns(cfg)



class AuxiliarySensorError(ValueError):
    """Required model inputs must never silently fall back to zeros."""


def add_auxiliary_sensors(df, cfg):
    names = auxiliary_sensor_columns(cfg)
    if not names:
        return df
    import pandas as pd
    out = df.copy()
    for name in CYLINDER_SENSORS:
        if name not in out:
            raise AuxiliarySensorError("필수 실린더 센서 누락: " + name)
        values = pd.to_numeric(out[name], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise AuxiliarySensorError("실린더 센서 비수치/NaN/무한값: " + name)
        out[name] = values
    if cfg.cylinder_diff:
        with np.errstate(over="ignore", invalid="ignore"):
            values = out[CYLINDER_SENSORS[0]].to_numpy(dtype=np.float64) - out[CYLINDER_SENSORS[1]].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise AuxiliarySensorError("실린더 차이 계산 결과가 유한하지 않습니다.")
        out[CYLINDER_DIFF_COLUMN] = values
    return out


def validate_auxiliary_record(rec, cfg):
    required = auxiliary_sensor_columns(cfg)
    if not required:
        return
    columns = list(rec.get("sensor_cols") or [])
    values = np.asarray(rec["sensors"], dtype=np.float64)
    for name in required:
        if columns.count(name) != 1 or values.ndim != 2 or columns.index(name) >= values.shape[1]:
            raise AuxiliarySensorError("필수 실린더 센서 누락/중복: " + name)
        if not np.isfinite(values[:, columns.index(name)]).all():
            raise AuxiliarySensorError("실린더 센서 비수치/NaN/무한값: " + name)


def configure_auxiliary_model(model, metadata):
    """Restore and validate NPZ/checkpoint schema before inference or conversion."""
    arch = dict(getattr(model, "arch", {}) or {})
    for key in ("cylinder_mode", "cylinder_diff", "step_time_mode"):
        if key in metadata:
            value = metadata[key]
            arch[key] = value.item() if isinstance(value, np.ndarray) else value
    restore_context_modes(model.cfg, arch)
    model.cylinder_mode = model.cfg.cylinder_mode
    model.cylinder_diff = model.cfg.cylinder_diff
    expected = list(auxiliary_sensor_columns(model.cfg))
    model.auxiliary_sensor_cols = expected
    for name in expected:
        if list(model.input_sensor_cols).count(name) != 1:
            raise AuxiliarySensorError("모델 메타의 필수 실린더 입력 열 누락/중복: " + name)
    targets = set(auxiliary_sensor_columns(model.cfg, "predict"))
    if any(name not in model.sensor_cols for name in targets) or any(name in model.sensor_cols for name in set(expected)-targets):
        raise AuxiliarySensorError("실린더 input/predict 역할과 출력 센서 메타가 다릅니다.")
    if "auxiliary_sensor_cols" in metadata and list(metadata["auxiliary_sensor_cols"]) != expected:
        raise AuxiliarySensorError("auxiliary_sensor_cols와 실린더 모드가 다릅니다.")
    definition = metadata.get("cylinder_diff_definition_json")
    if definition is not None:
        try:
            if isinstance(definition, np.ndarray): definition = definition.item()
            if isinstance(definition, bytes): definition = definition.decode("utf-8")
            definition = json.loads(definition)
            expected_def = auxiliary_sensor_metadata(model.cfg)["cylinder_diff_definition"]
            if model.cfg.cylinder_diff and any(definition.get(k) != expected_def[k] for k in ("column", "operation", "minuend", "subtrahend", "calculation_scale", "absolute")):
                raise ValueError("expected signed raw Right-Left")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise AuxiliarySensorError("cylinder_diff_definition_json 불일치: " + str(exc)) from exc
    def indices(key, default):
        values = np.asarray(metadata.get(key, default))
        if values.ndim != 1 or (values.size and (not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all())):
            raise AuxiliarySensorError("잘못된 모델 입력 인덱스: " + key)
        result = [int(v) for v in values]
        if len(set(result)) != len(result) or any(i < 0 or i >= len(model.input_sensor_cols) for i in result):
            raise AuxiliarySensorError("모델 입력 인덱스 범위/중복 오류: " + key)
        return result
    model.future_h_indices = indices("future_h_indices", model.h_indices)
    contexts = [model.input_sensor_cols.index(n) for n in auxiliary_sensor_columns(model.cfg, "input")]
    model.future_context_indices = indices("future_context_indices", contexts)
    if model.future_context_indices != contexts:
        raise AuxiliarySensorError("future_context_indices가 실린더 입력 열과 다릅니다.")
    future = (model.future_h_indices if str(model.future_h_mode if hasattr(model, "future_h_mode") else arch.get("future_h_mode", "input")) != "predict" else []) + contexts
    model.future_input_sensor_indices = indices("future_input_sensor_indices", future)
    if model.future_input_sensor_indices != future:
        raise AuxiliarySensorError("future_input_sensor_indices가 H/실린더 입력 열과 다릅니다.")
    if any(model.input_sensor_cols[i] in expected for i in model.future_h_indices):
        raise AuxiliarySensorError("실린더 입력을 H 입력 인덱스로 지정할 수 없습니다.")
    arch.update(auxiliary_sensor_metadata(model.cfg))
    model.arch = arch


def auxiliary_npz_metadata(model):
    configure_auxiliary_model(model, dict(future_h_indices=getattr(model, "future_h_indices", model.h_indices),
                                         future_context_indices=getattr(model, "future_context_indices", [])))
    meta = auxiliary_sensor_metadata(model.cfg)
    return dict(cylinder_mode=meta["cylinder_mode"], cylinder_diff=np.bool_(meta["cylinder_diff"]),
                cylinder_diff_definition_json=json.dumps(meta["cylinder_diff_definition"], ensure_ascii=False),
                auxiliary_sensor_cols=np.asarray(meta["auxiliary_sensor_cols"], dtype=object),
                future_h_indices=np.asarray(model.future_h_indices, dtype=np.int64),
                future_context_indices=np.asarray(model.future_context_indices, dtype=np.int64),
                future_input_sensor_indices=np.asarray(model.future_input_sensor_indices, dtype=np.int64))
