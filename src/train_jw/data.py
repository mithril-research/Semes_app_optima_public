"""Data loading and schema inference for IO-Recon-TCN."""
from __future__ import annotations


import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .sensor_schema import in_group, paired_zone
from .auxiliary_sensors import (add_auxiliary_sensors as _add_auxiliary_sensors,
                                auxiliary_sensor_columns, AuxiliarySensorError)

from .config import (
    IOConfig,
    SENSOR_ROLE_PROFILE_H_SPLIT_V1,
    h_sensor_keep_for_config,
    normalize_sensor_role_profile,
)
from .log2csv import convert_log_dataframe


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def _file_format(cfg: IOConfig) -> str:
    fmt = str(getattr(cfg, "file_format", "auto") or "auto").strip().lower()
    if fmt not in {"auto", "csv", "log"}:
        raise ValueError(f"지원하지 않는 file_format: {fmt}")
    return fmt


def _dataset_files(folder: Path, cfg: IOConfig) -> List[Path]:
    fmt = _file_format(cfg)
    suffixes = {".csv", ".log"} if fmt == "auto" else {"." + fmt}
    return sorted(
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in suffixes
    )


def _file_format_for_path(path: Path, cfg: IOConfig) -> str:
    fmt = _file_format(cfg)
    if fmt != "auto":
        return fmt
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".log":
        return "log"
    raise ValueError(f"auto file_format에서 지원하지 않는 확장자: {path}")


def _read_data_file(path: Path, cfg: IOConfig, **kwargs) -> pd.DataFrame:
    if _file_format_for_path(path, cfg) == "log":
        df = _read_csv(path, sep="\t", **kwargs)
        return convert_log_dataframe(df, with_io=True, only_htp=True)
    return _read_csv(path, **kwargs)


def _resolve_existing_dir_ci(path: Path) -> Path:
    """Resolve an existing directory even when path components differ by case."""
    path = Path(path)
    if path.exists():
        return path
    cur = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        direct = cur / part
        if direct.exists():
            cur = direct
            continue
        if not cur.exists() or not cur.is_dir():
            return path
        match = None
        try:
            for child in cur.iterdir():
                if child.name.lower() == part.lower():
                    match = child
                    break
        except OSError:
            return path
        if match is None:
            return path
        cur = match
    return cur


def _append_unique(dst: List[str], values) -> None:
    seen = {str(x).strip() for x in dst}
    for v in values:
        key = str(v).strip()
        if key not in seen:
            dst.append(v)
            seen.add(key)


def _looks_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    vals = pd.to_numeric(series, errors="coerce")
    return bool(vals.notna().any())


def _is_h_sensor_name(name: str, cfg: IOConfig) -> bool:
    text = str(name).strip()
    tail = re.split(r"[-\s]+", text)[-1]
    if re.match(r"^H\d+$", tail, flags=re.IGNORECASE):
        return True
    desc = re.sub(r"[\s\-]+[A-Za-z]{1,2}\d+\s*$", "", text).strip().lower()
    exact = {str(x).strip().lower() for x in
             getattr(cfg, "active_control_sensor_names", ()) or ()}
    if text.lower() in exact or desc in exact:
        return True
    prefixes = tuple(str(x) for x in
                     getattr(cfg, "active_control_sensor_prefixes", ()) or () if str(x))
    if prefixes and text.startswith(prefixes):
        return True
    low = text.lower()
    keywords = tuple(str(x).lower() for x in
                     getattr(cfg, "active_control_sensor_keywords", ("heat", "heater"))
                     if str(x))
    if any(k in low for k in keywords):
        return True
    pattern = str(getattr(cfg, "active_control_sensor_regex", "") or "")
    return bool(pattern and re.search(pattern, text, flags=re.IGNORECASE))


def _is_cart_h_sensor_name(name: str, cfg: IOConfig) -> bool:
    return _is_h_sensor_name(name, cfg) and in_group(name, "cart")


def _is_cart_sensor_name(name: str) -> bool:
    return in_group(name, "cart")


def _is_h_split_profile(cfg: IOConfig) -> bool:
    return normalize_sensor_role_profile(
        getattr(cfg, "sensor_role_profile", "")
    ) == SENSOR_ROLE_PROFILE_H_SPLIT_V1


def _h_sensor_keep_set(cfg: IOConfig) -> set[str]:
    return {str(x).strip() for x in h_sensor_keep_for_config(cfg)}


def _h_sensor_allowed(name: str, cfg: IOConfig) -> bool:
    keep = _h_sensor_keep_set(cfg)
    return not keep or str(name).strip() in keep


def _virtual_sensor_mode(cfg: IOConfig) -> str:
    mode = str(getattr(cfg, "virtual_sensor_mode", "off") or "off").lower()
    return mode if mode in {"input", "target", "both"} else "off"


def _virtual_sensor_names(cfg: IOConfig) -> list[str]:
    return [str(x) for x in getattr(cfg, "virtual_sensor_names", ()) or ()]


def _is_virtual_sensor_name(name: str, cfg: IOConfig) -> bool:
    return str(name).strip() in set(_virtual_sensor_names(cfg))


def _numeric_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)


def _mean_series(df: pd.DataFrame, names: list[str]) -> pd.Series:
    vals = [_numeric_series(df, name) for name in names]
    if not vals:
        return pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
    return pd.concat(vals, axis=1).mean(axis=1)


def add_virtual_sensors(df: pd.DataFrame, cfg: IOConfig) -> pd.DataFrame:
    """Create derived process sensors when virtual_sensor_mode is enabled."""
    if _virtual_sensor_mode(cfg) == "off":
        return df
    out = df.copy()
    out["v_001"] = _numeric_series(out, "t_002_PV") - _numeric_series(out, "t_003_PV")
    out["v_002"] = _numeric_series(out, "t_001_PV") - _numeric_series(out, "t_003_PV")
    out["v_003"] = _numeric_series(out, "t_002_PV") - _numeric_series(out, "t_001_PV")
    out["v_004"] = _numeric_series(out, "p_008_PV") - _numeric_series(out, "p_003_PV")
    out["v_005"] = _numeric_series(out, "p_007_PV") - _numeric_series(out, "p_003_PV")
    out["v_006"] = _numeric_series(out, "p_008_PV") - _numeric_series(out, "p_007_PV")
    out["v_007"] = _mean_series(out, [
        "h_0211_PV", "h_0212_PV", "h_0213_PV", "h_0214_PV"])
    out["v_008"] = _mean_series(out, [
        "h_0221_PV", "h_0222_PV", "h_0223_PV", "h_0224_PV"])
    out["v_009"] = _mean_series(out, [
        "h_0181_PV", "h_0182_PV", "h_0183_PV", "h_0184_PV"])
    out["v_010"] = _mean_series(out, [
        "h_0191_PV", "h_0192_PV", "h_0193_PV", "h_0194_PV"])
    return out


def split_columns(df: pd.DataFrame, cfg: IOConfig
                  ) -> Tuple[List[str], List[str], Optional[str]]:
    """(io_cols, sensor_cols, step_col) 로 분리.

    io     = 순수 숫자 이름 (cfg.io_name_regex)
    sensor = 그 외 수치 컬럼 (meta/step 제외)
    step   = process step 컬럼 (있으면)
    """
    rgx = re.compile(cfg.io_name_regex)
    lower_skip = set(cfg.step_names) | set(cfg.time_names)
    io_cols, sensor_cols, step_col = [], [], None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in cfg.step_names and step_col is None:
            step_col = c
            continue
        if cl in lower_skip:
            continue
        name = str(c).strip()
        if rgx.match(name):
            io_cols.append(c)
        elif _is_h_sensor_name(name, cfg) or _looks_numeric(df[c]):
            sensor_cols.append(c)
    # IO 센서 선택: 화이트리스트(실질 동작 채널)만 유지
    if cfg.io_keep:
        keep = {str(x).strip() for x in cfg.io_keep}
        io_cols = [c for c in io_cols if str(c).strip() in keep]
    # 연속 센서 선택: configured prefix/exact whitelist 기준으로 유지한다.
    exact = {str(x).strip() for x in getattr(cfg, "sensor_keep", ())}
    prefixes = tuple(str(x) for x in getattr(cfg, "sensor_keep_prefixes", ()) or ())
    exclude = {str(x).strip() for x in getattr(cfg, "sensor_exclude", ())}
    h_cols = [c for c in sensor_cols if _is_h_sensor_name(str(c), cfg)]
    h_cols = [c for c in h_cols if _h_sensor_allowed(str(c), cfg)]
    virtual_cols = [c for c in sensor_cols if _is_virtual_sensor_name(str(c), cfg)]
    mode = str(getattr(cfg, "h_cart_mode", "") or "").lower()
    h_split = _is_h_split_profile(cfg)
    if h_split and mode == "exclude":
        raise ValueError(
            "--sensor-role-profile h-split-v1은 Cart류 10종을 미래 입력으로 "
            "사용하므로 --h-cart-mode exclude와 함께 사용할 수 없습니다."
        )
    if mode == "exclude":
        h_cols = [c for c in h_cols if not _is_cart_h_sensor_name(str(c), cfg)]
    if exact:
        selected = [c for c in sensor_cols if str(c).strip() in exact]
    elif prefixes:
        selected = [c for c in sensor_cols if str(c).strip().startswith(prefixes)]
    else:
        selected = list(sensor_cols)
    _append_unique(selected, h_cols)
    if h_split:
        _append_unique(selected, [c for c in sensor_cols if _is_cart_sensor_name(str(c))])
    if _virtual_sensor_mode(cfg) != "off":
        _append_unique(selected, virtual_cols)
    h_keep = _h_sensor_keep_set(cfg)
    if h_keep:
        selected = [
            c for c in selected
            if not _is_h_sensor_name(str(c), cfg) or str(c).strip() in h_keep
        ]
    if mode == "exclude":
        selected = [c for c in selected if not _is_cart_h_sensor_name(str(c), cfg)]
    sensor_cols = selected
    if exclude:
        sensor_cols = [c for c in sensor_cols if str(c).strip() not in exclude]
    _append_unique(sensor_cols, auxiliary_sensor_columns(cfg))
    return io_cols, sensor_cols, step_col


def _infer_schema(files: List[Path], cfg: IOConfig) -> Tuple[List[str], List[str], Optional[str]]:
    io_cols, sensor_cols, step_col = [], [], None
    nrows = int(getattr(cfg, "schema_infer_rows", 512) or 512)
    for f in files:
        df = _add_auxiliary_sensors(add_virtual_sensors(_read_data_file(f, cfg, nrows=nrows), cfg), cfg)
        ios, sensors, step = split_columns(df, cfg)
        _append_unique(io_cols, ios)
        _append_unique(sensor_cols, sensors)
        if step_col is None and step is not None:
            step_col = step
    return io_cols, sensor_cols, step_col


def _infer_recipe(path: Path, folder: Path, cfg: IOConfig) -> str:
    """Infer recipe from subdir, manual mapping, then underscore-delimited filename token."""
    try:
        rel = path.relative_to(folder)
        if len(rel.parts) > 1:
            part = rel.parts[0]
            if part.lower() not in {"ok", "ng", "train", "val"}:
                return part
    except Exception:
        pass
    manual = getattr(cfg, "manual_recipe_by_filename", {}) or {}
    candidates = [
        path.name,
        path.stem,
        path.name.lower(),
        path.stem.lower(),
    ]
    try:
        rel = path.relative_to(folder).as_posix()
        candidates.extend([rel, rel.lower()])
    except Exception:
        pass
    lower_manual = {str(k).lower(): v for k, v in manual.items()}
    for key in candidates:
        if key in manual:
            return str(manual[key])
        lk = str(key).lower()
        if lk in lower_manual:
            return str(lower_manual[lk])
    name = path.stem.lower()
    tokenized = f"_{name}_"
    for r in getattr(cfg, "recipe_names", ()) or ():
        rr = str(r).lower()
        if rr and f"_{rr}_" in tokenized:
            return str(r)
    return "unknown"


def load_dataset(folder: Path, cfg: IOConfig
                 ) -> Tuple[List[dict], List[str], List[str], int]:
    """폴더의 CSV 들을 wafer 단위로 로드.

    Returns:
      records: [{wafer_id, io(T,n_io), sensors(T,n_sensor), sensor_cols, step(T,)|None}]
      io_cols, sensor_cols (전 파일 union schema, 누락은 0 채움)
      n_steps_global (스텝 one-hot 차원)
    """
    folder = _resolve_existing_dir_ci(Path(folder))
    files = _dataset_files(folder, cfg)
    if not files:
        return [], [], [], 0
    # 컬럼 스키마는 전체 파일 union 기준. 첫 파일에 없는 H 센서도 입력에 포함한다.
    io_cols, sensor_cols, step_col = _infer_schema(files, cfg)

    records = []
    max_step = 0
    for f in files:
        df = _add_auxiliary_sensors(add_virtual_sensors(_read_data_file(f, cfg), cfg), cfg)
        T = len(df)
        io = np.zeros((T, len(io_cols)), dtype=np.float64)
        for j, c in enumerate(io_cols):
            if c in df.columns:
                io[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0).values
        sen = np.zeros((T, len(sensor_cols)), dtype=np.float64)
        for j, c in enumerate(sensor_cols):
            if c in df.columns:
                sen[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0).values
        step = None
        if step_col and step_col in df.columns:
            step = pd.to_numeric(df[step_col], errors="coerce").fillna(0).astype(int).values
            max_step = max(max_step, int(step.max()) + 1 if len(step) else 0)
        records.append({"wafer_id": f.stem, "recipe": _infer_recipe(f, folder, cfg),
                        "source_path": str(f),
                        "io": io, "io_cols": list(io_cols),
                        "sensors": sen, "sensor_cols": list(sensor_cols), "step": step})
    return records, io_cols, sensor_cols, max_step
