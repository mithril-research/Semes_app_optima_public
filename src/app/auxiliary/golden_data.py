from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from src import sensor_names as SN
import pandas as pd

from .sensors import (
    CYLINDER_DIFF,
    CYLINDER_FEATURES,
    CYLINDER_LEFT,
    CYLINDER_RIGHT,
    P_SENSORS,
    T_SENSORS,
    VIRTUAL_PAIRS,
)


@dataclass
class GoldenSample:
    path: Path
    log_id: str
    equipment: str
    chamber: str
    recipe: str
    lot_id: str
    frame: pd.DataFrame


def _normalize_yyyymmdd(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) != 8:
        raise ValueError(f"날짜는 YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD 형식이어야 합니다: {value}")
    return int(digits)


def infer_file_date_yyyymmdd(path: Path) -> int | None:
    # Flattened 파일은 보통 파일명에 날짜가 있고, 원본 구조는 상위 폴더에
    # yyyymmddhh-SEQUS가 있다. 날짜 필터는 파일을 열기 전에 적용해야 하므로
    # 파일 내용이 아니라 경로/파일명 토큰만 사용한다.
    search_parts = [path.stem, *reversed(path.parts[:-1])]
    for part in search_parts:
        for token in re.split(r"[_\-\s()]+", part):
            if re.fullmatch(r"\d{8,14}", token):
                return int(token[:8])
        m = re.search(r"(?<!\d)(\d{8})(?:\d{2,6})?(?!\d)", part)
        if m:
            return int(m.group(1))
    return None


def iter_process_files(
    root: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Path]:
    start = _normalize_yyyymmdd(start_date)
    end = _normalize_yyyymmdd(end_date)
    if start is not None and end is not None and start > end:
        raise ValueError(f"시작 날짜가 종료 날짜보다 늦습니다: {start_date} > {end_date}")
    files: list[Path] = []
    for pattern in ("*.log", "*.csv"):
        files.extend(root.rglob(pattern))
    files = sorted(files, key=lambda p: (p.name, str(p)))
    if start is None and end is None:
        return files

    selected: list[Path] = []
    skipped_unknown = 0
    for path in files:
        file_date = infer_file_date_yyyymmdd(path)
        if file_date is None:
            skipped_unknown += 1
            continue
        if start is not None and file_date < start:
            continue
        if end is not None and file_date > end:
            continue
        selected.append(path)
    if skipped_unknown:
        print(f"[warn] 날짜를 추정할 수 없어 제외한 파일 수: {skipped_unknown}")
    return selected


def read_process_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".log":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"지원하지 않는 확장자: {path}")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    if "ProcessStep" not in df.columns:
        raise ValueError(f"필수 컬럼 누락(ProcessStep): {path}")
    df = df[pd.to_numeric(df["ProcessStep"], errors="coerce").notna()].copy()
    df["ProcessStep"] = pd.to_numeric(df["ProcessStep"], errors="coerce").astype(int)
    return SN.frame(df)


def parse_timestamp(df: pd.DataFrame) -> np.ndarray:
    if "Date" in df.columns and "Time" in df.columns:
        ts = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
            errors="coerce",
        )
        if ts.notna().any():
            sec = (ts - ts.dropna().iloc[0]).dt.total_seconds().to_numpy(np.float64)
            return np.nan_to_num(sec, nan=0.0)
    return np.arange(len(df), dtype=np.float64)


def infer_chamber(path: Path, chamber_regex: str = r"PMCUNIT\d+") -> str:
    text = "/".join(path.parts)
    m = re.search(chamber_regex, text, flags=re.IGNORECASE)
    return m.group(0).upper() if m else "UNKNOWN"


def _normalize_chamber_token(value: str) -> str:
    text = str(value or "").strip().upper()
    if re.fullmatch(r"\d+", text):
        return f"PMCUNIT{int(text)}"
    return text


def chamber_matches(chamber: str, requested: str | None) -> bool:
    if requested is None:
        return True
    text = str(requested).strip()
    if not text or text.lower() in {"all", "*"}:
        return True
    current = _normalize_chamber_token(chamber)
    wanted = [
        _normalize_chamber_token(tok)
        for tok in re.split(r"[,;\s]+", text)
        if str(tok).strip()
    ]
    return not wanted or current in wanted


def _infer_equipment_from_flattened_stem(stem: str) -> str | None:
    tokens = [tok for tok in re.split(r"_+", stem) if tok]
    date_idx = next((i for i, tok in enumerate(tokens) if re.fullmatch(r"\d{8,14}", tok)), None)
    if date_idx is None:
        return None
    prefix_tokens = tokens[:date_idx]
    for tok in prefix_tokens:
        if re.fullmatch(r"WUK\d+[A-Za-z0-9]*", tok, flags=re.IGNORECASE):
            return tok.upper()
    if prefix_tokens:
        return "_".join(prefix_tokens).upper()
    return None


def infer_equipment(path: Path, equipment_regex: str | None = None) -> str:
    text = "/".join(path.parts)
    if equipment_regex:
        m = re.search(equipment_regex, text, flags=re.IGNORECASE)
        if m:
            return (m.group(1) if m.groups() else m.group(0)).upper()
    # Flattened examples:
    #   WUK370A_2026091508_PMCUNIT1_29a_xxx
    #   WUK370A_info1_info2_2026091508_PMCUNIT1_29a_xxx
    #   001_WUK370A_2026091508_PMCUNIT1_29a_xxx
    flattened_equipment = _infer_equipment_from_flattened_stem(path.stem)
    if flattened_equipment:
        return flattened_equipment
    # Original folder format: root/equipment/yyyyMMddHH-SEQUS/unit/file.log.
    for idx, part in enumerate(path.parts):
        if re.match(r"\d{8,14}-SEQUS$", part, flags=re.IGNORECASE) and idx > 0:
            return path.parts[idx - 1].upper()
    m = re.search(r"(WUK\d+[A-Za-z]?)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "UNKNOWN_EQUIPMENT"


def infer_recipe(path: Path, df: pd.DataFrame) -> str:
    # Flattened files usually contain the recipe in the filename. Original logs
    # often carry it near column 9 in row 0, wrapped in quotes.
    stem = path.stem
    m = re.search(r"(?<![A-Za-z0-9])([0-9][0-9a-zA-Z]{2})(?![A-Za-z0-9])", stem)
    if m:
        return m.group(1).lower()
    if df.shape[1] > 9 and len(df) > 0:
        val = str(df.iloc[0, 9]).strip().strip("'\"")
        if val:
            return val[:3].lower()
    return "unknown"


def infer_lot_id(path: Path, lot_regex: str | None = None) -> str:
    text = path.stem
    if lot_regex:
        m = re.search(lot_regex, text)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return path.parent.name


def load_samples(
    root: Path,
    *,
    recipe_prefix: str | None,
    equipment_regex: str | None,
    chamber_regex: str,
    chamber_filter: str | None = None,
    lot_regex: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[GoldenSample]:
    samples: list[GoldenSample] = []
    for path in iter_process_files(
        root,
        start_date=start_date,
        end_date=end_date,
    ):
        df = read_process_file(path)
        recipe = infer_recipe(path, df)
        if recipe_prefix and recipe.lower()[: len(recipe_prefix)] != recipe_prefix.lower():
            continue
        chamber = infer_chamber(path, chamber_regex)
        if not chamber_matches(chamber, chamber_filter):
            continue
        samples.append(
            GoldenSample(
                path=path,
                log_id=path.stem,
                equipment=infer_equipment(path, equipment_regex),
                chamber=chamber,
                recipe=recipe,
                lot_id=infer_lot_id(path, lot_regex),
                frame=df,
            )
        )
    return samples


def feature_names(
    include_derivatives: bool = True,
    feature_profile: str = "process",
) -> list[str]:
    if feature_profile == "cylinder":
        return list(CYLINDER_FEATURES)
    if feature_profile != "process":
        raise ValueError(f"지원하지 않는 feature profile: {feature_profile}")
    base = list(P_SENSORS) + list(T_SENSORS) + [name for name, _, _ in VIRTUAL_PAIRS]
    if include_derivatives:
        base += [f"d_{name}" for name in base]
    return base


def build_feature_frame(
    df: pd.DataFrame,
    include_derivatives: bool = True,
    feature_profile: str = "process",
) -> pd.DataFrame:
    if feature_profile == "cylinder":
        missing = [name for name in (CYLINDER_RIGHT, CYLINDER_LEFT) if name not in df.columns]
        if missing:
            raise ValueError(f"실린더 필수 센서 컬럼 누락: {', '.join(missing)}")
        out = pd.DataFrame(index=df.index)
        for name in (CYLINDER_RIGHT, CYLINDER_LEFT):
            values = pd.to_numeric(df[name], errors="coerce").interpolate().ffill().bfill()
            if values.isna().any():
                raise ValueError(f"실린더 센서에 유효한 숫자값이 없습니다: {name}")
            out[name] = values
        out[CYLINDER_DIFF] = out[CYLINDER_RIGHT] - out[CYLINDER_LEFT]
        return out.astype(np.float32)
    if feature_profile != "process":
        raise ValueError(f"지원하지 않는 feature profile: {feature_profile}")

    out = pd.DataFrame(index=df.index)
    for name in P_SENSORS + T_SENSORS:
        if name in df.columns:
            out[name] = pd.to_numeric(df[name], errors="coerce").interpolate().ffill().bfill().fillna(0.0)
        else:
            out[name] = 0.0
    for virt, left, right in VIRTUAL_PAIRS:
        out[virt] = out[left] - out[right]
    if include_derivatives:
        t = parse_timestamp(df)
        dt = np.diff(t, prepend=t[0])
        valid = np.isfinite(dt) & (dt >= 0.03) & (dt <= 0.30)
        dt = np.where(valid, dt, np.nan)
        median_dt = float(np.nanmedian(dt[1:])) if np.isfinite(dt[1:]).any() else 0.1
        dt = np.nan_to_num(dt, nan=median_dt)
        dt[0] = median_dt
        base_cols = list(out.columns)
        for col in base_cols:
            vals = out[col].to_numpy(np.float64)
            der = np.diff(vals, prepend=vals[0]) / np.maximum(dt, 1e-6)
            der[0] = 0.0
            lo, hi = np.nanpercentile(der, [1.0, 99.0])
            out[f"d_{col}"] = np.clip(der, lo, hi)
    return out.astype(np.float32)


def fit_step_lengths(samples: list[GoldenSample], min_points: int = 4) -> dict[int, int]:
    counts: dict[int, list[int]] = {}
    for sample in samples:
        steps = sample.frame["ProcessStep"].to_numpy(int)
        for step in sorted(set(int(x) for x in steps if int(x) > 0)):
            counts.setdefault(step, []).append(int((steps == step).sum()))
    if not counts:
        raise ValueError("ProcessStep 1 이상인 데이터가 없습니다.")
    return {step: max(min_points, int(round(float(np.median(vals))))) for step, vals in counts.items()}


def step_offsets(step_lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    cur = 0
    offsets: dict[int, tuple[int, int]] = {}
    for step in sorted(step_lengths):
        nxt = cur + int(step_lengths[step])
        offsets[int(step)] = (cur, nxt)
        cur = nxt
    return offsets


def align_sample(
    sample: GoldenSample,
    step_lengths: dict[int, int],
    include_derivatives: bool,
    feature_profile: str = "process",
) -> np.ndarray:
    feat = build_feature_frame(
        sample.frame,
        include_derivatives=include_derivatives,
        feature_profile=feature_profile,
    )
    steps = sample.frame["ProcessStep"].to_numpy(int)
    aligned: list[np.ndarray] = []
    for step in sorted(step_lengths):
        n_out = int(step_lengths[step])
        part = feat.loc[steps == step].to_numpy(np.float32)
        if len(part) == 0:
            aligned.append(np.zeros((n_out, feat.shape[1]), dtype=np.float32))
            continue
        if len(part) == 1:
            aligned.append(np.repeat(part, n_out, axis=0).astype(np.float32))
            continue
        x_old = np.linspace(0.0, 1.0, len(part), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
        cols = [np.interp(x_new, x_old, part[:, j]) for j in range(part.shape[1])]
        aligned.append(np.stack(cols, axis=1).astype(np.float32))
    return np.concatenate(aligned, axis=0)


def original_feature_sequence(
    sample: GoldenSample,
    include_derivatives: bool,
    feature_profile: str = "process",
) -> tuple[np.ndarray, np.ndarray]:
    """Return features and ProcessStep values on the file's original time grid."""
    features = build_feature_frame(
        sample.frame,
        include_derivatives=include_derivatives,
        feature_profile=feature_profile,
    ).to_numpy(np.float32)
    steps = sample.frame["ProcessStep"].to_numpy(int)
    return features, steps


def sample_end_timestamp(sample: GoldenSample) -> pd.Timestamp | None:
    """Return the last valid Date/Time value recorded in a process file."""
    df = sample.frame
    if "Date" in df.columns and "Time" in df.columns:
        timestamps = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
            errors="coerce",
        )
        valid = timestamps.dropna()
        if not valid.empty:
            return pd.Timestamp(valid.iloc[-1])
    inferred = infer_file_date_yyyymmdd(sample.path)
    if inferred is None:
        return None
    return pd.to_datetime(str(inferred), format="%Y%m%d", errors="coerce")


def sample_row_timestamp(sample: GoldenSample, row_index: int) -> pd.Timestamp | None:
    """Return the Date/Time timestamp at an original file row."""
    if row_index < 0 or row_index >= len(sample.frame):
        return None
    df = sample.frame
    if "Date" not in df.columns or "Time" not in df.columns:
        return None
    value = pd.to_datetime(
        f"{str(df.iloc[row_index]['Date']).strip()} {str(df.iloc[row_index]['Time']).strip()}",
        errors="coerce",
    )
    return None if pd.isna(value) else pd.Timestamp(value)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[None, :]) / self.std[None, :]

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std[None, :] + self.mean[None, :]

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, data: dict) -> "Standardizer":
        return cls(np.asarray(data["mean"], dtype=np.float32), np.asarray(data["std"], dtype=np.float32))


def fit_standardizer(xs: Iterable[np.ndarray]) -> Standardizer:
    arr = np.concatenate([x.reshape(-1, x.shape[-1]) for x in xs], axis=0)
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return Standardizer(mean=mean, std=std)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
