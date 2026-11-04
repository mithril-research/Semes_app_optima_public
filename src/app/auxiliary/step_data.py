from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from src import sensor_names as SN
import pandas as pd


@dataclass(frozen=True)
class StepTimeSample:
    path: Path
    log_id: str
    equipment: str
    chamber: str
    recipe: str
    lot_id: str
    end_timestamp: pd.Timestamp
    durations: dict[int, float]


@dataclass(frozen=True)
class StepRun:
    """One contiguous ProcessStep occurrence in an original process log."""

    step: int
    occurrence: int
    start_index: int
    end_index: int
    boundary_index: int | None
    duration_seconds: float


def iter_process_files(root: Path) -> list[Path]:
    files = [*root.rglob("*.log"), *root.rglob("*.csv")]
    return sorted(files, key=lambda path: (path.name, str(path)))


def read_process_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".log":
        frame = pd.read_csv(path, sep="\t", low_memory=False)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {path}")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")].copy()
    missing = [name for name in ("Date", "Time", "ProcessStep") if name not in frame.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락({', '.join(missing)}): {path}")
    numeric_step = pd.to_numeric(frame["ProcessStep"], errors="coerce")
    frame = frame.loc[numeric_step.notna()].copy().reset_index(drop=True)
    frame["ProcessStep"] = numeric_step.loc[numeric_step.notna()].astype(int).to_numpy()
    if frame.empty:
        raise ValueError(f"유효한 ProcessStep 행이 없습니다: {path}")
    return SN.frame(frame)


def parse_timestamps(frame: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(
        frame["Date"].astype(str).str.strip()
        + " "
        + frame["Time"].astype(str).str.strip(),
        errors="coerce",
    )
    active = frame["ProcessStep"].to_numpy(int) > 0
    if not active.any():
        raise ValueError("ProcessStep 1 이상인 행이 없습니다.")
    if timestamps.loc[active].isna().any():
        count = int(timestamps.loc[active].isna().sum())
        raise ValueError(f"ProcessStep 1 이상 구간의 Date/Time 변환 실패: {count}행")
    return timestamps


def extract_step_runs(frame: pd.DataFrame) -> tuple[list[StepRun], pd.Timestamp]:
    """Return contiguous step occurrences without changing the legacy duration rule.

    A boundary sample belonging to the next step is retained separately.  It is
    useful for PV transition analysis because a controller can change
    ProcessStep on the same sample at which a pressure threshold is crossed.
    """
    timestamps = parse_timestamps(frame)
    steps = frame["ProcessStep"].to_numpy(int)
    active_indices = np.flatnonzero(steps > 0)
    active_timestamp_values = timestamps.iloc[active_indices]
    active_times = (
        active_timestamp_values - active_timestamp_values.iloc[0]
    ).dt.total_seconds().to_numpy(np.float64)
    positive_dt = np.diff(active_times)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
    typical_dt = float(np.median(positive_dt)) if positive_dt.size else 0.1
    typical_dt = max(typical_dt, 1e-6)

    runs: list[StepRun] = []
    occurrences: Counter[int] = Counter()
    run_start = int(active_indices[0])
    previous = run_start
    previous_step = int(steps[run_start])
    for current in active_indices[1:]:
        current = int(current)
        current_step = int(steps[current])
        if current != previous + 1 or current_step != previous_step:
            if current == previous + 1:
                elapsed = (
                    timestamps.iloc[current] - timestamps.iloc[run_start]
                ).total_seconds()
                boundary_index = current
            else:
                elapsed = (
                    timestamps.iloc[previous] - timestamps.iloc[run_start]
                ).total_seconds() + typical_dt
                boundary_index = None
            occurrences[previous_step] += 1
            runs.append(
                StepRun(
                    step=previous_step,
                    occurrence=int(occurrences[previous_step]),
                    start_index=run_start,
                    end_index=previous,
                    boundary_index=boundary_index,
                    duration_seconds=round(max(float(elapsed), typical_dt), 6),
                )
            )
            run_start = current
            previous_step = current_step
        previous = current
    elapsed = (
        timestamps.iloc[previous] - timestamps.iloc[run_start]
    ).total_seconds() + typical_dt
    occurrences[previous_step] += 1
    runs.append(
        StepRun(
            step=previous_step,
            occurrence=int(occurrences[previous_step]),
            start_index=run_start,
            end_index=previous,
            boundary_index=None,
            duration_seconds=round(max(float(elapsed), typical_dt), 6),
        )
    )
    return runs, pd.Timestamp(timestamps.iloc[active_indices[-1]])


def extract_step_durations(frame: pd.DataFrame) -> tuple[dict[int, float], pd.Timestamp]:
    runs, end_timestamp = extract_step_runs(frame)
    durations: dict[int, float] = {}
    for run in runs:
        durations[run.step] = durations.get(run.step, 0.0) + run.duration_seconds
    durations = {step: round(value, 6) for step, value in durations.items()}
    return durations, end_timestamp


def infer_equipment(path: Path, equipment_regex: str | None = None) -> str:
    text = "/".join(path.parts)
    if equipment_regex:
        match = re.search(equipment_regex, text, flags=re.IGNORECASE)
        if match:
            return (match.group(1) if match.groups() else match.group(0)).upper()
    match = re.search(r"(WUK\d+[A-Za-z0-9]*)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for index, part in enumerate(path.parts):
        if re.fullmatch(r"\d{8,14}-SEQUS", part, flags=re.IGNORECASE) and index > 0:
            return path.parts[index - 1].upper()
    return "UNKNOWN_EQUIPMENT"


def infer_chamber(path: Path, chamber_regex: str = r"PMCUNIT\d+") -> str:
    match = re.search(chamber_regex, "/".join(path.parts), flags=re.IGNORECASE)
    return match.group(0).upper() if match else "UNKNOWN_CHAMBER"


def infer_recipe(path: Path, frame: pd.DataFrame) -> str:
    match = re.search(
        r"(?<![A-Za-z0-9])([0-9][0-9A-Za-z]{2})(?![A-Za-z0-9])",
        path.stem,
    )
    if match:
        return match.group(1).lower()
    if len(frame) and frame.shape[1] > 9:
        value = str(frame.iloc[0, 9]).strip().strip("'\"")
        if value:
            return value[:3].lower()
    return "unknown"


def infer_lot(path: Path, lot_regex: str | None = None) -> str:
    if lot_regex:
        match = re.search(lot_regex, path.stem)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return path.parent.name


def _normalize_chamber(value: str) -> str:
    text = str(value).strip().upper()
    return f"PMCUNIT{int(text)}" if re.fullmatch(r"\d+", text) else text


def load_samples(
    root: Path,
    *,
    recipe: str | None = None,
    chamber: str | None = None,
    equipment_regex: str | None = None,
    chamber_regex: str = r"PMCUNIT\d+",
    lot_regex: str | None = None,
    on_sample: Callable[[StepTimeSample, pd.DataFrame], None] | None = None,
) -> tuple[list[StepTimeSample], list[dict[str, str]]]:
    from tqdm import tqdm
    samples: list[StepTimeSample] = []
    warnings: list[dict[str, str]] = []
    recipe_filter = None if not recipe or recipe.lower() in {"all", "*"} else recipe.lower()
    chamber_filter = None if not chamber or chamber.lower() in {"all", "*"} else {
        _normalize_chamber(value)
        for value in re.split(r"[,;\s]+", chamber)
        if value.strip()
    }
    files = iter_process_files(root)
    for path in tqdm(files, desc=f"read {root.name}", unit="file"):
        try:
            frame = read_process_file(path)
            sample_recipe = infer_recipe(path, frame)
            sample_chamber = infer_chamber(path, chamber_regex)
            if recipe_filter and not sample_recipe.startswith(recipe_filter):
                continue
            if chamber_filter and _normalize_chamber(sample_chamber) not in chamber_filter:
                continue
            durations, end_timestamp = extract_step_durations(frame)
            sample = StepTimeSample(
                path=path,
                log_id=path.stem,
                equipment=infer_equipment(path, equipment_regex),
                chamber=sample_chamber,
                recipe=sample_recipe,
                lot_id=infer_lot(path, lot_regex),
                end_timestamp=end_timestamp,
                durations=durations,
            )
            samples.append(sample)
            if on_sample is not None:
                on_sample(sample, frame)
        except Exception as exc:
            warnings.append({"path": str(path), "reason": str(exc)})
    return samples, warnings


def modal_step_schema(samples: list[StepTimeSample]) -> tuple[int, ...]:
    if not samples:
        raise ValueError("스텝 구성을 계산할 샘플이 없습니다.")
    counts = Counter(tuple(sorted(sample.durations)) for sample in samples)
    return counts.most_common(1)[0][0]


def vectorize_samples(
    samples: list[StepTimeSample],
    steps: tuple[int, ...],
    *,
    impute: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, ...]], list[tuple[int, ...]]]:
    expected = set(steps)
    matrix: list[list[float]] = []
    missing_rows: list[tuple[int, ...]] = []
    extra_rows: list[tuple[int, ...]] = []
    for sample in samples:
        actual = set(sample.durations)
        missing = tuple(sorted(expected - actual))
        extra = tuple(sorted(actual - expected))
        row = []
        for index, step in enumerate(steps):
            if step in sample.durations:
                row.append(float(sample.durations[step]))
            elif impute is not None:
                row.append(float(impute[index]))
            else:
                row.append(float("nan"))
        matrix.append(row)
        missing_rows.append(missing)
        extra_rows.append(extra)
    return np.asarray(matrix, dtype=np.float32), missing_rows, extra_rows
