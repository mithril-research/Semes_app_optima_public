"""Organize flattened SEMES log filenames into controller/unit/date folders.

Input filename:
    controller_yyyymmddhh_unit_recipe_original_name.log

Output path:
    controller/unit/yyyymmdd/yyyymmdd_001_recipeoriginal_name.log

Sequence numbers are assigned within each controller/unit/yyyymmdd folder by
the last row's Time column in each tab-separated .log file.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


INVALID_FILENAME_CHARS = '<>:"/\\|?*'
DATETIME_RE = re.compile(r"^\d{10}$")
TIME_RE = re.compile(r"^\s*(\d+):(\d{2}):(\d{2})(?:\.(\d{1,6}))?\s*$")


@dataclass(frozen=True)
class LogItem:
    src: Path
    controller: str
    yyyymmdd: str
    unit: str
    recipe: str
    original_stem: str
    last_time_key: int

    @property
    def group_key(self) -> tuple[str, str, str]:
        return self.controller, self.unit, self.yyyymmdd


def _safe_component(value: object) -> str:
    text = str(value).strip()
    trans = {ord(ch): "_" for ch in INVALID_FILENAME_CHARS}
    text = text.translate(trans)
    text = "".join("_" if ord(ch) < 32 else ch for ch in text)
    text = text.strip(" .")
    return text or "_"


def _parse_filename(path: Path) -> tuple[str, str, str, str, str]:
    parts = path.stem.split("_")
    dt_idx = next((i for i, part in enumerate(parts) if DATETIME_RE.match(part)), None)
    if dt_idx is None:
        raise ValueError("filename must include a 10-digit yyyymmddhh token")
    if dt_idx == 0:
        raise ValueError("controller token missing before yyyymmddhh")
    if len(parts) < dt_idx + 4:
        raise ValueError(
            "filename must be controller_yyyymmddhh_unit_recipe_original.log"
        )

    controller = "_".join(parts[:dt_idx])
    yyyymmddhh = parts[dt_idx]
    unit = parts[dt_idx + 1]
    recipe = parts[dt_idx + 2]
    original_stem = "_".join(parts[dt_idx + 3:])
    if not original_stem:
        raise ValueError("original filename token missing")
    return (
        _safe_component(controller),
        yyyymmddhh[:8],
        _safe_component(unit),
        _safe_component(recipe),
        _safe_component(original_stem),
    )


def _read_time_column(path: Path) -> pd.Series:
    last_error = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                encoding=enc,
                usecols=lambda c: str(c).strip() == "Time",
            )
            if df.empty or df.shape[1] < 1:
                raise ValueError("Time column missing or empty")
            return df.iloc[:, 0]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except ValueError:
            raise
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(f"failed to read Time column: {last_error}")


def _parse_time_value(value: object) -> int:
    text = str(value).strip()
    m = TIME_RE.match(text)
    if not m:
        raise ValueError(f"invalid Time value: {text!r}")
    hours = int(m.group(1))
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    fraction = (m.group(4) or "")[:6].ljust(6, "0")
    micros = int(fraction or "0")
    return (((hours * 60) + minutes) * 60 + seconds) * 1_000_000 + micros


def _last_time_key(path: Path) -> int:
    series = _read_time_column(path).dropna()
    if series.empty:
        raise ValueError("Time column has no valid rows")
    return _parse_time_value(series.iloc[-1])


def _make_item(path: Path) -> LogItem:
    controller, yyyymmdd, unit, recipe, original_stem = _parse_filename(path)
    return LogItem(
        src=path,
        controller=controller,
        yyyymmdd=yyyymmdd,
        unit=unit,
        recipe=recipe,
        original_stem=original_stem,
        last_time_key=_last_time_key(path),
    )


def _destination(item: LogItem, tgt_root: Path, seq: int) -> Path:
    name = f"{item.yyyymmdd}_{seq:03d}_{item.recipe}{item.original_stem}.log"
    return tgt_root / item.controller / item.unit / item.yyyymmdd / name


def _copy_or_move(src: Path, dst: Path, move: bool, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"target exists: {dst}")
        if dst.is_dir():
            raise IsADirectoryError(f"target is a directory: {dst}")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def _make_progress(total: int, action: str):
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=action, unit="file", file=sys.stdout)


def _progress_write(progress, message: str, file=None) -> None:
    if progress is None:
        print(message, file=file or sys.stdout)
        return
    progress.write(message, file=file)


def _print_errors(errors: list[str]) -> None:
    if not errors:
        return
    print(f"errors: {len(errors)}", file=sys.stderr)
    for msg in errors[:20]:
        print(f"  - {msg}", file=sys.stderr)
    if len(errors) > 20:
        print(f"  ... {len(errors) - 20} more", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Copy or move flattened .log files into "
            "controller/unit/yyyymmdd folders and renumber them by final Time."
        )
    )
    ap.add_argument("src", help="source directory containing flattened .log files")
    ap.add_argument("tgt", help="target root directory")
    ap.add_argument("--move", action="store_true",
                    help="move files instead of copying them")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing target files")
    ap.add_argument("--skip-invalid", action="store_true",
                    help="skip invalid files instead of returning failure")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    tgt_root = Path(args.tgt).resolve()
    if not src_root.exists() or not src_root.is_dir():
        print(f"[error] source directory not found: {src_root}", file=sys.stderr)
        return 1
    if src_root == tgt_root or src_root in tgt_root.parents:
        print("[error] target directory must not be inside source directory",
              file=sys.stderr)
        return 1

    files = sorted(
        p for p in src_root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".log"
    )
    if not files:
        print(f"[error] no .log files found: {src_root}", file=sys.stderr)
        return 1

    print(f"source: {src_root}", flush=True)
    print(f"target: {tgt_root}", flush=True)
    print(f"files: {len(files)} | action: {'move' if args.move else 'copy'}", flush=True)

    items: list[LogItem] = []
    errors: list[str] = []
    progress = _make_progress(len(files), "scan")
    try:
        for path in files:
            try:
                items.append(_make_item(path))
            except Exception as exc:
                msg = f"{path}: {exc}"
                errors.append(msg)
                _progress_write(progress, f"[invalid] {msg}", file=sys.stderr)
            finally:
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    if errors and not args.skip_invalid:
        _print_errors(errors)
        return 1
    if not items:
        _print_errors(errors)
        print("[error] no valid .log files to organize", file=sys.stderr)
        return 1

    items_by_group: dict[tuple[str, str, str], list[LogItem]] = {}
    for item in items:
        items_by_group.setdefault(item.group_key, []).append(item)

    planned: list[tuple[LogItem, Path]] = []
    for key in sorted(items_by_group):
        group_items = sorted(
            items_by_group[key],
            key=lambda x: (x.last_time_key, str(x.src)),
        )
        for seq, item in enumerate(group_items, start=1):
            planned.append((item, _destination(item, tgt_root, seq)))

    ok = 0
    copy_errors: list[str] = []
    progress = _make_progress(len(planned), "move" if args.move else "copy")
    try:
        for item, dst in planned:
            try:
                _copy_or_move(item.src, dst, args.move, args.overwrite)
                ok += 1
                _progress_write(progress, f"[ok] {item.src.name} -> {dst}")
            except Exception as exc:
                msg = f"{item.src}: {exc}"
                copy_errors.append(msg)
                _progress_write(progress, f"[fail] {msg}", file=sys.stderr)
            finally:
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    print(f"\ncompleted: {ok}/{len(planned)}")
    if errors:
        print(f"skipped invalid: {len(errors)}")
    if copy_errors:
        _print_errors(copy_errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
