"""Flatten SEMES SEQUS log files into one target folder.

Expected source layout:
    src/controller/datetime-SEQUS/unit/file.log

Output filename:
    controller_datetime_unit_recipe3_file.log

The recipe is read from df.iloc[0, 9] after loading the log with tab separator.
By default files are copied and originals are kept. Use --move to move them.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


INVALID_FILENAME_CHARS = '<>:"/\\|?*'
TARGET_UNITS = {f"PMCUNIT{i}" for i in range(7, 13)}


def _safe_component(value: object) -> str:
    text = str(value).strip()
    trans = {ord(ch): "_" for ch in INVALID_FILENAME_CHARS}
    text = text.translate(trans)
    text = "".join("_" if ord(ch) < 32 else ch for ch in text)
    text = text.strip(" .")
    return text or "_"


def _read_log_head(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            return pd.read_csv(path, sep="\t", encoding=enc, nrows=1)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep="\t", nrows=1)


def _recipe_prefix(path: Path) -> str:
    df = _read_log_head(path)
    if df.empty:
        raise ValueError("empty log")
    if df.shape[1] <= 9:
        raise ValueError(f"recipe column missing: columns={df.shape[1]}, need >=10")
    recipe = str(df.iloc[0, 9]).replace("'", "").strip()
    if not recipe or recipe.lower() == "nan":
        raise ValueError("empty recipe")
    return _safe_component(recipe[:3])


def _metadata_from_path(path: Path, src_root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(src_root)
    if len(rel.parts) < 4:
        raise ValueError(
            "path must be src/controller/datetime-SEQUS/unit/file.log"
        )
    controller = rel.parts[0]
    sequs_dir = rel.parts[1]
    unit = rel.parts[2]
    suffix = "-SEQUS"
    if not sequs_dir.upper().endswith(suffix):
        raise ValueError(f"datetime folder must end with {suffix}: {sequs_dir}")
    datetime = sequs_dir[: -len(suffix)]
    if not datetime:
        raise ValueError(f"empty datetime before {suffix}: {sequs_dir}")
    return (
        _safe_component(controller),
        _safe_component(datetime),
        _safe_component(unit),
    )


def _destination_for(path: Path, src_root: Path, tgt_root: Path) -> Path:
    controller, datetime, unit = _metadata_from_path(path, src_root)
    recipe3 = _recipe_prefix(path)
    filename = _safe_component(path.name)
    return tgt_root / f"{controller}_{datetime}_{unit}_{recipe3}_{filename}"


def _is_target_unit_log(path: Path, src_root: Path) -> bool:
    try:
        rel = path.relative_to(src_root)
    except ValueError:
        return False
    if len(rel.parts) < 4:
        return False
    unit = str(rel.parts[2]).strip().upper()
    return unit in TARGET_UNITS


def _log_files(src_root: Path) -> tuple[list[Path], int]:
    all_logs = sorted(
        p for p in src_root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".log"
    )
    selected = [p for p in all_logs if _is_target_unit_log(p, src_root)]
    return selected, len(all_logs) - len(selected)


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Copy or move src/controller/datetime-SEQUS/unit/*.log files into "
            "one flat target folder with controller_datetime_unit_recipe3 prefix."
        )
    )
    ap.add_argument("src", help="source root directory")
    ap.add_argument("tgt", help="target directory")
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

    files, skipped_units = _log_files(src_root)
    if not files:
        print(f"[error] no PMCUNIT7~PMCUNIT12 .log files found: {src_root}",
              file=sys.stderr)
        return 1

    action = "move" if args.move else "copy"
    print(f"source: {src_root}", flush=True)
    print(f"target: {tgt_root}", flush=True)
    print(
        f"files: {len(files)} | skipped non-target units: {skipped_units} "
        f"| action: {action}\n",
        flush=True,
    )

    ok = 0
    errors: list[str] = []
    progress = _make_progress(len(files), action)
    try:
        for src in files:
            try:
                dst = _destination_for(src, src_root, tgt_root)
                _copy_or_move(src, dst, args.move, args.overwrite)
                ok += 1
                _progress_write(
                    progress, f"[ok] {src.relative_to(src_root)} -> {dst.name}"
                )
            except Exception as exc:
                msg = f"{src}: {exc}"
                errors.append(msg)
                _progress_write(progress, f"[fail] {msg}", file=sys.stderr)
                if not args.skip_invalid:
                    continue
            finally:
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    print(f"\ncompleted: {ok}/{len(files)}")
    if errors:
        print(f"errors: {len(errors)}", file=sys.stderr)
        for msg in errors[:20]:
            print(f"  - {msg}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more", file=sys.stderr)
        return 0 if args.skip_invalid else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
