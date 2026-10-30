"""Copy or move logs for one recipe from SEQUS folders.

Expected source layout:
    root/yyyyMMddHH-SEQUS/unit/files.log

Output filename:
    equipment_yyyyMMddHH_recipefilename.log

The recipe is read from df.iloc[0, 9] after loading each .log file with a tab
separator. Quotes are stripped, then the first three characters are used for
matching and output filenames. By default files are copied and originals are
kept. Use --move to move them instead.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


INVALID_FILENAME_CHARS = '<>:"/\\|?*'
SEQUS_SUFFIX = "-SEQUS"


@dataclass(frozen=True)
class MatchedLog:
    src: Path
    sequs_datetime: str
    unit: str
    recipe: str
    recipe_key: str


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


def _read_recipe(path: Path) -> str:
    df = _read_log_head(path)
    if df.empty:
        raise ValueError("empty log")
    if df.shape[1] <= 9:
        raise ValueError(f"recipe column missing: columns={df.shape[1]}, need >=10")
    recipe = str(df.iloc[0, 9]).replace("'", "").strip()
    if not recipe or recipe.lower() == "nan":
        raise ValueError("empty recipe")
    return recipe


def _recipe_key(recipe: str) -> str:
    key = str(recipe).replace("'", "").strip()[:3]
    if not key or key.lower() == "nan":
        raise ValueError(f"empty recipe key from recipe: {recipe!r}")
    return key


def _sequs_datetime_from_dir(dirname: str) -> str:
    text = str(dirname).strip()
    if not text.upper().endswith(SEQUS_SUFFIX):
        raise ValueError(f"SEQUS folder must end with {SEQUS_SUFFIX}: {dirname}")
    value = text[: -len(SEQUS_SUFFIX)]
    if not value:
        raise ValueError(f"empty datetime before {SEQUS_SUFFIX}: {dirname}")
    return value


def _log_metadata(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    if len(rel.parts) < 3:
        raise ValueError("path must be root/yyyyMMddHH-SEQUS/unit/file.log")
    sequs_datetime = _sequs_datetime_from_dir(rel.parts[0])
    unit = rel.parts[1]
    return sequs_datetime, unit


def _recipe_matches(actual: str, wanted: str, case_sensitive: bool) -> bool:
    actual_key = _recipe_key(actual)
    wanted_key = _recipe_key(wanted)
    if case_sensitive:
        return actual_key == wanted_key
    return actual_key.lower() == wanted_key.lower()


def _iter_logs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".log"
    )


def _find_matching_logs(root: Path, recipe: str, case_sensitive: bool,
                        skip_invalid: bool) -> tuple[list[MatchedLog], list[str]]:
    matches: list[MatchedLog] = []
    errors: list[str] = []
    for path in _iter_logs(root):
        try:
            sequs_datetime, unit = _log_metadata(path, root)
            actual_recipe = _read_recipe(path)
        except Exception as exc:
            msg = f"{path}: {exc}"
            errors.append(msg)
            if not skip_invalid:
                continue
            continue
        if _recipe_matches(actual_recipe, recipe, case_sensitive):
            actual_recipe_key = _recipe_key(actual_recipe)
            matches.append(MatchedLog(
                src=path,
                sequs_datetime=sequs_datetime,
                unit=unit,
                recipe=actual_recipe,
                recipe_key=actual_recipe_key,
            ))
    return matches, errors


def _destination(item: MatchedLog, equipment: str, tgt_root: Path) -> Path:
    equipment_safe = _safe_component(equipment)
    datetime_safe = _safe_component(item.sequs_datetime)
    recipe_safe = _safe_component(item.recipe_key)
    filename_safe = _safe_component(item.src.name)
    return tgt_root / f"{equipment_safe}_{datetime_safe}_{recipe_safe}{filename_safe}"


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


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = [
        "status", "src", "dst", "equipment", "datetime", "unit",
        "recipe", "recipe_key", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
            "Copy or move logs matching one recipe from "
            "root/yyyyMMddHH-SEQUS/unit folders into one target folder."
        )
    )
    ap.add_argument("src", help="source root directory")
    ap.add_argument("tgt", help="target directory")
    ap.add_argument("--equipment", required=True, help="equipment name used in output filenames")
    ap.add_argument("--recipe", required=True,
                    help="recipe name/key. Quotes are stripped and the first three characters are matched")
    ap.add_argument("--move", action="store_true", help="move files instead of copying them")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing target files")
    ap.add_argument("--case-sensitive", action="store_true", help="match recipe with case sensitivity")
    ap.add_argument("--skip-invalid", action="store_true",
                    help="skip invalid files instead of returning failure")
    ap.add_argument("--summary-csv", default="copy_recipe_logs_summary.csv",
                    help="summary CSV filename/path. Relative paths are placed under target directory")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    tgt_root = Path(args.tgt).resolve()
    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = tgt_root / summary_csv

    if not src_root.exists() or not src_root.is_dir():
        print(f"[error] source directory not found: {src_root}", file=sys.stderr)
        return 1
    if src_root == tgt_root or src_root in tgt_root.parents:
        print("[error] target directory must not be inside source directory",
              file=sys.stderr)
        return 1

    matches, scan_errors = _find_matching_logs(
        src_root, args.recipe, bool(args.case_sensitive), bool(args.skip_invalid))
    if scan_errors and not args.skip_invalid:
        _print_errors(scan_errors)
        return 1
    if not matches:
        print(f"[error] no logs matched recipe {args.recipe!r}: {src_root}",
              file=sys.stderr)
        _print_errors(scan_errors)
        return 1

    action = "move" if args.move else "copy"
    print(f"source: {src_root}", flush=True)
    print(f"target: {tgt_root}", flush=True)
    print(f"equipment: {args.equipment}", flush=True)
    print(f"recipe: {args.recipe} (match key: {_recipe_key(args.recipe)})", flush=True)
    print(f"matched files: {len(matches)} | action: {action}", flush=True)
    if scan_errors:
        print(f"scan errors skipped: {len(scan_errors)}", flush=True)

    rows: list[dict] = []
    ok = 0
    copy_errors: list[str] = []
    progress = _make_progress(len(matches), action)
    try:
        for item in matches:
            dst = _destination(item, args.equipment, tgt_root)
            row = {
                "status": "ok",
                "src": str(item.src),
                "dst": str(dst),
                "equipment": args.equipment,
                "datetime": item.sequs_datetime,
                "unit": item.unit,
                "recipe": item.recipe,
                "recipe_key": item.recipe_key,
                "error": "",
            }
            try:
                _copy_or_move(item.src, dst, bool(args.move), bool(args.overwrite))
                ok += 1
                _progress_write(progress, f"[ok] {item.src.name} -> {dst.name}")
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                msg = f"{item.src}: {exc}"
                copy_errors.append(msg)
                _progress_write(progress, f"[fail] {msg}", file=sys.stderr)
            finally:
                rows.append(row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    _write_summary(summary_csv, rows)
    print(f"\ncompleted: {ok}/{len(matches)}")
    print(f"summary: {summary_csv}")
    if copy_errors:
        _print_errors(copy_errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
