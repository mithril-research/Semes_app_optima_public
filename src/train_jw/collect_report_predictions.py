"""Collect name-ordered prediction PNGs from nested tcn-explain reports.

Expected source layout:

    report_root/controller/unit/date/.../07_예측값_이름순.png

Output filename:

    controller_unit_date_예측값_이름순.png

This utility is independent from run.py and is intended to be convenient on
Windows paths as well as Linux paths.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


SOURCE_FILENAME = "07_예측값_이름순.png"
OUTPUT_SUFFIX = "예측값_이름순.png"
INVALID_FILENAME_CHARS = '<>:"/\\|?*'


@dataclass(frozen=True)
class PredictionPng:
    src: Path
    controller: str
    unit: str
    date: str

    @property
    def dst_name(self) -> str:
        return f"{self.controller}_{self.unit}_{self.date}_{OUTPUT_SUFFIX}"


def _safe_filename_component(value: object) -> str:
    text = str(value).strip()
    trans = {ord(ch): "_" for ch in INVALID_FILENAME_CHARS}
    text = text.translate(trans)
    text = "".join("_" if ord(ch) < 32 else ch for ch in text)
    text = text.strip(" .")
    return text or "_"


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


def _find_prediction_pngs(report_root: Path) -> list[PredictionPng]:
    items: list[PredictionPng] = []
    for src in sorted(report_root.rglob(SOURCE_FILENAME)):
        if not src.is_file():
            continue
        try:
            rel = src.relative_to(report_root)
        except ValueError:
            continue
        if len(rel.parts) < 4:
            continue
        controller, unit, date = rel.parts[:3]
        items.append(
            PredictionPng(
                src=src,
                controller=_safe_filename_component(controller),
                unit=_safe_filename_component(unit),
                date=_safe_filename_component(date),
            )
        )
    return items


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "status",
        "src",
        "dst",
        "controller",
        "unit",
        "date",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_one(item: PredictionPng, tgt_dir: Path, overwrite: bool,
              dry_run: bool) -> dict:
    dst = tgt_dir / item.dst_name
    row = {
        "status": "copied",
        "src": str(item.src),
        "dst": str(dst),
        "controller": item.controller,
        "unit": item.unit,
        "date": item.date,
        "error": "",
    }
    if dst.exists() and not overwrite:
        row["status"] = "error"
        row["error"] = "target exists; use --overwrite to replace it"
        return row
    if dry_run:
        row["status"] = "dry_run"
        return row
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.src, dst)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            f"Copy {SOURCE_FILENAME} files from report_root/controller/unit/date "
            "folders into one target folder."
        )
    )
    ap.add_argument("--report-root", required=True,
                    help="root directory containing controller/unit/date reports")
    ap.add_argument("--tgt-dir", required=True,
                    help="target directory where copied PNG files are stored")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing target PNG with the same output name")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be copied without writing files")
    ap.add_argument("--summary-csv", default="collect_report_predictions_summary.csv",
                    help="summary CSV filename/path. Relative paths are placed under --tgt-dir")
    args = ap.parse_args()

    report_root = Path(args.report_root).resolve()
    tgt_dir = Path(args.tgt_dir).resolve()
    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = tgt_dir / summary_csv

    if not report_root.exists() or not report_root.is_dir():
        print(f"[error] report root directory not found: {report_root}", file=sys.stderr)
        return 1
    if tgt_dir == report_root or report_root in tgt_dir.parents:
        print("[error] --tgt-dir must not be inside --report-root", file=sys.stderr)
        return 1

    items = _find_prediction_pngs(report_root)
    if not items:
        print(f"[error] no {SOURCE_FILENAME} files found under: {report_root}", file=sys.stderr)
        return 1

    progress = _make_progress(len(items), "copy predictions")
    rows: list[dict] = []
    for item in items:
        row = _copy_one(item, tgt_dir, args.overwrite, args.dry_run)
        rows.append(row)
        if row["status"] == "error":
            _progress_write(progress, f"[error] {row['src']} -> {row['error']}", file=sys.stderr)
        elif args.dry_run:
            _progress_write(progress, f"[dry-run] {row['src']} -> {row['dst']}")
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()

    _write_summary(summary_csv, rows)
    copied = sum(1 for row in rows if row["status"] == "copied")
    dry = sum(1 for row in rows if row["status"] == "dry_run")
    errors = [row for row in rows if row["status"] == "error"]
    print(f"found={len(items)} copied={copied} dry_run={dry} errors={len(errors)}")
    print(f"summary: {summary_csv}")
    if errors:
        print("copy failed because one or more target filenames already exist.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
