"""Run tcn-explain once for each lowest log folder under a root directory.

The script expects a directory tree such as:

    root/controller/unit/yyyymmdd/*.log

Every lowest directory that contains .log files is treated as one blind
tcn-explain input set. The original files are not modified. A temporary
dataset is made with symlinks by default, and each run writes a report under
the matching relative path below --report-root.
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


CONTROLLED_RUN_OPTIONS = {
    "--data-name",
    "--file-format",
    "--report-dir",
    "--report-name",
}


@dataclass(frozen=True)
class LeafRun:
    leaf_dir: Path
    rel_dir: Path
    logs: tuple[Path, ...]
    data_dir: Path
    report_dir: Path


def _safe_rel_path(path: Path) -> Path:
    safe_parts = []
    for part in path.parts:
        text = str(part).strip().strip(".")
        if not text:
            text = "_"
        for ch in '<>:"/\\|?*':
            text = text.replace(ch, "_")
        safe_parts.append(text)
    return Path(*safe_parts) if safe_parts else Path("_root")


def _log_dirs(root: Path) -> list[Path]:
    dirs = {p.parent for p in root.rglob("*.log") if p.is_file()}
    if not dirs:
        return []
    leaf_dirs = []
    for d in dirs:
        if not any(other != d and d in other.parents for other in dirs):
            leaf_dirs.append(d)
    return sorted(leaf_dirs, key=lambda p: p.relative_to(root).as_posix())


def _make_progress(total: int, action: str):
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=action, unit="folder", file=sys.stdout)


def _progress_write(progress, message: str, file=None) -> None:
    if progress is None:
        print(message, file=file or sys.stdout)
        return
    progress.write(message, file=file)


def _prepare_data_dir(run: LeafRun, copy_data: bool) -> None:
    if run.data_dir.exists():
        shutil.rmtree(run.data_dir)
    run.data_dir.mkdir(parents=True, exist_ok=True)
    for src in run.logs:
        dst = run.data_dir / src.name
        if copy_data:
            shutil.copy2(src, dst)
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _run_command(cmd: list[str], cwd: Path, dry_run: bool) -> tuple[int, float, str]:
    start = time.perf_counter()
    if dry_run:
        return 0, 0.0, ""
    proc = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.perf_counter() - start
    return int(proc.returncode), elapsed, ""


def _summary_row(run: LeafRun, idx: int, total: int, status: str,
                 returncode: int | str, elapsed: float | str,
                 cmd: list[str], error: str = "") -> dict:
    return {
        "index": idx,
        "total": total,
        "status": status,
        "returncode": returncode,
        "elapsed_sec": f"{elapsed:.3f}" if isinstance(elapsed, float) else elapsed,
        "relative_dir": run.rel_dir.as_posix(),
        "log_count": len(run.logs),
        "data_dir": str(run.data_dir),
        "report_dir": str(run.report_dir),
        "command": shlex.join(cmd),
        "error": error,
    }


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "index",
        "total",
        "status",
        "returncode",
        "elapsed_sec",
        "relative_dir",
        "log_count",
        "data_dir",
        "report_dir",
        "command",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _strip_separator(run_args: list[str]) -> list[str]:
    if run_args and run_args[0] == "--":
        return run_args[1:]
    return run_args


def _split_script_and_run_args(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1:]


def _warn_controlled_options(run_args: list[str]) -> None:
    seen = [opt for opt in CONTROLLED_RUN_OPTIONS if opt in run_args]
    if seen:
        print(
            "[warn] batch script overrides tcn-explain options: "
            + ", ".join(sorted(seen)),
            file=sys.stderr,
        )


def main() -> int:
    script_argv, forwarded_args = _split_script_and_run_args(sys.argv[1:])
    ap = argparse.ArgumentParser(
        description=(
            "Run run.py tcn-explain for every lowest folder containing .log files."
        )
    )
    ap.add_argument("root", help="organized log root directory")
    ap.add_argument("--report-root", default="batch_tcn_explain_reports",
                    help="root directory for per-folder tcn-explain reports")
    ap.add_argument("--work-root", default=".batch_tcn_explain_work",
                    help="temporary dataset root; symlinks are created here")
    ap.add_argument("--run-py", default=None,
                    help="path to run.py. Default: run.py next to this script")
    ap.add_argument("--python", default=sys.executable,
                    help="Python executable used to run run.py")
    ap.add_argument("--copy-data", action="store_true",
                    help="copy logs into the temporary dataset instead of symlinking")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failed tcn-explain run")
    ap.add_argument("--dry-run", action="store_true",
                    help="print and summarize commands without running tcn-explain")
    epilog = (
        "Arguments after '--' are passed to run.py tcn-explain. Example: "
        "python batch_tcn_explain_logs.py ROOT --report-root reports -- "
        "--h-cart-mode include --artifacts-dir artifacts"
    )
    ap.epilog = epilog
    args = ap.parse_args(script_argv)

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve()
    report_root = Path(args.report_root).resolve()
    work_root = Path(args.work_root).resolve()
    run_py = Path(args.run_py).resolve() if args.run_py else script_dir / "run.py"
    run_args = _strip_separator(list(forwarded_args))

    if not root.exists() or not root.is_dir():
        print(f"[error] root directory not found: {root}", file=sys.stderr)
        return 1
    if not run_py.exists():
        print(f"[error] run.py not found: {run_py}", file=sys.stderr)
        return 1
    if work_root == root or root in work_root.parents:
        print("[error] --work-root must not be inside the input root", file=sys.stderr)
        return 1
    _warn_controlled_options(run_args)

    dirs = _log_dirs(root)
    if not dirs:
        print(f"[error] no lowest .log folders found: {root}", file=sys.stderr)
        return 1

    runs: list[LeafRun] = []
    for leaf in dirs:
        rel = leaf.relative_to(root)
        safe_rel = _safe_rel_path(rel)
        logs = tuple(sorted(p for p in leaf.iterdir()
                            if p.is_file() and p.suffix.lower() == ".log"))
        if not logs:
            continue
        runs.append(LeafRun(
            leaf_dir=leaf,
            rel_dir=rel,
            logs=logs,
            data_dir=work_root / safe_rel,
            report_dir=report_root / safe_rel,
        ))

    if not runs:
        print(f"[error] no .log files found in leaf folders: {root}", file=sys.stderr)
        return 1

    print(f"root: {root}", flush=True)
    print(f"report_root: {report_root}", flush=True)
    print(f"work_root: {work_root}", flush=True)
    print(f"folders: {len(runs)} | dry_run={bool(args.dry_run)}", flush=True)

    rows: list[dict] = []
    progress = _make_progress(len(runs), "tcn-explain")
    failures = 0
    try:
        for idx, run in enumerate(runs, start=1):
            cmd = [
                str(args.python),
                str(run_py),
                "tcn-explain",
                *run_args,
                "--file-format",
                "log",
                "--data-name",
                str(run.data_dir),
                "--report-dir",
                str(run.report_dir),
            ]
            try:
                if args.dry_run:
                    print(shlex.join(cmd))
                else:
                    _prepare_data_dir(run, bool(args.copy_data))
                returncode, elapsed, error = _run_command(cmd, script_dir, bool(args.dry_run))
                status = "ok" if returncode == 0 else "failed"
                if returncode != 0:
                    failures += 1
                rows.append(_summary_row(
                    run, idx, len(runs), status, returncode, elapsed, cmd, error))
                _progress_write(
                    progress,
                    f"[{status}] {run.rel_dir.as_posix()} logs={len(run.logs)} "
                    f"returncode={returncode}",
                    file=sys.stderr if returncode else None,
                )
                if returncode != 0 and not args.keep_going:
                    break
            except Exception as exc:
                failures += 1
                rows.append(_summary_row(
                    run, idx, len(runs), "failed", "", "", cmd, str(exc)))
                _progress_write(progress, f"[failed] {run.rel_dir}: {exc}",
                                file=sys.stderr)
                if not args.keep_going:
                    break
            finally:
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    summary_path = report_root / "batch_tcn_explain_summary.csv"
    _write_summary(summary_path, rows)
    print(f"summary: {summary_path}")
    if failures:
        print(f"failed folders: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
