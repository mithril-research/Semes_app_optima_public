"""
이상(포화/죽음) 센서가 있는 CSV/LOG(wafer) 제외.
===============================================================================

제외(삭제) 기준 — 센서 컬럼 중 하나라도 아래에 해당하면 그 wafer 제외:
  (A) 값이 --high(기본 420) '이상'인 상태가 --min-run(기본 1500) 포인트 이상 연속
      → 포화/오버레인지/고착.
  (B) 값이 0(|값|<=--zero-tol)인 상태가 --min-run 포인트 이상 연속
      → 센서 꺼짐/죽음.

  - 검사 대상 = log2csv.py의 NO_HTP, IO_ORDER, 숫자명 IO 컬럼, 메타 컬럼을 제외한
    수치 센서 컬럼.
  - NaN 은 연속을 끊는다(판정 제외).

전역 죽은 채널 보호:
  거의 모든 파일(--global-frac, 기본 0.9 이상)에서 같은 센서가 걸리면, 그건 특정
  wafer 불량이 아니라 '채널 자체'(예: 늘 0인 미사용 센서) 문제 → wafer 삭제 대상에서
  자동 제외하고 안내. (강제 포함은 --no-global-skip)

안전:
  기본은 미리보기(아무것도 안 함). --apply 는 격리 폴더로 '이동'(원본 보존),
  --apply --delete 는 영구 삭제.

사용 (semes 폴더에서):
  python clean_stuck_sensors.py dataset --recursive                  # CSV/LOG 미리보기
  python clean_stuck_sensors.py dataset --recursive --apply          # 격리 이동
  python clean_stuck_sensors.py dataset --recursive --apply --delete # 삭제
  python clean_stuck_sensors.py dataset/train/OK --high 420 --min-run 1500
  python clean_stuck_sensors.py dataset/train/OK/sample.log --min-run 1500
  python clean_stuck_sensors.py dataset --recursive --ignore-cols LP2
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from log2csv import IO_ORDER, NO_HTP

# 검사에서 제외할 비센서 컬럼
_SKIP_NAMES = {"process step", "process_step", "processstep", "step", "phase",
               "date", "time", "datetime", "timestamp", "index"}
_SUPPORTED_SUFFIXES = {".csv", ".log"}
_EXCLUDE_SENSOR_NAMES = {str(x).strip() for x in IO_ORDER + NO_HTP}


def _is_skip_col(col) -> bool:
    name = str(col).strip()
    c = name.lower()
    if c.startswith("unnamed"):
        return True
    if c in _SKIP_NAMES:
        return True
    if name.isdigit():   # IO: 순수 숫자 이름
        return True
    if name in _EXCLUDE_SENSOR_NAMES:
        return True
    return False


def _max_true_run(mask) -> int:
    """boolean 배열에서 True 가 연속된 최장 구간 길이."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return 0
    z = m.astype(np.int8)
    padded = np.concatenate(([0], z, [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"지원하지 않는 파일 형식: {path.name}")
    read_kwargs = {"sep": "\t"} if suffix == ".log" else {}
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, **read_kwargs)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(path, **read_kwargs)
    if suffix == ".log":
        step_col = next(
            (c for c in df.columns
             if str(c).strip().lower().replace(" ", "").replace("_", "") == "processstep"),
            None,
        )
        if step_col is None:
            raise ValueError("LOG 필수 컬럼 누락: ProcessStep")
        df = df[pd.to_numeric(df[step_col], errors="coerce").notna()].copy()
    return df


def inspect_file(path: Path, min_run: int, high: float, zero_tol: float) -> list:
    """제외 사유 [(col, kind, run)] 반환 (비면 정상). kind='high'|'zero'."""
    df = _read_table(path)
    flagged = []
    for col in df.columns:
        if _is_skip_col(col):
            continue
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(v).all():
            continue
        with np.errstate(invalid="ignore"):
            m_high = v >= high            # NaN >= high → False
            m_zero = np.abs(v) <= zero_tol
        r_high = _max_true_run(m_high)
        r_zero = _max_true_run(m_zero)
        if r_high >= min_run:
            flagged.append((str(col), "high", int(r_high)))
        if r_zero >= min_run:
            flagged.append((str(col), "zero", int(r_zero)))
    return flagged


def main() -> None:
    ap = argparse.ArgumentParser(
        description="포화(≥high)/죽음(0) 상태가 오래 지속된 센서가 있는 wafer 제외")
    ap.add_argument("input", help="CSV/LOG 파일 또는 폴더")
    ap.add_argument("--min-run", type=int, default=1500,
                    help="이 포인트 이상 '연속'이면 제외 (기본 1500)")
    ap.add_argument("--high", type=float, default=420.0,
                    help="이 값 '이상'이 지속되면 포화로 판정 (기본 420)")
    ap.add_argument("--zero-tol", type=float, default=0.0,
                    help="|값|<=zero-tol 을 0 으로 간주 (기본 0=정확히 0)")
    ap.add_argument("--recursive", "-r", action="store_true",
                    help="폴더 입력 시 하위 폴더의 *.csv, *.log 까지 검사")
    ap.add_argument("--apply", action="store_true",
                    help="실제 처리 (미지정 시 미리보기만)")
    ap.add_argument("--delete", action="store_true",
                    help="--apply 와 함께: 격리 이동이 아니라 영구 삭제")
    ap.add_argument("--quarantine", default=None,
                    help="격리 폴더 (기본 <입력>_excluded). --delete 면 무시")
    ap.add_argument("--ignore-cols", default="",
                    help="검사에서 제외할 센서(콤마 구분). 예: LP2,E1")
    ap.add_argument("--global-frac", type=float, default=0.9,
                    help="이 비율 이상 파일에서 걸리는 센서는 '전역 채널 문제'로 보고 "
                         "wafer 삭제 대상에서 제외 (기본 0.9)")
    ap.add_argument("--no-global-skip", action="store_true",
                    help="전역 채널 문제도 wafer 삭제 판정에 포함(주의: 대량 삭제 가능)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[에러] 경로 없음: {inp}", file=sys.stderr); sys.exit(1)
    if inp.is_file():
        files = [inp]; base = inp.parent
    else:
        iterator = inp.rglob("*") if args.recursive else inp.glob("*")
        files = sorted(
            f for f in iterator
            if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        base = inp
    if not files:
        print(f"[에러] CSV/LOG 없음: {inp}"
              f"{' (하위폴더 포함하려면 --recursive)' if inp.is_dir() else ''}",
              file=sys.stderr)
        sys.exit(1)

    qdir = (Path(args.quarantine) if args.quarantine
            else base.parent / (base.name + "_excluded"))
    mode = ("미리보기" if not args.apply
            else ("영구 삭제" if args.delete else f"격리 이동 → {qdir}"))
    print(f"입력: {inp} | 파일 {len(files)}개 | 기준: (값≥{args.high}) 또는 "
          f"(값=0) 가 {args.min_run}포인트 이상 연속 | 동작: {mode}\n")

    # ── Pass 1: 전 파일 검사 + 센서별 빈도 집계 ──
    results = {}
    col_count = {}
    for f in files:
        try:
            flagged = inspect_file(f, args.min_run, args.high, args.zero_tol)
        except Exception as e:
            print(f"  ? {f.name}: 읽기 실패 {e}")
            continue
        results[f] = flagged
        for c, kind, r in flagged:
            col_count[c] = col_count.get(c, 0) + 1

    n_scanned = len(results) or 1
    ignore = {x.strip() for x in args.ignore_cols.split(",") if x.strip()}
    global_bad = {c for c, cnt in col_count.items()
                  if cnt / n_scanned >= args.global_frac}
    skip_cols = set(ignore)
    if not args.no_global_skip:
        skip_cols |= global_bad

    if global_bad:
        print("⚠ 전역 채널 문제(거의 모든 파일에서 걸림) — 특정 wafer 불량이 아니라 "
              "채널 자체(미사용/상시포화 등)일 가능성:")
        for c in sorted(global_bad):
            print(f"    {c}: {col_count[c]}/{n_scanned} 파일")
        if not args.no_global_skip:
            print("  → wafer 삭제 대상에서 제외. 이런 컬럼은 select_sensors.py 로 빼는 게 "
                  "맞습니다. (강제 포함: --no-global-skip)")
        print()

    _KIND = {"high": f"≥{args.high}", "zero": "=0"}
    bad_files = []
    for f, flagged in results.items():
        real = [(c, k, r) for c, k, r in flagged if c not in skip_cols]
        if not real:
            continue
        bad_files.append(f)
        _rel = (f.relative_to(base) if (base == f.parent or base in f.parents)
                else f.name)
        _desc = ", ".join(f"{c}({_KIND.get(k, k)} 연속{r})" for c, k, r in real[:5])
        print(f"  ✗ {_rel}  →  {_desc}{' …' if len(real) > 5 else ''}")

    print(f"\n제외 대상 wafer: {len(bad_files)}/{len(files)}"
          f"{f'  (전역 채널 {len(global_bad)}개 제외)' if (global_bad and not args.no_global_skip) else ''}")
    if not args.apply:
        print("→ 미리보기였습니다. 제거하려면 --apply (격리) 또는 --apply --delete (삭제).")
        return
    if not bad_files:
        print("제거할 파일 없음."); return

    done = 0
    for f in bad_files:
        try:
            if args.delete:
                f.unlink()
            else:
                try:
                    rel = f.relative_to(base)
                except ValueError:
                    rel = Path(f.name)
                dst = qdir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dst))
            done += 1
        except Exception as e:
            print(f"  ! {f.name} 처리 실패: {e}")
    print(f"{'삭제' if args.delete else '격리 이동'} 완료: {done}/{len(bad_files)}")


if __name__ == "__main__":
    main()
