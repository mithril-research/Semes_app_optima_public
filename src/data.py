"""Data loading and schema inference for IO-Recon-TCN."""
from __future__ import annotations


import os
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .sensor_schema import in_group, paired_zone
from .auxiliary_sensors import (add_auxiliary_sensors as _add_auxiliary_sensors,
                                auxiliary_sensor_columns, AuxiliarySensorError)

from .config import IOConfig


def step_timing(frame):
    cols = {str(c).strip().lower(): c for c in frame.columns}
    step_col = next((cols[c] for c in ('processstep', 'process_step', 'step') if c in cols), None)
    if 'time' not in cols or step_col is None:
        return {'version': 3, 'steps': {}, 'missing': 'Time 또는 ProcessStep 열 없음'}
    raw = frame[cols['time']]
    numeric = pd.to_numeric(raw, errors='coerce')
    if numeric.notna().any() and not raw.astype(str).str.contains(':', regex=False).any():
        seconds = numeric.to_numpy(dtype=float)
    else:
        seconds = pd.to_timedelta(raw.astype(str).str.strip(), errors='coerce').dt.total_seconds().to_numpy(copy=True)
        # Clock-only Time can cross midnight; smaller backwards jumps are invalid.
        offset = 0.0
        previous = None
        for i, value in enumerate(seconds.copy()):
            if np.isfinite(value):
                if previous is not None and previous - value > 43200:
                    offset += 86400
                previous = value
                seconds[i] = value + offset
    steps = pd.to_numeric(frame[step_col], errors='coerce').to_numpy()
    from .trend_sensors import boundary_zero_indices
    zeros = boundary_zero_indices(steps)
    zero_starts = {int(indices[0]): key for key, indices in zeros.items()}
    result = {}
    i = 0
    while i < len(steps):
        j = i + 1
        while j < len(steps) and steps[j] == steps[i]:
            j += 1
        st = steps[i]
        if i in zero_starts or np.isfinite(st) and st > 0 and st == int(st):
            key = zero_starts.get(i, str(int(st)))
            # The next step's first timestamp closes this step, including single-row steps.
            end = j if j < len(seconds) and np.isfinite(seconds[j]) else j - 1
            run = seconds[i:end + 1]
            valid = end > i and np.isfinite(run).all() and (np.diff(run) >= 0).all()
            duration = float(seconds[end] - seconds[i]) if valid else None
            reason = None if valid else ('종료 경계 없음(마지막 한 줄)' if end == i else 'Time 누락 또는 역행')
            prev = result.get(key, {'seconds': 0.0, 'visits': 0})
            result[key] = {'seconds': (prev['seconds'] + duration if prev['seconds'] is not None and duration is not None else None),
                           'visits': prev['visits'] + 1, 'missing': prev.get('missing') or reason}
        i = j
    return {'version': 3, 'steps': result, 'basis': '다음 스텝 시작−현재 스텝 시작; 마지막 스텝은 마지막 기록까지', 'missing': None}


def _safe_build(fn, p):
    """병렬 로드 워커 래퍼: 한 파일이 실패해도 None 반환 → 배치 전체 중단 방지(파일만 skip)."""
    try:
        return fn(p)
    except AuxiliarySensorError:
        raise
    except Exception as e:                                # noqa: BLE001
        print(f"[load] 파일 로드 실패 skip: {p} — {e}", flush=True)
        return None


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """CSV/LOG/TSV 리더. .log/.tsv 는 설비 원본(탭 구분)을 '변환 없이' 직접 읽는다.

    변환기가 하던 전처리를 메모리에서 동일 수행:
      · sep='\t'  · ProcessStep 비숫자 행(헤더/구분행) 제거  · 트레일링 탭 더미열(Unnamed) 제거
    → csv 파일 생성/삭제 없이 바로 추론 가능(대량 파일에서 3중 I/O 제거 → '준비 중' 단축).
    """
    is_log = Path(path).suffix.lower() in (".log", ".tsv")
    sep = "\t" if is_log else kwargs.get("sep", ",")
    if is_log:
        kwargs = {**kwargs, "sep": "\t"}

    bad = {"n": 0}                                       # 깨진(필드 초과) 행 수 — UI 경고 표시용

    def _count_over_fields(enc):
        """헤더보다 필드가 많은 행 수를 직접 세어 반환(=C파서가 죽던 라인 = skip 대상).

        필드가 적은 행은 pandas 가 NaN 으로 채워 오류가 안 나므로 세지 않는다.
        ParserError 가 난 파일에서만 호출(드묾) → 재스캔 비용 무시 가능."""
        import csv as _csv
        try:
            with open(path, encoding=enc or "utf-8", errors="replace", newline="") as fh:
                rdr = _csv.reader(fh, delimiter=sep)
                header = next(rdr, None)
                if not header:
                    return 0
                nc = len(header)
                return sum(1 for row in rdr if len(row) > nc)
        except Exception:                               # noqa: BLE001
            return 0

    def _read_tolerant(**opts):
        """C엔진으로 먼저 읽되, 행별 필드 수 불일치(ParserError)면 깨진 행을 건너뛰고 재시도.

        설비 로그가 대량 내보내기 중 간헐적으로 한 줄이 깨지거나(쓰기 경쟁) 두 줄이
        붙어(예: 143열 헤더인데 어떤 행이 255열) 나오면 C 파서는 배치 전체를 죽인다.
        → 그 '깨진 행만' 건너뛰고 나머지는 정상 추론(웨이퍼 수천 장 중 1줄 손실은 무시 가능).
        """
        try:
            return pd.read_csv(path, **opts)
        except pd.errors.ParserError:
            bad["n"] = _count_over_fields(opts.get("encoding"))   # UI 경고용 카운트
            skip = {**opts, "engine": "python"}
            try:                                        # pandas ≥1.3
                return pd.read_csv(path, on_bad_lines="skip", **skip)
            except TypeError:                           # pandas <1.3 (구버전 제어기)
                return pd.read_csv(path, error_bad_lines=False,
                                   warn_bad_lines=False, **skip)

    df = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            df = _read_tolerant(encoding=enc, **kwargs)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        df = _read_tolerant(**kwargs)
    if is_log:
        if "ProcessStep" in df.columns:                  # 숫자 아닌 행(중간 헤더 등) 제거
            df = df[pd.to_numeric(df["ProcessStep"], errors="coerce").notna()]
            df = df.reset_index(drop=True)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    try:
        df.attrs["n_bad_lines"] = int(bad["n"])          # load_dataset 가 rec 로 전달
    except Exception:                                    # noqa: BLE001
        pass
    return df


def _data_files(folder: Path):
    """폴더의 데이터 파일 목록 — 같은 이름(스템)이 중복이면 1개만.

    ★ 추론 입력은 .log/.tsv 원본만(우선순위 .log > .tsv). .csv 는 폴더에 .log/.tsv 가
    '하나도 없을 때만' 허용 — 레거시 수동투입 폴더·아카이브(npz) RCA 임시복원 폴더용.
    (.log 기반 폴더의 .csv 는 변환 잔여물/부산물이라 웨이퍼로 읽지 않는다 — 사용자 확정)"""
    pri = {".log": 0, ".tsv": 1, ".csv": 2}
    best = {}
    has_log = False
    for f in sorted(folder.rglob("*")):
        suf = f.suffix.lower()
        if suf not in pri or not f.is_file():
            continue
        if f.stem.lower() == "metadata":                 # 폴더 메타데이터 파일은 웨이퍼 데이터가 아님
            continue
        if suf != ".csv":
            has_log = True
        cur = best.get(f.stem)
        if cur is None or pri[suf] < pri[cur.suffix.lower()]:
            best[f.stem] = f
    if has_log:                                          # .log 기반 폴더 → .csv 완전 제외
        best = {st: f for st, f in best.items() if f.suffix.lower() != ".csv"}
    return sorted(best.values())


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
    virtual_cols = [c for c in sensor_cols if _is_virtual_sensor_name(str(c), cfg)]
    mode = str(getattr(cfg, "h_cart_mode", "") or "").lower()
    if mode == "exclude":
        h_cols = [c for c in h_cols if not _is_cart_h_sensor_name(str(c), cfg)]
    if exact:
        selected = [c for c in sensor_cols if str(c).strip() in exact]
    elif prefixes:
        selected = [c for c in sensor_cols if str(c).strip().startswith(prefixes)]
    else:
        selected = list(sensor_cols)
    _append_unique(selected, h_cols)
    if _virtual_sensor_mode(cfg) != "off":
        _append_unique(selected, virtual_cols)
    if mode == "exclude":
        selected = [c for c in selected if not _is_cart_h_sensor_name(str(c), cfg)]
    sensor_cols = selected
    if exclude:
        sensor_cols = [c for c in sensor_cols if str(c).strip() not in exclude]
    _append_unique(sensor_cols, auxiliary_sensor_columns(cfg))
    return io_cols, sensor_cols, step_col


_SCHEMA_CACHE: dict = {}                                 # (path,size,mtime) → (io,sensor,step)
_SCHEMA_CACHE_MAX = 20000


def _infer_schema(files: List[Path], cfg: IOConfig, cancel_check=None) -> Tuple[List[str], List[str], Optional[str]]:
    """파일별 스키마 기여를 캐시(경로+크기+mtime 키) — 자동감시가 같은 폴더를 반복
    스캔해도 '안 바뀐 파일'은 다시 읽지 않는다(대량 폴더 스캔 가속)."""
    io_cols, sensor_cols, step_col = [], [], None
    nrows = int(getattr(cfg, "schema_infer_rows", 512) or 512)
    for f in files:
        if cancel_check is not None:
            cancel_check()
        try:
            st = f.stat()
            key = (str(f), st.st_size, st.st_mtime, nrows, repr(cfg), "cylinder-v1")
        except OSError:
            key = None
        hit = _SCHEMA_CACHE.get(key) if key else None
        if hit is None:
            try:
                df = _add_auxiliary_sensors(add_virtual_sensors(_read_csv(f, nrows=nrows), cfg), cfg)
                hit = split_columns(df, cfg)
            except AuxiliarySensorError:
                raise
            except Exception as e:                       # noqa: BLE001 (빈/깨진 파일 → 스키마 기여 없이 skip)
                print(f"[schema] 스키마 추론 skip(파일 손상 의심): {f} — {e}", flush=True)
                continue                                 # 이 파일은 건너뜀(전체 로드가 크래시하지 않게)
            if key:
                if len(_SCHEMA_CACHE) >= _SCHEMA_CACHE_MAX:
                    _SCHEMA_CACHE.clear()                # 단순 상한(재시작 없는 장기 운영 보호)
                _SCHEMA_CACHE[key] = hit
        ios, sensors, step = hit
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


def _parse_cache_dir() -> Optional[Path]:
    """파싱 캐시 폴더(앱이 IO_PARSE_CACHE_DIR 로 지정). 미지정=캐시 끔(CLI 등)."""
    d = os.environ.get("IO_PARSE_CACHE_DIR") or ""
    return Path(d) if d else None


def _parsed_frame(f: Path, cfg: IOConfig) -> pd.DataFrame:
    """_read_csv + add_virtual_sensors 결과 프레임 — npz 캐시(★모델 무관).

    · 키 = 파일 경로 md5 + 원본 mtime/size(파일이 바뀌면 자동 무효) + 가상센서 버전(cfg).
    · 값은 to_numeric 후 float64 그대로 저장 → 원본 파싱과 수치 동일(판정 불변).
    · 모델을 바꿔 재검사해도 유효 — 재검사 시간의 절반가량인 파싱을 통째로 생략한다.
    """
    import hashlib
    cdir = _parse_cache_dir()
    key = vver = None
    if cdir is not None:
        try:
            st = f.stat()
            vver = "v4-cylinder-input:" + hashlib.md5(repr(cfg).encode("utf-8", "ignore")).hexdigest()[:8]
            key = cdir / (f"{f.stem}_"
                          f"{hashlib.md5(str(f.resolve()).encode('utf-8', 'ignore')).hexdigest()[:8]}.npz")
            if key.exists():
                with np.load(key, allow_pickle=False) as z:   # 핸들 닫기(Windows 삭제 잠금 방지)
                    if (int(z["mtime_ns"]) == st.st_mtime_ns and int(z["size"]) == st.st_size
                            and str(z["vver"]) == vver):
                        df = pd.DataFrame(np.asarray(z["vals"], dtype=np.float64),
                                          columns=[str(c) for c in z["cols"]])
                        df.attrs["n_bad_lines"] = int(z["n_bad"])
                        df.attrs["step_timing"] = json.loads(str(z["step_timing"]))
                        return df
        except Exception:                                 # noqa: BLE001 — 캐시 손상 → 원본 파싱
            pass
    raw_df = _read_csv(f)
    n_bad = int(raw_df.attrs.get("n_bad_lines", 0) or 0)
    timing = step_timing(raw_df)
    df = _add_auxiliary_sensors(add_virtual_sensors(raw_df, cfg), cfg)
    df.attrs["step_timing"] = timing
    if key is not None:
        try:                                              # 수치화 후 저장(원자적) — 실패해도 검사엔 무해
            num = df.apply(pd.to_numeric, errors="coerce")
            st = f.stat()
            tmp = key.with_name(key.stem + f".tmp{os.getpid()}.npz")
            np.savez_compressed(tmp, vals=num.values.astype(np.float64),
                                cols=np.asarray([str(c) for c in df.columns]),
                                n_bad=np.int64(n_bad), mtime_ns=np.int64(st.st_mtime_ns),
                                size=np.int64(st.st_size), vver=np.asarray(vver),
                                step_timing=np.asarray(json.dumps(timing)))
            os.replace(tmp, key)
        except Exception:                                 # noqa: BLE001
            pass
    df.attrs["n_bad_lines"] = n_bad
    return df


def load_dataset(folder: Path, cfg: IOConfig, stems=None, cancel_check=None, include_trend_source=False
                 ) -> Tuple[List[dict], List[str], List[str], int]:
    """폴더의 CSV 들을 wafer 단위로 로드.

    Returns:
      records: [{wafer_id, io(T,n_io), sensors(T,n_sensor), sensor_cols, step(T,)|None}]
      io_cols, sensor_cols (전 파일 union schema, 누락은 0 채움)
      n_steps_global (스텝 one-hot 차원)
    """
    folder = _resolve_existing_dir_ci(Path(folder))
    files = _data_files(folder)                          # .csv + .log/.tsv(직접 읽기)
    if stems is not None:                                # ★ 신규 파일만 로드(대량 폴더 스캔 가속)
        want = {str(x) for x in stems}
        files = [f for f in files if f.stem in want]
    if not files:
        return [], [], [], 0
    # 컬럼 스키마는 전체 파일 union 기준. 첫 파일에 없는 H 센서도 입력에 포함한다.
    io_cols, sensor_cols, step_col = _infer_schema(files, cfg, cancel_check=cancel_check)

    # ── 파일별 로드(읽기+피처+수치화)를 함수로 분리 → 스레드풀로 병렬 읽기 ──
    #   대량 폴더에서 검사시간의 ~절반이 CSV/로그 읽기라, I/O 대기 동안 다른 파일을 동시에 읽어
    #   전체를 크게 단축한다. pandas 파싱은 GIL 을 상당부분 놓으므로 스레드로 실효 가속.
    def _build_one(f):
        if cancel_check is not None:
            cancel_check()
        df = _parsed_frame(f, cfg)                        # ★ 파싱 캐시(모델 무관) — 재검사 가속
        n_bad = int(df.attrs.get("n_bad_lines", 0) or 0)   # 깨진 행 수(있으면 UI 경고)
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
        local_max = 0
        if step_col and step_col in df.columns:
            step = pd.to_numeric(df[step_col], errors="coerce").fillna(0).astype(int).values
            local_max = int(step.max()) + 1 if len(step) else 0
        rec = {"wafer_id": f.stem, "recipe": _infer_recipe(f, folder, cfg),
               "source_path": str(f), "n_bad_lines": n_bad,
               "io": io, "io_cols": list(io_cols),
               "sensors": sen, "sensor_cols": list(sensor_cols), "step": step}
        if include_trend_source:
            from .trend_sensors import trend_measurements
            rec["trend_measurements"] = trend_measurements(df)
        rec["step_timing"] = df.attrs.get("step_timing") or step_timing(df)
        return rec, local_max

    records = []
    max_step = 0
    try:
        _w = int(os.environ.get("IO_LOAD_WORKERS") or 0)
    except (TypeError, ValueError):
        _w = 0
    if _w <= 0:
        # 기본: 코어 수(최대 8) — I/O 바운드. ★ 앱의 '코어 예약' 설정을 따른다.
        _w = min(8, int(os.environ.get("OMP_NUM_THREADS") or 0) or (os.cpu_count() or 4))
    _w = max(1, min(_w, len(files)))
    if _w == 1 or len(files) == 1:                        # 파일 1개거나 병렬 끄면 순차
        for f in files:
            try:
                rec, lm = _build_one(f)
                records.append(rec); max_step = max(max_step, lm)
            except AuxiliarySensorError:
                raise
            except Exception as e:                        # noqa: BLE001 (한 파일 실패는 건너뛰고 계속)
                print(f"[load] 파일 로드 실패 skip: {f} — {e}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=_w) as ex:
            for res in ex.map(lambda p: _safe_build(_build_one, p), files):   # 순서 보존
                if res is None:                           # 실패 파일 skip(배치 전체 중단 방지)
                    continue
                rec, lm = res
                records.append(rec); max_step = max(max_step, lm)
    return records, io_cols, sensor_cols, max_step
