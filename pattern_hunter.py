"""패턴 탐사 데몬(pattern_hunter) — 가설 생성·검증 상시 에이전트.

사람이 하던 '패턴 발견'을 자동화한다: 불량 보고(CSV)를 앵커로, 보고 이전
2~3일/1~2달 창에서 지속되는 센서×스텝×특징 패턴을 전수 탐색해 가설을 세우고
(원장 SQLite — 같은 가설은 서명으로 원천 차단), 대조·홀드아웃·반증·선행성으로
검증해 기각/확정한다. LLM(로컬)은 의미부여·보고문만 담당하고 수치 판단은
결정적 엔진이 한다. 골든=답지 철학: 판정은 항상 대조(정상 기간) 상대 비교.

실행:  python pattern_hunter.py --data <로그 폴더;…> --csv <불량내역.csv>
         --out <탐사 작업 폴더> [--model <모델 폴더>] [--once] [--tick 600]
중단 내성: 특징 저장소·원장·앵커가 전부 디스크 — 재시작하면 이어서 진행.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

HUNTER_VER = "h7.2"

# ── 앱 코드 재사용(배포본 src/ · 개발본 io_recon_tcn/ 모두 인식 — main() 과 동일 규칙) ──


def _bootstrap(model_dir=None, cache_hint=None):
    cands = [Path(__file__).resolve().parent, Path.cwd()]
    cands += [p for i, p in enumerate(Path(__file__).resolve().parents) if i < 3]
    root = pkg = None
    for c in cands:
        for p in ("src", "io_recon_tcn"):
            if (c / p / "app" / "server.py").exists():
                root, pkg = c, p
                break
        if root is not None:
            break
    if root is None:
        raise SystemExit("앱 코드 폴더(src/io_recon_tcn)를 찾지 못했습니다 — "
                         "pattern_hunter.py 는 test_avg_waveform.py 와 같은 곳에 두세요")
    app_dir = root / pkg / "app"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(app_dir))
    if pkg != "io_recon_tcn":
        # 배포본은 폴더 이름이 src — io_recon_tcn 이름으로 import 되도록
        # 디스크에 얇은 별칭 패키지를 만든다(앱 main() 과 동일 방식).
        al = root / "io_recon_tcn"
        if not (al / "__init__.py").exists():
            al.mkdir(exist_ok=True)
            (al / "__init__.py").write_text(
                "# 자동 생성(배포본 src/ 를 io_recon_tcn 이름으로 읽기 위한 별칭)\n"
                "import os\n"
                "__path__ = [os.path.join(os.path.dirname(os.path.dirname(\n"
                "    os.path.abspath(__file__))), %r)]\n"
                "from .config import IOConfig\n"
                "__all__ = ['IOConfig']\n"
                "def __getattr__(name):\n"
                "    if name in ('TCNForecastModel', 'TCNForecastDetector'):\n"
                "        from . import model_tcn_forecast as _m\n"
                "        return getattr(_m, name)\n"
                "    raise AttributeError(name)\n" % pkg,
                encoding="utf-8")
            print(f"[setup] 배포 폴더 구조({pkg}/) 인식 — io_recon_tcn 별칭 생성",
                  flush=True)
    import test_avg_waveform as AW                     # noqa: E402
    from io_recon_tcn.app import server as S           # noqa: E402
    S._LOGGER_ON = True
    AW._install_virtual(S)
    AW._install_parse_cache(S, store_hint=cache_hint)
    if model_dir:
        ok, why = S._probe_model_dir(model_dir)
        if not ok:
            raise SystemExit(f"모델 폴더 사용 불가: {why}")
        S.ENGINE.select(model_dir)
    try:
        S.ENGINE.load()
    except Exception as e:                             # noqa: BLE001
        raise SystemExit(f"모델 로드 실패: {e}")
    if not S.ENGINE.ready:
        raise SystemExit(f"모델 로드 실패: {S.ENGINE.error}")
    return AW, S


# ── 특징 라이브러리(웨이퍼×센서×스텝 → 스칼라) — 기존 형상 함수 재사용 ──
F_NAMES = ["mean", "std", "slope", "min", "max", "range",
           "convex", "drop", "arch_area", "arch_n", "wbulge", "median"]


def _feats(AW, seg):
    """스텝 구간 곡선 → 특징 12종(불가한 것은 nan)."""
    out = np.full(len(F_NAMES), np.nan, dtype=np.float32)
    seg = np.asarray(seg, dtype=np.float64)
    m = np.isfinite(seg)
    if m.sum() < 8:
        return out
    y = seg[m]
    out[0] = np.mean(y)
    out[1] = np.std(y)
    x = np.arange(len(y), dtype=np.float64)
    out[2] = np.polyfit(x, y, 1)[0] * len(y)           # 구간 전체 기울기(원단위)
    out[3] = np.min(y)
    out[4] = np.max(y)
    out[5] = out[4] - out[3]
    try:
        r = AW._shape_convexity(y)
        if r:
            out[6], out[7] = r["ratio"], r["drop"]
    except Exception:                                  # noqa: BLE001
        pass
    try:
        a = AW._shape_arches(y, polarity=0)
        if a:
            out[8], out[9] = a["total_area"], float(a["n_arch"])
    except Exception:                                  # noqa: BLE001
        pass
    try:
        w = AW._shape_wbulge(y)
        if w:
            out[10] = w["bulge"]
    except Exception:                                  # noqa: BLE001
        pass
    out[11] = np.median(y)
    return out


def _feats_batch(AW, segs, device):
    """같은 길이 세그먼트 묶음 → (B, F) 특징 — torch 일괄(GPU 위주 전략 B).

    수식은 _feats(numpy)와 동일하게 재현: 기본 6종+중앙값·볼록비(현 중앙값
    끝점)·W면적(스무딩 argmin→선분 위 면적). 극값 기반 아치 2종만 세그먼트별
    CPU(_shape_arches) — GPU 부적합. NaN 포함 세그먼트는 호출부에서 CPU 폴백.
    """
    import torch
    import torch.nn.functional as TF
    B = len(segs)
    n = len(segs[0])
    out = np.full((B, len(F_NAMES)), np.nan, dtype=np.float32)
    y = torch.tensor(np.stack(segs), dtype=torch.float64, device=device)
    mean = y.mean(1)
    std = y.std(1, unbiased=False)                     # np.std(ddof=0) 동일
    mn = y.min(1).values
    mx = y.max(1).values
    x = torch.arange(n, dtype=torch.float64, device=device)
    xc = x - x.mean()
    slope = ((y * xc).sum(1) / (xc * xc).sum()) * n    # polyfit 기울기 × n
    med = y.quantile(0.5, dim=1)                       # np.median(선형 보간) 동일
    out[:, 0] = mean.cpu().numpy(); out[:, 1] = std.cpu().numpy()
    out[:, 2] = slope.cpu().numpy(); out[:, 3] = mn.cpu().numpy()
    out[:, 4] = mx.cpu().numpy(); out[:, 5] = (mx - mn).cpu().numpy()
    out[:, 11] = med.cpu().numpy()
    if n >= 12:
        k = min(5, max(1, n // 10))
        y0 = y[:, :k].quantile(0.5, 1)
        y1 = y[:, n - k:].quantile(0.5, 1)
        t = torch.linspace(0, 1, n, dtype=torch.float64, device=device)
        chord = y0[:, None] + (y1 - y0)[:, None] * t
        area = (chord - y).mean(1)
        drop = y0 - y1
        ratio = torch.where(drop.abs() > 1e-9, 2.0 * area / drop.abs(),
                            torch.zeros_like(area))
        out[:, 6] = ratio.cpu().numpy()
        out[:, 7] = drop.cpu().numpy()
        # W면적: 스무딩(np.convolve same 정렬 재현) → argmin → 선분 위 면적
        w = max(3, n // 100)
        ker = torch.ones(1, 1, w, dtype=torch.float64, device=device)
        num = TF.conv1d(y[:, None, :], ker, padding=w - 1)[:, 0, :]
        den = TF.conv1d(torch.ones_like(y)[:, None, :], ker,
                        padding=w - 1)[:, 0, :]
        s0 = (w - 1) // 2
        ys = (num / den)[:, s0:s0 + n]
        imin = ys.argmin(1)
        idxr = imin[:, None] + torch.arange(-2, 3, device=device)
        validw = idxr >= 0                             # np: y[max(0,imin-2):imin+3]
        gath = y.gather(1, idxr.clamp(0, n - 1))
        y0w = torch.where(validw, gath,
                          torch.full_like(gath, float("nan"))).nanquantile(0.5, 1)
        kw = ((n - imin) // 10).clamp(min=1, max=5)
        pos = torch.arange(n, device=device)[None, :]
        tailm = pos >= (n - kw)[:, None]
        ytail = torch.where(tailm, y, torch.full_like(y, float("nan")))
        y1w = ytail.nanquantile(0.5, 1)
        rise = y1w - y0w
        Lw = (n - imin).to(torch.float64)
        tw = (pos.to(torch.float64) - imin[:, None].to(torch.float64))             / (Lw[:, None] - 1).clamp(min=1)
        chw = y0w[:, None] + rise[:, None] * tw
        maskw = pos >= imin[:, None]
        bulge = ((y - chw).clamp(min=0) * maskw).sum(1) / maskw.sum(1)
        ok_w = (imin < n - 8).cpu().numpy()
        bw = bulge.cpu().numpy()
        out[ok_w, 10] = bw[ok_w]
    # 아치 2종(극값 루프)만 CPU
    for i9, seg9 in enumerate(segs):
        try:
            a9 = AW._shape_arches(seg9, polarity=0)
            if a9:
                out[i9, 8] = a9["total_area"]
                out[i9, 9] = float(a9["n_arch"])
        except Exception:                              # noqa: BLE001
            pass
    return out


def _resolve_device(dev):
    """--device auto/cpu/cuda → torch 디바이스 또는 None(numpy 기존 경로)."""
    if dev in (None, "", "numpy", "off"):
        return None
    try:
        import torch
    except Exception:                                  # noqa: BLE001
        print("[hunter] torch 없음 — --device 무시하고 CPU(numpy) 경로",
              flush=True)
        return None
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else None
    if dev == "cuda" and not torch.cuda.is_available():
        print("[hunter] CUDA 불가 — CPU(torch) 배치로 폴백", flush=True)
        return "cpu"
    return dev


# ── 특징 저장소: <out>/features/<day>.npz (웨이퍼층 — 디스크가 곧 장기 컨텍스트) ──


class FeatStore:
    def __init__(self, out):
        self.dir = Path(out) / "features"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}

    def day_fp(self, day):
        return self.dir / f"{day}.npz"

    def has(self, day, sig):
        fp = self.day_fp(day)
        if not fp.is_file():
            return False
        try:
            with np.load(fp, allow_pickle=True) as z:
                return str(z["sig"]) == sig
        except Exception:                              # noqa: BLE001
            return False

    def save(self, day, sig, ids, srcs, metas, sensors, steps, vals):
        tmp = self.day_fp(day).with_suffix(f".tmp{os.getpid()}.npz")
        np.savez_compressed(
            tmp, sig=np.asarray(sig), ids=np.asarray(ids), srcs=np.asarray(srcs),
            eqp=np.asarray([m["eqp"] for m in metas]),
            dsv=np.asarray([m["dsv"] for m in metas]),
            recipe=np.asarray([m["recipe"] for m in metas]),
            sensors=np.asarray(sensors), steps=np.asarray(steps, dtype=np.int32),
            vals=vals.astype(np.float32))
        os.replace(tmp, self.day_fp(day))
        self._cache.pop(day, None)

    def load(self, day):
        if day in self._cache:
            return self._cache[day]
        fp = self.day_fp(day)
        if not fp.is_file():
            return None
        with np.load(fp, allow_pickle=True) as z:
            d = {k: z[k] for k in ("ids", "srcs", "eqp", "dsv", "recipe",
                                   "sensors", "steps", "vals")}
        d["sensors"] = [str(v) for v in d["sensors"]]
        d["steps"] = [int(v) for v in d["steps"]]
        if len(self._cache) > 90:                      # 하루치 수 MB — 90일 LRU
            self._cache.pop(next(iter(self._cache)))
        self._cache[day] = d
        return d

    def days(self):
        return sorted(p.stem for p in self.dir.glob("*.npz"))


def _aggregate_day(AW, S, day, prs, device=None):
    """하루치 웨이퍼 → (ids, srcs, metas, sensors, steps, vals). 직렬·병렬 공용.

    device 지정 시(전략 B): 세그먼트를 길이별 버킷으로 모아 torch 일괄 계산
    (GPU 위주) — NaN 포함 세그먼트만 세그별 CPU(_feats) 폴백. 미지정=기존 경로.
    """
    ids, srcs, metas, rows = [], [], [], []
    sensors, steps = None, None
    pend = {}                                          # 길이 → [(row_i,si,pi,seg)]
    by_f = {}
    for f, st in prs:
        by_f.setdefault(f, []).append(st)
    for f, sts in by_f.items():
        for c0 in range(0, len(sts), 100):
            recs, _a, _b, _c = S._load_aligned(Path(f), stems=sts[c0:c0 + 100])
            for rec in recs:
                cols = [str(c) for c in (rec.get("sensor_cols") or [])]
                stp = np.asarray(rec.get("step"), dtype=float).astype(int)
                arr = np.asarray(rec["sensors"], dtype=np.float64)
                if sensors is None:
                    sensors = cols
                    steps = sorted(set(int(v) for v in stp))
                v = np.full((len(sensors), len(steps), len(F_NAMES)),
                            np.nan, dtype=np.float32)
                ri = len(rows)
                for si, sn in enumerate(sensors):
                    if sn not in cols:
                        continue
                    col = arr[:, cols.index(sn)]
                    for pi, sp in enumerate(steps):
                        m = stp == sp
                        if m.sum() < 8:
                            continue
                        seg = col[m]
                        if device is None or not np.isfinite(seg).all():
                            v[si, pi] = _feats(AW, seg)
                        else:
                            pend.setdefault(len(seg), []).append(
                                (ri, si, pi, seg))
                ids.append(str(rec["wafer_id"]))
                srcs.append(f)
                metas.append(AW._meta_of(S, str(rec["wafer_id"])))
                rows.append(v)
            del recs
    for n9, items in pend.items():                     # 길이 버킷별 일괄
        c0 = 0
        chunk = len(items)
        while c0 < len(items):
            part = items[c0:c0 + chunk]
            try:
                fb = _feats_batch(AW, [it[3] for it in part], device)
                _DEV["demote"] = 0
            except Exception as e9:                    # noqa: BLE001
                # GPU 메모리 부족(LLM 과 공유) 등 — 배치 반토막 재시도, 끝내
                # 안 되면 세그별 CPU 폴백. 연속 실패면 세션 동안 CPU 배치 강등.
                if chunk > 32:
                    chunk //= 2
                    continue
                _DEV["demote"] += 1
                if _DEV["demote"] == 3 and device == "cuda":
                    print("[feat] GPU 연속 실패 — 이번 실행은 torch CPU 배치로 "
                          "강등(판독 결과는 동일)", flush=True)
                print(f"[feat] torch 배치 불가 — CPU 폴백({str(e9)[:60]})",
                      flush=True)
                fb = np.stack([_feats(AW, it[3]) for it in part])
            for (ri, si, pi, _s9), fv in zip(part, fb):
                rows[ri][si, pi] = fv
            c0 += len(part)
    return ids, srcs, metas, sensors, steps, rows


_WORK = {}                                             # 병렬 워커 상태(프로세스별)
CFG = {}                                               # 메인 프로세스 실행 설정
_DEV = {"demote": 0}                                   # GPU 연속 실패 → CPU 배치 강등


def _worker_init(model_dir, out):
    import io as _io
    import contextlib
    try:
        os.nice(10)                                    # 대화형(앱 조작)에 CPU 양보
    except Exception:                                  # noqa: BLE001
        pass
    with contextlib.redirect_stdout(_io.StringIO()):   # 워커 부트 로그 억제
        AW, S = _bootstrap(model_dir, cache_hint=out)
    _WORK["AW"], _WORK["S"], _WORK["out"] = AW, S, out


def _worker_day(args):
    day, prs, sig = args
    AW, S = _WORK["AW"], _WORK["S"]
    ids, srcs, metas, sensors, steps, rows = _aggregate_day(AW, S, day, prs)
    if rows:
        FeatStore(_WORK["out"]).save(day, sig, ids, srcs, metas,
                                     sensors, steps, np.stack(rows))
    return day, len(rows)


def build_features(AW, S, store, folders, journal, workers=1,
                   model_dir=None, out=None):
    """폴더 전수 → 일별 특징 npz(증분 — 이미 있는 날은 건너뜀)."""
    journal(f"[scan] 파일 태깅 중 — 폴더 {len(folders)}곳 (파일이 많으면 수 분)")
    per, n_all, _n, _av = AW.select_stems_multi(S, folders, None, None, None, None)
    day_map = {}
    for fol, selx in per:
        stems = selx
        if stems is None:
            stems = [f.stem for f in Path(fol).iterdir()
                     if f.suffix.lower() in (".csv", ".log", ".tsv")]
        for st in stems:
            try:
                d = str(S._end_ts_and_date(st)[1] or "")
            except Exception:                          # noqa: BLE001
                d = ""
            if d:
                day_map.setdefault(d, []).append((str(fol), st))
    todo = []
    for day in sorted(day_map):
        prs = sorted(day_map[day], key=lambda x: x[1])
        sig = hashlib.md5(("|".join(st for _f, st in prs)).encode()).hexdigest()[:12]
        if not store.has(day, sig):
            todo.append(day)
    n_w = sum(len(v) for v in day_map.values())
    journal(f"[scan] 웨이퍼 {n_w}장 · {len(day_map)}일 — 새로 집계할 날 "
            f"{len(todo)}일 (약 1초/장, 한 번 하면 증분)")
    jobs = []
    for day in todo:
        prs = sorted(day_map[day], key=lambda x: x[1])
        sig = hashlib.md5(("|".join(st for _f, st in prs)).encode()).hexdigest()[:12]
        jobs.append((day, prs, sig))
    new = 0
    device = _resolve_device(CFG.get("device"))
    if device is not None:
        journal(f"[feat] GPU/torch 일괄 집계 — device={device} (워커 1, "
                f"배치가 병렬을 대체 · 아치 2종만 CPU)")
        for k9j, (day, prs, sig) in enumerate(jobs):
            if _DEV["demote"] >= 3 and device == "cuda":
                device = "cpu"
            journal(f"[feat] {day} 집계 시작 ({k9j + 1}/{len(jobs)}일 · "
                    f"{len(prs)}장)")
            ids, srcs, metas, sensors, steps, rows = _aggregate_day(
                AW, S, day, prs, device=device)
            if rows:
                store.save(day, sig, ids, srcs, metas, sensors, steps,
                           np.stack(rows))
                new += 1
                journal(f"[feat] {day} {len(rows)}장 집계 ({new}/{len(jobs)}일)")
        return new, sorted(day_map)
    if workers > 1 and len(jobs) > 1:
        # 일 단위 병렬(결과는 직렬과 동일 — 하루 안은 순서 그대로, 파일은 원자 교체)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as _mp
        journal(f"[feat] 병렬 집계 시작 — 워커 {workers}개(코어당 1일씩)")
        with ProcessPoolExecutor(
                max_workers=workers, mp_context=_mp.get_context("spawn"),
                initializer=_worker_init,
                initargs=(model_dir, str(out))) as ex:
            futs = {ex.submit(_worker_day, j): j[0] for j in jobs}
            # 하루 단위 완료라 첫 로그까지 오래 침묵 → 60초 하트비트
            from concurrent.futures import FIRST_COMPLETED, wait as _fwait
            t0h = time.time()
            pend_f = set(futs)
            while pend_f:
                done_f, pend_f = (lambda r: (r.done, r.not_done))(
                    _fwait(pend_f, timeout=60, return_when=FIRST_COMPLETED))
                for fu in done_f:
                    day, n_r = fu.result()
                    new += 1
                    journal(f"[feat] {day} {n_r}장 집계 ({new}/{len(jobs)}일)")
                if pend_f and not done_f:
                    journal(f"[feat] 진행 중 — 완료 {new}/{len(jobs)}일 · 경과 "
                            f"{int((time.time() - t0h) // 60)}분 (하루가 끝날 "
                            f"때마다 완료 로그가 나옵니다)")
        store._cache.clear()
    else:
        for day, prs, sig in jobs:
            ids, srcs, metas, sensors, steps, rows = _aggregate_day(AW, S, day, prs)
            if rows:
                store.save(day, sig, ids, srcs, metas, sensors, steps,
                           np.stack(rows))
                new += 1
                journal(f"[feat] {day} {len(rows)}장 집계 ({new}/{len(jobs)}일)")
    return new, sorted(day_map)


# ── 앵커: 불량 내역 CSV(설비명,dsv,불량원인,발생일자 — 8자리/하이픈 날짜) ──


def _to_date(v):
    v = str(v or "").strip().replace(".", "-").replace("/", "-")
    if len(v) == 8 and v.isdigit():
        return f"{v[:4]}-{v[4:6]}-{v[6:]}"
    try:
        return dt.date.fromisoformat(v).isoformat()
    except Exception:                                  # noqa: BLE001
        return ""


def read_anchors(csv_path, journal=None):
    """불량 내역 CSV → 앵커. 관용 파서(v154 — 조용한 0건 금지).

    인코딩 utf-8(-sig)/cp949/utf-16 자동, 구분자 콤마/탭/세미콜론, 엑셀
    따옴표 제거, 4열(설비,dsv,원인,날짜) 또는 3열(dsv 생략) 허용. 마지막
    열이 날짜여야 하며, 버린 행은 사유별로 집계해 journal 로 보고한다.
    """
    raw = None
    used_enc = "?"
    data9 = Path(csv_path).read_bytes()
    encs = (["utf-16"] if data9[:2] in (b"\xff\xfe", b"\xfe\xff")
            else []) + ["utf-8-sig", "utf-8", "cp949", "utf-16"]
    for enc in encs:
        try:
            raw = data9.decode(enc)
            used_enc = enc
            break
        except Exception:                              # noqa: BLE001
            continue
    if raw is None:
        raw = data9.decode("cp949", errors="replace")
        used_enc = "cp949(치환)"
    out = []
    skip = {}
    for ln in raw.splitlines():
        ln = ln.replace("\x00", "").strip()
        if not ln:
            continue
        sep = ("\t" if "\t" in ln else (";" if (";" in ln and "," not in ln)
                                          else ","))
        cs = [c.strip().strip('"').strip("'").strip() for c in ln.split(sep)]
        if cs and ("설비" in cs[0] or cs[0].lower() in ("eqp", "equipment")):
            continue                                   # 헤더
        if len(cs) < 3:
            skip["열 부족(3열 미만)"] = skip.get("열 부족(3열 미만)", 0) + 1
            continue
        dt9 = _to_date(cs[-1])
        if not dt9:
            skip["날짜 해석 실패(마지막 열)"] =                 skip.get("날짜 해석 실패(마지막 열)", 0) + 1
            continue
        if len(cs) >= 4:
            eqp9, d9, cz9 = cs[0], cs[1], ",".join(cs[2:-1])
        else:                                          # 3열: dsv 생략
            eqp9, d9, cz9 = cs[0], "", cs[1]
        d9 = d9.strip().lower()
        if d9.isdigit():
            d9 = "dsv" + d9
        out.append({"eqp": eqp9.strip().lower(), "dsv": d9,
                    "cause": cz9.strip(), "date": dt9})
    if journal:
        journal(f"[anchor] CSV 판독 — 인코딩 {used_enc} · 앵커 {len(out)}건"
                + ("".join(f" · 버림 {c9}건({k9})" for k9, c9 in skip.items())
                   if skip else ""))
    return out


# ── 원장(SQLite) — 가설·시행·앵커 전부 영속, 서명 UNIQUE 로 중복 가설 차단 ──


SCHEMA = """
CREATE TABLE IF NOT EXISTS hypothesis(
 id INTEGER PRIMARY KEY, signature TEXT UNIQUE, sensor TEXT, step INTEGER,
 feature TEXT, window_days INTEGER, direction INTEGER,
 anchor_eqp TEXT, anchor_dsv TEXT, anchor_cause TEXT, anchor_date TEXT,
 status TEXT, effect REAL, fdr_q REAL, persist_ratio REAL,
 evidence_json TEXT, reject_reason TEXT, llm_note TEXT,
 created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS trials(
 id INTEGER PRIMARY KEY, hyp_id INTEGER, kind TEXT, stat REAL, p REAL,
 passed INTEGER, detail TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS anchors(
 id INTEGER PRIMARY KEY, eqp TEXT, dsv TEXT, cause TEXT, date TEXT,
 explored INTEGER DEFAULT 0, UNIQUE(eqp,dsv,cause,date));
CREATE TABLE IF NOT EXISTS rounds(
 key TEXT, anchor_id INTEGER, done INTEGER DEFAULT 1, UNIQUE(key,anchor_id));
"""


class Ledger:
    def __init__(self, out):
        self.fp = Path(out) / "ledger.sqlite"
        self.db = sqlite3.connect(str(self.fp))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    @staticmethod
    def sig(sensor, step, feature, w, direction, cause):
        base = f"{sensor}|{step}|{feature}|{w}|{direction}|{cause}".lower()
        return hashlib.md5(base.encode()).hexdigest()

    def propose(self, h):
        now = dt.datetime.now().isoformat(timespec="seconds")
        cur = self.db.execute(
            "INSERT OR IGNORE INTO hypothesis(signature,sensor,step,feature,"
            "window_days,direction,anchor_eqp,anchor_dsv,anchor_cause,anchor_date,"
            "status,effect,fdr_q,persist_ratio,evidence_json,created,updated) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (h["sig"], h["sensor"], h["step"], h["feature"], h["w"], h["dir"],
             h["eqp"], h["dsv"], h["cause"], h["date"], "proposed",
             h["effect"], h["q"], h["persist"],
             json.dumps({"cmode": h.get("cmode", "self")}), now, now))
        self.db.commit()
        return cur.rowcount > 0                        # False = 이미 세웠던 가설

    def set_status(self, hid, status, reason=None, ev=None, note=None):
        now = dt.datetime.now().isoformat(timespec="seconds")
        q = "UPDATE hypothesis SET status=?, updated=?"
        args = [status, now]
        if reason is not None:
            q += ", reject_reason=?"
            args.append(reason)
        if ev is not None:
            q += ", evidence_json=?"
            args.append(json.dumps(ev))
        if note is not None:
            q += ", llm_note=?"
            args.append(note)
        self.db.execute(q + " WHERE id=?", args + [hid])
        self.db.commit()

    def trial(self, hid, kind, stat, p, passed, detail=""):
        self.db.execute(
            "INSERT INTO trials(hyp_id,kind,stat,p,passed,detail,created) "
            "VALUES(?,?,?,?,?,?,?)",
            (hid, kind, float(stat), float(p), int(passed), detail,
             dt.datetime.now().isoformat(timespec="seconds")))
        self.db.commit()

    def rows(self, status=None):
        q = "SELECT * FROM hypothesis"
        if status:
            q += f" WHERE status='{status}'"
        cur = self.db.execute(q + " ORDER BY id")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def add_anchors(self, anchors):
        for a in anchors:
            self.db.execute(
                "INSERT OR IGNORE INTO anchors(eqp,dsv,cause,date) VALUES(?,?,?,?)",
                (a["eqp"], a["dsv"], a["cause"], a["date"]))
        self.db.commit()

    def anchors(self, only_new=False):
        q = "SELECT id,eqp,dsv,cause,date,explored FROM anchors"
        if only_new:
            q += " WHERE explored=0"
        return [dict(zip(("id", "eqp", "dsv", "cause", "date", "explored"), r))
                for r in self.db.execute(q + " ORDER BY date").fetchall()]

    def mark_explored(self, aid):
        self.db.execute("UPDATE anchors SET explored=1 WHERE id=?", (aid,))
        self.db.commit()

    def round_done(self, key, aid):
        self.db.execute("INSERT OR IGNORE INTO rounds(key,anchor_id) VALUES(?,?)",
                        (key, aid))
        self.db.commit()

    def round_is_done(self, key, aid):
        return self.db.execute("SELECT 1 FROM rounds WHERE key=? AND anchor_id=?",
                               (key, aid)).fetchone() is not None

    def similar(self, sensor, limit=5):
        """토론용 — 같은 센서 토큰이 겹치는 기존 가설(확정·기각) 요약."""
        toks = [t for t in str(sensor).replace("\u2212", "_").split("_")
                if len(t) >= 3][:3]
        out = []
        for st in ("confirmed", "rejected"):
            for h in self.rows(st):
                if any(t.lower() in h["sensor"].lower() for t in toks):
                    out.append(f"[{st}] {h['sensor']} step{h['step']} "
                               f"{h['feature']} W{h['window_days']} "
                               f"d={h['effect']}"
                               + (f" — 기각: {h['reject_reason']}"
                                  if h.get("reject_reason") else ""))
                if sum(1 for o in out if o.startswith(f"[{st}]")) >= limit:
                    break
        return out[:limit * 2]


# ── 심화 라운드 사다리: 고정 데이터에서도 탐사 공간을 계속 확장 ──
# (같은 (라운드,앵커)는 rounds 테이블로 1회만 — 재시작해도 이어서)
ROUNDS = [
    {"key": "w2", "windows": [14, 45, 90]},
    {"key": "diffT", "pairs": "T"},                    # 온도군 A−B 파생(mean)
    {"key": "diffP", "pairs": "P"},                    # 압력군 A−B 파생(mean)
    {"key": "relax1", "d_min": 0.65},
    {"key": "relax1w", "d_min": 0.65, "windows": [14, 45, 90]},
    {"key": "diffT2", "pairs": "T", "windows": [14, 45, 90]},
    {"key": "diffP2", "pairs": "P", "windows": [14, 45, 90]},
    {"key": "relax2", "d_min": 0.5, "persist_min": 0.5},
    {"key": "w3", "windows": [5, 10, 21, 120]},
    {"key": "diffT3", "pairs": "T", "d_min": 0.6,
     "windows": [5, 10, 21, 120]},
    {"key": "relax3", "d_min": 0.45, "persist_min": 0.45,
     "windows": [5, 10, 14, 21, 45, 90, 120]},
]


# ── 통계 유틸(외부 의존 없음) ──


def _welch(a, b):
    """Welch t 근사 p(양측, 정규 근사) — (d 효과크기, p)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = math.sqrt(max((va + vb) / 2, 1e-24))
    d = (float(np.mean(a)) - float(np.mean(b))) / sp
    se = math.sqrt(max(va / len(a) + vb / len(b), 1e-24))
    t = (float(np.mean(a)) - float(np.mean(b))) / se
    p = math.erfc(abs(t) / math.sqrt(2))
    return d, p


def _bh_fdr(ps):
    """Benjamini-Hochberg q값."""
    ps = np.asarray(ps, dtype=np.float64)
    n = len(ps)
    order = np.argsort(ps)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, ps[i] * n / rank)
        q[i] = prev
    return q


# ── 탐색·검증 엔진 ──


class Engine:
    def __init__(self, AW, store, ledger, journal, cfg=None):
        self.AW, self.store, self.led, self.journal = AW, store, ledger, journal
        self.cfg = dict({"windows": [3, 7, 30, 60], "q_max": 0.05, "d_min": 0.8,
                         "persist_min": 0.6, "refute_max": 0.1, "buffer": 7},
                        **(cfg or {}))

    # 일 요약: (day, eqp, dsv) → 센서×스텝×특징 일 중앙값
    def _day_median(self, day, eqp, dsv):
        d = self.store.load(day)
        if d is None:
            return None, None, None
        m = np.ones(len(d["ids"]), dtype=bool)
        if eqp:
            m &= np.char.find(np.char.lower(d["eqp"].astype(str)), eqp) >= 0
        if dsv:
            m &= np.char.lower(d["dsv"].astype(str)) == dsv
        if not m.any():
            return None, None, None
        with np.errstate(all="ignore"):
            med = np.nanmedian(d["vals"][m], axis=0)
        return med, d["sensors"], d["steps"]

    def _wafers(self, days, eqp, dsv, peer=False):
        """일 목록 → 웨이퍼층 (Nw,센서,스텝,특징) + 웨이퍼별 날짜.

        peer=True 면 eqp '미일치' 웨이퍼(타 설비, 같은 dsv)를 모은다 — 같은
        설비에 무보고 기간이 없을 때의 대조군 폴백."""
        mats, day_of, sens, steps = [], [], None, None
        for day in days:
            d = self.store.load(day)
            if d is None:
                continue
            m = np.ones(len(d["ids"]), dtype=bool)
            if eqp:
                hit = np.char.find(np.char.lower(d["eqp"].astype(str)), eqp) >= 0
                m &= (~hit) if peer else hit           # peer=타 설비 대조
            if dsv:
                m &= np.char.lower(d["dsv"].astype(str)) == dsv
            if not m.any():
                continue
            if sens is None:
                sens, steps = d["sensors"], d["steps"]
            if d["sensors"] != sens or d["steps"] != steps:
                continue
            mats.append(d["vals"][m])
            day_of += [day] * int(m.sum())
        if not mats:
            return None, None, None, None
        return np.concatenate(mats), np.asarray(day_of), sens, steps

    def _series(self, days, eqp, dsv, peer=False):
        """일 목록 → (일수, 센서, 스텝, 특징) 중앙값 텐서 + 축."""
        mats, sens, steps, kept = [], None, None, []
        for day in days:
            med, s9, p9 = self._day_median(day, eqp, dsv)
            if med is None:
                continue
            if sens is None:
                sens, steps = s9, p9
            if s9 != sens or p9 != steps:
                continue                               # 축 불일치 일자는 제외
            mats.append(med)
            kept.append(day)
        if not mats:
            return None, None, None, []
        return np.stack(mats), sens, steps, kept

    def _contrast_days(self, all_days, anchors, eqp, dsv, exclude, W):
        """무보고 대조 일자: 해당 설비·dsv 앵커의 [보고-W-buffer, 보고+buffer] 밖.

        제외 폭은 '이 검정의 창 W' 기준 — 최대 창 기준으로 제외하면 데이터가
        짧을 때 대조군이 통째로 사라진다(설계 결함 수정).
        """
        bad = set(exclude)
        for a in anchors:
            if a["eqp"] and eqp and a["eqp"] not in eqp and eqp not in a["eqp"]:
                continue
            d0 = dt.date.fromisoformat(a["date"])
            for k in range(-W - self.cfg["buffer"], self.cfg["buffer"] + 1):
                bad.add((d0 + dt.timedelta(days=k)).isoformat())
        return [d for d in all_days if d not in bad]

    @staticmethod
    def _pairs_of(sens, group):
        """파생 차분 후보: 같은 물리군(T/P) 원시 센서끼리 A−B (기존 차분류 제외)."""
        idx = [i for i, sn in enumerate(sens)
               if sn[:1].upper() == group
               and not sn.lower().startswith(("dp", "dt"))]
        return [(i, j) for k, i in enumerate(idx) for j in idx[k + 1:]]

    def _contrast_pick(self, all_days, anchors, eqp, dsv, win, W):
        """대조군 선택: ①같은 설비 무보고 기간 ②부족하면 타 설비(같은 dsv)
        동일 기간(그 설비들의 보고 주변도 제외). 반환 (일자들, 모드, 진단문)."""
        ctr = self._contrast_days(all_days, anchors, eqp, dsv, set(win), W)
        if len(ctr) >= 3:
            return ctr, "self", ""
        n_all = len(all_days)
        # 타 설비 폴백: 모든 앵커 주변(설비 무관) 제외한 날 — peer 웨이퍼로 대조
        bad = set(win)
        for a in anchors:
            d0 = dt.date.fromisoformat(a["date"])
            for k in range(-W - self.cfg["buffer"], self.cfg["buffer"] + 1):
                bad.add((d0 + dt.timedelta(days=k)).isoformat())
        ctr2 = [d for d in all_days if d not in bad]
        if len(ctr2) < 3:
            ctr2 = [d for d in all_days if d not in set(win)]  # 최후: 창 밖 전부(타 설비)
        diag = (f"같은 설비 대조 {len(ctr)}일<3 (전체 {n_all}일, 보고 제외 후) — "
                f"타 설비 대조 {len(ctr2)}일로 폴백")
        return (ctr2, "peer", diag) if len(ctr2) >= 3 else ([], "none",
                f"대조 불가: 전체 {n_all}일 · 같은 설비 {len(ctr)}일 · 타 설비 {len(ctr2)}일 — "
                "데이터 기간을 넓히거나 보고 밀도가 낮은 구간을 포함하세요")

    def explore(self, anchor, all_days, over=None, round_key="base"):
        """앵커 1건 → 창×센서×스텝×특징 전수 검정 → FDR 통과분 가설 제안.

        over: 심화 라운드 오버라이드 {windows, d_min, persist_min, pairs}.
        pairs='T'/'P' 면 원시 센서를 검정하는 대신 같은 물리군의 A−B(mean)
        파생 채널을 검정한다(가설 좌표: sensor="A−B", feature="mean").
        """
        over = over or {}
        d_min = float(over.get("d_min", self.cfg["d_min"]))
        p_min = float(over.get("persist_min", self.cfg["persist_min"]))
        pairs_g = over.get("pairs")
        eqp, dsv = anchor["eqp"], anchor["dsv"]
        a_date = dt.date.fromisoformat(anchor["date"])
        anchors = self.led.anchors()
        n_new = n_dup = 0
        for W in over.get("windows", self.cfg["windows"]):
            win = [(a_date - dt.timedelta(days=k)).isoformat()
                   for k in range(W, 0, -1)]
            ctr, cmode, diag = self._contrast_pick(all_days, anchors, eqp, dsv,
                                                    win, W)
            if diag:
                self.journal(f"[explore] {anchor['date']} W{W}: {diag}")
            if cmode == "none":
                continue
            Aw, a_days, sens, steps = self._wafers(win, eqp, dsv)
            Ad, _s0, _p0, kept = self._series(win, eqp, dsv)
            Bw, _bd, _s1, _p1 = self._wafers(ctr, eqp, dsv,
                                             peer=(cmode == "peer"))
            if Aw is None or Bw is None or Ad is None \
                    or len(kept) < max(2, W // 4) or len(Aw) < 6:
                continue
            if pairs_g:                                # 파생 A−B(mean) 채널로 교체
                prs9 = self._pairs_of(sens, pairs_g)
                if not prs9:
                    continue
                fm9 = F_NAMES.index("mean")
                Aw = np.stack([Aw[:, i, :, fm9] - Aw[:, j, :, fm9]
                               for i, j in prs9], axis=1)[..., None]
                Ad = np.stack([Ad[:, i, :, fm9] - Ad[:, j, :, fm9]
                               for i, j in prs9], axis=1)[..., None]
                Bw = np.stack([Bw[:, i, :, fm9] - Bw[:, j, :, fm9]
                               for i, j in prs9], axis=1)[..., None]
                sens_l = [f"{sens[i]}\u2212{sens[j]}" for i, j in prs9]
                fnames = ["mean"]
            else:
                sens_l, fnames = sens, F_NAMES
            tests = []                                 # (si,pi,fi,d,p,persist)
            q1 = np.nanpercentile(Bw, 25, axis=0)      # 대조 '웨이퍼' 분포 울타리
            q3 = np.nanpercentile(Bw, 75, axis=0)
            iqr = np.maximum(q3 - q1, 1e-9)
            hi = Ad > (q3 + 1.5 * iqr)[None]           # 일 중앙값의 울타리 초과
            lo = Ad < (q1 - 1.5 * iqr)[None]
            for si in range(Aw.shape[1]):
                for pi in range(Aw.shape[2]):
                    for fi in range(Aw.shape[3]):
                        a9 = Aw[:, si, pi, fi]
                        b9 = Bw[:, si, pi, fi]
                        if np.isfinite(a9).sum() < 6 or np.isfinite(b9).sum() < 12:
                            continue
                        d, p = _welch(a9, b9)
                        if abs(d) < d_min:
                            continue
                        pr = float((hi if d > 0 else lo)[:, si, pi, fi].mean())
                        tests.append((si, pi, fi, d, p, pr))
            if not tests:
                continue
            qs = _bh_fdr([t[4] for t in tests])
            for (si, pi, fi, d, p, pr), q in zip(tests, qs):
                if q > self.cfg["q_max"] or pr < p_min:
                    continue
                h = {"sensor": sens_l[si], "step": steps[pi],
                     "feature": fnames[fi],
                     "w": W, "dir": (1 if d > 0 else -1), "eqp": eqp, "dsv": dsv,
                     "cause": anchor["cause"], "date": anchor["date"],
                     "effect": round(float(d), 3), "q": round(float(q), 5),
                     "persist": round(pr, 3), "cmode": cmode}
                h["sig"] = Ledger.sig(h["sensor"], h["step"], h["feature"],
                                      W, h["dir"], h["cause"])
                if self.led.propose(h):
                    n_new += 1
                else:
                    n_dup += 1
        self.journal(f"[explore:{round_key}] 앵커 {anchor['eqp']}/{anchor['dsv']} "
                     f"{anchor['date']} — 신규 가설 {n_new} · 중복 차단 {n_dup}")
        return n_new

    # ── 검증 ──
    def _hyp_series(self, h, days, peer=False):
        """가설 좌표의 웨이퍼값 1-D + 웨이퍼별 날짜. 파생(A−B) 좌표 지원."""
        Aw, a_days, sens, steps = self._wafers(days, h["anchor_eqp"],
                                               h["anchor_dsv"], peer=peer)
        if Aw is None or int(h["step"]) not in steps:
            return None, None
        pi = steps.index(int(h["step"]))
        fi = F_NAMES.index(h["feature"])
        sn = str(h["sensor"])
        if "\u2212" in sn:                            # 파생 차분 A−B
            a9, b9 = sn.split("\u2212", 1)
            if a9 not in sens or b9 not in sens:
                return None, None
            return (Aw[:, sens.index(a9), pi, fi]
                    - Aw[:, sens.index(b9), pi, fi]), a_days
        if sn not in sens:
            return None, None
        return Aw[:, sens.index(sn), pi, fi], a_days

    def verify(self, h, all_days):
        led, cfg = self.led, self.cfg
        led.set_status(h["id"], "testing")
        a_date = dt.date.fromisoformat(h["anchor_date"])
        W = int(h["window_days"])
        win = [(a_date - dt.timedelta(days=k)).isoformat() for k in range(W, 0, -1)]
        anchors = led.anchors()
        ctr_days, cmode, _dg = self._contrast_pick(
            all_days, anchors, h["anchor_eqp"], h["anchor_dsv"], win, W)
        if cmode == "none":
            led.set_status(h["id"], "rejected", reason="대조 일자 없음(데이터 기간 부족)")
            return "rejected"
        peer9 = cmode == "peer"
        aw, adays = self._hyp_series(h, win)
        cw, _k2 = self._hyp_series(h, ctr_days, peer=peer9)
        if aw is None or cw is None:
            led.set_status(h["id"], "rejected", reason="데이터 부족(창/대조 시계열 없음)")
            return "rejected"
        # 1) 반증: 무보고 기간의 같은 길이 창들에서 동일 방향·크기 효과 출현율
        d0, _p0 = _welch(aw, cw)
        false_hits = total = 0
        ctr_sorted = sorted(ctr_days)
        for s0 in range(0, max(0, len(ctr_sorted) - W), max(1, W // 2)):
            sub = ctr_sorted[s0:s0 + W]
            if len(sub) < max(2, W // 4):
                continue
            sw, _ = self._hyp_series(h, sub, peer=peer9)
            rest = [d for d in ctr_sorted if d not in set(sub)]
            rw, _ = self._hyp_series(h, rest, peer=peer9)
            if sw is None or rw is None:
                continue
            dd, _pp = _welch(sw, rw)
            total += 1
            if dd * h["direction"] >= cfg["d_min"]:
                false_hits += 1
        rate = (false_hits / total) if total else 1.0
        ok_refute = rate <= cfg["refute_max"]
        led.trial(h["id"], "refute", rate, rate, ok_refute,
                  f"무보고 창 {total}개 중 동일 패턴 {false_hits}")
        # 2) 선행성: 일 중앙값 시계열에서 이탈 시작일(+창 분류)
        q1, q3 = np.nanpercentile(cw, 25), np.nanpercentile(cw, 75)
        fence = (q3 + 1.5 * (q3 - q1)) if h["direction"] > 0 else \
                (q1 - 1.5 * (q3 - q1))
        dmed, ddays = [], []
        for day9 in win:
            m9 = adays == day9
            if m9.any() and np.isfinite(aw[m9]).any():
                dmed.append(float(np.nanmedian(aw[m9])))
                ddays.append(day9)
        dmed = np.asarray(dmed)
        beyond = (dmed > fence) if h["direction"] > 0 else (dmed < fence)
        onset = None
        for k9, b9 in enumerate(beyond):
            if b9 and beyond[k9:].mean() >= 0.5:
                onset = ddays[k9]
                break
        ok_precede = onset is not None
        lead = (a_date - dt.date.fromisoformat(onset)).days if onset else -1
        led.trial(h["id"], "precede", lead, 0.0 if ok_precede else 1.0, ok_precede,
                  f"이탈 시작 {onset or '없음'} (보고 {lead}일 전)")
        # 3) 홀드아웃: 같은 원인의 다른 앵커에서 재현(없으면 생략 표시)
        others = [a for a in anchors
                  if a["cause"] == h["anchor_cause"]
                  and (a["eqp"], a["date"]) != (h["anchor_eqp"], h["anchor_date"])]
        ok_hold, hold_note = None, "다른 앵커 없음(생략)"
        for a in others[:5]:
            od = dt.date.fromisoformat(a["date"])
            owin = [(od - dt.timedelta(days=k)).isoformat() for k in range(W, 0, -1)]
            h2 = dict(h, anchor_eqp=a["eqp"], anchor_dsv=a["dsv"])
            ow, _ = self._hyp_series(h2, owin)
            octr, ocm, _od = self._contrast_pick(all_days, anchors,
                                                 a["eqp"], a["dsv"], owin, W)
            if ocm == "none":
                continue
            oc, _ = self._hyp_series(h2, octr, peer=(ocm == "peer"))
            if ow is None or oc is None:
                continue
            dd, _pp = _welch(ow, oc)
            ok_hold = dd * h["direction"] >= 0.5
            hold_note = f"{a['eqp']} {a['date']} d={dd:+.2f}"
            led.trial(h["id"], "holdout", dd, 0.0, bool(ok_hold), hold_note)
            if ok_hold:
                break
        passed = ok_refute and ok_precede and (ok_hold is not False)
        if not passed:
            why = []
            if not ok_refute:
                why.append(f"반증 실패(무보고 기간 출현율 {rate:.2f})")
            if not ok_precede:
                why.append("선행성 없음(창 내 이탈 시작 없음)")
            if ok_hold is False:
                why.append(f"홀드아웃 재현 실패({hold_note})")
            led.set_status(h["id"], "rejected", reason=" / ".join(why))
            return "rejected"
        # 토론 게이트: 통계 통과 후 비판자(gpt-oss)와 합의해야 확정(사용자 지시)
        okd, why_d, note_d = self._debate(h, {
            "effect_d": d0, "fdr_q": h.get("fdr_q"),
            "persist": h.get("persist_ratio"), "refute_rate": round(rate, 3),
            "onset": onset, "lead_days": lead, "holdout": hold_note,
            "n_window": int(np.isfinite(aw).sum()),
            "n_contrast": int(np.isfinite(cw).sum())})
        led.trial(h["id"], "debate", 0.0, 0.0, bool(okd), (why_d or "")[:500])
        if not okd:
            led.set_status(h["id"], "rejected",
                           reason="비판자 미합의 — " + (why_d or "")[:300])
            return "rejected"
        if note_d:
            led.set_status(h["id"], "testing", note=note_d[:800])
        # 달력 증거: 창 앞뒤 포함 일별 이상 웨이퍼 목록(웨이퍼층 판정)
        ev = self._calendar_ev(h, fence, all_days, win)
        ev["onset"] = onset
        ev["lead_days"] = lead
        ev["kind"] = "단기(2~3일)" if W <= 3 else ("주간" if W <= 7 else "장기(1~2달)")
        ev["fence"] = round(float(fence), 4)
        ev["c_med"] = round(float(np.nanmedian(cw)), 4)   # 대조 분포(카드 z 표기용)
        ev["c_iqr"] = round(float(q3 - q1), 4)
        ev["holdout"] = hold_note
        ev["contrast"] = ("같은 설비 무보고 기간" if cmode == "self"
                          else "타 설비 동일 기간(같은 설비 무보고 기간 부족)")
        ev["refute_rate"] = round(rate, 3)
        led.set_status(h["id"], "confirmed", ev=ev)
        return "confirmed"

    def _debate(self, h, stats):
        """가설 1건 비판자 토론(최대 3라운드) — (합의여부, 요지, 합의노트).

        비판자=RCA_CRITIC_MODEL(gpt-oss 등). 반박은 주 모델(RCA_LLM_MODEL)이,
        주 모델이 없으면 증거 패킷 재제시(결정적)로 한다. 비판자 미설정이나
        LLM 오류는 '토론 생략'으로 통계 결과만으로 확정(완주 원칙).
        """
        critic = os.environ.get("RCA_CRITIC_MODEL", "").strip()
        if not critic:
            return True, "토론 생략(RCA_CRITIC_MODEL 미설정 — 통계 검증만)", None
        try:
            from io_recon_tcn import rca_causal as RC
            host = (os.environ.get("OLLAMA_HOST")
                    or "http://127.0.0.1:11434").rstrip("/")
            main = os.environ.get("RCA_LLM_MODEL", "").strip()
            tmo = int(os.environ.get("RCA_LLM_TIMEOUT", "300"))
            packet = {
                "가설": _desc(dict(h)),
                "앵커": f"{h['anchor_eqp']}/{h['anchor_dsv']} {h['anchor_date']}"
                        f" ({h['anchor_cause']})",
                "통계": stats,
                "기존_유사_가설": self.led.similar(h["sensor"]),
            }
            pj = json.dumps(packet, ensure_ascii=False, default=str)[:9000]
            sys_c = ("너는 반도체 설비 이상 패턴 가설의 적대적 검증자다. 통계 증거 "
                     "패킷만 근거로 가설을 공격하라: ①기존 유사 가설과 사실상 중복"
                     "(단, 기존_유사_가설 목록이 비어 있으면 중복 아님 — 이를 "
                     "의심 근거로 쓰지 마라) ②근거 대비 비약 ③대안 설명(반증 창 "
                     "출현율 등) ④수치 인용 오류.\n판단 기준표(이 기준을 충족하는 "
                     "항목을 '부족하다'고 공격하면 무효): 효과 |d|>=0.8 크다 / "
                     "FDR q<0.05 유의 / 지속률>=0.6 지속적 / 반증율<=0.1 특이 / "
                     "lead_days>0 선행. 패킷에 적힌 수치 사실만 근거로 쓰고, "
                     "일반론·추정 금지. 이 가설은 인과 단정이 아니라 '보고 이전 "
                     "기간에 지속된 통계적 패턴' 주장이다 — 인과 미확정을 이유로 "
                     "공격하면 무효다. 각 지적을 근거 수치와 함께 목록으로. "
                     "타당한 이의가 없으면 정확히 '이의 없음' 만 답하라.")
            attack = RC._ollama_chat(host, critic, sys_c,
                                     "증거 패킷:\n" + pj, tmo)
            transcript = []
            for rnd in range(3):
                if not attack or "이의 없음" in str(attack)[:30] \
                        or "동의" in str(attack)[:30]:
                    note = ("비판자 합의"
                            + (f"({rnd}라운드 후)" if rnd else "(즉시)")
                            + (": " + str(attack)[:300] if attack else ""))
                    return True, note, note
                transcript.append(str(attack)[:600])
                if main:
                    reb = RC._ollama_chat(
                        host, main,
                        "너는 가설 제안자다. 비판자의 공격 중 타당한 것은 인정하고 "
                        "증거 패킷의 수치로 반박 가능한 것만 반박하라. 패킷 밖 "
                        "사실 추가 금지. 간결한 목록으로.",
                        "증거 패킷:\n" + pj + "\n\n[비판자 공격]\n"
                        + str(attack), tmo) or ""
                else:
                    reb = ("(결정적 반박) 통계 재제시: " + json.dumps(
                        stats, ensure_ascii=False, default=str))
                attack = RC._ollama_chat(
                    host, critic, sys_c,
                    "증거 패킷:\n" + pj + "\n\n[직전 공격]\n"
                    + transcript[-1] + "\n\n[제안자 반박]\n" + str(reb)[:3000]
                    + "\n\n남은 타당한 이의만 목록으로. 없으면 정확히 "
                      "'이의 없음' 만 답하라.", tmo)
            # 최종 이진 판정: 목록 생성 습관 탓에 '이의 없음'을 못 내는 모델 대비
            final = RC._ollama_chat(
                host, critic,
                "너는 최종 판정자다. 한 단어로만 답하라.",
                "기준표: 효과 |d|>=0.8 크다 / q<0.05 유의 / 지속률>=0.6 / "
                "반증율<=0.1 / lead_days>0 선행. 인과 단정이 아닌 '지속 패턴' "
                "주장이다.\n증거 패킷:\n" + pj + "\n\n지금까지의 토론을 "
                "감안해, 이 가설을 데이터에 존재하는 지속 패턴으로 인정하는가? "
                "반드시 '인정' 또는 '불인정' 한 단어로만 답하라.", tmo) or ""
            if "불인정" not in final and "인정" in final:
                return True, ("비판자 최종 인정(3라운드 토론 후). 마지막 이의: "
                              + str(attack or "")[:200]), \
                       "비판자 최종 인정 — " + str(attack or "")[:300]
            return False, ("비판자 최종 불인정: "
                           + str(attack or transcript[-1])[:300]), None
        except Exception as e:                         # noqa: BLE001
            self.journal(f"[debate] LLM 오류 — 토론 생략(통계 검증만): {e}")
            return True, f"토론 생략(LLM 오류: {e})", None

    def _calendar_ev(self, h, fence, all_days, win):
        """가설 달력 증거: {date: {bad:[웨이퍼…], ok:n}} — 웨이퍼층 판정."""
        si_f = F_NAMES.index(h["feature"])
        d0 = dt.date.fromisoformat(win[0]) - dt.timedelta(days=7)
        d1 = dt.date.fromisoformat(h["anchor_date"]) + dt.timedelta(days=3)
        days = {}
        sn = str(h["sensor"])
        drv = sn.split("\u2212", 1) if "\u2212" in sn else None
        for day in all_days:
            if not (d0.isoformat() <= day <= d1.isoformat()):
                continue
            d = self.store.load(day)
            if d is None or int(h["step"]) not in d["steps"]:
                continue
            if drv:
                if drv[0] not in d["sensors"] or drv[1] not in d["sensors"]:
                    continue
            elif sn not in d["sensors"]:
                continue
            pi = d["steps"].index(int(h["step"]))
            m = np.ones(len(d["ids"]), dtype=bool)
            if h["anchor_eqp"]:
                m &= np.char.find(np.char.lower(d["eqp"].astype(str)),
                                  h["anchor_eqp"]) >= 0
            if h["anchor_dsv"]:
                m &= np.char.lower(d["dsv"].astype(str)) == h["anchor_dsv"]
            if not m.any():
                continue
            if drv:
                vals = (d["vals"][m, d["sensors"].index(drv[0]), pi, si_f]
                        - d["vals"][m, d["sensors"].index(drv[1]), pi, si_f])
            else:
                vals = d["vals"][m, d["sensors"].index(sn), pi, si_f]
            ids = d["ids"][m]
            bad = (vals > fence) if h["direction"] > 0 else (vals < fence)
            bad &= np.isfinite(vals)
            days[day] = {"bad": [str(v) for v in ids[bad][:50]],
                         "ok": int((~bad & np.isfinite(vals)).sum())}
        return {"days": days}


# ── LLM 레이어(로컬 — 없어도 완주) ──


def llm_note_for(h):
    try:
        from io_recon_tcn import rca_causal as RC
        facts = (f"센서 {h['sensor']} 스텝 {h['step']} 특징 {h['feature']} 이 "
                 f"보고일({h['anchor_date']}) 이전 {h['window_days']}일 동안 "
                 f"{'상승' if h['direction'] > 0 else '하락'} 지속"
                 f"(효과 d={h['effect']}, 지속률 {h['persist_ratio']}). "
                 f"불량원인 표기: {h['anchor_cause']}.")
        r = RC.narrate_pattern("[가설 확정 후보]\n" + facts)
        if r.get("engine") != "error" and r.get("narrative"):
            return str(r["narrative"])[:800]
    except Exception:                                  # noqa: BLE001
        pass
    return None


# ── 데몬 ──


def _desc(h):
    return (f"{h['sensor']} step{h['step']} {h['feature']} "
            f"{'상승' if h['direction'] > 0 else '하락'} — 보고 전 "
            f"{h['window_days']}일 지속 (효과 d={h['effect']}, q={h['fdr_q']})")


def write_outputs(out, led):
    """보고서(개조식 텍스트)·확정 제안 JSON — 사람 승인용."""
    props = Path(out) / "proposals"
    props.mkdir(exist_ok=True)
    conf = led.rows("confirmed")
    for h in conf:
        fp = props / f"hyp_{h['id']:04d}.json"
        if not fp.is_file():
            fp.write_text(json.dumps(
                {"id": h["id"], "sensor": h["sensor"], "step": h["step"],
                 "feature": h["feature"], "window_days": h["window_days"],
                 "direction": h["direction"], "effect": h["effect"],
                 "anchor": {"eqp": h["anchor_eqp"], "dsv": h["anchor_dsv"],
                            "cause": h["anchor_cause"], "date": h["anchor_date"]},
                 "desc": _desc(h), "llm_note": h.get("llm_note")},
                ensure_ascii=False, indent=1), encoding="utf-8")
    rows = led.rows()
    diag = []
    anc = led.anchors()
    diag.append(f"앵커 {len(anc)}건" + ("" if anc else " ← 0건이면 CSV 미인식(경로/형식)"))
    for a9 in anc[:10]:
        diag.append(f"  · {a9['eqp']}/{a9['dsv']} {a9['date']} ({a9['cause']}) "
                    f"탐색 {'완료' if a9['explored'] else '대기'}")
    why = {}
    for h9 in rows:
        if h9["status"] == "rejected":
            k9 = (h9.get("reject_reason") or "?")[:24]
            why[k9] = why.get(k9, 0) + 1
    for k9, c9 in sorted(why.items(), key=lambda x: -x[1])[:6]:
        diag.append(f"  · 기각 {c9}건 — {k9}…")
    rep = ["# 패턴 탐사 일지", f"생성 {dt.datetime.now().isoformat(timespec='seconds')}",
           "", "## 진단 요약", *("- " + d9 for d9 in diag), "",
           f"가설 총 {len(rows)} — 확정 {sum(1 for r in rows if r['status'] == 'confirmed')}"
           f" · 기각 {sum(1 for r in rows if r['status'] == 'rejected')}"
           f" · 검증중 {sum(1 for r in rows if r['status'] == 'testing')}"
           f" · 제안 {sum(1 for r in rows if r['status'] == 'proposed')}", ""]
    for h in rows:
        rep.append(f"- [{h['status']}] #{h['id']} {_desc(h)}"
                   + (f" — 기각: {h['reject_reason']}" if h["reject_reason"] else ""))
    (Path(out) / "reports").mkdir(exist_ok=True)
    (Path(out) / "reports" / "latest.md").write_text("\n".join(rep), encoding="utf-8")


def tick(AW, S, store, led, eng, folders, csv_path, out, journal):
    journal("[tick] 점검 시작 — ①특징 집계 ②앵커 ③탐색 ④검증 순서로 진행")
    n_new_days, all_days = build_features(
        AW, S, store, folders, journal,
        workers=int(CFG.get("workers") or 1),
        model_dir=CFG.get("model"), out=CFG.get("out"))
    if csv_path and Path(csv_path).is_file():
        rows_a = read_anchors(csv_path, journal=journal)
        if not rows_a:
            journal(f"[anchor] CSV 에서 앵커 0건 — 헤더(설비명,dsv,불량원인,"
                    f"발생일자)·날짜 8자리·인코딩을 확인하세요: {csv_path}")
        led.add_anchors(rows_a)
    elif csv_path:
        journal(f"[anchor] CSV 없음: {csv_path} — 앵커를 추가할 수 없습니다")
    all_days = store.days()
    n_act = 0
    for a in led.anchors(only_new=True):
        try:
            eng.explore(a, all_days)
            led.mark_explored(a["id"])
            n_act += 1
        except Exception:                              # noqa: BLE001
            import traceback
            journal(f"[err] 앵커 #{a['id']} 탐색 실패 — 이 앵커만 건너뛰고 계속\n"
                    + traceback.format_exc())
    # 심화 라운드: 새 앵커도, 검증 대기 가설도 없으면 탐사 공간을 한 단계 확장
    # (고정 데이터에서도 계속 가설을 만든다 — 사용자 지시. 라운드는 원장에 기록,
    #  재시작해도 이어서. 서명 차단으로 같은 가설은 절대 반복되지 않는다.)
    if not n_act and not led.anchors():
        journal("[deepen] 앵커가 0건 — 탐색을 시작할 수 없습니다. 불량 내역 CSV "
                "경로/형식을 확인하세요(가설·소진과 무관한 상태)")
    elif not n_act and not (led.rows("proposed") or led.rows("testing")):
        for r in ROUNDS:
            todo_a = [a for a in led.anchors()
                      if not led.round_is_done(r["key"], a["id"])]
            if not todo_a:
                continue
            journal(f"[deepen] 심화 라운드 '{r['key']}' 시작 — 앵커 {len(todo_a)}건 "
                    f"({ {k9: v9 for k9, v9 in r.items() if k9 != 'key'} })")
            for a in todo_a:
                try:
                    eng.explore(a, all_days, over=r, round_key=r["key"])
                except Exception:                      # noqa: BLE001
                    import traceback
                    journal(f"[err] 라운드 {r['key']} 앵커 #{a['id']} 실패 — "
                            f"건너뛰고 계속\n" + traceback.format_exc())
                led.round_done(r["key"], a["id"])
            n_act += 1
            break
        else:
            journal("[deepen] 심화 라운드 전부 소진 — 현 데이터의 탐사 공간을 "
                    "모두 검정했습니다(새 데이터/새 보고가 오면 자동 재개)")
    for h in led.rows("proposed") + led.rows("testing"):
        try:
            res = eng.verify(h, all_days)
        except Exception:                              # noqa: BLE001
            import traceback
            journal(f"[err] 가설 #{h['id']} 검증 실패 — 이 건만 대기로 두고 계속\n"
                    + traceback.format_exc())
            continue
        journal(f"[verify] #{h['id']} {_desc(h)} → {res}")
        if res == "confirmed":
            note = llm_note_for(h)
            if note:
                led.set_status(h["id"], "confirmed", note=note)
        n_act += 1
    if n_act or n_new_days:
        write_outputs(out, led)
    if not (n_act or n_new_days):
        journal("[tick] 신규 없음 — 주어진 데이터의 탐사·검증은 이미 완료 상태"
                "(새 날짜/새 CSV 행이 오면 자동 반영)")
    return n_act


def main():
    ap = argparse.ArgumentParser(description="패턴 탐사 데몬(가설 생성·검증)")
    ap.add_argument("--data", required=True, help="로그 폴더(;로 여러 개)")
    ap.add_argument("--csv", required=True, help="불량 내역 CSV(설비명,dsv,불량원인,발생일자)")
    ap.add_argument("--out", required=True, help="탐사 작업 폴더(원장·특징·보고서)")
    ap.add_argument("--model", help="모델 폴더(생략=앱 기본)")
    ap.add_argument("--once", action="store_true", help="1회 수행 후 종료(검증용)")
    ap.add_argument("--tick", type=int, default=600, help="주기(초, 기본 600)")
    ap.add_argument("--workers", type=int, default=0,
                    help="특징 집계 병렬 워커 수(0=자동: 코어/2, 최대 16)")
    ap.add_argument("--device", default="",
                    help="특징 계산 장치: cuda=GPU 위주(앱과 동시 운용 시 권장), "
                         "cpu=torch CPU 배치, auto=가능하면 cuda, 빈값=기존 numpy")
    a = ap.parse_args()
    # run_hunter.sh 가 도구 폴더로 cd 하므로 상대경로는 사용자 의도와 어긋난다
    # → 전부 절대경로로 확정하고 시작 로그에 표시(v154: 조용한 실패 제거)
    a.out = os.path.abspath(a.out)
    a.csv = os.path.abspath(a.csv)
    a.data = ";".join(os.path.abspath(p) for p in str(a.data).split(";") if p)
    if not Path(a.csv).is_file():
        raise SystemExit(f"[hunter] 불량 내역 CSV 를 찾을 수 없습니다: {a.csv}\n"
                         "  → 경로를 절대경로로 확인하세요(상대경로는 도구 폴더 기준)")
    for p9 in a.data.split(";"):
        if not Path(p9).is_dir():
            raise SystemExit(f"[hunter] 데이터 폴더가 없습니다: {p9}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_fp = out / "hunter.log"

    def journal(msg):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        try:
            with open(log_fp, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:                              # noqa: BLE001
            pass

    workers = a.workers or max(1, min(16, (os.cpu_count() or 2) // 2))
    if a.device:
        workers = 1                                    # GPU 모드: 배치가 병렬 대체
    CFG.update(workers=workers, model=a.model, out=str(out), device=a.device)
    journal(f"[hunter] {HUNTER_VER} 시작 — out={out} · csv={a.csv} · "
            f"집계 워커 {workers}개")
    AW, S = _bootstrap(a.model, cache_hint=str(out))
    import test_avg_waveform as AWm                    # noqa: F401
    folders = AW.split_folders(a.data)
    store = FeatStore(out)
    led = Ledger(out)
    eng = Engine(AW, store, led, journal)
    err_sig, err_rep = "", 0
    while True:
        n = 0
        try:
            n = tick(AW, S, store, led, eng, folders, a.csv, out, journal)
            journal(f"[tick] 처리 {n}건")
            err_sig, err_rep = "", 0
        except Exception as e:                         # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            journal(f"[err] tick 실패 — 원인 전체:\n{tb}")
            sig9 = f"{type(e).__name__}:{e}"[:200]
            err_rep = err_rep + 1 if sig9 == err_sig else 1
            err_sig = sig9
            if a.once or err_rep >= 3:
                journal(f"[hunter] 같은 오류 {err_rep}회 — 재시도해도 해소되지 "
                        f"않는 오류라 중단합니다. 위 원인 전체를 회신해 주세요: "
                        f"{sig9}")
                raise SystemExit(1)
        if a.once:
            journal("[hunter] --once 완료 — 결과: reports/latest.md · "
                    "앱 [패턴 탐사] 탭에서 열람")
            break
        if n:
            journal("[tick] 작업 남음 — 대기 없이 즉시 계속(가설·검증이 소진될 "
                    "때까지 주기를 기다리지 않습니다)")
            continue
        journal(f"[tick] 소진/신규 없음 — {max(30, a.tick)}초 후 재점검"
                f"(주기는 새 로그·새 CSV 행 감지용입니다)")
        time.sleep(max(30, a.tick))


if __name__ == "__main__":
    main()
