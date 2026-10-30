"""src 제어기 UX — FastAPI 백엔드 (초안)

고객사 제어기(HMI)에 들어갈 검사 화면의 서버.
  ① 데이터가 '특정 폴더(inbox)' 로 들어온다.
  ② [검사 시작] → 폴더의 모든 wafer CSV 를 forecast 모델로 추론.
  ③ 결과 테이블(이상점수·판정·핵심원인) 표시.
  ④ '상세 분석' 탭 → overlay(실제 vs 기대) + heatmap + RCA(응답분류).

실제 추론은 src 의 함수를 그대로 재사용한다(중복 구현 없음):
  - OnnxForecastModel.load        : 학습된 모델(artifacts/tcn_forecast.onnx)
  - TCNForecastModel.load         : torch 폴백(artifacts/tcn_forecast.pt)
  - load_dataset                  : 폴더 CSV → wafer record
  - model.assess                  : wafer 이상점수 + 센서별 원인
  - recon_explain.narrate_recon   : 자연어 RCA(응답분류)
  - recon_explain.plot_recon_*    : overlay / heatmap PNG

실행:
  cd /home/node/io_dev
  uvicorn src.app.server:app --host 0.0.0.0 --port 8000
  # 또는:  python -m src.app.server
"""
from __future__ import annotations

import csv
import datetime as dt
import gc
import hashlib
import math
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import queue
import tempfile
import threading
import time
import traceback
from contextlib import asynccontextmanager
from functools import wraps
from contextvars import copy_context
import inspect
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

# ── 직접 실행(python ...app/server.py / -m ...app.server) 지원 ──
if __package__ in (None, "", "app"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

# ★ numpy 보다 먼저 — OpenBLAS/MKL 은 import 시점에 스레드 수 환경변수를 읽는다.
#   여기서 늦으면 '코어 예약' 설정이 계산 라이브러리에 안 먹는다.
from src.app import cpu_limit as CPU
from src.app.ui_runtime import IDENTITY, FreshUIResponses
from src.app.chamber_models import (EngineRouter, discover_chamber_models,
                                    has_chamber_layout, artifact_digest, canonical_artifacts)
_CPU_BUDGET = CPU.apply_early()

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.auxiliary_sensors import auxiliary_sensor_columns, restore_context_modes, validate_auxiliary_record
from src.config import IOConfig
from src.wafer_score_override import AUDIT_FIELDS, normalize_override, effective_wafer_score, artifact_override, apply_selected_override, override_identity
from src.data import load_dataset, _data_files, _resolve_existing_dir_ci
# TCNForecastModel(=torch) 은 지연 import — onnx 백엔드만 쓰면 torch 불필요(배포 경량).
from src import recon_explain as RX

# ── 경로 (개발 실행 / PyInstaller exe 둘 다 지원) ──
#   frozen(exe): static·artifacts 는 읽기전용 번들(_MEIPASS),
#                inbox·reports 는 exe 옆 '쓰기 가능' 폴더로 분리.
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    _BUNDLE = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    PKG = _BUNDLE / "src"                  # 번들 내 리소스(읽기전용)
    _WRITABLE = Path(sys.executable).parent         # exe 옆(쓰기 가능)
else:
    PKG = Path(__file__).resolve().parents[1]       # src/
    _WRITABLE = PKG
STATIC = PKG / "app" / "static"

# ── 저장소 경로 — 배포판: 앱 옆 고정(항상성). 특수 상황은 환경변수로만 ──
STORAGE_KEYS = ("data", "archive", "log", "report", "backup", "rca", "train", "model")


def _storage_cfg() -> dict:
    """★ 배포판: storage.json 을 읽지 않는다 — 저장소 경로는 앱 옆 고정(항상성).
    개발/특수 상황은 환경변수(IO_LOG_DIR·IO_BACKUP 등)로만 바꾼다."""
    return {}


_STORAGE = _storage_cfg()
_STORAGE_FALLBACK: list = []                             # 쓸 수 없어 기본값으로 대체된 항목(UI 경고용)
_STALE_PATHS: dict = {}                                  # 설정에 남은 '이 PC 에 없는 경로'(다른 PC 에서 옮겨온 설정)


def _note_stale_path(key: str, want: str):
    """설정에 남아 있으나 이 PC 에서 찾을 수 없는 경로를 기록(설정 화면 경고용)."""
    if want:
        _STALE_PATHS[key] = str(want)


def _storage_rel(p: Path) -> str:
    """저장용 경로 문자열 — 앱 폴더(쓰기루트) 안이면 '상대경로'로 저장한다.

    ★ PC 를 바꾸거나 폴더째 옮겨도 그대로 동작하게 하기 위함(코드베이스 공통 규칙).
      앱 밖의 공용 경로(예: D 드라이브의 공용 폴더)만 절대경로로 남는다."""
    try:
        return str(Path(p).resolve().relative_to(Path(_WRITABLE).resolve()))
    except Exception:                                    # noqa: BLE001 — 앱 밖 → 절대경로 유지
        return str(p)


_PROBE_TIMEOUT = float(os.environ.get("IO_PATH_PROBE_SEC", "3"))


def _probe_writable(p: Path, timeout: float = None):
    """폴더 생성 + 쓰기 가능 검사. **반드시 timeout 안에 끝난다.**

    ★ 이게 없으면 앱이 시작조차 못 한다: 저장된 경로가 끊긴 네트워크 드라이브
      (\\서버\공유, 연결 끊긴 Z:) 면 mkdir/write 가 SMB 타임아웃까지 수십 초~수 분
      블록된다. 창 없는(windowed) exe 라 그동안 화면에는 아무것도 안 뜬다
      = "더블클릭해도 아무 반응 없음". 별도 스레드로 돌려 시간을 끊는다.
    """
    out = {}

    def _work():
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / f".w_{os.getpid()}.tmp"
            t.write_text("x", encoding="utf-8")
            t.unlink()
            out["ok"] = True
        except Exception as e:                           # noqa: BLE001
            out["ok"] = False
            out["err"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=_work, daemon=True)     # daemon: 응답 없어도 종료를 막지 않음
    th.start()
    th.join(timeout if timeout is not None else _PROBE_TIMEOUT)
    if th.is_alive():
        return False, (f"{_PROBE_TIMEOUT:.0f}초 안에 응답이 없습니다"
                       " (연결이 끊긴 네트워크 드라이브일 수 있습니다)")
    return bool(out.get("ok")), out.get("err", "")


def _exists_bounded(p: Path, timeout: float = None) -> bool:
    """폴더 존재 확인 — 응답 없는 네트워크 경로에서 멈추지 않도록 시간 제한.
    (끊긴 드라이브를 챔버 폴더로 지정해 두면 검사 요청이 통째로 멈춘다)"""
    out = {}

    def _work():
        try:
            out["v"] = p.exists() and p.is_dir()
        except Exception:                                # noqa: BLE001
            out["v"] = False

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(timeout if timeout is not None else _PROBE_TIMEOUT)
    if th.is_alive():
        print(f"[storage] 응답 없는 경로를 건너뜁니다: {p}", flush=True)
        _note_stale_path(f"slow:{p}", str(p))
        return False
    return bool(out.get("v"))


def _storage_dir(key: str, default: Path, env: str = "") -> Path:
    """저장소 경로 결정 — 환경변수 > storage.json > 기본값.

    · 상대경로는 앱 폴더 기준으로 복원(다른 PC 로 옮겨도 유지).
    · 지정 경로를 쓸 수 없으면(드라이브 없음 등) 조용히 죽지 말고 기본값으로 폴백 + 경고.
    """
    if env and os.environ.get(env):
        return Path(os.environ[env]).expanduser()
    v = _STORAGE.get(key)
    if not v:
        return Path(default)
    raw = str(v).strip()
    p = Path(raw).expanduser()
    # ★ 윈도우 절대경로(D:\... , \\서버\공유)는 리눅스에서 is_absolute() 가 False 라
    #   앱 폴더 밑에 이상한 이름의 폴더를 만들어 버린다 → 직접 판별한다.
    _win_abs = bool(re.match(r"^[A-Za-z]:[\\/]", raw)) or raw.startswith("\\\\")
    if not p.is_absolute() and not _win_abs:
        p = Path(_WRITABLE) / p                          # 상대 → 앱 폴더 기준
    elif _win_abs and not p.is_absolute():               # 이 OS 에서는 못 쓰는 경로
        print(f"[storage] '{key}' 는 이 PC 에서 쓸 수 없는 경로입니다({raw}) → 기본값 사용",
              flush=True)
        _STORAGE_FALLBACK.append({"key": key, "want": raw, "used": str(default)})
        return Path(default)
    ok, err = _probe_writable(p)
    if ok:
        return p
    print(f"[storage] '{key}' 경로를 쓸 수 없어 기본값으로 대체합니다: {p} ({err})", flush=True)
    _STORAGE_FALLBACK.append({"key": key, "want": str(p), "used": str(default), "why": err})
    return Path(default)


def _cpu_workers(cap: int = 8, lo: int = 1) -> int:
    """스레드/프로세스 풀 크기 — '코어 예약' 설정을 반영한다.

    예전에는 곳곳에서 min(8, os.cpu_count()) 를 따로 계산해 설정이 안 먹었다.
    이제 전부 이 함수를 지나간다.
    """
    try:
        n = int((_CPU_BUDGET or {}).get("threads") or 0) or (os.cpu_count() or lo)
    except Exception:                                    # noqa: BLE001
        n = os.cpu_count() or lo
    return max(lo, min(cap, n))


def _has_model(d: Path) -> bool:
    """모델 폴더 판정 — 이름 무관 .onnx 또는 .pt 가 있으면 유효(recon_A.* 강제 아님)."""
    d = Path(d)
    if has_chamber_layout(d):
        return True
    if any(re.match(r"^dsv[1-6][-_]", p.name, re.I) for p in d.glob("*.onnx")):
        return False  # removed layout: do not silently use one chamber file as a shared model
    return bool(list(d.glob("*.onnx")) or list(d.glob("*.pt")))


def _find_pt(d: Path):
    """폴더의 .pt 웨이트(tcn_forecast.pt 우선, 없으면 아무 .pt)."""
    d = Path(d)
    cand = d / "tcn_forecast.pt"
    return cand if cand.exists() else next(iter(sorted(d.glob("*.pt"))), None)


def _auto_convert_pt(art: Path) -> Dict:
    r"""모델 폴더에 .pt 만 있고 짝맞는 .onnx(+메타) 가 없으면 자동으로 ONNX 변환.

    반환: {"ok":bool, "converted":bool, "msg":str, "onnx":str|None}
      · converted=True  → 이번에 .pt→.onnx 로 변환함
      · converted=False, ok=True → 변환 불필요(이미 onnx 있음 / .pt 없음)
      · ok=False → .pt 는 있으나 변환 불가(torch/onnx 미설치 등) — msg 로 안내
    변환은 torch 가 있는 환경(내 PC·풀 파이썬 패키지)에서만 가능. torch 없는 exe
    에서는 ok=False 로 '풀 패키지 PC 에서 변환하세요' 안내를 돌려준다.
    """
    art = Path(art)
    try:
        from src.onnx_backend_forecast import OnnxForecastModel
        if OnnxForecastModel.available(art):
            return {"ok": True, "converted": False, "msg": "이미 ONNX 모델 있음", "onnx": None}
    except Exception:                                        # noqa: BLE001
        pass
    pt = _find_pt(art)
    if pt is None:
        return {"ok": True, "converted": False, "msg": ".pt 없음", "onnx": None}
    # 변환 필수 패키지 확인(torch·onnx·onnxscript). exe(torch 없음)면 여기서 멈춤.
    import importlib.util as _ilu
    miss = [m for m in ("torch", "onnx", "onnxscript") if _ilu.find_spec(m) is None]
    if miss:
        return {"ok": False, "converted": False, "onnx": None,
                "msg": (f".pt 자동변환 불가 — 이 환경에 {', '.join(miss)} 가 없습니다. "
                        f"torch 가 설치된 PC(풀 파이썬 패키지)에서 변환 후 .onnx 를 넣으세요.")}
    try:
        from src.export_onnx_forecast import export
        with _PLOT_LOCK:                                     # 변환은 직렬화(안전)
            onnx_path = export(pt_path=str(pt), out_dir=str(art))
        # ── 변환 결과가 실제 로드 가능한지 확인한 뒤에만 .pt 삭제(안전) ──
        deleted = []
        keep_pt = bool(_load_app_config().get("keep_pt_after_convert", False))
        if not keep_pt and OnnxForecastModel.available(art):
            for p in sorted(art.glob("*.pt")):
                try:
                    p.unlink()
                    deleted.append(p.name)
                except Exception:                            # noqa: BLE001
                    traceback.print_exc()
        msg = f".pt→ONNX 자동변환 완료: {Path(onnx_path).name}"
        if deleted:
            msg += f" · 원본 삭제: {', '.join(deleted)}"
        return {"ok": True, "converted": True, "onnx": str(onnx_path),
                "deleted": deleted, "msg": msg}
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        return {"ok": False, "converted": False, "onnx": None,
                "msg": f".pt→ONNX 변환 실패: {e}"}


def _model_storage_root():
    """모델 루트 추가 지정(환경변수 IO_MODEL_ROOT 전용 — 배포판은 UI 없음)."""
    v = _STORAGE.get("model") or os.environ.get("IO_MODEL_ROOT")
    if not v:
        return None
    try:
        p = Path(str(v)).expanduser()
        if not p.is_absolute():
            p = _WRITABLE / p
        return p
    except Exception:                                    # noqa: BLE001
        return None


def _model_roots() -> List[Path]:
    r"""'모델'(models) 루트 후보 폴더들 — 저장소 지정 > exe 옆 · 상위 · 현재 폴더.
    exe 는 하위폴더(웨이퍼품질검사시스템\)에 생성되므로, 설치 bat 옆(상위)에 둔
    모델\ 도 찾도록 여러 위치를 본다."""
    roots0 = []
    _ms = _model_storage_root()
    if _ms is not None:
        roots0.append(_ms)                                # 저장소 지정 모델 루트(최우선)
    bases = [_WRITABLE, _WRITABLE.parent]
    try:
        bases.append(Path.cwd())
    except Exception:                                    # noqa: BLE001
        pass
    roots, seen = [], set()
    for d in roots0:
        k = str(d)
        if k not in seen:
            seen.add(k); roots.append(d)
    for base in bases:
        for name in ("모델", "models", "model"):
            d = base / name
            k = str(d)
            if k not in seen:
                seen.add(k); roots.append(d)
    return roots


def _submodel_dirs() -> List[Path]:
    """모델 루트들 안의 '하위 폴더별 모델'(예: 모델\\0713, 모델\\0716). 이름 오름차순."""
    subs = []
    for root in _model_roots():
        if root.exists() and root.is_dir():
            for sub in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if sub.is_dir() and _has_model(sub):
                    subs.append(sub)
    return subs


def _resolve_art() -> Path:
    r"""모델 폴더를 매번 새로 탐색.
    우선순위: ① 지정한 '기본 모델'(설정)  ② 환경변수 IO_ARTIFACTS
             ③ 모델\ · model\ · artifacts\ (exe옆·상위·현재)  ④ 모델\<하위폴더>(최신)  ⑤ 번들.
    """
    try:
        _dm = _load_app_config().get("default_model")
    except Exception:                                    # noqa: BLE001
        _dm = None
    if _dm == "__bundled__":
        return PKG / "artifacts"
    if _dm and _has_model(_path_load(_dm)):
        return _path_load(_dm)
    cands = []
    if os.environ.get("IO_ARTIFACTS"):
        cands.append(Path(os.environ["IO_ARTIFACTS"]))
    _ms = _model_storage_root()
    if _ms is not None:
        cands.append(_ms)                                 # 저장소 지정 모델 루트(직접 모델)
    for base in (_WRITABLE, _WRITABLE.parent):           # 모델\ 직접(exe옆·상위)
        cands += [base / "모델", base / "model", base / "artifacts"]
    for d in cands:
        if _has_model(d):
            return d
    subs = _submodel_dirs()                              # 모델\<하위폴더> 중 최신(이름 마지막) 자동 로드
    if subs:
        return subs[-1]
    return PKG / "artifacts"                             # 번들 내장(기본)


ART = _resolve_art()
# 검사 대상 폴더(데이터가 들어오는 곳). 환경변수 IO_INBOX → 기본값.
#   실행 중 대시보드에서 폴더를 등록하면 data/app_config.json 에 저장되어 그 폴더를 쓴다.
INBOX_DEFAULT = Path(os.environ.get("IO_INBOX") or (_WRITABLE / "데이터" / "입력"))
INBOX = INBOX_DEFAULT                                # 하위호환(초기값). 실사용은 _inbox() 참조.
# 리포트 출력 폴더:  리포트/<모델>/<wafer>/*.png · 리포트/<모델>/predictions.csv
REPORT_DIR = _storage_dir("report", _WRITABLE / "리포트")
PRED_CSV = REPORT_DIR / "predictions.csv"
# ── 이력·백업·자동감시 (신규 기능) ──  (고객이 보는 폴더는 한국어)
DATA_DIR = _storage_dir("data", _WRITABLE / "데이터")
def _default_backup_dir() -> Path:
    """백업 기본 위치 — 데이터 폴더와 '별도'의 최상위 폴더(사용자 결정).

    예전 기본은 '데이터/백업' 이라 데이터 폴더 안에 섞여 있었다. 이제 기본은
    앱 폴더 바로 아래 '백업' 으로 데이터와 나란히 둔다.
    ★ 호환: 명시 설정이 없고, 옛 위치(데이터/백업)에 백업이 이미 쌓여 있으며
      새 위치가 아직 없으면 옛 위치를 계속 쓴다 — 기존 설치에서 백업이
      갑자기 안 보이는 일을 막는다(새 설치만 새 기본을 탄다).
    """
    new = _WRITABLE / "백업"
    legacy = DATA_DIR / "백업"
    try:
        if not new.exists() and legacy.is_dir() and any(legacy.iterdir()):
            print(f"[storage] 백업: 옛 위치 사용(호환) — {legacy} "
                  f"(기존 백업 위치 유지)", flush=True)
            return legacy
    except Exception:                                    # noqa: BLE001
        pass
    return new


BACKUP_DIR = _storage_dir("backup", _default_backup_dir(), "IO_BACKUP")   # 일 마감 파일 보관
AUTOPILOT_STATE = DATA_DIR / "autopilot.json"       # 자동감시 상태(현재 날짜 등) 영속화
ARCHIVE_DIR = _storage_dir("archive", _WRITABLE / "보관")   # 웨이퍼 최소데이터(이미지 아님) → 필요시 렌더
LOG_DIR = _storage_dir("log", _WRITABLE / "로그", "IO_LOG_DIR")        # 앱 동작 로그(콘솔 대체)
ARCHIVE_RETENTION_DAYS = int(os.environ.get("IO_ARCHIVE_DAYS", "0"))    # 기본=0=무제한(영구 보관). 양수면 그 일수
RETENTION_DAYS = int(os.environ.get("IO_RETENTION_DAYS", "60"))   # 백업·로그 보관 일수(기본 60일)
# 챔버별 데이터 경로 기본값 — 설비 원본 .log 가 챔버 폴더로 나뉘어 쌓인다.
CHAMBER_PATH_DEFAULTS = {f"dsv{i}": rf"D:\smarteq_logs\DSV{i}" for i in range(1, 7)}
WATCH_INTERVAL = float(os.environ.get("IO_WATCH_INTERVAL", "5"))  # inbox 폴링 주기(초)
STABLE_SECS = float(os.environ.get("IO_STABLE_SECS", "3"))        # 파일이 '완성'됐다고 볼 정지시간(초)
AUTO_START = os.environ.get("IO_AUTOWATCH", "1") != "0"           # 자동감시 기본 ON
SKIP_STEPS = int(os.environ.get("IO_SKIP_STEPS", "1"))           # 앞 N개 스텝 제외(1=0번 제외)
LOAD_MAX_RETRIES = int(os.environ.get("IO_LOAD_RETRIES", "6"))   # 정렬(_load_aligned) 실패 시 재시도 횟수(초과 시 스킵)
# 없으면 생성(사용자가 CSV 를 넣는 곳). ★ 네트워크 경로가 죽어 있어도 앱 시작이 멈추지 않도록
# 시간 제한을 둔다 — 이 한 줄 때문에 앱이 아예 안 뜨는 상황을 막는다.
_ib_ok, _ib_err = _probe_writable(INBOX)
if not _ib_ok:
    print(f"[storage] 입력 폴더를 준비하지 못했습니다: {INBOX} ({_ib_err})", flush=True)
    _note_stale_path("inbox", str(INBOX))
# ★ 배포판: 설정 파일을 '앱 폴더'에 둔다(데이터 폴더와 분리).
#   예전엔 데이터 폴더 안이라, 데이터 저장소를 외부로 옮기면 다른 PC 의 설정
#   (기본 모델·자동검사 상태 등)이 딸려 오는 사고가 났다. 설정은 이 PC 의 것.
#   이전(1회): 새 위치에 없고 옛 위치(데이터\app_config.json)에 있으면 복사.
APP_CONFIG_FILE = _WRITABLE / "app_config.json"
try:
    _legacy_cfg = DATA_DIR / "app_config.json"
    if not APP_CONFIG_FILE.exists() and _legacy_cfg.exists():
        shutil.copy2(_legacy_cfg, APP_CONFIG_FILE)
        print(f"[config] 설정 파일 이전: {_legacy_cfg} → {APP_CONFIG_FILE}", flush=True)
except Exception as _e:                                   # noqa: BLE001
    print(f"[config] 설정 이전 실패(기본값으로 시작): {_e}", flush=True)


def _is_foreign_abs(raw: str) -> bool:
    """이 OS 가 절대경로로 인식 못 하는 '다른 OS 의 절대경로'인가.
    (리눅스에서 'D:\\logs' 를 상대경로로 오인해 엉뚱한 폴더를 만들던 사고 방지)"""
    r = str(raw or "").strip()
    return bool(re.match(r"^[A-Za-z]:[\\/]", r)) or r.startswith("\\\\") or r.startswith("/")


def _path_store(pstr):
    """경로 저장용: exe(쓰기루트) 내부면 상대경로로 저장 → 폴더째 옮겨도 유지."""
    try:
        pp = Path(pstr).resolve()
        root = Path(_WRITABLE).resolve()
        return str(pp.relative_to(root))
    except Exception:                                    # noqa: BLE001
        return str(pstr)


def _path_load(pstr):
    """경로 로드용: 상대경로면 exe(쓰기루트) 기준으로 복원.
    단 다른 OS 의 절대경로(D:\\.. , \\\\서버\\..)는 '이 PC 에서 못 쓰는 경로'이므로
    쓰기루트에 이어붙이지 않는다(그러면 이름이 이상한 폴더가 생겨 버린다)."""
    raw = str(pstr)
    try:
        pp = Path(raw)
        if pp.is_absolute() or _is_foreign_abs(raw):
            return pp
        return Path(_WRITABLE) / pp
    except Exception:                                    # noqa: BLE001
        return Path(raw)


def _load_app_config() -> dict:
    try:
        import json as _j
        return _j.loads(APP_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}


def _save_app_config(cfg: dict):
    try:
        import json as _j
        APP_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        APP_CONFIG_FILE.write_text(_j.dumps(cfg, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass


def _inbox() -> Path:
    """현재 검사 대상 폴더(대시보드에서 등록했으면 그 폴더, 아니면 기본)."""
    d = _load_app_config().get("inbox")
    if d:
        p = _path_load(d)                                # 앱 폴더 안이면 상대→절대 복원
        if p.exists() and p.is_dir():
            return p
        _note_stale_path("inbox", str(d))                # 이전 PC 경로 등 → 기본값 사용
    return INBOX_DEFAULT


def _set_inbox(path: str):
    cfg = _load_app_config()
    cfg["inbox"] = _path_store(path)                     # 앱 폴더 안이면 상대 저장(이동 내성)
    _save_app_config(cfg)


# ── 챔버별 데이터 경로 (DSV1~6) ─────────────────────────────────────
def _chamber_paths() -> Dict:
    """챔버→폴더 경로. 설정에 없으면 기본(D:\\smarteq_logs\\DSVn)."""
    cfg = _load_app_config().get("chamber_paths") or {}
    out = {}
    for ch in CHAMBERS:
        v = cfg.get(ch)
        out[ch] = (str(_path_load(v)) if v else "") if ch in cfg else str(CHAMBER_PATH_DEFAULTS.get(ch, ""))
    return out


def _set_chamber_paths(paths: Dict, wuk380b=None):
    cfg = _load_app_config()
    cur = cfg.get("chamber_paths") or {}
    for ch in CHAMBERS:
        if ch in paths:
            value = str(paths[ch] or "").strip().strip('"')
            cur[ch] = _path_store(value) if value else ""
    cfg["chamber_paths"] = cur
    if wuk380b is not None:
        cfg["wuk380b_mapping"] = bool(wuk380b)
    _save_app_config(cfg)
    if _load_app_config() != cfg:
        raise HTTPException(500, "설정 파일 저장에 실패했습니다.")


#   챔버 경로(DSV1~6)는 '지금 쌓이는' 폴더라 6칸으로 고정인데, 지난 이력의 LOT 을
#   채우려면 예전에 검사했던 데이터셋 폴더들을 따로 알려줘야 한다. 개수 제한 없이 등록한다.
def _active_chamber_dirs() -> List:
    """실제 존재하는 챔버 폴더만 (챔버, Path) 목록으로."""
    out = []
    for ch, p in _chamber_paths().items():
        if p:
            pp = Path(p)
            if _exists_bounded(pp):
                out.append((ch, pp))
            elif (_load_app_config().get("chamber_paths") or {}).get(ch):
                # 설정에 등록돼 있는데 이 PC 에는 없는 폴더(예: 이전 PC 의 D:\...) → 경고용 기록
                _note_stale_path(f"chamber:{ch}", str(p))
    return out


def _scan_targets() -> List:
    """검사/감시 대상 폴더 목록 [(chamber|None, Path)].

    ① 존재하는 챔버 폴더(DSV1~6)  +  ② 기본 입력 폴더(데이터/입력) 를 모두 포함.
    즉 챔버 폴더를 지정하지 않아도 데이터/입력 은 항상 인식되고,
    챔버 폴더가 있으면 그 폴더도 함께 인식(파일 출처 폴더로 챔버 귀속)."""
    out = _active_chamber_dirs()
    ib = _inbox()
    if all(str(Path(p)) != str(ib) for _, p in out):     # inbox 가 챔버경로와 겹치지 않으면 추가
        out.append((None, ib))
    return out


# ── 파일명에서 '종료 시각' 파싱 → 이력 timestamp 로 사용 ────────────
#   형식: <날짜+끝시>_<시작시각>_<로트>_<끝시각...>
#   예1: 2026012619_183508_3_1913756
#        → 첫토큰 2026012619 = 2026-01-26 + 끝시(19시), 마지막토큰 1913756 → 19:13
#        → 종료시각 = 2026-01-26 19:13
#   예2: WUK720A_2026121618_1881148_19_184300
#        → 날짜 20261216 + 끝시(18), 마지막토큰 184300 → 18:43:00
def _parse_end_time(wafer_id):
    """파일명(스템)에서 검사 '종료시각'을 datetime 으로. 실패시 None.

    · 날짜(+끝시) = 앞 8자리(YYYYMMDD)(+2자리 시) 토큰
    · 끝 시:분(:초) = 뒤에서부터 'HHMM(SS)' 로 유효 해석되는 첫 토큰(시<24·분<60)
      (마지막 토큰이 로트번호 등이라 4자리 미만이면 그 앞 토큰으로)
    """
    toks = [t for t in re.split(r"[_\-.\s]+", str(wafer_id or "")) if t]
    if not toks:
        return None
    date8 = end_hour = None
    for t in toks:                                        # 앞 8자리 YYYYMMDD (+2자리 끝시)
        m = re.match(r"(\d{8})(\d{2})?", t)
        if m:
            date8 = m.group(1)
            end_hour = m.group(2)
            break
    if not date8:
        return None
    hh = mm = ss = None
    for t in reversed(toks):                              # 뒤에서부터 HHMM(SS) 유효한 토큰
        digs = re.sub(r"\D", "", t)
        # ★ 날짜+시각이 한 토큰(YYYYMMDDHHMMSS)인 파일명: 연도 '2026'을 20:26 으로
        #   오독해 모든 웨이퍼가 20:26 로 찍히던 버그 — 날짜 8자리를 떼고 나머지를 시각으로.
        if date8 and digs.startswith(date8):
            digs = digs[len(date8):]
            if not digs:
                continue
        if len(digs) >= 4:
            h, mi = int(digs[0:2]), int(digs[2:4])
            if h < 24 and mi < 60:
                hh, mm = h, mi
                if len(digs) >= 6 and int(digs[4:6]) < 60:
                    ss = int(digs[4:6])
                break
    if hh is None:                                        # 폴백: 날짜토큰의 끝시(hour)만
        if end_hour is None:
            return None
        hh, mm = int(end_hour), 0
    try:
        return dt.datetime.strptime(f"{date8}{hh:02d}{mm:02d}{ss or 0:02d}",
                                    "%Y%m%d%H%M%S")
    except Exception:                                     # noqa: BLE001
        return None


def _end_ts_and_date(wafer_id):
    """종료시각 (isoformat ts, YYYY-MM-DD).
    우선순위: ① 데이터 파일 A/B 마지막행(file_meta) → ② 파일명 파싱 → ③ (now, today)."""
    ets = _file_meta(wafer_id).get("end_ts")              # 데이터(A열 날짜+B열 시간) 우선
    if ets:
        try:
            d = dt.datetime.fromisoformat(ets)
            return d.isoformat(timespec="seconds"), d.strftime("%Y-%m-%d")
        except Exception:                                 # noqa: BLE001
            pass
    d = _parse_end_time(wafer_id)
    if d is None:
        return _now(), _today()
    return d.isoformat(timespec="seconds"), d.strftime("%Y-%m-%d")


# ── 레시피 태그: .log 특정 셀에서 추출한 파일별 레시피(풀네임) ──────
#   설비 원본 .log 의 지정 셀(기본 J2 = 2행·10열)에 레시피 풀네임(29aece39...)이 있다.
#   파일별로 다르므로 stem→recipe 로 태깅해 검사/이력/리포트가 실제 레시피명을 쓴다.
_RECIPE_TAGS: Dict[str, str] = {}
_RECIPE_TAGS_LOCK = threading.Lock()


def _recipe_tags_file() -> Path:
    return DATA_DIR / "recipe_tags.json"


def _load_recipe_tags():
    global _RECIPE_TAGS
    try:
        _RECIPE_TAGS = json.loads(_recipe_tags_file().read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        _RECIPE_TAGS = {}


def _clean_recipe(s):
    """레시피 문자열에서 따옴표·공백·BOM 등 잡문자 제거.
    예: 엑셀/설비가 붙이는 앞따옴표  '29aece39  →  29aece39  (접두 매칭이 깨지지 않게)."""
    if s is None:
        return s
    return str(s).strip().strip("'\"`‘’“”﻿ \t").strip()


def _save_recipe_tag(stem: str, recipe: str):
    recipe = _clean_recipe(recipe)
    if not stem or not recipe:
        return
    with _RECIPE_TAGS_LOCK:
        _RECIPE_TAGS[str(stem)] = str(recipe)
        try:
            f = _recipe_tags_file()
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(_RECIPE_TAGS, ensure_ascii=False, indent=0),
                         encoding="utf-8")
        except Exception:                                 # noqa: BLE001
            pass


def _recipe_tag(stem: str):
    """이 파일의 .log 에서 뽑은 레시피(있으면). 없으면 None. (따옴표 등 정리)"""
    if not _RECIPE_TAGS:
        _load_recipe_tags()
    return _clean_recipe(_RECIPE_TAGS.get(str(stem)))


# ── 파일 메타: .log/.csv 에서 뽑은 파일별 종료시각·챔버(파일명 파싱 대신 데이터 우선) ──
#   · 종료시각 = 데이터 '맨 마지막 행'의 A열(날짜)+B열(시간)   (요구: 파일명 아님)
#   · 챔버     = .log 의 C2(2행 3열, 예 'PMCUNIT11' 앞따옴표 무시) → dsvN
_FILE_META: Dict[str, Dict] = {}
_FILE_META_LOCK = threading.Lock()


def _file_meta_path() -> Path:
    return DATA_DIR / "file_meta.json"


def _load_file_meta():
    global _FILE_META
    try:
        _FILE_META = json.loads(_file_meta_path().read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        _FILE_META = {}


def _file_meta_fresh(src: Path, fm: Dict) -> bool:
    """★ 태그 캐시 유효성 — 같은 파일명으로 다시 내보낸(내용이 바뀐) 파일을 옛 종료시각·챔버로
    쓰던 문제. 항목의 mt(mtime_ns)가 현재 mtime 과 같아야 유효. 구형 항목(mt 없음)은 1회 재태깅."""
    try:
        st = src.stat()
    except Exception:                                     # noqa: BLE001
        return True
    mt = fm.get("mt")
    if mt is None:
        return False
    try:
        return int(mt) == int(st.st_mtime_ns)
    except Exception:                                     # noqa: BLE001
        return False


def _save_file_meta(stem: str, **kv):
    kv = {k: v for k, v in kv.items() if v is not None}
    if not stem or not kv:
        return
    with _FILE_META_LOCK:
        if not _FILE_META:
            _load_file_meta()
        cur = dict(_FILE_META.get(str(stem), {}))
        cur.update(kv)
        _FILE_META[str(stem)] = cur
        try:
            f = _file_meta_path(); f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(_FILE_META, ensure_ascii=False, indent=0), encoding="utf-8")
        except Exception:                                 # noqa: BLE001
            pass


_FILE_META_ERR: Dict[str, str] = {}                       # 마지막 저장 실패 사유(UI 노출용)


def _flush_file_meta() -> Optional[str]:
    """file_meta.json 을 '1회' 저장(대량 태깅 배치용 — 파일마다 안 쓰고 끝에 한 번).

    ★ 예전엔 실패를 조용히 삼켰다(except: pass). 그래서 데이터 폴더에 쓸 수 없는
      상황이면 LOT 을 다 읽고도 아무것도 저장되지 않는데 화면에는 '완료' 만 떴다.
      이제 사유를 남기고 콘솔에도 찍는다. 실패 시 사유 문자열을 반환한다.
    """
    with _FILE_META_LOCK:
        try:
            f = _file_meta_path(); f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.name + ".tmp")            # 원자적 교체(중간에 끊겨도 손상 없음)
            tmp.write_text(json.dumps(_FILE_META, ensure_ascii=False, indent=0), encoding="utf-8")
            os.replace(tmp, f)
            _FILE_META_ERR.pop("last", None)
            return None
        except Exception as e:                            # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if _FILE_META_ERR.get("last") != msg:         # 같은 사유를 반복해 찍지 않는다
                print(f"[meta] ★ file_meta.json 저장 실패 → {_file_meta_path()}\n       {msg}", flush=True)
            _FILE_META_ERR["last"] = msg
            return msg


def _flush_recipe_tags():
    """recipe_tags.json 을 '1회' 저장(대량 태깅 배치용)."""
    with _RECIPE_TAGS_LOCK:
        try:
            f = _recipe_tags_file(); f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(_RECIPE_TAGS, ensure_ascii=False, indent=0), encoding="utf-8")
        except Exception:                                 # noqa: BLE001
            pass


def _file_meta(stem: str) -> Dict:
    if not _FILE_META:
        _load_file_meta()
    return _FILE_META.get(str(stem), {}) or {}


def _chamber_key(r) -> str:
    """이력 1건의 챔버 키 — 항상 소문자 dsvN 형태로 통일한다.

    ★ 왜 필요한가 — 같은 챔버가 기록에 따라 'DSV1' / 'dsv1' 로 섞여 들어온다.
      챔버별 불량률 추이는 dsv1~6 목록과 문자열이 정확히 같아야 칸에 들어가는데,
      대문자면 어디에도 안 들어가 격자에서 통째로 사라졌다. 요약·산점도에는 나오므로
      "산점도엔 웨이퍼가 많은데 추이에는 표기가 안 된다"로 보였다.
      비교하는 곳마다 제각각 처리하지 말고 여기 한 곳에서 정한다.
    """
    ch = (_chamber(r.get("wafer_id")) if _is_wuk380b(r.get("wafer_id")) or _chamber_mapping_stale(r.get("wafer_id"))
          else r.get("chamber") or _chamber(r.get("wafer_id")))
    return str(ch or "").strip().lower()


# Routing is fixed for the process lifetime so an active batch cannot mix models.
_WUK380B_OVERRIDE = _load_app_config().get("wuk380b_mapping") is True


def _is_wuk380b(value) -> bool:
    # Exact equipment token in the filename, including legacy prefixed names.
    return _WUK380B_OVERRIDE or bool(re.search(r"(?i)(?<![a-z0-9])WUK380B(?![a-z0-9])", str(value or "")))


def _chamber_cell_tag(value) -> str:
    if _WUK380B_OVERRIDE:
        return "C2:WUK380B-app-v1"
    return "C2:WUK380B-even-v1" if _is_wuk380b(value) else "C2"


def _chamber_mapping_stale(wafer_id):
    return _file_meta(wafer_id).get("chamber_cell", "C2") != _chamber_cell_tag(wafer_id)


def _chamber_from_token(tok, equipment=None):
    """'PMCUNIT11'(앞따옴표 무시)·'DSV5' 등 → dsvN. 숫자/무관값이면 None."""
    s = _clean_recipe(tok) or ""
    m = re.search(r"(?i)dsv\s*_?([1-6])(?!\d)", s)
    if m:
        return f"dsv{m.group(1)}"
    m = re.search(r"(?i)PMC[A-Z_\s]*?(\d+)(?!\d)", s)             # PMCUNIT7~12 → dsv1~6
    if m:
        n = int(m.group(1))
        if _is_wuk380b(equipment):
            return f"dsv{n // 2}" if n in (2, 4, 6, 8, 10, 12) else None
        if 7 <= n <= 12:
            return f"dsv{n - 6}"
    return None


def _extract_chamber_from_log(src: Path):
    """.log(탭)의 C2(2행 3열)에서 챔버 추출 → dsvN. 실패시 None."""
    try:
        with open(src, "rb") as f:
            bom = f.read(4)
    except OSError:
        return None
    encodings = ("utf-16",) if bom.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "cp949")
    for enc in encodings:
        try:
            with open(src, encoding=enc) as f:
                row2 = None
                for i, line in enumerate(f):
                    if i == 1:
                        row2 = line; break
            if row2 is None:
                return None
            cells = next(csv.reader([row2.rstrip("\r\n")], delimiter="\t"))
            return _chamber_from_token(cells[2], src.stem) if len(cells) >= 3 else None
        except UnicodeDecodeError:
            continue
        except Exception:                                 # noqa: BLE001
            return None
    return None


def _lot_cell() -> tuple:
    """LOT 셀 위치 (row, col) 1-indexed. 기본 (2, 9) = I2. 설정으로 조정 가능.

    ★ 처음엔 G2(7열)로 알고 있었으나 실제로는 I2(9열)였다(사용자 확인).
      레시피 J2(10열) 바로 왼쪽 칸이다. 설비가 달라 또 바뀔 수 있으므로
      app_config 의 lot_log_row/lot_log_col 로 화면에서 바꿀 수 있게 해 두었다.
    """
    cfg = _load_app_config()
    try:
        r = int(cfg.get("lot_log_row", 2))
        c = int(cfg.get("lot_log_col", 9))
    except Exception:                                     # noqa: BLE001
        r, c = 2, 9
    return max(1, r), max(1, c)


def _lot_cell_tag() -> str:
    """현재 LOT 셀 표시 — 'R2C9'. '어느 칸으로 읽었는지' 를 기록에 남겨,
    칸 설정이 바뀌면 그 파일들을 자동으로 다시 읽게 하는 근거로 쓴다."""
    return "R%dC%d" % _lot_cell()


def _lot_cell_name() -> str:
    """사람이 읽는 셀 이름 — 'I2'. 27열(AA) 이상도 처리."""
    r, c = _lot_cell()
    name = ""
    while c > 0:
        c, rem = divmod(c - 1, 26)
        name = chr(65 + rem) + name
    return f"{name}{r}"


def _clean_lot(s):
    """LOT 값 정리 — 앞따옴표·공백 제거. 값이 없거나 자리표시자면 None.

    설비가 넣는 해시(7c218394…)를 그대로 쓴다. 같은 문자열이면 같은 LOT.
    대소문자만 다른 경우가 섞이면 다른 LOT 으로 갈리므로 소문자로 통일한다.
    """
    v = _clean_recipe(s) or ""
    if not v or v.lower() in ("-", "n/a", "na", "null", "none", "nan", "0"):
        return None
    return v.lower()


def _extract_lot_from_log(src: Path):
    """.log(탭)의 I2(2행 9열)에서 LOT 해시를 추출. 실패시 None.

    ★ 왜 I2 인가: 설비 원본 .log 의 이 칸에 LOT 해시가 들어 있다(사용자 확인).
      같은 해시 = 같은 LOT 이라, 산포 추세를 시간 간격이 아니라 LOT 단위로 묶는 근거가 된다.
    """
    row, col = _lot_cell()
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"):
        try:
            with open(src, encoding=enc) as f:
                line = None
                for i, ln in enumerate(f):
                    if i == row - 1:
                        line = ln; break
            if line is None:
                return None
            cells = line.rstrip("\r\n").split("\t")
            return _clean_lot(cells[col - 1]) if col - 1 < len(cells) else None
        except UnicodeDecodeError:
            continue
        except Exception:                                 # noqa: BLE001
            return None
    return None


def _lot_of(wafer_id, row=None):
    """웨이퍼의 LOT — 검사 기록에 박힌 값 우선, 없으면 파일 태깅 캐시(I2)."""
    if row is not None:
        v = _clean_lot(row.get("lot"))
        if v:
            return v
    return _clean_lot(_file_meta(str(wafer_id)).get("lot"))


def _parse_date_time(date_s, time_s):
    """A열 날짜 + B열 시간 → datetime(여러 포맷 대비). 실패시 None."""
    d = (_clean_recipe(date_s) or "").strip()
    t = (_clean_recipe(time_s) or "").strip()
    if not d:
        return None
    if t:                                                 # 소수초(밀리초) 제거: 09:15:00.123 → 09:15:00
        t = t.split(".")[0].split(",")[0].strip()
    combo = (d + " " + t).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M",
                "%Y%m%d %H%M%S", "%Y%m%d %H:%M:%S",
                "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
                "%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %I:%M %p"):
        try:
            return dt.datetime.strptime(combo, fmt)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d"):   # 날짜만(시각 소실 → 자정)
        try:
            return dt.datetime.strptime(d, fmt)
        except ValueError:
            continue
    return None


def _read_tail_lines(src: Path, n: int = 64, blocksize: int = 65536) -> List[str]:
    """파일 '끝부분'만 읽어 마지막 n개 비어있지 않은 줄 반환. 대형 파일 전체읽기 방지(준비중 가속).

    끝에서부터 블록 단위로 뒤로 읽어 줄이 충분히 모이면 멈춘다(수십MB 로그도 수십KB만 읽음)."""
    try:
        with open(src, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            data = b""
            while size > 0 and data.count(b"\n") <= n + 1:
                step = min(blocksize, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
    except Exception:                                     # noqa: BLE001
        return []
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return []
    return [ln for ln in text.splitlines() if ln.strip()][-n:]


def _extract_end_dt_from_file(src: Path):
    """데이터 파일의 마지막 데이터행 A(날짜)+B(시간) → 종료시각. (끝 64줄만 읽어 빠름)

    맨 끝 행의 시간셀이 비어있거나(요약행 등) 자정으로 파싱되면, 위쪽 행으로 거슬러
    올라가며 '실제 시각(자정 아님)'을 가진 마지막 행을 찾는다.
    """
    sep = "\t" if src.suffix.lower() in (".log", ".tsv") else ","
    try:
        lines = _read_tail_lines(src, n=64)               # ★ 전체읽기 대신 끝부분만
        if not lines:
            return None
        fallback = None
        for line in reversed(lines):
            cells = line.split(sep)
            if len(cells) < 2:
                continue
            dtv = _parse_date_time(cells[0], cells[1])
            if dtv is None:
                continue
            if fallback is None:
                fallback = dtv
            if (dtv.hour, dtv.minute, dtv.second) != (0, 0, 0):
                return dtv
        return fallback
    except Exception:                                     # noqa: BLE001
        return None


def _tag_file_meta(folder: Path, stem: str, force: bool = False, source_path: Path = None):
    """폴더의 stem 파일에서 종료시각(A/B 마지막행)+챔버(.log C2)를 뽑아 저장(있을 때만)."""
    try:
        _meta = _file_meta(stem)
        _prev = _meta.get("end_ts")
        # 이미 태깅됐어도, 자정(T00:00:00) 값은 옛 버전/요약행 탓일 수 있어 '한 번' 재추출한다.
        #   단 새 추출기로 이미 확정(ets_v>=2)했으면 자정이라도 재추출 안 함 → 반복 IO 방지(대량검사).
        _stale = (bool(_prev) and str(_prev).endswith("T00:00:00")
                  and int(_meta.get("ets_v", 0) or 0) < 2)
        if not force and _prev and not _stale and _meta.get("chamber_cell") == _chamber_cell_tag(stem):
            return                                        # 정상 태깅됨(중복 IO 방지)
        log, tsv, csv = (folder / f"{stem}.log", folder / f"{stem}.tsv", folder / f"{stem}.csv")
        raw = log if log.exists() else (tsv if tsv.exists() else None)
        if source_path is not None:
            source_path = Path(source_path)
            if source_path.is_file():
                if source_path.suffix.lower() in (".log", ".tsv"):
                    raw = source_path
                elif source_path.suffix.lower() == ".csv":
                    csv = source_path
        if raw is None and not csv.exists():
            return  # a guessed/missing path must never erase a chamber read from the real file
        chamber = _extract_chamber_from_log(raw) if raw is not None else (_chamber_from_token(stem, stem) if _is_wuk380b(stem) else None)
        end_dt = _extract_end_dt_from_file(raw) if raw is not None else None
        lot = _extract_lot_from_log(raw) if raw is not None else None
        if end_dt is None and csv.exists():
            end_dt = _extract_end_dt_from_file(csv)       # 원본 .log 없으면 변환/직접 .csv 로
        _save_file_meta(stem,
                        end_ts=(end_dt.isoformat(timespec="seconds") if end_dt else None),
                        chamber=chamber or "",
                        chamber_cell=_chamber_cell_tag(stem),
                        lot=lot,
                        # 값 확정 시 버전마킹(재추출 방지). lot_v 는 'LOT 칸을 한 번 봤다'는 표시라
                        # 값이 비어도 세운다(LOT 칸이 없는 파일을 매번 다시 읽지 않도록).
                        lot_v=(1 if raw is not None else None),
                        lot_c=(_lot_cell_tag() if raw is not None else None),
                        ets_v=(2 if end_dt is not None else None))
    except Exception:                                     # noqa: BLE001
        pass


def _recipe_cell() -> tuple:
    """레시피 셀 위치 (row, col) 1-indexed. 기본 (2, 10) = J2. 설정으로 조정 가능."""
    cfg = _load_app_config()
    try:
        r = int(cfg.get("recipe_log_row", 2))
        c = int(cfg.get("recipe_log_col", 10))
    except Exception:                                     # noqa: BLE001
        r, c = 2, 10
    return max(1, r), max(1, c)


def _extract_recipe_from_log(src: Path):
    """.log(탭 구분)의 지정 셀에서 레시피 풀네임을 추출. 실패시 None.

    기본 셀 = J2(2행 10열). 셀이 비면 2행에서 '레시피처럼 보이는' 최장 토큰으로 폴백.
    """
    row, col = _recipe_cell()
    lines = None
    # 데이터 리더(_read_csv)와 동일한 인코딩 폴백. latin1 은 어떤 바이트든 읽혀 마지막 안전망.
    #   (예전엔 3종만 시도 → 특정 인코딩 파일 1개만 유형 추출 실패해 unknown 되던 문제)
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"):
        try:
            with open(src, encoding=enc) as f:            # 여유있게 앞 몇 줄(맨 위 빈 줄/제목 줄로
                lines = [f.readline() for _ in range(max(row + 1, 3))]   # 행이 밀린 파일 대비)
            break
        except UnicodeDecodeError:
            lines = None
        except Exception:                                 # noqa: BLE001
            return None
    if not lines:
        return None

    def _cells(i):
        return lines[i].rstrip("\r\n").split("\t") if 0 <= i < len(lines) and lines[i] else []

    cells = _cells(row - 1)                              # ★ 지정행(J2 의 행 = 2행) '그 행만' 본다
    # ① 지정 셀(J2) 을 '정확히' 우선 — 값이 있으면 그대로 채택. 다른 칸/다른 행(J1 헤더 등)은 안 본다.
    #    (예전엔 알려진 레시피 스캔이 위아래 행까지 훑어, J1 의 '336…' 을 잡던 문제 → 제거)
    #    글자 포함(풀네임) 또는 3자리+ 숫자코드(336/521/232) 면 레시피로 인정.
    if col - 1 < len(cells):
        v = _clean_recipe(cells[col - 1])
        if v and (re.search(r"[A-Za-z]", v) or re.fullmatch(r"\d{3,}", v)):
            return v

    # ② J2 가 '비었을 때만' 최소 폴백 — 같은 행(2행) 안에서만. 다른 행은 절대 안 본다.
    known = []
    try:
        known += [str(k) for k in (ENGINE.thresholds or {}).keys()]
        known += [str(x) for x in (getattr(ENGINE.cfg, "recipe_names", ()) or ())]
    except Exception:                                     # noqa: BLE001
        pass
    known = [k for k in ((_clean_recipe(x) or "").lower() for x in known) if k]
    if known:
        for cellv in cells:                              # ★ cells = 2행만(다른 행 스캔 안 함)
            cv = _clean_recipe(cellv)
            if not cv:
                continue
            cl = cv.lower()
            if any(cl == k or (len(k) >= 3 and cl.startswith(k))
                   or (len(cl) >= 3 and k.startswith(cl)) for k in known):
                return cv

    # ③ 폴백(J2 가 비었을 때만): '같은 2행'에서 글자 포함 최장 토큰(날짜/시간/순수숫자 제외)
    cand = [c for c in (_clean_recipe(x) for x in cells)
            if c and re.search(r"[A-Za-z]", c)]
    if cand:
        return max(cand, key=len)
    # ④ 실패 → '눈으로 똑같은데 왜 unknown 인지' 확인용 진단 로그(실제 셀 내용 표시)
    _shown = cells[col - 1] if col - 1 < len(cells) else "(그 열 없음)"
    print(f"[recipe] 유형 인식 실패 → unknown: {src.name} | 셀 R{row}C{col}='{_shown}' | "
          f"R{row} 앞 12칸={[c[:16] for c in cells[:12]]} | 그 행 총열수={len(cells)}", flush=True)
    return None


# ── 파일 로깅(콘솔 대체) ─────────────────────────────────────────
#   콘솔창 없는(windowed) exe 에서 print 출력을 날짜별 로그파일로 남긴다.
#   로그 보관기간 = 데이터 보관기간(RETENTION_DAYS). 날짜 바뀌면 새 파일.
class _DailyLog:
    """stdout/stderr 대체: 로그\\app_<날짜>.log 에 append. 날짜 바뀌면 파일 교체."""

    encoding = "utf-8"

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self._day = None
        self._fh = None
        self._lock = threading.Lock()

    def isatty(self):
        return False

    def fileno(self):                                    # 일부 라이브러리가 조회
        raise OSError("no fileno")

    def _ensure(self):
        day = dt.date.today().isoformat()
        if self._fh is None or day != self._day:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                if self._fh:
                    try:
                        self._fh.close()
                    except Exception:            # noqa: BLE001
                        pass
                # ★ 한국어판 Windows 메모장은 BOM 없는 UTF-8 을 ANSI 로 읽어 한글이 깨진다.
                #   새 파일일 때만 BOM 을 붙인다(이어쓰기 파일 중간에 BOM 이 끼면 안 됨).
                _lp = self.dir / f"app_{day}.log"
                _new = not _lp.exists() or _lp.stat().st_size == 0
                self._fh = open(_lp, "a", encoding="utf-8-sig" if _new else "utf-8")
                self._day = day
            except Exception:                    # noqa: BLE001
                self._fh = None

    def write(self, s):
        if not s:
            return
        with self._lock:
            self._ensure()
            if self._fh:
                try:
                    self._fh.write(s)
                    self._fh.flush()
                except Exception:                # noqa: BLE001
                    pass

    def flush(self):
        with self._lock:
            if self._fh:
                try:
                    self._fh.flush()
                except Exception:                # noqa: BLE001
                    pass


_LOGGER_ON = False


def _setup_logging():
    """print 출력을 파일 로그로 보냄(콘솔 없는 exe 대비). 1회만."""
    global _LOGGER_ON
    if _LOGGER_ON:
        return
    try:
        w = _DailyLog(LOG_DIR)
        sys.stdout = w
        sys.stderr = w
        _LOGGER_ON = True
        print(f"\n===== 로그 시작 {_now()} (보관 {RETENTION_DAYS}일) =====")
    except Exception:                            # noqa: BLE001
        pass


def _prune_logs() -> int:
    """보관기간 지난 로그파일(app_<날짜>.log) 삭제. 반환=삭제 수.
    ★ UI '데이터 보관 기간'(archive_retention_days, 0=무제한)을 따른다 — 백업·로그도 동일 정책."""
    days = _archive_retention_days()
    if days <= 0:
        return 0                                          # 무제한 → 삭제 안 함
    cutoff = dt.date.today() - dt.timedelta(days=days)
    removed = 0
    try:
        for f in LOG_DIR.glob("app_*.log"):
            m = re.search(r"app_(\d{4}-\d{2}-\d{2})\.log$", f.name)
            if not m:
                continue
            try:
                d = dt.date.fromisoformat(m.group(1))
            except ValueError:
                continue
            if d < cutoff:
                try:
                    f.unlink()
                    removed += 1
                except Exception:                # noqa: BLE001
                    pass
    except Exception:                            # noqa: BLE001
        pass
    # 검사 캐시(파싱/점수)도 같은 보관기간으로 정리(0=무제한). 지워져도 다음 검사 때 재생성될 뿐.
    cutoff_ts = time.time() - days * 86400
    for cdir in (DATA_DIR / "parse_cache", DATA_DIR / "score_cache"):
        try:
            if not cdir.exists():
                continue
            for f in cdir.glob("*"):
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff_ts:
                        f.unlink()
                        removed += 1
                except Exception:                # noqa: BLE001
                    pass
        except Exception:                        # noqa: BLE001
            pass
    return removed


def _no_window_kwargs() -> dict:
    """Windows 에서 subprocess 실행 시 콘솔창(cmd)이 깜빡이지 않게 하는 옵션."""
    if os.name != "nt":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0                                    # SW_HIDE
    return {"startupinfo": si,
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


def _no_console_but_gui_kwargs() -> dict:
    """콘솔은 숨기되 '창은 띄워야 하는' 자식 프로세스용(폴더 선택 대화상자).

    ★ 왜 따로 두는가
      _no_window_kwargs 는 STARTUPINFO 에 SW_HIDE 를 넣는다. 이 값은 자식이 만드는
      '첫 창' 의 표시 방식으로 쓰여, GUI 창까지 숨거나 뒤로 밀릴 수 있다.
      콘솔을 없애는 데는 CREATE_NO_WINDOW 하나면 충분하다(powershell 은 콘솔 앱이라
      이 플래그만으로 콘솔 자체가 안 만들어진다). 그래서 SW_HIDE 는 빼고 쓴다.
    """
    if os.name != "nt":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


_PICK_ERR: Dict = {"msg": ""}                            # 폴더 선택 창 실패 사유(UI 안내용)


def _pick_folder_dialog(initial: str = "") -> str:
    """Windows 네이티브 '폴더 선택' 창을 띄우고 고른 경로를 반환(취소=빈 문자열).

    로컬 앱이므로 서버=사용자 PC → 창이 사용자 화면에 뜬다. PowerShell 의
    FolderBrowserDialog 사용(추가 패키지 불필요). 비-Windows 에선 빈 문자열.
    """
    if os.name != "nt":
        # ★ 리눅스(우분투 학습앱): zenity 폴더선택 창(GNOME 기본 포함). 없으면 빈 문자열(직접 입력).
        if sys.platform.startswith("linux"):
            try:
                args = ["zenity", "--file-selection", "--directory", "--title=IO 검사 - 폴더 선택"]
                init_l = (initial or "").strip()
                if init_l and Path(init_l).is_dir():
                    args.append(f"--filename={init_l.rstrip('/')}/")
                p = subprocess.run(args, capture_output=True, text=True, timeout=300)
                return (p.stdout or "").strip()
            except Exception as e:                       # noqa: BLE001
                print(f"[pick-folder] zenity 실패: {e}", flush=True)
        return ""
    init = (initial or "").strip()
    if not init:
        init = str(_WRITABLE)                            # ★ 지정 경로 없으면 '앱 실행 위치'에서 시작(요청)
    init = init.replace("'", "''")
    # 콘솔창 깜빡임 방지: powershell 을 창 없이 실행

    # ★ 폴더 선택 창이 앱 뒤에 숨던 문제
    #   예전에도 TopMost 소유자 폼을 만들긴 했는데 '보이게 하지 않아' 아무 효과가 없었다.
    #   윈도우는 백그라운드 프로세스(여기선 powershell)가 포그라운드를 뺏는 것을 막는다.
    #   그래서 소유자 폼을 화면 밖에 1x1 로 '실제로 띄우고' 활성화한 뒤 대화상자를 연다.
    #   TopMost 는 포커스가 아니라 z-순서 속성이라, 포그라운드를 못 뺏어도 그림은
    #   맨 앞에 그려진다 → 가려지지 않는다.
    #   SetForegroundWindow(P/Invoke)까지 쓰면 호출할 때마다 C# 을 컴파일해 1~3초씩
    #   느려지고, 최소 구성 PC 엔 컴파일러가 없을 수도 있어 쓰지 않는다.
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms | Out-Null;"
        "$owner = New-Object System.Windows.Forms.Form;"
        "$owner.TopMost = $true; $owner.ShowInTaskbar = $false;"
        "$owner.FormBorderStyle = 'None'; $owner.StartPosition = 'Manual';"
        # ★ 화면 '안' 에 둔다. 예전엔 -3000,-3000 으로 치웠는데, 윈도우는 화면 밖 창을
        #   제대로 활성화하지 못한다. 소유자가 활성화되지 않으면 그 위에 떠야 할
        #   대화상자도 앞으로 못 나온다. 1x1 이라 화면 구석에 점 하나로만 보인다.
        #   (투명도 0 은 쓰지 않는다 — 레이어드 창이 되어 오히려 활성화가 막힌다)
        "$owner.Left = 0; $owner.Top = 0; $owner.Width = 1; $owner.Height = 1;"
        "$owner.Show(); $owner.Activate(); $owner.BringToFront();"
        "[System.Windows.Forms.Application]::DoEvents();"
        "$f = New-Object System.Windows.Forms.FolderBrowserDialog;"
        "$f.Description = 'IO 검사 - 폴더 선택'; $f.ShowNewFolderButton = $true;"
        + (f"$f.SelectedPath = '{init}';" if init else "")
        + "$r = $f.ShowDialog($owner);"
        "if($r -eq [System.Windows.Forms.DialogResult]::OK){[Console]::Out.Write($f.SelectedPath)};"
        "$owner.Close(); $owner.Dispose();"
    )
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-STA", "-Command", ps],
            # ★ 여기서는 SW_HIDE 를 주지 않는다 — 대화상자까지 숨거나 뒤로 밀린다.
            capture_output=True, text=True, timeout=300, **_no_console_but_gui_kwargs())
        if p.returncode != 0:
            err = (p.stderr or "").strip()[:200]
            print(f"[pick-folder] powershell 실패(코드 {p.returncode}): {err}", flush=True)
            _PICK_ERR["msg"] = ("폴더 선택 창을 띄우지 못했습니다(PowerShell 실행 실패). "
                                "경로를 직접 입력해 주세요.")
            return ""
        _PICK_ERR["msg"] = ""
        return (p.stdout or "").strip()
    except FileNotFoundError:
        # ★ WES7(Windows Embedded Standard 7)은 구성요소를 골라 담는 OS 라
        #   PowerShell 이 아예 빠져 있을 수 있다. 이때 조용히 실패하면 사용자에겐
        #   '버튼을 눌러도 아무 일도 안 일어남'으로 보인다 → 이유를 돌려준다.
        print("[pick-folder] PowerShell 이 없습니다(WES7 최소 구성 가능성)", flush=True)
        _PICK_ERR["msg"] = ("이 PC 에는 PowerShell 이 없어 폴더 선택 창을 띄울 수 없습니다. "
                            "경로를 직접 입력해 주세요.")
        return ""
    except Exception as e:                               # noqa: BLE001
        print(f"[pick-folder] 실패: {e}", flush=True)
        _PICK_ERR["msg"] = f"폴더 선택 창을 띄우지 못했습니다({type(e).__name__}). 경로를 직접 입력해 주세요."
        return ""
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return dt.date.today().isoformat()


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _detect_score_norm(root, val, best) -> str:
    """모델이 점수를 '유효 스텝 수'로 나눠 임계를 잡았는지 판별 → 'none' | 'time' | 'final'.

    train_jw v0.95/0.96 의 --score-time-avg / --score-final-avg. 판별 근거 2가지:
      ① threshold/val 안의 score_time_avg · score_final_avg 불리언
      ② score_kind(=score_name) 접두 'time_avg_' / 'final_avg_'
    (최종 웨이퍼 점수는 두 방식이 같다 — 앱은 '나눠야 하는지'만 알면 된다)
    """
    def _b(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v) if _is_num(v) else False

    cands = []
    for src in (val, best, root if isinstance(root, dict) else {}):
        if isinstance(src, dict):
            cands.append(src)
            for k in ("threshold", "evaluation_criteria"):
                if isinstance(src.get(k), dict):
                    cands.append(src[k])
    for d in cands:
        if _b(d.get("score_final_avg")):
            return "final"
        if _b(d.get("score_time_avg")):
            return "time"
    for d in cands:                                       # 접두 폴백(구 저장 포맷 호환)
        for k in ("score_kind", "score_name"):
            kind = str(d.get(k) or "").strip().lower()
            if kind.startswith("final_avg_"):
                return "final"
            if kind.startswith("time_avg_"):
                return "time"
    return "none"


def _find_threshold(obj):
    """search_best.json 어디에 있든 스칼라 임계값을 찾는다(구조 무관).
    우선순위: threshold_raw > threshold > *_threshold. 재귀 탐색."""
    if isinstance(obj, dict):
        for key in ("threshold_raw", "threshold", "decision_threshold",
                    "global_threshold_raw", "global_threshold", "best_threshold"):
            if _is_num(obj.get(key)) and float(obj[key]) > 0:
                return float(obj[key])
        for v in obj.values():
            r = _find_threshold(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_threshold(v)
            if r is not None:
                return r
    return None


def _find_recipe_thresholds(obj):
    """threshold_by_recipe(레시피→임계 dict) 를 구조 어디에 있든 찾는다."""
    if isinstance(obj, dict):
        v = obj.get("threshold_by_recipe")
        if isinstance(v, dict) and v:
            return v
        for vv in obj.values():
            r = _find_recipe_thresholds(vv)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for vv in obj:
            r = _find_recipe_thresholds(vv)
            if r is not None:
                return r
    return None


# 'threshold' dict 안에서 레시피 임계가 아닌 메타/전역 키(추출 시 제외).
_THR_META_KEYS = {
    "threshold_sigma", "threshold_sigma_mode", "threshold_sigma_requested",
    "threshold", "threshold_raw", "decision_threshold", "best_threshold",
    "global_threshold", "global_threshold_raw", "default", "global", "sigma",
}


def _recipe_thr_map(obj) -> dict:
    """새 모델 포맷: 'threshold' 값이 '레시피→임계' dict 인 경우 레시피별 임계만 추출.

    예) {"29a":1.2,"521":1.0,"336":1.1,"232":1.0,
         "threshold_sigma":3.0,"threshold_sigma_mode":"...","threshold_sigma_requested":3.0}
        → {"29a":1.2,"521":1.0,"336":1.1,"232":1.0}
    (threshold_sigma* 등 메타키·전역키·비수치 값은 제외. 레시피명은 짧게(29a)든 풀네임이든 그대로 키로 둔다.)
    """
    if not isinstance(obj, dict):
        return {}
    return {str(k): float(v) for k, v in obj.items()
            if str(k) not in _THR_META_KEYS and _is_num(v)}


# ════════════════════════════════════════════════════════════════════
#  모델 + 설정 (lazy singleton)
# ════════════════════════════════════════════════════════════════════
class Engine:
    """학습된 recon 모델 + best 설정 + 판정 임계 보관(1회 로드)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.chamber_engines = {}
        self.chamber_artifacts = {}
        self.bundle_id = None
        self.artifact = None
        self.model = None                            # OnnxForecastModel | TCNForecastModel
        self.cfg: Optional[IOConfig] = None
        self.threshold: Optional[float] = None      # 단일/폴백 임계
        self.thresholds: Dict = {}                  # 레시피별 임계 {recipe: thr}
        self.threshold_mode: str = "unknown"        # 'per_recipe' | 'global' | 'unknown'(자율 인식)
        self.score_norm: str = "none"              # 'none'|'time'|'final' — 모델이 점수를 정규화했는지
        self.threshold_warning: Optional[str] = None  # 임계 미설정(0/None) 경고 문구 → UI 배너
        self.wafer_score_override = None
        self._override_in_json = False
        self.params: Dict = {}
        self.error: Optional[str] = None
        self.art: Path = ART                        # 실제 로드한 모델 폴더
        self.selected: Optional[Path] = None        # 드롭다운으로 고른 모델 폴더
        self.backend: str = "-"                     # 'onnx' | 'torch'

    def _build_cfg(self) -> IOConfig:
        """search_best.json 의 임계(+레시피별 임계)를 읽고, io_keep=() 로 전 IO 로드.

        forecast 모델은 학습 때 io_keep=() 였으므로 모든 IO 컬럼을 유지해야 정렬이 맞는다.
        앞 스텝(ProcessStep==0) 제외는 모델이 내부적으로 처리(score_exclude_boundary_step0)
        하므로 여기서 score_skip_steps 를 주입하지 않는다.
        """
        self.wafer_score_override = None
        self._override_in_json = False
        cfg = replace(IOConfig(), io_keep=())
        sb = self.artifact.config if self.artifact else self.art / "search_best.json"
        if not sb.exists():
            sb = self.art / "best_search.json"
        self._override_in_json, self.wafer_score_override = artifact_override(sb.parent, sb, include_standalone=False)
        if sb.exists():
            root = json.loads(sb.read_text(encoding="utf-8"))
            best = root.get("best", {}) if isinstance(root, dict) else {}
            self.params = best.get("params") or {}
            val = best.get("val") or {}
            # 레시피별 임계 — ① threshold_by_recipe(구 포맷) 어디에 있든 찾고,
            #                 ② 새 포맷(val/best 의 'threshold' 가 레시피 dict)도 병합.
            rec_thr = (val.get("threshold_by_recipe")
                       or best.get("threshold_by_recipe")
                       or (root.get("threshold_by_recipe") if isinstance(root, dict) else None)
                       or _find_recipe_thresholds(root))
            self.thresholds = {str(k): float(v) for k, v in (rec_thr or {}).items()
                               if _is_num(v)}
            # ★ v0.96 점수 정규화 감지 — 모델이 '누적 점수 ÷ 유효 스텝 수'로 임계를 산정했으면
            #   앱도 똑같이 나눠야 판정이 맞는다(안 나누면 점수가 ~1300배 커져 전부 NG).
            #   ONNX 메타(npz)엔 이 플래그가 없어 search_best.json 에서 읽는다.
            self.score_norm = _detect_score_norm(root, val, best)
            # 새 포맷 레시피 임계는 '정해진 위치'에서만 읽는다(val/best/root 의 'threshold').
            #   ※ 전체 재귀 탐색은 params.threshold 등 무관한 중첩 dict 를 레시피로 오인(오염)하므로 금지.
            new_map = (_recipe_thr_map(val.get("threshold"))
                       or _recipe_thr_map(best.get("threshold"))
                       or (_recipe_thr_map(root.get("threshold")) if isinstance(root, dict) else {}))
            for k, v in new_map.items():                   # 구 포맷 우선, 없는 레시피만 새 포맷으로 채움
                self.thresholds.setdefault(k, v)
            # 전역 임계 — 표준 경로 → 실패 시 JSON 전체에서 재귀 탐색(구조 달라도 OK).
            #   ★ 새 포맷은 'threshold' 가 dict(레시피별)이므로 스칼라로 쓰면 안 됨(float(dict) 크래시 방지).
            thr = val.get("threshold_raw")
            if thr is None:
                thr = val.get("threshold")
            if isinstance(thr, dict):
                thr = None
            if thr is None:
                thr = best.get("threshold")
                if isinstance(thr, dict):
                    thr = None
            if thr is None:                               # 통합 임계가 dict({"default":x}/{"global":x})로 저장된 경우
                for _src in (val.get("threshold"), best.get("threshold"),
                             root.get("threshold") if isinstance(root, dict) else None):
                    if isinstance(_src, dict):
                        for _k in ("default", "global", "global_threshold_raw",
                                   "global_threshold", "value"):
                            if _is_num(_src.get(_k)):
                                thr = float(_src[_k]); break
                    if thr is not None:
                        break
            if thr is None and self.thresholds:           # 레시피 못 찾을 때 폴백(최대=보수적)
                thr = max(self.thresholds.values())
            if thr is None:                               # ★ 어디에 있든 찾기(구조 무관)
                thr = _find_threshold(root)
            self.threshold = None if thr is None else float(thr)
            # ★ 자율 인식: 레시피별 임계가 있으면 'per_recipe', 없으면 'global'(통합) 모드로 자동 판정.
            self.threshold_mode = "per_recipe" if self.thresholds else "global"
            # ★ 임계 미설정 경고: 레시피별 임계도 없고 global 임계도 0/None 이면 판정 기준이 없음.
            #   → 모든 웨이퍼가 raw≥0 이라 NG 로 뜨고 점수도 정규화 불가 → 결과 신뢰 불가.
            _no_thr = (not self.thresholds) and (
                self.threshold is None or float(self.threshold) <= 1e-12)
            self.threshold_warning = (
                "모델에 판정 임계가 설정되어 있지 않습니다(global·레시피별 모두 없음). "
                "모든 웨이퍼가 불량(NG)으로 표시되고 이상점수를 신뢰할 수 없습니다. "
                "모델 폴더의 search_best.json 에서 threshold 를 확인하세요." if _no_thr else None)
            print(f"[engine] 임계 로드: 모드={self.threshold_mode} · global={self.threshold} · "
                  f"레시피별={len(self.thresholds)}개 {sorted(self.thresholds)}  (search_best.json)",
                  flush=True)
            if self.threshold_warning:
                print("[engine] ⚠ 경고: " + self.threshold_warning, flush=True)
            fields = set(IOConfig().__dataclass_fields__)
            cfg = replace(cfg, **{k: v for k, v in self.params.items()
                                  if k in fields and k != "io_keep"})
        return cfg

    def _sync_cfg_from_model(self):
        """모델 메타가 아는 학습 시 설정을 '데이터 로드용 cfg' 에 전파한다.

        ★ 왜 필요한가 — virtual_sensor_mode(파생 센서)는 데이터 로드 단계
          (add_virtual_sensors)에서 cfg 를 보고 만들어진다. 예전에는 이 값이
          search_best.json 의 best.params 에 우연히 들어 있어야만 켜졌다.
          파생 센서로 학습한 모델인데 params 에 없으면, 파생 입력이 전부 0 으로
          채워져 기대값이 조용히 어긋난다(오류도 안 난다). 메타(.npz)가 학습의
          사실을 기록하므로 메타를 우선 신뢰한다.
        """
        # Input schema is an artifact contract: restore each child independently, fail closed.
        self.cfg = replace(self.cfg)
        restore_context_modes(self.cfg, getattr(self.model, "arch", {}) or {})
        try:
            mode = str(getattr(self.model, "virtual_sensor_mode", "") or
                       (getattr(self.model, "arch", {}) or {}).get("virtual_sensor_mode", "")
                       ).strip().lower()
            if not mode:
                return
            cur = str(getattr(self.cfg, "virtual_sensor_mode", "off") or "off").lower()
            if cur != mode:
                self.cfg = replace(self.cfg, virtual_sensor_mode=mode)
                print(f"[engine] virtual_sensor_mode: cfg '{cur}' → 모델 메타 '{mode}' 로 동기화",
                      flush=True)
        except Exception as e:                             # noqa: BLE001 — 동기화 실패가 로드를 막으면 안 됨
            print(f"[engine] virtual_sensor_mode 동기화 실패(무시): {e}", flush=True)

    def load(self):
        """모델·임계 로드(1회). 실패해도 서버는 떠 있고 error 로 표면화."""
        if self.model is not None or self.error is not None:
            return
        with self._lock:
            if self.model is not None or self.error is not None:
                return
            try:
                if self.selected:
                    self.art = Path(self.selected)        # 드롭다운 선택 우선
                else:
                    self.art = _resolve_art()             # 아니면 자동 탐색
                if not _has_model(self.art):
                    raise FileNotFoundError(
                        f"학습된 모델 없음: {self.art} — model\\ 또는 artifacts\\ 폴더에 "
                        f"*.onnx(+*.npz 메타) 또는 *.pt 를 넣으세요.")
                if self.artifact is None and has_chamber_layout(self.art):
                    specs = discover_chamber_models(self.art)
                    if not specs:
                        raise ValueError("챔버별 ONNX 모델이 없습니다.")
                    ok, why = _probe_model_dir(str(self.art), timeout=330.0)
                    if not ok:
                        raise RuntimeError(why)
                    children = {}
                    for ch, spec in specs.items():
                        child = Engine()
                        child.selected = spec.onnx.parent
                        child.artifact = spec
                        child.load()
                        if not child.ready or child.error or child.threshold_warning:
                            raise RuntimeError(f"{ch.upper()}: {child.error or child.threshold_warning}")
                        children[ch] = child
                    first = next(iter(children.values()))
                    for name in ("model", "cfg", "threshold", "thresholds", "score_norm", "params"):
                        setattr(self, name, getattr(first, name))
                    self.chamber_engines = children
                    self.chamber_artifacts = specs
                    self.bundle_id = artifact_digest(specs)
                    if any(getattr(e.model, "normalization_mode", "standard") == "minmax"
                           for e in children.values()):
                        self.bundle_id += "-minmax-v1"  # exclude results from pre-minmax inference
                    if any(auxiliary_sensor_columns(e.cfg) for e in children.values()):
                        self.bundle_id += "-cylinder-input-v1"
                    _rules = {ch: override_identity(e.wafer_score_override) for ch, e in children.items() if e.wafer_score_override}
                    if _rules:
                        self.bundle_id += "-override-" + hashlib.sha256(json.dumps(_rules, sort_keys=True).encode()).hexdigest()[:12]
                    self.backend = "onnx-chambers"
                    self.threshold_mode = "per_chamber"
                    return
                # ★ 시작 시에도 '자식 프로세스에서 먼저 열어 보기'.
                #   torch/onnxruntime 이 네이티브 레벨에서 죽는 모델이면 앱이 통째로 사라져
                #   화면에 아무것도 안 남는다(콘솔이 그냥 닫힘). 자식이 대신 죽게 해서
                #   앱은 살아남고 이유를 error 로 표면화한다.
                # ★ 시작 사전 점검 스킵 캐시(영속) — 재시작이 오래 걸리던 주원인.
                #   사전 점검은 '자식 프로세스'를 띄우는데, 얼린 exe 에서 자식은 앱
                #   전체를 다시 로딩한다(수십 초, CPU 제한 시 더). 같은 모델(파일
                #   서명 동일)을 지난번에 정상 로드했다면 점검을 건너뛴다 —
                #   모델이 바뀌었을 때만 다시 점검한다.
                _sig_now = _model_sig(self.art)
                _last_ok = str(_load_app_config().get("model_probe_ok") or "")
                if self.artifact is not None or (_sig_now and _sig_now == _last_ok):
                    print("[engine] 모델 사전 점검 생략(지난 실행에서 정상 확인된 모델)",
                          flush=True)
                    _ok, _why = True, ""
                else:
                    _ok, _why = _probe_model_dir(str(self.art))
                if not _ok and not self.selected:
                    # ★ 사용자가 직접 고른 모델이 아니면(기본모델·자동탐색 — 예: 데이터
                    #   저장소를 옮기며 딸려 온 다른 PC 의 default_model) 앱을 오류로
                    #   두지 않는다. 그 폴더만 빼고 열리는 모델(내장 포함)로 폴백.
                    print(f"[engine] 자동 선택 모델 점검 실패 → 건너뛰고 재탐색: "
                          f"{self.art} ({_why})", flush=True)
                    try:
                        _bad = str(Path(self.art).resolve())
                    except Exception:                      # noqa: BLE001
                        _bad = str(self.art)
                    _alts = list(reversed(_submodel_dirs())) + [PKG / "artifacts"]
                    _ms = _model_storage_root()
                    if _ms is not None:
                        _alts.insert(0, _ms)
                    for _alt in _alts:
                        try:
                            if str(Path(_alt).resolve()) == _bad or not _has_model(_alt):
                                continue
                        except Exception:                  # noqa: BLE001
                            continue
                        _ok2, _ = _probe_model_dir(str(_alt))
                        if _ok2:
                            self.art = Path(_alt)
                            _ok, _why = True, ""
                            print(f"[engine] 폴백 모델 사용: {self.art}", flush=True)
                            break
                if not _ok:
                    raise RuntimeError(
                        f"{_why}\n  폴더: {self.art}\n"
                        f"  · 이 PC 의 CPU 나 라이브러리가 이 모델을 여는 데 필요한 명령을 "
                        f"지원하지 않을 수 있습니다.\n"
                        f"  · 다른 모델 폴더를 지정하거나, 설정에서 모델을 바꿔 주세요.")
                self.cfg = self._build_cfg()
                from src.onnx_backend_forecast import OnnxForecastModel
                # ── .pt 만 있으면 자동으로 ONNX 변환(torch 있는 환경) ──
                #    모델을 .pt 로 받아 폴더에 넣으면, 첫 로드 때 .onnx(+메타) 를 만들어
                #    이후로는 torch 없이 onnxruntime 로 추론한다.
                # ★ IO_BACKEND=torch 면 ONNX 를 아예 쳐다보지 않는다.
                #   WES7 빌드처럼 onnxruntime 이 그 OS 에서 안 도는 경우, .onnx 가 폴더에
                #   있으면 그쪽을 골랐다가 로드 시점에 죽는다. 강제 지정으로 그 경로를 막는다.
                _force = (os.environ.get("IO_BACKEND") or "").strip().lower()
                if self.artifact is not None and _force == "torch":
                    raise RuntimeError("챔버별 ONNX 모델은 ONNX 실행 환경이 필요합니다(IO_BACKEND=torch 해제).")
                if _force == "torch":
                    print("[engine] IO_BACKEND=torch — ONNX 백엔드를 건너뜁니다", flush=True)
                elif self.artifact is None and not OnnxForecastModel.available(self.art) and _find_pt(self.art) is not None:
                    conv = _auto_convert_pt(self.art)
                    if conv.get("msg"):
                        print(f"[engine] {conv['msg']}")   # 변환됨/불가 모두 로그 남김
                # ── onnx(경량, torch 불필요) 우선, 없으면 torch(.pt) 폴백 ──
                if _force != "torch" and (self.artifact is not None or OnnxForecastModel.available(self.art)):  # .onnx + 짝 .npz
                    self.model = (OnnxForecastModel(self.artifact.onnx, self.artifact.meta, self.cfg)
                                  if self.artifact else OnnxForecastModel.load(self.art, self.cfg))
                    self.backend = "onnx"
                    # ★ 학습이 점수를 정규화했으면 백엔드도 같은 단위로 계산하게 주입
                    self.model.score_norm = self.score_norm
                    self._sync_cfg_from_model()            # 메타의 virtual_sensor_mode → 로드 cfg
                else:
                    pt = _find_pt(self.art)
                    if pt is None and _force == "torch":
                        raise RuntimeError(
                            f"IO_BACKEND=torch 인데 모델 폴더에 .pt 가 없습니다: {self.art}\n"
                            f"  학습 결과의 tcn_forecast.pt 를 이 폴더에 넣으세요.")
                    if pt is None:                         # onnx 는 있는데 meta 없음 등
                        raise RuntimeError(
                            f"모델 폴더에 짝이 맞는 파일이 없습니다: {self.art}\n"
                            f"  · onnx 배포: *.onnx + 같은 이름/recon_meta.npz 가 함께 필요\n"
                            f"  · .pt 만 있으면 먼저 변환: python -m src.export_onnx_forecast")
                    try:
                        from src.model_tcn_forecast import TCNForecastModel
                    except ModuleNotFoundError:           # 배포 exe 엔 torch 없음
                        raise RuntimeError(
                            f"이 모델 폴더에는 *.onnx(+*.npz) 가 필요합니다.\n"
                            f"  폴더: {self.art}\n"
                            f"  .pt 만 있는 모델은 먼저 ONNX 로 변환하세요: "
                            f"python -m src.export_onnx_forecast")
                    # ★ torch 는 자기 기준(물리코어 수)으로 스레드를 잡는다 — 앱의 '코어 예약'
                    #   설정을 따르지 않는다. 환경변수(OMP_NUM_THREADS)도 무시하는 경우가 있어
                    #   여기서 명시적으로 지정한다.
                    try:
                        import torch as _t
                        _n = int((_CPU_BUDGET or {}).get("threads") or 0)
                        if _n > 0 and _CPU_BUDGET.get("use_cores", 0) < _CPU_BUDGET.get("physical", 0):
                            _t.set_num_threads(_n)
                            print(f"[engine] torch 스레드 {_t.get_num_threads()}개로 제한"
                                  f"(예약 {_CPU_BUDGET.get('reserve')}코어)", flush=True)
                    except Exception as _e:              # noqa: BLE001
                        print(f"[engine] torch 스레드 수 지정 실패: {_e}", flush=True)
                    self.model = TCNForecastModel.load(pt, self.cfg, device="cpu")
                    self.backend = "torch"
                    self._sync_cfg_from_model()            # 메타의 virtual_sensor_mode → 로드 cfg
                    try:                                  # ★ torch 폴백도 학습과 같은 점수 단위로
                        _c = getattr(self.model, "cfg", None)
                        if _c is not None:
                            _c.score_time_avg = (self.score_norm == "time")
                            _c.score_final_avg = (self.score_norm == "final")
                        # 센서 점수·누적 곡선(recon_explain)이 밴드와 같은 단위를 쓰도록
                        self.model.score_norm = self.score_norm
                    except Exception:                     # noqa: BLE001
                        pass
                if not self._override_in_json:
                    self.wafer_score_override = normalize_override(getattr(self.model, "wafer_score_override", None))
                    if self.wafer_score_override is None:
                        config = self.artifact.config if self.artifact else self.art / "search_best.json"
                        if not config.exists():
                            config = self.art / "best_search.json"
                        _, self.wafer_score_override = artifact_override(config.parent, config)
                if self.threshold is None:                # best.json 없을 때 폴백
                    self.threshold = 0.0
                # ★ search_best.json 이 아예 없으면 위쪽 임계 블록을 타지 않아
                #   경고가 설정되지 않았다(=사용자는 이유를 모른 채 등급이 '주의'로만 나옴).
                #   여기서 다시 판정해 경고를 남긴다.
                if (not self.thresholds) and float(self.threshold or 0.0) <= 1e-12:
                    self.threshold_mode = "none"
                    self.threshold_warning = (
                        "이 모델 폴더에 판정 임계가 없습니다(search_best.json 없음/threshold 미기재). "
                        "등급(주의·경고·심각)을 계산할 수 없어 불량이 모두 '주의'로만 표시됩니다. "
                        "학습 결과의 search_best.json 을 모델 폴더에 함께 넣어 주세요.")
                    print("[engine] ⚠ 경고: " + self.threshold_warning, flush=True)
                try:
                    RX._setup_font()                      # 그림 한글 폰트
                except Exception:                         # noqa: BLE001
                    pass
                print(f"[engine] 모델 로드({self.backend}): {self.art} · "
                      f"레시피임계 {self.thresholds or '단일'}")
                try:                                      # ★ 다음 실행의 사전 점검 생략용 서명 저장
                    _cfg2 = _load_app_config()
                    _s2 = _model_sig(self.art)
                    if _s2 and _cfg2.get("model_probe_ok") != _s2:
                        _cfg2["model_probe_ok"] = _s2
                        _save_app_config(_cfg2)
                except Exception:                         # noqa: BLE001
                    pass
            except Exception as e:                        # noqa: BLE001
                self.error = f"{e}"
                traceback.print_exc()

    def info(self) -> Dict:
        return {"ready": self.ready, "error": self.error, "backend": self.backend,
                "model_dir": str(self.art), "thresholds": self.thresholds,
                "threshold": self.threshold, "threshold_mode": self.threshold_mode,
                "chamber_models": {ch: {"onnx": str(spec.onnx), "meta": str(spec.meta),
                    "config": str(spec.config)} for ch, spec in self.chamber_artifacts.items()}}

    def reload(self) -> Dict:
        """폴더의 웨이트를 다시 읽어 모델 교체(재시작·재빌드 불필요)."""
        with self._lock:
            self.model = None; self.error = None
            self.threshold = None; self.thresholds = {}; self.threshold_mode = "unknown"
            self.score_norm = "none"
            self.threshold_warning = None
            self.params = {}
            self.cfg = None
        self.load()
        return {"ready": self.ready, "error": self.error, "backend": self.backend,
                "model_dir": str(self.art), "thresholds": self.thresholds,
                "threshold": self.threshold, "threshold_mode": self.threshold_mode}

    def select(self, model_dir: str) -> Dict:
        """드롭다운에서 고른 모델 폴더로 전환."""
        self.selected = Path(model_dir) if model_dir else None
        return self.reload()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.error is None


ENGINE = EngineRouter(Engine(), busy=lambda: any(
    (globals().get("STATE", {}).get(key) or {}).get("active")
    for key in ("progress", "progress_bg")) or globals().get("REPORT", {}).get("running", False))


def _engine_for_wafer(wid, row=None):
    ch = (row or {}).get("chamber") or _chamber(wid)
    return ENGINE.for_chamber(ch)


def _wafer_model(fn):
    """Keep model/schema/threshold consistent for one wafer, including archived-only data."""
    signature = inspect.signature(fn)
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with ENGINE.pinned():
            if not ENGINE.root.chamber_engines or ENGINE._scope.get() is not None:
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs).arguments
            rec = bound.get("rec") or {}
            wid = bound.get("wid") or bound.get("wafer_id") or rec.get("wafer_id")
            row = rec
            if bound.get("date") and wid:
                arch = _load_archive(bound["date"], wid)
                if arch:
                    row = arch["meta"]
            with ENGINE.using(_engine_for_wafer(wid, row)):
                return fn(*args, **kwargs)
    return wrapped


def _pin_model(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with ENGINE.pinned():
            return fn(*args, **kwargs)
    return wrapped


# 마지막 검사 결과 캐시(상세/이미지 endpoint 가 재로딩 없이 재사용)
STATE: Dict = {"folder": str(_inbox()), "results": [], "recs": {}, "inspected_at": None,
               "progress": {"active": False, "done": 0, "total": 0, "phase": "", "folder": ""},
               # 백그라운드 자동감시 진행률(전면 검사바와 분리 — 서로 번갈아 깜빡이지 않게)
               "progress_bg": {"active": False, "done": 0, "total": 0, "phase": "", "folder": ""}}

# STATE["recs"] 는 웨이퍼당 원본 배열(io/sensors)을 들고 있어 누적되면 메모리가 계속 점유된다.
# 최근 N개만 메모리에 유지(그 외는 아카이브 npz 로 상세 제공) → RSS 상한. 0/음수면 무제한.
#   ※ 상세/그래프는 아카이브(npz)로 조회되므로 recs 는 '최근 재검사 편의용' 캐시일 뿐 → 작게 유지.
RECS_MAX = int(os.environ.get("IO_RECS_MAX", "100"))


def _trim_recs():
    """STATE['recs'] 를 최근 RECS_MAX 개로 제한(오래된 것부터 제거). STATE_LOCK 안에서 호출."""
    if RECS_MAX <= 0:
        return
    recs = STATE.get("recs") or {}
    excess = len(recs) - RECS_MAX
    if excess > 0:
        for k in list(recs.keys())[:excess]:              # dict 삽입순=오래된 순 → 앞에서 제거
            recs.pop(k, None)


_LIBC = None


def _release_memory():
    """검사 후 메모리 반환. gc.collect() 만으로는 해제된 배열이 힙 아레나에 남아 RSS 가
    안 줄어든다(측정 확인).
      · Linux : malloc_trim(0) 으로 OS 에 실제 반환.
      · Windows: SetProcessWorkingSetSize(-1,-1) 로 워킹셋을 OS 에 반환(대량검사 RSS 우상향 방지).
    (예전엔 Windows 에 처리가 없어 6000장 처리 시 메모리 점유가 계속 올라갔음)"""
    global _LIBC
    gc.collect()
    try:
        if sys.platform.startswith("linux"):
            if _LIBC is None:
                import ctypes
                _LIBC = ctypes.CDLL("libc.so.6")
            _LIBC.malloc_trim(0)
        elif sys.platform.startswith("win"):
            import ctypes
            k = ctypes.windll.kernel32
            # (SIZE_T)-1, (SIZE_T)-1 = '워킹셋을 최소로 트림' → 미사용 페이지를 OS 로 반환
            k.SetProcessWorkingSetSize(k.GetCurrentProcess(),
                                       ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except Exception:                                     # noqa: BLE001
        pass


def _track_inspection_request(func):
    """Keep progress active between folders; score queries may run concurrently."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        with STATE_LOCK:
            STATE["score_inspect_requests"] = STATE.get("score_inspect_requests", 0) + 1
        try:
            return func(*args, **kwargs)
        finally:
            with STATE_LOCK:
                STATE["score_inspect_requests"] -= 1
    return wrapped


def _set_progress(active, done, total, phase, folder="", bg=False):
    """검사 진행률 기록. bg=True(자동감시)면 전면 검사바와 별개 슬롯에 기록해
    전면(수동/다시검사) 진행바와 '입력 ↔ 검사'로 번갈아 깜빡이지 않게 한다."""
    key = "progress_bg" if bg else "progress"
    now = time.time()
    with STATE_LOCK:
        prev = STATE.get(key) or {}
        t0 = prev.get("t0")
        # 시작 시점(재)설정: 비활성→활성 전환, done 리셋/역행, 또는 최초.
        if (not prev.get("active")) or t0 is None or int(done) == 0 or int(done) < int(prev.get("done") or 0):
            t0 = now
        elapsed = max(1e-6, now - t0)
        rate = (done / elapsed) if done > 0 else 0.0            # 장/초
        eta = ((total - done) / rate) if (rate > 0 and total > done) else 0.0
        STATE[key] = {"active": bool(active), "done": int(done),
                      "total": int(total), "phase": str(phase), "folder": str(folder),
                      "t0": t0, "elapsed_s": round(elapsed, 1),
                      "rate": round(rate, 2), "eta_s": round(eta)}
STATE_LOCK = threading.Lock()
# matplotlib(pyplot)은 전역 상태를 써서 스레드 안전하지 않다. 리포트 백그라운드
# 스레드가 그림을 그리는 동안 다른 탭(상세/이력)의 플롯 요청이 동시에 렌더하면
# pyplot 상태가 깨져 리포트가 중간에 실패한다("되다마네"). 모든 렌더를 이 락으로 직렬화.
_PLOT_LOCK = threading.Lock()
# ── RCA 실행 가드: 동시 1건 + 타임아웃(무한 '분석 중…'과 CPU 중첩 폭주 방지) ──
from concurrent.futures import ThreadPoolExecutor as _RcaPool, TimeoutError as FuturesTimeout
_RCA_POOL = _RcaPool(max_workers=1, thread_name_prefix="rca")
_RCA_STATE = {"busy": False}
_RCA_STATE_LOCK = threading.Lock()
_RCA_TIMEOUT_S = 300

# 검사 일시중단: set()=진행, clear()=일시중단. 검사 루프가 청크 경계마다 확인해 대기.
_PAUSE = threading.Event()
_PAUSE.set()


class InspectAborted(BaseException):
    """'완전 중단' 요청. BaseException 상속 — 배치추론/prefill 의 광범위한
    `except Exception` 에 삼켜지지 않고 검사 스택을 즉시 탈출하기 위함."""


_INSPECT_CONTEXT = threading.local()
_ABORT = threading.Event()                                # '완전 중단' 요청 플래그(전면 검사 전용)


def _check_inspect_cancelled():
    context = getattr(_INSPECT_CONTEXT, "auto", None)
    if context is not None:
        auto, generation = context
        if auto._stop.is_set() or not auto.enabled or auto._generation != generation:
            raise InspectAborted()
    elif _ABORT.is_set():
        raise InspectAborted()


def _reader_cancel_check():
    context = getattr(_INSPECT_CONTEXT, "auto", None)
    def check():
        if context is not None:
            auto, generation = context
            if auto._stop.is_set() or not auto.enabled or auto._generation != generation:
                raise InspectAborted()
        elif _ABORT.is_set():
            raise InspectAborted()
    return check


def _cancel_for_path_change():
    # Invalidate the captured scan plan, including an OFF -> ON race.
    if AUTOPILOT is not None:
        AUTOPILOT.cancel_scan()
    with STATE_LOCK:
        if STATE.get("score_inspect_requests", 0) or (STATE.get("progress") or {}).get("active"):
            _ABORT.set()
    _PAUSE.set()


def _wait_if_low_memory(tag: str = "", floor_mb: int = 500, max_wait_s: int = 120):
    """가용 메모리가 바닥이면 청크 경계에서 잠시 기다린다 — 죽지 않고 느리게.

    ★ 왜 필요한가 — 제어기에서 설비 프로그램이 메모리를 확 가져가는 순간에 검사가
      큰 할당을 하면 네이티브 bad_alloc(abort)로 앱이 흔적 없이 사라졌다.
      리소스가 모자라면 크래시가 아니라 대기/저속이어야 한다(사용자 요구).
    가용 메모리(물리·커밋 중 작은 쪽)가 floor_mb 미만이면 2초 간격으로 재확인,
    max_wait_s 가 지나면 그냥 진행한다(영구 대기 방지 — 이후는 분할 재시도가 받는다).
    """
    try:
        from src.onnx_backend_forecast import OnnxForecastModel as _M
        t0 = time.time()
        warned = False
        while time.time() - t0 < max_wait_s:
            _check_inspect_cancelled()
            avail = _M._avail_mem_bytes()
            if avail is None or avail >= floor_mb * 1024 * 1024:
                return
            if not warned:
                warned = True
                print(f"[inspect] 가용 메모리 {avail // (1024*1024)}MB < {floor_mb}MB — "
                      f"확보될 때까지 대기(최대 {max_wait_s}초){' · ' + tag if tag else ''}",
                      flush=True)
            time.sleep(2.0)
        print(f"[inspect] 메모리 대기 {max_wait_s}초 초과 — 분할 배치로 진행", flush=True)
    except Exception:                                     # noqa: BLE001 — 대기 실패가 검사를 막으면 안 됨
        pass


def _wait_if_paused(done, total, folder, bg=False):
    """일시중단 요청이 걸려 있으면 재개될 때까지 블록(진행바에 '일시중단' 표시).
    '완전 중단' 또는 자동검사 취소 요청 시 InspectAborted 로 탈출."""
    _check_inspect_cancelled()
    if _PAUSE.is_set():
        return
    while not _PAUSE.is_set():
        _check_inspect_cancelled()
        _set_progress(True, done, total, "일시중단", folder, bg=bg)
        _PAUSE.wait(timeout=0.4)
    _check_inspect_cancelled()
    _set_progress(True, done, total, "검사", folder, bg=bg)   # 재개 즉시 '검사'로 복귀


def _pause_gate():
    """Check cancellation for this worker, including while paused."""
    _check_inspect_cancelled()
    while not _PAUSE.is_set():
        _check_inspect_cancelled()
        _PAUSE.wait(timeout=0.3)
    _check_inspect_cancelled()

AUTOPILOT = None                                    # AutoPilot 인스턴스(아래에서 생성)


# ════════════════════════════════════════════════════════════════════
#  추론 로직
# ════════════════════════════════════════════════════════════════════

def _recipe_display(rc):
    """표시·그룹·필터용 유형명. ★ 사용자 요청(2026-08-06): 29a 로 간편화하지 말고 '풀네임 그대로'.

    예전엔 배율표 키(29a)로 축약해 표시했으나, 이제 원본 풀네임(29aece38bdydhsi273jd)을
    표시·필터·그룹 전부에 통일해서 쓴다. (배율/임계 매칭은 여전히 접두로 29a↔풀네임 호환)
    표시가 어디서나 같은 풀네임이므로 필터 옵션↔행↔매칭이 항상 일치한다."""
    return _clean_recipe(rc) or "unknown"


def _recipe_mult(recipe) -> float:
    """★ 배포판: 유형별 임계 '배수' 기능 제거(사용자 결정) — 항상 1.0.
    모델이 가진 레시피별 임계(search_best.json)는 그대로 쓴다. 배수 편집 UI/파일은
    항상성 사고(설정 파일 파편화)의 원인이라 배포판에서 뺐다."""
    return 1.0

def _model_recipe_mult(recipe) -> float:
    """모델이 레시피별 임계를 '직접' 제공하는 경우의 배수.

    사용자 결정: 모델 임계를 그대로 사용(배수 중립화). 단, 사용자가 배수를 레거시 기본값과
    '다르게' 명시 설정했다면 그 값을 존중한다.  → 기본(레거시 시드)과 같으면 1.0, 다르면 그 배수.
    """
    return 1.0                                            # 배포판: 배수 중립


def _plot_threshold_dict():
    """플롯 정규화용 임계 dict. base(학습 임계)=1.0 기준, 결정선=배수(29a→1.1).

    plot_* 는 base_threshold_by_recipe 로 값을 1 로 정규화하고, threshold_by_recipe
    (=base×배수) 로 결정선을 그린다 → 29a 는 1.1, 나머지는 1.0 위치에 선이 뜬다.
    """
    base = {}
    for k, v in (ENGINE.thresholds or {}).items():
        try:
            base[str(k)] = float(v)
        except Exception:                                # noqa: BLE001
            pass
    d: Dict = {}
    if base:
        # 모델 제공 임계는 배수 중립화(_model_recipe_mult) → 결정선이 실제 판정과 일치.
        mult = {r: _model_recipe_mult(r) for r in base}
        d["threshold_by_recipe"] = {r: base[r] * mult[r] for r in base}
        d["threshold_raw"] = dict(d["threshold_by_recipe"])
        d["base_threshold_by_recipe"] = dict(base)
        d["threshold_recipe_multipliers"] = dict(mult)
    g = float(ENGINE.threshold or 0.0)
    if g > 0:
        d["default"] = g
        d["global_threshold_raw"] = g
    return d or None


@_wafer_model
def _recipe_base_decision(wafer_id: str):
    """(recipe, base, decision) 반환.

    - base     : 그 레시피의 '학습/폴백 임계'(앱 배수 적용 前).
    - decision : base × 유효배수(=실제 '판정 임계'). verdict·산점도가 쓰는 값.
    누적 복원오차 그래프가 판정선을 base(=1)가 아니라 decision 으로 그리도록,
    base 와 decision 을 함께 돌려준다. (기존 _recipe_threshold 의 decision 계산과 100% 동일)
    """
    tds = ENGINE.thresholds or {}
    fallback = float(ENGINE.threshold or 0.0)
    # 레시피 식별 후보 = 파일의 임계 레시피 ∪ 설정의 recipe_names(임계 없어도 이름은 인식)
    known = list(tds.keys())
    try:
        known += [str(x) for x in getattr(ENGINE.cfg, "recipe_names", ()) or ()]
    except Exception:                                # noqa: BLE001
        pass
    # ① .log 에서 뽑은 파일별 레시피 태그 우선(풀네임). 파일명 토큰은 폴백.
    tag = _recipe_tag(str(wafer_id))
    if tag:
        m = _recipe_mult(tag)
        # ★ 모델이 레시피별 임계를 직접 제공하면 그 값을 그대로 결정임계로(배수 중립화).
        #   폴백(전용 임계 없음)일 때만 레거시 배수를 적용한다.
        if tag in tds:                               # 태그명이 임계표에 그대로 있으면
            b = float(tds[tag]); return tag, b, b * _model_recipe_mult(tag)
        tl = tag.lower()
        best = None                                  # (키길이, k, v) — 임계키가 태그의 접두인 것 중 최장
        for k, v in tds.items():                     # 대소문자/접두 매칭(29aece39↔29a)
            kl = k.lower()
            if (kl == tl or tl.startswith(kl)) and (best is None or len(kl) > best[0]):
                best = (len(kl), k, v)
        if best is not None:
            b = float(best[2]); return tag, b, b * _model_recipe_mult(best[1])
        return tag, fallback, fallback * m           # 이름은 알지만 전용 임계 없음 → 폴백×배수(레거시)
    if not known:
        return None, fallback, fallback
    toks = [t for t in re.split(r"[_\-.\s]+", str(wafer_id)) if t]
    tokset = set(toks)
    lowmap = {t.lower(): t for t in toks}
    recipe = None
    for k in known:                                  # 정확 토큰 일치 우선(긴 이름 우선)
        if k in tokset:
            recipe = k; break
    if recipe is None:
        for k in known:                              # 대소문자 무시
            if k.lower() in lowmap:
                recipe = k; break
    if recipe is None:
        # ★ 접두 매칭: 파일명 토큰(풀네임 '29asaeasd135ews')이 짧은 설정키('29a')로 시작하면 그 키로 인식.
        #   .log 레시피 태그 경로와 동일 규칙 → 두 경로 일관(가장 구체적인 최장 키 우선).
        best = None                                  # (키길이, 토큰, 키)
        for t in toks:
            tl = t.lower()
            for k in known:
                kl = k.lower()
                if kl and tl.startswith(kl) and (best is None or len(kl) > best[0]):
                    best = (len(kl), t, k)
        if best is not None:
            _kl, tok, key = best
            if key in tds:
                b = float(tds[key]); return tok, b, b * _model_recipe_mult(key)
            for kk, vv in tds.items():
                if kk.lower() == key.lower():
                    b = float(vv); return tok, b, b * _model_recipe_mult(kk)
            return tok, fallback, fallback * _recipe_mult(key)  # 이름만 알고 전용 임계 없음
    if recipe is None:
        return None, fallback, fallback              # 레시피 미검출 → 폴백
    # 임계: 그 레시피가 파일 dict 에 있으면 그 값(배수 중립화), 없으면 폴백×배수(레거시)
    m = _recipe_mult(recipe)
    if recipe in tds:
        b = float(tds[recipe]); return recipe, b, b * _model_recipe_mult(recipe)
    for k, v in tds.items():
        if k.lower() == recipe.lower():
            b = float(v); return recipe, b, b * _model_recipe_mult(recipe)
    return recipe, fallback, fallback * m            # 이름은 인식했지만 전용 임계 없음


def _recipe_threshold(wafer_id: str):
    """(recipe, decision) — decision=배수 적용된 '결정 임계'. base 가 필요하면 _recipe_base_decision."""
    r, _b, dec = _recipe_base_decision(wafer_id)
    return r, dec


def _waferaccum_threshold(wafer_id: str, rec) -> Dict:
    """누적 복원오차 그래프의 '판정선'을 실제 판정임계(base×앱배수)와 동기화한 플롯 임계 dict.

    문제: 전용 모델임계가 없는 레시피(global 모드/폴백)는 판정이 base×앱배수 인데,
    _plot_threshold_dict 엔 그 레시피 항목이 없어 판정선이 global(=선 1, 배수 미반영)로 떨어졌다.
    → 그래프는 1을 넘었는데 판정은 정상인 불일치. 여기서 이 웨이퍼의 base/decision 을
    '레코드 레시피 키'로 직접 지정 → 곡선은 base 로 정규화(배수와 무관하게 1), 판정선은
    decision(=base×앱배수)에 그려져 산점도·판정과 정확히 일치한다.
    """
    recipe, base, decision = _recipe_base_decision(wafer_id)
    try:
        b = float(base); d = float(decision)
    except (TypeError, ValueError):
        return _plot_threshold_dict()
    if not (np.isfinite(b) and b > 1e-12 and np.isfinite(d) and d > 1e-12):
        return _plot_threshold_dict()                     # 안전 폴백(기존 동작)
    # Overridden curves use the same decision threshold as their final displayed score.
    saved = rec or {}
    if saved.get("score_overridden"):
        effective = float(saved["effective_score_raw"])
        ratio = float(saved.get("override_ratio") or 1.01)
        b = d = effective / ratio
    elif effective_wafer_score(saved, 0.0, d, getattr(ENGINE, "wafer_score_override", None))["score_overridden"]:
        b = d
    rkey = str(saved.get("recipe") or recipe or "unknown") or "unknown"
    return {
        "base_threshold_by_recipe": {rkey: b},            # 곡선 정규화 기준(=base)
        "threshold_by_recipe": {rkey: d},                 # 결정선 = base×앱배수
        "threshold_raw": {rkey: d},
        "threshold_recipe_multipliers": {rkey: (d / b)},
        "default": b, "global_threshold_raw": b,          # 매칭 실패 시도 base 기준(최소 산점도와 정합)
    }


def _judge(score: float, wafer_id: str = None) -> str:
    _, thr = _recipe_threshold(wafer_id)
    return "NG" if score >= thr else "OK"


def _floor2(x) -> float:
    """점수 표시용 '버림'(내림, 2자리). 사용자 요청: 점수 숫자는 반올림 말고 무조건 버림.
    예) 0.989→0.98, 1.278→1.27. (판정은 raw 로 하므로 표시만 버림 — 경계 1.0 관계는 유지)"""
    try:
        scaled = float(x) * 100
        return math.floor(scaled + 2 * 2.220446049250313e-16 * max(1., abs(scaled))) / 100.0
    except (TypeError, ValueError):
        return x


@_wafer_model
def _score_view(wafer_id: str, raw_score: float, rec=None) -> Dict:
    """이상점수를 '임계=1' 기준으로 정규화한 표시값을 만든다.

    - score      : raw_score / base(레시피 학습 임계)  → 임계 1.0 에서 판정
    - threshold  : 배수 전부 1.0(중립화)               → 결정임계 = base 그대로
    - verdict    : raw_score >= base×배수(=base) 이면 NG
    raw 값(score_raw/threshold_raw)도 함께 담아 추적 가능하게 한다.
    """
    recipe, decision = _recipe_threshold(wafer_id)   # decision = base×배수 (raw 결정임계)
    rec = rec or STATE.get("recs", {}).get(wafer_id) or {"wafer_id": wafer_id}
    audit = effective_wafer_score(rec, raw_score, decision, getattr(ENGINE, "wafer_score_override", None))
    raw_score = audit["effective_score_raw"]
    # ★ 결정임계(base×배수)로 나눠 '임계=1' 로 통일. 레시피 배수를 바꿔도 기준은 항상 1.
    sn = raw_score / decision if decision and decision > 1e-12 else raw_score
    return {
        **audit,
        "source_path": rec.get("source_path"),
        "recipe": recipe,
        "score": audit["override_ratio"] if audit["score_overridden"] else _floor2(sn),                         # 정규화 점수(임계 1 기준, 2자리 '버림')
        "threshold": 1.0,                             # 임계 기준 항상 1
        "verdict": "NG" if raw_score >= decision else "OK",
        "score_raw": raw_score if audit["score_overridden"] else math.floor(raw_score * 1000) / 1000.0,     # raw 도 버림(3자리)
        "threshold_raw": math.floor(float(decision) * 1000) / 1000.0,
    }


def _load_aligned(folder: Path, stems=None):
    """★ load_dataset 로 폴더를 로드하고, 각 rec 의 IO 를 모델 io_cols 순서로 재정렬.

    forecast 모델은 센서를 rec["sensor_cols"] 이름으로 내부 매핑하므로 센서는 재정렬하지 않는다.
    IO 는 위치기반이므로 이름으로 모델 io_cols 순서에 맞춰 (T, len(model.io_cols)) 로 채운다.

    반환: (recs, io_cols, sensor_cols, coverage).
    """
    if ENGINE.root.chamber_engines and ENGINE._scope.get() is None:
        files = _data_files(_resolve_existing_dir_ci(Path(folder)))
        wanted = set(stems) if stems is not None else None
        groups = {}
        for f in files:
            if wanted is not None and f.stem not in wanted:
                continue
            _tag_file_meta(f.parent, f.stem, source_path=f)
            ch = (_extract_chamber_from_log(f) if f.suffix.lower() in LOG_EXTS
                  else _file_meta(f.stem).get("chamber"))
            if not ch:
                raise ValueError(f"챔버 미확인: {f} — 실제 2행 C열의 설비별 PMCUNIT 매핑을 확인하세요 (WUK380B: 2/4/6/8/10/12, 기타: 7~12).")
            ENGINE.for_chamber(ch)  # fail closed: never apply an arbitrary chamber's model
            groups.setdefault(ch, set()).add(f.stem)
        combined, coverages = [], {}
        io_all, sen_all = [], []
        for ch, group in sorted(groups.items()):
            with ENGINE.using(ENGINE.for_chamber(ch)):
                recs, io, sen, coverage = _load_aligned(folder, group)
                for rec in recs:
                    rec["chamber"] = ch
                    rec["model_chamber"] = ch
                combined.extend(recs)
                io_all.extend(c for c in io if c not in io_all)
                sen_all.extend(c for c in sen if c not in sen_all)
                coverages[ch] = coverage
        cov = dict(next((v for v in coverages.values() if v), {}))
        cov["chambers"] = coverages
        return combined, io_all, sen_all, cov
    A = ENGINE.model
    model_io = [str(c) for c in A.io_cols]
    input_sen = [str(c) for c in A.input_sensor_cols]     # 모델 입력(과거) 센서

    cfg = ENGINE.cfg                                       # 이미 io_keep=()
    _apply_cache_env()                                     # 파싱 캐시 on/off(설정) — 모든 로드 경로의 관문
    recs, io_cols, sensor_cols, _ = load_dataset(folder, cfg, stems=stems, cancel_check=_reader_cancel_check(), include_trend_source=True)   # stems=신규만(스캔 가속)
    if not recs:
        return [], model_io, input_sen, None

    # ── 센서/IO 별칭 적용: 데이터 컬럼명 → 모델 센서명(설정 탭에서 지정) ──
    #    설비마다 컬럼 헤더가 조금 달라도 모델이 인식하도록 이름을 바꿔준다.
    aliases = {str(k): str(v) for k, v in (_load_app_config().get("sensor_aliases") or {}).items()}
    if aliases:
        io_cols = [aliases.get(str(c), str(c)) for c in io_cols]
        sensor_cols = [aliases.get(str(c), str(c)) for c in sensor_cols]
        for rec in recs:
            if rec.get("sensor_cols") is not None:
                rec["sensor_cols"] = [aliases.get(str(c), str(c)) for c in rec["sensor_cols"]]

    load_io = [str(c) for c in io_cols]
    load_io_idx = {name: i for i, name in enumerate(load_io)}
    load_sen_set = set(str(c) for c in sensor_cols)

    for rec in recs:
        validate_auxiliary_record(rec, cfg)
        src_io = np.asarray(rec["io"], dtype=np.float64)
        T = src_io.shape[0]
        new_io = np.zeros((T, len(model_io)), dtype=np.float64)
        for j, name in enumerate(model_io):
            si = load_io_idx.get(name)
            if si is not None and si < src_io.shape[1]:
                new_io[:, j] = src_io[:, si]
        rec["io"] = new_io
        rec["io_cols"] = list(model_io)
        rec["recipe"] = _recipe_threshold(rec["wafer_id"])[0] or "unknown"  # 배수/정규화용
        source = Path(rec["source_path"]) if rec.get("source_path") else None
        _tag_file_meta(source.parent if source else folder, rec["wafer_id"], source_path=source)          # 종료시각·챔버 태깅(원본 .log 없어도 .csv 로)

    present_io = [c for c in model_io if c in load_io_idx]
    present_sen = [c for c in input_sen if c in load_sen_set]
    miss_sen = [c for c in input_sen if c not in load_sen_set]
    miss_io = [c for c in model_io if c not in load_io_idx]
    auxiliary = list(auxiliary_sensor_columns(cfg))
    cov = {
        "auxiliary_sensors": auxiliary,
        "missing_auxiliary_sensors": [c for c in auxiliary if c not in load_sen_set],
        "n_auxiliary_missing": sum(c not in load_sen_set for c in auxiliary),
        "n_sensor_expected": len(input_sen),
        "n_sensor_missing": len(miss_sen),
        "missing_sensors": miss_sen[:15],
        "sensor_coverage": round(len(present_sen) / max(1, len(input_sen)), 3),
        "n_io_expected": len(model_io),
        "n_io_missing": len(miss_io),
        "missing_io": miss_io[:15],
        "io_coverage": round(len(present_io) / max(1, len(model_io)), 3),
        "found_columns_sample": (load_io + list(sensor_cols))[:20],
    }
    if cov["sensor_coverage"] <= 0.0:
        raise RuntimeError(
            "CSV 의 컬럼명이 이 모델의 센서명과 하나도 일치하지 않습니다. "
            "이 모델로는 이 데이터를 검사할 수 없습니다(이 설비 데이터로 재학습 필요).\n"
            f"  · 모델이 기대하는 센서 예: {input_sen[:8]}\n"
            f"  · CSV 센서 컬럼 예: {list(sensor_cols)[:8]}")
    return recs, model_io, list(sensor_cols), cov


def _natkey(s):
    """파일명 자연정렬 키: 숫자는 숫자로 비교(W2 < W10)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(s))]


def _prefill_batch(recs, pause_check=None) -> None:
    """청크(여러 웨이퍼)를 배치로 한 번에 순전파해 각 rec 캐시를 채운다(GPU/CPU 가속).

    모델이 predict_batch 를 제공하면(ONNX 백엔드) 사용. 실패해도 예외 없이 넘어가고,
    이후 _assess→predict 가 웨이퍼별로 정상 추론(폴백)하므로 결과에는 영향 없음.
    pause_check: 배치 실행 직전 호출 — 일시중단이면 대기해 GPU 를 곧 유휴화(청크 중간에도 멈춤).
    """
    if ENGINE.root.chamber_engines and ENGINE._scope.get() is None:
        groups = {}
        for rec in recs:
            ch = rec.get("chamber") or _chamber(rec["wafer_id"])
            groups.setdefault(ch, []).append(rec)
        for ch, group in sorted(groups.items()):
            with ENGINE.using(ENGINE.for_chamber(ch)):
                _prefill_batch(group, pause_check)
        return
    try:
        m = ENGINE.model
        if m is not None and hasattr(m, "predict_batch") and recs:
            try:
                m.predict_batch(recs, pause_check=pause_check)
            except TypeError:                             # 구버전 백엔드(콜백 미지원) 호환
                m.predict_batch(recs)
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()


@_wafer_model
def _assess(rec) -> Dict:
    """forecast 모델의 assess + n_flag_sensors 를 한 dict 로 묶는다."""
    a = ENGINE.model.assess(rec)
    return {
        "score": float(a["score"]),
        "n_flag_sensor": int(ENGINE.model.n_flag_sensors(rec)),
        "sensors": a.get("sensors") or [],
    }


def _write_predictions_csv(results: List[Dict], path: Path = None,
                           inspected_at: str = None) -> Path:
    """전체 웨이퍼 예측을 predictions.csv 로 저장(검사 때마다 갱신). 모델별 폴더."""
    path = path or _pred_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = inspected_at or STATE.get("inspected_at") or _now()
    cols = ["wafer_id", "chamber", "recipe", "verdict", "score", "threshold",
            "score_raw", "threshold_raw", "score_w_raw", "threshold_w", "verdict_w",
            "n_flag_sensor", "inspected_at"] + list(AUDIT_FIELDS)
    _wmode = _score_mode() == "weighted"
    _thrw_cache: Dict = {}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # 엑셀 한글 대비 BOM
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(results, key=lambda x: _natkey(x["wafer_id"])):
            # 배점 컬럼: S_w + (보정돼 있으면) θ_w·배점판정 — 엑셀에서 배점 결과 검증용
            sw = r.get("score_w_raw")
            rc = r.get("recipe") or ""
            thr_w = _thrw_cache.get(rc)
            if thr_w is None:
                thr_w = _score_thr_for(rc) or 0.0; _thrw_cache[rc] = thr_w
            try:
                _swf = float(sw)
            except (TypeError, ValueError):
                _swf = None
            vw = ""
            if _swf is not None and thr_w and thr_w > 1e-12:
                vw = "NG" if _swf >= thr_w else "OK"
            w.writerow({
                "wafer_id": r["wafer_id"], "chamber": r.get("chamber") or "",
                "recipe": r.get("recipe") or "",
                "verdict": r["verdict"],
                "score": r["score"], "threshold": r["threshold"],
                **{k: r.get(k) for k in AUDIT_FIELDS},
                "score_raw": r.get("score_raw"), "threshold_raw": r.get("threshold_raw"),
                "score_w_raw": (sw if sw is not None else ""),
                "threshold_w": (round(float(thr_w), 3) if thr_w else ""),
                "verdict_w": vw,
                "n_flag_sensor": r["n_flag_sensor"], "inspected_at": ts,
            })
    return path


CHAMBERS = [f"dsv{i}" for i in range(1, 7)]           # dsv1 ~ dsv6 (6개 챔버)


def _chamber(wafer_id) -> Optional[str]:
    """C2 metadata first; equipment-aware filename fallback for untagged files."""
    meta = _file_meta(wafer_id)
    if meta.get("chamber") and not _chamber_mapping_stale(wafer_id):
        return meta["chamber"]
    return _chamber_from_token(str(wafer_id or ""), wafer_id)


def _weighted_score_for_rec(rec) -> Optional[float]:
    """배점 모드 제거(배포판) — 더 이상 계산·기록하지 않는다(과거 npz 의 score_w_raw 는 읽기만)."""
    return None


def _weighted_view_results(results: List[Dict]) -> List[Dict]:
    """검사 결과(raw 저장본)를 '현재 모드' 기준 표시로 변환.

    배점 모드 + θ_w 보정 완료 + score_w_raw 존재 → S_w/θ_w 로 재판정한 사본 반환.
    (개요/산점도는 이력 relive 로 이미 배점 반영되는데, '개별 검사' 탭·검사 직후 응답은
    STATE 결과(raw)를 그대로 보여줘 서로 다른 OK/NG 가 보이던 불일치 해소.)
    raw 모드/미보정이면 원본 그대로(복사 없음)."""
    if _score_mode() != "weighted" or not results:
        return results
    out = []
    changed = False
    thr_cache: Dict = {}
    for r in results:
        if r.get("score_overridden"):
            out.append(r); continue
        sw = r.get("score_w_raw")
        rc = r.get("recipe") or ""
        try:
            sw = float(sw)
        except (TypeError, ValueError):
            sw = None
        thr_w = thr_cache.get(rc)
        if thr_w is None:
            thr_w = _score_thr_for(rc); thr_cache[rc] = thr_w
        if sw is None or not np.isfinite(sw) or not thr_w or thr_w <= 1e-12:
            out.append(r); continue                       # 배점 불가 → raw 유지
        nr = dict(r)
        nr["score"] = _floor2(sw / thr_w)
        nr["ratio"] = nr["score"]
        nr["threshold"] = 1.0
        nr["threshold_raw"] = math.floor(float(thr_w) * 1000) / 1000.0
        nr["verdict"] = "NG" if sw >= thr_w else "OK"
        nr["score_mode"] = "weighted"
        out.append(nr); changed = True
    return out if changed else results


def _norm_cdf(z: float) -> float:
    """표준정규 누적분포 Φ(z). 오탐률 α = 1-Φ(z) 표시용."""
    try:
        return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))
    except Exception:                                    # noqa: BLE001
        return float("nan")


@_wafer_model
def _result_for(rec) -> Dict:
    """한 wafer record → 결과 dict(점수·판정). RCA(주요원인)는 제외."""
    wid = rec["wafer_id"]
    s = _assess(rec)
    sv = _score_view(wid, float(s["score"]), rec)         # 임계=1 정규화(29a→1.1)
    sw = _weighted_score_for_rec(rec)                # 배점 점수(S_w) — 판정은 배점모드에서 사용
    return {
        "wafer_id": wid,
        "score": sv["score"],
        "threshold": sv["threshold"],
        "recipe": sv["recipe"],
        "verdict": sv["verdict"],
        **{k: sv.get(k) for k in AUDIT_FIELDS},
        "score_raw": sv["score_raw"],
        "threshold_raw": sv["threshold_raw"],
        "score_w_raw": (math.floor(sw * 1000) / 1000.0 if sw is not None else None),  # 배점 점수(원본)
        "n_flag_sensor": s["n_flag_sensor"],
        "n_bad_lines": int(rec.get("n_bad_lines") or 0),   # 깨진(skip된) 행 수 → UI 경고표시
        "chamber": rec.get("chamber") or _chamber(wid),
        "model_chamber": rec.get("model_chamber"),
        "lot": _lot_of(wid),                         # .log I2 해시 — 산포 추세를 LOT 단위로 묶는 키
        "ratio": sv["score"],                        # 표시용(=정규화 점수)
        "end_ts": _end_ts_and_date(wid)[0],          # 파일명 끝시각(상세 조회 날짜용)
        "n_tp": _ntp_from_rec(rec),                  # 점수 구간 길이(타임포인트 정규화 모드용)
    }


def _merge_results(new_results: List[Dict], rec_map: Dict, folder: Path):
    """STATE 결과에 새 결과를 병합(같은 wafer_id 는 갱신) 후 점수 내림차순 정렬."""
    with STATE_LOCK:
        by_id = {r["wafer_id"]: r for r in STATE["results"]} if STATE.get("result_model") in (None, _model_id()) else {}
        for r in new_results:
            by_id[r["wafer_id"]] = r
        merged = sorted(by_id.values(), key=lambda r: _natkey(r["wafer_id"]))
        STATE["results"] = merged
        STATE["recs"].update(rec_map)
        STATE["folder"] = str(folder)
        STATE["inspected_at"] = _now()
        STATE["result_model"] = _model_id()
        _trim_recs()                                      # 최근 N개만 유지(자동검사 누적 방지)


# ── 원본 .log/.tsv(탭 구분) → .csv 자동 변환 ──
#   설비 원본은 탭 구분 .log(엑셀에서 열림). 검사 전 같은 이름 .csv 로 변환.
#   · ProcessStep 이 숫자가 아닌 행(헤더/구분행) 제거
#   · 마지막 탭으로 생기는 더미열(Unnamed) 제거
#   컬럼 선택/정렬은 하지 않는다(모델이 센서 '이름'으로 알아서 고름).
LOG_EXTS = (".log", ".tsv")


def _convert_log_to_csv(src: Path, dst: Path) -> None:
    import pandas as pd
    df = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):           # 인코딩 폴백(한글 Windows 대비)
        try:
            df = pd.read_csv(src, sep="\t", encoding=enc)
            break
        except UnicodeDecodeError:
            df = None
        except Exception:                                 # noqa: BLE001 (파서 문제는 아래서 처리)
            df = None
            break
    if df is None:
        df = pd.read_csv(src, sep="\t", engine="python")  # 최후 폴백
    if "ProcessStep" in df.columns:                       # 숫자 아닌 행 제거
        df = df[pd.to_numeric(df["ProcessStep"], errors="coerce").notna()]
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]   # 트레일링 탭 더미열 제거
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():                                      # 이전 크래시로 남은 부분파일 정리
        try:
            tmp.unlink()
        except OSError:
            pass
    df.to_csv(tmp, sep=",", index=False)
    tmp.replace(dst)                                      # 원자적 교체(부분 파일 방지)


def _convert_logs_in(folder: Path, min_age: float = 0.0, skip_stems=None,
                     progress_cb=None) -> List[Path]:
    """★ .log 직접 추론 전환: 더 이상 csv 로 변환하지 않는다(3중 I/O 제거 → '준비 중' 단축).
    이 함수는 신규 .log/.tsv 의 '태깅만' 수행: 레시피(J2)·종료시각(A/B 마지막행)·챔버(C2).
    min_age>0 이면 그만큼 정지(복사 완료)된 파일만. skip_stems 는 이미 처리한 스템.
    반환=이번에 태깅한 원본 파일 목록."""
    tagged: List[Path] = []
    _cell_tag = _lot_cell_tag()                           # 이번 태깅이 쓸 LOT 칸
    skip = set(skip_stems or ())
    try:
        # ★ 하위폴더 포함(rglob): '모든 데이터가 든 통합 폴더'를 월별 등 하위폴더로 정리해도
        #   태깅이 전부 되도록. (예전 iterdir(최상위만)은 하위폴더 파일이 검사 목록(rglob)에는
        #   잡히는데 태깅이 안 돼 metadata 가 비고 → 기간/챔버/유형 필터가 그 파일들을 제외했음)
        srcs = [p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in LOG_EXTS]
    except Exception:                                     # noqa: BLE001
        return tagged
    now = time.time()
    # 태깅 대상만 선별(이미 태깅/아직 쓰는 중 제외)
    todo = []
    for src in sorted(srcs):
        _fm = _file_meta(src.stem)
        # ★ LOT 칸을 아직 안 봤거나, 셀 설정이 바뀐(lot_c 불일치) 파일은 이미 태깅됐어도 다시 읽는다.
        #   (LOT 기능 추가 이전 이력 보강 + 셀 위치 변경 시 자동 재추출 — 재검사 불필요)
        if (src.stem in skip or (_fm.get("end_ts") and _fm.get("lot_c") == _cell_tag
                                 and _file_meta_fresh(src, _fm))) and _fm.get("chamber_cell") == _chamber_cell_tag(src.stem):
            continue
        if min_age and (now - src.stat().st_mtime) < min_age:
            continue
        todo.append(src)
    if not todo:
        return tagged
    if not _FILE_META:
        _load_file_meta()
    if not _RECIPE_TAGS:
        _load_recipe_tags()

    _prep_done = [0]

    def _extract(src):
        """파일 하나에서 (레시피·챔버·종료시각) 읽기만(쓰기 없음) → 병렬 안전."""
        _pause_gate()                                     # 준비중에도 일시중단 반영
        if progress_cb is not None:                       # ★ 파일 준비 진행 표시(대량 유입 시)
            _prep_done[0] += 1
            try:
                progress_cb(_prep_done[0], len(todo))
            except Exception:                             # noqa: BLE001
                pass
        try:
            rc = _extract_recipe_from_log(src)
            ch = _extract_chamber_from_log(src)
            lot = _extract_lot_from_log(src)              # I2 — 산포 추세를 LOT 단위로 묶는 키
            edt = _extract_end_dt_from_file(src)
            if edt is None:
                csvf = folder / f"{src.stem}.csv"
                if csvf.exists():
                    edt = _extract_end_dt_from_file(csvf)
            return (src, _clean_recipe(rc) if rc else None, ch,
                    edt.isoformat(timespec="seconds") if edt else None, lot)
        except Exception:                                 # noqa: BLE001
            return (src, None, None, None, None)

    # ★ 파일별 읽기를 스레드로 병렬(I/O 바운드). 쓰기는 아래서 '1회'만.
    try:
        _w = int(os.environ.get("IO_LOAD_WORKERS") or 0)
    except (TypeError, ValueError):
        _w = 0
    if _w <= 0:
        _w = _cpu_workers(8)
    _w = max(1, min(_w, len(todo)))

    # ★ 배치 단위 처리 + '중간 저장': 대량 폴더(수만 장) 첫 태깅이 수 분 걸릴 수 있는데,
    #   예전엔 전부 끝난 뒤 1회만 저장해서 도중에 중단/크래시 나면 진행분이 전부 소실됐다.
    #   → 배치(기본 500)마다 태그 캐시 flush → 어디서 끊겨도
    #   그때까지 태깅분은 보존, 다음 실행은 남은 파일만 이어서 태깅.
    try:
        BATCH = max(50, int(os.environ.get("IO_TAG_BATCH", "500")))
    except (TypeError, ValueError):
        BATCH = 500
    n_rc = n_lot = 0
    from concurrent.futures import ThreadPoolExecutor
    for bi in range(0, len(todo), BATCH):
        part = todo[bi:bi + BATCH]
        if _w == 1 or len(part) == 1:
            results = [_extract(s) for s in part]
        else:
            with ThreadPoolExecutor(max_workers=min(_w, len(part))) as ex:
                results = list(ex.map(_extract, part))
        # 메모리 dict 에 모아서 recipe_tags.json / file_meta.json 을 '배치당 1회'만 저장
        with _RECIPE_TAGS_LOCK:
            for src, rc, ch, ets, lot in results:
                if rc:
                    _RECIPE_TAGS[str(src.stem)] = rc; n_rc += 1
        with _FILE_META_LOCK:
            for src, rc, ch, ets, lot in results:
                cur = dict(_FILE_META.get(str(src.stem), {}))
                if ets:
                    cur["end_ts"] = ets; cur["ets_v"] = 2
                cur["chamber"] = ch or ""
                cur["chamber_cell"] = _chamber_cell_tag(src.stem)
                if lot:
                    cur["lot"] = lot; n_lot += 1
                if not lot:
                    cur.pop("lot", None)                  # 칸이 바뀌어 값이 없어졌으면 옛 값도 지운다
                cur["lot_v"] = 1                          # LOT 칸을 봤다는 표시(값이 없어도 재읽기 방지)
                cur["lot_c"] = _cell_tag                  # 어느 칸으로 읽었는지 — 바뀌면 다시 읽는다
                try:
                    cur["mt"] = int(src.stat().st_mtime_ns)   # 파일 변경 감지용
                except Exception:                         # noqa: BLE001
                    pass
                if cur:
                    _FILE_META[str(src.stem)] = cur
                tagged.append(src)
        _flush_recipe_tags()
        _flush_file_meta()
        if len(todo) > BATCH:                             # 대량일 때만 중간 metadata + 진행 로그
            try:
                _folder_metadata_update(folder, _data_files(_resolve_existing_dir_ci(folder)))
            except Exception:                             # noqa: BLE001
                traceback.print_exc()
            print(f"[tag] 중간저장 {min(bi + BATCH, len(todo))}/{len(todo)}", flush=True)
    print(f"[tag] {len(tagged)}개 태깅(병렬 {_w}스레드·배치쓰기) · recipe {n_rc} · LOT {n_lot}", flush=True)
    return tagged


def _delete_converted_csv(folder: Path, stems) -> int:
    """검사 끝난 '변환본' .csv 만 삭제(같은 이름 .log/.tsv 원본이 있는 것만).
    원본(.log/.tsv)과 사용자가 직접 넣은 .csv 는 유지. 반환=삭제 건수."""
    n = 0
    for st in set(str(s) for s in stems):
        csv = folder / f"{st}.csv"
        if not csv.exists():
            continue
        if any((folder / f"{st}{e}").exists() for e in LOG_EXTS):   # 원본 있는 변환본만
            try:
                csv.unlink(); n += 1
            except Exception:                             # noqa: BLE001
                pass
    return n


# ── 센서별 스텝 배점(가중 복원오차) 엔진 ───────────────────────────────
#   센서마다 '연관 IO 밸브' 불리언식이 참인 스텝은 복원오차 ×1.1, 아니면 ×0.9.
#   판정 점수 = Σ w·max(0,level) (모델 점수 정의와 동일, 가중만 추가). |=OR &=AND.
_SCORE_VALVE_RULES_DEFAULT = {
    "t_005_PV":    "9000002 & (9000006 | 9000007 | 9000008) & (9000003 | 9000004)",
    "t_002_PV":     "9000003 | 9000004",
    "t_001_PV":       "9000004",
    "t_003_PV":    "9000009",
    "t_004_PV": "9000010",
    "p_005_PV":    "9000002 & (9000006 | 9000007 | 9000008) & (9000003 | 9000004)",
    "p_006_PV": "9000002 & (9000006 | 9000007 | 9000008) & (9000003 | 9000004)",
    "p_008_PV":   "9000003",
    "p_007_PV":"9000004",
    "p_003_PV":    "9000009",
    "p_004_PV": "9000010",
}
_SCORE_W_MATCH_DEFAULT = 2.0                              # 밸브 조건 일치 구간 배율(사용자 조정 가능)
_SCORE_W_OTHER_DEFAULT = 0.5                              # 불일치 구간 배율(사용자 조정 가능)
_VALVE_RULE_CACHE: Dict = {}


def _score_valve_rules() -> Dict:
    """센서→밸브 불리언식(설정 override 가능, 없으면 기본)."""
    cfg = _load_app_config().get("score_valve_rules")
    if isinstance(cfg, dict) and cfg:
        return {str(k): str(v) for k, v in cfg.items() if str(v).strip()}
    return dict(_SCORE_VALVE_RULES_DEFAULT)


def _score_weights() -> tuple:
    cfg = _load_app_config()
    try:
        wm = float(cfg.get("score_w_match", _SCORE_W_MATCH_DEFAULT))
    except (TypeError, ValueError):
        wm = _SCORE_W_MATCH_DEFAULT
    try:
        wo = float(cfg.get("score_w_other", _SCORE_W_OTHER_DEFAULT))
    except (TypeError, ValueError):
        wo = _SCORE_W_OTHER_DEFAULT
    return wm, wo


def _needed_valves() -> set:
    vs = set()
    for expr in _score_valve_rules().values():
        vs.update(re.findall(r"\d+", str(expr)))
    return vs


def _compile_valve_rule(expr: str):
    """불리언식 → 함수(open_map, T)->bool[T]. &(AND)가 |(OR)보다 우선. 결과 캐시."""
    key = str(expr)
    if key in _VALVE_RULE_CACHE:
        return _VALVE_RULE_CACHE[key]
    toks = re.findall(r"\d+|[()&|]", key)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat():
        t = peek(); pos[0] += 1; return t

    def p_or():
        node = p_and()
        while peek() == "|":
            eat(); node = ("or", node, p_and())
        return node

    def p_and():
        node = p_atom()
        while peek() == "&":
            eat(); node = ("and", node, p_atom())
        return node

    def p_atom():
        t = peek()
        if t == "(":
            eat(); node = p_or()
            if peek() == ")":
                eat()
            return node
        eat(); return ("valve", t)

    try:
        tree = p_or()
    except Exception:                                    # noqa: BLE001 (식 오류 → 항상 False)
        tree = ("valve", "__none__")

    def ev(node, om, T):
        if node[0] == "valve":
            return om.get(node[1], np.zeros(T, dtype=bool))
        a = ev(node[1], om, T); b = ev(node[2], om, T)
        return (a & b) if node[0] == "and" else (a | b)

    fn = lambda om, T: ev(tree, om, T)                   # noqa: E731
    _VALVE_RULE_CACHE[key] = fn
    return fn


def _valve_open_map(rec) -> tuple:
    """rec → ({밸브ID: bool[T] 열림여부}, T). 열림 = 그 시점 io 값 > 0."""
    io = np.asarray(rec.get("io", []), dtype=np.float64)
    cols = [str(c) for c in rec.get("io_cols", [])]
    if io.ndim != 2 or not cols or io.shape[0] == 0:
        return {}, 0
    T = io.shape[0]
    idx = {c: k for k, c in enumerate(cols)}
    om: Dict = {}
    for v in _needed_valves():
        k = idx.get(v)
        om[v] = (np.nan_to_num(io[:, k]) > 0.0) if k is not None else np.zeros(T, dtype=bool)
    return om, T


def _weight_matrix_from_open(om, T, sensor_cols):
    """열림맵(om)+길이(T) → (T, n_sensors) 가중치. 규칙 참=1.1/거짓=0.9/규칙없음=1.0."""
    if not T:
        return None
    rules = _score_valve_rules()
    wm, wo = _score_weights()
    W = np.ones((int(T), len(sensor_cols)), dtype=np.float64)
    for j, s in enumerate(sensor_cols):
        expr = rules.get(str(s))
        if not expr:
            continue
        cond = _compile_valve_rule(expr)(om, int(T))
        if getattr(cond, "shape", (0,))[0] == int(T):
            W[:, j] = np.where(cond, wm, wo)
    return W


def _score_weight_matrix(rec, sensor_cols):
    """(T, n_sensors) 가중치 행렬(검사 시 rec io 기반). io 없으면 None."""
    om, T = _valve_open_map(rec)
    return _weight_matrix_from_open(om, T, sensor_cols) if T else None


# ── 배점 모드 & 배점 임계(θ_w) ─────────────────────────────────────────
def _score_mode() -> str:
    """배점(가중) 모드는 배포판에서 제거됨 — 항상 raw. (설정 UI·API 삭제, 잔존 호출부 호환용)"""
    return "raw"


def _score_thr_for(recipe):
    """레시피의 배점 임계 θ_w(없으면 접두매칭 → 글로벌). 미보정이면 None."""
    cfg = _load_app_config()
    d = cfg.get("score_thresholds") or {}
    rc = (_recipe_display(recipe) or "").strip()
    if rc in d:
        try:
            return float(d[rc])
        except (TypeError, ValueError):
            pass
    rl = rc.lower()
    best, blen = None, -1
    for k, v in d.items():
        kl = str(k).lower()
        if kl and (rl == kl or rl.startswith(kl)) and len(kl) > blen:
            try:
                best, blen = float(v), len(kl)
            except (TypeError, ValueError):
                pass
    if best is not None:
        return best
    g = cfg.get("score_threshold_global")
    try:
        return float(g) if g is not None else None
    except (TypeError, ValueError):
        return None


def _weighted_score_from_level(level, W) -> float:
    """S_w = Σ w·max(0,level). W=None 이면 raw(Σ max(0,level))."""
    lv = np.maximum(0.0, np.nan_to_num(np.asarray(level, dtype=np.float64)))
    if W is None or getattr(W, "shape", None) != lv.shape:
        return float(lv.sum())
    return float((np.asarray(W, dtype=np.float64) * lv).sum())


# ── 폴더 메타데이터(metadata.csv): 파일명·설비·챔버·레시피·끝시간 ─────────────
def _tool_from_name(stem: str) -> str:
    """설비명 = 파일명 첫 토큰. '*' 구분 우선(예 WUK370A*2026...), 없으면 '_' 구분.
    (Windows 파일명엔 '*' 가 금지라 실제로는 '_' 구분이 대부분 — 둘 다 지원)"""
    s = str(stem or "")
    for sep in ("*", "_"):
        if sep in s:
            return s.split(sep, 1)[0]
    return s


def _folder_metadata_update(folder: Path, files) -> Dict:
    """폴더 파일들의 메타(설비·챔버·레시피·LOT·끝시간) → {stem: meta} 반환.

    ★ 배포판: 어떤 CSV 파일도 만들지 않는다(사용자 결정 — 고객사 제어기·앱 폴더
      모두). 태그 캐시(_file_meta/_recipe_tag — 앱 데이터폴더의 파일 속성 저장소)
      에서 매번 메모리로 조립한다. 설비/챔버/유형/기간 필터가 이 dict 를 쓴다.
    예전 버전이 만든 metadata.csv(데이터셋 폴더·앱 폴더 모두)는 발견 즉시 지운다.
    """
    folder = Path(folder)
    for legacy in (folder / "metadata.csv",):
        try:
            if legacy.exists():
                legacy.unlink()
                print(f"[metadata] {folder.name}: 구버전 metadata.csv 제거(파일 미생성 정책)",
                      flush=True)
        except Exception:                                 # noqa: BLE001 — 읽기전용이면 그대로 둔다
            pass
    rows: Dict[str, Dict] = {}
    for f in list(files or []):
        st = f.stem
        fm = _file_meta(st)
        rows[st] = {"file": f.name,
                    "tool": _tool_from_name(st),
                    "chamber": str((_chamber(st) if _is_wuk380b(st) or _chamber_mapping_stale(st) else fm.get("chamber")) or ""),
                    "recipe": str(_recipe_tag(st) or ""),
                    "lot": str(fm.get("lot") or ""),
                    "end_ts": str(fm.get("end_ts") or "")}
    return rows


# ── 검사 캐시 ────────────────────────────────────────────────────────
#   ① 파싱 캐시(data._parsed_frame, 모델 무관): .log→배열 변환 결과 npz — 모델을 바꿔도 유효.
#   ② 점수 캐시(아래): 같은 모델·같은 파일이면 추론 생략, 이전 raw 점수 재사용.
#      키에 '모델 서명'이 들어가 모델이 바뀌면 자동 무효 → '이미 검사한 웨이퍼 스킵'과 달리
#      모델 교체 워크플로에서도 안전. 판정/배수는 raw 점수에서 매번 재계산(기존 정책 유지).
SCORE_CACHE_DIR = DATA_DIR / "score_cache"
_SCORE_CACHE_MEM: Dict = {"sig": None, "rows": {}}
_SCORE_CACHE_LOCK = threading.Lock()


def _inspect_cache_enabled() -> bool:
    """★ 배포판: 검사 캐시(점수 재사용·파싱 캐시) 전면 비활성 — 항상성 우선.
    캐시로 옛 점수가 재사용되는 경로 자체를 없앤다(사용자 결정). 항상 False."""
    return False

def _apply_cache_env() -> None:
    """파싱 캐시 폴더를 data 모듈에 env 로 전달(설정 토글 반영). 로드 직전마다 호출."""
    try:
        if _inspect_cache_enabled():
            d = DATA_DIR / "parse_cache"
            d.mkdir(parents=True, exist_ok=True)
            os.environ["IO_PARSE_CACHE_DIR"] = str(d)
        else:
            os.environ.pop("IO_PARSE_CACHE_DIR", None)
    except Exception:                                     # noqa: BLE001
        pass


def _model_sig(art=None) -> str:
    """모델 서명 = 아티팩트 폴더의 모델 파일들(이름·크기·mtime) + 센서별칭 설정."""
    try:
        d = Path(art if art is not None else ENGINE.root.art)
        parts = [str(d), "onnx-normalization-v1", "wafer-score-override-v1", "cylinder-input-v1"]
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".onnx", ".pt", ".json", ".npz"):
                st = p.stat()
                parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
        parts.append(json.dumps(_load_app_config().get("sensor_aliases") or {}, sort_keys=True))
        return hashlib.md5("|".join(parts).encode("utf-8", "ignore")).hexdigest()[:16]
    except Exception:                                     # noqa: BLE001
        return ""


def _rules_sig() -> str:
    """배점 규칙 서명(밸브 규칙 + 배율) — 바뀌면 S_w 캐시 무효(그 웨이퍼만 재추론)."""
    try:
        wm, wo = _score_weights()
        return hashlib.md5(json.dumps([_score_valve_rules(), wm, wo],
                                      sort_keys=True).encode()).hexdigest()[:12]
    except Exception:                                     # noqa: BLE001
        return ""


def _score_cache_path(sig: str) -> Path:
    return SCORE_CACHE_DIR / f"{sig}.jsonl"


def _score_cache_load(sig: str) -> Dict:
    """모델 서명별 점수 캐시(JSONL, 뒤 라인이 이김) → {stem: entry}. 메모리 1회 로드."""
    with _SCORE_CACHE_LOCK:
        if _SCORE_CACHE_MEM.get("sig") == sig:
            return dict(_SCORE_CACHE_MEM["rows"])
    rows: Dict[str, Dict] = {}
    try:
        p = _score_cache_path(sig)
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        e = json.loads(ln)
                        if e.get("w"):
                            rows[str(e["w"])] = e
                    except Exception:                     # noqa: BLE001
                        continue
    except Exception:                                     # noqa: BLE001
        rows = {}
    with _SCORE_CACHE_LOCK:
        _SCORE_CACHE_MEM["sig"] = sig
        _SCORE_CACHE_MEM["rows"] = rows
    return dict(rows)


def _score_cache_put(sig: str, rec, r: Dict, rsig: str) -> None:
    """검사 직후 캐시에 1행 append. raw 점수는 버림 전 전정밀(경계 판정 보존)."""
    try:
        f = Path(rec.get("source_path") or "")
        st = f.stat()
        sraw = float(ENGINE.model.assess(rec)["score"])   # _pred_std_cache 재사용 — 저렴
        e = {"w": r["wafer_id"], "mt": int(st.st_mtime_ns), "sz": int(st.st_size),
             "s": sraw, "nf": r.get("n_flag_sensor"), "sw": r.get("score_w_raw"),
             "rs": rsig, "nb": r.get("n_bad_lines"), "cm": _chamber_cell_tag(r["wafer_id"]),
             "d": _end_ts_and_date(r["wafer_id"])[1]}
        with _SCORE_CACHE_LOCK:
            SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(_score_cache_path(sig), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
            if _SCORE_CACHE_MEM.get("sig") == sig:
                _SCORE_CACHE_MEM["rows"][e["w"]] = e
    except Exception:                                     # noqa: BLE001
        pass


def _match_prefixes(value: str, prefixes) -> bool:
    """체크된 prefix 중 하나라도 value(소문자) 의 접두면 True. prefixes 비면 전체 허용."""
    if not prefixes:
        return True
    v = str(value or "").lower()
    return any(v.startswith(str(p).lower()) for p in prefixes if str(p).strip())


def _match_chamber(value: str, prefixes, equipment=None) -> bool:
    """챔버 prefix 매칭 — DSV↔PMCUNIT 별칭 상호 인식.

    메타의 챔버는 정규화된 'dsv5' 인데, 로그 원문(C2)은 'PMCUNIT 11' 식이라 사용자가
    'PMCUNIT11'/'PMCUNIT 11' 로 등록해도 매칭되도록 dsvN↔pmcunit(N+6) 별칭을 함께 비교.
    prefix 쪽은 공백/따옴표 제거 후 소문자 비교."""
    if not prefixes:
        return True
    v = str(value or "").lower()
    cands = [v]
    m = re.fullmatch(r"dsv([1-6])", v)
    if m:
        cands.append(f"pmcunit{int(m.group(1)) * 2 if _is_wuk380b(equipment) else int(m.group(1)) + 6}")     # dsv5 → pmcunit11
    norm = [re.sub(r"[\s'\"]+", "", str(p).lower()) for p in prefixes if str(p).strip()]
    return any(c.startswith(p) for c in cands for p in norm if p)


def _peek_recipe(path) -> str:
    """검사 전에 파일의 '유형(레시피)'을 싸게 알아낸다(전체 로드 없이 앞 몇 줄만).

    우선순위: .log/.tsv → J2 셀 → 태깅 캐시 → 파일명 토큰. 표시명(_recipe_display)으로 정규화.
    유형 체크박스 스캔·'선택 유형만 검사' 필터에 사용.
    """
    p = Path(path)
    rc = None
    if p.suffix.lower() in (".log", ".tsv"):
        try:
            rc = _extract_recipe_from_log(p)
        except Exception:                                # noqa: BLE001
            rc = None
    if not rc:
        rc = _recipe_tag(p.stem)                          # 이전 검사 때 저장된 태그
    if not rc:
        try:
            rc = _recipe_base_decision(p.stem)[0]         # 파일명 토큰 기반 인식
        except Exception:                                # noqa: BLE001
            rc = None
    return _recipe_display(rc) if rc else "unknown"


def _peek_recipes_map(files) -> Dict:
    """파일 리스트 → {stem: 표시유형} (병렬 peek). 대량 폴더에서도 빠르게."""
    files = list(files)
    out: Dict = {}
    if not files:
        return out
    try:
        from concurrent.futures import ThreadPoolExecutor
        _w = _cpu_workers(8, 2)
        with ThreadPoolExecutor(max_workers=_w) as ex:
            for f, rc in zip(files, ex.map(_peek_recipe, files)):
                out[Path(f).stem] = rc
    except Exception:                                    # noqa: BLE001
        for f in files:
            out[Path(f).stem] = _peek_recipe(f)
    return out


@_pin_model
def run_inspection(folder: Path, replace_results: bool = True, chamber=None,
                   trigger: str = "",
                   seen_stems: set = None, recipe_filter: set = None,
                   recipe_prefixes: set = None, tool_prefixes: set = None,
                   chamber_prefixes: set = None, date_from: str = None,
                   date_to: str = None) -> Dict:
    """폴더의 모든 wafer CSV 추론 → 결과 리스트. recs 는 STATE 에 보관 + 이력 기록.
    .log/.tsv(탭 구분) 원본이 있으면 먼저 .csv 로 자동 변환한다.
    chamber 지정시(챔버폴더 검사) 결과를 그 챔버로 귀속(파일명 무관).
    seen_stems 지정 시(다중 폴더 검사) 이미 다른 폴더에서 처리한 스템은 건너뛴다(폴더 간 중복 제거)."""
    print(f"[inspect] 검사 시작 — 폴더={folder} · 출처={trigger or '수동(직접 호출)'}",
          flush=True)
    ENGINE.load()
    if not ENGINE.ready:
        raise RuntimeError(ENGINE.error or "모델 미로드")
    if not folder.exists():
        raise FileNotFoundError(f"검사 폴더 없음: {folder}")

    _set_progress(True, 0, 0, "파일 확인", str(folder))     # .log 직접 읽기(변환 없음) — 태깅만
    _convert_logs_in(folder)                              # .log/.tsv → .csv
    # ★ 대량(예: 6000장) 검사 대비: 전체를 한 번에 메모리로 올리지 않고 '청크 단위'로
    #   로드→추론→스트리밍 기록→해제 반복 → 피크 메모리를 청크 크기로 제한.
    _files = _data_files(_resolve_existing_dir_ci(Path(folder)))
    all_stems = [f.stem for f in _files]
    _n_unique0 = len(all_stems)                           # 필터/폴더간중복 제외 '전' 유니크 수(처리요약용)
    _n_xdup = 0
    _n_filtered = 0
    if seen_stems is not None:                            # 폴더 간 중복(같은 파일명) 제거 → 한 번만 검사
        _dup = [s for s in all_stems if s in seen_stems]
        if _dup:
            print(f"[검사] 다른 폴더 중복 {len(_dup)}매 제외(같은 파일명): {folder}", flush=True)
        _n_xdup = len(_dup)
        all_stems = [s for s in all_stems if s not in seen_stems]
        seen_stems.update(all_stems)
    # ★ 폴더 메타데이터(메모리): 태그에서 조립 — 파일은 만들지 않는다(배포판).
    #   필터(설비/챔버/유형/기간)의 조회 소스로도 사용(파일 재파싱 없이 빠름).
    _fmeta = _folder_metadata_update(folder, _files)
    if (recipe_filter or recipe_prefixes or tool_prefixes or chamber_prefixes
            or date_from or date_to):                    # ★ 조건 만족 파일만 검사(카테고리 간 AND)
        before = len(all_stems)
        _fmap = {f.stem: f for f in _files}
        _rdisp_cache: Dict = {}

        def _rdisp(s):                                   # 유형 표시명(메타 → 필요시 peek 폴백)
            if s in _rdisp_cache:
                return _rdisp_cache[s]
            rc = (_fmeta.get(s) or {}).get("recipe") or ""
            d = _recipe_display(rc) if rc else (_peek_recipe(_fmap[s]) if s in _fmap else "unknown")
            _rdisp_cache[s] = d
            return d

        kept = []
        for s in all_stems:
            m = _fmeta.get(s) or {}
            if tool_prefixes and not _match_prefixes(m.get("tool") or _tool_from_name(s), tool_prefixes):
                continue                                  # 설비 prefix
            if chamber_prefixes and not _match_chamber(m.get("chamber"), chamber_prefixes, s):
                continue                                  # 챔버 prefix (C2 기반, DSV↔PMCUNIT 별칭 인식)
            if recipe_filter and _rdisp(s) not in {str(x) for x in recipe_filter}:
                continue                                  # 유형 정확일치(개별검사 '유형 스캔')
            if recipe_prefixes and not _match_prefixes(_rdisp(s), recipe_prefixes):
                continue                                  # 유형 prefix
            if date_from or date_to:                      # 검사기간(끝시간 날짜 기준, 미상=제외)
                ets = str(m.get("end_ts") or "")[:10]
                if not ets:
                    continue
                if (date_from and ets < str(date_from)) or (date_to and ets > str(date_to)):
                    continue
            kept.append(s)
        all_stems = kept
        skipped = before - len(all_stems)
        _n_filtered = skipped
        if skipped:
            print(f"[검사] 필터 외 {skipped}매 건너뜀(설비 {sorted(tool_prefixes or [])} · "
                  f"챔버 {sorted(chamber_prefixes or [])} · 유형 {sorted(recipe_prefixes or list(recipe_filter or []))} · "
                  f"기간 {date_from or ''}~{date_to or ''}): {folder}", flush=True)
    if not all_stems:
        diag = {"files": _n_unique0, "unique": _n_unique0, "inspected": 0,
                "dup_excluded": _n_xdup, "csv_excluded": 0,
                "filtered_excluded": _n_filtered, "failed": 0,
                "cache_hits": 0, "fail_reasons": [],
                "chambers_found": sorted({m.get("chamber") for m in _fmeta.values()
                                          if m.get("chamber")}),
                "chamber_unknown": sum(not m.get("chamber") for m in _fmeta.values())}
        _set_progress(False, 0, 0, "완료", str(folder))
        with STATE_LOCK:
            STATE["last_diag"] = diag
            if replace_results:
                STATE["results"] = []; STATE["recs"] = {}; STATE["folder"] = str(folder)
        return {"folder": str(folder), "n": 0, "results": [], "summary": _summary([]),
                "diag": diag}

    # ── ★ 점수 캐시(같은 모델 재검사 가속): 모델서명·파일(mtime/size)·배점규칙이 모두 같고
    #   상세보기용 아카이브(npz)가 남아 있으면 로드/추론 없이 이전 raw 점수를 재사용.
    #   판정·배수·배점 임계는 raw 에서 매번 재계산되므로 표시는 항상 현재 설정 기준.
    _cache_on = _inspect_cache_enabled()
    _msig = _model_sig() if _cache_on else ""
    _rsig = _rules_sig()
    _hits = []                                            # [(stem, cache entry)]
    if _cache_on and _msig:
        _cmap = _score_cache_load(_msig)
        if _cmap:
            _fmap2 = {f.stem: f for f in _files}
            _miss = []
            for s in all_stems:
                e = _cmap.get(s)
                f = _fmap2.get(s)
                ok = False
                if e is not None and f is not None:
                    try:
                        st = f.stat()
                        ok = (int(e.get("mt") or -1) == st.st_mtime_ns
                              and int(e.get("sz") or -1) == st.st_size
                              and (e.get("sw") is None or e.get("rs") == _rsig)
                              and e.get("cm", "C2") == _chamber_cell_tag(s)
                              and bool(e.get("d"))
                              and _archive_path(str(e["d"]), s).exists())
                    except OSError:
                        ok = False
                if ok:
                    _hits.append((s, dict(e, source_path=str(f))))
                else:
                    _miss.append(s)
            if _hits:
                print(f"[검사] 점수캐시 적중 {len(_hits)}/{len(all_stems)}매"
                      f"(같은 모델·같은 파일 → 추론 생략)", flush=True)
            all_stems = _miss

    if ENGINE.root.chamber_engines:
        all_stems.sort(key=lambda wid: (_file_meta(wid).get("chamber") or "", _natkey(wid)))
    CHUNK = max(1, int(os.environ.get("IO_INSPECT_CHUNK", "200")))
    total = len(all_stems) + len(_hits)
    results = []
    io_cols = sensor_cols = None
    cov = None
    fail_reasons: Dict[str, Dict] = {}                    # 판정 실패 사유 집계(유형→개수·예시)
    _set_progress(True, 0, total, "검사", str(folder))
    done = 0
    try:
        for s, e in _hits:                                # 캐시 적중분: 점수 재사용(파일도 안 읽음)
            _pause_gate()
            try:
                sv = _score_view(s, float(e["s"]), {"wafer_id": s, "source_path": e.get("source_path")})
                r = {"wafer_id": s, "score": sv["score"], "threshold": sv["threshold"],
                     "recipe": sv["recipe"], "verdict": sv["verdict"],
                     **{k: sv.get(k) for k in AUDIT_FIELDS},
                     "score_raw": sv["score_raw"], "threshold_raw": sv["threshold_raw"],
                     "score_w_raw": e.get("sw"),
                     "n_flag_sensor": int(e.get("nf") or 0),
                     "n_bad_lines": int(e.get("nb") or 0),
                     "chamber": _chamber(s) or chamber,
                     "ratio": sv["score"],
                     "end_ts": _end_ts_and_date(s)[0]}
                results.append(r)
                _append_history([r])                      # 이력은 (date,wafer) 최신 1건으로 dedup 됨
            except Exception as e2:                       # noqa: BLE001
                key = f"{type(e2).__name__}: {str(e2)[:120]}"
                ent = fail_reasons.setdefault(key, {"count": 0, "sample": s})
                ent["count"] += 1
            done += 1
            if done % 200 == 0 or done == total:
                _set_progress(True, done, total, "검사(캐시)", str(folder))
        if _hits:
            _wait_if_paused(done, total, str(folder))     # 캐시 구간에도 일시중단/중단 반영
        for ci in range(0, len(all_stems), CHUNK):
            _wait_if_paused(done, total, str(folder))     # 일시중단 요청 시 여기서 대기(청크 경계)
            _wait_if_low_memory(str(folder))              # ★ 메모리 부족이면 죽지 말고 여기서 대기
            chunk = set(all_stems[ci:ci + CHUNK])
            recs, io_c, sen_c, cov_c = _load_aligned(folder, stems=chunk)
            if io_cols is None and recs:                  # 표시용 스키마/커버리지는 첫 유효 청크 기준
                io_cols, sensor_cols, cov = io_c, sen_c, cov_c
            _prefill_batch(recs, pause_check=_pause_gate)  # ★ 청크 배치 순전파(배치마다 일시중단 확인→GPU 유휴화)
            chunk_rec_map = {}
            for rec in recs:
                _pause_gate()
                try:                                      # 한 장이 실패해도 배치 전체가 멈추지 않게(자동검사와 동일)
                    r = _result_for(rec)
                    _check_inspect_cancelled()
                    if chamber and not r.get("chamber"):  # .log C2 챔버 우선, 폴더 귀속은 폴백
                        r["chamber"] = chamber
                    results.append(r)
                    chunk_rec_map[rec["wafer_id"]] = rec
                    # ★ 스트리밍: 웨이퍼 한 장이 끝나는 즉시 이력 append + 아카이브 →
                    #   대시보드(이력 기반 산점도·목록·개요)가 검사 중에도 실시간으로 채워진다.
                    _append_history([r])
                    _archive_wafer(rec, r)
                    if _cache_on and _msig:               # ★ 점수 캐시 기록(다음 재검사 때 추론 생략)
                        _score_cache_put(_msig, rec, r, _rsig)
                    rec.pop("_level_cache", None)         # STATE 보관 rec 에 level 캐시 잔류 방지(RAM)
                except Exception as e:                    # noqa: BLE001 — 왜 실패했는지 사유 집계
                    key = f"{type(e).__name__}: {str(e)[:120]}"
                    ent = fail_reasons.setdefault(key, {"count": 0, "sample": rec.get("wafer_id")})
                    ent["count"] += 1
                    traceback.print_exc()
                done += 1
                _set_progress(True, done, total, "검사", str(folder))
            with STATE_LOCK:                              # 최근 N개만 메모리 유지(청크마다 정리)
                STATE["recs"].update(chunk_rec_map)
                _trim_recs()
            del recs, chunk_rec_map                       # 청크 원본 배열 즉시 해제 → 피크 메모리 제한
            _release_memory()                             # 해제분을 OS 에 실제 반환(malloc_trim)
    except BaseException:                                 # 어떤 예외든 진행바가 멈춘 채 남지 않게
        _set_progress(False, done, total, "중단", str(folder))
        _release_memory()
        raise

    _archive_flush()                                      # 비동기 아카이브 쓰기 완료 대기(유실 방지)
    if io_cols is None and _hits:                         # 전량 캐시 적중 → 표시용 스키마는 모델 기준
        io_cols = [str(c) for c in ENGINE.model.io_cols]
        sensor_cols = [str(c) for c in ENGINE.model.input_sensor_cols]
    results.sort(key=lambda r: _natkey(r["wafer_id"]))    # 파일명 순
    with STATE_LOCK:
        STATE["folder"] = str(folder)
        STATE["results"] = results if replace_results else \
            sorted({**{r["wafer_id"]: r for r in STATE["results"]},
                    **{r["wafer_id"]: r for r in results}}.values(),
                   key=lambda r: _natkey(r["wafer_id"]))
        STATE["io_cols"] = io_cols or []
        STATE["sensor_cols"] = sensor_cols or []
        STATE["coverage"] = cov
        STATE["inspected_at"] = _now()
        STATE["result_model"] = _model_id()
    _write_predictions_csv(STATE["results"])              # 전체 웨이퍼 predictions.csv 갱신
    # ★ .log 직접추론 전환으로 '변환본 csv' 는 더 이상 생기지 않는다.
    #   같은 스템 .log 가 있다고 csv 를 지우면 '사용자가 직접 넣은 csv' 를 오삭제할 수 있어 삭제 중단.
    if AUTOPILOT is not None:                             # 수동검사분도 자동감시 '처리됨'에 반영
        AUTOPILOT.mark_processed(r["wafer_id"] for r in results)
    # ── 처리 요약(투명성): '넣은 파일 수' vs '검사된 매수' 차이를 원인별로 정확히 구분 ──
    #   (a) 파일명(스템) 중복  (b) .log 기반 폴더의 .csv 부산물  (c) ★검사 필터로 제외
    #   (d) 손상 등으로 로드/판정 실패.  ※예전엔 (b)(c)까지 전부 '중복'으로 잘못 표시했음.
    try:
        _rawl = [p for p in _resolve_existing_dir_ci(Path(folder)).rglob("*")
                 if p.is_file() and p.suffix.lower() in (".csv", ".log", ".tsv")
                 and p.stem.lower() != "metadata"]
        _haslog = any(p.suffix.lower() in LOG_EXTS for p in _rawl)
        _csv_skip = sum(1 for p in _rawl if _haslog and p.suffix.lower() == ".csv")
        _raw = len(_rawl)
    except Exception:                                     # noqa: BLE001
        _raw, _csv_skip = _n_unique0, 0
    top_reasons = sorted(fail_reasons.items(), key=lambda kv: -kv[1]["count"])[:5]
    diag = {"files": int(_raw), "unique": int(_n_unique0), "inspected": len(results),
            "dup_excluded": max(0, int(_raw) - int(_csv_skip) - int(_n_unique0)) + int(_n_xdup),
            "csv_excluded": int(_csv_skip),
            "filtered_excluded": int(_n_filtered),
            "failed": max(0, (len(all_stems) + len(_hits)) - len(results)),
            "cache_hits": len(_hits),
            "fail_reasons": [{"reason": k, "count": v["count"], "sample": v["sample"]}
                             for k, v in top_reasons]}
    with STATE_LOCK:
        STATE["last_diag"] = diag
    if diag["dup_excluded"] or diag["failed"] or diag["filtered_excluded"] or diag["csv_excluded"]:
        print(f"[검사] 파일 {diag['files']}개 → 유니크 {diag['unique']} → 검사 {diag['inspected']}매 "
              f"(중복제외 {diag['dup_excluded']}, 필터제외 {diag['filtered_excluded']}, "
              f".csv부산물 {diag['csv_excluded']}, 로드/판정실패 {diag['failed']})", flush=True)
        for it in diag["fail_reasons"]:                   # ★ '왜' 실패했는지 사유별 개수·예시 로그
            print(f"  · 실패사유 [{it['count']}건] {it['reason']}  (예: {it['sample']})", flush=True)
    _set_progress(False, len(results), len(results), "완료", str(folder))
    out = {"folder": str(folder), "n": len(results),
           "results": STATE["results"], "summary": _summary(STATE["results"]),
           "coverage": cov, "inspected_at": STATE["inspected_at"], "diag": diag}
    # 검사 완료 → 남은 대용량 참조 회수 + OS 반환.
    del results
    _release_memory()
    return out


# ════════════════════════════════════════════════════════════════════
#  이력(일자별 이상점수) — append-only JSONL
# ════════════════════════════════════════════════════════════════════
_HIST_LOCK = threading.Lock()


_MODEL_ID_CACHE: Dict = {"key": None, "mid": None}


def _legacy_model_id() -> str:
    """현재 로드된 모델 식별자 — 이력을 모델별로 구분.

    ★ '모델 파일 내용' 기반(recon_meta.npz 해시) — 경로와 무관.
    예전 방식('폴더명@절대경로')은 exe/모델을 다른 PC·폴더로 옮기면 경로가 바뀌어
    같은 모델인데 이력이 전부 걸러지는 문제(이력검색 빈 화면)가 있었다.
    """
    if ENGINE.root.bundle_id:
        return ENGINE.root.bundle_id
    try:
        art = Path(ENGINE.art)
        meta = Path(getattr(ENGINE.model, "_meta_path", art / "recon_meta.npz"))
        norm = str(getattr(ENGINE.model, "normalization_mode", "standard"))
        if meta.exists():
            st = meta.stat()
            key = (str(meta), st.st_size, st.st_mtime_ns, norm)
            if _MODEL_ID_CACHE["key"] == key:
                return _MODEL_ID_CACHE["mid"]
            import hashlib
            payload = meta.read_bytes()
            if norm == "minmax":
                payload = b"minmax-v1:" + payload
            mid = "m@" + hashlib.md5(payload).hexdigest()[:12]
            _MODEL_ID_CACHE.update(key=key, mid=mid)
            return mid
        return f"{art.name}"                             # 메타 없음 → 폴더명(차선)
    except Exception:                                    # noqa: BLE001
        return str(getattr(ENGINE, "art", ""))


def _model_id() -> str:
    """Stable loaded-artifact identity; identical metadata alone is not a model."""
    root = ENGINE.root
    if root.bundle_id:
        return root.bundle_id
    cached = getattr(root, "_history_identity", None)
    if cached:
        return cached
    legacy = _legacy_model_id()
    art = Path(root.art)
    onnx = getattr(root.model, "_onnx_path", None)
    weights = Path(onnx) if onnx else _find_pt(art)
    if weights is None or not weights.is_file():
        return legacy
    digest = hashlib.sha256(b"single-model-history-v2:" + legacy.encode("utf-8"))
    override = override_identity(getattr(root, "wafer_score_override", None))
    if override:
        digest.update(json.dumps(override, sort_keys=True).encode("utf-8"))
    if auxiliary_sensor_columns(getattr(root, "cfg", None)):
        digest.update(b"cylinder-input-v1")
    config = art / "search_best.json"
    if not config.is_file():
        config = art / "best_search.json"
    for kind, path in ((b"weights", weights), (b"config", config)):
        digest.update(kind)
        if path.is_file():
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    mid = "m@v2-" + digest.hexdigest()[:24]
    root._history_identity = mid
    root._history_legacy_identity = legacy
    return mid


def _hist_entry(r: Dict, mid: str) -> Dict:
    """검사 결과 1건 → 이력 인덱스 행(예전 history.jsonl 한 줄과 동일 스키마)."""
    ts, day = _end_ts_and_date(r["wafer_id"])            # 파일명 끝시각 기준(없으면 now)
    return {
        "ts": ts, "date": day, "wafer_id": r["wafer_id"],
        "score": r.get("score"), "verdict": r.get("verdict"),
        "threshold": r.get("threshold"), "n_flag_sensor": r.get("n_flag_sensor"),
        "recipe": r.get("recipe"), "chamber": r.get("chamber"),
        "lot": (r.get("lot") or _lot_of(r["wafer_id"])),
        **{k: r.get(k) for k in AUDIT_FIELDS},
        "score_raw": r.get("score_raw"),
        "score_w_raw": r.get("score_w_raw"),
        "n_bad_lines": int(r.get("n_bad_lines") or 0),
        "n_tp": (int(r["n_tp"]) if _isnum(r.get("n_tp")) and int(r.get("n_tp") or 0) > 0
                 else None),
        "model": mid,
    }


def _append_history(results: List[Dict]):
    """검사 결과를 이력 '메모리 인덱스'에 반영한다.

    ★ 배포판 단일 소스 설계 — 이력 파일(history.jsonl)을 더 이상 만들지 않는다.
      영속 저장은 아카이브(웨이퍼 npz 의 meta_json)가 유일하고, 이 인덱스는 그
      아카이브의 메모리 사본일 뿐이다. 재시작하면 아카이브를 훑어 다시 만든다.
      → '이력은 지웠는데 어딘가 다른 파일이 남아 어긋나는' 사고가 원천 차단된다.
    """
    if not results:
        return
    mid = _model_id()
    _hist_index_ensure()
    with _HIST_LOCK:
        for r in results:
            e = _hist_entry(r, mid)
            _HIST_IDX["rows"][(e["date"], str(e["wafer_id"]), str(e.get("model") or ""))] = e
            _HIST_IDX["by_stem"].setdefault(str(e["wafer_id"]), set()).add(mid)
        _HIST_IDX["sorted"] = None                        # 다음 조회 때 재정렬
    _load_ntp()                                          # 사이드맵에도 반영(과거 이력 조회 시 재사용)
    _new = {str(r["wafer_id"]): int(r["n_tp"]) for r in results
            if _isnum(r.get("n_tp")) and int(r.get("n_tp") or 0) > 0}
    if _new:
        with _NTP_LOCK:
            _NTP.update(_new)
        _flush_ntp()


_HIST_IDX: Dict = {"built": False, "rows": {}, "by_stem": {}, "sorted": None}


def _hist_index_ensure():
    """아카이브를 1회 스캔해 이력 인덱스를 만든다(시작 후 첫 조회 시).

    각 npz 에서 meta_json 배열만 읽는다(압축 멤버 1개 — 파일당 수 ms).
    무제한 보관 설정에서는 보관된 전체 이력을 스캔한다. 실패한 파일은 건너뛴다.
    """
    if _HIST_IDX["built"]:
        return
    with _HIST_LOCK:
        if _HIST_IDX["built"]:
            return
        t0 = time.time()
        rows: Dict = {}
        by_stem: Dict = {}
        n_bad = 0
        try:
            days = sorted(d for d in ARCHIVE_DIR.iterdir() if d.is_dir()) \
                if ARCHIVE_DIR.exists() else []
        except Exception:                                 # noqa: BLE001
            days = []
        for d in days:
            for f in sorted(d.rglob("*.npz"), key=lambda p: (len(p.parts), str(p))):
                try:
                    with np.load(f, allow_pickle=False) as z:
                        meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
                    e = {
                        "ts": meta.get("ts"), "date": meta.get("date") or d.name,
                        "wafer_id": meta.get("wafer_id") or f.stem,
                        "score": meta.get("score"), "verdict": meta.get("verdict"),
                        "threshold": meta.get("threshold"),
                        "n_flag_sensor": meta.get("n_flag_sensor"),
                        "recipe": meta.get("recipe"), "chamber": meta.get("chamber"),
                        "lot": meta.get("lot") or _lot_of(meta.get("wafer_id") or f.stem),
                        **{k: meta.get(k) for k in AUDIT_FIELDS},
                        "score_raw": meta.get("score_raw"),
                        "score_w_raw": meta.get("score_w_raw"),
                        "n_bad_lines": int(meta.get("n_bad_lines") or 0),
                        "n_tp": meta.get("n_tp"),
                        "model": meta.get("model"),
                    }
                    rows[(e["date"], str(e["wafer_id"]), str(e.get("model") or ""))] = e
                    by_stem.setdefault(str(e["wafer_id"]), set()).add(str(meta.get("model") or ""))
                except Exception:                         # noqa: BLE001 — 손상 npz 는 건너뜀
                    n_bad += 1
        _HIST_IDX["rows"] = rows
        _HIST_IDX["by_stem"] = by_stem
        _HIST_IDX["sorted"] = None
        _HIST_IDX["built"] = True
        print(f"[hist] 아카이브 인덱스 구축 — {len(rows)}매"
              f"{f' · 손상 {n_bad}건 제외' if n_bad else ''}"
              f" ({time.time() - t0:.1f}초)", flush=True)


def _hist_index_reset():
    """전체 삭제 등 — 인덱스를 비우고 '구축됨' 상태로 둔다(재스캔 불필요)."""
    with _HIST_LOCK:
        _HIST_IDX["rows"] = {}
        _HIST_IDX["by_stem"] = {}
        _HIST_IDX["sorted"] = None
        _HIST_IDX["built"] = True


def _read_history() -> List[Dict]:
    """이력 조회 — 아카이브 메모리 인덱스(ts 오름차순). 스키마는 예전과 동일."""
    _hist_index_ensure()
    with _HIST_LOCK:
        if _HIST_IDX["sorted"] is None:
            _HIST_IDX["sorted"] = sorted(_HIST_IDX["rows"].values(),
                                         key=lambda e: str(e.get("ts") or ""))
        return list(_HIST_IDX["sorted"])


def _weighted_plot_threshold(rec, thr_w) -> Dict:
    """배점 그래프용 임계 dict: 곡선을 θ_w 로 정규화(선=1). 곡선=accum(z*W) 최종=S_w."""
    rkey = str((rec or {}).get("recipe") or "unknown") or "unknown"
    d = float(thr_w)
    return {"base_threshold_by_recipe": {rkey: d}, "threshold_by_recipe": {rkey: d},
            "threshold_raw": {rkey: d}, "default": d, "global_threshold_raw": d}


def _recipe_decision_by_name(recipe):
    """레시피명 → (base, decision=base×현재배수). _recipe_base_decision 의 태그 분기와 동일 규칙.
    이력 표시를 '현재 배율'로 재판정할 때 사용(레시피명이 이미 있으므로 파일명 재파싱 불필요)."""
    tds = ENGINE.thresholds or {}
    fallback = float(ENGINE.threshold or 0.0)
    tag = (_clean_recipe(recipe) or "")
    if not tag:
        return fallback, fallback
    m = _recipe_mult(tag)
    if tag in tds:
        b = float(tds[tag]); return b, b * _model_recipe_mult(tag)
    tl = tag.lower(); best = None
    for k, v in tds.items():
        kl = k.lower()
        if (kl == tl or tl.startswith(kl)) and (best is None or len(kl) > best[0]):
            best = (len(kl), k, v)
    if best is not None:
        b = float(best[2]); return b, b * _model_recipe_mult(best[1])
    return fallback, fallback * m


def _recipe_mult_sig():
    return "mult-off"                                     # 배포판: 배수 기능 제거 — 항상 중립

_RELIVE_CACHE: Dict = {"key": None, "rows": None}
# ── 집계 응답 캐시(개요·산포) ───────────────────────────────────────
#   체크박스를 바꿀 때마다 개요와 산포를 다시 계산하는데, 이력이 많으면 그대로 느려진다.
#   같은 조건이면 재사용한다. 키에 '판정에 영향을 주는 상태'를 전부 넣고,
#   상대 기간('1h' 등)은 시간이 흐르면 결과가 달라지므로 짧은 TTL 을 함께 둔다.
_AGG_CACHE: Dict = {}
_AGG_TTL = float(os.environ.get("IO_AGG_CACHE_SEC", "20"))


def _agg_sig() -> tuple:
    """집계 결과에 영향을 주는 '현재 상태' 서명."""
    return (_model_id(), _recipe_mult_sig(), _score_mode(),
            _disp_sig(_display_rule()),
            getattr(ENGINE, "ready", False),
            float(getattr(ENGINE, "threshold", 0.0) or 0.0),
            len(getattr(ENGINE, "thresholds", None) or {}))


def _agg_cached(name: str, params: tuple, n_rows: int):
    return None                                           # 배포판: 집계 캐시 제거(항상 재계산)


def _agg_store(name: str, params: tuple, n_rows: int, data):
    return data                                           # 배포판: 저장하지 않음
_SCATTER_CACHE: Dict = {"sig": None, "map": {}}          # 산점도 응답 캐시(이력 서명 동일 시 재사용)


# ════════════════════════════════════════════════════════════════════
#  표시 판정 규칙(display_rule) — 산점도·목록·챔버별 불량률 추이·1시간 현황 '단일 소스'
#  · 기록(history.jsonl)은 raw 그대로. 여기서 '표시용 판정'만 다시 계산한다.
#  · 이 규칙 하나만 바꾸면 모든 화면의 양/불 수가 동시에 바뀐다(요구 1·4).
#    mode      : raw    = 기존 이상점수(임계=1 정규화)
#                per_tp = 이상점수 / 타임포인트 수(점수 계산에 실제 쓰인 구간)
#    thr_scale : 레시피별 임계 배율(raw 모드) — 산점도에서 조절
#    thr_abs   : 레시피별 절대 임계(per_tp 모드) — 사용자 지정
#    bands     : 임계의 '배수'로 정의 → 임계가 바뀌면 밴드도 자동으로 따라간다(요구 2·4)
# ════════════════════════════════════════════════════════════════════
_DISP_ALL = "__all__"                                     # 레시피 미지정(전체 공통) 키
_DISPLAY_RULE_DEFAULT = {"mode": "raw", "thr_scale": {}, "thr_abs": {},
                         "bands": {"warn": 1.05, "crit": 1.2}, "ntp_estimate": True}


def _display_rule() -> Dict:
    d = dict(_DISPLAY_RULE_DEFAULT)
    try:
        cur = _load_app_config().get("display_rule") or {}
        if isinstance(cur, dict):
            d.update({k: v for k, v in cur.items() if k in _DISPLAY_RULE_DEFAULT})
    except Exception:                                     # noqa: BLE001
        pass
    # ★ 배포판: 표시 모드 = 항상 원점수(raw). '판정 표시·타임포인트' 기능 제거(사용자
    #   결정). 모델 자체의 타임포인트 정규화(score_norm)는 백엔드가 계속 처리한다 —
    #   그런 모델도 점수는 임계=1 정규화로 저장되므로 표시엔 영향이 없다.
    d["mode"] = "raw"
    for k in ("thr_scale", "thr_abs"):
        d[k] = {str(a): float(b) for a, b in (d.get(k) or {}).items()
                if _isnum(b) and float(b) > 0}
    b = d.get("bands") or {}
    warn = float(b.get("warn", 1.05)) if _isnum(b.get("warn")) else 1.05
    crit = float(b.get("crit", 1.2)) if _isnum(b.get("crit")) else 1.2
    warn = min(max(warn, 1.01), 1000.0)
    crit = min(max(crit, warn + 0.01), 1000.0)
    d["bands"] = {"warn": round(warn, 4), "crit": round(crit, 4)}
    d["ntp_estimate"] = bool(d.get("ntp_estimate", True))
    return d


def _isnum(v) -> bool:
    try:
        f = float(v)
        return bool(np.isfinite(f))
    except (TypeError, ValueError):
        return False


def _save_display_rule(patch: Dict) -> Dict:
    cur = _display_rule()
    for k in ("mode", "thr_scale", "thr_abs", "bands", "ntp_estimate"):
        if k in patch and patch[k] is not None:
            cur[k] = patch[k]
    cfg = _load_app_config()
    cfg["display_rule"] = cur
    _save_app_config(cfg)
    _RELIVE_CACHE["key"] = None                           # 즉시 재판정
    _SCATTER_CACHE.update(sig=None, map={})               # 산점도 응답 캐시도 무효화
    return _display_rule()


def _disp_lookup(table: Dict, recipe) -> Optional[float]:
    """레시피별 값 조회 — 정확일치 → 표시명 일치 → 전체(__all__) 순."""
    if not table:
        return None
    rc = str(recipe or "")
    if rc in table:
        return float(table[rc])
    disp = _recipe_display(rc) if rc else ""
    for k, v in table.items():
        if k != _DISP_ALL and _recipe_display(k) == disp and disp:
            return float(v)
    return float(table[_DISP_ALL]) if _DISP_ALL in table else None


def _disp_band(ratio: float, bands: Dict) -> str:
    """점수/임계 비율 → 등급. 1 미만=양품, 그 위로 주의→경고→심각."""
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return "ok"
    if not np.isfinite(r) or r < 1.0:
        return "ok"
    if r < float(bands["warn"]):
        return "caution"                                  # 주의(불량이나 경미)
    if r < float(bands["crit"]):
        return "warn"                                     # 경고
    return "crit"                                         # 심각


def _disp_sig(rule: Dict) -> tuple:
    return (rule["mode"],
            tuple(sorted(rule["thr_scale"].items())),
            tuple(sorted(rule["thr_abs"].items())),
            rule["bands"]["warn"], rule["bands"]["crit"])


# ── 타임포인트 수(n_tp): 점수 계산에 실제 쓰인 구간 길이(step != 0) ──
#   신규 검사는 결과에 바로 기록하고, 과거분은 아카이브(step 배열)에서 1회 백필한다.
NTP_FILE = DATA_DIR / "n_tp.json"
_NTP: Dict[str, int] = {}
_NTP_LOADED = {"v": False}
_NTP_LOCK = threading.Lock()


def _load_ntp():
    with _NTP_LOCK:
        if _NTP_LOADED["v"]:
            return
        try:
            if NTP_FILE.exists():
                _NTP.update({str(k): int(v) for k, v in
                             json.loads(NTP_FILE.read_text(encoding="utf-8")).items()})
        except Exception:                                 # noqa: BLE001
            traceback.print_exc()
        _NTP_LOADED["v"] = True


def _flush_ntp():
    try:
        NTP_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = NTP_FILE.with_name(NTP_FILE.name + ".tmp")
        with _NTP_LOCK:
            tmp.write_text(json.dumps(_NTP, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, NTP_FILE)
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()


def _ntp_of(wid: str, row: Dict = None) -> Optional[int]:
    """이 웨이퍼의 타임포인트 수 — 이력 행 → 사이드맵 순."""
    if row is not None:
        v = row.get("n_tp")
        if _isnum(v) and int(v) > 0:
            return int(v)
    _load_ntp()
    v = _NTP.get(str(wid))
    return int(v) if v else None


@_wafer_model
def _ntp_from_rec(rec) -> Optional[int]:
    """검사 중인 rec 에서 점수 구간 길이 계산(모델의 score keep mask 와 동일 정의)."""
    try:
        A = ENGINE.model
        keep = A._score_keep_mask(rec)
        n = int(np.asarray(keep, dtype=bool).sum())
        return n if n > 0 else None
    except Exception:                                     # noqa: BLE001
        return None


@_pin_model
def _relive_rows(rows: List[Dict]) -> List[Dict]:
    """표시용 재판정: 기록(raw)은 그대로 두고, score(임계=1 정규화)·threshold·verdict 를
    '현재 배율' 기준으로 다시 계산한 사본을 반환한다. → 배율을 바꾸면 재검사 없이도
    산점도 점·OK/NG 뱃지·목록·요약이 즉시 현재 배율로 재판정된다(기록 파일은 불변).
    (모델·이력길이·배율서명이 그대로면 캐시 재사용 → 폴링마다 재계산 안 함)"""
    mode = _score_mode()
    _thr_sig = tuple(sorted((str(k), float(v)) for k, v in
                            ((_load_app_config().get("score_thresholds") or {}).items()))) \
        if mode == "weighted" else None
    rule = _display_rule()                                # ★ 표시 판정 규칙(임계 배율·모드·밴드)
    # ★ 판정에는 모델 임계가 필요하다. 아직 안 올라왔으면 여기서 올린다.
    #   (개요·산점도가 모델 로드보다 먼저 호출되면 임계=0 → 전부 '판정 불가'로 원본 통과 →
    #    등급(band)이 비어 경고/위험 집계가 0이 되고, 그 결과가 캐시에 굳던 문제)
    if not getattr(ENGINE, "ready", False):
        try:
            ENGINE.load()
        except Exception:                                 # noqa: BLE001
            pass
    # ★ 모델 로드 상태·임계를 서명에 포함한다.
    #   (모델이 아직 안 올라온 시점에 이력을 읽으면 '판정 불가 → 원본 그대로' 결과가 캐시에 남아,
    #    이후 모델이 올라와도 재판정이 안 되고 등급(band)도 비어 경고/위험 집계가 0이 되던 버그)
    _eng_sig = (bool(getattr(ENGINE, "ready", False)),
                float(getattr(ENGINE, "threshold", 0.0) or 0.0),
                len(getattr(ENGINE, "thresholds", None) or {}))
    sig = (_model_id(), len(rows), _recipe_mult_sig(), mode, _thr_sig, _disp_sig(rule), _eng_sig)
    c = _RELIVE_CACHE
    if c["key"] == sig and c["rows"] is not None:
        return c["rows"]
    dec_cache: Dict = {}
    thrw_cache: Dict = {}
    bands = rule["bands"]
    per_tp = (rule["mode"] == "per_tp")
    # ★ n_tp 추정 폴백: 아카이브가 없어 백필이 안 되는 과거 웨이퍼(=판정 보류)를 살린다.
    #   공정 길이는 레시피별로 거의 일정하므로 '같은 레시피의 알려진 n_tp 중앙값'으로 추정하고
    #   추정치임을 표시(ntp_est)한다. 전역 중앙값은 레시피가 처음 보는 경우의 마지막 폴백.
    ntp_est_map: Dict[str, int] = {}
    ntp_est_all = 0
    if per_tp and rule.get("ntp_estimate", True):
        _byrc: Dict[str, List[int]] = {}
        _all: List[int] = []
        for r in rows:
            n = _ntp_of(r.get("wafer_id"), r)
            if n:
                _byrc.setdefault(_recipe_display(r.get("recipe") or ""), []).append(int(n))
                _all.append(int(n))
        for rc_, vs in _byrc.items():
            vs.sort(); ntp_est_map[rc_] = int(vs[len(vs) // 2])
        if _all:
            _all.sort(); ntp_est_all = int(_all[len(_all) // 2])
    out: List[Dict] = []
    for r in rows:
        if r.get("score_overridden") and _isnum(r.get("effective_score_raw")):
            ratio = float(r.get("override_ratio") or 1.01)
            nr = dict(r, score=ratio, ratio=ratio, ratio_thr=ratio, ratio_base=ratio,
                      threshold=1.0, verdict="NG" if ratio >= 1 else "OK",
                      score_raw=r["effective_score_raw"], band=_disp_band(ratio, bands))
            out.append(nr)
            continue
        rc = r.get("recipe") or ""
        cur_raw = None                                    # 현재 채점 모드의 '원점수'
        base_thr = None                                   # 그 모드의 기준 임계(배율 적용 前)
        tag = None
        # ── 배점 모드: S_w(score_w_raw) vs θ_w(레시피별). 미보정/구데이터면 raw 로 폴백 ──
        if mode == "weighted":
            sw = r.get("score_w_raw")
            try:
                sw = float(sw)
            except (TypeError, ValueError):
                sw = None
            thr_w = thrw_cache.get(rc)
            if thr_w is None:
                thr_w = _score_thr_for(rc); thrw_cache[rc] = thr_w
            if (sw is not None and np.isfinite(sw)
                    and thr_w is not None and thr_w > 1e-12 and np.isfinite(thr_w)):
                cur_raw, base_thr, tag = sw, float(thr_w), "weighted"
        if cur_raw is None:                               # ── raw 모드(또는 배점 폴백) ──
            eng = _engine_for_wafer(r.get("wafer_id"), r)
            decision_key = (r.get("chamber"), rc)
            hit = dec_cache.get(decision_key)
            if hit is None:
                with ENGINE.using(eng):
                    hit = _recipe_decision_by_name(rc)
                dec_cache[decision_key] = hit
            _base, decision = hit
            raw = r.get("score_raw")
            try:
                raw = float(raw)
            except (TypeError, ValueError):
                raw = None
            if raw is None:                               # 구(舊) 이력 폴백: score×threshold_raw = raw
                try:
                    raw = float(r.get("score")) * float(r.get("threshold_raw"))
                except (TypeError, ValueError):
                    raw = None
            if decision is None or not (decision > 1e-12) or not np.isfinite(decision):
                # ★ 현재 모델에 임계가 없을 때: 검사 당시 이력에 남은 임계로 등급을 계산한다.
                #   (예전 데이터를 불러왔는데 지금 모델에 임계가 없으면, 예전에는 재계산을
                #    포기해 band 가 비었고 → 개요가 불량을 전부 '주의'로 세어 경고·심각이 0 이 됐다)
                try:
                    _hist_thr = float(r.get("threshold_raw"))
                except (TypeError, ValueError):
                    _hist_thr = None
                if _hist_thr is not None and np.isfinite(_hist_thr) and _hist_thr > 1e-12:
                    decision = _hist_thr
                    _thr_from_history = True
                else:
                    out.append(r); continue               # 정말 근거가 없을 때만 원본 유지
            else:
                _thr_from_history = False
            if raw is None or not np.isfinite(raw):
                out.append(r); continue                   # 원점수 없음 → 재계산 불가
            cur_raw, base_thr = raw, float(decision)

        # ── ★ 표시 판정 규칙 적용 ──
        if per_tp:
            # 타임포인트 정규화: 점수 계산에 실제 쓰인 구간(step≠0) 길이로 나눈다.
            ntp = _ntp_of(r.get("wafer_id"), r)
            est = False
            if not ntp:                                   # 실측 없음 → 레시피 중앙값으로 추정
                ntp = ntp_est_map.get(_recipe_display(rc)) or ntp_est_all
                est = bool(ntp)
            thr = _disp_lookup(rule["thr_abs"], rc)
            if not ntp or thr is None or not (thr > 0):
                nr = dict(r)
                nr["disp_na"] = ("타임포인트 미상(백필 필요)" if not ntp else "임계 미설정")
                nr["band"] = "na"
                out.append(nr); continue                  # 값 없으면 판정 보류
            sc = cur_raw / float(ntp)
            _sc = math.floor(sc * 1e6) / 1e6              # 작은 값이라 6자리 유지
            _thr = float(thr)
            _vd = "NG" if sc >= _thr else "OK"
            _ratio = sc / _thr if _thr > 1e-12 else 0.0
        else:
            scale = _disp_lookup(rule["thr_scale"], rc) or 1.0
            _ratio = cur_raw / base_thr if base_thr > 1e-12 else 0.0
            _sc = _floor2(_ratio)                         # 모델 임계=1 기준 점수(표시용)
            _thr = round(float(scale), 4)                 # 조정된 판정 기준선(=임계 배율)
            _vd = "NG" if _ratio >= scale else "OK"
            # ★ 등급은 '조정된 임계' 대비 비율로 매긴다.
            #   (배율을 올려 판정이 완화됐는데 등급만 옛 기준으로 남아, 판정 OK 인데 '주의'로
            #    표시되던 불일치를 막는다. 밴드는 임계의 배수이므로 임계가 움직이면 함께 이동.)
            _ratio_base = _ratio                          # 모델 임계 대비 원비율(배율 적용 전)
            _ratio = (_ratio / scale) if scale > 1e-12 else _ratio
        _tr = math.floor(float(base_thr) * 1000) / 1000.0
        _bd = _disp_band(_ratio, bands)
        nr = dict(r)
        nr["score"] = _sc
        nr["threshold"] = _thr
        nr["threshold_raw"] = _tr
        nr["verdict"] = _vd
        nr["band"] = _bd
        if _thr_from_history:
            nr["thr_from_history"] = True                  # 현재 모델 임계가 없어 이력 임계로 판정
        # 임계 대비 비율(밴드 판정·산포 계산 근거). 6자리 — 4자리로 자르면 내보낸 HTML 의
        # 재계산값과 소수점이 어긋나 같은 화면인데 값이 미세하게 달라 보인다.
        nr["ratio_thr"] = round(float(_ratio), 6)
        if tag == "weighted":
            nr["score_raw"] = math.floor(float(cur_raw) * 1000) / 1000.0
            nr["score_mode"] = "weighted"
        nr["ratio_base"] = round(float(locals().get("_ratio_base", _ratio)), 6)
        if per_tp:
            nr["score_tp"] = _sc
            nr["n_tp"] = int(ntp)
            if est:
                nr["ntp_est"] = True                      # 추정치(실측 아님) — UI 에 표시
        out.append(nr)
    c["key"] = sig; c["rows"] = out
    return out


def _history_for_current_model() -> List[Dict]:
    """현재 모델의 이력만 반환(그래프 오염 방지).

    모델을 바꾸면 옛 모델의 큰 raw 점수(예: 19673)가 추세/산점도에 찍혀 스케일을
    망가뜨린다. 이력에 모델 태그가 하나라도 있으면 '현재 모델' 것만 남긴다.
    (전부 태그 없는 레거시면 그대로 — 기존 사용자 이력 보존.)
    """
    rows = _read_history()
    if not any(r.get("model") for r in rows):
        return _relive_rows(rows)                          # 전부 레거시 → 보존 + 현재 배율로 표시 재판정
    mid = _model_id()
    # Prefer new model-scoped rows over compatible pre-upgrade rows for the same wafer.
    matched = [r for r in rows if _archive_model_matches(r.get("model"), mid)]
    selected = {}
    for row in sorted(matched, key=lambda r: r.get("model") == mid):
        selected[(row.get("date"), str(row.get("wafer_id")))] = row
    return _relive_rows(list(selected.values())) if selected else []



_HIST_FALLBACK_WARNED = False


def _hist_ratio(entry) -> float:
    """이력 1건을 임계=1 정규화 값으로 변환.

    ★ 정밀값(ratio_thr, 6자리)이 있으면 그것을 쓴다.
      score 는 이미 '표시용 2자리 버림'을 거친 값이라, 그걸로 다시 나눈 뒤 화면에서
      또 버리면 한 칸 더 깎인다(4.1025 → 4.10 → 4.09). 산점도에서 고친 것과 같은
      함정이 이력 검색·달력·챔버 램프에도 있었다 — 근원인 여기서 한 번에 막는다.

    ratio_thr 이 없는 옛 이력은 예전처럼 score/threshold 로 계산한다(raw 점수 13297 ·
    raw 임계 1538 이든 정규화 8.6 · 임계 1.0 이든 같은 1 기준 값이 된다).
    """
    v = entry.get("ratio_thr")
    if _isnum(v):
        return float(v)
    try:
        sc = float(entry.get("score"))
        thr = float(entry.get("threshold"))
    except (TypeError, ValueError):
        return float("nan")
    return (sc / thr) if (np.isfinite(thr) and thr > 1e-12) else sc


def _within_hours(entry, hours: float, now_dt) -> bool:
    """이력 1건이 최근 N시간 이내인가(hours<=0 이면 전체)."""
    if not hours or hours <= 0:
        return True
    ts = entry.get("ts")
    if not ts:
        return True
    try:
        t = dt.datetime.fromisoformat(str(ts))
    except Exception:                                    # noqa: BLE001
        return True
    return (now_dt - t).total_seconds() <= hours * 3600.0


# 챔버 상태 램프 기준 — 환경변수로 조정 가능(현장에서 빡빡/느슨 조절)
_CH_RATE_DEFECT = float(os.environ.get("IO_CH_RATE_DEFECT", "5"))    # % — 이상(빨강) 선
_CH_RATE_WARN   = float(os.environ.get("IO_CH_RATE_WARN", "2"))      # % — 경고 선
_CH_MIN_N       = int(os.environ.get("IO_CH_MIN_N", "20"))           # 비율 판단 최소 표본


def _chamber_level(total: int, ng: int, max_ratio: float) -> str:
    """이상점수 추이 → 5단계: normal/warn1/warn2/warn3/defect (+none 데이터없음).

    ★ 예전에는 ng>0 이면 무조건 defect(빨강)이었다. 한 장만 걸려도 챔버 전체가
      빨개져 '항상 빨강'이 되고, 그러면 램프가 정보를 주지 못한다(사용자 지적).

    바꾼 기준 — '한 장'이 아니라 '비율'로 본다.
      · 불량률이 기준선을 넘으면 이상/경고. 기본 5% / 2%.
        근거: 이 시스템의 임계는 정상 분포의 μ+3σ 로 잡히므로 정상 공정에서도
        0.3% 안팎은 임계를 넘을 수 있다(오탐). 거기에 표본 변동을 감안하면
        수 % 대가 '평소와 다르다'고 말할 수 있는 선이다.
      · 다만 표본이 적으면 비율이 불안정하다(5장 중 1장=20%). 그래서
        _CH_MIN_N(기본 20장) 미만이면 비율로 단정하지 않고, 대신 '연속 신호'를
        요구한다 — 2장 이상 심각이어야 이상으로 본다.
      · 한 장뿐이면 경고까지만 올린다. 놓치지는 않되 빨강으로 단정하지 않는다.
    """
    if total <= 0:
        return "none"
    rate = 100.0 * ng / total if total else 0.0
    if total >= _CH_MIN_N:                                # 표본 충분 → 비율로 판단
        if rate >= _CH_RATE_DEFECT:
            return "defect"
        if rate >= _CH_RATE_WARN:
            return "warn3"
    else:                                                 # 표본 부족 → 건수로 보수적 판단
        if ng >= 2:
            return "defect"
    if ng >= 1:
        return "warn3"                                    # 1장 발생: 경고까지만(빨강 아님)
    r = float(max_ratio)                                  # 불량 0 → 임계 근접도로 예열 표시
    if r >= 0.90:
        return "warn2"
    if r >= 0.80:
        return "warn1"
    return "normal"


def chamber_overview(hours: float = 24, t0=None, t1=None) -> Dict:
    """챔버(dsv1~6)별 간략 현황: 대상 구간 내 불량/전체 + 5단계 위험도.

    ★ t0/t1 을 주면 '그 구간'을 본다. 안 주면 예전처럼 최근 N시간(now 기준).
      개요 화면은 반드시 t0/t1 로 부른다 — 예전에는 조회 구간의 '길이'만 넘기고
      기준점은 now 였다. 그래서 과거 구간을 조회하면(사용자 지정 기간, 또는 데이터가
      과거에서 끝나는 전체 기간) 화면의 나머지는 그 구간인데 챔버 타일만 최근
      데이터를 세는 불일치가 있었다. 같은 화면에서 대상이 다르면 안 된다.
    """
    rows = _history_for_current_model()
    now = dt.datetime.now()
    latest: Dict = {}                                     # (chamber, wafer) 최신 1건
    for r in rows:
        if t0 is not None and t1 is not None:
            _t = _parse_ts(r.get("ts"))
            if _t is None or not (t0 <= _t <= t1):
                continue
        elif not _within_hours(r, hours, now):
            continue
        ch = _chamber_key(r)                                   # 폴더귀속 우선(소문자 통일)
        key = (ch, r.get("wafer_id"))
        prev = latest.get(key)
        if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
            latest[key] = r
    agg = {c: {"total": 0, "ng": 0, "max_ratio": 0.0,
               "caution": 0, "warn": 0, "crit": 0} for c in CHAMBERS}
    for (ch, _wid), r in latest.items():
        if ch not in agg:
            continue                                      # dsv1~6 외(미인식)는 제외
        b = agg[ch]
        b["total"] += 1
        if r.get("verdict") == "NG":
            b["ng"] += 1
        _bd = str(r.get("band") or "")
        if _bd in ("caution", "warn", "crit"):
            b[_bd] += 1
        rr = _hist_ratio(r)
        if np.isfinite(rr):
            b["max_ratio"] = max(b["max_ratio"], float(rr))
    out = []
    for c in CHAMBERS:
        b = agg[c]
        out.append({
            "chamber": c, "total": b["total"], "ng": b["ng"],
            "caution": b["caution"], "warn": b["warn"], "crit": b["crit"],
            "max_ratio": round(b["max_ratio"], 3),
            # ★ 램프/상태는 '위험' 등급 기준(사용자 요청) — 경미한 초과로 전체가 빨개지지 않게.
            "level": _chamber_level(b["total"], b["crit"], b["max_ratio"]),
            "level_ng": _chamber_level(b["total"], b["ng"], b["max_ratio"]),   # 참고(전체 불량 기준)
        })
    return {"chambers": out, "hours": hours}


# ── 웨이퍼 아카이브: 이미지 대신 최소 데이터(actual/expected/z/step) 압축 저장 ──
def _archive_model_matches(stored, current):
    """Recognize only this model's tag, including its old directory-based tag."""
    if str(stored or "") == str(current or ""):
        return True
    if str(current).startswith("m@v2-") and stored == (getattr(ENGINE.root, "_history_legacy_identity", None) or _legacy_model_id()):
        return True
    if not stored or str(stored).startswith("m@"):
        return False
    art = Path(ENGINE.root.art)
    return str(stored) in (art.name, str(art), str(art.resolve()),
                           f"{art.name}@{art}", f"{art.name}@{art.resolve()}")


def _model_archive_path(date: str, wid: str) -> Path:
    # Same wafer inspected by two models must never share the output file.
    scope = hashlib.sha256(_model_id().encode("utf-8")).hexdigest()[:24]
    return ARCHIVE_DIR / date / ("model_" + scope) / (_safe(wid) + ".npz")


def _archive_write_path(date: str, wid: str) -> Path:
    path = _model_archive_path(date, wid)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _archive_path(date: str, wid: str) -> Path:
    """Read current model's archive; old flat files remain untouched."""
    path = _model_archive_path(date, wid)
    if path.is_file():
        return path
    legacy = ARCHIVE_DIR / date / (_safe(wid) + ".npz")
    if legacy.is_file():
        try:
            with np.load(legacy, allow_pickle=False) as data:
                meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
            if _archive_model_matches(meta.get("model"), _model_id()):
                return legacy
            # Untagged pre-model archives are readable only in wholly untagged history.
            if not meta.get("model") and not any(row.get("model") for row in _read_history()):
                return legacy
        except (OSError, ValueError, KeyError):
            pass
    return path


# ── 아카이브 비동기 기록: zlib 압축+디스크 쓰기를 백그라운드 워커로 → 추론 루프 지연 제거 ──
def _atomic_savez(p, arrays: Dict):
    """npz 를 임시파일에 쓴 뒤 원자적 교체(os.replace).

    쓰는 도중엔 최종 경로에 파일이 '없어서' 읽기 쪽이 불완전 파일을 np.load 하는 일이
    원천적으로 불가능(특히 Windows: 디스크 느림/백신으로 쓰기 창이 길어질 때 중요)."""
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:                            # 파일객체로 넘겨 .npz 자동확장 방지
        np.savez_compressed(f, **arrays)
    os.replace(tmp, p)                                    # 원자적 교체(리눅스/윈도우 공통)


_ARCHIVE_Q: Optional[queue.Queue] = None
_ARCHIVE_LOCK = threading.Lock()


def _archive_queue() -> queue.Queue:
    global _ARCHIVE_Q
    with _ARCHIVE_LOCK:
        if _ARCHIVE_Q is None:
            _ARCHIVE_Q = queue.Queue(maxsize=64)          # 백프레셔(디스크 느릴 때 무한 적체 방지)

            def _writer():
                while True:
                    p, arrays = _ARCHIVE_Q.get()
                    try:
                        _atomic_savez(p, arrays)
                    except Exception:                     # noqa: BLE001
                        traceback.print_exc()
                    finally:
                        _ARCHIVE_Q.task_done()

            threading.Thread(target=_writer, name="archive-writer", daemon=True).start()
        return _ARCHIVE_Q


def _archive_flush(timeout: float = 60.0):
    """대기 중인 아카이브 쓰기를 모두 마칠 때까지 대기(검사 완료 시 호출 — 파일 유실 방지)."""
    q = _ARCHIVE_Q
    if q is None:
        return
    end = time.time() + timeout
    while not q.empty() and time.time() < end:
        time.sleep(0.05)
    try:
        q.join()
    except Exception:                                    # noqa: BLE001
        pass


@_wafer_model
def _archive_wafer(rec, result: Dict):
    """검사한 웨이퍼의 렌더용 최소 데이터 + 메타를 압축 저장(장당 ~150KB).
    ★ 압축·디스크 쓰기는 백그라운드 워커에 넘겨 추론 루프를 막지 않는다(배열 준비만 여기서)."""
    try:
        A = ENGINE.model
        actual, expected, z = RX._compute(A, rec)
        step = rec.get("step")
        step = (np.asarray(step, dtype=np.int32) if step is not None
                else np.zeros(actual.shape[0], dtype=np.int32))
        meta = {k: result.get(k) for k in
                ("wafer_id", "recipe", "chamber", "score", "threshold",
                 "score_raw", "threshold_raw", "verdict", "n_flag_sensor",
                 "n_bad_lines", "score_w_raw", "n_tp", "model_chamber") + AUDIT_FIELDS}
        # ★ 단일 소스 설계: 이력 인덱스가 아카이브에서 재구축되므로, 이력 행에
        #   필요한 필드(LOT 포함)를 여기 meta 에 전부 싣는다.
        meta["lot"] = result.get("lot") or _lot_of(result.get("wafer_id"))
        _ets, _eday = _end_ts_and_date(result.get("wafer_id"))   # 파일명 끝시각 기준
        # 배점 그래프 재그리기용: 필요한 밸브들의 스텝별 열림상태(작음). z 와 같은 T 축.
        om, _T0 = _valve_open_map(rec)
        vcols = sorted(om.keys())
        if vcols and _T0 == z.shape[0]:
            vopen = np.stack([om[v] for v in vcols], axis=1).astype(np.uint8)
        else:
            vopen = np.zeros((z.shape[0], 0), dtype=np.uint8); vcols = []
        # ★ RCA용 원본 보존(기본 on): IO 전체(81ch)+입력센서(48) 를 함께 저장 →
        #   원본 .log 가 삭제/이동돼도 npz 만으로 RCA 가능(임시 CSV 복원). 장당 +수백 KB.
        #   설정 archive_rca_source=false 로 끌 수 있음.
        rca_io = None; rca_insen = None; rca_io_cols = []; rca_in_cols = []
        try:
            if bool(_load_app_config().get("archive_rca_source", True)):
                _io = np.asarray(rec.get("io"), dtype=np.float64)
                if _io.ndim == 2 and _io.shape[0] == z.shape[0]:
                    rca_io = _io.astype(np.float16)
                    rca_io_cols = [str(c) for c in (rec.get("io_cols") or A.io_cols)]
                _ins = np.asarray(A._input_sensors(rec), dtype=np.float64)
                if _ins.ndim == 2 and _ins.shape[0] == z.shape[0]:
                    rca_insen = _ins.astype(np.float16)
                    rca_in_cols = [str(c) for c in A.input_sensor_cols]
        except Exception:                                # noqa: BLE001 — RCA소스 실패해도 아카이브는 저장
            rca_io = rca_insen = None; rca_io_cols = []; rca_in_cols = []
        trend = rec.get("trend_measurements")
        if trend is not None:
            meta["trend_sensor_cols"] = trend["sensor_cols"]
            meta["trend_zero_version"] = 1
        meta["step_timing"] = rec.get("step_timing")
        meta["source_path"] = rec.get("source_path")
        meta.update(ts=_ets, date=_eday, model=_model_id(),
                    sensor_cols=[str(c) for c in A.sensor_cols],
                    valve_cols=vcols,
                    rca_io_cols=rca_io_cols, rca_in_cols=rca_in_cols,
                    interp_factor=int(rec.get("_interp_factor", 1) or 1))
        p = _archive_write_path(_eday, result["wafer_id"])      # 끝시각 날짜 폴더에 보관
        mj = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                           dtype=np.uint8)
        arrays = dict(actual=actual.astype(np.float16),
                      expected=expected.astype(np.float16),
                      z=z.astype(np.float16), app_score_level=z.astype(np.float32), step=step, meta_json=mj,
                      valve_open=vopen)
        if trend is not None:
            arrays["trend_actual"] = trend["actual"]
            arrays["trend_step"] = trend["step"]
        if rca_io is not None and rca_insen is not None:
            arrays["rca_io"] = rca_io                     # RCA 복원용 IO 전체
            arrays["rca_insen"] = rca_insen               # RCA 복원용 입력센서 원본
        try:
            _archive_queue().put((p, arrays), timeout=5.0)   # 비동기(백프레셔: 가득 차면 대기)
        except Exception:                                    # noqa: BLE001 — 큐 실패 시 동기 폴백
            _atomic_savez(p, arrays)
        rec.pop("trend_measurements", None)
    except Exception:                                    # noqa: BLE001
        traceback.print_exc()


def _load_archive(date: str, wid: str):
    p = _archive_path(date, wid)
    if not p.exists():
        # ★ 비동기 아카이브 도입 대응: 방금 검사된 웨이퍼는 npz 가 아직 쓰기 큐에 있을 수 있다.
        #   큐에 잔량이 있으면 잠깐 비우고 1회 재시도(검사 중 상세 클릭 시 일시 404 방지).
        q = _ARCHIVE_Q
        # empty() 는 '꺼내서 쓰는 중(in-flight)'을 못 본다 → unfinished_tasks(진행 중 포함)로 확인
        if q is not None and getattr(q, "unfinished_tasks", 0):
            _archive_flush(timeout=3.0)
        if not p.exists():
            return None
    d = np.load(p, allow_pickle=False)
    meta = json.loads(bytes(d["meta_json"].tobytes()).decode("utf-8"))
    try:
        vopen = d["valve_open"].astype(np.uint8)           # 구 아카이브엔 없음 → None
    except (KeyError, Exception):                          # noqa: BLE001
        vopen = None
    out = {"actual": d["actual"].astype(np.float64),
           "expected": d["expected"].astype(np.float64),
           "z": d["z"].astype(np.float64), "step": d["step"].astype(int),
           "meta": meta, "valve_open": vopen,
           "valve_cols": meta.get("valve_cols") or []}
    for k in ("rca_io", "rca_insen", "app_score_level", "trend_actual", "trend_step"):                      # RCA 복원용(신규 아카이브만 존재)
        try:
            out[k] = d[k].astype(np.float64)
        except (KeyError, Exception):                      # noqa: BLE001
            out[k] = None
    return out


def _rca_source_from_archive(date: str, wid: str):
    """원본 .log 가 없을 때, 아카이브 npz 의 RCA용 원본(IO+입력센서)으로 임시 CSV 를 복원.

    반환: 임시 CSV 경로(호출측이 부모 임시폴더 정리) 또는 None(구 아카이브 — RCA소스 미저장).
    """
    arch = _load_archive(date, wid)
    if arch is None:                                       # 날짜 폴더가 다르면 전체 탐색 폴백
        d2 = _archive_date_of(wid)
        arch = _load_archive(d2, wid) if d2 else None
    if not arch or arch.get("rca_io") is None or arch.get("rca_insen") is None:
        return None
    meta = arch["meta"]
    io_cols = [str(c) for c in (meta.get("rca_io_cols") or [])]
    in_cols = [str(c) for c in (meta.get("rca_in_cols") or [])]
    io = arch["rca_io"]; ins = arch["rca_insen"]; step = arch["step"]
    if (not io_cols or not in_cols or io.shape[1] != len(io_cols)
            or ins.shape[1] != len(in_cols) or io.shape[0] != ins.shape[0]):
        return None
    tmp = Path(tempfile.mkdtemp(prefix="rcasrc_"))
    csvf = tmp / f"{_safe(wid)}.csv"
    T = io.shape[0]
    with open(csvf, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(["Date", "Time", "Process Step"] + io_cols + in_cols) + "\n")
        for t in range(T):
            row = ["", "", f"{float(step[t] if t < len(step) else 0):.1f}"]
            row += [f"{v:.6g}" for v in io[t]]
            row += [f"{v:.6g}" for v in ins[t]]
            f.write(",".join(row) + "\n")
    return csvf


def _archive_weight_matrix(arch, sensor_cols):
    """아카이브의 밸브 열림상태 → (T, n_sensors) 가중치. 없으면 None(배점 그래프 불가)."""
    vo = arch.get("valve_open"); vc = arch.get("valve_cols") or []
    if vo is None or not len(vc) or getattr(vo, "ndim", 0) != 2 or vo.shape[0] == 0:
        return None
    T = vo.shape[0]
    om = {str(vc[k]): (vo[:, k] > 0) for k in range(min(len(vc), vo.shape[1]))}
    return _weight_matrix_from_open(om, T, sensor_cols)


def _archive_rec(arch) -> Dict:
    m = arch["meta"]
    return {**{k: m.get(k) for k in AUDIT_FIELDS},
            "wafer_id": m.get("wafer_id"), "step": arch["step"],
            "sensors": arch["actual"],                   # keep-mask 등 T·shape 용
            "recipe": m.get("recipe") or "unknown",
            "_interp_factor": int(m.get("interp_factor", 1) or 1)}


@_wafer_model
def render_archive_plot(date: str, wid: str, kind: str) -> Path:
    """아카이브 데이터로 상세 그림을 그때그때 렌더(디스크 영구저장 안 함, 임시파일 스트리밍)."""
    arch = _load_archive(date, wid)
    if arch is None:
        raise KeyError((date, wid))
    A = ENGINE.model
    if arch["z"].shape[1] != len(A.sensor_cols):
        raise RuntimeError("현재 로드된 모델과 검사 당시 센서 수가 달라 렌더 불가(그 모델을 로드하세요).")
    rec = _archive_rec(arch)
    actual, expected, z = arch["actual"], arch["expected"], arch["z"]
    out = Path(tempfile.mkdtemp(prefix="arch_")) / f"{_safe(kind)}.png"
    thr = _plot_threshold_dict()
    with _PLOT_LOCK:                                      # pyplot 직렬화
        try:
            RX._setup_font()
        except Exception:                                # noqa: BLE001
            pass
        if kind == "overlay":
            RX.plot_recon_overlay(A, rec, out, top_k=None, pre=(actual, expected, z), threshold=thr)
        elif kind == "heatmap":
            RX.plot_recon_heatmap(A, rec, out, z_pre=z)
        elif kind == "waferaccum":
            # ★ 판정선을 실제 판정임계(base×앱배수)와 동기화 → 그래프 1선과 판정 불일치 해소
            # ★ 배점 모드: 인터랙티브(wafer-series)와 동일하게 z×가중·θ_w (PNG 폴백 일관성)
            zg, thr_d, okb = z, _waferaccum_threshold(wid, rec), _ok_bands()
            if _score_mode() == "weighted" and not rec.get("score_overridden"):
                W = _archive_weight_matrix(arch, [str(c) for c in A.sensor_cols])
                thr_w = _score_thr_for(rec.get("recipe"))
                if W is not None and W.shape == z.shape and thr_w and thr_w > 1e-12:
                    zg = z * W; thr_d = _weighted_plot_threshold(rec, thr_w); okb = None
            RX.plot_wafer_accum_score(A, rec, zg, out, ok_bands=okb,
                                      threshold=thr_d,
                                      final_score=rec.get("effective_score_raw"),
                                      apply_final_score=bool(rec.get("score_overridden")), raw_rec=rec, raw_z=zg)
        else:
            raise ValueError(kind)
    return out


@_wafer_model
def render_archive_sensor(date: str, wid: str, rank: int) -> Path:
    arch = _load_archive(date, wid)
    if arch is None:
        raise KeyError((date, wid))
    A = ENGINE.model
    if arch["z"].shape[1] != len(A.sensor_cols):
        raise RuntimeError("모델 센서 수 불일치 — 검사 당시 모델을 로드하세요.")
    rec = _archive_rec(arch)
    actual, expected, z = arch["actual"], arch["expected"], arch["z"]
    order = _top_sensor_indices(z, max(1, int(rank) + 1))
    if int(rank) < 0 or int(rank) >= len(order):
        raise IndexError(rank)
    j = int(order[int(rank)])
    out = Path(tempfile.mkdtemp(prefix="archs_")) / f"{int(rank):02d}.png"
    with _PLOT_LOCK:                                      # pyplot 직렬화
        try:
            RX._setup_font()
        except Exception:                                # noqa: BLE001
            pass
        RX.plot_recon_sensor(A, rec, j, actual, expected, z, out,
                             ok_bands=_ok_bands(), threshold=_plot_threshold_dict(),
                             raw_rec=rec, raw_z=z)
    return out


RETENTION_MAX_DAYS = 60                                   # 일수 지정 상한. 0은 무제한 보관.


def _archive_retention_days() -> int:
    """보관 일수: 0(화면 입력 0000)은 무제한, 1~60은 기간 보관.
    미설정·음수·60 초과·손상값은 기본 60일로 처리한다."""
    try:
        d = int(_load_app_config().get("archive_retention_days"))
    except (TypeError, ValueError):
        d = RETENTION_MAX_DAYS
    if d < 0 or d > RETENTION_MAX_DAYS:
        return RETENTION_MAX_DAYS
    return d


def _prune_archive() -> int:
    """보관기간 지난 아카이브 날짜폴더 삭제. 반환=삭제한 날짜 수."""
    days = _archive_retention_days()
    if days <= 0 or not ARCHIVE_DIR.exists():
        return 0
    cutoff = dt.date.today() - dt.timedelta(days=days)
    removed = 0
    for d in list(ARCHIVE_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            dd = dt.date.fromisoformat(d.name)
        except ValueError:
            continue
        if dd < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def _prune_history() -> int:
    """보관기간 지난 이력 '인덱스 행' 정리 — 아카이브 폴더 정리(_prune_archive)와 짝."""
    days = _archive_retention_days()
    if days <= 0:
        return 0
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    _hist_index_ensure()
    with _HIST_LOCK:
        drop = [k for k in _HIST_IDX["rows"] if str(k[0]) < cutoff]
        for k in drop:
            _HIST_IDX["rows"].pop(k, None)
        if drop:
            _HIST_IDX["sorted"] = None
            _HIST_IDX["by_stem"] = {}
            for (d_, wid, model_), e in _HIST_IDX["rows"].items():
                _HIST_IDX["by_stem"].setdefault(str(wid), set()).add(str(e.get("model") or ""))
    return len(drop)

def _prune_file_meta() -> int:
    """보관기간 지난 file_meta(종료시각·챔버 태그) 항목 정리(무한 증가 방지). 반환=삭제 수.
    같은 이유로 recipe_tags 도 file_meta 에 없는(=만료된) 스템을 함께 정리한다."""
    days = _archive_retention_days()
    if days <= 0:
        return 0
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    expired = []
    with _FILE_META_LOCK:
        if not _FILE_META:
            _load_file_meta()
        for stem in list(_FILE_META.keys()):
            ets = str(_FILE_META[stem].get("end_ts") or "")
            if ets and ets[:10] < cutoff:
                del _FILE_META[stem]; expired.append(stem)
        if expired:
            try:
                _file_meta_path().write_text(
                    json.dumps(_FILE_META, ensure_ascii=False, indent=0), encoding="utf-8")
            except Exception:                             # noqa: BLE001
                pass
    if expired and _RECIPE_TAGS:                          # '만료로 삭제된 스템'의 태그만 정리(레거시 보존)
        with _RECIPE_TAGS_LOCK:
            for stem in expired:
                _RECIPE_TAGS.pop(stem, None)
            try:
                _recipe_tags_file().write_text(
                    json.dumps(_RECIPE_TAGS, ensure_ascii=False, indent=0), encoding="utf-8")
            except Exception:                             # noqa: BLE001
                pass
    return len(expired)


def history_daily(days: int = 30) -> Dict:
    """일자별 집계 — (date, wafer)당 최신 1건으로 중복 제거 후 날짜별 평균/최대/NG수.
    점수는 임계=1 정규화(score/threshold)로 집계. 현재 모델 이력만 사용(모델 교체 시 오염 방지)."""
    rows = _history_for_current_model()
    # (date, wafer) 최신만 유지
    latest: Dict = {}
    for r in rows:
        key = (r.get("date"), r.get("wafer_id"))
        prev = latest.get(key)
        if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
            latest[key] = r
    byday: Dict[str, list] = {}
    for r in latest.values():
        byday.setdefault(r["date"], []).append(r)

    out = []
    for day in sorted(byday):
        sc = np.array([_hist_ratio(x) for x in byday[day]], dtype=float)
        sc = sc[np.isfinite(sc)]
        if not sc.size:
            continue
        ng = sum(1 for x in byday[day] if x["verdict"] == "NG")
        out.append({
            "date": day, "n": int(sc.size), "ng": int(ng), "ok": int(sc.size - ng),
            "mean": round(float(sc.mean()), 3),
            "max": round(float(sc.max()), 3),
            "p95": round(float(np.percentile(sc, 95)), 3),
        })
    if days and len(out) > days:
        out = out[-days:]
    # 간단 추세: 최근 mean 이 앞구간 평균보다 높으면 상승
    trend = "flat"
    if len(out) >= 4:
        half = len(out) // 2
        early = np.mean([d["mean"] for d in out[:half]])
        late = np.mean([d["mean"] for d in out[half:]])
        if late > early * 1.05:
            trend = "up"
        elif late < early * 0.95:
            trend = "down"
    # 점수는 임계=1 정규화 값이므로 추세선 임계도 정규화 기준 1.0.
    return {"days": out, "threshold": 1.0, "trend": trend}


def _summary(results: List[Dict]) -> Dict:
    ng = sum(1 for r in results if r["verdict"] == "NG")
    # 요약 임계도 정규화(1 기준): base=1.0, 레시피별은 배수(29a=1.1).
    norm_thr = {str(r): round(_recipe_mult(r), 3) for r in (ENGINE.thresholds or {})}
    return {"total": len(results), "ng": ng, "ok": len(results) - ng,
            "threshold": 1.0,
            "thresholds": norm_thr}


def wafer_detail(wid: str) -> Dict:
    """상세 분석: 점수·판정·그림. RCA(자연어 원인/센서별 원인분석)는 제외."""
    rec = STATE["recs"].get(wid)
    if rec is None:
        # 대량검사 후 메모리에서 밀려난 웨이퍼 → 아카이브 메타로 대체(상세 조회 유지).
        _ts, _day = _end_ts_and_date(wid)
        arch = _load_archive(_day, wid) if _day else None
        if arch is not None:
            m = arch["meta"]
            return {**{k: m.get(k) for k in AUDIT_FIELDS}, "wafer_id": wid, "score": m.get("score"),
                    "threshold": m.get("threshold", 1.0), "recipe": m.get("recipe"),
                    "verdict": m.get("verdict"), "score_raw": m.get("score_raw"),
                    "threshold_raw": m.get("threshold_raw"),
                    "n_flag_sensor": m.get("n_flag_sensor"), "chamber": m.get("chamber"),
                    "n_bad_lines": int(m.get("n_bad_lines") or 0)}
        raise KeyError(wid)
    s = _assess(rec)
    sv = _score_view(wid, float(s["score"]), rec)         # 임계=1 정규화
    return {
        "wafer_id": wid,
        "score": sv["score"],
        "threshold": sv["threshold"],
        "recipe": sv["recipe"],
        "verdict": sv["verdict"],
        **{k: sv.get(k) for k in AUDIT_FIELDS},
        "score_raw": sv["score_raw"],
        "threshold_raw": sv["threshold_raw"],
        "n_flag_sensor": s["n_flag_sensor"],
        "n_bad_lines": int(rec.get("n_bad_lines") or 0),
        "chamber": _chamber(wid),
    }


def _round(v, n=2):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def _safe(name: str) -> str:
    """파일명 안전화(경로/특수문자 → _)."""
    return re.sub(r'[^\w.\-]', "_", str(name))


def _short_name(wid) -> str:
    """웨이퍼 '폴더명' 단축 — Windows 260자 경로 한계 대응.

    RCA/리포트 경로는 '…\\rca\\<웨이퍼폴더>\\slope\\slope_valve_reports\\<웨이퍼명>_....md' 처럼
    긴 웨이퍼명이 두 번 들어가, 앱이 깊은 폴더에 있으면 260자를 넘겨 '폴더는 생기는데
    파일 쓰기만 [Errno 2]로 실패'하는 문제가 났다. 폴더 세그먼트는 앞 22자+해시로 줄인다
    (파일명 쪽은 RCA 모듈이 정하므로 유지 — 폴더 단축만으로 ~35자 이상 절약)."""
    s = _safe(str(wid))
    if len(s) <= 28:
        return s
    return s[:22] + "~" + hashlib.md5(s.encode()).hexdigest()[:5]


def _model_slug() -> str:
    """현재 모델을 폴더명으로 안전화한 식별자. 리포트를 모델별로 분리 보관.
    _model_id()('__bundled__' 또는 '이름@전체경로')를 파일시스템 안전 슬러그로 변환."""
    mid = _model_id()
    if mid == "__bundled__":
        return "__bundled__"
    h = hashlib.md5(mid.encode("utf-8")).hexdigest()[:8]   # 경로 충돌 방지 해시
    name = _safe(mid.split("@", 1)[0]) or "model"
    return f"{name}_{h}"


def _report_root() -> Path:
    """현재 모델의 리포트 루트: reports/<model_slug>/  (모델별 분리)."""
    d = REPORT_DIR / _model_slug()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pred_csv() -> Path:
    """현재 모델의 predictions.csv 경로(모델별 분리)."""
    return _report_root() / "predictions.csv"


def _plot_path(wid: str, kind: str) -> Path:
    # 모델별·웨이퍼별 폴더: reports/<model>/<wafer>/waferaccum.png, overlay.png, heatmap.png
    d = _report_root() / _short_name(wid)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{kind}.png"


def _clear_report_images():
    """'현재 모델' 리포트 폴더만 삭제(다른 모델 리포트는 보존). 웨이트 재로딩 시 사용."""
    try:
        root = _report_root()
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    except Exception:                                    # noqa: BLE001
        pass


# ── OK 누적 참조 밴드 캐시 ──
#   배포 현장엔 train-OK 가 없으므로 모델 폴더의 prebuilt npz 를 로드해 재사용.
#   모델 교체(reload/select) 시 sentinel 로 무효화 → 다음 호출 때 재로딩.
_OKBANDS_SENTINEL = object()
_OKBANDS_CACHE = {"bands": _OKBANDS_SENTINEL}


def _ok_bands():
    """모델 폴더의 ok_accum_bands.npz 를 1회 로드(없거나 실패하면 None).

    load_or_build_ok_accum_bands(A, [], npz) 에 빈 ok_records 를 넘기면
    '호환 캐시 있으면 재사용' 경로를 타서 prebuilt band 를 그대로 돌려준다.
    그래도 못 얻으면 npz 를 직접 열어 _cache_to_ok_bands 로 변환한다.
    """
    cache = (ENGINE._scope.get() or ENGINE.root).__dict__.setdefault("_ok_bands_cache", {})
    if "bands" in cache:
        return cache["bands"]
    bands = None
    try:
        A = ENGINE.model
        npz = Path(ENGINE.art) / "ok_accum_bands.npz"
        if A is not None and npz.exists():
            try:
                bands = RX.load_or_build_ok_accum_bands(A, [], npz, threshold=None)
            except Exception:                             # noqa: BLE001
                traceback.print_exc()
                bands = None
            if bands is None:                             # 직접 로드 폴백
                try:
                    data = dict(np.load(npz, allow_pickle=True))
                    bands = RX._cache_to_ok_bands(data)
                except Exception:                         # noqa: BLE001
                    traceback.print_exc()
                    bands = None
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()
        bands = None
    cache["bands"] = bands
    return bands


def _invalidate_ok_bands():
    """모델 교체 시 OK 밴드 캐시 무효화(다음 호출에 재로딩)."""
    _OKBANDS_CACHE["bands"] = _OKBANDS_SENTINEL
    for eng in [ENGINE.root] + list(ENGINE.root.chamber_engines.values()):
        eng.__dict__.pop("_ok_bands_cache", None)


@_wafer_model
def ensure_plot(wid: str, kind: str) -> Path:
    """waferaccum/overlay/heatmap PNG 생성(없으면). 모델 재사용, 캐시."""
    rec = STATE["recs"].get(wid)
    if rec is None:
        raise KeyError(wid)
    path = _plot_path(wid, kind)
    if path.exists():
        return path
    A = ENGINE.model
    with _PLOT_LOCK:                                       # pyplot 직렬화(리포트/탭 동시렌더 충돌 방지)
        if path.exists():                                  # 락 대기 중 다른 요청이 이미 생성했을 수도
            return path
        try:
            RX._setup_font()
        except Exception:                                 # noqa: BLE001
            pass
        actual, expected, z = RX._compute(A, rec)         # wafer 1회 계산(전 kind 재사용)
        thr = _plot_threshold_dict()                      # base=1 정규화, 29a→1.1
        if kind == "overlay":
            RX.plot_recon_overlay(A, rec, path, top_k=None,
                                  pre=(actual, expected, z), threshold=thr)
        elif kind == "heatmap":
            RX.plot_recon_heatmap(A, rec, path, z_pre=z)
        elif kind == "waferaccum":
            # 웨이퍼 레벨 누적 복원오차 — 검정=현재, 파랑=OK평균, 하늘=OK 최소~최대.
            # ★ 판정선을 실제 판정임계(base×앱배수)와 동기화 → 그래프 1선과 판정 불일치 해소.
            # ★ 배점 모드: 인터랙티브(wafer-series)와 동일하게 z×가중·θ_w 로 그린다(PNG 폴백 일관성).
            zg, thr_d, okb = z, _waferaccum_threshold(wid, rec), _ok_bands()
            effective = _score_view(wid, A._score_record(rec), rec)
            if _score_mode() == "weighted" and not effective["score_overridden"]:
                W = _score_weight_matrix(rec, [str(c) for c in A.sensor_cols])
                thr_w = _score_thr_for(rec.get("recipe"))
                if W is not None and W.shape == z.shape and thr_w and thr_w > 1e-12:
                    zg = z * W; thr_d = _weighted_plot_threshold(rec, thr_w)
                    okb = None
            if effective["score_overridden"]:
                fsc = effective["effective_score_raw"]
            else:
                fsc = None
            RX.plot_wafer_accum_score(
                A, rec, zg, path, ok_bands=okb, threshold=thr_d,
                final_score=fsc, apply_final_score=effective["score_overridden"],
                raw_rec=rec, raw_z=zg)
        else:
            raise ValueError(kind)
    return path


def _sensor_dir(wid: str) -> Path:
    d = _report_root() / _short_name(wid) / "sensors"        # reports/<model>/<wafer>/sensors/
    d.mkdir(parents=True, exist_ok=True)
    return d


def _top_sensor_indices(z, n: int = 6) -> List[int]:
    """이탈 정도 순 상위 센서 인덱스(recon_explain._sensor_order)."""
    sig = float(getattr(ENGINE.model.cfg, "raw_error_level", 1.0))
    return RX._sensor_order(z, sig)[:int(n)]


def _sensor_all_scores(z, keep=None) -> List[float]:
    """모든 센서의 이상점수를 원래 순서대로 0~2 로 정규화(웨이퍼 내부, 최대 센서=2).

    이상점수 = '누적 초과오차 합' — 정상범위(sig) 를 넘은 만큼을 시간 전체로 합산.
    (웨이퍼 '복원오차 누적' 과 같은 관점: 오래·자주 벗어날수록 높음).
    keep: ProcessStep 0 등 비평가 구간을 제외하는 시간축 마스크(다른 모든 곡선과 동일 기준)."""
    za = np.asarray(z, dtype=float)
    if keep is not None:                                 # ProcessStep 0 제외(다른 곡선과 일관)
        keep = np.asarray(keep, dtype=bool)
        if keep.shape[0] == za.shape[0] and keep.any():
            za = za[keep]
    sig = float(getattr(ENGINE.model.cfg, "raw_error_level", 1.0))
    accum = np.clip(za - sig, 0, None).sum(axis=0)       # 센서별 초과분 누적합
    mx = float(accum.max()) if accum.size else 0.0
    if mx <= 1e-9:                                        # 초과 전혀 없음(정상) → 전부 0
        return [0.0] * int(accum.size)
    return [round(float(2.0 * a / mx), 3) for a in accum]


@_wafer_model
def wafer_sensors(wid: str, n: int = 6) -> Dict:
    """상위 N개 센서(이름+rank) — UI 갤러리 구성용."""
    rec = STATE["recs"].get(wid)
    if rec is None:
        raise KeyError(wid)
    A = ENGINE.model
    _, _, z = RX._compute(A, rec)
    idx = _top_sensor_indices(z, n)
    sensors = [{"rank": r, "sensor_index": int(j), "sensor": str(A.sensor_cols[int(j)])}
               for r, j in enumerate(idx)]
    return {"wafer_id": wid, "sensors": sensors}


@_wafer_model
def ensure_sensor_plot(wid: str, rank: int) -> Path:
    """rank-번째 상위 센서의 2단(실제vs기대 + 누적) PNG 생성(없으면). 캐시.
    파일명은 실제 센서명: reports/<wafer>/sensors/<랭크>_<센서명>.png (예: 00_p_005_PV.png)."""
    rec = STATE["recs"].get(wid)
    if rec is None:
        raise KeyError(wid)
    d = _sensor_dir(wid)
    hits = sorted(d.glob(f"{int(rank):02d}_*.png"))       # 랭크 접두로 캐시 조회
    if hits:
        return hits[0]
    A = ENGINE.model
    with _PLOT_LOCK:                                       # pyplot 직렬화
        hits = sorted(d.glob(f"{int(rank):02d}_*.png"))    # 락 대기 중 생성됐을 수도
        if hits:
            return hits[0]
        actual, expected, z = RX._compute(A, rec)
        order = _top_sensor_indices(z, max(1, int(rank) + 1))
        if int(rank) < 0 or int(rank) >= len(order):
            raise IndexError(rank)
        j = int(order[int(rank)])
        name = str(A.sensor_cols[j])
        path = d / f"{int(rank):02d}_{_safe(name)}.png"    # 실제 센서명 포함
        try:
            RX._setup_font()
        except Exception:                                 # noqa: BLE001
            pass
        RX.plot_recon_sensor(A, rec, j, actual, expected, z, path,
                             ok_bands=_ok_bands(), threshold=_plot_threshold_dict(),
                             raw_rec=rec, raw_z=z)
    return path


# ════════════════════════════════════════════════════════════════════
#  자동 조종(AutoPilot) — 파일 자동감시 + 일 마감 백업 + 30일 보관
# ════════════════════════════════════════════════════════════════════
class AutoPilot:
    """백그라운드 스레드 1개로 세 가지를 담당:
      1) 감시: inbox 에 '완성된' 신규 CSV 가 생기면 즉시 추론 → 결과/이력 반영
      2) 마감: 날짜가 바뀌면 그날의 inbox 파일을 backup/<날짜>/ 로 이동
      3) 보관: RETENTION_DAYS(30) 지난 backup/<날짜> 폴더 자동 삭제
    """

    def __init__(self):
        self.enabled = AUTO_START               # 감시 on/off (마감·보관은 항상 동작)
        self._stop = threading.Event()
        self._control_lock = threading.Lock()
        self._generation = 0
        self._thread: Optional[threading.Thread] = None
        self._plock = threading.Lock()
        # ★ 배포판: '처리됨' 판정은 아카이브 인덱스가 유일한 근거다.
        #   예전의 processed 영속 목록은 이력과 따로 놀아 '이력을 지워도 재검사가
        #   안 되는' 사고의 근본 원인이었다. processed 는 이제 '이번 실행 중 스킵'
        #   (스키마 실패 소진분 등) 만 담는 비영속 세트다.
        self.processed: set = set()             # 세션 한정 스킵(재시작하면 빈다)
        self._load_fail: dict = {}              # wafer_id → 정렬실패 누적횟수(임시, 영속화 안 함)
        self.model_id: Optional[str] = None     # 현재 처리 기준 모델 — 바뀌면 processed 리셋
        self.day = _today()
        self.last_event: Optional[str] = None
        self.last_scan: Optional[str] = None
        self._load_state()

    # ── 상태 영속화(재시작 시 날짜 이어받기) ──
    def _load_state(self):
        try:
            d = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
            self.day = d.get("day") or self.day
            if "enabled" in d:
                self.enabled = bool(d["enabled"])
        except Exception:                       # noqa: BLE001
            pass
        self._save_state()

    def _save_state(self):
        try:
            AUTOPILOT_STATE.parent.mkdir(parents=True, exist_ok=True)
            AUTOPILOT_STATE.write_text(json.dumps(
                {"day": self.day,
                 "enabled": self.enabled}, ensure_ascii=False), encoding="utf-8")
        except Exception:                       # noqa: BLE001
            pass

    def _is_processed(self, stem: str) -> bool:
        """이 파일을 현재 모델로 이미 검사했는가 — 아카이브 인덱스가 근거.

        모델이 바뀌면 같은 파일도 '미처리'가 되어 자동 재검사된다(예전
        _check_model_change 의 processed 리셋과 같은 효과, 상태 파일 없이).
        이력(아카이브)을 지우면 인덱스가 비므로 전체 재검사가 자동 성립한다.
        """
        if stem in self.processed:                        # 세션 스킵(스키마 실패 소진 등)
            return True
        _hist_index_ensure()
        try:
            mid = _model_id()
        except Exception:                                 # noqa: BLE001
            mid = ""
        with _HIST_LOCK:
            models = _HIST_IDX["by_stem"].get(str(stem))
        return bool(models) and (mid in models)

    def mark_processed(self, wids):
        with self._plock:
            for w in wids:
                self.processed.add(w)
        self._save_state()

    # ── 스레드 제어 ──
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="autopilot", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def cancel_scan(self):
        with self._control_lock:
            self._generation += 1

    def set_enabled(self, on: bool):
        with self._control_lock:
            self.enabled = bool(on)
            if not on:
                self._generation += 1
        self._save_state()
        self.last_event = f"{_now()} 자동감시 {'ON' if on else 'OFF'}"

    # ── 메인 루프 ──
    def _loop(self):
        try:
            self._retention()                   # 시작 시 1회 보관정리
            self._check_rollover()              # 재시작 catch-up(지난 날짜 마감)
        except Exception:                       # noqa: BLE001
            traceback.print_exc()
        while not self._stop.wait(WATCH_INTERVAL):
            try:
                self._ensure_model()            # 모델이 새로 들어오면 새로고침 없이 자동 로드
                self._check_rollover()
                if self.enabled:
                    self._scan_once()
            except InspectAborted:
                # A background worker must never clear the foreground stop signal.
                _set_progress(False, 0, 0, "중단", "", bg=True)
            except Exception:                   # noqa: BLE001
                traceback.print_exc()

    def _ensure_model(self):
        """모델 미로드 상태에서 모델 폴더에 모델이 생기면 자동으로 로드(폴링)."""
        if ENGINE.ready:
            return
        try:
            art = (Path(ENGINE.selected) if (ENGINE.selected and _has_model(Path(ENGINE.selected)))
                   else _resolve_art())
            if _has_model(art):                 # 실제로 모델이 있을 때만 재로드(로그 스팸 방지)
                info = ENGINE.reload()
                if info.get("ready"):
                    print(f"[engine] 모델 자동 로드: {ENGINE.art}")
        except Exception:                       # noqa: BLE001
            traceback.print_exc()

    # ── 1) 신규 파일 감시·추론 ──
    def _check_model_change(self):
        """모델 교체 감지 — 세션 스킵만 비운다(처리 판정은 인덱스가 모델별로 함)."""
        try:
            mid = _model_id()
        except Exception:                       # noqa: BLE001
            return
        if self.model_id is None:
            self.model_id = mid
            return
        if mid != self.model_id:
            print("[autopilot] 모델 교체 감지 → 새 모델 기준으로 자동 재검사", flush=True)
            with self._plock:
                self.processed = set()          # 세션 스킵 해제(새 모델로 다시 시도)
            self._load_fail = {}
            self.model_id = mid
            self.last_event = f"{_now()} 모델 교체 → 전체 재검사"

    def _pending_count(self, folder: Path) -> int:
        """이번 주기에 이 폴더에서 검사될 후보 수(가벼운 사전 계수 — 진행바 총계용)."""
        try:
            now = time.time()
            seen = set()
            n = 0
            for f in folder.iterdir():
                _pause_gate()
                if not f.is_file() or f.suffix.lower() not in (".csv", ".log", ".tsv"):
                    continue
                if f.stem.lower() == "metadata" or f.stem in seen:
                    continue
                if f.stem in self.processed or self._is_processed(f.stem):
                    continue
                if (now - f.stat().st_mtime) < STABLE_SECS:
                    continue
                seen.add(f.stem)
                n += 1
            return n
        except Exception:                       # noqa: BLE001
            return 0

    def _scan_once(self):
        with self._control_lock:
            generation = self._generation
        previous = getattr(_INSPECT_CONTEXT, "auto", None)
        _INSPECT_CONTEXT.auto = (self, generation)
        try:
            _pause_gate()
            return self._scan_once_current()
        finally:
            _INSPECT_CONTEXT.auto = previous

    def _scan_once_current(self):
        self._check_model_change()              # 모델 바뀌면 processed 리셋(새 모델로 재검사)
        # 챔버 폴더(DSV1~6, 존재분) + 기본 입력 폴더(데이터/입력) 를 모두 감시.
        folders = _scan_targets()
        self.last_scan = _now()
        # ★ 자동검사 1회 = dsv1~6 전부. 진행바도 '주기 전체'를 하나로 묶는다
        #   (사용자 요구) — 폴더마다 바가 따로 떴다 사라지지 않고, 전 폴더 합산
        #   총계 대비 진행이 한 바에 이어진다.
        plan = [(ch, fd, self._pending_count(fd)) for ch, fd in folders]
        total_all = sum(c for _, _, c in plan)
        done_off = 0
        total_new = 0
        touched = []
        for chamber, folder, cnt in plan:
            _pause_gate()
            try:
                n = self._scan_folder(folder, chamber,
                                      prog_off=done_off, prog_total=total_all)
                total_new += n
                if n:
                    touched.append(str(chamber or folder.name))
            except Exception:                   # noqa: BLE001
                traceback.print_exc()
            done_off += cnt
        if total_new == 0:
            _set_progress(False, 0, 0, "대기", bg=True)
        else:
            _set_progress(False, total_all, total_all, "완료", "", bg=True)
            with STATE_LOCK:                    # 주기 전체 완료 흔적(폴더 나열)
                STATE["bg_done"] = {"ts": time.time(), "n": total_new,
                                    "folder": ", ".join(touched)}

    @_pin_model
    def _scan_folder(self, folder: Path, chamber=None,
                     prog_off: int = 0, prog_total=None) -> int:
        """폴더 하나를 검사(신규·정지 파일만). chamber 지정시 결과를 그 챔버로 귀속."""
        now = time.time()
        # ★ '파일 준비' 단계부터 진행바에 표시(사용자 요구) — 대량 유입 시 태깅이
        #   길어져 자동검사가 멈춘 것처럼 보이던 문제. 준비 중에도 바가 움직인다.
        _lbl = str(chamber or folder.name)
        _tt0 = (prog_total if prog_total else 0)
        _convert_logs_in(folder, min_age=STABLE_SECS,
                         skip_stems={st for st in self.processed},
                         progress_cb=lambda i, k: _set_progress(
                             True, prog_off, _tt0 or k,
                             f"파일 준비 {_lbl} ({i}/{k})", str(folder), bg=True))
        try:
            files = sorted([p for p in folder.iterdir() if p.is_file()
                            and p.suffix.lower() in (".csv", ".log", ".tsv")
                            and p.stem.lower() != "metadata"])   # 폴더 메타파일은 웨이퍼 아님
            if any(p.suffix.lower() in LOG_EXTS for p in files):
                files = [p for p in files if p.suffix.lower() != ".csv"]   # .log 기반 → .csv 무시
        except Exception:                       # noqa: BLE001
            return 0
        try:                                    # ★ 자동검사(추론) 경로에도 폴더 metadata 생성/무결성
            _folder_metadata_update(folder, files)
        except Exception:                       # noqa: BLE001
            traceback.print_exc()
        seen_stems = set()
        cand = []
        for f in files:                          # .log 직접 검사(변환본 없음) · 스템당 1개
            _pause_gate()
            if f.stem in seen_stems or self._is_processed(f.stem):
                continue
            if (now - f.stat().st_mtime) < STABLE_SECS:   # 복사 완료(정지)된 것만
                continue
            seen_stems.add(f.stem)
            cand.append(f)
        if not cand:
            return 0
        ENGINE.load()
        if not ENGINE.ready:
            return 0
        groups = {}
        for f in cand:
            key = _file_meta(f.stem).get("chamber") if ENGINE.root.chamber_engines else "single"
            groups.setdefault(key or "unknown", []).append(f)
        count = offset = 0
        chunk_size = max(1, int(os.environ.get("IO_INSPECT_CHUNK", "200")))
        for key, group in sorted(groups.items()):
            for i in range(0, len(group), chunk_size):
                _pause_gate()
                batch = group[i:i + chunk_size]
                count += self._inspect_candidates(folder, batch, chamber, prog_off + offset,
                                                   prog_total or len(cand))
                offset += len(batch)
        return count

    def _inspect_candidates(self, folder, cand, chamber, prog_off, prog_total):
        """Bounded same-chamber batch; failures retain the existing per-file retry accounting."""
        _lbl = str(chamber or folder.name)
        stems = {f.stem for f in cand}
        _set_progress(True, prog_off, (prog_total if prog_total else len(cand)),
                      f"파일 읽는 중 {_lbl} ({len(cand)}건)", str(folder), bg=True)
        try:
            recs, io_cols, sensor_cols, cov = _load_aligned(folder, stems=stems)   # ★ 신규 파일만 로드
        except Exception as e:                  # noqa: BLE001 — 스키마 불일치 등
            # 즉시 processed 로 묻으면 '시작 직후 잘못된/미로드 모델' 같은 일시적 실패에도
            # 그 배치가 영영 재검사되지 않는다. 대신 파일별 실패횟수를 세어 LOAD_MAX_RETRIES
            # 초과분만 스킵 → 일시 오류는 회복되고, 진짜 불량 스키마는 무한 재시도되지 않는다.
            exhausted = []
            for s in stems:
                c = self._load_fail.get(s, 0) + 1
                self._load_fail[s] = c
                if c >= LOAD_MAX_RETRIES:
                    exhausted.append(s)
            self.last_event = (f"{_now()} 자동검사 보류({folder.name}): {e} "
                               f"[재시도 {LOAD_MAX_RETRIES}회 초과 {len(exhausted)}건 스킵]")
            print(f"[autopilot] {self.last_event}")
            if exhausted:
                self.mark_processed(exhausted)   # 재시도 소진분만 영구 스킵
                for s in exhausted:
                    self._load_fail.pop(s, None)
            return 0
        with STATE_LOCK:
            STATE["coverage"] = cov
        todo = [rec for rec in recs if rec["wafer_id"] in stems]
        total = len(todo)
        new_results, rec_map = [], {}
        _pause_gate()
        _prefill_batch(todo, pause_check=_pause_gate)       # ★ 배치 순전파 1회(GPU/CPU 가속)
        print(f"[autopilot] 자동검사 시작 — {chamber or folder.name} {total}건 "
              f"(감시 토글={'ON' if self.enabled else 'OFF'})", flush=True)
        _tt = (prog_total if prog_total else total)       # 주기 전체 총계(한 바로 이어짐)
        _set_progress(True, prog_off, _tt, f"자동검사 {chamber or folder.name}", str(folder), bg=True)
        for i, rec in enumerate(todo, 1):
            _wait_if_paused(i - 1, total, str(folder), bg=True)   # 일시중단 시 대기(자동검사)
            try:
                r = _result_for(rec)
                _check_inspect_cancelled()
                if chamber and not r.get("chamber"):   # .log C2 챔버 우선, 폴더 귀속은 폴백
                    r["chamber"] = chamber
                new_results.append(r)
                rec_map[rec["wafer_id"]] = rec
                # ★ 스트리밍: 한 장 끝나는 즉시 이력 append + 아카이브 → 대시보드 실시간 반영
                _append_history([r])
                _archive_wafer(rec, r)
                rec.pop("_level_cache", None)   # STATE 보관 rec 에 level 캐시 잔류 방지(RAM)
            except Exception:                   # noqa: BLE001
                traceback.print_exc()
            _set_progress(True, prog_off + i, _tt, f"자동검사 {chamber or folder.name}", str(folder), bg=True)
        _archive_flush()                                   # 비동기 아카이브 쓰기 완료 대기
        # ★ 성공=실패카운트 리셋+처리표시 / 실패(판정 예외)=재시도 카운트→소진분만 스킵.
        #   (예전엔 성공분만 processed 로 표시 → 판정 실패 웨이퍼가 매 스캔 무한 재시도되며 CPU 점유)
        done_wids = {r["wafer_id"] for r in new_results}
        failed_wids = [rec["wafer_id"] for rec in todo if rec["wafer_id"] not in done_wids]
        for s in done_wids:
            self._load_fail.pop(s, None)                   # 성공 → 실패카운트 초기화
        exhausted = []
        for s in failed_wids:
            c = self._load_fail.get(s, 0) + 1
            self._load_fail[s] = c
            if c >= LOAD_MAX_RETRIES:
                exhausted.append(s)
        if exhausted:                                      # 재시도 소진 → 영구 스킵(무한루프 차단)
            self.mark_processed(exhausted)
            for s in exhausted:
                self._load_fail.pop(s, None)
            print(f"[autopilot] 판정 실패 {len(exhausted)}건 재시도({LOAD_MAX_RETRIES}회) 소진 → 스킵", flush=True)
        if not new_results:
            return 0
        _merge_results(new_results, rec_map, folder)
        _write_predictions_csv(STATE["results"])         # predictions.csv 갱신
        self.mark_processed(done_wids)
        # (변환본 csv 없음 — 삭제 안 함. 사용자 csv 보호)
        # (진행바 종료·완료 흔적은 주기(_scan_once) 레벨에서 한 번만)
        ng = sum(1 for r in new_results if r["verdict"] == "NG")
        self.last_event = f"{_now()} 자동검사 {len(new_results)}건 (NG {ng}){f' · {chamber}' if chamber else ''}"
        print(f"[autopilot] {self.last_event}")
        del recs, todo, rec_map                           # 배치 원본 배열 회수 → 메모리 반환
        _release_memory()
        return len(new_results)

    # ── 2) 날짜 롤오버 → 마감 백업 ──
    def _check_rollover(self):
        today = _today()
        if today == self.day:
            return
        self._end_of_day(self.day)              # 바뀌기 전 날짜로 백업
        try:
            n = _prune_archive()                # 보관기간 지난 이력 아카이브 자동 정리
            if n:
                print(f"[archive] 보관기간 초과 {n}일치 정리")
            hn = _prune_history()               # history.jsonl 도 같은 보관기간으로 정리
            if hn:
                print(f"[history] 보관기간 초과 {hn}행 정리")
            fm = _prune_file_meta()             # 파일 메타(끝시각·챔버·레시피 태그)도 정리
            if fm:
                print(f"[file-meta] 보관기간 초과 {fm}건 정리")
        except Exception:                       # noqa: BLE001
            traceback.print_exc()
        self.day = today
        self._save_state()

    def _end_of_day(self, day: str):
        moved = 0
        dest = BACKUP_DIR / day                            # sources 비어도 참조 가능하게 선초기화
        # ★ 챔버 폴더(D:\smarteq_logs\DSV1~6)는 설비 원본 저장소 — 절대 이동·삭제하지 않는다.
        #   마감 이동은 '기본 입력 폴더(데이터/입력)'만. (요구: 기간 초과돼도 smarteq_logs 는 그대로)
        sources = [(ch, p) for ch, p in _scan_targets() if not ch]
        for chamber, src in sources:
            dest = BACKUP_DIR / day / chamber if chamber else BACKUP_DIR / day
            try:
                dest.mkdir(parents=True, exist_ok=True)
                src_files = (sorted(src.glob("*.csv"))    # 변환본 + 원본(.log/.tsv) 모두 마감 이동
                             + sorted(src.glob("*.log")) + sorted(src.glob("*.tsv")))
                src_files = [f for f in src_files
                             if f.stem.lower() != "metadata"]   # 폴더 메타파일은 이동 금지
                for f in src_files:
                    try:
                        target = dest / f.name
                        if target.exists():     # 동명 파일 있으면 시간 접미사
                            target = dest / f"{f.stem}__{int(time.time())}{f.suffix}"
                        shutil.move(str(f), str(target))
                        moved += 1
                    except Exception:           # noqa: BLE001
                        traceback.print_exc()
            except Exception:                   # noqa: BLE001
                traceback.print_exc()
        self._retention()
        self.last_event = f"{_now()} '{day}' 마감 백업 {moved}건 → {dest}"
        print(f"[autopilot] {self.last_event}")
        return moved

    # ── 3) 30일 보관 정리 ──
    def _retention(self):
        # ★ 백업(원본 .log) 삭제도 UI '데이터 보관 기간'(archive_retention_days)을 따른다.
        #   0=무제한이면 아무것도 지우지 않음. (예전엔 별도 상수 60일이 UI 설정을 무시하고
        #   원본을 지워, 60일 넘은 웨이퍼의 RCA 가 '원본 없음'으로 실패하던 문제의 원인)
        days = _archive_retention_days()
        removed = 0
        if days > 0:
            cutoff = dt.date.today() - dt.timedelta(days=days)
            for d in list(BACKUP_DIR.glob("*")):
                if not d.is_dir():
                    continue
                try:
                    dd = dt.date.fromisoformat(d.name)
                except ValueError:
                    continue                    # 날짜 형식 아닌 폴더는 건드리지 않음
                if dd < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            if removed:
                print(f"[autopilot] 보관정리: {days}일 지난 백업 {removed}개 삭제")
        _prune_logs()                           # 로그도 같은 보관기간으로 정리(0=무제한)
        return removed

    def backup_now(self) -> int:
        """수동 마감 백업(현재 날짜 폴더로). 데모/운영 강제 마감용."""
        n = self._end_of_day(self.day)
        self._save_state()
        return n

    def status(self) -> Dict:
        try:
            backups = sorted(p.name for p in BACKUP_DIR.glob("*") if p.is_dir())
        except Exception:                       # noqa: BLE001
            backups = []
        return {
            "enabled": self.enabled,
            "day": self.day,
            "watch_interval": WATCH_INTERVAL,
            "retention_days": _archive_retention_days(),   # UI 보관기간(0=무제한) — 백업도 동일 정책
            "last_scan": self.last_scan,
            "last_event": self.last_event,
            "processed_count": len(self.processed),
            "backup_dir": str(BACKUP_DIR),
            "backup_days": backups[-RETENTION_DAYS:],
        }


# ════════════════════════════════════════════════════════════════════
#  FastAPI
# ════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: "FastAPI"):
    global AUTOPILOT
    _setup_logging()                                      # 콘솔 대신 파일 로그
    _prune_logs()                                         # 시작 시 오래된 로그 정리
    _note = CPU.apply_affinity(_CPU_BUDGET)               # 코어 고정(설정에서 켠 경우만)
    STATE["cpu_affinity_note"] = _note
    if _CPU_BUDGET.get("use_cores", 0) < _CPU_BUDGET.get("physical", 0):
        print(f"[cpu] 물리코어 {_CPU_BUDGET['use_cores']}/{_CPU_BUDGET['physical']}개 사용 "
              f"(스레드 {_CPU_BUDGET['threads']}개, 예약 {_CPU_BUDGET['reserve']}개)"
              + (f" · {_note}" if _note else " · 코어 고정 미사용"))
    _load_recipe_tags()                                   # 레시피 태그 로드
    AUTOPILOT = AutoPilot()
    AUTOPILOT.start()
    print(f"[server] AutoPilot 시작 — 감시 {'ON' if AUTOPILOT.enabled else 'OFF'} · "
          f"backup={BACKUP_DIR} · logs={LOG_DIR} (보관 {RETENTION_DAYS}일)")
    try:
        yield
    finally:
        if AUTOPILOT is not None:
            AUTOPILOT.stop()


app = FastAPI(title="챔버 이상 검사 시스템", version="0.2", lifespan=lifespan)
app.add_middleware(FreshUIResponses)
from src.app.trend_transport import TrendResponseCompression
app.add_middleware(TrendResponseCompression)


class InspectReq(BaseModel):
    folder: Optional[str] = None                           # 미지정 → INBOX
    recipes: Optional[List[str]] = None                    # 지정 시 '이 유형'만 검사(체크박스). None/빈=전체
    recipe_prefixes: Optional[List[str]] = None            # 유형 prefix (예 ['29a'])
    tool_prefixes: Optional[List[str]] = None              # 설비 prefix (파일명 첫 토큰, 예 ['WUK370A'])
    chamber_prefixes: Optional[List[str]] = None           # 챔버 prefix (C2 기반, 예 ['dsv5'])
    date_from: Optional[str] = None                        # 검사기간 시작(YYYY-MM-DD, 끝시간 기준)
    date_to: Optional[str] = None                          # 검사기간 끝(YYYY-MM-DD)
    chambers: Optional[List[str]] = None                   # 지정 시 이 DSV 챔버 폴더만(예 ['dsv1','dsv3'])


@app.get("/api/status")
@_pin_model
def api_status():
    ENGINE.load()
    if ENGINE.ready:
        _restore_model_results()
    files = sorted(p.name for p in Path(STATE["folder"]).glob("*.csv")) \
        if Path(STATE["folder"]).exists() else []
    return {
        "chamber_models": ENGINE.root.info()["chamber_models"],
        "model_ready": ENGINE.ready,
        "error": ENGINE.error,
        "model_dir": str(ENGINE.art),
        "backend": ENGINE.backend,
        "model_is_external": (ENGINE.art != PKG / "artifacts"),
        "threshold": ENGINE.threshold,
        "thresholds": ENGINE.thresholds,
        "threshold_mode": ENGINE.threshold_mode,          # 자율 인식 결과: per_recipe | global
        "score_norm": getattr(ENGINE, "score_norm", "none"),   # 모델이 점수를 정규화했는지
        "threshold_warning": ENGINE.threshold_warning,    # 임계 미설정 시 경고(없으면 null)
        "ort_provider": getattr(ENGINE.model, "active_provider", None),  # 실행 프로바이더(GPU/CPU)
        "ort_fallback": getattr(ENGINE.model, "gpu_fallback_reason", None),  # GPU→CPU 내려간 사유
        "cylinder_mode": getattr(ENGINE.cfg, "cylinder_mode", "off"),
        "cylinder_diff": getattr(ENGINE.cfg, "cylinder_diff", False),
        "auxiliary_sensors": list(auxiliary_sensor_columns(ENGINE.cfg)),
        "wafer_score_override": ENGINE.wafer_score_override,
        "chamber_input_modes": {ch: {"cylinder_mode": child.cfg.cylinder_mode,
                                       "cylinder_diff": child.cfg.cylinder_diff,
                                       "score_norm": child.score_norm,
                                       "wafer_score_override": child.wafer_score_override,
                                       "auxiliary_sensors": list(auxiliary_sensor_columns(child.cfg))}
                                for ch, child in ENGINE.root.chamber_engines.items()},
        "last_diag": STATE.get("last_diag"),              # 최근 검사 처리요약(파일수/검사/제외)
        "params": ENGINE.params,
        "folder": STATE["folder"],
        "file_count": len(files),
        "files": files[:200],
        "has_results": bool(STATE["results"]),
        "inspected_at": STATE.get("inspected_at"),
        "skip_steps": SKIP_STEPS,
        "autopilot": (AUTOPILOT.status() if AUTOPILOT is not None else None),
    }


@app.get("/api/scan-recipes")
def api_scan_recipes(folder: str = "", chambers: str = ""):
    """검사 전 유형 체크박스용: 대상 폴더의 파일들을 '앞 몇 줄만' 읽어 유형별 개수를 센다.
    folder 지정 시 그 폴더, 아니면 DSV 상시 폴더(+입력) 전체. chambers=csv 로 DSV 일부만."""
    if folder.strip():
        dirs = [(None, Path(folder.strip().strip('"')))]
    else:
        dirs = _scan_targets()
        if chambers.strip():
            want = {c.strip().lower() for c in chambers.split(",") if c.strip()}
            dirs = [(ch, p) for ch, p in dirs if (str(ch or "").lower() in want)]
    files, seen = [], set()
    for _ch, fol in dirs:
        d = _resolve_existing_dir_ci(Path(fol))
        if not d or not Path(d).exists():
            continue
        for f in _data_files(d):
            if f.stem in seen:
                continue
            seen.add(f.stem); files.append(f)
    rmap = _peek_recipes_map(files)
    from collections import Counter
    cnt = Counter(rmap.values())
    items = sorted(({"recipe": k, "count": int(v)} for k, v in cnt.items()),
                   key=lambda x: (-x["count"], x["recipe"]))
    return {"total": len(files), "recipes": items,
            "folders": [str(p) for _c, p in dirs]}


@app.post("/api/inspect")
@_track_inspection_request
def api_inspect(req: InspectReq):
    try:
        _ABORT.clear()                                    # 새 검사 시작 → 이전 '완전 중단' 플래그 해제
        _PAUSE.set()
        # 유형 체크박스: 지정 시 '이 유형'만 검사(빈/None=전체)
        rfilter = {str(x) for x in (req.recipes or []) if str(x).strip()} or None
        pfilter = {str(x).strip() for x in (req.recipe_prefixes or []) if str(x).strip()} or None
        tfilter = {str(x).strip() for x in (req.tool_prefixes or []) if str(x).strip()} or None
        cfilter = {str(x).strip() for x in (req.chamber_prefixes or []) if str(x).strip()} or None
        d_from = (req.date_from or "").strip()[:10] or None
        d_to = (req.date_to or "").strip()[:10] or None
        _fkw = dict(recipe_filter=rfilter, recipe_prefixes=pfilter, tool_prefixes=tfilter,
                    chamber_prefixes=cfilter, date_from=d_from, date_to=d_to)

        def _viewed(res):
            # ★ 배점 모드면 응답의 results/summary 를 S_w·θ_w 재판정으로 통일
            #   (개요·산점도의 배점 재판정과 '개별 검사' 탭이 어긋나지 않게)
            if not isinstance(res, dict):
                return res
            vr = _weighted_view_results(res.get("results") or [])
            if vr is not res.get("results"):
                res = {**res, "results": vr, "summary": _summary(vr)}
            return res

        if req.folder:                                    # 특정 폴더 지정 검사
            return _viewed(run_inspection(Path(req.folder), trigger="수동 검사(폴더 지정)", **_fkw))
        dirs = _scan_targets()                            # 챔버폴더(존재분) + 기본 입력폴더 모두
        if req.chambers:                                  # 선택 DSV 챔버 폴더만
            want = {str(c).strip().lower() for c in req.chambers if str(c).strip()}
            dirs = [(ch, p) for ch, p in dirs if (str(ch or "").lower() in want)]
        # 폴더별 순차 검사(첫 폴더만 기존결과 치환, 이후는 병합).
        # seen 으로 폴더 간 같은 파일명(중복)을 한 번만 검사(50개×2폴더=100 → 50).
        # ★ n 은 '이번 검사에서 처리한 전체 매수'다 — 마지막 폴더의 값이 아니다.
        #   예전에는 last(마지막 폴더 결과)를 그대로 돌려줬다. dsv1 에서 500장을 검사해도
        #   뒤쪽 dsv6 폴더가 비어 있으면 n=0 이 되어 화면에 '검사할 파일이 없습니다' 가 떴다.
        #   사용자는 dsv1 검사까지 실패한 줄 알게 된다(사용자 지적).
        #   한 폴더라도 처리했으면 경고가 뜨면 안 된다 → 합계를 돌려주고,
        #   폴더별 내역(per_folder)도 함께 실어 어디가 비었는지는 조용히 확인만 가능하게 한다.
        last = None
        seen = set()
        total_n = 0
        per_folder = []
        _diag_sum = None                                  # 폴더별 처리요약 합계
        for i, (chamber, folder) in enumerate(dirs):
            last = run_inspection(folder, replace_results=(i == 0), chamber=chamber,
                                  trigger="수동 검사(챔버 일괄)",
                                  seen_stems=seen, **_fkw)
            _n = int((last or {}).get("n") or 0)
            total_n += _n
            per_folder.append({"chamber": chamber, "folder": str(folder), "n": _n,
                               "diag": (last or {}).get("diag") or {}})
            # ★ 처리요약(last_diag)도 마지막 폴더 것만 남으면 안 된다.
            #   dsv1 에서 500장을 검사해도, 마지막 dsv6 이 '중복 3개 제외 · 0매 검사' 면
            #   화면 배너가 그 숫자를 띄워 검사가 헛돈 것처럼 보인다(경고와 같은 함정).
            with STATE_LOCK:
                _d = dict(STATE.get("last_diag") or {})
            if _d:
                if _diag_sum is None:
                    _diag_sum = {k: (list(v) if isinstance(v, list) else v) for k, v in _d.items()}
                else:
                    for k in ("files", "unique", "inspected", "dup_excluded", "csv_excluded",
                              "filtered_excluded", "failed", "cache_hits"):
                        _diag_sum[k] = int(_diag_sum.get(k) or 0) + int(_d.get(k) or 0)
                    _mr = {r["reason"]: dict(r) for r in (_diag_sum.get("fail_reasons") or [])}
                    for r in (_d.get("fail_reasons") or []):
                        if r["reason"] in _mr:
                            _mr[r["reason"]]["count"] += r["count"]
                        else:
                            _mr[r["reason"]] = dict(r)
                    _diag_sum["fail_reasons"] = sorted(
                        _mr.values(), key=lambda x: -x["count"])[:5]
        if _diag_sum is not None:
            with STATE_LOCK:
                STATE["last_diag"] = _diag_sum            # 화면 배너는 '이번 검사 전체' 기준
        if last is not None:
            return {**_viewed(last), "n": total_n, "per_folder": per_folder,
                    "diag": _diag_sum or {}}
        return {"folder": "", "n": 0, "per_folder": per_folder,
                                           "results": _weighted_view_results(STATE["results"]),
                                           "summary": _summary(_weighted_view_results(STATE["results"]))}
    except InspectAborted:
        _ABORT.clear()                                    # 소진 — 다음 검사에 영향 없게
        # ★ '파일 확인' 단계는 run_inspection 의 정리(try) 밖이라, 그 단계에서 중단되면
        #   진행상태가 active/"파일 확인" 으로 남아 UI 가 멈춘 것처럼 보였음 → 여기서 확실히 닫는다.
        _set_progress(False, 0, 0, "중단", "")
        vr = _weighted_view_results(STATE["results"])
        return {"aborted": True, "folder": STATE.get("folder") or "",
                "n": len(vr), "results": vr, "summary": _summary(vr)}
    except Exception as e:                                 # noqa: BLE001
        traceback.print_exc()
        _set_progress(False, 0, 0, "오류", "")            # 준비단계 오류도 진행바가 남지 않게
        raise HTTPException(status_code=400, detail=str(e))


# ══════════════════════════════════════════════════════════════════
#  브라우저를 닫으면 앱도 끝나게 — 화면 연결 감시
# ══════════════════════════════════════════════════════════════════
#  문제: 브라우저로 여는 방식(launcher)은 창을 닫아도 uvicorn 이 계속 돈다.
#        사용자는 앱을 껐다고 생각하는데 프로세스가 작업관리자에 남고,
#        다시 실행하면 포트가 물려 있어 엉킨다.
#        (WebView2 창 방식(desktop.py)은 창을 닫으면 이미 종료된다 — 그쪽과 맞춘다)
#  방법: 화면이 주기적으로 살아 있다고 알리고(10초), 창을 닫을 때 작별을 보낸다.
#        일정 시간 아무 화면도 붙어 있지 않으면 앱을 종료한다.
#  안전장치
#    · 검사·리포트·학습 중에는 절대 종료하지 않는다(작업 유실 방지).
#    · 새로고침·탭 이동으로 잠깐 끊기는 것과 구분하려고 유예를 둔다(기본 45초).
#    · 아직 한 번도 화면이 붙지 않았으면 더 오래 기다린다(브라우저가 늦게 뜨는 경우).
#    · IO_KEEP_ALIVE=1 이면 이 기능을 끈다 — 화면 없이 자동감시만 계속 돌릴 때.
_CLIENTS: Dict[str, float] = {}                       # 화면 id → 마지막 신호 시각
_CLIENTS_LOCK = threading.Lock()
# ★ 45→180초: 화면을 잠그면 브라우저가 숨은 탭의 타이머를 분당 1회로 줄인다
#   (Chrome 집중 스로틀링). 신호 간격이 60초로 벌어지면 45초 기준으로는 '화면이
#   닫혔다'로 오인해 잠금만 해도 앱이 꺼졌다. 창을 정말 닫을 때는 별도의 즉시
#   신호(beforeunload)가 오므로, 이 값은 그 신호를 놓쳤을 때의 안전망일 뿐이다.
_EXIT_IDLE = float(os.environ.get("IO_EXIT_IDLE", "180"))       # 마지막 화면이 사라진 뒤 대기(초)
_EXIT_BOOT_GRACE = float(os.environ.get("IO_EXIT_BOOT", "300"))  # 첫 화면을 기다리는 시간(초)
_EXIT_ENABLED = os.environ.get("IO_KEEP_ALIVE", "0") != "1"


def _app_busy() -> tuple:
    """지금 끝내면 안 되는 작업이 도는가 → (bool, 사유)."""
    try:
        with STATE_LOCK:
            pg = dict(STATE.get("progress") or {})
            bg = dict(STATE.get("progress_bg") or {})
        if pg.get("active"):
            return True, "검사 중"
        if bg.get("active"):
            return True, "자동검사 중"
    except Exception:                                 # noqa: BLE001
        return True, "상태 확인 불가"
    try:
        if REPORT.get("running"):
            return True, "리포트 생성 중"
    except Exception:                                 # noqa: BLE001
        pass
    try:
        if TRAINER is not None:
            with TRAINER.lock:
                _js = list(TRAINER.jobs.values())
            for j in _js:
                if str(j.get("status")) in ("running", "paused", "finalizing"):
                    return True, "학습 중"
    except Exception:                                 # noqa: BLE001
        pass
    return False, ""


APP_MARK = "chamber-inspect"                           # 이미 도는 앱인지 확인하는 표식
APP_VERSION = os.environ.get("IO_APP_VERSION", "")     # 있으면 표기(없어도 동작)


@app.get("/api/ping-app")
def api_ping_app():
    """이 포트에 떠 있는 것이 '이 앱' 인지 확인용.

    ★ 왜 필요한가 — 앱을 껐다 켤 때마다 프로세스가 쌓이던 원인이 여기였다.
      예전에는 8000 번이 차 있으면 아무 말 없이 다른 빈 포트로 '새 앱'을 하나 더
      띄웠다. 브라우저만 닫고 앱을 다시 켜기를 반복하면 그때마다 앱이 하나씩
      늘어나 메모리를 먹고, 로그 파일도 여러 프로세스가 물고 있어 압축조차 안 됐다.
      이 표식으로 '이미 떠 있는 내 앱'을 알아보고, 그 화면만 열고 새로 띄우지 않는다.
      (8000 번을 다른 프로그램이 쓰고 있으면 표식이 안 맞으므로 예전처럼 다른 포트를 쓴다)
    """
    return {"app": APP_MARK, "pid": os.getpid(), "version": APP_VERSION, **IDENTITY}


@app.post("/api/alive")
@app.get("/api/alive")
def api_alive(cid: str = ""):
    """화면이 살아 있다는 신호(10초마다)."""
    if cid:
        with _CLIENTS_LOCK:
            _CLIENTS[str(cid)] = time.time()
            # 첫 5초 내 열고 닫아도 '아직 화면이 한 번도 안 열림'으로 오인하지 않는다.
            _exit_watch_loop._seen = True
    return {"ok": True, "idle_exit": _EXIT_ENABLED}


@app.post("/api/bye")
def api_bye(cid: str = ""):
    """화면을 닫았다는 작별(sendBeacon). 즉시 목록에서 뺀다."""
    if cid:
        with _CLIENTS_LOCK:
            _CLIENTS.pop(str(cid), None)
    return {"ok": True}


def _exit_watch_loop():
    """화면이 모두 사라지면 앱을 종료한다(작업 중이면 끝날 때까지 기다린다)."""
    t0 = time.time()
    _prev = time.time()
    while True:
        time.sleep(5.0)
        try:
            now = time.time()
            # ★ 절전/최대절전 감지 — 5초 자야 할 루프가 한참 뒤에 깨어났다면 PC 가
            #   잠들었던 것이다. 그 사이 화면 신호가 없던 건 브라우저 탓이 아니므로,
            #   모든 화면의 신호 시각을 지금으로 갱신해 방금 깬 PC 에서 앱이
            #   곧바로 꺼지는 일을 막는다(브라우저가 다시 신호를 보낼 시간을 준다).
            if now - _prev > 60.0:
                print(f"[exit] 절전에서 복귀 감지({now - _prev:.0f}초 공백) — "
                      f"화면 신호 대기 재시작", flush=True)
                with _CLIENTS_LOCK:
                    for k in list(_CLIENTS):
                        _CLIENTS[k] = now
            _prev = now
            with _CLIENTS_LOCK:
                for k, v in list(_CLIENTS.items()):
                    if now - v > _EXIT_IDLE:          # 신호가 끊긴 화면은 정리
                        _CLIENTS.pop(k, None)
                alive = len(_CLIENTS)
                seen_any = bool(_CLIENTS) or getattr(_exit_watch_loop, "_seen", False)
            if alive:
                _exit_watch_loop._seen = True         # 한 번이라도 붙었음
                continue
            if not getattr(_exit_watch_loop, "_seen", False):
                if now - t0 < _EXIT_BOOT_GRACE:       # 아직 브라우저가 뜨는 중일 수 있다
                    continue
            busy, why = _app_busy()
            if busy:
                continue                              # 작업 끝날 때까지 기다린다
            print(f"[exit] 열린 화면이 없어 앱을 종료합니다 "
                  f"(마지막 화면 종료 후 {_EXIT_IDLE:g}초 경과)", flush=True)
            _shutdown_now()
        except Exception:                             # noqa: BLE001
            traceback.print_exc()


def _shutdown_now(code: int = 0):
    """자식 프로세스까지 정리하고 즉시 완전 종료(desktop.py 의 _hard_shutdown 과 같은 뜻)."""
    try:
        if AUTOPILOT is not None:
            AUTOPILOT.stop()
    except Exception:                                 # noqa: BLE001
        pass
    try:
        import multiprocessing
        for ch in multiprocessing.active_children():
            try:
                ch.terminate()
            except Exception:                         # noqa: BLE001
                pass
    except Exception:                                 # noqa: BLE001
        pass
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


def start_exit_watch():
    """브라우저로 여는 방식(launcher)에서만 켠다."""
    if not _EXIT_ENABLED:
        print("[exit] IO_KEEP_ALIVE=1 — 화면을 닫아도 계속 실행합니다", flush=True)
        return
    threading.Thread(target=_exit_watch_loop, name="exit-watch", daemon=True).start()
    print(f"[exit] 화면 연결 감시 시작 — 모든 화면이 닫히고 {_EXIT_IDLE:g}초 뒤 종료"
          f"(검사·리포트·학습 중이면 끝난 뒤)", flush=True)


@app.get("/api/progress")
def api_progress():
    """전면 검사 진행률(수동/다시검사). UI progress bar 폴링용.
    백그라운드 자동감시 진행률은 별도 'bg' 필드로만 제공(전면바와 섞여 깜빡이지 않게)."""
    empty = {"active": False, "done": 0, "total": 0, "phase": "", "folder": ""}
    with STATE_LOCK:
        fg = dict(STATE.get("progress") or empty)
        fg["active"] = bool(fg["active"] or STATE.get("score_inspect_requests", 0))
        fg["bg"] = dict(STATE.get("progress_bg") or empty)
        _bd = STATE.get("bg_done")
        if _bd and (time.time() - _bd["ts"]) < 6.0:       # 최근 완료 흔적(6초 유지)
            fg["bg_done"] = dict(_bd)
    fg["paused"] = not _PAUSE.is_set()                    # 일시중단 상태(수동/자동 공통)
    return fg


class PauseReq(BaseModel):
    paused: bool


@app.post("/api/inspect/pause")
def api_inspect_pause(req: PauseReq):
    """검사 일시중단/재개. paused=true → 중단(현재 청크까지 마치고 대기), false → 재개."""
    if req.paused:
        _PAUSE.clear()
    else:
        _PAUSE.set()
    return {"paused": not _PAUSE.is_set()}


@app.post("/api/inspect/abort")
def api_inspect_abort():
    """Stop foreground and automatic inspection; preserve completed archives."""
    if AUTOPILOT is not None:
        AUTOPILOT.set_enabled(False)  # Persist OFF so restart cannot resume the old queue.
    _ABORT.set()
    _PAUSE.set()                                          # 일시중단 대기 스레드를 깨워 즉시 탈출시킴
    return {"aborting": True}


class ChamberPathsReq(BaseModel):
    paths: Dict[str, str]
    wuk380b_mapping: Optional[bool] = None


@app.get("/api/chamber-paths")
def api_chamber_paths():
    """챔버(dsv1~6)→데이터 폴더 경로 + 존재여부."""
    paths = _chamber_paths()
    return {"paths": paths, "defaults": CHAMBER_PATH_DEFAULTS,
            "wuk380b_mapping": _load_app_config().get("wuk380b_mapping") is True,
            "restart_required": (_load_app_config().get("wuk380b_mapping") is True) != _WUK380B_OVERRIDE,
            "exists": {ch: Path(p).is_dir() if p else False for ch, p in paths.items()}}


@app.post("/api/chamber-paths")
def api_set_chamber_paths(req: ChamberPathsReq):
    _cancel_for_path_change()
    _set_chamber_paths(req.paths or {}, req.wuk380b_mapping)
    _cancel_for_path_change()
    paths = _chamber_paths()
    return {"paths": paths,
            "wuk380b_mapping": _load_app_config().get("wuk380b_mapping") is True,
            "restart_required": (_load_app_config().get("wuk380b_mapping") is True) != _WUK380B_OVERRIDE,
            "exists": {ch: Path(p).is_dir() if p else False for ch, p in paths.items()}}


class PickReq(BaseModel):
    initial: Optional[str] = None


@app.get("/api/list-dirs")
def api_list_dirs(path: str = ""):
    """폴더 목록 — 화면 안 폴더 탐색기용.

    ★ 왜 OS 폴더 선택 창을 버렸는가
      윈도우 폴더 대화상자는 셸이 만드는 창이라 소유자의 '항상 위'를 물려받지 않고,
      윈도우7 의 포그라운드 잠금 때문에 백그라운드 프로세스(powershell)가 앞으로
      끌어올릴 수도 없다. 두 번 고쳐도 실기에서 계속 브라우저 뒤에 떴다.
      브라우저 '안' 에서 고르게 하면 z-순서 문제가 원천적으로 없다.
      (경로 입력이 서버로 오가지만 서버=사용자 PC 라 보안 경계는 그대로다)
    """
    p = (path or "").strip().strip('"')
    out = {"path": "", "parent": None, "dirs": [], "drives": [], "error": None}
    try:
        if os.name == "nt":
            out["drives"] = [f"{c}:\\" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                             if Path(f"{c}:\\").exists()]
        else:
            out["drives"] = ["/"]
        if not p:                                         # 시작 위치 = 앱 실행 폴더
            p = str(_WRITABLE)
        q = Path(p)
        if not q.is_dir():
            out["error"] = "폴더가 없습니다"; out["path"] = p
            return out
        q = q.resolve()
        out["path"] = str(q)
        out["parent"] = (str(q.parent) if q.parent != q else None)
        ds = []
        with os.scandir(q) as it:
            for e in it:
                try:
                    if e.is_dir() and not e.name.startswith(("$", "."))                       and e.name.lower() not in ("system volume information",):
                        ds.append(e.name)
                except OSError:
                    continue
                if len(ds) >= 800:                        # 폭주 방지(그 위는 직접 입력)
                    break
        ds.sort(key=str.lower)
        out["dirs"] = ds
    except PermissionError:
        out["error"] = "접근 권한이 없는 폴더입니다"
    except Exception as e:                                # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@app.post("/api/pick-folder")
def api_pick_folder(req: PickReq):
    """Windows 네이티브 '폴더 선택' 창을 띄우고 고른 경로를 반환(취소=빈 문자열).
    로컬 앱이라 창이 사용자 화면에 뜬다. 비-Windows 는 400."""
    if os.name != "nt":
        raise HTTPException(400, "폴더 선택 창은 Windows 에서만 지원됩니다. 경로를 직접 입력하세요.")
    _PICK_ERR["msg"] = ""
    path = _pick_folder_dialog(req.initial or "")
    # note 가 있으면 UI 가 '취소'가 아니라 '띄울 수 없음'으로 안내한다.
    return {"path": path, "note": _PICK_ERR.get("msg") or ""}


class InboxReq(BaseModel):
    folder: str


@app.post("/api/set-inbox")
def api_set_inbox(req: InboxReq):
    """검사 대상 폴더 등록(영속). 이후 자동감시·검사가 이 폴더를 본다."""
    d = (req.folder or "").strip().strip('"')
    p = Path(d)
    if not d or not p.exists() or not p.is_dir():
        raise HTTPException(400, f"폴더가 없습니다: {d or '(빈 값)'}")
    _cancel_for_path_change()
    _set_inbox(str(p))
    _cancel_for_path_change()
    with STATE_LOCK:
        STATE["folder"] = str(p)
    n = len(list(p.glob("*.csv")))
    return {"folder": str(p), "file_count": n}


def _list_models() -> List[Dict]:
    """선택 가능한 모델 목록.
      · 내장(기본)  · exe 옆 model\\ · artifacts\\  · models\\<이름>\\ 하위 폴더들
    각 항목은 recon_A.pt 가 있어야 유효. dir 로 중복 제거.
    """
    found, seen = [], set()

    def add(name, d, bundled=False):
        d = Path(d)
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen or not _has_model(d):
            return
        seen.add(key)
        found.append({"name": name, "dir": str(d), "bundled": bundled})

    add("내장 모델 (기본)", PKG / "artifacts", bundled=True)
    if os.environ.get("IO_ARTIFACTS"):
        add(Path(os.environ["IO_ARTIFACTS"]).name or "IO_ARTIFACTS",
            os.environ["IO_ARTIFACTS"])
    _ms = _model_storage_root()
    if _ms is not None:
        add(_ms.name or "모델(저장소)", _ms)              # 저장소 지정 모델 루트(직접)
    for base in (_WRITABLE, _WRITABLE.parent):           # 모델\ 직접(exe옆·상위)
        add("모델", base / "모델"); add("model", base / "model")
        add("artifacts", base / "artifacts")
    for root in _model_roots():
        if has_chamber_layout(root):
            add(root.name, root)  # 챔버 모델 묶음도 선택 목록에서는 하나의 모델
    for sub in _submodel_dirs():                         # 하위 폴더별 모델(모델\0713 …)
        add(sub.name, sub)
    for cd in _load_custom_dirs():                        # 사용자가 직접 입력한 경로
        add("📁 " + Path(cd).name, cd)
    bundles = {Path(m["dir"]).resolve() for m in found
               if has_chamber_layout(m["dir"])}
    return [m for m in found
            if not (re.fullmatch(r"dsv[1-6]", Path(m["dir"]).name, re.I)
                    and Path(m["dir"]).resolve().parent in bundles)]


@app.get("/api/models")
def api_models():
    dm = _load_app_config().get("default_model")
    default_dir = str(PKG / "artifacts") if dm == "__bundled__" else (str(_path_load(dm)) if dm else dm)
    return {"models": _list_models(), "current_dir": str(ENGINE.art),
            "default_dir": default_dir, "chamber_models": ENGINE.root.info()["chamber_models"]}


class SelectReq(BaseModel):
    dir: str


# 사용자가 직접 입력한 모델 폴더(드롭다운에 유지). data/custom_models.json 에 영속화.
_CUSTOM_FILE = DATA_DIR / "custom_models.json"


def _load_custom_dirs() -> List[str]:
    try:
        return list(json.loads(_CUSTOM_FILE.read_text(encoding="utf-8")))
    except Exception:                                    # noqa: BLE001
        return []


def _add_custom_dir(d: str):
    dirs = _load_custom_dirs()
    if d not in dirs:
        dirs.append(d)
        try:
            _CUSTOM_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CUSTOM_FILE.write_text(json.dumps(dirs, ensure_ascii=False), encoding="utf-8")
        except Exception:                                # noqa: BLE001
            pass


def _probe_model_worker(art: str, q):
    """자식 프로세스에서 모델을 한 번 로드해 본다(성공 여부만 큐로 보고)."""
    try:
        import os as _os
        _os.environ["IO_PROBE_CHILD"] = "1"                # 자식 안에서 또 프로브하지 않도록
        _os.environ.setdefault("IO_ORT_PROVIDER", "cpu")   # 프로브는 CPU 로만(자원 경합 방지)
        e = Engine()
        e.selected = str(art)
        e.load()
        # ★ load() 는 실패해도 예외 대신 error 를 남기고 넘어간다(세션 생성 실패 → CPU 폴백 등).
        #   ready 를 확인하지 않으면 '열리지 않는 모델'을 통과시켜 버린다.
        if not getattr(e, "ready", False):
            q.put(("err", str(getattr(e, "error", "") or "모델을 열지 못했습니다")))
        else:
            q.put(("ok", e.backend or "?"))
    except Exception as ex:                              # noqa: BLE001
        q.put(("err", f"{type(ex).__name__}: {ex}"))


def _probe_canary_worker(q):
    q.put("ok")


_PROBE_OK: Dict = {"checked": False, "usable": False}


def _probe_supported() -> bool:
    """이 실행 환경에서 '자식 프로세스 띄우기'가 되는지 1회 확인(캐시).

    ★ 반드시 필요하다: 자식 프로세스를 못 띄우는 환경(개발용 stdin 실행 등)에서는
      프로브가 항상 실패하는데, 그걸 '모델이 나쁘다'로 오해하면 멀쩡한 모델까지
      막아 버린다. 카나리아가 실패하면 프로브를 아예 쓰지 않는다.
    """
    if _PROBE_OK["checked"]:
        return _PROBE_OK["usable"]
    _PROBE_OK["checked"] = True
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        pr = ctx.Process(target=_probe_canary_worker, args=(q,), daemon=True)
        pr.start(); pr.join(60)
        if pr.is_alive():
            try:
                pr.kill()
            except Exception:                            # noqa: BLE001
                pass
            _PROBE_OK["usable"] = False
        else:
            _PROBE_OK["usable"] = (pr.exitcode == 0 and not q.empty())
    except Exception:                                    # noqa: BLE001
        _PROBE_OK["usable"] = False
    if not _PROBE_OK["usable"]:
        print("[model] 이 환경에서는 모델 사전 점검을 쓸 수 없어 건너뜁니다", flush=True)
    return _PROBE_OK["usable"]


_PROBE_LOCK = threading.Lock()
_PROBE_CACHE: Dict[str, tuple] = {}                       # {경로: (시각, ok, why)}
_PROBE_OK_TTL, _PROBE_ERR_TTL = 600.0, 60.0               # 성공 10분 · 실패 1분 캐시


def _probe_model_dir(art: str, timeout: float = 90.0):
    """모델 폴더를 '별도 프로세스'에서 먼저 열어 본다. 반환 (ok, 메시지).

    ★ 왜: 손상됐거나 이 PC 에서 지원되지 않는 모델을 로드하면 파이썬 예외가 아니라
      네이티브 레벨에서 프로세스가 통째로 죽는 경우가 있다. 그러면 앱이 사라지고
      화면에는 'Failed to fetch' 만 뜬다(원인 알 수 없음).
      자식 프로세스에서 미리 열어 보면, 죽더라도 자식만 죽고 앱은 살아서
      '이 모델은 열 수 없습니다' 라고 이유를 돌려줄 수 있다.
    프로브 자체가 불가능한 환경이면 (True, "") 로 통과시키고 기존처럼 직접 로드한다.
    """
    if has_chamber_layout(Path(art)):
        timeout = max(timeout, 330.0)
    if os.environ.get("IO_PROBE_CHILD") == "1":
        return True, ""                                  # 프로브 자식 자신 → 재귀 방지
    if not _probe_supported():
        return True, ""                                  # 점검 불가 환경 → 기존처럼 직접 로드
    # ★ 결과 캐시 — 같은 폴더를 방금 점검했으면 자식을 다시 띄우지 않는다.
    try:
        _key = str(Path(str(art)).resolve())
    except Exception:                                    # noqa: BLE001
        _key = str(art)
    _hit = _PROBE_CACHE.get(_key)
    if _hit and (time.time() - _hit[0]) < (_PROBE_OK_TTL if _hit[1] else _PROBE_ERR_TTL):
        return _hit[1], _hit[2]
    # ★ 동시 프로브는 '거절'이 아니라 '대기'다. 예전엔 즉시 '다른 점검 진행 중'을
    #   돌려줘서, 시작 자동 점검(깨진 폴더면 오래 걸림)이 도는 동안 모델 전환·
    #   기본모델 지정이 전부 실패했고, UI 는 이름만 바뀐 채 옛 모델이 계속 돌았다.
    if not _PROBE_LOCK.acquire(timeout=180.0):
        return False, "모델 점검 대기가 너무 깁니다 — 잠시 뒤 다시 시도하세요."
    try:
        _hit = _PROBE_CACHE.get(_key)                     # 줄 서는 동안 끝났으면 재사용
        if _hit and (time.time() - _hit[0]) < (_PROBE_OK_TTL if _hit[1] else _PROBE_ERR_TTL):
            return _hit[1], _hit[2]
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(target=_probe_model_worker, args=(str(art), q), daemon=True)
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:                            # noqa: BLE001
                pass
            _r = (False, f"모델을 여는 데 {int(timeout)}초를 넘겼습니다. 파일이 너무 크거나 손상됐을 수 있습니다.")
        elif proc.exitcode not in (0, None):
            _r = (False, ("모델을 여는 중 프로세스가 비정상 종료했습니다"
                          f"(코드 {proc.exitcode}). 모델 파일이 손상됐거나 이 PC 에서 "
                          "지원되지 않는 형식일 수 있습니다. 다른 모델 폴더를 지정해 주세요."))
        else:
            _r = (True, "")
            if not q.empty():
                kind, msg = q.get()
                if kind == "err":
                    _r = (False, f"모델을 열 수 없습니다 — {msg}")
        _PROBE_CACHE[_key] = (time.time(), *_r)
        return _r
    except Exception as e:                               # noqa: BLE001
        print(f"[model] 사전 점검을 건너뜁니다({e})", flush=True)
        return True, ""
    finally:
        _PROBE_LOCK.release()


def _restore_model_results(force=False):
    """Restore selected model's persisted result list without loading raw arrays."""
    with ENGINE._switch_lock:
        if getattr(ENGINE, "_root", ENGINE.root) is not ENGINE.root:
            return len(STATE["results"])  # a late request pinned to the previous model
        mid = _model_id()
        if not force and STATE.get("result_model") == mid:
            return len(STATE["results"])
        if any((STATE.get(key) or {}).get("active") for key in ("progress", "progress_bg")):
            return len(STATE["results"])
        _RELIVE_CACHE["key"] = None
        _AGG_CACHE.clear()
        rows = _history_for_current_model()
        with STATE_LOCK:
            STATE["results"] = [dict(row) for row in rows]
            STATE["recs"] = {}
            STATE["coverage"] = None
            STATE["inspected_at"] = None
            STATE["last_diag"] = None
            STATE["result_model"] = mid
        return len(rows)

@app.post("/api/model/select")
def api_model_select(req: SelectReq):
    """모델 폴더로 전환하고 해당 모델의 보관된 검사 결과를 복원."""
    d = (req.dir or "").strip().strip('"')
    p = Path(d)
    if not d or not p.exists() or not p.is_dir():
        raise HTTPException(400, f"폴더가 없습니다: {d or '(빈 값)'}")
    if not _has_model(p):
        raise HTTPException(400,
            f"이 폴더에 모델 파일이 없습니다: {d}\n"
            f"  · *.onnx + 짝 *.npz (권장)  또는  *.pt 가 있어야 합니다.")
    # ★ 먼저 자식 프로세스에서 열어 본다 — 죽는 모델이어도 앱은 살아남는다.
    print(f"[model] 전환 시도: {p}", flush=True)
    ok, why = _probe_model_dir(str(p))
    if not ok:
        print(f"[model] 사전 점검 실패: {why}", flush=True)
        raise HTTPException(400, why)
    with ENGINE._switch_lock:
        _add_custom_dir(str(p))                               # 드롭다운에 기억
        info = ENGINE.select(str(p))
        if not info.get("ready"):
            raise HTTPException(400, info.get("error") or "모델 로드 실패")
        print(f"[model] 전환 완료: backend={info.get('backend')} · ready={info.get('ready')}", flush=True)
        _invalidate_ok_bands()                                # 모델 바뀌면 OK 밴드 재로딩
        info["restored_results"] = _restore_model_results(force=True)
        dates = [str(row.get("date") or row.get("ts") or "")[:10] for row in STATE["results"]]
        dates = [day for day in dates if day]
        info["restored_range"] = {"start": min(dates), "end": max(dates)} if dates else None
        # 모델 변경 시 리포트는 '삭제하지 않는다' — 리포트는 모델별 폴더로 분리 보관되므로
        # 다른 모델(방금까지 쓰던 모델 포함)의 리포트는 그대로 남는다. 나중에 그 모델로
        # 되돌아오면 예전 리포트가 다시 보인다.
        if AUTOPILOT is not None:
            with AUTOPILOT._plock:
                AUTOPILOT.processed = set()
            AUTOPILOT._save_state()
        if not info["ready"]:
            raise HTTPException(400, info.get("error") or "모델 로드 실패")
        return info

@app.post("/api/model/set-default")
def api_model_set_default(req: SelectReq):
    """이 모델 폴더를 '기본 모델'로 저장 → 다음 실행부터 자동으로 이 모델을 로드."""
    d = (req.dir or "").strip().strip('"')
    p = Path(d)
    if not d or not p.exists() or not _has_model(p):
        raise HTTPException(400, f"이 폴더에 모델이 없습니다: {d or '(빈 값)'}")
    try:
        is_bundled = p.resolve() == (PKG / "artifacts").resolve()
    except Exception:                                    # noqa: BLE001
        is_bundled = False
    # ★ 기본 모델은 '다음 실행 때 자동 로드'된다. 열리지 않는 모델을 기본으로 저장하면
    #   다음 실행부터 앱이 계속 죽는다 → 저장 전에 자식 프로세스로 열어 본다.
    ok, why = _probe_model_dir(str(p))
    if not ok:
        raise HTTPException(400, "기본 모델로 지정할 수 없습니다 — " + why)
    val = "__bundled__" if is_bundled else str(p.resolve())
    if val != "__bundled__":
        val = _path_store(val)                            # exe 내부 경로는 상대 저장(이동 내성)
    cfg = _load_app_config(); cfg["default_model"] = val; _save_app_config(cfg)
    return {"ok": True, "default_model": val,
            "default_dir": str(PKG / "artifacts") if is_bundled else val}


@app.post("/api/model/clear-default")
def api_model_clear_default():
    """기본 모델 지정 해제 → 다음 실행부터 자동 탐색으로 복귀."""
    cfg = _load_app_config()
    cfg.pop("default_model", None)
    _save_app_config(cfg)
    return {"ok": True}


@app.get("/api/sensors")
def api_sensors():
    """현재 모델이 사용하는 '정해진 센서' 목록(순서대로) + IO + 센서 별칭.
    여러 센서가 들어와도 이 목록의 센서만, 이 순서로 뽑아서 검사한다."""
    A = ENGINE.model
    def _cols(attr):
        try:
            models = [e.model for e in ENGINE.root.chamber_engines.values()] or [A]
            return list(dict.fromkeys(str(c) for model in models for c in getattr(model, attr)))
        except Exception:                                 # noqa: BLE001
            return []
    aliases = _load_app_config().get("sensor_aliases") or {}
    return {
        "input_sensors": _cols("input_sensor_cols"),      # 모델 입력(과거) 센서
        "recon_sensors": _cols("sensor_cols"),            # 복원(예측) 센서
        "io": _cols("io_cols"),                           # IO 명령
        "aliases": aliases,                               # {데이터명: 모델센서명}
    }


class AliasReq(BaseModel):
    aliases: Dict[str, str] = {}


@app.post("/api/sensor-aliases")
def api_sensor_aliases(req: AliasReq):
    """센서 별칭 저장: 데이터 컬럼명 → 모델 센서명. 검사 시 정렬 전에 적용.
    빈 문자열/자기자신 매핑은 제거. 즉시 반영(다음 검사부터)."""
    clean = {}
    for k, v in (req.aliases or {}).items():
        k = str(k).strip(); v = str(v).strip()
        if k and v and k != v:
            clean[k] = v
    cfg = _load_app_config()
    cfg["sensor_aliases"] = clean
    _save_app_config(cfg)
    return {"ok": True, "aliases": clean}


@app.post("/api/model/reload")
def api_model_reload():
    """폴더의 웨이트를 다시 읽어 모델 교체 + '이 모델'의 그림 캐시만 비움.
    (같은 폴더의 웨이트가 바뀌었으므로 이 모델의 예전 그림은 무효 → 재생성 유도.
     다른 모델의 리포트는 보존.)"""
    with ENGINE._switch_lock:
        info = ENGINE.reload()
        if not info.get("ready"):
            raise HTTPException(400, info.get("error") or "모델 로드 실패")
        _invalidate_ok_bands()                               # 모델 바뀌면 OK 밴드 재로딩
        info["restored_results"] = _restore_model_results(force=True)
        dates = [str(row.get("date") or row.get("ts") or "")[:10] for row in STATE["results"]]
        dates = [day for day in dates if day]
        info["restored_range"] = {"start": min(dates), "end": max(dates)} if dates else None
        _clear_report_images()                               # 현재 모델 폴더만 삭제
        if AUTOPILOT is not None:                            # 재검사 위해 처리기록 초기화
            with AUTOPILOT._plock:
                AUTOPILOT.processed = set()
            AUTOPILOT._save_state()
        if not info["ready"]:
            raise HTTPException(400, info.get("error") or "모델 로드 실패")
        return info

class ThresholdReq(BaseModel):
    value: float


@app.post("/api/model/threshold")
def api_set_model_threshold(req: ThresholdReq):
    """현재 모델의 '판정 기준 임계(원점수)'를 설정 → 모델 폴더 search_best.json 에 기록 후 재로드.

    모델 폴더에 search_best.json 이 없으면 점수가 원본(정규화 안 됨)으로 나온다.
    학습 때의 판정 임계값(원점수)을 여기서 넣으면 점수 = 원점수/임계(=1 기준)로 정규화된다."""
    if ENGINE.root.chamber_engines:
        raise HTTPException(400, "챔버별 모델은 각 챔버 폴더의 모델·임계를 수정한 뒤 새로고침하세요.")
    art = Path(ENGINE.art)
    if not art.exists() or not art.is_dir():
        raise HTTPException(400, "모델 폴더가 없습니다. 먼저 모델을 로드하세요.")
    try:
        v = float(req.value)
        if not (v > 0 and np.isfinite(v)):
            raise ValueError
    except (TypeError, ValueError):
        raise HTTPException(400, "임계는 0보다 큰 숫자여야 합니다.")
    sb = art / "search_best.json"
    try:
        root = json.loads(sb.read_text(encoding="utf-8")) if sb.exists() else {}
    except Exception:                                    # noqa: BLE001
        root = {}
    if not isinstance(root, dict):
        root = {}
    val = root.setdefault("best", {}).setdefault("val", {})
    val["threshold_raw"] = v
    val["threshold"] = v
    try:
        sb.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:                               # noqa: BLE001
        raise HTTPException(400, f"저장 실패(모델 폴더 쓰기 권한 확인): {e}")
    info = ENGINE.reload()
    if not info.get("ready"):
        raise HTTPException(400, info.get("error") or "재로드 실패")
    return {"threshold": ENGINE.threshold, "ready": True}


@app.post("/api/model/convert")
def api_model_convert(req: SelectReq = None):
    """모델 폴더의 .pt → ONNX 수동 변환 후 재로드.

    dir 를 주면 그 폴더, 없으면 현재 로드된 모델 폴더를 변환한다.
    (설정 화면 '모델 변환(.pt→ONNX)' 버튼용. 변환은 torch 있는 환경에서만 가능.)"""
    if ENGINE.root.chamber_engines:
        raise HTTPException(400, "챔버별 모델은 각 챔버 폴더의 모델·임계를 수정한 뒤 새로고침하세요.")
    d = (req.dir or "").strip().strip('"') if (req and req.dir) else str(ENGINE.art)
    p = Path(d)
    if not p.exists() or not p.is_dir():
        raise HTTPException(400, f"폴더가 없습니다: {d}")
    if _find_pt(p) is None:
        raise HTTPException(400, f"변환할 .pt 파일이 없습니다: {d}")
    conv = _auto_convert_pt(p)
    if not conv.get("ok"):
        raise HTTPException(400, conv.get("msg") or ".pt 변환 실패")
    # 현재 로드된 모델 폴더를 변환했으면 재로드해 onnx 백엔드로 전환
    if str(p) == str(ENGINE.art) or (ENGINE.selected and str(p) == str(ENGINE.selected)):
        info = ENGINE.reload()
        conv["reloaded"] = True
        conv["backend"] = info.get("backend")
    return conv


@app.get("/api/results")
@_pin_model
def api_results():
    ENGINE.load()
    if ENGINE.ready:
        _restore_model_results()
    return {"folder": STATE["folder"], "results": STATE["results"],
            "summary": _summary(STATE["results"]),
            "coverage": STATE.get("coverage"),
            "inspected_at": STATE.get("inspected_at")}


# ════════════════════════════════════════════════════════════════════
#  리포트 — 전체 웨이퍼 overlay/heatmap 일괄 생성 + predictions.csv
# ════════════════════════════════════════════════════════════════════
REPORT: Dict = {"running": False, "done": 0, "total": 0, "started": None,
                "finished": None, "csv": None, "errors": 0, "current": None}
_REPORT_LOCK = threading.Lock()
REPORT_SENSOR_N = int(os.environ.get("IO_REPORT_SENSORS", "12"))   # 리포트에 담을 상위 센서 수


def _archive_date_of(wid: str):
    """아카이브에서 이 웨이퍼(.npz)가 있는 날짜폴더명. 없으면 끝시각 날짜."""
    safe = _safe(wid)
    try:
        for dd in sorted(ARCHIVE_DIR.glob("*"), reverse=True):
            if dd.is_dir() and _archive_path(dd.name, wid).exists():
                return dd.name
    except Exception:                                    # noqa: BLE001
        pass
    return _end_ts_and_date(wid)[1]


def _make_wafer_report(wid: str) -> List[str]:
    """웨이퍼 1장의 리포트 이미지 생성: 3종 + 상위 센서 overlay N종. 반환=만든 종류.
    현재 검사분(STATE)에 없으면 아카이브(보관 데이터)로 렌더해 리포트 폴더에 저장."""
    if wid in STATE["recs"]:
        made = []
        for kind in ("waferaccum", "overlay", "heatmap"):
            try:
                ensure_plot(wid, kind); made.append(kind)
            except Exception:                            # noqa: BLE001
                REPORT["errors"] += 1; traceback.print_exc()
        for rank in range(max(0, REPORT_SENSOR_N)):      # 개별 센서 overlay(상위 N)
            try:
                ensure_sensor_plot(wid, rank); made.append(f"sensor{rank}")
            except IndexError:
                break                                    # 센서 수 도달
            except Exception:                            # noqa: BLE001
                REPORT["errors"] += 1; traceback.print_exc()
        return made
    return _make_wafer_report_from_archive(wid)


def _make_wafer_report_from_archive(wid: str) -> List[str]:
    date = _archive_date_of(wid)
    if not date or _load_archive(date, wid) is None:
        REPORT["errors"] += 1
        return []
    made = []
    for kind in ("waferaccum", "overlay", "heatmap"):
        dst = _plot_path(wid, kind)
        if dst.exists():
            made.append(kind); continue
        tmp = None
        try:
            tmp = render_archive_plot(date, wid, kind)
            shutil.copy(str(tmp), str(dst))
            made.append(kind)
        except Exception:                                # noqa: BLE001
            REPORT["errors"] += 1; traceback.print_exc()
        finally:
            if tmp is not None:                          # copy 실패해도 temp 디렉터리 정리(누수 방지)
                shutil.rmtree(tmp.parent, ignore_errors=True)
    sd = _sensor_dir(wid)
    for rank in range(max(0, REPORT_SENSOR_N)):
        try:
            tmp = render_archive_sensor(date, wid, rank)
        except IndexError:
            break
        except Exception:                                # noqa: BLE001
            REPORT["errors"] += 1; traceback.print_exc(); continue
        try:
            sd.mkdir(parents=True, exist_ok=True)
            dst = sd / (f"{rank:02d}_" + tmp.name)
            if not dst.exists():
                shutil.copy(str(tmp), str(dst))
            made.append(f"sensor{rank}")
        except Exception:                                # noqa: BLE001
            REPORT["errors"] += 1
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)  # copy 실패해도 temp 정리
    return made


def _report_worker_init(art_path):
    """리포트 워커 프로세스 초기화 — 부모와 같은 모델을 1회 로드(그림 메타용).
    ★ 워커는 반드시 CPU 로 로드(IO_FORCE_CPU) — GPU 켜진 상태에서 워커마다 DirectML/CUDA 세션을
    만들면 Windows 에서 크래시/과부하. 리포트 그림은 아카이브 npz 기반이라 GPU 불필요."""
    try:
        os.environ["IO_FORCE_CPU"] = "1"
        import matplotlib                                 # noqa: F401
        matplotlib.use("Agg")
        if art_path:
            ENGINE.selected = Path(art_path)
            ENGINE.art = Path(art_path)
        ENGINE.load()
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()


def _report_worker_render(wid):
    """워커에서 웨이퍼 1장 그림 렌더(아카이브 기반, 모델 불필요분은 npz 사용). 반환 (wid, 만든수, 실패)."""
    try:
        made = _make_wafer_report_from_archive(str(wid))
        return (wid, len(made), 0 if made else 1)
    except Exception:                                     # noqa: BLE001
        return (wid, 0, 1)


def _write_wafer_score_report(wid, selections=None, history=None, references=None):
    from src.app.score_report import build_score_report, write_score_report
    from src.app.app_sensor_scores import load_reference
    date = _archive_date_of(wid)
    arch = _load_archive(date, wid) if date else None
    row = (history or {}).get(wid) if history is not None else None
    if row is None:
        row = next((r for r in _history_for_current_model() if r.get("wafer_id") == wid), {})
    engine = _engine_for_wafer(wid, (arch or {}).get("meta") or row)
    with ENGINE.using(engine):
        _, threshold = _recipe_decision_by_name(row.get("recipe") or "")
    context = dict(model_id=_model_id(), model_directory=str(engine.art), model_threshold=threshold,
                   wafer_score_override=getattr(engine, "wafer_score_override", None),
                   score_norm=getattr(engine.model, "score_norm", "none"), display_rule=_display_rule(),
                   exclude_step_zero=getattr(getattr(engine.model,"cfg",None),"score_exclude_boundary_step0",True))
    reference = None
    if references is not None:
        key = str(engine.art)
        if key not in references:
            try:
                references[key] = load_reference(engine.art)
            except (ValueError, KeyError, OSError):
                references[key] = None
        reference = references[key]
    report = build_score_report(wid, arch, engine.art, row, context, selections, reference)
    report["created_at"] = _now()
    write_score_report(_report_root() / _short_name(wid), report)


@_pin_model
def _run_report(wafer_ids=None, score_selections=None, include_images=True):
    if wafer_ids:                                        # 요청한 웨이퍼(이력 포함)
        ids = [str(w) for w in wafer_ids]
    else:                                               # 전체(현재 검사분)
        ids = [r["wafer_id"] for r in STATE["results"]]
    REPORT.update(running=True, done=0, total=len(ids), started=_now(),
                  finished=None, errors=0, current=None)
    score_errors = 0
    history = {str(r.get("wafer_id")): r for r in _history_for_current_model()}
    references = {}
    for wid in ids:
        try:
            _write_wafer_score_report(wid, score_selections, history, references)
        except Exception:
            score_errors += 1
            traceback.print_exc()
    # predictions.csv 먼저(빠름)
    try:
        REPORT["csv"] = str(_write_predictions_csv(STATE["results"]))
    except Exception:                                    # noqa: BLE001
        traceback.print_exc()

    REPORT["images_enabled"] = include_images
    if not include_images:
        REPORT.update(running=False, done=len(ids), finished=_now(), current=None, errors=score_errors)
        return

    # ★ 그림 렌더(matplotlib, CPU 바운드)를 '여러 프로세스'로 병렬 → 코어 수만큼 단축.
    #   pyplot 은 스레드 안전하지 않아 프로세스로 분리(각자 독립 matplotlib). 실패하면 순차 폴백.
    try:
        _rw = int(os.environ.get("IO_REPORT_WORKERS") or 0)
    except (TypeError, ValueError):
        _rw = 0
    if _rw <= 0:
        _rw = _cpu_workers(4)
    done_parallel = False
    if _rw > 1 and len(ids) > 1:
        try:
            import concurrent.futures as _cf
            import multiprocessing as _mp
            ctx = _mp.get_context("spawn")
            art = str(ENGINE.art)
            with _cf.ProcessPoolExecutor(max_workers=_rw, mp_context=ctx,
                                         initializer=_report_worker_init,
                                         initargs=(art,)) as ex:
                futs = {ex.submit(_report_worker_render, w): w for w in ids}
                for fut in _cf.as_completed(futs):
                    try:
                        w, nmade, err = fut.result()
                    except Exception:                     # noqa: BLE001
                        w, nmade, err = futs[fut], 0, 1
                    REPORT["current"] = w
                    if err:
                        REPORT["errors"] += 1
                    REPORT["done"] += 1
            # ★ 전부 실패(=병렬 자체가 안 됨: spawn 불가 등)면 순차로 다시. 부분성공은 병렬 인정.
            done_parallel = (REPORT["errors"] < len(ids))
            if done_parallel:
                print(f"[report] 병렬 렌더 완료({_rw} 프로세스)", flush=True)
            else:
                print("[report] 병렬 전부 실패 → 순차 렌더로 폴백", flush=True)
        except Exception:                                 # noqa: BLE001 (병렬 실패 → 순차 폴백)
            traceback.print_exc()
            print("[report] 병렬 실패 → 순차 렌더로 폴백", flush=True)

    if not done_parallel:                                # 순차(폴백 또는 워커 1)
        REPORT["done"] = 0; REPORT["errors"] = 0
        for wid in ids:                                  # 웨이퍼별 그림(3종 + 센서 overlay)
            REPORT["current"] = wid
            _make_wafer_report(wid)
            REPORT["done"] += 1
    REPORT["errors"] += score_errors
    REPORT.update(running=False, finished=_now(), current=None)
    print(f"[report] 완료: {REPORT['done']}/{REPORT['total']} 웨이퍼 · "
          f"오류 {REPORT['errors']} · {_pred_csv()}")


class ReportScoreSelection(BaseModel):
    total_basis: str = "selected"
    model_id: str
    steps: List[int]
    sensors: Optional[List[str]] = None


class ReportReq(BaseModel):
    score_selections: Optional[Dict[str, ReportScoreSelection]] = None
    wafers: Optional[List[str]] = None                  # 지정 시 그 웨이퍼만, 없으면 전체


@app.post("/api/report")
def api_report(req: ReportReq = None):
    """리포트 생성(웨이퍼별 폴더에 그림 저장 + predictions.csv). 백그라운드.
    body {"wafers": [...]} → 그 웨이퍼만, 없으면 전체."""
    wafers = (req.wafers if req else None) or None
    selections = {}
    for name, selected in ((req.score_selections if req else None) or {}).items():
        if name not in ("overview", "detail") or selected.model_id != _model_id():
            raise HTTPException(409, "리포트 선택 조건의 모델이 다릅니다. 화면을 다시 조회하세요.")
        if selected.total_basis not in ("original", "selected") or not selected.steps or 0 in selected.steps or selected.sensors == []:
            raise HTTPException(400, "리포트에 저장할 스텝과 센서를 선택하세요.")
        selections[name] = dict(steps=selected.steps, sensors=selected.sensors, total_basis=selected.total_basis)
    if not STATE["results"] and not wafers:
        raise HTTPException(400, "검사 결과가 없습니다. 먼저 검사하세요.")
    include_images = True
    try:
        import matplotlib  # noqa: F401
    except Exception:
        include_images = False  # Score reports and CSV do not need plotting libraries.
    with _REPORT_LOCK:
        if REPORT["running"]:
            return {"already_running": True, **REPORT}
        threading.Thread(target=_run_report, kwargs={"wafer_ids": wafers, "score_selections": selections, "include_images": include_images},
                         name="report", daemon=True).start()
    total = len(wafers) if wafers else len(STATE["results"])
    return {"started": True, "total": total, "report_dir": str(REPORT_DIR)}


@app.get("/api/report")
def api_report_status():
    return {**REPORT, "report_dir": str(REPORT_DIR)}


@app.post("/api/report/wafer/{wid}")
def api_report_wafer(wid: str):
    """웨이퍼 1장만 리포트 생성(3종 + 상위 센서 overlay). 즉시 처리."""
    if wid not in STATE["recs"]:
        raise HTTPException(404, f"검사 결과에 없는 웨이퍼: {wid}")
    _write_wafer_score_report(wid)
    made = _make_wafer_report(wid)
    return {"wafer_id": wid, "made": made, "report_dir": str(REPORT_DIR)}


@app.get("/api/predictions.csv")
def api_predictions_csv():
    """predictions.csv 다운로드(없으면 현재 결과로 즉석 생성). 현재 모델 폴더 기준."""
    pred = _pred_csv()
    if not pred.exists():
        if not STATE["results"]:
            raise HTTPException(404, "predictions.csv 없음 — 먼저 검사하세요.")
        _write_predictions_csv(STATE["results"])
    return FileResponse(pred, media_type="text/csv", filename="predictions.csv")


@app.get("/api/history")
def api_history(days: int = 30):
    """일자별 이상점수 집계(추세 탭). ?days=N 으로 최근 N일."""
    return history_daily(days=days)


# ── 이력 검색(월 달력 → 일별 → 웨이퍼 → 아카이브 상세) ──
def _calendar_month(month: str) -> Dict:
    """month=YYYY-MM. 일별 검사수/불량수/평균 이상점수(현재 모델 이력)."""
    rows = _history_for_current_model()
    latest: Dict = {}
    for r in rows:
        d = str(r.get("date") or "")
        if not d.startswith(month):
            continue
        key = (d, r.get("wafer_id"))
        prev = latest.get(key)
        if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
            latest[key] = r
    byday: Dict[str, list] = {}
    for (d, _w), r in latest.items():
        byday.setdefault(d, []).append(r)
    days = []
    for d in sorted(byday):
        rr = [x for x in (_hist_ratio(v) for v in byday[d]) if np.isfinite(x)]
        # ★ 달력의 '불량'도 개요와 같은 잣대(심각 등급) — 예전엔 임계 초과 전체
        #   (verdict=NG)를 세서, 전부 '주의'인 날도 달력엔 '불량 31장'으로 떠
        #   날짜를 눌러 본 목록(주의 31)과 어긋났다.
        ng = sum(1 for v in byday[d] if str(v.get("band") or "") == "crit")
        days.append({"date": d, "count": len(byday[d]), "ng": ng,
                     "avg": round(float(np.mean(rr)), 3) if rr else None,
                     "max": round(float(np.max(rr)), 3) if rr else None})
    # 이력 전체에서 가장 최근 검사 날짜 — 이번 달이 비어있으면 프론트가 그 달로 자동 이동
    all_dates = sorted({str(r.get("date") or "") for r in rows if r.get("date")})
    return {"month": month, "days": days,
            "latest_date": (all_dates[-1] if all_dates else None)}


def _calendar_day(date: str) -> Dict:
    rows = _history_for_current_model()
    latest: Dict = {}
    for r in rows:
        if str(r.get("date")) != date:
            continue
        w = r.get("wafer_id")
        prev = latest.get(w)
        if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
            latest[w] = r
    out = []
    for w, r in latest.items():
        rat = _hist_ratio(r)
        # ★ 표기·등급은 산점도와 같은 것을 내려준다.
        #   · ratio 를 3자리 반올림으로 내려주면 화면에서 2자리 버림 규칙과 어긋나고,
        #     경계(0.9996 → 1.0)에서 등급까지 갈릴 수 있다 → 같은 '버림 2자리'로.
        #   · band(주의/경고/심각)는 서버가 이미 판정한 값을 그대로 전달한다.
        #     이력 검색이 자체 기준으로 색을 정하면 같은 웨이퍼가 탭마다 달라진다.
        out.append({"wafer_id": w,
                    "ratio": _floor2(rat) if np.isfinite(rat) else None,
                    "band": r.get("band"),
                    "verdict": r.get("verdict"), "chamber": r.get("chamber"),
                    "recipe": _recipe_display(r.get("recipe")), "ts": r.get("ts"),   # 표시명 통일(풀네임→29a)
                    "has_archive": _archive_path(date, w).exists()})
    out.sort(key=lambda x: _natkey(x["wafer_id"]))
    return {"date": date, "wafers": out}


@app.get("/api/calendar")
def api_calendar(month: str):
    return _calendar_month(month)


@app.get("/api/calendar/day/{date}")
def api_calendar_day(date: str):
    return _calendar_day(date)


@app.get("/api/history/range")
def api_history_range(start: str, end: str):
    """기간(start~end, 날짜 포함) 내 검사한 웨이퍼 목록(웨이퍼당 최신 1건).
    종료 시각 오름차순. 각 항목에 date(아카이브 날짜)와 has_archive 포함 — 이상분석 기간 조회용."""
    if end < start:
        start, end = end, start
    rows = _history_for_current_model()
    # 조회 기간 밖의 첫 장도 포함해 판단 — 필터로 남은 두 번째 장을 첫 장으로 표시하지 않는다.
    lot_first_ids = _first_wafer_ids(rows)
    latest: Dict = {}
    for r in rows:
        d = str(r.get("date") or "")
        if not d or d < start or d > end:            # YYYY-MM-DD 문자열 비교(정렬 가능)
            continue
        w = r.get("wafer_id")
        if not w:
            continue
        prev = latest.get(w)
        if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
            latest[w] = r
    out = []
    for w, r in latest.items():
        rat = _hist_ratio(r)
        d = str(r.get("date") or "")
        # ★ 이상 분석 탭이 (없어진) 웨이퍼 목록의 펼침 상세와 '같은 정보'를 보여줘야 하므로
        #   등급(band)·정규화 점수·깨진 행 수·종료시각까지 함께 준다.
        out.append({"wafer_id": w, "date": d,
                    # 점수 표기는 '버림 2자리'(규약). 여기만 반올림이면 같은 웨이퍼가
                    # 탭마다 0.01 다르게 보인다.
                    "ratio": _floor2(rat) if np.isfinite(rat) else None,
                    "ratio_thr": r.get("ratio_thr"),
                    "band": r.get("band"),
                    "verdict": r.get("verdict"),
                    "chamber": r.get("chamber") or _chamber(w),
                    "recipe": _recipe_display(r.get("recipe")), "ts": r.get("ts"),   # 표시명 통일(풀네임→29a)
                    "end_ts": r.get("end_ts") or r.get("ts"),
                    "lot": _lot_of(w, r),
                    "lot_first": str(w) in lot_first_ids,
                    "n_bad_lines": r.get("n_bad_lines") or 0,
                    "has_archive": _archive_path(d, w).exists()})
    def end_order(item):
        stamp = _parse_ts(item.get("end_ts")) or _parse_ts(item.get("ts"))
        return (stamp.timestamp() if stamp is not None else float("inf"),
                _natkey(item["wafer_id"]))

    out.sort(key=end_order)
    return {"start": start, "end": end, "wafers": out}


@app.get("/api/archive/{date}/{wid}")
def api_archive_meta(date: str, wid: str):
    arch = _load_archive(date, wid)
    if arch is None:
        raise HTTPException(404, "아카이브 데이터 없음")
    m = arch["meta"]
    return {k: m.get(k) for k in
            ("wafer_id", "recipe", "chamber", "score", "threshold",
             "score_raw", "threshold_raw", "verdict", "n_flag_sensor", "ts", "date") + AUDIT_FIELDS}


@app.get("/api/archive/{date}/{wid}/plot/{kind}")
def api_archive_plot(date: str, wid: str, kind: str):
    try:
        p = render_archive_plot(date, wid, kind)
    except KeyError:
        raise HTTPException(404, "아카이브 데이터 없음")
    except Exception as e:                               # noqa: BLE001
        raise HTTPException(400, str(e))
    from starlette.background import BackgroundTask      # 전송 후 임시파일 삭제(누적 방지)
    return FileResponse(p, media_type="image/png",
                        background=BackgroundTask(shutil.rmtree, p.parent, ignore_errors=True))


@app.get("/api/archive/{date}/{wid}/sensors")
@_wafer_model
def api_archive_sensors(date: str, wid: str, n: int = 12):
    arch = _load_archive(date, wid)
    if arch is None:
        raise HTTPException(404, "아카이브 데이터 없음")
    cols = arch["meta"].get("sensor_cols") or [str(c) for c in ENGINE.model.sensor_cols]
    idx = _top_sensor_indices(arch["z"], n)
    return {"sensors": [{"rank": r, "sensor": cols[int(j)] if int(j) < len(cols) else str(j)}
                        for r, j in enumerate(idx)]}


@app.get("/api/archive/{date}/{wid}/sensors-all")
@_wafer_model
def api_archive_sensors_all(date: str, wid: str):
    """모든 센서(원래 순서) + 센서별 이상점수(임계 anchored, 1.0=그 센서 OK상한). 센서 바 뷰용."""
    arch = _load_archive(date, wid)
    if arch is None:
        raise HTTPException(404, "아카이브 데이터 없음")
    cols = arch["meta"].get("sensor_cols") or [str(c) for c in ENGINE.model.sensor_cols]
    # 센서별 점수 = 누적이상치 ÷ 그 센서 OK 상한. 1.0 = 상한. 2.0 에서만 자른다.
    #   ratio 는 자르기 전 원값 — 2.0 으로 잘린 센서가 실제로 얼마나 큰지 화면이 보여 준다.
    with _PLOT_LOCK:
        sc, dt_ = RX.sensor_scores(ENGINE.model, _archive_rec(arch), arch["z"],
                                   ok_bands=_ok_bands(), return_detail=True)
    return {"sensors": [{"index": i, "sensor": cols[i] if i < len(cols) else str(i),
                         "score": sc[i],
                         # 점수만으로는 '오래 조금'과 '잠깐 크게'가 구분되지 않는다 →
                         # 순간 최대 이탈 배수와 초과 지속 비율을 함께 보낸다.
                         "ratio": (dt_[i].get("ratio") if i < len(dt_) else None),
                         "peak": (dt_[i].get("peak") if i < len(dt_) else None),
                         "dur": (dt_[i].get("dur") if i < len(dt_) else None)}
                        for i in range(len(sc))],
            "threshold": 1.0}


def _archive_app_sensor_scores(date, wid, steps=None, sensors=None):
    """Anomaly detail only: each sensor × all nonzero steps, train OK μ+3.5σ."""
    from src.app.app_sensor_scores import sensor_scores, load_reference, AppScoreUnavailable
    arch = _load_archive(date, wid)
    if arch is None:
        raise HTTPException(404, "아카이브 데이터 없음")
    try:
        level = arch.get("app_score_level")
        legacy = level is None
        if legacy:
            level = arch["z"]
        cols = arch["meta"].get("sensor_cols") or list(ENGINE.model.sensor_cols)
        reference = load_reference(ENGINE.art)
        result = sensor_scores(ENGINE.art, level, arch["step"], cols, reference=reference,
                               selected_steps=steps, selected_sensors=sensors)
        result = apply_selected_override(result, dict(arch["meta"], wafer_id=wid), getattr(ENGINE, "wafer_score_override", None))
        result["available_steps"] = reference[1].tolist()
        result["available_sensors"] = reference[2]
        result["legacy_precision"] = legacy
        return result
    except AppScoreUnavailable as exc:
        return {"available": False, "method": "train_ok_step_sensor_sigma",
                "sensors": [], "note": str(exc)}


class DetailScoreSelectionReq(BaseModel):
    steps: List[int]
    sensors: List[str]


@app.get("/api/archive/{date}/{wid}/sensors-app-score")
@_wafer_model
def api_archive_app_sensor_scores(date: str, wid: str, selection: Optional[str] = None):
    if selection is None:
        return _archive_app_sensor_scores(date, wid)
    try:
        payload = json.loads(selection)
        req = DetailScoreSelectionReq(**payload)
    except (ValueError, TypeError):
        raise HTTPException(400, "스텝·센서 선택 형식이 올바르지 않습니다.")
    if not req.steps or 0 in req.steps or not req.sensors:
        raise HTTPException(400, "스텝 0을 제외한 스텝과 센서를 선택하세요.")
    result = _archive_app_sensor_scores(date, wid, req.steps, req.sensors)
    result["applied_selection"] = dict(steps=req.steps, sensors=req.sensors)
    return result


@app.post("/api/archive/{date}/{wid}/sensors-app-score")
@_wafer_model
def api_archive_app_sensor_selection(date: str, wid: str, req: DetailScoreSelectionReq):
    if not req.steps or 0 in req.steps or not req.sensors:
        raise HTTPException(400, "스텝 0을 제외한 스텝과 센서를 선택하세요.")
    return _archive_app_sensor_scores(date, wid, req.steps, req.sensors)


@app.get("/api/archive/{date}/{wid}/sensor-idx/{j}")
@_wafer_model
def api_archive_sensor_idx(date: str, wid: str, j: int):
    """센서 원래-인덱스 j 의 개별 overlay(누적 없음)."""
    arch = _load_archive(date, wid)
    if arch is None:
        raise HTTPException(404, "아카이브 데이터 없음")
    A = ENGINE.model
    rec = _archive_rec(arch)
    actual, expected, z = arch["actual"], arch["expected"], arch["z"]
    if j < 0 or j >= z.shape[1]:
        raise HTTPException(400, "센서 인덱스 범위 초과")
    out = Path(tempfile.mkdtemp(prefix="archsi_")) / f"{int(j)}.png"
    with _PLOT_LOCK:
        try:
            RX._setup_font()
        except Exception:                                # noqa: BLE001
            pass
        RX.plot_recon_sensor_overlay(A, rec, int(j), actual, expected, z, out,
                                     ok_bands=_ok_bands())
    from starlette.background import BackgroundTask
    return FileResponse(out, media_type="image/png",
                        background=BackgroundTask(shutil.rmtree, out.parent, ignore_errors=True))


@app.get("/api/archive/{date}/{wid}/sensor-series/{j}")
@_wafer_model
def api_archive_sensor_series(date: str, wid: str, j: int):
    """센서 j 인터랙티브 차트용 시계열(실제/기대/밴드/차이/누적). 이미지 아님."""
    arch = _load_archive(date, wid)
    if arch is None:
        raise HTTPException(404, "아카이브 데이터 없음")
    z = arch["z"]
    if j < 0 or j >= z.shape[1]:
        raise HTTPException(400, "센서 인덱스 범위 초과")
    # ★ 배점 모드: 해당 센서의 복원오차 z[:,j] 에 스텝별 가중(W[:,j])을 곱해 표시.
    if _score_mode() == "weighted":
        W = _archive_weight_matrix(arch, [str(c) for c in ENGINE.model.sensor_cols])
        if W is not None and W.shape == z.shape:
            z = z * W
    with _PLOT_LOCK:
        data = RX.sensor_series(ENGINE.model, _archive_rec(arch),
                                int(j), arch["actual"], arch["expected"], z)
    if data is None:
        raise HTTPException(400, "표시할 데이터 없음")
    return data


@app.get("/api/archive/{date}/{wid}/wafer-series")
@_wafer_model
def api_archive_wafer_series(date: str, wid: str):
    """웨이퍼 복원오차 누적(종합) 인터랙티브 차트용 시계열(누적/시점별 초과오차)."""
    arch = _load_archive(date, wid)
    if arch is None:
        raise HTTPException(404, "아카이브 데이터 없음")
    rec = _archive_rec(arch)
    with _PLOT_LOCK:
        # ★ 배점 모드: 복원오차 z 에 센서·스텝 가중(W)을 곱해 그린다 → 곡선 최종값=S_w,
        #   판정선=θ_w(정규화 1). 배점 임계·밸브상태가 있어야 하며, 없으면 raw 로 폴백.
        zg = arch["z"]; thr = _waferaccum_threshold(wid, rec); okb = _ok_bands()
        _wmode = False
        if _score_mode() == "weighted" and not rec.get("score_overridden"):
            W = _archive_weight_matrix(arch, [str(c) for c in ENGINE.model.sensor_cols])
            thr_w = _score_thr_for(rec.get("recipe"))
            if W is not None and W.shape == arch["z"].shape and thr_w and thr_w > 1e-12:
                zg = arch["z"] * W
                thr = _weighted_plot_threshold(rec, thr_w)
                _wmode = True                             # ★ OK밴드는 유지하되 '참고(raw 기준)'로 표시
        data = RX.wafer_series(ENGINE.model, rec, zg,
                               ok_bands=okb, threshold=thr,
                               final_score=arch["meta"].get("effective_score_raw") if arch["meta"].get("score_overridden") else None,
                               raw_rec=rec, raw_z=zg)
    if data is None:
        raise HTTPException(400, "표시할 데이터 없음")
    # 정상(OK) 참조가 왜 없는지/어떤 기준인지 화면에 안내
    if okb is None:
        data["ok_note"] = ("정상(OK) 참조 데이터 없음 — 모델 폴더에 ok_accum_bands.npz 가 없어 "
                           "정상범위·OK평균 곡선을 표시할 수 없습니다.")
    elif _wmode:
        data["ok_note"] = "참고: 정상범위/OK평균은 raw 기준(배점 미적용) — 배점 곡선과 직접 비교는 참고용입니다."
    return data


@app.get("/api/archive/{date}/{wid}/zone-contrib")
@_wafer_model
def api_zone_contrib(date: str, wid: str, a: float = 0.0, b: float = 1.0, top: int = 5):
    """종합 overlay 이상구간(a~b, 0~1 비율)의 '센서별 기여도' — 그 구간 복원오차 누적의 센서 분해.

    종합 누적 = Σ(센서별 누적)이므로, 구간 내 센서별 증가분/전체 증가분 = 기여율(%).
    '이 구간이 왜 이상구간인가'를 원인 센서 순위로 보여준다."""
    arch = _load_archive(date, wid)
    if arch is None:
        raise HTTPException(404, "아카이브 데이터 없음")
    rec = _archive_rec(arch)
    z = np.asarray(arch["z"], dtype=np.float64)
    keep = RX._plot_time_filter(ENGINE.model, rec, z.shape[0])[0]
    zc = z[keep]                                          # 차트와 동일 축(ProcessStep0 제외)
    T = zc.shape[0]
    if T < 2:
        return {"sensors": [], "total": 0.0}
    i0 = max(0, min(T - 1, int(round(float(a) * (T - 1)))))
    i1 = max(0, min(T - 1, int(round(float(b) * (T - 1)))))
    if i1 < i0:
        i0, i1 = i1, i0

    def _sum(lo, hi):
        seg = np.clip(zc[lo:hi + 1], 0, None)             # 누적곡선과 동일 기저(threshold_base=0)
        p = seg.sum(axis=0)
        return p, float(p.sum())

    per, tot = _sum(i0, i1)
    # ★ 종합 차트는 stride(n//2000)로 다운샘플되므로, 좁은 이상구간이 원해상도 단일 포인트로
    #   매핑되면 그 지점이 마침 증가분 0 이라 어떤 센서도 기여로 안 잡히던 문제(개별 센서 차트에
    #   구간이 안 뜸). tot≈0 이면 원인이 있는 인접 구간까지 좌우로 넓혀 반드시 귀속시킨다.
    if tot <= 1e-12:
        step = max(2, int(round(T * 0.02)))               # 한 번에 좌우로 넓히는 폭
        lo, hi = i0, i1
        while tot <= 1e-12 and (lo > 0 or hi < T - 1):
            lo = max(0, lo - step)
            hi = min(T - 1, hi + step)
            per, tot = _sum(lo, hi)
    cols = arch["meta"].get("sensor_cols") or [str(c) for c in ENGINE.model.sensor_cols]
    order = np.argsort(per)[::-1][:max(1, int(top))]
    out = [{"index": int(j), "sensor": (cols[j] if j < len(cols) else str(j)),
            "share": round(100.0 * float(per[j]) / tot, 1) if tot > 1e-12 else 0.0}
           for j in order]
    return {"sensors": out, "total": round(tot, 3)}


class ZoneBatchReq(BaseModel):
    windows: list = []                                   # [[a,b], ...] (0~1 비율)
    top: int = 999


@app.post("/api/archive/{date}/{wid}/zone-contrib-batch")
@_wafer_model
def api_zone_contrib_batch(date: str, wid: str, req: ZoneBatchReq):
    """여러 이상구간의 센서별 기여도를 한 번에 계산(아카이브 1회 로드). 개별 센서 차트 전 구간 색칠용.

    ★ 예전엔 프런트가 구간을 40개까지만 계산해서, 41번째 이후 구간이 어떤 센서에도 안 뜨던
      문제가 있었다. 이 배치로 '모든 구간'을 한 요청에 처리해 그 누락을 제거한다."""
    arch = _load_archive(date, wid)
    if arch is None:
        raise HTTPException(404, "아카이브 데이터 없음")
    rec = _archive_rec(arch)
    z = np.asarray(arch["z"], dtype=np.float64)
    keep = RX._plot_time_filter(ENGINE.model, rec, z.shape[0])[0]
    zc = z[keep]
    T = zc.shape[0]
    cols = arch["meta"].get("sensor_cols") or [str(c) for c in ENGINE.model.sensor_cols]
    topn = max(1, int(req.top or 999))
    if T < 2:
        return {"zones": [{"sensors": [], "total": 0.0} for _ in (req.windows or [])]}

    def _sum(lo, hi):
        p = np.clip(zc[lo:hi + 1], 0, None).sum(axis=0)
        return p, float(p.sum())

    step = max(2, int(round(T * 0.02)))
    zones = []
    for w in (req.windows or []):
        try:
            a, b = float(w[0]), float(w[1])
        except (TypeError, ValueError, IndexError):
            zones.append({"sensors": [], "total": 0.0}); continue
        i0 = max(0, min(T - 1, int(round(a * (T - 1)))))
        i1 = max(0, min(T - 1, int(round(b * (T - 1)))))
        if i1 < i0:
            i0, i1 = i1, i0
        per, tot = _sum(i0, i1)
        if tot <= 1e-12:                                  # 인접 구간까지 넓혀 반드시 귀속(단일센서 최소 1)
            lo, hi = i0, i1
            while tot <= 1e-12 and (lo > 0 or hi < T - 1):
                lo = max(0, lo - step); hi = min(T - 1, hi + step)
                per, tot = _sum(lo, hi)
        order = np.argsort(per)[::-1][:topn]
        zones.append({"sensors": [
            {"index": int(j), "sensor": (cols[j] if j < len(cols) else str(j)),
             "share": round(100.0 * float(per[j]) / tot, 1) if tot > 1e-12 else 0.0}
            for j in order], "total": round(tot, 3)})
    return {"zones": zones}


class RetentionReq(BaseModel):
    days: int = RETENTION_MAX_DAYS
    unlimited: bool = False


@app.get("/api/archive-retention")
def api_get_retention():
    return {"days": _archive_retention_days()}


@app.post("/api/archive-retention")
def api_set_retention(req: RetentionReq):
    """0000 입력은 unlimited=True로 전달. 일반 숫자 0은 기본 60일로 저장."""
    try:
        days = int(req.days)
    except (TypeError, ValueError):
        days = RETENTION_MAX_DAYS
    if req.unlimited:
        days = 0
    elif days <= 0 or days > RETENTION_MAX_DAYS:
        days = RETENTION_MAX_DAYS
    cfg = _load_app_config()
    cfg["archive_retention_days"] = days
    _save_app_config(cfg)
    pruned = _prune_archive()
    return {"days": days, "unlimited": days == 0, "pruned": pruned}


def _reset_recognition_state(wipe_tags: bool = True) -> Dict:
    """'처음 보는 데이터'로 되돌리는 공통 초기화.

    ★ 왜 필요한가 — 이력/아카이브만 지우면 겉보기는 비는데, 아래 인식 상태가
      남아 같은 파일을 '이미 처리됨'으로 건너뛰거나 옛 점수를 재사용했다.
      "초기화했는데 새 데이터로 인식을 못 한다"(사용자 보고)의 원인.
      · 자동검사 처리목록(AUTOPILOT.processed)
      · 점수 캐시(점수 재사용) · 파싱 캐시 · 집계 캐시
      · 화면 상태(STATE 결과)
      · (wipe_tags) 폴더 metadata(앱 폴더 저장분)·파일 태그·레시피 태그
    """
    out = {}
    try:
        if AUTOPILOT is not None:
            with AUTOPILOT._plock:
                out["processed"] = len(AUTOPILOT.processed)
                AUTOPILOT.processed = set()
            AUTOPILOT._load_fail = {}
            AUTOPILOT._save_state()
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()
    for name, d in (("score_cache", SCORE_CACHE_DIR),
                    ("parse_cache", DATA_DIR / "parse_cache"),
                    *((("metadata", DATA_DIR / "metadata_fallback"),) if wipe_tags else ())):
        try:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                out[name] = True
        except Exception:                                 # noqa: BLE001
            pass
    with _SCORE_CACHE_LOCK:
        _SCORE_CACHE_MEM["sig"] = None; _SCORE_CACHE_MEM["rows"] = {}
    if wipe_tags:
        try:
            global _FILE_META, _RECIPE_TAGS
            with _FILE_META_LOCK:
                _FILE_META = {}
            for f in (_file_meta_path(), _recipe_tags_file()):
                try:
                    f.unlink()
                except OSError:
                    pass
            _RECIPE_TAGS = {}
            out["tags"] = True
        except Exception:                                 # noqa: BLE001
            traceback.print_exc()
    try:
        _AGG_CACHE.clear()                                # 집계(개요·산포) 캐시
        _SCATTER_CACHE["sig"] = None; _SCATTER_CACHE["map"] = {}
    except Exception:                                     # noqa: BLE001
        pass
    with STATE_LOCK:
        STATE["results"] = []; STATE["recs"] = {}
        STATE["coverage"] = None; STATE["inspected_at"] = None
    return out


@app.post("/api/archive/clear")
def api_archive_clear():
    """검사 이력(아카이브) 전체 삭제 — 달력/검색 데이터가 사라진다. 되돌릴 수 없음.

    ★ 인식 상태(자동검사 처리목록·점수/파싱/집계 캐시·폴더 metadata·태그)까지
      함께 초기화한다 — 같은 파일이 다시 들어오면 '새 데이터'로 재검사된다."""
    removed = 0
    try:
        if ARCHIVE_DIR.exists():
            for d in list(ARCHIVE_DIR.iterdir()):
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
                else:
                    try:
                        d.unlink()
                    except OSError:
                        pass
    except Exception as e:                               # noqa: BLE001
        raise HTTPException(500, f"삭제 실패: {e}")
    # 구버전 이력 파일 잔재도 정리(단일 소스 = 아카이브)
    try:
        for b in [DATA_DIR / "history.jsonl", *DATA_DIR.glob("history_*.jsonl")]:
            try:
                b.unlink()
            except OSError:
                pass
    except Exception:                                     # noqa: BLE001
        pass
    _hist_index_reset()
    reset = _reset_recognition_state(wipe_tags=True)
    return {"ok": True, "removed_days": removed, "reset": reset}


@app.get("/api/archive/{date}/{wid}/sensor/{rank}")
def api_archive_sensor(date: str, wid: str, rank: int):
    try:
        p = render_archive_sensor(date, wid, rank)
    except KeyError:
        raise HTTPException(404, "아카이브 데이터 없음")
    except IndexError:
        raise HTTPException(404, "센서 rank 초과")
    except Exception as e:                               # noqa: BLE001
        raise HTTPException(400, str(e))
    from starlette.background import BackgroundTask
    return FileResponse(p, media_type="image/png",
                        background=BackgroundTask(shutil.rmtree, p.parent, ignore_errors=True))


def _known_recipes() -> List[str]:
    """모델/설정이 아는 레시피 이름 목록(임계 있는 레시피 + recipe_names)."""
    if ENGINE.root.chamber_engines and ENGINE._scope.get() is None:
        all_known = set()
        for engine in ENGINE.root.chamber_engines.values():
            with ENGINE.using(engine):
                all_known.update(_known_recipes())
        return sorted(all_known)
    known = []
    for k in (ENGINE.thresholds or {}):
        if str(k) not in known:
            known.append(str(k))
    try:
        for r in getattr(ENGINE.cfg, "recipe_names", ()) or ():
            if str(r) not in known:
                known.append(str(r))
    except Exception:                                    # noqa: BLE001
        pass
    return known


class RecipeMultReq(BaseModel):
    multipliers: Dict[str, float] = {}
    default: Optional[float] = None


def _ort_install_diag() -> dict:
    """CUDA 가 안 붙는 진짜 이유를 찾아 준다.

    ★ 가장 흔한 원인: venv 에 onnxruntime(CPU 판)과 onnxruntime-gpu 가 **둘 다** 깔린 것.
      두 패키지는 같은 모듈명을 쓰기 때문에 나중에 잡히는 쪽이 이기고, CPU 판이 이기면
      CUDAExecutionProvider 자체가 목록에 안 나온다(설정에서 CUDA 를 골라도 CPU 로 동작).
      → 해결: pip uninstall -y onnxruntime  (onnxruntime-gpu 만 남긴다)
    """
    out = {"packages": [], "problem": "", "fix": ""}
    try:
        from importlib import metadata as _md
        for name in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml",
                     "onnxruntime-silicon"):
            try:
                out["packages"].append({"name": name, "version": _md.version(name)})
            except Exception:                             # noqa: BLE001 — 미설치
                pass
    except Exception:                                     # noqa: BLE001
        pass
    names = {p["name"] for p in out["packages"]}
    avail = []
    try:
        import onnxruntime as _ort
        avail = list(_ort.get_available_providers())
        out["onnxruntime_file"] = getattr(_ort, "__file__", "")
    except Exception as e:                                # noqa: BLE001
        out["problem"] = f"onnxruntime 를 불러올 수 없습니다: {e}"
        return out
    if "onnxruntime" in names and len(names) > 1:
        out["problem"] = ("CPU 판(onnxruntime)과 GPU 판이 함께 설치돼 있습니다. "
                          "CPU 판이 우선 잡히면 CUDA 를 골라도 CPU 로 동작합니다.")
        out["fix"] = "pip uninstall -y onnxruntime   (GPU 판만 남기고 앱 재시작)"
    elif "CUDAExecutionProvider" not in avail and os.name != "nt":
        if "onnxruntime-gpu" not in names:
            out["problem"] = "GPU 판(onnxruntime-gpu)이 설치돼 있지 않습니다."
            out["fix"] = "pip install \"onnxruntime-gpu>=1.17,<1.23\"  (앱 재시작)"
        else:
            out["problem"] = ("onnxruntime-gpu 는 설치돼 있으나 CUDA 프로바이더가 목록에 없습니다 "
                              "— CUDA/cuDNN 라이브러리를 찾지 못한 경우입니다.")
            out["fix"] = ("torch 가 설치돼 있으면 앱이 torch 번들 CUDA 라이브러리를 자동 선로드합니다. "
                          "그래도 안 되면 onnxruntime-gpu 버전을 CUDA 버전에 맞추세요.")
    # 학습(torch)은 되는데 추론만 CPU 인지 구분 — 사용자 증상과 바로 연결되는 정보
    try:
        import torch as _t
        out["torch_cuda"] = bool(_t.cuda.is_available())
        out["torch_version"] = _t.__version__
    except Exception:                                     # noqa: BLE001
        out["torch_cuda"] = None
    return out


class CpuLimitReq(BaseModel):
    # ★ 통일 입력(사용자 결정): '사용할 코어 수' 하나. 0=제한 없음(전체 사용).
    #   제한을 걸면 affinity(코어 고정)까지 자동 적용 — '딱 그 코어 수만' 보장.
    use_cores: Optional[int] = None
    # 구버전 호환 필드(use_cores 가 오면 무시)
    reserve: int = 0                                      # 비워 둘 물리 코어 수
    max_cores: int = 0                                    # 사용할 최대 물리 코어 수(0=제한없음)
    affinity: bool = False                                # 코어 고정까지 적용할지


@app.get("/api/cpu-limit")
def api_cpu_limit_get():
    """CPU 사용 제한 현황(물리/논리 코어, 예약 수, 실제 적용 상태)."""
    cfg = _load_app_config()
    st = CPU.state()
    st.pop("masks", None)                                 # 마스크는 UI 에 불필요
    # ★ 검증용: OS 에 '지금 실제로' 묶여 있는 논리 프로세서 수를 되읽는다.
    #   설정값이 아니라 현재 프로세스의 사실 — 이 숫자가 곧 '제한이 걸렸는가'의 증거.
    now_aff = None
    try:
        now_aff = CPU.current_affinity_logical()
    except Exception:                                     # noqa: BLE001
        now_aff = None
    return {**st, "now_affinity_logical": now_aff,
            "saved": {"reserve": int(cfg.get("cpu_reserve") or 0),
                      "max_cores": int(cfg.get("cpu_max") or 0),
                      "affinity": bool(cfg.get("cpu_affinity", False))},
            "affinity_note": STATE.get("cpu_affinity_note", ""),
            "pool_workers": _cpu_workers(8),              # 실제 스레드풀 크기
            "restart_required": True}


@app.post("/api/cpu-limit")
def api_cpu_limit_set(req: CpuLimitReq):
    """CPU 사용 제한 저장. 스레드 수는 재시작 후 적용(라이브러리가 import 시점에 읽음)."""
    topo = CPU.topology()
    phys = int(topo["physical"])
    if req.use_cores is not None:
        # ★ 통일 방식: 사용할 코어 수 N(0=전체). N<물리코어면 코어 고정까지 자동으로 걸어
        #   '딱 N 코어만' 쓰게 한다(스레드 수 제한만으론 다른 코어로 새는 걸 못 막는다).
        mx = max(0, min(phys, int(req.use_cores)))
        reserve = 0
        aff = bool(mx and mx < phys)
    else:                                                 # 구버전 호환(예전 3필드)
        reserve = max(0, min(phys - 1, int(req.reserve or 0)))   # 최소 1코어는 남겨야 앱이 돈다
        mx = max(0, min(phys, int(req.max_cores or 0)))
        aff = bool(req.affinity)
    cfg = _load_app_config()
    cfg["cpu_reserve"] = reserve
    cfg["cpu_max"] = mx
    cfg["cpu_affinity"] = aff
    _save_app_config(cfg)
    b = CPU.budget(cfg)
    return {"ok": True, "saved": {"reserve": reserve, "max_cores": mx, "affinity": aff},
            "will_use_cores": b["use_cores"], "will_use_threads": b["threads"],
            "physical": phys, "restart_required": True}


# ════════════════════════════════════════════════════════════════════
#  표시 판정 규칙 API — 임계(레시피별)·모드·밴드 + n_tp 백필 + 자동 산출
# ════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════
#  산포(분산) 추세 — 임계로는 못 잡는 '서서히 나빠짐'을 산포가 벌어지는 것으로 조기 감지
#   · ★ 공정 그룹 = LOT (원본 .log 의 I2 해시. 같은 해시 = 같은 LOT — 사용자 확정)
#     예전엔 끝시각 간격(기본 10분)으로 잘랐는데, 그건 LOT 경계의 '추정'일 뿐이라
#     설비가 쉬거나 몰아서 돌면 한 LOT 이 쪼개지거나 두 LOT 이 붙었다.
#     이제 LOT 해시로 정확히 묶는다(gap 방식은 LOT 이 없는 옛 이력용 폴백으로만 남긴다).
#   · 레시피 기저 제거 = 레시피별 중앙값/MAD 로 표준화한 u 로 산포 계산(전체 구간 기준 — 사용자 확정)
#   · 지표 = IQR · max-min · p90-p50 (옵션)
#   · 산포비 D = 그룹 산포 / 전체 기준 산포 → 1이면 평소, 2면 두 배로 벌어짐
# ════════════════════════════════════════════════════════════════════
def _disp_metric(vals: np.ndarray, metric: str) -> float:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    if metric == "range":
        return float(a.max() - a.min())
    if metric == "p90p50":
        return float(np.percentile(a, 90) - np.percentile(a, 50))
    return float(np.percentile(a, 75) - np.percentile(a, 25))     # 기본 IQR


def _group_margin_sigma(vals) -> float:
    """임계까지 몇 σ 남았는지 — 로그공간(임계=1 → log 0) 기준.

    여유 = -median(log r) / σ_log   (양수=임계 아래=안전, 0=중앙값이 임계에 닿음)
    · 점수는 이미 '임계=1'로 정규화돼 있어 레시피 차이가 자동으로 사라진다.
    · 로그를 쓰는 이유는 복원오차가 곱셈적으로 퍼지기 때문(밴드 자동계산과 같은 척도).
    · 산포(분모)와 수준(분자)을 한 값에 담아, 흩어지거나 올라가면 함께 줄어든다.
    """
    a = np.asarray([v for v in vals if np.isfinite(v) and v > 0], dtype=np.float64)
    if a.size < 2:
        return float("nan")
    l = np.log(a)
    med = float(np.median(l))
    sig = float(np.median(np.abs(l - med))) * 1.4826
    if not np.isfinite(sig) or sig <= 1e-9:
        sig = float(np.std(l, ddof=1)) if a.size > 1 else 0.0
    if not np.isfinite(sig) or sig <= 1e-9:               # 산포가 0이면 거리만으로 판단
        return 99.0 if med < 0 else 0.0
    return float(min(99.0, max(-99.0, -med / sig)))


def _group_excess(vals, power: float = 1.0, base: float = 1.0) -> tuple:
    """초과 강도 — '기준선을 넘은 빈도 × 넘은 정도'를 한 값으로.

        초과 강도 = 평균( max(0, r - base) )   (그룹 전체 웨이퍼 기준)
                  = 초과율 × 평균 초과량

    base = 어디부터 '초과'로 볼 것인가.
      · 1.0  = 임계 그 자체(예전 기본). 임계를 살짝 넘긴 웨이퍼도 전부 잡혀서,
               칸이 온통 빨갛게 되어 어디가 진짜 문제인지 구분이 안 된다.
      · 심각 배수(현재 기본) = '위험' 등급을 넘은 것만 센다. 주의·경고는 산포에
               반영하지 않는다 — 산포 화면은 '심하게 벌어진 구간'을 찾는 화면이다.

    · 넘은 장수가 늘어도, 넘은 폭이 커져도 값이 커진다 → "점점 많아지고 커진다"를 그대로 표현.
    · 통계에서는 상방 부분적률(upper partial moment), 품질공학의 손실함수와 같은 계열.
    · power=2 면 크게 넘은 웨이퍼에 더 큰 벌점(악화 가속을 강조).
    · 한계: 아무도 안 넘으면 0 이라 '넘기 직전'은 못 잡는다 → 산포·여유가 그 구간을 담당.
    반환: (초과 강도, 초과율, 평균 초과량)
    """
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return float("nan"), 0.0, 0.0
    ex = np.maximum(0.0, a - float(base))
    over = ex > 0
    inten = float(np.mean(ex ** power))
    rate = float(over.mean())
    mean_ex = float(ex[over].mean()) if over.any() else 0.0
    return inten, rate, mean_ex


def _lot_runs(items: List, gap_min: float, group_by: str = "lot") -> List:
    """챔버 하나의 웨이퍼를 '한 번 돌린 묶음' 단위로 나눈다 → [(lot|None, [item,...]), ...]

    items 는 시간 오름차순, 각 원소의 [0] 이 끝시각(datetime)·[-1] 이 LOT(또는 None).

    ★ 산포 계산(_dispersion_groups)과 첫 장 표시(_first_wafer_ids)가 반드시 같은
      묶음을 봐야 한다. 예전에는 두 곳이 따로 묶어서, 산포는 구간마다 첫 장을 빼는데
      화면은 LOT 당 한 장만 흐려지는 어긋남이 생겼다. 여기서 한 번만 정한다.

    두 가지 규칙
      1) 같은 LOT 해시라도 끝시각 간격이 gap_min 을 넘으면 끊는다. 통째로 합치면
         묶음이 첫 구간에만 놓여, 그 LOT 이 다시 돈 날짜가 통째로 비어 버린다.
      2) LOT 을 모르는 웨이퍼가 LOT 을 아는 묶음에 시간상 붙어 있으면(간격 이내)
         그 묶음에 넣는다. 끊어 두면 진짜 첫 장에 태그가 없을 때 두 번째 장이
         첫 장으로 밀려 표시된다 — 실제로 그렇게 보였다.
    """
    if not items:
        return []
    seq = sorted(items, key=lambda x: x[0])
    if group_by != "lot":
        runs, cur = [], [seq[0]]
        for it in seq[1:]:
            if (it[0] - cur[-1][0]).total_seconds() > gap_min * 60.0:
                runs.append((None, cur)); cur = []
            cur.append(it)
        return runs + [(None, cur)]
    # 시간이 이어지는 덩어리로 먼저 자르고, 그 안에서 LOT 이 바뀌면 다시 자른다.
    runs: List = []
    cur: List = [seq[0]]
    cur_lot = seq[0][-1]
    for it in seq[1:]:
        gap = (it[0] - cur[-1][0]).total_seconds() > gap_min * 60.0
        lot = it[-1]
        # LOT 미상은 이어지는 동안 현재 묶음에 흡수한다(규칙 2).
        same = (lot is None) or (cur_lot is None) or (lot == cur_lot)
        if gap or not same:
            runs.append((cur_lot, cur)); cur = []; cur_lot = lot
        elif cur_lot is None and lot is not None:
            cur_lot = lot                                 # 미상으로 시작한 묶음의 LOT 확정
        cur.append(it)
    runs.append((cur_lot, cur))
    return runs


def _first_wafer_ids(rows: List[Dict], gap_min: float = 10.0,
                     group_by: str = "lot") -> set:
    """산포 계산에서 '첫 장'으로 빠지는 웨이퍼 id 집합.

    ★ 산점도의 흐리게 표시와 산포의 제외가 어긋나던 문제 때문에 만든다.
      예전에는 산점도가 '자기 결과 안에서 LOT 이 있는 것만' 첫 장을 골랐다. 그래서
        · LOT 미상 웨이퍼 — 산포는 시간 간격으로 묶어 첫 장을 빼는데 화면은 안 흐려졌다
        · 구간을 좁혀 본 산점도 — 그 창 안의 첫 장을 골라, LOT 의 진짜 첫 장과 달랐다
      여기서 산포(_dispersion_groups)와 '같은 규칙·같은 전체 이력'으로 한 번만 정한다.
    """
    parsed = []
    # ★ 점수 유효성은 따지지 않는다 — '첫 장'은 시간 순서의 문제이지 점수와 무관하다.
    #   (판정 보류 웨이퍼가 진짜 첫 장인데 건너뛰면 두 번째가 첫 장으로 찍힌다)
    for r in rows:
        ts = _parse_ts(r.get("ts"))                       # ts = 공정 종료시각
        wid = r.get("wafer_id")
        if ts is None or not wid:
            continue
        ch = _chamber_key(r) or "-"
        lot = _lot_of(wid, r) if group_by == "lot" else None
        parsed.append((ts, ch, lot, str(wid)))
    per_ch: Dict[str, List] = {}
    for it in parsed:
        per_ch.setdefault(it[1], []).append(it)
    first: set = set()
    for ch, items in per_ch.items():
        # ★ 산포와 똑같은 묶음을 쓴다(_lot_runs 하나로 통일). 따로 묶으면
        #   '계산에서 뺀 장'과 '흐리게 표시한 장'이 어긋난다.
        for lot, run in _lot_runs([(it[0], it[1], it[3], it[2]) for it in items],
                                  gap_min, group_by):
            # LOT 을 모르는 묶음은 표시하지 않는다 — LOT 경계가 아닌 자리에
            # '첫 장'이 찍혀 "점 중간에 첫 장" 이 되기 때문.
            if group_by == "lot" and not lot:
                continue
            first.add(run[0][2])                          # 묶음의 첫 장(종료시각 기준)
    return first


def _dispersion_groups(rows: List[Dict], gap_min: float, metric: str,
                       min_n: int, basis: str = "spread", group_by: str = "lot",
                       include_first: bool = True, excess_base: float = 1.0) -> Dict:
    """챔버별 공정 그룹 산포 시계열 + 전체 기준 산포(baseline) 반환.

    group_by="lot"  : 같은 LOT(.log I2 해시)끼리 묶는다. LOT 을 모르는 웨이퍼는
                      끝시각 간격(gap_min)으로 묶어 폴백한다(LOT 태깅 전 옛 이력 대비).
    group_by="gap"  : 예전 방식 — 끝시각 간격만으로 자른다.

    include_first=False : LOT 의 '첫 장'을 산포 계산에서 뺀다(사용자 옵션).
        ★ 왜 필요한가 — LOT 첫 장은 설비가 쉬다가 다시 도는 직후라 이전 공정 잔류
          (carryover)·예열 상태가 남아 점수가 높게 나오는 경향이 있다. 그것이 '공정
          이상'이 아니라 '시작 조건'이면 산포를 부풀린다. 빼고 보면 LOT 내부의
          진짜 흔들림이 드러난다. 대신 첫 장 자체를 숨기지는 않는다 — 화면에서
          같은 색으로 흐리게 표시해 '있었지만 계산에서 뺐다'를 남긴다.
    """
    # ── 레시피별 기저 제거(전체 구간 기준) ──
    by_rc: Dict[str, List[float]] = {}
    parsed = []
    first_ids = _first_wafer_ids(rows, gap_min, group_by)
    n_no_ts = n_no_score = 0                              # 진단: 왜 계산에서 빠졌나
    for r in rows:
        ts = _parse_ts(r.get("ts"))
        sc = r.get("ratio_thr") if _isnum(r.get("ratio_thr")) else r.get("score")
        if ts is None:
            n_no_ts += 1
            continue
        if not _isnum(sc):
            n_no_score += 1                               # 점수가 없어 산포에서 제외된 웨이퍼
            continue
        rc = _recipe_display(r.get("recipe") or "")
        ch = _chamber_key(r) or "-"
        lot = _lot_of(r.get("wafer_id"), r) if group_by == "lot" else None
        parsed.append((ts, ch, rc, float(sc), bool(r.get("lot_first", str(r.get("wafer_id")) in first_ids)), lot))
        by_rc.setdefault(rc, []).append(float(sc))
    base_rc: Dict[str, tuple] = {}
    for rc, vals in by_rc.items():
        a = np.asarray(vals, dtype=np.float64)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med))) * 1.4826
        if not (mad > 1e-9):
            sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
            mad = sd if sd > 1e-9 else 1.0
        base_rc[rc] = (med, mad)
    # ── 챔버별 그룹핑(LOT 해시 · LOT 없으면 끝시각 간격 폴백) ──
    per_ch: Dict[str, List] = {}
    n_lot = 0
    for ts, ch, rc, sc, first, lot in parsed:
        med, mad = base_rc.get(rc, (0.0, 1.0))
        if lot:
            n_lot += 1
        # u=레시피 표준화값(산포용) · sc=임계 대비 원비율(여유/Cpu 용) · lot=묶음 키
        per_ch.setdefault(ch, []).append((ts, (sc - med) / mad, sc, first, lot))
    groups: Dict[str, List[Dict]] = {}
    n_first_excluded = 0                                  # 첫 장 제외로 빠진 웨이퍼 수(표시용)
    allvals: List[float] = []
    gaps: List[float] = []                                # 챔버 내 연속 웨이퍼 간격(분) — 진단용
    n_small = 0                                           # 최소 장수 미달로 버린 그룹 수
    for ch, items in per_ch.items():
        items.sort(key=lambda x: x[0])
        cur: List = []
        out: List[Dict] = []

        def _close(cur, lot=None):
            nonlocal n_small, n_first_excluded
            # ★ 첫 장 제외는 'LOT 을 아는 묶음'에만 적용한다(화면 표시와 같은 규칙).
            #   LOT 미상은 시간 간격으로 묶은 것뿐이라 그 첫 장이 LOT 첫 장이 아니다.
            if not include_first and lot and len(cur) >= 1:
                # Use full-history identity, even when a view starts mid-LOT
                # or its first wafer has no selected score.
                n_first_excluded += sum(bool(c[3]) for c in cur)
                cur = [c for c in cur if not c[3]]
            # ★ 필요 장수는 기준마다 다르다. 흩어짐(IQR)은 최소 2장이 있어야 '벌어짐'이
            #   정의되지만, 초과 강도는 그룹 평균이라 1장으로도 성립한다. 2장을 일괄
            #   요구하면 2장짜리 LOT 에서 첫 장을 빼는 순간 그룹이 통째로 사라져,
            #   정작 초과한 웨이퍼가 있는데도 칸이 비어 버린다.
            _need = 1 if basis in ("excess", "excess2") else max(2, int(min_n))
            if len(cur) < _need:
                if cur:
                    n_small += 1
                return
            _rate = _mean_ex = None
            if basis in ("margin", "cpu"):                # 임계까지 몇 σ (작을수록 위험)
                d = _group_margin_sigma([c[2] for c in cur])
                if basis == "cpu":
                    d = d / 3.0                           # 공정능력지수(단측) 스케일
            elif basis in ("excess", "excess2"):          # 초과 강도(빈도×정도)
                d, _rate, _mean_ex = _group_excess(
                    [c[2] for c in cur], 2.0 if basis == "excess2" else 1.0,
                    base=excess_base)
            else:
                u = np.asarray([c[1] for c in cur], dtype=np.float64)
                d = _disp_metric(u, metric)
            if not np.isfinite(d):
                return
            out.append({"t0": cur[0][0].isoformat(timespec="seconds"),
                        "t1": cur[-1][0].isoformat(timespec="seconds"),
                        "n": len(cur), "disp": d,
                        "lot": lot,                       # LOT 해시(폴백 그룹이면 None)
                        "rate": (round(_rate, 4) if _rate is not None else None),
                        "mean_ex": (round(_mean_ex, 4) if _mean_ex is not None else None)})
            allvals.append(d)

        if group_by == "lot":
            # ★ LOT 단위: 같은 해시끼리 모은다. 같은 LOT 이 시간상 떨어져 다시 돌아도
            #   (설비가 중간에 다른 LOT 을 끼워 돌려도) 하나의 그룹으로 정확히 묶인다.
            # ★ 첫 장 표시(_first_wafer_ids)와 똑같은 묶음을 쓴다 — _lot_runs 하나로 통일.
            #   LOT 재실행은 구간마다 나누고, LOT 미상은 붙어 있으면 흡수된다.
            for _i in range(len(items) - 1):
                gaps.append((items[_i + 1][0] - items[_i][0]).total_seconds() / 60.0)
            for lot, run in _lot_runs(items, gap_min, "lot"):
                _close(run, lot)
            out.sort(key=lambda g: g["t0"])               # 시작시각 순(레인 그리기용)
        else:
            for it in items:
                if cur:
                    gaps.append((it[0] - cur[-1][0]).total_seconds() / 60.0)
                    if (it[0] - cur[-1][0]).total_seconds() > gap_min * 60.0:
                        _close(cur); cur = []
                cur.append(it)
            _close(cur)
        groups[ch] = out
    base = float(np.median([v for v in allvals if np.isfinite(v)])) if allvals else float("nan")
    if not (np.isfinite(base) and base > 1e-9):
        base = 1.0
    if basis in ("margin", "cpu", "excess", "excess2"):
        base = 1.0                                        # 여유·Cpu·초과강도는 절대 스케일
    for ch, gs in groups.items():
        for g in gs:
            g["ratio"] = round(g["disp"] / base, 6)                # 산포비 D 또는 여유/Cpu
            g["disp"] = round(g["disp"], 4)
    gs = sorted(gaps)
    stat = {"n_groups": sum(len(v) for v in groups.values()), "n_small": n_small,
            "excess_base": round(float(excess_base), 3),
            # ★ 빈 칸의 이유를 화면이 말할 수 있게 — 시각 없음·점수 없음으로 빠진 수
            "n_no_ts": n_no_ts, "n_no_score": n_no_score,
            "n_wafers": len(parsed),
            "group_by": group_by,
            "include_first": bool(include_first),
            "n_first_excluded": n_first_excluded,         # 첫 장 제외로 빠진 수
            "n_lot_known": n_lot,                         # LOT 을 아는 웨이퍼 수
            "n_lots": len({(ch, g.get("lot")) for ch, gs2 in groups.items()
                           for g in gs2 if g.get("lot")}),
            "gap_median": round(gs[len(gs) // 2], 1) if gs else None,
            "gap_p90": round(gs[int(len(gs) * 0.9)], 1) if gs else None}
    if stat["n_groups"] == 0 and gs:                      # 그룹이 안 만들어질 때 권장값 제시
        stat["suggest_gap"] = float(max(gap_min, round(gs[int(len(gs) * 0.9)] * 1.5, 0) or gap_min))
    return {"groups": groups, "baseline": round(base, 4), "stat": stat}


def _theil_sen(xs, ys) -> float:
    """로버스트 기울기(중앙값 기울기) — 소수 이상치에 흔들리지 않는 추세."""
    x = np.asarray(xs, dtype=np.float64); y = np.asarray(ys, dtype=np.float64)
    n = x.size
    if n < 3:
        return 0.0
    idx = np.arange(n)
    if n > 60:                                                    # 표본 과다 시 균등 축소
        idx = np.linspace(0, n - 1, 60).astype(int)
        x, y = x[idx], y[idx]; n = x.size
    sl = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        ok = dx > 0
        if ok.any():
            sl.extend(((y[i + 1:][ok] - y[i]) / dx[ok]).tolist())
    return float(np.median(sl)) if sl else 0.0


# 폴더별 원본 로그 수 캐시 — 세는 것 자체가 비싸서(수십만 파일) 한 번 센 값을 재사용한다.
_LOG_COUNT_CACHE: Dict[str, tuple] = {}
_LOG_COUNT_TTL = 300.0


def _count_logs_cached(path: str, ttl: float = _LOG_COUNT_TTL):
    now = time.time()
    hit = _LOG_COUNT_CACHE.get(path)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    q = Path(path)
    if not _exists_bounded(q):
        return None
    n = 0
    try:
        for f in q.rglob("*"):
            if f.is_file() and f.suffix.lower() in LOG_EXTS:
                n += 1
                if n >= 500000:                           # 폭주 방지 상한
                    break
    except Exception:                                     # noqa: BLE001
        return None
    _LOG_COUNT_CACHE[path] = (now, n)
    return n


@app.get("/api/dispersion")
def api_dispersion(rng: str = Query("all", alias="range"), scale: str = "month",
                   recipe: str = "all",
                   start: str = "", end: str = "", chambers: str = "", recipes: str = "",
                   metric: str = "iqr", gap_min: float = 10.0, min_n: int = 2,
                   slices: int = 0, basis: str = "excess", hold: int = 0,
                   group_by: str = "lot", include_first: int = 1, over: str = "crit",
                   month_start_day: int = 1):
    """버킷(기본 월)별 산포 추세 — 버킷 내부를 균등 slice 로 나눠 그라데이션용 값을 준다.

    월마다 길이가 달라도 slice 는 '비율'이라, 월 초→말 색 변화를 같은 척도로 볼 수 있다.
    ★ 묶는 단위는 LOT(.log I2 해시)이 기본. group_by=gap 이면 예전 시간간격 방식."""
    rows = _history_for_current_model()
    group_by = "lot" if str(group_by).lower() != "gap" else "gap"
    _incl = bool(int(include_first))                      # LOT 첫 장 포함 여부(체크박스)
    # ★ 초과를 어디부터 셀 것인가 — 기본은 '위험(심각) 배수'.
    #   임계(1.0)부터 세면 살짝 넘긴 웨이퍼까지 전부 잡혀 칸이 온통 빨개진다.
    #   산포 화면은 '심하게 벌어진 구간'을 찾는 화면이므로 심각 등급부터 센다.
    #   over=thr 로 부르면 예전처럼 임계 기준.
    _bands = _display_rule().get("bands") or {}
    _ebase = 1.0
    if str(over).lower() != "thr":
        try:
            _ebase = max(1.0, float(_bands.get("crit") or 1.2))
        except (TypeError, ValueError):
            _ebase = 1.2
    # ★ 유형 드롭다운(recipe)도 파라미터에 넣는다.
    #   예전에는 이 API 가 recipe 를 받지 않아 화면이 보내도 무시했다. 그러면 개요(격자)는
    #   그 유형만 세는데 산포는 전 유형으로 계산해, 두 응답의 조회 구간과 칸 개수가 어긋난다.
    #   화면은 칸을 '순번'으로 짝지으므로 산포 색이 엉뚱한 날짜에 붙거나 통째로 비어 보인다.
    #   캐시 키에도 넣어야 유형을 바꿨을 때 옛 결과가 나오지 않는다.
    _p = (rng, scale, recipe, start, end, chambers, recipes, metric, gap_min, min_n, slices,
          basis, hold, group_by, _incl, round(_ebase, 3), month_start_day)
    _n = len(rows)                                        # ★ 필터 전 행 수 — 저장 때도 같은 값
    _hit = _agg_cached("dispersion", _p, _n)
    if _hit is not None:
        return _hit
    ch_set = {c.strip().lower() for c in chambers.split(",") if c.strip()} or None
    rc_set = {c.strip() for c in recipes.split(",") if c.strip()} or None
    if ch_set or rc_set:
        rows = [r for r in rows
                if (ch_set is None or str(r.get("chamber") or _chamber(r.get("wafer_id")) or "").lower() in ch_set)
                and (rc_set is None or _recipe_display(r.get("recipe") or "") in rc_set)]
    now = dt.datetime.now()
    t0, t1 = _range_window(rows, rng, start, end, now)
    if rng == "custom":
        t1 += dt.timedelta(seconds=1)
    rows = [r for r in rows
            if (_parse_ts(r.get("ts")) is not None
                and t0 <= _parse_ts(r.get("ts")) < t1)]
    # ★ 유형 드롭다운은 '조회 구간을 정한 뒤' 적용한다 — 개요(api_overview)와 같은 순서.
    #   구간을 유형으로 좁히면 개요와 칸 개수가 달라지고, 화면이 두 배열을 나란히 쓰므로
    #   색이 밀리거나 뒤쪽이 비어 보인다. 구간은 같게, 계산 대상만 유형으로 거른다.
    _want_rc = (recipe or "all").strip()
    if _want_rc and _want_rc != "all":
        rows = [r for r in rows if _recipe_display(r.get("recipe") or "") == _want_rc]
    result = _dispersion_result(rows, t0, t1, scale, metric, gap_min, min_n,
                                slices, basis, hold, group_by, _incl, _ebase, clip=rng == "custom", month_start_day=month_start_day)
    return _agg_store("dispersion", _p, _n, result)


def _dispersion_result(rows, t0, t1, scale, metric="iqr", gap_min=10.0, min_n=2,
                       slices=0, basis="excess", hold=0, group_by="lot",
                       _incl=True, _ebase=1.2, clip=False, month_start_day=1):
    """Shared LOT aggregation; rows must already contain the intended score ratios."""
    # ★ 기준은 '초과 강도(excess)' 하나로 정리(사용자 결정). 나머지 값이 들어와도
    #   과거 저장본/HTML 데모 호환을 위해 계산은 유지하되, 기본값은 excess 다.
    basis = basis if basis in ("spread", "margin", "cpu", "excess", "excess2") else "excess"
    res = _dispersion_groups(rows, gap_min, metric, min_n, basis, group_by, _incl,
                             excess_base=_ebase)
    bounds = _bucket_bounds(t0, t1, scale, clip=clip, month_start_day=month_start_day)                         # 개요와 동일한 달력 경계
    # ★ 조각 수 — 지정이 없으면(0) 칸의 '자연스러운 하위 단위'에 맞춘다.
    #   월 칸을 20등분하면 한 조각이 1.55일이라, 25·26일만 벌어져도 24.2~27.4일이
    #   통째로 붉게 번진다. 언제 문제였는지 짚어 보라는 화면인데 날짜가 흐려진다.
    #     월 → 그 달의 일수(하루 = 한 조각) · 주 → 7(하루) · 일 → 24(한 시간)
    if not slices:
        if scale == "month":
            _sp = [(bounds[i + 1] - bounds[i]).days for i in range(len(bounds) - 1)]
            slices = max(_sp) if _sp else 30           # 28~31 — 달마다 다르므로 최대치
        elif scale == "week":
            slices = 7
        else:
            slices = 24
    slices = max(4, min(int(slices), 60))
    # ★ 버킷 안의 여러 LOT 을 한 칸으로 합칠 때 쓰는 함수 — 기준에 따라 다르다.
    #   흩어짐/여유는 '보통 어느 수준인가'라 중앙값(이상치에 안 흔들림)이 맞다.
    #
    #   초과 강도는 '가장 심했던 LOT' 을 쓴다(최댓값).
    #     · 중앙값으로 합치면 LOT 10개 중 1개가 크게 넘어도 0 이 된다 — 넘은 사실이 지워진다.
    #     · 평균도 긴 구간에서는 희석된다. 한 달 칸에서 25~26일 이틀만 심하게 벌어져도
    #       나머지 28일의 0 과 섞여 0.02 가 나오고, 색은 연두로 떨어진다. 정작 그 이틀은
    #       분명한 문제인데 화면은 '정상'으로 읽힌다(고객사 혼선).
    #     · 이 화면이 답해야 하는 질문은 "이 구간이 평균적으로 어땠나"가 아니라
    #       "이 구간에 문제가 있었나"다. 그러면 최댓값이 맞다.
    #   평균은 버리지 않고 ratio_mean 으로 함께 내려 툴팁에서 같이 보여 준다.
    _agg = np.max if basis in ("excess", "excess2") else np.median
    lanes = []
    for ch in CHAMBERS:
        gs = res["groups"].get(ch) or []
        # ★ LOT 묶음은 '시작 시각 한 점'이 아니라 '구간'이다(t0~t1).
        #   예전에는 시작 시각이 든 칸에만 넣었다. 그래서 5/21 밤에 시작해 5/22 새벽까지
        #   이어진 LOT 은 5/21 칸에만 들어가고, 5/22 는 웨이퍼가 44장이나 처리됐는데도
        #   빈 칸이 됐다(일반 보기에서는 44/44 로 정상 표시). 자정을 걸친 LOT 이 있으면
        #   반드시 생기는 구멍이다.
        #   → 묶음이 걸쳐 있는 칸·조각에 모두 넣는다. '이 구간에 문제가 있었나'가 질문이므로
        #     그 시간대에 돌고 있었으면 그 칸의 값에 포함되는 것이 맞다.
        pts = []
        for g in gs:
            _a = _parse_ts(g["t0"]); _b = _parse_ts(g.get("t1")) or _a
            if _a is None:
                continue
            if _b < _a:
                _b = _a
            pts.append((_a, _b, g["ratio"]))
        b1 = bounds[-1]
        cells = []
        for bi in range(len(bounds) - 1):
            b0, b1 = bounds[bi], bounds[bi + 1]
            span = (b1 - b0).total_seconds() or 1.0
            inb = [(a, b, v) for a, b, v in pts if a < b1 and b >= b0]   # 겹치면 포함
            sl = []
            for si in range(slices):
                s0 = b0 + dt.timedelta(seconds=span * si / slices)
                s1 = b0 + dt.timedelta(seconds=span * (si + 1) / slices)
                vv = [v for a, b, v in inb if a < s1 and b >= s0]        # 조각도 겹침 기준
                sl.append(round(float(_agg(vv)), 6) if vv else None)
            xs = [(max(a, b0) - b0).total_seconds() / span for a, _b, _v in inb]
            ys = [v for _a, _b, v in inb]
            def _ovl(g):                                  # 칸과 겹치는 묶음인가(위와 같은 기준)
                _a = _parse_ts(g["t0"]); _b = _parse_ts(g.get("t1")) or _a
                return _a is not None and _a < b1 and (_b if _b and _b >= _a else _a) >= b0
            _rates = [g["rate"] for g in gs if g.get("rate") is not None and _ovl(g)]
            _mexs = [g["mean_ex"] for g in gs if g.get("mean_ex") is not None and _ovl(g)]
            cells.append({"n_groups": len(inb),
                          "ratio": (round(float(_agg(ys)), 6) if ys else None),
                          # 참고용 평균 — 칸 값(최댓값)과 함께 툴팁에 보여 준다.
                          "ratio_mean": (round(float(np.mean(ys)), 6) if ys else None),
                          "slope": round(_theil_sen(xs, ys), 4) if len(ys) >= 3 else 0.0,
                          # 초과 기준일 때만: 넘은 비율 · 넘은 평균 폭(툴팁 설명용)
                          "rate": (round(float(np.mean(_rates)), 4) if _rates else None),
                          "mean_ex": (round(float(np.mean(_mexs)), 4) if _mexs else None),
                          "slices": sl})
        # 데이터가 없는 구간은 기본적으로 '칠하지 않는다'(언제 데이터가 있었는지 보이도록).
        #   hold=1 이면 직전에 관측된 상태를 이어 칠해 추세선을 끊김 없이 본다(옵션).
        n_hold = 0
        if hold:
            last = None
            for c in cells:
                sl2 = c["slices"]
                for i2, v2 in enumerate(sl2):
                    if v2 is None:
                        if last is not None:
                            sl2[i2] = last; n_hold += 1
                    else:
                        last = v2
        lanes.append({"chamber": ch, "cells": cells,
                      "n_groups": len(gs), "n_hold": n_hold})
    return {
        "lanes": lanes, "baseline": res["baseline"], "stat": res.get("stat") or {},
        "basis": basis, "metric": metric, "hold": int(bool(hold)),
        "excess_base": round(_ebase, 3),                   # 화면 표기용(무엇을 넘은 것인지)
        "group_by": group_by, "include_first": _incl,
        "gap_min": gap_min, "min_n": min_n, "slices": slices,
        "bounds": [b.isoformat(timespec="seconds") for b in bounds],
        "scale": scale}


def _saved_path_entries() -> list:
    """설정에 저장된 경로성 항목 목록(현재 이 PC 에서 쓸 수 있는지 포함)."""
    cfg = _load_app_config()
    ent = []

    def _add(key, raw, label, fallback):
        if not raw:
            return
        pp = _path_load(raw)
        ent.append({"key": key, "label": label, "saved": str(raw), "resolved": str(pp),
                    "exists": bool(pp.exists()), "relative": not Path(str(raw)).is_absolute()
                    and not _is_foreign_abs(str(raw)), "fallback": fallback})

    _add("inbox", cfg.get("inbox"), "CSV 입력 폴더", "앱 폴더의 기본 입력 폴더")
    for ch, v in (cfg.get("chamber_paths") or {}).items():
        _add(f"chamber:{ch}", v, f"챔버 {ch} 데이터 폴더", "감시 대상에서 제외")
    _add("default_model", cfg.get("default_model"), "기본 모델 폴더", "내장 모델")
    _add("rca_h_artifacts", cfg.get("rca_h_artifacts"), "RCA H 모델 폴더", "모델 옆 자동탐색")
    return ent


class DisplayRuleReq(BaseModel):
    mode: Optional[str] = None                            # raw | per_tp
    recipe: Optional[str] = None                          # 적용 대상 레시피(없으면 전체)
    thr_scale: Optional[float] = None                     # raw 모드 임계 배율
    thr_abs: Optional[float] = None                       # per_tp 모드 절대 임계
    warn: Optional[float] = None
    crit: Optional[float] = None
    reset: Optional[bool] = None


@app.get("/api/display-rule")
def api_display_rule():
    r = _display_rule()
    _load_ntp()
    return {**r, "ntp_known": len(_NTP),
            "model_score_norm": str(getattr(ENGINE, "score_norm", "none"))}


@app.post("/api/display-rule")
def api_set_display_rule(req: DisplayRuleReq):
    """임계(레시피별)·모드·밴드 저장 → 산점도·목록·불량률추이·1시간현황이 즉시 같은 기준으로."""
    rule = _display_rule()
    if req.reset:
        return _save_display_rule(dict(_DISPLAY_RULE_DEFAULT))
    if req.mode:
        if req.mode not in ("raw", "per_tp"):
            raise HTTPException(400, "mode 는 raw | per_tp")
        if req.mode == "per_tp" and str(getattr(ENGINE, "score_norm", "none")) in ("time", "final"):
            raise HTTPException(400,
                "이 모델은 학습 단계에서 이미 타임포인트로 정규화되어 판정합니다"
                "(--score-final-avg/--score-time-avg). 앱에서 또 나누면 이중 정규화가 되어"
                " 사용할 수 없습니다 — 원점수 모드 그대로 두세요.")
        # ★ per_tp 로 처음 전환할 때 임계가 없으면 전 웨이퍼가 '판정 보류'가 되어 화면이 텅 빈다.
        #   → 지금(아직 raw 모드) 이력으로 OK 기준 임계를 자동 산출해 함께 저장한다.
        #   (모드를 먼저 바꾸면 relive 가 보류 처리되어 산출이 불가능하므로 순서가 중요)
        if req.mode == "per_tp" and not rule["thr_abs"] and req.thr_abs is None:
            sug = _suggest_per_tp_thr()
            if sug:
                rule["thr_abs"] = {_DISP_ALL: float(sug)}
                print(f"[display] per_tp 전환 — 임계 자동 설정 {sug:.6g}", flush=True)
        rule["mode"] = req.mode
    key = str(req.recipe).strip() if req.recipe else _DISP_ALL
    if req.thr_scale is not None:
        v = float(req.thr_scale)
        if not (v > 0):
            raise HTTPException(400, "임계 배율은 0보다 커야 합니다")
        rule["thr_scale"] = {**rule["thr_scale"], key: round(v, 4)}
    if req.thr_abs is not None:
        v = float(req.thr_abs)
        if not (v > 0):
            raise HTTPException(400, "임계값은 0보다 커야 합니다")
        rule["thr_abs"] = {**rule["thr_abs"], key: v}
    if req.warn is not None or req.crit is not None:
        b = dict(rule["bands"])
        if req.warn is not None:
            b["warn"] = float(req.warn)
        if req.crit is not None:
            b["crit"] = float(req.crit)
        rule["bands"] = b
    return _save_display_rule(rule)


def _log_sigma(vals) -> Optional[float]:
    """로그공간 표준편차(로버스트) — 복원오차는 곱셈적으로 퍼져 log 를 취하면 정규에 가깝다.
    중앙값/MAD 로 추정해 소수의 이상치에 흔들리지 않게 한다."""
    a = np.asarray([v for v in vals if _isnum(v) and float(v) > 0], dtype=np.float64)
    if a.size < 8:
        return None
    l = np.log(a)
    med = float(np.median(l))
    mad = float(np.median(np.abs(l - med))) * 1.4826
    if not np.isfinite(mad) or mad <= 1e-9:
        sd = float(np.std(l, ddof=1)) if a.size > 1 else 0.0
        mad = sd if sd > 1e-9 else None
    return mad


def _per_tp_samples(rows) -> tuple:
    """(모든 웨이퍼의 s_tp 리스트, OK 만의 s_tp 리스트) — n_tp 없으면 레시피 중앙값으로 추정."""
    byrc: Dict[str, List[int]] = {}
    allv: List[int] = []
    for r in rows:
        n = _ntp_of(r.get("wafer_id"), r)
        if n:
            byrc.setdefault(_recipe_display(r.get("recipe") or ""), []).append(int(n))
            allv.append(int(n))
    med_rc = {}
    for k, v in byrc.items():
        v.sort(); med_rc[k] = v[len(v) // 2]
    med_all = (sorted(allv)[len(allv) // 2]) if allv else 0
    tps, oks = [], []
    for r in rows:
        n = _ntp_of(r.get("wafer_id"), r) or med_rc.get(_recipe_display(r.get("recipe") or "")) or med_all
        raw = r.get("score_raw")
        if n and _isnum(raw) and float(raw) > 0:
            v = float(raw) / n
            tps.append(v)
            if r.get("verdict") == "OK":
                oks.append(v)
    return tps, oks


def _suggest_per_tp_thr(recipe: str = "", method: str = "match") -> Optional[float]:
    """per_tp 기본 임계 제안. ※ mode 가 raw 인 상태에서 호출할 것(전환 뒤엔 표본이 없음).

    · match(기본) = **분위수 매칭** — 지금 raw 판정의 불량 개수를 그대로 유지하는 임계.
      모드만 바꿨는데 불량 수가 갑자기 0이 되거나 폭증하는 혼란을 막는다(배점 θ_w 와 같은 원칙).
    · sigma = OK 모집단 s_tp 의 로그공간 μ+3σ(모드와 무관한 절대 기준).
    """
    try:
        rows = _history_for_current_model()
        if recipe:
            want = _recipe_display(recipe)
            rows = [r for r in rows if _recipe_display(r.get("recipe") or "") == want]
        tps, oks = _per_tp_samples(rows)
        if method == "match" and tps:
            k = sum(1 for r in rows if r.get("verdict") == "NG")
            if k > 0:
                a = sorted(tps, reverse=True)
                k = min(k, len(a))
                # k 번째와 k+1 번째 사이 → 정확히 k 개가 임계 이상이 되도록
                v = a[k - 1] if k >= len(a) else (a[k - 1] + a[k]) / 2.0
                if np.isfinite(v) and v > 0:
                    return float(v)
        if len(oks) < 8:
            return None
        l = np.log(np.asarray(oks, dtype=np.float64))
        med = float(np.median(l))
        mad = float(np.median(np.abs(l - med))) * 1.4826
        if mad <= 1e-9:
            mad = float(np.std(l, ddof=1))
        v = float(np.exp(med + 3 * mad))
        return v if np.isfinite(v) and v > 0 else None
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()
        return None


@app.get("/api/display-rule/auto")
def api_display_rule_auto(recipe: str = ""):
    """밴드·임계 자동 제안 — ★OK 판정 웨이퍼 모집단 기준(사용자 확정).

    · 밴드: log 공간 σ 추정 → 경고=임계×exp(σ), 위험=임계×exp(2σ).
      (log s ~ N(μ,σ) 이고 임계가 μ+3σ 면 4σ·5σ 지점이 정확히 이 배수가 된다)
    · per_tp 임계: OK 의 s_tp 로그공간 μ+3σ.
    """
    rows = _history_for_current_model()
    if recipe:
        want = _recipe_display(recipe)
        rows = [r for r in rows if _recipe_display(r.get("recipe") or "") == want]
    oks = [r for r in rows if r.get("verdict") == "OK"]
    rule = _display_rule()
    out = {"n_ok": len(oks), "n_all": len(rows), "recipe": recipe or "전체"}
    # ── 밴드(임계 배수) ──
    ratios = [r.get("ratio_thr") if _isnum(r.get("ratio_thr")) else r.get("score") for r in oks]
    sig = _log_sigma(ratios)
    if sig:
        warn = float(np.exp(sig)); crit = float(np.exp(2 * sig))
        out["sigma_log"] = round(sig, 4)
        out["bands"] = {"warn": round(min(max(warn, 1.05), 20.0), 3),
                        "crit": round(min(max(crit, 1.1), 40.0), 3)}
    else:
        out["bands"] = None
        out["note"] = "OK 표본이 부족해(8장 미만) 자동 산출 불가 — 값을 직접 입력하세요"
    # ── per_tp 절대 임계(OK 의 μ+3σ) ──
    tps = []
    for r in oks:
        n = _ntp_of(r.get("wafer_id"), r)
        raw = r.get("score_raw")
        if n and _isnum(raw) and float(raw) > 0:
            tps.append(float(raw) / n)
    if len(tps) >= 8:
        l = np.log(np.asarray(tps, dtype=np.float64))
        med = float(np.median(l))
        mad = float(np.median(np.abs(l - med))) * 1.4826
        if mad <= 1e-9:
            mad = float(np.std(l, ddof=1))
        _match = _suggest_per_tp_thr(recipe, "match")
        out["per_tp"] = {"n": len(tps), "median": float(np.exp(med)),
                         "sigma_log": round(float(mad), 4),
                         "suggest_thr": float(_match or np.exp(med + 3 * mad)),
                         "method": ("현재 불량 개수 유지(분위수 매칭)" if _match else "OK 로그공간 μ+3σ"),
                         "suggest_sigma": float(np.exp(med + 3 * mad))}
    else:
        out["per_tp"] = None
    out["current"] = rule
    return out


class UiConfigReq(BaseModel):
    overlay_sensitivity: Optional[float] = None
    inspect_cache: Optional[bool] = None                  # 검사 캐시(파싱+점수) on/off


@app.get("/api/ui-config")
def api_ui_config():
    """UI 기본값(현재: overlay 문제구간 민감도 기본값 0~1). 파일 영속 — 재시작에도 유지."""
    # ★ 배포판: 민감도 기본값 고정(0.8)·캐시 상시 꺼짐 — 설정 패널 제거됨
    return {"overlay_sensitivity": 0.8, "inspect_cache": False}


@app.post("/api/ui-config")
def api_set_ui_config(req: UiConfigReq):
    """UI 기본값 저장. 다음 overlay 차트부터(재시작 포함) 이 민감도로 시작."""
    return api_ui_config()                                # 배포판: 저장할 UI 기본값 없음(고정)


@app.get("/api/chambers")
def api_chambers(hours: float = 24):
    """챔버(dsv1~6)별 간략 현황 — 최근 N시간 불량/전체 + 5단계 위험도."""
    return chamber_overview(hours=hours)


# ── Overview 대시보드용 집계 (챔버×시간버킷 불량률 + 요약 + 현재 상태) ──
_UNIT_MIN = {"day": 1440, "week": 10080}                     # 균등 버킷 단위(분) — month 는 달력 기준
_PRESET_HRS = {"1h": 1, "6h": 6, "12h": 12, "24h": 24, "7d": 168, "30d": 720}   # 프리셋 조회기간(시간)


def _bucket_bounds(t0, t1, scale, clip=False, month_start_day=1):
    """버킷 경계 리스트 [b0, b1, ..., bN] (달력 정렬). scale=day/week/month.
    day=자정, week=월요일 자정, month=매월 1일(가변 길이 — 달력 그대로)."""
    bounds = []
    if not isinstance(month_start_day, int) or not 1 <= month_start_day <= 31:
        raise HTTPException(400, "월 시작일은 1~31일 정수여야 합니다.")
    if scale == "month":
        import calendar
        def anchor(year, month):
            return dt.datetime(year, month, min(month_start_day, calendar.monthrange(year, month)[1]))
        cur = anchor(t0.year, t0.month)
        if cur > t0:
            cur = anchor(t0.year - (t0.month == 1), (t0.month - 2) % 12 + 1)
        while cur < t1:
            # Include the requested final anchor day in the preceding partial box.
            if not (clip and month_start_day != 1 and cur > t0
                    and cur.date() == (t1 - dt.timedelta(seconds=1)).date()):
                bounds.append(cur)
            cur = anchor(cur.year + (cur.month == 12), cur.month % 12 + 1)
        bounds.append(cur)
    else:
        step = dt.timedelta(days=7 if scale == "week" else 1)
        base = t0.replace(hour=0, minute=0, second=0, microsecond=0)
        if scale == "week":
            base -= dt.timedelta(days=base.weekday())        # 월요일 정렬
        cur = base
        while cur < t1:
            bounds.append(cur)
            cur += step
        bounds.append(cur)
    if clip:
        bounds = [t0] + [b for b in bounds if t0 < b < t1] + [t1]
    if len(bounds) > 401:                                    # 과도 버킷 방지(최근 400칸)
        bounds = bounds[-401:]
    return bounds


def _parse_ts(ts):
    try:
        return dt.datetime.fromisoformat(str(ts))
    except Exception:                                    # noqa: BLE001
        return None


def _recent_n_window(rows, n, now):
    """가장 최근 검사한 웨이퍼 n개를 커버하는 시간창 (t0, t1).

    웨이퍼당 최신 1건 기준, ts 최근순 n개 중 가장 이른 ts=t0, t1=now.
    데이터가 없으면 (now-1시간, now)."""
    latest = {}
    for r in rows:
        wid = r.get("wafer_id")
        if not wid:
            continue
        prev = latest.get(wid)
        if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
            latest[wid] = r
    tss = sorted((_parse_ts(r.get("ts")) for r in latest.values() if _parse_ts(r.get("ts"))),
                 reverse=True)
    if not tss:
        return now - dt.timedelta(hours=1), now
    t0 = tss[min(len(tss), max(1, int(n))) - 1]
    return t0, now


def _range_window(rows, rng, start, end, now):
    """조회기간 → (t0, t1). recent30/custom/all/프리셋(1h·6h·12h·24h) 지원."""
    if rng == "recent30":
        return _recent_n_window(rows, 30, now)
    if rng == "custom" and start and end:
        return (_parse_ts(start) or (now - dt.timedelta(hours=6)),
                _parse_ts(end) or now)
    if rng == "all":
        # ★ '전체 기간' = 데이터가 있는 전 기간. 1970 년부터 잡으면 일/주 단위에서
        #   버킷 상한(400칸)에 걸려 빈 칸만 잔뜩 그려지고, 내보낸 HTML(데이터 범위로
        #   잡는다)과 x축이 통째로 달라진다 — 같은 데이터인데 화면이 다르다.
        #   이력이 없을 때만 예전처럼 넓게 잡는다.
        _ts = [t for t in (_parse_ts(r.get("ts")) for r in (rows or [])) if t is not None]
        if _ts:
            return (min(_ts), max(_ts) + dt.timedelta(days=1))
        return (dt.datetime(1970, 1, 1), now + dt.timedelta(days=1))
    hrs = _PRESET_HRS.get(rng, 6)                          # 기본 6시간
    return (now - dt.timedelta(hours=hrs), now)


@app.get("/api/overview")
def api_overview(rng: str = Query("1h", alias="range"), scale: str = "min",
                 recipe: str = "all", start: str = "", end: str = "",
                 chambers: str = "", recipes: str = "", month_start_day: int = 1):
    """Overview 화면 한 방 집계: 조회기간(range)·스케일·레시피 기준으로
    챔버별 시간버킷 불량률(lanes) + 기간 요약 + 현재 챔버 상태를 반환.

    ★ chambers/recipes(csv)는 화면의 체크박스 상태다(산점도와 같은 규칙):
      ''=전체 · '__NONE__'=아무것도 선택 안 함 · 그 외=선택된 것만.
      이게 없으면 체크를 해제해도 불량률 추이 히트맵만 그대로 남아 산점도와 어긋난다.
    """
    rows = _history_for_current_model()
    ch_set = {c.strip().lower() for c in chambers.split(",") if c.strip()} or None
    rc_set = {c.strip() for c in recipes.split(",") if c.strip()} or None
    if ch_set or rc_set:
        rows = [r for r in rows
                if (ch_set is None
                    or str(r.get("chamber") or _chamber(r.get("wafer_id")) or "").lower() in ch_set)
                and (rc_set is None or _recipe_display(r.get("recipe") or "") in rc_set)]
    # ★ 같은 조건이면 재계산하지 않는다(체크박스 조작 때마다 O(N) 집계가 도는 것 방지)
    _p = (rng, scale, recipe, start, end, chambers, recipes, month_start_day)
    _n = len(rows)                                        # 조회·저장에 같은 값을 쓴다
    _hit = _agg_cached("overview", _p, _n)
    if _hit is not None:
        return _hit
    now = dt.datetime.now()

    # ── 조회기간(window) ──
    t0, t1 = _range_window(rows, rng, start, end, now)
    # ── 버킷 단위(일/주/월). 미지정·구단위면 창 크기에 맞춰 자동 ──
    total_days = max(0.01, (t1 - t0).total_seconds() / 86400.0)
    if scale not in ("day", "week", "month"):
        scale = "day" if total_days <= 31 else ("week" if total_days <= 183 else "month")
    if rng == "custom":
        t1 += dt.timedelta(seconds=1)  # UI end is inclusive to the second
    bounds = _bucket_bounds(t0, t1, scale, clip=rng == "custom", month_start_day=month_start_day)                # 달력 정렬 경계(월=가변 길이)
    base = bounds[0]
    count = len(bounds) - 1

    # 라벨: 일='MM-DD' · 주='MM-DD'(주 시작 월요일) · 월='YYYY-MM'
    buckets = []
    for i in range(count):
        bs = bounds[i]
        if scale == "month" and month_start_day != 1:
            last = bounds[i + 1] - dt.timedelta(seconds=1)
            buckets.append(f"{bs.month}/{bs.day}~{last.month}/{last.day}")
        else:
            buckets.append(f"{bs.year}-{bs.month:02d}" if scale == "month"
                           else f"{bs.month:02d}-{bs.day:02d}")

    def rmatch(r):
        return recipe == "all" or _recipe_display(r.get("recipe")) == recipe

    recipes = sorted({_recipe_display(r.get("recipe")) for r in rows
                      if r.get("recipe") and _recipe_display(r.get("recipe")) != "unknown"})

    # ★ 웨이퍼당 최신 1건만(같은 웨이퍼 재검사 중복 집계 방지 — 25장×3회=75장 방지)
    _latest = {}
    for r in rows:
        wid = r.get("wafer_id")
        if not wid:
            continue
        prev = _latest.get(wid)
        if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
            _latest[wid] = r
    uniq_rows = list(_latest.values())

    # 범위 내 이벤트를 (챔버, 버킷인덱스, NG여부) 로 1회 순회 집계
    import bisect
    # [total, bad, caution, warn, crit] — 등급별로 나눠 담아 '경고 포함/위험만' 보기를 지원
    lane_cells = {ch: [[0, 0, 0, 0, 0] for _ in range(count)] for ch in CHAMBERS}
    tot = good = bad = na = 0
    n_caution = n_warn = n_crit = 0
    n_hist_thr = 0
    n_unassigned = 0                                      # 격자에 자리가 없던 수
    for r in uniq_rows:
        if not rmatch(r):
            continue
        ts = _parse_ts(r.get("ts"))
        if ts is None or not bounds[0] <= ts < bounds[-1]:
            continue
        if r.get("band") == "na":                             # per_tp 인데 n_tp 미상 → 판정 보류
            na += 1
            continue
        ts = _parse_ts(r.get("ts"))
        if ts is None:
            continue
        # ★ 요약·lane 을 '화면에 그려지는 버킷 범위(bounds)'로 통일 — 버킷 상한(400칸) 때문에
        #   base(bounds[0]) 가 t0 보다 뒤로 밀려도 요약 총계와 lane 합이 어긋나지 않게.
        k = bisect.bisect_right(bounds, ts) - 1               # 달력 경계 기반 버킷 배정
        if not (0 <= k < count):
            continue
        is_ng = (r.get("verdict") == "NG")
        bd = str(r.get("band") or ("caution" if is_ng else "ok"))
        if r.get("thr_from_history"):
            n_hist_thr += 1
        tot += 1
        good += (0 if is_ng else 1)
        bad += (1 if is_ng else 0)
        n_caution += (1 if bd == "caution" else 0)
        n_warn += (1 if bd == "warn" else 0)
        n_crit += (1 if bd == "crit" else 0)
        ch = _chamber_key(r)                                   # 폴더귀속 우선(소문자 통일)
        if ch not in lane_cells:
            n_unassigned += 1                             # dsv1~6 밖(미상·범위 밖) — 격자에 자리가 없다
        if ch in lane_cells:
            c_ = lane_cells[ch][k]
            c_[0] += 1
            c_[1] += (1 if is_ng else 0)
            c_[2] += (1 if bd == "caution" else 0)
            c_[3] += (1 if bd == "warn" else 0)
            c_[4] += (1 if bd == "crit" else 0)

    lanes = []
    for ch in CHAMBERS:
        cells, ct, cb = [], 0, 0
        for total_, bad_, cau_, wrn_, crt_ in lane_cells[ch]:
            cells.append({"total": total_, "bad": bad_,
                          "caution": cau_, "warn": wrn_, "crit": crt_,
                          "defectPct": round(100 * bad_ / total_, 1) if total_ else 0.0})
            ct += total_; cb += bad_
        lanes.append({"chamber": ch, "avgDefect": round(100 * cb / ct, 1) if ct else 0.0,
                      "cells": cells})

    hours = max(0.05, (t1 - t0).total_seconds() / 3600.0)
    status = chamber_overview(hours=hours, t0=t0, t1=t1)["chambers"]   # 화면과 같은 구간
    # ── ★ '데이터 기준 최근 1시간' 블록(카드·실시간 챔버 상태용) ──
    #   벽시계가 아니라 '가장 최근 데이터의 끝시각'을 닻으로 1시간 창을 만든다.
    #   설비가 잠시 쉬거나 과거 데이터를 보는 중에도 카드가 비지 않고,
    #   조회기간과 무관하게 항상 '마지막 1시간 생산'을 보여준다(제목과 일치).
    #   (예전에는 이 카드가 조회기간 '전체' 요약을 보여줘 제목과 어긋났다)
    # ★ 벽시계 '지금' 기준 [now-1h, now] — 데이터 최신시각을 닻으로 쓰면 과거
    #   데이터를 넣기만 해도 '최근 1시간'에 올라오는 오해가 생겼다(사용자 지적).
    #   과거 데이터는 조회기간 화면(추이·산점도)에서 본다. 설비가 1시간 넘게
    #   쉬면 0 으로 비는 것이 맞는 표시다.
    hour = {"total": 0, "good": 0, "bad": 0, "caution": 0, "warn": 0, "crit": 0,
            "na": 0, "unassigned": 0, "goodRate": 0.0, "badRate": 0.0,
            "warnRate": 0.0, "critRate": 0.0, "anchor": None}
    hour_chambers = {c: {"total": 0, "ng": 0, "max_ratio": 0.0,
                         "caution": 0, "warn": 0, "crit": 0, "level": 0} for c in CHAMBERS}
    if True:
        _anchor = now
        _h0 = _anchor - dt.timedelta(hours=1)
        hour["anchor"] = _anchor.strftime("%Y-%m-%dT%H:%M:%S")
        _hlatest: Dict = {}
        for r in rows:
            _t = _parse_ts(r.get("ts"))
            if _t is None or not (_h0 <= _t <= _anchor) or not rmatch(r):
                continue
            wid = r.get("wafer_id")
            prev = _hlatest.get(wid)
            if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
                _hlatest[wid] = r
        for r in _hlatest.values():
            if r.get("band") == "na":                     # 판정 보류(요약과 같은 규칙)
                hour["na"] += 1
                continue
            is_ng = (r.get("verdict") == "NG")
            bd = str(r.get("band") or ("caution" if is_ng else "ok"))
            hour["total"] += 1
            hour["good"] += (0 if is_ng else 1)
            hour["bad"] += (1 if is_ng else 0)
            for k2 in ("caution", "warn", "crit"):
                hour[k2] += (1 if bd == k2 else 0)
            ch = _chamber_key(r)
            if ch not in hour_chambers:
                hour["unassigned"] += 1
            if ch in hour_chambers:
                b2 = hour_chambers[ch]
                b2["total"] += 1
                b2["ng"] += (1 if is_ng else 0)
                for k2 in ("caution", "warn", "crit"):
                    b2[k2] += (1 if bd == k2 else 0)
                try:
                    # 챔버 실시간 상태(chamber_overview)와 같은 정밀값(_hist_ratio)
                    b2["max_ratio"] = max(b2["max_ratio"], float(_hist_ratio(r)))
                except (TypeError, ValueError):
                    pass
        if hour["total"]:
            hour["goodRate"] = round(100 * hour["good"] / hour["total"], 1)
            hour["badRate"] = round(100 * hour["bad"] / hour["total"], 1)
            hour["warnRate"] = round(100 * hour["warn"] / hour["total"], 1)
            hour["critRate"] = round(100 * hour["crit"] / hour["total"], 1)
        pass
    return _agg_store("overview", _p, _n, {
        "range": rng, "scale": scale, "recipe": recipe,
        "base": base.isoformat(timespec="seconds"),       # 버킷0 시작(하위호환)
        "bounds": [b.isoformat(timespec="seconds") for b in bounds],   # 버킷 경계(클릭 산점도용)
        "count": count,
        "now": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {"total": tot, "good": good, "bad": bad, "na": na,
                    "caution": n_caution, "warn": n_warn, "crit": n_crit,
                    # 현재 모델에 임계가 없어 '검사 당시 임계'로 등급을 매긴 건수(0 이면 정상)
                    "thr_from_history": n_hist_thr,
                    # 격자(dsv1~6)에 자리가 없어 칸에 표시되지 못한 수 — 조용히 사라지지 않게
                    "unassigned": n_unassigned,
                    "goodRate": round(100 * good / tot, 1) if tot else 0.0,
                    "badRate": round(100 * bad / tot, 1) if tot else 0.0,
                    "warnRate": round(100 * n_warn / tot, 1) if tot else 0.0,
                    "critRate": round(100 * n_crit / tot, 1) if tot else 0.0},
        "bands": _display_rule()["bands"],
        "chambers": status,
        "hour": hour,                                     # 데이터 기준 최근 1시간 요약
        "hour_chambers": [dict(hour_chambers[c], chamber=c,
                               level=_chamber_level(hour_chambers[c]["total"],
                                                    hour_chambers[c]["ng"],
                                                    hour_chambers[c]["max_ratio"]),
                               max_ratio=round(hour_chambers[c]["max_ratio"], 3))
                          for c in CHAMBERS],
        "buckets": buckets,
        "lanes": lanes,
        "recipes": recipes,                               # 실제 레시피명 목록(필터 드롭다운)
    })


@app.get("/api/scatter")
def api_scatter(limit: int = 300, hours: float = 0, chamber: str = "all",
                start: str = "", end: str = "", recipe: str = "all",
                chambers: str = "", recipes: str = ""):
    """이상점수 추이(시간순) + 선형 추세.  y = score/threshold (임계=1 정규화).
    chamber(단일)/chambers(csv 다중, 체크박스) · recipe(단일)/recipes(csv 다중) · start/end(ISO).
    표시 순서 = 웨이퍼 '종료시각' 오름차순(왼쪽=가장 먼저 끝난 웨이퍼)."""
    rows = _history_for_current_model()
    # ★ 응답 캐시: 상한 제거 후 N점 전체를 매 폴링마다 재직렬화하면 대량(2만+)에서 무겁다.
    #   이력(모델·길이·배율·모드)이 그대로고 같은 파라미터면 직전 응답 재사용.
    _sig = (_model_id(), len(rows), _recipe_mult_sig(), _score_mode())
    _key = (limit, hours, chamber, start, end, recipe, chambers, recipes)
    _c = _SCATTER_CACHE
    if _c.get("sig") == _sig and _key in _c["map"]:
        return _c["map"][_key]
    now = dt.datetime.now()
    t0 = _parse_ts(start) if start else None
    t1 = _parse_ts(end) if end else None
    if chambers == "__NONE__" or recipes == "__NONE__":     # 전체 체크해제 → 아무것도 표시 안 함
        return {"points": [], "trend": {"slope": 0.0, "dir": "flat", "y0": None, "y1": None}}
    ch_set = ({c.strip() for c in chambers.split(",") if c.strip()} or None) if chambers else None
    rc_set = ({c.strip() for c in recipes.split(",") if c.strip()} or None) if recipes else None
    latest: Dict = {}                                       # 웨이퍼당 최신 1건
    for r in rows:
        wid = r.get("wafer_id")
        if not wid:
            continue
        rch = _chamber_key(r)
        if ch_set is not None and rch not in ch_set:
            continue                                        # 챔버 체크박스(통합 보기)
        if chamber != "all" and ch_set is None and rch != chamber:
            continue
        rrc = _recipe_display(r.get("recipe"))          # 표시·필터 정규화(풀네임→배율키)
        if rc_set is not None and rrc not in rc_set:
            continue                                        # 레시피 체크박스
        if recipe != "all" and rc_set is None and rrc != recipe:
            continue                                        # 레시피 필터(드롭다운 연동)
        ts = _parse_ts(r.get("ts"))
        if t0 is not None and (ts is None or ts < t0):     # 박스 기간 필터
            continue
        if t1 is not None and (ts is None or ts >= t1):
            continue
        if (t0 is None and t1 is None) and hours and not _within_hours(r, hours, now):
            continue
        prev = latest.get(wid)
        if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
            latest[wid] = r
    pts = []
    _rule_now = _display_rule()
    _bands = _rule_now["bands"]
    n_na = 0
    n_est = 0
    na_reason = ""
    for wid, r in latest.items():
        if r.get("band") == "na":                           # 판정 보류(임계 미설정·타임포인트 미상)
            n_na += 1
            na_reason = na_reason or str(r.get("disp_na") or "")
            continue
        if r.get("ntp_est"):
            n_est += 1
        try:
            thr = float(r.get("threshold")); sc = float(r.get("score"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(thr) and thr > 0 and np.isfinite(sc)):
            continue                                        # 임계 없음/이상치 → 제외
        ratio = sc / thr                                    # 임계=1 정규화
        # ★ 표시 점수는 '버림'이 규칙이다(사용자 결정). 다만 버림을 두 번 하면 안 된다.
        #   위 sc 는 이미 2자리로 버려진 값이라, 그걸 또 버리면 한 칸 더 깎인다.
        #   부동소수에서 4.10 은 4.0999999999999996 이므로 4.1025 → 4.10 → 4.09.
        #   정밀값(ratio_thr, 6자리)에서 '한 번만' 버린다 — 규칙은 그대로 버림이다.
        _disp_ratio = float(r["ratio_thr"]) if _isnum(r.get("ratio_thr")) else ratio
        pts.append({"ts": r.get("ts"), "wafer_id": wid,
                    "chamber": _chamber_key(r) or None,
                    "recipe": _recipe_display(r.get("recipe")),
                    "mean_recipe": str(r.get("recipe") or ""),
                    "ratio": _floor2(_disp_ratio),              # 점수 표시=버림(한 번만)
                    # 계산용 정밀값 — 산포 추세가 쓰는 값(ratio_thr)을 그대로 전달.
                    # ※ 위 score 는 이미 2자리 버림이라 여기서 다시 나누면 정밀도가 사라진다.
                    "ratio_x": (float(r["ratio_thr"]) if _isnum(r.get("ratio_thr"))
                                else round(float(ratio), 6)),
                    # 배율 적용 전 원비율 — 내보낸 HTML 이 임계·밴드를 다시 계산할 때 필요
                    "ratio_base": (float(r["ratio_base"]) if _isnum(r.get("ratio_base")) else None),
                    "score": round(sc, 6), "threshold": round(thr, 6),
                    "band": r.get("band") or _disp_band(ratio, _bands),   # 주의/경고/심각 등급
                    "n_tp": (int(r["n_tp"]) if _isnum(r.get("n_tp")) else None),
                    "lot": _lot_of(wid, r),                     # 산포 그룹 키(첫 장 판별에도 씀)
                    "verdict": r.get("verdict")})
    pts.sort(key=lambda p: str(p.get("ts") or ""))         # 종료시각 오름차순(왼쪽=먼저 끝난 웨이퍼)
    # ★ LOT 첫 장 표시 — '첫 장 포함' 체크를 끄면 산포 계산에서 빠지는 웨이퍼다.
    #   화면은 이 표시를 보고 같은 색으로 흐리게 그린다(숨기지 않는다 — 있었다는 사실은 남긴다).
    #   산포 그룹이 챔버×LOT 단위이므로 여기서도 같은 키로 첫 장을 고른다.
    #   ★ 판정은 '전체 이력'에 산포와 같은 규칙을 적용해 한 번에 구한다.
    #     자기 결과 안에서만 고르면 (1) LOT 미상 웨이퍼가 빠지고
    #     (2) 구간을 좁혀 본 산점도에서 그 창의 첫 장을 LOT 첫 장으로 착각한다.
    try:
        _firsts = _first_wafer_ids(_history_for_current_model())
    except Exception:                                      # noqa: BLE001
        _firsts = set()
    for _p in pts:
        _p["lot_first"] = str(_p.get("wafer_id")) in _firsts
    # ★ 점 개수 상한 없음(사용자 요청) — 조회기간·필터에 해당하는 모든 웨이퍼를 그대로 표시.
    #   과거엔 상한 초과 시 tail 만 남겨(pts[-limit:]) 초반 날짜(특정 챔버만 있는 구간)가
    #   통째로 사라졌음. 이제 전부 반환한다. (limit 파라미터는 하위호환용으로 유지하되 무시)
    # 선형 추세(최소자승) — x=순번, y=ratio(없으면 score)
    ys = [(p["ratio"] if p["ratio"] is not None else p["score"]) for p in pts]
    trend = {"slope": 0.0, "dir": "flat", "y0": None, "y1": None}
    n = len(ys)
    if n >= 2:
        x = np.arange(n, dtype=float); y = np.asarray(ys, dtype=float)
        xm, ym = x.mean(), y.mean()
        denom = float(((x - xm) ** 2).sum())
        slope = float(((x - xm) * (y - ym)).sum() / denom) if denom else 0.0
        inter = ym - slope * xm
        rng = (y.max() - y.min()) or 1.0
        d = "up" if slope > 0.02 * rng / max(1, n) else \
            ("down" if slope < -0.02 * rng / max(1, n) else "flat")
        trend = {"slope": round(slope, 6), "dir": d,
                 "y0": round(inter, 4), "y1": round(inter + slope * (n - 1), 4)}
    _resp = {"model_id": _model_id(), "points": pts, "trend": trend, "n": n, "n_na": n_na, "n_est": n_est,
             "na_reason": na_reason,
             "rule": {**_rule_now, "model_score_norm": str(getattr(ENGINE, "score_norm", "none"))}}
    if _c.get("sig") != _sig:
        _c["sig"] = _sig; _c["map"] = {}
    if len(_c["map"]) < 16:                               # 파라미터 조합 상한(메모리 보호)
        _c["map"][_key] = _resp
    return _resp



class ScatterSensorWaferReq(BaseModel):
    date: str
    wid: str


class ScatterSensorsReq(BaseModel):
    model_id: str
    wafers: List[ScatterSensorWaferReq]


@app.post("/api/scatter/sensors")
@_pin_model
def api_scatter_sensors(req: ScatterSensorsReq):
    """Bounded archive-only batches; no inference. Preserve missing samples as gaps."""
    if req.model_id != _model_id():
        raise HTTPException(409, "모델이 변경되었습니다. 산점도를 다시 생성하세요.")
    if len(req.wafers) > 64:
        raise HTTPException(400, "한 번에 최대 64개 웨이퍼를 조회할 수 있습니다.")
    valid = {str(r.get("wafer_id")) for r in _history_for_current_model()}
    rows = []
    for item in req.wafers:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item.date) or item.wid not in valid:
            raise HTTPException(400, "현재 모델의 유효한 웨이퍼와 날짜가 필요합니다.")
        scores = {}
        missing = None
        try:
            result = api_archive_sensors_all(item.date, item.wid)
            for sensor in result["sensors"]:
                value = sensor.get("ratio")
                if value is not None and np.isfinite(float(value)):
                    scores[str(sensor["sensor"])] = float(value)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            missing = "보관 데이터 없음"
        except (ValueError, KeyError, OSError) as exc:
            missing = "센서 점수 계산 불가"
            print(f"[scatter-sensors] {item.wid}: {exc}", flush=True)
        rows.append({"wid": item.wid, "scores": scores, "missing": missing})
    return {"model_id": req.model_id, "wafers": rows}


class ScatterMaxSensorsReq(ScatterSensorsReq):
    steps: Optional[List[int]] = None
    sensors: Optional[List[str]] = None


@app.post("/api/scatter/sensors-max")
@_pin_model
def api_scatter_sensors_max(req: ScatterMaxSensorsReq):
    """Overview sensors only. Wafer verdicts and anomaly-detail sigma stay unchanged."""
    from src.app.app_sensor_scores import load_reference, sensor_scores, AppScoreUnavailable
    if req.model_id != _model_id():
        raise HTTPException(409, "모델이 변경되었습니다. 산점도를 다시 생성하세요.")
    if len(req.wafers) > 64 or req.steps == [] or req.sensors == []:
        raise HTTPException(400, "최대 64개 웨이퍼와 비어 있지 않은 선택이 필요합니다.")
    valid = {str(r.get("wafer_id")) for r in _history_for_current_model()}
    references, rows = {}, []
    for item in req.wafers:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item.date) or item.wid not in valid:
            raise HTTPException(400, "현재 모델의 유효한 웨이퍼와 날짜가 필요합니다.")
        row = dict(wid=item.wid, scores={}, missing=None)
        try:
            arch = _load_archive(item.date, item.wid)
            if arch is None:
                raise AppScoreUnavailable('보관 데이터 없음')
            engine = _engine_for_wafer(item.wid, arch['meta'])
            key = str(engine.art)
            if key not in references:
                try:
                    references[key] = load_reference(engine.art)
                except AppScoreUnavailable as exc:
                    references[key] = exc
            if isinstance(references[key], Exception):
                raise references[key]
            level = arch.get('app_score_level')
            if level is None:
                level = arch['z']
            result = sensor_scores(engine.art, level, arch['step'],
                arch['meta'].get('sensor_cols') or list(engine.model.sensor_cols),
                reference=references[key], selected_steps=req.steps,
                selected_sensors=req.sensors, sensor_basis='train_ok_max')
            row['scores'] = {s['sensor']: s['ratio'] for s in result['sensors'] if s['ratio'] is not None}
            missing = [s['sensor'] for s in result['sensors'] if s['ratio'] is None]
            if missing:
                row['missing'] = '정상 학습 최댓값이 0인 센서: ' + ', '.join(missing)
        except AppScoreUnavailable as exc:
            row['missing'] = str(exc)
        except (ValueError, KeyError, OSError):
            row['missing'] = '보관 데이터에서 센서 점수를 계산할 수 없습니다.'
        rows.append(row)
    return dict(model_id=req.model_id, sensor_basis='train_ok_max', wafers=rows)


# Cache only small trend summaries, never reconstruction/score arrays.
from src.app.sensor_differences import TrendSummaryCache
_TREND_SUMMARY_CACHE = TrendSummaryCache()


def _trend_summary(date, wid, pairs=None):
    from src.app.sensor_differences import difference_series
    path = _archive_path(date, wid)

    def build():
        if not path.exists() and _ARCHIVE_Q is not None and getattr(_ARCHIVE_Q, "unfinished_tasks", 0):
            _archive_flush(timeout=3.0)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta_json"].tobytes()).decode("utf-8"))
            # NpzFile is lazy: do not decompress unused model input arrays.
            keys = set(data.files)
            arch = {"meta": meta, "step": data["step"],
                    "rca_io": data["rca_io"] if "rca_io" in keys else None}
            if {"trend_actual", "trend_step"} <= keys:
                arch.update(trend_actual=data["trend_actual"], trend_step=data["trend_step"])
            else:
                arch.update(actual=data["actual"],
                            rca_insen=data["rca_insen"] if "rca_insen" in keys else None)
            summary = difference_series(arch, pairs=pairs)
        return {"summary": summary, "meta": {k: meta[k] for k in ("step_timing", "source_path") if k in meta}}

    cache_key = (_model_id(), tuple(tuple(p) for p in (pairs or [])), 'sensor-block-v2-zero')
    result = _TREND_SUMMARY_CACHE.get(path, cache_key, build)
    if result is None or (result['summary'].get('sensor_scope') == 'full_block' and result['summary'].get('zero_step_version') == 1):
        return result
    source = _trend_original_path(result['meta'], wid)
    if source is None:
        return result
    def complete():
        from src.data import _read_csv
        from src.trend_sensors import trend_measurements, attach_trend_measurements
        frame = _read_csv(source)
        measurements = trend_measurements(frame)
        if measurements is None:
            return result
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data['meta_json'].tobytes()).decode('utf-8'))
            arch = dict(meta=meta, step=data['step'], rca_io=data['rca_io'] if 'rca_io' in data else None)
            summary = difference_series(attach_trend_measurements(arch, measurements), pairs=pairs)
        from src.data import step_timing
        return dict(summary=summary, meta=dict(result['meta'], step_timing=step_timing(frame)))
    try:
        return _TREND_SUMMARY_CACHE.get(source, (cache_key, _TREND_SUMMARY_CACHE.signature(path)), complete)
    except (ValueError, OSError, KeyError):
        return result


def _trend_original_path(meta, wid):
    direct = meta.get('source_path')
    if direct and Path(direct).is_file():
        return Path(direct)
    return _find_wafer_source(wid)



class ScatterDifferencesReq(ScatterSensorsReq):
    pairs: List[List[str]] = []


def _scatter_trend_rows(req, include_timing=False):
    """All requested wafers, bounded batches, archive measurements only."""
    if req.model_id != _model_id():
        raise HTTPException(409, "모델이 변경되었습니다. 산점도를 다시 생성하세요.")
    if len(req.wafers) > 64:
        raise HTTPException(400, "한 번에 최대 64개 웨이퍼를 조회할 수 있습니다.")
    pairs = list(getattr(req, "pairs", []))
    if include_timing:
        pairs.append(["d_001_PV", "d_004_PV"])
    if len(pairs) > 24 or any(len(p) != 2 or p[0] == p[1] or any(not n or len(n) > 200 for n in p) for p in pairs):
        raise HTTPException(400, "서로 다른 센서 두 개로 최대 24개 비교를 지정하세요.")
    valid = {str(r.get("wafer_id")) for r in _history_for_current_model()}
    rows = []
    for item in req.wafers:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item.date) or item.wid not in valid:
            raise HTTPException(400, "현재 모델의 유효한 웨이퍼와 날짜가 필요합니다.")
        row = dict(wid=item.wid, date=item.date, series=[], missing=None)
        summary = None
        try:
            summary = _trend_summary(item.date, item.wid, pairs=pairs)
            if summary is None:
                row['missing'] = '보관 데이터 없음'
            else:
                row.update(summary['summary'])
        except (ValueError, KeyError, OSError):
            row['missing'] = '보관된 실측 센서값을 읽을 수 없습니다.'
        if include_timing:
            from src.app.sensor_differences import step_sensor_metrics
            # Reuse this wafer's summary before it can be evicted from the LRU.
            row['cylinder_difference'] = next((v for v in row.get('series', []) if v.get('left') == 'd_001_PV' and v.get('right') == 'd_004_PV'), None)
            row['sensor_metrics'] = step_sensor_metrics(row)
            row['step_timing'] = (_wafer_step_timing(item.wid, {'meta': summary['meta']})
                                  if summary is not None else {'steps': {}, 'missing': '보관 데이터 없음'})
            row['peaks'] = {s['sensor']: {st: stat['max'] for st, stat in s.get('step_stats', {}).items()}
                            for s in row.get('sensor_series', [])}
            for name in ('series', 'sensor_series'):
                row.pop(name, None)
        rows.append(row)
    return dict(model_id=req.model_id, wafers=rows)



@_pin_model
def api_scatter_differences(req: ScatterDifferencesReq):
    return _scatter_trend_rows(req)


def _wafer_step_timing(wid, arch):
    saved = arch['meta'].get('step_timing')
    if saved is not None and saved.get('version') == 3 and saved.get('steps') and all(v.get('seconds') is not None for v in saved['steps'].values()):
        return saved
    direct = arch['meta'].get('source_path')
    source = Path(direct) if direct and Path(direct).is_file() else _find_wafer_source(wid)
    if source is not None:
        from src.data import _read_csv, step_timing
        try:
            return _TREND_SUMMARY_CACHE.get(source, ("timing", _model_id()),
                                            lambda: step_timing(_read_csv(source)))
        except (ValueError, OSError, KeyError) as exc:
            return {'steps': {}, 'missing': '원본 Time 읽기 실패: ' + str(exc)}
    if saved and saved.get('steps'):
        return dict(saved, missing='원본 없음: 기존 시간 요약 사용(재계산하려면 원본 복원 필요)')
    return {'steps': {}, 'missing': 'Time 자료 없음: 원본 로그 복원 또는 재검사 필요'}


@_pin_model
def api_scatter_step_times(req: ScatterSensorsReq):
    return _scatter_trend_rows(req, include_timing=True)


# Summaries contain only JSON-native values. Returning JSONResponse avoids
# FastAPI recursively re-encoding every statistic before serializing it again.
@app.post("/api/scatter/differences")
def api_scatter_differences_http(req: ScatterDifferencesReq):
    return JSONResponse(api_scatter_differences(req))


@app.post("/api/scatter/step-times")
def api_scatter_step_times_http(req: ScatterSensorsReq):
    return JSONResponse(api_scatter_step_times(req))


class ScatterStepScoresReq(ScatterSensorsReq):
    steps: List[int]
    sensors: Optional[List[str]] = None


@app.get("/api/scatter/steps")
@_pin_model
def api_scatter_steps():
    from src.app.app_sensor_scores import load_reference, AppScoreUnavailable
    steps, sensors, issues = set(), set(), []
    engines = list(ENGINE.root.chamber_engines.items()) or [("model", ENGINE)]
    for name, engine in engines:
        try:
            reference = load_reference(engine.art)
            steps.update(int(s) for s in reference[1])
            sensors.update(reference[2])
        except AppScoreUnavailable as exc:
            issues.append(str(name) + ": " + str(exc))
    return {"model_id": _model_id(), "steps": sorted(steps), "sensors": sorted(sensors), "issues": issues}


@app.post("/api/scatter/steps")
@_pin_model
def api_scatter_step_scores(req: ScatterStepScoresReq):
    from src.app.app_sensor_scores import load_reference, sensor_scores, AppScoreUnavailable
    if req.model_id != _model_id():
        raise HTTPException(409, "모델이 변경되었습니다. 산점도를 다시 생성하세요.")
    if len(req.wafers) > 64 or not req.steps or 0 in req.steps or req.sensors == []:
        raise HTTPException(400, "최대 64개 웨이퍼, ProcessStep 0을 제외한 스텝과 센서 선택이 필요합니다.")
    valid = {str(r.get("wafer_id")) for r in _history_for_current_model()}
    references, rows = {}, []
    for item in req.wafers:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item.date) or item.wid not in valid:
            raise HTTPException(400, "현재 모델의 유효한 웨이퍼와 날짜가 필요합니다.")
        row = {"wid": item.wid, "scores": {}, "ratio": None, "missing": None}
        try:
            arch = _load_archive(item.date, item.wid)
            if arch is None:
                raise AppScoreUnavailable("보관 데이터 없음")
            engine = _engine_for_wafer(item.wid, arch["meta"])
            key = str(engine.art)
            if key not in references:
                try:
                    references[key] = load_reference(engine.art)
                except AppScoreUnavailable as exc:
                    references[key] = exc
            if isinstance(references[key], Exception):
                raise references[key]
            level = arch.get("app_score_level")
            if level is None:
                level = arch["z"]
            result = sensor_scores(engine.art, level, arch["step"],
                                   arch["meta"].get("sensor_cols") or list(engine.model.sensor_cols),
                                   selected_steps=req.steps, reference=references[key], selected_sensors=req.sensors)
            result = apply_selected_override(result, dict(arch["meta"], wafer_id=item.wid), getattr(engine, "wafer_score_override", None))
            row.update(score_overridden=bool(result.get("score_overridden")),
                       matched_pattern=result.get("matched_pattern"), override_ratio=result.get("override_ratio"),
                       ratio=result["total_ratio"], score=result["total_score"],
                       threshold=result["total_threshold"],
                       scores={s["sensor"]: s["ratio"] for s in result["sensors"] if s["ratio"] is not None})
            if row["ratio"] is None:
                row["missing"] = "선택 스텝의 학습 임계가 0입니다."
        except AppScoreUnavailable as exc:
            row["missing"] = str(exc)
        except (ValueError, KeyError, OSError) as exc:
            row["ratio"], row["scores"] = None, {}
            row["missing"] = "보관 데이터에서 스텝 점수를 계산할 수 없습니다."
            print(f"[scatter-steps] {item.wid}: {exc}", flush=True)
        rows.append(row)
    return {"model_id": req.model_id, "steps": sorted(set(req.steps)), "sigma": 3.5, "wafers": rows}


@app.get("/api/debug/wafer-score-override")
@_pin_model
def api_debug_wafer_score_override(wid: str = ""):
    """Read-only trace of active filename policies; never rerun or alter inference."""
    from pathlib import PurePosixPath
    import src.wafer_score_override as policy_module
    history = _history_for_current_model()
    selected = [r for r in history if str(r.get("wafer_id")) == wid] if wid else history[-20:]
    if wid and not selected:
        raise HTTPException(404, "현재 모델 이력에서 해당 wafer_id를 찾지 못했습니다.")
    rows = []
    for saved in selected:
        wafer = str(saved.get("wafer_id") or "")
        date = str(saved.get("date") or saved.get("ts") or "")[:10]
        archive = _load_archive(date, wafer) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else None
        rec = dict((archive or {}).get("meta") or {}, **{k:v for k,v in saved.items() if v is not None})
        rec["wafer_id"] = wafer
        engine = _engine_for_wafer(wafer, rec)
        rule = normalize_override(getattr(engine, "wafer_score_override", None))
        source = str(rec.get("source_path") or "").strip()
        basename = PurePosixPath(source.replace("\\", "/")).name if source else wafer
        rules = (rule or {}).get("rules") or []
        sensitive = (rule or {}).get("case_sensitive", True)
        matched = [r for r in rules if (r["pattern"] if sensitive else r["pattern"].casefold()) in (basename if sensitive else basename.casefold())]
        with ENGINE.using(engine):
            _, threshold = _recipe_threshold(wafer)
        audit = effective_wafer_score(rec, 0., threshold, rule)
        reason = "applied" if audit["score_overridden"] else "rule_missing" if rule is None else "basename_not_matched" if not matched else "invalid_threshold"
        rows.append(dict(wafer_id=wafer, source_path=source, basename=basename,
                         model_directory=str(engine.art), rule=rule, matched_patterns=[r["pattern"] for r in matched],
                         matched_rule=matched[0] if matched else None, matched_pattern=audit.get("matched_pattern"),
                         decision_threshold=threshold, reason=reason,
                         expected_ratio=audit.get("override_ratio"),
                         saved_ratio=saved.get("ratio_thr", saved.get("score")),
                         saved_score_overridden=bool(saved.get("score_overridden"))))
    engines = ENGINE.root.chamber_engines or {"model": ENGINE.root}
    return dict(diagnostic_version="basename-override-v2", model_id=_model_id(),
                policy_module=str(policy_module.__file__),
                models={name:dict(directory=str(engine.art), rule=getattr(engine,"wafer_score_override",None)) for name,engine in engines.items()},
                wafers=rows, note="현재 규칙의 basename 매칭 진단입니다. 센서/스텝 선택 임계 계산과 실제 재추론은 실행하지 않습니다.")


class OverviewSelectionReq(BaseModel):
    model_id: str
    steps: List[int]
    sensors: List[str]
    month_start_day: int = 1
    rng: str = "1h"
    scale: str = "day"
    recipe: str = "all"
    start: str = ""
    end: str = ""
    chambers: str = ""
    recipes: str = ""
    dispersion: bool = False
    include_first: bool = True
    hold: bool = False


@app.post("/api/overview/selection")
@_pin_model
def api_overview_selection(req: OverviewSelectionReq):
    """Recount visible calendar buckets using the scatter selection, without editing history."""
    import copy
    import bisect
    if req.model_id != _model_id():
        raise HTTPException(409, "모델이 변경되었습니다. 다시 조회하세요.")
    if not req.steps or 0 in req.steps or not req.sensors:
        raise HTTPException(400, "스텝과 센서를 선택하세요.")
    overview = copy.deepcopy(api_overview(req.rng, req.scale, req.recipe, req.start,
                                         req.end, req.chambers, req.recipes, month_start_day=req.month_start_day))
    bounds = [_parse_ts(v) for v in overview["bounds"]]
    scatter = api_scatter(start=overview["bounds"][0], end=overview["bounds"][-1],
                          recipe=req.recipe, chambers=req.chambers, recipes=req.recipes)
    points = [p for p in scatter["points"]
              if _parse_ts(p.get("ts")) is not None and bounds[0] <= _parse_ts(p["ts"]) < bounds[-1]]
    scores = {}
    for offset in range(0, len(points), 64):
        result = api_scatter_step_scores(ScatterStepScoresReq(
            model_id=req.model_id, steps=req.steps, sensors=req.sensors,
            wafers=[dict(wid=p["wafer_id"], date=p["ts"][:10]) for p in points[offset:offset+64]]))
        scores.update({r["wid"]: r for r in result["wafers"]})
    lanes = {lane["chamber"]: lane for lane in overview["lanes"]}
    for lane in lanes.values():
        lane["cells"] = [dict(total=0, bad=0, caution=0, warn=0, crit=0, na=0, defectPct=0)
                         for _ in range(len(bounds)-1)]
    for p in points:
        lane = lanes.get(str(p.get("chamber") or "").lower())
        if lane is None:
            continue
        cell = lane["cells"][bisect.bisect_right(bounds, _parse_ts(p["ts"]))-1]
        ratio = scores.get(p["wafer_id"], {}).get("ratio")
        if ratio is None:
            cell["na"] += 1
            continue
        cell["total"] += 1
        cell["bad"] += int(ratio >= 1)
        band = _disp_band(ratio, overview["bands"])
        if band in ("caution", "warn", "crit"):
            cell[band] += 1
    for lane in lanes.values():
        for cell in lane["cells"]:
            cell["defectPct"] = round(100*cell["bad"]/cell["total"], 1) if cell["total"] else 0
        total = sum(c["total"] for c in lane["cells"])
        lane["avgDefect"] = round(100*sum(c["bad"] for c in lane["cells"])/total, 1) if total else 0
    overview["selection"] = dict(steps=req.steps, sensors=req.sensors, sigma=3.5)
    overview["model_id"] = req.model_id
    if req.dispersion:
        # Reuse the scores calculated for the cards. Never fall back to original
        # scores for missing selected scores: that would mix two calibrations.
        selected_rows = [dict(p, ratio_thr=scores.get(p["wafer_id"], {}).get("ratio"),
                              score=None) for p in points]
        dispersion = _dispersion_result(
            selected_rows, bounds[0], bounds[-1], overview.get("scale") or req.scale,
            hold=int(req.hold), _incl=req.include_first, clip=req.rng == "custom", month_start_day=req.month_start_day,
            _ebase=max(1.0, float(overview["bands"].get("crit") or 1.2)))
        dispersion["selection"] = overview["selection"]
        overview["dispersion"] = dispersion
    return overview


def _wt_item(r):
    """이력 row → 웨이퍼 목록 표시용 dict."""
    try:
        sc = float(r.get("score")); thr = float(r.get("threshold"))
    except (TypeError, ValueError):
        sc = thr = None
    d = _parse_end_time(r.get("wafer_id"))
    return {
        "wafer_id": r.get("wafer_id"),
        "chamber": _chamber_key(r) or None,
        "recipe": _recipe_display(r.get("recipe")),      # 표시 정규화(풀네임→배율키 29a)
        "score": (round(sc, 4) if sc is not None else None),
        "threshold": (round(thr, 4) if thr is not None else None),
        # 정밀값(ratio_thr)에서 한 번만 버린다 — score 는 이미 버려진 값이라 또 버리면
        # 한 칸 더 깎인다(4.1025 → 4.10 → 4.09).
        "ratio": (_floor2(float(r["ratio_thr"])) if _isnum(r.get("ratio_thr"))
                  else (_floor2(sc / thr) if (sc and thr and thr > 0)
                        else (_floor2(sc) if sc is not None else None))),
        "verdict": r.get("verdict"),
        # ★ 등급(주의/경고/심각) — 산점도 밴드와 같은 기준으로 목록·판정을 표시하기 위함
        "band": r.get("band"),
        "ratio_thr": (round(float(r["ratio_thr"]), 4) if _isnum(r.get("ratio_thr")) else None),
        "n_bad_lines": int(r.get("n_bad_lines") or 0),   # >0 이면 목록에 빨간 ❗ 표시
        "end_ts": (d.isoformat(timespec="seconds") if d else r.get("ts")),
    }


@app.get("/api/wafer-list")
def api_wafer_list(rng: str = Query("1h", alias="range"), scale: str = "min",
                   start: str = "", end: str = "", chamber: str = "all",
                   recipe: str = "all", q: str = "", page: int = 1,
                   page_size: int = 30, defects_only: bool = False,
                   chambers: str = "", recipes: str = ""):
    """'조회기간'(챔버 추이와 동일)에 검사한 모든 웨이퍼 목록 — 페이지당 최대 30.

    기간은 상단 조회기간을 그대로 따른다(range=10m/1h/24h 또는 custom+start/end).
    레시피/챔버/검색 필터는 '선택했을 때만' 적용(기본 all=전체). defects_only=NG만.
    반환 wafers 는 최근순(위=최근). span=현재 페이지가 커버하는 기간."""
    rows = _history_for_current_model()
    now = dt.datetime.now()
    # 기간 창(overview 와 동일 규칙). recent30=최근30매(시간필터 없이 최근순).
    if rng == "recent30" or rng == "all":
        t0 = dt.datetime(1970, 1, 1); t1 = now + dt.timedelta(days=1)
    else:
        t0, t1 = _range_window(rows, rng, start, end, now)

    latest: Dict = {}                                     # 웨이퍼당 최신 1건
    for r in rows:
        wid = r.get("wafer_id")
        if not wid:
            continue
        prev = latest.get(wid)
        if prev is None or str(r.get("ts", "")) >= str(prev.get("ts", "")):
            latest[wid] = r

    def endkey(r):
        d = _parse_end_time(r.get("wafer_id"))
        return d.isoformat() if d else str(r.get("ts") or "")

    ql = (q or "").strip().lower()

    if chambers == "__NONE__" or recipes == "__NONE__":     # 전체 체크해제 → 빈 목록
        return {"wafers": [], "total": 0, "ng_total": 0, "page": 1, "pages": 1,
                "page_size": page_size, "defects_only": defects_only, "recipes": [], "span": ""}
    _chs = ({c.strip() for c in chambers.split(",") if c.strip()} or None) if chambers else None
    _rcs = ({c.strip() for c in recipes.split(",") if c.strip()} or None) if recipes else None

    def keep(r):
        ch = _chamber_key(r)
        if _chs is not None:
            if ch not in _chs:                            # 챔버 체크박스(통합 보기)
                return False
        elif chamber != "all" and ch != chamber:
            return False
        rrc = _recipe_display(r.get("recipe"))
        if _rcs is not None:
            if rrc not in _rcs:                           # 레시피 체크박스
                return False
        elif recipe != "all" and rrc != recipe:
            return False
        if defects_only and r.get("verdict") != "NG":
            return False
        if ql and ql not in str(r.get("wafer_id") or "").lower():
            return False
        ts = _parse_ts(r.get("ts"))                       # 조회기간 필터
        if ts is None or ts < t0 or ts >= t1:
            return False
        return True

    items = [r for r in latest.values() if keep(r)]
    page_size = max(1, min(200, int(page_size)))
    if rng == "recent30":                                 # 최근 30개만 뽑되
        items.sort(key=endkey, reverse=True)
        items = items[:page_size]
    items.sort(key=endkey)                                # 표시는 '먼저 처리한 순'(오래된→최근, 위=먼저)
    total = len(items)
    ng_total = sum(1 for r in items if r.get("verdict") == "NG")
    pages = max(1, -(-total // page_size))
    page = max(1, min(pages, int(page)))
    sl = items[(page - 1) * page_size: page * page_size]
    wafers = [_wt_item(r) for r in sl]
    span = None
    if wafers:
        es = [w["end_ts"] for w in wafers if w["end_ts"]]
        if es:
            span = {"from": min(es), "to": max(es)}
    # ★ 필터 옵션도 행/매칭과 '같은' 표시명(_recipe_display: 풀네임→29a)으로 통일.
    #   (raw 풀네임을 옵션으로 주면, 행은 29a 라 선택해도 아무것도 안 걸리는 버그가 났었음)
    recipes = sorted({_recipe_display(r.get("recipe")) for r in latest.values()
                      if r.get("recipe") and _recipe_display(r.get("recipe")) != "unknown"})
    _rule = _display_rule()
    return {"bands": _rule["bands"], "mode": _rule["mode"],
            "total": total, "ng_total": ng_total, "page": page, "pages": pages,
            "page_size": page_size, "defects_only": bool(defects_only),
            "range": rng, "chamber": chamber, "recipe": recipe, "q": q,
            "span": span, "recipes": recipes, "wafers": wafers}


class ToggleReq(BaseModel):
    enabled: bool


@app.get("/api/autopilot")
def api_autopilot():
    if AUTOPILOT is None:
        raise HTTPException(503, "AutoPilot 미기동")
    return AUTOPILOT.status()


@app.post("/api/autopilot/toggle")
def api_autopilot_toggle(req: ToggleReq):
    if AUTOPILOT is None:
        raise HTTPException(503, "AutoPilot 미기동")
    AUTOPILOT.set_enabled(req.enabled)
    return AUTOPILOT.status()


@app.post("/api/autopilot/backup")
def api_autopilot_backup():
    """수동 마감 백업(현재 inbox → backup/<오늘>/) + 보관정리."""
    if AUTOPILOT is None:
        raise HTTPException(503, "AutoPilot 미기동")
    moved = AUTOPILOT.backup_now()
    return {"moved": moved, "autopilot": AUTOPILOT.status()}


@app.get("/api/wafer/{wid}")
def api_wafer(wid: str):
    try:
        return wafer_detail(wid)
    except KeyError:
        raise HTTPException(404, f"검사 결과에 없는 wafer: {wid} (먼저 검사 실행)")
    except Exception as e:                                 # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/wafer/{wid}/plot/{kind}")
def api_plot(wid: str, kind: str):
    try:
        path = ensure_plot(wid, kind)
        return FileResponse(path, media_type="image/png")
    except KeyError:
        raise HTTPException(404, f"검사 결과에 없는 wafer: {wid}")
    except Exception as e:                                 # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/wafer/{wid}/sensors")
def api_wafer_sensors(wid: str, n: int = 6):
    """상위 N개 이탈 센서 목록(rank+이름) — 센서별 갤러리 구성용."""
    try:
        return wafer_sensors(wid, n=n)
    except KeyError:
        raise HTTPException(404, f"검사 결과에 없는 wafer: {wid}")
    except Exception as e:                                 # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/wafer/{wid}/sensor/{rank}")
def api_wafer_sensor(wid: str, rank: int):
    """rank-번째 상위 센서의 복원오차 상세 PNG(실제vs기대 + 누적)."""
    try:
        path = ensure_sensor_plot(wid, rank)
        return FileResponse(path, media_type="image/png")
    except KeyError:
        raise HTTPException(404, f"검사 결과에 없는 wafer: {wid}")
    except IndexError:
        raise HTTPException(404, f"센서 rank 범위 초과: {rank}")
    except Exception as e:                                 # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/export-html")
@_pin_model
def api_export_html(detail_limit: int = 50, sensor_top: int = 12,
                    include_scores: bool = True, include_trends: bool = True,
                    include_detail: bool = True, trend_metrics: str = "mean,min,max,std",
                    include_chamber_means: bool = False,
                    include_differences: Optional[bool] = None,
                    include_equipment: Optional[bool] = None,
                    include_recipe: Optional[bool] = None, trend_pairs: str = "[]"):
    """현재 데이터 스냅샷을 넣은 '데모용 단일 HTML' 다운로드(서버 없이 브라우저에서 동작).

    대시보드와 이상분석 목록은 전체 검사 데이터를 포함한다.
    누적·센서·오버레이 상세만 종료시각 기준 최근 detail_limit 장(기본 50매)의 보관 자료를 포함한다.
    새 검사/저장 등 변경 동작은 데모 모드로 무효화(읽기전용)."""
    from src.app.export_html import build_export_html
    include_differences = include_trends if include_differences is None else include_differences
    include_equipment = include_trends if include_equipment is None else include_equipment
    include_recipe = include_trends if include_recipe is None else include_recipe
    include_trends = include_differences or include_equipment or include_recipe
    metrics = list(dict.fromkeys(trend_metrics.split(',')))
    if include_trends and (not metrics or any(m not in ('mean', 'min', 'max', 'std') for m in metrics)):
        raise HTTPException(status_code=400, detail="센서 추이 통계는 평균/min/max/표준편차 중 하나 이상 선택하세요.")
    try:
        export_pairs = json.loads(trend_pairs)
        if (not isinstance(export_pairs, list) or len(export_pairs) > 48 or
            any(not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1] or
                any(not isinstance(name, str) or not name or len(name) > 200 for name in pair)
                for pair in export_pairs)):
            raise ValueError('invalid pairs')
    except (ValueError, TypeError):
        raise HTTPException(400, "센서 차이 조합 형식이 올바르지 않습니다.")
    include_scores = include_scores or include_detail
    ENGINE.load()
    # ① WAFERS — 대시보드 전체(산점도 계산과 동일 소스)
    sc = api_scatter(limit=1000000)
    wafers = []
    for p in sc.get("points", []):
        ts = str(p.get("ts") or "")
        wafers.append({"wafer_id": p.get("wafer_id"), "chamber": p.get("chamber"),
                       # ★ LOT — 이력(history)에는 없고 file_meta 에 있으므로 여기서 조회해 담는다.
                       #   이게 없으면 내보낸 HTML 에서 산포 추세가 'LOT 미상'으로 떨어져
                       #   앱과 다른 화면이 된다(사용자 보고).
                       "lot": _lot_of(p.get("wafer_id")),
                       "lot_first": bool(p.get("lot_first")),   # 첫 장 흐리게 표시용
                       "recipe": p.get("recipe"), "mean_recipe": p.get("mean_recipe"), "ratio": p.get("ratio"),
                       "score": p.get("score", p.get("ratio")), "verdict": p.get("verdict"),
                       # ★ 등급(주의/경고/심각)을 그대로 담아야 내보낸 HTML 이 같은 화면이 된다
                       "band": p.get("band"), "n_tp": p.get("n_tp"),
                       "ratio_x": p.get("ratio_x"),            # 산포 계산용 정밀값(2자리 버림 방지)
                       "ratio_base": p.get("ratio_base"),      # 임계·밴드 재계산용(배율 적용 전)
                       "threshold": p.get("threshold", 1.0),
                       "ts": ts, "end_ts": ts, "date": ts[:10]})
    # Keep all summary rows so exported date queries can access older inspections.
    wafers.sort(key=lambda w: (str(w.get("end_ts") or w.get("ts") or ""),
                               str(w.get("wafer_id") or "")))
    # Compact sensor scores for every exported wafer, independent of detail_limit.
    scatter_scores = {}
    from src.app.app_sensor_scores import OfflineScoreExport
    offline_scores = OfflineScoreExport()
    from src.app.sensor_differences import difference_series, compact_trend_export, step_sensor_metrics
    sensor_differences = {}
    step_times = {}
    for start in range(0, len(wafers) if include_scores else 0, 64):
        batch = wafers[start:start + 64]
        result = api_scatter_sensors(ScatterSensorsReq(
            model_id=_model_id(),
            wafers=[ScatterSensorWaferReq(date=w["date"], wid=w["wafer_id"]) for w in batch]))
        for w, row in zip(batch, result["wafers"]):
            scatter_scores[w["date"] + "/" + w["wafer_id"]] = row
    for w in wafers if (include_scores or include_trends or include_chamber_means) else []:
        key = w["date"] + "/" + w["wafer_id"]
        try:
            arch = _load_archive(w["date"], w["wafer_id"])
            if arch is not None and (include_trends or include_chamber_means or include_scores):
                trend_summary = _trend_summary(w["date"], w["wafer_id"], pairs=(export_pairs if include_differences else []) + [["d_001_PV", "d_004_PV"]])
                full_stats = trend_summary["summary"] if trend_summary is not None else difference_series(arch, pairs=(export_pairs if include_differences else []) + [["d_001_PV", "d_004_PV"]])
                export_metrics = list(metrics) if include_trends else []
                if include_chamber_means and 'mean' not in export_metrics:
                    export_metrics.append('mean')
                sensor_differences[key] = compact_trend_export(full_stats, export_metrics)
                if ['d_001_PV', 'd_004_PV'] not in export_pairs:
                    sensor_differences[key]['series'] = [v for v in sensor_differences[key]['series'] if not (v.get('left') == 'd_001_PV' and v.get('right') == 'd_004_PV')]
                if not include_differences:
                    sensor_differences[key]['series'] = []
                if not (include_differences or include_equipment or include_chamber_means):
                    sensor_differences[key]['sensor_series'] = []
                if not (include_differences or include_equipment or include_chamber_means):
                    del sensor_differences[key]
            if arch is not None and (include_recipe or include_scores):
                step_times[key] = {'cylinder_difference': next((v for v in full_stats.get('series', []) if v.get('left') == 'd_001_PV' and v.get('right') == 'd_004_PV'), None), 'valve_phases': full_stats.get('valve_phases'), 'process_steps': full_stats.get('process_steps', []), 'zero_steps': full_stats.get('zero_steps', []), 'zero_step_version': full_stats.get('zero_step_version', 0), 'sensor_scope': full_stats.get('sensor_scope'), 'sensor_metrics': step_sensor_metrics(full_stats), 'step_timing': _wafer_step_timing(w['wafer_id'], {'meta': trend_summary['meta']} if trend_summary is not None else arch),
                                   'peaks': {s['sensor']: {st: stat['max'] for st, stat in s.get('step_stats', {}).items()}
                                             for s in full_stats.get('sensor_series', [])}}
            if not include_scores:
                continue
            if arch is None:
                offline_scores.wafers[key] = {"missing": "보관 데이터 없음"}
                continue
            engine = _engine_for_wafer(w["wafer_id"], arch["meta"])
            level = arch.get("app_score_level")
            offline_scores.add(key, engine.art, arch["z"] if level is None else level,
                               arch["step"], arch["meta"].get("sensor_cols") or list(engine.model.sensor_cols),
                               record=dict(arch["meta"], wafer_id=w["wafer_id"]),
                               override=getattr(engine, "wafer_score_override", None))
        except (ValueError, KeyError, OSError) as exc:
            offline_scores.wafers[key] = {"missing": str(exc)}
    # ② DETAIL — 최근 N장만(파일 크기 관리). 실제 아카이브 응답 + zone-contrib 용 z.
    detail: Dict = {}
    recent = wafers[-max(0, int(detail_limit)):] if detail_limit and include_detail else []
    for w in recent:
        date, wid = w["date"], w["wafer_id"]
        if not date or not wid:
            continue
        arch = _load_archive(date, wid)
        if arch is None:
            continue
        try:
            m = arch["meta"]
            model = _engine_for_wafer(wid, m).model
            ncols = len(model.sensor_cols)
            sa = api_archive_sensors_all(date, wid)
            # 기여 상위 센서만 sensor-series 구움(용량 절감), 클릭 시 그 센서 실제vs기대 표시
            top = sorted(sa.get("sensors", []), key=lambda x: -(x.get("score") or 0))[:max(1, int(sensor_top))]
            ss = {}
            for it in top:
                j = int(it.get("index", -1))
                if 0 <= j < ncols:
                    try:
                        ss[str(j)] = api_archive_sensor_series(date, wid, j)
                    except Exception:                         # noqa: BLE001
                        pass
            z = np.asarray(arch["z"], dtype=np.float64)
            keep = RX._plot_time_filter(model, _archive_rec(arch), z.shape[0])[0]
            zc = np.round(np.clip(z[keep], 0, None), 3).tolist()
            cols = m.get("sensor_cols") or [str(c) for c in model.sensor_cols]
            detail[date + "/" + wid] = {
                "meta": {"wafer_id": wid, "date": date, "recipe": m.get("recipe"),
                         "chamber": m.get("chamber"), "score": m.get("score"),
                         "threshold": m.get("threshold", 1.0), "verdict": m.get("verdict"),
                         "score_raw": m.get("score_raw"), "threshold_raw": m.get("threshold_raw"),
                         "ts": w["ts"], "n_flag_sensor": m.get("n_flag_sensor")},
                "wseries": api_archive_wafer_series(date, wid),
                "sensorsAll": sa, "appSensorScores": api_archive_app_sensor_scores(date, wid),
                "sensorSeries": ss, "z": zc, "cols": cols}
        except Exception:                                     # noqa: BLE001
            traceback.print_exc()
    # ③ CFG — 설정/상태 스냅샷
    mult = {}                                             # 배포판: 배수 기능 제거(중립)
    try:
        uisens = float(api_ui_config().get("overlay_sensitivity", 0.8))
    except Exception:                                         # noqa: BLE001
        uisens = 0.8
    try:
        sens = api_sensors()
    except Exception:                                         # noqa: BLE001
        sens = {"input_sensors": [], "io": []}
    # ★ 체크박스 목록은 '데이터에 실제로 있는 유형'이어야 한다.
    #   예전엔 모델이 아는 레시피(임계 키: 232·29a…)를 넣어서, 화면 표시명(풀네임)과
    #   달라 유형 체크를 하나라도 해제하면 매칭 0건 → 산점도·히트맵이 통째로 사라졌다.
    # 목록 출처는 '이력에 실제로 있는 유형'(개요/산점도가 쓰는 표시명과 같은 함수).
    #   스냅샷(wafers)만 보면 임계가 없어 산점도에서 빠진 웨이퍼의 유형이 누락된다.
    _rc_in_data = sorted({_recipe_display(r.get("recipe") or "")
                          for r in _history_for_current_model()
                          if r.get("recipe")}
                         | {str(w.get("recipe") or "") for w in wafers if w.get("recipe")})
    cfg = {"status": api_status(), "now": dt.datetime.now().replace(microsecond=0).isoformat(),
           "recipes": _rc_in_data or (mult.get("known_recipes") or []),
           "known_recipes": mult.get("known_recipes") or [],   # 배율 설정 화면용(짧은 키)
           "multipliers": mult.get("multipliers") or {},
           "mult_default": mult.get("default", 1.0), "uisens": uisens,
           "reten": (api_get_retention() or {}).get("days", 60),
           "chamber_paths": api_chamber_paths(), "models": api_models(),
           "sensors": sens.get("input_sensors") or [], "io": sens.get("io") or [],
           "recon_sensors": sens.get("recon_sensors") or [],
           "trend_export": {"metrics": metrics, "individual_steps": True},
           "export_features": {"scores": include_scores, "trends": include_trends, "differences": include_differences, "equipment": include_equipment, "recipe": include_recipe, "detail": include_detail, "chamber_means": include_chamber_means},
           "sensor_differences": sensor_differences, "step_times": step_times,
           "scatter_scores": scatter_scores, "offline_scores": offline_scores.payload(), "model_id": _model_id(),
           # ★ 표시 판정 규칙(밴드 경계·모드·임계)을 함께 저장 — 내보낸 HTML 이 기본값(2/3)으로
           #   떨어지지 않고 '지금 보고 있는 화면' 그대로 재현되게 한다.
           "rule": {**_display_rule(),
                    "model_score_norm": str(getattr(ENGINE, "score_norm", "none"))},
           "band_view": None, "export_manual_query": True}
    html_src = (STATIC / "index.html").read_text(encoding="utf-8")
    out = build_export_html(html_src, wafers, detail, cfg, static_dir=STATIC)
    from starlette.responses import Response
    from urllib.parse import quote
    fn = "챔버이상검사_데모_" + _today().replace("-", "") + ".html"
    # 헤더는 latin-1 만 허용 → 한글은 RFC5987 퍼센트 인코딩(filename*), ASCII 폴백 병기.
    disp = "attachment; filename=\"demo.html\"; filename*=UTF-8''" + quote(fn)
    return Response(content=out, media_type="text/html; charset=utf-8",
                    headers={"Content-Disposition": disp})


#  ── RCA(근본원인분석): 불량 웨이퍼의 .log 를 ONNX 로 재분석 → slope-밸브 리포트+이미지 ──
RCA_DIR = _storage_dir("rca", _WRITABLE / "rca")        # 생성 리포트/이미지 보관


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:120] or "wafer"


def _rca_h_artifacts() -> Optional[Path]:
    """H 센서 복원 모델(별도 ONNX) 아티팩트 폴더. 설정(rca_h_artifacts) → 모델 옆 artifacts_h → 없으면 None."""
    cfg = _load_app_config()
    p = cfg.get("rca_h_artifacts")
    if p:
        pp = _path_load(p)
        if pp.exists():
            return pp
        _note_stale_path("rca_h_artifacts", str(p))      # 이전 PC 경로 → 자동탐색으로 대체
    cand = Path(ENGINE.art).parent / "artifacts_h"
    return cand if cand.exists() else None


def _find_wafer_source(wid: str) -> Optional[Path]:
    """웨이퍼 원본 .log/.tsv/.csv 위치 탐색(RCA 는 밸브/IO 컬럼이 필요 → 아카이브 npz 로는 불가)."""
    rec = STATE["recs"].get(wid)
    if rec and rec.get("source_path") and Path(rec["source_path"]).exists():
        return Path(rec["source_path"])
    # ★ 백업 폴더 포함: 일마감(rollover)이 지난 날짜의 원본 .log 를 데이터\백업\<날짜>\<챔버>\ 로
    #   이동시키므로, 오래된 웨이퍼의 RCA 는 백업에서 찾아야 한다.
    #   (기간을 길게 잡아 옛 웨이퍼를 고르면 '원본 없음'으로 실패하던 문제 수정)
    dirs = ([Path(STATE.get("folder") or ""), _inbox()]
            + list(_chamber_paths().values()) + [BACKUP_DIR])
    seen = set()
    for d in dirs:
        try:
            d = Path(d)
        except Exception:                                # noqa: BLE001
            continue
        if not d or str(d) in seen or not d.exists():
            continue
        seen.add(str(d))
        for ext in (".log", ".tsv", ".csv"):
            f = d / f"{wid}{ext}"
            if f.exists():
                return f
        # 하위폴더까지(챔버 루트/백업에 하위 날짜폴더가 있을 수 있음)
        for ext in (".log", ".tsv", ".csv"):
            hit = next(iter(d.rglob(f"{wid}{ext}")), None)
            if hit:
                return hit
        # ★ 일마감 이동 때 동명 충돌로 '<wid>__<타임스탬프>.log' 로 개명된 백업 파일도 탐색
        #   (정확일치만 찾으면 개명본을 놓쳐 '원본 없음' 이 나던 허점)
        for ext in (".log", ".tsv", ".csv"):
            hit = next(iter(d.rglob(f"{wid}__*{ext}")), None)
            if hit:
                return hit
    return None


def _run_one_rca(mode: str, pt_art: Path, h_art: Path, src: Path, wid: str):
    """RCA 한 종류 실행 → {report_md, image_url} (실패해도 예외 대신 error 반환)."""
    from urllib.parse import quote
    import os as _os
    _os.environ.setdefault("MPLBACKEND", "Agg")
    from src.config import IOConfig
    out_dir = RCA_DIR / _short_name(wid) / ("cf" if mode == "h-counterfactual" else "slope")
    try:
        with _PLOT_LOCK:
            if mode == "h-counterfactual":
                from src import h_counterfactual_rca as HRCA
                HRCA._write_counterfactual_plot = lambda *a, **k: None   # matplotlib 스킵(앱 SVG 로 표시)
                HRCA.generate_h_counterfactual_rca(
                    pt_artifacts=pt_art, h_artifacts=h_art, data_dir=src.parent,
                    out_dir=out_dir, cfg=IOConfig(), wafer_id=wid,
                    prediction_filter="all", exclude_steps={0})
                reps = sorted(out_dir.rglob("*_h_counterfactual.md"))
            else:
                from src import rca_slope_valve as RSV
                RSV._plot_image = lambda *a, **k: None   # matplotlib 스킵(앱 SVG 로 표시 → 빠름·폰트문제 제거)
                RSV.generate_rca_slope_valve_reports(
                    pt_artifacts=pt_art, h_artifacts=h_art, data_dir=src.parent,
                    out_dir=out_dir, cfg=IOConfig(), wafer_id=wid,
                    prediction_filter="all", exclude_steps={0},
                    include_h_roles=True, h_role_top_k=2)
                reps = sorted(out_dir.glob("slope_valve_reports/*.md"))
    except Exception as e:                                # noqa: BLE001
        traceback.print_exc()
        # ★ 진단 강화: 예외 타입 + 파일경로(FileNotFoundError 의 filename) + 발생 위치까지 표시
        #   ("[Errno 2] ..." 만으로는 어떤 파일이 없는지 알 수 없던 문제)
        loc = ""
        try:
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                fr = tb[-1]
                loc = f" @ {Path(fr.filename).name}:{fr.lineno}"
        except Exception:                                # noqa: BLE001
            pass
        fn = getattr(e, "filename", None)
        return {"error": f"{type(e).__name__}: {e}" + (f" (파일: {fn})" if fn else "") + loc}
    if not reps:
        return {"error": "리포트가 생성되지 않았습니다."}
    out = {"report_md": reps[0].read_text(encoding="utf-8")}
    # ★ 앱 스타일 SVG + 요약(원인) 구조화 데이터.
    if mode != "h-counterfactual":
        try:
            import csv as _csv
            ev = out_dir / "rca_slope_valve_events.csv"
            cols = [str(c) for c in ENGINE.model.sensor_cols]
            idx_of = {c: i for i, c in enumerate(cols)}
            with open(ev, encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    if str(row.get("wafer_id")) != str(wid):
                        continue
                    out["focus_step"] = int(float(row.get("peak_step") or 0))
                    names = [s for s in str(row.get("related_sensors") or "").split(";") if s]
                    out["related"] = [{"index": idx_of[n], "sensor": n}
                                      for n in names if n in idx_of]
                    break
        except Exception:                                 # noqa: BLE001
            pass
        # 리포트 '판정 요약' 표(항목→내용) 파싱 → 원인/불량유형 요약용
        kv = {}
        for line in out["report_md"].splitlines():
            if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-", ":", " "}:
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 2 and parts[0] not in ("항목",):
                kv[parts[0]] = " ".join(parts[1:-1]) if len(parts) > 2 else parts[1]
        out["summary_kv"] = kv
    else:
        # 반사실: 제어변수별 개선율에서 '가장 개선되는' 추천 1건 추출
        try:
            import csv as _csv
            best = None
            for cand in out_dir.rglob("*summary*.csv"):
                for row in _csv.DictReader(open(cand, encoding="utf-8")):
                    if str(row.get("wafer_id")) != str(wid):
                        continue
                    try:
                        pct = float(row.get("pt_improvement_pct") or 0)
                    except (TypeError, ValueError):
                        pct = 0.0
                    if best is None or pct > best["pct"]:
                        best = {"pct": pct, "control": row.get("control", ""),
                                "step": row.get("recommended_step", ""),
                                "delta": row.get("recommended_delta", ""),
                                "value": row.get("recommended_value", "")}
                break
            out["best_cf"] = best
        except Exception:                                 # noqa: BLE001
            pass
    return out


@app.post("/api/rca/{date}/{wid}")
@_wafer_model
def api_rca_generate(date: str, wid: str, force: bool = False):
    """불량(NG) 웨이퍼 RCA 생성. slope-밸브 + 반사실(h-counterfactual) 둘 다 자동 실행.
    ★ 결과는 저장(result.json)되어 재클릭 시 즉시 반환 — force=true 면 다시 분석."""
    ENGINE.load()
    if not ENGINE.ready:
        raise HTTPException(503, "모델 미로드")
    _res_path = RCA_DIR / _short_name(wid) / "result.json"
    if not force and _res_path.exists():                  # 저장된 결과 즉시 반환(재분석 불필요)
        try:
            saved = json.loads(_res_path.read_text(encoding="utf-8"))
            if saved.get("model_id") == _model_id():
                saved["cached"] = True
                return saved
        except Exception:                                # noqa: BLE001 — 손상 시 재생성
            pass
    arch = _load_archive(date, wid)
    # ★ NG 게이트는 '현재 표시 판정'(배율·배점 모드 반영된 relive) 기준으로.
    #   아카이브 meta 의 verdict 는 검사 당시 raw 판정이라, 배점 모드에서 NG 로 표시되는
    #   웨이퍼(raw 는 OK)의 RCA 가 "NG 만 지원" 400 으로 거부되던 회귀 수정.
    verdict = None
    try:
        for r in reversed(_history_for_current_model()):   # 최신 행 우선
            if r.get("wafer_id") == wid:
                verdict = r.get("verdict"); break
    except Exception:                                     # noqa: BLE001
        verdict = None
    if verdict is None and arch:
        verdict = arch["meta"].get("verdict")             # 이력에 없으면 아카이브 폴백
    if verdict is not None and verdict != "NG":
        raise HTTPException(400, "RCA 는 불량(NG) 웨이퍼만 지원합니다. "
                                 "(현재 모드 기준 이 웨이퍼는 OK 판정입니다)")
    src = _find_wafer_source(wid)
    _tmp_dir = None
    if src is None:
        # ★ 원본이 없으면 아카이브(npz)의 RCA용 원본(IO+입력센서)으로 임시 CSV 복원 → RCA 가능
        src = _rca_source_from_archive(date, wid)
        if src is not None:
            _tmp_dir = Path(src).parent
    if src is None:
        raise HTTPException(404, "원본 .log/.csv 를 찾지 못했습니다 — 검사폴더·DSV 챔버폴더·백업폴더와 "
                                 "아카이브(npz) RCA 원본까지 모두 확인했습니다. "
                                 "(이 웨이퍼는 'RCA 원본 보존' 기능 이전에 검사된 과거분이라 npz 에도 "
                                 "IO 가 없습니다 — 원본을 다시 넣거나 다시 검사하면 가능합니다)")
    pt_art = Path(ENGINE.art)
    h_art = _rca_h_artifacts() or pt_art
    # ★ 동시 1건 + 벽시계 타임아웃. 예전에는 무제한 실행이라 오래 걸리면 화면은
    #   '분석 중…' 그대로, 서버는 CPU 만 태웠다(사용자가 재클릭하면 겹쳐 더 악화).
    #   진행 중이면 즉시 거절하고, 시간 초과면 실패로 응답한다(작업 스레드는 끝나는
    #   대로 busy 를 풀므로 그동안 새 분석이 겹치지 않는다).
    with _RCA_STATE_LOCK:
        if _RCA_STATE["busy"]:
            raise HTTPException(429, "다른 RCA 분석이 아직 진행 중입니다 — 잠시 후 다시 시도하세요")
        _RCA_STATE["busy"] = True

    chamber_spec = ENGINE.artifact
    def _rca_work():
        try:
            def run_pair(model_dir):
                h_dir = model_dir if h_art == pt_art else h_art
                s = _run_one_rca("slope-valve", model_dir, h_dir, src, wid)
                c = _run_one_rca("h-counterfactual", model_dir, h_dir, src, wid)
                return s, c
            if chamber_spec:
                with canonical_artifacts(chamber_spec) as model_dir:
                    return run_pair(model_dir)
            return run_pair(pt_art)
        finally:
            if _tmp_dir is not None:                      # 복원용 임시 CSV 정리
                shutil.rmtree(_tmp_dir, ignore_errors=True)
            with _RCA_STATE_LOCK:
                _RCA_STATE["busy"] = False

    fut = _RCA_POOL.submit(copy_context().run, _rca_work)
    try:
        slope, cf = fut.result(timeout=_RCA_TIMEOUT_S)
    except FuturesTimeout:
        print(f"[rca] 시간 초과({_RCA_TIMEOUT_S}s) — wid={wid}", flush=True)
        raise HTTPException(504, f"RCA 분석이 {_RCA_TIMEOUT_S}초 안에 끝나지 않아 중단 응답합니다. "
                                 "잠시 후 다시 시도하거나, CPU 사용 제한 설정을 확인하세요")
    if "error" in slope and "error" in cf:               # 둘 다 실패면 에러
        raise HTTPException(500, f"RCA 실패: slope={slope['error']} · cf={cf['error']}")
    # ── 공정 엔지니어용 3항목 요약: 원인 · 불량 유형 · 반사실적 설명 ──
    kv = slope.get("summary_kv") or {}
    first = (kv.get("주요 센서", "").split(",")[0] or "").strip()
    fam = ("온도" if first.upper().startswith("T") else "압력" if first.upper().startswith("P")
           else "히터/H" if (first.upper().startswith("H") or "Heater" in first) else "센서")
    dirn = kv.get("방향", "")
    dtxt = "초과(실제>기대)" if ">" in dirn else "미달(실제<기대)" if "<" in dirn else "-"
    cause = " · ".join([x for x in [
        (f"이상 집중 step {kv.get('Peak step')}" if kv.get("Peak step") else ""),
        (f"주요 센서 {kv.get('주요 센서')} ({dirn})" if kv.get("주요 센서") else ""),
        (f"의심 밸브 {kv.get('의심 밸브')}" if kv.get("의심 밸브") else ""),
        (f"1차 확인: {kv.get('1차 확인')}" if kv.get("1차 확인") else ""),
    ] if x]) or "-"
    defect_type = (f"{fam} {dtxt}" + (f" · {kv.get('1차 확인')}" if kv.get("1차 확인") else "")) if first else "-"
    best = cf.get("best_cf")
    if best and best.get("pct", 0) > 0.5:
        cf_text = (f"step {best['step']} 에서 {best['control']} 를 {best['delta']} 조정 시 "
                   f"P/T 복원오차 약 {best['pct']:.1f}% 개선 (조정 후 값 {best['value']})")
    else:
        cf_text = "H 제어변수(h_005) 조정만으로는 유의미한 개선 없음 → 밸브/유로 경로 점검 권장"
    summary = {"cause": cause, "defect_type": defect_type, "counterfactual": cf_text,
               "focus_step": slope.get("focus_step"),
               "top_sensors": (kv.get("주요 센서", "").split(",") if kv.get("주요 센서") else [])}
    resp = {"wafer_id": wid, "model_id": _model_id(), "summary": summary, "slope": slope, "counterfactual": cf}
    try:                                                  # ★ 결과 저장 → 다음 클릭은 즉시 표시
        _res_path.parent.mkdir(parents=True, exist_ok=True)
        _res_path.write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        traceback.print_exc()
    return resp


@app.get("/api/rca/{wid}/image")
def api_rca_image(wid: str, mode: str = "slope-valve"):
    """생성된 RCA 오버레이 이미지 반환(mode: slope-valve | h-counterfactual)."""
    base = RCA_DIR / _short_name(wid) / ("cf" if mode == "h-counterfactual" else "slope")
    imgs = sorted(base.rglob("*.png")) if base.exists() else []
    if not imgs:
        raise HTTPException(404, "RCA 이미지 없음")
    return FileResponse(imgs[0], media_type="image/png")


class HModelReq(BaseModel):
    dir: str = ""


@app.get("/api/rca/h-model")
def api_rca_h_model_get():
    """RCA 용 H 복원 모델(별도 ONNX) 현재 설정·자동탐색 상태."""
    cfg = _load_app_config()
    conf = cfg.get("rca_h_artifacts") or ""
    resolved = _rca_h_artifacts()
    avail = False
    try:
        from src.onnx_backend_forecast import OnnxForecastModel
        avail = bool(resolved and OnnxForecastModel.available(Path(resolved)))
    except Exception:                                     # noqa: BLE001
        pass
    if conf and Path(conf).exists():
        source = "지정 경로"
    elif (Path(ENGINE.art).parent / "artifacts_h").exists():
        source = "자동(모델 폴더 옆 artifacts_h)"
    else:
        source = "없음 → P/T 모델로 대체(H 역할 제한)"
    return {"path": conf, "resolved": (str(resolved) if resolved else ""),
            "available": avail, "source": source, "pt_dir": str(ENGINE.art)}


@app.post("/api/rca/h-model")
def api_rca_h_model_set(req: HModelReq):
    """RCA 용 H 복원 모델 아티팩트 폴더 지정(빈 값이면 해제 → 자동탐색)."""
    d = (req.dir or "").strip()
    cfg = _load_app_config()
    if not d:
        cfg.pop("rca_h_artifacts", None)
        _save_app_config(cfg)
        return {"ok": True, "path": "", "cleared": True}
    p = Path(d)
    if not p.exists():
        raise HTTPException(400, f"경로가 없습니다: {d}")
    try:
        from src.onnx_backend_forecast import OnnxForecastModel
        ok = OnnxForecastModel.available(p)
    except Exception:                                     # noqa: BLE001
        ok = False
    if not ok and not any(p.glob("*.onnx")):
        raise HTTPException(400, "그 폴더에 H 복원 ONNX 모델(.onnx + 메타 .npz)이 없습니다.")
    cfg["rca_h_artifacts"] = _path_store(p)
    _save_app_config(cfg)
    return {"ok": True, "path": str(p), "available": ok}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ════════════════════════════════════════════════════════════════════
#  학습(Train) — 우분투 내부용. train_jw 서브패키지를 GPU별 subprocess 로 실행.
#  · 설계 원칙: GPU/RAM 완전 반납은 '프로세스 종료'로 보장(앱 프로세스는 torch 미로드).
#  · 진행 보고: 학습 로그(log.txt)를 tail 파싱 → epoch/val_mse 곡선 + trial 아키텍처.
#  · 멀티 GPU: GPU 당 1작업, 점유 GPU 요청은 대기 큐 → 반납 시 자동 시작.
#  · 앱 재시작: job.json + pid 로 실행중 학습에 재연결(학습은 앱과 독립적으로 계속).
# ════════════════════════════════════════════════════════════════════
TRAIN_DIR = _storage_dir("train", DATA_DIR / "train_jobs")   # 학습 작업(로그·아티팩트)
#   ★ --save-trial-artifacts 를 쓰면 trial 마다 모델이 쌓여 수 GB 가 되므로 별도 디스크 지정 가능
_TRAIN_MIN_FREE_RAM_MB = int(os.environ.get("IO_TRAIN_MIN_RAM_MB", "4096"))
_TRAIN_MIN_FREE_VRAM_MB = int(os.environ.get("IO_TRAIN_MIN_VRAM_MB", "2048"))


def _tag_data_files_direct(files) -> int:
    """★ 파일 경로에서 직접 끝시각(+.log 는 챔버/레시피) 태깅 — .csv 도 포함.

    `_convert_logs_in` 은 .log/.tsv 만 태깅해서, **.csv 로 된 학습 폴더는 end_ts 가 비고**
    기간 필터가 전부 걸러내 '0매'가 되는 문제가 있었다(학습 시작 실패의 원인).
    여기서는 확장자와 무관하게 파일에서 직접 뽑고, 하위폴더 파일도 경로 그대로 처리한다.
    반환 = 이번에 새로 태깅한 개수."""
    todo = [f for f in (files or []) if not _file_meta(f.stem).get("end_ts")
            or _chamber_mapping_stale(f.stem)]
    if not todo:
        return 0
    if not _FILE_META:
        _load_file_meta()
    if not _RECIPE_TAGS:
        _load_recipe_tags()

    def _one(f: Path):
        _pause_gate()
        is_log = f.suffix.lower() in LOG_EXTS
        try:
            edt = _extract_end_dt_from_file(f)
            ch = _extract_chamber_from_log(f) if is_log else (_chamber_from_token(f.stem, f.stem) if _is_wuk380b(f.stem) else None)
            rc = _extract_recipe_from_log(f) if is_log else None
            return (f, edt.isoformat(timespec="seconds") if edt else None, ch,
                    _clean_recipe(rc) if rc else None)
        except Exception:                                 # noqa: BLE001
            return (f, None, None, None)

    try:
        _w = max(1, min(int(os.environ.get("IO_LOAD_WORKERS") or 0) or _cpu_workers(8),
                        len(todo)))
    except (TypeError, ValueError):
        _w = 1
    if _w == 1 or len(todo) == 1:
        res = [_one(f) for f in todo]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=_w) as ex:
            res = list(ex.map(_one, todo))
    n = 0
    with _FILE_META_LOCK:
        for f, ets, ch, _rc in res:
            cur = dict(_FILE_META.get(str(f.stem), {}))
            if ets:
                cur["end_ts"] = ets; cur["ets_v"] = 2; n += 1
            if ch or _is_wuk380b(f.stem) or _chamber_mapping_stale(f.stem):
                cur["chamber"] = ch or ""
                cur["chamber_cell"] = _chamber_cell_tag(f.stem)
            if cur:
                _FILE_META[str(f.stem)] = cur
    with _RECIPE_TAGS_LOCK:
        for f, _e, _c, rc in res:
            if rc:
                _RECIPE_TAGS[str(f.stem)] = rc
    _flush_file_meta()
    _flush_recipe_tags()
    print(f"[tag] 직접 태깅 {n}/{len(todo)}매(.csv 포함)", flush=True)
    return n


def _train_ok_dir(base: Path) -> tuple:
    """학습 base 폴더 → (OK 폴더, 제외할 하위폴더들, 레이아웃 설명).

    · OK/NG/VAL 은 **대소문자 무시**로 찾는다(리눅스는 case-sensitive 라 'ok' 폴더를 놓치면
      NG/VAL 웨이퍼까지 OK 로 학습되는 심각한 오류가 난다).
    · OK 하위폴더가 없으면 base 전체를 OK 로 보되, NG/VAL 하위트리는 반드시 제외한다."""
    subs = {p.name.lower(): p for p in base.iterdir() if p.is_dir()} if base.is_dir() else {}
    okd = subs.get("ok")
    excl = [p for k, p in subs.items() if k in ("ng", "val")]
    if okd is not None:
        return okd, [], f"{okd.name}/NG/VAL 구조"
    return base, excl, ("단일 폴더(전체=OK)" + (f" · {[p.name for p in excl]} 제외" if excl else ""))


def _train_files(base: Path) -> tuple:
    """학습 대상 파일 목록(OK 폴더 기준, NG/VAL 제외) + OK 폴더 + 레이아웃."""
    okd, excl, layout = _train_ok_dir(base)
    files = _data_files(okd)
    if excl:
        exr = [str(p.resolve()) for p in excl]
        files = [f for f in files
                 if not any(str(f.resolve()).startswith(e + os.sep) for e in exr)]
    return files, okd, layout


def _train_supported() -> Dict:
    """학습탭 지원 여부 — 리눅스 + torch + train_jw 가 있을 때만(윈도우 제어기엔 자동 숨김)."""
    if not sys.platform.startswith("linux"):
        return {"ok": False, "reason": "리눅스 전용(윈도우 제어기 앱에서는 비활성)"}
    import importlib.util as _il
    if _il.find_spec("torch") is None:
        return {"ok": False, "reason": "torch 미설치"}
    if not (PKG / "train_jw" / "run.py").exists():
        return {"ok": False, "reason": "train_jw 학습 패키지 없음"}
    return {"ok": True}


def _train_gpus() -> List[Dict]:
    """nvidia-smi 로 GPU 목록/메모리/사용률 조회(없으면 빈 목록)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        gpus = []
        for ln in out.strip().splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) >= 5:
                gpus.append({"index": int(p[0]), "name": p[1], "mem_total_mb": int(p[2]),
                             "mem_used_mb": int(p[3]), "util_pct": int(p[4])})
        return gpus
    except Exception:                                     # noqa: BLE001
        return []


def _free_ram_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) // 1024
    except Exception:                                     # noqa: BLE001
        pass
    return 1 << 20                                        # 알 수 없으면 통과


_EPOCH_RE = re.compile(r"\[tcn\] epoch (\d+)/(\d+) val_mse=([0-9.eE+-]+)"
                       r"(?:\s+pid=(\d+))?(?:\s+dev=(\S+))?")
# 멀티 GPU: 워커가 어떤 GPU 에서 어떤 trial 을 시작했는지 → pid 로 epoch 을 귀속시킨다.
_WORKER_RE = re.compile(r"\[worker pid=(\d+) gpu=(\S+)\] start trial (\d+)")
_FIT_RE = re.compile(r"\[tcn\] past=(\d+) future=(\d+) stride=(\d+) \| input (\d+)ch -> target (\d+)ch")
_TRIAL_RE = re.compile(r"\[\s*(\d+)/(\d+)\] val_mse=([0-9.eE+-]+).*?"
                       r"w(\d+) h(\d+) s(\d+).*?ch(\d+) blk(\d+) k(\d+)")


class TrainManager:
    """학습 작업 오케스트레이터 — 시작/큐/일시정지/중단/진행파싱/완료시 ONNX export."""

    def __init__(self):
        self.jobs: Dict[str, Dict] = {}                  # id → job(dict, 실행 필드는 _rt에)
        self.lock = threading.Lock()
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        self._load_existing()
        threading.Thread(target=self._monitor_loop, daemon=True, name="train-monitor").start()

    # ── 영속화 ──
    def _jdir(self, jid: str) -> Path:
        return TRAIN_DIR / jid

    def _save(self, job: Dict):
        try:
            d = {k: v for k, v in job.items() if not k.startswith("_")}
            p = self._jdir(job["id"]) / "job.json"
            tmp = p.with_name("job.json.tmp")
            tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, p)
        except Exception:                                 # noqa: BLE001
            traceback.print_exc()

    def _load_existing(self):
        for jd in sorted(TRAIN_DIR.glob("job_*")):
            try:
                job = json.loads((jd / "job.json").read_text(encoding="utf-8"))
            except Exception:                             # noqa: BLE001
                continue
            if job.get("status") == "finalizing":          # 마무리 중 앱이 꺼짐 → 재개 불가
                job["status"] = "failed"
                job["note"] = "마무리(ONNX/밴드) 중 앱이 종료됨 — 출력폴더 확인 후 재학습 필요"
            if job.get("status") in ("running", "paused"):
                pid = int(job.get("pid") or 0)
                if pid > 0 and self._alive(pid):
                    job["_reattached"] = True             # 앱 재시작 → 실행중 학습 재연결
                else:
                    job["status"] = "interrupted"
                    job["note"] = "앱 재시작 시 프로세스 없음(중단됨)"
            self.jobs[job["id"]] = job
            if job.get("status") in ("interrupted", "failed"):
                self._save(job)

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    # ── 시작/큐 ──
    def start(self, spec: Dict) -> Dict:
        sup = _train_supported()
        if not sup["ok"]:
            raise HTTPException(400, f"학습 미지원: {sup['reason']}")
        folder = Path(spec["folder"]).expanduser()
        if not folder.is_dir():
            raise HTTPException(404, f"학습 폴더 없음: {folder}")
        out_dir = Path(spec["out_dir"]).expanduser()
        gpus = _train_gpus()
        # ★ 멀티 GPU: gpus 리스트(예 [0,1]) — run.py 가 CUDA_VISIBLE_DEVICES 를 보고
        #   GPU당 1워커를 자동으로 띄워 trial 을 병렬 처리한다(작업 1개 = GPU 여러 장).
        glist = spec.get("gpus") or [int(spec.get("gpu", 0))]
        glist = sorted({int(x) for x in glist})
        if gpus:
            have = [g["index"] for g in gpus]
            miss = [g for g in glist if g not in have]
            if miss:
                raise HTTPException(400, f"GPU {miss} 없음 (감지: {have})")
        gidx = glist[0]
        fixed = spec.get("fixed_params") or None          # ★ 고정 학습: 단일값 search space 주입
        with self.lock:                                   # 같은 초 시작 대비 일련번호(ID 충돌 방지)
            self._seq = int(getattr(self, "_seq", 0)) + 1
            seq = self._seq
        jid = ("job_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{seq:03d}"
               + "_g" + "-".join(str(g) for g in glist))
        jdir = self._jdir(jid)
        (jdir).mkdir(parents=True, exist_ok=True)
        opts = self._validated_opts(spec)                 # ★ tcn-search 전체 옵션(README 기준) + 상호검증
        job = {"id": jid, "folder": str(folder), "out_dir": str(out_dir), "gpu": gidx,
               "gpus": glist, "fixed_params": fixed,
               "trials": (1 if fixed else int(spec.get("trials", 12))), "epochs": spec.get("epochs"),
               "seed": int(spec.get("seed", 42)),
               "h_cart_mode": opts["h_cart_mode"],
               "file_format": str(spec.get("file_format") or "auto"),
               "opts": opts,
               "tool_prefixes": spec.get("tool_prefixes") or [],
               "chamber_prefixes": spec.get("chamber_prefixes") or [],
               "recipe_prefixes": spec.get("recipe_prefixes") or [],
               "date_from": spec.get("date_from") or "", "date_to": spec.get("date_to") or "",
               "status": "queued", "created": _now(), "pid": 0,
               "progress": [], "trials_done": [], "n_files": None, "note": ""}
        warns = []
        busy = self._gpu_busy_map()
        for g_ in glist:
            if g_ in busy:
                warns.append(f"GPU {g_}는 작업 {busy[g_]} 학습 중 — 대기 큐에 넣고 반납 시 자동 시작")
        for g in gpus:
            if g["index"] in glist and (g["mem_total_mb"] - g["mem_used_mb"]) < _TRAIN_MIN_FREE_VRAM_MB:
                warns.append(f"GPU {g['index']} VRAM 여유 {g['mem_total_mb']-g['mem_used_mb']}MB — 부족하면 대기")
        if _free_ram_mb() < _TRAIN_MIN_FREE_RAM_MB:
            warns.append(f"시스템 RAM 여유 {_free_ram_mb()}MB < {_TRAIN_MIN_FREE_RAM_MB}MB — 확보 후 자동 시작")
        job["note"] = " · ".join(warns)
        with self.lock:
            self.jobs[jid] = job
        self._save(job)
        self._try_start_queued()                          # 여유 있으면 즉시 시작
        return {"id": jid, "status": job["status"], "warnings": warns}

    @staticmethod
    def _validated_opts(spec: Dict) -> Dict:
        """tcn-search 의 모든 CLI 옵션(README 표)을 받아 정규화 + run.py 와 동일한 상호검증.

        검증을 여기서 하는 이유: 학습 프로세스를 띄운 뒤 argparse 가 죽으면 사용자는
        '실패'만 보게 되므로, 시작 전에 UI 로 원인을 돌려준다."""
        def _s(k, default, choices=None):
            v = str(spec.get(k) or default).strip()
            if choices and v not in choices:
                raise HTTPException(400, f"{k}: '{v}' 는 허용되지 않음 (가능: {', '.join(choices)})")
            return v

        def _b(k, default=False):
            v = spec.get(k)
            return default if v is None else bool(v)

        def _i(k, default=None):
            v = spec.get(k)
            if v in (None, ""):
                return default
            try:
                return int(float(v))
            except (TypeError, ValueError):
                raise HTTPException(400, f"{k}: 정수가 아님 ({v})")

        def _f(k, default=None):
            v = spec.get(k)
            if v in (None, ""):
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"{k}: 숫자가 아님 ({v})")

        o = {
            "h_cart_mode": _s("h_cart_mode", "include", ["include", "exclude"]),
            "sensor_role_profile": _s("sensor_role_profile", "original", ["original", "h-split-v1"]),
            "h_sensor_preset": _s("h_sensor_preset", "all", ["all", "heater-block3", "heater-12"]),
            "future_h_mode": _s("future_h_mode", "input", ["input", "predict"]),
            "virtual_sensor_mode": _s("virtual_sensor_mode", "off",
                                      ["off", "input", "target", "both"]),
            "virtual_sensors": _b("virtual_sensors"),
            "search_objective": _s("search_objective", "correct", ["correct", "precision"]),
            # ★ v0.96 점수 정규화: 누적 점수를 '유효 time-step 수'로 나눠 레시피 길이 영향을 제거.
            #   단일 선택이라 --score-time-avg 와 --score-final-avg 동시 지정(에러)이 원천 차단된다.
            "score_norm": _s("score_norm", "none", ["none", "time", "final"]),
            "trial_artifact_dir": str(spec.get("trial_artifact_dir") or "").strip(),
            "threshold_wise": _s("threshold_wise", "recipe", ["all", "recipe", "sensor", "step"]),
            "threshold_source": _s("threshold_source", "val-search",
                                   ["val-search", "train-ok-sigma"]),
            "threshold_sigma": _f("threshold_sigma", -1.0),
            "threshold_sigma_min": _f("threshold_sigma_min"),
            "threshold_sigma_max": _f("threshold_sigma_max"),
            "threshold_recipe_multipliers": str(spec.get("threshold_recipe_multipliers") or "").strip(),
            "resid_agg": _s("resid_agg", "", ["", "sum", "p99", "max", "mean"]),
            "wafer_agg": _s("wafer_agg", "", ["", "sum", "topk_mean", "max", "mean"]),
            "topk": _i("topk"),
            "post_eval_extend_epochs": _i("post_eval_extend_epochs", 10),
            "post_eval_max_extends": _i("post_eval_max_extends", 0),
            "n_workers": _i("n_workers"),
            "checkpoint": _b("checkpoint", True),          # 기본 ON(README 기본) — best 즉시 저장
            "no_train": _b("no_train"),
            "resume": _b("resume"),
            "save_trial_artifacts": _b("save_trial_artifacts"),
            "stop_on_perfect_val": _b("stop_on_perfect_val"),
        }
        # ── run.py 와 동일한 상호검증 ──
        if o["sensor_role_profile"] == "h-split-v1":
            if o["h_cart_mode"] != "include":
                raise HTTPException(400, "h-split-v1 은 Cart류 10종을 미래 입력으로 쓰므로 "
                                         "H-Cart '포함(include)' 이어야 합니다")
            if o["h_sensor_preset"] != "all":
                raise HTTPException(400, "h-split-v1 은 H 역할을 직접 나누므로 "
                                         "H 센서 preset 은 'all' 이어야 합니다")
            if o["future_h_mode"] != "input":
                raise HTTPException(400, "h-split-v1 은 미래 H 입력 프로파일이므로 "
                                         "future-h-mode 는 'input' 이어야 합니다")
        rng = (o["threshold_sigma_min"] is not None) or (o["threshold_sigma_max"] is not None)
        if rng and float(o["threshold_sigma"]) >= 0:
            raise HTTPException(400, "sigma 하한/상한은 threshold-sigma = -1(자동 탐색)일 때만 "
                                     "사용할 수 있습니다 (고정값이면 범위를 비우세요)")
        for k in ("threshold_sigma_min", "threshold_sigma_max"):
            if o[k] is not None and o[k] < 0:
                raise HTTPException(400, f"{k} 는 0 이상이어야 합니다")
        if (o["threshold_sigma_min"] is not None and o["threshold_sigma_max"] is not None
                and o["threshold_sigma_min"] > o["threshold_sigma_max"]):
            raise HTTPException(400, "sigma 하한이 상한보다 큽니다")
        for tok in [t for t in re.split(r"[,\s]+", o["threshold_recipe_multipliers"]) if t]:
            if "=" not in tok:
                raise HTTPException(400, f"레시피 배율 형식 오류: '{tok}' (예: 29a=1.1,336=1.2)")
            try:
                float(tok.split("=", 1)[1])
            except ValueError:
                raise HTTPException(400, f"레시피 배율 값이 숫자가 아님: '{tok}'")
        return o

    @staticmethod
    def _opts_to_args(o: Dict) -> List[str]:
        """정규화된 옵션 → run.py tcn-search 인자 목록(기본값과 같으면 생략)."""
        a: List[str] = ["--h-cart-mode", o["h_cart_mode"]]
        if o["sensor_role_profile"] != "original":
            a += ["--sensor-role-profile", o["sensor_role_profile"]]
        if o["h_sensor_preset"] != "all":
            a += ["--h-sensor-preset", o["h_sensor_preset"]]
        if o["future_h_mode"] != "input":
            a += ["--future-h-mode", o["future_h_mode"]]
        if o["virtual_sensors"]:
            a += ["--virtual-sensors"]                     # both 로 강제(모드값은 무시됨)
        elif o["virtual_sensor_mode"] != "off":
            a += ["--virtual-sensor-mode", o["virtual_sensor_mode"]]
        if o["search_objective"] != "correct":
            a += ["--search-objective", o["search_objective"]]
        if o.get("score_norm") == "time":
            a += ["--score-time-avg"]
        elif o.get("score_norm") == "final":
            a += ["--score-final-avg"]
        if o["threshold_wise"] != "recipe":
            a += ["--threshold-wise", o["threshold_wise"]]
        if o["threshold_source"] != "val-search":
            a += ["--threshold-source", o["threshold_source"]]
            if float(o["threshold_sigma"]) != -1.0:        # sigma 는 train-ok-sigma 에서만 의미
                a += ["--threshold-sigma", str(o["threshold_sigma"])]
            if o["threshold_sigma_min"] is not None:
                a += ["--threshold-sigma-min", str(o["threshold_sigma_min"])]
            if o["threshold_sigma_max"] is not None:
                a += ["--threshold-sigma-max", str(o["threshold_sigma_max"])]
        for tok in [t for t in re.split(r"[,\s]+", o["threshold_recipe_multipliers"]) if t]:
            a += ["--threshold-recipe-multiplier", tok]
        if o["resid_agg"]:
            a += ["--resid-agg", o["resid_agg"]]
        if o["wafer_agg"]:
            a += ["--wafer-agg", o["wafer_agg"]]
        if o["topk"] is not None:
            a += ["--topk", str(o["topk"])]
        if int(o["post_eval_extend_epochs"]) != 10:
            a += ["--post-eval-extend-epochs", str(int(o["post_eval_extend_epochs"]))]
        if int(o["post_eval_max_extends"]) != 0:
            a += ["--post-eval-max-extends", str(int(o["post_eval_max_extends"]))]
        if not o["checkpoint"]:
            a += ["--no-checkpoint"]
        if o["no_train"]:
            a += ["--no-train"]
        if o["resume"]:
            a += ["--resume"]
        if o["save_trial_artifacts"]:
            a += ["--save-trial-artifacts"]
        if o["stop_on_perfect_val"]:
            a += ["--stop-on-perfect-val"]
        return a

    def _gpu_busy_map(self) -> Dict[int, str]:
        with self.lock:
            out: Dict[int, str] = {}
            for j in self.jobs.values():
                # finalizing 도 GPU 를 쓴다(OK 누적밴드 계산이 모델을 GPU 에 올림) → 점유로 계산
                if j.get("status") in ("running", "paused", "finalizing"):
                    for g in (j.get("gpus") or [j.get("gpu", 0)]):
                        out[int(g)] = j["id"]
            return out

    def _try_start_queued(self):
        with self.lock:
            queued = sorted([j for j in self.jobs.values() if j.get("status") == "queued"],
                            key=lambda j: j["id"])
        busy = self._gpu_busy_map()
        for job in queued:
            need = [int(g) for g in (job.get("gpus") or [job.get("gpu", 0)])]
            if any(g in busy for g in need):
                continue                                  # 요청 GPU 중 점유분 존재 → 대기 유지
            if _free_ram_mb() < _TRAIN_MIN_FREE_RAM_MB:
                continue                                  # RAM 부족 → 대기 유지
            gs = _train_gpus()
            if any((g["mem_total_mb"] - g["mem_used_mb"]) < _TRAIN_MIN_FREE_VRAM_MB
                   for g in gs if g["index"] in need):
                continue                                  # VRAM 부족(다른 사용자 프로세스) → 대기
            try:
                self._launch(job)
                for g_ in need:
                    busy[g_] = job["id"]
            except Exception as e:                        # noqa: BLE001
                job["status"] = "failed"; job["note"] = f"시작 실패: {e}"
                self._save(job)
                traceback.print_exc()

    # ── 데이터 준비: metadata 필터로 고른 파일만 심볼릭링크 (원본 무접촉) ──
    def _prepare_data(self, job: Dict) -> Path:
        base = Path(job["folder"])
        files, okd, _layout = _train_files(base)          # OK 폴더(대소문자 무시) · NG/VAL 제외
        _convert_logs_in(okd)                             # .log/.tsv 태깅
        files, okd, _layout = _train_files(base)
        _tag_data_files_direct(files)                     # ★ .csv 포함 직접 태깅(기간필터 0매 방지)
        meta = _folder_metadata_update(okd, files)        # metadata.csv 생성/무결성(요구 2)
        tp, cp, rp = job["tool_prefixes"], job["chamber_prefixes"], job["recipe_prefixes"]
        # ★ 기간 미선택(빈칸) = 전체 기간 학습. 둘 다 비면 end_ts 를 아예 보지 않으므로
        #   끝시간을 못 읽은 파일(태깅 실패 등)도 전부 학습에 포함된다.
        df_ = str(job.get("date_from") or "").strip()
        dt_ = str(job.get("date_to") or "").strip()
        use_date = bool(df_ or dt_)
        kept, drop = [], {"설비": 0, "챔버": 0, "유형": 0, "기간": 0, "끝시간 미상(기간 지정됨)": 0}
        for f in files:
            m = meta.get(f.stem) or {}
            if tp and not _match_prefixes(m.get("tool") or _tool_from_name(f.stem), tp):
                drop["설비"] += 1; continue
            if cp and not _match_chamber(m.get("chamber"), cp, f.stem):
                drop["챔버"] += 1; continue
            if rp and not _match_prefixes(_recipe_display(m.get("recipe") or ""), rp):
                drop["유형"] += 1; continue
            if use_date:                                  # 기간 지정 시에만 end_ts 검사
                ets = str(m.get("end_ts") or "")[:10]
                if not ets:
                    drop["끝시간 미상(기간 지정됨)"] += 1; continue
                if (df_ and ets < df_) or (dt_ and ets > dt_):
                    drop["기간"] += 1; continue
            kept.append(f)
        job["filter_drop"] = {k: v for k, v in drop.items() if v}
        job["period"] = (f"{df_ or '처음'} ~ {dt_ or '끝'}" if use_date else "전체 기간")
        print(f"[train] {job['id']} 학습대상 {len(kept)}/{len(files)}매 · 기간={job['period']}"
              + (f" · 제외 {job['filter_drop']}" if job["filter_drop"] else ""), flush=True)
        if not kept:
            det = ", ".join(f"{k} {v}매" for k, v in job["filter_drop"].items()) or "폴더가 비어 있음"
            raise RuntimeError(f"필터에 맞는 학습 파일이 0매입니다 (전체 {len(files)}매 중 제외: {det})")
        ddir = self._jdir(job["id"]) / "data"
        shutil.rmtree(ddir, ignore_errors=True)
        (ddir / "OK").mkdir(parents=True, exist_ok=True)
        for f in kept:
            (ddir / "OK" / f.name).symlink_to(f.resolve())
        for sub in ("NG", "VAL"):                         # 임계 검증용 폴더는 통째 링크(있으면)
            src = base / sub
            if src.is_dir():
                (ddir / sub).symlink_to(src.resolve())
        job["n_files"] = len(kept)
        return ddir

    def _launch(self, job: Dict):
        jdir = self._jdir(job["id"])
        ddir = self._prepare_data(job)
        art = jdir / "artifacts"
        o = job.get("opts") or self._validated_opts({"h_cart_mode": job.get("h_cart_mode")})
        cmd = [sys.executable, "-u", "-m", "src.train_jw.run", "tcn-search",
               "--trials", str(job["trials"]), "--seed", str(job["seed"]),
               "--data-name", str(ddir), "--artifacts-dir", str(art)]
        cmd += self._opts_to_args(o)                      # README 전체 옵션(기본값은 생략)
        if job.get("epochs"):
            cmd += ["--epochs", str(int(job["epochs"]))]
        if job.get("file_format") and job["file_format"] != "auto":
            cmd += ["--file-format", job["file_format"]]
        env = dict(os.environ)
        glist = [int(g) for g in (job.get("gpus") or [job.get("gpu", 0)])]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in glist)   # 멀티 GPU → GPU당 1워커 자동
        nw = o.get("n_workers") or (len(glist) if len(glist) > 1 else None)
        if nw:
            cmd += ["--n-workers", str(int(nw))]
        env.setdefault("IO_TCN_LOG_EVERY", "1")           # 현황판 loss 곡선을 epoch 단위로
        if job.get("fixed_params"):                       # ★ 고정 학습: 단일값 search space 파일 주입
            sp = self._fixed_space(job["fixed_params"])
            spp = jdir / "search_space_fixed.json"
            spp.write_text(json.dumps(sp, ensure_ascii=False, indent=1), encoding="utf-8")
            env["IO_SEARCH_SPACE"] = str(spp)
        logf = open(jdir / "log.txt", "ab", buffering=0)
        p = subprocess.Popen(cmd, cwd=str(PKG.parent), env=env, stdout=logf,
                             stderr=subprocess.STDOUT, start_new_session=True)   # 자체 프로세스그룹
        job["pid"] = p.pid
        job["status"] = "running"
        job["started"] = _now()
        job["_proc"] = p
        job["_log_pos"] = 0
        job["cmd"] = " ".join(shlex.quote(c) for c in cmd)   # 재현용 실행 커맨드(UI 표시)
        self._save(job)
        print(f"[train] {job['id']} 시작 — GPU{job['gpu']} · {job['n_files']}매 · pid {p.pid}", flush=True)

    @staticmethod
    def _fixed_space(fp: Dict) -> Dict:
        """고정 파라미터 → 단일값 search space (미지정 키는 기본 search space 첫 값)."""
        base = json.loads((PKG / "train_jw" / "search_space.json").read_text(encoding="utf-8"))
        out = {}
        for k, lst in base.items():
            v = fp.get(k)
            if v is None:
                out[k] = [lst[0]]
            else:
                if isinstance(lst[0], bool):
                    v = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "on")
                elif isinstance(lst[0], int) and not isinstance(lst[0], bool):
                    v = int(float(v))
                elif isinstance(lst[0], float):
                    v = float(v)
                out[k] = [v]
        return out

    # ── 제어: 일시정지/재개/완전중단 (프로세스그룹 단위 → 고아 0) ──
    def _pg(self, job: Dict) -> int:
        return int(job.get("pid") or 0)

    def pause(self, jid: str):
        job = self._get(jid)
        if job["status"] != "running":
            raise HTTPException(400, "실행 중인 작업만 일시정지 가능")
        os.killpg(self._pg(job), signal.SIGSTOP)          # GPU 연산 정지(VRAM 은 유지)
        job["status"] = "paused"; self._save(job)

    def resume(self, jid: str):
        job = self._get(jid)
        if job["status"] != "paused":
            raise HTTPException(400, "일시정지 상태가 아님")
        os.killpg(self._pg(job), signal.SIGCONT)
        job["status"] = "running"; self._save(job)

    def stop(self, jid: str):
        job = self._get(jid)
        if job["status"] not in ("running", "paused", "queued"):
            raise HTTPException(400, "실행/대기 중이 아님")
        if job["status"] == "queued":
            job["status"] = "stopped"; job["note"] = "대기 중 취소"; self._save(job)
            return
        pg = self._pg(job)
        proc = job.get("_proc")
        try:
            os.killpg(pg, signal.SIGCONT)                 # 일시정지 상태여도 종료 신호를 받도록
            os.killpg(pg, signal.SIGTERM)
        except OSError:
            pass
        def _force():
            try:
                if proc is not None:
                    try:
                        proc.wait(timeout=10)             # 정상 종료 대기 + 좀비 회수(reap)
                    except subprocess.TimeoutExpired:
                        os.killpg(pg, signal.SIGKILL)     # 10초 내 미종료 → 강제(고아 방지)
                        proc.wait(timeout=10)
                elif self._alive(pg):
                    time.sleep(10)
                    if self._alive(pg):
                        os.killpg(pg, signal.SIGKILL)
            except OSError:
                pass
        threading.Thread(target=_force, daemon=True).start()
        job["status"] = "stopped"; self._save(job)

    def _get(self, jid: str) -> Dict:
        with self.lock:
            job = self.jobs.get(jid)
        if not job:
            raise HTTPException(404, f"작업 없음: {jid}")
        return job

    # ── 감시: 로그 파싱(진행곡선) + 종료 처리(export) + 큐 스케줄 ──
    def _monitor_loop(self):
        while True:
            time.sleep(3)
            try:
                with self.lock:
                    active = [j for j in self.jobs.values() if j.get("status") in ("running", "paused")]
                    finished = [j for j in self.jobs.values()
                                if j.get("status") not in ("running", "paused") and j.get("_proc") is not None]
                for job in finished:                      # 종료된 작업 프로세스 회수(좀비 방지)
                    try:
                        if job["_proc"].poll() is not None:
                            job.pop("_proc", None)
                    except Exception:                     # noqa: BLE001
                        job.pop("_proc", None)
                for job in active:
                    self._pump_log(job)
                    pid = int(job.get("pid") or 0)
                    proc = job.get("_proc")
                    rc = proc.poll() if proc is not None else (None if self._alive(pid) else -1)
                    if rc is not None and job["status"] in ("running", "paused"):
                        self._pump_log(job)
                        self._finish(job, rc)
                self._try_start_queued()
            except Exception:                             # noqa: BLE001
                traceback.print_exc()

    def _pump_log(self, job: Dict):
        p = self._jdir(job["id"]) / "log.txt"
        if not p.exists():
            return
        try:
            pos = int(job.get("_log_pos") or 0)
            sz = p.stat().st_size
            if sz <= pos:
                return
            with open(p, "rb") as f:
                f.seek(pos)
                chunk = f.read(min(sz - pos, 2_000_000)).decode("utf-8", "replace")
            job["_log_pos"] = pos + len(chunk.encode("utf-8", "replace"))
            # ★ 줄 순서대로 처리 — 멀티 GPU 는 여러 워커 출력이 한 파일에 섞이므로
            #   'worker 시작' 줄을 먼저 만나 pid→(GPU·trial) 을 갱신한 뒤 그 pid 의 epoch 을 귀속시킨다.
            wmap = job.setdefault("wmap", {})              # pid(str) → {"gpu": ..., "trial": n}
            glist = [int(g) for g in (job.get("gpus") or [job.get("gpu", 0)])]
            trial_now = len(job.get("trials_done") or []) + 1
            for line in chunk.splitlines():
                mw = _WORKER_RE.search(line)
                if mw:
                    wmap[mw.group(1)] = {"gpu": mw.group(2), "trial": int(mw.group(3))}
                    continue
                mf = _FIT_RE.search(line)
                if mf:                                    # 진행 중 trial 의 구조(현황판 표기)
                    job["cur"] = {"trial": trial_now, "past_window": int(mf.group(1)),
                                  "future_window": int(mf.group(2)), "stride": int(mf.group(3)),
                                  "in_ch": int(mf.group(4)), "target_ch": int(mf.group(5))}
                    continue
                me = _EPOCH_RE.search(line)
                if me:
                    pid = me.group(4)
                    w = wmap.get(pid) if pid else None
                    if w:                                 # 멀티 워커: pid 로 GPU·trial 확정
                        gpu, trial = str(w["gpu"]), int(w["trial"])
                    else:                                 # 단일 워커: 순차 trial · 선택한 GPU
                        gpu, trial = (str(glist[0]) if glist else "0"), trial_now
                    job.setdefault("progress", []).append(
                        {"trial": trial, "epoch": int(me.group(1)), "epochs": int(me.group(2)),
                         "val_mse": float(me.group(3)), "gpu": gpu, "dev": me.group(5) or ""})
                    continue
                mt = _TRIAL_RE.search(line)
                if mt:
                    job.setdefault("trials_done", []).append(
                        {"trial": int(mt.group(1)), "of": int(mt.group(2)),
                         "val_mse": float(mt.group(3)),
                         "arch": {"past_window": int(mt.group(4)), "future_window": int(mt.group(5)),
                                  "stride": int(mt.group(6)), "channels": int(mt.group(7)),
                                  "blocks": int(mt.group(8)), "kernel": int(mt.group(9))}})
                    trial_now = len(job["trials_done"]) + 1
            if len(job.get("progress") or []) > 6000:     # RAM 상한(요구 8): 곡선 다운샘플
                job["progress"] = job["progress"][::2]
        except Exception:                                 # noqa: BLE001
            traceback.print_exc()

    def _finish(self, job: Dict, rc: int):
        if job.get("status") == "stopped" or rc in (-signal.SIGTERM, -signal.SIGKILL):
            job["status"] = "stopped"
            job["ended"] = _now()
            self._save(job)
            return
        if rc != 0:
            job["status"] = "failed"
            job["note"] = f"학습 프로세스 종료코드 {rc} — 로그 확인"
            job["ended"] = _now()
            self._save(job)
            return
        # ★ 마무리(ONNX 변환 + OK 누적밴드)는 수 분~수십 분 걸릴 수 있다.
        #   ① 이 구간을 'done' 으로 두면 출력폴더가 비었는데 완료로 보이므로 별도 상태로 표시하고,
        #   ② 감시 스레드에서 직접 돌리면 그동안 다른 GPU 작업의 진행표시·큐가 멈추므로 스레드로 분리한다.
        job["status"] = "finalizing"
        job["note"] = "마무리 중(ONNX 변환·OK 누적밴드 생성)"
        self._save(job)
        threading.Thread(target=self._finalize, args=(job,), daemon=True,
                         name=f"train-final-{job['id']}").start()

    def _finalize(self, job: Dict):
        try:                                              # 완료 → ONNX export → 출력폴더(요구 7)
            art = self._jdir(job["id"]) / "artifacts"
            out = Path(job["out_dir"])
            out.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([sys.executable, "-m", "src.train_jw.export_onnx_forecast",
                                str(art), "-o", str(out)], cwd=str(PKG.parent),
                               capture_output=True, text=True, timeout=1800)
            with open(self._jdir(job["id"]) / "log.txt", "ab") as lf:
                lf.write(("\n[export]\n" + (r.stdout or "") + (r.stderr or "")).encode("utf-8", "replace"))
            for extra in ("search_best.json", "best_search.json", "tcn_forecast.pt"):
                src = art / extra
                if src.exists():
                    shutil.copy2(src, out / extra)        # 임계(레시피dict)·RCA(H .pt)용
            job["export_ok"] = (r.returncode == 0)
            job["params"] = self._count_params(art / "tcn_forecast.pt")
            job["note"] = "" if r.returncode == 0 else "학습 성공·ONNX 변환 실패 — 로그 확인"
            # ★ OK 누적 밴드(ok_accum_bands.npz): 원래는 리포트(recon-explain)를 돌려야 생겼지만,
            #   검사 앱의 '센서별 복원오차 점수/정상 밴드'에 필요하므로 학습 직후 바로 만든다.
            if r.returncode == 0:
                self._save(job)                           # 밴드 생성 전 진행상태 반영
                job["ok_bands"] = self._build_ok_bands(job, art, out)
            job["status"] = "done" if r.returncode == 0 else "failed"
        except Exception as e:                            # noqa: BLE001
            job["status"] = "failed"; job["note"] = f"export 오류: {e}"
            traceback.print_exc()
        job["ended"] = _now()
        self._save(job)
        print(f"[train] {job['id']} {job['status']} → {job.get('out_dir')}", flush=True)

    def _build_ok_bands(self, job: Dict, art: Path, out: Path) -> bool:
        """학습에 쓴 OK 웨이퍼로 ok_accum_bands.npz 생성 → 출력폴더(리포트 실행 불필요).

        별도 프로세스(앱에 torch 미로드). 실패해도 모델 사용에는 지장 없으므로 경고만 남긴다."""
        okd = self._jdir(job["id"]) / "data" / "OK"
        try:
            o = dict(job.get("opts") or {})
            o.setdefault("h_cart_mode", job.get("h_cart_mode") or "include")
            o["file_format"] = job.get("file_format") or "auto"
            r = subprocess.run(
                [sys.executable, "-u", "-m", "src.app.build_ok_bands",
                 str(art), str(okd), str(out), "--json", json.dumps(o, ensure_ascii=False)],
                cwd=str(PKG.parent), capture_output=True, text=True, timeout=7200)
            with open(self._jdir(job["id"]) / "log.txt", "ab") as lf:
                lf.write(("\n[ok-bands]\n" + (r.stdout or "") + (r.stderr or ""))
                         .encode("utf-8", "replace"))
            ok = (r.returncode == 0) and (out / "ok_accum_bands.npz").exists()
            if not ok:
                job["note"] = ((job.get("note") or "") +
                               " · OK 누적밴드 생성 실패(센서별 점수/정상밴드 미표시) — 로그 확인").strip(" ·")
            return ok
        except Exception as e:                            # noqa: BLE001
            job["note"] = ((job.get("note") or "") + f" · OK 누적밴드 오류: {e}").strip(" ·")
            traceback.print_exc()
            return False

    @staticmethod
    def _count_params(pt: Path):
        """파라미터 수 — 별도 프로세스에서 계산(앱에 torch 를 로드하지 않음: RAM 반납 원칙)."""
        try:
            code = ("import torch,sys; d=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
                    "print(sum(int(v.numel()) for v in d['state'].values()))")
            r = subprocess.run([sys.executable, "-c", code, str(pt)],
                               capture_output=True, text=True, timeout=300)
            return int(r.stdout.strip()) if r.returncode == 0 else None
        except Exception:                                 # noqa: BLE001
            return None

    def snapshot(self) -> List[Dict]:
        with self.lock:
            jobs = [dict((k, v) for k, v in j.items() if not k.startswith("_"))
                    for j in self.jobs.values()]
        return sorted(jobs, key=lambda j: j["id"], reverse=True)


TRAINER: Optional[TrainManager] = None
if _train_supported()["ok"]:
    try:
        TRAINER = TrainManager()
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()


class TrainStartReq(BaseModel):
    folder: str
    out_dir: str
    gpu: int = 0
    gpus: Optional[List[int]] = None                      # 멀티 GPU(예 [0,1]) — GPU당 1워커 병렬
    fixed_params: Optional[Dict] = None                   # 고정 학습(서치 없음, trials=1)
    trials: int = 12
    epochs: Optional[int] = None
    seed: int = 42
    h_cart_mode: str = "include"
    file_format: str = "auto"
    # ── tcn-search 전체 옵션(README CLI 표) ──
    sensor_role_profile: Optional[str] = None             # original | h-split-v1
    h_sensor_preset: Optional[str] = None                 # all | heater-block3 | heater-12
    future_h_mode: Optional[str] = None                   # input | predict
    virtual_sensor_mode: Optional[str] = None             # off | input | target | both
    virtual_sensors: Optional[bool] = None
    search_objective: Optional[str] = None                # correct | precision
    score_norm: Optional[str] = None                      # none | time | final (점수 정규화)
    threshold_wise: Optional[str] = None                  # all | recipe | sensor | step
    threshold_source: Optional[str] = None                # val-search | train-ok-sigma
    threshold_sigma: Optional[float] = None
    threshold_sigma_min: Optional[float] = None
    threshold_sigma_max: Optional[float] = None
    threshold_recipe_multipliers: Optional[str] = None     # "29a=1.1,336=1.2"
    resid_agg: Optional[str] = None                       # sum | p99 | max | mean
    wafer_agg: Optional[str] = None                       # sum | topk_mean | max | mean
    topk: Optional[int] = None
    post_eval_extend_epochs: Optional[int] = None
    post_eval_max_extends: Optional[int] = None
    n_workers: Optional[int] = None
    checkpoint: Optional[bool] = None                     # False → --no-checkpoint
    no_train: Optional[bool] = None
    resume: Optional[bool] = None
    save_trial_artifacts: Optional[bool] = None
    stop_on_perfect_val: Optional[bool] = None
    tool_prefixes: Optional[List[str]] = None
    chamber_prefixes: Optional[List[str]] = None
    recipe_prefixes: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


@app.get("/api/train/support")
def api_train_support():
    return _train_supported()


@app.get("/api/train/gpus")
def api_train_gpus():
    busy = TRAINER._gpu_busy_map() if TRAINER else {}
    return {"gpus": _train_gpus(), "busy": {str(k): v for k, v in busy.items()},
            "free_ram_mb": _free_ram_mb()}


class TrainFolderReq(BaseModel):
    folder: str


def _train_scan(base: Path) -> Dict:
    """학습 폴더 스캔 — 태깅(.csv 포함) + metadata 생성 + 끝시간 커버리지 진단."""
    files, okd, layout = _train_files(base)
    _convert_logs_in(okd)
    files, okd, layout = _train_files(base)
    _tag_data_files_direct(files)
    meta = _folder_metadata_update(okd, files)
    dated = [str((meta.get(f.stem) or {}).get("end_ts") or "")[:10] for f in files]
    dated = sorted([d for d in dated if d])
    return {"folder": str(okd), "layout": layout, "files": len(files), "meta_rows": len(meta),
            "meta": meta, "files_list": files,
            "with_end_ts": len(dated), "no_end_ts": len(files) - len(dated),
            "date_min": dated[0] if dated else None, "date_max": dated[-1] if dated else None}


@app.post("/api/train/metadata")
def api_train_metadata(req: TrainFolderReq):
    """학습 폴더 metadata 생성/갱신(요구 2) — OK 하위폴더(대소문자 무시)가 있으면 거기에."""
    base = Path(req.folder or "").expanduser()
    if not base.is_dir():
        raise HTTPException(404, f"폴더 없음: {base}")
    s = _train_scan(base)
    return {k: v for k, v in s.items() if k not in ("meta", "files_list")}


class TrainPreviewReq(TrainStartReq):
    """미리보기 — start 와 같은 필드를 받되 folder/out_dir 만 실제로 쓴다."""
    out_dir: str = ""


@app.post("/api/train/preview")
def api_train_preview(req: TrainPreviewReq):
    """★ 학습 시작 전에 '몇 매가 학습에 참여하는지 + 무엇이 왜 제외되는지' 미리 확인."""
    base = Path(req.folder or "").expanduser()
    if not base.is_dir():
        raise HTTPException(404, f"폴더 없음: {base}")
    s = _train_scan(base)
    meta = s["meta"]
    tp = [x for x in (req.tool_prefixes or []) if str(x).strip()]
    cp = [x for x in (req.chamber_prefixes or []) if str(x).strip()]
    rp = [x for x in (req.recipe_prefixes or []) if str(x).strip()]
    df_ = str(req.date_from or "").strip()
    dt_ = str(req.date_to or "").strip()
    use_date = bool(df_ or dt_)
    kept, drop = [], {"설비": 0, "챔버": 0, "유형": 0, "기간": 0, "끝시간 미상(기간 지정됨)": 0}
    for f in s["files_list"]:
        m = meta.get(f.stem) or {}
        if tp and not _match_prefixes(m.get("tool") or _tool_from_name(f.stem), tp):
            drop["설비"] += 1; continue
        if cp and not _match_chamber(m.get("chamber"), cp, f.stem):
            drop["챔버"] += 1; continue
        if rp and not _match_prefixes(_recipe_display(m.get("recipe") or ""), rp):
            drop["유형"] += 1; continue
        if use_date:
            ets = str(m.get("end_ts") or "")[:10]
            if not ets:
                drop["끝시간 미상(기간 지정됨)"] += 1; continue
            if (df_ and ets < df_) or (dt_ and ets > dt_):
                drop["기간"] += 1; continue
        kept.append(f)
    tips = []
    if not kept:
        if s["files"] == 0:
            tips.append("폴더에서 데이터 파일(.log/.tsv/.csv)을 찾지 못했습니다 — 경로를 확인하세요")
        elif drop["끝시간 미상(기간 지정됨)"] == s["files"]:
            tips.append("모든 파일의 끝시간을 읽지 못했습니다 — 기간을 비우면(전체 기간) 학습 가능합니다")
        else:
            tips.append("필터 조건을 완화하세요(체크/입력을 비우면 그 항목은 전체 허용)")
    return {"folder": s["folder"], "layout": s["layout"], "files": s["files"],
            "period": (f"{df_ or '처음'} ~ {dt_ or '끝'}" if use_date else "전체 기간"),
            "date_min": s["date_min"], "date_max": s["date_max"], "no_end_ts": s["no_end_ts"],
            "kept": len(kept), "drop": {k: v for k, v in drop.items() if v},
            "samples": [f.name for f in kept[:5]], "tips": tips}


@app.post("/api/train/best-params")
def api_train_best_params(req: TrainFolderReq):
    """기존 search_best.json 에서 best.params 로드 — '고정 학습' 입력칸 채우기용.
    folder 에 파일 경로 또는 폴더(안의 search_best.json) 를 받는다."""
    p = Path(req.folder or "").expanduser()
    if p.is_dir():
        p = p / "search_best.json"
    if not p.exists():
        raise HTTPException(404, f"search_best.json 없음: {p}")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        best = d.get("best") or {}
        params = best.get("params") or {}
        if not params:
            raise ValueError("best.params 없음")
        return {"params": params, "n_params": best.get("n_params"),
                "val_mse": best.get("forecast_val_mse"), "source": str(p)}
    except HTTPException:
        raise
    except Exception as e:                                # noqa: BLE001
        raise HTTPException(400, f"search_best.json 파싱 실패: {e}")


@app.post("/api/train/start")
def api_train_start(req: TrainStartReq):
    if TRAINER is None:
        raise HTTPException(400, f"학습 미지원: {_train_supported().get('reason')}")
    return TRAINER.start(req.dict())


@app.get("/api/train/jobs")
def api_train_jobs():
    return {"jobs": TRAINER.snapshot() if TRAINER else []}


@app.post("/api/train/pause/{jid}")
def api_train_pause(jid: str):
    TRAINER.pause(jid); return {"ok": True}


@app.post("/api/train/resume/{jid}")
def api_train_resume(jid: str):
    TRAINER.resume(jid); return {"ok": True}


@app.post("/api/train/stop/{jid}")
def api_train_stop(jid: str):
    TRAINER.stop(jid); return {"ok": True}


@app.get("/api/train/log/{jid}")
def api_train_log(jid: str, tail: int = 120):
    if TRAINER is None:
        raise HTTPException(400, "학습 미지원")
    p = TRAIN_DIR / jid / "log.txt"
    if not p.exists():
        return {"lines": []}
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": lines[-max(1, min(int(tail), 2000)):]}
    except Exception as e:                                # noqa: BLE001
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    # 로컬 WebView2 클라이언트 전용 → 루프백 바인딩(LAN 노출 방지).
    uvicorn.run("src.app.server:app", host=os.environ.get("IO_HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", 8000)), reload=False)
