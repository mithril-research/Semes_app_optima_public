"""CPU 사용 제한 — 코어 몇 개를 '연산에 쓰지 않고' 남겨 둔다.

제어기는 검사 앱 말고도 설비 프로그램이 함께 돈다. 검사가 전 코어를 먹으면
다른 프로그램이 밀리므로, 코어 몇 개는 비워 두고 싶다는 요구가 있다.

이 모듈이 하는 일
  1) 물리 코어 / 논리 프로세서 개수를 센다(하이퍼스레딩 구분).
  2) 설정(예약 코어 수 또는 최대 사용 코어 수)에서 '쓸 수 있는 코어 수'를 정한다.
  3) 계산 라이브러리의 스레드 수 환경변수를 정한다.
     ★ 이건 **numpy 가 import 되기 전에** 해야 한다(OpenBLAS/MKL 은 import 시점에 읽는다).
        그래서 이 모듈은 server.py 최상단, numpy 보다 먼저 import 된다.
  4) (선택) 프로세스 affinity 를 걸어 지정한 코어에서만 실행되게 한다.
     환경변수로는 못 막는 스레드까지 포함해 '진짜' 제한이 되는 유일한 방법이다.
     윈도우는 SetProcessAffinityMask(Win7 포함 지원), 리눅스는 sched_setaffinity.

설정(app_config.json)
  cpu_reserve   : 연산에 쓰지 않고 비워 둘 **물리 코어 수** (기본 0)
  cpu_max       : 사용할 최대 물리 코어 수 (0=제한 없음). reserve 와 함께 쓰면 더 작은 쪽.
  cpu_affinity  : true 면 프로세스 affinity 까지 적용(기본 false — 스레드 수 제한만)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_STATE: dict = {"applied": False}


def _win_kernel32():
    """Use pointer-sized process handles and preserve Win32 last-error values."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.argtypes = []
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,
                                          ctypes.POINTER(ctypes.c_size_t),
                                          ctypes.POINTER(ctypes.c_size_t)]
    k32.GetProcessAffinityMask.restype = wintypes.BOOL
    return k32


def _win_process_mask():
    import ctypes
    k32 = _win_kernel32()
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(k32.GetCurrentProcess(),
                                     ctypes.byref(process_mask), ctypes.byref(system_mask)):
        error = ctypes.get_last_error()
        raise OSError(error, "GetProcessAffinityMask 실패 (Windows 오류 %d)" % error)
    return process_mask.value


def _win_set_affinity(mask):
    import ctypes
    k32 = _win_kernel32()
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.c_size_t(mask)):
        error = ctypes.get_last_error()
        raise OSError(error, "SetProcessAffinityMask 실패 (Windows 오류 %d)" % error)
    applied = _win_process_mask()
    if applied != mask:
        raise RuntimeError("affinity 검증 실패: 요청 0x%x / 실제 0x%x" % (mask, applied))


def current_affinity_logical():
    if os.name == "nt":
        return bin(_win_process_mask()).count("1")
    return len(os.sched_getaffinity(0))


# ── 코어 수 세기 ─────────────────────────────────────────────────────
def _win_topology():
    """윈도우: GetLogicalProcessorInformation 으로 (물리코어별 논리마스크) 목록.
    XP SP3 / Vista 부터 있는 API 라 윈도우 7 에서도 그대로 동작한다."""
    import ctypes
    from ctypes import wintypes

    class _U(ctypes.Union):
        _fields_ = [("ProcessorCore", ctypes.c_byte * 1),
                    ("NumaNode", ctypes.c_uint32),
                    ("Cache", ctypes.c_byte * 16),
                    ("Reserved", ctypes.c_uint64 * 2)]

    class SLPI(ctypes.Structure):
        _fields_ = [("ProcessorMask", ctypes.c_size_t),
                    ("Relationship", ctypes.c_uint32),
                    ("u", _U)]

    k32 = ctypes.windll.kernel32
    ln = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(ln))
    n = ln.value // ctypes.sizeof(SLPI)
    if n <= 0:
        return []
    buf = (SLPI * n)()
    if not k32.GetLogicalProcessorInformation(ctypes.byref(buf), ctypes.byref(ln)):
        return []
    return [int(e.ProcessorMask) for e in buf if e.Relationship == 0]   # 0 = RelationProcessorCore


def _linux_topology():
    """리눅스: sysfs 의 thread_siblings_list 로 물리 코어별 논리 CPU 묶음."""
    cores, seen = [], set()
    base = Path("/sys/devices/system/cpu")
    try:
        for d in sorted(base.glob("cpu[0-9]*")):
            f = d / "topology" / "thread_siblings_list"
            if not f.exists():
                continue
            ids = set()
            for part in f.read_text().strip().split(","):
                if "-" in part:
                    a, b = part.split("-")[:2]
                    ids.update(range(int(a), int(b) + 1))
                else:
                    ids.add(int(part))
            key = tuple(sorted(ids))
            if key not in seen:
                seen.add(key)
                cores.append(sum(1 << i for i in ids))
    except Exception:                                        # noqa: BLE001
        return []
    return cores


def topology() -> dict:
    """{'physical': 물리코어수, 'logical': 논리프로세서수, 'masks': [코어별 affinity 마스크]}"""
    masks = []
    try:
        masks = _win_topology() if os.name == "nt" else _linux_topology()
    except Exception:                                        # noqa: BLE001
        masks = []
    logical = os.cpu_count() or 1
    physical = len(masks) or logical
    return {"physical": physical, "logical": logical, "masks": masks}


# ── 설정 읽기(numpy 보다 먼저 도는 시점이라 server 모듈을 못 쓴다) ──
def _writable_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[1]


def _read_cfg() -> dict:
    """서버와 같은 앱 폴더 설정을 우선 읽고, 이전 데이터 폴더는 이전 호환용으로만 사용."""
    root = _writable_root()
    current = root / "app_config.json"
    if current.exists():
        try:
            return json.loads(current.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            return {}
    data_dir = root / "데이터"
    try:
        sj = root / "storage.json"
        if sj.exists():
            v = (json.loads(sj.read_text(encoding="utf-8")) or {}).get("data")
            if v:
                p = Path(str(v)).expanduser()
                if not p.is_absolute():
                    p = root / p
                data_dir = p
    except Exception:                                        # noqa: BLE001
        pass
    for cand in (data_dir / "app_config.json", root / "데이터" / "app_config.json"):
        try:
            if cand.exists():
                return json.loads(cand.read_text(encoding="utf-8")) or {}
        except Exception:                                    # noqa: BLE001
            continue
    return {}


def budget(cfg: dict = None) -> dict:
    """쓸 수 있는 코어 수 계산. 환경변수 IO_CPU_RESERVE / IO_CPU_MAX 가 설정보다 우선."""
    cfg = _read_cfg() if cfg is None else cfg
    topo = topology()
    phys = topo["physical"]

    def _int(v, d=0):
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return d

    reserve = _int(os.environ.get("IO_CPU_RESERVE") or cfg.get("cpu_reserve") or 0)
    cmax = _int(os.environ.get("IO_CPU_MAX") or cfg.get("cpu_max") or 0)
    use = phys - reserve
    if cmax:
        use = min(use, cmax)
    use = max(1, min(phys, use))                             # 최소 1코어는 남긴다(0이면 앱이 멈춘다)
    # 스레드 수 = '쓰기로 한 물리코어들이 실제로 가진 논리 프로세서 수'.
    # ★ logical//physical 로 나누면 P/E 코어가 섞인 CPU(코어마다 스레드 수가 다름)에서 틀린다.
    masks = topo["masks"]
    if masks:
        threads = sum(bin(m).count("1") for m in masks[:use])
    else:
        threads = max(1, (topo["logical"] * use) // max(1, phys))
    return {**topo, "reserve": reserve, "max": cmax, "use_cores": use,
            "threads": max(1, threads),
            "affinity": bool(cfg.get("cpu_affinity", False))}


# ── 적용 ────────────────────────────────────────────────────────────
_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def apply_early() -> dict:
    """numpy import 전에 호출 — 계산 라이브러리 스레드 수를 고정한다."""
    b = budget()
    if b["use_cores"] < b["physical"]:                       # 제한이 걸린 경우에만 건드린다
        for k in _ENV_KEYS:
            os.environ.setdefault(k, str(b["threads"]))
        os.environ.setdefault("IO_ORT_THREADS", str(b["threads"]))
    _STATE["budget"] = b
    _STATE["applied"] = True
    return b


def apply_affinity(b: dict = None) -> str:
    """프로세스를 허용 코어에만 묶는다. 반환=결과 설명(빈 문자열이면 적용 안 함).

    ★ 스레드 수 제한만으로는 '반드시 비워 둔다'가 보장되지 않는다(라이브러리가 자체
      스레드를 더 만들 수 있음). 코어를 확실히 남기려면 affinity 가 필요하다.
    """
    b = b or _STATE.get("budget") or budget()
    if not b.get("affinity"):
        return ""
    masks = b.get("masks") or []
    if not masks or b["use_cores"] >= b["physical"]:
        return ""
    mask = 0
    for m in masks[:b["use_cores"]]:                         # 앞쪽 코어부터 사용, 뒤쪽을 비워 둔다
        mask |= m
    if not mask:
        return ""
    try:
        if os.name == "nt":
            _win_set_affinity(mask)
        else:
            cpus = [i for i in range(b["logical"]) if mask >> i & 1]
            os.sched_setaffinity(0, cpus)                    # type: ignore[attr-defined]
        return (f"물리코어 {b['use_cores']}/{b['physical']}개만 사용 "
                f"(마스크 0x{mask:x}) — 나머지는 다른 프로그램 몫으로 비움")
    except Exception as e:                                   # noqa: BLE001
        return f"affinity 적용 실패: {e}"


def state() -> dict:
    b = _STATE.get("budget") or budget()
    return {**b, "applied": bool(_STATE.get("applied")),
            "env": {k: os.environ.get(k) for k in _ENV_KEYS if os.environ.get(k)}}
