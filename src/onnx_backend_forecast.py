"""torch 없이 onnxruntime 로 IO-Recon-TCN 추론 (배포 경량 경로).

TCNForecastModel 과 동일한 predict/actual/residual/error_level/z_residual/
score_records 를 numpy + onnxruntime 로 재현(수치 동일 — export_onnx_forecast 검증).
torch 를 import 하지 않음 → 배포 exe 에서 torch(≈450MB) 제거 가능.

score 는 sensor_graph.graph_adjust_abs_z(순수 numpy)를 그대로 재사용 → 그래프 부스트도 동일.
"""
from __future__ import annotations

import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np

from .auxiliary_sensors import configure_auxiliary_model, validate_auxiliary_record
from .config import IOConfig
from .wafer_score_override import override_from_npz
from .sensor_graph import graph_adjust_abs_z


class OnnxForecastModel:
    def __init__(self, onnx_path: Path, meta_path: Path, cfg: IOConfig):
        import onnxruntime as ort
        self._meta_path = Path(meta_path)
        d = np.load(meta_path, allow_pickle=True)
        self.wafer_score_override = override_from_npz(d)
        self.io_cols = [str(c) for c in d["io_cols"].tolist()]
        self.input_sensor_cols = [str(c) for c in d["input_sensor_cols"].tolist()]
        self.sensor_cols = [str(c) for c in d["sensor_cols"].tolist()]
        self.target_indices = list(np.asarray(d["target_indices"], dtype=int))
        self.h_indices = list(np.asarray(d["h_indices"], dtype=int))
        self.n_steps_global = int(d["n_steps_global"])
        self.schema_mode = str(d["schema_mode"])
        self.normalization_mode = str(d.get("normalization_mode", "standard")).strip().lower()
        if self.normalization_mode not in ("standard", "minmax"):
            raise ValueError(f"지원하지 않는 normalization_mode: {self.normalization_mode}")
        self._load_normalization(d)
        self.input_sen_mean = np.asarray(d.get("input_sen_mean", np.zeros(len(self.input_sensor_cols)))
                                        if self.normalization_mode == "minmax" else d["input_sen_mean"], dtype=np.float64)
        self.input_sen_std = np.asarray(d.get("input_sen_std", np.ones(len(self.input_sensor_cols)))
                                 if self.normalization_mode == "minmax" else d["input_sen_std"], dtype=np.float64)
        self.sen_mean = np.asarray(d.get("sen_mean", np.zeros(len(self.sensor_cols)))
                                 if self.normalization_mode == "minmax" else d["sen_mean"], dtype=np.float64)
        self.sen_std = np.asarray(d.get("sen_std", np.ones(len(self.sensor_cols)))
                                 if self.normalization_mode == "minmax" else d["sen_std"], dtype=np.float64)
        self.resid_abs_limit = np.asarray(d["resid_abs_limit"], dtype=np.float64)
        self.resid_sigma = (None if d["resid_sigma"] is None or d["resid_sigma"].dtype == object
                            else np.asarray(d["resid_sigma"], dtype=np.float64))
        self.resid_std = self.resid_sigma
        self._per_step_limit = bool(d["per_step_limit"])
        self._per_step_sigma = bool(d["per_step_sigma"])
        self.w = int(d["w"]); self.h = int(d["h"]); self.stride = int(d["stride"])
        self._n_in = int(d["n_in"])
        self.future_h_mode = str(d["future_h_mode"])
        self.score_norm = "none"                         # 앱이 search_best.json 감지 결과를 주입
        self.virtual_sensor_mode = str(d["virtual_sensor_mode"])
        self.arch = {"future_h_mode": self.future_h_mode,
                     "virtual_sensor_mode": self.virtual_sensor_mode}
        self.cfg = replace(cfg, use_step_onehot=bool(d["use_step_onehot"]))
        configure_auxiliary_model(self, d)
        d.close()
        # ★ CPU 추론 튜닝: 모든 코어로 그래프 최적화 + 병렬 실행 → 대량 웨이퍼 처리 가속.
        #   IO_ORT_THREADS 로 코어 수 조정 가능(기본=전체 코어). 0/미설정이면 ORT 기본.
        _so = ort.SessionOptions()
        _so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ★ CPU 메모리 아레나 끔(기본). 아레나는 배치 크기가 들쭉날쭉하면 최대 크기로
        #   계속 자라고 반납하지 않는다 → 대량 검사 중 피크가 누적되다 네이티브
        #   bad_alloc(abort)로 '프로세스가 흔적 없이 사라지는' 죽음의 주원인.
        #   끄면 할당/해제가 OS 로 바로 반납돼 느려짐은 미미하고 안정성이 크게 는다.
        #   IO_ORT_ARENA=1 로 예전 동작(아레나 사용) 복원 가능.
        if (os.environ.get("IO_ORT_ARENA") or "0").strip() != "1":
            _so.enable_cpu_mem_arena = False
        try:
            _nt = int(os.environ.get("IO_ORT_THREADS") or (os.cpu_count() or 0))
            if _nt > 0:
                _so.intra_op_num_threads = _nt
        except (TypeError, ValueError):
            pass
        self._onnx_path = str(onnx_path)
        self._so = _so
        self._run_lock = threading.Lock()                       # 동시 Run 직렬화(DirectML/CUDA 크래시 방지)
        self.active_provider = None
        self.gpu_fallback_reason = None                          # GPU→CPU 로 내려간 이유(상태표시용)
        self._build_session(self._pick_providers())              # 세션 생성(+GPU면 자가진단)
        self._in = self.sess.get_inputs()[0].name
        self._out = self.sess.get_outputs()[0].name

    def _load_normalization(self, meta):
        """Column-wise train-OK scaling contract; masks/one-hot never pass here."""
        if self.normalization_mode != "minmax":
            return
        def scalar(key):
            if key not in meta:
                raise ValueError(f"minmax 모델 메타 필드 누락: {key}")
            value = np.asarray(meta[key], dtype=np.float64)
            if value.size != 1 or not np.isfinite(value).all():
                raise ValueError(f"minmax 모델 메타 값 오류: {key}")
            return float(value.reshape(-1)[0])
        self.normalization_min = scalar("normalization_min")
        self.normalization_max = scalar("normalization_max")
        self._normalization_span = self.normalization_max - self.normalization_min
        if not np.isfinite(self._normalization_span) or self._normalization_span <= 0:
            raise ValueError("normalization_max는 normalization_min보다 커야 합니다.")
        for prefix, columns in (("io", self.io_cols), ("input_sen", self.input_sensor_cols),
                                ("sen", self.sensor_cols)):
            values = {}
            for suffix in ("min", "max", "scale"):
                key = prefix + "_" + suffix
                if key not in meta and columns:
                    raise ValueError(f"minmax 모델 메타 필드 누락: {key}")
                arr = np.asarray(meta.get(key, []), dtype=np.float64)
                if arr.shape != (len(columns),):
                    raise ValueError(f"minmax 모델 메타 열 수 불일치: {key} {arr.shape} != {(len(columns),)}")
                if suffix == "scale":
                    # Training saves 1.0 for constant/invalid scales. Be equally defensive.
                    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 1.0)
                elif not np.isfinite(arr).all():
                    raise ValueError(f"minmax 모델 메타 값 오류: {key}")
                values[suffix] = arr
                setattr(self, key, arr)
            if np.any(values["max"] < values["min"]):
                raise ValueError(f"minmax 모델 범위 역전: {prefix}")

    def _minmax(self, x, prefix):
        # No clipping: out-of-training-range values must remain below/above the range.
        out = self.normalization_min + (np.asarray(x, dtype=np.float64)
              - getattr(self, prefix + "_min")) / getattr(self, prefix + "_scale") * self._normalization_span
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_io(self, io):
        values = np.asarray(io, dtype=np.float64)
        return self._minmax(values, "io") if self.normalization_mode == "minmax" else values

    def _inverse_target(self, values):
        if self.normalization_mode == "minmax":
            return ((values - self.normalization_min) / self._normalization_span
                    * self.sen_scale + self.sen_min)
        return values * self.sen_std + self.sen_mean

    @staticmethod
    def _pick_providers():
        """실행 프로바이더 선택. ★ 기본은 CPU(안전). GPU 는 '명시적으로 켤 때만'.

        GPU(DirectML/CUDA)는 자원 부족·드라이버 문제로 앱이 튕길 수 있어 자동으로 켜지 않는다.
        IO_ORT_PROVIDER 로 opt-in: dml(DirectML) / cuda / auto(가능한 GPU 자동) / cpu(기본).
        (설정 화면의 'GPU 가속' 토글이 이 환경변수를 설정한다.)
        """
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        want = (os.environ.get("IO_ORT_PROVIDER") or "").strip().lower()
        if want == "dml":
            order = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif want == "cuda":
            order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif want == "amd":                                      # AMD: ROCm/MIGraphX(리눅스) → DirectML(윈도우)
            order = ["MIGraphXExecutionProvider", "ROCMExecutionProvider",
                     "DmlExecutionProvider", "CPUExecutionProvider"]
        elif want == "auto":                                     # 가능한 GPU 자동(DML→CUDA→ROCm)
            for p in ("DmlExecutionProvider", "CUDAExecutionProvider",
                      "ROCMExecutionProvider", "MIGraphXExecutionProvider"):
                if p in avail:
                    order = [p, "CPUExecutionProvider"]
                    break
            else:
                order = ["CPUExecutionProvider"]
        else:                                                    # "" / cpu / 그 외 → CPU(안전 기본)
            order = ["CPUExecutionProvider"]
        return [p for p in order if p in avail] or ["CPUExecutionProvider"]

    @staticmethod
    def _preload_cuda_dlls():
        """CUDA/cuDNN 라이브러리를 미리 로드 → 시스템에 CUDA 설치가 없어도 CUDA EP 가 동작.

        ★ 리눅스(우분투 학습앱)에서 "학습은 CUDA 되는데 추론만 CPU" 의 주원인:
          torch 는 자기 휠에 CUDA/cuDNN(.so)을 **번들해서** 들고 다니지만,
          onnxruntime-gpu 는 시스템에 설치된 libcudnn/libcublas 를 찾는다.
          시스템 cuDNN 이 없으면 CUDA EP 세션 생성이 실패해 조용히 CPU 로 떨어진다.
          → torch 가 깔아 둔 site-packages/nvidia/*/lib 의 .so 를 직접 선로드해서 해결한다.
        """
        try:
            import sys as _sys
            import onnxruntime as ort
            _dir = None
            base = getattr(_sys, "_MEIPASS", None)               # PyInstaller frozen 번들 루트
            if base:
                cand = os.path.join(base, "nvidia")
                if os.path.isdir(cand):
                    _dir = cand
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls(directory=_dir)             # None 이면 site-packages 자동 탐색
                except TypeError:                                # 구버전 시그니처
                    ort.preload_dlls()
        except Exception:                                        # noqa: BLE001
            pass
        if os.name == "nt":
            return
        # ── 리눅스: nvidia pip 패키지(.so)를 직접 선로드 ────────────────────
        #   순서가 중요하다(cublas → cudnn). 이미 로드돼 있으면 무해하게 넘어간다.
        try:
            import ctypes
            import glob as _glob
            import site as _site
            roots = []
            for base in (list(getattr(_site, "getsitepackages", lambda: [])())
                         + [os.path.dirname(os.path.dirname(__file__))]):
                cand = os.path.join(base, "nvidia")
                if os.path.isdir(cand):
                    roots.append(cand)
            loaded = []
            for pat in ("cublas/lib/libcublasLt.so*", "cublas/lib/libcublas.so*",
                        "cufft/lib/libcufft.so*", "curand/lib/libcurand.so*",
                        "cuda_runtime/lib/libcudart.so*", "cudnn/lib/libcudnn.so*"):
                for root in roots:
                    for so in sorted(_glob.glob(os.path.join(root, pat)), reverse=True):
                        try:
                            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                            loaded.append(os.path.basename(so))
                            break
                        except OSError:
                            continue
            if loaded:
                print(f"[onnx] CUDA 라이브러리 선로드: {len(loaded)}개 "
                      f"({', '.join(loaded[:3])}…)", flush=True)
        except Exception:                                        # noqa: BLE001
            pass

    def _build_session(self, providers):
        """주어진 프로바이더로 세션 생성. GPU면 자가진단(작은 추론)까지 통과해야 채택, 아니면 CPU."""
        import onnxruntime as ort
        if providers and providers[0] in ("CUDAExecutionProvider",
                                          "TensorrtExecutionProvider"):
            self._preload_cuda_dlls()                            # CUDA/cuDNN 번들 DLL 선로드
        prov_arg = list(providers)
        so = self._so
        if providers and providers[0] == "DmlExecutionProvider":
            so.enable_mem_pattern = False                        # DML 호환 위해 필수
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            _dev = os.environ.get("IO_ORT_DEVICE")               # 다중 GPU 시 어댑터 지정(NVIDIA=1 등)
            if _dev is not None and str(_dev).strip() != "":
                try:
                    prov_arg = [("DmlExecutionProvider", {"device_id": int(_dev)})] \
                        + [p for p in providers if p != "DmlExecutionProvider"]
                except (TypeError, ValueError):
                    pass
        try:
            self.sess = ort.InferenceSession(self._onnx_path, sess_options=so, providers=prov_arg)
        except Exception as e:                                   # noqa: BLE001
            print(f"[onnx] {providers} 세션 생성 실패({e}) → CPU 폴백", flush=True)
            self.gpu_fallback_reason = f"세션 생성 실패: {e}"
            self.sess = ort.InferenceSession(self._onnx_path, sess_options=so,
                                             providers=["CPUExecutionProvider"])
        self.active_provider = (self.sess.get_providers() or ["CPUExecutionProvider"])[0]
        gpu = self.active_provider != "CPUExecutionProvider"
        if gpu:                                                  # ★ 자가진단: 실제로 추론되는지 확인
            try:
                self._selftest()
            except Exception as e:                              # noqa: BLE001 (진단 실패 → CPU 로 안전 전환)
                print(f"[onnx] GPU 자가진단 실패({e}) → CPU 로 전환", flush=True)
                self.gpu_fallback_reason = f"자가진단 실패: {e}"
                self.sess = ort.InferenceSession(self._onnx_path, sess_options=so,
                                                 providers=["CPUExecutionProvider"])
                self.active_provider = "CPUExecutionProvider"
                gpu = False
        print(f"[onnx] 실행 프로바이더: {self.active_provider}"
              f"{'  (GPU 가속)' if gpu else '  (CPU)'}", flush=True)

    def _selftest(self):
        """작은 더미 입력으로 1회 추론 — GPU가 실제로 동작하는지 확인(자원부족/드라이버 문제 조기 검출)."""
        inn = self.sess.get_inputs()[0].name
        out = self.sess.get_outputs()[0].name
        x = np.zeros((1, self.n_in(), self.w + self.h), dtype=np.float32)
        self.sess.run([out], {inn: x})

    def _rebuild_cpu(self, reason=""):
        """런타임 중 GPU 추론이 실패하면 CPU 세션으로 교체(이후 안전하게 계속)."""
        import onnxruntime as ort
        self.sess = ort.InferenceSession(self._onnx_path, sess_options=self._so,
                                         providers=["CPUExecutionProvider"])
        self.active_provider = "CPUExecutionProvider"
        self.gpu_fallback_reason = reason or self.gpu_fallback_reason
        self._in = self.sess.get_inputs()[0].name
        self._out = self.sess.get_outputs()[0].name

    @staticmethod
    def _avail_mem_bytes():
        """OS 가용 물리 메모리(바이트). 못 구하면 None — 호출부는 보수 기본값 사용."""
        try:
            if os.name == "nt":
                import ctypes

                class _MS(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                                ("ullAvailExtendedVirtual", ctypes.c_uint64)]
                ms = _MS(); ms.dwLength = ctypes.sizeof(_MS)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                    # 물리와 커밋(페이지파일) 중 작은 쪽 — 커밋 한계 초과가 진짜 죽음의 원인
                    return int(min(ms.ullAvailPhys, ms.ullAvailPageFile))
            else:
                with open("/proc/meminfo") as f:
                    for ln in f:
                        if ln.startswith("MemAvailable:"):
                            return int(ln.split()[1]) * 1024
        except Exception:                                # noqa: BLE001
            pass
        return None

    def _mem_cap_blocks(self, fallback: int) -> int:
        """메모리 기준 배치 블록 상한.

        블록 하나 = n_in×(w+h) float32 입력 + n_out×(w+h) float32 출력 + ORT 내부 사본(~2배).
        가용 메모리의 20% 만 쓰도록 상한을 정한다 — 제어기는 설비 프로그램과 같이 돌므로
        검사가 메모리를 몰아 쓰면 안 된다. 못 구하면 fallback(기존 기본) 유지.
        """
        avail = self._avail_mem_bytes()
        if not avail:
            return fallback
        L = self.w + self.h
        per = (self.n_in() + len(self.sensor_cols)) * L * 4 * 4   # 입력+출력+내부사본 여유 4배
        cap = int(max(1, (avail * 0.20) // max(1, per)))
        return max(1, min(fallback, cap))

    def _run_mem_safe(self, X):
        """메모리 실패에 죽지 않는 실행 — 실패하면 배치를 반으로 쪼개 재귀 재시도.

        ★ 왜 필요한가 — ORT 네이티브 할당 실패는 파이썬 예외가 아니라 C++ bad_alloc
          abort 로 프로세스를 통째로 죽일 수 있고, 파이썬 쪽 MemoryError 로 나와도
          예전 코드는 그대로 전파돼 검사가 통째로 실패했다. 리소스가 모자라면
          죽는 게 아니라 '작게 나눠 느리게' 처리해야 한다(사용자 요구).
        단일 블록에서도 실패하면 MemoryError 로 올린다(호출부의 웨이퍼별 격리가 받음).
        """
        try:
            return self._run({self._in: X})
        except (MemoryError, RuntimeError, OSError) as e:
            low = str(e).lower()
            memlike = isinstance(e, MemoryError) or any(
                k in low for k in ("alloc", "memory", "enomem", "bad_alloc"))
            if not memlike:
                raise
            if X.shape[0] <= 1:
                raise MemoryError(f"메모리 부족: 블록 1개도 실행 불가 ({e})") from e
            h_ = X.shape[0] // 2
            print(f"[onnx] 메모리 부족 → 배치 {X.shape[0]}→{h_}+{X.shape[0]-h_} 로 분할 재시도",
                  flush=True)
            a = self._run_mem_safe(X[:h_])
            b = self._run_mem_safe(X[h_:])
            return np.concatenate([a, b], axis=0)

    def _run(self, feed):
        """추론 1회. ★ 모든 추론을 락으로 '직렬화' — DirectML/CUDA 세션은 여러 스레드가 동시에
        Run 하면 드라이버 레벨에서 크래시(앱 꺼짐)할 수 있다. 검사 중 상세보기·대시보드·자동검사가
        같은 세션을 동시에 돌려도 안전하게. (CPU EP 는 원래 동시안전이라 락 비용만 미미)
        GPU에서 실패(예외)하면 즉시 CPU로 전환 후 재시도 → 앱이 튕기지 않고 계속 동작."""
        with self._run_lock:
            try:
                return self.sess.run([self._out], feed)[0]
            except Exception as e:                               # noqa: BLE001
                if self.active_provider != "CPUExecutionProvider":
                    print(f"[onnx] GPU 추론 실패({e}) → CPU 로 전환 후 재시도", flush=True)
                    self._rebuild_cpu(reason=f"추론 실패: {e}")
                    return self.sess.run([self._out], feed)[0]
                raise

    def predict_batch(self, recs, pause_check=None) -> None:
        """여러 웨이퍼의 순전파를 '한 번의 session.run(들)' 으로 처리해 각 rec 에 결과를 캐시.

        웨이퍼별로 따로 추론하면 작은 run 이 웨이퍼 수만큼 반복돼 GPU/CPU 모두 손해다.
        모든 웨이퍼의 윈도우 블록을 B축으로 이어붙여 큰 배치 1회(또는 cap 단위 몇 회)로 실행 →
        각 rec['_pred_std_cache'] 채움. 이후 assess/archive 는 캐시를 그대로 사용(중복추론 0).
        결과는 웨이퍼별 _predict_std_full 과 수학적으로 동일. 실패해도 예외 없이 반환(웨이퍼별 폴백).

        pause_check: 각 배치 실행(session.run) 직전 호출되는 콜백. 일시중단이면 여기서
          블록(대기)해 다음 배치를 안 돌린다 → 일시중단 시 GPU 가 곧 유휴가 된다
          (현재 진행 중인 한 배치만 마치면 멈춤).
        """
        try:
            def _pc_ok(r):
                c = r.get("_pred_std_cache")
                return (isinstance(c, tuple) and len(c) == 3
                        and c[0] is r.get("sensors") and c[1] is r.get("io"))
            todo = [r for r in recs if isinstance(r, dict) and not _pc_ok(r)]
            if not todo:
                return
            _gpu = self.active_provider != "CPUExecutionProvider"
            # GPU 기본 배치는 작게(32). 모델이 커지면 256 배치의 순간 VRAM 이 16GB 를 넘겨
            #   Windows 가 시스템 RAM(공유 GPU 메모리)으로 스필 → PCIe 왕복으로 100배 느려져
            #   'GPU 3D 100% 인데 처리 0' 스래싱이 발생한다. IO_ORT_BATCH 로 상향/하향 조절.
            _defcap = 32 if _gpu else 2048
            cap = max(1, int(os.environ.get("IO_ORT_BATCH", str(_defcap)) or _defcap))
            if not _gpu:
                cap = self._mem_cap_blocks(cap)           # ★ 가용 메모리에 맞춰 자동 축소
            j = np.arange(self.w, self.w + self.h)
            buf_blocks: List = []
            buf_meta: List = []                                  # (rec, starts, T, b0)

            def _flush():
                if not buf_blocks:
                    return
                if pause_check is not None:                       # 일시중단이면 다음 배치 전에 대기(GPU 유휴화)
                    try:
                        pause_check()
                    except Exception:                            # noqa: BLE001
                        pass
                X = np.stack(buf_blocks).astype(np.float32)
                P = self._run_mem_safe(X)                # 메모리 부족 시 분할 재시도(죽지 않음)
                for rec, starts, T, b0 in buf_meta:
                    acc = np.zeros((len(self.sensor_cols), T), dtype=np.float64)
                    cnt = np.zeros(T, dtype=np.float64)
                    for k, a in enumerate(starts):
                        t = a - self.w + j
                        m = (t >= 0) & (t < T)
                        if m.any():
                            tm = t[m]
                            acc[:, tm] += P[b0 + k][:, j[m]]
                            cnt[tm] += 1
                    cnt[cnt < 1] = 1
                    try:
                        rec["_pred_std_cache"] = (rec.get("sensors"), rec.get("io"), acc / cnt)
                    except Exception:                            # noqa: BLE001
                        pass
                buf_blocks.clear()
                buf_meta.clear()

            for rec in todo:
                Xin = self._std_input_sensor(self._input_sensors(rec))
                io = self._normalize_io(rec["io"])
                T = Xin.shape[0]
                starts = self._starts(T)
                if buf_blocks and len(buf_blocks) + len(starts) > cap:   # cap 초과 전에 flush(메모리 제한)
                    _flush()
                b0 = len(buf_blocks)
                for a in starts:
                    buf_blocks.append(self._block_x(io, Xin, rec.get("step"), a, T))
                buf_meta.append((rec, starts, T, b0))
            _flush()
        except Exception:                                        # noqa: BLE001 (배치 실패 → 웨이퍼별 폴백)
            pass

    # ── 스키마 헬퍼 (TCNForecastModel 과 동일) ──
    def _predict_future_h(self) -> bool:
        return str(self.future_h_mode) == "predict"

    def _use_future_h_input(self) -> bool:
        return bool(self.h_indices) and not self._predict_future_h()

    @staticmethod
    def _record_sensors(rec, wanted_cols):
        arr = np.asarray(rec["sensors"], dtype=np.float64)
        wanted = list(wanted_cols)
        rec_cols = rec.get("sensor_cols")
        if rec_cols:
            idx = {str(c): i for i, c in enumerate(rec_cols)}
            out = np.zeros((arr.shape[0], len(wanted)), dtype=np.float64)
            for j, name in enumerate(wanted):
                src = idx.get(str(name))
                if src is not None and src < arr.shape[1]:
                    out[:, j] = arr[:, src]
            return out
        out = np.zeros((arr.shape[0], len(wanted)), dtype=np.float64)
        take = min(arr.shape[1], len(wanted))
        out[:, :take] = arr[:, :take]
        return out

    def actual(self, rec):
        return self._record_sensors(rec, self.sensor_cols)

    def _input_sensors(self, rec):
        validate_auxiliary_record(rec, self.cfg)
        return self._record_sensors(rec, self.input_sensor_cols)

    def _std_input_sensor(self, x):
        if self.normalization_mode == "minmax":
            return self._minmax(x, "input_sen")
        return np.nan_to_num((x - self.input_sen_mean) / self.input_sen_std,
                             nan=0.0, posinf=0.0, neginf=0.0)

    def _std_target_sensor(self, x):
        if self.normalization_mode == "minmax":
            return self._minmax(x, "sen")
        return np.nan_to_num((x - self.sen_mean) / self.sen_std,
                             nan=0.0, posinf=0.0, neginf=0.0)

    def _score_keep_mask(self, rec):
        T = int(rec["sensors"].shape[0])
        keep = np.ones(T, dtype=bool)
        if not bool(getattr(self.cfg, "score_exclude_boundary_step0", True)):
            return keep
        step = rec.get("step")
        if step is None or len(step) != T:
            return keep
        s = np.asarray(step).astype(int)
        keep[s == 0] = False
        return keep

    def n_in(self) -> int:
        return self._n_in

    def feature_names(self) -> List[str]:
        names = [f"IO[{c}]" for c in self.io_cols]
        names += [f"CTX[{c}]" for c in self.input_sensor_cols]
        names += ["past_context_mask"]
        if getattr(self.cfg, "use_step_onehot", True) and self.n_steps_global:
            names += [f"step={i}" for i in range(self.n_steps_global)]
        return names

    def _starts(self, T: int):
        s = min(max(1, int(self.stride)), max(1, int(self.h)))
        starts = list(range(0, T, s))
        if starts and starts[-1] < T - 1 and starts[-1] != T - self.h:
            starts.append(max(0, T - self.h))
        return starts or [0]

    def _block_x(self, io, Xstd, step, a: int, T: int):
        """입력 텐서 x 만 생성(추론용). TCNForecastModel._block 의 x 부분과 동일."""
        w, h, L = self.w, self.h, self.w + self.h
        n_io, n_in_sen = len(self.io_cols), len(self.input_sensor_cols)
        x = np.zeros((self.n_in(), L), dtype=np.float32)
        j = np.arange(L)
        t = a - w + j
        valid = (t >= 0) & (t < T)
        tv, jv = t[valid], j[valid]
        past = valid & (j < w)
        future = valid & (j >= w)
        tp, jp = t[past], j[past]
        tf, jf = t[future], j[future]
        if n_io and tv.size:
            x[:n_io, jv] = io[tv].T
        if n_in_sen and tp.size:
            x[n_io:n_io + n_in_sen, jp] = Xstd[tp].T
        if tf.size:
            for index in self.future_input_sensor_indices:
                x[n_io + index, jf] = Xstd[tf, index]
        x[n_io + n_in_sen, jp] = 1.0
        if getattr(self.cfg, "use_step_onehot", True) and step is not None \
                and self.n_steps_global and tv.size:
            base = n_io + n_in_sen + 1
            st = np.clip(np.asarray(step)[tv].astype(int), 0, self.n_steps_global - 1)
            x[base + st, jv] = 1.0
        return x

    def _predict_std_full(self, rec) -> np.ndarray:
        # ★ 순전파(ORT session.run) 결과를 rec 에 캐시. 한 웨이퍼는 assess(점수)와 archive(actual/
        #   expected/z)에서 predict/error_level 로 순전파를 여러 번 호출하므로, 캐시로 중복 추론 제거
        #   (검사 시간 ~30% 절감). 같은 입력→같은 출력이라 안전. rec 이 사라지면 캐시도 함께 해제.
        # ★ 캐시는 (sensors참조, io참조, 값) 튜플 — 입력 배열이 '같은 객체'일 때만 유효.
        #   RCA counterfactual 이 rec 사본의 sensors 를 바꿔도 예전 캐시가 반환돼
        #   '델타를 걸어도 예측이 안 변하는'(개선율 전부 0%) 버그가 있었음 → identity 검증으로 차단.
        _c = rec.get("_pred_std_cache")
        if (isinstance(_c, tuple) and len(_c) == 3
                and _c[0] is rec.get("sensors") and _c[1] is rec.get("io")):
            return _c[2]
        Xin = self._std_input_sensor(self._input_sensors(rec))
        io = self._normalize_io(rec["io"])
        T = Xin.shape[0]
        starts = self._starts(T)
        acc = np.zeros((len(self.sensor_cols), T), dtype=np.float64)
        cnt = np.zeros(T, dtype=np.float64)
        j = np.arange(self.w, self.w + self.h)
        # ★ 긴 웨이퍼(윈도우 수천 개)를 한 번에 stack 하면 입력+출력이 수백 MB 가 되어
        #   네이티브 할당 실패(abort)로 앱이 통째로 죽을 수 있다 → 메모리 상한 청크로 순회.
        _bs = self._mem_cap_blocks(2048)
        for ci in range(0, len(starts), _bs):
            chunk = starts[ci:ci + _bs]
            X = np.stack([self._block_x(io, Xin, rec.get("step"), a, T) for a in chunk]) \
                .astype(np.float32)
            P = self._run_mem_safe(X)                     # (B, n_out, L) — 부족 시 분할 재시도
            for b, a in enumerate(chunk):
                t = a - self.w + j
                m = (t >= 0) & (t < T)
                if m.any():
                    tm = t[m]
                    acc[:, tm] += P[b][:, j[m]]
                    cnt[tm] += 1
            del X, P
        cnt[cnt < 1] = 1
        out = acc / cnt
        try:
            rec["_pred_std_cache"] = (rec.get("sensors"), rec.get("io"), out)   # identity-키 캐시
        except Exception:                                         # noqa: BLE001 (rec 이 dict 가 아니면 무시)
            pass
        return out

    def predict(self, rec) -> np.ndarray:
        return self._inverse_target(self._predict_std_full(rec).T)

    def residual(self, rec) -> np.ndarray:
        return self.actual(rec) - self.predict(rec)

    def _limit_for_record(self, rec) -> np.ndarray:
        T = int(rec["sensors"].shape[0])
        lim = np.asarray(self.resid_abs_limit, dtype=np.float64)
        emin = float(getattr(self.cfg, "raw_error_min", 1e-9))
        if self._per_step_limit and rec.get("step") is not None and lim.ndim == 2:
            st = np.clip(np.asarray(rec["step"]).astype(int), 0, lim.shape[0] - 1)
            out = lim[st]
        elif lim.ndim == 1:
            out = np.repeat(lim[None, :], T, axis=0)
        else:
            out = np.ones((T, len(self.sensor_cols)), dtype=np.float64)
        out = np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)
        out[out < emin] = emin
        return out

    def _sigma_for_record(self, rec) -> np.ndarray:
        T = int(rec["sensors"].shape[0])
        ref = self.resid_sigma if self.resid_sigma is not None else self.resid_abs_limit
        sig = np.asarray(ref, dtype=np.float64)
        emin = float(getattr(self.cfg, "raw_error_min", 1e-9))
        if self._per_step_sigma and rec.get("step") is not None and sig.ndim == 2:
            st = np.clip(np.asarray(rec["step"]).astype(int), 0, sig.shape[0] - 1)
            out = sig[st]
        elif sig.ndim == 1:
            out = np.repeat(sig[None, :], T, axis=0)
        elif sig.ndim == 2 and sig.shape[0] == T:
            out = sig
        else:
            out = self._limit_for_record(rec)
        out = np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)
        out[out < emin] = emin
        return out

    def error_level(self, rec) -> np.ndarray:
        # ★ 웨이퍼당 1회 계산 캐시 — 검사 파이프라인이 assess/n_flag/배점/아카이브에서
        #   같은 rec 로 4회 호출하던 중복 연산 제거(결과 비트동일). 캐시 키에 sensors 배열의
        #   id 를 넣어, RCA counterfactual 처럼 sensors 를 바꾼 사본(rec copy)에서는 자동 재계산.
        #   (rec 은 청크 단위로 해제되므로 캐시도 함께 소멸 — 메모리 안전)
        try:
            _sarr = rec.get("sensors")
            _c = rec.get("_level_cache")
            if _c is not None and (_c[0] is _sarr):       # ★ 참조 동일성(is) — id 재사용 위험 없음
                return _c[1]
        except Exception:                                # noqa: BLE001
            _sarr = None
        level = np.abs(self.residual(rec)) / self._limit_for_record(rec)
        keep = self._score_keep_mask(rec)
        if keep is not None and len(keep) == level.shape[0]:
            level = level.copy(); level[~keep] = 0.0
        if _sarr is not None and isinstance(rec, dict):
            try:
                rec["_level_cache"] = (_sarr, level)      # 배열 참조를 키로 보관(GC id 재사용 차단)
            except Exception:                            # noqa: BLE001
                pass
        return level

    def z_residual(self, rec) -> np.ndarray:
        signed = self.residual(rec) / self._limit_for_record(rec)
        keep = self._score_keep_mask(rec)
        if keep is not None and len(keep) == signed.shape[0]:
            signed = signed.copy(); signed[~keep] = 0.0
        return signed

    def sigma_residual(self, rec) -> np.ndarray:
        signed = self.residual(rec) / self._sigma_for_record(rec)
        keep = self._score_keep_mask(rec)
        if keep is not None and len(keep) == signed.shape[0]:
            signed = signed.copy(); signed[~keep] = 0.0
        return signed

    def residual_batch(self, recs):
        return [self.residual(r) for r in recs]

    def assess(self, rec):
        """TCNForecastModel.assess 와 동일 — {wafer_id, score, sensors:[...]}."""
        level = graph_adjust_abs_z(self.error_level(rec), self.sensor_cols, self.cfg)
        per = level.max(axis=0) if level.size else np.zeros(len(self.sensor_cols))
        order = np.argsort(per)[::-1]
        actual = self.actual(rec); pred = self.predict(rec)
        sensors = []
        for j in order[:5]:
            t = int(np.argmax(level[:, j])) if level.size else 0
            sensors.append({
                "sensor": self.sensor_cols[j],
                "peak_step": int(rec["step"][t]) if rec.get("step") is not None else t,
                "error_level": round(float(level[t, j]), 3),
                "dir": "높음" if actual[t, j] > pred[t, j] else "낮음",
                "raw_value": round(float(actual[t, j]), 5),
                "expected_value": round(float(pred[t, j]), 5),
                "raw_abs_error": round(float(abs(actual[t, j] - pred[t, j])), 5),
            })
        return {"wafer_id": rec["wafer_id"], "score": round(self._score_record(rec), 6),
                "sensors": sensors}

    def n_flag_sensors(self, rec, thresh: float = 1.0) -> int:
        """이상 센서 수 = peak error_level 이 thresh(=OK tolerance 배수) 이상인 타깃 센서 수."""
        lvl = self.error_level(rec)
        if lvl.size == 0:
            return 0
        return int((lvl.max(axis=0) >= float(thresh)).sum())

    def _score_record(self, rec) -> float:
        # gh 모델과 동일: 전체 센서·시간 error_level 의 합(=웨이퍼 복원오차 누적)
        level = self.error_level(rec)
        if level.size == 0:
            return 0.0
        level = np.nan_to_num(np.asarray(level, dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
        score = float(np.maximum(0.0, level).sum())
        # ★ 학습이 --score-time-avg / --score-final-avg 로 임계를 잡았으면 같은 단위로 맞춘다.
        #   (train_jw v0.95+ 는 누적합을 '유효 스텝 수(step≠0)'로 나눈 값으로 임계를 산정한다.
        #    안 나누면 점수가 스텝 수만큼 커져 전 웨이퍼가 NG 가 된다. ONNX 메타엔 이 플래그가
        #    없어 앱이 search_best.json 에서 읽어 self.score_norm 으로 주입한다.)
        if str(getattr(self, "score_norm", "none")) in ("time", "final"):
            score /= float(max(1, self._valid_step_count(rec)))
        return score

    def _valid_step_count(self, rec) -> int:
        """점수 계산에 실제 쓰인 구간 길이(step≠0) — 학습쪽 분모와 동일 정의."""
        try:
            keep = self._score_keep_mask(rec)
            return int(np.asarray(keep, dtype=bool).sum())
        except Exception:                                # noqa: BLE001
            try:
                return int(np.asarray(rec["sensors"]).shape[0])
            except Exception:                            # noqa: BLE001
                return 1

    def score_records(self, recs):
        return np.asarray([self._score_record(r) for r in recs], dtype=np.float64)

    # ── 로더 ──
    @staticmethod
    def find_onnx(art_dir: Path):
        art_dir = Path(art_dir)
        cand = art_dir / "tcn_forecast.onnx"
        onnx = cand if cand.exists() else next(iter(sorted(art_dir.glob("*.onnx"))), None)
        if onnx is None:
            return None, None
        for m in (onnx.with_suffix(".npz"), art_dir / "recon_meta.npz"):
            if m.exists():
                return onnx, m
        return onnx, None  # unrelated NPZ (e.g. OK bands) is not model metadata

    @classmethod
    def load(cls, art_dir: Path, cfg: IOConfig) -> "OnnxForecastModel":
        onnx, meta = cls.find_onnx(art_dir)
        if onnx is None:
            raise FileNotFoundError(f"{art_dir} 에 .onnx 모델이 없습니다.")
        if meta is None:
            raise FileNotFoundError(
                f"{onnx.name} 에 짝 메타(.npz)가 없습니다. export_onnx_forecast 로 함께 생성하세요.")
        return cls(onnx, meta, cfg)

    @staticmethod
    def available(art_dir: Path) -> bool:
        onnx, meta = OnnxForecastModel.find_onnx(art_dir)
        return onnx is not None and meta is not None
