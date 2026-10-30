"""TCN future forecaster for OK-only anomaly detection.

Input:
  future_h_mode=input:   past T/P/H/IO + future H/IO -> future T/P
  future_h_mode=predict: past T/P/H/IO + future IO   -> future T/P/H
  sensor_role_profile=h-split-v1:
                         past T/P/target-H/input-H/IO + future input-H/IO
                         -> future T/P/target-H
  virtual_sensor_mode=both: past all+virtual+IO + future H/IO -> future T/P+virtual
  without H columns:     past T/P/IO   + future IO   -> future T/P

The future target sensors are masked from the input, so the network must infer
normal future T/P behavior from the past trajectory and actuator/control state.
"""
from __future__ import annotations


import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .sensor_schema import in_group, paired_zone
from .auxiliary_sensors import (auxiliary_sensor_columns, auxiliary_sensor_metadata,
                                configure_auxiliary_model, validate_auxiliary_record)

from .config import (
    h_SPLIT_CART_COUNT,
    h_SPLIT_HEATER_BLOCK_COUNT,
    h_SPLIT_HEATER_BLOCK_SENSORS,
    h_SPLIT_RESERVOIR_TARGET_SENSORS,
    h_SPLIT_VESSEL_HEATER_COUNT,
    IOConfig,
    SENSOR_ROLE_PROFILE_H_SPLIT_V1,
    h_sensor_keep_for_config,
    normalize_sensor_role_profile,
)
from .sensor_graph import graph_adjust_abs_z
from .tcn import IOTCN


def _score_agg_enabled(cfg: IOConfig = None) -> bool:
    return bool(getattr(cfg, "score_agg_enabled", False)) if cfg is not None else False


def _score_resid_agg(cfg: IOConfig = None) -> str:
    if not _score_agg_enabled(cfg):
        return "sum"
    mode = str(getattr(cfg, "resid_agg", "sum") or "sum").strip().lower()
    return mode if mode in {"sum", "p99", "max", "mean"} else "sum"


def _score_wafer_agg(cfg: IOConfig = None) -> str:
    if not _score_agg_enabled(cfg):
        return "sum"
    mode = str(getattr(cfg, "wafer_agg", "sum") or "sum").strip().lower()
    return mode if mode in {"sum", "topk_mean", "max", "mean"} else "sum"


def _score_topk(cfg: IOConfig = None) -> int:
    try:
        return max(0, int(getattr(cfg, "wafer_topk", 0) or 0))
    except Exception:
        return 0


def _score_time_avg(cfg: IOConfig = None) -> bool:
    return bool(getattr(cfg, "score_time_avg", False)) if cfg is not None else False


def _score_final_avg(cfg: IOConfig = None) -> bool:
    return bool(getattr(cfg, "score_final_avg", False)) if cfg is not None else False


def _score_time_denominator(level, valid_steps=None) -> float:
    if valid_steps is not None:
        if np.isscalar(valid_steps):
            try:
                value = int(valid_steps)
            except Exception:
                value = 0
        else:
            try:
                value = int(np.asarray(valid_steps, dtype=bool).sum())
            except Exception:
                value = 0
    else:
        arr = np.asarray(level)
        value = int(arr.shape[0]) if arr.ndim else 0
    return float(max(1, value))


def _per_sensor_reconstruction_scores(level, resid_agg: str = "sum") -> np.ndarray:
    level = np.asarray(level, dtype=np.float64)
    if level.ndim != 2 or level.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    level = np.nan_to_num(np.maximum(0.0, level), nan=0.0, posinf=0.0, neginf=0.0)
    resid_agg = str(resid_agg or "sum").strip().lower()
    if resid_agg == "p99":
        return np.percentile(level, 99.0, axis=0)
    if resid_agg == "max":
        return np.max(level, axis=0)
    if resid_agg == "mean":
        return np.mean(level, axis=0)
    return np.sum(level, axis=0)


def _aggregate_sensor_scores(per_sensor, wafer_agg: str = "sum", topk: int = 0) -> float:
    per_sensor = np.asarray(per_sensor, dtype=np.float64)
    if per_sensor.size == 0:
        return 0.0
    per_sensor = np.nan_to_num(per_sensor, nan=0.0, posinf=0.0, neginf=0.0)
    if int(topk or 0) > 0 and int(topk) < per_sensor.size:
        order = np.argsort(per_sensor)[::-1][:int(topk)]
        selected = per_sensor[order]
    else:
        selected = per_sensor
    if selected.size == 0:
        return 0.0
    wafer_agg = str(wafer_agg or "sum").strip().lower()
    if wafer_agg in {"topk_mean", "mean"}:
        return float(np.mean(selected))
    if wafer_agg == "max":
        return float(np.max(selected))
    return float(np.sum(selected))


def _reconstruction_score_from_level(level, cfg: IOConfig = None, valid_steps=None) -> float:
    per_sensor = _per_sensor_reconstruction_scores(level, _score_resid_agg(cfg))
    score = _aggregate_sensor_scores(per_sensor, _score_wafer_agg(cfg), _score_topk(cfg))
    if _score_time_avg(cfg) or _score_final_avg(cfg):
        score /= _score_time_denominator(level, valid_steps=valid_steps)
    return float(score)


class TCNForecastModel:
    def __init__(self, cfg: IOConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        self.use_amp = bool(getattr(cfg, "tcn_amp", True)) and self.device_type == "cuda"
        self.net: Optional[IOTCN] = None
        self.io_cols = []
        self.input_sensor_cols = []
        self.sensor_cols = []
        self.target_indices = []
        self.h_indices = []
        self.future_h_indices = []
        self.n_steps_global = 0
        self.schema_mode = ""
        self.input_sen_mean = self.input_sen_std = None
        self.sen_mean = self.sen_std = None
        self.resid_abs_limit = None
        self.resid_std = None
        self.resid_sigma = None
        self._per_step_limit = False
        self._per_step_sigma = False
        self.w = int(getattr(cfg, "tcn_past_window", 64))
        self.h = int(getattr(cfg, "tcn_future_window", 64))
        self.stride = int(getattr(cfg, "tcn_stride", 0)) or self.h
        self.arch = {}

    @staticmethod
    def _matches_prefix(name: str, prefixes) -> bool:
        prefixes = tuple(str(p) for p in (prefixes or ()) if str(p))
        return bool(prefixes) and str(name).strip().startswith(prefixes)

    def _is_h_sensor(self, name: str) -> bool:
        text = str(name).strip()
        tail = re.split(r"[-\s]+", text)[-1]
        if re.match(r"^H\d+$", tail, re.IGNORECASE):
            return True
        desc = re.sub(r"[\s\-]+[A-Za-z]{1,2}\d+\s*$", "", text).strip().lower()
        exact = {str(x).strip().lower() for x in
                 getattr(self.cfg, "active_control_sensor_names", ()) or ()}
        if text.lower() in exact or desc in exact:
            return True
        if self._matches_prefix(text, getattr(self.cfg, "active_control_sensor_prefixes", ())):
            return True
        low = text.lower()
        keywords = tuple(str(k).lower() for k in
                         getattr(self.cfg, "active_control_sensor_keywords", ("heat", "heater"))
                         if str(k))
        if any(k in low for k in keywords):
            return True
        pattern = str(getattr(self.cfg, "active_control_sensor_regex", "") or "")
        return bool(pattern and re.search(pattern, text, re.IGNORECASE))

    def _h_sensor_allowed(self, name: str) -> bool:
        keep = {str(x).strip() for x in h_sensor_keep_for_config(self.cfg)}
        return not keep or str(name).strip() in keep

    def _is_target_sensor(self, name: str) -> bool:
        return self._matches_prefix(name, getattr(self.cfg, "target_sensor_prefixes", ("p_", "t_")))

    def _sensor_role_profile(self) -> str:
        mode = self.arch.get("sensor_role_profile") if isinstance(self.arch, dict) else None
        if not mode:
            mode = getattr(self.cfg, "sensor_role_profile", "")
        return normalize_sensor_role_profile(mode)

    def _is_h_split_profile(self) -> bool:
        return self._sensor_role_profile() == SENSOR_ROLE_PROFILE_H_SPLIT_V1

    @staticmethod
    def _is_cart_sensor(name: str) -> bool:
        return in_group(name, "cart")

    @staticmethod
    def _is_h_split_heater_block_sensor(name: str) -> bool:
        return str(name).strip() in set(h_SPLIT_HEATER_BLOCK_SENSORS)

    @staticmethod
    def _is_h_split_reservoir_target_sensor(name: str) -> bool:
        return str(name).strip() in set(h_SPLIT_RESERVOIR_TARGET_SENSORS)

    @staticmethod
    def _is_h_split_vessel_heater_sensor(name: str) -> bool:
        return in_group(name, "response")

    @staticmethod
    def _validate_h_split_count(label: str, cols, expected: int) -> None:
        cols = [str(c) for c in cols]
        if len(cols) == int(expected):
            return
        shown = ", ".join(cols) if cols else "(none)"
        raise ValueError(
            f"--sensor-role-profile h-split-v1 requires exactly {expected} "
            f"{label} sensors, but found {len(cols)}: {shown}"
        )

    def _virtual_sensor_mode(self) -> str:
        mode = self.arch.get("virtual_sensor_mode") if isinstance(self.arch, dict) else None
        if not mode:
            mode = getattr(self.cfg, "virtual_sensor_mode", "off")
        mode = str(mode or "off").lower()
        return mode if mode in {"input", "target", "both"} else "off"

    def _is_virtual_sensor(self, name: str) -> bool:
        return str(name).strip() in {
            str(x) for x in getattr(self.cfg, "virtual_sensor_names", ()) or ()
        }

    def _predict_future_h(self) -> bool:
        mode = self.arch.get("future_h_mode") if isinstance(self.arch, dict) else None
        if not mode:
            mode = getattr(self.cfg, "future_h_mode", "input")
        return str(mode or "input").lower() == "predict"

    def _use_future_h_input(self) -> bool:
        future_h_indices = getattr(self, "future_h_indices", None)
        if future_h_indices is None:
            future_h_indices = self.h_indices
        return bool(future_h_indices) and not self._predict_future_h()

    def _setup_h_split_sensor_schema(self, sensor_cols, virtual_mode: str) -> None:
        if str(getattr(self.cfg, "h_cart_mode", "") or "").lower() == "exclude":
            raise ValueError(
                "--sensor-role-profile h-split-v1은 Cart류 10종을 미래 입력으로 "
                "사용하므로 --h-cart-mode exclude와 함께 사용할 수 없습니다."
            )
        all_cols = list(sensor_cols)
        heater_block_cols = [
            c for c in all_cols if self._is_h_split_heater_block_sensor(c)
        ]
        cart_cols = [c for c in all_cols if self._is_cart_sensor(c)]
        reservoir_cols = [
            c for c in all_cols if self._is_h_split_reservoir_target_sensor(c)
        ]
        vessel_heater_cols = [
            c for c in all_cols if self._is_h_split_vessel_heater_sensor(c)
        ]

        self._validate_h_split_count(
            "h_005", heater_block_cols, h_SPLIT_HEATER_BLOCK_COUNT)
        expected_heater_blocks = set(h_SPLIT_HEATER_BLOCK_SENSORS)
        found_heater_blocks = {str(c).strip() for c in heater_block_cols}
        if found_heater_blocks != expected_heater_blocks:
            raise ValueError(
                "--sensor-role-profile h-split-v1 h_005 sensors mismatch: "
                f"expected {sorted(expected_heater_blocks)}, "
                f"found {sorted(found_heater_blocks)}"
            )
        self._validate_h_split_count("Cart", cart_cols, h_SPLIT_CART_COUNT)
        self._validate_h_split_count(
            "Reservoir target", reservoir_cols,
            len(h_SPLIT_RESERVOIR_TARGET_SENSORS))
        self._validate_h_split_count(
            "Vessel heater target", vessel_heater_cols,
            h_SPLIT_VESSEL_HEATER_COUNT)

        future_h_names = {str(c).strip() for c in heater_block_cols + cart_cols}
        target_h_names = {
            str(c).strip() for c in reservoir_cols + vessel_heater_cols
        }
        input_virtual = virtual_mode in {"input", "both"}
        target_virtual = virtual_mode != "off"
        input_names = set(future_h_names) | set(target_h_names)
        target_names = set(target_h_names)
        target_cols = []
        for c in all_cols:
            name = str(c).strip()
            is_pt_target = self._is_target_sensor(c) and name not in future_h_names
            if is_pt_target or name in target_names or (
                target_virtual and self._is_virtual_sensor(c)
            ):
                target_cols.append(c)
            if is_pt_target:
                input_names.add(name)
        self.input_sensor_cols = [
            c for c in all_cols
            if str(c).strip() in input_names or (
                input_virtual and self._is_virtual_sensor(c)
            )
        ]
        self.target_indices = [all_cols.index(c) for c in target_cols]
        self.sensor_cols = list(target_cols)
        self.h_indices = [
            i for i, c in enumerate(self.input_sensor_cols)
            if self._is_h_sensor(c) or self._is_cart_sensor(c)
        ]
        self.future_h_indices = [
            i for i, c in enumerate(self.input_sensor_cols)
            if str(c).strip() in future_h_names
        ]
        if not self.sensor_cols:
            raise ValueError("No h-split-v1 target sensors selected.")
        suffix = "+virtual" if target_virtual else ""
        self.schema_mode = (
            "h-split-v1: past T/P/target-H/input-H"
            f"{suffix}+IO, future h_005/Cart+IO -> "
            f"future T/P/Reservoir/VesselH{suffix}"
        )

    def _setup_sensor_schema(self, sensor_cols) -> None:
        virtual_mode = self._virtual_sensor_mode()
        if self._is_h_split_profile():
            self._setup_h_split_sensor_schema(sensor_cols, virtual_mode)
            self._setup_auxiliary_schema(sensor_cols)
            return
        all_cols = [
            c for c in list(sensor_cols)
            if not self._is_h_sensor(c) or self._h_sensor_allowed(c)
        ]
        input_virtual = virtual_mode in {"input", "both"}
        self.input_sensor_cols = [
            c for c in all_cols
            if input_virtual or not self._is_virtual_sensor(c)
        ]
        self.h_indices = [i for i, c in enumerate(self.input_sensor_cols) if self._is_h_sensor(c)]
        self.future_h_indices = list(self.h_indices)
        target_idx = [i for i, c in enumerate(all_cols) if self._is_target_sensor(c) and c not in auxiliary_sensor_columns(self.cfg)]
        if self._predict_future_h():
            target_idx = sorted(set(target_idx) | {
                i for i, c in enumerate(all_cols) if self._is_h_sensor(c)
            })
        if virtual_mode != "off":
            target_idx = sorted(set(target_idx) | {
                i for i, c in enumerate(all_cols) if self._is_virtual_sensor(c)
            })
        if not target_idx:
            input_h = {str(self.input_sensor_cols[i]) for i in self.h_indices}
            target_idx = [i for i, c in enumerate(all_cols) if str(c) not in input_h and c not in auxiliary_sensor_columns(self.cfg)]
        if not target_idx and not auxiliary_sensor_columns(self.cfg, "predict"):
            raise ValueError("No T/P target sensors selected.")
        self.target_indices = [int(i) for i in target_idx]
        self.sensor_cols = [all_cols[i] for i in self.target_indices]
        suffix = "+virtual" if virtual_mode != "off" else ""
        if self.h_indices and self._predict_future_h():
            self.schema_mode = f"past T/P/H{suffix}+IO, future IO -> future T/P/H{suffix}"
        elif self.h_indices:
            self.schema_mode = f"past T/P/H{suffix}+IO, future H+IO -> future T/P{suffix}"
        else:
            self.schema_mode = f"past T/P{suffix}+IO, future IO -> future T/P{suffix}"
        self._setup_auxiliary_schema(sensor_cols)

    def _setup_auxiliary_schema(self, sensor_cols) -> None:
        columns = auxiliary_sensor_columns(self.cfg)
        future_cols = auxiliary_sensor_columns(self.cfg, "input")
        target_cols = auxiliary_sensor_columns(self.cfg, "predict")
        for name in columns:
            if name not in sensor_cols:
                raise ValueError(f"Missing required auxiliary sensor: {name}")
            if name not in self.input_sensor_cols:
                self.input_sensor_cols.append(name)
        self.sensor_cols = [c for c in self.sensor_cols if c not in columns]
        self.sensor_cols.extend(c for c in target_cols if c not in self.sensor_cols)
        if columns:
            self.target_indices = [list(sensor_cols).index(c) for c in self.sensor_cols]
        self.future_context_indices = [self.input_sensor_cols.index(c) for c in future_cols]
        if columns:
            self.schema_mode += ("; future context=" + ",".join(future_cols)
                                 + "; auxiliary targets=" + ",".join(target_cols))

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

    def actual(self, rec) -> np.ndarray:
        return self._record_sensors(rec, self.sensor_cols)

    def _input_sensors(self, rec) -> np.ndarray:
        validate_auxiliary_record(rec, self.cfg)
        return self._record_sensors(rec, self.input_sensor_cols)

    def _std_input_sensor(self, x):
        return np.nan_to_num((x - self.input_sen_mean) / self.input_sen_std,
                             nan=0.0, posinf=0.0, neginf=0.0)

    def _std_target_sensor(self, x):
        return np.nan_to_num((x - self.sen_mean) / self.sen_std,
                             nan=0.0, posinf=0.0, neginf=0.0)

    def _score_keep_mask(self, rec) -> np.ndarray:
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
        n_step = self.n_steps_global if getattr(self.cfg, "use_step_onehot", True) else 0
        return len(self.io_cols) + len(self.input_sensor_cols) + 1 + n_step

    def feature_names(self):
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

    def _block(self, io, Xstd, Ystd, step, keep_mask, a: int, T: int):
        w, h, L = self.w, self.h, self.w + self.h
        n_io, n_in_sen, n_out = len(self.io_cols), len(self.input_sensor_cols), len(self.sensor_cols)
        x = np.zeros((self.n_in(), L), dtype=np.float32)
        y = np.zeros((n_out, L), dtype=np.float32)
        m = np.zeros(L, dtype=np.float32)

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
        if self._use_future_h_input() and tf.size:
            for hi in getattr(self, "future_h_indices", self.h_indices):
                x[n_io + hi, jf] = Xstd[tf, hi]
        if tf.size:
            for index in getattr(self, "future_context_indices", []):
                x[n_io + index, jf] = Xstd[tf, index]
        x[n_io + n_in_sen, jp] = 1.0
        if getattr(self.cfg, "use_step_onehot", True) and step is not None and self.n_steps_global and tv.size:
            base = n_io + n_in_sen + 1
            st = np.clip(np.asarray(step)[tv].astype(int), 0, self.n_steps_global - 1)
            x[base + st, jv] = 1.0
        if n_out and tf.size:
            y[:, jf] = Ystd[tf].T
            m[jf] = keep_mask[tf].astype(np.float64)
        return x, y, m[None]

    def _build_samples(self, records):
        Xs, Ys, Ms = [], [], []
        for r in records:
            Xin = self._std_input_sensor(self._input_sensors(r))
            Yin = self._std_target_sensor(self.actual(r))
            io = np.asarray(r["io"], dtype=np.float64)
            T = Xin.shape[0]
            keep = self._score_keep_mask(r)
            for a in self._starts(T):
                x, y, m = self._block(io, Xin, Yin, r.get("step"), keep, a, T)
                if float(m.sum()) <= 0:
                    continue
                Xs.append(x)
                Ys.append(y)
                Ms.append(m)
        if not Xs:
            raise ValueError("No trainable TCN windows. Check ProcessStep 0 exclusion and window sizes.")
        def to(arr):
            stacked = np.stack(arr).astype(np.float32, copy=False)
            return torch.from_numpy(stacked)
        return to(Xs), to(Ys), to(Ms)

    def _setup_torch_runtime(self):
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        if self.device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = bool(
                getattr(self.cfg, "tcn_cudnn_benchmark", True))

    def _tensor_bytes(self, *tensors) -> int:
        return int(sum(t.numel() * t.element_size() for t in tensors if t is not None))

    def _gpu_cache_limit_bytes(self) -> int:
        cfg_limit = int(float(getattr(self.cfg, "tcn_cache_max_gb", 24.0)) * (1024 ** 3))
        if self.device_type != "cuda":
            return cfg_limit
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            return min(cfg_limit, int(free_bytes * 0.65))
        except Exception:
            return cfg_limit

    def _pin_if_cuda(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device_type != "cuda" or tensor.is_cuda:
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor

    def _move_batch(self, tensor: torch.Tensor, index=None) -> torch.Tensor:
        batch = tensor[index] if index is not None else tensor
        if batch.device.type == self.device_type:
            return batch
        return batch.to(self.device, non_blocking=(self.device_type == "cuda"))

    def _device_batch_from_arrays(self, arrays) -> torch.Tensor:
        arr = np.stack(arrays).astype(np.float32, copy=False)
        xt = torch.from_numpy(arr)
        xt = self._pin_if_cuda(xt)
        return xt.to(self.device, non_blocking=(self.device_type == "cuda"))

    def fit(self, ok_records, io_cols, sensor_cols, n_steps_global, val_records=None):
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device_type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self._setup_torch_runtime()
        self.io_cols = list(io_cols)
        self._setup_sensor_schema(sensor_cols)
        self.n_steps_global = int(n_steps_global or 0)
        self.w = max(1, int(getattr(self.cfg, "tcn_past_window", 64)))
        self.h = max(1, int(getattr(self.cfg, "tcn_future_window", 64)))
        self.stride = min(max(1, int(getattr(self.cfg, "tcn_stride", 0)) or self.h), self.h)
        self.arch = {
            **auxiliary_sensor_metadata(self.cfg),
            "past_window": self.w,
            "future_window": self.h,
            "stride": self.stride,
            "channels": int(getattr(self.cfg, "tcn_channels", 64)),
            "blocks": int(getattr(self.cfg, "tcn_blocks", 5)),
            "kernel": int(getattr(self.cfg, "tcn_kernel", 3)),
            "dropout": float(getattr(self.cfg, "tcn_dropout", 0.1)),
            "file_format": str(getattr(self.cfg, "file_format", "csv") or "csv"),
            "h_cart_mode": str(getattr(self.cfg, "h_cart_mode", "")),
            "h_sensor_preset": str(getattr(self.cfg, "h_sensor_preset", "") or ""),
            "h_sensor_keep": list(h_sensor_keep_for_config(self.cfg)),
            "sensor_role_profile": normalize_sensor_role_profile(
                getattr(self.cfg, "sensor_role_profile", "")),
            "future_h_mode": str(getattr(self.cfg, "future_h_mode", "input") or "input"),
            "virtual_sensor_mode": str(getattr(self.cfg, "virtual_sensor_mode", "off") or "off"),
        }

        all_in = np.concatenate([self._input_sensors(r) for r in ok_records], axis=0)
        self.input_sen_mean = np.nanmean(all_in, axis=0)
        self.input_sen_std = np.nanstd(all_in, axis=0)
        self.input_sen_std[self.input_sen_std < 1e-9] = 1.0
        all_target = np.concatenate([self.actual(r) for r in ok_records], axis=0)
        self.sen_mean = np.nanmean(all_target, axis=0)
        self.sen_std = np.nanstd(all_target, axis=0)
        self.sen_std[self.sen_std < 1e-9] = 1.0

        cart_h = sum(1 for i in self.h_indices if in_group(self.input_sensor_cols[i], "cart"))
        future_h = len(getattr(self, "future_h_indices", self.h_indices))
        target_h = sum(1 for name in self.sensor_cols if self._is_h_sensor(name))
        target_virtual = sum(1 for name in self.sensor_cols if self._is_virtual_sensor(name))
        input_virtual = sum(1 for name in self.input_sensor_cols if self._is_virtual_sensor(name))
        h_keep_label = (
            f" | h_sensor_keep={len(self.arch.get('h_sensor_keep') or [])}"
            if self.arch.get("h_sensor_keep") else ""
        )
        print(f"  [tcn-schema] mode={self.schema_mode} | h_cart_mode={self.arch['h_cart_mode']} "
              f"| file_format={self.arch.get('file_format', 'csv')} "
              f"| h_sensor_preset={self.arch.get('h_sensor_preset', '') or 'all'}{h_keep_label} "
              f"| sensor_role_profile={self.arch.get('sensor_role_profile', 'original')} "
              f"| future_h_mode={self.arch['future_h_mode']} "
              f"| virtual_sensor_mode={self.arch['virtual_sensor_mode']} "
              f"| OK wafer={len(ok_records)} | IO={len(self.io_cols)} "
              f"| input_sensor={len(self.input_sensor_cols)} | h_input={len(self.h_indices)} "
              f"(future H {future_h}, Cart H {cart_h}, virtual {input_virtual}) "
              f"| target_sensor={len(self.sensor_cols)} "
              f"(h_target {target_h}, virtual_target {target_virtual}) | steps={self.n_steps_global}",
              flush=True)

        t0 = time.perf_counter()
        print("  [tcn-runtime] build train windows on CPU...", flush=True)
        X, Y, M = self._build_samples(ok_records)
        print(f"  [tcn-runtime] build windows done in {time.perf_counter() - t0:.1f}s",
              flush=True)
        n = X.shape[0]
        self.net = IOTCN(self.n_in(), len(self.sensor_cols), self.arch["channels"],
                         self.arch["blocks"], self.arch["kernel"],
                         self.arch["dropout"]).to(self.device)
        print(f"  [tcn] past={self.w} future={self.h} stride={self.stride} "
              f"| input {self.n_in()}ch -> target {len(self.sensor_cols)}ch "
              f"| receptive≈{self.net.receptive_field} need={self.w + self.h} "
              f"| device={self.device}", flush=True)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_val = max(1, int(n * float(getattr(self.cfg, "tcn_val_frac", 0.15))))
        vi = torch.as_tensor(idx[:n_val])
        ti = torch.as_tensor(idx[n_val:])
        if len(ti) == 0:
            ti = vi
        Xtr, Ytr, Mtr = X[ti], Y[ti], M[ti]
        Xva, Yva, Mva = X[vi], Y[vi], M[vi]
        train_bytes = self._tensor_bytes(Xtr, Ytr, Mtr, Xva, Yva, Mva)
        cache_limit = self._gpu_cache_limit_bytes()
        cache_on_gpu = (
            self.device_type == "cuda"
            and bool(getattr(self.cfg, "tcn_cache_train_on_gpu", True))
            and train_bytes <= cache_limit
        )
        if cache_on_gpu:
            Xtr = Xtr.to(self.device, non_blocking=True)
            Ytr = Ytr.to(self.device, non_blocking=True)
            Mtr = Mtr.to(self.device, non_blocking=True)
            Xva = Xva.to(self.device, non_blocking=True)
            Yva = Yva.to(self.device, non_blocking=True)
            Mva = Mva.to(self.device, non_blocking=True)
            print(f"  [tcn-runtime] train windows={n:,} | cache=gpu "
                  f"({train_bytes / (1024 ** 3):.2f} GiB, "
                  f"gpu_limit={cache_limit / (1024 ** 3):.2f} GiB)", flush=True)
        else:
            Xtr, Ytr, Mtr = self._pin_if_cuda(Xtr), self._pin_if_cuda(Ytr), self._pin_if_cuda(Mtr)
            Xva, Yva, Mva = self._pin_if_cuda(Xva), self._pin_if_cuda(Yva), self._pin_if_cuda(Mva)
            cache_label = "pinned-cpu" if self.device_type == "cuda" else "cpu"
            print(f"  [tcn-runtime] train windows={n:,} | cache={cache_label} "
                  f"({train_bytes / (1024 ** 3):.2f} GiB, "
                  f"gpu_limit={cache_limit / (1024 ** 3):.2f} GiB)", flush=True)
        bs = max(1, int(getattr(self.cfg, "tcn_batch", 256)))
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(getattr(self.cfg, "tcn_lr", 1e-3)),
                                weight_decay=float(getattr(self.cfg, "tcn_weight_decay", 1e-4)))
        best, best_state, patience = float("inf"), None, 0
        epochs = int(getattr(self.cfg, "tcn_epochs", 60))
        done_epochs = 0
        for ep in range(epochs):
            done_epochs = ep + 1
            self.net.train()
            perm = torch.randperm(Xtr.shape[0], device=Xtr.device)
            for i in range(0, Xtr.shape[0], bs):
                bi = perm[i:i + bs]
                xb = self._move_batch(Xtr, bi)
                yb = self._move_batch(Ytr, bi)
                mb = self._move_batch(Mtr, bi)
                opt.zero_grad()
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16,
                                    enabled=self.use_amp):
                    pred = self.net(xb)
                    loss = (((pred - yb) ** 2) * mb).sum() / mb.sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                self.net.eval()
                tot, cnt = 0.0, 0.0
                for i in range(0, Xva.shape[0], bs):
                    sl = slice(i, i + bs)
                    xb = self._move_batch(Xva, sl)
                    yb = self._move_batch(Yva, sl)
                    mb = self._move_batch(Mva, sl)
                    with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16,
                                        enabled=self.use_amp):
                        pv = self.net(xb)
                    tot += float((((pv.float() - yb) ** 2) * mb).sum())
                    cnt += float(mb.sum())
                vloss = tot / max(cnt, 1.0)
            if vloss < best - 1e-6:
                best = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                patience = 0
            else:
                patience += 1
            import os as _os                              # 앱 현황판용: 로그 주기 env 조절(기본 5 — 동작 불변)
            if (ep + 1) % max(1, int(_os.environ.get("IO_TCN_LOG_EVERY", "5") or 5)) == 0 or ep == 0:
                # pid/dev 를 함께 남긴다 — 멀티 GPU(워커 여러 개)에서 로그가 한 파일에 섞여도
                # 앱이 'pid → 워커(GPU·trial)' 로 되짚어 GPU별 loss 곡선을 그릴 수 있다.
                print(f"  [tcn] epoch {ep + 1}/{epochs} val_mse={vloss:.6f} "
                      f"pid={_os.getpid()} dev={self.device}", flush=True)
            if patience >= int(getattr(self.cfg, "tcn_patience", 8)):
                break
        if best_state:
            self.net.load_state_dict(best_state)
        self.val_mse = float(best)
        self.trained_epochs = int(done_epochs)
        del X, Y, M, Xtr, Ytr, Mtr, Xva, Yva, Mva
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        t1 = time.perf_counter()
        print("  [tcn-score] fit OK residual limits/std...", flush=True)
        self._fit_error_limits(ok_records)
        print(f"  [tcn-score] residual limits/std done in {time.perf_counter() - t1:.1f}s",
              flush=True)
        return self

    def train_extra_epochs(self, ok_records, extra_epochs: int):
        """Continue training the current network for exactly extra_epochs epochs."""
        extra_epochs = max(0, int(extra_epochs or 0))
        if extra_epochs <= 0:
            return self
        if self.net is None:
            raise RuntimeError("TCN network is not initialized")
        self._setup_torch_runtime()
        t0 = time.perf_counter()
        print(f"  [tcn-post] build train windows for +{extra_epochs} epochs...", flush=True)
        X, Y, M = self._build_samples(ok_records)
        print(f"  [tcn-post] build windows done in {time.perf_counter() - t0:.1f}s",
              flush=True)
        n = X.shape[0]
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        n_val = max(1, int(n * float(getattr(self.cfg, "tcn_val_frac", 0.15))))
        vi = torch.as_tensor(idx[:n_val])
        ti = torch.as_tensor(idx[n_val:])
        if len(ti) == 0:
            ti = vi
        Xtr, Ytr, Mtr = X[ti], Y[ti], M[ti]
        Xva, Yva, Mva = X[vi], Y[vi], M[vi]
        train_bytes = self._tensor_bytes(Xtr, Ytr, Mtr, Xva, Yva, Mva)
        cache_limit = self._gpu_cache_limit_bytes()
        cache_on_gpu = (
            self.device_type == "cuda"
            and bool(getattr(self.cfg, "tcn_cache_train_on_gpu", True))
            and train_bytes <= cache_limit
        )
        if cache_on_gpu:
            Xtr = Xtr.to(self.device, non_blocking=True)
            Ytr = Ytr.to(self.device, non_blocking=True)
            Mtr = Mtr.to(self.device, non_blocking=True)
            Xva = Xva.to(self.device, non_blocking=True)
            Yva = Yva.to(self.device, non_blocking=True)
            Mva = Mva.to(self.device, non_blocking=True)
            cache_label = "gpu"
        else:
            Xtr, Ytr, Mtr = self._pin_if_cuda(Xtr), self._pin_if_cuda(Ytr), self._pin_if_cuda(Mtr)
            Xva, Yva, Mva = self._pin_if_cuda(Xva), self._pin_if_cuda(Yva), self._pin_if_cuda(Mva)
            cache_label = "pinned-cpu" if self.device_type == "cuda" else "cpu"
        print(f"  [tcn-post] train windows={n:,} | cache={cache_label} "
              f"({train_bytes / (1024 ** 3):.2f} GiB, "
              f"gpu_limit={cache_limit / (1024 ** 3):.2f} GiB)", flush=True)
        bs = max(1, int(getattr(self.cfg, "tcn_batch", 256)))
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(getattr(self.cfg, "tcn_lr", 1e-3)),
                                weight_decay=float(getattr(self.cfg, "tcn_weight_decay", 1e-4)))
        start_epoch = int(getattr(self, "trained_epochs", 0) or 0)
        torch.manual_seed(42 + start_epoch)
        if self.device_type == "cuda":
            torch.cuda.manual_seed_all(42 + start_epoch)
        last_vloss = float(getattr(self, "val_mse", float("nan")))
        for ep in range(extra_epochs):
            self.net.train()
            perm = torch.randperm(Xtr.shape[0], device=Xtr.device)
            for i in range(0, Xtr.shape[0], bs):
                bi = perm[i:i + bs]
                xb = self._move_batch(Xtr, bi)
                yb = self._move_batch(Ytr, bi)
                mb = self._move_batch(Mtr, bi)
                opt.zero_grad()
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16,
                                    enabled=self.use_amp):
                    pred = self.net(xb)
                    loss = (((pred - yb) ** 2) * mb).sum() / mb.sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                self.net.eval()
                tot, cnt = 0.0, 0.0
                for i in range(0, Xva.shape[0], bs):
                    sl = slice(i, i + bs)
                    xb = self._move_batch(Xva, sl)
                    yb = self._move_batch(Yva, sl)
                    mb = self._move_batch(Mva, sl)
                    with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16,
                                        enabled=self.use_amp):
                        pv = self.net(xb)
                    tot += float((((pv.float() - yb) ** 2) * mb).sum())
                    cnt += float(mb.sum())
                last_vloss = tot / max(cnt, 1.0)
            epoch_no = start_epoch + ep + 1
            if (ep + 1) % 5 == 0 or ep == 0 or ep + 1 == extra_epochs:
                print(f"  [tcn-post] epoch {epoch_no} (+{ep + 1}/{extra_epochs}) "
                      f"val_mse={last_vloss:.6f}", flush=True)
        self.val_mse = float(last_vloss)
        self.trained_epochs = int(start_epoch + extra_epochs)
        del X, Y, M, Xtr, Ytr, Mtr, Xva, Yva, Mva
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        t1 = time.perf_counter()
        print("  [tcn-post] refit OK residual limits/std...", flush=True)
        self._fit_error_limits(ok_records)
        print(f"  [tcn-post] residual limits/std done in {time.perf_counter() - t1:.1f}s",
              flush=True)
        return self

    def _prediction_window_batch_size(self, pred_batch: int = None) -> int:
        try:
            override = int(pred_batch if pred_batch is not None
                           else getattr(self.cfg, "pred_batch", 1) or 1)
        except Exception:
            override = 1
        if override > 1:
            return override
        return max(1, int(getattr(self.cfg, "tcn_batch", 256)))

    @torch.no_grad()
    def _predict_std_full(self, rec, pred_batch: int = None) -> np.ndarray:
        self.net.eval()
        Xin = self._std_input_sensor(self._input_sensors(rec))
        Yin = self._std_target_sensor(self.actual(rec))
        io = np.asarray(rec["io"], dtype=np.float64)
        T = Xin.shape[0]
        starts = self._starts(T)
        keep = np.ones(T, dtype=bool)
        acc = np.zeros((len(self.sensor_cols), T), dtype=np.float64)
        cnt = np.zeros(T, dtype=np.float64)
        bs = self._prediction_window_batch_size(pred_batch)
        j = np.arange(self.w, self.w + self.h)
        for i in range(0, len(starts), bs):
            chunk = starts[i:i + bs]
            Xt = self._device_batch_from_arrays(
                [self._block(io, Xin, Yin, rec.get("step"), keep, a, T)[0] for a in chunk])
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16,
                                enabled=self.use_amp):
                P = self.net(Xt)
            P = P.float().cpu().numpy()
            for b, a in enumerate(chunk):
                t = a - self.w + j
                m = (t >= 0) & (t < T)
                if m.any():
                    tm = t[m]
                    acc[:, tm] += P[b][:, j[m]]
                    cnt[tm] += 1
        cnt[cnt < 1] = 1
        return acc / cnt

    @torch.no_grad()
    def _predict_std_full_many(self, recs, pred_batch: int = None) -> list[np.ndarray]:
        """Predict several wafers by batching their forecast windows together."""
        recs = list(recs or [])
        if not recs:
            return []
        self.net.eval()
        states = []
        work = []
        for rec_i, rec in enumerate(recs):
            Xin = self._std_input_sensor(self._input_sensors(rec))
            Yin = self._std_target_sensor(self.actual(rec))
            io = np.asarray(rec["io"], dtype=np.float64)
            T = Xin.shape[0]
            states.append({
                "rec": rec,
                "Xin": Xin,
                "Yin": Yin,
                "io": io,
                "T": T,
                "keep": np.ones(T, dtype=bool),
                "acc": np.zeros((len(self.sensor_cols), T), dtype=np.float64),
                "cnt": np.zeros(T, dtype=np.float64),
            })
            for start in self._starts(T):
                work.append((rec_i, start))

        bs = self._prediction_window_batch_size(pred_batch)
        j = np.arange(self.w, self.w + self.h)
        for i in range(0, len(work), bs):
            chunk = work[i:i + bs]
            Xt = self._device_batch_from_arrays([
                self._block(
                    states[rec_i]["io"],
                    states[rec_i]["Xin"],
                    states[rec_i]["Yin"],
                    states[rec_i]["rec"].get("step"),
                    states[rec_i]["keep"],
                    start,
                    states[rec_i]["T"],
                )[0]
                for rec_i, start in chunk
            ])
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16,
                                enabled=self.use_amp):
                P = self.net(Xt)
            P = P.float().cpu().numpy()
            for b, (rec_i, start) in enumerate(chunk):
                state = states[rec_i]
                t = start - self.w + j
                m = (t >= 0) & (t < state["T"])
                if m.any():
                    tm = t[m]
                    state["acc"][:, tm] += P[b][:, j[m]]
                    state["cnt"][tm] += 1

        out = []
        for state in states:
            cnt = state["cnt"]
            cnt[cnt < 1] = 1
            out.append(state["acc"] / cnt)
        return out

    @torch.no_grad()
    def predict(self, rec, pred_batch: int = None) -> np.ndarray:
        pred_std = self._predict_std_full(rec, pred_batch=pred_batch).T
        return pred_std * self.sen_std + self.sen_mean

    @torch.no_grad()
    def predict_batch(self, recs, pred_batch: int = None) -> list[np.ndarray]:
        recs = list(recs or [])
        if not recs:
            return []
        try:
            window_batch = int(pred_batch if pred_batch is not None
                               else getattr(self.cfg, "pred_batch", 1) or 1)
        except Exception:
            window_batch = 1
        window_batch = max(1, window_batch)
        if window_batch <= 1:
            return [self.predict(r) for r in recs]
        out = []
        wafer_chunk = min(len(recs), 32)
        for i in range(0, len(recs), wafer_chunk):
            for pred_std in self._predict_std_full_many(
                    recs[i:i + wafer_chunk], pred_batch=window_batch):
                out.append(pred_std.T * self.sen_std + self.sen_mean)
        return out

    def residual(self, rec) -> np.ndarray:
        return self.actual(rec) - self.predict(rec)

    def _limit_for_record(self, rec) -> np.ndarray:
        T = int(rec["sensors"].shape[0])
        lim = np.asarray(self.resid_abs_limit, dtype=np.float64)
        if self._per_step_limit and rec.get("step") is not None and lim.ndim == 2:
            st = np.clip(np.asarray(rec["step"]).astype(int), 0, lim.shape[0] - 1)
            out = lim[st]
        elif lim.ndim == 1:
            out = np.repeat(lim[None, :], T, axis=0)
        else:
            out = np.ones((T, len(self.sensor_cols)), dtype=np.float64)
        out = np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)
        out[out < float(getattr(self.cfg, "raw_error_min", 1e-9))] = float(
            getattr(self.cfg, "raw_error_min", 1e-9))
        return out

    def _sigma_for_record(self, rec) -> np.ndarray:
        T = int(rec["sensors"].shape[0])
        sigma_ref = self.resid_sigma
        if sigma_ref is None:
            sigma_ref = self.resid_std if self.resid_std is not None else self.resid_abs_limit
        sig = np.asarray(sigma_ref, dtype=np.float64)
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
        out[out < float(getattr(self.cfg, "raw_error_min", 1e-9))] = float(
            getattr(self.cfg, "raw_error_min", 1e-9))
        return out

    def error_level(self, rec) -> np.ndarray:
        level = np.abs(self.residual(rec)) / self._limit_for_record(rec)
        keep = self._score_keep_mask(rec)
        if keep is not None and len(keep) == level.shape[0]:
            level = level.copy()
            level[~keep] = 0.0
        return level

    def z_residual(self, rec) -> np.ndarray:
        signed = self.residual(rec) / self._limit_for_record(rec)
        keep = self._score_keep_mask(rec)
        if keep is not None and len(keep) == signed.shape[0]:
            signed = signed.copy()
            signed[~keep] = 0.0
        return signed

    def sigma_residual(self, rec) -> np.ndarray:
        signed = self.residual(rec) / self._sigma_for_record(rec)
        keep = self._score_keep_mask(rec)
        if keep is not None and len(keep) == signed.shape[0]:
            signed = signed.copy()
            signed[~keep] = 0.0
        return signed

    def residual_batch(self, recs, pred_batch: int = None):
        recs = list(recs or [])
        preds = self.predict_batch(recs, pred_batch=pred_batch)
        return [self.actual(r) - p for r, p in zip(recs, preds)]

    def error_level_batch(self, recs, pred_batch: int = None):
        recs = list(recs or [])
        preds = self.predict_batch(recs, pred_batch=pred_batch)
        out = []
        for rec, pred in zip(recs, preds):
            level = np.abs(self.actual(rec) - pred) / self._limit_for_record(rec)
            keep = self._score_keep_mask(rec)
            if keep is not None and len(keep) == level.shape[0]:
                level = level.copy()
                level[~keep] = 0.0
            out.append(level)
        return out

    def _fit_error_limits(self, ok_records) -> None:
        q = float(getattr(self.cfg, "raw_error_quantile", 99.0))
        q = min(max(q, 50.0), 99.99)
        residuals = []
        signed_residuals = []
        steps = []
        for r in ok_records:
            signed = self.residual(r)
            res = np.abs(signed)
            keep = self._score_keep_mask(r)
            if keep is not None and keep.any():
                residuals.append(res[keep])
                signed_residuals.append(signed[keep])
                if r.get("step") is not None:
                    steps.append(np.asarray(r["step"])[keep].astype(int))
            else:
                residuals.append(res)
                signed_residuals.append(signed)
                if r.get("step") is not None:
                    steps.append(np.asarray(r["step"]).astype(int))
        all_res = np.concatenate(residuals, axis=0)
        all_signed = np.concatenate(signed_residuals, axis=0)
        global_lim = np.percentile(all_res, q, axis=0)
        global_lim[global_lim < float(getattr(self.cfg, "raw_error_min", 1e-9))] = float(
            getattr(self.cfg, "raw_error_min", 1e-9))
        global_sigma = np.nanstd(all_signed, axis=0)
        global_sigma[global_sigma < float(getattr(self.cfg, "raw_error_min", 1e-9))] = float(
            getattr(self.cfg, "raw_error_min", 1e-9))
        if self.n_steps_global > 0 and len(steps) == len(residuals):
            lim = np.repeat(global_lim[None, :], self.n_steps_global, axis=0)
            sigma = np.repeat(global_sigma[None, :], self.n_steps_global, axis=0)
            for s in range(self.n_steps_global):
                vals = [res[st == s] for res, st in zip(residuals, steps) if np.any(st == s)]
                if vals:
                    vv = np.concatenate(vals, axis=0)
                    if len(vv):
                        lim[s] = np.percentile(vv, q, axis=0)
                svals = [res[st == s] for res, st in zip(signed_residuals, steps) if np.any(st == s)]
                if svals:
                    sv = np.concatenate(svals, axis=0)
                    if len(sv):
                        sigma[s] = np.nanstd(sv, axis=0)
            lim[lim < float(getattr(self.cfg, "raw_error_min", 1e-9))] = float(
                getattr(self.cfg, "raw_error_min", 1e-9))
            sigma[sigma < float(getattr(self.cfg, "raw_error_min", 1e-9))] = float(
                getattr(self.cfg, "raw_error_min", 1e-9))
            self.resid_abs_limit = lim
            self.resid_sigma = sigma
            self._per_step_limit = True
            self._per_step_sigma = True
        else:
            self.resid_abs_limit = global_lim
            self.resid_sigma = global_sigma
            self._per_step_limit = False
            self._per_step_sigma = False
        self.resid_std = self.resid_sigma
        print(f"  [tcn-score] raw residual 기준: OK |actual-expected| "
              f"q{q:g} 초과분을 이탈로 계산, 설명 그림은 OK residual std를 별도 저장",
              flush=True)

    def _score_record(self, rec) -> float:
        level = self.error_level(rec)
        if level.size == 0:
            return 0.0
        return _reconstruction_score_from_level(
            level, self.cfg, valid_steps=self._score_keep_mask(rec))

    def score_records(self, recs):
        recs = list(recs or [])
        try:
            pred_batch = int(getattr(self.cfg, "pred_batch", 1) or 1)
        except Exception:
            pred_batch = 1
        if pred_batch <= 1:
            return np.asarray([self._score_record(r) for r in recs], dtype=np.float64)
        levels = self.error_level_batch(recs, pred_batch=pred_batch)
        return np.asarray([
            0.0 if level.size == 0 else _reconstruction_score_from_level(
                level, self.cfg, valid_steps=self._score_keep_mask(rec))
            for rec, level in zip(recs, levels)
        ], dtype=np.float64)

    def assess(self, rec):
        level = graph_adjust_abs_z(self.error_level(rec), self.sensor_cols, self.cfg)
        per = level.max(axis=0) if level.size else np.zeros(len(self.sensor_cols))
        order = np.argsort(per)[::-1]
        actual = self.actual(rec)
        pred = self.predict(rec)
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

    def io_attribution(self, rec, sensor_idx: int, at_t: Optional[int] = None) -> np.ndarray:
        self.net.eval()
        T = int(rec["sensors"].shape[0])
        starts = self._starts(T)
        if at_t is None:
            a = starts[len(starts) // 2]
            jpos = None
        else:
            at_t = int(np.clip(at_t, 0, T - 1))
            cands = [s for s in starts if s <= at_t < s + self.h]
            a = cands[0] if cands else min(starts, key=lambda s: abs((s + self.h // 2) - at_t))
            jpos = int(np.clip(self.w + (at_t - a), self.w, self.w + self.h - 1))
        Xin = self._std_input_sensor(self._input_sensors(rec))
        Yin = self._std_target_sensor(self.actual(rec))
        keep = np.ones(T, dtype=bool)
        x = self._block(np.asarray(rec["io"], dtype=np.float64), Xin, Yin, rec.get("step"),
                        keep, a, T)[0]
        xt = torch.tensor(x[None], dtype=torch.float32, device=self.device,
                          requires_grad=True)
        pred = self.net(xt)
        scalar = pred[0, sensor_idx, jpos].pow(2) if jpos is not None \
            else pred[0, sensor_idx, self.w:].pow(2).mean()
        self.net.zero_grad()
        scalar.backward()
        return xt.grad[0].abs().mean(dim=1).cpu().numpy()

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state": self.net.state_dict(),
            "io_cols": self.io_cols,
            "input_sensor_cols": self.input_sensor_cols,
            "sensor_cols": self.sensor_cols,
            "target_indices": self.target_indices,
            "h_indices": self.h_indices,
            "future_context_indices": getattr(self, "future_context_indices", []),
            "future_h_indices": self.future_h_indices,
            "n_steps_global": self.n_steps_global,
            "schema_mode": self.schema_mode,
            "input_sen_mean": self.input_sen_mean,
            "input_sen_std": self.input_sen_std,
            "sen_mean": self.sen_mean,
            "sen_std": self.sen_std,
            "resid_abs_limit": self.resid_abs_limit,
            "resid_std": self.resid_std,
            "resid_std_kind": "sigma",
            "resid_sigma": self.resid_sigma,
            "per_step_limit": self._per_step_limit,
            "per_step_sigma": self._per_step_sigma,
            "arch": self.arch,
            "val_mse": getattr(self, "val_mse", None),
            "trained_epochs": getattr(self, "trained_epochs", None),
        }, path)

    @classmethod
    def load(cls, path: Path, cfg: IOConfig, device: Optional[str] = None):
        d = torch.load(Path(path), map_location="cpu", weights_only=False)
        m = cls(cfg, device)
        m.io_cols = d["io_cols"]
        m.input_sensor_cols = d.get("input_sensor_cols", d["sensor_cols"])
        m.sensor_cols = d["sensor_cols"]
        m.target_indices = d.get("target_indices", list(range(len(m.sensor_cols))))
        m.h_indices = d.get("h_indices", [])
        m.future_h_indices = d.get("future_h_indices", list(m.h_indices))
        m.n_steps_global = int(d["n_steps_global"])
        m.schema_mode = d.get("schema_mode", "")
        m.input_sen_mean = d.get("input_sen_mean", d["sen_mean"])
        m.input_sen_std = d.get("input_sen_std", d["sen_std"])
        m.sen_mean = d["sen_mean"]
        m.sen_std = d["sen_std"]
        m.resid_abs_limit = d.get("resid_abs_limit", d.get("resid_std"))
        m.resid_std = d.get("resid_std", m.resid_abs_limit)
        m.resid_sigma = d.get("resid_sigma")
        if m.resid_sigma is None and d.get("resid_std_kind") == "sigma":
            m.resid_sigma = m.resid_std
        m._per_step_limit = bool(d.get("per_step_limit", False))
        m._per_step_sigma = bool(d.get("per_step_sigma", False))
        m.arch = d["arch"]
        m.arch.setdefault("file_format", "csv")
        m.arch.setdefault("future_h_mode", "input")
        m.arch.setdefault("virtual_sensor_mode", "off")
        m.arch.setdefault("h_sensor_preset", "")
        m.arch.setdefault("h_sensor_keep", [])
        m.arch.setdefault("sensor_role_profile", "original")
        try:
            m.cfg.h_cart_mode = m.arch.get("h_cart_mode", getattr(m.cfg, "h_cart_mode", ""))
            m.cfg.future_h_mode = m.arch["future_h_mode"]
            m.cfg.virtual_sensor_mode = m.arch["virtual_sensor_mode"]
            m.cfg.h_sensor_preset = m.arch.get("h_sensor_preset", "") or ""
            m.cfg.h_sensor_keep = tuple(m.arch.get("h_sensor_keep") or ())
            m.cfg.sensor_role_profile = normalize_sensor_role_profile(
                m.arch.get("sensor_role_profile", "original"))
        except Exception:
            pass
        configure_auxiliary_model(m, d)
        m.w = int(m.arch["past_window"])
        m.h = int(m.arch["future_window"])
        m.stride = int(m.arch["stride"])
        m.val_mse = d.get("val_mse")
        m.trained_epochs = d.get("trained_epochs")
        m.net = IOTCN(m.n_in(), len(m.sensor_cols), int(m.arch["channels"]),
                      int(m.arch["blocks"]), int(m.arch["kernel"]),
                      float(m.arch["dropout"])).to(m.device)
        m.net.load_state_dict(d["state"])
        return m


class TCNForecastDetector:
    """Wrapper matching export_report.build_report's detector contract."""

    def __init__(self, cfg: IOConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device
        self.A: Optional[TCNForecastModel] = None

    def fit(self, ok_records, io_cols, sensor_cols, n_steps_global, val_records=None):
        self.A = TCNForecastModel(self.cfg, device=self.device).fit(
            ok_records, io_cols, sensor_cols, n_steps_global, val_records=val_records)
        return self

    def score_records(self, recs):
        return self.A.score_records(recs)

    def assess(self, rec):
        return self.A.assess(rec)

    def save(self, adir: Path):
        adir = Path(adir)
        adir.mkdir(parents=True, exist_ok=True)
        self.A.save(adir / "tcn_forecast.pt")

    @classmethod
    def load(cls, adir: Path, cfg: IOConfig):
        h = cls(cfg)
        h.A = TCNForecastModel.load(Path(adir) / "tcn_forecast.pt", cfg)
        return h
