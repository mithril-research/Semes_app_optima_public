"""Export IO-Recon-TCN forecast checkpoints to ONNX plus metadata.

This mirrors the standalone export script used by the deployment checker:
only the neural network is exported to ONNX, while normalization, tolerance,
and score metadata are stored in a companion ``.npz`` file.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch

from .auxiliary_sensors import configure_auxiliary_model, auxiliary_npz_metadata
from .wafer_score_override import artifact_override
from .config import IOConfig
from .model_tcn_forecast import TCNForecastModel


def _resolve_pt(pt_path=None, art_dir=None) -> Path:
    if pt_path:
        p = Path(pt_path)
        if p.is_dir():
            c = p / "tcn_forecast.pt"
            p = c if c.exists() else next(iter(sorted(p.glob("*.pt"))), c)
        return p
    base = Path(art_dir) if art_dir else (Path(__file__).resolve().parent / "artifacts")
    return base / "tcn_forecast.pt"


def _embed_weights(onnx_path: Path):
    """Absorb external ONNX weight sidecars into the single .onnx file."""
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    onnx_path = Path(onnx_path)
    m = onnx.load(str(onnx_path))
    load_external_data_for_model(m, str(onnx_path.parent))
    for init in m.graph.initializer:
        init.ClearField("data_location")
        del init.external_data[:]
    onnx.save(m, str(onnx_path), save_as_external_data=False)
    side = onnx_path.parent / (onnx_path.name + ".data")
    if side.exists():
        side.unlink()


def _load_model_for_export(pt: Path, base_cfg):
    """Load checkpoint using the net dimensions saved in the actual state."""
    import re
    from dataclasses import replace as _replace
    from .tcn import IOTCN
    d = torch.load(Path(pt), map_location="cpu", weights_only=False)
    st = d["state"]
    c1 = next(k for k in st if re.search(r"tcn\.0\.conv1\.conv\.weight$", k))
    true_channels, true_n_in, true_kernel = (int(x) for x in st[c1].shape)
    n_out = int(st["head.weight"].shape[0])
    n_blocks = len({re.match(r"tcn\.(\d+)\.", k).group(1)
                    for k in st if re.match(r"tcn\.\d+\.", k)})

    io_n = len(d["io_cols"])
    in_sen_n = len(d.get("input_sensor_cols", d["sensor_cols"]))
    n_steps = int(d["n_steps_global"])
    base = io_n + in_sen_n + 1
    cfg = base_cfg
    if true_n_in == base + n_steps:
        cfg = _replace(base_cfg, use_step_onehot=True)
    elif true_n_in == base:
        cfg = _replace(base_cfg, use_step_onehot=False)

    try:
        A = TCNForecastModel.load(pt, cfg, device="cpu")
        if A.n_in() == true_n_in:
            return A, cfg
    except Exception as e:
        print(f"  [load] 표준 로드 실패({e}) -> state 크기로 net 재구성", flush=True)

    A = TCNForecastModel(cfg, device="cpu")
    A.io_cols = d["io_cols"]
    A.input_sensor_cols = d.get("input_sensor_cols", d["sensor_cols"])
    A.sensor_cols = d["sensor_cols"]
    A.target_indices = d.get("target_indices", list(range(len(A.sensor_cols))))
    A.h_indices = d.get("h_indices", [])
    A.future_h_indices = d.get("future_h_indices", list(A.h_indices))
    A.n_steps_global = n_steps
    A.schema_mode = d.get("schema_mode", "")
    A.input_sen_mean = d.get("input_sen_mean", d["sen_mean"])
    A.input_sen_std = d.get("input_sen_std", d["sen_std"])
    A.sen_mean = d["sen_mean"]
    A.sen_std = d["sen_std"]
    A.resid_abs_limit = d.get("resid_abs_limit", d.get("resid_std"))
    A.resid_std = d.get("resid_std", A.resid_abs_limit)
    A.resid_sigma = d.get("resid_sigma")
    A._per_step_limit = bool(d.get("per_step_limit", False))
    A._per_step_sigma = bool(d.get("per_step_sigma", False))
    A.arch = d["arch"]
    A.arch.setdefault("future_h_mode", "input")
    A.arch.setdefault("virtual_sensor_mode", "off")
    try:
        A.cfg.future_h_mode = A.arch["future_h_mode"]
        A.cfg.virtual_sensor_mode = A.arch["virtual_sensor_mode"]
    except Exception:
        pass
    configure_auxiliary_model(A, d)
    A.w = int(A.arch["past_window"])
    A.h = int(A.arch["future_window"])
    A.stride = int(A.arch["stride"])
    A.net = IOTCN(true_n_in, n_out, true_channels, n_blocks, true_kernel,
                  float(A.arch.get("dropout", 0.1))).to("cpu")
    A.net.load_state_dict(st)
    A._forced_n_in = true_n_in
    print(f"  [load] net 재구성: n_in={true_n_in} ch={true_channels} "
          f"blocks={n_blocks} kernel={true_kernel} n_out={n_out}", flush=True)
    return A, cfg


def export(pt_path=None, out_dir=None, art_dir=None, opset: int = 17) -> Path:
    pt = _resolve_pt(pt_path, art_dir)
    if not pt.exists():
        sys.exit(f"모델 없음: {pt}")
    out = Path(out_dir) if out_dir else pt.parent
    out.mkdir(parents=True, exist_ok=True)

    import importlib.util as _ilu
    miss = [p for p in ("onnx", "onnxscript") if _ilu.find_spec(p) is None]
    if miss:
        sys.exit("모델 변환에 필요한 패키지가 없습니다: " + ", ".join(miss) +
                 "\n  설치: pip install " + " ".join(miss))

    A, cfg = _load_model_for_export(pt, IOConfig())
    A.net.eval()
    _, score_override = artifact_override(pt.parent)

    n_in = int(getattr(A, "_forced_n_in", A.n_in()))
    L = A.w + A.h
    stem = pt.stem
    onnx_path = out / f"{stem}.onnx"
    dummy = torch.zeros(1, n_in, L, dtype=torch.float32)
    torch.onnx.export(
        A.net, dummy, str(onnx_path),
        input_names=["x"], output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
    )
    _embed_weights(onnx_path)
    print(f"[export] {onnx_path}  (n_in={n_in}, L={L}, 단일파일)")

    def _arr(x, dt=np.float64):
        return None if x is None else np.asarray(x, dtype=dt)

    meta = out / f"{stem}.npz"
    np.savez(
        meta,
        io_cols=np.array([str(c) for c in A.io_cols], dtype=object),
        input_sensor_cols=np.array([str(c) for c in A.input_sensor_cols], dtype=object),
        sensor_cols=np.array([str(c) for c in A.sensor_cols], dtype=object),
        target_indices=np.asarray(A.target_indices, dtype=np.int64),
        h_indices=np.asarray(A.h_indices, dtype=np.int64),
        **auxiliary_npz_metadata(A),
        wafer_score_override_json=json.dumps(score_override or {}, ensure_ascii=False),
        n_steps_global=np.int64(A.n_steps_global),
        schema_mode=str(A.schema_mode),
        input_sen_mean=_arr(A.input_sen_mean), input_sen_std=_arr(A.input_sen_std),
        sen_mean=_arr(A.sen_mean), sen_std=_arr(A.sen_std),
        resid_abs_limit=_arr(A.resid_abs_limit),
        resid_sigma=_arr(A.resid_sigma),
        resid_std=_arr(getattr(A, "resid_std", None)),
        per_step_limit=np.bool_(getattr(A, "_per_step_limit", False)),
        per_step_sigma=np.bool_(getattr(A, "_per_step_sigma", False)),
        use_step_onehot=np.bool_(getattr(A.cfg, "use_step_onehot", True)),
        w=np.int64(A.w), h=np.int64(A.h), stride=np.int64(A.stride),
        n_in=np.int64(n_in),
        future_h_mode=str(A.arch.get("future_h_mode", "input")),
        virtual_sensor_mode=str(A.arch.get("virtual_sensor_mode", "off")),
    )
    print(f"[export] {meta}  (torch-free 메타)")
    import shutil as _sh
    if stem != "recon_meta":
        _sh.copy(meta, out / "recon_meta.npz")
    return onnx_path


def verify(pt_path=None, out_dir=None, art_dir=None, n_trials: int = 20):
    """Compare neural-network forward outputs between torch and ONNX."""
    import onnxruntime as ort
    pt = _resolve_pt(pt_path, art_dir)
    out = Path(out_dir) if out_dir else pt.parent
    A, cfg = _load_model_for_export(pt, IOConfig())
    A.net.eval()
    n_in = int(getattr(A, "_forced_n_in", A.n_in()))
    L = A.w + A.h
    sess = ort.InferenceSession(str(out / f"{pt.stem}.onnx"),
                                providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(0)
    mx = 0.0
    for _ in range(n_trials):
        B = int(rng.randint(1, 5))
        x = rng.randn(B, n_in, L).astype(np.float32)
        with torch.no_grad():
            yt = A.net(torch.from_numpy(x)).numpy()
        yo = sess.run(["y"], {"x": x})[0]
        mx = max(mx, float(np.abs(yt - yo).max()))
    print(f"[verify:net] {n_trials} trials · max |torch-onnx| = {mx:.3e}")
    return mx


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IO-Recon-TCN .pt -> .onnx(+meta)")
    ap.add_argument("pt", nargs="?", default=None, help=".pt 파일 또는 폴더")
    ap.add_argument("-o", "--out", default=None, help="저장 폴더")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    export(pt_path=a.pt, out_dir=a.out, opset=a.opset)
    d = verify(pt_path=a.pt, out_dir=a.out)
    print("통과" if d < 1e-4 else f"차이 큼: {d:.3e}")
