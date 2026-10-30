"""학습 직후 OK 누적 밴드(ok_accum_bands.npz)를 생성한다 — 리포트(recon-explain) 없이.

원래 train_jw 는 `tcn-explain/recon-explain` 리포트를 돌려야 `ok_accum_bands.npz` 가
만들어졌다. 검사 앱은 이 파일이 있어야 '개별 센서 복원오차 점수(정상 상한 대비 비율)'와
정상 밴드 음영을 그릴 수 있어서, 학습만 하고 리포트를 안 돌리면 점수가 비었다.
→ 학습이 끝나면 앱이 이 스크립트를 별도 프로세스로 실행해 밴드를 만들고 출력폴더에 넣는다.

  python -m src.app.build_ok_bands <artifacts_dir> <ok_dir> <out_dir> [--json OPTS]

별도 프로세스인 이유: torch 를 앱 프로세스에 올리지 않기 위해(메모리/GPU 반납 원칙).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):                            # 직접 실행 지원
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train_jw.config import (                # noqa: E402
    h_SENSOR_PRESETS, IOConfig, normalize_h_sensor_preset, normalize_sensor_role_profile,
)
from src.train_jw import recon_explain as RX     # noqa: E402
from src.train_jw.data import load_dataset       # noqa: E402
from src.train_jw.model_tcn_forecast import TCNForecastDetector  # noqa: E402
from src.train_jw.recon_explain import (         # noqa: E402
    _apply_score_agg_metadata_to_cfg, _score_kind_from_cfg,
    _threshold_with_score_metadata, _uses_sum_reconstruction_threshold,
)
from src.train_jw.run import _artifact_threshold_and_cfg  # noqa: E402


def build(artifacts_dir: Path, ok_dir: Path, out_dir: Path, opts: dict) -> Path:
    artifacts_dir, ok_dir, out_dir = Path(artifacts_dir), Path(ok_dir), Path(out_dir)
    base = IOConfig()
    base.ok_dir = ok_dir
    base.h_cart_mode = str(opts.get("h_cart_mode") or "include")
    base.file_format = str(opts.get("file_format") or "auto")
    base.h_sensor_preset = normalize_h_sensor_preset(opts.get("h_sensor_preset") or "all")
    base.sensor_role_profile = normalize_sensor_role_profile(
        opts.get("sensor_role_profile") or "original")
    if base.h_sensor_preset:
        base.h_sensor_keep = tuple(h_SENSOR_PRESETS[base.h_sensor_preset])
    base.future_h_mode = str(opts.get("future_h_mode") or "input")
    base.virtual_sensor_mode = ("both" if opts.get("virtual_sensors")
                                else str(opts.get("virtual_sensor_mode") or "off"))

    # 학습 결과(search_best.json)에서 cfg/threshold/score_kind 복원 — 리포트와 동일 경로
    cfg, threshold, score_kind, best = _artifact_threshold_and_cfg(base, artifacts_dir)
    cfg.ok_dir = ok_dir
    val_payload = best.get("val") if isinstance(best.get("val"), dict) else {}
    eval_payload = (best.get("evaluation_criteria")
                    if isinstance(best.get("evaluation_criteria"), dict) else {})
    _apply_score_agg_metadata_to_cfg(cfg, threshold, val_payload, eval_payload, best)
    A = TCNForecastDetector.load(artifacts_dir, cfg).A

    if _uses_sum_reconstruction_threshold(threshold, score_kind):
        threshold = _threshold_with_score_metadata(threshold, getattr(A, "cfg", cfg))
        score_kind = _score_kind_from_cfg(getattr(A, "cfg", cfg))

    ok_records, _io, _sen, _ns = load_dataset(ok_dir, cfg)
    if not ok_records:
        raise SystemExit(f"OK 웨이퍼를 찾지 못했습니다: {ok_dir}")
    print(f"[ok-bands] train OK {len(ok_records)}장으로 누적 밴드 계산", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "ok_accum_bands.npz"
    bands = RX.load_or_build_ok_accum_bands(
        A, ok_records, dst, threshold=threshold, score_kind=score_kind,
        exclude_ok_over_threshold=bool(opts.get("ok_mean_exclude_over_threshold", True)))
    if bands is None or not dst.exists():
        raise SystemExit("OK 누적 밴드 생성 실패")
    print(f"[ok-bands] 저장: {dst}", flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser(description="학습 직후 ok_accum_bands.npz 생성(리포트 불필요)")
    ap.add_argument("artifacts_dir")
    ap.add_argument("ok_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--json", default="{}", help="학습 옵션 JSON(h_cart_mode 등)")
    a = ap.parse_args()
    build(Path(a.artifacts_dir), Path(a.ok_dir), Path(a.out_dir), json.loads(a.json or "{}"))


if __name__ == "__main__":
    main()
