r"""모델 변환기(.pt → ONNX) — 제어기 동봉 전용 실행 파일의 진입점.

메인 앱(웨이퍼품질검사.exe)은 용량을 줄이려 torch 없이 onnxruntime 로만 추론한다.
학습 모델은 보통 .pt 로 전달되므로, torch 를 포함한 이 별도 실행 파일(모델변환.exe)
로 .pt 를 .onnx(+메타) 로 바꾼 뒤 원본 .pt 를 삭제한다.

동작(인자 없이 실행 시):
  - exe 옆 모델 폴더(모델\ · model\ · artifacts\)를 훑어 .pt 가 있으면 변환.
  - 변환 결과 .onnx 가 실제 로드 가능함을 확인한 뒤에만 .pt 삭제(안전).
인자로 폴더나 .pt 파일(드래그&드롭)을 주면 그 대상만 변환한다.

빌드: 모델변환.spec (변환기빌드.bat) — torch/onnx/onnxscript 포함 onedir.
"""
from __future__ import annotations

import sys
from pathlib import Path

# 직접 실행/frozen 모두에서 패키지 import 가 되도록 경로 보정
if not getattr(sys, "frozen", False) and __package__ in (None, "", "app"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _import_api():
    """패키지명이 배포본은 src, 개발본은 io_recon_tcn 이라 둘 다 시도."""
    try:
        from src.export_onnx_forecast import export
        from src.onnx_backend_forecast import OnnxForecastModel
    except Exception:                                    # noqa: BLE001
        from io_recon_tcn.export_onnx_forecast import export
        from io_recon_tcn.onnx_backend_forecast import OnnxForecastModel
    return export, OnnxForecastModel


def _base_dir() -> Path:
    return (Path(sys.executable).parent if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parents[1])


def _model_dirs_near_exe():
    base = _base_dir()
    return [d for d in (base / "모델", base / "model", base / "artifacts") if d.is_dir()]


def convert_folder(folder: Path) -> int:
    """폴더의 .pt 를 onnx 로 변환하고, 성공 확인 후 .pt 삭제. 반환=삭제한 .pt 수."""
    export, OnnxForecastModel = _import_api()
    folder = Path(folder)
    pts = sorted(folder.glob("*.pt"))
    if not pts:
        print(f"  · .pt 없음 → 건너뜀: {folder}")
        return 0
    if OnnxForecastModel.available(folder):
        print(f"  · 이미 ONNX 모델 있음: {folder}")
    else:
        for pt in pts:
            print(f"  · 변환: {pt.name}")
            export(pt_path=str(pt), out_dir=str(folder))
    deleted = 0
    if OnnxForecastModel.available(folder):              # onnx 로드 가능 확인 후에만 삭제
        for pt in pts:
            try:
                pt.unlink()
                print(f"  · 원본 삭제: {pt.name}")
                deleted += 1
            except Exception as e:                       # noqa: BLE001
                print(f"  · 삭제 실패({pt.name}): {e}")
    else:
        print(f"  ! ONNX 생성 확인 실패 — .pt 는 보존합니다: {folder}")
    return deleted


def main():
    print("=" * 56)
    print("  모델 변환기  (.pt → ONNX)")
    print("=" * 56)
    args = [a for a in sys.argv[1:] if a.strip()]
    targets = [Path(a.strip('"')) for a in args] if args else _model_dirs_near_exe()
    if not targets:
        print("변환할 모델 폴더를 찾지 못했습니다.")
        print("  · 이 파일 위에 '모델 폴더' 또는 '.pt 파일'을 드래그&드롭 하거나")
        print("  · exe 옆에 '모델\\' 폴더를 두고 다시 실행하세요.")
    total = 0
    for t in targets:
        if t.is_file() and t.suffix.lower() == ".pt":
            total += convert_folder(t.parent)
        elif t.is_dir():
            print(f"[폴더] {t}")
            total += convert_folder(t)
        else:
            print(f"  ! 경로 없음: {t}")
    print("-" * 56)
    print(f"완료 — 변환/삭제한 .pt {total}개.")
    try:
        input("Enter 를 누르면 닫힙니다...")
    except Exception:                                    # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
