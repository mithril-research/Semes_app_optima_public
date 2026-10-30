"""ONNX 모델의 IR 버전을 낮춰 구버전 onnxruntime 이 읽을 수 있게 한다.

왜 필요한가
  Windows 7 에서 쓸 수 있는 onnxruntime 은 1.14.1 이 상한이다
  (1.15.1 부터 DLL 이 Windows 8 전용 API 인 CreateFile2 를 정적 임포트한다 — 실측 확인).
  그런데 학습앱이 만드는 .onnx 는 IR 버전 10(ONNX 1.16 계열)이라
  onnxruntime 1.14 가 "Unsupported model IR version" 으로 거절한다.

  IR 버전은 파일 맨 앞의 필드 하나다. 모델의 연산 내용은 그대로 두고 이 값만
  낮추면 구버전에서도 읽힌다. 단, 모델이 실제로 최신 IR 전용 기능을 쓰고 있으면
  안 되는데, 이 앱의 모델은 opset 17 에 Conv/Relu/Add/Slice 만 쓰므로 안전하다.
  (변환 후 실제로 로드·추론해 확인까지 한다)

사용법
    python onnx_ir_downgrade.py <모델폴더 또는 .onnx 파일> [--ir 8]
원본은 <이름>.onnx.bak 으로 백업한다.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TARGET_DEFAULT = 8          # onnxruntime 1.14 가 읽는 최대 IR


def _read_ir(data: bytes):
    """맨 앞 필드(ir_version, varint)를 읽어 (값, 본문 시작 오프셋) 반환."""
    if not data or data[0] != 0x08:                      # field 1, wire type 0
        return None, 0
    i, v, shift = 1, 0, 0
    while i < len(data):
        b = data[i]; i += 1
        v |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return v, i
    return None, 0


def downgrade(path: Path, target: int = TARGET_DEFAULT) -> str:
    data = path.read_bytes()
    cur, body = _read_ir(data)
    if cur is None:
        return f"  건너뜀 {path.name}: ONNX 형식이 아닌 것 같습니다"
    if cur <= target:
        return f"  그대로 {path.name}: IR {cur} (이미 {target} 이하)"
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_bytes(bytes([0x08, target]) + data[body:])
    return f"  변환됨 {path.name}: IR {cur} → {target}  (원본 백업: {bak.name})"


def verify(path: Path) -> str:
    """실제로 로드·추론해 본다(onnxruntime 이 있을 때만)."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return "  (확인 생략: 이 환경에 onnxruntime 이 없습니다)"
    try:
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {}
        for x in s.get_inputs():
            shape = [d if isinstance(d, int) else 1 for d in x.shape]
            feed[x.name] = np.zeros(shape, dtype=np.float32)
        s.run(None, feed)
        return f"  확인 완료: 로드·추론 정상 ({ort.__version__})"
    except Exception as e:                               # noqa: BLE001
        return f"  ★확인 실패: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="ONNX IR 버전 낮추기 (구버전 onnxruntime 대응)")
    ap.add_argument("target", help="모델 폴더 또는 .onnx 파일")
    ap.add_argument("--ir", type=int, default=TARGET_DEFAULT, help=f"목표 IR (기본 {TARGET_DEFAULT})")
    a = ap.parse_args()

    p = Path(a.target)
    if p.is_dir():
        files = sorted(p.glob("*.onnx"))
    elif p.suffix.lower() == ".onnx":
        files = [p]
    else:
        print(f"[X] .onnx 파일이나 모델 폴더를 지정하세요: {p}")
        return 2
    if not files:
        print(f"[X] .onnx 파일이 없습니다: {p}")
        return 2

    print("=" * 62)
    print("  ONNX IR 버전 낮추기 (Windows 7 용 onnxruntime 1.14 대응)")
    print("=" * 62)
    for f in files:
        print(downgrade(f, a.ir))
        print(verify(f))
    print("\n끝났습니다. 이 폴더를 앱의 모델 폴더로 지정하세요.")
    print("※ 되돌리려면 .onnx.bak 파일의 이름에서 .bak 을 지우면 됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
