"""PyInstaller 런타임 훅 — 이 exe 는 torch 백엔드로 빌드되었음을 앱에 알린다.

WES7 용 torch 빌드에만 포함된다(io_console_win7.spec 이 IO_WES7_BACKEND=torch 일 때만 넣는다).

왜 필요한가
  앱은 모델 폴더에 .onnx(+메타)가 있으면 그쪽을 먼저 고른다. 그런데 이 빌드에는
  onnxruntime 이 아예 들어있지 않으므로, 폴더에 .onnx 가 섞여 있으면 그것을 골랐다가
  로드 시점에 죽는다. 여기서 IO_BACKEND=torch 를 미리 박아 그 경로를 원천 차단한다.
  (사용자가 환경변수로 다르게 지정했다면 그 뜻을 존중해 덮어쓰지 않는다.)
"""
import os

os.environ.setdefault("IO_BACKEND", "torch")

# ★ OpenMP 런타임이 두 번 로드되면 즉시 abort 한다(OMP: Error #15).
#   torch·MKL·numpy 가 각자 libiomp5md.dll 을 들고 오는데, 얼린 exe 안에서는
#   같은 폴더에 여러 벌이 놓여 이 충돌이 잘 난다. 프로세스가 메시지도 없이
#   사라지는 대표적인 원인이라 미리 허용해 둔다(성능 영향 없음).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# MKL 이 GNU/Intel 스레딩을 오판해 죽는 경우 방지
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
