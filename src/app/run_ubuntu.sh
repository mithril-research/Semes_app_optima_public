#!/usr/bin/env bash
# 우분투 학습앱 실행 스크립트 (내부용 — 브라우저 접속, 네이티브 뷰 없음)
#   설치(최초 1회):  ./run_ubuntu.sh setup
#   실행:            ./run_ubuntu.sh          → http://127.0.0.1:8000 (Firefox 자동 오픈)
#   포트 변경:       PORT=8010 ./run_ubuntu.sh
#
#   저장소 위치(선택): 데이터·보관·로그·리포트·학습작업 폴더를 앱 밖 공용 경로로 두려면
#     이 스크립트와 같은 폴더(<패키지>/app/)에 storage.json 을 만들거나 설정 탭에서 지정.
#       {"data":"/mnt/io/데이터","archive":"/mnt/io/보관","train":"/mnt/io/train_jobs"}
#     앱을 새로 받아도 이 파일만 두면 기존 이력·학습 결과를 그대로 이어서 씁니다(복사 불필요).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"          # .../<패키지>/app
PKG_DIR="$(dirname "$HERE")"                   # 패키지 폴더
PKG_PARENT="$(dirname "$PKG_DIR")"             # 패키지의 부모(import 루트)
# ★ 패키지 폴더 이름을 고정하지 않는다. 저장소에서는 src/, 배포본에서는 io_recon_tcn/ 이라
#   이름을 박아 두면 한쪽에서 "No module named ..." 로 안 뜬다.
PKG_NAME="$(basename "$PKG_DIR")"
VENV="$HERE/venv"
PORT="${PORT:-8000}"

if [ "$1" = "setup" ]; then
  python3 -m venv "$VENV"
  . "$VENV/bin/activate"
  pip install -U pip
  pip install -r "$HERE/requirements-ubuntu.txt"
  # CUDA torch (장비 CUDA 에 맞게 — nvidia-smi 우상단 CUDA Version 확인)
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  # ★ CPU 판(onnxruntime)이 함께 깔려 있으면 GPU 판을 가려서 "학습은 CUDA, 추론만 CPU" 가 된다.
  #   두 패키지는 모듈명이 같아 나중에 잡히는 쪽이 이긴다 → CPU 판을 제거한다.
  if pip show onnxruntime >/dev/null 2>&1 && pip show onnxruntime-gpu >/dev/null 2>&1; then
    echo "⚠ onnxruntime(CPU)과 onnxruntime-gpu 가 함께 설치돼 있어 CPU 판을 제거합니다."
    pip uninstall -y onnxruntime
  fi
  echo "── 추론 GPU 확인 ──"
  python3 - <<'PYCHK'
try:
    import onnxruntime as ort
    ps = ort.get_available_providers()
    print("  onnxruntime", ort.__version__, "providers:", ps)
    print("  CUDA 추론:", "가능" if "CUDAExecutionProvider" in ps else
          "불가 — pip install \"onnxruntime-gpu>=1.17,<1.23\" 확인 필요")
except Exception as e:
    print("  onnxruntime 확인 실패:", e)
try:
    import torch
    print("  학습(torch) CUDA:", torch.cuda.is_available())
except Exception as e:
    print("  torch 확인 실패:", e)
PYCHK
  echo "✅ 설치 완료. ./run_ubuntu.sh 로 실행하세요."
  echo "   (설정 탭 > GPU 가속 에서 'CUDA' 선택 후 저장 → 자동 재로드)"
  exit 0
fi

[ -d "$VENV" ] || { echo "먼저 ./run_ubuntu.sh setup 을 실행하세요"; exit 1; }
. "$VENV/bin/activate"
cd "$PKG_PARENT"
export IO_HOST="${IO_HOST:-127.0.0.1}"         # 내부망 공유 시 IO_HOST=0.0.0.0 (방화벽 유의)
( sleep 3; command -v firefox >/dev/null && firefox "http://127.0.0.1:$PORT" >/dev/null 2>&1 & ) &
exec python3 -m "$PKG_NAME.app.server"
