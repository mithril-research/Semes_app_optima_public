#!/usr/bin/env bash
# 우분투 학습앱 진입점 — 내려받은 폴더 최상위에서 바로 실행한다.
#   최초 1회:  ./우분투_학습앱.sh setup     (venv + 패키지 + CUDA torch 설치)
#   실행:      ./우분투_학습앱.sh           → http://127.0.0.1:8000
#   포트 변경: PORT=8010 ./우분투_학습앱.sh
#
# 실제 내용은 src/app/run_ubuntu.sh 에 있다. 이 파일은 경로를 찾아 넘겨주기만 한다.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "$HERE/src/app/run_ubuntu.sh" "$@"
