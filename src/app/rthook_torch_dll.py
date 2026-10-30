"""PyInstaller 런타임 훅 — 얼린 exe 안에서 torch 의 DLL 검색 경로를 잡아 준다.

★ 왜 필요한가
  torch 는 평소(설치된 상태)에는 자기 패키지의 `torch\\lib` 폴더를
  os.add_dll_directory() 로 등록한 뒤 torch_cpu.dll·c10.dll 등을 불러온다.
  그런데 PyInstaller 로 얼리면 torch/__init__.py 의 그 경로 계산이 어긋나서,
  DLL 들이 번들 안에 들어 있어도 로더가 찾지 못한다.

  결과: torch 를 import 하는 순간 '지연 로드 DLL 을 찾지 못함'
  (종료코드 0xC06D007E) 으로 프로세스가 즉시 사라진다. 예외도 로그도 없다.

  이 훅은 앱 코드보다 먼저 실행되므로, import torch 가 일어나기 전에
  번들 폴더들을 DLL 검색 경로와 PATH 에 넣어 둔다.
"""
import os
import sys

if os.name == "nt":
    _base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    _cands = [
        _base,                                            # onedir 루트(대부분 여기에 풀린다)
        os.path.join(_base, "torch", "lib"),              # 원래 구조를 유지한 경우
        os.path.join(_base, "_internal"),                 # PyInstaller 6 계열 배치
        os.path.join(_base, "_internal", "torch", "lib"),
    ]
    _added = []
    for _p in _cands:
        try:
            if not os.path.isdir(_p):
                continue
            if hasattr(os, "add_dll_directory"):          # 파이썬 3.8+
                os.add_dll_directory(_p)
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
            _added.append(_p)
        except Exception:                                 # noqa: BLE001
            continue
    try:
        print(f"[dll] torch DLL 검색 경로 {len(_added)}곳 등록", flush=True)
    except Exception:                                     # noqa: BLE001
        pass
