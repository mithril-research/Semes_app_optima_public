"""PyInstaller 런타임 훅 — 시작 실패 흔적을 무조건 남긴다.

★ 왜 필요한가
  배포 exe 는 console=False(창 없음)라 stdout/stderr 가 없다. 이 상태에서
   · 파이썬 예외가 나면 → 화면에 아무것도 안 뜨고 종료
   · 네이티브 크래시(예: DirectML/DLL 로드 실패로 인한 접근위반)가 나면 → 흔적조차 없음
  둘 다 사용자에게는 똑같이 "더블클릭해도 아무 반응 없음" 으로 보인다.

이 훅은 앱 코드보다 **먼저** 실행되므로(런타임 훅), main() 안의 오류처리가
동작하기도 전에 죽는 경우까지 덮는다.
  · faulthandler → 네이티브 크래시의 파이썬 스택을 '앱_크래시.txt' 로 덤프
  · sys.excepthook → 잡히지 않은 파이썬 예외를 같은 파일에 기록
쓰기 권한이 없으면 임시 폴더로 물러난다(로그조차 못 남기는 일 방지).
"""
import os
import sys


def _crash_path():
    try:
        base = (os.path.dirname(sys.executable)
                if getattr(sys, "frozen", False) else os.getcwd())
        p = os.path.join(base, "앱_크래시.txt")
        with open(p, "a", encoding="utf-8-sig"):         # 쓰기 가능 여부 실제 확인
            pass
        return p
    except Exception:                                    # noqa: BLE001
        import tempfile
        return os.path.join(tempfile.gettempdir(), "앱_크래시.txt")


try:
    _PATH = _crash_path()
    # ★ 한국어판 Windows 메모장은 BOM 없는 UTF-8 을 ANSI 로 읽어 한글이 깨진다 → utf-8-sig
    _FP = open(_PATH, "a", buffering=1, encoding="utf-8-sig", errors="replace")

    import faulthandler
    faulthandler.enable(file=_FP, all_threads=True)      # ★ 네이티브 크래시 덤프

    import time
    _FP.write(f"\n===== 실행 시작 {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"pid={os.getpid()} =====\n")

    def _hook(etype, evalue, tb):
        try:
            import traceback
            _FP.write("--- 처리되지 않은 오류 ---\n")
            traceback.print_exception(etype, evalue, tb, file=_FP)
            _FP.flush()
        except Exception:                                # noqa: BLE001
            pass
        try:                                             # 사용자에게도 보이게
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, f"{etype.__name__}: {evalue}\n\n자세한 내용:\n{_PATH}",
                "챔버 이상 검사 시스템 — 시작 실패", 0x10)
        except Exception:                                # noqa: BLE001
            pass

    sys.excepthook = _hook
except Exception:                                        # noqa: BLE001
    pass                                                 # 진단 장치가 앱을 막으면 안 된다
