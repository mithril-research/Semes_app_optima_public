# -*- coding: utf-8 -*-
"""빌드 산출물 정리 — 실행 파일만 최상위에 남기고 지원 파일은 build\\ 로 옮긴다.

왜 필요한가
  PyInstaller onedir 은 exe 와 수천 개의 DLL·데이터를 한 폴더에 쏟아낸다.
  고객사에서 "어느 것을 실행해야 하느냐"는 문의가 반복됐다.
  PyInstaller 6.x 에는 지원 파일을 하위 폴더로 몰아주는 옵션(contents-directory)이
  있지만, 윈도우7 부트로더가 있는 마지막 버전이 5.13.2 라 쓸 수 없다.

어떻게 하는가
  exe 는 '자기가 놓인 폴더'에서 지원 파일을 찾는다. 그래서 exe 만 위로 빼면 실행되지
  않는다. 대신 이렇게 한다.

      ChamberInspect_WIN7\\
        ChamberInspect.exe        <- 새로 만든 '실행 전용' 파일(런처)
        build\\
          ChamberInspect.exe      <- 원본(지원 파일과 함께 있어야 동작)
          ... 지원 파일 수천 개

  최상위 런처는 build\\ChamberInspect.exe 를 그대로 띄우는 아주 작은 프로그램이다.
  인자·종료코드를 그대로 전달하므로 사용자에겐 차이가 없다.

런처를 만드는 방법(우선순위)
  ① 윈도우 네이티브 exe — 이 스크립트를 '빌드 PC'에서 돌릴 때 PyInstaller 로 즉석 생성.
     제어기에 파이썬이 없어도 되고, 아이콘도 그대로 쓴다.
  ② 실패 시 .bat — exe 를 못 만들면 '앱 실행.bat' 을 두고 안내한다(동작은 동일).

되돌리기
  build\\ 안의 원본 exe 를 그대로 실행해도 된다. 구조를 바꾸는 것이지 파일을 고치는
  것이 아니라, 언제든 build\\ 내용을 상위로 도로 옮기면 원래 상태가 된다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

APP = "ChamberInspect.exe"
SUB = "build"                       # 지원 파일을 담을 하위 폴더 이름
# 최상위에 함께 남길 것(사용자가 직접 열어야 하는 것들)
KEEP_TOP = {"데이터", "보관", "로그", "리포트", "storage.json",
            "앱_시작오류.txt", "앱_시작기록.txt", "앱_크래시.txt"}

LAUNCHER_SRC = r'''# -*- coding: utf-8 -*-
"""ChamberInspect 실행기 — build 폴더의 본체를 띄운다(사용자에겐 동일)."""
import os, subprocess, sys
from pathlib import Path

def main() -> int:
    here = Path(sys.executable).resolve().parent
    target = here / "build" / "ChamberInspect.exe"
    if not target.exists():
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, "build 폴더에서 실행 파일을 찾지 못했습니다.\n\n"
                   f"찾은 위치: {target}\n\n"
                   "build 폴더가 지워졌거나 이동된 것 같습니다. 앱을 다시 설치해 주세요.",
                "ChamberInspect", 0x10 | 0x40000 | 0x10000)
        except Exception:
            print(f"실행 파일 없음: {target}", file=sys.stderr)
        return 2
    # ★ 본체를 띄우고 런처는 곧바로 끝낸다.
    #   예전에는 subprocess.call 로 '끝날 때까지 기다려서', 앱을 켜는 동안
    #   런처가 계속 살아 있었다. 런처는 onefile 이라 부팅용 프로세스까지 두 개를
    #   차지하므로, 작업관리자에 ChamberInspect 가 본체까지 3개로 보였다.
    #   기다릴 이유가 없다 — 창은 본체가 띄우고, 종료도 본체가 알아서 한다.
    creation = 0
    if os.name == "nt":
        # 런처가 사라져도 본체가 함께 죽지 않도록 새 프로세스 그룹으로 분리
        creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200) \
                 | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    subprocess.Popen([str(target)] + sys.argv[1:], cwd=str(target.parent),
                     close_fds=True, creationflags=creation)
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


def _make_launcher_exe(out_dir: Path, icon: Path | None) -> bool:
    """PyInstaller 로 런처 exe 생성(빌드 PC 에서만 가능). 성공 시 True."""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("  · PyInstaller 없음 → 런처 exe 생성 불가")
        return False
    tmp = Path(tempfile.mkdtemp(prefix="chlaunch_"))
    try:
        src = tmp / "launch_app.py"
        src.write_text(LAUNCHER_SRC, encoding="utf-8")
        cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
               "--onefile", "--windowed", "--name", "ChamberInspect",
               "--distpath", str(tmp / "dist"), "--workpath", str(tmp / "work"),
               "--specpath", str(tmp)]
        if icon and icon.exists():
            cmd += ["--icon", str(icon)]
        cmd.append(str(src))
        r = subprocess.run(cmd, capture_output=True, text=True)
        # 윈도우는 ChamberInspect.exe, 그 외 OS 는 확장자 없이 나온다.
        made = tmp / "dist" / APP
        if not made.exists():
            alt = tmp / "dist" / Path(APP).stem
            if alt.exists():
                made = alt
        if r.returncode != 0 or not made.exists():
            print(f"  · 런처 exe 빌드 실패(코드 {r.returncode})")
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            for ln in tail:
                print(f"      {ln[:120]}")
            return False
        shutil.copy2(made, out_dir / APP)
        return True
    except Exception as e:                                # noqa: BLE001
        print(f"  · 런처 exe 생성 중 오류: {type(e).__name__}: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _make_launcher_bat(out_dir: Path) -> None:
    """exe 를 못 만들 때의 대안 — 배치 파일."""
    bat = ("@echo off\r\n"
           "chcp 949 >nul\r\n"
           "title ChamberInspect\r\n"
           "cd /d \"%~dp0build\"\r\n"
           "start \"\" \"ChamberInspect.exe\" %*\r\n")
    (out_dir / "앱 실행.bat").write_bytes(bat.encode("cp949"))


def arrange(root: Path) -> int:
    root = Path(root).resolve()
    if not root.is_dir():
        print(f"[X] 폴더가 없습니다: {root}")
        return 2
    sub = root / SUB
    if sub.is_dir() and (sub / APP).exists():
        print(f"  · 이미 정리된 폴더입니다: {sub}")
        return 0
    if not (root / APP).exists():
        print(f"[X] {APP} 를 찾지 못했습니다: {root}")
        return 2

    icon = None
    for cand in (root / "icon.ico", root / "app.ico"):
        if cand.exists():
            icon = cand; break

    sub.mkdir(parents=True, exist_ok=True)
    moved = 0
    for item in list(root.iterdir()):
        if item.name == SUB or item.name in KEEP_TOP:
            continue
        try:
            shutil.move(str(item), str(sub / item.name))
            moved += 1
        except Exception as e:                            # noqa: BLE001
            print(f"  · 이동 실패({item.name}): {e}")
    print(f"  · 지원 파일 {moved}개를 {SUB}\\ 로 옮겼습니다")

    if _make_launcher_exe(root, icon):
        print(f"  · 최상위에 실행 파일 생성: {APP}")
    else:
        _make_launcher_bat(root)
        print("  · 런처 exe 대신 '앱 실행.bat' 을 두었습니다")

    # ★ 남아 있는 프로세스 정리 — 예전 버전으로 실행해 두었던 것이 남아 있으면
    #   메모리를 먹고 로그 파일을 물고 있어 폴더 압축도 안 된다. 한 번 눌러 정리한다.
    #   (지금 버전은 브라우저를 닫으면 스스로 끝나고, 중복 실행도 막는다)
    (root / "프로세스 정리.bat").write_bytes(
        ("@echo off\r\n"
         "chcp 949 >nul\r\n"
         "echo.\r\n"
         "echo   ChamberInspect - 남아 있는 프로그램 정리\r\n"
         "echo   ------------------------------------------\r\n"
         "echo   실행 중인 ChamberInspect 를 모두 종료합니다.\r\n"
         "echo   (검사 중이면 그 작업도 중단됩니다)\r\n"
         "echo.\r\n"
         "pause\r\n"
         "tasklist /FI \"IMAGENAME eq ChamberInspect.exe\" 2>nul | find /I \"ChamberInspect.exe\" >nul\r\n"
         "if errorlevel 1 (\r\n"
         "  echo   남아 있는 프로그램이 없습니다.\r\n"
         ") else (\r\n"
         "  taskkill /F /IM ChamberInspect.exe /T\r\n"
         "  echo.\r\n"
         "  echo   정리했습니다.\r\n"
         ")\r\n"
         "echo.\r\n"
         "pause\r\n").encode("cp949"))

    (root / "읽어주세요.txt").write_bytes(
        ("ChamberInspect - 실행 방법\r\n"
         "==============================\r\n\r\n"
         "  ChamberInspect.exe 를 더블클릭하세요.\r\n\r\n"
         "  build 폴더에는 앱이 쓰는 파일들이 들어 있습니다.\r\n"
         "  지우거나 옮기면 앱이 실행되지 않습니다.\r\n\r\n"
         "  기록이 남는 곳\r\n"
         "    로그\\app_날짜.log    실행 기록\r\n"
         "    앱_시작오류.txt      시작에 실패했을 때\r\n\r\n"
         "  브라우저 창을 닫으면 프로그램도 함께 종료됩니다.\r\n"
         "  (검사 중이면 끝난 뒤에 종료됩니다)\r\n\r\n"
         "  이미 실행 중일 때 다시 누르면 새로 뜨지 않고 화면만 다시 열립니다.\r\n\r\n"
         "  예전 버전이 남아 있어 느려지거나 폴더 압축이 안 되면\r\n"
         "  '프로세스 정리.bat' 을 한 번 실행하세요.\r\n").encode("cp949"))
    return 0


if __name__ == "__main__":
    sys.exit(arrange(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
