"""빌드 결과물이 Windows 7 / WES7 에서 '로드는 되는지' 정적 검사.

왜 이 검사가 필요한가
  Windows 는 DLL 을 열 때 임포트 테이블에 적힌 함수를 **로드 시점에 전부** 해석한다.
  그 중 하나라도 그 OS 의 DLL 에 없으면 프로세스가 시작조차 못 하고 죽는다.
  창 없는 exe 면 화면에 아무것도 안 남아서 원인을 찾기가 매우 어렵다.
  (실제로 torch 2.4.1 의 torch_cpu.dll 이 kernel32!CreateFile2 — Windows 8 부터 있는 API —
   를 정적 임포트해서 Win7 에서 import torch 가 죽는다.)

  ★ 문자열 검색으로는 판별할 수 없다. GetProcAddress 로 동적 조회하는 코드는 문자열이
    들어있어도 Win7 에서 잘 돈다(예: torch 의 VirtualAlloc2, SetThreadDescription).
    반드시 '임포트 테이블'을 봐야 한다. 이 스크립트가 하는 일이 그것이다.

사용법 (빌드 PC 에서, OS 무관)
    python check_win7_compat.py ChamberInspect_WES7
    python check_win7_compat.py ChamberInspect_WES7 --json 결과.json

종료 코드: 0=문제 없음, 1=Win7 에서 로드 실패할 파일 있음, 2=검사 자체 실패
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# Windows 8 이상에서 추가된 API (Win7 kernel32/api-set 에 없음).
# 정적 임포트로 잡히면 그 DLL 은 Win7 에서 열리지 않는다.
WIN8_PLUS = {
    "createfile2":                 "Windows 8",
    "virtualalloc2":               "Windows 10 1803",
    "mapviewoffile3":              "Windows 10 1803",
    "getthreaddescription":        "Windows 10 1607",
    "setthreaddescription":        "Windows 10 1607",
    "pathcchcombineex":            "Windows 8",
    "pathcchcanonicalizeex":       "Windows 8",
    "pathcchappendex":             "Windows 8",
    "setprocessmitigationpolicy":  "Windows 8",
    "getsystemcpusetinformation":  "Windows 10",
    "discardvirtualmemory":        "Windows 8.1",
    "offervirtualmemory":          "Windows 8.1",
    "reclaimvirtualmemory":        "Windows 8.1",
    "waitonaddress":               "Windows 8",
    "wakebyaddresssingle":         "Windows 8",
    "wakebyaddressall":            "Windows 8",
    "getoverlappedresultex":       "Windows 8",
    "quirkisenabled":              "Windows 8",
    "compareobjecthandles":        "Windows 10",
    "createwaitabletimerexw":      "Windows 7",       # 참고용(있어도 무방)
}
# 위 목록 중 Win7 에 실제로 존재하는 것은 검사에서 뺀다.
WIN8_PLUS.pop("createwaitabletimerexw", None)


def _imports(data: bytes):
    """PE 파일에서 (종류, DLL명) -> [함수명] 을 뽑는다. 종류: import(정적) / delay(지연)."""
    if data[:2] != b"MZ":
        return None
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return None
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    optsz = struct.unpack_from("<H", data, pe + 20)[0]
    magic = struct.unpack_from("<H", data, pe + 24)[0]
    pe32plus = magic == 0x20B
    dd = pe + 24 + (112 if pe32plus else 96)
    secs = []
    so = pe + 24 + optsz
    for i in range(nsec):
        b = so + 40 * i
        va = struct.unpack_from("<I", data, b + 12)[0]
        rawsz = struct.unpack_from("<I", data, b + 16)[0]
        raw = struct.unpack_from("<I", data, b + 20)[0]
        secs.append((va, rawsz, raw))

    def off(rva):
        for va, rawsz, raw in secs:
            if va <= rva < va + rawsz:
                return raw + (rva - va)
        return None

    def cstr(o):
        e = data.index(b"\0", o)
        return data[o:e].decode("ascii", "replace")

    out = {}
    for idx, kind in ((1, "import"), (13, "delay")):
        rva, _size = struct.unpack_from("<II", data, dd + 8 * idx)
        if not rva:
            continue
        o = off(rva)
        if o is None:
            continue
        step = 20 if idx == 1 else 32
        while True:
            ent = data[o:o + step]
            if len(ent) < step or not any(ent):
                break
            if idx == 1:
                oft, _, _, nrva, frva = struct.unpack_from("<IIIII", data, o)
                thunk = oft or frva
            else:
                _, nrva, _, thunk = struct.unpack_from("<IIII", data, o)[:4]
            if not nrva:
                break
            no = off(nrva)
            if no is None:
                break
            dll = cstr(no).lower()
            fns = []
            to = off(thunk)
            if to is not None:
                while True:
                    if pe32plus:
                        v = struct.unpack_from("<Q", data, to)[0]
                        step2, hi = 8, 1 << 63
                    else:
                        v = struct.unpack_from("<I", data, to)[0]
                        step2, hi = 4, 1 << 31
                    if v == 0:
                        break
                    if not (v & hi):                     # 서수(ordinal) 임포트가 아니면 이름 있음
                        po = off(v & 0x7FFFFFFF)
                        if po is not None:
                            fns.append(cstr(po + 2))
                    to += step2
            out.setdefault((kind, dll), []).extend(fns)
            o += step
    return out


def scan(root: Path):
    files, problems, skipped = 0, [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".dll", ".pyd", ".exe"):
            continue
        try:
            data = p.read_bytes()
        except Exception as e:                           # noqa: BLE001
            skipped.append((str(p), f"읽기 실패: {e}"))
            continue
        try:
            imp = _imports(data)
        except Exception as e:                           # noqa: BLE001
            skipped.append((str(p.relative_to(root)), f"파싱 실패: {e}"))
            continue
        if imp is None:
            continue
        files += 1
        hits = []
        for (kind, dll), fns in imp.items():
            if kind != "import":                         # 지연로드는 실행 시점 조회 → Win7 무해
                continue
            for f in fns:
                since = WIN8_PLUS.get(f.lower())
                if since:
                    hits.append({"dll": dll, "func": f, "since": since})
        if hits:
            problems.append({"file": str(p.relative_to(root)), "imports": hits})
    return {"scanned": files, "problems": problems, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser(description="빌드 결과물의 Windows 7 로드 가능성 검사")
    ap.add_argument("folder", help="검사할 폴더 (예: ChamberInspect_WES7)")
    ap.add_argument("--json", help="결과를 JSON 으로 저장할 경로")
    a = ap.parse_args()

    root = Path(a.folder)
    if not root.is_dir():
        print(f"[X] 폴더가 없습니다: {root}")
        return 2

    print("=" * 66)
    print("  Windows 7 / WES7 로드 가능성 검사 (PE 임포트 테이블)")
    print(f"  대상: {root.resolve()}")
    print("=" * 66)
    r = scan(root)
    print(f"\n검사한 실행 파일: {r['scanned']}개")

    if r["problems"]:
        print(f"\n★ Windows 7 에서 로드에 실패할 파일 {len(r['problems'])}개:\n")
        for p in r["problems"]:
            print(f"  ✗ {p['file']}")
            for h in p["imports"]:
                print(f"      {h['dll']}!{h['func']}  ({h['since']} 부터 존재)")
        print("\n  → 이 파일들은 Windows 7 에서 프로세스가 시작조차 못 합니다.")
        print("    해당 패키지의 버전을 낮춰 다시 빌드하세요.")
        print("    (예: torch 는 2.3.1 이하 — 2.4.1 은 CreateFile2 를 정적 임포트합니다)")
    else:
        print("\n✅ 정적 임포트에서 Windows 8+ 전용 API 를 쓰는 파일이 없습니다.")
        print("   → DLL 로딩 단계는 통과할 것으로 보입니다.")

    # UCRT 가 앱 폴더에 들어 있는지(윈도우7 에서 KB2999226 없이 돌리려면 필요)
    ucrt = sorted(root.glob("api-ms-win-crt-*.dll"))
    base = (root / "ucrtbase.dll").exists()
    if ucrt and base:
        print(f"\n✅ UCRT 포함: api-ms-win-crt-*.dll {len(ucrt)}개 + ucrtbase.dll")
        print("   → 윈도우7 에 KB2999226 이 없어도 실행됩니다.")
    else:
        print("\n※ 앱 폴더에 UCRT 가 없습니다"
              f"(api-ms-win-crt-*: {len(ucrt)}개, ucrtbase.dll: {'있음' if base else '없음'}).")
        print("   윈도우7 제어기에 KB2999226(Universal C Runtime)을 설치해야 합니다.")
        print("   설치하지 않으면 'api-ms-win-crt-runtime-l1-1-0.dll 이 없습니다' 오류가 납니다.")

    if r["skipped"]:
        print(f"\n(참고) 검사하지 못한 파일 {len(r['skipped'])}개:")
        for f, why in r["skipped"][:5]:
            print(f"   - {f}: {why}")

    print("\n※ 이 검사는 '로드가 되는가'만 봅니다. 실행 중 다른 이유로 막힐 가능성까지")
    print("   배제하지는 못하므로, 실기(또는 Win7 VM)에서 한 번은 띄워 보세요.")

    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 저장: {a.json}")
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
