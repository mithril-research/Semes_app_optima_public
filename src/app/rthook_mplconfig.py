# PyInstaller 런타임 훅: matplotlib 설정/폰트캐시 디렉터리를 '쓰기 가능한 임시폴더'로 지정.
#   frozen exe 에서 matplotlib 자동 훅(pyi_rth_mplconfig)이 버전 불일치로 어긋나도
#   앱이 정상 부팅되도록 보강한다(MPLCONFIGDIR/ MPLBACKEND 를 우리가 직접 설정).
import os
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")               # GUI 백엔드 회피(서버·리포트용)
try:
    _d = os.path.join(tempfile.gettempdir(), "io_mplcfg")
    os.makedirs(_d, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _d                       # 쓰기 가능 위치(권한 문제 회피)
except Exception:                                        # noqa: BLE001
    pass
