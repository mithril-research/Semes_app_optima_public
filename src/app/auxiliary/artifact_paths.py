"""Portable, package-local metadata file references."""
from pathlib import Path, PureWindowsPath


def artifact_file(root, value):
    text = str(value).replace("\\", "/")
    if not text or Path(text).is_absolute() or PureWindowsPath(text).drive:
        raise ValueError("아티팩트 파일은 패키지 내부 상대경로여야 합니다: " + text)
    root = Path(root).resolve()
    path = (root / text).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("아티팩트 폴더 밖의 파일 경로: " + text)
    return path
