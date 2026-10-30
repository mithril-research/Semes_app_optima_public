"""Chamber artifact discovery and request-local engine selection (stdlib only)."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import threading


@dataclass(frozen=True)
class ChamberArtifact:
    chamber: str
    onnx: Path
    meta: Path
    config: Path


def discover_chamber_models(folder):
    """Discover models only inside DSV1 through DSV6 subdirectories."""
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    entries = list(folder.iterdir())
    candidates = {}
    for p in entries:
        if p.is_dir() and re.fullmatch(r'dsv[1-6]', p.name, re.I):
            models = sorted(q for q in p.iterdir() if q.suffix.lower() == '.onnx')
            candidates.setdefault(p.name.lower(), []).extend(models)
    out = {}
    for ch, models in sorted(candidates.items()):
        if len(models) != 1:
            raise ValueError(f'{ch.upper()}: ONNX 모델은 하나만 두세요 ({len(models)}개).')
        onnx = models[0]
        files = {p.name.lower(): p for p in onnx.parent.iterdir() if p.is_file()}
        meta_names = [onnx.with_suffix('.npz').name.lower(), 'recon_meta.npz']
        config_names = ['search_best.json', 'best_search.json']
        meta = next((files[n] for n in meta_names if n in files), None)
        config = next((files[n] for n in config_names if n in files), None)
        if meta is None:
            raise ValueError(f'{ch.upper()}: {onnx.name}의 센서·정규화 메타 NPZ가 없습니다. '
                             'recon_meta.npz 또는 모델과 같은 이름의 .npz를 함께 넣으세요.')
        if config is None:
            raise ValueError(f'{ch.upper()}: search_best.json 또는 best_search.json이 없습니다.')
        out[ch] = ChamberArtifact(ch, onnx, meta, config)
    return out


def has_chamber_layout(folder):
    folder = Path(folder)
    return folder.is_dir() and any(
        p.is_dir() and re.fullmatch(r'dsv[1-6]', p.name, re.I)
        for p in folder.iterdir())


def artifact_digest(artifacts):
    """Content-based bundle identity: model, schema AND decision config; independent of install path."""
    h = hashlib.sha256(b"chamber-routing:C2:v1")  # invalidate results inferred with the former C3 routing
    for ch, spec in sorted(artifacts.items()):
        h.update(ch.encode())
        for kind, path in [('onnx', spec.onnx), ('meta', spec.meta), ('config', spec.config)]:
            h.update(kind.encode())
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
    return 'm@chambers-' + h.hexdigest()[:24]


class EngineRouter:
    """Existing helpers read ENGINE; contexts isolate each chamber across concurrent requests."""
    def __init__(self, engine, busy=None):
        object.__setattr__(self, '_root', engine)
        object.__setattr__(self, '_busy', busy or (lambda: False))
        object.__setattr__(self, '_scope', ContextVar('chamber_engine', default=None))
        object.__setattr__(self, '_snapshot', ContextVar('engine_snapshot', default=None))
        object.__setattr__(self, '_switch_lock', threading.RLock())

    @property
    def root(self):
        return self._snapshot.get() or self._root

    def __getattr__(self, name):
        return getattr(self._scope.get() or self.root, name)

    def __setattr__(self, name, value):
        setattr(self._scope.get() or self.root, name, value)

    def for_chamber(self, chamber):
        root = self.root
        if not root.chamber_engines:
            return root
        ch = str(chamber or '').lower()
        if ch not in root.chamber_engines:
            raise ValueError(f'챔버 모델 없음: {ch or "로그 2행 C열 챔버 미확인"}. 해당 웨이퍼는 추론하지 않습니다.')
        return root.chamber_engines[ch]

    @contextmanager
    def using(self, engine):
        token = self._scope.set(engine)
        try:
            yield engine
        finally:
            self._scope.reset(token)

    @contextmanager
    def pinned(self):
        token = self._snapshot.set(self.root)
        try:
            yield
        finally:
            self._snapshot.reset(token)

    def select(self, model_dir):
        # Publish only fully loaded models. Ongoing requests retain their pinned snapshot.
        with self._switch_lock:
            if self._busy():
                return {'ready': False, 'error': '검사·리포트 작업 완료 후 모델을 변경하세요.'}
            candidate = type(self._root)()
            candidate.selected = Path(model_dir) if model_dir else None
            candidate.load()
            if not candidate.ready or candidate.error:
                return {'ready': False, 'error': candidate.error, 'backend': candidate.backend}
            if self._busy():
                return {'ready': False, 'error': '검사·리포트 작업이 시작되어 모델 변경을 보류했습니다. 완료 후 다시 적용하세요.'}
            object.__setattr__(self, '_root', candidate)
            return candidate.info()

    def reload(self):
        return self.select(str(self.root.selected or self.root.art))


@contextmanager
def canonical_artifacts(spec):
    """Give directory-based RCA tools only the selected chamber's files, without editing originals."""
    import os
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory(prefix='chamber-model-') as td:
        target = Path(td)
        for source, name in [(spec.onnx, 'tcn_forecast.onnx'),
                             (spec.meta, 'recon_meta.npz'),
                             (spec.config, 'search_best.json')]:
            try:
                os.link(str(source), str(target / name))
            except OSError:
                shutil.copy2(source, target / name)
        yield target
