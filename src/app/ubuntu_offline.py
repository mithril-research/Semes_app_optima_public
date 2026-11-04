"""Prepare on a matching online Linux PC; install without network or preinstalled pip."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import venv
import zipfile
from email.parser import Parser

HERE = Path(__file__).resolve().parent


def target():
    return {'python': f'{sys.version_info.major}.{sys.version_info.minor}',
            'machine': platform.machine(), 'system': platform.system()}


def run(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(folder):
    if target()['system'] != 'Linux' or not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise ValueError('Prepare on Linux using Python 3.10–3.12 matching the server.')
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError('Use an empty output directory to avoid mixing wheel versions.')
    wheels = folder / 'wheels'
    wheels.mkdir()
    pip = [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'download',
           '--only-binary=:all:', '--dest', wheels]
    run(pip + ['--no-deps', '--index-url', 'https://download.pytorch.org/whl/cpu',
               'torch==2.6.0+cpu'])
    constraint = folder / 'cpu-constraint.txt'
    constraint.write_text('torch==2.6.0+cpu\n')
    run(pip + ['--find-links', wheels, '-c', constraint,
               '-r', HERE / 'requirements-ubuntu-cpu.txt', 'pip>=24,<26', 'setuptools', 'wheel'])
    locked = []
    hashes = {}
    for wheel in sorted(wheels.glob('*.whl')):
        with zipfile.ZipFile(wheel) as z:
            metadata = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
            info = Parser().parsestr(z.read(metadata).decode('utf-8'))
        sha = digest(wheel)
        locked.append(f"{info['Name']}=={info['Version']} --hash=sha256:{sha}")
        hashes[wheel.name] = sha
    lock = folder / 'requirements.lock'
    lock.write_text('\n'.join(locked) + '\n', encoding='utf-8')
    (folder / 'manifest.json').write_text(json.dumps(
        {'target': target(), 'wheels': hashes, 'lock_sha256': digest(lock)}, indent=2))
    print(f'Offline packages ready: {folder} ({len(hashes)} wheels)', flush=True)


def verify(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['target'] != target():
        raise ValueError(f"Package target {manifest['target']} does not match server {target()}")
    if digest(folder / 'requirements.lock') != manifest['lock_sha256']:
        raise ValueError('requirements.lock checksum mismatch')
    actual = {p.name for p in (folder / 'wheels').iterdir()}
    if actual != set(manifest['wheels']):
        raise ValueError('Wheel files missing or unexpected; copy the complete original bundle.')
    for name, sha in manifest['wheels'].items():
        if Path(name).name != name or digest(folder / 'wheels' / name) != sha:
            raise ValueError(f'Wheel checksum mismatch: {name}')
    return manifest


def install(folder, destination):
    verify(folder)  # Check compatibility and integrity before changing the environment.
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use a new empty virtual environment; the existing environment is preserved.')
    venv.EnvBuilder(with_pip=False).create(destination)
    python = destination / 'bin/python'
    pip_wheels = list((folder / 'wheels').glob('pip-*.whl'))
    if len(pip_wheels) != 1:
        raise ValueError('Expected exactly one bundled pip wheel')
    env = dict(os.environ, PYTHONPATH=str(pip_wheels[0]), PIP_CONFIG_FILE=os.devnull,
               PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    run([python, '-m', 'pip', 'install', '--no-index', '--no-cache-dir',
         '--only-binary=:all:', '--find-links', folder / 'wheels', '--require-hashes',
         '-r', folder / 'requirements.lock'], env=env)
    env.pop('PYTHONPATH')
    run([python, '-m', 'pip', 'check'], env=env)
    run([python, '-c', 'import torch, onnxruntime, fastapi, numpy, pandas, matplotlib; '
         'print("CPU:", torch.ones(2).sum().item()); '
         'print("ONNX providers:", onnxruntime.get_available_providers())'], env=env)
    print(f'Offline installation complete: {destination}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'install', 'verify'])
    p.add_argument('folder', type=Path, help='Offline package directory')
    p.add_argument('--venv', type=Path, default=HERE / 'venv')
    a = p.parse_args()
    try:
        if a.action == 'prepare':
            prepare(a.folder.resolve())
        elif a.action == 'verify':
            verify(a.folder.resolve())
        else:
            install(a.folder.resolve(), a.venv.resolve())
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        p.exit(1, f'Offline setup failed: {e}\n')


if __name__ == '__main__':
    main()
