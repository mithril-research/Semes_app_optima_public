"""Prepare once, then build without pip installation (Python 3.8 x64)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[2]


def run(args, env):
    subprocess.run([str(x) for x in args], cwd=str(ROOT), env=env, check=True)


def run_build(command, env, log, heartbeat_seconds=10):
    """Tee child output to console/log, with a liveness timer during quiet steps."""
    started = time.monotonic()
    stop = threading.Event()
    output_lock = threading.Lock()
    with log.open('w', encoding='utf-8') as stream:
        def emit(text):
            with output_lock:
                print(text, end='', flush=True)
                stream.write(text)
                stream.flush()

        with subprocess.Popen([str(x) for x in command], cwd=str(ROOT), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, encoding='utf-8', errors='replace',
                              bufsize=1) as process:
            def heartbeat():
                while not stop.wait(heartbeat_seconds):
                    if process.poll() is not None:
                        return
                    elapsed = int(time.monotonic() - started)
                    emit('[BUILD] Process running; elapsed {:02d}:{:02d} (not a progress percentage)\n'.format(
                        elapsed // 60, elapsed % 60))
            timer = threading.Thread(target=heartbeat, daemon=True)
            timer.start()
            try:
                for line in process.stdout:
                    emit(line)
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                raise
            finally:
                stop.set()
                timer.join()
            if code:
                raise RuntimeError('PyInstaller failed (exit {}). See {}'.format(code, log))
            emit('[BUILD] PyInstaller finished successfully.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'build'])
    parser.add_argument('--wheels', type=Path, help='Offline package directory (prepare only)')
    args = parser.parse_args()
    if os.name != 'nt' or sys.version_info[:2] != (3, 8) or struct.calcsize('P') != 8:
        raise RuntimeError('Use Windows Python 3.8 64-bit.')
    if args.wheels and args.action != 'prepare':
        parser.error('--wheels is only valid for prepare')
    backend = 'torch' if os.environ.get('IO_WES7_BACKEND', '').lower() == 'torch' else 'onnx'
    # Separate from iobuild7, which the original build BAT deletes.
    base = Path(os.environ.get('SystemDrive', 'C:') + '/semes_win7_prepared')
    venv = base / ('venv_' + backend)
    python = venv / 'Scripts/python.exe'
    req = ROOT / 'src/app' / ('requirements-win7-torch.txt' if backend == 'torch' else 'requirements-win7.txt')
    env = dict(os.environ, PYTHONUTF8='1', PIP_DISABLE_PIP_VERSION_CHECK='1', IO_WES7_BACKEND=backend)
    env['IO_WES7_UI'] = 'webview' if env.get('IO_WES7_UI', '').lower() == 'webview' else 'browser'
    if args.action == 'prepare':
        wheels = args.wheels.resolve() if args.wheels else ROOT / 'offline/win7' / backend
        if not wheels.is_dir():
            raise RuntimeError('Offline packages not found: ' + str(wheels) +
                               '. Bundled packages support ONNX only; supply --wheels for Torch.')
        manifest_file = wheels / 'manifest.json'
        lock = wheels / 'requirements.lock'
        install_req = req
        hashed = []
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
            if manifest.get('backend') != backend:
                raise RuntimeError('Offline package backend mismatch.')
            if hashlib.sha256(req.read_bytes()).hexdigest() != manifest['requirements_sha256']:
                raise RuntimeError('Requirements changed; refresh the offline package bundle.')
            for item in manifest['packages']:
                package = wheels / item['file']
                if not package.is_file() or hashlib.sha256(package.read_bytes()).hexdigest() != item['sha256']:
                    raise RuntimeError('Missing/corrupt offline package: ' + str(package))
            if not lock.is_file():
                raise RuntimeError('Offline requirements.lock is missing.')
            install_req = lock
            hashed = ['--require-hashes']
        elif not args.wheels:
            raise RuntimeError('Bundled offline manifest is missing.')
        if not python.exists():
            base.mkdir(parents=True, exist_ok=True)
            run([sys.executable, '-m', 'venv', venv], env)
        # Keep bundled/current pip. All installation inputs are local wheels.
        run([python, '-m', 'pip', 'install', '--no-index', '--find-links', wheels,
             '--only-binary=:all:'] + hashed + ['-r', install_req], env)
        run([python, '-m', 'pip', 'check'], env)
        print('Ready. Use WIN7_build_only.bat for subsequent builds.')
        return
    if not python.exists():
        raise RuntimeError('Run WIN7_prepare.bat first. No packages will be installed during build.')
    print('[1/4] Checking prepared Python and packages...', flush=True)
    run([python, '-c', 'import sys,struct; assert sys.version_info[:2]==(3,8) and struct.calcsize("P")==8'], env)
    run([python, '-m', 'pip', 'check'], env)
    run([python, '-c', 'import PyInstaller; import ' + ('torch' if backend == 'torch' else 'onnxruntime')], env)
    # A unique directory avoids stale output and preserves previous builds on failure.
    work = base / ('build_' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()))
    work.mkdir(parents=True)
    output_root = ROOT
    log_root = ROOT / 'build'
    log_root.mkdir(parents=True, exist_ok=True)
    output = output_root / 'ChamberInspect_WIN7'
    log = log_root / (work.name + '.log')
    print('App folder:', ROOT, flush=True)
    print('Build output:', output, flush=True)
    print('Build log:', log, flush=True)
    if output.exists():
        backup = output_root / ('ChamberInspect_WIN7_backup_' + work.name)
        output.rename(backup)
        print('Previous output:', backup, flush=True)
    print('[2/4] Building executable directly into the app folder...', flush=True)
    try:
        run_build([python, '-u', '-m', 'PyInstaller', '--clean', '--noconfirm',
                   '--distpath', output_root, '--workpath', work / 'work',
                   ROOT / 'src/app/io_console_win7.spec'], env, log)
        executable = output / 'ChamberInspect.exe'
        if not executable.is_file():
            raise RuntimeError('Expected executable not found: ' + str(executable))
        print('[3/4] Checking Windows 7 binary compatibility...', flush=True)
        run([python, ROOT / 'src/app/check_win7_compat.py', output], env)
    except Exception:
        print('[FAILED] Output may be incomplete or incompatible:', output, flush=True)
        print('See build log:', log, flush=True)
        raise
    print('[4/4] Output ready.', flush=True)
    print('[DONE] Build completed.', flush=True)
    print('Built:', output)
    print('Run:', executable)
    print('Prepared environment retained:', venv)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(1)
