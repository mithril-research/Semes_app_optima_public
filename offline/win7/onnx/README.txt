Windows 7 / CPython 3.8 x64 / ONNX offline build packages

Run WIN7_prepare.bat, then WIN7_build_only.bat from the repository root.
No PyPI connection or pip upgrade is performed. Git must download these files first.
Existing build BAT files are unchanged and still use their original online workflow.
Python 3.8 x64 and Windows system prerequisites must already be installed.
Torch packages are NOT included.

Sources: https://pypi.org/simple/ (official PyPI).
50 wheels were downloaded directly. proxy_tools 0.1.0 is a pure-Python
wheel built with pip wheel from its official PyPI source distribution.
Licenses are retained inside the wheels. See each dist-info directory.
manifest.json records SHA256 hashes. requirements.lock pins every package.

Validation: all dependency markers evaluated for Windows Python 3.8.10;
142 PE binaries scanned with the repository Windows 7 compatibility checker.
No flagged Windows 8+ static API imports. This is NOT a real Windows 7 run.
Actual Windows installation/build and pip 21.1.1 execution remain unverified.
