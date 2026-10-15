# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the OllaVoice Windows executable.

Build it *on Windows* (PyInstaller cannot cross-compile)::

    scripts\\build-windows-exe.ps1

The result is a self-contained folder in dist\\OllaVoice\\ containing
OllaVoice.exe. Speech model weights are deliberately NOT bundled - they are
several hundred MB, they differ per machine tier, and setup already knows how to
fetch or restore them offline. The exe reads them from the usual place:

    %LOCALAPPDATA%\\OllaVoice\\models

Two things make this spec bigger than a typical one, and both are unavoidable:

* ``ctranslate2`` and ``onnxruntime`` ship large native libraries;
* ``piper`` and ``espeakng_loader`` each carry a ~370-file ``espeak-ng-data``
  tree that the phonemiser loads by path at runtime, so it must be collected as
  data rather than as Python modules.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# Packages that load data files / native libraries by path at runtime.
for package in ("piper", "espeakng_loader", "ctranslate2", "onnxruntime",
                "faster_whisper", "kokoro_onnx", "tokenizers", "soundfile",
                "huggingface_hub", "av", "numpy"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as exc:      # an optional extra may simply not be installed
        print(f"[ollavoice] skipping {package}: {exc}")

# Our own package data: the browser UI served by `ollavoice serve`.
datas += [("ollavoice/server/static", "ollavoice/server/static")]

# Submodules imported lazily (engine factories, GUI, server) must be listed.
hiddenimports += collect_submodules("ollavoice")
hiddenimports += [
    "ollavoice.gui.app",
    "ollavoice.gui.main_window",
    "ollavoice.gui.settings_dialog",
    "ollavoice.gui.workers",
    "ollavoice.gui.hotkeys",
    "ollavoice.gui.theme",
    "ollavoice.server",
    "ollavoice.repl",
    "ollavoice.setup",
    "sounddevice",
]

block_cipher = None

a = Analysis(
    ["run_gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Nothing here is used by OllaVoice, and each one costs tens of MB.
        "tkinter", "matplotlib", "scipy", "pandas", "PIL", "IPython", "jupyter",
        "notebook", "sympy", "torch", "tensorflow", "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OllaVoice",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                       # UPX corrupts some onnxruntime/ct2 DLLs
    console=False,                   # windowed app; logs go to %APPDATA%\OllaVoice\logs
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="resources/ollavoice.ico",   # add an .ico here if you have one
    version=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="OllaVoice",
)
