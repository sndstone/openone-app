# PyInstaller spec for the OpenOne desktop app (Windows and macOS).
#   pyinstaller --noconfirm --clean packaging/openone.spec        (run from the repo root)
# The UI process has no torch: inference runs in a separate runtime (see app/openone_app/runtime.py), so the bundle
# stays ~30-40 MB. core/ and worker/ are also shipped as plain .py files for that runtime.
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
APP_VERSION = "0.2.0"
IS_MAC = sys.platform == "darwin"

datas = [
    (str(ROOT / "app" / "openone_app" / "web"), "openone_app/web"),
    (str(ROOT / "core" / "openone_core"), "core/openone_core"),
    (str(ROOT / "worker" / "openone_worker.py"), "worker"),
]
datas += collect_data_files("jsonschema_specifications")
datas += collect_data_files("webview")
hidden = collect_submodules("openone_core") + collect_submodules("openone_app") + ["jsonschema", "referencing"]
if IS_MAC:
    hidden += ["webview.platforms.cocoa"]
else:
    hidden += ["webview.platforms.winforms", "webview.platforms.edgechromium", "clr"]

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT / "app"), str(ROOT / "core")],
    datas=datas,
    hiddenimports=hidden,
    excludes=["torch", "transformers", "numpy", "tkinter", "matplotlib", "PIL", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="OpenOne",
    console=False,
    icon=str(ROOT / "packaging" / "assets" / ("icon.icns" if IS_MAC else "icon.ico")),
    version=str(ROOT / "packaging" / "windows" / "version_info.txt") if not IS_MAC else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="OpenOne")
if IS_MAC:
    app = BUNDLE(
        coll,
        name="OpenOne.app",
        icon=str(ROOT / "packaging" / "assets" / "icon.icns"),
        bundle_identifier="local.openone.app",
        version=APP_VERSION,
        info_plist={
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            "NSHumanReadableCopyright": "Private build. Model weights are not included.",
        },
    )
