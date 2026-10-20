"""Where things live: bundled resources (dev checkout or PyInstaller bundle) and the per-user data folder."""
import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)


def resource_root():
    if FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]                       # repo root: app/, core/, worker/


def core_dir():
    return resource_root() / "core"


def worker_script():
    return resource_root() / "worker" / "openone_worker.py"


def web_dir():
    if FROZEN:
        return resource_root() / "openone_app" / "web"
    return Path(__file__).resolve().parent / "web"


def install_dir():
    """Folder of the installed app (where an offline installer puts runtime/)."""
    if FROZEN:
        exe = Path(sys.executable).resolve()
        if sys.platform == "darwin" and exe.parent.name == "MacOS":
            return exe.parent.parent / "Resources"
        return exe.parent
    return resource_root() / "dist-runtime"


def data_dir():
    env = os.environ.get("OPENONE_DATA_DIR")
    if env:
        d = Path(env)
    elif FROZEN and (install_dir() / "portable.txt").is_file():      # portable zip: keep everything next to the app
        d = install_dir() / "data"
    elif sys.platform == "win32":
        d = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "OpenOne"
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / "OpenOne"
    else:
        d = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "OpenOne"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sub(name):
    p = data_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p
