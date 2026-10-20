"""Inference runtimes (a standalone Python + torch + transformers) and hardware detection.

The UI process has no torch. Local inference runs in a runtime:
  - bundled with an offline installer (<install dir>/runtime/<variant>), or
  - installed at onboarding into <data>/runtimes/<variant>: python-build-standalone (pinned, sha256-checked) + pinned
    pip wheels for the chosen accelerator, or
  - any existing Python that already has torch + transformers ("custom").
"""
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from . import net, paths

PBS_TAG = "20261013"
PBS = {   # (platform key) -> (file, sha256)
    "win-x64": ("cpython-3.12.15+20261013-x86_64-pc-windows-msvc-install_only.tar.gz",
                "bbc902d4d187ac4914915bf538196a064b49109a33a104965b5164ced5b80a1e"),
    "mac-arm64": ("cpython-3.12.15+20261013-aarch64-apple-darwin-install_only.tar.gz",
                  "ba32036ad8f4c434d97a1ee7feef8b7edc0fac16777639b97e39c74a108e4440"),
    "mac-x64": ("cpython-3.12.15+20261013-x86_64-apple-darwin-install_only.tar.gz",
                "9686ddd8639b855905447ea4a16eb54a804a9ef12afbe7c9ecb1412d9c1fbc98"),
}
PBS_URL = "https://github.com/astral-sh/python-build-standalone/releases/download/{tag}/{file}"
TORCH = "2.14.1"
COMMON = ["transformers==5.20.2", "safetensors==0.8.0", "tokenizers==0.23.3", "huggingface_hub==2.2.0", "numpy==2.5.4"]
VARIANTS = {
    "cpu": {"label": "CPU", "index": "https://download.pytorch.org/whl/cpu", "download_gb": 0.3, "disk_gb": 1.0,
            "platforms": ["win-x64"]},
    "cuda": {"label": "NVIDIA CUDA 13.0 (driver 580+)", "index": "https://download.pytorch.org/whl/cu130",
             "download_gb": 2.8, "disk_gb": 5.0, "platforms": ["win-x64"]},
    "cuda126": {"label": "NVIDIA CUDA 12.6 (older drivers; no RTX 50-series)", "index": "https://download.pytorch.org/whl/cu126",
                "download_gb": 2.6, "disk_gb": 4.6, "platforms": ["win-x64"]},
    "xpu": {"label": "Intel Arc / Core Ultra GPU (XPU)", "index": "https://download.pytorch.org/whl/xpu",
            "download_gb": 1.6, "disk_gb": 4.3, "platforms": ["win-x64"]},
    "mps": {"label": "Apple Silicon (Metal / MPS)", "index": None, "download_gb": 0.2, "disk_gb": 0.9,
            "platforms": ["mac-arm64"]},
}
NOWIN = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def platform_key():
    m = platform.machine().lower()
    if sys.platform == "win32":
        return "win-x64" if m in ("amd64", "x86_64") else "win-arm64"
    if sys.platform == "darwin":
        return "mac-arm64" if m in ("arm64", "aarch64") else "mac-x64"
    return "linux-" + m


def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NOWIN)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_hardware():
    """No-torch hardware probe -> {"platform", "cpu", "ram_gb", "gpus": [{"vendor", "name", ...}], "recommend": {...}}."""
    pk = platform_key()
    hw = {"platform": pk, "os": f"{platform.system()} {platform.release()}", "cpu": platform.processor() or platform.machine(),
          "ram_gb": None, "gpus": []}
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            ms = MS()
            ms.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            hw["ram_gb"] = round(ms.total / 2**30, 1)
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                hw["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif sys.platform == "darwin":
            hw["ram_gb"] = round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1)
            hw["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or hw["cpu"]
    except Exception:                                      # noqa: BLE001 - best effort
        pass
    smi = shutil.which("nvidia-smi") or (r"C:\Windows\System32\nvidia-smi.exe" if sys.platform == "win32" else None)
    if smi and Path(smi).exists():
        out = _run([smi, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                cap = parts[3]
                try:
                    fp8 = tuple(int(x) for x in cap.split(".")) >= (8, 9)
                except ValueError:
                    fp8 = None
                hw["gpus"].append({"vendor": "nvidia", "name": parts[0], "memory_gb": round(float(parts[1]) / 1024, 1),
                                   "driver": parts[2], "compute_capability": cap, "fp8_capable": fp8})
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        for name in (n.strip() for n in out.splitlines()):
            low = name.lower()
            if not name or "nvidia" in low or "remote display" in low or "basic display" in low:
                continue
            vendor = "intel" if "intel" in low else "amd" if ("amd" in low or "radeon" in low) else "other"
            hw["gpus"].append({"vendor": vendor, "name": name,
                               "xpu": vendor == "intel" and bool(re.search(r"\barc\b|core\(tm\) ultra|graphics [0-9]", low))})
    elif pk == "mac-arm64":
        hw["gpus"].append({"vendor": "apple", "name": "Apple Silicon GPU", "mps": True})
    hw["recommend"] = recommend(hw)
    return hw


def recommend(hw):
    pk = hw["platform"]
    nv = [g for g in hw["gpus"] if g["vendor"] == "nvidia"]
    if pk == "mac-arm64":
        var, why = "mps", "Apple Silicon: the Metal (MPS) runtime runs the bf16 package on the GPU."
    elif pk == "mac-x64":
        var, why = None, ("Intel Mac: PyTorch no longer ships Intel-Mac builds that can run this model, so there is no "
                          "local runtime. Use a remote gate service, or point the app at your own Python with a working torch.")
    elif nv:
        drv = nv[0].get("driver") or "0"
        try:
            major = int(drv.split(".")[0])
        except ValueError:
            major = 0
        var = "cuda" if major >= 580 else "cuda126"
        why = f"NVIDIA {nv[0]['name']} (driver {drv}): CUDA runtime."
    elif any(g.get("xpu") for g in hw["gpus"]):
        g = next(g for g in hw["gpus"] if g.get("xpu"))
        var, why = "xpu", f"{g['name']}: Intel XPU runtime (about 0.12 s per decision on an Arc B580)."
    else:
        var, why = "cpu", "No supported GPU found: CPU runtime (about 0.8 s per decision on a fast 16-core CPU)."
    fp8_gpu = any(g.get("fp8_capable") for g in nv)
    pkg = ("bf16", "Use the bf16 package locally. The FP8 package needs vLLM on Linux; locally it only runs "
                   "dequantized to bf16 (fallback, no speed gain).")
    if fp8_gpu and sys.platform == "win32":
        pkg = ("bf16", "Use the bf16 package locally. Your NVIDIA GPU can run FP8 natively through the gate "
                       "service in Docker Desktop/WSL2 (remote backend at http://127.0.0.1:8130).")
    return {"runtime": var, "runtime_reason": why, "package": pkg[0], "package_reason": pkg[1],
            "available_runtimes": [v for v, d in VARIANTS.items() if pk in d["platforms"]]}


# ------------------------------------------------------------------------------------------------ runtimes
def python_in(root):
    root = Path(root)
    for c in (root / "python" / "python.exe", root / "python" / "bin" / "python3", root / "python.exe", root / "bin" / "python3"):
        if c.is_file():
            return c
    return None


def list_runtimes(custom=None):
    out = []
    for base, origin in ((paths.install_dir() / "runtime", "bundled"), (paths.data_dir() / "runtimes", "installed")):
        if base.is_dir():
            for d in sorted(base.iterdir()):
                py = python_in(d)
                if py and (d / "openone_runtime.json").is_file():
                    meta = json.loads((d / "openone_runtime.json").read_text(encoding="utf-8"))
                    out.append({"id": f"{origin}:{d.name}", "variant": meta.get("variant", d.name), "origin": origin,
                                "python": str(py), "meta": meta})
    if custom:
        out.append({"id": "custom", "variant": "custom", "origin": "custom", "python": custom, "meta": {}})
    return out


def _download(url, dest, sha256, progress, cancel):
    h = hashlib.sha256()
    with net.open_url(url, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if cancel():
                raise InterruptedError("cancelled")
            b = r.read(1 << 20)
            if not b:
                break
            h.update(b)
            f.write(b)
            done += len(b)
            progress(stage="download python", done=done, total=total)
    if h.hexdigest() != sha256:
        raise ValueError(f"python download checksum mismatch ({h.hexdigest()[:12]}…)")


def install_runtime(variant, progress, cancel, log, base=None, pk=None):
    """Download python-build-standalone + pip install the pinned packages into <data>/runtimes/<variant>
    (or <base>/<variant>; the offline build uses this to bundle a runtime next to the app)."""
    pk = pk or platform_key()
    if variant not in VARIANTS:
        raise ValueError(f"unknown runtime {variant!r}")
    if pk not in PBS:
        raise ValueError(f"no runtime download for platform {pk}; use a custom Python instead")
    spec = VARIANTS[variant]
    base = Path(base) if base else paths.sub("runtimes")
    base.mkdir(parents=True, exist_ok=True)
    final = base / variant
    part = base / (variant + ".partial")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir()
    try:
        fname, sha = PBS[pk]
        tgz = part / fname
        log(f"downloading {fname}")
        _download(PBS_URL.format(tag=PBS_TAG, file=fname), tgz, sha, progress, cancel)
        progress(stage="unpack python", done=0, total=1)
        with tarfile.open(tgz) as t:
            t.extractall(part, filter="data")
        tgz.unlink()
        py = python_in(part)
        if not py:
            raise RuntimeError("unpacked Python has no interpreter")
        steps = []
        torch_cmd = [str(py), "-m", "pip", "install", "--no-input", "--disable-pip-version-check", "--no-warn-script-location",
                     f"torch=={TORCH}"]
        if spec["index"]:
            torch_cmd += ["--index-url", spec["index"]]
        steps.append(("install torch", torch_cmd))
        steps.append(("install transformers", [str(py), "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
                                               "--no-warn-script-location"] + COMMON))
        for k, (stage, cmd) in enumerate(steps):
            progress(stage=stage, done=k, total=len(steps))
            log("$ " + " ".join(cmd))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=NOWIN,
                                 env={k2: v for k2, v in os.environ.items() if not k2.startswith(("PYTHON", "PIP_"))})
            for line in p.stdout:
                log(line.rstrip())
                m = re.search(r"(Downloading|Collecting|Installing collected packages)[: ]+(\S+)", line)
                if m:
                    progress(stage=stage, done=k, total=len(steps), message=f"{m.group(1)} {m.group(2)}"[:120])
                if cancel():
                    p.kill()
                    raise InterruptedError("cancelled")
            if p.wait() != 0:
                raise RuntimeError(f"{stage} failed (exit {p.returncode}); see the runtime log")
        meta = {"variant": variant, "label": spec["label"], "python_build": f"{fname}", "torch": TORCH,
                "packages": COMMON, "index": spec["index"]}
        (part / "openone_runtime.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    shutil.rmtree(final, ignore_errors=True)
    os.replace(part, final)
    return {"id": f"installed:{variant}", "python": str(python_in(final))}


def remove_runtime(rid):
    origin, _, name = rid.partition(":")
    if origin != "installed" or name not in VARIANTS:
        raise ValueError("only runtimes installed by the app can be removed here")
    base = (paths.data_dir() / "runtimes").resolve()
    d = (base / name).resolve()
    if d.parent != base:
        raise ValueError(f"invalid runtime {rid!r}")
    if d.is_dir():
        shutil.rmtree(d)
