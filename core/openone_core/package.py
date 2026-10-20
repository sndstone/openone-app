"""Model packages: read, verify against SHA256SUMS, and install from a zip or an extracted folder.

A package is one top folder holding model.safetensors, config.json, the tokenizer, openone.json, decider_config.json,
the decider/ prompt package and SHA256SUMS (one "sha256  relative/path" line per file). Installing streams every file
through sha256 while copying it, so a file is read once; any mismatch aborts and removes the partial copy.
"""
import hashlib
import json
import math
import os
import shutil
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED = ("config.json", "tokenizer.json", "model.safetensors", "openone.json", "SHA256SUMS")
DOC_SUFFIXES = (".md", ".txt")
CHUNK = 4 << 20
MAX_TOTAL_BYTES = 40 << 30
VENDORED = {                                     # decider/ files the app vendors (core/openone_core/vendor)
    "decider/prompt.py": "46f8158cd00c708408f6340340517c5a0c50fb0a339d0e46cd0a30f8801624cd",
    "decider/systemone.py": "884e4b0bae36f5e09dbd689b4390e94320cbbd77f7b8d675e50490b1d7c61d34",
}


class PackageError(ValueError):
    pass


class Cancelled(Exception):
    pass


def parse_sums(text):
    """SHA256SUMS text -> {relative posix path: lowercase hex}. Raises PackageError on a malformed line."""
    out = {}
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in parts[0]):
            raise PackageError(f"SHA256SUMS line {n} is not '<sha256>  <path>': {line[:80]!r}")
        rel = parts[1].strip().lstrip("*")
        _check_rel(rel, f"SHA256SUMS line {n}")
        out[rel] = parts[0].lower()
    if not out:
        raise PackageError("SHA256SUMS lists no files")
    return out


WIN_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def _check_rel(rel, where):
    p = PurePosixPath(rel)
    if not rel or p.is_absolute() or ".." in p.parts or ":" in rel or "\\" in rel:
        raise PackageError(f"{where}: unsafe path {rel!r}")
    for part in p.parts:
        if part != part.rstrip(" .") or part.split(".")[0].lower() in WIN_RESERVED or any(ord(c) < 32 or c in '<>"|?*' for c in part):
            raise PackageError(f"{where}: path {rel!r} is not portable (trailing dot/space, reserved or invalid name)")


def _check_collisions(rels):
    """Two paths that are the same file on a case-insensitive file system (Windows, default macOS)."""
    seen = {}
    for r in rels:
        k = r.casefold()
        if k in seen and seen[k] != r:
            raise PackageError(f"paths {seen[k]!r} and {r!r} collide on a case-insensitive file system")
        seen[k] = r


def read_package(path):
    """Installed or extracted package folder -> info dict (no hashing)."""
    d = Path(path)
    missing = [f for f in REQUIRED if not (d / f).is_file()]
    if missing:
        raise PackageError(f"{d.name}: missing {', '.join(missing)}")
    openone = json.loads((d / "openone.json").read_text(encoding="utf-8"))
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    dcfg_path = d / "decider_config.json"
    dcfg = json.loads(dcfg_path.read_text(encoding="utf-8")) if dcfg_path.is_file() else {}
    sums = parse_sums((d / "SHA256SUMS").read_text(encoding="utf-8"))
    quant = d / "openone_quant.json"
    fmt = "fp8" if quant.is_file() or "quantization_config" in config else "bf16"
    check_openone(openone)
    layout = dcfg.get("layout", "chat" if dcfg.get("chat_template") is True else "plain")
    return {
        "name": d.name,
        "path": str(d),
        "format": fmt,
        "local_runnable": layout == "plain",
        "local_mode": "native" if fmt == "bf16" else "fp8_dequantized_bf16",   # FP8 never runs natively in the local runtime
        "layout": layout,
        "architecture": (config.get("architectures") or ["?"])[0],
        "weights_sha256": sums.get("model.safetensors"),
        "weights_bytes": (d / "model.safetensors").stat().st_size,
        "openone": openone,
        "decider_version": dcfg.get("version"),
        "status": openone.get("status"),
        "prompt_code": prompt_code_check(d, sums),
    }


def check_openone(o):
    """The serving settings the app reads; raises PackageError naming the missing or bad key."""
    try:
        t = o["usecase_temperatures"]
        g = o["gate"]
        vals = {"usecase_temperatures.usecase_toolcall": t["usecase_toolcall"],
                "usecase_temperatures.usecase_bucket": t["usecase_bucket"],
                "gate.auto_run_if_p_allow_at_least": g["auto_run_if_p_allow_at_least"],
                "gate.alarm_if_p_block_at_least": g["alarm_if_p_block_at_least"]}
    except (KeyError, TypeError) as e:
        raise PackageError(f"openone.json lacks {e}") from None
    for k, v in vals.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not v > 0:
            raise PackageError(f"openone.json {k} must be a number > 0, got {v!r}")
    for k in ("gate.auto_run_if_p_allow_at_least", "gate.alarm_if_p_block_at_least"):
        if vals[k] > 1:
            raise PackageError(f"openone.json {k} must be <= 1, got {vals[k]!r}")


def prompt_code_check(d, sums):
    """Compare the package's decider prompt code with the app's vendored copy (the app always uses its own)."""
    out = {"matches_app": True, "differs": []}
    for rel, want in VENDORED.items():
        have = sums.get(rel)
        if have is None and (Path(d) / rel).is_file():
            have = sha256_file(Path(d) / rel)
        if have != want:
            out["matches_app"] = False
            out["differs"].append(rel)
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ sources
class _ZipSource:
    def __init__(self, path):
        try:
            self.z = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as e:
            raise PackageError(f"not a readable zip file: {e}") from None
        infos = [i for i in self.z.infolist() if not i.is_dir()]
        if not infos:
            raise PackageError("the zip is empty")
        tops = set()
        for i in infos:
            _check_rel(i.filename, "zip member")
            if stat.S_ISLNK(i.external_attr >> 16):
                raise PackageError(f"zip member {i.filename!r} is a symlink")
            tops.add(PurePosixPath(i.filename).parts[0])
        if len(tops) != 1 or len(PurePosixPath(infos[0].filename).parts) < 2:
            raise PackageError("the zip must hold exactly one top folder (the package)")
        self.top = tops.pop()
        self.files = {str(PurePosixPath(*PurePosixPath(i.filename).parts[1:])): i for i in infos}

    def names(self):
        return list(self.files)

    def size(self, rel):
        return self.files[rel].file_size

    def open(self, rel):
        return self.z.open(self.files[rel])

    def close(self):
        self.z.close()


class _DirSource:
    def __init__(self, path):
        self.root = Path(path)
        if not (self.root / "SHA256SUMS").is_file():
            subs = [p for p in self.root.iterdir() if p.is_dir() and (p / "SHA256SUMS").is_file()]
            if len(subs) == 1:
                self.root = subs[0]
            else:
                raise PackageError("the folder holds no SHA256SUMS (pick the package folder itself)")
        self.top = self.root.name
        self.files = {}
        for p in self.root.rglob("*"):
            if p.is_symlink():
                raise PackageError(f"{p.relative_to(self.root)} is a symlink")
            if p.is_file():
                self.files[p.relative_to(self.root).as_posix()] = p

    def names(self):
        return list(self.files)

    def size(self, rel):
        return self.files[rel].stat().st_size

    def open(self, rel):
        return open(self.files[rel], "rb")

    def close(self):
        pass


def open_source(path):
    p = Path(path)
    if p.is_dir():
        return _DirSource(p)
    if p.is_file():
        return _ZipSource(p)
    raise PackageError(f"{path} does not exist")


def inspect(path):
    """Cheap look at a zip/folder before installing: name, file count, bytes, what SHA256SUMS covers."""
    src = open_source(path)
    try:
        if "SHA256SUMS" not in src.files:
            raise PackageError("the package has no SHA256SUMS")
        with src.open("SHA256SUMS") as f:
            sums = parse_sums(f.read().decode("utf-8"))
        missing = [r for r in sums if r not in src.files]
        unlisted = sorted(r for r in src.names() if r not in sums and r != "SHA256SUMS")
        req = [r for r in REQUIRED if r not in src.files]
        return {"name": src.top, "files": len(sums), "bytes": sum(src.size(r) for r in sums if r in src.files),
                "missing": missing, "unlisted": unlisted, "missing_required": req}
    finally:
        src.close()


def install(path, models_dir, progress=None, cancel=None, replace=False):
    """Verify every file of a zip or folder against SHA256SUMS while copying it into models_dir/<package name>.

    progress(done_bytes, total_bytes, rel_path, files_done, files_total); cancel() -> True aborts.
    Returns {"info": read_package(...), "verified": n, "skipped": [unlisted files not installed], "seconds": s}.
    """
    t0 = time.time()
    src = open_source(path)
    models_dir = Path(models_dir)
    try:
        if "SHA256SUMS" not in src.files:
            raise PackageError("the package has no SHA256SUMS")
        with src.open("SHA256SUMS") as f:
            sums_bytes = f.read()
        sums = parse_sums(sums_bytes.decode("utf-8"))
        missing = [r for r in sums if r not in src.files]
        if missing:
            raise PackageError(f"files listed in SHA256SUMS are missing: {', '.join(missing[:5])}")
        req = [r for r in REQUIRED if r not in src.files]
        if req:
            raise PackageError(f"not an OpenOne package, missing {', '.join(req)}")
        unlisted = sorted(r for r in src.names() if r not in sums and r != "SHA256SUMS")
        docs = [r for r in unlisted if r.lower().endswith(DOC_SUFFIXES) and "/" not in r]
        skipped = [r for r in unlisted if r not in docs]
        _check_collisions(list(sums) + docs + ["SHA256SUMS"])
        total = sum(src.size(r) for r in sums)
        if total > MAX_TOTAL_BYTES:
            raise PackageError(f"package is {total / 2**30:.1f} GiB, over the {MAX_TOTAL_BYTES / 2**30:.0f} GiB limit")
        models_dir.mkdir(parents=True, exist_ok=True)
        final = models_dir / src.top
        if final.exists() and not replace:
            raise PackageError(f"{src.top} is already installed (remove it first or import with replace)")
        free = shutil.disk_usage(models_dir).free
        if free < total + (1 << 28):
            raise PackageError(f"not enough disk space: need {total / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB")
        part = models_dir / (src.top + ".partial")
        if part.exists():
            shutil.rmtree(part)
        part.mkdir()
        done, nfiles = 0, 0
        try:
            for rel in sorted(sums, key=lambda r: src.size(r)):          # small files first: fast failure on a bad package
                dest = part / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with src.open(rel) as fi, open(dest, "wb") as fo:
                    while True:
                        if cancel and cancel():
                            raise Cancelled()
                        b = fi.read(CHUNK)
                        if not b:
                            break
                        h.update(b)
                        fo.write(b)
                        done += len(b)
                        if progress:
                            progress(done, total, rel, nfiles, len(sums))
                if h.hexdigest() != sums[rel]:
                    raise PackageError(f"checksum mismatch for {rel}: expected {sums[rel][:12]}…, got {h.hexdigest()[:12]}…")
                nfiles += 1
                if progress:
                    progress(done, total, rel, nfiles, len(sums))
            (part / "SHA256SUMS").write_bytes(sums_bytes)
            for r in docs:
                with src.open(r) as fi, open(part / r, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            info = read_package(part)
        except BaseException:
            shutil.rmtree(part, ignore_errors=True)
            raise
        if final.exists():
            shutil.rmtree(final)
        os.replace(part, final)
        info = read_package(final)
        return {"info": info, "verified": nfiles, "skipped": skipped, "unverified_docs": docs, "seconds": round(time.time() - t0, 1)}
    finally:
        src.close()


def verify_installed(path, progress=None, cancel=None):
    """Re-hash an installed package. -> {"ok": bool, "bad": [paths], "missing": [paths]}."""
    d = Path(path)
    sums = parse_sums((d / "SHA256SUMS").read_text(encoding="utf-8"))
    total = sum((d / r).stat().st_size for r in sums if (d / r).is_file())
    done, bad, missing = 0, [], []
    for i, rel in enumerate(sorted(sums)):
        p = d / rel
        if not p.is_file():
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(CHUNK), b""):
                if cancel and cancel():
                    raise Cancelled()
                h.update(b)
                done += len(b)
                if progress:
                    progress(done, total, rel, i, len(sums))
        if h.hexdigest() != sums[rel]:
            bad.append(rel)
    return {"ok": not bad and not missing, "bad": bad, "missing": missing, "files": len(sums)}
