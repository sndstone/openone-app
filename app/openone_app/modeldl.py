"""Model packages from GitHub releases.

A model release is a release whose tag starts with "model-". It holds:
  - SHA256SUMS.txt: "<sha256>  <package>.zip" for every package (the hash of the whole zip);
  - the zip itself, or, when it is larger than GitHub's 2 GiB asset limit, parts "<package>.zip.001", ".002", …;
  - SHA256SUMS-parts.txt (optional): the hash of each part.

download() writes the parts straight into one zip at their offsets (no join step, no second copy), checks every part
against its hash, resumes after a cancel or a dropped connection (HTTP Range), and finally checks the whole zip against
SHA256SUMS.txt. The import that follows checks every file inside the package against the package's own SHA256SUMS."""
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from . import net

DEFAULT_REPO = "sndstone/openone-app"
REPO_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.zip$")
CHUNK = 1 << 20
SAVE_EVERY = 64 << 20
RETRIES = 3


class DownloadError(ValueError):
    pass


def _get(url, limit, headers=None):
    with net.open_url(url, headers, timeout=30) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise DownloadError(f"unexpectedly large response from {url}")
    return data


def parse_sums(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not _HEX64.match(parts[0].lower()):
            continue
        out[parts[1].lstrip("*").strip()] = parts[0].lower()
    return out


def catalog(repo=DEFAULT_REPO):
    """-> [{"repo", "tag", "release", "published", "file", "name", "format", "sha256", "size", "parts": [...]}]"""
    if not REPO_RE.match(repo or ""):
        raise DownloadError(f"not a GitHub repository name: {repo!r} (expected owner/name)")
    rels = []
    for page in range(1, 11):                           # up to 1,000 releases
        batch = json.loads(_get(f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}", 8 << 20,
                                {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}))
        rels += batch
        if len(batch) < 100:
            break
    out = []
    for rel in rels:
        tag = str(rel.get("tag_name") or "")
        if rel.get("draft") or not tag.startswith("model-"):
            continue
        assets = {a["name"]: a for a in rel.get("assets") or [] if a.get("state", "uploaded") == "uploaded"}
        if "SHA256SUMS.txt" not in assets:
            continue
        sums = parse_sums(_get(assets["SHA256SUMS.txt"]["browser_download_url"], 1 << 20).decode("utf-8", "replace"))
        psums = {}
        if "SHA256SUMS-parts.txt" in assets:
            psums = parse_sums(_get(assets["SHA256SUMS-parts.txt"]["browser_download_url"], 1 << 20).decode("utf-8", "replace"))
        for fname, sha in sums.items():
            if not _SAFE_FILE.match(fname):
                continue
            if fname in assets:
                parts = [assets[fname]]
            else:
                parts = sorted((a for n, a in assets.items() if re.fullmatch(re.escape(fname) + r"\.\d{3}", n)),
                               key=lambda a: a["name"])
                if not parts or [int(a["name"][-3:]) for a in parts] != list(range(1, len(parts) + 1)):
                    continue                            # missing or out-of-sequence parts: not offered
            out.append({"repo": repo, "tag": tag, "release": rel.get("name") or tag, "published": rel.get("published_at"),
                        "file": fname, "name": fname[:-4], "format": "fp8" if re.search(r"fp8", fname, re.I) else "bf16",
                        "sha256": sha, "size": sum(int(a["size"]) for a in parts),
                        "parts": [{"name": a["name"], "url": a["browser_download_url"], "size": int(a["size"]),
                                   "sha256": psums.get(a["name"])} for a in parts]})
    return out


def _state_path(dest):
    return dest.with_name(dest.name + ".download.json")


def _layout(entry):
    """What a resume state is valid for: the whole-zip hash plus the exact split (names, sizes, hashes)."""
    return {"sha256": entry["sha256"], "size": int(entry["size"]),
            "parts": [[p["name"], int(p["size"]), p.get("sha256")] for p in entry["parts"]]}


def partial_bytes(dest_dir, entry):
    """Bytes a new download of `entry` would not fetch again: an interrupted download, or the whole size when a
    complete zip is already there (downloaded but not installed). 0 if none."""
    dest = Path(dest_dir) / entry["file"]
    sp = _state_path(dest)
    try:
        st = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        try:
            return int(entry["size"]) if dest.stat().st_size == int(entry["size"]) else 0
        except OSError:
            return 0
    if st.get("layout") != _layout(entry):
        return 0
    return sum(int(v) for v in (st.get("have") or {}).values())


def _save(sp, st):
    tmp = sp.with_name(sp.name + ".tmp")
    tmp.write_text(json.dumps(st), encoding="utf-8")
    tmp.replace(sp)


def _hash_file(path, total, progress, stage, cancel, offset=0, length=None):
    h, done = hashlib.sha256(), 0
    length = total if length is None else length
    with open(path, "rb") as f:
        f.seek(offset)
        while done < length:
            if cancel():
                raise InterruptedError("cancelled")
            b = f.read(min(8 * CHUNK, length - done))
            if not b:
                break
            h.update(b)
            done += len(b)
            progress(stage=stage, done=done, total=length)
    return h, done


def download(entry, dest_dir, progress, cancel, log=lambda m: None, reserve=None):
    """Download `entry` (from catalog()) into dest_dir/<file>; -> Path of the verified zip.
    reserve: extra free bytes needed afterwards (the import copies the package into the models folder)."""
    if not _SAFE_FILE.match(entry["file"]):
        raise DownloadError(f"unsafe file name {entry['file']!r}")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest, total = dest_dir / entry["file"], int(entry["size"])
    sp = _state_path(dest)
    if not sp.exists() and dest.exists() and dest.stat().st_size == total:
        # a complete zip from an earlier run (its install failed or was stopped): reuse it if the hash still matches
        h, _ = _hash_file(dest, total, progress, "checking the downloaded package", cancel)
        if h.hexdigest() == entry["sha256"]:
            return dest
    try:
        st = json.loads(sp.read_text(encoding="utf-8"))
        if st.get("layout") != _layout(entry) or not dest.exists() or dest.stat().st_size != total:
            raise ValueError("stale")
        st["verified"] = []                             # re-checked from disk below; nothing is trusted blindly
    except (OSError, ValueError):
        st = {"layout": _layout(entry), "have": {}, "verified": []}
        dest.unlink(missing_ok=True)
    have_all = sum(int(v) for v in st["have"].values())
    need = (total - have_all) + (reserve if reserve is not None else total) + (256 << 20)
    free = shutil.disk_usage(dest_dir).free
    if free < need:
        raise DownloadError(f"not enough disk space: about {need / 1e9:.1f} GB needed (download + install), "
                            f"{free / 1e9:.1f} GB free in {dest_dir}")
    _save(sp, st)
    with open(dest, "r+b" if dest.exists() else "w+b") as f:
        f.truncate(total)
        offset = 0
        for i, part in enumerate(entry["parts"], 1):
            label = f"part {i}/{len(entry['parts'])}" if len(entry["parts"]) > 1 else entry["file"]
            if part["name"] in st["verified"]:
                offset += part["size"]
                continue
            for attempt in range(1, RETRIES + 1):
                try:
                    _fetch_part(f, dest, part, offset, st, sp, total, label, progress, cancel)
                    break
                except InterruptedError:                # a cancel (InterruptedError is an OSError): never retried
                    raise
                except (OSError, DownloadError) as e:
                    if isinstance(e, DownloadError) and "checksum" in str(e):
                        raise
                    if attempt == RETRIES or cancel():
                        raise DownloadError(f"download of {part['name']} failed: {e}") from None
                    log(f"download of {part['name']} interrupted ({e}); resuming (attempt {attempt + 1}/{RETRIES})")
                    time.sleep(2 * attempt)
            offset += part["size"]
    h, _ = _hash_file(dest, total, progress, "checking the whole package", cancel)
    if h.hexdigest() != entry["sha256"]:
        dest.unlink(missing_ok=True)
        sp.unlink(missing_ok=True)
        raise DownloadError(f"{entry['file']}: checksum mismatch after download ({h.hexdigest()[:12]}… vs "
                            f"{entry['sha256'][:12]}… in SHA256SUMS.txt); the file was deleted")
    sp.unlink(missing_ok=True)
    return dest


def _fetch_part(f, dest, part, offset, st, sp, total, label, progress, cancel):
    size, name = part["size"], part["name"]
    have = min(int(st["have"].get(name, 0)), size)
    h = hashlib.sha256()
    if have:                                            # resume: re-hash what is already on disk
        f.flush()
        h, got = _hash_file(dest, total, lambda **k: progress(stage=f"checking {label}", done=k["done"], total=k["total"]),
                            "checking", cancel, offset, have)
        if got != have:
            h, have = hashlib.sha256(), 0
    if have == size:                                    # complete on disk: verify it here, never ask for an empty range
        if not part.get("sha256") or h.hexdigest() == part["sha256"]:
            st["verified"].append(name)
            _save(sp, st)
            return
        h, have = hashlib.sha256(), 0                   # damaged: fetch this part again
        st["have"][name] = 0
        _save(sp, st)
    headers = {"Range": f"bytes={have}-"} if have else {}
    with net.open_url(part["url"], headers, timeout=60) as r:
        if have and getattr(r, "status", 200) != 206:   # the server ignored the range: start this part again
            h, have = hashlib.sha256(), 0
        f.seek(offset + have)
        since = 0
        try:
            while have < size:
                if cancel():
                    raise InterruptedError("cancelled")
                b = r.read(min(CHUNK, size - have))
                if not b:
                    break
                f.write(b)
                h.update(b)
                have += len(b)
                since += len(b)
                if since >= SAVE_EVERY:
                    f.flush()
                    st["have"][name] = have
                    _save(sp, st)
                    since = 0
                progress(stage=f"downloading {label}", done=offset + have, total=total)
        finally:                                        # cancel, dropped connection or done: the next try resumes here
            f.flush()
            st["have"][name] = have
            _save(sp, st)
    if have != size:
        raise DownloadError(f"{name}: connection closed after {have} of {size} bytes")
    if part.get("sha256") and h.hexdigest() != part["sha256"]:
        st["have"][name] = 0
        _save(sp, st)
        raise DownloadError(f"{name}: checksum mismatch ({h.hexdigest()[:12]}… vs {part['sha256'][:12]}…); "
                            "this part will be downloaded again next time")
    st["verified"].append(name)
    _save(sp, st)
