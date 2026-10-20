"""Settings, app log, decision history (opt-in) and the feedback log.

The remote API key is kept in its own file (owner-only where the OS allows) and never sent back to the UI.
"""
import json
import os
import threading
import time
import zipfile
from pathlib import Path

from . import paths
from openone_core import __version__ as CORE_VERSION

APP_VERSION = "0.1.0"
DEFAULTS = {
    "onboarded": False,
    "active_model": None,
    "backend": {"kind": "local", "runtime": None, "custom_python": None, "device": "auto", "dtype": "auto",
                "threads": None, "remote_url": "", "autostart": True},
    "dequant_cache": {},                 # model name -> true (keep the dequantized bf16 copy) / false (redo each load)
    "overrides": {},                     # advanced: {"temperature": {...}, "gate": {...}, "bucket": {...}}
    "policy": None,                      # advanced: custom allow/ask/block texts (None = defaults)
    "bucket_accept_at": 0.8,
    "save_history": False,
    "theme": "system",
}
_LOCK = threading.Lock()


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def _write_atomic(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Store:
    def __init__(self):
        self.dir = paths.data_dir()
        self.settings_path = self.dir / "settings.json"
        self.key_path = self.dir / "remote.key"
        self.logs = paths.sub("logs")
        self.feedback_path = self.dir / "feedback.jsonl"
        self.history_path = self.logs / "decisions.jsonl"
        try:
            raw = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        self.settings = _merge(DEFAULTS, raw)

    # ---- settings
    def save(self):
        with _LOCK:
            _write_atomic(self.settings_path, json.dumps(self.settings, indent=2))

    def update(self, patch):
        self.settings = _merge(self.settings, patch)
        self.save()
        return self.settings

    def api_key(self):
        try:
            return self.key_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def set_api_key(self, key):
        if key:
            _write_atomic(self.key_path, key.strip())
            try:
                os.chmod(self.key_path, 0o600)
            except OSError:
                pass
        elif self.key_path.exists():
            self.key_path.unlink()

    # ---- logs
    def log(self, msg, level="info"):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}\n"
        with _LOCK:
            with open(self.logs / "app.log", "a", encoding="utf-8") as f:
                f.write(line)

    def tail(self, name, lines=400):
        p = self.logs / name
        if not p.is_file():
            return ""
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            data = f.read().decode("utf-8", "replace")
        return "\n".join(data.splitlines()[-lines:])

    def history(self, record):
        if self.settings.get("save_history"):
            with _LOCK:
                with open(self.history_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- feedback
    def feedback(self, kind, request, response, validation, note=None, extra=None):
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, "app_version": APP_VERSION,
               "core_version": CORE_VERSION, "note": note, "request": request, "response": response,
               "validation": validation, **(extra or {})}
        with _LOCK:
            with open(self.feedback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    def feedback_items(self):
        if not self.feedback_path.is_file():
            return []
        out = []
        for line in self.feedback_path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def export_feedback(self, dest):
        """Zip feedback.jsonl + a manifest to dest. -> path."""
        dest = Path(dest)
        items = self.feedback_items()
        manifest = {"app_version": APP_VERSION, "core_version": CORE_VERSION, "items": len(items),
                    "kinds": {k: sum(1 for i in items if i["kind"] == k) for k in {i["kind"] for i in items}},
                    "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "note": "Contains the requests (states, commands) you scored. Review before sending."}
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
            z.writestr("feedback.jsonl", "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
        return str(dest)

    def clear_feedback(self):
        if self.feedback_path.exists():
            self.feedback_path.unlink()
