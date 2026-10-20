"""Inference backends as seen from the app: a local worker process (runtime Python + torch) or a remote gate service.

Both expose score(specs) -> raw per-question dicts; the app builds every response itself (openone_core.pipeline).
"""
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import paths


class BackendError(RuntimeError):
    pass


class LocalWorker:
    kind = "local"

    def __init__(self, python_exe, log_path, on_event=None):
        self.python_exe = str(Path(python_exe).resolve())
        self.log_path = Path(log_path)
        self.on_event = on_event or (lambda e: None)
        self.proc = None
        self.ids = itertools.count(1)
        self.pending = {}
        self.lock = threading.Lock()
        self.call_lock = threading.Lock()
        self.info = None
        self.load_info = None

    def start(self, timeout=60):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self.log.write(f"\n==== worker start {time.strftime('%Y-%m-%d %H:%M:%S')} {self.python_exe}\n")
        self.log.flush()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "PYINSTALLER", "_MEI", "TCL", "TK"))}
        env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   PYTORCH_ENABLE_MPS_FALLBACK="1")              # an op MPS lacks runs on the CPU instead of failing
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen([self.python_exe, "-I", str(paths.worker_script()), "--core", str(paths.core_dir())],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env,
                                     creationflags=flags, text=True, encoding="utf-8", bufsize=1)
        self.ready = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()
        if not self.ready.wait(timeout):
            self.stop()
            raise BackendError(f"the inference runtime did not start within {timeout}s (see {self.log_path.name})")

    def _reader(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("event") == "ready":
                self.ready.set()
            elif "event" in msg:
                self.on_event(msg)
            else:
                with self.lock:
                    q = self.pending.pop(msg.get("id"), None)
                if q:
                    q.put(msg)
        with self.lock:                                      # process ended: fail everything still waiting
            for q in self.pending.values():
                q.put({"ok": False, "error": "the inference runtime exited (see the worker log)", "type": "exit"})
            self.pending.clear()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def call(self, op, timeout=600, **kw):
        if not self.alive():
            raise BackendError("the inference runtime is not running")
        rid = next(self.ids)
        q = queue.Queue()
        with self.lock:
            self.pending[rid] = q
        with self.call_lock:
            self.proc.stdin.write(json.dumps(dict(kw, id=rid, op=op)) + "\n")
            self.proc.stdin.flush()
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            raise BackendError(f"{op} timed out after {timeout}s") from None
        if not msg.get("ok"):
            raise BackendError(msg.get("error", "worker error"))
        return msg["result"]

    def get_info(self):
        self.info = self.call("info", timeout=120)
        return self.info

    def load(self, **kw):
        self.load_info = self.call("load", timeout=1800, **kw)
        return self.load_info

    def score(self, specs):
        return self.call("score", timeout=600, specs=specs)

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"id": 0, "op": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(5)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                self.proc.kill()
        self.proc = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """The gate service never redirects; following one could hand the bearer key to another host."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused (the API key is not forwarded)",
                                     headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect)


class RemoteBackend:
    kind = "remote"

    def __init__(self, url, api_key=None, timeout=60):
        if not url.lower().startswith(("http://", "https://")):
            raise BackendError("the server URL must start with http:// or https://")
        self.url = url.rstrip("/")
        self.api_key = api_key or None
        self.timeout = timeout
        self.health_info = None

    def _req(self, method, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode()
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url + path, data=data, headers=h, method=method)
        try:
            with _OPENER.open(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if e.code == 401:
                raise BackendError("the server rejected the API key (401)") from None
            raise BackendError(f"server answered {e.code}: {detail}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise BackendError(f"cannot reach {self.url}: {getattr(e, 'reason', e)}") from None

    def health(self):
        h = self._req("GET", "/health", timeout=10)
        if not h.get("ok") or "model" not in h:
            raise BackendError("the server is not ready (its /health is not ok)")
        self.health_info = h
        return h

    def score(self, specs):
        return self._req("POST", "/v1/score", {"specs": specs})["results"]

    def alive(self):
        return self.health_info is not None

    def stop(self):
        self.health_info = None
