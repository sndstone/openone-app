"""Local HTTP server for the UI: static files + JSON API on 127.0.0.1 only.

Every /api call needs the per-session token (X-OpenOne-Token), and the Host header must be the loopback address, so
other web pages and DNS-rebinding tricks cannot drive the app.
"""
import json
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import paths
from .backends import BackendError
from .controller import Controller
from openone_core.package import PackageError
from openone_core.pipeline import parse_json_text

MAX_BODY = 64 << 20


class ApiError(Exception):
    def __init__(self, status, message, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


def _need(body, *keys):
    for k in keys:
        if k not in body:
            raise ApiError(400, f"missing field {k!r}")
    return [body[k] for k in keys]


def open_in_os(path):
    if sys.platform == "win32":
        os.startfile(path)                                   # noqa: S606 - a folder of our own data dir
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def build_routes(c: Controller, ui):
    def job_view(j):
        return j.view()

    def status(_b, _q):
        return {**c.status(), "native": ui.get("dialog") is not None}

    def hardware(b, _q):
        return c.detect_hardware(force=bool(b.get("force")))

    def runtime_install(b, _q):
        return job_view(c.install_runtime(_need(b, "variant")[0]))

    def runtime_remove(b, _q):
        from . import runtime
        runtime.remove_runtime(_need(b, "id")[0])
        return {"ok": True}

    def runtime_custom(b, _q):
        py = os.path.abspath(_need(b, "python")[0].strip())            # not resolve(): keep venv symlinks
        if not Path(py).is_file():
            raise ApiError(400, f"{py} does not exist")
        info = c.test_python(py)
        c.store.update({"backend": {"custom_python": py, "runtime": "custom"}})
        return info

    def models_inspect(b, _q):
        from openone_core import package
        return package.inspect(_need(b, "path")[0])

    def models_import(b, _q):
        if c.jobs.running("import"):
            raise ApiError(409, "an import is already running")
        return job_view(c.import_model(_need(b, "path")[0], bool(b.get("replace"))))

    def models_verify(b, _q):
        return job_view(c.verify_model(_need(b, "name")[0]))

    def models_remove(b, _q):
        c.remove_model(_need(b, "name")[0])
        return {"ok": True}

    def models_activate(b, _q):
        name = _need(b, "name")[0]
        if not c.model(name):
            raise ApiError(404, f"no model {name!r}")
        c.stop_backend()
        c.store.update({"active_model": name})
        return {"ok": True}

    def models_dequant(b, _q):
        name, keep = _need(b, "name", "keep")
        if not c.model(name):
            raise ApiError(404, f"no model {name!r}")
        c.store.update({"dequant_cache": {name: bool(keep)}})
        if not keep:
            c.clear_dequant_cache(name)
        return {"ok": True}

    def backend_configure(b, _q):
        allowed = {"kind", "runtime", "device", "dtype", "threads", "remote_url", "autostart"}
        patch = {k: v for k, v in b.items() if k in allowed}
        if "kind" in patch and patch["kind"] not in ("local", "remote"):
            raise ApiError(400, "kind must be local or remote")
        if "api_key" in b:
            c.store.set_api_key(b["api_key"] or None)
        c.store.update({"backend": patch})
        return {"ok": True}

    def backend_start(_b, _q):
        return job_view(c.start_backend())

    def backend_stop(_b, _q):
        c.stop_backend()
        return {"ok": True}

    def jobs_get(b, q):
        j = c.jobs.get((q.get("id") or [b.get("id")])[0])
        if not j:
            raise ApiError(404, "no such job")
        return j.view(with_logs=bool(q.get("logs")))

    def jobs_cancel(b, _q):
        j = c.jobs.get(_need(b, "id")[0])
        if j:
            j.cancel()
        return {"ok": True}

    def decide(b, _q):
        if "text" in b:
            req, err = __import__("openone_core.pipeline", fromlist=["x"]).parse_json_text(b["text"])
            if err:
                return {"request_validation": {"valid": False, "errors": [err], "warnings": []}, "response": None,
                        "response_validation": None, "raw": None}
        else:
            req = _need(b, "request")[0]
        try:
            return c.decide(req, raw=bool(b.get("raw")), use_overrides=b.get("use_overrides", True))
        except BackendError as e:
            raise ApiError(503, str(e)) from None

    def validate(b, _q):
        return c.validate_text(_need(b, "text")[0], b.get("which", "request"))

    def settings_post(b, _q):
        allowed = {"overrides", "policy", "save_history", "bucket_accept_at", "onboarded", "theme"}
        patch = {k: v for k, v in b.items() if k in allowed}
        if "overrides" in patch:
            s, _ = c.base_settings()
            if s is not None:
                from openone_core import policy
                policy.merge_overrides(s, patch["overrides"])      # raises ValueError on a bad number
            c.store.settings["overrides"] = patch.pop("overrides") or {}
        if "policy" in patch:
            pol = patch.pop("policy")
            if pol is not None and (not isinstance(pol, dict) or set(pol) != {"allow", "ask", "block"}
                                    or not all(isinstance(v, str) and v.strip() for v in pol.values())):
                raise ApiError(400, "policy must have non-empty allow, ask and block texts")
            c.store.settings["policy"] = pol
        if "bucket_accept_at" in patch:
            v = float(patch["bucket_accept_at"])
            if not 0 <= v <= 1:
                raise ApiError(400, "bucket_accept_at must be between 0 and 1")
            patch["bucket_accept_at"] = v
        c.store.update(patch)
        return {"ok": True}

    def settings_effective(_b, _q):
        base, warns = c.base_settings()
        eff, warns2 = c.effective_settings()
        return {"base": base, "effective": eff, "warnings": warns2, "overrides": c.store.settings.get("overrides") or {}}

    def batch_start(b, _q):
        text = b.get("text")
        if text is None and b.get("path"):
            p = Path(b["path"])
            if not p.is_file():
                raise ApiError(400, f"{p} does not exist")
            text = p.read_text(encoding="utf-8-sig")
        if not text:
            raise ApiError(400, "no batch content")
        try:
            return job_view(c.start_batch(text, b.get("use_overrides", True)))
        except BackendError as e:
            raise ApiError(503, str(e)) from None

    def batch_export(b, _q):
        jid, fmt = _need(b, "id", "format")
        if fmt not in ("jsonl", "csv"):
            raise ApiError(400, "format must be jsonl or csv")
        content = c.export_batch(jid, fmt)
        if b.get("path"):
            Path(b["path"]).write_text(content, encoding="utf-8", newline="")
            return {"path": b["path"], "bytes": len(content.encode())}
        return {"content": content, "filename": f"openone-batch-{time.strftime('%Y%m%d-%H%M%S')}.{fmt}"}

    def logs_get(_b, q):
        name = (q.get("name") or ["app.log"])[0]
        if name not in ("app.log", "worker.log", "runtime.log"):
            raise ApiError(400, "unknown log")
        return {"name": name, "text": c.store.tail(name)}

    def feedback_get(_b, _q):
        items = c.store.feedback_items()
        return {"count": len(items), "items": [{"time": i["time"], "kind": i["kind"], "note": i.get("note"),
                                                 "errors": (i.get("validation") or {}).get("errors", [])[:5]} for i in items[-200:]]}

    def feedback_report(b, _q):
        c.store.feedback("user_report", b.get("request"), b.get("response"), b.get("validation"), note=b.get("note"))
        return {"ok": True}

    def feedback_export(b, _q):
        dest = b.get("path")
        if not dest:
            dl = Path.home() / "Downloads"
            dl = dl if dl.is_dir() else paths.data_dir()
            dest = dl / f"openone-feedback-{time.strftime('%Y%m%d-%H%M%S')}.zip"
        return {"path": c.store.export_feedback(dest)}

    def feedback_clear(_b, _q):
        c.store.clear_feedback()
        return {"ok": True}

    def open_folder(b, _q):
        what = b.get("what", "data")
        p = {"data": paths.data_dir(), "logs": c.store.logs}.get(what)
        if p is None:
            raise ApiError(400, "unknown folder")
        open_in_os(str(p))
        return {"ok": True}

    def dialog(b, _q):
        if ui.get("dialog") is None:
            raise ApiError(501, "no native dialogs in browser mode; type the path")
        return {"paths": ui["dialog"](b.get("kind", "open"), b.get("filter"), b.get("filename"))}

    return {
        ("GET", "/api/status"): status, ("POST", "/api/hardware"): hardware,
        ("POST", "/api/runtime/install"): runtime_install, ("POST", "/api/runtime/remove"): runtime_remove,
        ("POST", "/api/runtime/custom"): runtime_custom,
        ("POST", "/api/models/inspect"): models_inspect, ("POST", "/api/models/import"): models_import,
        ("POST", "/api/models/verify"): models_verify, ("POST", "/api/models/remove"): models_remove,
        ("POST", "/api/models/activate"): models_activate, ("POST", "/api/models/dequant"): models_dequant,
        ("POST", "/api/backend/configure"): backend_configure, ("POST", "/api/backend/start"): backend_start,
        ("POST", "/api/backend/stop"): backend_stop,
        ("GET", "/api/job"): jobs_get, ("POST", "/api/job/cancel"): jobs_cancel,
        ("POST", "/api/decide"): decide, ("POST", "/api/validate"): validate,
        ("GET", "/api/presets"): lambda b, q: c.presets(), ("GET", "/api/schemas"): lambda b, q: c.schemas(),
        ("POST", "/api/settings"): settings_post, ("GET", "/api/settings/effective"): settings_effective,
        ("POST", "/api/batch"): batch_start, ("POST", "/api/batch/export"): batch_export,
        ("GET", "/api/logs"): logs_get, ("GET", "/api/feedback"): feedback_get,
        ("POST", "/api/feedback/report"): feedback_report, ("POST", "/api/feedback/export"): feedback_export,
        ("POST", "/api/feedback/clear"): feedback_clear, ("POST", "/api/open"): open_folder, ("POST", "/api/dialog"): dialog,
    }


def make_server(controller, port=0, token=None, ui=None):
    token = token or secrets.token_urlsafe(24)
    ui = ui if ui is not None else {}
    routes = build_routes(controller, ui)
    web = paths.web_dir().resolve()

    class H(BaseHTTPRequestHandler):
        server_version = "OpenOne"

        def log_message(self, fmt, *args):
            pass

        def _send(self, status, body, ctype="application/json; charset=utf-8", extra=None):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _host_ok(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in ("127.0.0.1", "localhost")

        def _handle(self, method):
            if not self._host_ok():
                return self._send(403, {"error": "bad host"})
            u = urlparse(self.path)
            if not u.path.startswith("/api/"):
                if method != "GET":
                    return self._send(405, {"error": "method not allowed"})
                rel = u.path.lstrip("/") or "index.html"
                f = (web / rel).resolve()
                if web not in f.parents or not f.is_file():
                    f = web / "index.html"
                ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                return self._send(200, f.read_bytes(), ctype)
            if not secrets.compare_digest(self.headers.get("X-OpenOne-Token", ""), token):
                return self._send(401, {"error": "missing or wrong session token"})
            fn = routes.get((method, u.path))
            if fn is None:
                return self._send(404, {"error": f"no route {method} {u.path}"})
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._send(413, {"error": "request too large"})
                try:
                    raw = (self.rfile.read(n) or b"{}").decode("utf-8")
                except UnicodeDecodeError:
                    return self._send(400, {"error": "body is not UTF-8"})
                body, err = parse_json_text(raw)                 # strict: no duplicate keys, NaN or out-of-range numbers
                if err:
                    return self._send(400, {"error": f"body is not valid JSON: {err['message']}"})
                if not isinstance(body, dict):
                    return self._send(400, {"error": "body must be a JSON object"})
            try:
                return self._send(200, fn(body, parse_qs(u.query)))
            except ApiError as e:
                return self._send(e.status, {"error": e.message, **e.extra})
            except (ValueError, PackageError, BackendError, OSError) as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:                            # noqa: BLE001
                controller.store.log(traceback.format_exc(), "error")
                return self._send(500, {"error": f"internal error: {type(e).__name__}: {e}"})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    return srv, token


def serve_in_thread(controller, port=0, ui=None):
    srv, token = make_server(controller, port, ui=ui)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, token
