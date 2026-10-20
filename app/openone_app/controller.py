"""App state and actions behind the HTTP API: models, runtimes, backend lifecycle, decisions, batch, feedback."""
import csv
import io
import json
import shutil
import statistics
import threading
import time
from pathlib import Path

from openone_core import package, pipeline, policy
from openone_core.schemas import REQUEST_SCHEMA, RESPONSE_SCHEMA

from . import paths, presets, runtime
from .backends import BackendError, LocalWorker, RemoteBackend
from .jobs import Jobs
from .store import APP_VERSION, Store

MAX_BATCH_LINES = 50000


class Controller:
    def __init__(self):
        self.store = Store()
        self.jobs = Jobs(on_log=lambda m: self.store.log(m))
        self.backend = None
        self.backend_state = {"state": "stopped"}
        self._gen = 0                          # backend generation: a newer start or a stop supersedes an in-flight start
        self._lock = threading.RLock()         # guards _gen, backend, backend_state, _starting
        self._starting = {}                    # gen -> LocalWorker still loading (killed by stop/supersede)
        self.hardware = None
        self.batches = {}
        self._cleanup_temp()
        self.store.log(f"app {APP_VERSION} start, data {paths.data_dir()}")

    # ------------------------------------------------------------------------------------------------ models
    def models_dir(self):
        return paths.sub("models")

    def models(self):
        out = []
        for d in sorted(self.models_dir().iterdir()):
            if d.is_dir() and not d.name.endswith(".partial"):
                try:
                    info = package.read_package(d)
                    info["dequant_cached"] = bool(self._dequant_marker(info))
                    out.append(info)
                except (package.PackageError, ValueError, OSError) as e:
                    out.append({"name": d.name, "path": str(d), "broken": str(e)})
        return out

    def model(self, name):
        for m in self.models():
            if m["name"] == name and "broken" not in m:
                return m
        return None

    @staticmethod
    def _inside(base, name):
        """base/name, refusing names that leave base (.., absolute paths, separators)."""
        base = Path(base).resolve()
        p = (base / str(name)).resolve()
        if not str(name) or str(name) in (".", "..") or p.parent != base:
            raise ValueError(f"invalid name {name!r}")
        return p

    def _dequant_dir(self, name):
        return self._inside(paths.sub("cache") / "dequant", name)

    def _dequant_marker(self, info):
        if info.get("format") != "fp8":
            return None
        from openone_core import dequant
        return dequant.read_marker(self._dequant_dir(info["name"]), info["weights_sha256"])

    def _cleanup_temp(self):
        base = paths.sub("cache") / "dequant"
        base.mkdir(parents=True, exist_ok=True)
        for d in base.iterdir():
            if ".tmp-" in d.name or ".partial" in d.name:
                shutil.rmtree(d, ignore_errors=True)

    def import_model(self, path, replace=False, activate=True):
        if replace:
            # a starting backend may be about to load this package: stop it as well (starts are refused until the
            # import finishes, see start_backend)
            name = package.inspect(path)["name"]
            with self._lock:
                st = self.backend_state
                hit = (st.get("state") == "starting" or (st.get("model") or {}).get("name") == name
                       or self.store.settings.get("active_model") == name)
            if hit:
                self.stop_backend()

        def run(job):
            job.update(stage="verifying", message=Path(path).name)
            res = package.install(path, self.models_dir(), replace=replace, cancel=job.cancelled,
                                  progress=lambda d, t, r, i, n: job.update(stage="verifying", done=d, total=t,
                                                                            message=f"{r} ({i}/{n} files)"))
            info = res["info"]
            self.store.log(f"imported {info['name']} ({info['format']}), {res['verified']} files verified in {res['seconds']}s")
            if activate:
                self.store.update({"active_model": info["name"]})
            return {k: res[k] for k in ("verified", "skipped", "unverified_docs", "seconds")} | {"name": info["name"],
                                                                                                    "format": info["format"]}
        return self.jobs.start("import", run)

    def verify_model(self, name):
        m = self.model(name)
        if not m:
            raise ValueError(f"no model {name!r}")

        def run(job):
            return package.verify_installed(m["path"], cancel=job.cancelled,
                                            progress=lambda d, t, r, i, n: job.update(stage="verifying", done=d, total=t, message=r))
        return self.jobs.start("verify", run)

    def remove_model(self, name):
        d = self._inside(self.models_dir(), name)
        if not d.is_dir():
            raise ValueError(f"no model {name!r}")
        if self.store.settings["active_model"] == name:
            self.stop_backend()
            self.store.update({"active_model": None})
        shutil.rmtree(d)
        shutil.rmtree(self._dequant_dir(name), ignore_errors=True)
        self.store.log(f"removed model {name}")

    def clear_dequant_cache(self, name):
        if not self.model(name):
            raise ValueError(f"no model {name!r}")
        shutil.rmtree(self._dequant_dir(name), ignore_errors=True)

    # ------------------------------------------------------------------------------------------------ runtimes
    def detect_hardware(self, force=False):
        if self.hardware is None or force:
            self.hardware = runtime.detect_hardware()
        return self.hardware

    def runtimes(self):
        return runtime.list_runtimes(self.store.settings["backend"].get("custom_python"))

    def install_runtime(self, variant):
        if self.jobs.running("runtime"):
            raise ValueError("a runtime install is already running")
        rlog = self.store.logs / "runtime.log"

        def run(job):
            def log(line):
                with open(rlog, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            res = runtime.install_runtime(variant, lambda **kw: job.update(**kw), job.cancelled, log)
            job.update(stage="checking runtime", done=0, total=1)
            w = LocalWorker(res["python"], self.store.logs / "worker.log")
            w.start(timeout=120)
            try:
                info = w.get_info()
            finally:
                w.stop()
            self.store.update({"backend": {"runtime": res["id"]}})
            return {"id": res["id"], "info": info}
        return self.jobs.start("runtime", run)

    def test_python(self, python):
        w = LocalWorker(python, self.store.logs / "worker.log")
        w.start(timeout=120)
        try:
            return w.get_info()
        finally:
            w.stop()

    # ------------------------------------------------------------------------------------------------ backend
    def _detach(self, state):
        """Allocate a new generation, detach the current backend and any loading workers, publish `state`; all in one
        locked step. The detached processes are stopped outside the lock. -> the new generation."""
        with self._lock:
            self._gen += 1
            gen = self._gen
            old, self.backend = self.backend, None
            loading = list(self._starting.values())
            self._starting.clear()
            self.backend_state = state
        if old:
            old.stop()
        for w in loading:                          # busy loading: it cannot read a shutdown request, so kill it
            w.kill()
        return gen

    def stop_backend(self):
        self._detach({"state": "stopped"})

    def start_backend(self):
        if self.jobs.running("import"):
            raise ValueError("a model import is running; start the model after it finishes")
        b = dict(self.store.settings["backend"])
        gen = self._detach({"state": "starting", "kind": b["kind"]})

        def run(job):
            try:
                return self._start(job, b, gen)
            except InterruptedError:
                raise
            except Exception as e:
                with self._lock:
                    if gen != self._gen:
                        raise InterruptedError("superseded by a newer start") from None
                    self.backend_state = {"state": "error", "kind": b["kind"], "error": str(e)}
                    self._starting.pop(gen, None)
                self.store.log(f"backend start failed: {e}", "error")
                raise
        return self.jobs.start("backend", run)

    def _loading(self, gen, worker):
        """Register a worker that is loading, so a stop or a newer start can kill it."""
        with self._lock:
            current = gen == self._gen
            if current:
                self._starting[gen] = worker
        if not current:
            worker.kill()
            raise InterruptedError("superseded by a newer start")

    def _install(self, gen, backend, state):
        """Make a started backend current, unless a newer start or a stop came in meanwhile (atomic with stop)."""
        with self._lock:
            current = gen == self._gen
            if current:
                self._starting.pop(gen, None)
                self.backend = backend
                self.backend_state = state
        if not current:
            backend.stop()
            raise InterruptedError("superseded by a newer start")

    def _start(self, job, b, gen):
        if b["kind"] == "remote":
            if not b.get("remote_url"):
                raise ValueError("no server URL configured")
            job.update(stage="connecting", message=b["remote_url"])
            rb = RemoteBackend(b["remote_url"], self.store.api_key())
            h = rb.health()
            m = h["model"]
            warns = list(h.get("warnings") or [])
            local = self.model(self.store.settings.get("active_model") or "")
            if local and local.get("weights_sha256") != m.get("weights_sha256"):
                warns.append(f"the server runs {m['name']} ({(m.get('weights_sha256') or '')[:12]}…), not your active "
                             f"package {local['name']} ({(local.get('weights_sha256') or '')[:12]}…)")
            self._install(gen, rb, {"state": "ready", "kind": "remote", "url": rb.url, "device": f"cuda ({h['gpu']['name']})",
                                    "runtime": h["runtime"], "mode": m["mode"],
                                    "model": {k: m[k] for k in ("name", "format", "mode", "weights_sha256")},
                                    "server_settings": h["settings"], "base_settings": h["settings"],
                                    "warnings": warns, "auth": h.get("auth")})
            self.store.log(f"connected to {rb.url}: {m['name']} {m['mode']}")
            return self.backend_state
        name = self.store.settings.get("active_model")
        info = self.model(name) if name else None
        if not info:
            raise ValueError("no model imported yet (import a package first)")
        if not info["local_runnable"]:
            raise ValueError(f"{name} uses the {info['layout']} prompt layout, which this app does not run")
        rts = {r["id"]: r for r in self.runtimes()}
        rid = b.get("runtime") or (next(iter(rts)) if rts else None)
        if rid not in rts:
            raise ValueError("no inference runtime installed (install one in onboarding or settings)")
        mode = info["local_mode"]
        keep = self.store.settings["dequant_cache"].get(name, True)
        job.update(stage="starting runtime", message=rts[rid]["python"])

        def ev(e):
            job.update(stage=e.get("stage", ""), done=e.get("done", 0), total=e.get("total", 0), message=e.get("message", ""))
        w = LocalWorker(rts[rid]["python"], self.store.logs / "worker.log", on_event=ev)
        w.start(timeout=180)
        self._loading(gen, w)
        try:
            winfo = w.get_info()
            li = w.load(model_dir=info["path"], mode=mode, device=b.get("device") or "auto", dtype=b.get("dtype") or "auto",
                        threads=b.get("threads"), dequant={"cache_dir": str(self._dequant_dir(name)), "keep": keep,
                                                           "source_weights_sha256": info["weights_sha256"]})
        except Exception:
            w.kill()
            raise
        s_used, warns = policy.resolve_settings(info, mode, self.models())
        if mode == "fp8_dequantized_bf16" and "fitted under vLLM" in s_used["source"]["gate"]:
            warns = warns + [policy.VLLM_FIT_NOTE]
        self._install(gen, w, {"state": "ready", "kind": "local", "runtime_id": rid, "device": li["device"], "dtype": li["dtype"],
                              "runtime": li["runtime"], "mode": mode, "load_seconds": li["load_seconds"],
                              "warmup_ms": li.get("warmup_ms"), "dequant": li.get("dequant"), "worker": winfo,
                              "model": {"name": info["name"], "format": info["format"], "mode": mode,
                                        "weights_sha256": info["weights_sha256"]}, "warnings": warns,
                               "base_settings": s_used})
        self.store.log(f"local backend ready: {name} {mode} on {li['device']} ({li['runtime']}) in {li['load_seconds']}s")
        return self.backend_state

    def backend_ready(self):
        if not self.backend or self.backend_state.get("state") != "ready":
            raise BackendError("the model is not running (start it in Settings or finish onboarding)")
        if self.backend.kind == "local" and not self.backend.alive():
            self.backend_state = {"state": "error", "kind": "local", "error": "the inference runtime exited (see worker.log)"}
            raise BackendError(self.backend_state["error"])

    # ------------------------------------------------------------------------------------------------ settings
    def base_settings(self):
        """Settings before advanced overrides (what "reset to defaults" restores) -> (settings, warnings)."""
        with self._lock:
            st = self.backend_state
        if st.get("state") == "ready":
            # the numbers captured when this backend started: re-reading the package from disk could pair a running
            # model with a replacement package's settings
            s = json.loads(json.dumps(st["base_settings"]))
            warns = list(st.get("warnings") or [])
        else:
            name = self.store.settings.get("active_model")
            info = self.model(name) if name else None
            if not info:
                return None, []
            s, warns = policy.resolve_settings(info, info["local_mode"], self.models())
            if info["local_mode"] == "fp8_dequantized_bf16" and "fitted under vLLM" in s["source"]["gate"]:
                warns = warns + [policy.VLLM_FIT_NOTE]
        s["bucket"] = {"accept_at": float(self.store.settings.get("bucket_accept_at", policy.BUCKET_ACCEPT_DEFAULT))}
        if s["bucket"]["accept_at"] != policy.BUCKET_ACCEPT_DEFAULT:
            s["source"]["bucket"] = "app setting"
        s.setdefault("provisional", False)
        return s, warns

    def effective_settings(self, use_overrides=True):
        s, warns = self.base_settings()
        if s is None:
            return None, warns
        if use_overrides and self.store.settings.get("overrides"):
            s = policy.merge_overrides(s, self.store.settings["overrides"])
            warns = warns + ["advanced overrides are active (temperatures/thresholds differ from the package)"]
        return s, warns

    # ------------------------------------------------------------------------------------------------ decisions
    def _snapshot(self, use_overrides=True):
        """(backend, context) taken together under the lock, so scores and settings always come from the same model."""
        with self._lock:
            return self.backend, self._ctx(use_overrides)

    def _ctx(self, use_overrides=True):
        self.backend_ready()
        st = self.backend_state
        s, warns = self.effective_settings(use_overrides)
        backend = {"kind": st["kind"], "device": st["device"], "runtime": st["runtime"]}
        if st["kind"] == "remote":
            backend["url"] = st["url"]
        return {"model": dict(st["model"]), "backend": backend, "settings": s, "warnings": warns}

    def decide(self, req, raw=False, use_overrides=True, source="app"):
        backend, ctx = self._snapshot(use_overrides)
        out = pipeline.decide(req, backend.score, ctx, want_raw=raw)
        resp = out["response"]
        if resp is not None and not out["response_validation"]["valid"]:
            self.store.feedback("invalid_response", req, resp, out["response_validation"], extra={"source": source})
            self.store.log(f"response failed validation: {out['response_validation']['errors'][:3]}", "error")
            out["feedback_logged"] = True
        if resp is not None:
            self.store.history({"request": req, "response": resp})
        return out

    def validate_text(self, text, which="request"):
        val, err = pipeline.parse_json_text(text)
        if err:
            return {"valid": False, "errors": [err], "warnings": []}
        if which == "response":
            return pipeline.validate_response(val) | {"warnings": []}
        return pipeline.validate_request(val)

    # ------------------------------------------------------------------------------------------------ batch
    @staticmethod
    def _batch_rows(text):
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            if len(rows) >= MAX_BATCH_LINES:
                raise ValueError(f"more than {MAX_BATCH_LINES} lines")
            val, err = pipeline.parse_json_text(line)
            if err:
                rows.append({"line": n, "error": f"line {n}: {err['message']} (column {err.get('column', '?')})"})
                continue
            if not isinstance(val, dict):
                rows.append({"line": n, "error": f"line {n}: not a JSON object"})
                continue
            rid, expected = val.get("id"), val.get("expected")
            if "question" in val and "questions" not in val:            # eval-row form {"id", "state", "question", "expected"}
                req = {"state": val.get("state"), "questions": {"q": val["question"]}}
                expected = {"q": expected} if expected is not None else None
            else:
                req = {k: v for k, v in val.items() if k not in ("id", "expected", "family", "group", "split_role")}
                if expected is not None and not isinstance(expected, dict):
                    qs = val.get("questions") or {}
                    expected = {next(iter(qs)): expected} if len(qs) == 1 else None
            rows.append({"line": n, "id": rid, "request": req, "expected": expected})
        if not rows:
            raise ValueError("the file holds no JSON lines")
        return rows

    def start_batch(self, text, use_overrides=True):
        rows = self._batch_rows(text)
        self.backend_ready()

        def run(job):
            results = []
            for k, r in enumerate(rows):
                if job.cancelled():
                    break
                job.update(stage="scoring", done=k, total=len(rows), message=f"line {r['line']}")
                res = {"line": r["line"], "id": r.get("id")}
                if "error" in r:
                    res.update(status="bad_json", errors=[r["error"]])
                else:
                    t = time.time()
                    try:
                        out = self.decide(r["request"], raw=False, use_overrides=use_overrides, source="batch")
                    except BackendError as e:
                        res.update(status="backend_error", errors=[str(e)])
                        results.append(res)
                        continue
                    res["latency_ms"] = round((time.time() - t) * 1000, 1)
                    if not out["request_validation"]["valid"]:
                        res.update(status="invalid_request", errors=[f"{e['path'] or '/'}: {e['message']}" for e in out["request_validation"]["errors"]])
                    else:
                        resp = out["response"]
                        res.update(status="ok" if out["response_validation"]["valid"] else "invalid_response",
                                   response=resp, response_valid=out["response_validation"]["valid"],
                                   errors=[f"{e['path']}: {e['message']}" for e in out["response_validation"]["errors"]])
                        if r.get("expected"):
                            res["expected"] = r["expected"]
                            res["correct"] = all(resp["answers"].get(q, {}).get("choice") == v for q, v in r["expected"].items())
                results.append(res)
            job.update(stage="done", done=len(results), total=len(rows))
            summary = self._batch_summary(results)
            self.batches[job.id] = results
            return summary
        return self.jobs.start("batch", run)

    @staticmethod
    def _batch_summary(results):
        lat = [r["latency_ms"] for r in results if "latency_ms" in r]
        st = {}
        for r in results:
            st[r["status"]] = st.get(r["status"], 0) + 1
        actions = {}
        for r in results:
            for a in (r.get("response") or {}).get("answers", {}).values():
                key = a["verdict"]["action"] if a["use_case"] == "toolcall" else ("accepted" if a["accepted"] else "below_threshold")
                actions[key] = actions.get(key, 0) + 1
        graded = [r for r in results if "correct" in r]
        q = sorted(lat)
        return {"rows": len(results), "status": st, "actions": actions,
                "accuracy": round(sum(r["correct"] for r in graded) / len(graded), 4) if graded else None, "graded": len(graded),
                "latency_ms": {"mean": round(statistics.mean(lat), 1), "p50": q[len(q) // 2], "p95": q[min(len(q) - 1, int(len(q) * 0.95))],
                               "max": q[-1]} if lat else None,
                "preview": [{k: r.get(k) for k in ("line", "id", "status", "latency_ms", "correct", "errors")} |
                            {"answers": {q2: {"choice": a["choice"], "p": a["x_p_max"],
                                              "action": a.get("verdict", {}).get("action") or ("accepted" if a.get("accepted") else "below")}
                                         for q2, a in (r.get("response") or {}).get("answers", {}).items()}}
                            for r in results[:500]]}

    def export_batch(self, jid, fmt):
        results = self.batches.get(jid)
        if results is None:
            raise ValueError("no finished batch with that id")
        if fmt == "jsonl":
            return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["line", "id", "status", "question", "use_case", "choice", "p_choice", "action", "probabilities",
                    "expected", "correct", "latency_ms", "response_valid", "errors"])
        for r in results:
            answers = (r.get("response") or {}).get("answers") or {None: None}
            for qid, a in answers.items():
                a = a or {}
                act = a.get("verdict", {}).get("action") if a.get("use_case") == "toolcall" else (
                    ("accepted" if a.get("accepted") else "below_threshold") if a else "")
                w.writerow([r["line"], r.get("id") or "", r["status"], qid or "", a.get("use_case", ""), a.get("choice", ""),
                            a.get("x_p_max", ""), act or "", json.dumps(a.get("probabilities")) if a else "",
                            (r.get("expected") or {}).get(qid, "") if qid else "", r.get("correct", ""), r.get("latency_ms", ""),
                            r.get("response_valid", ""), " | ".join(r.get("errors") or [])])
        return buf.getvalue()

    # ------------------------------------------------------------------------------------------------ status
    def status(self):
        s = self.store.settings
        b = dict(s["backend"])
        b["remote_key_set"] = self.store.api_key() is not None
        eff, warns = self.effective_settings() if self.store.settings.get("active_model") or \
            self.backend_state.get("state") == "ready" else (None, [])
        return {"app_version": APP_VERSION, "data_dir": str(paths.data_dir()), "onboarded": s["onboarded"],
                "active_model": s["active_model"], "backend_config": b, "backend": self.backend_state,
                "models": self.models(), "runtimes": self.runtimes(), "hardware": self.hardware,
                "settings": eff, "settings_warnings": warns, "overrides": s.get("overrides") or {},
                "policy": s.get("policy") or policy.DEFAULT_POLICY, "default_policy": policy.DEFAULT_POLICY,
                "gate_instructions": policy.GATE_INSTRUCTIONS, "bucket_instructions": policy.BUCKET_INSTRUCTIONS,
                "save_history": s.get("save_history"), "bucket_accept_at": s.get("bucket_accept_at"),
                "dequant_cache": s.get("dequant_cache") or {}, "feedback_count": len(self.store.feedback_items()),
                "jobs": {k: j.view() for k, j in self.jobs.jobs.items() if j.state == "running"},
                "fallback_warning": policy.FALLBACK_WARNING}

    def presets(self):
        return {"gate": presets.GATE, "bucket": presets.BUCKET}

    def schemas(self):
        return {"request": REQUEST_SCHEMA, "response": RESPONSE_SCHEMA}
