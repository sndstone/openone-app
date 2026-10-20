"""Background jobs with progress (model import, runtime install, backend start, batch scoring)."""
import itertools
import threading
import time
import traceback

_ids = itertools.count(1)


class Job:
    def __init__(self, kind, fn, on_log=None):
        self.id = f"job{next(_ids)}"
        self.kind = kind
        self.state = "running"
        self.progress = {"stage": "starting", "done": 0, "total": 0, "message": ""}
        self.result = None
        self.error = None
        self.logs = []
        self.started = time.time()
        self.finished = None
        self._cancel = threading.Event()
        self._on_log = on_log
        self.thread = threading.Thread(target=self._run, args=(fn,), daemon=True)

    def _run(self, fn):
        try:
            self.result = fn(self)
            self.state = "done"
        except InterruptedError:
            self.state = "cancelled"
        except Exception as e:                                # noqa: BLE001 - reported to the UI
            self.error = f"{e}" if str(e) else type(e).__name__
            self.state = "error"
            self.log(traceback.format_exc())
        finally:
            self.finished = time.time()

    def update(self, **kw):
        self.progress = {**self.progress, **kw}

    def log(self, line):
        self.logs.append(line)
        if len(self.logs) > 2000:
            del self.logs[:500]
        if self._on_log:
            self._on_log(f"[{self.kind}] {line}")

    def cancel(self):
        self._cancel.set()

    def cancelled(self):
        return self._cancel.is_set()

    def view(self, with_logs=False):
        out = {"id": self.id, "kind": self.kind, "state": self.state, "progress": self.progress, "error": self.error,
               "result": self.result if self.state == "done" else None,
               "elapsed": round((self.finished or time.time()) - self.started, 1)}
        if with_logs:
            out["logs"] = self.logs[-200:]
        return out


class Jobs:
    def __init__(self, on_log=None):
        self.jobs = {}
        self.on_log = on_log

    def start(self, kind, fn):
        j = Job(kind, fn, self.on_log)
        self.jobs[j.id] = j                    # registered before it runs: running() sees it at once
        j.thread.start()
        return j

    def start_exclusive(self, kind, fn, lock):
        """Start unless a job of `kind` is running; check and registration happen under `lock` (the controller's
        lock, which backend starts also take)."""
        with lock:
            if self.running(kind):
                raise ValueError(f"another {kind} is already running")
            return self.start(kind, fn)

    def get(self, jid):
        return self.jobs.get(jid)

    def running(self, kind):
        return next((j for j in self.jobs.values() if j.kind == kind and j.state == "running"), None)
