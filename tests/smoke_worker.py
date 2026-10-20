"""Smoke test: app worker client -> runtime worker -> pipeline, on a real installed package.

python tests/smoke_worker.py RUNTIME_PYTHON MODEL_DIR DEVICE [native|fp8_dequantized_bf16]
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "core")]
from openone_app.backends import LocalWorker  # noqa: E402
from openone_core import package, pipeline, policy  # noqa: E402

py, model_dir, device = sys.argv[1:4]
mode = sys.argv[4] if len(sys.argv) > 4 else "native"
info = package.read_package(model_dir)
settings, warns = policy.resolve_settings(info, mode)
log = Path(tempfile.gettempdir()) / "openone-smoke-worker.log"
w = LocalWorker(py, log, on_event=lambda e: print("  event", e.get("stage"), e.get("done"), e.get("total"), flush=True)
                if e.get("done") in (0, e.get("total")) else None)
w.start()
print("info", json.dumps(w.get_info())[:300])
t = time.time()
cache = str(Path(tempfile.gettempdir()) / "openone-smoke-dequant")
li = w.load(model_dir=model_dir, mode=mode, device=device,
            dequant={"cache_dir": cache, "keep": False, "source_weights_sha256": info["weights_sha256"]})
print("load", json.dumps(li)[:600], f"{time.time() - t:.1f}s")
ctx = {"model": {"name": info["name"], "format": info["format"], "mode": mode, "weights_sha256": info["weights_sha256"]},
       "backend": {"kind": "local", "device": li["device"], "runtime": li["runtime"]}, "settings": settings, "warnings": warns}
reqs = [
    {"state": {"agent": "claude-code", "os": "windows", "shell": "powershell", "workspace_root": "C:\\work\\demo-app",
               "cwd": "C:\\work\\demo-app", "tool": "PowerShell", "input": {"command": "Remove-Item .\\dist -Recurse -Force"},
               "user_task": "Clean the build output and rebuild"},
     "questions": {"verdict": {"type": "choice", "instructions": policy.GATE_INSTRUCTIONS, "criteria": policy.DEFAULT_POLICY}}},
    {"state": {"text": "The app freezes when I open settings"},
     "questions": {"bucket": {"type": "choice", "instructions": policy.BUCKET_INSTRUCTIONS,
                              "criteria": {"billing": "Payments, invoices, refunds.", "technical": {"what": "Crashes, errors, bugs", "examples": ["login fails"]},
                                           "other": None}}}},
    {"state": {"agent": "codex", "os": "linux", "shell": "bash", "workspace_root": "/srv/demo", "cwd": "/srv/demo", "tool": "Write",
               "input": {"file_path": "/srv/demo/notes.txt", "content": "x" * 30000}, "user_task": "write notes"},
     "questions": {"verdict": {"instructions": policy.GATE_INSTRUCTIONS, "criteria": policy.DEFAULT_POLICY}}},
    {"state": "", "questions": {"q": {"criteria": ["a"]}}},
]
for r in reqs:
    out = pipeline.decide(r, w.score, ctx, want_raw=True)
    rv, resp = out["request_validation"], out["response"]
    print("request valid", rv["valid"], rv["errors"][:3], rv["warnings"][:2])
    if resp:
        for qid, a in resp["answers"].items():
            print("  ", qid, a["choice"], a["probabilities"], a.get("verdict", {}).get("action"), a.get("accepted"),
                  "tokens", a["input_tokens"], "elided", a["elided"], "trunc", a["truncated"])
        print("  response valid", out["response_validation"], "latency", resp["latency_ms"], "warnings", resp["warnings"][:3])
        v = next(iter(out["raw"]["questions"].values()))
        print("   raw", {k: v[k] for k in ("letter_logprobs", "temperature", "rule", "forward_ms")}, "prompt tail:", repr(v["prompt_text"][-60:]))
w.stop()
