"""OpenOne inference worker: JSON lines on stdin/stdout, run by the app with the runtime's Python.

    python -I openone_worker.py --core DIR

Requests  {"id": n, "op": "info" | "load" | "score" | "shutdown", ...}
Replies   {"id": n, "ok": true, "result": ...} or {"id": n, "ok": false, "error": "...", "type": "..."}
Events    {"event": "progress", "stage": ..., "done": i, "total": n} and {"event": "log", "message": ...}
stdout carries only protocol lines; everything else (library warnings, progress bars) goes to stderr.
"""
import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback

PROTO = sys.stdout
sys.stdout = sys.stderr


def send(obj):
    PROTO.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
    PROTO.flush()


def event(**kw):
    send(dict(event=kw.pop("kind", "progress"), **kw))


STATE = {"engine": None, "info": None}


def op_info(_req):
    import torch
    import transformers
    from openone_core.scorers import devices
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "torch": torch.__version__, "transformers": transformers.__version__, "devices": devices(),
            "pid": os.getpid()}


def op_load(req):
    from openone_core.engine import Engine
    from openone_core.scorers import TorchScorer
    from openone_core import dequant
    model_dir = req["model_dir"]
    mode = req.get("mode", "native")
    info = {"mode": mode}
    load_dir, tmp = model_dir, None
    if mode == "fp8_dequantized_bf16":
        dq = req.get("dequant") or {}
        cache = dq.get("cache_dir")
        sha = dq.get("source_weights_sha256")
        keep = bool(dq.get("keep", True))
        marker = dequant.read_marker(cache, sha) if (cache and keep) else None
        if marker:
            info["dequant"] = dict(marker, cached=True)
            load_dir = cache
        else:
            target = cache if keep else f"{cache}.tmp-{os.getpid()}"
            event(stage="dequantize", done=0, total=1, message="dequantizing FP8 weights to bf16")
            marker = dequant.write_bf16_snapshot(model_dir, target, sha, progress=lambda i, n, k: (
                event(stage="dequantize", done=i, total=n, message=k) if i % 16 == 0 or i == n else None))
            info["dequant"] = dict(marker, cached=keep)
            load_dir = marker.get("path", target)
            tmp = None if keep else target
    elif mode != "native":
        raise ValueError(f"unknown mode {mode!r}")
    event(stage="load", done=0, total=1, message="loading weights")
    STATE["engine"] = None
    scorer = TorchScorer(load_dir, req.get("device", "auto"), req.get("dtype", "auto"), req.get("threads"))
    STATE["engine"] = Engine(scorer)
    if tmp:
        try:
            shutil.rmtree(tmp)
            info["dequant"]["temp_removed"] = True
        except OSError:
            info["dequant"]["temp_removed"] = False          # memory-mapped on this OS; the app removes it on the next start
    info.update(device=scorer.device, dtype=scorer.dtype_name, load_seconds=scorer.load_seconds, runtime=scorer.runtime)
    t = time.time()
    warm = STATE["engine"].score([{"state": {"text": "warm-up"}, "question": {
        "type": "choice", "instructions": "Pick one.", "criteria": {"a": None, "b": None}}}])
    info["warmup_ms"] = round((time.time() - t) * 1000, 1)
    assert len(warm) == 1
    event(stage="load", done=1, total=1, message="ready")
    STATE["info"] = info
    return info


def op_score(req):
    if STATE["engine"] is None:
        raise RuntimeError("no model loaded")
    return STATE["engine"].score(req["specs"])


OPS = {"info": op_info, "load": op_load, "score": op_score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.core)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    send({"event": "ready", "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            send({"id": None, "ok": False, "error": "bad JSON line", "type": "protocol"})
            continue
        rid, op = req.get("id"), req.get("op")
        if op == "shutdown":
            send({"id": rid, "ok": True, "result": None})
            break
        fn = OPS.get(op)
        if fn is None:
            send({"id": rid, "ok": False, "error": f"unknown op {op!r}", "type": "protocol"})
            continue
        try:
            send({"id": rid, "ok": True, "result": fn(req)})
        except Exception as e:                                   # noqa: BLE001 - every failure goes back to the app
            traceback.print_exc()
            send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "type": type(e).__name__})


if __name__ == "__main__":
    main()
