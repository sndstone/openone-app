"""OpenOne remote gate service: vLLM-backed scoring for the desktop app (remote backend) and for agent harnesses.

    OPENONE_MODEL_DIR=/model uvicorn gate_server:app --host 0.0.0.0 --port 8130

Environment
  OPENONE_MODEL_DIR       installed/extracted package folder (verified against SHA256SUMS at start-up unless
                          OPENONE_SKIP_VERIFY=1)
  OPENONE_GATE_KEY_FILE   file holding the API key (or OPENONE_GATE_KEY). Required: start-up fails without a key unless
                          OPENONE_ALLOW_NO_AUTH=1
  OPENONE_MODE            auto | native | fp8_dequantized_bf16. auto: native, except an FP8 package on a GPU without FP8
                          (compute capability < 8.9) is dequantized to bf16 first
  OPENONE_DEQUANT_DIR     where the dequantized snapshot is cached (default /cache/dequant)
  OPENONE_SETTINGS_FILE   optional openone.json-style file overriding the serving numbers (e.g. the vLLM-fitted
                          values for a bf16 package); reported as the settings source
  OPENONE_VLLM_MEM        vLLM gpu_memory_utilization (default 0.45)

Routes
  GET  /health            model, mode, settings in force (and where they come from), warnings, runtime
  POST /v1/score          {"specs": [{"state", "question", "want_text"}]} -> {"results": [raw letter log-probs ...]}
                          (the desktop app applies temperature/thresholds itself, from the settings in /health)
  POST /v1/systemone      INTEGRATION.md request -> the full validated OpenOne response (422 with errors if invalid)
  POST /v1/decide         same, but returns {request_validation, response, response_validation, raw?}; ?raw=1 for raw mode
"""
import hmac
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from openone_core import __version__, dequant, package, pipeline, policy
from openone_core.engine import Engine
from openone_core.prompting import MAX_STATE_TOKENS

MODEL_DIR = os.environ.get("OPENONE_MODEL_DIR", "/model")
MAX_SPECS = 64
STATE = {"engine": None, "info": None, "mode": None, "settings": None, "warnings": [], "started": None, "error": None,
         "dequant": None, "gpu": None}
LOCK = threading.Lock()


def _key():
    """The API key. Fails closed: a configured key file that is missing or empty stops start-up, and running without a
    key needs OPENONE_ALLOW_NO_AUTH=1 (for a loopback-only dev setup)."""
    f = os.environ.get("OPENONE_GATE_KEY_FILE")
    if f:
        k = Path(f).read_text(encoding="utf-8").strip() if Path(f).is_file() else ""
        if not k:
            raise SystemExit(f"OPENONE_GATE_KEY_FILE={f} is missing or empty; refusing to start without auth")
        return k
    if os.environ.get("OPENONE_GATE_KEY", "").strip():
        return os.environ["OPENONE_GATE_KEY"].strip()
    if os.environ.get("OPENONE_ALLOW_NO_AUTH") == "1":
        return None
    raise SystemExit("no API key configured (OPENONE_GATE_KEY_FILE / OPENONE_GATE_KEY); set OPENONE_ALLOW_NO_AUTH=1 to run without")


KEY = _key()


def _gpu():
    import torch
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "capability": f"{p.major}.{p.minor}", "fp8_native": (p.major, p.minor) >= (8, 9),
            "memory_gb": round(p.total_memory / 2**30, 1)}


def load():
    from openone_core.scorers import VllmScorer
    t0 = time.time()
    if os.environ.get("OPENONE_SKIP_VERIFY") != "1":
        v = package.verify_installed(MODEL_DIR)
        if not v["ok"]:
            raise RuntimeError(f"package failed SHA256SUMS verification: bad {v['bad']}, missing {v['missing']}")
    info = package.read_package(MODEL_DIR)
    gpu = _gpu()
    want = os.environ.get("OPENONE_MODE", "auto")
    mode = "native"
    if info["format"] == "fp8" and (want == "fp8_dequantized_bf16" or (want == "auto" and not gpu["fp8_native"])):
        mode = "fp8_dequantized_bf16"
    load_dir = MODEL_DIR
    if mode == "fp8_dequantized_bf16":
        cache = os.environ.get("OPENONE_DEQUANT_DIR", "/cache/dequant")
        marker = dequant.read_marker(cache, info["weights_sha256"]) or \
            dequant.write_bf16_snapshot(MODEL_DIR, cache, info["weights_sha256"])
        STATE["dequant"] = marker
        load_dir = marker.get("path", cache)
    settings, warnings = policy.resolve_settings(info, mode)
    sf = os.environ.get("OPENONE_SETTINGS_FILE")
    if sf:
        o = json.loads(Path(sf).read_text(encoding="utf-8"))
        package.check_openone(o)
        settings = policy.settings_from_openone(o, f"server override {Path(sf).name}")
    elif info["format"] == "bf16":
        warnings.append("bf16 package served by vLLM: its openone.json numbers were fitted on the torch path; the bundle "
                        "README gives vLLM-fitted values. Set OPENONE_SETTINGS_FILE to use them.")
    scorer = VllmScorer(load_dir)
    STATE.update(engine=Engine(scorer), info=info, mode=mode, settings=settings, warnings=warnings, gpu=gpu,
                 runtime=scorer.runtime, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 load_seconds=round(time.time() - t0, 1))
    print("[openone-gate] ready", json.dumps({"model": info["name"], "mode": mode, "settings": settings, "gpu": gpu}), flush=True)


app = FastAPI(title="OpenOne gate service", version=__version__)


@app.on_event("startup")
def _startup():
    try:
        load()
    except Exception as e:                                        # noqa: BLE001 - surfaced in /health
        STATE["error"] = f"{type(e).__name__}: {e}"
        print("[openone-gate] load failed:", STATE["error"], flush=True)


@app.middleware("http")
async def _auth(request: Request, call_next):
    if KEY and request.url.path != "/health":
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got.encode(), f"Bearer {KEY}".encode()):
            return JSONResponse({"detail": "missing or wrong API key"}, status_code=401)
    return await call_next(request)


def _model_block():
    i = STATE["info"]
    return {"name": i["name"], "format": i["format"], "mode": STATE["mode"], "weights_sha256": i["weights_sha256"]}


@app.get("/health")
def health():
    if STATE["engine"] is None:
        return {"ok": False, "error": STATE["error"] or "loading", "service": "openone-gate", "version": __version__}
    i = STATE["info"]
    return {"ok": True, "service": "openone-gate", "version": __version__, "auth": bool(KEY),
            "model": dict(_model_block(), openone=i["openone"], status=i.get("status")),
            "settings": STATE["settings"], "warnings": STATE["warnings"], "dequant": STATE["dequant"],
            "runtime": STATE["runtime"], "gpu": STATE["gpu"], "max_state_tokens": MAX_STATE_TOKENS,
            "started": STATE["started"], "load_seconds": STATE["load_seconds"]}


def _engine():
    if STATE["engine"] is None:
        raise HTTPException(503, STATE["error"] or "model is loading")
    return STATE["engine"]


def _score(specs):
    eng = _engine()
    with LOCK:
        return eng.score(specs)


@app.post("/v1/score")
def score(body: dict):
    specs = body.get("specs")
    if not isinstance(specs, list) or not 1 <= len(specs) <= MAX_SPECS:
        raise HTTPException(422, f"specs must be a list of 1..{MAX_SPECS} items")
    for k, s in enumerate(specs):
        if not isinstance(s, dict):
            raise HTTPException(422, f"specs[{k}] must be an object")
        rv = pipeline.validate_request({"state": s.get("state"), "questions": {"q": s.get("question")}})
        if not rv["valid"]:
            raise HTTPException(422, {"spec": k, "errors": rv["errors"]})
    try:
        return {"results": _score([{"state": s["state"], "question": s["question"], "want_text": bool(s.get("want_text"))}
                                   for s in specs])}
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _decide(req, want_raw):
    ctx = {"model": _model_block(), "backend": {"kind": "remote", "device": f"cuda ({STATE['gpu']['name']})",
                                                 "runtime": STATE["runtime"]},
           "settings": STATE["settings"], "warnings": STATE["warnings"]}
    try:
        return pipeline.decide(req, _score, ctx, want_raw=want_raw)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/v1/systemone")
def systemone(body: dict):
    out = _decide(body, False)
    if not out["request_validation"]["valid"]:
        raise HTTPException(422, {"errors": out["request_validation"]["errors"]})
    if not out["response_validation"]["valid"]:
        print("[openone-gate] RESPONSE FAILED VALIDATION", json.dumps(out["response_validation"]), flush=True)
    return out["response"]


@app.post("/v1/decide")
def decide(body: dict, raw: int = 0):
    return _decide(body, bool(raw))
