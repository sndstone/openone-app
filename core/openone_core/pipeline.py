"""Decision pipeline: validate the request, score, apply temperature and thresholds, build and validate the response.

The backend only supplies letter log-probs (plus prompt facts); every response field is computed here, so local and
remote backends produce byte-compatible responses. Every response is validated against RESPONSE_SCHEMA plus semantic
checks; a failure is reported to the caller (the app writes it to the feedback log).
"""
import json
import math
import secrets
import time

from . import SCHEMA_VERSION
from .policy import GATE_OPTIONS, gate_action, is_gate, softmax, use_case
from .schemas import REQUEST_SCHEMA, RESPONSE_SCHEMA
from .vendor.decider import systemone as S1

GATE_STATE_FIELDS = ("agent", "os", "shell", "workspace_root", "cwd", "tool", "input")


# ------------------------------------------------------------------------------------------------ JSON text
class _Dup(ValueError):
    pass


def parse_json_text(text):
    """Strict JSON: -> (value, None) or (None, error dict with line/column). Rejects NaN/Infinity and duplicate keys,
    which json.loads accepts silently."""
    def pairs(kv):
        seen = {}
        for k, v in kv:
            if k in seen:
                raise _Dup(k)
            seen[k] = v
        return seen

    def const(name):
        raise ValueError(f"{name} is not valid JSON")

    def num(txt):
        v = float(txt)
        if not math.isfinite(v):
            raise ValueError(f"number {txt[:40]} is out of range (not a finite double)")
        return v

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=const, parse_float=num), None
    except json.JSONDecodeError as e:
        return None, {"kind": "syntax", "message": e.msg, "line": e.lineno, "column": e.colno, "offset": e.pos, "path": ""}
    except _Dup as e:
        return None, {"kind": "syntax", "message": f"duplicate key {str(e)!r}", "path": "", **_find_key(text, str(e))}
    except ValueError as e:
        return None, {"kind": "syntax", "message": str(e), "path": ""}


def _find_key(text, key):
    needle = json.dumps(key) + ""
    first = text.find(needle)
    pos = text.find(needle, first + 1) if first >= 0 else -1
    if pos < 0:
        return {}
    line = text.count("\n", 0, pos) + 1
    return {"line": line, "column": pos - (text.rfind("\n", 0, pos) + 1) + 1, "offset": pos}


def dumps_strict(obj, indent=None):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=indent)


# ------------------------------------------------------------------------------------------------ validation
def _pointer(path):
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in path)


def _schema_errors(schema, value):
    from jsonschema import Draft202012Validator
    v = Draft202012Validator(schema)
    out = []
    for e in sorted(v.iter_errors(value), key=lambda e: (list(map(str, e.absolute_path)), e.message)):
        leaf = e
        msg = None
        if e.context:                                          # oneOf/anyOf: report the most specific branch error
            types = [c.schema.get("type") for c in e.context if c.validator == "type" and not c.relative_path]
            if types and len(types) == len(e.context):
                got = {dict: "an object", list: "an array", str: "a string", bool: "a boolean", type(None): "null"}.get(
                    type(e.instance), "a number")
                msg = f"must be {' or '.join(str(t) for t in types)}, got {got}"
            else:
                leaf = max(e.context, key=lambda c: (len(c.absolute_path), -len(c.message)))
        msg = msg or leaf.message
        if len(msg) > 300:
            msg = msg[:300] + "…"
        out.append({"kind": "schema", "path": _pointer(leaf.absolute_path), "message": msg, "keyword": leaf.validator})
    return out


def _nonfinite(x, path=""):
    """JSON pointers of float values that are NaN or infinite (strict JSON cannot carry them)."""
    if isinstance(x, float) and not math.isfinite(x):
        return [path]
    if isinstance(x, dict):
        return [p for k, v in x.items() for p in _nonfinite(v, f"{path}/{_pointer([k])[1:]}")]
    if isinstance(x, list):
        return [p for i, v in enumerate(x) for p in _nonfinite(v, f"{path}/{i}")]
    return []


def validate_request(req):
    """-> {"valid", "errors": [{kind, path, message}], "warnings": [...]}"""
    bad = _nonfinite(req)
    if bad:
        return {"valid": False, "warnings": [],
                "errors": [{"kind": "syntax", "path": p, "message": "number is NaN or infinite, which JSON cannot carry"} for p in bad]}
    errors = _schema_errors(REQUEST_SCHEMA, req)
    warnings = []
    if not errors:
        for qid, q in req["questions"].items():
            crit = q["criteria"]
            names = list(crit) if isinstance(crit, dict) else crit
            stripped = [n.strip() for n in names]
            if len(set(stripped)) != len(stripped):
                errors.append({"kind": "semantic", "path": f"/questions/{qid}/criteria",
                               "message": "two options have the same name once surrounding spaces are removed"})
            try:
                uc = use_case(q)
                S1.render_question({k: q[k] for k in ("type", "instructions", "criteria") if k in q})
            except ValueError as e:
                errors.append({"kind": "semantic", "path": f"/questions/{qid}", "message": str(e)})
                continue
            if uc == "toolcall":
                st = req["state"]
                if not isinstance(st, dict):
                    warnings.append({"path": "/state", "message": "tool-call state should be an object with agent, os, "
                                     "shell, workspace_root, cwd, tool, input and user_task (INTEGRATION.md)"})
                else:
                    miss = [f for f in GATE_STATE_FIELDS if f not in st]
                    if miss:
                        warnings.append({"path": "/state", "message": "tool-call state lacks required field(s): " + ", ".join(miss)})
                    if "user_task" not in st:
                        warnings.append({"path": "/state", "message": "no user_task: recommended, the gate cannot see "
                                         "limits the user stated without it"})
            elif set(n.lower() for n in names) == set(GATE_OPTIONS):
                warnings.append({"path": f"/questions/{qid}/criteria", "message": "options look like allow/ask/block "
                                 "but do not match exactly (case); scored as a bucket question"})
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def _near(a, b, tol=1e-4):
    return abs(a - b) <= tol


def validate_response(resp, req=None):
    """Schema + semantic checks + strict JSON round trip. -> {"valid", "errors"}."""
    errors = []
    try:
        text = dumps_strict(resp)
        back, err = parse_json_text(text)
        if err:
            errors.append(err)
        elif back != resp:
            errors.append({"kind": "syntax", "path": "", "message": "response does not survive a JSON round trip"})
    except (TypeError, ValueError) as e:
        errors.append({"kind": "syntax", "path": "", "message": f"response is not serialisable as strict JSON: {e}"})
        return {"valid": False, "errors": errors}
    errors += _schema_errors(RESPONSE_SCHEMA, resp)
    if errors:
        return {"valid": False, "errors": errors}
    s = resp["settings"]
    total_in = 0
    if req is not None and set(resp["answers"]) != set(req.get("questions", {})):
        errors.append({"kind": "semantic", "path": "/answers", "message": "answer ids differ from the request's question ids"})
    for qid, a in resp["answers"].items():
        at = f"/answers/{qid}"
        p = a["probabilities"]
        n = len(p)
        tot = sum(p.values())
        if abs(tot - 1) > max(1e-3, n * 5e-5):
            errors.append({"kind": "semantic", "path": at + "/probabilities", "message": f"probabilities sum to {tot:.6f}, not 1"})
        if a["choice"] not in p:
            errors.append({"kind": "semantic", "path": at + "/choice", "message": f"choice {a['choice']!r} is not one of the options"})
            continue
        if p[a["choice"]] < max(p.values()) - 1e-9:
            errors.append({"kind": "semantic", "path": at + "/choice", "message": "choice is not the most probable option"})
        if not _near(a["x_p_max"], p[a["choice"]]):
            errors.append({"kind": "semantic", "path": at + "/x_p_max", "message": "x_p_max differs from p(choice)"})
        if req is not None and qid in req.get("questions", {}):
            q = req["questions"][qid]
            names = [str(x) for x in (q["criteria"] if isinstance(q["criteria"], list) else list(q["criteria"]))]
            if list(p) != names:
                errors.append({"kind": "semantic", "path": at + "/probabilities", "message": "probability keys differ from the request's options"})
            if a["use_case"] != use_case(q):
                errors.append({"kind": "semantic", "path": at + "/use_case", "message": "use_case does not match the question"})
        want_t = s["temperature"][a["use_case"]]
        if not _near(a["temperature"], want_t, 1e-9):
            errors.append({"kind": "semantic", "path": at + "/temperature", "message": f"temperature {a['temperature']} differs from settings {want_t}"})
        if a["use_case"] == "toolcall":
            v = a["verdict"]
            g = s["gate"]
            if not (_near(v["auto_run_at"], g["auto_run_at"], 1e-12) and _near(v["alarm_at"], g["alarm_at"], 1e-12)):
                errors.append({"kind": "semantic", "path": at + "/verdict", "message": "verdict thresholds differ from settings"})
            exp, _ = gate_action(p, g, oversized=a["truncated"])
            if a["truncated"] and v["action"] == "auto_run":
                errors.append({"kind": "semantic", "path": at + "/verdict/action",
                               "message": "a truncated tool-call state must never auto-run"})
            elif exp != v["action"]:
                # probabilities are rounded to 4 decimals: the claimed action must still be reachable, within 1e-4,
                # on every condition the rule needs for it
                tol, pb, pa = 1e-4, p["block"], p["allow"]
                act = v["action"]
                if act == "alarm":
                    explained = pb >= g["alarm_at"] - tol
                elif act == "auto_run":
                    explained = pb < g["alarm_at"] + tol and pa >= g["auto_run_at"] - tol and not a["truncated"]
                else:                                          # ask
                    explained = pb < g["alarm_at"] + tol and (a["truncated"] or pa < g["auto_run_at"] + tol)
                if not explained:
                    errors.append({"kind": "semantic", "path": at + "/verdict/action",
                                   "message": f"action {v['action']!r} does not follow from the probabilities and thresholds (expected {exp!r})"})
        else:
            if not _near(a["accept_at"], s["bucket"]["accept_at"], 1e-12):
                errors.append({"kind": "semantic", "path": at + "/accept_at", "message": "accept_at differs from settings"})
            exp = p[a["choice"]] >= a["accept_at"]
            if exp != a["accepted"] and abs(p[a["choice"]] - a["accept_at"]) > 1e-4:
                errors.append({"kind": "semantic", "path": at + "/accepted", "message": "accepted does not follow from p(choice) and accept_at"})
        if a["elided"] and a["use_case"] != "toolcall":
            errors.append({"kind": "semantic", "path": at + "/elided", "message": "only tool-call states are elided"})
        total_in += a["input_tokens"]
    if resp["usage"]["input_tokens"] != total_in:
        errors.append({"kind": "semantic", "path": "/usage/input_tokens", "message": "usage.input_tokens is not the sum over answers"})
    return {"valid": not errors, "errors": errors}


# ------------------------------------------------------------------------------------------------ answers
def build_answer(question, raw, settings, nd=4):
    """question: request question; raw: one backend result. -> (answer, raw view for the raw-mode panel)."""
    uc = use_case(question)
    T = settings["temperature"][uc]
    lp = raw["letter_logprobs"]
    names = raw["names"]
    p1 = softmax(lp, 1.0)
    pT = softmax(lp, T)
    rq = S1.render_question({k: question[k] for k in ("type", "instructions", "criteria") if k in question})
    base = S1.format_answer(rq, pT, nd)                     # choice, confidence, x_p_max, certainty, probabilities
    ans = {"type": "choice", "use_case": uc, "choice": str(base["choice"]),
           "probabilities": {str(k): v for k, v in base["probabilities"].items()},
           "x_p_max": base["x_p_max"], "confidence": base["confidence"], "certainty": base["certainty"],
           "temperature": T, "input_tokens": raw["input_tokens"], "state_tokens": raw["state_tokens"],
           "max_state_tokens": raw["max_state_tokens"], "elided": raw["elided"], "truncated": raw["truncated"]}
    full = dict(zip(names, pT))
    if uc == "toolcall":
        g = settings["gate"]
        action, rule = gate_action(full, g, oversized=raw["truncated"])
        ans["verdict"] = {"action": action, "rule": rule, "auto_run_at": g["auto_run_at"], "alarm_at": g["alarm_at"]}
        thresholds = dict(g)
    else:
        acc = settings["bucket"]["accept_at"]
        ans["accepted"] = full[ans["choice"]] >= acc
        ans["accept_at"] = acc
        thresholds = {"accept_at": acc}
        rule = (f"p({ans['choice']}) {full[ans['choice']]:.4f} {'>=' if ans['accepted'] else '<'} accept threshold {acc:.4f}")
    view = {"use_case": uc, "option_letters": dict(zip(names, raw["letters"])),
            "letter_logprobs": {n: round(v, 6) for n, v in zip(names, lp)},
            "probabilities_T1": {n: round(v, 6) for n, v in zip(names, p1)},
            "temperature": T, "probabilities_T": {n: round(v, 6) for n, v in zip(names, pT)},
            "thresholds": thresholds, "rule": rule, "prompt_tokens": raw["input_tokens"],
            "state_tokens": raw["state_tokens"], "max_state_tokens": raw["max_state_tokens"],
            "elided": raw["elided"], "truncated": raw["truncated"], "forward_ms": raw.get("forward_ms"),
            "prompt_text": raw.get("prompt_text")}
    return ans, view


def new_id():
    return "dec_" + secrets.token_hex(12)


def now_iso():
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


def decide(req, score, ctx, want_raw=False):
    """req: parsed request; score(list of {"state", "question", "want_text"}) -> list of raw dicts;
    ctx: {"model": {...}, "backend": {...}, "settings": {...}, "warnings": [...]}.
    -> {"request_validation", "response", "response_validation", "raw"} (response None when the request is invalid)."""
    rv = validate_request(req)
    out = {"request_validation": rv, "response": None, "response_validation": None, "raw": None}
    if not rv["valid"]:
        return out
    t0 = time.time()
    qids = list(req["questions"])
    raws = score([{"state": req["state"], "question": req["questions"][q], "want_text": want_raw} for q in qids])
    answers, views = {}, {}
    warnings = list(ctx.get("warnings", []))
    for qid, raw in zip(qids, raws):
        a, v = build_answer(req["questions"][qid], raw, ctx["settings"])
        answers[qid], views[qid] = a, v
        if raw["truncated"]:
            warnings.append(f"{qid}: state is {raw['state_tokens']} tokens, over the {raw['max_state_tokens']}-token limit; "
                            "the model saw a cut state" + (" (tool call: never auto-run)" if a["use_case"] == "toolcall" else ""))
        elif raw["elided"]:
            warnings.append(f"{qid}: long string fields were elided to fit the {raw['max_state_tokens']}-token state limit")
    latency = round((time.time() - t0) * 1000, 2)
    s = ctx["settings"]
    resp = {"object": "openone.decision", "schema_version": SCHEMA_VERSION, "id": new_id(), "created": now_iso(),
            "model": ctx["model"], "backend": ctx["backend"],
            "settings": {"temperature": dict(s["temperature"]), "gate": dict(s["gate"]), "bucket": dict(s["bucket"]),
                         "source": dict(s["source"]), "provisional": bool(s.get("provisional", False))},
            "answers": answers, "usage": {"input_tokens": sum(a["input_tokens"] for a in answers.values())},
            "latency_ms": latency, "routing": {"model": "decider"}, "warnings": [str(w) for w in warnings]}
    out["response"] = resp
    out["response_validation"] = validate_response(resp, req)
    if want_raw:
        out["raw"] = {"questions": views, "backend_ms": latency}
    return out


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)
