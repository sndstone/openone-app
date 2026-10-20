"""Unit tests for openone_core (no model needed; dequant tests need torch + safetensors and are skipped otherwise).

python -m pytest tests/test_core.py -q
"""
import copy
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "app")]
from openone_core import package, pipeline, policy  # noqa: E402

OPENONE = {"usecase_temperatures": {"usecase_bucket": 1.34, "usecase_toolcall": 1.28},
           "gate": {"auto_run_if_p_allow_at_least": 0.8, "alarm_if_p_block_at_least": 0.4532}}


# ------------------------------------------------------------------------------------------------ packages
def make_pkg(root, name="pkg-bf16", extra=None, bad=None):
    d = root / name
    (d / "decider").mkdir(parents=True)
    files = {"config.json": json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}), "tokenizer.json": "{}",
             "model.safetensors": "weights" * 1000, "openone.json": json.dumps(OPENONE), "decider_config.json": "{}",
             "decider/prompt.py": "x"}
    files.update(extra or {})
    for rel, txt in files.items():
        (d / rel).write_text(txt, encoding="utf-8")
    sums = "".join(f"{hashlib.sha256((d / r).read_bytes()).hexdigest()}  {r}\n" for r in files)
    (d / "SHA256SUMS").write_text(sums, encoding="utf-8")
    (d / "README.md").write_text("doc", encoding="utf-8")
    if bad:
        (d / bad).write_text("tampered", encoding="utf-8")
    return d


def zip_dir(d, out):
    with zipfile.ZipFile(out, "w") as z:
        for p in d.rglob("*"):
            if p.is_file():
                z.write(p, f"{d.name}/{p.relative_to(d).as_posix()}")
    return out


def test_install_zip_and_folder(tmp_path):
    d = make_pkg(tmp_path / "src")
    res = package.install(zip_dir(d, tmp_path / "p.zip"), tmp_path / "models")
    assert res["verified"] == 6 and res["unverified_docs"] == ["README.md"]
    info = res["info"]
    assert info["format"] == "bf16" and info["local_mode"] == "native"
    assert not info["prompt_code"]["matches_app"]                       # our fake prompt.py differs from the vendored one
    with pytest.raises(package.PackageError, match="already installed"):
        package.install(d, tmp_path / "models")
    assert package.install(d, tmp_path / "models", replace=True)["verified"] == 6
    assert package.verify_installed(tmp_path / "models" / "pkg-bf16")["ok"]


def test_tampered_file_is_rejected_and_cleaned(tmp_path):
    d = make_pkg(tmp_path / "src", bad="model.safetensors")
    with pytest.raises(package.PackageError, match="checksum mismatch for model.safetensors"):
        package.install(zip_dir(d, tmp_path / "p.zip"), tmp_path / "models")
    assert not any((tmp_path / "models").iterdir())                      # partial copy removed


def test_unsafe_zip_member(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("pkg/SHA256SUMS", "0" * 64 + "  ../x\n")
        f.writestr("pkg/../../x", "boom")
    with pytest.raises(package.PackageError, match="unsafe path"):
        package.install(z, tmp_path / "models")


def test_fp8_detection(tmp_path):
    d = make_pkg(tmp_path / "src", "pkg-fp8", extra={"openone_quant.json": "{}"})
    info = package.read_package(d)
    assert info["format"] == "fp8" and info["local_mode"] == "fp8_dequantized_bf16"


def test_bad_openone(tmp_path):
    with pytest.raises(package.PackageError, match="alarm_if_p_block_at_least"):
        package.check_openone({"usecase_temperatures": {"usecase_bucket": 1, "usecase_toolcall": 1},
                               "gate": {"auto_run_if_p_allow_at_least": 0.8, "alarm_if_p_block_at_least": 1.5}})


# ------------------------------------------------------------------------------------------------ settings
FP8_SHA = "085b0834751a20ff2abf6389f605ce38e0634e7825f8623d9cc45a72f86cdd8e"
BF16_SHA = "6352dc26ab6d4a0afd60114448f7dcb1fe3ab38a86f200c886969ae598ed7b26"


def test_fallback_settings_fitted_for_known_fp8():
    fp8 = {"name": "x-fp8", "format": "fp8", "weights_sha256": FP8_SHA,
           "openone": dict(OPENONE, source_bf16_weights_sha256=BF16_SHA, gate={"auto_run_if_p_allow_at_least": 0.8,
                                                                               "alarm_if_p_block_at_least": 0.4019})}
    s, w = policy.resolve_settings(fp8, "fp8_dequantized_bf16")
    assert not s["provisional"] and s["gate"]["alarm_at"] == 0.4133 and s["gate"]["auto_run_at"] == 0.8
    assert s["temperature"] == {"toolcall": 1.28, "bucket": 1.34} and "fitted under vLLM" in s["source"]["gate"]
    assert w == [policy.FALLBACK_WARNING] and "close to the bf16 model, slightly below it" in w[0]
    fo = policy.fallback_openone(fp8["openone"], FP8_SHA)
    assert fo["runtime"] == "bf16 (dequantized from FP8)" and fo["gate"]["alarm_if_p_block_at_least"] == 0.4133
    package.check_openone(fo)


def test_fallback_settings_provisional_for_unknown_fp8():
    fp8 = {"name": "y-fp8", "format": "fp8", "weights_sha256": "f" * 64, "openone": dict(OPENONE, source_bf16_weights_sha256=BF16_SHA)}
    s, w = policy.resolve_settings(fp8, "fp8_dequantized_bf16")
    assert s["provisional"] and s["gate"]["alarm_at"] == pytest.approx(0.4532229741804979)
    assert w[0] == policy.FALLBACK_WARNING
    bf16 = {"name": "mine", "format": "bf16", "weights_sha256": BF16_SHA,
            "openone": dict(OPENONE, gate={"auto_run_if_p_allow_at_least": 0.8, "alarm_if_p_block_at_least": 0.5})}
    s, _ = policy.resolve_settings(fp8, "fp8_dequantized_bf16", [bf16])
    assert s["gate"]["alarm_at"] == 0.5 and "mine/openone.json" in s["source"]["gate"]
    assert policy.fallback_openone(fp8["openone"], "f" * 64)["status"].startswith("provisional")
    s, w = policy.resolve_settings(bf16, "native")
    assert not s["provisional"] and not w


def test_gate_action_precedence():
    g = {"auto_run_at": 0.5, "alarm_at": 0.3}
    assert policy.gate_action({"allow": 0.55, "ask": 0.1, "block": 0.35}, g)[0] == "alarm"
    assert policy.gate_action({"allow": 0.9, "ask": 0.05, "block": 0.05}, g)[0] == "auto_run"
    assert policy.gate_action({"allow": 0.9, "ask": 0.05, "block": 0.05}, g, oversized=True)[0] == "ask"


def test_overrides_validated():
    base = policy.settings_from_openone(OPENONE)
    assert policy.merge_overrides(base, {"temperature": {"bucket": 2}})["temperature"]["bucket"] == 2.0
    with pytest.raises(ValueError):
        policy.merge_overrides(base, {"gate": {"alarm_at": 2}})


# ------------------------------------------------------------------------------------------------ JSON + validation
def test_strict_json():
    assert pipeline.parse_json_text('{"a": 1}') == ({"a": 1}, None)
    v, e = pipeline.parse_json_text('{"a": 1, "a": 2}')
    assert v is None and "duplicate" in e["message"] and e["line"] == 1
    v, e = pipeline.parse_json_text('{"a": NaN}')
    assert v is None
    v, e = pipeline.parse_json_text('{\n"a": [1,]\n}')
    assert e["line"] == 2


def fake_score(specs):
    out = []
    for s in specs:
        names = policy.criteria_names(s["question"])
        lp = policy.log_softmax([float(-i) for i in range(len(names))])
        out.append({"names": names, "letters": [chr(65 + i) for i in range(len(names))], "letter_logprobs": lp,
                    "input_tokens": 100, "state_tokens": 50, "max_state_tokens": 2048, "elided": False, "truncated": False})
    return out


CTX = {"model": {"name": "m", "format": "bf16", "mode": "native", "weights_sha256": None},
       "backend": {"kind": "local", "device": "cpu", "runtime": "test"}, "settings": policy.settings_from_openone(OPENONE), "warnings": []}
GATE_REQ = {"state": {"agent": "codex", "os": "linux", "shell": "bash", "workspace_root": "/w", "cwd": "/w", "tool": "shell",
                      "input": {"command": ["ls"]}, "user_task": "x"},
            "questions": {"v": {"type": "choice", "instructions": "Decide.", "criteria": policy.DEFAULT_POLICY},
                          "b": {"instructions": "Pick.", "criteria": ["a", "b", "c", "d"]}}}


def test_decide_and_response_valid():
    out = pipeline.decide(GATE_REQ, fake_score, CTX, want_raw=True)
    assert out["request_validation"]["valid"] and out["response_validation"]["valid"], out["response_validation"]
    r = out["response"]
    assert r["answers"]["v"]["use_case"] == "toolcall" and r["answers"]["v"]["verdict"]["action"] == "ask"
    assert r["answers"]["b"]["use_case"] == "bucket" and r["answers"]["b"]["choice"] == "a"
    assert r["usage"]["input_tokens"] == 200


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r["answers"]["v"]["probabilities"].update(allow=0.9), "sum to"),
    (lambda r: r["answers"]["v"].update(choice="block"), "most probable"),
    (lambda r: r["answers"]["v"]["verdict"].update(action="auto_run"), "does not follow"),
    (lambda r: r["answers"]["b"].update(accepted=True), "accepted does not follow"),
    (lambda r: r["answers"]["b"].update(temperature=9.9), "temperature"),
    (lambda r: r["usage"].update(input_tokens=1), "usage.input_tokens"),
    (lambda r: r["answers"]["v"]["probabilities"].update(allow=float("nan")), "strict JSON"),
    (lambda r: r.update(extra=1), "Additional properties"),
    (lambda r: r["answers"].pop("b"), "answer ids"),
])
def test_corrupted_responses_fail(mutate, needle):
    out = pipeline.decide(GATE_REQ, fake_score, CTX)
    resp = copy.deepcopy(out["response"])
    mutate(resp)
    v = pipeline.validate_response(resp, GATE_REQ)
    assert not v["valid"] and any(needle in e["message"] for e in v["errors"]), v["errors"]


def test_request_errors_and_warnings():
    v = pipeline.validate_request({"state": {"text": "x"}, "questions": {"q": {"type": "noul", "instructions": "", "criteria": {"a": 1}}}})
    paths = {e["path"] for e in v["errors"]}
    assert "/questions/q/type" in paths and "/questions/q/instructions" in paths
    v = pipeline.validate_request({"state": {"tool": "Bash"}, "questions": {"v": {"instructions": "x", "criteria": policy.DEFAULT_POLICY}}})
    assert v["valid"] and any("lacks required" in w["message"] for w in v["warnings"])
    v = pipeline.validate_request({"state": "x", "questions": {"v": {"instructions": "x", "criteria": {"Allow": "a", "Ask": "b", "Block": "c"}}}})
    assert v["valid"] and any("case" in w["message"] for w in v["warnings"])
    v = pipeline.validate_request({"state": "x", "questions": {"v": {"instructions": "x", "criteria": {"a ": "x", "a": "y"}}}})
    assert not v["valid"]


# ------------------------------------------------------------------------------------------------ dequant (synthetic)
def test_dequant_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from openone_core import dequant
    w = torch.randn(8, 4)
    scale = (w.abs().amax(1, keepdim=True) / 448).to(torch.bfloat16)
    q = (w / scale.float()).to(torch.float8_e4m3fn)
    src = tmp_path / "fp8"
    src.mkdir()
    st.save_file({"l.weight": q, "l.weight_scale": scale, "n.weight": torch.ones(4, dtype=torch.bfloat16)}, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"a": 1, "quantization_config": {}}))
    (src / "openone.json").write_text(json.dumps(OPENONE))
    m = dequant.write_bf16_snapshot(src, tmp_path / "out", "abc")
    assert json.loads((tmp_path / "out" / "openone.json").read_text())["runtime"] == "bf16 (dequantized from FP8)"
    out = st.load_file(str(tmp_path / "out" / "model.safetensors"))
    assert set(out) == {"l.weight", "n.weight"} and out["l.weight"].dtype == torch.bfloat16
    assert torch.allclose(out["l.weight"].float(), (q.float() * scale.float()).bfloat16().float())
    assert "quantization_config" not in json.loads((tmp_path / "out" / "config.json").read_text())
    assert dequant.read_marker(tmp_path / "out", "abc")["tensors"] == 2 and dequant.read_marker(tmp_path / "out", "zzz") is None


def test_dequant_rejects_unknown_layout(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from openone_core import dequant
    src = tmp_path / "fp8"
    src.mkdir()
    st.save_file({"l.weight": torch.zeros(4, 4).to(torch.float8_e4m3fn), "l.weight_scale": torch.ones(1, 1, dtype=torch.bfloat16)},
                 str(src / "model.safetensors"))
    with pytest.raises(ValueError, match="per output channel"):
        dequant.dequantize_state_dict(src)


# ------------------------------------------------------------------------------------------------ review round 1 regressions
def test_truncated_gate_never_auto_runs_even_at_threshold():
    out = pipeline.decide(GATE_REQ, fake_score, CTX)
    resp = copy.deepcopy(out["response"])
    a = resp["answers"]["v"]
    a["truncated"] = True
    a["probabilities"] = {"allow": 0.8, "ask": 0.15, "block": 0.05}
    a["x_p_max"] = 0.8
    a["choice"] = "allow"
    a["verdict"]["action"] = "auto_run"
    v = pipeline.validate_response(resp, GATE_REQ)
    assert not v["valid"] and any("never auto-run" in e["message"] for e in v["errors"])


def test_out_of_range_numbers_rejected():
    v, e = pipeline.parse_json_text('{"state": {"x": 1e400}, "questions": {}}')
    assert v is None and "finite" in e["message"]
    r = pipeline.validate_request({"state": {"x": float("inf")}, "questions": {"q": {"instructions": "x", "criteria": ["a", "b"]}}})
    assert not r["valid"] and r["errors"][0]["path"] == "/state/x"
    with pytest.raises(package.PackageError):
        package.check_openone({"usecase_temperatures": {"usecase_bucket": float("inf"), "usecase_toolcall": 1},
                               "gate": {"auto_run_if_p_allow_at_least": 0.8, "alarm_if_p_block_at_least": 0.4}})


def test_case_collision_and_reserved_names(tmp_path):
    d = make_pkg(tmp_path / "src")
    z = zip_dir(d, tmp_path / "p.zip")
    sums = (d / "SHA256SUMS").read_text() + hashlib.sha256(b"other").hexdigest() + "  MODEL.SAFETENSORS\n"
    with zipfile.ZipFile(z, "a") as f:
        f.writestr("pkg-bf16/MODEL.SAFETENSORS", b"other")
    zz = tmp_path / "p2.zip"
    with zipfile.ZipFile(z) as fi, zipfile.ZipFile(zz, "w") as fo:          # replace SHA256SUMS with one listing both names
        for i in fi.infolist():
            fo.writestr(i.filename, sums if i.filename.endswith("SHA256SUMS") else fi.read(i))
    with pytest.raises(package.PackageError, match="collide"):
        package.install(zz, tmp_path / "models")
    with pytest.raises(package.PackageError, match="not portable"):
        package.parse_sums("0" * 64 + "  decider/aux.py\n")
    with pytest.raises(package.PackageError, match="not portable"):
        package.parse_sums("0" * 64 + "  model.safetensors.\n")


# ------------------------------------------------------------------------------------------------ review round 2 regressions
@pytest.mark.parametrize("probs, action, ok", [
    ({"allow": 0.2, "ask": 0.3867, "block": 0.4133}, "auto_run", False),    # near alarm, but allow far below auto-run
    ({"allow": 0.2, "ask": 0.3868, "block": 0.4132}, "alarm", True),        # within rounding of the alarm threshold
    ({"allow": 0.8, "ask": 0.15, "block": 0.05}, "ask", True),              # exactly at auto-run: rounding may explain ask
    ({"allow": 0.5, "ask": 0.45, "block": 0.05}, "alarm", False),
])
def test_verdict_must_be_attainable(probs, action, ok):
    ctx = dict(CTX, settings=policy.merge_overrides(policy.settings_from_openone(OPENONE), {"gate": {"alarm_at": 0.4133, "auto_run_at": 0.8}}))
    resp = copy.deepcopy(pipeline.decide(GATE_REQ, fake_score, ctx)["response"])
    a = resp["answers"]["v"]
    a["probabilities"] = probs
    a["choice"] = max(probs, key=probs.get)
    a["x_p_max"] = probs[a["choice"]]
    a["verdict"]["action"] = action
    errs = [e for e in pipeline.validate_response(resp, GATE_REQ)["errors"] if e["path"].endswith("/verdict/action")]
    assert (not errs) == ok, errs


def _controller(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENONE_DATA_DIR", str(tmp_path))
    from openone_app.controller import Controller
    return Controller()


def test_concurrent_starts_get_unique_generations(tmp_path, monkeypatch):
    import threading
    import time
    c = _controller(tmp_path, monkeypatch)
    seen, gate = [], threading.Event()

    def fake_start(job, b, gen):
        seen.append(gen)
        gate.wait(5)
        class B:
            kind = "local"
            def stop(self): pass
            def alive(self): return True
        c._install(gen, B(), {"state": "ready", "kind": "local", "model": {}, "base_settings": {}})
        return gen
    monkeypatch.setattr(c, "_start", fake_start)
    jobs = [c.start_backend(), c.start_backend(), c.start_backend()]
    time.sleep(0.3)
    gate.set()
    for j in jobs:
        j.thread.join(5)
    assert len(set(seen)) == 3 and [j.state for j in jobs] == ["cancelled", "cancelled", "done"], (seen, [j.state for j in jobs])
    assert c.backend_state["state"] == "ready"


def test_settings_captured_at_start(tmp_path, monkeypatch):
    c = _controller(tmp_path, monkeypatch)
    captured = policy.settings_from_openone(OPENONE)

    class B:
        kind = "local"
        def stop(self): pass
        def alive(self): return True
    gen = c._detach({"state": "starting"})
    c._install(gen, B(), {"state": "ready", "kind": "local", "device": "cpu", "runtime": "t",
                          "model": {"name": "m", "format": "bf16", "mode": "native", "weights_sha256": None},
                          "base_settings": captured, "warnings": []})
    c.store.settings["active_model"] = "something-else-on-disk"
    s, _ = c.base_settings()
    assert s["gate"]["alarm_at"] == 0.4532 and s["temperature"]["toolcall"] == 1.28


def test_no_start_during_import_and_replace_stops_starting_backend(tmp_path, monkeypatch):
    import threading
    c = _controller(tmp_path, monkeypatch)
    release = threading.Event()
    j = c.jobs.start("import", lambda job: release.wait(5))
    with pytest.raises(ValueError, match="import is running"):
        c.start_backend()
    release.set()
    j.thread.join(5)
    src = make_pkg(tmp_path / "src")
    package.install(src, c.models_dir())
    gen = c._detach({"state": "starting", "kind": "local"})
    job = c.import_model(str(src), replace=True)
    job.thread.join(10)
    assert job.state == "done" and c._gen > gen and c.backend_state["state"] == "stopped"


@pytest.mark.skipif(sys.platform == "win32", reason="symlinked venv interpreters are a POSIX thing")
def test_worker_keeps_venv_interpreter_symlink(tmp_path):
    # a venv's bin/python links to the base interpreter; resolving it would lose the venv's site-packages
    from openone_app.backends import LocalWorker
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(sys.executable)
    assert LocalWorker(str(link), tmp_path / "w.log").python_exe == str(link)


# ------------------------------------------------------------------------------------------------ bucket file batches
from openone_app import records  # noqa: E402


def test_records_csv_sniff_quotes_and_duplicate_headers():
    r = records.parse('id;text;text\n1;"refund please; twice\nthanks";x\n\n2;app crashes;y\n', filename="t.csv")
    assert r["format"] == "csv" and r["fields"] == ["id", "text", "text_2"]
    assert r["records"][0]["text"] == "refund please; twice\nthanks" and len(r["records"]) == 2


def test_records_json_wrapper_strings_and_jsonl_errors():
    assert records.parse('{"items": [{"a": 1}, "plain"]}')["records"] == [{"a": 1}, {"text": "plain"}]
    assert records.parse('[{"t": "x"}]')["format"] == "json"
    assert records.parse('{"t": "x"}\n{"t": "y"}\n')["format"] == "jsonl"
    with pytest.raises(records.RecordError, match="line 2"):
        records.parse('{"t": "x"}\n{"t": }\n', fmt="jsonl")
    with pytest.raises(records.RecordError, match="line"):
        records.parse('[{"t": 1},]', fmt="json")


def test_records_xml_wrappers_attributes_and_repeats():
    x = """<export><tickets>
      <ticket id="7"><subject>Refund</subject><tag>a</tag><tag>b</tag></ticket>
      <ticket id="8"><subject>Crash</subject></ticket>
    </tickets></export>"""
    r = records.parse(x)
    assert r["format"] == "xml" and r["record_tag"] == "export/tickets/ticket" and len(r["records"]) == 2
    assert r["records"][0] == {"@id": "7", "subject": "Refund", "tag": ["a", "b"]}
    assert records.parse(x, record_tag="subject")["records"] == [{"text": "Refund"}, {"text": "Crash"}]
    with pytest.raises(records.RecordError, match="DOCTYPE"):
        records.parse('<!DOCTYPE x [<!ENTITY a "aaaa">]><r><i>&a;</i></r>')
    with pytest.raises(records.RecordError, match="XML line"):
        records.parse("<r><i></r>")


def test_records_state_for():
    rec = {"id": "1", "subject": "Refund", "body": "", "n": 3}
    assert records.state_for(rec, ["subject"]) == {"text": "Refund"}
    assert records.state_for(rec, ["n"]) == {"n": 3}
    assert records.state_for(rec, ["subject", "body"]) == {"subject": "Refund", "body": ""}
    assert records.state_for(rec, ["*"]) == rec
    assert records.state_for(rec, ["body"]) is None


ROWS = [{"index": 0, "record": {"t": "=HYPERLINK(1)", "n": "-5"}, "status": "ok", "bucket": "b", "p": 0.9, "accepted": True,
         "probabilities": {"a": 0.1, "b": 0.9}},
        {"index": 1, "record": {"t": "x", "n": "2"}, "status": "ok", "bucket": "a", "p": 0.6, "accepted": False,
         "probabilities": {"a": 0.6, "b": 0.4}},
        {"index": 2, "record": {"t": "y", "n": "3"}, "status": "ok", "bucket": "a", "p": 0.95, "accepted": True,
         "probabilities": {"a": 0.95, "b": 0.05}},
        {"index": 3, "record": {"t": "", "n": "4"}, "status": "empty", "error": "nothing to classify"}]


def test_records_sort_and_exports():
    assert [r["index"] for r in records.sort_rows(ROWS, ["a", "b"], "bucket")] == [2, 1, 0, 3]
    assert [r["index"] for r in records.sort_rows(ROWS, ["a", "b"], "confidence")] == [2, 0, 1, 3]
    out = records.to_csv(ROWS, ["a", "b"], ["t", "n"])
    assert out.startswith("﻿")
    lines = list(__import__("csv").reader(io.StringIO(out[1:])))
    assert lines[0] == ["t", "n", "bucket", "p_bucket", "accepted", "status", "p_a", "p_b", "error"]
    assert lines[1][:2] == ["'=HYPERLINK(1)", "-5"]                 # formula guarded, negative number untouched
    z = zipfile.ZipFile(io.BytesIO(records.to_zip(ROWS, ["a", "b"], ["t", "n"])))
    assert z.namelist() == ["a.csv", "b.csv", "below_threshold.csv", "errors.csv"]
    import xml.etree.ElementTree as ET
    root = ET.fromstring(records.to_xml(ROWS, ["a", "b"], {"records": 4}))
    assert [e.get("bucket") for e in root] == ["b", "a", "a", ""] and root[0].find("data/t").text == "=HYPERLINK(1)"
    xr = [{"@id": "7", "subject": "Refund", "tag": ["a", "b"], "meta": {"lang": "en"}, "bad key": "x"}]
    back = records.parse(records.to_xml([dict(ROWS[0], record=xr[0])], ["a", "b"], {}), record_tag="data")["records"]
    assert back == [{"@id": "7", "subject": "Refund", "tag": ["a", "b"], "meta": {"lang": "en"},
                     "field": {"@name": "bad key", "text": "x"}}]
    j = json.loads(records.to_json(ROWS, ["a", "b"], {"records": 4}))
    assert j["records"][0]["_openone"]["bucket"] == "b" and j["buckets"] == ["a", "b"]
    z = zipfile.ZipFile(io.BytesIO(records.to_zip([dict(ROWS[0], bucket="a b"), dict(ROWS[2], bucket="a_b")], ["a b", "a_b"], ["t"])))
    assert z.namelist() == ["a_b.csv", "a_b_2.csv"]


def test_bucket_batch_end_to_end_with_fake_backend(tmp_path, monkeypatch):
    c = _controller(tmp_path, monkeypatch)

    def score(specs):                                  # the bucket named in the text wins
        out = []
        for s in specs:
            names, text = policy.criteria_names(s["question"]), json.dumps(s["state"]).lower()
            lg = [0.0 if n in text else -4.0 for n in names]
            out.append({"names": names, "letters": [chr(65 + i) for i in range(len(names))], "letter_logprobs": policy.log_softmax(lg),
                        "input_tokens": 60, "state_tokens": 20, "max_state_tokens": 2048, "elided": False, "truncated": False})
        return out

    B = type("B", (), {"kind": "local", "score": staticmethod(score)})
    s = policy.settings_from_openone(OPENONE) | {"bucket": {"accept_at": 0.8}}
    monkeypatch.setattr(c, "backend_ready", lambda: None)
    monkeypatch.setattr(c, "_snapshot", lambda use_overrides=True: (B, dict(CTX, settings=s)))
    monkeypatch.setattr(c, "effective_settings", lambda use_overrides=True: (s, []))
    parsed = records.parse("ref,msg,label\nT1,need billing help,billing\nT2,technical issue,technical\nT3,,other\nT4,hello,other\n")
    q = {"type": "choice", "instructions": "Route it.", "criteria": {"billing": "money", "technical": "bugs", "other": None}}
    with pytest.raises(ValueError, match="unknown field"):
        c.start_bucket_batch(parsed, q, ["nope"])
    job = c.start_bucket_batch(parsed, q, ["msg"], id_field="ref", expected_field="label")
    job.thread.join(10)
    assert job.state == "done", job.error
    res = job.result
    assert res["counts"]["billing"]["accepted"] == 1 and res["counts"]["technical"]["accepted"] == 1
    assert res["status"] == {"ok": 3, "empty": 1} and res["graded"] == 3 and res["accuracy"] == round(2 / 3, 4)
    assert [r["id"] for r in res["rows"]] == ["T1", "T2", "T3", "T4"] and res["rows"][3]["accepted"] is False
    data, ext = c.export_bucket_batch(job.id, "csv", "bucket")
    rows = list(__import__("csv").DictReader(io.StringIO(data.decode("utf-8-sig"))))
    assert ext == "csv" and [r["ref"] for r in rows] == ["T1", "T4", "T2", "T3"]
    assert rows[0]["bucket"] == "billing" and rows[0]["correct"] == "true"
    with pytest.raises(ValueError):
        c.export_batch(job.id, "csv")                  # not a JSONL batch


# bucket batch review round 1 regressions
def test_bb_review_csv_names_and_delimiter():
    r = records.parse("a,a,a_2\n1,2,3\n", fmt="csv")                       # generated names never collide
    assert r["fields"] == ["a", "a_2", "a_2_2"] and r["records"][0] == {"a": "1", "a_2": "2", "a_2_2": "3"}
    r = records.parse("x,extra_1\n1,2,3\n", fmt="csv")
    assert r["records"][0] == {"x": "1", "extra_1": "2", "extra_1_2": "3"} and r["warnings"]
    r = records.parse("id;text\n1;hello;extra\n2;bye\n", fmt="csv")       # sniffer fails on ragged rows
    assert r["fields"][:2] == ["id", "text"] and r["records"][1] == {"id": "2", "text": "bye"} and r["warnings"]


def test_bb_review_json_shapes_and_reimport():
    one = '{"id": 1, "text": "Refund", "tags": ["billing", "urgent"]}'
    assert records.parse(one)["records"] == [{"id": 1, "text": "Refund", "tags": ["billing", "urgent"]}]
    exp = records.to_json(ROWS[:2], ["a", "b"], {"records": 2})
    back = records.parse(exp, fmt="json")
    assert [x["t"] for x in back["records"]] == ["=HYPERLINK(1)", "x"] and back["warnings"]
    again = json.loads(records.to_json([dict(ROWS[0], record=back["records"][0])], ["a", "b"], {}))
    rec = again["records"][0]
    assert rec["_openone"] == back["records"][0]["_openone"] and rec["_openone_2"]["bucket"] == "b"   # old result kept
    assert back["result_fields"] == ["_openone"]
    assert records.state_for(back["records"][0], ["*"], back["result_fields"]) == {"t": "=HYPERLINK(1)", "n": "-5"}
    big = "\n".join('{"t": "x"}' for _ in range(11))
    with pytest.raises(records.RecordError, match="more than 10"):
        records.parse(big, fmt="jsonl", limit=10)
    with pytest.raises(records.RecordError, match="more than 3"):
        records.parse("a\nb\nc\nd\n", fmt="txt", limit=3)


def test_bb_review_xml_levels_mixed_and_decl():
    r = records.parse('<items><item id="7"><subject>Refund</subject><body>Hello</body></item></items>')
    assert r["records"] == [{"@id": "7", "subject": "Refund", "body": "Hello"}] and r["record_tag"] == "items/item"
    r = records.parse("<r><t><body>Please <b>refund</b> me</body></t><t><body>x</body></t></r>")
    assert r["records"][0]["body"] == {"b": "refund", "text": "Please refund me"}
    r = records.parse("<r><t>1</t><t>2</t><note>n</note></r>")
    assert r["records"] == [{"text": "1"}, {"text": "2"}] and r["record_tag"] == "r/t"
    with pytest.raises(records.RecordError, match="DOCTYPE"):
        records.parse("<!--" + "x" * 200000 + '--><!DOCTYPE r [<!ENTITY a "b">]><r><i>&a;</i></r>')


def test_bb_review_export_collisions_and_escaping():
    rows = [dict(ROWS[0], record={"bucket": "orig", "status": "s", "=1+1": "v"})]
    out = records.to_csv(rows, ["a", "bucket"], ["bucket", "status", "=1+1"])
    head = list(__import__("csv").reader(io.StringIO(out[1:])))[0]
    assert head[:3] == ["bucket", "status", "'=1+1"]                          # original columns kept, header guarded
    assert "bucket_openone" in head and "status_openone" in head and "p_bucket" in head and "p_bucket_openone" in head
    assert len(set(head)) == len(head)
    z = zipfile.ZipFile(io.BytesIO(records.to_zip(
        [dict(ROWS[0], bucket="errors"), ROWS[1], ROWS[3]], ["errors", "b"], ["t", "n"])))
    assert z.namelist() == ["errors.csv", "below_threshold.csv", "errors_2.csv"]
    import xml.etree.ElementTree as ET
    bad = [dict(ROWS[0], record={"t": "nul\x00here\x07", "ok": "fine"})]
    root = ET.fromstring(records.to_xml(bad, ["a", "b"], {}))                 # still well-formed
    assert root[0].find("data/t").text == "nul�here�"


# bucket batch review round 2 regressions
def test_bb_review2_xml_record_level():
    assert records.parse('<items><item id="7"><tag>a</tag><tag>b</tag></item></items>')["records"] == \
        [{"@id": "7", "tag": ["a", "b"]}]
    assert records.parse('<item id="7"><subject>Refund</subject><body>Hello</body></item>')["records"] == \
        [{"@id": "7", "subject": "Refund", "body": "Hello"}]
    assert records.parse('<item><subject>Refund</subject><body>Hello</body></item>')["records"] == \
        [{"subject": "Refund", "body": "Hello"}]
    assert records.parse("<x><item><s>1</s></item><item><s>2</s></item></x>", record_tag="x")["records"] == \
        [{"item": [{"s": "1"}, {"s": "2"}]}]
    exp = records.to_xml(ROWS[:2], ["a", "b"], {"records": 2})
    back = records.parse(exp)
    assert back["record_tag"] == "openone_buckets/record/data" and back["records"] == [ROWS[0]["record"], ROWS[1]["record"]]


def test_bb_review2_result_fields_by_shape():
    r = records.parse('[{"_openone": "Refund please"}]')
    assert r["result_fields"] == [] and records.state_for(r["records"][0], ["*"], r["result_fields"]) == {"_openone": "Refund please"}
    once = records.to_json(ROWS[:1], ["a", "b"], {})
    twice = records.to_json([dict(ROWS[0], record=records.parse(once)["records"][0])], ["a", "b"], {})
    r = records.parse(twice)
    assert r["result_fields"] == ["_openone", "_openone_2"]
    assert records.state_for(r["records"][0], ["*"], r["result_fields"]) == {"t": "=HYPERLINK(1)", "n": "-5"}
    csv_text = records.to_csv(ROWS[:3], ["a", "b"], ["t", "n"])
    r = records.parse(csv_text, fmt="csv")
    assert r["result_fields"] == ["bucket", "p_bucket", "accepted", "status", "p_a", "p_b"]
    assert set(records.state_for(r["records"][1], ["*"], r["result_fields"])) == {"t", "n"}


# bucket batch review round 3 regressions
def test_bb_review3_csv_result_blocks():
    once = records.parse(records.to_csv(ROWS[:3], ["a", "b"], ["t", "n"]), fmt="csv")
    rows2 = [dict(ROWS[i], record=once["records"][i]) for i in range(3)]
    twice = records.parse(records.to_csv(rows2, ["a", "b"], once["fields"]), fmt="csv")
    assert set(twice["result_fields"]) == set(twice["fields"]) - {"t", "n"}               # both blocks found
    assert set(records.state_for(twice["records"][0], ["*"], twice["result_fields"])) == {"t", "n"}
    user = records.parse("bucket,p_bucket,accepted,status,description\nHardware,not a probability,yes,New,Refund\n")
    assert user["result_fields"] == []                                                    # ordinary columns stay
    user = records.parse("bucket,p_bucket,accepted,status,p_x,p_y\nA,0.5,true,ok,0.5,0.5\nB,0.4,false,closed,0.4,0.6\n")
    assert user["result_fields"] == []                                                    # unknown status value
    assert records.parse('[{"r": {"bucket": "a", "p_bucket": "high", "accepted": "y", "status": "ok", '
                         '"probabilities": {}}}]')["result_fields"] == []


def test_bb_review3_xml_collections_and_choice():
    rss = ('<rss version="2.0"><channel><title>News</title><link>x</link>'
           '<item><title>A</title><description>refund</description></item>'
           '<item><title>B</title><description>crash</description></item></channel></rss>')
    r = records.parse(rss)
    assert r["record_tag"] == "rss/channel/item" and [x["title"] for x in r["records"]] == ["A", "B"]
    exp = ('<export><header><created>today</created></header>'
           '<items><item><s>1</s></item><item><s>2</s></item><item><s>3</s></item></items></export>')
    r = records.parse(exp)
    assert r["record_tag"] == "export/items/item" and len(r["records"]) == 3
    order = '<order id="1"><line><sku>a</sku></line><line><sku>b</sku></line></order>'
    r = records.parse(order)
    assert [c["path"] for c in r["record_candidates"]] == ["order/line"]
    assert records.parse(order, record_tag="order")["records"] == [{"@id": "1", "line": [{"sku": "a"}, {"sku": "b"}]}]
    with pytest.raises(records.RecordError):
        records.parse(order, record_tag="nope/line")
