"""Unit tests for openone_core (no model needed; dequant tests need torch + safetensors and are skipped otherwise).

python -m pytest tests/test_core.py -q
"""
import copy
import hashlib
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
