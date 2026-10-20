"""Inference-path smoke test without the private weights: a tiny randomly initialised Qwen3.5 (same layer pattern:
gated-delta + full attention) with the real tokenizer, run through the worker, an FP8 dequant round trip and the
pipeline. The answers are random; the test checks that every step runs on this platform/device and that every
response validates.

python tests/tiny_model_smoke.py --tokenizer-dir DIR | --hf-repo Mapika/decider-2b  [--device auto|cpu|mps|xpu|cuda]
Needs torch, transformers, safetensors and jsonschema in the running Python (it also runs the worker with it).
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "core")]
from openone_app.backends import LocalWorker  # noqa: E402
from openone_core import package, pipeline, policy  # noqa: E402

TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
CONFIG = {
    "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", "attention_bias": False, "attention_dropout": 0.0,
    "attn_output_gate": True, "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 64, "hidden_act": "silu",
    "hidden_size": 128, "initializer_range": 0.02, "intermediate_size": 256,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
    "linear_value_head_dim": 32, "mamba_ssm_dtype": "float32", "max_position_embeddings": 8192, "mlp_only_layers": [],
    "mtp_num_hidden_layers": 0, "num_attention_heads": 2, "num_hidden_layers": 4, "num_key_value_heads": 1,
    "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-6,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [3, 3, 2], "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000, "rope_type": "default"},
    "tie_word_embeddings": True, "use_cache": True, "vocab_size": 248320, "dtype": "bfloat16",
}
OPENONE = {"usecase_temperatures": {"usecase_bucket": 1.34, "usecase_toolcall": 1.28},
           "gate": {"auto_run_if_p_allow_at_least": 0.8, "alarm_if_p_block_at_least": 0.4532}}


def sums(d):
    lines = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(d).as_posix()}")
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make(tmp, tok_dir):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    bf16 = tmp / "tiny-bf16"
    bf16.mkdir()
    (bf16 / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(bf16)).to(torch.bfloat16)
    m.save_pretrained(bf16, safe_serialization=True)
    for f in TOK_FILES:
        if (tok_dir / f).is_file():
            shutil.copy2(tok_dir / f, bf16 / f)
    (bf16 / "openone.json").write_text(json.dumps(OPENONE), encoding="utf-8")
    (bf16 / "decider_config.json").write_text(json.dumps({"layout": "plain"}), encoding="utf-8")
    sums(bf16)
    # an FP8 twin in the compressed-tensors layout (per-channel e4m3 weights + bf16 [out, 1] scales)
    from safetensors.torch import load_file, save_file
    fp8 = tmp / "tiny-fp8"
    shutil.copytree(bf16, fp8)
    sd = load_file(str(bf16 / "model.safetensors"))
    out = {}
    for k, w in sd.items():
        if k.endswith("proj.weight") and w.dim() == 2:
            s = (w.float().abs().amax(1, keepdim=True).clamp(min=1e-8) / 448).to(torch.bfloat16)
            out[k] = (w.float() / s.float()).to(torch.float8_e4m3fn)
            out[k[:-len("weight")] + "weight_scale"] = s
        else:
            out[k] = w
    save_file(out, str(fp8 / "model.safetensors"), metadata={"format": "pt"})
    cfg = dict(CONFIG, quantization_config={"format": "float-quantized"})
    (fp8 / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (fp8 / "openone_quant.json").write_text("{}", encoding="utf-8")
    sums(fp8)
    return bf16, fp8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--hf-repo")
    ap.add_argument("--hf-revision", default="533964dae8be954c5b5e19fa4948e48408094c1e")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="openone-tiny-"))
    if a.tokenizer_dir:
        tok_dir = Path(a.tokenizer_dir)
    else:
        from huggingface_hub import hf_hub_download
        tok_dir = tmp / "tok"
        tok_dir.mkdir()
        for f in TOK_FILES:
            try:
                shutil.copy2(hf_hub_download(a.hf_repo, f, revision=a.hf_revision), tok_dir / f)
            except Exception as e:                                  # noqa: BLE001 - chat template is optional
                print("  (skip", f, e.__class__.__name__ + ")")
    bf16, fp8 = make(tmp, tok_dir)
    fails = []
    req = {"state": {"agent": "claude-code", "os": "macos", "shell": "zsh", "workspace_root": "/Users/dev/demo", "cwd": "/Users/dev/demo",
                     "tool": "Bash", "input": {"command": "npm test"}, "user_task": "run tests"},
           "questions": {"v": {"type": "choice", "instructions": policy.GATE_INSTRUCTIONS, "criteria": policy.DEFAULT_POLICY},
                         "b": {"instructions": "Pick a bucket.", "criteria": [f"bucket{i}" for i in range(12)]}}}
    for d, mode in ((bf16, "native"), (fp8, "fp8_dequantized_bf16")):
        info = package.read_package(d)
        w = LocalWorker(sys.executable, tmp / "worker.log")
        w.start(timeout=120)
        try:
            li = w.load(model_dir=str(d), mode=mode, device=a.device,
                        dequant={"cache_dir": str(tmp / "dq"), "keep": False, "source_weights_sha256": info["weights_sha256"]})
            settings, warns = policy.resolve_settings(info, mode)
            ctx = {"model": {"name": info["name"], "format": info["format"], "mode": mode, "weights_sha256": info["weights_sha256"]},
                   "backend": {"kind": "local", "device": li["device"], "runtime": li["runtime"]}, "settings": settings, "warnings": warns}
            out = pipeline.decide(req, w.score, ctx, want_raw=True)
            ok = out["response_validation"]["valid"]
            print(f"{mode:22s} device={li['device']} dtype={li['dtype']} warmup={li['warmup_ms']}ms valid={ok} "
                  f"verdict={out['response']['answers']['v']['verdict']['action']} bucket={out['response']['answers']['b']['choice']}")
            if not ok:
                fails.append(mode)
                print(json.dumps(out["response_validation"], indent=1))
        except Exception as e:                                      # noqa: BLE001
            fails.append(f"{mode}: {e}")
            print("FAIL", mode, e)
            print((tmp / "worker.log").read_text(errors="replace")[-3000:])
        finally:
            w.stop()
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
