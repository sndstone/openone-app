"""FP8 (compressed-tensors FP8_DYNAMIC, per-channel symmetric weights) -> bf16 snapshot, for hardware without FP8.

    w_bf16 = (w_fp8.float() * weight_scale.float()).bfloat16()        weight_scale: bf16 [out, 1]

The *.weight_scale tensors are dropped, every other tensor is already bf16 and is copied unchanged, and
quantization_config is removed from config.json. The result has the bf16 package's tensor names and shapes (checked on
the v9 r16 bundles: 320 tensors). Input activations were dynamic per-token FP8 at run time; in bf16 compute they are
simply not quantized. Needs torch and safetensors (runs in the inference runtime, not in the UI process).
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

COPY = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "decider_config.json")
MARKER_VERSION = 2          # 2: the snapshot carries the fallback's own openone.json
MARKER = "openone_fallback.json"
METHOD = "w_bf16 = (w_fp8.float() * weight_scale.float()).bfloat16(); weight_scale keys dropped; quantization_config removed"


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dequantize_state_dict(fp8_dir, progress=None):
    """-> {name: bf16 tensor}. Raises ValueError on any tensor layout this function does not know."""
    import torch
    from safetensors import safe_open
    src = Path(fp8_dir) / "model.safetensors"
    out = {}
    with safe_open(str(src), framework="pt") as f:
        keys = list(f.keys())
        scales = {k for k in keys if k.endswith(".weight_scale")}
        other_q = [k for k in keys if k.endswith(("_zero_point", ".input_scale", ".weight_global_scale"))]
        if other_q:
            raise ValueError(f"unsupported quantization tensors (only per-channel FP8 weights are handled): {other_q[:3]}")
        for i, k in enumerate(keys):
            if k in scales:
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                sk = k[:-len("weight")] + "weight_scale"
                if not k.endswith(".weight") or sk not in scales:
                    raise ValueError(f"FP8 tensor {k} has no {sk}")
                s = f.get_tensor(sk)
                if list(s.shape) != [t.shape[0], 1]:
                    raise ValueError(f"{sk} has shape {list(s.shape)}, expected [{t.shape[0]}, 1] (per output channel)")
                t = (t.to(torch.float32) * s.to(torch.float32)).to(torch.bfloat16)
            elif t.dtype != torch.bfloat16:
                raise ValueError(f"{k} is {t.dtype}; expected bf16 or float8_e4m3fn")
            out[k] = t.contiguous()
            if progress:
                progress(i + 1, len(keys), k)
        orphan = [s for s in scales if s[:-len("weight_scale")] + "weight" not in out]
        if orphan:
            raise ValueError(f"weight_scale without a weight: {orphan[:3]}")
    return out


def write_bf16_snapshot(fp8_dir, out_dir, source_weights_sha256=None, progress=None):
    """Write a loadable bf16 snapshot of an FP8 package into out_dir (replaced atomically). -> marker dict."""
    from safetensors.torch import save_file
    t0 = time.time()
    fp8_dir, out_dir = Path(fp8_dir), Path(out_dir)
    part = out_dir.with_name(f"{out_dir.name}.partial-{os.getpid()}")
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    try:
        sd = dequantize_state_dict(fp8_dir, progress)
        save_file(sd, str(part / "model.safetensors"), metadata={"format": "pt"})
        n = len(sd)
        del sd
        cfg = json.loads((fp8_dir / "config.json").read_text(encoding="utf-8"))
        cfg.pop("quantization_config", None)
        (part / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        for name in COPY:
            if (fp8_dir / name).is_file():
                shutil.copy2(fp8_dir / name, part / name)
        from .policy import fallback_openone
        fo = fallback_openone(json.loads((fp8_dir / "openone.json").read_text(encoding="utf-8")), source_weights_sha256)
        (part / "openone.json").write_text(json.dumps(fo, indent=2) + "\n", encoding="utf-8")
        marker = {"kind": "fp8_dequantized_bf16", "version": MARKER_VERSION, "source_package": fp8_dir.name,
                  "source_weights_sha256": source_weights_sha256, "tensors": n, "method": METHOD,
                  "weights_sha256": _sha(part / "model.safetensors"),
                  "weights_bytes": (part / "model.safetensors").stat().st_size,
                  "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)}
        (part / MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    try:
        part.rename(out_dir)
    except OSError:                                   # another process finished the same snapshot first: use ours in place
        return dict(marker, path=str(part))
    return dict(marker, path=str(out_dir))


def read_marker(out_dir, source_weights_sha256=None):
    """The marker of a cached snapshot if it is complete and was made from these source weights, else None."""
    p = Path(out_dir) / MARKER
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    w = Path(out_dir) / "model.safetensors"
    if m.get("version") != MARKER_VERSION or not w.is_file() or w.stat().st_size != m.get("weights_bytes"):
        return None
    if source_weights_sha256 and m.get("source_weights_sha256") != source_weights_sha256:
        return None
    return m
