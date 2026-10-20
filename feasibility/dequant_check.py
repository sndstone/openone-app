"""Dequantize the FP8 package and compare it with the bf16 package: names, shapes, dtypes, weight error.

python dequant_check.py FP8_DIR BF16_DIR OUT_DIR
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from openone_core import dequant  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

fp8, bf16, out = map(Path, sys.argv[1:4])
t = time.time()
m = dequant.write_bf16_snapshot(fp8, out, source_weights_sha256=None)
print("snapshot", json.dumps(m), f"{time.time() - t:.1f}s")
with safe_open(str(out / "model.safetensors"), "pt") as a, safe_open(str(bf16 / "model.safetensors"), "pt") as b:
    ka, kb = set(a.keys()), set(b.keys())
    print("keys equal:", ka == kb, len(ka))
    worst, rel, nq = 0.0, [], 0
    for k in sorted(ka):
        x, y = a.get_tensor(k), b.get_tensor(k)
        assert x.shape == y.shape and x.dtype == y.dtype == torch.bfloat16, k
        if torch.equal(x, y):
            continue
        nq += 1
        d = (x.float() - y.float()).norm() / y.float().norm()
        rel.append(d.item()); worst = max(worst, d.item())
    print(f"tensors that differ from bf16: {nq}; relative L2 error mean {sum(rel) / max(1, len(rel)):.4f}, max {worst:.4f}")
cfg_a = json.loads((out / "config.json").read_text()); cfg_b = json.loads((bf16 / "config.json").read_text())
print("config equal to bf16:", cfg_a == cfg_b)
