"""Letter-logit scorers. Each returns, per prompt item, the log-softmax over the item's option letters at the answer slot
(T = 1). Temperature and thresholds are applied later by openone_core.pipeline, identically for every backend.

TorchScorer: transformers + torch on cuda / xpu / mps / cpu (bf16 packages; FP8 packages are dequantized to bf16 first).
VllmScorer:  vLLM 0.31 (FP8 compressed-tensors or bf16) on Linux + CUDA; letter-restricted next-token log-probs, as
             export/eval_vllm.py.
"""
import os
import time

from .policy import log_softmax


def pick_device(requested="auto"):
    import torch
    req = (requested or "auto").lower()
    have = {"cuda": torch.cuda.is_available(),
            "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
            "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            "cpu": True}
    if req == "auto":
        return next(d for d in ("cuda", "xpu", "mps", "cpu") if have[d])
    kind = req.split(":")[0]
    if kind not in have:
        raise ValueError(f"unknown device {requested!r}; use auto, cuda, xpu, mps or cpu")
    if not have[kind]:
        raise ValueError(f"device {requested!r} is not available in this runtime")
    return req


def devices():
    """What this torch build can use, with names."""
    import torch
    out = [{"device": "cpu", "name": "CPU", "threads": torch.get_num_threads()}]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append({"device": f"cuda:{i}" if i else "cuda", "name": p.name, "memory_gb": round(p.total_memory / 2**30, 1),
                        "capability": f"{p.major}.{p.minor}", "fp8_native": (p.major, p.minor) >= (8, 9)})
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        out.append({"device": "xpu", "name": torch.xpu.get_device_name(0)})
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        out.append({"device": "mps", "name": "Apple GPU (Metal)"})
    return out


class TorchScorer:
    def __init__(self, model_dir, device="auto", dtype="auto", threads=None, max_batch=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        t0 = time.time()
        self.device = pick_device(device)
        kind = self.device.split(":")[0]
        if dtype in (None, "auto"):
            dtype = "fp16" if kind == "mps" else "bf16"      # as decider.serve.resolve_device
        self.dtype_name = dtype
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        if threads:
            torch.set_num_threads(int(threads))
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.lm = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt).to(self.device).eval()
        self.W = self.lm.lm_head.weight.detach()
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.max_batch = max_batch or (1 if kind == "cpu" else 8)
        self.load_seconds = round(time.time() - t0, 2)
        self.runtime = f"torch {torch.__version__}, transformers {__import__('transformers').__version__}"

    def _sync(self):
        kind = self.device.split(":")[0]
        if kind == "cuda":
            self.torch.cuda.synchronize()
        elif kind == "xpu":
            self.torch.xpu.synchronize()
        elif kind == "mps":
            self.torch.mps.synchronize()

    def score(self, items):
        """items: prompting.Builder.build outputs. -> list of (log-softmax over letters, forward ms per item)."""
        torch = self.torch
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
        res = [None] * len(items)
        k = 0
        while k < len(order):
            chunk = order[k:k + self.max_batch]
            T = max(len(items[i]["ids"]) for i in chunk)
            while len(chunk) > 1 and T * len(chunk) > 16384:
                chunk = chunk[:-1]
            k += len(chunk)
            ids = torch.full((len(chunk), T), self.pad, dtype=torch.long)
            for b, i in enumerate(chunk):                    # right padding: causal layers never see later pads
                ids[b, :len(items[i]["ids"])] = torch.tensor(items[i]["ids"])
            t = time.time()
            with torch.no_grad():
                h = self.lm.model(input_ids=ids.to(self.device), use_cache=False).last_hidden_state
                rows = torch.tensor([items[i]["slot"] for i in chunk], device=h.device)
                hs = h[torch.arange(len(chunk), device=h.device), rows]
                outs = []
                for b, i in enumerate(chunk):
                    w = self.W[torch.tensor(items[i]["letter_ids"], device=h.device)]
                    outs.append(torch.nn.functional.linear(hs[b], w).float().cpu())
            self._sync()
            ms = (time.time() - t) * 1000 / len(chunk)
            for b, i in enumerate(chunk):
                res[i] = (log_softmax(outs[b].double().tolist()), round(ms, 2))
        return res


class VllmScorer:
    def __init__(self, model_dir, gpu_memory_utilization=None, max_model_len=8192):
        from transformers import AutoTokenizer
        from vllm import LLM
        import vllm
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        mem = float(gpu_memory_utilization or os.environ.get("OPENONE_VLLM_MEM", "0.45"))
        self.llm = LLM(model=model_dir, tokenizer=model_dir, dtype="bfloat16", max_model_len=max_model_len, max_logprobs=256,
                       logprobs_mode="processed_logprobs", gpu_memory_utilization=mem, enable_prefix_caching=True, seed=0)
        self.max_model_len = max_model_len
        self.device = "cuda"
        self.dtype_name = "vllm"
        self.load_seconds = round(time.time() - t0, 2)
        self.runtime = f"vLLM {vllm.__version__}"

    def score(self, items):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        for it in items:
            if len(it["ids"]) >= self.max_model_len:
                raise ValueError(f"prompt has {len(it['ids'])} tokens; the server allows {self.max_model_len - 1}")
        prompts = [TokensPrompt(prompt_token_ids=it["ids"]) for it in items]
        params = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(it["letter_ids"]),
                                 allowed_token_ids=list(it["letter_ids"])) for it in items]
        t = time.time()
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        ms = (time.time() - t) * 1000 / max(1, len(items))
        res = []
        for it, out in zip(items, outs):
            lp = out.outputs[0].logprobs[0]
            res.append((log_softmax([lp[t].logprob for t in it["letter_ids"]]), round(ms, 2)))
        return res
