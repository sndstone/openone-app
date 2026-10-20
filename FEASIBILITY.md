# OpenOne app: backend feasibility (2026-10-20)

## What one decision needs
- **Prompt.** `decider.systemone.render_state` / `render_question`, then `decider.prompt.build`:
  - plain layout, options in caller order (FixedOrder), up to 255 options;
  - the state is capped at 2,048 tokens of `"Context:\n" + state`.
  - These are the exact settings of `eval_decider_rows.py`, which the `openone.json` numbers were fitted with.
- **Scoring.** One forward pass. Read the hidden state at the final `" ("` token and project it onto the option-letter
  rows of the tied `lm_head`. This equals vLLM's next-token log-probs restricted to the letters (`eval_vllm.py`).
- **Temperature and gate.** `softmax(letter_logits / T)` with `T` = `openone.json` `usecase_temperatures`:
  - `usecase_toolcall` for allow/ask/block questions;
  - `usecase_bucket` for everything else;
  - then the gate thresholds from `openone.json` (`gate.auto_run_if_p_allow_at_least`, `gate.alarm_if_p_block_at_least`).
- **Dependencies.** Gated DeltaNet needs no special kernels. Stock `transformers` 5.x falls back to
  `torch_chunk_gated_delta_rule` and a reference causal conv; `fla` and `causal_conv1d` are optional.

## Measured (same 3 synthetic decisions, bf16 package `03db485d…`)

| Platform / backend | Status | Load | Gate decision, ~420 tokens | Numerics vs CUDA |
| --- | --- | --- | --- | --- |
| Linux, CUDA, RTX 4060 Ti, torch 2.13 + transformers 5.17 (vLLM image) | **tested** | 4.1 s | 67 ms | reference |
| Windows 11, CPU (Ryzen 9 8945HX), torch 2.14.1+cpu, transformers 5.20.2, bf16 | **tested** | 3.4 s | 757 ms | log-probs equal to 4 dp (one value 0.06 off at −7.5) |
| Windows 11, Intel Arc B580 (XPU), torch 2.14.1+xpu, bf16 | **tested** | 5.1 s | 122 ms | log-probs within 0.06 |
| Windows 11, Intel Arc B580, fp16 (stand-in for the MPS dtype) | **tested** | 5.2 s | 100 ms | within 0.04; no overflow |
| macOS, Apple Silicon, MPS | **not tested (no Mac here)** | – | – | – |
| Windows, NVIDIA CUDA | **not tested (this PC has no NVIDIA GPU)** | – | – | – |

Notes:
- This PC has an **Intel Arc B580 and an AMD 610M iGPU, no NVIDIA GPU**, so "local GPU" on this machine means Intel XPU.
- macOS should work. torch on arm64 ships MPS, the transformers fallback kernels are pure torch, and the package's own
  `decider/mps_ops.py` adds MPS kernels (optional MLX). The Decider serves MPS in fp16; fp16 was checked above as a
  stand-in. Expect roughly 0.1–0.3 s per decision on M-series chips (an estimate, not measured).
- **FP8 cannot run locally on Windows or macOS.** It is a compressed-tensors checkpoint for vLLM 0.31, which needs
  Linux + CUDA on Ada/Hopper/Blackwell. It runs through the remote gate service (Linux GPU box), or on a Windows PC
  with an NVIDIA GPU through Docker Desktop/WSL2 running the same service image.
- The bf16 `openone.json` values (T_U2 1.28, θ 0.4532) were fitted on the **torch** path, which is what the local app
  runs. Under vLLM the bundle README gives T_U2 1.27, θ 0.4593. The FP8 `openone.json` is fitted under vLLM. The app
  always reads the values from the package or server; nothing is hard-coded.

## What works, plainly

| Use | Package | Backend | Verdict |
| --- | --- | --- | --- |
| Windows, any PC | bf16 | local CPU | works, ~0.8 s/decision (tested) |
| Windows, Intel Arc | bf16 | local XPU | works, ~0.12 s/decision (tested) |
| Windows, NVIDIA | bf16 | local CUDA | expected to work like the Linux CUDA test; not tested on Windows |
| macOS, Apple Silicon | bf16 | local MPS | expected to work; built, not tested |
| macOS, Intel | bf16 | local CPU | expected to work (slow); optional build |
| Any OS | FP8 | remote vLLM gate service on a Linux NVIDIA box | works (tested against ai-cluster, see README) |
| Any OS | bf16 | remote gate service | works, but then use the vLLM-fitted numbers (the server reports what it applies) |
| Windows + NVIDIA | FP8 | Docker Desktop/WSL2 running the gate service image | possible; not tested here |
| Windows/macOS | FP8 | local, native | **not possible** |
| Any OS, any local device | FP8 | local, dequantized to bf16 (fallback) | works (tested on CPU and XPU); same speed as bf16; own fitted thresholds (vLLM) |

## FP8 fallback on hardware without FP8 (added 2026-10-20)
An FP8-only install still runs locally by dequantizing to bf16, using `core/openone_core/dequant.py`:
- **Method.** `w = (w_fp8.float() * weight_scale.float()).bfloat16()` for the 112 `F8_E4M3` weights (scales are bf16
  `[out, 1]`, per channel, symmetric). The `weight_scale` keys are dropped and `quantization_config` is removed.
- **Checked against the bf16 package:**
  - same 320 tensor names, shapes and dtypes;
  - `config.json` is identical once `quantization_config` is removed;
  - tokenizer, chat template and `decider_config.json` are byte-identical;
  - the file is the same size (3,763,692,048 bytes).
- **Cost.** 6.6 s on this PC. The output is deterministic: dequantized weights sha256 `a14fc531…aba3d`.
- **Probe on XPU.** It agrees with bf16 to about 0.01 in probability on the 3 probe cases. On the block case,
  p(block) is 0.941 against bf16's 0.949.
- **Finding, which also affects true FP8.** Besides the 112 quantized weights, the FP8 checkpoint's **61 RMSNorm weights
  differ from bf16** (input/post-attention/q/k/final norms, relative error 1–2 %).
  - They are exactly `bf16(1 + w) − 1` of the bf16 weights. Qwen3.5's offset norms took a bf16 round trip when
    llm-compressor saved the file.
  - These weights are not quantized, so the dequantizer copies them unchanged. The fallback therefore reproduces the FP8
    checkpoint exactly.
  - This round trip is a possible contributor to the FP8 drift (RT3 R6) and is cheap to fix at export time.
- **Validated on the cluster (2026-10-20).** The cluster reference `quant/dequant_fp8.py` produces byte-identical
  weights (sha256 `a14fc531…`). The largest weight difference from bf16 is 0.026 (FP8 rounding).
- **Fitted numbers, under vLLM bf16:** T bucket 1.34, T tool call 1.28, auto-run 0.80, alarm 0.4133.
  - Quality on real traffic: mean |Δp| vs bf16 0.0065–0.0078 (FP8 0.011–0.015), about half of FP8's argmax flips.
  - False alarms: bf16 108, fallback 114, FP8 120. Block leaks: bf16 10, fallback 11, FP8 9 (of 1,089).
