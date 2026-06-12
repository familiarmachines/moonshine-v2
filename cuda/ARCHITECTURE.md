# moonshine-streaming-tiny — exact inference spec

Sources: `UsefulSensors/moonshine-streaming-tiny` (config.json + safetensors header),
`transformers` reference (`modular_moonshine_streaming.py`, `modeling_moonshine.py`,
fetched 2026-06-12, saved as `ref-*.py`). Cross-checked against this repo's ORT
pipeline (`core/moonshine-streaming-model.cpp`, `test-assets/tiny-streaming-en/streaming_config.json`).

Dims: hidden 320, 6 enc + 6 dec layers, 8 heads, head_dim 40, vocab 32768,
bos=1, eos=2, fp32 weights (44.1M params).

## Frontend (`model.encoder.embedder`), per 80-sample frame @16 kHz (200 Hz)

1. Frame audio: `x = audio.reshape(T200, 80)` (drop remainder <80 samples; streaming
   keeps a 79-sample carry buffer — see ORT `sample_buffer`).
2. Per-frame CMVN: `m = mean(x); c = x - m; x = c / sqrt(mean(c^2) + 1e-6)`.
3. Asinh compression: `x = asinh(exp(log_k) * x)`, `log_k` scalar (`embedder.comp.log_k`).
4. `h = silu(linear(x))`, `linear.weight [320, 80]`, no bias → `[T200, 320]`.
5. `conv1`: causal Conv1d (left-pad 4, zero history at t=0), 320→640, k=5, stride 2,
   bias → `silu` → 100 Hz.
6. `conv2`: causal Conv1d (left-pad 4), 640→320, k=5, stride 2, bias, **no activation**
   → features `[T50, 320]` at 50 Hz (20 ms per feature frame).

Streaming state (must match ORT semantics): 79-sample remainder buffer, conv1 input
history `[320,4]`, conv2 input history `[640,4]`.

## Encoder (6 layers, pre-norm, no positional encoding)

Per layer:
- `x += self_attn(LNu(x))` — plain MHA, scale `40^-0.5`, **sliding-window mask**
  per layer; `(left, right)` from config `sliding_windows =
  [(16,4),(16,4),(16,0),(16,0),(16,4),(16,4)]`:
  attend iff `0 <= q-kv < left` OR `0 < kv-q < right`.
- `x += mlp(LNu(x))` — fc1 (320→1280, bias) → GELU (exact erf) → fc2 (1280→320, bias).

`LNu` = unit-offset norm: `LayerNorm(eps=1e-5, affine=False)(x) * (gamma + 1)`,
weights named `*.gamma` (stored centered at 0). `final_norm` (LNu) after last layer.

Streaming emission (ORT behavior): only frames `< total - 16` are "stable"
(`total_lookahead=16` from streaming_config.json; config-derived lookahead sum is 12 —
core uses 16, keep 16 for parity). Re-encode with `16*6 = 96` frames of left context,
emit only new stable frames.

## Adapter == `decoder.pos_emb`

`memory[t] = enc_out[t] + pos_emb.weight[t_abs]`, `pos_emb [4096, 320]` (learned,
absolute over the whole utterance — streaming keeps running `pos_offset`).
`proj` = Identity (enc dim == dec dim).

## Decoder (6 layers, pre-norm; norms are `LayerNorm(bias=False)` WITH affine weight, eps 1e-5)

Token embed: `embed_tokens [32768, 320]`, **no scaling**.

Per layer:
- `x += self_attn(input_layernorm(x))` — causal, KV-cached, **RoPE** on q,k.
- `x += encoder_attn(post_attention_layernorm(x))` — K,V from memory, computed once
  and cached (== ORT `cross_kv` session); **no RoPE** on cross-attention.
- `x += mlp(final_layernorm(x))` — fc1 (320→2560, bias) → split halves `(val, gate)`
  → `silu(gate) * val` → fc2 (1280→320, bias).
  NOTE the order: FIRST half is the value, SECOND half is the gate.

Final: `norm` (LayerNorm bias=False) → logits = `proj_out.weight @ h` (untied, no bias).

## RoPE (decoder self-attention only)

Partial + interleaved: rotary_dim = 0.8 * 40 = 32 dims, theta 10000.
`inv_freq[j] = 10000^(-2j/32)`, j = 0..15. For position p, pair j applied to dims
(2j, 2j+1):
```
q'[2j]   = q[2j]*cos(p*f_j) - q[2j+1]*sin(p*f_j)
q'[2j+1] = q[2j+1]*cos(p*f_j) + q[2j]*sin(p*f_j)
```
Dims 32..39 pass through unrotated.

## Generation

Greedy argmax, start token 1, stop at 2; cap tokens at
`ceil(duration_sec * max_tokens_per_second)` (core uses 6.5, max 256) for parity
with the ORT benchmark.

## Validation plan

System-level: transcripts on `test-assets/beckett.wav`, `two_cities_16k.wav` vs the
known-good ORT pipeline output (`core/build/benchmark`, quantized model — expect
near-identical but not bit-identical text). Tensor-level (debug): run individual
`.ort` sessions from `test-assets/tiny-streaming-en` to dump
features/encoded/memory/logits; quantization makes these approximate references only.
