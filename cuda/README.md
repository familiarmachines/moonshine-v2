# moonshine-cuda

Standalone C++/CUDA inference engine for
[`UsefulSensors/moonshine-streaming-tiny`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny),
loading weights directly from HuggingFace safetensors. **No PyTorch, no
Transformers, no ONNX Runtime** — the only dependencies are CUDA Toolkit
(cuBLAS) for the GPU path, and nothing at all for the CPU reference.

See `ARCHITECTURE.md` for the exact model spec, derived from the
`transformers` reference implementation (`ref-*.py`) and cross-checked
against this repo's ORT pipeline.

## Layout

| File | Purpose |
| ---- | ------- |
| `safetensors.{h,cpp}` | mmap safetensors reader (fp32) |
| `tokenizer.{h,cpp}` | BinTokenizer-format decoder |
| `wav.{h,cpp}` | 16 kHz mono PCM16 WAV loader |
| `model.{h,cpp}` | weight mapping + CPU fp32 reference forward pass |
| `model_cuda.{h,cu}` | CUDA fp32 engine (cuBLAS + custom kernels) and `GpuStream` incremental API |
| `main-ref.cpp` | CPU reference CLI (`moonshine-ref`) |
| `main-cuda.cpp` | one-shot GPU CLI (`moonshine-cuda`) |
| `main-bench.cpp` | streaming benchmark (`moonshine-bench`) |

## Setup

```bash
# Weights (one-time, ~176 MB)
mkdir -p models/streaming-tiny && cd models/streaming-tiny
for f in config.json tokenizer.json model.safetensors; do
  curl -sL "https://huggingface.co/UsefulSensors/moonshine-streaming-tiny/resolve/main/$f" -O
done
# tokenizer.json -> tokenizer.bin (decodes <0xNN> byte-fallback tokens)
python3 ../../convert_hf_tokenizer.py tokenizer.json tokenizer.bin
cd ../..

# Build (CUDA target needs nvcc on PATH; CPU reference builds regardless)
mkdir -p build && cd build
PATH=/usr/local/cuda-12.6/bin:$PATH cmake .. && cmake --build . -j6
```

Note: `scripts/convert_tokenizer.py` in the repo root keeps `<0xNN>`
byte-fallback tokens literal; `convert_hf_tokenizer.py` here decodes them to
raw bytes, matching the shipped `test-assets/*/tokenizer.bin` byte-for-byte
(except one stripped U+2028 token).

## Run

```bash
./moonshine-ref  ../models/streaming-tiny ../../test-assets/beckett.wav   # CPU
./moonshine-cuda ../models/streaming-tiny ../../test-assets/beckett.wav   # GPU
./moonshine-bench ../models/streaming-tiny ../../test-assets/two_cities_16k.wav \
    --segment-duration 10 --update-interval 0.5 --chunk-ms 80
```

`moonshine-bench` mirrors `core/build/benchmark` semantics minus VAD: audio is
fed in 80 ms chunks, segmented at a fixed duration (a VAD would do this
adaptively), the encoder runs incrementally while "speech" streams in
(96-frame left-context window, 16-frame lookahead holdback — same as the ORT
pipeline), interim transcripts refresh at the update interval, and the
reported latency is the segment-end flush + final decode.

## Results (Jetson Orin, 8 SM, cc 8.7, fp32 — measured 2026-06-12)

| Engine | beckett.wav (10 s) | two_cities (44 s) |
| ------ | ------------------ | ------------------ |
| CPU reference (1 thread) | 7.36 s (74% RT) | — |
| CUDA one-shot | 0.141 s (1.4% RT) | — |
| CUDA streaming bench | — | 12.9% of audio duration, **112 ms** avg end-of-segment latency |
| `core/build/benchmark` (ORT CPU int8, VAD + speaker ID) | — | 32.9% of audio duration, 171 ms avg latency |

Caveats for the comparison row: the ORT pipeline includes VAD + speaker
embedding + clustering, runs int8-quantized models on CPU, and segments
adaptively; this engine is fp32 GPU, model-only, fixed segmentation.

Transcripts are identical between CPU reference and CUDA on the test WAVs,
and match the ORT pipeline output near-verbatim (boundary words differ where
fixed segmentation cuts mid-phrase).

## Encoder kernel findings (2026-06-12)

`model_cuda.cu` contains a fused cooperative megakernel (`k_encoder_fused`,
opt-in via `MOONSHINE_FUSED=1`) that runs all 6 encoder layers in one launch:
frame-partitioned blocks, fp16-stored/fp32-math weights, FT=16 register tiles,
phase-locked weight streaming with 2 grid syncs per layer. It is numerically
correct (transcripts match) but currently ~2x SLOWER than the unfused cuBLAS
path on the steady-state streaming window (~10 ms vs 4.9 ms for len=137), so
the cuBLAS path stays the default.

Measured facts (clock64 stage profiling, `MOONSHINE_ENC_PROF=1` with the
`enc-only` harness; cooperative launch 12 us, grid.sync ~4 us each):

- Per layer: QKV 360 us | attention+O 300 us | MLP 1280 us -> ~10 ms/call.
- fp16 weights changed nothing -> NOT bandwidth-bound (L2 is 2 MB, but
  phase-locked streams fit).
- The kernel is instruction-issue bound: each FMA pairs with one shared-mem
  load (50% issue ceiling) x ~62% thread utilization in 320-wide loops x 56%
  tile padding (ft=9 of FT=16) ~= 220 GFLOPS, which matches measurement.

Roofline: the encode tick is ~1.9 GFLOP. Orin Nano fp32 peak is ~1.3 TFLOPS,
so even a perfect fp32 kernel cannot go below ~1.5 ms/call, and cuBLAS already
runs at ~30% of peak. The real lever is the tensor cores (~4 TFLOPS fp16-in /
fp32-acc): an `mma.sync` (m16n8k16) rewrite with fp16 activations staged in
shared memory targets ~1-1.5 ms/call, a ~3-4x win. That is the planned v2.

### v2 update: wmma tensor cores landed (2026-06-12, later)

The GEMM stages (QKV, O-proj, fc1, fc2) now run on tensor cores via
`nvcuda::wmma` m16n16k16 (fp16 operands, fp32 accumulate; fp16 activation
tiles in shared, fp32 residual stream in global). Transcripts unchanged.
Clock-invariant cycle profile (len=137): QKV 360->173, attn+O 300->214,
MLP 1280->470 per layer; whole kernel 2.3x faster.

Pinned-clock A/B (`sudo jetson_clocks`, 1020 MHz, 100 calls): fused wmma
5.80 ms/call vs unfused cuBLAS 5.61 ms/call — parity.

Overhead attribution (`MOONSHINE_ENC_ATTR=1`, per-phase sync timing): window
memcpy ~30us, emit (stable rows + pos_emb) ~34us, kernel ~5.17ms wall. So
there is NO meaningful host-side overhead to recover — the earlier "1.4ms
gap" was an artifact of comparing block-0 clock64 cycle totals (which miss
launch ramp/tail, ~0.7ms) against wall time. All remaining headroom is inside
the kernel: attention (~25% of cycles, uncoalesced K/V gathers) and wmma GEMM
pipelining (no cp.async double-buffering, A-fragments reloaded per n-tile).
As of the fc2 register-promotion commit the fused kernel leads cuBLAS
(4.67 vs 5.3-5.6 ms/call at pinned 1020 MHz) and is now the DEFAULT.
`MOONSHINE_NO_FUSED=1` forces the unfused cuBLAS path; the fallback is also
automatic where cooperative launch is unavailable. Verified before flipping:
fused and unfused produce byte-identical transcripts on beckett/intent/
two_cities one-shot (multi-tile long windows) and on the streaming benchmark
(short windows incl. remainder tiles).

Two infrastructure gotchas worth knowing:
- **CMake was silently building for compute_52** (`enable_language(CUDA)`
  pins the toolkit default before a later `if(NOT DEFINED
  CMAKE_CUDA_ARCHITECTURES)` can run). Every measurement before the fix went
  through driver JIT from generic PTX, and wmma would not compile. Now set to
  87 before `enable_language`.
- **Jetson DVFS confounds wall-clock numbers**: the GPU governor sat at the
  306 MHz floor (max 1020) through entire benchmark runs on this box.
  Cross-session comparisons need `sudo jetson_clocks` (root) or cycle-based
  profiling (`MOONSHINE_ENC_PROF=1`, which reports clock64 counts).

Next per the task plan: attention is now ~30% of the kernel (stage K/V tiles
to shared, or wmma-ize scores); then counter-based sync replacing grid.sync,
cp.async weight pipelining, and the decode-loop megakernel/graphs.

## Status / next steps

- fp32 only; fp16 tensor-core path and CUDA-graph capture of the decode loop
  (~125 kernel launches/token) are the obvious next optimizations.
- Single stream per `GpuModel` (decode scratch is shared).
- Greedy decode only; no repetition heuristics beyond the
  `max_tokens_per_second` cap (parity with core).
- Larger checkpoints (`-small`, `-medium`) differ only in dims/layer counts —
  the loader hardcodes tiny's config; generalizing means reading
  `config.json` instead of `ModelConfig` defaults.
