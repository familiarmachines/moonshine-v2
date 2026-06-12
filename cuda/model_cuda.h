// CUDA (fp32) implementation of moonshine-streaming-tiny inference.
// Mirrors the CPU reference in model.h; weights are uploaded once at load.
#ifndef MOONSHINE_CUDA_MODEL_CUDA_H
#define MOONSHINE_CUDA_MODEL_CUDA_H

#include <vector>

#include "model.h"

struct GpuModelImpl;

struct GpuModel {
  const ModelConfig cfg;
  GpuModelImpl *impl;

  explicit GpuModel(const Model &weights);  // uploads weights to device
  ~GpuModel();
  GpuModel(const GpuModel &) = delete;
  GpuModel &operator=(const GpuModel &) = delete;

  // Full pipeline on GPU: audio -> token ids. Decode loop runs on host with
  // one device sync per token.
  std::vector<int> transcribe(const float *audio, size_t n_samples);
};

// Incremental streaming session against a GpuModel, mirroring the semantics
// of core/moonshine-streaming-model.cpp: frontend runs only on new audio,
// the encoder re-encodes a 96-frame left-context window and emits only newly
// stabilized frames (holding back 16 lookahead frames until finalize), and
// the decoder re-decodes from scratch against the accumulated memory.
struct GpuStreamImpl;

struct GpuStream {
  GpuStreamImpl *impl;

  // max_seconds bounds device buffer allocation for one segment.
  GpuStream(GpuModel &model, float max_seconds);
  ~GpuStream();
  GpuStream(const GpuStream &) = delete;
  GpuStream &operator=(const GpuStream &) = delete;

  void reset();                                   // start a new segment
  void add_audio(const float *audio, size_t n);   // frontend on new samples
  void encode(bool is_final);                     // emit stable frames
  std::vector<int> decode(int max_tokens);        // greedy, from scratch
  int memory_frames() const;                      // current memory length
};

#endif
