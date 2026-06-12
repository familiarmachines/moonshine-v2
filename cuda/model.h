// moonshine-streaming-tiny weights + CPU fp32 reference forward pass.
// Architecture spec: see ARCHITECTURE.md. All pointers borrow from the
// mmap'd SafeTensors file.
#ifndef MOONSHINE_CUDA_MODEL_H
#define MOONSHINE_CUDA_MODEL_H

#include <memory>
#include <string>
#include <vector>

#include "safetensors.h"

struct ModelConfig {
  int hidden = 320;
  int heads = 8;
  int head_dim = 40;
  int enc_layers = 6;
  int dec_layers = 6;
  int enc_ffn = 1280;
  int dec_ffn = 1280;  // fc1 outputs 2*dec_ffn (value‖gate)
  int vocab = 32768;
  int frame_len = 80;  // 5 ms @ 16 kHz
  int rotary_dim = 32;  // 0.8 * head_dim
  float rope_theta = 10000.0f;
  int max_positions = 4096;
  int bos_id = 1;
  int eos_id = 2;
  float max_tokens_per_second = 6.5f;  // parity with core benchmark
  int max_tokens_cap = 256;
  // Per-layer (left, right) sliding attention windows.
  int win_left[6] = {16, 16, 16, 16, 16, 16};
  int win_right[6] = {4, 4, 0, 0, 4, 4};
};

struct EncLayerWeights {
  const float *ln1_g, *ln2_g;            // [D] unit-offset gammas
  const float *q_w, *k_w, *v_w, *o_w;    // [D, D]
  const float *fc1_w, *fc1_b;            // [FF, D], [FF]
  const float *fc2_w, *fc2_b;            // [D, FF], [D]
};

struct DecLayerWeights {
  const float *ln1_w, *ln2_w, *ln3_w;    // [D] affine LayerNorm weights
  const float *sq_w, *sk_w, *sv_w, *so_w;  // self-attention [D, D]
  const float *cq_w, *ck_w, *cv_w, *co_w;  // cross-attention [D, D]
  const float *fc1_w, *fc1_b;            // [2*FF, D], [2*FF]
  const float *fc2_w, *fc2_b;            // [D, FF], [D]
};

struct Model {
  ModelConfig cfg;
  std::unique_ptr<SafeTensors> st;

  // Frontend
  float log_k;
  const float *fe_linear_w;              // [D, frame_len]
  const float *conv1_w, *conv1_b;        // [2D, D, 5], [2D]
  const float *conv2_w, *conv2_b;        // [D, 2D, 5], [D]

  EncLayerWeights enc[6];
  const float *enc_final_g;              // [D]

  const float *embed;                    // [V, D]
  const float *pos_emb;                  // [max_positions, D]
  DecLayerWeights dec[6];
  const float *dec_norm_w;               // [D]
  const float *proj_out;                 // [V, D]

  explicit Model(const std::string &safetensors_path);
};

// audio (16 kHz mono float) -> feature frames [T50, D] at 50 Hz.
std::vector<float> frontend_forward(const Model &m,
                                    const float *audio, size_t n_samples,
                                    int *t50_out);

// features [T, D] -> encoder output [T, D] (full bidirectional-window pass).
std::vector<float> encoder_forward(const Model &m, const float *features,
                                   int T);

// Greedy decode against memory[T, D] (= encoder output + pos_emb, already
// summed). Returns token ids without BOS/EOS.
std::vector<int> decode_greedy(const Model &m, const float *memory, int T,
                               int max_tokens);

// Convenience: full pipeline, returns token ids.
std::vector<int> transcribe(const Model &m, const float *audio,
                            size_t n_samples);

#endif
