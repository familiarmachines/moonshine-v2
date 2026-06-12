#include "model.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

// y[M] = W[M, K] * x[K]  (linear layer with row-major [out, in] weights)
void matvec(const float *__restrict W, const float *__restrict x,
            float *__restrict y, int M, int K) {
  for (int m = 0; m < M; m++) {
    const float *w = W + (size_t)m * K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += w[k] * x[k];
    y[m] = acc;
  }
}

// C[T, M] = X[T, K] * W[M, K]^T, optional bias[M].
void linear(const float *__restrict X, const float *__restrict W,
            const float *__restrict bias, float *__restrict C, int T, int M,
            int K) {
  for (int t = 0; t < T; t++) {
    const float *x = X + (size_t)t * K;
    float *c = C + (size_t)t * M;
    matvec(W, x, c, M, K);
    if (bias)
      for (int m = 0; m < M; m++) c[m] += bias[m];
  }
}

inline float silu(float x) { return x / (1.0f + expf(-x)); }
inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }

// LayerNorm(affine=False) * (gamma + 1), eps 1e-5 (encoder unit-offset norm).
void norm_unit_offset(const float *__restrict x, const float *__restrict gamma,
                      float *__restrict y, int D) {
  float mean = 0.0f;
  for (int i = 0; i < D; i++) mean += x[i];
  mean /= D;
  float var = 0.0f;
  for (int i = 0; i < D; i++) {
    float c = x[i] - mean;
    var += c * c;
  }
  var /= D;
  float inv = 1.0f / sqrtf(var + 1e-5f);
  for (int i = 0; i < D; i++) y[i] = (x[i] - mean) * inv * (gamma[i] + 1.0f);
}

// Standard LayerNorm with affine weight, no bias, eps 1e-5 (decoder).
void norm_affine(const float *__restrict x, const float *__restrict w,
                 float *__restrict y, int D) {
  float mean = 0.0f;
  for (int i = 0; i < D; i++) mean += x[i];
  mean /= D;
  float var = 0.0f;
  for (int i = 0; i < D; i++) {
    float c = x[i] - mean;
    var += c * c;
  }
  var /= D;
  float inv = 1.0f / sqrtf(var + 1e-5f);
  for (int i = 0; i < D; i++) y[i] = (x[i] - mean) * inv * w[i];
}

void softmax_inplace(float *x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; i++) mx = std::max(mx, x[i]);
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - mx);
    sum += x[i];
  }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) x[i] *= inv;
}

// Interleaved partial RoPE on one [head_dim] vector at position pos.
void rope_inplace(float *q, int pos, const ModelConfig &cfg) {
  for (int j = 0; j < cfg.rotary_dim / 2; j++) {
    float freq = powf(cfg.rope_theta, -2.0f * j / cfg.rotary_dim);
    float angle = pos * freq;
    float c = cosf(angle), s = sinf(angle);
    float a = q[2 * j], b = q[2 * j + 1];
    q[2 * j] = a * c - b * s;
    q[2 * j + 1] = b * c + a * s;
  }
}

// Causal Conv1d, left-pad (k-1) zeros, stride 2.
// x [T_in, C_in] row-major frames; w [C_out, C_in, k]; y [T_out, C_out].
void causal_conv_s2(const float *__restrict x, int T_in, int C_in,
                    const float *__restrict w, const float *__restrict b,
                    int C_out, int k, float *__restrict y, int T_out) {
  for (int t = 0; t < T_out; t++) {
    float *out = y + (size_t)t * C_out;
    for (int oc = 0; oc < C_out; oc++) out[oc] = b[oc];
    for (int ki = 0; ki < k; ki++) {
      int src = 2 * t - (k - 1) + ki;
      if (src < 0 || src >= T_in) continue;
      const float *xin = x + (size_t)src * C_in;
      for (int oc = 0; oc < C_out; oc++) {
        const float *wrow = w + ((size_t)oc * C_in) * k + ki;
        float acc = 0.0f;
        for (int ic = 0; ic < C_in; ic++) acc += wrow[(size_t)ic * k] * xin[ic];
        out[oc] += acc;
      }
    }
  }
}

}  // namespace

Model::Model(const std::string &path) : st(new SafeTensors(path.c_str())) {
  const ModelConfig &c = cfg;
  auto t1 = [&](const std::string &n) { return st->get(n, 1).data; };
  auto t2 = [&](const std::string &n) { return st->get(n, 2).data; };
  auto t3 = [&](const std::string &n) { return st->get(n, 3).data; };

  const std::string fe = "model.encoder.embedder.";
  log_k = *st->get(fe + "comp.log_k", 0).data;
  fe_linear_w = t2(fe + "linear.weight");
  conv1_w = t3(fe + "conv1.weight");
  conv1_b = t1(fe + "conv1.bias");
  conv2_w = t3(fe + "conv2.weight");
  conv2_b = t1(fe + "conv2.bias");

  for (int l = 0; l < c.enc_layers; l++) {
    const std::string p = "model.encoder.layers." + std::to_string(l) + ".";
    enc[l].ln1_g = t1(p + "input_layernorm.gamma");
    enc[l].ln2_g = t1(p + "post_attention_layernorm.gamma");
    enc[l].q_w = t2(p + "self_attn.q_proj.weight");
    enc[l].k_w = t2(p + "self_attn.k_proj.weight");
    enc[l].v_w = t2(p + "self_attn.v_proj.weight");
    enc[l].o_w = t2(p + "self_attn.o_proj.weight");
    enc[l].fc1_w = t2(p + "mlp.fc1.weight");
    enc[l].fc1_b = t1(p + "mlp.fc1.bias");
    enc[l].fc2_w = t2(p + "mlp.fc2.weight");
    enc[l].fc2_b = t1(p + "mlp.fc2.bias");
  }
  enc_final_g = t1("model.encoder.final_norm.gamma");

  embed = t2("model.decoder.embed_tokens.weight");
  pos_emb = t2("model.decoder.pos_emb.weight");
  for (int l = 0; l < c.dec_layers; l++) {
    const std::string p = "model.decoder.layers." + std::to_string(l) + ".";
    dec[l].ln1_w = t1(p + "input_layernorm.weight");
    dec[l].ln2_w = t1(p + "post_attention_layernorm.weight");
    dec[l].ln3_w = t1(p + "final_layernorm.weight");
    dec[l].sq_w = t2(p + "self_attn.q_proj.weight");
    dec[l].sk_w = t2(p + "self_attn.k_proj.weight");
    dec[l].sv_w = t2(p + "self_attn.v_proj.weight");
    dec[l].so_w = t2(p + "self_attn.o_proj.weight");
    dec[l].cq_w = t2(p + "encoder_attn.q_proj.weight");
    dec[l].ck_w = t2(p + "encoder_attn.k_proj.weight");
    dec[l].cv_w = t2(p + "encoder_attn.v_proj.weight");
    dec[l].co_w = t2(p + "encoder_attn.o_proj.weight");
    dec[l].fc1_w = t2(p + "mlp.fc1.weight");
    dec[l].fc1_b = t1(p + "mlp.fc1.bias");
    dec[l].fc2_w = t2(p + "mlp.fc2.weight");
    dec[l].fc2_b = t1(p + "mlp.fc2.bias");
  }
  dec_norm_w = t1("model.decoder.norm.weight");
  proj_out = t2("proj_out.weight");
}

std::vector<float> frontend_forward(const Model &m, const float *audio,
                                    size_t n_samples, int *t50_out) {
  const ModelConfig &c = m.cfg;
  const int D = c.hidden;
  const int T200 = (int)(n_samples / c.frame_len);
  if (T200 < 4) throw std::runtime_error("audio too short");

  // CMVN + asinh compression + linear + silu, per 80-sample frame.
  std::vector<float> h200((size_t)T200 * D);
  std::vector<float> frame(c.frame_len);
  const float ek = expf(m.log_k);
  for (int t = 0; t < T200; t++) {
    const float *x = audio + (size_t)t * c.frame_len;
    float mean = 0.0f;
    for (int i = 0; i < c.frame_len; i++) mean += x[i];
    mean /= c.frame_len;
    float var = 0.0f;
    for (int i = 0; i < c.frame_len; i++) {
      float cc = x[i] - mean;
      var += cc * cc;
    }
    var /= c.frame_len;
    float inv = 1.0f / sqrtf(var + 1e-6f);
    for (int i = 0; i < c.frame_len; i++)
      frame[i] = asinhf(ek * (x[i] - mean) * inv);
    float *h = h200.data() + (size_t)t * D;
    matvec(m.fe_linear_w, frame.data(), h, D, c.frame_len);
    for (int i = 0; i < D; i++) h[i] = silu(h[i]);
  }

  // conv1 (D -> 2D, k5 s2 causal) + silu
  const int T100 = (T200 - 1) / 2 + 1;
  std::vector<float> h100((size_t)T100 * 2 * D);
  causal_conv_s2(h200.data(), T200, D, m.conv1_w, m.conv1_b, 2 * D, 5,
                 h100.data(), T100);
  for (float &v : h100) v = silu(v);

  // conv2 (2D -> D, k5 s2 causal), no activation
  const int T50 = (T100 - 1) / 2 + 1;
  std::vector<float> h50((size_t)T50 * D);
  causal_conv_s2(h100.data(), T100, 2 * D, m.conv2_w, m.conv2_b, D, 5,
                 h50.data(), T50);
  *t50_out = T50;
  return h50;
}

std::vector<float> encoder_forward(const Model &m, const float *features,
                                   int T) {
  const ModelConfig &c = m.cfg;
  const int D = c.hidden, H = c.heads, HD = c.head_dim;
  const float scale = 1.0f / sqrtf((float)HD);

  std::vector<float> x(features, features + (size_t)T * D);
  std::vector<float> xn((size_t)T * D), q((size_t)T * D), k((size_t)T * D),
      v((size_t)T * D), attn_out((size_t)T * D), ff((size_t)T * c.enc_ffn);
  std::vector<float> scores(T);

  for (int l = 0; l < c.enc_layers; l++) {
    const EncLayerWeights &w = m.enc[l];
    // --- self-attention block ---
    for (int t = 0; t < T; t++)
      norm_unit_offset(x.data() + (size_t)t * D, w.ln1_g,
                       xn.data() + (size_t)t * D, D);
    linear(xn.data(), w.q_w, nullptr, q.data(), T, D, D);
    linear(xn.data(), w.k_w, nullptr, k.data(), T, D, D);
    linear(xn.data(), w.v_w, nullptr, v.data(), T, D, D);

    const int left = c.win_left[l], right = c.win_right[l];
    for (int t = 0; t < T; t++) {
      // window: kv in [t-left+1, t] U (t, t+right-1]
      int lo = std::max(0, t - left + 1);
      int hi = std::min(T - 1, right > 0 ? t + right - 1 : t);
      int n = hi - lo + 1;
      float *out = attn_out.data() + (size_t)t * D;
      for (int h = 0; h < H; h++) {
        const float *qh = q.data() + (size_t)t * D + h * HD;
        for (int s = 0; s < n; s++) {
          const float *kh = k.data() + (size_t)(lo + s) * D + h * HD;
          float acc = 0.0f;
          for (int d = 0; d < HD; d++) acc += qh[d] * kh[d];
          scores[s] = acc * scale;
        }
        softmax_inplace(scores.data(), n);
        float *oh = out + h * HD;
        for (int d = 0; d < HD; d++) oh[d] = 0.0f;
        for (int s = 0; s < n; s++) {
          const float *vh = v.data() + (size_t)(lo + s) * D + h * HD;
          float p = scores[s];
          for (int d = 0; d < HD; d++) oh[d] += p * vh[d];
        }
      }
    }
    // o_proj + residual (reuse xn as projection buffer)
    linear(attn_out.data(), w.o_w, nullptr, xn.data(), T, D, D);
    for (size_t i = 0; i < x.size(); i++) x[i] += xn[i];

    // --- MLP block ---
    for (int t = 0; t < T; t++)
      norm_unit_offset(x.data() + (size_t)t * D, w.ln2_g,
                       xn.data() + (size_t)t * D, D);
    linear(xn.data(), w.fc1_w, w.fc1_b, ff.data(), T, c.enc_ffn, D);
    for (float &val : ff) val = gelu(val);
    linear(ff.data(), w.fc2_w, w.fc2_b, xn.data(), T, D, c.enc_ffn);
    for (size_t i = 0; i < x.size(); i++) x[i] += xn[i];
  }

  std::vector<float> out((size_t)T * D);
  for (int t = 0; t < T; t++)
    norm_unit_offset(x.data() + (size_t)t * D, m.enc_final_g,
                     out.data() + (size_t)t * D, D);
  return out;
}

std::vector<int> decode_greedy(const Model &m, const float *memory, int T,
                               int max_tokens) {
  const ModelConfig &c = m.cfg;
  const int D = c.hidden, H = c.heads, HD = c.head_dim, L = c.dec_layers;
  const float scale = 1.0f / sqrtf((float)HD);

  // Precompute cross-attention K/V per layer: [T, D].
  std::vector<std::vector<float>> k_cross(L), v_cross(L);
  for (int l = 0; l < L; l++) {
    k_cross[l].resize((size_t)T * D);
    v_cross[l].resize((size_t)T * D);
    linear(memory, m.dec[l].ck_w, nullptr, k_cross[l].data(), T, D, D);
    linear(memory, m.dec[l].cv_w, nullptr, v_cross[l].data(), T, D, D);
  }

  // Self-attention KV caches, grown per step: [step, D].
  std::vector<std::vector<float>> k_self(L), v_self(L);

  std::vector<float> x(D), xn(D), q(D), kv(D), attn(D), proj(D);
  std::vector<float> ff((size_t)2 * c.dec_ffn);
  std::vector<float> scores(std::max(T, max_tokens + 1));
  std::vector<float> logits(c.vocab);

  std::vector<int> out_tokens;
  int token = c.bos_id;
  for (int step = 0; step < max_tokens; step++) {
    memcpy(x.data(), m.embed + (size_t)token * D, D * sizeof(float));

    for (int l = 0; l < L; l++) {
      const DecLayerWeights &w = m.dec[l];
      // --- causal self-attention with KV cache ---
      norm_affine(x.data(), w.ln1_w, xn.data(), D);
      matvec(w.sq_w, xn.data(), q.data(), D, D);
      for (int h = 0; h < H; h++) rope_inplace(q.data() + h * HD, step, c);
      matvec(w.sk_w, xn.data(), kv.data(), D, D);
      for (int h = 0; h < H; h++) rope_inplace(kv.data() + h * HD, step, c);
      k_self[l].insert(k_self[l].end(), kv.begin(), kv.end());
      matvec(w.sv_w, xn.data(), kv.data(), D, D);
      v_self[l].insert(v_self[l].end(), kv.begin(), kv.end());
      const int n = step + 1;
      for (int h = 0; h < H; h++) {
        const float *qh = q.data() + h * HD;
        for (int s = 0; s < n; s++) {
          const float *kh = k_self[l].data() + (size_t)s * D + h * HD;
          float acc = 0.0f;
          for (int d = 0; d < HD; d++) acc += qh[d] * kh[d];
          scores[s] = acc * scale;
        }
        softmax_inplace(scores.data(), n);
        float *oh = attn.data() + h * HD;
        for (int d = 0; d < HD; d++) oh[d] = 0.0f;
        for (int s = 0; s < n; s++) {
          const float *vh = v_self[l].data() + (size_t)s * D + h * HD;
          float p = scores[s];
          for (int d = 0; d < HD; d++) oh[d] += p * vh[d];
        }
      }
      matvec(w.so_w, attn.data(), proj.data(), D, D);
      for (int i = 0; i < D; i++) x[i] += proj[i];

      // --- cross-attention (no RoPE) ---
      norm_affine(x.data(), w.ln2_w, xn.data(), D);
      matvec(w.cq_w, xn.data(), q.data(), D, D);
      for (int h = 0; h < H; h++) {
        const float *qh = q.data() + h * HD;
        for (int s = 0; s < T; s++) {
          const float *kh = k_cross[l].data() + (size_t)s * D + h * HD;
          float acc = 0.0f;
          for (int d = 0; d < HD; d++) acc += qh[d] * kh[d];
          scores[s] = acc * scale;
        }
        softmax_inplace(scores.data(), T);
        float *oh = attn.data() + h * HD;
        for (int d = 0; d < HD; d++) oh[d] = 0.0f;
        for (int s = 0; s < T; s++) {
          const float *vh = v_cross[l].data() + (size_t)s * D + h * HD;
          float p = scores[s];
          for (int d = 0; d < HD; d++) oh[d] += p * vh[d];
        }
      }
      matvec(w.co_w, attn.data(), proj.data(), D, D);
      for (int i = 0; i < D; i++) x[i] += proj[i];

      // --- SwiGLU MLP: fc1 -> (value, gate) halves -> silu(gate)*value ---
      norm_affine(x.data(), w.ln3_w, xn.data(), D);
      matvec(w.fc1_w, xn.data(), ff.data(), 2 * c.dec_ffn, D);
      for (int i = 0; i < 2 * c.dec_ffn; i++) ff[i] += w.fc1_b[i];
      for (int i = 0; i < c.dec_ffn; i++)
        ff[i] = silu(ff[c.dec_ffn + i]) * ff[i];
      matvec(w.fc2_w, ff.data(), proj.data(), D, c.dec_ffn);
      for (int i = 0; i < D; i++) x[i] += proj[i] + w.fc2_b[i];
    }

    norm_affine(x.data(), m.dec_norm_w, xn.data(), D);
    matvec(m.proj_out, xn.data(), logits.data(), c.vocab, D);
    int best = 0;
    for (int i = 1; i < c.vocab; i++)
      if (logits[i] > logits[best]) best = i;
    if (best == c.eos_id) break;
    out_tokens.push_back(best);
    token = best;
  }
  return out_tokens;
}

std::vector<int> transcribe(const Model &m, const float *audio,
                            size_t n_samples) {
  int T = 0;
  std::vector<float> features = frontend_forward(m, audio, n_samples, &T);
  std::vector<float> enc = encoder_forward(m, features.data(), T);
  if (T > m.cfg.max_positions)
    throw std::runtime_error("audio too long for pos_emb table");
  // Adapter: memory = encoder output + learned absolute position embedding.
  for (int t = 0; t < T; t++)
    for (int d = 0; d < m.cfg.hidden; d++)
      enc[(size_t)t * m.cfg.hidden + d] +=
          m.pos_emb[(size_t)t * m.cfg.hidden + d];

  float dur = n_samples / 16000.0f;
  int max_tokens = std::min(m.cfg.max_tokens_cap,
                            (int)ceilf(dur * m.cfg.max_tokens_per_second));
  return decode_greedy(m, enc.data(), T, max_tokens);
}
