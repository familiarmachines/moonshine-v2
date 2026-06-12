#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <stdexcept>
#include <vector>

#include "model_cuda.h"

namespace cg = cooperative_groups;

#define CUDA_CHECK(x)                                                       \
  do {                                                                      \
    cudaError_t err_ = (x);                                                 \
    if (err_ != cudaSuccess)                                                \
      throw std::runtime_error(std::string("CUDA: ") +                     \
                               cudaGetErrorString(err_) + " at " __FILE__   \
                               ":" + std::to_string(__LINE__));             \
  } while (0)

#define CUBLAS_CHECK(x)                                                     \
  do {                                                                      \
    cublasStatus_t st_ = (x);                                               \
    if (st_ != CUBLAS_STATUS_SUCCESS)                                       \
      throw std::runtime_error("cuBLAS error at " __FILE__ ":" +            \
                               std::to_string(__LINE__));                   \
  } while (0)

namespace {

constexpr int D = 320, H = 8, HD = 40, FF = 1280, V = 32768, FRAME = 80;

// ---------------------------------------------------------------- kernels

__device__ inline float silu_d(float x) { return x / (1.0f + expf(-x)); }
__device__ inline float gelu_d(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678f));
}

// One 80-sample frame -> CMVN -> asinh(e^log_k * x) -> linear(80->320) -> silu
// grid: T200 blocks, block: 320 threads.
__global__ void k_frame_embed(const float *__restrict__ audio,
                              const float *__restrict__ w,  // [320, 80]
                              float ek, float *__restrict__ out) {
  __shared__ float frame[FRAME];
  __shared__ float stats[2];
  const float *x = audio + (size_t)blockIdx.x * FRAME;
  if (threadIdx.x < FRAME) frame[threadIdx.x] = x[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    float mean = 0.0f;
    for (int i = 0; i < FRAME; i++) mean += frame[i];
    mean /= FRAME;
    float var = 0.0f;
    for (int i = 0; i < FRAME; i++) {
      float c = frame[i] - mean;
      var += c * c;
    }
    stats[0] = mean;
    stats[1] = rsqrtf(var / FRAME + 1e-6f);
  }
  __syncthreads();
  if (threadIdx.x < FRAME)
    frame[threadIdx.x] = asinhf(ek * (frame[threadIdx.x] - stats[0]) * stats[1]);
  __syncthreads();
  const float *wr = w + (size_t)threadIdx.x * FRAME;
  float acc = 0.0f;
  for (int i = 0; i < FRAME; i++) acc += wr[i] * frame[i];
  out[(size_t)blockIdx.x * D + threadIdx.x] = silu_d(acc);
}

// Causal Conv1d stride 2 with pre-transposed weights wt[k][C_in][C_out].
// Computes output rows [t0, t0 + gridDim.x) against the full input buffer.
// grid: T_out blocks, block: C_out threads. Shared: 5*C_in floats.
template <int C_IN, int C_OUT, bool SILU>
__global__ void k_causal_conv_s2(const float *__restrict__ x, int T_in,
                                 const float *__restrict__ wt,
                                 const float *__restrict__ bias,
                                 float *__restrict__ y, int t0 = 0) {
  extern __shared__ float xs[];  // [5][C_IN]
  const int t = t0 + blockIdx.x;
  for (int idx = threadIdx.x; idx < 5 * C_IN; idx += blockDim.x) {
    int ki = idx / C_IN, ic = idx % C_IN;
    int src = 2 * t - 4 + ki;
    xs[idx] = (src >= 0 && src < T_in) ? x[(size_t)src * C_IN + ic] : 0.0f;
  }
  __syncthreads();
  const int oc = threadIdx.x;
  float acc = bias[oc];
  for (int ki = 0; ki < 5; ki++) {
    const float *wk = wt + ((size_t)ki * C_IN) * C_OUT + oc;
    const float *xk = xs + ki * C_IN;
    for (int ic = 0; ic < C_IN; ic++) acc += wk[(size_t)ic * C_OUT] * xk[ic];
  }
  y[(size_t)t * C_OUT + oc] = SILU ? silu_d(acc) : acc;
}

// LayerNorm over D dims. unit_offset: scale = (g + 1) (encoder), else g.
// grid: rows, block: 256.
template <bool UNIT_OFFSET>
__global__ void k_layernorm(const float *__restrict__ x,
                            const float *__restrict__ g,
                            float *__restrict__ y, int rows) {
  __shared__ float red[256];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float *xr = x + (size_t)row * D;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) sum += xr[i];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float mean = red[0] / D;
  __syncthreads();
  float vs = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float c = xr[i] - mean;
    vs += c * c;
  }
  red[threadIdx.x] = vs;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / D + 1e-5f);
  float *yr = y + (size_t)row * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float scale = UNIT_OFFSET ? g[i] + 1.0f : g[i];
    yr[i] = (xr[i] - mean) * inv * scale;
  }
}

// y[i] += x[i]
__global__ void k_add(float *__restrict__ y, const float *__restrict__ x,
                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += x[i];
}

// y[i] = act(y[i] + b[i % bn]); ACT: 0 none, 1 gelu, 2 silu
template <int ACT>
__global__ void k_bias_act(float *__restrict__ y, const float *__restrict__ b,
                           int n, int bn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = y[i] + b[i % bn];
  if (ACT == 1) v = gelu_d(v);
  if (ACT == 2) v = silu_d(v);
  y[i] = v;
}

// x[i] += y[i] + b[i % bn]  (residual with biased projection)
__global__ void k_residual_bias(float *__restrict__ x,
                                const float *__restrict__ y,
                                const float *__restrict__ b, int n, int bn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += y[i] + b[i % bn];
}

// Encoder sliding-window attention. grid: (T, H), block: 64.
// Window: kv in [t-left+1, t] U (t, t+right-1].
__global__ void k_enc_attn(const float *__restrict__ q,
                           const float *__restrict__ k,
                           const float *__restrict__ v, int T, int left,
                           int right, float *__restrict__ out) {
  __shared__ float sc[32];
  __shared__ float qs[HD];
  const int t = blockIdx.x, h = blockIdx.y;
  const int lo = max(0, t - left + 1);
  const int hi = min(T - 1, right > 0 ? t + right - 1 : t);
  const int n = hi - lo + 1;
  if (threadIdx.x < HD) qs[threadIdx.x] = q[(size_t)t * D + h * HD + threadIdx.x];
  __syncthreads();
  if (threadIdx.x < n) {
    const float *kh = k + (size_t)(lo + threadIdx.x) * D + h * HD;
    float acc = 0.0f;
    for (int d = 0; d < HD; d++) acc += qs[d] * kh[d];
    sc[threadIdx.x] = acc * 0.15811388f;  // 1/sqrt(40)
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float mx = sc[0];
    for (int s = 1; s < n; s++) mx = fmaxf(mx, sc[s]);
    float sum = 0.0f;
    for (int s = 0; s < n; s++) {
      sc[s] = expf(sc[s] - mx);
      sum += sc[s];
    }
    float inv = 1.0f / sum;
    for (int s = 0; s < n; s++) sc[s] *= inv;
  }
  __syncthreads();
  if (threadIdx.x < HD) {
    float acc = 0.0f;
    for (int s = 0; s < n; s++)
      acc += sc[s] * v[(size_t)(lo + s) * D + h * HD + threadIdx.x];
    out[(size_t)t * D + h * HD + threadIdx.x] = acc;
  }
}

// memory[t][d] += pos_emb[t][d]
__global__ void k_add_pos(float *__restrict__ mem,
                          const float *__restrict__ pos, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) mem[i] += pos[i];
}

// Interleaved partial RoPE on one [D] q-or-k vector. block: 128 (8 heads x 16
// pairs).
__global__ void k_rope(float *__restrict__ x, int pos) {
  const int h = threadIdx.x / 16, j = threadIdx.x % 16;
  if (h >= H) return;
  const float freq = powf(10000.0f, -2.0f * j / 32.0f);
  float c, s;
  sincosf(pos * freq, &s, &c);
  float *p = x + h * HD + 2 * j;
  float a = p[0], b = p[1];
  p[0] = a * c - b * s;
  p[1] = b * c + a * s;
}

// Softmax over scores[H][n] (already scaled). grid: H, block: 256.
__global__ void k_softmax_rows(float *__restrict__ scores, int n) {
  __shared__ float red[256];
  float *row = scores + (size_t)blockIdx.x * n;
  float mx = -1e30f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, row[i]);
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    row[i] = expf(row[i] - mx);
    sum += row[i];
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / red[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] *= inv;
}

// SwiGLU halves: ff[i] = silu(ff[FF + i]) * ff[i], i < FF.
__global__ void k_swiglu(float *__restrict__ ff) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < FF) ff[i] = silu_d(ff[FF + i]) * ff[i];
}

// ------------------------------------------------ fused encoder megakernel
//
// The whole 6-layer encoder in ONE cooperative launch. Each block owns a
// contiguous chunk of frames; activations x stay in global memory but are
// only ever touched by the owning block. The sliding-window attention is the
// only cross-block dependency, so each layer needs exactly two grid syncs:
// after Q/K/V are published (neighbors read K/V halos) and after the layer's
// output is complete (before the next layer overwrites Q/K/V).
//
// Throughput hinges on weight reuse: frames are processed in register-tiled
// groups of FT=16, so one weight load from L2/DRAM feeds 16 FMAs. Weight
// matrices are pre-transposed to [in][out] so consecutive threads (one output
// column each) read consecutive addresses.

struct EncFusedWeights {
  // Weight matrices are stored fp16 (halves DRAM traffic and keeps each
  // phase-locked stage L2-resident on a 2 MB L2); math is fp32.
  const __half *wqkv_t[6];  // [320][960]  columns: q | k | v
  const __half *wo_t[6];    // [320][320]
  const __half *fc1_t[6];   // [320][1280]
  const __half *fc2_t[6];   // [1280][320]
  const float *fc1_b[6], *fc2_b[6];
  const float *ln1_g[6], *ln2_g[6];
  const float *final_g;
  int win_left[6], win_right[6];
};

constexpr int FT = 16;  // frames per register tile

// Stage timing for MOONSHINE_ENC_PROF=1 (block 0 records clock64 deltas).
__device__ unsigned long long g_enc_prof[32];
#define PROF(slot)                                  \
  if (blockIdx.x == 0 && threadIdx.x == 0) g_enc_prof[slot] = clock64();

// Warp-cooperative unit-offset LayerNorm of one frame row; output fp32 or
// fp16 depending on destination type.
template <typename T>
__device__ inline void warp_ln_unit(const float *__restrict__ x,
                                    const float *__restrict__ g,
                                    T *__restrict__ out, int lane) {
  float sum = 0.0f;
  for (int i = lane; i < D; i += 32) sum += x[i];
  for (int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, s);
  const float mean = sum / D;
  float vs = 0.0f;
  for (int i = lane; i < D; i += 32) {
    float c = x[i] - mean;
    vs += c * c;
  }
  for (int s = 16; s > 0; s >>= 1) vs += __shfl_xor_sync(0xffffffffu, vs, s);
  const float inv = rsqrtf(vs / D + 1e-5f);
  for (int i = lane; i < D; i += 32)
    out[i] = (T)((x[i] - mean) * inv * (g[i] + 1.0f));
}

using namespace nvcuda;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> FragA;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> FragB;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> FragC;

// GEMM stages run on tensor cores: out[16 frames, N outs] = A[16, K] x B[K, N]
// with A (fp16) staged in shared, B (fp16) streamed from global (L2-resident
// per the phase-locked schedule), fp32 accumulate. The residual stream x
// stays fp32 in global.
__global__ void __launch_bounds__(256, 2)
k_encoder_fused(float *__restrict__ x, __half *__restrict__ q_g,
                __half *__restrict__ k_g, __half *__restrict__ v_g,
                float *__restrict__ out, int len, EncFusedWeights W) {
  cg::grid_group grid = cg::this_grid();
  __shared__ __half buf1[FT][D];      // LN output tile (A operand)
  __shared__ __half buf2[FT][D];      // attention out / MLP hidden chunk
  __shared__ float scr[8][16][16];    // per-warp accumulator staging

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int chunk = (len + gridDim.x - 1) / gridDim.x;
  const int f0 = min(len, (int)blockIdx.x * chunk);
  const int f1 = min(len, f0 + chunk);
  const int tiles = (chunk + FT - 1) / FT;

  PROF(0);
  for (int l = 0; l < 6; l++) {
    // ---- stage A: LN1 + QKV (tensor cores) -> q_g/k_g/v_g (fp16) ----
    for (int tile = 0; tile < tiles; tile++) {
      const int fs = f0 + tile * FT;
      const int ft = max(0, min(FT, f1 - fs));
      if (ft > 0) {
        for (int f = warp; f < FT; f += 8) {
          if (f < ft)
            warp_ln_unit(x + (size_t)(fs + f) * D, W.ln1_g[l], buf1[f], lane);
          else
            for (int i = lane; i < D; i += 32) buf1[f][i] = __half(0);
        }
        __syncthreads();
        for (int nt = warp; nt < 3 * D / 16; nt += 8) {
          FragC c;
          wmma::fill_fragment(c, 0.0f);
          for (int ks = 0; ks < D / 16; ks++) {
            FragA a;
            FragB b;
            wmma::load_matrix_sync(a, &buf1[0][ks * 16], D);
            wmma::load_matrix_sync(b, W.wqkv_t[l] + (size_t)(ks * 16) * 3 * D +
                                          nt * 16, 3 * D);
            wmma::mma_sync(c, a, b, c);
          }
          wmma::store_matrix_sync(&scr[warp][0][0], c, 16, wmma::mem_row_major);
          for (int e = lane; e < 16 * 16; e += 32) {
            const int r = e / 16, o = nt * 16 + e % 16;
            if (r < ft) {
              __half *dst = (o < D) ? q_g : (o < 2 * D) ? k_g : v_g;
              dst[(size_t)(fs + r) * D + (o % D)] = __float2half(scr[warp][0][e]);
            }
          }
        }
        __syncthreads();
      }
    }
    grid.sync();
    PROF(1 + l * 3);

    // ---- stage B: attention | O-proj | MLP ----
    for (int tile = 0; tile < tiles; tile++) {
      const int fs = f0 + tile * FT;
      const int ft = max(0, min(FT, f1 - fs));
      if (ft > 0) {
        // sliding-window attention: one warp per (frame, head)
        for (int task = warp; task < ft * H; task += 8) {
          const int f = fs + task / H, h = task % H;
          const int lo = max(0, f - W.win_left[l] + 1);
          const int hi =
              min(len - 1, W.win_right[l] > 0 ? f + W.win_right[l] - 1 : f);
          const int n = hi - lo + 1;
          float score = -1e30f;
          if (lane < n) {
            const __half2 *qh =
                (const __half2 *)(q_g + (size_t)f * D + h * HD);
            const __half2 *kh =
                (const __half2 *)(k_g + (size_t)(lo + lane) * D + h * HD);
            float acc = 0.0f;
#pragma unroll
            for (int d = 0; d < HD / 2; d++) {
              const float2 qq = __half22float2(qh[d]);
              const float2 kk = __half22float2(kh[d]);
              acc += qq.x * kk.x + qq.y * kk.y;
            }
            score = acc * 0.15811388f;
          }
          float mx = score;
          for (int s = 16; s > 0; s >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
          const float e = (lane < n) ? expf(score - mx) : 0.0f;
          float sum = e;
          for (int s = 16; s > 0; s >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, s);
          const float p = e / sum;
          // Each lane owns dims (2*lane, 2*lane+1) via one half2 load.
          float acc0 = 0.0f, acc1 = 0.0f;
          for (int s = 0; s < n; s++) {
            const float ps = __shfl_sync(0xffffffffu, p, s);
            if (lane < HD / 2) {
              const float2 vv = __half22float2(*(const __half2 *)(
                  v_g + (size_t)(lo + s) * D + h * HD + 2 * lane));
              acc0 += ps * vv.x;
              acc1 += ps * vv.y;
            }
          }
          if (lane < HD / 2) {
            buf2[task / H][h * HD + 2 * lane] = __float2half(acc0);
            buf2[task / H][h * HD + 2 * lane + 1] = __float2half(acc1);
          }
        }
        __syncthreads();
        for (int f = ft + (int)warp; f < FT; f += 8)
          for (int i = lane; i < D; i += 32) buf2[f][i] = __half(0);
        __syncthreads();
        // O-proj (tensor cores) + residual into fp32 x
        for (int nt = warp; nt < D / 16; nt += 8) {
          FragC c;
          wmma::fill_fragment(c, 0.0f);
          for (int ks = 0; ks < D / 16; ks++) {
            FragA a;
            FragB b;
            wmma::load_matrix_sync(a, &buf2[0][ks * 16], D);
            wmma::load_matrix_sync(b, W.wo_t[l] + (size_t)(ks * 16) * D + nt * 16,
                                   D);
            wmma::mma_sync(c, a, b, c);
          }
          wmma::store_matrix_sync(&scr[warp][0][0], c, 16, wmma::mem_row_major);
          for (int e = lane; e < 16 * 16; e += 32) {
            const int r = e / 16, o = nt * 16 + e % 16;
            if (r < ft) x[(size_t)(fs + r) * D + o] += scr[warp][0][e];
          }
        }
        __syncthreads();
        // LN2 -> buf1 (fp16)
        for (int f = warp; f < FT; f += 8) {
          if (f < ft)
            warp_ln_unit(x + (size_t)(fs + f) * D, W.ln2_g[l], buf1[f], lane);
          else
            for (int i = lane; i < D; i += 32) buf1[f][i] = __half(0);
        }
        __syncthreads();
      }
      grid.sync();
      if (tile == 0) PROF(2 + l * 3);

      // MLP: fc1 in 4 column chunks; fc2 accumulated in persistent fragments.
      // Up to ceil(20 n-tiles / 8 warps) = 3 per warp. ALL loops over
      // c2/b fragments are compile-time unrolled with clamped indices so the
      // arrays stay in registers (dynamic indexing puts them on the stack).
      FragC c2[3];
#pragma unroll
      for (int i = 0; i < 3; i++) wmma::fill_fragment(c2[i], 0.0f);
      for (int cc = 0; cc < 4; cc++) {
        if (ft > 0) {
          // fc1 chunk -> gelu -> buf2 (fp16)
          for (int nt = warp; nt < D / 16; nt += 8) {
            FragC c;
            wmma::fill_fragment(c, 0.0f);
            for (int ks = 0; ks < D / 16; ks++) {
              FragA a;
              FragB b;
              wmma::load_matrix_sync(a, &buf1[0][ks * 16], D);
              wmma::load_matrix_sync(
                  b, W.fc1_t[l] + (size_t)(ks * 16) * FF + cc * D + nt * 16, FF);
              wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(&scr[warp][0][0], c, 16,
                                    wmma::mem_row_major);
            for (int e = lane; e < 16 * 16; e += 32) {
              const int r = e / 16, o = nt * 16 + e % 16;
              buf2[r][o] = __float2half(
                  (r < ft) ? gelu_d(scr[warp][0][e] + W.fc1_b[l][cc * D + o])
                           : 0.0f);
            }
          }
          __syncthreads();
          // fc2 partial: rows cc*320..+319, accumulate into c2 fragments
          for (int ks = 0; ks < D / 16; ks++) {
            FragA a;
            wmma::load_matrix_sync(a, &buf2[0][ks * 16], D);
            FragB b[3];
#pragma unroll
            for (int i = 0; i < 3; i++) {
              const int nt = min(warp + i * 8, D / 16 - 1);  // clamped
              wmma::load_matrix_sync(b[i],
                                     W.fc2_t[l] +
                                         (size_t)(cc * D + ks * 16) * D +
                                         nt * 16, D);
            }
#pragma unroll
            for (int i = 0; i < 3; i++)
              wmma::mma_sync(c2[i], a, b[i], c2[i]);
          }
          __syncthreads();
        }
        grid.sync();
      }
      if (ft > 0) {
#pragma unroll
        for (int i = 0; i < 3; i++) {
          const int nt = warp + i * 8;
          if (nt >= D / 16) continue;
          wmma::store_matrix_sync(&scr[warp][0][0], c2[i], 16,
                                  wmma::mem_row_major);
          for (int e = lane; e < 16 * 16; e += 32) {
            const int r = e / 16, o = nt * 16 + e % 16;
            if (r < ft)
              x[(size_t)(fs + r) * D + o] += scr[warp][0][e] + W.fc2_b[l][o];
          }
        }
        __syncthreads();
      }
    }
    grid.sync();
    PROF(3 + l * 3);
  }

  // final unit-offset norm -> fp32 out
  for (int f = f0 + (int)warp; f < f1; f += 8)
    warp_ln_unit(x + (size_t)f * D, W.final_g, out + (size_t)f * D, lane);
}

// Single-block argmax over n logits. block: 1024.
__global__ void k_argmax(const float *__restrict__ logits, int n,
                         int *__restrict__ out) {
  __shared__ float bv[1024];
  __shared__ int bi[1024];
  float best = -1e30f;
  int besti = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (logits[i] > best) {
      best = logits[i];
      besti = i;
    }
  }
  bv[threadIdx.x] = best;
  bi[threadIdx.x] = besti;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && bv[threadIdx.x + s] > bv[threadIdx.x]) {
      bv[threadIdx.x] = bv[threadIdx.x + s];
      bi[threadIdx.x] = bi[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = bi[0];
}

}  // namespace

// ------------------------------------------------------------ host side

namespace {

float *upload(const float *src, size_t n) {
  float *d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}

// Conv weights [C_out, C_in, k] -> [k][C_in][C_out] for coalesced access.
float *upload_conv_transposed(const float *w, int c_out, int c_in, int k) {
  std::vector<float> t((size_t)k * c_in * c_out);
  for (int oc = 0; oc < c_out; oc++)
    for (int ic = 0; ic < c_in; ic++)
      for (int ki = 0; ki < k; ki++)
        t[((size_t)ki * c_in + ic) * c_out + oc] =
            w[((size_t)oc * c_in + ic) * k + ki];
  return upload(t.data(), t.size());
}

__half *upload_half(const std::vector<float> &v) {
  std::vector<__half> h(v.size());
  for (size_t i = 0; i < v.size(); i++) h[i] = __float2half(v[i]);
  __half *d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  return d;
}

struct GpuEncLayer {
  float *ln1_g, *ln2_g, *q_w, *k_w, *v_w, *o_w, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};
struct GpuDecLayer {
  float *ln1_w, *ln2_w, *ln3_w;
  float *sq_w, *sk_w, *sv_w, *so_w, *cq_w, *ck_w, *cv_w, *co_w;
  float *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};

// Row-major C[T,M] = X[T,K] * W[M,K]^T via column-major cuBLAS.
void gemm_linear(cublasHandle_t h, const float *X, const float *W, float *C,
                 int T, int M, int K) {
  const float one = 1.0f, zero = 0.0f;
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, W, K, X,
                           K, &zero, C, M));
}

// y[M] = W[M,K] * x[K]; beta=1 accumulates into y (used for fused bias).
void gemv_linear(cublasHandle_t h, const float *W, const float *x, float *y,
                 int M, int K, float beta = 0.0f) {
  const float one = 1.0f;
  CUBLAS_CHECK(cublasSgemv(h, CUBLAS_OP_T, K, M, &one, W, K, x, 1, &beta, y, 1));
}

}  // namespace

struct GpuModelImpl {
  cublasHandle_t blas = nullptr;

  // Weights (device)
  float log_k;
  float *fe_linear_w, *conv1_wt, *conv1_b, *conv2_wt, *conv2_b;
  GpuEncLayer enc[6];
  float *enc_final_g;
  float *embed, *pos_emb;
  GpuDecLayer dec[6];
  float *dec_norm_w, *proj_out;

  // Decode scratch (device), sized at construction. Shared across streams;
  // one decode at a time per GpuModel.
  static constexpr int MAX_TOKENS = 320;
  float *x, *xn, *q, *kv, *attn, *proj, *ff, *logits, *scores;
  int *d_best;
  float *k_self[6], *v_self[6];

  ModelConfig cfg;

  // Fused encoder
  EncFusedWeights fused_w = {};
  bool fused_ok = false;
  int fused_blocks = 0;  // co-resident grid size for cooperative launch

  explicit GpuModelImpl(const Model &m) : cfg(m.cfg) {
    CUBLAS_CHECK(cublasCreate(&blas));
    log_k = m.log_k;
    fe_linear_w = upload(m.fe_linear_w, (size_t)D * FRAME);
    conv1_wt = upload_conv_transposed(m.conv1_w, 2 * D, D, 5);
    conv1_b = upload(m.conv1_b, 2 * D);
    conv2_wt = upload_conv_transposed(m.conv2_w, D, 2 * D, 5);
    conv2_b = upload(m.conv2_b, D);
    for (int l = 0; l < 6; l++) {
      const EncLayerWeights &w = m.enc[l];
      enc[l] = {upload(w.ln1_g, D),          upload(w.ln2_g, D),
                upload(w.q_w, (size_t)D * D), upload(w.k_w, (size_t)D * D),
                upload(w.v_w, (size_t)D * D), upload(w.o_w, (size_t)D * D),
                upload(w.fc1_w, (size_t)FF * D), upload(w.fc1_b, FF),
                upload(w.fc2_w, (size_t)D * FF), upload(w.fc2_b, D)};
    }
    enc_final_g = upload(m.enc_final_g, D);
    embed = upload(m.embed, (size_t)V * D);
    pos_emb = upload(m.pos_emb, (size_t)cfg.max_positions * D);
    for (int l = 0; l < 6; l++) {
      const DecLayerWeights &w = m.dec[l];
      dec[l] = {upload(w.ln1_w, D),           upload(w.ln2_w, D),
                upload(w.ln3_w, D),           upload(w.sq_w, (size_t)D * D),
                upload(w.sk_w, (size_t)D * D), upload(w.sv_w, (size_t)D * D),
                upload(w.so_w, (size_t)D * D), upload(w.cq_w, (size_t)D * D),
                upload(w.ck_w, (size_t)D * D), upload(w.cv_w, (size_t)D * D),
                upload(w.co_w, (size_t)D * D), upload(w.fc1_w, (size_t)2 * FF * D),
                upload(w.fc1_b, 2 * FF),      upload(w.fc2_w, (size_t)D * FF),
                upload(w.fc2_b, D)};
    }
    dec_norm_w = upload(m.dec_norm_w, D);
    proj_out = upload(m.proj_out, (size_t)V * D);

    // Fused-encoder weights: transpose to [in][out] for coalesced columns.
    for (int l = 0; l < 6; l++) {
      const EncLayerWeights &w = m.enc[l];
      std::vector<float> qkv((size_t)D * 3 * D);
      for (int k = 0; k < D; k++)
        for (int o = 0; o < D; o++) {
          qkv[(size_t)k * 3 * D + o] = w.q_w[(size_t)o * D + k];
          qkv[(size_t)k * 3 * D + D + o] = w.k_w[(size_t)o * D + k];
          qkv[(size_t)k * 3 * D + 2 * D + o] = w.v_w[(size_t)o * D + k];
        }
      fused_w.wqkv_t[l] = upload_half(qkv);
      std::vector<float> ot((size_t)D * D);
      for (int k = 0; k < D; k++)
        for (int o = 0; o < D; o++)
          ot[(size_t)k * D + o] = w.o_w[(size_t)o * D + k];
      fused_w.wo_t[l] = upload_half(ot);
      std::vector<float> f1((size_t)D * FF);
      for (int k = 0; k < D; k++)
        for (int o = 0; o < FF; o++)
          f1[(size_t)k * FF + o] = w.fc1_w[(size_t)o * D + k];
      fused_w.fc1_t[l] = upload_half(f1);
      std::vector<float> f2((size_t)FF * D);
      for (int k = 0; k < FF; k++)
        for (int o = 0; o < D; o++)
          f2[(size_t)k * D + o] = w.fc2_w[(size_t)o * FF + k];
      fused_w.fc2_t[l] = upload_half(f2);
      fused_w.fc1_b[l] = enc[l].fc1_b;
      fused_w.fc2_b[l] = enc[l].fc2_b;
      fused_w.ln1_g[l] = enc[l].ln1_g;
      fused_w.ln2_g[l] = enc[l].ln2_g;
      fused_w.win_left[l] = cfg.win_left[l];
      fused_w.win_right[l] = cfg.win_right[l];
    }
    fused_w.final_g = enc_final_g;

    int coop = 0;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
                                                  k_encoder_fused, 256, 0);
    fused_blocks = prop.multiProcessorCount * blocks_per_sm;
    fused_ok = coop != 0 && fused_blocks >= 1;
    if (fused_ok && std::getenv("MOONSHINE_FUSED"))
      fprintf(stderr, "moonshine-cuda: fused encoder, %d blocks (%d/SM)\n",
              fused_blocks, blocks_per_sm);

    CUDA_CHECK(cudaMalloc(&x, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&xn, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&kv, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attn, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&proj, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ff, 2 * FF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&logits, V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(int)));
    scores = nullptr;  // sized per transcribe call (max(T, MAX_TOKENS))
    for (int l = 0; l < 6; l++) {
      CUDA_CHECK(cudaMalloc(&k_self[l], (size_t)MAX_TOKENS * D * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&v_self[l], (size_t)MAX_TOKENS * D * sizeof(float)));
    }
  }

  ~GpuModelImpl() {
    // Process teardown frees device memory; skip per-pointer cleanup.
    if (blas) cublasDestroy(blas);
  }

  // Batched per-head attention of one query against a [n, D] cache using
  // strided cuBLAS GEMMs: scores[h] = scale * K_h^T q_h; out_h = V_h probs_h.
  void cached_attention(const float *q_vec, const float *k_cache,
                        const float *v_cache, int n, float *score_buf,
                        float *out) {
    const float scale = 0.15811388f, zero = 0.0f, one = 1.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, HD, &scale, k_cache, D, HD,
        q_vec, HD, HD, &zero, score_buf, n, n, H));
    k_softmax_rows<<<H, 256>>>(score_buf, n);
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, 1, n, &one, v_cache, D, HD,
        score_buf, n, n, &zero, out, HD, HD, H));
  }

  // 6 encoder layers + final norm over x[len, D] (in place); final-normed
  // output lands in xn_t. Scratch buffers must hold len*D (ffb: len*FF).
  void encoder_window(float *x, int len, float *xn_t, float *qb, float *kb,
                      float *vb, float *ab, float *ffb) {
    // The fused megakernel is currently slower than the cuBLAS path
    // (issue-bound; see README "Encoder kernel findings"). Opt-in until the
    // tensor-core version lands.
    if (fused_ok && std::getenv("MOONSHINE_FUSED")) {
      __half *qh = (__half *)qb, *kh = (__half *)kb, *vh = (__half *)vb;
      void *args[] = {(void *)&x,  (void *)&qh,  (void *)&kh, (void *)&vh,
                      (void *)&xn_t, (void *)&len, (void *)&fused_w};
      CUDA_CHECK(cudaLaunchCooperativeKernel((void *)k_encoder_fused,
                                             dim3(fused_blocks), dim3(256),
                                             args, 0, nullptr));
      if (std::getenv("MOONSHINE_ENC_PROF")) {
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long prof[32] = {};
        CUDA_CHECK(cudaMemcpyFromSymbol(prof, g_enc_prof, sizeof(prof)));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        const double us = 1000.0 / prop.clockRate;  // kHz -> us per cycle
        fprintf(stderr, "enc prof len=%d:", len);
        for (int l = 0; l < 6; l++) {
          const double a = (prof[1 + l * 3] - (l ? prof[l * 3] : prof[0])) * us;
          const double b = (prof[2 + l * 3] - prof[1 + l * 3]) * us;
          const double c = (prof[3 + l * 3] - prof[2 + l * 3]) * us;
          fprintf(stderr, " L%d[qkv %.0f attnO %.0f mlp %.0f]", l, a, b, c);
        }
        fprintf(stderr, " total %.0fus\n", (prof[16] - prof[0]) * us);
      }
      return;
    }
    const int TD = len * D;
    const int eltBlocks = (TD + 255) / 256;
    for (int l = 0; l < 6; l++) {
      const GpuEncLayer &w = enc[l];
      k_layernorm<true><<<len, 256>>>(x, w.ln1_g, xn_t, len);
      gemm_linear(blas, xn_t, w.q_w, qb, len, D, D);
      gemm_linear(blas, xn_t, w.k_w, kb, len, D, D);
      gemm_linear(blas, xn_t, w.v_w, vb, len, D, D);
      k_enc_attn<<<dim3(len, H), 64>>>(qb, kb, vb, len, cfg.win_left[l],
                                       cfg.win_right[l], ab);
      gemm_linear(blas, ab, w.o_w, xn_t, len, D, D);
      k_add<<<eltBlocks, 256>>>(x, xn_t, TD);

      k_layernorm<true><<<len, 256>>>(x, w.ln2_g, xn_t, len);
      gemm_linear(blas, xn_t, w.fc1_w, ffb, len, FF, D);
      k_bias_act<1><<<(len * FF + 255) / 256, 256>>>(ffb, w.fc1_b, len * FF,
                                                     FF);
      gemm_linear(blas, ffb, w.fc2_w, xn_t, len, D, FF);
      k_residual_bias<<<eltBlocks, 256>>>(x, xn_t, w.fc2_b, TD, D);
    }
    k_layernorm<true><<<len, 256>>>(x, enc_final_g, xn_t, len);
  }

  // Greedy decode against memory cross K/V. score_buf holds
  // H * max(mem_len, steps) floats. Returns tokens without BOS/EOS.
  std::vector<int> greedy_decode(int mem_len, float *const k_cross[6],
                                 float *const v_cross[6], int max_tokens,
                                 float *score_buf) {
    std::vector<int> out_tokens;
    int token = cfg.bos_id;
    max_tokens = std::min(max_tokens, MAX_TOKENS);
    for (int step = 0; step < max_tokens; step++) {
      CUDA_CHECK(cudaMemcpy(x, embed + (size_t)token * D, D * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      for (int l = 0; l < 6; l++) {
        const GpuDecLayer &w = dec[l];
        // self-attention
        k_layernorm<false><<<1, 256>>>(x, w.ln1_w, xn, 1);
        gemv_linear(blas, w.sq_w, xn, q, D, D);
        k_rope<<<1, 128>>>(q, step);
        gemv_linear(blas, w.sk_w, xn, kv, D, D);
        k_rope<<<1, 128>>>(kv, step);
        CUDA_CHECK(cudaMemcpy(k_self[l] + (size_t)step * D, kv,
                              D * sizeof(float), cudaMemcpyDeviceToDevice));
        gemv_linear(blas, w.sv_w, xn, kv, D, D);
        CUDA_CHECK(cudaMemcpy(v_self[l] + (size_t)step * D, kv,
                              D * sizeof(float), cudaMemcpyDeviceToDevice));
        cached_attention(q, k_self[l], v_self[l], step + 1, score_buf, attn);
        gemv_linear(blas, w.so_w, attn, proj, D, D);
        k_add<<<2, 256>>>(x, proj, D);

        // cross-attention
        k_layernorm<false><<<1, 256>>>(x, w.ln2_w, xn, 1);
        gemv_linear(blas, w.cq_w, xn, q, D, D);
        cached_attention(q, k_cross[l], v_cross[l], mem_len, score_buf, attn);
        gemv_linear(blas, w.co_w, attn, proj, D, D);
        k_add<<<2, 256>>>(x, proj, D);

        // SwiGLU MLP (bias fused into gemv via beta=1)
        k_layernorm<false><<<1, 256>>>(x, w.ln3_w, xn, 1);
        CUDA_CHECK(cudaMemcpy(ff, w.fc1_b, 2 * FF * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        gemv_linear(blas, w.fc1_w, xn, ff, 2 * FF, D, 1.0f);
        k_swiglu<<<(FF + 255) / 256, 256>>>(ff);
        CUDA_CHECK(cudaMemcpy(proj, w.fc2_b, D * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        gemv_linear(blas, w.fc2_w, ff, proj, D, FF, 1.0f);
        k_add<<<2, 256>>>(x, proj, D);
      }
      k_layernorm<false><<<1, 256>>>(x, dec_norm_w, xn, 1);
      gemv_linear(blas, proj_out, xn, logits, V, D);
      k_argmax<<<1, 1024>>>(logits, V, d_best);
      int best = 0;
      CUDA_CHECK(
          cudaMemcpy(&best, d_best, sizeof(int), cudaMemcpyDeviceToHost));
      if (best == cfg.eos_id) break;
      out_tokens.push_back(best);
      token = best;
    }
    return out_tokens;
  }
};

GpuModel::GpuModel(const Model &weights)
    : cfg(weights.cfg), impl(new GpuModelImpl(weights)) {}

GpuModel::~GpuModel() { delete impl; }

// ------------------------------------------------------------ GpuStream

struct GpuStreamImpl {
  GpuModelImpl &g;
  ModelConfig cfg;
  int max_t200, max_t100, max_t50;

  // Persistent device buffers for one segment.
  float *d_staging;            // raw samples of new complete frames
  float *h200, *h100, *f50;    // frontend activations (full history)
  float *memory;               // adapter output [t50, D]
  float *k_cross[6], *v_cross[6];
  // Encoder window scratch.
  float *win, *xn_t, *qb, *kb, *vb, *ab, *ffb, *scores;

  // Host-side state.
  std::vector<float> remainder;  // < 80 leftover samples
  int t200 = 0, t100 = 0, t50 = 0, emitted = 0;
  bool cross_valid = false;

  GpuStreamImpl(GpuModelImpl &g_, float max_seconds) : g(g_), cfg(g_.cfg) {
    max_t200 = (int)(max_seconds * 16000) / FRAME + 1;
    max_t100 = (max_t200 - 1) / 2 + 1;
    max_t50 = (max_t100 - 1) / 2 + 1;
    auto alloc = [](float **p, size_t n) {
      CUDA_CHECK(cudaMalloc(p, n * sizeof(float)));
    };
    alloc(&d_staging, (size_t)max_t200 * FRAME);
    alloc(&h200, (size_t)max_t200 * D);
    alloc(&h100, (size_t)max_t100 * 2 * D);
    alloc(&f50, (size_t)max_t50 * D);
    alloc(&memory, (size_t)max_t50 * D);
    for (int l = 0; l < 6; l++) {
      alloc(&k_cross[l], (size_t)max_t50 * D);
      alloc(&v_cross[l], (size_t)max_t50 * D);
    }
    alloc(&win, (size_t)max_t50 * D);
    alloc(&xn_t, (size_t)max_t50 * D);
    alloc(&qb, (size_t)max_t50 * D);
    alloc(&kb, (size_t)max_t50 * D);
    alloc(&vb, (size_t)max_t50 * D);
    alloc(&ab, (size_t)max_t50 * D);
    alloc(&ffb, (size_t)max_t50 * FF);
    alloc(&scores,
          (size_t)H * std::max(max_t50, GpuModelImpl::MAX_TOKENS + 1));
    remainder.reserve(FRAME);
  }

  ~GpuStreamImpl() {
    cudaFree(d_staging);
    cudaFree(h200);
    cudaFree(h100);
    cudaFree(f50);
    cudaFree(memory);
    for (int l = 0; l < 6; l++) {
      cudaFree(k_cross[l]);
      cudaFree(v_cross[l]);
    }
    cudaFree(win);
    cudaFree(xn_t);
    cudaFree(qb);
    cudaFree(kb);
    cudaFree(vb);
    cudaFree(ab);
    cudaFree(ffb);
    cudaFree(scores);
  }

  void reset() {
    remainder.clear();
    t200 = t100 = t50 = emitted = 0;
    cross_valid = false;
  }

  // Frontend on new samples only: frame-embed new 200 Hz frames, then extend
  // the conv pyramids. Causal stride-2 convs depend only on past rows of the
  // persistent buffers, so newly computable rows are final immediately.
  void add_audio(const float *audio, size_t n) {
    std::vector<float> buf;
    const float *src = audio;
    size_t avail = n;
    if (!remainder.empty()) {
      buf.assign(remainder.begin(), remainder.end());
      buf.insert(buf.end(), audio, audio + n);
      src = buf.data();
      avail = buf.size();
    }
    const int new_frames = (int)(avail / FRAME);
    const size_t used = (size_t)new_frames * FRAME;
    remainder.assign(src + used, src + avail);
    if (new_frames == 0) return;
    if (t200 + new_frames > max_t200)
      throw std::runtime_error("stream segment exceeds max_seconds");

    CUDA_CHECK(cudaMemcpy(d_staging, src, used * sizeof(float),
                          cudaMemcpyHostToDevice));
    k_frame_embed<<<new_frames, D>>>(d_staging, g.fe_linear_w, expf(g.log_k),
                                     h200 + (size_t)t200 * D);
    t200 += new_frames;

    const int new_t100 = (t200 - 1) / 2 + 1;
    if (new_t100 > t100) {
      k_causal_conv_s2<D, 2 * D, true>
          <<<new_t100 - t100, 2 * D, 5 * D * sizeof(float)>>>(
              h200, t200, g.conv1_wt, g.conv1_b, h100, t100);
      t100 = new_t100;
    }
    const int new_t50 = (t100 - 1) / 2 + 1;
    if (new_t50 > t50) {
      k_causal_conv_s2<2 * D, D, false>
          <<<new_t50 - t50, D, 5 * 2 * D * sizeof(float)>>>(
              h100, t100, g.conv2_wt, g.conv2_b, f50, t50);
      t50 = new_t50;
    }
  }

  // Re-encode a window with 96 frames of left context and emit newly
  // stabilized frames into memory (holding back 16 lookahead frames unless
  // final). Mirrors MoonshineStreamingModel::encode.
  void encode(bool is_final) {
    const int stable = is_final ? t50 : std::max(0, t50 - 16);
    if (stable <= emitted) return;
    if (stable > cfg.max_positions)
      throw std::runtime_error("segment exceeds pos_emb table");
    const bool attr = std::getenv("MOONSHINE_ENC_ATTR") != nullptr;
    auto mark = [&](const char *what, auto &&fn) {
      if (!attr) { fn(); return; }
      cudaDeviceSynchronize();
      auto t0 = std::chrono::steady_clock::now();
      fn();
      cudaDeviceSynchronize();
      fprintf(stderr, " %s %.0fus", what,
              1e6 * std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count());
    };
    const int ws = std::max(0, emitted - 96);
    const int len = t50 - ws;
    mark("wincpy", [&] {
      CUDA_CHECK(cudaMemcpy(win, f50 + (size_t)ws * D,
                            (size_t)len * D * sizeof(float),
                            cudaMemcpyDeviceToDevice));
    });
    mark("kernel", [&] { g.encoder_window(win, len, xn_t, qb, kb, vb, ab, ffb); });
    const int new_frames = stable - emitted;
    mark("emit", [&] {
      CUDA_CHECK(cudaMemcpy(memory + (size_t)emitted * D,
                            xn_t + (size_t)(emitted - ws) * D,
                            (size_t)new_frames * D * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      k_add_pos<<<(new_frames * D + 255) / 256, 256>>>(
          memory + (size_t)emitted * D, g.pos_emb + (size_t)emitted * D,
          new_frames * D);
    });
    if (attr) fprintf(stderr, "\n");
    emitted = stable;
    cross_valid = false;
  }

  std::vector<int> decode(int max_tokens) {
    if (emitted == 0) return {};
    if (!cross_valid) {
      for (int l = 0; l < 6; l++) {
        gemm_linear(g.blas, memory, g.dec[l].ck_w, k_cross[l], emitted, D, D);
        gemm_linear(g.blas, memory, g.dec[l].cv_w, v_cross[l], emitted, D, D);
      }
      cross_valid = true;
    }
    return g.greedy_decode(emitted, k_cross, v_cross, max_tokens, scores);
  }
};

GpuStream::GpuStream(GpuModel &model, float max_seconds)
    : impl(new GpuStreamImpl(*model.impl, max_seconds)) {}
GpuStream::~GpuStream() { delete impl; }
void GpuStream::reset() { impl->reset(); }
void GpuStream::add_audio(const float *audio, size_t n) {
  impl->add_audio(audio, n);
}
void GpuStream::encode(bool is_final) { impl->encode(is_final); }
std::vector<int> GpuStream::decode(int max_tokens) {
  return impl->decode(max_tokens);
}
int GpuStream::memory_frames() const { return impl->emitted; }

std::vector<int> GpuModel::transcribe(const float *audio, size_t n_samples) {
  const float dur = n_samples / 16000.0f;
  if ((int)(n_samples / FRAME) < 4) throw std::runtime_error("audio too short");
  GpuStream stream(*this, dur + 1.0f);
  stream.add_audio(audio, n_samples);
  stream.encode(true);
  int max_tokens = std::min(cfg.max_tokens_cap,
                            (int)ceilf(dur * cfg.max_tokens_per_second));
  return stream.decode(max_tokens);
}
