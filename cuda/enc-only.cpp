// Encode-only loop for profiling: simulates steady-state streaming ticks.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include "model.h"
#include "model_cuda.h"
#include "wav.h"
int main(int argc, char **argv) {
  Model weights(std::string(argv[1]) + "/model.safetensors");
  GpuModel model(weights);
  GpuStream stream(model, 12.0f);
  std::vector<float> audio = load_wav_16k_mono(argv[2]);
  // Build steady state: feed 3.5s, encode
  stream.add_audio(audio.data(), 56000);
  stream.encode(false);
  cudaDeviceSynchronize();
  size_t pos = 56000;
  const int iters = (argc > 3) ? atoi(argv[3]) : 8;
  double total = 0;
  for (int i = 0; i < iters; i++) {
    if (pos + 8000 > 56000 + 15 * 8000 || pos + 8000 > audio.size()) {
      // Stream near capacity (or end of file): rebuild steady state.
      stream.reset();
      stream.add_audio(audio.data(), 56000);
      stream.encode(false);
      pos = 56000;
    }
    stream.add_audio(audio.data() + pos, 8000);  // 0.5s
    pos += 8000;
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    stream.encode(false);
    cudaDeviceSynchronize();
    total += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  printf("encode: %.2f ms/call (%d calls)\n", 1000.0 * total / iters, iters);
  return 0;
}
