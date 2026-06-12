// CUDA fp32: transcribe a 16 kHz mono WAV with moonshine-streaming-tiny
// weights from HF safetensors. Same CLI as moonshine-ref.
#include <chrono>
#include <cstdio>
#include <string>

#include "model.h"
#include "model_cuda.h"
#include "tokenizer.h"
#include "wav.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <model_dir> <wav_path>\n", argv[0]);
    return 1;
  }
  const std::string model_dir = argv[1];
  try {
    Model weights(model_dir + "/model.safetensors");
    Tokenizer tokenizer((model_dir + "/tokenizer.bin").c_str());
    GpuModel model(weights);
    std::vector<float> audio = load_wav_16k_mono(argv[2]);
    const float duration = audio.size() / 16000.0f;

    // Warm-up run (context/kernel compilation), then timed run.
    model.transcribe(audio.data(), std::min(audio.size(), (size_t)32000));
    auto start = std::chrono::steady_clock::now();
    std::vector<int> tokens = model.transcribe(audio.data(), audio.size());
    auto end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();

    printf("%s\n", tokenizer.decode(tokens).c_str());
    fprintf(stderr,
            "[%zu tokens, %.1fs audio, %.3fs compute, %.1f%% of real time]\n",
            tokens.size(), duration, secs, 100.0 * secs / duration);
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
