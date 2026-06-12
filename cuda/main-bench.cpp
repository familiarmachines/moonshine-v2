// Model-only streaming benchmark for moonshine-streaming-tiny in C/CUDA,
// mirroring core/benchmark.cpp semantics minus VAD: audio is fed in 80 ms
// chunks, split into fixed-length segments; within each segment the encoder
// runs incrementally and the transcript is refreshed at the update interval;
// at segment end the lookahead is flushed and a final decode runs.
//
// Reported latency = wall time of the segment-end flush + decode, i.e. the
// time between "the user stopped talking" and the final transcript -- the
// same metric core/build/benchmark reports.
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"
#include "model_cuda.h"
#include "tokenizer.h"
#include "wav.h"

namespace {

double now_secs() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string model_dir, wav_path;
  float segment_secs = 10.0f, update_secs = 0.5f, chunk_ms = 80.0f;
  for (int i = 1; i < argc; i++) {
    auto arg_value = [&](const char *name) -> const char * {
      if (strcmp(argv[i], name) == 0 && i + 1 < argc) return argv[++i];
      return nullptr;
    };
    if (const char *v = arg_value("--segment-duration")) {
      segment_secs = atof(v);
    } else if (const char *v = arg_value("--update-interval")) {
      update_secs = atof(v);
    } else if (const char *v = arg_value("--chunk-ms")) {
      chunk_ms = atof(v);
    } else if (model_dir.empty()) {
      model_dir = argv[i];
    } else if (wav_path.empty()) {
      wav_path = argv[i];
    }
  }
  if (model_dir.empty() || wav_path.empty()) {
    fprintf(stderr,
            "usage: %s <model_dir> <wav> [--segment-duration S] "
            "[--update-interval S] [--chunk-ms MS]\n",
            argv[0]);
    return 1;
  }

  try {
    Model weights(model_dir + "/model.safetensors");
    Tokenizer tokenizer((model_dir + "/tokenizer.bin").c_str());
    GpuModel model(weights);
    GpuStream stream(model, segment_secs + 1.0f);
    std::vector<float> audio = load_wav_16k_mono(wav_path.c_str());
    const double audio_dur = audio.size() / 16000.0;

    // Warm-up.
    {
      std::vector<float> warm(audio.begin(),
                              audio.begin() + std::min(audio.size(),
                                                       (size_t)32000));
      stream.reset();
      stream.add_audio(warm.data(), warm.size());
      stream.encode(true);
      stream.decode(16);
    }

    const size_t chunk = (size_t)(16000.0f * chunk_ms / 1000.0f);
    const size_t seg_samples = (size_t)(segment_secs * 16000.0f);
    double total_compute = 0.0, latency_sum = 0.0;
    double t_frontend = 0.0, t_encode = 0.0, t_decode = 0.0;
    int n_segments = 0, n_encodes = 0;

    for (size_t seg0 = 0; seg0 < audio.size(); seg0 += seg_samples) {
      const size_t seg1 = std::min(audio.size(), seg0 + seg_samples);
      const double seg_dur = (seg1 - seg0) / 16000.0;
      const int cap = std::min(
          256, (int)ceilf((float)seg_dur * model.cfg.max_tokens_per_second));
      stream.reset();

      double since_update = 0.0;
      for (size_t pos = seg0; pos < seg1; pos += chunk) {
        const size_t n = std::min(chunk, seg1 - pos);
        double t0 = now_secs();
        stream.add_audio(audio.data() + pos, n);
        cudaDeviceSynchronize();
        double t1 = now_secs();
        t_frontend += t1 - t0;
        since_update += n / 16000.0;
        if (since_update >= update_secs && pos + n < seg1) {
          stream.encode(false);
          cudaDeviceSynchronize();
          double t2 = now_secs();
          t_encode += t2 - t1;
          n_encodes++;
          stream.decode(cap);  // interim transcript, as the library would
          t_decode += now_secs() - t2;
          since_update = 0.0;
        }
        total_compute += now_secs() - t0;
      }

      // Segment end: flush lookahead, final decode. This is the latency the
      // user perceives between end-of-speech and final text.
      double t0 = now_secs();
      stream.encode(true);
      cudaDeviceSynchronize();
      double t1 = now_secs();
      t_encode += t1 - t0;
      n_encodes++;
      std::vector<int> tokens = stream.decode(cap);
      const double t2 = now_secs();
      t_decode += t2 - t1;
      const double latency = t2 - t0;
      total_compute += latency;
      latency_sum += latency;
      n_segments++;

      printf("[%6.1fs] (%.1fms) %s\n", seg0 / 16000.0, latency * 1000.0,
             tokenizer.decode(tokens).c_str());
    }

    printf("\n");
    printf("Audio duration:      %.2fs\n", audio_dur);
    printf("Transcription took:  %.2fs (%.2f%% of audio duration)\n",
           total_compute, 100.0 * total_compute / audio_dur);
    printf("Average latency:     %.0fms over %d segments\n",
           1000.0 * latency_sum / n_segments, n_segments);
    printf("Phase breakdown:     frontend %.0fms | encode %.0fms (%d calls, "
           "%.2fms/call) | decode %.0fms\n",
           1000.0 * t_frontend, 1000.0 * t_encode, n_encodes,
           1000.0 * t_encode / n_encodes, 1000.0 * t_decode);
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
