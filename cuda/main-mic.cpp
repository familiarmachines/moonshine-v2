// Live microphone transcriber for moonshine-streaming-tiny on the CUDA
// encoder+decoder. Captures 16 kHz mono audio (ALSA), segments it with a
// simple adaptive energy VAD, and drives GpuStream incrementally: the encoder
// runs as audio arrives, interim transcripts refresh on a fixed cadence, and
// the lookahead is flushed for a final decode at end of utterance.
//
// The cuda/ engine intentionally has no VAD (see ../INTEGRATION.md), so the
// energy gate here is a pragmatic v1 stand-in for Silero, not a port of it.
//
// The same feed()/segmentation/transcription core is driven by either the
// microphone (--device, default) or a WAV file (--wav PATH). The WAV path
// exercises the entire pipeline identically to the mic, so it can be tested
// on machines without a capture device.
//
// Usage:
//   moonshine-mic <model_dir> [--device NAME] [--wav PATH] [--meter]
//                 [--list-devices] [--vad-threshold RMS] [--hangover S]
//                 [--max-segment S] [--update-interval S]
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"
#include "model_cuda.h"
#include "tokenizer.h"
#include "wav.h"

#ifdef MOONSHINE_HAVE_ALSA
#include <alsa/asoundlib.h>
#endif

namespace {

constexpr int SR = 16000;
constexpr int VAD_FRAME = 480;  // 30 ms VAD granularity

std::atomic<bool> g_stop{false};
void on_sigint(int) { g_stop.store(true); }

float rms(const float *x, size_t n) {
  double acc = 0.0;
  for (size_t i = 0; i < n; i++) acc += (double)x[i] * x[i];
  return (float)std::sqrt(acc / (double)n);
}

// Live transcript line that overwrites in place until finalized.
class LinePrinter {
  size_t last_len_ = 0;

 public:
  void partial(const std::string &text) {
    std::string out = "  " + text;
    fputc('\r', stdout);
    fputs(out.c_str(), stdout);
    for (size_t i = out.size(); i < last_len_; i++) fputc(' ', stdout);
    last_len_ = out.size();
    fflush(stdout);
  }
  void final(const std::string &text) {
    std::string out = "> " + text;
    fputc('\r', stdout);
    fputs(out.c_str(), stdout);
    for (size_t i = out.size(); i < last_len_; i++) fputc(' ', stdout);
    fputc('\n', stdout);
    last_len_ = 0;
    fflush(stdout);
  }
};

struct Config {
  float vad_threshold = 0.012f;  // RMS floor for "voiced" (normalized audio)
  float vad_factor = 3.0f;       // voiced if rms > noise_floor * factor
  float onset_secs = 0.12f;      // voiced run needed to open a segment
  float hangover_secs = 0.7f;    // trailing silence that ends a segment
  float max_segment_secs = 15.0f;
  float update_interval_secs = 0.4f;
  float preroll_secs = 0.3f;     // audio kept before onset (avoid clipping)
};

class LiveTranscriber {
 public:
  LiveTranscriber(GpuModel &model, Tokenizer &tok, const Config &cfg)
      : model_(model),
        tok_(tok),
        cfg_(cfg),
        stream_(model, cfg.max_segment_secs + 1.0f) {
    preroll_cap_ = (size_t)(cfg.preroll_secs * SR);
    noise_floor_ = cfg.vad_threshold;
  }

  // Process an arbitrary-length chunk of 16 kHz mono float audio.
  void feed(const float *x, size_t n) {
    pending_.insert(pending_.end(), x, x + n);
    size_t off = 0;
    while (pending_.size() - off >= (size_t)VAD_FRAME) {
      process_frame(pending_.data() + off);
      off += VAD_FRAME;
    }
    pending_.erase(pending_.begin(), pending_.begin() + off);
  }

  void finalize_if_active() {
    if (active_) finalize();
  }

 private:
  int token_cap() const {
    double dur = seg_samples_ / (double)SR;
    int cap = (int)std::ceil(dur * model_.cfg.max_tokens_per_second);
    return std::max(8, std::min(256, cap));
  }

  void process_frame(const float *frame) {
    const float level = rms(frame, VAD_FRAME);
    const float thresh = std::max(cfg_.vad_threshold, noise_floor_ * cfg_.vad_factor);
    const bool voiced = level > thresh;

    // Rolling pre-roll history of recent audio.
    preroll_.insert(preroll_.end(), frame, frame + VAD_FRAME);
    if (preroll_.size() > preroll_cap_)
      preroll_.erase(preroll_.begin(), preroll_.end() - preroll_cap_);

    if (!active_) {
      if (!voiced)  // adapt the noise floor only on quiet frames
        noise_floor_ = 0.95f * noise_floor_ + 0.05f * level;
      voiced_run_ = voiced ? voiced_run_ + VAD_FRAME : 0;
      if (voiced_run_ >= (size_t)(cfg_.onset_secs * SR)) {
        // Open a segment, priming it with the pre-roll so the onset isn't clipped.
        stream_.reset();
        seg_samples_ = 0;
        silence_samples_ = 0;
        since_update_ = 0;
        stream_.add_audio(preroll_.data(), preroll_.size());
        seg_samples_ += preroll_.size();
        active_ = true;
      }
      return;
    }

    // Active segment: feed audio, refresh interim transcript, watch for EOU.
    stream_.add_audio(frame, VAD_FRAME);
    seg_samples_ += VAD_FRAME;
    since_update_ += VAD_FRAME;
    silence_samples_ = voiced ? 0 : silence_samples_ + VAD_FRAME;

    if (since_update_ >= (size_t)(cfg_.update_interval_secs * SR)) {
      stream_.encode(false);
      printer_.partial(tok_.decode(stream_.decode(token_cap())));
      since_update_ = 0;
    }
    if (silence_samples_ >= (size_t)(cfg_.hangover_secs * SR) ||
        seg_samples_ >= (size_t)(cfg_.max_segment_secs * SR)) {
      finalize();
    }
  }

  void finalize() {
    stream_.encode(true);
    std::string text = tok_.decode(stream_.decode(token_cap()));
    if (!text.empty())
      printer_.final(text);
    else
      printer_.partial("");  // clear an empty partial line
    active_ = false;
    voiced_run_ = 0;
  }

  GpuModel &model_;
  Tokenizer &tok_;
  Config cfg_;
  GpuStream stream_;
  LinePrinter printer_;

  std::vector<float> pending_;
  std::vector<float> preroll_;
  size_t preroll_cap_;
  bool active_ = false;
  size_t voiced_run_ = 0;
  size_t seg_samples_ = 0;
  size_t silence_samples_ = 0;
  size_t since_update_ = 0;
  float noise_floor_;
};

// Live RMS meter (helps pick --vad-threshold without a transcription run).
void run_meter(const float *x, size_t n) {
  for (size_t i = 0; i + VAD_FRAME <= n; i += VAD_FRAME) {
    float level = rms(x + i, VAD_FRAME);
    int bars = std::min(50, (int)(level * 400));
    fprintf(stderr, "\rRMS %.4f |%s%s", level, std::string(bars, '#').c_str(),
            std::string(50 - bars, ' ').c_str());
  }
  fprintf(stderr, "\n");
}

#ifdef MOONSHINE_HAVE_ALSA
void list_devices() {
  void **hints = nullptr;
  if (snd_device_name_hint(-1, "pcm", &hints) != 0) return;
  printf("Capture devices:\n");
  for (void **h = hints; *h; h++) {
    char *name = snd_device_name_get_hint(*h, "NAME");
    char *ioid = snd_device_name_get_hint(*h, "IOID");
    char *desc = snd_device_name_get_hint(*h, "DESC");
    if (!ioid || strcmp(ioid, "Input") == 0)
      printf("  %s  (%s)\n", name ? name : "?", desc ? desc : "");
    free(name);
    free(ioid);
    free(desc);
  }
  snd_device_name_free_hint(hints);
}

// Open ALSA capture as 16 kHz mono S16_LE (soft-resampling enabled).
snd_pcm_t *open_capture(const char *device) {
  snd_pcm_t *pcm = nullptr;
  int err = snd_pcm_open(&pcm, device, SND_PCM_STREAM_CAPTURE, 0);
  if (err < 0) {
    fprintf(stderr, "cannot open capture device '%s': %s\n", device,
            snd_strerror(err));
    return nullptr;
  }
  err = snd_pcm_set_params(pcm, SND_PCM_FORMAT_S16_LE,
                           SND_PCM_ACCESS_RW_INTERLEAVED, 1, SR, 1, 200000);
  if (err < 0) {
    fprintf(stderr, "cannot configure capture (16 kHz mono): %s\n",
            snd_strerror(err));
    snd_pcm_close(pcm);
    return nullptr;
  }
  return pcm;
}

// Returns false on unrecoverable error or stop.
bool capture_loop(snd_pcm_t *pcm, bool meter, LiveTranscriber *tr) {
  const int frames = 1600;  // 100 ms
  std::vector<int16_t> pcm_buf(frames);
  std::vector<float> f(frames);
  while (!g_stop.load()) {
    snd_pcm_sframes_t r = snd_pcm_readi(pcm, pcm_buf.data(), frames);
    if (r == -EPIPE || r == -ESTRPIPE) {
      snd_pcm_recover(pcm, (int)r, 1);
      continue;
    }
    if (r < 0) {
      fprintf(stderr, "\ncapture error: %s\n", snd_strerror((int)r));
      return false;
    }
    for (snd_pcm_sframes_t i = 0; i < r; i++) f[i] = pcm_buf[i] / 32768.0f;
    if (meter)
      run_meter(f.data(), (size_t)r);
    else
      tr->feed(f.data(), (size_t)r);
  }
  return true;
}
#endif  // MOONSHINE_HAVE_ALSA

}  // namespace

int main(int argc, char *argv[]) {
  std::string model_dir, device = "default", wav_path;
  bool meter = false, list = false;
  Config cfg;
  for (int i = 1; i < argc; i++) {
    auto val = [&](const char *name) -> const char * {
      return (strcmp(argv[i], name) == 0 && i + 1 < argc) ? argv[++i] : nullptr;
    };
    if (const char *v = val("--device"))
      device = v;
    else if (const char *v = val("--wav"))
      wav_path = v;
    else if (const char *v = val("--vad-threshold"))
      cfg.vad_threshold = atof(v);
    else if (const char *v = val("--hangover"))
      cfg.hangover_secs = atof(v);
    else if (const char *v = val("--max-segment"))
      cfg.max_segment_secs = atof(v);
    else if (const char *v = val("--update-interval"))
      cfg.update_interval_secs = atof(v);
    else if (strcmp(argv[i], "--meter") == 0)
      meter = true;
    else if (strcmp(argv[i], "--list-devices") == 0)
      list = true;
    else if (model_dir.empty())
      model_dir = argv[i];
  }

#ifdef MOONSHINE_HAVE_ALSA
  if (list) {
    list_devices();
    return 0;
  }
#else
  if (list) {
    fprintf(stderr, "built without ALSA; --list-devices unavailable\n");
    return 1;
  }
#endif

  if (model_dir.empty()) {
    fprintf(stderr,
            "usage: %s <model_dir> [--device NAME | --wav PATH] [--meter]\n"
            "         [--list-devices] [--vad-threshold RMS] [--hangover S]\n"
            "         [--max-segment S] [--update-interval S]\n",
            argv[0]);
    return 1;
  }

  try {
    Model weights(model_dir + "/model.safetensors");
    Tokenizer tokenizer((model_dir + "/tokenizer.bin").c_str());
    GpuModel model(weights);

    // Make it obvious the encoder+decoder run on the GPU.
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
      fprintf(stderr, "CUDA device: %s (%d SMs, cc %d.%d) — %s encoder\n",
              prop.name, prop.multiProcessorCount, prop.major, prop.minor,
              getenv("MOONSHINE_NO_FUSED") ? "unfused cuBLAS"
                                           : "fused tensor-core");

    // Warm up the GPU pipeline (first cooperative launch is slow).
    {
      GpuStream warm(model, 2.5f);
      std::vector<float> z(SR, 0.0f);
      warm.add_audio(z.data(), z.size());
      warm.encode(true);
      warm.decode(8);
    }

    signal(SIGINT, on_sigint);
    LiveTranscriber tr(model, tokenizer, cfg);

    if (!wav_path.empty()) {
      // File source: drive the identical feed()/segmentation path. Audio-time
      // logic is pacing-independent, so we run flat out (no real-time sleep).
      std::vector<float> audio = load_wav_16k_mono(wav_path.c_str());
      if (meter) {
        run_meter(audio.data(), audio.size());
        return 0;
      }
      fprintf(stderr, "transcribing %s (%.1fs)...\n", wav_path.c_str(),
              audio.size() / (double)SR);
      const size_t chunk = 1600;
      for (size_t p = 0; p < audio.size() && !g_stop.load(); p += chunk)
        tr.feed(audio.data() + p, std::min(chunk, audio.size() - p));
      tr.finalize_if_active();
      return 0;
    }

#ifdef MOONSHINE_HAVE_ALSA
    snd_pcm_t *pcm = open_capture(device.c_str());
    if (!pcm) {
      fprintf(stderr, "\nNo usable capture device. Try --list-devices, or\n"
                      "test the pipeline with --wav <file>.\n");
      return 1;
    }
    if (meter)
      fprintf(stderr, "RMS meter (Ctrl-C to stop). Speak to calibrate "
                      "--vad-threshold.\n");
    else
      fprintf(stderr, "Listening on '%s' (Ctrl-C to stop). vad-threshold=%.4f\n",
              device.c_str(), cfg.vad_threshold);
    capture_loop(pcm, meter, &tr);
    snd_pcm_drop(pcm);
    snd_pcm_close(pcm);
    if (!meter) tr.finalize_if_active();
    fprintf(stderr, "\nstopped.\n");
    return 0;
#else
    fprintf(stderr, "built without ALSA: microphone capture unavailable.\n"
                    "Use --wav <file> to test the transcription pipeline.\n");
    return 1;
#endif
  } catch (const std::exception &e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
