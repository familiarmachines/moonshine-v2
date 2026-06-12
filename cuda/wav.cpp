#include "wav.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

std::vector<float> load_wav_16k_mono(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  auto fail = [&](const char *msg) {
    fclose(f);
    throw std::runtime_error(std::string(path) + ": " + msg);
  };

  uint8_t riff[12];
  if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 ||
      memcmp(riff + 8, "WAVE", 4) != 0)
    fail("not a RIFF/WAVE file");

  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  std::vector<float> samples;
  uint8_t hdr[8];
  while (fread(hdr, 1, 8, f) == 8) {
    uint32_t chunk_size;
    memcpy(&chunk_size, hdr + 4, 4);
    if (memcmp(hdr, "fmt ", 4) == 0) {
      uint8_t fmt[16];
      if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16) fail("bad fmt chunk");
      memcpy(&format, fmt, 2);
      memcpy(&channels, fmt + 2, 2);
      memcpy(&rate, fmt + 4, 4);
      memcpy(&bits, fmt + 14, 2);
      if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
    } else if (memcmp(hdr, "data", 4) == 0) {
      if (format != 1 || channels != 1 || rate != 16000 || bits != 16)
        fail("need mono PCM16 @ 16 kHz");
      std::vector<int16_t> pcm(chunk_size / 2);
      if (fread(pcm.data(), 1, chunk_size, f) != chunk_size)
        fail("truncated data chunk");
      samples.resize(pcm.size());
      for (size_t i = 0; i < pcm.size(); i++) samples[i] = pcm[i] / 32768.0f;
      break;
    } else {
      fseek(f, chunk_size + (chunk_size & 1), SEEK_CUR);
    }
  }
  fclose(f);
  if (samples.empty()) fail("no data chunk");
  return samples;
}
