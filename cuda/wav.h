// Minimal RIFF/WAV reader: 16 kHz mono PCM16 -> float32 in [-1, 1].
#ifndef MOONSHINE_CUDA_WAV_H
#define MOONSHINE_CUDA_WAV_H

#include <vector>

// Throws std::runtime_error unless the file is mono PCM16 @ 16000 Hz.
std::vector<float> load_wav_16k_mono(const char *path);

#endif
