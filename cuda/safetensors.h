// Minimal mmap-based safetensors reader (fp32 only).
#ifndef MOONSHINE_CUDA_SAFETENSORS_H
#define MOONSHINE_CUDA_SAFETENSORS_H

#include <cstdint>
#include <map>
#include <string>

struct StTensor {
  const float *data = nullptr;
  int64_t shape[4] = {0, 0, 0, 0};
  int ndim = 0;
  int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
  }
};

struct SafeTensors {
  const uint8_t *mapped = nullptr;
  size_t mapped_size = 0;
  std::map<std::string, StTensor> tensors;

  // Throws std::runtime_error on failure.
  explicit SafeTensors(const char *path);
  ~SafeTensors();

  // Fetch a tensor, checking dimensionality. Throws if missing or wrong ndim.
  const StTensor &get(const std::string &name, int expected_ndim) const;
};

#endif
