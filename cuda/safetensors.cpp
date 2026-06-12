#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace {

// Tiny scanner for the safetensors JSON header. The header is machine-written
// flat JSON: {"name": {"dtype": "F32", "shape": [..], "data_offsets": [a, b]},
// ..., "__metadata__": {...}}. We only need string keys, string values,
// integer arrays, and the ability to skip nested objects.
struct Scanner {
  const char *p;
  const char *end;

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
      p++;
  }
  bool eat(char c) {
    skip_ws();
    if (p < end && *p == c) {
      p++;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!eat(c))
      throw std::runtime_error(std::string("safetensors header: expected '") +
                               c + "'");
  }
  std::string string() {
    expect('"');
    std::string s;
    while (p < end && *p != '"') {
      if (*p == '\\' && p + 1 < end) p++;
      s.push_back(*p++);
    }
    expect('"');
    return s;
  }
  int64_t integer() {
    skip_ws();
    bool neg = eat('-');
    int64_t v = 0;
    while (p < end && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    return neg ? -v : v;
  }
  // Skip any value (used for __metadata__).
  void skip_value() {
    skip_ws();
    if (*p == '"') {
      string();
    } else if (*p == '{' || *p == '[') {
      char open = *p, close = (*p == '{') ? '}' : ']';
      int depth = 0;
      bool in_str = false;
      while (p < end) {
        if (in_str) {
          if (*p == '\\')
            p++;
          else if (*p == '"')
            in_str = false;
        } else if (*p == '"') {
          in_str = true;
        } else if (*p == open) {
          depth++;
        } else if (*p == close) {
          if (--depth == 0) {
            p++;
            return;
          }
        }
        p++;
      }
      throw std::runtime_error("safetensors header: unterminated value");
    } else {
      while (p < end && *p != ',' && *p != '}' && *p != ']') p++;
    }
  }
};

}  // namespace

SafeTensors::SafeTensors(const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0)
    throw std::runtime_error(std::string("cannot open ") + path);
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error(std::string("cannot stat ") + path);
  }
  mapped_size = (size_t)st.st_size;
  mapped = (const uint8_t *)mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE,
                                 fd, 0);
  close(fd);
  if (mapped == MAP_FAILED)
    throw std::runtime_error(std::string("cannot mmap ") + path);

  uint64_t header_len;
  memcpy(&header_len, mapped, 8);
  if (8 + header_len > mapped_size)
    throw std::runtime_error("safetensors: header overruns file");
  const uint8_t *payload = mapped + 8 + header_len;
  const size_t payload_size = mapped_size - 8 - header_len;

  Scanner s{(const char *)(mapped + 8), (const char *)(mapped + 8 + header_len)};
  s.expect('{');
  if (s.eat('}')) return;
  do {
    std::string key = s.string();
    s.expect(':');
    if (key == "__metadata__") {
      s.skip_value();
      continue;
    }
    StTensor t;
    int64_t begin = -1, end_off = -1;
    s.expect('{');
    do {
      std::string field = s.string();
      s.expect(':');
      if (field == "dtype") {
        std::string dtype = s.string();
        if (dtype != "F32")
          throw std::runtime_error("tensor " + key + ": unsupported dtype " +
                                   dtype);
      } else if (field == "shape") {
        s.expect('[');
        if (!s.eat(']')) {
          do {
            if (t.ndim >= 4)
              throw std::runtime_error("tensor " + key + ": ndim > 4");
            t.shape[t.ndim++] = s.integer();
          } while (s.eat(','));
          s.expect(']');
        }
      } else if (field == "data_offsets") {
        s.expect('[');
        begin = s.integer();
        s.expect(',');
        end_off = s.integer();
        s.expect(']');
      } else {
        s.skip_value();
      }
    } while (s.eat(','));
    s.expect('}');
    if (begin < 0 || end_off < begin || (size_t)end_off > payload_size)
      throw std::runtime_error("tensor " + key + ": bad data_offsets");
    if ((end_off - begin) != t.numel() * (int64_t)sizeof(float))
      throw std::runtime_error("tensor " + key + ": size mismatch");
    t.data = (const float *)(payload + begin);
    tensors.emplace(std::move(key), t);
  } while (s.eat(','));
  s.expect('}');
}

SafeTensors::~SafeTensors() {
  if (mapped && mapped != MAP_FAILED)
    munmap((void *)mapped, mapped_size);
}

const StTensor &SafeTensors::get(const std::string &name,
                                 int expected_ndim) const {
  auto it = tensors.find(name);
  if (it == tensors.end())
    throw std::runtime_error("missing tensor: " + name);
  if (it->second.ndim != expected_ndim)
    throw std::runtime_error("tensor " + name + ": expected ndim " +
                             std::to_string(expected_ndim) + ", got " +
                             std::to_string(it->second.ndim));
  return it->second;
}
