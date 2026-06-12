#include "tokenizer.h"

#include <cstdio>
#include <stdexcept>

Tokenizer::Tokenizer(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> data((size_t)size);
  if (fread(data.data(), 1, (size_t)size, f) != (size_t)size) {
    fclose(f);
    throw std::runtime_error(std::string("cannot read ") + path);
  }
  fclose(f);

  size_t i = 0;
  while (i < data.size()) {
    size_t len = data[i++];
    if (len >= 128) {
      if (i >= data.size()) throw std::runtime_error("truncated tokenizer");
      len = data[i++] * 128 + len - 128;
    }
    if (i + len > data.size()) throw std::runtime_error("truncated tokenizer");
    id_to_bytes.emplace_back((const char *)&data[i], len);
    i += len;
  }
}

std::string Tokenizer::decode(const std::vector<int> &ids) const {
  static const std::string kSpace = "\xe2\x96\x81";  // "▁"
  std::string raw;
  for (int id : ids) {
    if (id < 0 || (size_t)id >= id_to_bytes.size()) continue;
    const std::string &b = id_to_bytes[id];
    if (b.size() > 2 && b.front() == '<' && b.back() == '>') continue;
    raw += b;
  }
  std::string out;
  for (size_t i = 0; i < raw.size();) {
    if (raw.compare(i, kSpace.size(), kSpace) == 0) {
      out += ' ';
      i += kSpace.size();
    } else {
      out += raw[i++];
    }
  }
  size_t begin = out.find_first_not_of(' ');
  size_t end = out.find_last_not_of(' ');
  if (begin == std::string::npos) return "";
  return out.substr(begin, end - begin + 1);
}
