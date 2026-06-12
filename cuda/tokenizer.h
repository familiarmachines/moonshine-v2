// Decoder-only tokenizer for the BinTokenizer format (length-prefixed token
// byte strings, SentencePiece-style "▁" word boundaries).
#ifndef MOONSHINE_CUDA_TOKENIZER_H
#define MOONSHINE_CUDA_TOKENIZER_H

#include <cstdint>
#include <string>
#include <vector>

struct Tokenizer {
  std::vector<std::string> id_to_bytes;

  explicit Tokenizer(const char *path);  // throws on failure
  // Concatenate token bytes, skipping <...> specials, "▁" -> " ", trimmed.
  std::string decode(const std::vector<int> &ids) const;
};

#endif
