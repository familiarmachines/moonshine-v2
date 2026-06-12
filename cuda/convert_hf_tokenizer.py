#!/usr/bin/env python3
"""Convert a HuggingFace tokenizer.json to BinTokenizer format, decoding
<0xNN> byte-fallback tokens to raw bytes (matching the tokenizer.bin files
shipped with moonshine-v2 models). Stdlib only.

Usage: python3 convert_hf_tokenizer.py tokenizer.json tokenizer.bin
"""
import json
import re
import sys

VOCAB_SIZE = 32768


def token_bytes(s: str) -> bytes:
    m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", s)
    return bytes([int(m.group(1), 16)]) if m else s.encode("utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    tokens = [b""] * VOCAB_SIZE
    for tok, idx in data["model"]["vocab"].items():
        if idx < VOCAB_SIZE:
            tokens[idx] = token_bytes(tok)
    for added in data.get("added_tokens", []):
        if added["id"] < VOCAB_SIZE:
            tokens[added["id"]] = token_bytes(added["content"])
    with open(sys.argv[2], "wb") as f:
        for tb in tokens:
            n = len(tb)
            if n == 0:
                f.write(b"\x00")
            elif n < 128:
                f.write(bytes([n]) + tb)
            else:
                f.write(bytes([(n % 128) + 128, n // 128]) + tb)
    print(f"wrote {sys.argv[2]} ({VOCAB_SIZE} tokens)")


if __name__ == "__main__":
    main()
