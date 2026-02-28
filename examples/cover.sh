#!/bin/bash
# Cover mode: decode precomputed audio_codes to WAV (no LLM).
# Use cover.json as-is, or replace audio_codes with output from a previous run:
#   ../build/ace-qwen3 --request simple.json --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf
#   # then use simple0.json as input, or copy its audio_codes into cover.json
set -eu
cd "$(dirname "$0")"

../build/dit-vae \
    --request cover.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf

echo "Done. Check cover0.wav"
