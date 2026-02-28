#!/bin/bash
# Cover mode with reference timbre: audio_codes + reference_audio (WAV or MP3).
# Put a WAV/MP3 at reference.wav (or reference.mp3) or set reference_audio in cover-reference.json.
# Requires VAE GGUF with encoder weights (same as request-reference / test-reference).
set -eu
cd "$(dirname "$0")"

if [ ! -f "reference.wav" ] && [ ! -f "reference.mp3" ]; then
    echo "No reference.wav or reference.mp3 found. Copy a file to reference.wav (or .mp3), or set reference_audio in cover-reference.json."
    echo "Then run: $0"
    exit 1
fi

../build/dit-vae \
    --request cover-reference.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf

echo "Done. Check cover-reference0.wav"
