#!/usr/bin/env bash
# Run the same generation tests as the GitHub Action (test-generation.yml).
# Use this to validate locally before pushing. No assumptions: build and models required.
#
# From repo root:
#   ./models.sh              # once: download Q8_0 + VAE into models/
#   mkdir -p build && cd build && cmake .. && cmake --build . --config Release
#   cd .. && tests/run-generation-tests.sh

set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# --- Build ---
if [ ! -f build/dit-vae ] || [ ! -f build/ace-qwen3 ]; then
    echo "Missing build/dit-vae or build/ace-qwen3. Build first:"
    echo "  mkdir -p build && cd build && cmake .. && cmake --build . --config Release"
    exit 1
fi

# --- Models ---
TEXT_ENC="models/Qwen3-Embedding-0.6B-Q8_0.gguf"
DIT="models/acestep-v15-turbo-Q8_0.gguf"
VAE="models/vae-BF16.gguf"
LM="models/acestep-5Hz-lm-4B-Q8_0.gguf"
for f in "$TEXT_ENC" "$DIT" "$VAE"; do
    if [ ! -f "$f" ]; then
        echo "Missing $f. Download models once: ./models.sh"
        exit 1
    fi
done

echo "[1/3] Test mode text2music (short)"
./build/dit-vae \
    --request tests/fixtures/ci-text2music.json \
    --text-encoder "$TEXT_ENC" \
    --dit "$DIT" \
    --vae "$VAE"
if [ ! -f tests/fixtures/ci-text2music0.wav ]; then
    echo "FAIL: tests/fixtures/ci-text2music0.wav not created"
    exit 1
fi
echo "  text2music WAV OK"

echo "[2/3] Test mode cover with WAV reference (short)"
./build/dit-vae \
    --request tests/fixtures/ci-cover.json \
    --text-encoder "$TEXT_ENC" \
    --dit "$DIT" \
    --vae "$VAE"
if [ ! -f tests/fixtures/ci-cover0.wav ]; then
    echo "FAIL: tests/fixtures/ci-cover0.wav not created"
    exit 1
fi
echo "  cover WAV OK"

echo "[3/3] Test full pipeline (LLM + DiT, short)"
if [ ! -f "$LM" ]; then
    echo "Missing $LM; skipping full pipeline. Run ./models.sh to include LM."
    exit 1
fi
# ace-qwen3 names output from input path (e.g. request.json -> request0.json)
cp tests/fixtures/ci-text2music.json request.json
./build/ace-qwen3 \
    --request request.json \
    --model "$LM"
if [ ! -f request0.json ]; then
    echo "FAIL: request0.json not created by ace-qwen3"
    exit 1
fi
./build/dit-vae \
    --request request0.json \
    --text-encoder "$TEXT_ENC" \
    --dit "$DIT" \
    --vae "$VAE"
if [ ! -f request00.wav ]; then
    echo "FAIL: request00.wav not created"
    exit 1
fi
echo "  full pipeline WAV OK"

echo ""
echo "All generation tests passed locally. Safe to rely on CI for the same checks."
