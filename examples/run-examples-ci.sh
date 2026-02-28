#!/bin/bash
# Run all example scripts with short-duration CI fixtures (from repo root).
# Prereqs: build/ and models/ present; run after build and ./models.sh.
set -eu
cd "$(dirname "$0")/.."
EXAMPLES=examples
cd "$EXAMPLES"

run() { echo "== $*" && "$@"; }

# 1) DiT-only (no LLM), 5s
run cp ../tests/fixtures/ci-dit-only.json dit-only.json
run ./dit-only.sh
test -f dit-only0.wav && echo "dit-only OK"

# 2) Cover from precomputed audio_codes (existing cover.json, 10s)
run ./cover.sh
test -f cover0.wav && echo "cover OK"

# 3) reference.wav for cover-reference and test-reference
run cp cover0.wav reference.wav

# 4) Cover + reference timbre
run ./cover-reference.sh
test -f cover-reference0.wav && echo "cover-reference OK"

# 5) text2music with reference_audio
run cp ../tests/fixtures/ci-request-reference.json request-reference.json
run ./test-reference.sh
test -f request-reference0.wav && echo "test-reference OK"

# 6) Simple (caption only, LLM fills), 5s
run cp ../tests/fixtures/ci-text2music.json simple.json
run ./simple.sh
test -f simple00.wav && echo "simple OK"

# 7) Partial (caption + lyrics + duration), 5s
run cp ../tests/fixtures/ci-partial.json partial.json
run ./partial.sh
test -f partial00.wav && echo "partial OK"

# 8) Full (all metadata), 5s
run cp ../tests/fixtures/ci-full.json full.json
run ./full.sh
test -f full00.wav && echo "full OK"

echo "All example scripts passed."
