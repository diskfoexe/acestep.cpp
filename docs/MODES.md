# ACE-Step 1.5 built-in modes (acestep.cpp)

This document maps the [ACE-Step 1.5 Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md) built-in modes to the current C++ implementation.

## Task types (Tutorial: Input Control)

| `task_type`   | Description | Turbo/SFT | Base only | C++ status |
|---------------|-------------|-----------|-----------|------------|
| **text2music** | Generate from caption/lyrics (and optional reference) | ✅ | — | ✅ **Supported** |
| **cover**      | Re-synthesize with structure from source; optional timbre from reference | ✅ | — | ✅ **Supported** (audio_codes or src_audio WAV/MP3) |
| **repaint**    | Local edit in time range using source as context | ✅ | — | ❌ Not implemented |
| **lego**       | Add new tracks to existing audio | — | ✅ | ❌ Base model only |
| **extract**    | Extract single track from mix | — | ✅ | ❌ Base model only |
| **complete**   | Add accompaniment to single track | — | ✅ | ❌ Base model only |

We only ship Turbo and SFT DiT weights; **lego**, **extract**, **complete** require the Base DiT and are out of scope for now.

---

## What we support today

### text2music (default)
- **Input**: `caption`, optional `lyrics`, metadata (bpm, duration, keyscale, …).
- **Flow**: LM (optional) → CoT + audio codes → DiT (context = silence) → VAE → WAV.
- **Timbre**: Optional **reference_audio** (WAV/MP3) → VAE encode → CondEncoder timbre; else built-in silence.

### cover (when `audio_codes` or `src_audio` are provided)
- **Input**: Same as text2music, plus either **precomputed** `audio_codes` or **`src_audio`** (WAV/MP3 path). Optional **reference_audio** for timbre.
- **Flow**: If `src_audio` set and no `audio_codes`: load WAV/MP3 → VAE encode → FSQ nearest-codeword encode → codes. Then decode codes to latents → DiT context (blend with silence) → DiT → VAE → WAV. No Python.
- **reference_audio** and **audio_cover_strength**: Implemented (timbre; blend).
---

## What’s not implemented yet

### repaint
- **Tutorial**: Specify `repainting_start` / `repainting_end` (seconds); model uses source audio as context and only generates in that interval (3–90 s).
- **C++**: Would require **masked diffusion**: context carries “given” frames; ODE only updates the repaint region. DiT’s context has a 64-channel “mask” that we currently set to 1.0; repaint would set mask per frame and the generation loop would only update unmasked frames. Not implemented.

---

## Request fields (aligned with Tutorial)

All of these are in `AceRequest` and parsed from / written to JSON. Backend behavior is as above.

| Field | Type | Purpose |
|-------|------|--------|
| `task_type` | string | `"text2music"` \| `"cover"` \| `"repaint"` \| … |
| `reference_audio` | string | Path to WAV or MP3 for timbre (implemented) |
| `src_audio` | string | Path to WAV or MP3 for cover source; encoded to codes internally (implemented) |
| `audio_codes` | string | Comma-separated FSQ codes; non-empty ⇒ cover path (or from `src_audio`) |
| `audio_cover_strength` | float | 0.0–1.0 blend of decoded context with silence (implemented) |
| `repainting_start` | float | Start time (s) for repaint (not used yet) |
| `repainting_end` | float | End time (s) for repaint (not used yet) |

See `request.h` and the README “Request JSON reference” for the full list.

---

## Summary

- **Fully supported**: text2music (with optional reference_audio for timbre); cover from **precomputed** `audio_codes` or from **WAV/MP3** via `src_audio` (VAE encode + FSQ nearest-codeword encode); reference_audio (timbre); audio_cover_strength (blend).
- **Schema only** (no backend): `repainting_start`/`repainting_end`.
- **To support repaint**: implement masked DiT generation (context mask + ODE only on repaint interval).
