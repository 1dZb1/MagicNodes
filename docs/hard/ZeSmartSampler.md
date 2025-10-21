# MG_ZeSmartSampler (v1.1)

Custom sampler that builds hybrid sigma schedules (Karras/Beta blend), adds tiny schedule jitter, and optionally applies a PC2‑like predictor‑corrector shaping.

## Overview
- Inputs/Outputs match a standard KSampler: `MODEL / SEED / STEPS / CFG / base_sampler / schedule / CONDITIONING / LATENT` → `LATENT`.
- `hybrid_mix` blends the tail toward Beta; `tail_smooth` softens tail jumps adaptively.
- `jitter_sigma` introduces a tiny monotonic noise to schedules for de‑ringing; remains deterministic with fixed seed.
- PC2‑style shaping is available via `smart_strength/target_error/curv_sensitivity` (kept conservative by default).

## Controls (high‑level)
- `base_sampler` and `schedule` (karras/beta/hybrid)
- `hybrid_mix` ∈ [0..1]
- `jitter_sigma` ∈ [0..0.1]
- `tail_smooth` ∈ [0..1]
- `smart_strength`, `target_error`, `curv_sensitivity`

## Tips
- Start hybrid at `hybrid_mix≈0.3` for 2D work; 0.5–0.7 for photo‑like.
- Keep `jitter_sigma` very small (≈0.005–0.01) to avoid destabilizing steps.
- If using inside CADE (`scheduler=MGHybrid`), CADE will construct the schedule and run the custom path automatically.

