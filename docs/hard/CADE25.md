# CADE 2.5 (ComfyAdaptiveDetailEnhancer25)

CADE 2.5 is a refined adaptive enhancer with a single clean iteration loop, optional reference‑driven polishing, and flexible sampler scheduling. It can run standalone or as part of multi‑step pipelines (e.g., with ControlFusion masks in between passes).

This document describes the Hard variant — the full‑surface node that exposes advanced controls. For a minimal, preset‑driven experience, use the Easy variant or the `MG_SuperSimple` orchestrator.

## Overview
- Iterative latent refinement with configurable steps/CFG/denoise
- Optional guidance override (Rescale/CFGZero‑style, FDG/NAG ideas, epsilon scaling)
- Hybrid schedule path (`MGHybrid`) that builds ZeSmart‑style sigma stacks
- Local spatial guidance via CLIPSeg prompts
- Reference polishing with CLIP‑Vision (preserves low‑frequency structure)
- Optional upscaling mid‑run, detail stabilization, and gentle sharpening
- Determinism helpers: CLIPSeg pinned to CPU, mask state cleared per run

## Inputs
- `model` (MODEL)
- `positive` (CONDITIONING), `negative` (CONDITIONING)
- `vae` (VAE)
- `latent` (LATENT)
- `reference_image` (IMAGE, optional)
- `clip_vision` (CLIP_VISION, optional)

## Outputs
- `LATENT`: refined latent
- `IMAGE`: decoded image after the last internal iteration
- `mask_preview` (IMAGE): last fused mask preview (RGB 0..1)
- Internal values like effective `steps/cfg/denoise` are tracked across the loop (the Easy wrapper surfaces them if needed).

## Core Controls (essentials)
- `seed` (with control_after_generate)
- `steps`, `cfg`, `denoise`
- `sampler_name` (e.g., `ddim`)
- `scheduler` (`MGHybrid` recommended for smooth tails)

Typical starting points
- General: steps≈25, cfg≈7.0, denoise≈0.7, sampler=`euler_ancestral`, scheduler=`MGHybrid`
- As the first pass of a multi‑step pipeline: denoise=1.0 (full rewrite pass)

## MGHybrid schedule
When `scheduler = MGHybrid`, CADE builds a hybrid sigma schedule compatible with the internal KSampler path. It follows ZeSmart principles (hybrid mix and smooth tail), then calls a custom sampler entry — falling back to `nodes.common_ksampler` if anything goes wrong. The behavior remains deterministic under fixed `seed/steps/cfg/denoise`.

## Local guidance (CLIPSeg)
- CLIPSeg prompts (comma‑separated) produce a soft mask that can attenuate denoise/CFG.
- CLIPSeg inference is pinned to CPU by default for reproducibility.

## Reference polish (CLIP‑Vision)
Provide `reference_image` and `clip_vision` to preserve global form while refining details. CADE encodes the current and reference images and reduces denoise/CFG when they diverge; in polish mode it also mixes low frequencies from the reference using a blur‑based split.

## Advanced features (high‑level)
- Guidance override wrapper (rescale curves, momentum, perpendicular dampers)
- FDG/ZeRes‑inspired options with adaptive thresholds
- Mid‑run upscale support via `MagicUpscaleModule` with post‑adjusted CFG/denoise
- Post passes: `IntelligentDetailStabilizer`, optional mild sharpen

## Related
- QSilk (micrograin stabilizer + AQClip): a lightweight latent‑space regularizer that suppresses rare activation tails while preserving micro‑texture. Works plug‑and‑play inside CADE 2.5 and synergizes with ZeResFDG by allowing slightly higher effective CFG without speckle. See preprint draft in `Arxiv_QSilk/` (source: [Arxiv_QSilk/main_qsilk.tex](../../Arxiv_QSilk/main_qsilk.tex)). Replace with arXiv link when available.

## Tips
- Keep `vae` consistent across passes; CADE re‑encodes when scale changes.
- For multi‑step flows (e.g., with ControlFusion), feed the current decoded `IMAGE` into CF, update `positive/negative`, then run CADE again with the latest `LATENT`.
- If you rely on presets, consider the Easy wrapper or `MG_SuperSimple` to avoid UI/preset drift.

## Quickstart (Hard)
1) Connect `MODEL / VAE / CONDITIONING / LATENT`.
2) Set `seed`, `steps≈25`, `cfg≈7.0`, `denoise≈0.7`, `sampler=euler_ancestral`, `scheduler=MGHybrid`.
3) (Optional) Add `reference_image` and `clip_vision`, and a CLIPSeg prompt.
4) Run and fine‑tune denoise/CFG first; only then adjust sampler/schedule.

Notes
- The node clears internal masks and patches at the end of a run even on errors.
- Some experimental toggles are intentionally conservative in default configs to avoid destabilizing results.
