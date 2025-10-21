# ControlFusion (Hard)

Builds a fused control mask from Depth and Pyramid Canny Edges, then injects it into ControlNet for both positive and negative conditionings. Designed to be resolution‑aware (keeps aspect), with optional split application (Depth then Edges) and a rich set of edge post‑processing knobs.

For minimal usage, see the Easy wrapper documented in `docs/EasyNodes.md`.

## Overview
- Depth: Depth Anything v2 if available (vendored/local/aux fallbacks), otherwise pseudo‑depth from luminance + blur.
- Edges: multi‑scale Pyramid Canny with optional thinning, width adjust, smoothing, single‑line collapse, and depth‑based gating.
- Blending: `normal` (weighted mix), `max`, or `edge_over_depth` prior to ControlNet.
- Application: single fused hint or `split_apply` (Depth first, then Edges) with independent strengths and schedules.
- Preview: aspect‑kept visualization with optional strength reflection (display‑only).

## Inputs
- `image` (IMAGE, BHWC 0..1)
- `positive` (CONDITIONING), `negative` (CONDITIONING)
- `control_net` (CONTROL_NET)
- `vae` (VAE)

## Outputs
- `positive` (CONDITIONING), `negative` (CONDITIONING) — updated with ControlNet hint
- `Mask_Preview` (IMAGE) — fused mask preview (RGB 0..1)

## Core Controls
Depth
- `enable_depth` (bool)
- `depth_model_path` (pth for Depth Anything v2)
- `depth_resolution` (min‑side target; hires mode keeps aspect)

Edges (PyraCanny)
- `enable_pyra` (bool), `pyra_low`, `pyra_high`, `pyra_resolution`
- `edge_thin_iter` (thinning passes, auto‑tuned in smart mode)
- `edge_alpha` (pre‑blend opacity), `edge_boost` (micro‑contrast), `smart_tune`, `smart_boost`

Blend and Strength
- `blend_mode`: `normal` | `max` | `edge_over_depth`
- `blend_factor` (for `normal`)
- `strength_pos`, `strength_neg` (global)
- `start_percent`, `end_percent` (schedule window 0..1)

Preview and Quality
- `preview_res` (min‑side), `mask_brightness`
- `preview_show_strength` with `preview_strength_branch` = `positive` | `negative` | `max` | `avg`
- `hires_mask_auto` (keep aspect and higher caps)

Application Options
- `apply_to_uncond` (mirror ControlNet hint to uncond)
- `stack_prev_control` (stack with previous ControlNet in the cond dict)
- `split_apply` (Depth first, Edges second)
- Separate schedules and multipliers when split:
  - Depth: `depth_start_percent`, `depth_end_percent`, `depth_strength_mul`
  - Edges: `edge_start_percent`, `edge_end_percent`, `edge_strength_mul`

Extra Edge Controls
- `edge_width` (thin/thicken), `edge_smooth` (reduce pixelation)
- `edge_single_line`, `edge_single_strength` (collapse double outlines)
- `edge_depth_gate`, `edge_depth_gamma` (weigh edges by depth)

## Behavior Notes
- Depth min‑side is capped (default 1024) and aspect is preserved to avoid distortions.
- In `split_apply`, the order is deterministic: Depth → Edges.
- Preview image reflects strength only if `preview_show_strength` is enabled; it does not affect the hint itself.
- When both Depth and Edges are disabled, the node passes inputs through and returns a zero preview.

## Quickstart
1) Connect `image/positive/negative/control_net/vae`.
2) Enable Depth and/or PyraCanny. Start with `edge_alpha≈1.0`, `blend_mode=normal`, `blend_factor≈0.02`.
3) Schedule the apply window (`start_percent/end_percent`) and tune `strength_pos/neg`.
4) Use `split_apply` if you want Depth to anchor structure and Edges to refine contours separately.

