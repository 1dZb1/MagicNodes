# IntelligentDetailStabilizer (IDS)

Gentle, fast post‑pass for stabilizing micro‑detail and suppressing noise while preserving sharpness.

## Overview
- Two‑stage blur/sharpen split with strength‑controlled recombination.
- Uses SciPy Gaussian if available; otherwise a portable PyTorch separable blur.
- Operates on images (BHWC, 0..1) and returns a single stabilized `IMAGE`.

## Inputs
- `image` (IMAGE)
- `ids_strength` (float, default 0.5, range −1.0..1.0)

## Outputs
- `IMAGE` — stabilized image

## Tips
- Start around `ids_strength≈0.5` for gentle cleanup.
- Negative values bias toward more smoothing; positive increases sharpening of denoised base.

