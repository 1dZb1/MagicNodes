# MagicUpscaleModule

Lightweight latent‑space upscaler that keeps shapes aligned to the VAE stride to avoid border artifacts.

## Overview
- Decodes latent to image, resamples with selected filter, and re‑encodes.
- Aligns target size up to the VAE spatial compression stride to keep shapes consistent.
- Clears GPU/RAM caches to minimize fragmentation before heavy resizes.

## Inputs
- `samples` (LATENT)
- `vae` (VAE)
- `upscale_method` in `nearest-exact | bilinear | area | bicubic | lanczos`
- `scale_by` (float)

## Outputs
- `LATENT` — upscaled latent
- `Upscaled Image` — convenience decoded image

## Tips
- Use modest `scale_by` first (e.g., 1.2–1.5) and chain passes if needed.
- Keep the same `vae` before and after upscale in a larger pipeline.

