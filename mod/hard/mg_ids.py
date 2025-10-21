from __future__ import annotations

import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _torch_gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    # image: BHWC in [0,1]
    if sigma <= 0.0:
        return image
    device = image.device
    dtype = image.dtype
    radius = max(1, int(3.0 * float(sigma)))
    ksize = radius * 2 + 1
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(x * x) / (2.0 * (sigma ** 2)))
    g1 = (g1 / g1.sum()).view(1, 1, 1, -1)
    g2 = g1.transpose(2, 3)
    xch = image.movedim(-1, 1)  # BCHW
    pad = (radius, radius, radius, radius)
    out = torch.nn.functional.conv2d(torch.nn.functional.pad(xch, pad, mode="reflect"), g1.repeat(xch.shape[1], 1, 1, 1), groups=xch.shape[1])
    out = torch.nn.functional.conv2d(torch.nn.functional.pad(out, pad, mode="reflect"), g2.repeat(out.shape[1], 1, 1, 1), groups=out.shape[1])
    return out.movedim(1, -1)


class IntelligentDetailStabilizer:
    """Alias-preserving move of IDS into mod/ as mg_ids.py.
    Keeps class/key name for backward compatibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "ids_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "stabilize"
    CATEGORY = "MagicNodes"

    def stabilize(self, image: torch.Tensor, ids_strength: float = 0.5):
        sigma = max(float(ids_strength) * 2.0, 1e-3)
        if _HAVE_SCIPY:
            img_np = image.detach().cpu().numpy()
            denoised = _scipy_gaussian_filter(img_np, sigma=(0, sigma, sigma, 0))
            blurred = _scipy_gaussian_filter(denoised, sigma=(0, 1.0, 1.0, 0))
            sharpen = denoised + ids_strength * (denoised - blurred)
            sharpen = np.clip(sharpen, 0.0, 1.0)
            out = torch.from_numpy(sharpen).to(image.device, dtype=image.dtype)
        else:
            denoised = _torch_gaussian_blur(image, sigma=sigma)
            blurred = _torch_gaussian_blur(denoised, sigma=1.0)
            out = (denoised + ids_strength * (denoised - blurred)).clamp(0, 1)
        return (out,)
