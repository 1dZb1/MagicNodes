"""Adaptive sampler helper node (moved to mod/).

Keeps class/key name AdaptiveSamplerHelper for backward compatibility.
"""

import numpy as np
from scipy.ndimage import laplace


class AdaptiveSamplerHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("steps", "cfg", "denoise")
    FUNCTION = "tune"
    CATEGORY = "MagicNodes"

    def tune(self, image, steps, cfg, denoise):
        img = image[0].cpu().numpy()
        gray = img.mean(axis=2)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        sharpness = float(np.var(laplace(gray)))

        tuned_steps = int(max(1, round(steps + sharpness * 10)))
        tuned_cfg = float(cfg + contrast * 2.0)
        tuned_denoise = float(np.clip(denoise + (0.5 - brightness), 0.0, 1.0))

        return (tuned_steps, tuned_cfg, tuned_denoise)

