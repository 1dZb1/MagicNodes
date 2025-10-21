"""
Simple latent generator for ComfyUI.
The ``MagicSeedLatent`` class creates a random latent tensor of the specified size.
If ``mix_image`` is enabled, the input image is encoded with a VAE and mixed with noise.
"""

from __future__ import annotations

import torch


class MagicSeedLatent:
    """Generate a latent tensor with optional image mixing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "bias": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "mix_image": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "vae": ("VAE", {}),
                "image": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "MagicNodes"

    def generate(
        self,
        width: int,
        height: int,
        batch_size: int,
        sigma: float,
        bias: float,
        mix_image: bool = False,
        vae=None,
        image=None,
    ):
        """Generate a random latent tensor and optionally mix it with an image."""

        lat = torch.randn(batch_size, 4, height // 8, width // 8) * sigma + bias
        if mix_image and vae is not None and image is not None:
            encoded = vae.encode(image[:, :, :, :3])
            lat = encoded + lat
        return ({"samples": lat},)