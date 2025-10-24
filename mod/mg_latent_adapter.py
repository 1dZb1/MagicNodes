"""
MagicLatentAdapter: two-in-one latent adapter for ComfyUI.

- Mode "generate": creates a latent of appropriate grid size for the target model
  (optionally mixing an input image via VAE), then adapts channels.
- Mode "adapt": takes an incoming LATENT and adapts channel count to match the model.

Family switch: "auto / SD / SDXL / FLUX" influences only stride fallback when VAE
is not provided. In AUTO we query VAE stride if possible and fall back to 8.

No file re-encodings are performed; all code is ASCII/English as requested.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.sample as _sample


class MagicLatentAdapter:
    """Generate or adapt a LATENT to fit the target model's expectations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "mode": (["generate", "adapt"], {"default": "generate"}),
                "family": (["auto", "SD", "SDXL", "FLUX"], {"default": "auto"}),

                # Generation params (ignored in adapt mode)
                "width": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "bias": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "mix_image": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # For adapt mode
                "latent": ("LATENT", {}),
                # For image mixing in generate mode
                "vae": ("VAE", {}),
                "image": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "MagicNodes"

    @staticmethod
    def _detect_stride(vae, family: str) -> int:
        # Prefer VAE stride if available
        if vae is not None:
            try:
                s = int(vae.spacial_compression_decode())
                if s > 0:
                    return s
            except Exception:
                pass
        # Fallback per-family (conservative)
        fam = (family or "auto").lower()
        if fam in ("sd", "sdxl", "flux"):
            return 8
        return 8  # sensible default

    @staticmethod
    def _latent_format(model) -> tuple[int, int]:
        """Return (channels, dimensions) from model.latent_format.
        dimensions: 2 -> NCHW, 3 -> NCDHW.
        """
        try:
            lf = model.get_model_object("latent_format")
            ch = int(getattr(lf, "latent_channels", 4))
            dims = int(getattr(lf, "latent_dimensions", 2))
            if dims not in (2, 3):
                dims = 2
            return ch, dims
        except Exception:
            return 4, 2

    @staticmethod
    def _adapt_channels(model, z: torch.Tensor, preserve_zero: bool = False) -> torch.Tensor:
        """Adapts channel count and dims to the model's latent_format.
        If preserve_zero and the latent is all zeros, pad with zeros instead of noise.
        """
        target_c, target_dims = MagicLatentAdapter._latent_format(model)

        # First, let Comfy add depth dim for empty latents when needed
        try:
            z = _sample.fix_empty_latent_channels(model, z)
        except Exception:
            pass

        # Align dimensions
        if target_dims == 3 and z.ndim == 4:
            z = z.unsqueeze(2)  # N C 1 H W
        elif target_dims == 2 and z.ndim == 5:
            if z.shape[2] == 1:
                z = z.squeeze(2)
            else:
                z = z[:, :, :1].squeeze(2)

        # Align channels
        if z.ndim == 4:
            B, C, H, W = z.shape
            if C == target_c:
                return z
            if C > target_c:
                return z[:, :target_c]
            dev, dt = z.device, z.dtype
            if preserve_zero and torch.count_nonzero(z) == 0:
                pad = torch.zeros(B, target_c - C, H, W, device=dev, dtype=dt)
            else:
                pad = torch.randn(B, target_c - C, H, W, device=dev, dtype=dt)
            return torch.cat([z, pad], dim=1)
        elif z.ndim == 5:
            B, C, D, H, W = z.shape
            if C == target_c:
                return z
            if C > target_c:
                return z[:, :target_c]
            dev, dt = z.device, z.dtype
            if preserve_zero and torch.count_nonzero(z) == 0:
                pad = torch.zeros(B, target_c - C, D, H, W, device=dev, dtype=dt)
            else:
                pad = torch.randn(B, target_c - C, D, H, W, device=dev, dtype=dt)
            return torch.cat([z, pad], dim=1)
        else:
            return z

    @staticmethod
    def _mix_image_into_latent(vae, image_bhwc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if vae is None or image_bhwc is None:
            return z
        try:
            # Align image spatial to VAE grid by padding (mirror) if needed
            try:
                stride = int(vae.spacial_compression_decode())
            except Exception:
                stride = 8
            h, w = image_bhwc.shape[1:3]
            def _align_up(x, s):
                return int(((x + s - 1) // s) * s)
            Ht, Wt = _align_up(h, stride), _align_up(w, stride)
            x = image_bhwc
            if (Ht != h) or (Wt != w):
                pad_h = Ht - h
                pad_w = Wt - w
                x_nchw = x.movedim(-1, 1)
                x_nchw = F.pad(x_nchw, (0, pad_w, 0, pad_h), mode='replicate')
                x = x_nchw.movedim(1, -1)
            enc = vae.encode(x[:, :, :, :3])
            # If batch mismatches, use first encoding and tile
            while enc.ndim < z.ndim:
                enc = enc.unsqueeze(2)  # add depth dim if needed
            while enc.ndim > z.ndim:
                # reduce extra depth dims
                if enc.ndim == 5 and enc.shape[2] == 1:
                    enc = enc.squeeze(2)
                else:
                    enc = enc[(slice(None), slice(None)) + (slice(0,1),) * (enc.ndim-2)]
                    if enc.ndim == 5:
                        enc = enc.squeeze(2)
            if enc.shape[0] != z.shape[0]:
                enc = enc[:1]
                enc = enc.repeat(z.shape[0], *([1] * (enc.ndim - 1)))
            # Resize spatial if needed (nearest)
            if enc.ndim == 4:
                if enc.shape[2:] != z.shape[2:]:
                    enc = F.interpolate(enc, size=z.shape[2:], mode="nearest")
            elif enc.ndim == 5:
                if enc.shape[2:] != z.shape[2:]:
                    enc = F.interpolate(enc, size=z.shape[2:], mode="nearest")
            # Channel adapt for mixing safety
            if enc.shape[1] != z.shape[1]:
                cmin = min(enc.shape[1], z.shape[1])
                enc = enc[:, :cmin]
                z = z[:, :cmin]
            return enc + z
        except Exception:
            return z

    def run(
        self,
        model,
        mode: str,
        family: str,
        width: int,
        height: int,
        batch_size: int,
        sigma: float,
        bias: float,
        mix_image: bool = False,
        latent=None,
        vae=None,
        image=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mode == "adapt":
            if latent is None or "samples" not in latent:
                # Produce an empty latent matching model's latent_format
                stride = self._detect_stride(vae, family)
                h8, w8 = max(1, height // stride), max(1, width // stride)
                target_c, target_dims = self._latent_format(model)
                if target_dims == 3:
                    z = torch.zeros(batch_size, target_c, 1, h8, w8, device=device)
                else:
                    z = torch.zeros(batch_size, target_c, h8, w8, device=device)
            else:
                z = latent["samples"].to(device)
            z = self._adapt_channels(model, z, preserve_zero=True)
            return ({"samples": z},)

        # generate
        stride = self._detect_stride(vae, family)
        h8, w8 = max(1, height // stride), max(1, width // stride)
        target_c, target_dims = self._latent_format(model)
        if target_dims == 3:
            z = torch.randn(batch_size, target_c, 1, h8, w8, device=device) * float(sigma) + float(bias)
        else:
            z = torch.randn(batch_size, target_c, h8, w8, device=device) * float(sigma) + float(bias)
        if mix_image and (vae is not None) and (image is not None):
            # image is BHWC 0..1
            img = image.to(device)
            z = self._mix_image_into_latent(vae, img, z)
        # Final channel adaptation
        z = self._adapt_channels(model, z, preserve_zero=False)
        return ({"samples": z},)
