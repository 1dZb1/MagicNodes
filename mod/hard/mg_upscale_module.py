import comfy.utils
import torch
import gc
import logging
import comfy.model_management as model_management


def clear_gpu_and_ram_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _smart_decode(vae, latent, tile_size=512):
    try:
        images = vae.decode(latent["samples"])
    except model_management.OOM_EXCEPTION:
        logging.warning("VAE decode OOM, using tiled decode")
        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(
            latent["samples"],
            tile_x=tile_size // compression,
            tile_y=tile_size // compression,
            overlap=(tile_size // 4) // compression,
        )
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


class MagicUpscaleModule:
    """Moved into mod/ as mg_upscale_module keeping class/key name."""
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {}),
                "vae": ("VAE", {}),
                "upscale_method": (cls.upscale_methods, {"default": "bilinear"}),
                "scale_by": ("FLOAT", {"default": 1.2, "min": 0.01, "max": 8.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("LATENT", "Upscaled Image")
    FUNCTION = "process_upscale"
    CATEGORY = "MagicNodes"

    def process_upscale(self, samples, vae, upscale_method, scale_by):
        clear_gpu_and_ram_cache()
        images = _smart_decode(vae, samples)
        samples_t = images.movedim(-1, 1)
        width = round(samples_t.shape[3] * scale_by)
        height = round(samples_t.shape[2] * scale_by)
        # Align to VAE stride to avoid border artifacts/shape drift
        try:
            stride = int(vae.spacial_compression_decode())
        except Exception:
            stride = 8
        if stride <= 0:
            stride = 8
        def _align_up(x, s):
            return int(((x + s - 1) // s) * s)
        width_al = _align_up(width, stride)
        height_al = _align_up(height, stride)
        up = comfy.utils.common_upscale(samples_t, width_al, height_al, upscale_method, "disabled")
        up = up.movedim(1, -1)
        encoded = vae.encode(up[:, :, :, :3])
        return ({"samples": encoded}, up)
