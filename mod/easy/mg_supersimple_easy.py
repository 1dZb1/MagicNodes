from __future__ import annotations

"""MG_SuperSimple: orchestrates a 1..4 step pipeline over CF -> CADE pairs.

Step 1: CADE with the "Step 1" preset (denoise is forced to 1.0).
Steps 2..N: ControlFusion (CF) with the "Step N" preset, then CADE with the same "Step N" preset.
When custom=True: visible CADE controls (seed/steps/cfg/denoise/sampler/scheduler/clipseg_text) override the corresponding Step presets across all steps (Step 1 still uses denoise=1.0).
When custom=False: CADE values come from Step presets; node UI values are ignored. CF always uses its Step presets (kept minimal here).
Inputs:

model/vae/latent/positive/negative: standard Comfy connectors
control_net: ControlNet module for CF (required)
reference_image/clip_vision: forwarded into CADE (optional)
Outputs:

(LATENT, IMAGE) from the final executed step
"""


import torch

from .mg_cade25_easy import ComfyAdaptiveDetailEnhancer25 as _CADE
from .mg_controlfusion_easy import MG_ControlFusion as _CF
from .mg_cade25_easy import _sampler_names as _sampler_names
from .mg_cade25_easy import _scheduler_names as _scheduler_names
import comfy.model_management as model_management
from ..hard.mg_upscale_module import clear_gpu_and_ram_cache


class MG_SuperSimple:
    CATEGORY = "MagicNodes/Easy"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # High-level pipeline control
                "step_count": ("INT", {"default": 4, "min": 1, "max": 4, "tooltip": "Number of steps to run (1..4)."}),
                "custom": ("BOOLEAN", {"default": False, "tooltip": "When enabled, CADE UI values below override Step presets across all steps (denoise on Step 1 is still forced to 1.0)."}),

                # Connectors
                "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "vae": ("VAE", {}),
                "latent": ("LATENT", {}),
                "control_net": ("CONTROL_NET", {"tooltip": "ControlNet module used by ControlFusion."}),

                # Shared CADE surface controls
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Seed 0 = SmartSeed (Sobol + light probe). Non-zero = fixed seed (deterministic)."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000, "tooltip": "KSampler steps for CADE (applies to all steps)."}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                # Denoise is clamped; Step 1 uses 1.0 regardless
                "denoise": ("FLOAT", {"default": 0.65, "min": 0.35, "max": 0.9, "step": 0.0001}),
                "sampler_name": (_sampler_names(), {"default": _sampler_names()[0]}),
                "scheduler": (_scheduler_names(), {"default": "MGHybrid"}),
                "clipseg_text": ("STRING", {"default": "hand, feet, face", "multiline": False, "tooltip": "Focus terms for CLIPSeg (comma-separated)."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {}),
                "clip_vision": ("CLIP_VISION", {}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("LATENT", "IMAGE")
    FUNCTION = "run"

    def _cade(self,
              preset_step: str,
              custom_override: bool,
              model, vae, positive, negative, latent,
              seed: int, steps: int, cfg: float, denoise: float,
              sampler_name: str, scheduler: str,
              clipseg_text: str,
              reference_image=None, clip_vision=None):
        # CADE core call mirrors CADEEasyUI -> apply_cade2
        lat, img, _s, _c, _d, _mask = _CADE().apply_cade2(
            model, vae, positive, negative, latent,
            int(seed), int(steps), float(cfg), float(denoise),
            str(sampler_name), str(scheduler), 0.0,
            preset_step=str(preset_step), custom_override=bool(custom_override),
            clipseg_text=str(clipseg_text),
            reference_image=reference_image, clip_vision=clip_vision,
        )
        return lat, img

    def _cf(self,
            preset_step: str,
            image, positive, negative, control_net, vae):
        # Keep CF strictly on presets for SuperSimple (no extra UI),
        # so pass custom_override=False intentionally.
        pos, neg, _prev = _CF().apply(
            image=image, positive=positive, negative=negative,
            control_net=control_net, vae=vae,
            preset_step=str(preset_step), custom_override=False,
        )
        return pos, neg

    def run(self,
             step_count, custom,
             model, positive, negative, vae, latent, control_net,
             seed, steps, cfg, denoise, sampler_name, scheduler, clipseg_text,
             reference_image=None, clip_vision=None):
        # Cooperative cancel before any heavy work
        model_management.throw_exception_if_processing_interrupted()

        # Clamp step_count to 1..4
        n = int(max(1, min(4, step_count)))

        cur_latent = latent
        cur_image = None
        cur_pos = positive
        cur_neg = negative

        try:
            # Step 1: CADE with Step 1 preset, denoise forced to 1.0
            denoise_step1 = 1.0
            lat1, img1 = self._cade(
                preset_step="Step 1",
                custom_override=bool(custom),
                model=model, vae=vae, positive=cur_pos, negative=cur_neg, latent=cur_latent,
                seed=seed, steps=steps, cfg=cfg, denoise=denoise_step1,
                sampler_name=sampler_name, scheduler=scheduler,
                clipseg_text=clipseg_text,
                reference_image=reference_image, clip_vision=clip_vision,
            )
            cur_latent, cur_image = lat1, img1

            # Steps 2..n: CF -> CADE per step
            for i in range(2, n + 1):
                # allow user cancel between steps
                model_management.throw_exception_if_processing_interrupted()
                # ControlFusion on current image/conds
                cur_pos, cur_neg = self._cf(
                    preset_step=f"Step {i}",
                    image=cur_image, positive=cur_pos, negative=cur_neg,
                    control_net=control_net, vae=vae,
                )
                # CADE with shared controls
                # If no external reference_image is provided, use the previous step image
                # so that reference_clean / CLIP-Vision gating can take effect.
                ref_img = reference_image if (reference_image is not None) else cur_image
                lat_i, img_i = self._cade(
                    preset_step=f"Step {i}",
                    custom_override=bool(custom),
                    model=model, vae=vae, positive=cur_pos, negative=cur_neg, latent=cur_latent,
                    seed=seed, steps=steps, cfg=cfg, denoise=denoise,
                    sampler_name=sampler_name, scheduler=scheduler,
                    clipseg_text=clipseg_text,
                    reference_image=ref_img, clip_vision=clip_vision,
                )
                cur_latent, cur_image = lat_i, img_i
            return (cur_latent, cur_image)
        finally:
            # try to free memory on cancel or normal exit
            try:
                clear_gpu_and_ram_cache()
            except Exception:
                pass
