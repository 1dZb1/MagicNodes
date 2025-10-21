from __future__ import annotations

import math
import torch
import torch.nn.functional as F  # noqa: F401

import comfy.utils as _utils
import comfy.sample as _sample
import comfy.samplers as _samplers
from comfy.k_diffusion import sampling as _kds

import nodes  # latent preview callback


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    return x * x * (3.0 - 2.0 * x)


def _build_hybrid_sigmas(model, steps: int, base_sampler: str, mode: str,
                         mix: float, denoise: float, jitter: float, seed: int,
                         _debug: bool = False, tail_smooth: float = 0.0,
                         auto_hybrid_tail: bool = True, auto_tail_strength: float = 0.35):
    """Return 1D tensor of sigmas (len == steps+1), strictly descending and ending with 0.

    mode: 'karras' | 'beta' | 'hybrid'. If 'hybrid', blend tail toward beta by `mix`.
    We DO NOT apply 'drop penultimate' until the very end to preserve denoise math.
    """
    ms = model.get_model_object("model_sampling")
    steps = int(steps)
    assert steps >= 1

    # --- base tracks ---
    sig_k = _samplers.calculate_sigmas(ms, "karras", steps)
    sig_b = _samplers.calculate_sigmas(ms, "beta",   steps)

    mode = str(mode).lower()
    if mode == "karras":
        sig = sig_k
    elif mode == "beta":
        sig = sig_b
    else:
        n = sig_k.shape[0]
        t = torch.linspace(0.0, 1.0, n, device=sig_k.device, dtype=sig_k.dtype)
        m = float(max(0.0, min(1.0, mix)))
        eps = 1e-6 if m < 1e-6 else m
        w = torch.clamp((t - (1.0 - m)) / eps, 0.0, 1.0)
        w = _smoothstep01(w)
        sig = sig_k * (1.0 - w) + sig_b * w

    # --- Comfy denoise semantics: recompute a "full" track and take the tail of desired length ---
    sig_k_base = sig_k
    sig_b_base = sig_b
    if denoise is not None and 0.0 < float(denoise) < 0.999999:
        new_steps = max(1, int(steps / max(1e-6, float(denoise))))
        sk = _samplers.calculate_sigmas(ms, "karras", new_steps)
        sb = _samplers.calculate_sigmas(ms, "beta",   new_steps)
        if mode == "karras":
            sig_full = sk
        elif mode == "beta":
            sig_full = sb
        else:
            n2 = sk.shape[0]
            t2 = torch.linspace(0.0, 1.0, n2, device=sk.device, dtype=sk.dtype)
            m = float(max(0.0, min(1.0, mix)))
            eps = 1e-6 if m < 1e-6 else m
            w2 = torch.clamp((t2 - (1.0 - m)) / eps, 0.0, 1.0)
            w2 = _smoothstep01(w2)
            sig_full = sk * (1.0 - w2) + sb * w2
        need = steps + 1
        if sig_full.shape[0] >= need:
            sig = sig_full[-need:]
            sig_k_base = sk[-need:]
            sig_b_base = sb[-need:]
        else:
            # Worst case: trust what we got; we will still guarantee the last sigma is zero later
            sig = sig_full
            tail = min(need, sk.shape[0])
            sig_k_base = sk[-tail:]
            sig_b_base = sb[-tail:]

    # --- auto-hybrid tail: blend beta into the tail when the steps become brittle ---
    if bool(auto_hybrid_tail) and sig.numel() > 2:
        n = sig.shape[0]
        t = torch.linspace(0.0, 1.0, n, device=sig.device, dtype=sig.dtype)
        m = float(max(0.0, min(1.0, mix)))
        if mode == "hybrid":
            eps = 1e-6 if m < 1e-6 else m
            w_m = torch.clamp((t - (1.0 - m)) / eps, 0.0, 1.0)
            w_m = _smoothstep01(w_m)
        elif mode == "beta":
            w_m = torch.ones_like(t)
        else:
            w_m = torch.zeros_like(t)
        dif = (sig[1:] - sig[:-1]).abs() / sig[:-1].abs().clamp_min(1e-8)
        dif = torch.cat([dif, dif[-1:]], dim=0)
        dif = (dif - dif.min()) / (dif.max() - dif.min() + 1e-8)
        ramp = _smoothstep01(torch.clamp((t - 0.7) / 0.3, 0.0, 1.0))
        w_a = dif * ramp
        g = float(max(0.0, min(1.0, auto_tail_strength)))
        u = w_m + g * w_a - w_m * g * w_a
        sig = sig_k_base * (1.0 - u) + sig_b_base * u

    # --- tiny schedule jitter ---
    j = float(max(0.0, min(0.1, float(jitter))))
    if j > 0.0 and sig.numel() > 1:
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed) ^ 0x5EEDCAFE)
        noise = torch.randn(sig.shape, generator=gen, device='cpu').to(sig.device, sig.dtype)
        amp = j * float(sig[0].item() - sig[-1].item()) * 1e-3
        sig = sig + noise * amp
        sig, _ = torch.sort(sig, descending=True)

    # --- hard guarantee of ending with exact zero ---
    if sig[-1].abs() > 1e-12:
        sig = torch.cat([sig[:-1], sig.new_zeros(1)], dim=0)

    # --- and only now drop-penultimate for respective samplers ---
    # --- gentle smoothing of sigma tail (adaptive, safe for monotonic decrease) ---
    ts = float(max(0.0, min(1.0, tail_smooth)))
    if ts > 0.0 and sig.numel() > 2:
        s = sig.clone()
        n = int(s.shape[0])
        t = torch.linspace(0.0, 1.0, n, device=s.device, dtype=s.dtype)
        w = (t.pow(2) * ts).clamp(0.0, 1.0)
        for i in range(n - 2, -1, -1):
            a = float(min(0.5, 0.5 * w[i].item()))
            s[i] = (1.0 - a) * s[i] + a * s[i + 1]
        sig = s

    if base_sampler in _samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS and sig.numel() >= 2:
        sig = torch.cat([sig[:-2], sig[-1:]], dim=0)

    sig = sig.to(model.load_device)

    # Lightweight debug: schedule summary
    if _debug:
        try:
            desc_ok = bool((sig[:-1] > sig[1:]).all().item()) if sig.numel() > 1 else True
            head = ", ".join(f"{float(v):.4g}" for v in sig[:3].tolist()) if sig.numel() >= 3 else \
                   ", ".join(f"{float(v):.4g}" for v in sig.tolist())
            tail = ", ".join(f"{float(v):.4g}" for v in sig[-3:].tolist()) if sig.numel() >= 3 else head
            print(f"[ZeSmart][dbg] sigmas len={sig.numel()} desc={desc_ok} first={float(sig[0]):.6g} last={float(sig[-1]):.6g}")
            print(f"[ZeSmart][dbg] head: [{head}]  tail: [{tail}]")
        except Exception:
            pass

    return sig


class MG_ZeSmartSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63-1, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 4096}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "base_sampler": (_samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "schedule": (["karras", "beta", "hybrid"], {"default": "hybrid", "tooltip": "Sigma curve: karras — soft start; beta — stable tail; hybrid — their mix."}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "latent": ("LATENT", {}),
            },
            "optional": {
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Path shortening: 1.0 = full; <1.0 = take the last steps only. Useful for inpaint/mixing."}),
                "hybrid_mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "For schedule=hybrid: tail fraction blended toward beta (0=karras, 1=beta)."}),
                "jitter_sigma": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Tiny sigma jitter to kill moiré/banding on backgrounds. 0–0.02 is usually enough."}),
                "tail_smooth": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Smooth the sigma tail — reduces wobble/banding. Too high may soften details."}),
                "auto_hybrid_tail": ("BOOLEAN", {"default": True, "tooltip": "Auto‑blend beta on the tail when steps become brittle."}),
                "auto_tail_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of auto beta‑mix on the tail (0=off, 1=max)."}),
                "debug_probe": ("BOOLEAN", {"default": False, "tooltip": "Print sigma summary (length, first/last, head/tail)."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "MagicNodes/Experimental"

    def apply(self, model, seed, steps, cfg, base_sampler, schedule,
              positive, negative, latent, denoise=1.0, hybrid_mix=0.5,
              jitter_sigma=0.02, tail_smooth=0.07,
              auto_hybrid_tail=True, auto_tail_strength=0.3,
              debug_probe=False):
        # Prepare latent + noise
        lat_img = latent["samples"]
        lat_img = _sample.fix_empty_latent_channels(model, lat_img)
        batch_inds = latent.get("batch_index", None)
        noise = _sample.prepare_noise(lat_img, seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)

        # Custom sigmas
        sigmas = _build_hybrid_sigmas(model, int(steps), str(base_sampler), str(schedule),
                                      float(hybrid_mix), float(denoise), float(jitter_sigma), int(seed),
                                      _debug=bool(debug_probe), tail_smooth=float(tail_smooth),
                                      auto_hybrid_tail=bool(auto_hybrid_tail),
                                      auto_tail_strength=float(auto_tail_strength))

        # Use native sampler; all tweaks happen in sigma schedule only.
        sampler_obj = _samplers.sampler_object(str(base_sampler))
        callback = nodes.latent_preview.prepare_callback(model, int(steps))
        disable_pbar = not _utils.PROGRESS_BAR_ENABLED
        samples = _sample.sample_custom(model, noise, float(cfg), sampler_obj, sigmas,
                                        positive, negative, lat_img,
                                        noise_mask=noise_mask, callback=callback,
                                        disable_pbar=disable_pbar, seed=seed)
        out = {**latent}
        out["samples"] = samples
        return (out,)
