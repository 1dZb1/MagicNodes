"""CADE 2.5: refined adaptive enhancer with reference clean and accumulation override.
"""

from __future__ import annotations  # moved/renamed module: mg_cade25

import torch
import os
import numpy as np
import torch.nn.functional as F

import nodes
import comfy.model_management as model_management

from ..hard.mg_adaptive import AdaptiveSamplerHelper
from ..hard.mg_zesmart_sampler_v1_1 import _build_hybrid_sigmas
import comfy.sample as _sample
import comfy.samplers as _samplers
import comfy.utils as _utils
from ..hard.mg_upscale_module import MagicUpscaleModule, clear_gpu_and_ram_cache
from ..hard.mg_controlfusion import _build_depth_map as _cf_build_depth_map
from ..hard.mg_ids import IntelligentDetailStabilizer
from .. import mg_sagpu_attention as sa_patch
from .preset_loader import get as load_preset
from ..hard.mg_controlfusion import _pyracanny as _cf_pyracanny, _build_depth_map as _cf_build_depth
# FDG/NAG experimental paths removed for now; keeping code lean

_ONNX_RT = None
_ONNX_SESS = {}  # name -> onnxruntime.InferenceSession
_ONNX_WARNED = False
_ONNX_DEBUG = False
_ONNX_FORCE_CPU = True  # pin ONNX to CPU for deterministic behavior
_ONNX_COUNT_DEBUG = True  # print detected counts (faces/hands/persons) when True (temporarily forced ON)

# Lazy CLIPSeg cache
_CLIPSEG_MODEL = None
_CLIPSEG_PROC = None
_CLIPSEG_DEV = "cpu"
_CLIPSEG_FORCE_CPU = True  # pin CLIPSeg to CPU to avoid device drift

# ONNX keypoints (wholebody/pose) parsing toggles (set by UI at runtime)
_ONNX_KPTS_ENABLE = False
_ONNX_KPTS_SIGMA = 2.5
_ONNX_KPTS_GAIN = 1.5
_ONNX_KPTS_CONF = 0.20

# Per-iteration spatial guidance mask (B,1,H,W) in [0,1]; used by cfg_func when enabled
CURRENT_ONNX_MASK_BCHW = None


# --- AQClip-Lite: adaptive soft quantile clipping in latent space (tile overlap) ---
@torch.no_grad()
def _aqclip_lite(latent_bchw: torch.Tensor,
                 tile: int = 32,
                 stride: int = 16,
                 alpha: float = 2.0,
                 ema_state: dict | None = None,
                 ema_beta: float = 0.8) -> tuple[torch.Tensor, dict]:
    try:
        z = latent_bchw
        B, C, H, W = z.shape
        dev, dt = z.device, z.dtype
        ksize = max(8, min(int(tile), min(H, W)))
        kstride = max(1, min(int(stride), ksize))

        # Confidence proxy: gradient magnitude on channel-mean latent
        zm = z.mean(dim=1, keepdim=True)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=dev, dtype=dt).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=dev, dtype=dt).view(1, 1, 3, 3)
        gx = F.conv2d(zm, kx, padding=1)
        gy = F.conv2d(zm, ky, padding=1)
        gmag = torch.sqrt(gx * gx + gy * gy)
        gpool = F.avg_pool2d(gmag, kernel_size=ksize, stride=kstride)
        gmax = gpool.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        Hn = (gpool / gmax).squeeze(1)  # B,h',w'
        L = Hn.shape[1] * Hn.shape[2]
        Hn = Hn.reshape(B, L)

        # Map confidence -> quantiles
        ql = 0.5 * (Hn ** 2)
        qh = 1.0 - 0.5 * ((1.0 - Hn) ** 2)

        # Per-tile mean/std
        unf = F.unfold(z, kernel_size=ksize, stride=kstride)  # B, C*ksize*ksize, L
        M = unf.shape[1]
        mu = unf.mean(dim=1).to(torch.float32)  # B,L
        var = (unf.to(torch.float32) - mu.unsqueeze(1)).pow(2).mean(dim=1)
        sigma = (var + 1e-12).sqrt()

        # Normal inverse approximation: ndtri(q) = sqrt(2)*erfinv(2q-1)
        def _ndtri(q: torch.Tensor) -> torch.Tensor:
            return (2.0 ** 0.5) * torch.special.erfinv(q.mul(2.0).sub(1.0).clamp(-0.999999, 0.999999))
        k_neg = _ndtri(ql).abs()
        k_pos = _ndtri(qh).abs()
        lo = mu - k_neg * sigma
        hi = mu + k_pos * sigma

        # EMA smooth
        if ema_state is None:
            ema_state = {}
        b = float(max(0.0, min(0.999, ema_beta)))
        if 'lo' in ema_state and 'hi' in ema_state and ema_state['lo'].shape == lo.shape:
            lo = b * ema_state['lo'] + (1.0 - b) * lo
            hi = b * ema_state['hi'] + (1.0 - b) * hi
        ema_state['lo'] = lo.detach()
        ema_state['hi'] = hi.detach()

        # Soft tanh clip (vectorized in unfold domain)
        mid = (lo + hi) * 0.5
        half = (hi - lo) * 0.5
        half = half.clamp_min(1e-6)
        y = (unf.to(torch.float32) - mid.unsqueeze(1)) / half.unsqueeze(1)
        y = torch.tanh(float(alpha) * y)
        unf_clipped = mid.unsqueeze(1) + half.unsqueeze(1) * y
        unf_clipped = unf_clipped.to(dt)

        out = F.fold(unf_clipped, output_size=(H, W), kernel_size=ksize, stride=kstride)
        ones = torch.ones((B, M, L), device=dev, dtype=dt)
        w = F.fold(ones, output_size=(H, W), kernel_size=ksize, stride=kstride).clamp_min(1e-6)
        out = out / w
        return out, ema_state
    except Exception:
        return latent_bchw, (ema_state or {})


def _try_init_onnx(models_dir: str):
    """Initialize onnxruntime and load all .onnx models in models_dir.
    We prefer GPU providers when available, but gracefully fall back to CPU.
    """
    global _ONNX_RT, _ONNX_SESS, _ONNX_WARNED
    import os
    if _ONNX_RT is None:
        try:
            import onnxruntime as ort
            _ONNX_RT = ort
        except Exception:
            if not _ONNX_WARNED:
                print("[CADE2.5][ONNX] onnxruntime not available, skipping ONNX detectors.")
                _ONNX_WARNED = True
            return False

    # Build provider preference list
    try:
        avail = set(_ONNX_RT.get_available_providers())
    except Exception:
        avail = set()
    pref = []
    if _ONNX_FORCE_CPU:
        pref = ["CPUExecutionProvider"]
    else:
        for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
            if p in avail or p == "CPUExecutionProvider":
                pref.append(p)
    if _ONNX_DEBUG:
        try:
            print(f"[CADE2.5][ONNX] Available providers: {sorted(list(avail))}")
            print(f"[CADE2.5][ONNX] Provider preference: {pref}")
        except Exception:
            pass

    # Load any .onnx in models_dir
    try:
        for fname in os.listdir(models_dir):
            if not fname.lower().endswith('.onnx'):
                continue
            if fname in _ONNX_SESS:
                continue
            full = os.path.join(models_dir, fname)
            try:
                _ONNX_SESS[fname] = _ONNX_RT.InferenceSession(full, providers=pref)
                if _ONNX_DEBUG:
                    try:
                        print(f"[CADE2.5][ONNX] Loaded model: {fname}")
                    except Exception:
                        pass
            except Exception as e:
                if not _ONNX_WARNED:
                    print(f"[CADE2.5][ONNX] failed to load {fname}: {e}")
    except Exception as e:
        if not _ONNX_WARNED:
            print(f"[CADE2.5][ONNX] cannot list models in {models_dir}: {e}")
    if not _ONNX_SESS and not _ONNX_WARNED:
        print("[CADE2.5][ONNX] No ONNX models found in", models_dir)
        _ONNX_WARNED = True
    return len(_ONNX_SESS) > 0


def _try_init_clipseg():
    """Lazy-load CLIPSeg processor + model and choose device.
    Returns True on success.
    """
    global _CLIPSEG_MODEL, _CLIPSEG_PROC, _CLIPSEG_DEV
    if (_CLIPSEG_MODEL is not None) and (_CLIPSEG_PROC is not None):
        return True
    try:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation  # type: ignore
    except Exception:
        if not globals().get("_CLIPSEG_WARNED", False):
            print("[CADE2.5][CLIPSeg] transformers not available; CLIPSeg disabled.")
            globals()["_CLIPSEG_WARNED"] = True
        return False
    try:
        _CLIPSEG_PROC = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        _CLIPSEG_MODEL = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        if _CLIPSEG_FORCE_CPU:
            _CLIPSEG_DEV = "cpu"
        else:
            _CLIPSEG_DEV = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIPSEG_MODEL = _CLIPSEG_MODEL.to(_CLIPSEG_DEV)
        _CLIPSEG_MODEL.eval()
        return True
    except Exception as e:
        print(f"[CADE2.5][CLIPSeg] failed to load model: {e}")
        return False


def _clipseg_build_mask(image_bhwc: torch.Tensor,
                        text: str,
                        preview: int = 224,
                        threshold: float = 0.4,
                        blur: float = 7.0,
                        dilate: int = 4,
                        gain: float = 1.0,
                        ref_embed: torch.Tensor | None = None,
                        clip_vision=None,
                        ref_threshold: float = 0.03) -> torch.Tensor | None:
    """Return BHWC single-channel mask [0,1] from CLIPSeg.
    - Uses cached CLIPSeg model; gracefully returns None on failure.
    - Applies optional threshold/blur/dilate and scaling gain.
    - If clip_vision + ref_embed provided, gates mask by CLIP-Vision distance.
    """
    if not text or not isinstance(text, str):
        return None
    if not _try_init_clipseg():
        return None
    try:
        # Prepare preview image (CPU PIL)
        target = int(max(16, min(1024, preview)))
        img = image_bhwc.detach().to('cpu')
        if img.ndim == 5:
            # squeeze depth if present
            if img.shape[1] == 1:
                img = img[:, 0]
            else:
                img = img[:, 0]
        B, H, W, C = img.shape
        x = img[0].movedim(-1, 0).unsqueeze(0)  # 1,C,H,W
        x = F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)
        x = x.clamp(0, 1)
        arr = (x[0].movedim(0, -1).numpy() * 255.0).astype('uint8')
        from PIL import Image  # lazy import
        pil_img = Image.fromarray(arr)

        # Run CLIPSeg
        import re
        prompts = [t.strip() for t in re.split(r"[\|,;\n]+", text) if t.strip()]
        if not prompts:
            prompts = [text.strip()]
        prompts = prompts[:8]
        inputs = _CLIPSEG_PROC(text=prompts, images=[pil_img] * len(prompts), return_tensors="pt")
        inputs = {k: v.to(_CLIPSEG_DEV) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = _CLIPSEG_MODEL(**inputs)  # type: ignore
            # logits: [N, H', W'] for N prompts
            logits = outputs.logits  # [N,h,w]
            if logits.ndim == 2:
                logits = logits.unsqueeze(0)
            prob = torch.sigmoid(logits)  # [N,h,w]
            # Soft-OR fuse across prompts
            prob = 1.0 - torch.prod(1.0 - prob.clamp(0, 1), dim=0, keepdim=True)  # [1,h,w]
            prob = prob.unsqueeze(1)  # [1,1,h,w]
        # Resize to original image size
        prob = F.interpolate(prob, size=(H, W), mode='bilinear', align_corners=False)
        m = prob[0, 0].to(dtype=image_bhwc.dtype, device=image_bhwc.device)
        # Threshold + blur (approx)
        if threshold > 0.0:
            m = torch.where(m > float(threshold), m, torch.zeros_like(m))
        # Gaussian blur via our depthwise helper
        if blur > 0.0:
            rad = int(max(1, min(7, round(blur))))
            m = _gaussian_blur_nchw(m.unsqueeze(0).unsqueeze(0), sigma=float(max(0.5, blur)), radius=rad)[0, 0]
        # Dilation via max-pool
        if int(dilate) > 0:
            k = int(dilate) * 2 + 1
            p = int(dilate)
            m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=p)[0, 0]
        # Optional CLIP-Vision gating by reference distance
        if (clip_vision is not None) and (ref_embed is not None):
            try:
                cur = _encode_clip_image(image_bhwc, clip_vision, target_res=224)
                dist = _clip_cosine_distance(cur, ref_embed)
                if dist > float(ref_threshold):
                    # up to +50% gain if distance exceeds the reference threshold
                    gate = 1.0 + min(0.5, (dist - float(ref_threshold)) * 4.0)
                    m = m * gate
            except Exception:
                pass
        m = (m * float(max(0.0, gain))).clamp(0, 1)
        out_mask = m.unsqueeze(0).unsqueeze(-1)  # BHWC with B=1,C=1
        # Best-effort release of temporaries to reduce RAM peak
        try:
            del inputs
        except Exception:
            pass
        try:
            del outputs
        except Exception:
            pass
        try:
            del logits
        except Exception:
            pass
        try:
            del prob
        except Exception:
            pass
        try:
            del pil_img
        except Exception:
            pass
        try:
            del arr
        except Exception:
            pass
        try:
            del x
        except Exception:
            pass
        try:
            del img
        except Exception:
            pass
        return out_mask
    except Exception as e:
        if not globals().get("_CLIPSEG_WARNED", False):
            print(f"[CADE2.5][CLIPSeg] mask failed: {e}")
            globals()["_CLIPSEG_WARNED"] = True
        return None


def _np_to_mask_tensor(np_map: np.ndarray, out_h: int, out_w: int, device, dtype):
    """Convert numpy heatmap [H,W] or [1,H,W] or [H,W,1] to BHWC torch mask with B=1 and resize to out_h,out_w."""
    if np_map.ndim == 3:
        np_map = np_map.reshape(np_map.shape[-2], np_map.shape[-1]) if (np_map.shape[0] == 1) else np_map.squeeze()
    if np_map.ndim != 2:
        return None
    t = torch.from_numpy(np_map.astype(np.float32))
    t = t.clamp_min(0.0)
    t = t.unsqueeze(0).unsqueeze(0)  # B=1,C=1,H,W
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    t = t.permute(0, 2, 3, 1).to(device=device, dtype=dtype)  # B,H,W,C
    return t.clamp(0, 1)


# --- Firefly/Hot-pixel remover (image space, BHWC in 0..1) ---
def _median_pool3x3_bhwc(img_bhwc: torch.Tensor) -> torch.Tensor:
    B, H, W, C = img_bhwc.shape
    x = img_bhwc.permute(0, 3, 1, 2)  # B,C,H,W
    unfold = F.unfold(x, kernel_size=3, padding=1)  # B, 9*C, H*W
    unfold = unfold.view(B, x.shape[1], 9, H, W)    # B,C,9,H,W
    med, _ = torch.median(unfold, dim=2)            # B,C,H,W
    return med.permute(0, 2, 3, 1)                  # B,H,W,C


def _despeckle_fireflies(img_bhwc: torch.Tensor,
                         thr: float = 0.985,
                         max_iso: float | None = None,
                         grad_gate: float = 0.25) -> torch.Tensor:
    try:
        dev, dt = img_bhwc.device, img_bhwc.dtype
        B, H, W, C = img_bhwc.shape
        # Scale-aware window
        s = max(H, W) / 1024.0
        k = 3 if s <= 1.1 else (5 if s <= 2.0 else 7)
        pad = k // 2
        # Value/Saturation from RGB (fast, no colorspace conv required)
        R, G, Bc = img_bhwc[..., 0], img_bhwc[..., 1], img_bhwc[..., 2]
        V = torch.maximum(R, torch.maximum(G, Bc))
        m = torch.minimum(R, torch.minimum(G, Bc))
        S = 1.0 - (m / (V + 1e-6))
        # Dynamic bright threshold from top tail; allow manual override for very high thr
        try:
            q = float(torch.quantile(V.reshape(-1), 0.9995).item())
            thr_eff = float(thr) if float(thr) >= 0.99 else max(float(thr), min(0.997, q))
        except Exception:
            thr_eff = float(thr)
        v_thr = max(0.985, thr_eff)
        s_thr = 0.06
        cand = (V > v_thr) & (S < s_thr)
        # gradient gate to protect real edges/highlights
        lum = (0.2126 * R + 0.7152 * G + 0.0722 * Bc)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=dev, dtype=dt).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=dev, dtype=dt).view(1, 1, 3, 3)
        gx = F.conv2d(lum.unsqueeze(1), kx, padding=1)
        gy = F.conv2d(lum.unsqueeze(1), ky, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy).squeeze(1)
        safe_gate = float(grad_gate) * (k / 3.0) ** 0.5
        cand = cand & (grad < safe_gate)
        if not cand.any():
            return img_bhwc
        # Prefer connected components (OpenCV) to drop small bright specks
        try:
            import cv2
            masks = []
            for b in range(cand.shape[0]):
                msk = cand[b].detach().to('cpu').numpy().astype('uint8') * 255
                num, labels, stats, _ = cv2.connectedComponentsWithStats(msk, connectivity=8)
                rem = np.zeros_like(msk, dtype='uint8')
                # Size threshold grows with k
                area_max = int(max(3, round((k * k) * 0.8)))
                for lbl in range(1, num):
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    if area <= area_max:
                        rem[labels == lbl] = 255
                masks.append(torch.from_numpy(rem > 0))
            rm = torch.stack(masks, dim=0).to(device=dev)  # B,H,W (bool)
            rm = rm.unsqueeze(-1)  # B,H,W,1
            if not rm.any():
                return img_bhwc
            med = _median_pool3x3_bhwc(img_bhwc)
            return torch.where(rm, med, img_bhwc)
        except Exception:
            # Fallback: isolation via local density
            dens = F.avg_pool2d(cand.float().unsqueeze(1), k, 1, pad).squeeze(1)
            max_iso_eff = (2.0 / (k * k)) if (max_iso is None) else float(max_iso)
            iso = cand & (dens < max_iso_eff)
            if not iso.any():
                return img_bhwc
            med = _median_pool3x3_bhwc(img_bhwc)
            return torch.where(iso.unsqueeze(-1), med, img_bhwc)
    except Exception:
        return img_bhwc


def _try_heatmap_from_outputs(outputs: list, preview_hw: tuple[int, int]):
    """Return [H,W] heatmap from model outputs if possible.
    Supports:
      - Segmentation logits/probabilities (NCHW / NHWC)
      - Keypoints arrays -> gaussian disks on points
      - Bounding boxes -> soft rectangles
    """
    if not outputs:
        return None

    Ht, Wt = int(preview_hw[0]), int(preview_hw[1])

    def to_float(arr):
        if arr.dtype not in (np.float32, np.float64):
            try:
                arr = arr.astype(np.float32)
            except Exception:
                return None
        return arr

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # 1) Prefer any spatial heatmap first
    for out in outputs:
        try:
            arr = np.asarray(out)
        except Exception:
            continue
        arr = to_float(arr)
        if arr is None:
            continue
        if arr.ndim == 4:
            n, a, b, c = arr.shape
            if c <= 4 and a >= 8 and b >= 8:
                if c == 1:
                    hm = sigmoid(arr[0, :, :, 0]) if np.max(np.abs(arr)) > 1.5 else arr[0, :, :, 0]
                else:
                    ex = np.exp(arr[0] - np.max(arr[0], axis=-1, keepdims=True))
                    prob = ex / np.clip(ex.sum(axis=-1, keepdims=True), 1e-6, None)
                    hm = 1.0 - prob[..., 0] if prob.shape[-1] > 1 else prob[..., 0]
                return hm.astype(np.float32)
            else:
                if a == 1:
                    ch = arr[0, 0]
                    hm = sigmoid(ch) if np.max(np.abs(ch)) > 1.5 else ch
                    return hm.astype(np.float32)
                else:
                    x = arr[0]
                    x = x - np.max(x, axis=0, keepdims=True)
                    ex = np.exp(x)
                    prob = ex / np.clip(np.sum(ex, axis=0, keepdims=True), 1e-6, None)
                    bg = prob[0] if prob.shape[0] > 1 else prob[0]
                    hm = 1.0 - bg
                    return hm.astype(np.float32)
        if arr.ndim == 3:
            if arr.shape[0] == 1 and arr.shape[1] >= 8 and arr.shape[2] >= 8:
                return arr[0].astype(np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 8 and arr.shape[1] >= 8:
            return arr.astype(np.float32)

    # 2) Try keypoints and boxes
    heat = np.zeros((Ht, Wt), dtype=np.float32)

    def draw_gaussian(hm, cx, cy, sigma=2.5, amp=1.0):
        r = max(1, int(3 * sigma))
        xs = np.arange(-r, r + 1, dtype=np.float32)
        ys = np.arange(-r, r + 1, dtype=np.float32)
        gx = np.exp(-(xs**2) / (2 * sigma * sigma))
        gy = np.exp(-(ys**2) / (2 * sigma * sigma))
        g = np.outer(gy, gx) * float(amp)
        x0 = int(round(cx)) - r
        y0 = int(round(cy)) - r
        x1 = x0 + g.shape[1]
        y1 = y0 + g.shape[0]
        if x1 < 0 or y1 < 0 or x0 >= Wt or y0 >= Ht:
            return
        xs0 = max(0, x0)
        ys0 = max(0, y0)
        xs1 = min(Wt, x1)
        ys1 = min(Ht, y1)
        gx0 = xs0 - x0
        gy0 = ys0 - y0
        gx1 = gx0 + (xs1 - xs0)
        gy1 = gy0 + (ys1 - ys0)
        hm[ys0:ys1, xs0:xs1] = np.maximum(hm[ys0:ys1, xs0:xs1], g[gy0:gy1, gx0:gx1])

    def draw_soft_rect(hm, x0, y0, x1, y1, edge=3.0):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if x1 <= 0 or y1 <= 0 or x0 >= Wt or y0 >= Ht:
            return
        xs0 = max(0, min(x0, x1))
        ys0 = max(0, min(y0, y1))
        xs1 = min(Wt, max(x0, x1))
        ys1 = min(Ht, max(y0, y1))
        if xs1 - xs0 <= 0 or ys1 - ys0 <= 0:
            return
        hm[ys0:ys1, xs0:xs1] = np.maximum(hm[ys0:ys1, xs0:xs1], 1.0)
        # feather edges with simple blur-like falloff
        if edge > 0:
            rad = int(edge)
            if rad > 0:
                # quick separable triangle filter
                line = np.linspace(0, 1, rad + 1, dtype=np.float32)[1:]
                for d in range(1, rad + 1):
                    w = line[d - 1]
                    if ys0 - d >= 0:
                        hm[ys0 - d:ys0, xs0:xs1] = np.maximum(hm[ys0 - d:ys0, xs0:xs1], w)
                    if ys1 + d <= Ht:
                        hm[ys1:ys1 + d, xs0:xs1] = np.maximum(hm[ys1:ys1 + d, xs0:xs1], w)
                    if xs0 - d >= 0:
                        hm[max(0, ys0 - d):min(Ht, ys1 + d), xs0 - d:xs0] = np.maximum(
                            hm[max(0, ys0 - d):min(Ht, ys1 + d), xs0 - d:xs0], w)
                    if xs1 + d <= Wt:
                        hm[max(0, ys0 - d):min(Ht, ys1 + d), xs1:xs1 + d] = np.maximum(
                            hm[max(0, ys0 - d):min(Ht, ys1 + d), xs1:xs1 + d], w)

    # Inspect outputs to find plausible keypoints/boxes
    for out in outputs:
        try:
            arr = np.asarray(out)
        except Exception:
            continue
        arr = to_float(arr)
        if arr is None:
            continue
        a = arr
        # Squeeze batch dims like [1,N,4] -> [N,4]
        while a.ndim > 2 and a.shape[0] == 1:
            a = np.squeeze(a, axis=0)
        # Keypoints: [N,2] or [N,3] or [K, N, 2/3] (relax N limit; subsample if huge)
        if a.ndim == 2 and a.shape[-1] in (2, 3):
            pts = a
        elif a.ndim == 3 and a.shape[-1] in (2, 3):
            pts = a.reshape(-1, a.shape[-1])
        else:
            pts = None
        if pts is not None:
            # Coordinates range guess: if max>1.2 -> absolute; else normalized
            maxv = float(np.nanmax(np.abs(pts[:, :2]))) if pts.size else 0.0
            for px, py, *rest in pts:
                if np.isnan(px) or np.isnan(py):
                    continue
                if maxv <= 1.2:
                    cx = float(px) * (Wt - 1)
                    cy = float(py) * (Ht - 1)
                else:
                    cx = float(px)
                    cy = float(py)
                base_sig = max(1.5, min(Ht, Wt) / 128.0)
                if _ONNX_KPTS_ENABLE:
                    draw_gaussian(heat, cx, cy, sigma=base_sig * float(_ONNX_KPTS_SIGMA), amp=float(_ONNX_KPTS_GAIN))
                else:
                    draw_gaussian(heat, cx, cy, sigma=base_sig)
            continue

        # Wholebody-style packed keypoints: [N, K*3] with triples (x,y,conf)
        if _ONNX_KPTS_ENABLE and a.ndim == 2 and a.shape[-1] >= 6 and (a.shape[-1] % 3) == 0:
            K = a.shape[-1] // 3
            if K >= 5 and K <= 256:
                # Guess coordinate range once
                with np.errstate(invalid='ignore'):
                    maxv = float(np.nanmax(np.abs(a[:, :2]))) if a.size else 0.0
                for i in range(a.shape[0]):
                    row = a[i]
                    kp = row.reshape(K, 3)
                    for (px, py, pc) in kp:
                        if np.isnan(px) or np.isnan(py):
                            continue
                        if np.isfinite(pc) and pc < float(_ONNX_KPTS_CONF):
                            continue
                        if maxv <= 1.2:
                            cx = float(px) * (Wt - 1)
                            cy = float(py) * (Ht - 1)
                        else:
                            cx = float(px)
                            cy = float(py)
                        base_sig = max(1.0, min(Ht, Wt) / 128.0)
                        draw_gaussian(heat, cx, cy, sigma=base_sig * float(_ONNX_KPTS_SIGMA), amp=float(_ONNX_KPTS_GAIN))
                continue
        # 1D edge-case: single detection row (post-processed model output)
        if _ONNX_KPTS_ENABLE and a.ndim == 1 and a.shape[0] >= 6:
            D = a.shape[0]
            parsed = False
            # try [xyxy, conf, cls, kpts] or [xyxy, conf, kpts] or [xyxy, kpts] or [kpts]
            for offset in (6, 5, 4, 0):
                t = D - offset
                if t >= 6 and (t % 3) == 0:
                    k = t // 3
                    kp = a[offset:offset + 3 * k].reshape(k, 3)
                    parsed = True
                    break
            if parsed:
                with np.errstate(invalid='ignore'):
                    maxv = float(np.nanmax(np.abs(kp[:, :2]))) if kp.size else 0.0
                for (px, py, pc) in kp:
                    if np.isnan(px) or np.isnan(py):
                        continue
                    if np.isfinite(pc) and pc < float(_ONNX_KPTS_CONF):
                        continue
                    if maxv <= 1.2:
                        cx = float(px) * (Wt - 1)
                        cy = float(py) * (Ht - 1)
                    else:
                        cx = float(px)
                        cy = float(py)
                    base_sig = max(1.0, min(Ht, Wt) / 128.0)
                    draw_gaussian(heat, cx, cy, sigma=base_sig * float(_ONNX_KPTS_SIGMA), amp=float(_ONNX_KPTS_GAIN))
                continue
        # Boxes: [N,4+] (x0,y0,x1,y1) or [N, (x,y,w,h, [conf, ...])]; relax N limit (handle YOLO-style outputs)
        if a.ndim == 2 and a.shape[-1] >= 4:
            boxes = a
        elif a.ndim == 3 and a.shape[-1] >= 4:
            # choose the smallest first two dims as N
            if a.shape[0] == 1:
                boxes = a.reshape(-1, a.shape[-1])
            else:
                boxes = a.reshape(-1, a.shape[-1])
        else:
            boxes = None
        if boxes is not None:
            # Optional score gating (try to find a confidence column)
            score = None
            if boxes.shape[-1] >= 5:
                score = boxes[:, 4]
                # If trailing columns look like probabilities in [0,1], mix the best one; if they look like class ids, ignore
                if boxes.shape[-1] > 5:
                    try:
                        tail = boxes[:, 5:]
                        tmax = np.max(tail, axis=-1)
                        # Heuristic: treat as probs if within [0,1] and not integer-like; else assume class ids
                        maybe_prob = np.all((tmax >= 0.0) & (tmax <= 1.0))
                        frac = np.abs(tmax - np.round(tmax))
                        maybe_classid = (np.mean(frac < 1e-6) > 0.9) and (np.max(tmax) >= 1.0)
                        if maybe_prob and not maybe_classid:
                            score = score * tmax
                    except Exception:
                        pass
            # Keep top-K by score if available
            if score is not None:
                try:
                    order = np.argsort(-score)
                    keep = order[: min(12, order.shape[0])]
                    boxes = boxes[keep]
                    score = score[keep]
                except Exception:
                    score = None

            xy = boxes[:, :4]
            maxv = float(np.nanmax(np.abs(xy))) if xy.size else 0.0
            if maxv <= 1.2:
                x0 = xy[:, 0] * (Wt - 1)
                y0 = xy[:, 1] * (Ht - 1)
                x1 = xy[:, 2] * (Wt - 1)
                y1 = xy[:, 3] * (Ht - 1)
            else:
                x0, y0, x1, y1 = xy[:, 0], xy[:, 1], xy[:, 2], xy[:, 3]
            # Heuristic: if many boxes are inverted, treat as [x,y,w,h]
            invalid = np.sum((x1 <= x0) | (y1 <= y0))
            if invalid > 0.5 * x0.shape[0]:
                x, y, w, h = x0, y0, x1, y1
                x0 = x - w * 0.5
                y0 = y - h * 0.5
                x1 = x + w * 0.5
                y1 = y + h * 0.5
            for i in range(x0.shape[0]):
                if score is not None and np.isfinite(score[i]) and score[i] < 0.05:
                    continue
                draw_soft_rect(heat, x0[i], y0[i], x1[i], y1[i], edge=3.0)

            # Embedded keypoints in YOLO-style rows: try to parse trailing triples (x,y,conf)
            if _ONNX_KPTS_ENABLE and boxes.shape[-1] > 6:
                D = boxes.shape[-1]
                for i in range(boxes.shape[0]):
                    row = boxes[i]
                    parsed = False
                    # try [xyxy, conf, cls, kpts] or [xyxy, conf, kpts] or [xyxy, kpts]
                    for offset in (6, 5, 4):
                        t = D - offset
                        if t >= 6 and t % 3 == 0:
                            k = t // 3
                            kp = row[offset:offset + 3 * k].reshape(k, 3)
                            parsed = True
                            break
                    if not parsed:
                        continue
                    for (px, py, pc) in kp:
                        if np.isnan(px) or np.isnan(py):
                            continue
                        if pc < float(_ONNX_KPTS_CONF):
                            continue
                        if maxv <= 1.2:
                            cx = float(px) * (Wt - 1)
                            cy = float(py) * (Ht - 1)
                        else:
                            cx = float(px)
                            cy = float(py)
                        base_sig = max(1.0, min(Ht, Wt) / 128.0)
                        draw_gaussian(heat, cx, cy, sigma=base_sig * float(_ONNX_KPTS_SIGMA), amp=float(_ONNX_KPTS_GAIN))

    if heat.max() > 0:
        heat = np.clip(heat, 0.0, 1.0)
        return heat
    return None


def _onnx_build_mask(image_bhwc: torch.Tensor, preview: int, sensitivity: float, models_dir: str, anomaly_gain: float = 1.0) -> torch.Tensor:
    """Return BHWC single-channel mask [0,1] fused from all auto-detected ONNX models.
    - Auto-loads any .onnx in models_dir.
    - Heuristically extracts spatial heatmaps; non-spatial outputs are ignored.
    - Uses soft-OR fusion across models. Models whose filename contains 'anomaly' are scaled by anomaly_gain.
    """
    if not _try_init_onnx(models_dir):
        # Explicit hint when debugging counts
        try:
            if globals().get("_ONNX_COUNT_DEBUG", False):
                print("[CADE2.5][ONNX] inactive: onnxruntime not available or init failed")
        except Exception:
            pass
        return torch.zeros((image_bhwc.shape[0], image_bhwc.shape[1], image_bhwc.shape[2], 1), device=image_bhwc.device, dtype=image_bhwc.dtype)

    if not _ONNX_SESS:
        # Explicit hint when debugging counts
        try:
            if globals().get("_ONNX_COUNT_DEBUG", False):
                print(f"[CADE2.5][ONNX] inactive: no .onnx models loaded from {models_dir}")
        except Exception:
            pass
        return torch.zeros((image_bhwc.shape[0], image_bhwc.shape[1], image_bhwc.shape[2], 1), device=image_bhwc.device, dtype=image_bhwc.dtype)

    # One-time session summary when counting is enabled
    if globals().get("_ONNX_COUNT_DEBUG", False):
        try:
            names = list(_ONNX_SESS.keys())
            short = names[:3]
            more = "..." if len(names) > 3 else ""
            print(f"[CADE2.5][ONNX] sessions={len(names)} models={short}{more}")
        except Exception:
            pass

    B, H, W, C = image_bhwc.shape
    device = image_bhwc.device
    dtype = image_bhwc.dtype

    # Process per-batch image
    masks = []
    img_cpu = image_bhwc.detach().to('cpu')
    for b in range(B):
        masks_b = []
        counts_b: dict[str, int] = {}
        if globals().get("_ONNX_COUNT_DEBUG", False):
            try:
                print(f"[CADE2.5][ONNX] build mask image[{b}] preview={int(max(16, min(1024, preview)))}")
            except Exception:
                pass
        # Prepare base BCHW tensor and default preview size; per-model resize comes later
        target = int(max(16, min(1024, preview)))
        xb = img_cpu[b].movedim(-1, 0).unsqueeze(0)  # 1,C,H,W
        if _ONNX_DEBUG:
            try:
                print(f"[CADE2.5][ONNX] Build mask for image[{b}] -> preview {target}x{target}")
            except Exception:
                pass

        for name, sess in list(_ONNX_SESS.items()):
            try:
                inputs = sess.get_inputs()
                if not inputs:
                    continue
                in_name = inputs[0].name
                in_shape = inputs[0].shape if hasattr(inputs[0], 'shape') else None
                # Choose layout automatically based on the presence of channel dim=3
                if isinstance(in_shape, (list, tuple)) and len(in_shape) == 4:
                    dim_vals = []
                    for d in in_shape:
                        try:
                            dim_vals.append(int(d))
                        except Exception:
                            dim_vals.append(-1)
                    if dim_vals[-1] == 3:
                        layout = "NHWC"
                    else:
                        layout = "NCHW"
                else:
                    layout = "NCHW?"
                if _ONNX_DEBUG:
                    try:
                        print(f"[CADE2.5][ONNX] Model '{name}' in_shape={in_shape} layout={layout}")
                    except Exception:
                        pass
                # Build per-model sized variants (respect fixed input shapes when provided)
                th, tw = target, target
                try:
                    if isinstance(in_shape, (list, tuple)) and len(in_shape) == 4:
                        dd = []
                        for d in in_shape:
                            try:
                                dd.append(int(d))
                            except Exception:
                                dd.append(-1)
                        if layout == "NCHW" and dd[2] > 8 and dd[3] > 8:
                            th, tw = int(dd[2]), int(dd[3])
                        if layout.startswith("NHWC") and dd[1] > 8 and dd[2] > 8:
                            th, tw = int(dd[1]), int(dd[2])
                except Exception:
                    th, tw = target, target

                x_stretch_m = F.interpolate(xb, size=(th, tw), mode='bilinear', align_corners=False).clamp(0, 1)
                if th == tw:
                    x_letter_m = _letterbox_nchw(xb, th).clamp(0, 1)
                else:
                    sq = max(th, tw)
                    x_letter_sq = _letterbox_nchw(xb, sq).clamp(0, 1)
                    x_letter_m = F.interpolate(x_letter_sq, size=(th, tw), mode='bilinear', align_corners=False).clamp(0, 1)

                variants = [
                    ("stretch-RGB", x_stretch_m),
                    ("letterbox-RGB", x_letter_m),
                    ("stretch-BGR", x_stretch_m[:, [2, 1, 0], :, :]),
                    ("letterbox-BGR", x_letter_m[:, [2, 1, 0], :, :]),
                ]

                # Try multiple input variants and scales
                hm = None
                chosen = None
                for vname, vx in variants:
                    if layout.startswith("NHWC"):
                        xin = vx.permute(0, 2, 3, 1)
                    else:
                        xin = vx
                    for scale in (1.0, 255.0):
                        inp = (xin * float(scale)).numpy().astype(np.float32)
                        feed = {in_name: inp}
                        outs = sess.run(None, feed)
                        if _ONNX_DEBUG:
                            try:
                                shapes = []
                                for o in outs:
                                    try:
                                        shapes.append(tuple(np.asarray(o).shape))
                                    except Exception:
                                        shapes.append("?")
                                print(f"[CADE2.5][ONNX] '{name}' {vname} scale={scale} -> outs shapes {shapes}")
                            except Exception:
                                pass
                        hm = _try_heatmap_from_outputs(outs, (target, target))
                        if _ONNX_DEBUG:
                            try:
                                if hm is None:
                                    print(f"[CADE2.5][ONNX] '{name}' {vname} scale={scale}: no spatial heatmap detected")
                                else:
                                    print(f"[CADE2.5][ONNX] '{name}' {vname} scale={scale}: heat stats min={np.min(hm):.4f} max={np.max(hm):.4f} mean={np.mean(hm):.4f}")
                            except Exception:
                                pass
                        if hm is not None and np.max(hm) > 0:
                            chosen = (vname, scale)
                            break
                    if hm is not None and np.max(hm) > 0:
                        break
                if hm is None:
                    continue
                # Scale by sensitivity and optional anomaly gain
                gain = float(max(0.0, sensitivity))
                if 'anomaly' in name.lower():
                    gain *= float(max(0.0, anomaly_gain))
                hm = np.clip(hm * gain, 0.0, 1.0)
                # Heuristic rejection of stripe artifacts (horizontal/vertical banding)
                try:
                    rm = np.mean(hm, axis=1)  # row means (H)
                    cm = np.mean(hm, axis=0)  # col means (W)
                    rstd = float(np.std(rm))
                    cstd = float(np.std(cm))
                    zig_r = float(np.mean(np.abs(np.diff(rm))))
                    zig_c = float(np.mean(np.abs(np.diff(cm))))
                    horiz_bands = (rstd > 10.0 * max(cstd, 1e-6)) and (zig_r > 0.02)
                    vert_bands  = (cstd > 10.0 * max(rstd, 1e-6)) and (zig_c > 0.02)
                    if horiz_bands or vert_bands:
                        if _ONNX_DEBUG:
                            print(f"[CADE2.5][ONNX] '{name}' rejected as stripe artifact (rstd={rstd:.4f} cstd={cstd:.4f} zig_r={zig_r:.4f} zig_c={zig_c:.4f})")
                        hm = None
                except Exception:
                    pass
                tmask = _np_to_mask_tensor(hm, H, W, device, dtype)
                if tmask is not None:
                    masks_b.append(tmask)
                    # Optional counting for debugging: derive counts from raw outputs if possible
                    try:
                        if globals().get("_ONNX_COUNT_DEBUG", False):
                            lower = name.lower()
                            is_face = ("face" in lower)
                            is_hand = ("hand" in lower)
                            is_pose = any(w in lower for w in ["wholebody", "pose", "person", "body"]) and not (is_face or is_hand)

                            boxes_cnt = 0
                            persons_cnt = 0
                            wrists_cnt = 0

                            try:
                                for outx in outs:
                                    arr0 = np.asarray(outx)
                                    if arr0 is None:
                                        continue
                                    a = arr0.astype(np.float32, copy=False)
                                    while a.ndim > 2 and a.shape[0] == 1:
                                        try:
                                            a = np.squeeze(a, axis=0)
                                        except Exception:
                                            break

                                    # Face/Hand: count only plausible (N,D) detection tables, not spatial maps/grids
                                    if (is_face or is_hand) and a.ndim == 2:
                                        N, D = a.shape
                                        if D >= 4 and D <= 64 and N <= 512 and not (N >= 32 and D >= 32):
                                            n = N
                                            # Try to locate a confidence column
                                            conf = None
                                            if D >= 5:
                                                c = a[:, 4]
                                                if np.all(np.isfinite(c)) and 0.0 <= float(np.nanmin(c)) and float(np.nanmax(c)) <= 1.0:
                                                    conf = c
                                            if conf is None:
                                                c = a[:, -1]
                                                if np.all(np.isfinite(c)) and 0.0 <= float(np.nanmin(c)) and float(np.nanmax(c)) <= 1.0:
                                                    conf = c
                                            if conf is not None:
                                                n = int(np.sum(conf >= 0.25))
                                            boxes_cnt = max(boxes_cnt, int(n))

                                    # Pose: prefer keypoints formats only
                                    if is_pose:
                                        # Packed per row [N, K*3]
                                        if a.ndim == 2 and a.shape[-1] >= 6 and (a.shape[-1] % 3) == 0:
                                            persons_cnt = max(persons_cnt, int(a.shape[0]))
                                            try:
                                                K = a.shape[-1] // 3
                                                if K >= 11:
                                                    lw = a[:, 9*3 + 2]
                                                    rw = a[:, 10*3 + 2]
                                                    wrists_cnt = max(wrists_cnt, int(np.sum(lw >= 0.2) + np.sum(rw >= 0.2)))
                                            except Exception:
                                                pass
                                        # [N,K,2/3]
                                        if a.ndim == 3 and a.shape[-1] in (2, 3):
                                            persons_cnt = max(persons_cnt, int(a.shape[0]))
                                            try:
                                                if a.shape[1] >= 11 and a.shape[-1] == 3:
                                                    lw = a[:, 9, 2]
                                                    rw = a[:, 10, 2]
                                                    wrists_cnt = max(wrists_cnt, int(np.sum(lw >= 0.2) + np.sum(rw >= 0.2)))
                                            except Exception:
                                                pass
                            except Exception:
                                pass

                            # Map to categories by model name using derived counts
                            if is_face:
                                if boxes_cnt > 0:
                                    counts_b["faces"] = counts_b.get("faces", 0) + boxes_cnt
                            elif is_hand:
                                if boxes_cnt > 0:
                                    counts_b["hands"] = counts_b.get("hands", 0) + boxes_cnt
                            elif is_pose:
                                if persons_cnt > 0:
                                    counts_b["persons"] = counts_b.get("persons", 0) + persons_cnt
                                # Fallback hands from wrists or 2 per person
                                if wrists_cnt > 0:
                                    counts_b["hands"] = counts_b.get("hands", 0) + wrists_cnt
                                elif persons_cnt > 0:
                                    counts_b["hands"] = counts_b.get("hands", 0) + (2 * persons_cnt)
                    except Exception:
                        pass
                    if _ONNX_DEBUG:
                        try:
                            area = float(tmask.movedim(-1,1).mean().item())
                            if chosen is not None:
                                vname, scale = chosen
                                print(f"[CADE2.5][ONNX] '{name}' via {vname} x{scale} area={area:.4f}")
                            else:
                                print(f"[CADE2.5][ONNX] '{name}' contribution area={area:.4f}")
                        except Exception:
                            pass
            except Exception:
                # Ignore failing models
                continue
        if not masks_b:
            masks.append(torch.zeros((1, H, W, 1), device=device, dtype=dtype))
            if _ONNX_DEBUG or globals().get("_ONNX_COUNT_DEBUG", False):
                try:
                    print(f"[CADE2.5][ONNX] Detected (image[{b}]): none (no contributing models)")
                except Exception:
                    pass
        else:
            # Soft-OR fusion: 1 - prod(1 - m)
            stack = torch.stack([masks_b[i] for i in range(len(masks_b))], dim=0)  # M,1,H,W,1? actually B dims kept as 1
            fused = 1.0 - torch.prod(1.0 - stack.clamp(0, 1), dim=0)
            # Light smoothing via bilinear down/up (anti alias)
            ch = fused.permute(0, 3, 1, 2)  # B=1,C=1,H,W
            dd = F.interpolate(ch, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False)
            uu = F.interpolate(dd, size=(H, W), mode='bilinear', align_corners=False)
            fused = uu.permute(0, 2, 3, 1).clamp(0, 1)
            if _ONNX_DEBUG or globals().get("_ONNX_COUNT_DEBUG", False):
                try:
                    area = float(fused.movedim(-1,1).mean().item())
                    if _ONNX_DEBUG:
                        print(f"[CADE2.5][ONNX] Fused area (image[{b}])={area:.4f}")
                    # Print per-image counts if requested
                    if globals().get("_ONNX_COUNT_DEBUG", False):
                        if counts_b:
                            faces = counts_b.get("faces", 0)
                            hands = counts_b.get("hands", 0)
                            persons = counts_b.get("persons", 0)
                            print(f"[CADE2.5][ONNX] Detected (image[{b}]): faces={faces} hands={hands} persons={persons}")
                        else:
                            print(f"[CADE2.5][ONNX] Detected (image[{b}]): counts unavailable (no categories or cv2 missing), area={area:.4f}")
                except Exception:
                    pass
            masks.append(fused)

    return torch.cat(masks, dim=0)

def _sampler_names():
    try:
        import comfy.samplers
        return comfy.samplers.KSampler.SAMPLERS
    except Exception:
        return ["euler"]


def _scheduler_names():
    try:
        import comfy.samplers
        scheds = list(comfy.samplers.KSampler.SCHEDULERS)
        if "MGHybrid" not in scheds:
            scheds.append("MGHybrid")
        return scheds
    except Exception:
        return ["normal", "MGHybrid"]


def safe_decode(vae, lat, tile=512, ovlp=64):
    # Ensure we don't build autograd graphs during final decode steps
    with torch.inference_mode():
        h, w = lat["samples"].shape[-2:]
        if min(h, w) > 1024:
            # Increase overlap for ultra-hires to reduce seam artifacts
            ov = 128 if max(h, w) > 2048 else ovlp
            out = vae.decode_tiled(lat["samples"], tile_x=tile, tile_y=tile, overlap=ov)
        else:
            out = vae.decode(lat["samples"])
    # Move to CPU and detach to release VRAM/graphs early
    try:
        try:
            out = out.detach()
        except Exception:
            pass
        out_cpu = out
        try:
            out_cpu = out_cpu.to('cpu')
        except Exception:
            pass
        try:
            del out
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return out_cpu
    except Exception:
        return out


def safe_encode(vae, img, tile=512, ovlp=64):
    import math, torch.nn.functional as F
    h, w = img.shape[1:3]
    try:
        stride = int(vae.spacial_compression_decode())
    except Exception:
        stride = 8
    if stride <= 0:
        stride = 8
    def _align_up(x, s):
        return int(((x + s - 1) // s) * s)
    Ht = _align_up(h, stride)
    Wt = _align_up(w, stride)
    x = img
    if (Ht != h) or (Wt != w):
        # pad on bottom/right using replicate to avoid black borders
        pad_h = Ht - h
        pad_w = Wt - w
        x_nchw = img.movedim(-1, 1)
        x_nchw = F.pad(x_nchw, (0, pad_w, 0, pad_h), mode='replicate')
        x = x_nchw.movedim(1, -1)
    with torch.inference_mode():
        if min(Ht, Wt) > 1024:
            ov = 128 if max(Ht, Wt) > 2048 else ovlp
            out = vae.encode_tiled(x[:, :, :, :3], tile_x=tile, tile_y=tile, overlap=ov)
        else:
            out = vae.encode(x[:, :, :, :3])
    try:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    except Exception:
        pass
    return out
    


def _gaussian_kernel(kernel_size: int, sigma: float, device=None):
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, kernel_size, device=device),
        torch.linspace(-1, 1, kernel_size, device=device),
        indexing="ij",
    )
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()


def _sharpen_image(image: torch.Tensor, sharpen_radius: int, sigma: float, alpha: float):
    if sharpen_radius == 0:
        return (image,)

    image = image.to(model_management.get_torch_device())
    batch_size, height, width, channels = image.shape

    kernel_size = sharpen_radius * 2 + 1
    kernel = _gaussian_kernel(kernel_size, sigma, device=image.device) * -(alpha * 10)
    kernel = kernel.to(dtype=image.dtype)
    center = kernel_size // 2
    kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
    kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

    tensor_image = image.permute(0, 3, 1, 2)
    tensor_image = F.pad(tensor_image, (sharpen_radius, sharpen_radius, sharpen_radius, sharpen_radius), 'reflect')
    sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)[:, :, sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
    sharpened = sharpened.permute(0, 2, 3, 1)

    result = torch.clamp(sharpened, 0, 1)
    return (result.to(model_management.intermediate_device()),)


def _encode_clip_image(image: torch.Tensor, clip_vision, target_res: int) -> torch.Tensor:
    # image: BHWC in [0,1]
    img = image.movedim(-1, 1)  # BCHW
    img = F.interpolate(img, size=(target_res, target_res), mode="bilinear", align_corners=False)
    img = (img * 2.0) - 1.0
    embeds = clip_vision.encode_image(img)["image_embeds"]
    embeds = F.normalize(embeds, dim=-1)
    return embeds


def _clip_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        m = min(a.shape[0], b.shape[0])
        a = a[:m]
        b = b[:m]
    sim = (a * b).sum(dim=-1).mean().clamp(-1.0, 1.0).item()
    return 1.0 - sim


def _soft_symmetry_blend(image_bhwc: torch.Tensor,
                         mask_bhwc: torch.Tensor,
                         alpha: float = 0.03,
                         lp_sigma: float = 1.5) -> torch.Tensor:
    """Gently mix a mirrored low-frequency component inside mask.
    - image_bhwc: [B,H,W,C] in [0,1]
    - mask_bhwc: [B,H,W,1] in [0,1]
    """
    try:
        if image_bhwc is None or mask_bhwc is None:
            return image_bhwc
        if image_bhwc.ndim != 4 or mask_bhwc.ndim != 4:
            return image_bhwc
        B, H, W, C = image_bhwc.shape
        if C < 3:
            return image_bhwc
        # Mirror along width
        mirror = torch.flip(image_bhwc, dims=[2])
        # Low-pass both
        x = image_bhwc.movedim(-1, 1)
        y = mirror.movedim(-1, 1)
        rad = max(1, int(round(lp_sigma)))
        x_lp = _gaussian_blur_nchw(x, sigma=float(lp_sigma), radius=rad)
        y_lp = _gaussian_blur_nchw(y, sigma=float(lp_sigma), radius=rad)
        # High-pass from original
        hp = x - x_lp
        # Blend LPs inside mask
        m = mask_bhwc.movedim(-1, 1).clamp(0, 1)
        a = float(max(0.0, min(0.2, alpha)))
        base = (1.0 - a * m) * x_lp + (a * m) * y_lp
        res = (base + hp).movedim(1, -1).clamp(0, 1)
        return res.to(image_bhwc.dtype)
    except Exception:
        return image_bhwc


def _gaussian_blur_nchw(x: torch.Tensor, sigma: float = 1.0, radius: int = 1) -> torch.Tensor:
    """Lightweight depthwise Gaussian blur for NCHW or NCDHW tensors.
    Uses reflect padding and a normalized kernel built by _gaussian_kernel.
    """
    if radius <= 0:
        return x
    ksz = radius * 2 + 1
    kernel = _gaussian_kernel(ksz, sigma, device=x.device).to(dtype=x.dtype)
    # Support 5D by folding depth into batch
    if x.ndim == 5:
        b, c, d, h, w = x.shape
        x2 = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        k = kernel.repeat(c, 1, 1).unsqueeze(1)  # [C,1,K,K]
        x_pad = F.pad(x2, (radius, radius, radius, radius), mode='reflect')
        y2 = F.conv2d(x_pad, k, padding=0, groups=c)
        y = y2.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)
        return y
    # 4D path
    if x.ndim == 4:
        b, c, h, w = x.shape
        k = kernel.repeat(c, 1, 1).unsqueeze(1)  # [C,1,K,K]
        x_pad = F.pad(x, (radius, radius, radius, radius), mode='reflect')
        y = F.conv2d(x_pad, k, padding=0, groups=c)
        return y
    # Fallback: return input if unexpected dims
    return x


def _letterbox_nchw(x: torch.Tensor, target: int, pad_val: float = 114.0 / 255.0) -> torch.Tensor:
    """Letterbox a BCHW tensor to target x target with constant padding (YOLO-style).
    Preserves aspect ratio, centers content, pads with pad_val.
    """
    if x.ndim != 4:
        return F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)
    b, c, h, w = x.shape
    if h == 0 or w == 0:
        return F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)
    r = float(min(target / max(1, h), target / max(1, w)))
    nh = max(1, int(round(h * r)))
    nw = max(1, int(round(w * r)))
    y = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
    pt = (target - nh) // 2
    pb = target - nh - pt
    pl = (target - nw) // 2
    pr = target - nw - pl
    if pt < 0 or pb < 0 or pl < 0 or pr < 0:
        # Fallback stretch if rounding went weird
        return F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)
    return F.pad(y, (pl, pr, pt, pb), mode='constant', value=float(pad_val))


def _fdg_filter(delta: torch.Tensor, low_gain: float, high_gain: float, sigma: float = 1.0, radius: int = 1) -> torch.Tensor:
    """Frequency-Decoupled Guidance: split delta into low/high bands and reweight.
    delta: [B,C,H,W]
    """
    low = _gaussian_blur_nchw(delta, sigma=sigma, radius=radius)
    high = delta - low
    return low * float(low_gain) + high * float(high_gain)


def _fdg_energy_fraction(delta: torch.Tensor, sigma: float = 1.0, radius: int = 1) -> torch.Tensor:
    """Return fraction of high-frequency energy: E_high / (E_low + E_high)."""
    low = _gaussian_blur_nchw(delta, sigma=sigma, radius=radius)
    high = delta - low
    e_low = (low * low).mean(dim=(1, 2, 3), keepdim=True)
    e_high = (high * high).mean(dim=(1, 2, 3), keepdim=True)
    frac = e_high / (e_low + e_high + 1e-8)
    return frac


def _fdg_split_three(delta: torch.Tensor,
                     sigma_lo: float = 0.8,
                     sigma_hi: float = 2.0,
                     radius: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sig_lo = float(max(0.05, sigma_lo))
    sig_hi = float(max(sig_lo + 1e-3, sigma_hi))
    blur_lo = _gaussian_blur_nchw(delta, sigma=sig_lo, radius=radius)
    blur_hi = _gaussian_blur_nchw(delta, sigma=sig_hi, radius=radius)
    low = blur_hi
    mid = blur_lo - blur_hi
    high = delta - blur_lo
    return low, mid, high


def _wrap_model_with_guidance(model, guidance_mode: str, rescale_multiplier: float, momentum_beta: float, cfg_curve: float, perp_damp: float, use_zero_init: bool=False, zero_init_steps: int=0, fdg_low: float = 0.6, fdg_high: float = 1.3, fdg_sigma: float = 1.0, ze_zero_steps: int = 0, ze_adaptive: bool = False, ze_r_switch_hi: float = 0.6, ze_r_switch_lo: float = 0.45, fdg_low_adaptive: bool = False, fdg_low_min: float = 0.45, fdg_low_max: float = 0.7, fdg_ema_beta: float = 0.8, use_local_mask: bool = False, mask_inside: float = 1.0, mask_outside: float = 1.0,
                                midfreq_enable: bool = False, midfreq_gain: float = 0.0, midfreq_sigma_lo: float = 0.8, midfreq_sigma_hi: float = 2.0,
                                mahiro_plus_enable: bool = False, mahiro_plus_strength: float = 0.5,
                                eps_scale_enable: bool = False, eps_scale: float = 0.0,
                                cfg_sched_type: str = "off", cfg_sched_min: float = 0.0, cfg_sched_max: float = 0.0,
                                cfg_sched_gamma: float = 1.5, cfg_sched_u_pow: float = 1.0):

    """Clone model and attach a cfg mixing function implementing RescaleCFG/FDG, CFGZero*/FD, or hybrid ZeResFDG.
    guidance_mode: 'default' | 'RescaleCFG' | 'RescaleFDG' | 'CFGZero*' | 'CFGZeroFD' | 'ZeResFDG'
    """
    if guidance_mode == "default":
        return model
    m = model.clone()

    # State for momentum and sigma normalization across steps
    prev_delta = {"t": None}
    sigma_seen = {"max": None, "min": None}
    # Spectral switching/adaptive low state
    spec_state = {"ema": None, "mode": "CFGZeroFD"}

    # External reset hook to emulate fresh state per iteration without re-cloning the model
    def _reset_state():
        try:
            prev_delta["t"] = None
            sigma_seen["max"] = None
            sigma_seen["min"] = None
            spec_state["ema"] = None
            spec_state["mode"] = "CFGZeroFD"
        except Exception:
            pass
    try:
        setattr(m, "mg_guidance_reset", _reset_state)
    except Exception:
        pass

    def cfg_func(args):
        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]
        sigma = args.get("sigma", None)
        x_orig = args.get("input", None)

        # Local spatial gain from CURRENT_ONNX_MASK_BCHW, resized to cond spatial size
        def _local_gain_for(hw):
            if not bool(use_local_mask):
                return None
            m = globals().get("CURRENT_ONNX_MASK_BCHW", None)
            if m is None:
                return None
            try:
                Ht, Wt = int(hw[0]), int(hw[1])
                g = m.to(device=cond.device, dtype=cond.dtype)
                if g.shape[-2] != Ht or g.shape[-1] != Wt:
                    g = F.interpolate(g, size=(Ht, Wt), mode='bilinear', align_corners=False)
                gi = float(mask_inside)
                go = float(mask_outside)
                gain = g * gi + (1.0 - g) * go  # [B,1,H,W]
                return gain
            except Exception:
                return None

        # Allow hybrid switch per-step
        mode = guidance_mode
        if guidance_mode == "ZeResFDG":
            if bool(ze_adaptive):
                try:
                    delta_raw = args["cond"] - args["uncond"]
                    frac_b = _fdg_energy_fraction(delta_raw, sigma=float(fdg_sigma), radius=1)  # [B,1,1,1]
                    frac = float(frac_b.mean().clamp(0.0, 1.0).item())
                except Exception:
                    frac = 0.0
                if spec_state["ema"] is None:
                    spec_state["ema"] = frac
                else:
                    beta = float(max(0.0, min(0.99, fdg_ema_beta)))
                    spec_state["ema"] = beta * float(spec_state["ema"]) + (1.0 - beta) * frac
                r = float(spec_state["ema"])
                # Hysteresis: switch up/down with two thresholds
                if spec_state["mode"] == "CFGZeroFD" and r >= float(ze_r_switch_hi):
                    spec_state["mode"] = "RescaleFDG"
                elif spec_state["mode"] == "RescaleFDG" and r <= float(ze_r_switch_lo):
                    spec_state["mode"] = "CFGZeroFD"
                mode = spec_state["mode"]
            else:
                try:
                    sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
                    matched_idx = (sigmas == args["timestep"][0]).nonzero()
                    if len(matched_idx) > 0:
                        current_idx = matched_idx.item()
                    else:
                        current_idx = 0
                except Exception:
                    current_idx = 0
                mode = "CFGZeroFD" if current_idx <= int(ze_zero_steps) else "RescaleFDG"

        if mode in ("CFGZero*", "CFGZeroFD"):
            # Optional zero-init for the first N steps
            if use_zero_init and "model_options" in args and args.get("timestep") is not None:
                try:
                    sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
                    matched_idx = (sigmas == args["timestep"][0]).nonzero()
                    if len(matched_idx) > 0:
                        current_idx = matched_idx.item()
                    else:
                        # fallback lookup
                        current_idx = 0
                    if current_idx <= int(zero_init_steps):
                        return cond * 0.0
                except Exception:
                    pass
            # Project cond onto uncond subspace (batch-wise alpha)
            bsz = cond.shape[0]
            pos_flat = cond.view(bsz, -1)
            neg_flat = uncond.view(bsz, -1)
            dot = torch.sum(pos_flat * neg_flat, dim=1, keepdim=True)
            denom = torch.sum(neg_flat * neg_flat, dim=1, keepdim=True).clamp_min(1e-8)
            alpha = (dot / denom).view(bsz, *([1] * (cond.dim() - 1)))
            resid = cond - uncond * alpha
            # Adaptive low gain if enabled
            low_gain_eff = float(fdg_low)
            if bool(fdg_low_adaptive) and spec_state["ema"] is not None:
                s = float(spec_state["ema"])  # 0..1 fraction of high-frequency energy
                lmin = float(fdg_low_min)
                lmax = float(fdg_low_max)
                low_gain_eff = max(0.0, min(2.0, lmin + (lmax - lmin) * s))
            if mode == "CFGZeroFD":
                resid = _fdg_filter(resid, low_gain=low_gain_eff, high_gain=fdg_high, sigma=float(fdg_sigma), radius=1)
            # Apply local spatial gain to residual guidance
            lg = _local_gain_for((cond.shape[-2], cond.shape[-1]))
            if lg is not None:
                resid = resid * lg.expand(-1, resid.shape[1], -1, -1)
            noise_pred = uncond * alpha + cond_scale * resid
            return noise_pred

        # RescaleCFG/FDG path (with optional momentum/perp damping and S-curve shaping)
        delta = cond - uncond
        pd = float(max(0.0, min(1.0, perp_damp)))
        if pd > 0.0 and (prev_delta["t"] is not None) and (prev_delta["t"].shape == delta.shape):
            prev = prev_delta["t"]
            denom = (prev * prev).sum(dim=(1,2,3), keepdim=True).clamp_min(1e-6)
            coeff = ((delta * prev).sum(dim=(1,2,3), keepdim=True) / denom)
            parallel = coeff * prev
            delta = delta - pd * parallel
        beta = float(max(0.0, min(0.95, momentum_beta)))
        if beta > 0.0:
            if prev_delta["t"] is None or prev_delta["t"].shape != delta.shape:
                prev_delta["t"] = delta.detach()
            delta = (1.0 - beta) * delta + beta * prev_delta["t"]
            prev_delta["t"] = delta.detach()
            cond = uncond + delta
        else:
            prev_delta["t"] = delta.detach()
        # After momentum: optionally apply FDG and rebuild cond
        if mode == "RescaleFDG":
            # Adaptive low gain if enabled
            low_gain_eff = float(fdg_low)
            if bool(fdg_low_adaptive) and spec_state["ema"] is not None:
                s = float(spec_state["ema"])  # 0..1
                lmin = float(fdg_low_min)
                lmax = float(fdg_low_max)
                low_gain_eff = max(0.0, min(2.0, lmin + (lmax - lmin) * s))
            delta_fdg = _fdg_filter(delta, low_gain=low_gain_eff, high_gain=fdg_high, sigma=float(fdg_sigma), radius=1)
            # Optional mid-frequency emphasis blended on top
            if bool(midfreq_enable) and abs(float(midfreq_gain)) > 1e-6:
                lo_b, mid_b, hi_b = _fdg_split_three(delta, sigma_lo=float(midfreq_sigma_lo), sigma_hi=float(midfreq_sigma_hi), radius=1)
                lg = _local_gain_for((cond.shape[-2], cond.shape[-1]))
                if lg is not None:
                    mid_b = mid_b * lg.expand(-1, mid_b.shape[1], -1, -1)
                delta_fdg = delta_fdg + float(midfreq_gain) * mid_b
            lg = _local_gain_for((cond.shape[-2], cond.shape[-1]))
            if lg is not None:
                delta_fdg = delta_fdg * lg.expand(-1, delta_fdg.shape[1], -1, -1)
            cond = uncond + delta_fdg
        else:
            lg = _local_gain_for((cond.shape[-2], cond.shape[-1]))
            if lg is not None:
                delta = delta * lg.expand(-1, delta.shape[1], -1, -1)
            cond = uncond + delta

        cond_scale_eff = cond_scale
        curve_gain = 1.0
        if cfg_curve > 0.0 and (sigma is not None):
            s = sigma
            if s.ndim > 1:
                s = s.flatten()
            s_max = float(torch.max(s).item())
            s_min = float(torch.min(s).item())
            if sigma_seen["max"] is None:
                sigma_seen["max"] = s_max
                sigma_seen["min"] = s_min
            else:
                sigma_seen["max"] = max(sigma_seen["max"], s_max)
                sigma_seen["min"] = min(sigma_seen["min"], s_min)
            lo = max(1e-6, sigma_seen["min"])
            hi = max(lo * (1.0 + 1e-6), sigma_seen["max"])
            t = (torch.log(s + 1e-6) - torch.log(torch.tensor(lo, device=sigma.device))) / (torch.log(torch.tensor(hi, device=sigma.device)) - torch.log(torch.tensor(lo, device=sigma.device)) + 1e-6)
            t = t.clamp(0.0, 1.0)
            k = 6.0 * float(cfg_curve)
            s_curve = torch.tanh((t - 0.5) * k)
            g = 1.0 + 0.15 * float(cfg_curve) * s_curve
            if g.ndim > 0:
                g = g.mean().item()
            curve_gain = float(g)
            cond_scale_eff = cond_scale * curve_gain

        # Per-step CFG schedule (cosine/warmup/U) using normalized sigma progress
        if isinstance(cfg_sched_type, str) and cfg_sched_type.lower() != "off" and (sigma is not None):
            try:
                s = sigma
                if s.ndim > 1:
                    s = s.flatten()
                s_max = float(torch.max(s).item())
                s_min = float(torch.min(s).item())
                if sigma_seen["max"] is None:
                    sigma_seen["max"] = s_max
                    sigma_seen["min"] = s_min
                else:
                    sigma_seen["max"] = max(sigma_seen["max"], s_max)
                    sigma_seen["min"] = min(sigma_seen["min"], s_min)
                lo = max(1e-6, sigma_seen["min"])
                hi = max(lo * (1.0 + 1e-6), sigma_seen["max"])
                t = (torch.log(s + 1e-6) - torch.log(torch.tensor(lo, device=sigma.device))) / (torch.log(torch.tensor(hi, device=sigma.device)) - torch.log(torch.tensor(lo, device=sigma.device)) + 1e-6)
                t = t.clamp(0.0, 1.0)
                if t.ndim > 0:
                    t_val = float(t.mean().item())
                else:
                    t_val = float(t.item())
                cmin = float(max(0.0, cfg_sched_min))
                cmax = float(max(cmin, cfg_sched_max))
                tp = cfg_sched_type.lower()
                if tp == "cosine":
                    import math
                    cfg_val = cmax - (cmax - cmin) * 0.5 * (1.0 + math.cos(math.pi * t_val))
                elif tp in ("warmup", "warm-up", "linear"):
                    g = float(max(0.0, min(1.0, t_val))) ** float(max(0.1, cfg_sched_gamma))
                    cfg_val = cmin + (cmax - cmin) * g
                elif tp in ("u", "u-shape", "ushape"):
                    # edges high, middle low; power to control concavity
                    e = 4.0 * (t_val - 0.5) * (t_val - 0.5)
                    e = float(min(1.0, max(0.0, e)))
                    e = e ** float(max(0.1, cfg_sched_u_pow))
                    cfg_val = cmin + (cmax - cmin) * e
                else:
                    cfg_val = cond_scale_eff
                # Keep curve shaping as a multiplier on top of scheduled absolute value
                shape = (cond_scale_eff / float(cond_scale)) if float(cond_scale) != 0.0 else 1.0
                cond_scale_eff = float(cfg_val) * float(shape)
            except Exception:
                pass

        # Epsilon scaling (exposure bias correction): early steps get multiplier closer to (1 + eps_scale)
        eps_mult = 1.0
        if bool(eps_scale_enable) and (sigma is not None):
            try:
                s = sigma
                if s.ndim > 1:
                    s = s.flatten()
                s_max = float(torch.max(s).item())
                s_min = float(torch.min(s).item())
                if sigma_seen["max"] is None:
                    sigma_seen["max"] = s_max
                    sigma_seen["min"] = s_min
                else:
                    sigma_seen["max"] = max(sigma_seen["max"], s_max)
                    sigma_seen["min"] = min(sigma_seen["min"], s_min)
                lo = max(1e-6, sigma_seen["min"])
                hi = max(lo * (1.0 + 1e-6), sigma_seen["max"])
                t_lin = (torch.log(s + 1e-6) - torch.log(torch.tensor(lo, device=sigma.device))) / (torch.log(torch.tensor(hi, device=sigma.device)) - torch.log(torch.tensor(lo, device=sigma.device)) + 1e-6)
                t_lin = t_lin.clamp(0.0, 1.0)
                w_early = (1.0 - t_lin).mean().item()
                eps_mult = float(1.0 + eps_scale * w_early)
            except Exception:
                eps_mult = float(1.0 + eps_scale)

        if sigma is None or x_orig is None:
            return uncond + cond_scale * (cond - uncond)
        sigma_ = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
        x = x_orig / (sigma_ * sigma_ + 1.0)
        v_cond = ((x - (x_orig - cond)) * (sigma_ ** 2 + 1.0) ** 0.5) / (sigma_)
        v_uncond = ((x - (x_orig - uncond)) * (sigma_ ** 2 + 1.0) ** 0.5) / (sigma_)
        v_cfg = v_uncond + cond_scale_eff * (v_cond - v_uncond)
        ro_pos = torch.std(v_cond, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(v_cfg, dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        v_rescaled = v_cfg * (ro_pos / ro_cfg)
        v_final = float(rescale_multiplier) * v_rescaled + (1.0 - float(rescale_multiplier)) * v_cfg
        eps = x_orig - (x - (v_final * eps_mult) * sigma_ / (sigma_ * sigma_ + 1.0) ** 0.5)
        return eps

    m.set_model_sampler_cfg_function(cfg_func, disable_cfg1_optimization=True)

    # Note: ControlNet class-label injection wrapper removed to keep CADE neutral.

    # Optional directional post-mix (Muse Blend), global, no ONNX
    if bool(mahiro_plus_enable):
        s_clamp = float(max(0.0, min(1.0, mahiro_plus_strength)))
        mb_state = {"ema": None}

        def _sqrt_sign(x: torch.Tensor) -> torch.Tensor:
            return x.sign() * torch.sqrt(x.abs().clamp_min(1e-12))

        def _hp_split(x: torch.Tensor, radius: int = 1, sigma: float = 1.0):
            low = _gaussian_blur_nchw(x, sigma=sigma, radius=radius)
            high = x - low
            return low, high

        def _sched_gain(args) -> float:
            # Gentle mid-steps boost: triangle peak at the middle of schedule
            try:
                sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
                idx_t = args.get("timestep", None)
                if idx_t is None:
                    return 1.0
                matched = (sigmas == idx_t[0]).nonzero()
                if len(matched) == 0:
                    return 1.0
                i = float(matched.item())
                n = float(sigmas.shape[0])
                if n <= 1:
                    return 1.0
                phase = i / (n - 1.0)
                tri = 1.0 - abs(2.0 * phase - 1.0)
                return float(0.6 + 0.4 * tri)  # 0.6 at edges -> 1.0 mid
            except Exception:
                return 1.0

        def mahiro_plus_post(args):
            try:
                scale = args.get('cond_scale', 1.0)
                cond_p = args['cond_denoised']
                uncond_p = args['uncond_denoised']
                cfg = args['denoised']

                # Orthogonalize positive to negative direction (batch-wise)
                bsz = cond_p.shape[0]
                pos_flat = cond_p.view(bsz, -1)
                neg_flat = uncond_p.view(bsz, -1)
                dot = torch.sum(pos_flat * neg_flat, dim=1, keepdim=True)
                denom = torch.sum(neg_flat * neg_flat, dim=1, keepdim=True).clamp_min(1e-8)
                alpha = (dot / denom).view(bsz, *([1] * (cond_p.dim() - 1)))
                c_orth = cond_p - uncond_p * alpha

                leap_raw = float(scale) * c_orth
                # Light high-pass emphasis for detail, protect low-frequency tone
                low, high = _hp_split(leap_raw, radius=1, sigma=1.0)
                leap = 0.35 * low + 1.00 * high

                # Directional agreement (global cosine over flattened dims)
                u_leap = float(scale) * uncond_p
                merge = 0.5 * (leap + cfg)
                nu = _sqrt_sign(u_leap).flatten(1)
                nm = _sqrt_sign(merge).flatten(1)
                sim = F.cosine_similarity(nu, nm, dim=1).mean()
                a = torch.clamp((sim + 1.0) * 0.5, 0.0, 1.0)
                # Small EMA for temporal smoothness
                if mb_state["ema"] is None:
                    mb_state["ema"] = float(a)
                else:
                    mb_state["ema"] = 0.8 * float(mb_state["ema"]) + 0.2 * float(a)
                a_eff = float(mb_state["ema"])
                w = a_eff * cfg + (1.0 - a_eff) * leap

                # Gentle energy match to CFG
                dims = tuple(range(1, w.dim()))
                ro_w = torch.std(w, dim=dims, keepdim=True).clamp_min(1e-6)
                ro_cfg = torch.std(cfg, dim=dims, keepdim=True).clamp_min(1e-6)
                w_res = w * (ro_cfg / ro_w)

                # Schedule gain over steps (mid stronger)
                s_eff = s_clamp * _sched_gain(args)
                out = (1.0 - s_eff) * cfg + s_eff * w_res
                return out
            except Exception:
                return args['denoised']

        try:
            m.set_model_sampler_post_cfg_function(mahiro_plus_post)
        except Exception:
            pass

    # Quantile clamp stabilizer (per-sample): soft range limit for denoised tensor
    # Always on, under the hood. Helps prevent rare exploding values.
    def _qclamp_post(args):
        try:
            x = args.get("denoised", None)
            if x is None:
                return args["denoised"]
            dt = x.dtype
            xf = x.to(dtype=torch.float32)
            B = xf.shape[0]
            lo_q, hi_q = 0.001, 0.999
            out = []
            for i in range(B):
                t = xf[i].reshape(-1)
                try:
                    lo = torch.quantile(t, lo_q)
                    hi = torch.quantile(t, hi_q)
                except Exception:
                    n = t.numel()
                    k_lo = max(1, int(n * lo_q))
                    k_hi = max(1, int(n * hi_q))
                    lo = torch.kthvalue(t, k_lo).values
                    hi = torch.kthvalue(t, k_hi).values
                out.append(xf[i].clamp(min=lo, max=hi))
            y = torch.stack(out, dim=0).to(dtype=dt)
            return y
        except Exception:
            return args["denoised"]

    try:
        m.set_model_sampler_post_cfg_function(_qclamp_post)
    except Exception:
        pass

    return m


def _edge_mask(image_bhwc: torch.Tensor,
               threshold: float = 0.20,
               blur: float = 1.0) -> torch.Tensor:
    """Return a simple edge mask BHWC [0,1] using Sobel magnitude on luminance.
    It is resolution-agnostic and intentionally lightweight.
    """
    try:
        img = image_bhwc
        lum = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
        g = torch.sqrt((F.conv2d(lum.unsqueeze(1), kx, padding=1) ** 2) + (F.conv2d(lum.unsqueeze(1), ky, padding=1) ** 2))
        g = g.squeeze(1)
        # Robust normalization via 98th percentile
        try:
            q = torch.quantile(g.flatten(), 0.98).clamp_min(1e-6)
        except Exception:
            q = torch.topk(g.flatten(), max(1, int(g.numel() * 0.02))).values.min().clamp_min(1e-6)
        m = (g / q).clamp(0, 1)
        if threshold > 0.0:
            m = (m > float(threshold)).to(img.dtype)
        if blur > 0.0:
            rad = int(max(1, min(5, round(float(blur)))))
            m = _gaussian_blur_nchw(m.unsqueeze(1), sigma=float(max(0.5, blur)), radius=rad).squeeze(1)
        return m.unsqueeze(-1)
    except Exception:
        return torch.zeros((image_bhwc.shape[0], image_bhwc.shape[1], image_bhwc.shape[2], 1), device=image_bhwc.device, dtype=image_bhwc.dtype)



def _cf_edges_post(acc_t: torch.Tensor,
                   edge_width: float,
                   edge_smooth: float,
                   edge_single_line: bool,
                   edge_single_strength: float) -> torch.Tensor:
    try:
        import cv2, numpy as _np
        img = (acc_t.clamp(0,1).detach().to('cpu').numpy()*255.0).astype(_np.uint8)
        # Thickness adjust
        if float(edge_width) != 0.0:
            s = float(edge_width)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            op = cv2.dilate if s > 0 else cv2.erode
            it = int(max(0, min(6, round(abs(s) * 4.0))))
            frac = abs(s) * 4.0 - it
            for _ in range(max(0, it)):
                img = op(img, k, iterations=1)
            if frac > 1e-6:
                y2 = op(img, k, iterations=1)
                img = ((1.0-frac)*img.astype(_np.float32) + frac*y2.astype(_np.float32)).astype(_np.uint8)
        # Collapse double lines to single centerline
        if bool(edge_single_line) and float(edge_single_strength) > 1e-6:
            try:
                s = float(edge_single_strength)
                close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
                if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
                    sk = cv2.ximgproc.thinning(close)
                else:
                    iters = max(1, int(round(2 + 6*s)))
                    sk = _np.zeros_like(close)
                    src = close.copy()
                    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                    for _ in range(iters):
                        er = cv2.erode(src, elem, iterations=1)
                        opn = cv2.morphologyEx(er, cv2.MORPH_OPEN, elem)
                        tmp = cv2.subtract(er, opn)
                        sk = cv2.bitwise_or(sk, tmp)
                        src = er
                        if not _np.any(src):
                            break
                img = ((1.0 - s) * img.astype(_np.float32) + s * sk.astype(_np.float32)).astype(_np.uint8)
            except Exception:
                pass
        # Smooth
        if float(edge_smooth) > 1e-6:
            sigma = max(0.1, min(2.0, float(edge_smooth) * 1.2))
            img = cv2.GaussianBlur(img, (0,0), sigmaX=sigma)
        out = torch.from_numpy((img.astype(_np.float32)/255.0)).to(device=acc_t.device, dtype=acc_t.dtype)
        return out.clamp(0,1)
    except Exception:
        # Torch fallback: light blur-only and basic thicken/thin
        y = acc_t
        if float(edge_width) > 1e-6:
            k = max(1, int(round(float(edge_width) * 2)))
            p = k
            y = F.max_pool2d(y.unsqueeze(0).unsqueeze(0), kernel_size=2*k+1, stride=1, padding=p)[0,0]
        if float(edge_width) < -1e-6:
            k = max(1, int(round(abs(float(edge_width)) * 2)))
            p = k
            maxed = F.max_pool2d((1.0 - y).unsqueeze(0).unsqueeze(0), kernel_size=2*k+1, stride=1, padding=p)[0,0]
            y = 1.0 - maxed
        if float(edge_smooth) > 1e-6:
            s = max(1, int(round(float(edge_smooth)*2)))
            y = F.avg_pool2d(y.unsqueeze(0).unsqueeze(0), kernel_size=2*s+1, stride=1, padding=s)[0,0]
        return y.clamp(0,1)


def _build_cf_edge_mask_from_step(image_bhwc: torch.Tensor, preset_step: str) -> torch.Tensor | None:
    try:
        p = load_preset("mg_controlfusion", preset_step)
        # Safe converters (preset values may be blank strings)
        def _safe_int(val, default):
            try:
                iv = int(val)
                return iv if iv > 0 else default
            except Exception:
                return default
        def _safe_float(val, default):
            try:
                return float(val)
            except Exception:
                return default

        # Read CF params with safe defaults
        enable_depth = bool(p.get('enable_depth', True))
        depth_model_path = str(p.get('depth_model_path', ''))
        depth_resolution = _safe_int(p.get('depth_resolution', 768), 768)
        hires_mask_auto = bool(p.get('hires_mask_auto', True))
        pyra_low = _safe_int(p.get('pyra_low', 109), 109)
        pyra_high = _safe_int(p.get('pyra_high', 147), 147)
        pyra_resolution = _safe_int(p.get('pyra_resolution', 1024), 1024)
        edge_thin_iter = int(p.get('edge_thin_iter', 0))
        edge_boost = _safe_float(p.get('edge_boost', 0.0), 0.0)
        smart_tune = bool(p.get('smart_tune', False))
        smart_boost = _safe_float(p.get('smart_boost', 0.2), 0.2)
        edge_width = _safe_float(p.get('edge_width', 0.0), 0.0)
        edge_smooth = _safe_float(p.get('edge_smooth', 0.0), 0.0)
        edge_single_line = bool(p.get('edge_single_line', False))
        edge_single_strength = _safe_float(p.get('edge_single_strength', 0.0), 0.0)
        edge_depth_gate = bool(p.get('edge_depth_gate', False))
        edge_depth_gamma = _safe_float(p.get('edge_depth_gamma', 1.5), 1.5)
        edge_alpha = _safe_float(p.get('edge_alpha', 1.0), 1.0)
        # Treat blend_factor as extra gain for edges (depth is not mixed here)
        blend_factor = _safe_float(p.get('blend_factor', 0.02), 0.02)
        # ControlNet multipliers: use edge_strength_mul as an additional gain for the edge mask
        edge_strength_mul = _safe_float(p.get('edge_strength_mul', 1.0), 1.0)

        # Build edges with CF PyraCanny
        ed = _cf_pyracanny(image_bhwc, pyra_low, pyra_high, pyra_resolution,
                           edge_thin_iter, edge_boost, smart_tune, smart_boost,
                           preserve_aspect=bool(hires_mask_auto))
        ed = _cf_edges_post(ed, edge_width, edge_smooth, edge_single_line, edge_single_strength)
        # Depth-gate edges if enabled
        if edge_depth_gate and enable_depth:
            try:
                depth = _cf_build_depth(image_bhwc, int(depth_resolution), str(depth_model_path), bool(hires_mask_auto))
                g = depth.clamp(0,1) ** float(edge_depth_gamma)
                ed = (ed * g).clamp(0,1)
            except Exception:
                pass
        # Apply opacity + edge strength (keep blend_factor only for ControlNet stage, not for mask amplitude)
        total_gain = max(0.0, float(edge_alpha)) * max(0.0, float(edge_strength_mul))
        ed = (ed * total_gain).clamp(0,1)
        # Return BHWC single-channel
        return ed.unsqueeze(0).unsqueeze(-1)
    except Exception:
        return None

def _mask_to_like(mask_bhw1: torch.Tensor, like_bhwc: torch.Tensor) -> torch.Tensor:
    try:
        if mask_bhw1 is None or like_bhwc is None:
            return mask_bhw1
        if mask_bhw1.ndim != 4 or like_bhwc.ndim != 4:
            return mask_bhw1
        _, Ht, Wt, _ = like_bhwc.shape
        _, Hm, Wm, C = mask_bhw1.shape
        if (Hm, Wm) == (Ht, Wt):
            return mask_bhw1
        m = mask_bhw1.movedim(-1, 1)
        m = F.interpolate(m, size=(Ht, Wt), mode='bilinear', align_corners=False)
        return m.movedim(1, -1).clamp(0, 1)
    except Exception:
        return mask_bhw1

def _align_mask_pair(a_bhw1: torch.Tensor, b_bhw1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        if a_bhw1 is None or b_bhw1 is None:
            return a_bhw1, b_bhw1
        if a_bhw1.ndim != 4 or b_bhw1.ndim != 4:
            return a_bhw1, b_bhw1
        _, Ha, Wa, Ca = a_bhw1.shape
        _, Hb, Wb, Cb = b_bhw1.shape
        if (Ha, Wa) == (Hb, Wb):
            return a_bhw1, b_bhw1
        m = b_bhw1.movedim(-1, 1)
        m = F.interpolate(m, size=(Ha, Wa), mode='bilinear', align_corners=False)
        return a_bhw1, m.movedim(1, -1).clamp(0, 1)
    except Exception:
        return a_bhw1, b_bhw1
def _mask_dilate(mask_bhw1: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return mask_bhw1
    m = mask_bhw1.movedim(-1, 1)
    m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
    return m.movedim(1, -1)


def _mask_erode(mask_bhw1: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return mask_bhw1
    m = mask_bhw1.movedim(-1, 1)
    e = 1.0 - F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=k // 2)
    return e.movedim(1, -1)


class ComfyAdaptiveDetailEnhancer25:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_step": (["Step 1", "Step 2", "Step 3", "Step 4"], {"default": "Step 1", "tooltip": "Choose the Step preset. Toggle Custom below to apply UI values; otherwise Step preset values are used."}),
                "custom": ("BOOLEAN", {"default": False, "tooltip": "Custom override: when enabled, your UI values override the selected Step for visible controls; hidden parameters still come from the Step preset."}), "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "vae": ("VAE", {}),
                "latent": ("LATENT", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "sampler_name": (_sampler_names(), {"default": _sampler_names()[0]}),
                "scheduler": (_scheduler_names(), {"default": _scheduler_names()[0]}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "steps_delta": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "cfg_delta": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "denoise_delta": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.0001}),
                "apply_sharpen": ("BOOLEAN", {"default": False}),
                "apply_upscale": ("BOOLEAN", {"default": False}),
                "apply_ids": ("BOOLEAN", {"default": False}),
                "clip_clean": ("BOOLEAN", {"default": False}),
                "ids_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_method": (MagicUpscaleModule.upscale_methods, {"default": "lanczos"}),
                "scale_by": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 8.0, "step": 0.01}),
                "scale_delta": ("FLOAT", {"default": 0.0, "min": -8.0, "max": 8.0, "step": 0.01}),
                "noise_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "threshold": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "RMS latent drift threshold (smaller = more damping)."}),
            },
            "optional": {
                "Sharpnes_strenght": ("FLOAT", {"default": 0.300, "min": 0.0, "max": 1.0, "step": 0.001}),
                "latent_compare": ("BOOLEAN", {"default": False, "tooltip": "Use latent drift to gently damp params (safer than overwriting latents)."}),
                "accumulation": (["default", "fp32+fp16", "fp32+fp32"], {"default": "default", "tooltip": "Override SageAttention PV accumulation mode for this node run."}),
                "reference_clean": ("BOOLEAN", {"default": False, "tooltip": "Use CLIP-Vision similarity to a reference image to stabilize output."}),
                "reference_image": ("IMAGE", {}),
                "clip_vision": ("CLIP_VISION", {}),
                "ref_preview": ("INT", {"default": 224, "min": 64, "max": 512, "step": 16}),
                "ref_threshold": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.2, "step": 0.001}),
                "ref_cooldown": ("INT", {"default": 1, "min": 1, "max": 8}),

                # ONNX detectors (beta) unified toggle for Hands/Face/Pose
                "onnx_detectors": ("BOOLEAN", {"default": False, "tooltip": "Use auto ONNX detectors (any .onnx in models) to refine artifact mask."}),
                "onnx_preview": ("INT", {"default": 224, "min": 64, "max": 512, "step": 16, "tooltip": "Square preview size fed to ONNX models."}),
                "onnx_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Global gain for fused ONNX mask."}),
                "onnx_anomaly_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Extra gain for 'anomaly' models (e.g., anomaly_det.onnx)."}),

                # Guidance controls
                "guidance_mode": (["default", "RescaleCFG", "RescaleFDG", "CFGZero*", "CFGZeroFD", "ZeResFDG"], {"default": "RescaleCFG", "tooltip": "Rescale (stable), RescaleFDG (spectral), CFGZero*, CFGZeroFD, or hybrid ZeResFDG."}),
                "rescale_multiplier": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend between rescaled and plain CFG (like comfy RescaleCFG)."}),
                "momentum_beta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "EMA momentum in eps-space for (cond-uncond), 0 to disable."}),
                "cfg_curve": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "S-curve shaping of cond_scale across steps (0=flat)."}),
                "perp_damp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Remove a small portion of the component parallel to previous delta (0-1)."}),

                # NAG (Normalized Attention Guidance) toggles
                "use_nag": ("BOOLEAN", {"default": False, "tooltip": "Apply NAG inside CrossAttention (positive branch) during this node."}),
                "nag_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),

                # CFGZero* extras
                "use_zero_init": ("BOOLEAN", {"default": False, "tooltip": "For CFGZero*, zero out first few steps."}),
                "zero_init_steps": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),

                # FDG controls (placed last to avoid reordering existing fields)
                "fdg_low": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Low-frequency gain (<1 to restrain masses)."}),
                "fdg_high": ("FLOAT", {"default": 1.3, "min": 0.5, "max": 2.5, "step": 0.01, "tooltip": "High-frequency gain (>1 to boost details)."}),
                "fdg_sigma": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.5, "step": 0.05, "tooltip": "Gaussian sigma for FDG low-pass split."}),
                "ze_res_zero_steps": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1, "tooltip": "Hybrid: number of initial steps to use CFGZeroFD before switching to RescaleFDG."}),

                # Adaptive spectral switch (ZeRes) and adaptive low gain
                "ze_adaptive": ("BOOLEAN", {"default": False, "tooltip": "Enable spectral switch: CFGZeroFD, RescaleFDG by HF/LF ratio (EMA)."}),
                "ze_r_switch_hi": ("FLOAT", {"default": 0.60, "min": 0.10, "max": 0.95, "step": 0.01, "tooltip": "Switch to RescaleFDG when EMA fraction of high-frequency."}),
                "ze_r_switch_lo": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 0.90, "step": 0.01, "tooltip": "Switch back to CFGZeroFD when EMA fraction (hysteresis)."}),
                "fdg_low_adaptive": ("BOOLEAN", {"default": False, "tooltip": "Adapt fdg_low by HF fraction (EMA)."}),
                "fdg_low_min": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Lower bound for adaptive fdg_low."}),
                "fdg_low_max": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Upper bound for adaptive fdg_low."}),
                "fdg_ema_beta": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "EMA smoothing for spectral ratio (higher = smoother)."}),

                # ONNX local guidance (placed last to avoid reordering)
                "onnx_local_guidance": ("BOOLEAN", {"default": False, "tooltip": "Modulate guidance spatially by ONNX mask."}),
                "onnx_mask_inside": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Multiplier for guidance inside mask (protects)."}),
                "onnx_mask_outside": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "Multiplier for guidance outside mask."}),
                "onnx_debug": ("BOOLEAN", {"default": False, "tooltip": "Print ONNX mask area per iteration."}),

                # ONNX wholebody keypoints local heatmap (placed last)
                "onnx_kpts_enable": ("BOOLEAN", {"default": False, "tooltip": "Parse YOLO wholebody keypoints and add local heatmap."}),
                "onnx_kpts_sigma": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.1, "tooltip": "Keypoint Gaussian sigma multiplier."}),
                "onnx_kpts_gain": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Keypoint heat amplitude multiplier."}),
                "onnx_kpts_conf": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Keypoint confidence threshold."}),

                # Muse Blend global directional post-mix
                "muse_blend": ("BOOLEAN", {"default": False, "tooltip": "Enable Muse Blend: gentle directional positive blend (global)."}),
                "muse_blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Overall influence of Muse Blend over baseline CFG (0..1)."}),
                # Exposure Bias Correction (epsilon scaling)
                "eps_scale_enable": ("BOOLEAN", {"default": False, "tooltip": "Exposure Bias Correction: scale predicted noise early in schedule."}),
                "eps_scale": ("FLOAT", {"default": 0.005, "min": -1.0, "max": 1.0, "step": 0.0005, "tooltip": "Signed scaling near early steps (recommended ~0.0045; use with care)."}),
                "clipseg_enable": ("BOOLEAN", {"default": False, "tooltip": "Use CLIPSeg to build a text-driven mask (e.g., 'eyes | hands | face')."}),
                "clipseg_text": ("STRING", {"default": "", "multiline": False}),
                "clipseg_preview": ("INT", {"default": 224, "min": 64, "max": 512, "step": 16}),
                "clipseg_threshold": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}),
                "clipseg_blur": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 15.0, "step": 0.1}),
                "clipseg_dilate": ("INT", {"default": 4, "min": 0, "max": 10, "step": 1}),
                "clipseg_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "clipseg_blend": (["fuse", "replace", "intersect"], {"default": "fuse", "tooltip": "How to combine CLIPSeg with ONNX mask."}),
                "clipseg_ref_gate": ("BOOLEAN", {"default": False, "tooltip": "If reference provided, boost mask when far from reference (CLIP-Vision)."}),
                "clipseg_ref_threshold": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.2, "step": 0.001}),
                # Preview/output image cap (helps RAM during save/preview)
                "preview_downscale": ("BOOLEAN", {"default": True, "tooltip": "Cap final IMAGE to max 1920 on the longer side to reduce RAM spike during save/preview. Disable for full-res output."}),
                # Under-the-hood saving (disabled by default to avoid duplicate saves)
                "auto_save": ("BOOLEAN", {"default": False, "tooltip": "Save final IMAGE directly from CADE (uses low PNG compress to reduce RAM)."}),
                "save_prefix": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "save_compress": ("INT", {"default": 1, "min": 0, "max": 9, "step": 1}),

                # Polish mode (final hi-res refinement)
                "polish_enable": ("BOOLEAN", {"default": False, "tooltip": "Polish: keep low-frequency shape from reference while allowing high-frequency details to refine."}),
                "polish_keep_low": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much low-frequency (global form, lighting) to take from reference image (0=use current, 1=use reference)."}),
                "polish_edge_lock": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Edge lock strength: protects edges from sideways drift (0=off, 1=strong)."}),
                "polish_sigma": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.1, "tooltip": "Radius for low/high split: larger keeps bigger shapes as 'low' (global form)."}),
                "polish_start_after": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1, "tooltip": "Enable polish after N iterations (0=immediately)."}),
                "polish_keep_low_ramp": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Starting share of low-frequency mix; ramps to polish_keep_low over remaining iterations."}),

            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "INT", "FLOAT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("LATENT", "IMAGE", "steps", "cfg", "denoise", "mask_preview")
    FUNCTION = "apply_cade2"
    CATEGORY = "MagicNodes"

    def apply_cade2(self, model, vae, positive, negative, latent, seed, steps, cfg, denoise,
                     sampler_name, scheduler, noise_offset, iterations=1, steps_delta=0.0,
                     cfg_delta=0.0, denoise_delta=0.0, apply_sharpen=False,
                     apply_upscale=False, apply_ids=False, clip_clean=False,
                     ids_strength=0.5, upscale_method="lanczos", scale_by=1.2, scale_delta=0.0,
                    Sharpnes_strenght=0.300, threshold=0.03, latent_compare=False, accumulation="default",
                     reference_clean=False, reference_image=None, clip_vision=None, ref_preview=224, ref_threshold=0.03, ref_cooldown=1,
                     onnx_detectors=False, onnx_preview=224, onnx_sensitivity=0.5, onnx_anomaly_gain=1.0,
                     guidance_mode="RescaleCFG", rescale_multiplier=0.7, momentum_beta=0.0, cfg_curve=0.0, perp_damp=0.0,
                     use_nag=False, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25,
                     use_zero_init=False, zero_init_steps=0,
                     fdg_low=0.6, fdg_high=1.3, fdg_sigma=1.0, ze_res_zero_steps=2,
                     ze_adaptive=False, ze_r_switch_hi=0.60, ze_r_switch_lo=0.45,
                     fdg_low_adaptive=False, fdg_low_min=0.45, fdg_low_max=0.70, fdg_ema_beta=0.80,
                     onnx_local_guidance=False, onnx_mask_inside=1.0, onnx_mask_outside=1.0, onnx_debug=False,
                     onnx_kpts_enable=False, onnx_kpts_sigma=2.5, onnx_kpts_gain=1.5, onnx_kpts_conf=0.20,
                     muse_blend=False, muse_blend_strength=0.5,
                     eps_scale_enable=False, eps_scale=0.005,
                     clipseg_enable=False, clipseg_text="", clipseg_preview=224,
                     clipseg_threshold=0.40, clipseg_blur=7.0, clipseg_dilate=4,
                     clipseg_gain=1.0, clipseg_blend="fuse", clipseg_ref_gate=False, clipseg_ref_threshold=0.03,
                     polish_enable=False, polish_keep_low=0.4, polish_edge_lock=0.2, polish_sigma=1.0,
                    polish_start_after=1, polish_keep_low_ramp=0.2,
                     preview_downscale=False,
                     auto_save=False, save_prefix="ComfyUI", save_compress=1,
                     preset_step="Step 1", custom_override=False):
        # Cooperative cancel before any heavy work
        model_management.throw_exception_if_processing_interrupted()

        # Load base preset for the selected Step. When custom_override is True,
        # visible UI controls (top-level) are kept from UI; hidden ones still come from preset.
        try:
            p = load_preset("mg_cade25", preset_step) if isinstance(preset_step, str) else {}
        except Exception:
            p = {}
        def pv(name, cur, top=False):
            return cur if (top and bool(custom_override)) else p.get(name, cur)
        seed = int(pv("seed", seed, top=True))
        steps = int(pv("steps", steps, top=True))
        cfg = float(pv("cfg", cfg, top=True))
        denoise = float(pv("denoise", denoise, top=True))
        sampler_name = str(pv("sampler_name", sampler_name, top=True))
        scheduler = str(pv("scheduler", scheduler, top=True))
        iterations = int(pv("iterations", iterations))
        # Smart-seed per-step toggle (defaults to True if not present in preset)
        smart_seed_enable = bool(pv("smart_seed_enable", True))
        steps_delta = float(pv("steps_delta", steps_delta))
        cfg_delta = float(pv("cfg_delta", cfg_delta))
        denoise_delta = float(pv("denoise_delta", denoise_delta))
        apply_sharpen = bool(pv("apply_sharpen", apply_sharpen))
        apply_upscale = bool(pv("apply_upscale", apply_upscale))
        apply_ids = bool(pv("apply_ids", apply_ids))
        clip_clean = bool(pv("clip_clean", clip_clean))
        ids_strength = float(pv("ids_strength", ids_strength))
        upscale_method = str(pv("upscale_method", upscale_method))
        scale_by = float(pv("scale_by", scale_by))
        scale_delta = float(pv("scale_delta", scale_delta))
        noise_offset = float(pv("noise_offset", noise_offset))
        threshold = float(pv("threshold", threshold))
        Sharpnes_strenght = float(pv("Sharpnes_strenght", Sharpnes_strenght))
        latent_compare = bool(pv("latent_compare", latent_compare))
        accumulation = str(pv("accumulation", accumulation))
        reference_clean = bool(pv("reference_clean", reference_clean))
        ref_preview = int(pv("ref_preview", ref_preview))
        ref_threshold = float(pv("ref_threshold", ref_threshold))
        ref_cooldown = int(pv("ref_cooldown", ref_cooldown))
        onnx_detectors = bool(pv("onnx_detectors", onnx_detectors))
        onnx_preview = int(pv("onnx_preview", onnx_preview))
        onnx_sensitivity = float(pv("onnx_sensitivity", onnx_sensitivity))
        onnx_anomaly_gain = float(pv("onnx_anomaly_gain", onnx_anomaly_gain))
        guidance_mode = str(pv("guidance_mode", guidance_mode))
        rescale_multiplier = float(pv("rescale_multiplier", rescale_multiplier))
        momentum_beta = float(pv("momentum_beta", momentum_beta))
        cfg_curve = float(pv("cfg_curve", cfg_curve))
        perp_damp = float(pv("perp_damp", perp_damp))
        use_nag = bool(pv("use_nag", use_nag))
        nag_scale = float(pv("nag_scale", nag_scale))
        nag_tau = float(pv("nag_tau", nag_tau))
        nag_alpha = float(pv("nag_alpha", nag_alpha))
        use_zero_init = bool(pv("use_zero_init", use_zero_init))
        zero_init_steps = int(pv("zero_init_steps", zero_init_steps))
        fdg_low = float(pv("fdg_low", fdg_low))
        fdg_high = float(pv("fdg_high", fdg_high))
        fdg_sigma = float(pv("fdg_sigma", fdg_sigma))
        ze_res_zero_steps = int(pv("ze_res_zero_steps", ze_res_zero_steps))
        ze_adaptive = bool(pv("ze_adaptive", ze_adaptive))
        ze_r_switch_hi = float(pv("ze_r_switch_hi", ze_r_switch_hi))
        ze_r_switch_lo = float(pv("ze_r_switch_lo", ze_r_switch_lo))
        fdg_low_adaptive = bool(pv("fdg_low_adaptive", fdg_low_adaptive))
        fdg_low_min = float(pv("fdg_low_min", fdg_low_min))
        fdg_low_max = float(pv("fdg_low_max", fdg_low_max))
        fdg_ema_beta = float(pv("fdg_ema_beta", fdg_ema_beta))
        # AQClip-Lite (hidden in Easy UI, controllable via presets)
        aqclip_enable = bool(pv("aqclip_enable", False))
        aq_tile = int(pv("aq_tile", 32))
        aq_stride = int(pv("aq_stride", 16))
        aq_alpha = float(pv("aq_alpha", 2.0))
        aq_ema_beta = float(pv("aq_ema_beta", 0.85))
        midfreq_enable = bool(pv("midfreq_enable", False))
        midfreq_gain = float(pv("midfreq_gain", 0.0))
        midfreq_sigma_lo = float(pv("midfreq_sigma_lo", 0.8))
        midfreq_sigma_hi = float(pv("midfreq_sigma_hi", 2.0))
        onnx_local_guidance = bool(pv("onnx_local_guidance", onnx_local_guidance))
        onnx_mask_inside = float(pv("onnx_mask_inside", onnx_mask_inside))
        onnx_mask_outside = float(pv("onnx_mask_outside", onnx_mask_outside))
        onnx_debug = bool(pv("onnx_debug", onnx_debug))
        onnx_kpts_enable = bool(pv("onnx_kpts_enable", onnx_kpts_enable))
        onnx_kpts_sigma = float(pv("onnx_kpts_sigma", onnx_kpts_sigma))
        onnx_kpts_gain = float(pv("onnx_kpts_gain", onnx_kpts_gain))
        onnx_kpts_conf = float(pv("onnx_kpts_conf", onnx_kpts_conf))
        muse_blend = bool(pv("muse_blend", muse_blend))
        muse_blend_strength = float(pv("muse_blend_strength", muse_blend_strength))
        eps_scale_enable = bool(pv("eps_scale_enable", eps_scale_enable))
        eps_scale = float(pv("eps_scale", eps_scale))
        clipseg_enable = bool(pv("clipseg_enable", clipseg_enable))
        clipseg_text = str(pv("clipseg_text", clipseg_text, top=True))
        clipseg_preview = int(pv("clipseg_preview", clipseg_preview))
        clipseg_threshold = float(pv("clipseg_threshold", clipseg_threshold))
        clipseg_blur = float(pv("clipseg_blur", clipseg_blur))
        clipseg_dilate = int(pv("clipseg_dilate", clipseg_dilate))
        clipseg_gain = float(pv("clipseg_gain", clipseg_gain))
        clipseg_blend = str(pv("clipseg_blend", clipseg_blend))
        clipseg_ref_gate = bool(pv("clipseg_ref_gate", clipseg_ref_gate))
        clipseg_ref_threshold = float(pv("clipseg_ref_threshold", clipseg_ref_threshold))
        # CFG scheduling (internal-only; configured via presets)
        cfg_sched = str(pv("cfg_sched", "off"))
        cfg_sched_min = float(pv("cfg_sched_min", max(0.0, cfg * 0.5)))
        cfg_sched_max = float(pv("cfg_sched_max", cfg))
        cfg_sched_gamma = float(pv("cfg_sched_gamma", 1.5))
        cfg_sched_u_pow = float(pv("cfg_sched_u_pow", 1.0))
        # Latent buffer (internal-only; configured via presets)
        latent_buffer = bool(pv("latent_buffer", True))
        lb_inject = float(pv("lb_inject", 0.25))
        lb_ema = float(pv("lb_ema", 0.75))
        lb_every = int(pv("lb_every", 1))
        lb_anchor_every = int(pv("lb_anchor_every", 6))
        lb_masked = bool(pv("lb_masked", True))
        lb_rebase_thresh = float(pv("lb_rebase_thresh", 0.10))
        lb_rebase_rate = float(pv("lb_rebase_rate", 0.25))
        polish_enable = bool(pv("polish_enable", polish_enable))
        polish_keep_low = float(pv("polish_keep_low", polish_keep_low))
        polish_edge_lock = float(pv("polish_edge_lock", polish_edge_lock))
        polish_sigma = float(pv("polish_sigma", polish_sigma))
        polish_start_after = int(pv("polish_start_after", polish_start_after))
        polish_keep_low_ramp = float(pv("polish_keep_low_ramp", polish_keep_low_ramp))
        # CADE Seg: per-step toggle to include CF edges into Seg mask
        seg_use_cf_edges = bool(pv("seg_use_cf_edges", True))
        # Hard reset of any sticky globals from prior runs
        try:
            global CURRENT_ONNX_MASK_BCHW, _ONNX_KPTS_ENABLE, _ONNX_KPTS_SIGMA, _ONNX_KPTS_GAIN, _ONNX_KPTS_CONF
            CURRENT_ONNX_MASK_BCHW = None
            # Reset KPTS toggles to sane defaults; they will be set again if enabled below
            _ONNX_KPTS_ENABLE = False
            _ONNX_KPTS_SIGMA = 2.5
            _ONNX_KPTS_GAIN = 1.5
            _ONNX_KPTS_CONF = 0.20
        except Exception:
            pass

        image = safe_decode(vae, latent)
        # allow user cancel right after initial decode
        model_management.throw_exception_if_processing_interrupted()

        tuned_steps, tuned_cfg, tuned_denoise = AdaptiveSamplerHelper().tune(
            image, steps, cfg, denoise)

        current_steps = tuned_steps
        current_cfg = tuned_cfg
        current_denoise = tuned_denoise
        # Work on a detached copy to avoid mutating input latent across runs
        try:
            current_latent = {"samples": latent["samples"].clone()}
        except Exception:
            current_latent = {"samples": latent["samples"]}
        current_scale = scale_by

        # Derive a user-friendly step tag for logs
        try:
            _ps = str(preset_step)
            _num = ''.join(ch for ch in _ps if ch.isdigit())
            step_tag = f"Step:{_num}" if _num else _ps
        except Exception:
            step_tag = str(preset_step)

        # Smart seed selection (Sobol + light probing) when effective seed==0 and not in custom override mode
        try:
            if int(seed) == 0 and not bool(custom_override) and bool(smart_seed_enable):
                seed = _smart_seed_select(
                    model, vae, positive, negative, current_latent,
                    str(sampler_name), str(scheduler), float(current_cfg), float(current_denoise),
                    base_seed=0, step_tag=step_tag,
                    # Reduce candidate count and probe cost for Easy UI
                    k=3, probe_steps=3,
                    clip_vision=clip_vision, reference_image=reference_image, clipseg_text=str(clipseg_text))
        except Exception as e:
            # propagate user cancel; swallow only non-interrupt errors
            if isinstance(e, model_management.InterruptProcessingException):
                raise
            pass

        # Visual separation and start marker after seed is finalized
        try:
            print("")
        except Exception:
            pass
        try:
            print(f"\x1b[32m==== {step_tag}, Starting main job ====\x1b[0m")
        except Exception:
            pass

        ref_embed = None
        if reference_clean and (clip_vision is not None) and (reference_image is not None):
            try:
                ref_embed = _encode_clip_image(reference_image, clip_vision, ref_preview)
            except Exception:
                ref_embed = None

        # Pre-disable any lingering NAG patch from previous runs and set PV accumulation for this node
        try:
            sa_patch.enable_crossattention_nag_patch(False)
        except Exception:
            pass
        prev_accum = getattr(sa_patch, "CURRENT_PV_ACCUM", None)
        sa_patch.CURRENT_PV_ACCUM = None if accumulation == "default" else accumulation
        # Enable NAG patch if requested
        try:
            sa_patch.enable_crossattention_nag_patch(bool(use_nag), float(nag_scale), float(nag_tau), float(nag_alpha))
        except Exception:
            pass

        # Enable attention-entropy probe for AQClip Attn-mode (read from preset)
        try:
            aq_attn = bool(p.get("aq_attn", False)) if isinstance(p, dict) else False
            if hasattr(sa_patch, "enable_attention_entropy_capture"):
                sa_patch.enable_attention_entropy_capture(aq_attn, max_tokens=1024, max_heads=4)
        except Exception:
            pass

        # Enable KV pruning for self-attention (read from preset)
        try:
            kv_enable = bool(p.get("kv_prune_enable", False)) if isinstance(p, dict) else False
            kv_keep = float(p.get("kv_keep", 0.85)) if isinstance(p, dict) else 0.85
            kv_min_tokens = int(p.get("kv_min_tokens", 128)) if isinstance(p, dict) else 128
            if hasattr(sa_patch, "set_kv_prune"):
                sa_patch.set_kv_prune(kv_enable, kv_keep, kv_min_tokens)
        except Exception:
            pass

        onnx_mask_last = None
        try:
            with torch.inference_mode():
                __cade_noop = 0  # ensure non-empty with-block
                # Latent buffer runtime state
                lb_state = {"z_ema": None, "anchor": None, "drift_last": None, "ref_dist_last": None}

                # Preflight: reset sticky state and build external masks once (CPU-pinned)
                try:
                    CURRENT_ONNX_MASK_BCHW = None
                except Exception:
                    pass
                pre_mask = None
                pre_area = 0.0
                # ONNX detectors disabled in Easy: prefer CLIPSeg + edge fusion
                onnx_detectors = False
                # Build CLIPSeg mask once
                if bool(clipseg_enable) and isinstance(clipseg_text, str) and clipseg_text.strip() != "":
                    try:
                        cmask = _clipseg_build_mask(image, clipseg_text, int(clipseg_preview), float(clipseg_threshold), float(clipseg_blur), int(clipseg_dilate), float(clipseg_gain), None, None, float(clipseg_ref_threshold))
                        if cmask is not None:
                            if pre_mask is not None:
                                pre_mask = _mask_to_like(pre_mask, image)
                            cmask = _mask_to_like(cmask, image)
                            if pre_mask is None:
                                pre_mask = cmask
                            else:
                                pre_mask, cmask = _align_mask_pair(pre_mask, cmask)
                                if clipseg_blend == "replace":
                                    pre_mask = cmask
                                elif clipseg_blend == "intersect":
                                    pre_mask = (pre_mask * cmask).clamp(0, 1)
                                else:
                                    pre_mask = (1.0 - (1.0 - pre_mask) * (1.0 - cmask)).clamp(0, 1)
                    except Exception:
                        pass
                # Edge mask from ControlFusion Step (with depth gating) when enabled; fallback to Sobel
                if bool(seg_use_cf_edges):
                    try:
                        emask = _build_cf_edge_mask_from_step(image, str(preset_step))
                    except Exception:
                        emask = None
                    if emask is None:
                        try:
                            emask = _edge_mask(image, threshold=0.20, blur=1.0)
                        except Exception:
                            emask = None
                    if emask is not None:
                        if pre_mask is not None:
                            pre_mask, emask = _align_mask_pair(pre_mask, emask)
                        pre_mask = emask if pre_mask is None else (1.0 - (1.0 - pre_mask) * (1.0 - emask)).clamp(0, 1)
                if pre_mask is not None:
                    onnx_mask_last = pre_mask
                    om = pre_mask.movedim(-1, 1)
                    pre_area = float(om.mean().item())
                    if bool(onnx_local_guidance):
                        try:
                            if 0.02 <= pre_area <= 0.35:
                                CURRENT_ONNX_MASK_BCHW = om.clamp(0, 1).to(model_management.get_torch_device())
                            else:
                                CURRENT_ONNX_MASK_BCHW = None
                        except Exception:
                            CURRENT_ONNX_MASK_BCHW = None
                        try:
                            del onnx_mask
                        except Exception:
                            pass
                        try:
                            del om
                        except Exception:
                            pass
                        try:
                            del img_preview
                        except Exception:
                            pass
                    # One-time damping from area (disabled by default)
                    if False:
                        try:
                            if pre_area > 0.005:
                                damp = 1.0 - min(0.04, 0.008 + pre_area * 0.02)
                                current_denoise = max(0.10, current_denoise * damp)
                                current_cfg = max(1.0, current_cfg * (1.0 - 0.003))
                        except Exception:
                            pass
                # Preflight symmetry disabled (kept for experiments only)
                if False:
                    try:
                        img0 = image
                        sym_mask = _clipseg_build_mask(img0, "face | head | torso | shoulders", preview=int(clipseg_preview), threshold=0.45, blur=5.0, dilate=2, gain=1.0)
                        if sym_mask is not None:
                            img_sym = _soft_symmetry_blend(img0, sym_mask, alpha=0.012, lp_sigma=1.75)
                            current_latent = {"samples": safe_encode(vae, img_sym)}
                            image = img_sym
                    except Exception:
                        pass
                # Compact status
                try:
                    provs = []
                    if _ONNX_RT is not None:
                        provs = list(_ONNX_RT.get_available_providers())
                    clipseg_status = "on" if bool(clipseg_enable) and isinstance(clipseg_text, str) and clipseg_text.strip() != "" else "off"
                    kpts = f"kpts={'on' if bool(onnx_kpts_enable) else 'off'} sigma={float(onnx_kpts_sigma):.2f} gain={float(onnx_kpts_gain):.2f} conf={float(onnx_kpts_conf):.2f}"
                    # print preflight info only in debug sessions (muted by default)
                    if False:
                        print(f"[CADE2.5][preflight] onnx_sessions={len(_ONNX_SESS)} providers={provs if provs else ['CPU']} clipseg={clipseg_status} device={'cpu' if _CLIPSEG_FORCE_CPU else _CLIPSEG_DEV} mask_area={pre_area:.4f} {kpts}")
                except Exception:
                    pass
                # Freeze per-iteration external mask rebuild
                onnx_detectors = False
                clipseg_enable = False
                # Depth gate cache for micro-detail injection (reuse per resolution)
                depth_gate_cache = {"size": None, "mask": None}
                # Prepare guided sampler once per node run to avoid cloning model each iteration
                sampler_model = _wrap_model_with_guidance(
                      model, guidance_mode, rescale_multiplier, momentum_beta, cfg_curve, perp_damp,
                      use_zero_init=bool(use_zero_init), zero_init_steps=int(zero_init_steps),
                      fdg_low=float(fdg_low), fdg_high=float(fdg_high), fdg_sigma=float(fdg_sigma),
                      midfreq_enable=bool(midfreq_enable), midfreq_gain=float(midfreq_gain), midfreq_sigma_lo=float(midfreq_sigma_lo), midfreq_sigma_hi=float(midfreq_sigma_hi),
                      ze_zero_steps=int(ze_res_zero_steps),
                      ze_adaptive=bool(ze_adaptive), ze_r_switch_hi=float(ze_r_switch_hi), ze_r_switch_lo=float(ze_r_switch_lo),
                      fdg_low_adaptive=bool(fdg_low_adaptive), fdg_low_min=float(fdg_low_min), fdg_low_max=float(fdg_low_max), fdg_ema_beta=float(fdg_ema_beta),
                      use_local_mask=bool(onnx_local_guidance), mask_inside=float(onnx_mask_inside), mask_outside=float(onnx_mask_outside),
                      mahiro_plus_enable=bool(muse_blend), mahiro_plus_strength=float(muse_blend_strength),
                      eps_scale_enable=bool(eps_scale_enable), eps_scale=float(eps_scale),
                      cfg_sched_type=str(cfg_sched), cfg_sched_min=float(cfg_sched_min), cfg_sched_max=float(cfg_sched_max),
                      cfg_sched_gamma=float(cfg_sched_gamma), cfg_sched_u_pow=float(cfg_sched_u_pow)
                  )
                # check once more right before the loop starts
                model_management.throw_exception_if_processing_interrupted()
                for i in range(iterations):
                    # cooperative cancel at the start of each iteration
                    model_management.throw_exception_if_processing_interrupted()
                    if i % 2 == 0:
                        clear_gpu_and_ram_cache()

                    # Reset guidance internal state so each iteration starts clean
                    try:
                        if hasattr(sampler_model, "mg_guidance_reset"):
                            sampler_model.mg_guidance_reset()
                    except Exception:
                        pass

                    prev_samples = current_latent["samples"].clone().detach()

                    iter_seed = seed + i * 7777
                    if noise_offset > 0.0:
                        # Deterministic noise offset tied to iter_seed
                        fade = 1.0 - (i / max(1, iterations))
                        try:
                            gen = torch.Generator(device='cpu')
                        except Exception:
                            gen = torch.Generator()
                        gen.manual_seed(int(iter_seed) & 0xFFFFFFFF)
                        eps = torch.randn(
                            size=current_latent["samples"].shape,
                            dtype=current_latent["samples"].dtype,
                            device='cpu',
                            generator=gen,
                        ).to(current_latent["samples"].device)
                        current_latent["samples"] = current_latent["samples"] + (noise_offset * fade) * eps
                        try:
                            del eps
                        except Exception:
                            pass

                    # Pre-sampling ONNX detectors: handled once below (kept compact)

                    # Pre-sampling ONNX detectors (build mask and optionally adjust params for this iteration)
                    if onnx_detectors and (i % max(1, 1) == 0):
                        try:
                            import os
                            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
                            img_preview = safe_decode(vae, current_latent)
                            # Set toggles for this iteration
                            globals()["_ONNX_DEBUG"] = bool(onnx_debug)
                            globals()["_ONNX_COUNT_DEBUG"] = True  # force counts ON for debugging session
                            globals()["_ONNX_KPTS_ENABLE"] = bool(onnx_kpts_enable)
                            globals()["_ONNX_KPTS_SIGMA"] = float(onnx_kpts_sigma)
                            globals()["_ONNX_KPTS_GAIN"] = float(onnx_kpts_gain)
                            globals()["_ONNX_KPTS_CONF"] = float(onnx_kpts_conf)
                            onnx_mask = _onnx_build_mask(img_preview, int(onnx_preview), float(onnx_sensitivity), models_dir, float(onnx_anomaly_gain))
                            if onnx_mask is not None:
                                onnx_mask_last = onnx_mask
                                om = onnx_mask.movedim(-1, 1)
                                area = float(om.mean().item())
                                if bool(onnx_debug):
                                    print(f"[CADE2.5][ONNX] iter={i} mask_area={area:.4f}")
                                if area > 0.005:
                                    damp = 1.0 - min(0.25, 0.06 + onnx_sensitivity * 0.05 + area * 0.25)
                                    current_denoise = max(0.10, current_denoise * damp)
                                    current_cfg = max(1.0, current_cfg * (1.0 - 0.015 * onnx_sensitivity))
                                # Prepare spatial mask for cfg_func if requested
                                if bool(onnx_local_guidance):
                                    # store BCHW mask in global for this iteration (only for reasonable areas)
                                    if 0.02 <= area <= 0.35:
                                        CURRENT_ONNX_MASK_BCHW = om.clamp(0, 1).to(model_management.get_torch_device())
                                    else:
                                        CURRENT_ONNX_MASK_BCHW = None
                            else:
                                CURRENT_ONNX_MASK_BCHW = None
                        except Exception:
                            CURRENT_ONNX_MASK_BCHW = None

                    # CF edge mask (from current image) and fusion (only when enabled)
                    if bool(seg_use_cf_edges):
                        try:
                            img_prev2 = safe_decode(vae, current_latent)
                            em2 = _build_cf_edge_mask_from_step(img_prev2, str(preset_step))
                            if em2 is not None:
                                if onnx_mask_last is None:
                                    onnx_mask_last = em2
                                else:
                                    onnx_mask_last, em2 = _align_mask_pair(onnx_mask_last, em2)
                                    onnx_mask_last = (1.0 - (1.0 - onnx_mask_last) * (1.0 - em2)).clamp(0, 1)
                                om = onnx_mask_last.movedim(-1, 1)
                                area = float(om.mean().item())
                                if bool(onnx_local_guidance):
                                    if 0.02 <= area <= 0.35:
                                        CURRENT_ONNX_MASK_BCHW = om.clamp(0, 1).to(model_management.get_torch_device())
                                    else:
                                        CURRENT_ONNX_MASK_BCHW = None
                        except Exception:
                            pass

                    # CLIPSeg mask (optional) and fusion with ONNX
                    try:
                        if bool(clipseg_enable) and isinstance(clipseg_text, str) and clipseg_text.strip() != "":
                            cmask = _clipseg_build_mask(img_prev2, clipseg_text, int(clipseg_preview), float(clipseg_threshold), float(clipseg_blur), int(clipseg_dilate), float(clipseg_gain), ref_embed if bool(clipseg_ref_gate) else None, clip_vision if bool(clipseg_ref_gate) else None, float(clipseg_ref_threshold))
                            if cmask is not None:
                                if onnx_mask_last is None:
                                    fused = cmask
                                else:
                                    if clipseg_blend == "replace":
                                        fused = cmask
                                    elif clipseg_blend == "intersect":
                                        onnx_mask_last, cmask = _align_mask_pair(onnx_mask_last, cmask)
                                        fused = (onnx_mask_last * cmask).clamp(0, 1)
                                    else:
                                        onnx_mask_last, cmask = _align_mask_pair(onnx_mask_last, cmask)
                                        fused = (1.0 - (1.0 - onnx_mask_last) * (1.0 - cmask)).clamp(0, 1)
                                onnx_mask_last = fused
                                om = fused.movedim(-1, 1)
                                area = float(om.mean().item())
                                if bool(onnx_debug):
                                    print(f"[CADE2.5][MASK] iter={i} mask_area={area:.4f}")
                                if area > 0.005:
                                    damp = 1.0 - min(0.10, 0.02 + float(onnx_sensitivity) * 0.02 + area * 0.08)
                                    current_denoise = max(0.10, current_denoise * damp)
                                    current_cfg = max(1.0, current_cfg * (1.0 - 0.008 * float(onnx_sensitivity)))
                                if bool(onnx_local_guidance):
                                    if 0.02 <= area <= 0.35:
                                        CURRENT_ONNX_MASK_BCHW = om.clamp(0, 1).to(model_management.get_torch_device())
                                    else:
                                        CURRENT_ONNX_MASK_BCHW = None
                    except Exception:
                        pass
                    try:
                        del img_prev2
                    except Exception:
                        pass
                    try:
                        del em2
                    except Exception:
                        pass
                    try:
                        del cmask
                        del fused
                        del om
                    except Exception:
                        pass

                    # Latent buffer: pre-sampling injection
                    if bool(latent_buffer) and (iterations > 1) and (i > 0) and (i % max(1, lb_every) == 0):
                        try:
                            z = current_latent["samples"]
                            if lb_state["z_ema"] is None or tuple(lb_state["z_ema"].shape) != tuple(z.shape):
                                lb_state["z_ema"] = z.clone().detach()
                            inj = float(max(0.0, min(0.95, lb_inject)))
                            # optional boost when far from reference (uses last known distance)
                            try:
                                if (ref_embed is not None) and (lb_state.get("ref_dist_last") is not None) and (lb_state["ref_dist_last"] > float(ref_threshold)):
                                    inj = min(0.95, inj * (1.0 + min(0.5, (lb_state["ref_dist_last"] - float(ref_threshold)) * 0.75)))
                            except Exception:
                                pass
                            z_ema = lb_state["z_ema"]
                            if bool(lb_masked) and (onnx_mask_last is not None):
                                m = onnx_mask_last.movedim(-1, 1)
                                m = F.interpolate(m, size=(z.shape[-2], z.shape[-1]), mode='bilinear', align_corners=False).clamp(0, 1)
                                m = m.expand(-1, z.shape[1], -1, -1)
                                z = z * (1.0 - inj * m) + z_ema * (inj * m)
                            else:
                                z = z * (1.0 - inj) + z_ema * inj
                            current_latent["samples"] = z
                        except Exception:
                            pass

                    # Sampler model prepared once above; reused across iterations (no-op here)
                    sampler_model = sampler_model

                    # Local best-of-2 in ROI (hands/face), only on early iteration to limit overhead
                    try:
                        do_local_refine = False  # disable local best-of-2 by default
                        if do_local_refine:
                            img_roi = safe_decode(vae, current_latent)
                            roi = _clipseg_build_mask(img_roi, "hand | hands | face", preview=max(192, int(clipseg_preview//2)), threshold=0.40, blur=5.0, dilate=2, gain=1.0)
                            if roi is None and onnx_mask_last is not None:
                                roi = torch.clamp(onnx_mask_last, 0.0, 1.0)
                            if roi is not None:
                                # Area gating
                                try:
                                    ra = float(roi.mean().item())
                                except Exception:
                                    ra = 0.0
                                if not (0.02 <= ra <= 0.15):
                                    raise Exception("ROI area out of range; skip local_refine")
                                # Light erosion to avoid halo influence
                                try:
                                    m = roi[..., 0].unsqueeze(1)
                                    # disable erosion effect (kernel=1)
                                    ero = 1.0 - F.max_pool2d(1.0 - m, kernel_size=1, stride=1, padding=0)
                                    roi = ero.clamp(0, 1).movedim(1, -1)
                                except Exception:
                                    pass
                                # micro sampling params
                                micro_steps = int(max(2, min(4, round(max(1, current_steps) * 0.05))))
                                micro_denoise = float(min(0.22, max(0.10, current_denoise * 0.30)))
                                s1 = int((iter_seed ^ 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF)
                                s2 = int((iter_seed ^ 0x85EBCA77) & 0xFFFFFFFFFFFFFFFF)
                                # Candidate A
                                lat_in_a = {"samples": current_latent["samples"].clone()}
                                lat_a, = nodes.common_ksampler(
                                    sampler_model, s1, micro_steps, current_cfg, sampler_name, scheduler,
                                    positive, negative, lat_in_a, denoise=micro_denoise)
                                img_a = safe_decode(vae, lat_a)
                                # Candidate B
                                lat_in_b = {"samples": current_latent["samples"].clone()}
                                lat_b, = nodes.common_ksampler(
                                    sampler_model, s2, micro_steps, current_cfg, sampler_name, scheduler,
                                    positive, negative, lat_in_b, denoise=micro_denoise)
                                img_b = safe_decode(vae, lat_b)

                                # Score inside ROI
                                def _roi_stats(img, roi_mask):
                                    try:
                                        m = roi_mask[..., 0].clamp(0, 1)
                                        R, G, Bc = img[..., 0], img[..., 1], img[..., 2]
                                        lum = (0.2126 * R + 0.7152 * G + 0.0722 * Bc)
                                        # edges
                                        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=img.device, dtype=img.dtype).view(1,1,3,3)
                                        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=img.device, dtype=img.dtype).view(1,1,3,3)
                                        gx = F.conv2d(lum.unsqueeze(1), kx, padding=1)
                                        gy = F.conv2d(lum.unsqueeze(1), ky, padding=1)
                                        g = torch.sqrt(gx*gx + gy*gy).squeeze(1)
                                        wmean = (g*m).mean() / (m.mean()+1e-6)
                                        # speckles
                                        V = torch.maximum(R, torch.maximum(G, Bc))
                                        mi = torch.minimum(R, torch.minimum(G, Bc))
                                        S = 1.0 - (mi / (V + 1e-6))
                                        cand = (V > 0.98) & (S < 0.12)
                                        speck = (cand.float()*m).mean() / (m.mean()+1e-6)
                                        lmean = (lum*m).mean() / (m.mean()+1e-6)
                                        return float(wmean.item()), float(speck.item()), float(lmean.item())
                                    except Exception:
                                        return 0.0, 0.5, 0.5

                                ed_a, sp_a, lm_a = _roi_stats(img_a, roi)
                                ed_b, sp_b, lm_b = _roi_stats(img_b, roi)
                                edge_target = 0.08
                                score_a = -abs(ed_a - edge_target) - 0.8*sp_a - 0.10*abs(lm_a - 0.5)
                                score_b = -abs(ed_b - edge_target) - 0.8*sp_b - 0.10*abs(lm_b - 0.5)

                                # Optional CLIP-Vision ref boost
                                if ref_embed is not None and clip_vision is not None:
                                    try:
                                        emb_a = _encode_clip_image(img_a, clip_vision, target_res=224)
                                        emb_b = _encode_clip_image(img_b, clip_vision, target_res=224)
                                        sim_a = float((emb_a * ref_embed).sum(dim=-1).mean().clamp(-1.0, 1.0).item())
                                        sim_b = float((emb_b * ref_embed).sum(dim=-1).mean().clamp(-1.0, 1.0).item())
                                        score_a += 0.25 * (0.5*(sim_a+1.0))
                                        score_b += 0.25 * (0.5*(sim_b+1.0))
                                    except Exception:
                                        pass

                                if score_b > score_a:
                                    current_latent = lat_b
                                else:
                                    current_latent = lat_a
                                try:
                                    del img_roi
                                except Exception:
                                    pass
                                try:
                                    del roi
                                except Exception:
                                    pass
                                try:
                                    del lat_in_a
                                    del lat_a
                                    del img_a
                                except Exception:
                                    pass
                                try:
                                    del lat_in_b
                                    del lat_b
                                    del img_b
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    if str(scheduler) == "MGHybrid":
                        try:
                            # Build ZeSmart hybrid sigmas with safe defaults
                            sigmas = _build_hybrid_sigmas(
                                sampler_model, int(current_steps), str(sampler_name), "hybrid",
                                mix=0.5, denoise=float(current_denoise), jitter=0.01, seed=int(iter_seed),
                                _debug=False, tail_smooth=0.15, auto_hybrid_tail=True, auto_tail_strength=0.4,
                            )
                            # Prepare latent + noise like in MG_ZeSmartSampler
                            lat_img = current_latent["samples"]
                            lat_img = _sample.fix_empty_latent_channels(sampler_model, lat_img)
                            batch_inds = current_latent.get("batch_index", None)
                            noise = _sample.prepare_noise(lat_img, int(iter_seed), batch_inds)
                            noise_mask = current_latent.get("noise_mask", None)
                            callback = _wrap_interruptible_callback(sampler_model, int(current_steps))
                            # cooperative cancel just before entering sampler
                            model_management.throw_exception_if_processing_interrupted()
                            disable_pbar = not _utils.PROGRESS_BAR_ENABLED
                            sampler_obj = _samplers.sampler_object(str(sampler_name))
                            samples = _sample.sample_custom(
                                sampler_model, noise, float(current_cfg), sampler_obj, sigmas,
                                positive, negative, lat_img,
                                noise_mask=noise_mask, callback=callback,
                                disable_pbar=disable_pbar, seed=int(iter_seed)
                            )
                            current_latent = {**current_latent}
                            current_latent["samples"] = samples
                        except Exception as e:
                            # Before any fallback, propagate user cancel if set
                            try:
                                model_management.throw_exception_if_processing_interrupted()
                            except Exception:
                                globals()["_MG_CANCEL_REQUESTED"] = False
                                raise
                            # Do not swallow user interruption; also check sentinel just in case
                            if isinstance(e, model_management.InterruptProcessingException) or globals().get("_MG_CANCEL_REQUESTED", False):
                                globals()["_MG_CANCEL_REQUESTED"] = False
                                raise
                            # Fallback to original path if anything goes wrong
                            print(f"[CADE2.5][MGHybrid] fallback to common_ksampler due to: {e}")
                            current_latent, = _interruptible_ksampler(
                                sampler_model, iter_seed, int(current_steps), current_cfg, sampler_name, _scheduler_names()[0],
                                positive, negative, current_latent, denoise=current_denoise)
                    else:
                        current_latent, = _interruptible_ksampler(
                            sampler_model, iter_seed, int(current_steps), current_cfg, sampler_name, scheduler,
                            positive, negative, current_latent, denoise=current_denoise)

                    # cooperative cancel immediately after sampling
                    model_management.throw_exception_if_processing_interrupted()
                    # Release heavy temporaries from sampler path
                    try:
                        del lat_img
                    except Exception:
                        pass
                    try:
                        del noise
                    except Exception:
                        pass
                    try:
                        del noise_mask
                    except Exception:
                        pass
                    try:
                        del callback
                    except Exception:
                        pass
                    try:
                        del sampler_obj
                    except Exception:
                        pass
                    try:
                        del sigmas
                    except Exception:
                        pass

                    # Latent buffer: post-sampling EMA update and drift measure
                    try:
                        z_now = current_latent["samples"].detach()
                        if lb_state["z_ema"] is None or tuple(lb_state["z_ema"].shape) != tuple(z_now.shape):
                            lb_state["z_ema"] = z_now.clone()
                            lb_state["anchor"] = z_now.clone()
                        else:
                            lb = float(max(0.0, min(0.99, lb_ema)))
                            lb_state["z_ema"] = lb * lb_state["z_ema"] + (1.0 - lb) * z_now
                        if int(lb_anchor_every) > 0 and ((i + 1) % int(lb_anchor_every) == 0):
                            lb_state["anchor"] = lb_state["z_ema"].clone()
                    except Exception:
                        pass
                    # local RMS drift (independent of UI)
                    try:
                        _cur = current_latent["samples"]
                        _prev = prev_samples
                        if _prev.device != _cur.device:
                            _prev = _prev.to(_cur.device)
                        if _prev.dtype != _cur.dtype:
                            _prev = _prev.to(dtype=_cur.dtype)
                        _diff = _cur - _prev
                        lb_state["drift_last"] = float(torch.sqrt(torch.mean(_diff * _diff)).item())
                    except Exception:
                        pass

                    if bool(latent_compare):
                        _cur = current_latent["samples"]
                        _prev = prev_samples
                        try:
                            if _prev.device != _cur.device:
                                _prev = _prev.to(_cur.device)
                            if _prev.dtype != _cur.dtype:
                                _prev = _prev.to(dtype=_cur.dtype)
                        except Exception:
                            pass
                        latent_diff = _cur - _prev
                        rms = torch.sqrt(torch.mean(latent_diff * latent_diff))
                        drift = float(rms.item())
                        if drift > float(threshold):
                            overshoot = max(0.0, drift - float(threshold))
                            damp = 1.0 - min(0.15, overshoot * 2.0)
                            current_denoise = max(0.20, current_denoise * damp)
                            cfg_damp = 0.997 if damp > 0.9 else 0.99
                            current_cfg = max(1.0, current_cfg * cfg_damp)
                    # Latent buffer: optional rebase toward anchor on overshoot
                    if bool(latent_buffer) and (iterations > 1) and (lb_state.get("anchor") is not None):
                        try:
                            dval = lb_state.get("drift_last", None)
                            if (dval is not None) and (dval > float(lb_rebase_thresh)):
                                rb = float(max(0.0, min(1.0, lb_rebase_rate)))
                                z = current_latent["samples"]
                                a = lb_state["anchor"]
                                if bool(lb_masked) and (onnx_mask_last is not None):
                                    m = onnx_mask_last.movedim(-1, 1)
                                    m = F.interpolate(m, size=(z.shape[-2], z.shape[-1]), mode='bilinear', align_corners=False).clamp(0, 1)
                                    m = m.expand(-1, z.shape[1], -1, -1)
                                    z = z * (1.0 - rb * m) + a * (rb * m)
                                else:
                                    z = z * (1.0 - rb) + a * rb
                                current_latent["samples"] = z
                        except Exception:
                            pass
                    try:
                        del prev_samples
                    except Exception:
                        pass

                    # AQClip-Lite: adaptive soft clipping in latent space (before decode)
                    try:
                        if bool(aqclip_enable):
                            if 'aq_state' not in locals():
                                aq_state = None
                            H_override = None
                            try:
                                if bool(aq_attn) and hasattr(sa_patch, "get_attention_entropy_map"):
                                    Hm = sa_patch.get_attention_entropy_map(clear=False)
                                    if Hm is not None:
                                        H_override = F.interpolate(Hm, size=(current_latent["samples"].shape[-2], current_latent["samples"].shape[-1]), mode="bilinear", align_corners=False)
                            except Exception:
                                H_override = None
                            z_new, aq_state = _aqclip_lite(
                                current_latent["samples"],
                                tile=int(aq_tile), stride=int(aq_stride),
                                alpha=float(aq_alpha), ema_state=aq_state, ema_beta=float(aq_ema_beta),
                                H_override=H_override,
                            )
                            current_latent["samples"] = z_new
                            try:
                                del H_override
                            except Exception:
                                pass
                            try:
                                del Hm
                            except Exception:
                                pass
                    except Exception:
                        pass

                    image = safe_decode(vae, current_latent)
                    # and again after decode before post-processing
                    model_management.throw_exception_if_processing_interrupted()

                    # Polish mode: keep global form (low frequencies) from reference while letting details refine
                    if bool(polish_enable) and (i >= int(polish_start_after)):
                        try:
                            # Prepare tensors
                            img = image
                            ref = reference_image if (reference_image is not None) else img
                            if ref.shape[1] != img.shape[1] or ref.shape[2] != img.shape[2]:
                                # resize reference to match current image
                                ref_n = ref.movedim(-1, 1)
                                ref_n = F.interpolate(ref_n, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
                                ref = ref_n.movedim(1, -1)
                            x = img.movedim(-1, 1)
                            r = ref.movedim(-1, 1)
                            # Low/high split via Gaussian blur
                            rad = max(1, int(round(float(polish_sigma) * 2)))
                            low_x = _gaussian_blur_nchw(x, sigma=float(polish_sigma), radius=rad)
                            low_r = _gaussian_blur_nchw(r, sigma=float(polish_sigma), radius=rad)
                            high_x = x - low_x
                            # Mix low from reference and current with ramp
                            # a starts from polish_keep_low_ramp and linearly ramps to polish_keep_low over remaining iterations
                            try:
                                denom = max(1, int(iterations) - int(polish_start_after))
                                t = max(0.0, min(1.0, (i - int(polish_start_after)) / denom))
                            except Exception:
                                t = 1.0
                            a0 = float(polish_keep_low_ramp)
                            at = float(polish_keep_low)
                            a = a0 + (at - a0) * t
                            low_mix = low_r * a + low_x * (1.0 - a)
                            new = low_mix + high_x
                            # Micro-detail injection on tail: very light HF boost gated by edges+depth
                            try:
                                phase = (i + 1) / max(1, int(iterations))
                                ramp = max(0.0, min(1.0, (phase - 0.70) / 0.30))
                                if ramp > 0.0:
                                    micro = x - _gaussian_blur_nchw(x, sigma=0.6, radius=1)
                                    gray = x.mean(dim=1, keepdim=True)
                                    sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=gray.dtype, device=gray.device).unsqueeze(1)
                                    sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=gray.dtype, device=gray.device).unsqueeze(1)
                                    gx = F.conv2d(gray, sobel_x, padding=1)
                                    gy = F.conv2d(gray, sobel_y, padding=1)
                                    mag = torch.sqrt(gx*gx + gy*gy)
                                    m_edge = (mag - mag.amin()) / (mag.amax() - mag.amin() + 1e-8)
                                    g_edge = (1.0 - m_edge).clamp(0.0, 1.0).pow(0.65)
                                    try:
                                        sz = (int(img.shape[1]), int(img.shape[2]))
                                        if depth_gate_cache.get("size") != sz or depth_gate_cache.get("mask") is None:
                                            model_path = os.path.join(os.path.dirname(__file__), '..', 'depth-anything', 'depth_anything_v2_vitl.pth')
                                            dm = _cf_build_depth_map(img, res=512, model_path=model_path, hires_mode=True)
                                            depth_gate_cache = {"size": sz, "mask": dm}
                                        dm = depth_gate_cache.get("mask")
                                        if dm is not None:
                                            g_depth = (dm.movedim(-1, 1).clamp(0,1)) ** 1.35
                                        else:
                                            g_depth = torch.ones_like(g_edge)
                                    except Exception:
                                        g_depth = torch.ones_like(g_edge)
                                    g = (g_edge * g_depth).clamp(0.0, 1.0)
                                    micro_boost = 0.018 * ramp
                                    new = new + micro_boost * (micro * g)
                            except Exception:
                                pass
                            # Edge-lock: protect edges from drift by biasing toward low_mix along edges
                            el = float(polish_edge_lock)
                            if el > 1e-6:
                                # Sobel edge magnitude on grayscale
                                gray = x.mean(dim=1, keepdim=True)
                                sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=gray.dtype, device=gray.device).unsqueeze(1)
                                sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=gray.dtype, device=gray.device).unsqueeze(1)
                                gx = F.conv2d(gray, sobel_x, padding=1)
                                gy = F.conv2d(gray, sobel_y, padding=1)
                                mag = torch.sqrt(gx*gx + gy*gy)
                                m = (mag - mag.amin()) / (mag.amax() - mag.amin() + 1e-8)
                                # Blend toward low_mix near edges
                                new = new * (1.0 - el*m) + (low_mix) * (el*m)
                            img2 = new.movedim(1, -1).clamp(0,1)
                            # Feed back to latent for next steps
                            current_latent = {"samples": safe_encode(vae, img2)}
                            image = img2
                            try:
                                del x
                                del r
                                del low_x
                                del low_r
                                del high_x
                                del low_mix
                                del new
                                del micro
                                del gray
                                del sobel_x
                                del sobel_y
                                del gx
                                del gy
                                del mag
                                del m_edge
                                del g_depth
                                del g
                                del ref_n
                                del ref
                                del img
                            except Exception:
                                pass
                            try:
                                clear_gpu_and_ram_cache()
                            except Exception:
                                pass
                        except Exception:
                            pass

                    # ONNX detectors (beta): fuse hands/face/pose mask if available (post-sampling; skip if already set)
                    if onnx_detectors and (i % max(1, 1) == 0) and (onnx_mask_last is None):
                        try:
                            import os
                            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
                            globals()["_ONNX_DEBUG"] = False
                            globals()["_ONNX_KPTS_ENABLE"] = bool(onnx_kpts_enable)
                            globals()["_ONNX_KPTS_SIGMA"] = float(onnx_kpts_sigma)
                            globals()["_ONNX_KPTS_GAIN"] = float(onnx_kpts_gain)
                            globals()["_ONNX_KPTS_CONF"] = float(onnx_kpts_conf)
                            onnx_mask = _onnx_build_mask(image, int(onnx_preview), float(onnx_sensitivity), models_dir, float(onnx_anomaly_gain))
                            if onnx_mask is not None:
                                onnx_mask_last = onnx_mask
                                om = onnx_mask.movedim(-1,1)
                                area = float(om.mean().item())
                                # verbose post-mask log removed; keep single compact log above
                                if area > 0.005:
                                    damp = 1.0 - min(0.25, 0.06 + onnx_sensitivity*0.05 + area*0.25)
                                    current_denoise = max(0.10, current_denoise * damp)
                                    current_cfg = max(1.0, current_cfg * (1.0 - 0.015*onnx_sensitivity))
                        except Exception:
                            pass

                    if reference_clean and (ref_embed is not None) and (i % max(1, ref_cooldown) == 0):
                        try:
                            cur_embed = _encode_clip_image(image, clip_vision, ref_preview)
                            dist = _clip_cosine_distance(cur_embed, ref_embed)
                            if dist > ref_threshold:
                                current_denoise = max(0.10, current_denoise * 0.9)
                                current_cfg = max(1.0, current_cfg * 0.99)
                            # store for next-iter latent buffer injection boost
                            try:
                                lb_state["ref_dist_last"] = float(dist)
                            except Exception:
                                pass
                        except Exception:
                            pass

                    if apply_upscale and current_scale != 1.0:
                        current_latent, image = MagicUpscaleModule().process_upscale(
                            current_latent, vae, upscale_method, current_scale)
                        # After upscale at large sizes, add a tiny HF sprinkle gated by edges+depth
                        try:
                            H, W = int(image.shape[1]), int(image.shape[2])
                            if max(H, W) > 1536:
                                # Simple BHWC blur
                                def _gb_bhwc(im: torch.Tensor, radius: float, sigma: float) -> torch.Tensor:
                                    if radius <= 0.0:
                                        return im
                                    pad = int(max(1, round(radius)))
                                    ksz = pad * 2 + 1
                                    k = _gaussian_kernel(ksz, sigma, device=im.device).to(dtype=im.dtype)
                                    k = k.unsqueeze(0).unsqueeze(0)
                                    b, h, w, c = im.shape
                                    xch = im.permute(0, 3, 1, 2)
                                    y = F.conv2d(F.pad(xch, (pad, pad, pad, pad), mode='reflect'), k.repeat(c, 1, 1, 1), groups=c)
                                    return y.permute(0, 2, 3, 1)
                                blur = _gb_bhwc(image, radius=1.0, sigma=0.8)
                                hf = (image - blur).clamp(-1, 1)
                                lum = (0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2])
                                kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=lum.device, dtype=lum.dtype).view(1, 1, 3, 3)
                                ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=lum.device, dtype=lum.dtype).view(1, 1, 3, 3)
                                g = torch.sqrt(F.conv2d(lum.unsqueeze(1), kx, padding=1)**2 + F.conv2d(lum.unsqueeze(1), ky, padding=1)**2).squeeze(1)
                                m = (g - g.amin()) / (g.amax() - g.amin() + 1e-8)
                                g_edge = (1.0 - m).clamp(0,1).pow(0.5).unsqueeze(-1)
                                try:
                                    sz = (H, W)
                                    if depth_gate_cache.get("size") != sz or depth_gate_cache.get("mask") is None:
                                        model_path = os.path.join(os.path.dirname(__file__), '..', 'depth-anything', 'depth_anything_v2_vitl.pth')
                                        dm = _cf_build_depth_map(image, res=512, model_path=model_path, hires_mode=True)
                                        depth_gate_cache = {"size": sz, "mask": dm}
                                    dm = depth_gate_cache.get("mask")
                                    if dm is not None:
                                        g_depth = dm.clamp(0,1) ** 1.35
                                    else:
                                        g_depth = torch.ones_like(g_edge)
                                except Exception:
                                    g_depth = torch.ones_like(g_edge)
                                g_tot = (g_edge * g_depth).clamp(0,1)
                                image = (image + 0.045 * hf * g_tot).clamp(0,1)
                        except Exception:
                            pass
                        current_cfg = max(4.0, current_cfg * (1.0 / current_scale))
                        current_denoise = max(0.15, current_denoise * (1.0 / current_scale))

                    current_steps = max(1, current_steps - steps_delta)
                    current_cfg = max(0.0, current_cfg - cfg_delta)
                    current_denoise = max(0.0, current_denoise - denoise_delta)
                    current_scale = max(1.0, current_scale - scale_delta)

                    if apply_upscale and current_scale != 1.0 and max(image.shape[1:3]) > 1024:
                        current_latent = {"samples": safe_encode(vae, image)}

        finally:
            # Always disable NAG patch and clear local mask, even on errors
            try:
                sa_patch.enable_crossattention_nag_patch(False)
            except Exception:
                pass
            # Turn off attention-entropy probe (AQClip Attn-mode) to avoid holding last maps
            try:
                if hasattr(sa_patch, "enable_attention_entropy_capture"):
                    sa_patch.enable_attention_entropy_capture(False)
            except Exception:
                pass
            # Disable KV pruning as well (avoid leaking state)
            try:
                if hasattr(sa_patch, "set_kv_prune"):
                    sa_patch.set_kv_prune(False, 1.0, 128)
            except Exception:
                pass
            try:
                sa_patch.CURRENT_PV_ACCUM = prev_accum
            except Exception:
                pass
            try:
                CURRENT_ONNX_MASK_BCHW = None
            except Exception:
                pass
            try:
                globals()["_MG_CANCEL_REQUESTED"] = False
                clear_gpu_and_ram_cache()
            except Exception:
                pass
            # best-effort cache cleanup on cancel or error
            try:
                clear_gpu_and_ram_cache()
            except Exception:
                pass

        if apply_ids:
            image, = IntelligentDetailStabilizer().stabilize(image, ids_strength)

        if apply_sharpen:
            image, = _sharpen_image(image, 2, 1.0, Sharpnes_strenght)

        # ONNX mask preview as IMAGE (RGB)
        if onnx_mask_last is None:
            onnx_mask_last = torch.zeros((image.shape[0], image.shape[1], image.shape[2], 1), device=image.device, dtype=image.dtype)
        onnx_mask_img = onnx_mask_last.repeat(1, 1, 1, 3).clamp(0, 1)

        # Final pass: remove isolated hot whites ("fireflies") without touching real edges/highlights 6.0/9.0, 0.05
        try:
            image = _despeckle_fireflies(image, thr=0.998, max_iso=4.0/9.0, grad_gate=0.15)
        except Exception:
            pass

        # Under-the-hood preview downscale for UI/output IMAGE to cap RAM during save/preview
        preview_downscale = False  # hard-coded default (can be toggled here if needed)
        try:
            if bool(preview_downscale):
                B, H, W, C = image.shape
                max_side = max(int(H), int(W))
                cap = 1920
                if max_side > cap:
                    scale = float(cap) / float(max_side)
                    nh = max(1, int(round(H * scale)))
                    nw = max(1, int(round(W * scale)))
                    x = image.movedim(-1, 1)
                    x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
                    image = x.movedim(1, -1).clamp(0, 1).to(dtype=image.dtype)
        except Exception:
            pass

        # Optional: save from node with low PNG compress to reduce RAM spike; ignore UI wiring
        try:
            if bool(auto_save):
                from comfy_api.latest._ui import ImageSaveHelper, FolderType
                _ = ImageSaveHelper.save_images(
                    [image], filename_prefix=str(save_prefix), folder_type=FolderType.output,
                    cls=CADEEasyUI, compress_level=int(save_compress))
        except Exception:
            pass

        return current_latent, image, int(current_steps), float(current_cfg), float(current_denoise), onnx_mask_img


# === Easy UI wrapper: show only top-level controls ===
class CADEEasyUI(ComfyAdaptiveDetailEnhancer25):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_step": (["Step 1", "Step 2", "Step 3", "Step 4"], {"default": "Step 1", "tooltip": "Choose the Step preset. Toggle Custom below to apply UI values; otherwise Step preset values are used."}),
                "custom": ("BOOLEAN", {"default": False, "tooltip": "Custom override: when enabled, your UI values override the selected Step for visible controls; hidden parameters still come from the Step preset."}),
                "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "vae": ("VAE", {}),
                "latent": ("LATENT", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Seed 0 = SmartSeed (Sobol + light probe). Non?zero = fixed seed (deterministic)."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "sampler_name": (_sampler_names(), {"default": _sampler_names()[0]}),
                "scheduler": (_scheduler_names(), {"default": "MGHybrid"}),
            },
            "optional": {
                # Reference inputs must remain available in Easy
                "reference_image": ("IMAGE", {}),
                "clip_vision": ("CLIP_VISION", {}),
                # CLIPSeg prompt
                "clipseg_text": ("STRING", {"default": "", "multiline": False, "tooltip": "This field tells what the step should focus on (e.g., hand, feet, face). Separate with commas."}),
            }
        }

    # Easy outputs (hide steps/cfg/denoise)
    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("LATENT", "IMAGE", "mask_preview")
    FUNCTION = "apply_easy"

    def apply_easy(self,
                   preset_step,
                   model, positive, negative, vae, latent,
                   seed, steps, cfg, denoise, sampler_name, scheduler,
                   clipseg_text="", reference_image=None, clip_vision=None, custom=False):
        lat, img, _s, _c, _d, mask = super().apply_cade2(
            model, vae, positive, negative, latent,
            int(seed), int(steps), float(cfg), float(denoise),
            str(sampler_name), str(scheduler), 0.0,
            preset_step=str(preset_step), custom_override=bool(custom), clipseg_text=str(clipseg_text),
            reference_image=reference_image,
            clip_vision=clip_vision,
        )
        return lat, img, mask

        # Show simpler outputs in Easy variant
        RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE")
        RETURN_NAMES = ("LATENT", "IMAGE", "mask_preview")
        FUNCTION = "apply_easy"

        def apply_easy(self,
                       preset_step,
                       model, positive, negative, vae, latent,
                       seed, steps, cfg, denoise, sampler_name, scheduler,
                       clipseg_text=""):
            lat, img, _s, _c, _d, mask = super().apply_cade2(
                model, vae, positive, negative, latent,
                int(seed), int(steps), float(cfg), float(denoise),
                str(sampler_name), str(scheduler), 0.0,
                preset_step=str(preset_step), custom_override=bool(custom), clipseg_text=str(clipseg_text),
            )
            return lat, img, mask





# === Smart seed helpers (Sobol/Halton + light probing) ===
def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF

def _halton_single(index: int, base: int) -> float:
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f = f / base
        r = r + f * (i % base)
        i //= base
    return r

def _sobol_like_2d(n: int, anchor: int) -> tuple[float, float]:
    # lightweight 2D low-discrepancy via Halton(2,3) scrambled by anchor
    i = n + 1 + (anchor % 9973)
    return (_halton_single(i, 2), _halton_single(i, 3))

def _edge_density(img_bhwc: torch.Tensor) -> float:
    lum = (0.2126 * img_bhwc[..., 0] + 0.7152 * img_bhwc[..., 1] + 0.0722 * img_bhwc[..., 2])
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img_bhwc.device, dtype=img_bhwc.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img_bhwc.device, dtype=img_bhwc.dtype).view(1,1,3,3)
    gx = F.conv2d(lum.unsqueeze(1), kx, padding=1)
    gy = F.conv2d(lum.unsqueeze(1), ky, padding=1)
    g = torch.sqrt(gx*gx + gy*gy)
    return float(g.mean().item())

def _speckle_fraction(img_bhwc: torch.Tensor) -> float:
    # reuse S/V-based candidate mask from despeckle logic (no replacement) to estimate fraction
    R, G, Bc = img_bhwc[..., 0], img_bhwc[..., 1], img_bhwc[..., 2]
    V = torch.maximum(R, torch.maximum(G, Bc))
    mi = torch.minimum(R, torch.minimum(G, Bc))
    S = 1.0 - (mi / (V + 1e-6))
    v_thr = 0.98
    s_thr = 0.12
    cand = (V > v_thr) & (S < s_thr)
    return float(cand.float().mean().item())

def _smart_seed_state_path() -> str:
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "state")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "smart_seed.json")

def _smart_seed_counter(anchor: int) -> int:
    import os, json
    path = _smart_seed_state_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    key = hex(anchor & 0xFFFFFFFFFFFFFFFF)
    n = int(data.get(key, 0))
    data[key] = n + 1
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return n

def _smart_seed_select(model,
                       vae,
                       positive,
                       negative,
                       latent,
                       sampler_name: str,
                       scheduler: str,
                       cfg: float,
                       denoise: float,
                       base_seed: int | None = None,
                       k: int = 6,
                       probe_steps: int = 6,
                       clip_vision=None,
                       reference_image=None,
                       clipseg_text: str = "",
                       step_tag: str | None = None) -> int:
    # Log start of SmartSeed selection
    try:
        # cooperative cancel before any smart-seed work
        model_management.throw_exception_if_processing_interrupted()
        try:
            # Visual separation before SmartSeed block
            print("")
            print("")
            if step_tag:
                print(f"\x1b[34m==== {step_tag}, Smart_seed_random: Start ====\x1b[0m")
            else:
                print("\x1b[34m==== Smart_seed_random: Start ====\x1b[0m")
        except Exception:
            pass

        # Optional: precompute CLIP-Vision embedding of reference image
        ref_embed = None
        if (clip_vision is not None) and (reference_image is not None):
            try:
                ref_embed = _encode_clip_image(reference_image, clip_vision, target_res=224)
            except Exception:
                ref_embed = None

        # Anchor from latent shape + sampler/scheduler (+ cfg/denoise)
        sh = latent["samples"].shape if isinstance(latent, dict) and "samples" in latent else None
        anchor = 1469598103934665603  # FNV offset basis
        for v in (sh[2] if sh else 0, sh[3] if sh else 0, len(str(sampler_name)), len(str(scheduler)), int(cfg * 1000), int(denoise * 1000)):
            anchor = _splitmix64(anchor ^ int(v))
        # Advance a persistent counter per anchor to vary indices between runs
        offset = _smart_seed_counter(anchor) * 7

        # Build K candidate seeds from Halton(2,3)
        cands: list[int] = []
        for i in range(k):
            u, v = _sobol_like_2d(offset + i, anchor)
            lo = int(u * (1 << 32)) & 0xFFFFFFFF
            hi = int(v * (1 << 32)) & 0xFFFFFFFF
            seed64 = _splitmix64((hi << 32) ^ lo ^ anchor)
            cands.append(seed64 & 0xFFFFFFFFFFFFFFFF)

        best_seed = cands[0]
        best_score = -1e9
        for sd in cands:
            # allow user to cancel between candidates
            model_management.throw_exception_if_processing_interrupted()
            try:
                # quick KSampler preview at low steps
                lat_in = {"samples": latent["samples"].clone()} if isinstance(latent, dict) else latent
                lat_out, = _interruptible_ksampler(
                    model, int(sd), int(probe_steps), float(cfg), str(sampler_name), str(scheduler),
                    positive, negative, lat_in, denoise=float(min(denoise, 0.65))
                )
                img = safe_decode(vae, lat_out)
                # and again right after decode
                model_management.throw_exception_if_processing_interrupted()
                # Base score: edge density toward a target + low speckle + balanced exposure
                ed = _edge_density(img)
                speck = _speckle_fraction(img)
                lum = float(img.mean().item())
                edge_target = 0.10
                score = -abs(ed - edge_target) - 2.0 * speck - 0.5 * abs(lum - 0.5)

                # Perceptual metrics: luminance std and Laplacian variance (downscaled)
                try:
                    lum_t = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])
                    lstd = float(lum_t.std().item())
                    lch = lum_t.unsqueeze(1)
                    lch_small = F.interpolate(lch, size=(128, 128), mode='bilinear', align_corners=False)
                    lap_k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=lch_small.device, dtype=lch_small.dtype).view(1, 1, 3, 3)
                    lap = F.conv2d(lch_small, lap_k, padding=1)
                    lap_var = float(lap.var().item())
                    score += 0.15 * lstd + 0.10 * lap_var
                except Exception:
                    pass

                # Semantic alignment via CLIP-Vision when available
                if ref_embed is not None and clip_vision is not None:
                    try:
                        cand_embed = _encode_clip_image(img, clip_vision, target_res=224)
                        sim = float((cand_embed * ref_embed).sum(dim=-1).mean().clamp(-1.0, 1.0).item())
                        sim01 = 0.5 * (sim + 1.0)
                        score += 0.75 * sim01
                    except Exception:
                        pass

                # Focus coverage via CLIPSeg when text provided
                if isinstance(clipseg_text, str) and clipseg_text.strip() != "":
                    try:
                        cmask = _clipseg_build_mask(img, clipseg_text, preview=192, threshold=0.40, blur=5.0, dilate=2, gain=1.0)
                        if cmask is not None:
                            area = float(cmask.mean().item())
                            cov_target = 0.06
                            cov_score = 1.0 - min(1.0, abs(area - cov_target) / max(cov_target, 1e-3))
                            score += 0.30 * cov_score
                    except Exception:
                        pass
                if score > best_score:
                    best_score = score
                    best_seed = sd
                try:
                    del img
                except Exception:
                    pass
                try:
                    del lat_out
                except Exception:
                    pass
                try:
                    del lat_in
                except Exception:
                    pass
                try:
                    del lch_small
                except Exception:
                    pass
                try:
                    del lap
                except Exception:
                    pass
                try:
                    del cand_embed
                except Exception:
                    pass
                try:
                    del cmask
                except Exception:
                    pass
            except Exception as e:
                # do not swallow user interruption; also honour sentinel
                if isinstance(e, model_management.InterruptProcessingException) or globals().get("_MG_CANCEL_REQUESTED", False):
                    globals()["_MG_CANCEL_REQUESTED"] = False
                    raise
                continue

        # Log end with selected seed
        try:
            if step_tag:
                print(f"\x1b[34m==== {step_tag}, Smart_seed_random: End. Seed is: {int(best_seed & 0xFFFFFFFFFFFFFFFF)} ====\x1b[0m")
            else:
                print(f"\x1b[34m==== Smart_seed_random: End. Seed is: {int(best_seed & 0xFFFFFFFFFFFFFFFF)} ====\x1b[0m")
        except Exception:
            pass
        return int(best_seed & 0xFFFFFFFFFFFFFFFF)
    except Exception as e:
        if isinstance(e, model_management.InterruptProcessingException) or globals().get("_MG_CANCEL_REQUESTED", False):
            globals()["_MG_CANCEL_REQUESTED"] = False
            # propagate cancel to stop the whole prompt cleanly
            raise
        # Fallback to time-based random
        try:
            import time
            fallback_seed = int(_splitmix64(int(time.time_ns())))
        except Exception:
            fallback_seed = int(base_seed or 0)
        try:
            if step_tag:
                print(f"\x1b[34m==== {step_tag}, Smart_seed_random: End. Seed is: {fallback_seed} ====\x1b[0m")
            else:
                print(f"\x1b[34m==== Smart_seed_random: End. Seed is: {fallback_seed} ====\x1b[0m")
        except Exception:
            pass
        return fallback_seed


def _wrap_interruptible_callback(model, steps):
    base_cb = nodes.latent_preview.prepare_callback(model, int(steps))
    def _cb(step, x0, x, total_steps):
        model_management.throw_exception_if_processing_interrupted()
        return base_cb(step, x0, x, total_steps)
    return _cb

def _interruptible_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                            positive, negative, latent, denoise=1.0):
    lat_img = _sample.fix_empty_latent_channels(model, latent["samples"])
    batch_inds = latent.get("batch_index", None)
    noise = _sample.prepare_noise(lat_img, int(seed), batch_inds)
    noise_mask = latent.get("noise_mask", None)
    callback = _wrap_interruptible_callback(model, int(steps))
    # cooperative cancel just before sampler entry
    model_management.throw_exception_if_processing_interrupted()
    disable_pbar = not _utils.PROGRESS_BAR_ENABLED
    samples = _sample.sample(
        model, noise, int(steps), float(cfg), str(sampler_name), str(scheduler),
        positive, negative, lat_img,
        denoise=float(denoise), disable_noise=False, start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=noise_mask, callback=callback,
        disable_pbar=disable_pbar, seed=int(seed)
    )
    out = {**latent}
    out["samples"] = samples
    return (out,)
