import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np

import comfy.model_management as model_management


_DEPTH_INIT = False
_DEPTH_MODEL = None
_DEPTH_PROC = None


def _insert_aux_path():
    try:
        base = os.path.dirname(os.path.dirname(__file__))  # .../custom_nodes
        aux_root = os.path.join(base, 'comfyui_controlnet_aux')
        aux_src = os.path.join(aux_root, 'src')
        for p in (aux_src, aux_root):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
    except Exception:
        pass


def _try_init_depth_anything(model_path: str):
    global _DEPTH_INIT, _DEPTH_MODEL, _DEPTH_PROC
    if _DEPTH_INIT:
        return _DEPTH_MODEL is not None
    _DEPTH_INIT = True
    # Resolve model path: allow 'auto' or directory and prefer vitl>vitb>vits>vitg
    try:
        def _prefer_order(paths):
            order = ["vitl", "vitb", "vits", "vitg"]
            scored = []
            for p in paths:
                name = os.path.basename(p).lower()
                score = 100
                for i, tag in enumerate(order):
                    if tag in name:
                        score = i
                        break
                scored.append((score, p))
            scored.sort(key=lambda x: x[0])
            return [p for _, p in scored]

        def _resolve_path(mp: str) -> str:
            if isinstance(mp, str) and mp.strip().lower() == "auto":
                mp = ""
            if mp and os.path.isfile(mp):
                return mp
            search_dirs = []
            if mp and os.path.isdir(mp):
                search_dirs.append(mp)
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'depth-anything')
            search_dirs.append(base_dir)
            cand = []
            for d in search_dirs:
                try:
                    for fn in os.listdir(d):
                        if fn.lower().endswith('.pth') and 'depth_anything_v2' in fn.lower():
                            cand.append(os.path.join(d, fn))
                except Exception:
                    pass
            if cand:
                return _prefer_order(cand)[0]
            return mp

        model_path = _resolve_path(model_path)
    except Exception:
        pass
    # Prefer our vendored implementation first
    try:
        from ...vendor.depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        # Guess config from filename
        fname = os.path.basename(model_path or '')
        cfgs = {
            'depth_anything_v2_vits.pth': dict(encoder='vits', features=64, out_channels=[48,96,192,384]),
            'depth_anything_v2_vitb.pth': dict(encoder='vitb', features=128, out_channels=[96,192,384,768]),
            'depth_anything_v2_vitl.pth': dict(encoder='vitl', features=256, out_channels=[256,512,1024,1024]),
            'depth_anything_v2_vitg.pth': dict(encoder='vitg', features=384, out_channels=[1536,1536,1536,1536]),
            'depth_anything_v2_metric_vkitti_vitl.pth': dict(encoder='vitl', features=256, out_channels=[256,512,1024,1024]),
            'depth_anything_v2_metric_hypersim_vitl.pth': dict(encoder='vitl', features=256, out_channels=[256,512,1024,1024]),
        }
        # fallback to vitl if unknown
        cfg = cfgs.get(fname, cfgs['depth_anything_v2_vitl.pth'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        m = DepthAnythingV2(**cfg)
        sd = torch.load(model_path, map_location='cpu')
        m.load_state_dict(sd)
        _DEPTH_MODEL = m.to(device).eval()
        _DEPTH_PROC = True
        return True
    except Exception:
        # Try local checkout of comfyui_controlnet_aux (if present)
        _insert_aux_path()
        try:
            from custom_controlnet_aux.depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
            fname = os.path.basename(model_path or '')
            cfgs = {
                'depth_anything_v2_vits.pth': dict(encoder='vits', features=64, out_channels=[48,96,192,384]),
                'depth_anything_v2_vitb.pth': dict(encoder='vitb', features=128, out_channels=[96,192,384,768]),
                'depth_anything_v2_vitl.pth': dict(encoder='vitl', features=256, out_channels=[256,512,1024,1024]),
                'depth_anything_v2_vitg.pth': dict(encoder='vitg', features=384, out_channels=[1536,1536,1536,1536]),
                'depth_anything_v2_metric_vkitti_vitl.pth': dict(encoder='vitl', features=256, out_channels=[256,512,1024,1024]),
                'depth_anything_v2_metric_hypersim_vitl.pth': dict(encoder='vitl', features=256, out_channels=[256,512,1024,1024]),
            }
            cfg = cfgs.get(fname, cfgs['depth_anything_v2_vitl.pth'])
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            m = DepthAnythingV2(**cfg)
            sd = torch.load(model_path, map_location='cpu')
            m.load_state_dict(sd)
            _DEPTH_MODEL = m.to(device).eval()
            _DEPTH_PROC = True
            return True
        except Exception:
            # Fallback: packaged auxiliary API
            try:
                from controlnet_aux.depth_anything import DepthAnythingDetector, DepthAnythingV2  # type: ignore
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                _DEPTH_MODEL = DepthAnythingV2(model_path=model_path, device=device)
                _DEPTH_PROC = True
                return True
            except Exception:
                _DEPTH_MODEL = None
                _DEPTH_PROC = False
                return False


def _build_depth_map(image_bhwc: torch.Tensor, res: int, model_path: str, hires_mode: bool = True) -> torch.Tensor:
    B, H, W, C = image_bhwc.shape
    dev = image_bhwc.device
    dtype = image_bhwc.dtype
    # Choose target min-side for processing. In hires mode we allow higher caps and keep aspect.
    # DepthAnything v2 can be memory-hungry on large inputs; cap min-side at 1024
    cap = 1024
    target = int(max(16, min(cap, res)))
    if _try_init_depth_anything(model_path):
        try:
            # to CPU uint8
            img = image_bhwc.detach().to('cpu')
            x = img[0].movedim(-1, 0).unsqueeze(0)
            # keep aspect ratio: scale so that min(H,W) == target
            _, Cc, Ht, Wt = x.shape
            min_side = max(1, min(Ht, Wt))
            scale = float(target) / float(min_side)
            out_h = max(1, int(round(Ht * scale)))
            out_w = max(1, int(round(Wt * scale)))
            x = F.interpolate(x, size=(out_h, out_w), mode='bilinear', align_corners=False)
            # make channels-last and ensure contiguous layout for OpenCV
            arr = (x[0].movedim(0, -1).contiguous().numpy() * 255.0).astype('uint8')
            # Prefer direct DepthAnythingV2 inference if model has infer_image
            if hasattr(_DEPTH_MODEL, 'infer_image'):
                import cv2
                # Drive input_size from desired depth resolution (min side), let DA keep aspect
                input_sz = int(max(224, min(cap, res)))
                depth = _DEPTH_MODEL.infer_image(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), input_size=input_sz, max_depth=20.0)
                d = np.asarray(depth, dtype=np.float32)
                # Normalize DepthAnythingV2 output (0..max_depth) to 0..1
                d = d / 20.0
            else:
                depth = _DEPTH_MODEL(arr)
                d = np.asarray(depth, dtype=np.float32)
            if d.max() > 1.0:
                d = d / 255.0
            d = torch.from_numpy(d)[None, None]  # 1,1,h,w
            d = F.interpolate(d, size=(H, W), mode='bilinear', align_corners=False)
            d = d[0, 0].to(device=dev, dtype=dtype)
            # Heuristic de-banding: sometimes upstream returns column-wise banding
            # Detect when column variance >> row variance, indicating vertical stripes
            try:
                with torch.no_grad():
                    dr = d.clamp(0,1)
                    # compute per-axis variance summaries in a small, cheap way
                    vcol = torch.var(dr, dim=0).mean()
                    vrow = torch.var(dr, dim=1).mean()
                    if torch.isfinite(vcol) and torch.isfinite(vrow) and (vcol > 8.0 * (vrow + 1e-6)):
                        # Apply mild horizontal smoothing only (reduce vertical banding)
                        k = max(3, int(round(min(W, 21) // 2 * 2 + 1)))  # odd kernel up to ~21
                        dr2 = F.avg_pool2d(dr.unsqueeze(0).unsqueeze(0), kernel_size=(1, k), stride=1, padding=(0, k//2))[0,0]
                        # preserve global contrast by gentle blend
                        d = (0.6 * dr + 0.4 * dr2).clamp(0,1)
            except Exception:
                pass
            d = d.clamp(0, 1)
            return d
        except Exception:
            pass
    # Fallback pseudo-depth: luminance + gentle blur
    lum = (0.2126 * image_bhwc[..., 0] + 0.7152 * image_bhwc[..., 1] + 0.0722 * image_bhwc[..., 2]).to(dtype=dtype)
    x = lum.movedim(-1, 0).unsqueeze(0) if lum.ndim == 3 else lum.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
    x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x[0, 0].clamp(0, 1)


def _pyracanny(image_bhwc: torch.Tensor,
               low: int,
               high: int,
               res: int,
               thin_iter: int = 0,
               edge_boost: float = 0.0,
               smart_tune: bool = False,
               smart_boost: float = 0.2,
               preserve_aspect: bool = True) -> torch.Tensor:
    try:
        import cv2
    except Exception:
        # Fallback: simple Sobel magnitude
        x = image_bhwc.movedim(-1, 1)
        xg = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(xg, torch.tensor([[[-1, 0, 1],[-2,0,2],[-1,0,1]]], dtype=x.dtype, device=x.device).unsqueeze(1), padding=1)
        gy = F.conv2d(xg, torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=x.dtype, device=x.device).unsqueeze(1), padding=1)
        mag = torch.sqrt(gx*gx + gy*gy)
        mag = (mag - mag.amin())/(mag.amax()-mag.amin()+1e-6)
        return mag[0,0].clamp(0,1)
    B,H,W,C = image_bhwc.shape
    img = (image_bhwc.detach().to('cpu')[0].contiguous().numpy()*255.0).astype('uint8')
    cap = 4096
    target = int(max(64, min(cap, res)))
    if preserve_aspect:
        scale = float(target) / float(max(1, min(H, W)))
        out_h = max(8, int(round(H * scale)))
        out_w = max(8, int(round(W * scale)))
        img_res = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_res = cv2.resize(img, (target, target), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
    pyr_scales = [1.0, 0.5, 0.25]
    acc = None
    for s in pyr_scales:
        if preserve_aspect:
            sz = (max(8, int(round(img_res.shape[1]*s))), max(8, int(round(img_res.shape[0]*s))))
        else:
            sz = (max(8, int(target*s)), max(8, int(target*s)))
        g = cv2.resize(gray, sz, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (5,5), 0)
        e = cv2.Canny(g, threshold1=int(low*s), threshold2=int(high*s))
        e = cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR)
        e = (e.astype(np.float32)/255.0)
        acc = e if acc is None else np.maximum(acc, e)
    # Estimate density and sharpness for smart tuning
    edensity_pre = None
    try:
        edensity_pre = float(np.mean(acc)) if acc is not None else None
    except Exception:
        edensity_pre = None
    lap_var = None
    try:
        g32 = gray.astype(np.float32) / 255.0
        lap = cv2.Laplacian(g32, cv2.CV_32F)
        lap_var = float(lap.var())
    except Exception:
        lap_var = None

    # optional thinning
    try:
        thin_iter_eff = int(thin_iter)
        if smart_tune:
            # simple heuristic: more thinning on high res and dense edges
            auto = 0
            if target >= 1024:
                auto += 1
            if target >= 1400:
                auto += 1
            if edensity_pre is not None and edensity_pre > 0.12:
                auto += 1
            if edensity_pre is not None and edensity_pre < 0.05:
                auto = max(0, auto - 1)
            thin_iter_eff = max(thin_iter_eff, min(3, auto))
        if thin_iter_eff > 0:
            import cv2
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
                th = acc.copy()
                th = (th*255).astype('uint8')
                th = cv2.ximgproc.thinning(th)
                acc = th.astype(np.float32)/255.0
            else:
                # simple erosion-based thinning approximation
                kernel = np.ones((3,3), np.uint8)
                t = (acc*255).astype('uint8')
                for _ in range(int(thin_iter_eff)):
                    t = cv2.erode(t, kernel, iterations=1)
                acc = t.astype(np.float32)/255.0
    except Exception:
        pass
    # optional edge boost (unsharp on edge map)
    # We fix a gentle boost for micro‑contrast; smart_tune may nudge it slightly
    boost_eff = 0.10
    if smart_tune:
        try:
            lv = 0.0 if lap_var is None else max(0.0, min(1.0, lap_var / 2.0))
            dens = 0.0 if edensity_pre is None else float(max(0.0, min(1.0, edensity_pre)))
            boost_eff = max(0.05, min(0.20, boost_eff + (1.0 - dens) * 0.05 + (1.0 - lv) * 0.02))
        except Exception:
            pass
    if boost_eff and boost_eff != 0.0:
        try:
            import cv2
            blur = cv2.GaussianBlur(acc, (0,0), sigmaX=1.0)
            acc = np.clip(acc + float(boost_eff)*(acc - blur), 0.0, 1.0)
        except Exception:
            pass
    ed = torch.from_numpy(acc).to(device=image_bhwc.device, dtype=image_bhwc.dtype)
    return ed.clamp(0,1)


def _blend(depth: torch.Tensor, edges: torch.Tensor, mode: str, factor: float) -> torch.Tensor:
    depth = depth.clamp(0,1)
    edges = edges.clamp(0,1)
    if mode == 'max':
        return torch.maximum(depth, edges)
    if mode == 'edge_over_depth':
        # edges override depth (edge=1) while preserving depth elsewhere
        return (depth * (1.0 - edges) + edges).clamp(0,1)
    # normal
    f = float(max(0.0, min(1.0, factor)))
    return (depth*(1.0-f) + edges*f).clamp(0,1)


def _apply_controlnet_separate(positive, negative, control_net, image_bhwc: torch.Tensor,
                               strength_pos: float, strength_neg: float,
                               start_percent: float, end_percent: float, vae=None,
                               apply_to_uncond: bool = False,
                               stack_prev_control: bool = False):
    control_hint = image_bhwc.movedim(-1,1)
    out_pos = []
    out_neg = []
    # POS
    for t in positive:
        d = t[1].copy()
        prev = d.get('control', None) if stack_prev_control else None
        c_net = control_net.copy().set_cond_hint(control_hint, float(strength_pos), (start_percent, end_percent), vae=vae, extra_concat=[])
        c_net.set_previous_controlnet(prev)
        d['control'] = c_net
        d['control_apply_to_uncond'] = bool(apply_to_uncond)
        out_pos.append([t[0], d])
    # NEG
    for t in negative:
        d = t[1].copy()
        prev = d.get('control', None) if stack_prev_control else None
        c_net = control_net.copy().set_cond_hint(control_hint, float(strength_neg), (start_percent, end_percent), vae=vae, extra_concat=[])
        c_net.set_previous_controlnet(prev)
        d['control'] = c_net
        d['control_apply_to_uncond'] = bool(apply_to_uncond)
        out_neg.append([t[0], d])
    return out_pos, out_neg


class MG_ControlFusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input RGB image (B,H,W,3) in 0..1."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning to apply ControlNet to."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning to apply ControlNet to."}),
                "control_net": ("CONTROL_NET", {"tooltip": "ControlNet module receiving the fused mask as hint."}),
                "vae": ("VAE", {"tooltip": "VAE used by ControlNet when encoding the hint."}),
            },
            "optional": {
                "enable_depth": ("BOOLEAN", {"default": True, "tooltip": "Enable depth map fusion (Depth Anything v2 if available)."}),
                "depth_model_path": ("STRING", {"default": os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MagicNodes','depth-anything','depth_anything_v2_vitl.pth') if False else os.path.join(os.path.dirname(__file__), '..','depth-anything','depth_anything_v2_vitl.pth'), "tooltip": "Path to Depth Anything v2 .pth weights (vits/vitb/vitl/vitg)."}),
                "depth_resolution": ("INT", {"default": 768, "min": 64, "max": 1024, "step": 64, "tooltip": "Depth min-side resolution (cap 1024). In Hi‑Res mode drives DepthAnything input_size."}),
                "enable_pyra": ("BOOLEAN", {"default": True, "tooltip": "Enable PyraCanny edge detector."}),
                "pyra_low": ("INT", {"default": 109, "min": 0, "max": 255, "tooltip": "Canny low threshold (0..255)."}),
                "pyra_high": ("INT", {"default": 147, "min": 0, "max": 255, "tooltip": "Canny high threshold (0..255)."}),
                "pyra_resolution": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64, "tooltip": "Working resolution for edges (min side, keeps aspect)."}),
                "edge_thin_iter": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "Thinning iterations for edges (skeletonize). 0 = off."}),
                "edge_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Opacity for edges before blending (0..1)."}),
                "edge_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Deprecated: internal boost fixed (~0.10); use edge_alpha instead."}),
                "smart_tune": ("BOOLEAN", {"default": False, "tooltip": "Auto-adjust thinning/boost from image edge density and sharpness."}),
                "smart_boost": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Scale for auto edge boost when Smart Tune is on."}),
                "blend_mode": (["normal","max","edge_over_depth"], {"default": "normal", "tooltip": "Depth+edges merge: normal (mix), max (strongest), edge_over_depth (edges overlay)."}),
                "blend_factor": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Blend strength for edges into depth (depends on mode)."}),
                "strength_pos": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "ControlNet strength for positive branch."}),
                "strength_neg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "ControlNet strength for negative branch."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Start percentage along the sampling schedule."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "End percentage along the sampling schedule."}),
                "preview_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64, "tooltip": "Preview minimum side (keeps aspect ratio)."}),
                "mask_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Preview brightness multiplier (visualization only)."}),
                "preview_show_strength": ("BOOLEAN", {"default": True, "tooltip": "Multiply preview by ControlNet strength for visualization."}),
                "preview_strength_branch": (["positive","negative","max","avg"], {"default": "max", "tooltip": "Which strength to reflect in preview (display only)."}),
                "hires_mask_auto": ("BOOLEAN", {"default": True, "tooltip": "High‑res mask: keep aspect ratio, scale by minimal side for depth/edges, and drive DepthAnything with your depth_resolution (no 2K cap)."}),
                "apply_to_uncond": ("BOOLEAN", {"default": False, "tooltip": "Apply ControlNet hint to the unconditional branch as well (stronger global hold on very large images)."}),
                "stack_prev_control": ("BOOLEAN", {"default": False, "tooltip": "Chain with any previously attached ControlNet in the conditioning (advanced). Off = replace to avoid memory bloat."}),
                # Split apply: chain Depth and Edges with separate schedules/strengths (fixed order: depth -> edges)
                "split_apply": ("BOOLEAN", {"default": False, "tooltip": "Apply Depth and Edges as two chained ControlNets (fixed order: depth then edges)."}),
                "edge_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Edges start percent (when split is enabled)."}),
                "edge_end_percent": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Edges end percent (when split is enabled)."}),
                "depth_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Depth start percent (when split is enabled)."}),
                "depth_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Depth end percent (when split is enabled)."}),
                "edge_strength_mul": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Multiply global strength for Edges when split is enabled."}),
                "depth_strength_mul": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Multiply global strength for Depth when split is enabled."}),
                # Extra edge controls (bottom)
                "edge_width": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 1.5, "step": 0.05, "tooltip": "Edge thickness adjust: negative thins, positive thickens."}),
                "edge_smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Small smooth on edges to reduce pixelation (0..1)."}),
                "edge_single_line": ("BOOLEAN", {"default": False, "tooltip": "Try to collapse double outlines into a single centerline."}),
                "edge_single_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of single-line collapse (0..1). 0 = off, 1 = strong."}),
                "edge_depth_gate": ("BOOLEAN", {"default": False, "tooltip": "Weigh edges by depth so distant lines are fainter."}),
                "edge_depth_gamma": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 4.0, "step": 0.1, "tooltip": "Gamma for depth gating: edges *= (1−depth)^gamma."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","IMAGE")
    RETURN_NAMES = ("positive","negative","mask_preview")
    FUNCTION = "apply"
    CATEGORY = "MagicNodes"

    def apply(self, image, positive, negative, control_net, vae,
              enable_depth=True, depth_model_path="", depth_resolution=1024,
              enable_pyra=True, pyra_low=109, pyra_high=147, pyra_resolution=1024,
              edge_thin_iter=0, edge_alpha=1.0, edge_boost=0.0,
              smart_tune=False, smart_boost=0.2,
              blend_mode="normal", blend_factor=0.02,
              strength_pos=1.0, strength_neg=1.0, start_percent=0.0, end_percent=1.0,
              preview_res=1024, mask_brightness=1.0,
              preview_show_strength=True, preview_strength_branch="max",
              hires_mask_auto=True, apply_to_uncond=False, stack_prev_control=False,
              edge_width=0.0, edge_smooth=0.0, edge_single_line=False, edge_single_strength=0.0,
              edge_depth_gate=False, edge_depth_gamma=1.5,
              split_apply=False, edge_start_percent=0.0, edge_end_percent=0.6,
              depth_start_percent=0.0, depth_end_percent=1.0,
              edge_strength_mul=1.0, depth_strength_mul=1.0):

        dev = image.device
        dtype = image.dtype
        B,H,W,C = image.shape
        # Build depth/edges
        depth = None
        edges = None
        if enable_depth:
            model_path = depth_model_path or os.path.join(os.path.dirname(__file__), '..','depth-anything','depth_anything_v2_vitl.pth')
            depth = _build_depth_map(image, int(depth_resolution), model_path, bool(hires_mask_auto))
        if enable_pyra:
            edges = _pyracanny(image,
                               int(pyra_low), int(pyra_high), int(pyra_resolution),
                               int(edge_thin_iter), float(edge_boost),
                               bool(smart_tune), float(smart_boost), bool(hires_mask_auto))
        if depth is None and edges is None:
            # Nothing to do: return inputs and zero preview
            prev = torch.zeros((B, max(H,1), max(W,1), 3), device=dev, dtype=dtype)
            return positive, negative, prev

        if depth is None:
            depth = torch.zeros_like(edges)
        if edges is None:
            edges = torch.zeros_like(depth)

        # Edge post-process: width/single-line/smooth
        def _edges_post(acc_t: torch.Tensor) -> torch.Tensor:
            try:
                import cv2, numpy as _np
                acc = acc_t.detach().to('cpu').numpy()
                img = (acc*255.0).astype(_np.uint8)
                k = _np.ones((3,3), _np.uint8)
                # Adjust thickness
                w = float(edge_width)
                if abs(w) > 1e-6:
                    it = int(abs(w))
                    frac = abs(w) - it
                    op = cv2.dilate if w > 0 else cv2.erode
                    y = img.copy()
                    for _ in range(max(0, it)):
                        y = op(y, k, iterations=1)
                    if frac > 1e-6:
                        y2 = op(y, k, iterations=1)
                        y = ((1.0-frac)*y.astype(_np.float32) + frac*y2.astype(_np.float32)).astype(_np.uint8)
                    img = y
                # Collapse double lines to single centerline
                if bool(edge_single_line) and float(edge_single_strength) > 1e-6:
                    try:
                        s = float(edge_single_strength)
                        close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=1)
                        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
                            sk = cv2.ximgproc.thinning(close)
                        else:
                            # limited-iteration morphological skeletonization
                            iters = max(1, int(round(2 + 6*s)))
                            sk = _np.zeros_like(close)
                            src = close.copy()
                            elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                            for _ in range(iters):
                                er = cv2.erode(src, elem, iterations=1)
                                op = cv2.morphologyEx(er, cv2.MORPH_OPEN, elem)
                                tmp = cv2.subtract(er, op)
                                sk = cv2.bitwise_or(sk, tmp)
                                src = er
                                if not _np.any(src):
                                    break
                        # Blend skeleton back with original according to strength
                        img = ((_np.float32(1.0 - s) * img.astype(_np.float32)) + (_np.float32(s) * sk.astype(_np.float32))).astype(_np.uint8)
                    except Exception:
                        pass
                # Smooth
                if float(edge_smooth) > 1e-6:
                    sigma = max(0.1, min(2.0, float(edge_smooth) * 1.2))
                    img = cv2.GaussianBlur(img, (0,0), sigmaX=sigma)
                out = torch.from_numpy((img.astype(_np.float32)/255.0)).to(device=acc_t.device, dtype=acc_t.dtype)
                return out.clamp(0,1)
            except Exception:
                # Torch fallback: light blur-only
                if float(edge_smooth) > 1e-6:
                    s = max(1, int(round(float(edge_smooth)*2)))
                    return F.avg_pool2d(acc_t.unsqueeze(0).unsqueeze(0), kernel_size=2*s+1, stride=1, padding=s)[0,0].clamp(0,1)
                return acc_t

        edges = _edges_post(edges)

        # Depth gating of edges
        if bool(edge_depth_gate):
            # Inverted gating per feedback: use depth^gamma (nearer = stronger if depth is larger)
            g = (depth.clamp(0,1)) ** float(edge_depth_gamma)
            edges = (edges * g).clamp(0,1)

        # Apply edge alpha before blending
        edges = (edges * float(edge_alpha)).clamp(0,1)

        fused = _blend(depth, edges, str(blend_mode), float(blend_factor))
        # Apply as split (Edges then Depth) or single fused hint
        if bool(split_apply):
            # Fixed order for determinism: Depth first, then Edges
            hint_edges = edges.unsqueeze(-1).repeat(1,1,1,3)
            hint_depth = depth.unsqueeze(-1).repeat(1,1,1,3)
            # Depth first
            pos_mid, neg_mid = _apply_controlnet_separate(
                positive, negative, control_net, hint_depth,
                float(strength_pos) * float(depth_strength_mul),
                float(strength_neg) * float(depth_strength_mul),
                float(depth_start_percent), float(depth_end_percent), vae,
                bool(apply_to_uncond), True
            )
            # Then edges
            pos_out, neg_out = _apply_controlnet_separate(
                pos_mid, neg_mid, control_net, hint_edges,
                float(strength_pos) * float(edge_strength_mul),
                float(strength_neg) * float(edge_strength_mul),
                float(edge_start_percent), float(edge_end_percent), vae,
                bool(apply_to_uncond), True
            )
        else:
            hint = fused.unsqueeze(-1).repeat(1,1,1,3)
            pos_out, neg_out = _apply_controlnet_separate(
                positive, negative, control_net, hint,
                float(strength_pos), float(strength_neg),
                float(start_percent), float(end_percent), vae,
                bool(apply_to_uncond), bool(stack_prev_control)
            )
        # Build preview: keep aspect ratio, set minimal side
        prev_res = int(max(256, min(2048, preview_res)))
        scale = prev_res / float(min(H, W))
        out_h = max(1, int(round(H * scale)))
        out_w = max(1, int(round(W * scale)))
        prev = F.interpolate(fused.unsqueeze(0).unsqueeze(0), size=(out_h, out_w), mode='bilinear', align_corners=False)[0,0]
        # Optionally reflect ControlNet strength in preview (display only)
        if bool(preview_show_strength):
            br = str(preview_strength_branch)
            sp = float(strength_pos)
            sn = float(strength_neg)
            if br == 'negative':
                s_vis = sn
            elif br == 'max':
                s_vis = max(sp, sn)
            elif br == 'avg':
                s_vis = 0.5 * (sp + sn)
            else:
                s_vis = sp
            # clamp for display range
            s_vis = max(0.0, min(1.0, s_vis))
            prev = prev * s_vis
        # Apply visualization brightness only for preview
        prev = (prev * float(mask_brightness)).clamp(0.0, 1.0)
        prev = prev.unsqueeze(-1).repeat(1,1,3).to(device=dev, dtype=dtype).unsqueeze(0)
        return (pos_out, neg_out, prev)
