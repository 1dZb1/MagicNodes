from comfy.ldm.modules import attention as comfy_attention
import logging
import comfy.model_patcher
import comfy.utils
import comfy.sd
import torch
import comfy.model_management as mm
from comfy.cli_args import args

sageattn_modes = [
    "disabled",
    "auto",
    "auto_speed",
    "auto_quality",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
]

_initialized = False
# Avoid spamming logs each attention call
_sage_warned_once = False
_sage_generic_warned_once = False
_original_functions = {}

# Runtime override knobs (may be set by other nodes, e.g., CADE2 Beta)
# CURRENT_PV_ACCUM can be None, "fp32+fp16" or "fp32+fp32"
CURRENT_PV_ACCUM = None

# Lightweight attention-entropy probe (for AQClip Attn-mode)
_attn_entropy_enabled = False
_attn_entropy_last = None  # torch.Tensor | None, shape (B,1,h',w') in [0,1]
_attn_probe_heads_cap = 4
_attn_probe_tokens_cap = 1024

def enable_attention_entropy_capture(enable: bool, max_tokens: int = 1024, max_heads: int = 4):
    """Toggle capturing a tiny attention entropy map during optimized_attention.
    Stores a normalized map per forward pass; consumer may upsample to latent size.
    """
    global _attn_entropy_enabled, _attn_probe_tokens_cap, _attn_probe_heads_cap, _attn_entropy_last
    _attn_entropy_enabled = bool(enable)
    _attn_probe_tokens_cap = int(max(128, min(16384, max_tokens)))
    _attn_probe_heads_cap = int(max(1, min(32, max_heads)))
    if not _attn_entropy_enabled:
        _attn_entropy_last = None

def get_attention_entropy_map(clear: bool = False):
    """Return last captured attention entropy map (B,1,h',w') in [0,1] or None."""
    global _attn_entropy_last
    out = _attn_entropy_last
    if clear:
        _attn_entropy_last = None
    return out

# ------------------------ KV pruning (self-attention) ------------------------
_kv_prune_enabled = False
_kv_prune_keep = 0.85
_kv_prune_min_tokens = 128

def set_kv_prune(enable: bool, keep: float = 0.85, min_tokens: int = 128):
    """Enable lightweight K/V token pruning inside optimized attention.
    - Applies only to self-attention (len(Q)==len(K)).
    - Keeps top-`keep` fraction of keys/values by L2 energy of K, averaged over heads.
    - Skips pruning when an attention mask is provided (shape mismatch risk).
    """
    global _kv_prune_enabled, _kv_prune_keep, _kv_prune_min_tokens
    _kv_prune_enabled = bool(enable)
    try:
        _kv_prune_keep = float(max(0.5, min(1.0, keep)))
    except Exception:
        _kv_prune_keep = 0.85
    try:
        _kv_prune_min_tokens = int(max(1, min_tokens))
    except Exception:
        _kv_prune_min_tokens = 128

if not _initialized:
    _original_functions["orig_attention"] = comfy_attention.optimized_attention
    _original_functions["original_patch_model"] = comfy.model_patcher.ModelPatcher.patch_model
    _original_functions["original_load_lora_for_models"] = comfy.sd.load_lora_for_models
    _initialized = True

class MGSagpuBaseLoader:
    original_linear = None
    cublas_patched = False

    @torch.compiler.disable()
    def _patch_modules(self, patch_cublaslinear, sage_attention):
        from comfy.ops import disable_weight_init, CastWeightBiasOp, cast_bias_weight

        if sage_attention != "disabled":
            print("Patching comfy attention to use sageattn")
            try:
                from sageattention import sageattn
                from sageattention import (
                    sageattn_qk_int8_pv_fp16_cuda,
                    sageattn_qk_int8_pv_fp16_triton,
                    sageattn_qk_int8_pv_fp8_cuda,
                    sageattn_qk_int8_pv_fp8_cuda_sm90,
                )
            except ImportError:
                from SageAttention import sageattn
                from SageAttention import (
                    sageattn_qk_int8_pv_fp16_cuda,
                    sageattn_qk_int8_pv_fp16_triton,
                    sageattn_qk_int8_pv_fp8_cuda,
                    sageattn_qk_int8_pv_fp8_cuda_sm90,
                )
            def set_sage_func(sage_attention):
                # Helper: pick best kernel for current GPU
                def select_auto(quality: bool):
                    def _auto(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device()) if torch.cuda.is_available() else (0, 0)
                        try:
                            if major == 12 and minor == 0:
                                # RTX 50 series
                                pv = "fp32+fp32" if quality else "fp32+fp16"
                                return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype=pv, tensor_layout=tensor_layout)
                            elif major == 9:
                                # H100 family
                                pv = "fp32+fp32" if quality else "fp32+fp32"
                                return sageattn_qk_int8_pv_fp8_cuda_sm90(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype=pv, tensor_layout=tensor_layout)
                            elif major == 8 and minor == 9:
                                pv = "fp32+fp32" if quality else "fp32+fp16"
                                return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype=pv, tensor_layout=tensor_layout)
                            elif major == 8 and minor in (0, 6):
                                # Ampere
                                # Prefer CUDA kernel when possible
                                return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
                        except Exception:
                            pass
                        # Generic auto (library decides), works across arch when available
                        return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    return _auto
                if sage_attention == "auto":
                    return select_auto(quality=False)
                if sage_attention == "auto_speed":
                    return select_auto(quality=False)
                if sage_attention == "auto_quality":
                    return select_auto(quality=True)
                elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
                    # using imported sageattn_qk_int8_pv_fp8_cuda above (name alias consistent for both module names)
                    # This variant requires SM89 (Ada 8.9). On newer GPUs (e.g., SM90),
                    # fall back to generic auto selection to avoid kernel assertion.
                    try:
                        if torch.cuda.is_available():
                            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
                            if not (major == 8 and minor == 9):
                                logging.warning(f"sageattn_qk_int8_pv_fp8_cuda++ requires SM89, but detected SM{major}{minor}. Falling back to auto kernel selection.")
                                def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                                    return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                                return func
                    except Exception:
                        pass
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
                    return func

            sage_func = set_sage_func(sage_attention)

            @torch.compiler.disable()
            def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, transformer_options=None, **kwargs):
                if skip_reshape:
                    b, _, _, dim_head = q.shape
                    tensor_layout="HND"
                else:
                    b, _, dim_head = q.shape
                    dim_head //= heads
                    q, k, v = map(
                        lambda t: t.view(b, -1, heads, dim_head),
                        (q, k, v),
                    )
                    tensor_layout="NHD"
                if mask is not None:
                    # add a batch dimension if there isn't already one
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    # add a heads dimension if there isn't already one
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(1)
                # Prefer trying sage kernels; allow runtime overrides via transformer_options or CURRENT_PV_ACCUM

                # Optional K/V pruning for self-attention (token-level top-k)
                try:
                    if _kv_prune_enabled and (mask is None):
                        import math
                        if tensor_layout == "NHD":
                            # q,k,v: B,N,H,D
                            Bn, Nq, Hn, Dh = q.shape
                            Nk = k.shape[1]
                            if Nq == Nk and Nk >= _kv_prune_min_tokens:
                                keep = max(1, int(math.ceil(float(_kv_prune_keep) * Nk)))
                                if keep < Nk:
                                    # importance: mean over heads of L2 norm of K per token
                                    imp = (k.pow(2).sum(dim=-1)).mean(dim=2)  # B,N
                                    top = torch.topk(imp, k=keep, dim=1, largest=True, sorted=False).indices
                                    idx = top.unsqueeze(-1).unsqueeze(-1).expand(Bn, keep, Hn, Dh)
                                    k = torch.gather(k, dim=1, index=idx)
                                    v = torch.gather(v, dim=1, index=idx)
                        else:
                            # HND: q,k,v: B,H,N,D
                            Bb, Hn, Nq, Dh = q.shape
                            Nk = k.shape[2]
                            if Nq == Nk and Nk >= _kv_prune_min_tokens:
                                keep = max(1, int(math.ceil(float(_kv_prune_keep) * Nk)))
                                if keep < Nk:
                                    imp = (k.pow(2).sum(dim=-1)).mean(dim=1)  # B,N
                                    top = torch.topk(imp, k=keep, dim=1, largest=True, sorted=False).indices
                                    idx = top.unsqueeze(1).unsqueeze(-1).expand(Bb, Hn, keep, Dh)
                                    k = torch.gather(k, dim=2, index=idx)
                                    v = torch.gather(v, dim=2, index=idx)
                except Exception:
                    # On any issue, skip pruning silently
                    pass

                try:
                    pv_override = None
                    if transformer_options and isinstance(transformer_options, dict):
                        so = transformer_options.get("sageattn")
                        if isinstance(so, dict):
                            pv_override = so.get("pv_accum_dtype", None)
                    if pv_override is None:
                        pv_override = CURRENT_PV_ACCUM

                    if pv_override is not None:
                        out = sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout, pv_accum_dtype=pv_override)
                    else:
                        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
                except Exception as e:
                    global _sage_generic_warned_once
                    if not _sage_generic_warned_once:
                        logging.warning(f"Error running sage attention: {e}. Falling back.")
                        _sage_generic_warned_once = True
                    try:
                        out = sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
                    except Exception:
                        # Final fallback to PyTorch attention, silent after first warning
                        if tensor_layout == "NHD":
                            q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
                        return comfy_attention.attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=True, skip_output_reshape=skip_output_reshape, transformer_options=transformer_options, **kwargs)
                # Optional tiny attention-entropy probe (avoid heavy compute)
                try:
                    if _attn_entropy_enabled:
                        import torch
                        with torch.inference_mode():
                            if tensor_layout == "HND":
                                # q: B,H,N,D -> B,N,H,D for uniform handling
                                q_probe = q.transpose(1, 2)
                                k_probe = k.transpose(1, 2)
                            else:
                                q_probe = q
                                k_probe = k
                            B_, N_, H_, Dh = q_probe.shape
                            # Cap heads and tokens
                            h_cap = min(H_, _attn_probe_heads_cap)
                            step = max(1, N_ // _attn_probe_tokens_cap)
                            q_s = q_probe[:, ::step, :h_cap, :].transpose(1, 2)  # B,h,q,d
                            k_s = k_probe[:, ::step, :h_cap, :].transpose(1, 2)  # B,h,k,d
                            scale = (float(Dh) ** -0.5)
                            # logits: B,h,q,k
                            logits = torch.matmul(q_s * scale, k_s.transpose(-1, -2))
                            p = torch.softmax(logits, dim=-1)
                            # entropy per query
                            eps = 1e-9
                            Hq = -(p * (p.clamp_min(eps).log())).sum(dim=-1)  # B,h,q
                            Hq = Hq.mean(dim=1)  # B,q
                            # reshape to approx grid
                            import math
                            Q = Hq.shape[-1]
                            w = int(math.sqrt(Q))
                            w = max(1, w)
                            h = max(1, Q // w)
                            if h * w > Q:
                                Hq = Hq[..., : (h * w)]
                            elif h * w < Q:
                                # pad with last
                                pad = (h * w) - Q
                                if pad > 0:
                                    Hq = torch.cat([Hq, Hq[..., -1:].expand(B_, pad)], dim=-1)
                            Hmap = Hq.reshape(B_, 1, h, w)
                            # normalize per-sample to [0,1]
                            Hmin = Hmap.amin(dim=(2, 3), keepdim=True)
                            Hmax = Hmap.amax(dim=(2, 3), keepdim=True)
                            Hn = (Hmap - Hmin) / (Hmax - Hmin + 1e-6)
                            global _attn_entropy_last
                            _attn_entropy_last = Hn.detach()
                except Exception:
                    pass

                if tensor_layout == "HND":
                    if not skip_output_reshape:
                        out = (
                            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                        )
                else:
                    if skip_output_reshape:
                        out = out.transpose(1, 2)
                    else:
                        out = out.reshape(b, -1, heads * dim_head)
                return out

            comfy_attention.optimized_attention = attention_sage
            comfy.ldm.hunyuan_video.model.optimized_attention = attention_sage
            comfy.ldm.flux.math.optimized_attention = attention_sage
            comfy.ldm.genmo.joint_model.asymm_models_joint.optimized_attention = attention_sage
            comfy.ldm.cosmos.blocks.optimized_attention = attention_sage
            comfy.ldm.wan.model.optimized_attention = attention_sage

        else:
            print("Restoring initial comfy attention")
            comfy_attention.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.hunyuan_video.model.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.flux.math.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.genmo.joint_model.asymm_models_joint.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.cosmos.blocks.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.wan.model.optimized_attention = _original_functions.get("orig_attention")

        if patch_cublaslinear:
            if not MGSagpuBaseLoader.cublas_patched:
                MGSagpuBaseLoader.original_linear = disable_weight_init.Linear
                try:
                    from cublas_ops import CublasLinear
                except ImportError:
                    raise Exception("Can't import 'torch-cublas-hgemm', install it from here https://github.com/aredden/torch-cublas-hgemm")

                class PatchedLinear(CublasLinear, CastWeightBiasOp):
                    def reset_parameters(self):
                        pass

                    def forward_comfy_cast_weights(self, input):
                        weight, bias = cast_bias_weight(self, input)
                        return torch.nn.functional.linear(input, weight, bias)

                    def forward(self, *args, **kwargs):
                        if self.comfy_cast_weights:
                            return self.forward_comfy_cast_weights(*args, **kwargs)
                        else:
                            return super().forward(*args, **kwargs)

                disable_weight_init.Linear = PatchedLinear
                MGSagpuBaseLoader.cublas_patched = True
        else:
            if MGSagpuBaseLoader.cublas_patched:
                disable_weight_init.Linear = MGSagpuBaseLoader.original_linear
                MGSagpuBaseLoader.cublas_patched = False

from comfy.patcher_extension import CallbacksMP
class MGSagpuAttention(MGSagpuBaseLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."}),
        }}

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"
    DESCRIPTION = "Node for patching attention mode. This doesn't use the model patching system and thus can't be disabled without running the node again with 'disabled' option."
    EXPERIMENTAL = False
    CATEGORY = "MagicNodes"

    def patch(self, model, sage_attention):
        model_clone = model.clone()
        @torch.compiler.disable()
        def patch_attention_enable(model):
            self._patch_modules(False, sage_attention)
        @torch.compiler.disable()
        def patch_attention_disable(model):
            self._patch_modules(False, "disabled")
        
        model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_attention_enable)
        model_clone.add_callback(CallbacksMP.ON_CLEANUP, patch_attention_disable)
        
        return model_clone,
 


# Legacy compile helpers removed

# Legacy video helpers removed
import inspect as _inspect
try:
    from comfy.ldm.modules import attention as _cm_attn
except Exception as _e:
    _cm_attn = None

_nag_patch_active = False
_nag_params = {"scale": 5.0, "tau": 2.5, "alpha": 0.25}
_original_functions.setdefault("orig_crossattn_forward", None)
_original_functions.setdefault("orig_crossattn_sig", None)

def _call_orig_crossattn(self, x, context=None, **kwargs):
    #\"\"\"Call the original CrossAttention.forward with kwargs filtered to its signature.\"\"\"
    f = _original_functions.get("orig_crossattn_forward", None)
    if f is None:
        # Should not happen; just try current method
        return self.__class__.forward(self, x, context=context, **kwargs)
    sig = _original_functions.get("orig_crossattn_sig", None)
    if sig is None:
        try:
            sig = _inspect.signature(f)
            _original_functions["orig_crossattn_sig"] = sig
        except Exception:
            sig = None
    if sig is not None:
        allowed = set(sig.parameters.keys())
        fkwargs = {k: v for k, v in kwargs.items() if k in allowed}
    else:
        fkwargs = kwargs
    try:
        return f(self, x, context=context, **fkwargs)
    except TypeError:
        # Some builds have (x, context=None, value=None, mask=None) only
        fkwargs.pop("attn_precision", None)
        fkwargs.pop("transformer_options", None)
        try:
            return f(self, x, context=context, **fkwargs)
        except Exception:
            # Give up; call current method (unpatched) to avoid crashing
            return self.__class__.forward(self, x, context=context, **kwargs)

def _kj_crossattn_forward_nag(self, x, context=None, value=None, mask=None, **kwargs):
    # If patch not active or context not having cond/uncond, defer to original.
    if (not _nag_patch_active) or (_cm_attn is None):
        return _call_orig_crossattn(self, x, context=context, value=value, mask=mask, **kwargs)
    try:
        if context is None or not torch.is_tensor(context):
            return _call_orig_crossattn(self, x, context=context, value=value, mask=mask, **kwargs)

        # Expect batch 2 with [uncond, cond]; if not, fall back
        if context.shape[0] < 2:
            return _call_orig_crossattn(self, x, context=context, value=value, mask=mask, **kwargs)

        # Split branches. In most samplers order is [uncond, cond].
        # If x has batch==2, split it likewise; else use the same x for both calls.
        x_has_pair = (torch.is_tensor(x) and x.shape[0] == 2)
        x_u = x[0:1] if x_has_pair else x
        x_c = x[1:2] if x_has_pair else x

        c_u, c_c = context[0:1], context[1:2]

        # value may also be batched
        v = kwargs.get("value", value)
        if torch.is_tensor(v) and v.shape[0] == 2:
            v_u, v_c = v[0:1], v[1:2]
        else:
            v_u = v_c = v

        # Get per-branch outputs using the ORIGINAL forward
        # - Neg branch (for real uncond stream)
        out_u = _call_orig_crossattn(self, x_u, context=c_u, value=v_u, mask=mask, **kwargs)
        # - Pos branch
        z_pos = _call_orig_crossattn(self, x_c, context=c_c, value=v_c, mask=mask, **kwargs)
        # - "Neg guidance" term computed with *positive query but negative context*
        z_neg = _call_orig_crossattn(self, x_c, context=c_u, value=v_u, mask=mask, **kwargs)

        # NAG mixing in the attention output space
        phi = float(_nag_params.get("scale", 5.0))
        tau = float(_nag_params.get("tau", 2.5))
        alpha = float(_nag_params.get("alpha", 0.25))

        g = z_pos * phi - z_neg * (phi - 1.0)
        # L1-norm based clipping to limit deviation from Z+
        def _l1_norm(t):
            return torch.sum(torch.abs(t), dim=-1, keepdim=True).clamp_min(1e-6)
        s_pos = _l1_norm(z_pos)
        s_g   = _l1_norm(g)
        scale = (s_pos * tau) / s_g
        g = torch.where((s_g > s_pos * tau), g * scale, g)

        z_guided = g * alpha + z_pos * (1.0 - alpha)
        if x_has_pair:
            return torch.cat([out_u, z_guided], dim=0)
        else:
            return z_guided
    except Exception as e:
        # If anything goes wrong, use the original forward.
        return _call_orig_crossattn(self, x, context=context, value=value, mask=mask, **kwargs)

def enable_crossattention_nag_patch(enable: bool, nag_scale: float = 5.0, nag_tau: float = 2.5, nag_alpha: float = 0.25):
    #\"\"\"Enable/disable a safe CrossAttention forward wrapper that applies NAG to the positive branch only.
    #This does not modify model weights and is fully reversible. The wrapper preserves
    #unknown kwargs (filters per-signature) to avoid errors on older Comfy builds.
    #\"\"\"
    global _nag_patch_active, _nag_params
    if _cm_attn is None:
        return False
    if enable:
        _nag_params = {"scale": float(nag_scale), "tau": float(nag_tau), "alpha": float(nag_alpha)}
        if _original_functions.get("orig_crossattn_forward", None) is None:
            try:
                _original_functions["orig_crossattn_forward"] = _cm_attn.CrossAttention.forward
                try:
                    _original_functions["orig_crossattn_sig"] = _inspect.signature(_cm_attn.CrossAttention.forward)
                except Exception:
                    _original_functions["orig_crossattn_sig"] = None
            except Exception:
                return False
        # Patch in our wrapper
        try:
            _cm_attn.CrossAttention.forward = _kj_crossattn_forward_nag
            _nag_patch_active = True
            return True
        except Exception:
            return False
    else:
        # Restore original if we have it
        if _original_functions.get("orig_crossattn_forward", None) is not None:
            try:
                _cm_attn.CrossAttention.forward = _original_functions["orig_crossattn_forward"]
            except Exception:
                pass
        _nag_patch_active = False
        return True
# ===============================================================================

PatchSageAttention = MGSagpuAttention







