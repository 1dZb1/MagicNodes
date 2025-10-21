import comfy.sd
import comfy.clip_vision
import folder_paths
import comfy.utils
import torch
import random
from datetime import datetime
import random
import gc
import os
import json
import re

from .hard.mg_upscale_module import clear_gpu_and_ram_cache

# Module level caches to reuse loaded models and LoRAs between invocations
_checkpoint_cache = {}
_loaded_checkpoint = None
_lora_cache = {}
_active_lora_names = set()


def _clear_unused_loras(active_names):
    """Remove unused LoRAs from cache and clear GPU memory."""
    unused = [n for n in _lora_cache if n not in active_names]
    for n in unused:
        del _lora_cache[n]
    if unused:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_checkpoint(path):
    """Load checkpoint from cache or disk."""
    if path in _checkpoint_cache:
        return _checkpoint_cache[path]
    model, clip, vae = comfy.sd.load_checkpoint_guess_config(
        path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )[:3]
    _checkpoint_cache[path] = (model, clip, vae)
    return model, clip, vae


def _unload_old_checkpoint(current_path):
    """Unload checkpoint if it's different from the current one."""
    global _loaded_checkpoint
    if _loaded_checkpoint and _loaded_checkpoint != current_path:
        _checkpoint_cache.pop(_loaded_checkpoint, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _loaded_checkpoint = current_path



class MagicNodesCombiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            


            # --- Checkpoint ---
            "use_checkpoint": ("BOOLEAN", {"default": True}),
            "checkpoint": (folder_paths.get_filename_list("checkpoints"), {}),
            "clear_cache": ("BOOLEAN", {"default": False}),

            # --- LoRA 1 ---
            "use_lora_1": ("BOOLEAN", {"default": True}),
            "lora_1": (folder_paths.get_filename_list("loras"), {}),
            "strength_model_1": ("FLOAT", {"default": 1.0, "min": -1.5, "max": 1.5, "step": 0.01,}),
            "strength_clip_1": ("FLOAT", {"default": 1.0, "min": -1.5, "max": 1.5, "step": 0.01,}),

            # --- LoRA 2 ---
            "use_lora_2": ("BOOLEAN", {"default": False}),
            "lora_2": (folder_paths.get_filename_list("loras"), {}),
            "strength_model_2": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),
            "strength_clip_2": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),

            # --- LoRA 3 ---
            "use_lora_3": ("BOOLEAN", {"default": False}),
            "lora_3": (folder_paths.get_filename_list("loras"), {}),
            "strength_model_3": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),
            "strength_clip_3": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),

            # --- LoRA 4 ---
            "use_lora_4": ("BOOLEAN", {"default": False}),
            "lora_4": (folder_paths.get_filename_list("loras"), {}),
            "strength_model_4": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),
            "strength_clip_4": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),

            # --- LoRA 5 ---
            "use_lora_5": ("BOOLEAN", {"default": False}),
            "lora_5": (folder_paths.get_filename_list("loras"), {}),
            "strength_model_5": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),
            "strength_clip_5": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),

            # --- LoRA 6 ---
            "use_lora_6": ("BOOLEAN", {"default": False}),
            "lora_6": (folder_paths.get_filename_list("loras"), {}),
            "strength_model_6": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),
            "strength_clip_6": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,}),
            },
            "optional": {
            "model_in": ("MODEL", {}),
            "clip_in": ("CLIP", {}),
            "vae_in": ("VAE", {}),

            # --- Prompts --- (controlled dynamic expansion inside node for determinism)
            "positive_prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
            "negative_prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),

            # Optional external conditioning (bypass internal text encode)
            "positive_in": ("CONDITIONING", {}),
            "negative_in": ("CONDITIONING", {}),

            # --- CLIP Layers ---
            "clip_set_last_layer_positive": ("INT", {"default": -2, "min": -20, "max": 0}),
            "clip_set_last_layer_negative": ("INT", {"default": -2, "min": -20, "max": 0}),

            # --- Recipes ---
            "recipe_slot": (["Off", "Slot 1", "Slot 2", "Slot 3", "Slot 4"], {"default": "Off", "tooltip": "Choose slot to save/load assembled setup."}),
            "recipe_save": ("BOOLEAN", {"default": False, "tooltip": "Save current setup into the selected slot."}),
            "recipe_use": ("BOOLEAN", {"default": False, "tooltip": "Load and apply setup from the selected slot for this run."}),

            # --- Standard pipeline (match classic node order for CLIP) ---
            "standard_pipeline": ("BOOLEAN", {"default": False, "tooltip": "Use vanilla order for CLIP: Set Last Layer -> Load LoRA -> Encode (same CLIP logic as standard ComfyUI)."}),

            # CLIP LoRA gains per branch (effective only when standard_pipeline=true)
            "clip_lora_pos_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Multiplier for CLIP-LoRA strength on positive branch (standard pipeline)."}),
            "clip_lora_neg_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Multiplier for CLIP-LoRA strength on negative branch (standard pipeline)."}),

            # Deterministic dynamic prompts
            "dynamic_pos": ("BOOLEAN", {"default": False, "tooltip": "Deterministically expand choices in positive prompt (uses dyn_seed)."}),
            "dynamic_neg": ("BOOLEAN", {"default": False, "tooltip": "Deterministically expand choices in negative prompt (uses dyn_seed)."}),
            "dyn_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Seed for dynamic prompt expansion (same seed used for both prompts)."}),
            "dynamic_break_freeze": ("BOOLEAN", {"default": True, "tooltip": "If enabled, do not expand choices before the first |BREAK| marker; dynamic applies only after it."}),
            "show_expanded_prompts": ("BOOLEAN", {"default": False, "tooltip": "Print expanded Positive/Negative prompts to console when dynamic is enabled."}),
            "save_expanded_prompts": ("BOOLEAN", {"default": False, "tooltip": "Save expanded prompts to mod/dynPrompt/SEED_dd_mm_yyyy.txt when dynamic is enabled."}),
            }
        }


    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "Positive", "Negative", "VAE")
    #RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "VAE")
    #RETURN_NAMES = ("MODEL", "Positive", "Negative", "VAE")
    FUNCTION = "apply_magic_node"
    CATEGORY = "MagicNodes"

    def apply_magic_node(self, model_in=None, clip_in=None, checkpoint=None,
                     use_checkpoint=True, clear_cache=False,
                     use_lora_1=True, lora_1=None, strength_model_1=1.0, strength_clip_1=1.0,
                     use_lora_2=False, lora_2=None, strength_model_2=0.0, strength_clip_2=0.0,
                     use_lora_3=False, lora_3=None, strength_model_3=0.0, strength_clip_3=0.0,
                     use_lora_4=False, lora_4=None, strength_model_4=0.0, strength_clip_4=0.0,
                     use_lora_5=False, lora_5=None, strength_model_5=0.0, strength_clip_5=0.0,
                     use_lora_6=False, lora_6=None, strength_model_6=0.0, strength_clip_6=0.0,
                     positive_prompt="", negative_prompt="",
                     clip_set_last_layer_positive=-2, clip_set_last_layer_negative=-2,
                     vae_in=None,
                     recipe_slot="Off", recipe_save=False, recipe_use=False,
                     standard_pipeline=False,
                     clip_lora_pos_gain=1.0, clip_lora_neg_gain=1.0,
                     positive_in=None, negative_in=None,
                     dynamic_pos=False, dynamic_neg=False, dyn_seed=0, dynamic_break_freeze=True,
                     show_expanded_prompts=False, save_expanded_prompts=False):

        global _loaded_checkpoint

        # hard scrub of checkpoint cache each call (prevent hidden state)
        _checkpoint_cache.clear()
        if clear_cache:
            _lora_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Recipe helpers
        def _recipes_path():
            base = os.path.join(os.path.dirname(__file__), "state")
            os.makedirs(base, exist_ok=True)
            return os.path.join(base, "combinode_recipes.json")
        def _recipes_load():
            try:
                with open(_recipes_path(), "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        def _recipes_save(data: dict):
            try:
                with open(_recipes_path(), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # Apply recipe if requested
        slot_idx = {"Off": 0, "Slot 1": 1, "Slot 2": 2, "Slot 3": 3, "Slot 4": 4}.get(str(recipe_slot), 0)
        if slot_idx and bool(recipe_use):
            rec = _recipes_load().get(str(slot_idx), None)
            if rec is not None:
                try:
                    use_checkpoint = rec.get("use_checkpoint", use_checkpoint)
                    checkpoint = rec.get("checkpoint", checkpoint)
                    clip_set_last_layer_positive = rec.get("clip_pos", clip_set_last_layer_positive)
                    clip_set_last_layer_negative = rec.get("clip_neg", clip_set_last_layer_negative)
                    positive_prompt = rec.get("pos_text", positive_prompt)
                    negative_prompt = rec.get("neg_text", negative_prompt)
                    rls = rec.get("loras", [])
                    if len(rls) >= 4:
                        (use_lora_1, lora_1, strength_model_1, strength_clip_1) = rls[0]
                        (use_lora_2, lora_2, strength_model_2, strength_clip_2) = rls[1]
                        (use_lora_3, lora_3, strength_model_3, strength_clip_3) = rls[2]
                        (use_lora_4, lora_4, strength_model_4, strength_clip_4) = rls[3]
                    if len(rls) >= 5:
                        (use_lora_5, lora_5, strength_model_5, strength_clip_5) = rls[4]
                    if len(rls) >= 6:
                        (use_lora_6, lora_6, strength_model_6, strength_clip_6) = rls[5]
                    print(f"[CombiNode] Loaded recipe Slot {slot_idx}.")
                except Exception:
                    print(f"[CombiNode] Failed to apply recipe Slot {slot_idx}.")

        # Prompt normalization helper (keeps '|' intact)
        def _norm_prompt(s: str) -> str:
            if not isinstance(s, str) or not s:
                return s or ""
            s2 = s.replace("\r", " ").replace("\n", " ")
            s2 = re.sub(r"\s+", " ", s2)
            s2 = re.sub(r"\s*,\s*", ", ", s2)
            s2 = re.sub(r"(,\s*){2,}", ", ", s2)
            return s2.strip()

        # Deterministic dynamic prompt expansion: supports {...}, (...), [...] with '|' choices
        def _expand_dynamic(text: str, seed_val: int, freeze_before_break: bool = True) -> str:
            if not isinstance(text, str) or (text.find('|') < 0):
                return text
            # Honor |BREAK|: keep first segment intact when requested
            if freeze_before_break and ('|BREAK|' in text):
                pre, post = text.split('|BREAK|', 1)
                return pre + '|BREAK|' + _expand_dynamic(post, seed_val, freeze_before_break=False)
            rng = random.Random(int(seed_val) & 0xFFFFFFFF)
            def _expand_pattern(t: str, pat: re.Pattern) -> str:
                prev = None
                cur = t
                while prev != cur:
                    prev = cur
                    def repl(m):
                        body = m.group(1)
                        choices = [c.strip() for c in body.split('|') if c.strip()]
                        if not choices:
                            return m.group(0)
                        return rng.choice(choices)
                    cur = pat.sub(repl, cur)
                return cur
            for rx in (
                re.compile(r"\{([^{}]+)\}"),
                re.compile(r"\(([^()]+)\)"),
                re.compile(r"\[([^\[\]]+)\]"),
            ):
                text = _expand_pattern(text, rx)
            return text

        # Precompute expanded (or original) texts once
        pos_text_expanded = _norm_prompt(_expand_dynamic(positive_prompt, int(dyn_seed), bool(dynamic_break_freeze)) if bool(dynamic_pos) else positive_prompt)
        neg_text_expanded = _norm_prompt(_expand_dynamic(negative_prompt, int(dyn_seed), bool(dynamic_break_freeze)) if bool(dynamic_neg) else negative_prompt)

        if use_checkpoint and checkpoint:
            checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint)
            _unload_old_checkpoint(checkpoint_path)
            base_model, base_clip, vae = _load_checkpoint(checkpoint_path)
            model = base_model.clone()
            clip = base_clip.clone()
            clip_clean = base_clip.clone()  # keep pristine CLIP for standard pipeline path

        elif model_in and clip_in:
            _unload_old_checkpoint(None)
            model = model_in.clone()
            clip = clip_in.clone()
            clip_clean = clip_in.clone()
            vae = vae_in
        else:
            raise Exception("No model selected!")

        # single clear at start is enough; avoid double-clearing here

        # Применение цепочки LoRA
        loras = [
            (use_lora_1, lora_1, strength_model_1, strength_clip_1),
            (use_lora_2, lora_2, strength_model_2, strength_clip_2),
            (use_lora_3, lora_3, strength_model_3, strength_clip_3),
            (use_lora_4, lora_4, strength_model_4, strength_clip_4),
            (use_lora_5, lora_5, strength_model_5, strength_clip_5),
            (use_lora_6, lora_6, strength_model_6, strength_clip_6),
        ]

        active_lora_paths = []
        lora_stack = []  # list of (lora_file, sc, sm)
        defer_clip = bool(standard_pipeline)
        for use_lora, lora_name, sm, sc in loras:
            if use_lora and lora_name:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                active_lora_paths.append(lora_path)
                # keep lora object to avoid reloading
                if lora_path in _lora_cache:
                    lora_file = _lora_cache[lora_path]
                else:
                    lora_file = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    _lora_cache[lora_path] = lora_file
                lora_stack.append((lora_file, float(sc), float(sm)))
                sc_apply = 0.0 if defer_clip else sc
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora_file, sm, sc_apply)

        _clear_unused_loras(active_lora_paths)
        # Warn about duplicate LoRA selections across slots
        try:
            counts = {}
            for p in active_lora_paths:
                counts[p] = counts.get(p, 0) + 1
            dups = [p for p, c in counts.items() if c > 1]
            if dups:
                print(f"[CombiNode] Duplicate LoRA detected across slots: {len(dups)} file(s).")
        except Exception:
            pass

        # Embeddings Positive и Negative
        # Standard pipeline: optionally use a shared CLIP after clip_layer + CLIP-LoRA
        # Select CLIP source for encoding: pristine when standard pipeline is enabled
        src_clip = clip_clean if bool(standard_pipeline) else clip

        pos_gain = float(clip_lora_pos_gain)
        neg_gain = float(clip_lora_neg_gain)
        skips_equal = int(clip_set_last_layer_positive) == int(clip_set_last_layer_negative)
        # Use shared CLIP only if gains are equal and skips equal
        use_shared = bool(standard_pipeline) and skips_equal and (abs(pos_gain - neg_gain) < 1e-6)

        if (positive_in is None) and (negative_in is None) and use_shared:
            shared_clip = src_clip.clone()
            shared_clip.clip_layer(clip_set_last_layer_positive)
            for lora_file, sc, sm in lora_stack:
                try:
                    _m_unused, shared_clip = comfy.sd.load_lora_for_models(model, shared_clip, lora_file, 0.0, sc * pos_gain)
                except Exception:
                    pass
            tokens_pos = shared_clip.tokenize(pos_text_expanded)
            cond_pos = shared_clip.encode_from_tokens_scheduled(tokens_pos)
            tokens_neg = shared_clip.tokenize(neg_text_expanded)
            cond_neg = shared_clip.encode_from_tokens_scheduled(tokens_neg)
        else:
            # CLIP Set Last Layer + Positive conditioning
            clip_pos = src_clip.clone()
            clip_pos.clip_layer(clip_set_last_layer_positive)
            if bool(standard_pipeline):
                for lora_file, sc, sm in lora_stack:
                    try:
                        _m_unused, clip_pos = comfy.sd.load_lora_for_models(model, clip_pos, lora_file, 0.0, sc * pos_gain)
                    except Exception:
                        pass
            if positive_in is not None:
                cond_pos = positive_in
            else:
                tokens_pos = clip_pos.tokenize(pos_text_expanded)
                cond_pos = clip_pos.encode_from_tokens_scheduled(tokens_pos)

            # CLIP Set Last Layer + Negative conditioning
            clip_neg = src_clip.clone()
            clip_neg.clip_layer(clip_set_last_layer_negative)
            if bool(standard_pipeline):
                for lora_file, sc, sm in lora_stack:
                    try:
                        _m_unused, clip_neg = comfy.sd.load_lora_for_models(model, clip_neg, lora_file, 0.0, sc * neg_gain)
                    except Exception:
                        pass
            if negative_in is not None:
                cond_neg = negative_in
            else:
                tokens_neg = clip_neg.tokenize(neg_text_expanded)
                cond_neg = clip_neg.encode_from_tokens_scheduled(tokens_neg)

        # Optional: show/save expanded prompts if dynamic used anywhere
        dyn_used = bool(dynamic_pos) or bool(dynamic_neg)
        if dyn_used and (bool(show_expanded_prompts) or bool(save_expanded_prompts)):
            # Console print
            if bool(show_expanded_prompts):
                try:
                    print(f"[CombiNode] Expanded prompts (dyn_seed={int(dyn_seed)}):")
                    def _print_block(name, src, expanded):
                        print(name + ":")
                        if bool(dynamic_break_freeze) and ('|BREAK|' in src) and ((name=="Positive" and bool(dynamic_pos)) or (name=="Negative" and bool(dynamic_neg))):
                            print("  static")
                        print("  " + expanded)
                    _print_block("Positive", positive_prompt, pos_text_expanded)
                    _print_block("Negative", negative_prompt, neg_text_expanded)
                except Exception:
                    pass
            # File save
            if bool(save_expanded_prompts):
                try:
                    base = os.path.join(os.path.dirname(__file__), "dynPrompt")
                    os.makedirs(base, exist_ok=True)
                    now = datetime.now()
                    fname = f"{int(dyn_seed)}_{now.day:02d}_{now.month:02d}_{now.year}.txt"
                    path = os.path.join(base, fname)
                    lines = []
                    def _append_block(name, src, expanded):
                        lines.append(name + ":\n")
                        if bool(dynamic_break_freeze) and ('|BREAK|' in src) and ((name=="Positive" and bool(dynamic_pos)) or (name=="Negative" and bool(dynamic_neg))):
                            lines.append("static\n")
                        lines.append(expanded + "\n\n")
                    _append_block("Positive", positive_prompt, pos_text_expanded)
                    _append_block("Negative", negative_prompt, neg_text_expanded)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                except Exception:
                    pass

        # Save recipe if requested
        if slot_idx and bool(recipe_save):
            data = _recipes_load()
            data[str(slot_idx)] = {
                "use_checkpoint": bool(use_checkpoint),
                "checkpoint": checkpoint,
                "clip_pos": int(clip_set_last_layer_positive),
                "clip_neg": int(clip_set_last_layer_negative),
                "pos_text": str(positive_prompt),
                "neg_text": str(negative_prompt),
                "loras": [
                    [bool(use_lora_1), lora_1, float(strength_model_1), float(strength_clip_1)],
                    [bool(use_lora_2), lora_2, float(strength_model_2), float(strength_clip_2)],
                    [bool(use_lora_3), lora_3, float(strength_model_3), float(strength_clip_3)],
                    [bool(use_lora_4), lora_4, float(strength_model_4), float(strength_clip_4)],
                    [bool(use_lora_5), lora_5, float(strength_model_5), float(strength_clip_5)],
                    [bool(use_lora_6), lora_6, float(strength_model_6), float(strength_clip_6)],
                ],
            }
            _recipes_save(data)
            print(f"[CombiNode] Saved recipe Slot {slot_idx}.")

        # Return the CLIP instance consistent with encoding path
        return (model, src_clip if bool(standard_pipeline) else clip, cond_pos, cond_neg, vae)




