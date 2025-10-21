import os
from typing import Dict, Tuple

_CACHE: Dict[str, Tuple[float, Dict[str, Dict[str, object]]]] = {}

_MSG_PREFIX = "[MagicNodes][Presets]"

def _root_dir() -> str:
    # .../MagicNodes/mod/easy -> .../MagicNodes
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def _pressets_dir() -> str:
    return os.path.join(_root_dir(), "pressets")

def _cfg_path(kind: str) -> str:
    # kind examples: "mg_cade25", "mg_controlfusion"
    return os.path.join(_pressets_dir(), f"{kind}.cfg")

def _parse_value(raw: str):
    s = raw.strip()
    if not s:
        return ""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except Exception:
        pass
    # variable substitution
    s = s.replace("$(ROOT)", _root_dir())
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s

def _load_kind(kind: str) -> Dict[str, Dict[str, object]]:
    path = _cfg_path(kind)
    if not os.path.isfile(path):
        print(f"{_MSG_PREFIX} No configuration file for '{kind}' found; loaded defaults — results may be unpredictable!")
        return {}
    try:
        mtime = os.path.getmtime(path)
        cached = _CACHE.get(path)
        if cached and cached[0] == mtime:
            return cached[1]

        data: Dict[str, Dict[str, object]] = {}
        cur_section = None
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    cur_section = line[1:-1].strip().lower()
                    data.setdefault(cur_section, {})
                    continue
                if ":" in line:
                    if cur_section is None:
                        print(f"{_MSG_PREFIX} Parse warning at line {ln}: key outside of any [section]; ignored")
                        continue
                    k, v = line.split(":", 1)
                    key = k.strip()
                    try:
                        val = _parse_value(v)
                    except Exception:
                        print(f"{_MSG_PREFIX} Missing or invalid parameter '{key}'; this may affect results!")
                        continue
                    data[cur_section][key] = val
                else:
                    print(f"{_MSG_PREFIX} Unknown line at {ln}: '{line}'; ignored")

        _CACHE[path] = (mtime, data)
        return data
    except Exception as e:
        print(f"{_MSG_PREFIX} Failed to read '{path}': {e}. Loaded defaults — results may be unpredictable!")
        return {}

def get(kind: str, step: str) -> Dict[str, object]:
    """Return dict of parameters for a given kind and step.
    step accepts 'Step 1', '1', 'step1', case-insensitive.
    """
    data = _load_kind(kind)
    if not data:
        return {}
    label = step.strip().lower().replace(" ", "")
    if label.startswith("step"):
        key = label
    elif label.isdigit():
        key = f"step{label}"
    else:
        key = f"step{label}"

    if key not in data:
        # Special case: CF is intentionally not applied on Step 1 in this pipeline.
        # Suppress noisy log for missing 'Step 1' in mg_controlfusion.
        if kind == "mg_controlfusion" and key in ("step1", "1"):
            return {}
        print(f"{_MSG_PREFIX} Preset step '{step}' not found for '{kind}'; using defaults")
        return {}
    res = dict(data[key])
    # Side-effect: when CADE presets are loaded, optionally enable KV pruning in attention
    try:
        if kind == "mg_cade25":
            from .. import mg_sagpu_attention as sa_patch  # local import to avoid cycles
            kv_enable = bool(res.get("kv_prune_enable", False))
            kv_keep = float(res.get("kv_keep", 0.85))
            kv_min = int(res.get("kv_min_tokens", 128)) if "kv_min_tokens" in res else 128
            if hasattr(sa_patch, "set_kv_prune"):
                sa_patch.set_kv_prune(kv_enable, kv_keep, kv_min)
    except Exception:
        pass
    return res
