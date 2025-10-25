import os
import gc
import time
import platform
import ctypes
from ctypes import wintypes
import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.sample as _sample
import comfy.samplers as _samplers
import comfy.utils as _utils

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


def _get_ram_mb() -> float:
    try:
        if psutil is not None:
            p = psutil.Process(os.getpid())
            rss = float(p.memory_info().rss)
            try:
                private = getattr(p.memory_full_info(), "private", None)
                if isinstance(private, (int, float)) and private > 0:
                    rss = float(private)
            except Exception:
                pass
            return rss / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _get_vram_mb_per_device() -> list[tuple[int, float, float]]:
    out = []
    try:
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                try:
                    reserved = float(torch.cuda.memory_reserved(d)) / (1024.0 * 1024.0)
                    allocated = float(torch.cuda.memory_allocated(d)) / (1024.0 * 1024.0)
                except Exception:
                    reserved = 0.0
                    allocated = 0.0
                out.append((d, reserved, allocated))
    except Exception:
        pass
    return out


def _trim_working_set_windows():
    try:
        if platform.system().lower().startswith("win"):
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            proc = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSize(proc, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except Exception:
        pass


def _enable_win_privileges(names):
    """Best-effort enable a set of Windows privileges for the current process."""
    try:
        if not platform.system().lower().startswith('win'):
            return False
        advapi32 = ctypes.windll.advapi32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        token = wintypes.HANDLE()
        TOKEN_ADJUST_PRIVILEGES = 0x20
        TOKEN_QUERY = 0x8
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ctypes.byref(token)):
            return False

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

        class LUID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

        class TOKEN_PRIVILEGES(ctypes.Structure):
            _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]

        SE_PRIVILEGE_ENABLED = 0x2
        success = False
        for name in names:
            luid = LUID()
            if not advapi32.LookupPrivilegeValueW(None, ctypes.c_wchar_p(name), ctypes.byref(luid)):
                continue
            tp = TOKEN_PRIVILEGES()
            tp.PrivilegeCount = 1
            tp.Privileges[0].Luid = luid
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
            if advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None):
                success = True
        return success
    except Exception:
        return False


def _system_cache_trim_windows():
    """Attempt to purge standby/file caches on Windows (requires privileges)."""
    try:
        if not platform.system().lower().startswith('win'):
            return False
        _enable_win_privileges([
            'SeIncreaseQuotaPrivilege',
            'SeProfileSingleProcessPrivilege',
            'SeDebugPrivilege',
        ])
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            SIZE_T = ctypes.c_size_t
            kernel32.SetSystemFileCacheSize(SIZE_T(-1), SIZE_T(-1), wintypes.DWORD(0))
        except Exception:
            pass
        try:
            ntdll = ctypes.windll.ntdll  # type: ignore[attr-defined]
            SystemMemoryListInformation = 0x50
            MemoryPurgeStandbyList = ctypes.c_ulong(4)
            ntdll.NtSetSystemInformation(wintypes.ULONG(SystemMemoryListInformation), ctypes.byref(MemoryPurgeStandbyList), ctypes.sizeof(MemoryPurgeStandbyList))
        except Exception:
            pass
        return True
    except Exception:
        return False


def cleanup_memory(sync_cuda: bool = True, hard_trim: bool = True) -> dict:
    """Run a best-effort cleanup of RAM/VRAM. Returns stats dict with before/after deltas."""
    stats: dict = {"ram_before_mb": 0.0, "ram_after_mb": 0.0, "ram_freed_mb": 0.0, "gpu": []}
    stats["ram_before_mb"] = _get_ram_mb()
    gpu_before = _get_vram_mb_per_device()
    try:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        import comfy.model_management as mm
        if hasattr(mm, 'soft_empty_cache'):
            mm.soft_empty_cache()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        time.sleep(0)
    except Exception:
        pass
    if hard_trim:
        try:
            import comfy.model_management as mm
            if hasattr(mm, 'unload_all_models'):
                mm.unload_all_models()
        except Exception:
            pass
        try:
            for _ in range(2):
                time.sleep(0)
                gc.collect()
        except Exception:
            pass
        try:
            if hasattr(_utils, 'cleanup_lru_caches'):
                _utils.cleanup_lru_caches()
        except Exception:
            pass
        try:
            _trim_working_set_windows()
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
        except Exception:
            pass
        try:
            if platform.system().lower().startswith('linux'):
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
        except Exception:
            pass
        try:
            _system_cache_trim_windows()
        except Exception:
            pass
    stats["ram_after_mb"] = _get_ram_mb()
    stats["ram_freed_mb"] = max(0.0, stats["ram_before_mb"] - stats["ram_after_mb"])
    gpu_after = _get_vram_mb_per_device()
    device_map = {d: (r, a) for d, r, a in gpu_before}
    gpu_stats = []
    for d, r_after, a_after in gpu_after:
        r_before, a_before = device_map.get(d, (0.0, 0.0))
        gpu_stats.append({
            "device": d,
            "reserved_before_mb": r_before,
            "reserved_after_mb": r_after,
            "reserved_freed_mb": max(0.0, r_before - r_after),
            "allocated_before_mb": a_before,
            "allocated_after_mb": a_after,
            "allocated_freed_mb": max(0.0, a_before - a_after),
        })
    stats["gpu"] = gpu_stats
    return stats


class MG_CleanUp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {}),
            },
            "optional": {
                "hard_trim": ("BOOLEAN", {"default": True, "tooltip": "Aggressively free RAM/VRAM and ask OS to return pages to the system."}),
                "sync_cuda": ("BOOLEAN", {"default": True, "tooltip": "Synchronize CUDA before cleanup to flush pending kernels."}),
                "hires_only_threshold": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 64, "tooltip": "Apply only when latent longest side >= threshold (0 == always)."}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("samples", "Preview")
    FUNCTION = "apply"
    CATEGORY = "MagicNodes"

    def apply(self, samples, hard_trim=True, sync_cuda=True, hires_only_threshold=0,
              model=None, positive=None, negative=None, vae=None):
        img_prev = None
        try:
            if (model is not None) and (positive is not None) and (negative is not None) and (vae is not None):
                lat = samples.get("samples", None)
                if lat is not None and isinstance(lat, torch.Tensor) and lat.ndim == 4:
                    z = lat
                    B, C, H, W = z.shape
                    target = 32
                    z_ds = z if (H == target and W == target) else F.interpolate(z, size=(target, target), mode='bilinear', align_corners=False)
                    lat_img = _sample.fix_empty_latent_channels(model, z_ds) if hasattr(_sample, 'fix_empty_latent_channels') else z_ds
                    batch_inds = samples.get("batch_index", None)
                    noise = _sample.prepare_noise(lat_img, int(0), batch_inds)
                    steps = 1
                    out = _sample.sample(
                        model, noise, int(steps), float(1.0), "ddim", "normal",
                        positive, negative, lat_img,
                        denoise=float(0.10), disable_noise=False, start_step=None, last_step=None,
                        force_full_denoise=False, noise_mask=None, callback=None,
                        disable_pbar=not _utils.PROGRESS_BAR_ENABLED, seed=int(0)
                    )
                    try:
                        img_prev = vae.decode(out)
                        if len(img_prev.shape) == 5:
                            img_prev = img_prev.reshape(-1, img_prev.shape[-3], img_prev.shape[-2], img_prev.shape[-1])
                    except Exception:
                        img_prev = None
        except Exception:
            img_prev = None

        try:
            do_cleanup = True
            try:
                if int(hires_only_threshold) > 0:
                    z = samples.get("samples", None)
                    if z is not None and hasattr(z, "shape") and len(z.shape) >= 4:
                        _, _, H, W = z.shape
                        if max(int(H), int(W)) < int(hires_only_threshold):
                            do_cleanup = False
            except Exception:
                pass
            if do_cleanup:
                print("=== CleanUP RAM and GPU ===")
                stats = cleanup_memory(sync_cuda=bool(sync_cuda), hard_trim=bool(hard_trim))
                try:
                    print(f"RAM freed: {stats['ram_freed_mb']:.1f} MB (before {stats['ram_before_mb']:.1f} -> after {stats['ram_after_mb']:.1f})")
                except Exception:
                    pass
                try:
                    for g in stats.get("gpu", []):
                        print(
                            f"GPU{g['device']}: reserved freed {g['reserved_freed_mb']:.1f} MB ( {g['reserved_before_mb']:.1f} -> {g['reserved_after_mb']:.1f} ), "
                            f"allocated freed {g['allocated_freed_mb']:.1f} MB ( {g['allocated_before_mb']:.1f} -> {g['allocated_after_mb']:.1f} )"
                        )
                except Exception:
                    pass
                # Second pass after short delay to catch late releasers
                try:
                    time.sleep(0.150)
                    stats2 = cleanup_memory(sync_cuda=False, hard_trim=bool(hard_trim))
                    if stats2 and float(stats2.get('ram_freed_mb', 0.0)) > 0.0:
                        print(f"2nd pass: RAM freed +{stats2['ram_freed_mb']:.1f} MB")
                    try:
                        for g in stats2.get('gpu', []):
                            if float(g.get('reserved_freed_mb', 0.0)) > 0.0 or float(g.get('allocated_freed_mb', 0.0)) > 0.0:
                                print(f"2nd pass GPU{g['device']}: reserved +{g['reserved_freed_mb']:.1f} MB, allocated +{g['allocated_freed_mb']:.1f} MB")
                    except Exception:
                        pass
                except Exception:
                    pass
                print("done.")
        except Exception:
            pass

        if img_prev is None:
            try:
                device = model_management.intermediate_device() if hasattr(model_management, 'intermediate_device') else 'cpu'
                img_prev = torch.zeros((1, 32, 32, 3), dtype=torch.float32, device=device)
            except Exception:
                img_prev = torch.zeros((1, 32, 32, 3))
        return (samples, img_prev)



