import os, sys, importlib.util

# Normalize package name so relative imports work even if loaded by absolute path
if __name__ != 'MagicNodes':
    sys.modules['MagicNodes'] = sys.modules[__name__]
    __package__ = 'MagicNodes'
    # Precreate subpackage alias MagicNodes.mod
    _mod_pkg_name = 'MagicNodes.mod'
    _mod_pkg_dir = os.path.join(os.path.dirname(__file__), 'mod')
    _mod_pkg_file = os.path.join(_mod_pkg_dir, '__init__.py')
    if _mod_pkg_name not in sys.modules and os.path.isfile(_mod_pkg_file):
        _spec = importlib.util.spec_from_file_location(
            _mod_pkg_name, _mod_pkg_file, submodule_search_locations=[_mod_pkg_dir]
        )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_mod_pkg_name] = _mod
        assert _spec.loader is not None
        _spec.loader.exec_module(_mod)

# Imports of active nodes
from .mod.mg_combinode import MagicNodesCombiNode
from .mod.hard.mg_upscale_module import MagicUpscaleModule
from .mod.hard.mg_adaptive import AdaptiveSamplerHelper
from .mod.hard.mg_cade25 import ComfyAdaptiveDetailEnhancer25
from .mod.hard.mg_ids import IntelligentDetailStabilizer
from .mod.mg_seed_latent import MagicSeedLatent
from .mod.mg_sagpu_attention import PatchSageAttention
from .mod.hard.mg_controlfusion import MG_ControlFusion
from .mod.hard.mg_zesmart_sampler_v1_1 import MG_ZeSmartSampler
from .mod.easy.mg_cade25_easy import CADEEasyUI as ComfyAdaptiveDetailEnhancer25_Easy
from .mod.easy.mg_controlfusion_easy import MG_ControlFusionEasyUI as MG_ControlFusion_Easy
from .mod.easy.mg_supersimple_easy import MG_SuperSimple

# Place Easy/Hard variants under dedicated UI categories
try:
    ComfyAdaptiveDetailEnhancer25_Easy.CATEGORY = "MagicNodes/Easy"
except Exception:
    pass
try:
    MG_ControlFusion_Easy.CATEGORY = "MagicNodes/Easy"
except Exception:
    pass
try:
    MG_SuperSimple.CATEGORY = "MagicNodes/Easy"
except Exception:
    pass
try:
    ComfyAdaptiveDetailEnhancer25.CATEGORY = "MagicNodes/Hard"
    IntelligentDetailStabilizer.CATEGORY = "MagicNodes/Hard"
    MagicUpscaleModule.CATEGORY = "MagicNodes/Hard"
    AdaptiveSamplerHelper.CATEGORY = "MagicNodes/Hard"
    PatchSageAttention.CATEGORY = "MagicNodes"
    MG_ControlFusion.CATEGORY = "MagicNodes/Hard"
    MG_ZeSmartSampler.CATEGORY = "MagicNodes/Hard"
except Exception:
    pass

NODE_CLASS_MAPPINGS = {
    "MagicNodesCombiNode": MagicNodesCombiNode,
    "MagicSeedLatent": MagicSeedLatent,
    "PatchSageAttention": PatchSageAttention,
    "MagicUpscaleModule": MagicUpscaleModule,
    "ComfyAdaptiveDetailEnhancer25": ComfyAdaptiveDetailEnhancer25,
    "IntelligentDetailStabilizer": IntelligentDetailStabilizer,
    "MG_ControlFusion": MG_ControlFusion,
    "MG_ZeSmartSampler": MG_ZeSmartSampler,
    # Easy variants (limited-surface controls)
    "ComfyAdaptiveDetailEnhancer25_Easy": ComfyAdaptiveDetailEnhancer25_Easy,
    "MG_ControlFusion_Easy": MG_ControlFusion_Easy,
    "MG_SuperSimple": MG_SuperSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicNodesCombiNode": "MG_CombiNode",
    "MagicSeedLatent": "MG_SeedLatent",
    # TDE removed from this build
    "PatchSageAttention": "MG_AccelAttention",
    "ComfyAdaptiveDetailEnhancer25": "MG_CADE 2.5",
    "MG_ControlFusion": "MG_ControlFusion",
    "MG_ZeSmartSampler": "MG_ZeSmartSampler",
    "IntelligentDetailStabilizer": "MG_IDS",
    "MagicUpscaleModule": "MG_UpscaleModule",
    # Easy variants (grouped under MagicNodes/Easy)
    "ComfyAdaptiveDetailEnhancer25_Easy": "MG_CADE 2.5 (Easy)",
    "MG_ControlFusion_Easy": "MG_ControlFusion (Easy)",
    "MG_SuperSimple": "MG_SuperSimple",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
]





