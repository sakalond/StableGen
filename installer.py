#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import requests
import shutil
from pathlib import Path
from typing import Dict, List, Set, Any

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# --- Compatibility Patches ---
# Polyfill for Path.is_junction() which requires Python 3.12+.
# comfy_env uses it but many ComfyUI installs still run Python 3.10/3.11.
_PATCH_PY310_IS_JUNCTION = (
    "# -- StableGen patch: Path.is_junction polyfill for Python < 3.12 --\n"
    "import pathlib as _pathlib\n"
    "if not hasattr(_pathlib.Path, 'is_junction'):\n"
    "    import os as _os, stat as _stat\n"
    "    def _is_junction(self):\n"
    "        try:\n"
    "            return bool(\n"
    "                _os.lstat(str(self)).st_file_attributes\n"
    "                & _stat.FILE_ATTRIBUTE_REPARSE_POINT\n"
    "            )\n"
    "        except (OSError, AttributeError):\n"
    "            return False\n"
    "    _pathlib.Path.is_junction = _is_junction\n"
    "# -- End StableGen patch --\n\n"
)

# Patch for lazy_manager.py: Add attn_backend to config comparison.
# Without this, switching attention backend (e.g. xformers -> flash_attn)
# doesn't recreate the model manager, leaving stale config.
_PATCH_TRELLIS_ATTN_CMP_ANCHOR = (
    "    elif (_LAZY_MANAGER.model_name != model_name or\n"
    "          _LAZY_MANAGER.resolution != resolution or\n"
    "          _LAZY_MANAGER.vram_mode != vram_mode):\n"
    "        # Config changed, recreate manager\n"
    "        _LAZY_MANAGER.cleanup()\n"
)

_PATCH_TRELLIS_ATTN_CMP_REPLACE = (
    "    elif (_LAZY_MANAGER.model_name != model_name or\n"
    "          _LAZY_MANAGER.resolution != resolution or\n"
    "          _LAZY_MANAGER.vram_mode != vram_mode or\n"
    "          _LAZY_MANAGER.attn_backend != attn_backend):\n"
    "        # Config changed, recreate manager\n"
    "        print(f\"[TRELLIS2] Config changed, recreating model manager...\", file=sys.stderr)\n"
    "        _LAZY_MANAGER.cleanup()\n"
)

# Patch for stages.py: Wrap DinoV3 feature extraction in torch.no_grad().
# Without this, autograd retains ViT-L intermediate activations on GPU,
# leaking ~5-9 GB between runs in the persistent comfy-env worker.
_PATCH_TRELLIS_NOGRAD_ANCHOR = (
    "    # Load DinoV3 and extract features\n"
    "    model = manager.get_dinov3(device)\n"
    "\n"
    "    # Get 512px conditioning\n"
    "    model.image_size = 512\n"
    "    cond_512 = model([pil_image])\n"
    "\n"
    "    # Get 1024px conditioning if requested\n"
    "    cond_1024 = None\n"
    "    if include_1024:\n"
    "        model.image_size = 1024\n"
    "        cond_1024 = model([pil_image])\n"
    "\n"
    "    # Unload DinoV3 immediately\n"
    "    manager.unload_dinov3()\n"
    "\n"
    "    # Create negative conditioning\n"
    "    neg_cond = torch.zeros_like(cond_512)\n"
    "\n"
    "    conditioning = {\n"
    "        'cond_512': cond_512.cpu(),\n"
    "        'neg_cond': neg_cond.cpu(),\n"
    "    }\n"
    "    if cond_1024 is not None:\n"
    "        conditioning['cond_1024'] = cond_1024.cpu()\n"
)

_PATCH_TRELLIS_NOGRAD_REPLACE = (
    "    # Load DinoV3 and extract features\n"
    "    model = manager.get_dinov3(device)\n"
    "\n"
    "    # Use no_grad to prevent autograd from retaining intermediate activations\n"
    "    # on GPU. Without this, the computation graph keeps ViT-L activations alive\n"
    "    # in the comfy-env worker, leaking ~5-9 GB between runs.\n"
    "    with torch.no_grad():\n"
    "        # Get 512px conditioning\n"
    "        model.image_size = 512\n"
    "        cond_512 = model([pil_image])\n"
    "\n"
    "        # Get 1024px conditioning if requested\n"
    "        cond_1024 = None\n"
    "        if include_1024:\n"
    "            model.image_size = 1024\n"
    "            cond_1024 = model([pil_image])\n"
    "\n"
    "    # Unload DinoV3 immediately\n"
    "    manager.unload_dinov3()\n"
    "\n"
    "    # Create negative conditioning\n"
    "    neg_cond = torch.zeros_like(cond_512)\n"
    "\n"
    "    # .detach() breaks any remaining graph references before moving to CPU\n"
    "    conditioning = {\n"
    "        'cond_512': cond_512.detach().cpu(),\n"
    "        'neg_cond': neg_cond.detach().cpu(),\n"
    "    }\n"
    "    if cond_1024 is not None:\n"
    "        conditioning['cond_1024'] = cond_1024.detach().cpu()\n"
    "\n"
    "    # Free the GPU originals immediately\n"
    "    del cond_512, cond_1024, neg_cond\n"
    "    gc.collect()\n"
    "    torch.cuda.empty_cache()\n"
)

# ---------------------------------------------------------------------------
# comfy-env VRAM leak fix patches
# ---------------------------------------------------------------------------
# comfy-env hooks torch.nn.Module.to() and .cuda() to auto-register every
# model that lands on CUDA in a permanent _model_registry (with no unregister
# API).  Between ComfyUI node calls the model manager issues
# "Requested to load SubprocessModel" which calls model_to_device("cuda"),
# moving ALL registered models back to GPU.  This leaks ~7 GB of model
# weights between runs.  The fix: unregister from the registry before
# moving models to CPU / deleting them.
# ---------------------------------------------------------------------------

# -- base.py _unload_model: unregister from comfy-env + model.cpu() before del
_PATCH_TRELLIS_UNLOAD_CPU_ANCHOR = (
    "            # Delete the model entirely\n"
    "            self.models[model_key] = None\n"
    "            del model\n"
)

_PATCH_TRELLIS_UNLOAD_CPU_REPLACE = (
    "            # Unregister from comfy-env's model registry BEFORE moving to CPU.\n"
    "            # comfy-env auto-registers every nn.Module that lands on CUDA via\n"
    "            # hooked Module.to(). Without unregistering, the registry keeps a\n"
    "            # permanent reference and ComfyUI's model manager re-loads the model\n"
    "            # to GPU between calls, leaking the full model weights per run.\n"
    "            # Replace with zero-param dummy so host's model_to_device succeeds\n"
    "            # but consumes 0 VRAM.  Registry accessed via closure on\n"
    "            # sys.modules['comfy_worker'].register_model.\n"
    "            try:\n"
    "                _cw = sys.modules.get('comfy_worker')\n"
    "                if _cw is not None:\n"
    "                    _reg_fn = getattr(_cw, 'register_model', None)\n"
    "                    if _reg_fn is not None and hasattr(_reg_fn, '__closure__'):\n"
    "                        _fv = _reg_fn.__code__.co_freevars\n"
    "                        _cl = _reg_fn.__closure__\n"
    "                        _by_obj = _cl[_fv.index('_model_id_by_obj')].cell_contents\n"
    "                        _registry = _cl[_fv.index('_model_registry')].cell_contents\n"
    "                        _meta = _cl[_fv.index('_model_registry_meta')].cell_contents\n"
    "                        obj_id = id(model)\n"
    "                        if obj_id in _by_obj:\n"
    "                            ce_model_id = _by_obj.pop(obj_id)\n"
    "                            dummy = torch.nn.Module()\n"
    "                            _registry[ce_model_id] = dummy\n"
    "                            _meta[ce_model_id] = {'size': 0, 'kind': 'other'}\n"
    "                            _by_obj[id(dummy)] = ce_model_id\n"
    '                            print(f"[TRELLIS2] Unregistered {model_key} from comfy-env registry (id={ce_model_id})", file=sys.stderr, flush=True)\n'
    "            except Exception:\n"
    "                pass\n"
    "            # Force all parameters and buffers off GPU.\n"
    "            try:\n"
    "                model.cpu()\n"
    "            except Exception:\n"
    "                pass\n"
    "            # Delete the model entirely\n"
    "            self.models[model_key] = None\n"
    "            del model\n"
)

# -- lazy_manager.py: insert _unregister_from_comfy_env helper function
_PATCH_TRELLIS_COMFY_HELPER_ANCHOR = (
    "# Global model manager instance\n"
    '_LAZY_MANAGER: Optional["LazyModelManager"] = None\n'
)

_PATCH_TRELLIS_COMFY_HELPER_REPLACE = (
    "\n"
    "def _unregister_from_comfy_env(model, label: str = \"\"):\n"
    '    """Remove an nn.Module from comfy-env\'s worker-side model registry.\n'
    "\n"
    "    comfy-env hooks Module.to() to auto-register every model that lands on\n"
    "    CUDA.  The registry keeps a permanent reference, and ComfyUI's model\n"
    "    manager will re-load the model to GPU between calls via\n"
    "    'Requested to load SubprocessModel'.  We replace the model with a\n"
    "    zero-param dummy so the host's model_to_device succeeds (no-op) but\n"
    "    no VRAM is consumed.  Registry accessed via closure on\n"
    "    sys.modules['comfy_worker'].register_model.\n"
    '    """\n'
    "    try:\n"
    "        _cw = sys.modules.get('comfy_worker')\n"
    "        if _cw is None:\n"
    "            return\n"
    "        _reg_fn = getattr(_cw, 'register_model', None)\n"
    "        if _reg_fn is None or not hasattr(_reg_fn, '__closure__'):\n"
    "            return\n"
    "        _fv = _reg_fn.__code__.co_freevars\n"
    "        _cl = _reg_fn.__closure__\n"
    "        _by_obj = _cl[_fv.index('_model_id_by_obj')].cell_contents\n"
    "        _registry = _cl[_fv.index('_model_registry')].cell_contents\n"
    "        _meta = _cl[_fv.index('_model_registry_meta')].cell_contents\n"
    "        obj_id = id(model)\n"
    "        if obj_id in _by_obj:\n"
    "            model_id = _by_obj.pop(obj_id)\n"
    "            dummy = torch.nn.Module()\n"
    "            _registry[model_id] = dummy\n"
    "            _meta[model_id] = {'size': 0, 'kind': 'other'}\n"
    "            _by_obj[id(dummy)] = model_id\n"
    '            print(f"[TRELLIS2] Unregistered {label} from comfy-env registry (id={model_id})", file=sys.stderr)\n'
    "    except Exception:\n"
    "        pass\n"
    "\n"
    "\n"
    "# Global model manager instance\n"
    '_LAZY_MANAGER: Optional["LazyModelManager"] = None\n'
)

# -- lazy_manager.py unload_dinov3: add comfy-env unregistration
_PATCH_TRELLIS_UNLOAD_DINOV3_ANCHOR = (
    "    def unload_dinov3(self):\n"
    '        """Unload DinoV3 to free VRAM."""\n'
    "        if self.dinov3_model is not None:\n"
    "            self.dinov3_model.cpu()\n"
    "            self.dinov3_model = None\n"
    "            gc.collect()\n"
    "            torch.cuda.empty_cache()\n"
    '            print(f"[TRELLIS2] DinoV3 offloaded", file=sys.stderr)\n'
)

_PATCH_TRELLIS_UNLOAD_DINOV3_REPLACE = (
    "    def unload_dinov3(self):\n"
    '        """Unload DinoV3 to free VRAM."""\n'
    "        if self.dinov3_model is not None:\n"
    "            # Unregister the inner nn.Module (DINOv3ViTModel) from comfy-env.\n"
    "            # DinoV3FeatureExtractor is a plain class; comfy-env hooks on the\n"
    "            # inner .model which is the actual nn.Module that got .to(cuda).\n"
    "            inner = getattr(self.dinov3_model, 'model', self.dinov3_model)\n"
    '            _unregister_from_comfy_env(inner, "dinov3")\n'
    "            self.dinov3_model.cpu()\n"
    "            self.dinov3_model = None\n"
    "            gc.collect()\n"
    "            torch.cuda.empty_cache()\n"
    '            print(f"[TRELLIS2] DinoV3 offloaded", file=sys.stderr)\n'
)

# -- lazy_manager.py unload_shape_pipeline: iterate models + comfy-env unreg
_PATCH_TRELLIS_UNLOAD_SHAPE_ANCHOR = (
    "    def unload_shape_pipeline(self):\n"
    '        """Unload shape pipeline to free VRAM."""\n'
    "        if self.shape_pipeline is not None:\n"
    "            self.shape_pipeline = None\n"
    "            gc.collect()\n"
    "            torch.cuda.empty_cache()\n"
    '            print(f"[TRELLIS2] Shape pipeline offloaded", file=sys.stderr)\n'
)

_PATCH_TRELLIS_UNLOAD_SHAPE_REPLACE = (
    "    def unload_shape_pipeline(self):\n"
    '        """Unload shape pipeline to free VRAM."""\n'
    "        if self.shape_pipeline is not None:\n"
    "            # Force any remaining models off GPU before dropping pipeline\n"
    "            if hasattr(self.shape_pipeline, 'models'):\n"
    "                for key in list(self.shape_pipeline.models.keys()):\n"
    "                    model = self.shape_pipeline.models[key]\n"
    "                    if model is not None:\n"
    '                        _unregister_from_comfy_env(model, f"shape/{key}")\n'
    "                        try:\n"
    "                            model.cpu()\n"
    "                        except Exception:\n"
    "                            pass\n"
    "                        self.shape_pipeline.models[key] = None\n"
    "                        del model\n"
    "                self.shape_pipeline.models.clear()\n"
    "            self.shape_pipeline = None\n"
    "            gc.collect()\n"
    "            torch.cuda.empty_cache()\n"
    '            print(f"[TRELLIS2] Shape pipeline offloaded", file=sys.stderr)\n'
)

# -- lazy_manager.py unload_texture_pipeline: iterate models + comfy-env unreg
_PATCH_TRELLIS_UNLOAD_TEX_ANCHOR = (
    "    def unload_texture_pipeline(self):\n"
    '        """Unload texture pipeline to free VRAM."""\n'
    "        if self.texture_pipeline is not None:\n"
    "            self.texture_pipeline = None\n"
    "            gc.collect()\n"
    "            torch.cuda.empty_cache()\n"
    '            print(f"[TRELLIS2] Texture pipeline offloaded", file=sys.stderr)\n'
)

_PATCH_TRELLIS_UNLOAD_TEX_REPLACE = (
    "    def unload_texture_pipeline(self):\n"
    '        """Unload texture pipeline to free VRAM."""\n'
    "        if self.texture_pipeline is not None:\n"
    "            # Force any remaining models off GPU before dropping pipeline\n"
    "            if hasattr(self.texture_pipeline, 'models'):\n"
    "                for key in list(self.texture_pipeline.models.keys()):\n"
    "                    model = self.texture_pipeline.models[key]\n"
    "                    if model is not None:\n"
    '                        _unregister_from_comfy_env(model, f"texture/{key}")\n'
    "                        try:\n"
    "                            model.cpu()\n"
    "                        except Exception:\n"
    "                            pass\n"
    "                        self.texture_pipeline.models[key] = None\n"
    "                        del model\n"
    "                self.texture_pipeline.models.clear()\n"
    "            self.texture_pipeline = None\n"
    "            gc.collect()\n"
    "            torch.cuda.empty_cache()\n"
    '            print(f"[TRELLIS2] Texture pipeline offloaded", file=sys.stderr)\n'
)

# Patch for stages.py: clean up IPC tensor files from previous generations
# inside _save_to_disk so they don't accumulate and fill the disk.
_PATCH_TRELLIS_TEMP_CLEANUP_ANCHOR = (
    "def _save_to_disk(data, prefix):\n"
    "    path = os.path.join(_get_temp_dir(), f'{prefix}_{uuid.uuid4().hex[:8]}.pt')\n"
    "    torch.save(data, path)\n"
    "    return {'_tensor_file': path}"
)

_PATCH_TRELLIS_TEMP_CLEANUP_REPLACE = (
    "def _save_to_disk(data, prefix):\n"
    "    import glob as _glob\n"
    "    temp_dir = _get_temp_dir()\n"
    "    # StableGen patch: clean up files from previous generations\n"
    "    for _old in _glob.glob(os.path.join(temp_dir, f'{prefix}_*.pt')):\n"
    "        try:\n"
    "            os.remove(_old)\n"
    "        except OSError:\n"
    "            pass\n"
    "    path = os.path.join(temp_dir, f'{prefix}_{uuid.uuid4().hex[:8]}.pt')\n"
    "    torch.save(data, path)\n"
    "    return {'_tensor_file': path}"
)

# --- Configuration: Dependencies Data ---
# Sizes are in MB.
DEPENDENCIES: Dict[str, Dict[str, Any]] = {
    # --- Custom Nodes ---
    "cn_ipadapter_plus": {
        "id": "cn_ipadapter_plus", "type": "node", "name": "ComfyUI IPAdapter Plus",
        "git_url": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI_IPAdapter_plus",
        "license": "GPL-3.0", "packages": ["core"]
    },
    # --- Models ---
    # Core Models
    "model_ipadapter_plus_sdxl_vit_h": {
        "id": "model_ipadapter_plus_sdxl_vit_h", "type": "model", "name": "IPAdapter Plus SDXL ViT-H",
        "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors?download=true",
        "target_path_relative": "models/ipadapter", "filename": "ip-adapter-plus_sdxl_vit-h.safetensors",
        "license": "Apache 2.0", "size_mb": 850, "packages": ["core"]
    },
    "model_clip_vision_h": {
        "id": "model_clip_vision_h", "type": "model", "name": "IPAdapter CLIP Vision ViT-H",
        "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
        "target_path_relative": "models/clip_vision", "filename": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        "rename_from": "model.safetensors", "license": "Apache 2.0", "size_mb": 2500, "packages": ["core"]
    },
    "model_clip_vision_g": {
        "id": "model_clip_vision_g", "type": "model", "name": "IPAdapter CLIP Vision ViT-G",
        "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors",
        "target_path_relative": "models/clip_vision", "filename": "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors",
        "rename_from": "model.safetensors", "license": "Apache 2.0", "size_mb": 3500, "packages": ["core"]
    },
    "lora_sdxl_lightning_8step": {
        "id": "lora_sdxl_lightning_8step", "type": "model", "name": "SDXL Lightning 8-Step LoRA",
        "url": "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "sdxl_lightning_8step_lora.safetensors",
        "license": "OpenRAIL++", "size_mb": 400, "packages": ["core"]
    },
    # Preset Essentials
    "controlnet_depth_sdxl_preset": {
        "id": "controlnet_depth_sdxl_preset", "type": "model", "name": "ControlNet Depth SDXL (for presets)",
        "url": "https://huggingface.co/xinsir/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true",
        "target_path_relative": "models/controlnet", "filename": "controlnet_depth_sdxl.safetensors",
        "rename_from": "diffusion_pytorch_model.safetensors", "license": "Apache 2.0", "size_mb": 2500, "packages": ["preset_essentials"]
    },
    # Extended SDXL Optional Models
    "lora_sdxl_lightning_4step": {
        "id": "lora_sdxl_lightning_4step", "type": "model", "name": "SDXL Lightning 4-Step LoRA",
        "url": "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "sdxl_lightning_4step_lora.safetensors",
        "license": "OpenRAIL++", "size_mb": 400, "packages": ["extended_optional"]
    },
    "lora_sdxl_lightning_2step": {
        "id": "lora_sdxl_lightning_2step", "type": "model", "name": "SDXL Lightning 2-Step LoRA",
        "url": "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_2step_lora.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "sdxl_lightning_2step_lora.safetensors",
        "license": "OpenRAIL++", "size_mb": 400, "packages": ["extended_optional"]
    },
    "lora_hyper_sdxl_8step": {
        "id": "lora_hyper_sdxl_8step", "type": "model", "name": "Hyper-SDXL 8-Steps LoRA",
        "url": "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-lora.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "Hyper-SDXL-8steps-lora.safetensors",
        "license": "Unknown (User to verify)", "size_mb": 800, "packages": ["extended_optional"]
    },
    "lora_hyper_sdxl_4step": {
        "id": "lora_hyper_sdxl_4step", "type": "model", "name": "Hyper-SDXL 4-Steps LoRA",
        "url": "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-4steps-lora.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "Hyper-SDXL-4steps-lora.safetensors",
        "license": "Unknown (User to verify)", "size_mb": 800, "packages": ["extended_optional"]
    },
    "lora_hyper_sdxl_1step": {
        "id": "lora_hyper_sdxl_1step", "type": "model", "name": "Hyper-SDXL 1-Step LoRA",
        "url": "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-1step-lora.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "Hyper-SDXL-1step-lora.safetensors",
        "license": "Unknown (User to verify)", "size_mb": 800, "packages": ["extended_optional"]
    },
    "controlnet_depth_sdxl_fp16_alt": {
        "id": "controlnet_depth_sdxl_fp16_alt", "type": "model", "name": "ControlNet Depth SDXL fp16 (alternative)",
        "url": "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true",
        "target_path_relative": "models/controlnet", "filename": "diffusion_pytorch_model.fp16.safetensors",
        "license": "OpenRAIL++", "size_mb": 2500, "packages": ["extended_optional"]
    },
    "controlnet_union_promax": {
        "id": "controlnet_union_promax", "type": "model", "name": "ControlNet Union SDXL ProMax",
        "url": "https://huggingface.co/brad-twinkl/controlnet-union-sdxl-1.0-promax/resolve/main/diffusion_pytorch_model.safetensors?download=true",
        "target_path_relative": "models/controlnet", "filename": "sdxl_promax.safetensors",
        "rename_from": "diffusion_pytorch_model.safetensors", "license": "Apache 2.0", "size_mb": 2500, "packages": ["extended_optional"]
    },
    # Checkpoints
    "checkpoint_realvis_v5": {
        "id": "checkpoint_realvis_v5", "type": "model", "name": "RealVisXL V5.0 fp16 Checkpoint",
        "url": "https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors?download=true",
        "target_path_relative": "models/checkpoints", "filename": "RealVisXL_V5.0_fp16.safetensors",
        "license": "OpenRAIL++", "size_mb": 6500, "packages": ["checkpoint_realvis"]
    },
    # Qwen Core
    "cn_comfyui_gguf": {
        "id": "cn_comfyui_gguf", "type": "node", "name": "ComfyUI GGUF Loader",
        "git_url": "https://github.com/city96/ComfyUI-GGUF.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-GGUF",
        "license": "Apache 2.0", "packages": ["qwen_core"]
    },
    "model_qwen_unet_q3_k_m": {
        "id": "model_qwen_unet_q3_k_m", "type": "model", "name": "Qwen Image Edit 2509 UNet (Q3_K_M)",
        "url": "https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/resolve/main/Qwen-Image-Edit-2509-Q3_K_M.gguf?download=true",
        "target_path_relative": "models/unet", "filename": "Qwen-Image-Edit-2509-Q3_K_M.gguf",
        "license": "Apache 2.0", "size_mb": 9760, "packages": ["qwen_core"]
    },
    "model_qwen_vae": {
        "id": "model_qwen_vae", "type": "model", "name": "Qwen Image VAE",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors?download=true",
        "target_path_relative": "models/vae", "filename": "qwen_image_vae.safetensors",
        "license": "Apache 2.0", "size_mb": 254, "packages": ["qwen_core"]
    },
    "model_qwen_text_encoder_fp8": {
        "id": "model_qwen_text_encoder_fp8", "type": "model", "name": "Qwen 2.5 VL 7B Text Encoder (FP8)",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors?download=true",
        "target_path_relative": "models/clip", "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "license": "Apache 2.0", "size_mb": 9380, "packages": ["qwen_core"]
    },
    "lora_qwen_lightning_4step_edit": {
        "id": "lora_qwen_lightning_4step_edit", "type": "model", "name": "Qwen Image Edit Lightning 4-Step LoRA (bf16)",
        "url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        "license": "Apache 2.0", "size_mb": 850, "packages": ["qwen_core"]
    },
    # Qwen Extras
    "lora_qwen_lightning_8step": {
        "id": "lora_qwen_lightning_8step", "type": "model", "name": "Qwen Image Edit Lightning 8-Step LoRA (bf16)",
        "url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
        "license": "Apache 2.0", "size_mb": 850, "packages": ["qwen_extras"]
    },
    "lora_qwen_lightning_4step": {
        "id": "lora_qwen_lightning_4step", "type": "model", "name": "Qwen Image Lightning 4-Step LoRA (bf16)",
        "url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors",
        "license": "Apache 2.0", "size_mb": 850, "packages": ["qwen_extras"]
    },
    # Nunchaku Qwen
    "cn_nunchaku": {
        "id": "cn_nunchaku", "type": "node", "name": "ComfyUI Nunchaku",
        "git_url": "https://github.com/nunchaku-tech/ComfyUI-nunchaku.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-nunchaku",
        "license": "Apache 2.0", "packages": ["qwen_nunchaku"],
        "pip_packages": ["nunchaku"]
    },
    "cn_qwen_lora_loader": {
        "id": "cn_qwen_lora_loader", "type": "node", "name": "ComfyUI Qwen Image LoRA Loader",
        "git_url": "https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-QwenImageLoraLoader",
        "license": "Apache 2.0", "packages": ["qwen_nunchaku"]
    },
    "model_nunchaku_qwen": {
        "id": "model_nunchaku_qwen", "type": "model", "name": "Nunchaku Qwen Image Edit 2509 (Int4)",
        "url": "https://huggingface.co/nunchaku-tech/nunchaku-qwen-image-edit-2509/resolve/main/lightning-251115/svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors?download=true",
        "target_path_relative": "models/diffusion_models", "filename": "svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors",
        "license": "Apache 2.0", "size_mb": 12700, "packages": ["qwen_nunchaku"]
    },
    # --- PBR Decomposition (Marigold IID) ---
    "cn_comfyui_marigold": {
        "id": "cn_comfyui_marigold", "type": "node", "name": "ComfyUI-Marigold (IID/Depth/Normal)",
        "git_url": "https://github.com/kijai/ComfyUI-Marigold.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-Marigold",
        "license": "GPL-3.0", "packages": ["pbr_marigold"],
        "pip_packages": ["diffusers>=0.28", "matplotlib"] + (["triton-windows"] if sys.platform == "win32" else []),
        "run_install_script": True,
    },
    # --- StableDelight (specular-free albedo) ---
    "cn_comfyui_stabledelight": {
        "id": "cn_comfyui_stabledelight", "type": "node", "name": "ComfyUI StableDelight (Delighting)",
        "git_url": "https://github.com/lldacing/ComfyUI_StableDelight_ll.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI_StableDelight_ll",
        "license": "Apache-2.0", "packages": ["pbr_stabledelight"],
        "post_clone_patches": [
            {
                "file": "nodes/BaseNode.py",
                "marker": "local_files_only=False",
                "anchor": "local_files_only=True",
                "patch": "local_files_only=False",
                "mode": "replace",
            },
        ],
    },
    "model_stabledelight": {
        "id": "model_stabledelight", "type": "hf_model",
        "name": "StableDelight Model (yoso-delight-v0-4-base)",
        "hf_repo": "Stable-X/yoso-delight-v0-4-base",
        "local_dir": "Stable-X--yoso-delight-v0-4-base",
        "license": "Apache-2.0", "size_mb": 3300,
        "packages": ["pbr_stabledelight"],
    },
    # --- TRELLIS.2 ---
    "cn_trellis2": {
        "id": "cn_trellis2", "type": "node", "name": "ComfyUI TRELLIS.2",
        "git_url": "https://github.com/sakalond/ComfyUI-TRELLIS2.git",
        "commit": "6b0b2148f45bbafa0b86bfd25c63602b63a7aae0",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-TRELLIS2",
        "license": "MIT (Note: textured pipeline uses NVIDIA non-commercial libs)", "packages": ["trellis2"],
        "pip_packages": ["comfy-env==0.2.0", "numpy>=2.0.0", "scipy>=1.14.0"],
        "clean_envs": True,
        "run_install_script": True,
        "post_clone_patches": [
            {
                "file": "__init__.py",
                "marker": "StableGen patch: Path.is_junction",
                "anchor": "from comfy_env import",
                "patch": _PATCH_PY310_IS_JUNCTION,
            },
            {
                "file": "prestartup_script.py",
                "marker": "StableGen patch: Path.is_junction",
                "anchor": "from comfy_env import",
                "patch": _PATCH_PY310_IS_JUNCTION,
            },
            {
                "file": "nodes/trellis_utils/lazy_manager.py",
                "marker": "_LAZY_MANAGER.attn_backend != attn_backend",
                "anchor": _PATCH_TRELLIS_ATTN_CMP_ANCHOR,
                "patch": _PATCH_TRELLIS_ATTN_CMP_REPLACE,
                "mode": "replace",
            },
            {
                "file": "nodes/trellis_utils/stages.py",
                "marker": "# Use no_grad to prevent autograd",
                "anchor": _PATCH_TRELLIS_NOGRAD_ANCHOR,
                "patch": _PATCH_TRELLIS_NOGRAD_REPLACE,
                "mode": "replace",
            },
            {
                "file": "nodes/trellis2/pipelines/base.py",
                "marker": "_model_id_by_obj",
                "anchor": _PATCH_TRELLIS_UNLOAD_CPU_ANCHOR,
                "patch": _PATCH_TRELLIS_UNLOAD_CPU_REPLACE,
                "mode": "replace",
            },
            {
                "file": "nodes/trellis_utils/lazy_manager.py",
                "marker": "_unregister_from_comfy_env",
                "anchor": _PATCH_TRELLIS_COMFY_HELPER_ANCHOR,
                "patch": _PATCH_TRELLIS_COMFY_HELPER_REPLACE,
                "mode": "replace",
            },
            {
                "file": "nodes/trellis_utils/lazy_manager.py",
                "marker": '_unregister_from_comfy_env(self.dinov3_model',
                "anchor": _PATCH_TRELLIS_UNLOAD_DINOV3_ANCHOR,
                "patch": _PATCH_TRELLIS_UNLOAD_DINOV3_REPLACE,
                "mode": "replace",
            },
            {
                "file": "nodes/trellis_utils/lazy_manager.py",
                "marker": '_unregister_from_comfy_env(model, f"shape/',
                "anchor": _PATCH_TRELLIS_UNLOAD_SHAPE_ANCHOR,
                "patch": _PATCH_TRELLIS_UNLOAD_SHAPE_REPLACE,
                "mode": "replace",
            },
            {
                "file": "nodes/trellis_utils/lazy_manager.py",
                "marker": '_unregister_from_comfy_env(model, f"texture/',
                "anchor": _PATCH_TRELLIS_UNLOAD_TEX_ANCHOR,
                "patch": _PATCH_TRELLIS_UNLOAD_TEX_REPLACE,
                "mode": "replace",
            },
            {
                "file": "nodes/trellis_utils/stages.py",
                "marker": "# StableGen patch: clean up files from previous generations",
                "anchor": _PATCH_TRELLIS_TEMP_CLEANUP_ANCHOR,
                "patch": _PATCH_TRELLIS_TEMP_CLEANUP_REPLACE,
                "mode": "replace",
            },
            {
                "file": "nodes/rembg/BiRefNet.py",
                "marker": "StableGen patch: BiRefNet fp32",
                "anchor": "self.model.eval()",
                "patch": "self.model.eval()\n        self.model.float()  # StableGen patch: BiRefNet fp32 — avoid fp16/fp32 dtype mismatch",
                "mode": "replace",
            },
            {
                "file": "nodes/trellis2/pipelines/rembg/BiRefNet.py",
                "marker": "StableGen patch: timm.layers compat",
                "anchor": "import torch",
                "patch": (
                    "import torch\n"
                    "import sys as _sys  # StableGen patch: timm.layers compat\n"
                    "if 'timm.layers' not in _sys.modules:\n"
                    "    try:\n"
                    "        import timm.models.layers as _tl; _sys.modules['timm.layers'] = _tl  # timm<0.9 compat for BiRefNet\n"
                    "    except ImportError: pass\n"
                ),
                "mode": "replace",
            },
        ]
    },
    # --- FLUX.2 Klein 4B ---
    "model_flux2_klein_4b": {
        "id": "model_flux2_klein_4b", "type": "model",
        "name": "FLUX.2 Klein 4B Diffusion Model (FP8)",
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors?download=true",
        "target_path_relative": "models/diffusion_models",
        "filename": "flux-2-klein-base-4b-fp8.safetensors",
        "license": "Apache 2.0", "size_mb": 4070, "packages": ["flux2_klein"],
    },
    "model_flux2_text_encoder": {
        "id": "model_flux2_text_encoder", "type": "model",
        "name": "FLUX.2 Klein Qwen 3 4B Text Encoder (bf16)",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors?download=true",
        "target_path_relative": "models/text_encoders",
        "filename": "qwen_3_4b.safetensors",
        "license": "Apache 2.0", "size_mb": 8050, "packages": ["flux2_klein"],
    },
    "model_flux2_vae": {
        "id": "model_flux2_vae", "type": "model",
        "name": "FLUX.2 Klein VAE",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors?download=true",
        "target_path_relative": "models/vae",
        "filename": "flux2-vae.safetensors",
        "license": "Apache 2.0", "size_mb": 321, "packages": ["flux2_klein"],
    },
}

# Define what items each menu option entails by listing package tags
# The script will collect all unique items based on the selected package tags.
MENU_PACKAGES: Dict[str, Dict[str, Any]] = {
    '1': {"name": "[MINIMAL CORE] Basic Requirements",
          "tags": ["core"],
          "size_gb": 7.3,
          "description_suffix": "*You will still need to manually download your own SDXL checkpoint(s) and all ControlNet models for full functionality and preset usage.*"},
    '2': {"name": "[ESSENTIAL] Core + Preset Essentials",
          "tags": ["core", "preset_essentials"],
          "size_gb": 9.8,
          "description_suffix": "*All models for preset functionality. You will still need to manually download your own SDXL checkpoint(s).*"},
    '3': {"name": "[RECOMMENDED] Full SDXL Setup (No Checkpoints)",
          "tags": ["core", "preset_essentials", "extended_optional", "pbr_marigold", "pbr_stabledelight"],
          "size_gb": 19.3,
          "description_suffix": "*Downloads optional ControlNet and LoRA models + PBR decomposition nodes. You will still need to manually download your own SDXL checkpoint(s).*"},
    '4': {"name": "[COMPLETE SDXL] Full SDXL Setup + RealVisXL V5.0 Checkpoint",
          "tags": ["core", "preset_essentials", "extended_optional", "pbr_marigold", "pbr_stabledelight", "checkpoint_realvis"],
        "size_gb": 26.3,
        "description_suffix": ""},
    '5': {"name": "[QWEN CORE] Models + GGUF Node",
        "tags": ["qwen_core"],
        "size_gb": 20.3,
        "description_suffix": "*Installs Qwen Image Edit UNet, VAE, text encoder, core LoRA, and GGUF ComfyUI node.*"},
    '6': {"name": "[QWEN EXTRAS] Core + Lightning LoRAs",
        "tags": ["qwen_core", "qwen_extras"],
        "size_gb": 22.6,
        "description_suffix": "*Adds additional Qwen Lightning LoRAs on top of the Qwen core install.*"},
    '7': {"name": "[QWEN NUNCHAKU] Nunchaku Nodes + Model",
        "tags": ["qwen_core", "qwen_nunchaku"],
        "size_gb": 33.0,
        "description_suffix": "*Installs Qwen Core components plus Nunchaku nodes and the Int4 quantized model (12.7GB).*"},
    '8': {"name": "[TRELLIS.2] Image-to-3D Node",
        "tags": ["trellis2"],
        "size_gb": 0.1,
        "description_suffix": "*Installs ComfyUI-TRELLIS2 custom node. Models are downloaded automatically on first use by the node.*\n"
                              "    *LICENSE NOTICE: Only the 'Native (TRELLIS.2)' texture mode uses nvdiffrast/nvdiffrec*\n"
                              "    *(NVIDIA Source Code License — non-commercial use only). See README.md for full details.*",
    },
    '9': {"name": "[PBR DECOMPOSITION] Marigold IID Node",
        "tags": ["pbr_marigold"],
        "size_gb": 0.01,
        "description_suffix": "*Installs ComfyUI-Marigold custom node for PBR decomposition (albedo, roughness, metallic).*\n"
                              "    *IID models (~2GB each) are downloaded automatically from HuggingFace on first use.*",
    },
    '10': {"name": "[PBR DECOMPOSITION] StableDelight Node + Model",
        "tags": ["pbr_marigold", "pbr_stabledelight"],
        "size_gb": 3.3,
        "description_suffix": "*Installs ComfyUI_StableDelight_ll custom node + downloads the*\n"
                              "    *Stable-X/yoso-delight-v0-4-base model (~3.3GB fp16) for specular-free albedo.*",
    },
    '11': {"name": "[FLUX.2 KLEIN] Klein 4B FP8 + Qwen 3 Text Encoder + VAE",
        "tags": ["flux2_klein"],
        "size_gb": 12.4,
        "description_suffix": "*Downloads FLUX.2 Klein 4B FP8 diffusion model (~4.1GB), Qwen 3 4B text encoder bf16 (~8.0GB),*\n"
                              "    *and FLUX.2 VAE (~0.3GB). All ComfyUI nodes are built-in (no custom nodes needed).*\n"
                              "    *Apache 2.0 license. Requires ~13GB VRAM.*",
    },
}

# --- Helper Functions ---
def print_header(title: str):
    print(f"\n{'='*10} {title} {'='*10}")

def print_separator(char='-', length=70):
    print(char * length)

def get_comfyui_path_from_args() -> Path:
    parser = argparse.ArgumentParser(description="StableGen Dependency Installer Script.")
    parser.add_argument("comfyuipath", nargs='?', default=None,
                        help="Full path to your ComfyUI installation directory. If not provided, will be prompted.")
    args = parser.parse_args()

    comfyui_path_str = args.comfyuipath
    while not comfyui_path_str:
        comfyui_path_str = input("Please enter the full path to your ComfyUI installation directory: ").strip()

    comfyui_path = Path(comfyui_path_str).resolve() # Get absolute path

    if not comfyui_path.is_dir():
        print(f"Error: ComfyUI path '{comfyui_path}' not found or not a directory.")
        sys.exit(1)
    if not (comfyui_path / "models").is_dir() or not (comfyui_path / "custom_nodes").is_dir():
        print(f"Error: '{comfyui_path}' does not look like a valid ComfyUI directory (missing 'models' or 'custom_nodes' subfolder).")
        sys.exit(1)
    return comfyui_path

def find_comfyui_python(comfyui_path: Path) -> str:
    """Detect the Python executable used by ComfyUI.

    Checks (in order):
      1. Windows portable: python_embedded/python.exe
      2. Virtual-env:      venv/Scripts/python.exe  or  venv/bin/python
      3. Fallback:         the Python running this script (sys.executable)
    """
    # Windows portable build
    embedded = comfyui_path / "python_embedded" / "python.exe"
    if embedded.is_file():
        return str(embedded)
    # venv (Windows)
    venv_win = comfyui_path / "venv" / "Scripts" / "python.exe"
    if venv_win.is_file():
        return str(venv_win)
    # venv (Linux / macOS)
    venv_unix = comfyui_path / "venv" / "bin" / "python"
    if venv_unix.is_file():
        return str(venv_unix)
    # Fallback
    return sys.executable


def install_pip_packages(pip_packages: List[str], comfyui_path: Path):
    """Install pip packages into ComfyUI's Python environment."""
    python_exe = find_comfyui_python(comfyui_path)
    print(f"  Installing pip packages into ComfyUI Python: {python_exe}")
    for pkg in pip_packages:
        print(f"    pip install {pkg} ...")
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install", pkg],
                check=True,
            )
            print(f"    Successfully installed '{pkg}'.")
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: Failed to install '{pkg}' (exit code {e.returncode}).")
            print(f"    You may need to install it manually: pip install {pkg}")
        except FileNotFoundError:
            print(f"    ERROR: Python executable not found at '{python_exe}'.")
            print(f"    Please install '{pkg}' manually into your ComfyUI Python environment.")
            break


def _patch_comfy_env_platform_tag(comfyui_path: Path):
    """Fix comfy-env 0.2.0 Linux platform tag matching bug.

    comfy-env's _platform_tag() returns "linux_x86_64" but pre-built CUDA
    wheels use "manylinux_2_34_x86_64" filenames.  The substring check
    ``"linux_x86_64" in "manylinux_2_34_x86_64"`` fails because the glibc
    version sits between "linux_" and "x86_64".

    Fix: return "linux" instead, which IS a substring of "manylinux_*".
    """
    python_exe = find_comfyui_python(comfyui_path)
    result = subprocess.run(
        [python_exe, "-c",
         "import comfy_env.packages.cuda_wheels as m; print(m.__file__)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  WARNING: Could not locate comfy_env cuda_wheels.py, "
              "skipping platform tag patch")
        return

    cuda_wheels_path = Path(result.stdout.strip())
    if not cuda_wheels_path.is_file():
        print(f"  WARNING: {cuda_wheels_path} not found, "
              "skipping platform tag patch")
        return

    content = cuda_wheels_path.read_text(encoding="utf-8")
    marker = "# StableGen patch: linux platform tag"
    if marker in content:
        print("  comfy-env platform tag patch already applied")
        return

    old = 'return "linux_x86_64"'
    if old not in content:
        print("  WARNING: Could not find platform tag to patch in "
              "cuda_wheels.py (already fixed upstream?)")
        return

    content = content.replace(old, f'return "linux"  {marker}')
    cuda_wheels_path.write_text(content, encoding="utf-8")
    print("  Patched comfy-env _platform_tag() for Linux manylinux matching")


def _patch_comfy_env_wheel_fallback(comfyui_path: Path):
    """Fix comfy-env get_wheel_url missing torch version fallback.

    Some packages (e.g. flash-attn) lack wheels for the exact torch version
    comfy-env pins (e.g. torch 2.4 on Linux, though torch 2.5+ exist).
    Patch get_wheel_url to try higher torch versions when exact match fails.
    """
    python_exe = find_comfyui_python(comfyui_path)
    result = subprocess.run(
        [python_exe, "-c",
         "import comfy_env.packages.cuda_wheels as m; print(m.__file__)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  WARNING: Could not locate comfy_env cuda_wheels.py, "
              "skipping wheel fallback patch")
        return

    cuda_wheels_path = Path(result.stdout.strip())
    if not cuda_wheels_path.is_file():
        print(f"  WARNING: {cuda_wheels_path} not found, "
              "skipping wheel fallback patch")
        return

    content = cuda_wheels_path.read_text(encoding="utf-8")
    marker = "# StableGen patch: torch version fallback"
    if marker in content:
        print("  comfy-env wheel fallback patch already applied")
        return

    anchor = "    return None\n\n\ndef find_available_wheels"
    if anchor not in content:
        print("  WARNING: Could not find get_wheel_url anchor in "
              "cuda_wheels.py (already fixed upstream?)")
        return

    fallback = "\n".join([
        "    " + marker,
        "    _cu_re = re.compile(r'\\+cu' + cuda_short + r'torch(\\d+)')",
        "    for pkg_dir in _pkg_variants(package):",
        "        try:",
        '            with urllib.request.urlopen(f"{CUDA_WHEELS_INDEX}{pkg_dir}/", timeout=10) as resp:',
        '                html = resp.read().decode("utf-8")',
        "        except Exception: continue",
        "        _best_url, _best_tv = None, None",
        "        for match in link_pattern.finditer(html):",
        "            wheel_url, display = match.group(1), match.group(2)",
        "            _tv_m = _cu_re.search(display)",
        "            if not _tv_m: continue",
        "            _tv = int(_tv_m.group(1))",
        "            if _tv <= int(torch_short): continue",
        "            if py_tag not in display: continue",
        "            if platform_tag and platform_tag not in display: continue",
        "            if _best_tv is None or _tv < _best_tv:",
        "                _best_tv = _tv",
        '                _best_url = wheel_url if wheel_url.startswith("http") else f"{CUDA_WHEELS_INDEX}{pkg_dir}/{wheel_url}"',
        "        if _best_url:",
        "            return _best_url",
    ]) + "\n"

    content = content.replace(anchor, fallback + anchor)
    cuda_wheels_path.write_text(content, encoding="utf-8")
    print("  Patched comfy-env get_wheel_url() for torch version fallback")


def _patch_comfy_env_user_site_isolation(comfyui_path: Path):
    """Prevent user site-packages from leaking into comfy-env workers.

    comfy-env's build_isolation_env() never sets PYTHONNOUSERSITE, so
    packages installed in ~/.local/lib/pythonX.Y/site-packages/ leak into
    the isolated worker subprocess.  Old versions of scipy, tensorboard,
    etc. from user site-packages then clash with the env's numpy 2.x.

    Fix: set PYTHONNOUSERSITE=1 in build_isolation_env() so the worker
    subprocess ignores the user site-packages directory entirely.
    """
    python_exe = find_comfyui_python(comfyui_path)
    result = subprocess.run(
        [python_exe, "-c",
         "import comfy_env.isolation.wrap as m; print(m.__file__)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  WARNING: Could not locate comfy_env wrap.py, "
              "skipping user site isolation patch")
        return

    wrap_path = Path(result.stdout.strip())
    if not wrap_path.is_file():
        print(f"  WARNING: {wrap_path} not found, "
              "skipping user site isolation patch")
        return

    content = wrap_path.read_text(encoding="utf-8")
    marker = "# StableGen patch: block user site-packages"
    if marker in content:
        print("  comfy-env user site isolation patch already applied")
        return

    anchor = '    env["COMFYUI_ISOLATION_WORKER"] = "1"'
    if anchor not in content:
        print("  WARNING: Could not find build_isolation_env anchor in "
              "wrap.py (already fixed upstream?)")
        return

    patch_line = f'    env["PYTHONNOUSERSITE"] = "1"  {marker}'
    content = content.replace(anchor, anchor + "\n" + patch_line)
    wrap_path.write_text(content, encoding="utf-8")
    print("  Patched comfy-env build_isolation_env() to block user site-packages")


def _patch_flex_gemm_autotuner(repo_path: Path):
    """Fix flex_gemm TritonPersistentCacheAutotuner for triton 3.0+.

    flex_gemm 1.0.0 passes a ``do_bench`` argument to triton's
    ``Autotuner.__init__`` which was removed in triton 3.0.  Drop it so
    the call signature matches.
    """
    import glob as _glob
    env_dirs = _glob.glob(str(repo_path / "nodes" / "_env_*"))
    if not env_dirs:
        return
    for env_dir in env_dirs:
        matches = _glob.glob(
            str(Path(env_dir) / "lib" / "python*" / "site-packages"
                / "flex_gemm" / "utils" / "autotuner.py"))
        for autotuner_path in matches:
            autotuner_path = Path(autotuner_path)
            content = autotuner_path.read_text(encoding="utf-8")
            marker = "# StableGen patch: drop do_bench for triton 3.0+"
            if marker in content:
                print("  flex_gemm autotuner patch already applied")
                continue
            old = (
                "            warmup,\n"
                "            rep,\n"
                "            use_cuda_graph,\n"
                "            do_bench,\n"
                "        )"
            )
            if old not in content:
                print("  WARNING: Could not find do_bench in flex_gemm "
                      "autotuner.py (already fixed upstream?)")
                continue
            new = (
                "            warmup if warmup is not None else 25,  # StableGen patch: triton 3.0 needs int\n"
                "            rep if rep is not None else 100,\n"
                "            use_cuda_graph,\n"
                f"        )  {marker}\n"
                "        if not hasattr(self, 'keys'):  # StableGen patch: triton 3.0 compat\n"
                "            self.keys = key"
            )
            content = content.replace(old, new)
            autotuner_path.write_text(content, encoding="utf-8")
            print("  Patched flex_gemm autotuner: removed do_bench for triton 3.0+")


def run_node_install_script(item_details: Dict[str, Any], comfyui_path: Path):
    """Run a custom node's install.py to resolve all its dependencies.

    Some nodes (e.g. ComfyUI-TRELLIS2) use comfy_env which manages CUDA
    wheel installation from custom indices.  Running their install.py
    after cloning lets comfy_env handle everything automatically.

    The first attempt may fail if the auto-detected CUDA version (e.g.
    12.8) lacks wheels on the cuda-wheels index.  The node's install.py
    also downgrades comfy-env to 0.1.92 (via requirements.txt of both
    TRELLIS2 and GeometryPack), wiping our patches from the on-disk
    package.  Within a single subprocess the in-memory modules survive,
    but a retry needs freshly re-applied patches on disk.
    """
    repo_path = comfyui_path / item_details["target_dir_relative"] / item_details["repo_name"]
    install_script = repo_path / "install.py"
    if not install_script.is_file():
        print(f"  WARNING: No install.py found in '{repo_path}'. Skipping dependency install.")
        return

    python_exe = find_comfyui_python(comfyui_path)
    print(f"  Running install.py for {item_details['name']}...")
    print(f"    {python_exe} {install_script}")

    try:
        subprocess.run(
            [python_exe, str(install_script)],
            check=True,
            cwd=str(repo_path),
        )
        print(f"  Successfully ran install.py for '{item_details['name']}'.")
    except subprocess.CalledProcessError:
        # The cuda-wheels index may not have wheels for the auto-detected
        # CUDA version yet (e.g. cu128).  Retry with CUDA 12.4 which has
        # the widest wheel coverage for flex_gemm / nvdiffrast.
        #
        # The first attempt's requirements.txt installs downgraded
        # comfy-env to 0.1.92 on disk (patches wiped).  Restore 0.2.0
        # with patches before retrying so the new subprocess sees them.
        print("  Install failed. Restoring comfy-env 0.2.0 + patches for retry...")
        install_pip_packages(["comfy-env==0.2.0"], comfyui_path)
        _patch_comfy_env_platform_tag(comfyui_path)
        _patch_comfy_env_wheel_fallback(comfyui_path)
        _patch_comfy_env_user_site_isolation(comfyui_path)

        print("  Retrying with CUDA 12.4 fallback...")
        env = os.environ.copy()
        env["COMFY_ENV_CUDA_VERSION"] = "12.4"
        try:
            subprocess.run(
                [python_exe, str(install_script)],
                check=True,
                cwd=str(repo_path),
                env=env,
            )
            print(f"  Successfully ran install.py for '{item_details['name']}' (CUDA 12.4 fallback).")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: install.py failed (exit code {e.returncode}).")
            print(f"  You may need to run it manually:")
            print(f"    cd {repo_path}")
            print(f"    {python_exe} install.py")
    except FileNotFoundError:
        print(f"  ERROR: Python executable not found at '{python_exe}'.")


def apply_post_clone_patches(item_details: Dict[str, Any], comfyui_path: Path):
    """Apply compatibility patches to a cloned custom node repository.

    Patches are defined in the dependency entry under 'post_clone_patches'.
    Each patch dict has:
      file    – relative path inside the repo
      marker  – string checked to avoid re-applying
      mode    – 'insert' (default) or 'replace'

    Insert mode (default):
      anchor  – text before which the patch is inserted
      patch   – code string to insert

    Replace mode:
      anchor  – exact text to replace
      patch   – replacement text
    """
    patches = item_details.get("post_clone_patches")
    if not patches:
        return

    repo_path = (comfyui_path / item_details["target_dir_relative"]
                 / item_details["repo_name"])

    for p in patches:
        target_file = repo_path / p["file"]
        if not target_file.is_file():
            print(f"  WARNING: Patch target '{p['file']}' not found. Skipping.")
            continue

        content = target_file.read_text(encoding="utf-8")

        # Already patched?
        marker = p.get("marker", "")
        if marker and marker in content:
            print(f"  Compat patch already applied to '{p['file']}'.")
            continue

        anchor = p.get("anchor", "")
        patch_code = p.get("patch", "")
        mode = p.get("mode", "insert")

        if anchor and anchor in content:
            if mode == "replace":
                content = content.replace(anchor, patch_code, 1)
            else:
                content = content.replace(anchor, patch_code + anchor, 1)
            target_file.write_text(content, encoding="utf-8")
            print(f"  Applied compat patch ({mode}) to '{p['file']}'.")
        else:
            print(f"  WARNING: Anchor not found in '{p['file']}'. Patch skipped.")


def create_dir_if_not_exists(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _get_remote_file_size(url: str) -> int:
    """Return the remote file's total size in bytes via a HEAD request.
    Returns 0 if the server does not provide Content-Length.
    """
    try:
        r = requests.head(url, allow_redirects=True, timeout=15)
        if r.status_code == 200:
            return int(r.headers.get('content-length', 0))
    except Exception:
        pass
    return 0


def _rename_if_needed(item_details: Dict[str, Any], temp_filepath: Path, final_filepath: Path):
    """Move temp_filepath -> final_filepath when rename_from is set."""
    if item_details.get("rename_from") and temp_filepath != final_filepath:
        if final_filepath.exists():
            final_filepath.unlink()
        shutil.move(str(temp_filepath), str(final_filepath))
        print(f"  Renamed to '{final_filepath.name}'.")


def download_file(item_details: Dict[str, Any], comfyui_path: Path):
    url = item_details["url"]
    target_dir = comfyui_path / item_details["target_path_relative"]
    final_filename = item_details["filename"]
    final_filepath = target_dir / final_filename
    temp_filename = item_details.get("rename_from", final_filename)
    temp_filepath = target_dir / temp_filename
    # File we actually write to during the download
    download_target = temp_filepath if item_details.get("rename_from") else final_filepath

    create_dir_if_not_exists(target_dir)

    # ------------------------------------------------------------------
    # Step 1: Check whether the final file already exists and is complete.
    # ------------------------------------------------------------------
    if final_filepath.exists():
        remote_size = _get_remote_file_size(url)
        local_size = final_filepath.stat().st_size
        if remote_size > 0:
            if local_size >= remote_size:
                print(f"INFO: '{final_filename}' already complete "
                      f"({local_size / 1024**2:.1f} MB). Skipping.")
                return
            else:
                print(f"WARNING: '{final_filename}' is incomplete "
                      f"({local_size / 1024**2:.1f} / {remote_size / 1024**2:.1f} MB). "
                      f"Will attempt to resume.")
                # If download_target differs from final_filepath (rename_from case),
                # the partial final file is unusable — delete it and resume via
                # download_target instead.  If they are the same file, keep it for resume.
                if download_target != final_filepath:
                    final_filepath.unlink()
        else:
            # Server did not return Content-Length on HEAD; trust the file's existence.
            print(f"INFO: '{final_filename}' exists "
                  f"(remote size unavailable, assuming complete). Skipping.")
            print(f"      License: {item_details['license']}")
            return

    # ------------------------------------------------------------------
    # Step 2: Check for an existing partial download to resume from.
    # ------------------------------------------------------------------
    existing_bytes = 0
    if download_target.exists():
        existing_bytes = download_target.stat().st_size
        if existing_bytes > 0:
            print(f"  Found partial file '{download_target.name}' "
                  f"({existing_bytes / 1024**2:.1f} MB already downloaded). "
                  f"Attempting to resume...")

    size_mb = item_details.get("size_mb", "N/A")
    print(f"  Downloading: {item_details['name']} (~{size_mb} MB)"
          f" - License: {item_details['license']}")
    print(f"  From: {url}")
    print(f"  To:   {final_filepath}")
    if item_details.get("rename_from"):
        print(f"  (Downloading as '{temp_filename}', will rename to '{final_filename}')")

    try:
        req_headers = {}
        if existing_bytes > 0:
            req_headers['Range'] = f'bytes={existing_bytes}-'

        response = requests.get(url, stream=True, timeout=30, headers=req_headers)

        # 416 Range Not Satisfiable: our offset is already at or beyond the file end.
        if response.status_code == 416:
            print(f"  File is already fully downloaded on disk.")
            _rename_if_needed(item_details, temp_filepath, final_filepath)
            return

        if response.status_code == 206:
            # Server supports partial content — append to the existing partial file.
            open_mode = 'ab'
            content_range = response.headers.get('Content-Range', '')
            total_size = int(content_range.split('/')[-1]) if '/' in content_range else 0
        elif response.status_code == 200:
            # Server returned the full file (no resume support, or no Range was sent).
            if existing_bytes > 0:
                print(f"  Server does not support resume. Restarting download from scratch.")
                existing_bytes = 0
            open_mode = 'wb'
            total_size = int(response.headers.get('content-length', 0))
        else:
            response.raise_for_status()
            return

        with open(download_target, open_mode) as f:
            if TQDM_AVAILABLE and total_size > 0:
                with tqdm(total=total_size, initial=existing_bytes,
                          unit='B', unit_scale=True, unit_divisor=1024,
                          desc=item_details['name'], ascii=True) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                downloaded = existing_bytes
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        print(f"\r  {downloaded / 1024**2:.0f} / "
                              f"{total_size / 1024**2:.0f} MB "
                              f"({pct:.0f}%)", end='', flush=True)
                if total_size > 0:
                    print()

        print(f"  Download complete: {download_target.name}")
        _rename_if_needed(item_details, temp_filepath, final_filepath)

    except requests.exceptions.RequestException as e:
        print(f"  ERROR downloading {item_details['name']}: {e}")
    except Exception as e:
        print(f"  An unexpected error occurred during download of {item_details['name']}: {e}")


def download_hf_model(item_details: Dict[str, Any], comfyui_path: Path):
    """Download a HuggingFace diffusers model repo into ComfyUI/models/diffusers/.

    Uses ``huggingface_hub.snapshot_download`` if available, otherwise falls
    back to ``huggingface-cli download``.

    Required keys in *item_details*:
        hf_repo   – e.g. ``'Stable-X/yoso-delight-v0-4-base'``
        local_dir – directory name under ``models/diffusers/``,
                    e.g. ``'Stable-X--yoso-delight-v0-4-base'``
    """
    hf_repo = item_details["hf_repo"]
    local_dir_name = item_details["local_dir"]
    target_dir = comfyui_path / "models" / "diffusers" / local_dir_name

    if target_dir.is_dir() and (target_dir / "model_index.json").exists():
        print(f"INFO: HuggingFace model '{hf_repo}' already exists at '{target_dir}'. Skipping.")
        return

    create_dir_if_not_exists(target_dir.parent)
    print(f"  Downloading HuggingFace model: {hf_repo} (~{item_details.get('size_mb', '?')} MB)")
    print(f"  License: {item_details.get('license', 'unknown')}")
    print(f"  Destination: {target_dir}")

    # Try Python API first
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            # Download config files + fp16 weight variant only (skip fp32)
            allow_patterns=[
                "*.json", "*.txt", "*.yaml", "*.md", "*.py", "*.model",
                "*.fp16.safetensors", "*.fp16.bin",
                "tokenizer/**",
            ],
        )
        print(f"  HuggingFace model download complete: {local_dir_name}")
        return
    except ImportError:
        print("  huggingface_hub not available, trying CLI fallback...")
    except Exception as e:
        print(f"  snapshot_download failed: {e}. Trying CLI fallback...")

    # Fallback: huggingface-cli
    try:
        python_exe = find_comfyui_python(comfyui_path)
        cmd = [
            str(python_exe), "-m", "huggingface_hub.commands.huggingface_cli",
            "download", hf_repo,
            "--local-dir", str(target_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"  HuggingFace model download complete: {local_dir_name}")
        else:
            print(f"  ERROR downloading HuggingFace model: {result.stderr}")
    except Exception as e:
        print(f"  ERROR downloading HuggingFace model '{hf_repo}': {e}")
        print(f"  You can manually download it with:")
        print(f"    huggingface-cli download {hf_repo} --local-dir {target_dir}")


def _clean_stale_comfy_envs(repo_path: Path):
    """Remove stale comfy-env virtualenvs (_env_*) from a node tree.

    comfy-env creates isolated virtualenvs named _env_<hash> based on the
    comfy-env.toml dependency manifest.  When the node is switched to a
    different commit whose manifest lists different packages (e.g. timm
    was added), the OLD _env_* dirs are never rebuilt and keep serving
    outdated packages.  Deleting them forces comfy-env to recreate
    correct environments on the next ComfyUI startup.
    """
    removed = 0
    for env_dir in repo_path.rglob("_env_*"):
        if env_dir.is_dir():
            try:
                shutil.rmtree(str(env_dir))
                print(f"  Removed stale comfy-env '{env_dir.name}' (will be rebuilt on next launch).")
                removed += 1
            except Exception as e:
                print(f"  WARNING: Could not remove '{env_dir}': {e}")
    if removed:
        print(f"  Cleaned {removed} stale environment(s). comfy-env will recreate them on next ComfyUI start.")


def clone_git_repo(item_details: Dict[str, Any], comfyui_path: Path):
    git_url = item_details["git_url"]
    target_parent_dir = comfyui_path / item_details["target_dir_relative"]
    repo_name = item_details["repo_name"]
    final_repo_path = target_parent_dir / repo_name
    pinned_commit = item_details.get("commit")

    if final_repo_path.is_dir():
        # If a commit is pinned, verify the checkout matches
        if pinned_commit:
            try:
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                    check=True, cwd=str(final_repo_path)
                )
                current = result.stdout.strip()
                if current != pinned_commit:
                    print(f"  Pinning '{repo_name}' to verified commit {pinned_commit[:12]}...")
                    subprocess.run(['git', 'fetch', 'origin'], check=True, cwd=str(final_repo_path))
                    subprocess.run(['git', 'checkout', pinned_commit], check=True, cwd=str(final_repo_path))
                    print(f"  Successfully pinned to {pinned_commit[:12]}.")
                    # Commit changed — old comfy-env virtualenvs may have stale deps
                    if item_details.get("clean_envs"):
                        _clean_stale_comfy_envs(final_repo_path)
                else:
                    print(f"INFO: '{repo_name}' already at pinned commit {pinned_commit[:12]}. Skipping.")
            except Exception as e:
                print(f"  WARNING: Could not verify/pin commit for '{repo_name}': {e}")
        else:
            print(f"INFO: Custom node directory '{repo_name}' already exists at '{final_repo_path}'. Skipping clone.")
            print(f"      Please ensure it's the correct repository and up-to-date if you encounter issues (License: {item_details['license']}).")
        return

    create_dir_if_not_exists(target_parent_dir)
    print(f"  Cloning: {item_details['name']} - License: {item_details['license']}")
    print(f"  From: {git_url}")
    print(f"  To:   {final_repo_path}")
    if pinned_commit:
        print(f"  Pinned commit: {pinned_commit[:12]}")

    try:
        subprocess.run(['git', 'clone', git_url, str(final_repo_path)], check=True, cwd=str(target_parent_dir))

        # Checkout pinned commit if specified
        if pinned_commit:
            subprocess.run(['git', 'checkout', pinned_commit], check=True, cwd=str(final_repo_path))
            print(f"  Successfully cloned and pinned '{repo_name}' to {pinned_commit[:12]}.")
        else:
            print(f"  Successfully cloned '{repo_name}'.")

        print(f"  IMPORTANT: Restart ComfyUI if it was running to load this new custom node.")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR cloning {item_details['name']}: Git command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"  Git stderr: {e.stderr.decode(errors='ignore')}")
        if e.stdout:
            print(f"  Git stdout: {e.stdout.decode(errors='ignore')}")

    except FileNotFoundError:
        print("  ERROR: Git command not found. Please ensure Git is installed and in your system's PATH.")
    except Exception as e:
        print(f"  An unexpected error occurred during git clone of {item_details['name']}: {e}")


def display_menu(comfyui_path: Path):
    print_header("StableGen Dependency Installer")
    print(f"Target ComfyUI Directory: {comfyui_path}")
    print_separator()
    print("This script helps download and set up essential and optional models for StableGen.")
    print("For FLUX.1 model setup (which requires manual download due to licensing),")
    print("please refer to the main README.md on our GitHub page: https://github.com/sakalond/stablegen")
    print_separator()
    print("Note: Download sizes are estimates. Actual sizes may vary slightly.")
    print()
    print("Upon entering a choice, the full list of components will be shown for approval before proceeding.")
    print_separator()
    print("Please choose an installation package:")
    print()

    for key, val in MENU_PACKAGES.items():
        print(f"\n{key}. {val['name']}")
        print_separator(char='.')
        # Dynamically list contents for clarity (optional, can make menu long)
        # items_in_package = get_items_for_package_tags(val['tags'])
        # for item_id in items_in_package:
        #     print(f"    - {DEPENDENCIES[item_id]['name']}")
        print(f"    *Approximate total download size: ~{val['size_gb']:.1f} GB*")
        if val.get("description_suffix"):
            print(f"    {val['description_suffix']}")
    print("\nq. Quit")
    print_separator()


def get_unique_item_ids_for_tags(selected_tags: List[str]) -> Set[str]:
    item_ids: Set[str] = set()
    for item_id, details in DEPENDENCIES.items():
        # An item is included if any of its 'packages' tags are in selected_tags
        if any(tag in details["packages"] for tag in selected_tags):
            item_ids.add(item_id)
    return item_ids

# --- Main Script Logic ---
def main():
    comfyui_base_path = get_comfyui_path_from_args()

    processed_during_this_session: Set[str] = set() # To avoid re-processing in same session if menu is re-shown

    while True:
        display_menu(comfyui_base_path)
        choice = input("Enter your choice (1-12, or q to quit): ").strip().lower()

        if choice == 'q':
            print("Exiting installer.")
            break

        selected_option = MENU_PACKAGES.get(choice)
        if not selected_option:
            print("Invalid choice. Please try again.")
            continue

        print_separator()
        print(f"You selected: {selected_option['name']}")
        print(f"Estimated download for this package (if items not present): ~{selected_option['size_gb']:.1f} GB")
        
        current_selection_item_ids = get_unique_item_ids_for_tags(selected_option["tags"])
        items_to_process_this_round: List[Dict[str, Any]] = []
        
        print("This package will check/install the following components:")
        for item_id in sorted(list(current_selection_item_ids)): # Sort for consistent display
            if item_id not in processed_during_this_session and item_id in DEPENDENCIES:
                print(f"  - {DEPENDENCIES[item_id]['name']}")
                items_to_process_this_round.append(DEPENDENCIES[item_id])
        
        if not items_to_process_this_round:
            print("All components for this selection appear to have been processed or are not defined. Nothing new to install for this choice.")
            print("If you expected installations, ensure the components are not already present from a previous run or selection.")
            print_separator()
            continue

        confirm_all = input("Proceed with checking/installing these components? (y/n, Enter for yes): ").strip().lower()
        if not (confirm_all == "" or confirm_all == 'y'):
            print("Installation of this package cancelled by user.")
            print_separator()
            continue

        # Display TRELLIS.2 license notice if installing that package
        if "trellis2" in selected_option["tags"]:
            print_separator(char='!')
            print("TRELLIS.2 THIRD-PARTY LICENSE NOTICE")
            print_separator(char='!')
            print("NOTE: This notice applies ONLY to the TRELLIS.2 Image-to-3D feature.")
            print("StableGen's standard texturing pipelines (SDXL, FLUX, Qwen) are unaffected.")
            print()
            print("ComfyUI-TRELLIS2 and the TRELLIS.2 model are MIT-licensed.")
            print("However, the TRELLIS.2 TEXTURED mesh pipeline uses the following NVIDIA")
            print("libraries that are restricted to NON-COMMERCIAL use only:")
            print()
            print("  - nvdiffrast  (NVIDIA Source Code License, 1-Way Commercial)")
            print("  - nvdiffrec   (NVIDIA Source Code License)")
            print()
            print("These licenses restrict usage to 'research or evaluation purposes only")
            print("and not for any direct or indirect monetary gain' (Section 3.3).")
            print("Only NVIDIA and its affiliates may use them commercially.")
            print()
            print("These libraries are ONLY used when Texture Mode is set to")
            print("'Native (TRELLIS.2)' for UV rasterization and PBR texture baking.")
            print()
            print("All other modes do not introduce additional NVIDIA restrictions:")
            print("  - Texture Mode 'None' (shape-only) - permissively licensed")
            print("  - Projection texture modes (SDXL, Flux, Qwen) - the selected")
            print("    diffusion model's own license applies as usual")
            print()
            print("For full details, see the README.md License section.")
            print_separator(char='!')
            ack = input("Acknowledge and continue? (y/n, Enter for yes): ").strip().lower()
            if not (ack == "" or ack == 'y'):
                print("Installation cancelled.")
                print_separator()
                continue

        print_separator(char='*')
        for item_details in items_to_process_this_round:
            if item_details["id"] in processed_during_this_session: # Should not happen if items_to_process_this_round is built correctly
                continue

            print_separator(char='.')
            print(f"Processing: {item_details['name']}")
            
            target_full_path: Path
            if item_details["type"] == "node":
                target_full_path = comfyui_base_path / item_details["target_dir_relative"] / item_details["repo_name"]
                clone_git_repo(item_details, comfyui_base_path)
                # Install pip dependencies required by this custom node
                if item_details.get("pip_packages"):
                    install_pip_packages(item_details["pip_packages"], comfyui_base_path)
                # Apply comfy-env patches BEFORE running install.py,
                # since the install script needs them (e.g. platform_tag for Linux wheel lookup)
                if item_details.get("pip_packages") and any("comfy-env" in p for p in item_details["pip_packages"]):
                    _patch_comfy_env_platform_tag(comfyui_base_path)
                    _patch_comfy_env_wheel_fallback(comfyui_base_path)
                    _patch_comfy_env_user_site_isolation(comfyui_base_path)
                # Apply compatibility patches (e.g. Python 3.10 polyfills)
                if item_details.get("post_clone_patches"):
                    apply_post_clone_patches(item_details, comfyui_base_path)
                # Run node's install.py for full dependency resolution (e.g. CUDA wheels)
                if item_details.get("run_install_script"):
                    run_node_install_script(item_details, comfyui_base_path)
                    # Fix flex_gemm/triton version mismatch in comfy-env isolated envs
                    _patch_flex_gemm_autotuner(target_full_path)
                # Force comfy-env back to 0.2.0 and re-apply patches AFTER install.py,
                # since the node's requirements.txt may pin an older version
                if item_details.get("pip_packages") and any("comfy-env" in p for p in item_details["pip_packages"]):
                    print("  Re-installing comfy-env==0.2.0 (overriding node requirements.txt)...")
                    install_pip_packages(["comfy-env==0.2.0"], comfyui_base_path)
                    _patch_comfy_env_platform_tag(comfyui_base_path)
                    _patch_comfy_env_wheel_fallback(comfyui_base_path)
                    _patch_comfy_env_user_site_isolation(comfyui_base_path)
            elif item_details["type"] == "model":
                target_full_path = comfyui_base_path / item_details["target_path_relative"] / item_details["filename"]
                download_file(item_details, comfyui_base_path)
            elif item_details["type"] == "hf_model":
                target_full_path = comfyui_base_path / "models" / "diffusers" / item_details["local_dir"]
                download_hf_model(item_details, comfyui_base_path)
            
            processed_during_this_session.add(item_details["id"])
        
        print_separator(char='*')
        print("Processing for selected package complete.")
        print("Please check messages above for status of each item.")
        if any(item["type"] == "node" for item in items_to_process_this_round):
             print("IMPORTANT: If any custom nodes were newly cloned, restart ComfyUI to load them.")
        print_separator()
        print("All done. Exiting installer.")
        break

if __name__ == "__main__":
    if not TQDM_AVAILABLE:
        print("NOTE: 'tqdm' library not found. Download progress will not be shown as a bar.")
        print("      You can install it with: pip install tqdm")
    main()