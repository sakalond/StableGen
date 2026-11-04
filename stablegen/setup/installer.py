#!/usr/bin/env python3
"""StableGen dependency installer: custom nodes and models for a ComfyUI server.

    python installer.py <ComfyUI folder>                      menu
    python installer.py <ComfyUI folder> --packages 3,8 --yes unattended
    python installer.py --list                                 what each package installs

Needs only Python 3.8+ and Git (for custom nodes). `tqdm`, when installed, draws progress bars. The StableGen
add-on ships this file and runs it from Blender; it stays a single file so it can be downloaded on its own.

Machine-readable modes (used by the add-on; nothing asks a question in them):

    --status --json    one JSON document: ComfyUI's Python, GPU, Git, and the state of every package
    --plan --json      one JSON document: what installing --packages would add and change (nothing is changed)
    --events           JSON lines on stdout, each prefixed '@sg ', while installing; --log FILE takes every other
                       line (this script's messages and pip, git and install.py output)

Events, each {"event": <kind>, ...}:
    plan      packages, items [{id, name, type, size_mb, present}], download_mb, free_mb
    item      id, name, phase ('start' or 'done'); on done also status and rows [{name, status, message}]
    progress  id, file, done, total (bytes; at most twice a second per file)
    error     code ('args', 'path', 'python', 'git_missing', 'disk_space', 'notice'), message, plus details
    verify    ok, missing_files, missing_nodes, import_failed
    summary   ok, rows [{name, status, message}], restart (custom nodes changed), dry_run
Statuses are 'installed', 'updated', 'present', 'warning' and 'failed'. License notices are accepted up front
with --accept-notices (keys of LICENSE_NOTICES).
"""
import os
import sys
import argparse
import fnmatch
import hashlib
import json
import socket
import subprocess
import tempfile
import time
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Captured subprocess output is read as UTF-8 and child Pythons are asked to write
# it: the console codepage (cp1251 on a Russian Windows) cannot decode what pip, git
# and a Python print for a non-ASCII path, and the reader thread's crash lost the
# output (#115).
UTF8_TEXT = dict(text=True, encoding="utf-8", errors="replace")
UTF8_ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

# --- StableGen's TRELLIS.2 voxel budget node ---
# ComfyUI's native Trellis2UpsampleStage has no voxel budget yet (Comfy-Org/ComfyUI#17445); the original pipeline
# lowers the 1536 cascade until the upsampled shape fits max_num_tokens. The installer writes this node into
# custom_nodes (no Git needed) and StableGen wires its output into the upsample stage's target_resolution.
_TRELLIS2_BUDGET_NODE = r'''"""StableGen's voxel budget for ComfyUI's native TRELLIS.2 cascade (written by the StableGen installer).

The original TRELLIS.2 pipeline lowers the cascade resolution until the upsampled shape stays under a voxel budget
(max_num_tokens). ComfyUI's native Trellis2UpsampleStage has no budget yet, so StableGen wires this node's
resolution into its target_resolution. If the native helpers it reuses change, it passes the target through.
"""
import logging

from comfy_api.latest import ComfyExtension, IO


class Trellis2TokenBudget(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="StableGenTrellis2TokenBudget",
            display_name="TRELLIS.2 Voxel Budget (StableGen)",
            category="model/conditioning/trellis",
            inputs=[
                IO.Latent.Input("shape_latent", tooltip="The 512 shape latent, as for Trellis2UpsampleStage."),
                IO.Vae.Input("vae"),
                IO.Int.Input("target_resolution", default=1536, min=1024, max=2048, step=128),
                IO.Int.Input("max_tokens", default=49152, min=16384, max=262144, step=4096,
                             tooltip="Sparse-voxel budget of the high-resolution pass (TRELLIS.2's max_num_tokens)."),
                IO.Conditioning.Input("positive", optional=True,
                                      tooltip="Only for Pixal3D, whose coordinates quantize differently."),
            ],
            outputs=[IO.Int.Output(display_name="resolution"), IO.Int.Output(display_name="tokens")],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, shape_latent, vae, target_resolution, max_tokens, positive=None):
        try:
            resolution, tokens = _budget(shape_latent, vae, target_resolution, max_tokens, positive)
        except Exception as e:      # the native helpers moved: run without a budget rather than fail the job
            logging.warning(f"[StableGen] TRELLIS.2 voxel budget unavailable ({e}); using {target_resolution}")
            return IO.NodeOutput(target_resolution, 0)
        note = (f"{target_resolution} -> {resolution}: {tokens:,} voxels" if resolution != target_resolution
                else f"{resolution}: {tokens:,} voxels")
        logging.info(f"[StableGen] TRELLIS.2 voxel budget {note}")
        if cls.hidden.unique_id:
            from server import PromptServer
            PromptServer.instance.send_progress_text(note, cls.hidden.unique_id)
        return IO.NodeOutput(resolution, tokens)


def _budget(shape_latent, vae, target_resolution, max_tokens, positive):
    import comfy.model_management
    from comfy_extras import nodes_trellis2 as T
    device = comfy.model_management.get_torch_device()
    vae.prepare_decode(shape_latent["samples"].shape)
    pixal3d = positive is not None and T._proj_pack_from_conditioning(positive) is not None
    items = T.split_batched_sparse_latent(shape_latent["samples"], shape_latent["coords"],
                                          shape_latent.get("coord_counts"))
    hr = []
    for feats, coords in items:             # the same upsample Trellis2UpsampleStage runs, once
        coords = coords.to(device).clone()
        coords[:, 0] = 0
        slat = T.shape_norm(feats.to(device), coords)
        hr.append(vae.first_stage_model.upsample_shape(slat.to(vae.vae_dtype), upsample_times=4))
    resolution = target_resolution
    while True:                             # TRELLIS.2's loop: cheap, it only re-quantizes the same points
        tokens = max(T.Trellis2UpsampleStage._quantize_unique(c, 512, resolution, pixal3d).shape[0] for c in hr)
        if tokens < max_tokens or resolution <= 1024:
            return resolution, tokens
        resolution -= 128


class StableGenTrellis2Extension(ComfyExtension):
    async def get_node_list(self):
        return [Trellis2TokenBudget]


async def comfy_entrypoint():
    return StableGenTrellis2Extension()
'''

# --- Configuration: Dependencies Data ---
# Sizes are in MB. Every custom node is pinned to a commit StableGen was tested with ("commit"). A fresh clone
# checks out that commit. An existing clone at another commit is left alone with a note, unless the node sets
# "enforce_pin" (StableGen's patches depend on that exact code) or the installer runs with --update-nodes.
# "embedded_node" items are small nodes whose source is in this file; they are written, not cloned.
DEPENDENCIES: Dict[str, Dict[str, Any]] = {
    # --- Custom Nodes ---
    "cn_ipadapter_plus": {
        "id": "cn_ipadapter_plus", "type": "node", "name": "ComfyUI IPAdapter Plus",
        "git_url": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI_IPAdapter_plus",
        "commit": "a0f451a5113cf9becb0847b92884cb10cbdec0ef",
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
        "commit": "6ea2651e7df66d7585f6ffee804b20e92fb38b8a",
        "license": "Apache 2.0", "packages": ["qwen_core"],
        "pip_packages": ["gguf"]
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
        "commit": "b600f879a777fdef1466c362e850f7e8c31bf89a",
        "license": "Apache 2.0", "packages": ["qwen_nunchaku"],
        # The engine is not the PyPI package called "nunchaku" (an unrelated project): it ships as wheels built
        # per Python, torch and platform, picked with the node's own version list.
        "post_install": "nunchaku_wheel",
    },
    "cn_qwen_lora_loader": {
        "id": "cn_qwen_lora_loader", "type": "node", "name": "ComfyUI Qwen Image LoRA Loader",
        "git_url": "https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-QwenImageLoraLoader",
        "commit": "ccb55860b80d41835110609748000f4c64932635",  # v2.6.7
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
        "commit": "2adfec75c430b0751a98a82453dc2dd8077bca85",
        "license": "GPL-3.0", "packages": ["pbr_marigold"],
        "pip_packages": ["diffusers>=0.28", "accelerate", "matplotlib"] + (["triton-windows"] if sys.platform == "win32" else []),
        "run_install_script": True,
    },
    # --- StableDelight (specular-free albedo) ---
    "cn_comfyui_stabledelight": {
        "id": "cn_comfyui_stabledelight", "type": "node", "name": "ComfyUI StableDelight (Delighting)",
        "git_url": "https://github.com/lldacing/ComfyUI_StableDelight_ll.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI_StableDelight_ll",
        "commit": "42c0463cc65554509804cbf577a1675c61499ec4",
        "license": "Apache-2.0", "packages": ["pbr_stabledelight"],
        "pip_packages": ["diffusers>=0.28", "accelerate", "matplotlib"] + (["triton-windows"] if sys.platform == "win32" else []),
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
    # --- Marigold V2 (albedo + normals; ComfyUI core nodes, LoRAs on Qwen-Image-Edit-2509) ---
    "model_marigold_v2_base": {
        "id": "model_marigold_v2_base", "type": "model",
        "name": "Marigold V2 base: Qwen Image Edit 2509 (int8)",
        "url": "https://huggingface.co/Comfy-Org/marigold-v2-0/resolve/main/diffusion_models/qwen_image_edit_2509_int8_convrot.safetensors?download=true",
        "target_path_relative": "models/diffusion_models", "filename": "qwen_image_edit_2509_int8_convrot.safetensors",
        "license": "Apache 2.0", "size_mb": 19550, "packages": ["pbr_marigold_v2"],
    },
    **{
        f"model_marigold_v2_{task}_{kind}": {
            "id": f"model_marigold_v2_{task}_{kind}", "type": "model",
            "name": f"Marigold V2 {task.capitalize()} {label}",
            "url": f"https://huggingface.co/Comfy-Org/marigold-v2-0/resolve/main/{folder}/{filename}?download=true",
            "target_path_relative": f"models/{folder}", "filename": filename,
            "license": "Apache 2.0", "size_mb": size_mb, "packages": ["pbr_marigold_v2"],
        }
        for task in ("albedo", "normals")
        for kind, label, folder, filename, size_mb in (
            ("lora", "LoRA", "loras", f"marigold_v2_{task}.safetensors", 1625),
            ("vae", "VAE", "vae", f"marigold_v2_{task}_vae.safetensors", 242),
            ("cond", "Conditioning", "embeddings", f"marigold_v2_{task}_conditioning.safetensors", 3),
        )
    },
    # --- TRELLIS.2 (ComfyUI's native nodes, ComfyUI 0.42 or newer) ---
    "cn_trellis2_budget": {
        "id": "cn_trellis2_budget", "type": "embedded_node", "name": "StableGen TRELLIS.2 voxel budget node",
        "target_dir_relative": "custom_nodes", "repo_name": "ComfyUI-StableGen-TRELLIS2",
        "files": {"__init__.py": _TRELLIS2_BUDGET_NODE}, "stablegen_node": True,
        "license": "GPL-3.0", "packages": ["trellis2"],
    },
    "model_trellis2": {
        "id": "model_trellis2", "type": "model", "name": "TRELLIS.2 4B Diffusion Model (bf16)",
        "url": "https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/diffusion_models/trellis_2_bf16.safetensors",
        "target_path_relative": "models/diffusion_models", "filename": "trellis_2_bf16.safetensors",
        "license": "MIT", "size_mb": 9860, "packages": ["trellis2"],
    },
    "model_trellis2_shape_vae": {
        "id": "model_trellis2_shape_vae", "type": "model", "name": "TRELLIS.2 Shape VAE",
        "url": "https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/vae/trellis_2_shape_vae_bf16.safetensors",
        "target_path_relative": "models/vae", "filename": "trellis_2_shape_vae_bf16.safetensors",
        "license": "MIT", "size_mb": 1046, "packages": ["trellis2"],
    },
    "model_trellis2_texture_vae": {
        "id": "model_trellis2_texture_vae", "type": "model", "name": "TRELLIS.2 Texture VAE",
        "url": "https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/vae/trellis_2_texture_vae_bf16.safetensors",
        "target_path_relative": "models/vae", "filename": "trellis_2_texture_vae_bf16.safetensors",
        "license": "MIT", "size_mb": 905, "packages": ["trellis2"],
    },
    "model_dino_v3": {
        "id": "model_dino_v3", "type": "model", "name": "DINOv3 ViT-L Image Encoder",
        "url": "https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/clip_vision/dino_v3_vit_l.safetensors",
        "target_path_relative": "models/clip_vision", "filename": "dino_v3_vit_l.safetensors",
        "license": "DINOv3 License", "size_mb": 1157, "packages": ["trellis2"],
    },
    "model_birefnet": {
        "id": "model_birefnet", "type": "model", "name": "BiRefNet Background Removal",
        "url": "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors",
        "target_path_relative": "models/background_removal", "filename": "birefnet.safetensors",
        "license": "MIT", "size_mb": 424, "packages": ["trellis2"],
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
        "name": "FLUX.2 VAE",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors?download=true",
        "target_path_relative": "models/vae",
        "filename": "flux2-vae.safetensors",
        "license": "Apache 2.0", "size_mb": 321, "packages": ["flux2_klein", "fibo_edit"],
    },
    # --- Qwen Image 2.1 (ComfyUI core nodes; StableGen expects these text encoder and VAE file names) ---
    "model_qwen21_diffusion": {
        "id": "model_qwen21_diffusion", "type": "model", "name": "Qwen Image 2.1 Diffusion Model (int8)",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors?download=true",
        "target_path_relative": "models/diffusion_models", "filename": "qwen_image_2.1_int8_convrot.safetensors",
        "license": "Qwen Research License (non-commercial)", "size_mb": 7260, "packages": ["qwen_image_21"],
    },
    "model_qwen21_text_encoder": {
        "id": "model_qwen21_text_encoder", "type": "model", "name": "Qwen Image 2.1 Text Encoder (Qwen3-VL 8B, w4a8)",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_w4a8.safetensors?download=true",
        "target_path_relative": "models/text_encoders", "filename": "qwen3vl_8b_w4a8.safetensors",
        "license": "Qwen Research License (non-commercial)", "size_mb": 6310,
        "packages": ["qwen_image_21", "qwen_image_21_turbo"],
    },
    "model_qwen21_vae": {
        "id": "model_qwen21_vae", "type": "model", "name": "Qwen Image 2.1 VAE",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors?download=true",
        "target_path_relative": "models/vae", "filename": "qwen_image_2.1_vae_bf16.safetensors",
        "license": "Qwen Research License (non-commercial)", "size_mb": 680,
        "packages": ["qwen_image_21", "qwen_image_21_turbo"],
    },
    # --- Experimental architectures (ComfyUI core nodes, Comfy-Org repacks; file names as StableGen loads them) ---
    "model_qwen21_turbo_diffusion": {
        "id": "model_qwen21_turbo_diffusion", "type": "model", "name": "Qwen Image 2.1 Turbo Diffusion Model (int8)",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors?download=true",
        "target_path_relative": "models/diffusion_models", "filename": "qwen_image_2.1_turbo_int8_convrot.safetensors",
        "license": "Qwen Research License (non-commercial)", "size_mb": 6920, "packages": ["qwen_image_21_turbo"],
    },
    "model_moss_diffusion": {
        "id": "model_moss_diffusion", "type": "model", "name": "MOSS-Image Diffusion Model (int8)",
        "url": "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/diffusion_models/moss_image_sft_int8_convrot.safetensors?download=true",
        "target_path_relative": "models/diffusion_models", "filename": "moss_image_sft_int8_convrot.safetensors",
        "license": "Apache 2.0", "size_mb": 5162, "packages": ["moss_image"],
    },
    "model_moss_text_encoder": {
        "id": "model_moss_text_encoder", "type": "model", "name": "MOSS-Image Text Encoder (Qwen3.5 9B, int8)",
        "url": "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors?download=true",
        "target_path_relative": "models/text_encoders", "filename": "qwen3.5_9b_int8_convrot.safetensors",
        "license": "Apache 2.0", "size_mb": 9296, "packages": ["moss_image"],
    },
    "model_moss_vae": {
        "id": "model_moss_vae", "type": "model", "name": "MOSS-Image VAE (FLUX.2 small)",
        "url": "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae_small.safetensors?download=true",
        "target_path_relative": "models/vae", "filename": "flux2-vae_small.safetensors",
        "license": "Apache 2.0", "size_mb": 237, "packages": ["moss_image"],
    },
    "model_moss_depth_control": {
        "id": "model_moss_depth_control", "type": "model", "name": "MOSS-Image Color/Depth Control Adapter",
        "url": "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/loras/moss_image_color_depth_control.safetensors?download=true",
        "target_path_relative": "models/loras", "filename": "moss_image_color_depth_control.safetensors",
        "license": "Apache 2.0", "size_mb": 298, "packages": ["moss_image"],
    },
    "model_fibo_diffusion": {
        "id": "model_fibo_diffusion", "type": "model", "name": "FIBO 2.0 Edit Turbo Diffusion Model (int8)",
        "url": "https://huggingface.co/Comfy-Org/Fibo-2.0/resolve/main/diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors?download=true",
        "target_path_relative": "models/diffusion_models", "filename": "fibo2_edit_turbo_int8_convrot.safetensors",
        "license": "CC BY-NC 4.0 (non-commercial)", "size_mb": 8019, "packages": ["fibo_edit"],
    },
    "model_fibo_text_encoder": {
        "id": "model_fibo_text_encoder", "type": "model", "name": "FIBO 2.0 Text Encoder (Qwen3-VL 4B, int8)",
        "url": "https://huggingface.co/Comfy-Org/Fibo-2.0/resolve/main/text_encoders/qwen3vl_4b_int8_convrot.safetensors?download=true",
        "target_path_relative": "models/text_encoders", "filename": "qwen3vl_4b_int8_convrot.safetensors",
        "license": "CC BY-NC 4.0 (non-commercial)", "size_mb": 4633, "packages": ["fibo_edit"],
    },
    # --- SkinTokens auto-rigging ---
    "cn_skintokens": {
        "id": "cn_skintokens", "type": "node", "name": "ComfyUI StableGen SkinTokens",
        "git_url": "https://github.com/sakalond/ComfyUI-StableGen-SkinTokens.git", "stablegen_node": True,
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-StableGen-SkinTokens",
        "commit": "dd0a272686b77226e6702281b6fd7eb24231abe6",
        "license": "GPL-3.0 (SkinTokens code and weights: MIT)", "packages": ["skintokens"],
        # The node's own setup builds SkinTokens in a separate environment (vendor/skintokens_env) with its own
        # PyTorch and downloads the checkpoints, so ComfyUI's Python is left untouched.
        "post_install": "skintokens_runtime",
    },
    # --- LATO.2 low-poly remeshing (ComfyUI-LATO2) ---
    "cn_lato2": {
        "id": "cn_lato2", "type": "node", "name": "ComfyUI-LATO2 (LATO.2 low-poly remesh)",
        "git_url": "https://github.com/sakalond/ComfyUI-LATO2.git", "stablegen_node": True,
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-LATO2",
        "commit": "b9056ca27e6dbdfe546de4700f98afe8c1e7bcf5",
        "license": "GPL-3.0 (LATO.2 code and weights: MIT; DINOv2: Apache 2.0)", "packages": ["lato2"],
        # The node's own setup builds LATO.2 in a separate environment (envs/lato2: PyTorch 2.8 for CUDA 12.8) with
        # uv and downloads its weights and DINOv2 into models/lato2, so ComfyUI's Python only gets uv.
        "pip_packages": ["uv"],
        "post_install": "lato2_runtime",
        "data_dirs": ["models/lato2"],      # removing the package removes the weights too
        # installed once the setup's last step is done: DINOv2's checkpoint, after the environment and LATO.2's weights
        "ready": ["models/lato2/dinov2/checkpoints/*.pth"],
    },
}

# ComfyUI node types each package provides or needs (checked by --verify against /object_info), and the ComfyUI
# core nodes a package needs, which only exist in recent ComfyUI versions (checked before installing).
PACKAGE_NODES: Dict[str, List[str]] = {
    "core": ["IPAdapterUnifiedLoader", "IPAdapter"],
    "qwen_core": ["UnetLoaderGGUF", "TextEncodeQwenImageEditPlus"],
    "qwen_nunchaku": ["NunchakuQwenImageDiTLoader", "NunchakuQwenImageLoraLoader"],
    "pbr_marigold": ["MarigoldModelLoader", "MarigoldDepthEstimation_v2"],
    "pbr_stabledelight": ["LoadStableDelightModel", "ApplyStableDelight"],
    "pbr_marigold_v2": ["MarigoldV2PostProcess", "ConditioningLoader"],
    "trellis2": ["Trellis2Conditioning", "EmptyTrellis2LatentStructure", "VaeDecodeStructureTrellis2",
    "Trellis2ShapeStage", "Trellis2UpsampleStage", "VaeDecodeShapeTrellis", "Trellis2TextureStage",
    "VaeDecodeTextureTrellis", "RemeshMesh", "DecimateMesh", "MeshSmoothNormals", "UnwrapMesh",
    "BakeTextureFromVoxel", "ApplyTextureToMesh", "SaveGLB", "LoadBackgroundRemovalModel",
    "RemoveBackground", "ImageCropToMask", "StableGenTrellis2TokenBudget"],
    "flux2_klein": ["Flux2Scheduler", "EmptyFlux2LatentImage"],
    "qwen_image_21": ["TextEncodeQwenImage21", "QwenImage21Cache"],
    "skintokens": ["StableGenSkinTokensAutoRig"],
    "lato2": ["LATO2Remesh"],
    "qwen_image_21_turbo": ["TextEncodeQwenImage21", "QwenImage21Cache"],
    "moss_image": ["MOSSImageControl", "EmptyFlux2LatentImage"],
    "fibo_edit": ["TextEncodeFibo2"],
}
CORE_NODES: Dict[str, List[str]] = {
    "qwen_core": ["TextEncodeQwenImageEditPlus"],
    "trellis2": ["Trellis2Conditioning", "EmptyTrellis2LatentStructure", "VaeDecodeStructureTrellis2",
    "Trellis2ShapeStage", "Trellis2UpsampleStage", "VaeDecodeShapeTrellis", "Trellis2TextureStage",
    "VaeDecodeTextureTrellis", "RemeshMesh", "DecimateMesh", "MeshSmoothNormals", "UnwrapMesh",
    "BakeTextureFromVoxel", "ApplyTextureToMesh", "SaveGLB", "LoadBackgroundRemovalModel",
    "RemoveBackground", "ImageCropToMask"],     # ComfyUI 0.42 or newer
    "pbr_marigold_v2": ["MarigoldV2PostProcess", "ConditioningLoader"],
    "flux2_klein": ["Flux2Scheduler", "EmptyFlux2LatentImage"],
    "qwen_image_21": ["TextEncodeQwenImage21", "QwenImage21Cache"],
    "qwen_image_21_turbo": ["TextEncodeQwenImage21", "QwenImage21Cache"],
    "moss_image": ["MOSSImageControl", "EmptyFlux2LatentImage"],     # ComfyUI 0.42.3 or newer
    "fibo_edit": ["TextEncodeFibo2"],                                   # ComfyUI master from 2026-10-21
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
        "description_suffix": "*Adds the 8-step Qwen Image Edit Lightning LoRA and base Qwen Image's 4-step one. No*\n"
                              "    *preset uses them: the presets use the 4-step Edit LoRA, which Qwen core installs.*"},
    '7': {"name": "[QWEN NUNCHAKU] Nunchaku Nodes + Model",
        "tags": ["qwen_core", "qwen_nunchaku"],
        "size_gb": 33.0,
        "description_suffix": "*Installs Qwen Core components plus Nunchaku nodes and the Int4 quantized model (12.7GB).*"},
    '8': {"name": "[TRELLIS.2] Image-to-3D (ComfyUI's native nodes)",
        "tags": ["trellis2"],
        "size_gb": 13.1,
        "description_suffix": "*Needs ComfyUI 0.42 or newer. Installs the TRELLIS.2 4B model, its VAEs, the DINOv3 image*\n"
                              "    *encoder, BiRefNet background removal and StableGen's voxel-budget node. TRELLIS.2 and*\n"
                              "    *BiRefNet are MIT; DINOv3 is under Meta's DINOv3 License (commercial use allowed, not*\n"
                              "    *for military or weapons uses).*",
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
    '11': {"name": "[DEPRECATED] FLUX.2 Klein 4B FP8 + Qwen 3 Text Encoder + VAE",
        "tags": ["flux2_klein"],
        "size_gb": 12.4,
        "description_suffix": "*Downloads FLUX.2 Klein 4B FP8 diffusion model (~4.1GB), Qwen 3 4B text encoder bf16 (~8.0GB),*\n"
                              "    *and FLUX.2 VAE (~0.3GB). All ComfyUI nodes are built-in (no custom nodes needed).*\n"
                              "    *Deprecated: StableGen shows FLUX.2 Klein only with Enable Deprecated Architectures.*\n"
                              "    *Apache 2.0 license. Requires ~13GB VRAM.*",
    },
    '12': {"name": "[PBR DECOMPOSITION] Marigold V2 Models (Albedo + Normals)",
        "tags": ["pbr_marigold_v2"],
        "size_gb": 23.3,
        "description_suffix": "*Downloads Marigold V2 (Apache 2.0): the int8 Qwen Image Edit 2509 base (~19.5GB) plus albedo*\n"
                              "    *and normals LoRAs, VAEs and conditionings. Uses ComfyUI core nodes (September 2026 or newer).*\n"
                              "    *Roughness and metallic still need the Marigold IID node (option 9).*",
    },
    '13': {"name": "[QWEN IMAGE 2.1] Diffusion Model + Text Encoder + VAE",
        "tags": ["qwen_image_21"],
        "size_gb": 14.3,
        "notice": "qwen21",
        "description_suffix": "*Downloads Qwen Image 2.1 int8 (~7.3GB), the Qwen3-VL 8B text encoder (~6.3GB) and VAE*\n"
                              "    *(~0.7GB). Uses ComfyUI core nodes (September 2026 or newer).*\n"
                              "    *LICENSE NOTICE: Qwen Research License, non-commercial use only.*",
    },
    '14': {"name": "[AUTO-RIGGING] SkinTokens Node + Runtime",
        "tags": ["skintokens"],
        "size_gb": 9.0,
        "description_suffix": "*Installs the StableGen SkinTokens node and builds SkinTokens in its own environment*\n"
                              "    *(PyTorch, flash-attn when available, checkpoints ~1.6GB). Rigs TRELLIS.2 meshes; NVIDIA GPU.*",
    },
    '15': {"name": "[EXPERIMENTAL] Qwen Image 2.1 Turbo",
        "tags": ["qwen_image_21_turbo"],
        "size_gb": 13.9,
        "notice": "qwen21",
        "description_suffix": "*Downloads the Qwen Image 2.1 Turbo int8 model (~6.9GB) and, unless package 13 already did,*\n"
                              "    *the Qwen Image 2.1 text encoder and VAE (~7.0GB). Uses ComfyUI core nodes.*\n"
                              "    *LICENSE NOTICE: Qwen Research License, non-commercial use only.*",
    },
    '16': {"name": "[EXPERIMENTAL] MOSS Image + Depth Control",
        "tags": ["moss_image"],
        "size_gb": 15.0,
        "description_suffix": "*Downloads MOSS-Image int8 (~5.2GB), its Qwen3.5 9B text encoder (~9.3GB), VAE (~0.2GB) and*\n"
                              "    *color/depth control adapter (~0.3GB). Needs ComfyUI 0.42.3 or newer. Apache 2.0.*",
    },
    '18': {"name": "[LATO.2] Low-Poly Remesh (ComfyUI-LATO2 node)",
        "tags": ["lato2"],
        "size_gb": 12.2,
        "description_suffix": "*Clones ComfyUI-LATO2 and runs its setup: LATO.2 in its own Python environment (PyTorch*\n"
                              "    *2.8, CUDA 12.8, ~8 GB, built with uv), its weights (3.4 GB) and DINOv2 (1.1 GB). Needs an*\n"
                              "    *NVIDIA GPU. ComfyUI's Python only gets uv. LATO.2 and its weights are MIT.*",
    },
    '17': {"name": "[EXPERIMENTAL] FIBO 2.0 Edit",
        "tags": ["fibo_edit"],
        "size_gb": 13.0,
        "notice": "fibo",
        "description_suffix": "*Downloads Bria FIBO 2.0 Edit Turbo int8 (~8.0GB), its Qwen3-VL 4B text encoder (~4.6GB) and*\n"
                              "    *the FLUX.2 VAE (~0.3GB). Needs a ComfyUI build with core FIBO 2.0 support (master, October 2026).*\n"
                              "    *LICENSE NOTICE: CC BY-NC 4.0, non-commercial use only.*",
    },
}

LICENSE_NOTICES: Dict[str, List[str]] = {     # licenses that limit use (non-commercial), accepted before installing
    "qwen21": [
        "QWEN IMAGE 2.1 LICENSE NOTICE",
        "Qwen Image 2.1 is released under the Qwen Research License, which allows non-commercial use only",
        "(research or evaluation). Commercial use needs a separate license from the Qwen team.",
        "Qwen Image Edit 2509 (packages 5-7) is Apache 2.0 and not affected.",
        "",
        "License: https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
    ],
    "fibo": [
        "FIBO 2.0 LICENSE NOTICE",
        "Bria FIBO 2.0 is released under CC BY-NC 4.0, which allows non-commercial use only.",
        "Commercial use needs an agreement with Bria.",
        "",
        "License: https://huggingface.co/briaai/Fibo-2.0-edit-turbo",
    ],
}

# --- Helper Functions ---
PYTHON_OVERRIDE: Optional[str] = None   # --python
ASSUME_YES = False                      # --yes
NONINTERACTIVE = False                  # --events or --json: never ask, report instead
EVENTS = False                          # --events
ACCEPTED_NOTICES: Set[str] = set()      # --accept-notices
IGNORE_DISK_SPACE = False               # --ignore-disk-space
GIT_EXE = "git"                         # --git
CHILD_IO: Dict[str, Any] = {}           # --log: where pip, git and install.py write
_EVENT_STREAM = None                    # the real stdout while --log takes print()
_CURRENT_ITEM = ""                      # the item being installed, for progress events

# Child processes that can hang (a stalled pip or git, an install.py waiting on a lock) end after these
GIT_TIMEOUT = 30 * 60
PIP_TIMEOUT = 60 * 60
SCRIPT_TIMEOUT = 2 * 60 * 60


def emit(kind: str, **data):
    """One --events line."""
    if not EVENTS:
        return
    stream = _EVENT_STREAM or sys.stdout
    stream.write("@sg " + json.dumps({"event": kind, **data}, ensure_ascii=False) + "\n")
    stream.flush()


def timed_out(e: subprocess.TimeoutExpired) -> str:
    return f"stopped after {int(e.timeout) // 60} min without finishing"


def print_header(title: str):
    print(f"\n{'='*10} {title} {'='*10}")

def print_separator(char='-', length=70):
    print(char * length)


def ask(question: str, default_yes: bool = True, yes_answers: bool = True) -> bool:
    """Ask a yes/no question. --yes answers it, unless yes_answers is False (then --yes means no)."""
    if ASSUME_YES:
        print(f"{question} {'yes' if yes_answers else 'no'} (--yes)")
        return yes_answers
    if NONINTERACTIVE:
        print(f"{question} no (nothing can answer in --events/--json mode)")
        return False
    suffix = "(y/n, Enter for yes)" if default_yes else "(y/n)"
    try:
        answer = input(f"{question} {suffix}: ").strip().lower()
    except EOFError:
        return False
    return answer == "y" or (default_yes and answer == "")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StableGen dependency installer for ComfyUI.")
    parser.add_argument("comfyuipath", nargs='?', default=None,
                        help="Your ComfyUI folder (the one with 'models' and 'custom_nodes'). Asked for if omitted.")
    parser.add_argument("--packages",
                        help="Package numbers to install without the menu, comma-separated (e.g. 3,8). See --list.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Answer yes to every question, including the license notices.")
    parser.add_argument("--list", action="store_true", help="List the packages and what each installs, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be installed; change nothing.")
    parser.add_argument("--update-nodes", action="store_true",
                        help="Move custom nodes that are already installed to the commits StableGen was tested with.")
    parser.add_argument("--verify", action="store_true",
                        help="Check the installation: model files are in place and ComfyUI registers every node the "
                             "packages need. With --packages it runs after installing them (add --dry-run to only "
                             "check them); alone it checks every package that looks installed.")
    parser.add_argument("--server",
                        help="Verify against this running ComfyUI (e.g. 127.0.0.1:8188) instead of starting one.")
    parser.add_argument("--verify-timeout", type=int, default=900,
                        help="Seconds to wait for ComfyUI to start during --verify (default 900: the first start "
                             "after installing TRELLIS.2 builds its environment).")
    parser.add_argument("--verify-hashes", action="store_true",
                        help="Check the SHA-256 of Hugging Face downloads, including files already present (slow).")
    parser.add_argument("--python", help="Python executable of your ComfyUI environment (detected otherwise).")
    parser.add_argument("--repair", action="store_true",
                        help="No longer needed: TRELLIS.2 now runs on ComfyUI's native nodes. Kept for old scripts.")
    parser.add_argument("--check", action="store_true", help="No longer needed; kept for old scripts.")
    parser.add_argument("--accept-notices", default="",
                        help="Accept these license notices without asking, comma-separated (qwen21, fibo).")
    parser.add_argument("--ignore-disk-space", action="store_true",
                        help="Install even when the estimate says the drive is too full.")
    parser.add_argument("--git", help="Git executable to use (default: git on PATH).")
    parser.add_argument("--events", action="store_true",
                        help="Report progress as JSON lines prefixed '@sg ' (for the StableGen add-on); never ask.")
    parser.add_argument("--log", help="With --events: write every other line (and pip, git and install.py output) "
                                      "to this file.")
    parser.add_argument("--status", action="store_true",
                        help="Report ComfyUI's Python, GPU and the state of every package; change nothing.")
    parser.add_argument("--plan", action="store_true",
                        help="Report what installing --packages would add and change in this ComfyUI; change nothing.")
    parser.add_argument("--json", action="store_true", help="With --status or --plan: print one JSON document.")
    parser.add_argument("--create-comfyui", metavar="FOLDER",
                        help="Install StableGen's own ComfyUI into FOLDER (its own Python, torch for this GPU and a "
                             "tested ComfyUI release), or repair and update the one there. Then install packages "
                             "into FOLDER/ComfyUI as usual.")
    return parser.parse_args(argv)


def get_comfyui_path_from_args(args: argparse.Namespace) -> Path:
    comfyui_path_str = args.comfyuipath
    while not comfyui_path_str:
        if ASSUME_YES or NONINTERACTIVE:
            sys.exit("Error: pass the ComfyUI folder as an argument when using --yes, --events or --json.")
        comfyui_path_str = input("Please enter the full path to your ComfyUI installation directory: ").strip().strip('"')

    comfyui_path = Path(comfyui_path_str).expanduser().resolve()

    if not comfyui_path.is_dir():
        sys.exit(f"Error: ComfyUI path '{comfyui_path}' not found or not a directory.")
    if not (comfyui_path / "models").is_dir() or not (comfyui_path / "custom_nodes").is_dir():
        inner = comfyui_path / "ComfyUI"     # the portable build's outer folder
        if (inner / "models").is_dir() and (inner / "custom_nodes").is_dir():
            print(f"Using the ComfyUI folder inside it: {inner}")
            return inner
        sys.exit(f"Error: '{comfyui_path}' does not look like a ComfyUI folder (no 'models' or 'custom_nodes').")
    return comfyui_path


def python_candidates(comfyui_path: Path) -> List[Path]:
    """Where ComfyUI's Python usually lives. The Windows portable build keeps it next to the ComfyUI folder, in
    'python_embeded' (sic); venvs sit inside the folder or next to it (ComfyUI Desktop uses '.venv')."""
    out = []
    for base in (comfyui_path.parent, comfyui_path):
        for name in ("python_embeded", "python_embedded"):
            out.append(base / name / "python.exe")
    for base in (comfyui_path, comfyui_path.parent):
        for venv in ("venv", ".venv"):
            out.append(base / venv / "Scripts" / "python.exe")
            out.append(base / venv / "bin" / "python")
    return out


def find_comfyui_python(comfyui_path: Path) -> str:
    """The Python ComfyUI runs with: --python, a portable build's embedded Python, a venv, else this Python."""
    if PYTHON_OVERRIDE:
        return PYTHON_OVERRIDE
    for candidate in python_candidates(comfyui_path):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


_PYTHON_REPORT = (
    "import sys\n"
    "v = sys.version_info\n"
    "print(f'{v.major}.{v.minor}.{v.micro}')\n"
    "try:\n"
    "    import importlib.metadata as m\n"
    "    print(m.version('torch'))\n"
    "except Exception:\n"
    "    print('not installed')\n"
)


def describe_python(comfyui_path: Path) -> bool:
    """Print the Python the installer will use. False when none was found and this script's own Python is used."""
    python = find_comfyui_python(comfyui_path)
    detected = bool(PYTHON_OVERRIDE) or any(c.is_file() for c in python_candidates(comfyui_path))
    try:
        out = subprocess.run([python, "-c", _PYTHON_REPORT], capture_output=True, env=UTF8_ENV, timeout=120,
                             **UTF8_TEXT)
        version, torch = (out.stdout.split() + ["?", "?"])[:2]
    except (OSError, subprocess.TimeoutExpired):
        version, torch = "?", "?"
    print(f"ComfyUI Python: {python} (Python {version}, torch {torch})")
    if not detected:
        print("WARNING: ComfyUI's own Python was not found (a portable build's python_embeded or a venv), so the")
        print("         Python running this installer is used. If ComfyUI runs with a different Python, re-run")
        print("         with --python <path to that python>.")
    return detected


def install_pip_packages(pip_packages: List[str], comfyui_path: Path, force_reinstall: bool = False) -> bool:
    """Install pip packages into ComfyUI's Python environment. Returns True when all of them installed."""
    python_exe = find_comfyui_python(comfyui_path)
    print(f"  Installing pip packages into ComfyUI Python: {python_exe}")
    ok = True
    for pkg in pip_packages:
        print(f"    pip install {pkg} ...")
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install"] + (["--force-reinstall", "--no-deps"] if force_reinstall else []) + [pkg],
                check=True, timeout=PIP_TIMEOUT, **CHILD_IO,
            )
            print(f"    Successfully installed '{pkg}'.")
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: Failed to install '{pkg}' (exit code {e.returncode}).")
            print(f"    You may need to install it manually: pip install {pkg}")
            ok = False
        except subprocess.TimeoutExpired as e:
            print(f"    ERROR: pip install {pkg} {timed_out(e)}.")
            ok = False
        except FileNotFoundError:
            print(f"    ERROR: Python executable not found at '{python_exe}'.")
            print(f"    Please install '{pkg}' manually into your ComfyUI Python environment.")
            return False
    return ok


def run_node_install_script(item_details: Dict[str, Any], comfyui_path: Path) -> bool:
    """Run a custom node's install.py to resolve its dependencies. Returns True on success."""
    repo_path = comfyui_path / item_details["target_dir_relative"] / item_details["repo_name"]
    install_script = repo_path / "install.py"
    if not install_script.is_file():
        print(f"  WARNING: No install.py found in '{repo_path}'. Skipping dependency install.")
        return True

    python_exe = find_comfyui_python(comfyui_path)
    print(f"  Running install.py for {item_details['name']}...")
    print(f"    {python_exe} {install_script}")
    try:
        subprocess.run([python_exe, str(install_script)], check=True, cwd=str(repo_path), timeout=SCRIPT_TIMEOUT,
                       **CHILD_IO)
        print(f"  Successfully ran install.py for '{item_details['name']}'.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: install.py failed (exit code {e.returncode}). Run it manually to see why:")
        print(f"    cd {repo_path}")
        print(f"    {python_exe} install.py")
    except subprocess.TimeoutExpired as e:
        print(f"  ERROR: install.py {timed_out(e)}.")
    except FileNotFoundError:
        print(f"  ERROR: Python executable not found at '{python_exe}'.")
    return False


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


# --- Downloads ---
# Plain urllib, so the installer needs no third-party packages. Files download to '<name>.part' and are renamed
# once complete; an interrupted download resumes from the .part file on the next attempt or the next run.
USER_AGENT = "StableGen-installer"
DOWNLOAD_ATTEMPTS = 4
CHUNK = 1024 * 1024


class _RedirectRecorder(urllib.request.HTTPRedirectHandler):
    """Follows redirects, keeps Hugging Face's x-linked-* headers (only the first response has them), and does not
    forward an Authorization header to another host (the CDN URLs are pre-signed)."""

    def __init__(self):
        super().__init__()
        self.headers: Dict[str, str] = {}

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        for key in ("x-linked-etag", "x-linked-size"):
            if headers.get(key) and key not in self.headers:
                self.headers[key] = headers.get(key)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).hostname != urllib.parse.urlparse(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def _open(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, timeout: int = 60):
    """Open a URL; returns (response, headers recorded on redirects). HF_TOKEN authenticates Hugging Face."""
    recorder = _RedirectRecorder()
    opener = urllib.request.build_opener(recorder)
    request_headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and urllib.parse.urlparse(url).hostname == "huggingface.co":
        request_headers["Authorization"] = f"Bearer {token}"
    request_headers.update(headers or {})
    response = opener.open(urllib.request.Request(url, method=method, headers=request_headers), timeout=timeout)
    return response, recorder.headers


def remote_info(url: str) -> Tuple[int, Optional[str]]:
    """(size in bytes or 0, SHA-256 or None) of a remote file. Hugging Face reports both for LFS files."""
    try:
        response, linked = _open(url, method="HEAD", timeout=30)
        with response:
            size = int(response.headers.get("Content-Length") or 0)
    except Exception:
        return 0, None
    size = int(linked.get("x-linked-size") or 0) or size
    etag = (linked.get("x-linked-etag") or "").strip().strip('"')
    return size, (etag.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", etag) else None)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


class _PlainProgress:
    """Progress line for when tqdm is not installed."""

    def __init__(self, total: int, initial: int):
        self.total, self.done, self.shown = total, initial, 0.0

    def update(self, n: int):
        self.done += n
        now = time.time()
        if now - self.shown >= 1:
            self.shown = now
            pct = f" ({self.done / self.total:.0%})" if self.total else ""
            print(f"\r  {self.done / 1024**2:,.0f} / {self.total / 1024**2:,.0f} MB{pct}", end="", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        print()


class _EventProgress:
    """--events: progress events for one file, at most twice a second."""

    def __init__(self, name: str, total: int, initial: int):
        self.name, self.total, self.done, self.shown = name, total, initial, 0.0

    def update(self, n: int):
        self.done += n
        now = time.time()
        if now - self.shown >= 0.5:
            self.shown = now
            emit("progress", id=_CURRENT_ITEM, file=self.name, done=self.done, total=self.total)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        emit("progress", id=_CURRENT_ITEM, file=self.name, done=self.done, total=self.total)


def _progress(name: str, total: int, initial: int):
    if EVENTS:
        return _EventProgress(name, total, initial)
    if TQDM_AVAILABLE and total:
        return tqdm(total=total, initial=initial, unit='B', unit_scale=True, unit_divisor=1024, desc=name[:40],
                    ascii=True)
    return _PlainProgress(total, initial)


def download_url(url: str, final_path: Path, name: str, verify_hash: bool = False) -> Tuple[str, str]:
    """Download url to final_path, resuming and retrying. Returns (status, message); status is 'installed',
    'present', 'warning' or 'failed'."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    part = final_path.with_name(final_path.name + ".part")
    remote_size, remote_sha = remote_info(url)

    adopted = False     # resuming a file this installer did not write: it may not be the start of this one
    if final_path.exists():
        local = final_path.stat().st_size
        if not model_file_complete(final_path) and (not remote_size or local == remote_size):
            print(f"  '{final_path.name}' is damaged (shorter than its header says); downloading it again.")
            final_path.unlink()
        elif not remote_size or local == remote_size:
            if verify_hash and remote_sha:
                print(f"  Checking the SHA-256 of '{final_path.name}'...")
                if file_sha256(final_path) == remote_sha:
                    return "present", "SHA-256 verified"
                print(f"  '{final_path.name}' does not match the published SHA-256; downloading it again.")
                final_path.unlink()
            else:
                return "present", "" if remote_size else "server did not report a size; assumed complete"
        elif local < remote_size:
            print(f"  '{final_path.name}' is incomplete ({local / 1024**2:,.0f} of {remote_size / 1024**2:,.0f} MB); "
                  f"resuming.")
            if part.exists():
                part.unlink()
            final_path.rename(part)
            adopted = True
        else:
            return "warning", (f"local file is {local:,} bytes but the server's is {remote_size:,}; left as is "
                               f"(delete it to download again)")

    error = ""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        if remote_size and have == remote_size:
            break
        if remote_size and have > remote_size:
            part.unlink()
            have = 0
        try:
            response, _ = _open(url, headers={"Range": f"bytes={have}-"} if have else None)
            with response:
                if have and response.status != 206:
                    have = 0        # the server ignored the range request: start over
                length = int(response.headers.get("Content-Length") or 0)
                total = remote_size or (have + length if length else 0)
                with open(part, "ab" if have else "wb") as f, _progress(name, total, have) as bar:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        bar.update(len(block))
            done = part.stat().st_size
            if (remote_size and done < remote_size) or (not remote_size and length and done < have + length):
                raise ConnectionError(f"connection closed after {done / 1024**2:,.0f} MB")
            break
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:
                break       # nothing left to fetch
            if e.code in (401, 403):
                return "failed", (f"HTTP {e.code}: access denied (a gated model needs its license accepted on "
                                  f"Hugging Face and HF_TOKEN set)")
            if e.code == 404:
                return "failed", "HTTP 404: the file is no longer at this address"
            error = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError, ConnectionError, socket.timeout) as e:
            error = str(getattr(e, "reason", e))
        if attempt < DOWNLOAD_ATTEMPTS:
            wait = 5 * attempt
            print(f"  Download interrupted ({error}); retrying in {wait} s ({attempt}/{DOWNLOAD_ATTEMPTS - 1})...")
            time.sleep(wait)
        else:
            return "failed", f"download failed after {DOWNLOAD_ATTEMPTS} attempts ({error}); run again to resume"

    size = part.stat().st_size if part.exists() else 0
    if remote_size and size != remote_size:
        return "failed", f"incomplete ({size:,} of {remote_size:,} bytes); run again to resume"
    checked = (verify_hash or adopted) and bool(remote_sha)
    if checked:
        print(f"  Checking the SHA-256 of '{final_path.name}'...")
        if file_sha256(part) != remote_sha:
            part.unlink()
            return "failed", "SHA-256 mismatch; the download was deleted, run again"
    os.replace(part, final_path)
    return "installed", f"{size / 1024**2:,.0f} MB" + (", SHA-256 verified" if checked else "")


def download_file(item_details: Dict[str, Any], comfyui_path: Path, verify_hash: bool = False) -> Tuple[str, str]:
    url = item_details["url"]
    final_filepath = comfyui_path / item_details["target_path_relative"] / item_details["filename"]
    print(f"  {item_details['name']} (~{item_details.get('size_mb', '?')} MB) - License: {item_details['license']}")
    print(f"  From: {url}")
    print(f"  To:   {final_filepath}")
    status, message = download_url(url, final_filepath, item_details['name'], verify_hash)
    print(f"  {status}{': ' + message if message else ''}")
    return status, message


HF_REPO_PATTERNS = ["*.json", "*.txt", "*.yaml", "*.md", "*.py", "*.model", "*.fp16.safetensors", "*.fp16.bin",
                    "tokenizer/*"]


def _hf_repo_files(repo: str) -> List[str]:
    response, _ = _open(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", timeout=30)
    with response:
        entries = json.loads(response.read().decode("utf-8"))
    return [e["path"] for e in entries if e.get("type") == "file"]


def download_hf_model(item_details: Dict[str, Any], comfyui_path: Path, verify_hash: bool = False) -> Tuple[str, str]:
    """Download a Hugging Face diffusers repo (configs and fp16 weights) into ComfyUI/models/diffusers/<local_dir>.

    Required keys in *item_details*:
        hf_repo   – e.g. ``'Stable-X/yoso-delight-v0-4-base'``
        local_dir – directory name under ``models/diffusers/``
    """
    hf_repo = item_details["hf_repo"]
    target_dir = comfyui_path / "models" / "diffusers" / item_details["local_dir"]
    print(f"  Hugging Face model {hf_repo} (~{item_details.get('size_mb', '?')} MB) - "
          f"License: {item_details.get('license', 'unknown')}")
    print(f"  To:   {target_dir}")
    try:
        files = [f for f in _hf_repo_files(hf_repo)
                 if any(fnmatch.fnmatch(f, p) or fnmatch.fnmatch(f.split("/")[-1], p) for p in HF_REPO_PATTERNS)]
    except Exception as e:
        if (target_dir / "model_index.json").exists():
            return "present", f"could not list the repository ({e}); kept the existing copy"
        return "failed", f"could not list the repository: {e}"
    statuses = []
    for path in files:
        url = f"https://huggingface.co/{hf_repo}/resolve/main/{urllib.parse.quote(path)}"
        status, message = download_url(url, target_dir / path, path, verify_hash)
        if status == "failed":
            print(f"  {path}: {message}")
            return "failed", f"{path}: {message}"
        statuses.append(status)
    if "installed" in statuses:
        return "installed", f"{statuses.count('installed')} of {len(files)} files downloaded"
    return "present", f"{len(files)} files"


def _git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([GIT_EXE] + args, cwd=str(cwd), capture_output=True, timeout=GIT_TIMEOUT, **UTF8_TEXT)


def _rmtree(path: Path):
    """shutil.rmtree that also removes Git's read-only object files on Windows."""
    def make_writable(func, p, _exc):
        os.chmod(p, 0o700)
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(str(path), onexc=make_writable)
    else:
        shutil.rmtree(str(path), onerror=make_writable)


def clone_git_repo(item_details: Dict[str, Any], comfyui_path: Path, update_nodes: bool = False) -> Tuple[str, str]:
    """Clone a custom node at its tested commit, or check an existing clone. Returns (status, message)."""
    git_url = item_details["git_url"]
    repo_name = item_details["repo_name"]
    repo_path = comfyui_path / item_details["target_dir_relative"] / repo_name
    pinned = item_details.get("commit")

    try:
        if repo_path.is_dir():
            if not (repo_path / ".git").exists():
                return "present", "installed without Git (e.g. through ComfyUI-Manager); left as is"
            head = _git(["rev-parse", "HEAD"], repo_path).stdout.strip()
            if not pinned or head == pinned:
                return "present", f"at {head[:12]}" if head else ""
            if not (item_details.get("enforce_pin") or update_nodes):
                return "warning", (f"at {head[:12]}; StableGen was tested with {pinned[:12]} "
                                   f"(re-run with --update-nodes to switch)")
            print(f"  Moving '{repo_name}' to the tested commit {pinned[:12]}...")
            _git(["fetch", "origin"], repo_path)
            checkout = _git(["checkout", pinned], repo_path)
            if checkout.returncode != 0:
                reason = (checkout.stderr.strip().splitlines() or ["unknown error"])[-1]
                return "failed", (f"could not check out {pinned[:12]}: {reason} (commit or discard local changes "
                                  f"in {repo_path})")
            return "updated", f"{head[:12]} -> {pinned[:12]}"

        create_dir_if_not_exists(repo_path.parent)
        print(f"  Cloning: {item_details['name']} - License: {item_details['license']}")
        print(f"  From: {git_url}")
        print(f"  To:   {repo_path}")
        # Clone next to the target and move it in only when complete: an interrupted clone must not leave a
        # folder that later runs take for an installed node
        staging = repo_path.with_name(f".{repo_name}.stablegen-clone")
        if staging.exists():
            _rmtree(staging)
        clone = subprocess.run([GIT_EXE, 'clone', git_url, str(staging)], cwd=str(repo_path.parent),
                               timeout=GIT_TIMEOUT, **CHILD_IO)
        if clone.returncode != 0:
            if staging.exists():
                _rmtree(staging)
            return "failed", "git clone failed: check your network and that the repository is public"
        if pinned:
            checkout = _git(["checkout", pinned], staging)
            if checkout.returncode != 0:
                _rmtree(staging)
                return "failed", f"cloned, but the tested commit {pinned[:12]} is missing upstream"
        staging.rename(repo_path)
        return "installed", f"at {pinned[:12]}" if pinned else "latest commit"
    except FileNotFoundError:
        return "failed", "Git is not installed or not on PATH"
    except subprocess.TimeoutExpired as e:
        return "failed", f"git {timed_out(e)}"


# --- Post-install steps for nodes that set up their own runtime ---
_NUNCHAKU_WHEEL_SCRIPT = r'''"""Install the Nunchaku engine wheel matching this Python, torch and platform (StableGen installer)."""
import importlib.metadata
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("nunchaku_installers", sys.argv[1])
installers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installers)

try:
    dist = importlib.metadata.distribution("nunchaku")
    summary = (dist.metadata.get("Summary") or "").lower()
    if "piece-wise" not in summary and not dist.version.startswith("0.16"):
        print(f"Nunchaku {dist.version} is already installed.")
        sys.exit(0)
    print(f"Replacing the unrelated PyPI package 'nunchaku' {dist.version}.")
except importlib.metadata.PackageNotFoundError:
    pass

config = installers.generate_and_save_config() or installers.load_version_config()
versions = config.get("versions", [])
if not versions:
    sys.exit("Could not read Nunchaku's version list.")
from packaging.version import parse
version = max(versions, key=parse)
info = installers.get_system_info()
if not info.get("torch_version"):
    sys.exit("torch is not installed in this Python.")
backend = installers.get_install_backend()
for source in ("github", "huggingface", "modelscope"):
    wheel = installers.construct_compatible_wheel_info(version, source, info, config)
    if not wheel:
        continue
    try:
        print(f"Installing {wheel['name']} from {source}...")
        installers.install_wheel(wheel["url"], backend)
        print(f"Installed Nunchaku {version}.")
        sys.exit(0)
    except Exception as e:
        print(f"  {source}: {e}")
sys.exit(f"No Nunchaku {version} wheel for Python {info['python_version']}, {info['torch_version']}, "
         f"{info['platform_tag']}.")
'''


def _install_nunchaku_wheel(item_details: Dict[str, Any], comfyui_path: Path) -> Tuple[bool, str]:
    repo_path = comfyui_path / item_details["target_dir_relative"] / item_details["repo_name"]
    installers = repo_path / "nodes" / "tools" / "installers.py"
    if not installers.is_file():
        return False, f"{installers} not found"
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "stablegen_nunchaku_wheel.py"
        script.write_text(_NUNCHAKU_WHEEL_SCRIPT, encoding="utf-8")
        try:
            result = subprocess.run([find_comfyui_python(comfyui_path), str(script), str(installers)],
                                    timeout=PIP_TIMEOUT, **CHILD_IO)
        except subprocess.TimeoutExpired as e:
            return False, timed_out(e)
    return result.returncode == 0, "" if result.returncode == 0 else "see the messages above"


def scripts_first_env(comfyui_path: Path) -> Dict[str, str]:
    """The environment for a node's setup script: ComfyUI's Python's own scripts folder first on PATH, so a tool
    pip installed there (uv) runs as itself, not through a launcher shim. pyenv-win's shims are .bat files whose
    arguments pass through cmd, which reads the '<' in 'numpy<2.3' as an input redirection."""
    env = dict(UTF8_ENV)
    try:
        out = subprocess.run([find_comfyui_python(comfyui_path), "-c", "import sysconfig; print(sysconfig.get_path('scripts'))"],
                             capture_output=True, env=UTF8_ENV, timeout=60, **UTF8_TEXT)
        scripts = out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else ""
    except (OSError, subprocess.TimeoutExpired):
        scripts = ""
    if scripts and Path(scripts).is_dir():
        env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
    return env


def _install_skintokens_runtime(item_details: Dict[str, Any], comfyui_path: Path) -> Tuple[bool, str]:
    repo_path = comfyui_path / item_details["target_dir_relative"] / item_details["repo_name"]
    script = repo_path / "install_skintokens.py"
    if not script.is_file():
        return False, f"{script} not found"
    print("  Building the SkinTokens runtime (its own Python environment, PyTorch and checkpoints).")
    print("  This takes a while the first time.")
    try:
        result = subprocess.run([find_comfyui_python(comfyui_path), str(script), "--target", "managed"],
                                cwd=str(repo_path), timeout=SCRIPT_TIMEOUT,
                                env=scripts_first_env(comfyui_path), **CHILD_IO)
    except subprocess.TimeoutExpired as e:
        return False, timed_out(e)
    return result.returncode == 0, "" if result.returncode == 0 else "see the messages above"


def _install_lato2_runtime(item_details: Dict[str, Any], comfyui_path: Path) -> Tuple[bool, str]:
    repo_path = comfyui_path / item_details["target_dir_relative"] / item_details["repo_name"]
    script = repo_path / "install_lato2.py"
    if not script.is_file():
        return False, f"{script} not found"
    print("  Building the LATO.2 runtime (its own Python environment and PyTorch) and downloading its weights.")
    print("  This takes a while the first time (about 12 GB).")
    try:
        result = subprocess.run([find_comfyui_python(comfyui_path), str(script)],
                                cwd=str(repo_path), timeout=SCRIPT_TIMEOUT,
                                env=scripts_first_env(comfyui_path), **CHILD_IO)
    except subprocess.TimeoutExpired as e:
        return False, timed_out(e)
    return result.returncode == 0, "" if result.returncode == 0 else "see the messages above"


POST_INSTALL = {
    "nunchaku_wheel": ("Nunchaku engine", _install_nunchaku_wheel),
    "skintokens_runtime": ("SkinTokens runtime", _install_skintokens_runtime),
    "lato2_runtime": ("LATO.2 runtime", _install_lato2_runtime),
}


def install_embedded_node(item_details: Dict[str, Any], comfyui_path: Path) -> Tuple[str, str]:
    """Write a node StableGen ships inside this installer (no Git needed). Returns (status, message)."""
    folder = item_target(item_details, comfyui_path)
    existed = folder.is_dir()
    changed = []
    for relative, text in item_details["files"].items():
        target = folder / relative
        if target.is_file() and target.read_text(encoding="utf-8") == text:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        changed.append(relative)
    if not changed:
        return "present", ""
    return ("updated" if existed else "installed"), f"wrote {', '.join(changed)} in {folder}"


def install_node(item_details: Dict[str, Any], comfyui_path: Path, update_nodes: bool) -> List[Tuple[str, str, str]]:
    """Clone or check a custom node, then install what it needs. Returns (name, status, message) rows."""
    name = item_details["name"]
    status, message = clone_git_repo(item_details, comfyui_path, update_nodes)
    print(f"  {status}{': ' + message if message else ''}")
    rows = [(name, status, message)]
    if status == "failed":
        return rows
    if item_details.get("pip_packages") and not install_pip_packages(item_details["pip_packages"], comfyui_path):
        rows.append((f"{name}: Python packages", "failed", "pip install failed (see above)"))
    if item_details.get("post_clone_patches"):
        apply_post_clone_patches(item_details, comfyui_path)
    if item_details.get("run_install_script") and not run_node_install_script(item_details, comfyui_path):
        rows.append((f"{name}: install.py", "failed", "the node's dependency install failed (see above)"))
    hook = item_details.get("post_install")
    if hook:
        label, run = POST_INSTALL[hook]
        ok, message = run(item_details, comfyui_path)
        rows.append((f"{name}: {label}", "installed" if ok else "failed", message))
    return rows


# --- Packages ---
def display_menu(comfyui_path: Path):
    print_header("StableGen Dependency Installer")
    print(f"Target ComfyUI Directory: {comfyui_path}")
    print_separator()
    print("This script downloads and sets up the custom nodes and models StableGen uses.")
    print("To install components by hand instead, see the manual installation guide:")
    print("https://github.com/sakalond/StableGen/wiki/Manual-installation")
    print_separator()
    print("Download sizes are estimates. The full list of components is shown for approval before anything changes.")
    print_separator()

    for key, val in MENU_PACKAGES.items():
        print(f"\n{key}. {val['name']}")
        print_separator(char='.')
        print(f"    *Approximate total download size: ~{val['size_gb']:.1f} GB*")
        if val.get("description_suffix"):
            print(f"    {val['description_suffix']}")
    print("\nq. Quit")
    print_separator()


def print_package_list():
    """--list: every package and the components it installs."""
    for key, val in MENU_PACKAGES.items():
        print(f"{key:>3}. {val['name']} (~{val['size_gb']:.1f} GB)")
        for item in resolve_items([key]):
            kind = {"node": "node", "embedded_node": "node", "model": "model", "hf_model": "model"}[item["type"]]
            print(f"       {kind:<5} {item['name']}")


def parse_package_choice(text: str) -> List[str]:
    """'3, 8' -> ['3', '8']. Package tags ('trellis2') work too. Raises ValueError for unknown ones."""
    keys = [k.strip() for k in text.replace(";", ",").split(",") if k.strip()]
    unknown = [k for k in keys if k not in MENU_PACKAGES and k not in TAG_LABELS]
    if not keys or unknown:
        raise ValueError(f"unknown package(s): {', '.join(unknown) or text!r}; choose from {', '.join(MENU_PACKAGES)} "
                         f"or a package tag ({', '.join(TAG_LABELS)})")
    return list(dict.fromkeys(keys))


def package_info(key: str) -> Dict[str, Any]:
    """A menu package ('8'), or a package tag ('trellis2') as a package of its own: the add-on installs by tag."""
    if key in MENU_PACKAGES:
        return MENU_PACKAGES[key]
    notices = [p["notice"] for p in MENU_PACKAGES.values() if p.get("notice") and key in p["tags"]]
    size_mb = sum(i.get("size_mb", 0) for i in DEPENDENCIES.values() if key in i["packages"]) + ENV_SIZE_MB.get(key, 0)
    return {"name": TAG_LABELS[key], "tags": [key], "size_gb": round(size_mb / 1024, 1),
            **({"notice": notices[0]} if notices else {})}


def tags_for(keys: List[str]) -> List[str]:
    return list(dict.fromkeys(tag for key in keys for tag in package_info(key)["tags"]))


def get_unique_item_ids_for_tags(selected_tags: List[str]) -> Set[str]:
    return {item_id for item_id, details in DEPENDENCIES.items()
            if any(tag in details["packages"] for tag in selected_tags)}


def resolve_items(keys: List[str]) -> List[Dict[str, Any]]:
    """The components of the chosen packages, each once, in DEPENDENCIES order."""
    ids = get_unique_item_ids_for_tags(tags_for(keys))
    return [item for item_id, item in DEPENDENCIES.items() if item_id in ids]


def item_target(item: Dict[str, Any], comfyui_path: Path) -> Path:
    if item["type"] in ("node", "embedded_node"):
        return comfyui_path / item["target_dir_relative"] / item["repo_name"]
    if item["type"] == "model":
        return comfyui_path / item["target_path_relative"] / item["filename"]
    return comfyui_path / "models" / "diffusers" / item["local_dir"]


def item_present(item: Dict[str, Any], comfyui_path: Path) -> bool:
    target = item_target(item, comfyui_path)
    if item["type"] == "hf_model":
        return (target / "model_index.json").exists()
    if item["type"] == "embedded_node":
        return all((target / relative).is_file() for relative in item["files"])
    if item["type"] == "model":
        return target.exists() and model_file_complete(target)
    # a node whose setup builds more than its clone counts once that is done ("ready": globs under the ComfyUI folder)
    return target.exists() and all(any(comfyui_path.glob(pattern)) for pattern in item.get("ready", []))


_COMPLETE: Dict[Tuple[str, int, int], bool] = {}


def model_file_complete(path: Path) -> bool:
    """False for a .safetensors file shorter than its own header says: a download cut short, or a copy that stopped.
    Other formats are taken as complete. Remembered per size and modification time."""
    if path.suffix != ".safetensors":
        return True
    try:
        stat = path.stat()
    except OSError:
        return False
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _COMPLETE:
        _COMPLETE[key] = _safetensors_complete(path, stat.st_size)
    return _COMPLETE[key]


def _safetensors_complete(path: Path, size: int) -> bool:
    """A safetensors file is an 8-byte header length, a JSON header and the tensors' bytes at the offsets it lists."""
    try:
        with open(path, "rb") as f:
            length = int.from_bytes(f.read(8), "little")
            if not 0 < length <= 100 * 1024**2 or 8 + length > size:
                return False
            header = json.loads(f.read(length))
    except (OSError, ValueError):
        return False
    end = max((entry["data_offsets"][1] for name, entry in header.items()
               if name != "__metadata__" and isinstance(entry, dict) and "data_offsets" in entry), default=0)
    return size >= 8 + length + end


TAG_LABELS = {
    "core": "SDXL core", "preset_essentials": "SDXL preset essentials", "extended_optional": "SDXL extras",
    "checkpoint_realvis": "RealVisXL checkpoint", "qwen_core": "Qwen Image Edit core", "qwen_extras": "Qwen extras",
    "qwen_nunchaku": "Qwen Nunchaku", "pbr_marigold": "Marigold IID", "pbr_stabledelight": "StableDelight",
    "pbr_marigold_v2": "Marigold V2", "trellis2": "TRELLIS.2", "flux2_klein": "FLUX.2 Klein",
    "qwen_image_21": "Qwen Image 2.1", "skintokens": "SkinTokens",
    "qwen_image_21_turbo": "Qwen Image 2.1 Turbo", "moss_image": "MOSS Image", "fibo_edit": "FIBO 2.0 Edit",
    "lato2": "LATO.2",
}


def installed_tags(comfyui_path: Path) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """For --verify without --packages: package tags with every component present, and the partly installed ones
    as {tag: (present, total)}."""
    complete, partial = [], {}
    for tag in TAG_LABELS:
        items = [item for item in DEPENDENCIES.values() if tag in item["packages"]]
        present = sum(item_present(item, comfyui_path) for item in items)
        if items and present == len(items):
            complete.append(tag)
        elif present:
            partial[tag] = (present, len(items))
    return complete, partial


# --- Checks before installing ---
def missing_core_nodes(comfyui_path: Path, tags: List[str]) -> Dict[str, List[str]]:
    """ComfyUI core nodes the packages need that this ComfyUI version does not have, by package tag."""
    sources = [comfyui_path / "nodes.py"] + sorted((comfyui_path / "comfy_extras").glob("*.py"))
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in sources if p.is_file())
    if not text:
        return {}       # not a source checkout (e.g. ComfyUI Desktop): can't tell offline
    missing = {}
    for tag in tags:
        absent = [n for n in CORE_NODES.get(tag, []) if not re.search(rf"[\"']{re.escape(n)}[\"']|class {re.escape(n)}\b", text)]
        if absent:
            missing[tag] = absent
    return missing


ENV_SIZE_MB = {"skintokens": 9000, "lato2": 12500}


def estimate_download_mb(items: List[Dict[str, Any]], tags: List[str], comfyui_path: Path) -> int:
    """MB still to download: models that are not present yet, plus environments the nodes build."""
    total = sum(item.get("size_mb", 0) for item in items
                if item["type"] in ("model", "hf_model") and not item_present(item, comfyui_path))
    for tag in tags:
        node_items = [i for i in items if i["type"] == "node" and tag in i["packages"]]
        if tag in ENV_SIZE_MB and not all(item_present(i, comfyui_path) for i in node_items):
            total += ENV_SIZE_MB[tag]
    return total


def show_notice(key: str) -> bool:
    print_separator(char='!')
    for line in LICENSE_NOTICES[key]:
        print(line)
    print_separator(char='!')
    if key in ACCEPTED_NOTICES:
        print("Accepted (--accept-notices).")
        return True
    if NONINTERACTIVE and not ASSUME_YES:
        emit("error", code="notice", key=key, lines=LICENSE_NOTICES[key],
             message=f"the '{key}' license notice was not accepted (--accept-notices {key})")
        return False
    return ask("Acknowledge and continue?")


# --- Verification ---
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fetch_object_info(address: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(f"http://{address}/object_info", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _stop_process_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def start_comfyui_for_verify(comfyui_path: Path, timeout: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Start ComfyUI on a free local port, read /object_info, stop it. Returns (object_info or None, custom node
    folders that failed to import)."""
    main_py = comfyui_path / "main.py"
    if not main_py.is_file():
        print(f"  {main_py} not found (ComfyUI Desktop?). Start ComfyUI yourself and pass --server <address>.")
        return None, []
    python = find_comfyui_python(comfyui_path)
    port = _free_port()
    log_path = Path(tempfile.gettempdir()) / f"stablegen-verify-comfyui-{port}.log"
    cmd = [python] + (["-s"] if "python_embed" in python else []) + [
        str(main_py), "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch"]
    print(f"  Starting ComfyUI on port {port} to read its node list (log: {log_path})...")
    kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000200 | 0x08000000    # new process group, no console window
    else:
        kwargs["start_new_session"] = True
    info = None
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(comfyui_path), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, **kwargs)
        start = time.time()
        reported = 0
        try:
            while time.time() - start < timeout:
                if proc.poll() is not None:
                    print(f"  ComfyUI exited early (code {proc.returncode}); see the log.")
                    break
                info = _fetch_object_info(f"127.0.0.1:{port}", timeout=5)
                if info is not None:
                    break
                elapsed = int(time.time() - start)
                if elapsed - reported >= 60:
                    reported = elapsed
                    print(f"  ...still starting ({elapsed} s)")
                time.sleep(3)
            else:
                print(f"  ComfyUI did not answer within {timeout} s; see the log.")
        finally:
            _stop_process_tree(proc)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failed = re.findall(r"\(IMPORT FAILED\)[^\n]*?custom_nodes[\\/]([^\\/\n]+)", text)
    return info, sorted(set(failed))


def verify_installation(comfyui_path: Path, tags: List[str], server: Optional[str], timeout: int) -> bool:
    print_header("Verifying the installation")
    ok = True
    missing_files: List[str] = []
    items = [item for item in DEPENDENCIES.values() if any(t in item["packages"] for t in tags)]
    for item in items:
        if item["type"] in ("node", "embedded_node"):
            continue
        target = item_target(item, comfyui_path)
        if not item_present(item, comfyui_path):
            print(f"  MISSING  {item['name']} ({target})")
            missing_files.append(str(target))
            ok = False
        elif item["type"] == "model" and target.with_name(target.name + ".part").exists():
            print(f"  WARNING  {item['name']}: an unfinished download is next to it")
    required = list(dict.fromkeys(n for t in tags for n in PACKAGE_NODES.get(t, [])))
    if not required:
        emit("verify", ok=ok, missing_files=missing_files, missing_nodes=[], import_failed=[])
        return ok
    if server:
        info, failed = _fetch_object_info(server), []
        if info is None:
            print(f"  Could not reach ComfyUI at {server}.")
    else:
        info, failed = start_comfyui_for_verify(comfyui_path, timeout)
    ours = {item["repo_name"] for item in items if item["type"] in ("node", "embedded_node")}
    for folder in failed:
        if folder in ours:
            print(f"  FAILED   custom node '{folder}' did not import (see the ComfyUI log)")
    others = [folder for folder in failed if folder not in ours]
    if others:
        print(f"  Note: other custom nodes failed to import (not StableGen's): {', '.join(others)}")
    failed = [folder for folder in failed if folder in ours]
    if info is None:
        emit("verify", ok=False, missing_files=missing_files, missing_nodes=required, import_failed=failed,
             message="ComfyUI did not answer")
        return False
    missing = [n for n in required if n not in info]
    emit("verify", ok=ok and not missing and not failed, missing_files=missing_files, missing_nodes=missing,
         import_failed=failed)
    for tag in tags:
        absent = [n for n in PACKAGE_NODES.get(tag, []) if n in missing]
        if absent:
            core = set(CORE_NODES.get(tag, []))
            hint = "update ComfyUI" if set(absent) <= core else "the custom node did not load"
            print(f"  MISSING  {', '.join(absent)} ({hint})")
    print(f"  {len(required) - len(missing)} of {len(required)} required node types are registered.")
    return ok and not missing and not failed


# --- StableGen's own ComfyUI (--create-comfyui) ---
# A private ComfyUI built with uv: its own Python, torch for this GPU and a tested ComfyUI release, all inside one
# folder. Each step records what it installed in stablegen-comfyui.json and is skipped while that is in place, so
# running it again repairs a broken install, and moves an old one to the pins below.
UV_VERSION = "0.13.1"
UV_ASSETS = {       # platform -> (release asset, SHA-256, MB)
    "win32": ("uv-x86_64-pc-windows-msvc.zip", "3b2661a463a5baea7f5076ea2b25ab71d77f36a6c0d809e5350651792f80a672", 15),
    "linux": ("uv-x86_64-unknown-linux-gnu.tar.gz",
              "eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220", 16),
    "darwin": ("uv-aarch64-apple-darwin.tar.gz", "cfe454573412ff7380d15fc7dae5997ab53ffce72fce0a78e91ddbc84001acd2", 14),
}
# Windows has no Git of its own; elsewhere the system's is used
MINGIT_VERSION = "2.56.0.2"
MINGIT = ("https://github.com/git-for-windows/git/releases/download/v2.56.0.windows.2/MinGit-2.56.0.2-64-bit.zip",
          "da35e72aa21c005a5a0d298cfbae110bc1609a815730ea0dde84b01a1b3cd3be", 38)
COMFYUI_GIT = "https://github.com/Comfy-Org/ComfyUI.git"
COMFYUI_TAG, COMFYUI_COMMIT = "v0.43.0", "426f5eea52a7ece1dd599fff4cb6cd1f352ddd24"    # has every CORE_NODES entry
PYTHON_VERSION = "3.12"
TORCH_PINS = ["torch==2.9.0", "torchvision==0.24.0", "torchaudio==2.9.0"]
OWN_MARKER = "stablegen-comfyui.json"


def nvidia_driver() -> Optional[Tuple[int, int]]:
    """The NVIDIA driver version, or None without an NVIDIA GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                             timeout=60, **UTF8_TEXT).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.match(r"\s*(\d+)\.(\d+)", out or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def torch_backend(driver: Optional[Tuple[int, int]]) -> str:
    """The newest CUDA build this NVIDIA driver runs (CUDA 13 needs driver 580), else uv's own pick (ROCm, CPU,
    Apple's MPS)."""
    if driver is None:
        return "auto"
    return "cu130" if driver >= (580, 0) else "cu128" if driver >= (570, 0) else "cu126"


def own_python(comfy: Path) -> Path:
    return comfy / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _uv_env(root: Path) -> Dict[str, str]:
    env = dict(UTF8_ENV, UV_PYTHON_INSTALL_DIR=str(root / "tools" / "python"), UV_CACHE_DIR=str(root / "cache" / "uv"),
               UV_NO_CONFIG="1", UV_PYTHON_PREFERENCE="only-managed", UV_PYTHON_DOWNLOADS="automatic")
    for name in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX"):
        env.pop(name, None)
    return env


def _run_step(cmd: List[str], env: Dict[str, str], timeout: int, cwd: Optional[Path] = None) -> Tuple[str, str]:
    print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, timeout=timeout, cwd=str(cwd) if cwd else None,
                                stdin=subprocess.DEVNULL, **CHILD_IO)
    except subprocess.TimeoutExpired as e:
        return "failed", timed_out(e)
    except OSError as e:
        return "failed", str(e)
    return ("installed", "") if result.returncode == 0 else ("failed", f"exit code {result.returncode}; see the log")


def _fetch_pinned(url: str, sha: str, target: Path, name: str) -> Tuple[str, str]:
    """Download a release archive and check it against its pinned SHA-256."""
    status, message = download_url(url, target, name)
    if status == "failed":
        return status, message
    if file_sha256(target) != sha:
        target.unlink()
        return "failed", f"{target.name} does not match its published SHA-256; deleted, run again"
    return "installed", ""


def _extract(archive: Path, target: Path):
    import tarfile
    import zipfile
    if target.exists():
        _rmtree(target)
    target.mkdir(parents=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target, **({"filter": "data"} if sys.version_info >= (3, 12) else {}))


def create_comfyui(root: Path, dry_run: bool = False) -> int:
    """--create-comfyui: StableGen's own ComfyUI in root/ComfyUI, with uv, Python and Git in root/tools."""
    global GIT_EXE, _CURRENT_ITEM
    root = root.expanduser().resolve()
    comfy, tools = root / "ComfyUI", root / "tools"
    if sys.platform not in UV_ASSETS:
        return _fail("platform", f"StableGen cannot set up its own ComfyUI on {sys.platform}")
    root.mkdir(parents=True, exist_ok=True)
    marker_path = root / OWN_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        marker = {}
    uv_asset, uv_sha, uv_mb = UV_ASSETS[sys.platform]
    uv_exe = tools / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")
    backend = torch_backend(nvidia_driver())
    torch_id = f"{' '.join(TORCH_PINS)} {backend}"
    python = own_python(comfy)
    requirements = comfy / "requirements.txt"
    windows = sys.platform == "win32"
    git_exe = tools / "git" / "cmd" / "git.exe"

    def head() -> Optional[str]:
        if not (comfy / ".git").exists() or not git_executable():
            return None
        try:
            return _git(["rev-parse", "HEAD"], comfy).stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            return None

    # (id, name, MB to download, already in place?)
    if windows:
        GIT_EXE = str(git_exe) if git_exe.is_file() else GIT_EXE
    steps = [
        ("uv", f"uv {UV_VERSION}", uv_mb, marker.get("uv") == UV_VERSION and uv_exe.is_file()),
        ("git", f"Git {MINGIT_VERSION}" if windows else "Git", MINGIT[2] if windows else 0,
         (marker.get("git") == MINGIT_VERSION and git_exe.is_file()) if windows else bool(shutil.which("git"))),
        ("comfyui", f"ComfyUI {COMFYUI_TAG}", 120, head() == COMFYUI_COMMIT),
        ("python", f"Python {PYTHON_VERSION}", 40, marker.get("python") == PYTHON_VERSION and python.is_file()),
        ("torch", f"torch {TORCH_PINS[0].split('==')[1]} ({backend})", 2800,
         marker.get("torch") == torch_id and python.is_file()),
        ("requirements", "ComfyUI's Python packages", 500,
         python.is_file() and requirements.is_file() and marker.get("requirements") == file_sha256(requirements)),
    ]
    need_mb = sum(mb for _, _, mb, present in steps if not present)
    free_mb = shutil.disk_usage(root).free // 1024**2
    print_header("StableGen's own ComfyUI")
    print(f"Folder: {root}  (download ~{need_mb / 1024:.1f} GB; {free_mb / 1024:.1f} GB free)")
    emit("plan", packages=["comfyui"], download_mb=need_mb, free_mb=free_mb,
         items=[{"id": i, "name": n, "type": "step", "size_mb": mb, "present": p} for i, n, mb, p in steps])
    # torch and the packages take about three times their download once installed
    if need_mb * 3 > free_mb and not IGNORE_DISK_SPACE:
        message = f"not enough free disk space (~{need_mb * 3 / 1024:.0f} GB needed, {free_mb / 1024:.1f} GB free)"
        if NONINTERACTIVE or not ask(f"{message}. Continue anyway?", default_yes=False, yes_answers=False):
            return _fail("disk_space", message, need_mb=need_mb * 3, free_mb=free_mb)
    if dry_run:
        print("Dry run: nothing was changed.")
        emit("summary", ok=True, rows=[], restart=False, dry_run=True)
        return 0

    env = _uv_env(root)
    rows: List[Tuple[str, str, str]] = []
    for step_id, name, _mb, present in steps:
        _CURRENT_ITEM = step_id
        emit("item", id=step_id, name=name, phase="start")
        print_separator(char='.')
        print(f"Step: {name}")
        if present:
            status, message = "present", ""
        elif step_id == "uv":
            archive = tools / "downloads" / uv_asset
            status, message = _fetch_pinned(f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
                                            f"{uv_asset}", uv_sha, archive, uv_asset)
            if status != "failed":
                _extract(archive, tools / "uv-unpacked")
                found = next((tools / "uv-unpacked").rglob(uv_exe.name), None)
                if found is None:
                    status, message = "failed", f"no {uv_exe.name} in {uv_asset}"
                else:
                    uv_exe.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(found, uv_exe)
                    os.chmod(uv_exe, 0o755)
                    _rmtree(tools / "uv-unpacked")
                    marker["uv"] = UV_VERSION
        elif step_id == "git":
            if windows:
                archive = tools / "downloads" / Path(MINGIT[0]).name
                status, message = _fetch_pinned(MINGIT[0], MINGIT[1], archive, archive.name)
                if status != "failed":
                    _extract(archive, tools / "git")
                    GIT_EXE = str(git_exe)
                    marker["git"] = MINGIT_VERSION
            else:
                status, message = "failed", "Git is not installed (for example: sudo apt install git)"
        elif step_id == "comfyui":
            item = {"name": "ComfyUI", "git_url": COMFYUI_GIT, "target_dir_relative": ".", "repo_name": "ComfyUI",
                    "commit": COMFYUI_COMMIT, "enforce_pin": True, "license": "GPL-3.0"}
            status, message = clone_git_repo(item, root)
        elif step_id == "python":
            status, message = _run_step([str(uv_exe), "venv", "--seed", "--clear", "--python", PYTHON_VERSION,
                                         str(comfy / ".venv")], env, PIP_TIMEOUT)
            if status == "installed":
                marker["python"] = PYTHON_VERSION
        elif step_id == "torch":
            status, message = _run_step([str(uv_exe), "pip", "install", "--python", str(python), *TORCH_PINS,
                                         f"--torch-backend={backend}"], env, PIP_TIMEOUT)
            if status == "installed":
                marker["torch"] = torch_id
        else:
            constraints = root / "torch-constraints.txt"     # ComfyUI's unpinned torch must stay the build above
            constraints.write_text("\n".join(TORCH_PINS) + "\n", encoding="utf-8")
            status, message = _run_step([str(uv_exe), "pip", "install", "--python", str(python), "-r",
                                         str(requirements), "-c", str(constraints), f"--torch-backend={backend}"],
                                        env, PIP_TIMEOUT)
            if status == "installed":
                marker["requirements"] = file_sha256(requirements)
        print(f"  {status}{': ' + message if message else ''}")
        rows.append((name, status, message))
        emit("item", id=step_id, name=name, phase="done", status=status,
             rows=[{"name": name, "status": status, "message": message}])
        marker_path.write_text(json.dumps(dict(marker, comfyui=COMFYUI_TAG, root=str(root)), indent=1),
                               encoding="utf-8")
        if status == "failed":
            break
    _CURRENT_ITEM = ""
    ok = print_summary(rows) and len(rows) == len(steps)
    emit("summary", ok=ok, rows=[{"name": n, "status": s, "message": m} for n, s, m in rows], restart=True,
         dry_run=False)
    if ok:
        print(f"ComfyUI is ready in {comfy}. Install packages into it with:  installer.py \"{comfy}\" --packages ...")
    return 0 if ok else 1


# --- Reports for the add-on (--status, --plan) ---
_ENV_REPORT = (
    "import json, sys\n"
    "import importlib.metadata as m\n"
    "v = sys.version_info\n"
    "r = {'version': f'{v.major}.{v.minor}.{v.micro}', 'torch': None, 'cuda': None, 'gpu': None, 'vram_mb': None,\n"
    "     'packages': {}}\n"
    "for name in sys.argv[1:]:\n"
    "    try:\n"
    "        r['packages'][name] = m.version(name)\n"
    "    except Exception:\n"
    "        pass\n"
    "try:\n"
    "    import torch\n"
    "    r['torch'], r['cuda'] = torch.__version__, torch.version.cuda\n"
    "    if torch.cuda.is_available():\n"
    "        p = torch.cuda.get_device_properties(0)\n"
    "        r['gpu'], r['vram_mb'] = p.name, p.total_memory // 2**20\n"
    "    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():\n"
    "        r['gpu'] = 'Apple GPU (MPS)'\n"
    "except Exception as e:\n"
    "    r['torch_error'] = str(e)[:300]\n"
    "print(json.dumps(r))\n"
)


def probe_environment(comfyui_path: Path, distributions: List[str] = ()) -> Dict[str, Any]:
    """ComfyUI's Python: version, torch, CUDA, GPU and the installed versions of `distributions`."""
    python = find_comfyui_python(comfyui_path)
    report: Dict[str, Any] = {"path": python,
                              "detected": bool(PYTHON_OVERRIDE) or any(c.is_file() for c in python_candidates(comfyui_path))}
    try:
        out = subprocess.run([python, "-c", _ENV_REPORT, *distributions], capture_output=True, env=UTF8_ENV,
                             timeout=300, **UTF8_TEXT)
        lines = out.stdout.strip().splitlines()
        report.update(json.loads(lines[-1]) if out.returncode == 0 and lines else
                      {"error": (out.stderr.strip().splitlines() or ["no output"])[-1]})
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        report["error"] = str(e)
    return report


def git_executable() -> Optional[str]:
    return GIT_EXE if Path(GIT_EXE).is_file() else shutil.which(GIT_EXE)


MODEL_TYPES = ("model", "hf_model")
NODE_TYPES = ("node", "embedded_node")


def removed_with_package(item: Dict[str, Any], nodes: str) -> bool:
    """Whether removing a package deletes this item. Models always; nodes with nodes='all' (a ComfyUI StableGen
    built), or StableGen's own nodes with nodes='stablegen' (someone's own ComfyUI, where the other nodes may
    predate StableGen and serve other workflows)."""
    return item["type"] in MODEL_TYPES or item["type"] in NODE_TYPES and (
        nodes == "all" or nodes == "stablegen" and bool(item.get("stablegen_node")))


def removal_plan(comfyui_path: Path, tags: List[str], busy_tags: List[str] = (), nodes: str = "") -> List[Path]:
    """What removing these package tags deletes: their model files, Hugging Face model folders and unfinished
    downloads, and the nodes the nodes policy names (removed_with_package). Anything they share stays while another
    package still uses it: one with something of its own on disk, or one being installed (busy_tags). Only paths
    inside the ComfyUI folder."""
    candidates = [i for i in DEPENDENCIES.values() if removed_with_package(i, nodes) and set(i["packages"]) & set(tags)]
    in_use = set(busy_tags) - set(tags)
    for item in DEPENDENCIES.values():
        if removed_with_package(item, nodes) and item not in candidates and item_present(item, comfyui_path):
            in_use |= set(item["packages"])
    root = os.path.normcase(os.path.realpath(str(comfyui_path)))
    paths = []
    for item in candidates:
        if set(item["packages"]) & in_use:
            continue
        target = item_target(item, comfyui_path)
        # a node's setup may download into data_dirs (LATO.2's weights in models/lato2): they go with the node
        extra = [comfyui_path / d for d in item.get("data_dirs", [])] if item["type"] in NODE_TYPES else []
        for path in ((target, Path(str(target) + ".part")) if item["type"] in MODEL_TYPES else (target, *extra)):
            inside = os.path.normcase(os.path.realpath(str(path))).startswith(root + os.sep)
            if inside and path.exists():
                paths.append(path)
    return paths


def unfinished_downloads(items: List[Dict[str, Any]], comfyui_path: Path) -> List[str]:
    """.part files of these models: downloads that stopped and resume on the next run (a Hugging Face model's are
    inside its folder)."""
    parts = [Path(str(item_target(i, comfyui_path)) + ".part") for i in items if i["type"] == "model"]
    found = [str(p) for p in parts if p.exists()]
    for item in items:
        folder = item_target(item, comfyui_path)
        if item["type"] == "hf_model" and folder.is_dir():
            found += sorted(str(p) for p in folder.rglob("*.part"))
    return found


def status_report(comfyui_path: Path) -> Dict[str, Any]:
    """--status: everything the add-on shows about this ComfyUI, read without changing anything. 'packages' has
    the menu packages and 'tags' every package tag on its own (what the add-on installs)."""
    return {"comfyui": str(comfyui_path), "main_py": (comfyui_path / "main.py").is_file(),
            "python": probe_environment(comfyui_path), "git": git_executable(), **package_report(comfyui_path)}


def package_report(comfyui_path: Path) -> Dict[str, Any]:
    """The part of --status about the packages and free space: files only, so the add-on also reads it in-process
    whenever its model lists refresh (ComfyUI's Python is not run)."""
    packages = {}
    for key, package in [*MENU_PACKAGES.items(), *((tag, package_info(tag)) for tag in TAG_LABELS)]:
        items = resolve_items([key])
        present = sum(item_present(i, comfyui_path) for i in items)
        packages[key] = {
            "name": package["name"], "tags": package["tags"], "size_gb": package["size_gb"],
            "notice": package.get("notice"), "present": present, "total": len(items),
            "models_present": sum(item_present(i, comfyui_path) for i in items if i["type"] in MODEL_TYPES),
            "models_total": sum(i["type"] in MODEL_TYPES for i in items),
            "state": "installed" if present == len(items) else "partial" if present else "missing",
            "download_mb": estimate_download_mb(items, package["tags"], comfyui_path),
            "missing_core_nodes": sorted({n for ns in missing_core_nodes(comfyui_path, package["tags"]).values()
                                          for n in ns}),
            "unfinished": unfinished_downloads(items, comfyui_path),
        }
    return {"free_mb": shutil.disk_usage(comfyui_path).free // 1024**2,
            "packages": {k: v for k, v in packages.items() if k in MENU_PACKAGES},
            "tags": {k: v for k, v in packages.items() if k in TAG_LABELS}}


def _distribution(spec: str) -> str:
    """'trimesh[easy]>=4' -> 'trimesh'."""
    return re.match(r"[A-Za-z0-9_.\-]+", spec.strip()).group(0)


def plan_report(comfyui_path: Path, keys: List[str]) -> Dict[str, Any]:
    """--plan: what installing these packages would add to and change in this ComfyUI. Nothing is changed."""
    items = resolve_items(keys)
    tags = tags_for(keys)
    nodes, pip, patches, steps, models = [], [], [], [], []
    git = git_executable()
    for item in items:
        target = item_target(item, comfyui_path)
        if item["type"] == "embedded_node":
            same = all((target / r).is_file() and (target / r).read_text(encoding="utf-8") == text
                       for r, text in item["files"].items())
            nodes.append({"id": item["id"], "name": item["name"], "path": str(target),
                          "action": "keep" if same else "update" if target.is_dir() else "write",
                          "commit": None, "head": None, "license": item.get("license")})
            continue
        if item["type"] != "node":
            part = target.with_name(target.name + ".part")
            state = ("present" if item_present(item, comfyui_path) else
                     "resume" if item["type"] == "model" and part.exists() else "download")
            models.append({"id": item["id"], "name": item["name"], "path": str(target),
                           "size_mb": item.get("size_mb", 0), "state": state, "license": item.get("license")})
            continue
        pinned = item.get("commit")
        head = None
        if not target.is_dir():
            action = "clone"
        elif not (target / ".git").exists():
            action = "keep"         # installed without Git; left as is
        else:
            if git:
                try:
                    head = _git(["rev-parse", "HEAD"], target).stdout.strip() or None
                except (OSError, subprocess.TimeoutExpired):
                    head = None
            action = ("keep" if not pinned or head == pinned else
                      "move to pin" if item.get("enforce_pin") else "different commit")
        nodes.append({"id": item["id"], "name": item["name"], "path": str(target), "action": action,
                      "commit": pinned, "head": head, "license": item.get("license")})
        for spec in item.get("pip_packages", []):
            pip.append({"spec": spec, "distribution": _distribution(spec), "for": item["name"]})
        for p in item.get("post_clone_patches", []):
            patches.append({"node": item["name"], "file": p["file"]})
        if item.get("run_install_script"):
            steps.append({"node": item["name"], "step": "runs the node's install.py"})
        if item.get("post_install"):
            steps.append({"node": item["name"], "step": POST_INSTALL[item["post_install"]][0]})
    env = probe_environment(comfyui_path, sorted({p["distribution"] for p in pip}))
    for p in pip:
        p["installed"] = env.get("packages", {}).get(p["distribution"])
    return {"comfyui": str(comfyui_path), "packages": keys,
            "notices": list(dict.fromkeys(package_info(k)["notice"] for k in keys if package_info(k).get("notice"))),
            "nodes": nodes, "pip": pip, "patches": patches, "steps": steps, "models": models,
            "changes_python": bool(pip or steps), "git_missing": bool(nodes) and git is None,
            "python": {k: env.get(k) for k in ("path", "detected", "version", "torch", "error")},
            "download_mb": estimate_download_mb(items, tags, comfyui_path),
            "free_mb": shutil.disk_usage(comfyui_path).free // 1024**2,
            "missing_core_nodes": missing_core_nodes(comfyui_path, tags)}


def print_json(document: Dict[str, Any]):
    stream = _EVENT_STREAM or sys.stdout
    stream.write(json.dumps(document, indent=1, ensure_ascii=False) + "\n")
    stream.flush()


# --- Summary ---
STATUS_LABELS ={"installed": "installed", "updated": "updated", "present": "present", "warning": "WARNING",
                 "failed": "FAILED"}


def print_summary(rows: List[Tuple[str, str, str]]) -> bool:
    print_header("Summary")
    for name, status, message in rows:
        print(f"  {STATUS_LABELS[status]:<10} {name}" + (f" - {message}" if message else ""))
    counts = {s: sum(1 for _, st, _ in rows if st == s) for s in STATUS_LABELS}
    print_separator()
    print(", ".join(f"{n} {STATUS_LABELS[s].lower()}" for s, n in counts.items() if n))
    return counts["failed"] == 0


# --- Main Script Logic ---
def choose_packages_interactively(comfyui_path: Path) -> Optional[List[str]]:
    while True:
        display_menu(comfyui_path)
        try:
            choice = input("Enter one or more package numbers (e.g. 3 or 3,8), or q to quit: ").strip().lower()
        except EOFError:
            return None
        if choice == 'q':
            return None
        try:
            return parse_package_choice(choice)
        except ValueError as e:
            print(f"Invalid choice: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    """The CLI. --events/--json switch to the machine-readable modes; --log moves every other line to a file."""
    global ASSUME_YES, PYTHON_OVERRIDE, NONINTERACTIVE, EVENTS, ACCEPTED_NOTICES, IGNORE_DISK_SPACE, GIT_EXE
    global CHILD_IO, _EVENT_STREAM, _CURRENT_ITEM
    args = parse_args(argv)
    if args.list:
        print_package_list()
        return 0
    EVENTS = args.events
    NONINTERACTIVE = args.events or args.json
    ACCEPTED_NOTICES = {k.strip() for k in args.accept_notices.split(",") if k.strip()}
    IGNORE_DISK_SPACE = args.ignore_disk_space
    if args.git:
        GIT_EXE = str(Path(args.git).resolve())
        # node install scripts and pip (git+ URLs) look for git on PATH too
        path = str(Path(GIT_EXE).parent) + os.pathsep + os.environ.get("PATH", "")
        os.environ["PATH"] = UTF8_ENV["PATH"] = path
    saved = sys.stdout, sys.stderr
    log = None
    try:
        if NONINTERACTIVE:
            _EVENT_STREAM = sys.stdout
            if args.log:
                Path(args.log).parent.mkdir(parents=True, exist_ok=True)
                log = open(args.log, "a", encoding="utf-8", errors="replace", buffering=1)
                sys.stdout = sys.stderr = log
                CHILD_IO = {"stdout": log, "stderr": subprocess.STDOUT}
            elif args.json:
                sys.stdout = sys.stderr      # the JSON document is the only thing on stdout
        return _main(args)
    finally:
        sys.stdout, sys.stderr = saved
        if log:
            log.close()
        # in-process callers (tests, the add-on reading reports) get the defaults back
        NONINTERACTIVE = EVENTS = IGNORE_DISK_SPACE = False
        ACCEPTED_NOTICES, CHILD_IO, _EVENT_STREAM, _CURRENT_ITEM, GIT_EXE = set(), {}, None, "", "git"


def _fail(code: str, message: str, exit_code: int = 1, **details) -> int:
    """An error that ends the run: printed, and reported as an event or JSON document."""
    print(f"ERROR: {message}")
    emit("error", code=code, message=message, **details)
    if NONINTERACTIVE and not EVENTS:
        print_json({"error": code, "message": message, **details})
    return exit_code


def _main(args: argparse.Namespace) -> int:
    global ASSUME_YES, PYTHON_OVERRIDE, _CURRENT_ITEM
    ASSUME_YES = args.yes
    if args.create_comfyui:
        return create_comfyui(Path(args.create_comfyui), dry_run=args.dry_run)
    if args.python:
        if not Path(args.python).is_file():
            return _fail("python", f"--python '{args.python}' does not exist.", 2)
        PYTHON_OVERRIDE = str(Path(args.python).resolve())
    keys: Optional[List[str]] = None
    if args.packages:
        try:
            keys = parse_package_choice(args.packages)
        except ValueError as e:
            return _fail("args", str(e), 2)
    if NONINTERACTIVE and keys is None and not (args.status or args.verify or args.repair or args.check):
        return _fail("args", "--events and --plan need --packages", 2)

    try:
        comfyui_base_path = get_comfyui_path_from_args(args)
    except SystemExit as e:
        if not NONINTERACTIVE:
            raise
        return _fail("path", str(e.code).replace("Error: ", ""))
    print(f"ComfyUI: {comfyui_base_path}")
    if args.status:
        print_json(status_report(comfyui_base_path))
        return 0
    if args.plan:
        print_json(plan_report(comfyui_base_path, keys))
        return 0
    if not describe_python(comfyui_base_path) and not args.check:
        if NONINTERACTIVE:
            return _fail("python", "ComfyUI's own Python was not found; pass --python")
        if not ask("Continue with this Python?", default_yes=False):
            return 1

    if args.check or args.repair:
        print("Nothing to check or repair: StableGen now uses ComfyUI's native TRELLIS.2 nodes, which have no "
              "environment of their own. The old ComfyUI-TRELLIS2 custom node can be deleted from custom_nodes.")
        return 0

    if args.verify and keys is None:
        tags, partial = installed_tags(comfyui_base_path)
        if partial:
            print("Partly installed, not checked: " + ", ".join(
                f"{TAG_LABELS[t]} ({have} of {total} components)" for t, (have, total) in partial.items()))
        if not tags:
            print("No StableGen package is completely installed here yet.")
            return 1
        print("Checking: " + ", ".join(TAG_LABELS[t] for t in tags))
        return 0 if verify_installation(comfyui_base_path, tags, args.server, args.verify_timeout) else 1

    if keys is None:
        keys = choose_packages_interactively(comfyui_base_path)
        if keys is None:
            print("Exiting installer.")
            return 0

    # License notices can drop packages
    accepted = []
    answers: Dict[str, bool] = {}       # a notice shared by two packages (Qwen 2.1 and Turbo) is asked once
    for key in keys:
        package = package_info(key)
        notice = package.get("notice")
        if notice and notice not in answers:
            answers[notice] = show_notice(notice)
        if notice and not answers[notice]:
            print(f"Skipping package {key}.")
            continue
        accepted.append(key)
    refused = NONINTERACTIVE and len(accepted) < len(keys)     # reported as error events above
    if not accepted:
        print("Nothing to install.")
        emit("summary", ok=not refused, rows=[], restart=False, dry_run=args.dry_run)
        return 1 if refused else 0
    tags = tags_for(accepted)
    items = resolve_items(accepted)

    print_separator()
    for key in accepted:
        print(f"Package {key}: {package_info(key)['name']}")
    print("Components:")
    for item in items:
        state = "present" if item_present(item, comfyui_base_path) else "to install"
        print(f"  - {item['name']} ({state})")
    need_mb = estimate_download_mb(items, tags, comfyui_base_path)
    free_mb = shutil.disk_usage(comfyui_base_path).free // 1024**2
    print(f"Estimated download: ~{need_mb / 1024:.1f} GB; free on this drive: {free_mb / 1024:.1f} GB")
    emit("plan", packages=accepted, download_mb=need_mb, free_mb=free_mb,
         items=[{"id": i["id"], "name": i["name"], "type": i["type"], "size_mb": i.get("size_mb", 0),
                 "present": item_present(i, comfyui_base_path)} for i in items])

    for tag, absent in missing_core_nodes(comfyui_base_path, tags).items():
        print(f"WARNING: this ComfyUI lacks {', '.join(absent)}; update ComfyUI before using these models.")
    if any(item["type"] == "node" for item in items) and git_executable() is None:
        return _fail("git_missing", "Git is needed to install custom nodes and was not found on PATH "
                                    "(https://git-scm.com).")
    if need_mb > free_mb:
        message = f"not enough free disk space ({need_mb / 1024:.1f} GB needed, {free_mb / 1024:.1f} GB free)"
        if IGNORE_DISK_SPACE:
            print(f"WARNING: {message}; continuing (--ignore-disk-space).")
        elif NONINTERACTIVE:
            return _fail("disk_space", message, need_mb=need_mb, free_mb=free_mb)
        else:
            print(f"ERROR: {message}.")
            if not ask("Continue anyway?", default_yes=False, yes_answers=False):
                return 1

    if args.dry_run:
        print("Dry run: nothing was changed.")
        emit("summary", ok=not refused, rows=[], restart=False, dry_run=True)
        if args.verify:
            return 0 if verify_installation(comfyui_base_path, tags, args.server, args.verify_timeout) else 1
        return 1 if refused else 0
    if not NONINTERACTIVE and not ask("Proceed with checking/installing these components?"):
        print("Installation cancelled.")
        return 0

    rows: List[Tuple[str, str, str]] = []
    for item_details in items:
        print_separator(char='.')
        print(f"Processing: {item_details['name']}")
        _CURRENT_ITEM = item_details["id"]
        emit("item", id=item_details["id"], name=item_details["name"], phase="start")
        if item_details["type"] == "node":
            item_rows = install_node(item_details, comfyui_base_path, args.update_nodes)
        elif item_details["type"] == "embedded_node":
            item_rows = [(item_details["name"],) + install_embedded_node(item_details, comfyui_base_path)]
        elif item_details["type"] == "model":
            item_rows = [(item_details["name"],) + download_file(item_details, comfyui_base_path, args.verify_hashes)]
        else:
            status, message = download_hf_model(item_details, comfyui_base_path, args.verify_hashes)
            print(f"  {status}{': ' + message if message else ''}")
            item_rows = [(item_details["name"], status, message)]
        rows.extend(item_rows)
        statuses = [status for _, status, _ in item_rows]
        worst = next((s for s in ("failed", "warning", "updated", "installed") if s in statuses), "present")
        emit("item", id=item_details["id"], name=item_details["name"], phase="done", status=worst,
             rows=[{"name": n, "status": s, "message": m} for n, s, m in item_rows])
    _CURRENT_ITEM = ""

    ok = print_summary(rows) and not refused
    node_names = {item["name"] for item in items if item["type"] in ("node", "embedded_node")}
    restart = any(status in ("installed", "updated") and name in node_names for name, status, _ in rows)
    if restart:
        print("Restart ComfyUI so it loads new or updated custom nodes.")
    emit("summary", ok=ok, rows=[{"name": n, "status": s, "message": m} for n, s, m in rows], restart=restart,
         dry_run=False)
    if args.verify:
        ok = verify_installation(comfyui_base_path, tags, args.server, args.verify_timeout) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    # stdout carries only JSON in the machine-readable modes
    if not TQDM_AVAILABLE and not {"--events", "--json", "--status", "--plan"} & set(sys.argv):
        print("NOTE: 'tqdm' is not installed, so downloads show a progress line instead of a bar.")
    sys.exit(main())
