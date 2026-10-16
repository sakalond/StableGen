#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import requests
import glob
import re
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

_PATCH_TRELLIS_PRESTARTUP = (
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
    "# -- StableGen patch: Auto-heal comfy-env site-packages isolation at startup --\n"
    "try:\n"
    "    import importlib.util as _u, os as _os, sys as _sys\n"
    "    _s = _u.find_spec('comfy_env')\n"
    "    if _s and _s.origin:\n"
    "        _wrap_py = _os.path.join(_os.path.dirname(_s.origin), 'isolation', 'wrap.py')\n"
    "        if _os.path.exists(_wrap_py):\n"
    "            with open(_wrap_py, 'r', encoding='utf-8') as _f:\n"
    "                _code = _f.read()\n"
    "            if 'PYTHONNOUSERSITE' not in _code:\n"
    "                _anchor = '    env[\"COMFYUI_ISOLATION_WORKER\"] = \"1\"'\n"
    "                if _anchor in _code:\n"
    "                    _code = _code.replace(_anchor, _anchor + '\\n    env[\"PYTHONNOUSERSITE\"] = \"1\"')\n"
    "                    with open(_wrap_py, 'w', encoding='utf-8') as _f:\n"
    "                        _f.write(_code)\n"
    "                    print('[StableGen] Auto-applied comfy-env user site-packages isolation patch at boot', file=_sys.stderr)\n"
    "except Exception:\n"
    "    pass\n"
    "# -- End StableGen patch --\n\n"
)

# --- comfy-env core sync ---
# comfy-env runs nodes like ComfyUI-TRELLIS2 in isolated pixi environments,
# but their workers still import ComfyUI's own `comfy` package. ComfyUI core
# pins some packages exactly (e.g. comfy-aimdo, comfy-kitchen), while the
# node's comfy-env.toml asks for "*" and pixi's lock keeps whatever was current
# at install time. After a ComfyUI update the worker then fails with errors
# like "No module named 'comfy_aimdo.storage'".
#
# This module brings those pins into every isolated env of a node. The
# installer writes it into the node folder, runs it with ComfyUI's Python, and
# hooks it into the node's prestartup script so it re-checks on every ComfyUI
# start (cheap: it only reads dist-info metadata unless something is out of date).
_COMFY_ENV_SYNC_FILENAME = "stablegen_env_sync.py"
_COMFY_ENV_SYNC_SOURCE = r'''"""Keep comfy-env isolated environments in step with ComfyUI core's pinned packages.

Written by StableGen's installer. Run as:
    python stablegen_env_sync.py <ComfyUI dir> <custom node dir> [--check]
or import and call sync(comfyui_dir, node_dir). Safe to run repeatedly.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TAG = "[StableGen env sync]"
SKIP_DIRS = {".git", ".pixi", "node_modules", "__pycache__"}


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _log(msg):
    print(f"{TAG} {msg}", file=sys.stderr, flush=True)


try:
    from packaging.specifiers import SpecifierSet
except ImportError:
    try:
        from pip._vendor.packaging.specifiers import SpecifierSet
    except ImportError:
        SpecifierSet = None


def _satisfies(version, spec):
    if SpecifierSet is None:  # no packaging library: only exact pins can be compared
        return not spec.startswith("==") or version == spec[2:]
    try:
        return SpecifierSet(spec).contains(version, prereleases=True)
    except Exception:
        return True


def _main_version(name):
    """Version installed in the Python running this module (ComfyUI's own environment)."""
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def core_requirements(comfyui_dir):
    """Versioned requirements from ComfyUI's requirements.txt, as {name: specifier}.

    Workers import ComfyUI's `comfy` and `comfy_api` packages, so whatever version
    rules ComfyUI core sets must hold inside isolated envs too. Unversioned lines
    are skipped, and front-end/template packages (comfyui-*) only matter to the
    server, not to workers.
    """
    req = Path(comfyui_dir) / "requirements.txt"
    reqs = {}
    if not req.is_file():
        return reqs
    for line in req.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*((?:[<>=!~]=?|===)\s*[^\s,]+(?:\s*,\s*(?:[<>=!~]=?)\s*[^\s,]+)*)", line)
        if m and not _norm(m.group(1)).startswith("comfyui-"):
            reqs[_norm(m.group(1))] = re.sub(r"\s+", "", m.group(2))
    return reqs


def _target(name, spec):
    """What to install for a requirement: ComfyUI's own version when it satisfies the rule."""
    main = _main_version(name)
    return f"{name}=={main}" if main and _satisfies(main, spec) else f"{name}{spec}"


def _workspace_root():
    override = os.environ.get("COMFY_ENV_ROOT")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "Programs" / "comfy-env"
    return Path.home() / ".ce"


def _env_name(node_dir, config_path):
    """comfy-env's naming rule; used only when comfy_env itself can't be imported."""
    plugin = Path(node_dir).name
    for prefix in ("ComfyUI-", "ComfyUI_", "comfyui-", "comfyui_"):
        if plugin.startswith(prefix):
            plugin = plugin[len(prefix):]
            break
    clean = lambda s: re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", s.lower())).strip("-")
    name = clean(plugin)
    parent = Path(config_path).parent.resolve()
    if parent != Path(node_dir).resolve():
        suffix = clean(parent.name)
        if suffix:
            name = f"{name}-{suffix}" if name else suffix
    return name


def env_dirs(node_dir):
    """(config file, materialized env dir) for each isolated env of a node."""
    node_dir = Path(node_dir)
    configs = []
    for root, dirs, files in os.walk(node_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith("_env_")]
        if "comfy-env.toml" in files:
            configs.append(Path(root) / "comfy-env.toml")
    try:
        from comfy_env.environment.cache import get_env_name, get_workspace_env_dir
        resolve = lambda cfg: get_workspace_env_dir(None, get_env_name(node_dir, cfg))
    except Exception:
        resolve = lambda cfg: _workspace_root() / "envs" / _env_name(node_dir, cfg) / ".pixi" / "envs" / "default"
    return [(cfg, Path(resolve(cfg))) for cfg in configs]


def _env_python(env_dir):
    exe = env_dir / ("python.exe" if sys.platform == "win32" else "bin/python")
    return exe if exe.is_file() else None


def _site_packages(env_dir):
    return [p for p in [env_dir / "Lib" / "site-packages", *env_dir.glob("lib/python3*/site-packages")] if p.is_dir()]


def installed_versions(env_dir):
    from importlib import metadata
    versions = {}
    for dist in metadata.distributions(path=[str(p) for p in _site_packages(env_dir)]):
        name = dist.metadata.get("Name")
        if name:
            versions[_norm(name)] = dist.version
    return versions


def declared_packages(config_path):
    """Package names listed under any *pypi-dependencies* table of a comfy-env.toml."""
    names, active = set(), False
    for line in Path(config_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        header = re.fullmatch(r"\[([^\]]+)\]", line)
        if header:
            active = "pypi-dependencies" in header.group(1)
            continue
        m = re.match(r"([A-Za-z0-9_.\-]+)\s*=", line)
        if active and m:
            names.add(_norm(m.group(1)))
    return names


def sync(comfyui_dir, node_dir, check_only=False):
    """Make the node's envs satisfy ComfyUI core's requirements. Returns a list of problems left over.

    Only packages an env already has (or declares) are touched, and only when their
    version breaks ComfyUI's rule; they get the version ComfyUI's own env runs.
    """
    reqs = core_requirements(comfyui_dir)
    if not reqs:
        _log(f"no versioned requirements found in {Path(comfyui_dir) / 'requirements.txt'}; nothing to sync")
        return []
    problems = []
    for config, env_dir in env_dirs(node_dir):
        python = _env_python(env_dir)
        if not python:
            continue  # not materialized yet; comfy-env builds it on first use
        installed = installed_versions(env_dir)
        relevant = installed.keys() | declared_packages(config)
        wanted = {n: _target(n, s) for n, s in reqs.items()
                  if n in relevant and not (n in installed and _satisfies(installed[n], s))}
        if not wanted:
            continue
        summary = ", ".join(f"{n} {installed.get(n, 'missing')} -> {t[len(n):]}" for n, t in wanted.items())
        if check_only:
            problems.append(f"{env_dir.parent.parent.parent.name}: {summary}")
            continue
        _log(f"{env_dir.parent.parent.parent.name}: {summary}")
        # Hold every other package at its current version so pip can't swap the
        # env's CUDA torch build or anything else while resolving.
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write("\n".join(f"{n}=={v}" for n, v in installed.items() if n not in wanted))
            constraints = fh.name
        try:
            # Packages pixi took from conda (e.g. av) may have no build for this
            # Python that satisfies ComfyUI; pip replaces them with the PyPI wheel.
            result = subprocess.run(
                [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                 "-c", constraints, *wanted.values()],
                env=dict(os.environ, PYTHONNOUSERSITE="1"), capture_output=True, text=True, timeout=900)
        finally:
            os.unlink(constraints)
        after = installed_versions(env_dir)
        left = [t for n, t in wanted.items() if not (n in after and _satisfies(after[n], reqs[n]))]
        if result.returncode != 0 or left:
            tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
            problems.append(f"{env_dir}: could not install {', '.join(left) or 'all packages'}: "
                            + " | ".join(tail))
        else:
            _log(f"{env_dir.parent.parent.parent.name}: up to date")
    for p in problems:
        _log(("OUT OF DATE " if check_only else "FAILED ") + p)
    return problems


_VERIFY = (
    "import sys; sys.argv = ['main.py']; sys.path[:0] = [{comfy!r}]\n"
    "import comfy.model_management, comfy.utils, comfy_api.latest\n"
    "print('ok')\n"
)


def verify(comfyui_dir, node_dir):
    """Import ComfyUI core with each env's own Python, the way the node's worker does."""
    problems = []
    for _, env_dir in env_dirs(node_dir):
        python = _env_python(env_dir)
        if not python:
            continue
        result = subprocess.run([str(python), "-c", _VERIFY.format(comfy=str(Path(comfyui_dir).resolve()))],
                                env=dict(os.environ, PYTHONNOUSERSITE="1"), capture_output=True, text=True,
                                timeout=600, cwd=str(comfyui_dir))
        if result.returncode != 0:
            output = (result.stderr or result.stdout).strip()
            last = output.splitlines()[-1:] or ["unknown error"]
            if "WinError 1455" in output or "MemoryError" in output:
                # Not an env problem: the machine is out of memory (e.g. a generation is running).
                _log(f"{env_dir.parent.parent.parent.name}: could not verify, system is low on memory; "
                     "close other GPU/RAM-heavy programs and re-run --check")
                continue
            problems.append(f"{env_dir.parent.parent.parent.name}: ComfyUI core does not import: {last[0]}")
        else:
            _log(f"{env_dir.parent.parent.parent.name}: ComfyUI core imports cleanly")
    for p in problems:
        _log("FAILED " + p)
    return problems


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    left = sync(args[0], args[1], check_only="--check" in sys.argv)
    if not left:
        left = verify(args[0], args[1])
    sys.exit(1 if left else 0)
'''

_PATCH_COMFY_ENV_SYNC_PRESTARTUP = (
    "# -- StableGen patch: keep comfy-env envs in step with ComfyUI core --\n"
    "try:\n"
    "    import os as _os, sys as _sys\n"
    "    _here = _os.path.dirname(_os.path.abspath(__file__))\n"
    "    _sys.path.insert(0, _here)\n"
    "    try:\n"
    "        import stablegen_env_sync as _sg_env_sync\n"
    "        _sg_env_sync.sync(_os.path.dirname(_os.path.dirname(_here)), _here)\n"
    "    finally:\n"
    "        _sys.path.remove(_here)\n"
    "except Exception as _e:\n"
    "    print(f'[StableGen env sync] skipped: {_e}', file=_sys.stderr)\n"
    "# -- End StableGen patch --\n\n"
)

# Older StableGen releases also patched nodes/trellis_utils/ and nodes/trellis2/pipelines/ (VRAM-leak and
# temp-file fixes). Upstream replaced that code with ComfyUI's native model loading in February 2026, so those
# patches no longer have anything to apply to and were removed when the node was pinned below.

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
        "pip_packages": ["diffusers>=0.28", "accelerate", "matplotlib"] + (["triton-windows"] if sys.platform == "win32" else []),
        "run_install_script": True,
    },
    # --- StableDelight (specular-free albedo) ---
    "cn_comfyui_stabledelight": {
        "id": "cn_comfyui_stabledelight", "type": "node", "name": "ComfyUI StableDelight (Delighting)",
        "git_url": "https://github.com/lldacing/ComfyUI_StableDelight_ll.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI_StableDelight_ll",
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
    # --- TRELLIS.2 ---
    "cn_trellis2": {
        "id": "cn_trellis2", "type": "node", "name": "ComfyUI TRELLIS.2",
        "git_url": "https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-TRELLIS2",
        # Verified with ComfyUI 0.37 and comfy-env 0.4.1 (October 2026). Upstream refactors
        # regularly break the patches below, so move this pin only after re-testing.
        "commit": "9b878516f2dc2fd873f4f6cceadba403dd12d83e",
        "license": "MIT (Note: textured pipeline uses NVIDIA non-commercial libs)", "packages": ["trellis2"],
        "pip_packages": ["comfy-env", "comfy-sparse-attn", "comfy-3d-viewers"],
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
                "marker": "StableGen patch: Auto-heal comfy-env",
                "anchor": "from comfy_env import",
                "patch": _PATCH_TRELLIS_PRESTARTUP,
            },
            {
                "file": "prestartup_script.py",
                "marker": "StableGen patch: keep comfy-env envs in step",
                "anchor": "from comfy_env import",
                "patch": _PATCH_COMFY_ENV_SYNC_PRESTARTUP,
            },
            {
                "file": "nodes/rembg/BiRefNet.py",
                "marker": "StableGen patch: BiRefNet fp32",
                "anchor": "self.model.eval()",
                "patch": "self.model.eval()\n        self.model.float()  # StableGen patch: BiRefNet fp32 — avoid fp16/fp32 dtype mismatch",
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
        "size_gb": 20.4,
        "description_suffix": "*Installs ComfyUI-TRELLIS2 custom node + isolated Python environment (~5 GB).*\n"
                              "    *TRELLIS.2 models (~15.4 GB) are downloaded automatically from HuggingFace on first use.*\n"
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
    '12': {"name": "[PBR DECOMPOSITION] Marigold V2 Models (Albedo + Normals)",
        "tags": ["pbr_marigold_v2"],
        "size_gb": 23.3,
        "description_suffix": "*Downloads Marigold V2 (Apache 2.0): the int8 Qwen Image Edit 2509 base (~19.5GB) plus albedo*\n"
                              "    *and normals LoRAs, VAEs and conditionings. Uses ComfyUI core nodes (September 2026 or newer).*\n"
                              "    *Roughness and metallic still need the Marigold IID node (option 9).*",
    },
}

# --- Helper Functions ---
def print_header(title: str):
    print(f"\n{'='*10} {title} {'='*10}")

def print_separator(char='-', length=70):
    print(char * length)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StableGen Dependency Installer Script.")
    parser.add_argument("comfyuipath", nargs='?', default=None,
                        help="Full path to your ComfyUI installation directory. If not provided, will be prompted.")
    parser.add_argument("--repair", action="store_true",
                        help="Repair installed TRELLIS.2 nodes without the menu: re-apply StableGen's patches and "
                             "sync their isolated environments with ComfyUI core. Run after updating ComfyUI or "
                             "the TRELLIS.2 node.")
    parser.add_argument("--check", action="store_true",
                        help="Only report whether installed TRELLIS.2 environments are out of date; change nothing.")
    return parser.parse_args()


def get_comfyui_path_from_args(args: argparse.Namespace) -> Path:

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


def install_pip_packages(pip_packages: List[str], comfyui_path: Path, force_reinstall: bool = False):
    """Install pip packages into ComfyUI's Python environment."""
    python_exe = find_comfyui_python(comfyui_path)
    print(f"  Installing pip packages into ComfyUI Python: {python_exe}")
    for pkg in pip_packages:
        print(f"    pip install {pkg} ...")
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install"] + (["--force-reinstall", "--no-deps"] if force_reinstall else []) + [pkg],
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


def _apply_all_comfy_env_patches(comfyui_path: Path):
    """Apply all comfy-env patches (platform tag, wheel fallback, site isolation, dist-info, validator)."""
    _patch_comfy_env_platform_tag(comfyui_path)
    _patch_comfy_env_wheel_fallback(comfyui_path)
    _patch_comfy_env_user_site_isolation(comfyui_path)
    _patch_comfy_env_distinfo_normalize(comfyui_path)
    _patch_comfy_env_validator(comfyui_path)


def _patch_comfy_env_validator(comfyui_path: Path):
    """Patch comfy-env's _validate_node_config to strip torch/torchvision instead of crashing."""
    python_exe = find_comfyui_python(comfyui_path)
    result = subprocess.run(
        [python_exe, "-c",
         "import importlib.util as u, os; s=u.find_spec('comfy_env'); print(os.path.join(os.path.dirname(s.origin), 'packages', 'toml_generator.py') if s and s.origin else '')"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  WARNING: Could not locate comfy_env toml_generator.py, skipping validator patch")
        return

    toml_gen_path = Path(result.stdout.strip().splitlines()[-1])
    if not toml_gen_path.is_file():
        print(f"  WARNING: {toml_gen_path} not found, skipping validator patch")
        return

    content = toml_gen_path.read_text(encoding="utf-8")
    marker = "# StableGen patch: bypass toml validation crash by stripping torch/torchvision"
    if marker in content:
        print("  comfy-env validator patch already applied")
        return

    old = (
        "def _validate_node_config(name: str, cfg: ComfyEnvConfig) -> None:\n"
        "    \"\"\"Reject node configs that try to redefine the workspace torch pin.\"\"\"\n"
        "    bad = [p for p in cfg.cuda_packages if p in _TORCH_PKGS]\n"
        "    if bad:\n"
        "        raise ValueError(\n"
        "            f\"[{name}] comfy-env.toml has {bad} under [cuda] packages. \"\n"
        "            \"Plain torch/torchvision/torchaudio are pinned workspace-wide \"\n"
        "            \"(replicated into every feature so the rattler cache dedupes). \"\n"
        "            \"Remove them from [cuda] packages -- keep only CUDA-only wheels there \"\n"
        "            \"(cumesh, flash-attn, cc_torch, nvdiffrast, etc.).\"\n"
        "        )"
    )

    if old not in content:
        old_alt = (
            "def _validate_node_config(name: str, cfg: ComfyEnvConfig) -> None:\n"
            "    \"\"\"Reject node configs that try to redefine the workspace torch pin.\"\"\"\n"
            "    bad = [p for p in cfg.cuda_packages if p in _TORCH_PKGS]\n"
            "    if bad:\n"
            "        raise ValueError(\n"
            "            f\"[{name}] comfy-env.toml has {bad} under [cuda] packages. \"\n"
            "            \"Plain torch/torchvision/torchaudio are pinned workspace-wide \"\n"
            "            \"(replicated into every feature so the rattler cache dedupes). \"\n"
            "            \"Remove them from [cuda] packages -- keep only CUDA-only wheels there \"\n"
            "            \"(cumesh, flash-attn, cc_torch, nvdiffrast, etc.).\"\n"
            "        )\n"
        )
        if old_alt in content:
            old = old_alt
        else:
            print("  WARNING: Could not find validator anchor in toml_generator.py (already modified?)")
            return

    new = (
        "def _validate_node_config(name: str, cfg: ComfyEnvConfig) -> None:\n"
        "    " + marker + "\n"
        "    bad = [p for p in cfg.cuda_packages if p in _TORCH_PKGS]\n"
        "    if bad:\n"
        "        for p in bad:\n"
        "            try:\n"
        "                cfg.cuda_packages.remove(p)\n"
        "            except ValueError:\n"
        "                pass\n"
        "        print(f'[comfy-env] [{name}] stripped plain torch/torchvision from [cuda] packages list', file=sys.stderr)\n"
    )

    content = content.replace(old, new)
    toml_gen_path.write_text(content, encoding="utf-8")
    print("  Patched comfy-env _validate_node_config() to bypass invalid torch/torchvision declarations")


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
         "import importlib.util as u, os; s=u.find_spec('comfy_env'); print(os.path.join(os.path.dirname(s.origin), 'packages', 'cuda_wheels.py') if s and s.origin else '')"],
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
         "import importlib.util as u, os; s=u.find_spec('comfy_env'); print(os.path.join(os.path.dirname(s.origin), 'packages', 'cuda_wheels.py') if s and s.origin else '')"],
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
        "            if _tv <= int(torch_short.replace('.', '')): continue",
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
         "import importlib.util as u, os; s=u.find_spec('comfy_env'); print(os.path.join(os.path.dirname(s.origin), 'isolation', 'wrap.py') if s and s.origin else '')"],
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


def _patch_comfy_env_distinfo_normalize(comfyui_path: Path):
    """Fix any dist-info folders crashing pixi rebuilds.

    If flex_gemm or nvdiffrec-render is installed in a pixi environment, its wheel creates a
    .dist-info map with an unnormalized hyphen (flex-gemm-...) which crashes
    subsequent uv pip install runs due to strict PEP440 string validation.
    We patch comfy_env to monkeypatch subprocess.run so this fix runs after every pip install.
    """
    python_exe = find_comfyui_python(comfyui_path)
    result = subprocess.run(
        [python_exe, "-c",
         "import importlib.util as u, os; s=u.find_spec('comfy_env'); print(os.path.join(os.path.dirname(s.origin), 'install.py') if s and s.origin else '')"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: Could not locate comfy_env install.py: {result.stderr}")
        return

    install_path = Path(result.stdout.strip().splitlines()[-1])
    if not install_path.is_file():
        print(f"  WARNING: comfy_env install.py does not exist at {install_path}")
        return

    content = install_path.read_text(encoding="utf-8")
    marker = "# StableGen patch v7: aggressive dist-info cleanup via subprocess monkeypatch"
    if marker in content:
        print("  comfy-env universal dist-info patch already applied")
        return

    if "# StableGen patch" in content:
        print("  Fixing previously broken comfy-env installation...")
        subprocess.run([python_exe, "-m", "pip", "install", "--force-reinstall", "--no-deps", "comfy-env==0.2.0"], check=True, capture_output=True)
        _patch_comfy_env_platform_tag(comfyui_path)
        _patch_comfy_env_wheel_fallback(comfyui_path)
        _patch_comfy_env_user_site_isolation(comfyui_path)
        content = install_path.read_text(encoding="utf-8")

    anchor = 'log(f"[comfy-env] build_dir={build_dir}")'
    if anchor not in content:
        print(f"  WARNING: Could not find anchor in comfy_env install.py")
        return

    # Get the correct indentation from the anchor dynamically
    _indent_match = re.search(r"([ \t]*)" + re.escape(anchor), content)
    _ind = _indent_match.group(1) if _indent_match else "    "

    # Format patch properly
    patch_lines = []
    
    # We use a raw string for the regex backslashes, but f-string needed for anchor/marker.
    # To keep it simple, we use a normal string and .replace() instead of f-string
    _raw_patch = r"""ANCHOR
MARKER
try:
    _orig_subprocess_run = subprocess.run
    def _patched_subprocess_run(*args, **kwargs):
        _res = _orig_subprocess_run(*args, **kwargs)
        try:
            import sys as _sys_patch
            import pathlib as _pathlib_patch
            import re as _re_patch
            def _fix_dist_info(_p):
                if not _p.is_dir() or not _p.name.endswith(".dist-info"): return
                _m = _re_patch.match(r"^([a-zA-Z0-9_\-]+?)-([0-9].*)\.dist-info$", _p.name)
                if _m and "-" in _m.group(1):
                    _new_name = _m.group(1).replace("-", "_") + "-" + _m.group(2) + ".dist-info"
                    _p.rename(_p.parent / _new_name)

            for _sp in _sys_patch.path:
                _sp_path = _pathlib_patch.Path(_sp)
                if _sp_path.is_dir() and "site-packages" in _sp_path.name:
                    for _p in _sp_path.glob("*.dist-info"):
                        _fix_dist_info(_p)

            _sp_dir = build_dir / ".pixi" / "envs" / "default" / ("Lib" if _sys_patch.platform == "win32" else "lib")
            if _sys_patch.platform != "win32":
                for _p in _sp_dir.glob("python*"): _sp_dir = _p
            _sp_dir = _sp_dir / "site-packages"
            if _sp_dir.exists():
                for _p in _sp_dir.iterdir():
                    _fix_dist_info(_p)
        except Exception as _e:
            log(f"Failed to normalize dist-info naming: {_e}")
        return _res
    subprocess.run = _patched_subprocess_run
except Exception as _e:
    pass"""
    
    _raw_patch = _raw_patch.replace("ANCHOR", anchor).replace("MARKER", marker)

    for i, line in enumerate(_raw_patch.split("\n")):
        if i == 0:
            patch_lines.append(line)
        else:
            patch_lines.append(_ind + line)
    patch = "\n".join(patch_lines)

    content = content.replace(anchor, patch)
    install_path.write_text(content, encoding="utf-8")
    print("  Patched comfy-env installation to monkeypatch subprocess.run for dist-info normalizations")


def _patch_flex_gemm_autotuner(repo_path: Path):
    """Fix flex_gemm TritonPersistentCacheAutotuner for triton 3.0+.

    flex_gemm 1.0.0 passes a ``do_bench`` argument to triton's
    ``Autotuner.__init__`` which was removed in triton 3.0.  Drop it so
    the call signature matches.
    """
    env_dirs = glob.glob(str(repo_path / "nodes" / "_env_*"))
    if not env_dirs:
        return
    for env_dir in env_dirs:
        matches = glob.glob(
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
        # comfy-env on disk (patches wiped).  Restore comfy-env
        # with patches before retrying so the new subprocess sees them.
        print("  Install failed. Restoring comfy-env + patches for retry...")
        install_pip_packages(["comfy-env"], comfyui_path, force_reinstall=True)
        _apply_all_comfy_env_patches(comfyui_path)

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


def _uses_comfy_env(item_details: Dict[str, Any]) -> bool:
    return any("comfy-env" in p for p in item_details.get("pip_packages", []))


def sync_comfy_env_core(item_details: Dict[str, Any], comfyui_path: Path, check_only: bool = False) -> bool:
    """Bring ComfyUI core's exact pins into a comfy-env node's isolated environments.

    Writes the sync module into the node folder (the prestartup patch imports it
    on every ComfyUI start) and runs it once now with ComfyUI's own Python, so it
    resolves env paths with the installed comfy-env. Returns True when everything
    is in step.
    """
    repo_path = comfyui_path / item_details["target_dir_relative"] / item_details["repo_name"]
    if not repo_path.is_dir():
        return True
    module_path = repo_path / _COMFY_ENV_SYNC_FILENAME
    if not module_path.is_file() or module_path.read_text(encoding="utf-8") != _COMFY_ENV_SYNC_SOURCE:
        module_path.write_text(_COMFY_ENV_SYNC_SOURCE, encoding="utf-8")
    python_exe = find_comfyui_python(comfyui_path)
    action = "Checking" if check_only else "Syncing"
    print(f"  {action} {item_details['name']} isolated environment(s) with ComfyUI core requirements...")
    try:
        result = subprocess.run(
            [python_exe, str(module_path), str(comfyui_path), str(repo_path)] + (["--check"] if check_only else []),
            cwd=str(repo_path), timeout=1800,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  WARNING: Environment sync did not complete: {e}")
        return False
    if result.returncode == 0:
        print("  Isolated environment(s) match ComfyUI core.")
        return True
    print("  WARNING: Some isolated packages are still out of date (see messages above).")
    print(f"  Re-run with:  python installer.py \"{comfyui_path}\" --repair")
    return False


def repair_comfy_env_nodes(comfyui_path: Path, check_only: bool = False) -> bool:
    """Re-apply StableGen's patches and env sync to every installed comfy-env node.

    Node updates (git pull, ComfyUI-Manager) overwrite patched files, and ComfyUI
    updates move core's pinned packages ahead of the isolated envs; both break
    TRELLIS.2 until repaired. Safe to run any number of times.
    """
    ok = True
    for item in DEPENDENCIES.values():
        if item.get("type") != "node" or not _uses_comfy_env(item):
            continue
        repo_path = comfyui_path / item["target_dir_relative"] / item["repo_name"]
        if not repo_path.is_dir():
            continue
        print_separator(char='.')
        print(f"{'Checking' if check_only else 'Repairing'}: {item['name']}")
        if not check_only:
            apply_post_clone_patches(item, comfyui_path)
        ok = sync_comfy_env_core(item, comfyui_path, check_only=check_only) and ok
    return ok


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
    args = parse_args()
    comfyui_base_path = get_comfyui_path_from_args(args)

    if args.check:
        print_header("Checking TRELLIS.2 environments")
        sys.exit(0 if repair_comfy_env_nodes(comfyui_base_path, check_only=True) else 1)

    # Ensure all comfy-env patches are applied if comfy-env is already installed
    has_comfy_env = False
    try:
        python_exe = find_comfyui_python(comfyui_base_path)
        result = subprocess.run(
            [python_exe, "-c", "import comfy_env"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            has_comfy_env = True
            print("  Detected existing comfy-env installation. Ensuring all patches are applied...")
            _apply_all_comfy_env_patches(comfyui_base_path)
    except Exception as e:
        print(f"  WARNING: Could not check/patch existing comfy-env: {e}")

    # Already-installed TRELLIS.2 nodes drift out of step whenever ComfyUI or
    # the node updates; repair them on every installer run.
    if has_comfy_env or args.repair:
        print_header("Repairing installed TRELLIS.2 nodes")
        ok = repair_comfy_env_nodes(comfyui_base_path)
        if args.repair:
            print("Repair complete." if ok else "Repair finished with warnings (see above).")
            print("Restart ComfyUI so the TRELLIS.2 worker picks up the changes.")
            sys.exit(0 if ok else 1)

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
            # Check target environment's Python version
            try:
                python_exe = find_comfyui_python(comfyui_base_path)
                py_version_res = subprocess.run(
                    [python_exe, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                    capture_output=True, text=True, check=True
                )
                py_ver = py_version_res.stdout.strip()
                is_python_313_plus = [int(x) for x in py_ver.split(".")] >= [3, 13]
            except Exception:
                is_python_313_plus = sys.version_info >= (3, 13)

            if is_python_313_plus:
                print_separator(char='!')
                print("WARNING: PYTHON 3.13 COMPATIBILITY ALERT")
                print_separator(char='!')
                print("You are running Python 3.13+ in ComfyUI.")
                print("TRELLIS.2 requires binary CUDA extensions (cumesh, flex_gemm_ap, nvdiffrec, etc.)")
                print("which DO NOT currently have pre-compiled wheels for Python 3.13 on the wheel index.")
                print("Compiling them from source requires a local MSVC compiler and CUDA Toolkit.")
                print()
                print("It is HIGHLY RECOMMENDED to use Python 3.12 (or a ComfyUI portable build")
                print("that uses Python 3.12) to run the TRELLIS.2 mesh generation pipeline.")
                print_separator(char='!')
                ack_313 = input("Do you still want to attempt installation on Python 3.13? (y/n): ").strip().lower()
                if ack_313 != 'y':
                    print("Installation cancelled.")
                    print_separator()
                    continue

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
            print("For full details, see docs/trellis2.md (Third-party licenses) in the StableGen repository.")
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
                    _apply_all_comfy_env_patches(comfyui_base_path)
                # Apply compatibility patches (e.g. Python 3.10 polyfills)
                if item_details.get("post_clone_patches"):
                    apply_post_clone_patches(item_details, comfyui_base_path)
                # Run node's install.py for full dependency resolution (e.g. CUDA wheels)
                if item_details.get("run_install_script"):
                    run_node_install_script(item_details, comfyui_base_path)
                    # Fix flex_gemm/triton version mismatch in comfy-env isolated envs
                    _patch_flex_gemm_autotuner(target_full_path)
                # Force comfy-env (latest) and re-apply patches AFTER install.py,
                # since the node's requirements.txt may pin an older version
                if item_details.get("pip_packages") and any("comfy-env" in p for p in item_details["pip_packages"]):
                    print("  Re-installing comfy-env (overriding node requirements.txt)...")
                    install_pip_packages(["comfy-env"], comfyui_base_path, force_reinstall=True)
                    _apply_all_comfy_env_patches(comfyui_base_path)
                    # The isolated env was just solved from the node's own manifest;
                    # bring ComfyUI core's exact pins into it.
                    sync_comfy_env_core(item_details, comfyui_base_path)
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
