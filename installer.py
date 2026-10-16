#!/usr/bin/env python3
"""StableGen dependency installer: custom nodes and models for a ComfyUI server.

    python installer.py <ComfyUI folder>                      menu
    python installer.py <ComfyUI folder> --packages 3,8 --yes unattended
    python installer.py --list                                 what each package installs

Needs only Python 3.8+ and Git (for custom nodes). `tqdm`, when installed, draws progress bars.
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
import glob
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

# comfy-env runs TRELLIS.2 in its isolated environment. The patches below and the pinned TRELLIS.2 commit were
# verified with this version; newer releases change the code the patches anchor on.
COMFY_ENV = "comfy-env==0.4.1"

# --- Configuration: Dependencies Data ---
# Sizes are in MB. Every custom node is pinned to a commit StableGen was tested with ("commit"). A fresh clone
# checks out that commit. An existing clone at another commit is left alone with a note, unless the node sets
# "enforce_pin" (StableGen's patches depend on that exact code) or the installer runs with --update-nodes.
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
    # --- TRELLIS.2 ---
    "cn_trellis2": {
        "id": "cn_trellis2", "type": "node", "name": "ComfyUI TRELLIS.2",
        "git_url": "https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-TRELLIS2",
        # Verified with ComfyUI 0.37 and comfy-env 0.4.1 (October 2026). Upstream refactors
        # regularly break the patches below, so move this pin only after re-testing.
        "commit": "9b878516f2dc2fd873f4f6cceadba403dd12d83e",
        "enforce_pin": True,
        "license": "MIT (Note: textured pipeline uses NVIDIA non-commercial libs)", "packages": ["trellis2"],
        "pip_packages": [COMFY_ENV, "comfy-sparse-attn", "comfy-3d-viewers"],
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
        "license": "Qwen Research License (non-commercial)", "size_mb": 6310, "packages": ["qwen_image_21"],
    },
    "model_qwen21_vae": {
        "id": "model_qwen21_vae", "type": "model", "name": "Qwen Image 2.1 VAE",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors?download=true",
        "target_path_relative": "models/vae", "filename": "qwen_image_2.1_vae_bf16.safetensors",
        "license": "Qwen Research License (non-commercial)", "size_mb": 680, "packages": ["qwen_image_21"],
    },
    # --- SkinTokens auto-rigging ---
    "cn_skintokens": {
        "id": "cn_skintokens", "type": "node", "name": "ComfyUI StableGen SkinTokens",
        "git_url": "https://github.com/sakalond/ComfyUI-StableGen-SkinTokens.git",
        "target_dir_relative": "custom_nodes",
        "repo_name": "ComfyUI-StableGen-SkinTokens",
        "commit": "dd0a272686b77226e6702281b6fd7eb24231abe6",
        "license": "GPL-3.0 (SkinTokens code and weights: MIT)", "packages": ["skintokens"],
        # The node's own setup builds SkinTokens in a separate environment (vendor/skintokens_env) with its own
        # PyTorch and downloads the checkpoints, so ComfyUI's Python is left untouched.
        "post_install": "skintokens_runtime",
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
    "trellis2": ["LoadTrellis2Models", "Trellis2ImageToShape", "Trellis2GetConditioning", "Trellis2RemoveBackground",
                 "Trellis2ExportTrimesh", "Trellis2Simplify", "Trellis2ShapeToTexturedMesh", "Trellis2RasterizePBR",
                 "Trellis2ProcessMesh"],
    "flux2_klein": ["Flux2Scheduler", "EmptyFlux2LatentImage"],
    "qwen_image_21": ["TextEncodeQwenImage21", "QwenImage21Cache"],
    "skintokens": ["StableGenSkinTokensAutoRig"],
}
CORE_NODES: Dict[str, List[str]] = {
    "qwen_core": ["TextEncodeQwenImageEditPlus"],
    "pbr_marigold_v2": ["MarigoldV2PostProcess", "ConditioningLoader"],
    "flux2_klein": ["Flux2Scheduler", "EmptyFlux2LatentImage"],
    "qwen_image_21": ["TextEncodeQwenImage21", "QwenImage21Cache"],
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
        "notice": "trellis2",
        "description_suffix": "*Installs ComfyUI-TRELLIS2 custom node + isolated Python environment (~5 GB).*\n"
                              "    *TRELLIS.2 models (~15.4 GB) are downloaded automatically from HuggingFace on first use.*\n"
                              "    *LICENSE NOTICE: Only the 'Native (TRELLIS.2)' texture mode uses nvdiffrast/nvdiffrec*\n"
                              "    *(NVIDIA Source Code License — non-commercial use only). Details in the TRELLIS.2 wiki page.*",
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
}

LICENSE_NOTICES: Dict[str, List[str]] = {
    "trellis2": [
        "TRELLIS.2 THIRD-PARTY LICENSE NOTICE",
        "This notice applies only to the TRELLIS.2 feature. StableGen's projection texturing is unaffected.",
        "",
        "ComfyUI-TRELLIS2 and the TRELLIS.2 model are MIT-licensed. The TRELLIS.2 Native texture mode uses",
        "NVIDIA's nvdiffrast and nvdiffrec, licensed for 'research or evaluation purposes only and not for any",
        "direct or indirect monetary gain' (Section 3.3). Only that texture mode uses them; shape-only and",
        "projection texture modes do not.",
        "",
        "Details: https://github.com/sakalond/StableGen/wiki/TRELLIS.2-mesh-generation#third-party-licenses",
    ],
    "qwen21": [
        "QWEN IMAGE 2.1 LICENSE NOTICE",
        "Qwen Image 2.1 is released under the Qwen Research License, which allows non-commercial use only",
        "(research or evaluation). Commercial use needs a separate license from the Qwen team.",
        "Qwen Image Edit 2509 (packages 5-7) is Apache 2.0 and not affected.",
        "",
        "License: https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
    ],
}

# --- Helper Functions ---
PYTHON_OVERRIDE: Optional[str] = None   # --python
ASSUME_YES = False                      # --yes


def print_header(title: str):
    print(f"\n{'='*10} {title} {'='*10}")

def print_separator(char='-', length=70):
    print(char * length)


def ask(question: str, default_yes: bool = True, yes_answers: bool = True) -> bool:
    """Ask a yes/no question. --yes answers it, unless yes_answers is False (then --yes means no)."""
    if ASSUME_YES:
        print(f"{question} {'yes' if yes_answers else 'no'} (--yes)")
        return yes_answers
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
                        help="Repair installed TRELLIS.2 nodes without the menu: re-apply StableGen's patches and "
                             "sync their isolated environments with ComfyUI core. Run after updating ComfyUI or "
                             "the TRELLIS.2 node.")
    parser.add_argument("--check", action="store_true",
                        help="Only report whether installed TRELLIS.2 environments are out of date; change nothing.")
    return parser.parse_args(argv)


def get_comfyui_path_from_args(args: argparse.Namespace) -> Path:
    comfyui_path_str = args.comfyuipath
    while not comfyui_path_str:
        if ASSUME_YES:
            sys.exit("Error: pass the ComfyUI folder as an argument when using --yes.")
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
        out = subprocess.run([python, "-c", _PYTHON_REPORT], capture_output=True, text=True, timeout=120)
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
                check=True,
            )
            print(f"    Successfully installed '{pkg}'.")
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: Failed to install '{pkg}' (exit code {e.returncode}).")
            print(f"    You may need to install it manually: pip install {pkg}")
            ok = False
        except FileNotFoundError:
            print(f"    ERROR: Python executable not found at '{python_exe}'.")
            print(f"    Please install '{pkg}' manually into your ComfyUI Python environment.")
            return False
    return ok


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
        subprocess.run([python_exe, "-m", "pip", "install", "--force-reinstall", "--no-deps", COMFY_ENV], check=True, capture_output=True)
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


def run_node_install_script(item_details: Dict[str, Any], comfyui_path: Path) -> bool:
    """Run a custom node's install.py to resolve all its dependencies. Returns True on success.

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
        return True

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
        return True
    except subprocess.CalledProcessError:
        if not _uses_comfy_env(item_details):
            print(f"  ERROR: install.py failed. Run it manually to see why:")
            print(f"    cd {repo_path}")
            print(f"    {python_exe} install.py")
            return False
        # The cuda-wheels index may not have wheels for the auto-detected
        # CUDA version yet (e.g. cu128).  Retry with CUDA 12.4 which has
        # the widest wheel coverage for flex_gemm / nvdiffrast.
        #
        # The first attempt's requirements.txt installs downgraded
        # comfy-env on disk (patches wiped).  Restore comfy-env
        # with patches before retrying so the new subprocess sees them.
        print("  Install failed. Restoring comfy-env + patches for retry...")
        install_pip_packages([COMFY_ENV], comfyui_path, force_reinstall=True)
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
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: install.py failed (exit code {e.returncode}).")
            print(f"  You may need to run it manually:")
            print(f"    cd {repo_path}")
            print(f"    {python_exe} install.py")
            return False
    except FileNotFoundError:
        print(f"  ERROR: Python executable not found at '{python_exe}'.")
        return False


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


def _progress(name: str, total: int, initial: int):
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

    if final_path.exists():
        local = final_path.stat().st_size
        if not remote_size or local == remote_size:
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
    if verify_hash and remote_sha:
        print(f"  Checking the SHA-256 of '{final_path.name}'...")
        if file_sha256(part) != remote_sha:
            part.unlink()
            return "failed", "SHA-256 mismatch; the download was deleted, run again"
    os.replace(part, final_path)
    return "installed", f"{size / 1024**2:,.0f} MB" + (", SHA-256 verified" if verify_hash and remote_sha else "")


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


def _git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True)


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
            if item_details.get("clean_envs"):
                _clean_stale_comfy_envs(repo_path)
            return "updated", f"{head[:12]} -> {pinned[:12]}"

        create_dir_if_not_exists(repo_path.parent)
        print(f"  Cloning: {item_details['name']} - License: {item_details['license']}")
        print(f"  From: {git_url}")
        print(f"  To:   {repo_path}")
        clone = subprocess.run(['git', 'clone', git_url, str(repo_path)], cwd=str(repo_path.parent))
        if clone.returncode != 0:
            return "failed", "git clone failed: check your network and that the repository is public"
        if pinned:
            checkout = _git(["checkout", pinned], repo_path)
            if checkout.returncode != 0:
                return "failed", f"cloned, but the tested commit {pinned[:12]} is missing upstream"
        return "installed", f"at {pinned[:12]}" if pinned else "latest commit"
    except FileNotFoundError:
        return "failed", "Git is not installed or not on PATH"


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
        result = subprocess.run([find_comfyui_python(comfyui_path), str(script), str(installers)])
    return result.returncode == 0, "" if result.returncode == 0 else "see the messages above"


def _install_skintokens_runtime(item_details: Dict[str, Any], comfyui_path: Path) -> Tuple[bool, str]:
    repo_path = comfyui_path / item_details["target_dir_relative"] / item_details["repo_name"]
    script = repo_path / "install_skintokens.py"
    if not script.is_file():
        return False, f"{script} not found"
    print("  Building the SkinTokens runtime (its own Python environment, PyTorch and checkpoints).")
    print("  This takes a while the first time.")
    result = subprocess.run([find_comfyui_python(comfyui_path), str(script), "--target", "managed"], cwd=str(repo_path))
    return result.returncode == 0, "" if result.returncode == 0 else "see the messages above"


POST_INSTALL = {
    "nunchaku_wheel": ("Nunchaku engine", _install_nunchaku_wheel),
    "skintokens_runtime": ("SkinTokens runtime", _install_skintokens_runtime),
}


def install_node(item_details: Dict[str, Any], comfyui_path: Path, update_nodes: bool) -> List[Tuple[str, str, str]]:
    """Clone or check a custom node, then install what it needs. Returns (name, status, message) rows."""
    name = item_details["name"]
    status, message = clone_git_repo(item_details, comfyui_path, update_nodes)
    print(f"  {status}{': ' + message if message else ''}")
    rows = [(name, status, message)]
    if status == "failed":
        return rows
    repo_path = comfyui_path / item_details["target_dir_relative"] / item_details["repo_name"]
    uses_env = _uses_comfy_env(item_details)
    if item_details.get("pip_packages") and not install_pip_packages(item_details["pip_packages"], comfyui_path):
        rows.append((f"{name}: Python packages", "failed", "pip install failed (see above)"))
    # comfy-env patches go in BEFORE install.py, which needs them (e.g. the Linux platform tag for wheel lookup)
    if uses_env:
        _apply_all_comfy_env_patches(comfyui_path)
    if item_details.get("post_clone_patches"):
        apply_post_clone_patches(item_details, comfyui_path)
    if item_details.get("run_install_script"):
        if not run_node_install_script(item_details, comfyui_path):
            rows.append((f"{name}: install.py", "failed", "the node's dependency install failed (see above)"))
        _patch_flex_gemm_autotuner(repo_path)
    if uses_env:
        # The node's requirements.txt may pin an older comfy-env: restore the latest and re-apply the patches,
        # then bring ComfyUI core's exact pins into the isolated env that install.py just solved.
        print(f"  Re-installing {COMFY_ENV} (overriding the node's requirements.txt)...")
        install_pip_packages([COMFY_ENV], comfyui_path, force_reinstall=True)
        _apply_all_comfy_env_patches(comfyui_path)
        if not sync_comfy_env_core(item_details, comfyui_path):
            rows.append((f"{name}: isolated environment", "warning",
                         "out of step with ComfyUI core; run the installer with --repair"))
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
    print("For FLUX.1 (manual download because of its license) see the manual installation guide:")
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
            kind = {"node": "node", "model": "model", "hf_model": "model"}[item["type"]]
            print(f"       {kind:<5} {item['name']}")


def parse_package_choice(text: str) -> List[str]:
    """'3, 8' -> ['3', '8']. Raises ValueError for unknown numbers."""
    keys = [k.strip() for k in text.replace(";", ",").split(",") if k.strip()]
    unknown = [k for k in keys if k not in MENU_PACKAGES]
    if not keys or unknown:
        raise ValueError(f"unknown package(s): {', '.join(unknown) or text!r}; choose from {', '.join(MENU_PACKAGES)}")
    return list(dict.fromkeys(keys))


def tags_for(keys: List[str]) -> List[str]:
    return list(dict.fromkeys(tag for key in keys for tag in MENU_PACKAGES[key]["tags"]))


def get_unique_item_ids_for_tags(selected_tags: List[str]) -> Set[str]:
    return {item_id for item_id, details in DEPENDENCIES.items()
            if any(tag in details["packages"] for tag in selected_tags)}


def resolve_items(keys: List[str]) -> List[Dict[str, Any]]:
    """The components of the chosen packages, each once, in DEPENDENCIES order."""
    ids = get_unique_item_ids_for_tags(tags_for(keys))
    return [item for item_id, item in DEPENDENCIES.items() if item_id in ids]


def item_target(item: Dict[str, Any], comfyui_path: Path) -> Path:
    if item["type"] == "node":
        return comfyui_path / item["target_dir_relative"] / item["repo_name"]
    if item["type"] == "model":
        return comfyui_path / item["target_path_relative"] / item["filename"]
    return comfyui_path / "models" / "diffusers" / item["local_dir"]


def item_present(item: Dict[str, Any], comfyui_path: Path) -> bool:
    target = item_target(item, comfyui_path)
    if item["type"] == "hf_model":
        return (target / "model_index.json").exists()
    return target.exists()


TAG_LABELS = {
    "core": "SDXL core", "preset_essentials": "SDXL preset essentials", "extended_optional": "SDXL extras",
    "checkpoint_realvis": "RealVisXL checkpoint", "qwen_core": "Qwen Image Edit core", "qwen_extras": "Qwen extras",
    "qwen_nunchaku": "Qwen Nunchaku", "pbr_marigold": "Marigold IID", "pbr_stabledelight": "StableDelight",
    "pbr_marigold_v2": "Marigold V2", "trellis2": "TRELLIS.2", "flux2_klein": "FLUX.2 Klein",
    "qwen_image_21": "Qwen Image 2.1", "skintokens": "SkinTokens",
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


ENV_SIZE_MB = {"trellis2": 5000, "skintokens": 9000}


def estimate_download_mb(items: List[Dict[str, Any]], tags: List[str], comfyui_path: Path) -> int:
    """MB still to download: models that are not present yet, plus environments the nodes build."""
    total = sum(item.get("size_mb", 0) for item in items
                if item["type"] in ("model", "hf_model") and not item_present(item, comfyui_path))
    for tag in tags:
        node_items = [i for i in items if i["type"] == "node" and tag in i["packages"]]
        if tag in ENV_SIZE_MB and not all(item_present(i, comfyui_path) for i in node_items):
            total += ENV_SIZE_MB[tag]
    return total


def trellis_python_ok(comfyui_path: Path) -> bool:
    """TRELLIS.2's CUDA extensions have no wheels for Python 3.13 yet; ask before installing there."""
    try:
        out = subprocess.run([find_comfyui_python(comfyui_path), "-c",
                              "import sys; print(sys.version_info[0], sys.version_info[1])"],
                             capture_output=True, text=True, check=True)
        is_313_plus = tuple(int(x) for x in out.stdout.split()) >= (3, 13)
    except Exception:
        is_313_plus = sys.version_info >= (3, 13)
    if not is_313_plus:
        return True
    print_separator(char='!')
    print("WARNING: TRELLIS.2 needs binary CUDA extensions (cumesh, flex_gemm, nvdiffrec, ...) that have no")
    print("pre-built wheels for Python 3.13 yet; building them needs MSVC and the CUDA Toolkit.")
    print("A ComfyUI with Python 3.12 (e.g. a portable build) is strongly recommended.")
    print_separator(char='!')
    return ask("Attempt the TRELLIS.2 installation on Python 3.13 anyway?", default_yes=False)


def show_notice(key: str) -> bool:
    print_separator(char='!')
    for line in LICENSE_NOTICES[key]:
        print(line)
    print_separator(char='!')
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
    items = [item for item in DEPENDENCIES.values() if any(t in item["packages"] for t in tags)]
    for item in items:
        if item["type"] == "node":
            continue
        target = item_target(item, comfyui_path)
        if not item_present(item, comfyui_path):
            print(f"  MISSING  {item['name']} ({target})")
            ok = False
        elif item["type"] == "model" and target.with_name(target.name + ".part").exists():
            print(f"  WARNING  {item['name']}: an unfinished download is next to it")
    required = list(dict.fromkeys(n for t in tags for n in PACKAGE_NODES.get(t, [])))
    if not required:
        return ok
    if server:
        info, failed = _fetch_object_info(server), []
        if info is None:
            print(f"  Could not reach ComfyUI at {server}.")
    else:
        info, failed = start_comfyui_for_verify(comfyui_path, timeout)
    ours = {item["repo_name"] for item in items if item["type"] == "node"}
    for folder in failed:
        if folder in ours:
            print(f"  FAILED   custom node '{folder}' did not import (see the ComfyUI log)")
    others = [folder for folder in failed if folder not in ours]
    if others:
        print(f"  Note: other custom nodes failed to import (not StableGen's): {', '.join(others)}")
    failed = [folder for folder in failed if folder in ours]
    if info is None:
        return False
    missing = [n for n in required if n not in info]
    for tag in tags:
        absent = [n for n in PACKAGE_NODES.get(tag, []) if n in missing]
        if absent:
            core = set(CORE_NODES.get(tag, []))
            hint = "update ComfyUI" if set(absent) <= core else "the custom node did not load"
            print(f"  MISSING  {', '.join(absent)} ({hint})")
    print(f"  {len(required) - len(missing)} of {len(required)} required node types are registered.")
    return ok and not missing and not failed


# --- Summary ---
STATUS_LABELS = {"installed": "installed", "updated": "updated", "present": "present", "warning": "WARNING",
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
    global ASSUME_YES, PYTHON_OVERRIDE
    args = parse_args(argv)
    if args.list:
        print_package_list()
        return 0
    ASSUME_YES = args.yes
    if args.python:
        if not Path(args.python).is_file():
            print(f"Error: --python '{args.python}' does not exist.")
            return 2
        PYTHON_OVERRIDE = str(Path(args.python).resolve())
    keys: Optional[List[str]] = None
    if args.packages:
        try:
            keys = parse_package_choice(args.packages)
        except ValueError as e:
            print(f"Error: {e}")
            return 2

    comfyui_base_path = get_comfyui_path_from_args(args)
    print(f"ComfyUI: {comfyui_base_path}")
    if not describe_python(comfyui_base_path) and not args.check:
        if not ask("Continue with this Python?", default_yes=False):
            return 1

    if args.check:
        print_header("Checking TRELLIS.2 environments")
        return 0 if repair_comfy_env_nodes(comfyui_base_path, check_only=True) else 1

    if args.verify and keys is None and not args.repair:
        tags, partial = installed_tags(comfyui_base_path)
        if partial:
            print("Partly installed, not checked: " + ", ".join(
                f"{TAG_LABELS[t]} ({have} of {total} components)" for t, (have, total) in partial.items()))
        if not tags:
            print("No StableGen package is completely installed here yet.")
            return 1
        print("Checking: " + ", ".join(TAG_LABELS[t] for t in tags))
        return 0 if verify_installation(comfyui_base_path, tags, args.server, args.verify_timeout) else 1

    if not args.dry_run:
        # Ensure all comfy-env patches are applied if comfy-env is already installed
        has_comfy_env = False
        try:
            result = subprocess.run([find_comfyui_python(comfyui_base_path), "-c", "import comfy_env"],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                has_comfy_env = True
                print("  Detected existing comfy-env installation. Ensuring all patches are applied...")
                _apply_all_comfy_env_patches(comfyui_base_path)
        except Exception as e:
            print(f"  WARNING: Could not check/patch existing comfy-env: {e}")

        # Already-installed TRELLIS.2 nodes drift out of step whenever ComfyUI or the node updates; repair them
        # on every installer run.
        if has_comfy_env or args.repair:
            print_header("Repairing installed TRELLIS.2 nodes")
            ok = repair_comfy_env_nodes(comfyui_base_path)
            if args.repair:
                print("Repair complete." if ok else "Repair finished with warnings (see above).")
                print("Restart ComfyUI so the TRELLIS.2 worker picks up the changes.")
                return 0 if ok else 1
    elif args.repair:
        print("--repair changes files; nothing to show in a dry run.")
        return 0

    if keys is None:
        keys = choose_packages_interactively(comfyui_base_path)
        if keys is None:
            print("Exiting installer.")
            return 0

    # License notices and the Python 3.13 check can drop packages
    accepted = []
    for key in keys:
        package = MENU_PACKAGES[key]
        if "trellis2" in package["tags"] and not trellis_python_ok(comfyui_base_path):
            print(f"Skipping package {key}.")
            continue
        if package.get("notice") and not show_notice(package["notice"]):
            print(f"Skipping package {key}.")
            continue
        accepted.append(key)
    if not accepted:
        print("Nothing to install.")
        return 0
    tags = tags_for(accepted)
    items = resolve_items(accepted)

    print_separator()
    for key in accepted:
        print(f"Package {key}: {MENU_PACKAGES[key]['name']}")
    print("Components:")
    for item in items:
        state = "present" if item_present(item, comfyui_base_path) else "to install"
        print(f"  - {item['name']} ({state})")
    need_mb = estimate_download_mb(items, tags, comfyui_base_path)
    free_mb = shutil.disk_usage(comfyui_base_path).free // 1024**2
    print(f"Estimated download: ~{need_mb / 1024:.1f} GB; free on this drive: {free_mb / 1024:.1f} GB")

    problems = False
    for tag, absent in missing_core_nodes(comfyui_base_path, tags).items():
        print(f"WARNING: this ComfyUI lacks {', '.join(absent)}; update ComfyUI before using these models.")
    if any(item["type"] == "node" for item in items) and shutil.which("git") is None:
        print("ERROR: Git is needed to install custom nodes and was not found on PATH (https://git-scm.com).")
        problems = True
    if need_mb > free_mb:
        print(f"ERROR: not enough free disk space ({need_mb / 1024:.1f} GB needed, {free_mb / 1024:.1f} GB free).")
        problems = problems or not ask("Continue anyway?", default_yes=False, yes_answers=False)
    if problems:
        return 1

    if args.dry_run:
        print("Dry run: nothing was changed.")
        if args.verify:
            return 0 if verify_installation(comfyui_base_path, tags, args.server, args.verify_timeout) else 1
        return 0
    if not ask("Proceed with checking/installing these components?"):
        print("Installation cancelled.")
        return 0

    rows: List[Tuple[str, str, str]] = []
    for item_details in items:
        print_separator(char='.')
        print(f"Processing: {item_details['name']}")
        if item_details["type"] == "node":
            rows.extend(install_node(item_details, comfyui_base_path, args.update_nodes))
        elif item_details["type"] == "model":
            rows.append((item_details["name"],) + download_file(item_details, comfyui_base_path, args.verify_hashes))
        elif item_details["type"] == "hf_model":
            status, message = download_hf_model(item_details, comfyui_base_path, args.verify_hashes)
            print(f"  {status}{': ' + message if message else ''}")
            rows.append((item_details["name"], status, message))

    ok = print_summary(rows)
    node_names = {item["name"] for item in items if item["type"] == "node"}
    if any(status in ("installed", "updated") and name in node_names for name, status, _ in rows):
        print("Restart ComfyUI so it loads new or updated custom nodes.")
    if args.verify:
        ok = verify_installation(comfyui_base_path, tags, args.server, args.verify_timeout) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    if not TQDM_AVAILABLE:
        print("NOTE: 'tqdm' is not installed, so downloads show a progress line instead of a bar.")
    sys.exit(main())
