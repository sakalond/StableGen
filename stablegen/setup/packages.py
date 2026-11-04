"""The models the setup offers to install: rows grouped by what they are for, each one or more installer package
tags (installer.py installs by tag), with the licenses their files come under. Also the tags each architecture
needs, for the missing-model hint. No bpy here."""
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from . import installer

# Packages whose nodes only build on NVIDIA's CUDA: not offered on macOS
CUDA_ONLY = {"qwen_nunchaku", "skintokens", "lato2"}

ROWS: Tuple[Tuple[str, Tuple[str, ...], str, str], ...] = (
    # group, tags, model, what it does
    ("Texturing", ("core", "preset_essentials", "checkpoint_realvis"), "SDXL", "fast, fits 8 GB GPUs"),
    ("Texturing", ("extended_optional",), "SDXL extras", "more LoRAs and ControlNets"),
    ("Texturing", ("qwen_image_21",), "Qwen Image 2.1", "best quality"),
    ("Texturing", ("qwen_core",), "Qwen Image Edit", "2509 GGUF Q3_K_M"),
    # the presets use Qwen Image Edit's own 4-step LoRA; these are the 8-step one, to pick by hand, and base Qwen Image's
    ("Texturing", ("qwen_extras",), "Qwen Image Edit Lightning", "optional 8-step LoRAs"),
    ("Texturing", ("qwen_nunchaku",), "Qwen Image Edit Nunchaku", "4-bit, for smaller GPUs"),
    ("Experimental", ("qwen_image_21_turbo",), "Qwen Image 2.1 Turbo", "few-step sampling"),
    ("Experimental", ("moss_image",), "MOSS Image", "with depth control"),
    ("Experimental", ("fibo_edit",), "FIBO 2.0 Edit", "Bria's edit model"),
    ("Deprecated", ("flux2_klein",), "FLUX.2 Klein", "no longer supported"),
    ("3D, maps and rigging", ("trellis2",), "TRELLIS.2", "image to 3D"),
    ("3D, maps and rigging", ("pbr_marigold", "pbr_stabledelight"), "Marigold + StableDelight", "PBR maps"),
    ("3D, maps and rigging", ("pbr_marigold_v2",), "Marigold V2", "albedo and normal maps"),
    ("3D, maps and rigging", ("skintokens",), "SkinTokens", "auto-rigging"),
    ("3D, maps and rigging", ("lato2",), "LATO.2", "low-poly remesh"),
)

# What each architecture needs on the server (a checkpoint of the user's own counts for SDXL)
ARCH_TAGS: Dict[str, Tuple[str, ...]] = {
    'sdxl': ("core", "preset_essentials"),
    'qwen_image_edit': ("qwen_core",),
    'qwen_image_21': ("qwen_image_21",),
    'qwen_image_21_turbo': ("qwen_image_21_turbo",),
    'moss_image': ("moss_image",),
    'fibo_edit': ("fibo_edit",),
    'flux2_klein': ("flux2_klein",),
    'trellis2': ("trellis2",),
}

# The installer's license notes, short enough for a row; None leaves one out
LICENSE_NAMES = {
    "Qwen Research License (non-commercial)": "Qwen Research, non-commercial",
    "CC BY-NC 4.0 (non-commercial)": "CC BY-NC 4.0, non-commercial",
    "Apache-2.0": "Apache 2.0",
    "DINOv3 License": "DINOv3",
    "GPL-3.0 (SkinTokens code and weights: MIT)": "MIT",
    "GPL-3.0 (LATO.2 code and weights: MIT; DINOv2: Apache 2.0)": "MIT + Apache 2.0",
    "Unknown (User to verify)": None,
}
LICENSE_SHARE = 0.05    # a license under 5% of a row's download is left out (the notices cover the details)


def visible_rows(deprecated: bool = False, experimental: bool = True,
                 platform: str = sys.platform) -> Iterator[Tuple[str, Tuple[str, ...], str, str]]:
    for group, tags, title, purpose in ROWS:
        if (group == "Deprecated" and not deprecated) or (group == "Experimental" and not experimental):
            continue
        if platform == "darwin" and CUDA_ONLY & set(tags):
            continue
        yield group, tags, title, purpose


def license_text(tags: Sequence[str]) -> str:
    """'MIT + DINOv3': the licenses of a row's model files, largest share first (its nodes' when it has none)."""
    items = [i for i in installer.DEPENDENCIES.values() if set(i["packages"]) & set(tags)]
    models = [i for i in items if i["type"] in ("model", "hf_model")] or items
    shares: Dict[str, float] = {}
    for item in models:
        name = LICENSE_NAMES.get(item["license"], item["license"].split(" (")[0])
        if name:
            shares[name] = shares.get(name, 0) + max(1, item.get("size_mb", 0))
    total = sum(shares.values()) or 1
    names = [n for n, mb in sorted(shares.items(), key=lambda kv: -kv[1]) if mb / total >= LICENSE_SHARE]
    return " + ".join(names[:2])


def row_state(tags: Sequence[str], report: Optional[dict]) -> Dict[str, object]:
    """'installed', 'partial', 'missing' or 'unknown' (no report yet), with the sizes to show and the unfinished
    downloads (.part files) a partial one left."""
    entries = [(report or {}).get("tags", {}).get(t) for t in tags]
    if not report or any(e is None for e in entries):
        return {"state": "unknown", "download_mb": 0, "size_mb": 0, "missing_core_nodes": [], "unfinished": []}
    states = {e["state"] for e in entries}
    state = "installed" if states == {"installed"} else "missing" if states == {"missing"} else "partial"
    unfinished: List[str] = list(dict.fromkeys(path for e in entries for path in e.get("unfinished", [])))
    models_present = sum(e.get("models_present", 0) for e in entries)
    only_nodes = all(e["state"] == "missing" or (e.get("models_total", 0) and not e.get("models_present", 0))
                     for e in entries)
    if state == "partial" and only_nodes and not unfinished:
        state = "missing"       # only its nodes are there (already in the ComfyUI, or left after removing it)
    return {"state": state,
            "download_mb": sum(e["download_mb"] for e in entries),
            "size_mb": sum(e["size_gb"] for e in entries) * 1024,
            "missing_core_nodes": sorted({n for e in entries for n in e.get("missing_core_nodes", [])}),
            "unfinished": unfinished, "models_present": models_present}


def missing_for_architecture(architecture: str, report: Optional[dict]) -> Optional[Tuple[Tuple[str, ...], str]]:
    """(tags, title) to install when this architecture's packages are not complete in report, else None."""
    tags = ARCH_TAGS.get(architecture)
    if not tags or not report or row_state(tags, report)["state"] in ("installed", "unknown"):
        return None
    title = next((t for _, row_tags, t, _ in ROWS if set(tags) <= set(row_tags)), architecture)
    return tags, title


def removable_state(tags: Sequence[str], state: Dict[str, object], nodes: str = "") -> bool:
    """A row with something to remove on disk: models or unfinished downloads, or nodes the nodes policy removes
    (installer.removed_with_package)."""
    if state.get("models_present") or state["unfinished"]:
        return True
    return state["state"] in ("installed", "partial") and any(
        i["type"] in installer.NODE_TYPES and installer.removed_with_package(i, nodes)
        for i in installer.DEPENDENCIES.values() if set(i["packages"]) & set(tags))


def queue_text(place: int) -> str:
    """'next', '2nd in line', ...: where a queued install waits (0 is next)."""
    if place == 0:
        return "next"
    n = place + 1
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')} in line"


def size_text(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{max(1, round(mb))} MB"


def meta_text(purpose: str, tags: Sequence[str], size_mb: float) -> str:
    """'Image to 3D  ·  MIT + DINOv3  ·  13.1 GB': what a row does, its license and its size."""
    parts = [purpose[:1].upper() + purpose[1:], license_text(tags)] + ([size_text(size_mb)] if size_mb else [])
    return "  ·  ".join(p for p in parts if p)
