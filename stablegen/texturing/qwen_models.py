"""Model-family detection and defaults for StableGen's Qwen workflows."""

from __future__ import annotations

import re


QWEN_IMAGE_21_TEXT_ENCODER = "qwen3vl_8b_w4a8.safetensors"
QWEN_IMAGE_21_VAE = "qwen_image_2.1_vae_bf16.safetensors"
QWEN_IMAGE_EDIT_ARCHITECTURE = "qwen_image_edit"
QWEN_IMAGE_21_ARCHITECTURE = "qwen_image_21"
QWEN_IMAGE_21_TURBO_ARCHITECTURE = "qwen_image_21_turbo"
FIBO_EDIT_ARCHITECTURE = "fibo_edit"
# Qwen-Image 2.1 Turbo is distilled to one fixed 8-step schedule
# (``sample_sigmas`` in its model_index.json, used by the official ComfyUI
# templates through ManualSigmas). Standard schedulers do not reproduce it.
QWEN_IMAGE_21_TURBO_STEPS = 8
QWEN_IMAGE_21_TURBO_SIGMAS = (
    "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
)
# Bria FIBO 2.0, as packaged by Comfy-Org/Fibo-2.0 for ComfyUI's core nodes.
FIBO_TEXT_ENCODER = "qwen3vl_4b_int8_convrot.safetensors"
FIBO_VAE = "flux2-vae.safetensors"
QWEN_IMAGE_21_RGBA_PREFIX = "This is an RGBA format image with transparency."
QWEN_IMAGE_21_RGBA_SUFFIX = (
    "The image has an alpha channel and a transparent background."
)

QWEN_PROMPT_TEMPLATE_ITEMS = (
    (
        "QWEN_EDIT_LEGACY",
        "Qwen Image Edit Legacy",
        "Use the original StableGen Qwen-Image-Edit transfer wording",
    ),
    (
        "QWEN_21_DEFAULT",
        "Qwen 2.1 Default",
        "Use the newer Qwen-Image 2.1 create-and-guide wording used by the sampler tests",
    ),
    (
        "CANVAS_EDIT",
        "Canvas Edit",
        "Treat the target view as the canvas and transfer appearance while preserving its geometry",
    ),
    (
        "REFERENCE_RECONSTRUCTION",
        "Reference Reconstruction",
        "Reconstruct the requested subject in the target view using explicit image roles",
    ),
    (
        "CONCISE_TRANSFER",
        "Concise Transfer",
        "Use a short appearance-transfer instruction with strong target-view preservation",
    ),
    (
        "RESULT_DESCRIPTION",
        "Result Description",
        "Describe the desired result while assigning structure and appearance references",
    ),
    (
        "CANVAS_AUTHORITY",
        "Canvas Authority",
        "State which image controls geometry and which image controls appearance",
    ),
    (
        "MINIMAL_EDIT",
        "Minimal Edit",
        "Use the strongest concise general-purpose variant from the Qwen 2.1 prompt tests",
    ),
    (
        "CUSTOM",
        "Custom",
        "Write separate guidance prompts for initial and sequential views",
    ),
)

QWEN_PROMPT_TEMPLATE_IDS = frozenset(item[0] for item in QWEN_PROMPT_TEMPLATE_ITEMS)


def is_qwen_image_21_model(model_name: str) -> bool:
    """Return whether a selected diffusion model belongs to Qwen-Image 2.1.

    Model files in ComfyUI can be nested in folders and community conversions
    use dots, dashes, or underscores in the version. Filename detection is
    retained for filtering model choices and migrating files saved before the
    dedicated architecture existed; generation dispatch uses the architecture.
    """
    return bool(re.search(r"qwen[-_. ]*image[-_. ]*2[-_. ]*1(?:\D|$)", _model_basename(model_name)))


def is_qwen_image_21_turbo_model(model_name: str) -> bool:
    """Return whether a Qwen-Image 2.1 model file is the few-step Turbo variant."""
    return is_qwen_image_21_model(model_name) and "turbo" in _model_basename(model_name)


def is_moss_image_model(model_name: str) -> bool:
    """Return whether a diffusion model file belongs to MOSS-Image."""
    return bool(re.search(r"moss[-_. ]*image", _model_basename(model_name)))


def is_fibo_model(model_name: str) -> bool:
    """Return whether a diffusion model file belongs to Bria FIBO."""
    return "fibo" in _model_basename(model_name)


def _model_basename(model_name: str) -> str:
    return (model_name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def is_qwen_architecture(architecture: str) -> bool:
    """Return whether *architecture* uses StableGen's shared Qwen UI path."""
    return architecture in {
        QWEN_IMAGE_EDIT_ARCHITECTURE,
        QWEN_IMAGE_21_ARCHITECTURE,
        QWEN_IMAGE_21_TURBO_ARCHITECTURE,
        FIBO_EDIT_ARCHITECTURE,
    }


def qwen_prepared_prompt(
    template_id: str,
    *,
    canvas_image: int,
    structure_image: int | None = None,
    appearance_image: int | None = None,
    has_magenta_regions: bool = False,
    gray_background: bool = False,
) -> str:
    """Build one of the prompt variants evaluated for Qwen image guidance.

    Image numbers use StableGen's semantic Qwen-Image-Edit ordering.  The
    Qwen-Image 2.1 graph adapter remaps them after it makes projected context
    image 1, so a single template remains correct for both architectures.
    The returned string deliberately keeps ``{main_prompt}`` for the caller.
    """
    if template_id not in QWEN_PROMPT_TEMPLATE_IDS:
        template_id = "QWEN_EDIT_LEGACY"
    if template_id in {"QWEN_EDIT_LEGACY", "QWEN_21_DEFAULT", "CUSTOM"}:
        raise ValueError(f"{template_id} is not a prepared prompt template")

    canvas = f"image {canvas_image}"
    structure = (
        f"image {structure_image}"
        if structure_image is not None and structure_image != canvas_image
        else None
    )
    appearance = (
        f"image {appearance_image}"
        if appearance_image is not None and appearance_image != canvas_image
        else None
    )

    preservation = (
        f"Preserve {canvas}'s camera viewpoint, subject orientation, pose, "
        "silhouette, proportions, framing, and background."
    )
    structure_role = (
        f" Use {structure} only as the geometry and composition guide."
        if structure
        else ""
    )
    appearance_role = (
        f" Use {appearance} only as the identity and appearance reference "
        "for the subject."
        if appearance
        else ""
    )
    completion = ""
    if has_magenta_regions:
        completion = (
            " Preserve its existing textured regions and replace every solid "
            "magenta region with texture that seamlessly continues the "
            "surrounding surface."
        )
    background = " Replace the background with solid gray." if gray_background else ""

    if template_id == "MINIMAL_EDIT":
        return (
            f"Edit {canvas} to satisfy this instruction: {{main_prompt}}. "
            f"{preservation}{completion}{structure_role}{appearance_role}{background}"
        )

    if template_id == "CANVAS_AUTHORITY":
        prompt = (
            f"Use {canvas} as the canvas. Apply this edit: {{main_prompt}}. "
            f"{canvas} is authoritative for the camera viewpoint, subject "
            "orientation, pose, silhouette, proportions, framing, and background."
        )
        if structure:
            prompt += f" {structure} is authoritative for geometry and composition."
        if appearance:
            prompt += (
                f" {appearance} is authoritative for the subject's identity, "
                "styling, colors, materials, clothing design, and surface detail."
            )
        return (
            f"{prompt}{completion}{background} Produce a coherent, fully textured "
            "version of the canvas."
        )

    if template_id == "CANVAS_EDIT":
        prompt = (
            f"Edit {canvas} as the canvas. Preserve its exact camera viewpoint, "
            "framing, subject orientation, pose, silhouette, proportions, and "
            "background. Apply this instruction: {main_prompt}."
        )
        if appearance:
            prompt += (
                f" Transfer the identity, styling, colors, materials, and surface "
                f"details from {appearance}, fitting them to the visible surfaces "
                f"of {canvas}."
            )
        return f"{prompt}{completion}{structure_role}{background}"

    if template_id == "REFERENCE_RECONSTRUCTION":
        prompt = (
            f"Reconstruct {{main_prompt}} in the exact camera view shown by {canvas}. "
            f"Use {canvas} for the camera, pose, proportions, silhouette, framing, "
            "and background."
        )
        if appearance:
            prompt += (
                f" Use {appearance} for identity, styling, colors, materials, and "
                "surface detail."
            )
        return (
            f"{prompt}{completion}{structure_role}{background} Produce a coherent "
            "render with the requested appearance correctly resolved in this view."
        )

    if template_id == "CONCISE_TRANSFER":
        if appearance:
            lead = (
                f"Apply the identity and appearance from {appearance} to {canvas} "
                "while satisfying this instruction: {main_prompt}."
            )
        else:
            lead = f"Apply this instruction to {canvas}: {{main_prompt}}."
        return f"{lead} {preservation}{completion}{structure_role}{background}"

    # RESULT_DESCRIPTION
    prompt = (
        f"Create a coherent render of {{main_prompt}} that follows the camera "
        f"viewpoint, pose, silhouette, proportions, framing, and background of {canvas}."
    )
    if appearance:
        prompt += (
            f" Depict the same identity, styling, colors, materials, and surface "
            f"details shown in {appearance}."
        )
    return f"{prompt}{completion}{structure_role}{background}"


def qwen_image_21_prompt(text: str) -> str:
    """Convert legacy ``Image 1`` references to Qwen 2.1 image tokens."""
    return re.sub(
        r"(?<!<)\bimage\s*([1-9]\d*)\b(?!>)",
        lambda match: f"<image{match.group(1)}>",
        text,
        flags=re.IGNORECASE,
    )


def qwen_image_21_rgba_prompt(text: str) -> str:
    """Wrap a Qwen 2.1 prompt for native transparent RGBA output.

    The wording follows the official Qwen-Image 2.1 ComfyUI template.  Keep
    this idempotent because custom prompts and archived workflows may already
    include the wrapper.
    """
    prompt = re.sub(
        r"\s*(?:keep|replace)\s+the\s+background\s+"
        r"(?:with\s+)?solid(?:\s+neutral)?\s+gray\.?",
        "",
        (text or "").strip(),
        flags=re.IGNORECASE,
    ).strip()
    lowered = prompt.casefold()
    has_prefix = "rgba format image with transparency" in lowered
    has_suffix = (
        "alpha channel" in lowered and "transparent background" in lowered
    )
    if not has_prefix:
        prompt = f"{QWEN_IMAGE_21_RGBA_PREFIX} {prompt}".strip()
    if not has_suffix:
        prompt = f"{prompt} {QWEN_IMAGE_21_RGBA_SUFFIX}".strip()
    return prompt


def remap_qwen_image_21_prompt(text: str, image_mapping: dict[int, int]) -> str:
    """Convert image references and renumber them without replacement clashes.

    StableGen's original Qwen Edit workflow uses image 1 as a structure guide
    and image 2 as projected context.  Qwen-Image 2.1 instead treats its first
    image as the edit target, so later context-guided views swap those slots.
    This helper keeps custom driving prompts aligned with the swapped images.
    """
    converted = qwen_image_21_prompt(text)
    return re.sub(
        r"<image([1-9]\d*)>",
        lambda match: (
            f"<image{image_mapping.get(int(match.group(1)), int(match.group(1)))}>"
        ),
        converted,
        flags=re.IGNORECASE,
    )


def fibo_prompt(text: str, image_mapping: dict[int, int] | None = None) -> str:
    """Convert image references to FIBO 2.0's explicit ``<image_N>`` indices.

    FIBO refers to its inputs by position only, so every reference is made
    explicit. ``image_mapping`` renumbers them after the graph adapter has
    moved the projected context into the first (edited) slot.
    """
    converted = remap_qwen_image_21_prompt(text, image_mapping or {})
    return re.sub(r"<image([1-9]\d*)>", r"<image_\1>", converted, flags=re.IGNORECASE)
