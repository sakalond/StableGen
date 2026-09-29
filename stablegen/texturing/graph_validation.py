"""Pure validation helpers for dynamically edited ComfyUI workflow graphs."""

from __future__ import annotations

from collections.abc import Mapping


def _is_node_reference(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def validate_workflow_graph(prompt: Mapping, semantic_node_ids=()) -> list[str]:
    """Return graph errors without changing the workflow.

    Checks every ComfyUI link target and, for Qwen text encoders, verifies
    that image slots are consistent and do not accidentally alias one another.
    """
    issues: list[str] = []
    node_ids = {str(node_id) for node_id in prompt}
    for node_id, node in prompt.items():
        for input_name, value in node.get("inputs", {}).items():
            if _is_node_reference(value) and str(value[0]) not in node_ids:
                issues.append(
                    f"node {node_id} input {input_name} references missing node {value[0]}"
                )

    semantic_refs = {}
    for node_id in semantic_node_ids:
        node = prompt.get(str(node_id))
        if node is None:
            issues.append(f"semantic text encoder node {node_id} is missing")
            continue
        inputs = node.get("inputs", {})
        slots = {
            key: tuple(inputs[key])
            for key in ("image1", "image2", "image3")
            if key in inputs and _is_node_reference(inputs[key])
        }
        reverse = {}
        for slot, reference in slots.items():
            reverse.setdefault(reference, []).append(slot)
        for reference, aliases in reverse.items():
            if len(aliases) > 1:
                issues.append(
                    f"semantic node {node_id} aliases {', '.join(aliases)} to {reference[0]}"
                )
        semantic_refs[str(node_id)] = slots

    if len(semantic_refs) > 1:
        reference_slots = next(iter(semantic_refs.values()))
        for node_id, slots in list(semantic_refs.items())[1:]:
            if slots != reference_slots:
                issues.append(
                    f"semantic image slots differ between text encoders and node {node_id}"
                )
    return issues


def qwen_style_semantic_slot(context_mode: str, is_initial_image: bool) -> str:
    """Return the Qwen text-encoder slot occupied by the style reference."""
    if context_mode == "ADDITIONAL" and not is_initial_image:
        return "image3"
    return "image2"


def qwen_reference_latent_slots(
    context_mode: str,
    is_initial_image: bool,
    global_style_vlm_only: bool,
    context_vlm_only: bool = False,
) -> tuple[str, ...]:
    """Return Qwen image slots that should receive spatial reference latents.

    Omitted slots remain connected to ``TextEncodeQwenImageEditPlus``.
    These experiments remove only manual full-resolution ReferenceLatent
    conditioning. Image 2 is context only on later views in a context mode;
    initial/external style references must keep their existing behavior.
    """
    slots = ["image1", "image2", "image3"]
    if (
        context_vlm_only
        and context_mode in {"REPLACE_STYLE", "ADDITIONAL"}
        and not is_initial_image
    ):
        slots.remove("image2")
    if (
        global_style_vlm_only
        and context_mode == "ADDITIONAL"
        and not is_initial_image
    ):
        slots.remove("image3")
    return tuple(slots)


def attach_qwen_latent_lock(
    prompt: dict,
    *,
    sampler_node_id: str,
    vae_node_id: str,
    context_image_name: str,
    repair_mask_name: str,
    start_node_id: int = 650,
    neutral_fill: bool = False,
) -> dict[str, str]:
    """Attach sampler-level Qwen context inpainting to ``prompt``.

    The context render is VAE-encoded as the sampler's starting latent.  The
    exported sequential visibility image uses StableGen's repair convention:
    white pixels may denoise and black pixels remain locked.  ComfyUI's generic
    noise-mask sampler restores the protected source latent after every step.

    Dedicated loader/encode nodes are intentional.  Qwen's semantic image
    slots are dynamically reordered between Replace Style and Additional
    modes; keeping the latent-lock path separate prevents semantic slot changes
    from silently changing which image is protected.

    Optional neutral fill removes the repair sentinel before VAE encoding.
    Encoding magenta directly can contaminate protected boundary latents even
    when the sampler mask is correct. The inpaint encoder uses neutral gray in
    repair pixels; grow_mask_by=0 and the explicit mask node preserve the same
    protected region as the original experiment.
    """
    sampler_node_id = str(sampler_node_id)
    vae_node_id = str(vae_node_id)
    if sampler_node_id not in prompt:
        raise ValueError(f"Qwen latent lock sampler node {sampler_node_id} is missing")
    if vae_node_id not in prompt:
        raise ValueError(f"Qwen latent lock VAE node {vae_node_id} is missing")
    if not context_image_name:
        raise ValueError("Qwen latent lock requires a context image")
    if not repair_mask_name:
        raise ValueError("Qwen latent lock requires a repair mask")

    next_id = int(start_node_id)

    def allocate() -> str:
        nonlocal next_id
        while str(next_id) in prompt:
            next_id += 1
        node_id = str(next_id)
        next_id += 1
        return node_id

    context_loader = allocate()
    context_encode = allocate()
    mask_loader = allocate()
    image_to_mask = allocate()
    set_noise_mask = allocate()

    prompt[context_loader] = {
        "inputs": {"image": context_image_name},
        "class_type": "LoadImage",
        "_meta": {"title": "Qwen Latent Lock - Context"},
    }
    prompt[context_encode] = {
        "inputs": {
            "pixels": [context_loader, 0],
            "vae": [vae_node_id, 0],
        },
        "class_type": "VAEEncode",
        "_meta": {"title": "Qwen Latent Lock - Encode Context"},
    }
    prompt[mask_loader] = {
        "inputs": {"image": repair_mask_name},
        "class_type": "LoadImage",
        "_meta": {"title": "Qwen Latent Lock - Repair Mask"},
    }
    prompt[image_to_mask] = {
        "inputs": {
            "image": [mask_loader, 0],
            "channel": "red",
        },
        "class_type": "ImageToMask",
        "_meta": {"title": "Qwen Latent Lock - Image to Mask"},
    }
    if neutral_fill:
        prompt[context_encode]["class_type"] = "VAEEncodeForInpaint"
        prompt[context_encode]["inputs"].update({
            "mask": [image_to_mask, 0],
            "grow_mask_by": 0,
        })
        prompt[context_encode]["_meta"]["title"] = "Qwen Latent Lock - Neutral Repair Fill"
    prompt[set_noise_mask] = {
        "inputs": {
            "samples": [context_encode, 0],
            "mask": [image_to_mask, 0],
        },
        "class_type": "SetLatentNoiseMask",
        "_meta": {"title": "Qwen Latent Lock - Set Noise Mask"},
    }
    prompt[sampler_node_id].setdefault("inputs", {})["latent_image"] = [
        set_noise_mask,
        0,
    ]
    return {
        "context_loader": context_loader,
        "context_encode": context_encode,
        "mask_loader": mask_loader,
        "image_to_mask": image_to_mask,
        "set_noise_mask": set_noise_mask,
    }


def sdxl_final_mask_reference(blur_mask: bool, nodes: Mapping) -> list:
    """Return the one processed mask consumed by all SDXL inpaint nodes."""
    node_id = nodes['image_to_mask'] if blur_mask else nodes['grow_mask']
    return [node_id, 0]
