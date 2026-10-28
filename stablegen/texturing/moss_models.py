"""MOSS-Image model defaults and the SDXL-graph adapter.

MOSS-Image reuses StableGen's SDXL workflows: txt2img, img2img refinement,
classical inpainting and Differential Diffusion. The adapter below swaps only
the model layer of a finished SDXL graph and removes the SDXL-only extensions
(LoRA, CLIP skip, IPAdapter and ControlNet) that have no MOSS counterpart.

Two ComfyUI setups are supported. ComfyUI 0.42 and later load MOSS-Image
with the core loaders and already apply its flow shift; 0.42.3 adds the
``MOSSImageControl`` node for the color/depth control adapter. Older servers
use the OpenMOSS ``moss-image`` custom node, whose model needs an explicit
``ModelSamplingSD3`` shift. The core model must not receive that node: it
would also change the timestep multiplier.
"""

from __future__ import annotations

MOSS_IMAGE_ARCHITECTURE = "moss_image"
MOSS_IMAGE_TEXT_ENCODER = "qwen3.5_9b_int8_convrot.safetensors"
MOSS_IMAGE_VAE = "flux2-vae_small.safetensors"
# darask0/MOSS-Image-color-depth-control, as repacked by Comfy-Org.
MOSS_IMAGE_CONTROL_LORA = "moss_image_color_depth_control.safetensors"
MOSS_IMAGE_SHIFT = 5.0
# The model card skips block 22 on the unconditional branch.
MOSS_IMAGE_UNCOND_SKIP_LAYERS = "22"
# Nodes that tell the two server setups apart.
MOSS_PLUGIN_LOADER = "MOSSImageUNETLoader"
MOSS_CONTROL_NODE = "MOSSImageControl"

# Nodes removed from SDXL graphs, mapped as {output index: passthrough input}.
_BYPASSED_NODES = {
    "CLIPSetLastLayer": {0: "clip"},
    "LoraLoader": {0: "model", 1: "clip"},
    "LoraLoaderModelOnly": {0: "model"},
    "IPAdapterUnifiedLoader": {0: "model"},
    "IPAdapter": {0: "model"},
    "ControlNetApplyAdvanced": {0: "positive", 1: "negative"},
}
_OUTPUT_NODES = {"SaveImageWebsocket", "SaveImage", "PreviewImage"}
_SAMPLERS = {"KSampler", "KSamplerAdvanced"}


def _is_reference(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _next_id(prompt, start):
    node_id = start
    while str(node_id) in prompt:
        node_id += 1
    return str(node_id)


def _rewire(prompt, mapping, skip=()):
    """Replace every link ``(node, output)`` found in *mapping*."""
    for node_id, node in prompt.items():
        if node_id in skip:
            continue
        inputs = node.get("inputs", {})
        for name, value in list(inputs.items()):
            if _is_reference(value):
                target = mapping.get((str(value[0]), value[1]))
                if target is not None:
                    inputs[name] = list(target)


def _drop_unreachable(prompt):
    """Remove nodes that no output node depends on (orphaned loaders)."""
    reachable = set()
    pending = [node_id for node_id, node in prompt.items()
               if node.get("class_type") in _OUTPUT_NODES]
    while pending:
        node_id = str(pending.pop())
        if node_id in reachable or node_id not in prompt:
            continue
        reachable.add(node_id)
        for value in prompt[node_id].get("inputs", {}).values():
            if _is_reference(value):
                pending.append(str(value[0]))
    if reachable:
        for node_id in [node_id for node_id in prompt if node_id not in reachable]:
            del prompt[node_id]


def _add(prompt, start, class_type, title, inputs):
    node_id = _next_id(prompt, start)
    prompt[node_id] = {"inputs": inputs, "class_type": class_type, "_meta": {"title": title}}
    return node_id


def convert_sdxl_graph_to_moss(
    prompt,
    model_name: str,
    *,
    native: bool = True,
    depth_image: str | None = None,
    depth_strength: float = 1.0,
    depth_end: float = 0.5,
    text_encoder: str = MOSS_IMAGE_TEXT_ENCODER,
    vae: str = MOSS_IMAGE_VAE,
    control_lora: str = MOSS_IMAGE_CONTROL_LORA,
):
    """Rewrite a StableGen SDXL API graph in place to run MOSS-Image.

    Every ``CheckpointLoaderSimple`` becomes a MOSS diffusion-model loader
    with separate Qwen3.5 text-encoder and FLUX.2 VAE loaders, for the core
    nodes (``native``) or the older custom node. SDXL-only extensions are
    bypassed so their conditioning and model links reconnect to their inputs.
    ``InpaintModelConditioning`` becomes a VAE encode with a latent noise
    mask: its concat latent is the channel MOSS reads its control maps from.
    Empty latents switch to the FLUX.2 format, and samplers get the model
    card's unconditional layer skip. With ``depth_image`` (an uploaded image
    name, core nodes only) the color/depth adapter conditions every sampler.
    Returns the removed SDXL-only class types for reporting.
    """
    removed = []
    for node_id, node in list(prompt.items()):
        if node.get("class_type") != "CheckpointLoaderSimple":
            continue
        if native:
            clip_id = _add(prompt, 800, "CLIPLoader", "Load MOSS-Image Qwen3.5 Text Encoder",
                           {"clip_name": text_encoder, "type": "moss_image", "device": "default"})
            loader = "UNETLoader"
        else:
            clip_id = _add(prompt, 800, "MOSSImageCLIPLoader", "Load MOSS-Image Qwen3.5 Text Encoder",
                           {"clip_name": text_encoder, "device": "default"})
            loader = MOSS_PLUGIN_LOADER
        vae_id = _add(prompt, 800, "VAELoader", "Load MOSS-Image VAE", {"vae_name": vae})
        prompt[node_id] = {
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
            "class_type": loader,
            "_meta": {"title": "Load MOSS-Image Diffusion Model"},
        }
        _rewire(prompt, {(node_id, 1): (clip_id, 0), (node_id, 2): (vae_id, 0)})

    # Bypass repeatedly: LoRA and ControlNet chains link to one another.
    changed = True
    while changed:
        changed = False
        for node_id, node in list(prompt.items()):
            passthrough = _BYPASSED_NODES.get(node.get("class_type"))
            if passthrough is None:
                continue
            inputs = node.get("inputs", {})
            mapping = {
                (node_id, output): tuple(inputs[name])
                for output, name in passthrough.items()
                if _is_reference(inputs.get(name))
            }
            del prompt[node_id]
            _rewire(prompt, mapping)
            removed.append(node["class_type"])
            changed = True

    for node_id, node in list(prompt.items()):
        if node.get("class_type") != "InpaintModelConditioning":
            continue
        inputs = node["inputs"]
        latent_id = _add(prompt, 840, "VAEEncode", "Encode Inpaint Image",
                         {"pixels": inputs["pixels"], "vae": inputs["vae"]})
        if inputs.get("noise_mask", True):
            latent_id = _add(prompt, 840, "SetLatentNoiseMask", "Inpaint Noise Mask",
                             {"samples": [latent_id, 0], "mask": inputs["mask"]})
        del prompt[node_id]
        _rewire(prompt, {
            (node_id, 0): tuple(inputs["positive"]),
            (node_id, 1): tuple(inputs["negative"]),
            (node_id, 2): (latent_id, 0),
        })

    for node in prompt.values():
        if node.get("class_type") == "EmptyLatentImage":
            node["class_type"] = "EmptyFlux2LatentImage"
            node.setdefault("_meta", {})["title"] = "Empty FLUX.2 Latent"

    if depth_image and native:
        for node_id, node in list(prompt.items()):
            if node.get("class_type") != "UNETLoader":
                continue
            lora_id = _add(prompt, 860, "LoraLoaderModelOnly", "MOSS-Image Color/Depth Control Adapter",
                           {"lora_name": control_lora, "strength_model": 1.0, "model": [node_id, 0]})
            _rewire(prompt, {(node_id, 0): (lora_id, 0)}, skip={lora_id})

    for node_id, node in list(prompt.items()):
        if node.get("class_type") not in _SAMPLERS:
            continue
        inputs = node["inputs"]
        model = inputs.get("model")
        if not native and _is_reference(model):
            model = [_add(prompt, 820, "ModelSamplingSD3", "MOSS-Image Flow Shift",
                          {"shift": MOSS_IMAGE_SHIFT, "model": model}), 0]
        if _is_reference(model):
            inputs["model"] = [_add(prompt, 820, "SkipLayerGuidanceDiTSimple", "MOSS-Image Uncond Layer Skip", {
                "double_layers": "",
                "single_layers": MOSS_IMAGE_UNCOND_SKIP_LAYERS,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "model": model,
            }), 0]
        if depth_image and native:
            decoder = next((n for n in prompt.values() if n.get("class_type") == "VAEDecode"), None)
            depth_id = _add(prompt, 880, "LoadImage", "Load Depth Map", {"image": depth_image})
            control_id = _add(prompt, 880, MOSS_CONTROL_NODE, "MOSS-Image Depth Control", {
                "positive": inputs["positive"],
                "negative": inputs["negative"],
                "vae": decoder["inputs"]["vae"],
                "depth_map": [depth_id, 0],
                "strength": depth_strength,
                "start_percent": 0.0,
                "end_percent": depth_end,
            })
            inputs["positive"] = [control_id, 0]
            inputs["negative"] = [control_id, 1]

    _drop_unreachable(prompt)
    return removed
