"""helpers.py - backward-compatibility shim.

Canonical location: stablegen/util/workflow_templates.py
"""
from .workflow_templates import (  # noqa: F401
    prompt_text,
    prompt_text_img2img,
    prompt_text_flux,
    prompt_text_img2img_flux,
    ipadapter_flux,
    depth_lora_flux,
    gguf_unet_loader,
    prompt_text_qwen_image_edit,
    prompt_text_flux2_klein,
)
