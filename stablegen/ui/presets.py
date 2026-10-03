"""Preset definitions, callbacks, and preset operators."""

import math
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
from ..utils import sg_modal_active
from ..feature_flags import TEXTURING_DEFAULTS, EXPERIMENTAL_FEATURE_DEFAULTS, experiments_enabled
from ..architecture_options import (
    DEPRECATION_NOTICE, deprecated_architectures_enabled,
    preset_uses_deprecated_architecture,
)

PRESETS = {
    "DEFAULT": {"description": "Default settings for general purpose generation", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)),  "discard_factor": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "hm-mvgd-hm", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.7, "sequential_factor_smooth": 0.15, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "pbr_decomposition": False, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "MODEL IS IMPORTANT": {"description": "Same as default, but is more guided by the model", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.7, "sequential_factor_smooth": 0.15, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "pbr_decomposition": False, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.75, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "CHARACTERS": {"description": "Optimized settings for character generation", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.7, "sequential_factor_smooth": 0.1, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "pbr_decomposition": False, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "CHARACTERS (ALTERNATIVE MASKING)": {"description": "Optimized for character generation. Uses alternative masking parameters to be more consistent between images, but may produce more artifacts. Try if \"Characters\" fails.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.5, "sequential_factor_smooth": 0.3499999940395355, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 10, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "pbr_decomposition": False, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "QUICK DRAFT": {"description": "Optimized for speed", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 4, "cfg": 1.0, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "grid", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.7, "sequential_factor_smooth": 0.1, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 2, "blur_mask_sigma": 1.0, "grow_mask_by": 2, "canny_threshold_low": 0, "canny_threshold_high": 80, "pbr_decomposition": False, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'Hyper-SDXL-4steps-lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "UV INPAINTING": {"description": "Recommended UV Inpainting setup. It is recommended to bake texutures manually before running the generation to fine-tune unwrapping and avoid lag when generating.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 10, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "uv_inpaint", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "3,0,1,2", "sequential_factor": 0.6000000238418579, "sequential_factor_smooth": 0.11000001430511475, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 3, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "pbr_decomposition": False, "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]}, # No ControlNet for UV Inpainting by default
    "ARCHITECTURE": {"description": "Prioritizes only the most straight-on camera for each point. This means details generated on flat surfaces will not get blurred by getting generated differently from two or more viewpoints. Does not use visibility masking. Each picture will get generated as new, consistency depends on IPAdapter + geometry.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "weight_exponent": 10.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "separate", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": False, "sequential_custom_camera_order": "", "sequential_factor": 0.75, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "ipadapter_weight_type": "style", "ipadapter_strength": 0.800000011920929, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": False, "blur_mask": True, "blur_mask_radius": 3, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "pbr_decomposition": False, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "QWEN PRECISE": {"description": "Precise detail when camera overlap is good. Relies on context renders plus the prompt, so sparse coverage can still introduce artifacts.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit","qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "pbr_decomposition": False, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "QWEN SAFE": {"description": "Safer fallback when coverage is limited. Uses the initial generated view as the reference for every later camera instead of context renders, improving global appearance consistency at the cost of extra conditioning time.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 6.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "pbr_decomposition": False, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "QWEN ALT": {"description": "Balanced option that mixes additional context renders with sequential references to smooth out coverage while keeping detail reasonable.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "recent", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "ADDITIONAL", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "early_priority": True, "early_priority_strength": 0.5, "pbr_decomposition": False, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},     
    "QWEN VORONOI": {"description": "Voronoi projection mode with exponent 1000 for hard camera segmentation during generation, then resets to 15 for softer blending. Each surface point is dominated by its closest camera. Based on Qwen Precise.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 1000.0, "weight_exponent_generation_only": True, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": True, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "pbr_decomposition": False, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "QWEN PRECISE (NUNCHAKU)": {"description": "Precise detail using Nunchaku. Meant to be used with the Nunchaku model which has the 4-step Lightning LoRA included. Requires Nunchaku nodes.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "pbr_decomposition": False, "lora_units": []},
    "QWEN SAFE (NUNCHAKU)": {"description": "Safer fallback using Nunchaku. Meant to be used with the Nunchaku model which has the 4-step Lightning LoRA included. Requires Nunchaku nodes.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 6.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "pbr_decomposition": False, "lora_units": []},
    "QWEN ALT (NUNCHAKU)": {"description": "Balanced option using Nunchaku. Meant to be used with the Nunchaku model which has the 4-step Lightning LoRA included. Requires Nunchaku nodes.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "recent", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "ADDITIONAL", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "early_priority": True, "early_priority_strength": 0.5, "pbr_decomposition": False, "lora_units": []},
    "QWEN VORONOI (NUNCHAKU)": {"description": "Voronoi projection mode using Nunchaku. Exponent 1000 for hard camera segmentation during generation, then resets to 15 for softer blending. Requires Nunchaku nodes.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 1000.0, "weight_exponent_generation_only": True, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": True, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "pbr_decomposition": False, "lora_units": []},
    "KLEIN PRECISE": {"description": "FLUX.2 Klein with precise detail when camera overlap is good. Uses depth reference images with CFGGuider (cfg=1). Best for well-covered geometry.", "control_after_generate": "fixed", "model_architecture": "flux2_klein", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour.", "qwen_custom_prompt_seq_none": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour, adopting the visual style from image 2.", "qwen_custom_prompt_seq_replace": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray.", "qwen_custom_prompt_seq_additional": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray. Image 3 represents the overall style.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "pbr_decomposition": False, "lora_units": []},
    "KLEIN SAFE": {"description": "FLUX.2 Klein safer fallback when coverage is limited. Uses the previous view instead of context renders with depth references and CFGGuider (cfg=1).", "control_after_generate": "fixed", "model_architecture": "flux2_klein", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 6.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour.", "qwen_custom_prompt_seq_none": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour, adopting the visual style from image 2.", "qwen_custom_prompt_seq_replace": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray.", "qwen_custom_prompt_seq_additional": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray. Image 3 represents the overall style.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "pbr_decomposition": False, "lora_units": []},
    "KLEIN ALT": {"description": "FLUX.2 Klein balanced option that mixes additional context renders with sequential references. Uses depth references and CFGGuider (cfg=1).", "control_after_generate": "fixed", "model_architecture": "flux2_klein", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "recent", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "ADDITIONAL", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour.", "qwen_custom_prompt_seq_none": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour, adopting the visual style from image 2.", "qwen_custom_prompt_seq_replace": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray.", "qwen_custom_prompt_seq_additional": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray. Image 3 represents the overall style.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "early_priority": True, "early_priority_strength": 0.5, "pbr_decomposition": False, "lora_units": []},
    "KLEIN VORONOI": {"description": "FLUX.2 Klein voronoi projection mode. Exponent 1000 for hard camera segmentation during generation, then resets to 15 for softer blending. Uses depth references and CFGGuider (cfg=1).", "control_after_generate": "fixed", "model_architecture": "flux2_klein", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 1000.0, "weight_exponent_generation_only": True, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": True, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour.", "qwen_custom_prompt_seq_none": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour, adopting the visual style from image 2.", "qwen_custom_prompt_seq_replace": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray.", "qwen_custom_prompt_seq_additional": "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray. Image 3 represents the overall style.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "pbr_decomposition": False, "lora_units": []},
    "LOCAL REFINE": {"description": "Uses the SDXL local edit mode to improve detail / refine specific areas. Use new set of cameras or a single camera pointed at the area you want to refine.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "local_edit", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.10000000149011612, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "original_render", "sequential_desaturate_factor": 0.10000000149011612, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": True, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 5.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05, "visibility_vignette": True, "visibility_vignette_width": 0.3, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "pbr_decomposition": False, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "LOCAL EDIT (QWEN)": {"description": "Uses Qwen to make targeted edits to specific areas. Point cameras at what you want to change — you can alter colors, style, rewrite text, add details, and more. Untouched areas are preserved.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "local_edit", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 1.0, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": False, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": False, "refine_angle_ramp_active": False, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05, "visibility_vignette": True, "visibility_vignette_width": 0.1, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6, "refine_edge_feather_projection": True, "refine_edge_feather_width": 15, "refine_edge_feather_softness": 1.0, "pbr_decomposition": False, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "REFINE (QWEN)": {"description": "Uses Qwen to restyle or globally modify the entire existing texture. Applies changes uniformly across all camera views — ideal for changing the overall color scheme, art style, or surface appearance.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "refine", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 1.0, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": False, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": False, "pbr_decomposition": False, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "DEFAULT (MESH + TEXTURE)": {"description": "Uses TRELLIS.2 and SDXL to generate a textured mesh. Optimized for general object generation. May not work ideally for the cases which have their specialized presets.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 75.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 85.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.10000000149011612, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "trellis2_input", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 3, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 5.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "qwen_generation_method": "generate", "qwen_refine_use_prev_ref": False, "qwen_refine_use_depth": False, "qwen_timestep_zero_ref": False, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05000000074505806, "visibility_vignette": True, "visibility_vignette_width": 0.15000000596046448, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6000000238418579, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "trellis2_texture_mode": "sdxl", "trellis2_initial_image_arch": "sdxl", "trellis2_camera_count": 10, "trellis2_placement_mode": "normal_weighted", "trellis2_auto_prompts": True, "trellis2_exclude_bottom": False, "trellis2_exclude_bottom_angle": 1.5533000230789185, "trellis2_auto_aspect": "per_camera", "trellis2_occlusion_mode": "none", "trellis2_consider_existing": False, "trellis2_delete_cameras": False, "trellis2_coverage_target": 0.949999988079071, "trellis2_max_auto_cameras": 12, "trellis2_fan_angle": 90.0, "trellis2_resolution": "1024_cascade", "trellis2_precision": "bf16", "trellis2_attn_backend": "auto", "trellis2_ss_guidance": 7.5, "trellis2_ss_steps": 12, "trellis2_shape_guidance": 7.5, "trellis2_shape_steps": 12, "trellis2_tex_guidance": 3.0, "trellis2_tex_steps": 12, "trellis2_max_tokens": 49152, "trellis2_texture_size": 4096, "trellis2_decimation": 1000000, "trellis2_remesh": "qdc", "trellis2_post_processing_enabled": True, "trellis2_bg_removal": "auto", "trellis2_background_color": "black", "trellis2_import_scale": 2.0, "trellis2_clamp_elevation": False, "trellis2_max_elevation": 1.2216999530792236, "trellis2_min_elevation": -1.0471975803375244, "trellis2_auto_lighting": True, "use_ipadapter": False, "sequential_ipadapter_regenerate_wo_controlnet": False, "allow_modify_existing_textures": False, "ask_object_prompts": True, "weight_exponent_mask": False, "mask_blocky": False, "architecture_mode": "trellis2", "use_camera_prompts": True, "sg_use_custom_camera_order": False, "pbr_decomposition": False, "generation_mode": "standard", "texture_objects": "all", "use_flux_lora": True, "qwen_use_trellis2_style": False, "qwen_trellis2_style_initial_only": False, "trellis2_skip_texture": True, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "TRELLIS.2 (MESH ONLY)": {"description": "Uses TRELLIS.2 to generate a mesh without texturing. Useful when you only need geometry or plan to texture manually.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 75.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 85.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.10000000149011612, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "trellis2_input", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 3, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 5.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "qwen_generation_method": "generate", "qwen_refine_use_prev_ref": False, "qwen_refine_use_depth": False, "qwen_timestep_zero_ref": False, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05000000074505806, "visibility_vignette": True, "visibility_vignette_width": 0.15000000596046448, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6000000238418579, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "trellis2_texture_mode": "none", "trellis2_initial_image_arch": "sdxl", "trellis2_camera_count": 10, "trellis2_placement_mode": "normal_weighted", "trellis2_auto_prompts": True, "trellis2_exclude_bottom": False, "trellis2_exclude_bottom_angle": 1.5533000230789185, "trellis2_auto_aspect": "per_camera", "trellis2_occlusion_mode": "none", "trellis2_consider_existing": False, "trellis2_delete_cameras": False, "trellis2_coverage_target": 0.949999988079071, "trellis2_max_auto_cameras": 12, "trellis2_fan_angle": 90.0, "trellis2_resolution": "1024_cascade", "trellis2_precision": "bf16", "trellis2_attn_backend": "auto", "trellis2_ss_guidance": 7.5, "trellis2_ss_steps": 12, "trellis2_shape_guidance": 7.5, "trellis2_shape_steps": 12, "trellis2_tex_guidance": 3.0, "trellis2_tex_steps": 12, "trellis2_max_tokens": 49152, "trellis2_texture_size": 4096, "trellis2_decimation": 1000000, "trellis2_remesh": "qdc", "trellis2_post_processing_enabled": True, "trellis2_bg_removal": "auto", "trellis2_background_color": "black", "trellis2_import_scale": 2.0, "trellis2_clamp_elevation": False, "trellis2_max_elevation": 1.2216999530792236, "trellis2_min_elevation": -1.0471975803375244, "trellis2_auto_lighting": True, "use_ipadapter": False, "sequential_ipadapter_regenerate_wo_controlnet": False, "allow_modify_existing_textures": False, "ask_object_prompts": True, "weight_exponent_mask": False, "mask_blocky": False, "architecture_mode": "trellis2", "use_camera_prompts": True, "sg_use_custom_camera_order": False, "pbr_decomposition": False, "generation_mode": "standard", "texture_objects": "all", "use_flux_lora": True, "qwen_use_trellis2_style": False, "qwen_trellis2_style_initial_only": False, "trellis2_skip_texture": True, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "CHARACTERS (MESH + TEXTURE)": {"description": "Uses TRELLIS.2 and SDXL to generate a textured mesh. Optimized settings for generating characters. Will not texture bottom facing faces.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 75.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 85.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.10000000149011612, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "trellis2_input", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 5.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "qwen_generation_method": "generate", "qwen_refine_use_prev_ref": False, "qwen_refine_use_depth": False, "qwen_timestep_zero_ref": False, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05000000074505806, "visibility_vignette": True, "visibility_vignette_width": 0.15000000596046448, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6000000238418579, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "trellis2_texture_mode": "sdxl", "trellis2_initial_image_arch": "sdxl", "trellis2_camera_count": 8, "trellis2_placement_mode": "normal_weighted", "trellis2_auto_prompts": True, "trellis2_exclude_bottom": True, "trellis2_exclude_bottom_angle": 1.5533000230789185, "trellis2_auto_aspect": "per_camera", "trellis2_occlusion_mode": "none", "trellis2_consider_existing": False, "trellis2_delete_cameras": False, "trellis2_coverage_target": 0.949999988079071, "trellis2_max_auto_cameras": 12, "trellis2_fan_angle": 90.0, "trellis2_resolution": "1024_cascade", "trellis2_precision": "bf16", "trellis2_attn_backend": "auto", "trellis2_ss_guidance": 7.5, "trellis2_ss_steps": 12, "trellis2_shape_guidance": 7.5, "trellis2_shape_steps": 12, "trellis2_tex_guidance": 3.0, "trellis2_tex_steps": 12, "trellis2_max_tokens": 49152, "trellis2_texture_size": 4096, "trellis2_decimation": 1000000, "trellis2_remesh": "qdc", "trellis2_post_processing_enabled": True, "trellis2_bg_removal": "auto", "trellis2_background_color": "black", "trellis2_import_scale": 2.0, "trellis2_clamp_elevation": True, "trellis2_max_elevation": 0.8726646259971648, "trellis2_min_elevation": -0.8726646259971648, "trellis2_auto_lighting": True, "use_ipadapter": False, "sequential_ipadapter_regenerate_wo_controlnet": False, "allow_modify_existing_textures": False, "ask_object_prompts": True, "weight_exponent_mask": False, "mask_blocky": False, "architecture_mode": "trellis2", "use_camera_prompts": True, "sg_use_custom_camera_order": False, "pbr_decomposition": False, "generation_mode": "standard", "texture_objects": "all", "use_flux_lora": True, "qwen_use_trellis2_style": False, "qwen_trellis2_style_initial_only": False, "trellis2_skip_texture": True, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "ARCHITECTURE (MESH + TEXTURE)": {"description": "Uses TRELLIS.2 and SDXL to generate a textured mesh. Optimized settings for architecture, and other models with flat walls and sharp angles.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 85.0, "weight_exponent": 10.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "separate", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": False, "sequential_custom_camera_order": "", "sequential_factor": 0.75, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "trellis2_input", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 0.800000011920929, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": False, "blur_mask": True, "blur_mask_radius": 3, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 5.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "qwen_generation_method": "generate", "qwen_refine_use_prev_ref": False, "qwen_refine_use_depth": False, "qwen_timestep_zero_ref": False, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05000000074505806, "visibility_vignette": True, "visibility_vignette_width": 0.15000000596046448, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6000000238418579, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "trellis2_texture_mode": "sdxl", "trellis2_initial_image_arch": "sdxl", "trellis2_camera_count": 8, "trellis2_placement_mode": "normal_weighted", "trellis2_auto_prompts": True, "trellis2_exclude_bottom": True, "trellis2_exclude_bottom_angle": 1.5533000230789185, "trellis2_auto_aspect": "per_camera", "trellis2_occlusion_mode": "none", "trellis2_consider_existing": False, "trellis2_delete_cameras": False, "trellis2_coverage_target": 0.949999988079071, "trellis2_max_auto_cameras": 12, "trellis2_fan_angle": 90.0, "trellis2_resolution": "1024_cascade", "trellis2_precision": "bf16", "trellis2_attn_backend": "auto", "trellis2_ss_guidance": 7.5, "trellis2_ss_steps": 12, "trellis2_shape_guidance": 7.5, "trellis2_shape_steps": 12, "trellis2_tex_guidance": 3.0, "trellis2_tex_steps": 12, "trellis2_max_tokens": 49152, "trellis2_texture_size": 4096, "trellis2_decimation": 1000000, "trellis2_remesh": "qdc", "trellis2_post_processing_enabled": True, "trellis2_bg_removal": "auto", "trellis2_background_color": "black", "trellis2_import_scale": 2.0, "trellis2_clamp_elevation": False, "trellis2_max_elevation": 1.2216999530792236, "trellis2_min_elevation": -1.0471975803375244, "trellis2_auto_lighting": True, "use_ipadapter": False, "sequential_ipadapter_regenerate_wo_controlnet": False, "allow_modify_existing_textures": False, "ask_object_prompts": True, "weight_exponent_mask": False, "mask_blocky": False, "architecture_mode": "trellis2", "use_camera_prompts": True, "sg_use_custom_camera_order": False, "pbr_decomposition": False, "generation_mode": "standard", "texture_objects": "all", "use_flux_lora": True, "qwen_use_trellis2_style": False, "qwen_trellis2_style_initial_only": False, "trellis2_skip_texture": True, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "QWEN PRECISE (MESH + TEXTURE)": {"description": "Uses TRELLIS.2 and Qwen Image Edit to generate a textured mesh. Precise detail when camera overlap is good. Relies on context renders plus the prompt.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "qwen_generation_method": "generate", "qwen_refine_use_prev_ref": False, "qwen_refine_use_depth": False, "qwen_timestep_zero_ref": False, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05000000074505806, "visibility_vignette": True, "visibility_vignette_width": 0.15000000596046448, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "sg_silhouette_margin": 3, "sg_silhouette_depth": 0.05000000074505806, "sg_silhouette_rays": "4", "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6000000238418579, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "trellis2_texture_mode": "qwen_image_edit", "trellis2_initial_image_arch": "sdxl", "trellis2_camera_count": 8, "trellis2_placement_mode": "normal_weighted", "trellis2_auto_prompts": True, "trellis2_exclude_bottom": True, "trellis2_exclude_bottom_angle": 1.5533000230789185, "trellis2_auto_aspect": "per_camera", "trellis2_occlusion_mode": "none", "trellis2_consider_existing": False, "trellis2_delete_cameras": False, "trellis2_coverage_target": 0.949999988079071, "trellis2_max_auto_cameras": 12, "trellis2_fan_angle": 90.0, "trellis2_resolution": "1024_cascade", "trellis2_precision": "bf16", "trellis2_attn_backend": "auto", "trellis2_seed": 0, "trellis2_ss_guidance": 7.5, "trellis2_ss_steps": 12, "trellis2_shape_guidance": 7.5, "trellis2_shape_steps": 12, "trellis2_tex_guidance": 3.0, "trellis2_tex_steps": 12, "trellis2_max_tokens": 49152, "trellis2_texture_size": 4096, "trellis2_decimation": 1000000, "trellis2_remesh": "qdc", "trellis2_post_processing_enabled": True, "trellis2_bg_removal": "auto", "trellis2_background_color": "black", "trellis2_import_scale": 2.0, "trellis2_clamp_elevation": True, "trellis2_max_elevation": 0.8726646259971648, "trellis2_min_elevation": -0.8726646259971648, "trellis2_auto_lighting": True, "use_ipadapter": False, "sequential_ipadapter_regenerate_wo_controlnet": False, "allow_modify_existing_textures": False, "ask_object_prompts": True, "weight_exponent_mask": False, "mask_blocky": False, "architecture_mode": "trellis2", "use_camera_prompts": True, "sg_use_custom_camera_order": False, "pbr_decomposition": False, "generation_mode": "standard", "texture_objects": "all", "use_flux_lora": True, "qwen_use_trellis2_style": True, "qwen_trellis2_style_initial_only": True, "trellis2_skip_texture": True, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "QWEN SAFE (MESH + TEXTURE)": {"description": "Uses TRELLIS.2 and Qwen Image Edit to generate a textured mesh. Safer fallback when coverage is limited. Uses TRELLIS.2 input image as sequential style reference for global coherence.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 6.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "qwen_generation_method": "generate", "qwen_refine_use_prev_ref": False, "qwen_refine_use_depth": False, "qwen_timestep_zero_ref": False, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05000000074505806, "visibility_vignette": True, "visibility_vignette_width": 0.15000000596046448, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "sg_silhouette_margin": 3, "sg_silhouette_depth": 0.05000000074505806, "sg_silhouette_rays": "4", "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6000000238418579, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "trellis2_texture_mode": "qwen_image_edit", "trellis2_initial_image_arch": "sdxl", "trellis2_camera_count": 8, "trellis2_placement_mode": "normal_weighted", "trellis2_auto_prompts": True, "trellis2_exclude_bottom": True, "trellis2_exclude_bottom_angle": 1.5533000230789185, "trellis2_auto_aspect": "per_camera", "trellis2_occlusion_mode": "none", "trellis2_consider_existing": False, "trellis2_delete_cameras": False, "trellis2_coverage_target": 0.949999988079071, "trellis2_max_auto_cameras": 12, "trellis2_fan_angle": 90.0, "trellis2_resolution": "1024_cascade", "trellis2_precision": "bf16", "trellis2_attn_backend": "auto", "trellis2_seed": 0, "trellis2_ss_guidance": 7.5, "trellis2_ss_steps": 12, "trellis2_shape_guidance": 7.5, "trellis2_shape_steps": 12, "trellis2_tex_guidance": 3.0, "trellis2_tex_steps": 12, "trellis2_max_tokens": 49152, "trellis2_texture_size": 4096, "trellis2_decimation": 1000000, "trellis2_remesh": "qdc", "trellis2_post_processing_enabled": True, "trellis2_bg_removal": "auto", "trellis2_background_color": "black", "trellis2_import_scale": 2.0, "trellis2_clamp_elevation": True, "trellis2_max_elevation": 0.8726646259971648, "trellis2_min_elevation": -0.8726646259971648, "trellis2_auto_lighting": True, "use_ipadapter": False, "sequential_ipadapter_regenerate_wo_controlnet": False, "allow_modify_existing_textures": False, "ask_object_prompts": True, "weight_exponent_mask": False, "mask_blocky": False, "architecture_mode": "trellis2", "use_camera_prompts": True, "sg_use_custom_camera_order": False, "pbr_decomposition": False, "generation_mode": "standard", "texture_objects": "all", "use_flux_lora": True, "qwen_use_trellis2_style": True, "qwen_trellis2_style_initial_only": False, "trellis2_skip_texture": True, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "QWEN ALT (MESH + TEXTURE)": {"description": "Uses TRELLIS.2 and Qwen Image Edit to generate a textured mesh. Uses ADDITIONAL context renders with TRELLIS.2 style transfer for consistency. Good general-purpose Qwen pipeline.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": True, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "ADDITIONAL", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "qwen_generation_method": "generate", "qwen_refine_use_prev_ref": False, "qwen_refine_use_depth": False, "qwen_timestep_zero_ref": False, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05000000074505806, "visibility_vignette": True, "visibility_vignette_width": 0.15000000596046448, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "sg_silhouette_margin": 3, "sg_silhouette_depth": 0.05000000074505806, "sg_silhouette_rays": "4", "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6000000238418579, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "trellis2_texture_mode": "qwen_image_edit", "trellis2_initial_image_arch": "sdxl", "trellis2_camera_count": 8, "trellis2_placement_mode": "normal_weighted", "trellis2_auto_prompts": True, "trellis2_exclude_bottom": True, "trellis2_exclude_bottom_angle": 1.5533000230789185, "trellis2_auto_aspect": "per_camera", "trellis2_occlusion_mode": "none", "trellis2_consider_existing": False, "trellis2_delete_cameras": False, "trellis2_coverage_target": 0.949999988079071, "trellis2_max_auto_cameras": 12, "trellis2_fan_angle": 90.0, "trellis2_resolution": "1024_cascade", "trellis2_precision": "bf16", "trellis2_attn_backend": "auto", "trellis2_seed": 0, "trellis2_ss_guidance": 7.5, "trellis2_ss_steps": 12, "trellis2_shape_guidance": 7.5, "trellis2_shape_steps": 12, "trellis2_tex_guidance": 3.0, "trellis2_tex_steps": 12, "trellis2_max_tokens": 49152, "trellis2_texture_size": 4096, "trellis2_decimation": 1000000, "trellis2_remesh": "qdc", "trellis2_post_processing_enabled": True, "trellis2_bg_removal": "auto", "trellis2_background_color": "black", "trellis2_import_scale": 2.0, "trellis2_clamp_elevation": True, "trellis2_max_elevation": 0.8726646259971648, "trellis2_min_elevation": -0.8726646259971648, "trellis2_auto_lighting": True, "use_ipadapter": False, "sequential_ipadapter_regenerate_wo_controlnet": False, "allow_modify_existing_textures": False, "ask_object_prompts": True, "weight_exponent_mask": False, "mask_blocky": False, "architecture_mode": "trellis2", "use_camera_prompts": True, "sg_use_custom_camera_order": False, "pbr_decomposition": False, "generation_mode": "standard", "texture_objects": "all", "use_flux_lora": True, "qwen_use_trellis2_style": True, "qwen_trellis2_style_initial_only": False, "trellis2_skip_texture": True, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "QWEN VORONOI (MESH + TEXTURE)": {"description": "Uses TRELLIS.2 and Qwen Image Edit to generate a textured mesh. Voronoi projection with exponent 1000 for hard camera segmentation during generation, then resets to 15 for softer blending.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 1000.0, "weight_exponent_generation_only": True, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": True, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_prompt_gray_background": True, "qwen_rescale_alignment": True, "qwen_generation_method": "generate", "qwen_refine_use_prev_ref": False, "qwen_refine_use_depth": False, "qwen_timestep_zero_ref": False, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05000000074505806, "visibility_vignette": True, "visibility_vignette_width": 0.15000000596046448, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "sg_silhouette_margin": 3, "sg_silhouette_depth": 0.05000000074505806, "sg_silhouette_rays": "4", "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6000000238418579, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "trellis2_texture_mode": "qwen_image_edit", "trellis2_initial_image_arch": "sdxl", "trellis2_camera_count": 8, "trellis2_placement_mode": "normal_weighted", "trellis2_auto_prompts": True, "trellis2_exclude_bottom": True, "trellis2_exclude_bottom_angle": 1.5533000230789185, "trellis2_auto_aspect": "per_camera", "trellis2_occlusion_mode": "none", "trellis2_consider_existing": False, "trellis2_delete_cameras": False, "trellis2_coverage_target": 0.949999988079071, "trellis2_max_auto_cameras": 12, "trellis2_fan_angle": 90.0, "trellis2_resolution": "1024_cascade", "trellis2_precision": "bf16", "trellis2_attn_backend": "auto", "trellis2_seed": 0, "trellis2_ss_guidance": 7.5, "trellis2_ss_steps": 12, "trellis2_shape_guidance": 7.5, "trellis2_shape_steps": 12, "trellis2_tex_guidance": 3.0, "trellis2_tex_steps": 12, "trellis2_max_tokens": 49152, "trellis2_texture_size": 4096, "trellis2_decimation": 1000000, "trellis2_remesh": "qdc", "trellis2_post_processing_enabled": True, "trellis2_bg_removal": "auto", "trellis2_background_color": "black", "trellis2_import_scale": 2.0, "trellis2_clamp_elevation": True, "trellis2_max_elevation": 0.8726646259971648, "trellis2_min_elevation": -0.8726646259971648, "trellis2_auto_lighting": True, "use_ipadapter": False, "sequential_ipadapter_regenerate_wo_controlnet": False, "allow_modify_existing_textures": False, "ask_object_prompts": True, "weight_exponent_mask": False, "mask_blocky": False, "architecture_mode": "trellis2", "use_camera_prompts": True, "sg_use_custom_camera_order": False, "pbr_decomposition": False, "generation_mode": "standard", "texture_objects": "all", "use_flux_lora": True, "qwen_use_trellis2_style": True, "qwen_trellis2_style_initial_only": False, "trellis2_skip_texture": True, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
}

# Diagnostic base-model preset: keep QWEN PRECISE's camera/context/projection
# behavior, but remove the 4-step Lightning LoRA and run the 2509 base model
# with the reference 40-step / CFG 4 sampling settings.  It deliberately does
# not enable any texturing experiments; those remain independently testable.
PRESETS["QWEN 2509 FULL STEPS (EXPERIMENTAL)"] = {
    **PRESETS["QWEN PRECISE"],
    "description": "Experimental Qwen 2509 base-model comparison. Matches QWEN PRECISE except it removes the Lightning LoRA and uses 40 steps at CFG 4. Requires a regular Qwen-Image-Edit-2509 base model, not a Nunchaku checkpoint with Lightning baked in.",
    "steps": 40,
    "cfg": 4.0,
    "lora_units": [],
}

# Regular controls are available without enabling research features.
PRESETS["QWEN PRECISE+"] = {
    **PRESETS["QWEN PRECISE"],
    "description": (
        "Qwen Precise with the evaluated combination of three context and projection controls for sequential multi-view texturing. "
        "Confidence-Gated Context (CGC) keeps reliable existing texture as context and marks low-quality regions for repair. "
        "Acquisition Quality Weighting (AQW) favors source views with better projected detail and acquisition quality when blending textures. "
        "Depth/Part Projection Guard (DPPG) reduces projection contributions near depth, object and material boundaries to limit foreground silhouettes leaking onto rear surfaces. "
        "Retains Qwen Precise's 4-step Lightning setup, sampler and context settings; uses a 0.50 trust threshold, a 3 px silhouette margin and backface projection. "
        "Experimental options are off. Intended to improve context reliability and projection consistency, but fine patterns and material continuity can still vary, and some assets may look better with standard Qwen Precise."
    ),
    "qwen_use_confidence_context": True,
    "sg_rework_acquisition_quality": True,
    "sg_rework_projection_guard": True,
    "qwen_trust_threshold": 0.5,
    "sg_silhouette_margin": 3,
    "sg_allow_backface_projection": True,
    **EXPERIMENTAL_FEATURE_DEFAULTS,
}

# Qwen-Image 2.1 has its own architecture and native sampler presets. These
# retain the evaluated StableGen view/context strategies and use the sampler
# combination that performed best in the projection-oriented comparison.
PRESETS["QWEN 2.1 PRECISE"] = {
    **PRESETS["QWEN PRECISE"],
    "weight_exponent": 7.0,
    "description": (
        "Qwen-Image 2.1 multi-view texturing using projected context renders "
        "to preserve already textured regions and repair uncovered areas. "
        "Uses the tested projection-oriented configuration of 20 steps, CFG 1, "
        "DPM++ SDE and SGM Uniform with no Lightning LoRA. This combination "
        "produced broad, coherent panels that followed the model silhouette and "
        "continued well into later views. Best when adjacent cameras overlap "
        "well; sparse coverage can still introduce artifacts. Qwen 2.1 is also "
        "prompted to generate native RGBA views, independently of the sampler, "
        "and transparent pixels are excluded from texture projection."
    ),
    "model_architecture": "qwen_image_21",
    "steps": 20,
    "cfg": 1.0,
    "sampler": "dpmpp_sde",
    "scheduler": "sgm_uniform",
    "qwen_image_21_native_rgba": True,
    "qwen_guidance_background_color": mathutils.Color((0.5, 0.5, 0.5)),
    "lora_units": [],
}

PRESETS["QWEN 2.1 PRECISE+"] = {
    **PRESETS["QWEN 2.1 PRECISE"],
    "description": (
        "Qwen-Image 2.1 Precise with Confidence-Gated Context (CGC), Acquisition "
        "Quality Weighting (AQW) and Depth/Part Projection Guard (DPPG). CGC "
        "retains reliable projected texture as context and marks low-confidence "
        "regions for repair. AQW favors views with better projected detail when "
        "blending textures. DPPG limits foreground silhouettes leaking onto "
        "surfaces behind them at depth, object and material boundaries. "
        "Uses a 0.50 context trust threshold, a 3 px silhouette margin and "
        "backface projection. Retains Qwen 2.1 Precise's projected-context "
        "strategy, 20 steps, CFG 1.0, DPM++ SDE, SGM Uniform, blending exponent "
        "7 and native RGBA output, with no Lightning LoRA. Transparent pixels "
        "are excluded from projection. Research options are off. Best suited "
        "to overlapping camera views; fine patterns and material continuity "
        "can still vary, and standard Precise can work better on some assets."
    ),
    "architecture_mode": "qwen_image_21",
    "generation_mode": "standard",
    "qwen_use_confidence_context": True,
    "sg_rework_acquisition_quality": True,
    "sg_rework_projection_guard": True,
    "qwen_trust_threshold": 0.5,
    "sg_silhouette_margin": 3,
    "sg_allow_backface_projection": True,
    **EXPERIMENTAL_FEATURE_DEFAULTS,
}

PRESETS["QWEN 2.1 SAFE"] = {
    **PRESETS["QWEN SAFE"],
    "weight_exponent": 7.0,
    "description": (
        "Qwen-Image 2.1 multi-view texturing that supplies the initial generated "
        "view as an appearance reference for every later camera. This was the "
        "strongest evaluated reference strategy for retaining distinctive colors "
        "and details around the object. Uses the tested projection-oriented "
        "configuration of 20 steps, CFG 1, DPM++ SDE and SGM Uniform with no "
        "Lightning LoRA. It favors broad, geometry-following texture regions. "
        "Qwen 2.1 is prompted to generate native RGBA views independently of the "
        "sampler, and transparent pixels are excluded from texture projection. "
        "It is slower when the initial reference is tall or large."
    ),
    "model_architecture": "qwen_image_21",
    "steps": 20,
    "cfg": 1.0,
    "sampler": "dpmpp_sde",
    "scheduler": "sgm_uniform",
    "qwen_image_21_native_rgba": True,
    "sequential_ipadapter": True,
    "sequential_ipadapter_mode": "first",
    "qwen_context_render_mode": "NONE",
    "qwen_guidance_background_color": mathutils.Color((0.5, 0.5, 0.5)),
    "lora_units": [],
}

PRESETS["QWEN 2.1 CONTEXT + REFERENCE"] = {
    **PRESETS["QWEN 2.1 PRECISE"],
    "description": (
        "Qwen-Image 2.1 sequential texturing combining the live projected context "
        "with the first generated view as a persistent appearance reference. "
        "Uses Additional Context and the Qwen 2.1 Default prompt template: the "
        "model fills magenta regions while retaining existing texture, with a "
        "depth guide for geometry and the initial view for overall appearance. "
        "A 50-degree discard angle excludes grazing projections during generation, "
        "then resets to 90 degrees for the final material; blending exponent is 7. "
        "Uses 20 steps, CFG 1.0, DPM++ SDE, SGM Uniform, native RGBA output and no "
        "LoRA. Transparent generated pixels are excluded from projection. "
        "Confidence-Gated Context, Acquisition Quality Weighting and Depth/Part "
        "Projection Guard are off; legacy fallback dilation is 1 px, cleanup is "
        "off and backface projection is allowed. Captures the current manually "
        "tuned setup; intended for comparing context coverage and appearance "
        "consistency across overlapping views."
    ),
    "architecture_mode": "qwen_image_21",
    "generation_mode": "standard",
    "discard_factor": 50.0,
    "sequential_ipadapter": True,
    "sequential_ipadapter_mode": "first",
    "qwen_context_render_mode": "ADDITIONAL",
    "qwen_timestep_zero_ref": False,
    "auto_rescale_target_mp": 1.0,
    "early_priority": False,
    "early_priority_strength": 0.5,
    "view_blend_use_color_match": False,
    "view_blend_color_match_method": "hm-mvgd-hm",
    "view_blend_color_match_strength": 1.0,
    "allow_modify_existing_textures": False,
    "weight_exponent_mask": False,
    "mask_blocky": False,
    "sg_silhouette_margin": 3,
    "sg_silhouette_depth": 0.05,
    "sg_silhouette_rays": "4",
    "qwen_use_trellis2_style": False,
    **TEXTURING_DEFAULTS,
}

PRESETS["LOCAL EDIT (QWEN 2.1)"] = {
    **PRESETS["LOCAL EDIT (QWEN)"],
    "description": (
        "Use Qwen-Image 2.1 to make targeted changes to an existing texture. "
        "Point detail cameras at the area to edit and describe the desired "
        "change, such as a new color, lettering, style or surface detail. "
        "Preserves the original Local Edit camera coverage and edge-feathering "
        "settings so the edited projections blend into the existing material. "
        "Uses 20 steps, CFG 1.0, DPM++ SDE, SGM Uniform, blending exponent 7 "
        "and native RGBA output, with no Lightning LoRA. Transparent generated "
        "pixels do not contribute to projection. This edits camera-selected "
        "areas; use Masked Edit generation mode for painted-mask selection."
    ),
    "architecture_mode": "qwen_image_21",
    "model_architecture": "qwen_image_21",
    "generation_mode": "standard",
    "weight_exponent": 7.0,
    "steps": 20,
    "cfg": 1.0,
    "sampler": "dpmpp_sde",
    "scheduler": "sgm_uniform",
    "qwen_image_21_native_rgba": True,
    "qwen_guidance_background_color": mathutils.Color((0.5, 0.5, 0.5)),
    "lora_units": [],
}

PRESETS["REFINE (QWEN 2.1)"] = {
    **PRESETS["REFINE (QWEN)"],
    "description": (
        "Use Qwen-Image 2.1 to restyle or modify the existing texture across "
        "the scene's camera views. Each view uses the current textured render "
        "as its editing input, making this suitable for overall color, art "
        "style or surface-appearance changes. Preserves the original Qwen "
        "Refine workflow settings. Uses 20 steps, CFG 1.0, DPM++ SDE, SGM "
        "Uniform, blending exponent 7 and native RGBA output, with no Lightning "
        "LoRA. Transparent generated pixels are excluded from projection. "
        "Use Local Edit for camera-targeted changes or Masked Edit generation "
        "mode to select an area by painting."
    ),
    "architecture_mode": "qwen_image_21",
    "model_architecture": "qwen_image_21",
    "generation_mode": "standard",
    "weight_exponent": 7.0,
    "steps": 20,
    "cfg": 1.0,
    "sampler": "dpmpp_sde",
    "scheduler": "sgm_uniform",
    "qwen_image_21_native_rgba": True,
    "qwen_guidance_background_color": mathutils.Color((0.5, 0.5, 0.5)),
    "lora_units": [],
}

# Stock presets select their architecture's native guidance wording. Including
# the enum makes applying one reset a previously selected comparison template.
for _preset_values in PRESETS.values():
    _preset_architecture = _preset_values.get("model_architecture")
    if _preset_architecture == "qwen_image_edit":
        _preset_values.setdefault("qwen_prompt_template", "QWEN_EDIT_LEGACY")
    elif _preset_architecture == "qwen_image_21":
        _preset_values.setdefault("qwen_prompt_template", "QWEN_21_DEFAULT")
    if _preset_architecture in {"qwen_image_edit", "qwen_image_21"}:
        # Prepared-template fields are derived live. Keeping stale copies in a
        # stock preset would overwrite them after the template is selected.
        for _prompt_field in (
            "qwen_custom_prompt_initial",
            "qwen_custom_prompt_seq_none",
            "qwen_custom_prompt_seq_replace",
            "qwen_custom_prompt_seq_additional",
        ):
            _preset_values.pop(_prompt_field, None)

# Global list of all generation parameter names to check for a preset.
GEN_PARAMETERS = [
    'control_after_generate',
    'model_architecture',
    'steps',
    'cfg',
    'sampler',
    'scheduler',
    'fallback_color',
    'discard_factor',
    'discard_factor_generation_only',
    'discard_factor_after_generation',
    'weight_exponent',
    'weight_exponent_generation_only',
    'weight_exponent_after_generation',
    'view_blend_use_color_match',
    'view_blend_color_match_method',
    'view_blend_color_match_strength',
    'clip_skip',
    'auto_rescale',
    'auto_rescale_target_mp',
    'overwrite_material',
    'generation_method',
    'refine_images',
    'refine_steps',
    'refine_sampler',
    'refine_scheduler',
    'denoise',
    'refine_cfg',
    'refine_prompt',
    'refine_upscale_method',
    'sequential_smooth',
    'sequential_custom_camera_order',
    'sequential_factor',
    'sequential_factor_smooth',
    'sequential_factor_smooth_2',
    'sequential_ipadapter',
    'sequential_ipadapter_mode',
    'sequential_desaturate_factor',
    'sequential_contrast_factor',
    'sequential_ipadapter_regenerate',
    'ipadapter_weight_type',
    'ipadapter_strength',
    'ipadapter_start',
    'ipadapter_end',
    'early_priority',
    'early_priority_strength',
    'sg_auto_view_sequence',
    'sg_anchor_camera',
    'sg_texturing_frame',
    'sg_rework_aligned_resolution',
    'sg_rework_full_resolution_scale',
    'sg_rework_acquisition_quality',
    'sg_rework_projection_guard',
    'qwen_global_style_vlm_only',
    'qwen_context_vlm_only',
    'qwen_latent_locked_context',
    'qwen_latent_lock_neutral_fill',
    'sg_rework_final_display',
    'differential_diffusion',
    'differential_noise',
    'blur_mask',
    'blur_mask_radius',
    'blur_mask_sigma',
    'grow_mask_by',
    'canny_threshold_low',
    'canny_threshold_high',
    'qwen_guidance_map_type',
    'qwen_voronoi_mode',
    'qwen_use_external_style_image',
    'qwen_external_style_image',
    'qwen_context_render_mode',
    'qwen_external_style_initial_only',
    'qwen_use_custom_prompts',
    'qwen_prompt_template',
    'qwen_custom_prompt_initial',
    'qwen_custom_prompt_seq_none',
    'qwen_custom_prompt_seq_replace',
    'qwen_custom_prompt_seq_additional',
    'qwen_guidance_fallback_color',
    'qwen_guidance_background_color',
    'qwen_context_cleanup',
    'qwen_context_cleanup_hue_tolerance',
    'qwen_context_cleanup_value_adjust',
    'qwen_context_fallback_dilation',
    'qwen_use_confidence_context',
    'qwen_regularize_context_mask',
    'qwen_use_projection_write_masks',
    'qwen_reject_residual_fallback',
    'qwen_trust_threshold',
    'qwen_mask_min_island_area',
    'qwen_mask_min_hole_area',
    'qwen_mask_min_strip_width',
    'qwen_mask_uncertainty_dilation',
    'qwen_mask_keep_largest_components',
    'qwen_residual_fallback_tolerance',
    'qwen_prompt_gray_background',
    'qwen_rescale_alignment',
    'qwen_image_21_native_rgba',
    'qwen_generation_method',
    'qwen_refine_use_prev_ref',
    'qwen_refine_use_depth',
    'qwen_timestep_zero_ref',
    'refine_angle_ramp_active',
    'refine_angle_ramp_pos_0',
    'refine_angle_ramp_pos_1',
    'visibility_vignette',
    'visibility_vignette_width',
    'visibility_vignette_softness',
    'visibility_vignette_blur',
    'sg_silhouette_margin',
    'sg_silhouette_feather',
    'sg_projection_guard_scale',
    'sg_guard_feather_method',
    'sg_guard_directional',
    'sg_guard_foreground_margin',
    'sg_guard_coverage_rescue',
    'sg_silhouette_depth',
    'sg_silhouette_rays',
    'sg_allow_backface_projection',
    'refine_feather_ramp_pos_0',
    'refine_feather_ramp_pos_1',
    'refine_edge_feather_projection',
    'refine_edge_feather_width',
    'refine_edge_feather_softness',
    'trellis2_texture_mode',
    'trellis2_initial_image_arch',
    'trellis2_camera_count',
    'trellis2_placement_mode',
    'trellis2_auto_prompts',
    'trellis2_exclude_bottom',
    'trellis2_exclude_bottom_angle',
    'trellis2_auto_aspect',
    'trellis2_occlusion_mode',
    'trellis2_consider_existing',
    'trellis2_delete_cameras',
    'trellis2_coverage_target',
    'trellis2_max_auto_cameras',
    'trellis2_fan_angle',
    'trellis2_resolution',
    'trellis2_precision',
    'trellis2_attn_backend',
    'trellis2_seed',
    'trellis2_ss_guidance',
    'trellis2_ss_steps',
    'trellis2_shape_guidance',
    'trellis2_shape_steps',
    'trellis2_tex_guidance',
    'trellis2_tex_steps',
    'trellis2_max_tokens',
    'trellis2_texture_size',
    'trellis2_decimation',
    'trellis2_remesh',
    'trellis2_decimate_method',
    'trellis2_remesh_method',
    'trellis2_post_processing_enabled',
    'trellis2_autorig_enabled',
    'trellis2_autorig_use_transfer',
    'trellis2_autorig_use_skeleton',
    'trellis2_autorig_use_postprocess',
    'trellis2_autorig_cleanup_cross_limb_weights',
    'trellis2_autorig_garment_mode',
    'trellis2_autorig_bone_display_type',
    'trellis2_autorig_bone_heuristic',
    'trellis2_autorig_top_k',
    'trellis2_autorig_top_p',
    'trellis2_autorig_temperature',
    'trellis2_autorig_repetition_penalty',
    'trellis2_autorig_num_beams',
    'trellis2_autorig_model_ckpt',
    'trellis2_autorig_hf_path',
    'trellis2_bg_removal',
    'trellis2_background_color',
    'trellis2_import_scale',
    'trellis2_clamp_elevation',
    'trellis2_max_elevation',
    'trellis2_min_elevation',
    'trellis2_auto_lighting',
    'use_ipadapter',
    'sequential_ipadapter_regenerate_wo_controlnet',
    'allow_modify_existing_textures',
    'ask_object_prompts',
    'weight_exponent_mask',
    'mask_blocky',
    'architecture_mode',
    'use_camera_prompts',
    'sg_use_custom_camera_order',
    'generation_mode',
    'texture_objects',
    'use_flux_lora',
    'qwen_use_trellis2_style',
    'qwen_trellis2_style_initial_only',
    'trellis2_skip_texture',
    'pbr_decomposition',
    'pbr_albedo_saturation_mode',
]

# ── Preset diff preview helpers ──────────────────────────────────────────


def _build_preset_items(context, include_deprecated, selected=None):
    # Group presets by architecture for easier navigation.
    _ARCH_GROUP_ORDER = [
        ('sdxl',            'SDXL'),
        ('qwen_image_edit', 'Qwen Image Edit'),
        ('qwen_image_21',   'Qwen Image 2.1'),
        ('flux2_klein',     'FLUX.2 Klein (Deprecated)'),
        ('trellis2',        'TRELLIS.2 Pipeline'),
    ]
    _known_groups = {k for k, _ in _ARCH_GROUP_ORDER}

    # Classify each preset into its architecture group.
    groups = {}  # group_key -> list of (identifier, name, description)
    for key, preset in PRESETS.items():
        if '(EXPERIMENTAL)' in key and not experiments_enabled(context):
            continue
        deprecated = preset_uses_deprecated_architecture(preset)
        if deprecated and not include_deprecated and key != selected:
            continue
        if preset.get('architecture_mode') == 'trellis2':
            grp = 'trellis2'
        else:
            grp = preset.get('model_architecture', 'sdxl')
        groups.setdefault(grp, []).append(
            (key, key.title() + (' (Deprecated)' if deprecated else ''),
             preset.get('description', f'Preset {key}')
             + (f'. {DEPRECATION_NOTICE}' if deprecated else ''))
        )

    qwen_edit_items = groups.get('qwen_image_edit', [])
    qwen_edit_ids = [item[0] for item in qwen_edit_items]
    if 'QWEN PRECISE' in qwen_edit_ids and 'QWEN PRECISE+' in qwen_edit_ids:
        precise_plus = qwen_edit_items.pop(qwen_edit_ids.index('QWEN PRECISE+'))
        precise_index = next(i for i, item in enumerate(qwen_edit_items)
                             if item[0] == 'QWEN PRECISE')
        qwen_edit_items.insert(precise_index + 1, precise_plus)

    items = []
    first_group = True
    for grp_key, _label in _ARCH_GROUP_ORDER:
        if grp_key not in groups:
            continue
        if not first_group:
            items.append(('', '', ''))          # horizontal separator
        # Section header: empty identifier + label name renders as header
        # in Blender 4.x enum popup menus.
        items.append(('', _label, ''))
        first_group = False
        items.extend(groups[grp_key])

    # Any architectures not explicitly listed above (future-proofing).
    for grp_key, grp_items in groups.items():
        if grp_key not in _known_groups:
            items.append(('', '', ''))
            label = ('FLUX.1 (Deprecated)' if grp_key == 'flux1'
                     else grp_key.replace('_', ' ').title())
            items.append(('', label, ''))
            items.extend(grp_items)

    items.append(('', '', ''))                  # separator before Custom
    items.append(('CUSTOM', 'Custom', 'Custom configuration'))
    return items


_PRESET_ENUM_CACHE = {}
_READING_PRESET_SELECTION = False


def get_preset_items(self, context):
    global _READING_PRESET_SELECTION
    # Keep the old, unfiltered menu's numeric values. Hiding Klein presets must
    # not silently turn a saved TRELLIS preset (or Custom) into another preset.
    full_items = _build_preset_items(context, include_deprecated=True)
    numbers = {item[0]: index for index, item in enumerate(full_items) if item[0]}
    if _READING_PRESET_SELECTION:
        items = tuple(item + (numbers[item[0]],) if item[0] else item
                      for item in full_items)
        return _PRESET_ENUM_CACHE.setdefault(items, items)
    # Reading through RNA is portable across Blender's old and new property
    # storage. The nested items callback above temporarily supplies the full menu.
    _READING_PRESET_SELECTION = True
    try:
        selected = getattr(self, 'stablegen_preset', None)
    finally:
        _READING_PRESET_SELECTION = False
    visible_items = _build_preset_items(
        context, deprecated_architectures_enabled(context), selected=selected,
    )
    items = tuple(item + (numbers[item[0]],) if item[0] else item
                  for item in visible_items)
    # Blender's dynamic enums borrow their strings; keep every returned variant.
    return _PRESET_ENUM_CACHE.setdefault(items, items)

def normalize_value(key, value):
    if key == "trellis2_remesh" and isinstance(value, bool):
        return "qdc" if value else "none"
    return value

def update_parameters(self, context):
    scene = context.scene
    # Build a dictionary of current parameter values
    current = {key: normalize_value(key, getattr(scene, key)) for key in GEN_PARAMETERS if hasattr(scene, key)}
    
    # Compare current values with every stock preset's stored values
    for name, preset in PRESETS.items():
        if '(EXPERIMENTAL)' in name and not experiments_enabled(context):
            continue
        if (preset_uses_deprecated_architecture(preset)
                and not deprecated_architectures_enabled(context)):
            continue
        expected = preset if preset.get('custom', False) else {**TEXTURING_DEFAULTS, **preset}
        # First check regular parameters
        if all(
            (lambda v1, v2: math.isclose(v1, v2, rel_tol=1e-7, abs_tol=0.0) if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else v1 == v2)
            (current.get(key), normalize_value(key, expected.get(key)))
            for key in GEN_PARAMETERS if key in expected
        ):
            units_match = True

            # Now check ControlNet units if present in the preset
            if "controlnet_units" in preset:
                # Get current controlnet units as comparable data
                current_units = []
                for unit in scene.controlnet_units:
                    unit_data = {
                        "unit_type": unit.unit_type,
                        "model_name": unit.model_name,
                        "strength": unit.strength,
                        "start_percent": unit.start_percent,
                        "end_percent": unit.end_percent,
                        "is_union": unit.is_union,
                        "use_union_type": unit.use_union_type
                    }
                    
                    current_units.append(unit_data)
                
                # Compare unit count
                if len(current_units) != len(preset["controlnet_units"]):
                    continue  # Different number of units, not a match
                
                # Compare each unit's properties
                units_match = True
                for i, unit_data in enumerate(current_units):
                    preset_unit = preset["controlnet_units"][i]
                    for key, value in unit_data.items():
                        if key not in preset_unit or preset_unit[key] != value:
                            units_match = False
                            break
                    if not units_match:
                        break
                
            if not units_match:
                continue  # ControlNet units don't match, try next preset

            # Now check LoRA units if present in the preset
            if "lora_units" in preset:
                current_lora_units_data = []
                for lora_unit_obj in scene.lora_units:
                    current_lora_units_data.append({
                        "model_name": lora_unit_obj.model_name,
                        "model_strength": round(lora_unit_obj.model_strength, 7),
                        "clip_strength": round(lora_unit_obj.clip_strength, 7),
                    })

                preset_lora_units_data = preset["lora_units"]
                if len(current_lora_units_data) != len(preset_lora_units_data):
                    continue # Different number of LoRA units

                lora_units_match = True
                for i, current_lora_unit_data in enumerate(current_lora_units_data):
                    preset_lora_unit_data = preset_lora_units_data[i]
                    for key, value in current_lora_unit_data.items():
                        if key not in preset_lora_unit_data or preset_lora_unit_data[key] != value:
                            lora_units_match = False
                            break
                    if not lora_units_match:
                        break
                
                if not lora_units_match:
                    continue # LoRA units don't match, try next preset
            
            # All parameters and ControlNet and LoRA units match
            if scene.stablegen_preset != name:
                scene.stablegen_preset = name
                scene.active_preset = name
            return

    # No match found, set to custom
    scene.active_preset = "CUSTOM"
    scene.stablegen_preset = "CUSTOM"


class ResetQwenPrompt(bpy.types.Operator):
    """Resets a guidance prompt to its default value"""
    bl_idname = "stablegen.reset_qwen_prompt"
    bl_label = "Reset Qwen Prompt"
    bl_description = "Reset this prompt to its default value based on the current settings"

    prompt_type: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        from ..workflows import WorkflowManager
        # We need an operator instance to call the helper, a bit of a workaround
        # This doesn't run generation, just gives us access to the method
        wm = WorkflowManager(self) 

        # Choose the correct default prompt builder for the active architecture
        is_klein = getattr(context.scene, 'model_architecture', '') == 'flux2_klein'
        get_defaults = wm._get_klein_default_prompts if is_klein else wm._get_qwen_default_prompts

        def built_in_prompt(is_initial, context_mode=None):
            if is_klein:
                return get_defaults(context, is_initial_image=is_initial)
            selected = context.scene.qwen_prompt_template
            if selected == 'CUSTOM':
                selected = (
                    'QWEN_21_DEFAULT'
                    if context.scene.model_architecture == 'qwen_image_21'
                    else 'QWEN_EDIT_LEGACY'
                )
            return get_defaults(
                context,
                is_initial_image=is_initial,
                template_id=selected,
                context_mode_override=context_mode,
            )

        # Determine which prompt to reset
        if self.prompt_type == 'initial':
            default_prompt = built_in_prompt(True)
            context.scene.qwen_custom_prompt_initial = default_prompt
        elif self.prompt_type == 'seq_none':
            default_prompt = built_in_prompt(False, 'NONE')
            context.scene.qwen_custom_prompt_seq_none = default_prompt
        elif self.prompt_type == 'seq_replace':
            default_prompt = built_in_prompt(False, 'REPLACE_STYLE')
            context.scene.qwen_custom_prompt_seq_replace = default_prompt
        elif self.prompt_type == 'seq_additional':
            default_prompt = built_in_prompt(False, 'ADDITIONAL')
            context.scene.qwen_custom_prompt_seq_additional = default_prompt
        
        self.report({'INFO'}, "Prompt reset to default.")
        return {'FINISHED'}


class ResetTexturingExperiments(bpy.types.Operator):
    """Reset research settings without changing graduated controls."""

    bl_idname = "stablegen.reset_texturing_experiments"
    bl_label = "Disable All Texturing Experiments"
    bl_description = "Reset experimental texturing settings; keep regular features and their values"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for name, default in EXPERIMENTAL_FEATURE_DEFAULTS.items():
            if hasattr(context.scene, name):
                setattr(context.scene, name, default)
        self.report({'INFO'}, "Experimental settings reset; regular texturing settings retained")
        return {'FINISHED'}

class ApplyPreset(bpy.types.Operator):
    """Apply selected preset values to parameters"""
    bl_idname = "stablegen.apply_preset"
    bl_label = "Apply Preset"
    bl_description = "Set multiple parameters based on selected preset for easier configuration"

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        preset = context.scene.stablegen_preset
        if '(EXPERIMENTAL)' in preset and not experiments_enabled(context):
            self.report({'ERROR'}, "Enable experimental texturing in add-on preferences first")
            return {'CANCELLED'}
        if preset in PRESETS:
            values = PRESETS[preset]
            if (preset_uses_deprecated_architecture(values)
                    and not deprecated_architectures_enabled(context)):
                self.report({'ERROR'}, "Enable deprecated architectures in add-on preferences first; these architectures no longer receive support")
                return {'CANCELLED'}

            if not values.get("custom", False):
                for key, value in TEXTURING_DEFAULTS.items():
                    if hasattr(context.scene, key):
                        setattr(context.scene, key, value)
            
            # Apply architecture_mode first so dynamic enums that
            # depend on it (e.g. sequential_ipadapter_mode) are valid
            # when their values are set in the main loop.
            if "architecture_mode" in values and hasattr(context.scene, "architecture_mode"):
                try:
                    setattr(context.scene, "architecture_mode", values["architecture_mode"])
                except (TypeError, AttributeError):
                    pass

            # Apply regular parameters
            skipped = []
            for key, value in values.items():
                if key not in ["controlnet_units", "lora_units", "description"] and hasattr(context.scene, key):
                    try:
                        if key == "trellis2_decimate_method" and value == "planar":
                            setattr(context.scene, "trellis2_decimate_method", "none")
                            continue
                        if key == "trellis2_remesh":
                            if isinstance(value, bool):
                                legacy_str = "qdc" if value else "none"
                            else:
                                legacy_str = value
                            setattr(context.scene, "trellis2_remesh", legacy_str)
                            if legacy_str == "qdc":
                                setattr(context.scene, "trellis2_decimate_method", "server")
                                setattr(context.scene, "trellis2_remesh_method", "qdc")
                            elif legacy_str in ("none", "decimate_planar"):
                                setattr(context.scene, "trellis2_decimate_method", "none")
                                setattr(context.scene, "trellis2_remesh_method", "none")
                            elif legacy_str == "decimate_collapse":
                                setattr(context.scene, "trellis2_decimate_method", "collapse")
                                setattr(context.scene, "trellis2_remesh_method", "none")
                            elif legacy_str == "quadriflow":
                                setattr(context.scene, "trellis2_decimate_method", "none")
                                setattr(context.scene, "trellis2_remesh_method", "quadriflow")
                            elif legacy_str == "instant_meshes":
                                setattr(context.scene, "trellis2_decimate_method", "none")
                                setattr(context.scene, "trellis2_remesh_method", "quadriflow")
                            continue
                        setattr(context.scene, key, value)
                    except (TypeError, AttributeError):
                        # Dynamic enum values (e.g. trellis2_input) may not
                        # exist when the current architecture differs.
                        skipped.append(key)
            
            # Apply ControlNet units if present in the preset
            if "controlnet_units" in values:
                # Clear existing units
                context.scene.controlnet_units.clear()
                
                # Add new units from preset
                controlnet_units = values["controlnet_units"]
                for unit_data in controlnet_units:
                    new_unit = context.scene.controlnet_units.add()
                    for key, value in unit_data.items():
                        try:
                            setattr(new_unit, key, value)
                        except TypeError:
                            self.report({'ERROR'}, f"Failed to set {key} for ControlNet unit: {value}. Model might be missing or might not be named correctly.")
                            return {'CANCELLED'}
                        
            if "lora_units" in values:
                # Clear existing LoRA units
                context.scene.lora_units.clear()
                
                # Add new LoRA units from preset
                lora_units = values["lora_units"]
                for lora_data in lora_units:
                    new_lora = context.scene.lora_units.add()
                    for key, value in lora_data.items():
                        try:
                            setattr(new_lora, key, value)
                        except TypeError:
                            self.report({'ERROR'}, f"Failed to set {key} for LoRA unit: {value}. Model might be missing or might not be named correctly.")
                            context.scene.lora_units.remove(len(context.scene.lora_units) - 1)
                            return {'CANCELLED'}
                        
            # Reverse-sync: if the preset set model_architecture but didn't
            # include architecture_mode, update the visible dropdown to match.
            if "architecture_mode" not in values:
                arch = context.scene.model_architecture
                if arch in ('sdxl', 'flux1', 'qwen_image_edit', 'qwen_image_21', 'flux2_klein'):
                    context.scene.architecture_mode = arch

            if skipped:
                self.report({'WARNING'}, f"Preset '{preset}' applied (skipped {len(skipped)} incompatible setting(s)).")
            else:
                self.report({'INFO'}, f"Preset '{preset}' applied.")
        else:
            self.report({'INFO'}, "Custom preset active.")
        
        # Force update to ensure preset detection is correct after list changes
        update_parameters(self, context)

        return {'FINISHED'}

class SwitchToMeshGeneration(bpy.types.Operator):
    """Switch to mesh generation mode using TRELLIS.2"""
    bl_idname = "stablegen.switch_to_mesh_generation"
    bl_label = "Switch to Mesh Generation"
    bl_description = "Apply the TRELLIS.2 (Mesh Only) preset to generate 3D meshes"

    @classmethod
    def poll(cls, context):
        return (not sg_modal_active(context)
                and "TRELLIS.2 (MESH ONLY)" in PRESETS
                and getattr(context.scene, 'trellis2_available', False))

    def execute(self, context):
        context.scene.stablegen_preset = "TRELLIS.2 (MESH ONLY)"
        bpy.ops.stablegen.apply_preset()
        return {'FINISHED'}


class SavePreset(bpy.types.Operator):
    """Save the current parameter values as a custom preset"""
    bl_idname = "stablegen.save_preset"
    bl_label = "Save Custom Preset"

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    preset_name: bpy.props.StringProperty(
        name="Preset Name",
        default="MyPreset"
    ) # type: ignore

    preset_description: bpy.props.StringProperty(
        name="Description",
        description="A short description of what this preset is good for",
        default=""
    ) # type: ignore
    
    include_controlnet: bpy.props.BoolProperty(
        name="Include ControlNet Units",
        default=True,
        description="Include ControlNet units in the preset"
    ) # type: ignore

    include_loras: bpy.props.BoolProperty(
        name="Include LoRA Units",
        default=True
    ) # type: ignore

    def execute(self, context):
        scene = context.scene
        key = self.preset_name.upper()
        
        # Save all parameters defined in GEN_PARAMETERS
        PRESETS[key] = {param: getattr(scene, param) for param in GEN_PARAMETERS if hasattr(scene, param)}
        
        # Add description
        PRESETS[key]["description"] = self.preset_description

        # Add custom flag
        PRESETS[key]["custom"] = True

        if self.include_controlnet:
            # Save ControlNet units
            controlnet_units = []
            for unit in scene.controlnet_units:
                unit_data = {
                    "unit_type": unit.unit_type,
                    "model_name": unit.model_name,
                    "strength": unit.strength,
                    "start_percent": unit.start_percent,
                    "end_percent": unit.end_percent,
                    "is_union": unit.is_union,
                    "use_union_type": unit.use_union_type
                }
                controlnet_units.append(unit_data)
            
            # Add controlnet units to the preset
            PRESETS[key]["controlnet_units"] = controlnet_units

        if self.include_loras: # Save LoRA units
            lora_units_data = []
            for lora_unit in scene.lora_units:
                lora_units_data.append({
                    "model_name": lora_unit.model_name,
                    "model_strength": lora_unit.model_strength,
                    "clip_strength": lora_unit.clip_strength,
                })

            # Add LoRA units to the preset
            PRESETS[key]["lora_units"] = lora_units_data
        
        scene.stablegen_preset = key
        scene.active_preset = key
        self.report({'INFO'}, f"Preset '{self.preset_name}' saved.")
        
        # Print in the console for debugging
        print(f'"{key}": {{', end="")
        print(f'"description": "{self.preset_description}", ', end="")
        for param in GEN_PARAMETERS:
            if hasattr(scene, param):
                value = getattr(scene, param)
                if isinstance(value, str):
                    print(f'"{param}": "{value}", ', end="")
                else:
                    print(f'"{param}": {value}, ', end="")
        
        # Print controlnet units in a compact format if included
        if self.include_controlnet:
            print(f'"controlnet_units": {controlnet_units},', end="")
        print("},")

        # Print LoRA units in a compact format if included
        if self.include_loras:
            print(f'"lora_units": {lora_units_data},', end="")
        print("},")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preset_name")
        layout.prop(self, "preset_description")
        layout.prop(self, "include_controlnet")
        layout.prop(self, "include_loras")

class DeletePreset(bpy.types.Operator):
    """Delete a custom preset"""
    bl_idname = "stablegen.delete_preset"
    bl_label = "Delete Preset"

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        preset = context.scene.stablegen_preset
        if preset in PRESETS:
            del PRESETS[preset]
            context.scene.stablegen_preset = "CUSTOM"
            self.report({'INFO'}, f"Preset '{preset}' deleted.")
            update_parameters(self, context)
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "Preset not found.")
            return {'CANCELLED'}


# =====================================================================
# Scene Queue System
# =====================================================================
# Allows users to queue multiple Blender scenes for unattended batch
# processing.  Each queued scene is processed in order — the manager
# switches to the scene, invokes the appropriate generation operator
# (TRELLIS.2 or standard texturing), waits for completion, optionally
# saves a .blend copy, then advances to the next item.
