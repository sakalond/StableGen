""" This file contains the operators and panels for the StableGen addon """
# disable import-error because pylint doesn't recognize the blenders internal modules
import os
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
import math  # pylint: disable=import-error

# Stock presets
PRESETS = {
    "DEFAULT": {"description": "Default settings for general purpose generation", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)),  "discard_factor": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "hm-mvgd-hm", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.7, "sequential_factor_smooth": 0.15, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "MODEL IS IMPORTANT": {"description": "Same as default, but is more guided by the model", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.7, "sequential_factor_smooth": 0.15, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.75, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "CHARACTERS": {"description": "Optimized settings for character generation", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.7, "sequential_factor_smooth": 0.1, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "CHARACTERS (ALTERNATIVE MASKING)": {"description": "Optimized for character generation. Uses alternative masking parameters to be more consistent between images, but may produce more artifacts. Try if \"Characters\" fails.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.5, "sequential_factor_smooth": 0.3499999940395355, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 10, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "QUICK DRAFT": {"description": "Optimized for speed", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 4, "cfg": 1.0, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "grid", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.7, "sequential_factor_smooth": 0.1, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 2, "blur_mask_sigma": 1.0, "grow_mask_by": 2, "canny_threshold_low": 0, "canny_threshold_high": 80, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'Hyper-SDXL-4steps-lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "UV INPAINTING": {"description": "Recommended UV Inpainting setup. It is recommended to bake texutures manually before running the generation to fine-tune unwrapping and avoid lag when generating.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 10, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "uv_inpaint", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "3,0,1,2", "sequential_factor": 0.6000000238418579, "sequential_factor_smooth": 0.11000001430511475, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 3, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]}, # No ControlNet for UV Inpainting by default
    "ARCHITECTURE": {"description": "Prioritizes only the most straight-on camera for each point. This means details generated on flat surfaces will not get blurred by getting generated differently from two or more viewpoints. Does not use visibility masking. Each picture will get generated as new, consistency depends on IPAdapter + geometry.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "weight_exponent": 10.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "separate", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": False, "sequential_custom_camera_order": "", "sequential_factor": 0.75, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "ipadapter_weight_type": "style", "ipadapter_strength": 0.800000011920929, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": False, "blur_mask": True, "blur_mask_radius": 3, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.6000000238418579, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "QWEN PRECISE": {"description": "Precise detail when camera overlap is good. Relies on context renders plus the prompt, so sparse coverage can still introduce artifacts.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit","qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": True, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "QWEN SAFE": {"description": "Safer fallback when coverage is limited. Uses the previous view (recent mode) instead of context renders to keep global look coherent, at the cost of some fine-detail persistence.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 6.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": True, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "QWEN ALT": {"description": "Balanced option that mixes additional context renders with sequential references to smooth out coverage while keeping detail reasonable.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "recent", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "ADDITIONAL", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": True, "early_priority": True, "early_priority_strength": 0.5, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},     
    "QWEN VORONOI": {"description": "Voronoi projection mode with exponent 1000 for hard camera segmentation during generation, then resets to 15 for softer blending. Each surface point is dominated by its closest camera. Based on Qwen Precise.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 1000.0, "weight_exponent_generation_only": True, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": True, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": True, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "QWEN PRECISE (NUNCHAKU)": {"description": "Precise detail using Nunchaku. Meant to be used with the Nunchaku model which has the 4-step Lightning LoRA included. Requires Nunchaku nodes.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": True, "lora_units": []},
    "QWEN SAFE (NUNCHAKU)": {"description": "Safer fallback using Nunchaku. Meant to be used with the Nunchaku model which has the 4-step Lightning LoRA included. Requires Nunchaku nodes.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 6.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "NONE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": True, "lora_units": []},
    "QWEN ALT (NUNCHAKU)": {"description": "Balanced option using Nunchaku. Meant to be used with the Nunchaku model which has the 4-step Lightning LoRA included. Requires Nunchaku nodes.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": False, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "recent", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "ADDITIONAL", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": True, "early_priority": True, "early_priority_strength": 0.5, "lora_units": []},
    "QWEN VORONOI (NUNCHAKU)": {"description": "Voronoi projection mode using Nunchaku. Exponent 1000 for hard camera segmentation during generation, then resets to 15 for softer blending. Requires Nunchaku nodes.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "generate", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 1000.0, "weight_exponent_generation_only": True, "weight_exponent_after_generation": 15.0, "qwen_voronoi_mode": True, "clip_skip": 1, "auto_rescale": True, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.8, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": True, "lora_units": []},
    "LOCAL REFINE": {"description": "Uses the SDXL local edit mode to improve detail / refine specific areas. Use new set of cameras or a single camera pointed at the area you want to refine.", "control_after_generate": "fixed", "model_architecture": "sdxl", "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2s_ancestral", "scheduler": "sgm_uniform", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 80.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "local_edit", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 0.800000011920929, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": True, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.10000000149011612, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": True, "sequential_ipadapter_mode": "original_render", "sequential_desaturate_factor": 0.10000000149011612, "sequential_contrast_factor": 0.10000000149011612, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": True, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 5.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": True, "refine_angle_ramp_active": True, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05, "visibility_vignette": True, "visibility_vignette_width": 0.3, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6, "refine_edge_feather_projection": True, "refine_edge_feather_width": 30, "refine_edge_feather_softness": 1.0, "controlnet_units": [{'unit_type': 'depth', 'model_name': 'controlnet_depth_sdxl.safetensors', 'strength': 0.5, 'start_percent': 0.0, 'end_percent': 1.0, 'is_union': False, 'use_union_type': True}], "lora_units": [{'model_name': 'sdxl_lightning_8step_lora.safetensors', 'model_strength': 1.0, 'clip_strength': 1.0}]},
    "LOCAL EDIT (QWEN)": {"description": "Uses Qwen to make targeted edits to specific areas. Point cameras at what you want to change — you can alter colors, style, rewrite text, add details, and more. Untouched areas are preserved.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "local_edit", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 1.0, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": False, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": False, "refine_angle_ramp_active": False, "refine_angle_ramp_pos_0": 0.0, "refine_angle_ramp_pos_1": 0.05, "visibility_vignette": True, "visibility_vignette_width": 0.1, "visibility_vignette_softness": 1.0, "visibility_vignette_blur": False, "refine_feather_ramp_pos_0": 0.0, "refine_feather_ramp_pos_1": 0.6, "refine_edge_feather_projection": True, "refine_edge_feather_width": 15, "refine_edge_feather_softness": 1.0, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
    "REFINE (QWEN)": {"description": "Uses Qwen to restyle or globally modify the entire existing texture. Applies changes uniformly across all camera views — ideal for changing the overall color scheme, art style, or surface appearance.", "control_after_generate": "fixed", "model_architecture": "qwen_image_edit", "qwen_generation_method": "refine", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "fallback_color": mathutils.Color((0.0, 0.0, 0.0)), "discard_factor": 70.0, "discard_factor_generation_only": True, "discard_factor_after_generation": 90.0, "weight_exponent": 3.0, "weight_exponent_generation_only": False, "weight_exponent_after_generation": 15.0, "view_blend_use_color_match": False, "view_blend_color_match_method": "reinhard", "view_blend_color_match_strength": 1.0, "clip_skip": 1, "auto_rescale": True, "auto_rescale_target_mp": 1.0, "overwrite_material": True, "generation_method": "sequential", "refine_images": False, "refine_steps": 8, "refine_sampler": "dpmpp_2s_ancestral", "refine_scheduler": "sgm_uniform", "denoise": 1.0, "refine_cfg": 1.5, "refine_prompt": "", "refine_upscale_method": "lanczos", "sequential_smooth": False, "sequential_custom_camera_order": "", "sequential_factor": 0.699999988079071, "sequential_factor_smooth": 0.15000000596046448, "sequential_factor_smooth_2": 1.0, "sequential_ipadapter": False, "sequential_ipadapter_mode": "first", "sequential_desaturate_factor": 0.0, "sequential_contrast_factor": 0.0, "sequential_ipadapter_regenerate": False, "ipadapter_weight_type": "style", "ipadapter_strength": 1.0, "ipadapter_start": 0.0, "ipadapter_end": 1.0, "early_priority": False, "early_priority_strength": 0.5, "differential_diffusion": True, "differential_noise": True, "blur_mask": True, "blur_mask_radius": 1, "blur_mask_sigma": 1.0, "grow_mask_by": 3, "canny_threshold_low": 0, "canny_threshold_high": 80, "qwen_guidance_map_type": "depth", "qwen_voronoi_mode": False, "qwen_use_external_style_image": False, "qwen_external_style_image": "", "qwen_context_render_mode": "REPLACE_STYLE", "qwen_external_style_initial_only": False, "qwen_use_custom_prompts": False, "qwen_custom_prompt_initial": "Change the format of image 1 to '{main_prompt}'", "qwen_custom_prompt_seq_none": "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2", "qwen_custom_prompt_seq_replace": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.", "qwen_custom_prompt_seq_additional": "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.", "qwen_guidance_fallback_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_guidance_background_color": mathutils.Color((1.0, 0.0, 1.0)), "qwen_context_cleanup": False, "qwen_context_cleanup_hue_tolerance": 0.0, "qwen_context_cleanup_value_adjust": 0.0, "qwen_context_fallback_dilation": 1, "qwen_rescale_alignment": False, "lora_units": [{'model_name': 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors', 'model_strength': 1.0, 'clip_strength': 0.0}]},
}

# Global list of all generation parameter names to check for a preset.
GEN_PARAMETERS = [
    "control_after_generate",
    "model_architecture",
    "steps",
    "cfg",
    "sampler",
    "scheduler",
    "fallback_color",
    "discard_factor",
    "discard_factor_generation_only",
    "discard_factor_after_generation",
    "weight_exponent",
    "weight_exponent_generation_only",
    "weight_exponent_after_generation",
    "view_blend_use_color_match",
    "view_blend_color_match_method",
    "view_blend_color_match_strength",
    "clip_skip",
    "auto_rescale",
    "auto_rescale_target_mp",
    "overwrite_material",
    "generation_method", 
    "refine_images",
    "refine_steps",
    "refine_sampler",
    "refine_scheduler",
    "denoise",
    "refine_cfg",
    "refine_prompt",
    "refine_upscale_method",
    "sequential_smooth",
    "sequential_custom_camera_order",
    "sequential_factor",
    "sequential_factor_smooth",
    "sequential_factor_smooth_2",
    "sequential_ipadapter",
    "sequential_ipadapter_mode",
    "sequential_desaturate_factor",
    "sequential_contrast_factor",
    "sequential_ipadapter_regenerate",
    "ipadapter_weight_type",
    "ipadapter_strength",
    "ipadapter_start",
    "ipadapter_end",
    "early_priority",
    "early_priority_strength",
    "differential_diffusion",
    "differential_noise",
    "blur_mask",
    "blur_mask_radius",
    "blur_mask_sigma",
    "grow_mask_by",
    "canny_threshold_low",
    "canny_threshold_high",
    "qwen_guidance_map_type",
    "qwen_voronoi_mode",
    "qwen_use_external_style_image",
    "qwen_external_style_image",
    "qwen_context_render_mode",
    "qwen_external_style_initial_only",
    "qwen_use_custom_prompts",
    "qwen_custom_prompt_initial",
    "qwen_custom_prompt_seq_none",
    "qwen_custom_prompt_seq_replace",
    "qwen_custom_prompt_seq_additional",
    "qwen_guidance_fallback_color",
    "qwen_guidance_background_color",
    "qwen_context_cleanup",
    "qwen_context_cleanup_hue_tolerance",
    "qwen_context_cleanup_value_adjust",
    "qwen_context_fallback_dilation",
    "qwen_rescale_alignment",
    "qwen_generation_method",
    "qwen_refine_use_prev_ref",
    "qwen_refine_use_depth",
    "qwen_timestep_zero_ref",
    "refine_angle_ramp_active",
    "refine_angle_ramp_pos_0",
    "refine_angle_ramp_pos_1",
    "visibility_vignette",
    "visibility_vignette_width",
    "visibility_vignette_softness",
    "visibility_vignette_blur",
    "refine_feather_ramp_pos_0",
    "refine_feather_ramp_pos_1",
    "refine_edge_feather_projection",
    "refine_edge_feather_width",
    "refine_edge_feather_softness",
]

def get_preset_items(self, context):
    items = []
    for key in PRESETS.keys():
        description = PRESETS[key].get("description", f"Preset {key}")
        items.append((key, key.title(), description))
    items.append(("CUSTOM", "Custom", "Custom configuration"))
    return items

def update_parameters(self, context):
    scene = context.scene
    # Build a dictionary of current parameter values
    current = {key: getattr(scene, key) for key in GEN_PARAMETERS if hasattr(scene, key)}
    
    # Compare current values with every stock preset's stored values
    for name, preset in PRESETS.items():
        # First check regular parameters
        if all(
            (lambda v1, v2: math.isclose(v1, v2, rel_tol=1e-7, abs_tol=0.0) if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else v1 == v2)
            (current.get(key), preset.get(key))
            for key in GEN_PARAMETERS if key in preset
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

class StableGenPanel(bpy.types.Panel):
    """     
    Creates the main UI panel for the StableGen addon.     
    """
    bl_label = "StableGen"
    bl_idname = "OBJECT_PT_stablegen"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "StableGen"
    bl_context = "objectmode"
    bl_ui_units_x = 600

    def draw_header(self, _):
        """     
        Draws the header of the panel.         
        :param _: Unused parameter.         
        :return: None     
        """
        self.layout.label(icon="WORLD_DATA")

    def draw(self, context):
        """     
        Draws the panel with reorganized Advanced Parameters.         
        :param context: Blender context.         
        :return: None     
        """
        layout = self.layout
        scene = context.scene # Get the scene for easier access

        # Detect the current width of the panel
        region = context.region
        width = region.width
        width_mode = 'narrow' if width < 420 else 'wide'
        
         # --- Action Buttons & Progress ---
        cam_tools_row = layout.row()
        cam_tools_row.operator("object.add_cameras", text="Add Cameras", icon="CAMERA_DATA")
        if width_mode == 'narrow':
            cam_tools_row = layout.row() 
        cam_tools_row.operator("object.collect_camera_prompts", text="Collect Camera Prompts", icon="FILE_TEXT")

        cam_extra_row = layout.row(align=True)
        cam_extra_row.operator("object.clone_camera", text="Clone Camera", icon="DUPLICATE")
        cam_extra_row.operator("object.mirror_camera", text="Mirror", icon="MOD_MIRROR")
        cam_extra_row.operator("object.toggle_camera_labels", text="Labels", icon="FONT_DATA")
        

        addon_prefs = context.preferences.addons[__package__].preferences
        config_error_message = None

        if not os.path.exists(addon_prefs.output_dir):
            config_error_message = "Output Path Invalid"
        elif not addon_prefs.server_address:
            config_error_message = "Server Address Missing"
        elif not addon_prefs.server_online:
            config_error_message = "Cannot reach server"

        # Determine if we are in TRELLIS.2 mode (used by generate button & later sections)
        _arch_mode = getattr(scene, 'architecture_mode', 'sdxl')
        _is_trellis2_mode = (_arch_mode == 'trellis2')

        action_row = layout.row()
        action_row.scale_y = 2.0 # Scale the row vertically

        if _is_trellis2_mode:
            # --- TRELLIS.2 Generate Button ---
            trellis2_op = next(
                (op for win in context.window_manager.windows
                 for op in win.modal_operators
                 if op.bl_idname == 'OBJECT_OT_trellis2_generate'),
                None
            )
            # Also look for ComfyUIGenerate running as the texturing phase
            comfy_tex_op = next(
                (op for win in context.window_manager.windows
                 for op in win.modal_operators
                 if op.bl_idname == 'OBJECT_OT_test_stable'),
                None
            ) if not trellis2_op else None

            if config_error_message:
                if config_error_message == "Cannot reach server":
                    split = action_row.split(factor=0.85, align=True)
                    split.operator("object.trellis2_generate", text="Cannot generate: " + config_error_message, icon="ERROR")
                    split.operator("stablegen.check_server_status", text="", icon='FILE_REFRESH')
                else:
                    action_row.operator("object.trellis2_generate", text="Cannot generate: " + config_error_message, icon="ERROR")
                    action_row.enabled = False

            elif trellis2_op:
                # ── Phases 1 & 2: Trellis2Generate is alive ──
                action_row.operator("object.trellis2_generate", text="Cancel TRELLIS.2", icon="CANCEL")
                progress_col = layout.column()

                # Bar 1 — Overall
                overall_pct = getattr(trellis2_op, '_overall_progress', 0)
                overall_label = getattr(trellis2_op, '_overall_stage', 'Initializing')
                progress_col.progress(
                    text=f"{overall_label} ({overall_pct:.0f}%)",
                    factor=max(0.0, min(overall_pct / 100.0, 1.0))
                )

                # Bar 2 — Phase
                phase_pct = getattr(trellis2_op, '_phase_progress', 0)
                phase_label = getattr(trellis2_op, '_phase_stage', '')
                if phase_label:
                    progress_col.progress(
                        text=f"{phase_label} ({phase_pct:.0f}%)",
                        factor=max(0.0, min(phase_pct / 100.0, 1.0))
                    )

                # Bar 3 — Detail (only when there's actual sampler progress)
                detail_pct = getattr(trellis2_op, '_detail_progress', 0)
                detail_label = getattr(trellis2_op, '_detail_stage', '')
                if detail_label and detail_pct > 0:
                    progress_col.progress(
                        text=f"{detail_label} ({detail_pct:.0f}%)",
                        factor=max(0.0, min(detail_pct / 100.0, 1.0))
                    )

            elif comfy_tex_op and getattr(scene, 'trellis2_pipeline_active', False):
                # ── Phase 3: Texturing via ComfyUIGenerate ──
                action_row.operator("object.test_stable", text="Cancel Texturing", icon="CANCEL")
                progress_col = layout.column()

                # Compute overall from scene pipeline props + ComfyUI progress
                phase_start = getattr(scene, 'trellis2_pipeline_phase_start_pct', 65.0)
                phase_weight = 100.0 - phase_start
                comfy_progress = getattr(comfy_tex_op, '_progress', 0) / 100.0
                total_imgs = getattr(comfy_tex_op, '_total_images', 0)
                cur_img_idx = getattr(comfy_tex_op, '_current_image', 0)
                if total_imgs > 1:
                    comfy_overall = (cur_img_idx + comfy_progress) / total_imgs
                else:
                    comfy_overall = comfy_progress
                overall_pct = phase_start + comfy_overall * phase_weight
                overall_pct = max(0.0, min(overall_pct, 100.0))

                total_phases = getattr(scene, 'trellis2_pipeline_total_phases', 3)
                # Bar 1 — Overall
                progress_col.progress(
                    text=f"Phase {total_phases}/{total_phases}: Texturing ({overall_pct:.0f}%)",
                    factor=max(0.0, min(overall_pct / 100.0, 1.0))
                )

                # Bar 2 — Per-image (same as normal ComfyUI bar)
                stage = getattr(comfy_tex_op, '_stage', 'Generating')
                img_pct = getattr(comfy_tex_op, '_progress', 0)
                progress_col.progress(
                    text=f"{stage} ({img_pct:.0f}%)",
                    factor=max(0.0, min(img_pct / 100.0, 1.0))
                )

                # Bar 3 — Image N/M (same as normal ComfyUI overall bar)
                if total_imgs > 1:
                    cur_img = min(cur_img_idx + 1, total_imgs)
                    img_overall = max(0.0, min(comfy_overall, 1.0))
                    progress_col.progress(
                        text=f"Overall: Image {cur_img}/{total_imgs}",
                        factor=img_overall
                    )

            elif scene.trellis2_generate_from == 'image' and not scene.trellis2_input_image:
                action_row.operator("object.trellis2_generate", text="Select an image first", icon="ERROR")
                action_row.enabled = False
            elif not getattr(scene, 'trellis2_available', False):
                action_row.operator("object.trellis2_generate", text="TRELLIS.2 nodes not found on server", icon="ERROR")
                action_row.enabled = False
            else:
                action_row.operator("object.trellis2_generate", text="Generate 3D Mesh", icon="MESH_ICOSPHERE")
        else:
            # --- Standard Diffusion Generate Button ---
            if config_error_message:
                # Split the row to have the error message/disabled button and the refresh button
                if config_error_message == "Cannot reach server":
                    split = action_row.split(factor=0.85, align=True) # Adjust factor as needed
                    split.operator("object.test_stable", text="Cannot generate: " + config_error_message, icon="ERROR") # Use ERROR icon
                    # Use the operator from __init__.py
                    split.operator("stablegen.check_server_status", text="", icon='FILE_REFRESH')
                else:
                    action_row.operator("object.test_stable", text="Cannot generate: " + config_error_message, icon="ERROR")
                    action_row.enabled = False
            else:
                action_row.enabled = True
                if not bpy.app.online_access:
                    action_row.operator("object.test_stable", text="Enable online access in preferences", icon="ERROR")
                    action_row.enabled = False
                elif not scene.model_name or scene.model_name == "NONE_FOUND":
                    action_row.operator("object.test_stable", text="Cannot generate: Model Directory Empty", icon="ERROR")
                    action_row.enabled = False
                elif scene.generation_status == 'idle':
                    # Check if any cameras are selected and if there is existing output
                    selected_cameras = [obj for obj in context.selected_objects if obj.type == 'CAMERA']
                    if not selected_cameras or scene.get("output_timestamp") == "":
                        action_row.operator("object.test_stable", text="Generate", icon="PLAY")
                    else:
                        # Use the regenerate operator
                        action_row.operator("object.stablegen_regenerate", text="Regenerate Selected Views", icon="PLAY")
                elif scene.generation_status == 'running':
                    action_row.operator("object.test_stable", text="Cancel Generation", icon="CANCEL")

                    operator_instance = next((op for win in context.window_manager.windows for op in win.modal_operators if op.bl_idname == 'OBJECT_OT_test_stable'), None)
                    if operator_instance:
                        progress_col = layout.column()
                        progress_text = f"{getattr(operator_instance, '_stage', 'Generating')} ({getattr(operator_instance, '_progress', 0):.0f}%)"
                        progress_factor = getattr(operator_instance, '_progress', 0) / 100.0
                        progress_col.progress(text=progress_text, factor=max(0.0, min(progress_factor, 1.0))) # Ensure factor is <= 1.0 (logic maintained)

                        total_images = getattr(operator_instance, '_total_images', 0)
                        if total_images > 1:
                            current_image_idx = getattr(operator_instance, '_current_image', 0)
                            current_image_decimal_progress = max(0.0, min(progress_factor, 1.0))
                            
                            # Ensure total_images is not zero to prevent division by zero
                            overall_progress_factor = (current_image_idx + current_image_decimal_progress) / total_images if total_images > 0 else 0
                            overall_progress_factor_clamped = max(0.0, min(overall_progress_factor, 1.0))

                            current_img = min(current_image_idx + 1, total_images)  # Clamp to total_images

                            progress_col.progress(
                                text=f"Overall: Image {current_img}/{total_images}",
                                factor=overall_progress_factor_clamped # Ensure factor is <= 1.0 (logic maintained)
                            )
                            
                elif context.scene.generation_status == 'waiting':
                    action_row.operator("object.test_stable", text="Waiting for Cancellation", icon="TIME")
                else:
                    action_row.operator("object.test_stable", text="Fix Issues to Generate", icon="ERROR")
                    action_row.enabled = False
        
        bake_row = layout.row()
        if config_error_message:
            bake_row.operator("object.bake_textures", text="Cannot Bake: " + config_error_message, icon="ERROR")
            bake_row.enabled = False
        else:
            bake_row.operator("object.bake_textures", text="Bake Textures", icon="RENDER_STILL")
            bake_row.enabled = True
        bake_operator = next((op for win in context.window_manager.windows for op in win.modal_operators if op.bl_idname == 'OBJECT_OT_bake_textures'), None)
        if bake_operator:
            bake_progress_col = layout.column()
            bake_stage = getattr(bake_operator, '_stage', 'Baking')
            bake_progress = getattr(bake_operator, '_progress', 0) / 100.0
            bake_progress_col.progress(text=bake_stage, factor=bake_progress if bake_progress <=1.0 else 1.0) # Ensure factor is <= 1.0
            
            total_objects = getattr(bake_operator, '_total_objects', 0)
            if total_objects > 1:
                current_object = getattr(bake_operator, '_current_object', 0)
                # Ensure total_objects is not zero
                overall_bake_progress = ((current_object + bake_progress) / total_objects) if total_objects > 0 else 0
                bake_progress_col.progress(
                    text=f"{bake_stage}: Object {current_object + 1}/{total_objects}",
                    factor=overall_bake_progress if overall_bake_progress <=1.0 else 1.0 # Ensure factor is <= 1.0
                )

        # --- Preset Management ---
        preset_box = layout.box()
        row = preset_box.row(align=True)
        row.prop(scene, "stablegen_preset", text="Preset")
        
        # Conditional button: Apply for stock presets, Save for custom preset
        if not hasattr(scene, 'active_preset'):
            scene.active_preset = scene.stablegen_preset

        if scene.stablegen_preset == "CUSTOM":
            row.operator("stablegen.save_preset", text="Save Preset", icon="PLUS")
        else:
            if scene.active_preset != scene.stablegen_preset:
                row.operator("stablegen.apply_preset", text="Apply Preset", icon="CHECKMARK")
            
            is_stock_preset = PRESETS.get(scene.stablegen_preset, {}).get("custom", False) is False
            if not is_stock_preset and scene.stablegen_preset != "DEFAULT": 
                 row.operator("stablegen.delete_preset", text="Delete", icon="TRASH")


        # --- Main Parameters section ---
        if not hasattr(scene, 'show_generation_params'): 
            scene.show_generation_params = True

        is_trellis2 = getattr(scene, 'architecture_mode', 'sdxl') == 'trellis2'
        trellis2_tex_mode = getattr(scene, 'trellis2_texture_mode', 'native')
        trellis2_diffusion_texturing = is_trellis2 and trellis2_tex_mode in ('sdxl', 'flux1', 'qwen_image_edit')
            
        main_params_box = layout.box()
        main_params_col = main_params_box.column()
        main_params_col.prop(scene, "show_generation_params", text="Main Parameters", icon="TRIA_DOWN" if scene.show_generation_params else "TRIA_RIGHT", emboss=False)
        if scene.show_generation_params:
            params_container = main_params_col.box()
            # Split for prompt
            split = params_container.split(factor=0.25)
            split.label(text="Prompt:")
            split.prop(scene, "comfyui_prompt", text="")

            # Architecture selector (architecture_mode — includes TRELLIS.2)
            split = params_container.split(factor=0.5)
            split.label(text="Architecture:")
            split.prop(scene, "architecture_mode", text="")

            if is_trellis2:
                # --- TRELLIS.2 specific layout ---
                # Warning if TRELLIS.2 nodes not detected
                if not getattr(scene, 'trellis2_available', False):
                    warn_row = params_container.row()
                    warn_row.alert = True
                    warn_row.label(text="TRELLIS.2 nodes not detected on server", icon="ERROR")

                # Generate From toggle (Image / Prompt)
                split = params_container.split(factor=0.5)
                split.label(text="Generate From:")
                split.prop(scene, "trellis2_generate_from", text="")

                # Input image picker (only when generate_from = image)
                if scene.trellis2_generate_from == 'image':
                    split = params_container.split(factor=0.25)
                    split.label(text="Input Image:")
                    split.prop(scene, "trellis2_input_image", text="")

                # Texture Generation Mode
                split = params_container.split(factor=0.5)
                split.label(text="Texture Mode:")
                split.prop(scene, "trellis2_texture_mode", text="")

                # When diffusion texturing: show checkpoint, generation mode, camera count
                if trellis2_diffusion_texturing:
                    split = params_container.split(factor=0.25)
                    split.label(text="Checkpoint:")
                    row = split.row(align=True)
                    row.prop(scene, "model_name", text="")
                    row.operator("stablegen.refresh_checkpoint_list", text="", icon='FILE_REFRESH')

                    split = params_container.split(factor=0.5)
                    split.label(text="Generation Mode:")
                    if scene.model_architecture.startswith("qwen"):
                        split.prop(scene, "qwen_generation_method", text="")
                    else:
                        split.prop(scene, "generation_method", text="")

                    row = params_container.row()
                    row.prop(scene, "trellis2_camera_count", text="Camera Count")
            else:
                # --- Standard diffusion layout ---
                # Split for model name
                split = params_container.split(factor=0.25)
                split.label(text="Checkpoint:")
                row = split.row(align=True)
                row.prop(scene, "model_name", text="")
                row.operator("stablegen.refresh_checkpoint_list", text="", icon='FILE_REFRESH')

                # Split for generation method
                split = params_container.split(factor=0.5)
                split.label(text="Generation Mode:")
                if scene.model_architecture.startswith("qwen"):
                    split.prop(scene, "qwen_generation_method", text="")
                else:
                    split.prop(scene, "generation_method", text="")

                # Split for object selection
                split = params_container.split(factor=0.5)
                split.label(text="Target Objects:")
                split.prop(scene, "texture_objects", text="")

        # Helper to create collapsible sections
        def draw_collapsible_section(parent_layout, toggle_prop_name, title, icon="NONE"):
            if not hasattr(scene, toggle_prop_name):
                setattr(bpy.types.Scene, toggle_prop_name, bpy.props.BoolProperty(name=title, default=False))

            box = parent_layout.box()
            col = box.column()
            is_expanded = getattr(scene, toggle_prop_name, False)
            col.prop(scene, toggle_prop_name, text=title, icon="TRIA_DOWN" if is_expanded else "TRIA_RIGHT", emboss=False)
            if is_expanded:
                return col.box() # Return a new box for content if expanded
            return None

        core_settings_props = [
            "show_core_settings", "show_lora_settings", "show_camera_options",
            "show_scene_understanding_settings", 
            "show_output_material_settings", "show_image_guidance_settings",
            "show_masking_inpainting_settings", "show_mode_specific_settings"
        ]
        for prop_name in core_settings_props:
            if not hasattr(scene, prop_name):
                setattr(bpy.types.Scene, prop_name, bpy.props.BoolProperty(name=prop_name.replace("_", " ").title(), default=False))

        # --- ADVANCED PARAMETERS ---
        advanced_params_box = layout.box()
        advanced_params_box = advanced_params_box.column()
        advanced_params_box.prop(scene, "show_advanced_params", text="Advanced Parameters", icon="TRIA_DOWN" if scene.show_advanced_params else "TRIA_RIGHT", emboss=False)
        if context.scene.show_advanced_params:

            # --- TRELLIS.2: Mesh Generation Settings ---
            if is_trellis2:
                content_box = draw_collapsible_section(advanced_params_box, "show_trellis2_mesh_settings", "Mesh Generation Settings", icon="MESH_DATA")
                if content_box:
                    # Core mesh params
                    row = content_box.row()
                    row.prop(scene, "trellis2_seed", text="Seed")

                    content_box.separator()

                    # Post-processing (ComfyUI-side decimation + remesh)
                    content_box.label(text="Post-Processing:", icon="OUTLINER_OB_MESH")
                    row = content_box.row()
                    row.prop(scene, "trellis2_post_processing_enabled", text="Enable Post-Processing", toggle=True, icon="MOD_DECIM")

                    if scene.trellis2_post_processing_enabled:
                        row = content_box.row()
                        row.prop(scene, "trellis2_decimation", text="Decimation Target")
                        row = content_box.row()
                        row.prop(scene, "trellis2_remesh", text="Remesh", toggle=True, icon="MOD_REMESH")
                    row = content_box.row()
                    row.prop(scene, "trellis2_fill_holes", text="Fill Holes", toggle=True, icon="MESH_GRID")

                    content_box.separator()

                    # Model settings
                    content_box.label(text="Model:", icon="SETTINGS")
                    split = content_box.split(factor=0.5)
                    split.label(text="Resolution:")
                    split.prop(scene, "trellis2_resolution", text="")

                    split = content_box.split(factor=0.5)
                    split.label(text="VRAM Mode:")
                    split.prop(scene, "trellis2_vram_mode", text="")

                    split = content_box.split(factor=0.5)
                    split.label(text="Attention:")
                    split.prop(scene, "trellis2_attn_backend", text="")

                    content_box.separator()

                    # Shape generation
                    content_box.label(text="Shape Generation:", icon="MESH_ICOSPHERE")
                    row = content_box.row(align=True)
                    row.prop(scene, "trellis2_ss_guidance", text="SS Guidance")
                    row.prop(scene, "trellis2_ss_steps", text="SS Steps")
                    row = content_box.row(align=True)
                    row.prop(scene, "trellis2_shape_guidance", text="Shape Guidance")
                    row.prop(scene, "trellis2_shape_steps", text="Shape Steps")
                    row = content_box.row()
                    row.prop(scene, "trellis2_max_tokens", text="Max Tokens (VRAM)")

                    content_box.separator()

                    # Conditioning
                    content_box.label(text="Conditioning:", icon="IMAGE_DATA")
                    split = content_box.split(factor=0.5)
                    split.label(text="Background:")
                    split.prop(scene, "trellis2_background_color", text="")
                    row = content_box.row()
                    row.prop(scene, "trellis2_include_1024", text="Include 1024 Conditioning", toggle=True, icon="IMAGE_PLANE")

                    content_box.separator()

                    # Misc
                    content_box.label(text="Misc:", icon="PREFERENCES")
                    row = content_box.row()
                    row.prop(scene, "trellis2_low_vram", text="Low VRAM Background Removal", toggle=True, icon="GHOST_ENABLED")

            # --- TRELLIS.2: Native Texture Settings ---
            if is_trellis2 and trellis2_tex_mode == 'native':
                content_box = draw_collapsible_section(advanced_params_box, "show_trellis2_texture_settings", "Texture Settings (TRELLIS.2 Native)", icon="TEXTURE")
                if content_box:
                    row = content_box.row(align=True)
                    row.prop(scene, "trellis2_tex_guidance", text="Tex Guidance")
                    row.prop(scene, "trellis2_tex_steps", text="Tex Steps")
                    row = content_box.row()
                    row.prop(scene, "trellis2_texture_size", text="Texture Size")

            # --- Diffusion-based advanced sections ---
            # Each is individually guarded: shown for standard arches or TRELLIS.2 with diffusion texturing
            _show_diffusion_sections = not is_trellis2 or trellis2_diffusion_texturing

            # --- Core Generation Settings ---
            
            if _show_diffusion_sections:
                content_box = draw_collapsible_section(advanced_params_box, "show_core_settings", "Core Generation Settings", icon="SETTINGS")
            else:
                content_box = None
            if content_box:
                row = content_box.row()
                row.prop(scene, "seed", text="Seed")
                if width_mode == 'narrow':
                    row = content_box.row()
                row.prop(scene, "steps", text="Steps")
                if width_mode == 'narrow':
                    row = content_box.row()
                row.prop(scene, "cfg", text="CFG")

                split = content_box.split(factor=0.5)
                split.label(text="Negative Prompt:")
                split.prop(scene, "comfyui_negative_prompt", text="")
                
                split = content_box.split(factor=0.5)
                split.label(text="Control After Generate:")
                split.prop(scene, "control_after_generate", text="")

                split = content_box.split(factor=0.5)
                split.label(text="Sampler:")
                split.prop(scene, "sampler", text="")

                split = content_box.split(factor=0.5)
                split.label(text="Scheduler:")
                split.prop(scene, "scheduler", text="")
                
                row = content_box.row()
                row.prop(scene, "clip_skip", text="Clip Skip")

           # --- LoRA Settings ---
            if _show_diffusion_sections:
                content_box = draw_collapsible_section(advanced_params_box, "show_lora_settings", "LoRA Management", icon="MODIFIER")
            else:
                content_box = None
            if content_box:
                row = content_box.row()
                row.alignment = 'CENTER'
                row.label(text="LoRA Units", icon="BRUSHES_ALL") # Using decimate icon for LoRA

                if scene.lora_units:
                    for i, lora_unit in enumerate(scene.lora_units):
                        is_selected_lora = (scene.lora_units_index == i)
                        unit_box = content_box.box()
                        row = unit_box.row()
                        row.prop(lora_unit, "model_name", text=f"LoRA {i+1}") # Shows selected model
                        
                        sub_row = unit_box.row(align=True)
                        sub_row.prop(lora_unit, "model_strength", text="Model Strength")
                        if not scene.model_architecture.startswith("qwen"): # Qwen uses model only loras
                            sub_row.prop(lora_unit, "clip_strength", text="CLIP Strength")

                        # Icon to indicate selection more clearly alongside the alert state
                        select_icon = 'CHECKBOX_HLT' if is_selected_lora else 'CHECKBOX_DEHLT'
                        
                        # Selection button (now more like a radio button)
                        op_select_lora = row.operator("wm.context_set_int", text="", icon=select_icon, emboss=True) # Keep emboss for the button itself
                        op_select_lora.data_path = "scene.lora_units_index"
                        op_select_lora.value = i

                btn_row_lora = content_box.row(align=True)

                if not scene.lora_units:
                    # Only one button if no LoRA units are present
                    button_text = "Add LoRA Unit" # Default text
                    
                    # Draw the operator with the dynamically determined text
                    btn_row_lora.operator("stablegen.add_lora_unit", text=button_text, icon="ADD")
                    # The enabled state (greying out) will be handled by AddLoRAUnit.poll()
                else:
                    # Multiple buttons if LoRA units exist
                    btn_row_lora.operator("stablegen.add_lora_unit", text="Add Another LoRA", icon="ADD")
                    btn_row_lora.operator("stablegen.remove_lora_unit", text="Remove Selected", icon="REMOVE")

            # --- Camera Options ---
            if _show_diffusion_sections:
                content_box = draw_collapsible_section(advanced_params_box, "show_camera_options", "Camera Settings", icon="CAMERA_DATA")
            else:
                content_box = None
            if content_box:
                row = content_box.row(align=True)
                row.prop(scene, "use_camera_prompts", text="Use Camera Prompts", toggle=True, icon="OUTLINER_OB_CAMERA")

                # ── Camera Generation Order ──
                row = content_box.row(align=True)
                row.prop(scene, "sg_use_custom_camera_order",
                         text="Custom Generation Order", toggle=True, icon="SORTALPHA")
                if scene.sg_use_custom_camera_order:
                    order_box = content_box.box()
                    # Preset strategy selector + sync
                    row = order_box.row(align=True)
                    row.operator_menu_enum(
                        "stablegen.apply_camera_order_preset", "strategy",
                        text="Sort Preset", icon="PRESET")
                    row.operator("stablegen.sync_camera_order",
                                 text="", icon="FILE_REFRESH")

                    # UIList with move buttons
                    row = order_box.row()
                    row.template_list(
                        "SG_UL_CameraOrderList", "",
                        scene, "sg_camera_order",
                        scene, "sg_camera_order_index",
                        rows=4, maxrows=8)
                    col = row.column(align=True)
                    op = col.operator("stablegen.move_camera_order",
                                      text="", icon="TRIA_UP")
                    op.direction = 'UP'
                    op = col.operator("stablegen.move_camera_order",
                                      text="", icon="TRIA_DOWN")
                    op.direction = 'DOWN'

            # --- Viewpoint Blending Settings ---
            if _show_diffusion_sections:
                content_box = draw_collapsible_section(advanced_params_box, "show_scene_understanding_settings", "Viewpoint Blending Settings", icon="ZOOM_IN")
            else:
                content_box = None
            if content_box:
                # Row 1: Discard-Over Angle | Weight Exponent
                split = content_box.split(factor=0.5, align=True)
                split.prop(scene, "discard_factor", text="Discard Angle")
                split.prop(scene, "weight_exponent", text="Exponent")

                # Row 2 & 3: Reset toggles and values (only for sequential / qwen)
                if scene.generation_method == 'sequential' or scene.model_architecture == 'qwen_image_edit':
                    split = content_box.split(factor=0.5, align=True)
                    split.prop(scene, "discard_factor_generation_only", text="Reset Angle", toggle=True)
                    split.prop(scene, "weight_exponent_generation_only", text="Reset Exponent", toggle=True)

                    if scene.discard_factor_generation_only or scene.weight_exponent_generation_only:
                        split = content_box.split(factor=0.5, align=True)
                        if scene.discard_factor_generation_only:
                            split.prop(scene, "discard_factor_after_generation", text="Angle After")
                        else:
                            split.label(text="")
                        if scene.weight_exponent_generation_only:
                            split.prop(scene, "weight_exponent_after_generation", text="Exp. After")
                        else:
                            split.label(text="")

                row = content_box.row()
                row.prop(scene, "early_priority", text="Prioritize Initial Views", toggle=True, icon="REW")
                if scene.early_priority:
                    row = content_box.row()
                    row.prop(scene, "early_priority_strength", text="Priority Strength")
                

            # --- Output & Material Settings ---
            if _show_diffusion_sections:
                content_box = draw_collapsible_section(advanced_params_box, "show_output_material_settings", "Output & Material Settings", icon="MATERIAL")
            else:
                content_box = None
            if content_box:
                split = content_box.split(factor=0.5)
                split.label(text="Fallback Color:")
                split.prop(scene, "fallback_color", text="")
                
               
                row = content_box.row()
                row.prop(scene, "apply_bsdf", text="Apply BSDF", toggle=True, icon="SHADING_RENDERED")
                
                row = content_box.row()
                row.prop(scene, "auto_rescale", text="Auto Rescale Resolution", toggle=True, icon="ARROW_LEFTRIGHT")
                if scene.auto_rescale:
                    sub_box_rescale = content_box.box()
                    row = sub_box_rescale.row()
                    row.prop(scene, "auto_rescale_target_mp", text="Target Megapixels")
                    if scene.model_architecture.startswith('qwen'):
                        row = sub_box_rescale.row()
                        row.prop(scene, "qwen_rescale_alignment", text="Qwen VL-Aligned Rescale (112px)", toggle=True, icon="SNAP_INCREMENT")
                row = content_box.row()
                row.prop(scene, "overwrite_material", text="Overwrite Material", toggle=True, icon="FILE_REFRESH")

            # --- Image Guidance (IPAdapter & ControlNet) ---
            if _show_diffusion_sections:
                if scene.model_architecture in ['sdxl', 'flux1']:
                    content_box = draw_collapsible_section(advanced_params_box, "show_image_guidance_settings", "Image Guidance (IPAdapter & ControlNet)", icon="MODIFIER")
                else: # Qwen Image Edit
                    content_box = draw_collapsible_section(advanced_params_box, "show_image_guidance_settings", "Qwen-Image-Edit Guidance", icon="MODIFIER")
            else:
                content_box = None
            if content_box:
                if scene.model_architecture == 'qwen_image_edit':
                    if scene.qwen_generation_method == 'generate':
                        split = content_box.split(factor=0.5)
                        split.label(text="Guidance Map:")
                        split.prop(scene, "qwen_guidance_map_type", text="")

                        row = content_box.row()
                        row.prop(scene, "qwen_use_external_style_image", text="Use External Image as Style", toggle=True, icon="FILE_IMAGE")

                        # TRELLIS.2 input as style (only in trellis2 architecture mode)
                        if getattr(scene, 'architecture_mode', '') == 'trellis2':
                            row = content_box.row()
                            row.prop(scene, "qwen_use_trellis2_style", text="Use TRELLIS.2 Input as Style", toggle=True, icon="MESH_MONKEY")

                        if scene.qwen_use_external_style_image:
                            style_box = content_box.box()
                            row = style_box.row()
                            row.prop(scene, "qwen_external_style_image", text="Style Image")
                            row = style_box.row()
                            row.prop(scene, "qwen_external_style_initial_only", text="External for Initial Only", toggle=True)

                        if scene.qwen_use_external_style_image and scene.qwen_external_style_initial_only:
                            subsequent_box = style_box.box()
                            split = subsequent_box.split(factor=0.5)
                            split.label(text="Subsequent mode:")
                            split.prop(scene, "sequential_ipadapter_mode", text="")
                            if scene.sequential_ipadapter_mode == 'recent':
                                subsequent_box.prop(scene, "sequential_desaturate_factor", text="Desaturate")
                                subsequent_box.prop(scene, "sequential_contrast_factor", text="Reduce Contrast")

                        if not scene.qwen_use_external_style_image and scene.generation_method in ['sequential', 'separate']:
                            row = content_box.row()
                            row.prop(scene, "sequential_ipadapter", text="Use Previous Image as Style", toggle=True, icon="MODIFIER")
                            if scene.sequential_ipadapter:
                                sub_ip_box = content_box.box()
                                split = sub_ip_box.split(factor=0.5)
                                split.label(text="Mode:")
                                split.prop(scene, "sequential_ipadapter_mode", text="")
                                if scene.sequential_ipadapter_mode == 'recent':
                                    sub_ip_box.prop(scene, "sequential_desaturate_factor", text="Desaturate")
                                    sub_ip_box.prop(scene, "sequential_contrast_factor", text="Reduce Contrast")

                        if scene.generation_method == 'sequential':
                            split = content_box.split(factor=0.5)
                            split.label(text="Context Render:")
                            split.prop(scene, "qwen_context_render_mode", text="")

                            row = content_box.row()
                            row.prop(scene, "qwen_voronoi_mode", text="Voronoi Projection", toggle=True, icon="MESH_GRID")
                    
                    elif scene.qwen_generation_method in ('refine', 'local_edit'):
                        row = content_box.row()
                        row.prop(scene, "qwen_refine_use_prev_ref", text="Use Previous Refined View", toggle=True)
                        
                        row = content_box.row()
                        row.prop(scene, "qwen_refine_use_depth", text="Use Depth Map", toggle=True, icon="MODIFIER")
                        
                        row = content_box.row()
                        row.prop(scene, "qwen_use_external_style_image", text="Use External Image as Style", toggle=True, icon="FILE_IMAGE")

                        # TRELLIS.2 input as style (only in trellis2 architecture mode)
                        if getattr(scene, 'architecture_mode', '') == 'trellis2':
                            row = content_box.row()
                            row.prop(scene, "qwen_use_trellis2_style", text="Use TRELLIS.2 Input as Style", toggle=True, icon="MESH_MONKEY")

                        if scene.qwen_use_external_style_image:
                            style_box = content_box.box()
                            row = style_box.row()
                            row.prop(scene, "qwen_external_style_image", text="Style Image")
                    
                    # Timestep-zero reference method — shared across all Qwen modes
                    row = content_box.row()
                    row.prop(scene, "qwen_timestep_zero_ref", text="Timestep-Zero References (color shift fix)", toggle=True, icon="COLORSET_08_VEC")

                    row = content_box.row()
                    row.prop(scene, "qwen_use_custom_prompts", text="Custom Guidance Prompts", toggle=True, icon="TEXT")
                    if scene.qwen_use_custom_prompts:
                        custom_prompt_box = content_box.box()
                        
                        # Initial Image Prompt
                        col = custom_prompt_box.column()
                        col.label(text="Initial Image Prompt:")
                        row = col.row(align=True)
                        row.prop(scene, "qwen_custom_prompt_initial", text="")
                        op = row.operator("stablegen.reset_qwen_prompt", text="", icon='FILE_REFRESH')
                        op.prompt_type = 'initial'

                        # Subsequent Images Prompt (conditional)
                        if scene.generation_method == 'sequential' or (scene.qwen_generation_method in ('refine', 'local_edit') and scene.qwen_refine_mode == 'sequential'):
                            col = custom_prompt_box.column()
                            col.label(text="Subsequent Images Prompt:")
                            row = col.row(align=True)
                            
                            if scene.qwen_context_render_mode == 'NONE' and scene.qwen_generation_method == 'generate':
                                row.prop(scene, "qwen_custom_prompt_seq_none", text="")
                                op_prop = 'seq_none'
                            elif scene.qwen_context_render_mode == 'REPLACE_STYLE' and scene.qwen_generation_method == 'generate':
                                row.prop(scene, "qwen_custom_prompt_seq_replace", text="")
                                op_prop = 'seq_replace'
                            elif scene.qwen_context_render_mode == 'ADDITIONAL' and scene.qwen_generation_method == 'generate':
                                row.prop(scene, "qwen_custom_prompt_seq_additional", text="")
                                op_prop = 'seq_additional'
                            else: # Refine mode or other
                                row.prop(scene, "qwen_custom_prompt_seq_none", text="")
                                op_prop = 'seq_none'
                            
                            op = row.operator("stablegen.reset_qwen_prompt", text="", icon='FILE_REFRESH')
                            op.prompt_type = op_prop

                    if (scene.generation_method == 'sequential' and scene.qwen_generation_method == 'generate' and
                            scene.qwen_context_render_mode in {'REPLACE_STYLE', 'ADDITIONAL'}):
                        context_box = content_box.box()
                        context_box.label(text="Context Render Options")

                        if scene.qwen_use_custom_prompts:
                            colors_box = context_box.box()
                            colors_box.label(text="Context Render Colors")
                            row = colors_box.row()
                            row.prop(scene, "qwen_guidance_background_color", text="Background")
                            row = colors_box.row()
                            row.prop(scene, "qwen_guidance_fallback_color", text="Fallback")

                        dilation_row = context_box.row()
                        dilation_row.prop(scene, "qwen_context_fallback_dilation", text="Fallback Dilate (px)")

                        cleanup_row = context_box.row()
                        cleanup_row.prop(scene, "qwen_context_cleanup", text="Apply Cleanup", toggle=True, icon="BRUSH_DATA")
                        if scene.qwen_context_cleanup:
                            row = context_box.row()
                            row.prop(scene, "qwen_context_cleanup_hue_tolerance", text="Hue Tol (°)")
                            row = context_box.row()
                            row.prop(scene, "qwen_context_cleanup_value_adjust", text="Value Adjust")

                elif scene.model_architecture == 'sdxl' or scene.model_architecture == 'flux1':
                    # IPAdapter Parameters
                    if not scene.generation_method == 'uv_inpaint':
                        ipadapter_main_box = content_box.box() # Group IPAdapter settings together
                        if scene.model_architecture == 'flux1':
                            row = ipadapter_main_box.row()
                            row.prop(scene, "use_flux_lora", text="Use Flux Depth LoRA", toggle=True, icon="MODIFIER")
                        row = ipadapter_main_box.row()
                        row.prop(scene, "use_ipadapter", text="Use IPAdapter (External image)", toggle=True, icon="MOD_MULTIRES")
                        if scene.use_ipadapter:
                            sub_ip_box = ipadapter_main_box.box() 
                            row = sub_ip_box.row()
                            row.prop(scene, "ipadapter_image", text="Image")
                            row = sub_ip_box.row()
                            row.prop(scene, "ipadapter_strength", text="Strength")
                            if width_mode == 'narrow':
                                row = sub_ip_box.row()
                            row.prop(scene, "ipadapter_start", text="Start")
                            if width_mode == 'narrow':
                                row = sub_ip_box.row()
                            row.prop(scene, "ipadapter_end", text="End")
                            split = sub_ip_box.split(factor=0.5)
                            if context.scene.model_architecture == 'sdxl':
                                split.label(text="Weight Type:")
                                split.prop(scene, "ipadapter_weight_type", text="")
                    
                    content_box.separator() # Separator between IPAdapter and ControlNet if both are shown
                    # ControlNet Parameters
                    if not (scene.model_architecture == 'flux1' and scene.use_flux_lora):
                        cn_box = content_box.box()
                        row = cn_box.row()
                        row.alignment = 'CENTER'
                        row.label(text="ControlNet Units", icon="NODETREE")
                        for i, unit in enumerate(scene.controlnet_units): 
                            sub_unit_box = cn_box.box() # Each unit gets its own box
                            row = sub_unit_box.row()
                            row.label(text=f"Unit: {unit.unit_type.replace('_', ' ').title()}", icon="DOT") 
                            row.alignment = 'LEFT' 
                            
                            if width_mode == 'narrow':
                                split = sub_unit_box.split(factor=0.35, align=True) 
                            else:
                                split = sub_unit_box.split(factor=0.2, align=True) 
                            split.label(text="Model:")
                            split.prop(unit, "model_name", text="")
                            
                            row = sub_unit_box.row()
                            row.prop(unit, "strength", text="Strength")
                            if width_mode == 'narrow':
                                row = sub_unit_box.row()
                            row.prop(unit, "start_percent", text="Start")
                            if width_mode == 'narrow':
                                row = sub_unit_box.row()
                            row.prop(unit, "end_percent", text="End")
                            
                            if unit.unit_type == 'canny':
                                row = sub_unit_box.row()
                                row.prop(scene, "canny_threshold_low", text="Canny Low")
                                if width_mode == 'narrow':
                                    row = sub_unit_box.row()
                                row.prop(scene, "canny_threshold_high", text="Canny High")
                            if hasattr(unit, 'is_union') and unit.is_union: 
                                row = sub_unit_box.row()
                                row.prop(unit, "use_union_type", text="Set Union Type", toggle=True, icon="MOD_BOOLEAN")
                        
                        btn_row = cn_box.row(align=True) 
                        if width_mode == 'wide':
                            btn_row.operator("stablegen.add_controlnet_unit", text="Add Unit", icon="ADD")
                            btn_row.operator("stablegen.remove_controlnet_unit", text="Remove Unit", icon="REMOVE")
                        else:
                            cn_box.operator("stablegen.add_controlnet_unit", text="Add ControlNet Unit", icon="ADD")
                            cn_box.operator("stablegen.remove_controlnet_unit", text="Remove Last ControlNet Unit", icon="REMOVE")

            if _show_diffusion_sections and not scene.model_architecture == 'qwen_image_edit':
                # --- Inpainting Options (Conditional) ---
                if scene.generation_method == 'uv_inpaint' or scene.generation_method == 'sequential':
                    content_box = draw_collapsible_section(advanced_params_box, "show_masking_inpainting_settings", "Inpainting Options", icon="MOD_MASK")
                    if content_box: # content_box is the container for these settings
                        row = content_box.row()
                        row.prop(scene, "differential_diffusion", text="Use Differential Diffusion", toggle=True, icon="SMOOTHCURVE")
                        
                        if scene.differential_diffusion:
                            row = content_box.row()
                            row.prop(scene, "differential_noise", text="Add Latent Noise Mask", toggle=True, icon="MOD_NOISE")

                        if not (scene.differential_diffusion and not scene.differential_noise): 
                            row = content_box.row()
                            row.prop(scene, "mask_blocky", text="Use Blocky Mask", icon="MOD_MASK") 
                            
                            if width_mode == 'narrow':
                                row = content_box.row()
                                
                            row.prop(scene, "blur_mask", text="Blur Mask", toggle=True, icon="SURFACE_NSPHERE")

                            if scene.blur_mask:
                                row = content_box.row()
                                row.prop(scene, "blur_mask_radius", text="Blur Radius")
                                if width_mode == 'narrow':
                                    row = content_box.row()
                                row.prop(scene, "blur_mask_sigma", text="Blur Sigma")

                            row = content_box.row() # Draw directly in content_box
                            row.prop(scene, "grow_mask_by", text="Grow Mask By")


            # --- Generation Mode Specifics ---
            if _show_diffusion_sections:
                mode_specific_outer_box = draw_collapsible_section(advanced_params_box, "show_mode_specific_settings", "Generation Mode Specifics", icon="OPTIONS")
            else:
                mode_specific_outer_box = None
            if mode_specific_outer_box: # This is the box where all mode-specific UIs should go
                
                # Qwen Local Edit Mode Parameters
                if scene.model_architecture.startswith('qwen') and scene.qwen_generation_method == 'local_edit':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Qwen Local Edit Parameters", icon='BRUSH_DATA')
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "denoise", text="Denoise")
                    
                    # Angle Ramp Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "refine_angle_ramp_active", text="Use Angle-Based Blending", icon="DRIVER")
                    if scene.refine_angle_ramp_active:
                        row = box.row()
                        row.prop(scene, "refine_angle_ramp_pos_0", text="Black Point")
                        row.prop(scene, "refine_angle_ramp_pos_1", text="White Point")
                    
                    # Feather Ramp Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "visibility_vignette", text="Use Vignette Blending", icon="DRIVER")
                    if scene.visibility_vignette:
                        row = box.row()
                        row.prop(scene, "refine_feather_ramp_pos_0", text="Black Point")
                        row.prop(scene, "refine_feather_ramp_pos_1", text="White Point")
                        row = box.row()
                        row.prop(scene, "visibility_vignette_width", text="Feather Width")
                        if width_mode == 'narrow':
                            row = box.row()
                        row.prop(scene, "visibility_vignette_softness", text="Feather Softness")
                        row = box.row()
                        row.prop(scene, "visibility_vignette_blur", text="Blur Mask", icon="SURFACE_NSPHERE")

                    # Edge Feather Projection Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "refine_edge_feather_projection", text="Edge Feather (Projection)", icon="MOD_EDGESPLIT")
                    if scene.refine_edge_feather_projection:
                        row = box.row()
                        row.prop(scene, "refine_edge_feather_width", text="Feather Width (px)")
                        row = box.row()
                        row.prop(scene, "refine_edge_feather_softness", text="Feather Softness")

                    # Color Matching
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "view_blend_use_color_match", text="Match Colors to Viewport", toggle=True, icon="COLOR")
                    if scene.view_blend_use_color_match:
                        row = box.row(align=True)
                        row.prop(scene, "view_blend_color_match_method", text="Method")
                        row = box.row()
                        row.prop(scene, "view_blend_color_match_strength", text="Strength")

                # Qwen Refine Mode Parameters
                elif scene.model_architecture.startswith('qwen') and scene.qwen_generation_method == 'refine':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Qwen Refine Parameters", icon='SHADERFX')
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "denoise", text="Denoise")

                # Grid Mode Parameters
                elif scene.generation_method == 'grid':
                    # Draw Grid parameters directly into mode_specific_outer_box
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Grid Mode Parameters", icon="MESH_GRID")
                    
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "refine_images", text="Refine Images", toggle=True, icon="SHADERFX")
                    if scene.refine_images:
                        split = mode_specific_outer_box.split(factor=0.5)
                        split.label(text="Refine Sampler:")
                        split.prop(scene, "refine_sampler", text="")
                        
                        split = mode_specific_outer_box.split(factor=0.5)
                        split.label(text="Refine Scheduler:")
                        split.prop(scene, "refine_scheduler", text="")
                        
                        row = mode_specific_outer_box.row()
                        row.prop(scene, "denoise", text="Denoise")
                        if width_mode == 'narrow':
                            row = mode_specific_outer_box.row()
                        row.prop(scene, "refine_cfg", text="Refine CFG")
                        if width_mode == 'narrow':
                            row = mode_specific_outer_box.row()
                        row.prop(scene, "refine_steps", text="Refine Steps")

                        row = mode_specific_outer_box.row() 
                        split = mode_specific_outer_box.split(factor=0.25)
                        split.label(text="Refine Prompt:")
                        split.prop(scene, "refine_prompt", text="")
                        
                        split = mode_specific_outer_box.split(factor=0.5) 
                        split.label(text="Refine Upscale:") 
                        split.prop(scene, "refine_upscale_method", text="")

                # Separate Mode Parameters
                elif scene.generation_method == 'separate':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Separate Mode Parameters", icon='FORCE_FORCE')
                    
                    row = mode_specific_outer_box.row() 
                    row.prop(scene, "sequential_ipadapter", text="Use IPAdapter for Separate Mode", toggle=True, icon="MODIFIER")
                    if scene.sequential_ipadapter: 
                        sub_ip_box_separate = mode_specific_outer_box.box()
                        
                        split = sub_ip_box_separate.split(factor=0.5) 
                        split.label(text="Mode:")
                        split.prop(scene, "sequential_ipadapter_mode", text="") 

                        if scene.sequential_ipadapter_mode == 'recent':
                            sub_ip_box_separate.prop(scene, "sequential_desaturate_factor", text="Desaturate")
                            sub_ip_box_separate.prop(scene, "sequential_contrast_factor", text="Reduce Contrast")

                        if context.scene.model_architecture != 'qwen_image_edit':
                            split = sub_ip_box_separate.split(factor=0.5) 
                            if context.scene.model_architecture == 'sdxl':
                                split.label(text="Weight Type:")
                                split.prop(scene, "ipadapter_weight_type", text="")
                        
                        row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_strength", text="Strength")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_start", text="Start")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_end", text="End")    
                        
                        if context.scene.sequential_ipadapter_mode == 'first':
                            row = sub_ip_box_separate.row()
                            row.prop(scene, "sequential_ipadapter_regenerate", text="Regenerate First Image", toggle=True, icon="FILE_REFRESH")
                            if context.scene.sequential_ipadapter_regenerate:
                                row = sub_ip_box_separate.row()
                                row.prop(scene, "sequential_ipadapter_regenerate_wo_controlnet", text="Generate reference without ControlNet", toggle=True, icon="HIDE_OFF")

                # Refine Mode Parameters
                elif scene.generation_method == 'refine':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Refine Mode Parameters", icon='SHADERFX')
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "denoise", text="Denoise") 
                    row = mode_specific_outer_box.row() 
                    row.prop(scene, "sequential_ipadapter", text="Use IPAdapter for Refine Mode", toggle=True, icon="MODIFIER")
                    if scene.sequential_ipadapter: 
                        sub_ip_box_separate = mode_specific_outer_box.box()
                        
                        split = sub_ip_box_separate.split(factor=0.5) 
                        split.label(text="Mode:")
                        split.prop(scene, "sequential_ipadapter_mode", text="") 

                        split = sub_ip_box_separate.split(factor=0.5) 
                        if context.scene.model_architecture == 'sdxl':
                            split.label(text="Weight Type:")
                            split.prop(scene, "ipadapter_weight_type", text="")
                        
                        row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_strength", text="Strength")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_start", text="Start")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_end", text="End")    
                        
                        if context.scene.sequential_ipadapter_mode == 'first':
                            row = sub_ip_box_separate.row()
                            row.prop(scene, "sequential_ipadapter_regenerate", text="Regenerate First Image", toggle=True, icon="FILE_REFRESH")
                            if context.scene.sequential_ipadapter_regenerate:
                                row = sub_ip_box_separate.row()
                                row.prop(scene, "sequential_ipadapter_regenerate_wo_controlnet", text="Generate reference without ControlNet", toggle=True, icon="HIDE_OFF")

                # Local Edit Mode Parameters
                elif scene.generation_method == 'local_edit':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Local Edit Parameters", icon='BRUSH_DATA')
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "denoise", text="Denoise")
                    
                    # Angle Ramp Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "refine_angle_ramp_active", text="Use Angle-Based Blending", icon="DRIVER")
                    if scene.refine_angle_ramp_active:
                        row = box.row()
                        row.prop(scene, "refine_angle_ramp_pos_0", text="Black Point")
                        row.prop(scene, "refine_angle_ramp_pos_1", text="White Point")
                    
                    # Feather Ramp Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "visibility_vignette", text="Use Vignette Blending", icon="DRIVER")
                    if scene.visibility_vignette:
                        row = box.row()
                        row.prop(scene, "refine_feather_ramp_pos_0", text="Black Point")
                        row.prop(scene, "refine_feather_ramp_pos_1", text="White Point")
                        row = box.row()
                        row.prop(scene, "visibility_vignette_width", text="Feather Width")
                        if width_mode == 'narrow':
                            row = box.row()
                        row.prop(scene, "visibility_vignette_softness", text="Feather Softness")
                        row = box.row()
                        row.prop(scene, "visibility_vignette_blur", text="Blur Mask", icon="SURFACE_NSPHERE")

                    # Edge Feather Projection Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "refine_edge_feather_projection", text="Edge Feather (Projection)", icon="MOD_EDGESPLIT")
                    if scene.refine_edge_feather_projection:
                        row = box.row()
                        row.prop(scene, "refine_edge_feather_width", text="Feather Width (px)")
                        row = box.row()
                        row.prop(scene, "refine_edge_feather_softness", text="Feather Softness")

                    # Color Matching
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "view_blend_use_color_match", text="Match Colors to Viewport", toggle=True, icon="COLOR")
                    if scene.view_blend_use_color_match:
                        row = box.row(align=True)
                        row.prop(scene, "view_blend_color_match_method", text="Method")
                        row = box.row()
                        row.prop(scene, "view_blend_color_match_strength", text="Strength")

                    row = mode_specific_outer_box.row() 
                    row.prop(scene, "sequential_ipadapter", text="Use IPAdapter for Local Edit", toggle=True, icon="MODIFIER")
                    if scene.sequential_ipadapter: 
                        sub_ip_box_separate = mode_specific_outer_box.box()
                        
                        split = sub_ip_box_separate.split(factor=0.5) 
                        split.label(text="Mode:")
                        split.prop(scene, "sequential_ipadapter_mode", text="") 

                        split = sub_ip_box_separate.split(factor=0.5) 
                        if context.scene.model_architecture == 'sdxl':
                            split.label(text="Weight Type:")
                            split.prop(scene, "ipadapter_weight_type", text="")
                        
                        row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_strength", text="Strength")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_start", text="Start")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_end", text="End")    
                        
                        if context.scene.sequential_ipadapter_mode == 'first':
                            row = sub_ip_box_separate.row()
                            row.prop(scene, "sequential_ipadapter_regenerate", text="Regenerate First Image", toggle=True, icon="FILE_REFRESH")
                            if context.scene.sequential_ipadapter_regenerate:
                                row = sub_ip_box_separate.row()
                                row.prop(scene, "sequential_ipadapter_regenerate_wo_controlnet", text="Generate reference without ControlNet", toggle=True, icon="HIDE_OFF")
                
                # UV Inpainting Parameters
                elif scene.generation_method == 'uv_inpaint':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="UV Inpainting Parameters", icon="IMAGE_PLANE")
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "allow_modify_existing_textures", text="Allow Modifying Existing Textures", toggle=True, icon="TEXTURE")
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "ask_object_prompts", text="Ask for Object Specific Prompts", toggle=True, icon="QUESTION")

                # Sequential Mode Parameters
                elif scene.generation_method == 'sequential':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Sequential Mode Parameters", icon="SEQUENCE")
                    
                    if not (scene.differential_diffusion and not scene.differential_noise): 
                        row = mode_specific_outer_box.row()
                        row.prop(scene, "sequential_smooth", text="Use Smooth Visibility Map", toggle=True, icon="MOD_SMOOTH")
                        if width_mode == 'narrow':
                            row = mode_specific_outer_box.row()
                        row.prop(scene, "weight_exponent_mask", text="Exponent for Visibility Map", toggle=True, icon="IPO_EXPO") 
                        
                        if not scene.sequential_smooth:
                            row = mode_specific_outer_box.row()
                            row.prop(scene, "sequential_factor", text="Visibility Threshold") 
                        else:
                            row = mode_specific_outer_box.row()
                            row.prop(scene, "sequential_factor_smooth", text="Smooth Visibility Black Point")
                            if width_mode == 'narrow':
                                row = mode_specific_outer_box.row()
                            row.prop(scene, "sequential_factor_smooth_2", text="Smooth Visibility White Point")
                    
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "sequential_ipadapter", text="Use IPAdapter for Sequential Mode", toggle=True, icon="MODIFIER")
                    if scene.sequential_ipadapter:
                        sub_ip_seq_box = mode_specific_outer_box.box()
                        
                        split = sub_ip_seq_box.split(factor=0.5)
                        split.label(text="Mode:")
                        split.prop(scene, "sequential_ipadapter_mode", text="")

                        if scene.sequential_ipadapter_mode == 'recent':
                            sub_ip_seq_box.prop(scene, "sequential_desaturate_factor", text="Desaturate")
                            sub_ip_seq_box.prop(scene, "sequential_contrast_factor", text="Reduce Contrast")

                        if context.scene.model_architecture != 'qwen_image_edit':
                            split = sub_ip_seq_box.split(factor=0.5)
                            if context.scene.model_architecture == 'sdxl':
                                split.label(text="Weight Type:")
                                split.prop(scene, "ipadapter_weight_type", text="")
                        
                        row = sub_ip_seq_box.row()
                        row.prop(scene, "ipadapter_strength", text="Strength")
                        if width_mode == 'narrow':
                            row = sub_ip_seq_box.row()
                        row.prop(scene, "ipadapter_start", text="Start")
                        if width_mode == 'narrow':  
                            row = sub_ip_seq_box.row()
                        row.prop(scene, "ipadapter_end", text="End")     
                        
                        if context.scene.sequential_ipadapter_mode == 'first':
                            row = sub_ip_seq_box.row()
                            row.prop(scene, "sequential_ipadapter_regenerate", text="Regenerate First Image", toggle=True, icon="FILE_REFRESH")
                            if context.scene.sequential_ipadapter_regenerate:
                                row = sub_ip_seq_box.row()
                                row.prop(scene, "sequential_ipadapter_regenerate_wo_controlnet", text="Generate reference without ControlNet", toggle=True, icon="HIDE_OFF")   

        # --- Tools ---
        layout.separator()
        tools_box = layout.box()
        row = tools_box.row()
        row.alignment = 'CENTER'
        row.label(text="Tools", icon="TOOL_SETTINGS")
        
        row = tools_box.row() 
        row.operator("object.switch_material", text="Switch Material", icon="MATERIAL_DATA")
        if width_mode == 'narrow':
            row = tools_box.row()
        row.operator("object.add_hdri", text="Add HDRI Light", icon="WORLD")
        
        row = tools_box.row()
        row.operator("object.apply_all_mesh_modifiers", text="Apply All Modifiers", icon="MODIFIER_DATA") 
        if width_mode == 'narrow':
            row = tools_box.row()
        row.operator("object.curves_to_mesh", text="Convert Curves to Mesh", icon="CURVE_DATA")
        
        row = tools_box.row()
        if config_error_message:
            row.enabled = False
            row.operator("object.export_orbit_gif", text=f"Cannot Export: {config_error_message}", icon="ERROR")
        else:
            row.enabled = True
            row.operator("object.export_orbit_gif", text="Export Orbit GIF/MP4", icon="RENDER_ANIMATION")

        if width_mode == 'narrow':
            row = tools_box.row()
        row.operator("object.stablegen_reproject", text="Reproject Textures", icon="FILE_REFRESH")

        row = tools_box.row()
        row.operator("object.stablegen_mirror_reproject", text="Mirror Last Projection", icon="MOD_MIRROR")

        # --- Debug Tools ---
        prefs = context.preferences.addons.get(__package__)
        if prefs and prefs.preferences.enable_debug:
            layout.separator()
            debug_box = layout.box()
            row = debug_box.row()
            row.alignment = 'CENTER'
            row.label(text="Debug / Diagnostics", icon="GHOST_ENABLED")

            row = debug_box.row()
            row.operator("stablegen.debug_solid_colors", text="Draw Solid Colors", icon="COLOR")
            if width_mode == 'narrow':
                row = debug_box.row()
            row.operator("stablegen.debug_grid_pattern", text="Grid Pattern", icon="MESH_GRID")

            row = debug_box.row()
            row.operator("stablegen.debug_coverage_heatmap", text="Coverage Heatmap", icon="AREA_SWAP")
            if width_mode == 'narrow':
                row = debug_box.row()
            row.operator("stablegen.debug_visibility_material", text="Visibility Material", icon="HIDE_OFF")

            row = debug_box.row()
            row.operator("stablegen.debug_uv_seam_viz", text="UV Seam Visualizer", icon="UV")
            if width_mode == 'narrow':
                row = debug_box.row()
            row.operator("stablegen.debug_restore_material", text="Remove Debug Mats", icon="TRASH")

            row = debug_box.row()
            op = row.operator("stablegen.debug_per_camera_weight", text="Per-Camera Weight", icon="CAMERA_DATA")
            if width_mode == 'narrow':
                row = debug_box.row()
            op2 = row.operator("stablegen.debug_feather_preview", text="Feather Preview", icon="MOD_SMOOTH")

        layout.separator()
          

class ResetQwenPrompt(bpy.types.Operator):
    """Resets a Qwen guidance prompt to its default value"""
    bl_idname = "stablegen.reset_qwen_prompt"
    bl_label = "Reset Qwen Prompt"
    bl_description = "Reset this prompt to its default value based on the current settings"

    prompt_type: bpy.props.StringProperty()

    def execute(self, context):
        from .workflows import WorkflowManager
        # We need an operator instance to call the helper, a bit of a workaround
        # This doesn't run generation, just gives us access to the method
        wm = WorkflowManager(self) 

        # Determine which prompt to reset
        if self.prompt_type == 'initial':
            default_prompt = wm._get_qwen_default_prompts(context, is_initial_image=True)
            context.scene.qwen_custom_prompt_initial = default_prompt
        elif self.prompt_type == 'seq_none':
            default_prompt = wm._get_qwen_default_prompts(context, is_initial_image=False)
            context.scene.qwen_custom_prompt_seq_none = default_prompt
        elif self.prompt_type == 'seq_replace':
            default_prompt = wm._get_qwen_default_prompts(context, is_initial_image=False)
            context.scene.qwen_custom_prompt_seq_replace = default_prompt
        elif self.prompt_type == 'seq_additional':
            default_prompt = wm._get_qwen_default_prompts(context, is_initial_image=False)
            context.scene.qwen_custom_prompt_seq_additional = default_prompt
        
        self.report({'INFO'}, "Prompt reset to default.")
        return {'FINISHED'}

class ApplyPreset(bpy.types.Operator):
    """Apply selected preset values to parameters"""
    bl_idname = "stablegen.apply_preset"
    bl_label = "Apply Preset"
    bl_description = "Set multiple parameters based on selected preset for easier configuration"

    def execute(self, context):
        preset = context.scene.stablegen_preset
        if preset in PRESETS:
            values = PRESETS[preset]
            
            # Apply regular parameters
            for key, value in values.items():
                if key not in ["controlnet_units", "lora_units", "description"] and hasattr(context.scene, key):
                    setattr(context.scene, key, value)
            
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
                        
            self.report({'INFO'}, f"Preset '{preset}' applied.")
        else:
            self.report({'INFO'}, "Custom preset active.")
        
        # Force update to ensure preset detection is correct after list changes
        update_parameters(self, context)

        return {'FINISHED'}

class SavePreset(bpy.types.Operator):
    """Save the current parameter values as a custom preset"""
    bl_idname = "stablegen.save_preset"
    bl_label = "Save Custom Preset"
    
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
