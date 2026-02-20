import bpy
import os
import json
import uuid
import random
import struct
import websocket
import socket
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

from .util.helpers import prompt_text_qwen_image_edit, prompt_text_trellis2, prompt_text_trellis2_shape_only
from .utils import get_generation_dirs
from .timeout_config import get_timeout

from io import BytesIO
import numpy as np
from PIL import Image

class WorkflowManager:
    def __init__(self, operator):
        """
        Initializes the WorkflowManager.

        Args:
            operator: The instance of the ComfyUIGenerate operator.
        """
        self.operator = operator

    def _crop_and_vignette(
        self,
        img_bytes,
        border_px: int = 8,
        feather: float = 0.15,
        gamma: float = 0.7,
    ):
        """
        Crop a constant border from the image and apply an *alpha* vignette
        that fades the image out toward the edges.

        IMPORTANT:
          - RGB is NOT darkened here, only alpha is shaped.
          - This assumes the shader uses alpha to blend with the underlying
            surface (e.g. Mix using alpha as Fac).

        border_px : how many pixels to remove on each side.
        feather   : fraction of min(width, height) that is used as the feather band.
        gamma     : exponent applied to the feather mask (1.0 = linear).
        """

        if img_bytes is None:
            return None

        # --- Load image as RGBA without touching color ---
        buf = BytesIO(img_bytes)
        try:
            img = Image.open(buf).convert("RGBA")
        except Exception as e:
            print(f"[StableGen] _crop_and_vignette: cannot decode image "
                  f"({len(img_bytes)} bytes): {e}  — returning raw bytes.")
            return img_bytes

        w, h = img.size
        if w <= 2 * border_px or h <= 2 * border_px:
            # Too small to crop; just return as-is
            out_buf = BytesIO()
            img.save(out_buf, format="PNG")
            return out_buf.getvalue()

        # --- Crop hard Comfy border first ---
        left   = border_px
        top    = border_px
        right  = w - border_px
        bottom = h - border_px
        img = img.crop((left, top, right, bottom))
        w, h = img.size

        # Convert to numpy for mask math
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 4]
        rgb = arr[..., :3]
        alpha_orig = arr[..., 3]

        # If there was no alpha, assume fully opaque as starting point
        if np.all(alpha_orig == 0):
            alpha_orig = np.ones_like(alpha_orig, dtype=np.float32)

        # --- Build rectangular feather mask based on distance to nearest edge ---

        # Normalized distance (in pixels) to each edge
        yy, xx = np.mgrid[0:h, 0:w]
        dist_to_left   = xx
        dist_to_right  = (w - 1) - xx
        dist_to_top    = yy
        dist_to_bottom = (h - 1) - yy

        dist_to_edge = np.minimum(
            np.minimum(dist_to_left, dist_to_right),
            np.minimum(dist_to_top, dist_to_bottom),
        ).astype(np.float32)

        # Feather band thickness in pixels
        # (0.0–0.5 of the smaller dimension is reasonable)
        min_dim = float(min(w, h))
        feather_px = max(1.0, feather * 0.5 * min_dim)

        # 0 at the border, 1 in the interior beyond the feather band
        mask = dist_to_edge / feather_px
        mask = np.clip(mask, 0.0, 1.0)

        # Shape the transition with gamma (only on the mask, not on RGB!)
        if gamma != 1.0:
            # stronger bias toward 1.0 in mid-range
            mask = np.clip(mask, 1e-6, 1.0)
            mask = np.power(mask, gamma)

        # Compose final alpha: original alpha * vignette mask
        alpha_new = alpha_orig * mask

        # Reassemble RGBA, leaving RGB completely unchanged
        out = np.empty_like(arr)
        out[..., :3] = rgb
        out[..., 3] = alpha_new
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

        img_out = Image.fromarray(out, mode="RGBA")

        out_buf = BytesIO()
        img_out.save(out_buf, format="PNG")
        return out_buf.getvalue()

    def _build_qwen_reference_latent_chain(self, prompt, NODES, start_node_id=700):
        """
        Build VAEEncode + ReferenceLatent chain for the Qwen unzoom fix.

        When the VAE input is disconnected from TextEncodeQwenImageEditPlus,
        the node no longer creates reference latents (and no longer forces
        a 1 MP rescale).  We replicate the reference-latent injection
        manually so the latents match the actual output resolution.

        For each active image (image1 / image2 / image3) still present in
        the positive text-encode node, a VAEEncode node is created and a
        ReferenceLatent node is chained onto both positive and negative
        conditioning.  The KSampler inputs are then updated to point at
        the tail of the chain.

        When ``qwen_timestep_zero_ref`` is enabled on the scene, a
        FluxKontextMultiReferenceLatentMethod node is appended with
        ``index_timestep_zero`` so the model treats reference latents as
        fully-denoised tokens, reducing color shift / over-saturation.
        """
        pos_node = NODES['pos_prompt']
        neg_node = NODES['neg_prompt']
        sampler_node = NODES['sampler']
        vae_ref = ["4", 0]  # VAELoader node

        # Collect active image references from the pos text-encode node.
        # image1 is always present; image2/image3 may have been deleted.
        active_refs = []
        pos_inputs = prompt[pos_node]['inputs']
        for key in ('image1', 'image2', 'image3'):
            if key in pos_inputs:
                active_refs.append(pos_inputs[key])

        if not active_refs:
            return

        node_id = start_node_id
        # Avoid collisions with existing dynamic nodes
        while str(node_id) in prompt:
            node_id += 1

        pos_cond_ref = [pos_node, 0]
        neg_cond_ref = [neg_node, 0]

        for i, img_ref in enumerate(active_refs):
            # --- VAEEncode for this image ---
            # For image1 we can reuse the existing node 8 which already
            # encodes the same image (also used as KSampler latent_image).
            if i == 0:
                vae_enc_ref = ["8", 0]
            else:
                vae_enc_id = str(node_id)
                node_id += 1
                prompt[vae_enc_id] = {
                    "inputs": {
                        "pixels": img_ref,
                        "vae": vae_ref
                    },
                    "class_type": "VAEEncode",
                    "_meta": {"title": f"VAE Encode (Ref {i + 1})"}
                }
                vae_enc_ref = [vae_enc_id, 0]

            # --- ReferenceLatent for positive ---
            ref_pos_id = str(node_id)
            node_id += 1
            prompt[ref_pos_id] = {
                "inputs": {
                    "conditioning": pos_cond_ref,
                    "latent": vae_enc_ref
                },
                "class_type": "ReferenceLatent",
                "_meta": {"title": f"Reference Latent Pos {i + 1}"}
            }
            pos_cond_ref = [ref_pos_id, 0]

            # --- ReferenceLatent for negative ---
            ref_neg_id = str(node_id)
            node_id += 1
            prompt[ref_neg_id] = {
                "inputs": {
                    "conditioning": neg_cond_ref,
                    "latent": vae_enc_ref
                },
                "class_type": "ReferenceLatent",
                "_meta": {"title": f"Reference Latent Neg {i + 1}"}
            }
            neg_cond_ref = [ref_neg_id, 0]

        # Point KSampler at the final chained conditioning
        prompt[sampler_node]['inputs']['positive'] = pos_cond_ref
        prompt[sampler_node]['inputs']['negative'] = neg_cond_ref

        # ── Timestep-zero reference method (colour-shift reduction) ─────
        context = bpy.context
        if getattr(context.scene, 'qwen_timestep_zero_ref', False):
            # Positive conditioning
            tz_pos_id = str(node_id)
            node_id += 1
            prompt[tz_pos_id] = {
                "inputs": {
                    "conditioning": pos_cond_ref,
                    "reference_latents_method": "index_timestep_zero"
                },
                "class_type": "FluxKontextMultiReferenceLatentMethod",
                "_meta": {"title": "Ref Method (Pos)"}
            }
            prompt[sampler_node]['inputs']['positive'] = [tz_pos_id, 0]

            # Negative conditioning
            tz_neg_id = str(node_id)
            node_id += 1
            prompt[tz_neg_id] = {
                "inputs": {
                    "conditioning": neg_cond_ref,
                    "reference_latents_method": "index_timestep_zero"
                },
                "class_type": "FluxKontextMultiReferenceLatentMethod",
                "_meta": {"title": "Ref Method (Neg)"}
            }
            prompt[sampler_node]['inputs']['negative'] = [tz_neg_id, 0]

    def _get_qwen_default_prompts(self, context, is_initial_image):
        """Gets the default Qwen prompts based on the current context."""
        # Check if we are in a real generation context vs. just resetting a prompt
        is_generating = hasattr(self.operator, '_current_image')
        is_subsequent_in_sequence = is_generating and self.operator._current_image > 0

        if is_initial_image:
            # First frame only has a style reference when an explicit external image is supplied.
            style_image_provided = context.scene.qwen_use_external_style_image
        else:
            # Later frames can draw style from previous renders, external sources, or context renders.
            style_image_provided = (
                context.scene.qwen_use_external_style_image or
                context.scene.sequential_ipadapter or
                context.scene.qwen_context_render_mode in {'REPLACE_STYLE', 'ADDITIONAL'}
            )
        context_mode = context.scene.qwen_context_render_mode

        if is_initial_image:
            if not style_image_provided:
                return "Change the format of image 1 to '{main_prompt}'"
            else:
                return "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2"
        else: # Subsequent image
            if context_mode == 'ADDITIONAL':
                return "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object."
            elif context_mode == 'REPLACE_STYLE':
                return "Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas."
            else: # NONE or other cases
                if not style_image_provided:
                     return "Change the format of image 1 to '{main_prompt}'"
                else:
                    return "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2"


    def generate_qwen_refine(self, context, camera_id=None):
        """Generates an image using the Qwen-Image-Edit workflow for refinement."""
        server_address = context.preferences.addons[__package__].preferences.server_address
        client_id = str(uuid.uuid4())
        revision_dir = get_generation_dirs(context)["revision"]

        prompt = json.loads(prompt_text_qwen_image_edit)

        NODES = {
            'sampler': "1",
            'save_image': "5",
            'model_sampler': "6", # ModelSamplingAuraFlow
            'cfg_norm': "7", # CFGNorm
            'vae_encode': "8",
            'pos_prompt': "12",
            'neg_prompt': "11",
            'unet_loader': "13",
            'guidance_map_loader': "14", # Image 1 (structure)
            'style_map_loader': "15",   # Image 2 (style)
            'context_render_loader': "16", # Image 3 (context render)
        }

        # --- Build LoRA chain ---
        initial_model_input = [NODES['unet_loader'], 0]
        dummy_clip_input = [NODES['unet_loader'], 0] 

        is_nunchaku = context.scene.model_name.lower().endswith('.safetensors')
        lora_class = "NunchakuQwenImageLoraLoader" if is_nunchaku else "LoraLoaderModelOnly"

        prompt, final_lora_model_out, _ = self._build_lora_chain(
            prompt, context,
            initial_model_input, dummy_clip_input,
            start_node_id=500, 
            lora_class_type=lora_class 
        )

        prompt[NODES['model_sampler']]['inputs']['model'] = final_lora_model_out

        # --- Configure Inputs ---
        # Image 1: The current render (structure)
        render_info = self.operator._get_uploaded_image_info(context, "inpaint", subtype="render", camera_id=camera_id)
        if not render_info:
            self.operator._error = f"Could not find or upload render for camera {camera_id}."
            return {"error": "conn_failed"}
        prompt[NODES['guidance_map_loader']]['inputs']['image'] = render_info['name']

        # --- Configure Style Image (Image 2) ---
        use_prev_ref = context.scene.qwen_refine_use_prev_ref
        style_image_info = None
        
        if use_prev_ref and camera_id > 0:
             # Use previous generated image
             style_image_info = self.operator._get_uploaded_image_info(context, "generated", camera_id=camera_id - 1, material_id=self.operator._material_id)
        
        if style_image_info:
            prompt[NODES['style_map_loader']]['inputs']['image'] = style_image_info['name']
        else:
            # Fallback to external style image if configured
            if context.scene.qwen_use_external_style_image:
                 style_image_info = self.operator._get_uploaded_image_info(context, "custom", filename=bpy.path.abspath(context.scene.qwen_external_style_image))
                 if style_image_info:
                     prompt[NODES['style_map_loader']]['inputs']['image'] = style_image_info['name']
            
            # Fallback to TRELLIS.2 input image as style
            if not style_image_info and getattr(context.scene, 'qwen_use_trellis2_style', False):
                t2_path = getattr(context.scene, 'trellis2_last_input_image', '')
                if t2_path and os.path.exists(bpy.path.abspath(t2_path)):
                    style_image_info = self.operator._get_uploaded_image_info(context, "custom", filename=bpy.path.abspath(t2_path))
                    if style_image_info:
                        prompt[NODES['style_map_loader']]['inputs']['image'] = style_image_info['name']
            
            # If still no style image, remove Image 2 inputs
            if not style_image_info:
                del prompt[NODES['style_map_loader']]
                del prompt[NODES['pos_prompt']]['inputs']['image2']
                del prompt[NODES['neg_prompt']]['inputs']['image2']

        # --- Configure Depth Map (Image 3) ---
        use_depth = context.scene.qwen_refine_use_depth
        depth_info = None
        if use_depth:
            depth_info = self.operator._get_uploaded_image_info(context, "controlnet", subtype="depth", camera_id=camera_id)
        
        if depth_info:
            prompt[NODES['context_render_loader']]['inputs']['image'] = depth_info['name']
        else:
            # Remove Context Render (Image 3) if not used
            del prompt[NODES['context_render_loader']]
            del prompt[NODES['pos_prompt']]['inputs']['image3']
            del prompt[NODES['neg_prompt']]['inputs']['image3']

        # --- Prompt ---
        user_prompt = context.scene.comfyui_prompt
        final_prompt = f"Modify image1 to {user_prompt}"
        if depth_info:
            final_prompt += ". Use image3 as depth map reference."
        
        prompt[NODES['pos_prompt']]['inputs']['prompt'] = final_prompt

        # --- Reference Latent chain (Qwen unzoom fix) ---
        self._build_qwen_reference_latent_chain(prompt, NODES)
        
        # --- Save and Execute ---
        self._save_prompt_to_file(prompt, revision_dir)
        
        ws = self._connect_to_websocket(server_address, client_id)
        if ws is None:
            return {"error": "conn_failed"}

        images = None
        try:
            images = self._execute_prompt_and_get_images(ws, prompt, client_id, server_address, NODES)
        finally:
            if ws:
                ws.close()

        if images is None or isinstance(images, dict) and "error" in images:
            return {"error": "conn_failed"}
        
        return images[NODES['save_image']][0]

    def generate_qwen_edit(self, context, camera_id=None):
        """Generates an image using the Qwen-Image-Edit workflow."""
        server_address = context.preferences.addons[__package__].preferences.server_address
        client_id = str(uuid.uuid4())
        revision_dir = get_generation_dirs(context)["revision"]

        prompt = json.loads(prompt_text_qwen_image_edit)

        NODES = {
            'sampler': "1",
            'save_image': "5",
            'model_sampler': "6", # ModelSamplingAuraFlow
            'cfg_norm': "7", # CFGNorm
            'vae_encode': "8",
            'pos_prompt': "12",
            'neg_prompt': "11",
            'unet_loader': "13",
            'guidance_map_loader': "14", # Image 1 (structure)
            'style_map_loader': "15",   # Image 2 (style)
            'context_render_loader': "16", # Image 3 (context render)
        }

        # --- Build LoRA chain ---
        # The Qwen workflow uses model-only LoRAs. We can reuse the existing
        # chain builder by providing a dummy CLIP input that won't be used.
        initial_model_input = [NODES['unet_loader'], 0]
        dummy_clip_input = [NODES['unet_loader'], 0] # Dummy, not used by LoraLoaderModelOnly

        is_nunchaku = context.scene.model_name.lower().endswith('.safetensors')
        lora_class = "NunchakuQwenImageLoraLoader" if is_nunchaku else "LoraLoaderModelOnly"

        prompt, final_lora_model_out, _ = self._build_lora_chain(
            prompt, context,
            initial_model_input, dummy_clip_input,
            start_node_id=500, # Use a high starting ID to avoid conflicts
            lora_class_type=lora_class # Specify model-only loader
        )

        # Connect the output of the LoRA chain to the next node in the model path
        prompt[NODES['model_sampler']]['inputs']['model'] = final_lora_model_out


        # --- Configure Inputs ---
        guidance_map_type = context.scene.qwen_guidance_map_type
        guidance_map_info = self.operator._get_uploaded_image_info(context, "controlnet", subtype=guidance_map_type, camera_id=camera_id)
        if not guidance_map_info:
            self.operator._error = f"Could not find or upload {guidance_map_type} map for camera {camera_id}."
            return {"error": "conn_failed"}
        prompt[NODES['guidance_map_loader']]['inputs']['image'] = guidance_map_info['name']

        # --- Configure Style Image (Image 2) and Prompts ---
        user_prompt = context.scene.comfyui_prompt
        style_image_info = None
        context_render_info = None
        context_mode = context.scene.qwen_context_render_mode
        remove_context = False

        # --- Camera Prompt Injection ---
        if context.scene.use_camera_prompts and self.operator._cameras and self.operator._current_image < len(self.operator._cameras):
            current_camera_name = self.operator._cameras[self.operator._current_image].name
            # Find the prompt in the collection
            prompt_item = next((item for item in context.scene.camera_prompts if item.name == current_camera_name), None)
            if prompt_item and prompt_item.prompt:
                view_desc = prompt_item.prompt
                # Prepend the view description
                user_prompt = f"{view_desc}, {user_prompt}"

        # --- Handle Context Render (Image 3) ---
        # This is only active in sequential mode after the first image
        is_initial_image = not self.operator._current_image > 0
        if not is_initial_image and context_mode != 'NONE':
            context_render_info = self.operator._get_uploaded_image_info(context, "inpaint", subtype="render", camera_id=camera_id)
            if not context_render_info:
                self.operator._error = f"Qwen context render enabled, but could not find context render for camera {camera_id}."
                return {"error": "conn_failed"}
            
            if context_mode == 'ADDITIONAL':
                # Switch context loader and style loader ids
                NODES['context_render_loader'], NODES['style_map_loader'] = NODES['style_map_loader'], NODES['context_render_loader']
                prompt[NODES['context_render_loader']]['inputs']['image'] = context_render_info['name']
                # The prompt needs to reference image 3
                if context.scene.qwen_use_custom_prompts:
                    pos_prompt_text = context.scene.qwen_custom_prompt_seq_additional.format(main_prompt=user_prompt)
                else:
                    pos_prompt_text = self._get_qwen_default_prompts(context, is_initial_image).format(main_prompt=user_prompt)
                prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text
            # If mode is REPLACE_STYLE, we will handle it in the style image section below.
            else:
                remove_context = True
        else:
            remove_context = True

        if remove_context:
            del prompt[NODES['context_render_loader']]
            del prompt[NODES['pos_prompt']]['inputs']['image3']
            del prompt[NODES['neg_prompt']]['inputs']['image3']


        # --- Handle Style Image (Image 2) ---
        # Determine if we should use the external style image for this specific frame
        use_external_this_frame = context.scene.qwen_use_external_style_image
        if use_external_this_frame and not is_initial_image and context.scene.qwen_external_style_initial_only:
            # If it's a subsequent image AND the "initial only" flag is set, DON'T use the external image.
            use_external_this_frame = False
            context.scene.sequential_ipadapter = True # Force using previous image as style

        # Case 1: External Style Image for this frame
        if use_external_this_frame:
            style_image_info = self.operator._get_uploaded_image_info(context, "custom", filename=bpy.path.abspath(context.scene.qwen_external_style_image))
            if not style_image_info:
                self.operator._error = "External style image enabled, but file not found or could not be uploaded."
                return {"error": "conn_failed"}
            if context_mode != 'ADDITIONAL':
                if context.scene.qwen_use_custom_prompts:
                    pos_prompt_text = (context.scene.qwen_custom_prompt_initial if is_initial_image else context.scene.qwen_custom_prompt_seq_none).format(main_prompt=user_prompt)
                else:
                    pos_prompt_text = self._get_qwen_default_prompts(context, is_initial_image).format(main_prompt=user_prompt)
                prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text
            else: # Additional mode
                if context.scene.qwen_use_custom_prompts:
                    pos_prompt_text = ()
                else:
                    pos_prompt_text = self._get_qwen_default_prompts(context, is_initial_image).format(main_prompt=user_prompt)
                prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text

        # Case 1b: TRELLIS.2 input image as style (when no external style configured)
        elif getattr(context.scene, 'qwen_use_trellis2_style', False):
            use_t2_this_frame = True
            if not is_initial_image and getattr(context.scene, 'qwen_trellis2_style_initial_only', False):
                # Initial-only mode: fall back to sequential style for subsequent frames
                use_t2_this_frame = False
                context.scene.sequential_ipadapter = True
            if use_t2_this_frame:
                t2_path = getattr(context.scene, 'trellis2_last_input_image', '')
                if t2_path and os.path.exists(bpy.path.abspath(t2_path)):
                    style_image_info = self.operator._get_uploaded_image_info(context, "custom", filename=bpy.path.abspath(t2_path))
                if style_image_info:
                    if context.scene.qwen_use_custom_prompts:
                        pos_prompt_text = (context.scene.qwen_custom_prompt_initial).format(main_prompt=user_prompt)
                    else:
                        pos_prompt_text = self._get_qwen_default_prompts(context, is_initial_image).format(main_prompt=user_prompt)
                    prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text

        # Case 2: Sequential generation (after first image)
        elif not is_initial_image:
            if context_mode == 'REPLACE_STYLE':
                # The context render becomes the style image
                style_image_info = context_render_info
                if context.scene.qwen_use_custom_prompts:
                    pos_prompt_text = context.scene.qwen_custom_prompt_seq_replace.format(main_prompt=user_prompt)
                else:
                    pos_prompt_text = self._get_qwen_default_prompts(context, is_initial_image).format(main_prompt=user_prompt)
                prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text
            elif context.scene.sequential_ipadapter: # Use previous generated image
                if context.scene.sequential_ipadapter_mode == 'original_render':
                    # For original_render in Qwen context, fall back to first mode since
                    # the render is already used as the guidance map (Image 1).
                    ref_cam_id = 0
                else:
                    ref_cam_id = 0 if context.scene.sequential_ipadapter_mode == 'first' else self.operator._current_image - 1
                style_image_info = self.operator._get_uploaded_image_info(context, "generated", camera_id=ref_cam_id, material_id=self.operator._material_id)
                if not style_image_info:
                    self.operator._error = f"Sequential mode error: Could not find previous image for camera {ref_cam_id} to use as style."
                    return {"error": "conn_failed"}
                if context_mode != 'ADDITIONAL':
                    if context.scene.qwen_use_custom_prompts:
                        pos_prompt_text = context.scene.qwen_custom_prompt_seq_none.format(main_prompt=user_prompt)
                    else:
                        pos_prompt_text = self._get_qwen_default_prompts(context, is_initial_image).format(main_prompt=user_prompt)
                    prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text
            # If neither of the above, style_image_info remains None, handled below

        # Case 3: First image of a sequence, or separate generation, or no style source in sequential
        if style_image_info is None:
            # No style image is provided. For the first image, we remove image2 entirely.
            del prompt[NODES['style_map_loader']]
            del prompt[NODES['pos_prompt']]['inputs']['image2']
            del prompt[NODES['neg_prompt']]['inputs']['image2']
            if context.scene.qwen_use_custom_prompts:
                pos_prompt_text = (context.scene.qwen_custom_prompt_initial if is_initial_image else context.scene.qwen_custom_prompt_seq_none).format(main_prompt=user_prompt)
            else:
                pos_prompt_text = self._get_qwen_default_prompts(context, is_initial_image).format(main_prompt=user_prompt)
            prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text
        else:
            # A style image is provided, so set the loader input.
            prompt[NODES['style_map_loader']]['inputs']['image'] = style_image_info['name']
            # External user-selected images can be arbitrarily large (10+ MP).
            # Scale them down to ~1 MP so the VAEEncode in the reference
            # latent chain doesn't blow up VRAM.  StableGen's own renders
            # are already roughly 1 MP and don't need this.
            if use_external_this_frame:
                scale_node_int = 600
                while str(scale_node_int) in prompt:
                    scale_node_int += 1
                scale_node_key = str(scale_node_int)
                prompt[scale_node_key] = {
                    "inputs": {
                        "upscale_method": "lanczos",
                        "megapixels": 1,
                        "resolution_steps": 1,
                        "image": [NODES['style_map_loader'], 0]
                    },
                    "class_type": "ImageScaleToTotalPixels",
                    "_meta": {
                        "title": "Scale External Style to 1MP"
                    }
                }
                # Point text-encode image2 at the scaled version so both
                # VL processing and the reference latent chain see ~1 MP.
                prompt[NODES['pos_prompt']]['inputs']['image2'] = [scale_node_key, 0]
                prompt[NODES['neg_prompt']]['inputs']['image2'] = [scale_node_key, 0]

        # --- Reference Latent chain (Qwen unzoom fix) ---
        self._build_qwen_reference_latent_chain(prompt, NODES)

        # --- Configure Sampler ---
        prompt[NODES['sampler']]['inputs']['seed'] = context.scene.seed
        prompt[NODES['sampler']]['inputs']['steps'] = context.scene.steps
        prompt[NODES['sampler']]['inputs']['cfg'] = context.scene.cfg
        prompt[NODES['sampler']]['inputs']['sampler_name'] = context.scene.sampler
        prompt[NODES['sampler']]['inputs']['scheduler'] = context.scene.scheduler
        prompt[NODES['sampler']]['inputs']['denoise'] = 1.0 # Typically 1.0 for this kind of edit

        # --- Set UNET model ---
        if is_nunchaku:
             prompt[NODES['unet_loader']] = {
                "inputs": {
                    "model_name": context.scene.model_name,
                    "cpu_offload": "auto",
                    "num_blocks_on_gpu": 1,
                    "use_pin_memory": "enable"
                },
                "class_type": "NunchakuQwenImageDiTLoader",
                "_meta": {
                    "title": "Nunchaku Qwen-Image DiT Loader"
                }
             }
        else:
            prompt[NODES['unet_loader']]['inputs']['unet_name'] = context.scene.model_name

        # --- Execute ---
        self._save_prompt_to_file(prompt, revision_dir)
        ws = self._connect_to_websocket(server_address, client_id)
        if ws is None:
            return {"error": "conn_failed"}

        images = None
        try:
            images = self._execute_prompt_and_get_images(ws, prompt, client_id, server_address, NODES)
        finally:
            if ws:
                ws.close()

        if images is None or (isinstance(images, dict) and "error" in images):
            return {"error": "conn_failed"}

        print(f"Qwen image generated with prompt: {prompt[NODES['pos_prompt']]['inputs']['prompt']}")
        return images[NODES['save_image']][0]
    

    def generate(self, context, controlnet_info=None, ipadapter_ref_info=None):
        """     
        Generates the image using ComfyUI.         
        :param context: Blender context.
        :param controlnet_info: Dict of uploaded controlnet image info.
        :param ipadapter_ref_info: Uploaded IPAdapter reference image info.
        :return: Generated image binary data.     
        """

        # Setup connection parameters
        server_address = context.preferences.addons[__package__].preferences.server_address
        client_id = str(uuid.uuid4())
        # Get revision dir for debug file
        revision_dir = get_generation_dirs(context)["revision"]

        # Initialize the prompt template and get node mappings
        prompt, NODES = self._create_base_prompt(context)
        
        # Set model resolution
        self._configure_resolution(prompt, context, NODES)

        if ipadapter_ref_info:
            # Configure IPAdapter settings
            self._configure_ipadapter(prompt, context, ipadapter_ref_info, NODES)
        else:
            # Remove IPAdapter nodes if not used
            for node_id in ['235', '236', '237']:
                if node_id in prompt:
                    del prompt[node_id]
        
        # Build controlnet chain
        prompt = self._build_controlnet_chain(prompt, context, controlnet_info, NODES)
        
        # Save prompt for debugging (in revision dir)
        self._save_prompt_to_file(prompt, revision_dir)

        # Execute generation and get results
        ws = self._connect_to_websocket(server_address, client_id)

        if ws is None:
            return {"error": "conn_failed"} # Connection error

        images = None
        try:
            images = self._execute_prompt_and_get_images(ws, prompt, client_id, server_address, NODES)
        finally:
            if ws:
                ws.close()

        if images is None or isinstance(images, dict) and "error" in images:
            return {"error": "conn_failed"}
        
        print(f"Image generated with prompt: {context.scene.comfyui_prompt}")
        
        # Return the generated image from the save_image node
        return images[NODES['save_image']][0]

    def _check_server_alive(self, server_address, timeout=None):
        """Return True if the ComfyUI server responds to a lightweight request."""
        if timeout is None:
            timeout = get_timeout('ping')
        try:
            req = urllib.request.Request(
                f"http://{server_address}/system_stats",
                method='GET'
            )
            urllib.request.urlopen(req, timeout=timeout)
            return True
        except Exception:
            return False

    def _get_vram_stats(self, server_address):
        """Return (vram_free, vram_total) in MB from ComfyUI /system_stats."""
        try:
            req = urllib.request.Request(
                f"http://{server_address}/system_stats", method='GET'
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=get_timeout('api')).read())
            dev = resp.get("devices", [{}])[0]
            vram_free = dev.get("vram_free", 0) / (1024 * 1024)
            vram_total = dev.get("vram_total", 0) / (1024 * 1024)
            return vram_free, vram_total
        except Exception:
            return None, None

    def _flush_comfyui_vram(self, server_address, retries=3, label="Post-generation"):
        """Unload all models and free VRAM on the ComfyUI server.

        The ``/free`` endpoint only **sets flags** on the prompt queue; the
        actual unloading happens asynchronously in ComfyUI's execution loop.
        This method therefore:
          1. Sends ``/free`` (unload models + free memory).
          2. Clears the execution cache history and queue.
          3. Sends ``/free`` *again* — the first round triggers model unloads
             whose CUDA tensors may still reside in PyTorch's cache; the
             second round triggers ``soft_empty_cache`` which calls
             ``torch.cuda.empty_cache()`` on the now-freed allocations.
          4. Polls ``/system_stats`` waiting for VRAM to increase.
        Retries the whole sequence up to *retries* times.
        """
        import time

        vram_before, vram_total = self._get_vram_stats(server_address)
        if vram_before is not None:
            print(f"[TRELLIS2] {label} VRAM before flush: {vram_before:.0f} MB free / {vram_total:.0f} MB total")

            # Skip the flush entirely if VRAM is already ≥90% free.
            if vram_total and vram_before > vram_total * 0.9:
                print(f"[TRELLIS2] {label} VRAM already ≥90% free — skipping flush.")
                return True

        def _send_free():
            d = json.dumps({"unload_models": True, "free_memory": True}).encode('utf-8')
            r = urllib.request.Request(
                f"http://{server_address}/free", data=d,
                headers={'Content-Type': 'application/json'}, method='POST'
            )
            urllib.request.urlopen(r, timeout=get_timeout('api'))

        def _clear_history():
            d = json.dumps({"clear": True}).encode('utf-8')
            r = urllib.request.Request(
                f"http://{server_address}/history", data=d,
                headers={'Content-Type': 'application/json'}, method='POST'
            )
            urllib.request.urlopen(r, timeout=get_timeout('api'))

        def _clear_queue():
            d = json.dumps({"clear": True}).encode('utf-8')
            r = urllib.request.Request(
                f"http://{server_address}/queue", data=d,
                headers={'Content-Type': 'application/json'}, method='POST'
            )
            urllib.request.urlopen(r, timeout=get_timeout('api'))

        for attempt in range(1, retries + 1):
            try:
                # Round 1: Unload models
                _send_free()
                time.sleep(1)

                # Clear caches
                try:
                    _clear_history()
                except Exception:
                    pass
                try:
                    _clear_queue()
                except Exception:
                    pass

                # Round 2: Free the now-released CUDA allocations
                _send_free()

                # Poll VRAM (up to 8s)
                for check in range(8):
                    time.sleep(1)
                    vram_after, _ = self._get_vram_stats(server_address)
                    if vram_after is not None:
                        print(f"[TRELLIS2] {label} VRAM after flush (check {check+1}): {vram_after:.0f} MB free")
                        if vram_before is not None and vram_after > vram_before + 500:
                            print(f"[TRELLIS2] {label} flush verified: freed {vram_after - vram_before:.0f} MB")
                            return True
                        if vram_total is not None and vram_after > vram_total * 0.7:
                            print(f"[TRELLIS2] {label} flush OK: >70% VRAM free")
                            return True

                print(f"[TRELLIS2] {label} flush attempt {attempt}/{retries}: VRAM didn't free enough — retrying")
            except Exception as e:
                print(f"[TRELLIS2] {label} flush attempt {attempt}/{retries} failed: {e}")

            if attempt < retries:
                time.sleep(2)

        print(f"[TRELLIS2] Warning: {label} VRAM flush unverified after {retries} attempts")
        return False

    # ------------------------------------------------------------------
    #  Txt2Img — lightweight single-image generation for prompt-to-image
    # ------------------------------------------------------------------

    def generate_txt2img(self, context, seed_override=None):
        """Generate a single image from the scene prompt and checkpoint.

        Builds a minimal txt2img ComfyUI workflow (with LoRA support) and
        returns the generated image as raw PNG bytes delivered via the
        ``SaveImageWebsocket`` node.

        Args:
            seed_override: If not None, overrides ``scene.seed`` in the
                built workflow.  Useful for preview-gallery generation from
                a background thread where ``scene.seed`` may not propagate.

        Returns:
            bytes: Raw image data on success.
            dict:  ``{"error": "..."}`` on failure.
        """
        server_address = context.preferences.addons[__package__].preferences.server_address
        client_id = str(uuid.uuid4())
        scene = context.scene
        architecture = scene.model_architecture  # synced from texture_mode

        # When using TRELLIS.2 with native/none texturing the backbone is
        # synced from trellis2_initial_image_arch instead of texture_mode.
        # Double-check here in case the property wasn't synced yet.
        if getattr(scene, 'architecture_mode', '') == 'trellis2':
            tex_mode = getattr(scene, 'trellis2_texture_mode', 'native')
            if tex_mode not in ('sdxl', 'flux1', 'qwen_image_edit'):
                architecture = getattr(scene, 'trellis2_initial_image_arch', 'sdxl')

        if architecture == 'qwen_image_edit':
            prompt, save_node = self._build_qwen_txt2img(context)
        elif architecture == 'flux1':
            prompt, save_node = self._build_flux_txt2img(context)
        else:
            prompt, save_node = self._build_sdxl_txt2img(context)

        # Patch seed if caller supplies an override (thread-safe)
        if seed_override is not None:
            for _node in prompt.values():
                _inp = _node.get('inputs', {})
                if 'seed' in _inp:
                    _inp['seed'] = seed_override
                if 'noise_seed' in _inp:
                    _inp['noise_seed'] = seed_override

        ws = self._connect_to_websocket(server_address, client_id)
        if ws is None:
            return {"error": "WebSocket connection failed"}

        # Let the operator close this WS on cancel
        if hasattr(self.operator, '_active_ws'):
            self.operator._active_ws = ws
            # Also set on class level so cancel from a new instance can find it
            type(self.operator)._active_ws = ws

        try:
            images = self._execute_prompt_and_get_images(
                ws, prompt, client_id, server_address, {"save_image": save_node}
            )
        finally:
            if hasattr(self.operator, '_active_ws'):
                self.operator._active_ws = None
                type(self.operator)._active_ws = None
            try:
                ws.close()
            except Exception:
                pass

        if images is None or not isinstance(images, dict) or not images:
            return {"error": "txt2img generation failed"}

        if save_node not in images or not images[save_node]:
            return {"error": "No image received from txt2img"}

        return images[save_node][0]

    def _build_sdxl_txt2img(self, context):
        """Return (prompt_dict, save_node_id) for a minimal SDXL txt2img with LoRA support."""
        scene = context.scene
        prompt = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": scene.model_name}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": scene.comfyui_prompt, "clip": ["1", 1]}
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": scene.comfyui_negative_prompt, "clip": ["1", 1]}
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": scene.seed if scene.seed != 0 else random.randint(0, 2**31),
                    "steps": scene.steps,
                    "cfg": scene.cfg,
                    "sampler_name": scene.sampler,
                    "scheduler": scene.scheduler,
                    "denoise": 1.0
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
            },
            "7": {
                "class_type": "SaveImageWebsocket",
                "inputs": {"images": ["6", 0]}
            }
        }

        # LoRA chain (LoraLoader): model + clip
        prompt, final_model, final_clip = self._build_lora_chain(
            prompt, context,
            initial_model_input=["1", 0],
            initial_clip_input=["1", 1],
            start_node_id=300,
            lora_class_type="LoraLoader"
        )
        prompt["5"]["inputs"]["model"] = final_model
        prompt["2"]["inputs"]["clip"] = final_clip
        prompt["3"]["inputs"]["clip"] = final_clip

        return prompt, "7"

    # ── PBR Decomposition (Marigold IID) ───────────────────────────────
    def generate_pbr_maps(self, context, input_image_path, model_name=None,
                          force_native_resolution=False):
        """Run a Marigold model on a generated texture and collect the output.

        Uploads *input_image_path* to the ComfyUI server, runs it through the
        ``MarigoldModelLoader`` + ``MarigoldDepthEstimation_v2`` pipeline,
        and collects the resulting map(s) via ``SaveImageWebsocket``.

        Different models produce different outputs:

        * **IID-Appearance** → 3 images: [Albedo, Roughness, Metallicity]
        * **IID-Lighting**   → 3 images: [Albedo, Shading, Residual]
        * **Normals**        → 1 image:  [Normal map]
        * **Depth**          → 1 image:  [Depth map]

        Args:
            input_image_path: Absolute path to the generated texture PNG.
            model_name: HuggingFace model repo name to use, e.g.
                ``'prs-eth/marigold-iid-appearance-v1-1'``.  If *None*,
                defaults to the IID-Appearance model.
            force_native_resolution: If *True*, always process at the
                image's native longest edge (rounded to 64) regardless
                of the user-configured resolution settings.  Used by
                the tiling path so each tile gets full-detail processing.

        Returns:
            list[bytes] | dict: A list of raw-PNG byte buffers on success,
                or ``{"error": "..."}`` on failure.
        """
        server_address = context.preferences.addons[__package__].preferences.server_address
        client_id = str(uuid.uuid4())
        scene = context.scene

        if model_name is None:
            model_name = 'prs-eth/marigold-iid-appearance-v1-1'

        # Pre-flight: verify that ComfyUI-Marigold nodes are installed
        required_nodes = ['MarigoldModelLoader', 'MarigoldDepthEstimation_v2']
        for node_class in required_nodes:
            try:
                resp = urllib.request.urlopen(
                    f"http://{server_address}/object_info/{node_class}",
                    timeout=get_timeout('api')
                )
                data = json.loads(resp.read())
                if node_class not in data:
                    return {"error": f"ComfyUI-Marigold node '{node_class}' not found. "
                                     f"Install ComfyUI-Marigold (run installer option 9 "
                                     f"or git clone https://github.com/kijai/ComfyUI-Marigold "
                                     f"into ComfyUI/custom_nodes/) and ensure 'diffusers>=0.28' "
                                     f"is installed in ComfyUI's Python, then restart ComfyUI."}
            except Exception:
                return {"error": f"ComfyUI-Marigold node '{node_class}' not found. "
                                 f"Install ComfyUI-Marigold (run installer option 9 "
                                 f"or git clone https://github.com/kijai/ComfyUI-Marigold "
                                 f"into ComfyUI/custom_nodes/) and ensure 'diffusers>=0.28' "
                                 f"is installed in ComfyUI's Python, then restart ComfyUI."}

        # Upload the source image to ComfyUI's input folder
        from .generator import upload_image_to_comfyui
        image_info = upload_image_to_comfyui(server_address, input_image_path)
        if image_info is None:
            return {"error": f"Failed to upload image for PBR decomposition: {input_image_path}"}

        uploaded_name = image_info.get("name", os.path.basename(input_image_path))

        # Determine the processing resolution.
        # When 'Native Resolution' is enabled (or forced by tiling), use
        # the image's own longest edge (rounded to 64) so Marigold
        # processes at full detail.
        # Otherwise fall back to the user-specified fixed resolution.
        use_native = force_native_resolution or getattr(
            scene, 'pbr_use_native_resolution', False)
        if use_native:
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(input_image_path) as _img:
                    _w, _h = _img.size
                proc_res = max(_w, _h)
                proc_res = ((proc_res + 63) // 64) * 64
                print(f"[StableGen] Marigold: native resolution {proc_res}px "
                      f"(input {_w}\u00d7{_h})")
            except Exception:
                proc_res = scene.pbr_processing_resolution
        else:
            proc_res = scene.pbr_processing_resolution

        # Build the minimal ComfyUI workflow
        prompt = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_name}
            },
            "2": {
                "class_type": "MarigoldModelLoader",
                "inputs": {
                    "model": model_name,
                }
            },
            "3": {
                "class_type": "MarigoldDepthEstimation_v2",
                "inputs": {
                    "marigold_model": ["2", 0],
                    "image": ["1", 0],
                    "seed": scene.seed if scene.seed != 0 else random.randint(0, 2**31),
                    "denoise_steps": scene.pbr_denoise_steps,
                    "ensemble_size": scene.pbr_ensemble_size,
                    "processing_resolution": proc_res,
                    "scheduler": "LCMScheduler",
                    "use_taesd_vae": False,
                    # Keep loaded to avoid offloading to CPU — newer diffusers
                    # raises ValueError when moving a float16 pipeline to CPU.
                    "keep_model_loaded": True,
                }
            },
            "4": {
                "class_type": "SaveImageWebsocket",
                "inputs": {"images": ["3", 0]}
            }
        }

        NODES = {"save_image": "4"}

        ws = self._connect_to_websocket(server_address, client_id)
        if ws is None:
            return {"error": "WebSocket connection failed for PBR decomposition"}

        if hasattr(self.operator, '_active_ws'):
            self.operator._active_ws = ws
            type(self.operator)._active_ws = ws

        try:
            images = self._execute_prompt_and_get_images(
                ws, prompt, client_id, server_address, NODES
            )
        finally:
            if hasattr(self.operator, '_active_ws'):
                self.operator._active_ws = None
                type(self.operator)._active_ws = None
            try:
                ws.close()
            except Exception:
                pass

        if images is None or not isinstance(images, dict) or not images:
            return {"error": "PBR decomposition failed — no output received"}

        save_node = NODES['save_image']
        if save_node not in images or not images[save_node]:
            return {"error": "PBR decomposition failed — no images from save node"}

        pbr_maps = images[save_node]  # list of raw PNG bytes
        print(f"[StableGen] Marigold '{model_name}' returned {len(pbr_maps)} image(s)")

        return pbr_maps

    def generate_delight_map(self, context, input_image_path):
        """Run StableDelight on an image to produce a specular-free (delighted) version.

        Uses the ``LoadStableDelightModel`` + ``ApplyStableDelight`` nodes
        from the *ComfyUI_StableDelight_ll* custom-node package.

        The delighted image preserves diffuse shading and texture detail
        while removing specular highlights — making it a high-quality
        alternative to Marigold IID's flat albedo for the Base Color slot.

        Args:
            input_image_path: Absolute path to the generated texture PNG.

        Returns:
            bytes | dict: Raw PNG bytes of the delighted image on success,
                or ``{"error": "..."}`` on failure.
        """
        server_address = context.preferences.addons[__package__].preferences.server_address
        client_id = str(uuid.uuid4())
        scene = context.scene

        # Pre-flight: verify that StableDelight nodes are installed
        required_nodes = ['LoadStableDelightModel', 'ApplyStableDelight']
        for node_class in required_nodes:
            try:
                resp = urllib.request.urlopen(
                    f"http://{server_address}/object_info/{node_class}",
                    timeout=get_timeout('api')
                )
                data = json.loads(resp.read())
                if node_class not in data:
                    return {"error": f"StableDelight node '{node_class}' not found. "
                                     f"Install ComfyUI_StableDelight_ll (run installer "
                                     f"option 10 or git clone "
                                     f"https://github.com/lldacing/ComfyUI_StableDelight_ll "
                                     f"into ComfyUI/custom_nodes/) and download the model "
                                     f"'Stable-X/yoso-delight-v0-4-base' into "
                                     f"ComfyUI/models/diffusers/, then restart ComfyUI."}
            except Exception:
                return {"error": f"StableDelight node '{node_class}' not found. "
                                 f"Install ComfyUI_StableDelight_ll (run installer "
                                 f"option 10 or git clone "
                                 f"https://github.com/lldacing/ComfyUI_StableDelight_ll "
                                 f"into ComfyUI/custom_nodes/) and download the model "
                                 f"'Stable-X/yoso-delight-v0-4-base' into "
                                 f"ComfyUI/models/diffusers/, then restart ComfyUI."}

        # Resolve model path — the node expects a relative path under
        # ComfyUI/models/diffusers/ containing model_index.json.
        delight_model_path = "Stable-X--yoso-delight-v0-4-base"

        # Upload the source image to ComfyUI's input folder
        from .generator import upload_image_to_comfyui
        image_info = upload_image_to_comfyui(server_address, input_image_path)
        if image_info is None:
            return {"error": f"Failed to upload image for StableDelight: {input_image_path}"}

        uploaded_name = image_info.get("name", os.path.basename(input_image_path))

        # Determine the processing resolution.
        #
        # The official Stable-X Predictor uses processing_resolution=2048,
        # while the ComfyUI node defaults to 1024.  Both use
        # MarigoldImageProcessor.preprocess() → resize_to_max_edge() which
        # scales the image so its *longest edge* equals processing_resolution,
        # then pads to a multiple of vae_scale_factor (8).
        #
        # Problems with a fixed resolution:
        # - Too small (e.g. 1024): images larger than 1024px get downscaled
        #   then bilinearly upscaled back → blurry output.
        # - Too large: small images get massively upscaled → wastes VRAM
        #   and can exceed available memory.
        # - resolution=0: triggers skip_preprocess=True in the node, which
        #   has a 3D/4D tensor bug in the pipeline's VAE encoder.
        #
        # Solution: use max(native_longest_edge, 1024).  This ensures:
        # - Large images (>1024) are processed at native res (no downscale).
        # - Small images (<1024) are upscaled to 1024 — giving the SD1.5-
        #   based model enough spatial tokens to produce clean output.
        # The value is rounded up to the nearest multiple of 64 (not just 8)
        # to match the official Stable-X resize_image() convention, which
        # aligns to 64 for clean integer dimensions through every UNet level
        # (VAE /8, then 3 downsampling stages at /2 each = 64× total).
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(input_image_path) as _img:
                _w, _h = _img.size
            # Floor: at least 1024 (the model's ComfyUI default)
            # Ceiling: native resolution (avoid downscaling)
            proc_res = max(max(_w, _h), 1024)
            # Round up to next multiple of 64 (official alignment)
            proc_res = ((proc_res + 63) // 64) * 64
            print(f"[StableGen] StableDelight: processing at {proc_res}px "
                  f"(input {_w}×{_h})")
        except Exception:
            proc_res = 1024  # safe fallback

        # Build the ComfyUI workflow
        prompt = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_name}
            },
            "2": {
                "class_type": "LoadStableDelightModel",
                "inputs": {
                    "model": delight_model_path,
                    "device": "AUTO",
                }
            },
            "3": {
                "class_type": "ApplyStableDelight",
                "inputs": {
                    "model": ["2", 0],
                    "images": ["1", 0],
                    "strength": getattr(scene, 'pbr_delight_strength', 1.0),
                    "resolution": proc_res,
                    "upscale_method": "bilinear",
                }
            },
            "4": {
                "class_type": "SaveImageWebsocket",
                "inputs": {"images": ["3", 0]}
            }
        }

        NODES = {"save_image": "4"}

        ws = self._connect_to_websocket(server_address, client_id)
        if ws is None:
            return {"error": "WebSocket connection failed for StableDelight"}

        if hasattr(self.operator, '_active_ws'):
            self.operator._active_ws = ws
            type(self.operator)._active_ws = ws

        try:
            images = self._execute_prompt_and_get_images(
                ws, prompt, client_id, server_address, NODES
            )
        finally:
            if hasattr(self.operator, '_active_ws'):
                self.operator._active_ws = None
                type(self.operator)._active_ws = None
            try:
                ws.close()
            except Exception:
                pass

        if images is None or not isinstance(images, dict) or not images:
            return {"error": "StableDelight failed — no output received"}

        save_node = NODES['save_image']
        if save_node not in images or not images[save_node]:
            return {"error": "StableDelight failed — no images from save node"}

        delight_images = images[save_node]
        print(f"[StableGen] StableDelight returned {len(delight_images)} image(s)")

        # StableDelight returns a single image
        return delight_images[0] if delight_images else {"error": "StableDelight returned empty"}

    def _build_flux_txt2img(self, context):
        """Return (prompt_dict, save_node_id) for a minimal Flux txt2img with LoRA support."""
        scene = context.scene
        unet_name = scene.model_name
        is_gguf = ".gguf" in unet_name.lower()
        unet_loader_class = "UnetLoaderGGUF" if is_gguf else "UNETLoader"

        prompt = {
            "10": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ae.sft"}
            },
            "11": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                    "clip_name2": "clip_l.safetensors",
                    "type": "flux",
                    "device": "default"
                }
            },
            "12": {
                "class_type": unet_loader_class,
                "inputs": {"unet_name": unet_name}
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": scene.comfyui_prompt, "clip": ["11", 0]}
            },
            "26": {
                "class_type": "FluxGuidance",
                "inputs": {"guidance": scene.cfg, "conditioning": ["6", 0]}
            },
            "25": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": scene.seed if scene.seed != 0 else random.randint(0, 2**31)}
            },
            "16": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": scene.sampler}
            },
            "17": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "scheduler": scene.scheduler,
                    "steps": scene.steps,
                    "denoise": 1.0,
                    "model": ["12", 0]
                }
            },
            "22": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["12", 0], "conditioning": ["26", 0]}
            },
            "30": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
            },
            "13": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["25", 0],
                    "guider": ["22", 0],
                    "sampler": ["16", 0],
                    "sigmas": ["17", 0],
                    "latent_image": ["30", 0]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["13", 0], "vae": ["10", 0]}
            },
            "7": {
                "class_type": "SaveImageWebsocket",
                "inputs": {"images": ["8", 0]}
            }
        }

        # LoRA chain (LoraLoaderModelOnly): model only, no clip
        prompt, final_model, _ = self._build_lora_chain(
            prompt, context,
            initial_model_input=["12", 0],
            initial_clip_input=["12", 0],   # dummy, not used by LoraLoaderModelOnly
            start_node_id=300,
            lora_class_type="LoraLoaderModelOnly"
        )
        prompt["17"]["inputs"]["model"] = final_model
        prompt["22"]["inputs"]["model"] = final_model

        return prompt, "7"

    def _build_qwen_txt2img(self, context):
        """Return (prompt_dict, save_node_id) for a minimal Qwen txt2img.

        Qwen-Image-Edit can generate images from text alone when no input
        images are provided to the ``TextEncodeQwenImageEditPlus`` nodes.
        """
        scene = context.scene
        unet_name = scene.model_name
        is_gguf = ".gguf" in unet_name.lower()
        unet_loader_class = "UnetLoaderGGUF" if is_gguf else "UNETLoader"

        prompt = {
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                    "type": "qwen_image",
                    "device": "default"
                }
            },
            "4": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "qwen_image_vae.safetensors"}
            },
            "13": {
                "class_type": unet_loader_class,
                "inputs": {"unet_name": unet_name}
            },
            "6": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {"shift": 3, "model": ["13", 0]}
            },
            "9": {
                "class_type": "CFGNorm",
                "inputs": {"strength": 1, "model": ["6", 0]}
            },
            "12": {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {
                    "prompt": scene.comfyui_prompt,
                    "clip": ["3", 0]
                }
            },
            "11": {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {
                    "prompt": "",
                    "clip": ["3", 0]
                }
            },
            "30": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
            },
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["9", 0],
                    "positive": ["12", 0],
                    "negative": ["11", 0],
                    "latent_image": ["30", 0],
                    "seed": scene.seed if scene.seed != 0 else random.randint(0, 2**31),
                    "steps": scene.steps,
                    "cfg": scene.cfg,
                    "sampler_name": scene.sampler,
                    "scheduler": scene.scheduler,
                    "denoise": 1.0
                }
            },
            "2": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["1", 0], "vae": ["4", 0]}
            },
            "7": {
                "class_type": "SaveImageWebsocket",
                "inputs": {"images": ["2", 0]}
            }
        }

        # LoRA chain: Qwen uses NunchakuQwenImageLoraLoader for nunchaku
        # models, LoraLoaderModelOnly otherwise
        is_nunchaku = unet_name.lower().endswith('.safetensors')
        lora_class = "NunchakuQwenImageLoraLoader" if is_nunchaku else "LoraLoaderModelOnly"

        prompt, final_model, _ = self._build_lora_chain(
            prompt, context,
            initial_model_input=["13", 0],
            initial_clip_input=["13", 0],   # dummy
            start_node_id=500,
            lora_class_type=lora_class
        )
        # ModelSamplingAuraFlow takes the final LoRA output
        prompt["6"]["inputs"]["model"] = final_model

        return prompt, "7"

    # ── GLB mesh validation ───────────────────────────────────────────
    _GLB_VERTEX_THRESHOLD = 10.0    # absolute coordinate cap
    _GLB_OUTLIER_SIGMA   = 8.0     # for absolute-value outlier check
    _GLB_LAPLACIAN_SIGMA = 8.0    # fallback — overridden by scene property
    _GLB_SPIKE_FRACTION  = 0.002  # 0.02% — very tight because at 10σ virtually all
                                    # flagged vertices are genuine artifacts
    _GLB_SPIKE_ABS_MAX   = 25      # fallback — overridden by scene property

    @staticmethod
    def _validate_glb_mesh(glb_bytes, threshold=None, sigma=None):
        """
        GLB mesh integrity check that catches both extreme values and
        subtle spiky artifacts (the cumesh/Dual Contouring Windows bug).

        Three-layer validation:
        1. **Hard limits** – NaN, Inf, absolute coordinate > *threshold*.
        2. **Statistical outliers** – coords beyond *sigma* σ from mean.
        3. **Laplacian smoothness** – for each vertex, measure displacement
           from the average of its mesh neighbors.  Vertices displaced by
           more than ``_GLB_LAPLACIAN_SIGMA`` σ of the mesh-wide Laplacian
           distribution are flagged.  If more than ``_GLB_SPIKE_FRACTION``
           of all vertices are flagged, the mesh is considered corrupt.

        Returns ``(True, "")`` on success or ``(False, reason)`` on failure.
        """
        if threshold is None:
            threshold = WorkflowManager._GLB_VERTEX_THRESHOLD
        if sigma is None:
            sigma = WorkflowManager._GLB_OUTLIER_SIGMA
        # Read user-facing settings if available, else use class defaults
        _scene = getattr(bpy.context, 'scene', None)
        lap_sigma = getattr(_scene, 'trellis2_artifact_laplacian_sigma',
                            WorkflowManager._GLB_LAPLACIAN_SIGMA)
        spike_frac = WorkflowManager._GLB_SPIKE_FRACTION

        if not glb_bytes or len(glb_bytes) < 20:
            return False, "GLB data too small"

        try:
            # ── Parse GLB container ──────────────────────────────────
            magic, version, total_len = struct.unpack_from('<III', glb_bytes, 0)
            if magic != 0x46546C67:  # 'glTF'
                return False, "Not a valid GLB (bad magic)"

            json_len, json_type = struct.unpack_from('<II', glb_bytes, 12)
            if json_type != 0x4E4F534A:  # 'JSON'
                return False, "First GLB chunk is not JSON"
            gltf = json.loads(glb_bytes[20:20 + json_len])

            bin_offset = 20 + json_len
            if bin_offset % 4:
                bin_offset += 4 - (bin_offset % 4)
            if bin_offset + 8 > len(glb_bytes):
                return False, "No binary chunk found"
            bin_len, bin_type = struct.unpack_from('<II', glb_bytes, bin_offset)
            bin_data = glb_bytes[bin_offset + 8: bin_offset + 8 + bin_len]

            accessors = gltf.get('accessors', [])
            buffer_views = gltf.get('bufferViews', [])
            meshes = gltf.get('meshes', [])

            # ── Collect all primitives (position accessor + index accessor)
            primitives_info = []
            for mesh in meshes:
                for prim in mesh.get('primitives', []):
                    pos_idx = prim.get('attributes', {}).get('POSITION')
                    idx_idx = prim.get('indices')
                    if pos_idx is not None:
                        primitives_info.append((pos_idx, idx_idx))

            if not primitives_info:
                return False, "No POSITION accessor found in GLB"

            total_verts = 0
            total_spikes = 0

            for pos_acc_idx, idx_acc_idx in primitives_info:
                # ── Read vertex positions ────────────────────────────
                acc = accessors[pos_acc_idx]
                bv = buffer_views[acc['bufferView']]
                off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
                count = acc['count']
                total_verts += count
                num_floats = count * 3
                end = off + num_floats * 4
                if end > len(bin_data):
                    return False, (f"POSITION data overflows binary chunk "
                                   f"(need {end}, have {len(bin_data)})")

                raw = struct.unpack_from(f'<{num_floats}f', bin_data, off)

                # --- Layer 1: hard checks ---
                for i, v in enumerate(raw):
                    if v != v:
                        return False, f"NaN vertex at float index {i}"
                    if abs(v) == float('inf'):
                        return False, f"Inf vertex at float index {i}"
                    if abs(v) > threshold:
                        return False, (f"Extreme vertex {v:.2e} at float "
                                       f"index {i} (threshold {threshold})")

                # --- Layer 2: absolute statistical outlier ---
                if count >= 10:
                    mean_v = sum(raw) / len(raw)
                    var_v = sum((v - mean_v) ** 2 for v in raw) / len(raw)
                    std_v = var_v ** 0.5
                    if std_v > 1e-9:
                        for i, v in enumerate(raw):
                            dev = abs(v - mean_v) / std_v
                            if dev > sigma:
                                return False, (
                                    f"Outlier vertex: {v:.4f} is {dev:.1f}σ "
                                    f"from mean {mean_v:.4f} at float idx {i}")

                # --- Layer 3: Laplacian smoothness (needs indices) ----
                if idx_acc_idx is None or count < 10:
                    continue  # can't do topology check without indices

                # Build positions list as (x, y, z) tuples
                positions = [(raw[i*3], raw[i*3+1], raw[i*3+2])
                             for i in range(count)]

                # Read triangle indices
                idx_acc = accessors[idx_acc_idx]
                idx_bv = buffer_views[idx_acc['bufferView']]
                idx_off = idx_bv.get('byteOffset', 0) + idx_acc.get('byteOffset', 0)
                idx_count = idx_acc['count']
                comp_type = idx_acc.get('componentType', 5123)
                # 5121=UNSIGNED_BYTE, 5123=UNSIGNED_SHORT, 5125=UNSIGNED_INT
                if comp_type == 5121:
                    fmt, sz = 'B', 1
                elif comp_type == 5123:
                    fmt, sz = 'H', 2
                else:  # 5125
                    fmt, sz = 'I', 4

                idx_end = idx_off + idx_count * sz
                if idx_end > len(bin_data):
                    continue  # skip Laplacian if index data is bad
                indices = struct.unpack_from(
                    f'<{idx_count}{fmt}', bin_data, idx_off)

                # Build neighbor sets from triangles
                neighbors = [set() for _ in range(count)]
                for t in range(0, idx_count, 3):
                    if t + 2 >= idx_count:
                        break
                    a, b, c = indices[t], indices[t+1], indices[t+2]
                    if a < count and b < count and c < count:
                        neighbors[a].update((b, c))
                        neighbors[b].update((a, c))
                        neighbors[c].update((a, b))

                # Compute Laplacian displacement magnitudes
                laplacians = []
                for vi in range(count):
                    nbrs = neighbors[vi]
                    if not nbrs:
                        continue
                    px, py, pz = positions[vi]
                    ax = sum(positions[n][0] for n in nbrs) / len(nbrs)
                    ay = sum(positions[n][1] for n in nbrs) / len(nbrs)
                    az = sum(positions[n][2] for n in nbrs) / len(nbrs)
                    dx, dy, dz = px - ax, py - ay, pz - az
                    laplacians.append((dx*dx + dy*dy + dz*dz) ** 0.5)

                if len(laplacians) < 10:
                    continue

                lap_mean = sum(laplacians) / len(laplacians)
                lap_var = sum((v - lap_mean)**2 for v in laplacians) / len(laplacians)
                lap_std = lap_var ** 0.5

                if lap_std < 1e-12:
                    continue  # perfectly smooth mesh, nothing to flag

                spike_count = sum(
                    1 for v in laplacians
                    if (v - lap_mean) / lap_std > lap_sigma
                )
                total_spikes += spike_count

            # Check overall spike count (absolute cap + fraction)
            if total_verts > 0 and total_spikes > 0:
                frac = total_spikes / total_verts
                abs_max = getattr(_scene, 'trellis2_artifact_spike_abs_max',
                                  WorkflowManager._GLB_SPIKE_ABS_MAX)
                if total_spikes > abs_max:
                    return False, (
                        f"Mesh has {total_spikes} Laplacian-outlier vertices "
                        f"(exceeds absolute cap of {abs_max}). "
                        f"Likely cumesh corruption."
                    )
                if frac > spike_frac:
                    return False, (
                        f"Mesh has {total_spikes} Laplacian-outlier vertices "
                        f"({frac*100:.2f}% of {total_verts} — threshold "
                        f"{spike_frac*100:.2f}%). Likely cumesh corruption."
                    )

            print(f"[TRELLIS2] Mesh validation passed "
                  f"({total_verts} verts, {total_spikes} minor spikes).")
            return True, ""
        except (struct.error, json.JSONDecodeError, KeyError, IndexError) as e:
            return False, f"GLB parse error: {e}"

    @staticmethod
    def _is_local_server(server_address):
        """Return True if server_address points to localhost."""
        host = server_address.split(':')[0].strip()
        return host in ('127.0.0.1', 'localhost', '0.0.0.0', '::1', '')

    @staticmethod
    def _clear_triton_cache():
        """
        Delete the Triton JIT kernel cache.
        On Windows this lives at ``%USERPROFILE%\\.triton\\cache``.
        Returns True if anything was deleted.
        """
        import shutil
        import pathlib

        candidates = []
        # Standard location
        home = pathlib.Path.home()
        candidates.append(home / '.triton' / 'cache')
        # TRITON_CACHE_DIR env override
        env_dir = os.environ.get('TRITON_CACHE_DIR')
        if env_dir:
            candidates.append(pathlib.Path(env_dir))

        cleared = False
        for cache_dir in candidates:
            if cache_dir.is_dir():
                try:
                    count = sum(1 for _ in cache_dir.rglob('*') if _.is_file())
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    print(f"[TRELLIS2] Cleared Triton cache: {cache_dir} "
                          f"({count} files)")
                    cleared = True
                except Exception as e:
                    print(f"[TRELLIS2] Failed to clear Triton cache {cache_dir}: {e}")
        return cleared

    def _reboot_comfyui(self, server_address, poll_timeout=None):
        """
        Attempt to reboot ComfyUI via the ComfyUI Manager ``/manager/reboot``
        endpoint, then wait for the server to come back online.

        Returns True if the server rebooted and is reachable again.
        Returns False if ComfyUI Manager is not installed (404), the request
        failed, or the server did not come back within *poll_timeout* seconds.
        """
        import time

        if poll_timeout is None:
            poll_timeout = get_timeout('reboot')

        reboot_url = f"http://{server_address}/manager/reboot"
        print(f"[TRELLIS2] Requesting ComfyUI reboot via {reboot_url} ...")

        try:
            req = urllib.request.Request(reboot_url, method='GET')
            resp = urllib.request.urlopen(req, timeout=get_timeout('api'))
            status = resp.getcode()
            if status not in (200, 201, 204):
                print(f"[TRELLIS2] Reboot endpoint returned unexpected status {status}")
                return False
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"[TRELLIS2] Reboot endpoint not found (HTTP 404). "
                      f"ComfyUI Manager may not be installed.")
                return False
            # Other HTTP errors (5xx etc.) might mean the server is already
            # shutting down — treat as potentially successful.
            print(f"[TRELLIS2] Reboot endpoint returned HTTP {e.code}, "
                  f"proceeding to poll (server may be restarting)...")
        except Exception as e:
            # Connection reset / broken pipe / timeout are EXPECTED when the
            # server process dies mid-response.  Treat as likely success.
            print(f"[TRELLIS2] Connection dropped after reboot request "
                  f"({type(e).__name__}: {e}) — server is likely restarting.")

        print("[TRELLIS2] Reboot request sent. Waiting for server to go down...")

        # Phase 1: Wait for the server to actually go down (up to 15s).
        # If it never goes down, the reboot may not have taken effect.
        down_deadline = time.time() + 15
        went_down = False
        time.sleep(2)  # give the process a moment to start shutting down
        while time.time() < down_deadline:
            if not self._check_server_alive(server_address, timeout=2):
                went_down = True
                break
            time.sleep(1)

        if not went_down:
            # Server never went down — maybe the reboot was a no-op
            print("[TRELLIS2] Server did not appear to go down. "
                  "Continuing anyway (it may have rebooted very quickly).")

        # Phase 2: Poll until the server is back up.
        print("[TRELLIS2] Waiting for server to come back up...")
        up_deadline = time.time() + poll_timeout
        while time.time() < up_deadline:
            if self._check_server_alive(server_address, timeout=3):
                print("[TRELLIS2] Server is back online!")
                # Give it a few extra seconds to finish loading custom nodes
                time.sleep(5)
                return True
            time.sleep(3)

        print(f"[TRELLIS2] Server did not come back within {poll_timeout}s.")
        return False

    def generate_trellis2(self, context, input_image_path):
        """
        Generates a 3D mesh using TRELLIS.2 via ComfyUI.

        Uploads an input image, runs the TRELLIS.2 pipeline (background removal,
        conditioning, shape generation, optionally texture generation, GLB export),
        and downloads the resulting GLB file from the ComfyUI server.
        VRAM is always flushed before AND after generation.

        If the resulting mesh contains degenerate vertices (a known cumesh /
        Dual Contouring issue on Windows, caused by corrupted Triton JIT
        cache), the addon will:

        1. Clear the Triton cache (if the server is local).
        2. Reboot ComfyUI via the ComfyUI Manager ``/manager/reboot``
           endpoint (works for both local and remote servers).
        3. Run one final generation attempt with a fresh process.

        Args:
            context: Blender context.
            input_image_path: Local path to the input image file.

        Returns:
            bytes: GLB file binary data on success.
            dict: {"error": "message"} on failure.
        """
        import urllib.parse
        import time

        server_address = context.preferences.addons[__package__].preferences.server_address

        # Pre-generation flush: free any loaded diffusion/other models so
        # TRELLIS.2 has maximum VRAM available.
        print("[TRELLIS2] Pre-generation VRAM flush — freeing loaded models...")
        self._flush_comfyui_vram(server_address, label="Pre-generation")

        time.sleep(1)  # Brief pause for CUDA memory to be released

        # Verify the server is alive before proceeding
        if not self._check_server_alive(server_address):
            return {"error": "ComfyUI server is not responding. Please restart it and try again."}

        is_local = self._is_local_server(server_address)
        original_seed = context.scene.trellis2_seed
        max_retries = getattr(context.scene, 'trellis2_artifact_max_retries', 1)

        try:
            # ── First attempt ──────────────────────────────────────────
            client_id = str(uuid.uuid4())
            result = self._generate_trellis2_inner(
                context, input_image_path, server_address, client_id)

            if isinstance(result, dict) and "error" in result:
                return result

            ok, reason = self._validate_glb_mesh(result)
            if ok:
                return result

            # Mesh is corrupt — Triton JIT cache is the most common cause.
            print(f"[TRELLIS2] Mesh validation FAILED: {reason}")

            if max_retries <= 0:
                print("[TRELLIS2] Artifact retries disabled (max_retries=0).")
            else:
                print(f"[TRELLIS2] Attempting automatic recovery "
                      f"(up to {max_retries} retries, Triton cache clear + ComfyUI reboot)...")

            rebooted = False
            for retry_i in range(max_retries):
                # ── Recovery: clear cache + reboot ─────────────────────
                if is_local:
                    self._clear_triton_cache()

                rebooted = self._reboot_comfyui(server_address)

                if not rebooted:
                    print(f"[TRELLIS2] Could not reboot ComfyUI — aborting retries.")
                    break

                print(f"[TRELLIS2] Retry {retry_i + 1}/{max_retries} after reboot...")
                new_seed = random.randint(0, 2**31 - 1)
                context.scene.trellis2_seed = new_seed
                client_id = str(uuid.uuid4())
                result = self._generate_trellis2_inner(
                    context, input_image_path, server_address, client_id)

                if isinstance(result, dict) and "error" in result:
                    return result

                ok, reason = self._validate_glb_mesh(result)
                if ok:
                    return result

                print(f"[TRELLIS2] Retry {retry_i + 1}/{max_retries} also failed: {reason}")

            # ── Truly exhausted ────────────────────────────────────────
            if is_local:
                warning_msg = (
                    "Mesh has artifacts (corrupt Triton JIT cache). "
                    f"Cache was cleared and ComfyUI was "
                    f"{'rebooted' if rebooted else 'NOT rebooted (install ComfyUI Manager for auto-reboot)'}. "
                    "Try manually restarting ComfyUI."
                )
            else:
                warning_msg = (
                    "Mesh has artifacts (corrupt Triton JIT cache). "
                    f"{'ComfyUI was rebooted but issue persists.' if rebooted else 'Could not auto-reboot (ComfyUI Manager not installed?).'} "
                    "On the ComfyUI host, delete C:\\Users\\<USERNAME>\\.triton\\cache "
                    "and restart ComfyUI."
                )
            print(f"[TRELLIS2] WARNING: {warning_msg}")
            self.operator._warning = warning_msg
            return result
        finally:
            # Restore the original seed so the UI isn't silently changed
            context.scene.trellis2_seed = original_seed
            self._flush_comfyui_vram(server_address, label="Post-generation")

    def _generate_trellis2_inner(self, context, input_image_path, server_address, client_id):
        """Inner implementation of generate_trellis2, called within a try/finally VRAM flush."""
        import urllib.parse

        # Upload the input image to ComfyUI
        from .generator import upload_image_to_comfyui
        image_info = upload_image_to_comfyui(server_address, input_image_path)
        if image_info is None:
            self.operator._error = f"Failed to upload input image: {input_image_path}"
            return {"error": self.operator._error}

        scene = context.scene
        skip_texture = scene.trellis2_skip_texture

        # Load the appropriate workflow template
        if skip_texture:
            prompt = json.loads(prompt_text_trellis2_shape_only)
            NODES = {
                'input_image': '1',
                'load_models': '2',
                'remove_bg': '3',
                'get_conditioning': '4',
                'image_to_shape': '5',
                'simplify': '6',
                'export_trimesh': '7',
            }
            export_node_key = 'export_trimesh'
        else:
            prompt = json.loads(prompt_text_trellis2)
            NODES = {
                'input_image': '1',
                'load_models': '2',
                'remove_bg': '3',
                'get_conditioning': '4',
                'image_to_shape': '5',
                'shape_to_textured_mesh': '6',
                'export_glb': '7',
            }
            export_node_key = 'export_glb'

        # Set input image
        prompt[NODES['input_image']]["inputs"]["image"] = image_info['name']

        # Configure model settings from scene properties
        prompt[NODES['load_models']]["inputs"]["resolution"] = scene.trellis2_resolution
        prompt[NODES['load_models']]["inputs"]["vram_mode"] = scene.trellis2_vram_mode
        prompt[NODES['load_models']]["inputs"]["attn_backend"] = scene.trellis2_attn_backend

        # Configure background removal (or bypass it)
        skip_bg = getattr(scene, 'trellis2_bg_removal', 'auto') == 'skip'
        if skip_bg:
            # Remove the RemoveBackground node from the workflow.
            # Wire GetConditioning to use LoadImage outputs directly:
            #   LoadImage[0] → image,  LoadImage[1] → mask (alpha channel).
            # If the input has no alpha, ComfyUI's LoadImage provides an
            # all-white mask (= entire image is foreground).
            remove_bg_node_id = NODES.pop('remove_bg')
            del prompt[remove_bg_node_id]
            prompt[NODES['get_conditioning']]["inputs"]["image"] = [NODES['input_image'], 0]
            prompt[NODES['get_conditioning']]["inputs"]["mask"] = [NODES['input_image'], 1]
        else:
            prompt[NODES['remove_bg']]["inputs"]["low_vram"] = scene.trellis2_low_vram

        # Configure conditioning
        prompt[NODES['get_conditioning']]["inputs"]["background_color"] = scene.trellis2_background_color
        # Auto-determine whether 1024 conditioning is needed from resolution mode
        prompt[NODES['get_conditioning']]["inputs"]["include_1024"] = scene.trellis2_resolution != '512'

        # Configure shape generation
        seed = scene.trellis2_seed
        prompt[NODES['image_to_shape']]["inputs"]["seed"] = seed
        prompt[NODES['image_to_shape']]["inputs"]["ss_guidance_strength"] = scene.trellis2_ss_guidance
        prompt[NODES['image_to_shape']]["inputs"]["ss_sampling_steps"] = scene.trellis2_ss_steps
        prompt[NODES['image_to_shape']]["inputs"]["shape_guidance_strength"] = scene.trellis2_shape_guidance
        prompt[NODES['image_to_shape']]["inputs"]["shape_sampling_steps"] = scene.trellis2_shape_steps
        prompt[NODES['image_to_shape']]["inputs"]["max_tokens"] = scene.trellis2_max_tokens

        # Configure export with unique prefix for identification
        unique_prefix = f"trellis2_{uuid.uuid4().hex[:8]}"

        # When post-processing is disabled, maximize the decimation target
        # and disable remeshing so the user gets the rawest possible mesh.
        use_pp = getattr(scene, 'trellis2_post_processing_enabled', True)

        if skip_texture:
            # Configure simplify node
            if use_pp:
                prompt[NODES['simplify']]["inputs"]["target_face_count"] = scene.trellis2_decimation
                prompt[NODES['simplify']]["inputs"]["remesh"] = scene.trellis2_remesh
            else:
                # Raw mesh — max faces, no remesh
                prompt[NODES['simplify']]["inputs"]["target_face_count"] = 5000000
                prompt[NODES['simplify']]["inputs"]["remesh"] = False
            prompt[NODES['simplify']]["inputs"]["fill_holes"] = scene.trellis2_fill_holes
            prompt[NODES['export_trimesh']]["inputs"]["filename_prefix"] = unique_prefix
        else:
            # Configure texture generation
            prompt[NODES['shape_to_textured_mesh']]["inputs"]["seed"] = seed
            prompt[NODES['shape_to_textured_mesh']]["inputs"]["tex_guidance_strength"] = scene.trellis2_tex_guidance
            prompt[NODES['shape_to_textured_mesh']]["inputs"]["tex_sampling_steps"] = scene.trellis2_tex_steps

            # Configure GLB export
            if use_pp:
                prompt[NODES['export_glb']]["inputs"]["decimation_target"] = scene.trellis2_decimation
                prompt[NODES['export_glb']]["inputs"]["remesh"] = scene.trellis2_remesh
            else:
                prompt[NODES['export_glb']]["inputs"]["decimation_target"] = 5000000
                prompt[NODES['export_glb']]["inputs"]["remesh"] = False
            prompt[NODES['export_glb']]["inputs"]["texture_size"] = scene.trellis2_texture_size
            prompt[NODES['export_glb']]["inputs"]["filename_prefix"] = unique_prefix

        # Save prompt for debugging
        revision_dir = get_generation_dirs(context).get("revision", "")
        if revision_dir:
            self._save_prompt_to_file(prompt, revision_dir)

        # --- Two-phase VRAM management for textured path ---
        # NOTE: Two-phase execution (shape first, flush, then texture) was
        # removed because ComfyUI requires at least one OUTPUT_NODE per prompt
        # and the shape-only subset has none.  The TRELLIS pipeline stages
        # handle VRAM management internally (unload_shape_pipeline before
        # loading texture models), so a single full-prompt submission works.

        # Connect WebSocket
        ws = self._connect_to_websocket(server_address, client_id)
        if ws is None:
            return {"error": "conn_failed"}

        # TRELLIS.2 simplification / post-processing can take several
        # minutes without sending any WS messages.  Use the user-
        # configurable mesh generation timeout.
        ws.settimeout(get_timeout('mesh_gen'))

        # Let the operator close this WS on cancel
        if hasattr(self.operator, '_active_ws'):
            self.operator._active_ws = ws

        prompt_id = None
        try:
            # Queue prompt
            prompt_id = self._queue_prompt(prompt, client_id, server_address)

            # Node-level progress (isolated subprocess doesn't emit within-node progress)
            # Weights approximate actual time spent per node
            if skip_texture:
                NODE_PROGRESS = {
                    NODES['input_image']:        2,
                    NODES['load_models']:        3,
                    NODES['get_conditioning']:   25,
                    NODES['image_to_shape']:     80,
                    NODES['simplify']:           90,
                    NODES['export_trimesh']:     98,
                }
                if not skip_bg:
                    NODE_PROGRESS[NODES['remove_bg']] = 12
            else:
                # Full textured pipeline (single submission)
                NODE_PROGRESS = {
                    NODES['input_image']:              2,
                    NODES['load_models']:              3,
                    NODES['get_conditioning']:         18,
                    NODES['image_to_shape']:           55,
                    NODES['shape_to_textured_mesh']:   85,
                    NODES['export_glb']:               98,
                }
                if not skip_bg:
                    NODE_PROGRESS[NODES['remove_bg']] = 10

            # Wait for execution to complete via WebSocket
            # Also capture 'executed' events which may contain the output path
            export_node_id = NODES[export_node_key]
            glb_output_path = None

            # Friendly node labels for the progress bars
            NODE_LABELS = {
                'input_image':              'Loading Image',
                'load_models':              'Loading Models',
                'remove_bg':                'Removing Background',
                'get_conditioning':         'Conditioning',
                'image_to_shape':           'Generating Shape',
                'shape_to_textured_mesh':   'Generating Texture',
                'simplify':                 'Simplifying Mesh',
                'export_trimesh':           'Exporting Mesh',
                'export_glb':               'Exporting GLB',
            }

            # Build a lookup of known step counts so we can identify
            # which sampling sub-phase a within-node progress event
            # belongs to (the TRELLIS nodes may emit separate runs of
            # progress events for SS, shape, and texture sampling).
            _ss_steps    = scene.trellis2_ss_steps
            _shape_steps = scene.trellis2_shape_steps
            _tex_steps   = getattr(scene, 'trellis2_tex_steps', 0)
            _current_exec_node = None  # node-key of the currently executing node

            # Track accumulated sub-phases within a node so we can
            # compute a node-internal overall progress.
            _sub_phase_idx = 0     # resets when the executing node changes
            _sub_phase_count = 1   # how many sub-phases this node has

            while True:
                try:
                    out = ws.recv()
                except websocket.WebSocketTimeoutException:
                    # Timeout on recv — server is still alive but a heavy
                    # step (e.g. simplification) took longer than expected.
                    # Keep waiting instead of treating it as a crash.
                    print("[TRELLIS2] WebSocket recv timed out — "
                          "server may still be processing. Retrying...")
                    continue
                except (ConnectionError, OSError, Exception) as ws_err:
                    err_name = type(ws_err).__name__
                    print(f"[TRELLIS2] WebSocket died ({err_name}): {ws_err}")
                    # Distinguish user-initiated cancel from a real crash
                    if getattr(self.operator, '_cancelled', False):
                        return {"error": "cancelled"}
                    return {"error": (
                        "ComfyUI server crashed during TRELLIS.2 generation "
                        "(likely VRAM exhaustion). Please restart ComfyUI, "
                        "reduce max_tokens or resolution, and try again."
                    )}
                if isinstance(out, str):
                    message = json.loads(out)

                    if message['type'] == 'executing':
                        data = message['data']
                        if data['prompt_id'] == prompt_id:
                            if data['node'] is None:
                                self.operator._phase_progress = 100
                                self.operator._detail_progress = 100
                                if hasattr(self.operator, '_update_overall'):
                                    self.operator._update_overall()
                                break  # Execution complete
                            else:
                                node_id = data['node']
                                node_names = {v: k for k, v in NODES.items()}
                                node_key = node_names.get(node_id, node_id)
                                node_label = NODE_LABELS.get(node_key, node_key)
                                print(f"[TRELLIS2] Executing: {node_label}")
                                self.operator._phase_stage = f"TRELLIS.2: {node_label}"
                                # Update progress based on node weight
                                if node_id in NODE_PROGRESS:
                                    self.operator._phase_progress = NODE_PROGRESS[node_id]
                                    if hasattr(self.operator, '_update_overall'):
                                        self.operator._update_overall()
                                self.operator._detail_stage = node_label
                                self.operator._detail_progress = 0

                                # Reset sub-phase tracking for the new node
                                _current_exec_node = node_key
                                _sub_phase_idx = 0
                                if node_key == 'image_to_shape':
                                    # SS sampling + shape sampling = 2 sub-phases
                                    _sub_phase_count = 2
                                elif node_key == 'shape_to_textured_mesh':
                                    _sub_phase_count = 1
                                else:
                                    _sub_phase_count = 1

                    elif message['type'] == 'executed':
                        # Capture output from export node (newer ComfyUI versions)
                        data = message.get('data', {})
                        if data.get('node') == export_node_id:
                            ws_output = data.get('output', {})
                            if ws_output:
                                for key in ['glb_path', 'file_path', 'text', 'string']:
                                    val = ws_output.get(key)
                                    if val:
                                        if isinstance(val, list) and len(val) > 0:
                                            val = val[0]
                                        if isinstance(val, str) and val:
                                            glb_output_path = val
                                            print(f"[TRELLIS2] Got path from WS executed event: {glb_output_path}")
                                            break
                                # Also check any key with a file-like value
                                if not glb_output_path:
                                    for key, val in ws_output.items():
                                        if isinstance(val, list) and len(val) > 0:
                                            val = val[0]
                                        if isinstance(val, str) and val.endswith(('.glb', '.obj', '.ply')):
                                            glb_output_path = val
                                            print(f"[TRELLIS2] Got path from WS (key={key}): {glb_output_path}")
                                            break

                    elif message['type'] == 'progress':
                        # Within-node progress (sampler steps).
                        # TRELLIS.2 nodes may emit separate runs of progress
                        # events for each internal sampling phase.  We detect
                        # the sub-phase by matching ``max`` against known step
                        # counts (SS, shape, texture).
                        p_data = message['data']
                        p_value = p_data['value']
                        p_max   = p_data['max']
                        step_progress = (p_value / p_max) * 100 if p_max else 0

                        # ── Sub-phase identification ──
                        sub_label = ""
                        if _current_exec_node == 'image_to_shape':
                            if p_max == _ss_steps:
                                sub_label = "Sampling SS"
                                # First sub-phase: reset index if we see it again
                                if _sub_phase_idx == 0 or p_value == 1:
                                    _sub_phase_idx = 0
                            elif p_max == _shape_steps:
                                sub_label = "Sampling Shape SLat"
                                if _sub_phase_idx < 1:
                                    _sub_phase_idx = 1
                            else:
                                # Unknown max — show generic label
                                sub_label = "Sampling"
                        elif _current_exec_node == 'shape_to_textured_mesh':
                            if p_max == _tex_steps:
                                sub_label = "Sampling Texture"
                            else:
                                sub_label = "Sampling"
                        else:
                            sub_label = "Processing"

                        # Compute node-internal overall progress accounting for sub-phases
                        if _sub_phase_count > 1:
                            node_overall = ((_sub_phase_idx + step_progress / 100.0)
                                            / _sub_phase_count) * 100.0
                        else:
                            node_overall = step_progress

                        if step_progress != 0:
                            self.operator._detail_progress = node_overall
                            self.operator._detail_stage = (
                                f"{sub_label}: Step {p_value}/{p_max}"
                            )
                            print(f"[TRELLIS2] {sub_label}: Step {p_value}/{p_max} ({step_progress:.0f}%)")

                    elif message['type'] == 'execution_error':
                        error_data = message.get('data', {})
                        error_msg = error_data.get('exception_message', 'Unknown error')
                        self.operator._error = f"TRELLIS.2 execution error: {error_msg}"
                        print(f"[TRELLIS2] Error: {self.operator._error}")
                        return {"error": self.operator._error}
        finally:
            if hasattr(self.operator, '_active_ws'):
                self.operator._active_ws = None
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass

        # ── Retrieve the GLB file ──────────────────────────────────────
        # Strategy 1: path from WS 'executed' event (already captured above)
        # Strategy 2: path from history API
        # Strategy 3: direct file read from disk (local ComfyUI)
        # Strategy 4: HTTP download via /view endpoint
        # Strategy 5: scan ComfyUI output dir for our prefix

        glb_source_path = glb_output_path  # May already be set from WS event

        # Helper: bail early when the user hits Cancel
        def _cancelled():
            return getattr(self.operator, '_cancelled', False)

        # Strategy 2: Query history for the output path
        if not glb_source_path and not _cancelled():
            try:
                history_url = f"http://{server_address}/history/{prompt_id}"
                history_response = json.loads(urllib.request.urlopen(
                    history_url, timeout=get_timeout('api')).read())

                if prompt_id in history_response:
                    outputs = history_response[prompt_id].get("outputs", {})
                    export_output = outputs.get(export_node_id, {})
                    print(f"[TRELLIS2] History output for node {export_node_id}: {export_output}")

                    for key in ['glb_path', 'file_path', 'text', 'string']:
                        val = export_output.get(key)
                        if val:
                            if isinstance(val, list) and len(val) > 0:
                                val = val[0]
                            if isinstance(val, str) and val:
                                glb_source_path = val
                                print(f"[TRELLIS2] Got path from history (key={key}): {glb_source_path}")
                                break

                    # Check any key with a path-like value
                    if not glb_source_path:
                        for key, val in export_output.items():
                            if isinstance(val, list) and len(val) > 0:
                                val = val[0]
                            if isinstance(val, str) and val.endswith(('.glb', '.obj', '.ply')):
                                glb_source_path = val
                                print(f"[TRELLIS2] Got path from history (key={key}): {glb_source_path}")
                                break
            except Exception as e:
                print(f"[TRELLIS2] History query failed: {e}")

        # Strategy 3: Read directly from disk (works when ComfyUI is local)
        if glb_source_path and not _cancelled() and os.path.isfile(glb_source_path):
            try:
                print(f"[TRELLIS2] Reading GLB directly from disk: {glb_source_path}")
                with open(glb_source_path, 'rb') as f:
                    glb_data = f.read()
                print(f"[TRELLIS2] Read {len(glb_data)} bytes from disk")
                if glb_data and len(glb_data) > 0:
                    return glb_data
            except Exception as e:
                print(f"[TRELLIS2] Direct file read failed: {e}")

        # Strategy 4: HTTP download via /view endpoint
        if glb_source_path and not _cancelled():
            glb_filename = os.path.basename(glb_source_path)
            try:
                view_url = f"http://{server_address}/view?filename={urllib.parse.quote(glb_filename)}&type=output"
                print(f"[TRELLIS2] Downloading GLB via HTTP: {view_url}")
                glb_response = urllib.request.urlopen(view_url, timeout=get_timeout('transfer'))
                glb_data = glb_response.read()
                print(f"[TRELLIS2] Downloaded GLB: {len(glb_data)} bytes")
                if glb_data and len(glb_data) > 0:
                    return glb_data
            except Exception as e:
                print(f"[TRELLIS2] HTTP download failed: {e}")

        if _cancelled():
            return {"error": "cancelled"}

        # Strategy 5: Scan ComfyUI output directory for files matching our prefix
        # This handles the case where the export node path wasn't captured via API
        print(f"[TRELLIS2] Scanning for files with prefix '{unique_prefix}'...")

        # 5a: Try to discover ComfyUI's output directory from the server
        comfyui_output_dir = None
        if not _cancelled():
            try:
                system_url = f"http://{server_address}/system_stats"
                system_response = json.loads(urllib.request.urlopen(
                    system_url, timeout=get_timeout('api')).read())
                # Some ComfyUI versions include directory info
                comfyui_output_dir = system_response.get("output_dir")
            except Exception:
                pass

        # 5b: Try common local paths relative to server
        if not comfyui_output_dir:
            # Check if server is localhost - if so, try to find output dir
            host = server_address.split(':')[0]
            if host in ('127.0.0.1', 'localhost', '0.0.0.0', '::1'):
                # Try to discover via the /view endpoint with a known file
                # Or just check common ComfyUI locations
                common_paths = [
                    os.path.join(os.environ.get('COMFYUI_PATH', ''), 'output'),
                    'E:/tools/ComfyUI/output',
                    'C:/ComfyUI/output',
                    os.path.expanduser('~/ComfyUI/output'),
                ]
                for candidate in common_paths:
                    if candidate and os.path.isdir(candidate):
                        comfyui_output_dir = candidate
                        break

        if comfyui_output_dir and os.path.isdir(comfyui_output_dir):
            print(f"[TRELLIS2] Scanning output dir: {comfyui_output_dir}")
            try:
                matching_files = sorted(
                    [f for f in os.listdir(comfyui_output_dir)
                     if f.startswith(unique_prefix) and f.endswith(('.glb', '.obj', '.ply'))],
                    key=lambda f: os.path.getmtime(os.path.join(comfyui_output_dir, f)),
                    reverse=True
                )
                if matching_files:
                    found_path = os.path.join(comfyui_output_dir, matching_files[0])
                    print(f"[TRELLIS2] Found matching file: {found_path}")
                    with open(found_path, 'rb') as f:
                        glb_data = f.read()
                    print(f"[TRELLIS2] Read {len(glb_data)} bytes from disk")
                    if glb_data and len(glb_data) > 0:
                        return glb_data
            except Exception as e:
                print(f"[TRELLIS2] Output dir scan failed: {e}")

        # 5c: Try HTTP download with timestamp-based filename guesses
        # The export nodes use format: {prefix}_{YYYYMMDD_HHMMSS}.glb
        # Use a short timeout per request so slow remote connections don't
        # block for minutes, and check for cancellation each iteration.
        is_remote = not self._is_local_server(server_address)
        scan_range = 30 if is_remote else 120  # fewer guesses for remote
        scan_timeout = 5 if is_remote else 10   # per-request timeout (seconds)
        now = datetime.now()
        for delta_seconds in range(0, scan_range):
            if _cancelled():
                return {"error": "cancelled"}
            for delta in [timedelta(seconds=-delta_seconds), timedelta(seconds=delta_seconds)]:
                candidate_time = now + delta
                candidate_name = f"{unique_prefix}_{candidate_time.strftime('%Y%m%d_%H%M%S')}.glb"
                try:
                    view_url = f"http://{server_address}/view?filename={urllib.parse.quote(candidate_name)}&type=output"
                    glb_response = urllib.request.urlopen(view_url, timeout=scan_timeout)
                    glb_data = glb_response.read()
                    if glb_data and len(glb_data) > 0:
                        print(f"[TRELLIS2] Found GLB via timestamp scan: {candidate_name} ({len(glb_data)} bytes)")
                        return glb_data
                except Exception:
                    continue

        if _cancelled():
            return {"error": "cancelled"}

        self.operator._error = (
            f"Failed to retrieve GLB from ComfyUI. "
            f"The workflow completed but the output file could not be located. "
            f"Prefix: {unique_prefix}"
        )
        return {"error": self.operator._error}

    def _create_base_prompt(self, context):
        """Creates and configures the base prompt with user settings."""
        from .util.helpers import prompt_text
        
        # Load the base prompt template
        prompt = json.loads(prompt_text)
        
        # Node IDs organized by functional category
        NODES = {
            # Text Prompting
            'pos_prompt': "9",
            'neg_prompt': "10",
            'clip_skip': "247",
            
            # Sampling Control
            'sampler': "15",
            'seed_control': "15",  # Same as sampler node but for seed parameter
            
            # Model Loading
            'checkpoint': "6",
            
            # Latent Space
            'latent': "16",
            
            # Image Output
            'save_image': "25",

            # IPAdapter
            'ipadapter_loader': "235",
            'ipadapter': "236",
            'ipadapter_image': "237",
        }
        
        base_prompt_text = context.scene.comfyui_prompt
        # Camera Prompt Injection
        if context.scene.use_camera_prompts and context.scene.generation_method in ['separate', 'sequential', 'refine', 'local_edit'] and self.operator._cameras and self.operator._current_image < len(self.operator._cameras):
            current_camera_name = self.operator._cameras[self.operator._current_image].name
            # Find the prompt in the collection
            prompt_item = next((item for item in context.scene.camera_prompts if item.name == current_camera_name), None)
            if prompt_item and prompt_item.prompt:
                view_desc = prompt_item.prompt
                # Prepend the view description
                base_prompt_text = f"{view_desc}, {base_prompt_text}"
        
        # Set text prompts
        prompt[NODES['pos_prompt']]["inputs"]["text"] = base_prompt_text
        prompt[NODES['neg_prompt']]["inputs"]["text"] = context.scene.comfyui_negative_prompt
        
        # Set sampling parameters
        prompt[NODES['sampler']]["inputs"]["seed"] = context.scene.seed
        prompt[NODES['sampler']]["inputs"]["steps"] = context.scene.steps
        prompt[NODES['sampler']]["inputs"]["cfg"] = context.scene.cfg
        prompt[NODES['sampler']]["inputs"]["sampler_name"] = context.scene.sampler
        prompt[NODES['sampler']]["inputs"]["scheduler"] = context.scene.scheduler
        
        # Set clip skip
        prompt[NODES['clip_skip']]["inputs"]["stop_at_clip_layer"] = -context.scene.clip_skip
        
        # Set the model name
        prompt[NODES['checkpoint']]["inputs"]["ckpt_name"] = context.scene.model_name

        # Build LoRA chain
        initial_model_input_lora = [NODES['checkpoint'], 0]
        initial_clip_input_lora = [NODES['checkpoint'], 1]

        prompt, final_lora_model_out, final_lora_clip_out = self._build_lora_chain(
            prompt, context,
            initial_model_input_lora, initial_clip_input_lora,
            start_node_id=400 # Starting node ID for LoRA chain
        )

        current_model_out = final_lora_model_out

        # Set the input for the clip skip node
        prompt[NODES['clip_skip']]["inputs"]["clip"] = final_lora_clip_out

        # If using IPAdapter, set the model input
        _is_trellis2_input_ipadapter = (
            context.scene.sequential_ipadapter
            and context.scene.sequential_ipadapter_mode == 'trellis2_input'
        )
        if (context.scene.use_ipadapter
                or _is_trellis2_input_ipadapter
                or (context.scene.generation_method == 'separate'
                    and context.scene.sequential_ipadapter
                    and self.operator._current_image > 0)):
            # Set the model input for IPAdapter
            prompt[NODES['ipadapter_loader']]["inputs"]["model"] = current_model_out
            current_model_out = [NODES['ipadapter'], 0]

        # Set the model for sampler node
        prompt[NODES['sampler']]["inputs"]["model"] = current_model_out

        return prompt, NODES

    def _configure_resolution(self, prompt, context, NODES):
        """Sets the generation resolution based on mode."""
        if context.scene.generation_method == 'grid':
            # Use the resolution of the grid image
            prompt[NODES['latent']]["inputs"]["width"] = self.operator._grid_width
            prompt[NODES['latent']]["inputs"]["height"] = self.operator._grid_height
        else:
            # Use current render resolution
            prompt[NODES['latent']]["inputs"]["width"] = context.scene.render.resolution_x
            prompt[NODES['latent']]["inputs"]["height"] = context.scene.render.resolution_y

    def _configure_ipadapter(self, prompt, context, ipadapter_ref_info, NODES):
        # Configure IPAdapter if enabled
        
        # Connect IPAdapter output to the appropriate node
        prompt[NODES['sampler']]["inputs"]["model"] = [NODES['ipadapter'], 0]
        
        # Set IPAdapter image source
        prompt[NODES['ipadapter_image']]["inputs"]["image"] = ipadapter_ref_info['name']

        # Connect ipadapter image to the input
        prompt[NODES['ipadapter']]["inputs"]["image"] = [NODES['ipadapter_image'], 0]
        
        # Configure IPAdapter settings
        prompt[NODES['ipadapter']]["inputs"]["weight"] = context.scene.ipadapter_strength
        prompt[NODES['ipadapter']]["inputs"]["start_at"] = context.scene.ipadapter_start
        prompt[NODES['ipadapter']]["inputs"]["end_at"] = context.scene.ipadapter_end
        
        # Set weight type
        weight_type_mapping = {
            'standard': "standard",
            'prompt': "prompt is more important",
            'style': "style transfer"
        }
        prompt[NODES['ipadapter']]["inputs"]["weight_type"] = weight_type_mapping.get(context.scene.ipadapter_weight_type, "standard")

    def _build_controlnet_chain_extended(self, context, base_prompt, pos_input, neg_input, vae_input, controlnet_info_dict):
        """
        Builds a chain of ControlNet units dynamically based on scene settings.

        Args:
            context: Blender context, used to access addon preferences and scene data.
            base_prompt (dict): The ComfyUI prompt dictionary to be modified.
            pos_input (list): The [node_id, output_idx] for the initial positive conditioning.
            neg_input (list): The [node_id, output_idx] for the initial negative conditioning.
            vae_input (list): The [node_id, output_idx] for the VAE, used by ControlNetApplyAdvanced.
                                Typically, this is [checkpoint_node_id, 2] for SDXL or 
                                [checkpoint_node_id, 0] for some VAE loaders.
            controlnet_info_dict (dict): A dictionary mapping ControlNet types (e.g., "depth", 
                                "canny") to their corresponding uploaded image info.

        Returns:
            tuple: (modified_prompt, final_positive_conditioning, final_negative_conditioning)
        """
        addon_prefs = context.preferences.addons[__package__].preferences
        try:
            mapping = json.loads(addon_prefs.controlnet_mapping)
        except Exception:
            mapping = {}
        
        # Get the dynamic collection of ControlNet units
        controlnet_units = getattr(context.scene, "controlnet_units", [])
        current_pos = pos_input
        current_neg = neg_input
        has_union = False
        for idx, unit in enumerate(controlnet_units):
            # Get uploaded info for this unit's type
            uploaded_info = controlnet_info_dict.get(unit.unit_type)
            if not uploaded_info:
                print(f"Warning: Uploaded info for ControlNet type '{unit.unit_type}' not found. Skipping unit.")
                continue # Skip this unit

            # Generate unique keys for nodes in this chain unit.
            load_key = str(200 + idx * 3)       # LoadImage node
            loader_key = str(200 + idx * 3 + 1)   # ControlNetLoader node
            apply_key = str(200 + idx * 3 + 2)    # ControlNetApplyAdvanced node

            # Create the LoadImage node.
            base_prompt[load_key] = {
                "inputs": {
                    "image": uploaded_info['name'],
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": f"Load Image ({unit.unit_type})"
                }
            }
            # Create the ControlNetLoader node.
            base_prompt[loader_key] = {
                "inputs": {
                    "control_net_name": unit.model_name  # updated to use selected property
                },
                "class_type": "ControlNetLoader",
                "_meta": {
                    "title": f"Load ControlNet ({unit.unit_type})"
                }
            }
            # Create the ControlNetApplyAdvanced node.
            base_prompt[apply_key] = {
                "inputs": {
                    "strength": unit.strength,
                    "start_percent": unit.start_percent,
                    "end_percent": unit.end_percent,
                    "positive": [current_pos, 0],
                    "negative": [current_neg, 1] if (idx > 0 or current_neg == "228" or current_neg == "51") else [current_neg, 0],
                    "control_net": [loader_key, 0],
                    "image": [load_key, 0],
                    "vae": [vae_input, 2] if context.scene.model_architecture == "sdxl" else [vae_input, 0],
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {
                    "title": f"Apply ControlNet ({unit.unit_type})"
                }
            }
            # Update chain inputs: the output of this apply node becomes the new input.
            current_pos = apply_key
            current_neg = apply_key
            # If the controlnet is of the union type, connect the ControlNetApplyAdvanced input into the SetUnionControlNetType node (239)
            if unit.is_union and unit.use_union_type: 
                base_prompt[apply_key]["inputs"]["control_net"] = ["239", 0]
                base_prompt["239"]["inputs"]["control_net"] = [loader_key, 0]
                if unit.unit_type == "depth":
                    base_prompt["239"]["inputs"]["type"] = "depth" 
                elif unit.unit_type == "canny":
                    base_prompt["239"]["inputs"]["type"] = "canny/lineart/anime_lineart/mlsd"
                elif unit.unit_type == "normal":
                    base_prompt["239"]["inputs"]["type"] = "normal"
                has_union = True
        if not has_union:
            # Remove the node
            if "239" in base_prompt:
                del base_prompt["239"]

        return base_prompt, current_pos

    def _build_lora_chain(self, prompt, context, initial_model_input, initial_clip_input, start_node_id=300, lora_class_type="LoraLoader"):
        """
        Builds a chain of LoRA loaders dynamically.

        Args:
            prompt (dict): The ComfyUI prompt dictionary to modify.
            context: Blender context.
            initial_model_input (list): The [node_id, output_idx] for the initial model.
            initial_clip_input (list): The [node_id, output_idx] for the initial CLIP.
            start_node_id (int): The starting integer for generating unique LoRA node IDs.
            lora_class_type (str): The class type of the LoRA loader node to use.

        Returns:
            tuple: (modified_prompt, final_model_output, final_clip_output)
            
            The final model and CLIP outputs are [node_id, output_idx] lists.
        """
        scene = context.scene
        
        current_model_out = initial_model_input
        current_clip_out = initial_clip_input

        if not scene.lora_units:
            return prompt, current_model_out, current_clip_out

        for i, lora_unit in enumerate(scene.lora_units):
            if not lora_unit.model_name or lora_unit.model_name == "NONE":
                continue # Skip if no LoRA model is selected for this unit

            lora_node_id_str = str(start_node_id + i)
            
            lora_inputs = {
                "lora_name": lora_unit.model_name,
                "strength_model": lora_unit.model_strength,
                "model": current_model_out,
            }

            if lora_class_type == "LoraLoader":
                lora_inputs["strength_clip"] = lora_unit.clip_strength
                lora_inputs["clip"] = current_clip_out
            elif lora_class_type == "NunchakuQwenImageLoraLoader":
                lora_inputs = {
                    "lora_name": lora_unit.model_name,
                    "lora_strength": lora_unit.model_strength,
                    "cpu_offload": "disable",
                    "model": current_model_out
                }

            prompt[lora_node_id_str] = {
                "inputs": lora_inputs,
                "class_type": lora_class_type,
                "_meta": {
                    "title": f"Load LoRA {i+1} ({lora_unit.model_name[:20]})"
                }
            }
            # Update outputs for the next LoRA in the chain
            current_model_out = [lora_node_id_str, 0]
            if lora_class_type == "LoraLoader":
                current_clip_out = [lora_node_id_str, 1]
            
        return prompt, current_model_out, current_clip_out

    def _build_controlnet_chain(self, prompt, context, controlnet_info, NODES):
        """Builds the ControlNet processing chain."""
        # Build controlnet chain with guidance images
        prompt, final_node = self._build_controlnet_chain_extended(
            context, prompt, NODES['pos_prompt'], NODES['neg_prompt'], NODES['checkpoint'],
            controlnet_info
        )
        
        # Connect final node outputs to the KSampler
        prompt[NODES['sampler']]["inputs"]["positive"] = [final_node, 0]
        prompt[NODES['sampler']]["inputs"]["negative"] = [final_node, 1]
        
        return prompt

    def _save_prompt_to_file(self, prompt, output_dir):
        """Saves the prompt to a file for debugging."""
        try:
            with open(os.path.join(output_dir, "prompt.json"), 'w') as f:
                json.dump(prompt, f, indent=2)  # Added indent for better readability
        except Exception as e:
            print(f"Failed to save prompt to file: {str(e)}")

    @staticmethod
    def _validate_image_bytes(img_bytes):
        """Return True if *img_bytes* can be decoded as an image by PIL."""
        if not img_bytes:
            return False
        try:
            Image.open(BytesIO(img_bytes)).verify()
            return True
        except Exception:
            return False

    def _get_images_via_http(self, server_address, prompt_id, node_id):
        """
        Fallback: fetch output image(s) for *node_id* from ComfyUI's
        /history + /view HTTP endpoints.  Returns a list of bytes objects,
        or an empty list on failure.
        """
        try:
            history_url = f"http://{server_address}/history/{prompt_id}"
            history = json.loads(urllib.request.urlopen(history_url, timeout=get_timeout('transfer')).read())
            outputs = history.get(prompt_id, {}).get("outputs", {}).get(node_id, {})
            image_list = outputs.get("images", [])
            result = []
            for img_meta in image_list:
                filename  = img_meta.get("filename", "")
                subfolder = img_meta.get("subfolder", "")
                img_type  = img_meta.get("type", "output")
                view_url  = (f"http://{server_address}/view?"
                             f"filename={urllib.parse.quote(filename)}"
                             f"&subfolder={urllib.parse.quote(subfolder)}"
                             f"&type={urllib.parse.quote(img_type)}")
                data = urllib.request.urlopen(view_url, timeout=get_timeout('transfer')).read()
                result.append(data)
            return result
        except Exception as e:
            print(f"[StableGen] HTTP image fallback failed: {e}")
            return []

    def _connect_to_websocket(self, server_address, client_id):
        """Establishes WebSocket connection to ComfyUI server."""
        try:
            ws = websocket.WebSocket()
            # Use a short timeout for the initial TCP+WS handshake.
            ws.settimeout(get_timeout('api'))
            ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
            # Switch to a longer timeout for recv() during generation —
            # ComfyUI may take a while between progress messages (model
            # loading, TRELLIS.2 processing, etc.).
            ws.settimeout(get_timeout('transfer'))
            return ws
        except ConnectionRefusedError:
            self._error = f"Connection to ComfyUI WebSocket was refused at {server_address}. Is ComfyUI running and accessible?"
            return None
        except (socket.gaierror, websocket.WebSocketAddressException): # Catch getaddrinfo errors specifically
            self._error = f"Could not resolve ComfyUI server address: '{server_address}'. Please check the hostname/IP and port in preferences and your network settings."
            return None
        except websocket.WebSocketTimeoutException:
            self._error = f"Connection to ComfyUI WebSocket timed out at {server_address}."
            return None
        except websocket.WebSocketBadStatusException as e: # More specific catch for handshake errors
            # e.status_code will be 404 in this case
            if e.status_code == 404:
                self._error = (f"ComfyUI endpoint not found at {server_address} (404 Not Found).")
            else:
                self._error = (f"WebSocket handshake failed with ComfyUI server at {server_address}. "
                            f"Status: {e.status_code}. The server might not be a ComfyUI instance or is misconfigured.")
            return None
        except Exception as e: # Catch-all for truly unexpected issues during connect
            self._error = f"An unexpected error occurred connecting WebSocket: {e}"
            return None

    @staticmethod
    def _inject_save_image_fallback(prompt, save_ws_node_id):
        """
        Inject a regular ``SaveImage`` node into *prompt* that mirrors the
        input of the ``SaveImageWebsocket`` node.  This ensures the image is
        also written to disk so it can be retrieved via HTTP if the WebSocket
        binary transfer is corrupted (e.g. packet loss on a remote network).

        Returns the node-ID of the injected node, or *None* if injection
        was not possible.
        """
        ws_node = prompt.get(save_ws_node_id)
        if not ws_node:
            return None

        images_input = ws_node.get("inputs", {}).get("images")
        if not images_input:
            return None

        # Pick a node-ID that is guaranteed not to collide
        fallback_id = f"_sg_fallback_{save_ws_node_id}"
        prompt[fallback_id] = {
            "inputs": {
                "filename_prefix": "StableGen_fallback",
                "images": images_input,
            },
            "class_type": "SaveImage",
            "_meta": {"title": "StableGen HTTP fallback"},
        }
        return fallback_id

    def _execute_prompt_and_get_images(self, ws, prompt, client_id, server_address, NODES):
        """Executes the prompt and collects generated images."""

        # Inject a disk-saving fallback node so we can recover via HTTP
        # if the WebSocket binary data arrives corrupted.
        fallback_node_id = self._inject_save_image_fallback(
            prompt, NODES.get('save_image', ''))

        # Send the prompt to the queue
        prompt_id = self._queue_prompt(prompt, client_id, server_address)
        
        # Process the WebSocket messages and collect images
        output_images = {}
        current_node = ""
        
        while True:
            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                print(f"[StableGen] WebSocket recv timed out after "
                      f"{get_timeout('transfer')}s. The server may be "
                      f"slow or the connection was lost.")
                break
            except (ConnectionError, OSError, Exception) as ws_err:
                print(f"[StableGen] WebSocket error during generation: {ws_err}")
                break

            if isinstance(out, str):
                message = json.loads(out)
                
                if message['type'] == 'executing':
                    data = message['data']
                    if data['prompt_id'] == prompt_id:
                        if data['node'] is None:
                            break  # Execution is complete
                        else:
                            current_node = data['node']
                            print(f"Executing node: {current_node}")
                            
                elif message['type'] == 'progress':
                    progress = (message['data']['value'] / message['data']['max']) * 100
                    if progress != 0:
                        self.operator._progress = progress  # Update progress for UI
                        # Transition stage from "Uploading" to "Generating" on
                        # first actual sampler progress from the server.
                        if self.operator._stage == "Uploading to Server":
                            self.operator._stage = "Generating Image"
                        # Also update 3-tier detail when called from Trellis2Generate
                        if hasattr(self.operator, '_detail_progress'):
                            self.operator._detail_progress = progress
                            self.operator._detail_stage = f"Step {message['data']['value']}/{message['data']['max']}"
                            # Remap progress into a sub-range when gallery mode is active
                            remap = getattr(self.operator, '_progress_remap', None)
                            if remap:
                                base, span = remap
                                self.operator._phase_progress = base + (progress / 100.0) * span
                            else:
                                self.operator._phase_progress = progress
                            if hasattr(self.operator, '_update_overall'):
                                self.operator._update_overall()
                        print(f"Progress: {progress:.1f}%")
            else:
                # Binary data (image)
                if current_node == NODES['save_image']:  # SaveImageWebsocket node
                    print("Receiving generated image")
                    images_output = output_images.get(current_node, [])
                    images_output.append(out[8:])  # Skip the first 8 bytes (header)
                    output_images[current_node] = images_output

        # --- Validate received images; fall back to HTTP if corrupt ----------
        save_node = NODES.get('save_image')
        if save_node and save_node in output_images:
            needs_http = False
            for img_bytes in output_images[save_node]:
                if not self._validate_image_bytes(img_bytes):
                    needs_http = True
                    break

            if needs_http and fallback_node_id:
                print("[StableGen] WebSocket image data is corrupt "
                      "(likely packet loss on remote network). "
                      "Falling back to HTTP /view download...")
                http_images = self._get_images_via_http(
                    server_address, prompt_id, fallback_node_id)
                if http_images:
                    output_images[save_node] = http_images
                    print(f"[StableGen] Successfully retrieved "
                          f"{len(http_images)} image(s) via HTTP fallback.")
                else:
                    print("[StableGen] HTTP fallback also failed. "
                          "The image data may be unrecoverable.")
            elif needs_http:
                print("[StableGen] WebSocket image data is corrupt "
                      "and no HTTP fallback node was injected.")

        return output_images

    def _queue_prompt(self, prompt, client_id, server_address):
        """Queues the prompt for processing by ComfyUI."""
        try:
            data = json.dumps({
                "prompt": prompt,
                "client_id": client_id
            }).encode('utf-8')
            
            req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
            response = json.loads(urllib.request.urlopen(req, timeout=get_timeout('api')).read())
            
            return response['prompt_id']
        except Exception as e:
            print(f"Failed to queue prompt: {str(e)}")
            raise

    def refine(self, context, controlnet_info=None, mask_info=None, render_info=None, ipadapter_ref_info=None):
        """     
        Refines the image using ComfyUI.         
        :param context: Blender context.         
        :param controlnet_info: Dict of uploaded controlnet image info.
        :param mask_info: Uploaded mask image info.
        :param render_info: Uploaded render image info.
        :param ipadapter_ref_info: Uploaded IPAdapter reference image info.
        :return: Refined image.     
        """
        # Setup connection parameters
        server_address = context.preferences.addons[__package__].preferences.server_address
        client_id = str(uuid.uuid4())
        output_dir = context.preferences.addons[__package__].preferences.output_dir

        revision_dir = get_generation_dirs(context)["revision"]

        # Initialize the img2img prompt template and configure base settings
        prompt, NODES = self._create_img2img_base_prompt(context)
        
        # Configure based on generation method
        self._configure_refinement_mode(prompt, context, render_info, mask_info, NODES)

        if ipadapter_ref_info and context.scene.generation_method != 'uv_inpaint':
            # Configure IPAdapter settings
            self._configure_ipadapter_refine(prompt, context, ipadapter_ref_info, NODES)
        else:
            # Remove IPAdapter nodes if not used
            for node_id in ["235", "236", "237"]:
                if node_id in prompt:
                    del prompt[node_id]
        
        # Set up image inputs for different controlnet types
        self._refine_configure_images(prompt, render_info, NODES)
        
        # Build controlnet chain for refinement if needed
        if not context.scene.generation_method == 'uv_inpaint':
            prompt = self._refine_build_controlnet_chain(prompt, context, controlnet_info, NODES)
        else:
            if context.scene.differential_diffusion:
                prompt[NODES['sampler']]["inputs"]["positive"] = [NODES['inpaint_conditioning'], 0]
                prompt[NODES['sampler']]["inputs"]["negative"] = [NODES['inpaint_conditioning'], 1]
            else:
                prompt[NODES['sampler']]["inputs"]["positive"] = [NODES['pos_prompt'], 0]
                prompt[NODES['sampler']]["inputs"]["negative"] = [NODES['neg_prompt'], 0]
        
        # Save prompt for debugging
        with open(os.path.join(output_dir, "prompt.json"), 'w') as f:
            json.dump(prompt, f)
        
        # Execute generation and get results
        ws = self._connect_to_websocket(server_address, client_id)

        if ws is None:
            return {"error": "conn_failed"} # Connection error

        images = None
        try:
            images = self._execute_prompt_and_get_images(ws, prompt, client_id, server_address, NODES)
        finally:
            if ws:
                ws.close()

        if images is None or isinstance(images, dict) and "error" in images:
            return {"error": "conn_failed"}
        
        print(f"Image refined with prompt: ...")

        img_bytes = images[NODES['save_image']][0]

        # Remove Comfy’s hard border and add a soft alpha vignette
        img_bytes = self._crop_and_vignette(
            img_bytes,
            border_px=8,   # tweak this if your Comfy border is wider/narrower
            feather=0.08,  # thicker feather band
            gamma=0.5,     # stronger fade near edge
        )

        return img_bytes

    def _create_img2img_base_prompt(self, context):
        """Creates and configures the base prompt for img2img refinement."""
        from .util.helpers import prompt_text_img2img
        
        prompt = json.loads(prompt_text_img2img)
        
        # Node IDs organized by functional category
        NODES = {
            # Text Prompting
            'pos_prompt': "102",
            'neg_prompt': "103",
            'clip_skip': "247",
            
            # Sampling Control
            'sampler': "105",
            
            # Model Loading
            'checkpoint': "38",
            
            # Image Processing
            'upscale_grid': "118",
            'upscale_uv': "23",
            'vae_encode': "116",
            'vae_encode_inpaint': "13",
            'inpaint_conditioning': "228",
            
            # Input Images
            'input_image': "1",
            'mask_image': "12",
            'render_image': "117",
            
            # Mask Processing
            'grow_mask': "224",
            'blur': "226",
            'image_to_mask': "227",
            
            # Advanced Features
            'differential_diffusion': "229",
            'ipadapter_loader': "235",
            'ipadapter': "236",
            'ipadapter_image': "237",
            
            # Output
            'save_image': "111"
        }
        
        base_prompt_text = context.scene.comfyui_prompt
        # Camera Prompt Injection
        if context.scene.use_camera_prompts and context.scene.generation_method in ['separate', 'sequential', 'refine', 'local_edit'] and self.operator._cameras and self.operator._current_image < len(self.operator._cameras):
            current_camera_name = self.operator._cameras[self.operator._current_image].name
            # Find the prompt in the collection
            prompt_item = next((item for item in context.scene.camera_prompts if item.name == current_camera_name), None)
            if prompt_item and prompt_item.prompt:
                view_desc = prompt_item.prompt
                # Prepend the view description
                base_prompt_text = f"{view_desc}, {base_prompt_text}"
        
        # Set positive prompt based on generation method
        if context.scene.generation_method in ['refine', 'local_edit', 'uv_inpaint', 'sequential']:
            prompt[NODES['pos_prompt']]["inputs"]["text"] = base_prompt_text
        else:
            prompt[NODES['pos_prompt']]["inputs"]["text"] = context.scene.refine_prompt if context.scene.refine_prompt != "" else context.scene.comfyui_prompt
        
        # Set negative prompt
        prompt[NODES['neg_prompt']]["inputs"]["text"] = context.scene.comfyui_negative_prompt
        
        # Set sampling parameters
        prompt[NODES['sampler']]["inputs"]["seed"] = context.scene.seed
        prompt[NODES['sampler']]["inputs"]["steps"] = context.scene.refine_steps if context.scene.generation_method == 'grid' else context.scene.steps
        prompt[NODES['sampler']]["inputs"]["cfg"] = context.scene.refine_cfg if context.scene.generation_method == 'grid' else context.scene.cfg
        prompt[NODES['sampler']]["inputs"]["sampler_name"] = context.scene.refine_sampler if context.scene.generation_method == 'grid' else context.scene.sampler
        prompt[NODES['sampler']]["inputs"]["scheduler"] = context.scene.refine_scheduler if context.scene.generation_method == 'grid' else context.scene.scheduler
        if context.scene.generation_method in ('grid', 'refine', 'local_edit'):
            prompt[NODES['sampler']]["inputs"]["denoise"] = context.scene.denoise
        else:
            prompt[NODES['sampler']]["inputs"]["denoise"] = 1.0
        
        # Set clip skip
        prompt[NODES['clip_skip']]["inputs"]["stop_at_clip_layer"] = -context.scene.clip_skip
        
        # Set upscale method and dimensions
        prompt[NODES['upscale_grid']]["inputs"]["upscale_method"] = context.scene.refine_upscale_method
        prompt[NODES['upscale_grid']]["inputs"]["width"] = context.scene.render.resolution_x
        prompt[NODES['upscale_grid']]["inputs"]["height"] = context.scene.render.resolution_y
        prompt[NODES['upscale_uv']]["inputs"]["upscale_method"] = "nearest-exact"
        prompt[NODES['upscale_uv']]["inputs"]["width"] = 1024
        prompt[NODES['upscale_uv']]["inputs"]["height"] = 1024

        # Set the model name
        prompt[NODES['checkpoint']]["inputs"]["ckpt_name"] = context.scene.model_name
        
        # Build LoRA chain
        initial_model_input_lora = [NODES['checkpoint'], 0]
        initial_clip_input_lora = [NODES['checkpoint'], 1]

        prompt, final_lora_model_out, final_lora_clip_out = self._build_lora_chain(
            prompt, context,
            initial_model_input_lora, initial_clip_input_lora,
            start_node_id=400 # Starting node ID for LoRA chain
        )

        current_model_out = final_lora_model_out

        # Set the input for the clip skip node
        prompt[NODES['clip_skip']]["inputs"]["clip"] = final_lora_clip_out

        # If using IPAdapter, set the model input
        _is_original_render_ipadapter = (
            context.scene.sequential_ipadapter
            and context.scene.sequential_ipadapter_mode == 'original_render'
            and context.scene.generation_method == 'local_edit'
        )
        _is_trellis2_input_ipadapter = (
            context.scene.sequential_ipadapter
            and context.scene.sequential_ipadapter_mode == 'trellis2_input'
        )
        if (context.scene.use_ipadapter or _is_original_render_ipadapter or _is_trellis2_input_ipadapter or (context.scene.sequential_ipadapter and self.operator._current_image > 0)) and context.scene.generation_method != 'uv_inpaint':
            # Set the model input for IPAdapter
            prompt[NODES['ipadapter_loader']]["inputs"]["model"] = current_model_out
            current_model_out = [NODES['ipadapter'], 0]

        if context.scene.differential_diffusion and NODES['differential_diffusion'] in prompt and context.scene.generation_method not in ('refine', 'local_edit'):
            # Set model input for differential diffusion
            prompt[NODES['differential_diffusion']]["inputs"]["model"] = current_model_out
            current_model_out = [NODES['differential_diffusion'], 0]

        # Set the model for sampler node
        prompt[NODES['sampler']]["inputs"]["model"] = current_model_out

        return prompt, NODES

    def _configure_refinement_mode(self, prompt, context, render_info, mask_info, NODES):
        """Configures the prompt based on the specific refinement mode."""
        # Configure based on generation method
        if context.scene.generation_method in ('refine', 'local_edit'):
            prompt[NODES['vae_encode']]["inputs"]["pixels"] = [NODES['render_image'], 0]  # Use render directly
        
        elif context.scene.generation_method == 'uv_inpaint' or context.scene.generation_method == 'sequential':
            # Connect latent to KSampler
            prompt[NODES['sampler']]["inputs"]["latent_image"] = [NODES['vae_encode_inpaint'], 0] if not context.scene.differential_diffusion else [NODES['inpaint_conditioning'], 2]
            
            # Configure differential diffusion if enabled
            if context.scene.differential_diffusion:
                prompt[NODES['sampler']]["inputs"]["model"] = [NODES['differential_diffusion'], 0]
            
            # Configure mask settings
            prompt[NODES['mask_image']]["inputs"]["image"] = mask_info['name']
            prompt[NODES['input_image']]["inputs"]["image"] = render_info['name']
            
            # Configure mask blur settings
            if not context.scene.blur_mask:
                prompt[NODES['inpaint_conditioning']]["inputs"]["mask"] = [NODES['grow_mask'], 0]  # Direct connection
                prompt[NODES['vae_encode_inpaint']]["inputs"]["mask"] = [NODES['grow_mask'], 0]   # Direct connection
            
            # Set blur parameters
            prompt[NODES['blur']]["inputs"]["sigma"] = context.scene.blur_mask_sigma
            prompt[NODES['blur']]["inputs"]["blur_radius"] = context.scene.blur_mask_radius
            
            # Set grow mask parameter
            prompt[NODES['grow_mask']]["inputs"]["expand"] = context.scene.grow_mask_by
            
            if context.scene.generation_method == 'uv_inpaint':
                # Configure UV inpainting specific prompts
                self._configure_uv_inpainting_mode(prompt, context, render_info, NODES)
            else:  # Sequential mode
                # Configure sequential mode settings
                self._configure_sequential_mode(prompt, context, NODES)

    def _configure_uv_inpainting_mode(self, prompt, context, render_info, NODES):
        """Configures the prompts for UV inpainting mode."""
        # Connect upscale to VAE / InpaintConditioning
        if not context.scene.differential_diffusion:
            prompt[NODES['vae_encode_inpaint']]["inputs"]["pixels"] = [NODES['upscale_uv'], 0]
        else:
            prompt[NODES['inpaint_conditioning']]["inputs"]["pixels"] = [NODES['upscale_uv'], 0]
            # Set the noise_mask flag according to context.scene.differential_noise
            prompt[NODES['inpaint_conditioning']]["inputs"]["noise_mask"] = context.scene.differential_noise

        # Create base UV prompt
        uv_prompt = f"seamless (UV-unwrapped texture) of {context.scene.comfyui_prompt}, consistent material continuity, no visible seams or stretching, PBR material properties"
        uv_prompt_neg = f"seam, stitch, visible edge, texture stretching, repeating pattern, {context.scene.comfyui_negative_prompt}"
        
        prompt[NODES['pos_prompt']]["inputs"]["text"] = uv_prompt
        prompt[NODES['neg_prompt']]["inputs"]["text"] = uv_prompt_neg
        
        # Get the current object name from the file path
        if render_info and 'name' in render_info:
            current_object_name = os.path.basename(render_info['name']).split('.')[0]
        
        # Use the object-specific prompt if available
        object_prompt = self.operator._object_prompts.get(current_object_name, context.scene.comfyui_prompt)
        if object_prompt:
            uv_prompt = f"(UV-unwrapped texture) of {object_prompt}, consistent material continuity, no visible seams or stretching, PBR material properties"
            uv_prompt_neg = f"seam, stitch, visible edge, texture stretching, repeating pattern, {context.scene.comfyui_negative_prompt}"
            prompt[NODES['pos_prompt']]["inputs"]["text"] = uv_prompt
            prompt[NODES['neg_prompt']]["inputs"]["text"] = uv_prompt_neg

    def _configure_ipadapter_refine(self, prompt, context, ipadapter_ref_info, NODES):
        """Configures IPAdapter settings for refinement mode."""
        # Connect IPAdapter output to the appropriate node
        if context.scene.differential_diffusion and context.scene.generation_method not in ('refine', 'local_edit'):
            prompt[NODES['differential_diffusion']]["inputs"]["model"] = [NODES['ipadapter'], 0]
        else:
            prompt[NODES['sampler']]["inputs"]["model"] = [NODES['ipadapter'], 0]
        
        # Set IPAdapter image source
        prompt[NODES['ipadapter_image']]["inputs"]["image"] = ipadapter_ref_info['name']
        
        
        # Connect ipadapter image to the input
        prompt[NODES['ipadapter']]["inputs"]["image"] = [NODES['ipadapter_image'], 0]
        
        # Configure IPAdapter settings
        prompt[NODES['ipadapter']]["inputs"]["weight"] = context.scene.ipadapter_strength
        prompt[NODES['ipadapter']]["inputs"]["start_at"] = context.scene.ipadapter_start
        prompt[NODES['ipadapter']]["inputs"]["end_at"] = context.scene.ipadapter_end
        
        # Set weight type
        weight_type_mapping = {
            'standard': "standard",
            'prompt': "prompt is more important",
            'style': "style transfer"
        }
        prompt[NODES['ipadapter']]["inputs"]["weight_type"] = weight_type_mapping.get(context.scene.ipadapter_weight_type, "standard")

    def _configure_sequential_mode(self, prompt, context, NODES):
        """Configures the prompt for sequential generation mode."""
        # Connect image directly to VAE
        prompt[NODES['vae_encode_inpaint']]["inputs"]["pixels"] = [NODES['input_image'], 0]
        if context.scene.differential_diffusion:
            # Set the noise_mask flag according to context.scene.differential_noise
            prompt[NODES['inpaint_conditioning']]["inputs"]["noise_mask"] = context.scene.differential_noise

    def _refine_configure_images(self, prompt, render_info, NODES):
        """Configures the input images for the refinement process."""
        # Set render image
        if render_info:
            prompt[NODES['render_image']]["inputs"]["image"] = render_info['name']

    def _refine_build_controlnet_chain(self, prompt, context, controlnet_info, NODES):
        """Builds the ControlNet chain for refinement process."""
        # Determine inputs for ControlNet chain
        pos_input = NODES['pos_prompt'] if (not context.scene.differential_diffusion or 
                                context.scene.generation_method in ["grid", "refine", "local_edit"]) else NODES['inpaint_conditioning']
        neg_input = NODES['neg_prompt'] if (not context.scene.differential_diffusion or 
                                context.scene.generation_method in ["grid", "refine", "local_edit"]) else NODES['inpaint_conditioning']
        vae_input = NODES['checkpoint']
        
        # Build the ControlNet chain
        prompt, final = self._build_controlnet_chain_extended(
            context, prompt, pos_input, neg_input, vae_input, 
            controlnet_info
        )
        
        # Connect final outputs to KSampler
        prompt[NODES['sampler']]["inputs"]["positive"] = [final, 0]
        prompt[NODES['sampler']]["inputs"]["negative"] = [final, 1]
        
        return prompt

    def create_base_prompt_flux(self, context):
        """Creates and configures the base Flux prompt.
        Uses prompt_text_flux and does not include negative prompt or LoRA configuration.
        """
        from .util.helpers import prompt_text_flux
        prompt = json.loads(prompt_text_flux)
        # Define node IDs for Flux
        NODES = {
            'pos_prompt': "6",          # CLIPTextEncode for positive prompt
            'vae_loader': "10",         # VAELoader
            'dual_clip': "11",          # DualCLIPLoader
            'unet_loader': "12",        # UNETLoader
            'sampler': "13",            # SamplerCustomAdvanced
            'ksampler': "16",           # KSamplerSelect
            'scheduler': "17",          # BasicScheduler
            'guider': "22",             # BasicGuider
            'noise': "25",              # RandomNoise
            'flux_guidance': "26",      # FluxGuidance
            'latent': "30",             # EmptyLatentImage
            'save_image': "32"          # SaveImageWebsocket
        }
        
        base_prompt_text = context.scene.comfyui_prompt
        # Camera Prompt Injection
        if context.scene.use_camera_prompts and context.scene.generation_method in ['separate', 'sequential', 'refine', 'local_edit'] and self.operator._cameras and self.operator._current_image < len(self.operator._cameras):
            current_camera_name = self.operator._cameras[self.operator._current_image].name
            # Find the prompt in the collection
            prompt_item = next((item for item in context.scene.camera_prompts if item.name == current_camera_name), None)
            if prompt_item and prompt_item.prompt:
                view_desc = prompt_item.prompt
                # Prepend the view description
                base_prompt_text = f"{view_desc}, {base_prompt_text}"
        
        # Set positive prompt only (Flux doesn't use negative prompt)
        prompt[NODES['pos_prompt']]["inputs"]["text"] = base_prompt_text
        
        # Configure sampler parameters
        prompt[NODES['noise']]["inputs"]["noise_seed"] = context.scene.seed
        prompt[NODES['scheduler']]["inputs"]["steps"] = context.scene.steps
        prompt[NODES['scheduler']]["inputs"]["scheduler"] = context.scene.scheduler
        prompt[NODES['flux_guidance']]["inputs"]["guidance"] = context.scene.cfg
        prompt[NODES['ksampler']]["inputs"]["sampler_name"] = context.scene.sampler

        # Replace unet_loader with UNETLoaderGGUF if using GGUF model
        if ".gguf" in context.scene.model_name:
            del prompt[NODES['unet_loader']]
            from .util.helpers import gguf_unet_loader
            unet_loader_dict = json.loads(gguf_unet_loader)
            prompt.update(unet_loader_dict)

        # Set the model name
        prompt[NODES['unet_loader']]["inputs"]["unet_name"] = context.scene.model_name

        # Flux does not use negative prompt or LoRA.
        return prompt, NODES

    def configure_ipadapter_flux(self, prompt, context, ipadapter_ref_info, NODES):
        # Configure IPAdapter if enabled
        from .util.helpers import ipadapter_flux
        ipadapter_dict = json.loads(ipadapter_flux)
        prompt.update(ipadapter_dict)
        
        # Label nodes
        NODES['ipadapter_loader'] = "242"  # IPAdapterFluxLoader
        NODES['ipadapter'] = "243"          # ApplyIPAdapterFlux
        NODES['ipadapter_image'] = "244"    # LoadImage for IPAdapter input
        
        # Connect IPAdapter output to guider and scheduler
        prompt[NODES['guider']]["inputs"]["model"] = [NODES['ipadapter'], 0]
        prompt[NODES['scheduler']]["inputs"]["model"] = [NODES['ipadapter'], 0]
        
        # Set IPAdapter image source
        prompt[NODES['ipadapter_image']]["inputs"]["image"] = ipadapter_ref_info['name']

        # Connect ipadapter image to the input
        prompt[NODES['ipadapter']]["inputs"]["image"] = [NODES['ipadapter_image'], 0]
        
        # Configure IPAdapter settings
        prompt[NODES['ipadapter']]["inputs"]["weight"] = context.scene.ipadapter_strength
        prompt[NODES['ipadapter']]["inputs"]["start_percent"] = context.scene.ipadapter_start
        prompt[NODES['ipadapter']]["inputs"]["end_percent"] = context.scene.ipadapter_end
        
        # There is no weight type for Flux IPAdapter
        
    def generate_flux(self, context, controlnet_info=None, ipadapter_ref_info=None):
        """Generates an image using Flux 1.
        Similar in structure to generate() but uses Flux nodes, skips negative prompt and LoRA.
        """
        from .util.helpers import prompt_text_flux
        server_address = context.preferences.addons[__package__].preferences.server_address
        client_id = str(uuid.uuid4())
        output_dir = context.preferences.addons[__package__].preferences.output_dir

        revision_dir = get_generation_dirs(context)["revision"]

        # Build Flux base prompt and node mapping.
        prompt, NODES = self.create_base_prompt_flux(context)
        
        self._configure_resolution(prompt, context, NODES)
        
        # Configure IPAdapter for Flux if enabled
        if ipadapter_ref_info:
            self.configure_ipadapter_flux(prompt, context, ipadapter_ref_info, NODES)

        # Build ControlNet chain if not using Depth LoRA
        if not context.scene.use_flux_lora:
            prompt, final_node = self._build_controlnet_chain_extended(
                context, prompt, NODES['pos_prompt'], NODES['pos_prompt'], NODES['vae_loader'],
                controlnet_info
            )
        else: # If using Depth LoRA instead of ControlNet, we do not build a ControlNet chain
            final_node = NODES['pos_prompt']  # Use positive prompt directly if not using ControlNet
            # Add Required nodes for the FLUX.1-Depth-dev LoRA
            from .util.helpers import depth_lora_flux
            depth_lora_dict = json.loads(depth_lora_flux)
            prompt.update(depth_lora_dict)

            # Label nodes
            NODES['flux_lora_image'] = "245"  # LoadImage
            NODES['instruct_pix'] = "246"  # InstructPixToPixConditioning
            NODES['flux_lora'] = "247"  # LoraLoaderModelOnly

            # Connect nodes 
            final_node = NODES['instruct_pix'] # To be connected to flux_guidance
            prompt[NODES['sampler']]["inputs"]["latent_image"] = [NODES['instruct_pix'], 2]

            # If using ipadapter, set the apply_ipadapter_flux node to use the flux_lora_image
            _is_original_render_ipadapter = (
                context.scene.sequential_ipadapter
                and context.scene.sequential_ipadapter_mode == 'original_render'
                and context.scene.generation_method == 'local_edit'
            )
            if context.scene.use_ipadapter or _is_original_render_ipadapter or (context.scene.generation_method == 'separate' and context.scene.sequential_ipadapter and self.operator._current_image > 0):
                prompt[NODES['ipadapter']]["inputs"]["model"] = [NODES['flux_lora'], 0]
            else:
                prompt[NODES['guider']]["inputs"]["model"] = [NODES['flux_lora'], 0]
                prompt[NODES['scheduler']]["inputs"]["model"] = [NODES['flux_lora'], 0]

            # Delete unnecessary nodes
            if "239" in prompt:
                del prompt["239"] # SetUnionControlNetType
            if "30" in prompt:
                del prompt["30"] # EmptyLatentImage

            if controlnet_info and "depth" in controlnet_info:
                prompt[NODES['flux_lora_image']]["inputs"]["image"] = controlnet_info["depth"]['name']

        # Connect final node to FluxGuidance
        prompt[NODES['flux_guidance']]["inputs"]["conditioning"] = [final_node, 0]
        # Note: No negative prompt is connected.

        # Save prompt for debugging.
        self._save_prompt_to_file(prompt,  revision_dir)

        # Execute generation via websocket.
        # Execute generation and get results
        ws = self._connect_to_websocket(server_address, client_id)

        if ws is None:
            return {"error": "conn_failed"} # Connection error

        images = None
        try:
            images = self._execute_prompt_and_get_images(ws, prompt, client_id, server_address, NODES)
        finally:
            if ws:
                ws.close()

        if images is None or isinstance(images, dict) and "error" in images:
            return {"error": "conn_failed"}
        
        print(f"Flux image generated with prompt: {context.scene.comfyui_prompt}")

        return images[NODES['save_image']][0]

    def _create_img2img_base_prompt_flux(self, context):
        """Creates and configures the base Flux prompt for img2img refinement."""
        from .util.helpers import prompt_text_img2img_flux
        
        prompt = json.loads(prompt_text_img2img_flux)
        
        # Node IDs organized by functional category for Flux
        NODES = {
            # Text Prompting
            'pos_prompt': "6",          # CLIPTextEncode for positive prompt
            
            # Model Components
            'vae_loader': "10",         # VAELoader
            'dual_clip': "11",          # DualCLIPLoader
            'unet_loader': "12",        # UNETLoader
            
            # Sampling Control
            'sampler': "13",            # SamplerCustomAdvanced
            'ksampler': "16",           # KSamplerSelect
            'scheduler': "17",          # BasicScheduler
            'guider': "22",             # BasicGuider
            'noise': "25",              # RandomNoise
            'flux_guidance': "26",      # FluxGuidance
            
            # Image Processing
            'vae_decode': "8",          # VAEDecode
            'vae_encode': "116",        # VAEEncode
            'vae_encode_inpaint': "44", # VAEEncodeForInpaint
            'upscale': "118",           # ImageScale for upscaling
            'upscale_uv': "43",         # ImageScale for UV maps
            
            # Input Images
            'input_image': "1",         # LoadImage for input
            'mask_image': "42",         # LoadImage for mask
            'render_image': "117",      # LoadImage for render
            
            # Mask Processing
            'grow_mask': "224",         # GrowMask
            'blur': "226",              # ImageBlur
            'image_to_mask': "227",     # ImageToMask
            'mask_to_image': "225",     # MaskToImage
            
            # Advanced Features
            'differential_diffusion': "50", # DifferentialDiffusion for Flux
            'inpaint_conditioning': "51",   # InpaintModelConditioning for Flux
            
            # Latent Space
            'latent': "30",             # EmptyLatentImage
            
            # Output
            'save_image': "32"          # SaveImageWebsocket
        }
        
        base_prompt_text = context.scene.comfyui_prompt
        # Camera Prompt Injection
        if context.scene.use_camera_prompts and context.scene.generation_method in ['separate', 'sequential', 'refine', 'local_edit', 'grid'] and self.operator._cameras and self.operator._current_image < len(self.operator._cameras):
            current_camera_name = self.operator._cameras[self.operator._current_image].name
            # Find the prompt in the collection
            prompt_item = next((item for item in context.scene.camera_prompts if item.name == current_camera_name), None)
            if prompt_item and prompt_item.prompt:
                view_desc = prompt_item.prompt
                # Prepend the view description
                base_prompt_text = f"{view_desc}, {base_prompt_text}"
        
        # Set positive prompt (Flux doesn't use negative prompt)
        prompt[NODES['pos_prompt']]["inputs"]["text"] = base_prompt_text
        
        # Configure sampler parameters
        prompt[NODES['noise']]["inputs"]["noise_seed"] = context.scene.seed
        prompt[NODES['scheduler']]["inputs"]["steps"] = context.scene.refine_steps if context.scene.generation_method == 'grid' else context.scene.steps
        prompt[NODES['scheduler']]["inputs"]["denoise"] = context.scene.denoise if context.scene.generation_method in ['grid', 'refine', 'local_edit'] else 1.0
        prompt[NODES['flux_guidance']]["inputs"]["guidance"] = context.scene.refine_cfg if context.scene.generation_method == 'grid' else context.scene.cfg
        prompt[NODES['ksampler']]["inputs"]["sampler_name"] = context.scene.refine_sampler if context.scene.generation_method == 'grid' else context.scene.sampler
        prompt[NODES['scheduler']]["inputs"]["scheduler"] = context.scene.refine_scheduler if context.scene.generation_method == 'grid' else context.scene.scheduler

        # Replace unet_loader with UNETLoaderGGUF if using GGUF model
        if ".gguf" in context.scene.model_name:
            del prompt[NODES['unet_loader']]
            from .util.helpers import gguf_unet_loader
            unet_loader_dict = json.loads(gguf_unet_loader)
            prompt.update(unet_loader_dict)

        # Set the model name
        prompt[NODES['unet_loader']]["inputs"]["unet_name"] = context.scene.model_name
        
        # Configure upscale settings
        prompt[NODES['upscale']]["inputs"]["upscale_method"] = context.scene.refine_upscale_method
        prompt[NODES['upscale']]["inputs"]["width"] = context.scene.render.resolution_x
        prompt[NODES['upscale']]["inputs"]["height"] = context.scene.render.resolution_y
        
        # Configure UV upscale settings
        prompt[NODES['upscale_uv']]["inputs"]["upscale_method"] = "nearest-exact"
        prompt[NODES['upscale_uv']]["inputs"]["width"] = 1024
        prompt[NODES['upscale_uv']]["inputs"]["height"] = 1024
        
        # Configure mask settings
        prompt[NODES['grow_mask']]["inputs"]["expand"] = context.scene.grow_mask_by
        prompt[NODES['blur']]["inputs"]["blur_radius"] = context.scene.blur_mask_radius
        prompt[NODES['blur']]["inputs"]["sigma"] = context.scene.blur_mask_sigma
        
        return prompt, NODES

    def refine_flux(self, context, controlnet_info=None, mask_info=None, render_info=None, ipadapter_ref_info=None):
        """     
        Refines the image using Flux 1 in ComfyUI.         
        :param context: Blender context.         
        :param controlnet_info: Dict of uploaded controlnet image info.
        :param mask_info: Uploaded mask image info.
        :param render_info: Uploaded render image info.
        :param ipadapter_ref_info: Uploaded IPAdapter reference image info.
        :return: Refined image.     
        """
        # Setup connection parameters
        server_address = context.preferences.addons[__package__].preferences.server_address
        client_id = str(uuid.uuid4())
        output_dir = context.preferences.addons[__package__].preferences.output_dir

        revision_dir = get_generation_dirs(context)["revision"]

        # Initialize the img2img prompt template for Flux
        prompt, NODES = self._create_img2img_base_prompt_flux(context)
        
        # Configure IPAdapter for Flux if enabled
        if ipadapter_ref_info and context.scene.generation_method != 'uv_inpaint':
            self.configure_ipadapter_flux(prompt, context, ipadapter_ref_info, NODES)
        
        # Configure based on generation method
        self._configure_refinement_mode_flux(prompt, context, render_info, mask_info, ipadapter_ref_info, NODES)
        
        # Set up image inputs for different controlnet types
        self._refine_configure_images_flux(prompt, render_info, NODES)
        
        # Build ControlNet chain if not using Depth LoRA
        if not context.scene.generation_method == 'uv_inpaint':
            if not context.scene.use_flux_lora:
                prompt = self._refine_build_controlnet_chain_flux(
                    context, prompt, controlnet_info, NODES
                )
            else: # If using Depth LoRA instead of ControlNet, we do not build a ControlNet chain
                final_node = NODES['pos_prompt']  # Use positive prompt directly if not using ControlNet
                # Add Required nodes for the FLUX.1-Depth-dev LoRA
                from .util.helpers import depth_lora_flux
                depth_lora_dict = json.loads(depth_lora_flux)
                prompt.update(depth_lora_dict)

                # Label nodes
                NODES['flux_lora_image'] = "245"  # LoadImage
                NODES['instruct_pix'] = "246"  # InstructPixToPixConditioning
                NODES['flux_lora'] = "247"  # LoraLoaderModelOnly

                # Configure InstructPixToPixConditioning inputs to InpaintModelConditioning if using differential diffusion
                if context.scene.differential_diffusion:
                    prompt[NODES['instruct_pix']]["inputs"]["positive"] = [NODES['inpaint_conditioning'], 0]
                    prompt[NODES['instruct_pix']]["inputs"]["negative"] = [NODES['inpaint_conditioning'], 1]

                # Connect nodes 
                prompt[NODES['flux_guidance']]["inputs"]["conditioning"] = [NODES['instruct_pix'], 0]
                # prompt[NODES['sampler']]["inputs"]["latent_image"] = [NODES['instruct_pix'], 2] # Not doing since we need to respect the mask

                # If using ipadapter, set the apply_ipadapter_flux node to use the flux_lora_image
                if ipadapter_ref_info and context.scene.generation_method != 'uv_inpaint':
                    prompt[NODES['ipadapter']]["inputs"]["model"] = [NODES['flux_lora'], 0]
                    prompt[NODES['differential_diffusion']]["inputs"]["model"] = [NODES['ipadapter'], 0]
                else:
                    prompt[NODES['guider']]["inputs"]["model"] = [NODES['flux_lora'], 0]
                    prompt[NODES['scheduler']]["inputs"]["model"] = [NODES['flux_lora'], 0]
                    prompt[NODES['differential_diffusion']]["inputs"]["model"] = [NODES['flux_lora'], 0]

                # Delete unnecessary nodes
                if "239" in prompt:
                    del prompt["239"] # SetUnionControlNetType
                if "30" in prompt:
                    del prompt["30"] # EmptyLatentImage

                # Set the image for the Flux LoRA
                if controlnet_info and "depth" in controlnet_info:
                    prompt[NODES['flux_lora_image']]["inputs"]["image"] = controlnet_info["depth"]['name']
        
        # Save prompt for debugging
        self._save_prompt_to_file(prompt, revision_dir)
        
        # Execute generation and get results
        ws = self._connect_to_websocket(server_address, client_id)

        if ws is None:
            return {"error": "conn_failed"} # Connection error

        images = None
        try:
            images = self._execute_prompt_and_get_images(ws, prompt, client_id, server_address, NODES)
        finally:
            if ws:
                ws.close()

        if images is None or isinstance(images, dict) and "error" in images:
            return {"error": "conn_failed"}
        
        print(f"Image refined with Flux using prompt: {context.scene.comfyui_prompt}")
        
        # Return the refined image
        return images[NODES['save_image']][0]

    def _configure_refinement_mode_flux(self, prompt, context, render_info, mask_info, ipadapter_ref_info, NODES):
        """Configures the prompt based on the specific refinement mode for Flux."""
        # Configure based on generation method
        if context.scene.generation_method in ('refine', 'local_edit'):
            # Configure for refine mode - load render directly
            if render_info:
                prompt[NODES['render_image']]["inputs"]["image"] = render_info['name']
                prompt[NODES['vae_encode']]["inputs"]["pixels"] = [NODES['render_image'], 0]
            # Connect latent to sampler
            prompt[NODES['sampler']]["inputs"]["latent_image"] = [NODES['vae_encode'], 0]
        
        elif context.scene.generation_method in ['uv_inpaint', 'sequential']:
            # Configure for inpainting modes
            if mask_info:
                prompt[NODES['mask_image']]["inputs"]["image"] = mask_info['name']
            if render_info:
                prompt[NODES['input_image']]["inputs"]["image"] = render_info['name']
            
            # Configure mask processing
            if not context.scene.blur_mask:
                prompt[NODES['vae_encode_inpaint']]["inputs"]["mask"] = [NODES['grow_mask'], 0]
                if context.scene.differential_diffusion:
                    prompt[NODES['inpaint_conditioning']]["inputs"]["mask"] = [NODES['grow_mask'], 0]
            else:
                # Configure blur chain
                prompt[NODES['image_to_mask']]["inputs"]["image"] = [NODES['blur'], 0]
                prompt[NODES['vae_encode_inpaint']]["inputs"]["mask"] = [NODES['image_to_mask'], 0]
            
            # Different setups based on differential diffusion
            if context.scene.differential_diffusion:
                # Connect differential diffusion between model loader and other components
                prompt[NODES['guider']]["inputs"]["model"] = [NODES['differential_diffusion'], 0]
                prompt[NODES['scheduler']]["inputs"]["model"] = [NODES['differential_diffusion'], 0]
                
                # Connect inpaint conditioning to differential diffusion
                if ipadapter_ref_info:
                    prompt[NODES['differential_diffusion']]["inputs"]["model"] = [NODES['ipadapter'], 0]
                else:
                    prompt[NODES['differential_diffusion']]["inputs"]["model"] = [NODES['unet_loader'], 0]
                
                # Configure inpaint conditioning with proper input image and mask
                prompt[NODES['inpaint_conditioning']]["inputs"]["pixels"] = [
                    NODES['upscale_uv'], 0
                ] if context.scene.generation_method == 'uv_inpaint' else [NODES['input_image'], 0]
                
                # Connect latent to sampler from inpaint conditioning
                prompt[NODES['sampler']]["inputs"]["latent_image"] = [NODES['inpaint_conditioning'], 2]
                
                # Connect conditioning to flux_guidance
                prompt[NODES['flux_guidance']]["inputs"]["conditioning"] = [NODES['inpaint_conditioning'], 0]
            else:
                # Standard setup without differential diffusion
                prompt[NODES['sampler']]["inputs"]["latent_image"] = [NODES['vae_encode_inpaint'], 0]
            
            if context.scene.generation_method == 'uv_inpaint':
                self._configure_uv_inpainting_mode_flux(prompt, context, render_info, NODES)
            else:  # Sequential mode
                self._configure_sequential_mode_flux(prompt, context, NODES)

    def _configure_uv_inpainting_mode_flux(self, prompt, context, render_info, NODES):
        """Configures the prompts for UV inpainting mode in Flux."""
        # UV inpainting specific configuration
        prompt[NODES['upscale_uv']]["inputs"]["image"] = [NODES['input_image'], 0]
        
        if not context.scene.differential_diffusion:
            prompt[NODES['vae_encode_inpaint']]["inputs"]["pixels"] = [NODES['upscale_uv'], 0]
        else:
            # Set the noise_mask flag according to context.scene.differential_noise
            prompt[NODES['inpaint_conditioning']]["inputs"]["noise_mask"] = context.scene.differential_noise
        
        # Create UV-specific prompt
        uv_prompt = f"seamless (UV-unwrapped texture) of {context.scene.comfyui_prompt}, consistent material continuity, no visible seams or stretching"
        prompt[NODES['pos_prompt']]["inputs"]["text"] = uv_prompt
        
        # Object-specific prompt if available
        if render_info and 'name' in render_info:
            current_object_name = os.path.basename(render_info['name']).split('.')[0]
            object_prompt = self.operator._object_prompts.get(current_object_name, context.scene.comfyui_prompt)
            if object_prompt:
                uv_prompt = f"(UV-unwrapped texture) of {object_prompt}, consistent material continuity, no visible seams or stretching"
                prompt[NODES['pos_prompt']]["inputs"]["text"] = uv_prompt

    def _configure_sequential_mode_flux(self, prompt, context, NODES):
        """Configures the prompt for sequential generation mode in Flux."""
        # Direct connection for sequential mode
        if not context.scene.differential_diffusion:
            prompt[NODES['vae_encode_inpaint']]["inputs"]["pixels"] = [NODES['input_image'], 0]
        else:
            # Set the noise_mask flag according to context.scene.differential_noise
            prompt[NODES['inpaint_conditioning']]["inputs"]["noise_mask"] = context.scene.differential_noise

    def _refine_configure_images_flux(self, prompt, render_info, NODES):
        """Configures the input images for the refinement process in Flux."""
        # Set render image if provided
        if render_info:
            prompt[NODES['render_image']]["inputs"]["image"] = render_info['name']
        
        # Control images are handled by the controlnet chain builder

    def _refine_build_controlnet_chain_flux(self, context, prompt, controlnet_info, NODES):
        """Builds the ControlNet chain for refinement process with Flux."""
        input = NODES['pos_prompt'] if not context.scene.differential_diffusion else NODES['inpaint_conditioning']
        # For Flux, the controlnet chain connects to the guidance node
        prompt, final_node = self._build_controlnet_chain_extended(
            context, prompt, input, input, NODES['vae_loader'],
            controlnet_info
        )
        # Connect final node to FluxGuidance conditioning input
        prompt[NODES['flux_guidance']]["inputs"]["conditioning"] = [final_node, 0]
        return prompt
