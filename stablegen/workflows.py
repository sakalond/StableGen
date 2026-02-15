import bpy
import os
import json
import uuid
import websocket
import socket
import urllib.request

from .util.helpers import prompt_text_qwen_image_edit
from .utils import get_generation_dirs

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
        img = Image.open(buf).convert("RGBA")

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
        if context.scene.use_ipadapter or (context.scene.generation_method == 'separate' and context.scene.sequential_ipadapter and self.operator._current_image > 0):
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

    def _connect_to_websocket(self, server_address, client_id):
        """Establishes WebSocket connection to ComfyUI server."""
        try:
            ws = websocket.WebSocket()
            ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
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

    def _execute_prompt_and_get_images(self, ws, prompt, client_id, server_address, NODES):
        """Executes the prompt and collects generated images."""
        # Send the prompt to the queue
        prompt_id = self._queue_prompt(prompt, client_id, server_address)
        
        # Process the WebSocket messages and collect images
        output_images = {}
        current_node = ""
        
        while True:
            out = ws.recv()
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
                        print(f"Progress: {progress:.1f}%")
            else:
                # Binary data (image)
                if current_node == NODES['save_image']:  # SaveImageWebsocket node
                    print("Receiving generated image")
                    images_output = output_images.get(current_node, [])
                    images_output.append(out[8:])  # Skip the first 8 bytes (header)
                    output_images[current_node] = images_output
        
        return output_images

    def _queue_prompt(self, prompt, client_id, server_address):
        """Queues the prompt for processing by ComfyUI."""
        try:
            data = json.dumps({
                "prompt": prompt,
                "client_id": client_id
            }).encode('utf-8')
            
            req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
            response = json.loads(urllib.request.urlopen(req).read())
            
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
        if (context.scene.use_ipadapter or _is_original_render_ipadapter or (context.scene.sequential_ipadapter and self.operator._current_image > 0)) and context.scene.generation_method != 'uv_inpaint':
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
