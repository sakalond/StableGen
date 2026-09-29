from ..feature_flags import experimental_enabled
import bpy
import os
import json
import uuid
import random
import urllib.request
import urllib.parse
from datetime import datetime

from ..util.workflow_templates import prompt_text_qwen_image_edit, prompt_text_flux2_klein
from ..utils import get_generation_dirs
from ..timeout_config import get_timeout
from .graph_validation import (
    validate_workflow_graph,
    qwen_style_semantic_slot,
    qwen_reference_latent_slots,
    attach_qwen_latent_lock,
    sdxl_final_mask_reference,
)
from .qwen_models import (
    QWEN_IMAGE_21_TEXT_ENCODER,
    QWEN_IMAGE_21_VAE,
    qwen_prepared_prompt,
    qwen_image_21_prompt,
    qwen_image_21_rgba_prompt,
    remap_qwen_image_21_prompt,
)

from io import BytesIO
import numpy as np
from PIL import Image

_ADDON_PKG = __package__.rsplit('.', 1)[0]


def _immutable_view_prompt(operator):
    """Return the prompt frozen into the canonical generation view plan."""
    index = getattr(operator, '_current_image', -1)
    plan = getattr(operator, '_view_plan', ()) or ()
    if 0 <= index < len(plan):
        return plan[index].prompt or ""
    # Legacy camera-prompt lookup when the canonical-plan experiment is off.
    cameras = getattr(operator, '_cameras', ()) or ()
    if 0 <= index < len(cameras):
        camera_name = cameras[index].name
        item = next(
            (entry for entry in bpy.context.scene.camera_prompts
             if entry.name == camera_name),
            None,
        )
        return item.prompt if item and item.prompt else ""
    return ""


def _texturing_prompt(scene):
    """Return the texturing portion of the prompt.

    Priority order:
    1. Dedicated ``texture_prompt`` field (if the separate-prompt toggle is on
       and the field is non-empty).
    2. Legacy ``||`` separator in the main prompt (text after ``||``).
    3. The full main prompt as-is.
    """
    if getattr(scene, 'use_separate_texture_prompt', False):
        tex = getattr(scene, 'texture_prompt', '').strip()
        if tex:
            return tex
    raw = scene.comfyui_prompt
    if '||' in raw:
        return raw.split('||', 1)[1].strip()
    return raw


def _generation_prompt(scene):
    """Return the generation / mesh-creation portion of the prompt.

    When the separate-texture-prompt toggle is on the entire main prompt is
    treated as the generation prompt (no ``||`` splitting).  Otherwise the
    legacy ``||`` separator is respected.
    """
    raw = scene.comfyui_prompt
    if getattr(scene, 'use_separate_texture_prompt', False):
        return raw.strip()
    if '||' in raw:
        return raw.split('||', 1)[0].strip()
    return raw


class _TexturingWorkflowMixin:
    def _queue_prompt(self, prompt, client_id, server_address):
        """Validate every dynamically edited texturing graph at submission."""
        issues = validate_workflow_graph(prompt)
        if issues:
            message = "Invalid texturing workflow: " + "; ".join(issues)
            self.operator._error = message
            raise ValueError(message)
        return super()._queue_prompt(prompt, client_id, server_address)

    """Texturing workflow methods for WorkflowManager."""

    def _convert_to_qwen_image_21(
        self, prompt, nodes, context, *, projected_context_target=False
    ):
        """Replace the legacy Qwen Edit graph with Qwen-Image 2.1 nodes.

        The caller first performs StableGen's normal reference routing.  This
        adapter reads those semantic image slots, retains their loader/resize
        dependencies, and swaps only the ComfyUI model layer.  Camera planning,
        magenta guides, depth references and sequential projection therefore
        remain shared with the established Qwen path.
        """
        old_encoder = prompt[nodes['pos_prompt']]
        old_inputs = old_encoder.get('inputs', {})
        image_refs = [
            old_inputs[key]
            for key in ('image1', 'image2', 'image3')
            if key in old_inputs
        ]
        prompt_text = old_inputs.get('prompt', '')
        if projected_context_target and len(image_refs) >= 2:
            # The legacy graph routes structure/depth as image 1 and projected
            # context as image 2. Qwen 2.1 VAE-encodes image 1 as the output
            # target, so put the context there and retain structure as image 2.
            image_refs[0], image_refs[1] = image_refs[1], image_refs[0]
            final_prompt = remap_qwen_image_21_prompt(
                prompt_text, {1: 2, 2: 1}
            )
        else:
            final_prompt = qwen_image_21_prompt(prompt_text)
        if getattr(context.scene, 'qwen_image_21_native_rgba', True):
            final_prompt = qwen_image_21_rgba_prompt(final_prompt)

        graph = {
            nodes['unet_loader']: {
                'inputs': {
                    'unet_name': context.scene.model_name,
                    'weight_dtype': 'default',
                },
                'class_type': 'UNETLoader',
                '_meta': {'title': 'Load Qwen-Image 2.1 Diffusion Model'},
            },
            '3': {
                'inputs': {
                    'clip_name': QWEN_IMAGE_21_TEXT_ENCODER,
                    'type': 'qwen_image',
                    'device': 'default',
                },
                'class_type': 'CLIPLoader',
                '_meta': {'title': 'Load Qwen3-VL 8B Text Encoder'},
            },
            '4': {
                'inputs': {'vae_name': QWEN_IMAGE_21_VAE},
                'class_type': 'VAELoader',
                '_meta': {'title': 'Load Qwen-Image 2.1 VAE'},
            },
            nodes['pos_prompt']: {
                'inputs': {
                    'clip': ['3', 0],
                    'prompt': final_prompt,
                    'negative_prompt': getattr(
                        context.scene, 'comfyui_negative_prompt', ''
                    ),
                    # A camera reference defines the output canvas. Preserve
                    # its 32px-aligned size instead of rescaling it to 1 MP.
                    # Text-only generation has no reference size, so retain
                    # ComfyUI's explicit 1024px square latent fallback.
                    'resolution': 0 if image_refs else 1024,
                    'vae': ['4', 0],
                },
                'class_type': 'TextEncodeQwenImage21',
                '_meta': {'title': 'Qwen-Image 2.1 Multi-Image Conditioning'},
            },
            nodes['model_sampler']: {
                'inputs': {
                    'model': [nodes['unet_loader'], 0],
                    'device': 'auto',
                    'dtype': 'default',
                },
                'class_type': 'QwenImage21Cache',
                '_meta': {'title': 'Qwen-Image 2.1 Cache'},
            },
            nodes['sampler']: {
                'inputs': {
                    'model': [nodes['model_sampler'], 0],
                    'seed': context.scene.seed,
                    'steps': context.scene.steps,
                    'cfg': context.scene.cfg,
                    'sampler_name': context.scene.sampler,
                    'scheduler': context.scene.scheduler,
                    'positive': [nodes['pos_prompt'], 0],
                    'negative': [nodes['pos_prompt'], 1],
                    'latent_image': [nodes['pos_prompt'], 2],
                    'denoise': 1.0,
                },
                'class_type': 'KSampler',
                '_meta': {'title': 'Qwen-Image 2.1 Sampler'},
            },
            '2': {
                'inputs': {
                    'samples': [nodes['sampler'], 0],
                    'vae': ['4', 0],
                },
                'class_type': 'VAEDecode',
                '_meta': {'title': 'VAE Decode'},
            },
            nodes['save_image']: {
                'inputs': {'images': ['2', 0]},
                'class_type': 'SaveImageWebsocket',
                '_meta': {'title': 'SaveImageWebsocket'},
            },
        }

        # Dynamic Qwen 2.1 image inputs are named images.image_1, etc.  Copy
        # any resize/load chain used by a semantic slot into the replacement
        # graph so external-style scaling continues to work.
        def retain_reference(reference):
            node_id = str(reference[0])
            if node_id in graph:
                return
            source = prompt.get(node_id)
            if source is None:
                return
            graph[node_id] = source
            for value in source.get('inputs', {}).values():
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and isinstance(value[0], (str, int))
                    and isinstance(value[1], int)
                ):
                    retain_reference(value)

        for index, reference in enumerate(image_refs, 1):
            retain_reference(reference)
            graph[nodes['pos_prompt']]['inputs'][f'images.image_{index}'] = reference

        prompt.clear()
        prompt.update(graph)
        print(
            '[StableGen] Qwen-Image 2.1 architecture: using '
            f'{QWEN_IMAGE_21_TEXT_ENCODER}, {QWEN_IMAGE_21_VAE}, '
            f'{graph[nodes["sampler"]]["inputs"]["steps"]} steps and no legacy LoRA.'
        )

    def _crop_and_vignette(
        self,
        img_bytes,
        border_px: int = 0,
        feather: float = 0.15,
        gamma: float = 0.7,
    ):
        """
        Apply an alpha vignette without changing image registration.

        IMPORTANT:
          - RGB is NOT darkened here, only alpha is shaped.
          - This assumes the shader uses alpha to blend with the underlying
            surface (e.g. Mix using alpha as Fac).

        border_px : deprecated and ignored; sequential outputs stay full-size.
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

        # Corrected mode never crops because projection registration is not
        # adjusted for a changed image extent.
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

    def _build_qwen_reference_latent_chain(
        self,
        prompt,
        NODES,
        start_node_id=700,
        latent_slots=("image1", "image2", "image3"),
    ):
        """
        Build VAEEncode + ReferenceLatent chain for the Qwen unzoom fix.

        When the VAE input is disconnected from TextEncodeQwenImageEditPlus,
        the node no longer creates reference latents (and no longer forces
        a 1 MP rescale).  We replicate the reference-latent injection
        manually so the latents match the actual output resolution.

        For each selected, active image still present in the positive
        text-encode node, a VAEEncode node is created and a ReferenceLatent
        node is chained onto both positive and negative conditioning. The
        KSampler inputs are then updated to point at the tail of the chain.
        A slot omitted from ``latent_slots`` remains available to the Qwen
        visual-language encoder but exerts no full-resolution reference-latent
        pressure.

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
            if key in pos_inputs and key in latent_slots:
                active_refs.append((key, pos_inputs[key]))

        if not active_refs:
            return

        node_id = start_node_id
        # Avoid collisions with existing dynamic nodes
        while str(node_id) in prompt:
            node_id += 1

        pos_cond_ref = [pos_node, 0]
        neg_cond_ref = [neg_node, 0]

        for i, (slot, img_ref) in enumerate(active_refs):
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
                    "_meta": {"title": f"VAE Encode ({slot})"}
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
                "_meta": {"title": f"Reference Latent Pos ({slot})"}
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
                "_meta": {"title": f"Reference Latent Neg ({slot})"}
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

    # ------------------------------------------------------------------
    # FLUX.2 Klein – build ReferenceLatent chain for multi-reference editing
    # ------------------------------------------------------------------
    def _build_flux2_reference_chain(self, prompt, NODES, ref_image_infos, start_node_id=700):
        """Build a ReferenceLatent chain for FLUX.2 Klein (based on official workflow).

        For each reference image, creates:
            LoadImage (already present) → VAEEncode
                → ReferenceLatent (positive conditioning)
                → ReferenceLatent (negative conditioning)

        The ReferenceLatent nodes are chained so each successive reference
        appends to the previous one.  The final modified positive and negative
        conditioning outputs are wired into CFGGuider.

        Args:
            prompt:          The ComfyUI prompt dict (modified in-place).
            NODES:           Node-ID mapping dict.
            ref_image_infos: List of dicts, each with at least a ``loader_node``
                             key pointing to the LoadImage node ID.
            start_node_id:   First dynamic node ID to use.
        """
        if not ref_image_infos:
            return

        pos_node = NODES['pos_prompt']
        neg_node = NODES['neg_prompt']
        guider_node = NODES['guider']
        vae_ref = [NODES['vae_loader'], 0]

        node_id = start_node_id
        while str(node_id) in prompt:
            node_id += 1

        pos_cond_ref = [pos_node, 0]
        neg_cond_ref = [neg_node, 0]

        for i, ref_info in enumerate(ref_image_infos):
            loader_node = ref_info['loader_node']

            # --- VAEEncode (reference image → latent) ---
            vae_enc_id = str(node_id); node_id += 1
            prompt[vae_enc_id] = {
                "inputs": {
                    "pixels": [loader_node, 0],
                    "vae": vae_ref
                },
                "class_type": "VAEEncode",
                "_meta": {"title": f"VAE Encode Ref {i + 1}"}
            }

            # --- ReferenceLatent (positive) ---
            ref_pos_id = str(node_id); node_id += 1
            prompt[ref_pos_id] = {
                "inputs": {
                    "conditioning": pos_cond_ref,
                    "latent": [vae_enc_id, 0]
                },
                "class_type": "ReferenceLatent",
                "_meta": {"title": f"Reference Latent Pos {i + 1}"}
            }
            pos_cond_ref = [ref_pos_id, 0]

            # --- ReferenceLatent (negative) ---
            ref_neg_id = str(node_id); node_id += 1
            prompt[ref_neg_id] = {
                "inputs": {
                    "conditioning": neg_cond_ref,
                    "latent": [vae_enc_id, 0]
                },
                "class_type": "ReferenceLatent",
                "_meta": {"title": f"Reference Latent Neg {i + 1}"}
            }
            neg_cond_ref = [ref_neg_id, 0]

        # Wire final pos/neg conditioning into CFGGuider
        prompt[guider_node]['inputs']['positive'] = pos_cond_ref
        prompt[guider_node]['inputs']['negative'] = neg_cond_ref

    def _get_klein_default_prompts(self, context, is_initial_image):
        """Gets the default Klein prompts based on the current context.

        Follows BFL's official Klein prompting guide: write like a novelist,
        front-load the subject, reference image numbers explicitly, and
        describe the desired transformation rather than the image contents.
        Camera prompts are placed after the main prompt via {camera_suffix}
        to reduce their weighting (Klein pays more attention to what comes
        first).
        """
        is_generating = hasattr(self.operator, '_current_image')
        is_subsequent_in_sequence = is_generating and self.operator._current_image > 0

        if is_initial_image:
            style_image_provided = context.scene.qwen_use_external_style_image
        else:
            style_image_provided = (
                context.scene.qwen_use_external_style_image or
                context.scene.sequential_ipadapter or
                context.scene.qwen_context_render_mode in {'REPLACE_STYLE', 'ADDITIONAL'}
            )
        context_mode = context.scene.qwen_context_render_mode

        if is_initial_image:
            if not style_image_provided:
                return "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour."
            else:
                return "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour, adopting the visual style from image 2."
        else:
            if context_mode == 'ADDITIONAL':
                return "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray. Image 3 represents the overall style."
            elif context_mode == 'REPLACE_STYLE':
                return "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour. In image 2, replace all solid magenta areas with content that continues the surrounding style. Replace the background with solid gray."
            else:
                if not style_image_provided:
                    return "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour."
                else:
                    return "Reskin this into {main_prompt}{camera_suffix}, while preserve identity keep likeness replica contour, adopting the visual style from image 2."

    def _get_qwen_default_prompts(
        self,
        context,
        is_initial_image,
        template_id=None,
        context_mode_override=None,
    ):
        """Get the selected Qwen guidance template for the current image roles.

        ``template_id`` is mainly used by the reset operator to retrieve the
        architecture's built-in wording while the UI is set to Custom.
        """
        # Check if we are in a real generation context vs. just resetting a prompt
        is_generating = hasattr(self.operator, '_current_image')
        is_subsequent_in_sequence = is_generating and self.operator._current_image > 0

        # TRELLIS.2 style reference counts as a style image for prompt selection.
        _t2_style = (
            getattr(context.scene, 'architecture_mode', '') == 'trellis2'
            and getattr(context.scene, 'qwen_use_trellis2_style', False)
        )
        context_mode = (
            context_mode_override
            if context_mode_override is not None
            else context.scene.qwen_context_render_mode
        )

        if is_initial_image:
            # First frame has a style reference from an external image or the TRELLIS.2 input.
            style_image_provided = (
                context.scene.qwen_use_external_style_image
                or _t2_style
            )
        else:
            # Later frames can draw style from previous renders, external sources,
            # context renders, or the TRELLIS.2 input (when not initial-only).
            style_image_provided = (
                context.scene.qwen_use_external_style_image or
                context.scene.sequential_ipadapter or
                context_mode in {'REPLACE_STYLE', 'ADDITIONAL'} or
                (_t2_style and not getattr(context.scene, 'qwen_trellis2_style_initial_only', False))
            )
        is_qwen21 = context.scene.model_architecture == 'qwen_image_21'

        architecture_default = (
            'QWEN_21_DEFAULT' if is_qwen21 else 'QWEN_EDIT_LEGACY'
        )
        selected_template = template_id or getattr(
            context.scene, 'qwen_prompt_template', architecture_default
        )
        # Compatibility with scenes saved by the short-lived combined entry.
        if selected_template == 'STABLEGEN_DEFAULT':
            selected_template = architecture_default
        if selected_template == 'CUSTOM':
            if is_initial_image:
                return context.scene.qwen_custom_prompt_initial
            if context_mode == 'ADDITIONAL':
                return context.scene.qwen_custom_prompt_seq_additional
            if context_mode == 'REPLACE_STYLE':
                return context.scene.qwen_custom_prompt_seq_replace
            return context.scene.qwen_custom_prompt_seq_none

        if selected_template not in {
            'QWEN_EDIT_LEGACY', 'QWEN_21_DEFAULT'
        }:
            # Initial-only TRELLIS styling falls back to the first generated
            # view on later cameras, so an appearance reference still exists.
            trellis_style_available = _t2_style
            appearance_reference = (
                context.scene.qwen_use_external_style_image
                or trellis_style_available
                or (
                    not is_initial_image
                    and context.scene.sequential_ipadapter
                )
            )
            has_context_canvas = (
                not is_initial_image
                and context_mode in {'REPLACE_STYLE', 'ADDITIONAL'}
            )
            return qwen_prepared_prompt(
                selected_template,
                canvas_image=2 if has_context_canvas else 1,
                structure_image=1 if has_context_canvas else None,
                appearance_image=(
                    3 if has_context_canvas and context_mode == 'ADDITIONAL'
                    and appearance_reference
                    else 2 if not has_context_canvas and appearance_reference
                    else None
                ),
                has_magenta_regions=has_context_canvas,
                gray_background=(
                    has_context_canvas
                    and not is_qwen21
                    and getattr(
                        context.scene, 'qwen_prompt_gray_background', True
                    )
                ),
            )

        # Optional sentence injected into context-render prompts.
        gray_bg = " Replace the background with solid gray." if getattr(context.scene, 'qwen_prompt_gray_background', True) else ""

        if selected_template == 'QWEN_21_DEFAULT' and not is_initial_image and context_mode in {
            'REPLACE_STYLE', 'ADDITIONAL'
        }:
            # These refer to the legacy semantic slots before conversion:
            # image 2 is projected context, image 1 is structure. The Qwen 2.1
            # adapter swaps them so the context becomes its required image-1
            # edit target and also renumbers custom/default prompt references.
            prompt = (
                "Edit image 2 into '{main_prompt}'. Preserve its existing "
                "non-magenta textured regions. Replace every solid magenta "
                "region with texture that seamlessly continues the surrounding "
                "surface. Use image 1 only as a geometry and composition guide."
            )
            if context_mode == 'ADDITIONAL':
                prompt += " Use image 3 as the overall appearance reference."
            return prompt

        if selected_template == 'QWEN_21_DEFAULT' and is_initial_image:
            prompt = (
                "Create '{main_prompt}' using image 1 as the geometry and "
                "composition guide."
            )
            if style_image_provided:
                prompt += " Use image 2 as the overall appearance reference."
            return prompt

        if (selected_template == 'QWEN_21_DEFAULT'
                and context_mode not in {'REPLACE_STYLE', 'ADDITIONAL'}):
            prompt = (
                "Create '{main_prompt}' using image 1 as the geometry and "
                "composition guide."
            )
            if style_image_provided:
                prompt += " Use image 2 as the overall appearance reference."
            return prompt

        if is_initial_image:
            if not style_image_provided:
                return "Change the format of image 1 to '{main_prompt}'"
            else:
                return "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2"
        else: # Subsequent image
            if context_mode == 'ADDITIONAL':
                return f"Change and transfer the format of image 1 to '{{main_prompt}}'. Replace all solid magenta areas in image 2.{gray_bg} The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object."
            elif context_mode == 'REPLACE_STYLE':
                return f"Change and transfer the format of image 1 to '{{main_prompt}}'. Replace all solid magenta areas in image 2.{gray_bg} The style from image 2 should smoothly continue into the previously magenta areas."
            else: # NONE or other cases
                if not style_image_provided:
                     return "Change the format of image 1 to '{main_prompt}'"
                else:
                    return "Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2"


    def generate_qwen_refine(self, context, camera_id=None, masked_edit=None):
        """Generates an image using the Qwen-Image-Edit workflow for refinement."""
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
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
        is_qwen21 = context.scene.model_architecture == 'qwen_image_21'

        # --- Build LoRA chain ---
        is_nunchaku = (
            not is_qwen21
            and context.scene.model_name.lower().endswith('.safetensors')
        )
        if not is_qwen21:
            initial_model_input = [NODES['unet_loader'], 0]
            dummy_clip_input = [NODES['unet_loader'], 0]
            lora_class = (
                "NunchakuQwenImageLoraLoader"
                if is_nunchaku else "LoraLoaderModelOnly"
            )
            prompt, final_lora_model_out, _ = self._build_lora_chain(
                prompt, context,
                initial_model_input, dummy_clip_input,
                start_node_id=500,
                lora_class_type=lora_class,
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
        use_prev_ref = context.scene.qwen_refine_use_prev_ref and masked_edit is None
        style_image_info = None
        
        if use_prev_ref and camera_id > 0:
             # Use previous generated image
             style_image_info = self.operator._get_uploaded_image_info(context, "generated", camera_id=camera_id - 1, material_id=self.operator._material_id)
        
        if style_image_info:
            prompt[NODES['style_map_loader']]['inputs']['image'] = style_image_info['name']
        else:
            # Fallback to external style image if configured
            if masked_edit is not None:
                style_image_info = masked_edit.get('overlay')
                if style_image_info:
                    prompt[NODES['style_map_loader']]['inputs']['image'] = style_image_info['name']
            elif context.scene.qwen_use_external_style_image:
                 style_image_info = self.operator._get_uploaded_image_info(context, "custom", filename=bpy.path.abspath(context.scene.qwen_external_style_image))
                 if style_image_info:
                     prompt[NODES['style_map_loader']]['inputs']['image'] = style_image_info['name']
            
            # Fallback to TRELLIS.2 input image as style
            if masked_edit is None and not style_image_info and getattr(context.scene, 'architecture_mode', '') == 'trellis2' and getattr(context.scene, 'qwen_use_trellis2_style', False):
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
        use_depth = (bool(masked_edit.get('depth')) if masked_edit is not None
                     else context.scene.qwen_refine_use_depth)
        depth_info = masked_edit.get('depth') if masked_edit is not None else None
        if use_depth and masked_edit is None:
            depth_info = self.operator._get_uploaded_image_info(context, "controlnet", subtype="depth", camera_id=camera_id)
        
        if depth_info:
            prompt[NODES['context_render_loader']]['inputs']['image'] = depth_info['name']
        else:
            # Remove Context Render (Image 3) if not used
            del prompt[NODES['context_render_loader']]
            del prompt[NODES['pos_prompt']]['inputs']['image3']
            del prompt[NODES['neg_prompt']]['inputs']['image3']

        # --- Prompt ---
        user_prompt = _texturing_prompt(context.scene)
        final_prompt = f"Modify image1 to {user_prompt}"
        if masked_edit is not None:
            from .masked_edit import qwen_edit_instruction
            final_prompt = qwen_edit_instruction(
                masked_edit['prompt'], bool(style_image_info), masked_edit.get('prompt_template'))
        if depth_info:
            final_prompt += "\n\nImage3 is a depth map of the same view; use it to preserve scene geometry."
        
        prompt[NODES['pos_prompt']]['inputs']['prompt'] = final_prompt

        if is_qwen21:
            self._convert_to_qwen_image_21(prompt, NODES, context)
        else:
            # --- Reference Latent chain (Qwen unzoom fix) ---
            # The overlay is a semantic selection guide, never an appearance latent.
            self._build_qwen_reference_latent_chain(
                prompt, NODES,
                **({'latent_slots': ('image1',)} if masked_edit is not None else {}),
            )
            prompt[NODES['sampler']]['inputs']['seed'] = context.scene.seed
            prompt[NODES['sampler']]['inputs']['steps'] = context.scene.steps
            prompt[NODES['sampler']]['inputs']['cfg'] = context.scene.cfg
            prompt[NODES['sampler']]['inputs']['sampler_name'] = context.scene.sampler
            prompt[NODES['sampler']]['inputs']['scheduler'] = context.scene.scheduler
            prompt[NODES['sampler']]['inputs']['denoise'] = 1.0

        if is_nunchaku:
            prompt[NODES['unet_loader']] = {
                "inputs": {
                    "model_name": context.scene.model_name,
                    "cpu_offload": "auto",
                    "num_blocks_on_gpu": 1,
                    "use_pin_memory": "enable",
                },
                "class_type": "NunchakuQwenImageDiTLoader",
                "_meta": {"title": "Nunchaku Qwen-Image DiT Loader"},
            }
        elif not is_qwen21:
            prompt[NODES['unet_loader']]['inputs']['unet_name'] = context.scene.model_name

        graph_issues = (validate_workflow_graph(
            prompt,
            ((NODES['pos_prompt'],) if is_qwen21
             else (NODES['pos_prompt'], NODES['neg_prompt'])),
        ))
        if graph_issues:
            self.operator._error = "Invalid Qwen refine workflow: " + "; ".join(graph_issues)
            return {"error": "invalid_workflow"}
        
        # --- Save and Execute ---
        self._save_prompt_to_file(
            prompt, revision_dir, workflow="qwen_refine", camera_id=camera_id
        )
        
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
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
        client_id = str(uuid.uuid4())
        revision_dir = get_generation_dirs(context)["revision"]

        prompt = json.loads(prompt_text_qwen_image_edit)

        NODES = {
            'sampler': "1",
            'vae_loader': "4",
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
        is_qwen21 = context.scene.model_architecture == 'qwen_image_21'

        # --- Build LoRA chain ---
        # The Qwen workflow uses model-only LoRAs. We can reuse the existing
        # chain builder by providing a dummy CLIP input that won't be used.
        is_nunchaku = (
            not is_qwen21
            and context.scene.model_name.lower().endswith('.safetensors')
        )
        if not is_qwen21:
            initial_model_input = [NODES['unet_loader'], 0]
            dummy_clip_input = [NODES['unet_loader'], 0]
            lora_class = (
                "NunchakuQwenImageLoraLoader"
                if is_nunchaku else "LoraLoaderModelOnly"
            )
            prompt, final_lora_model_out, _ = self._build_lora_chain(
                prompt, context,
                initial_model_input, dummy_clip_input,
                start_node_id=500,
                lora_class_type=lora_class,
            )
            # Connect the output of the LoRA chain to the next model node.
            prompt[NODES['model_sampler']]['inputs']['model'] = final_lora_model_out


        # --- Configure Inputs ---
        guidance_map_type = context.scene.qwen_guidance_map_type
        guidance_map_info = self.operator._get_uploaded_image_info(context, "controlnet", subtype=guidance_map_type, camera_id=camera_id)
        if not guidance_map_info:
            self.operator._error = f"Could not find or upload {guidance_map_type} map for camera {camera_id}."
            return {"error": "conn_failed"}
        prompt[NODES['guidance_map_loader']]['inputs']['image'] = guidance_map_info['name']

        # --- Configure Style Image (Image 2) and Prompts ---
        user_prompt = _texturing_prompt(context.scene)
        style_image_info = None
        context_render_info = None
        context_mode = context.scene.qwen_context_render_mode
        remove_context = False

        # --- Camera Prompt Injection ---
        if context.scene.use_camera_prompts:
            view_desc = _immutable_view_prompt(self.operator)
            if view_desc:
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
                pos_prompt_text = self._get_qwen_default_prompts(
                    context, is_initial_image
                ).format(main_prompt=user_prompt)
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
                pos_prompt_text = self._get_qwen_default_prompts(
                    context, is_initial_image
                ).format(main_prompt=user_prompt)
                prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text
            else: # Additional mode
                pos_prompt_text = self._get_qwen_default_prompts(
                    context, is_initial_image
                ).format(main_prompt=user_prompt)
                prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text

        # Case 1b: TRELLIS.2 input image as style (when no external style configured)
        trellis_style_branch = (
            getattr(context.scene, 'architecture_mode', '') == 'trellis2'
            and getattr(context.scene, 'qwen_use_trellis2_style', False)
        )
        if not use_external_this_frame and trellis_style_branch:
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
                    pos_prompt_text = self._get_qwen_default_prompts(
                        context, is_initial_image
                    ).format(main_prompt=user_prompt)
                    prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text

        # Case 2: Sequential generation (after first image). The rigging
        # branch used an ``elif`` after the TRELLIS branch, so an enabled
        # initial-only TRELLIS style did not fall through on later views.
        # Preserve that quirk unless the corrected routing experiment is on.
        use_sequential_style = style_image_info is None and not is_initial_image
        if use_sequential_style:
            if context_mode == 'REPLACE_STYLE':
                # The context render becomes the style image
                style_image_info = context_render_info
                pos_prompt_text = self._get_qwen_default_prompts(
                    context, is_initial_image
                ).format(main_prompt=user_prompt)
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
                    pos_prompt_text = self._get_qwen_default_prompts(
                        context, is_initial_image
                    ).format(main_prompt=user_prompt)
                    prompt[NODES['pos_prompt']]['inputs']['prompt'] = pos_prompt_text
            # If neither of the above, style_image_info remains None, handled below

        # Case 3: First image of a sequence, or separate generation, or no style source in sequential
        if style_image_info is None:
            # No style image is provided — remove the style loader node and
            # its reference from the prompt encoders.
            del prompt[NODES['style_map_loader']]
            if context_mode == 'ADDITIONAL' and not is_initial_image:
                # In ADDITIONAL mode the context/style loader IDs were swapped
                # above, so the style loader now sits on the node that image3
                # originally pointed to.  Remove image3 (dangling after the
                # delete) but keep image2 (the context render).
                for nid in (NODES['pos_prompt'], NODES['neg_prompt']):
                    prompt[nid]['inputs'].pop('image3', None)
            else:
                del prompt[NODES['pos_prompt']]['inputs']['image2']
                del prompt[NODES['neg_prompt']]['inputs']['image2']
            pos_prompt_text = self._get_qwen_default_prompts(
                context, is_initial_image
            ).format(main_prompt=user_prompt)
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
                style_slot = (
                    qwen_style_semantic_slot(context_mode, is_initial_image)
                )
                prompt[NODES['pos_prompt']]['inputs'][style_slot] = [scale_node_key, 0]
                prompt[NODES['neg_prompt']]['inputs'][style_slot] = [scale_node_key, 0]

        if is_qwen21:
            self._convert_to_qwen_image_21(
                prompt,
                NODES,
                context,
                projected_context_target=(
                    not is_initial_image
                    and context_mode in {'REPLACE_STYLE', 'ADDITIONAL'}
                    and context_render_info is not None
                ),
            )
        else:
            # --- Reference Latent chain (Qwen unzoom fix) ---
            latent_slots = qwen_reference_latent_slots(
                context_mode,
                is_initial_image,
                experimental_enabled(context.scene, 'qwen_global_style_vlm_only'),
                bool(
                    experimental_enabled(context.scene, 'qwen_context_vlm_only')
                    and context_render_info
                    and (context_mode == 'ADDITIONAL' or style_image_info == context_render_info)
                ),
            )
            if 'image2' not in latent_slots and 'image2' in prompt[NODES['pos_prompt']]['inputs']:
                print("[StableGen] Qwen projected context is VLM-only; skipping its ReferenceLatent")
            if 'image3' not in latent_slots and 'image3' in prompt[NODES['pos_prompt']]['inputs']:
                print(
                    "[StableGen] Qwen Image 3 global style is VLM-only; "
                    "skipping its ReferenceLatent"
                )
            self._build_qwen_reference_latent_chain(
                prompt,
                NODES,
                latent_slots=latent_slots,
            )

        # --- Sampler-level latent-locked context inpainting ---
        # Unlike the RGB/magenta conditioning alone, this makes the exported
        # visibility repair mask authoritative inside the sampler.  It is
        # available only for later sequential views that actually have a
        # context render; the initial view keeps the legacy depth-start latent.
        latent_lock_enabled = bool(experimental_enabled(context.scene, 'qwen_latent_locked_context'))
        latent_lock_active = (
            latent_lock_enabled
            and not is_initial_image
            and context_mode in {'REPLACE_STYLE', 'ADDITIONAL'}
            and context_render_info is not None
        )
        if latent_lock_active:
            repair_mask_info = self.operator._get_uploaded_image_info(
                context,
                "inpaint",
                subtype="visibility",
                camera_id=camera_id,
            )
            if not repair_mask_info:
                self.operator._error = (
                    "Qwen latent-locked context is enabled, but the sequential "
                    f"repair mask for camera {camera_id} is missing."
                )
                return {"error": "conn_failed"}
            attach_qwen_latent_lock(
                prompt,
                sampler_node_id=NODES['sampler'],
                vae_node_id=NODES['vae_loader'],
                context_image_name=context_render_info['name'],
                repair_mask_name=repair_mask_info['name'],
                neutral_fill=bool(experimental_enabled(context.scene, 'qwen_latent_lock_neutral_fill')),
            )
            print(
                "[StableGen] Qwen latent-locked context active for camera "
                f"{camera_id}: trusted pixels are sampler-locked; repair "
                "pixels denoise."
            )
        elif latent_lock_enabled and is_initial_image:
            print(
                "[StableGen] Qwen latent-locked context enabled; initial view "
                "has no prior context and uses the normal sampler latent."
            )
        elif latent_lock_enabled:
            print(
                "[StableGen] Qwen latent-locked context enabled but inactive: "
                "use a later sequential view with Replace Style or Additional "
                "context."
            )

        # --- Configure Sampler ---
        if not is_qwen21:
            prompt[NODES['sampler']]['inputs']['seed'] = context.scene.seed
            prompt[NODES['sampler']]['inputs']['steps'] = context.scene.steps
            prompt[NODES['sampler']]['inputs']['cfg'] = context.scene.cfg
            prompt[NODES['sampler']]['inputs']['sampler_name'] = context.scene.sampler
            prompt[NODES['sampler']]['inputs']['scheduler'] = context.scene.scheduler
            prompt[NODES['sampler']]['inputs']['denoise'] = 1.0

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
        elif not is_qwen21:
            prompt[NODES['unet_loader']]['inputs']['unet_name'] = context.scene.model_name

        graph_issues = (validate_workflow_graph(
            prompt,
            ((NODES['pos_prompt'],) if is_qwen21
             else (NODES['pos_prompt'], NODES['neg_prompt'])),
        ))
        if graph_issues:
            self.operator._error = "Invalid Qwen edit workflow: " + "; ".join(graph_issues)
            return {"error": "invalid_workflow"}

        # --- Execute ---
        self._save_prompt_to_file(
            prompt, revision_dir, workflow="qwen_edit", camera_id=camera_id
        )
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

        print(f"[StableGen] Qwen image generated with prompt: {prompt[NODES['pos_prompt']]['inputs']['prompt']}")
        return images[NODES['save_image']][0]

    # ------------------------------------------------------------------
    # FLUX.2 Klein – multi-reference edit generation
    # ------------------------------------------------------------------
    def generate_flux2_klein(self, context, camera_id=None):
        """Generates an image using the FLUX.2 Klein multi-reference edit workflow.

        Klein injects reference images via ReferenceLatent chains and
        understands image references ("image 1", "image 2") in prompts.
        Following BFL's official prompting guide, prompts are written as
        natural prose describing the desired transformation, front-loading
        the subject and referencing images by number.

        Reference layout:
            Ref 1 → structure / guidance map  (depth, normal, workbench …)
            Ref 2 → style reference           (external image or prev gen)
            Ref 3 → context render            (sequential mode only)
        """
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
        client_id = str(uuid.uuid4())
        revision_dir = get_generation_dirs(context)["revision"]

        prompt = json.loads(prompt_text_flux2_klein)

        NODES = {
            'unet_loader':  "1",
            'clip_loader':  "2",
            'vae_loader':   "3",
            'pos_prompt':   "4",   # CLIPTextEncode
            'neg_prompt':   "5",   # CLIPTextEncode (empty)
            'empty_latent': "6",   # EmptyFlux2LatentImage
            'scheduler':    "7",   # Flux2Scheduler
            'ksampler':     "8",   # KSamplerSelect
            'noise':        "9",   # RandomNoise
            'guider':       "10",  # CFGGuider
            'sampler':      "11",  # SamplerCustomAdvanced
            'vae_decode':   "12",  # VAEDecode
            'save_image':   "13",  # SaveImageWebsocket
            'ref1_loader':  "14",  # LoadImage – structure (guidance map)
            'ref2_loader':  "15",  # LoadImage – style
            'ref3_loader':  "16",  # LoadImage – context render
        }

        # --- Resolution (must be multiple of 16 for FLUX.2 latent) ---
        if context.scene.generation_method == 'grid':
            width = self.operator._grid_width
            height = self.operator._grid_height
        else:
            width = context.scene.render.resolution_x
            height = context.scene.render.resolution_y
        width = (width // 16) * 16
        height = (height // 16) * 16
        prompt[NODES['empty_latent']]['inputs']['width'] = width
        prompt[NODES['empty_latent']]['inputs']['height'] = height
        prompt[NODES['scheduler']]['inputs']['width'] = width
        prompt[NODES['scheduler']]['inputs']['height'] = height

        # --- Sampler settings ---
        prompt[NODES['noise']]['inputs']['noise_seed'] = context.scene.seed
        prompt[NODES['scheduler']]['inputs']['steps'] = context.scene.steps
        prompt[NODES['ksampler']]['inputs']['sampler_name'] = context.scene.sampler
        prompt[NODES['guider']]['inputs']['cfg'] = context.scene.cfg

        # --- Model ---
        prompt[NODES['unet_loader']]['inputs']['unet_name'] = context.scene.model_name

        # --- CLIP: Detect 9B vs 4B from model name ---
        # 9B uses a larger Qwen 3 8B text encoder instead of 4B.
        # Default to fp8mixed (8.66 GB) for practical VRAM usage.
        # Full-precision qwen_3_8b.safetensors (16.4 GB) is also compatible.
        # Download from: Comfy-Org/vae-text-encorder-for-flux-klein-9b
        model_lower = context.scene.model_name.lower()
        if '9b' in model_lower:
            prompt[NODES['clip_loader']]['inputs']['clip_name'] = "qwen_3_8b_fp8mixed.safetensors"

        # --- Base user prompt + camera injection ---
        user_prompt = _texturing_prompt(context.scene)
        camera_suffix = ""
        if context.scene.use_camera_prompts:
            view_desc = _immutable_view_prompt(self.operator)
            if view_desc:
                camera_suffix = f", {view_desc}"

        # ─── Collect reference images ────────────────────────────────
        ref_images = []
        remove_ref1 = True
        remove_ref2 = True
        remove_ref3 = True

        is_initial_image = not self.operator._current_image > 0
        context_mode = context.scene.qwen_context_render_mode
        context_render_info = None
        style_image_info = None

        # --- Ref 1: Structure / guidance map (always attempt) ---
        guidance_map_type = context.scene.qwen_guidance_map_type
        guidance_map_info = self.operator._get_uploaded_image_info(
            context, "controlnet", subtype=guidance_map_type, camera_id=camera_id)
        if guidance_map_info:
            prompt[NODES['ref1_loader']]['inputs']['image'] = guidance_map_info['name']
            ref_images.append({'loader_node': NODES['ref1_loader']})
            remove_ref1 = False

        # --- Ref 3: Context render (sequential subsequent only) ---
        # Handled BEFORE Ref 2 so ADDITIONAL mode can swap slot ids (mirrors Qwen).
        remove_context = True
        if not is_initial_image and context_mode != 'NONE':
            context_render_info = self.operator._get_uploaded_image_info(
                context, "inpaint", subtype="render", camera_id=camera_id)
            if not context_render_info:
                self.operator._error = f"Klein context render enabled, but could not find context render for camera {camera_id}."
                return {"error": "conn_failed"}
            if context_render_info and context_mode == 'ADDITIONAL':
                # Swap loaders so context render is Ref 2 and style is Ref 3
                NODES['ref3_loader'], NODES['ref2_loader'] = NODES['ref2_loader'], NODES['ref3_loader']
                prompt[NODES['ref3_loader']]['inputs']['image'] = context_render_info['name']
                ref_images.append({'loader_node': NODES['ref3_loader']})
                remove_context = False
                if context.scene.qwen_use_custom_prompts:
                    pos_prompt_text = context.scene.qwen_custom_prompt_seq_additional.format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                else:
                    pos_prompt_text = self._get_klein_default_prompts(context, is_initial_image).format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                prompt[NODES['pos_prompt']]['inputs']['text'] = pos_prompt_text

        if remove_context and not remove_ref3:
            pass  # already assigned
        elif remove_context:
            pass  # will be deleted below

        # --- Ref 2: Style reference (optional) ---
        # Determine if we should use the external style image for this frame
        use_external = context.scene.qwen_use_external_style_image
        if use_external and not is_initial_image and context.scene.qwen_external_style_initial_only:
            use_external = False
            context.scene.sequential_ipadapter = True

        # Case 1: External Style Image
        if use_external:
            style_image_info = self.operator._get_uploaded_image_info(
                context, "custom",
                filename=bpy.path.abspath(context.scene.qwen_external_style_image))
            if style_image_info and context_mode != 'ADDITIONAL':
                if context.scene.qwen_use_custom_prompts:
                    pos_prompt_text = (context.scene.qwen_custom_prompt_initial if is_initial_image
                                       else context.scene.qwen_custom_prompt_seq_none).format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                else:
                    pos_prompt_text = self._get_klein_default_prompts(context, is_initial_image).format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                prompt[NODES['pos_prompt']]['inputs']['text'] = pos_prompt_text

        # Case 1b: TRELLIS.2 input image as style
        if not style_image_info and getattr(context.scene, 'architecture_mode', '') == 'trellis2' and getattr(context.scene, 'qwen_use_trellis2_style', False):
            use_t2 = True
            if not is_initial_image and getattr(context.scene, 'qwen_trellis2_style_initial_only', False):
                use_t2 = False
                context.scene.sequential_ipadapter = True
            if use_t2:
                t2_path = getattr(context.scene, 'trellis2_last_input_image', '')
                if t2_path and os.path.exists(bpy.path.abspath(t2_path)):
                    style_image_info = self.operator._get_uploaded_image_info(
                        context, "custom", filename=bpy.path.abspath(t2_path))
                    if style_image_info:
                        if context.scene.qwen_use_custom_prompts:
                            pos_prompt_text = context.scene.qwen_custom_prompt_initial.format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                        else:
                            pos_prompt_text = self._get_klein_default_prompts(context, is_initial_image).format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                        prompt[NODES['pos_prompt']]['inputs']['text'] = pos_prompt_text

        # Case 2: Sequential (after first image)
        if not style_image_info and not is_initial_image:
            if context_mode == 'REPLACE_STYLE':
                style_image_info = context_render_info
                if context.scene.qwen_use_custom_prompts:
                    pos_prompt_text = context.scene.qwen_custom_prompt_seq_replace.format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                else:
                    pos_prompt_text = self._get_klein_default_prompts(context, is_initial_image).format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                prompt[NODES['pos_prompt']]['inputs']['text'] = pos_prompt_text
            elif context.scene.sequential_ipadapter:
                if context.scene.sequential_ipadapter_mode == 'original_render':
                    # For original_render in Klein context, fall back to first mode since
                    # the render is already used as the guidance map (Ref 1).
                    ref_cam = 0
                else:
                    ref_cam = (0 if context.scene.sequential_ipadapter_mode == 'first'
                               else self.operator._current_image - 1)
                style_image_info = self.operator._get_uploaded_image_info(
                    context, "generated", camera_id=ref_cam,
                    material_id=self.operator._material_id)
                if context_mode != 'ADDITIONAL':
                    if context.scene.qwen_use_custom_prompts:
                        pos_prompt_text = context.scene.qwen_custom_prompt_seq_none.format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                    else:
                        pos_prompt_text = self._get_klein_default_prompts(context, is_initial_image).format(main_prompt=user_prompt, camera_suffix=camera_suffix)
                    prompt[NODES['pos_prompt']]['inputs']['text'] = pos_prompt_text

        # Case 3: No style image – first image or no style source
        if style_image_info is None:
            if context.scene.qwen_use_custom_prompts:
                pos_prompt_text = (context.scene.qwen_custom_prompt_initial if is_initial_image
                                   else context.scene.qwen_custom_prompt_seq_none).format(main_prompt=user_prompt, camera_suffix=camera_suffix)
            else:
                pos_prompt_text = self._get_klein_default_prompts(context, is_initial_image).format(main_prompt=user_prompt, camera_suffix=camera_suffix)
            prompt[NODES['pos_prompt']]['inputs']['text'] = pos_prompt_text
        else:
            prompt[NODES['ref2_loader']]['inputs']['image'] = style_image_info['name']
            ref_images.append({'loader_node': NODES['ref2_loader']})
            remove_ref2 = False

        # Remove unused LoadImage nodes
        if remove_ref1:
            del prompt[NODES['ref1_loader']]
        if remove_ref2:
            del prompt[NODES['ref2_loader']]
        if remove_context:
            del prompt[NODES['ref3_loader']]

        # --- Build ReferenceLatent chain ---
        self._build_flux2_reference_chain(prompt, NODES, ref_images)

        # --- Execute ---
        self._save_prompt_to_file(
            prompt, revision_dir, workflow="flux2_klein", camera_id=camera_id
        )
        ws = self._connect_to_websocket(server_address, client_id)
        if ws is None:
            return {"error": "conn_failed"}

        images = None
        try:
            images = self._execute_prompt_and_get_images(
                ws, prompt, client_id, server_address, NODES)
        finally:
            if ws:
                ws.close()

        if images is None or (isinstance(images, dict) and "error" in images):
            return {"error": "conn_failed"}

        print(f"[StableGen] FLUX.2 Klein image generated with prompt: {prompt[NODES['pos_prompt']]['inputs']['text']}")
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
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
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
        self._save_prompt_to_file(prompt, revision_dir, workflow="sdxl_generate")

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
        
        print(f"[StableGen] Image generated with prompt: {context.scene.comfyui_prompt}")
        
        # Return the generated image from the save_image node
        return images[NODES['save_image']][0]

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
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
        client_id = str(uuid.uuid4())
        scene = context.scene
        architecture = scene.model_architecture  # synced from texture_mode

        # When using TRELLIS.2 with native/none texturing the backbone is
        # synced from trellis2_initial_image_arch instead of texture_mode.
        # Double-check here in case the property wasn't synced yet.
        if getattr(scene, 'architecture_mode', '') == 'trellis2':
            tex_mode = getattr(scene, 'trellis2_texture_mode', 'native')
            if tex_mode not in ('sdxl', 'flux1', 'qwen_image_edit', 'qwen_image_21', 'flux2_klein'):
                architecture = getattr(scene, 'trellis2_initial_image_arch', 'sdxl')

        if architecture == 'qwen_image_21':
            prompt, save_node = self._build_qwen_image_21_txt2img(context)
        elif architecture == 'qwen_image_edit':
            prompt, save_node = self._build_qwen_txt2img(context)
        elif architecture == 'flux1':
            prompt, save_node = self._build_flux_txt2img(context)
        elif architecture == 'flux2_klein':
            prompt, save_node = self._build_klein_txt2img(context)
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
                "inputs": {"text": _generation_prompt(scene), "clip": ["1", 1]}
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
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
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
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return {"error": f"ComfyUI-Marigold node '{node_class}' not found. "
                                     f"Install ComfyUI-Marigold (run installer option 9 "
                                     f"or git clone https://github.com/kijai/ComfyUI-Marigold "
                                     f"into ComfyUI/custom_nodes/) and ensure 'diffusers>=0.28' "
                                     f"is installed in ComfyUI's Python, then restart ComfyUI."}
                return {"error": f"PBR model error: failed to query ComfyUI for node "
                                 f"'{node_class}' (HTTP {e.code}). Is ComfyUI running?"}
            except Exception as e:
                return {"error": f"PBR model error: could not reach ComfyUI at "
                                 f"'{server_address}' while checking for node '{node_class}' "
                                 f"({type(e).__name__}: {e}). Verify ComfyUI is running and "
                                 f"the server address is correct."}

        # Upload the source image to ComfyUI's input folder
        from .._generator_utils import upload_image_to_comfyui
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
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
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
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return {"error": f"StableDelight node '{node_class}' not found. "
                                     f"Install ComfyUI_StableDelight_ll (run installer "
                                     f"option 10 or git clone "
                                     f"https://github.com/lldacing/ComfyUI_StableDelight_ll "
                                     f"into ComfyUI/custom_nodes/) and download the model "
                                     f"'Stable-X/yoso-delight-v0-4-base' into "
                                     f"ComfyUI/models/diffusers/, then restart ComfyUI."}
                return {"error": f"Normal estimation error: failed to query ComfyUI for node "
                                 f"'{node_class}' (HTTP {e.code}). Is ComfyUI running?"}
            except Exception as e:
                return {"error": f"Normal estimation error: could not reach ComfyUI at "
                                 f"'{server_address}' while checking for node '{node_class}' "
                                 f"({type(e).__name__}: {e}). Verify ComfyUI is running and "
                                 f"the server address is correct."}

        # Resolve model path — the node expects a relative path under
        # ComfyUI/models/diffusers/ containing model_index.json.
        delight_model_path = "Stable-X--yoso-delight-v0-4-base"

        # Upload the source image to ComfyUI's input folder
        from .._generator_utils import upload_image_to_comfyui
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
                    # Official Stable-X Predictor uses LANCZOS for resize.
                    # F.interpolate (used by MarigoldImageProcessor) doesn't
                    # support lanczos, but "bicubic" is the closest PyTorch
                    # equivalent and much sharper than "bilinear" for both
                    # input downscale and output upscale.
                    "upscale_method": "bicubic",
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

    def _build_klein_txt2img(self, context):
        """Return (prompt_dict, save_node_id) for a minimal FLUX.2 Klein txt2img.

        Uses the same node types as the full Klein multi-reference workflow
        (UNETLoader, CLIPLoader with type flux2, VAELoader, Flux2Scheduler,
        CFGGuider, SamplerCustomAdvanced) but without any reference images.
        """
        scene = context.scene
        unet_name = scene.model_name

        # Detect 9B vs 4B CLIP from model name
        model_lower = unet_name.lower()
        clip_name = "qwen_3_8b_fp8mixed.safetensors" if '9b' in model_lower else "qwen_3_4b.safetensors"

        prompt = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": unet_name, "weight_dtype": "default"}
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": clip_name, "type": "flux2"}
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "flux2-vae.safetensors"}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": _generation_prompt(scene), "clip": ["2", 0]}
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["2", 0]}
            },
            "6": {
                "class_type": "EmptyFlux2LatentImage",
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
            },
            "7": {
                "class_type": "Flux2Scheduler",
                "inputs": {"steps": scene.steps, "width": 1024, "height": 1024}
            },
            "8": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": scene.sampler}
            },
            "9": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": scene.seed if scene.seed != 0 else random.randint(0, 2**31)}
            },
            "10": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "cfg": scene.cfg
                }
            },
            "11": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["9", 0],
                    "guider": ["10", 0],
                    "sampler": ["8", 0],
                    "sigmas": ["7", 0],
                    "latent_image": ["6", 0]
                }
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["11", 0], "vae": ["3", 0]}
            },
            "13": {
                "class_type": "SaveImageWebsocket",
                "inputs": {"images": ["12", 0]}
            }
        }

        return prompt, "13"

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
                "inputs": {"text": _generation_prompt(scene), "clip": ["11", 0]}
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
                    "prompt": _generation_prompt(scene),
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

    def _build_qwen_image_21_txt2img(self, context):
        """Return a native Qwen-Image 2.1 text-to-image graph."""
        prompt = json.loads(prompt_text_qwen_image_edit)
        nodes = {
            'sampler': "1",
            'save_image': "5",
            'model_sampler': "6",
            'pos_prompt': "12",
            'neg_prompt': "11",
            'unet_loader': "13",
            'guidance_map_loader': "14",
            'style_map_loader': "15",
            'context_render_loader': "16",
        }

        prompt[nodes['pos_prompt']]['inputs']['prompt'] = _generation_prompt(context.scene)
        for encoder_id in (nodes['pos_prompt'], nodes['neg_prompt']):
            for slot in ('image1', 'image2', 'image3'):
                prompt[encoder_id]['inputs'].pop(slot, None)
        for loader_id in (
            nodes['guidance_map_loader'],
            nodes['style_map_loader'],
            nodes['context_render_loader'],
        ):
            prompt.pop(loader_id, None)

        self._convert_to_qwen_image_21(prompt, nodes, context)
        return prompt, nodes['save_image']

    def _create_base_prompt(self, context):
        """Creates and configures the base prompt with user settings."""
        from ..util.workflow_templates import prompt_text
        
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
        
        base_prompt_text = _texturing_prompt(context.scene)
        # Camera Prompt Injection
        if context.scene.use_camera_prompts and context.scene.generation_method in ['separate', 'sequential', 'refine', 'local_edit']:
            view_desc = _immutable_view_prompt(self.operator)
            if view_desc:
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
        addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
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
                print(f"[StableGen] Warning: Uploaded info for ControlNet type '{unit.unit_type}' not found. Skipping unit.")
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

    def generate_masked_sdxl(self, context, render_info, mask_info, instruction):
        """Masked image-space inpainting, independent of the selected base preset mode."""
        prompt, nodes = self._create_img2img_base_prompt(context)
        for key in ('235', '236', '237', '239'):
            prompt.pop(key, None)
        sampler = prompt[nodes['sampler']]['inputs']
        # The base builder may insert Differential Diffusion for the scene mode.
        dd = prompt.pop(nodes['differential_diffusion'], None)
        if sampler['model'] == [nodes['differential_diffusion'], 0]:
            sampler['model'] = dd['inputs']['model']
        prompt[nodes['pos_prompt']]['inputs']['text'] = instruction
        prompt[nodes['render_image']]['inputs']['image'] = render_info['name']
        prompt[nodes['input_image']]['inputs']['image'] = render_info['name']
        prompt[nodes['mask_image']]['inputs']['image'] = mask_info['name']
        # Mask expansion/feathering is prepared once by Masked Edit, not by preset settings.
        prompt[nodes['grow_mask']]['inputs']['expand'] = 0
        mask = [nodes['image_to_mask'], 0]
        encoder = prompt[nodes['vae_encode_inpaint']]['inputs']
        encoder['pixels'] = [nodes['input_image'], 0]
        encoder['mask'] = mask
        encoder['grow_mask_by'] = 0
        sampler['latent_image'] = [nodes['vae_encode_inpaint'], 0]
        sampler['positive'] = [nodes['pos_prompt'], 0]
        sampler['negative'] = [nodes['neg_prompt'], 0]
        sampler['denoise'] = context.scene.sg_masked_strength
        self._save_prompt_to_file(prompt, get_generation_dirs(context)['revision'], workflow='masked_sdxl')
        server = context.preferences.addons[_ADDON_PKG].preferences.server_address
        client_id = str(uuid.uuid4())
        ws = self._connect_to_websocket(server, client_id)
        if ws is None:
            raise RuntimeError('Could not connect to ComfyUI')
        try:
            images = self._execute_prompt_and_get_images(ws, prompt, client_id, server, nodes)
        finally:
            ws.close()
        if not images or 'error' in images:
            raise RuntimeError(self.operator._error or 'Masked inpainting failed')
        return images[nodes['save_image']][0]

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
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
        client_id = str(uuid.uuid4())
        output_dir = context.preferences.addons[_ADDON_PKG].preferences.output_dir

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
            # UV textures do not use ControlNet; the template's union selector
            # references a loader that only the ControlNet builder creates.
            prompt.pop('239', None)
            if context.scene.differential_diffusion:
                prompt[NODES['sampler']]["inputs"]["positive"] = [NODES['inpaint_conditioning'], 0]
                prompt[NODES['sampler']]["inputs"]["negative"] = [NODES['inpaint_conditioning'], 1]
            else:
                prompt[NODES['sampler']]["inputs"]["positive"] = [NODES['pos_prompt'], 0]
                prompt[NODES['sampler']]["inputs"]["negative"] = [NODES['neg_prompt'], 0]
        
        # Save prompt for debugging
        self._save_prompt_to_file(prompt, output_dir, workflow="sdxl_refine")
        
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
        
        print(f"[StableGen] Image refined with prompt: ...")

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
        from ..util.workflow_templates import prompt_text_img2img
        
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
        
        base_prompt_text = _texturing_prompt(context.scene)
        # Camera Prompt Injection
        if context.scene.use_camera_prompts and context.scene.generation_method in ['separate', 'sequential', 'refine', 'local_edit']:
            view_desc = _immutable_view_prompt(self.operator)
            if view_desc:
                # Prepend the view description
                base_prompt_text = f"{view_desc}, {base_prompt_text}"
        
        # Set positive prompt based on generation method
        if context.scene.generation_method in ['refine', 'local_edit', 'uv_inpaint', 'sequential']:
            prompt[NODES['pos_prompt']]["inputs"]["text"] = base_prompt_text
        else:
            prompt[NODES['pos_prompt']]["inputs"]["text"] = context.scene.refine_prompt if context.scene.refine_prompt != "" else _texturing_prompt(context.scene)
        
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

        # Prepare IPAdapter's upstream model, but route through it only when
        # refine() has an actual reference image. Otherwise its nodes are removed.
        prompt[NODES['ipadapter_loader']]["inputs"]["model"] = current_model_out

        if context.scene.differential_diffusion and NODES['differential_diffusion'] in prompt and context.scene.generation_method not in ('refine', 'local_edit'):
            # Set model input for differential diffusion
            prompt[NODES['differential_diffusion']]["inputs"]["model"] = current_model_out
            current_model_out = [NODES['differential_diffusion'], 0]
        else:
            # Local/global img2img refinement does not use Differential Diffusion.
            # Its template input points at IPAdapter, which may be removed later.
            prompt.pop(NODES['differential_diffusion'], None)

        # Set the model for sampler node
        prompt[NODES['sampler']]["inputs"]["model"] = current_model_out

        return prompt, NODES

    def _configure_refinement_mode(self, prompt, context, render_info, mask_info, NODES):
        """Configures the prompt based on the specific refinement mode."""
        if not render_info or not render_info.get('name'):
            raise ValueError('Refinement input image is missing or could not be uploaded to ComfyUI')
        if context.scene.generation_method in ('uv_inpaint', 'sequential') and (
                not mask_info or not mask_info.get('name')):
            raise ValueError('Inpainting visibility mask is missing or could not be uploaded to ComfyUI')
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
            
            final_mask = sdxl_final_mask_reference(context.scene.blur_mask, NODES)
            prompt[NODES['inpaint_conditioning']]["inputs"]["mask"] = final_mask
            prompt[NODES['vae_encode_inpaint']]["inputs"]["mask"] = final_mask
            
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
        uv_prompt = f"seamless (UV-unwrapped texture) of {_texturing_prompt(context.scene)}, consistent material continuity, no visible seams or stretching, PBR material properties"
        uv_prompt_neg = f"seam, stitch, visible edge, texture stretching, repeating pattern, {context.scene.comfyui_negative_prompt}"
        
        prompt[NODES['pos_prompt']]["inputs"]["text"] = uv_prompt
        prompt[NODES['neg_prompt']]["inputs"]["text"] = uv_prompt_neg
        
        # Get the current object name from the file path
        current_object_name = None
        if render_info and 'name' in render_info:
            current_object_name = os.path.basename(render_info['name']).split('.')[0]
        
        # Use the object-specific prompt if available
        object_prompt = self.operator._object_prompts.get(current_object_name, _texturing_prompt(context.scene)) if current_object_name else _texturing_prompt(context.scene)
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
            controlnet_info or {}
        )
        
        # Connect final outputs to KSampler
        prompt[NODES['sampler']]["inputs"]["positive"] = [final, 0]
        prompt[NODES['sampler']]["inputs"]["negative"] = (
            [neg_input, 1 if neg_input == NODES['inpaint_conditioning'] else 0]
            if final == pos_input else [final, 1]
        )
        
        return prompt

    def create_base_prompt_flux(self, context):
        """Creates and configures the base Flux prompt.
        Uses prompt_text_flux and does not include negative prompt or LoRA configuration.
        """
        from ..util.workflow_templates import prompt_text_flux
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
        
        base_prompt_text = _texturing_prompt(context.scene)
        # Camera Prompt Injection
        if context.scene.use_camera_prompts and context.scene.generation_method in ['separate', 'sequential', 'refine', 'local_edit']:
            view_desc = _immutable_view_prompt(self.operator)
            if view_desc:
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
            from ..util.workflow_templates import gguf_unet_loader
            unet_loader_dict = json.loads(gguf_unet_loader)
            prompt.update(unet_loader_dict)

        # Set the model name
        prompt[NODES['unet_loader']]["inputs"]["unet_name"] = context.scene.model_name

        # Flux does not use negative prompt or LoRA.
        return prompt, NODES

    def configure_ipadapter_flux(self, prompt, context, ipadapter_ref_info, NODES):
        # Configure IPAdapter if enabled
        from ..util.workflow_templates import ipadapter_flux
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
        from ..util.workflow_templates import prompt_text_flux
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
        client_id = str(uuid.uuid4())
        output_dir = context.preferences.addons[_ADDON_PKG].preferences.output_dir

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
            from ..util.workflow_templates import depth_lora_flux
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
        self._save_prompt_to_file(prompt, revision_dir, workflow="flux_generate")

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
        
        print(f"[StableGen] Flux image generated with prompt: {context.scene.comfyui_prompt}")

        return images[NODES['save_image']][0]

    def _create_img2img_base_prompt_flux(self, context):
        """Creates and configures the base Flux prompt for img2img refinement."""
        from ..util.workflow_templates import prompt_text_img2img_flux
        
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
        
        base_prompt_text = _texturing_prompt(context.scene)
        # Camera Prompt Injection
        if context.scene.use_camera_prompts and context.scene.generation_method in ['separate', 'sequential', 'refine', 'local_edit', 'grid']:
            view_desc = _immutable_view_prompt(self.operator)
            if view_desc:
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
            from ..util.workflow_templates import gguf_unet_loader
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
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
        client_id = str(uuid.uuid4())
        output_dir = context.preferences.addons[_ADDON_PKG].preferences.output_dir

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
                from ..util.workflow_templates import depth_lora_flux
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
        self._save_prompt_to_file(prompt, revision_dir, workflow="flux_refine")
        
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
        
        print(f"[StableGen] Image refined with Flux using prompt: {context.scene.comfyui_prompt}")
        
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
        uv_prompt = f"seamless (UV-unwrapped texture) of {_texturing_prompt(context.scene)}, consistent material continuity, no visible seams or stretching"
        prompt[NODES['pos_prompt']]["inputs"]["text"] = uv_prompt
        
        # Object-specific prompt if available
        if render_info and 'name' in render_info:
            current_object_name = os.path.basename(render_info['name']).split('.')[0]
            object_prompt = self.operator._object_prompts.get(current_object_name, _texturing_prompt(context.scene))
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
