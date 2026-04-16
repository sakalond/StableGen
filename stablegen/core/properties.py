"""Scene and WindowManager property definitions for the StableGen addon.

All ``bpy.types.Scene.*`` and ``bpy.types.WindowManager.*`` properties that
were previously defined inside the monolithic ``register()`` function are
collected here in ``register_properties()`` and ``unregister_properties()``.
"""

import bpy  # pylint: disable=import-error

from ..ui.presets import update_parameters, get_preset_items
from ..cameras.prompts import CameraPromptItem, CameraOrderItem
from .callbacks import (
    update_architecture_mode,
    update_combined,
    update_trellis2_generate_from,
    update_trellis2_initial_image_arch,
    update_trellis2_texture_mode,
)
from . import ADDON_PKG


# ── Helper(s) used by properties ───────────────────────────────────────────

def _get_ipadapter_mode_items(self, context):
    items = [
        ('first', 'Use first generated image', '', 0),
        ('recent', 'Use most recent generated image', '', 1),
    ]
    if context:
        is_local_edit = (
            context.scene.generation_method == 'local_edit'
            or (context.scene.model_architecture.startswith('qwen')
                and context.scene.qwen_generation_method == 'local_edit')
        )
        if is_local_edit:
            items.append(('original_render', 'Use original render',
                          'Uses the existing texture render from each camera viewpoint as IPAdapter reference', 2))
        if getattr(context.scene, 'architecture_mode', '') == 'trellis2':
            items.append(('trellis2_input', 'Use TRELLIS.2 input image',
                          'Uses the input image from TRELLIS.2 mesh generation as IPAdapter reference', 3))
    return items


# ── Registration ───────────────────────────────────────────────────────────

def register_properties(update_model_list, ControlNetUnit, LoRAUnit,
                        SceneQueueItem, load_handler, _sg_queue_load_handler,
                        _sg_queue_load):
    """Register all Scene / WindowManager properties and load_post handlers.

    Parameters are passed in to avoid circular imports — the caller
    (``__init__.register``) already has them resolved.
    """

    # --- Scene Queue properties (on WindowManager so they're global) ---
    bpy.types.WindowManager.sg_scene_queue = bpy.props.CollectionProperty(
        type=SceneQueueItem, name="Scene Queue"
    )
    bpy.types.WindowManager.sg_scene_queue_index = bpy.props.IntProperty(
        name="Queue Index", default=0
    )
    bpy.types.WindowManager.sg_show_queue = bpy.props.BoolProperty(
        name="Show Queue", default=False
    )

    # --- Queue GIF Export settings ---
    bpy.types.WindowManager.sg_queue_gif_export = bpy.props.BoolProperty(
        name="Export GIF After Each Item", default=False,
        description="Automatically export an orbit GIF/MP4 after each queue item completes"
    )
    bpy.types.WindowManager.sg_queue_gif_duration = bpy.props.FloatProperty(
        name="Duration (s)", default=5.0, min=0.1, max=60.0,
        description="Duration of the 360-degree orbit"
    )
    bpy.types.WindowManager.sg_queue_gif_fps = bpy.props.IntProperty(
        name="FPS", default=24, min=1, max=60,
        description="Frames per second"
    )
    bpy.types.WindowManager.sg_queue_gif_resolution = bpy.props.IntProperty(
        name="Resolution %", default=50, min=10, max=100, subtype='PERCENTAGE',
        description="Percentage of scene render resolution"
    )
    bpy.types.WindowManager.sg_queue_gif_samples = bpy.props.IntProperty(
        name="Samples", default=32, min=1, max=4096,
        description="Render samples per frame"
    )
    bpy.types.WindowManager.sg_queue_gif_engine = bpy.props.EnumProperty(
        name="Engine",
        items=[
            ('BLENDER_WORKBENCH', "Workbench", "Fast preview renderer — no lighting, flat shading"),
            ('EEVEE', "Eevee", "Real-time renderer — good quality with fast render times"),
            ('CYCLES', "Cycles", "Path-traced renderer — highest quality but slowest"),
        ],
        default='CYCLES'
    )
    bpy.types.WindowManager.sg_queue_gif_interpolation = bpy.props.EnumProperty(
        name="Rotation Curve",
        items=[
            ('LINEAR', "Linear", "Constant rotation speed throughout the orbit"),
            ('BEZIER', "Ease In/Out", "Smooth acceleration and deceleration for a polished look"),
        ],
        default='LINEAR'
    )
    bpy.types.WindowManager.sg_queue_gif_use_hdri = bpy.props.BoolProperty(
        name="HDRI Environment", default=False
    )
    bpy.types.WindowManager.sg_queue_gif_hdri_path = bpy.props.StringProperty(
        name="HDRI File",
        description="Path to the HDRI image file (.hdr / .exr). "
                    "Applied via AddHDRI before each GIF export",
        default="", subtype='FILE_PATH',
    )
    bpy.types.WindowManager.sg_queue_gif_hdri_strength = bpy.props.FloatProperty(
        name="HDRI Strength",
        description="Brightness of the HDRI environment lighting",
        default=1.0, min=0.01, max=10.0,
    )
    bpy.types.WindowManager.sg_queue_gif_hdri_rotation = bpy.props.FloatProperty(
        name="HDRI Rotation", default=0.0, min=0.0, max=360.0, subtype='ANGLE'
    )
    bpy.types.WindowManager.sg_queue_gif_env_mode = bpy.props.EnumProperty(
        name="Environment Mode",
        items=[
            ('FIXED', "Fixed", "HDRI stays in place while the camera orbits"),
            ('LOCKED', "Locked", "HDRI rotates with the camera to keep lighting consistent"),
            ('COUNTER', "Counter-Rotate", "HDRI rotates opposite to the camera for dynamic lighting"),
            ('ENV_ONLY', "Environment Only", "Render only the HDRI environment without the object"),
        ],
        default='FIXED'
    )
    bpy.types.WindowManager.sg_queue_gif_denoiser = bpy.props.BoolProperty(
        name="Denoise", default=True
    )
    bpy.types.WindowManager.sg_queue_gif_use_gpu = bpy.props.BoolProperty(
        name="GPU Compute", default=True,
        description="Use GPU for Cycles rendering (Blender 5.1+ only)"
    )
    bpy.types.WindowManager.sg_queue_gif_also_no_pbr = bpy.props.BoolProperty(
        name="Also Export Without PBR", default=False,
        description="After the main GIF export, disable PBR, reproject, and export a second GIF with emission-only shading (1 sample)"
    )

    # --- Initial server refresh (timer) ---
    def initial_refresh():
        print("[StableGen] StableGen: Performing initial model list refresh...")
        try:
            bpy.ops.stablegen.check_server_status('INVOKE_DEFAULT')
            prefs = bpy.context.preferences.addons.get(ADDON_PKG)
            if not prefs or not prefs.preferences.server_online:
                print("[StableGen] StableGen: Server not reachable during initial refresh.")
                return None
            if prefs and prefs.preferences.server_address:
                bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')
                bpy.ops.stablegen.refresh_lora_list('INVOKE_DEFAULT')
                bpy.ops.stablegen.refresh_controlnet_mappings('INVOKE_DEFAULT')
            else:
                print("[StableGen] StableGen: Server address not set, skipping initial refresh.")
            load_handler(None)
            _sg_queue_load()
        except Exception as e:
            print(f"[StableGen] StableGen: Error during initial refresh: {e}")
        return None

    bpy.app.timers.register(initial_refresh, first_interval=1.0)

    # ── Core generation properties ─────────────────────────────────────
    bpy.types.Scene.comfyui_prompt = bpy.props.StringProperty(
        name="ComfyUI Prompt",
        description="Text prompt for generation (also used for texturing unless a separate texture prompt is provided)",
        default="gold cube",
        update=update_parameters
    )
    bpy.types.Scene.use_separate_texture_prompt = bpy.props.BoolProperty(
        name="Use Separate Texture Prompt",
        description="Enable a dedicated prompt field for the texturing pass instead of reusing the main prompt",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.texture_prompt = bpy.props.StringProperty(
        name="Texture Prompt",
        description="Prompt used for the per-camera texturing step. "
                    "Avoid view-specific details here (e.g. 'logo on the front') — "
                    "this prompt is applied from every camera angle. "
                    "Describe side-specific features in the main prompt or in per-camera prompts instead. "
                    "When empty, the main prompt is used",
        default="",
        update=update_parameters
    )
    bpy.types.Scene.comfyui_negative_prompt = bpy.props.StringProperty(
        name="ComfyUI Negative Prompt",
        description="Enter the negative text prompt for ComfyUI generation",
        default="",
        update=update_parameters
    )
    bpy.types.Scene.model_name = bpy.props.EnumProperty(
        name="Model Name",
        description="Select the SDXL checkpoint",
        items=update_model_list,
        update=update_parameters
    )
    bpy.types.Scene.sg_model_name_backup = bpy.props.StringProperty(
        name="Model Name Backup",
        description="Internal: stores the last known-good model_name identifier",
        default="",
        options={'HIDDEN'},
    )
    bpy.types.Scene.seed = bpy.props.IntProperty(
        name="Seed",
        description="Seed for image generation",
        default=42, min=0, max=1000000,
        update=update_parameters
    )
    bpy.types.Scene.control_after_generate = bpy.props.EnumProperty(
        name="Control After Generate",
        description="Control behavior after generation",
        items=[
            ('fixed', 'Fixed', 'Keep the same seed every generation'),
            ('increment', 'Increment', 'Add 1 to the seed after each generation'),
            ('decrement', 'Decrement', 'Subtract 1 from the seed after each generation'),
            ('randomize', 'Randomize', 'Pick a random seed for every generation')
        ],
        default='fixed',
        update=update_parameters
    )
    bpy.types.Scene.steps = bpy.props.IntProperty(
        name="Steps",
        description="Number of denoising steps. Higher values improve detail and coherence but take longer",
        default=8, min=0, max=200,
        update=update_parameters
    )
    bpy.types.Scene.cfg = bpy.props.FloatProperty(
        name="CFG",
        description="Classifier-Free Guidance scale. Higher values follow the prompt more strictly but may reduce quality; lower values give more creative freedom",
        default=1.5, min=0.0, max=100.0,
        update=update_parameters
    )
    bpy.types.Scene.sampler = bpy.props.EnumProperty(
        name="Sampler",
        description="Sampler for generation",
        items=[
            ('euler', 'Euler', ''),
            ('euler_ancestral', 'Euler A', ''),
            ('dpmpp_sde', 'DPM++ SDE', ''),
            ('dpmpp_2m', 'DPM++ 2M', ''),
            ('dpmpp_2s_ancestral', 'DPM++ 2S Ancestral', ''),
        ],
        default='dpmpp_2s_ancestral',
        update=update_parameters
    )
    bpy.types.Scene.scheduler = bpy.props.EnumProperty(
        name="Scheduler",
        description="Scheduler for generation",
        items=[
            ('sgm_uniform', 'SGM Uniform', ''),
            ('karras', 'Karras', ''),
            ('beta', 'Beta', ''),
            ('normal', 'Normal', ''),
            ('simple', 'Simple', ''),
        ],
        default='sgm_uniform',
        update=update_parameters
    )
    bpy.types.Scene.show_advanced_params = bpy.props.BoolProperty(
        name="Show Advanced Parameters",
        description="Show or hide advanced parameters",
        default=False, update=update_parameters
    )
    bpy.types.Scene.show_generation_params = bpy.props.BoolProperty(
        name="Show Generation Parameters",
        description="Toggle visibility of core generation settings (steps, CFG, sampler, scheduler, seed)",
        default=True, update=update_parameters
    )
    bpy.types.Scene.auto_rescale = bpy.props.BoolProperty(
        name="Auto Rescale Resolution",
        description="Automatically rescale resolution to appropriate size for the selected model",
        default=True, update=update_parameters
    )
    bpy.types.Scene.qwen_rescale_alignment = bpy.props.BoolProperty(
        name="Qwen VL-Aligned Rescale",
        description="Round resolution to multiples of 112 instead of 8 when auto-rescaling. "
                    "112 is the window size used by the Qwen2.5-VL vision encoder "
                    "(LCM of VAE factor 8, ViT patch 14, spatial merge 28). "
                    "The official diffusers pipeline rounds to 32; using 112 is stricter "
                    "and may reduce subtle zoom / pixel-shift artifacts in some cases",
        default=True, update=update_parameters
    )
    bpy.types.Scene.auto_rescale_target_mp = bpy.props.FloatProperty(
        name="Target Megapixels",
        description=(
            "Target total megapixels when auto-rescaling the render resolution.\n\n"
            "Default: 1.0 MP (~1024\u00d71024). This is the native resolution for SDXL "
            "and the recommended default for Qwen Image Edit.\n\n"
            "For SDXL / Flux models 1.0 MP is a safe default that matches "
            "the training resolution. Higher values are possible if your "
            "model and hardware support it, but may reduce quality or cause "
            "out-of-memory issues.\n\n"
            "For Qwen Image Edit the vision encoder becomes unreliable above "
            "~1.4 MP and the model often fails to follow the prompt above "
            "~1.2 MP. Values below ~0.3 MP also cause failures.\n\n"
            "Adjust with care \u2013 staying close to 1.0 MP works best for most setups"
        ),
        default=1.0, min=0.1, max=4.0, step=10, precision=2,
        update=update_parameters
    )

    # ── IPAdapter ──────────────────────────────────────────────────────
    bpy.types.Scene.use_ipadapter = bpy.props.BoolProperty(
        name="Use IPAdapter",
        description="Use IPAdapter for image generation. Requires an external reference image. Can improve consistency, can be useful for generating images with similar styles.\n\n - Has priority over mode specific IPAdapter.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.ipadapter_image = bpy.props.StringProperty(
        name="Reference Image",
        description="Path to the reference image",
        default="", subtype='FILE_PATH',
        update=update_parameters
    )
    bpy.types.Scene.ipadapter_strength = bpy.props.FloatProperty(
        name="IPAdapter Strength",
        description="Strength for IPAdapter",
        default=1.0, min=-1.0, max=3.0,
        update=update_parameters
    )
    bpy.types.Scene.ipadapter_start = bpy.props.FloatProperty(
        name="IPAdapter Start",
        description="Start percentage for IPAdapter (/100)",
        default=0.0, min=0.0, max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.ipadapter_end = bpy.props.FloatProperty(
        name="IPAdapter End",
        description="End percentage for IPAdapter (/100)",
        default=1.0, min=0.0, max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.ipadapter_weight_type = bpy.props.EnumProperty(
        name="IPAdapter Weight Type",
        description="Weight type for IPAdapter",
        items=[
            ('standard', 'Standard', ''),
            ('prompt', 'Prompt is more important', ''),
            ('style', 'Style transfer', ''),
        ],
        default='style',
        update=update_parameters
    )
    bpy.types.Scene.sequential_ipadapter = bpy.props.BoolProperty(
        name="Use IPAdapter",
        description="Uses IPAdapter to improve consistency between images.\n\n - Applicable for Separate, Sequential and Refine modes.\n - Uses either the first generated image or the most recent one as a reference for the rest of the images.\n - If 'Regenerate IPAdapter' is enabled, the first viewpoint will be regenerated with IPAdapter to match the rest of the images.\n - If 'Use IPAdapter (External Image)' is enabled, this setting is effectively overriden.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.sequential_ipadapter_mode = bpy.props.EnumProperty(
        name="IPAdapter Mode",
        description="Mode for IPAdapter in sequential generation",
        items=_get_ipadapter_mode_items,
        update=update_parameters
    )
    bpy.types.Scene.sequential_desaturate_factor = bpy.props.FloatProperty(
        name="Desaturate Recent Image",
        description="Desaturation factor for the 'most recent' image to prevent color stacking. 0.0 is no change, 1.0 is fully desaturated",
        default=0.0, min=0.0, max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.sequential_contrast_factor = bpy.props.FloatProperty(
        name="Reduce Contrast of Recent Image",
        description="Contrast reduction factor for the 'most recent' image to prevent contrast stacking. 0.0 is no change, 1.0 is maximum reduction (grey)",
        default=0.0, min=0.0, max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.sequential_ipadapter_regenerate = bpy.props.BoolProperty(
        name="Regenerate IPAdapter",
        description="IPAdapter generations may differ from the original image. This option regenerates the first viewpoint with IPAdapter to match the rest of the images.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.sequential_ipadapter_regenerate_wo_controlnet = bpy.props.BoolProperty(
        name="Generate IPAdapter reference without ControlNet",
        description="Generate the first viewpoint with IPAdapter without ControlNet. This is useful for generating a reference image that is not affected by ControlNet. Can possibly generate higher quality reference.",
        default=False, update=update_parameters
    )

    # ── Generation method ──────────────────────────────────────────────
    bpy.types.Scene.generation_method = bpy.props.EnumProperty(
        name="Generation Mode",
        description="Choose the mode for generating images",
        items=[
            ('separate', 'Generate Separately', 'Generates images one by one for each viewpoint. Each image is generated independently using only its own control signals (e.g., depth map) without context from other views. All images are applied at the end.'),
            ('sequential', 'Generate Sequentially', 'Generates images viewpoint by viewpoint. After the first view, each subsequent view is generated using inpainting, guided by a visibility mask and an RGB render of the texture projected from previous viewpoints to maintain consistency.'),
            ('grid', 'Generate Using Grid', 'Combines control signals from all viewpoints into a single grid, generates a single image, then splits it back into individual viewpoint textures. Faster but lower resolution per view. Includes an optional second pass to refine each split image individually at full resolution for improved quality.'),
            ('refine', 'Refine/Restyle Texture (Img2Img)', 'Uses the current rendered texture appearance as input for an img2img generation pass. Replaces the previous material with the new result. Good for restyling or globally changing the look of an existing texture. Works on any existing material setup.'),
            ('local_edit', 'Local Edit', 'Make localized changes to an existing texture. Point cameras at areas you want to modify — the new generation blends over the original using angle and vignette-based feathering, preserving untouched areas. Works only with StableGen generated textures.'),
            ('uv_inpaint', 'UV Inpaint Missing Areas', 'Identifies untextured areas on a standard UV map using a visibility calculation. Performs baking if not baked already. Performs diffusion inpainting directly on the UV texture map to fill only these missing regions, using the surrounding texture as context.'),
        ],
        default='sequential',
        update=update_parameters
    )
    bpy.types.Scene.qwen_generation_method = bpy.props.EnumProperty(
        name="Generation Mode",
        description="Choose the mode for generating images with Qwen",
        items=[
            ('generate', 'Generate', 'Standard generation mode'),
            ('refine', 'Refine', 'Refine/restyle the entire texture using Qwen Image Edit. Replaces the existing material with the new result. Describe the desired look in the prompt — you can completely change the style, color scheme, or overall appearance.'),
            ('local_edit', 'Local Edit', 'Make targeted changes to specific areas of the texture. Point cameras at what you want to change and describe the edit — you can change colors, add details, sharpen, alter text, or restyle selected parts. Untouched areas are preserved.'),
        ],
        default='generate',
        update=update_parameters
    )
    bpy.types.Scene.qwen_refine_use_prev_ref = bpy.props.BoolProperty(
        name="Use Previous Refined View",
        description="Use the previous modified/refined view output as a second image for reference (Image 2)",
        default=False, update=update_parameters
    )
    bpy.types.Scene.qwen_refine_use_depth = bpy.props.BoolProperty(
        name="Use Depth Map",
        description="Include depth map as an additional reference image (Image 3)",
        default=False, update=update_parameters
    )
    bpy.types.Scene.qwen_timestep_zero_ref = bpy.props.BoolProperty(
        name="Timestep-Zero References",
        description="Tell the model that reference images are fully clean (timestep 0) instead of noisy. "
                    "Reduces color shift and over-saturation, especially with Qwen 2511. "
                    "Requires the FluxKontextMultiReferenceLatentMethod node in ComfyUI",
        default=False, update=update_parameters
    )

    # ── Refine settings ────────────────────────────────────────────────
    bpy.types.Scene.refine_images = bpy.props.BoolProperty(
        name="Refine Images",
        description="Refine images after generation",
        default=False, update=update_parameters
    )
    bpy.types.Scene.refine_steps = bpy.props.IntProperty(
        name="Refine Steps", description="Number of steps for refining",
        default=8, min=0, max=200, update=update_parameters
    )
    bpy.types.Scene.refine_sampler = bpy.props.EnumProperty(
        name="Refine Sampler", description="Sampler for refining",
        items=[
            ('euler', 'Euler', ''),
            ('euler_ancestral', 'Euler A', ''),
            ('dpmpp_sde', 'DPM++ SDE', ''),
            ('dpmpp_2m', 'DPM++ 2M', ''),
            ('dpmpp_2s_ancestral', 'DPM++ 2S Ancestral', ''),
        ],
        default='dpmpp_2s_ancestral', update=update_parameters
    )
    bpy.types.Scene.refine_scheduler = bpy.props.EnumProperty(
        name="Refine Scheduler", description="Scheduler for refining",
        items=[
            ('sgm_uniform', 'SGM Uniform', ''),
            ('karras', 'Karras', ''),
            ('beta', 'Beta', ''),
            ('normal', 'Normal', ''),
            ('simple', 'Simple', ''),
        ],
        default='sgm_uniform', update=update_parameters
    )
    bpy.types.Scene.denoise = bpy.props.FloatProperty(
        name="Denoise", description="Denoise level for refining",
        default=0.8, min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.refine_cfg = bpy.props.FloatProperty(
        name="Refine CFG", description="Classifier-Free Guidance scale for refining",
        default=1.5, min=0.0, max=100.0, update=update_parameters
    )
    bpy.types.Scene.refine_prompt = bpy.props.StringProperty(
        name="Refine Prompt",
        description="Prompt for refining (leave empty to use same prompt as generation)",
        default="", update=update_parameters
    )
    bpy.types.Scene.refine_upscale_method = bpy.props.EnumProperty(
        name="Refine Upscale Method", description="Upscale method for refining",
        items=[
            ('nearest-exact', 'Nearest Exact', ''),
            ('bilinear', 'Bilinear', ''),
            ('bicubic', 'Bicubic', ''),
            ('lanczos', 'Lanczos', ''),
        ],
        default='lanczos', update=update_parameters
    )

    # ── Status / progress ──────────────────────────────────────────────
    bpy.types.Scene.generation_status = bpy.props.EnumProperty(
        name="Generation Status", description="Status of the generation process",
        items=[
            ('idle', 'Idle', ''),
            ('running', 'Running', ''),
            ('waiting', 'Waiting for cancel', ''),
            ('error', 'Error', '')
        ],
        default='idle', update=update_parameters
    )
    bpy.types.Scene.sg_last_gen_error = bpy.props.BoolProperty(
        name="Last Generation Error",
        description="True if the most recent generation ended with an error",
        default=False,
    )
    bpy.types.Scene.generation_progress = bpy.props.FloatProperty(
        name="Generation Progress", description="Current progress of image generation",
        default=0.0, min=0.0, max=100.0, update=update_parameters
    )

    # ── Material / blending ────────────────────────────────────────────
    bpy.types.Scene.overwrite_material = bpy.props.BoolProperty(
        name="Overwrite Material", description="Overwrite existing material",
        default=True, update=update_parameters
    )
    bpy.types.Scene.bake_visibility_weights = bpy.props.BoolProperty(
        name="Bake Visibility Weights",
        description="Pre-compute per-vertex visibility weights in Python instead of "
                    "using shader Raycast nodes. Makes projected textures survive "
                    "object transforms (move/rotate/scale) at the cost of vertex-level "
                    "weight resolution instead of per-pixel. Recommended for meshes "
                    "with >500 faces",
        default=False, update=update_parameters
    )
    bpy.types.Scene.discard_factor = bpy.props.FloatProperty(
        name="Discard Factor",
        description="If the texture is facing the camera at an angle greater than this value, it will be discarded. This is useful for preventing artifacts from the very edge of the generated texture appearing when keeping high discard factor (use ~65 for best results when generating textures around an object)",
        default=90.0, min=0.0, max=180.0, update=update_parameters
    )
    bpy.types.Scene.discard_factor_generation_only = bpy.props.BoolProperty(
        name="Reset Discard Angle After Generation",
        description="If enabled, the 'Discard Factor' will be reset to a specified value after generation completes. Useful for sequential/Qwen modes where a low discard angle is needed during generation but not for final blending",
        default=False, update=update_parameters
    )
    bpy.types.Scene.discard_factor_after_generation = bpy.props.FloatProperty(
        name="Discard Factor After Generation",
        description="The value to set the 'Discard Factor' to after generation is complete",
        default=90.0, min=0.0, max=180.0, update=update_parameters
    )
    bpy.types.Scene.weight_exponent_generation_only = bpy.props.BoolProperty(
        name="Reset Exponent After Generation",
        description="If enabled, the Weight Exponent will be reset to a specified value after generation completes. Useful for Voronoi projection mode where a high exponent produces sharp segmentation during generation but softer blending is preferred for the final result",
        default=False, update=update_parameters
    )
    bpy.types.Scene.weight_exponent_after_generation = bpy.props.FloatProperty(
        name="Exponent After Generation",
        description="The value to set the Weight Exponent to after generation is complete",
        default=15.0, min=0.01, max=1000.0, update=update_parameters
    )
    bpy.types.Scene.view_blend_use_color_match = bpy.props.BoolProperty(
        name="Match Colors to Viewport",
        description="Match each generated view's colors to the current viewport texture before blending",
        default=False, update=update_parameters,
    )
    bpy.types.Scene.view_blend_color_match_method = bpy.props.EnumProperty(
        name="Color Match Method",
        description="Algorithm used when matching view colors to the viewport texture",
        items=[
            ("mkl",        "MKL",           ""),
            ("hm",         "Histogram",     ""),
            ("reinhard",   "Reinhard",      ""),
            ("mvgd",       "MVGD",          ""),
            ("hm-mvgd-hm", "HM\u2013MVGD\u2013HM",    ""),
            ("hm-mkl-hm",  "HM\u2013MKL\u2013HM",     ""),
        ],
        default="hm-mvgd-hm", update=update_parameters,
    )
    bpy.types.Scene.view_blend_color_match_strength = bpy.props.FloatProperty(
        name="Match Strength",
        description="Blend between original and viewport-matched colors",
        default=1.0, min=0.0, max=2.0, update=update_parameters,
    )
    bpy.types.Scene.weight_exponent = bpy.props.FloatProperty(
        name="Weight Exponent",
        description="Controls the falloff curve for viewpoint weighting based on the angle to the surface normal (\u03b8). "
                    "Weight = |cos(\u03b8)|^Exponent. Higher values prioritize straight-on views more strongly, creating sharper transitions. "
                    "1.0 = standard |cos(\u03b8)| weighting.",
        default=3.0, min=0.1, max=1000.0, update=update_parameters
    )
    bpy.types.Scene.allow_modify_existing_textures = bpy.props.BoolProperty(
        name="Allow modifying existing textures",
        description="Disconnect compare node in export_visibility so that smooth output is not pure 1 areas",
        default=False, update=update_parameters
    )
    bpy.types.Scene.ask_object_prompts = bpy.props.BoolProperty(
        name="Ask for object prompts",
        description="Use object-specific prompts; if disabled, the normal prompt is used for all objects",
        default=True, update=update_parameters
    )
    bpy.types.Scene.fallback_color = bpy.props.FloatVectorProperty(
        name="Fallback Color",
        description="Color to use as fallback in texture generation",
        subtype='COLOR', default=(0.0, 0.0, 0.0), min=0.0, max=1.0,
        update=update_parameters
    )

    # ── Sequential / masking / inpainting ──────────────────────────────
    bpy.types.Scene.sequential_smooth = bpy.props.BoolProperty(
        name="Sequential Smooth",
        description="Use smooth visibility map for sequential generation mode. Disabling this uses a binary visibility map and may need more mask blurring to reduce artifacts.\n\n - Visibility map is a mask that indicates which pixels have textures already projected from previous viewpoints.\n - Both methods are using weights which are calculated based on the angle between the surface normal and the camera view direction.\n - 'Smooth' uses these calculated weights directly (0.0-1.0 range, giving gradual transitions). The transition point can be further tuned by the 'Smooth Factor' parameters.\n - Disabling 'Smooth' thresholds these weights to create a hard-edged binary mask (0.0 or 1.0).",
        default=True, update=update_parameters
    )
    bpy.types.Scene.weight_exponent_mask = bpy.props.BoolProperty(
        name="Weight Exponent Mask",
        description="Use weight exponent for visibility map generation. Uses 1.0 if disabled.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.canny_threshold_low = bpy.props.IntProperty(
        name="Canny Threshold Low", description="Low threshold for Canny edge detection",
        default=0, min=0, max=255, update=update_parameters
    )
    bpy.types.Scene.canny_threshold_high = bpy.props.IntProperty(
        name="Canny Threshold High", description="High threshold for Canny edge detection",
        default=80, min=0, max=255, update=update_parameters
    )
    bpy.types.Scene.sequential_factor_smooth = bpy.props.FloatProperty(
        name="Smooth Visibility Black Point",
        description="Controls the black point (start) of the Color Ramp used for the smooth visibility mask in sequential mode. Defines the weight threshold below which areas are considered fully invisible/untextured from previous views. Higher values create a sharper transition start.",
        default=0.15, min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.sequential_factor_smooth_2 = bpy.props.FloatProperty(
        name="Smooth Visibility White Point",
        description="Controls the white point (end) of the Color Ramp used for the smooth visibility mask in sequential mode. Defines the weight threshold above which areas are considered fully visible/textured from previous views. Lower values create a sharper transition end.",
        default=1.0, min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.sequential_factor = bpy.props.FloatProperty(
        name="Binary Visibility Threshold",
        description="Threshold value used when 'Sequential Smooth' is OFF. Calculated visibility weights below this value are treated as 0 (invisible), and those above as 1 (visible), creating a hard-edged binary mask.",
        default=0.7, min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.differential_noise = bpy.props.BoolProperty(
        name="Differential Noise",
        description="Adds latent noise mask to the image before inpainting. This must be used with low factor smooth mask or with a high blur mask radius. Disabling this effectively discrads the mask and only uses the inapaint conditioning.",
        default=True, update=update_parameters
    )
    bpy.types.Scene.grow_mask_by = bpy.props.IntProperty(
        name="Grow Mask By", description="Grow mask by this amount (ComfyUI)",
        default=3, min=0, update=update_parameters
    )
    bpy.types.Scene.mask_blocky = bpy.props.BoolProperty(
        name="Blocky Visibility Map",
        description="Uses a blocky visibility map. This will downscale the visibility map according to the 8x8 grid which Stable Diffusion uses in latent space. Highly experimental.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.visibility_vignette = bpy.props.BoolProperty(
        name="Feather Visibility Edges",
        description="Blend refinement edges using a vignette mask to reduce seams",
        default=True, update=update_parameters,
    )
    bpy.types.Scene.visibility_vignette_width = bpy.props.FloatProperty(
        name="Vignette Width",
        description="Fraction of the image radius used as a feather band (0 = no feather, 0.5 = very soft edges)",
        default=0.15, min=0.0, max=0.5, update=update_parameters,
    )
    bpy.types.Scene.visibility_vignette_softness = bpy.props.FloatProperty(
        name="Vignette Softness",
        description="Exponent shaping feather falloff (<1 = sharper transition, >1 = softer/wider transition)",
        default=1.0, min=0.1, max=5.0, update=update_parameters,
    )
    bpy.types.Scene.visibility_vignette_blur = bpy.props.BoolProperty(
        name="Blur Vignette Mask",
        description="Apply a Gaussian blur to the vignette mask to soften edges",
        default=False, update=update_parameters,
    )
    bpy.types.Scene.sg_silhouette_margin = bpy.props.IntProperty(
        name="Silhouette Margin",
        description="Pixel margin around occluder silhouettes where projection is suppressed. "
                    "Prevents thin outlines of foreground geometry (e.g. hands) bleeding onto surfaces behind them.  0 = disabled",
        default=3, min=0, max=20, update=update_parameters,
    )
    bpy.types.Scene.sg_silhouette_depth = bpy.props.FloatProperty(
        name="Silhouette Depth",
        description="Minimum depth difference (Blender units) between the vertex and an offset ray hit to consider it a silhouette edge. Raise if too much is eroded, lower if outlines remain",
        default=0.05, min=0.001, max=1.0, step=1, precision=3, update=update_parameters,
    )
    bpy.types.Scene.sg_silhouette_rays = bpy.props.EnumProperty(
        name="Silhouette Rays",
        description="Number of offset directions for silhouette detection. 4 = cardinal only (faster), 8 = cardinal + diagonal (better coverage)",
        items=[
            ('4', "4 (Cardinal)", "Right, Left, Up, Down"),
            ('8', "8 (Cardinal + Diagonal)", "Cardinal + 4 diagonal directions"),
        ],
        default='4', update=update_parameters,
    )

    # ── Refine mode ramp controls ──────────────────────────────────────
    bpy.types.Scene.refine_angle_ramp_active = bpy.props.BoolProperty(
        name="Use Angle Ramp", description="Blend refinement based on surface angle",
        default=True, update=update_parameters
    )
    bpy.types.Scene.refine_angle_ramp_pos_0 = bpy.props.FloatProperty(
        name="Angle Ramp Black Point", description="Position of the black point (Invisible) for the angle ramp",
        default=0.0, min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.refine_angle_ramp_pos_1 = bpy.props.FloatProperty(
        name="Angle Ramp White Point", description="Position of the white point (Visible) for the angle ramp",
        default=0.05, min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.refine_feather_ramp_pos_0 = bpy.props.FloatProperty(
        name="Feather Ramp Black Point", description="Position of the black point (Invisible) for the feather ramp",
        default=0.0, min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.refine_feather_ramp_pos_1 = bpy.props.FloatProperty(
        name="Feather Ramp White Point", description="Position of the white point (Visible) for the feather ramp",
        default=0.6, min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.refine_edge_feather_projection = bpy.props.BoolProperty(
        name="Edge Feather (Projection)",
        description="Add a screen-space distance-transform ramp at the projection silhouette boundary as an additional multiplier on top of the angle\u00d7feather weight. Interior surfaces keep full strength; only the geometric edge is softened",
        default=True, update=update_parameters
    )
    bpy.types.Scene.refine_edge_feather_width = bpy.props.IntProperty(
        name="Edge Feather Width",
        description="Width in pixels of the transition band at projection boundaries. Larger values produce a wider blend zone",
        default=30, min=1, max=200, update=update_parameters
    )
    bpy.types.Scene.refine_edge_feather_softness = bpy.props.FloatProperty(
        name="Edge Feather Softness",
        description="Rounds off the sharp corners at both ends of the linear feather ramp using a Gaussian blur. 0 = raw linear ramp (hard kinks at edge and interior boundary). 1 = moderate smoothing. Higher values give progressively gentler transitions without shifting the ramp position",
        default=1.0, min=0.0, max=5.0, step=10, precision=2, update=update_parameters
    )
    bpy.types.Scene.differential_diffusion = bpy.props.BoolProperty(
        name="Differential Diffusion",
        description="Replace standard inpainting with a differential diffusion based workflow\n\n - Generally works better and reduces artifacts.\n - Using a Smooth Visibilty Map is recommended for Sequential Mode.",
        default=True, update=update_parameters
    )
    bpy.types.Scene.blur_mask = bpy.props.BoolProperty(
        name="Blur Mask", description="Blur mask before inpainting (ComfyUI)",
        default=True, update=update_parameters
    )
    bpy.types.Scene.blur_mask_radius = bpy.props.IntProperty(
        name="Blur Mask Radius", description="Radius for mask blurring (ComfyUI)",
        default=1, min=1, max=31, update=update_parameters
    )
    bpy.types.Scene.blur_mask_sigma = bpy.props.FloatProperty(
        name="Blur Mask Sigma", description="Sigma for mask blurring (ComfyUI)",
        default=1.0, min=0.1, update=update_parameters
    )
    bpy.types.Scene.sequential_custom_camera_order = bpy.props.StringProperty(
        name="Custom Camera Order",
        description="Custom camera order for Sequential Mode. Format: 'index1,index2,index3,...'\n\n - This will permanently change the order of the cameras in the scene.",
        default="", update=update_parameters
    )
    bpy.types.Scene.clip_skip = bpy.props.IntProperty(
        name="CLIP Skip", description="CLIP skip value for generation",
        default=1, min=1, update=update_parameters
    )

    # ── Presets ────────────────────────────────────────────────────────
    bpy.types.Scene.stablegen_preset = bpy.props.EnumProperty(
        name="Preset", description="Select a preset for easy mode",
        items=get_preset_items, default=0
    )
    bpy.types.Scene.active_preset = bpy.props.StringProperty(
        name="Active Preset", default="DEFAULT"
    )

    # ── Architecture ───────────────────────────────────────────────────
    bpy.types.Scene.model_architecture = bpy.props.EnumProperty(
        name="Model Architecture",
        description="Select the model architecture to use for generation",
        items=[
            ('sdxl', 'SDXL', ''),
            ('flux1', 'FLUX.1', ''),
            ('qwen_image_edit', 'Qwen Image Edit', ''),
            ('flux2_klein', 'FLUX.2 Klein', '')
        ],
        default='sdxl',
        update=update_combined
    )
    bpy.types.Scene.architecture_mode = bpy.props.EnumProperty(
        name="Architecture",
        description="Select the overall generation architecture",
        items=[
            ('sdxl', 'SDXL', 'Stable Diffusion XL'),
            ('flux1', 'FLUX.1', 'FLUX.1 architecture'),
            ('qwen_image_edit', 'Qwen Image Edit', 'Qwen Image Edit architecture'),
            ('flux2_klein', 'FLUX.2 Klein', 'FLUX.2 Klein multi-reference edit model (Apache 2.0)'),
            ('trellis2', 'TRELLIS.2', 'Image to 3D mesh generation with TRELLIS.2'),
        ],
        default='sdxl',
        update=update_architecture_mode
    )

    # ── TRELLIS.2 ──────────────────────────────────────────────────────
    bpy.types.Scene.trellis2_generate_from = bpy.props.EnumProperty(
        name="Generate From", description="Source for the TRELLIS.2 input",
        items=[
            ('image', 'Image', 'Use an existing image as input'),
            ('prompt', 'Prompt', 'Generate an image from the prompt first, then feed to TRELLIS.2'),
        ],
        default='image', update=update_trellis2_generate_from
    )
    bpy.types.Scene.trellis2_texture_mode = bpy.props.EnumProperty(
        name="Texture Mode", description="How to generate textures for the 3D mesh",
        items=[
            ('none', 'None', 'Shape only, no texture generation'),
            ('native', 'Native (TRELLIS.2)', 'Use TRELLIS.2 built-in texture generation'),
            ('sdxl', 'SDXL', 'Use SDXL for camera-based texture projection'),
            ('flux1', 'FLUX.1', 'Use FLUX.1 for camera-based texture projection'),
            ('qwen_image_edit', 'Qwen Image Edit', 'Use Qwen for camera-based texture projection'),
            ('flux2_klein', 'FLUX.2 Klein', 'Use FLUX.2 Klein for camera-based texture projection'),
        ],
        default='native', update=update_trellis2_texture_mode
    )
    bpy.types.Scene.trellis2_initial_image_arch = bpy.props.EnumProperty(
        name="Initial Image Architecture",
        description="Diffusion architecture used to generate the initial image when Generate From is set to Prompt and Texture Mode is Native or None",
        items=[
            ('sdxl', 'SDXL', 'Stable Diffusion XL'),
            ('flux1', 'FLUX.1', 'FLUX.1 architecture'),
            ('qwen_image_edit', 'Qwen Image Edit', 'Qwen Image Edit architecture'),
            ('flux2_klein', 'FLUX.2 Klein', 'FLUX.2 Klein multi-reference edit'),
        ],
        default='sdxl', update=update_trellis2_initial_image_arch
    )
    bpy.types.Scene.trellis2_camera_count = bpy.props.IntProperty(
        name="Camera Count", description="Number of cameras to place around the generated mesh for texture projection",
        default=8, min=2, max=32, update=update_parameters
    )
    bpy.types.Scene.trellis2_placement_mode = bpy.props.EnumProperty(
        name="Placement Mode", description="Strategy for placing cameras around the generated mesh",
        items=[
            ('orbit_ring', "Orbit Ring", "Place cameras in a circle around the center"),
            ('hemisphere', "Sphere Coverage", "Distribute cameras evenly across a sphere using a Fibonacci spiral"),
            ('normal_weighted', "Normal-Weighted", "Automatically place cameras to cover the most surface area, using K-means on face normals weighted by area"),
            ('pca_axes', "PCA Axes", "Place cameras along the mesh's principal axes of variation"),
            ('greedy_coverage', "Greedy Coverage", "Iteratively add cameras that maximise new visible surface. Automatically determines the number of cameras needed"),
            ('fan_from_camera', "Fan from Camera", "Spread cameras in an arc around the active camera's orbit position"),
        ],
        default='normal_weighted', update=update_parameters
    )
    bpy.types.Scene.trellis2_auto_prompts = bpy.props.BoolProperty(
        name="Auto View Prompts", description="Automatically generate view-direction prompts for each camera (e.g. 'front view', 'rear view, from above')",
        default=True, update=update_parameters
    )
    bpy.types.Scene.trellis2_exclude_bottom = bpy.props.BoolProperty(
        name="Exclude Bottom Faces", description="Ignore downward-facing geometry when placing cameras",
        default=True, update=update_parameters
    )
    bpy.types.Scene.trellis2_exclude_bottom_angle = bpy.props.FloatProperty(
        name="Bottom Angle Threshold",
        description="Faces whose normal points more than this many degrees below the horizon are excluded",
        default=1.5533, min=0.1745, max=1.5708, subtype='ANGLE', unit='ROTATION', update=update_parameters
    )
    bpy.types.Scene.trellis2_auto_aspect = bpy.props.EnumProperty(
        name="Auto Aspect Ratio", description="Automatically adjust render aspect ratio to match the mesh silhouette",
        items=[
            ('off', "Off", "Use current scene resolution for all cameras"),
            ('shared', "Shared", "Average silhouette aspect across all cameras and set a single scene resolution"),
            ('per_camera', "Per Camera", "Each camera gets its own optimal aspect ratio"),
        ],
        default='per_camera', update=update_parameters
    )
    bpy.types.Scene.trellis2_occlusion_mode = bpy.props.EnumProperty(
        name="Occlusion Handling", description="How to account for self-occlusion when choosing camera directions",
        items=[
            ('none', "None (Fast)", "Back-face culling only \u2014 ignores self-occlusion"),
            ('full_matrix', "Full Occlusion Matrix", "Build a complete BVH-validated visibility matrix. Most accurate but slower"),
            ('two_pass', "Two-Pass Refinement", "Fast back-face pass, then targeted BVH refinement"),
            ('vis_weighted', "Visibility-Weighted", "Weight faces by visibility fraction; mostly-occluded faces have reduced influence"),
        ],
        default='none', update=update_parameters
    )
    bpy.types.Scene.trellis2_consider_existing = bpy.props.BoolProperty(
        name="Consider Existing Cameras",
        description="Treat existing cameras as already-placed directions so auto modes avoid duplicating their coverage",
        default=True, update=update_parameters
    )
    bpy.types.Scene.trellis2_delete_cameras = bpy.props.BoolProperty(
        name="Delete Cameras After",
        description="Automatically delete cameras placed during the TRELLIS.2 pipeline once generation and texturing are complete",
        default=False, update=update_parameters
    )
    bpy.types.Scene.trellis2_coverage_target = bpy.props.FloatProperty(
        name="Coverage Target", description="Stop adding cameras when this fraction of surface area is visible (Greedy mode)",
        default=0.95, min=0.5, max=1.0, subtype='FACTOR', update=update_parameters
    )
    bpy.types.Scene.trellis2_max_auto_cameras = bpy.props.IntProperty(
        name="Max Cameras (Greedy)", description="Upper limit on cameras for Greedy Coverage mode",
        default=12, min=2, max=50, update=update_parameters
    )
    bpy.types.Scene.trellis2_fan_angle = bpy.props.FloatProperty(
        name="Fan Angle", description="Total angular spread of the fan in degrees",
        default=90.0, min=10.0, max=350.0, update=update_parameters
    )
    bpy.types.Scene.trellis2_import_scale = bpy.props.FloatProperty(
        name="Import Scale (BU)",
        description="Scale imported TRELLIS.2 meshes so the longest axis equals this many Blender units. Set to 0 to keep the original size",
        default=2.0, min=0.0, max=100.0, soft_min=0.0, soft_max=10.0, update=update_parameters
    )
    bpy.types.Scene.trellis2_shade_mode = bpy.props.EnumProperty(
        name="Shading", description="Shading mode applied to imported TRELLIS.2 meshes",
        items=[
            ('flat', "Flat", "Keep flat shading (default)"),
            ('smooth', "Smooth", "Apply smooth shading to the entire mesh"),
            ('auto_smooth', "Auto Smooth", "Apply auto smooth shading by angle"),
        ],
        default='flat', update=update_parameters
    )
    bpy.types.Scene.trellis2_clamp_elevation = bpy.props.BoolProperty(
        name="Clamp Elevation",
        description="Restrict camera elevation to avoid extreme top-down or bottom-up views that diffusion models often struggle with",
        default=False, update=update_parameters
    )
    bpy.types.Scene.trellis2_max_elevation = bpy.props.FloatProperty(
        name="Max Elevation", description="Maximum upward elevation angle. Cameras looking further up will be clamped",
        default=1.2217, min=0.0, max=1.5708, subtype='ANGLE', unit='ROTATION', update=update_parameters
    )
    bpy.types.Scene.trellis2_min_elevation = bpy.props.FloatProperty(
        name="Min Elevation", description="Minimum downward elevation angle. Cameras looking further down will be clamped",
        default=-0.1745, min=-1.5708, max=0.0, subtype='ANGLE', unit='ROTATION', update=update_parameters
    )
    bpy.types.Scene.trellis2_preview_gallery_enabled = bpy.props.BoolProperty(
        name="Preview Gallery",
        description="When in prompt mode, generate multiple images with different seeds and let you pick the best one before proceeding to 3D generation",
        default=False, update=update_parameters
    )
    bpy.types.Scene.trellis2_preview_gallery_count = bpy.props.IntProperty(
        name="Gallery Count", description="Number of images to generate per batch in the preview gallery",
        default=4, min=1, max=16, update=update_parameters
    )

    # ── Qwen-specific ──────────────────────────────────────────────────
    bpy.types.Scene.qwen_guidance_map_type = bpy.props.EnumProperty(
        name="Guidance Map", description="The type of guidance map to use for Qwen Image Edit",
        items=[
            ('depth', 'Depth Map', 'Use depth map for structural guidance'),
            ('normal', 'Normal Map', 'Use normal map for structural guidance'),
            ('workbench', 'Workbench Render', 'Use workbench render for structural guidance'),
            ('viewport', 'Viewport Render', 'Use viewport render (OpenGL) for structural guidance')
        ],
        default='depth', update=update_parameters
    )
    bpy.types.Scene.qwen_voronoi_mode = bpy.props.BoolProperty(
        name="Voronoi Projection",
        description="Instead of zeroing weights of non-generated cameras, keep natural angle-based weights and project magenta from cameras that have not been generated yet. Use a high Weight Exponent to achieve Voronoi-like segmentation where each surface point is dominated by its closest camera. This eliminates the need for low discard-over-angle thresholds",
        default=False, update=update_parameters
    )
    bpy.types.Scene.qwen_context_render_mode = bpy.props.EnumProperty(
        name="Context Render", description="How to use the RGB context render in sequential mode for Qwen",
        items=[
            ('NONE', 'Disabled', 'Do not use the RGB context render'),
            ('REPLACE_STYLE', 'Replace Style Image', 'Use context render instead of the previous generated image as the style reference'),
            ('ADDITIONAL', 'Additional Context', 'Use context render as an additional image input (image 3) for context')
        ],
        default='NONE', update=update_parameters
    )
    bpy.types.Scene.qwen_use_external_style_image = bpy.props.BoolProperty(
        name="Use External Style Image",
        description="Use a separate, external image as the style reference for all viewpoints",
        default=False, update=update_parameters
    )
    bpy.types.Scene.qwen_external_style_image = bpy.props.StringProperty(
        name="Style Reference Image", description="Path to the external style reference image",
        default="", subtype='FILE_PATH', update=update_parameters
    )
    bpy.types.Scene.qwen_external_style_initial_only = bpy.props.BoolProperty(
        name="External for Initial Only",
        description="Use the external style image for the first image, then use the previously generated image for subsequent images",
        default=False, update=update_parameters
    )
    bpy.types.Scene.qwen_use_custom_prompts = bpy.props.BoolProperty(
        name="Use Custom Qwen Prompts",
        description="Enable to override the default guidance prompts for the Qwen Image Edit workflow",
        default=False, update=update_parameters
    )
    bpy.types.Scene.qwen_custom_prompt_initial = bpy.props.StringProperty(
        name="Initial Image Prompt",
        description="Custom prompt for the first generated image. Use {main_prompt} to insert the main prompt text.",
        default="Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2",
        update=update_parameters
    )
    bpy.types.Scene.qwen_custom_prompt_seq_none = bpy.props.StringProperty(
        name="Sequential Prompt (No Context)",
        description="Custom prompt for subsequent images when Context Render is 'Disabled'. Use {main_prompt} to insert the main prompt text.",
        default="Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2",
        update=update_parameters
    )
    bpy.types.Scene.qwen_custom_prompt_seq_replace = bpy.props.StringProperty(
        name="Sequential Prompt (Replace Style)",
        description="Custom prompt for subsequent images when Context Render is 'Replace Style'. Use {main_prompt} to insert the main prompt text.",
        default="Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.",
        update=update_parameters
    )
    bpy.types.Scene.qwen_custom_prompt_seq_additional = bpy.props.StringProperty(
        name="Sequential Prompt (Additional Context)",
        description="Custom prompt for subsequent images when Context Render is 'Additional Context'. Use {main_prompt} to insert the main prompt text.",
        default="Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.",
        update=update_parameters
    )
    bpy.types.Scene.qwen_guidance_fallback_color = bpy.props.FloatVectorProperty(
        name="Guidance Fallback Color", description="Color used for fallback regions in the Qwen context render",
        subtype='COLOR', default=(1.0, 0.0, 1.0), min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.qwen_guidance_background_color = bpy.props.FloatVectorProperty(
        name="Guidance Background Color", description="Background color for the Qwen context render",
        subtype='COLOR', default=(1.0, 0.0, 1.0), min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.qwen_context_cleanup = bpy.props.BoolProperty(
        name="Use Context Cleanup",
        description="Replace fallback color in subsequent Qwen renders before projection",
        default=False, update=update_parameters
    )
    bpy.types.Scene.qwen_context_cleanup_hue_tolerance = bpy.props.FloatProperty(
        name="Cleanup Hue Tolerance", description="Hue tolerance in degrees for identifying fallback regions during cleanup",
        default=5.0, min=0.0, max=180.0, update=update_parameters
    )
    bpy.types.Scene.qwen_context_cleanup_value_adjust = bpy.props.FloatProperty(
        name="Cleanup Value Adjustment",
        description="Adjust value (brightness) for cleaned pixels. -1 darkens to black, 1 brightens to white.",
        default=0.0, min=-1.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.qwen_context_fallback_dilation = bpy.props.IntProperty(
        name="Fallback Dilation",
        description="Dilate fallback color regions in the context render before sending to Qwen.",
        default=1, min=0, max=64, update=update_parameters
    )
    bpy.types.Scene.qwen_prompt_gray_background = bpy.props.BoolProperty(
        name="Gray Background Prompt",
        description="Include 'Replace the background with solid gray' in the default context-render prompt. Disable to let the model choose the background freely",
        default=True, update=update_parameters
    )

    # ── Misc ───────────────────────────────────────────────────────────
    bpy.types.Scene.output_timestamp = bpy.props.StringProperty(
        name="Output Timestamp", description="Timestamp for generation output directory", default=""
    )
    bpy.types.Scene.camera_prompts = bpy.props.CollectionProperty(
        type=CameraPromptItem, name="Camera Prompts",
        description="Stores viewpoint descriptions for each camera"
    )
    bpy.types.Scene.use_camera_prompts = bpy.props.BoolProperty(
        name="Use Camera Prompts", description="Use camera prompts for generating images",
        default=True, update=update_parameters
    )
    bpy.types.Scene.sg_camera_order = bpy.props.CollectionProperty(
        type=CameraOrderItem, name="Camera Generation Order",
        description="Defines the order in which cameras are processed during generation"
    )
    bpy.types.Scene.sg_camera_order_index = bpy.props.IntProperty(
        name="Active Camera Order Index", default=0
    )
    bpy.types.Scene.sg_use_custom_camera_order = bpy.props.BoolProperty(
        name="Use Custom Camera Order",
        description="When enabled, generation uses the custom camera order list instead of alphabetical sorting",
        default=False, update=update_parameters
    )

    # ── UI section toggles ─────────────────────────────────────────────
    bpy.types.Scene.show_core_settings = bpy.props.BoolProperty(
        name="Core Generation Settings",
        description="Parameters used for the image generation process. Also includes LoRAs for faster generation.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.show_lora_settings = bpy.props.BoolProperty(
        name="LoRA Settings", description="Settings for custom LoRA management.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.show_camera_options = bpy.props.BoolProperty(
        name="Camera Settings", description="Camera prompt and generation order settings.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.show_scene_understanding_settings = bpy.props.BoolProperty(
        name="Viewpoint Blending Settings",
        description="Settings for how the addon blends different viewpoints together.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.show_output_material_settings = bpy.props.BoolProperty(
        name="Output & Material Settings",
        description="Settings for output characteristics and material handling, including texture processing and final image resolution.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.show_image_guidance_settings = bpy.props.BoolProperty(
        name="Image Guidance (IPAdapter & ControlNet)",
        description="Configuration for advanced image guidance techniques, allowing more precise control via reference images or structural inputs.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.show_masking_inpainting_settings = bpy.props.BoolProperty(
        name="Inpainting Options",
        description="Parameters for inpainting and mask manipulation to refine specific image areas. (Visible for UV Inpaint & Sequential modes).",
        default=False, update=update_parameters
    )
    bpy.types.Scene.show_mode_specific_settings = bpy.props.BoolProperty(
        name="Generation Mode Specifics",
        description="Parameters exclusively available for the selected Generation Mode, allowing tailored control over mode-dependent behaviors.",
        default=False, update=update_parameters
    )

    # ── Generation mode / priority ─────────────────────────────────────
    bpy.types.Scene.generation_mode = bpy.props.EnumProperty(
        name="Generation Mode", description="Controls the generation behavior",
        items=[
            ('standard', 'Standard', 'Standard generation process'),
            ('regenerate_selected', 'Regenerate Selected', 'Regenerate only specific viewpoints, keeping the rest from the previous run'),
            ('project_only', 'Project Only', 'Only project existing textures onto the model without generating new ones')
        ],
        default='standard', update=update_parameters
    )
    bpy.types.Scene.early_priority_strength = bpy.props.FloatProperty(
        name="Prioritize Initial Views",
        description="Strength of the priority applied to initial views. Higher values will make the earlier cameras more important than the later ones. Every view will be prioritized over the next one.\n    - Very high values may cause various artifacts.",
        default=0.5, min=0.0, max=1.0, update=update_parameters
    )
    bpy.types.Scene.early_priority = bpy.props.BoolProperty(
        name="Priority Strength",
        description="Enable blending priority for earlier cameras.\n    - This may prevent artifacts caused by later cameras overwriting earlier ones.\n    - You will have to place the important cameras first.",
        default=False, update=update_parameters
    )
    bpy.types.Scene.texture_objects = bpy.props.EnumProperty(
        name="Objects to Texture", description="Select the objects to texture",
        items=[
            ('all', 'All Visible', 'Texture all visible objects in the scene'),
            ('selected', 'Selected', 'Texture only selected objects'),
        ],
        default='all', update=update_parameters
    )
    bpy.types.Scene.use_flux_lora = bpy.props.BoolProperty(
        name="Use FLUX Depth LoRA",
        description="Use FLUX.1-Depth-dev LoRA for depth conditioning instead of ControlNet. This disables all ControlNet units.",
        default=True, update=update_parameters
    )

    # ── ControlNet / LoRA collections ──────────────────────────────────
    bpy.types.Scene.controlnet_units = bpy.props.CollectionProperty(type=ControlNetUnit)
    bpy.types.Scene.lora_units = bpy.props.CollectionProperty(type=LoRAUnit)
    bpy.types.Scene.controlnet_units_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.lora_units_index = bpy.props.IntProperty(default=0)
    bpy.app.handlers.load_post.append(load_handler)
    bpy.app.handlers.load_post.append(_sg_queue_load_handler)

    # ── TRELLIS.2 status / flags ───────────────────────────────────────
    bpy.types.Scene.trellis2_available = bpy.props.BoolProperty(
        name="TRELLIS.2 Available",
        description="Whether TRELLIS.2 nodes are available on the ComfyUI server (auto-detected)",
        default=False
    )
    bpy.types.Scene.pbr_nodes_available = bpy.props.BoolProperty(
        name="PBR Nodes Available",
        description="Whether PBR decomposition nodes (Marigold IID + StableDelight) are available on the ComfyUI server (auto-detected)",
        default=False
    )
    bpy.types.Scene.show_trellis2_params = bpy.props.BoolProperty(
        name="Show TRELLIS.2 Section", description="Toggle TRELLIS.2 Image-to-3D section", default=False
    )
    bpy.types.Scene.show_trellis2_advanced = bpy.props.BoolProperty(
        name="Show TRELLIS.2 Settings", description="Toggle TRELLIS.2 advanced settings", default=False
    )
    bpy.types.Scene.show_trellis2_mesh_settings = bpy.props.BoolProperty(
        name="Show Mesh Generation Settings", description="Toggle TRELLIS.2 mesh generation advanced settings", default=False
    )
    bpy.types.Scene.show_trellis2_texture_settings = bpy.props.BoolProperty(
        name="Show Texture Settings", description="Toggle TRELLIS.2 native texture settings", default=False
    )
    bpy.types.Scene.show_trellis2_camera_settings = bpy.props.BoolProperty(
        name="Show Camera Placement Settings", description="Toggle TRELLIS.2 camera placement settings", default=False
    )
    bpy.types.Scene.trellis2_last_input_image = bpy.props.StringProperty(
        name="TRELLIS.2 Last Input Image",
        description="Path to the image most recently used as TRELLIS.2 input (set automatically after generation)",
        subtype='FILE_PATH', default=""
    )
    bpy.types.Scene.qwen_use_trellis2_style = bpy.props.BoolProperty(
        name="Use TRELLIS.2 Input as Style",
        description="Use the TRELLIS.2 input image as the Qwen style reference",
        default=False, update=update_parameters
    )
    bpy.types.Scene.qwen_trellis2_style_initial_only = bpy.props.BoolProperty(
        name="TRELLIS.2 Style for Initial Only",
        description="Use the TRELLIS.2 input image only for the first image, then fall back to sequential style for subsequent images",
        default=False, update=update_parameters
    )
    bpy.types.Scene.trellis2_pipeline_active = bpy.props.BoolProperty(
        name="TRELLIS.2 Pipeline Active",
        description="True while the TRELLIS.2 pipeline (incl. texturing) is running",
        default=False
    )
    bpy.types.Scene.trellis2_pipeline_phase_start_pct = bpy.props.FloatProperty(
        name="Phase Start %", description="Overall-% at which the current texturing phase begins",
        default=0.0, min=0.0, max=100.0
    )
    bpy.types.Scene.trellis2_pipeline_total_phases = bpy.props.IntProperty(
        name="Total Phases", description="Total number of phases in the TRELLIS.2 pipeline",
        default=3, min=1, max=3
    )
    bpy.types.Scene.trellis2_input_image = bpy.props.StringProperty(
        name="Input Image", description="Path to the reference image for 3D mesh generation",
        subtype='FILE_PATH', default=""
    )
    # --- Batch generation ---
    bpy.types.Scene.trellis2_batch_folder = bpy.props.StringProperty(
        name="Batch Folder",
        description="Folder containing images for batch TRELLIS.2 generation",
        subtype='DIR_PATH', default=""
    )
    bpy.types.Scene.trellis2_batch_count = bpy.props.IntProperty(
        name="Batch Image Count",
        description="Number of supported images found in the batch folder",
        default=0, min=0
    )
    bpy.types.Scene.trellis2_batch_rename_meshes = bpy.props.BoolProperty(
        name="Name meshes after input files",
        description="Rename each imported mesh to the stem of its source image filename",
        default=True
    )
    bpy.types.Scene.trellis2_batch_bake_textures = bpy.props.BoolProperty(
        name="Bake Textures after each model",
        description="Automatically bake textures after each model is generated",
        default=True
    )
    bpy.types.Scene.trellis2_batch_bake_pbr = bpy.props.BoolProperty(
        name="Bake PBR Maps",
        description="Bake individual PBR channel maps (BaseColor, Roughness, Metallic, Normal, Emission, Height, AO)",
        default=True
    )
    bpy.types.Scene.trellis2_batch_bake_resolution = bpy.props.IntProperty(
        name="Texture Resolution",
        description="Resolution of the baked textures",
        default=2048, min=128, max=8192
    )
    bpy.types.Scene.trellis2_batch_bake_try_unwrap = bpy.props.EnumProperty(
        name="Unwrap Method",
        description="Method to unwrap UVs before baking",
        items=[
            ('none',      'None',            'Skip UV unwrapping'),
            ('cube',      'Cube Project',    'Fast cube projection'),
            ('smart',     'Smart UV Project','Smart UV Project'),
            ('basic',     'Basic Unwrap',    'Angle-based unwrap'),
            ('lightmap',  'Lightmap Pack',   'Lightmap Pack'),
            ('pack',      'Pack Islands',    'Pack Islands'),
        ],
        default='smart'
    )
    bpy.types.Scene.trellis2_batch_bake_overlap_only = bpy.props.BoolProperty(
        name="Overlap Only",
        description="Only unwrap objects with overlapping UVs",
        default=False
    )
    bpy.types.Scene.trellis2_batch_bake_export_orm = bpy.props.BoolProperty(
        name="Pack ORM Texture",
        description="Create a packed ORM texture (R=AO, G=Roughness, B=Metallic) for Unreal Engine / glTF",
        default=False
    )
    bpy.types.Scene.trellis2_batch_bake_normal_convention = bpy.props.EnumProperty(
        name="Normal Convention",
        description="Normal map Y-axis convention",
        items=[
            ('opengl',  'OpenGL (Y+)',  'Standard OpenGL / glTF / Unity / Blender convention'),
            ('directx', 'DirectX (Y-)', 'DirectX / Unreal Engine convention'),
        ],
        default='opengl'
    )
    bpy.types.Scene.trellis2_batch_bake_add_material = bpy.props.BoolProperty(
        name="Add Material",
        description="Add the baked texture as a material to the objects",
        default=True
    )
    bpy.types.Scene.trellis2_batch_bake_flatten = bpy.props.BoolProperty(
        name="Bake & Continue Refining",
        description="After baking, apply the baked texture to the StableGen projection material",
        default=False
    )
    # WindowManager props for live batch progress display
    bpy.types.WindowManager.sg_batch_running = bpy.props.BoolProperty(default=False)
    bpy.types.WindowManager.sg_batch_index = bpy.props.IntProperty(default=0)
    bpy.types.WindowManager.sg_batch_total = bpy.props.IntProperty(default=0)
    bpy.types.Scene.trellis2_resolution = bpy.props.EnumProperty(
        name="Resolution", description="Model resolution for generation. Higher values use more VRAM",
        items=[
            ('512', '512', 'Low resolution, fast, less VRAM'),
            ('1024', '1024', 'Direct 1024 generation with higher sparse structure resolution'),
            ('1024_cascade', '1024 Cascade', 'Medium resolution with cascade (recommended)'),
            ('1536_cascade', '1536 Cascade', 'High resolution with cascade, most VRAM'),
        ],
        default='1024_cascade', update=update_parameters
    )
    bpy.types.Scene.trellis2_vram_mode = bpy.props.EnumProperty(
        name="VRAM Mode", description="Controls model offloading strategy to manage VRAM usage",
        items=[
            ('keep_loaded', 'Keep Loaded', 'Keep all models in VRAM (fastest, ~12GB VRAM)'),
            ('disk_offload', 'Disk Offload', 'Load models on demand from disk (recommended for <=16GB VRAM)'),
        ],
        default='disk_offload', update=update_parameters
    )
    bpy.types.Scene.trellis2_attn_backend = bpy.props.EnumProperty(
        name="Attention Backend", description="Attention implementation to use. flash_attn is fastest but requires CUDA",
        items=[
            ('flash_attn', 'Flash Attention', 'Fastest (requires flash-attn package)'),
            ('xformers', 'xFormers', 'Fast (requires xformers package)'),
            ('sdpa', 'SDPA', 'PyTorch native, always available'),
            ('sageattn', 'SageAttention', 'SageAttention backend (requires sageattn package)'),
        ],
        default='flash_attn', update=update_parameters
    )
    bpy.types.Scene.trellis2_seed = bpy.props.IntProperty(
        name="Seed", description="Random seed for generation (0 = random)",
        default=0, min=0, max=2147483647
    )
    bpy.types.Scene.trellis2_ss_guidance = bpy.props.FloatProperty(
        name="SS Guidance", description="Sparse structure CFG scale. Higher = stronger adherence to input image",
        default=7.5, min=1.0, max=20.0, step=10, update=update_parameters
    )
    bpy.types.Scene.trellis2_ss_steps = bpy.props.IntProperty(
        name="SS Steps", description="Sparse structure sampling steps. More steps = better quality but slower",
        default=12, min=1, max=50, update=update_parameters
    )
    bpy.types.Scene.trellis2_shape_guidance = bpy.props.FloatProperty(
        name="Shape Guidance", description="Shape CFG scale. Higher = stronger adherence to input image",
        default=7.5, min=1.0, max=20.0, step=10, update=update_parameters
    )
    bpy.types.Scene.trellis2_shape_steps = bpy.props.IntProperty(
        name="Shape Steps", description="Shape sampling steps. More steps = better quality but slower",
        default=12, min=1, max=50, update=update_parameters
    )
    bpy.types.Scene.trellis2_tex_guidance = bpy.props.FloatProperty(
        name="Texture Guidance", description="Texture CFG scale. Higher = stronger adherence to input image",
        default=7.5, min=1.0, max=20.0, step=10, update=update_parameters
    )
    bpy.types.Scene.trellis2_tex_steps = bpy.props.IntProperty(
        name="Texture Steps", description="Texture sampling steps. More steps = better quality but slower",
        default=12, min=1, max=50, update=update_parameters
    )
    bpy.types.Scene.trellis2_max_tokens = bpy.props.IntProperty(
        name="Max Tokens",
        description="Max sparse-voxel tokens during cascade upsampling (only affects cascade modes). "
                    "Higher = more detail but more VRAM. Increase to 49152 for maximum detail if VRAM allows",
        default=32768, min=16384, max=65536, step=4096, update=update_parameters
    )
    bpy.types.Scene.trellis2_texture_size = bpy.props.IntProperty(
        name="Texture Size",
        description="Resolution of the UV-baked texture in pixels. Higher values reduce aliasing at UV seams. "
                    "The native voxel grid is 1024, so values above 1024 interpolate existing data",
        default=4096, min=512, max=8192, step=512, update=update_parameters
    )
    bpy.types.Scene.trellis2_decimation = bpy.props.IntProperty(
        name="Decimation Target", description="Target polygon count for mesh simplification. Lower = simpler mesh",
        default=100000, min=1000, max=5000000, step=10000, update=update_parameters
    )
    bpy.types.Scene.trellis2_remesh = bpy.props.BoolProperty(
        name="Remesh", description="Apply remeshing for cleaner topology",
        default=True, update=update_parameters
    )
    bpy.types.Scene.trellis2_post_processing_enabled = bpy.props.BoolProperty(
        name="Post-Processing",
        description="Run ComfyUI-side mesh post-processing (decimation + remeshing). Disable to import the raw mesh for manual retopology",
        default=True, update=update_parameters
    )
    bpy.types.Scene.trellis2_auto_lighting = bpy.props.BoolProperty(
        name="Auto Studio Lighting",
        description="Create a three-point studio lighting rig (key, fill, rim) after import to showcase PBR materials",
        default=True, update=update_parameters
    )
    bpy.types.Scene.trellis2_skip_texture = bpy.props.BoolProperty(
        name="Skip Texture", description="Export shape-only mesh (no PBR textures). Much faster and uses less VRAM",
        default=False, update=update_parameters
    )
    bpy.types.Scene.trellis2_bg_removal = bpy.props.EnumProperty(
        name="Background Removal", description="Method for removing the image background before TRELLIS.2 processing",
        items=[
            ('auto', 'Auto (BiRefNet)', 'Automatically remove background using BiRefNet model'),
            ('skip', 'Skip (Use Alpha)', 'Skip background removal \u2014 use the input image\'s alpha channel as mask. '
             'Use this when your image already has a transparent background'),
        ],
        default='auto', update=update_parameters
    )
    bpy.types.Scene.trellis2_background_color = bpy.props.EnumProperty(
        name="Background Color", description="Background color for image conditioning",
        items=[
            ('black', 'Black', 'Black background (default)'),
            ('gray', 'Gray', 'Gray background'),
            ('white', 'White', 'White background'),
        ],
        default='black', update=update_parameters
    )

    # ── PBR Decomposition ──────────────────────────────────────────────
    bpy.types.Scene.pbr_decomposition = bpy.props.BoolProperty(
        name="PBR Decomposition",
        description="Run Marigold decomposition on each generated image to produce PBR material maps (albedo, roughness, metallic, normal, depth)",
        default=False, update=update_parameters
    )
    bpy.types.Scene.pbr_albedo_source = bpy.props.EnumProperty(
        name="Albedo Source", description="Model to use for extracting the Base Color / albedo map",
        items=[
            ('marigold', "Marigold IID (Flat Albedo)", "True albedo from Marigold IID-Appearance \u2014 removes all lighting but may lose texture detail"),
            ('delight', "StableDelight (Delighted)", "Specular-removed image via StableDelight \u2014 preserves diffuse shading and texture detail, only strips highlights"),
            ('lighting', "Marigold IID-Lighting (Vibrant)", "Albedo from Marigold IID-Lighting \u2014 tends to produce more vibrant, saturated colours than IID-Appearance"),
        ],
        default='marigold', update=update_parameters,
    )
    bpy.types.Scene.pbr_map_albedo = bpy.props.BoolProperty(
        name="Albedo", description="Extract albedo (diffuse colour without lighting) and use it as Base Color",
        default=True, update=update_parameters
    )
    bpy.types.Scene.pbr_map_roughness = bpy.props.BoolProperty(
        name="Roughness", description="Extract roughness map (0 = mirror, 1 = rough)",
        default=True, update=update_parameters
    )
    bpy.types.Scene.pbr_map_metallic = bpy.props.BoolProperty(
        name="Metallic", description="Extract metallic map (0 = dielectric, 1 = metal)",
        default=True, update=update_parameters
    )
    bpy.types.Scene.pbr_map_normal = bpy.props.BoolProperty(
        name="Normal",
        description="Extract surface normal map for detail and bump. Warning: may cause triangle artifacts on voxel-remeshed geometry",
        default=True, update=update_parameters
    )
    bpy.types.Scene.pbr_normal_strength = bpy.props.FloatProperty(
        name="Normal Strength",
        description="How strongly the normal map perturbs the surface shading. Lower values reduce triangle artifacts on poor topology",
        default=1.0, min=0.0, max=2.0, step=0.05, update=update_parameters
    )
    bpy.types.Scene.pbr_delight_strength = bpy.props.FloatProperty(
        name="Delight Strength",
        description="How strongly StableDelight removes specular reflections. Lower values preserve more original texture detail. 1.0 = full delighting, 0.5 = subtle",
        default=1.0, min=0.01, max=5.0, step=0.1, update=update_parameters
    )
    bpy.types.Scene.pbr_map_height = bpy.props.BoolProperty(
        name="Height", description="Extract height/displacement map via Marigold depth estimation",
        default=False, update=update_parameters
    )
    bpy.types.Scene.pbr_height_scale = bpy.props.FloatProperty(
        name="Height Scale", description="Displacement scale for the height map. Higher values produce more pronounced surface displacement",
        default=0.1, min=0.001, max=2.0, step=0.01, precision=3, update=update_parameters
    )
    bpy.types.Scene.pbr_map_ao = bpy.props.BoolProperty(
        name="AO", description="Bake an Ambient Occlusion map from mesh geometry (uses Blender's built-in bake, no ML required)",
        default=False, update=update_parameters
    )
    bpy.types.Scene.pbr_ao_samples = bpy.props.IntProperty(
        name="AO Samples", description="Number of samples for AO baking. Higher = cleaner but slower",
        default=16, min=1, max=128, update=update_parameters
    )
    bpy.types.Scene.pbr_ao_distance = bpy.props.FloatProperty(
        name="AO Distance", description="Maximum ray distance for AO baking. 0 = scene size (automatic)",
        default=0.0, min=0.0, max=100.0, step=0.1, update=update_parameters
    )
    bpy.types.Scene.pbr_map_emission = bpy.props.BoolProperty(
        name="Emission", description="Extract an emission map (experimental). Identifies self-illuminating regions in the generated texture",
        default=False, update=update_parameters
    )
    bpy.types.Scene.pbr_emission_method = bpy.props.EnumProperty(
        name="Emission Method", description="Algorithm used to detect emissive regions",
        items=[
            ('residual', "IID-Lighting Residual", "Run Marigold IID-Lighting and extract the residual (image \u2212 albedo \u00d7 shading). Most accurate but requires an additional model pass"),
            ('hsv', "HSV Threshold", "Isolate pixels with high saturation + high value in HSV space. Fast, zero model cost, best for neon/sci-fi styles"),
        ],
        default='residual', update=update_parameters
    )
    bpy.types.Scene.pbr_emission_threshold = bpy.props.FloatProperty(
        name="Emission Threshold",
        description="Minimum value to consider as emission. In Residual mode, fades out low-luminance residuals. In HSV mode, zeroes the blurred mask below this cutoff. Higher = fewer false positives, lower = catches subtle glow",
        default=0.2, min=0.0, max=1.0, step=0.05, precision=2, update=update_parameters
    )
    bpy.types.Scene.pbr_emission_saturation_min = bpy.props.FloatProperty(
        name="Saturation Min",
        description="(HSV method) Minimum saturation for a pixel to be considered emissive. Glowing objects retain colour intensity while lit surfaces wash out",
        default=0.5, min=0.0, max=1.0, step=0.05, precision=2, update=update_parameters
    )
    bpy.types.Scene.pbr_emission_value_min = bpy.props.FloatProperty(
        name="Value Min", description="(HSV method) Minimum value/brightness for a pixel to be considered emissive",
        default=0.85, min=0.0, max=1.0, step=0.05, precision=2, update=update_parameters
    )
    bpy.types.Scene.pbr_emission_bloom = bpy.props.FloatProperty(
        name="Bloom Radius",
        description="(HSV method) Gaussian blur radius applied to the emission mask to simulate bloom / glow bleed. 0 = sharp mask, higher = more bloom",
        default=5.0, min=0.0, max=50.0, step=1.0, precision=1, update=update_parameters
    )
    bpy.types.Scene.pbr_emission_strength = bpy.props.FloatProperty(
        name="Emission Strength", description="Strength of the emission channel in the Principled BSDF",
        default=2.5, min=0.0, max=100.0, step=0.5, precision=1, update=update_parameters
    )
    bpy.types.Scene.pbr_use_native_resolution = bpy.props.BoolProperty(
        name="Use Native Resolution",
        description="Process PBR maps at the image's native resolution (longest edge, rounded to 64px). Produces sharper results but uses more VRAM. When disabled, the fixed Processing Resolution is used",
        default=True, update=update_parameters
    )
    bpy.types.Scene.pbr_tiling = bpy.props.EnumProperty(
        name="Tiling", description="Tile-based super-resolution for PBR maps. Each tile is upscaled to the full image resolution before processing, producing N\u00b2\u00d7 the effective detail",
        items=[
            ('off', "Off", "No tiling \u2014 process the full image in one pass"),
            ('selective', "Selective", "Tile albedo only (StableDelight and/or IID albedo). Normals, roughness, metallic and height are processed normally"),
            ('all', "All", "Tile every PBR model including normals and height"),
            ('custom', "Custom", "Choose which maps to tile individually. Models are run as efficiently as possible"),
        ],
        default='selective', update=update_parameters
    )
    bpy.types.Scene.pbr_tile_albedo = bpy.props.BoolProperty(
        name="Tile Albedo", description="Tile the albedo map for higher detail",
        default=True, update=update_parameters
    )
    bpy.types.Scene.pbr_tile_material = bpy.props.BoolProperty(
        name="Tile Material", description="Tile the roughness and metallic maps (both come from the same IID-Appearance material output)",
        default=False, update=update_parameters
    )
    bpy.types.Scene.pbr_tile_normal = bpy.props.BoolProperty(
        name="Tile Normal", description="Tile the normal map for higher detail",
        default=False, update=update_parameters
    )
    bpy.types.Scene.pbr_tile_height = bpy.props.BoolProperty(
        name="Tile Height", description="Tile the height/displacement map for higher detail",
        default=False, update=update_parameters
    )
    bpy.types.Scene.pbr_tile_emission = bpy.props.BoolProperty(
        name="Tile Emission",
        description="Tile the IID-Lighting residual used for emission (only applies to the Residual emission method)",
        default=False, update=update_parameters
    )
    bpy.types.Scene.pbr_tile_grid = bpy.props.IntProperty(
        name="Tile Grid",
        description="N\u00d7N grid size for tiling. 2 = 4 tiles (4\u00d7 detail), 3 = 9 tiles (9\u00d7 detail), 4 = 16 tiles (16\u00d7 detail). Processing time scales with N\u00b2",
        default=2, min=2, max=4, update=update_parameters
    )
    bpy.types.Scene.pbr_tile_superres = bpy.props.BoolProperty(
        name="Super Resolution",
        description="When enabled, the stitched PBR maps are kept at the upscaled tile resolution (~N\u00d7 the original image size). When disabled (default), tiles are scaled back to the original image resolution \u2014 still higher detail from tiled processing, but matching the source texture size",
        default=True, update=update_parameters
    )
    bpy.types.Scene.pbr_processing_resolution = bpy.props.IntProperty(
        name="Processing Resolution",
        description="Internal processing resolution for the Marigold model. Output is upscaled back to the original resolution. 768 is the default for the model's training resolution",
        default=768, min=256, max=2048, step=64, update=update_parameters
    )
    bpy.types.Scene.pbr_denoise_steps = bpy.props.IntProperty(
        name="Denoise Steps",
        description="Number of denoising steps. More steps = better quality but slower. 4 is a good balance",
        default=4, min=1, max=50, update=update_parameters
    )
    bpy.types.Scene.pbr_ensemble_size = bpy.props.IntProperty(
        name="Ensemble Size",
        description="Number of ensemble predictions to average. Higher = more stable results but linearly slower. 1 is usually sufficient",
        default=1, min=1, max=10, update=update_parameters
    )
    bpy.types.Scene.pbr_albedo_auto_saturation = bpy.props.BoolProperty(
        name="Correct Albedo Saturation",
        description="Automatically correct albedo saturation by comparing the PBR albedo against the original rendered image and boosting to match. Averaged across all cameras for uniform results by default. Recommended for Marigold IID (Flat Albedo) which tends to desaturate. Usually not needed for StableDelight or IID-Lighting",
        default=True, update=update_parameters
    )
    bpy.types.Scene.pbr_albedo_saturation_mode = bpy.props.EnumProperty(
        name="Saturation Mode",
        description="How to compute the albedo saturation correction ratio. Mean averages all camera ratios (sensitive to outliers). Median takes the middle value (robust to outliers). Per Camera applies an individual ratio to each camera",
        items=[
            ('MEDIAN', "Median", "Use the median ratio across cameras (recommended, outlier-resistant)"),
            ('MEAN', "Mean", "Use the arithmetic mean of all camera ratios"),
            ('PER_CAMERA', "Per Camera", "Compute and apply an individual ratio for each camera"),
        ],
        default='MEDIAN', update=update_parameters
    )
    bpy.types.Scene.pbr_replace_color_with_albedo = bpy.props.BoolProperty(
        name="Use Albedo as Base Color",
        description="Replace the projected colour texture with the albedo map. This effectively delights the texture",
        default=True, update=update_parameters
    )
    bpy.types.Scene.pbr_auto_lighting = bpy.props.BoolProperty(
        name="Studio Lighting",
        description="Create a three-point studio lighting rig (key, fill, rim) after PBR projection to showcase PBR materials",
        default=False, update=update_parameters
    )


# ── Unregistration ─────────────────────────────────────────────────────────

def unregister_properties(load_handler, _sg_queue_load_handler):
    """Remove all properties registered by ``register_properties``."""

    if hasattr(bpy.types.Scene, 'controlnet_model_mappings'):
        del bpy.types.Scene.controlnet_model_mappings
    if hasattr(bpy.types.Scene, 'controlnet_mapping_index'):
        del bpy.types.Scene.controlnet_mapping_index

    _simple_scene_props = [
        'use_flux_lora',
        'comfyui_prompt', 'use_separate_texture_prompt', 'texture_prompt',
        'comfyui_negative_prompt', 'model_name', 'sg_model_name_backup',
        'seed', 'control_after_generate', 'steps', 'cfg',
        'sampler', 'scheduler', 'show_advanced_params', 'show_generation_params',
        'auto_rescale', 'qwen_rescale_alignment', 'auto_rescale_target_mp',
        'generation_method', 'use_ipadapter',
        'refine_images', 'refine_steps', 'refine_sampler', 'refine_scheduler',
        'denoise', 'refine_cfg', 'refine_prompt', 'refine_upscale_method',
        'generation_status', 'sg_last_gen_error', 'generation_progress',
        'overwrite_material', 'bake_visibility_weights',
        'discard_factor', 'discard_factor_generation_only', 'discard_factor_after_generation',
        'weight_exponent', 'weight_exponent_generation_only', 'weight_exponent_after_generation',
        'view_blend_use_color_match', 'view_blend_color_match_method', 'view_blend_color_match_strength',
        'allow_modify_existing_textures', 'ask_object_prompts', 'fallback_color',
        'controlnet_units', 'controlnet_units_index', 'lora_units', 'lora_units_index',
        'weight_exponent_mask', 'sequential_smooth',
        'canny_threshold_low', 'canny_threshold_high',
        'sequential_factor_smooth', 'sequential_factor_smooth_2', 'sequential_factor',
        'grow_mask_by', 'mask_blocky',
        'visibility_vignette', 'visibility_vignette_width', 'visibility_vignette_softness',
        'visibility_vignette_blur',
        'sg_silhouette_margin', 'sg_silhouette_depth', 'sg_silhouette_rays',
        'refine_angle_ramp_active', 'refine_angle_ramp_pos_0', 'refine_angle_ramp_pos_1',
        'refine_feather_ramp_pos_0', 'refine_feather_ramp_pos_1',
        'refine_edge_feather_projection', 'refine_edge_feather_width', 'refine_edge_feather_softness',
        'differential_diffusion', 'differential_noise',
        'blur_mask', 'blur_mask_radius', 'blur_mask_sigma',
        'sequential_custom_camera_order',
        'ipadapter_strength', 'ipadapter_start', 'ipadapter_end',
        'ipadapter_image', 'ipadapter_weight_type',
        'sequential_ipadapter', 'sequential_ipadapter_mode',
        'sequential_desaturate_factor', 'sequential_contrast_factor',
        'sequential_ipadapter_regenerate', 'sequential_ipadapter_regenerate_wo_controlnet',
        'clip_skip', 'stablegen_preset', 'active_preset',
        'model_architecture', 'architecture_mode',
        'output_timestamp', 'camera_prompts', 'use_camera_prompts',
        'sg_camera_order', 'sg_camera_order_index', 'sg_use_custom_camera_order',
        'show_core_settings', 'show_lora_settings', 'show_camera_options',
        'show_scene_understanding_settings', 'show_output_material_settings',
        'show_image_guidance_settings', 'show_masking_inpainting_settings',
        'show_mode_specific_settings',
        'generation_mode', 'early_priority_strength', 'early_priority', 'texture_objects',
        'qwen_generation_method', 'qwen_refine_use_prev_ref', 'qwen_refine_use_depth',
        'qwen_timestep_zero_ref',
        'qwen_guidance_map_type', 'qwen_voronoi_mode', 'qwen_context_render_mode',
        'qwen_use_external_style_image', 'qwen_external_style_image',
        'qwen_external_style_initial_only',
        'qwen_use_custom_prompts', 'qwen_custom_prompt_initial',
        'qwen_custom_prompt_seq_none', 'qwen_custom_prompt_seq_replace',
        'qwen_custom_prompt_seq_additional',
        'qwen_guidance_fallback_color', 'qwen_guidance_background_color',
        'qwen_context_cleanup', 'qwen_context_cleanup_hue_tolerance',
        'qwen_context_cleanup_value_adjust', 'qwen_context_fallback_dilation',
        'qwen_prompt_gray_background',
    ]
    for prop in _simple_scene_props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # --- PBR Decomposition Properties ---
    pbr_props = [
        'pbr_decomposition', 'pbr_albedo_source',
        'pbr_map_albedo', 'pbr_map_roughness', 'pbr_map_metallic',
        'pbr_map_normal', 'pbr_map_height', 'pbr_height_scale',
        'pbr_map_ao', 'pbr_ao_samples', 'pbr_ao_distance',
        'pbr_map_emission', 'pbr_emission_method',
        'pbr_emission_threshold', 'pbr_emission_saturation_min',
        'pbr_emission_value_min', 'pbr_emission_bloom',
        'pbr_emission_strength',
        'pbr_normal_strength', 'pbr_delight_strength',
        'pbr_use_native_resolution', 'pbr_tiling',
        'pbr_tile_albedo', 'pbr_tile_material', 'pbr_tile_normal', 'pbr_tile_height',
        'pbr_tile_emission', 'pbr_tile_grid', 'pbr_tile_superres',
        'pbr_processing_resolution', 'pbr_denoise_steps', 'pbr_ensemble_size',
        'pbr_albedo_auto_saturation', 'pbr_albedo_saturation_mode',
        'pbr_replace_color_with_albedo', 'pbr_auto_lighting',
        'pbr_model_variant',  # legacy, kept for compat
    ]
    for prop in pbr_props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # --- TRELLIS.2 Properties ---
    trellis2_props = [
        'trellis2_available', 'pbr_nodes_available',
        'show_trellis2_params', 'show_trellis2_advanced',
        'show_trellis2_mesh_settings', 'show_trellis2_texture_settings',
        'show_trellis2_camera_settings',
        'trellis2_last_input_image', 'qwen_use_trellis2_style', 'qwen_trellis2_style_initial_only',
        'trellis2_pipeline_active', 'trellis2_pipeline_phase_start_pct',
        'trellis2_pipeline_total_phases',
        'trellis2_generate_from', 'trellis2_texture_mode', 'trellis2_initial_image_arch',
        'trellis2_camera_count',
        'trellis2_placement_mode', 'trellis2_auto_prompts',
        'trellis2_exclude_bottom', 'trellis2_exclude_bottom_angle',
        'trellis2_auto_aspect', 'trellis2_occlusion_mode',
        'trellis2_consider_existing', 'trellis2_delete_cameras',
        'trellis2_coverage_target',
        'trellis2_max_auto_cameras', 'trellis2_fan_angle',
        'trellis2_import_scale', 'trellis2_shade_mode',
        'trellis2_clamp_elevation', 'trellis2_max_elevation', 'trellis2_min_elevation',
        'trellis2_preview_gallery_enabled', 'trellis2_preview_gallery_count',
        'trellis2_input_image', 'trellis2_batch_folder', 'trellis2_batch_count',
        'trellis2_batch_rename_meshes',
        'trellis2_batch_bake_textures', 'trellis2_batch_bake_pbr',
        'trellis2_batch_bake_resolution', 'trellis2_batch_bake_try_unwrap',
        'trellis2_batch_bake_overlap_only', 'trellis2_batch_bake_export_orm',
        'trellis2_batch_bake_normal_convention', 'trellis2_batch_bake_add_material',
        'trellis2_batch_bake_flatten',
        'trellis2_resolution', 'trellis2_vram_mode',
        'trellis2_attn_backend', 'trellis2_seed', 'trellis2_ss_guidance',
        'trellis2_ss_steps', 'trellis2_shape_guidance', 'trellis2_shape_steps',
        'trellis2_tex_guidance', 'trellis2_tex_steps', 'trellis2_max_tokens',
        'trellis2_texture_size', 'trellis2_decimation', 'trellis2_remesh',
        'trellis2_post_processing_enabled',
        'trellis2_auto_lighting',
        'trellis2_skip_texture', 'trellis2_bg_removal', 'trellis2_background_color',
    ]
    for prop in trellis2_props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # --- Batch generation cleanup ---
    for attr in ('sg_batch_running', 'sg_batch_index', 'sg_batch_total'):
        if hasattr(bpy.types.WindowManager, attr):
            delattr(bpy.types.WindowManager, attr)

    # --- Scene Queue cleanup ---
    if hasattr(bpy.types.WindowManager, 'sg_scene_queue'):
        del bpy.types.WindowManager.sg_scene_queue
    if hasattr(bpy.types.WindowManager, 'sg_scene_queue_index'):
        del bpy.types.WindowManager.sg_scene_queue_index
    if hasattr(bpy.types.WindowManager, 'sg_show_queue'):
        del bpy.types.WindowManager.sg_show_queue

    for attr in (
        'sg_queue_gif_export', 'sg_queue_gif_duration', 'sg_queue_gif_fps',
        'sg_queue_gif_resolution', 'sg_queue_gif_samples', 'sg_queue_gif_engine',
        'sg_queue_gif_interpolation', 'sg_queue_gif_use_hdri',
        'sg_queue_gif_hdri_path', 'sg_queue_gif_hdri_strength',
        'sg_queue_gif_hdri_rotation', 'sg_queue_gif_env_mode',
        'sg_queue_gif_denoiser', 'sg_queue_gif_use_gpu',
        'sg_queue_gif_also_no_pbr',
    ):
        if hasattr(bpy.types.WindowManager, attr):
            delattr(bpy.types.WindowManager, attr)

    # --- Load handlers ---
    if load_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(load_handler)
    if _sg_queue_load_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_sg_queue_load_handler)
